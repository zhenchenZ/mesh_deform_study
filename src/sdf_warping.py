import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utils: coordinate grids
# ----------------------------
def make_normalized_grid(D: int, H: int, W: int, device=None, dtype=torch.float32):
    """
    Returns normalized grid in [-1, 1] for grid_sample.
    Shape: (1, D, H, W, 3) with order (x, y, z) in last dim for grid_sample 5D.
    """
    zs = torch.linspace(-1, 1, D, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
    z, y, x = torch.meshgrid(zs, ys, xs, indexing="ij")
    grid = torch.stack([x, y, z], dim=-1)  # (D,H,W,3)
    return grid.unsqueeze(0)  # (1,D,H,W,3)


def normalized_to_world(grid_norm, bbox_min, bbox_max):
    """
    Convert normalized coords [-1,1] to world coords.
    grid_norm: (..., 3)
    bbox_min/max: (3,)
    """
    # t in [0,1]
    t = (grid_norm + 1) * 0.5
    return bbox_min + t * (bbox_max - bbox_min)


def world_to_normalized(x_world, bbox_min, bbox_max):
    """
    Convert world coords to normalized [-1,1].
    x_world: (..., 3)
    bbox_min/max: (3,)
    """
    t = (x_world - bbox_min) / (bbox_max - bbox_min)
    return t * 2 - 1


# ----------------------------
# Fourier warp field
# ----------------------------
class FourierWarpField(nn.Module):
    """
    Low-frequency smooth displacement field u(x) defined by a truncated 3D Fourier basis.

    u(x) = sum_{k in K} a_k * sin(2π k·x) + b_k * cos(2π k·x)
    where x is in normalized world space [0,1]^3 (not [-1,1]).

    We store learnable/random coefficients for each component (x,y,z displacement).
    """
    def __init__(self,
                 modes: Tuple[int, int, int] = (3, 3, 3),
                 amplitude: float = 0.03,
                 seed: Optional[int] = None,
                 device=None,
                 dtype=torch.float32):
        super().__init__()
        self.Kx, self.Ky, self.Kz = modes
        self.amplitude = amplitude

        if seed is not None:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
        else:
            g = None

        # coefficients for sin and cos for each displacement component
        # shape: (3, Kx, Ky, Kz)
        self.a = nn.Parameter(torch.empty(3, self.Kx, self.Ky, self.Kz, device=device, dtype=dtype))
        self.b = nn.Parameter(torch.empty(3, self.Kx, self.Ky, self.Kz, device=device, dtype=dtype))

        # init small random
        nn.init.normal_(self.a, mean=0.0, std=1.0)
        nn.init.normal_(self.b, mean=0.0, std=1.0)

    def forward(self, x01):
        """
        x01: normalized coordinate in [0,1], shape (..., 3)
        returns u(x) in normalized units (same space as x01), shape (..., 3)
        """
        # build frequency indices
        kx = torch.arange(1, self.Kx + 1, device=x01.device, dtype=x01.dtype)
        ky = torch.arange(1, self.Ky + 1, device=x01.device, dtype=x01.dtype)
        kz = torch.arange(1, self.Kz + 1, device=x01.device, dtype=x01.dtype)

        # (...,1,1,1) etc for broadcast
        x = x01[..., 0][..., None, None, None]
        y = x01[..., 1][..., None, None, None]
        z = x01[..., 2][..., None, None, None]

        phase = 2 * math.pi * (
            kx[None, :, None, None] * x +
            ky[None, None, :, None] * y +
            kz[None, None, None, :] * z
        )  # (..., Kx, Ky, Kz)

        sin_term = torch.sin(phase)
        cos_term = torch.cos(phase)

        # weighted sum over modes for each component
        # a,b shape (3,Kx,Ky,Kz) -> broadcast to (...,3,Kx,Ky,Kz)
        u = (self.a * sin_term.unsqueeze(-4) + self.b * cos_term.unsqueeze(-4)).sum(dim=(-1, -2, -3))
        # normalize and scale
        u = u / (self.Kx * self.Ky * self.Kz)
        u = u * self.amplitude
        return u


# ----------------------------
# SDF warp
# ----------------------------
def warp_sdf_grid(phi: torch.Tensor,
                  warp_field: nn.Module,
                  bbox_min: torch.Tensor,
                  bbox_max: torch.Tensor,
                  padding_mode: str = "border",
                  align_corners: bool = True):
    """
    phi: (1,1,D,H,W) torch tensor
    warp_field: module mapping x01 -> u(x) in [0,1] coords
    bbox_min/max: (3,) world bounds that phi covers
    return: phi_warped same shape
    """
    assert phi.ndim == 5 and phi.shape[:2] == (1, 1)

    _, _, D, H, W = phi.shape
    device = phi.device
    dtype = phi.dtype

    grid_norm = make_normalized_grid(D, H, W, device=device, dtype=dtype)  # (1,D,H,W,3)
    x_world = normalized_to_world(grid_norm, bbox_min, bbox_max)  # (1,D,H,W,3)

    # map to [0,1]^3 for Fourier basis stability
    x01 = (grid_norm + 1) * 0.5  # (1,D,H,W,3)

    u01 = warp_field(x01)  # (1,D,H,W,3) in [0,1] coord units
    # convert u from [0,1] units to world units then to normalized units
    u_world = u01 * (bbox_max - bbox_min)
    x_warp_world = x_world - u_world

    x_warp_norm = world_to_normalized(x_warp_world, bbox_min, bbox_max)

    # sample old phi at warped coords
    phi_warp = F.grid_sample(
        phi, x_warp_norm,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners
    )
    return phi_warp


# ----------------------------
# Extraction (fallback)
# ----------------------------
def extract_mesh_marching_cubes(phi: torch.Tensor,
                                bbox_min: torch.Tensor,
                                bbox_max: torch.Tensor,
                                level: float = 0.0):
    """
    Fallback extraction with marching cubes (skimage).
    Returns vertices (N,3), faces (M,3) as numpy arrays.
    Swap this with dual contouring if you want.
    """
    import numpy as np
    from skimage.measure import marching_cubes

    phi_np = phi.squeeze().detach().cpu().numpy()  # (D,H,W)
    verts, faces, normals, _ = marching_cubes(phi_np, level=level)

    print(f"[DEBUG] verts min/max after mc:", verts.min(axis=0), verts.max(axis=0))

    # verts are in voxel coords [0, D/H/W)
    D, H, W = phi_np.shape
    verts_norm01 = np.stack([
        verts[:, 0] / (D - 1),
        verts[:, 1] / (H - 1),
        verts[:, 2] / (W - 1),
    ], axis=1)  # to (x,y,z) in [0,1]

    print(f"[DEBUG] verts_norm01 min/max :", verts_norm01.min(axis=0), verts_norm01.max(axis=0))

    bbox_min_np = bbox_min.detach().cpu().numpy()
    bbox_max_np = bbox_max.detach().cpu().numpy()
    verts_world = bbox_min_np + verts_norm01 * (bbox_max_np - bbox_min_np)
    return verts_world, faces
