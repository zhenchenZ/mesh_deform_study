"""
This is a diagnostics kit so you can:
    - understand Trellis SDF error patterns
    - decide what classes of synthetic deformation to simulate on CAD SDFs

    
B.2 What these metrics tell you

- Eikonal stats (mean_eikonal_abs_band)
    If this is large for Trellis SDF, it's not behaving like a real distance field.
    Typical neural artifact.

- Band mean / std
    Shows systematic surface shift (bias) vs random noise.

- Radial power spectrum
    power concentrated at low freq ⇒ global drift / bulging
    power at mid/high freq ⇒ spiky / noisy regression errors

- Compare to GT CAD SDF (optional but ideal)
    Gives you:
        * bias (surface offset)
        * error scale in narrow band


"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def finite_diff_grad(phi: torch.Tensor, spacing: Tuple[float,float,float]=(1,1,1)):
    """
    phi: (1,1,D,H,W)
    returns grad: (1,3,D,H,W)
    """
    assert phi.ndim == 5
    dz, dy, dx = spacing

    # central differences with padding
    phi_zp = F.pad(phi, (0,0,0,0,1,1), mode="replicate")
    phi_yp = F.pad(phi, (0,0,1,1,0,0), mode="replicate")
    phi_xp = F.pad(phi, (1,1,0,0,0,0), mode="replicate")

    dphi_dz = (phi_zp[:,:,2:,:,:] - phi_zp[:,:,:-2,:,:]) / (2*dz)
    dphi_dy = (phi_yp[:,:,:,2:,:] - phi_yp[:,:,:,:-2,:]) / (2*dy)
    dphi_dx = (phi_xp[:,:,:,:,2:] - phi_xp[:,:,:,:,:-2]) / (2*dx)

    grad = torch.cat([dphi_dx, dphi_dy, dphi_dz], dim=1)
    return grad


def eikonal_error(phi: torch.Tensor, band: float=0.05) -> Dict[str, float]:
    """
    Computes ||grad phi|| and deviation from 1 in a narrow band around 0-level set.
    band is in *normalized SDF units* (depends on your scaling).
    """
    grad = finite_diff_grad(phi)                          # (1,3,D,H,W)
    gnorm = torch.linalg.norm(grad, dim=1, keepdim=True)  # (1,1,D,H,W)

    band_mask = (phi.abs() < band).float()
    denom = band_mask.sum().clamp_min(1.0)

    mean_g = (gnorm * band_mask).sum() / denom
    mean_eik = ((gnorm - 1).abs() * band_mask).sum() / denom
    max_eik = ((gnorm - 1).abs() * band_mask).max()

    return {
        "mean_grad_norm_band": float(mean_g.item()),
        "mean_eikonal_abs_band": float(mean_eik.item()),
        "max_eikonal_abs_band": float(max_eik.item())
    }


def narrow_band_stats(phi: torch.Tensor, band: float=0.05) -> Dict[str, float]:
    if isinstance(phi, np.ndarray):
        phi = torch.from_numpy(phi).float()
    
    mask = (phi.abs() < band)
    vals = phi[mask]
    if vals.numel() == 0:
        return {"band_fraction": 0.0}
    return {
        "band_fraction": float(mask.float().mean().item()),
        "band_mean": float(vals.mean().item()),
        "band_std": float(vals.std().item()),
        "band_p95_abs": float(vals.abs().quantile(0.95).item()),
    }


def power_spectrum_radial(phi: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Rough frequency analysis: 3D FFT magnitude and radial binning.
    Good for seeing if errors are low-freq drift vs high-freq spikes.
    """
    # (D,H,W)
    f = torch.fft.fftn(phi.squeeze(0).squeeze(0), norm="ortho")
    mag = (f.real**2 + f.imag**2).sqrt()

    D,H,W = mag.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(D, device=mag.device),
        torch.arange(H, device=mag.device),
        torch.arange(W, device=mag.device),
        indexing="ij"
    )
    # shift freq center
    zz = zz - D//2
    yy = yy - H//2
    xx = xx - W//2
    r = torch.sqrt(xx.float()**2 + yy.float()**2 + zz.float()**2)

    rmax = int(r.max().item())
    bins = torch.linspace(0, rmax, rmax+1, device=mag.device)
    ps = torch.zeros(rmax, device=mag.device)
    counts = torch.zeros(rmax, device=mag.device)

    for i in range(rmax):
        m = (r >= bins[i]) & (r < bins[i+1])
        counts[i] = m.sum()
        if counts[i] > 0:
            ps[i] = mag[m].mean()

    return {
        "radial_freq": torch.arange(rmax, device=mag.device),
        "radial_power": ps,
        "counts": counts
    }


def compare_to_gt(phi_pred: torch.Tensor,
                  phi_gt: torch.Tensor,
                  band: float=0.05) -> Dict[str,float]:
    """
    If you have GT SDF (e.g., from CAD) aligned to pred grid, compute errors.
    """
    err = phi_pred - phi_gt
    mask = (phi_gt.abs() < band)

    denom = mask.sum().clamp_min(1)
    mae_band = err.abs()[mask].sum() / denom
    bias_band = err[mask].sum() / denom
    rmse = (err**2).mean().sqrt()

    return {
        "rmse_full": float(rmse.item()),
        "mae_band": float(mae_band.item()),
        "bias_band": float(bias_band.item())
    }
