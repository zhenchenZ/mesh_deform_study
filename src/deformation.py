# mesh_deformation.py

from typing import Optional

import numpy as np
from copy import deepcopy
import open3d as o3d
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

from src.mesh_io import normalize_mesh_with_transform


def _mesh_arrays(mesh: o3d.geometry.TriangleMesh):
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


# ---------- Laplacian & spectral displacement ----------

def build_uniform_laplacian(mesh: o3d.geometry.TriangleMesh):
    """
    Build a simple graph Laplacian L = D - A with uniform weights.

    Returns:
        L: sparse matrix of shape (n, n) where n is number of vertices
    """
    V, F = _mesh_arrays(mesh)
    n = V.shape[0]
    # edges via faces
    e01 = F[:, [0, 1]]
    e12 = F[:, [1, 2]]
    e20 = F[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    i = edges[:, 0]
    j = edges[:, 1]
    data = np.ones(len(edges))
    # adjacency (symmetric)
    row = np.concatenate([i, j])
    col = np.concatenate([j, i])
    data_full = np.concatenate([data, data])
    A = coo_matrix((data_full, (row, col)), shape=(n, n)).tocsr()
    deg = np.array(A.sum(axis=1)).ravel()
    D = diags(deg)
    L = D - A
    return L


def spectral_displacement(
    mesh: o3d.geometry.TriangleMesh,
    num_eig: int = 8,
    amplitude: float = 0.01,
    random_state: Optional[np.random.RandomState] = None,
) -> o3d.geometry.TriangleMesh:
    """
    Low-frequency smooth deformation using Laplacian eigenfunctions.
    Displaces vertices along normals with a random combination of low
    frequency eigenmodes.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    V, F = _mesh_arrays(mesh)
    n = V.shape[0]
    L = build_uniform_laplacian(mesh) # (nV, nV)

    k = min(num_eig + 1, n - 1)  # +1 to drop the constant mode
    # smallest eigenvalues (smoothest)
    eigvals, eigvecs = eigsh(L, k=k, which="SM") # find k smallest eigenvalues and eigenvectors of the Laplacian L
                                                 # i.e. solve L * v = lambda * v for the smallest lambda(s)
                                                 # each eigenvec of shape (nV, ) is a scalar function defined on at all vertices
                                                 # (num_eig + 1, ), (nV, num_eig + 1)

    # drop constant eigenvector (eigval ~ 0)
    idx_sorted = np.argsort(eigvals)     # (num_eig + 1, )
    eigvals = eigvals[idx_sorted]        # (num_eig + 1, )
    eigvecs = eigvecs[:, idx_sorted]     # (nV, num_eig + 1)
    eigvecs = eigvecs[:, 1:num_eig + 1]  # skip first, shape (nV, num_eig)

    coeffs = random_state.randn(eigvecs.shape[1]) # (num_eig, )
    field = eigvecs @ coeffs                      # (nV, ), this operation combines eigenvectors linearly, i.e. a weighted sum
    field = (field - field.mean()) / (field.std() + 1e-8)

    mesh = deepcopy(mesh)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    disp = amplitude * field[:, None] * normals
    V_new = V + disp
    mesh.vertices = o3d.utility.Vector3dVector(V_new)
    return mesh


# ---------- Primitive-aware perturbations ----------

def perturb_planes(
    mesh: o3d.geometry.TriangleMesh,
    face_labels: np.ndarray,
    plane_label: int = 0,
    amplitude: float = 0.01,
    freq_u: int = 1,
    freq_v: int = 1,
) -> o3d.geometry.TriangleMesh:
    """
    Add gentle warping / bumps to planar regions.

    For simplicity:
    - group all vertices that belong to any plane face,
    - build a local PCA basis,
    - apply a small 2D sinusoidal height field along the plane normal.
    """
    V, F = _mesh_arrays(mesh)
    assert face_labels.shape[0] == F.shape[0]
    mesh = deepcopy(mesh)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)

    plane_faces = np.where(face_labels == plane_label)[0]
    if len(plane_faces) == 0:
        return mesh

    vert_mask = np.zeros(V.shape[0], dtype=bool)
    vert_mask[F[plane_faces].ravel()] = True
    verts_plane = V[vert_mask]

    # PCA to get dominant plane (u, v, n)
    centroid = verts_plane.mean(axis=0)
    X = verts_plane - centroid
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    basis = vh                  # rows: principal directions
    # n = basis[2]              # but we already have vertex normals; use them for per-vertex displacement
    uv = X @ basis[:2].T        # (n_plane_verts, 2), this gives coordinates in local plane basis

    u_coord = uv[:, 0]
    v_coord = uv[:, 1]
    u_norm = (u_coord - u_coord.min()) / (u_coord.ptp() + 1e-8)
    v_norm = (v_coord - v_coord.min()) / (v_coord.ptp() + 1e-8)

    height = amplitude * np.sin(freq_u * 2 * np.pi * u_norm) * np.sin(freq_v * 2 * np.pi * v_norm)

    # apply along normals for these vertices
    idx_plane_verts = np.where(vert_mask)[0]
    disp = np.zeros_like(V)
    disp[idx_plane_verts] = height[:, None] * normals[idx_plane_verts]

    V_new = V + disp
    mesh.vertices = o3d.utility.Vector3dVector(V_new)
    return mesh


def perturb_cylinders(
    mesh: o3d.geometry.TriangleMesh,
    face_labels: np.ndarray,
    cylinder_label: int = 1,
    radius_amp: float = 0.01,
    wobble_amp: float = 0.01,
) -> o3d.geometry.TriangleMesh:
    """
    Radius variation + axis wobble for cylindrical regions.

    Very approximate:
    - take all vertices that belong to cylinder faces,
    - use PCA to get main axis,
    - use sinusoidal modulation of radius and lateral position along axis.
    """
    V, F = _mesh_arrays(mesh)
    assert face_labels.shape[0] == F.shape[0]
    mesh = deepcopy(mesh)

    cyl_faces = np.where(face_labels == cylinder_label)[0]
    if len(cyl_faces) == 0:
        return mesh

    vert_mask = np.zeros(V.shape[0], dtype=bool)
    vert_mask[F[cyl_faces].ravel()] = True
    verts_cyl = V[vert_mask]

    centroid = verts_cyl.mean(axis=0)
    X = verts_cyl - centroid
    cov = X.T @ X / X.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)

    z = X @ axis               # coordinate along axis
    z_min, z_max = z.min(), z.max()
    L = z_max - z_min + 1e-8
    z_norm = (z - z_min) / L   # in [0, 1]

    proj = np.outer(z, axis)
    radial = X - proj
    r = np.linalg.norm(radial, axis=1) + 1e-8
    radial_dir = radial / r[:, None]

    # radius modulation
    r_mod = r * (1.0 + radius_amp * np.sin(2 * np.pi * z_norm))

    # wobble: shift axis center sideways
    # pick any vector orthogonal to axis
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, axis)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    wobble_dir = np.cross(axis, tmp)
    wobble_dir /= np.linalg.norm(wobble_dir)
    wobble = wobble_amp * np.sin(2 * np.pi * z_norm)[:, None] * wobble_dir[None, :]

    # new positions for cylindrical verts
    X_new = proj + r_mod[:, None] * radial_dir + wobble

    V_new = V.copy()
    idx_cyl = np.where(vert_mask)[0]
    V_new[idx_cyl] = X_new + centroid
    mesh.vertices = o3d.utility.Vector3dVector(V_new)
    return mesh


# ---------- Tessellation-like augmentations ----------

def subdivide_midpoint(
    mesh: o3d.geometry.TriangleMesh,
    iterations: int = 1,
) -> o3d.geometry.TriangleMesh:
    """
    Wrapper around Open3D midpoint subdivision.
    """
    out = mesh
    for _ in range(iterations):
        out = out.subdivide_midpoint(number_of_iterations=1)
    return out


def jitter_vertices(
    mesh: o3d.geometry.TriangleMesh,
    scale: float = 0.001,
) -> o3d.geometry.TriangleMesh:
    """
    Small random vertex noise (absolute scale relative to bbox diagonal).
    """
    mesh = deepcopy(mesh)
    V, _ = _mesh_arrays(mesh)
    bb = mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bb.get_max_bound() - bb.get_min_bound())
    eps = scale * diag
    noise = np.random.randn(*V.shape) * eps
    V_new = V + noise
    mesh.vertices = o3d.utility.Vector3dVector(V_new)
    return mesh


def trellis_like_deform(
    mesh: o3d.geometry.TriangleMesh,
    face_labels: Optional[np.ndarray] = None,
    spectral_amp: float = 0.01,
    plane_amp: float = 0.01,
    cyl_radius_amp: float = 0.02,
    cyl_wobble_amp: float = 0.02,
    jitter_scale: float = 0.0005,
    subdiv_iterations: int = 1,
) -> o3d.geometry.TriangleMesh:
    """
    Full pipeline:
      0) normalize mesh,
      1) global smooth displacement,
      2) primitive-aware local perturbations,
      3) tessellation tweaks (subdivision + jitter).
    """
    mesh, _, _ = normalize_mesh_with_transform(mesh)

    out = spectral_displacement(mesh, amplitude=spectral_amp)

    if face_labels is not None:
        out = perturb_planes(out, face_labels, plane_label=0, amplitude=plane_amp)
        out = perturb_cylinders(
            out,
            face_labels,
            cylinder_label=1,
            radius_amp=cyl_radius_amp,
            wobble_amp=cyl_wobble_amp,
        )

    out = subdivide_midpoint(out, iterations=subdiv_iterations)
    out = jitter_vertices(out, scale=jitter_scale)
    return out
