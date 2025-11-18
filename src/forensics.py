# mesh_forensics.py

import json
from typing import Optional, Dict, Any, Tuple

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from src.mesh_io import mesh_arrays, normalize_mesh_with_transform


# ---------- Sampling ----------

def sample_points_uniform_trimesh(
    mesh_o3d: o3d.geometry.TriangleMesh,
    n_points: int = 50_000,
):
    """
    Use trimesh internally because it returns face indices.
    """
    # Convert Open3D mesh to Trimesh
    V = np.asarray(mesh_o3d.vertices)
    F = np.asarray(mesh_o3d.triangles)
    tm = trimesh.Trimesh(vertices=V, faces=F, process=False)

    # Sample uniformly with area weighting
    points, face_idx = tm.sample(n_points, return_index=True)

    return points, face_idx


# ---------- Triangle & edge stats ----------

def compute_edge_data(mesh: o3d.geometry.TriangleMesh):
    """
    Build unique edges and their incident faces.
    Returns:
        edges: (nE, 2) int
        edge_face_indices: list of lists of face indices, length nE
                           If edge_face_indices[i] = [f0, f1], then edges[i] is shared by faces f0 and f1.
    """
    _, F = mesh_arrays(mesh)

    # Build all directed edges then uniq
    e01 = F[:, [0, 1]]
    e12 = F[:, [1, 2]]
    e20 = F[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])             # shape (3*number_of_faces, 2)
    face_ids = np.repeat(np.arange(F.shape[0]), 3) # shape (3*number_of_faces), track which face each edge in `edges` comes from
                                                   # e.g. if we have 2 faces, then face_ids = [0,0,0,1,1,1] for 6 edges
                                                   # first 3 edges from face 0, next 3 from face 1

    # sort endpoints so undirected edges deduplicate
    edges_sorted = np.sort(edges, axis=1)                                       # in this way, (i,j) becomes (j,i) if j<i, all edges are undirected
    uniq_edges, inv = np.unique(edges_sorted, axis=0, return_inverse=True)      # dedup the edges
                                                                                # inv, shape (n_edges_sorted=num_faces*3,)
                                                                                # it tells for each row in edges_sorted, which row in uniq_edges it maps to
                                                                                # recall that edges_sorted[0:3] are edges from face 0, edges_sorted[3:6] are from face 1, etc.
    edge_face_indices = [[] for _ in range(uniq_edges.shape[0])]
    for i_edge, f_idx in zip(inv, face_ids):
        edge_face_indices[i_edge].append(int(f_idx))

    return uniq_edges, edge_face_indices


def edge_lengths(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    V, _ = mesh_arrays(mesh)
    edges, _ = compute_edge_data(mesh)
    e = V[edges[:, 0]] - V[edges[:, 1]]
    return np.linalg.norm(e, axis=1)


def triangle_aspect_ratios(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    Simple aspect ratio: longest edge / shortest altitude.
    """
    V, F = mesh_arrays(mesh)
    tri = V[F]  # (nF, 3, 3)

    a = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    b = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    c = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    longest = np.maximum.reduce([a, b, c])

    # triangle area
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                         tri[:, 2] - tri[:, 0]), axis=1)
    # altitude h = 2A / base (use longest as base)
    altitude = 2.0 * area / (longest + 1e-12)
    ratio = longest / (altitude + 1e-12)
    return ratio


# ---------- Curvature (very simple discrete mean curvature proxy) ----------

def vertex_curvatures(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    Very simple curvature estimate: ‖Laplacian(v)‖ with uniform weights.
    Good enough for comparing CAD vs Trellis, not for differential geometry papers :)
    """
    V, F = mesh_arrays(mesh)
    n_verts = V.shape[0]

    # Build neighbor lists
    neighbors = [[] for _ in range(n_verts)]
    for tri in F:
        i, j, k = tri
        neighbors[i].extend([j, k])
        neighbors[j].extend([i, k])
        neighbors[k].extend([i, j])
    neighbors = [np.unique(nbrs) for nbrs in neighbors]

    lap = np.zeros_like(V)
    for i in range(n_verts):
        nbr = neighbors[i]
        if len(nbr) == 0:
            continue
        lap[i] = V[nbr].mean(axis=0) - V[i]
    curv = np.linalg.norm(lap, axis=1)
    return curv


# ---------- Planarity scores ----------

def planarity_scores(
    points: np.ndarray,
    k: int = 30,
) -> np.ndarray:
    """
    For each point, fit a plane to its k-NN and return RMS residual.
    """
    tree = cKDTree(points)
    n = points.shape[0]
    rms = np.zeros(n)
    for i in range(n):
        d, idx = tree.query(points[i], k=k)
        nbrs = points[idx]
        centroid = nbrs.mean(axis=0)
        X = nbrs - centroid
        # PCA via SVD
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        normal = vh[-1]
        dist = X @ normal
        rms[i] = np.sqrt(np.mean(dist ** 2))
    return rms


# ---------- Cylinder-likeness (PCA-based quick fit) ----------

def cylinder_fit_error(points: np.ndarray) -> float:
    """
    Very simple cylinder fit for a patch:
    - axis = first PCA component
    - radius = mean distance to axis
    - error = RMS distance to that radius
    This is not robust to strong noise but OK for comparison.
    """
    centroid = points.mean(axis=0)
    X = points - centroid
    # PCA
    cov = X.T @ X / X.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)

    # project onto axis to get z, residual radial vector
    z = X @ axis
    proj = np.outer(z, axis)
    radial = X - proj
    r = np.linalg.norm(radial, axis=1)
    r_mean = r.mean()
    err = np.sqrt(np.mean((r - r_mean) ** 2))
    return float(err)


# ---------- Dihedral angles ----------

def dihedral_angles(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    mesh.compute_triangle_normals()
    V, F = mesh_arrays(mesh)
    tris = F
    tri_normals = np.asarray(mesh.triangle_normals)

    edges, edge_face_indices = compute_edge_data(mesh)
    angles = []

    for ef, faces in zip(edges, edge_face_indices):
        if len(faces) != 2:
            continue  # boundary or non-manifold
        f0, f1 = faces
        n0 = tri_normals[f0]
        n1 = tri_normals[f1]
        cos_phi = np.clip(np.dot(n0, n1) /
                          (np.linalg.norm(n0) * np.linalg.norm(n1) + 1e-12),
                          -1.0, 1.0)
        phi = np.degrees(np.arccos(cos_phi))
        # Convert to dihedral "sharpness": 0° for flat, 180° for opposite
        # For our purpose, keep angle between normals:
        angles.append(phi)
    return np.array(angles)


# ---------- High-level analysis ----------

def summarize_stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def analyze_mesh(
    mesh: o3d.geometry.TriangleMesh,
    face_labels: Optional[np.ndarray] = None,
    n_samples: int = 50_000,
    k_planarity: int = 30,
    cylinder_label: int = 1,
) -> Dict[str, Any]:
    """
    Run the full forensics pass on a single mesh.
    """
    # diagonal normalization
    mesh, _, _ = normalize_mesh_with_transform(mesh)

    V, F = mesh_arrays(mesh)
    if face_labels is not None:
        assert face_labels.shape[0] == F.shape[0], \
            "face_labels must have length = n_faces"

    # sampling
    pts, face_idx = sample_points_uniform_trimesh(mesh, n_points=n_samples, return_face_indices=True)

    # basic stats
    edges_len = edge_lengths(mesh)
    tri_ar = triangle_aspect_ratios(mesh)
    curv_v = vertex_curvatures(mesh)
    planarity = planarity_scores(pts, k=k_planarity)
    dih = dihedral_angles(mesh)

    result: Dict[str, Any] = {
        "edge_length": summarize_stats(edges_len),
        "triangle_aspect_ratio": summarize_stats(tri_ar),
        "vertex_curvature": summarize_stats(curv_v),
        "planarity_rms": summarize_stats(planarity),
        "dihedral_angle": summarize_stats(dih),
    }

    # per-primitive curvature / planarity / cylinder error
    if face_labels is not None and face_idx is not None:
        point_labels = face_labels[face_idx]
        primitives = np.unique(point_labels)
        per_prim = {}
        for lab in primitives:
            mask = point_labels == lab
            if mask.sum() < 10:
                continue
            per_prim[int(lab)] = {
                "curvature": summarize_stats(curv_v),  # vertex-level approx
                "planarity_rms": summarize_stats(planarity[mask]),
            }
        result["per_primitive"] = per_prim

        # cylinder-like score for all points on cylindrical faces
        cyl_mask = point_labels == cylinder_label
        if cyl_mask.sum() > 30:
            cyl_err = cylinder_fit_error(pts[cyl_mask])
            result["cylinder_fit_error"] = float(cyl_err)

    return result


# ---------- Simple plotting helpers ----------

def plot_histogram(data: np.ndarray, title: str, bins: int = 50):
    plt.figure(figsize=(4, 3))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.tight_layout()


def quick_plots(mesh: o3d.geometry.TriangleMesh):
    """
    Quick-and-dirty visual inspection: histograms of edge length,
    triangle AR, curvature, dihedral.
    """

    mesh, _, _ = normalize_mesh_with_transform(mesh)

    edges_len = edge_lengths(mesh)
    tri_ar = triangle_aspect_ratios(mesh)
    curv_v = vertex_curvatures(mesh)
    dih = dihedral_angles(mesh)

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(2, 2, 1)
    ax.hist(edges_len, bins=50)
    ax.set_title("Edge lengths")

    ax = plt.subplot(2, 2, 2)
    ax.hist(tri_ar, bins=50)
    ax.set_title("Triangle aspect ratio")

    ax = plt.subplot(2, 2, 3)
    ax.hist(curv_v, bins=50)
    ax.set_title("Vertex curvature (proxy)")

    ax = plt.subplot(2, 2, 4)
    ax.hist(dih, bins=50)
    ax.set_title("Dihedral angles (deg)")

    plt.tight_layout()


def save_summary(summary: Dict[str, Any], path: str):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
