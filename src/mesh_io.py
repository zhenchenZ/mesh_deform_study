from typing import Optional, Dict, Any, Tuple

import numpy as np
from copy import deepcopy
import open3d as o3d
import trimesh

# ---------- I/O & basic helpers ----------

def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    """
    Load an OBJ / PLY mesh with Open3D.
    """
    mesh = o3d.io.read_triangle_mesh(path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    return mesh


def mesh_arrays(mesh: o3d.geometry.TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience: return (V, F) numpy arrays.
    """
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    return V, F


def trimesh_to_o3d(mesh_tm: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """
    Convert a trimesh.Trimesh to an open3d.geometry.TriangleMesh.
    """
    V = mesh_tm.vertices
    F = mesh_tm.faces
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(V)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(F)
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.compute_triangle_normals()
    return mesh_o3d


def normalize_mesh_with_transform(mesh: o3d.geometry.TriangleMesh) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, float]:
    """
    Normalize mesh so that its AABB is centered at origin and has diagonal length 1.

    NOTE: normalization is based on diagonal to be more robust to orientation.

    Returns:
        mesh_norm: normalized mesh
        center: center used for normalization, useful for denormalization
        diag: diagonal length used for normalization, useful for denormalization
    """
    mesh = deepcopy(mesh)
    V = np.asarray(mesh.vertices)

    bbox = mesh.get_axis_aligned_bounding_box()
    min_b = bbox.get_min_bound()
    max_b = bbox.get_max_bound()
    center = (min_b + max_b) * 0.5
    diag = np.linalg.norm(max_b - min_b)

    if diag < 1e-12:
        raise ValueError("Zero-size bbox")

    V_norm = (V - center) / diag
    mesh.vertices = o3d.utility.Vector3dVector(V_norm)
    mesh.compute_vertex_normals()
    return mesh, center, diag


def denormalize_mesh(mesh_norm, center, diag):
    mesh = deepcopy(mesh_norm)
    V = np.asarray(mesh.vertices)
    V_denorm = V * diag + center
    mesh.vertices = o3d.utility.Vector3dVector(V_denorm)
    mesh.compute_vertex_normals()
    return mesh