"""Vertex ambient-occlusion baking over a triangle BVH."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bvh import BVH


@dataclass
class BakeResult:
    """Result of :func:`bake_vertex_ao`.

    Attributes:
        ao: Per-input-vertex AO in [0, 1]; duplicated vertices (same
            quantized position + normal) share identical values.
        unique_ao: AO values for the merged unique vertices.
        unique_positions: Positions of the merged unique vertices.
        unique_normals: Normals of the merged unique vertices.
        vertex_to_unique: Maps every input vertex to its unique index.
        stats: Diagnostic counters (degenerate triangles, merge counts).
    """

    ao: np.ndarray
    unique_ao: np.ndarray
    unique_positions: np.ndarray
    unique_normals: np.ndarray
    vertex_to_unique: np.ndarray
    stats: dict = field(default_factory=dict)


def _merge_vertices(positions, normals, position_tol, normal_tol):
    """Group vertices by quantized (position, normal).

    Returns (unique_indices, vertex_to_unique) where unique_indices[i] is
    the representative input vertex of unique group i (first occurrence).
    """
    pos_key = np.round(positions / position_tol).astype(np.int64)
    nrm_key = np.round(normals / normal_tol).astype(np.int64)

    key_to_group = {}
    vertex_to_unique = np.empty(len(positions), dtype=np.int64)
    unique_indices = []
    for i in range(len(positions)):
        key = (
            int(pos_key[i, 0]), int(pos_key[i, 1]), int(pos_key[i, 2]),
            int(nrm_key[i, 0]), int(nrm_key[i, 1]), int(nrm_key[i, 2]),
        )
        group = key_to_group.get(key)
        if group is None:
            group = len(unique_indices)
            key_to_group[key] = group
            unique_indices.append(i)
        vertex_to_unique[i] = group
    return np.asarray(unique_indices, dtype=np.int64), vertex_to_unique


def _tangent_frame(normal):
    """Orthonormal basis (tangent, bitangent) for a unit normal."""
    if abs(normal[2]) < 0.999:
        tangent = np.cross(normal, (0.0, 0.0, 1.0))
    else:
        tangent = np.cross(normal, (1.0, 0.0, 0.0))
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    return tangent, bitangent


def _cosine_hemisphere_directions(normals, num_samples, rng):
    """Cosine-weighted hemisphere samples around each normal, shape (V, S, 3)."""
    u1 = rng.random((len(normals), num_samples))
    u2 = rng.random((len(normals), num_samples))
    r = np.sqrt(u1)
    phi = 2.0 * np.pi * u2
    local = np.stack(
        [r * np.cos(phi), r * np.sin(phi), np.sqrt(np.maximum(1.0 - u1, 0.0))],
        axis=-1,
    )
    directions = np.empty_like(local)
    for i, n in enumerate(normals):
        tangent, bitangent = _tangent_frame(n)
        basis = np.stack([tangent, bitangent, n], axis=1)
        directions[i] = local[i] @ basis.T
    return directions


def bake_vertex_ao(
    positions,
    triangles,
    normals,
    num_samples=64,
    max_distance=10.0,
    falloff=1.0,
    seed=0,
    position_tol=1e-5,
    normal_tol=1e-4,
    ray_epsilon=None,
):
    """Bake per-vertex ambient occlusion for a triangle mesh.

    Args:
        positions: (N, 3) vertex positions in world space.
        triangles: (M, 3) triangle indices into ``positions``.
        normals: (N, 3) vertex normals (need not be unit length; zero-length
            normals raise ``ValueError``).
        num_samples: Hemisphere rays per unique vertex.
        max_distance: Rays are clipped to this distance; hits beyond it count
            as fully visible (open sky).
        falloff: Exponent shaping the distance attenuation of occluders.
            A hit at distance ``d`` contributes occlusion
            ``(1 - d / max_distance) ** falloff``; larger values localize
            the darkening to close geometry.
        seed: RNG seed; identical inputs and seed give identical results.
        position_tol: World-space tolerance used to merge duplicate
            (e.g. UV-split) vertices before baking.
        normal_tol: Tolerance used to merge vertex normals.
        ray_epsilon: Ray origin offset along the normal and minimum hit
            distance. Defaults to 1e-4 of the scene diagonal. Hits at
            ``t <= ray_epsilon`` are ignored, which makes coplanar
            overlapping surfaces at the shading plane non-occluding.

    Returns:
        BakeResult with per-vertex AO in [0, 1] (1 = fully unoccluded).
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)

    if len(positions) != len(normals):
        raise ValueError("positions and normals must have the same length")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if max_distance <= 0:
        raise ValueError("max_distance must be > 0")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(positions)):
        raise ValueError("triangle indices out of range")

    lengths = np.linalg.norm(normals, axis=1)
    if (lengths == 0.0).any():
        raise ValueError("normals must be non-zero")
    unit_normals = normals / lengths[:, None]

    unique_indices, vertex_to_unique = _merge_vertices(
        positions, unit_normals, position_tol, normal_tol
    )
    unique_positions = positions[unique_indices]
    unique_normals = unit_normals[unique_indices]

    bvh = BVH(positions, triangles)

    if ray_epsilon is None:
        if len(positions):
            diag = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
        else:
            diag = 0.0
        ray_epsilon = max(1e-6, 1e-4 * diag)

    rng = np.random.default_rng(seed)
    directions = _cosine_hemisphere_directions(unique_normals, num_samples, rng)

    unique_ao = np.ones(len(unique_indices), dtype=np.float64)
    for i in range(len(unique_indices)):
        origin = unique_positions[i] + unique_normals[i] * ray_epsilon
        visibility = 0.0
        for s in range(num_samples):
            t = bvh.closest_hit(origin, directions[i, s], max_distance, ray_epsilon)
            if t is None:
                visibility += 1.0
            else:
                occlusion = max(0.0, 1.0 - t / max_distance) ** falloff
                visibility += 1.0 - occlusion
        unique_ao[i] = visibility / num_samples

    ao = unique_ao[vertex_to_unique]
    stats = {
        "num_vertices": len(positions),
        "num_unique_vertices": len(unique_indices),
        "num_triangles": len(triangles),
        "num_degenerate_triangles": bvh.num_degenerate,
        "num_samples": num_samples,
        "max_distance": max_distance,
        "ray_epsilon": ray_epsilon,
    }
    return BakeResult(
        ao=ao,
        unique_ao=unique_ao,
        unique_positions=unique_positions,
        unique_normals=unique_normals,
        vertex_to_unique=vertex_to_unique,
        stats=stats,
    )
