"""Bounding volume hierarchy over triangles for AO ray queries."""

from __future__ import annotations

import numpy as np

_LEAF_SIZE = 4
_DET_EPS = 1e-12


class BVH:
    """Median-split BVH over (non-degenerate) triangles.

    Triangles are reordered during construction so that every leaf
    references a contiguous range of the compacted triangle arrays.
    Zero-area triangles are excluded from the tree so they can never
    occlude anything; the excluded count is ``num_degenerate``.
    """

    def __init__(self, positions, triangles):
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)

        v0 = positions[triangles[:, 0]]
        v1 = positions[triangles[:, 1]]
        v2 = positions[triangles[:, 2]]

        cross = np.cross(v1 - v0, v2 - v0)
        keep = np.linalg.norm(cross, axis=1) > _DET_EPS
        self.num_degenerate = int((~keep).sum())

        self.v0 = np.ascontiguousarray(v0[keep])
        self.v1 = np.ascontiguousarray(v1[keep])
        self.v2 = np.ascontiguousarray(v2[keep])

        # Node arrays. A node is a leaf iff _count > 0.
        self._bmin = []
        self._bmax = []
        self._left = []
        self._right = []
        self._start = []
        self._count = []
        self._leaf_indices = []  # per-leaf triangle index lists, pre-compaction

        if len(self.v0):
            self._build(np.arange(len(self.v0)))
            self._compact_leaves()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _new_node(self, bmin, bmax):
        node = len(self._bmin)
        self._bmin.append(bmin)
        self._bmax.append(bmax)
        self._left.append(-1)
        self._right.append(-1)
        self._start.append(-1)
        self._count.append(0)
        return node

    def _build(self, indices):
        v0 = self.v0[indices]
        v1 = self.v1[indices]
        v2 = self.v2[indices]
        tri_min = np.minimum(np.minimum(v0, v1), v2)
        tri_max = np.maximum(np.maximum(v0, v1), v2)
        node = self._new_node(tri_min.min(axis=0), tri_max.max(axis=0))

        if len(indices) <= _LEAF_SIZE:
            self._count[node] = len(indices)
            self._start[node] = len(self._leaf_indices)
            self._leaf_indices.append(indices)
            return node

        centroid = (v0 + v1 + v2) / 3.0
        extent = centroid.max(axis=0) - centroid.min(axis=0)
        axis = int(np.argmax(extent))
        order = indices[np.argsort(centroid[:, axis], kind="stable")]
        mid = len(order) // 2
        self._left[node] = self._build(order[:mid])
        self._right[node] = self._build(order[mid:])
        return node

    def _compact_leaves(self):
        """Reorder triangles so each leaf owns a contiguous index range."""
        order = np.concatenate(self._leaf_indices)
        sizes = np.array([len(x) for x in self._leaf_indices], dtype=np.int64)
        offsets = np.zeros(len(sizes), dtype=np.int64)
        offsets[1:] = np.cumsum(sizes)[:-1]
        for node in range(len(self._bmin)):
            if self._count[node] > 0:
                self._start[node] = int(offsets[self._start[node]])
        self.v0 = self.v0[order]
        self.v1 = self.v1[order]
        self.v2 = self.v2[order]
        self._leaf_indices = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def closest_hit(self, origin, direction, max_t, eps):
        """Return distance to the closest hit in (eps, max_t), or None."""
        if not self._bmin:
            return None
        best = max_t
        stack = [0]
        bmin = self._bmin
        bmax = self._bmax
        left = self._left
        right = self._right
        start = self._start
        count = self._count
        inv = 1.0 / np.where(direction == 0.0, 1e-300, direction)

        while stack:
            node = stack.pop()
            t0 = (bmin[node] - origin) * inv
            t1 = (bmax[node] - origin) * inv
            tmin = float(np.minimum(t0, t1).max())
            tmax = float(np.maximum(t0, t1).min())
            if tmax < max(tmin, eps) or tmin > best:
                continue
            if count[node] > 0:
                hit_t = self._intersect_leaf(origin, direction, node, eps, best)
                if hit_t is not None and hit_t < best:
                    best = hit_t
            else:
                stack.append(left[node])
                stack.append(right[node])
        return None if best == max_t else best

    def _intersect_leaf(self, origin, direction, node, eps, max_t):
        s = self._start[node]
        n = self._count[node]
        v0 = self.v0[s : s + n]
        e1 = self.v1[s : s + n] - v0
        e2 = self.v2[s : s + n] - v0

        p = np.cross(direction, e2)
        det = (e1 * p).sum(axis=1)
        valid = np.abs(det) > _DET_EPS
        if not valid.any():
            return None
        inv_det = 1.0 / np.where(valid, det, 1.0)
        tvec = origin - v0
        u = (tvec * p).sum(axis=1) * inv_det
        q = np.cross(tvec, e1)
        v = (direction * q).sum(axis=1) * inv_det
        t = (e2 * q).sum(axis=1) * inv_det

        hit = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > eps) & (t < max_t)
        if not hit.any():
            return None
        return float(t[hit].min())
