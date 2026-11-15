"""Tests for the vertex AO baker (run with: python -m unittest)."""

import unittest

import numpy as np

from vertex_ao import bake_vertex_ao


def make_grid(size=4.0, divisions=8, y=0.0):
    """Ground grid on the XZ plane with +Y normals."""
    xs = np.linspace(-size / 2, size / 2, divisions + 1)
    positions = []
    for z in xs:
        for x in xs:
            positions.append((x, y, z))
    n = divisions + 1
    triangles = []
    for j in range(divisions):
        for i in range(divisions):
            a = j * n + i
            b = a + 1
            c = a + n
            d = c + 1
            triangles.append((a, c, b))
            triangles.append((b, c, d))
    normals = np.tile(np.array([[0.0, 1.0, 0.0]]), (len(positions), 1))
    return np.array(positions), np.array(triangles), normals


def make_wall(width=4.0, height=2.0, x=2.0):
    """Vertical wall on the plane x=x, facing -X (towards the ground)."""
    positions = np.array(
        [
            (x, 0.0, -width / 2),
            (x, 0.0, width / 2),
            (x, height, -width / 2),
            (x, height, width / 2),
        ]
    )
    triangles = np.array([(0, 1, 2), (1, 3, 2)])
    normals = np.tile(np.array([[-1.0, 0.0, 0.0]]), (4, 1))
    return positions, triangles, normals


def make_plate(size=1.0, y=1.0, cx=0.0, cz=0.0, normal=(0.0, -1.0, 0.0)):
    """Horizontal quad occluder facing down."""
    h = size / 2
    positions = np.array(
        [
            (cx - h, y, cz - h),
            (cx + h, y, cz - h),
            (cx - h, y, cz + h),
            (cx + h, y, cz + h),
        ]
    )
    triangles = np.array([(0, 3, 1), (0, 2, 3)])
    normals = np.tile(np.array([normal]), (4, 1))
    return positions, triangles, normals


def combine(*meshes):
    positions, triangles, normals = [], [], []
    offset = 0
    for p, t, n in meshes:
        positions.append(p)
        triangles.append(np.asarray(t) + offset)
        normals.append(n)
        offset += len(p)
    return (
        np.vstack(positions),
        np.vstack(triangles),
        np.vstack(normals),
    )


class TestUnoccluded(unittest.TestCase):
    def test_ground_plane_is_fully_lit(self):
        p, t, n = make_grid()
        result = bake_vertex_ao(p, t, n, num_samples=32, max_distance=5.0)
        # Hemisphere above the ground never hits anything.
        np.testing.assert_allclose(result.ao, 1.0, atol=1e-12)

    def test_ao_values_are_unit_range(self):
        ground = make_grid()
        plate = make_plate(size=2.0, y=0.5)
        p, t, n = combine(ground, plate)
        result = bake_vertex_ao(p, t, n, num_samples=32, max_distance=5.0)
        self.assertTrue(np.all(result.ao >= 0.0))
        self.assertTrue(np.all(result.ao <= 1.0))


class TestWallCorner(unittest.TestCase):
    def test_corner_is_darker_than_open_ground(self):
        ground = make_grid(size=4.0, divisions=8)
        wall = make_wall(x=2.0)
        p, t, n = combine(ground, wall)
        result = bake_vertex_ao(p, t, n, num_samples=64, max_distance=4.0)

        ground_ao = result.ao[: len(ground[0])]
        xs = ground[0][:, 0]
        near_wall = ground_ao[xs > 1.0].mean()
        far_side = ground_ao[xs < -1.0].mean()
        self.assertLess(near_wall, far_side)
        self.assertLess(near_wall, 0.95)
        self.assertGreater(far_side, 0.95)


class TestOccluderPlate(unittest.TestCase):
    def test_plate_darkens_ground_below(self):
        ground = make_grid(size=4.0, divisions=8)
        plate = make_plate(size=2.0, y=0.8)
        p, t, n = combine(ground, plate)
        result = bake_vertex_ao(p, t, n, num_samples=64, max_distance=4.0)

        ground_pos = ground[0]
        ground_ao = result.ao[: len(ground_pos)]
        under = (np.abs(ground_pos[:, 0]) < 0.5) & (np.abs(ground_pos[:, 2]) < 0.5)
        away = np.linalg.norm(ground_pos[:, [0, 2]], axis=1) > 1.5
        self.assertLess(ground_ao[under].mean(), ground_ao[away].mean())

    def test_distance_falloff(self):
        ground = make_grid(size=4.0, divisions=8)
        close_plate = make_plate(size=2.0, y=0.4)
        far_plate = make_plate(size=2.0, y=1.6)
        p1, t1, n1 = combine(ground, close_plate)
        p2, t2, n2 = combine(ground, far_plate)
        kwargs = dict(num_samples=64, max_distance=4.0)
        ao_close = bake_vertex_ao(p1, t1, n1, **kwargs).ao[: len(ground[0])]
        ao_far = bake_vertex_ao(p2, t2, n2, **kwargs).ao[: len(ground[0])]
        # Closer occluder darkens more.
        self.assertLess(ao_close.mean(), ao_far.mean())

    def test_occluder_beyond_max_distance_is_ignored(self):
        ground = make_grid(size=4.0, divisions=8)
        plate = make_plate(size=2.0, y=10.0)
        p, t, n = combine(ground, plate)
        result = bake_vertex_ao(p, t, n, num_samples=32, max_distance=5.0)
        np.testing.assert_allclose(result.ao, 1.0, atol=1e-12)


class TestHalfOpenMesh(unittest.TestCase):
    def test_open_box_floor_is_occluded_but_finite(self):
        # Box without its top face: open boundary must not crash or
        # produce out-of-range values; the floor sees the walls.
        ground = make_grid(size=2.0, divisions=4)
        wall_e = make_wall(width=2.0, height=2.0, x=1.0)
        wall_w = make_wall(width=2.0, height=2.0, x=-1.0)
        wall_w = (wall_w[0], wall_w[1], np.tile([[1.0, 0.0, 0.0]], (4, 1)))
        p, t, n = combine(ground, wall_e, wall_w)
        # Include a degenerate (zero-area) triangle: it must be skipped.
        t = np.vstack([t, [[0, 0, 1]]])
        result = bake_vertex_ao(p, t, n, num_samples=32, max_distance=5.0)

        self.assertEqual(result.stats["num_degenerate_triangles"], 1)
        self.assertTrue(np.all(np.isfinite(result.ao)))
        self.assertTrue(np.all(result.ao >= 0.0))
        self.assertTrue(np.all(result.ao <= 1.0))

        floor_ao = result.ao[: len(ground[0])]
        center = floor_ao[
            (np.abs(ground[0][:, 0]) < 0.6) & (np.abs(ground[0][:, 2]) < 0.6)
        ]
        self.assertLess(center.mean(), 1.0)


class TestDuplicateVertices(unittest.TestCase):
    def test_uv_split_seam_is_continuous(self):
        # Two quads sharing an edge but duplicating the seam vertices,
        # as produced by a UV split. AO must match across the seam.
        left = (
            np.array([(-1.0, 0.0, -1.0), (0.0, 0.0, -1.0),
                      (-1.0, 0.0, 1.0), (0.0, 0.0, 1.0)]),
            np.array([(0, 2, 1), (1, 2, 3)]),
            np.tile([[0.0, 1.0, 0.0]], (4, 1)),
        )
        right = (
            np.array([(0.0, 0.0, -1.0), (1.0, 0.0, -1.0),
                      (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]),
            np.array([(0, 2, 1), (1, 2, 3)]),
            np.tile([[0.0, 1.0, 0.0]], (4, 1)),
        )
        wall = make_wall(width=2.0, height=2.0, x=1.0)
        p, t, n = combine(left, right, wall)
        result = bake_vertex_ao(p, t, n, num_samples=64, max_distance=4.0)

        # Seam vertices: left[1] <-> right[0], left[3] <-> right[2].
        self.assertAlmostEqual(result.ao[1], result.ao[4], places=12)
        self.assertAlmostEqual(result.ao[3], result.ao[6], places=12)
        # The merge actually happened: 8 ground vertices collapse to 6,
        # plus the 4 wall vertices.
        self.assertEqual(result.stats["num_unique_vertices"], 10)

    def test_different_normals_are_not_merged(self):
        # Same position, different normals -> independent AO values.
        p = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
        t = np.zeros((0, 3), dtype=int)
        n = np.array([(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
        result = bake_vertex_ao(p, t, n, num_samples=8, max_distance=1.0)
        self.assertEqual(result.stats["num_unique_vertices"], 2)

    def test_rebake_is_deterministic(self):
        ground = make_grid(size=4.0, divisions=6)
        plate = make_plate(size=2.0, y=0.8)
        p, t, n = combine(ground, plate)
        kwargs = dict(num_samples=48, max_distance=4.0, seed=7)
        first = bake_vertex_ao(p, t, n, **kwargs)
        second = bake_vertex_ao(p, t, n, **kwargs)
        np.testing.assert_array_equal(first.ao, second.ao)


class TestOverlappingSurfaces(unittest.TestCase):
    def test_coplanar_duplicate_geometry_does_not_self_occlude(self):
        # Two copies of the same ground plane (z-fighting overlap):
        # hits at t <= ray_epsilon are ignored, so AO stays ~1.
        ground = make_grid(size=4.0, divisions=6)
        p, t, n = combine(ground, ground)
        result = bake_vertex_ao(p, t, n, num_samples=32, max_distance=5.0)
        ground_ao = result.ao[: len(ground[0])]
        np.testing.assert_allclose(ground_ao, 1.0, atol=1e-12)


class TestValidation(unittest.TestCase):
    def test_zero_normal_raises(self):
        p = np.array([(0.0, 0.0, 0.0)])
        t = np.zeros((0, 3), dtype=int)
        n = np.array([(0.0, 0.0, 0.0)])
        with self.assertRaises(ValueError):
            bake_vertex_ao(p, t, n)

    def test_index_out_of_range_raises(self):
        p = np.array([(0.0, 0.0, 0.0)])
        t = np.array([(0, 1, 2)])
        n = np.array([(0.0, 1.0, 0.0)])
        with self.assertRaises(ValueError):
            bake_vertex_ao(p, t, n)


if __name__ == "__main__":
    unittest.main()
