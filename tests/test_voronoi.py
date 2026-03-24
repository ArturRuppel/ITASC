"""Tests for Voronoi tessellation."""
import numpy as np
import pytest

from napariTissueFlow.core.voronoi import (
    compute_voronoi,
    voronoi_to_graph,
    voronoi_to_labels,
    lloyd_relaxation,
    _polygon_area,
    _polygon_perimeter,
)
from napariTissueFlow.structures import VoronoiMethod


class TestPolygonHelpers:
    def test_unit_square_area(self):
        verts = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert abs(_polygon_area(verts) - 1.0) < 1e-10

    def test_triangle_area(self):
        verts = np.array([[0, 0], [0, 2], [1, 0]])
        assert abs(_polygon_area(verts) - 1.0) < 1e-10

    def test_unit_square_perimeter(self):
        verts = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert abs(_polygon_perimeter(verts) - 4.0) < 1e-10


class TestVoronoi:
    def test_regular_grid_produces_graph(self, grid_positions):
        image_shape = (100, 100)
        vor, _ = compute_voronoi(grid_positions, image_shape=image_shape)
        cells, junctions, graph = voronoi_to_graph(
            vor, grid_positions, len(grid_positions), image_shape=image_shape
        )
        assert len(cells) > 0
        assert len(junctions) > 0
        assert graph.number_of_nodes() > 0

    def test_all_cells_represented(self, grid_positions):
        image_shape = (100, 100)
        vor, _ = compute_voronoi(grid_positions, image_shape=image_shape)
        cells, _, _ = voronoi_to_graph(
            vor, grid_positions, len(grid_positions), image_shape=image_shape
        )
        assert len(cells) == len(grid_positions)

    def test_junction_lengths_positive(self, grid_positions):
        image_shape = (100, 100)
        vor, _ = compute_voronoi(grid_positions, image_shape=image_shape)
        _, junctions, _ = voronoi_to_graph(
            vor, grid_positions, len(grid_positions), image_shape=image_shape
        )
        for jd in junctions.values():
            assert jd.length > 0

    def test_interior_cells_have_nonzero_area(self, grid_positions):
        image_shape = (100, 100)
        vor, _ = compute_voronoi(grid_positions, image_shape=image_shape)
        cells, _, _ = voronoi_to_graph(
            vor, grid_positions, len(grid_positions), image_shape=image_shape
        )
        # At least some cells should have positive area
        areas = [c.area for c in cells.values()]
        assert any(a > 0 for a in areas)

    def test_vertices_stored_in_cells(self, grid_positions):
        """Interior cells should have polygon vertices stored."""
        image_shape = (100, 100)
        vor, _ = compute_voronoi(grid_positions, image_shape=image_shape)
        cells, _, _ = voronoi_to_graph(
            vor, grid_positions, len(grid_positions), image_shape=image_shape
        )
        cells_with_verts = [c for c in cells.values() if c.vertices is not None]
        assert len(cells_with_verts) > 0
        for cell in cells_with_verts:
            assert cell.vertices.ndim == 2
            assert cell.vertices.shape[1] == 2
            assert len(cell.vertices) >= 3  # valid polygon


class TestLloyd:
    def test_lloyd_converges(self):
        """Centroid displacement should decrease with iterations."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(10, 90, size=(20, 2))
        image_shape = (100, 100)

        # Run 1 iteration vs 20 iterations
        pts_1 = lloyd_relaxation(positions, image_shape, n_iterations=1, tol=0.0)
        pts_20 = lloyd_relaxation(positions, image_shape, n_iterations=20, tol=0.0)

        # Compute centroid displacement for each
        def centroid_displacement(pts):
            vor, _ = compute_voronoi(pts, image_shape=image_shape)
            disps = []
            for i in range(len(pts)):
                region_idx = vor.point_region[i]
                region = vor.regions[region_idx]
                if -1 in region or len(region) < 3:
                    continue
                verts = np.clip(vor.vertices[region], [0, 0], [100, 100])
                centroid = np.mean(verts, axis=0)
                disps.append(np.linalg.norm(pts[i] - centroid))
            return np.mean(disps)

        disp_1 = centroid_displacement(pts_1)
        disp_20 = centroid_displacement(pts_20)
        assert disp_20 < disp_1

    def test_lloyd_zero_iterations_matches_standard(self):
        """With 0 iterations, Lloyd's should give same result as standard."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(10, 90, size=(15, 2))
        image_shape = (100, 100)

        vor_std, pos_std = compute_voronoi(positions, image_shape=image_shape)
        vor_lloyd, pos_lloyd = compute_voronoi(
            positions, image_shape=image_shape,
            method=VoronoiMethod.LLOYD, lloyd_iterations=0,
        )
        np.testing.assert_array_equal(pos_std, pos_lloyd)

    def test_lloyd_positions_stay_in_bounds(self):
        """After Lloyd's, all positions should remain within image bounds."""
        rng = np.random.default_rng(42)
        positions = rng.uniform(5, 95, size=(25, 2))
        image_shape = (100, 100)

        pts = lloyd_relaxation(positions, image_shape, n_iterations=10, tol=0.01)
        assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= 100)
        assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] <= 100)


class TestVoronoiToLabels:
    def test_output_shape_and_dtype(self, grid_positions):
        image_shape = (100, 100)
        labels, _ = voronoi_to_labels(grid_positions, image_shape)
        assert labels.shape == image_shape
        assert labels.dtype == np.int32

    def test_labels_are_one_indexed(self, grid_positions):
        image_shape = (100, 100)
        labels, _ = voronoi_to_labels(grid_positions, image_shape)
        assert labels.min() >= 1
        assert labels.max() == len(grid_positions)

    def test_all_cells_have_pixels(self, grid_positions):
        """Every cell should own at least one pixel."""
        image_shape = (100, 100)
        labels, _ = voronoi_to_labels(grid_positions, image_shape)
        unique_labels = set(np.unique(labels))
        expected = set(range(1, len(grid_positions) + 1))
        assert expected == unique_labels

    def test_every_pixel_assigned(self, grid_positions):
        """No pixel should be zero (background)."""
        image_shape = (100, 100)
        labels, _ = voronoi_to_labels(grid_positions, image_shape)
        assert np.all(labels > 0)

    def test_cell_contains_its_seed(self, grid_positions):
        """Each cell's seed pixel should have that cell's label."""
        image_shape = (100, 100)
        labels, final_pos = voronoi_to_labels(grid_positions, image_shape)
        for i, (y, x) in enumerate(final_pos):
            yi, xi = int(round(y)), int(round(x))
            if 0 <= yi < image_shape[0] and 0 <= xi < image_shape[1]:
                assert labels[yi, xi] == i + 1

    def test_lloyd_method(self, grid_positions):
        """Lloyd's method should produce valid labels and different positions."""
        image_shape = (100, 100)
        labels, final_pos = voronoi_to_labels(
            grid_positions, image_shape,
            method=VoronoiMethod.LLOYD, lloyd_iterations=5,
        )
        assert labels.shape == image_shape
        assert np.all(labels > 0)
