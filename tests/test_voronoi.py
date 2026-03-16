"""Tests for Voronoi tessellation."""
import numpy as np
import pytest

from napariTissueGraph.core.voronoi import (
    compute_voronoi,
    voronoi_to_graph,
    _polygon_area,
    _polygon_perimeter,
)


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
        vor = compute_voronoi(grid_positions, image_shape=image_shape)
        cells, junctions, graph = voronoi_to_graph(
            vor, grid_positions, len(grid_positions), image_shape=image_shape
        )
        assert len(cells) > 0
        assert len(junctions) > 0
        assert graph.number_of_nodes() > 0

    def test_all_cells_represented(self, grid_positions):
        image_shape = (100, 100)
        vor = compute_voronoi(grid_positions, image_shape=image_shape)
        cells, _, _ = voronoi_to_graph(
            vor, grid_positions, len(grid_positions), image_shape=image_shape
        )
        assert len(cells) == len(grid_positions)

    def test_junction_lengths_positive(self, grid_positions):
        image_shape = (100, 100)
        vor = compute_voronoi(grid_positions, image_shape=image_shape)
        _, junctions, _ = voronoi_to_graph(
            vor, grid_positions, len(grid_positions), image_shape=image_shape
        )
        for jd in junctions.values():
            assert jd.length > 0

    def test_interior_cells_have_nonzero_area(self, grid_positions):
        image_shape = (100, 100)
        vor = compute_voronoi(grid_positions, image_shape=image_shape)
        cells, _, _ = voronoi_to_graph(
            vor, grid_positions, len(grid_positions), image_shape=image_shape
        )
        # At least some cells should have positive area
        areas = [c.area for c in cells.values()]
        assert any(a > 0 for a in areas)
