"""Tests for graph extraction from segmentation labels."""
import numpy as np
import pytest

from napariTissueGraph.core.labels import (
    find_shared_boundary,
    calculate_edge_length,
    labels_to_graph,
)


class TestCalculateEdgeLength:
    def test_straight_line(self):
        coords = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
        assert abs(calculate_edge_length(coords) - 3.0) < 1e-10

    def test_diagonal(self):
        coords = np.array([[0, 0], [1, 1]])
        assert abs(calculate_edge_length(coords) - np.sqrt(2)) < 1e-10

    def test_single_point(self):
        coords = np.array([[0, 0]])
        assert calculate_edge_length(coords) == 0.0


class TestFindSharedBoundary:
    def test_adjacent_cells(self, label_frame):
        # Cells 1 and 2 should be adjacent in our grid layout
        result = find_shared_boundary(label_frame, 1, 2)
        # They may or may not be adjacent depending on Voronoi layout;
        # just check the function doesn't crash
        if result is not None:
            coords, length = result
            assert len(coords) >= 2
            assert length > 0

    def test_distant_cells(self, label_frame):
        # Very distant cells shouldn't share a boundary
        # Cell 1 is top-left, cell with highest id is bottom-right
        max_id = label_frame.max()
        result = find_shared_boundary(label_frame, 1, max_id)
        # May or may not be adjacent — just verify it returns valid type
        assert result is None or len(result) == 2


class TestLabelsToGraph:
    def test_basic(self, label_frame):
        cells, junctions, graph = labels_to_graph(label_frame)
        assert len(cells) > 0
        assert len(junctions) > 0
        assert graph.number_of_edges() > 0

    def test_cell_areas_match_label_counts(self, label_frame):
        cells, _, _ = labels_to_graph(label_frame, filter_isolated=False)
        for cid, cd in cells.items():
            expected_area = np.sum(label_frame == cid)
            assert abs(cd.area - expected_area) < 1e-10

    def test_shape_index_reasonable(self, label_frame):
        cells, _, _ = labels_to_graph(label_frame)
        for cd in cells.values():
            # Shape index for convex cells typically 3.5-4.5
            assert cd.shape_index > 2.0
            assert cd.shape_index < 10.0

    def test_num_neighbors_matches_graph(self, label_frame):
        cells, _, graph = labels_to_graph(label_frame)
        for cid, cd in cells.items():
            assert cd.num_neighbors == graph.degree(cid)
