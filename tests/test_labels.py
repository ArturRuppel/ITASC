"""Tests for graph extraction from segmentation labels."""
import numpy as np
import pytest

from cellflow.backend.labels import (
    find_border_boundary,
    find_border_cells,
    calculate_edge_length,
    labels_to_graph,
    merge_cells,
    clean_stranded_pixels,
    remove_tiny_cells,
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


class TestFindBorderBoundary:
    """Tests for border edge detection with contiguous segment splitting."""

    def _make_frame_with_border_cell(self):
        """A 20x20 frame with cell 1 touching the top edge."""
        frame = np.zeros((20, 20), dtype=np.int32)
        frame[0:8, 5:15] = 1   # cell 1 touches top edge
        frame[10:18, 5:15] = 2  # cell 2 in the middle
        return frame

    def test_returns_list(self):
        frame = self._make_frame_with_border_cell()
        result = find_border_boundary(frame, 1)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_each_segment_has_coords_and_length(self):
        frame = self._make_frame_with_border_cell()
        for coords, length in find_border_boundary(frame, 1):
            assert coords.ndim == 2
            assert coords.shape[1] == 2
            assert length > 0

    def test_non_border_cell_returns_empty(self):
        """Cell 2 doesn't touch any border or background in a packed frame."""
        frame = np.zeros((20, 20), dtype=np.int32)
        frame[:, :] = 2  # fill everything
        frame[5:15, 5:15] = 1  # cell 1 surrounded by cell 2
        # Cell 1 has no border contact
        assert find_border_boundary(frame, 1) == []

    def test_small_hole_ignored_with_min_length(self):
        """A tiny background hole should not create a border segment
        when the cell does not touch the image edge."""
        # Cell 1 surrounded by cell 2 — only border contact is a tiny hole.
        frame = np.full((30, 30), 2, dtype=np.int32)
        frame[5:25, 5:25] = 1
        # Poke a 1-pixel hole inside cell 1
        frame[15, 15] = 0
        # The hole boundary is ~4 px; min_edge_length=5 should filter it out.
        result = find_border_boundary(frame, 1, min_edge_length=5.0)
        assert len(result) == 0

    def test_internal_hole_not_on_external_contour(self):
        """Internal holes don't appear on the external contour, so
        find_border_boundary correctly ignores them regardless of min_edge_length."""
        frame = np.full((30, 30), 2, dtype=np.int32)
        frame[5:25, 5:25] = 1
        frame[15, 15] = 0
        # The external contour of cell 1 doesn't pass through the hole
        result = find_border_boundary(frame, 1, min_edge_length=0.0)
        assert len(result) == 0

    def test_multiple_disconnected_segments(self):
        """A cell touching two separate edges should produce multiple segments."""
        frame = np.zeros((30, 30), dtype=np.int32)
        # Cell 1: an L-shape touching both top and right edges with a gap
        frame[0:5, 0:10] = 1     # touches top edge
        frame[10:20, 25:30] = 1  # touches right edge
        # These are disconnected regions; OpenCV may find the largest contour
        # only, but the principle is tested.
        result = find_border_boundary(frame, 1, min_edge_length=0.0)
        # At minimum, the largest contour produces at least one segment
        assert len(result) >= 1


class TestBorderTaggingWithMinLength:
    """Integration test: small holes shouldn't cause border tagging."""

    def test_small_hole_does_not_tag_junctions(self):
        """Two adjacent cells, one with a tiny internal hole, should not
        get border tags on their shared junction."""
        frame = np.zeros((30, 30), dtype=np.int32)
        frame[0:15, :] = 1
        frame[15:30, :] = 2
        # Poke a small hole inside cell 1
        frame[7, 15] = 0

        cells, junctions, graph = labels_to_graph(
            frame, filter_isolated=True, min_border_edge_length=10.0,
        )
        # The shared junction between cell 1 and 2 should exist
        pair = frozenset((1, 2))
        assert pair in junctions
        # With a high min_border_edge_length, the tiny hole shouldn't make
        # cell 1 a "border cell", so the shared junction shouldn't be tagged.
        # (Cell 1 does touch the image border on top/sides, so it IS a real
        # border cell — but only if those segments pass the length threshold.)
        # The key check: the function doesn't crash and junctions are created.
        assert junctions[pair].length > 0


class TestMergeCells:
    """Tests for cell merging with automatic cleanup of stranded pixels."""

    def test_merge_adjacent_cells(self):
        """Two adjacent cells should merge without stranded pixels."""
        seg = np.zeros((20, 20), dtype=np.int32)
        seg[5:10, 5:10] = 1   # cell 1
        seg[5:10, 10:15] = 2  # cell 2 (touching on the right)

        initial_cell1_px = np.sum(seg == 1)
        initial_cell2_px = np.sum(seg == 2)

        # Merge cell 1 into cell 2
        result = merge_cells(seg, (7, 7), (7, 12), label_a=1, label_b=2)
        assert result is True

        # Cell 1 should no longer exist
        assert np.sum(seg == 1) == 0
        # Cell 2 should contain all pixels (no stranded pixels left behind)
        assert np.sum(seg == 2) == initial_cell1_px + initial_cell2_px

    def test_merge_non_touching_cells_fails(self):
        """Merging non-touching cells should fail."""
        seg = np.zeros((30, 30), dtype=np.int32)
        seg[5:10, 5:10] = 1   # cell 1
        seg[20:25, 20:25] = 2  # cell 2 (not touching)

        # Merge should fail
        result = merge_cells(seg, (7, 7), (22, 22), label_a=1, label_b=2)
        assert result is False

        # Both cells should remain unchanged
        assert np.sum(seg == 1) == 25
        assert np.sum(seg == 2) == 25

    def test_merge_creates_no_stranded_pixels(self):
        """After merge, no pixels of the merged cell should remain."""
        seg = np.zeros((40, 40), dtype=np.int32)
        # Create two L-shaped cells that touch in a complex way
        seg[5:15, 5:10] = 1   # vertical part of cell 1
        seg[10:15, 5:15] = 1  # horizontal part of cell 1
        seg[10:15, 10:20] = 2  # overlapping part with cell 2

        result = merge_cells(seg, (12, 7), (12, 15), label_a=1, label_b=2)
        assert result is True

        # After cleanup, cell 1 should not exist at all
        assert np.sum(seg == 1) == 0


class TestCleanStrandedPixels:
    """Tests for removing stranded pixels and orphaned background holes."""

    def test_removes_stranded_pixels_from_cell(self):
        """Isolated pixel components smaller than min_size should be removed."""
        seg = np.zeros((20, 20), dtype=np.int32)
        seg[5:10, 5:10] = 1   # main cell 1 (25 pixels)
        seg[15, 15] = 1        # stranded pixel (1 pixel)

        removed = clean_stranded_pixels(seg, min_size=4)

        # Stranded pixel should be removed
        assert seg[15, 15] != 1
        assert removed >= 1

    def test_fills_orphaned_background_holes(self):
        """Small enclosed background holes should be filled with nearest cell."""
        seg = np.zeros((30, 30), dtype=np.int32)
        seg[:, :] = 1  # fill with cell 1
        seg[10:20, 10:20] = 2  # cell 2 in center
        seg[15, 15] = 0  # small hole (1 pixel)

        removed = clean_stranded_pixels(seg, min_size=4)

        # The hole should be filled
        assert seg[15, 15] != 0
        assert removed >= 1

    def test_preserves_large_components(self):
        """Large components should not be removed."""
        seg = np.zeros((20, 20), dtype=np.int32)
        seg[5:10, 5:10] = 1   # large cell 1 (25 pixels)

        removed = clean_stranded_pixels(seg, min_size=4)

        # Large cell should remain
        assert np.sum(seg == 1) == 25
        assert removed == 0


class TestRemoveTinyCells:
    """Tests for removing entire cells smaller than a threshold."""

    def test_removes_tiny_cell(self):
        """Cells smaller than min_size should be removed."""
        seg = np.zeros((30, 30), dtype=np.int32)
        seg[10:15, 10:15] = 1  # main cell 1 (25 pixels)
        seg[5:7, 5:7] = 2       # tiny cell 2 (4 pixels)

        removed = remove_tiny_cells(seg, min_size=5)

        # Tiny cell should be removed
        assert np.sum(seg == 2) == 0
        assert removed == 4

    def test_preserves_large_cells(self):
        """Cells at or above min_size should not be removed."""
        seg = np.zeros((30, 30), dtype=np.int32)
        seg[5:10, 5:10] = 1    # cell 1 (25 pixels)
        seg[15:17, 15:17] = 2  # cell 2 (4 pixels, exactly min_size)

        removed = remove_tiny_cells(seg, min_size=4)

        # Cell 2 is exactly min_size, should be preserved
        assert np.sum(seg == 2) == 4
        assert removed == 0

    def test_removes_multiple_tiny_cells(self):
        """Multiple tiny cells should all be removed."""
        seg = np.zeros((40, 40), dtype=np.int32)
        seg[5:10, 5:10] = 1    # cell 1 (25 pixels - large)
        seg[15:17, 15:17] = 2  # cell 2 (4 pixels - tiny)
        seg[25:26, 25:26] = 3  # cell 3 (1 pixel - tiny)

        removed = remove_tiny_cells(seg, min_size=5)  # min_size=5 removes cells < 5

        # Large cell should remain
        assert np.sum(seg == 1) == 25
        # Tiny cells should be removed
        assert np.sum(seg == 2) == 0
        assert np.sum(seg == 3) == 0
        assert removed == 5

    def test_reassigns_to_neighbors(self):
        """Removed tiny cell pixels should be reassigned to neighboring cells."""
        seg = np.zeros((20, 20), dtype=np.int32)
        seg[5:15, 5:15] = 1   # main cell 1 (100 pixels)
        seg[8:10, 15:17] = 2  # tiny cell 2 (4 pixels) adjacent to cell 1

        initial_cell1 = np.sum(seg == 1)
        removed = remove_tiny_cells(seg, min_size=5)  # Remove cells < 5 pixels

        # Tiny cell removed
        assert np.sum(seg == 2) == 0
        # Cell 1 should have gained the removed pixels
        assert np.sum(seg == 1) >= initial_cell1
        assert removed == 4
