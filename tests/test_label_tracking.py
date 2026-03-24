"""Tests for label-based cell tracking."""
import numpy as np
import pytest

from napariTissueFlow.core.label_tracking import (
    match_labels,
    assign_track_ids,
    label_to_vertices,
)


def _make_simple_frame(positions, size=100):
    """Make a label frame from cell center positions using nearest-neighbor."""
    frame = np.zeros((size, size), dtype=np.int32)
    yy, xx = np.mgrid[0:size, 0:size]
    coords = np.column_stack([yy.ravel(), xx.ravel()])
    dists = np.sum((coords[:, None, :] - np.array(positions)[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1) + 1
    frame = labels.reshape(size, size)
    return frame


class TestMatchLabels:
    def test_identical_frames(self):
        """Identical frames should match all labels to themselves."""
        frame = _make_simple_frame([[25, 25], [75, 75], [25, 75]])
        mapping = match_labels(frame, frame, min_iou=0.3)
        for label, matched in mapping.items():
            assert matched == label

    def test_shifted_cells(self):
        """Slightly shifted cells should still match."""
        pos_t = [[25, 25], [75, 75], [25, 75]]
        pos_t1 = [[27, 27], [73, 73], [27, 73]]  # small shift
        frame_t = _make_simple_frame(pos_t)
        frame_t1 = _make_simple_frame(pos_t1)
        mapping = match_labels(frame_t, frame_t1, min_iou=0.3)
        # Each cell should match
        for label, matched in mapping.items():
            assert matched is not None

    def test_disappearing_cell(self):
        """A cell that disappears should have no match."""
        pos_t = [[25, 25], [75, 75], [25, 75]]
        pos_t1 = [[25, 25], [75, 75]]  # cell 3 gone
        frame_t = _make_simple_frame(pos_t)
        frame_t1 = _make_simple_frame(pos_t1)
        mapping = match_labels(frame_t, frame_t1, min_iou=0.3)
        # At least one cell should not match (the one that disappeared)
        unmatched = [k for k, v in mapping.items() if v is None]
        assert len(unmatched) >= 1

    def test_high_threshold_rejects(self):
        """Very high IoU threshold should reject imperfect matches."""
        # Use non-overlapping rectangles so IoU is 0
        frame_t = np.zeros((100, 100), dtype=np.int32)
        frame_t[10:30, 10:30] = 1
        frame_t[60:80, 60:80] = 2
        frame_t1 = np.zeros((100, 100), dtype=np.int32)
        frame_t1[30:50, 30:50] = 1  # shifted away from original label 1
        frame_t1[80:100, 80:100] = 2  # shifted away from original label 2
        mapping = match_labels(frame_t, frame_t1, min_iou=0.3)
        # No overlap at all, so nothing should match
        assert all(v is None for v in mapping.values())

    def test_max_area_change_rejects_large_change(self):
        """Matches with area change beyond threshold should be rejected."""
        # Label 1 is small in frame_t, large in frame_t1
        frame_t = np.zeros((100, 100), dtype=np.int32)
        frame_t[40:60, 40:60] = 1  # 20x20 = 400 pixels
        frame_t1 = np.zeros((100, 100), dtype=np.int32)
        frame_t1[10:90, 10:90] = 1  # 80x80 = 6400 pixels
        # They overlap, so IoU > 0
        mapping = match_labels(frame_t, frame_t1, min_iou=0.01, max_area_change=2.0)
        # Area ratio is 6400/400 = 16, way above 2.0
        assert mapping[1] is None

    def test_max_area_change_inf_preserves_behavior(self):
        """Default max_area_change=inf should not reject any matches."""
        frame = _make_simple_frame([[25, 25], [75, 75]])
        mapping = match_labels(frame, frame, min_iou=0.3, max_area_change=float('inf'))
        for label, matched in mapping.items():
            assert matched == label


class TestAssignTrackIds:
    def test_static_cells_same_track(self):
        """Cells that don't move should keep the same track ID across frames."""
        frame = _make_simple_frame([[25, 25], [75, 75]])
        stack = np.stack([frame] * 3)
        tracks = assign_track_ids(stack, min_iou=0.3)
        # Track IDs for label 1 should be consistent
        track_for_label1 = [tracks[f][1] for f in range(3)]
        assert len(set(track_for_label1)) == 1

    def test_new_cell_gets_new_track(self):
        """A cell appearing in frame 2 should get a new track ID."""
        pos_1 = [[25, 25], [75, 75]]
        pos_2 = [[25, 25], [75, 75], [50, 50]]  # new cell appears
        frame1 = _make_simple_frame(pos_1)
        frame2 = _make_simple_frame(pos_2, size=100)
        stack = np.stack([frame1, frame2])
        tracks = assign_track_ids(stack, min_iou=0.3)

        # Frame 0 should have 2 track IDs
        assert len(tracks[0]) == 2
        # Frame 1 should have 3 track IDs
        assert len(tracks[1]) == 3
        # The new cell's track should not match any from frame 0
        track_ids_0 = set(tracks[0].values())
        track_ids_1 = set(tracks[1].values())
        new_ids = track_ids_1 - track_ids_0
        assert len(new_ids) >= 1

    def test_disappearing_cell_across_3_frames(self):
        """Cell disappears mid-sequence; other cells keep their tracks."""
        pos_all = [[25, 25], [75, 75], [50, 25]]
        pos_reduced = [[25, 25], [75, 75]]  # cell 3 gone
        frame_full = _make_simple_frame(pos_all)
        frame_reduced = _make_simple_frame(pos_reduced)
        stack = np.stack([frame_full, frame_full, frame_reduced])
        tracks = assign_track_ids(stack, min_iou=0.3)

        # Frames 0 and 1 should have 3 track IDs
        assert len(tracks[0]) == 3
        assert len(tracks[1]) == 3
        # Frame 2 should have 2 track IDs
        assert len(tracks[2]) == 2


class TestLabelToVertices:
    def test_square_region(self):
        """A simple square region should produce a valid polygon."""
        frame = np.zeros((50, 50), dtype=np.int32)
        frame[10:30, 10:30] = 1
        verts = label_to_vertices(frame, 1)
        assert verts is not None
        assert verts.ndim == 2
        assert verts.shape[1] == 2
        assert len(verts) >= 4  # at least 4 corners

    def test_nonexistent_label(self):
        """Non-existent label should return None."""
        frame = np.zeros((50, 50), dtype=np.int32)
        frame[10:20, 10:20] = 1
        verts = label_to_vertices(frame, 99)
        assert verts is None

    def test_vertices_are_yx_order(self):
        """Vertices should be in (y, x) order."""
        frame = np.zeros((100, 100), dtype=np.int32)
        frame[20:40, 60:80] = 1  # rectangle offset in x
        verts = label_to_vertices(frame, 1)
        assert verts is not None
        # y values should be around 20-40, x around 60-80
        assert np.min(verts[:, 0]) >= 19
        assert np.max(verts[:, 0]) <= 40
        assert np.min(verts[:, 1]) >= 59
        assert np.max(verts[:, 1]) <= 80
