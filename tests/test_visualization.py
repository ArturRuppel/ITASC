"""Tests for visualization functions."""
import numpy as np
import pytest

from napariTissueGraph.core.graph import (
    build_from_labels,
    extract_graphs_from_labels,
    assign_tracking_labels,
)
from napariTissueGraph.napari.visualization import (
    build_tracked_centroids,
    build_track_breaks,
    build_trajectory_lines,
    build_all_centroids,
)


class TestTrackedCentroids:
    def test_shapes_match(self, label_stack):
        series = build_from_labels(label_stack)
        positions, colors, cmap = build_tracked_centroids(series)
        assert positions.shape[0] == colors.shape[0]
        assert positions.shape[1] == 3
        assert colors.shape[1] == 4

    def test_untracked_cells_get_gray(self, label_stack):
        """Cells with track_id=None should get gray [0.5, 0.5, 0.5, 0.5]."""
        series = extract_graphs_from_labels(label_stack)
        # No tracking assigned — all should be gray
        positions, colors, cmap = build_tracked_centroids(series)
        assert len(cmap) == 0  # no tracks
        expected_gray = np.array([0.5, 0.5, 0.5, 0.5])
        for i in range(len(colors)):
            np.testing.assert_allclose(colors[i], expected_gray)

    def test_tracked_cells_not_gray(self, label_stack):
        """Tracked cells should have non-gray colors."""
        series = build_from_labels(label_stack)
        positions, colors, cmap = build_tracked_centroids(series)
        # At least some cells should have non-gray colors
        assert len(cmap) > 0
        gray = np.array([0.5, 0.5, 0.5, 0.5])
        n_colored = sum(1 for c in colors if not np.allclose(c, gray))
        assert n_colored > 0

    def test_matches_centroid_count(self, label_stack):
        """Should have same number of points as build_all_centroids."""
        series = build_from_labels(label_stack)
        plain = build_all_centroids(series)
        tracked, _, _ = build_tracked_centroids(series)
        assert len(tracked) == len(plain)


class TestTrackBreaks:
    def test_no_breaks_for_stable_data(self, label_stack):
        """Identical frames → all tracks span full range → no breaks."""
        series = build_from_labels(label_stack)
        positions, types = build_track_breaks(series)
        assert len(positions) == 0
        assert len(types) == 0

    def test_empty_series(self):
        """Single frame should return empty."""
        from napariTissueGraph.structures import TissueGraphTimeSeries
        series = TissueGraphTimeSeries(frames={})
        positions, types = build_track_breaks(series)
        assert len(positions) == 0


class TestTrajectoryLines:
    def test_output_format(self, label_stack):
        """Output should be a list of arrays + color array."""
        series = build_from_labels(label_stack)
        lines, colors = build_trajectory_lines(series)
        # With no trajectories built, all should be gray
        if len(lines) > 0:
            assert colors.shape[1] == 4
            for line in lines:
                assert line.shape[1] == 3  # (frame, y, x)

    def test_same_line_count_as_junction_lines(self, label_stack):
        """Should produce same number of lines as build_all_junction_lines."""
        from napariTissueGraph.napari.visualization import build_all_junction_lines
        series = build_from_labels(label_stack)
        j_lines, _ = build_all_junction_lines(series)
        t_lines, _ = build_trajectory_lines(series)
        assert len(j_lines) == len(t_lines)
