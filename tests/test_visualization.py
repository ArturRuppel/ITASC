"""Tests for visualization functions."""
import numpy as np
import pytest

from napariTissueGraph.core.graph import (
    build_from_labels,
    extract_graphs_from_labels,
    assign_tracking_labels,
)
from napariTissueGraph.core.label_tracking import assign_track_ids
from napariTissueGraph.napari.visualization import (
    build_tracked_centroids,
    build_tracked_labels,
    build_track_breaks,
    build_trajectory_lines,
    build_trajectory_lines_with_features,
    build_all_centroids,
)


class TestTrackedLabels:
    def test_output_shape_matches_input(self, label_stack):
        track_map = assign_track_ids(label_stack)
        result = build_tracked_labels(label_stack, track_map)
        assert result.shape == label_stack.shape

    def test_background_stays_zero(self, label_stack):
        track_map = assign_track_ids(label_stack)
        result = build_tracked_labels(label_stack, track_map)
        bg_mask = label_stack == 0
        assert np.all(result[bg_mask] == 0)

    def test_tracked_cells_get_track_ids(self, label_stack):
        track_map = assign_track_ids(label_stack)
        result = build_tracked_labels(label_stack, track_map)
        for frame_idx in range(len(label_stack)):
            frame_tracks = track_map.get(frame_idx, {})
            for cell_id, track_id in frame_tracks.items():
                mask = label_stack[frame_idx] == cell_id
                if mask.any():
                    assert np.all(result[frame_idx][mask] == track_id)

    def test_untracked_cells_get_unique_ids(self, label_stack):
        """Untracked cells should get IDs above max track_id."""
        # Use an empty track_map so all cells are untracked
        result = build_tracked_labels(label_stack, {})
        # All non-zero pixels should have IDs >= 1
        assert np.all(result[label_stack > 0] >= 1)
        # Each unique cell should have a unique ID
        for frame_idx in range(len(label_stack)):
            cell_ids = np.unique(label_stack[frame_idx])
            cell_ids = cell_ids[cell_ids > 0]
            result_ids = set()
            for cid in cell_ids:
                mask = label_stack[frame_idx] == cid
                rid = result[frame_idx][mask][0]
                result_ids.add(rid)
            assert len(result_ids) == len(cell_ids)

    def test_no_cells_returns_zeros(self):
        """Empty label stack should return zeros."""
        stack = np.zeros((3, 10, 10), dtype=np.int32)
        result = build_tracked_labels(stack, {})
        assert np.all(result == 0)


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


class TestTrajectoryLinesWithFeatures:
    def test_output_format(self, label_stack):
        series = build_from_labels(label_stack)
        from napariTissueGraph.core.topology import detect_t1_events
        from napariTissueGraph.analysis.trajectories import build_edge_trajectories
        detect_t1_events(series)
        build_edge_trajectories(series, series.t1_events)

        lines, colors, features = build_trajectory_lines_with_features(series)
        assert colors.shape[1] == 4  # RGBA
        assert len(lines) == len(features)
        assert len(lines) == len(colors)
        for line in lines:
            assert line.shape[1] == 3  # (frame, y, x)

    def test_features_columns(self, label_stack):
        series = build_from_labels(label_stack)
        from napariTissueGraph.core.topology import detect_t1_events
        from napariTissueGraph.analysis.trajectories import build_edge_trajectories
        detect_t1_events(series)
        build_edge_trajectories(series, series.t1_events)

        _, _, features = build_trajectory_lines_with_features(series)
        assert "trajectory_id" in features.columns
        assert "cell_pair_a" in features.columns
        assert "cell_pair_b" in features.columns
        assert "frame" in features.columns
        assert "tags" in features.columns
        assert "name" in features.columns

    def test_one_shape_per_junction_per_frame(self, label_stack):
        series = build_from_labels(label_stack)
        from napariTissueGraph.core.topology import detect_t1_events
        from napariTissueGraph.analysis.trajectories import build_edge_trajectories
        detect_t1_events(series)
        build_edge_trajectories(series, series.t1_events)

        lines, _, _ = build_trajectory_lines_with_features(series)
        # Count junctions with >= 2 coordinate points
        total = sum(
            1 for f in series.frames.values()
            for jd in f.junctions.values()
            if len(jd.coordinates) >= 2
        )
        assert len(lines) == total

    def test_show_only_tagged_filters(self, label_stack):
        series = build_from_labels(label_stack)
        from napariTissueGraph.core.topology import detect_t1_events
        from napariTissueGraph.analysis.trajectories import build_edge_trajectories
        detect_t1_events(series)
        build_edge_trajectories(series, series.t1_events)

        # No tags → show_only_tagged returns empty
        lines, _, _ = build_trajectory_lines_with_features(
            series, show_only_tagged=True,
        )
        assert len(lines) == 0

    def test_color_by_tags(self, label_stack):
        series = build_from_labels(label_stack)
        from napariTissueGraph.core.topology import detect_t1_events
        from napariTissueGraph.analysis.trajectories import build_edge_trajectories
        from napariTissueGraph.analysis.tagging import tag_trajectory
        detect_t1_events(series)
        build_edge_trajectories(series, series.t1_events)

        traj_ids = sorted(series.edge_trajectories.keys())
        if traj_ids:
            tag_trajectory(series, traj_ids[0], "central")
            _, colors, features = build_trajectory_lines_with_features(
                series, color_by_tags=True,
            )
            # Tagged shapes should have tag color, not trajectory color
            gray = np.array([0.5, 0.5, 0.5, 0.5])
            n_colored = sum(1 for c in colors if not np.allclose(c, gray))
            assert n_colored > 0

    def test_show_only_tagged_returns_tagged(self, label_stack):
        series = build_from_labels(label_stack)
        from napariTissueGraph.core.topology import detect_t1_events
        from napariTissueGraph.analysis.trajectories import build_edge_trajectories
        from napariTissueGraph.analysis.tagging import tag_trajectory
        detect_t1_events(series)
        build_edge_trajectories(series, series.t1_events)

        traj_ids = sorted(series.edge_trajectories.keys())
        if traj_ids:
            tag_trajectory(series, traj_ids[0], "central")
            _, _, features = build_trajectory_lines_with_features(
                series, show_only_tagged=True,
            )
            assert len(features) > 0
            for _, row in features.iterrows():
                assert row["tags"] != ""

    def test_backward_compat_with_build_trajectory_lines(self, label_stack):
        """build_trajectory_lines should still work and return same line count."""
        from napariTissueGraph.napari.visualization import build_all_junction_lines
        series = build_from_labels(label_stack)
        j_lines, _ = build_all_junction_lines(series)
        t_lines, _ = build_trajectory_lines(series)
        assert len(j_lines) == len(t_lines)
