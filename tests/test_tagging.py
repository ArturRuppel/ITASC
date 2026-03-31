"""Tests for junction and trajectory tagging API."""
import numpy as np
import pytest

from cellflow.structures import (
    EdgeTrajectory,
    JunctionData,
    TissueGraphFrame,
    TissueGraphTimeSeries,
    InputType,
)
from cellflow.analysis.tagging import (
    tag_trajectory,
    untag_trajectory,
    name_trajectory,
    get_trajectories_by_tag,
    get_trajectory_by_name,
    tag_junction,
    untag_junction,
    get_junctions_by_tag,
    tag_trajectories_near,
    get_all_tags,
    clear_tag,
)


def _make_junction(pair, midpoint=(50.0, 50.0)):
    return JunctionData(
        cell_pair=tuple(sorted(pair)),
        length=10.0,
        coordinates=np.array([[0, 0], [1, 1]], dtype=float),
        midpoint=np.array(midpoint, dtype=float),
    )


def _make_frame(frame_idx, junctions_spec):
    """Create a frame with given junctions. junctions_spec: list of (pair, midpoint)."""
    import networkx as nx
    junctions = {}
    graph = nx.Graph()
    cells = {}
    for pair, midpoint in junctions_spec:
        key = frozenset(pair)
        junctions[key] = _make_junction(pair, midpoint)
        graph.add_edge(*pair)
    return TissueGraphFrame(
        frame=frame_idx,
        graph=graph,
        cells=cells,
        junctions=junctions,
        input_type=InputType.SEGMENTATION,
    )


def _make_series_with_trajectories():
    """Create a series with 2 frames and 3 trajectories."""
    frame0 = _make_frame(0, [
        ((1, 2), (10.0, 10.0)),
        ((2, 3), (50.0, 50.0)),
        ((3, 4), (90.0, 90.0)),
    ])
    frame1 = _make_frame(1, [
        ((1, 2), (10.0, 10.0)),
        ((2, 3), (50.0, 50.0)),
        ((3, 4), (90.0, 90.0)),
    ])

    series = TissueGraphTimeSeries(frames={0: frame0, 1: frame1})

    series.edge_trajectories = {
        1: EdgeTrajectory(
            trajectory_id=1,
            frames=[0, 1],
            cell_pairs=[(1, 2), (1, 2)],
            signed_lengths=[10.0, 10.0],
            coordinates=[np.array([[0, 0]]), np.array([[0, 0]])],
        ),
        2: EdgeTrajectory(
            trajectory_id=2,
            frames=[0, 1],
            cell_pairs=[(2, 3), (2, 3)],
            signed_lengths=[10.0, 10.0],
            coordinates=[np.array([[0, 0]]), np.array([[0, 0]])],
        ),
        3: EdgeTrajectory(
            trajectory_id=3,
            frames=[0, 1],
            cell_pairs=[(3, 4), (3, 4)],
            signed_lengths=[10.0, 10.0],
            coordinates=[np.array([[0, 0]]), np.array([[0, 0]])],
        ),
    }

    return series


# ------------------------------------------------------------------
# Trajectory tagging
# ------------------------------------------------------------------

class TestTrajectoryTagging:
    def test_tag_and_query(self):
        series = _make_series_with_trajectories()
        tag_trajectory(series, 1, "central")
        tag_trajectory(series, 2, "central")
        tag_trajectory(series, 3, "peripheral")

        result = get_trajectories_by_tag(series, "central")
        assert len(result) == 2
        assert {t.trajectory_id for t in result} == {1, 2}

    def test_untag(self):
        series = _make_series_with_trajectories()
        tag_trajectory(series, 1, "central")
        untag_trajectory(series, 1, "central")
        assert get_trajectories_by_tag(series, "central") == []

    def test_untag_nonexistent_tag_is_noop(self):
        series = _make_series_with_trajectories()
        untag_trajectory(series, 1, "nonexistent")  # should not raise

    def test_multiple_tags(self):
        series = _make_series_with_trajectories()
        tag_trajectory(series, 1, "central")
        tag_trajectory(series, 1, "interesting")
        assert series.edge_trajectories[1].tags == {"central", "interesting"}

    def test_name_trajectory(self):
        series = _make_series_with_trajectories()
        name_trajectory(series, 1, "junction_alpha")
        result = get_trajectory_by_name(series, "junction_alpha")
        assert result is not None
        assert result.trajectory_id == 1

    def test_name_not_found(self):
        series = _make_series_with_trajectories()
        assert get_trajectory_by_name(series, "nonexistent") is None

    def test_rename(self):
        series = _make_series_with_trajectories()
        name_trajectory(series, 1, "old_name")
        name_trajectory(series, 1, "new_name")
        assert get_trajectory_by_name(series, "old_name") is None
        assert get_trajectory_by_name(series, "new_name").trajectory_id == 1

    def test_clear_name(self):
        series = _make_series_with_trajectories()
        name_trajectory(series, 1, "temp")
        name_trajectory(series, 1, None)
        assert get_trajectory_by_name(series, "temp") is None


# ------------------------------------------------------------------
# Junction tagging
# ------------------------------------------------------------------

class TestJunctionTagging:
    def test_tag_and_query(self):
        frame = _make_frame(0, [((1, 2), (10, 10)), ((3, 4), (50, 50))])
        tag_junction(frame, (1, 2), "important")
        result = get_junctions_by_tag(frame, "important")
        assert len(result) == 1
        assert frozenset((1, 2)) in result

    def test_untag(self):
        frame = _make_frame(0, [((1, 2), (10, 10))])
        tag_junction(frame, (1, 2), "important")
        untag_junction(frame, (1, 2), "important")
        assert get_junctions_by_tag(frame, "important") == []

    def test_tag_multiple_junctions(self):
        frame = _make_frame(0, [((1, 2), (10, 10)), ((3, 4), (50, 50))])
        tag_junction(frame, (1, 2), "row1")
        tag_junction(frame, (3, 4), "row1")
        result = get_junctions_by_tag(frame, "row1")
        assert len(result) == 2


# ------------------------------------------------------------------
# Bulk tagging
# ------------------------------------------------------------------

class TestBulkTagging:
    def test_tag_near(self):
        series = _make_series_with_trajectories()
        tagged = tag_trajectories_near(
            series,
            location=np.array([10.0, 10.0]),
            radius=15.0,
            tag="nearby",
        )
        assert 1 in tagged
        assert 2 not in tagged
        assert 3 not in tagged

    def test_tag_near_large_radius(self):
        series = _make_series_with_trajectories()
        tagged = tag_trajectories_near(
            series,
            location=np.array([50.0, 50.0]),
            radius=100.0,
            tag="all",
        )
        assert len(tagged) == 3


# ------------------------------------------------------------------
# Global tag queries
# ------------------------------------------------------------------

class TestGlobalQueries:
    def test_get_all_tags(self):
        series = _make_series_with_trajectories()
        tag_trajectory(series, 1, "central")
        tag_junction(series.frames[0], (2, 3), "boundary")
        tags = get_all_tags(series)
        assert tags == {"central", "boundary"}

    def test_get_all_tags_empty(self):
        series = _make_series_with_trajectories()
        assert get_all_tags(series) == set()

    def test_clear_tag(self):
        series = _make_series_with_trajectories()
        tag_trajectory(series, 1, "central")
        tag_trajectory(series, 2, "central")
        tag_junction(series.frames[0], (1, 2), "central")
        count = clear_tag(series, "central")
        assert count == 3
        assert get_all_tags(series) == set()
