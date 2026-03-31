"""Tests for T1 detection and edge trajectory construction."""
import numpy as np
import networkx as nx
import pytest

from cellflow.structures import (
    CellData,
    JunctionData,
    TissueGraphFrame,
    TissueGraphTimeSeries,
    InputType,
)
from cellflow.core.topology import (
    detect_t1_events,
    _validate_t1_transition,
    _detect_t1_between_frames,
)
from cellflow.analysis.trajectories import (
    build_edge_trajectories,
    get_t1_trajectories,
    get_stable_trajectories,
    filter_trajectories,
)


def _make_cell(cell_id, y, x, neighbors):
    """Helper to create a CellData with minimal info."""
    return CellData(
        cell_id=cell_id,
        position=np.array([y, x]),
        area=100.0,
        perimeter=40.0,
        shape_index=4.0,
        num_neighbors=neighbors,
    )


def _make_junction(c1, c2, length=10.0):
    """Helper to create a JunctionData."""
    pair = tuple(sorted((c1, c2)))
    mid = np.array([50.0, 50.0])
    coords = np.array([[50.0, 45.0], [50.0, 55.0]])
    return frozenset(pair), JunctionData(
        cell_pair=pair,
        length=length,
        coordinates=coords,
        midpoint=mid,
    )


def _make_frame(frame_idx, edges, cell_positions=None):
    """Build a TissueGraphFrame from a list of (c1, c2) edges.

    Parameters
    ----------
    frame_idx : int
    edges : list of (int, int) tuples
    cell_positions : dict of cell_id -> (y, x), optional
    """
    G = nx.Graph()
    all_cells = set()
    for c1, c2 in edges:
        G.add_edge(c1, c2)
        all_cells.add(c1)
        all_cells.add(c2)

    if cell_positions is None:
        cell_positions = {c: (50.0, 50.0) for c in all_cells}

    cells = {}
    for c in all_cells:
        y, x = cell_positions.get(c, (50.0, 50.0))
        cells[c] = _make_cell(c, y, x, G.degree(c))

    junctions = {}
    for c1, c2 in edges:
        key, jdata = _make_junction(c1, c2)
        junctions[key] = jdata

    return TissueGraphFrame(
        frame=frame_idx,
        graph=G,
        cells=cells,
        junctions=junctions,
        input_type=InputType.SEGMENTATION,
    )


def _make_t1_series():
    """Create a 3-frame series with a T1 event between frames 0 and 1.

    Topology:
      Frame 0: edges (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
               — all 4 cells connected, including edge (1,2)
      Frame 1: edge (1,2) removed, edge (3,4) stays, NEW edge between...
               Actually let's do a classic 4-cell rosette:

      Frame 0: 1-2 connected, 3-4 not directly connected
               edges: (1,3), (1,4), (2,3), (2,4), (1,2)
      Frame 1: 1-2 loses contact, 3-4 gains contact
               edges: (1,3), (1,4), (2,3), (2,4), (3,4)
      Frame 2: same as frame 1 (stable)
    """
    # Frame 0: cells 1,2 share an edge; 3,4 do not
    edges_0 = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
    # Frame 1: cells 1,2 lose contact; 3,4 gain contact (T1 event)
    edges_1 = [(3, 4), (1, 3), (1, 4), (2, 3), (2, 4)]
    # Frame 2: same topology, stable
    edges_2 = [(3, 4), (1, 3), (1, 4), (2, 3), (2, 4)]

    positions = {1: (30, 50), 2: (70, 50), 3: (50, 30), 4: (50, 70)}

    f0 = _make_frame(0, edges_0, positions)
    f1 = _make_frame(1, edges_1, positions)
    f2 = _make_frame(2, edges_2, positions)

    return TissueGraphTimeSeries(
        frames={0: f0, 1: f1, 2: f2},
        input_type=InputType.SEGMENTATION,
    )


# ---- Tests for _validate_t1_transition ----

class TestValidateT1:
    def test_valid_t1(self):
        """A proper 4-cell T1 should validate."""
        lost = frozenset({1, 2})
        gained = frozenset({3, 4})
        # Connecting edges: (1,3), (1,4), (2,3), (2,4) must exist in both
        edges_prev = {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)}
        edges_next = {(3, 4), (1, 3), (1, 4), (2, 3), (2, 4)}
        assert _validate_t1_transition(lost, gained, edges_prev, edges_next)

    def test_not_four_unique_cells(self):
        """If lost and gained share a cell, not a valid T1."""
        lost = frozenset({1, 2})
        gained = frozenset({2, 3})  # cell 2 in both
        edges_prev = {(1, 2), (1, 3), (2, 3)}
        edges_next = {(2, 3), (1, 3)}
        assert not _validate_t1_transition(lost, gained, edges_prev, edges_next)

    def test_missing_connecting_edge(self):
        """If a connecting edge is missing, not a valid T1."""
        lost = frozenset({1, 2})
        gained = frozenset({3, 4})
        # Missing (2,4) in prev frame
        edges_prev = {(1, 2), (1, 3), (1, 4), (2, 3)}
        edges_next = {(3, 4), (1, 3), (1, 4), (2, 3), (2, 4)}
        assert not _validate_t1_transition(lost, gained, edges_prev, edges_next)


# ---- Tests for detect_t1_events ----

class TestDetectT1:
    def test_detects_single_t1(self):
        """Should find exactly one T1 event in the synthetic series."""
        series = _make_t1_series()
        events = detect_t1_events(series)

        assert len(events) == 1
        event = events[0]
        assert event.frame == 0
        assert set(event.losing_pair) == {1, 2}
        assert set(event.gaining_pair) == {3, 4}
        assert event.all_cells == {1, 2, 3, 4}

    def test_no_t1_in_stable_series(self):
        """No T1 events when topology doesn't change."""
        edges = [(1, 2), (1, 3), (2, 3)]
        f0 = _make_frame(0, edges)
        f1 = _make_frame(1, edges)
        series = TissueGraphTimeSeries(
            frames={0: f0, 1: f1}, input_type=InputType.SEGMENTATION
        )
        events = detect_t1_events(series)
        assert len(events) == 0

    def test_events_stored_on_series(self):
        """detect_t1_events should also store results on the series object."""
        series = _make_t1_series()
        events = detect_t1_events(series)
        assert series.t1_events is events

    def test_t1_location(self):
        """T1 event location should be the midpoint of the lost junction."""
        series = _make_t1_series()
        events = detect_t1_events(series)
        # Our _make_junction always sets midpoint to (50, 50)
        np.testing.assert_array_equal(events[0].location, [50.0, 50.0])

    def test_edge_added_and_removed_but_not_t1(self):
        """An edge disappearing and another appearing with shared cells is not a T1."""
        # Cell 1-2 disappears, cell 2-3 appears — only 3 unique cells
        edges_0 = [(1, 2), (1, 3), (2, 3)]
        edges_1 = [(1, 3), (2, 3), (2, 4)]  # (1,2) gone, (2,4) new
        # But cells {1,2} union {2,4} = {1,2,4} — only 3 cells, fails validation
        f0 = _make_frame(0, edges_0)
        f1 = _make_frame(1, edges_1)
        series = TissueGraphTimeSeries(
            frames={0: f0, 1: f1}, input_type=InputType.SEGMENTATION
        )
        events = detect_t1_events(series)
        assert len(events) == 0


# ---- Tests for edge trajectories ----

class TestEdgeTrajectories:
    def test_stable_trajectories(self):
        """Edges that persist across frames should get continuous trajectories."""
        edges = [(1, 2), (1, 3), (2, 3)]
        f0 = _make_frame(0, edges)
        f1 = _make_frame(1, edges)
        f2 = _make_frame(2, edges)
        series = TissueGraphTimeSeries(
            frames={0: f0, 1: f1, 2: f2}, input_type=InputType.SEGMENTATION
        )

        trajs = build_edge_trajectories(series, [])

        assert len(trajs) == 3  # 3 edges
        for traj in trajs.values():
            assert len(traj.frames) == 3
            assert traj.t1_events == []
            # All lengths positive (no T1)
            assert all(sl > 0 for sl in traj.signed_lengths)

    def test_t1_sign_convention(self):
        """Junction length should flip sign at a T1 event."""
        series = _make_t1_series()
        events = detect_t1_events(series)
        trajs = build_edge_trajectories(series, events)

        # Find the trajectory involved in the T1
        t1_trajs = [t for t in trajs.values() if t.t1_events]
        assert len(t1_trajs) == 1

        t1_traj = t1_trajs[0]
        # The losing edge (1,2) exists at frame 0 → positive
        # The gaining edge (3,4) exists at frames 1,2 → negative (after T1)
        assert len(t1_traj.frames) == 3  # frame 0, 1, 2

        # Frame 0: positive (before T1)
        idx_0 = t1_traj.frames.index(0)
        assert t1_traj.signed_lengths[idx_0] > 0

        # Frames 1 and 2: negative (after T1)
        idx_1 = t1_traj.frames.index(1)
        idx_2 = t1_traj.frames.index(2)
        assert t1_traj.signed_lengths[idx_1] < 0
        assert t1_traj.signed_lengths[idx_2] < 0

    def test_get_t1_trajectories(self):
        """get_t1_trajectories should return only trajectories with T1 events."""
        series = _make_t1_series()
        events = detect_t1_events(series)
        build_edge_trajectories(series, events)

        t1_trajs = get_t1_trajectories(series)
        assert len(t1_trajs) == 1
        assert t1_trajs[0].t1_events[0].frame == 0

    def test_get_stable_trajectories(self):
        """get_stable_trajectories should exclude T1 trajectories."""
        series = _make_t1_series()
        events = detect_t1_events(series)
        build_edge_trajectories(series, events)

        stable = get_stable_trajectories(series, min_frames=2)
        # 4 connecting edges persist across all 3 frames, none have T1s
        assert len(stable) == 4
        for traj in stable:
            assert traj.t1_events == []
            assert len(traj.frames) >= 2

    def test_trajectories_stored_on_series(self):
        """build_edge_trajectories should store results on the series."""
        series = _make_t1_series()
        events = detect_t1_events(series)
        trajs = build_edge_trajectories(series, events)
        assert series.edge_trajectories is trajs


# ---- Tests for T1 detection parameters ----

class TestT1DetectionParams:
    def test_min_junction_length_filters_short_edges(self):
        """Short junctions below threshold should be excluded from T1 detection."""
        series = _make_t1_series()
        # Default junction length from _make_junction is 10.0
        # With threshold above 10, the T1 should not be detected
        events = detect_t1_events(series, min_junction_length=15.0)
        assert len(events) == 0

    def test_min_junction_length_zero_preserves_behavior(self):
        """Default min_junction_length=0 should detect T1 as before."""
        series = _make_t1_series()
        events = detect_t1_events(series, min_junction_length=0.0)
        assert len(events) == 1

    def test_max_t1_distance_filters_distant_events(self):
        """T1 events with distant lost/gained midpoints should be filtered."""
        # Create a series where lost and gained midpoints are far apart
        edges_0 = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
        edges_1 = [(3, 4), (1, 3), (1, 4), (2, 3), (2, 4)]

        positions = {1: (0, 0), 2: (10, 0), 3: (0, 100), 4: (10, 100)}

        f0 = _make_frame(0, edges_0, positions)
        f1 = _make_frame(1, edges_1, positions)

        # Set different midpoints for lost and gained junctions
        lost_key = frozenset({1, 2})
        gained_key = frozenset({3, 4})
        f0.junctions[lost_key] = JunctionData(
            cell_pair=(1, 2), length=10.0,
            coordinates=np.array([[0, 0], [10, 0]]),
            midpoint=np.array([5.0, 0.0]),
        )
        f1.junctions[gained_key] = JunctionData(
            cell_pair=(3, 4), length=10.0,
            coordinates=np.array([[0, 100], [10, 100]]),
            midpoint=np.array([5.0, 100.0]),
        )

        series = TissueGraphTimeSeries(
            frames={0: f0, 1: f1}, input_type=InputType.SEGMENTATION
        )

        # With a tight distance constraint, the T1 should be filtered out
        events = detect_t1_events(series, max_t1_distance=10.0)
        assert len(events) == 0

        # With no constraint, it should be detected
        events = detect_t1_events(series)
        assert len(events) == 1

    def test_max_t1_distance_inf_preserves_behavior(self):
        """Default max_t1_distance=inf should detect T1 as before."""
        series = _make_t1_series()
        events = detect_t1_events(series, max_t1_distance=float('inf'))
        assert len(events) == 1


# ---- Tests for filter_trajectories ----

class TestFilterTrajectories:
    def _make_series_with_trajectories(self):
        """Build a series and trajectories for filtering tests."""
        series = _make_t1_series()
        events = detect_t1_events(series)
        build_edge_trajectories(series, events)
        return series

    def test_min_frames_filter(self):
        """Trajectories shorter than min_frames should be excluded."""
        series = self._make_series_with_trajectories()
        # All stable trajectories span 3 frames, T1 trajectory spans 3 frames
        filtered = filter_trajectories(series, min_frames=4)
        assert len(filtered) == 0

        filtered = filter_trajectories(series, min_frames=3)
        assert len(filtered) == len(series.edge_trajectories)

    def test_min_completeness_filter(self):
        """Trajectories below completeness threshold should be excluded."""
        series = self._make_series_with_trajectories()
        # Series has 3 frames, all trajectories span 3 frames = 100% complete
        filtered = filter_trajectories(series, min_completeness=1.0)
        assert len(filtered) == len(series.edge_trajectories)

        # Add a short trajectory (only 1 frame)
        from cellflow.structures import EdgeTrajectory
        series.edge_trajectories[999] = EdgeTrajectory(
            trajectory_id=999, frames=[0], cell_pairs=[(10, 11)],
            signed_lengths=[5.0], coordinates=[np.zeros((2, 2))],
            t1_events=[],
        )
        filtered = filter_trajectories(series, min_completeness=0.5)
        assert 999 not in filtered

    def test_max_gap_filter(self):
        """Trajectories with gaps exceeding max_gap should be excluded."""
        series = self._make_series_with_trajectories()
        # Add a trajectory with a gap (exists in frame 0 and 2 but not 1)
        from cellflow.structures import EdgeTrajectory
        series.edge_trajectories[998] = EdgeTrajectory(
            trajectory_id=998, frames=[0, 2], cell_pairs=[(10, 11), (10, 11)],
            signed_lengths=[5.0, 5.0], coordinates=[np.zeros((2, 2))] * 2,
            t1_events=[],
        )

        # max_gap=0: no gaps allowed, trajectory 998 should be excluded
        filtered = filter_trajectories(series, max_gap=0)
        assert 998 not in filtered

        # max_gap=1: 1-frame gap allowed, trajectory 998 should pass
        filtered = filter_trajectories(series, max_gap=1)
        assert 998 in filtered

    def test_default_params_keep_all(self):
        """Default parameters should keep all trajectories."""
        series = self._make_series_with_trajectories()
        filtered = filter_trajectories(series)
        assert len(filtered) == len(series.edge_trajectories)
