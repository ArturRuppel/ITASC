"""Tests for the signed central junction-length reaction coordinate."""
import numpy as np

from cellflow.aggregate_quantification.contacts.energetics import (
    signed_central_junction_lengths,
)
from cellflow.aggregate_quantification.contacts.reader import PositionContactAnalysis


def _analysis(edges, t1_events) -> PositionContactAnalysis:
    """A minimal PositionContactAnalysis carrying just edges + t1_events."""
    return PositionContactAnalysis(
        cells={},
        edges=edges,
        t1_events=t1_events,
        cell_tracked_labels_path="cells.tif",
        nucleus_tracked_labels_path="nuclei.tif",
        _edge_coord_y=np.empty(0),
        _edge_coord_x=np.empty(0),
    )


def _edges(rows):
    """Column-major edges table from (frame, cell_a, cell_b, length) rows."""
    frame, a, b, length = zip(*rows)
    return {
        "frame": np.asarray(frame, dtype=np.int64),
        "cell_a": np.asarray(a, dtype=np.int64),
        "cell_b": np.asarray(b, dtype=np.int64),
        "length": np.asarray(length, dtype=float),
    }


def _one_event(losing=(1, 2), gaining=(3, 4)):
    return {
        "t1_event_id": np.array([7], dtype=np.int64),
        "losing_cell_a": np.array([losing[0]], dtype=np.int64),
        "losing_cell_b": np.array([losing[1]], dtype=np.int64),
        "gaining_cell_a": np.array([gaining[0]], dtype=np.int64),
        "gaining_cell_b": np.array([gaining[1]], dtype=np.int64),
    }


def test_losing_edge_is_negative_gaining_is_positive():
    # The losing edge (1,2) exists in frames 0-1 (shrinking); the gaining edge
    # (3,4) exists in frames 2-3 (growing).
    edges = _edges([
        (0, 1, 2, 5.0),
        (1, 1, 2, 2.0),
        (2, 3, 4, 2.0),
        (3, 3, 4, 5.0),
    ])
    table = signed_central_junction_lengths(_analysis(edges, _one_event()))

    by_frame = dict(zip(table["frame"].tolist(), table["signed_length"].tolist()))
    assert by_frame == {0: -5.0, 1: -2.0, 2: 2.0, 3: 5.0}
    roles = dict(zip(table["frame"].tolist(), table["role"].tolist()))
    assert roles == {0: "losing", 1: "losing", 2: "gaining", 3: "gaining"}
    assert set(table["t1_event_id"].tolist()) == {7}


def test_pixel_size_scales_magnitude_keeps_sign():
    edges = _edges([(0, 1, 2, 4.0), (1, 3, 4, 6.0)])
    table = signed_central_junction_lengths(
        _analysis(edges, _one_event()), pixel_size_um=0.5
    )
    by_frame = dict(zip(table["frame"].tolist(), table["signed_length"].tolist()))
    assert by_frame == {0: -2.0, 1: 3.0}


def test_event_with_no_matching_edges_contributes_nothing():
    # Only unrelated edges present; the event's losing/gaining pairs never appear.
    edges = _edges([(0, 9, 10, 5.0)])
    table = signed_central_junction_lengths(_analysis(edges, _one_event()))
    assert table["signed_length"].size == 0
    assert table["role"].dtype == object  # still typed


def test_no_events_returns_empty_typed_table():
    edges = _edges([(0, 1, 2, 5.0)])
    empty_events = {
        "t1_event_id": np.empty(0, dtype=np.int64),
        "losing_cell_a": np.empty(0, dtype=np.int64),
        "losing_cell_b": np.empty(0, dtype=np.int64),
        "gaining_cell_a": np.empty(0, dtype=np.int64),
        "gaining_cell_b": np.empty(0, dtype=np.int64),
    }
    table = signed_central_junction_lengths(_analysis(edges, empty_events))
    assert set(table) == {"t1_event_id", "frame", "signed_length", "role"}
    assert all(v.size == 0 for v in table.values())


def test_cell_pair_order_is_ignored():
    # The edge is stored as (2,1) but the event names the losing pair (1,2).
    edges = _edges([(0, 2, 1, 5.0)])
    table = signed_central_junction_lengths(_analysis(edges, _one_event(losing=(1, 2))))
    assert table["signed_length"].tolist() == [-5.0]


def test_all_frames_an_edge_exists_are_pooled():
    # The losing pair persists across many frames (no ± window) — every frame is a
    # sample.
    edges = _edges([(f, 1, 2, float(10 - f)) for f in range(6)])
    table = signed_central_junction_lengths(_analysis(edges, _one_event()))
    assert table["signed_length"].size == 6
    assert (table["signed_length"] < 0).all()
