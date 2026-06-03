"""Unit tests for cellflow.segmentation.lineage_graph."""
from __future__ import annotations

import numpy as np

from cellflow.segmentation.lineage_graph import (
    assign_columns,
    build_lineage_graph,
)


def _stack(t: int = 4, size: int = 12) -> np.ndarray:
    return np.zeros((t, size, size), dtype=np.uint32)


def test_one_node_per_occupied_frame() -> None:
    arr = _stack()
    for t in range(4):
        arr[t, 2:4, 2:4] = 1
    arr[0, 8, 8] = 2  # a second track at frame 0 only

    graph = build_lineage_graph(arr)

    assert graph.n_frames == 4
    track1 = [n.t for n in graph.nodes if n.cell_id == 1]
    assert track1 == [0, 1, 2, 3]
    assert [n.t for n in graph.nodes if n.cell_id == 2] == [0]


def test_edges_link_consecutive_present_frames() -> None:
    arr = _stack()
    for t in range(3):
        arr[t, 2:4, 2:4] = 1

    edges = [(e.t0, e.t1) for e in build_lineage_graph(arr).edges if e.cell_id == 1]

    assert edges == [(0, 1), (1, 2)]


def test_edge_skips_a_gap() -> None:
    arr = _stack()
    arr[0, 2:4, 2:4] = 1
    arr[1, 2:4, 2:4] = 1
    # frame 2 missing
    arr[3, 2:4, 2:4] = 1

    edges = [(e.t0, e.t1) for e in build_lineage_graph(arr).edges if e.cell_id == 1]

    assert edges == [(0, 1), (1, 3)]  # the gap is bridged by one edge


def test_assign_columns_orders_by_first_frame_then_id() -> None:
    arr = _stack()
    arr[2, 0, 0] = 5
    arr[0, 1, 1] = 9
    arr[0, 2, 2] = 3

    columns = assign_columns(build_lineage_graph(arr))

    assert columns == {3: 0, 9: 1, 5: 2}


def test_nodes_by_track_sorted_by_frame() -> None:
    arr = _stack()
    for t in range(4):
        arr[t, 0:2, 0:2] = 7

    by_track = build_lineage_graph(arr).nodes_by_track()

    assert [n.t for n in by_track[7]] == [0, 1, 2, 3]


def test_singleton_z_axis_is_squeezed() -> None:
    arr = np.zeros((3, 1, 12, 12), dtype=np.uint32)
    for t in range(3):
        arr[t, 0, 0, 0] = 1

    graph = build_lineage_graph(arr)

    assert graph.n_frames == 3
    assert [n.t for n in graph.nodes if n.cell_id == 1] == [0, 1, 2]


def test_empty_stack_has_no_nodes() -> None:
    graph = build_lineage_graph(_stack())
    assert graph.nodes == ()
    assert graph.edges == ()
    assert graph.n_frames == 4
