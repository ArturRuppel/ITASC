"""Tests for the label-agnostic neighborhood & density derivations."""
import numpy as np
import pytest

from itasc.contact_analysis.contacts.neighborhood import (
    cell_density,
    cell_neighbor_counts,
)
from itasc.contact_analysis.contacts.reader import PositionContactAnalysis


def _analysis(cells, edges) -> PositionContactAnalysis:
    return PositionContactAnalysis(
        cells=cells,
        edges=edges,
        t1_events={},
        cell_tracked_labels_path="cells.tif",
        nucleus_tracked_labels_path="nuclei.tif",
        _edge_coord_y=np.empty(0),
        _edge_coord_x=np.empty(0),
    )


def _cells(rows):
    """Column-major cells table from (frame, cell_id) rows."""
    if rows:
        frame, cid = zip(*rows)
    else:
        frame, cid = (), ()
    return {
        "frame": np.asarray(frame, dtype=np.int64),
        "cell_id": np.asarray(cid, dtype=np.int64),
    }


def _edges(rows):
    """Column-major edges from (frame, cell_a, cell_b[, kind]) rows; kind defaults cell_cell."""
    frame, a, b, kind, length = [], [], [], [], []
    for row in rows:
        fr, ca, cb = row[0], row[1], row[2]
        k = row[3] if len(row) > 3 else "cell_cell"
        frame.append(fr)
        a.append(ca)
        b.append(cb)
        kind.append(k)
        length.append(1.0)
    return {
        "frame": np.asarray(frame, dtype=np.int64),
        "cell_a": np.asarray(a, dtype=np.int64),
        "cell_b": np.asarray(b, dtype=np.int64),
        "kind": np.asarray(kind, dtype=object),
        "length": np.asarray(length, dtype=float),
    }


# --------------------------------------------------------------- neighbor counts
def test_degree_counts_distinct_cell_cell_neighbors():
    cells = _cells([(0, 1), (0, 2), (0, 3), (0, 4)])
    edges = _edges([(0, 1, 2), (0, 2, 3)])
    table = cell_neighbor_counts(_analysis(cells, edges))
    degree = dict(zip(table["cell_id"].tolist(), table["n_neighbors"].tolist()))
    assert degree == {1: 1, 2: 2, 3: 1, 4: 0}  # cell 4 isolated → degree 0


def test_border_edges_excluded_from_degree():
    cells = _cells([(0, 1), (0, 2)])
    edges = _edges([(0, 1, 2), (0, 1, 0, "border")])
    table = cell_neighbor_counts(_analysis(cells, edges))
    degree = dict(zip(table["cell_id"].tolist(), table["n_neighbors"].tolist()))
    assert degree == {1: 1, 2: 1}  # the border edge added no neighbor


def test_fragmented_boundary_counts_as_one_neighbor():
    cells = _cells([(0, 1), (0, 2)])
    # The 1–2 boundary split across three edge rows must count as one neighbor.
    edges = _edges([(0, 1, 2), (0, 1, 2), (0, 1, 2)])
    table = cell_neighbor_counts(_analysis(cells, edges))
    degree = dict(zip(table["cell_id"].tolist(), table["n_neighbors"].tolist()))
    assert degree == {1: 1, 2: 1}


# -------------------------------------------------------------------- density
def test_density_is_cells_over_area():
    # cell_density counts straight off a frame -> cell_ids map (the cell labels),
    # not the contact graph. One ``all`` row per frame (label-agnostic).
    frame_cells = {0: [1, 2, 3, 4]}
    table = cell_density(frame_cells, fov_area_mm2=2.0)
    rows = {
        str(lab): (int(n), float(d))
        for lab, n, d in zip(table["label"], table["n_cells"], table["density"])
    }
    assert rows == {"all": (4, 2.0)}  # every cell, one total row


def test_density_one_all_row_per_frame():
    table = cell_density({0: [1, 2], 1: [1, 2, 3]}, fov_area_mm2=1.0)
    assert table["label"].tolist() == ["all", "all"]
    assert table["frame"].tolist() == [0, 1]
    assert table["n_cells"].tolist() == [2, 3]


def test_density_requires_a_positive_area():
    # The field-of-view area is required — no silent fallback.
    with pytest.raises(ValueError):
        cell_density({0: [1, 2]}, fov_area_mm2=0.0)
