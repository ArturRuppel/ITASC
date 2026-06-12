"""Tests for the neighborhood & density derivations."""
import numpy as np
import pytest

from cellflow.aggregate_quantification.contacts.neighborhood import (
    cell_density,
    cell_neighbor_counts,
    contact_type_zscores,
    neighbor_enrichment,
)
from cellflow.aggregate_quantification.contacts.reader import PositionContactAnalysis


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


# ------------------------------------------------------------ neighbor enrichment
def test_enrichment_matches_expected_self_excluded():
    # 4 cells, two A two B; A's only neighbor is the other A (fully sorted).
    cells = _cells([(0, 1), (0, 2), (0, 3), (0, 4)])
    edges = _edges([(0, 1, 2), (0, 3, 4)])
    labels = {1: "A", 2: "A", 3: "B", 4: "B"}
    table = neighbor_enrichment(_analysis(cells, edges), labels)

    rows = {
        (int(c), str(f), str(n)): (int(o), float(e), float(en))
        for c, f, n, o, e, en in zip(
            table["cell_id"], table["focal_label"], table["neighbor_label"],
            table["observed"], table["expected"], table["enrichment"],
        )
    }
    # Focal cell 1 (A), degree 1: expected A = 1*(2-1)/(4-1) = 1/3; observed 1 → 3.
    obs_aa, exp_aa, enr_aa = rows[(1, "A", "A")]
    assert obs_aa == 1
    assert exp_aa == 1 / 3
    assert enr_aa == 3.0
    # Expected B = 1*2/3 = 2/3; observed 0 → enrichment 0 (heterotypic avoided).
    obs_ab, exp_ab, enr_ab = rows[(1, "A", "B")]
    assert obs_ab == 0
    assert exp_ab == 2 / 3
    assert enr_ab == 0.0


def test_homotypic_enriched_heterotypic_depleted_when_sorted():
    cells = _cells([(0, 1), (0, 2), (0, 3), (0, 4)])
    edges = _edges([(0, 1, 2), (0, 3, 4)])
    labels = {1: "A", 2: "A", 3: "B", 4: "B"}
    table = neighbor_enrichment(_analysis(cells, edges), labels)
    homo = table["enrichment"][table["focal_label"] == table["neighbor_label"]]
    hetero = table["enrichment"][table["focal_label"] != table["neighbor_label"]]
    assert (homo > 1).all()
    assert (hetero < 1).all()


def test_enrichment_nan_when_expected_zero():
    # An isolated labeled cell has degree 0 → expected 0 → enrichment NaN.
    cells = _cells([(0, 1), (0, 2), (0, 3)])
    edges = _edges([(0, 2, 3)])  # cell 1 has no edges
    labels = {1: "A", 2: "A", 3: "B"}
    table = neighbor_enrichment(_analysis(cells, edges), labels)
    cell1 = table["enrichment"][table["cell_id"] == 1]
    assert cell1.size > 0
    assert np.isnan(cell1).all()


def test_unclassified_neighbor_dropped_and_excluded_from_abundance():
    # Cell 3 is unclassified: it is not a neighbor-type, nor counted in N.
    cells = _cells([(0, 1), (0, 2), (0, 3)])
    edges = _edges([(0, 1, 2), (0, 1, 3)])  # cell 1 touches A(2) and unclassified(3)
    labels = {1: "A", 2: "A"}
    table = neighbor_enrichment(_analysis(cells, edges), labels)
    # Only label A present → only (A, A) rows; no "unclassified" neighbor label.
    assert set(table["neighbor_label"].tolist()) == {"A"}
    # Cell 1's labeled degree is 1 (only cell 2); abundance N = 2 (cells 1,2).
    obs = table["observed"][table["cell_id"] == 1]
    exp = table["expected"][table["cell_id"] == 1]
    assert obs.tolist() == [1]
    assert exp.tolist() == [1 * (2 - 1) / (2 - 1)]  # = 1.0


def test_unclassified_focal_cell_emits_no_rows():
    cells = _cells([(0, 1), (0, 2)])
    edges = _edges([(0, 1, 2)])
    labels = {1: "A"}  # cell 2 unclassified
    table = neighbor_enrichment(_analysis(cells, edges), labels)
    assert set(table["cell_id"].tolist()) == {1}


# ------------------------------------------------------------- contact z-scores
def _zrows(table):
    return {
        (int(f), str(t)): dict(
            observed=int(o), mean=float(m), sd=float(s), z=float(z),
            obs_frac=float(of), exp_frac=float(ef),
        )
        for f, t, o, m, s, z, of, ef in zip(
            table["frame"], table["contact_type"], table["observed_count"],
            table["mean_null"], table["sd_null"], table["z_score"],
            table["observed_fraction"], table["expected_fraction"],
        )
    }


def test_segregated_layout_gives_positive_homotypic_z():
    cells = _cells([(0, 1), (0, 2), (0, 3), (0, 4)])
    edges = _edges([(0, 1, 2), (0, 3, 4)])  # AA and BB only
    labels = {1: "A", 2: "A", 3: "B", 4: "B"}
    rows = _zrows(contact_type_zscores(_analysis(cells, edges), labels, n_shuffles=2000, seed=1))
    assert rows[(0, "A·A")]["z"] > 0
    assert rows[(0, "B·B")]["z"] > 0
    assert rows[(0, "A·B")]["z"] < 0  # heterotypic depleted vs chance


def test_mixed_layout_gives_positive_heterotypic_z():
    cells = _cells([(0, 1), (0, 2), (0, 3), (0, 4)])
    edges = _edges([(0, 1, 3), (0, 2, 4)])  # all heterotypic AB
    labels = {1: "A", 2: "A", 3: "B", 4: "B"}
    rows = _zrows(contact_type_zscores(_analysis(cells, edges), labels, n_shuffles=2000, seed=1))
    assert rows[(0, "A·B")]["z"] > 0
    assert rows[(0, "A·A")]["z"] < 0


def test_expected_fraction_is_analytic_label_product():
    cells = _cells([(0, 1), (0, 2), (0, 3), (0, 4)])
    edges = _edges([(0, 1, 2), (0, 3, 4)])
    labels = {1: "A", 2: "A", 3: "B", 4: "B"}  # f_A = f_B = 0.5
    rows = _zrows(contact_type_zscores(_analysis(cells, edges), labels, n_shuffles=50))
    assert rows[(0, "A·A")]["exp_frac"] == 0.25
    assert rows[(0, "B·B")]["exp_frac"] == 0.25
    assert rows[(0, "A·B")]["exp_frac"] == 0.5
    # Observed fraction: 1 of 2 contacts is AA.
    assert rows[(0, "A·A")]["obs_frac"] == 0.5


def test_zscore_is_deterministic_under_seed():
    cells = _cells([(0, 1), (0, 2), (0, 3), (0, 4)])
    edges = _edges([(0, 1, 2), (0, 3, 4), (0, 2, 3)])
    labels = {1: "A", 2: "A", 3: "B", 4: "B"}
    a = contact_type_zscores(_analysis(cells, edges), labels, n_shuffles=500, seed=7)
    b = contact_type_zscores(_analysis(cells, edges), labels, n_shuffles=500, seed=7)
    np.testing.assert_array_equal(a["z_score"], b["z_score"])
    np.testing.assert_array_equal(a["mean_null"], b["mean_null"])


def test_zscore_nan_when_sd_null_zero_single_label():
    cells = _cells([(0, 1), (0, 2)])
    edges = _edges([(0, 1, 2)])
    labels = {1: "A", 2: "A"}  # one label only → no shuffle variance
    rows = _zrows(contact_type_zscores(_analysis(cells, edges), labels, n_shuffles=100))
    assert rows[(0, "A·A")]["sd"] == 0.0
    assert np.isnan(rows[(0, "A·A")]["z"])


def test_zscore_excludes_unclassified_edges():
    cells = _cells([(0, 1), (0, 2), (0, 3)])
    # Edge (1,3) touches unclassified cell 3 → not counted.
    edges = _edges([(0, 1, 2), (0, 1, 3)])
    labels = {1: "A", 2: "A"}
    rows = _zrows(contact_type_zscores(_analysis(cells, edges), labels, n_shuffles=10))
    assert rows[(0, "A·A")]["observed"] == 1  # only the (1,2) contact


# -------------------------------------------------------------------- density
def test_density_is_cells_over_area():
    # cell_density counts straight off a frame -> cell_ids map (the cell labels),
    # not the contact graph.
    frame_cells = {0: [1, 2, 3, 4]}
    labels = {1: "A", 2: "A", 3: "B"}  # cell 4 unclassified
    table = cell_density(frame_cells, labels, fov_area_mm2=2.0)
    rows = {
        str(lab): (int(n), float(d))
        for lab, n, d in zip(table["label"], table["n_cells"], table["density"])
    }
    assert rows["all"] == (4, 2.0)  # every cell incl. unclassified
    assert rows["A"] == (2, 1.0)
    assert rows["B"] == (1, 0.5)
    # Per-label counts sum to the labeled-cell count.
    assert rows["A"][0] + rows["B"][0] == 3


def test_density_empty_labels_only_all_row():
    table = cell_density({0: [1, 2]}, {}, fov_area_mm2=1.0)
    assert table["label"].tolist() == ["all"]
    assert table["n_cells"].tolist() == [2]


def test_density_requires_a_positive_area():
    # The field-of-view area is required — no silent fallback.
    with pytest.raises(ValueError):
        cell_density({0: [1, 2]}, {1: "A"}, fov_area_mm2=0.0)
