"""Tests for propagating NLS subpopulation labels onto contacts."""
import numpy as np

from itasc.contact_analysis.contacts.contact_labels import label_contacts
from itasc.contact_analysis.contacts.reader import PositionContactAnalysis


def _analysis(edges) -> PositionContactAnalysis:
    """A minimal PositionContactAnalysis carrying just the edges table."""
    return PositionContactAnalysis(
        cells={},
        edges=edges,
        t1_events={},
        cell_tracked_labels_path="cells.tif",
        nucleus_tracked_labels_path="nuclei.tif",
        _edge_coord_y=np.empty(0),
        _edge_coord_x=np.empty(0),
    )


def _edges(rows):
    """Column-major edges table from (frame, edge_id, cell_a, cell_b, kind, length)."""
    if not rows:
        return {
            "frame": np.empty(0, dtype=np.int64),
            "edge_id": np.empty(0, dtype=np.int64),
            "cell_a": np.empty(0, dtype=np.int64),
            "cell_b": np.empty(0, dtype=np.int64),
            "kind": np.empty(0, dtype=object),
            "length": np.empty(0, dtype=float),
        }
    frame, edge_id, a, b, kind, length = zip(*rows)
    return {
        "frame": np.asarray(frame, dtype=np.int64),
        "edge_id": np.asarray(edge_id, dtype=np.int64),
        "cell_a": np.asarray(a, dtype=np.int64),
        "cell_b": np.asarray(b, dtype=np.int64),
        "kind": np.asarray(kind, dtype=object),
        "length": np.asarray(length, dtype=float),
    }


def test_homotypic_and_heterotypic_pairs():
    edges = _edges([
        (0, 0, 1, 2, "cell_cell", 5.0),  # A-A
        (0, 1, 3, 4, "cell_cell", 6.0),  # B-B
        (0, 2, 1, 3, "cell_cell", 7.0),  # A-B
    ])
    labels = {1: "A", 2: "A", 3: "B", 4: "B"}
    table = label_contacts(_analysis(edges), labels)

    assert table["contact_label"].tolist() == ["A-A", "B-B", "A-B"]
    assert table["homotypic"].tolist() == [True, True, False]
    assert table["fully_classified"].all()
    assert table["length"].tolist() == [5.0, 6.0, 7.0]


def test_pair_is_sorted_regardless_of_cell_order():
    # Edge stored as (3, 1) with labels B, A → still "A-B".
    edges = _edges([(0, 0, 1, 3, "cell_cell", 5.0)])
    table = label_contacts(_analysis(edges), {1: "B", 3: "A"})
    assert table["contact_label"].tolist() == ["A-B"]
    assert table["label_a"].tolist() == ["B"]  # endpoint labels stay with the cell
    assert table["label_b"].tolist() == ["A"]


def test_generic_vocabulary_not_just_positive_negative():
    edges = _edges([
        (0, 0, 1, 2, "cell_cell", 1.0),  # B-C
        (0, 1, 2, 3, "cell_cell", 1.0),  # A-C
    ])
    table = label_contacts(_analysis(edges), {1: "B", 2: "C", 3: "A"})
    assert table["contact_label"].tolist() == ["B-C", "A-C"]


def test_unclassified_cell_gets_token_and_is_not_fully_classified():
    # cell 2 has no label.
    edges = _edges([(0, 0, 1, 2, "cell_cell", 5.0)])
    table = label_contacts(_analysis(edges), {1: "A"})
    assert table["label_b"].tolist() == ["unclassified"]
    assert table["contact_label"].tolist() == ["A-unclassified"]
    assert table["homotypic"].tolist() == [False]
    assert table["fully_classified"].tolist() == [False]


def test_two_unclassified_cells_are_homotypic_but_not_fully_classified():
    edges = _edges([(0, 0, 1, 2, "cell_cell", 5.0)])
    table = label_contacts(_analysis(edges), {})
    assert table["contact_label"].tolist() == ["unclassified-unclassified"]
    assert table["homotypic"].tolist() == [True]
    assert table["fully_classified"].tolist() == [False]


def test_custom_unclassified_token():
    edges = _edges([(0, 0, 1, 2, "cell_cell", 5.0)])
    table = label_contacts(_analysis(edges), {1: "A"}, unclassified="none")
    assert table["label_b"].tolist() == ["none"]
    assert table["contact_label"].tolist() == ["A-none"]


def test_border_edges_excluded():
    edges = _edges([
        (0, 0, 1, 2, "cell_cell", 5.0),
        (0, 1, 3, 0, "border", 9.0),  # not a cell-cell contact
    ])
    table = label_contacts(_analysis(edges), {1: "A", 2: "A", 3: "A"})
    assert table["frame"].size == 1
    assert table["cell_b"].tolist() == [2]


def test_fragments_stay_separate_with_same_label():
    # Two fragments of the same (frame, 1, 2) boundary, different edge_id/length.
    edges = _edges([
        (0, 0, 1, 2, "cell_cell", 3.0),
        (0, 1, 1, 2, "cell_cell", 4.0),
    ])
    table = label_contacts(_analysis(edges), {1: "A", 2: "B"})
    assert table["contact_label"].tolist() == ["A-B", "A-B"]
    assert table["edge_id"].tolist() == [0, 1]
    assert table["length"].tolist() == [3.0, 4.0]


def test_empty_edges_returns_empty_typed_table():
    table = label_contacts(_analysis(_edges([])), {1: "A"})
    expected = {
        "frame", "edge_id", "cell_a", "cell_b", "label_a", "label_b",
        "contact_label", "homotypic", "fully_classified", "length",
    }
    assert set(table) == expected
    assert all(v.size == 0 for v in table.values())
    assert table["contact_label"].dtype == object
    assert table["homotypic"].dtype == bool


def test_border_only_edges_returns_empty_typed_table():
    edges = _edges([(0, 0, 1, 0, "border", 9.0)])
    table = label_contacts(_analysis(edges), {1: "A"})
    assert table["frame"].size == 0
    assert table["fully_classified"].dtype == bool
