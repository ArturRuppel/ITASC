"""Contact cell-type labels — propagate a per-cell label onto contacts.

Each cell–cell contact (a row of the ``edges`` table) is labelled by the
unordered pair of its two cells' labels, supplied as a generic ``cell_id -> label``
map. A contact between cell types ``A`` and ``B`` is labelled ``"A-B"`` (sorted), so
a consumer can ask how subpopulations contact each other — homotypic vs
heterotypic — without that aggregation living here. The label map is the caller's
concern (e.g. a downstream, dataset-specific classification); this module is
label-agnostic.

Headless and Qt-free, like
:mod:`cellflow.contact_analysis.contacts.signed_contact_length`: it operates on an
already-read
:class:`~cellflow.contact_analysis.contacts.reader.PositionContactAnalysis`
plus a ``cell_id -> label`` map, so it never opens HDF5 itself and runs unchanged in
scripts, notebooks, and plugins.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from cellflow.contact_analysis.contacts.reader import PositionContactAnalysis

#: Columns the label table carries, in order. Kept as a constant so the empty
#: table and the populated one cannot drift apart.
_COLUMNS = (
    "frame",
    "edge_id",
    "cell_a",
    "cell_b",
    "label_a",
    "label_b",
    "contact_label",
    "homotypic",
    "fully_classified",
    "length",
)


def contact_label_for(
    labels: Mapping[int, str],
    cell_a: int,
    cell_b: int,
    *,
    unclassified: str = "unclassified",
) -> str:
    """The sorted ``"label_a-label_b"`` contact label for one cell pair.

    Each cell's label is looked up in *labels* (``cell_id -> label``); a cell
    absent from the map takes *unclassified*. The pair is sorted so the contact
    type is orientation-independent (``"A-B" == "B-A"``). This is the single
    definition shared by :func:`label_contacts` (the per-edge table) and the
    signed-contact-length reaction coordinate, so a contact types identically
    wherever it is seen.
    """
    label_a = labels.get(int(cell_a), unclassified)
    label_b = labels.get(int(cell_b), unclassified)
    return "-".join(sorted((label_a, label_b)))


def label_contacts(
    analysis: PositionContactAnalysis,
    labels: Mapping[int, str],
    *,
    unclassified: str = "unclassified",
) -> dict[str, np.ndarray]:
    """Label every cell–cell contact by its two cells' NLS subpopulation labels.

    For each ``kind == "cell_cell"`` edge, each endpoint's label is looked up in
    *labels* (``cell_id -> label``; a cell absent from the map takes
    *unclassified*) and the unordered-pair contact label
    ``"-".join(sorted([label_a, label_b]))`` is formed. The vocabulary is whatever
    *labels* holds — nothing here is hard-wired to positive/negative.

    Border edges (``kind == "border"``, ``cell_b == 0``) are not contacts between
    two cells and are excluded. Fragments are **not** joined: a boundary split
    across several edge rows yields several labelled rows that share the same
    ``(frame, cell_a, cell_b)`` and therefore the same label; ``edge_id`` and
    ``length`` are carried through so a consumer can join or length-weight later.

    Columns (column-major, all equal length, one row per cell–cell edge):

    * ``frame``               — the edge's frame.
    * ``edge_id``             — the edge's id within its frame.
    * ``cell_a`` / ``cell_b`` — the contacting cell ids (as stored, ``a < b``).
    * ``label_a`` / ``label_b`` — each cell's NLS label, or *unclassified*.
    * ``contact_label``       — sorted ``"label_a-label_b"`` pair.
    * ``homotypic``           — ``label_a == label_b`` (True for two unclassified;
      gate on ``fully_classified`` if that matters).
    * ``fully_classified``    — both cells had a label in *labels*.
    * ``length``              — the edge length, carried through for weighting.

    Returns empty (but typed) arrays when there are no cell–cell edges.
    """
    edges = analysis.edges
    frame = np.asarray(edges.get("frame", ()), dtype=np.int64)
    edge_id = np.asarray(edges.get("edge_id", ()), dtype=np.int64)
    cell_a = np.asarray(edges.get("cell_a", ()), dtype=np.int64)
    cell_b = np.asarray(edges.get("cell_b", ()), dtype=np.int64)
    length = np.asarray(edges.get("length", ()), dtype=float)
    kind = np.asarray(edges.get("kind", np.full(frame.shape, "cell_cell")), dtype=object)

    out: dict[str, list] = {name: [] for name in _COLUMNS}

    for i in range(frame.size):
        if str(kind[i]) != "cell_cell":
            continue
        ca, cb = int(cell_a[i]), int(cell_b[i])
        present_a, present_b = ca in labels, cb in labels
        label_a = labels.get(ca, unclassified)
        label_b = labels.get(cb, unclassified)
        out["frame"].append(int(frame[i]))
        out["edge_id"].append(int(edge_id[i]) if edge_id.size else 0)
        out["cell_a"].append(ca)
        out["cell_b"].append(cb)
        out["label_a"].append(label_a)
        out["label_b"].append(label_b)
        out["contact_label"].append(contact_label_for(labels, ca, cb, unclassified=unclassified))
        out["homotypic"].append(label_a == label_b)
        out["fully_classified"].append(present_a and present_b)
        out["length"].append(float(length[i]))

    return {
        "frame": np.asarray(out["frame"], dtype=np.int64),
        "edge_id": np.asarray(out["edge_id"], dtype=np.int64),
        "cell_a": np.asarray(out["cell_a"], dtype=np.int64),
        "cell_b": np.asarray(out["cell_b"], dtype=np.int64),
        "label_a": np.asarray(out["label_a"], dtype=object),
        "label_b": np.asarray(out["label_b"], dtype=object),
        "contact_label": np.asarray(out["contact_label"], dtype=object),
        "homotypic": np.asarray(out["homotypic"], dtype=bool),
        "fully_classified": np.asarray(out["fully_classified"], dtype=bool),
        "length": np.asarray(out["length"], dtype=float),
    }
