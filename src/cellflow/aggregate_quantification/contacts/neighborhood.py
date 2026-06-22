"""Neighborhood & density derivations over the cell–cell contact graph.

Two headless, Qt-free, **label-agnostic** derivations on an already-read
:class:`~cellflow.aggregate_quantification.contacts.reader.PositionContactAnalysis`,
each returning a tidy column-major table (``dict[str, np.ndarray]``):

* :func:`cell_neighbor_counts` — adjacency degree per cell (how many neighbors).
* :func:`cell_density` — cells per unit field-of-view area. (Counts straight off
  a ``frame -> [cell_id, …]`` map, so it needs only the cell labels, not the
  contact graph.)

The ``cell_cell`` edges of the contacts ``edges`` table *are* the adjacency
graph; degree is the count of incident ``cell_cell`` edges, deduped per neighbor
(a boundary split across several edge rows still counts as one neighbor).

The classification-dependent derivations (neighbour enrichment; contact-type
z-score vs a label-shuffle null) are **not** here — they consume a per-cell
subpopulation ``class_label`` and so live with the dataset that defines it, not in
this label-agnostic library.

Like :mod:`...contacts.signed_contact_length` and :mod:`...contacts.contact_labels`, these
operate on the in-memory analysis object only: they never open HDF5, so they run
unchanged in scripts, notebooks, and the napari plugin.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from cellflow.aggregate_quantification.contacts.reader import PositionContactAnalysis


def _frame_adjacency(analysis: PositionContactAnalysis) -> dict[int, dict[int, set[int]]]:
    """``frame -> {cell_id -> set(neighbor cell_ids)}`` from ``cell_cell`` edges.

    Only ``kind == "cell_cell"`` rows contribute; border edges (``kind ==
    "border"``) are excluded. Each edge ``(cell_a, cell_b)`` adds each endpoint to
    the other's neighbor *set*, so a boundary fragmented into several edge rows
    still counts as a single neighbor relation (the set dedupes). Degree is
    ``len(neighbors[cell_id])``.
    """
    edges = analysis.edges
    frame = np.asarray(edges.get("frame", ()), dtype=np.int64)
    cell_a = np.asarray(edges.get("cell_a", ()), dtype=np.int64)
    cell_b = np.asarray(edges.get("cell_b", ()), dtype=np.int64)
    kind = np.asarray(edges.get("kind", np.full(frame.shape, "cell_cell")), dtype=object)

    adjacency: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for i in range(frame.size):
        if str(kind[i]) != "cell_cell":
            continue
        fr, ca, cb = int(frame[i]), int(cell_a[i]), int(cell_b[i])
        adjacency[fr][ca].add(cb)
        adjacency[fr][cb].add(ca)
    return adjacency


def _frame_cells(analysis: PositionContactAnalysis) -> dict[int, list[int]]:
    """``frame -> [cell_id, …]`` (ascending) from the ``cells`` table.

    The cells table is the authority on which cells are present in a frame, so an
    isolated cell (degree 0) still counts toward degree, abundance, and density.
    """
    cells = analysis.cells
    frame = np.asarray(cells.get("frame", ()), dtype=np.int64)
    cell_id = np.asarray(cells.get("cell_id", ()), dtype=np.int64)
    out: dict[int, list[int]] = defaultdict(list)
    for fr, cid in zip(frame, cell_id):
        out[int(fr)].append(int(cid))
    for ids in out.values():
        ids.sort()
    return out


def cell_neighbor_counts(analysis: PositionContactAnalysis) -> dict[str, np.ndarray]:
    """Per ``(frame, cell_id)`` adjacency degree — the headline neighbor count.

    One row per cell present in the ``cells`` table for that frame, with
    ``n_neighbors`` its number of distinct ``cell_cell`` neighbors (0 for an
    isolated cell). This count is also the numerator for density.

    Columns (column-major, equal length): ``frame``, ``cell_id``,
    ``n_neighbors`` (int).
    """
    adjacency = _frame_adjacency(analysis)
    frame_cells = _frame_cells(analysis)

    out_frame: list[int] = []
    out_cell: list[int] = []
    out_n: list[int] = []
    for fr in sorted(frame_cells):
        neighbors = adjacency.get(fr, {})
        for cid in frame_cells[fr]:
            out_frame.append(fr)
            out_cell.append(cid)
            out_n.append(len(neighbors.get(cid, ())))
    return {
        "frame": np.asarray(out_frame, dtype=np.int64),
        "cell_id": np.asarray(out_cell, dtype=np.int64),
        "n_neighbors": np.asarray(out_n, dtype=np.int64),
    }


def cell_density(
    frame_cells: Mapping[int, Sequence[int]],
    *,
    fov_area_mm2: float,
) -> dict[str, np.ndarray]:
    """Per ``frame`` cell count and ``density = n_cells / fov_area_mm2``.

    Emits one ``label="all"`` row per frame that counts **every** cell in the
    frame. ``density`` is in cells/mm²; *fov_area_mm2* is the user's field-of-view
    area and is **required** (a positive number) — there is no silent image-area
    fallback.

    *frame_cells* maps ``frame -> [cell_id, …]`` (the cell labels present in that
    frame), so this counts straight off the cell labels with no contacts
    dependency.

    Columns: ``frame``, ``label`` (str, always ``"all"``), ``n_cells`` (int),
    ``density`` (float, cells/mm²).
    """
    area = float(fov_area_mm2)
    if not area > 0:
        raise ValueError("cell_density requires a positive fov_area_mm2")

    out_frame: list[int] = []
    out_label: list[str] = []
    out_n: list[int] = []
    out_density: list[float] = []

    for fr in sorted(frame_cells):
        n_cells = len(frame_cells[fr])
        out_frame.append(fr)
        out_label.append("all")
        out_n.append(n_cells)
        out_density.append(n_cells / area)

    return {
        "frame": np.asarray(out_frame, dtype=np.int64),
        "label": np.asarray(out_label, dtype=object),
        "n_cells": np.asarray(out_n, dtype=np.int64),
        "density": np.asarray(out_density, dtype=float),
    }
