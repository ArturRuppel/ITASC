"""Neighborhood & density derivations over the cell–cell contact graph.

Four headless, Qt-free derivations on an already-read
:class:`~cellflow.aggregate_quantification.contacts.reader.PositionContactAnalysis`,
each returning a tidy column-major table (``dict[str, np.ndarray]``) shaped for
:class:`~cellflow.aggregate_quantification.plotting.PositionSource` /
:func:`~cellflow.aggregate_quantification.plotting.pool_object_tables`:

* :func:`cell_neighbor_counts` — adjacency degree per cell (how many neighbors).
* :func:`neighbor_enrichment` — per-cell observed/expected neighbor-type ratio
  (do cell types sort or mix, seen from each focal cell).
* :func:`contact_type_zscores` — observed contact-type counts vs a label-shuffle
  null (the rigorous "more than chance" statistic).
* :func:`cell_density` — cells per unit field-of-view area. (Counts straight off
  a ``frame -> [cell_id, …]`` map, so it needs only the cell labels, not the
  contact graph.)

The ``cell_cell`` edges of the contacts ``edges`` table *are* the adjacency
graph; degree is the count of incident ``cell_cell`` edges, deduped per neighbor
(a boundary split across several edge rows still counts as one neighbor). Cell
types come from the NLS ``class_label`` sidecar CSV
(:func:`...nls_classification.read_nls_classification_csv`), passed in as a
``labels: cell_id -> label`` map; a cell absent from it is **unclassified** —
still counted as a neighbor and a cell, but excluded from the enrichment /
z-score maths (its edges drop, abundances count labeled cells only).

Like :mod:`...contacts.signed_contact_length` and :mod:`...contacts.contact_labels`, these
operate on the in-memory analysis object only: they never open HDF5 or the CSV,
so they run unchanged in scripts, notebooks, and the napari plugin.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from cellflow.aggregate_quantification.contacts.reader import PositionContactAnalysis

#: Bucket for a cell with no entry in the NLS ``labels`` map (kept consistent
#: with the plotting backend's ``UNCLASSIFIED``).
UNCLASSIFIED = "unclassified"


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
    isolated cell (degree 0) and a labeled-but-edgeless cell still count toward
    degree, abundance, and density.
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
    isolated cell). No labels are needed here — ``class_label`` is attached later
    by the standard per-position join in
    :func:`...plotting.pool_object_tables`. This count is also the numerator for
    density.

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


def neighbor_enrichment(
    analysis: PositionContactAnalysis,
    labels: Mapping[int, str],
) -> dict[str, np.ndarray]:
    """Long per ``(frame, cell_id, neighbor_label)`` neighbor-enrichment table.

    For a focal cell of type ``i`` with labeled degree ``d`` (its number of
    *labeled* neighbors), in a frame with ``N`` labeled cells and ``N_j`` cells of
    type ``j``::

        expected_j = d * f_j'      where
            f_j' = N_j / (N - 1)            for j != i
            f_i' = (N_i - 1) / (N - 1)      for j == i   (self-excluded)
        observed_j = # of the focal cell's neighbors that are type j
        enrichment = observed_j / expected_j     (NaN when expected_j == 0)

    Because the abundances ``f_j'`` sum to 1, ``sum_j expected_j == d``, matching
    ``sum_j observed_j``. ``enrichment > 1`` ⇒ that neighbor type is
    over-represented around this cell (sorting); ``< 1`` ⇒ avoided (mixing).

    Emitted per **labeled** focal cell × per label ``j`` present (among labeled
    cells) in the frame, so a two-type frame yields up to four
    ``(focal, neighbor)`` combinations (``aa``, ``ab``, ``ba``, ``bb``). ``ab`` and
    ``ba`` cover the same physical edges from each endpoint but keep distinct
    per-cell denominators. Edges to ``unclassified`` neighbors are dropped from
    ``observed``; abundances count labeled cells only; an ``unclassified`` focal
    cell emits no rows.

    Columns: ``frame``, ``cell_id``, ``focal_label``, ``neighbor_label``,
    ``observed`` (int), ``expected`` (float), ``enrichment`` (float).
    """
    adjacency = _frame_adjacency(analysis)
    frame_cells = _frame_cells(analysis)

    out_frame: list[int] = []
    out_cell: list[int] = []
    out_focal: list[str] = []
    out_neighbor: list[str] = []
    out_observed: list[int] = []
    out_expected: list[float] = []
    out_enrichment: list[float] = []

    for fr in sorted(frame_cells):
        labeled = [c for c in frame_cells[fr] if c in labels]
        n_total = len(labeled)
        counts: dict[str, int] = defaultdict(int)
        for c in labeled:
            counts[labels[c]] += 1
        labels_present = sorted(counts)
        denom = n_total - 1
        neighbors_of = adjacency.get(fr, {})

        for focal in labeled:
            focal_label = labels[focal]
            labeled_neighbors = [n for n in neighbors_of.get(focal, ()) if n in labels]
            degree = len(labeled_neighbors)
            observed: dict[str, int] = defaultdict(int)
            for n in labeled_neighbors:
                observed[labels[n]] += 1
            for j in labels_present:
                n_j = counts[j]
                if denom <= 0:
                    f_j = float("nan")
                elif j == focal_label:
                    f_j = (n_j - 1) / denom
                else:
                    f_j = n_j / denom
                expected = degree * f_j
                obs_j = observed.get(j, 0)
                enrichment = obs_j / expected if expected > 0 else float("nan")
                out_frame.append(fr)
                out_cell.append(focal)
                out_focal.append(focal_label)
                out_neighbor.append(j)
                out_observed.append(obs_j)
                out_expected.append(float(expected))
                out_enrichment.append(float(enrichment))

    return {
        "frame": np.asarray(out_frame, dtype=np.int64),
        "cell_id": np.asarray(out_cell, dtype=np.int64),
        "focal_label": np.asarray(out_focal, dtype=object),
        "neighbor_label": np.asarray(out_neighbor, dtype=object),
        "observed": np.asarray(out_observed, dtype=np.int64),
        "expected": np.asarray(out_expected, dtype=float),
        "enrichment": np.asarray(out_enrichment, dtype=float),
    }


def contact_type_zscores(
    analysis: PositionContactAnalysis,
    labels: Mapping[int, str],
    *,
    n_shuffles: int = 1000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Per ``(frame, contact_type)`` observed counts vs a label-shuffle null.

    Each ``cell_cell`` edge between two labeled cells is typed by its sorted label
    pair (``aa | ab | bb``). For each frame: count each type, then permute the
    labels across that frame's labeled cells ``n_shuffles`` times and recount,
    giving a null mean/sd per type. ``z = (observed - mean_null) / sd_null``; a
    large positive homotypic ``z`` is sorting, a large negative one is mixing.

    The shuffle is vectorized: the frame's edge endpoint indices are fixed, so a
    shuffle only relabels nodes. The whole ``(n_shuffles, n_nodes)`` permuted
    label matrix is built once and both endpoints indexed into it, tallying every
    shuffle's type counts at once — cheap per frame.

    ``z_score = NaN`` when ``sd_null == 0`` (a degenerate frame: one label only,
    or too few cells). Edges touching ``unclassified`` cells are excluded
    throughout, and abundances count labeled cells only.

    Columns: ``frame``, ``contact_type`` (the sorted ``"<label>·<label>"`` pair),
    ``observed_count`` (int), ``mean_null`` (float), ``sd_null`` (float),
    ``z_score`` (float), ``observed_fraction`` (float), ``expected_fraction``
    (float, the analytic ``f_i·f_j`` homotypic / ``2·f_i·f_j`` heterotypic
    carried for reference alongside the empirical null).
    """
    adjacency = _frame_adjacency(analysis)
    frame_pairs = _frame_unique_pairs(adjacency)
    frame_cells = _frame_cells(analysis)
    rng = np.random.default_rng(seed)

    out_frame: list[int] = []
    out_type: list[str] = []
    out_observed: list[int] = []
    out_mean: list[float] = []
    out_sd: list[float] = []
    out_z: list[float] = []
    out_obs_frac: list[float] = []
    out_exp_frac: list[float] = []

    for fr in sorted(frame_cells):
        nodes = [c for c in frame_cells[fr] if c in labels]
        if not nodes:
            continue
        local_labels = sorted({labels[c] for c in nodes})
        lab_id = {lab: i for i, lab in enumerate(local_labels)}
        n_labels = len(local_labels)
        node_label_id = np.asarray([lab_id[labels[c]] for c in nodes], dtype=np.int64)
        node_index = {c: i for i, c in enumerate(nodes)}

        ea: list[int] = []
        eb: list[int] = []
        for ca, cb in frame_pairs.get(fr, ()):
            ia, ib = node_index.get(ca), node_index.get(cb)
            if ia is not None and ib is not None:
                ea.append(ia)
                eb.append(ib)
        ea_arr = np.asarray(ea, dtype=np.int64)
        eb_arr = np.asarray(eb, dtype=np.int64)
        n_edges = ea_arr.size

        # (label_i, label_j) sorted pair -> type id, and the reverse for naming.
        type_pairs = [(i, j) for i in range(n_labels) for j in range(i, n_labels)]
        n_types = len(type_pairs)
        pair_lookup = np.zeros((n_labels, n_labels), dtype=np.int64)
        for t, (i, j) in enumerate(type_pairs):
            pair_lookup[i, j] = t

        def _tally(label_vec: np.ndarray) -> np.ndarray:
            counts = np.zeros(n_types, dtype=np.int64)
            if n_edges:
                la, lb = label_vec[ea_arr], label_vec[eb_arr]
                lo = np.minimum(la, lb)
                hi = np.maximum(la, lb)
                np.add.at(counts, pair_lookup[lo, hi], 1)
            return counts

        observed = _tally(node_label_id)

        if n_edges:
            tiled = np.broadcast_to(node_label_id, (n_shuffles, node_label_id.size))
            shuffled = rng.permuted(tiled, axis=1)
            la = shuffled[:, ea_arr]
            lb = shuffled[:, eb_arr]
            lo = np.minimum(la, lb)
            hi = np.maximum(la, lb)
            tid = pair_lookup[lo, hi]  # (n_shuffles, n_edges)
            null_counts = np.empty((n_shuffles, n_types), dtype=float)
            for t in range(n_types):
                null_counts[:, t] = (tid == t).sum(axis=1)
        else:
            null_counts = np.zeros((n_shuffles, n_types), dtype=float)

        mean_null = null_counts.mean(axis=0)
        sd_null = null_counts.std(axis=0)
        z = np.where(sd_null > 0, (observed - mean_null) / np.where(sd_null > 0, sd_null, 1.0), np.nan)

        n_total = len(nodes)
        type_count = np.bincount(node_label_id, minlength=n_labels).astype(float)
        f = type_count / n_total if n_total else np.zeros(n_labels)

        for t, (i, j) in enumerate(type_pairs):
            exp_frac = f[i] * f[j] if i == j else 2.0 * f[i] * f[j]
            obs_frac = observed[t] / n_edges if n_edges else float("nan")
            out_frame.append(fr)
            out_type.append(f"{local_labels[i]}·{local_labels[j]}")
            out_observed.append(int(observed[t]))
            out_mean.append(float(mean_null[t]))
            out_sd.append(float(sd_null[t]))
            out_z.append(float(z[t]))
            out_obs_frac.append(float(obs_frac))
            out_exp_frac.append(float(exp_frac))

    return {
        "frame": np.asarray(out_frame, dtype=np.int64),
        "contact_type": np.asarray(out_type, dtype=object),
        "observed_count": np.asarray(out_observed, dtype=np.int64),
        "mean_null": np.asarray(out_mean, dtype=float),
        "sd_null": np.asarray(out_sd, dtype=float),
        "z_score": np.asarray(out_z, dtype=float),
        "observed_fraction": np.asarray(out_obs_frac, dtype=float),
        "expected_fraction": np.asarray(out_exp_frac, dtype=float),
    }


def cell_density(
    frame_cells: Mapping[int, Sequence[int]],
    labels: Mapping[int, str],
    *,
    fov_area_mm2: float,
) -> dict[str, np.ndarray]:
    """Per ``(frame, label)`` cell counts and ``density = n_cells / fov_area_mm2``.

    Emits one row per label present (among labeled cells) plus a ``label="all"``
    total row that counts **every** cell in the frame, including
    ``unclassified``. ``density`` is in cells/mm²; *fov_area_mm2* is the user's
    field-of-view area and is **required** (a positive number) — there is no
    silent image-area fallback. An empty *labels* map yields only the ``all`` row
    per frame.

    *frame_cells* maps ``frame -> [cell_id, …]`` (the cell labels present in that
    frame), so this counts straight off the cell labels with no contacts
    dependency.

    Columns: ``frame``, ``label`` (str), ``n_cells`` (int), ``density``
    (float, cells/mm²).
    """
    area = float(fov_area_mm2)
    if not area > 0:
        raise ValueError("cell_density requires a positive fov_area_mm2")

    out_frame: list[int] = []
    out_label: list[str] = []
    out_n: list[int] = []
    out_density: list[float] = []

    def _emit(fr: int, label: str, n_cells: int) -> None:
        out_frame.append(fr)
        out_label.append(label)
        out_n.append(n_cells)
        out_density.append(n_cells / area)

    for fr in sorted(frame_cells):
        cells = frame_cells[fr]
        _emit(fr, "all", len(cells))
        counts: dict[str, int] = defaultdict(int)
        for c in cells:
            if c in labels:
                counts[labels[c]] += 1
        for label in sorted(counts):
            _emit(fr, label, counts[label])

    return {
        "frame": np.asarray(out_frame, dtype=np.int64),
        "label": np.asarray(out_label, dtype=object),
        "n_cells": np.asarray(out_n, dtype=np.int64),
        "density": np.asarray(out_density, dtype=float),
    }


def _frame_unique_pairs(
    adjacency: dict[int, dict[int, set[int]]],
) -> dict[int, list[tuple[int, int]]]:
    """``frame -> [(a, b), …]`` of unique ``a < b`` adjacency relations.

    Derived from :func:`_frame_adjacency`, so a boundary fragmented into several
    edge rows is one contact pair here too — the z-score counts adjacency
    relations, consistent with degree, not raw (possibly fragmented) edge rows.
    """
    out: dict[int, list[tuple[int, int]]] = {}
    for fr, neighbors in adjacency.items():
        pairs = {(a, b) if a < b else (b, a) for a, ns in neighbors.items() for b in ns}
        out[fr] = sorted(pairs)
    return out
