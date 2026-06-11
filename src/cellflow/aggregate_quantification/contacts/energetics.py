"""Contacts energetics — the signed central junction-length reaction coordinate.

Boltzmann-inverted (see
:func:`cellflow.aggregate_quantification.plotting.potential_landscape`) this
coordinate yields the effective potential / barrier ``ΔE_eff`` of T1 transitions:
the energy a junction climbs to reach the four-fold vertex. It is the CellFlow
analogue of the reference's ``extract_central_junction_lengths`` /
``plot_signed_lengths_neg_log_p_histogram`` (``morphogenesis-on-chip_analysis``),
but the sign comes from the ``t1_events`` table (``losing`` ↔ ``gaining`` pairs)
rather than curated "quad" JSONs.

Headless and Qt-free: it operates on an already-read
:class:`~cellflow.aggregate_quantification.contacts.reader.PositionContactAnalysis`,
so it never opens HDF5 itself and runs unchanged in scripts, notebooks, and the
napari plugin.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from cellflow.aggregate_quantification.contacts.reader import PositionContactAnalysis


def signed_central_junction_lengths(
    analysis: PositionContactAnalysis, *, pixel_size_um: float | None = None
) -> dict[str, np.ndarray]:
    """Signed central junction length per T1 event, per frame.

    For each T1 event the central junction is the edge that flips from the
    *losing* cell pair (pre-transition) to the *gaining* pair (post-transition).
    Every frame in which an event's losing edge exists contributes a **negative**
    sample (``−length``); every frame its gaining edge exists contributes a
    **positive** one (``+length``). The magnitude is the edge length; it crosses
    zero at the four-fold vertex, so pooled and inverted these reproduce the
    reference's double-well potential without curated quads.

    All frames an edge exists are used (no ± window) — matching the reference,
    which histograms the whole movie.

    Columns (column-major, all equal length):

    * ``t1_event_id`` — the event each sample belongs to.
    * ``frame`` — the frame the sample is read from.
    * ``signed_length`` — ``±length``, in µm when *pixel_size_um* is given else px.
    * ``role`` — ``"losing"`` (negative) or ``"gaining"`` (positive).

    Returns empty (but typed) arrays when there are no events or no matching
    edges; an event whose losing/gaining edges never appear in ``edges`` simply
    contributes nothing.
    """
    edges = analysis.edges
    events = analysis.t1_events
    scale = float(pixel_size_um) if pixel_size_um else 1.0

    # Unordered cell-pair -> [(frame, length), …], so each event reads its losing
    # and gaining edge in one lookup over all the frames they appear in.
    e_frame = np.asarray(edges.get("frame", ()), dtype=np.int64)
    e_a = np.asarray(edges.get("cell_a", ()), dtype=np.int64)
    e_b = np.asarray(edges.get("cell_b", ()), dtype=np.int64)
    e_len = np.asarray(edges.get("length", ()), dtype=float)
    pair_frames: dict[frozenset[int], list[tuple[int, float]]] = defaultdict(list)
    for fr, ca, cb, ln in zip(e_frame, e_a, e_b, e_len):
        pair_frames[frozenset((int(ca), int(cb)))].append((int(fr), float(ln)))

    ev_id = np.asarray(events.get("t1_event_id", ()), dtype=np.int64)
    l_a = np.asarray(events.get("losing_cell_a", ()), dtype=np.int64)
    l_b = np.asarray(events.get("losing_cell_b", ()), dtype=np.int64)
    g_a = np.asarray(events.get("gaining_cell_a", ()), dtype=np.int64)
    g_b = np.asarray(events.get("gaining_cell_b", ()), dtype=np.int64)

    out_event: list[int] = []
    out_frame: list[int] = []
    out_signed: list[float] = []
    out_role: list[str] = []

    for i in range(ev_id.size):
        eid = int(ev_id[i])
        for sign, role, pair in (
            (-1.0, "losing", frozenset((int(l_a[i]), int(l_b[i])))),
            (+1.0, "gaining", frozenset((int(g_a[i]), int(g_b[i])))),
        ):
            for frame, length in pair_frames.get(pair, ()):
                out_event.append(eid)
                out_frame.append(frame)
                out_signed.append(sign * length * scale)
                out_role.append(role)

    return {
        "t1_event_id": np.asarray(out_event, dtype=np.int64),
        "frame": np.asarray(out_frame, dtype=np.int64),
        "signed_length": np.asarray(out_signed, dtype=float),
        "role": np.asarray(out_role, dtype=object),
    }
