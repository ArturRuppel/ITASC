"""Signed contact length — the signed central junction-length reaction coordinate.

Boltzmann-inversion of this coordinate yields the effective potential / barrier
``ΔE_eff`` of T1 transitions: the energy a junction climbs to reach the
four-fold vertex. That inversion and the resulting potential-landscape plot are
downstream concerns owned by the data repo, not this package — this
module only computes the signed lengths themselves. It is the ITASC
analogue of the reference's ``extract_central_junction_lengths`` /
``plot_signed_lengths_neg_log_p_histogram`` (``morphogenesis-on-chip_analysis``),
but the sign comes from the ``t1_events`` table (``losing`` ↔ ``gaining`` pairs)
rather than curated "quad" JSONs.

Headless and Qt-free: it operates on an already-read
:class:`~itasc.contact_analysis.contacts.reader.PositionContactAnalysis`,
so it never opens HDF5 itself and runs unchanged in scripts, notebooks, and the
napari plugin.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import numpy as np

from itasc.contact_analysis.contacts.contact_labels import contact_label_for
from itasc.contact_analysis.contacts.reader import PositionContactAnalysis


def signed_central_junction_lengths(
    analysis: PositionContactAnalysis,
    *,
    pixel_size_um: float | None = None,
    labels: Mapping[int, str] | None = None,
) -> dict[str, np.ndarray]:
    """Signed central junction length per T1 event, per frame.

    For each T1 event the central junction is the edge that flips from the
    *losing* cell pair (pre-transition) to the *gaining* pair (post-transition).
    Every frame in which an event's losing edge exists contributes a **negative**
    sample (``−length``); every frame its gaining edge exists contributes a
    **positive** one (``+length``). The magnitude is the edge length; it crosses
    zero at the four-fold vertex, so pooled and inverted these reproduce the
    reference's double-well potential without curated quads.

    Fragmented contacts are joined first: the build splits a single cell-cell
    boundary into several edge rows (one per disconnected segment from
    ``_coordinate_segments``), so the lengths of all rows sharing a
    ``(frame, cell-pair)`` are **summed** into one total junction length before
    signing. Otherwise each fragment would enter the landscape as its own
    (short) sample. This is the headless analogue of the v1
    ``find_shared_boundary`` / ``order_boundary_pixels`` join that produced one
    length per junction.

    All frames an edge exists are used (no ± window) — matching the reference,
    which histograms the whole movie.

    Columns (column-major, all equal length):

    * ``t1_event_id`` — the event each sample belongs to.
    * ``frame`` — the frame the sample is read from.
    * ``signed_length`` — ``±length``, in µm when *pixel_size_um* is given else px.
    * ``role`` — ``"losing"`` (negative) or ``"gaining"`` (positive).
    * ``contact_type`` — the event's **transition pair** ``"<losing>→<gaining>"``
      (e.g. ``"A-A→A-B"``), where each side is the contact label
      (:func:`...contact_labels.contact_label_for`) of that junction's cell pair.
      ``""`` when no *labels* map is given (the label-agnostic default). A single
      per-event label is used (not per-side), so both the negative losing lobe and
      the positive gaining lobe of an event share it and a grouped curve still
      spans ``L = 0`` for the barrier.

    *labels* maps ``cell_id -> label`` (an optional, caller-supplied per-cell
    classification); a cell absent from it is ``"unclassified"``.

    Returns empty (but typed) arrays when there are no events or no matching
    edges; an event whose losing/gaining edges never appear in ``edges`` simply
    contributes nothing.
    """
    edges = analysis.edges
    events = analysis.t1_events
    scale = float(pixel_size_um) if pixel_size_um else 1.0

    e_frame = np.asarray(edges.get("frame", ()), dtype=np.int64)
    e_a = np.asarray(edges.get("cell_a", ()), dtype=np.int64)
    e_b = np.asarray(edges.get("cell_b", ()), dtype=np.int64)
    e_len = np.asarray(edges.get("length", ()), dtype=float)

    # Join fragments: sum every edge row sharing a (pair, frame) into one total
    # junction length, so a boundary split across segments enters the landscape
    # once at its real length rather than as several short samples.
    total_length: dict[tuple[frozenset[int], int], float] = defaultdict(float)
    for fr, ca, cb, ln in zip(e_frame, e_a, e_b, e_len):
        total_length[(frozenset((int(ca), int(cb))), int(fr))] += float(ln)

    # Unordered cell-pair -> [(frame, total_length), …], frame-sorted, so each
    # event reads its losing and gaining edge in one lookup over all the frames
    # they appear in.
    pair_frames: dict[frozenset[int], list[tuple[int, float]]] = defaultdict(list)
    for (pair, fr), total in total_length.items():
        pair_frames[pair].append((fr, total))
    for entries in pair_frames.values():
        entries.sort()

    ev_id = np.asarray(events.get("t1_event_id", ()), dtype=np.int64)
    l_a = np.asarray(events.get("losing_cell_a", ()), dtype=np.int64)
    l_b = np.asarray(events.get("losing_cell_b", ()), dtype=np.int64)
    g_a = np.asarray(events.get("gaining_cell_a", ()), dtype=np.int64)
    g_b = np.asarray(events.get("gaining_cell_b", ()), dtype=np.int64)

    out_event: list[int] = []
    out_frame: list[int] = []
    out_signed: list[float] = []
    out_role: list[str] = []
    out_type: list[str] = []

    for i in range(ev_id.size):
        eid = int(ev_id[i])
        # One transition label per event ("<losing>→<gaining>"), shared by both
        # lobes so a grouped curve keeps its losing (−) and gaining (+) sides
        # together and the barrier at L=0 stays defined.
        if labels:
            losing_type = contact_label_for(labels, int(l_a[i]), int(l_b[i]))
            gaining_type = contact_label_for(labels, int(g_a[i]), int(g_b[i]))
            contact_type = f"{losing_type}→{gaining_type}"
        else:
            contact_type = ""
        for sign, role, pair in (
            (-1.0, "losing", frozenset((int(l_a[i]), int(l_b[i])))),
            (+1.0, "gaining", frozenset((int(g_a[i]), int(g_b[i])))),
        ):
            for frame, length in pair_frames.get(pair, ()):
                out_event.append(eid)
                out_frame.append(frame)
                out_signed.append(sign * length * scale)
                out_role.append(role)
                out_type.append(contact_type)

    return {
        "t1_event_id": np.asarray(out_event, dtype=np.int64),
        "frame": np.asarray(out_frame, dtype=np.int64),
        "signed_length": np.asarray(out_signed, dtype=float),
        "role": np.asarray(out_role, dtype=object),
        "contact_type": np.asarray(out_type, dtype=object),
    }
