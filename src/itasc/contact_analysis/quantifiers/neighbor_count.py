"""Neighbor-count quantifier — per-cell adjacency degree from the contact graph.

A contacts-derived **pooled** quantity: :meth:`compute_object_table` reads the
position's ``contact_analysis.h5`` and returns the per ``(frame, cell_id)``
neighbor count in memory (no per-position artifact). The NLS subpopulation label
is *not* stored here — it is left-joined by ``cell_id`` at pool time, exactly as
the shape family does.
"""
from __future__ import annotations

from itasc.contact_analysis.contacts.neighborhood import cell_neighbor_counts
from itasc.contact_analysis.quantifier import Quantifier
from itasc.contact_analysis.quantifiers import _contacts_derived as derived


class NeighborCountQuantifier(Quantifier):
    """Per ``(frame, cell_id)`` number of distinct cell-cell neighbors."""

    quantity_id = "neighbor_count"
    display_name = "Neighbor count"
    requires = ("contact_analysis_path",)
    # Per-cell adjacency degree, keyed (frame, cell_id).
    table_keys = ("frame", "cell_id")

    def compute_object_table(self, inputs, *, params=None):
        return dict(cell_neighbor_counts(derived.load_analysis(inputs)))
