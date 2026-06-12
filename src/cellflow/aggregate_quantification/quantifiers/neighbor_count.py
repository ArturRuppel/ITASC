"""Neighbor-count quantifier — per-cell adjacency degree from the contact graph.

A contacts-derived Build product: reads the position's ``contact_analysis.h5`` and
persists the per ``(frame, cell_id)`` neighbor count as a tidy CSV. The NLS
subpopulation label is *not* stored here — it is left-joined by ``cell_id`` at
pool time (the standard ``join_class`` path), exactly as the shape family does.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from cellflow.aggregate_quantification.contacts.neighborhood import cell_neighbor_counts
from cellflow.aggregate_quantification.quantifier import PositionInputs, Quantifier
from cellflow.aggregate_quantification.quantifiers import _contacts_derived as derived


class NeighborCountQuantifier(Quantifier):
    """Per ``(frame, cell_id)`` number of distinct cell-cell neighbors."""

    quantity_id = "neighbor_count"
    display_name = "Neighbor count"
    requires = ("contact_analysis_path",)
    default_output_name = "neighbor_count.csv"

    def build(
        self,
        inputs: PositionInputs,
        output_path: Path,
        *,
        params: dict | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        analysis = derived.load_analysis(inputs)
        return derived.persist(output_path, cell_neighbor_counts(analysis))

    def read(self, output_path: Path) -> dict[str, np.ndarray]:
        return derived.read_derived_table(output_path)

    def object_table(self, output_path: Path) -> Mapping[str, np.ndarray]:
        return derived.read_derived_table(output_path)
