"""Neighbor-enrichment quantifier — sorting-vs-mixing of the contact graph.

A contacts-derived Build product. Requires the position's NLS subpopulation
labels and (matching the previous plot-time behaviour) only the two-type case:
when a position is unclassified or carries more than two labels, an empty — but
typed — table is persisted so it pools to nothing instead of erroring.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from cellflow.aggregate_quantification.contacts.neighborhood import neighbor_enrichment
from cellflow.aggregate_quantification.quantifier import PositionInputs, Quantifier
from cellflow.aggregate_quantification.quantifiers import _contacts_derived as derived


class NeighborEnrichmentQuantifier(Quantifier):
    """Per ``(frame, cell_id, neighbor_label)`` neighbor-type enrichment."""

    quantity_id = "neighbor_enrichment"
    display_name = "Neighbor enrichment"
    requires = ("contact_analysis_path",)
    default_output_name = "neighbor_enrichment.csv"
    # Long per (frame, cell_id, focal_label, neighbor_label) enrichment table.
    table_keys = ("frame", "cell_id", "focal_label", "neighbor_label")

    def build(
        self,
        inputs: PositionInputs,
        output_path: Path,
        *,
        params: dict | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        analysis = derived.load_analysis(inputs)
        labels = derived.load_labels(inputs)
        # Typed view: only the two-subpopulation case is well-defined; an empty
        # labels map yields a valid empty table from the compute function.
        if not labels or len(set(labels.values())) > 2:
            labels = {}
        return derived.persist(output_path, neighbor_enrichment(analysis, labels))

    def read(self, output_path: Path) -> dict[str, np.ndarray]:
        return derived.read_derived_table(output_path)

    def object_table(self, output_path: Path) -> Mapping[str, np.ndarray]:
        return derived.read_derived_table(output_path)
