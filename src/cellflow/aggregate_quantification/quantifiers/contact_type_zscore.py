"""Contact-type z-score quantifier — observed contact types vs a label-shuffle null.

A contacts-derived Build product, and the one that most needed moving off the plot
path: it runs an ``n_shuffles`` permutation null per frame. The shuffle count comes
from the studio's shared params bar (``wants_build_params``); like enrichment it is
a typed (two-subpopulation) view, so an unclassified / >2-label position persists an
empty table.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from cellflow.aggregate_quantification.contacts.neighborhood import contact_type_zscores
from cellflow.aggregate_quantification.quantifier import PositionInputs, Quantifier
from cellflow.aggregate_quantification.quantifiers import _contacts_derived as derived

#: Permutation count used when the shared params bar supplies none.
_DEFAULT_SHUFFLES = 1000


class ContactTypeZScoreQuantifier(Quantifier):
    """Per ``(frame, contact_type)`` sorting/mixing z-score against a label null."""

    quantity_id = "contact_type_zscore"
    display_name = "Contact-type z-score"
    requires = ("contact_analysis_path",)
    default_output_name = "contact_type_zscore.csv"
    # Per (frame, contact_type) z-scores get their own per-contact-type table.
    shape_table = "contact_types_by_frame"
    table_keys = ("frame", "contact_type")
    wants_build_params = True

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
        if not labels or len(set(labels.values())) > 2:
            labels = {}
        shuffles = int((params or {}).get("shuffles") or _DEFAULT_SHUFFLES)
        table = contact_type_zscores(analysis, labels, n_shuffles=shuffles)
        return derived.persist(output_path, table)

    def read(self, output_path: Path) -> dict[str, np.ndarray]:
        return derived.read_derived_table(output_path)

    def object_table(self, output_path: Path) -> Mapping[str, np.ndarray]:
        return derived.read_derived_table(output_path)
