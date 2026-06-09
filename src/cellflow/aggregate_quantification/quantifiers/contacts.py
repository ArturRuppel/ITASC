"""Cell-cell contacts quantifier — the registry adapter over the contacts code.

Wraps :mod:`cellflow.aggregate_quantification.contacts` so the studio can build
and read contacts through the generic :class:`Quantifier` interface. Its chosen
persistence is the ``contact_analysis.h5`` schema, unchanged from before the
rename.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from cellflow.aggregate_quantification.contacts.build import build_contact_analysis
from cellflow.aggregate_quantification.contacts.reader import (
    read_position_contact_analysis,
)
from cellflow.aggregate_quantification.quantifier import PositionInputs, Quantifier


class ContactsQuantifier(Quantifier):
    """Quantifies cell-cell contacts: edges, T1 events, NLS classes."""

    quantity_id = "contacts"
    display_name = "Cell–cell contacts"
    # Cell labels are mandatory; nucleus labels are optional (when present they
    # enable the cell_id == nucleus_id invariant check and NLS classification).
    requires = ("cell_labels_path",)

    #: Default artifact name when a position does not dictate one.
    default_output_name = "contact_analysis.h5"

    def default_output(self, inputs: PositionInputs) -> Path:
        return inputs.position_dir / self.default_output_name

    def build(
        self,
        inputs: PositionInputs,
        output_path: Path,
        *,
        params: dict | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        return build_contact_analysis(
            cell_labels_path=inputs.cell_labels_path,
            nucleus_labels_path=inputs.nucleus_labels_path,
            output_path=output_path,
            source_path=inputs.position_dir,
            edge_extraction_params=params,
            progress_cb=progress_cb,
        )

    def read(self, output_path: Path) -> Any:
        return read_position_contact_analysis(output_path)
