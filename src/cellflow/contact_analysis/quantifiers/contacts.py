"""Cell-cell contacts quantifier — the registry adapter over the contacts code.

Wraps :mod:`cellflow.contact_analysis.contacts` so the studio can build
and read contacts through the generic :class:`Quantifier` interface. Its chosen
persistence is the ``contact_analysis.h5`` schema, unchanged from before the
rename.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cellflow.contact_analysis.contacts.build import build_contacts
from cellflow.contact_analysis.contacts.reader import (
    read_position_contacts,
)
from cellflow.contact_analysis.quantifier import PositionInputs, Quantifier


class ContactsQuantifier(Quantifier):
    """Quantifies cell-cell contacts: edges, T1 events, NLS classes."""

    quantity_id = "contacts"
    display_name = "Cell–cell contacts"
    # Cell labels are mandatory; nucleus labels are optional (when present they
    # enable the cell_id == nucleus_id invariant check and NLS classification).
    requires = ("cell_labels_path",)
    #: The contacts artifact is the input the contacts-derived quantifiers
    #: (neighbor count / enrichment / z-score / density / signed contact length) consume.
    produces = "contact_analysis_path"

    #: Default artifact name when a position does not dictate one.
    default_output_name = "contact_analysis.h5"

    def build(
        self,
        inputs: PositionInputs,
        output_path: Path,
        *,
        params: dict | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        return build_contacts(
            cell_labels_path=inputs.cell_labels_path,
            nucleus_labels_path=inputs.nucleus_labels_path,
            output_path=output_path,
            source_path=inputs.position_dir,
            edge_extraction_params=params,
            progress_cb=progress_cb,
        )

    def default_output(self, inputs: PositionInputs) -> Path:
        """The contacts ``.h5`` lives in the position base folder, beside the
        committed ``cell_labels.tif`` / ``nucleus_labels.tif`` — not under the
        shared :data:`~cellflow.contact_analysis.quantifier.OUTPUT_SUBDIR` that the
        other (dynamics) quantifiers persist into. One homogeneous per-position
        layout for the downstream-stable inputs and this derived-from-them graph.
        """
        return inputs.position_dir / self.default_output_name

    def read(self, output_path: Path) -> Any:
        return read_position_contacts(output_path)

    def object_table(self, output_path: Path) -> Mapping[str, np.ndarray]:
        """The per-cell ``cells`` table (``frame`` · ``cell_id`` · morphometry).

        The subpopulation label is no longer carried here — it lives in the NLS
        sidecar CSV and is joined by ``cell_id`` at pool time."""
        return read_position_contacts(output_path).cells
