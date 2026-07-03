"""Relational shape quantifier — nucleus-vs-cell paired morphology.

Wraps :func:`cellflow.contact_analysis.shape.build_relational`, which
pairs each nucleus with its cell on the shared ``(frame, id)`` key and emits
relational quantities (nuclear:cell area ratio, centroid offset, orientation
delta, …). Needs **both** label stacks plus a pixel size; persists
``aggregate_quantification/shape_relational.csv``. The object-key column is ``cell_id``,
so the relational table pools and plots through the same path as the per-source
shape tables.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from cellflow.contact_analysis.quantifier import PositionInputs, Quantifier
from cellflow.contact_analysis.shape import (
    build_relational,
    read_relational_table,
)


class ShapeRelationalQuantifier(Quantifier):
    """Quantifies per-(frame, cell) relational nucleus-vs-cell shape."""

    quantity_id = "shape_relational"
    display_name = "Nucleus–cell shape"
    # Both label stacks (to pair); pixel size (to emit physical units) is a
    # global build param set in the Parameters panel.
    requires = ("cell_labels_path", "nucleus_labels_path")
    required_build_params = {"pixel_size_um": "pixel size (µm/px)"}

    default_output_name = "shape_relational.csv"
    # Per-cell nucleus↔cell relational descriptors, keyed (frame, cell_id).
    table_keys = ("frame", "cell_id")

    def build(
        self,
        inputs: PositionInputs,
        output_path: Path,
        *,
        params: dict | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        return build_relational(
            inputs.cell_labels_path,
            inputs.nucleus_labels_path,
            output_path,
            pixel_size_um=inputs.pixel_size_um,
            source_path=inputs.position_dir,
            params=params,
            quantity_id=self.quantity_id,
            progress_cb=progress_cb,
        )

    def read(self, output_path: Path) -> dict[str, np.ndarray]:
        return read_relational_table(output_path)

    def object_table(self, output_path: Path) -> Mapping[str, np.ndarray]:
        return read_relational_table(output_path)
