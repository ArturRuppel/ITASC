"""Cell-shape quantifier — the registry adapter over the shape core.

Exposes per-cell morphology through the generic :class:`Quantifier` interface as
a **pooled** quantity: :meth:`compute_object_table` runs the shape core over the
cell label stack in memory and hands the tidy table straight to the pooling layer
(no per-position artifact is persisted). The object-key column is ``cell_id`` (the
shared track id).
"""
from __future__ import annotations

from cellflow.contact_analysis.quantifier import Quantifier
from cellflow.contact_analysis.shape import compute_object_shape


class CellShapeQuantifier(Quantifier):
    """Quantifies per-cell, per-frame shape descriptors from cell labels."""

    quantity_id = "cell_shape"
    display_name = "Cell shape"
    # Cell labels only; no nucleus / contacts dependency. Pixel size (to emit
    # physical µm / µm²) is a global build param set in the Parameters panel.
    requires = ("cell_labels_path",)
    required_build_params = {"pixel_size_um": "pixel size (µm/px)"}
    #: Per-cell, per-frame shape descriptors, keyed (frame, cell_id).
    table_keys = ("frame", "cell_id")

    def compute_object_table(self, inputs, *, params=None):
        return compute_object_shape(
            inputs.cell_labels_path,
            pixel_size_um=inputs.pixel_size_um,
            object_key="cell_id",
        )
