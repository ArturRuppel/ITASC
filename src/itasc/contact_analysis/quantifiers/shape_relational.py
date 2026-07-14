"""Relational shape quantifier — nucleus-vs-cell paired morphology.

Wraps :func:`itasc.contact_analysis.shape.compute_relational_table`, which
pairs each nucleus with its cell on the shared ``(frame, id)`` key and emits
relational quantities (nuclear:cell area ratio, centroid offset, orientation
delta, …). Needs **both** label stacks plus a pixel size. A **pooled** quantity:
:meth:`compute_object_table` builds the tidy table in memory (no per-position
artifact). The object-key column is ``cell_id``, so the relational table pools
through the same path as the per-source shape tables.
"""
from __future__ import annotations

from itasc.contact_analysis.quantifier import Quantifier
from itasc.contact_analysis.shape import compute_relational_table


class ShapeRelationalQuantifier(Quantifier):
    """Quantifies per-(frame, cell) relational nucleus-vs-cell shape."""

    quantity_id = "shape_relational"
    display_name = "Nucleus–cell shape"
    # Both label stacks (to pair); pixel size (to emit physical units) is a
    # global build param set in the Parameters panel.
    requires = ("cell_labels_path", "nucleus_labels_path")
    required_build_params = {"pixel_size_um": "pixel size (µm/px)"}
    # Per-cell nucleus↔cell relational descriptors, keyed (frame, cell_id).
    table_keys = ("frame", "cell_id")

    def compute_object_table(self, inputs, *, params=None):
        return compute_relational_table(
            inputs.cell_labels_path,
            inputs.nucleus_labels_path,
            pixel_size_um=inputs.pixel_size_um,
        )
