"""Nucleus-shape quantifier — the registry adapter over the shape core.

The nucleus twin of :mod:`.cell_shape`: :meth:`compute_object_table` runs the same
label-agnostic :func:`compute_object_shape` over the **nucleus** label stack
instead of the cell one and pools the tidy table in memory (no per-position
artifact). The object-key column stays ``cell_id`` — a nucleus is nucleus-seeded
so it carries its cell's shared track id — so nothing in the pooling layer treats
it differently.
"""
from __future__ import annotations

from itasc.contact_analysis.quantifier import Quantifier
from itasc.contact_analysis.shape import compute_object_shape


class NucleusShapeQuantifier(Quantifier):
    """Quantifies per-nucleus, per-frame shape descriptors from nucleus labels."""

    quantity_id = "nucleus_shape"
    display_name = "Nucleus shape"
    # Nucleus labels only; pixel size (to emit physical µm / µm²) is a global
    # build param set in the Parameters panel.
    requires = ("nucleus_labels_path",)
    required_build_params = {"pixel_size_um": "pixel size (µm/px)"}
    # Nucleus shape is keyed on its cell's shared track id (frame, cell_id); its
    # descriptors are namespaced by quantity_id (so the nucleus ``area`` never
    # collides with the cell ``area`` in a joined view).
    table_keys = ("frame", "cell_id")

    def compute_object_table(self, inputs, *, params=None):
        return compute_object_shape(
            inputs.nucleus_labels_path,
            pixel_size_um=inputs.pixel_size_um,
            object_key="cell_id",
        )
