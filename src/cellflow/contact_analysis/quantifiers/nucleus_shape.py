"""Nucleus-shape quantifier — the registry adapter over the shape core.

The nucleus twin of :mod:`.cell_shape`: it runs the same label-agnostic
:func:`build_object_shape` over the **nucleus** label stack instead of the cell
one, persisting ``4_contact_analysis/nucleus_shape.csv``. The object-key column
stays ``cell_id`` — a nucleus is nucleus-seeded so it carries its cell's shared
track id — so nothing in the pooling/plotting layer treats it differently.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from cellflow.contact_analysis.quantifier import PositionInputs, Quantifier
from cellflow.contact_analysis.shape import (
    build_object_shape,
    compute_object_shape,
    read_shape_table,
)


class NucleusShapeQuantifier(Quantifier):
    """Quantifies per-nucleus, per-frame shape descriptors from nucleus labels."""

    quantity_id = "nucleus_shape"
    display_name = "Nucleus shape"
    # Nucleus labels only; pixel size (to emit physical µm / µm²) is a global
    # build param set in the Parameters panel.
    requires = ("nucleus_labels_path",)
    required_build_params = {"pixel_size_um": "pixel size (µm/px)"}

    default_output_name = "nucleus_shape.csv"
    # Nucleus shape is keyed on its cell's shared track id (frame, cell_id); its
    # descriptors are namespaced by quantity_id (so the nucleus ``area`` never
    # collides with the cell ``area`` in a joined view).
    table_keys = ("frame", "cell_id")

    def build(
        self,
        inputs: PositionInputs,
        output_path: Path,
        *,
        params: dict | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        return build_object_shape(
            inputs.nucleus_labels_path,
            output_path,
            pixel_size_um=inputs.pixel_size_um,
            object_key="cell_id",
            source_path=inputs.position_dir,
            params=params,
            quantity_id=self.quantity_id,
            progress_cb=progress_cb,
        )

    def read(self, output_path: Path) -> dict[str, np.ndarray]:
        return read_shape_table(output_path)

    def object_table(self, output_path: Path) -> Mapping[str, np.ndarray]:
        return read_shape_table(output_path)

    def compute_object_table(self, inputs, *, params=None):
        return compute_object_shape(
            inputs.nucleus_labels_path,
            pixel_size_um=inputs.pixel_size_um,
            object_key="cell_id",
        )
