"""Cell-shape quantifier — the registry adapter over the shape core.

Wraps :mod:`cellflow.aggregate_quantification.shape` so the studio can build and
read per-cell morphology through the generic :class:`Quantifier` interface. Its
persistence is a tidy ``aggregate_quantification/cell_shape.csv``;
:meth:`object_table` exposes that table to the plotting backend. The object-key
column is ``cell_id`` (the shared track id).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from cellflow.aggregate_quantification.quantifier import PositionInputs, Quantifier
from cellflow.aggregate_quantification.shape import (
    build_object_shape,
    read_shape_table,
)


class CellShapeQuantifier(Quantifier):
    """Quantifies per-cell, per-frame shape descriptors from cell labels."""

    quantity_id = "cell_shape"
    display_name = "Cell shape"
    # Cell labels only; no nucleus / contacts dependency. Pixel size (to emit
    # physical µm / µm²) is a global build param set in the Parameters panel.
    requires = ("cell_labels_path",)
    required_build_params = {"pixel_size_um": "pixel size (µm/px)"}

    #: Default artifact name; ``default_output`` nests it under the shared
    #: ``aggregate_quantification/`` folder. The builder mkdirs the parent.
    default_output_name = "cell_shape.csv"
    #: Per-cell, per-frame shape descriptors pool into the cells table.
    shape_table = "cells_by_frame"
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
            inputs.cell_labels_path,
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
