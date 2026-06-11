"""Cell-shape quantifier — the registry adapter over the cell_shape core.

Wraps :mod:`cellflow.aggregate_quantification.cell_shape` so the studio can build
and read per-cell morphology through the generic :class:`Quantifier` interface.
Its persistence is ``cell_shape.h5`` (a tidy ``shape/table``); :meth:`object_table`
exposes that table to the plotting backend.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cellflow.aggregate_quantification.cell_shape.build import (
    build_cell_shape,
    read_cell_shape,
)
from cellflow.aggregate_quantification.quantifier import PositionInputs, Quantifier


class CellShapeQuantifier(Quantifier):
    """Quantifies per-cell, per-frame shape descriptors from cell labels."""

    quantity_id = "cell_shape"
    display_name = "Cell shape"
    # Cell labels plus a pixel size (to emit physical µm / µm²); no nucleus /
    # contacts dependency. A position with no resolvable pixel size is not
    # buildable until one is supplied.
    requires = ("cell_labels_path", "pixel_size_um")

    #: Default artifact name when a position does not dictate one.
    default_output_name = "cell_shape.h5"

    def build(
        self,
        inputs: PositionInputs,
        output_path: Path,
        *,
        params: dict | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        return build_cell_shape(
            cell_labels_path=inputs.cell_labels_path,
            output_path=output_path,
            pixel_size_um=inputs.pixel_size_um,
            source_path=inputs.position_dir,
            params=params,
            progress_cb=progress_cb,
        )

    def read(self, output_path: Path) -> dict[str, np.ndarray]:
        return read_cell_shape(output_path)

    def object_table(self, output_path: Path) -> Mapping[str, np.ndarray]:
        return read_cell_shape(output_path)
