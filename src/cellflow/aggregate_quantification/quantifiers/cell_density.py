"""Cell-density quantifier — per-label cell counts per field-of-view area.

Counts cells straight off the tracked **cell labels** (unique non-zero labels per
frame), so it is a cell-labels Build product, not a contacts-derived one. The
field-of-view area is **required** — it comes from the shared params bar and there
is no silent image-area fallback; ``density = n_cells / fov_area_mm2`` in
cells/mm². When an NLS classification sidecar exists the counts are also broken
down per class; without it only the ``all`` total row is emitted.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from cellflow.aggregate_quantification.contacts.neighborhood import cell_density
from cellflow.aggregate_quantification.quantifier import PositionInputs, Quantifier
from cellflow.aggregate_quantification.quantifiers._contacts_derived import (
    nls_labels_for_position,
)
from cellflow.aggregate_quantification.quantifiers._tidy_table import (
    persist,
    read_derived_table,
)
from cellflow.aggregate_quantification.shape.core import read_label_stack


class CellDensityQuantifier(Quantifier):
    """Per ``(frame, label)`` cell count and density (cells/mm²)."""

    quantity_id = "cell_density"
    display_name = "Cell density"
    # Cell labels are the only hard input — counts come straight off them. The
    # field-of-view area is a required build *param* (validated in ``build``), not
    # a PositionInputs field, so it is enforced there rather than gated here.
    requires = ("cell_labels_path",)
    default_output_name = "cell_density.csv"
    wants_build_params = True

    def build(
        self,
        inputs: PositionInputs,
        output_path: Path,
        *,
        params: dict | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        fov = (params or {}).get("fov_area_mm2")
        if not (isinstance(fov, (int, float)) and fov > 0):
            raise ValueError(
                "Cell density requires a field-of-view area — set 'FOV area (mm²)' "
                "in Parameters before building."
            )
        frame_cells = _frame_cells_from_labels(inputs.cell_labels_path)
        labels = nls_labels_for_position(inputs.position_dir) or {}
        table = cell_density(frame_cells, labels, fov_area_mm2=float(fov))
        return persist(output_path, table)

    def read(self, output_path: Path) -> dict[str, np.ndarray]:
        return read_derived_table(output_path)

    def object_table(self, output_path: Path) -> Mapping[str, np.ndarray]:
        return read_derived_table(output_path)


def _frame_cells_from_labels(labels_path: Path | None) -> dict[int, list[int]]:
    """``frame -> [cell_id, …]`` from a tracked cell-label TIFF.

    The unique non-zero labels of frame *i* are the cells present in it — the same
    set the contacts ``cells`` table holds (it is one ``regionprops`` row per
    label), so this reproduces the old per-frame counts without the artifact.
    """
    stack = read_label_stack(Path(labels_path))
    out: dict[int, list[int]] = {}
    for frame_idx, frame in enumerate(stack):
        ids = np.unique(frame)
        out[frame_idx] = [int(i) for i in ids if i != 0]
    return out
