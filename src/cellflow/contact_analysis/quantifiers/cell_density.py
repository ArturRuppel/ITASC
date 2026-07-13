"""Cell-density quantifier — per-frame cell counts per field-of-view area.

Counts cells straight off the tracked **cell labels** (unique non-zero labels per
frame), so it is a cell-labels quantity, not a contacts-derived one. The
field-of-view area is **required** — it comes from the shared params bar and there
is no silent image-area fallback; ``density = n_cells / fov_area_mm2`` in
cells/mm². One ``all`` total row per frame (label-agnostic). A **pooled** quantity:
:meth:`compute_object_table` builds the tidy table in memory (no per-position
artifact).
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from cellflow.contact_analysis.contacts.neighborhood import cell_density
from cellflow.contact_analysis.quantifier import PositionInputs, Quantifier
from cellflow.contact_analysis.shape.core import read_label_stack


class CellDensityQuantifier(Quantifier):
    """Per ``(frame, label)`` cell count and density (cells/mm²)."""

    quantity_id = "cell_density"
    display_name = "Cell density"
    # Cell labels are the only hard input — counts come straight off them. The
    # field-of-view area is a required build *param* (validated in
    # ``compute_object_table``), not a PositionInputs field.
    requires = ("cell_labels_path",)
    # Density is keyed per (frame, label); ``label`` is always ``"all"``.
    table_keys = ("frame", "label")
    wants_build_params = True
    # The field-of-view area has no image-area fallback, so it is a hard build
    # requirement: the pooling loop skips Cell density (and the UI greys it out)
    # until it is set, instead of letting the compute raise.
    required_build_params = {"fov_area_mm2": "FOV area (mm²)"}

    def compute_object_table(
        self, inputs: PositionInputs, *, params: dict | None = None
    ) -> Mapping[str, np.ndarray]:
        fov = (params or {}).get("fov_area_mm2")
        if not (isinstance(fov, (int, float)) and fov > 0):
            raise ValueError(
                "Cell density requires a field-of-view area — set 'FOV area (mm²)' "
                "in Parameters before building."
            )
        frame_cells = _frame_cells_from_labels(inputs.cell_labels_path)
        return cell_density(frame_cells, fov_area_mm2=float(fov))


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
