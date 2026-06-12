"""Cell-density quantifier — per-label cell counts per field-of-view area.

A contacts-derived Build product. The field-of-view area comes from the shared
params bar when set (one value for all positions); otherwise it is the position's
full image area ``H·W·pixel² / 1e6`` mm², resolved from the contacts artifact's
cell-label TIFF and the position's pixel size. ``density`` is ``NaN`` when no area
can be resolved (the plot then simply has nothing to show for that position).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from cellflow.aggregate_quantification.contacts.neighborhood import cell_density
from cellflow.aggregate_quantification.contacts.reader import PositionContactAnalysis
from cellflow.aggregate_quantification.quantifier import PositionInputs, Quantifier
from cellflow.aggregate_quantification.quantifiers import _contacts_derived as derived


class CellDensityQuantifier(Quantifier):
    """Per ``(frame, label)`` cell count and density (cells/mm²)."""

    quantity_id = "cell_density"
    display_name = "Cell density"
    requires = ("contact_analysis_path",)
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
        analysis = derived.load_analysis(inputs)
        labels = derived.load_labels(inputs) or {}
        fov = _fov_area_mm2(inputs, analysis, (params or {}).get("fov_area_mm2"))
        table = cell_density(analysis, labels, fov_area_mm2=fov)
        return derived.persist(output_path, table)

    def read(self, output_path: Path) -> dict[str, np.ndarray]:
        return derived.read_derived_table(output_path)

    def object_table(self, output_path: Path) -> Mapping[str, np.ndarray]:
        return derived.read_derived_table(output_path)


def _fov_area_mm2(
    inputs: PositionInputs, analysis: PositionContactAnalysis, override: float | None
) -> float | None:
    """The field-of-view area in mm² — *override* when given, else the image area."""
    if override is not None and override > 0:
        return float(override)
    pixel = inputs.pixel_size_um
    if not pixel or pixel <= 0:
        return None
    shape = _image_shape_hw(analysis.cell_tracked_labels_path)
    if shape is None:
        return None
    height, width = shape
    return height * width * pixel * pixel / 1e6


def _image_shape_hw(path: str) -> tuple[int, int] | None:
    """``(height, width)`` of a label TIFF without loading pixels; ``None`` on failure."""
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tf:
            shape = tf.series[0].shape
    except Exception:  # pragma: no cover - unreadable/missing TIFF → no default FOV
        return None
    if len(shape) >= 2:
        return int(shape[-2]), int(shape[-1])
    return None
