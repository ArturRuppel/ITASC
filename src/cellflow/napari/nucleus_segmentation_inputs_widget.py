"""Segmentation input parameter section for the nucleus workflow widget."""
from __future__ import annotations

from qtpy.QtWidgets import QWidget

from cellflow.napari._widget_helpers import (
    RangeThumbProxy as _RangeThumbProxy,
    drslider as _drslider,
    dslider as _dslider,
    heading as _heading,
)
from cellflow.napari.ui_style import (
    add_section_header,
    add_section_pair_row,
    section_grid,
)
from cellflow.napari.widgets import CollapsibleSection


class NucleusSegmentationInputsWidget(QWidget):
    """Qt controls for nucleus segmentation input generation parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        inner = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(0, 0, 0, 0)
        # Range slider should take more horizontal space than step slider.
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(3, 1)
        inner.setLayout(grid)

        # ─── Range sliders (min/max combined) + step sliders ─────────
        self.source_contour_threshold_range = _drslider(
            0, 1, 0.1, 0.5, 0.05, 2,
            "Normalized contour threshold range for the source sweep.",
        )
        self.source_contour_threshold_step_spin = _dslider(
            0.001, 1, 0.1, 0.05, 3,
            "Step size for normalized contour source thresholds.",
        )

        self.source_foreground_threshold_range = _drslider(
            0, 1, 0.1, 0.5, 0.05, 2,
            "Normalized foreground-score threshold range for the source sweep.",
        )
        self.source_foreground_threshold_step_spin = _dslider(
            0.001, 1, 0.1, 0.05, 3,
            "Step size for normalized foreground source thresholds.",
        )

        # ─── Backwards-compatible per-thumb proxies ──────────────────
        # External callers (state save/load, threshold computation, tests)
        # access min/max values via *_min_spin / *_max_spin attributes;
        # expose those names by wrapping each thumb of the range sliders.
        self.source_contour_threshold_min_spin = _RangeThumbProxy(
            self.source_contour_threshold_range, 0
        )
        self.source_contour_threshold_max_spin = _RangeThumbProxy(
            self.source_contour_threshold_range, 1
        )
        self.source_foreground_threshold_min_spin = _RangeThumbProxy(
            self.source_foreground_threshold_range, 0
        )
        self.source_foreground_threshold_max_spin = _RangeThumbProxy(
            self.source_foreground_threshold_range, 1
        )

        # ─── Pack into the unified grid ─────────────────────────────
        row = 0
        add_section_header(grid, row, _heading("Source Sweep")); row += 1
        add_section_pair_row(
            grid, row,
            "Contour:", self.source_contour_threshold_range,
            "Step:", self.source_contour_threshold_step_spin,
        ); row += 1
        add_section_pair_row(
            grid, row,
            "Foreground:", self.source_foreground_threshold_range,
            "Step:", self.source_foreground_threshold_step_spin,
        ); row += 1

        self.section = CollapsibleSection(
            "Ultrack Input Parameters",
            inner,
            expanded=True,
        )
