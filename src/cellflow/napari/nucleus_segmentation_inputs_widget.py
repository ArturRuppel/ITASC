"""Segmentation input parameter section for the nucleus workflow widget."""
from __future__ import annotations

from qtpy.QtWidgets import QWidget

from cellflow.napari._widget_helpers import (
    RangeThumbProxy as _RangeThumbProxy,
    _force_handle_label_width,
    drslider as _drslider,
    dslider as _dslider,
    heading as _heading,
    irslider as _irslider,
    islider as _islider,
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
        self.map_cellprob_range = _drslider(
            -20, 20, -3.0, 0.0, 1.0, 1,
            "Cellpose probability threshold range for averaged-map generation.",
        )
        self.map_cellprob_step_spin = _dslider(
            0.1, 10, 1.0, 0.5, 1,
            "Cellpose probability threshold step for averaged-map generation.",
        )

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

        # ─── Z slices: a range slider whose limits are set from the
        # loaded input file's z dimension via `set_z_extent`. ────────
        self.map_z_range = _irslider(
            0, 999, 0, 999, 1,
            "Z slice range (inclusive) for averaged-map generation.",
        )
        self.map_z_step_spin = _islider(
            1, 999, 1,
            tooltip="Z-slice step for averaged-map generation.",
        )

        # ─── Backwards-compatible per-thumb proxies ──────────────────
        # External callers (state save/load, threshold computation, tests)
        # access min/max values via *_min_spin / *_max_spin attributes;
        # expose those names by wrapping each thumb of the range sliders.
        self.map_cellprob_min_spin = _RangeThumbProxy(self.map_cellprob_range, 0)
        self.map_cellprob_max_spin = _RangeThumbProxy(self.map_cellprob_range, 1)
        self.map_z_start_spin = _RangeThumbProxy(self.map_z_range, 0)
        self.map_z_stop_spin = _RangeThumbProxy(self.map_z_range, 1)
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
        add_section_header(grid, row, _heading("Averaged Map")); row += 1
        add_section_pair_row(
            grid, row,
            "Cellprob:", self.map_cellprob_range,
            "Step:", self.map_cellprob_step_spin,
        ); row += 1

        add_section_header(grid, row, _heading("Z Slices")); row += 1
        add_section_pair_row(
            grid, row,
            "Z range:", self.map_z_range,
            "Step:", self.map_z_step_spin,
        ); row += 1

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
            "Segmentation Input Parameters",
            inner,
            expanded=True,
        )

    def set_z_extent(self, z_size: int | None) -> None:
        """Update the z range slider's limits to match the input volume.

        ``z_size`` is the number of z slices (so valid indices are
        ``[0, z_size - 1]``). Pass ``None`` to leave the current range
        unchanged. The thumbs are snapped into the new range while
        preserving the user's selection where possible; a fresh slider
        (still at its default open range) is collapsed onto the full
        extent."""
        if z_size is None or z_size <= 0:
            return
        new_max = int(z_size) - 1
        cur_lo, cur_hi = self.map_z_range.value()
        cur_min = self.map_z_range.minimum()
        cur_max = self.map_z_range.maximum()
        at_default = (cur_lo == cur_min and cur_hi == cur_max)
        self.map_z_range.setRange(0, new_max)
        if at_default:
            self.map_z_range.setValue((0, new_max))
        else:
            self.map_z_range.setValue((
                max(0, min(int(cur_lo), new_max)),
                max(0, min(int(cur_hi), new_max)),
            ))
        _force_handle_label_width(self.map_z_range)
        self.map_z_step_spin.setRange(1, max(1, new_max))
