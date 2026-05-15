"""Segmentation input parameter section for the nucleus workflow widget."""
from __future__ import annotations

from qtpy.QtWidgets import QVBoxLayout, QWidget

from cellflow.napari._widget_helpers import dspin as _dspin, ispin as _ispin
from cellflow.napari.ui_style import add_sweep_parameter_row, sweep_parameter_grid
from cellflow.napari.widgets import CollapsibleSection


class NucleusSegmentationInputsWidget(QWidget):
    """Qt controls for nucleus segmentation input generation parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        inner = QWidget(self)
        params_lay = QVBoxLayout(inner)
        params_lay.setContentsMargins(0, 0, 0, 0)
        params_lay.setSpacing(6)

        g = sweep_parameter_grid(horizontal_spacing=12)
        self.map_cellprob_min_spin = _dspin(
            -20, 20, -3.0, 1.0, 1,
            "Minimum Cellpose probability threshold for averaged-map generation.",
        )
        self.map_cellprob_max_spin = _dspin(
            -20, 20, 0.0, 1.0, 1,
            "Maximum Cellpose probability threshold for averaged-map generation.",
        )
        self.map_cellprob_step_spin = _dspin(
            0.1, 10, 1.0, 0.5, 1,
            "Cellpose probability threshold step for averaged-map generation.",
        )
        self.map_z_start_spin = _ispin(
            0, 999, 0,
            tooltip="First z slice included in averaged-map generation.",
        )
        self.map_z_stop_spin = _ispin(
            -1, 999, -1,
            tooltip="Last z slice included. -1 means all z slices.",
        )
        self.map_z_step_spin = _ispin(
            1, 999, 1,
            tooltip="Z-slice step for averaged-map generation.",
        )
        self.source_contour_threshold_min_spin = _dspin(
            0, 1, 0.1, 0.05, 2,
            "Minimum normalized contour threshold for the source sweep.",
        )
        self.source_contour_threshold_max_spin = _dspin(
            0, 1, 0.5, 0.05, 2,
            "Maximum normalized contour threshold for the source sweep.",
        )
        self.source_contour_threshold_step_spin = _dspin(
            0.001, 1, 0.1, 0.05, 3,
            "Step size for normalized contour source thresholds.",
        )
        self.source_foreground_threshold_min_spin = _dspin(
            0, 1, 0.1, 0.05, 2,
            "Minimum normalized foreground-score threshold for the source sweep.",
        )
        self.source_foreground_threshold_max_spin = _dspin(
            0, 1, 0.5, 0.05, 2,
            "Maximum normalized foreground-score threshold for the source sweep.",
        )
        self.source_foreground_threshold_step_spin = _dspin(
            0.001, 1, 0.1, 0.05, 3,
            "Step size for normalized foreground source thresholds.",
        )
        add_sweep_parameter_row(
            g, 1, "Cellprob:",
            self.map_cellprob_min_spin,
            self.map_cellprob_max_spin,
            self.map_cellprob_step_spin,
        )
        add_sweep_parameter_row(
            g, 2, "Z:",
            self.map_z_start_spin,
            self.map_z_stop_spin,
            self.map_z_step_spin,
        )
        add_sweep_parameter_row(
            g, 3, "Contour:",
            self.source_contour_threshold_min_spin,
            self.source_contour_threshold_max_spin,
            self.source_contour_threshold_step_spin,
        )
        add_sweep_parameter_row(
            g, 4, "Foreground:",
            self.source_foreground_threshold_min_spin,
            self.source_foreground_threshold_max_spin,
            self.source_foreground_threshold_step_spin,
        )
        params_lay.addLayout(g)

        self.section = CollapsibleSection(
            "Segmentation Input Parameters",
            inner,
            expanded=True,
            title_role="params",
            title_level=1,
        )
