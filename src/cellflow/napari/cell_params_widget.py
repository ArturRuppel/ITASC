"""Cell pipeline parameter section widget — sliders and config-builder methods."""
from __future__ import annotations

import os

from qtpy.QtWidgets import QVBoxLayout, QWidget

from cellflow.napari._widget_helpers import (
    dslider as _dslider,
    heading as _heading,
    islider as _islider,
)
from cellflow.napari.ui_style import (
    add_section_header,
    add_section_pair_row,
    section_grid,
)
from cellflow.napari.widgets import CollapsibleSection
from cellflow.segmentation import FlowFollowingParams


class CellParamsWidget(QWidget):
    """Qt controls for all cell pipeline parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ─── Build all sliders up front ─────────────────────────────
        # Flow filtering
        self.ff_median_time_spin = _islider(1, 15, 3)
        self.ff_median_space_spin = _islider(1, 15, 5)
        self.ff_gauss_time_spin = _dslider(0, 10, 0, 0.1, 1)
        self.ff_gauss_space_spin = _dslider(0, 10, 0, 0.1, 1)

        # Foreground
        self.fg_cellprob_threshold_spin = _dslider(
            0, 1, 0.5, 0.01, 2,
            "Cellpose probability threshold (sigmoid space).",
        )

        # Contour — cellprob sweep
        self.cp_min_spin = _dslider(0, 1, 0.05, 0.05, 2)
        self.cp_max_spin = _dslider(0, 1, 0.50, 0.05, 2)
        self.cp_step_spin = _dslider(0.01, 1, 0.05, 0.01, 2)

        # Contour — flow-following
        self.ff_flow_weight_spin = _dslider(
            0, 1, 0.5, 0.05, 2,
            "Blend: flow direction (1) vs EDT gravity (0).",
        )
        self.ff_step_scale_spin = _dslider(0.01, 2, 0.2, 0.05, 2, "Step-size multiplier.")
        self.ff_max_iter_spin = _islider(10, 2000, 100, 10, "Max integration steps.")

        # Contour — gamma averaging
        self.gamma_min_spin = _dslider(0.05, 5, 1.0, 0.05, 2)
        self.gamma_max_spin = _dslider(0.05, 5, 1.0, 0.05, 2)
        self.gamma_step_spin = _dslider(0.05, 2, 0.25, 0.05, 2)

        # Contour — temporal stabilization
        self.memory_tau_spin = _dslider(
            0, 1, 0, 0.01, 3,
            "Contour memory τ. 0 = disabled.",
        )
        self.memory_floor_spin = _dslider(
            0.001, 0.5, 0.01, 0.005, 3,
            "Minimum alpha — prevents permanent ghosting.",
        )

        # Segmentation — ICM
        self.alpha_unary_spin = _dslider(
            0, 1000, 4.0, 0.1, 2,
            "Contour weight: 1 + α·contour.",
        )
        self.lambda_s_spin = _dslider(0, 1000, 1.0, 0.1, 2, "Spatial Potts weight.")
        self.beta_s_spin = _dslider(
            0, 1000, 5.0, 0.1, 2,
            "Contour sensitivity: exp(-β·avg_contour).",
        )
        self.lambda_t_spin = _dslider(0, 1000, 1.0, 0.1, 2, "Temporal Potts weight.")
        self.gamma_unary_spin = _dslider(
            0, 100, 0, 0.1, 2,
            "(1 − foreground_score) weight. 0 = contour-only.",
        )
        self.n_workers_spin = _islider(
            1, max(1, os.cpu_count() or 1),
            min(4, os.cpu_count() or 1),
            tooltip="Parallel workers for geodesic computation.",
        )

        # ─── Pack into stage-specific grids ─────────────────────────
        flow_inner = QWidget(self)
        flow_grid = section_grid()
        flow_grid.setContentsMargins(0, 0, 0, 0)
        flow_inner.setLayout(flow_grid)
        row = 0
        add_section_header(flow_grid, row, _heading("Flow Filtering")); row += 1
        add_section_pair_row(
            flow_grid, row,
            "Median t:", self.ff_median_time_spin,
            "Median xy:", self.ff_median_space_spin,
        ); row += 1
        add_section_pair_row(
            flow_grid, row,
            "Gauss t σ:", self.ff_gauss_time_spin,
            "Gauss xy σ:", self.ff_gauss_space_spin,
        ); row += 1

        foreground_inner = QWidget(self)
        foreground_grid = section_grid()
        foreground_grid.setContentsMargins(0, 0, 0, 0)
        foreground_inner.setLayout(foreground_grid)
        row = 0
        add_section_header(foreground_grid, row, _heading("Foreground")); row += 1
        add_section_pair_row(
            foreground_grid, row,
            "Cellprob thr:", self.fg_cellprob_threshold_spin,
        ); row += 1

        contour_inner = QWidget(self)
        contour_grid = section_grid()
        contour_grid.setContentsMargins(0, 0, 0, 0)
        contour_inner.setLayout(contour_grid)
        row = 0
        add_section_header(contour_grid, row, _heading("Contour — Cellprob Sweep")); row += 1
        add_section_pair_row(
            contour_grid, row,
            "Min:", self.cp_min_spin,
            "Max:", self.cp_max_spin,
        ); row += 1
        add_section_pair_row(contour_grid, row, "Step:", self.cp_step_spin); row += 1

        add_section_header(contour_grid, row, _heading("Contour — Flow-Following")); row += 1
        add_section_pair_row(
            contour_grid, row,
            "Flow weight:", self.ff_flow_weight_spin,
            "Step scale:", self.ff_step_scale_spin,
        ); row += 1
        add_section_pair_row(contour_grid, row, "Max iter:", self.ff_max_iter_spin); row += 1

        add_section_header(contour_grid, row, _heading("Contour — Gamma Averaging")); row += 1
        add_section_pair_row(
            contour_grid, row,
            "Min:", self.gamma_min_spin,
            "Max:", self.gamma_max_spin,
        ); row += 1
        add_section_pair_row(contour_grid, row, "Step:", self.gamma_step_spin); row += 1

        add_section_header(contour_grid, row, _heading("Contour — Temporal Stabilization")); row += 1
        add_section_pair_row(
            contour_grid, row,
            "Memory τ:", self.memory_tau_spin,
            "Floor:", self.memory_floor_spin,
        ); row += 1

        segmentation_inner = QWidget(self)
        segmentation_grid = section_grid()
        segmentation_grid.setContentsMargins(0, 0, 0, 0)
        segmentation_inner.setLayout(segmentation_grid)
        row = 0
        add_section_header(segmentation_grid, row, _heading("Segmentation")); row += 1
        add_section_pair_row(
            segmentation_grid, row,
            "α unary:", self.alpha_unary_spin,
            "λ spatial:", self.lambda_s_spin,
        ); row += 1
        add_section_pair_row(
            segmentation_grid, row,
            "β spatial:", self.beta_s_spin,
            "λ temporal:", self.lambda_t_spin,
        ); row += 1
        add_section_pair_row(
            segmentation_grid, row,
            "γ unary:", self.gamma_unary_spin,
            "Workers:", self.n_workers_spin,
        ); row += 1

        self.flow_filter_section = CollapsibleSection(
            "Flow Filtering",
            flow_inner,
            expanded=False,
        )
        self.foreground_section = CollapsibleSection(
            "Foreground Masks",
            foreground_inner,
            expanded=False,
        )
        self.contour_section = CollapsibleSection(
            "Contours",
            contour_inner,
            expanded=False,
        )
        self.segmentation_section = CollapsibleSection(
            "Segmentation",
            segmentation_inner,
            expanded=False,
        )

        container = QWidget(self)
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        for section in (
            self.flow_filter_section,
            self.foreground_section,
            self.contour_section,
            self.segmentation_section,
        ):
            lay.addWidget(section)

        self.section = CollapsibleSection(
            "Parameters",
            container,
            expanded=False,
        )

    # ── Config-builder methods ────────────────────────────────────────────────

    def flow_filter_params(self) -> FlowFollowingParams:
        """Build FlowFollowingParams from the flow-filtering controls."""
        return FlowFollowingParams(
            median_kernel_time=int(self.ff_median_time_spin.value()),
            median_kernel_space=int(self.ff_median_space_spin.value()),
            gaussian_sigma_time=float(self.ff_gauss_time_spin.value()),
            gaussian_sigma_space=float(self.ff_gauss_space_spin.value()),
        )

    def cellprob_thresholds(self) -> list[float]:
        """Return the cellprob sweep thresholds from the current control values."""
        import numpy as np
        step = self.cp_step_spin.value()
        return list(np.arange(self.cp_min_spin.value(), self.cp_max_spin.value() + step / 2, step))

    def gammas(self) -> list[float]:
        """Return the gamma sweep values from the current control values."""
        import numpy as np
        step = self.gamma_step_spin.value()
        return list(np.arange(self.gamma_min_spin.value(), self.gamma_max_spin.value() + step / 2, step))

    def contour_ff_params(self) -> FlowFollowingParams:
        """Build FlowFollowingParams for contour flow-following from the current controls."""
        return FlowFollowingParams(
            median_kernel_time=1, median_kernel_space=1,
            gaussian_sigma_time=0.0, gaussian_sigma_space=0.0,
            flow_weight=self.ff_flow_weight_spin.value(),
            flow_step_scale=self.ff_step_scale_spin.value(),
            max_iterations=self.ff_max_iter_spin.value(),
        )
