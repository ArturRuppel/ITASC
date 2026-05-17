"""Cell pipeline parameter section widget — sliders and config-builder methods."""
from __future__ import annotations

import os

from qtpy.QtWidgets import QWidget

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

        inner = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(0, 0, 0, 0)
        inner.setLayout(grid)

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

        # ─── Pack into the unified grid ─────────────────────────────
        row = 0

        add_section_header(grid, row, _heading("Flow Filtering")); row += 1
        add_section_pair_row(
            grid, row,
            "Median t:", self.ff_median_time_spin,
            "Median xy:", self.ff_median_space_spin,
        ); row += 1
        add_section_pair_row(
            grid, row,
            "Gauss t σ:", self.ff_gauss_time_spin,
            "Gauss xy σ:", self.ff_gauss_space_spin,
        ); row += 1

        add_section_header(grid, row, _heading("Foreground")); row += 1
        add_section_pair_row(grid, row, "Cellprob thr:", self.fg_cellprob_threshold_spin); row += 1

        add_section_header(grid, row, _heading("Contour — Cellprob Sweep")); row += 1
        add_section_pair_row(
            grid, row,
            "Min:", self.cp_min_spin,
            "Max:", self.cp_max_spin,
        ); row += 1
        add_section_pair_row(grid, row, "Step:", self.cp_step_spin); row += 1

        add_section_header(grid, row, _heading("Contour — Flow-Following")); row += 1
        add_section_pair_row(
            grid, row,
            "Flow weight:", self.ff_flow_weight_spin,
            "Step scale:", self.ff_step_scale_spin,
        ); row += 1
        add_section_pair_row(grid, row, "Max iter:", self.ff_max_iter_spin); row += 1

        add_section_header(grid, row, _heading("Contour — Gamma Averaging")); row += 1
        add_section_pair_row(
            grid, row,
            "Min:", self.gamma_min_spin,
            "Max:", self.gamma_max_spin,
        ); row += 1
        add_section_pair_row(grid, row, "Step:", self.gamma_step_spin); row += 1

        add_section_header(grid, row, _heading("Contour — Temporal Stabilization")); row += 1
        add_section_pair_row(
            grid, row,
            "Memory τ:", self.memory_tau_spin,
            "Floor:", self.memory_floor_spin,
        ); row += 1

        add_section_header(grid, row, _heading("Segmentation")); row += 1
        add_section_pair_row(
            grid, row,
            "α unary:", self.alpha_unary_spin,
            "λ spatial:", self.lambda_s_spin,
        ); row += 1
        add_section_pair_row(
            grid, row,
            "β spatial:", self.beta_s_spin,
            "λ temporal:", self.lambda_t_spin,
        ); row += 1
        add_section_pair_row(
            grid, row,
            "γ unary:", self.gamma_unary_spin,
            "Workers:", self.n_workers_spin,
        ); row += 1

        self.section = CollapsibleSection(
            "Parameters",
            inner,
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
