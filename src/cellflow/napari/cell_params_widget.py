"""Cell pipeline parameter section widget — spinboxes and config-builder methods."""
from __future__ import annotations

import os

from qtpy.QtWidgets import QVBoxLayout, QWidget

from cellflow.napari._widget_helpers import dspin as _dspin, heading as _heading, ispin as _ispin
from cellflow.napari.ui_style import add_block_pair_row, block_grid, compact_spinbox
from cellflow.napari.widgets import CollapsibleSection
from cellflow.segmentation import FlowFollowingParams


class CellParamsWidget(QWidget):
    """Qt controls for all cell pipeline parameters.

    Owns:
    - Flow-filtering spinboxes
    - Foreground cellprob threshold
    - Contour cellprob sweep spinboxes
    - Contour flow-following spinboxes
    - Gamma averaging spinboxes
    - Temporal stabilization spinboxes
    - Segmentation (ICM) spinboxes

    The ``section`` attribute is the wrapping ``CollapsibleSection`` that callers
    should add to their layout.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        inner = QWidget(self)
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # Flow filtering
        lay.addWidget(_heading("Flow Filtering"))
        g = block_grid(horizontal_spacing=12)
        self.ff_median_time_spin = _ispin(1, 15, 3)
        self.ff_median_space_spin = _ispin(1, 15, 5)
        self.ff_gauss_time_spin = _dspin(0, 10, 0, 0.1, 1)
        self.ff_gauss_space_spin = _dspin(0, 10, 0, 0.1, 1)
        add_block_pair_row(g, 0,
            "Median t:", compact_spinbox(self.ff_median_time_spin),
            "Median xy:", compact_spinbox(self.ff_median_space_spin))
        add_block_pair_row(g, 1,
            "Gauss t σ:", compact_spinbox(self.ff_gauss_time_spin),
            "Gauss xy σ:", compact_spinbox(self.ff_gauss_space_spin))
        lay.addLayout(g)

        # Foreground
        lay.addWidget(_heading("Foreground"))
        g = block_grid(horizontal_spacing=12)
        self.fg_cellprob_threshold_spin = _dspin(
            0, 1, 0.5, 0.01, 2,
            "Cellpose probability threshold (sigmoid space).",
        )
        add_block_pair_row(g, 0,
            "Cellprob thr:", compact_spinbox(self.fg_cellprob_threshold_spin))
        lay.addLayout(g)

        # Contour maps — consensus sweep
        lay.addWidget(_heading("Contour — Cellprob Sweep"))
        g = block_grid(horizontal_spacing=12)
        self.cp_min_spin = _dspin(0, 1, 0.05, 0.05)
        self.cp_max_spin = _dspin(0, 1, 0.50, 0.05)
        self.cp_step_spin = _dspin(0.01, 1, 0.05, 0.01)
        add_block_pair_row(g, 0,
            "Min:", compact_spinbox(self.cp_min_spin),
            "Max:", compact_spinbox(self.cp_max_spin))
        add_block_pair_row(g, 1,
            "Step:", compact_spinbox(self.cp_step_spin))
        lay.addLayout(g)

        # Contour — flow-following
        lay.addWidget(_heading("Contour — Flow-Following"))
        g = block_grid(horizontal_spacing=12)
        self.ff_flow_weight_spin = _dspin(
            0, 1, 0.5, 0.05, 2,
            "Blend: flow direction (1) vs EDT gravity (0).",
        )
        self.ff_step_scale_spin = _dspin(0.01, 2, 0.2, 0.05, 2, "Step-size multiplier.")
        self.ff_max_iter_spin = _ispin(10, 2000, 100, 10, "Max integration steps.")
        add_block_pair_row(g, 0,
            "Flow weight:", compact_spinbox(self.ff_flow_weight_spin),
            "Step scale:", compact_spinbox(self.ff_step_scale_spin))
        add_block_pair_row(g, 1,
            "Max iter:", compact_spinbox(self.ff_max_iter_spin))
        lay.addLayout(g)

        # Contour — gamma averaging
        lay.addWidget(_heading("Contour — Gamma Averaging"))
        g = block_grid(horizontal_spacing=12)
        self.gamma_min_spin = _dspin(0.05, 5, 1.0, 0.05)
        self.gamma_max_spin = _dspin(0.05, 5, 1.0, 0.05)
        self.gamma_step_spin = _dspin(0.05, 2, 0.25, 0.05)
        add_block_pair_row(g, 0,
            "Min:", compact_spinbox(self.gamma_min_spin),
            "Max:", compact_spinbox(self.gamma_max_spin))
        add_block_pair_row(g, 1,
            "Step:", compact_spinbox(self.gamma_step_spin))
        lay.addLayout(g)

        # Contour — temporal stabilization
        lay.addWidget(_heading("Contour — Temporal Stabilization"))
        g = block_grid(horizontal_spacing=12)
        self.memory_tau_spin = _dspin(
            0, 1, 0, 0.01, 3,
            "Contour memory τ. 0 = disabled.",
        )
        self.memory_floor_spin = _dspin(
            0.001, 0.5, 0.01, 0.005, 3,
            "Minimum alpha — prevents permanent ghosting.",
        )
        add_block_pair_row(g, 0,
            "Memory τ:", compact_spinbox(self.memory_tau_spin),
            "Floor:", compact_spinbox(self.memory_floor_spin))
        lay.addLayout(g)

        # Segmentation — ICM
        lay.addWidget(_heading("Segmentation"))
        g = block_grid(horizontal_spacing=12)
        self.alpha_unary_spin = _dspin(
            0, 1000, 4.0, 0.1, 2,
            "Contour weight: 1 + α·contour.",
        )
        self.lambda_s_spin = _dspin(0, 1000, 1.0, 0.1, 2, "Spatial Potts weight.")
        self.beta_s_spin = _dspin(
            0, 1000, 5.0, 0.1, 2,
            "Contour sensitivity: exp(-β·avg_contour).",
        )
        self.lambda_t_spin = _dspin(0, 1000, 1.0, 0.1, 2, "Temporal Potts weight.")
        self.gamma_unary_spin = _dspin(
            0, 100, 0, 0.1, 2,
            "(1 − foreground_score) weight. 0 = contour-only.",
        )
        self.n_workers_spin = _ispin(
            1, max(1, os.cpu_count() or 1),
            min(4, os.cpu_count() or 1),
            tooltip="Parallel workers for geodesic computation.",
        )
        add_block_pair_row(g, 0,
            "α unary:", compact_spinbox(self.alpha_unary_spin),
            "λ spatial:", compact_spinbox(self.lambda_s_spin),
            field_width=92)
        add_block_pair_row(g, 1,
            "β spatial:", compact_spinbox(self.beta_s_spin),
            "λ temporal:", compact_spinbox(self.lambda_t_spin),
            field_width=92)
        add_block_pair_row(g, 2,
            "γ unary:", compact_spinbox(self.gamma_unary_spin),
            "Workers:", compact_spinbox(self.n_workers_spin),
            field_width=92)
        lay.addLayout(g)

        self.section = CollapsibleSection(
            "Parameters",
            inner,
            expanded=False,
        )

    # ── Config-builder methods ────────────────────────────────────────────────

    def flow_filter_params(self) -> FlowFollowingParams:
        """Build FlowFollowingParams from the flow-filtering spinboxes."""
        return FlowFollowingParams(
            median_kernel_time=int(self.ff_median_time_spin.value()),
            median_kernel_space=int(self.ff_median_space_spin.value()),
            gaussian_sigma_time=float(self.ff_gauss_time_spin.value()),
            gaussian_sigma_space=float(self.ff_gauss_space_spin.value()),
        )

    def cellprob_thresholds(self) -> list[float]:
        """Return the cellprob sweep thresholds from the current spinbox values."""
        import numpy as np
        step = self.cp_step_spin.value()
        return list(np.arange(self.cp_min_spin.value(), self.cp_max_spin.value() + step / 2, step))

    def gammas(self) -> list[float]:
        """Return the gamma sweep values from the current spinbox values."""
        import numpy as np
        step = self.gamma_step_spin.value()
        return list(np.arange(self.gamma_min_spin.value(), self.gamma_max_spin.value() + step / 2, step))

    def contour_ff_params(self) -> FlowFollowingParams:
        """Build FlowFollowingParams for contour flow-following from the current spinboxes."""
        return FlowFollowingParams(
            median_kernel_time=1, median_kernel_space=1,
            gaussian_sigma_time=0.0, gaussian_sigma_space=0.0,
            flow_weight=self.ff_flow_weight_spin.value(),
            flow_step_scale=self.ff_step_scale_spin.value(),
            max_iterations=self.ff_max_iter_spin.value(),
        )
