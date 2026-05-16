"""Ultrack tracking and database-generation parameter section for the nucleus workflow widget."""
from __future__ import annotations

import os

from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QVBoxLayout, QWidget

from cellflow.napari._widget_helpers import dspin as _dspin, heading as _heading, ispin as _ispin
from cellflow.napari.ui_style import (
    add_block_checkbox_row,
    add_block_pair_row,
    block_grid,
    compact_spinbox,
)
from cellflow.napari.widgets import CollapsibleSection
from cellflow.tracking_ultrack.config import TrackingConfig as _UltrackConfig


class NucleusTrackingInputsWidget(QWidget):
    """Qt controls for Ultrack tracking and database-generation parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        params_inner = QWidget(self)
        lay = QVBoxLayout(params_inner)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # DB Generation — Candidates
        lay.addWidget(_heading("DB Generation — Candidates"))
        g = block_grid(horizontal_spacing=12)
        self.db_gen_min_area_spin = _ispin(0, 1_000_000, 300, tooltip="Minimum segment area in pixels.")
        self.db_gen_max_area_spin = _ispin(0, 10_000_000, 100_000, tooltip="Maximum segment area in pixels.")
        self.db_gen_min_frontier_spin = _dspin(
            0, 1, 0.0, 0.01, 3,
            "Minimum boundary fraction to keep a candidate.",
        )
        self.db_gen_ws_hierarchy_combo = QComboBox()
        self.db_gen_ws_hierarchy_combo.addItems(["area", "dynamics", "volume"])
        self.db_gen_n_workers_spin = _ispin(
            1, max(1, os.cpu_count() or 1), 1,
            tooltip="Parallel workers for segmentation.",
        )
        add_block_pair_row(g, 0,
            "Min area:", compact_spinbox(self.db_gen_min_area_spin),
            "Max area:", compact_spinbox(self.db_gen_max_area_spin))
        add_block_pair_row(g, 1,
            "Min frontier:", compact_spinbox(self.db_gen_min_frontier_spin))
        add_block_pair_row(g, 2,
            "WS hierarchy:", self.db_gen_ws_hierarchy_combo,
            "Workers:", compact_spinbox(self.db_gen_n_workers_spin))
        lay.addLayout(g)

        # DB Generation — Linking
        lay.addWidget(_heading("DB Generation — Linking"))
        g = block_grid(horizontal_spacing=12)
        self.db_gen_max_dist_spin = _dspin(0, 500, 15.0, 1.0, 1)
        self.db_gen_max_neighbors_spin = _ispin(1, 50, 5)
        self.db_gen_linking_mode_combo = QComboBox()
        self.db_gen_linking_mode_combo.addItems(["default", "shape"])
        self.db_gen_area_weight_spin = _dspin(0, 10, 1.0, 0.1, 2)
        self.db_gen_area_weight_spin.setEnabled(False)
        self.db_gen_iou_weight_spin = _dspin(0, 10, 1.0, 0.1, 2)
        self.db_gen_iou_weight_spin.setEnabled(False)
        self.db_gen_distance_weight_spin = _dspin(0, 10, 0.05, 0.01, 3)
        self.db_gen_distance_weight_spin.setEnabled(False)
        add_block_pair_row(g, 0,
            "Max dist:", compact_spinbox(self.db_gen_max_dist_spin),
            "Max neighbors:", compact_spinbox(self.db_gen_max_neighbors_spin))
        add_block_pair_row(g, 1,
            "Linking mode:", self.db_gen_linking_mode_combo,
            "Area wt:", compact_spinbox(self.db_gen_area_weight_spin))
        add_block_pair_row(g, 2,
            "IoU wt:", compact_spinbox(self.db_gen_iou_weight_spin),
            "Dist wt:", compact_spinbox(self.db_gen_distance_weight_spin))
        lay.addLayout(g)

        # DB Generation — Scoring
        lay.addWidget(_heading("DB Generation — Scoring"))
        g = block_grid(horizontal_spacing=12)
        self.db_gen_quality_weight_spin = _dspin(
            0, 10, 1.0, 0.05, 2,
            "Weight applied to signal-based segmentation quality.",
        )
        self.db_gen_quality_exp_spin = _dspin(
            0.1, 50, 8.0, 0.5, 2,
            "Raises signal-based quality before storing as node_prob.",
        )
        self.db_gen_circularity_weight_spin = _dspin(
            0, 10, 0.25, 0.05, 2,
            "Weight applied to shape circularity.",
        )
        add_block_pair_row(g, 0,
            "Quality wt:", compact_spinbox(self.db_gen_quality_weight_spin),
            "Quality exp:", compact_spinbox(self.db_gen_quality_exp_spin))
        add_block_pair_row(g, 1,
            "Circularity wt:", compact_spinbox(self.db_gen_circularity_weight_spin))
        lay.addLayout(g)

        # DB Generation — Validated Seed Prior
        lay.addWidget(_heading("DB Generation — Validated Seed Prior"))
        g = block_grid(horizontal_spacing=12)
        self.db_gen_use_validated_check = QCheckBox("Use validated corrections")
        add_block_checkbox_row(g, 0, self.db_gen_use_validated_check)
        lay.addLayout(g)

        from cellflow.napari._widget_helpers import separator as _separator
        lay.addWidget(_separator())

        # Ultrack — Track Scope
        lay.addWidget(_heading("Ultrack — Track Scope"))
        g = block_grid(horizontal_spacing=12)
        self.ultrack_max_partitions_spin = _ispin(
            0, 1000, 30, tooltip="0 = use all partitions.")
        self.ultrack_n_frames_spin = _ispin(
            0, 10000, 0, tooltip="0 = process all frames.")
        add_block_pair_row(g, 0,
            "Max partitions:", compact_spinbox(self.ultrack_max_partitions_spin),
            "N frames:", compact_spinbox(self.ultrack_n_frames_spin))
        lay.addLayout(g)

        # Ultrack — Event Penalties
        lay.addWidget(_heading("Ultrack — Event Penalties"))
        g = block_grid(horizontal_spacing=12)
        self.ultrack_appear_spin = _dspin(-10, 0, -0.1, 0.05, 3)
        self.ultrack_disappear_spin = _dspin(-10, 0, -0.1, 0.05, 3)
        self.ultrack_division_spin = _dspin(
            -10, 0, -0.001, 0.05, 3,
            "ILP penalty for divisions. More negative = fewer divisions.",
        )
        add_block_pair_row(g, 0,
            "Appear:", compact_spinbox(self.ultrack_appear_spin),
            "Disappear:", compact_spinbox(self.ultrack_disappear_spin))
        add_block_pair_row(g, 1,
            "Division:", compact_spinbox(self.ultrack_division_spin))
        lay.addLayout(g)

        # Ultrack — Solver
        lay.addWidget(_heading("Ultrack — Solver"))
        g = block_grid(horizontal_spacing=12)
        self.ultrack_power_spin = _dspin(
            0.1, 20, 4.0, 0.5, 2,
            "Solver transform for node_prob and link weights (link_function=power).",
        )
        self.ultrack_bias_spin = _dspin(
            -10, 10, 0.0, 0.05, 3,
            "Constant offset applied by Ultrack tracking_config.bias.",
        )
        self.ultrack_solver_lbl = QLabel("—")
        add_block_pair_row(g, 0,
            "Power:", compact_spinbox(self.ultrack_power_spin),
            "Solver:", self.ultrack_solver_lbl)
        add_block_pair_row(g, 1,
            "Bias:", compact_spinbox(self.ultrack_bias_spin))
        lay.addLayout(g)

        self.section = CollapsibleSection(
            "Ultrack Parameters",
            params_inner,
            expanded=False,
        )

        self.db_gen_linking_mode_combo.currentTextChanged.connect(
            self._on_db_gen_mode_changed
        )

    def _on_db_gen_mode_changed(self, mode: str) -> None:
        enabled = mode == "shape"
        self.db_gen_area_weight_spin.setEnabled(enabled)
        self.db_gen_iou_weight_spin.setEnabled(enabled)
        self.db_gen_distance_weight_spin.setEnabled(enabled)

    def db_gen_config(self) -> _UltrackConfig:
        """Build a TrackingConfig from the current DB-generation controls."""
        return _UltrackConfig(
            seg_min_area=self.db_gen_min_area_spin.value(),
            seg_max_area=self.db_gen_max_area_spin.value(),
            seg_foreground_threshold=0.0,
            seg_min_frontier=self.db_gen_min_frontier_spin.value(),
            seg_ws_hierarchy=self.db_gen_ws_hierarchy_combo.currentText(),
            seg_n_workers=self.db_gen_n_workers_spin.value(),
            max_distance=self.db_gen_max_dist_spin.value(),
            max_neighbors=self.db_gen_max_neighbors_spin.value(),
            linking_mode=self.db_gen_linking_mode_combo.currentText(),
            area_weight=self.db_gen_area_weight_spin.value(),
            iou_weight=self.db_gen_iou_weight_spin.value(),
            distance_weight=self.db_gen_distance_weight_spin.value(),
            quality_weight=self.db_gen_quality_weight_spin.value(),
            quality_exponent=self.db_gen_quality_exp_spin.value(),
            circularity_weight=self.db_gen_circularity_weight_spin.value(),
            link_n_workers=self.db_gen_n_workers_spin.value(),
        )

    def ultrack_config(self) -> _UltrackConfig:
        """Build a TrackingConfig from all tracking controls (DB-gen + Ultrack solver)."""
        return _UltrackConfig(
            seg_min_area=self.db_gen_min_area_spin.value(),
            seg_max_area=self.db_gen_max_area_spin.value(),
            seg_foreground_threshold=0.0,
            seg_min_frontier=self.db_gen_min_frontier_spin.value(),
            seg_ws_hierarchy=self.db_gen_ws_hierarchy_combo.currentText(),
            seg_n_workers=self.db_gen_n_workers_spin.value(),
            max_distance=self.db_gen_max_dist_spin.value(),
            max_neighbors=self.db_gen_max_neighbors_spin.value(),
            linking_mode=self.db_gen_linking_mode_combo.currentText(),
            area_weight=self.db_gen_area_weight_spin.value(),
            iou_weight=self.db_gen_iou_weight_spin.value(),
            distance_weight=self.db_gen_distance_weight_spin.value(),
            quality_weight=self.db_gen_quality_weight_spin.value(),
            quality_exponent=self.db_gen_quality_exp_spin.value(),
            circularity_weight=self.db_gen_circularity_weight_spin.value(),
            power=self.ultrack_power_spin.value(),
            bias=self.ultrack_bias_spin.value(),
            appear_weight=self.ultrack_appear_spin.value(),
            disappear_weight=self.ultrack_disappear_spin.value(),
            division_weight=self.ultrack_division_spin.value(),
        )
