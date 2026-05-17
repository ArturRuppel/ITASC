"""Ultrack tracking and database-generation parameter section for the nucleus workflow widget."""
from __future__ import annotations

import os

from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QWidget

from cellflow.napari._widget_helpers import (
    dslider as _dslider,
    heading as _heading,
    islider as _islider,
    separator as _separator,
)
from cellflow.napari.ui_style import (
    add_section_full_row,
    add_section_header,
    add_section_pair_row,
    section_grid,
)
from cellflow.napari.widgets import CollapsibleSection
from cellflow.tracking_ultrack.config import TrackingConfig as _UltrackConfig


class NucleusTrackingInputsWidget(QWidget):
    """Qt controls for Ultrack tracking and database-generation parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        params_inner = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(0, 0, 0, 0)
        params_inner.setLayout(grid)

        # ─── DB Generation — Candidates ─────────────────────────────
        self.db_gen_min_area_spin = _islider(
            0, 1_000_000, 300, tooltip="Minimum segment area in pixels.")
        self.db_gen_max_area_spin = _islider(
            0, 10_000_000, 100_000, tooltip="Maximum segment area in pixels.")
        self.db_gen_min_frontier_spin = _dslider(
            0, 1, 0.0, 0.01, 3,
            "Minimum boundary fraction to keep a candidate.",
        )
        self.db_gen_ws_hierarchy_combo = QComboBox()
        self.db_gen_ws_hierarchy_combo.addItems(["area", "dynamics", "volume"])
        self.db_gen_n_workers_spin = _islider(
            1, max(1, os.cpu_count() or 1), 1,
            tooltip="Parallel workers for segmentation.",
        )

        # ─── DB Generation — Linking ────────────────────────────────
        self.db_gen_max_dist_spin = _dslider(0, 500, 15.0, 1.0, 1)
        self.db_gen_max_neighbors_spin = _islider(1, 50, 5)
        self.db_gen_linking_mode_combo = QComboBox()
        self.db_gen_linking_mode_combo.addItems(["default", "shape"])
        self.db_gen_area_weight_spin = _dslider(0, 10, 1.0, 0.1, 2)
        self.db_gen_area_weight_spin.setEnabled(False)
        self.db_gen_iou_weight_spin = _dslider(0, 10, 1.0, 0.1, 2)
        self.db_gen_iou_weight_spin.setEnabled(False)
        self.db_gen_distance_weight_spin = _dslider(0, 10, 0.05, 0.01, 3)
        self.db_gen_distance_weight_spin.setEnabled(False)

        # ─── DB Generation — Scoring ────────────────────────────────
        self.db_gen_quality_weight_spin = _dslider(
            0, 10, 1.0, 0.05, 2,
            "Weight applied to signal-based segmentation quality.",
        )
        self.db_gen_quality_exp_spin = _dslider(
            0.1, 50, 8.0, 0.5, 2,
            "Raises signal-based quality before storing as node_prob.",
        )
        self.db_gen_circularity_weight_spin = _dslider(
            0, 10, 0.25, 0.05, 2,
            "Weight applied to shape circularity.",
        )

        # ─── DB Generation — Validated Seed Prior ───────────────────
        self.db_gen_use_validated_check = QCheckBox("Use validated corrections")

        # ─── Ultrack — Track Scope ──────────────────────────────────
        self.ultrack_max_partitions_spin = _islider(
            0, 1000, 30, tooltip="0 = use all partitions.")
        self.ultrack_n_frames_spin = _islider(
            0, 10000, 0, tooltip="0 = process all frames.")

        # ─── Ultrack — Event Penalties ──────────────────────────────
        self.ultrack_appear_spin = _dslider(
            -10, 0, -0.1, 0.05, 3,
            "ILP penalty for cells appearing. More negative = fewer appearances.",
        )
        self.ultrack_disappear_spin = _dslider(
            -10, 0, -0.1, 0.05, 3,
            "ILP penalty for cells disappearing. More negative = fewer disappearances.",
        )
        self.ultrack_division_spin = _dslider(
            -10, 0, -0.001, 0.05, 3,
            "ILP penalty for divisions. More negative = fewer divisions.",
        )

        # ─── Ultrack — Solver ───────────────────────────────────────
        self.ultrack_power_spin = _dslider(
            0.1, 20, 4.0, 0.5, 2,
            "Solver transform for node_prob and link weights (link_function=power).",
        )
        self.ultrack_bias_spin = _dslider(
            -10, 10, 0.0, 0.05, 3,
            "Constant offset applied by Ultrack tracking_config.bias.",
        )
        self.ultrack_solver_lbl = QLabel("—")

        # ─── Pack everything into the unified grid ──────────────────
        row = 0

        add_section_header(grid, row, _heading("DB Generation — Candidates")); row += 1
        add_section_pair_row(
            grid, row,
            "Min area:", self.db_gen_min_area_spin,
            "Max area:", self.db_gen_max_area_spin,
        ); row += 1
        add_section_pair_row(
            grid, row,
            "Min frontier:", self.db_gen_min_frontier_spin,
            "WS hierarchy:", self.db_gen_ws_hierarchy_combo,
        ); row += 1
        add_section_pair_row(grid, row, "Workers:", self.db_gen_n_workers_spin); row += 1

        add_section_header(grid, row, _heading("DB Generation — Linking")); row += 1
        add_section_pair_row(
            grid, row,
            "Max dist:", self.db_gen_max_dist_spin,
            "Max neighbors:", self.db_gen_max_neighbors_spin,
        ); row += 1
        add_section_pair_row(
            grid, row,
            "Linking mode:", self.db_gen_linking_mode_combo,
            "Area wt:", self.db_gen_area_weight_spin,
        ); row += 1
        add_section_pair_row(
            grid, row,
            "IoU wt:", self.db_gen_iou_weight_spin,
            "Dist wt:", self.db_gen_distance_weight_spin,
        ); row += 1

        add_section_header(grid, row, _heading("DB Generation — Scoring")); row += 1
        add_section_pair_row(
            grid, row,
            "Quality wt:", self.db_gen_quality_weight_spin,
            "Quality exp:", self.db_gen_quality_exp_spin,
        ); row += 1
        add_section_pair_row(
            grid, row,
            "Circularity wt:", self.db_gen_circularity_weight_spin,
        ); row += 1

        add_section_header(grid, row, _heading("DB Generation — Validated Seed Prior")); row += 1
        add_section_full_row(grid, row, self.db_gen_use_validated_check); row += 1

        add_section_full_row(grid, row, _separator()); row += 1

        add_section_header(grid, row, _heading("Ultrack — Track Scope")); row += 1
        add_section_pair_row(
            grid, row,
            "Max partitions:", self.ultrack_max_partitions_spin,
            "N frames:", self.ultrack_n_frames_spin,
        ); row += 1

        add_section_header(grid, row, _heading("Ultrack — Event Penalties")); row += 1
        add_section_pair_row(
            grid, row,
            "Appear:", self.ultrack_appear_spin,
            "Disappear:", self.ultrack_disappear_spin,
        ); row += 1
        add_section_pair_row(grid, row, "Division:", self.ultrack_division_spin); row += 1

        add_section_header(grid, row, _heading("Ultrack — Solver")); row += 1
        add_section_pair_row(
            grid, row,
            "Power:", self.ultrack_power_spin,
            "Bias:", self.ultrack_bias_spin,
        ); row += 1
        add_section_pair_row(grid, row, "Solver:", self.ultrack_solver_lbl); row += 1

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
