"""Ultrack tracking and database-generation parameter section for the nucleus workflow widget."""
from __future__ import annotations

import os

from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from cellflow.napari._widget_helpers import (
    dslider as _dslider,
    heading as _heading,
    islider as _islider,
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

        db_inner = QWidget(self)
        db_grid = section_grid()
        db_grid.setContentsMargins(0, 0, 0, 0)
        db_inner.setLayout(db_grid)

        solve_inner = QWidget(self)
        solve_grid = section_grid()
        solve_grid.setContentsMargins(0, 0, 0, 0)
        solve_inner.setLayout(solve_grid)

        # ─── DB Generation — Source Thresholds ───────────────────────
        self.source_contour_threshold_spin = _dslider(
            0, 1, 0.1, 0.01, 3,
            "Contour threshold for the current Ultrack source.",
        )
        self.source_foreground_threshold_spin = _dslider(
            0, 1, 0.1, 0.01, 3,
            "Foreground threshold for the current Ultrack source.",
        )
        self.source_threshold_preview_check = QCheckBox("Preview")
        self.source_threshold_preview_check.setToolTip(
            "Preview the current threshold pair and update when thresholds change."
        )
        self.source_threshold_add_btn = QPushButton("Add")
        self.source_threshold_add_btn.setToolTip("Add the current threshold pair.")
        self.source_threshold_remove_btn = QPushButton("Remove")
        self.source_threshold_remove_btn.setToolTip("Remove the selected pair.")
        self.source_threshold_clear_btn = QPushButton("Clear")
        self.source_threshold_clear_btn.setToolTip("Clear all threshold pairs.")
        self.source_threshold_status_lbl = QLabel("")
        self.source_threshold_status_lbl.setWordWrap(True)

        self.source_threshold_pairs_table = QTableWidget(0, 2)
        self.source_threshold_pairs_table.setHorizontalHeaderLabels(
            ["Contour", "Foreground"]
        )
        self.source_threshold_pairs_table.verticalHeader().hide()
        self.source_threshold_pairs_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.source_threshold_pairs_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.source_threshold_pairs_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.source_threshold_pairs_table.setMaximumHeight(96)

        threshold_button_row = QWidget(self)
        threshold_button_layout = QHBoxLayout(threshold_button_row)
        threshold_button_layout.setContentsMargins(0, 0, 0, 0)
        threshold_button_layout.setSpacing(4)
        threshold_button_layout.addWidget(self.source_threshold_preview_check)
        threshold_button_layout.addWidget(self.source_threshold_add_btn)
        threshold_button_layout.addWidget(self.source_threshold_remove_btn)
        threshold_button_layout.addWidget(self.source_threshold_clear_btn)

        # ─── DB Generation — Candidates ─────────────────────────────
        self.db_gen_min_area_spin = _islider(
            0, 1_000_000, 300, tooltip="Minimum segment area in pixels.")
        self.db_gen_max_area_spin = _islider(
            0, 10_000_000, 100_000, tooltip="Maximum segment area in pixels.")
        self.db_gen_min_frontier_spin = _dslider(
            0, 1, 0.0, 0.01, 2,
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
        self.db_gen_distance_weight_spin = _dslider(0, 10, 0.05, 0.01, 2)
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

        # ─── Ultrack — Track Scope ──────────────────────────────────
        self.ultrack_max_partitions_spin = _islider(
            0, 1000, 30, tooltip="0 = use all partitions.")
        self.ultrack_n_frames_spin = _islider(
            0, 10000, 0, tooltip="0 = process all frames.")

        # ─── Ultrack — Event Penalties ──────────────────────────────
        self.ultrack_appear_spin = _dslider(
            -10, 0, -0.1, 0.05, 2,
            "ILP penalty for cells appearing. More negative = fewer appearances.",
        )
        self.ultrack_disappear_spin = _dslider(
            -10, 0, -0.1, 0.05, 2,
            "ILP penalty for cells disappearing. More negative = fewer disappearances.",
        )
        self.ultrack_division_spin = _dslider(
            -10, 0, -0.01, 0.05, 2,
            "ILP penalty for divisions. More negative = fewer divisions.",
        )

        # ─── Ultrack — Solver ───────────────────────────────────────
        self.ultrack_power_spin = _dslider(
            0.1, 20, 4.0, 0.5, 2,
            "Solver transform for node_prob and link weights (link_function=power).",
        )
        self.ultrack_bias_spin = _dslider(
            -10, 10, 0.0, 0.05, 2,
            "Constant offset applied by Ultrack tracking_config.bias.",
        )
        self.ultrack_solver_lbl = QLabel("—")
        self.solve_use_validated_check = QCheckBox("Use validated corrections")
        self.solve_use_validated_check.setToolTip(
            "Before solving, reapply saved validations/corrections as Ultrack "
            "annotations and preserve them during export."
        )
        # Compatibility alias for saved state and older tests/callers.
        self.db_gen_use_validated_check = self.solve_use_validated_check

        # ─── Pack DB Generation controls ────────────────────────────
        row = 0

        add_section_header(db_grid, row, _heading("Source Thresholds")); row += 1
        add_section_pair_row(
            db_grid, row,
            "Contour:", self.source_contour_threshold_spin,
            "Foreground:", self.source_foreground_threshold_spin,
        ); row += 1
        add_section_full_row(db_grid, row, threshold_button_row); row += 1
        add_section_full_row(db_grid, row, self.source_threshold_pairs_table); row += 1
        add_section_full_row(db_grid, row, self.source_threshold_status_lbl); row += 1

        add_section_header(db_grid, row, _heading("Candidates")); row += 1
        add_section_pair_row(
            db_grid, row,
            "Min area:", self.db_gen_min_area_spin,
            "Max area:", self.db_gen_max_area_spin,
        ); row += 1
        add_section_pair_row(
            db_grid, row,
            "Min frontier:", self.db_gen_min_frontier_spin,
            "WS hierarchy:", self.db_gen_ws_hierarchy_combo,
        ); row += 1
        add_section_pair_row(db_grid, row, "Workers:", self.db_gen_n_workers_spin); row += 1

        add_section_header(db_grid, row, _heading("Linking")); row += 1
        add_section_pair_row(
            db_grid, row,
            "Max distance:", self.db_gen_max_dist_spin,
            "Max neighbors:", self.db_gen_max_neighbors_spin,
        ); row += 1
        add_section_pair_row(
            db_grid, row,
            "Linking mode:", self.db_gen_linking_mode_combo,
            "Area weight:", self.db_gen_area_weight_spin,
        ); row += 1
        add_section_pair_row(
            db_grid, row,
            "IoU weight:", self.db_gen_iou_weight_spin,
            "Distance weight:", self.db_gen_distance_weight_spin,
        ); row += 1

        add_section_header(db_grid, row, _heading("Scoring")); row += 1
        add_section_pair_row(
            db_grid, row,
            "Quality weight:", self.db_gen_quality_weight_spin,
            "Quality exponent:", self.db_gen_quality_exp_spin,
        ); row += 1
        add_section_pair_row(
            db_grid, row,
            "Circularity weight:", self.db_gen_circularity_weight_spin,
        ); row += 1

        # ─── Pack Ultrack solver controls ───────────────────────────
        row = 0

        add_section_header(solve_grid, row, _heading("Track Scope")); row += 1
        add_section_pair_row(
            solve_grid, row,
            "Max partitions:", self.ultrack_max_partitions_spin,
            "N frames:", self.ultrack_n_frames_spin,
        ); row += 1

        add_section_header(solve_grid, row, _heading("Event Penalties")); row += 1
        add_section_pair_row(
            solve_grid, row,
            "Appear:", self.ultrack_appear_spin,
            "Disappear:", self.ultrack_disappear_spin,
        ); row += 1
        add_section_pair_row(solve_grid, row, "Division:", self.ultrack_division_spin); row += 1

        add_section_header(solve_grid, row, _heading("Solver")); row += 1
        add_section_pair_row(
            solve_grid, row,
            "Power:", self.ultrack_power_spin,
            "Bias:", self.ultrack_bias_spin,
        ); row += 1
        add_section_pair_row(solve_grid, row, "Solver:", self.ultrack_solver_lbl); row += 1

        add_section_header(solve_grid, row, _heading("Validated Corrections")); row += 1
        add_section_full_row(solve_grid, row, self.solve_use_validated_check); row += 1

        self.db_section = CollapsibleSection(
            "Database Generation Parameters",
            db_inner,
            expanded=False,
        )
        self.solve_section = CollapsibleSection(
            "Ultrack Solver Parameters",
            solve_inner,
            expanded=False,
        )

        self.db_gen_linking_mode_combo.currentTextChanged.connect(
            self._on_db_gen_mode_changed
        )
        self.source_threshold_add_btn.clicked.connect(self.add_threshold_pair)
        self.source_threshold_remove_btn.clicked.connect(
            self.remove_selected_threshold_pair
        )
        self.source_threshold_clear_btn.clicked.connect(self.clear_threshold_pairs)

    def _on_db_gen_mode_changed(self, mode: str) -> None:
        enabled = mode == "shape"
        self.db_gen_area_weight_spin.setEnabled(enabled)
        self.db_gen_iou_weight_spin.setEnabled(enabled)
        self.db_gen_distance_weight_spin.setEnabled(enabled)

    def current_threshold_pair(self) -> dict[str, float]:
        return {
            "contour_threshold": float(self.source_contour_threshold_spin.value()),
            "foreground_threshold": float(
                self.source_foreground_threshold_spin.value()
            ),
        }

    def threshold_pairs(self) -> list[dict[str, float]]:
        pairs: list[dict[str, float]] = []
        for row in range(self.source_threshold_pairs_table.rowCount()):
            contour_item = self.source_threshold_pairs_table.item(row, 0)
            foreground_item = self.source_threshold_pairs_table.item(row, 1)
            if contour_item is None or foreground_item is None:
                continue
            pairs.append(
                {
                    "contour_threshold": float(contour_item.data(256)),
                    "foreground_threshold": float(foreground_item.data(256)),
                }
            )
        return pairs

    def set_threshold_pairs(self, pairs: list[dict[str, float]]) -> None:
        self.source_threshold_pairs_table.setRowCount(0)
        for pair in pairs:
            self._append_threshold_pair(
                float(pair["contour_threshold"]),
                float(pair["foreground_threshold"]),
            )
        self._set_threshold_status("")

    def add_threshold_pair(self) -> bool:
        pair = self.current_threshold_pair()
        for existing in self.threshold_pairs():
            if existing == pair:
                self._set_threshold_status("Threshold pair already added.")
                return False
        self._append_threshold_pair(
            pair["contour_threshold"],
            pair["foreground_threshold"],
        )
        self._set_threshold_status("")
        return True

    def remove_selected_threshold_pair(self) -> bool:
        selected_rows = self.source_threshold_pairs_table.selectionModel().selectedRows()
        if not selected_rows:
            self._set_threshold_status("Select a threshold pair to remove.")
            return False
        self.source_threshold_pairs_table.removeRow(selected_rows[0].row())
        self._set_threshold_status("")
        return True

    def clear_threshold_pairs(self) -> None:
        self.source_threshold_pairs_table.setRowCount(0)
        self._set_threshold_status("")

    def _append_threshold_pair(
        self,
        contour_threshold: float,
        foreground_threshold: float,
    ) -> None:
        row = self.source_threshold_pairs_table.rowCount()
        self.source_threshold_pairs_table.insertRow(row)
        for col, value in enumerate((contour_threshold, foreground_threshold)):
            item = QTableWidgetItem(f"{value:.3f}")
            item.setData(256, float(value))
            self.source_threshold_pairs_table.setItem(row, col, item)

    def _set_threshold_status(self, text: str) -> None:
        self.source_threshold_status_lbl.setText(text)
        self.source_threshold_status_lbl.setVisible(bool(text))

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
