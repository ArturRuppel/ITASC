"""Nucleus workflow widget for hypothesis generation and tracking in CellFlow v2."""
from __future__ import annotations

import logging
from pathlib import Path

import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class NucleusWorkflowWidget(QWidget):
    """Nucleus hypothesis generation and tracking management."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        # ── 1. Hypothesis Generation ──────────────────────────────────────
        gen_group = QGroupBox("1. Hypothesis Generation")
        gen_lay = QVBoxLayout(gen_group)
        gen_lay.setSpacing(6)

        # Shared Controls (Above Tabs)
        shared_lay = QVBoxLayout()
        
        row_seeds = QHBoxLayout()
        row_seeds.addWidget(QLabel("Seed Source:"))
        self.seed_source_combo = QComboBox()
        self.seed_source_combo.addItems(["Peak local max", "Active Layer", "Disk (Corrected)"])
        row_seeds.addWidget(self.seed_source_combo)
        
        row_seeds.addWidget(QLabel("Seed Dist:"))
        self.seed_dist_spin = QSpinBox()
        self.seed_dist_spin.setRange(1, 500)
        self.seed_dist_spin.setValue(5)
        row_seeds.addWidget(self.seed_dist_spin)
        shared_lay.addLayout(row_seeds)

        self.overwrite_check = QCheckBox("Overwrite existing in DB")
        self.overwrite_check.setChecked(False)
        shared_lay.addWidget(self.overwrite_check)
        gen_lay.addLayout(shared_lay)

        # Tabs for Single vs Sweep
        self.gen_tabs = QTabWidget()
        
        # Tab 1: Single ("Tuning")
        single_tab = QWidget()
        single_lay = QVBoxLayout(single_tab)
        
        def _add_single_param(label, min_val, max_val, default, step, decimals=1):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox() if decimals > 0 else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(default)
            if decimals > 0:
                spin.setDecimals(decimals)
                spin.setSingleStep(step)
            row.addWidget(spin)
            single_lay.addLayout(row)
            return spin

        self.single_thr = _add_single_param("Threshold (%)", 0.0, 100.0, 30.0, 1.0)
        self.single_cmp = _add_single_param("Compactness", 0.0, 1.0, 0.0, 0.01, 2)
        self.single_sigma = _add_single_param("Smooth Sigma", 0.0, 10.0, 0.5, 0.1, 1)

        btn_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.save_db_btn = QPushButton("Save to DB")
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.save_db_btn)
        single_lay.addLayout(btn_row)

        self.use_as_tracked_btn = QPushButton("Use as Tracked")
        self.use_as_tracked_btn.setToolTip("Copy preview to tracked labels for current frame")
        single_lay.addWidget(self.use_as_tracked_btn)
        
        self.gen_tabs.addTab(single_tab, "Tuning (Single)")

        # Tab 2: Parameter Sweep ("Batch")
        sweep_tab = QWidget()
        sweep_lay = QVBoxLayout(sweep_tab)
        
        def _add_sweep_row(label, d_min, d_max, d_step, decimals=1):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            min_s = QDoubleSpinBox() if decimals > 0 else QSpinBox()
            max_s = QDoubleSpinBox() if decimals > 0 else QSpinBox()
            step_s = QDoubleSpinBox() if decimals > 0 else QSpinBox()
            for s in (min_s, max_s, step_s):
                s.setRange(0, 100 if "Threshold" in label else 10)
                if decimals > 0:
                    s.setDecimals(decimals)
            min_s.setValue(d_min)
            max_s.setValue(d_max)
            step_s.setValue(d_step)
            row.addWidget(QLabel("min"))
            row.addWidget(min_s)
            row.addWidget(QLabel("max"))
            row.addWidget(max_s)
            row.addWidget(QLabel("step"))
            row.addWidget(step_s)
            sweep_lay.addLayout(row)
            return min_s, max_s, step_s

        self.sweep_thr = _add_sweep_row("Threshold (%)", 10, 50, 10)
        self.sweep_cmp = _add_sweep_row("Compactness", 0, 0.1, 0.05, 2)
        self.sweep_sigma = _add_sweep_row("Smooth Sigma", 0, 1.0, 0.5, 1)

        sweep_btn_row = QHBoxLayout()
        self.run_sweep_btn = QPushButton("Run Batch Sweep")
        self.run_terminal_btn = QPushButton("Run in Terminal")
        sweep_btn_row.addWidget(self.run_sweep_btn)
        sweep_btn_row.addWidget(self.run_terminal_btn)
        sweep_lay.addLayout(sweep_btn_row)

        self.gen_tabs.addTab(sweep_tab, "Batch (Sweep)")
        gen_lay.addWidget(self.gen_tabs)
        layout.addWidget(gen_group)

        # ── 2. Database Browser ──────────────────────────────────────────
        db_group = QGroupBox("2. Seeding & Browser")
        db_lay = QVBoxLayout(db_group)
        
        row_h = QHBoxLayout()
        row_h.addWidget(QLabel("Hypothesis:"))
        self.hyp_spin = QSpinBox()
        self.hyp_spin.setRange(0, 0)
        row_h.addWidget(self.hyp_spin)
        self.hyp_meta_lbl = QLabel("p000: ---")
        row_h.addWidget(self.hyp_meta_lbl)
        db_lay.addLayout(row_h)
        
        self.set_seed_btn = QPushButton("Set as Tracking Seed")
        db_lay.addWidget(self.set_seed_btn)
        layout.addWidget(db_group)

        # ── 3. Automated Search ──────────────────────────────────────────
        search_group = QGroupBox("3. Automated Search")
        search_lay = QVBoxLayout(search_group)
        
        row_iou = QHBoxLayout()
        row_iou.addWidget(QLabel("IoU Threshold:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0, 1)
        self.iou_spin.setValue(0.5)
        self.iou_spin.setSingleStep(0.1)
        row_iou.addWidget(self.iou_spin)
        search_lay.addLayout(row_iou)

        row_dist = QHBoxLayout()
        row_dist.addWidget(QLabel("Max Dist (µm):"))
        self.dist_spin = QDoubleSpinBox()
        self.dist_spin.setRange(0, 1000)
        self.dist_spin.setValue(20.0)
        row_dist.addWidget(self.dist_spin)
        search_lay.addLayout(row_dist)

        prop_row = QHBoxLayout()
        self.prop_next_btn = QPushButton("Propagate Next")
        self.prop_all_btn = QPushButton("Propagate All")
        self.stop_btn = QPushButton("Stop")
        prop_row.addWidget(self.prop_next_btn)
        prop_row.addWidget(self.prop_all_btn)
        prop_row.addWidget(self.stop_btn)
        search_lay.addLayout(prop_row)
        layout.addWidget(search_group)

        # ── 4. Manual Correction Integration ──────────────────────────────
        corr_group = QGroupBox("4. Manual Correction")
        corr_lay = QVBoxLayout(corr_group)
        self.jump_corr_btn = QPushButton("Correct Current Frame")
        self.jump_corr_btn.setStyleSheet("font-weight: bold; min-height: 28px;")
        corr_lay.addWidget(self.jump_corr_btn)
        layout.addWidget(corr_group)

    def get_state(self) -> dict:
        """Return the current UI state as a dictionary."""
        return {
            "seed_source": self.seed_source_combo.currentText(),
            "seed_dist": self.seed_dist_spin.value(),
            "overwrite": self.overwrite_check.isChecked(),
            "single": {
                "threshold": self.single_thr.value(),
                "compactness": self.single_cmp.value(),
                "sigma": self.single_sigma.value(),
            },
            "sweep": {
                "thr_min": self.sweep_thr[0].value(),
                "thr_max": self.sweep_thr[1].value(),
                "thr_step": self.sweep_thr[2].value(),
                "cmp_min": self.sweep_cmp[0].value(),
                "cmp_max": self.sweep_cmp[1].value(),
                "cmp_step": self.sweep_cmp[2].value(),
                "sigma_min": self.sweep_sigma[0].value(),
                "sigma_max": self.sweep_sigma[1].value(),
                "sigma_step": self.sweep_sigma[2].value(),
            },
            "search": {
                "iou_threshold": self.iou_spin.value(),
                "max_dist_um": self.dist_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        """Update the UI state from a dictionary."""
        if "seed_source" in state:
            self.seed_source_combo.setCurrentText(state["seed_source"])
        if "seed_dist" in state:
            self.seed_dist_spin.setValue(state["seed_dist"])
        if "overwrite" in state:
            self.overwrite_check.setChecked(state["overwrite"])
        
        if "single" in state:
            s = state["single"]
            if "threshold" in s: self.single_thr.setValue(s["threshold"])
            if "compactness" in s: self.single_cmp.setValue(s["compactness"])
            if "sigma" in s: self.single_sigma.setValue(s["sigma"])
        
        if "sweep" in state:
            sw = state["sweep"]
            if "thr_min" in sw: self.sweep_thr[0].setValue(sw["thr_min"])
            if "thr_max" in sw: self.sweep_thr[1].setValue(sw["thr_max"])
            if "thr_step" in sw: self.sweep_thr[2].setValue(sw["thr_step"])
            if "cmp_min" in sw: self.sweep_cmp[0].setValue(sw["cmp_min"])
            if "cmp_max" in sw: self.sweep_cmp[1].setValue(sw["cmp_max"])
            if "cmp_step" in sw: self.sweep_cmp[2].setValue(sw["cmp_step"])
            if "sigma_min" in sw: self.sweep_sigma[0].setValue(sw["sigma_min"])
            if "sigma_max" in sw: self.sweep_sigma[1].setValue(sw["sigma_max"])
            if "sigma_step" in sw: self.sweep_sigma[2].setValue(sw["sigma_step"])

        if "search" in state:
            se = state["search"]
            if "iou_threshold" in se: self.iou_spin.setValue(se["iou_threshold"])
            if "max_dist_um" in se: self.dist_spin.setValue(se["max_dist_um"])
