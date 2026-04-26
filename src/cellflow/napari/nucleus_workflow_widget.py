"""Nucleus workflow widget for hypothesis generation and tracking in CellFlow v2."""
from __future__ import annotations

import logging
import shlex
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cellflow.database.hypotheses import (
    ContourWatershedSweepSpec,
    HypothesisRecord,
    build_contour_watershed_parameter_sets,
    delete_hypothesis_parameter,
    iter_contour_watershed_records,
    list_hypotheses,
    read_full_hypothesis_stack,
    read_hypothesis_labels,
    write_hypothesis_sweep_h5,
)
from cellflow.database.tracked import (
    read_full_tracked_stack,
    read_tracked_frame,
    tracked_frame_exists,
    write_tracked_frame,
)
from cellflow.database.validation import (
    invalidate_frame,
    is_validated,
    read_validated_frames,
    validate_frame,
)
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.segmentation import ContourWatershedParams, compute_contour_watershed
from cellflow.tracking import propagate_one_frame
from cellflow.tracking.retracker import retrack_frame

logger = logging.getLogger(__name__)

_PREVIEW_LAYER = "Preview: Nucleus"
_HYP_LAYER = "Hypothesis: Nucleus"
_TRACKED_LAYER = "Tracked: Nucleus"
_CONTOUR_LAYER = "Contour Map: Nucleus"
_CELL_ZAVG_LAYER = "Cell z-avg"
_NUC_ZAVG_LAYER = "Nucleus z-avg"


class NucleusWorkflowWidget(QWidget):
    """Nucleus hypothesis generation and tracking management."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._stop_flag: bool = False
        self._build_worker = None
        self._sweep_worker = None
        self._current_db_p: int | None = None
        self._db_param_map: dict[tuple[int, float], int] = {}
        self._db_seed_dist_vals: list[int] = []
        self._db_fg_thr_vals: list[float] = []
        self._setup_ui()
        self._connect_signals()

    # ──────────────────────────────────────────────────────────────────────────
    # UI setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        # ── Inputs ────────────────────────────────────────────────────────
        self.input_files = PipelineFilesWidget([
            ("Inputs", [
                ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
                ("1_cellpose/nucleus_dp_3dt.tif",  "Nucleus dp 3D+t"),
            ]),
        ])
        layout.addWidget(self.input_files)

        # ── 1. Contour Maps ───────────────────────────────────────────────
        _contour_inner = QWidget()
        contour_lay = QVBoxLayout(_contour_inner)
        contour_lay.setContentsMargins(4, 4, 4, 4)
        contour_lay.setSpacing(4)

        cp_params_lay = contour_lay

        cp_row = QHBoxLayout()
        cp_row.addWidget(QLabel("Cellprob:"))
        cp_row.addWidget(QLabel("min"))
        self.cp_min_spin = QDoubleSpinBox()
        self.cp_min_spin.setRange(-20.0, 20.0)
        self.cp_min_spin.setValue(-3.0)
        self.cp_min_spin.setDecimals(1)
        self.cp_min_spin.setSingleStep(1.0)
        cp_row.addWidget(self.cp_min_spin)
        cp_row.addWidget(QLabel("max"))
        self.cp_max_spin = QDoubleSpinBox()
        self.cp_max_spin.setRange(-20.0, 20.0)
        self.cp_max_spin.setValue(0.0)
        self.cp_max_spin.setDecimals(1)
        self.cp_max_spin.setSingleStep(1.0)
        cp_row.addWidget(self.cp_max_spin)
        cp_row.addWidget(QLabel("step"))
        self.cp_step_spin = QDoubleSpinBox()
        self.cp_step_spin.setRange(0.1, 10.0)
        self.cp_step_spin.setValue(1.0)
        self.cp_step_spin.setDecimals(1)
        self.cp_step_spin.setSingleStep(0.5)
        cp_row.addWidget(self.cp_step_spin)
        cp_params_lay.addLayout(cp_row)

        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Gamma:"))
        gamma_row.addWidget(QLabel("min"))
        self.cp_gamma_min_spin = QDoubleSpinBox()
        self.cp_gamma_min_spin.setRange(0.05, 5.0)
        self.cp_gamma_min_spin.setValue(1.0)
        self.cp_gamma_min_spin.setDecimals(2)
        self.cp_gamma_min_spin.setSingleStep(0.05)
        gamma_row.addWidget(self.cp_gamma_min_spin)
        gamma_row.addWidget(QLabel("max"))
        self.cp_gamma_max_spin = QDoubleSpinBox()
        self.cp_gamma_max_spin.setRange(0.05, 5.0)
        self.cp_gamma_max_spin.setValue(1.0)
        self.cp_gamma_max_spin.setDecimals(2)
        self.cp_gamma_max_spin.setSingleStep(0.05)
        gamma_row.addWidget(self.cp_gamma_max_spin)
        gamma_row.addWidget(QLabel("step"))
        self.cp_gamma_step_spin = QDoubleSpinBox()
        self.cp_gamma_step_spin.setRange(0.05, 2.0)
        self.cp_gamma_step_spin.setValue(0.25)
        self.cp_gamma_step_spin.setDecimals(2)
        self.cp_gamma_step_spin.setSingleStep(0.05)
        _gamma_tip = (
            "Gamma correction on Cellpose probability logits before boundary building. "
            "<1 boosts dim signals; >1 suppresses them. 1.0 = no correction. "
            "Contour maps are averaged over all gamma values in [min, max]."
        )
        for _w in (self.cp_gamma_min_spin, self.cp_gamma_max_spin, self.cp_gamma_step_spin):
            _w.setToolTip(_gamma_tip)
        gamma_row.addWidget(self.cp_gamma_step_spin)
        cp_params_lay.addLayout(gamma_row)

        build_btn_row = QHBoxLayout()
        self.build_btn = QPushButton("Build")
        self.preview_contour_btn = QPushButton("Preview")
        self.preview_contour_btn.setToolTip(
            "Build contour maps for the current frame only and display in napari"
        )
        self.cancel_build_btn = QPushButton("Cancel")
        self.cancel_build_btn.setEnabled(False)
        build_btn_row.addWidget(self.build_btn)
        build_btn_row.addWidget(self.preview_contour_btn)
        build_btn_row.addWidget(self.cancel_build_btn)
        contour_lay.addLayout(build_btn_row)

        self.build_progress_bar = QProgressBar()
        self.build_progress_bar.setRange(0, 100)
        self.build_progress_bar.setValue(0)
        self.build_progress_bar.setVisible(False)
        contour_lay.addWidget(self.build_progress_bar)

        self.contour_files = PipelineFilesWidget([
            ("", [
                ("2_nucleus/contour_maps.tif",   "Contour maps"),
                ("2_nucleus/foreground_maps.tif", "Foreground maps"),
            ]),
        ])
        contour_lay.addWidget(self.contour_files)
        self.contour_section = CollapsibleSection(
            "1. Contour Maps", _contour_inner, expanded=False
        )
        layout.addWidget(self.contour_section)

        # ── 2. Hypothesis Generation ──────────────────────────────────────
        gen_group = QGroupBox("2. Hypothesis Generation")
        gen_lay = QVBoxLayout(gen_group)
        gen_lay.setSpacing(6)

        shared_row = QHBoxLayout()
        shared_row.addWidget(QLabel("Min Cell Size (px):"))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 100000)
        self.min_size_spin.setValue(0)
        self.min_size_spin.setToolTip("Remove regions smaller than this many pixels (0 = keep all)")
        shared_row.addWidget(self.min_size_spin)
        shared_row.addWidget(QLabel("Min Circularity:"))
        self.min_circularity_spin = QDoubleSpinBox()
        self.min_circularity_spin.setRange(0.0, 1.0)
        self.min_circularity_spin.setValue(0.0)
        self.min_circularity_spin.setDecimals(2)
        self.min_circularity_spin.setSingleStep(0.05)
        self.min_circularity_spin.setToolTip(
            "Remove regions with circularity (4π·area/perimeter²) below this value (0 = keep all, 1 = perfect circle)"
        )
        shared_row.addWidget(self.min_circularity_spin)
        shared_row.addStretch()
        self.overwrite_check = QCheckBox("Overwrite existing")
        shared_row.addWidget(self.overwrite_check)
        gen_lay.addLayout(shared_row)

        self.gen_tabs = QTabWidget()

        # Tab: Tuning
        tuning_tab = QWidget()
        tuning_lay = QVBoxLayout(tuning_tab)

        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel("Seed Distance:"))
        self.single_seed_dist = QSpinBox()
        self.single_seed_dist.setRange(1, 500)
        self.single_seed_dist.setValue(10)
        dist_row.addWidget(self.single_seed_dist)
        tuning_lay.addLayout(dist_row)

        fg_row = QHBoxLayout()
        fg_row.addWidget(QLabel("Foreground Threshold:"))
        self.single_fg_threshold = QDoubleSpinBox()
        self.single_fg_threshold.setRange(0.01, 0.99)
        self.single_fg_threshold.setValue(0.5)
        self.single_fg_threshold.setDecimals(2)
        self.single_fg_threshold.setSingleStep(0.05)
        self.single_fg_threshold.setToolTip(
            "Sigmoid foreground probability cutoff — pixels below this are excluded from segmentation and seeding."
        )
        fg_row.addWidget(self.single_fg_threshold)
        tuning_lay.addLayout(fg_row)

        noise_row = QHBoxLayout()
        noise_row.addWidget(QLabel("Noise Scale:"))
        self.single_noise_scale = QDoubleSpinBox()
        self.single_noise_scale.setRange(0.0, 1.0)
        self.single_noise_scale.setValue(0.0)
        self.single_noise_scale.setDecimals(2)
        self.single_noise_scale.setSingleStep(0.01)
        self.single_noise_scale.setToolTip("Stochastic perturbation level for segmentation diversity.")
        noise_row.addWidget(self.single_noise_scale)

        noise_row.addWidget(QLabel("Blur Sigma:"))
        self.single_noise_blur = QDoubleSpinBox()
        self.single_noise_blur.setRange(0.0, 10.0)
        self.single_noise_blur.setValue(0.0)
        self.single_noise_blur.setDecimals(1)
        self.single_noise_blur.setSingleStep(0.5)
        self.single_noise_blur.setToolTip("Sigma for correlating noise (higher = larger structures).")
        noise_row.addWidget(self.single_noise_blur)
        tuning_lay.addLayout(noise_row)

        tuning_btn_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.save_db_btn = QPushButton("Save to DB")
        tuning_btn_row.addWidget(self.preview_btn)
        tuning_btn_row.addWidget(self.save_db_btn)
        tuning_lay.addLayout(tuning_btn_row)
        self.gen_tabs.addTab(tuning_tab, "Tuning")

        # Tab: Sweep
        sweep_tab = QWidget()
        sweep_lay = QVBoxLayout(sweep_tab)

        def _sweep_row(label, d_min, d_max, d_step, decimals=0):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            make = QDoubleSpinBox if decimals > 0 else QSpinBox
            min_s, max_s, step_s = make(), make(), make()
            for s in (min_s, max_s, step_s):
                s.setRange(1 if decimals == 0 else 0.0, 500 if decimals == 0 else 20.0)
                if decimals > 0:
                    s.setDecimals(decimals)
            min_s.setValue(d_min)
            max_s.setValue(d_max)
            step_s.setValue(d_step)
            row.addWidget(QLabel("min")); row.addWidget(min_s)
            row.addWidget(QLabel("max")); row.addWidget(max_s)
            row.addWidget(QLabel("step")); row.addWidget(step_s)
            sweep_lay.addLayout(row)
            return min_s, max_s, step_s

        self.sweep_seed_dist  = _sweep_row("Seed Dist",            8,    14,   2,     decimals=0)
        self.sweep_fg_thr     = _sweep_row("Foreground Threshold", 0.4,  0.6,  0.05,  decimals=2)

        sweep_noise_row = QHBoxLayout()
        sweep_noise_row.addWidget(QLabel("Noise Scale:"))
        self.sweep_noise_scale = QDoubleSpinBox()
        self.sweep_noise_scale.setRange(0.0, 1.0)
        self.sweep_noise_scale.setValue(0.0)
        self.sweep_noise_scale.setDecimals(2)
        self.sweep_noise_scale.setSingleStep(0.01)
        self.sweep_noise_scale.setToolTip("Stochastic perturbation level for segmentation diversity.")
        sweep_noise_row.addWidget(self.sweep_noise_scale)
        sweep_noise_row.addWidget(QLabel("Blur Sigma:"))
        self.sweep_noise_blur = QDoubleSpinBox()
        self.sweep_noise_blur.setRange(0.0, 10.0)
        self.sweep_noise_blur.setValue(0.0)
        self.sweep_noise_blur.setDecimals(1)
        self.sweep_noise_blur.setSingleStep(0.5)
        self.sweep_noise_blur.setToolTip("Sigma for correlating noise (higher = larger structures).")
        sweep_noise_row.addWidget(self.sweep_noise_blur)
        sweep_lay.addLayout(sweep_noise_row)

        sweep_runs_row = QHBoxLayout()
        sweep_runs_row.addWidget(QLabel("Runs:"))
        self.sweep_n_runs = QSpinBox()
        self.sweep_n_runs.setRange(1, 100)
        self.sweep_n_runs.setValue(1)
        self.sweep_n_runs.setToolTip(
            "How many times to run the sweep. With noise > 0 each run produces "
            "different stochastic hypotheses stored as separate parameter sets."
        )
        sweep_runs_row.addWidget(self.sweep_n_runs)
        sweep_runs_row.addStretch()
        sweep_lay.addLayout(sweep_runs_row)

        sweep_btn_row = QHBoxLayout()
        self.run_sweep_btn    = QPushButton("Run Sweep")
        self.run_terminal_btn = QPushButton("Run in Terminal")
        self.cancel_sweep_btn = QPushButton("Cancel")
        self.cancel_sweep_btn.setEnabled(False)
        sweep_btn_row.addWidget(self.run_sweep_btn)
        sweep_btn_row.addWidget(self.run_terminal_btn)
        sweep_btn_row.addWidget(self.cancel_sweep_btn)
        sweep_lay.addLayout(sweep_btn_row)
        self.gen_tabs.addTab(sweep_tab, "Sweep")

        gen_lay.addWidget(self.gen_tabs)
        layout.addWidget(gen_group)

        # ── 3. Database Browser ──────────────────────────────────────────
        db_group = QGroupBox("3. Database Browser")
        db_lay = QVBoxLayout(db_group)

        hdr_row = QHBoxLayout()
        self.db_activate_btn = QPushButton("Activate")
        self.db_activate_btn.setCheckable(True)
        self.db_activate_btn.setChecked(False)
        self.db_activate_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        hdr_row.addWidget(self.db_activate_btn)
        hdr_row.addStretch()
        self.db_refresh_btn = QPushButton()
        self.db_refresh_btn.setToolTip("Refresh database browser")
        self.db_refresh_btn.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_BrowserReload)
        )
        self.db_refresh_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        hdr_row.addWidget(self.db_refresh_btn)
        db_lay.addLayout(hdr_row)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Seed Dist:"))
        self.db_seed_dist_spin = QSpinBox()
        self.db_seed_dist_spin.setRange(1, 500)
        self.db_seed_dist_spin.setValue(10)
        self.db_seed_dist_spin.setEnabled(False)
        param_row.addWidget(self.db_seed_dist_spin)
        param_row.addWidget(QLabel("FG Thr:"))
        self.db_fg_thr_spin = QDoubleSpinBox()
        self.db_fg_thr_spin.setRange(0.01, 0.99)
        self.db_fg_thr_spin.setValue(0.5)
        self.db_fg_thr_spin.setDecimals(2)
        self.db_fg_thr_spin.setSingleStep(0.05)
        self.db_fg_thr_spin.setEnabled(False)
        param_row.addWidget(self.db_fg_thr_spin)
        db_lay.addLayout(param_row)

        self.db_info_lbl = QLabel("—")
        self.db_info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        db_lay.addWidget(self.db_info_lbl)

        db_btn_row = QHBoxLayout()
        self.set_seed_btn = QPushButton("Set as Tracking Seed")
        db_btn_row.addWidget(self.set_seed_btn)
        db_lay.addLayout(db_btn_row)

        db_del_row = QHBoxLayout()
        self.del_stack_btn = QPushButton("Remove Stack")
        self.del_stack_btn.setStyleSheet(
            "QPushButton { color: #cc3333; }"
            "QPushButton:hover { background-color: #4a1111; color: white; }"
        )
        db_del_row.addWidget(self.del_stack_btn)
        db_lay.addLayout(db_del_row)
        layout.addWidget(db_group)

        # ── 4. Automated Search ──────────────────────────────────────────
        search_group = QGroupBox("4. Automated Search")
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

        row_vel = QHBoxLayout()
        row_vel.addWidget(QLabel("Velocity σ (px):"))
        self.vel_sigma_spin = QDoubleSpinBox()
        self.vel_sigma_spin.setRange(1, 500)
        self.vel_sigma_spin.setValue(25.0)
        self.vel_sigma_spin.setSingleStep(5.0)
        row_vel.addWidget(self.vel_sigma_spin)
        search_lay.addLayout(row_vel)

        def _weight_spin(default):
            w = QDoubleSpinBox()
            w.setRange(0.0, 10.0)
            w.setValue(default)
            w.setSingleStep(0.5)
            w.setDecimals(1)
            return w

        row_iou_w = QHBoxLayout()
        row_iou_w.addWidget(QLabel("IoU Weight:"))
        self.iou_weight_spin = _weight_spin(1.0)
        row_iou_w.addWidget(self.iou_weight_spin)
        search_lay.addLayout(row_iou_w)

        row_area_w = QHBoxLayout()
        row_area_w.addWidget(QLabel("Area Weight:"))
        self.area_weight_spin = _weight_spin(1.0)
        row_area_w.addWidget(self.area_weight_spin)
        search_lay.addLayout(row_area_w)

        row_vel_w = QHBoxLayout()
        row_vel_w.addWidget(QLabel("Velocity Weight:"))
        self.vel_weight_spin = _weight_spin(1.0)
        row_vel_w.addWidget(self.vel_weight_spin)
        search_lay.addLayout(row_vel_w)

        row_pos_w = QHBoxLayout()
        row_pos_w.addWidget(QLabel("Position Weight:"))
        self.pos_weight_spin = _weight_spin(1.0)
        row_pos_w.addWidget(self.pos_weight_spin)
        search_lay.addLayout(row_pos_w)

        row_unmatched = QHBoxLayout()
        row_unmatched.addWidget(QLabel("Unmatched Score:"))
        self.unmatched_spin = QDoubleSpinBox()
        self.unmatched_spin.setRange(0.0, 1.0)
        self.unmatched_spin.setValue(0.1)
        self.unmatched_spin.setSingleStep(0.01)
        row_unmatched.addWidget(self.unmatched_spin)
        search_lay.addLayout(row_unmatched)

        prop_row = QHBoxLayout()
        self.prop_next_btn = QPushButton("Propagate Next")
        self.prop_all_btn  = QPushButton("Propagate All")
        self.stop_btn      = QPushButton("Stop")
        prop_row.addWidget(self.prop_next_btn)
        prop_row.addWidget(self.prop_all_btn)
        prop_row.addWidget(self.stop_btn)
        search_lay.addLayout(prop_row)

        self.load_tracked_btn = QPushButton("Load Tracked Labels")
        search_lay.addWidget(self.load_tracked_btn)
        layout.addWidget(search_group)

        # ── 5. Manual Correction ──────────────────────────────────────────
        corr_group = QGroupBox("5. Manual Correction")
        corr_lay = QVBoxLayout(corr_group)
        self.jump_corr_btn = QPushButton("Correct Current Frame")
        self.jump_corr_btn.setStyleSheet("font-weight: bold; min-height: 28px;")
        corr_lay.addWidget(self.jump_corr_btn)

        retrack_row = QHBoxLayout()
        self.retrack_btn = QPushButton("Retrack Frame")
        retrack_row.addWidget(self.retrack_btn)
        self.validate_btn = QPushButton("Validate Frame")
        self.validate_btn.setCheckable(True)
        retrack_row.addWidget(self.validate_btn)
        corr_lay.addLayout(retrack_row)
        layout.addWidget(corr_group)

        # ── Status label ──────────────────────────────────────────────────
        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        layout.addWidget(self.status_lbl)

        # ── Outputs ───────────────────────────────────────────────────────
        self.output_files = PipelineFilesWidget([
            ("Outputs", [
                ("2_nucleus/hypotheses.h5", "Hypotheses DB"),
            ]),
        ])
        layout.addWidget(self.output_files)

    # ──────────────────────────────────────────────────────────────────────────
    # Signal wiring
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.build_btn.clicked.connect(self._on_build_contour_maps)
        self.preview_contour_btn.clicked.connect(self._on_preview_contour_maps)
        self.cancel_build_btn.clicked.connect(self._on_cancel_build)
        self.preview_btn.clicked.connect(self._on_preview)
        self.save_db_btn.clicked.connect(self._on_save_db)
        self.run_sweep_btn.clicked.connect(self._on_run_sweep)
        self.run_terminal_btn.clicked.connect(self._on_run_terminal)
        self.cancel_sweep_btn.clicked.connect(self._on_cancel_sweep)
        self.db_seed_dist_spin.valueChanged.connect(self._on_db_param_changed)
        self.db_fg_thr_spin.valueChanged.connect(self._on_db_param_changed)
        self.set_seed_btn.clicked.connect(self._on_set_seed)
        self.db_activate_btn.toggled.connect(self._on_db_activate_toggled)
        self.db_refresh_btn.clicked.connect(lambda: self._refresh_db_browser())
        self.del_stack_btn.clicked.connect(self._on_remove_stack)
        self.prop_next_btn.clicked.connect(self._on_propagate_next)
        self.prop_all_btn.clicked.connect(self._on_propagate_all)
        self.stop_btn.clicked.connect(lambda: setattr(self, "_stop_flag", True))
        self.load_tracked_btn.clicked.connect(self._on_load_tracked)
        self.jump_corr_btn.clicked.connect(self._on_jump_correction)
        self.retrack_btn.clicked.connect(self._on_retrack_frame)
        self.validate_btn.toggled.connect(self._on_validate_toggled)
        self.viewer.dims.events.current_step.connect(self._on_dims_step_changed)

    # ──────────────────────────────────────────────────────────────────────────
    # Public refresh
    # ──────────────────────────────────────────────────────────────────────────

    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self.input_files.refresh(pos_dir)
        self.contour_files.refresh(pos_dir)
        self.output_files.refresh(pos_dir)
        if pos_dir is None:
            return
        self._refresh_db_browser()
        self._refresh_validate_btn()

    # ──────────────────────────────────────────────────────────────────────────
    # Path helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _hyp_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "hypotheses.h5" if self._pos_dir else None

    def _tracked_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    def _prob_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif" if self._pos_dir else None

    def _dp_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif" if self._pos_dir else None

    def _contour_maps_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "contour_maps.tif" if self._pos_dir else None

    def _foreground_maps_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "foreground_maps.tif" if self._pos_dir else None

    def _cell_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "cell_zavg.tif" if self._pos_dir else None

    def _nucleus_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "nucleus_zavg.tif" if self._pos_dir else None

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _current_t(self) -> int:
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) >= 1 else 0

    def _get_nz(self) -> int:
        for layer in self.viewer.layers:
            if hasattr(layer, "data") and layer.data.ndim == 4:
                return layer.data.shape[1]
        hyp_path = self._hyp_path()
        if hyp_path and hyp_path.exists():
            try:
                return read_hypothesis_labels(hyp_path, 0, 0).shape[0]
            except Exception:
                pass
        return 1

    def _update_tracked_display(
        self,
        labels: np.ndarray,
        t: int | None = None,
        tracked_path: Path | None = None,
    ) -> None:
        if _TRACKED_LAYER in self.viewer.layers and t is not None:
            layer = self.viewer.layers[_TRACKED_LAYER]
            if layer.data.ndim == 3:
                if t < layer.data.shape[0]:
                    new_data = layer.data.copy()
                    new_data[t] = labels
                    layer.data = new_data
                    return
                if tracked_path is not None and tracked_path.exists():
                    layer.data = read_full_tracked_stack(tracked_path)
                    return
        display = labels[np.newaxis].copy() if labels.ndim == 2 else labels
        self._update_layer(_TRACKED_LAYER, display)

    def _update_layer(self, name: str, data: np.ndarray) -> None:
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            self.viewer.add_labels(data, name=name)

    def _refresh_db_browser(self) -> None:
        self._db_param_map = {}
        self._db_seed_dist_vals = []
        self._db_fg_thr_vals = []
        self._current_db_p = None
        self.db_seed_dist_spin.setEnabled(False)
        self.db_fg_thr_spin.setEnabled(False)
        self.db_info_lbl.setText("—")

        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            self.status_lbl.setText("Hypothesis DB: not found.")
            return
        try:
            n_p, params_by_p = list_hypotheses(hyp_path)
        except Exception as e:
            logger.warning("Could not read hypotheses.h5: %s", e)
            self.status_lbl.setText(f"Hypothesis DB: read error — {e}")
            return

        contour_entries = {
            p: info for p, info in params_by_p.items()
            if str(info.get("method", "")) == "contour_watershed"
        }
        if not contour_entries:
            self.status_lbl.setText(f"Hypothesis DB: {n_p} parameter set(s) (no contour_watershed).")
            return

        seed_dist_set: set[int] = set()
        fg_thr_set: set[float] = set()
        for p_idx, info in contour_entries.items():
            d = int(info.get("seed_distance", 10))
            fg = round(float(info.get("foreground_threshold", 0.5)), 4)
            seed_dist_set.add(d)
            fg_thr_set.add(fg)
            self._db_param_map[(d, fg)] = p_idx

        self._db_seed_dist_vals = sorted(seed_dist_set)
        self._db_fg_thr_vals = sorted(fg_thr_set)

        self.db_seed_dist_spin.blockSignals(True)
        self.db_seed_dist_spin.setMinimum(self._db_seed_dist_vals[0])
        self.db_seed_dist_spin.setMaximum(self._db_seed_dist_vals[-1])
        step_d = (self._db_seed_dist_vals[1] - self._db_seed_dist_vals[0]) if len(self._db_seed_dist_vals) > 1 else 1
        self.db_seed_dist_spin.setSingleStep(step_d)
        self.db_seed_dist_spin.setValue(self._db_seed_dist_vals[0])
        self.db_seed_dist_spin.setEnabled(True)
        self.db_seed_dist_spin.blockSignals(False)

        self.db_fg_thr_spin.blockSignals(True)
        self.db_fg_thr_spin.setMinimum(self._db_fg_thr_vals[0])
        self.db_fg_thr_spin.setMaximum(self._db_fg_thr_vals[-1])
        step_fg = round(self._db_fg_thr_vals[1] - self._db_fg_thr_vals[0], 4) if len(self._db_fg_thr_vals) > 1 else 0.05
        self.db_fg_thr_spin.setSingleStep(step_fg)
        self.db_fg_thr_spin.setValue(self._db_fg_thr_vals[0])
        self.db_fg_thr_spin.setEnabled(True)
        self.db_fg_thr_spin.blockSignals(False)

        self._update_db_info_lbl()
        self.status_lbl.setText(f"Hypothesis DB: {n_p} parameter set(s).")

    def _set_status(self, msg: str) -> None:
        self.status_lbl.setText(msg)
        logger.info(msg)

    def _cp_gammas(self) -> list[float]:
        """Gamma values to iterate during consensus boundary building."""
        gmin  = self.cp_gamma_min_spin.value()
        gmax  = self.cp_gamma_max_spin.value()
        gstep = self.cp_gamma_step_spin.value()
        return list(np.arange(gmin, gmax + gstep / 2, gstep))

    def _contour_sweep_params(self) -> ContourWatershedParams:
        return ContourWatershedParams(
            seed_distance=self.single_seed_dist.value(),
            foreground_threshold=self.single_fg_threshold.value(),
            min_size=self.min_size_spin.value(),
            min_circularity=self.min_circularity_spin.value(),
            noise_scale=self.single_noise_scale.value(),
            noise_blur_sigma=self.single_noise_blur.value(),
        )

    def _contour_sweep_spec(self) -> ContourWatershedSweepSpec:
        return ContourWatershedSweepSpec(
            seed_distance=self.sweep_seed_dist[0].value(),
            seed_distance_min=self.sweep_seed_dist[0].value(),
            seed_distance_max=self.sweep_seed_dist[1].value(),
            seed_distance_step=self.sweep_seed_dist[2].value(),
            foreground_threshold=self.sweep_fg_thr[0].value(),
            foreground_threshold_min=self.sweep_fg_thr[0].value(),
            foreground_threshold_max=self.sweep_fg_thr[1].value(),
            foreground_threshold_step=self.sweep_fg_thr[2].value(),
            noise_scale=self.sweep_noise_scale.value(),
            noise_blur_sigma=self.sweep_noise_blur.value(),
            n_runs=self.sweep_n_runs.value(),
            min_size=self.min_size_spin.value(),
            min_circularity=self.min_circularity_spin.value(),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Contour map build
    # ──────────────────────────────────────────────────────────────────────────

    def _build_consensus_boundary_averaged(
        self,
        prob_3d: np.ndarray,
        dp_3d: np.ndarray,
        thresholds: list[float],
        gammas: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        from cellflow.segmentation import build_consensus_boundary

        boundary_sum  = None
        foreground_sum = None
        for g in gammas:
            b, fg = build_consensus_boundary(prob_3d, dp_3d, thresholds, gamma=g)
            if boundary_sum is None:
                boundary_sum  = b.copy()
                foreground_sum = fg.copy()
            else:
                boundary_sum  += b
                foreground_sum += fg
        n = len(gammas)
        return boundary_sum / n, foreground_sum / n

    def _on_build_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path   = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._set_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_status(f"Missing: {dp_path}")
            return

        thresholds      = list(np.arange(self.cp_min_spin.value(), self.cp_max_spin.value() + self.cp_step_spin.value() / 2, self.cp_step_spin.value()))
        gammas          = self._cp_gammas()
        contour_path    = self._contour_maps_path()
        foreground_path = self._foreground_maps_path()
        pos_dir         = self._pos_dir
        build_fn        = self._build_consensus_boundary_averaged

        @thread_worker(connect={
            "yielded":   self._on_build_progress,
            "returned":  self._on_build_done,
            "errored":   self._on_worker_error,
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            dp_stack   = np.asarray(tifffile.imread(str(dp_path)),   dtype=np.float32)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis]
            if dp_stack.ndim == 4:
                dp_stack = dp_stack[np.newaxis]

            n_t = prob_stack.shape[0]
            contour_frames:    list[np.ndarray] = []
            foreground_frames: list[np.ndarray] = []

            for t in range(n_t):
                yield (t + 1, n_t, f"Building contour maps: frame {t + 1}/{n_t}…")
                boundary, fg = build_fn(prob_stack[t], dp_stack[t], thresholds, gammas)
                contour_frames.append(boundary)
                foreground_frames.append(fg)

            contour_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(contour_path),    np.stack(contour_frames),    compression="zlib")
            tifffile.imwrite(str(foreground_path), np.stack(foreground_frames), compression="zlib")
            return pos_dir

        gamma_desc = f"γ={gammas[0]:.2f}" if len(gammas) == 1 else f"γ={gammas[0]:.2f}–{gammas[-1]:.2f} ({len(gammas)} steps)"
        self._set_status(f"Building contour maps ({len(thresholds)} cellprob thresholds, {gamma_desc})…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _on_build_done(self, pos_dir: Path) -> None:
        self._build_worker = None
        self._set_build_buttons_running(False)
        self.contour_files.refresh(pos_dir)
        self._set_status("Contour maps built.")

    def _on_cancel_build(self) -> None:
        if self._build_worker is not None:
            self._build_worker.quit()
        self._build_worker = None
        self._set_build_buttons_running(False)
        self._set_status("Build cancelled.")

    def _set_build_buttons_running(self, running: bool) -> None:
        self.build_btn.setEnabled(not running)
        self.preview_contour_btn.setEnabled(not running)
        self.cancel_build_btn.setEnabled(running)
        self.build_progress_bar.setVisible(running)
        if not running:
            self.build_progress_bar.setValue(0)

    def _on_build_progress(self, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            if total > 0:
                self.build_progress_bar.setRange(0, total)
                self.build_progress_bar.setValue(done)
            self._set_status(msg)
        else:
            self._set_status(str(data))

    def _on_preview_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path   = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._set_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_status(f"Missing: {dp_path}")
            return

        t_frame    = self._current_t()
        thresholds = list(np.arange(self.cp_min_spin.value(), self.cp_max_spin.value() + self.cp_step_spin.value() / 2, self.cp_step_spin.value()))
        gammas     = self._cp_gammas()
        build_fn   = self._build_consensus_boundary_averaged

        def _on_preview_done(result):
            self._build_worker = None
            self._set_build_buttons_running(False)
            boundary, foreground = result
            data = boundary[np.newaxis]
            if _CONTOUR_LAYER in self.viewer.layers:
                self.viewer.layers[_CONTOUR_LAYER].data = data
            else:
                self.viewer.add_image(data, name=_CONTOUR_LAYER, colormap="magma", visible=True)
            self._set_status(
                f"Preview contour map t={t_frame} — "
                f"{len(thresholds)} cellprob thresholds, "
                f"{len(gammas)} gamma value(s)"
            )

        @thread_worker(connect={
            "returned": _on_preview_done,
            "errored":  self._on_worker_error,
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            dp_stack   = np.asarray(tifffile.imread(str(dp_path)),   dtype=np.float32)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis]
            if dp_stack.ndim == 4:
                dp_stack = dp_stack[np.newaxis]
            t_idx = min(t_frame, prob_stack.shape[0] - 1)
            return build_fn(prob_stack[t_idx], dp_stack[t_idx], thresholds, gammas)

        self._set_status(f"Previewing contour map for frame t={t_frame}…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Hypothesis generation
    # ──────────────────────────────────────────────────────────────────────────

    def _load_contour_maps(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Read contour and foreground maps from disk. Returns None and sets status on error."""
        contour_path    = self._contour_maps_path()
        foreground_path = self._foreground_maps_path()
        if contour_path is None or not contour_path.exists():
            self._set_status("Contour maps not found — run Build first.")
            return None
        if foreground_path is None or not foreground_path.exists():
            self._set_status("Foreground maps not found — run Build first.")
            return None
        contour    = np.asarray(tifffile.imread(str(contour_path)),    dtype=np.float32)
        foreground = np.asarray(tifffile.imread(str(foreground_path)), dtype=np.float32)
        return contour, foreground

    def _on_preview(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        maps = self._load_contour_maps()
        if maps is None:
            return
        contour, foreground = maps
        t = min(self._current_t(), contour.shape[0] - 1)
        params = self._contour_sweep_params()

        if _CONTOUR_LAYER not in self.viewer.layers:
            self.viewer.add_image(contour, name=_CONTOUR_LAYER, colormap="magma", visible=True)
        elif self.viewer.layers[_CONTOUR_LAYER].data is not contour:
            self.viewer.layers[_CONTOUR_LAYER].data = contour

        try:
            labels = compute_contour_watershed(contour[t], foreground[t], params)
        except Exception as e:
            self._set_status(f"Segmentation failed: {e}")
            return

        self._update_layer(_PREVIEW_LAYER, labels)
        self._set_status(
            f"Preview t={t}: {int(labels.max())} cells  "
            f"(dist={params.seed_distance})"
        )

    def _on_save_db(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        maps = self._load_contour_maps()
        if maps is None:
            return
        contour, foreground = maps

        params   = self._contour_sweep_params()
        overwrite = self.overwrite_check.isChecked()
        output_path = self._hyp_path()
        pos_dir     = self._pos_dir

        @thread_worker(connect={"returned": self._on_save_done, "errored": self._on_worker_error})
        def _worker():
            spec = ContourWatershedSweepSpec(
                seed_distance=params.seed_distance,
                seed_distance_min=params.seed_distance,
                seed_distance_max=params.seed_distance,
                seed_distance_step=1,
                foreground_threshold=params.foreground_threshold,
                foreground_threshold_min=params.foreground_threshold,
                foreground_threshold_max=params.foreground_threshold,
                foreground_threshold_step=0.05,
                noise_scale=params.noise_scale,
                noise_blur_sigma=params.noise_blur_sigma,
                min_size=params.min_size,
                min_circularity=params.min_circularity,
            )
            records = iter_contour_watershed_records(contour, foreground, spec)
            write_hypothesis_sweep_h5(output_path, records, overwrite=overwrite, n_t=None, n_p=1)
            return pos_dir

        self._set_status("Saving to DB…")
        _worker()

    def _on_save_done(self, pos_dir: Path) -> None:
        self.output_files.refresh(pos_dir)
        self._set_status("Saved to hypotheses.h5.")
        self.refresh(pos_dir)

    def _on_run_sweep(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        contour_path    = self._contour_maps_path()
        foreground_path = self._foreground_maps_path()
        if contour_path is None or not contour_path.exists():
            self._set_status("Contour maps not found — run Build first.")
            return
        if foreground_path is None or not foreground_path.exists():
            self._set_status("Foreground maps not found — run Build first.")
            return

        spec      = self._contour_sweep_spec()
        overwrite = self.overwrite_check.isChecked()
        output_path = self._hyp_path()
        pos_dir     = self._pos_dir

        def _on_sweep_done(result):
            self._sweep_worker = None
            self._set_sweep_buttons_running(False)
            self._on_save_done(result)

        def _on_sweep_aborted():
            self._sweep_worker = None
            self._set_sweep_buttons_running(False)
            self._set_status("Sweep cancelled.")

        def _on_sweep_error(exc):
            self._sweep_worker = None
            self._set_sweep_buttons_running(False)
            self._on_worker_error(exc)

        @thread_worker(connect={
            "yielded":  self._set_status,
            "returned": _on_sweep_done,
            "aborted":  _on_sweep_aborted,
            "errored":  _on_sweep_error,
        })
        def _worker():
            import json as _json
            params_list = build_contour_watershed_parameter_sets(spec)

            if not overwrite and output_path.exists():
                try:
                    _, existing = list_hypotheses(output_path)
                    existing_jsons = {
                        attrs["parameter_json"]
                        for attrs in existing.values()
                        if "parameter_json" in attrs
                    }
                    params_list = [
                        p for p in params_list
                        if _json.dumps(p.to_dict(), sort_keys=True) not in existing_jsons
                    ]
                except Exception:
                    pass

            n_full = len(build_contour_watershed_parameter_sets(spec))
            n_skip = n_full - len(params_list)
            if not params_list:
                yield f"Sweep: all {n_full} parameter set(s) already present, nothing to do."
                return pos_dir
            if n_skip:
                yield f"Sweep: skipping {n_skip} existing, computing {len(params_list)} new…"

            contour_stack    = np.asarray(tifffile.imread(str(contour_path)),    dtype=np.float32)
            foreground_stack = np.asarray(tifffile.imread(str(foreground_path)), dtype=np.float32)

            n_t = contour_stack.shape[0]
            total = n_t * len(params_list)
            collected: list[HypothesisRecord] = []
            for done, record in enumerate(
                iter_contour_watershed_records(contour_stack, foreground_stack, spec), 1
            ):
                collected.append(record)
                yield f"Sweep {done}/{total}…"
            write_hypothesis_sweep_h5(output_path, iter(collected), overwrite=overwrite)
            return pos_dir

        self._set_status("Running sweep…")
        self._set_sweep_buttons_running(True)
        self._sweep_worker = _worker()

    def _set_sweep_buttons_running(self, running: bool) -> None:
        self.run_sweep_btn.setEnabled(not running)
        self.run_terminal_btn.setEnabled(not running)
        self.cancel_sweep_btn.setEnabled(running)

    def _on_cancel_sweep(self) -> None:
        if self._sweep_worker is not None:
            self._sweep_worker.quit()

    def _on_run_terminal(self) -> None:
        import sys
        import tempfile

        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        contour_path    = self._contour_maps_path()
        foreground_path = self._foreground_maps_path()
        output_path     = self._hyp_path()
        if contour_path is None or not contour_path.exists():
            self._set_status("Contour maps not found — run Build first.")
            return
        if foreground_path is None or not foreground_path.exists():
            self._set_status("Foreground maps not found — run Build first.")
            return

        spec      = self._contour_sweep_spec()
        overwrite = self.overwrite_check.isChecked()

        python_code = (
            "import tifffile, numpy as np\n"
            "from cellflow.database.hypotheses import (\n"
            "    ContourWatershedSweepSpec, iter_contour_watershed_records,\n"
            "    build_contour_watershed_parameter_sets, list_hypotheses,\n"
            "    write_hypothesis_sweep_h5)\n"
            "import json, pathlib\n"
            f"contour    = tifffile.imread({str(contour_path)!r}).astype('float32')\n"
            f"foreground = tifffile.imread({str(foreground_path)!r}).astype('float32')\n"
            f"output_path = pathlib.Path({str(output_path)!r})\n"
            f"overwrite = {overwrite!r}\n"
            f"spec = ContourWatershedSweepSpec(\n"
            f"    seed_distance={spec.seed_distance},\n"
            f"    seed_distance_min={spec.seed_distance_min}, seed_distance_max={spec.seed_distance_max},\n"
            f"    seed_distance_step={spec.seed_distance_step},\n"
            f"    foreground_threshold={spec.foreground_threshold},\n"
            f"    foreground_threshold_min={spec.foreground_threshold_min},\n"
            f"    foreground_threshold_max={spec.foreground_threshold_max},\n"
            f"    foreground_threshold_step={spec.foreground_threshold_step},\n"
            f"    noise_scale={spec.noise_scale},\n"
            f"    noise_blur_sigma={spec.noise_blur_sigma},\n"
            f"    n_runs={spec.n_runs},\n"
            f"    min_size={spec.min_size},\n"
            f"    min_circularity={spec.min_circularity},\n"
            ")\n"
            "params_list = build_contour_watershed_parameter_sets(spec)\n"
            "if not overwrite and output_path.exists():\n"
            "    try:\n"
            "        _, existing = list_hypotheses(output_path)\n"
            "        existing_jsons = {attrs['parameter_json'] for attrs in existing.values() if 'parameter_json' in attrs}\n"
            "        params_list = [p for p in params_list if json.dumps(p.to_dict(), sort_keys=True) not in existing_jsons]\n"
            "    except Exception:\n"
            "        pass\n"
            "n_t = contour.shape[0]\n"
            "total = n_t * len(params_list)\n"
            "records = []\n"
            "for done, rec in enumerate(iter_contour_watershed_records(contour, foreground, spec), 1):\n"
            "    records.append(rec)\n"
            "    print(f'Sweep {done}/{total}…', flush=True)\n"
            "write_hypothesis_sweep_h5(str(output_path), iter(records), overwrite=overwrite)\n"
            "print('Done.')\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="cellflow_sweep_", delete=False
        ) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path)}"
        try:
            from cellflow.napari.utils import launch_in_terminal
            launch_in_terminal(cmd)
            self._set_status("Command launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(cmd)
            self._set_status("Copied command to clipboard (terminal launch unavailable).")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Database Browser
    # ──────────────────────────────────────────────────────────────────────────

    def _on_db_activate_toggled(self, active: bool) -> None:
        self.db_activate_btn.setText("Deactivate" if active else "Activate")

    def _lookup_db_p(self) -> int | None:
        if not self._db_param_map:
            return None
        d = self.db_seed_dist_spin.value()
        fg = round(self.db_fg_thr_spin.value(), 4)
        if self._db_seed_dist_vals:
            d = min(self._db_seed_dist_vals, key=lambda x: abs(x - d))
        if self._db_fg_thr_vals:
            fg = round(min(self._db_fg_thr_vals, key=lambda x: abs(x - fg)), 4)
        return self._db_param_map.get((d, fg))

    def _update_db_info_lbl(self) -> None:
        p = self._lookup_db_p()
        self._current_db_p = p
        self.db_info_lbl.setText(f"p={p:03d}" if p is not None else "—")

    def _on_db_param_changed(self) -> None:
        self._update_db_info_lbl()
        if self.db_activate_btn.isChecked() and self._current_db_p is not None:
            self._load_db_stack(self._current_db_p)

    def _load_db_stack(self, p: int) -> None:
        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            return
        self._set_status(f"Loading p={p}…")

        @thread_worker(connect={"returned": self._on_load_stack_done, "errored": self._on_worker_error})
        def _worker():
            return p, read_full_hypothesis_stack(hyp_path, p)

        _worker()

    def _on_load_stack_done(self, result: tuple) -> None:
        p, stack = result
        if stack.ndim == 4:
            stack = stack[:, 0]  # contour_watershed stores (1, Y, X) per frame
        if _HYP_LAYER in self.viewer.layers:
            self.viewer.layers[_HYP_LAYER].data = stack
        else:
            self.viewer.add_labels(stack, name=_HYP_LAYER)
        n_cells = int(stack.max()) if stack.size > 0 else 0
        self.db_info_lbl.setText(f"p={p:03d}  |  {n_cells} cells")
        self._set_status(f"Loaded p={p} → {stack.shape} into napari.")

    def _on_set_seed(self) -> None:
        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_status("No hypothesis DB found.")
            return
        p = self._current_db_p
        if p is None:
            self._set_status("No parameter set selected in the DB browser.")
            return
        t = self._current_t()
        try:
            volume = read_hypothesis_labels(hyp_path, t, p)  # (1, Y, X) for contour_watershed
            slice_2d = volume[0]
            tracked_path = self._tracked_path()
            write_tracked_frame(tracked_path, t, slice_2d)
            self._update_tracked_display(slice_2d, t=t)
            self._set_status(f"Hypothesis p={p} set as tracking seed at t={t}.")
        except Exception as e:
            self._set_status(f"Error setting seed: {e}")

    def _on_remove_stack(self) -> None:
        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_status("No hypothesis DB found.")
            return
        p = self._current_db_p
        if p is None:
            self._set_status("No parameter set selected in the DB browser.")
            return
        try:
            delete_hypothesis_parameter(hyp_path, p)
        except Exception as e:
            self._set_status(f"Remove stack failed: {e}")
            return
        if _HYP_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_HYP_LAYER])
        self._current_db_p = None
        self._set_status(f"Removed p={p}.")
        self.refresh(self._pos_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Automated search / propagation
    # ──────────────────────────────────────────────────────────────────────────

    def _on_propagate_next(self) -> None:
        hyp_path     = self._hyp_path()
        tracked_path = self._tracked_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_status("No hypothesis DB found.")
            return
        if tracked_path is None or not tracked_path.exists():
            self._set_status("No tracked labels file. Set a tracking seed first.")
            return

        t = self._current_t()
        if not tracked_frame_exists(tracked_path, t):
            self._set_status(f"No tracked frame at t={t}. Set a seed first.")
            return

        if _TRACKED_LAYER in self.viewer.layers:
            layer = self.viewer.layers[_TRACKED_LAYER]
            if layer.data.ndim == 3 and t < layer.data.shape[0]:
                write_tracked_frame(tracked_path, t, np.asarray(layer.data[t]))

        try:
            winner = propagate_one_frame(
                hyp_path, tracked_path, t,
                iou_threshold=self.iou_spin.value(),
                max_dist_px=self.dist_spin.value(),
                velocity_sigma_px=self.vel_sigma_spin.value(),
                iou_weight=self.iou_weight_spin.value(),
                area_weight=self.area_weight_spin.value(),
                velocity_weight=self.vel_weight_spin.value(),
                pos_weight=self.pos_weight_spin.value(),
                unmatched_score=self.unmatched_spin.value(),
            )
        except Exception as e:
            self._set_status(f"Propagation failed: {e}")
            return

        if winner is None:
            self._set_status(f"No suitable hypothesis found for t={t + 1}.")
            return

        try:
            labels = read_tracked_frame(tracked_path, t + 1)
            self._update_tracked_display(labels, t=t + 1, tracked_path=tracked_path)
            step = list(self.viewer.dims.current_step)
            step[0] = t + 1
            self.viewer.dims.current_step = tuple(step)
        except Exception as e:
            self._set_status(f"Could not load t={t + 1}: {e}")
            return

        self._set_status(f"Propagated t={t}→{t + 1} using p={winner}.")

    def _on_propagate_all(self) -> None:
        hyp_path     = self._hyp_path()
        tracked_path = self._tracked_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_status("No hypothesis DB found.")
            return
        if tracked_path is None or not tracked_path.exists():
            self._set_status("No tracked labels file. Set a tracking seed first.")
            return

        n_p, _ = list_hypotheses(hyp_path)
        if n_p == 0:
            self._set_status("Hypothesis DB is empty.")
            return

        t_start   = self._current_t()
        iou_thr   = self.iou_spin.value()
        max_dist  = self.dist_spin.value()
        vel_sigma = self.vel_sigma_spin.value()
        iou_w     = self.iou_weight_spin.value()
        area_w    = self.area_weight_spin.value()
        vel_w     = self.vel_weight_spin.value()
        pos_w     = self.pos_weight_spin.value()
        unmatch_s = self.unmatched_spin.value()
        self._stop_flag = False

        if _TRACKED_LAYER in self.viewer.layers:
            layer = self.viewer.layers[_TRACKED_LAYER]
            if layer.data.ndim == 3 and t_start < layer.data.shape[0]:
                write_tracked_frame(tracked_path, t_start, np.asarray(layer.data[t_start]))

        @thread_worker(connect={"yielded": self._on_prop_progress, "finished": self._on_prop_done, "errored": self._on_worker_error})
        def _worker():
            t = t_start
            while not self._stop_flag:
                if not tracked_frame_exists(tracked_path, t):
                    break
                winner = propagate_one_frame(
                    hyp_path, tracked_path, t, iou_thr, max_dist,
                    velocity_sigma_px=vel_sigma,
                    iou_weight=iou_w,
                    area_weight=area_w,
                    velocity_weight=vel_w,
                    pos_weight=pos_w,
                    unmatched_score=unmatch_s,
                )
                if winner is None:
                    yield (t, None)
                    break
                yield (t, winner)
                t += 1

        self._set_status("Propagating…")
        _worker()

    def _on_prop_progress(self, result: tuple[int, int | None]) -> None:
        t, winner = result
        if winner is None:
            self._set_status(f"Propagation stopped at t={t}: no suitable hypothesis.")
        else:
            self._set_status(f"Propagated t={t}→{t + 1} (p={winner})")
            try:
                tracked_path = self._tracked_path()
                labels = read_tracked_frame(tracked_path, t + 1)
                self._update_tracked_display(labels, t=t + 1, tracked_path=tracked_path)
                step = list(self.viewer.dims.current_step)
                step[0] = t + 1
                self.viewer.dims.current_step = tuple(step)
            except Exception:
                pass

    def _on_prop_done(self) -> None:
        self._set_status("Propagation complete.")

    def _on_load_tracked(self) -> None:
        tracked_path   = self._tracked_path()
        cell_zavg_path = self._cell_zavg_path()
        nuc_zavg_path  = self._nucleus_zavg_path()
        if tracked_path is None or not tracked_path.exists():
            self._set_status("No tracked labels file found.")
            return
        self._set_status("Loading tracked labels…")

        @thread_worker(connect={"returned": self._on_load_tracked_done, "errored": self._on_worker_error})
        def _worker():
            stack = read_full_tracked_stack(tracked_path)
            cell_zavg = (
                np.asarray(tifffile.imread(str(cell_zavg_path)), dtype=np.float32)
                if cell_zavg_path and cell_zavg_path.exists() else None
            )
            nuc_zavg = (
                np.asarray(tifffile.imread(str(nuc_zavg_path)), dtype=np.float32)
                if nuc_zavg_path and nuc_zavg_path.exists() else None
            )
            return stack, cell_zavg, nuc_zavg

        _worker()

    def _on_load_tracked_done(self, result: tuple) -> None:
        stack, cell_zavg, nuc_zavg = result
        nt = stack.shape[0]
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = stack
        else:
            self.viewer.add_labels(stack, name=_TRACKED_LAYER)

        for zavg_data, layer_name, cmap in (
            (cell_zavg, _CELL_ZAVG_LAYER, "gray"),
            (nuc_zavg,  _NUC_ZAVG_LAYER,  "bop orange"),
        ):
            if zavg_data is None:
                continue
            if zavg_data.ndim == 2:
                broadcast_zavg = np.broadcast_to(zavg_data[np.newaxis], (nt,) + zavg_data.shape).copy()
            else:
                broadcast_zavg = zavg_data
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = broadcast_zavg
            else:
                self.viewer.add_image(broadcast_zavg, name=layer_name, colormap=cmap, blending="additive")

        self._set_status(f"Loaded tracked stack {stack.shape} into napari.")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Manual correction
    # ──────────────────────────────────────────────────────────────────────────

    def _on_dims_step_changed(self, event=None) -> None:
        self._refresh_validate_btn()

    def _refresh_validate_btn(self) -> None:
        if self._pos_dir is None:
            self.validate_btn.setChecked(False)
            return
        t = self._current_t()
        validated = is_validated(self._pos_dir, t)
        self.validate_btn.blockSignals(True)
        self.validate_btn.setChecked(validated)
        self.validate_btn.blockSignals(False)

    def _on_validate_toggled(self, checked: bool) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        t = self._current_t()
        if checked:
            validate_frame(self._pos_dir, t)
            self._set_status(f"Frame t={t} marked as validated.")
        else:
            invalidate_frame(self._pos_dir, t)
            self._set_status(f"Frame t={t} validation removed.")

    def _on_retrack_frame(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        tracked_path = self._tracked_path()
        if tracked_path is None or not tracked_path.exists():
            self._set_status("No tracked labels file found.")
            return

        t = self._current_t()
        if is_validated(self._pos_dir, t):
            self._set_status(f"Frame t={t} is validated — unvalidate it first to retrack.")
            return
        if not tracked_frame_exists(tracked_path, t):
            self._set_status(f"No tracked frame at t={t}.")
            return

        validated = sorted(
            [v for v in read_validated_frames(self._pos_dir) if v < t], reverse=True
        )
        t_ref = validated[0] if validated else (t - 1)
        if t_ref < 0:
            self._set_status("No reference frame available (t=0 has no predecessor).")
            return
        if not tracked_frame_exists(tracked_path, t_ref):
            self._set_status(f"Reference frame t={t_ref} does not exist yet.")
            return

        if _TRACKED_LAYER in self.viewer.layers:
            layer = self.viewer.layers[_TRACKED_LAYER]
            if layer.data.ndim == 3:
                for flush_t in (t, t_ref):
                    if flush_t < layer.data.shape[0]:
                        write_tracked_frame(tracked_path, flush_t, np.asarray(layer.data[flush_t]))

        try:
            ref_labels = read_tracked_frame(tracked_path, t_ref)
            tgt_labels = read_tracked_frame(tracked_path, t)
        except Exception as e:
            self._set_status(f"Could not read frames: {e}")
            return

        remapped = retrack_frame(ref_labels, tgt_labels, max_dist_px=self.dist_spin.value())
        new_ids = set(int(i) for i in np.unique(remapped) if i != 0)
        ref_ids = set(int(i) for i in np.unique(ref_labels) if i != 0)
        n_matched = len(new_ids & ref_ids)
        n_new     = len(new_ids - ref_ids)

        write_tracked_frame(tracked_path, t, remapped)
        self._update_tracked_display(remapped, t=t, tracked_path=tracked_path)
        self._set_status(
            f"Retracked t={t} using t={t_ref}: {n_matched} matched, {n_new} new ID(s)."
        )

    def _on_jump_correction(self) -> None:
        self._set_status("Manual correction widget not yet connected.")

    # ──────────────────────────────────────────────────────────────────────────
    # Error handler
    # ──────────────────────────────────────────────────────────────────────────

    def _on_worker_error(self, exc: Exception) -> None:
        self._set_status(f"Error: {exc}")
        logger.exception("Worker error", exc_info=exc)

    # ──────────────────────────────────────────────────────────────────────────
    # State persistence
    # ──────────────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "overwrite":        self.overwrite_check.isChecked(),
            "min_size":         self.min_size_spin.value(),
            "min_circularity":  self.min_circularity_spin.value(),
            "cellprob": {
                "min":       self.cp_min_spin.value(),
                "max":       self.cp_max_spin.value(),
                "step":      self.cp_step_spin.value(),
                "gamma_min": self.cp_gamma_min_spin.value(),
                "gamma_max": self.cp_gamma_max_spin.value(),
                "gamma_step": self.cp_gamma_step_spin.value(),
            },
            "tuning": {
                "seed_dist":      self.single_seed_dist.value(),
                "fg_threshold":   self.single_fg_threshold.value(),
                "noise_scale":    self.single_noise_scale.value(),
                "noise_blur_sigma": self.single_noise_blur.value(),
            },
            "sweep": {
                "seed_dist_min":  self.sweep_seed_dist[0].value(),
                "seed_dist_max":  self.sweep_seed_dist[1].value(),
                "seed_dist_step": self.sweep_seed_dist[2].value(),
                "fg_thr_min":     self.sweep_fg_thr[0].value(),
                "fg_thr_max":     self.sweep_fg_thr[1].value(),
                "fg_thr_step":    self.sweep_fg_thr[2].value(),
                "noise_scale":    self.sweep_noise_scale.value(),
                "noise_blur_sigma": self.sweep_noise_blur.value(),
                "n_runs":         self.sweep_n_runs.value(),
            },
            "db_browser": {
                "seed_dist":    self.db_seed_dist_spin.value(),
                "fg_threshold": self.db_fg_thr_spin.value(),
            },
            "search": {
                "iou_threshold":    self.iou_spin.value(),
                "max_dist_um":      self.dist_spin.value(),
                "velocity_sigma_px": self.vel_sigma_spin.value(),
                "iou_weight":       self.iou_weight_spin.value(),
                "area_weight":      self.area_weight_spin.value(),
                "velocity_weight":  self.vel_weight_spin.value(),
                "pos_weight":       self.pos_weight_spin.value(),
                "unmatched_score":  self.unmatched_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if "overwrite" in state:
            self.overwrite_check.setChecked(state["overwrite"])
        if "min_size" in state:
            self.min_size_spin.setValue(state["min_size"])
        if "min_circularity" in state:
            self.min_circularity_spin.setValue(state["min_circularity"])
        if "cellprob" in state:
            cp = state["cellprob"]
            if "min"        in cp: self.cp_min_spin.setValue(cp["min"])
            if "max"        in cp: self.cp_max_spin.setValue(cp["max"])
            if "step"       in cp: self.cp_step_spin.setValue(cp["step"])
            if "gamma_min"  in cp: self.cp_gamma_min_spin.setValue(cp["gamma_min"])
            if "gamma_max"  in cp: self.cp_gamma_max_spin.setValue(cp["gamma_max"])
            if "gamma_step" in cp: self.cp_gamma_step_spin.setValue(cp["gamma_step"])
        if "tuning" in state:
            t = state["tuning"]
            if "seed_dist"    in t: self.single_seed_dist.setValue(t["seed_dist"])
            if "fg_threshold" in t: self.single_fg_threshold.setValue(t["fg_threshold"])
            if "noise_scale"  in t: self.single_noise_scale.setValue(t["noise_scale"])
            if "noise_blur_sigma" in t: self.single_noise_blur.setValue(t["noise_blur_sigma"])
        if "sweep" in state:
            sw = state["sweep"]
            if "seed_dist_min"    in sw: self.sweep_seed_dist[0].setValue(sw["seed_dist_min"])
            if "seed_dist_max"    in sw: self.sweep_seed_dist[1].setValue(sw["seed_dist_max"])
            if "seed_dist_step"   in sw: self.sweep_seed_dist[2].setValue(sw["seed_dist_step"])
            if "fg_thr_min"       in sw: self.sweep_fg_thr[0].setValue(sw["fg_thr_min"])
            if "fg_thr_max"       in sw: self.sweep_fg_thr[1].setValue(sw["fg_thr_max"])
            if "fg_thr_step"      in sw: self.sweep_fg_thr[2].setValue(sw["fg_thr_step"])
            if "noise_scale"      in sw: self.sweep_noise_scale.setValue(sw["noise_scale"])
            if "noise_blur_sigma" in sw: self.sweep_noise_blur.setValue(sw["noise_blur_sigma"])
            if "n_runs"           in sw: self.sweep_n_runs.setValue(sw["n_runs"])
        if "db_browser" in state:
            db = state["db_browser"]
            if "seed_dist"    in db: self.db_seed_dist_spin.setValue(db["seed_dist"])
            if "fg_threshold" in db: self.db_fg_thr_spin.setValue(db["fg_threshold"])
        if "search" in state:
            se = state["search"]
            if "iou_threshold"     in se: self.iou_spin.setValue(se["iou_threshold"])
            if "max_dist_um"       in se: self.dist_spin.setValue(se["max_dist_um"])
            if "velocity_sigma_px" in se: self.vel_sigma_spin.setValue(se["velocity_sigma_px"])
            if "iou_weight"        in se: self.iou_weight_spin.setValue(se["iou_weight"])
            if "area_weight"       in se: self.area_weight_spin.setValue(se["area_weight"])
            if "velocity_weight"   in se: self.vel_weight_spin.setValue(se["velocity_weight"])
            if "pos_weight"        in se: self.pos_weight_spin.setValue(se["pos_weight"])
            if "unmatched_score"   in se: self.unmatched_spin.setValue(se["unmatched_score"])
