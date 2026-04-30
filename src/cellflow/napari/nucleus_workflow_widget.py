"""Nucleus workflow widget for hypothesis generation and tracking in CellFlow v2."""
from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from napari.utils.colormaps import direct_colormap
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QScrollArea,
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
    write_tracked_frame,
)
from cellflow.database.validation import (
    invalidate_track,
    is_track_validated,
    is_validated,
    read_validated_cells_at_frame,
    read_validated_frames,
    read_validated_tracks,
    remap_validated_tracks,
    validate_track,
)
from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.napari.ui_style import (
    add_block_button_row,
    add_block_checkbox_row,
    add_block_pair_row,
    add_sweep_parameter_row,
    block_grid,
    compact_spinbox,
    danger_button,
    sweep_parameter_grid,
)
from cellflow.segmentation import ContourWatershedParams, compute_contour_watershed
from cellflow.tracking.retracker import retrack_frame_constrained
from cellflow.tracking_ultrack.config import TrackingConfig as UltrackConfig
from cellflow.tracking_ultrack.export import export_tracked_labels
from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db, _select_solver
from cellflow.tracking_ultrack.linking import run_linking
from cellflow.tracking_ultrack.extend import extend_track
from cellflow.tracking_ultrack.reseed import resolve_with_validation
from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs
from cellflow.tracking_ultrack.solve import run_solve

logger = logging.getLogger(__name__)

_PREVIEW_LAYER = "Preview: Nucleus"
_HYP_LAYER = "Hypothesis: Nucleus"
_TRACKED_LAYER = "Tracked: Nucleus"
_VALIDATED_OVERLAY = "Validated: Nucleus"
_CONTOUR_LAYER = "Contour Map: Nucleus"
_CELL_ZAVG_LAYER = "Cell z-avg"
_NUC_ZAVG_LAYER = "Nucleus z-avg"
_CONTOUR_SWEEP_WIDTH = 60
_CONTOUR_SWEEP_MIN_WIDTH = int(_CONTOUR_SWEEP_WIDTH * 0.9)


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
        self._db_param_map: dict[tuple[int, float, float, int], int] = {}
        self._db_seed_dist_vals: list[int] = []
        self._db_fg_thr_vals: list[float] = []
        self._db_ridge_thr_vals: list[float] = []
        self._db_run_vals: list[int] = []
        self._setup_ui()
        self._connect_signals()

    # ──────────────────────────────────────────────────────────────────────────
    # UI setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        # ── Compact layout helpers ────────────────────────────────────────
        SPIN_MAX_W = 70

        def _compact(spin, w=SPIN_MAX_W):
            return compact_spinbox(spin, w)

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
        contour_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        cp_params_scroll = QScrollArea()
        cp_params_scroll.setWidgetResizable(True)
        cp_params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        cp_params_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        cp_params_scroll.setFrameShape(QFrame.NoFrame)
        cp_params_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        cp_params_widget = QWidget()
        cp_params_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        cp_params_lay = QVBoxLayout(cp_params_widget)
        cp_params_lay.setContentsMargins(0, 0, 0, 0)
        cp_params_lay.setSpacing(4)
        cp_params_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        contour_sweep_grid = sweep_parameter_grid(spin_width=_CONTOUR_SWEEP_WIDTH)
        self.cp_min_spin = QDoubleSpinBox()
        self.cp_min_spin.setRange(-20.0, 20.0)
        self.cp_min_spin.setValue(-3.0)
        self.cp_min_spin.setDecimals(1)
        self.cp_min_spin.setSingleStep(1.0)
        self.cp_min_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_min_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.cp_max_spin = QDoubleSpinBox()
        self.cp_max_spin.setRange(-20.0, 20.0)
        self.cp_max_spin.setValue(0.0)
        self.cp_max_spin.setDecimals(1)
        self.cp_max_spin.setSingleStep(1.0)
        self.cp_max_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_max_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.cp_step_spin = QDoubleSpinBox()
        self.cp_step_spin.setRange(0.1, 10.0)
        self.cp_step_spin.setValue(1.0)
        self.cp_step_spin.setDecimals(1)
        self.cp_step_spin.setSingleStep(0.5)
        self.cp_step_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_step_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        contour_sweep_grid.addWidget(QLabel("Cellprob:"), 1, 0)
        contour_sweep_grid.addWidget(self.cp_min_spin, 1, 1)
        contour_sweep_grid.addWidget(self.cp_max_spin, 1, 2)
        contour_sweep_grid.addWidget(self.cp_step_spin, 1, 3)
        contour_sweep_grid.setColumnStretch(1, 1)
        contour_sweep_grid.setColumnStretch(2, 1)
        contour_sweep_grid.setColumnStretch(3, 1)

        self.cp_gamma_min_spin = QDoubleSpinBox()
        self.cp_gamma_min_spin.setRange(0.05, 5.0)
        self.cp_gamma_min_spin.setValue(1.0)
        self.cp_gamma_min_spin.setDecimals(2)
        self.cp_gamma_min_spin.setSingleStep(0.05)
        self.cp_gamma_min_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_gamma_min_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.cp_gamma_max_spin = QDoubleSpinBox()
        self.cp_gamma_max_spin.setRange(0.05, 5.0)
        self.cp_gamma_max_spin.setValue(1.0)
        self.cp_gamma_max_spin.setDecimals(2)
        self.cp_gamma_max_spin.setSingleStep(0.05)
        self.cp_gamma_max_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_gamma_max_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.cp_gamma_step_spin = QDoubleSpinBox()
        self.cp_gamma_step_spin.setRange(0.05, 2.0)
        self.cp_gamma_step_spin.setValue(0.25)
        self.cp_gamma_step_spin.setDecimals(2)
        self.cp_gamma_step_spin.setSingleStep(0.05)
        self.cp_gamma_step_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_gamma_step_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        _gamma_tip = (
            "Gamma correction on Cellpose probability logits before boundary building. "
            "<1 boosts dim signals; >1 suppresses them. 1.0 = no correction. "
            "Contour maps are averaged over all gamma values in [min, max]."
        )
        for _w in (self.cp_gamma_min_spin, self.cp_gamma_max_spin, self.cp_gamma_step_spin):
            _w.setToolTip(_gamma_tip)
        contour_sweep_grid.addWidget(QLabel("Gamma:"), 2, 0)
        contour_sweep_grid.addWidget(self.cp_gamma_min_spin, 2, 1)
        contour_sweep_grid.addWidget(self.cp_gamma_max_spin, 2, 2)
        contour_sweep_grid.addWidget(self.cp_gamma_step_spin, 2, 3)
        contour_sweep_grid.setColumnStretch(1, 1)
        contour_sweep_grid.setColumnStretch(2, 1)
        contour_sweep_grid.setColumnStretch(3, 1)
        cp_params_lay.addLayout(contour_sweep_grid)

        self.save_source_check = QCheckBox("Save label images")
        self.save_source_check.setToolTip("Save all label images used for contour building in 2_nucleus/source_labels/")
        self.save_source_check.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        contour_sweep_grid.addWidget(QLabel(""), 3, 0)
        contour_sweep_grid.addWidget(
            self.save_source_check,
            3,
            1,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        self.preview_contour_btn = QPushButton("Preview")
        self.preview_contour_btn.setToolTip(
            "Build contour maps for the current frame only and display in napari"
        )
        self.build_btn = QPushButton("Build")
        self.cancel_build_btn = QPushButton("Cancel")
        self.cancel_build_btn.setEnabled(False)

        for button in (self.preview_contour_btn, self.build_btn, self.cancel_build_btn):
            button.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )

        contour_sweep_grid.addWidget(
            QLabel(""),
            4,
            0,
        )
        contour_sweep_grid.addWidget(
            self.preview_contour_btn,
            4,
            1,
        )
        contour_sweep_grid.addWidget(
            self.build_btn,
            4,
            2,
        )
        contour_sweep_grid.addWidget(
            self.cancel_build_btn,
            4,
            3,
        )

        self.build_progress_bar = QProgressBar()
        self.build_progress_bar.setRange(0, 100)
        self.build_progress_bar.setValue(0)
        self.build_progress_bar.setVisible(False)
        self.contour_files = PipelineFilesWidget([
            ("", [
                ("2_nucleus/contour_maps.tif",   "Contour maps"),
                ("2_nucleus/foreground_maps.tif", "Foreground maps"),
            ]),
        ])
        cp_params_lay.addWidget(self.build_progress_bar)
        cp_params_lay.addWidget(self.contour_files)

        cp_params_scroll.setWidget(cp_params_widget)
        contour_lay.addWidget(cp_params_scroll)
        self.contour_section = CollapsibleSection(
            "1. Contour Maps", _contour_inner, expanded=False
        )
        layout.addWidget(self.contour_section)

        # ── 2. Hypothesis Generation ──────────────────────────────────────
        _gen_inner = QWidget()
        gen_lay = QVBoxLayout(_gen_inner)
        gen_lay.setContentsMargins(4, 4, 4, 4)
        gen_lay.setSpacing(6)
        gen_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        gen_params_grid = block_grid(horizontal_spacing=12)
        gen_params_grid.setColumnStretch(1, 1)
        gen_params_grid.setColumnStretch(3, 1)

        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 100000)
        self.min_size_spin.setValue(0)
        self.min_size_spin.setToolTip("Remove regions smaller than this many pixels (0 = keep all)")
        self.min_circularity_spin = QDoubleSpinBox()
        self.min_circularity_spin.setRange(0.0, 1.0)
        self.min_circularity_spin.setValue(0.0)
        self.min_circularity_spin.setDecimals(2)
        self.min_circularity_spin.setSingleStep(0.05)
        self.min_circularity_spin.setToolTip(
            "Remove regions with circularity (4π·area/perimeter²) below this value (0 = keep all, 1 = perfect circle)"
        )
        for spin in (self.min_size_spin, self.min_circularity_spin):
            spin.setMinimumWidth(80)
            spin.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        gen_params_grid.addWidget(QLabel("Min Cell Size (px):"), 0, 0)
        gen_params_grid.addWidget(self.min_size_spin, 0, 1)
        gen_params_grid.addWidget(QLabel("Min Circularity:"), 0, 2)
        gen_params_grid.addWidget(self.min_circularity_spin, 0, 3)

        self.noise_scale = QDoubleSpinBox()
        self.noise_scale.setRange(0.0, 1.0)
        self.noise_scale.setValue(0.0)
        self.noise_scale.setDecimals(2)
        self.noise_scale.setSingleStep(0.01)
        self.noise_scale.setToolTip("Stochastic perturbation level for segmentation diversity.")
        self.noise_blur = QDoubleSpinBox()
        self.noise_blur.setRange(0.0, 10.0)
        self.noise_blur.setValue(0.0)
        self.noise_blur.setDecimals(1)
        self.noise_blur.setSingleStep(0.5)
        self.noise_blur.setToolTip("Sigma for correlating noise (higher = larger structures).")
        for spin in (self.noise_scale, self.noise_blur):
            spin.setMinimumWidth(80)
            spin.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        gen_params_grid.addWidget(QLabel("Noise Scale:"), 1, 0)
        gen_params_grid.addWidget(self.noise_scale, 1, 1)
        gen_params_grid.addWidget(QLabel("Blur Sigma:"), 1, 2)
        gen_params_grid.addWidget(self.noise_blur, 1, 3)

        gen_lay.addLayout(gen_params_grid)

        self.overwrite_check = QCheckBox("Overwrite existing")
        overwrite_grid = block_grid(horizontal_spacing=12)
        add_block_checkbox_row(overwrite_grid, 0, self.overwrite_check)
        gen_lay.addLayout(overwrite_grid)

        self.gen_tabs = QTabWidget()

        # Tab: Tuning
        tuning_tab = QWidget()
        tuning_lay = QVBoxLayout(tuning_tab)

        tuning_params_grid = block_grid(horizontal_spacing=12)

        self.single_seed_dist = QSpinBox()
        self.single_seed_dist.setRange(1, 500)
        self.single_seed_dist.setValue(10)
        self.single_fg_threshold = QDoubleSpinBox()
        self.single_fg_threshold.setRange(0.01, 0.99)
        self.single_fg_threshold.setValue(0.5)
        self.single_fg_threshold.setDecimals(2)
        self.single_fg_threshold.setSingleStep(0.05)
        self.single_fg_threshold.setToolTip(
            "Sigmoid foreground probability cutoff — pixels below this are excluded from segmentation and seeding."
        )
        add_block_pair_row(
            tuning_params_grid,
            0,
            "Seed Distance:",
            _compact(self.single_seed_dist),
            "Foreground Threshold:",
            _compact(self.single_fg_threshold),
        )

        self.single_ridge_threshold = QDoubleSpinBox()
        self.single_ridge_threshold.setRange(0.0, 1.0)
        self.single_ridge_threshold.setValue(0.5)
        self.single_ridge_threshold.setDecimals(2)
        self.single_ridge_threshold.setSingleStep(0.05)
        self.single_ridge_threshold.setToolTip(
            "Contour boundary fraction cutoff — pixels with boundary ≥ this are carved out of the seeding mask."
        )
        add_block_pair_row(
            tuning_params_grid,
            1,
            "Ridge Threshold:",
            _compact(self.single_ridge_threshold),
        )
        tuning_lay.addLayout(tuning_params_grid)

        self.preview_btn = QPushButton("Preview")
        self.save_db_btn = QPushButton("Save to DB")
        for button in (self.preview_btn, self.save_db_btn):
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        tuning_params_grid.addWidget(QLabel(""), 2, 0)
        tuning_params_grid.addWidget(
            self.preview_btn,
            2,
            1,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        tuning_params_grid.addWidget(QLabel(""), 2, 2)
        tuning_params_grid.addWidget(
            self.save_db_btn,
            2,
            3,
            1,
            1,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        self.gen_tabs.addTab(tuning_tab, "Tuning")

        # Tab: Sweep
        sweep_tab = QWidget()
        sweep_lay = QVBoxLayout(sweep_tab)

        def _make_sweep_spins(d_min, d_max, d_step, decimals=0):
            make = QDoubleSpinBox if decimals > 0 else QSpinBox
            min_s, max_s, step_s = make(), make(), make()
            for s in (min_s, max_s, step_s):
                s.setRange(1 if decimals == 0 else 0.0, 500 if decimals == 0 else 20.0)
                if decimals > 0:
                    s.setDecimals(decimals)
                s.setMaximumWidth(62)
                s.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            min_s.setValue(d_min)
            max_s.setValue(d_max)
            step_s.setValue(d_step)
            return min_s, max_s, step_s

        sweep_grid = sweep_parameter_grid()
        sd_min, sd_max, sd_step = _make_sweep_spins(8, 14, 2)
        add_sweep_parameter_row(sweep_grid, 1, "Seed Dist:", sd_min, sd_max, sd_step)
        self.sweep_seed_dist = (sd_min, sd_max, sd_step)

        fg_min, fg_max, fg_step = _make_sweep_spins(0.4, 0.6, 0.05, decimals=2)
        add_sweep_parameter_row(
            sweep_grid, 2, "Foreground Threshold:", fg_min, fg_max, fg_step
        )
        self.sweep_fg_thr = (fg_min, fg_max, fg_step)

        ridge_min, ridge_max, ridge_step = _make_sweep_spins(0.5, 0.5, 0.05, decimals=2)
        add_sweep_parameter_row(
            sweep_grid, 3, "Ridge Threshold:", ridge_min, ridge_max, ridge_step
        )
        self.sweep_ridge_thr = (ridge_min, ridge_max, ridge_step)

        sweep_lay.addLayout(sweep_grid)

        sweep_runs_row = block_grid(horizontal_spacing=12)
        self.sweep_n_runs = QSpinBox()
        self.sweep_n_runs.setRange(1, 100)
        self.sweep_n_runs.setValue(1)
        self.sweep_n_runs.setToolTip(
            "How many times to run the sweep. With noise > 0 each run produces "
            "different stochastic hypotheses stored as separate parameter sets."
        )
        self.sweep_n_workers = QSpinBox()
        self.sweep_n_workers.setRange(1, max(1, os.cpu_count() or 1))
        self.sweep_n_workers.setValue(1)
        self.sweep_n_workers.setToolTip(
            "Number of parallel threads for the sweep. "
            "scipy/skimage release the GIL so threading scales well."
        )
        add_block_pair_row(
            sweep_runs_row,
            0,
            "Runs:",
            _compact(self.sweep_n_runs),
            "Workers:",
            _compact(self.sweep_n_workers),
        )
        sweep_lay.addLayout(sweep_runs_row)

        sweep_btn_row = block_grid(horizontal_spacing=12)
        self.run_sweep_btn    = QPushButton("Run Sweep")
        self.run_terminal_btn = QPushButton("Run in Terminal")
        self.cancel_sweep_btn = QPushButton("Cancel")
        self.cancel_sweep_btn.setEnabled(False)
        add_block_button_row(
            sweep_btn_row, 0, self.run_sweep_btn, self.run_terminal_btn, self.cancel_sweep_btn
        )
        sweep_lay.addLayout(sweep_btn_row)
        self.gen_tabs.addTab(sweep_tab, "Sweep")

        gen_lay.addWidget(self.gen_tabs)
        self.gen_section = CollapsibleSection(
            "2. Hypothesis Generation", _gen_inner, expanded=False
        )
        layout.addWidget(self.gen_section)

        # ── 3. Database Browser ──────────────────────────────────────────
        _db_inner = QWidget()
        db_lay = QVBoxLayout(_db_inner)
        db_lay.setContentsMargins(4, 4, 4, 4)
        db_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        hdr_row = block_grid(horizontal_spacing=12)
        self.db_activate_btn = QPushButton("Activate")
        self.db_activate_btn.setCheckable(True)
        self.db_activate_btn.setChecked(False)
        self.db_activate_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.db_refresh_btn = QPushButton()
        self.db_refresh_btn.setToolTip("Refresh database browser")
        self.db_refresh_btn.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_BrowserReload)
        )
        self.db_refresh_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        add_block_button_row(hdr_row, 0, self.db_activate_btn, self.db_refresh_btn)
        db_lay.addLayout(hdr_row)

        # ── contour_watershed params panel ────────────────────────────────
        self._db_cw_panel = QWidget()
        cw_lay = block_grid(horizontal_spacing=12)
        self._db_cw_panel.setLayout(cw_lay)
        cw_lay.setContentsMargins(0, 0, 0, 0)
        self.db_seed_dist_spin = QSpinBox()
        self.db_seed_dist_spin.setRange(1, 500)
        self.db_seed_dist_spin.setValue(10)
        self.db_seed_dist_spin.setEnabled(False)
        self.db_fg_thr_spin = QDoubleSpinBox()
        self.db_fg_thr_spin.setRange(0.01, 0.99)
        self.db_fg_thr_spin.setValue(0.5)
        self.db_fg_thr_spin.setDecimals(2)
        self.db_fg_thr_spin.setSingleStep(0.05)
        self.db_fg_thr_spin.setEnabled(False)
        add_block_pair_row(
            cw_lay,
            0,
            "Seed Dist:",
            _compact(self.db_seed_dist_spin),
            "FG Thr:",
            _compact(self.db_fg_thr_spin),
        )
        self.db_ridge_thr_spin = QDoubleSpinBox()
        self.db_ridge_thr_spin.setRange(0.0, 1.0)
        self.db_ridge_thr_spin.setValue(0.5)
        self.db_ridge_thr_spin.setDecimals(2)
        self.db_ridge_thr_spin.setSingleStep(0.05)
        self.db_ridge_thr_spin.setEnabled(False)
        self.db_run_spin = QSpinBox()
        self.db_run_spin.setRange(0, 99)
        self.db_run_spin.setValue(0)
        self.db_run_spin.setEnabled(False)
        add_block_pair_row(
            cw_lay,
            1,
            "Ridge Thr:",
            _compact(self.db_ridge_thr_spin),
            "Run:",
            _compact(self.db_run_spin),
        )
        db_lay.addWidget(self._db_cw_panel)

        self.db_info_lbl = QLabel("—")
        self.db_info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        db_lay.addWidget(self.db_info_lbl)

        db_btn_row = block_grid(horizontal_spacing=12)
        self.set_seed_btn = QPushButton("Set as Tracking Seed")
        add_block_button_row(db_btn_row, 0, self.set_seed_btn)
        db_lay.addLayout(db_btn_row)

        db_del_row = block_grid(horizontal_spacing=12)
        self.del_stack_btn = QPushButton("Remove Stack")
        danger_button(self.del_stack_btn)
        add_block_button_row(db_del_row, 0, self.del_stack_btn)
        db_lay.addLayout(db_del_row)
        self.db_section = CollapsibleSection(
            "3. Database Browser", _db_inner, expanded=False
        )
        layout.addWidget(self.db_section)

        # ── 4. Tracking & Correction ───────────────────────────────────────
        _tracking_correction_inner = QWidget()
        tracking_correction_lay = QVBoxLayout(_tracking_correction_inner)
        tracking_correction_lay.setContentsMargins(4, 4, 4, 4)
        tracking_correction_lay.setSpacing(4)
        tracking_correction_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        _ultrack_inner = QWidget()
        ultrack_lay = QVBoxLayout(_ultrack_inner)
        ultrack_lay.setContentsMargins(0, 0, 0, 0)
        ultrack_lay.setSpacing(4)
        ultrack_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        tracking_grid = block_grid(horizontal_spacing=12)
        tracking_grid.setContentsMargins(0, 0, 0, 0)

        self.ultrack_min_area_spin = QSpinBox()
        self.ultrack_min_area_spin.setRange(0, 100000)
        self.ultrack_min_area_spin.setValue(300)
        self.ultrack_min_area_spin.setSingleStep(50)

        self.ultrack_max_partitions_spin = QSpinBox()
        self.ultrack_max_partitions_spin.setRange(0, 1000)
        self.ultrack_max_partitions_spin.setValue(30)
        self.ultrack_max_partitions_spin.setToolTip("0 = use all partitions")

        self.ultrack_n_frames_spin = QSpinBox()
        self.ultrack_n_frames_spin.setRange(0, 10000)
        self.ultrack_n_frames_spin.setValue(0)
        self.ultrack_n_frames_spin.setToolTip("0 = process all frames")

        self.ultrack_linking_mode_combo = QComboBox()
        self.ultrack_linking_mode_combo.addItems(["default", "iou"])

        self.ultrack_max_dist_spin = QDoubleSpinBox()
        self.ultrack_max_dist_spin.setRange(0.0, 500.0)
        self.ultrack_max_dist_spin.setValue(15.0)
        self.ultrack_max_dist_spin.setSingleStep(1.0)
        self.ultrack_max_dist_spin.setDecimals(1)

        self.ultrack_iou_weight_spin = QDoubleSpinBox()
        self.ultrack_iou_weight_spin.setRange(0.0, 1.0)
        self.ultrack_iou_weight_spin.setValue(1.0)
        self.ultrack_iou_weight_spin.setSingleStep(0.05)
        self.ultrack_iou_weight_spin.setDecimals(2)

        self.ultrack_appear_spin = QDoubleSpinBox()
        self.ultrack_appear_spin.setRange(-10.0, 0.0)
        self.ultrack_appear_spin.setValue(-0.1)
        self.ultrack_appear_spin.setSingleStep(0.05)
        self.ultrack_appear_spin.setDecimals(3)

        self.ultrack_disappear_spin = QDoubleSpinBox()
        self.ultrack_disappear_spin.setRange(-10.0, 0.0)
        self.ultrack_disappear_spin.setValue(-0.1)
        self.ultrack_disappear_spin.setSingleStep(0.05)
        self.ultrack_disappear_spin.setDecimals(3)

        self.ultrack_division_spin = QDoubleSpinBox()
        self.ultrack_division_spin.setRange(-10.0, 0.0)
        self.ultrack_division_spin.setValue(-0.001)
        self.ultrack_division_spin.setSingleStep(0.05)
        self.ultrack_division_spin.setDecimals(3)
        self.ultrack_division_spin.setToolTip(
            "ILP penalty for cell division events. More negative = fewer divisions allowed."
        )

        self.ultrack_max_neighbors_spin = QSpinBox()
        self.ultrack_max_neighbors_spin.setRange(1, 50)
        self.ultrack_max_neighbors_spin.setValue(5)
        self.ultrack_max_neighbors_spin.setToolTip(
            "Maximum number of candidate predecessor nodes considered during linking."
        )

        self.ultrack_power_spin = QDoubleSpinBox()
        self.ultrack_power_spin.setRange(0.1, 20.0)
        self.ultrack_power_spin.setValue(4.0)
        self.ultrack_power_spin.setSingleStep(0.5)
        self.ultrack_power_spin.setDecimals(2)
        self.ultrack_power_spin.setToolTip(
            "Ultrack's solver transform for node_prob and link weights. "
            "With link_function=power, stored weights are raised to this power during solving."
        )

        self.ultrack_quality_exp_spin = QDoubleSpinBox()
        self.ultrack_quality_exp_spin.setRange(0.1, 50.0)
        self.ultrack_quality_exp_spin.setValue(8.0)
        self.ultrack_quality_exp_spin.setSingleStep(0.5)
        self.ultrack_quality_exp_spin.setDecimals(2)
        self.ultrack_quality_exp_spin.setToolTip(
            "Raises the signal-based segmentation quality before storing it as node_prob. "
            "Higher values favor high-confidence whole-object candidates over fragments."
        )

        self.ultrack_seed_weight_spin = QDoubleSpinBox()
        self.ultrack_seed_weight_spin.setRange(0.0, 10.0)
        self.ultrack_seed_weight_spin.setValue(0.5)
        self.ultrack_seed_weight_spin.setSingleStep(0.1)
        self.ultrack_seed_weight_spin.setDecimals(2)
        self.ultrack_seed_weight_spin.setToolTip(
            "Additive reward for candidates similar to nearby validated cells. "
            "Zero disables the seed-local bonus."
        )

        self.ultrack_seed_space_spin = QDoubleSpinBox()
        self.ultrack_seed_space_spin.setRange(1.0, 500.0)
        self.ultrack_seed_space_spin.setValue(25.0)
        self.ultrack_seed_space_spin.setSingleStep(5.0)
        self.ultrack_seed_space_spin.setDecimals(1)
        self.ultrack_seed_space_spin.setToolTip(
            "Spatial decay scale for seed proximity. Larger values let validated cells influence candidates farther away."
        )

        self.ultrack_seed_time_spin = QDoubleSpinBox()
        self.ultrack_seed_time_spin.setRange(0.1, 50.0)
        self.ultrack_seed_time_spin.setValue(2.0)
        self.ultrack_seed_time_spin.setSingleStep(0.5)
        self.ultrack_seed_time_spin.setDecimals(1)
        self.ultrack_seed_time_spin.setToolTip(
            "Temporal decay scale in frames. Larger values let validated cells influence more distant frames within the seed window."
        )

        self.ultrack_seed_window_spin = QSpinBox()
        self.ultrack_seed_window_spin.setRange(0, 100)
        self.ultrack_seed_window_spin.setValue(5)
        self.ultrack_seed_window_spin.setToolTip(
            "Maximum frame distance from a validated cell used for seed affinity."
        )

        self.ultrack_solver_lbl = QLabel("—")
        add_block_pair_row(
            tracking_grid,
            0,
            "Min Area (px):",
            _compact(self.ultrack_min_area_spin, 80),
            "Appear Penalty:",
            _compact(self.ultrack_appear_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            tracking_grid,
            1,
            "Max Partitions/frame:",
            _compact(self.ultrack_max_partitions_spin, 80),
            "Disappear Penalty:",
            _compact(self.ultrack_disappear_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            tracking_grid,
            2,
            "First N frames:",
            _compact(self.ultrack_n_frames_spin, 80),
            "Division Penalty:",
            _compact(self.ultrack_division_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            tracking_grid,
            3,
            "Linking Mode:",
            self.ultrack_linking_mode_combo,
            "Max Neighbors:",
            _compact(self.ultrack_max_neighbors_spin, 80),
            field_width=None,
        )
        add_block_pair_row(
            tracking_grid,
            4,
            "Max Distance (px):",
            _compact(self.ultrack_max_dist_spin, 80),
            "Solver:",
            self.ultrack_solver_lbl,
            field_width=None,
        )
        add_block_pair_row(
            tracking_grid,
            5,
            "IoU Weight:",
            _compact(self.ultrack_iou_weight_spin, 80),
            field_width=80,
        )
        ultrack_lay.addLayout(tracking_grid)

        self.ultrack_route_check = QCheckBox("Resolve from validated")
        route_grid = block_grid(horizontal_spacing=12)
        add_block_checkbox_row(route_grid, 0, self.ultrack_route_check)
        ultrack_lay.addLayout(route_grid)

        resolve_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(
            resolve_grid,
            0,
            "Ultrack Power:",
            _compact(self.ultrack_power_spin, 80),
            "Quality Exp:",
            _compact(self.ultrack_quality_exp_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            resolve_grid,
            1,
            "Seed Weight:",
            _compact(self.ultrack_seed_weight_spin, 80),
            "Seed Space (px):",
            _compact(self.ultrack_seed_space_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            resolve_grid,
            2,
            "Seed Time:",
            _compact(self.ultrack_seed_time_spin, 80),
            "Seed Window:",
            _compact(self.ultrack_seed_window_spin, 80),
            field_width=80,
        )
        ultrack_lay.addLayout(resolve_grid)

        ultrack_run_row = block_grid(horizontal_spacing=12)
        self.run_ultrack_btn = QPushButton("Run Ultrack Tracking")
        self.ultrack_terminal_btn = QPushButton("Run in Terminal")
        add_block_button_row(ultrack_run_row, 0, self.run_ultrack_btn, self.ultrack_terminal_btn)
        ultrack_lay.addLayout(ultrack_run_row)

        self.ultrack_status_lbl = QLabel("")
        self.ultrack_status_lbl.setWordWrap(True)
        ultrack_lay.addWidget(self.ultrack_status_lbl)

        self.ultrack_progress_bar = QProgressBar()
        self.ultrack_progress_bar.setRange(0, 100)
        self.ultrack_progress_bar.setValue(0)
        self.ultrack_progress_bar.setVisible(False)
        ultrack_lay.addWidget(self.ultrack_progress_bar)

        self.ultrack_section = CollapsibleSection(
            "Ultrack Tracking", _ultrack_inner, expanded=False
        )
        tracking_correction_lay.addWidget(self.ultrack_section)

        _corr_inner = QWidget()
        _corr_inner_lay = QVBoxLayout(_corr_inner)
        _corr_inner_lay.setContentsMargins(0, 0, 0, 0)
        _corr_inner_lay.setSpacing(4)

        extend_row = block_grid(horizontal_spacing=12)
        self.extend_back_btn = QPushButton("◀ Extend (A)")
        self.extend_fwd_btn = QPushButton("Extend (D) ▶")
        add_block_button_row(extend_row, 0, self.extend_back_btn, self.extend_fwd_btn)
        _corr_inner_lay.addLayout(extend_row)

        retrack_row = block_grid(horizontal_spacing=12)
        self.retrack_back_btn = QPushButton("◀ Retrack (Q)")
        self.retrack_fwd_btn = QPushButton("Retrack (E) ▶")
        add_block_button_row(retrack_row, 0, self.retrack_back_btn, self.retrack_fwd_btn)
        _corr_inner_lay.addLayout(retrack_row)

        save_load_row = block_grid(horizontal_spacing=12)
        self.save_tracked_btn = QPushButton("Save Tracked Labels")
        self.load_tracked_btn = QPushButton("Load Tracked Labels")
        add_block_button_row(save_load_row, 0, self.save_tracked_btn, self.load_tracked_btn)
        _corr_inner_lay.addLayout(save_load_row)

        reassign_row = block_grid(horizontal_spacing=12)
        self.reassign_ids_btn = QPushButton("Reassign IDs")
        add_block_button_row(reassign_row, 0, self.reassign_ids_btn)
        _corr_inner_lay.addLayout(reassign_row)

        extend_params_inner = QWidget()
        extend_params_lay = QVBoxLayout(extend_params_inner)
        extend_params_lay.setContentsMargins(0, 0, 0, 0)
        extend_params_lay.setSpacing(4)
        extend_params_form = block_grid(horizontal_spacing=12)
        self.extend_max_dist_spin = QDoubleSpinBox()
        self.extend_max_dist_spin.setRange(0.0, 500.0)
        self.extend_max_dist_spin.setValue(40.0)
        self.extend_max_dist_spin.setSingleStep(1.0)
        self.extend_max_dist_spin.setDecimals(1)
        add_block_pair_row(
            extend_params_form,
            0,
            "Max Distance (px):",
            _compact(self.extend_max_dist_spin, 80),
            field_width=80,
        )
        extend_params_lay.addLayout(extend_params_form)
        self.extend_params_section = CollapsibleSection(
            "Extend Parameters", extend_params_inner, expanded=False
        )
        _corr_inner_lay.addWidget(self.extend_params_section)

        retrack_params_inner = QWidget()
        retrack_params_lay = QVBoxLayout(retrack_params_inner)
        retrack_params_lay.setContentsMargins(0, 0, 0, 0)
        retrack_params_lay.setSpacing(4)
        retrack_params_form = block_grid(horizontal_spacing=12)
        self.retrack_max_dist_spin = QDoubleSpinBox()
        self.retrack_max_dist_spin.setRange(0.0, 500.0)
        self.retrack_max_dist_spin.setValue(20.0)
        self.retrack_max_dist_spin.setSingleStep(1.0)
        self.retrack_max_dist_spin.setDecimals(1)
        add_block_pair_row(
            retrack_params_form,
            0,
            "Max Distance (px):",
            _compact(self.retrack_max_dist_spin, 80),
            field_width=80,
        )
        retrack_params_lay.addLayout(retrack_params_form)
        self.retrack_params_section = CollapsibleSection(
            "Retrack Parameters", retrack_params_inner, expanded=False
        )
        _corr_inner_lay.addWidget(self.retrack_params_section)

        self.validation_counter_lbl = QLabel("")
        self.validation_counter_lbl.setWordWrap(True)
        _corr_inner_lay.addWidget(self.validation_counter_lbl)

        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
        )
        self.correction_widget.set_edit_callback(self._on_cells_edited)
        _corr_inner_lay.addWidget(self.correction_widget)
        self.correction_shortcuts_section = CollapsibleSection(
            "Correction Shortcuts",
            self.correction_widget.build_shortcuts_widget(),
            expanded=True,
        )
        _corr_inner_lay.addWidget(self.correction_shortcuts_section)

        self.correction_section = CollapsibleSection(
            "Correction", _corr_inner, expanded=False
        )
        tracking_correction_lay.addWidget(self.correction_section)

        self.tracking_correction_section = CollapsibleSection(
            "4. Tracking & Correction", _tracking_correction_inner, expanded=False
        )
        layout.addWidget(self.tracking_correction_section)

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
        self.db_ridge_thr_spin.valueChanged.connect(self._on_db_param_changed)
        self.db_run_spin.valueChanged.connect(self._on_db_param_changed)
        self.set_seed_btn.clicked.connect(self._on_set_seed)
        self.db_activate_btn.toggled.connect(self._on_db_activate_toggled)
        self.db_refresh_btn.clicked.connect(lambda: self._refresh_db_browser())
        self.del_stack_btn.clicked.connect(self._on_remove_stack)
        self.run_ultrack_btn.clicked.connect(self._on_run_tracking_route)
        self.ultrack_terminal_btn.clicked.connect(self._on_run_tracking_route_terminal)
        self.save_tracked_btn.clicked.connect(self._on_save_tracked)
        self.load_tracked_btn.clicked.connect(self._on_load_tracked)
        self.reassign_ids_btn.clicked.connect(self._on_reassign_ids)
        self.ultrack_linking_mode_combo.currentTextChanged.connect(self._on_ultrack_mode_changed)
        self.ultrack_route_check.toggled.connect(self._set_resolve_prior_controls_enabled)
        self.retrack_back_btn.clicked.connect(self._on_retrack_backward)
        self.retrack_fwd_btn.clicked.connect(self._on_retrack_forward)
        self.extend_back_btn.clicked.connect(self._on_extend_backward)
        self.extend_fwd_btn.clicked.connect(self._on_extend_forward)
        self.viewer.dims.events.current_step.connect(self._on_dims_step_changed)
        self.viewer.bind_key("V", self._kb_toggle_cell_validation, overwrite=True)
        self._install_correction_shortcuts()
        self.correction_widget._activate_btn.toggled.connect(self._on_correction_mode_toggled)
        # Set initial state for solver label and IoU weight enablement
        solver = _select_solver()
        solver_display = "Gurobi (licensed)" if solver == "GUROBI" else "CBC"
        self.ultrack_solver_lbl.setText(solver_display)
        self._on_ultrack_mode_changed(self.ultrack_linking_mode_combo.currentText())
        self._set_resolve_prior_controls_enabled(self.ultrack_route_check.isChecked())

    # ──────────────────────────────────────────────────────────────────────────
    # Public refresh
    # ──────────────────────────────────────────────────────────────────────────

    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self.input_files.refresh(pos_dir)
        self.contour_files.refresh(pos_dir)
        self.output_files.refresh(pos_dir)
        if pos_dir is None:
            self.correction_widget.deactivate()
            return
        self._refresh_db_browser()
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

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

    def _cellprob_zavg_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_prob_zavg.tif" if self._pos_dir else None

    def _ultrack_workdir(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "ultrack_workdir" if self._pos_dir else None

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
    ) -> None:
        if _TRACKED_LAYER in self.viewer.layers and t is not None:
            layer = self.viewer.layers[_TRACKED_LAYER]
            if layer.data.ndim == 3:
                if t < layer.data.shape[0]:
                    new_data = layer.data.copy()
                    new_data[t] = labels
                    layer.data = new_data
                    return
                # Extend the in-memory stack rather than reloading from disk.
                new_data = np.concatenate(
                    [layer.data, labels[np.newaxis].astype(layer.data.dtype)], axis=0
                )
                layer.data = new_data
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
        self._db_ridge_thr_vals = []
        self._db_run_vals = []
        self._current_db_p = None
        self.db_seed_dist_spin.setEnabled(False)
        self.db_fg_thr_spin.setEnabled(False)
        self.db_ridge_thr_spin.setEnabled(False)
        self.db_run_spin.setEnabled(False)
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
            self.status_lbl.setText(f"Hypothesis DB: {n_p} parameter set(s) (no browsable entries).")
            return

        seed_dist_set: set[int] = set()
        fg_thr_set: set[float] = set()
        ridge_thr_set: set[float] = set()
        run_set: set[int] = set()
        for p_idx, info in contour_entries.items():
            d = int(info.get("seed_distance", 10))
            fg = round(float(info.get("foreground_threshold", 0.5)), 4)
            ridge = round(float(info.get("ridge_threshold", 0.5)), 4)
            run = int(info.get("run_index", 0))
            seed_dist_set.add(d)
            fg_thr_set.add(fg)
            ridge_thr_set.add(ridge)
            run_set.add(run)
            self._db_param_map[(d, fg, ridge, run)] = p_idx
        self._db_seed_dist_vals = sorted(seed_dist_set)
        self._db_fg_thr_vals = sorted(fg_thr_set)
        self._db_ridge_thr_vals = sorted(ridge_thr_set)
        self._db_run_vals = sorted(run_set)

        self._apply_method_panel()
        self.status_lbl.setText(f"Hypothesis DB: {n_p} parameter set(s).")

    def _apply_method_panel(self) -> None:
        if not self._db_seed_dist_vals:
            return

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

        self.db_ridge_thr_spin.blockSignals(True)
        self.db_ridge_thr_spin.setMinimum(self._db_ridge_thr_vals[0])
        self.db_ridge_thr_spin.setMaximum(self._db_ridge_thr_vals[-1])
        step_ridge = round(self._db_ridge_thr_vals[1] - self._db_ridge_thr_vals[0], 4) if len(self._db_ridge_thr_vals) > 1 else 0.05
        self.db_ridge_thr_spin.setSingleStep(step_ridge)
        self.db_ridge_thr_spin.setValue(self._db_ridge_thr_vals[0])
        self.db_ridge_thr_spin.setEnabled(True)
        self.db_ridge_thr_spin.blockSignals(False)

        self.db_run_spin.blockSignals(True)
        self.db_run_spin.setMinimum(self._db_run_vals[0])
        self.db_run_spin.setMaximum(self._db_run_vals[-1])
        self.db_run_spin.setSingleStep(1)
        self.db_run_spin.setValue(self._db_run_vals[0])
        self.db_run_spin.setEnabled(len(self._db_run_vals) > 1)
        self.db_run_spin.blockSignals(False)

        self._update_db_info_lbl()

    def _set_status(self, msg: str) -> None:
        self.status_lbl.setText(msg)

    def _set_ultrack_status(self, msg: str) -> None:
        self.ultrack_status_lbl.setText(msg)
        self._set_status(msg)
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
            ridge_threshold=self.single_ridge_threshold.value(),
            min_size=self.min_size_spin.value(),
            min_circularity=self.min_circularity_spin.value(),
            noise_scale=self.noise_scale.value(),
            noise_blur_sigma=self.noise_blur.value(),
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
            ridge_threshold=self.sweep_ridge_thr[0].value(),
            ridge_threshold_min=self.sweep_ridge_thr[0].value(),
            ridge_threshold_max=self.sweep_ridge_thr[1].value(),
            ridge_threshold_step=self.sweep_ridge_thr[2].value(),
            noise_scale=self.noise_scale.value(),
            noise_blur_sigma=self.noise_blur.value(),
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
        *,
        mask_callback=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from cellflow.segmentation import build_consensus_boundary

        boundary_sum  = None
        foreground_sum = None
        for g_idx, g in enumerate(gammas):
            cb = None
            if mask_callback is not None:
                def cb(masks, i_thresh, *, _gi=g_idx):
                    mask_callback(masks, _gi, i_thresh)
            b, fg = build_consensus_boundary(prob_3d, dp_3d, thresholds, gamma=g, mask_callback=cb)
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
        save_source     = self.save_source_check.isChecked()
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
            source_dir = pos_dir / "2_nucleus/source_labels"

            for t in range(n_t):
                yield (t + 1, n_t, f"Building contour maps: frame {t + 1}/{n_t}…")
                mask_cb = None
                if save_source:
                    source_dir.mkdir(parents=True, exist_ok=True)
                    def mask_cb(masks, g_idx, thresh_idx, *, _t=t):
                        tifffile.imwrite(
                            source_dir / f"masks_t{_t:04d}_g{g_idx:02d}_thr{thresh_idx:02d}.tif",
                            masks, compression="zlib",
                        )
                boundary, fg = build_fn(prob_stack[t], dp_stack[t], thresholds, gammas, mask_callback=mask_cb)
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
                ridge_threshold=params.ridge_threshold,
                ridge_threshold_min=params.ridge_threshold,
                ridge_threshold_max=params.ridge_threshold,
                ridge_threshold_step=0.05,
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
        n_workers = self.sweep_n_workers.value()
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
                iter_contour_watershed_records(contour_stack, foreground_stack, spec, n_workers=n_workers), 1
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
        n_workers = self.sweep_n_workers.value()
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
            f"    ridge_threshold={spec.ridge_threshold},\n"
            f"    ridge_threshold_min={spec.ridge_threshold_min},\n"
            f"    ridge_threshold_max={spec.ridge_threshold_max},\n"
            f"    ridge_threshold_step={spec.ridge_threshold_step},\n"
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
            f"for done, rec in enumerate(iter_contour_watershed_records(contour, foreground, spec, n_workers={n_workers}), 1):\n"
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
        if active and self._current_db_p is not None:
            self._load_db_stack(self._current_db_p)

    def _lookup_db_p(self) -> int | None:
        if not self._db_param_map:
            return None
        d = self.db_seed_dist_spin.value()
        fg = round(self.db_fg_thr_spin.value(), 4)
        ridge = round(self.db_ridge_thr_spin.value(), 4)
        run = self.db_run_spin.value()
        if self._db_seed_dist_vals:
            d = min(self._db_seed_dist_vals, key=lambda x: abs(x - d))
        if self._db_fg_thr_vals:
            fg = round(min(self._db_fg_thr_vals, key=lambda x: abs(x - fg)), 4)
        if self._db_ridge_thr_vals:
            ridge = round(min(self._db_ridge_thr_vals, key=lambda x: abs(x - ridge)), 4)
        if self._db_run_vals:
            run = min(self._db_run_vals, key=lambda x: abs(x - run))
        return self._db_param_map.get((d, fg, ridge, run))

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
        cell_zavg_path = self._cell_zavg_path()
        nuc_zavg_path  = self._nucleus_zavg_path()
        self._set_status(f"Loading p={p}…")

        @thread_worker(connect={"returned": self._on_load_stack_done, "errored": self._on_worker_error})
        def _worker():
            stack = read_full_hypothesis_stack(hyp_path, p)
            cell_zavg = (
                np.asarray(tifffile.imread(str(cell_zavg_path)), dtype=np.float32)
                if cell_zavg_path and cell_zavg_path.exists() else None
            )
            nuc_zavg = (
                np.asarray(tifffile.imread(str(nuc_zavg_path)), dtype=np.float32)
                if nuc_zavg_path and nuc_zavg_path.exists() else None
            )
            return p, stack, cell_zavg, nuc_zavg

        _worker()

    def _on_load_stack_done(self, result: tuple) -> None:
        p, stack, cell_zavg, nuc_zavg = result
        if stack.ndim == 4:
            stack = stack[:, 0]  # contour_watershed stores (1, Y, X) per frame
        nt = stack.shape[0]
        if _HYP_LAYER in self.viewer.layers:
            self.viewer.layers[_HYP_LAYER].data = stack
        else:
            self.viewer.add_labels(stack, name=_HYP_LAYER)
        n_cells = int(stack.max()) if stack.size > 0 else 0
        self.db_info_lbl.setText(f"p={p:03d}  |  {n_cells} cells")
        self._set_status(f"Loaded p={p} → {stack.shape} into napari.")

        for zavg_data, layer_name, cmap in (
            (cell_zavg, _CELL_ZAVG_LAYER, "gray"),
            (nuc_zavg,  _NUC_ZAVG_LAYER,  "bop orange"),
        ):
            if zavg_data is None:
                continue
            if zavg_data.ndim == 2:
                zavg_data = np.broadcast_to(zavg_data[np.newaxis], (nt,) + zavg_data.shape).copy()
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = zavg_data
            else:
                self.viewer.add_image(zavg_data, name=layer_name, colormap=cmap, blending="additive")

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

    def _on_save_tracked(self) -> None:
        tracked_path = self._tracked_path()
        if tracked_path is None:
            self._set_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_status("No tracked layer to save.")
            return
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3:
            self._set_status("Tracked layer is not a 3D stack.")
            return
        n = layer.data.shape[0]
        for t in range(n):
            write_tracked_frame(tracked_path, t, np.asarray(layer.data[t]))
        self._set_status(f"Saved {n} frame(s) to {tracked_path.name}.")

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
        layer = self.viewer.layers[_TRACKED_LAYER]
        self.correction_widget.activate_layer(layer)
        self.correction_section.expand()

    def _on_reassign_ids(self) -> None:
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_status("No tracked layer loaded.")
            return
        stack = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)
        self._set_status("Reassigning cell IDs to contiguous range…")

        @thread_worker(connect={"returned": self._on_reassign_ids_done, "errored": self._on_worker_error})
        def _worker():
            unique_ids = np.unique(stack)
            unique_ids = unique_ids[unique_ids != 0]
            if unique_ids.size == 0:
                return stack, 0, {}
            lut = np.zeros(int(unique_ids.max()) + 1, dtype=np.uint32)
            old_to_new: dict[int, int] = {}
            for new_id, old_id in enumerate(unique_ids, start=1):
                lut[old_id] = new_id
                old_to_new[int(old_id)] = new_id
            return lut[stack], len(unique_ids), old_to_new

        _worker()

    def _on_reassign_ids_done(self, result: tuple) -> None:
        remapped, n_cells, old_to_new = result
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = remapped
        if self._pos_dir is not None and old_to_new:
            remap_validated_tracks(self._pos_dir, old_to_new)
        self._set_status(f"Reassigned {n_cells} cell IDs to contiguous range 1–{n_cells}. Unsaved.")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Tracking & Correction
    # ──────────────────────────────────────────────────────────────────────────

    def _on_ultrack_mode_changed(self, mode: str) -> None:
        self.ultrack_iou_weight_spin.setEnabled(mode == "iou")

    def _set_resolve_prior_controls_enabled(self, enabled: bool) -> None:
        for control in (
            self.ultrack_quality_exp_spin,
            self.ultrack_seed_weight_spin,
            self.ultrack_seed_space_spin,
            self.ultrack_seed_time_spin,
            self.ultrack_seed_window_spin,
        ):
            control.setEnabled(enabled)

    def _ultrack_config_from_controls(self) -> UltrackConfig:
        return UltrackConfig(
            min_area=self.ultrack_min_area_spin.value(),
            max_distance=self.ultrack_max_dist_spin.value(),
            linking_mode=self.ultrack_linking_mode_combo.currentText(),
            iou_weight=self.ultrack_iou_weight_spin.value(),
            appear_weight=self.ultrack_appear_spin.value(),
            disappear_weight=self.ultrack_disappear_spin.value(),
            division_weight=self.ultrack_division_spin.value(),
            max_neighbors=self.ultrack_max_neighbors_spin.value(),
            power=self.ultrack_power_spin.value(),
            quality_exponent=self.ultrack_quality_exp_spin.value(),
            seed_weight=self.ultrack_seed_weight_spin.value(),
            seed_sigma_space=self.ultrack_seed_space_spin.value(),
            seed_tau_time=self.ultrack_seed_time_spin.value(),
            seed_max_dt=self.ultrack_seed_window_spin.value(),
        )

    def _on_run_tracking_route(self) -> None:
        if self.ultrack_route_check.isChecked():
            self._on_resolve_with_validation()
        else:
            self._on_run_ultrack()

    def _on_run_tracking_route_terminal(self) -> None:
        if self.ultrack_route_check.isChecked():
            self._on_resolve_terminal()
        else:
            self._on_ultrack_terminal()

    def _on_run_ultrack(self) -> None:
        if self._pos_dir is None:
            self._set_ultrack_status("No project open.")
            return
        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_ultrack_status("Hypothesis DB not found — run the sweep first.")
            return
        cellprob_zavg_path = self._cellprob_zavg_path()
        if cellprob_zavg_path is None or not cellprob_zavg_path.exists():
            self._set_ultrack_status("Cellpose cellprob z-avg image not found — run Cellpose first.")
            return
        working_dir = self._ultrack_workdir()
        tracked_path = self._tracked_path()

        # Capture all widget values before entering the worker closure
        max_partitions_raw = self.ultrack_max_partitions_spin.value()
        max_partitions = None if max_partitions_raw == 0 else max_partitions_raw
        n_frames_raw = self.ultrack_n_frames_spin.value()
        n_frames = None if n_frames_raw == 0 else n_frames_raw
        cfg = self._ultrack_config_from_controls()

        self.ultrack_progress_bar.setRange(0, 100)
        self.ultrack_progress_bar.setVisible(True)
        self.ultrack_progress_bar.setValue(0)
        self._set_ultrack_status("Starting Ultrack tracking…")

        @thread_worker(connect={
            "yielded":  self._on_ultrack_progress,
            "returned": self._on_run_ultrack_done,
            "errored":  self._on_worker_error,
        })
        def _worker():
            # Stage 1: ingest
            yield ("ingest", 0, 3, "Ingesting hypotheses…")
            ingest_hypotheses_to_db(
                hyp_path, working_dir, cfg,
                overwrite=True,
                max_partitions=max_partitions,
                n_frames=n_frames,
            )

            yield ("score", 0, 1, "Scoring segmentation quality…")
            write_seed_prior_node_probs(working_dir, cellprob_zavg_path, cfg)

            # Stage 2: linking (IS a generator — relay each progress tuple)
            for step, total, label in run_linking(working_dir, cfg):
                yield ("link", step, total, label)

            # Stage 3: solve (IS a generator — relay each progress tuple)
            for step, total, label in run_solve(working_dir, cfg, overwrite=True):
                yield ("solve", step, total, label)

            # Stage 4: export
            yield ("export", 0, 1, "Exporting tracked labels…")
            return export_tracked_labels(working_dir, cfg, tracked_path)

        _worker()

    def _on_ultrack_progress(self, payload: tuple) -> None:
        stage, step, total, label = payload
        self._set_ultrack_status(f"[{stage}] {label}")
        if total > 0:
            self.ultrack_progress_bar.setValue(int(100 * step / total))

    def _on_run_ultrack_done(self, labels: np.ndarray | None) -> None:
        self.ultrack_progress_bar.setVisible(False)
        if labels is None:
            self._set_ultrack_status("Ultrack tracking failed (no output).")
            return
        # Normalize (T, 1, Y, X) → (T, Y, X)
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        nt = labels.shape[0]
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = labels
        else:
            self.viewer.add_labels(labels, name=_TRACKED_LAYER)
        layer = self.viewer.layers[_TRACKED_LAYER]
        self.correction_widget.activate_layer(layer)
        self._set_ultrack_status(f"Ultrack tracking done: {nt} frame(s). Unsaved.")

    def _on_ultrack_terminal(self) -> None:
        import sys
        import tempfile

        if self._pos_dir is None:
            self._set_ultrack_status("No project open.")
            return
        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_ultrack_status("Hypothesis DB not found — run the sweep first.")
            return
        cellprob_zavg_path = self._cellprob_zavg_path()
        if cellprob_zavg_path is None or not cellprob_zavg_path.exists():
            self._set_ultrack_status("Cellpose cellprob z-avg image not found — run Cellpose first.")
            return
        working_dir = self._ultrack_workdir()
        tracked_path = self._tracked_path()

        # Capture all widget values
        cfg = self._ultrack_config_from_controls()
        max_partitions_raw = self.ultrack_max_partitions_spin.value()
        max_partitions = None if max_partitions_raw == 0 else max_partitions_raw
        n_frames_raw = self.ultrack_n_frames_spin.value()
        n_frames = None if n_frames_raw == 0 else n_frames_raw

        # NOTE: body must live under `if __name__ == "__main__":` because
        # Ultrack's linker uses spawn-based multiprocessing, which re-executes
        # this script in each child via runpy with run_name="__mp_main__".
        # Without the guard, every worker re-runs the full pipeline and races
        # the parent on the SQLite DB.
        python_code = (
            "import sys, pathlib\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))\n"
            "from cellflow.tracking_ultrack.config import TrackingConfig\n"
            "from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db\n"
            "from cellflow.tracking_ultrack.linking import run_linking\n"
            "from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs\n"
            "from cellflow.tracking_ultrack.solve import run_solve\n"
            "from cellflow.tracking_ultrack.export import export_tracked_labels\n"
            "\n"
            "if __name__ == '__main__':\n"
            f"    hyp_path    = pathlib.Path({str(hyp_path)!r})\n"
            f"    working_dir = pathlib.Path({str(working_dir)!r})\n"
            f"    tracked_path= pathlib.Path({str(tracked_path)!r})\n"
            f"    cellprob_zavg_path = pathlib.Path({str(cellprob_zavg_path)!r})\n"
            f"    cfg = TrackingConfig(\n"
            f"        min_area={cfg.min_area},\n"
            f"        max_distance={cfg.max_distance},\n"
            f"        linking_mode={cfg.linking_mode!r},\n"
            f"        iou_weight={cfg.iou_weight},\n"
            f"        appear_weight={cfg.appear_weight},\n"
            f"        disappear_weight={cfg.disappear_weight},\n"
            f"        division_weight={cfg.division_weight},\n"
            f"        max_neighbors={cfg.max_neighbors},\n"
            f"        power={cfg.power},\n"
            f"        quality_exponent={cfg.quality_exponent},\n"
            f"        seed_weight={cfg.seed_weight},\n"
            f"        seed_sigma_space={cfg.seed_sigma_space},\n"
            f"        seed_tau_time={cfg.seed_tau_time},\n"
            f"        seed_max_dt={cfg.seed_max_dt},\n"
            f"    )\n"
            f"    max_partitions = {max_partitions!r}\n"
            f"    n_frames       = {n_frames!r}\n"
            "    print('[1/4] Ingesting…', flush=True)\n"
            "    ingest_hypotheses_to_db(hyp_path, working_dir, cfg, overwrite=True,\n"
            "        max_partitions=max_partitions, n_frames=n_frames)\n"
            "    print('[2/5] Scoring segmentation quality…', flush=True)\n"
            "    write_seed_prior_node_probs(working_dir, cellprob_zavg_path, cfg)\n"
            "    print('[3/5] Linking…', flush=True)\n"
            "    for step, total, label in run_linking(working_dir, cfg):\n"
            "        print(f'  [{step}/{total}] {label}', flush=True)\n"
            "    print('[4/5] Solving ILP…', flush=True)\n"
            "    for step, total, label in run_solve(working_dir, cfg, overwrite=True):\n"
            "        print(f'  [{step}/{total}] {label}', flush=True)\n"
            "    print('[5/5] Exporting…', flush=True)\n"
            "    labels = export_tracked_labels(working_dir, cfg, tracked_path)\n"
            f"    print(f'Done — {{labels.shape}} written to {{tracked_path}}', flush=True)\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="cellflow_ultrack_", delete=False
        ) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path)}"
        try:
            from cellflow.napari.utils import launch_in_terminal
            launch_in_terminal(cmd)
            self._set_ultrack_status("Ultrack command launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(cmd)
            self._set_ultrack_status("Copied Ultrack command to clipboard (terminal launch unavailable).")

    def _on_resolve_terminal(self) -> None:
        import sys
        import tempfile

        if self._pos_dir is None:
            self._set_ultrack_status("No project open.")
            return
        validated_tracks = read_validated_tracks(self._pos_dir)
        if not validated_tracks:
            self._set_status("No validated tracks — validate some cells first (press V).")
            return
        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_ultrack_status("hypotheses.h5 not found — generate hypotheses first.")
            return
        tracked_path = self._tracked_path()
        if tracked_path is None or not tracked_path.exists():
            self._set_ultrack_status("Tracked labels not found.")
            return
        cellprob_zavg_path = self._cellprob_zavg_path()
        if cellprob_zavg_path is None or not cellprob_zavg_path.exists():
            self._set_ultrack_status("Cellpose cellprob z-avg image not found — run Cellpose first.")
            return
        pos_dir = self._pos_dir

        # Capture widget values (same as _on_resolve_with_validation)
        cfg = self._ultrack_config_from_controls()

        python_code = (
            "import sys, pathlib\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))\n"
            "from cellflow.tracking_ultrack.config import TrackingConfig\n"
            "from cellflow.tracking_ultrack.reseed import resolve_with_validation\n"
            "from cellflow.database.tracked import read_full_tracked_stack, write_tracked_frame\n"
            "from cellflow.database.validation import (\n"
            "    read_validated_tracks, invalidate_track, validate_track)\n"
            "import numpy as np\n"
            "\n"
            "if __name__ == '__main__':\n"
            f"    pos_dir      = pathlib.Path({str(pos_dir)!r})\n"
            f"    hyp_path     = pathlib.Path({str(hyp_path)!r})\n"
            f"    tracked_path = pathlib.Path({str(tracked_path)!r})\n"
            f"    cellprob_zavg_path = pathlib.Path({str(cellprob_zavg_path)!r})\n"
            f"    cfg = TrackingConfig(\n"
            f"        min_area={cfg.min_area},\n"
            f"        max_distance={cfg.max_distance},\n"
            f"        linking_mode={cfg.linking_mode!r},\n"
            f"        iou_weight={cfg.iou_weight},\n"
            f"        appear_weight={cfg.appear_weight},\n"
            f"        disappear_weight={cfg.disappear_weight},\n"
            f"        division_weight={cfg.division_weight},\n"
            f"        max_neighbors={cfg.max_neighbors},\n"
            f"        power={cfg.power},\n"
            f"        quality_exponent={cfg.quality_exponent},\n"
            f"        seed_weight={cfg.seed_weight},\n"
            f"        seed_sigma_space={cfg.seed_sigma_space},\n"
            f"        seed_tau_time={cfg.seed_tau_time},\n"
            f"        seed_max_dt={cfg.seed_max_dt},\n"
            f"    )\n"
            "    validated_tracks = read_validated_tracks(pos_dir)\n"
            "    print(f'Loaded {len(validated_tracks)} validated track(s).', flush=True)\n"
            "    tracked_labels = read_full_tracked_stack(tracked_path)\n"
            "    print(f'Loaded tracked labels: {tracked_labels.shape}', flush=True)\n"
            "    new_labels, id_map = resolve_with_validation(\n"
            "        hyp_path, validated_tracks, tracked_labels, cfg,\n"
            "        progress_cb=lambda msg: print(msg, flush=True),\n"
            "        intensity_image_path=cellprob_zavg_path,\n"
            "    )\n"
            "    if new_labels.ndim == 4 and new_labels.shape[1] == 1:\n"
            "        new_labels = new_labels[:, 0]\n"
            "    print('Saving tracked labels…', flush=True)\n"
            "    for t in range(new_labels.shape[0]):\n"
            "        write_tracked_frame(tracked_path, t, np.asarray(new_labels[t]))\n"
            "    if id_map:\n"
            "        for old_id, new_id in id_map.items():\n"
            "            old_frames = validated_tracks.get(old_id, set())\n"
            "            invalidate_track(pos_dir, old_id)\n"
            "            if old_frames:\n"
            "                validate_track(pos_dir, new_id, old_frames)\n"
            "    n_validated = len(validated_tracks)\n"
            "    n_total = int(np.unique(new_labels[new_labels != 0]).size)\n"
            "    print(\n"
            "        f'Done — {n_validated} validated track(s) preserved, '\n"
            "        f'{n_total} total track(s). Saved to {tracked_path}',\n"
            "        flush=True,\n"
            "    )\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="cellflow_resolve_", delete=False
        ) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path)}"
        try:
            from cellflow.napari.utils import launch_in_terminal
            launch_in_terminal(cmd)
            self._set_ultrack_status("Re-solve command launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(cmd)
            self._set_ultrack_status("Copied Re-solve command to clipboard (terminal launch unavailable).")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Manual correction
    # ──────────────────────────────────────────────────────────────────────────

    def _on_dims_step_changed(self, event=None) -> None:
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    @staticmethod
    def _frame_view_2d(arr: np.ndarray, t: int) -> np.ndarray | None:
        """Return a 2D (Y, X) view of frame t from a (T, Y, X) or (T, 1, Y, X) stack."""
        if arr.ndim < 3 or t < 0 or t >= arr.shape[0]:
            return None
        v = arr[t]
        while v.ndim > 2:
            if v.shape[0] != 1:
                return None
            v = v[0]
        return v

    def _current_cell_ids(self, t: int) -> set[int]:
        """Return the set of non-zero cell IDs in the tracked layer at frame t."""
        if _TRACKED_LAYER not in self.viewer.layers:
            return set()
        layer = self.viewer.layers[_TRACKED_LAYER]
        frame = self._frame_view_2d(layer.data, t)
        if frame is None:
            return set()
        return set(int(v) for v in np.unique(frame)) - {0}

    def _refresh_validated_overlay(self) -> None:
        """Rebuild the green overlay layer from current frame's validated cells."""
        if self._pos_dir is None or _TRACKED_LAYER not in self.viewer.layers:
            if _VALIDATED_OVERLAY in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[_VALIDATED_OVERLAY])
            return
        tracked = self.viewer.layers[_TRACKED_LAYER]
        if tracked.data.ndim < 3:
            return
        t = self._current_t()
        if t >= tracked.data.shape[0]:
            return
        frame = self._frame_view_2d(tracked.data, t)
        if frame is None:
            return
        validated_ids = read_validated_cells_at_frame(self._pos_dir, t)
        overlay_exists = _VALIDATED_OVERLAY in self.viewer.layers
        if not validated_ids and not overlay_exists:
            # Nothing to draw and no overlay yet — skip creating one. This avoids
            # adding a layer during napari's own layer-insertion event chain
            # (which would re-enter and crash vispy's _reorder_layers).
            return
        if validated_ids:
            mask2d = np.isin(frame, list(validated_ids)).astype(np.uint8)
        else:
            mask2d = np.zeros(frame.shape, dtype=np.uint8)
        full = np.zeros(tracked.data.shape, dtype=np.uint8)
        full[t] = mask2d
        if overlay_exists:
            self.viewer.layers[_VALIDATED_OVERLAY].data = full
        else:
            from qtpy.QtCore import QTimer
            # Defer the add so we don't run inside napari's insert-event chain.
            QTimer.singleShot(0, lambda data=full: self._add_validated_overlay(data))

    def _add_validated_overlay(self, data: np.ndarray) -> None:
        if _VALIDATED_OVERLAY in self.viewer.layers:
            self.viewer.layers[_VALIDATED_OVERLAY].data = data
            return
        ov = self.viewer.add_labels(
            data,
            name=_VALIDATED_OVERLAY,
            opacity=1.0,
            colormap=direct_colormap({None: (0, 0, 0, 0), 1: "#00ff00"}),
        )
        # Send the active layer back to tracked so corrections still target it.
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[_TRACKED_LAYER]

    def _refresh_validation_counter(self) -> None:
        """Update 'N tracks validated, M cell-frames covered' label."""
        if self._pos_dir is None or _TRACKED_LAYER not in self.viewer.layers:
            self.validation_counter_lbl.setText("")
            return
        validated_tracks = read_validated_tracks(self._pos_dir)
        n_tracks = len(validated_tracks)
        n_cellframes = sum(len(frames) for frames in validated_tracks.values())
        self.validation_counter_lbl.setText(
            f"{n_tracks} track(s) validated, {n_cellframes} cell-frame(s) covered"
        )

    def _on_cells_edited(self, t: int, changed_ids: set[int]) -> None:
        """Callback registered with CorrectionWidget. Invalidate any edited cell IDs."""
        if self._pos_dir is None:
            return
        for cell_id in changed_ids:
            invalidate_track(self._pos_dir, cell_id)
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _frames_with_cell(self, cell_id: int) -> list[int]:
        """Return sorted list of frame indices where cell_id is present in the tracked layer."""
        if cell_id == 0 or _TRACKED_LAYER not in self.viewer.layers:
            return []
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim < 3:
            return []
        # Compare on the whole stack at once — np.any over the spatial axes is cheap.
        nt = layer.data.shape[0]
        spatial_axes = tuple(range(1, layer.data.ndim))
        present = np.any(layer.data == cell_id, axis=spatial_axes)
        return [int(t) for t in np.where(present)[0]]

    def _install_correction_shortcuts(self) -> None:
        specs = [
            ("A", lambda: self._on_extend(direction="backward")),
            ("D", lambda: self._on_extend(direction="forward")),
            ("Q", self._on_retrack_backward),
            ("E", self._on_retrack_forward),
        ]
        self._correction_shortcuts: list[QShortcut] = []
        for key, slot in specs:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.setEnabled(False)
            sc.activated.connect(slot)
            self._correction_shortcuts.append(sc)

    def _on_correction_mode_toggled(self, active: bool) -> None:
        for sc in self._correction_shortcuts:
            sc.setEnabled(active)

    def _kb_toggle_cell_validation(self, _viewer) -> None:
        if self._pos_dir is None:
            return
        sel = self.correction_widget._selected_label
        if not sel:
            self._set_status("Validation toggle: no cell selected (left-click a cell first).")
            return
        t = self._current_t()
        if sel not in self._current_cell_ids(t):
            self._set_status(f"Cell {sel} not present at t={t}.")
            return
        frames = self._frames_with_cell(sel)
        if not frames:
            return
        currently_validated = is_track_validated(self._pos_dir, sel)
        if currently_validated:
            invalidate_track(self._pos_dir, sel)
            self._set_status(f"Cell {sel} invalidated across {len(frames)} frame(s).")
        else:
            validate_track(self._pos_dir, sel, frames)
            self._set_status(f"Cell {sel} validated across {len(frames)} frame(s).")
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _on_extend_backward(self) -> None:
        self._on_extend(direction="backward")

    def _on_extend_forward(self) -> None:
        self._on_extend(direction="forward")

    def _on_extend(self, direction: str) -> None:
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_status("No tracked layer loaded.")
            return

        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_status("No project open.")
            return

        source_id = self.correction_widget._selected_label
        if not source_id:
            self._set_status("Extend: no cell selected (left-click a cell first).")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        t = self._current_t()
        tracked = np.asarray(layer.data)
        T = tracked.shape[0]

        target_frame = t + (1 if direction == "forward" else -1)
        if direction == "forward" and t >= T - 1:
            self._set_status("Already at last frame")
            return
        if direction == "backward" and t <= 0:
            self._set_status("Already at first frame")
            return

        if not np.any(tracked[t] == source_id):
            self._set_status(f"Cell {source_id} not present at t={t}")
            return

        result = extend_track(
            source_id=source_id,
            source_frame=t,
            direction=direction,
            tracked_labels=tracked,
            hypotheses_path=hyp_path,
            d_max=float(self.extend_max_dist_spin.value()),
        )

        if result is None:
            self._set_status(
                f"No hypothesis within {self.extend_max_dist_spin.value():g}px at t={target_frame}"
            )
            return

        frame = layer.data[result.target_frame]
        frame[frame == source_id] = 0
        paintable = result.mask_2d & (frame == 0)
        frame[paintable] = source_id
        layer.refresh()

        step = list(self.viewer.dims.current_step)
        step[0] = result.target_frame
        self.viewer.dims.current_step = tuple(step)

        self._set_status(
            f"Extended cell {source_id} → t={result.target_frame} "
            f"(dist={result.centroid_distance:.1f}px, area={result.area_ratio:.2f}, "
            f"overlap={result.existing_overlap:.2f})"
        )

    def _on_retrack_forward(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_status("No tracked layer loaded.")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._set_status("Tracked layer must be a stack of at least 2 frames.")
            return

        t0 = int(self.viewer.dims.current_step[0])
        if t0 >= layer.data.shape[0] - 1:
            self._set_status("Already at last frame — nothing to retrack forward.")
            return

        T = layer.data.shape[0]
        stack = layer.data.copy()
        fully_validated = read_validated_frames(self._pos_dir)
        reserved_ids = set(read_validated_tracks(self._pos_dir))

        n_retracked = 0
        n_skipped = 0
        for t in range(t0 + 1, T):
            if t in fully_validated:
                n_skipped += 1
                continue
            ref = stack[t - 1]
            tgt = stack[t]
            locked = read_validated_cells_at_frame(self._pos_dir, t)
            stack[t] = retrack_frame_constrained(
                ref,
                tgt,
                locked,
                max_dist_px=float(self.retrack_max_dist_spin.value()),
                reserved_ids=reserved_ids,
            )
            n_retracked += 1

        layer.data = stack
        self._set_status(
            f"Retracked forward from t={t0 + 1}: {n_retracked} frame(s) updated, "
            f"{n_skipped} fully-validated frame(s) skipped. Unsaved."
        )

    def _on_retrack_backward(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_status("No tracked layer loaded.")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._set_status("Tracked layer must be a stack of at least 2 frames.")
            return

        t0 = int(self.viewer.dims.current_step[0])
        if t0 <= 0:
            self._set_status("Already at first frame — nothing to retrack backward.")
            return

        stack = layer.data.copy()
        fully_validated = read_validated_frames(self._pos_dir)
        reserved_ids = set(read_validated_tracks(self._pos_dir))

        n_retracked = 0
        n_skipped = 0
        for t in range(t0 - 1, -1, -1):
            if t in fully_validated:
                n_skipped += 1
                continue
            ref = stack[t + 1]
            tgt = stack[t]
            locked = read_validated_cells_at_frame(self._pos_dir, t)
            stack[t] = retrack_frame_constrained(
                ref,
                tgt,
                locked,
                max_dist_px=float(self.retrack_max_dist_spin.value()),
                reserved_ids=reserved_ids,
            )
            n_retracked += 1

        layer.data = stack
        self._set_status(
            f"Retracked backward from t={t0 - 1}: {n_retracked} frame(s) updated, "
            f"{n_skipped} fully-validated frame(s) skipped. Unsaved."
        )

    def _on_resolve_with_validation(self) -> None:
        if self._pos_dir is None:
            self._set_ultrack_status("No project open.")
            return

        validated_tracks = read_validated_tracks(self._pos_dir)
        if not validated_tracks:
            self._set_ultrack_status("No validated tracks found — validate some cells first (press V).")
            return

        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_ultrack_status("No tracked layer loaded.")
            return

        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_ultrack_status("hypotheses.h5 not found — generate hypotheses first.")
            return
        cellprob_zavg_path = self._cellprob_zavg_path()
        if cellprob_zavg_path is None or not cellprob_zavg_path.exists():
            self._set_ultrack_status("Cellpose cellprob z-avg image not found — run Cellpose first.")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        tracked_labels = np.asarray(layer.data)

        cfg = self._ultrack_config_from_controls()

        n_validated = len(validated_tracks)
        self.run_ultrack_btn.setEnabled(False)
        self.ultrack_terminal_btn.setEnabled(False)
        self.ultrack_progress_bar.setRange(0, 0)
        self.ultrack_progress_bar.setVisible(True)
        self._set_ultrack_status(
            f"Re-solving with {n_validated} validated track(s) preserved…"
        )

        def _on_resolve_done(result: tuple) -> None:
            self.run_ultrack_btn.setEnabled(True)
            self.ultrack_terminal_btn.setEnabled(True)
            self.ultrack_progress_bar.setVisible(False)
            self.ultrack_progress_bar.setRange(0, 100)
            if result is None:
                self._set_ultrack_status("Re-solve failed (no output).")
                return
            new_labels, id_map = result
            # Normalize (T, 1, Y, X) → (T, Y, X) if needed
            if new_labels.ndim == 4 and new_labels.shape[1] == 1:
                new_labels = new_labels[:, 0]
            # Save labelmap to disk BEFORE updating the JSON so they stay in sync.
            # If the save fails, abort — the JSON must not be updated with IDs
            # that don't exist on disk.
            pos_dir = self._pos_dir
            tracked_path = self._tracked_path()
            if tracked_path is None or pos_dir is None:
                self._set_ultrack_status("Re-solve complete but no project path — not saved.")
                return
            self._set_ultrack_status("Saving tracked labels…")
            try:
                for t in range(new_labels.shape[0]):
                    write_tracked_frame(tracked_path, t, np.asarray(new_labels[t]))
            except Exception as exc:
                self._set_ultrack_status(f"Save failed — JSON not updated. {exc}")
                return
            # Labelmap is on disk with new IDs; now update the napari layer and JSON.
            if _TRACKED_LAYER in self.viewer.layers:
                self.viewer.layers[_TRACKED_LAYER].data = new_labels
            else:
                self.viewer.add_labels(new_labels, name=_TRACKED_LAYER)
            if id_map:
                for old_id, new_id in id_map.items():
                    old_frames = validated_tracks.get(old_id, set())
                    invalidate_track(pos_dir, old_id)
                    if old_frames:
                        validate_track(pos_dir, new_id, old_frames)
                self._refresh_validated_overlay()
                self._refresh_validation_counter()
            n_total_tracks = int(np.unique(new_labels[new_labels != 0]).size)
            self._set_ultrack_status(
                f"Re-solve complete: {n_validated} validated track(s) preserved, "
                f"{n_total_tracks} total track(s) in output. Saved."
            )

        def _on_resolve_progress(msg: str) -> None:
            self._set_ultrack_status(msg)

        def _on_resolve_error(exc: Exception) -> None:
            self.run_ultrack_btn.setEnabled(True)
            self.ultrack_terminal_btn.setEnabled(True)
            self.ultrack_progress_bar.setVisible(False)
            self.ultrack_progress_bar.setRange(0, 100)
            self._on_worker_error(exc)

        @thread_worker(connect={
            "returned": _on_resolve_done,
            "yielded":  _on_resolve_progress,
            "errored":  _on_resolve_error,
        })
        def _worker():
            status_msgs = []

            def _cb(msg: str) -> None:
                status_msgs.append(msg)

            # resolve_with_validation is not a generator, so we call it with a
            # progress_cb that collects messages.  After each internal stage the
            # callback appends a message; we yield them all once the function
            # returns so the UI gets updated between calls.  To emit progress
            # *during* the solve we run it in steps via the callback trick:
            # yield a sentinel before calling, collect inside.
            # Simpler approach: just yield the stage strings ourselves and call
            # resolve_with_validation with a progress_cb that does a thread-safe
            # yield via a queue.  But thread_worker yields must come from the
            # generator itself.  So we use the progress_cb to collect messages
            # and yield them after the call completes.
            # Best practical approach: call resolve_with_validation with
            # progress_cb that stores messages, and yield each after the call.
            # This gives incremental feedback between stages.

            import queue as _queue

            msg_queue: _queue.SimpleQueue = _queue.SimpleQueue()

            def _progress(msg: str) -> None:
                msg_queue.put(msg)

            import threading

            result_holder: list = []
            exc_holder: list = []

            def _run() -> None:
                try:
                    result_holder.append(
                        resolve_with_validation(
                            hyp_path, validated_tracks, tracked_labels, cfg,
                            progress_cb=_progress,
                            intensity_image_path=cellprob_zavg_path,
                        )
                    )
                except Exception as e:
                    exc_holder.append(e)

            t = threading.Thread(target=_run, daemon=True)
            t.start()

            while t.is_alive() or not msg_queue.empty():
                try:
                    yield msg_queue.get_nowait()
                except _queue.Empty:
                    t.join(timeout=0.05)

            if exc_holder:
                raise exc_holder[0]

            return result_holder[0] if result_holder else None

        _worker()

    # ──────────────────────────────────────────────────────────────────────────
    # Error handler
    # ──────────────────────────────────────────────────────────────────────────

    def _on_worker_error(self, exc: Exception) -> None:
        self.ultrack_progress_bar.setVisible(False)
        self._set_status(f"Error: {exc}")
        logger.exception("Worker error", exc_info=exc)

    # ──────────────────────────────────────────────────────────────────────────
    # State persistence
    # ──────────────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "overwrite":        self.overwrite_check.isChecked(),
            "save_source":      self.save_source_check.isChecked(),
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
                "seed_dist":        self.single_seed_dist.value(),
                "fg_threshold":     self.single_fg_threshold.value(),
                "ridge_threshold":  self.single_ridge_threshold.value(),
                "noise_scale":      self.noise_scale.value(),
                "noise_blur_sigma": self.noise_blur.value(),
            },
            "sweep": {
                "seed_dist_min":    self.sweep_seed_dist[0].value(),
                "seed_dist_max":    self.sweep_seed_dist[1].value(),
                "seed_dist_step":   self.sweep_seed_dist[2].value(),
                "fg_thr_min":       self.sweep_fg_thr[0].value(),
                "fg_thr_max":       self.sweep_fg_thr[1].value(),
                "fg_thr_step":      self.sweep_fg_thr[2].value(),
                "ridge_thr_min":    self.sweep_ridge_thr[0].value(),
                "ridge_thr_max":    self.sweep_ridge_thr[1].value(),
                "ridge_thr_step":   self.sweep_ridge_thr[2].value(),
                "noise_scale":      self.noise_scale.value(),
                "noise_blur_sigma": self.noise_blur.value(),
                "n_runs":           self.sweep_n_runs.value(),
                "n_workers":        self.sweep_n_workers.value(),
            },
            "db_browser": {
                "seed_dist":      self.db_seed_dist_spin.value(),
                "fg_threshold":   self.db_fg_thr_spin.value(),
                "ridge_threshold": self.db_ridge_thr_spin.value(),
                "run_index":      self.db_run_spin.value(),
            },
            "ultrack": {
                "min_area":         self.ultrack_min_area_spin.value(),
                "max_partitions":   self.ultrack_max_partitions_spin.value(),
                "n_frames":         self.ultrack_n_frames_spin.value(),
                "max_distance":     self.ultrack_max_dist_spin.value(),
                "linking_mode":     self.ultrack_linking_mode_combo.currentText(),
                "iou_weight":       self.ultrack_iou_weight_spin.value(),
                "appear_weight":    self.ultrack_appear_spin.value(),
                "disappear_weight": self.ultrack_disappear_spin.value(),
                "division_weight":  self.ultrack_division_spin.value(),
                "max_neighbors":    self.ultrack_max_neighbors_spin.value(),
                "resolve_only":     self.ultrack_route_check.isChecked(),
                "power":            self.ultrack_power_spin.value(),
                "quality_exponent": self.ultrack_quality_exp_spin.value(),
                "seed_weight":      self.ultrack_seed_weight_spin.value(),
                "seed_sigma_space": self.ultrack_seed_space_spin.value(),
                "seed_tau_time":    self.ultrack_seed_time_spin.value(),
                "seed_max_dt":      self.ultrack_seed_window_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if "overwrite" in state:
            self.overwrite_check.setChecked(state["overwrite"])
        if "save_source" in state:
            self.save_source_check.setChecked(state["save_source"])
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
            if "seed_dist"       in t: self.single_seed_dist.setValue(t["seed_dist"])
            if "fg_threshold"    in t: self.single_fg_threshold.setValue(t["fg_threshold"])
            if "ridge_threshold" in t: self.single_ridge_threshold.setValue(t["ridge_threshold"])
            if "noise_scale"     in t: self.noise_scale.setValue(t["noise_scale"])
            if "noise_blur_sigma" in t: self.noise_blur.setValue(t["noise_blur_sigma"])
        if "sweep" in state:
            sw = state["sweep"]
            if "seed_dist_min"  in sw: self.sweep_seed_dist[0].setValue(sw["seed_dist_min"])
            if "seed_dist_max"  in sw: self.sweep_seed_dist[1].setValue(sw["seed_dist_max"])
            if "seed_dist_step" in sw: self.sweep_seed_dist[2].setValue(sw["seed_dist_step"])
            if "fg_thr_min"     in sw: self.sweep_fg_thr[0].setValue(sw["fg_thr_min"])
            if "fg_thr_max"     in sw: self.sweep_fg_thr[1].setValue(sw["fg_thr_max"])
            if "fg_thr_step"    in sw: self.sweep_fg_thr[2].setValue(sw["fg_thr_step"])
            if "ridge_thr_min"  in sw: self.sweep_ridge_thr[0].setValue(sw["ridge_thr_min"])
            if "ridge_thr_max"  in sw: self.sweep_ridge_thr[1].setValue(sw["ridge_thr_max"])
            if "ridge_thr_step" in sw: self.sweep_ridge_thr[2].setValue(sw["ridge_thr_step"])
            if "noise_scale"    in sw: self.noise_scale.setValue(sw["noise_scale"])
            if "noise_blur_sigma" in sw: self.noise_blur.setValue(sw["noise_blur_sigma"])
            if "n_runs"         in sw: self.sweep_n_runs.setValue(sw["n_runs"])
            if "n_workers"      in sw: self.sweep_n_workers.setValue(sw["n_workers"])
        if "db_browser" in state:
            db = state["db_browser"]
            if "seed_dist"       in db: self.db_seed_dist_spin.setValue(db["seed_dist"])
            if "fg_threshold"    in db: self.db_fg_thr_spin.setValue(db["fg_threshold"])
            if "ridge_threshold" in db: self.db_ridge_thr_spin.setValue(db["ridge_threshold"])
            if "run_index"       in db: self.db_run_spin.setValue(db["run_index"])
        if "search" in state:
            pass  # Old propagator state — silently skip
        if "search_v2" in state:
            pass  # Old propagator v2 state — silently skip
        if "ultrack" in state:
            ul = state["ultrack"]
            if "min_area"         in ul: self.ultrack_min_area_spin.setValue(ul["min_area"])
            if "max_partitions"   in ul: self.ultrack_max_partitions_spin.setValue(ul["max_partitions"])
            if "n_frames"         in ul: self.ultrack_n_frames_spin.setValue(ul["n_frames"])
            if "max_distance"     in ul: self.ultrack_max_dist_spin.setValue(ul["max_distance"])
            if "linking_mode"     in ul:
                idx = self.ultrack_linking_mode_combo.findText(ul["linking_mode"])
                if idx >= 0:
                    self.ultrack_linking_mode_combo.setCurrentIndex(idx)
            if "iou_weight"       in ul: self.ultrack_iou_weight_spin.setValue(ul["iou_weight"])
            if "appear_weight"    in ul: self.ultrack_appear_spin.setValue(ul["appear_weight"])
            if "disappear_weight" in ul: self.ultrack_disappear_spin.setValue(ul["disappear_weight"])
            if "division_weight"  in ul: self.ultrack_division_spin.setValue(ul["division_weight"])
            if "max_neighbors"    in ul: self.ultrack_max_neighbors_spin.setValue(ul["max_neighbors"])
            if "resolve_only"     in ul: self.ultrack_route_check.setChecked(ul["resolve_only"])
            if "power"            in ul: self.ultrack_power_spin.setValue(ul["power"])
            if "quality_exponent" in ul: self.ultrack_quality_exp_spin.setValue(ul["quality_exponent"])
            if "seed_weight"      in ul: self.ultrack_seed_weight_spin.setValue(ul["seed_weight"])
            if "seed_sigma_space" in ul: self.ultrack_seed_space_spin.setValue(ul["seed_sigma_space"])
            if "seed_tau_time"    in ul: self.ultrack_seed_time_spin.setValue(ul["seed_tau_time"])
            if "seed_max_dt"      in ul: self.ultrack_seed_window_spin.setValue(ul["seed_max_dt"])
