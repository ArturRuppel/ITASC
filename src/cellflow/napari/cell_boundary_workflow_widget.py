"""Track-conditioned cell boundary selection widget for CellFlow.

Three-stage workflow:
  1. **Initialize** — compute geodesic unary costs and pairwise weights,
     build initial labels, display in the viewer.
  2. **Refine** — run N ICM sweeps on the current viewer labels (repeatable,
     interleaves with manual correction).
  3. **Commit** — write the current viewer labels to disk.
"""

from __future__ import annotations

import logging
import queue
import threading
from enum import Enum
from pathlib import Path
import os

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.napari.ui_style import (
    add_block_button_row,
    add_block_pair_row,
    block_grid,
    compact_spinbox,
    status_label,
)
from cellflow.segmentation import (
    apply_gamma,
    build_consensus_boundary_2d,
    build_consensus_boundary_flow_following,
    FlowFollowingParams,
)

logger = logging.getLogger(__name__)

_CELL_SEG_LAYER = "Cell Segmentation"
_CELL_CONTOUR_LAYER = "Contour Map: Cell"
_CELL_FOREGROUND_SCORE_LAYER = "Foreground Score: Cell"
_CELL_FOREGROUND_LAYER = "Foreground Mask: Cell"
_CONTOUR_SWEEP_WIDTH = 60


class ContourMethod(str, Enum):
    """Selectable contour-map creation strategy."""

    CELLPOSE = "Cellpose Native"
    FLOW_FOLLOWING = "Flow-Following (EDT Gravity)"


class CellBoundaryWorkflowWidget(QWidget):
    """Track-conditioned cell boundary selection workflow."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None

        # Worker references
        self._contour_worker = None
        self._initialize_worker = None
        self._refine_worker = None

        # Cached ICM state (set by Initialize, consumed by Refine)
        self._icm_state = None  # CellICMState | None

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        # ---- 1. Contour Maps ----
        self._setup_contour_section(layout)

        # ---- 2. Boundary Selection (Initialize → Refine → Commit) ----
        self._setup_boundary_selection_section(layout)

        # ---- 3. Correction ----
        self._setup_correction_section(layout)

        layout.addStretch()

    # -- Contour Maps section (unchanged from original) -------------------

    def _setup_contour_section(self, layout: QVBoxLayout) -> None:
        def _stage_files(group_label, entries):
            return PipelineFilesWidget(
                [(group_label, entries)], viewer=self.viewer
            )

        def _stage_status():
            lbl = QLabel("")
            lbl.setWordWrap(True)
            lbl.setVisible(False)
            status_label(lbl)
            return lbl

        def _stage_progress():
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setVisible(False)
            return bar

        def _spin_width(widget, width=_CONTOUR_SWEEP_WIDTH):
            widget.setMinimumWidth(width)
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
            )
            return widget

        def _param_group_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight: 600;")
            return lbl

        contour_inner = QWidget()
        contour_lay = QVBoxLayout(contour_inner)
        contour_lay.setContentsMargins(0, 0, 0, 0)
        contour_lay.setSpacing(4)
        contour_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.contour_input_files = _stage_files("Inputs", [
            ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
            ("3_cell/filtered_dp.tif", "Filtered flow vectors"),
        ])
        contour_lay.addWidget(self.contour_input_files)

        # ── Method selector ──────────────────────────────────────────────
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._contour_method_combo = QComboBox()
        for m in ContourMethod:
            self._contour_method_combo.addItem(m.value)
        self._contour_method_combo.setCurrentText(ContourMethod.CELLPOSE.value)
        self._contour_method_combo.currentTextChanged.connect(
            self._on_contour_method_changed,
        )
        method_row.addWidget(self._contour_method_combo)
        contour_lay.addLayout(method_row)

        # ── Cellpose-specific parameter container ────────────────────────
        # Wrap the EXISTING flow_threshold and niter spinboxes so they can
        # be shown/hidden as a group.
        self._cellpose_params_container = QWidget()
        cp_lay = QHBoxLayout(self._cellpose_params_container)
        cp_lay.setContentsMargins(0, 0, 0, 0)

        # Cellpose mask sweep params
        self.cp_min_spin = _spin_width(QDoubleSpinBox())
        self.cp_min_spin.setRange(-20.0, 20.0)
        self.cp_min_spin.setValue(-3.0)
        self.cp_min_spin.setDecimals(1)
        self.cp_min_spin.setSingleStep(1.0)
        self.cp_max_spin = _spin_width(QDoubleSpinBox())
        self.cp_max_spin.setRange(-20.0, 20.0)
        self.cp_max_spin.setValue(0.0)
        self.cp_max_spin.setDecimals(1)
        self.cp_max_spin.setSingleStep(1.0)
        self.cp_step_spin = _spin_width(QDoubleSpinBox())
        self.cp_step_spin.setRange(0.1, 10.0)
        self.cp_step_spin.setValue(1.0)
        self.cp_step_spin.setDecimals(1)
        self.cp_step_spin.setSingleStep(0.5)
        self.contour_flow_threshold_spin = _spin_width(QDoubleSpinBox())
        self.contour_flow_threshold_spin.setRange(0.0, 10.0)
        self.contour_flow_threshold_spin.setValue(0.0)
        self.contour_flow_threshold_spin.setDecimals(2)
        self.contour_flow_threshold_spin.setSingleStep(0.1)
        self.contour_niter_spin = _spin_width(QSpinBox())
        self.contour_niter_spin.setRange(0, 2000)
        self.contour_niter_spin.setValue(200)

        # Build cellpose parameters container
        sweep_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(sweep_grid, 0,
            "Cellprob min:", compact_spinbox(self.cp_min_spin),
            "Cellprob max:", compact_spinbox(self.cp_max_spin))
        add_block_pair_row(sweep_grid, 1,
            "Cellprob step:", compact_spinbox(self.cp_step_spin),
            "Flow threshold:", compact_spinbox(self.contour_flow_threshold_spin))
        add_block_pair_row(sweep_grid, 2,
            "Niter:", compact_spinbox(self.contour_niter_spin))

        cellpose_container = QWidget()
        cellpose_lay = QVBoxLayout(cellpose_container)
        cellpose_lay.setContentsMargins(0, 0, 0, 0)
        cellpose_lay.setSpacing(4)
        cellpose_lay.addWidget(_param_group_label("Cellpose mask sweep"))
        cellpose_lay.addLayout(sweep_grid)

        self._cellpose_params_container = cellpose_container
        contour_lay.addWidget(self._cellpose_params_container)

        # ── Flow-following parameter container ───────────────────────────
        self._ff_params_container = QWidget()
        ff_lay = QHBoxLayout(self._ff_params_container)
        ff_lay.setContentsMargins(0, 0, 0, 0)

        ff_lay.addWidget(QLabel("Flow weight:"))
        self._ff_flow_weight_spin = QDoubleSpinBox()
        self._ff_flow_weight_spin.setRange(0.0, 1.0)
        self._ff_flow_weight_spin.setSingleStep(0.05)
        self._ff_flow_weight_spin.setValue(0.5)
        self._ff_flow_weight_spin.setToolTip(
            "Blend between flow direction (1.0) and EDT gravity toward "
            "nearest nucleus (0.0)."
        )
        ff_lay.addWidget(self._ff_flow_weight_spin)

        ff_lay.addWidget(QLabel("Step scale:"))
        self._ff_step_scale_spin = QDoubleSpinBox()
        self._ff_step_scale_spin.setRange(0.01, 2.0)
        self._ff_step_scale_spin.setSingleStep(0.05)
        self._ff_step_scale_spin.setValue(0.2)
        self._ff_step_scale_spin.setToolTip("Integration step-size multiplier.")
        ff_lay.addWidget(self._ff_step_scale_spin)

        ff_lay.addWidget(QLabel("Max iter:"))
        self._ff_max_iter_spin = QSpinBox()
        self._ff_max_iter_spin.setRange(10, 2000)
        self._ff_max_iter_spin.setSingleStep(10)
        self._ff_max_iter_spin.setValue(100)
        self._ff_max_iter_spin.setToolTip(
            "Maximum integration steps per pixel before giving up."
        )
        ff_lay.addWidget(self._ff_max_iter_spin)

        ff_lay.addWidget(QLabel("Capture r:"))
        self._ff_capture_radius_spin = QDoubleSpinBox()
        self._ff_capture_radius_spin.setRange(0.0, 30.0)
        self._ff_capture_radius_spin.setSingleStep(0.5)
        self._ff_capture_radius_spin.setValue(0.0)
        self._ff_capture_radius_spin.setToolTip(
            "0 = progressive shell assignment (recommended).\n"
            "> 0 = legacy fixed-radius capture at the given distance (px)."
        )
        ff_lay.addWidget(self._ff_capture_radius_spin)

        # Starts hidden — shown only when Flow-Following is selected
        self._ff_params_container.setVisible(False)
        contour_lay.addWidget(self._ff_params_container)

        # Gamma averaging params
        self.cp_gamma_min_spin = _spin_width(QDoubleSpinBox())
        self.cp_gamma_min_spin.setRange(0.05, 5.0)
        self.cp_gamma_min_spin.setValue(1.0)
        self.cp_gamma_min_spin.setDecimals(2)
        self.cp_gamma_min_spin.setSingleStep(0.05)
        self.cp_gamma_max_spin = _spin_width(QDoubleSpinBox())
        self.cp_gamma_max_spin.setRange(0.05, 5.0)
        self.cp_gamma_max_spin.setValue(1.0)
        self.cp_gamma_max_spin.setDecimals(2)
        self.cp_gamma_max_spin.setSingleStep(0.05)
        self.cp_gamma_step_spin = _spin_width(QDoubleSpinBox())
        self.cp_gamma_step_spin.setRange(0.05, 2.0)
        self.cp_gamma_step_spin.setValue(0.25)
        self.cp_gamma_step_spin.setDecimals(2)
        self.cp_gamma_step_spin.setSingleStep(0.05)

        gamma_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(gamma_grid, 0,
            "Gamma min:", compact_spinbox(self.cp_gamma_min_spin),
            "Gamma max:", compact_spinbox(self.cp_gamma_max_spin))
        add_block_pair_row(gamma_grid, 1,
            "Gamma step:", compact_spinbox(self.cp_gamma_step_spin))
        contour_lay.addWidget(_param_group_label("Gamma averaging"))
        contour_lay.addLayout(gamma_grid)

        # Foreground output
        self.contour_fg_threshold_spin = _spin_width(QDoubleSpinBox())
        self.contour_fg_threshold_spin.setRange(0.0, 1.0)
        self.contour_fg_threshold_spin.setValue(0.5)
        self.contour_fg_threshold_spin.setDecimals(2)
        self.contour_fg_threshold_spin.setSingleStep(0.01)

        fg_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(fg_grid, 0,
            "FG threshold:", compact_spinbox(self.contour_fg_threshold_spin))
        contour_lay.addWidget(_param_group_label("Foreground output"))
        contour_lay.addLayout(fg_grid)

        # Output files
        self.contour_output_files = _stage_files("Outputs", [
            ("3_cell/contour_maps.tif", "Contour maps"),
            ("3_cell/foreground_scores.tif", "Foreground scores"),
            ("3_cell/foreground_masks.tif", "Foreground masks"),
        ])
        contour_lay.addWidget(self.contour_output_files)

        # Buttons
        contour_btn_row = block_grid(horizontal_spacing=12)
        self.preview_contour_btn = QPushButton("Preview")
        self.preview_contour_btn.setToolTip(
            "Build contour maps for the current frame only and display in napari"
        )
        self.preview_contour_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.build_contour_maps_btn = QPushButton("Build Contour Maps")
        self.build_contour_maps_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        add_block_pair_row(contour_btn_row, 0,
            "", self.preview_contour_btn,
            "", self.build_contour_maps_btn)
        contour_lay.addLayout(contour_btn_row)

        self.contour_status_lbl = _stage_status()
        contour_lay.addWidget(self.contour_status_lbl)
        self.contour_progress_bar = _stage_progress()
        contour_lay.addWidget(self.contour_progress_bar)

        self.contour_section = CollapsibleSection(
            "1. Contour Maps", contour_inner, expanded=False
        )
        layout.addWidget(self.contour_section)

    # -- Boundary Selection section (Initialize → Refine → Commit) --------

    def _setup_boundary_selection_section(self, layout: QVBoxLayout) -> None:
        def _stage_files(group_label, entries):
            return PipelineFilesWidget(
                [(group_label, entries)], viewer=self.viewer
            )

        def _stage_status():
            lbl = QLabel("")
            lbl.setWordWrap(True)
            lbl.setVisible(False)
            status_label(lbl)
            return lbl

        def _stage_progress():
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setVisible(False)
            return bar

        def _float_spin(lo, hi, val, tooltip, *, decimals=2, step=0.1):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setDecimals(decimals)
            s.setSingleStep(step)
            s.setToolTip(tooltip)
            s.setMinimumWidth(_CONTOUR_SWEEP_WIDTH)
            s.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            return s

        def _int_spin(lo, hi, val, tooltip):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setToolTip(tooltip)
            s.setMinimumWidth(_CONTOUR_SWEEP_WIDTH)
            s.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            return s

        def _param_group_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight: 600;")
            return lbl

        sel_inner = QWidget()
        sel_lay = QVBoxLayout(sel_inner)
        sel_lay.setContentsMargins(0, 0, 0, 0)
        sel_lay.setSpacing(4)
        sel_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Input files
        self.boundary_selection_input_files = _stage_files("Inputs", [
            ("2_nucleus/tracked_labels.tif", "Nucleus tracked labels"),
            ("3_cell/contour_maps.tif", "Contour maps"),
            ("3_cell/foreground_scores.tif", "Foreground scores"),
            ("3_cell/foreground_masks.tif", "Foreground masks"),
        ])
        sel_lay.addWidget(self.boundary_selection_input_files)

        # ── Initialize parameters ─────────────────────────────────────
        sel_lay.addWidget(_param_group_label("Initialize"))

        self.alpha_unary_spin = _float_spin(
            0.0, 1000.0, 4.0,
            "Contour weight in the geodesic cost field: 1 + α·contour.",
        )
        self.lambda_s_spin = _float_spin(
            0.0, 1000.0, 1.0, "Spatial pairwise Potts weight.",
        )
        self.beta_s_spin = _float_spin(
            0.0, 1000.0, 5.0,
            "Contour sensitivity in spatial pairwise: exp(-β·avg_contour).",
        )
        self.lambda_t_spin = _float_spin(
            0.0, 1000.0, 1.0, "Temporal pairwise Potts weight.",
        )
        self.gamma_unary_spin = _float_spin(
            0.0, 100.0, 0.0,
            "Weight for (1 − foreground_score) in the geodesic cost field. "
            "0 = contour-only (default).",
        )
        self.init_mode_combo = QComboBox()
        self.init_mode_combo.addItems(["nuclei", "unary", "watershed"])
        self.init_mode_combo.setCurrentText("nuclei")
        self.init_mode_combo.setToolTip(
            "nuclei: only nucleus pixels labelled at init, cells grow via ICM.\n"
            "unary: init from per-pixel argmin of geodesic cost.\n"
            "watershed: seeded watershed on geodesic elevation."
        )
        self.init_mode_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        # In _setup_boundary_selection_section, after init_mode_combo:

        self.n_workers_spin = _int_spin(
            1, max(1, os.cpu_count() or 1), min(4, os.cpu_count() or 1),
            "Parallel workers for geodesic unary computation. "
            "Uses fork-based multiprocessing (Linux).",
        )

        init_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(init_grid, 3,
            "n_workers:", compact_spinbox(self.n_workers_spin),
            field_width=92)
        add_block_pair_row(init_grid, 0,
            "alpha_unary:", compact_spinbox(self.alpha_unary_spin),
            "lambda_s:", compact_spinbox(self.lambda_s_spin),
            field_width=92)
        add_block_pair_row(init_grid, 1,
            "beta_s:", compact_spinbox(self.beta_s_spin),
            "lambda_t:", compact_spinbox(self.lambda_t_spin),
            field_width=92)
        add_block_pair_row(init_grid, 2,
            "init_mode:", self.init_mode_combo,
            "gamma_unary:", compact_spinbox(self.gamma_unary_spin),
            field_width=92)
        sel_lay.addLayout(init_grid)

        init_btn_row = block_grid(horizontal_spacing=12)
        self.initialize_btn = QPushButton("Initialize")
        self.initialize_btn.setToolTip(
            "Compute geodesic unary costs and pairwise weights, then build "
            "initial labels. This is the expensive step."
        )
        self.initialize_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        add_block_button_row(init_btn_row, 0, self.initialize_btn)
        sel_lay.addLayout(init_btn_row)

        self.initialize_status_lbl = _stage_status()
        sel_lay.addWidget(self.initialize_status_lbl)
        self.initialize_progress_bar = _stage_progress()
        sel_lay.addWidget(self.initialize_progress_bar)

        # ── Refine parameters ─────────────────────────────────────────
        sel_lay.addWidget(_param_group_label("Refine"))

        self.n_iters_spin = _int_spin(
            1, 100, 3,
            "Number of ICM Gauss-Seidel sweeps per Refine press.",
        )
        self.min_round_flips_spin = _int_spin(
            0, 1_000_000, 0,
            "Stop early if a round produces fewer flips than this.",
        )
        self.lambda_area_spin = _float_spin(
            0.0, 10.0, 0.0,
            "Per-label frame-to-frame area-change penalty. 0 = disabled.",
            decimals=4, step=0.0001,
        )

        refine_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(refine_grid, 0,
            "n_iters:", compact_spinbox(self.n_iters_spin),
            "min_round_flips:", compact_spinbox(self.min_round_flips_spin),
            field_width=92)
        add_block_pair_row(refine_grid, 1,
            "lambda_area:", compact_spinbox(self.lambda_area_spin),
            field_width=92)
        sel_lay.addLayout(refine_grid)

        refine_btn_row = block_grid(horizontal_spacing=12)
        self.refine_btn = QPushButton("Refine")
        self.refine_btn.setToolTip(
            "Run ICM sweeps on the current viewer labels. Press repeatedly "
            "for incremental refinement; hand-correct between presses."
        )
        self.refine_btn.setEnabled(False)
        self.refine_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        add_block_button_row(refine_btn_row, 0, self.refine_btn)
        sel_lay.addLayout(refine_btn_row)

        self.refine_status_lbl = _stage_status()
        sel_lay.addWidget(self.refine_status_lbl)

        # ── Commit ────────────────────────────────────────────────────
        sel_lay.addWidget(_param_group_label("Commit"))

        commit_btn_row = block_grid(horizontal_spacing=12)
        self.commit_btn = QPushButton("Commit to disk")
        self.commit_btn.setToolTip(
            "Write the current viewer labels to 3_cell/tracked_labels.tif."
        )
        self.commit_btn.setEnabled(False)
        self.commit_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        add_block_button_row(commit_btn_row, 0, self.commit_btn)
        sel_lay.addLayout(commit_btn_row)

        self.commit_status_lbl = _stage_status()
        sel_lay.addWidget(self.commit_status_lbl)

        # Output files
        self.boundary_selection_output_files = _stage_files("Outputs", [
            ("3_cell/tracked_labels.tif", "Cell labels"),
        ])
        sel_lay.addWidget(self.boundary_selection_output_files)

        self.boundary_selection_section = CollapsibleSection(
            "2. Track-Conditioned Boundary Selection",
            sel_inner,
            expanded=False,
        )
        layout.addWidget(self.boundary_selection_section)

    # -- Correction section -----------------------------------------------

    def _setup_correction_section(self, layout: QVBoxLayout) -> None:
        correction_inner = QWidget()
        correction_lay = QVBoxLayout(correction_inner)
        correction_lay.setContentsMargins(0, 0, 0, 0)
        correction_lay.setSpacing(4)
        correction_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
        )
        correction_lay.addWidget(self.correction_widget)

        self.correction_section = CollapsibleSection(
            "3. Correction", correction_inner, expanded=False
        )
        layout.addWidget(self.correction_section)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        # Contour
        self.preview_contour_btn.clicked.connect(self._on_preview_contour_maps)
        self.build_contour_maps_btn.clicked.connect(self._on_build_contour_maps)
        # Boundary selection stages
        self.initialize_btn.clicked.connect(self._on_initialize)
        self.refine_btn.clicked.connect(self._on_refine)
        self.commit_btn.clicked.connect(self._on_commit)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _contour_maps_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "contour_maps.tif" if self._pos_dir else None

    def _foreground_scores_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "foreground_scores.tif" if self._pos_dir else None

    def _foreground_masks_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "foreground_masks.tif" if self._pos_dir else None

    def _nucleus_labels_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    def _cell_labels_output_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "tracked_labels.tif" if self._pos_dir else None

    def _prob_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_prob_3dt.tif" if self._pos_dir else None

    def _filtered_dp_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "filtered_dp.tif" if self._pos_dir else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        # Clear cached state when switching positions
        if self._icm_state is not None:
            self._icm_state = None
            self._update_stage_enabled()
        self._refresh_stage_files(pos_dir)
        if pos_dir is None:
            self.correction_widget.deactivate()

    def _refresh_stage_files(self, pos_dir: Path | None = None) -> None:
        if pos_dir is None:
            pos_dir = self._pos_dir
        for fw in (
            self.contour_input_files,
            self.contour_output_files,
            self.boundary_selection_input_files,
            self.boundary_selection_output_files,
        ):
            fw.refresh(pos_dir)

    def get_state(self) -> dict:
        return {
            # Contour params
            "cp_min": self.cp_min_spin.value(),
            "cp_max": self.cp_max_spin.value(),
            "cp_step": self.cp_step_spin.value(),
            "contour_flow_threshold": self.contour_flow_threshold_spin.value(),
            "contour_niter": self.contour_niter_spin.value(),
            "cp_gamma_min": self.cp_gamma_min_spin.value(),
            "cp_gamma_max": self.cp_gamma_max_spin.value(),
            "cp_gamma_step": self.cp_gamma_step_spin.value(),
            "contour_fg_threshold": self.contour_fg_threshold_spin.value(),
            # --- flow-following state ---
            "contour_method": self._contour_method_combo.currentText(),
            "ff_flow_weight": self._ff_flow_weight_spin.value(),
            "ff_step_scale": self._ff_step_scale_spin.value(),
            "ff_max_iter": self._ff_max_iter_spin.value(),
            "ff_capture_radius": self._ff_capture_radius_spin.value(),
            # Initialize params
            "alpha_unary": self.alpha_unary_spin.value(),
            "lambda_s": self.lambda_s_spin.value(),
            "beta_s": self.beta_s_spin.value(),
            "lambda_t": self.lambda_t_spin.value(),
            "gamma_unary": self.gamma_unary_spin.value(),
            "init_mode": self.init_mode_combo.currentText(),
            # Refine params
            "n_iters": self.n_iters_spin.value(),
            "min_round_flips": self.min_round_flips_spin.value(),
            "lambda_area": self.lambda_area_spin.value(),
            "n_workers": self.n_workers_spin.value(),

        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return

        _spin_map = {
            "cp_min": self.cp_min_spin,
            "cp_max": self.cp_max_spin,
            "cp_step": self.cp_step_spin,
            "contour_flow_threshold": self.contour_flow_threshold_spin,
            "contour_niter": self.contour_niter_spin,
            "cp_gamma_min": self.cp_gamma_min_spin,
            "cp_gamma_max": self.cp_gamma_max_spin,
            "cp_gamma_step": self.cp_gamma_step_spin,
            "contour_fg_threshold": self.contour_fg_threshold_spin,
            "alpha_unary": self.alpha_unary_spin,
            "lambda_s": self.lambda_s_spin,
            "beta_s": self.beta_s_spin,
            "lambda_t": self.lambda_t_spin,
            "gamma_unary": self.gamma_unary_spin,
            "n_iters": self.n_iters_spin,
            "min_round_flips": self.min_round_flips_spin,
            "lambda_area": self.lambda_area_spin,
            "n_workers": self.n_workers_spin.value(),

        }
        # Backward-compat: map old graphcut_* keys
        _legacy_map = {
            "graphcut_alpha_unary": "alpha_unary",
            "graphcut_lambda_s": "lambda_s",
            "graphcut_beta_s": "beta_s",
            "graphcut_lambda_t": "lambda_t",
            "graphcut_n_iters": "n_iters",
            "graphcut_min_round_flips": "min_round_flips",
            "graphcut_init_mode": "init_mode",
        }
        for old_key, new_key in _legacy_map.items():
            if old_key in state and new_key not in state:
                state[new_key] = state[old_key]

        for key, widget in _spin_map.items():
            if key in state:
                widget.setValue(state[key])
        if "init_mode" in state:
            self.init_mode_combo.setCurrentText(str(state["init_mode"]))

        # --- flow-following state ---
        if "contour_method" in state:
            self._contour_method_combo.setCurrentText(state["contour_method"])
        if "ff_flow_weight" in state:
            self._ff_flow_weight_spin.setValue(state["ff_flow_weight"])
        if "ff_step_scale" in state:
            self._ff_step_scale_spin.setValue(state["ff_step_scale"])
        if "ff_max_iter" in state:
            self._ff_max_iter_spin.setValue(state["ff_max_iter"])
        if "ff_capture_radius" in state:
            self._ff_capture_radius_spin.setValue(state["ff_capture_radius"])

    # ------------------------------------------------------------------
    # Stage enable/disable
    # ------------------------------------------------------------------
    def _update_stage_enabled(self) -> None:
        """Enable/disable Refine and Commit based on whether Initialize has run."""
        has_state = self._icm_state is not None
        has_layer = _CELL_SEG_LAYER in self.viewer.layers
        self.refine_btn.setEnabled(has_state and has_layer)
        self.commit_btn.setEnabled(has_layer)

    def _set_all_buttons_enabled(self, enabled: bool) -> None:
        """Disable all stage buttons during a long-running operation."""
        self.initialize_btn.setEnabled(enabled)
        self.refine_btn.setEnabled(enabled and self._icm_state is not None)
        self.commit_btn.setEnabled(enabled and _CELL_SEG_LAYER in self.viewer.layers)

    # ------------------------------------------------------------------
    # Status / layer helpers
    # ------------------------------------------------------------------
    def _set_contour_status(self, msg: str) -> None:
        self.contour_status_lbl.setText(msg)
        self.contour_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_initialize_status(self, msg: str) -> None:
        self.initialize_status_lbl.setText(msg)
        self.initialize_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_refine_status(self, msg: str) -> None:
        self.refine_status_lbl.setText(msg)
        self.refine_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_commit_status(self, msg: str) -> None:
        self.commit_status_lbl.setText(msg)
        self.commit_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _show_layer(self, layer_name, data, kwargs, adder):
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = data
        else:
            adder(data, name=layer_name, **kwargs)

    def _current_t(self) -> int:
        dims = getattr(self.viewer, "dims", None)
        step = getattr(dims, "current_step", (0,))
        return int(step[0]) if len(step) >= 1 else 0

    # ------------------------------------------------------------------
    # 1. Contour Maps (Preview / Build)
    # ------------------------------------------------------------------
    def _cellprob_thresholds(self) -> list[float]:
        step = self.cp_step_spin.value()
        return list(np.arange(
            self.cp_min_spin.value(),
            self.cp_max_spin.value() + step / 2,
            step,
        ))

    def _cp_gammas(self) -> list[float]:
        step = self.cp_gamma_step_spin.value()
        return list(np.arange(
            self.cp_gamma_min_spin.value(),
            self.cp_gamma_max_spin.value() + step / 2,
            step,
        ))

    def _build_consensus_boundary_averaged(
        self, prob_3d, dp_2d, thresholds, gammas,
        *, flow_threshold, niter,
    ):
        boundary_accum = foreground_accum = None
        for gamma in gammas:
            prob_2d = apply_gamma(prob_3d, gamma).mean(axis=0)
            b, fg = build_consensus_boundary_2d(
                prob_2d, dp_2d, thresholds,
                flow_threshold=flow_threshold, reduction="mean", niter=niter,
            )
            if boundary_accum is None:
                boundary_accum = b.copy()
                foreground_accum = fg.copy()
            else:
                boundary_accum += b
                foreground_accum += fg
        n = len(gammas)
        return boundary_accum / n, foreground_accum / n

    def _set_contour_buttons_running(self, running: bool) -> None:
        self.build_contour_maps_btn.setEnabled(not running)
        self.preview_contour_btn.setEnabled(not running)
        self.contour_progress_bar.setVisible(running)
        if not running:
            self.contour_progress_bar.setValue(0)

    # ------------------------------------------------------------------
    # Contour method toggle
    # ------------------------------------------------------------------

    def _on_contour_method_changed(self, text: str) -> None:
        """Show / hide parameter rows that belong to the selected method."""
        is_ff = text == ContourMethod.FLOW_FOLLOWING.value
        self._cellpose_params_container.setVisible(not is_ff)
        self._ff_params_container.setVisible(is_ff)

    # ------------------------------------------------------------------
    # Flow-following helpers
    # ------------------------------------------------------------------

    def _load_prob_frame(self, t: int) -> np.ndarray:
        """Load a single probability frame (Z, Y, X)."""
        prob_path = self._prob_path()
        if prob_path is None or not prob_path.exists():
            raise FileNotFoundError(f"Probability file not found: {prob_path}")
        prob_stack = tifffile.imread(str(prob_path))
        if prob_stack.ndim == 3:
            prob_stack = prob_stack[np.newaxis]
        return prob_stack[t].astype(np.float32)

    def _load_dp_frame(self, t: int) -> np.ndarray:
        """Load a single DP flow frame (2, Y, X)."""
        dp_path = self._filtered_dp_path()
        if dp_path is None or not dp_path.exists():
            raise FileNotFoundError(f"DP file not found: {dp_path}")
        dp_stack = tifffile.imread(str(dp_path))
        if dp_stack.ndim == 3:
            dp_stack = dp_stack[np.newaxis]
        return dp_stack[t].astype(np.float32)

    def _show_error(self, title: str, message: str) -> None:
        """Display an error dialog (simple text message for now)."""
        self._set_contour_status(f"{title}: {message}")

    def _current_ff_params(self) -> FlowFollowingParams:
        """Read the flow-following spinboxes and return a frozen dataclass."""
        return FlowFollowingParams(
            median_kernel_time=1,
            median_kernel_space=1,
            gaussian_sigma_time=0.0,
            gaussian_sigma_space=0.0,
            flow_weight=self._ff_flow_weight_spin.value(),
            flow_step_scale=self._ff_step_scale_spin.value(),
            max_iterations=self._ff_max_iter_spin.value(),
            capture_radius=self._ff_capture_radius_spin.value(),
        )

    def _build_consensus_boundary_ff_averaged(
        self,
        prob_3d: np.ndarray,
        dp_2d: np.ndarray,
        labels_yx: np.ndarray,
        thresholds: list[float],
        gammas: list[float],
        *,
        ff_params: FlowFollowingParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gamma-averaged consensus boundary via flow-following.

        Mirrors ``_build_consensus_boundary_averaged`` but dispatches to
        :func:`build_consensus_boundary_flow_following` instead of
        Cellpose's ``compute_masks``.
        """
        boundary_accum = None
        foreground_accum = None
        n = 0
        for gamma in gammas:
            prob_2d = apply_gamma(prob_3d, gamma).mean(axis=0)
            b, fg = build_consensus_boundary_flow_following(
                prob_2d,
                dp_2d,
                labels_yx,
                thresholds,
                params=ff_params,
                reduction="mean",
            )
            if boundary_accum is None:
                boundary_accum = b.copy()
                foreground_accum = fg.copy()
            else:
                boundary_accum += b
                foreground_accum += fg
            n += 1
        if n > 0:
            boundary_accum /= n
            foreground_accum /= n
        return boundary_accum, foreground_accum

    def _on_build_contour_maps(self) -> None:
        """Launch the contour-map worker, dispatching by selected method."""
        # ── shared parameter gathering ───────────────────
        thresholds = self._cellprob_thresholds()
        gammas = self._cp_gammas()
        fg_threshold = self.contour_fg_threshold_spin.value()

        method = self._contour_method_combo.currentText()

        # ── method-specific gathering ────────────────────────────────
        if method == ContourMethod.FLOW_FOLLOWING.value:
            nuc_path = self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None
            if nuc_path is None or not nuc_path.exists():
                self._show_error(
                    "Nucleus tracked labels not found",
                    f"Flow-following requires nucleus tracked labels at:\n"
                    f"{nuc_path}\n\n"
                    f"Run the Nucleus Segmentation & Tracking step first.",
                )
                return
            nuc_labels = tifffile.imread(str(nuc_path))  # (T, Y, X)
            ff_params = self._current_ff_params()
            extra_kw = dict(
                method="flow_following",
                nuc_labels=nuc_labels,
                ff_params=ff_params,
            )
        else:
            extra_kw = dict(
                method="cellpose",
                flow_threshold=self.contour_flow_threshold_spin.value(),
                niter=int(self.contour_niter_spin.value()),
            )

        # ── launch worker (napari thread worker) ─────────────────────
        prob_path = self._prob_path()
        filtered_dp_path = self._filtered_dp_path()
        contour_path = self._contour_maps_path()
        score_path = self._foreground_scores_path()
        foreground_path = self._foreground_masks_path()
        for path, name in [
            (prob_path, "cell_prob_3dt.tif"),
            (filtered_dp_path, "filtered_dp.tif"),
        ]:
            if path is None or not path.exists():
                self._set_contour_status(f"Missing: {name}")
                return
        if contour_path is None or score_path is None or foreground_path is None:
            self._set_contour_status("No project open.")
            return

        pos_dir = self._pos_dir

        def _on_done(result):
            self._contour_worker = None
            self._set_contour_buttons_running(False)
            contours, scores, foreground = result
            self._show_layer(_CELL_CONTOUR_LAYER, contours,
                             {"colormap": "magma", "visible": True}, self.viewer.add_image)
            self._show_layer(_CELL_FOREGROUND_SCORE_LAYER, scores,
                             {"colormap": "viridis", "visible": True}, self.viewer.add_image)
            self._show_layer(_CELL_FOREGROUND_LAYER, foreground,
                             {}, self.viewer.add_labels)
            self._refresh_stage_files(pos_dir)
            self._set_contour_status("Contour maps complete.")

        def _on_progress(data):
            if isinstance(data, tuple):
                done, total, msg = data
                if total > 0:
                    self.contour_progress_bar.setVisible(True)
                    self.contour_progress_bar.setRange(0, total)
                    self.contour_progress_bar.setValue(done)
                self._set_contour_status(msg)
            else:
                self._set_contour_status(str(data))

        @thread_worker(connect={
            "yielded": _on_progress,
            "returned": _on_done,
            "errored": lambda exc: self._on_contour_error(exc),
        })
        def _worker():
            prob_path = self._prob_path()
            filtered_dp_path = self._filtered_dp_path()
            contour_path = self._contour_maps_path()
            score_path = self._foreground_scores_path()
            foreground_path = self._foreground_masks_path()

            prob_stack = tifffile.imread(str(prob_path))  # (T, Z, Y, X)
            dp_stack = tifffile.imread(str(filtered_dp_path))  # (T, 2, Y, X)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis]
            if dp_stack.ndim == 3:
                dp_stack = dp_stack[np.newaxis]
            T = prob_stack.shape[0]

            contour_maps = np.zeros((T, *prob_stack.shape[2:]), dtype=np.float32)
            fg_scores = np.zeros_like(contour_maps)
            fg_masks = np.zeros_like(contour_maps, dtype=bool)

            method = extra_kw.get("method", "cellpose")
            nuc_labels = extra_kw.get("nuc_labels", None)
            ff_params = extra_kw.get("ff_params", None)
            flow_threshold = extra_kw.get("flow_threshold", 0.0)
            niter = extra_kw.get("niter", 200)

            for t in range(T):
                yield (t + 1, T, f"Building contour maps: frame {t + 1}/{T}...")
                prob_3d = prob_stack[t]  # (Z, Y, X)
                dp_2d = dp_stack[t]     # (2, Y, X)

                if method == "flow_following":
                    labels_t = nuc_labels[t]  # (Y, X)
                    b, fg = self._build_consensus_boundary_ff_averaged(
                        prob_3d,
                        dp_2d,
                        labels_t,
                        thresholds,
                        gammas,
                        ff_params=ff_params,
                    )
                else:  # "cellpose"
                    b, fg = self._build_consensus_boundary_averaged(
                        prob_3d,
                        dp_2d,
                        thresholds,
                        gammas,
                        flow_threshold=flow_threshold,
                        niter=niter,
                    )

                contour_maps[t] = b
                fg_scores[t] = fg
                fg_masks[t] = fg > fg_threshold

            contour_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(contour_path), contour_maps, compression="zlib")
            tifffile.imwrite(str(score_path), fg_scores, compression="zlib")
            tifffile.imwrite(str(foreground_path), fg_masks.astype(np.uint8), compression="zlib")
            return contour_maps, fg_scores, fg_masks.astype(np.uint8)

        self._set_contour_status(
            f"Building contour maps ({len(thresholds)} thresholds, "
            f"{len(gammas)} gamma value(s))..."
        )
        self._set_contour_buttons_running(True)
        self._contour_worker = _worker()

    def _on_preview_contour_maps(self) -> None:
        """Preview the contour map for the current frame only."""
        t = self._current_t()
        thresholds = self._cellprob_thresholds()
        gammas = self._cp_gammas()
        method = self._contour_method_combo.currentText()

        prob_3d = self._load_prob_frame(t)  # (Z, Y, X)
        dp_2d = self._load_dp_frame(t)     # (2, Y, X)

        if method == ContourMethod.FLOW_FOLLOWING.value:
            nuc_path = self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None
            if nuc_path is None or not nuc_path.exists():
                self._show_error(
                    "Nucleus tracked labels not found",
                    f"Flow-following requires:\n{nuc_path}",
                )
                return
            nuc_labels_t = tifffile.imread(str(nuc_path))[t]
            ff_params = self._current_ff_params()
            b, fg = self._build_consensus_boundary_ff_averaged(
                prob_3d,
                dp_2d,
                nuc_labels_t,
                thresholds,
                gammas,
                ff_params=ff_params,
            )
        else:
            b, fg = self._build_consensus_boundary_averaged(
                prob_3d,
                dp_2d,
                thresholds,
                gammas,
                flow_threshold=self.contour_flow_threshold_spin.value(),
                niter=int(self.contour_niter_spin.value()),
            )

        # Display preview
        foreground_threshold = self.contour_fg_threshold_spin.value()

        # Load the full stack to get dimensions
        prob_path = self._prob_path()
        prob_stack = tifffile.imread(str(prob_path))
        if prob_stack.ndim == 3:
            prob_stack = prob_stack[np.newaxis]
        n_t = prob_stack.shape[0]

        contour_data = np.zeros((n_t,) + b.shape, dtype=np.float32)
        contour_data[t] = b
        score_data = np.zeros((n_t,) + fg.shape, dtype=np.float32)
        score_data[t] = fg
        mask_data = (score_data >= foreground_threshold).astype(np.uint8)
        self._show_layer(_CELL_CONTOUR_LAYER, contour_data,
                         {"colormap": "magma", "visible": True}, self.viewer.add_image)
        self._show_layer(_CELL_FOREGROUND_SCORE_LAYER, score_data,
                         {"colormap": "viridis", "visible": True}, self.viewer.add_image)
        self._show_layer(_CELL_FOREGROUND_LAYER, mask_data, {}, self.viewer.add_labels)
        self._set_contour_status(
            f"Preview t={t} — {len(thresholds)} thresholds, "
            f"{len(gammas)} gamma(s)"
        )

    def _on_contour_error(self, exc: Exception) -> None:
        self._contour_worker = None
        self._set_contour_buttons_running(False)
        self._set_contour_status(f"Error: {exc}")
        logger.exception("Cell contour worker error", exc_info=exc)

    # ------------------------------------------------------------------
    # 2a. Initialize
    # ------------------------------------------------------------------
    def _on_initialize(self) -> None:
        if self._pos_dir is None:
            self._set_initialize_status("No project open.")
            return

        required = [
            (self._nucleus_labels_path(), "tracked_labels.tif (nucleus)"),
            (self._contour_maps_path(), "contour_maps.tif"),
            (self._foreground_masks_path(), "foreground_masks.tif"),
        ]
        for path, name in required:
            if path is None or not path.exists():
                self._set_initialize_status(f"Missing: {name}")
                return

        # Collect paths and params before entering the thread
        nuc_path = self._nucleus_labels_path()
        fg_path = self._foreground_masks_path()
        ct_path = self._contour_maps_path()
        score_path = self._foreground_scores_path()
        pos_dir = self._pos_dir

        from cellflow.segmentation.cell_label_icm import (
            CellLabelICMParams,
            initialize_icm,
        )

        params = CellLabelICMParams(
            alpha_unary=self.alpha_unary_spin.value(),
            lambda_s=self.lambda_s_spin.value(),
            beta_s=self.beta_s_spin.value(),
            lambda_t=self.lambda_t_spin.value(),
            gamma_unary=self.gamma_unary_spin.value(),
            init_mode=self.init_mode_combo.currentText(),
            n_workers=self.n_workers_spin.value(),  # ← new
        )

        def _on_done(result):
            self._initialize_worker = None
            state, init_labels = result
            self._icm_state = state
            self._show_layer(
                _CELL_SEG_LAYER, init_labels, {"visible": True},
                self.viewer.add_labels,
            )
            self._set_all_buttons_enabled(True)
            self._update_stage_enabled()
            self.initialize_progress_bar.setVisible(False)
            self._set_initialize_status(
                f"Initialized: {state.n_labels} labels, "
                f"{'×'.join(str(d) for d in state.shape)}. "
                f"Ready for refinement."
            )

        def _on_error(exc):
            self._initialize_worker = None
            self._set_all_buttons_enabled(True)
            self._update_stage_enabled()
            self.initialize_progress_bar.setVisible(False)
            self._set_initialize_status(f"Error: {exc}")
            logger.exception("Initialize error", exc_info=exc)

        def _on_yielded(msg):
            self._set_initialize_status(str(msg))

        @thread_worker(connect={
            "yielded": _on_yielded,
            "returned": _on_done,
            "errored": _on_error,
        })
        def _worker():
            from cellflow.segmentation.cell_label_icm import _load_pos_dir_inputs

            msg_q: queue.SimpleQueue = queue.SimpleQueue()
            result_holder: list = []
            exc_holder: list = []

            def _run():
                try:
                    nuc, fg, ct, fg_scores = _load_pos_dir_inputs(pos_dir)
                    s, init = initialize_icm(
                        nuc, fg, ct, params,
                        foreground_scores=fg_scores,
                        progress_cb=lambda m: msg_q.put(m),
                    )
                    result_holder.append((s, init))
                except Exception as e:
                    exc_holder.append(e)

            yield "Loading inputs..."
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            while t.is_alive() or not msg_q.empty():
                try:
                    yield msg_q.get_nowait()
                except queue.Empty:
                    t.join(timeout=0.05)
            if exc_holder:
                raise exc_holder[0]
            return result_holder[0]

        self._set_initialize_status("Initializing...")
        self.initialize_progress_bar.setRange(0, 0)  # indeterminate
        self.initialize_progress_bar.setVisible(True)
        self._set_all_buttons_enabled(False)
        self._initialize_worker = _worker()

    # ------------------------------------------------------------------
    # 2b. Refine
    # ------------------------------------------------------------------
    def _on_refine(self) -> None:
        if self._icm_state is None:
            self._set_refine_status("Run Initialize first.")
            return
        if _CELL_SEG_LAYER not in self.viewer.layers:
            self._set_refine_status("No label layer — run Initialize first.")
            return

        current_labels = np.asarray(
            self.viewer.layers[_CELL_SEG_LAYER].data, dtype=np.uint32,
        )
        state = self._icm_state
        n_iters = self.n_iters_spin.value()
        min_flips = self.min_round_flips_spin.value()
        lambda_area = self.lambda_area_spin.value()

        from cellflow.segmentation.cell_label_icm import refine_icm

        def _on_done(result):
            self._refine_worker = None
            new_labels, energy_log = result
            self.viewer.layers[_CELL_SEG_LAYER].data = new_labels
            self._set_all_buttons_enabled(True)
            self._update_stage_enabled()

            # Summarise
            total_flips = sum(e["flips"] for e in energy_log)
            rounds = len(energy_log)
            detail = ", ".join(
                f"r{e['iteration']}={e['flips']}" for e in energy_log
            )
            self._set_refine_status(
                f"{rounds} round(s), {total_flips} total flips. [{detail}]"
            )

        def _on_error(exc):
            self._refine_worker = None
            self._set_all_buttons_enabled(True)
            self._update_stage_enabled()
            self._set_refine_status(f"Error: {exc}")
            logger.exception("Refine error", exc_info=exc)

        def _on_yielded(msg):
            self._set_refine_status(str(msg))

        @thread_worker(connect={
            "yielded": _on_yielded,
            "returned": _on_done,
            "errored": _on_error,
        })
        def _worker():
            msg_q: queue.SimpleQueue = queue.SimpleQueue()
            result_holder: list = []
            exc_holder: list = []

            def _run():
                try:
                    result_holder.append(
                        refine_icm(
                            state, current_labels,
                            n_iters=n_iters,
                            min_round_flips=min_flips,
                            lambda_area=lambda_area,
                            progress_cb=lambda m: msg_q.put(m),
                        )
                    )
                except Exception as e:
                    exc_holder.append(e)

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            while t.is_alive() or not msg_q.empty():
                try:
                    yield msg_q.get_nowait()
                except queue.Empty:
                    t.join(timeout=0.05)
            if exc_holder:
                raise exc_holder[0]
            return result_holder[0]

        self._set_refine_status(f"Refining ({n_iters} iterations)...")
        self._set_all_buttons_enabled(False)
        self._refine_worker = _worker()

    # ------------------------------------------------------------------
    # 2c. Commit
    # ------------------------------------------------------------------
    def _on_commit(self) -> None:
        if _CELL_SEG_LAYER not in self.viewer.layers:
            self._set_commit_status("No label layer to save.")
            return
        output_path = self._cell_labels_output_path()
        if output_path is None:
            self._set_commit_status("No project open.")
            return

        from cellflow.segmentation.cell_label_icm import commit_labels

        labels = np.asarray(self.viewer.layers[_CELL_SEG_LAYER].data)
        commit_labels(labels, output_path)
        self._refresh_stage_files(self._pos_dir)
        self._set_commit_status(f"Saved to {output_path.name}.")