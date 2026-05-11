"""Track-conditioned cell boundary selection widget for CellFlow."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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
from cellflow.segmentation import apply_gamma, build_consensus_boundary_2d

logger = logging.getLogger(__name__)

_CELL_LABELS_LAYER = "Cell Labels"
_TRACKED_CELL_LAYER = "Tracked: Cell"
_CELL_CONTOUR_LAYER = "Contour Map: Cell"
_CELL_FOREGROUND_SCORE_LAYER = "Foreground Score: Cell"
_CELL_FOREGROUND_LAYER = "Foreground Mask: Cell"
_GRAPHCUT_CELL_LABELS_LAYER = "cell_labels_graphcut"
_CONTOUR_SWEEP_WIDTH = 60


class CellBoundaryWorkflowWidget(QWidget):
    """Track-conditioned cell boundary selection workflow."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._contour_worker = None
        self._boundary_selection_worker = None
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        def _stage_files(
            group_label: str, entries: list[tuple[str, str]]
        ) -> PipelineFilesWidget:
            return PipelineFilesWidget(
                [(group_label, entries)], viewer=self.viewer
            )

        def _stage_status() -> QLabel:
            label = QLabel("")
            label.setWordWrap(True)
            label.setVisible(False)
            status_label(label)
            return label

        def _stage_progress() -> QProgressBar:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setVisible(False)
            return bar

        def _spin_width(widget, width=_CONTOUR_SWEEP_WIDTH):
            widget.setMinimumWidth(width)
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            return widget

        def _combo(items: list[str], current: str, tooltip: str) -> QComboBox:
            combo = QComboBox()
            combo.addItems(items)
            combo.setCurrentText(current)
            combo.setToolTip(tooltip)
            combo.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            return combo

        def _int_spin(
            minimum: int,
            maximum: int,
            value: int,
            tooltip: str,
        ) -> QSpinBox:
            spin = _spin_width(QSpinBox())
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            spin.setToolTip(tooltip)
            return spin

        def _float_spin(
            minimum: float,
            maximum: float,
            value: float,
            tooltip: str,
            *,
            decimals: int = 2,
            step: float = 0.1,
        ) -> QDoubleSpinBox:
            spin = _spin_width(QDoubleSpinBox())
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setToolTip(tooltip)
            return spin

        def _param_group_label(text: str) -> QLabel:
            label = QLabel(text)
            label.setStyleSheet("font-weight: 600;")
            return label

        self._icm_disabled_widgets: list[QWidget] = []

        # ---- 1. Contour Maps ----
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
        self.contour_niter_spin.setToolTip("Cellpose flow ODE integration steps.")

        contour_sweep_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(
            contour_sweep_grid,
            0,
            "Cellprob min:",
            compact_spinbox(self.cp_min_spin),
            "Cellprob max:",
            compact_spinbox(self.cp_max_spin),
        )
        add_block_pair_row(
            contour_sweep_grid,
            1,
            "Cellprob step:",
            compact_spinbox(self.cp_step_spin),
            "Flow threshold:",
            compact_spinbox(self.contour_flow_threshold_spin),
        )
        add_block_pair_row(
            contour_sweep_grid,
            2,
            "Niter:",
            compact_spinbox(self.contour_niter_spin),
        )
        contour_lay.addWidget(_param_group_label("Cellpose mask sweep"))
        contour_lay.addLayout(contour_sweep_grid)

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
        add_block_pair_row(
            gamma_grid,
            0,
            "Gamma min:",
            compact_spinbox(self.cp_gamma_min_spin),
            "Gamma max:",
            compact_spinbox(self.cp_gamma_max_spin),
        )
        add_block_pair_row(
            gamma_grid,
            1,
            "Gamma step:",
            compact_spinbox(self.cp_gamma_step_spin),
        )
        contour_lay.addWidget(_param_group_label("Gamma averaging"))
        contour_lay.addLayout(gamma_grid)

        self.contour_fg_threshold_spin = _spin_width(QDoubleSpinBox())
        self.contour_fg_threshold_spin.setRange(0.0, 1.0)
        self.contour_fg_threshold_spin.setValue(0.5)
        self.contour_fg_threshold_spin.setDecimals(2)
        self.contour_fg_threshold_spin.setSingleStep(0.01)
        fg_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(
            fg_grid,
            0,
            "FG threshold:",
            compact_spinbox(self.contour_fg_threshold_spin),
        )
        contour_lay.addWidget(_param_group_label("Foreground output"))
        contour_lay.addLayout(fg_grid)

        self.contour_output_files = _stage_files("Outputs", [
            ("3_cell/contour_maps.tif", "Contour maps"),
            ("3_cell/foreground_scores.tif", "Foreground scores"),
            ("3_cell/foreground_masks.tif", "Foreground masks"),
        ])
        contour_lay.addWidget(self.contour_output_files)

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
        add_block_pair_row(
            contour_btn_row,
            0,
            "",
            self.preview_contour_btn,
            "",
            self.build_contour_maps_btn,
        )
        contour_lay.addLayout(contour_btn_row)
        self.contour_status_lbl = _stage_status()
        contour_lay.addWidget(self.contour_status_lbl)
        self.contour_progress_bar = _stage_progress()
        contour_lay.addWidget(self.contour_progress_bar)

        self.contour_section = CollapsibleSection(
            "1. Contour Maps", contour_inner, expanded=False
        )
        layout.addWidget(self.contour_section)

        # ---- 2. Track-Conditioned Boundary Selection ----
        selection_inner = QWidget()
        selection_lay = QVBoxLayout(selection_inner)
        selection_lay.setContentsMargins(0, 0, 0, 0)
        selection_lay.setSpacing(4)
        selection_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.boundary_selection_input_files = _stage_files("Inputs", [
            ("2_nucleus/tracked_labels.tif", "Nucleus tracked labels"),
            ("3_cell/contour_maps.tif", "Contour maps"),
            ("3_cell/foreground_scores.tif", "Foreground scores"),
            ("3_cell/foreground_masks.tif", "Foreground masks"),
            ("1_cellpose/cell_dp_3dt.tif", "Raw Cellpose cell flow"),
        ])
        selection_lay.addWidget(self.boundary_selection_input_files)

        self.graphcut_solver_combo = _combo(
            ["graphcut", "icm"],
            "graphcut",
            "Solver used by the graphcut experiment script.",
        )
        self.graphcut_unary_mode_combo = _combo(
            ["flow", "geodesic_flow", "geodesic", "euclidean"],
            "flow",
            "Unary term. flow uses raw Cellpose flows from 1_cellpose/cell_dp_3dt.tif.",
        )
        self.graphcut_boundary_mode_combo = _combo(
            ["contour", "foreground_inverse"],
            "contour",
            "Boundary signal for pairwise costs.",
        )
        self.graphcut_n_iters_spin = _int_spin(
            1, 100, 1, "Number of solver iterations or alpha-expansion rounds."
        )
        self.graphcut_n_workers_spin = _int_spin(
            1, 128, 1, "Parallel worker processes for graphcut expansion moves."
        )

        core_grid = block_grid(horizontal_spacing=12)
        _, _, graphcut_unary_label, _ = add_block_pair_row(
            core_grid,
            0,
            "Solver:",
            self.graphcut_solver_combo,
            "Unary:",
            self.graphcut_unary_mode_combo,
            field_width=110,
        )
        boundary_label, _, _, _ = add_block_pair_row(
            core_grid,
            1,
            "Boundary:",
            self.graphcut_boundary_mode_combo,
            "Iters:",
            compact_spinbox(self.graphcut_n_iters_spin),
            field_width=110,
        )
        workers_label, _, _, _ = add_block_pair_row(
            core_grid,
            2,
            "Workers:",
            compact_spinbox(self.graphcut_n_workers_spin),
            field_width=110,
        )
        self._icm_disabled_widgets.extend([
            graphcut_unary_label,
            self.graphcut_unary_mode_combo,
            boundary_label,
            self.graphcut_boundary_mode_combo,
            workers_label,
            self.graphcut_n_workers_spin,
        ])
        selection_lay.addWidget(_param_group_label("Graphcut run"))
        selection_lay.addLayout(core_grid)

        advanced_inner = QWidget()
        advanced_lay = QVBoxLayout(advanced_inner)
        advanced_lay.setContentsMargins(0, 0, 0, 0)
        advanced_lay.setSpacing(4)

        self.graphcut_alpha_unary_spin = _float_spin(
            0.0, 1000.0, 4.0, "Contour weight in the geodesic unary cost field."
        )
        self.graphcut_lambda_geodesic_spin = _float_spin(
            0.0, 1000.0, 1.0, "Geodesic unary weight for geodesic_flow mode."
        )
        self.graphcut_lambda_flow_spin = _float_spin(
            0.0, 1000.0, 1.0, "Flow endpoint unary weight for geodesic_flow mode."
        )
        unary_grid = block_grid(horizontal_spacing=12)
        _, _, lambda_geodesic_label, _ = add_block_pair_row(
            unary_grid,
            0,
            "alpha_unary:",
            compact_spinbox(self.graphcut_alpha_unary_spin),
            "lambda_geodesic:",
            compact_spinbox(self.graphcut_lambda_geodesic_spin),
            field_width=92,
        )
        lambda_flow_label, _, _, _ = add_block_pair_row(
            unary_grid,
            1,
            "lambda_flow:",
            compact_spinbox(self.graphcut_lambda_flow_spin),
            field_width=92,
        )
        self._icm_disabled_widgets.extend([
            lambda_geodesic_label,
            self.graphcut_lambda_geodesic_spin,
            lambda_flow_label,
            self.graphcut_lambda_flow_spin,
        ])
        advanced_lay.addWidget(_param_group_label("Unary"))
        advanced_lay.addLayout(unary_grid)

        self.graphcut_lambda_s_spin = _float_spin(
            0.0, 1000.0, 1.0, "Spatial pairwise weight."
        )
        self.graphcut_beta_s_spin = _float_spin(
            0.0, 1000.0, 5.0, "Contour sensitivity for spatial pairwise costs."
        )
        self.graphcut_lambda_contour_spin = _float_spin(
            0.0, 1000.0, 0.0, "Extra contour-weighted pairwise term."
        )
        spatial_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(
            spatial_grid,
            0,
            "lambda_s:",
            compact_spinbox(self.graphcut_lambda_s_spin),
            "beta_s:",
            compact_spinbox(self.graphcut_beta_s_spin),
            field_width=92,
        )
        lambda_contour_label, _, _, _ = add_block_pair_row(
            spatial_grid,
            1,
            "lambda_contour:",
            compact_spinbox(self.graphcut_lambda_contour_spin),
            field_width=92,
        )
        self._icm_disabled_widgets.extend([
            lambda_contour_label,
            self.graphcut_lambda_contour_spin,
        ])
        advanced_lay.addWidget(_param_group_label("Spatial Pairwise"))
        advanced_lay.addLayout(spatial_grid)

        self.graphcut_lambda_t_spin = _float_spin(
            0.0, 1000.0, 1.0, "Temporal pairwise weight."
        )
        temporal_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(
            temporal_grid,
            0,
            "lambda_t:",
            compact_spinbox(self.graphcut_lambda_t_spin),
            field_width=92,
        )
        advanced_lay.addWidget(_param_group_label("Temporal"))
        advanced_lay.addLayout(temporal_grid)

        self.graphcut_init_mode_combo = _combo(
            ["nuclei", "unary", "euclidean", "geodesic"],
            "nuclei",
            "Initialization used before iterative optimization. "
            "'nuclei' (default) labels only nucleus pixels and lets the "
            "algorithm grow each cell naturally from its nucleus.",
        )
        self.graphcut_min_round_flips_spin = _int_spin(
            0, 1_000_000, 0, "Stop after a round with fewer flips than this value."
        )
        solver_grid = block_grid(horizontal_spacing=12)
        init_mode_label, _, _, _ = add_block_pair_row(
            solver_grid,
            0,
            "init_mode:",
            self.graphcut_init_mode_combo,
            "min_round_flips:",
            compact_spinbox(self.graphcut_min_round_flips_spin),
            field_width=92,
        )
        self._icm_disabled_widgets.extend([
            init_mode_label,
            self.graphcut_init_mode_combo,
        ])
        advanced_lay.addWidget(_param_group_label("Solver"))
        advanced_lay.addLayout(solver_grid)

        self.graphcut_advanced_section = CollapsibleSection(
            "Advanced Graphcut Parameters", advanced_inner, expanded=True
        )
        selection_lay.addWidget(self.graphcut_advanced_section)

        selection_run_row = block_grid(horizontal_spacing=12)
        self.run_boundary_selection_btn = QPushButton(
            "Run Track-Conditioned Boundary Selection"
        )
        self.run_boundary_selection_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        add_block_button_row(
            selection_run_row, 0, self.run_boundary_selection_btn
        )
        selection_lay.addLayout(selection_run_row)

        self.boundary_selection_status_lbl = _stage_status()
        selection_lay.addWidget(self.boundary_selection_status_lbl)

        self.boundary_selection_progress_bar = _stage_progress()
        selection_lay.addWidget(self.boundary_selection_progress_bar)

        self.boundary_selection_output_files = _stage_files("Outputs", [
            ("3_cell/tracked_labels.tif", "Cell labels"),
        ])
        selection_lay.addWidget(self.boundary_selection_output_files)

        self.boundary_selection_section = CollapsibleSection(
            "2. Track-Conditioned Boundary Selection",
            selection_inner,
            expanded=False,
        )
        layout.addWidget(self.boundary_selection_section)

        # ---- 3. Correction ----
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

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.preview_contour_btn.clicked.connect(self._on_preview_contour_maps)
        self.build_contour_maps_btn.clicked.connect(self._on_build_contour_maps)
        self.run_boundary_selection_btn.clicked.connect(
            self._on_run_boundary_selection
        )
        self.graphcut_solver_combo.currentTextChanged.connect(
            self._on_solver_changed
        )
        self._on_solver_changed(self.graphcut_solver_combo.currentText())

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _contour_maps_path(self) -> Path | None:
        return (
            self._pos_dir / "3_cell" / "contour_maps.tif"
            if self._pos_dir
            else None
        )

    def _foreground_scores_path(self) -> Path | None:
        return (
            self._pos_dir / "3_cell" / "foreground_scores.tif"
            if self._pos_dir
            else None
        )

    def _foreground_masks_path(self) -> Path | None:
        return (
            self._pos_dir / "3_cell" / "foreground_masks.tif"
            if self._pos_dir
            else None
        )

    def _nucleus_labels_path(self) -> Path | None:
        return (
            self._pos_dir / "2_nucleus" / "tracked_labels.tif"
            if self._pos_dir
            else None
        )

    def _cell_labels_output_path(self) -> Path | None:
        return (
            self._pos_dir / "3_cell" / "tracked_labels.tif"
            if self._pos_dir
            else None
        )

    def _prob_path(self) -> Path | None:
        return (
            self._pos_dir / "1_cellpose" / "cell_prob_3dt.tif"
            if self._pos_dir
            else None
        )

    def _dp_path(self) -> Path | None:
        return (
            self._pos_dir / "1_cellpose" / "cell_dp_3dt.tif"
            if self._pos_dir
            else None
        )

    def _filtered_dp_path(self) -> Path | None:
        return (
            self._pos_dir / "3_cell" / "filtered_dp.tif"
            if self._pos_dir
            else None
        )

    def _graphcut_script_path(self) -> Path:
        return (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "experiment_cell_2d_t_multilabel_graphcut.py"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._refresh_stage_files(pos_dir)
        if pos_dir is None:
            self.correction_widget.deactivate()

    def _refresh_stage_files(self, pos_dir: Path | None = None) -> None:
        if pos_dir is None:
            pos_dir = self._pos_dir
        for files_widget in (
            self.contour_input_files,
            self.contour_output_files,
            self.boundary_selection_input_files,
            self.boundary_selection_output_files,
        ):
            files_widget.refresh(pos_dir)

    def get_state(self) -> dict:
        return {
            "cp_min": self.cp_min_spin.value(),
            "cp_max": self.cp_max_spin.value(),
            "cp_step": self.cp_step_spin.value(),
            "contour_flow_threshold": self.contour_flow_threshold_spin.value(),
            "contour_niter": self.contour_niter_spin.value(),
            "cp_gamma_min": self.cp_gamma_min_spin.value(),
            "cp_gamma_max": self.cp_gamma_max_spin.value(),
            "cp_gamma_step": self.cp_gamma_step_spin.value(),
            "contour_fg_threshold": self.contour_fg_threshold_spin.value(),
            "graphcut_solver": self.graphcut_solver_combo.currentText(),
            "graphcut_unary_mode": self.graphcut_unary_mode_combo.currentText(),
            "graphcut_boundary_mode": self.graphcut_boundary_mode_combo.currentText(),
            "graphcut_n_iters": self.graphcut_n_iters_spin.value(),
            "graphcut_n_workers": self.graphcut_n_workers_spin.value(),
            "graphcut_alpha_unary": self.graphcut_alpha_unary_spin.value(),
            "graphcut_lambda_geodesic": self.graphcut_lambda_geodesic_spin.value(),
            "graphcut_lambda_flow": self.graphcut_lambda_flow_spin.value(),
            "graphcut_lambda_s": self.graphcut_lambda_s_spin.value(),
            "graphcut_beta_s": self.graphcut_beta_s_spin.value(),
            "graphcut_lambda_contour": self.graphcut_lambda_contour_spin.value(),
            "graphcut_lambda_t": self.graphcut_lambda_t_spin.value(),
            "graphcut_init_mode": self.graphcut_init_mode_combo.currentText(),
            "graphcut_min_round_flips": self.graphcut_min_round_flips_spin.value(),
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        spin_values = {
            "cp_min": self.cp_min_spin,
            "cp_max": self.cp_max_spin,
            "cp_step": self.cp_step_spin,
            "contour_flow_threshold": self.contour_flow_threshold_spin,
            "contour_niter": self.contour_niter_spin,
            "cp_gamma_min": self.cp_gamma_min_spin,
            "cp_gamma_max": self.cp_gamma_max_spin,
            "cp_gamma_step": self.cp_gamma_step_spin,
            "contour_fg_threshold": self.contour_fg_threshold_spin,
        }
        for key, widget in spin_values.items():
            if key in state:
                widget.setValue(state[key])

        combo_values = {
            "graphcut_solver": self.graphcut_solver_combo,
            "graphcut_unary_mode": self.graphcut_unary_mode_combo,
            "graphcut_boundary_mode": self.graphcut_boundary_mode_combo,
            "graphcut_init_mode": self.graphcut_init_mode_combo,
        }
        for key, widget in combo_values.items():
            if key in state:
                widget.setCurrentText(str(state[key]))

        graphcut_spin_values = {
            "graphcut_n_iters": self.graphcut_n_iters_spin,
            "graphcut_n_workers": self.graphcut_n_workers_spin,
            "graphcut_alpha_unary": self.graphcut_alpha_unary_spin,
            "graphcut_lambda_geodesic": self.graphcut_lambda_geodesic_spin,
            "graphcut_lambda_flow": self.graphcut_lambda_flow_spin,
            "graphcut_lambda_s": self.graphcut_lambda_s_spin,
            "graphcut_beta_s": self.graphcut_beta_s_spin,
            "graphcut_lambda_contour": self.graphcut_lambda_contour_spin,
            "graphcut_lambda_t": self.graphcut_lambda_t_spin,
            "graphcut_min_round_flips": self.graphcut_min_round_flips_spin,
        }
        for key, widget in graphcut_spin_values.items():
            if key in state:
                widget.setValue(state[key])

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def _set_contour_status(self, msg: str) -> None:
        self.contour_status_lbl.setText(msg)
        self.contour_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_selection_status(self, msg: str) -> None:
        self.boundary_selection_status_lbl.setText(msg)
        self.boundary_selection_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _on_stage_progress(self, bar: QProgressBar, set_status, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            if total > 0:
                bar.setVisible(True)
                bar.setRange(0, total)
                bar.setValue(done)
            set_status(msg)
        else:
            set_status(str(data))

    def _show_layer(self, layer_name: str, data: np.ndarray, kwargs: dict, adder) -> None:
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = data
        else:
            adder(data, name=layer_name, **kwargs)

    def _current_t(self) -> int:
        dims = getattr(self.viewer, "dims", None)
        step = getattr(dims, "current_step", (0,))
        return int(step[0]) if len(step) >= 1 else 0

    # ------------------------------------------------------------------
    # 1. Contour Maps
    # ------------------------------------------------------------------
    def _cellprob_thresholds(self) -> list[float]:
        step = self.cp_step_spin.value()
        return list(
            np.arange(
                self.cp_min_spin.value(),
                self.cp_max_spin.value() + step / 2,
                step,
            )
        )

    def _cp_gammas(self) -> list[float]:
        step = self.cp_gamma_step_spin.value()
        return list(
            np.arange(
                self.cp_gamma_min_spin.value(),
                self.cp_gamma_max_spin.value() + step / 2,
                step,
            )
        )

    def _build_consensus_boundary_averaged(
        self,
        prob_3d: np.ndarray,
        dp_2d: np.ndarray,
        thresholds: list[float],
        gammas: list[float],
        *,
        flow_threshold: float,
        niter: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        boundary_accum = None
        foreground_accum = None
        for gamma in gammas:
            prob_2d = apply_gamma(prob_3d, gamma).mean(axis=0)
            b, fg = build_consensus_boundary_2d(
                prob_2d,
                dp_2d,
                thresholds,
                flow_threshold=flow_threshold,
                reduction="mean",
                niter=niter,
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

    def _on_build_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return

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
        thresholds = self._cellprob_thresholds()
        gammas = self._cp_gammas()
        flow_threshold = self.contour_flow_threshold_spin.value()
        niter = self.contour_niter_spin.value()
        foreground_threshold = self.contour_fg_threshold_spin.value()
        build_fn = self._build_consensus_boundary_averaged

        def _on_done(result):
            self._contour_worker = None
            self._set_contour_buttons_running(False)
            contours, scores, foreground = result
            self._show_layer(
                _CELL_CONTOUR_LAYER,
                contours,
                {"colormap": "magma", "visible": True},
                self.viewer.add_image,
            )
            self._show_layer(
                _CELL_FOREGROUND_SCORE_LAYER,
                scores,
                {"colormap": "viridis", "visible": True},
                self.viewer.add_image,
            )
            self._show_layer(
                _CELL_FOREGROUND_LAYER,
                foreground,
                {},
                self.viewer.add_labels,
            )
            self._refresh_stage_files(pos_dir)
            self._set_contour_status("Contour maps complete.")

        @thread_worker(connect={
            "yielded": lambda data: self._on_stage_progress(
                self.contour_progress_bar, self._set_contour_status, data
            ),
            "returned": _on_done,
            "errored": lambda exc: self._on_contour_error(exc),
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            filtered_dp_stack = np.asarray(tifffile.imread(str(filtered_dp_path)), dtype=np.float32)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis]
            if filtered_dp_stack.ndim == 3:
                filtered_dp_stack = filtered_dp_stack[np.newaxis]

            n_t = min(prob_stack.shape[0], filtered_dp_stack.shape[0])
            contour_frames: list[np.ndarray] = []
            score_frames: list[np.ndarray] = []
            foreground_frames: list[np.ndarray] = []
            for t in range(n_t):
                yield (t + 1, n_t, f"Building contour maps: frame {t + 1}/{n_t}...")
                contour, foreground_score = build_fn(
                    prob_stack[t],          # (Z, Y, X)
                    filtered_dp_stack[t],   # (2, Y, X)
                    thresholds,
                    gammas,
                    flow_threshold=flow_threshold,
                    niter=niter,
                )
                contour_frames.append(contour.astype(np.float32, copy=False))
                foreground_score = foreground_score.astype(np.float32, copy=False)
                score_frames.append(foreground_score)
                foreground_frames.append(
                    (foreground_score >= foreground_threshold).astype(np.uint8)
                )

            contour_arr = np.stack(contour_frames)
            score_arr = np.stack(score_frames)
            foreground_arr = np.stack(foreground_frames)
            contour_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(contour_path), contour_arr, compression="zlib")
            tifffile.imwrite(str(score_path), score_arr, compression="zlib")
            tifffile.imwrite(str(foreground_path), foreground_arr, compression="zlib")
            return contour_arr, score_arr, foreground_arr

        self._set_contour_status(
            f"Building contour maps ({len(thresholds)} thresholds, {len(gammas)} gamma value(s))..."
        )
        self._set_contour_buttons_running(True)
        self._contour_worker = _worker()

    def _on_preview_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return

        prob_path = self._prob_path()
        filtered_dp_path = self._filtered_dp_path()
        for path, name in [
            (prob_path, "cell_prob_3dt.tif"),
            (filtered_dp_path, "filtered_dp.tif"),
        ]:
            if path is None or not path.exists():
                self._set_contour_status(f"Missing: {name}")
                return

        t_frame = self._current_t()
        thresholds = self._cellprob_thresholds()
        gammas = self._cp_gammas()
        flow_threshold = self.contour_flow_threshold_spin.value()
        niter = self.contour_niter_spin.value()
        foreground_threshold = self.contour_fg_threshold_spin.value()
        build_fn = self._build_consensus_boundary_averaged

        def _on_preview_done(result):
            self._contour_worker = None
            self._set_contour_buttons_running(False)
            contour, foreground_score, n_t, t_idx = result
            contour_data = np.zeros((n_t,) + contour.shape, dtype=np.float32)
            contour_data[t_idx] = contour
            score_data = np.zeros((n_t,) + foreground_score.shape, dtype=np.float32)
            score_data[t_idx] = foreground_score
            mask_data = (score_data >= foreground_threshold).astype(np.uint8)
            self._show_layer(
                _CELL_CONTOUR_LAYER,
                contour_data,
                {"colormap": "magma", "visible": True},
                self.viewer.add_image,
            )
            self._show_layer(
                _CELL_FOREGROUND_SCORE_LAYER,
                score_data,
                {"colormap": "viridis", "visible": True},
                self.viewer.add_image,
            )
            self._show_layer(
                _CELL_FOREGROUND_LAYER,
                mask_data,
                {},
                self.viewer.add_labels,
            )
            self._set_contour_status(
                f"Preview t={t_idx} — {len(thresholds)} thresholds, {len(gammas)} gamma value(s)"
            )

        @thread_worker(connect={
            "returned": _on_preview_done,
            "errored": lambda exc: self._on_contour_error(exc),
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            filtered_dp_stack = np.asarray(tifffile.imread(str(filtered_dp_path)), dtype=np.float32)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis]
            if filtered_dp_stack.ndim == 3:
                filtered_dp_stack = filtered_dp_stack[np.newaxis]
            n_t = min(prob_stack.shape[0], filtered_dp_stack.shape[0])
            t_idx = min(max(t_frame, 0), n_t - 1)
            contour, foreground_score = build_fn(
                prob_stack[t_idx],          # (Z, Y, X)
                filtered_dp_stack[t_idx],   # (2, Y, X)
                thresholds,
                gammas,
                flow_threshold=flow_threshold,
                niter=niter,
            )
            return (
                contour.astype(np.float32, copy=False),
                foreground_score.astype(np.float32, copy=False),
                n_t,
                t_idx,
            )

        self._set_contour_status(f"Previewing contour map for frame t={t_frame}...")
        self._set_contour_buttons_running(True)
        self._contour_worker = _worker()

    def _on_contour_error(self, exc: Exception) -> None:
        self._contour_worker = None
        self._set_contour_buttons_running(False)
        self._set_contour_status(f"Error: {exc}")
        logger.exception("Cell contour worker error", exc_info=exc)

    # ------------------------------------------------------------------
    # 2. Track-Conditioned Boundary Selection
    # ------------------------------------------------------------------
    def _set_boundary_selection_running(self, running: bool) -> None:
        self.run_boundary_selection_btn.setEnabled(not running)
        self.boundary_selection_progress_bar.setVisible(running)
        if running:
            self.boundary_selection_progress_bar.setRange(0, 0)
        else:
            self.boundary_selection_progress_bar.setRange(0, 100)
            self.boundary_selection_progress_bar.setValue(0)

    def _on_solver_changed(self, solver: str) -> None:
        enabled = solver != "icm"
        for widget in self._icm_disabled_widgets:
            if widget is not None:
                widget.setEnabled(enabled)

    def _build_graphcut_command(self, timestamp: str) -> tuple[list[str], Path]:
        if self._pos_dir is None:
            raise RuntimeError("No project open.")
        output_dir = self._pos_dir / "4_cell_graphcut" / timestamp
        cmd = [
            sys.executable,
            str(self._graphcut_script_path()),
            "--pos-dir",
            str(self._pos_dir),
            "--solver",
            self.graphcut_solver_combo.currentText(),
            "--unary-mode",
            self.graphcut_unary_mode_combo.currentText(),
            "--flow-field-path",
            str(self._dp_path()),
            "--boundary-mode",
            self.graphcut_boundary_mode_combo.currentText(),
            "--n-iters",
            str(self.graphcut_n_iters_spin.value()),
            "--n-workers",
            str(self.graphcut_n_workers_spin.value()),
            "--alpha-unary",
            str(self.graphcut_alpha_unary_spin.value()),
            "--lambda-geodesic",
            str(self.graphcut_lambda_geodesic_spin.value()),
            "--lambda-flow",
            str(self.graphcut_lambda_flow_spin.value()),
            "--lambda-s",
            str(self.graphcut_lambda_s_spin.value()),
            "--beta-s",
            str(self.graphcut_beta_s_spin.value()),
            "--lambda-contour",
            str(self.graphcut_lambda_contour_spin.value()),
            "--lambda-t",
            str(self.graphcut_lambda_t_spin.value()),
            "--init-mode",
            self.graphcut_init_mode_combo.currentText(),
            "--min-round-flips",
            str(self.graphcut_min_round_flips_spin.value()),
            "--timestamp",
            timestamp,
            "--overwrite",
        ]
        if self.graphcut_boundary_mode_combo.currentText() == "foreground_inverse":
            cmd.extend([
                "--foreground-score-path",
                str(self._foreground_scores_path()),
            ])
        return cmd, output_dir

    def _on_run_boundary_selection(self) -> None:
        if self._pos_dir is None:
            self._set_selection_status("No project open.")
            return

        solver = self.graphcut_solver_combo.currentText()
        required_files = [
            (self._nucleus_labels_path(), "tracked_labels.tif (nucleus)"),
            (self._contour_maps_path(), "contour_maps.tif"),
            (self._foreground_scores_path(), "foreground_scores.tif"),
            (self._foreground_masks_path(), "foreground_masks.tif"),
        ]
        if solver != "icm":
            required_files.append((self._dp_path(), "cell_dp_3dt.tif"))
        for path, name in required_files:
            if path is None or not path.exists():
                self._set_selection_status(f"Missing: {name}")
                return

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if solver == "icm":
            self._run_boundary_selection_icm(timestamp)
        else:
            self._run_boundary_selection_graphcut(timestamp)

    def _run_boundary_selection_graphcut(self, timestamp: str) -> None:
        cmd, output_dir = self._build_graphcut_command(timestamp)
        canonical_output = self._cell_labels_output_path()
        pos_dir = self._pos_dir

        def _on_done(labels: np.ndarray) -> None:
            self._boundary_selection_worker = None
            self._set_boundary_selection_running(False)
            self._show_layer(
                _GRAPHCUT_CELL_LABELS_LAYER,
                labels,
                {"visible": True},
                self.viewer.add_labels,
            )
            self._refresh_stage_files(pos_dir)
            self._set_selection_status(
                f"Graphcut boundary selection complete: {canonical_output}"
            )

        @thread_worker(connect={
            "yielded": lambda data: self._on_stage_progress(
                self.boundary_selection_progress_bar,
                self._set_selection_status,
                data,
            ),
            "returned": _on_done,
            "errored": lambda exc: self._on_boundary_selection_error(exc),
        })
        def _worker():
            yield "Running graphcut boundary selection..."
            completed = subprocess.run(
                cmd,
                cwd=str(self._graphcut_script_path().parents[1]),
                capture_output=True,
                text=True,
            )
            if completed.stdout:
                yield completed.stdout.strip().splitlines()[-1]
            if completed.returncode != 0:
                msg = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(msg or f"Graphcut command failed with exit code {completed.returncode}")

            graphcut_output = output_dir / "cell_labels.tif"
            if not graphcut_output.exists():
                raise FileNotFoundError(graphcut_output)
            if canonical_output is None:
                raise RuntimeError("No project open.")
            canonical_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(graphcut_output, canonical_output)
            labels = np.asarray(tifffile.imread(str(canonical_output)))
            return labels

        self._set_selection_status("Running graphcut boundary selection...")
        self._set_boundary_selection_running(True)
        self._boundary_selection_worker = _worker()

    def _run_boundary_selection_icm(self, timestamp: str) -> None:
        if self._pos_dir is None:
            raise RuntimeError("No project open.")
        from cellflow.segmentation import CellLabelICMParams

        output_dir = self._pos_dir / "4_cell_graphcut" / timestamp
        canonical_output = self._cell_labels_output_path()
        pos_dir = self._pos_dir
        params = CellLabelICMParams(
            alpha_unary=self.graphcut_alpha_unary_spin.value(),
            lambda_s=self.graphcut_lambda_s_spin.value(),
            beta_s=self.graphcut_beta_s_spin.value(),
            lambda_t=self.graphcut_lambda_t_spin.value(),
            n_iters=self.graphcut_n_iters_spin.value(),
            min_round_flips=self.graphcut_min_round_flips_spin.value(),
        )

        def _on_done(labels: np.ndarray) -> None:
            self._boundary_selection_worker = None
            self._set_boundary_selection_running(False)
            self._show_layer(
                _GRAPHCUT_CELL_LABELS_LAYER,
                labels,
                {"visible": True},
                self.viewer.add_labels,
            )
            self._refresh_stage_files(pos_dir)
            self._set_selection_status(
                f"Graphcut boundary selection complete: {canonical_output}"
            )

        @thread_worker(connect={
            "yielded": lambda data: self._on_stage_progress(
                self.boundary_selection_progress_bar,
                self._set_selection_status,
                data,
            ),
            "returned": _on_done,
            "errored": lambda exc: self._on_boundary_selection_error(exc),
        })
        def _worker():
            from cellflow.segmentation.cell_label_icm import run_cell_icm_from_pos_dir

            yield "Running ICM boundary selection..."
            labels = run_cell_icm_from_pos_dir(pos_dir, params)
            graphcut_output = output_dir / "cell_labels.tif"
            graphcut_output.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(graphcut_output), labels, compression="zlib")
            if canonical_output is None:
                raise RuntimeError("No project open.")
            canonical_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(graphcut_output, canonical_output)
            labels = np.asarray(tifffile.imread(str(canonical_output)))
            return labels

        self._set_selection_status("Running ICM boundary selection...")
        self._set_boundary_selection_running(True)
        self._boundary_selection_worker = _worker()

    def _on_boundary_selection_error(self, exc: Exception) -> None:
        self._boundary_selection_worker = None
        self._set_boundary_selection_running(False)
        self._set_selection_status(f"Error: {exc}")
        logger.exception(
            "Boundary selection worker error", exc_info=exc
        )
