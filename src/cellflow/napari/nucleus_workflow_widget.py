"""Nucleus workflow widget for hypothesis generation and tracking in CellFlow v2."""
from __future__ import annotations

import logging
import os
import pickle
import shlex
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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
    QMessageBox,
    QProgressBar,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
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
    add_parameter_grid_row,
    block_grid,
    compact_spinbox,
    danger_button,
    muted_label,
    status_label,
)
from cellflow.segmentation import ContourWatershedParams, compute_contour_watershed
from cellflow.tracking.retracker import retrack_frame_constrained
from cellflow.tracking_ultrack.config import TrackingConfig as UltrackConfig
from cellflow.tracking_ultrack.db_build import build_ultrack_database
from cellflow.tracking_ultrack.export import export_tracked_labels
from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db, _select_solver
from cellflow.tracking_ultrack.linking import run_linking
from cellflow.tracking_ultrack.extend import extend_track, extend_track_from_db
from cellflow.tracking_ultrack.reseed import resolve_with_canonical_segment
from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs
from cellflow.tracking_ultrack.solve import database_has_annotations, run_solve

logger = logging.getLogger(__name__)

try:
    from ultrack.core.segmentation.processing import segment as _ultrack_segment
except ImportError:
    _ultrack_segment = None  # type: ignore[assignment]

_PREVIEW_LAYER = "Preview: Nucleus"
_HYP_LAYER = "Hypothesis: Nucleus"
_TRACKED_LAYER = "Tracked: Nucleus"
_VALIDATED_OVERLAY = "Validated: Nucleus"
_SPOTLIGHT_LAYER = "CellSpotlight"
_VALIDATED_OVERLAY_OPACITY = 0.4
_CONTOUR_LAYER = "Contour Map: Nucleus"
_CELLPROB_LAYER = "Cellprob Map: Nucleus"
_FOREGROUND_SCORE_LAYER = "Foreground Score: Nucleus"
_FOREGROUND_MASK_LAYER = "Foreground Mask: Nucleus"
_CELL_ZAVG_LAYER = "Cell z-avg"
_NUC_ZAVG_LAYER = "Nucleus z-avg"
_ULTRACK_DB_PREVIEW_LAYER = "Ultrack DB Preview"
_ULTRACK_DB_SELECTION_LAYER = "Ultrack DB Selection"
_ULTRACK_DB_ANNOTATION_LAYER = "Ultrack DB Annotations"
_CONTOUR_MAPS_DB_LAYER = "Contour Maps: Nucleus"
_FOREGROUND_MASKS_DB_LAYER = "Foreground Masks: Nucleus"
_CONTOUR_SWEEP_WIDTH = 60
_CONTOUR_SWEEP_MIN_WIDTH = int(_CONTOUR_SWEEP_WIDTH * 0.9)


@dataclass(frozen=True)
class _HierarchyCutState:
    node_ids: tuple[int, ...]
    height: float | None


class NucleusWorkflowWidget(QWidget):
    """Nucleus hypothesis generation and tracking management."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._stop_flag: bool = False
        self._build_worker = None
        self._sweep_worker = None
        self._ultrack_db_preview_cache: dict[
            tuple,
            tuple[np.ndarray, str]
            | tuple[np.ndarray, str, dict[int, float]]
            | tuple[
                np.ndarray,
                str,
                dict[int, float],
                dict[int, int],
                dict[int, int],
            ],
        ] = {}
        self._ultrack_db_height_values_cache: dict[tuple, tuple[float, ...]] = {}
        self._ultrack_db_cut_state_cache: dict[tuple, tuple[_HierarchyCutState, ...]] = {}
        self._ultrack_db_browser_active: bool = False
        self._ultrack_db_frame_initialized: bool = False
        self._ultrack_db_selected_node_id: int | None = None
        self._ultrack_db_selected_frame: int | None = None
        self._ultrack_db_label_to_node_id: dict[int, int] = {}
        self._ultrack_db_node_id_to_label: dict[int, int] = {}
        self._ultrack_db_node_annotations: dict[int, str] = {}
        self._ultrack_db_preview_labels: np.ndarray | None = None
        self._ultrack_db_preview_mouse_callback = None
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

        def _stage_files(group_label: str, entries: list[tuple[str, str]]) -> PipelineFilesWidget:
            return PipelineFilesWidget([(group_label, entries)], viewer=self.viewer)

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

        def _param_grid():
            grid = block_grid(horizontal_spacing=12, vertical_spacing=4)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)
            return grid

        def _param_group_label(text: str) -> QLabel:
            label = QLabel(text)
            label.setStyleSheet("font-weight: 600;")
            return label

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
        cp_params_widget.setMinimumWidth(520)
        cp_params_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        cp_params_lay = QVBoxLayout(cp_params_widget)
        cp_params_lay.setContentsMargins(0, 0, 0, 0)
        cp_params_lay.setSpacing(4)
        cp_params_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.contour_input_files = _stage_files("Inputs", [
            ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
            ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
        ])
        cp_params_lay.addWidget(self.contour_input_files)

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
        self.contour_flow_threshold_spin = QDoubleSpinBox()
        self.contour_flow_threshold_spin.setRange(0.0, 10.0)
        self.contour_flow_threshold_spin.setValue(0.0)
        self.contour_flow_threshold_spin.setDecimals(2)
        self.contour_flow_threshold_spin.setSingleStep(0.1)
        self.contour_flow_threshold_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.contour_flow_threshold_spin.setToolTip(
            "Cellpose flow error threshold passed to compute_masks. 0 disables filtering."
        )
        self.contour_flow_threshold_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

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
        self.contour_fg_threshold_spin = QDoubleSpinBox()
        self.contour_fg_threshold_spin.setRange(0.0, 1.0)
        self.contour_fg_threshold_spin.setValue(0.5)
        self.contour_fg_threshold_spin.setDecimals(2)
        self.contour_fg_threshold_spin.setSingleStep(0.01)
        self.contour_fg_threshold_spin.setToolTip(
            "Threshold applied to the fuzzy foreground score written by Contour Maps"
        )
        self.save_source_check = QCheckBox("Save label images")
        self.save_source_check.setToolTip("Save all label images used for contour building in 2_nucleus/source_labels/")
        self.save_source_check.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        contour_sweep_grid = _param_grid()
        add_parameter_grid_row(contour_sweep_grid, 0, 0, "Cellprob min:", self.cp_min_spin)
        add_parameter_grid_row(contour_sweep_grid, 0, 1, "Cellprob max:", self.cp_max_spin)
        add_parameter_grid_row(contour_sweep_grid, 1, 0, "Cellprob step:", self.cp_step_spin)
        add_parameter_grid_row(contour_sweep_grid, 1, 1, "Flow threshold:", self.contour_flow_threshold_spin)
        cp_params_lay.addWidget(_param_group_label("Cellpose mask sweep"))
        cp_params_lay.addLayout(contour_sweep_grid)

        contour_gamma_grid = _param_grid()
        add_parameter_grid_row(contour_gamma_grid, 0, 0, "Gamma min:", self.cp_gamma_min_spin)
        add_parameter_grid_row(contour_gamma_grid, 0, 1, "Gamma max:", self.cp_gamma_max_spin)
        add_parameter_grid_row(contour_gamma_grid, 1, 0, "Gamma step:", self.cp_gamma_step_spin)
        cp_params_lay.addWidget(_param_group_label("Gamma averaging"))
        cp_params_lay.addLayout(contour_gamma_grid)

        contour_output_grid = _param_grid()
        add_parameter_grid_row(contour_output_grid, 0, 0, "FG threshold:", self.contour_fg_threshold_spin)
        contour_output_grid.addWidget(
            self.save_source_check,
            0,
            2,
            1,
            2,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        cp_params_lay.addWidget(_param_group_label("Foreground output"))
        cp_params_lay.addLayout(contour_output_grid)
        for spin in (
            self.cp_min_spin,
            self.cp_max_spin,
            self.cp_step_spin,
            self.contour_flow_threshold_spin,
            self.cp_gamma_min_spin,
            self.cp_gamma_max_spin,
            self.cp_gamma_step_spin,
            self.contour_fg_threshold_spin,
        ):
            spin.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )

        self.preview_contour_btn = QPushButton("Preview")
        self.preview_contour_btn.setToolTip(
            "Build contour maps for the current frame only and display in napari"
        )
        self.build_btn = QPushButton("Build")
        self.contour_terminal_btn = QPushButton("Run in Terminal")
        self.cancel_build_btn = QPushButton("Cancel")
        self.cancel_build_btn.setEnabled(False)

        for button in (
            self.preview_contour_btn,
            self.build_btn,
            self.contour_terminal_btn,
            self.cancel_build_btn,
        ):
            button.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )

        contour_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(
            contour_btn_row,
            0,
            self.preview_contour_btn,
            self.build_btn,
            self.contour_terminal_btn,
            self.cancel_build_btn,
        )
        cp_params_lay.addLayout(contour_btn_row)

        contour_filter_grid = _param_grid()
        self.contour_filter_median_time_spin = QSpinBox()
        self.contour_filter_median_time_spin.setRange(1, 15)
        self.contour_filter_median_time_spin.setValue(1)
        self.contour_filter_median_time_spin.setSingleStep(1)
        self.contour_filter_median_space_spin = QSpinBox()
        self.contour_filter_median_space_spin.setRange(1, 15)
        self.contour_filter_median_space_spin.setValue(1)
        self.contour_filter_median_space_spin.setSingleStep(1)
        self.contour_filter_gauss_time_spin = QDoubleSpinBox()
        self.contour_filter_gauss_time_spin.setRange(0.0, 10.0)
        self.contour_filter_gauss_time_spin.setValue(0.0)
        self.contour_filter_gauss_time_spin.setDecimals(1)
        self.contour_filter_gauss_time_spin.setSingleStep(0.1)
        self.contour_filter_gauss_space_spin = QDoubleSpinBox()
        self.contour_filter_gauss_space_spin.setRange(0.0, 10.0)
        self.contour_filter_gauss_space_spin.setValue(0.0)
        self.contour_filter_gauss_space_spin.setDecimals(1)
        self.contour_filter_gauss_space_spin.setSingleStep(0.1)
        for spin in (
            self.contour_filter_median_time_spin,
            self.contour_filter_median_space_spin,
            self.contour_filter_gauss_time_spin,
            self.contour_filter_gauss_space_spin,
        ):
            spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            spin.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        add_parameter_grid_row(contour_filter_grid, 0, 0, "Median t kernel:", self.contour_filter_median_time_spin)
        add_parameter_grid_row(contour_filter_grid, 0, 1, "Median xy kernel:", self.contour_filter_median_space_spin)
        add_parameter_grid_row(contour_filter_grid, 1, 0, "Gaussian t sigma:", self.contour_filter_gauss_time_spin)
        add_parameter_grid_row(contour_filter_grid, 1, 1, "Gaussian xy sigma:", self.contour_filter_gauss_space_spin)
        for spin in (
            self.contour_filter_median_time_spin,
            self.contour_filter_median_space_spin,
            self.contour_filter_gauss_time_spin,
            self.contour_filter_gauss_space_spin,
        ):
            spin.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        cp_params_lay.addWidget(_param_group_label("Post-filter contour maps"))
        cp_params_lay.addLayout(contour_filter_grid)

        self.preview_contour_filter_btn = QPushButton("Preview Filter")
        self.preview_contour_filter_btn.setToolTip(
            "Preview filtered contour_maps.tif in napari without overwriting it"
        )
        self.run_contour_filter_btn = QPushButton("Run Filter")
        self.run_contour_filter_btn.setToolTip(
            "Filter contour_maps.tif and overwrite contour_maps.tif"
        )
        for button in (self.preview_contour_filter_btn, self.run_contour_filter_btn):
            button.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        contour_filter_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(
            contour_filter_btn_row,
            0,
            self.preview_contour_filter_btn,
            self.run_contour_filter_btn,
        )
        cp_params_lay.addLayout(contour_filter_btn_row)

        self.contour_status_lbl = _stage_status()
        cp_params_lay.addWidget(self.contour_status_lbl)

        self.build_progress_bar = _stage_progress()
        self.contour_output_files = _stage_files("Outputs", [
            ("2_nucleus/contour_maps.tif", "Contour maps"),
            ("2_nucleus/foreground_scores.tif", "Foreground scores"),
            ("2_nucleus/foreground_masks.tif", "Foreground masks"),
        ])
        cp_params_lay.addWidget(self.build_progress_bar)
        cp_params_lay.addWidget(self.contour_output_files)

        cp_params_scroll.setWidget(cp_params_widget)
        contour_lay.addWidget(cp_params_scroll)
        self.contour_section = CollapsibleSection(
            "1. Contour Maps", _contour_inner, expanded=False
        )
        layout.addWidget(self.contour_section)

        # ── 2. Ultrack Database Generation ────────────────────────────────
        _db_gen_inner = QWidget()
        db_gen_lay = QVBoxLayout(_db_gen_inner)
        db_gen_lay.setContentsMargins(0, 0, 0, 0)
        db_gen_lay.setSpacing(4)
        db_gen_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.db_gen_input_files = _stage_files("Inputs", [
            ("2_nucleus/contour_maps.tif", "Contour maps"),
            ("2_nucleus/foreground_masks.tif", "Foreground masks"),
            ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
        ])
        db_gen_lay.addWidget(self.db_gen_input_files)

        db_gen_grid = block_grid(horizontal_spacing=12)
        db_gen_grid.setContentsMargins(0, 0, 0, 0)

        self.db_gen_min_area_spin = QSpinBox()
        self.db_gen_min_area_spin.setRange(0, 1_000_000)
        self.db_gen_min_area_spin.setValue(300)

        self.db_gen_max_area_spin = QSpinBox()
        self.db_gen_max_area_spin.setRange(0, 10_000_000)
        self.db_gen_max_area_spin.setValue(100_000)

        self.db_gen_fg_thr_spin = QDoubleSpinBox()
        self.db_gen_fg_thr_spin.setRange(-5.0, 1.0)
        self.db_gen_fg_thr_spin.setValue(0.5)
        self.db_gen_fg_thr_spin.setDecimals(2)
        self.db_gen_fg_thr_spin.setSingleStep(0.05)
        self.db_gen_fg_thr_spin.setToolTip(
            "Pixel-level foreground threshold for ultrack segmentation (threshold in segmentation_config)"
        )

        self.db_gen_min_frontier_spin = QDoubleSpinBox()
        self.db_gen_min_frontier_spin.setRange(0.0, 1.0)
        self.db_gen_min_frontier_spin.setValue(0.0)
        self.db_gen_min_frontier_spin.setDecimals(3)
        self.db_gen_min_frontier_spin.setSingleStep(0.01)
        self.db_gen_min_frontier_spin.setToolTip(
            "Minimum boundary fraction to keep a candidate (min_frontier in segmentation_config)"
        )

        self.db_gen_ws_hierarchy_combo = QComboBox()
        self.db_gen_ws_hierarchy_combo.addItems(["area", "dynamics", "volume"])

        self.db_gen_n_workers_spin = QSpinBox()
        self.db_gen_n_workers_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.db_gen_n_workers_spin.setValue(1)
        self.db_gen_n_workers_spin.setToolTip("Parallel workers for segmentation")

        self.db_gen_max_dist_spin = QDoubleSpinBox()
        self.db_gen_max_dist_spin.setRange(0.0, 500.0)
        self.db_gen_max_dist_spin.setValue(15.0)
        self.db_gen_max_dist_spin.setDecimals(1)

        self.db_gen_max_neighbors_spin = QSpinBox()
        self.db_gen_max_neighbors_spin.setRange(1, 50)
        self.db_gen_max_neighbors_spin.setValue(5)

        self.db_gen_linking_mode_combo = QComboBox()
        self.db_gen_linking_mode_combo.addItems(["default", "iou"])

        self.db_gen_iou_weight_spin = QDoubleSpinBox()
        self.db_gen_iou_weight_spin.setRange(0.0, 1.0)
        self.db_gen_iou_weight_spin.setValue(1.0)
        self.db_gen_iou_weight_spin.setDecimals(2)
        self.db_gen_iou_weight_spin.setEnabled(False)

        self.db_gen_quality_weight_spin = QDoubleSpinBox()
        self.db_gen_quality_weight_spin.setRange(0.0, 10.0)
        self.db_gen_quality_weight_spin.setValue(1.0)
        self.db_gen_quality_weight_spin.setDecimals(2)
        self.db_gen_quality_weight_spin.setSingleStep(0.05)
        self.db_gen_quality_weight_spin.setToolTip(
            "Weight applied to signal-based segmentation quality before storing node_prob"
        )

        self.db_gen_quality_exp_spin = QDoubleSpinBox()
        self.db_gen_quality_exp_spin.setRange(0.1, 50.0)
        self.db_gen_quality_exp_spin.setValue(8.0)
        self.db_gen_quality_exp_spin.setDecimals(2)
        self.db_gen_quality_exp_spin.setToolTip(
            "Raises signal-based quality before storing as node_prob"
        )

        self.db_gen_circularity_weight_spin = QDoubleSpinBox()
        self.db_gen_circularity_weight_spin.setRange(0.0, 10.0)
        self.db_gen_circularity_weight_spin.setValue(0.25)
        self.db_gen_circularity_weight_spin.setDecimals(2)
        self.db_gen_circularity_weight_spin.setSingleStep(0.05)
        self.db_gen_circularity_weight_spin.setToolTip(
            "Weight applied to shape circularity before storing node_prob"
        )

        self.db_gen_power_spin = QDoubleSpinBox()
        self.db_gen_power_spin.setRange(0.1, 20.0)
        self.db_gen_power_spin.setValue(4.0)
        self.db_gen_power_spin.setDecimals(2)
        self.db_gen_power_spin.setToolTip(
            "Deprecated duplicate of the solver power control; solver transform for stored weights"
        )
        self.db_gen_power_spin.setVisible(False)

        add_block_pair_row(db_gen_grid, 0, "Min Area (px):", _compact(self.db_gen_min_area_spin), "Max Area (px):", _compact(self.db_gen_max_area_spin))
        add_block_pair_row(db_gen_grid, 1, "FG Threshold:", _compact(self.db_gen_fg_thr_spin), "Min Frontier:", _compact(self.db_gen_min_frontier_spin))
        add_block_pair_row(db_gen_grid, 2, "WS Hierarchy:", self.db_gen_ws_hierarchy_combo, "N Workers:", _compact(self.db_gen_n_workers_spin))
        add_block_pair_row(db_gen_grid, 3, "Max Distance (px):", _compact(self.db_gen_max_dist_spin), "Max Neighbors:", _compact(self.db_gen_max_neighbors_spin))
        add_block_pair_row(db_gen_grid, 4, "Linking Mode:", self.db_gen_linking_mode_combo, "IoU Weight:", _compact(self.db_gen_iou_weight_spin))
        add_block_pair_row(db_gen_grid, 5, "Quality Weight:", _compact(self.db_gen_quality_weight_spin), "Quality Exp:", _compact(self.db_gen_quality_exp_spin))
        add_block_pair_row(db_gen_grid, 6, "Circularity Weight:", _compact(self.db_gen_circularity_weight_spin), "", QWidget())
        db_gen_lay.addLayout(db_gen_grid)

        self.db_gen_use_validated_check = QCheckBox("Use validated corrections")
        db_gen_validated_grid = block_grid(horizontal_spacing=12)
        add_block_checkbox_row(db_gen_validated_grid, 0, self.db_gen_use_validated_check)
        db_gen_lay.addLayout(db_gen_validated_grid)

        db_gen_run_row = block_grid(horizontal_spacing=12)
        self.run_db_gen_btn = QPushButton("Run DB Generation")
        self.run_db_gen_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.db_gen_terminal_btn = QPushButton("Run in Terminal")
        self.db_gen_terminal_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        add_block_button_row(db_gen_run_row, 0, self.run_db_gen_btn, self.db_gen_terminal_btn)
        db_gen_lay.addLayout(db_gen_run_row)

        self.db_gen_status_lbl = _stage_status()
        db_gen_lay.addWidget(self.db_gen_status_lbl)

        self.db_gen_progress_bar = _stage_progress()
        db_gen_lay.addWidget(self.db_gen_progress_bar)

        self.db_gen_output_files = _stage_files("Outputs", [
            ("2_nucleus/ultrack_workdir/data.db", "Ultrack database"),
        ])
        db_gen_lay.addWidget(self.db_gen_output_files)

        self.db_gen_section = CollapsibleSection(
            "2. Ultrack Database Generation", _db_gen_inner, expanded=False
        )
        layout.addWidget(self.db_gen_section)

        # ── Optional Ultrack Database Browser ──────────────────────────────
        _ultrack_db_browser_inner = QWidget()
        ultrack_db_browser_lay = QVBoxLayout(_ultrack_db_browser_inner)
        ultrack_db_browser_lay.setContentsMargins(0, 0, 0, 0)
        ultrack_db_browser_lay.setSpacing(4)

        from qtpy.QtGui import QIcon
        from qtpy.QtCore import Qt as _Qt
        self.ultrack_db_info_lbl = QLabel("—")
        self.ultrack_db_info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ultrack_db_info_lbl.setWordWrap(True)
        self.ultrack_db_info_lbl.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        ultrack_db_browser_lay.addWidget(self.ultrack_db_info_lbl)

        ultrack_db_grid = block_grid(horizontal_spacing=12)
        ultrack_db_grid.setContentsMargins(0, 0, 0, 0)
        self.ultrack_db_mode_combo = QComboBox()
        self.ultrack_db_mode_combo.addItems([
            "Summary only",
            "Hierarchy cut",
        ])
        add_block_pair_row(
            ultrack_db_grid,
            0,
            "Mode:",
            self.ultrack_db_mode_combo,
            "",
            QWidget(),
        )

        self.ultrack_db_hierarchy_slider = QSlider(_Qt.Horizontal)
        self.ultrack_db_hierarchy_slider.setRange(0, 100)
        self.ultrack_db_hierarchy_slider.setValue(50)
        self.ultrack_db_hierarchy_slider.setToolTip(
            "Hierarchy cut level: 0 = most split, 1 = most merged"
        )
        self.ultrack_db_height_lbl = QLabel("0.50")
        self.ultrack_db_height_lbl.setFixedWidth(48)
        self._ultrack_db_slider_row = QWidget()
        _slider_lay = QHBoxLayout(self._ultrack_db_slider_row)
        _slider_lay.setContentsMargins(0, 0, 0, 0)
        _slider_lay.addWidget(self.ultrack_db_hierarchy_slider)
        _slider_lay.addWidget(self.ultrack_db_height_lbl)
        ultrack_db_grid.addWidget(self._ultrack_db_slider_row, 1, 0, 1, 4)
        self._ultrack_db_slider_row.setVisible(False)

        ultrack_db_browser_lay.addLayout(ultrack_db_grid)

        _db_btn_row = QWidget()
        _db_btn_lay = QHBoxLayout(_db_btn_row)
        _db_btn_lay.setContentsMargins(0, 0, 0, 0)
        _db_btn_lay.setSpacing(4)
        self.ultrack_db_active_btn = QPushButton("Activate")
        self.ultrack_db_active_btn.setCheckable(True)
        self.ultrack_db_active_btn.setChecked(False)
        self.ultrack_db_active_btn.setToolTip("Load contour maps and foreground masks into viewer and enable DB preview")
        self.ultrack_db_refresh_btn = QPushButton()
        self.ultrack_db_refresh_btn.setToolTip("Refresh Ultrack database browser")
        self.ultrack_db_refresh_btn.setIcon(QIcon.fromTheme("view-refresh"))
        self.ultrack_db_refresh_btn.setEnabled(False)
        _db_btn_lay.addWidget(self.ultrack_db_active_btn)
        _db_btn_lay.addWidget(self.ultrack_db_refresh_btn)
        ultrack_db_browser_lay.addWidget(_db_btn_row)
        self.ultrack_db_mode_combo.setEnabled(False)
        self.ultrack_db_hierarchy_slider.setEnabled(False)
        self.ultrack_db_prob_alpha_check = QCheckBox("Node prob transparency")
        self.ultrack_db_prob_alpha_check.setToolTip("Modulate label opacity by node probability (higher quality = more opaque)")
        self.ultrack_db_prob_alpha_check.setEnabled(False)
        self.ultrack_db_connected_focus_check = QCheckBox("Connected focus")
        self.ultrack_db_connected_focus_check.setToolTip(
            "Focus the DB preview on a selected node and its temporal neighbors"
        )
        self.ultrack_db_connected_focus_check.setEnabled(False)
        self.ultrack_db_edge_alpha_check = QCheckBox("Edge weight transparency")
        self.ultrack_db_edge_alpha_check.setToolTip(
            "Modulate connected-neighbor opacity by link weight"
        )
        self.ultrack_db_edge_alpha_check.setEnabled(False)
        self.ultrack_db_show_validated_check = QCheckBox("Show validated nodes")
        self.ultrack_db_show_validated_check.setChecked(True)
        self.ultrack_db_show_validated_check.setEnabled(False)
        self.ultrack_db_show_fake_check = QCheckBox("Show fake nodes")
        self.ultrack_db_show_fake_check.setChecked(False)
        self.ultrack_db_show_fake_check.setEnabled(False)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_prob_alpha_check)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_connected_focus_check)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_edge_alpha_check)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_show_validated_check)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_show_fake_check)

        self.ultrack_db_section_status_lbl = QLabel("")
        self.ultrack_db_section_status_lbl.setWordWrap(True)
        self.ultrack_db_section_status_lbl.setVisible(False)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_section_status_lbl)

        self.ultrack_db_browser_section = CollapsibleSection(
            "Ultrack Database Browser", _ultrack_db_browser_inner, expanded=False
        )

        # ── 4. Ultrack Tracking ───────────────────────────────────────────

        _ultrack_inner = QWidget()
        ultrack_lay = QVBoxLayout(_ultrack_inner)
        ultrack_lay.setContentsMargins(0, 0, 0, 0)
        ultrack_lay.setSpacing(4)
        ultrack_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.ultrack_input_files = _stage_files("Inputs", [
            ("2_nucleus/ultrack_workdir/data.db", "Ultrack database"),
        ])
        ultrack_lay.addWidget(self.ultrack_input_files)

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
        self.ultrack_route_check.setVisible(False)
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

        self.ultrack_status_lbl = _stage_status()
        ultrack_lay.addWidget(self.ultrack_status_lbl)

        self.ultrack_progress_bar = _stage_progress()
        ultrack_lay.addWidget(self.ultrack_progress_bar)

        self.ultrack_output_files = _stage_files("Outputs", [
            ("2_nucleus/tracked_labels.tif", "Tracked labels"),
        ])
        ultrack_lay.addWidget(self.ultrack_output_files)

        ultrack_attrib = QLabel(
            "Ultrack tracking is powered by the "
            '<a href="https://github.com/royerlab/ultrack">Ultrack</a> project.'
        )
        ultrack_attrib.setOpenExternalLinks(True)
        ultrack_attrib.setWordWrap(True)
        muted_label(ultrack_attrib, size_pt=9)
        ultrack_lay.addWidget(ultrack_attrib)

        self.ultrack_section = CollapsibleSection(
            "4. Ultrack Tracking", _ultrack_inner, expanded=False
        )
        layout.addWidget(self.ultrack_section)

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
        self.extend_area_weight_spin = QDoubleSpinBox()
        self.extend_area_weight_spin.setRange(0.0, 10.0)
        self.extend_area_weight_spin.setValue(1.0)
        self.extend_area_weight_spin.setSingleStep(0.1)
        self.extend_area_weight_spin.setDecimals(2)
        self.extend_iou_weight_spin = QDoubleSpinBox()
        self.extend_iou_weight_spin.setRange(0.0, 10.0)
        self.extend_iou_weight_spin.setValue(1.0)
        self.extend_iou_weight_spin.setSingleStep(0.1)
        self.extend_iou_weight_spin.setDecimals(2)
        self.extend_distance_weight_spin = QDoubleSpinBox()
        self.extend_distance_weight_spin.setRange(0.0, 10.0)
        self.extend_distance_weight_spin.setValue(0.25)
        self.extend_distance_weight_spin.setSingleStep(0.05)
        self.extend_distance_weight_spin.setDecimals(2)
        self.extend_overlap_penalty_spin = QDoubleSpinBox()
        self.extend_overlap_penalty_spin.setRange(0.0, 10.0)
        self.extend_overlap_penalty_spin.setValue(1.0)
        self.extend_overlap_penalty_spin.setSingleStep(0.1)
        self.extend_overlap_penalty_spin.setDecimals(2)
        self.extend_greedy_overwrite_check = QCheckBox("Greedy overwrite")
        add_block_pair_row(
            extend_params_form,
            0,
            "Max Distance (px):",
            _compact(self.extend_max_dist_spin, 80),
            "Area Weight:",
            _compact(self.extend_area_weight_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            extend_params_form,
            1,
            "IoU Weight:",
            _compact(self.extend_iou_weight_spin, 80),
            "Distance Weight:",
            _compact(self.extend_distance_weight_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            extend_params_form,
            2,
            "Overlap Penalty:",
            _compact(self.extend_overlap_penalty_spin, 80),
            field_width=80,
        )
        add_block_checkbox_row(extend_params_form, 3, self.extend_greedy_overwrite_check)
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

        self.correction_status_lbl = QLabel("")
        self.correction_status_lbl.setWordWrap(True)
        self.correction_status_lbl.setVisible(False)
        _corr_inner_lay.addWidget(self.correction_status_lbl)

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
            expanded=False,
        )
        _corr_inner_lay.addWidget(self.correction_shortcuts_section)

        self.correction_section = CollapsibleSection(
            "5. Correction", _corr_inner, expanded=False
        )
        layout.addWidget(self.correction_section)
        layout.addWidget(self.ultrack_db_browser_section)

    # ──────────────────────────────────────────────────────────────────────────
    # Signal wiring
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.build_btn.clicked.connect(self._on_build_contour_maps)
        self.preview_contour_btn.clicked.connect(self._on_preview_contour_maps)
        self.contour_terminal_btn.clicked.connect(self._on_run_contour_terminal)
        self.preview_contour_filter_btn.clicked.connect(self._on_preview_contour_filter)
        self.run_contour_filter_btn.clicked.connect(self._on_run_contour_filter)
        self.cancel_build_btn.clicked.connect(self._on_cancel_build)
        self.run_db_gen_btn.clicked.connect(self._on_run_db_generation)
        self.db_gen_terminal_btn.clicked.connect(self._on_db_gen_terminal)
        self.db_gen_linking_mode_combo.currentTextChanged.connect(self._on_db_gen_mode_changed)
        self.db_gen_use_validated_check.toggled.connect(self._set_resolve_prior_controls_enabled)
        self.ultrack_db_active_btn.toggled.connect(self._on_ultrack_db_activate)
        self.ultrack_db_refresh_btn.clicked.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_mode_combo.currentTextChanged.connect(self._on_ultrack_db_mode_changed)
        self.ultrack_db_hierarchy_slider.valueChanged.connect(self._on_ultrack_db_slider_changed)
        self.ultrack_db_prob_alpha_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_connected_focus_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_edge_alpha_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_show_validated_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_show_fake_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.run_ultrack_btn.clicked.connect(self._on_run_tracking_route)
        self.ultrack_terminal_btn.clicked.connect(self._on_run_tracking_route_terminal)
        self.save_tracked_btn.clicked.connect(self._on_save_tracked)
        self.load_tracked_btn.clicked.connect(self._on_load_tracked)
        self.reassign_ids_btn.clicked.connect(self._on_reassign_ids)
        self.ultrack_linking_mode_combo.currentTextChanged.connect(self._on_ultrack_mode_changed)
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
        self._set_resolve_prior_controls_enabled(self.db_gen_use_validated_check.isChecked())

    # ──────────────────────────────────────────────────────────────────────────
    # Public refresh
    # ──────────────────────────────────────────────────────────────────────────

    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._refresh_stage_files(pos_dir)
        if pos_dir is None:
            self.correction_widget.deactivate()
            return
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _refresh_stage_files(self, pos_dir: Path | None = None) -> None:
        if pos_dir is None:
            pos_dir = self._pos_dir
        for files_widget in (
            self.contour_input_files,
            self.contour_output_files,
            self.db_gen_input_files,
            self.db_gen_output_files,
            self.ultrack_input_files,
            self.ultrack_output_files,
        ):
            files_widget.refresh(pos_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Path helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _tracked_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    def _prob_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif" if self._pos_dir else None

    def _dp_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif" if self._pos_dir else None

    def _contour_maps_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "contour_maps.tif" if self._pos_dir else None

    def _foreground_scores_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "foreground_scores.tif" if self._pos_dir else None

    def _cell_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "cell_zavg.tif" if self._pos_dir else None

    def _nucleus_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "nucleus_zavg.tif" if self._pos_dir else None

    def _ultrack_workdir(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "ultrack_workdir" if self._pos_dir else None

    def _ultrack_db_path(self) -> Path | None:
        workdir = self._ultrack_workdir()
        return workdir / "data.db" if workdir else None

    def _foreground_masks_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "foreground_masks.tif" if self._pos_dir else None

    def _nucleus_prob_zavg_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif" if self._pos_dir else None

    # ── DB Generation section ─────────────────────────────────────────────────

    def _db_gen_config_from_controls(self) -> UltrackConfig:
        return UltrackConfig(
            seg_min_area=self.db_gen_min_area_spin.value(),
            seg_max_area=self.db_gen_max_area_spin.value(),
            seg_foreground_threshold=self.db_gen_fg_thr_spin.value(),
            seg_min_frontier=self.db_gen_min_frontier_spin.value(),
            seg_ws_hierarchy=self.db_gen_ws_hierarchy_combo.currentText(),
            seg_n_workers=self.db_gen_n_workers_spin.value(),
            max_distance=self.db_gen_max_dist_spin.value(),
            max_neighbors=self.db_gen_max_neighbors_spin.value(),
            linking_mode=self.db_gen_linking_mode_combo.currentText(),
            iou_weight=self.db_gen_iou_weight_spin.value(),
            quality_weight=self.db_gen_quality_weight_spin.value(),
            quality_exponent=self.db_gen_quality_exp_spin.value(),
            circularity_weight=self.db_gen_circularity_weight_spin.value(),
            link_n_workers=self.db_gen_n_workers_spin.value(),
            seed_weight=self.ultrack_seed_weight_spin.value(),
            seed_sigma_space=self.ultrack_seed_space_spin.value(),
            seed_tau_time=self.ultrack_seed_time_spin.value(),
            seed_max_dt=self.ultrack_seed_window_spin.value(),
        )

    def _on_run_db_generation(self) -> None:
        if self._pos_dir is None:
            self._set_db_gen_status("No project open.")
            return
        contour_path = self._contour_maps_path()
        fg_path = self._foreground_masks_path()
        nuc_zavg_path = self._nucleus_prob_zavg_path()
        if contour_path is None or not contour_path.exists():
            self._set_db_gen_status("Missing: contour_maps.tif — run Contour Maps first.")
            return
        if fg_path is None or not fg_path.exists():
            self._set_db_gen_status(
                "Missing: foreground_masks.tif (foreground mask) — run Contour Maps first."
            )
            return
        if nuc_zavg_path is None or not nuc_zavg_path.exists():
            self._set_db_gen_status("Missing: nucleus_prob_zavg.tif — run Cellpose first.")
            return
        if _ultrack_segment is None:
            self._set_db_gen_status("ultrack not installed — activate the cellflow conda environment.")
            return

        cfg = self._db_gen_config_from_controls()
        working_dir = self._ultrack_workdir()
        pos_dir = self._pos_dir
        use_validated = self.db_gen_use_validated_check.isChecked()
        validated_tracks: dict[int, set[int]] | None = None
        tracked_labels: np.ndarray | None = None
        if use_validated:
            validated_tracks = read_validated_tracks(pos_dir)
            if not validated_tracks:
                self._set_db_gen_status("No validated tracks found — validate some cells first (press V).")
                return
            if _TRACKED_LAYER not in self.viewer.layers:
                self._set_db_gen_status("No tracked layer loaded for validated DB generation.")
                return
            tracked_labels = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)

        self.db_gen_progress_bar.setRange(0, 0)
        self.db_gen_progress_bar.setVisible(True)
        self._set_db_gen_status("Starting DB generation…")
        self.run_db_gen_btn.setEnabled(False)
        self.db_gen_terminal_btn.setEnabled(False)

        @thread_worker(connect={
            "yielded": self._on_db_gen_progress,
            "returned": self._on_db_gen_done,
            "errored": self._on_db_gen_worker_error,
        })
        def _worker():
            import queue as _queue
            import threading

            msg_queue: _queue.SimpleQueue = _queue.SimpleQueue()
            result_holder: list = []
            exc_holder: list = []

            def _progress(msg: str) -> None:
                msg_queue.put(msg)

            def _run() -> None:
                try:
                    result_holder.append(
                        build_ultrack_database(
                            contour_maps_path=contour_path,
                            foreground_masks_path=fg_path,
                            nucleus_prob_zavg_path=nuc_zavg_path,
                            working_dir=working_dir,
                            cfg=cfg,
                            validated_tracks=validated_tracks,
                            tracked_labels=tracked_labels,
                            use_validated=use_validated,
                            progress_cb=_progress,
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
            return pos_dir

        _worker()

    def _on_db_gen_terminal(self) -> None:
        import sys
        import tempfile

        if self._pos_dir is None:
            self._set_db_gen_status("No project open.")
            return
        contour_path = self._contour_maps_path()
        fg_path = self._foreground_masks_path()
        nuc_zavg_path = self._nucleus_prob_zavg_path()
        if contour_path is None or not contour_path.exists():
            self._set_db_gen_status("Missing: contour_maps.tif")
            return
        if fg_path is None or not fg_path.exists():
            self._set_db_gen_status("Missing: foreground_masks.tif (foreground mask) — run Contour Maps first.")
            return
        if nuc_zavg_path is None or not nuc_zavg_path.exists():
            self._set_db_gen_status("Missing: nucleus_prob_zavg.tif")
            return

        cfg = self._db_gen_config_from_controls()
        working_dir = self._ultrack_workdir()
        use_validated = self.db_gen_use_validated_check.isChecked()
        tracked_path = self._tracked_path()
        if use_validated:
            validated_tracks = read_validated_tracks(self._pos_dir)
            if not validated_tracks:
                self._set_db_gen_status("No validated tracks found — validate some cells first (press V).")
                return
            if tracked_path is None or not tracked_path.exists():
                self._set_db_gen_status("Tracked labels not found for validated DB generation.")
                return

        python_code = (
            "import pathlib, sys\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))\n"
            "from cellflow.database.tracked import read_full_tracked_stack\n"
            "from cellflow.database.validation import read_validated_tracks\n"
            "from cellflow.tracking_ultrack.config import TrackingConfig\n"
            "from cellflow.tracking_ultrack.db_build import build_ultrack_database\n"
            "from cellflow.tracking_ultrack.linking import run_linking\n"
            "from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs\n"
            "from ultrack.core.segmentation.processing import segment as _ultrack_segment\n"
            "\n"
            "if __name__ == '__main__':\n"
            f"    pos_dir = pathlib.Path({str(self._pos_dir)!r})\n"
            f"    contour_path = pathlib.Path({str(contour_path)!r})\n"
            f"    foreground_masks_path = pathlib.Path({str(fg_path)!r})\n"
            f"    nucleus_prob_zavg_path = pathlib.Path({str(nuc_zavg_path)!r})\n"
            f"    working_dir = pathlib.Path({str(working_dir)!r})\n"
            f"    tracked_path = pathlib.Path({str(tracked_path)!r})\n"
            f"    use_validated = {bool(use_validated)!r}\n"
            "    cfg = TrackingConfig(\n"
            f"        seg_min_area={cfg.seg_min_area},\n"
            f"        seg_max_area={cfg.seg_max_area},\n"
            f"        seg_foreground_threshold={cfg.seg_foreground_threshold},\n"
            f"        seg_min_frontier={cfg.seg_min_frontier},\n"
            f"        seg_ws_hierarchy={cfg.seg_ws_hierarchy!r},\n"
            f"        seg_n_workers={cfg.seg_n_workers},\n"
            f"        max_distance={cfg.max_distance},\n"
            f"        max_neighbors={cfg.max_neighbors},\n"
            f"        linking_mode={cfg.linking_mode!r},\n"
            f"        iou_weight={cfg.iou_weight},\n"
            f"        quality_weight={cfg.quality_weight},\n"
            f"        quality_exponent={cfg.quality_exponent},\n"
            f"        circularity_weight={cfg.circularity_weight},\n"
            f"        link_n_workers={cfg.link_n_workers},\n"
            f"        seed_weight={cfg.seed_weight},\n"
            f"        seed_sigma_space={cfg.seed_sigma_space},\n"
            f"        seed_tau_time={cfg.seed_tau_time},\n"
            f"        seed_max_dt={cfg.seed_max_dt},\n"
            "    )\n"
            "    validated_tracks = read_validated_tracks(pos_dir) if use_validated else None\n"
            "    tracked_labels = read_full_tracked_stack(tracked_path) if use_validated else None\n"
            "    report = build_ultrack_database(\n"
            "        contour_maps_path=contour_path,\n"
            "        foreground_masks_path=foreground_masks_path,\n"
            "        nucleus_prob_zavg_path=nucleus_prob_zavg_path,\n"
            "        working_dir=working_dir,\n"
            "        cfg=cfg,\n"
            "        validated_tracks=validated_tracks,\n"
            "        tracked_labels=tracked_labels,\n"
            "        use_validated=use_validated,\n"
            "        progress_cb=lambda msg: print(msg, flush=True),\n"
            "    )\n"
            "    print(f'Done. {report}', flush=True)\n"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix="cellflow_db_gen_", delete=False) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path)}"
        try:
            from cellflow.napari.utils import launch_in_terminal
            launch_in_terminal(cmd)
            self._set_db_gen_status("DB generation launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(cmd)
            self._set_db_gen_status("Copied DB generation command to clipboard.")

    def _on_db_gen_mode_changed(self, mode: str) -> None:
        self.db_gen_iou_weight_spin.setEnabled(mode == "iou")

    def _on_db_gen_progress(self, msg: str) -> None:
        self._set_db_gen_status(msg)

    def _on_db_gen_done(self, pos_dir: Path) -> None:
        self.db_gen_progress_bar.setVisible(False)
        self.run_db_gen_btn.setEnabled(True)
        self.db_gen_terminal_btn.setEnabled(True)
        self._set_db_gen_status("DB generation complete.")
        self._refresh_stage_files(pos_dir)
        self._refresh_ultrack_db_browser()

    def _on_db_gen_worker_error(self, exc: Exception) -> None:
        self.db_gen_progress_bar.setVisible(False)
        self.run_db_gen_btn.setEnabled(True)
        self.db_gen_terminal_btn.setEnabled(True)
        self._set_db_gen_status(f"Error: {exc}")
        logger.exception("DB generation worker error", exc_info=exc)

    def _set_db_gen_status(self, msg: str) -> None:
        self.db_gen_status_lbl.setText(msg)
        self.db_gen_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    # ── Ultrack DB Browser section ────────────────────────────────────────────

    def _set_ultrack_db_status(self, msg: str) -> None:
        self.ultrack_db_section_status_lbl.setText(msg)
        self.ultrack_db_section_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _on_ultrack_db_browser_param_changed(self, *_args) -> None:
        self._ultrack_db_preview_cache.clear()

    def _on_ultrack_db_mode_changed(self, mode: str) -> None:
        self._ultrack_db_preview_cache.clear()
        self._ultrack_db_slider_row.setVisible(mode == "Hierarchy cut")

    def _on_ultrack_db_slider_changed(self, value: int) -> None:
        if not self._ultrack_db_browser_active:
            return
        db_path = self._ultrack_db_path()
        if db_path is not None and db_path.exists():
            try:
                mtime_ns = db_path.stat().st_mtime_ns
                heights = self._query_distinct_heights(db_path, mtime_ns)
                index = min(max(int(value), 0), max(len(heights) - 1, 0))
                if heights:
                    self._set_ultrack_db_height_label(index, heights[index], len(heights))
                else:
                    self.ultrack_db_height_lbl.setText("—")
            except Exception:
                self.ultrack_db_height_lbl.setText(str(value))
        else:
            self.ultrack_db_height_lbl.setText(str(value))
        self._ultrack_db_preview_cache.clear()
        from qtpy.QtCore import QTimer
        QTimer.singleShot(150, self._refresh_ultrack_db_browser)

    def _on_ultrack_db_activate(self, checked: bool) -> None:
        self._ultrack_db_browser_active = checked
        self.ultrack_db_active_btn.setText("Deactivate" if checked else "Activate")
        self.ultrack_db_refresh_btn.setEnabled(checked)
        self.ultrack_db_mode_combo.setEnabled(checked)
        self.ultrack_db_hierarchy_slider.setEnabled(checked)
        self.ultrack_db_prob_alpha_check.setEnabled(checked)
        self.ultrack_db_connected_focus_check.setEnabled(checked)
        self.ultrack_db_edge_alpha_check.setEnabled(checked)
        self.ultrack_db_show_validated_check.setEnabled(checked)
        self.ultrack_db_show_fake_check.setEnabled(checked)
        if checked:
            self._ultrack_db_frame_initialized = False
            self._refresh_ultrack_db_browser()
        else:
            self._remove_ultrack_db_browser_layers()

    def _remove_ultrack_db_browser_layers(self) -> None:
        self._remove_ultrack_db_preview_selector()
        for name in (
            _ULTRACK_DB_PREVIEW_LAYER,
            _ULTRACK_DB_ANNOTATION_LAYER,
            _CONTOUR_MAPS_DB_LAYER,
            _FOREGROUND_MASKS_DB_LAYER,
        ):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        if _ULTRACK_DB_SELECTION_LAYER in self.viewer.layers:
            self.viewer.layers.remove(_ULTRACK_DB_SELECTION_LAYER)
        self.ultrack_db_info_lbl.setText("—")
        self._set_ultrack_db_status("")

    def _ensure_ultrack_db_browser_layers_loaded(self) -> None:
        contour_path = self._contour_maps_path()
        fg_path = self._foreground_masks_path()
        try:
            if (
                contour_path and contour_path.exists()
                and _CONTOUR_MAPS_DB_LAYER not in self.viewer.layers
            ):
                data = np.asarray(tifffile.imread(str(contour_path)), dtype=np.float32)
                self.viewer.add_image(data, name=_CONTOUR_MAPS_DB_LAYER, colormap="magma", visible=True)
            if (
                fg_path and fg_path.exists()
                and _FOREGROUND_MASKS_DB_LAYER not in self.viewer.layers
            ):
                data = np.asarray(tifffile.imread(str(fg_path)), dtype=np.float32)
                self.viewer.add_image(data, name=_FOREGROUND_MASKS_DB_LAYER, colormap="gray", visible=True)
        except Exception as e:
            logger.warning("Failed to load DB browser layers: %s", e)

    def _ultrack_db_middle_frame(self, db_path: Path) -> int | None:
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                frames = sorted(
                    int(r[0]) for r in session.query(NodeDB.t).distinct().all()
                )
        except Exception:
            return None
        finally:
            engine.dispose()
        if not frames:
            return None
        return frames[len(frames) // 2]

    def _refresh_ultrack_db_browser(self) -> None:
        if not self._ultrack_db_browser_active:
            return
        self.ultrack_db_info_lbl.setText("—")
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._set_ultrack_db_status("data.db not found — run DB generation first.")
            return
        self._ensure_ultrack_db_browser_layers_loaded()
        frame = self._current_t()
        if not self._ultrack_db_frame_initialized:
            self._ultrack_db_frame_initialized = True
            if frame == 0:
                mid = self._ultrack_db_middle_frame(db_path)
                if mid is not None and mid > 0:
                    frame = mid
                    self._set_viewer_frame(frame)
        try:
            self.ultrack_db_info_lbl.setText(self._ultrack_db_summary_text(db_path, frame))
            mode = self.ultrack_db_mode_combo.currentText()
            if mode == "Summary only":
                self._set_ultrack_db_status("Summary refreshed.")
                return

            mtime_ns = db_path.stat().st_mtime_ns
            states = self._configure_ultrack_db_hierarchy_slider(db_path, mtime_ns, frame)
            if not states:
                labels = self._empty_ultrack_db_preview()
                self._update_layer(_ULTRACK_DB_PREVIEW_LAYER, labels)
                self._set_ultrack_db_status(f"No hierarchy states for frame {frame}.")
                return

            slider_int = int(self.ultrack_db_hierarchy_slider.value())
            state = states[slider_int]
            key = (
                str(db_path.resolve()),
                mtime_ns,
                frame,
                slider_int,
                state,
                self.ultrack_db_show_validated_check.isChecked(),
                self.ultrack_db_show_fake_check.isChecked(),
            )
            cached = self._ultrack_db_preview_cache.get(key)
            if cached is None:
                cached = self._render_hierarchy_cut_state(db_path, frame, state)
                self._ultrack_db_preview_cache[key] = cached
            labels, status, prob_dict, label_to_node_id, node_id_to_label, node_annotations = (
                self._normalize_ultrack_db_preview(cached)
            )
            self._ultrack_db_label_to_node_id = label_to_node_id
            self._ultrack_db_node_id_to_label = node_id_to_label
            self._ultrack_db_node_annotations = node_annotations
            alpha_dict: dict[int, float] = {}
            if self.ultrack_db_connected_focus_check.isChecked():
                labels, status, alpha_dict = self._render_ultrack_db_connected_focus(
                    db_path,
                    frame,
                    labels,
                    status,
                    prob_dict,
                    label_to_node_id,
                    node_id_to_label,
                )
            self._ultrack_db_preview_labels = labels.astype(np.uint32, copy=False)
            self._update_ultrack_db_preview_layer(
                self._ultrack_db_preview_labels, prob_dict, alpha_dict
            )
            self._update_ultrack_db_annotation_layer(
                self._ultrack_db_preview_labels,
                label_to_node_id,
                node_annotations,
            )
            self._install_ultrack_db_preview_selector()
            if not self.ultrack_db_connected_focus_check.isChecked():
                status = self._refresh_ultrack_db_selection_highlight(
                    self._ultrack_db_preview_labels,
                    status,
                    node_id_to_label,
                    frame,
                )
            self._set_ultrack_db_status(status)
        except Exception as e:
            self._set_ultrack_db_status(f"DB read error: {e}")
            logger.warning("DB browser error: %s", e)

    @staticmethod
    def _normalize_ultrack_db_preview(
        cached: tuple[np.ndarray, str]
        | tuple[np.ndarray, str, dict[int, float]]
        | tuple[
            np.ndarray,
            str,
            dict[int, float],
            dict[int, int],
            dict[int, int],
            dict[int, str],
        ],
    ) -> tuple[np.ndarray, str, dict[int, float], dict[int, int], dict[int, int], dict[int, str]]:
        if len(cached) == 2:
            labels, status = cached
            return labels, status, {}, {}, {}, {}
        if len(cached) == 3:
            labels, status, prob_dict = cached
            return labels, status, prob_dict, {}, {}, {}
        if len(cached) == 5:
            labels, status, prob_dict, label_to_node_id, node_id_to_label = cached
            return labels, status, prob_dict, label_to_node_id, node_id_to_label, {}
        labels, status, prob_dict, label_to_node_id, node_id_to_label, node_annotations = cached
        return labels, status, prob_dict, label_to_node_id, node_id_to_label, node_annotations

    def _update_ultrack_db_preview_layer(
        self,
        labels: np.ndarray,
        prob_dict: dict[int, float],
        alpha_dict: dict[int, float] | None = None,
    ) -> None:
        if alpha_dict:
            data = self._ultrack_db_alpha_rgba(labels, alpha_dict)
            self._update_image_layer(_ULTRACK_DB_PREVIEW_LAYER, data, rgb=True)
            return
        if self.ultrack_db_prob_alpha_check.isChecked() and prob_dict:
            data = self._ultrack_db_probability_rgba(labels, prob_dict)
            self._update_image_layer(_ULTRACK_DB_PREVIEW_LAYER, data, rgb=True)
            return
        self._update_labels_layer(_ULTRACK_DB_PREVIEW_LAYER, labels)

    def _update_ultrack_db_annotation_layer(
        self,
        labels: np.ndarray,
        label_to_node_id: dict[int, int],
        node_annotations: dict[int, str],
    ) -> None:
        overlay = np.zeros_like(labels, dtype=np.uint8)
        for label_id, node_id in label_to_node_id.items():
            annot = node_annotations.get(int(node_id), "UNKNOWN")
            if annot == "REAL":
                overlay[labels == int(label_id)] = 1
            elif annot == "FAKE":
                overlay[labels == int(label_id)] = 2
        if not np.any(overlay):
            if _ULTRACK_DB_ANNOTATION_LAYER in self.viewer.layers:
                self.viewer.layers.remove(_ULTRACK_DB_ANNOTATION_LAYER)
            return
        self._update_labels_layer(_ULTRACK_DB_ANNOTATION_LAYER, overlay)

    def _update_labels_layer(self, name: str, data: np.ndarray) -> None:
        from napari.layers import Labels

        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Labels):
            self.viewer.layers[name].data = data
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        self.viewer.add_labels(data, name=name)

    def _update_image_layer(self, name: str, data: np.ndarray, *, rgb: bool = False) -> None:
        from napari.layers import Image

        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Image):
            self.viewer.layers[name].data = data
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        self.viewer.add_image(data, name=name, rgb=rgb, blending="translucent")

    @staticmethod
    def _ultrack_db_probability_rgba(
        labels: np.ndarray, prob_dict: dict[int, float]
    ) -> np.ndarray:
        from napari.utils.colormaps import label_colormap

        rgba = np.zeros(labels.shape + (4,), dtype=np.float32)
        if labels.size == 0 or not prob_dict:
            return rgba

        probs = [float(v) for v in prob_dict.values()]
        min_p = min(probs)
        max_p = max(probs)
        denom = max(max_p - min_p, 1e-9)
        cmap = label_colormap(max(prob_dict.keys()) + 1)
        for label_id, prob in prob_dict.items():
            label_mask = labels == int(label_id)
            if not np.any(label_mask):
                continue
            color = np.asarray(cmap.map(int(label_id)), dtype=np.float32)
            alpha = 0.15 + 0.85 * (float(prob) - min_p) / denom
            color[3] = float(np.clip(alpha, 0.15, 1.0))
            rgba[label_mask] = color
        return rgba

    @staticmethod
    def _ultrack_db_alpha_rgba(
        labels: np.ndarray, alpha_dict: dict[int, float]
    ) -> np.ndarray:
        from napari.utils.colormaps import label_colormap

        rgba = np.zeros(labels.shape + (4,), dtype=np.float32)
        if labels.size == 0 or not alpha_dict:
            return rgba

        cmap = label_colormap(max(alpha_dict.keys()) + 1)
        for label_id, alpha in alpha_dict.items():
            label_mask = labels == int(label_id)
            if not np.any(label_mask):
                continue
            color = np.asarray(cmap.map(int(label_id)), dtype=np.float32)
            color[3] = float(np.clip(alpha, 0.0, 1.0))
            rgba[label_mask] = color
        return rgba

    def _install_ultrack_db_preview_selector(self) -> None:
        if _ULTRACK_DB_PREVIEW_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[_ULTRACK_DB_PREVIEW_LAYER]
        self._remove_ultrack_db_preview_selector()

        def _on_drag(_layer, event):
            if getattr(event, "type", None) != "mouse_press":
                return
            if getattr(event, "button", None) != 1:
                return
            if getattr(event, "modifiers", set()):
                return
            labels = self._ultrack_db_preview_labels
            if labels is None or labels.size == 0:
                return
            pos = _layer.world_to_data(event.position)
            y = int(round(float(pos[-2])))
            x = int(round(float(pos[-1])))
            if y < 0 or x < 0 or y >= labels.shape[-2] or x >= labels.shape[-1]:
                return
            display_label = int(labels[y, x])
            if display_label == 0:
                return
            self._select_ultrack_db_preview_label(display_label, frame=self._current_t())
            yield

        layer.mouse_drag_callbacks.append(_on_drag)
        self._ultrack_db_preview_mouse_callback = _on_drag

    def _remove_ultrack_db_preview_selector(self) -> None:
        callback = self._ultrack_db_preview_mouse_callback
        if callback is None or _ULTRACK_DB_PREVIEW_LAYER not in self.viewer.layers:
            self._ultrack_db_preview_mouse_callback = None
            return
        layer = self.viewer.layers[_ULTRACK_DB_PREVIEW_LAYER]
        try:
            layer.mouse_drag_callbacks.remove(callback)
        except ValueError:
            pass
        self._ultrack_db_preview_mouse_callback = None

    def _select_ultrack_db_preview_label(
        self, display_label: int, *, frame: int | None = None
    ) -> None:
        node_id = self._ultrack_db_label_to_node_id.get(int(display_label))
        if node_id is None:
            self._set_ultrack_db_status(f"No DB node mapped to label {display_label}.")
            self._clear_ultrack_db_highlight()
            return
        selected_frame = self._current_t() if frame is None else int(frame)
        self._ultrack_db_selected_node_id = int(node_id)
        self._ultrack_db_selected_frame = selected_frame
        self._update_ultrack_db_highlight(self._ultrack_db_preview_labels, int(display_label))
        annot = self._ultrack_db_node_annotations.get(int(node_id), "UNKNOWN")
        annot_suffix = "" if annot == "UNKNOWN" else f" [{annot}]"
        self._set_ultrack_db_status(
            f"Selected node {node_id}{annot_suffix} at t={selected_frame}."
        )
        if self.ultrack_db_connected_focus_check.isChecked():
            self._refresh_ultrack_db_browser()

    def _refresh_ultrack_db_selection_highlight(
        self,
        labels: np.ndarray,
        status: str,
        node_id_to_label: dict[int, int],
        frame: int,
    ) -> str:
        selected_node_id = self._ultrack_db_selected_node_id
        if selected_node_id is None:
            self._clear_ultrack_db_highlight()
            return status
        display_label = node_id_to_label.get(int(selected_node_id))
        if display_label is None:
            self._clear_ultrack_db_highlight()
            annot = self._query_ultrack_db_node_annotation_for_status(
                node_id_to_label, selected_node_id
            )
            if annot in {"REAL", "FAKE"}:
                return (
                    f"{status} Selected node {selected_node_id} [{annot}] is hidden "
                    f"by annotation filter at frame {frame}."
                )
            return (
                f"{status} Selected node {selected_node_id} is hidden "
                f"at frame {frame} and the current hierarchy threshold."
            )
        self._update_ultrack_db_highlight(labels, int(display_label))
        return status

    def _query_ultrack_db_node_annotation_for_status(
        self, node_id_to_label: dict[int, int], selected_node_id: int
    ) -> str:
        return self._ultrack_db_node_annotations.get(int(selected_node_id), "UNKNOWN")

    def _get_ultrack_db_highlight_layer(self):
        if _ULTRACK_DB_SELECTION_LAYER in self.viewer.layers:
            return self.viewer.layers[_ULTRACK_DB_SELECTION_LAYER]
        layer = self.viewer.add_shapes(
            name=_ULTRACK_DB_SELECTION_LAYER,
            ndim=2,
            edge_color="cyan",
            edge_width=2,
            face_color="transparent",
        )
        layer.visible = False
        return layer

    def _update_ultrack_db_highlight(
        self, labels: np.ndarray | None, display_label: int
    ) -> None:
        layer = self._get_ultrack_db_highlight_layer()
        if labels is None or display_label == 0:
            layer.data = []
            layer.visible = False
            return
        mask = (labels == int(display_label)).astype(np.uint8)
        if not np.any(mask):
            layer.data = []
            layer.visible = False
            return
        from skimage.measure import find_contours

        contours = find_contours(mask, level=0.5)
        if not contours:
            layer.data = []
            layer.visible = False
            return
        layer.data = [max(contours, key=len)]
        layer.shape_type = ["polygon"]
        layer.visible = True

    def _clear_ultrack_db_highlight(self) -> None:
        if _ULTRACK_DB_SELECTION_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[_ULTRACK_DB_SELECTION_LAYER]
        layer.data = []
        layer.visible = False

    def _query_ultrack_db_connected_nodes(
        self, db_path: Path, selected_node_id: int
    ) -> tuple[dict[int, float], dict[int, float]]:
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import LinkDB

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        predecessors: dict[int, float] = {}
        successors: dict[int, float] = {}
        try:
            with Session(engine) as session:
                rows = (
                    session.query(LinkDB.source_id, LinkDB.target_id, LinkDB.weight)
                    .filter(
                        (LinkDB.source_id == int(selected_node_id))
                        | (LinkDB.target_id == int(selected_node_id))
                    )
                    .all()
                )
                for source_id, target_id, weight in rows:
                    weight_f = float(weight if weight is not None else 1.0)
                    if int(target_id) == int(selected_node_id):
                        source_i = int(source_id)
                        predecessors[source_i] = predecessors.get(source_i, 1.0) * weight_f
                    if int(source_id) == int(selected_node_id):
                        target_i = int(target_id)
                        successors[target_i] = successors.get(target_i, 1.0) * weight_f
        finally:
            engine.dispose()
        return predecessors, successors

    def _render_ultrack_db_connected_focus(
        self,
        db_path: Path,
        frame: int,
        labels: np.ndarray,
        status: str,
        prob_dict: dict[int, float],
        label_to_node_id: dict[int, int],
        node_id_to_label: dict[int, int],
    ) -> tuple[np.ndarray, str, dict[int, float]]:
        selected_node_id = self._ultrack_db_selected_node_id
        selected_frame = self._ultrack_db_selected_frame
        if selected_node_id is None or selected_frame is None:
            self._clear_ultrack_db_highlight()
            return labels, f"{status} Click a DB preview node to focus links.", {}

        predecessors, successors = self._query_ultrack_db_connected_nodes(
            db_path, selected_node_id
        )
        if frame == selected_frame:
            relation = "selected"
            allowed_weights = {selected_node_id: 1.0}
            if int(selected_node_id) not in node_id_to_label:
                self._clear_ultrack_db_highlight()
                empty = np.zeros_like(labels, dtype=np.uint32)
                annot = self._ultrack_db_node_annotations.get(int(selected_node_id), "UNKNOWN")
                annot_suffix = "" if annot == "UNKNOWN" else f" [{annot}]"
                return (
                    empty,
                    f"Selected node {selected_node_id}{annot_suffix} at t={selected_frame} is "
                    "hidden by the current threshold or annotation filter.",
                    {},
                )
        elif frame == selected_frame - 1:
            relation = "t-1"
            allowed_weights = predecessors
        elif frame == selected_frame + 1:
            relation = "t+1"
            allowed_weights = successors
        else:
            empty = np.zeros_like(labels, dtype=np.uint32)
            self._clear_ultrack_db_highlight()
            return (
                empty,
                f"Selected node {selected_node_id} at t={selected_frame} | "
                f"frame {frame}: outside connected focus.",
                {},
            )

        focused = np.zeros_like(labels, dtype=np.uint32)
        alpha_dict: dict[int, float] = {}
        for label_id, node_id in label_to_node_id.items():
            label_i = int(label_id)
            node_i = int(node_id)
            if node_i not in allowed_weights:
                continue
            focused[labels == label_i] = label_i
            alpha_enabled = (
                self.ultrack_db_edge_alpha_check.isChecked()
                or self.ultrack_db_prob_alpha_check.isChecked()
            )
            if alpha_enabled:
                if node_i == selected_node_id:
                    alpha_dict[label_i] = 1.0
                else:
                    alpha_dict[label_i] = self._ultrack_db_connected_alpha(
                        label_i,
                        float(allowed_weights[node_i]),
                        prob_dict,
                    )

        selected_label = node_id_to_label.get(int(selected_node_id))
        if frame == selected_frame and selected_label is not None:
            self._update_ultrack_db_highlight(focused, int(selected_label))
        else:
            self._clear_ultrack_db_highlight()

        edge_values = [
            float(v)
            for node_id, v in allowed_weights.items()
            if node_id in node_id_to_label and node_id != selected_node_id
        ]
        if edge_values:
            edge_summary = (
                f" | edge product range {min(edge_values):.2f}-{max(edge_values):.2f}"
            )
        else:
            edge_summary = ""
        count = int(np.unique(focused[focused != 0]).size)
        annot = self._ultrack_db_node_annotations.get(int(selected_node_id), "UNKNOWN")
        annot_suffix = "" if annot == "UNKNOWN" else f" [{annot}]"
        return (
            focused,
            f"Selected node {selected_node_id}{annot_suffix} at t={selected_frame} | "
            f"{relation}: {count} connected node(s){edge_summary}",
            alpha_dict,
        )

    def _ultrack_db_connected_alpha(
        self,
        label_id: int,
        edge_weight: float,
        prob_dict: dict[int, float],
    ) -> float:
        alpha = 1.0
        if self.ultrack_db_edge_alpha_check.isChecked():
            alpha *= float(edge_weight)
        if self.ultrack_db_prob_alpha_check.isChecked() and prob_dict:
            probs = [float(v) for v in prob_dict.values()]
            min_p = min(probs)
            max_p = max(probs)
            denom = max(max_p - min_p, 1e-9)
            prob = float(prob_dict.get(int(label_id), 1.0))
            alpha *= 0.15 + 0.85 * (prob - min_p) / denom
        return float(np.clip(alpha, 0.05, 1.0))

    def _ultrack_db_summary_text(self, db_path: Path, frame: int) -> str:
        import sqlalchemy as sqla
        from sqlalchemy import func
        from sqlalchemy.orm import Session
        from ultrack.core.database import LinkDB, NodeDB, VarAnnotation

        try:
            from ultrack.core.database import OverlapDB
        except Exception:
            OverlapDB = None

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                n_nodes = int(session.query(func.count(NodeDB.id)).scalar() or 0)
                n_links = int(session.query(func.count(LinkDB.source_id)).scalar() or 0)
                n_real = int(
                    session.query(func.count(NodeDB.id))
                    .filter(NodeDB.node_annot == VarAnnotation.REAL)
                    .scalar() or 0
                )
                n_fake = int(
                    session.query(func.count(NodeDB.id))
                    .filter(NodeDB.node_annot == VarAnnotation.FAKE)
                    .scalar() or 0
                )
                frame_nodes = session.query(NodeDB).filter(NodeDB.t == frame).all()
                selected = sum(1 for n in frame_nodes if getattr(n, "selected", False))
                node_ids = [int(n.id) for n in frame_nodes]
                outgoing = incoming = overlaps = 0
                if node_ids:
                    outgoing = int(
                        session.query(func.count(LinkDB.source_id))
                        .filter(LinkDB.source_id.in_(node_ids))
                        .scalar() or 0
                    )
                    incoming = int(
                        session.query(func.count(LinkDB.target_id))
                        .filter(LinkDB.target_id.in_(node_ids))
                        .scalar() or 0
                    )
                    if OverlapDB is not None:
                        try:
                            overlaps = int(
                                session.query(func.count(OverlapDB.node_id))
                                .filter(
                                    OverlapDB.node_id.in_(node_ids)
                                    | OverlapDB.ancestor_id.in_(node_ids)
                                )
                                .scalar() or 0
                            )
                        except Exception:
                            overlaps = 0
            return (
                f"{n_nodes} nodes | {n_links} links | REAL {n_real} | FAKE {n_fake} | frame {frame}: "
                f"{len(node_ids)} nodes, {selected} selected, "
                f"{incoming} in/{outgoing} out links, {overlaps} overlaps"
            )
        finally:
            engine.dispose()

    def _query_distinct_heights(self, db_path: Path, mtime_ns: int) -> tuple[float, ...]:
        key = (str(db_path.resolve()), mtime_ns)
        cached = self._ultrack_db_height_values_cache.get(key)
        if cached is not None:
            return cached
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                heights = tuple(
                    float(row[0])
                    for row in session.query(NodeDB.height)
                    .distinct()
                    .order_by(NodeDB.height)
                    .all()
                    if row[0] is not None
                )
        finally:
            engine.dispose()
        self._ultrack_db_height_values_cache[key] = heights
        return heights

    def _query_hierarchy_cut_states(
        self, db_path: Path, mtime_ns: int, frame: int
    ) -> tuple[_HierarchyCutState, ...]:
        key = (str(db_path.resolve()), mtime_ns, frame)
        cached = self._ultrack_db_cut_state_cache.get(key)
        if cached is not None:
            return cached

        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        from ultrack.utils.constants import NO_PARENT

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                rows = [
                    (int(node_id), int(parent_id), float(height))
                    for node_id, parent_id, height in session.query(
                        NodeDB.id, NodeDB.hier_parent_id, NodeDB.height
                    )
                    .filter(NodeDB.t == frame)
                    .order_by(NodeDB.height, NodeDB.id)
                    .all()
                    if height is not None
                ]
        except Exception:
            heights = self._query_distinct_heights(db_path, mtime_ns)
            return tuple(_HierarchyCutState((), float(height)) for height in heights)
        finally:
            engine.dispose()

        if not rows:
            self._ultrack_db_cut_state_cache[key] = ()
            return ()

        node_ids = {node_id for node_id, _parent_id, _height in rows}
        heights_by_id = {node_id: height for node_id, _parent_id, height in rows}
        parent_by_id = {
            node_id: parent_id
            for node_id, parent_id, _height in rows
            if parent_id != NO_PARENT and parent_id in node_ids
        }
        children_by_parent: dict[int, set[int]] = {}
        for child_id, parent_id in parent_by_id.items():
            children_by_parent.setdefault(parent_id, set()).add(child_id)

        active = {
            node_id for node_id, _parent_id, _height in rows
            if node_id not in children_by_parent
        }
        if not active:
            active = set(node_ids)

        states: list[_HierarchyCutState] = []
        seen_states: set[tuple[int, ...]] = set()

        def _append_state() -> None:
            ordered = tuple(
                sorted(active, key=lambda node_id: (heights_by_id[node_id], node_id))
            )
            if ordered in seen_states:
                return
            seen_states.add(ordered)
            height = max((heights_by_id[node_id] for node_id in ordered), default=None)
            states.append(_HierarchyCutState(ordered, height))

        _append_state()
        while True:
            promotable = [
                parent_id
                for parent_id, child_ids in children_by_parent.items()
                if parent_id not in active and child_ids and child_ids.issubset(active)
            ]
            if not promotable:
                break
            min_height = min(heights_by_id[parent_id] for parent_id in promotable)
            promote_now = [
                parent_id
                for parent_id in promotable
                if heights_by_id[parent_id] == min_height
            ]
            for parent_id in sorted(promote_now):
                active.difference_update(children_by_parent[parent_id])
                active.add(parent_id)
            _append_state()

        result = tuple(states)
        self._ultrack_db_cut_state_cache[key] = result
        return result

    def _configure_ultrack_db_hierarchy_slider(
        self, db_path: Path, mtime_ns: int, frame: int
    ) -> tuple[_HierarchyCutState, ...]:
        states = self._query_hierarchy_cut_states(db_path, mtime_ns, frame)
        maximum = max(len(states) - 1, 0)
        value = min(max(int(self.ultrack_db_hierarchy_slider.value()), 0), maximum)

        old_blocked = self.ultrack_db_hierarchy_slider.blockSignals(True)
        try:
            self.ultrack_db_hierarchy_slider.setRange(0, maximum)
            self.ultrack_db_hierarchy_slider.setValue(value)
        finally:
            self.ultrack_db_hierarchy_slider.blockSignals(old_blocked)

        if states:
            self._set_ultrack_db_height_label(value, states[value].height, len(states))
        else:
            self.ultrack_db_height_lbl.setText("—")
        return states

    def _set_ultrack_db_height_label(
        self, index: int, height: float | None, total: int
    ) -> None:
        height_text = "—" if height is None else f"{height:.2f}"
        self.ultrack_db_height_lbl.setText(
            f"i={index} h={height_text} ({index + 1}/{total})"
        )

    def _render_hierarchy_cut(
        self, db_path: Path, frame: int, h_actual: float
    ) -> tuple[
        np.ndarray,
        str,
        dict[int, float],
        dict[int, int],
        dict[int, int],
        dict[int, str],
    ]:
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session, aliased
        from ultrack.core.database import NodeDB
        from ultrack.utils.constants import NO_PARENT

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                P = aliased(NodeDB)
                C = aliased(NodeDB)
                same_height_child_exists = (
                    session.query(C.id)
                    .where(C.hier_parent_id == NodeDB.id)
                    .where(C.height == NodeDB.height)
                    .where(NodeDB.height == h_actual)
                    .exists()
                )
                nodes = (
                    session.query(NodeDB)
                    .outerjoin(P, NodeDB.hier_parent_id == P.id)
                    .where(NodeDB.t == frame)
                    .where(NodeDB.height <= h_actual)
                    .where(
                        (NodeDB.hier_parent_id == NO_PARENT)
                        | ((NodeDB.height < h_actual) & (P.height > h_actual))
                        | ((NodeDB.height == h_actual) & (P.height >= h_actual))
                    )
                    .where(~same_height_child_exists)
                    .all()
                )
        finally:
            engine.dispose()

        return self._finalize_hierarchy_nodes(
            nodes,
            frame,
            empty_msg=f"No segments at this threshold for frame {frame}.",
            status_suffix=f"at h={h_actual:.2f}",
        )

    def _render_hierarchy_cut_state(
        self, db_path: Path, frame: int, state: _HierarchyCutState
    ) -> tuple[
        np.ndarray,
        str,
        dict[int, float],
        dict[int, int],
        dict[int, int],
        dict[int, str],
    ]:
        if not state.node_ids:
            return self._render_hierarchy_cut(db_path, frame, float(state.height or 0.0))

        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                rows = (
                    session.query(NodeDB)
                    .where(NodeDB.t == frame)
                    .where(NodeDB.id.in_(state.node_ids))
                    .all()
                )
        finally:
            engine.dispose()

        nodes_by_id = {int(node.id): node for node in rows}
        nodes = [
            nodes_by_id[node_id]
            for node_id in state.node_ids
            if node_id in nodes_by_id
        ]
        height_text = "—" if state.height is None else f"{state.height:.2f}"
        return self._finalize_hierarchy_nodes(
            nodes,
            frame,
            empty_msg=f"No hierarchy state segments for frame {frame}.",
            status_suffix=f"at cut state h={height_text}",
        )

    def _finalize_hierarchy_nodes(
        self,
        nodes: list,
        frame: int,
        *,
        empty_msg: str,
        status_suffix: str,
    ) -> tuple[
        np.ndarray,
        str,
        dict[int, float],
        dict[int, int],
        dict[int, int],
        dict[int, str],
    ]:
        if not nodes:
            return self._empty_ultrack_db_preview(), empty_msg, {}, {}, {}, {}
        show_validated = self.ultrack_db_show_validated_check.isChecked()
        show_fake = self.ultrack_db_show_fake_check.isChecked()
        filtered_nodes = []
        hidden_real = hidden_fake = 0
        for node in nodes:
            annot = self._ultrack_db_annotation_name(getattr(node, "node_annot", None))
            if annot == "REAL" and not show_validated:
                hidden_real += 1
                continue
            if annot == "FAKE" and not show_fake:
                hidden_fake += 1
                continue
            filtered_nodes.append(node)
        if not filtered_nodes:
            return self._empty_ultrack_db_preview(), (
                f"Frame {frame}: annotation filters hid all {len(nodes)} segment(s)."
            ), {}, {}, {}, {}
        labels = self._paint_ultrack_db_nodes(filtered_nodes)
        prob_dict, label_to_node_id, node_id_to_label = (
            self._ultrack_db_node_preview_metadata(filtered_nodes)
        )
        node_annotations = self._ultrack_db_node_annotation_metadata(filtered_nodes)
        hidden_summary = ""
        if hidden_real or hidden_fake:
            hidden_summary = f" Hidden by annotation filter: REAL {hidden_real}, FAKE {hidden_fake}."
        return labels, (
            f"Frame {frame}: {len(filtered_nodes)} segment(s) {status_suffix}."
            f"{hidden_summary}"
        ), prob_dict, label_to_node_id, node_id_to_label, node_annotations

    @staticmethod
    def _ultrack_db_annotation_name(value) -> str:
        if value is None:
            return "UNKNOWN"
        raw = getattr(value, "value", value)
        if raw is None:
            return "UNKNOWN"
        name = str(raw).split(".")[-1].upper()
        if name in {"REAL", "FAKE"}:
            return name
        return "UNKNOWN"

    @staticmethod
    def _ultrack_db_node_preview_metadata(
        nodes: list,
    ) -> tuple[dict[int, float], dict[int, int], dict[int, int]]:
        prob_dict: dict[int, float] = {}
        label_to_node_id: dict[int, int] = {}
        node_id_to_label: dict[int, int] = {}
        for label, node in enumerate(nodes, start=1):
            try:
                prob = float(node.node_prob if node.node_prob is not None else 1.0)
            except (TypeError, ValueError):
                prob = 1.0
            prob_dict[label] = prob
            try:
                node_id = int(node.id)
            except (TypeError, ValueError):
                continue
            label_to_node_id[label] = node_id
            node_id_to_label[node_id] = label
        return prob_dict, label_to_node_id, node_id_to_label

    @staticmethod
    def _ultrack_db_node_annotation_metadata(nodes: list) -> dict[int, str]:
        node_annotations: dict[int, str] = {}
        for node in nodes:
            try:
                node_id = int(node.id)
            except (TypeError, ValueError):
                continue
            node_annotations[node_id] = NucleusWorkflowWidget._ultrack_db_annotation_name(
                getattr(node, "node_annot", None)
            )
        return node_annotations

    def _empty_ultrack_db_preview(self) -> np.ndarray:
        shape = self._viewer_plane_shape()
        return np.zeros(shape, dtype=np.uint32)

    def _viewer_plane_shape(self) -> tuple[int, int]:
        for layer in self.viewer.layers:
            data = getattr(layer, "data", None)
            if isinstance(data, np.ndarray) and data.ndim >= 2:
                return tuple(int(v) for v in data.shape[-2:])
        return (1, 1)

    def _paint_ultrack_db_nodes(self, nodes: list) -> np.ndarray:
        masks: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
        max_y = max_x = 0
        for label, node in enumerate(nodes, start=1):
            parsed = self._node_mask_and_bbox(node)
            if parsed is None:
                continue
            bbox, mask = parsed
            y0, x0, y1, x1 = bbox
            max_y = max(max_y, y1)
            max_x = max(max_x, x1)
            masks.append((label, bbox, mask))

        base_y, base_x = self._viewer_plane_shape()
        labels = np.zeros((max(base_y, max_y, 1), max(base_x, max_x, 1)), dtype=np.uint32)
        for label, (y0, x0, y1, x1), mask in masks:
            target = labels[y0:y1, x0:x1]
            if target.shape != mask.shape:
                continue
            target[mask.astype(bool)] = label
        return labels

    @staticmethod
    def _node_mask_and_bbox(node) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        try:
            # MaybePickleType already unpickles on read; only call pickle.loads if raw bytes
            node_obj = node.pickle
            if isinstance(node_obj, (bytes, memoryview)):
                node_obj = pickle.loads(bytes(node_obj))
            if node_obj is None:
                return None
        except Exception:
            return None

        if isinstance(node_obj, dict):
            bbox = node_obj.get("bbox")
            mask = node_obj.get("mask")
        elif isinstance(node_obj, tuple) and len(node_obj) >= 2:
            bbox, mask = node_obj[0], node_obj[1]
        else:
            bbox = getattr(node_obj, "bbox", None)
            mask = getattr(node_obj, "mask", None)

        if bbox is None or mask is None:
            return None
        bbox_arr = np.asarray(bbox, dtype=int).ravel()
        if bbox_arr.size >= 6:
            y0, x0, y1, x1 = int(bbox_arr[1]), int(bbox_arr[2]), int(bbox_arr[4]), int(bbox_arr[5])
        elif bbox_arr.size >= 4:
            y0, x0, y1, x1 = (int(v) for v in bbox_arr[:4])
        else:
            return None

        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
            mask_arr = mask_arr[0]
        elif mask_arr.ndim > 2:
            mask_arr = np.squeeze(mask_arr)
        if mask_arr.ndim != 2:
            return None
        if mask_arr.shape != (y1 - y0, x1 - x0):
            return None
        return (y0, x0, y1, x1), mask_arr.astype(bool, copy=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _current_t(self) -> int:
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) >= 1 else 0

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
        self._update_labels_layer(name, data)

    def _set_viewer_frame(self, t: int) -> None:
        step = list(self.viewer.dims.current_step)
        if not step:
            return
        step[0] = int(t)
        self.viewer.dims.current_step = tuple(step)

    @staticmethod
    def _sigmoid_zavg(stack: np.ndarray) -> np.ndarray:
        zavg_logits = np.asarray(stack, dtype=np.float32).mean(axis=1)
        return (1.0 / (1.0 + np.exp(-zavg_logits))).astype(np.float32)

    def _set_contour_status(self, msg: str) -> None:
        self.contour_status_lbl.setText(msg)
        self.contour_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_correction_status(self, msg: str) -> None:
        self.correction_status_lbl.setText(msg)
        self.correction_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_ultrack_status(self, msg: str) -> None:
        self.ultrack_status_lbl.setText(msg)
        self.ultrack_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _on_contour_worker_error(self, exc: Exception) -> None:
        self.build_progress_bar.setVisible(False)
        self._set_contour_status(f"Error: {exc}")
        logger.exception("Contour worker error", exc_info=exc)

    def _on_correction_worker_error(self, exc: Exception) -> None:
        self._set_correction_status(f"Error: {exc}")
        logger.exception("Correction worker error", exc_info=exc)

    def _on_ultrack_worker_error(self, exc: Exception) -> None:
        self.ultrack_progress_bar.setVisible(False)
        self.ultrack_progress_bar.setRange(0, 100)
        self.run_ultrack_btn.setEnabled(True)
        self.ultrack_terminal_btn.setEnabled(True)
        self._set_ultrack_status(f"Error: {exc}")
        logger.exception("Ultrack worker error", exc_info=exc)

    def _cp_gammas(self) -> list[float]:
        """Gamma values to iterate during consensus boundary building."""
        gmin  = self.cp_gamma_min_spin.value()
        gmax  = self.cp_gamma_max_spin.value()
        gstep = self.cp_gamma_step_spin.value()
        return list(np.arange(gmin, gmax + gstep / 2, gstep))


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
        flow_threshold: float = 0.0,
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
            b, fg = build_consensus_boundary(
                prob_3d,
                dp_3d,
                thresholds,
                gamma=g,
                flow_threshold=flow_threshold,
                mask_callback=cb,
            )
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
            self._set_contour_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path   = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return

        thresholds      = list(np.arange(self.cp_min_spin.value(), self.cp_max_spin.value() + self.cp_step_spin.value() / 2, self.cp_step_spin.value()))
        gammas          = self._cp_gammas()
        contour_path    = self._contour_maps_path()
        score_path      = self._foreground_scores_path()
        mask_path       = self._foreground_masks_path()
        foreground_threshold = self.contour_fg_threshold_spin.value()
        flow_threshold = self.contour_flow_threshold_spin.value()
        save_source     = self.save_source_check.isChecked()
        pos_dir         = self._pos_dir
        build_fn        = self._build_consensus_boundary_averaged
        if contour_path is None or score_path is None or mask_path is None:
            self._set_contour_status("No project open.")
            return

        @thread_worker(connect={
            "yielded":   self._on_build_progress,
            "returned":  self._on_build_done,
            "errored":   self._on_contour_worker_error,
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
            foreground_score_frames: list[np.ndarray] = []
            foreground_mask_frames: list[np.ndarray] = []
            source_dir = pos_dir / "2_nucleus/source_labels"

            for t in range(n_t):
                yield (t + 1, n_t, f"Building contour maps and foreground masks: frame {t + 1}/{n_t}…")
                mask_cb = None
                if save_source:
                    source_dir.mkdir(parents=True, exist_ok=True)
                    def mask_cb(masks, g_idx, thresh_idx, *, _t=t):
                        tifffile.imwrite(
                            source_dir / f"masks_t{_t:04d}_g{g_idx:02d}_thr{thresh_idx:02d}.tif",
                            masks, compression="zlib",
                        )
                boundary, foreground_score = build_fn(
                    prob_stack[t],
                    dp_stack[t],
                    thresholds,
                    gammas,
                    flow_threshold=flow_threshold,
                    mask_callback=mask_cb,
                )
                contour_frames.append(boundary.astype(np.float32, copy=False))
                foreground_score = foreground_score.astype(np.float32, copy=False)
                foreground_score_frames.append(foreground_score)
                foreground_mask_frames.append(
                    (foreground_score >= foreground_threshold).astype(np.uint8)
                )

            contour_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(contour_path), np.stack(contour_frames), compression="zlib")
            tifffile.imwrite(str(score_path), np.stack(foreground_score_frames), compression="zlib")
            tifffile.imwrite(str(mask_path), np.stack(foreground_mask_frames), compression="zlib")
            return pos_dir

        gamma_desc = f"γ={gammas[0]:.2f}" if len(gammas) == 1 else f"γ={gammas[0]:.2f}–{gammas[-1]:.2f} ({len(gammas)} steps)"
        self._set_contour_status(f"Building contour maps and foreground masks ({len(thresholds)} cellprob thresholds, {gamma_desc})…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _on_build_done(self, pos_dir: Path) -> None:
        self._build_worker = None
        self._set_build_buttons_running(False)
        self._refresh_stage_files(pos_dir)
        self._set_contour_status("Contour maps and foreground masks built.")

    def _on_cancel_build(self) -> None:
        if self._build_worker is not None:
            self._build_worker.quit()
        self._build_worker = None
        self._set_build_buttons_running(False)
        self._set_contour_status("Build cancelled.")

    def _set_build_buttons_running(self, running: bool) -> None:
        self.build_btn.setEnabled(not running)
        self.preview_contour_btn.setEnabled(not running)
        self.contour_terminal_btn.setEnabled(not running)
        self.preview_contour_filter_btn.setEnabled(not running)
        self.run_contour_filter_btn.setEnabled(not running)
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
            self._set_contour_status(msg)
        else:
            self._set_contour_status(str(data))

    def _on_preview_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path   = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return

        t_frame    = self._current_t()
        thresholds = list(np.arange(self.cp_min_spin.value(), self.cp_max_spin.value() + self.cp_step_spin.value() / 2, self.cp_step_spin.value()))
        gammas     = self._cp_gammas()
        flow_threshold = self.contour_flow_threshold_spin.value()
        build_fn   = self._build_consensus_boundary_averaged

        def _on_preview_done(result):
            self._build_worker = None
            self._set_build_buttons_running(False)
            boundary, foreground, cellprob_zavg, t_idx = result
            data = np.zeros((cellprob_zavg.shape[0],) + boundary.shape, dtype=boundary.dtype)
            data[t_idx] = boundary
            foreground_score_data = np.zeros(
                (cellprob_zavg.shape[0],) + foreground.shape, dtype=np.float32
            )
            foreground_score_data[t_idx] = foreground
            foreground_mask_data = (
                foreground_score_data >= self.contour_fg_threshold_spin.value()
            ).astype(np.uint8)
            if _CELLPROB_LAYER in self.viewer.layers:
                self.viewer.layers[_CELLPROB_LAYER].data = cellprob_zavg
            else:
                self.viewer.add_image(
                    cellprob_zavg,
                    name=_CELLPROB_LAYER,
                    colormap="inferno",
                    blending="additive",
                    visible=True,
                )
            if _CONTOUR_LAYER in self.viewer.layers:
                self.viewer.layers[_CONTOUR_LAYER].data = data
            else:
                self.viewer.add_image(data, name=_CONTOUR_LAYER, colormap="magma", visible=True)
            if _FOREGROUND_SCORE_LAYER in self.viewer.layers:
                self.viewer.layers[_FOREGROUND_SCORE_LAYER].data = foreground_score_data
            else:
                self.viewer.add_image(
                    foreground_score_data,
                    name=_FOREGROUND_SCORE_LAYER,
                    colormap="viridis",
                    visible=True,
                )
            self._update_layer(_FOREGROUND_MASK_LAYER, foreground_mask_data)
            self._set_viewer_frame(t_idx)
            self._set_contour_status(
                f"Preview contour map and foreground mask t={t_idx} — "
                f"{len(thresholds)} cellprob thresholds, "
                f"{len(gammas)} gamma value(s)"
            )

        @thread_worker(connect={
            "returned": _on_preview_done,
            "errored":  self._on_contour_worker_error,
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            dp_stack   = np.asarray(tifffile.imread(str(dp_path)),   dtype=np.float32)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis]
            if dp_stack.ndim == 4:
                dp_stack = dp_stack[np.newaxis]
            n_t = min(prob_stack.shape[0], dp_stack.shape[0])
            t_idx = min(max(t_frame, 0), n_t - 1)
            boundary, foreground = build_fn(
                prob_stack[t_idx],
                dp_stack[t_idx],
                thresholds,
                gammas,
                flow_threshold=flow_threshold,
            )
            return boundary, foreground, self._sigmoid_zavg(prob_stack), t_idx

        self._set_contour_status(f"Previewing contour map for frame t={t_frame}…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _contour_filter_params_from_ui(self):
        from cellflow.segmentation import ContourFilterParams

        return ContourFilterParams(
            median_kernel_time=int(self.contour_filter_median_time_spin.value()),
            median_kernel_space=int(self.contour_filter_median_space_spin.value()),
            gaussian_sigma_time=float(self.contour_filter_gauss_time_spin.value()),
            gaussian_sigma_space=float(self.contour_filter_gauss_space_spin.value()),
        )

    def _update_contour_image_layer(self, data: np.ndarray) -> None:
        if _CONTOUR_LAYER in self.viewer.layers:
            self.viewer.layers[_CONTOUR_LAYER].data = data
        else:
            self.viewer.add_image(
                data,
                name=_CONTOUR_LAYER,
                colormap="magma",
                visible=True,
            )

    def _on_preview_contour_filter(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        contour_path = self._contour_maps_path()
        if contour_path is None or not contour_path.exists():
            self._set_contour_status(
                "Missing: contour_maps.tif — run Contour Maps first."
            )
            return

        params = self._contour_filter_params_from_ui()

        def _on_preview_done(result):
            self._build_worker = None
            self._set_build_buttons_running(False)
            filtered = result
            self._update_contour_image_layer(filtered)
            self._set_contour_status("Previewed filtered contour maps.")

        @thread_worker(connect={
            "returned": _on_preview_done,
            "errored":  self._on_contour_worker_error,
        })
        def _worker():
            from cellflow.segmentation import compute_filtered_contour_maps

            contours = np.asarray(tifffile.imread(str(contour_path)), dtype=np.float32)
            return compute_filtered_contour_maps(contours, params)

        self._set_contour_status("Previewing filtered contour maps…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _on_run_contour_filter(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        contour_path = self._contour_maps_path()
        if contour_path is None or not contour_path.exists():
            self._set_contour_status(
                "Missing: contour_maps.tif — run Contour Maps first."
            )
            return

        params = self._contour_filter_params_from_ui()
        pos_dir = self._pos_dir

        def _on_filter_done(result):
            self._build_worker = None
            self._set_build_buttons_running(False)
            pos_dir, filtered = result
            self._refresh_stage_files(pos_dir)
            self._update_contour_image_layer(filtered)
            self._set_contour_status("Filtered contour maps written to contour_maps.tif.")

        @thread_worker(connect={
            "returned": _on_filter_done,
            "errored":  self._on_contour_worker_error,
        })
        def _worker():
            from cellflow.segmentation import compute_filtered_contour_maps

            contours = np.asarray(tifffile.imread(str(contour_path)), dtype=np.float32)
            filtered = compute_filtered_contour_maps(contours, params)
            tifffile.imwrite(
                str(contour_path),
                filtered.astype(np.float32, copy=False),
                compression="zlib",
                photometric="minisblack",
            )
            return pos_dir, filtered

        self._set_contour_status("Filtering contour_maps.tif…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _on_run_contour_terminal(self) -> None:
        import sys
        import tempfile

        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path = self._dp_path()
        contour_path = self._contour_maps_path()
        score_path = self._foreground_scores_path()
        mask_path = self._foreground_masks_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return
        if contour_path is None or score_path is None or mask_path is None:
            self._set_contour_status("No project open.")
            return

        thresholds = list(
            np.arange(
                self.cp_min_spin.value(),
                self.cp_max_spin.value() + self.cp_step_spin.value() / 2,
                self.cp_step_spin.value(),
            )
        )
        gammas = self._cp_gammas()
        foreground_threshold = self.contour_fg_threshold_spin.value()
        flow_threshold = self.contour_flow_threshold_spin.value()
        save_source = self.save_source_check.isChecked()
        pos_dir = self._pos_dir

        python_code = (
            "import pathlib\n"
            "import numpy as np\n"
            "import tifffile\n"
            "from cellflow.segmentation import build_consensus_boundary\n"
            f"prob_path = pathlib.Path({str(prob_path)!r})\n"
            f"dp_path = pathlib.Path({str(dp_path)!r})\n"
            f"contour_path = pathlib.Path({str(contour_path)!r})\n"
            f"score_path = pathlib.Path({str(score_path)!r})\n"
            f"mask_path = pathlib.Path({str(mask_path)!r})\n"
            f"save_source = {save_source!r}\n"
            f"source_dir = pathlib.Path({str(pos_dir / '2_nucleus/source_labels')!r})\n"
            f"thresholds = {thresholds!r}\n"
            f"gammas = {gammas!r}\n"
            f"foreground_threshold = {foreground_threshold!r}\n"
            f"flow_threshold = {flow_threshold!r}\n"
            "def build_consensus_boundary_averaged(prob_3d, dp_3d, thresholds, gammas, flow_threshold=0.0, mask_callback=None):\n"
            "    boundary_sum = None\n"
            "    foreground_sum = None\n"
            "    for g_idx, g in enumerate(gammas):\n"
            "        cb = None\n"
            "        if mask_callback is not None:\n"
            "            def cb(masks, i_thresh, *, _gi=g_idx):\n"
            "                mask_callback(masks, _gi, i_thresh)\n"
            "        boundary, foreground = build_consensus_boundary(\n"
            "            prob_3d,\n"
            "            dp_3d,\n"
            "            thresholds,\n"
            "            gamma=g,\n"
            "            flow_threshold=flow_threshold,\n"
            "            mask_callback=cb,\n"
            "        )\n"
            "        if boundary_sum is None:\n"
            "            boundary_sum = boundary.copy()\n"
            "            foreground_sum = foreground.copy()\n"
            "        else:\n"
            "            boundary_sum += boundary\n"
            "            foreground_sum += foreground\n"
            "    n = len(gammas)\n"
            "    return boundary_sum / n, foreground_sum / n\n"
            "prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)\n"
            "dp_stack = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)\n"
            "if prob_stack.ndim == 3:\n"
            "    prob_stack = prob_stack[np.newaxis]\n"
            "if dp_stack.ndim == 4:\n"
            "    dp_stack = dp_stack[np.newaxis]\n"
            "n_t = prob_stack.shape[0]\n"
            "contour_frames = []\n"
            "foreground_score_frames = []\n"
            "foreground_mask_frames = []\n"
            "for t in range(n_t):\n"
            "    print(f'Building contour maps and foreground masks: frame {t + 1}/{n_t}...', flush=True)\n"
            "    mask_cb = None\n"
            "    if save_source:\n"
            "        source_dir.mkdir(parents=True, exist_ok=True)\n"
            "        def mask_cb(masks, g_idx, thresh_idx, *, _t=t):\n"
            "            tifffile.imwrite(\n"
            "                source_dir / f'masks_t{_t:04d}_g{g_idx:02d}_thr{thresh_idx:02d}.tif',\n"
            "                masks,\n"
            "                compression='zlib',\n"
            "            )\n"
            "    boundary, foreground = build_consensus_boundary_averaged(\n"
            "        prob_stack[t],\n"
            "        dp_stack[t],\n"
            "        thresholds,\n"
            "        gammas,\n"
            "        flow_threshold=flow_threshold,\n"
            "        mask_callback=mask_cb,\n"
            "    )\n"
            "    contour_frames.append(boundary.astype(np.float32, copy=False))\n"
            "    foreground = foreground.astype(np.float32, copy=False)\n"
            "    foreground_score_frames.append(foreground)\n"
            "    foreground_mask_frames.append((foreground >= foreground_threshold).astype(np.uint8))\n"
            "contour_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "print('Writing contour maps and foreground masks...', flush=True)\n"
            "tifffile.imwrite(str(contour_path), np.stack(contour_frames), compression='zlib')\n"
            "tifffile.imwrite(str(score_path), np.stack(foreground_score_frames), compression='zlib')\n"
            "tifffile.imwrite(str(mask_path), np.stack(foreground_mask_frames), compression='zlib')\n"
            "print('Done.')\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="cellflow_contour_build_", delete=False
        ) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path)}"
        try:
            from cellflow.napari.utils import launch_in_terminal
            launch_in_terminal(cmd)
            self._set_contour_status("Contour build command launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(cmd)
            self._set_contour_status(
                "Copied contour build command to clipboard (terminal launch unavailable)."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Automated search / propagation
    # ──────────────────────────────────────────────────────────────────────────

    def _on_save_tracked(self) -> None:
        tracked_path = self._tracked_path()
        if tracked_path is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer to save.")
            return
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3:
            self._set_correction_status("Tracked layer is not a 3D stack.")
            return
        n = layer.data.shape[0]
        for t in range(n):
            write_tracked_frame(tracked_path, t, np.asarray(layer.data[t]))
        self._set_correction_status(f"Saved {n} frame(s) to {tracked_path.name}.")

    def _on_load_tracked(self) -> None:
        tracked_path   = self._tracked_path()
        cell_zavg_path = self._cell_zavg_path()
        nuc_zavg_path  = self._nucleus_zavg_path()
        if tracked_path is None or not tracked_path.exists():
            self._set_correction_status("No tracked labels file found.")
            return
        self._set_correction_status("Loading tracked labels…")

        @thread_worker(connect={"returned": self._on_load_tracked_done, "errored": self._on_correction_worker_error})
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

        self._set_correction_status(f"Loaded tracked stack {stack.shape} into napari.")
        layer = self.viewer.layers[_TRACKED_LAYER]
        self.correction_widget.activate_layer(layer)
        self.correction_section.expand()

    def _on_reassign_ids(self) -> None:
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer loaded.")
            return
        stack = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)
        self._set_correction_status("Reassigning cell IDs to contiguous range…")

        @thread_worker(connect={"returned": self._on_reassign_ids_done, "errored": self._on_correction_worker_error})
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
        self._set_correction_status(f"Reassigned {n_cells} cell IDs to contiguous range 1–{n_cells}. Unsaved.")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Tracking & Correction
    # ──────────────────────────────────────────────────────────────────────────

    def _on_ultrack_mode_changed(self, mode: str) -> None:
        self.ultrack_iou_weight_spin.setEnabled(mode == "iou")

    def _set_resolve_prior_controls_enabled(self, enabled: bool) -> None:
        for control in (
            self.db_gen_quality_weight_spin,
            self.db_gen_quality_exp_spin,
            self.db_gen_circularity_weight_spin,
            self.ultrack_quality_exp_spin,
            self.ultrack_seed_weight_spin,
            self.ultrack_seed_space_spin,
            self.ultrack_seed_time_spin,
            self.ultrack_seed_window_spin,
        ):
            control.setEnabled(enabled)

    def _ultrack_config_from_controls(self) -> UltrackConfig:
        return UltrackConfig(
            seg_min_area=self.db_gen_min_area_spin.value(),
            seg_max_area=self.db_gen_max_area_spin.value(),
            seg_foreground_threshold=self.db_gen_fg_thr_spin.value(),
            seg_min_frontier=self.db_gen_min_frontier_spin.value(),
            seg_ws_hierarchy=self.db_gen_ws_hierarchy_combo.currentText(),
            seg_n_workers=self.db_gen_n_workers_spin.value(),
            max_distance=self.db_gen_max_dist_spin.value(),
            max_neighbors=self.db_gen_max_neighbors_spin.value(),
            linking_mode=self.db_gen_linking_mode_combo.currentText(),
            iou_weight=self.db_gen_iou_weight_spin.value(),
            quality_weight=self.db_gen_quality_weight_spin.value(),
            quality_exponent=self.db_gen_quality_exp_spin.value(),
            circularity_weight=self.db_gen_circularity_weight_spin.value(),
            power=self.ultrack_power_spin.value(),
            appear_weight=self.ultrack_appear_spin.value(),
            disappear_weight=self.ultrack_disappear_spin.value(),
            division_weight=self.ultrack_division_spin.value(),
            seed_weight=self.ultrack_seed_weight_spin.value(),
            seed_sigma_space=self.ultrack_seed_space_spin.value(),
            seed_tau_time=self.ultrack_seed_time_spin.value(),
            seed_max_dt=self.ultrack_seed_window_spin.value(),
        )

    def _on_run_tracking_route(self) -> None:
        self._on_run_ultrack()

    def _on_run_tracking_route_terminal(self) -> None:
        self._on_ultrack_terminal()

    def _on_run_ultrack(self) -> None:
        if self._pos_dir is None:
            self._set_ultrack_status("No project open.")
            return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._set_ultrack_status("data.db not found — run DB generation first.")
            return
        working_dir = self._ultrack_workdir()
        tracked_path = self._tracked_path()

        cfg = self._ultrack_config_from_controls()
        needs_validated_export = database_has_annotations(working_dir)
        validated_tracks = None
        tracked_labels = None
        if needs_validated_export:
            validated_tracks = read_validated_tracks(self._pos_dir)
            if not validated_tracks:
                self._set_ultrack_status(
                    "Annotated data.db requires validated tracks for ID-preserving export."
                )
                return
            if _TRACKED_LAYER not in self.viewer.layers:
                self._set_ultrack_status(
                    "Annotated data.db requires the current tracked layer for ID-preserving export."
                )
                return
            tracked_labels = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)

        self.ultrack_progress_bar.setRange(0, 100)
        self.ultrack_progress_bar.setVisible(True)
        self.ultrack_progress_bar.setValue(0)
        self._set_ultrack_status("Starting Ultrack solve…")
        self.run_ultrack_btn.setEnabled(False)
        self.ultrack_terminal_btn.setEnabled(False)

        @thread_worker(connect={
            "yielded":  self._on_ultrack_progress,
            "returned": self._on_run_ultrack_done,
            "errored":  self._on_ultrack_worker_error,
        })
        def _worker():
            for step, total, label in run_solve(working_dir, cfg, overwrite=True):
                yield ("solve", step, total, label)
            yield ("export", 0, 1, "Exporting tracked labels…")
            return export_tracked_labels(
                working_dir,
                cfg,
                tracked_path,
                validated_tracks=validated_tracks,
                tracked_labels=tracked_labels,
            )

        _worker()

    def _on_ultrack_progress(self, payload: tuple) -> None:
        stage, step, total, label = payload
        self._set_ultrack_status(f"[{stage}] {label}")
        if total > 0:
            self.ultrack_progress_bar.setValue(int(100 * step / total))

    def _on_run_ultrack_done(self, labels: np.ndarray | None) -> None:
        self.ultrack_progress_bar.setVisible(False)
        self.run_ultrack_btn.setEnabled(True)
        self.ultrack_terminal_btn.setEnabled(True)
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
        self._refresh_stage_files()
        self._set_ultrack_status(f"Tracking done: {nt} frame(s). Unsaved.")

    def _on_ultrack_terminal(self) -> None:
        import sys
        import tempfile

        if self._pos_dir is None:
            self._set_ultrack_status("No project open.")
            return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._set_ultrack_status("data.db not found — run DB generation first.")
            return
        working_dir = self._ultrack_workdir()
        tracked_path = self._tracked_path()

        cfg = self._ultrack_config_from_controls()
        needs_validated_export = database_has_annotations(working_dir)
        if needs_validated_export:
            validated_tracks = read_validated_tracks(self._pos_dir)
            if not validated_tracks:
                self._set_ultrack_status(
                    "Annotated data.db requires validated tracks for ID-preserving export."
                )
                return
            if tracked_path is None or not tracked_path.exists():
                self._set_ultrack_status(
                    "Annotated data.db requires current tracked labels for ID-preserving export."
                )
                return

        # NOTE: body must live under `if __name__ == "__main__":` because
        # Ultrack's linker uses spawn-based multiprocessing, which re-executes
        # this script in each child via runpy with run_name="__mp_main__".
        # Without the guard, every worker re-runs the full pipeline and races
        # the parent on the SQLite DB.
        python_code = (
            "import sys, pathlib\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))\n"
            "from cellflow.tracking_ultrack.config import TrackingConfig\n"
            "from cellflow.tracking_ultrack.solve import run_solve\n"
            "from cellflow.tracking_ultrack.export import export_tracked_labels\n"
            "from cellflow.database.tracked import read_full_tracked_stack\n"
            "from cellflow.database.validation import read_validated_tracks\n"
            "\n"
            "if __name__ == '__main__':\n"
            f"    pos_dir = pathlib.Path({str(self._pos_dir)!r})\n"
            f"    working_dir = pathlib.Path({str(working_dir)!r})\n"
            f"    tracked_path= pathlib.Path({str(tracked_path)!r})\n"
            f"    needs_validated_export = {bool(needs_validated_export)!r}\n"
            f"    cfg = TrackingConfig(\n"
            f"        power={cfg.power},\n"
            f"        appear_weight={cfg.appear_weight},\n"
            f"        disappear_weight={cfg.disappear_weight},\n"
            f"        division_weight={cfg.division_weight},\n"
            f"        solution_gap={cfg.solution_gap},\n"
            f"        time_limit={cfg.time_limit},\n"
            f"        window_size={cfg.window_size},\n"
            f"    )\n"
            "    print('[1/2] Solving ILP…', flush=True)\n"
            "    for step, total, label in run_solve(working_dir, cfg, overwrite=True):\n"
            "        print(f'  [{step}/{total}] {label}', flush=True)\n"
            "    print('[2/2] Exporting…', flush=True)\n"
            "    validated_tracks = read_validated_tracks(pos_dir) if needs_validated_export else None\n"
            "    tracked_labels = read_full_tracked_stack(tracked_path) if needs_validated_export else None\n"
            "    labels = export_tracked_labels(\n"
            "        working_dir,\n"
            "        cfg,\n"
            "        tracked_path,\n"
            "        validated_tracks=validated_tracks,\n"
            "        tracked_labels=tracked_labels,\n"
            "    )\n"
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
            self._set_ultrack_status("No validated tracks — validate some cells first (press V).")
            return

        tracked_path = self._tracked_path()
        if tracked_path is None or not tracked_path.exists():
            self._set_ultrack_status("Tracked labels not found.")
            return
        contour_path = self._contour_maps_path()
        if contour_path is None or not contour_path.exists():
            self._set_ultrack_status("Missing: contour_maps.tif")
            return
        fg_path = self._foreground_masks_path()
        if fg_path is None or not fg_path.exists():
            self._set_ultrack_status("Missing: foreground_masks.tif (foreground mask) — run Contour Maps first.")
            return
        nucleus_prob_zavg_path = self._nucleus_prob_zavg_path()
        pos_dir = self._pos_dir

        # Capture widget values (same as _on_resolve_with_validation)
        cfg = self._ultrack_config_from_controls()
        cfg.power = self.db_gen_power_spin.value()

        python_code = (
            "import sys, pathlib\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))\n"
            "from cellflow.tracking_ultrack.config import TrackingConfig\n"
            "from cellflow.tracking_ultrack.reseed import resolve_with_canonical_segment\n"
            "from cellflow.database.tracked import read_full_tracked_stack\n"
            "from cellflow.database.validation import read_validated_tracks\n"
            "import numpy as np\n"
            "import tifffile\n"
            "\n"
            "if __name__ == '__main__':\n"
            f"    pos_dir      = pathlib.Path({str(pos_dir)!r})\n"
            f"    contour_path = pathlib.Path({str(contour_path)!r})\n"
            f"    foreground_masks_path = pathlib.Path({str(fg_path)!r})\n"
            f"    tracked_path = pathlib.Path({str(tracked_path)!r})\n"
            "    preview_path = tracked_path.with_name('tracked_labels_resolve_preview.tif')\n"
            f"    nucleus_prob_zavg_path = pathlib.Path({str(nucleus_prob_zavg_path)!r})\n"
            f"    cfg = TrackingConfig(\n"
            f"        seg_min_area={cfg.seg_min_area},\n"
            f"        seg_max_area={cfg.seg_max_area},\n"
            f"        seg_foreground_threshold={cfg.seg_foreground_threshold},\n"
            f"        seg_min_frontier={cfg.seg_min_frontier},\n"
            f"        seg_ws_hierarchy={cfg.seg_ws_hierarchy!r},\n"
            f"        seg_n_workers={cfg.seg_n_workers},\n"
            f"        max_distance={cfg.max_distance},\n"
            f"        max_neighbors={cfg.max_neighbors},\n"
            f"        linking_mode={cfg.linking_mode!r},\n"
            f"        iou_weight={cfg.iou_weight},\n"
            f"        quality_weight={cfg.quality_weight},\n"
            f"        quality_exponent={cfg.quality_exponent},\n"
            f"        circularity_weight={cfg.circularity_weight},\n"
            f"        power={cfg.power},\n"
            f"        appear_weight={cfg.appear_weight},\n"
            f"        disappear_weight={cfg.disappear_weight},\n"
            f"        division_weight={cfg.division_weight},\n"
            f"        seed_weight={cfg.seed_weight},\n"
            f"        seed_sigma_space={cfg.seed_sigma_space},\n"
            f"        seed_tau_time={cfg.seed_tau_time},\n"
            f"        seed_max_dt={cfg.seed_max_dt},\n"
            f"    )\n"
            "    validated_tracks = read_validated_tracks(pos_dir)\n"
            "    print(f'Loaded {len(validated_tracks)} validated track(s).', flush=True)\n"
            "    tracked_labels = read_full_tracked_stack(tracked_path)\n"
            "    print(f'Loaded tracked labels: {tracked_labels.shape}', flush=True)\n"
            "    new_labels, _id_map = resolve_with_canonical_segment(\n"
            "        contour_maps_path=contour_path,\n"
            "        foreground_masks_path=foreground_masks_path,\n"
            "        validated_tracks=validated_tracks,\n"
            "        tracked_labels=tracked_labels,\n"
            "        cfg=cfg,\n"
            "        progress_cb=lambda msg: print(msg, flush=True),\n"
            "        intensity_image_path=nucleus_prob_zavg_path,\n"
            "    )\n"
            "    if new_labels.ndim == 4 and new_labels.shape[1] == 1:\n"
            "        new_labels = new_labels[:, 0]\n"
            "    tifffile.imwrite(str(preview_path), new_labels, compression='zlib')\n"
            "    n_validated = len(validated_tracks)\n"
            "    n_total = int(np.unique(new_labels[new_labels != 0]).size)\n"
            "    print(\n"
            "        f'Done — {n_validated} validated track(s) preserved, '\n"
            "        f'{n_total} total track(s). Preview saved to {preview_path}; tracked_labels.tif not saved.',\n"
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
        if (
            self.ultrack_db_browser_section.is_expanded
            and self.ultrack_db_mode_combo.currentText() == "Hierarchy cut"
        ):
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, self._refresh_ultrack_db_browser)

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
            layer = self.viewer.layers[_VALIDATED_OVERLAY]
            layer.data = data
            layer.opacity = _VALIDATED_OVERLAY_OPACITY
            self._place_validated_overlay_below_spotlight()
            return
        ov = self.viewer.add_labels(
            data,
            name=_VALIDATED_OVERLAY,
            opacity=_VALIDATED_OVERLAY_OPACITY,
            colormap=direct_colormap({None: (0, 0, 0, 0), 1: "#00ff00"}),
        )
        self._place_validated_overlay_below_spotlight()
        # Send the active layer back to tracked so corrections still target it.
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[_TRACKED_LAYER]

    def _place_validated_overlay_below_spotlight(self) -> None:
        if _VALIDATED_OVERLAY not in self.viewer.layers or _SPOTLIGHT_LAYER not in self.viewer.layers:
            return
        validated_index = self.viewer.layers.index(_VALIDATED_OVERLAY)
        spotlight_index = self.viewer.layers.index(_SPOTLIGHT_LAYER)
        if validated_index > spotlight_index:
            self.viewer.layers.move(validated_index, spotlight_index)

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
            self._set_correction_status("Validation toggle: no cell selected (left-click a cell first).")
            return
        t = self._current_t()
        if sel not in self._current_cell_ids(t):
            self._set_correction_status(f"Cell {sel} not present at t={t}.")
            return
        frames = self._frames_with_cell(sel)
        if not frames:
            return
        currently_validated = is_track_validated(self._pos_dir, sel)
        if currently_validated:
            invalidate_track(self._pos_dir, sel)
            self._set_correction_status(f"Cell {sel} invalidated across {len(frames)} frame(s).")
        else:
            validate_track(self._pos_dir, sel, frames)
            self._set_correction_status(f"Cell {sel} validated across {len(frames)} frame(s).")
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _on_extend_backward(self) -> None:
        self._on_extend(direction="backward")

    def _on_extend_forward(self) -> None:
        self._on_extend(direction="forward")

    def _on_extend(self, direction: str) -> None:
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer loaded.")
            return

        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._set_correction_status(
                "Extend: data.db not found — run DB generation first."
            )
            return

        source_id = self.correction_widget._selected_label
        if not source_id:
            self._set_correction_status("Extend: no cell selected (left-click a cell first).")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        t = self._current_t()
        tracked = np.asarray(layer.data)
        T = tracked.shape[0]

        target_frame = t + (1 if direction == "forward" else -1)
        if direction == "forward" and t >= T - 1:
            self._set_correction_status("Already at last frame")
            return
        if direction == "backward" and t <= 0:
            self._set_correction_status("Already at first frame")
            return

        if not np.any(tracked[t] == source_id):
            self._set_correction_status(f"Cell {source_id} not present at t={t}")
            return

        result = extend_track_from_db(
            source_id=source_id,
            source_frame=t,
            direction=direction,
            tracked_labels=tracked,
            db_path=db_path,
            d_max=float(self.extend_max_dist_spin.value()),
            area_weight=float(self.extend_area_weight_spin.value()),
            iou_weight=float(self.extend_iou_weight_spin.value()),
            distance_weight=float(self.extend_distance_weight_spin.value()),
            overlap_penalty=float(self.extend_overlap_penalty_spin.value()),
            greedy_overwrite=self.extend_greedy_overwrite_check.isChecked(),
        )

        if result is None:
            self._set_correction_status(
                f"No candidate within {self.extend_max_dist_spin.value():g}px at t={target_frame}"
            )
            return

        assignments = result.assignments or ()
        if not assignments:
            assignments = (
                SimpleNamespace(cell_id=source_id, mask_2d=result.mask_2d),
            )

        frame = layer.data[result.target_frame]
        changed_ids = {int(assignment.cell_id) for assignment in assignments}
        for cell_id in changed_ids:
            frame[frame == cell_id] = 0

        if self.extend_greedy_overwrite_check.isChecked():
            for assignment in assignments:
                frame[assignment.mask_2d] = int(assignment.cell_id)
        else:
            for assignment in assignments:
                paintable = assignment.mask_2d & (frame == 0)
                frame[paintable] = int(assignment.cell_id)
        layer.refresh()

        step = list(self.viewer.dims.current_step)
        step[0] = result.target_frame
        self.viewer.dims.current_step = tuple(step)

        moved_text = (
            f", reassigned {len(changed_ids) - 1} conflict(s)"
            if len(changed_ids) > 1 else ""
        )
        self._set_correction_status(
            f"Extended cell {source_id} → t={result.target_frame}{moved_text} "
            f"(dist={result.centroid_distance:.1f}px, area={result.area_ratio:.2f}, "
            f"iou={result.centroid_corrected_iou:.2f}, overlap={result.existing_overlap:.2f})"
        )

    def _on_retrack_forward(self) -> None:
        if self._pos_dir is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer loaded.")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._set_correction_status("Tracked layer must be a stack of at least 2 frames.")
            return

        t0 = int(self.viewer.dims.current_step[0])
        if t0 >= layer.data.shape[0] - 1:
            self._set_correction_status("Already at last frame — nothing to retrack forward.")
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
        self._set_correction_status(
            f"Retracked forward from t={t0 + 1}: {n_retracked} frame(s) updated, "
            f"{n_skipped} fully-validated frame(s) skipped. Unsaved."
        )

    def _on_retrack_backward(self) -> None:
        if self._pos_dir is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer loaded.")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._set_correction_status("Tracked layer must be a stack of at least 2 frames.")
            return

        t0 = int(self.viewer.dims.current_step[0])
        if t0 <= 0:
            self._set_correction_status("Already at first frame — nothing to retrack backward.")
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
        self._set_correction_status(
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

        # Show confirmation dialog before overwriting tracked_labels.tif
        msg = QMessageBox(self.viewer.window._qt_window)
        msg.setWindowTitle("Overwrite tracked labels?")
        msg.setText(
            "Resolve will overwrite `tracked_labels.tif`. If you want to preserve the current tracking, "
            "copy the file first.\n\nContinue?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        if msg.exec() != QMessageBox.Yes:
            self._set_ultrack_status("Resolve cancelled.")
            return

        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_ultrack_status("No tracked layer loaded.")
            return

        contour_path = self._contour_maps_path()
        if contour_path is None or not contour_path.exists():
            self._set_ultrack_status("Missing: contour_maps.tif — run Contour Maps first.")
            return
        fg_path = self._foreground_masks_path()
        if fg_path is None or not fg_path.exists():
            self._set_ultrack_status(
                "Missing: foreground_masks.tif (foreground mask) — run Contour Maps first."
            )
            return
        nucleus_prob_zavg_path = self._nucleus_prob_zavg_path()
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
            new_labels, _id_map = result
            # Normalize (T, 1, Y, X) → (T, Y, X) if needed
            if new_labels.ndim == 4 and new_labels.shape[1] == 1:
                new_labels = new_labels[:, 0]
            if _TRACKED_LAYER in self.viewer.layers:
                self.viewer.layers[_TRACKED_LAYER].data = new_labels
            else:
                self.viewer.add_labels(new_labels, name=_TRACKED_LAYER)
            layer = self.viewer.layers[_TRACKED_LAYER]
            self.correction_widget.activate_layer(layer)
            self._refresh_validated_overlay()
            self._refresh_validation_counter()
            n_total_tracks = int(np.unique(new_labels[new_labels != 0]).size)
            self._set_ultrack_status(
                f"Re-solve complete: {n_validated} validated track(s) preserved, "
                f"{n_total_tracks} total track(s) in output. Unsaved."
            )

        def _on_resolve_progress(msg: str) -> None:
            self._set_ultrack_status(msg)

        def _on_resolve_error(exc: Exception) -> None:
            self.run_ultrack_btn.setEnabled(True)
            self.ultrack_terminal_btn.setEnabled(True)
            self.ultrack_progress_bar.setVisible(False)
            self.ultrack_progress_bar.setRange(0, 100)
            self._on_ultrack_worker_error(exc)

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
                        resolve_with_canonical_segment(
                            contour_maps_path=contour_path,
                            foreground_masks_path=fg_path,
                            validated_tracks=validated_tracks,
                            tracked_labels=tracked_labels,
                            cfg=cfg,
                            progress_cb=_progress,
                            intensity_image_path=nucleus_prob_zavg_path,
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
    # State persistence
    # ──────────────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "save_source":      self.save_source_check.isChecked(),
            "cellprob": {
                "min":       self.cp_min_spin.value(),
                "max":       self.cp_max_spin.value(),
                "step":      self.cp_step_spin.value(),
                "gamma_min": self.cp_gamma_min_spin.value(),
                "gamma_max": self.cp_gamma_max_spin.value(),
                "gamma_step": self.cp_gamma_step_spin.value(),
                "foreground_threshold": self.contour_fg_threshold_spin.value(),
                "flow_threshold": self.contour_flow_threshold_spin.value(),
            },
            "contour_filter": {
                "median_time": self.contour_filter_median_time_spin.value(),
                "median_space": self.contour_filter_median_space_spin.value(),
                "gauss_time": self.contour_filter_gauss_time_spin.value(),
                "gauss_space": self.contour_filter_gauss_space_spin.value(),
            },
            "db_generation": {
                "min_area":         self.db_gen_min_area_spin.value(),
                "max_area":         self.db_gen_max_area_spin.value(),
                "fg_threshold":     self.db_gen_fg_thr_spin.value(),
                "min_frontier":     self.db_gen_min_frontier_spin.value(),
                "ws_hierarchy":     self.db_gen_ws_hierarchy_combo.currentText(),
                "max_distance":     self.db_gen_max_dist_spin.value(),
                "max_neighbors":    self.db_gen_max_neighbors_spin.value(),
                "linking_mode":     self.db_gen_linking_mode_combo.currentText(),
                "iou_weight":       self.db_gen_iou_weight_spin.value(),
                "quality_weight":   self.db_gen_quality_weight_spin.value(),
                "quality_exponent": self.db_gen_quality_exp_spin.value(),
                "circularity_weight": self.db_gen_circularity_weight_spin.value(),
                "power":            self.db_gen_power_spin.value(),
                "n_workers":        self.db_gen_n_workers_spin.value(),
                "use_validated":    self.db_gen_use_validated_check.isChecked(),
            },
            "extend": {
                "max_distance":     self.extend_max_dist_spin.value(),
                "area_weight":      self.extend_area_weight_spin.value(),
                "iou_weight":       self.extend_iou_weight_spin.value(),
                "distance_weight":  self.extend_distance_weight_spin.value(),
                "overlap_penalty":  self.extend_overlap_penalty_spin.value(),
                "greedy_overwrite": self.extend_greedy_overwrite_check.isChecked(),
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
                "power":            self.ultrack_power_spin.value(),
                "quality_exponent": self.ultrack_quality_exp_spin.value(),
                "seed_weight":      self.ultrack_seed_weight_spin.value(),
                "seed_sigma_space": self.ultrack_seed_space_spin.value(),
                "seed_tau_time":    self.ultrack_seed_time_spin.value(),
                "seed_max_dt":      self.ultrack_seed_window_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if "save_source" in state:
            self.save_source_check.setChecked(state["save_source"])
        if "cellprob" in state:
            cp = state["cellprob"]
            if "min"        in cp: self.cp_min_spin.setValue(cp["min"])
            if "max"        in cp: self.cp_max_spin.setValue(cp["max"])
            if "step"       in cp: self.cp_step_spin.setValue(cp["step"])
            if "gamma_min"  in cp: self.cp_gamma_min_spin.setValue(cp["gamma_min"])
            if "gamma_max"  in cp: self.cp_gamma_max_spin.setValue(cp["gamma_max"])
            if "gamma_step" in cp: self.cp_gamma_step_spin.setValue(cp["gamma_step"])
            if "flow_threshold" in cp:
                self.contour_flow_threshold_spin.setValue(cp["flow_threshold"])
            if "foreground_threshold" in cp:
                self.contour_fg_threshold_spin.setValue(cp["foreground_threshold"])
        if "contour_filter" in state:
            cf = state["contour_filter"]
            if "median_time" in cf:
                self.contour_filter_median_time_spin.setValue(cf["median_time"])
            if "median_space" in cf:
                self.contour_filter_median_space_spin.setValue(cf["median_space"])
            if "gauss_time" in cf:
                self.contour_filter_gauss_time_spin.setValue(cf["gauss_time"])
            if "gauss_space" in cf:
                self.contour_filter_gauss_space_spin.setValue(cf["gauss_space"])
        if "db_generation" in state:
            dbg = state["db_generation"]
            if "min_area"         in dbg: self.db_gen_min_area_spin.setValue(dbg["min_area"])
            if "max_area"         in dbg: self.db_gen_max_area_spin.setValue(dbg["max_area"])
            if "fg_threshold"     in dbg: self.db_gen_fg_thr_spin.setValue(dbg["fg_threshold"])
            if "min_frontier"     in dbg: self.db_gen_min_frontier_spin.setValue(dbg["min_frontier"])
            if "ws_hierarchy"     in dbg:
                idx = self.db_gen_ws_hierarchy_combo.findText(dbg["ws_hierarchy"])
                if idx >= 0:
                    self.db_gen_ws_hierarchy_combo.setCurrentIndex(idx)
            if "max_distance"     in dbg: self.db_gen_max_dist_spin.setValue(dbg["max_distance"])
            if "max_neighbors"    in dbg: self.db_gen_max_neighbors_spin.setValue(dbg["max_neighbors"])
            if "linking_mode"     in dbg:
                idx = self.db_gen_linking_mode_combo.findText(dbg["linking_mode"])
                if idx >= 0:
                    self.db_gen_linking_mode_combo.setCurrentIndex(idx)
            if "iou_weight"       in dbg: self.db_gen_iou_weight_spin.setValue(dbg["iou_weight"])
            if "quality_weight"   in dbg: self.db_gen_quality_weight_spin.setValue(dbg["quality_weight"])
            if "quality_exponent" in dbg: self.db_gen_quality_exp_spin.setValue(dbg["quality_exponent"])
            if "circularity_weight" in dbg: self.db_gen_circularity_weight_spin.setValue(dbg["circularity_weight"])
            if "power"            in dbg: self.db_gen_power_spin.setValue(dbg["power"])
            if "n_workers"        in dbg: self.db_gen_n_workers_spin.setValue(dbg["n_workers"])
            if "use_validated"    in dbg: self.db_gen_use_validated_check.setChecked(dbg["use_validated"])
        if "extend" in state:
            ext = state["extend"]
            if "max_distance"    in ext: self.extend_max_dist_spin.setValue(ext["max_distance"])
            if "area_weight"     in ext: self.extend_area_weight_spin.setValue(ext["area_weight"])
            if "iou_weight"      in ext: self.extend_iou_weight_spin.setValue(ext["iou_weight"])
            if "distance_weight" in ext: self.extend_distance_weight_spin.setValue(ext["distance_weight"])
            if "overlap_penalty" in ext: self.extend_overlap_penalty_spin.setValue(ext["overlap_penalty"])
            if "greedy_overwrite" in ext: self.extend_greedy_overwrite_check.setChecked(ext["greedy_overwrite"])
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
            if "power"            in ul: self.ultrack_power_spin.setValue(ul["power"])
            if "quality_exponent" in ul: self.ultrack_quality_exp_spin.setValue(ul["quality_exponent"])
            if "seed_weight"      in ul: self.ultrack_seed_weight_spin.setValue(ul["seed_weight"])
            if "seed_sigma_space" in ul: self.ultrack_seed_space_spin.setValue(ul["seed_sigma_space"])
            if "seed_tau_time"    in ul: self.ultrack_seed_time_spin.setValue(ul["seed_tau_time"])
            if "seed_max_dt"      in ul: self.ultrack_seed_window_spin.setValue(ul["seed_max_dt"])
