"""Nucleus workflow widget for hypothesis generation and tracking in CellFlow.

Flat layout — action buttons in a two-column grid at top, one collapsible
parameter panel, Ultrack DB browser, correction section at bottom.

Stages:
  1. Contour Maps → ``contour_maps.tif``, ``foreground_scores.tif``,
     ``foreground_masks.tif``
  2. Ultrack Database Generation → ``data.db``
  3. Ultrack Tracking → ``tracked_labels.tif``
  4. Correction (load / save / extend / retrack / reassign / remove unvalidated)
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from napari.utils.colormaps import direct_colormap
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QKeySequence
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.correction.labels import best_overlapping_label
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
from cellflow.napari.ui_style import (
    action_button,
    add_block_checkbox_row,
    add_block_pair_row,
    block_grid,
    compact_spinbox,
    danger_button,
    muted_label,
    status_label,
)
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.tracking.retracker import retrack_frame_constrained
from cellflow.tracking_ultrack.config import TrackingConfig as UltrackConfig
from cellflow.tracking_ultrack.db_build import build_ultrack_database
from cellflow.tracking_ultrack.export import export_tracked_labels
from cellflow.tracking_ultrack.extend import extend_track_from_db
from cellflow.tracking_ultrack.ingest import _select_solver
from cellflow.tracking_ultrack.solve import database_has_annotations, run_solve

logger = logging.getLogger(__name__)

try:
    from ultrack.core.segmentation.processing import segment as _ultrack_segment
except ImportError:
    _ultrack_segment = None  # type: ignore[assignment]

# ── Layer name constants ──────────────────────────────────────────────────────
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


# ── Tiny helpers ──────────────────────────────────────────────────────────────

def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #555;")
    return line


def _heading(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: 600;")
    return lbl


def _make_status() -> QLabel:
    lbl = QLabel("")
    lbl.setWordWrap(True)
    lbl.setVisible(False)
    status_label(lbl)
    return lbl


def _make_progress() -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(True)
    bar.setVisible(False)
    return bar


def _dspin(lo, hi, val, step=0.1, decimals=2, tooltip=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
    s.setDecimals(decimals); s.setToolTip(tooltip)
    return s


def _ispin(lo, hi, val, step=1, tooltip=""):
    s = QSpinBox()
    s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
    s.setToolTip(tooltip)
    return s


def _btn(text, tooltip=""):
    b = QPushButton(text)
    b.setToolTip(tooltip)
    action_button(b, expand=True)
    return b


def _button_grid(*rows: tuple[QPushButton, ...]) -> QGridLayout:
    grid = QGridLayout()
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(4)
    for r, buttons in enumerate(rows):
        for c, btn in enumerate(buttons):
            span = 2 - c if c == len(buttons) - 1 and len(buttons) == 1 else 1
            grid.addWidget(btn, r, c, 1, span)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)
    return grid


@dataclass(frozen=True)
class _HierarchyCutState:
    node_ids: tuple[int, ...]
    height: float | None


# ══════════════════════════════════════════════════════════════════════════════


class NucleusWorkflowWidget(QWidget):
    """Nucleus hypothesis generation and tracking — flat action-button layout."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._stop_flag: bool = False

        self._contour_worker = None
        self._db_gen_worker = None
        self._ultrack_worker = None

        self._ultrack_db_preview_cache: dict = {}
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

    # ================================================================
    # UI
    # ================================================================
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)

        # ── Pipeline files (single deduplicated panel) ────────────────
        self._files_widget = PipelineFilesWidget(
            [
                ("Inputs", [
                    ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
                    ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
                    ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
                    ("0_input/cell_zavg.tif", "Cell z-avg"),
                    ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
                ]),
                ("Intermediates", [
                    ("2_nucleus/contour_maps.tif", "Contour maps"),
                    ("2_nucleus/foreground_scores.tif", "Foreground scores"),
                    ("2_nucleus/foreground_masks.tif", "Foreground masks"),
                    ("2_nucleus/ultrack_workdir/data.db", "Ultrack database"),
                ]),
                ("Output", [
                    ("2_nucleus/tracked_labels.tif", "Tracked labels"),
                ]),
            ],
            viewer=self.viewer,
        )
        root.addWidget(
            CollapsibleSection("Pipeline Files", self._files_widget, expanded=False)
        )

        # ── Pipeline action buttons (2-column grid) ──────────────────
        self.preview_contour_btn = _btn(
            "Preview Contours",
            "Build contour map for the current frame only and display in napari.",
        )
        self.build_btn = _btn(
            "Build Contours",
            "Build consensus contour maps and foreground masks for all frames.",
        )
        self.preview_contour_filter_btn = _btn(
            "Preview Filter",
            "Preview filtered contour_maps.tif in napari without overwriting.",
        )
        self.run_contour_filter_btn = _btn(
            "Run Filter",
            "Filter contour_maps.tif and overwrite the file on disk.",
        )
        self.run_db_gen_btn = _btn(
            "Run DB Generation",
            "Build Ultrack candidate database from contour maps.",
        )
        self.run_ultrack_btn = _btn(
            "Run Ultrack",
            "Solve ILP tracking and export tracked_labels.tif.",
        )
        self.cancel_btn = _btn("Cancel", "Cancel the currently running pipeline step.")
        self.cancel_btn.setEnabled(False)

        root.addLayout(_button_grid(
            (self.preview_contour_btn, self.build_btn),
            (self.preview_contour_filter_btn, self.run_contour_filter_btn),
            (self.run_db_gen_btn, self.run_ultrack_btn),
            (self.cancel_btn,),
        ))

        self.pipeline_status_lbl = _make_status()
        root.addWidget(self.pipeline_status_lbl)
        self.pipeline_progress_bar = _make_progress()
        root.addWidget(self.pipeline_progress_bar)

        # ── Single collapsible parameter panel ───────────────────────
        self._build_params_section(root)

        # ── Ultrack Database Browser ─────────────────────────────────
        self._build_db_browser_section(root)

        # ── Correction (group box) ───────────────────────────────────
        self._build_correction_section(root)

        root.addStretch()

    # -- Parameters --------------------------------------------------------

    def _build_params_section(self, root: QVBoxLayout) -> None:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # Contour — Cellprob Sweep
        lay.addWidget(_heading("Contour — Cellprob Sweep"))
        g = block_grid(horizontal_spacing=12)
        self.cp_min_spin = _dspin(-20, 20, -3.0, 1.0, 1)
        self.cp_max_spin = _dspin(-20, 20, 0.0, 1.0, 1)
        self.cp_step_spin = _dspin(0.1, 10, 1.0, 0.5, 1)
        self.contour_flow_threshold_spin = _dspin(
            0, 10, 0.0, 0.1, 2,
            "Cellpose flow error threshold passed to compute_masks. 0 disables filtering.",
        )
        add_block_pair_row(g, 0,
            "Min:", compact_spinbox(self.cp_min_spin),
            "Max:", compact_spinbox(self.cp_max_spin))
        add_block_pair_row(g, 1,
            "Step:", compact_spinbox(self.cp_step_spin),
            "Flow thr:", compact_spinbox(self.contour_flow_threshold_spin))
        lay.addLayout(g)

        # Contour — Gamma Averaging
        _gamma_tip = (
            "Gamma correction on Cellpose probability logits. "
            "<1 boosts dim signals; >1 suppresses. 1.0 = no correction. "
            "Contour maps are averaged over all gamma values in [min, max]."
        )
        lay.addWidget(_heading("Contour — Gamma Averaging"))
        g = block_grid(horizontal_spacing=12)
        self.cp_gamma_min_spin = _dspin(0.05, 5, 1.0, 0.05, 2, _gamma_tip)
        self.cp_gamma_max_spin = _dspin(0.05, 5, 1.0, 0.05, 2, _gamma_tip)
        self.cp_gamma_step_spin = _dspin(0.05, 2, 0.25, 0.05, 2, _gamma_tip)
        add_block_pair_row(g, 0,
            "Min:", compact_spinbox(self.cp_gamma_min_spin),
            "Max:", compact_spinbox(self.cp_gamma_max_spin))
        add_block_pair_row(g, 1,
            "Step:", compact_spinbox(self.cp_gamma_step_spin))
        lay.addLayout(g)

        # Contour — Foreground Output
        lay.addWidget(_heading("Contour — Foreground Output"))
        g = block_grid(horizontal_spacing=12)
        self.contour_fg_threshold_spin = _dspin(
            0, 1, 0.5, 0.01, 2,
            "Threshold applied to the fuzzy foreground score.",
        )
        self.save_source_check = QCheckBox("Save label images")
        self.save_source_check.setToolTip(
            "Save all label images used for contour building in 2_nucleus/source_labels/"
        )
        add_block_pair_row(g, 0,
            "FG threshold:", compact_spinbox(self.contour_fg_threshold_spin))
        add_block_checkbox_row(g, 1, self.save_source_check)
        lay.addLayout(g)

        # Contour — Post-Filter
        lay.addWidget(_heading("Contour — Post-Filter"))
        g = block_grid(horizontal_spacing=12)
        self.contour_filter_median_time_spin = _ispin(1, 15, 1)
        self.contour_filter_median_space_spin = _ispin(1, 15, 1)
        self.contour_filter_gauss_time_spin = _dspin(0, 10, 0.0, 0.1, 1)
        self.contour_filter_gauss_space_spin = _dspin(0, 10, 0.0, 0.1, 1)
        add_block_pair_row(g, 0,
            "Median t:", compact_spinbox(self.contour_filter_median_time_spin),
            "Median xy:", compact_spinbox(self.contour_filter_median_space_spin))
        add_block_pair_row(g, 1,
            "Gauss t σ:", compact_spinbox(self.contour_filter_gauss_time_spin),
            "Gauss xy σ:", compact_spinbox(self.contour_filter_gauss_space_spin))
        lay.addLayout(g)

        lay.addWidget(_separator())

        # DB Generation — Candidates
        lay.addWidget(_heading("DB Generation — Candidates"))
        g = block_grid(horizontal_spacing=12)
        self.db_gen_min_area_spin = _ispin(0, 1_000_000, 300, tooltip="Minimum segment area in pixels.")
        self.db_gen_max_area_spin = _ispin(0, 10_000_000, 100_000, tooltip="Maximum segment area in pixels.")
        self.db_gen_fg_thr_spin = _dspin(
            -5, 1, 0.5, 0.05, 2,
            "Pixel-level foreground threshold for ultrack segmentation.",
        )
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
            "FG thr:", compact_spinbox(self.db_gen_fg_thr_spin),
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
        self.db_gen_linking_mode_combo.addItems(["default", "iou"])
        self.db_gen_iou_weight_spin = _dspin(0, 1, 1.0, 0.05, 2)
        self.db_gen_iou_weight_spin.setEnabled(False)
        add_block_pair_row(g, 0,
            "Max dist:", compact_spinbox(self.db_gen_max_dist_spin),
            "Max neighbors:", compact_spinbox(self.db_gen_max_neighbors_spin))
        add_block_pair_row(g, 1,
            "Linking mode:", self.db_gen_linking_mode_combo,
            "IoU weight:", compact_spinbox(self.db_gen_iou_weight_spin))
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

        # Hidden/deprecated — needed for state persistence only
        self.db_gen_power_spin = _dspin(0.1, 20, 4.0, 0.5, 2)

        # DB Generation — Validated Seed Prior
        lay.addWidget(_heading("DB Generation — Validated Seed Prior"))
        g = block_grid(horizontal_spacing=12)
        self.db_gen_use_validated_check = QCheckBox("Use validated corrections")
        self.ultrack_seed_weight_spin = _dspin(
            0, 10, 0.5, 0.1, 2,
            "Additive reward for candidates similar to nearby validated cells. 0 disables.",
        )
        self.ultrack_seed_space_spin = _dspin(
            1, 500, 25.0, 5.0, 1,
            "Spatial decay scale for seed proximity.",
        )
        self.ultrack_seed_time_spin = _dspin(
            0.1, 50, 2.0, 0.5, 1,
            "Temporal decay scale in frames.",
        )
        self.ultrack_seed_window_spin = _ispin(
            0, 100, 5,
            tooltip="Max frame distance from a validated cell used for seed affinity.",
        )
        add_block_checkbox_row(g, 0, self.db_gen_use_validated_check)
        add_block_pair_row(g, 1,
            "Seed weight:", compact_spinbox(self.ultrack_seed_weight_spin),
            "Seed space:", compact_spinbox(self.ultrack_seed_space_spin))
        add_block_pair_row(g, 2,
            "Seed time:", compact_spinbox(self.ultrack_seed_time_spin),
            "Seed window:", compact_spinbox(self.ultrack_seed_window_spin))
        lay.addLayout(g)

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
        self.ultrack_solver_lbl = QLabel("—")
        add_block_pair_row(g, 0,
            "Power:", compact_spinbox(self.ultrack_power_spin),
            "Solver:", self.ultrack_solver_lbl)
        lay.addLayout(g)

        # Attribution
        ultrack_attrib = QLabel(
            "Ultrack tracking is powered by the "
            '<a href="https://github.com/royerlab/ultrack">Ultrack</a> project.'
        )
        ultrack_attrib.setOpenExternalLinks(True)
        ultrack_attrib.setWordWrap(True)
        muted_label(ultrack_attrib, size_pt=9)
        lay.addWidget(ultrack_attrib)

        root.addWidget(CollapsibleSection("Parameters", inner, expanded=False))

    # -- Ultrack Database Browser ------------------------------------------

    def _build_db_browser_section(self, root: QVBoxLayout) -> None:
        _inner = QWidget()
        lay = QVBoxLayout(_inner)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.ultrack_db_info_lbl = QLabel("—")
        self.ultrack_db_info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ultrack_db_info_lbl.setWordWrap(True)
        self.ultrack_db_info_lbl.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum,
        )
        lay.addWidget(self.ultrack_db_info_lbl)

        self.ultrack_db_hierarchy_slider = QSlider(Qt.Horizontal)
        self.ultrack_db_hierarchy_slider.setRange(0, 100)
        self.ultrack_db_hierarchy_slider.setValue(50)
        self.ultrack_db_hierarchy_slider.setToolTip(
            "Hierarchy cut level: 0 = most split, 1 = most merged"
        )
        self.ultrack_db_hierarchy_slider.setEnabled(False)
        self.ultrack_db_height_lbl = QLabel("0.50")
        self.ultrack_db_height_lbl.setFixedWidth(48)
        self._ultrack_db_slider_row = QWidget()
        _slider_lay = QHBoxLayout(self._ultrack_db_slider_row)
        _slider_lay.setContentsMargins(0, 0, 0, 0)
        _slider_lay.addWidget(self.ultrack_db_hierarchy_slider)
        _slider_lay.addWidget(self.ultrack_db_height_lbl)
        lay.addWidget(self._ultrack_db_slider_row)

        _db_btn_row = QWidget()
        _db_btn_lay = QHBoxLayout(_db_btn_row)
        _db_btn_lay.setContentsMargins(0, 0, 0, 0)
        _db_btn_lay.setSpacing(4)
        self.ultrack_db_active_btn = QPushButton("Activate")
        self.ultrack_db_active_btn.setCheckable(True)
        self.ultrack_db_active_btn.setChecked(False)
        self.ultrack_db_active_btn.setToolTip(
            "Load contour maps and foreground masks into viewer and enable DB preview"
        )
        self.ultrack_db_refresh_btn = QPushButton()
        self.ultrack_db_refresh_btn.setToolTip("Refresh Ultrack database browser")
        self.ultrack_db_refresh_btn.setIcon(QIcon.fromTheme("view-refresh"))
        self.ultrack_db_refresh_btn.setEnabled(False)
        _db_btn_lay.addWidget(self.ultrack_db_active_btn)
        _db_btn_lay.addWidget(self.ultrack_db_refresh_btn)
        lay.addWidget(_db_btn_row)

        self.ultrack_db_prob_alpha_check = QCheckBox("Node prob transparency")
        self.ultrack_db_prob_alpha_check.setToolTip(
            "Modulate label opacity by node probability"
        )
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
        for cb in (
            self.ultrack_db_prob_alpha_check,
            self.ultrack_db_connected_focus_check,
            self.ultrack_db_edge_alpha_check,
            self.ultrack_db_show_validated_check,
            self.ultrack_db_show_fake_check,
        ):
            lay.addWidget(cb)

        self.ultrack_db_section_status_lbl = QLabel("")
        self.ultrack_db_section_status_lbl.setWordWrap(True)
        self.ultrack_db_section_status_lbl.setVisible(False)
        lay.addWidget(self.ultrack_db_section_status_lbl)

        self.ultrack_db_browser_section = CollapsibleSection(
            "Ultrack Database Browser", _inner, expanded=False,
        )
        root.addWidget(self.ultrack_db_browser_section)

    # -- Correction --------------------------------------------------------

    def _build_correction_section(self, root: QVBoxLayout) -> None:
        group = QGroupBox("Correction")
        group.setStyleSheet(
            "QGroupBox { font-weight: 600; margin-top: 8px; padding-top: 14px; }"
        )
        group_lay = QVBoxLayout(group)
        group_lay.setContentsMargins(8, 16, 8, 8)
        group_lay.setSpacing(6)

        # ── Action buttons — 2-column grid ────────────────────────
        self.load_tracked_btn = _btn(
            "Load Labels", "Load tracked nucleus labels from disk.")
        self.save_tracked_btn = _btn(
            "Save Labels", "Save tracked nucleus labels to disk.")
        self.extend_back_btn = _btn(
            "◀ Extend (A)", "Extend selected track one frame backward.")
        self.extend_fwd_btn = _btn(
            "Extend (D) ▶", "Extend selected track one frame forward.")
        self.retrack_back_btn = _btn(
            "◀ Retrack (Q)", "Retrack all labels backward from current frame.")
        self.retrack_fwd_btn = _btn(
            "Retrack (E) ▶", "Retrack all labels forward from current frame.")
        self.reassign_ids_btn = _btn(
            "Reassign IDs", "Reassign cell IDs to contiguous range 1–N.")
        self.remove_unvalidated_btn = _btn(
            "Remove Unvalidated",
            "Remove nucleus label pixels not marked validated for their frame.",
        )
        danger_button(self.remove_unvalidated_btn)

        group_lay.addLayout(_button_grid(
            (self.load_tracked_btn, self.save_tracked_btn),
            (self.extend_back_btn, self.extend_fwd_btn),
            (self.retrack_back_btn, self.retrack_fwd_btn),
            (self.reassign_ids_btn, self.remove_unvalidated_btn),
        ))

        self.correction_status_lbl = _make_status()
        group_lay.addWidget(self.correction_status_lbl)

        self.validation_counter_lbl = QLabel("")
        self.validation_counter_lbl.setWordWrap(True)
        group_lay.addWidget(self.validation_counter_lbl)

        # ── Extend parameters (collapsible) ───────────────────────
        extend_inner = QWidget()
        extend_lay = QVBoxLayout(extend_inner)
        extend_lay.setContentsMargins(0, 0, 0, 0)
        extend_lay.setSpacing(4)
        g = block_grid(horizontal_spacing=12)
        self.extend_max_dist_spin = _dspin(0, 500, 40.0, 1.0, 1)
        self.extend_area_weight_spin = _dspin(0, 10, 1.0, 0.1, 2)
        self.extend_iou_weight_spin = _dspin(0, 10, 1.0, 0.1, 2)
        self.extend_distance_weight_spin = _dspin(0, 10, 0.25, 0.05, 2)
        self.extend_overlap_penalty_spin = _dspin(0, 10, 1.0, 0.1, 2)
        self.extend_greedy_overwrite_check = QCheckBox("Greedy overwrite")
        add_block_pair_row(g, 0,
            "Max dist:", compact_spinbox(self.extend_max_dist_spin),
            "Area wt:", compact_spinbox(self.extend_area_weight_spin))
        add_block_pair_row(g, 1,
            "IoU wt:", compact_spinbox(self.extend_iou_weight_spin),
            "Dist wt:", compact_spinbox(self.extend_distance_weight_spin))
        add_block_pair_row(g, 2,
            "Overlap pen:", compact_spinbox(self.extend_overlap_penalty_spin))
        add_block_checkbox_row(g, 3, self.extend_greedy_overwrite_check)
        extend_lay.addLayout(g)
        group_lay.addWidget(
            CollapsibleSection("Extend Parameters", extend_inner, expanded=False)
        )

        # ── Retrack parameters (collapsible) ──────────────────────
        retrack_inner = QWidget()
        retrack_lay = QVBoxLayout(retrack_inner)
        retrack_lay.setContentsMargins(0, 0, 0, 0)
        retrack_lay.setSpacing(4)
        g = block_grid(horizontal_spacing=12)
        self.retrack_max_dist_spin = _dspin(0, 500, 20.0, 1.0, 1)
        add_block_pair_row(g, 0,
            "Max dist:", compact_spinbox(self.retrack_max_dist_spin))
        retrack_lay.addLayout(g)
        group_lay.addWidget(
            CollapsibleSection("Retrack Parameters", retrack_inner, expanded=False)
        )

        # ── Inline CorrectionWidget ───────────────────────────────
        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
        )
        self.correction_widget.set_edit_callback(self._on_cells_edited)
        group_lay.addWidget(self.correction_widget)

        group_lay.addWidget(CollapsibleSection(
            "Correction Shortcuts",
            self.correction_widget.build_shortcuts_widget(),
            expanded=False,
        ))

        root.addWidget(group)

    # ================================================================
    # Signals
    # ================================================================
    def _connect_signals(self) -> None:
        # Pipeline buttons
        self.preview_contour_btn.clicked.connect(self._on_preview_contour_maps)
        self.build_btn.clicked.connect(self._on_build_contour_maps)
        self.preview_contour_filter_btn.clicked.connect(self._on_preview_contour_filter)
        self.run_contour_filter_btn.clicked.connect(self._on_run_contour_filter)
        self.run_db_gen_btn.clicked.connect(self._on_run_db_generation)
        self.run_ultrack_btn.clicked.connect(self._on_run_ultrack)
        self.cancel_btn.clicked.connect(self._on_cancel)

        # Parameter interactions
        self.db_gen_linking_mode_combo.currentTextChanged.connect(
            self._on_db_gen_mode_changed
        )
        self.db_gen_use_validated_check.toggled.connect(
            self._set_resolve_prior_controls_enabled
        )

        # DB Browser
        self.ultrack_db_active_btn.toggled.connect(self._on_ultrack_db_activate)
        self.ultrack_db_refresh_btn.clicked.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_hierarchy_slider.valueChanged.connect(
            self._on_ultrack_db_slider_changed
        )
        self.ultrack_db_prob_alpha_check.toggled.connect(
            self._refresh_ultrack_db_browser
        )
        self.ultrack_db_connected_focus_check.toggled.connect(
            self._refresh_ultrack_db_browser
        )
        self.ultrack_db_edge_alpha_check.toggled.connect(
            self._refresh_ultrack_db_browser
        )
        self.ultrack_db_show_validated_check.toggled.connect(
            self._refresh_ultrack_db_browser
        )
        self.ultrack_db_show_fake_check.toggled.connect(
            self._refresh_ultrack_db_browser
        )

        # Correction
        self.load_tracked_btn.clicked.connect(self._on_load_tracked)
        self.save_tracked_btn.clicked.connect(self._on_save_tracked)
        self.reassign_ids_btn.clicked.connect(self._on_reassign_ids)
        self.extend_back_btn.clicked.connect(self._on_extend_backward)
        self.extend_fwd_btn.clicked.connect(self._on_extend_forward)
        self.retrack_back_btn.clicked.connect(self._on_retrack_backward)
        self.retrack_fwd_btn.clicked.connect(self._on_retrack_forward)
        self.remove_unvalidated_btn.clicked.connect(
            self._on_remove_unvalidated_labels
        )

        # Viewer events & keyboard
        self.viewer.dims.events.current_step.connect(self._on_dims_step_changed)
        self.viewer.bind_key("V", self._kb_toggle_cell_validation, overwrite=True)
        self._install_correction_shortcuts()
        self.correction_widget._activate_btn.toggled.connect(
            self._on_correction_mode_toggled
        )

        # Initial state
        solver = _select_solver()
        solver_display = "Gurobi (licensed)" if solver == "GUROBI" else "CBC"
        self.ultrack_solver_lbl.setText(solver_display)
        self._set_resolve_prior_controls_enabled()

    # ================================================================
    # Path helpers
    # ================================================================
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
        wd = self._ultrack_workdir()
        return wd / "data.db" if wd else None

    def _foreground_masks_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "foreground_masks.tif" if self._pos_dir else None

    def _nucleus_prob_zavg_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif" if self._pos_dir else None

    # ================================================================
    # Public API
    # ================================================================
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._files_widget.refresh(pos_dir)
        if pos_dir is None:
            self.correction_widget.deactivate()
            return
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def get_state(self) -> dict:
        return {
            "save_source": self.save_source_check.isChecked(),
            "cellprob": {
                "min": self.cp_min_spin.value(),
                "max": self.cp_max_spin.value(),
                "step": self.cp_step_spin.value(),
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
                "min_area": self.db_gen_min_area_spin.value(),
                "max_area": self.db_gen_max_area_spin.value(),
                "fg_threshold": self.db_gen_fg_thr_spin.value(),
                "min_frontier": self.db_gen_min_frontier_spin.value(),
                "ws_hierarchy": self.db_gen_ws_hierarchy_combo.currentText(),
                "max_distance": self.db_gen_max_dist_spin.value(),
                "max_neighbors": self.db_gen_max_neighbors_spin.value(),
                "linking_mode": self.db_gen_linking_mode_combo.currentText(),
                "iou_weight": self.db_gen_iou_weight_spin.value(),
                "quality_weight": self.db_gen_quality_weight_spin.value(),
                "quality_exponent": self.db_gen_quality_exp_spin.value(),
                "circularity_weight": self.db_gen_circularity_weight_spin.value(),
                "power": self.db_gen_power_spin.value(),
                "n_workers": self.db_gen_n_workers_spin.value(),
                "use_validated": self.db_gen_use_validated_check.isChecked(),
                "seed_weight": self.ultrack_seed_weight_spin.value(),
                "seed_sigma_space": self.ultrack_seed_space_spin.value(),
                "seed_tau_time": self.ultrack_seed_time_spin.value(),
                "seed_max_dt": self.ultrack_seed_window_spin.value(),
            },
            "extend": {
                "max_distance": self.extend_max_dist_spin.value(),
                "area_weight": self.extend_area_weight_spin.value(),
                "iou_weight": self.extend_iou_weight_spin.value(),
                "distance_weight": self.extend_distance_weight_spin.value(),
                "overlap_penalty": self.extend_overlap_penalty_spin.value(),
                "greedy_overwrite": self.extend_greedy_overwrite_check.isChecked(),
            },
            "ultrack": {
                "max_partitions": self.ultrack_max_partitions_spin.value(),
                "n_frames": self.ultrack_n_frames_spin.value(),
                "appear_weight": self.ultrack_appear_spin.value(),
                "disappear_weight": self.ultrack_disappear_spin.value(),
                "division_weight": self.ultrack_division_spin.value(),
                "power": self.ultrack_power_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if "save_source" in state:
            self.save_source_check.setChecked(state["save_source"])
        if "cellprob" in state:
            cp = state["cellprob"]
            if "min" in cp: self.cp_min_spin.setValue(cp["min"])
            if "max" in cp: self.cp_max_spin.setValue(cp["max"])
            if "step" in cp: self.cp_step_spin.setValue(cp["step"])
            if "gamma_min" in cp: self.cp_gamma_min_spin.setValue(cp["gamma_min"])
            if "gamma_max" in cp: self.cp_gamma_max_spin.setValue(cp["gamma_max"])
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
            if "min_area" in dbg: self.db_gen_min_area_spin.setValue(dbg["min_area"])
            if "max_area" in dbg: self.db_gen_max_area_spin.setValue(dbg["max_area"])
            if "fg_threshold" in dbg: self.db_gen_fg_thr_spin.setValue(dbg["fg_threshold"])
            if "min_frontier" in dbg: self.db_gen_min_frontier_spin.setValue(dbg["min_frontier"])
            if "ws_hierarchy" in dbg:
                idx = self.db_gen_ws_hierarchy_combo.findText(dbg["ws_hierarchy"])
                if idx >= 0: self.db_gen_ws_hierarchy_combo.setCurrentIndex(idx)
            if "max_distance" in dbg: self.db_gen_max_dist_spin.setValue(dbg["max_distance"])
            if "max_neighbors" in dbg: self.db_gen_max_neighbors_spin.setValue(dbg["max_neighbors"])
            if "linking_mode" in dbg:
                idx = self.db_gen_linking_mode_combo.findText(dbg["linking_mode"])
                if idx >= 0: self.db_gen_linking_mode_combo.setCurrentIndex(idx)
            if "iou_weight" in dbg: self.db_gen_iou_weight_spin.setValue(dbg["iou_weight"])
            if "quality_weight" in dbg: self.db_gen_quality_weight_spin.setValue(dbg["quality_weight"])
            if "quality_exponent" in dbg: self.db_gen_quality_exp_spin.setValue(dbg["quality_exponent"])
            if "circularity_weight" in dbg: self.db_gen_circularity_weight_spin.setValue(dbg["circularity_weight"])
            if "power" in dbg: self.db_gen_power_spin.setValue(dbg["power"])
            if "n_workers" in dbg: self.db_gen_n_workers_spin.setValue(dbg["n_workers"])
            if "use_validated" in dbg: self.db_gen_use_validated_check.setChecked(dbg["use_validated"])
            if "seed_weight" in dbg: self.ultrack_seed_weight_spin.setValue(dbg["seed_weight"])
            if "seed_sigma_space" in dbg: self.ultrack_seed_space_spin.setValue(dbg["seed_sigma_space"])
            if "seed_tau_time" in dbg: self.ultrack_seed_time_spin.setValue(dbg["seed_tau_time"])
            if "seed_max_dt" in dbg: self.ultrack_seed_window_spin.setValue(dbg["seed_max_dt"])
        if "extend" in state:
            ext = state["extend"]
            if "max_distance" in ext: self.extend_max_dist_spin.setValue(ext["max_distance"])
            if "area_weight" in ext: self.extend_area_weight_spin.setValue(ext["area_weight"])
            if "iou_weight" in ext: self.extend_iou_weight_spin.setValue(ext["iou_weight"])
            if "distance_weight" in ext: self.extend_distance_weight_spin.setValue(ext["distance_weight"])
            if "overlap_penalty" in ext: self.extend_overlap_penalty_spin.setValue(ext["overlap_penalty"])
            if "greedy_overwrite" in ext: self.extend_greedy_overwrite_check.setChecked(ext["greedy_overwrite"])
        if "ultrack" in state:
            ul = state["ultrack"]
            if "min_area" in ul and (
                "db_generation" not in state or "min_area" not in state["db_generation"]
            ):
                self.db_gen_min_area_spin.setValue(ul["min_area"])
            if "max_partitions" in ul: self.ultrack_max_partitions_spin.setValue(ul["max_partitions"])
            if "n_frames" in ul: self.ultrack_n_frames_spin.setValue(ul["n_frames"])
            if "max_distance" in ul and (
                "db_generation" not in state or "max_distance" not in state["db_generation"]
            ):
                self.db_gen_max_dist_spin.setValue(ul["max_distance"])
            if "linking_mode" in ul and (
                "db_generation" not in state or "linking_mode" not in state["db_generation"]
            ):
                idx = self.db_gen_linking_mode_combo.findText(ul["linking_mode"])
                if idx >= 0: self.db_gen_linking_mode_combo.setCurrentIndex(idx)
            if "iou_weight" in ul and (
                "db_generation" not in state or "iou_weight" not in state["db_generation"]
            ):
                self.db_gen_iou_weight_spin.setValue(ul["iou_weight"])
            if "appear_weight" in ul: self.ultrack_appear_spin.setValue(ul["appear_weight"])
            if "disappear_weight" in ul: self.ultrack_disappear_spin.setValue(ul["disappear_weight"])
            if "division_weight" in ul: self.ultrack_division_spin.setValue(ul["division_weight"])
            if "max_neighbors" in ul and (
                "db_generation" not in state or "max_neighbors" not in state["db_generation"]
            ):
                self.db_gen_max_neighbors_spin.setValue(ul["max_neighbors"])
            if "power" in ul: self.ultrack_power_spin.setValue(ul["power"])
            if "resolve_from_validated" in ul and (
                "db_generation" not in state or "use_validated" not in state["db_generation"]
            ):
                self.db_gen_use_validated_check.setChecked(ul["resolve_from_validated"])
            if "quality_exponent" in ul and (
                "db_generation" not in state or "quality_exponent" not in state["db_generation"]
            ):
                self.db_gen_quality_exp_spin.setValue(ul["quality_exponent"])
            if "seed_weight" in ul and (
                "db_generation" not in state or "seed_weight" not in state["db_generation"]
            ):
                self.ultrack_seed_weight_spin.setValue(ul["seed_weight"])
            if "seed_sigma_space" in ul and (
                "db_generation" not in state or "seed_sigma_space" not in state["db_generation"]
            ):
                self.ultrack_seed_space_spin.setValue(ul["seed_sigma_space"])
            if "seed_tau_time" in ul and (
                "db_generation" not in state or "seed_tau_time" not in state["db_generation"]
            ):
                self.ultrack_seed_time_spin.setValue(ul["seed_tau_time"])
            if "seed_max_dt" in ul and (
                "db_generation" not in state or "seed_max_dt" not in state["db_generation"]
            ):
                self.ultrack_seed_window_spin.setValue(ul["seed_max_dt"])

    def set_selection_callback(self, fn) -> None:
        self.correction_widget.set_selection_callback(fn)

    def select_matching_nucleus_label(
        self, t: int, source_label: int,
        *, source_labels: np.ndarray | None = None,
    ) -> None:
        if _TRACKED_LAYER not in self.viewer.layers:
            return
        if source_labels is None:
            if "Tracked: Cell" not in self.viewer.layers:
                return
            source_labels = np.asarray(self.viewer.layers["Tracked: Cell"].data)
        target = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)
        matched = best_overlapping_label(target, source_labels, t, source_label)
        self.correction_widget.select_label(t, matched, notify=False)

    # ================================================================
    # Status / progress / button helpers
    # ================================================================
    def _status(self, msg: str) -> None:
        self.pipeline_status_lbl.setText(msg)
        self.pipeline_status_lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    def _correction_status(self, msg: str) -> None:
        self.correction_status_lbl.setText(msg)
        self.correction_status_lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    def _progress(self, done: int, total: int, msg: str) -> None:
        self.pipeline_progress_bar.setVisible(True)
        self.pipeline_progress_bar.setRange(0, total)
        self.pipeline_progress_bar.setValue(done)
        self._status(msg)

    def _on_progress(self, data) -> None:
        if isinstance(data, tuple):
            self._progress(*data)
        else:
            self._status(str(data))

    def _clear_progress(self) -> None:
        self.pipeline_progress_bar.setValue(0)
        self.pipeline_progress_bar.setVisible(False)

    def _set_pipeline_buttons_enabled(self, enabled: bool) -> None:
        for btn in (
            self.preview_contour_btn,
            self.build_btn,
            self.preview_contour_filter_btn,
            self.run_contour_filter_btn,
            self.run_db_gen_btn,
            self.run_ultrack_btn,
        ):
            btn.setEnabled(enabled)
        self.cancel_btn.setEnabled(not enabled)

    # ================================================================
    # Viewer helpers
    # ================================================================
    def _current_t(self) -> int:
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) >= 1 else 0

    def _update_tracked_display(
        self, labels: np.ndarray, t: int | None = None,
    ) -> None:
        if _TRACKED_LAYER in self.viewer.layers and t is not None:
            layer = self.viewer.layers[_TRACKED_LAYER]
            if layer.data.ndim == 3:
                if t < layer.data.shape[0]:
                    new_data = layer.data.copy()
                    new_data[t] = labels
                    layer.data = new_data
                    return
                new_data = np.concatenate(
                    [layer.data, labels[np.newaxis].astype(layer.data.dtype)], axis=0,
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

    @staticmethod
    def _frame_view_2d(arr: np.ndarray, t: int) -> np.ndarray | None:
        if arr.ndim < 3 or t < 0 or t >= arr.shape[0]:
            return None
        v = arr[t]
        while v.ndim > 2:
            if v.shape[0] != 1:
                return None
            v = v[0]
        return v

    def _current_cell_ids(self, t: int) -> set[int]:
        if _TRACKED_LAYER not in self.viewer.layers:
            return set()
        layer = self.viewer.layers[_TRACKED_LAYER]
        frame = self._frame_view_2d(layer.data, t)
        if frame is None:
            return set()
        return set(int(v) for v in np.unique(frame)) - {0}

    # ================================================================
    # 1. Contour Maps
    # ================================================================
    def _cp_gammas(self) -> list[float]:
        gmin = self.cp_gamma_min_spin.value()
        gmax = self.cp_gamma_max_spin.value()
        gstep = self.cp_gamma_step_spin.value()
        return list(np.arange(gmin, gmax + gstep / 2, gstep))

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

        boundary_sum = fg_sum = None
        for g_idx, g in enumerate(gammas):
            cb = None
            if mask_callback is not None:
                def cb(masks, i_thresh, *, _gi=g_idx):
                    mask_callback(masks, _gi, i_thresh)
            b, fg = build_consensus_boundary(
                prob_3d, dp_3d, thresholds,
                gamma=g, flow_threshold=flow_threshold, mask_callback=cb,
            )
            if boundary_sum is None:
                boundary_sum, fg_sum = b.copy(), fg.copy()
            else:
                boundary_sum += b; fg_sum += fg
        n = len(gammas)
        return boundary_sum / n, fg_sum / n

    def _on_build_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._status("No project open."); return
        prob_path = self._prob_path()
        dp_path = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._status(f"Missing: {prob_path}"); return
        if dp_path is None or not dp_path.exists():
            self._status(f"Missing: {dp_path}"); return

        thresholds = list(np.arange(
            self.cp_min_spin.value(),
            self.cp_max_spin.value() + self.cp_step_spin.value() / 2,
            self.cp_step_spin.value(),
        ))
        gammas = self._cp_gammas()
        contour_path = self._contour_maps_path()
        score_path = self._foreground_scores_path()
        mask_path = self._foreground_masks_path()
        fg_threshold = self.contour_fg_threshold_spin.value()
        flow_threshold = self.contour_flow_threshold_spin.value()
        save_source = self.save_source_check.isChecked()
        pos_dir = self._pos_dir
        build_fn = self._build_consensus_boundary_averaged
        if contour_path is None or score_path is None or mask_path is None:
            self._status("No project open."); return

        def _done(pos_dir_result):
            self._contour_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            self._files_widget.refresh(pos_dir_result)
            self._status("Contour maps and foreground masks built.")

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": self._on_contour_worker_error,
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            dp_stack = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
            if prob_stack.ndim == 3: prob_stack = prob_stack[np.newaxis]
            if dp_stack.ndim == 4: dp_stack = dp_stack[np.newaxis]

            n_t = prob_stack.shape[0]
            contour_frames: list[np.ndarray] = []
            fg_score_frames: list[np.ndarray] = []
            fg_mask_frames: list[np.ndarray] = []
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
                boundary, fg_score = build_fn(
                    prob_stack[t], dp_stack[t], thresholds, gammas,
                    flow_threshold=flow_threshold, mask_callback=mask_cb,
                )
                contour_frames.append(boundary.astype(np.float32, copy=False))
                fg_score = fg_score.astype(np.float32, copy=False)
                fg_score_frames.append(fg_score)
                fg_mask_frames.append((fg_score >= fg_threshold).astype(np.uint8))

            yield (n_t, n_t, "Saving…")
            contour_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(contour_path), np.stack(contour_frames), compression="zlib")
            tifffile.imwrite(str(score_path), np.stack(fg_score_frames), compression="zlib")
            tifffile.imwrite(str(mask_path), np.stack(fg_mask_frames), compression="zlib")
            return pos_dir

        gamma_desc = (
            f"γ={gammas[0]:.2f}" if len(gammas) == 1
            else f"γ={gammas[0]:.2f}–{gammas[-1]:.2f} ({len(gammas)} steps)"
        )
        self._status(
            f"Building contour maps ({len(thresholds)} cellprob thresholds, {gamma_desc})…"
        )
        self._set_pipeline_buttons_enabled(False)
        self._contour_worker = _worker()

    def _on_preview_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._status("No project open."); return
        prob_path = self._prob_path()
        dp_path = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._status(f"Missing: {prob_path}"); return
        if dp_path is None or not dp_path.exists():
            self._status(f"Missing: {dp_path}"); return

        t_frame = self._current_t()
        thresholds = list(np.arange(
            self.cp_min_spin.value(),
            self.cp_max_spin.value() + self.cp_step_spin.value() / 2,
            self.cp_step_spin.value(),
        ))
        gammas = self._cp_gammas()
        flow_threshold = self.contour_flow_threshold_spin.value()
        build_fn = self._build_consensus_boundary_averaged

        def _done(result):
            self._contour_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            boundary, foreground, cellprob_zavg, t_idx = result
            data = np.zeros((cellprob_zavg.shape[0],) + boundary.shape, dtype=boundary.dtype)
            data[t_idx] = boundary
            fg_score_data = np.zeros(
                (cellprob_zavg.shape[0],) + foreground.shape, dtype=np.float32,
            )
            fg_score_data[t_idx] = foreground
            fg_mask_data = (
                fg_score_data >= self.contour_fg_threshold_spin.value()
            ).astype(np.uint8)
            if _CELLPROB_LAYER in self.viewer.layers:
                self.viewer.layers[_CELLPROB_LAYER].data = cellprob_zavg
            else:
                self.viewer.add_image(
                    cellprob_zavg, name=_CELLPROB_LAYER,
                    colormap="inferno", blending="additive", visible=True,
                )
            if _CONTOUR_LAYER in self.viewer.layers:
                self.viewer.layers[_CONTOUR_LAYER].data = data
            else:
                self.viewer.add_image(data, name=_CONTOUR_LAYER, colormap="magma", visible=True)
            if _FOREGROUND_SCORE_LAYER in self.viewer.layers:
                self.viewer.layers[_FOREGROUND_SCORE_LAYER].data = fg_score_data
            else:
                self.viewer.add_image(
                    fg_score_data, name=_FOREGROUND_SCORE_LAYER,
                    colormap="viridis", visible=True,
                )
            self._update_layer(_FOREGROUND_MASK_LAYER, fg_mask_data)
            self._set_viewer_frame(t_idx)
            self._status(
                f"Preview contour t={t_idx} — "
                f"{len(thresholds)} thresholds, {len(gammas)} gamma(s)"
            )

        @thread_worker(connect={
            "returned": _done, "errored": self._on_contour_worker_error,
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            dp_stack = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
            if prob_stack.ndim == 3: prob_stack = prob_stack[np.newaxis]
            if dp_stack.ndim == 4: dp_stack = dp_stack[np.newaxis]
            n_t = min(prob_stack.shape[0], dp_stack.shape[0])
            t_idx = min(max(t_frame, 0), n_t - 1)
            boundary, foreground = build_fn(
                prob_stack[t_idx], dp_stack[t_idx], thresholds, gammas,
                flow_threshold=flow_threshold,
            )
            return boundary, foreground, self._sigmoid_zavg(prob_stack), t_idx

        self._status(f"Previewing contour map for frame t={t_frame}…")
        self._set_pipeline_buttons_enabled(False)
        self._contour_worker = _worker()

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
            self.viewer.add_image(data, name=_CONTOUR_LAYER, colormap="magma", visible=True)

    def _on_preview_contour_filter(self) -> None:
        if self._pos_dir is None:
            self._status("No project open."); return
        contour_path = self._contour_maps_path()
        if contour_path is None or not contour_path.exists():
            self._status("Missing: contour_maps.tif — run Build Contours first."); return
        params = self._contour_filter_params_from_ui()

        def _done(filtered):
            self._contour_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            self._update_contour_image_layer(filtered)
            self._status("Previewed filtered contour maps.")

        @thread_worker(connect={
            "returned": _done, "errored": self._on_contour_worker_error,
        })
        def _worker():
            from cellflow.segmentation import compute_filtered_contour_maps
            contours = np.asarray(tifffile.imread(str(contour_path)), dtype=np.float32)
            return compute_filtered_contour_maps(contours, params)

        self._status("Previewing filtered contour maps…")
        self._set_pipeline_buttons_enabled(False)
        self._contour_worker = _worker()

    def _on_run_contour_filter(self) -> None:
        if self._pos_dir is None:
            self._status("No project open."); return
        contour_path = self._contour_maps_path()
        if contour_path is None or not contour_path.exists():
            self._status("Missing: contour_maps.tif — run Build Contours first."); return
        params = self._contour_filter_params_from_ui()
        pos_dir = self._pos_dir

        def _done(result):
            self._contour_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            _pos_dir, filtered = result
            self._files_widget.refresh(_pos_dir)
            self._update_contour_image_layer(filtered)
            self._status("Filtered contour maps written to contour_maps.tif.")

        @thread_worker(connect={
            "returned": _done, "errored": self._on_contour_worker_error,
        })
        def _worker():
            from cellflow.segmentation import compute_filtered_contour_maps
            contours = np.asarray(tifffile.imread(str(contour_path)), dtype=np.float32)
            filtered = compute_filtered_contour_maps(contours, params)
            tifffile.imwrite(
                str(contour_path),
                filtered.astype(np.float32, copy=False),
                compression="zlib", photometric="minisblack",
            )
            return pos_dir, filtered

        self._status("Filtering contour_maps.tif…")
        self._set_pipeline_buttons_enabled(False)
        self._contour_worker = _worker()

    def _on_contour_worker_error(self, exc: Exception) -> None:
        self._contour_worker = None
        self._set_pipeline_buttons_enabled(True)
        self._clear_progress()
        self._status(f"Error: {exc}")
        logger.exception("Contour worker error", exc_info=exc)

    # ================================================================
    # 2. DB Generation
    # ================================================================
    def _on_db_gen_mode_changed(self, mode: str) -> None:
        self.db_gen_iou_weight_spin.setEnabled(mode == "iou")

    def _set_resolve_prior_controls_enabled(self, _checked: bool | None = None) -> None:
        enabled = self.db_gen_use_validated_check.isChecked()
        for w in (
            self.ultrack_seed_weight_spin,
            self.ultrack_seed_space_spin,
            self.ultrack_seed_time_spin,
            self.ultrack_seed_window_spin,
        ):
            w.setEnabled(enabled)

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
            self._status("No project open."); return
        contour_path = self._contour_maps_path()
        fg_path = self._foreground_masks_path()
        nuc_zavg_path = self._nucleus_prob_zavg_path()
        if contour_path is None or not contour_path.exists():
            self._status("Missing: contour_maps.tif — run Build Contours first."); return
        if fg_path is None or not fg_path.exists():
            self._status("Missing: foreground_masks.tif — run Build Contours first."); return
        if nuc_zavg_path is None or not nuc_zavg_path.exists():
            self._status("Missing: nucleus_prob_zavg.tif — run Cellpose first."); return
        if _ultrack_segment is None:
            self._status("ultrack not installed — activate the cellflow conda environment."); return

        cfg = self._db_gen_config_from_controls()
        working_dir = self._ultrack_workdir()
        pos_dir = self._pos_dir
        use_validated = self.db_gen_use_validated_check.isChecked()
        validated_tracks: dict[int, set[int]] | None = None
        tracked_labels: np.ndarray | None = None
        if use_validated:
            validated_tracks = read_validated_tracks(pos_dir)
            if not validated_tracks:
                self._status("No validated tracks found — validate some cells first (press V)."); return
            if _TRACKED_LAYER not in self.viewer.layers:
                self._status("No tracked layer loaded for validated DB generation."); return
            tracked_labels = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)

        self.pipeline_progress_bar.setRange(0, 0)
        self.pipeline_progress_bar.setVisible(True)
        self._status("Starting DB generation…")
        self._set_pipeline_buttons_enabled(False)

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": self._on_db_gen_done,
            "errored": self._on_db_gen_worker_error,
        })
        def _worker():
            import queue as _queue
            import threading

            msg_queue: _queue.SimpleQueue = _queue.SimpleQueue()
            result_holder: list = []
            exc_holder: list = []

            def _progress_cb(msg: str) -> None:
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
                            progress_cb=_progress_cb,
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

        self._db_gen_worker = _worker()

    def _on_db_gen_done(self, pos_dir: Path) -> None:
        self._db_gen_worker = None
        self._set_pipeline_buttons_enabled(True)
        self._clear_progress()
        self._status("DB generation complete.")
        self._files_widget.refresh(pos_dir)
        self._refresh_ultrack_db_browser()

    def _on_db_gen_worker_error(self, exc: Exception) -> None:
        self._db_gen_worker = None
        self._set_pipeline_buttons_enabled(True)
        self._clear_progress()
        self._status(f"Error: {exc}")
        logger.exception("DB generation worker error", exc_info=exc)

    # ================================================================
    # 3. Ultrack Tracking
    # ================================================================
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
        )

    def _on_run_ultrack(self) -> None:
        if self._pos_dir is None:
            self._status("No project open."); return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._status("data.db not found — run DB Generation first."); return
        working_dir = self._ultrack_workdir()
        tracked_path = self._tracked_path()

        cfg = self._ultrack_config_from_controls()
        needs_validated_export = database_has_annotations(working_dir)
        validated_tracks = None
        tracked_labels = None
        if needs_validated_export:
            validated_tracks = read_validated_tracks(self._pos_dir)
            if not validated_tracks:
                self._status(
                    "Annotated data.db requires validated tracks for ID-preserving export."
                ); return
            if _TRACKED_LAYER not in self.viewer.layers:
                self._status(
                    "Annotated data.db requires the current tracked layer for export."
                ); return
            tracked_labels = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)

        self.pipeline_progress_bar.setRange(0, 100)
        self.pipeline_progress_bar.setVisible(True)
        self.pipeline_progress_bar.setValue(0)
        self._status("Starting Ultrack solve…")
        self._set_pipeline_buttons_enabled(False)

        @thread_worker(connect={
            "yielded": self._on_ultrack_progress,
            "returned": self._on_run_ultrack_done,
            "errored": self._on_ultrack_worker_error,
        })
        def _worker():
            for step, total, label in run_solve(working_dir, cfg, overwrite=True):
                yield (step, total, f"[solve] {label}")
            yield "Exporting tracked labels…"
            return export_tracked_labels(
                working_dir, cfg, tracked_path,
                validated_tracks=validated_tracks,
                tracked_labels=tracked_labels,
            )

        self._ultrack_worker = _worker()

    def _on_ultrack_progress(self, data) -> None:
        if isinstance(data, tuple):
            step, total, msg = data
            self._status(msg)
            if total > 0:
                self.pipeline_progress_bar.setRange(0, total)
                self.pipeline_progress_bar.setValue(step)
        else:
            self._status(str(data))

    def _on_run_ultrack_done(self, labels: np.ndarray | None) -> None:
        self._ultrack_worker = None
        self._set_pipeline_buttons_enabled(True)
        self._clear_progress()
        if labels is None:
            self._status("Ultrack tracking failed (no output)."); return
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        nt = labels.shape[0]
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = labels
        else:
            self.viewer.add_labels(labels, name=_TRACKED_LAYER)
        layer = self.viewer.layers[_TRACKED_LAYER]
        self.correction_widget.activate_layer(layer)
        self._files_widget.refresh(self._pos_dir)
        self._status(f"Tracking done: {nt} frame(s). Unsaved.")

    def _on_ultrack_worker_error(self, exc: Exception) -> None:
        self._ultrack_worker = None
        self._set_pipeline_buttons_enabled(True)
        self._clear_progress()
        self._status(f"Error: {exc}")
        logger.exception("Ultrack worker error", exc_info=exc)

    # ================================================================
    # Cancel
    # ================================================================
    def _on_cancel(self) -> None:
        cancelled = False
        for attr in ("_contour_worker", "_db_gen_worker", "_ultrack_worker"):
            worker = getattr(self, attr, None)
            if worker is not None:
                worker.quit()
                setattr(self, attr, None)
                cancelled = True
        self._set_pipeline_buttons_enabled(True)
        self._clear_progress()
        self._status("Cancelled." if cancelled else "Nothing running.")

    # ================================================================
    # Ultrack DB Browser
    # ================================================================
    def _set_ultrack_db_status(self, msg: str) -> None:
        self.ultrack_db_section_status_lbl.setText(msg)
        self.ultrack_db_section_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _on_ultrack_db_browser_param_changed(self, *_args) -> None:
        self._ultrack_db_preview_cache.clear()

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
        for name in (_ULTRACK_DB_PREVIEW_LAYER, _ULTRACK_DB_ANNOTATION_LAYER):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        if _ULTRACK_DB_SELECTION_LAYER in self.viewer.layers:
            self.viewer.layers.remove(_ULTRACK_DB_SELECTION_LAYER)
        self.ultrack_db_info_lbl.setText("—")
        self._set_ultrack_db_status("")

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
        return frames[len(frames) // 2] if frames else None

    def _refresh_ultrack_db_browser(self) -> None:
        if not self._ultrack_db_browser_active:
            return
        self.ultrack_db_info_lbl.setText("—")
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._set_ultrack_db_status("data.db not found — run DB Generation first.")
            return
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
                str(db_path.resolve()), mtime_ns, frame, slider_int, state,
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
                    db_path, frame, labels, status, prob_dict,
                    label_to_node_id, node_id_to_label,
                )
            self._ultrack_db_preview_labels = labels.astype(np.uint32, copy=False)
            self._update_ultrack_db_preview_layer(
                self._ultrack_db_preview_labels, prob_dict, alpha_dict,
            )
            self._update_ultrack_db_annotation_layer(
                self._ultrack_db_preview_labels, label_to_node_id, node_annotations,
            )
            self._install_ultrack_db_preview_selector()
            if not self.ultrack_db_connected_focus_check.isChecked():
                status = self._refresh_ultrack_db_selection_highlight(
                    self._ultrack_db_preview_labels, status, node_id_to_label, frame,
                )
            self._set_ultrack_db_status(status)
        except Exception as e:
            self._set_ultrack_db_status(f"DB read error: {e}")
            logger.warning("DB browser error: %s", e)

    @staticmethod
    def _normalize_ultrack_db_preview(cached):
        if len(cached) == 2:
            labels, status = cached
            return labels, status, {}, {}, {}, {}
        if len(cached) == 3:
            labels, status, prob_dict = cached
            return labels, status, prob_dict, {}, {}, {}
        if len(cached) == 5:
            labels, status, prob_dict, l2n, n2l = cached
            return labels, status, prob_dict, l2n, n2l, {}
        labels, status, prob_dict, l2n, n2l, annots = cached
        return labels, status, prob_dict, l2n, n2l, annots

    def _update_ultrack_db_preview_layer(self, labels, prob_dict, alpha_dict=None):
        if alpha_dict:
            data = self._ultrack_db_alpha_rgba(labels, alpha_dict)
            self._update_image_layer(_ULTRACK_DB_PREVIEW_LAYER, data, rgb=True)
            return
        if self.ultrack_db_prob_alpha_check.isChecked() and prob_dict:
            data = self._ultrack_db_probability_rgba(labels, prob_dict)
            self._update_image_layer(_ULTRACK_DB_PREVIEW_LAYER, data, rgb=True)
            return
        self._update_labels_layer(_ULTRACK_DB_PREVIEW_LAYER, labels)

    def _update_ultrack_db_annotation_layer(self, labels, label_to_node_id, node_annotations):
        overlay = np.zeros_like(labels, dtype=np.uint8)
        for lid, nid in label_to_node_id.items():
            annot = node_annotations.get(int(nid), "UNKNOWN")
            if annot == "REAL":
                overlay[labels == int(lid)] = 1
            elif annot == "FAKE":
                overlay[labels == int(lid)] = 2
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
    def _ultrack_db_probability_rgba(labels, prob_dict):
        from napari.utils.colormaps import label_colormap
        rgba = np.zeros(labels.shape + (4,), dtype=np.float32)
        if labels.size == 0 or not prob_dict:
            return rgba
        probs = [float(v) for v in prob_dict.values()]
        min_p, max_p = min(probs), max(probs)
        denom = max(max_p - min_p, 1e-9)
        cmap = label_colormap(max(prob_dict.keys()) + 1)
        for lid, prob in prob_dict.items():
            mask = labels == int(lid)
            if not np.any(mask): continue
            color = np.asarray(cmap.map(int(lid)), dtype=np.float32)
            alpha = 0.15 + 0.85 * (float(prob) - min_p) / denom
            color[3] = float(np.clip(alpha, 0.15, 1.0))
            rgba[mask] = color
        return rgba

    @staticmethod
    def _ultrack_db_alpha_rgba(labels, alpha_dict):
        from napari.utils.colormaps import label_colormap
        rgba = np.zeros(labels.shape + (4,), dtype=np.float32)
        if labels.size == 0 or not alpha_dict:
            return rgba
        cmap = label_colormap(max(alpha_dict.keys()) + 1)
        for lid, alpha in alpha_dict.items():
            mask = labels == int(lid)
            if not np.any(mask): continue
            color = np.asarray(cmap.map(int(lid)), dtype=np.float32)
            color[3] = float(np.clip(alpha, 0.0, 1.0))
            rgba[mask] = color
        return rgba

    def _install_ultrack_db_preview_selector(self) -> None:
        if _ULTRACK_DB_PREVIEW_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[_ULTRACK_DB_PREVIEW_LAYER]
        self._remove_ultrack_db_preview_selector()

        def _on_drag(_layer, event):
            if getattr(event, "type", None) != "mouse_press": return
            if getattr(event, "button", None) != 1: return
            if getattr(event, "modifiers", set()): return
            labels = self._ultrack_db_preview_labels
            if labels is None or labels.size == 0: return
            pos = _layer.world_to_data(event.position)
            y, x = int(round(float(pos[-2]))), int(round(float(pos[-1])))
            if y < 0 or x < 0 or y >= labels.shape[-2] or x >= labels.shape[-1]: return
            display_label = int(labels[y, x])
            if display_label == 0: return
            self._select_ultrack_db_preview_label(display_label, frame=self._current_t())
            yield

        layer.mouse_drag_callbacks.append(_on_drag)
        self._ultrack_db_preview_mouse_callback = _on_drag

    def _remove_ultrack_db_preview_selector(self) -> None:
        cb = self._ultrack_db_preview_mouse_callback
        if cb is None or _ULTRACK_DB_PREVIEW_LAYER not in self.viewer.layers:
            self._ultrack_db_preview_mouse_callback = None
            return
        layer = self.viewer.layers[_ULTRACK_DB_PREVIEW_LAYER]
        try:
            layer.mouse_drag_callbacks.remove(cb)
        except ValueError:
            pass
        self._ultrack_db_preview_mouse_callback = None

    def _select_ultrack_db_preview_label(self, display_label, *, frame=None):
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
        self._set_ultrack_db_status(f"Selected node {node_id}{annot_suffix} at t={selected_frame}.")
        if self.ultrack_db_connected_focus_check.isChecked():
            self._refresh_ultrack_db_browser()

    def _refresh_ultrack_db_selection_highlight(self, labels, status, node_id_to_label, frame):
        sel = self._ultrack_db_selected_node_id
        if sel is None:
            self._clear_ultrack_db_highlight()
            return status
        dl = node_id_to_label.get(int(sel))
        if dl is None:
            self._clear_ultrack_db_highlight()
            annot = self._query_ultrack_db_node_annotation_for_status(node_id_to_label, sel)
            if annot in {"REAL", "FAKE"}:
                return (
                    f"{status} Selected node {sel} [{annot}] is hidden "
                    f"by annotation filter at frame {frame}."
                )
            return (
                f"{status} Selected node {sel} is hidden "
                f"at frame {frame} and the current hierarchy threshold."
            )
        self._update_ultrack_db_highlight(labels, int(dl))
        return status

    def _query_ultrack_db_node_annotation_for_status(self, node_id_to_label, selected_node_id):
        return self._ultrack_db_node_annotations.get(int(selected_node_id), "UNKNOWN")

    def _get_ultrack_db_highlight_layer(self):
        if _ULTRACK_DB_SELECTION_LAYER in self.viewer.layers:
            return self.viewer.layers[_ULTRACK_DB_SELECTION_LAYER]
        layer = self.viewer.add_shapes(
            name=_ULTRACK_DB_SELECTION_LAYER, ndim=2,
            edge_color="cyan", edge_width=2, face_color="transparent",
        )
        layer.visible = False
        return layer

    def _update_ultrack_db_highlight(self, labels, display_label):
        layer = self._get_ultrack_db_highlight_layer()
        if labels is None or display_label == 0:
            layer.data = []; layer.visible = False; return
        mask = (labels == int(display_label)).astype(np.uint8)
        if not np.any(mask):
            layer.data = []; layer.visible = False; return
        from skimage.measure import find_contours
        contours = find_contours(mask, level=0.5)
        if not contours:
            layer.data = []; layer.visible = False; return
        layer.data = [max(contours, key=len)]
        layer.shape_type = ["polygon"]
        layer.visible = True

    def _clear_ultrack_db_highlight(self) -> None:
        if _ULTRACK_DB_SELECTION_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[_ULTRACK_DB_SELECTION_LAYER]
        layer.data = []; layer.visible = False

    def _query_ultrack_db_connected_nodes(self, db_path, selected_node_id):
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
                    ).all()
                )
                for src, tgt, w in rows:
                    wf = float(w if w is not None else 1.0)
                    if int(tgt) == int(selected_node_id):
                        si = int(src)
                        predecessors[si] = predecessors.get(si, 1.0) * wf
                    if int(src) == int(selected_node_id):
                        ti = int(tgt)
                        successors[ti] = successors.get(ti, 1.0) * wf
        finally:
            engine.dispose()
        return predecessors, successors

    def _render_ultrack_db_connected_focus(
        self, db_path, frame, labels, status, prob_dict, label_to_node_id, node_id_to_label,
    ):
        sel_nid = self._ultrack_db_selected_node_id
        sel_frame = self._ultrack_db_selected_frame
        if sel_nid is None or sel_frame is None:
            self._clear_ultrack_db_highlight()
            return labels, f"{status} Click a DB preview node to focus links.", {}
        predecessors, successors = self._query_ultrack_db_connected_nodes(db_path, sel_nid)
        if frame == sel_frame:
            relation = "selected"
            allowed = {sel_nid: 1.0}
            if int(sel_nid) not in node_id_to_label:
                self._clear_ultrack_db_highlight()
                empty = np.zeros_like(labels, dtype=np.uint32)
                annot = self._ultrack_db_node_annotations.get(int(sel_nid), "UNKNOWN")
                suf = "" if annot == "UNKNOWN" else f" [{annot}]"
                return empty, (
                    f"Selected node {sel_nid}{suf} at t={sel_frame} is hidden."
                ), {}
        elif frame == sel_frame - 1:
            relation = "t-1"; allowed = predecessors
        elif frame == sel_frame + 1:
            relation = "t+1"; allowed = successors
        else:
            self._clear_ultrack_db_highlight()
            return np.zeros_like(labels, dtype=np.uint32), (
                f"Selected node {sel_nid} at t={sel_frame} | frame {frame}: outside focus."
            ), {}

        focused = np.zeros_like(labels, dtype=np.uint32)
        alpha_dict: dict[int, float] = {}
        for lid, nid in label_to_node_id.items():
            li, ni = int(lid), int(nid)
            if ni not in allowed: continue
            focused[labels == li] = li
            alpha_on = (
                self.ultrack_db_edge_alpha_check.isChecked()
                or self.ultrack_db_prob_alpha_check.isChecked()
            )
            if alpha_on:
                alpha_dict[li] = (
                    1.0 if ni == sel_nid
                    else self._ultrack_db_connected_alpha(li, float(allowed[ni]), prob_dict)
                )

        sel_label = node_id_to_label.get(int(sel_nid))
        if frame == sel_frame and sel_label is not None:
            self._update_ultrack_db_highlight(focused, int(sel_label))
        else:
            self._clear_ultrack_db_highlight()

        edge_vals = [
            float(v) for nid, v in allowed.items()
            if nid in node_id_to_label and nid != sel_nid
        ]
        edge_summary = (
            f" | edge range {min(edge_vals):.2f}-{max(edge_vals):.2f}" if edge_vals else ""
        )
        count = int(np.unique(focused[focused != 0]).size)
        annot = self._ultrack_db_node_annotations.get(int(sel_nid), "UNKNOWN")
        suf = "" if annot == "UNKNOWN" else f" [{annot}]"
        return focused, (
            f"Selected node {sel_nid}{suf} at t={sel_frame} | "
            f"{relation}: {count} connected{edge_summary}"
        ), alpha_dict

    def _ultrack_db_connected_alpha(self, label_id, edge_weight, prob_dict):
        alpha = 1.0
        if self.ultrack_db_edge_alpha_check.isChecked():
            alpha *= float(edge_weight)
        if self.ultrack_db_prob_alpha_check.isChecked() and prob_dict:
            probs = [float(v) for v in prob_dict.values()]
            min_p, max_p = min(probs), max(probs)
            denom = max(max_p - min_p, 1e-9)
            prob = float(prob_dict.get(int(label_id), 1.0))
            alpha *= 0.15 + 0.85 * (prob - min_p) / denom
        return float(np.clip(alpha, 0.05, 1.0))

    def _ultrack_db_summary_text(self, db_path, frame):
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
                    .filter(NodeDB.node_annot == VarAnnotation.REAL).scalar() or 0
                )
                n_fake = int(
                    session.query(func.count(NodeDB.id))
                    .filter(NodeDB.node_annot == VarAnnotation.FAKE).scalar() or 0
                )
                frame_nodes = session.query(NodeDB).filter(NodeDB.t == frame).all()
                selected = sum(1 for n in frame_nodes if getattr(n, "selected", False))
                node_ids = [int(n.id) for n in frame_nodes]
                outgoing = incoming = overlaps = 0
                if node_ids:
                    outgoing = int(
                        session.query(func.count(LinkDB.source_id))
                        .filter(LinkDB.source_id.in_(node_ids)).scalar() or 0
                    )
                    incoming = int(
                        session.query(func.count(LinkDB.target_id))
                        .filter(LinkDB.target_id.in_(node_ids)).scalar() or 0
                    )
                    if OverlapDB is not None:
                        try:
                            overlaps = int(
                                session.query(func.count(OverlapDB.node_id))
                                .filter(
                                    OverlapDB.node_id.in_(node_ids)
                                    | OverlapDB.ancestor_id.in_(node_ids)
                                ).scalar() or 0
                            )
                        except Exception:
                            overlaps = 0
            return (
                f"{n_nodes} nodes | {n_links} links | REAL {n_real} | FAKE {n_fake} | "
                f"frame {frame}: {len(node_ids)} nodes, {selected} selected, "
                f"{incoming} in/{outgoing} out links, {overlaps} overlaps"
            )
        finally:
            engine.dispose()

    def _query_distinct_heights(self, db_path, mtime_ns):
        key = (str(db_path.resolve()), mtime_ns)
        cached = self._ultrack_db_height_values_cache.get(key)
        if cached is not None: return cached
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                heights = tuple(
                    float(r[0]) for r in
                    session.query(NodeDB.height).distinct().order_by(NodeDB.height).all()
                    if r[0] is not None
                )
        finally:
            engine.dispose()
        self._ultrack_db_height_values_cache[key] = heights
        return heights

    def _query_hierarchy_cut_states(self, db_path, mtime_ns, frame):
        key = (str(db_path.resolve()), mtime_ns, frame)
        cached = self._ultrack_db_cut_state_cache.get(key)
        if cached is not None: return cached
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
                    (int(nid), int(pid), float(h))
                    for nid, pid, h in session.query(
                        NodeDB.id, NodeDB.hier_parent_id, NodeDB.height
                    ).filter(NodeDB.t == frame).order_by(NodeDB.height, NodeDB.id).all()
                    if h is not None
                ]
        except Exception:
            heights = self._query_distinct_heights(db_path, mtime_ns)
            return tuple(_HierarchyCutState((), float(h)) for h in heights)
        finally:
            engine.dispose()
        if not rows:
            self._ultrack_db_cut_state_cache[key] = ()
            return ()

        node_ids = {nid for nid, _, _ in rows}
        heights_by_id = {nid: h for nid, _, h in rows}
        parent_by_id = {
            nid: pid for nid, pid, _ in rows
            if pid != NO_PARENT and pid in node_ids
        }
        children: dict[int, set[int]] = {}
        for cid, pid in parent_by_id.items():
            children.setdefault(pid, set()).add(cid)

        active = {nid for nid, _, _ in rows if nid not in children}
        if not active:
            active = set(node_ids)

        states: list[_HierarchyCutState] = []
        seen: set[tuple[int, ...]] = set()

        def _append():
            ordered = tuple(sorted(active, key=lambda n: (heights_by_id[n], n)))
            if ordered in seen: return
            seen.add(ordered)
            h = max((heights_by_id[n] for n in ordered), default=None)
            states.append(_HierarchyCutState(ordered, h))

        _append()
        while True:
            promotable = [
                pid for pid, cids in children.items()
                if pid not in active and cids and cids.issubset(active)
            ]
            if not promotable: break
            min_h = min(heights_by_id[pid] for pid in promotable)
            for pid in sorted(p for p in promotable if heights_by_id[p] == min_h):
                active.difference_update(children[pid])
                active.add(pid)
            _append()

        result = tuple(states)
        self._ultrack_db_cut_state_cache[key] = result
        return result

    def _configure_ultrack_db_hierarchy_slider(self, db_path, mtime_ns, frame):
        states = self._query_hierarchy_cut_states(db_path, mtime_ns, frame)
        maximum = max(len(states) - 1, 0)
        value = min(max(int(self.ultrack_db_hierarchy_slider.value()), 0), maximum)
        old = self.ultrack_db_hierarchy_slider.blockSignals(True)
        try:
            self.ultrack_db_hierarchy_slider.setRange(0, maximum)
            self.ultrack_db_hierarchy_slider.setValue(value)
        finally:
            self.ultrack_db_hierarchy_slider.blockSignals(old)
        if states:
            self._set_ultrack_db_height_label(value, states[value].height, len(states))
        else:
            self.ultrack_db_height_lbl.setText("—")
        return states

    def _set_ultrack_db_height_label(self, index, height, total):
        ht = "—" if height is None else f"{height:.2f}"
        self.ultrack_db_height_lbl.setText(f"i={index} h={ht} ({index + 1}/{total})")

    def _render_hierarchy_cut(self, db_path, frame, h_actual):
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session, aliased
        from ultrack.core.database import NodeDB
        from ultrack.utils.constants import NO_PARENT
        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                P = aliased(NodeDB); C = aliased(NodeDB)
                same_child = (
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
                    .where(~same_child)
                    .all()
                )
        finally:
            engine.dispose()
        return self._finalize_hierarchy_nodes(
            nodes, frame,
            empty_msg=f"No segments at this threshold for frame {frame}.",
            status_suffix=f"at h={h_actual:.2f}",
        )

    def _render_hierarchy_cut_state(self, db_path, frame, state):
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
        by_id = {int(n.id): n for n in rows}
        nodes = [by_id[nid] for nid in state.node_ids if nid in by_id]
        ht = "—" if state.height is None else f"{state.height:.2f}"
        return self._finalize_hierarchy_nodes(
            nodes, frame,
            empty_msg=f"No hierarchy state segments for frame {frame}.",
            status_suffix=f"at cut state h={ht}",
        )

    def _finalize_hierarchy_nodes(self, nodes, frame, *, empty_msg, status_suffix):
        if not nodes:
            return self._empty_ultrack_db_preview(), empty_msg, {}, {}, {}, {}
        show_val = self.ultrack_db_show_validated_check.isChecked()
        show_fake = self.ultrack_db_show_fake_check.isChecked()
        filtered, h_real, h_fake = [], 0, 0
        for n in nodes:
            a = self._ultrack_db_annotation_name(getattr(n, "node_annot", None))
            if a == "REAL" and not show_val:
                h_real += 1; continue
            if a == "FAKE" and not show_fake:
                h_fake += 1; continue
            filtered.append(n)
        if not filtered:
            return self._empty_ultrack_db_preview(), (
                f"Frame {frame}: annotation filters hid all {len(nodes)} segment(s)."
            ), {}, {}, {}, {}
        labels = self._paint_ultrack_db_nodes(filtered)
        prob_dict, l2n, n2l = self._ultrack_db_node_preview_metadata(filtered)
        annots = self._ultrack_db_node_annotation_metadata(filtered)
        hidden = ""
        if h_real or h_fake:
            hidden = f" Hidden: REAL {h_real}, FAKE {h_fake}."
        return labels, (
            f"Frame {frame}: {len(filtered)} segment(s) {status_suffix}.{hidden}"
        ), prob_dict, l2n, n2l, annots

    @staticmethod
    def _ultrack_db_annotation_name(value):
        if value is None: return "UNKNOWN"
        raw = getattr(value, "value", value)
        if raw is None: return "UNKNOWN"
        name = str(raw).split(".")[-1].upper()
        return name if name in {"REAL", "FAKE"} else "UNKNOWN"

    @staticmethod
    def _ultrack_db_node_preview_metadata(nodes):
        prob_dict, l2n, n2l = {}, {}, {}
        for label, node in enumerate(nodes, start=1):
            try:
                prob = float(node.node_prob if node.node_prob is not None else 1.0)
            except (TypeError, ValueError):
                prob = 1.0
            prob_dict[label] = prob
            try:
                nid = int(node.id)
            except (TypeError, ValueError):
                continue
            l2n[label] = nid; n2l[nid] = label
        return prob_dict, l2n, n2l

    @staticmethod
    def _ultrack_db_node_annotation_metadata(nodes):
        annots: dict[int, str] = {}
        for node in nodes:
            try:
                nid = int(node.id)
            except (TypeError, ValueError):
                continue
            annots[nid] = NucleusWorkflowWidget._ultrack_db_annotation_name(
                getattr(node, "node_annot", None)
            )
        return annots

    def _empty_ultrack_db_preview(self):
        return np.zeros(self._viewer_plane_shape(), dtype=np.uint32)

    def _viewer_plane_shape(self):
        for layer in self.viewer.layers:
            data = getattr(layer, "data", None)
            if isinstance(data, np.ndarray) and data.ndim >= 2:
                return tuple(int(v) for v in data.shape[-2:])
        return (1, 1)

    def _paint_ultrack_db_nodes(self, nodes):
        masks: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
        max_y = max_x = 0
        for label, node in enumerate(nodes, start=1):
            parsed = self._node_mask_and_bbox(node)
            if parsed is None: continue
            bbox, mask = parsed
            y0, x0, y1, x1 = bbox
            max_y = max(max_y, y1); max_x = max(max_x, x1)
            masks.append((label, bbox, mask))
        base_y, base_x = self._viewer_plane_shape()
        labels = np.zeros(
            (max(base_y, max_y, 1), max(base_x, max_x, 1)), dtype=np.uint32,
        )
        for label, (y0, x0, y1, x1), mask in masks:
            target = labels[y0:y1, x0:x1]
            if target.shape != mask.shape: continue
            target[mask.astype(bool)] = label
        return labels

    @staticmethod
    def _node_mask_and_bbox(node):
        try:
            node_obj = node.pickle
            if isinstance(node_obj, (bytes, memoryview)):
                node_obj = pickle.loads(bytes(node_obj))
            if node_obj is None: return None
        except Exception:
            return None
        if isinstance(node_obj, dict):
            bbox, mask = node_obj.get("bbox"), node_obj.get("mask")
        elif isinstance(node_obj, tuple) and len(node_obj) >= 2:
            bbox, mask = node_obj[0], node_obj[1]
        else:
            bbox = getattr(node_obj, "bbox", None)
            mask = getattr(node_obj, "mask", None)
        if bbox is None or mask is None: return None
        ba = np.asarray(bbox, dtype=int).ravel()
        if ba.size >= 6:
            y0, x0, y1, x1 = int(ba[1]), int(ba[2]), int(ba[4]), int(ba[5])
        elif ba.size >= 4:
            y0, x0, y1, x1 = (int(v) for v in ba[:4])
        else:
            return None
        ma = np.asarray(mask)
        if ma.ndim == 3 and ma.shape[0] == 1: ma = ma[0]
        elif ma.ndim > 2: ma = np.squeeze(ma)
        if ma.ndim != 2: return None
        if ma.shape != (y1 - y0, x1 - x0): return None
        return (y0, x0, y1, x1), ma.astype(bool, copy=False)

    # ================================================================
    # 4. Correction
    # ================================================================
    def _on_save_tracked(self) -> None:
        tracked_path = self._tracked_path()
        if tracked_path is None:
            self._correction_status("No project open."); return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._correction_status("No tracked layer to save."); return
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3:
            self._correction_status("Tracked layer is not a 3D stack."); return
        n = layer.data.shape[0]
        for t in range(n):
            write_tracked_frame(tracked_path, t, np.asarray(layer.data[t]))
        self._correction_status(f"Saved {n} frame(s) to {tracked_path.name}.")

    def _on_load_tracked(self) -> None:
        tracked_path = self._tracked_path()
        cell_zavg_path = self._cell_zavg_path()
        nuc_zavg_path = self._nucleus_zavg_path()
        if tracked_path is None or not tracked_path.exists():
            self._correction_status("No tracked labels file found."); return
        self._correction_status("Loading tracked labels…")

        @thread_worker(connect={
            "returned": self._on_load_tracked_done,
            "errored": self._on_correction_worker_error,
        })
        def _worker():
            stack = read_full_tracked_stack(tracked_path)
            cz = (
                np.asarray(tifffile.imread(str(cell_zavg_path)), dtype=np.float32)
                if cell_zavg_path and cell_zavg_path.exists() else None
            )
            nz = (
                np.asarray(tifffile.imread(str(nuc_zavg_path)), dtype=np.float32)
                if nuc_zavg_path and nuc_zavg_path.exists() else None
            )
            return stack, cz, nz

        _worker()

    def _on_load_tracked_done(self, result: tuple) -> None:
        stack, cell_zavg, nuc_zavg = result
        nt = stack.shape[0]
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = stack
        else:
            self.viewer.add_labels(stack, name=_TRACKED_LAYER)

        for zavg, name, cmap in (
            (cell_zavg, _CELL_ZAVG_LAYER, "gray"),
            (nuc_zavg, _NUC_ZAVG_LAYER, "bop orange"),
        ):
            if zavg is None: continue
            if zavg.ndim == 2:
                bcast = np.broadcast_to(zavg[np.newaxis], (nt,) + zavg.shape).copy()
            else:
                bcast = zavg
            if name in self.viewer.layers:
                self.viewer.layers[name].data = bcast
            else:
                self.viewer.add_image(bcast, name=name, colormap=cmap, blending="additive")

        self._correction_status(f"Loaded tracked stack {stack.shape} into napari.")
        layer = self.viewer.layers[_TRACKED_LAYER]
        self.correction_widget.activate_layer(layer)

    def _on_reassign_ids(self) -> None:
        if _TRACKED_LAYER not in self.viewer.layers:
            self._correction_status("No tracked layer loaded."); return
        stack = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)
        self._correction_status("Reassigning cell IDs…")

        @thread_worker(connect={
            "returned": self._on_reassign_ids_done,
            "errored": self._on_correction_worker_error,
        })
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
        self._correction_status(
            f"Reassigned {n_cells} cell IDs to range 1–{n_cells}. Unsaved."
        )

    def _on_extend_backward(self) -> None:
        self._on_extend(direction="backward")

    def _on_extend_forward(self) -> None:
        self._on_extend(direction="forward")

    def _on_extend(self, direction: str) -> None:
        if _TRACKED_LAYER not in self.viewer.layers:
            self._correction_status("No tracked layer loaded."); return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._correction_status("Extend: data.db not found — run DB Generation first."); return
        source_id = self.correction_widget._selected_label
        if not source_id:
            self._correction_status("Extend: no cell selected (left-click first)."); return

        layer = self.viewer.layers[_TRACKED_LAYER]
        t = self._current_t()
        tracked = np.asarray(layer.data)
        T = tracked.shape[0]

        target_frame = t + (1 if direction == "forward" else -1)
        if direction == "forward" and t >= T - 1:
            self._correction_status("Already at last frame."); return
        if direction == "backward" and t <= 0:
            self._correction_status("Already at first frame."); return
        if not np.any(tracked[t] == source_id):
            self._correction_status(f"Cell {source_id} not present at t={t}."); return

        validated_tracks = (
            read_validated_tracks(self._pos_dir) if self._pos_dir is not None else {}
        )
        result = extend_track_from_db(
            source_id=source_id, source_frame=t, direction=direction,
            tracked_labels=tracked, db_path=db_path,
            d_max=float(self.extend_max_dist_spin.value()),
            area_weight=float(self.extend_area_weight_spin.value()),
            iou_weight=float(self.extend_iou_weight_spin.value()),
            distance_weight=float(self.extend_distance_weight_spin.value()),
            overlap_penalty=float(self.extend_overlap_penalty_spin.value()),
            greedy_overwrite=self.extend_greedy_overwrite_check.isChecked(),
            validated_tracks=validated_tracks,
        )

        if result is None:
            self._correction_status(
                f"No candidate within {self.extend_max_dist_spin.value():g}px at t={target_frame}."
            ); return

        assignments = result.assignments or ()
        if not assignments:
            assignments = (SimpleNamespace(cell_id=source_id, mask_2d=result.mask_2d),)

        frame = layer.data[result.target_frame]
        changed_ids = {int(a.cell_id) for a in assignments}
        for cid in changed_ids:
            frame[frame == cid] = 0
        if self.extend_greedy_overwrite_check.isChecked():
            for a in assignments:
                frame[a.mask_2d] = int(a.cell_id)
        else:
            for a in assignments:
                frame[a.mask_2d & (frame == 0)] = int(a.cell_id)
        layer.refresh()

        step = list(self.viewer.dims.current_step)
        step[0] = result.target_frame
        self.viewer.dims.current_step = tuple(step)

        moved = (
            f", reassigned {len(changed_ids) - 1} conflict(s)"
            if len(changed_ids) > 1 else ""
        )
        self._correction_status(
            f"Extended cell {source_id} → t={result.target_frame}{moved} "
            f"(dist={result.centroid_distance:.1f}px, area={result.area_ratio:.2f}, "
            f"iou={result.centroid_corrected_iou:.2f}, overlap={result.existing_overlap:.2f})"
        )

    def _on_retrack_forward(self) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._correction_status("No tracked layer loaded."); return
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._correction_status("Need ≥ 2 frames to retrack."); return
        t0 = int(self.viewer.dims.current_step[0])
        if t0 >= layer.data.shape[0] - 1:
            self._correction_status("Already at last frame."); return

        T = layer.data.shape[0]
        stack = layer.data.copy()
        fully_validated = read_validated_frames(self._pos_dir)
        reserved_ids = set(read_validated_tracks(self._pos_dir))
        n_retracked = n_skipped = 0
        for t in range(t0 + 1, T):
            if t in fully_validated:
                n_skipped += 1; continue
            locked = read_validated_cells_at_frame(self._pos_dir, t)
            stack[t] = retrack_frame_constrained(
                stack[t - 1], stack[t], locked,
                max_dist_px=float(self.retrack_max_dist_spin.value()),
                reserved_ids=reserved_ids,
            )
            n_retracked += 1
        layer.data = stack
        self._correction_status(
            f"Retracked forward from t={t0 + 1}: {n_retracked} updated, "
            f"{n_skipped} validated skipped. Unsaved."
        )

    def _on_retrack_backward(self) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._correction_status("No tracked layer loaded."); return
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._correction_status("Need ≥ 2 frames to retrack."); return
        t0 = int(self.viewer.dims.current_step[0])
        if t0 <= 0:
            self._correction_status("Already at first frame."); return

        stack = layer.data.copy()
        fully_validated = read_validated_frames(self._pos_dir)
        reserved_ids = set(read_validated_tracks(self._pos_dir))
        n_retracked = n_skipped = 0
        for t in range(t0 - 1, -1, -1):
            if t in fully_validated:
                n_skipped += 1; continue
            locked = read_validated_cells_at_frame(self._pos_dir, t)
            stack[t] = retrack_frame_constrained(
                stack[t + 1], stack[t], locked,
                max_dist_px=float(self.retrack_max_dist_spin.value()),
                reserved_ids=reserved_ids,
            )
            n_retracked += 1
        layer.data = stack
        self._correction_status(
            f"Retracked backward from t={t0 - 1}: {n_retracked} updated, "
            f"{n_skipped} validated skipped. Unsaved."
        )

    def _on_remove_unvalidated_labels(self) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._correction_status("No tracked layer loaded."); return
        layer = self.viewer.layers[_TRACKED_LAYER]
        data = np.asarray(layer.data)
        if data.ndim < 2:
            self._correction_status("Tracked layer has no image data."); return

        validated_tracks = read_validated_tracks(self._pos_dir)
        frame_count = int(data.shape[0]) if data.ndim >= 3 else 1
        changed_pixels = changed_frames = 0
        for t in range(frame_count):
            frame = self._frame_view_2d(data, t) if data.ndim >= 3 else data
            if frame is None:
                self._correction_status("Tracked layer must be a time-first stack."); return
            validated_ids = {
                cid for cid, frames in validated_tracks.items() if t in frames
            }
            remove_mask = frame != 0
            if validated_ids:
                remove_mask &= ~np.isin(frame, list(validated_ids))
            n_remove = int(np.count_nonzero(remove_mask))
            if not n_remove: continue
            frame[remove_mask] = 0
            changed_pixels += n_remove
            changed_frames += 1

        if not changed_pixels:
            self._correction_status("No unvalidated labels found."); return
        layer.refresh()
        if self.correction_widget._selected_label:
            ct = self._current_t()
            if self.correction_widget._selected_label not in self._current_cell_ids(ct):
                self.correction_widget.select_label(ct, 0)
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        self._correction_status(
            f"Removed unvalidated labels in {changed_frames} frame(s), "
            f"{changed_pixels} px changed. Unsaved."
        )

    def _on_correction_worker_error(self, exc: Exception) -> None:
        self._correction_status(f"Error: {exc}")
        logger.exception("Correction worker error", exc_info=exc)

    # ================================================================
    # Keyboard / Validation
    # ================================================================
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
            self._correction_status(
                "Validation toggle: no cell selected (left-click first)."
            ); return
        t = self._current_t()
        if sel not in self._current_cell_ids(t):
            self._correction_status(f"Cell {sel} not present at t={t}."); return
        frames = self._frames_with_cell(sel)
        if not frames:
            return
        if is_track_validated(self._pos_dir, sel):
            invalidate_track(self._pos_dir, sel)
            self._correction_status(
                f"Cell {sel} invalidated across {len(frames)} frame(s)."
            )
        else:
            validate_track(self._pos_dir, sel, frames)
            self._correction_status(
                f"Cell {sel} validated across {len(frames)} frame(s)."
            )
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _on_dims_step_changed(self, event=None) -> None:
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        if self.ultrack_db_browser_section.is_expanded:
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, self._refresh_ultrack_db_browser)

    def _refresh_validated_overlay(self) -> None:
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
            return
        mask2d = (
            np.isin(frame, list(validated_ids)).astype(np.uint8) if validated_ids
            else np.zeros(frame.shape, dtype=np.uint8)
        )
        full = np.zeros(tracked.data.shape, dtype=np.uint8)
        full[t] = mask2d
        if overlay_exists:
            self.viewer.layers[_VALIDATED_OVERLAY].data = full
        else:
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, lambda data=full: self._add_validated_overlay(data))

    def _add_validated_overlay(self, data: np.ndarray) -> None:
        if _VALIDATED_OVERLAY in self.viewer.layers:
            layer = self.viewer.layers[_VALIDATED_OVERLAY]
            layer.data = data
            layer.opacity = _VALIDATED_OVERLAY_OPACITY
            self._place_validated_overlay_below_spotlight()
            return
        self.viewer.add_labels(
            data, name=_VALIDATED_OVERLAY,
            opacity=_VALIDATED_OVERLAY_OPACITY,
            colormap=direct_colormap({None: (0, 0, 0, 0), 1: "#00ff00"}),
        )
        self._place_validated_overlay_below_spotlight()
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[_TRACKED_LAYER]

    def _place_validated_overlay_below_spotlight(self) -> None:
        if _VALIDATED_OVERLAY not in self.viewer.layers:
            return
        if _SPOTLIGHT_LAYER not in self.viewer.layers:
            return
        vi = self.viewer.layers.index(_VALIDATED_OVERLAY)
        si = self.viewer.layers.index(_SPOTLIGHT_LAYER)
        if vi > si:
            self.viewer.layers.move(vi, si)

    def _refresh_validation_counter(self) -> None:
        if self._pos_dir is None or _TRACKED_LAYER not in self.viewer.layers:
            self.validation_counter_lbl.setText(""); return
        validated_tracks = read_validated_tracks(self._pos_dir)
        n_tracks = len(validated_tracks)
        n_cf = sum(len(f) for f in validated_tracks.values())
        self.validation_counter_lbl.setText(
            f"{n_tracks} track(s) validated, {n_cf} cell-frame(s) covered"
        )

    def _on_cells_edited(self, t: int, changed_ids: set[int]) -> None:
        if self._pos_dir is None:
            return
        for cid in changed_ids:
            invalidate_track(self._pos_dir, cid)
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _frames_with_cell(self, cell_id: int) -> list[int]:
        if cell_id == 0 or _TRACKED_LAYER not in self.viewer.layers:
            return []
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim < 3:
            return []
        spatial_axes = tuple(range(1, layer.data.ndim))
        present = np.any(layer.data == cell_id, axis=spatial_axes)
        return [int(t) for t in np.where(present)[0]]