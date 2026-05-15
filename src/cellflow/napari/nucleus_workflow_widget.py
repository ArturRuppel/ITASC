"""Nucleus workflow widget for hypothesis generation and tracking in CellFlow.

Simplified workflow layout with action buttons grouped into their owning
sections: segmentation inputs, tracking/Ultrack, database browser, correction.

Stages:
  1. Segmentation inputs → ``contours.tif`` / ``foreground_scores.tif``
  2. Source stacks → ``contour_sources.tif`` / ``foreground_sources.tif``
  3. Ultrack database + solve → ``data.db`` / ``tracked_labels.tif``
  4. Correction (load / save / extend / retrack / reassign / remove unvalidated)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from napari.utils.colormaps import Colormap
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QKeySequence
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from cellflow.correction.labels import best_overlapping_label
from cellflow.database.tracked import (
    read_full_tracked_stack,
    write_tracked_frame,
)
from cellflow.database.validation import (
    add_anchor,
    add_correction,
    invalidate_track,
    is_track_validated,
    read_corrections,
    read_validated_cells_at_frame,
    read_validated_frames,
    read_validated_tracks,
    remap_validated_tracks,
    write_corrections,
)
from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.radial_refinement_widget import RadialRefinementWidget
from cellflow.napari._paths import NucleusArtifactPaths
from cellflow.napari._state import dump_state, load_state
from cellflow.napari import _thresholds
from cellflow.napari.artifact_visualization import (
    _categorical_colors,
    _nucleus_centroids_by_track,
    _rasterize_track_image,
)
from cellflow.napari._widget_helpers import (
    btn as _btn,
    button_grid as _button_grid,
    dspin as _dspin,
    heading as _heading,
    ispin as _ispin,
    make_progress as _make_progress,
    make_status as _make_status,
    separator as _separator,
)
from cellflow.napari.ui_style import (
    add_block_checkbox_row,
    add_block_pair_row,
    add_sweep_parameter_row,
    block_grid,
    compact_spinbox,
    danger_button,
    muted_label,
    sweep_parameter_grid,
)
from cellflow.napari.validated_overlay_controller import (
    ValidatedOverlayController,
)
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.segmentation import build_consensus_boundary, build_nucleus_averaged_maps
from cellflow.tracking_ultrack.config import TrackingConfig as UltrackConfig
from cellflow.tracking_ultrack.corrections import Correction
from cellflow.tracking_ultrack.db_query import (
    HierarchyCutState as _HierarchyCutState,
    annotation_name as _ultrack_db_annotation_name,
    node_annotation_metadata as _ultrack_db_node_annotation_metadata,
    node_mask_and_bbox as _node_mask_and_bbox,
    node_preview_metadata as _ultrack_db_node_preview_metadata,
    paint_nodes as _paint_ultrack_db_nodes,
    query_available_sources as _query_available_sources,
    query_connected_nodes as _query_ultrack_db_connected_nodes,
    query_distinct_heights as _query_distinct_heights,
    query_hierarchy_cut_states as _query_hierarchy_cut_states,
    query_middle_frame as _query_ultrack_db_middle_frame,
    render_hierarchy_cut as _render_hierarchy_cut,
    render_hierarchy_cut_state as _render_hierarchy_cut_state,
    summary_text as _ultrack_db_summary_text,
)
from cellflow.tracking_ultrack.db_build import apply_annotations_and_score
from cellflow.tracking_ultrack.export import export_tracked_labels
from cellflow.tracking_ultrack.extend import extend_track_from_db
from cellflow.tracking_ultrack.swap_candidate import (
    SwapCandidate as _SwapCandidate,
    _SwapCursor,
    list_swap_candidates,
    step_smaller as _step_smaller,
    step_larger as _step_larger,
)
from cellflow.tracking_ultrack.ingest import _select_solver
from cellflow.tracking_ultrack.multi_threshold import (
    build_ultrack_database_from_sources,
    preview_ultrack_source_stack_frame,
    write_ultrack_source_stacks,
)
from cellflow.tracking_ultrack.retracker import retrack_frame_constrained
from cellflow.tracking_ultrack.solve import run_solve

logger = logging.getLogger(__name__)

try:
    from ultrack.core.segmentation.processing import segment as _ultrack_segment
except ImportError:
    _ultrack_segment = None  # type: ignore[assignment]

# ── Layer name constants ──────────────────────────────────────────────────────
_PREVIEW_LAYER = "Preview: Nucleus"
_HYP_LAYER = "Hypothesis: Nucleus"
_TRACKED_LAYER = "Tracked: Nucleus"
_CONTOUR_LAYER = "Contour Map: Nucleus"
_CELLPROB_LAYER = "Cellprob Map: Nucleus"
_FOREGROUND_SCORE_LAYER = "Foreground Score: Nucleus"
_CELL_ZAVG_LAYER = "Cell z-avg"
_NUC_ZAVG_LAYER = "Nucleus z-avg"
_ULTRACK_DB_PREVIEW_LAYER = "Ultrack DB Preview"
_ULTRACK_DB_SELECTION_LAYER = "Ultrack DB Selection"
_ULTRACK_DB_ANNOTATION_LAYER = "Ultrack DB Annotations"

# Correction-owned layer constants
_CORRECTION_TRACKED_LAYER = "[Correction] Tracked: Nucleus"
_CORRECTION_TRACK_LAYER = "[Correction] Nucleus tracks"
_CORRECTION_CELL_ZAVG_LAYER = "[Correction] Cell z-avg"
_CORRECTION_NUC_ZAVG_LAYER = "[Correction] Nucleus z-avg"
_CORRECTION_NLS_ZAVG_LAYER = "[Correction] NLS z-avg"


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
        self._ultrack_db_sources_cache: dict[tuple, tuple[int, ...]] = {}
        self._ultrack_db_browser_active: bool = False
        self._ultrack_db_frame_initialized: bool = False
        self._ultrack_db_selected_node_id: int | None = None
        self._ultrack_db_selected_frame: int | None = None
        self._ultrack_db_label_to_node_id: dict[int, int] = {}
        self._ultrack_db_node_id_to_label: dict[int, int] = {}
        self._ultrack_db_node_annotations: dict[int, str] = {}
        self._ultrack_db_preview_labels: np.ndarray | None = None
        self._ultrack_db_preview_mouse_callback = None

        self._correction_owned_layers: set[str] = set()
        self._correction_view_state: dict | None = None
        self._swap_cursor: _SwapCursor | None = None
        self._validated_overlay = ValidatedOverlayController(
            self.viewer,
            tracked_layer_provider=self._correction_tracked_layer,
            pos_dir_provider=lambda: self._pos_dir,
            current_t_provider=self._current_t,
            owned_layers=self._correction_owned_layers,
        )

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
                    ("2_nucleus/contours.tif", "Contours"),
                    ("2_nucleus/foreground_scores.tif", "Foreground scores"),
                    ("2_nucleus/contour_sources.tif", "Contour sources"),
                    ("2_nucleus/foreground_sources.tif", "Foreground sources"),
                    ("2_nucleus/ultrack_workdir/data.db", "Ultrack database"),
                ]),
                ("Output", [
                    ("2_nucleus/tracked_labels.tif", "Tracked labels"),
                ]),
            ],
            viewer=self.viewer,
        )
        root.addWidget(
            CollapsibleSection(
                "Pipeline Files",
                self._files_widget,
                expanded=False,
                title_role="stage",
                title_level=1,
            )
        )

        # ── Workflow sections ────────────────────────────────────────
        self._build_segmentation_inputs_section(root)
        self._build_tracking_ultrack_section(root)

        # ── Ultrack Database Browser ─────────────────────────────────
        self._build_db_browser_section(root)

        # ── Refinement ───────────────────────────────────────────────
        self._build_refinement_section(root)

        # ── Correction (group box) ───────────────────────────────────
        self._build_correction_section(root)

        root.addStretch()

    # -- Parameters --------------------------------------------------------

    def _build_segmentation_inputs_section(self, root: QVBoxLayout) -> None:
        params_inner = QWidget()
        params_lay = QVBoxLayout(params_inner)
        params_lay.setContentsMargins(0, 0, 0, 0)
        params_lay.setSpacing(6)

        g = sweep_parameter_grid(horizontal_spacing=12)
        self.map_cellprob_min_spin = _dspin(
            -20, 20, -3.0, 1.0, 1,
            "Minimum Cellpose probability threshold for averaged-map generation.",
        )
        self.map_cellprob_max_spin = _dspin(
            -20, 20, 0.0, 1.0, 1,
            "Maximum Cellpose probability threshold for averaged-map generation.",
        )
        self.map_cellprob_step_spin = _dspin(
            0.1, 10, 1.0, 0.5, 1,
            "Cellpose probability threshold step for averaged-map generation.",
        )
        self.map_z_start_spin = _ispin(
            0, 999, 0,
            tooltip="First z slice included in averaged-map generation.",
        )
        self.map_z_stop_spin = _ispin(
            -1, 999, -1,
            tooltip="Last z slice included. -1 means all z slices.",
        )
        self.map_z_step_spin = _ispin(
            1, 999, 1,
            tooltip="Z-slice step for averaged-map generation.",
        )
        self.source_contour_threshold_min_spin = _dspin(
            0, 1, 0.1, 0.05, 2,
            "Minimum normalized contour threshold for the source sweep.",
        )
        self.source_contour_threshold_max_spin = _dspin(
            0, 1, 0.5, 0.05, 2,
            "Maximum normalized contour threshold for the source sweep.",
        )
        self.source_contour_threshold_step_spin = _dspin(
            0.001, 1, 0.1, 0.05, 3,
            "Step size for normalized contour source thresholds.",
        )
        self.source_foreground_threshold_min_spin = _dspin(
            0, 1, 0.1, 0.05, 2,
            "Minimum normalized foreground-score threshold for the source sweep.",
        )
        self.source_foreground_threshold_max_spin = _dspin(
            0, 1, 0.5, 0.05, 2,
            "Maximum normalized foreground-score threshold for the source sweep.",
        )
        self.source_foreground_threshold_step_spin = _dspin(
            0.001, 1, 0.1, 0.05, 3,
            "Step size for normalized foreground source thresholds.",
        )
        add_sweep_parameter_row(
            g, 1, "Cellprob:",
            self.map_cellprob_min_spin,
            self.map_cellprob_max_spin,
            self.map_cellprob_step_spin,
        )
        add_sweep_parameter_row(
            g, 2, "Z:",
            self.map_z_start_spin,
            self.map_z_stop_spin,
            self.map_z_step_spin,
        )
        add_sweep_parameter_row(
            g, 3, "Contour:",
            self.source_contour_threshold_min_spin,
            self.source_contour_threshold_max_spin,
            self.source_contour_threshold_step_spin,
        )
        add_sweep_parameter_row(
            g, 4, "Foreground:",
            self.source_foreground_threshold_min_spin,
            self.source_foreground_threshold_max_spin,
            self.source_foreground_threshold_step_spin,
        )
        params_lay.addLayout(g)
        self.db_gen_threshold_min_spin = self.source_contour_threshold_min_spin
        self.db_gen_threshold_max_spin = self.source_contour_threshold_max_spin
        self.db_gen_threshold_step_spin = self.source_contour_threshold_step_spin

        self.segmentation_inputs_parameters_section = CollapsibleSection(
            "Segmentation Input Parameters",
            params_inner,
            expanded=True,
            title_role="params",
            title_level=1,
        )
        self.segmentation_inputs_section = self.segmentation_inputs_parameters_section

        self.preview_contour_btn = _btn(
            "Preview Segmentation Inputs",
            "Build the current frame's segmentation input source sweep in memory and display it in napari.",
        )
        self.build_btn = _btn(
            "Build Segmentation Inputs",
            "Build averaged maps, then contour and foreground source stacks from segmentation inputs.",
        )
        self.build_maps_btn = self.build_btn

        self.pipeline_status_lbl = _make_status()
        self.pipeline_progress_bar = _make_progress()
        root.addWidget(self.segmentation_inputs_parameters_section)

    def _build_tracking_ultrack_section(self, root: QVBoxLayout) -> None:
        params_inner = QWidget()
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

        self.run_db_gen_btn = _btn(
            "Build Ultrack Database",
            "Build an Ultrack candidate database from explicit source-stack artifacts.",
        )
        self.run_ultrack_btn = _btn(
            "Run Ultrack",
            "Solve ILP tracking and export tracked_labels.tif.",
        )
        self.cancel_btn = _btn("Cancel", "Cancel the currently running pipeline step.")
        self.cancel_btn.setEnabled(False)

        self.tracking_ultrack_parameters_section = CollapsibleSection(
            "Ultrack Parameters",
            params_inner,
            expanded=False,
            title_role="params",
            title_level=1,
        )
        self.tracking_ultrack_section = self.tracking_ultrack_parameters_section
        root.addWidget(self.tracking_ultrack_parameters_section)
        root.addLayout(_button_grid((self.preview_contour_btn, self.build_btn)))
        root.addLayout(_button_grid(
            (self.run_db_gen_btn, self.run_ultrack_btn),
            (self.cancel_btn,),
        ))
        root.addWidget(self.pipeline_status_lbl)
        root.addWidget(self.pipeline_progress_bar)

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

        # ── Threshold source slider (for multi-threshold merged DBs) ─────
        self.ultrack_db_source_slider = QSlider(Qt.Horizontal)
        self.ultrack_db_source_slider.setRange(0, 0)
        self.ultrack_db_source_slider.setValue(0)
        self.ultrack_db_source_slider.setToolTip(
            "Select threshold source: 0 = lowest threshold, higher = more stringent"
        )
        self.ultrack_db_source_slider.setEnabled(False)
        self.ultrack_db_source_lbl = QLabel("all")
        self.ultrack_db_source_lbl.setFixedWidth(48)
        self._ultrack_db_source_slider_row = QWidget()
        _source_slider_lay = QHBoxLayout(self._ultrack_db_source_slider_row)
        _source_slider_lay.setContentsMargins(0, 0, 0, 0)
        _source_slider_lay.addWidget(self.ultrack_db_source_slider)
        _source_slider_lay.addWidget(self.ultrack_db_source_lbl)
        lay.addWidget(self._ultrack_db_source_slider_row)

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
            "Database Browser",
            _inner,
            expanded=False,
            title_role="indicators",
            title_level=1,
        )
        root.addWidget(self.ultrack_db_browser_section)

    # -- Refinement --------------------------------------------------------

    def _build_refinement_section(self, root: QVBoxLayout) -> None:
        self.refinement_widget = RadialRefinementWidget(
            self.viewer,
            pos_dir_provider=lambda: self._pos_dir,
        )
        self.refinement_widget.set_correction_active_provider(
            lambda: self.correction_active_btn.isChecked()
        )
        self.refinement_widget.set_on_promoted_callback(
            self._on_refinement_promoted
        )
        self.refinement_section = CollapsibleSection(
            "Refinement",
            self.refinement_widget,
            expanded=False,
            title_role="stage",
            title_level=1,
        )
        root.addWidget(self.refinement_section)

    def _on_refinement_promoted(self) -> None:
        if self._pos_dir is not None:
            self._files_widget.refresh(self._pos_dir)

    # -- Correction --------------------------------------------------------

    def _build_correction_section(self, root: QVBoxLayout) -> None:
        inner = QWidget()
        group_lay = QVBoxLayout(inner)
        group_lay.setContentsMargins(0, 0, 0, 0)
        group_lay.setSpacing(6)

        self.correction_active_btn = QPushButton("Activate Correction")
        self.correction_active_btn.setCheckable(True)
        self.correction_active_btn.setToolTip(
            "Activate correction mode and show correction layers and controls."
        )

        # ── Action buttons — 2-column grid ────────────────────────
        self.save_tracked_btn = _btn(
            "Save tracked (S)", "Save corrected tracked nucleus labels to disk."
        )
        self.extend_back_btn = _btn(
            "◀ Extend (A)", "Extend selected track one frame backward."
        )
        self.extend_fwd_btn = _btn(
            "Extend (D) ▶", "Extend selected track one frame forward."
        )
        self.retrack_back_btn = _btn(
            "◀ Retrack (Q)", "Retrack all labels backward from current frame."
        )
        self.retrack_fwd_btn = _btn(
            "Retrack (E) ▶", "Retrack all labels forward from current frame."
        )
        self.reassign_ids_btn = _btn(
            "Reassign ID", "Reassign cell IDs to contiguous range 1-N."
        )
        self.validate_track_btn = _btn(
            "Validate track", "Lock selected cell geometry in every frame where it appears."
        )
        self.anchor_here_btn = _btn(
            "Anchor here", "Anchor selected cell identity at the current frame."
        )
        self.remove_unvalidated_btn = _btn(
            "Remove unvalidated",
            "Remove nucleus label pixels not marked validated for their frame.",
        )
        danger_button(self.remove_unvalidated_btn)

        group_lay.addLayout(_button_grid(
            (self.save_tracked_btn,),
            (self.extend_back_btn, self.extend_fwd_btn),
            (self.retrack_back_btn, self.retrack_fwd_btn),
            (self.validate_track_btn, self.anchor_here_btn),
            (self.reassign_ids_btn,),
            (self.remove_unvalidated_btn,),
        ))

        self.correction_status_lbl = _make_status()
        group_lay.addWidget(self.correction_status_lbl)

        self.validation_counter_lbl = QLabel("")
        self.validation_counter_lbl.setWordWrap(True)
        group_lay.addWidget(self.validation_counter_lbl)

        # ── Extend / retrack parameters (collapsible) ─────────────
        extend_retrack_inner = QWidget()
        extend_retrack_lay = QVBoxLayout(extend_retrack_inner)
        extend_retrack_lay.setContentsMargins(0, 0, 0, 0)
        extend_retrack_lay.setSpacing(6)

        extend_retrack_lay.addWidget(_heading("Extend"))
        g = block_grid(horizontal_spacing=12)
        self.extend_max_dist_spin = _dspin(0, 500, 40.0, 1.0, 1)
        self.extend_area_weight_spin = _dspin(0, 10, 1.0, 0.1, 2)
        self.extend_iou_weight_spin = _dspin(0, 10, 1.0, 0.1, 2)
        self.extend_distance_weight_spin = _dspin(0, 10, 0.05, 0.01, 3)
        self.extend_overlap_penalty_spin = _dspin(0, 10, 1.0, 0.1, 2)
        self.extend_greedy_overwrite_check = QCheckBox("Greedy overwrite")
        add_block_pair_row(g, 0,
            "Max dist:", compact_spinbox(self.extend_max_dist_spin),
            "Area wt:", compact_spinbox(self.extend_area_weight_spin))
        add_block_pair_row(g, 1,
            "IoU wt:", compact_spinbox(self.extend_iou_weight_spin),
            "Dist wt:", compact_spinbox(self.extend_distance_weight_spin))
        self.swap_radius_spin = _dspin(0, 500, 40.0, 1.0, 1)
        add_block_pair_row(g, 2,
            "Overlap pen:", compact_spinbox(self.extend_overlap_penalty_spin))
        add_block_pair_row(g, 3,
            "Swap radius:", compact_spinbox(self.swap_radius_spin))
        add_block_checkbox_row(g, 4, self.extend_greedy_overwrite_check)
        swap_hint = QLabel("Z / C — swap selection with smaller / larger hypothesis fragment.")
        swap_hint.setWordWrap(True)
        muted_label(swap_hint)
        extend_retrack_lay.addLayout(g)
        extend_retrack_lay.addWidget(swap_hint)

        extend_retrack_lay.addWidget(_heading("Retrack"))
        g = block_grid(horizontal_spacing=12)
        self.retrack_max_dist_spin = _dspin(0, 500, 20.0, 1.0, 1)
        add_block_pair_row(g, 0,
            "Max dist:", compact_spinbox(self.retrack_max_dist_spin))
        extend_retrack_lay.addLayout(g)
        self.extend_retrack_params_section = CollapsibleSection(
            "Extend / Retrack Parameters",
            extend_retrack_inner,
            expanded=False,
            title_role="params",
            title_level=2,
        )
        self.extend_params_section = self.extend_retrack_params_section
        self.retrack_params_section = self.extend_retrack_params_section
        group_lay.addWidget(self.extend_retrack_params_section)

        # ── Inline CorrectionWidget ───────────────────────────────
        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
            show_cleanup=False,
        )
        self.correction_widget.set_edit_callback(self._on_cells_edited)

        self.correction_shortcuts_section = CollapsibleSection(
            "Correction Shortcuts",
            self.correction_widget.build_shortcuts_widget(),
            expanded=True,
            title_role="actions",
            title_level=2,
        )
        group_lay.addWidget(self.correction_shortcuts_section)

        self.artifact_cleanup_section = CollapsibleSection(
            "Artifact Cleanup",
            self.correction_widget._cleanup_container,
            expanded=False,
            title_role="params",
            title_level=2,
        )
        group_lay.addWidget(self.artifact_cleanup_section)
        group_lay.addWidget(self.correction_widget)

        self.correction_mode_section = CollapsibleSection(
            "Correction",
            inner,
            expanded=False,
            title_role="stage",
            title_level=1,
        )
        self.correction_mode_section._toggle.setVisible(False)
        self.correction_mode_section._toggle.setEnabled(False)
        root.addWidget(self.correction_active_btn)
        root.addWidget(self.correction_mode_section)

    # ================================================================
    # Signals
    # ================================================================
    def _connect_signals(self) -> None:
        # Pipeline buttons
        self.preview_contour_btn.clicked.connect(self._on_preview_contour_maps)
        self.build_btn.clicked.connect(self._on_build_segmentation_inputs)
        self.run_db_gen_btn.clicked.connect(self._on_run_db_generation)
        self.run_ultrack_btn.clicked.connect(self._on_run_ultrack)
        self.cancel_btn.clicked.connect(self._on_cancel)

        # Parameter interactions
        self.db_gen_linking_mode_combo.currentTextChanged.connect(
            self._on_db_gen_mode_changed
        )
        # DB Browser
        self.ultrack_db_active_btn.toggled.connect(self._on_ultrack_db_activate)
        self.ultrack_db_refresh_btn.clicked.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_source_slider.valueChanged.connect(
            self._on_ultrack_db_source_changed
        )
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
        self.save_tracked_btn.clicked.connect(self._on_save_tracked)
        self.reassign_ids_btn.clicked.connect(self._on_reassign_ids)
        self.validate_track_btn.clicked.connect(self._on_validate_track)
        self.anchor_here_btn.clicked.connect(self._on_anchor_here)
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
        self.correction_active_btn.toggled.connect(
            self._on_correction_active_button_toggled
        )
        self.correction_widget._activate_btn.toggled.connect(
            self._on_correction_mode_toggled
        )
        self.set_selection_callback(None)

        # Initial state
        solver = _select_solver()
        solver_display = "Gurobi (licensed)" if solver == "GUROBI" else "CBC"
        self.ultrack_solver_lbl.setText(solver_display)

    # ================================================================
    # Path helpers
    # ================================================================
    @property
    def _paths(self) -> NucleusArtifactPaths | None:
        return NucleusArtifactPaths(self._pos_dir) if self._pos_dir else None

    def _tracked_path(self) -> Path | None:
        return self._paths.tracked if self._paths else None

    def _ensure_tracked_layer_data(self) -> np.ndarray | None:
        """Return the tracked labelmap from the viewer layer if present, else
        read it from disk. Does not add anything to the viewer."""
        if _TRACKED_LAYER in self.viewer.layers:
            return np.asarray(self.viewer.layers[_TRACKED_LAYER].data)
        tracked_path = self._tracked_path()
        if tracked_path is None or not tracked_path.exists():
            return None
        self._status(f"Reading {tracked_path.name} from disk…")
        labels = np.asarray(tifffile.imread(str(tracked_path)), dtype=np.uint32)
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        return labels

    def _prob_path(self) -> Path | None:
        return self._paths.prob if self._paths else None

    def _dp_path(self) -> Path | None:
        return self._paths.dp if self._paths else None

    def _contours_path(self) -> Path | None:
        return self._paths.contours if self._paths else None

    def _contour_maps_path(self) -> Path | None:
        return self._contours_path()

    def _contour_sources_path(self) -> Path | None:
        return self._paths.contour_sources if self._paths else None

    def _foreground_sources_path(self) -> Path | None:
        return self._paths.foreground_sources if self._paths else None

    def _foreground_scores_path(self) -> Path | None:
        return self._paths.foreground_scores if self._paths else None

    def _cell_zavg_path(self) -> Path | None:
        return self._paths.cell_zavg if self._paths else None

    def _nucleus_zavg_path(self) -> Path | None:
        return self._paths.nucleus_zavg if self._paths else None

    def _nls_zavg_path(self) -> Path | None:
        return self._paths.nls_zavg if self._paths else None

    def _ultrack_workdir(self) -> Path | None:
        return self._paths.ultrack_workdir if self._paths else None

    def _ultrack_db_path(self) -> Path | None:
        return self._paths.ultrack_db if self._paths else None

    def _nucleus_prob_zavg_path(self) -> Path | None:
        return self._paths.nucleus_prob_zavg if self._paths else None

    # ================================================================
    # Public API
    # ================================================================
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._files_widget.refresh(pos_dir)
        if hasattr(self, "refinement_widget"):
            self.refinement_widget.refresh()
        if pos_dir is None:
            if self.correction_active_btn.isChecked():
                self.correction_active_btn.setChecked(False)
            else:
                self.correction_widget.deactivate()
                self._remove_correction_owned_layers()
            return
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def get_state(self) -> dict:
        return dump_state(self)

    def set_state(self, state: dict) -> None:
        load_state(self, state)


    def set_selection_callback(self, fn) -> None:
        def composed(t, label):
            self._swap_cursor = None
            if fn is not None:
                fn(t, label)
        self.correction_widget.set_selection_callback(composed)

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
            self.build_maps_btn,
            self.preview_contour_btn,
            self.build_btn,
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

    @staticmethod
    def _preview_frame_from_step(
        current_step: tuple[int, ...],
        frame_count: int,
        *,
        source_time_axes: bool,
    ) -> int:
        axis = 1 if source_time_axes and len(current_step) >= 2 else 0
        if not current_step:
            return 0
        return min(max(int(current_step[axis]), 0), frame_count - 1)

    def _segmentation_preview_has_source_time_axes(self) -> bool:
        for name in (_CONTOUR_LAYER, _FOREGROUND_SCORE_LAYER):
            if name not in self.viewer.layers:
                continue
            data = np.asarray(self.viewer.layers[name].data)
            if data.ndim == 4:
                return True
        return False

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

    def _ensure_nucleus_zavg_layer(self) -> None:
        if _NUC_ZAVG_LAYER in self.viewer.layers:
            return
        zavg_path = self._nucleus_zavg_path()
        if zavg_path is None or not zavg_path.exists():
            return
        data = np.asarray(tifffile.imread(str(zavg_path)), dtype=np.float32)
        self.viewer.add_image(
            data,
            name=_NUC_ZAVG_LAYER,
            colormap="bop orange",
            visible=True,
        )

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
        layer = self._correction_tracked_layer()
        if layer is None:
            return set()
        frame = self._frame_view_2d(layer.data, t)
        if frame is None:
            return set()
        return set(int(v) for v in np.unique(frame)) - {0}

    # ================================================================
    # 1. Source Stacks
    # ================================================================
    def _on_build_segmentation_inputs(self) -> None:
        if self._pos_dir is None:
            self._status("No project open."); return
        prob_path = self._prob_path()
        dp_path = self._dp_path()
        contours_path = self._pos_dir / "2_nucleus" / "contours.tif"
        score_path = self._foreground_scores_path()
        contour_sources_path = self._contour_sources_path()
        foreground_sources_path = self._foreground_sources_path()
        if prob_path is None or not prob_path.exists():
            self._status(f"Missing: {prob_path}"); return
        if dp_path is None or not dp_path.exists():
            self._status(f"Missing: {dp_path}"); return
        if score_path is None or contour_sources_path is None or foreground_sources_path is None:
            self._status("No project open."); return
        try:
            map_thresholds = self._map_cellprob_thresholds_from_controls()
            z_indices = self._map_z_indices_from_controls()
            contour_thresholds = self._source_contour_thresholds_from_controls()
            foreground_thresholds = self._source_foreground_thresholds_from_controls()
        except ValueError as exc:
            self._status(str(exc)); return
        pos_dir = self._pos_dir

        def _done(result):
            report, n_sources = result
            self._contour_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            self._files_widget.refresh(pos_dir)
            frames = int(getattr(report, "frames", 0))
            self._status(f"Segmentation inputs built ({frames} frames, {n_sources} sources).")

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": self._on_contour_worker_error,
        })
        def _worker():
            import queue as _queue
            import threading

            msg_queue: _queue.SimpleQueue = _queue.SimpleQueue()
            result_holder: list = []
            exc_holder: list = []

            def _progress_cb(done: int, total: int, msg: str) -> None:
                msg_queue.put((done, total + 1, msg))

            def _run_maps() -> None:
                try:
                    result_holder.append(
                        build_nucleus_averaged_maps(
                            prob_path,
                            dp_path,
                            contours_path,
                            score_path,
                            cellprob_thresholds=map_thresholds,
                            z_indices=z_indices,
                            progress_cb=_progress_cb,
                        )
                    )
                except Exception as e:
                    exc_holder.append(e)

            t = threading.Thread(target=_run_maps, daemon=True)
            t.start()
            yield (0, 1, "Starting averaged-map build...")
            while t.is_alive() or not msg_queue.empty():
                try:
                    yield msg_queue.get_nowait()
                except _queue.Empty:
                    t.join(timeout=0.05)
            if exc_holder:
                raise exc_holder[0]
            report = result_holder[0]
            map_frames = max(1, int(getattr(report, "frames", 0)))
            yield (map_frames, map_frames + 1, "Building Ultrack source stacks...")
            metadata = write_ultrack_source_stacks(
                contours_path,
                score_path,
                contour_sources_path,
                foreground_sources_path,
                contour_thresholds=contour_thresholds,
                foreground_thresholds=foreground_thresholds,
            )
            yield (map_frames + 1, map_frames + 1, "Saved segmentation inputs.")
            return report, len(metadata)

        n_sources = len(contour_thresholds) * len(foreground_thresholds)
        self._status(
            f"Building segmentation inputs "
            f"({len(map_thresholds)} cellprob thresholds, {n_sources} sources)..."
        )
        self._set_pipeline_buttons_enabled(False)
        self._contour_worker = _worker()

    def _on_build_nucleus_maps(self) -> None:
        if self._pos_dir is None:
            self._status("No project open."); return
        prob_path = self._prob_path()
        dp_path = self._dp_path()
        contours_path = self._pos_dir / "2_nucleus" / "contours.tif"
        score_path = self._foreground_scores_path()
        if prob_path is None or not prob_path.exists():
            self._status(f"Missing: {prob_path}"); return
        if dp_path is None or not dp_path.exists():
            self._status(f"Missing: {dp_path}"); return
        if score_path is None:
            self._status("No project open."); return
        try:
            thresholds = self._map_cellprob_thresholds_from_controls()
            z_indices = self._map_z_indices_from_controls()
        except ValueError as exc:
            self._status(str(exc)); return
        pos_dir = self._pos_dir

        def _done(report):
            self._contour_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            self._files_widget.refresh(pos_dir)
            frames = int(getattr(report, "frames", 0))
            self._status(f"Averaged maps built ({frames} frames).")

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": self._on_contour_worker_error,
        })
        def _worker():
            return build_nucleus_averaged_maps(
                prob_path,
                dp_path,
                contours_path,
                score_path,
                cellprob_thresholds=thresholds,
                z_indices=z_indices,
            )

        self._status(f"Building averaged maps ({len(thresholds)} cellprob thresholds)…")
        self._set_pipeline_buttons_enabled(False)
        self._contour_worker = _worker()

    def _on_build_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._status("No project open."); return
        contours_path = self._contours_path()
        score_path = self._foreground_scores_path()
        contour_sources_path = self._contour_sources_path()
        foreground_sources_path = self._foreground_sources_path()
        if contours_path is None or score_path is None:
            self._status("No project open."); return
        if not contours_path.exists():
            self._status("Missing: contours.tif (or legacy contour_maps.tif) — build segmentation inputs first."); return
        if not score_path.exists():
            self._status("Missing: foreground_scores.tif — build segmentation inputs first."); return
        if contour_sources_path is None or foreground_sources_path is None:
            self._status("No project open."); return

        try:
            contour_thresholds = self._source_contour_thresholds_from_controls()
            foreground_thresholds = self._source_foreground_thresholds_from_controls()
        except ValueError as exc:
            self._status(str(exc)); return
        pos_dir = self._pos_dir

        def _done(result):
            pos_dir_result, n_sources = result
            self._contour_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            self._files_widget.refresh(pos_dir_result)
            self._status(f"Ultrack source stacks built ({n_sources} sources).")

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": self._on_contour_worker_error,
        })
        def _worker():
            yield (0, 1, "Building Ultrack source stacks…")
            metadata = write_ultrack_source_stacks(
                contours_path,
                score_path,
                contour_sources_path,
                foreground_sources_path,
                contour_thresholds=contour_thresholds,
                foreground_thresholds=foreground_thresholds,
            )
            yield (1, 1, "Saved Ultrack source stacks.")
            return pos_dir, len(metadata)

        n_sources = len(contour_thresholds) * len(foreground_thresholds)
        self._status(f"Building Ultrack source stacks ({n_sources} sources)…")
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

        current_step = tuple(int(v) for v in self.viewer.dims.current_step)
        self._ensure_nucleus_zavg_layer()
        try:
            map_thresholds = self._map_cellprob_thresholds_from_controls()
            z_indices = self._map_z_indices_from_controls()
            contour_thresholds = self._source_contour_thresholds_from_controls()
            foreground_thresholds = self._source_foreground_thresholds_from_controls()
        except ValueError as exc:
            self._status(str(exc)); return

        def _done(result):
            self._contour_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            contour_data, foreground_data, t_idx, frame_count, n_sources = result
            contour_data = np.asarray(contour_data)
            foreground_data = np.asarray(foreground_data)
            if contour_data.ndim == 2:
                contour_data = contour_data[np.newaxis, ...]
            if foreground_data.ndim == 2:
                foreground_data = foreground_data[np.newaxis, ...]
            if contour_data.ndim != 3 or foreground_data.ndim != 3:
                raise ValueError("Preview source frames must be PxYxX or YxX.")
            contour_stack = np.zeros(
                (contour_data.shape[0], frame_count) + contour_data.shape[1:],
                dtype=contour_data.dtype,
            )
            foreground_stack = np.zeros(
                (foreground_data.shape[0], frame_count) + foreground_data.shape[1:],
                dtype=foreground_data.dtype,
            )
            contour_stack[:, t_idx] = contour_data
            foreground_stack[:, t_idx] = foreground_data
            if _CONTOUR_LAYER in self.viewer.layers:
                self.viewer.layers[_CONTOUR_LAYER].data = contour_stack
            else:
                self.viewer.add_image(contour_stack, name=_CONTOUR_LAYER, colormap="magma", visible=True)
            self._update_labels_layer(_FOREGROUND_SCORE_LAYER, foreground_stack)
            self._files_widget.refresh(self._pos_dir)
            self._status(f"Preview segmentation inputs t={t_idx} — {n_sources} sources")

        @thread_worker(connect={
            "returned": _done, "errored": self._on_contour_worker_error,
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            dp_stack = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis, ...]
            if dp_stack.ndim == 4:
                dp_stack = dp_stack[np.newaxis, ...]
            if prob_stack.ndim != 4:
                raise ValueError("nucleus_prob must be ZxYxX or TxZxYxX.")
            if dp_stack.ndim != 5 or dp_stack.shape[2] != 2:
                raise ValueError("nucleus_dp must be Zx2xYxX or TxZx2xYxX.")
            if prob_stack.shape[0] != dp_stack.shape[0]:
                raise ValueError("nucleus_prob and nucleus_dp must have the same frame count.")
            if prob_stack.shape[1] != dp_stack.shape[1]:
                raise ValueError("nucleus_prob and nucleus_dp must have the same z count.")
            if prob_stack.shape[2:] != dp_stack.shape[3:]:
                raise ValueError("nucleus_prob and nucleus_dp must have the same YxX shape.")

            preview_t = self._preview_frame_from_step(
                current_step,
                prob_stack.shape[0],
                source_time_axes=preview_has_source_time_axes,
            )
            if z_indices is None:
                z_sel = tuple(range(prob_stack.shape[1]))
            elif isinstance(z_indices, slice):
                start = 0 if z_indices.start is None else int(z_indices.start)
                stop = prob_stack.shape[1] if z_indices.stop is None else int(z_indices.stop)
                step = 1 if z_indices.step is None else int(z_indices.step)
                z_sel = tuple(range(start, stop, step))
            else:
                z_sel = tuple(int(z) for z in z_indices)
            bad_z = [z for z in z_sel if z < 0 or z >= prob_stack.shape[1]]
            if bad_z:
                raise ValueError(f"Z indices out of range for {prob_stack.shape[1]} z slices: {bad_z}")
            contours, foreground_scores = build_consensus_boundary(
                prob_stack[preview_t, z_sel],
                dp_stack[preview_t, z_sel],
                list(map_thresholds),
                gamma=1.0,
                flow_threshold=0.0,
            )
            contour_frame, foreground_frame, _, metadata = preview_ultrack_source_stack_frame(
                contours[np.newaxis, ...],
                foreground_scores[np.newaxis, ...],
                contour_thresholds=contour_thresholds,
                foreground_thresholds=foreground_thresholds,
                frame_index=0,
            )
            return contour_frame, foreground_frame, preview_t, prob_stack.shape[0], len(metadata)

        n_sources = len(contour_thresholds) * len(foreground_thresholds)
        preview_has_source_time_axes = self._segmentation_preview_has_source_time_axes()
        preview_axis = 1 if preview_has_source_time_axes and len(current_step) >= 2 else 0
        t_frame = int(current_step[preview_axis]) if current_step else 0
        self._status(
            f"Previewing segmentation inputs for frame t={t_frame} "
            f"({len(map_thresholds)} cellprob thresholds, {n_sources} sources)..."
        )
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
        enabled = mode == "shape"
        self.db_gen_area_weight_spin.setEnabled(enabled)
        self.db_gen_iou_weight_spin.setEnabled(enabled)
        self.db_gen_distance_weight_spin.setEnabled(enabled)

    def _thresholds_from_controls(
        self,
        threshold_min: float,
        threshold_max: float,
        threshold_step: float,
        *,
        label: str,
    ) -> np.ndarray:
        return _thresholds.thresholds_from_values(
            threshold_min,
            threshold_max,
            threshold_step,
            label=label,
        )

    def _source_contour_thresholds_from_controls(self) -> np.ndarray:
        return _thresholds.source_contour_thresholds(self)

    def _source_foreground_thresholds_from_controls(self) -> np.ndarray:
        return _thresholds.source_foreground_thresholds(self)

    def _db_gen_thresholds_from_controls(self) -> np.ndarray:
        return self._source_contour_thresholds_from_controls()

    def _map_cellprob_thresholds_from_controls(self) -> np.ndarray:
        return _thresholds.map_cellprob_thresholds(self)

    def _map_z_indices_from_controls(self) -> list[int] | slice | None:
        return _thresholds.map_z_indices(self)

    def _db_gen_config_from_controls(self) -> UltrackConfig:
        return UltrackConfig(
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

    def _on_run_db_generation(self) -> None:
        if self._pos_dir is None:
            self._status("No project open."); return
        contour_sources_path = self._contour_sources_path()
        foreground_sources_path = self._foreground_sources_path()
        if contour_sources_path is None or not contour_sources_path.exists():
            self._status("Missing: contour_sources.tif — run Build Sources first."); return
        if foreground_sources_path is None or not foreground_sources_path.exists():
            self._status("Missing: foreground_sources.tif — run Build Sources first."); return
        if _ultrack_segment is None:
            self._status("ultrack not installed — activate the cellflow conda environment."); return

        cfg = self._db_gen_config_from_controls()
        working_dir = self._ultrack_workdir()
        pos_dir = self._pos_dir

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
                        build_ultrack_database_from_sources(
                            contour_sources_path=contour_sources_path,
                            foreground_sources_path=foreground_sources_path,
                            working_dir=working_dir,
                            cfg=cfg,
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

    def _on_run_ultrack(self) -> None:
        if self._pos_dir is None:
            self._status("No project open."); return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._status("data.db not found — run DB Generation first."); return
        score_path = self._foreground_scores_path()
        if score_path is None or not score_path.exists():
            self._status("Missing: foreground_scores.tif — build segmentation inputs first."); return
        working_dir = self._ultrack_workdir()
        tracked_path = self._tracked_path()

        cfg = self._ultrack_config_from_controls()
        use_corrections = self.db_gen_use_validated_check.isChecked()
        corrections = read_corrections(self._pos_dir) if use_corrections else None
        validated_tracks = (
            read_validated_tracks(self._pos_dir)
            if use_corrections and not corrections
            else None
        )
        tracked_labels = None
        if corrections or validated_tracks:
            tracked_labels = self._ensure_tracked_layer_data()
            if tracked_labels is None:
                self._status(
                    "Correction-aware solve requires tracked_labels.tif "
                    "(layer not loaded and file not on disk)."
                ); return

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
            yield "Applying annotations and scoring…"
            apply_annotations_and_score(
                working_dir=working_dir,
                cfg=cfg,
                score_signal_path=score_path,
                corrections=corrections,
                validated_tracks=validated_tracks,
                tracked_labels=tracked_labels,
            )
            for step, total, label in run_solve(working_dir, cfg, overwrite=True):
                yield (step, total, f"[solve] {label}")
            yield "Exporting tracked labels…"
            return export_tracked_labels(
                working_dir, cfg, tracked_path,
                corrections=corrections,
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
        self._files_widget.refresh(self._pos_dir)
        self._status(f"Tracking done: {nt} frame(s).")

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

    def _on_ultrack_db_source_changed(self, value: int) -> None:
        """Handle threshold source slider change."""
        if not self._ultrack_db_browser_active:
            return
        # Update label to show current source index
        max_source = self.ultrack_db_source_slider.maximum()
        if max_source > 0:
            self.ultrack_db_source_lbl.setText(f"{value}/{max_source}")
        else:
            self.ultrack_db_source_lbl.setText("all")
        # Clear cache and refresh to show selected source
        self._ultrack_db_preview_cache.clear()
        from qtpy.QtCore import QTimer
        QTimer.singleShot(150, self._refresh_ultrack_db_browser)

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
        self.ultrack_db_source_slider.setEnabled(checked)
        self.ultrack_db_hierarchy_slider.setEnabled(checked)
        self.ultrack_db_prob_alpha_check.setEnabled(checked)
        self.ultrack_db_connected_focus_check.setEnabled(checked)
        self.ultrack_db_edge_alpha_check.setEnabled(checked)
        self.ultrack_db_show_validated_check.setEnabled(checked)
        self.ultrack_db_show_fake_check.setEnabled(checked)
        if checked:
            self.ultrack_db_browser_section.expand()
            self._ultrack_db_frame_initialized = False
            self._refresh_ultrack_db_browser()
        else:
            self._remove_ultrack_db_browser_layers()
            self.ultrack_db_browser_section.collapse()

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
        return _query_ultrack_db_middle_frame(db_path)

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
            # Configure source slider first
            self._configure_ultrack_db_source_slider(db_path, mtime_ns)
            # Then configure hierarchy slider (may depend on selected source)
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
        return _query_ultrack_db_connected_nodes(db_path, selected_node_id)

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
        return _ultrack_db_summary_text(db_path, frame)

    def _query_distinct_heights(self, db_path, mtime_ns):
        key = (str(db_path.resolve()), mtime_ns)
        cached = self._ultrack_db_height_values_cache.get(key)
        if cached is not None: return cached
        heights = _query_distinct_heights(db_path)
        self._ultrack_db_height_values_cache[key] = heights
        return heights

    def _query_hierarchy_cut_states(self, db_path, mtime_ns, frame):
        source_idx = self.ultrack_db_source_slider.value()
        max_source = self.ultrack_db_source_slider.maximum()
        source_key = int(source_idx) if max_source > 0 else None
        key = (str(db_path.resolve()), mtime_ns, frame, source_key)
        cached = self._ultrack_db_cut_state_cache.get(key)
        if cached is not None: return cached
        result = _query_hierarchy_cut_states(db_path, frame, source_index=source_key)
        self._ultrack_db_cut_state_cache[key] = result
        return result

    def _query_available_sources(self, db_path, mtime_ns):
        """Query distinct source indices from merge metadata."""
        key = (str(db_path.resolve()), mtime_ns, "sources")
        cached = self._ultrack_db_sources_cache.get(key)
        if cached is not None:
            return cached
        sources = _query_available_sources(db_path)
        self._ultrack_db_sources_cache[key] = sources
        return sources

    def _configure_ultrack_db_source_slider(self, db_path, mtime_ns):
        """Configure source slider based on available sources in DB."""
        sources = self._query_available_sources(db_path, mtime_ns)
        if not sources:
            self.ultrack_db_source_slider.setRange(0, 0)
            self.ultrack_db_source_lbl.setText("all")
            return False  # Not a multi-source DB
        max_source = max(sources)
        current = min(max(int(self.ultrack_db_source_slider.value()), 0), max_source)
        old = self.ultrack_db_source_slider.blockSignals(True)
        try:
            self.ultrack_db_source_slider.setRange(0, max_source)
            self.ultrack_db_source_slider.setValue(current)
        finally:
            self.ultrack_db_source_slider.blockSignals(old)
        self.ultrack_db_source_lbl.setText(f"{current}/{max_source}")
        return len(sources) > 1  # Multi-source DB

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
        return _render_hierarchy_cut(
            db_path,
            frame,
            h_actual,
            plane_shape=self._viewer_plane_shape(),
            show_validated=self.ultrack_db_show_validated_check.isChecked(),
            show_fake=self.ultrack_db_show_fake_check.isChecked(),
        ).as_tuple()

    def _render_hierarchy_cut_state(self, db_path, frame, state):
        return _render_hierarchy_cut_state(
            db_path,
            frame,
            state,
            plane_shape=self._viewer_plane_shape(),
            show_validated=self.ultrack_db_show_validated_check.isChecked(),
            show_fake=self.ultrack_db_show_fake_check.isChecked(),
        ).as_tuple()

    def _finalize_hierarchy_nodes(self, nodes, frame, *, empty_msg, status_suffix):
        from cellflow.tracking_ultrack.db_query import finalize_hierarchy_nodes

        return finalize_hierarchy_nodes(
            nodes,
            frame,
            plane_shape=self._viewer_plane_shape(),
            show_validated=self.ultrack_db_show_validated_check.isChecked(),
            show_fake=self.ultrack_db_show_fake_check.isChecked(),
            empty_msg=empty_msg,
            status_suffix=status_suffix,
        ).as_tuple()

    @staticmethod
    def _ultrack_db_annotation_name(value):
        return _ultrack_db_annotation_name(value)

    @staticmethod
    def _ultrack_db_node_preview_metadata(nodes):
        return _ultrack_db_node_preview_metadata(nodes)

    @staticmethod
    def _ultrack_db_node_annotation_metadata(nodes):
        return _ultrack_db_node_annotation_metadata(nodes)

    def _empty_ultrack_db_preview(self):
        return np.zeros(self._viewer_plane_shape(), dtype=np.uint32)

    def _viewer_plane_shape(self):
        for layer in self.viewer.layers:
            data = getattr(layer, "data", None)
            if isinstance(data, np.ndarray) and data.ndim >= 2:
                return tuple(int(v) for v in data.shape[-2:])
        return (1, 1)

    def _paint_ultrack_db_nodes(self, nodes):
        return _paint_ultrack_db_nodes(nodes, self._viewer_plane_shape())

    @staticmethod
    def _node_mask_and_bbox(node):
        return _node_mask_and_bbox(node)

    # ================================================================
    # Correction mode helpers
    # ================================================================
    def _correction_tracked_layer(self):
        if _CORRECTION_TRACKED_LAYER in self.viewer.layers:
            return self.viewer.layers[_CORRECTION_TRACKED_LAYER]
        if _TRACKED_LAYER in self.viewer.layers:
            return self.viewer.layers[_TRACKED_LAYER]
        return None

    def _contrast_limits_for_image(self, data: np.ndarray):
        arr = np.asarray(data, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        p_lo, hi = np.percentile(finite, [0.05, 99.5])
        if not (np.isfinite(p_lo) and np.isfinite(hi) and hi > p_lo):
            data_min = float(np.min(finite))
            data_max = float(np.max(finite))
            if data_max > data_min:
                return (data_min, data_max)
            return None
        # Background = mode of the histogram between the 0.05th and 99.5th
        # percentiles. In fluorescence z-avg images this peak is the empty
        # background; using it as the lower contrast limit visually removes
        # it without mutating pixel data. Pad by 1 std of the sub-mode tail
        # so residual background noise stays dark.
        counts, edges = np.histogram(finite, bins=256, range=(float(p_lo), float(hi)))
        mode = float(edges[int(np.argmax(counts))] + (edges[1] - edges[0]) * 0.5)
        below = finite[finite <= mode]
        pad = float(np.std(below)) if below.size > 1 else 0.0
        lo = mode + pad
        if not (np.isfinite(lo) and hi > lo):
            lo = float(p_lo)
        return (float(lo), float(hi))

    def _capture_correction_view_state(self) -> None:
        selected = [layer.name for layer in self.viewer.layers.selection]
        active = self.viewer.layers.selection.active
        self._correction_view_state = {
            "visibility": {layer.name: bool(layer.visible) for layer in self.viewer.layers},
            "active": active.name if active is not None else None,
            "selected": selected,
        }

    def _restore_correction_view_state(self) -> None:
        state = self._correction_view_state or {}
        visibility = state.get("visibility", {})
        for name, visible in visibility.items():
            if name in self.viewer.layers:
                self.viewer.layers[name].visible = bool(visible)
        self.viewer.layers.selection.clear()
        for name in state.get("selected", ()):
            if name in self.viewer.layers:
                self.viewer.layers.selection.add(self.viewer.layers[name])
        active_name = state.get("active")
        if active_name in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[active_name]
        self._correction_view_state = None

    def _remove_correction_owned_layers(self) -> None:
        for name in list(self._correction_owned_layers):
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])
        self._correction_owned_layers.clear()

    def _add_correction_image_layer(self, data: np.ndarray, name: str, colormap: str) -> None:
        arr = np.asarray(data, dtype=np.float32)
        if colormap == "bop_blue":
            colormap = Colormap(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.25, 1.0, 1.0]],
                name="bop_blue",
            )
        kwargs = {"name": name, "colormap": colormap, "blending": "additive"}
        limits = self._contrast_limits_for_image(arr)
        if limits is not None:
            kwargs["contrast_limits"] = limits
        self.viewer.add_image(arr, **kwargs)
        self._correction_owned_layers.add(name)

    def _add_correction_track_layer(self, labels: np.ndarray) -> None:
        labels = np.asarray(labels)
        label_ids = np.asarray(
            sorted(int(v) for v in np.unique(labels) if int(v) != 0)
        )
        label_colors = _categorical_colors(label_ids)
        color_map: dict[int | None, tuple[float, float, float, float] | str] = {
            None: "transparent",
            0: "transparent",
        }
        for label_id, color in zip(label_ids, label_colors, strict=True):
            color_map[int(label_id)] = tuple(float(c) for c in color)

        shape = (
            (1, int(labels.shape[0]), int(labels.shape[1]))
            if labels.ndim == 2
            else tuple(int(v) for v in labels.shape[:3])
        )
        track_image = _rasterize_track_image(
            _nucleus_centroids_by_track(labels),
            color_map,
            shape,
        )
        self.viewer.add_image(
            track_image,
            name=_CORRECTION_TRACK_LAYER,
            rgb=True,
            opacity=0.9,
            blending="additive",
        )
        self._correction_owned_layers.add(_CORRECTION_TRACK_LAYER)

    # ================================================================
    # 4. Correction
    # ================================================================
    def _on_save_tracked(self) -> None:
        tracked_path = self._tracked_path()
        if tracked_path is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer to save."); return
        if layer.data.ndim != 3:
            self._correction_status("Tracked layer is not a 3D stack."); return
        n = layer.data.shape[0]
        for t in range(n):
            write_tracked_frame(tracked_path, t, np.asarray(layer.data[t]))
        self._correction_status(f"Saved {n} frame(s) to {tracked_path.name}.")

    def _load_correction_layers_from_disk(self) -> bool:
        tracked_path = self._tracked_path()
        if tracked_path is None or not tracked_path.exists():
            self._correction_status("No tracked labels file found.")
            return False

        self._remove_correction_owned_layers()
        self._remove_other_correction_prefix_layers()
        stack = read_full_tracked_stack(tracked_path)
        self.viewer.add_labels(stack, name=_CORRECTION_TRACKED_LAYER)
        self._correction_owned_layers.add(_CORRECTION_TRACKED_LAYER)
        self._add_correction_track_layer(stack)

        for path, name, cmap in (
            (self._cell_zavg_path(), _CORRECTION_CELL_ZAVG_LAYER, "gray"),
            (self._nucleus_zavg_path(), _CORRECTION_NUC_ZAVG_LAYER, "bop orange"),
            (self._nls_zavg_path(), _CORRECTION_NLS_ZAVG_LAYER, "bop_blue"),
        ):
            if path is None or not path.exists():
                continue
            self._add_correction_image_layer(
                np.asarray(tifffile.imread(str(path)), dtype=np.float32),
                name,
                cmap,
            )

        self._correction_status(f"Loaded tracked stack {stack.shape} into correction mode.")
        return True

    def _remove_other_correction_prefix_layers(self) -> None:
        for layer in list(self.viewer.layers):
            if layer.name.startswith("[Correction]") and layer.name not in self._correction_owned_layers:
                if isinstance(layer, napari.layers.Labels):
                    self.viewer.layers.remove(layer)

    def _on_load_tracked(self) -> None:
        self._load_correction_layers_from_disk()

    def _on_reassign_ids(self) -> None:
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        stack = np.asarray(layer.data)
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
        layer = self._correction_tracked_layer()
        if layer is not None:
            layer.data = remapped
        if self._pos_dir is not None and old_to_new:
            remap_validated_tracks(self._pos_dir, old_to_new)
        self._correction_status(
            f"Reassigned {n_cells} cell IDs to range 1–{n_cells}. Unsaved."
        )

    def _selected_correction_target(self) -> tuple[int, int, float, float] | None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return None
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return None
        cell_id = int(getattr(self.correction_widget, "_selected_label", 0) or 0)
        if cell_id == 0:
            self._correction_status("No cell selected (left-click first)."); return None
        t = self._current_t()
        data = np.asarray(layer.data)
        frame = self._frame_view_2d(data, t) if data.ndim >= 3 else data
        if frame is None or not np.any(frame == cell_id):
            self._correction_status(f"Cell {cell_id} not present at t={t}."); return None
        yy, xx = np.nonzero(frame == cell_id)
        return cell_id, t, float(np.mean(yy)), float(np.mean(xx))

    def _validated_correction_for_frame(
        self, cell_id: int, t: int, data: np.ndarray
    ) -> Correction | None:
        frame = self._frame_view_2d(data, t) if data.ndim >= 3 else data
        if frame is None or not np.any(frame == cell_id):
            return None
        yy, xx = np.nonzero(frame == cell_id)
        return Correction(
            cell_id=int(cell_id),
            t=int(t),
            kind="validated",
            y=float(np.mean(yy)),
            x=float(np.mean(xx)),
        )

    def _on_validate_track(self) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        cell_id = int(getattr(self.correction_widget, "_selected_label", 0) or 0)
        if cell_id == 0:
            self._correction_status("No cell selected (left-click first)."); return
        data = np.asarray(layer.data)
        frames = self._frames_with_cell(cell_id)
        if not frames:
            self._correction_status(f"Cell {cell_id} not present in tracked labels."); return
        for t in frames:
            correction = self._validated_correction_for_frame(cell_id, t, data)
            if correction is not None:
                add_correction(self._pos_dir, correction)
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        self._correction_status(
            f"Validated track {cell_id} across {len(frames)} frame(s)."
        )

    def _on_anchor_here(self) -> None:
        target = self._selected_correction_target()
        if target is None or self._pos_dir is None:
            return
        cell_id, t, y, x = target
        corrections = read_corrections(self._pos_dir)
        remaining = [
            correction
            for correction in corrections
            if not (
                int(correction.cell_id) == int(cell_id)
                and int(correction.t) == int(t)
                and correction.kind == "anchor"
            )
        ]
        if len(remaining) != len(corrections):
            write_corrections(self._pos_dir, remaining)
            self._refresh_validated_overlay()
            self._correction_status(f"Unanchored cell {cell_id} at t={t}.")
            return
        layer = self._correction_tracked_layer()
        if layer is None:
            add_correction(
                self._pos_dir,
                Correction(cell_id=cell_id, t=t, kind="anchor", y=y, x=x),
            )
            self._refresh_validated_overlay()
            self._correction_status(f"Anchored cell {cell_id} at t={t}.")
            return
        filled = add_anchor(
            self._pos_dir,
            cell_id=cell_id,
            t=t,
            y=y,
            x=x,
            tracked_labels=np.asarray(layer.data),
        )
        self._refresh_validated_overlay()
        suffix = f" (gap-filled {filled} frame(s))" if filled else ""
        self._correction_status(f"Anchored cell {cell_id} at t={t}.{suffix}")

    def _on_extend_backward(self) -> None:
        self._on_extend(direction="backward")

    def _on_extend_forward(self) -> None:
        self._on_extend(direction="forward")

    def _on_extend(self, direction: str) -> None:
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._correction_status("Extend: data.db not found — run DB Generation first."); return
        source_id = self.correction_widget._selected_label
        if not source_id:
            self._correction_status("Extend: no cell selected (left-click first)."); return

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

        # Build a mask of all validated *and* anchored cells in the target frame;
        # greedy overwrite must respect both kinds of pin.
        protected_ids_at_target: set[int] = set()
        for cell_id, frames in validated_tracks.items():
            if result.target_frame in frames:
                protected_ids_at_target.add(cell_id)
        if self._pos_dir is not None:
            for c in read_corrections(self._pos_dir):
                if c.kind == "anchor" and int(c.t) == int(result.target_frame):
                    protected_ids_at_target.add(int(c.cell_id))
        protected_mask = np.zeros_like(frame, dtype=bool)
        for pid in protected_ids_at_target:
            protected_mask |= (frame == pid)

        changed_ids = {int(a.cell_id) for a in assignments}
        for cid in changed_ids:
            frame[frame == cid] = 0
        if self.extend_greedy_overwrite_check.isChecked():
            for a in assignments:
                frame[a.mask_2d & ~protected_mask] = int(a.cell_id)
        else:
            for a in assignments:
                frame[a.mask_2d & (frame == 0)] = int(a.cell_id)
        layer.refresh()

        step = list(self.viewer.dims.current_step)
        step[0] = result.target_frame
        self.viewer.dims.current_step = tuple(step)

        self._correction_status(
            f"Extended cell {source_id} → t={result.target_frame} "
            f"(dist={result.centroid_distance:.1f}px, area={result.area_ratio:.2f}, "
            f"iou={result.centroid_corrected_iou:.2f}, overlap={result.existing_overlap:.2f})"
        )

    def _on_swap_step(self, direction: str) -> None:
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._correction_status("data.db not found — run DB Generation first."); return
        source_id = self.correction_widget._selected_label
        if not source_id:
            self._correction_status("Swap: no cell selected (left-click first)."); return

        t = self._current_t()
        tracked = np.asarray(layer.data)
        source_mask = tracked[t] == source_id
        if not source_mask.any():
            self._correction_status(f"Cell {source_id} not present at t={t}."); return

        validated_tracks = (
            read_validated_tracks(self._pos_dir) if self._pos_dir is not None else {}
        )
        if source_id in validated_tracks:
            self._correction_status("Cannot swap a validated cell."); return

        if self._swap_cursor is None:
            from skimage.measure import regionprops as _regionprops
            props = _regionprops(source_mask.astype(np.uint8))
            if not props:
                self._correction_status("Cannot compute centroid for source cell."); return
            src_cy, src_cx = props[0].centroid
            src_area = int(props[0].area)
            source_centroid = (float(src_cy), float(src_cx))

            protected_ids: set[int] = set()
            for cell_id, frames in validated_tracks.items():
                if t in frames and cell_id != source_id:
                    protected_ids.add(cell_id)
            if self._pos_dir is not None:
                for c in read_corrections(self._pos_dir):
                    if c.kind == "anchor" and int(c.t) == t and int(c.cell_id) != source_id:
                        protected_ids.add(int(c.cell_id))
            protected_mask = (
                np.isin(tracked[t], list(protected_ids))
                if protected_ids
                else np.zeros(tracked.shape[1:], dtype=bool)
            )

            radius_px = float(self.swap_radius_spin.value())
            candidates = list_swap_candidates(
                db_path=db_path,
                frame=t,
                source_centroid=source_centroid,
                radius_px=radius_px,
                frame_shape=tuple(tracked.shape[1:]),
                protected_mask=protected_mask,
            )
            if not candidates:
                self._correction_status(
                    f"No swap candidates within {radius_px:g}px."
                ); return

            self._swap_cursor = _SwapCursor(
                source_id=source_id,
                frame=t,
                source_centroid=source_centroid,
                source_area=src_area,
                candidates=tuple(candidates),
                displayed_area=src_area,
                cursor=None,
            )

        cursor = self._swap_cursor
        if direction == "smaller":
            idx = _step_smaller(cursor.candidates, cursor.displayed_area)
            no_move_msg = "No smaller candidate."
        else:
            idx = _step_larger(cursor.candidates, cursor.displayed_area)
            no_move_msg = "No larger candidate."

        if idx is None:
            self._correction_status(no_move_msg); return

        candidate = cursor.candidates[idx]
        validated_tracks_full = (
            read_validated_tracks(self._pos_dir) if self._pos_dir is not None else {}
        )
        self._apply_swap(layer, t, source_id, candidate, validated_tracks_full)
        cursor.cursor = idx
        cursor.displayed_area = candidate.area
        self._correction_status(
            f"Swapped cell {source_id} → candidate {idx + 1}/{len(cursor.candidates)}"
            f" (area={candidate.area} px)"
        )

    def _apply_swap(self, layer, t: int, source_id: int, candidate: _SwapCandidate, validated_tracks: dict) -> None:
        frame = layer.data[t]
        before = frame.copy()

        protected_ids: set[int] = set()
        for cell_id, frames in validated_tracks.items():
            if t in frames and cell_id != source_id:
                protected_ids.add(cell_id)
        if self._pos_dir is not None:
            for c in read_corrections(self._pos_dir):
                if c.kind == "anchor" and int(c.t) == t and int(c.cell_id) != source_id:
                    protected_ids.add(int(c.cell_id))
        protected_mask = (
            np.isin(frame, list(protected_ids))
            if protected_ids
            else np.zeros_like(frame, dtype=bool)
        )

        frame[frame == source_id] = 0
        paintable = candidate.mask_2d & ~protected_mask
        frame[paintable] = source_id

        self.correction_widget._record_history(layer, t, before)
        layer.refresh()

    def _on_retrack_forward(self) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
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
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
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
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
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
            ("B", self._on_anchor_here),
            ("S", self._on_save_tracked),
            ("Z", lambda: self._on_swap_step(direction="smaller")),
            ("C", lambda: self._on_swap_step(direction="larger")),
        ]
        self._correction_shortcuts: list[QShortcut] = []
        for key, slot in specs:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.setEnabled(False)
            sc.activated.connect(slot)
            self._correction_shortcuts.append(sc)

    def _on_correction_active_button_toggled(self, active: bool) -> None:
        if active:
            self._capture_correction_view_state()
            for layer in list(self.viewer.layers):
                layer.visible = False

            if not self._load_correction_layers_from_disk():
                self._restore_correction_view_state()
                old = self.correction_active_btn.blockSignals(True)
                try:
                    self.correction_active_btn.setChecked(False)
                finally:
                    self.correction_active_btn.blockSignals(old)
                self.correction_widget.deactivate()
                self.correction_mode_section.collapse()
                if hasattr(self, "refinement_widget"):
                    self.refinement_widget.refresh()
                return
            layer = self.viewer.layers[_CORRECTION_TRACKED_LAYER]
            layer.visible = True
            self.viewer.layers.selection.active = layer
            self.correction_widget.activate_layer(layer)
            self.correction_mode_section.expand()
            if hasattr(self, "refinement_widget"):
                self.refinement_widget.refresh()
            return

        self.correction_widget.deactivate()
        for sc in getattr(self, "_correction_shortcuts", []):
            sc.setEnabled(False)
        self._remove_correction_owned_layers()
        self._restore_correction_view_state()
        self.correction_mode_section.collapse()
        if hasattr(self, "refinement_widget"):
            self.refinement_widget.refresh()

    def _on_correction_mode_toggled(self, active: bool) -> None:
        if not active:
            self._swap_cursor = None
        for sc in self._correction_shortcuts:
            sc.setEnabled(active)
        self.correction_mode_section.expand() if active else self.correction_mode_section.collapse()

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
            layer = self._correction_tracked_layer()
            if layer is None:
                return
            data = np.asarray(layer.data)
            for frame in frames:
                correction = self._validated_correction_for_frame(sel, frame, data)
                if correction is not None:
                    add_correction(self._pos_dir, correction)
            self._correction_status(
                f"Cell {sel} validated across {len(frames)} frame(s)."
            )
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _on_dims_step_changed(self, event=None) -> None:
        self._swap_cursor = None
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        if self.ultrack_db_browser_section.is_expanded:
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, self._refresh_ultrack_db_browser)

    def _refresh_validated_overlay(self) -> None:
        self._validated_overlay.refresh_overlay(self._frame_view_2d)

    def _add_validated_overlay(self, data: np.ndarray) -> None:
        self._validated_overlay.add_overlay(data)

    def _place_validated_overlay_below_spotlight(self) -> None:
        self._validated_overlay.place_below_spotlight()

    def _refresh_validation_counter(self) -> None:
        self._validated_overlay.refresh_counter(self.validation_counter_lbl)

    def _on_cells_edited(self, t: int, changed_ids: set[int]) -> None:
        self._validated_overlay.on_cells_edited(
            t,
            changed_ids,
            frame_view_2d=self._frame_view_2d,
            counter_label=self.validation_counter_lbl,
        )

    def _frames_with_cell(self, cell_id: int) -> list[int]:
        return self._validated_overlay.frames_with_cell(cell_id)
