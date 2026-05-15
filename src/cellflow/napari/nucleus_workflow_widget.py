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

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from cellflow.correction.labels import best_overlapping_label
from cellflow.database.validation import (
    read_corrections,
    read_validated_tracks,
)
from cellflow.napari.nucleus_correction_widget import NucleusCorrectionWidget
from cellflow.napari.nucleus_db_browser_widget import (
    NucleusUltrackDbBrowserMixin,
    _HierarchyCutState,  # noqa: F401 - legacy module-level test helper export
)
from cellflow.napari.nucleus_segmentation_inputs_widget import (
    NucleusSegmentationInputsWidget,
)
from cellflow.napari.radial_refinement_widget import RadialRefinementWidget
from cellflow.napari._paths import NucleusArtifactPaths
from cellflow.napari._state import dump_state, load_state
from cellflow.napari import _thresholds
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
    block_grid,
    compact_spinbox,
)
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.segmentation import build_consensus_boundary, build_nucleus_averaged_maps
from cellflow.tracking_ultrack.config import TrackingConfig as UltrackConfig
from cellflow.tracking_ultrack.db_build import apply_annotations_and_score
from cellflow.tracking_ultrack.export import export_tracked_labels
from cellflow.tracking_ultrack.extend import extend_track_from_db  # noqa: F401
from cellflow.tracking_ultrack.ingest import _select_solver
from cellflow.tracking_ultrack.multi_threshold import (
    build_ultrack_database_from_sources,
    preview_ultrack_source_stack_frame,
    write_ultrack_source_stacks,
)
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
# Correction-owned layer constants
_CORRECTION_TRACKED_LAYER = "[Correction] Tracked: Nucleus"
_CORRECTION_TRACK_LAYER = "[Correction] Nucleus tracks"
_CORRECTION_CELL_ZAVG_LAYER = "[Correction] Cell z-avg"
_CORRECTION_NUC_ZAVG_LAYER = "[Correction] Nucleus z-avg"
_CORRECTION_NLS_ZAVG_LAYER = "[Correction] NLS z-avg"


# ══════════════════════════════════════════════════════════════════════════════


class NucleusWorkflowWidget(NucleusUltrackDbBrowserMixin, QWidget):
    """Nucleus hypothesis generation and tracking — flat action-button layout."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._stop_flag: bool = False

        self._contour_worker = None
        self._db_gen_worker = None
        self._ultrack_worker = None

        self._init_ultrack_db_browser_state()

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
        self.nucleus_segmentation_inputs_widget = NucleusSegmentationInputsWidget(self)
        segmentation_inputs = self.nucleus_segmentation_inputs_widget
        self.segmentation_inputs_parameters_section = segmentation_inputs.section
        self.segmentation_inputs_section = segmentation_inputs.section
        self.map_cellprob_min_spin = segmentation_inputs.map_cellprob_min_spin
        self.map_cellprob_max_spin = segmentation_inputs.map_cellprob_max_spin
        self.map_cellprob_step_spin = segmentation_inputs.map_cellprob_step_spin
        self.map_z_start_spin = segmentation_inputs.map_z_start_spin
        self.map_z_stop_spin = segmentation_inputs.map_z_stop_spin
        self.map_z_step_spin = segmentation_inputs.map_z_step_spin
        self.source_contour_threshold_min_spin = (
            segmentation_inputs.source_contour_threshold_min_spin
        )
        self.source_contour_threshold_max_spin = (
            segmentation_inputs.source_contour_threshold_max_spin
        )
        self.source_contour_threshold_step_spin = (
            segmentation_inputs.source_contour_threshold_step_spin
        )
        self.source_foreground_threshold_min_spin = (
            segmentation_inputs.source_foreground_threshold_min_spin
        )
        self.source_foreground_threshold_max_spin = (
            segmentation_inputs.source_foreground_threshold_max_spin
        )
        self.source_foreground_threshold_step_spin = (
            segmentation_inputs.source_foreground_threshold_step_spin
        )
        self.db_gen_threshold_min_spin = self.source_contour_threshold_min_spin
        self.db_gen_threshold_max_spin = self.source_contour_threshold_max_spin
        self.db_gen_threshold_step_spin = self.source_contour_threshold_step_spin

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
        root.addWidget(self.segmentation_inputs_section)

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
        self.nucleus_correction_widget = NucleusCorrectionWidget(
            self.viewer,
            pos_dir_provider=lambda: self._pos_dir,
            refresh_refinement_callback=lambda: self.refinement_widget.refresh(),
            parent=self,
        )
        self._alias_correction_controls()
        root.addWidget(self.correction_active_btn)
        root.addWidget(self.correction_mode_section)

    def _alias_correction_controls(self) -> None:
        correction = self.nucleus_correction_widget
        self.correction_active_btn = correction.active_btn
        self.save_tracked_btn = correction.save_tracked_btn
        self.extend_back_btn = correction.extend_back_btn
        self.extend_fwd_btn = correction.extend_fwd_btn
        self.retrack_back_btn = correction.retrack_back_btn
        self.retrack_fwd_btn = correction.retrack_fwd_btn
        self.reassign_ids_btn = correction.reassign_ids_btn
        self.validate_track_btn = correction.validate_track_btn
        self.anchor_here_btn = correction.anchor_here_btn
        self.remove_unvalidated_btn = correction.remove_unvalidated_btn
        self.correction_status_lbl = correction.status_lbl
        self.validation_counter_lbl = correction.validation_counter_lbl
        self.extend_max_dist_spin = correction.extend_max_dist_spin
        self.extend_area_weight_spin = correction.extend_area_weight_spin
        self.extend_iou_weight_spin = correction.extend_iou_weight_spin
        self.extend_distance_weight_spin = correction.extend_distance_weight_spin
        self.extend_overlap_penalty_spin = correction.extend_overlap_penalty_spin
        self.extend_greedy_overwrite_check = correction.extend_greedy_overwrite_check
        self.swap_radius_spin = correction.swap_radius_spin
        self.retrack_max_dist_spin = correction.retrack_max_dist_spin
        self.extend_retrack_params_section = correction.extend_retrack_params_section
        self.extend_params_section = correction.extend_params_section
        self.retrack_params_section = correction.retrack_params_section
        self.correction_widget = correction.correction_widget
        self.correction_shortcuts_section = correction.shortcuts_section
        self.artifact_cleanup_section = correction.artifact_cleanup_section
        self.correction_mode_section = correction.section
        self._correction_owned_layers = correction._correction_owned_layers
        self._validated_overlay = correction._validated_overlay
        for name in (
            "_correction_tracked_layer",
            "_correction_status",
            "_capture_correction_view_state",
            "_restore_correction_view_state",
            "_remove_correction_owned_layers",
            "_add_correction_image_layer",
            "_add_correction_track_layer",
            "_on_save_tracked",
            "_load_correction_layers_from_disk",
            "_remove_other_correction_prefix_layers",
            "_on_load_tracked",
            "_on_reassign_ids",
            "_on_reassign_ids_done",
            "_selected_correction_target",
            "_validated_correction_for_frame",
            "_on_validate_track",
            "_on_anchor_here",
            "_on_extend_backward",
            "_on_extend_forward",
            "_on_extend",
            "_on_swap_step",
            "_apply_swap",
            "_on_retrack_forward",
            "_on_retrack_backward",
            "_on_remove_unvalidated_labels",
            "_on_correction_worker_error",
            "_install_correction_shortcuts",
            "_on_correction_active_button_toggled",
            "_on_correction_mode_toggled",
            "_kb_toggle_cell_validation",
            "_refresh_validated_overlay",
            "_add_validated_overlay",
            "_place_validated_overlay_below_spotlight",
            "_refresh_validation_counter",
            "_on_cells_edited",
            "_frames_with_cell",
        ):
            setattr(self, name, getattr(correction, name))

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

        # Viewer events & keyboard
        self.viewer.dims.events.current_step.connect(self._on_dims_step_changed)
        self.viewer.bind_key("V", self._kb_toggle_cell_validation, overwrite=True)
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
            self.nucleus_correction_widget._swap_cursor = None
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
        self._update_tracked_display(labels)
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
    # Correction / DB browser coordination
    # ================================================================
    def _on_dims_step_changed(self, event=None) -> None:
        self.nucleus_correction_widget.on_dims_step_changed()
        if self.ultrack_db_browser_section.is_expanded:
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, self._refresh_ultrack_db_browser)
