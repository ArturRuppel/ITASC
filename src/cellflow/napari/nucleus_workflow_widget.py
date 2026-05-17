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
from pathlib import Path

import napari
import numpy as np
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

from cellflow.correction.labels import best_overlapping_label
from cellflow.napari.nucleus_correction_widget import NucleusCorrectionWidget
from cellflow.napari.nucleus_db_browser_widget import (
    NucleusUltrackDbBrowserMixin,
    _HierarchyCutState,  # noqa: F401 - legacy module-level test helper export
)
from cellflow.napari.nucleus_pipeline_widget import NucleusPipelineWidget
from cellflow.napari.nucleus_segmentation_inputs_widget import (
    NucleusSegmentationInputsWidget,
)
from cellflow.napari.nucleus_tracking_inputs_widget import NucleusTrackingInputsWidget
from cellflow.napari.radial_refinement_widget import RadialRefinementWidget
from cellflow.napari._paths import NucleusArtifactPaths
from cellflow.napari._state import dump_state, load_state
from cellflow.napari.widgets import (
    CollapsibleSection,
    PipelineFilesWidget,
    make_pipeline_files_header,
)
from cellflow.tracking_ultrack.extend import extend_track_from_db  # noqa: F401
from cellflow.tracking_ultrack.ingest import _select_solver

logger = logging.getLogger(__name__)

# ── Layer name constants ──────────────────────────────────────────────────────
_TRACKED_LAYER = "Tracked: Nucleus"


# ══════════════════════════════════════════════════════════════════════════════


class NucleusWorkflowWidget(NucleusUltrackDbBrowserMixin, QWidget):
    """Nucleus hypothesis generation and tracking — flat action-button layout."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._stop_flag: bool = False

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
                    ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
                    ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
                    ("1_cellpose/cell_prob_zavg.tif", "Cell prob z-avg"),
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
        self._pipeline_files_section = CollapsibleSection(
            "Pipeline Files",
            self._files_widget,
            expanded=False,
        )
        (
            self.pipeline_files_header,
            self.pipeline_files_header_lbl,
            self.pipeline_files_toggle_btn,
        ) = make_pipeline_files_header(
            self._pipeline_files_section,
            stage_key="nucleus",
            parent=self,
        )
        root.addWidget(self.pipeline_files_header)
        root.addWidget(self._pipeline_files_section)

        # ── Workflow sections ────────────────────────────────────────
        self._build_segmentation_inputs_section(root)
        self._build_tracking_ultrack_section(root)

        # ── Ultrack Database Browser ─────────────────────────────────
        self._build_db_browser_section(root)

        # ── Refinement (deprecated; widget no longer rendered) ───────
        # self._build_refinement_section(root)

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
        self.map_z_range = segmentation_inputs.map_z_range
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
        # Range-slider widgets (used by tests/state when the underlying widget
        # itself is needed rather than a per-thumb proxy).
        self.map_cellprob_range = segmentation_inputs.map_cellprob_range
        self.source_contour_threshold_range = (
            segmentation_inputs.source_contour_threshold_range
        )
        self.source_foreground_threshold_range = (
            segmentation_inputs.source_foreground_threshold_range
        )
        self.nucleus_segmentation_inputs_widget.hide()

    def _build_tracking_ultrack_section(self, root: QVBoxLayout) -> None:
        self.nucleus_tracking_inputs_widget = NucleusTrackingInputsWidget(self)
        self._alias_tracking_inputs_controls()

        self.nucleus_pipeline_widget = NucleusPipelineWidget(
            self.viewer,
            pos_dir_provider=lambda: self._pos_dir,
            seg_inputs_provider=lambda: self.nucleus_segmentation_inputs_widget,
            tracking_inputs_provider=lambda: self.nucleus_tracking_inputs_widget,
            refresh_files_callback=lambda pos: self._files_widget.refresh(pos),
            refresh_db_browser_callback=lambda: self._refresh_ultrack_db_browser(),
            parent=self,
        )
        self._alias_pipeline_controls()
        self.nucleus_tracking_inputs_widget.hide()
        self.nucleus_pipeline_widget.hide()

        # Hide the built-in section headers — the stage-row ⚙ buttons drive
        # expand/collapse instead.
        self.segmentation_inputs_section.set_header_visible(False)
        self.tracking_db_section.set_header_visible(False)
        self.tracking_solve_section.set_header_visible(False)

        # Collapse all three inline params blocks by default.
        self.segmentation_inputs_section.collapse()
        self.tracking_db_section.collapse()
        self.tracking_solve_section.collapse()

        pipeline_block = self.nucleus_pipeline_widget.build_pipeline_block(
            seg_section=self.segmentation_inputs_section,
            db_section=self.tracking_db_section,
            solve_section=self.tracking_solve_section,
        )
        root.addWidget(pipeline_block)
        root.addWidget(self.pipeline_status_lbl)
        root.addWidget(self.pipeline_progress_bar)

    def _alias_tracking_inputs_controls(self) -> None:
        ti = self.nucleus_tracking_inputs_widget
        self.tracking_db_section = ti.db_section
        self.tracking_solve_section = ti.solve_section
        self.db_gen_min_area_spin = ti.db_gen_min_area_spin
        self.db_gen_max_area_spin = ti.db_gen_max_area_spin
        self.db_gen_min_frontier_spin = ti.db_gen_min_frontier_spin
        self.db_gen_ws_hierarchy_combo = ti.db_gen_ws_hierarchy_combo
        self.db_gen_n_workers_spin = ti.db_gen_n_workers_spin
        self.db_gen_max_dist_spin = ti.db_gen_max_dist_spin
        self.db_gen_max_neighbors_spin = ti.db_gen_max_neighbors_spin
        self.db_gen_linking_mode_combo = ti.db_gen_linking_mode_combo
        self.db_gen_area_weight_spin = ti.db_gen_area_weight_spin
        self.db_gen_iou_weight_spin = ti.db_gen_iou_weight_spin
        self.db_gen_distance_weight_spin = ti.db_gen_distance_weight_spin
        self.db_gen_quality_weight_spin = ti.db_gen_quality_weight_spin
        self.db_gen_quality_exp_spin = ti.db_gen_quality_exp_spin
        self.db_gen_circularity_weight_spin = ti.db_gen_circularity_weight_spin
        self.db_gen_use_validated_check = ti.db_gen_use_validated_check
        self.ultrack_max_partitions_spin = ti.ultrack_max_partitions_spin
        self.ultrack_n_frames_spin = ti.ultrack_n_frames_spin
        self.ultrack_appear_spin = ti.ultrack_appear_spin
        self.ultrack_disappear_spin = ti.ultrack_disappear_spin
        self.ultrack_division_spin = ti.ultrack_division_spin
        self.ultrack_power_spin = ti.ultrack_power_spin
        self.ultrack_bias_spin = ti.ultrack_bias_spin
        self.ultrack_solver_lbl = ti.ultrack_solver_lbl

    def _alias_pipeline_controls(self) -> None:
        pl = self.nucleus_pipeline_widget
        self.preview_contour_btn = pl.preview_contour_btn
        self.seg_preview_btn = pl.seg_preview_btn
        self.seg_params_btn = pl.seg_params_btn
        self.seg_run_btn = pl.seg_run_btn
        self.db_params_btn = pl.db_params_btn
        self.db_run_btn = pl.db_run_btn
        self.solve_params_btn = pl.solve_params_btn
        self.solve_run_btn = pl.solve_run_btn
        self.pipeline_status_lbl = pl.pipeline_status_lbl
        self.pipeline_progress_bar = pl.pipeline_progress_bar
        for name in (
            "_on_build_segmentation_inputs",
            "_on_build_nucleus_maps",
            "_on_build_contour_maps",
            "_on_preview_contour_maps",
            "_on_contour_worker_error",
            "_on_run_db_generation",
            "_on_db_gen_done",
            "_on_db_gen_worker_error",
            "_on_run_ultrack",
            "_on_ultrack_progress",
            "_on_run_ultrack_done",
            "_on_ultrack_worker_error",
            "_on_cancel",
            "_status",
            "_progress",
            "_on_progress",
            "_clear_progress",
            "_set_pipeline_buttons_enabled",
            "_set_running_stage",
        ):
            setattr(self, name, getattr(pl, name))

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
            refresh_refinement_callback=lambda: None,
            parent=self,
        )
        self._alias_correction_controls()
        # NucleusCorrectionWidget acts as a controller here. Its visible controls
        # are reparented into this layout below, so keep the owner widget hidden
        # to prevent its default geometry from intercepting header clicks.
        self.nucleus_correction_widget.hide()
        root.addWidget(self.ultrack_db_browser_header)
        root.addWidget(self.ultrack_db_browser_section)
        root.addWidget(self.correction_header)
        root.addWidget(self.correction_mode_section)

    def _alias_correction_controls(self) -> None:
        correction = self.nucleus_correction_widget
        self.correction_header = correction.header
        self.correction_header_lbl = correction.header_lbl
        self.correction_params_btn = correction.params_btn
        self.correction_active_btn = correction.active_btn
        self.correction_toolbar = correction.toolbar
        self.save_tracked_btn = correction.save_tracked_btn
        self.extend_back_btn = correction.extend_back_btn
        self.extend_fwd_btn = correction.extend_fwd_btn
        self.retrack_back_btn = correction.retrack_back_btn
        self.retrack_fwd_btn = correction.retrack_fwd_btn
        self.reassign_ids_btn = correction.reassign_ids_btn
        self.validate_track_btn = correction.validate_track_btn
        self.anchor_here_btn = correction.anchor_here_btn
        self.remove_unvalidated_btn = correction.remove_unvalidated_btn
        self.commit_btn = correction.commit_btn
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
            "_commit_reassign_ids",
            "_on_commit",
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
            "_remove_unvalidated_from_layer",
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

        # Qt 5 skips shortcuts whose parent widget is hidden, even for
        # ApplicationShortcut context.  Reparent them to this widget (which
        # stays visible) so they continue to fire once correction mode is
        # activated.
        for sc in correction._correction_shortcuts:
            sc.setParent(self)

    # ================================================================
    # Signals
    # ================================================================
    def _connect_signals(self) -> None:
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

    def _ultrack_db_path(self) -> Path | None:
        # Required by NucleusUltrackDbBrowserMixin.
        return self._paths.ultrack_db if self._paths else None

    def _nucleus_zavg_path(self) -> Path | None:
        # Required by NucleusUltrackDbBrowserMixin.
        return self._paths.nucleus_zavg if self._paths else None

    # These delegate to the pipeline widget so that tests that call path helpers
    # on the workflow widget (legacy seam tests) continue to pass.
    def _contours_path(self) -> Path | None:
        return self.nucleus_pipeline_widget._contours_path()

    def _contour_maps_path(self) -> Path | None:
        return self._contours_path()

    def _contour_sources_path(self) -> Path | None:
        return self.nucleus_pipeline_widget._contour_sources_path()

    def _foreground_sources_path(self) -> Path | None:
        return self.nucleus_pipeline_widget._foreground_sources_path()

    def _foreground_scores_path(self) -> Path | None:
        return self.nucleus_pipeline_widget._foreground_scores_path()

    # ================================================================
    # Public API
    # ================================================================
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._files_widget.refresh(pos_dir)
        self._refresh_z_extent_from_inputs(pos_dir)
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

    def _refresh_z_extent_from_inputs(self, pos_dir: Path | None) -> None:
        """Read the z dimension of the 3D+t cellpose probability stack
        and update the segmentation widget's z range slider accordingly."""
        if pos_dir is None:
            return
        prob_path = pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif"
        if not prob_path.exists():
            return
        try:
            import tifffile
            with tifffile.TiffFile(prob_path) as tf:
                shape = tf.series[0].shape
        except Exception:
            logger.debug("Failed to read z extent from %s", prob_path, exc_info=True)
            return
        # nucleus_prob_3dt is (T, Z, Y, X) so z size is shape[-3].
        if len(shape) < 3:
            return
        z_size = int(shape[-3])
        self.nucleus_segmentation_inputs_widget.set_z_extent(z_size)

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
    # Viewer helpers (correction-owned)
    # ================================================================
    def _current_t(self) -> int:
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) >= 1 else 0

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

    def _set_viewer_frame(self, t: int) -> None:
        step = list(self.viewer.dims.current_step)
        if not step:
            return
        step[0] = int(t)
        self.viewer.dims.current_step = tuple(step)

    # ================================================================
    # Backward-compat delegates (tests call these on the workflow widget)
    # ================================================================
    def _db_gen_thresholds_from_controls(self):
        return self.nucleus_pipeline_widget._source_contour_thresholds_from_controls()

    def _db_gen_config_from_controls(self):
        return self.nucleus_pipeline_widget._db_gen_config_from_controls()

    def _ultrack_config_from_controls(self):
        return self.nucleus_pipeline_widget._ultrack_config_from_controls()

    def _map_z_indices_from_controls(self):
        return self.nucleus_pipeline_widget._map_z_indices_from_controls()

    # Pipeline handler methods are owned by NucleusPipelineWidget and aliased
    # onto this instance by _alias_pipeline_controls() during __init__.


    # ================================================================
    # Correction / DB browser coordination
    # ================================================================
    def _on_dims_step_changed(self, event=None) -> None:
        self.nucleus_correction_widget.on_dims_step_changed()
        if self.ultrack_db_browser_section.is_expanded:
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, self._refresh_ultrack_db_browser)
