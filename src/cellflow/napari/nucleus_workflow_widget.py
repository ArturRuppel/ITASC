"""Nucleus workflow widget for candidate generation and tracking in CellFlow.

Simplified workflow layout with action buttons grouped into their owning
sections: segmentation inputs, tracking/Ultrack, database browser, correction.

Stages:
  1. Cellpose maps → ``nucleus_contours.tif`` / ``nucleus_foreground.tif``
  2. Source sweep preview → in-memory napari layers
  3. Ultrack database + solve → ``data.db`` / ``tracked_labels.tif``
  4. Correction (load / save / extend / retrack / reassign / remove unvalidated)
"""
from __future__ import annotations

import logging
from pathlib import Path

import napari
import numpy as np
from qtpy.QtCore import QSettings, QTimer
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QSizePolicy,
    QWidget,
)

from cellflow.correction.labels import best_overlapping_label
from cellflow.napari.nucleus_correction_widget import NucleusCorrectionWidget
from cellflow.napari.nucleus_atom_extraction_widget import (
    NucleusAtomExtractionMixin,
    NucleusAtomExtractionWidget,
)
from cellflow.napari.nucleus_db_browser_widget import (
    NucleusUltrackDbBrowserMixin,
)
from cellflow.napari.nucleus_pipeline_widget import NucleusPipelineWidget
from cellflow.napari.nucleus_segmentation_inputs_widget import (
    NucleusSegmentationInputsWidget,
)
from cellflow.napari.nucleus_tracking_inputs_widget import NucleusTrackingInputsWidget
from cellflow.napari.ui_gate import ControlClass, UiGate
from cellflow.napari._paths import NucleusWorkspace
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

_NUCLEUS_PIPELINE_FILE_GROUPS = [
    ("Inputs", [
        ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
        ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
        ("1_cellpose/nucleus_contours.tif", "Nucleus contours"),
        ("1_cellpose/nucleus_foreground.tif", "Nucleus foreground"),
    ]),
    ("Intermediates", [
        ("2_nucleus/atoms.tif", "Atoms"),
        ("2_nucleus/ultrack_workdir/data.db", "Ultrack database"),
    ]),
    ("Output", [
        ("2_nucleus/tracked_labels.tif", "Tracked labels"),
    ]),
]


# ══════════════════════════════════════════════════════════════════════════════


class NucleusWorkflowWidget(NucleusUltrackDbBrowserMixin, NucleusAtomExtractionMixin, QWidget):
    """Nucleus candidate generation and tracking — flat action-button layout."""

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: QWidget | None = None,
        gate: UiGate | None = None,
        standalone: bool = False,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        #: Standalone mode: the piece runs on its own (own working-directory
        #: picker + config), instead of being driven by the orchestrator.
        self._standalone = standalone
        #: App-wide UI gate. A private gate is created when none is injected so
        #: the widget still works standalone (tests, isolated use).
        self.gate = gate if gate is not None else UiGate(self)
        #: Position directory (orchestrated mode) — drives the staged files panel.
        self._pos_dir: Path | None = None
        #: The active nucleus workspace (artifacts + annotation store). Derived
        #: from ``_pos_dir`` in orchestrated mode; set directly to a flat working
        #: directory by the standalone entry point.
        self._workspace: NucleusWorkspace | None = None
        self._stop_flag: bool = False
        self._dims_step_refresh_pending: bool = False

        self._init_ultrack_db_browser_state()
        self._init_atom_extraction_state()

        self._setup_ui()
        self._connect_signals()
        self._register_gate_controls()

        if self._standalone:
            self._load_standalone_settings()

    # ================================================================
    # UI
    # ================================================================
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # ── Standalone working-directory picker ───────────────────────
        # Only shown when the piece runs on its own; the orchestrator drives the
        # workspace through refresh()/set_context() instead.
        self._workdir_container = QWidget()
        workdir_row = QHBoxLayout(self._workdir_container)
        workdir_row.setContentsMargins(0, 0, 0, 0)
        workdir_row.addWidget(QLabel("Working dir:"))
        self._workdir_edit = QLineEdit()
        self._workdir_edit.setPlaceholderText(
            "Folder with foreground.tif + contours.tif"
        )
        self._workdir_edit.editingFinished.connect(self._on_workdir_edited)
        workdir_row.addWidget(self._workdir_edit, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_work_dir)
        workdir_row.addWidget(browse_btn)
        root.addWidget(self._workdir_container)
        self._workdir_container.setVisible(self._standalone)

        # ── Pipeline files (single deduplicated panel) ────────────────
        self._files_widget = PipelineFilesWidget(
            _NUCLEUS_PIPELINE_FILE_GROUPS,
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
        # The staged files panel lists 1_cellpose/2_nucleus paths that don't
        # exist in the standalone flat layout.
        self.pipeline_files_header.setVisible(not self._standalone)
        self._pipeline_files_section.setVisible(not self._standalone)

        # ── Atom Extraction ──────────────────────────────────────────
        self._build_atom_extraction_section(root)

        # ── Workflow sections ────────────────────────────────────────
        self._build_segmentation_inputs_section(root)
        self._build_tracking_ultrack_section(root)

        # ── Ultrack Database Browser ─────────────────────────────────
        self._build_db_browser_section(root)

        # ── Correction (group box) ───────────────────────────────────
        self._build_correction_section(root)

    # -- Parameters --------------------------------------------------------

    def _build_segmentation_inputs_section(self, root: QVBoxLayout) -> None:
        self.nucleus_segmentation_inputs_widget = NucleusSegmentationInputsWidget(self)
        segmentation_inputs = self.nucleus_segmentation_inputs_widget
        self.segmentation_inputs_parameters_section = segmentation_inputs.section
        self.segmentation_inputs_section = segmentation_inputs.section
        self.nucleus_segmentation_inputs_widget.hide()

    def _build_tracking_ultrack_section(self, root: QVBoxLayout) -> None:
        self.nucleus_tracking_inputs_widget = NucleusTrackingInputsWidget(self)
        self._alias_tracking_inputs_controls()

        self.nucleus_pipeline_widget = NucleusPipelineWidget(
            self.viewer,
            workspace_provider=lambda: self._workspace,
            seg_inputs_provider=lambda: self.nucleus_segmentation_inputs_widget,
            tracking_inputs_provider=lambda: self.nucleus_tracking_inputs_widget,
            refresh_files_callback=lambda: self._files_widget.refresh(self._pos_dir),
            refresh_db_browser_callback=lambda: self._refresh_ultrack_db_browser(),
            sync_viewer_activity_callback=lambda: self.gate.recompute(),
            parent=self,
        )
        self._alias_pipeline_controls()
        self.nucleus_tracking_inputs_widget.hide()
        self.nucleus_pipeline_widget.hide()

        # Hide the built-in section headers — the stage-row ⚙ buttons drive
        # expand/collapse instead.
        self.tracking_db_section.set_header_visible(False)
        self.tracking_solve_section.set_header_visible(False)

        # Collapse inline params blocks by default.
        self.tracking_db_section.collapse()
        self.tracking_solve_section.collapse()

        pipeline_block = self.nucleus_pipeline_widget.build_pipeline_block(
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
        self.atom_union_max_atoms_spin = ti.atom_union_max_atoms_spin
        self.atom_union_max_area_spin = ti.atom_union_max_area_spin
        self.db_gen_max_dist_spin = ti.db_gen_max_dist_spin
        self.db_gen_max_neighbors_spin = ti.db_gen_max_neighbors_spin
        self.db_gen_linking_mode_combo = ti.db_gen_linking_mode_combo
        self.db_gen_area_weight_spin = ti.db_gen_area_weight_spin
        self.db_gen_iou_weight_spin = ti.db_gen_iou_weight_spin
        self.db_gen_distance_weight_spin = ti.db_gen_distance_weight_spin
        self.db_gen_quality_weight_spin = ti.db_gen_quality_weight_spin
        self.db_gen_quality_exp_spin = ti.db_gen_quality_exp_spin
        self.db_gen_circularity_weight_spin = ti.db_gen_circularity_weight_spin
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
            "_set_running_stage",
        ):
            setattr(self, name, getattr(pl, name))

    # -- Ultrack Database Browser ------------------------------------------

    # -- Correction --------------------------------------------------------

    def _build_correction_section(self, root: QVBoxLayout) -> None:
        self.nucleus_correction_widget = NucleusCorrectionWidget(
            self.viewer,
            workspace_provider=lambda: self._workspace,
            ultrack_config_provider=lambda: self._ultrack_config_from_controls(),
            focus_takeover_callback=self._set_correction_focus_takeover,
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

    def _plugin_dock(self):
        """The napari QDockWidget hosting this whole workflow panel, if any."""
        from qtpy.QtWidgets import QDockWidget

        widget = self.parentWidget()
        while widget is not None:
            if isinstance(widget, QDockWidget):
                return widget
            widget = widget.parentWidget()
        return None

    def _set_correction_focus_takeover(self, on: bool) -> None:
        """Hand the whole window to the correction workspace, or give it back.

        Driven by NucleusCorrectionWidget once activation succeeds (and again,
        with ``False``, on deactivate). In focus mode the correction controls are
        reparented out into their own workspace dock, so we hide this entire
        plugin dock; deactivate re-inserts the correction header + section into
        this layout (their original trailing position) and shows the dock again.
        """
        layout = self.layout()
        dock = self._plugin_dock()
        if on:
            if dock is not None:
                dock.setVisible(False)
            return
        if layout is not None:
            for widget in (
                getattr(self, "correction_header", None),
                getattr(self, "correction_mode_section", None),
            ):
                if widget is not None:
                    layout.addWidget(widget)
        if dock is not None:
            dock.setVisible(True)

    def _alias_correction_controls(self) -> None:
        correction = self.nucleus_correction_widget
        self.correction_header = correction.header
        self.correction_header_lbl = correction.header_lbl
        self.correction_shortcuts_btn = correction.shortcuts_btn
        self.correction_params_btn = correction.params_btn
        self.correction_active_btn = correction.active_btn
        self.correction_toolbar = correction.toolbar
        self.save_tracked_btn = correction.save_tracked_btn
        self.extend_back_btn = correction.extend_back_btn
        self.extend_fwd_btn = correction.extend_fwd_btn
        self.retrack_back_btn = correction.retrack_back_btn
        self.retrack_fwd_btn = correction.retrack_fwd_btn
        self.swap_smaller_btn = correction.swap_smaller_btn
        self.swap_larger_btn = correction.swap_larger_btn
        self.reassign_ids_btn = correction.reassign_ids_btn
        self.validate_track_btn = correction.validate_track_btn
        self.anchor_here_btn = correction.anchor_here_btn
        self.annotate_db_btn = correction.annotate_db_btn
        self.remove_unvalidated_btn = correction.remove_unvalidated_btn
        self.commit_btn = correction.commit_btn
        self.correction_status_lbl = correction.status_lbl
        self.validation_counter_lbl = correction.validation_counter_lbl
        self.extend_area_weight_spin = correction.extend_area_weight_spin
        self.extend_iou_weight_spin = correction.extend_iou_weight_spin
        self.extend_distance_weight_spin = correction.extend_distance_weight_spin
        self.extend_greedy_overwrite_check = correction.extend_greedy_overwrite_check
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
            "_on_annotate_database",
            "_on_annotate_database_done",
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

    # -- Atom Extraction ---------------------------------------------------

    def _build_atom_extraction_section(self, root: QVBoxLayout) -> None:
        self.atom_extraction_widget = NucleusAtomExtractionWidget(self)
        self.atom_extraction_section = self.atom_extraction_widget.section
        self._alias_atom_extraction_controls()
        root.addWidget(self.atom_extraction_widget.header)
        root.addWidget(self.atom_extraction_widget.section)

    def _atom_fg_path(self):
        return self._workspace.foreground if self._workspace else None

    def _atom_contour_path(self):
        return self._workspace.contours if self._workspace else None

    def _atom_output_path(self):
        return self._workspace.atoms if self._workspace else None

    # ================================================================
    # Signals
    # ================================================================
    def _connect_signals(self) -> None:
        # DB Browser
        self.ultrack_db_active_btn.toggled.connect(
            self._on_guarded_ultrack_db_activate
        )
        try:
            self.correction_active_btn.toggled.disconnect(
                self.nucleus_correction_widget._on_correction_active_button_toggled
            )
        except (TypeError, RuntimeError):
            pass
        self.correction_active_btn.toggled.connect(
            self._on_guarded_correction_active_button_toggled
        )
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
        self.ultrack_db_annotation_check.toggled.connect(
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
        # Initial enablement is applied by ``_register_gate_controls`` (called
        # right after ``_connect_signals`` in ``__init__``).

    # ================================================================
    # Viewer activity guard (driven by the shared UI gate)
    # ================================================================
    def _register_gate_controls(self) -> None:
        """Register this section's controls with the app-wide UI gate.

        Correction and the database browser are mutually-exclusive viewer
        owners; the pipeline run/params buttons rebuild the data those owners
        view, so they are blocked while either owner is active. All enablement
        flows from :class:`~cellflow.napari.ui_gate.UiGate` from here on.
        """
        g = self.gate
        g.register_owner(
            "correction:nucleus",
            "correction mode",
            exit_fn=lambda: self.correction_active_btn.setChecked(False),
        )
        g.register_owner(
            "db_browser",
            "database browser mode",
            exit_fn=lambda: self.ultrack_db_active_btn.setChecked(False),
        )
        g.register(
            self.correction_active_btn,
            ControlClass.VIEWER_OWNER,
            owner_token="correction:nucleus",
        )
        g.register(
            self.ultrack_db_active_btn,
            ControlClass.VIEWER_OWNER,
            owner_token="db_browser",
        )
        pl = self.nucleus_pipeline_widget
        run_stage_of = {
            self.seg_run_btn: "seg",
            self.db_run_btn: "db",
            self.solve_run_btn: "ultrack",
        }
        for button, stage in run_stage_of.items():
            g.register(
                button,
                ControlClass.RUN_VIEWER,
                when=lambda s=stage: pl._running_stage in (None, s),
            )
        # ⚙ params just open/close a parameter panel — no viewer/context side
        # effect — so they stay available during modes and runs.
        for button in (self.seg_params_btn, self.db_params_btn, self.solve_params_btn):
            g.register(button, ControlClass.HARMLESS)
        self._register_atom_gate_controls()
        g.recompute()

    def _on_guarded_ultrack_db_activate(self, checked: bool) -> None:
        self._on_ultrack_db_activate(checked)
        if checked:
            self.gate.claim_viewer("db_browser")
        else:
            self.gate.release_viewer("db_browser")

    def _on_guarded_correction_active_button_toggled(self, checked: bool) -> None:
        self._on_correction_active_button_toggled(checked)
        # Activation can bail out — e.g. no tracked data on disk yet — reverting
        # the button to unchecked from inside the handler. Claim/release the
        # viewer from the *resulting* button state, not the incoming toggle, so a
        # bailed-out activation never leaves the gate owning the viewer with the
        # button already off (banner + disabled controls and no way to release).
        if self.correction_active_btn.isChecked():
            self.gate.claim_viewer("correction:nucleus")
        else:
            self.gate.release_viewer("correction:nucleus")

    # ================================================================
    # Path helpers
    # ================================================================
    def _ultrack_db_path(self) -> Path | None:
        # Required by NucleusUltrackDbBrowserMixin.
        return self._workspace.ultrack_db if self._workspace else None

    def _nucleus_foreground_path(self) -> Path | None:
        # Required by NucleusUltrackDbBrowserMixin.
        return self._workspace.foreground if self._workspace else None

    # These delegate to the pipeline widget so that tests that call path helpers
    # on the workflow widget (legacy seam tests) continue to pass.
    def _contours_path(self) -> Path | None:
        return self.nucleus_pipeline_widget._contours_path()


    # ================================================================
    # Public API
    # ================================================================
    def refresh(self, pos_dir: Path | None) -> None:
        """Orchestrated seam: drive the piece from a staged position directory."""
        self._pos_dir = pos_dir
        workspace = (
            NucleusWorkspace.staged(pos_dir) if pos_dir is not None else None
        )
        self._apply_workspace(workspace, files_root=pos_dir)

    def set_context(self, *, work_dir: Path | str | None) -> None:
        """Standalone seam: drive the piece from one flat working directory.

        The directory holds ``foreground.tif`` + ``contours.tif`` and receives
        every output (``data.db``, ``tracked_labels.tif``, ``atoms.tif``, the
        ``*.json`` annotations). There is no staged ``2_nucleus`` subfolder.
        """
        self._pos_dir = None
        workspace = (
            NucleusWorkspace.flat(work_dir) if work_dir else None
        )
        if self._standalone:
            self._workdir_edit.setText(str(work_dir) if work_dir else "")
            self._save_standalone_settings()
        # The staged files panel is meaningless without a position dir.
        self._apply_workspace(workspace, files_root=None)

    def _apply_workspace(
        self, workspace: NucleusWorkspace | None, *, files_root: Path | None
    ) -> None:
        self._workspace = workspace
        self._files_widget.refresh(files_root)
        if workspace is None:
            if self.correction_active_btn.isChecked():
                self.correction_active_btn.setChecked(False)
            else:
                self.correction_widget.deactivate()
                self._remove_correction_owned_layers()
            return
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    # ── Standalone helpers ────────────────────────────────────────────────────
    def _on_browse_work_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select working directory")
        if path:
            self.set_context(work_dir=path)

    def _on_workdir_edited(self) -> None:
        text = self._workdir_edit.text().strip()
        self.set_context(work_dir=text or None)

    def _settings(self) -> QSettings:
        return QSettings("cellflow", "cellflow_tracking")

    def _load_standalone_settings(self) -> None:
        work_dir = self._settings().value("work_dir", "", type=str)
        if work_dir:
            self.set_context(work_dir=work_dir)

    def _save_standalone_settings(self) -> None:
        self._settings().setValue("work_dir", self._workdir_edit.text().strip())

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
    def _db_gen_config_from_controls(self):
        return self.nucleus_pipeline_widget._db_gen_config_from_controls()

    def _ultrack_config_from_controls(self):
        return self.nucleus_pipeline_widget._ultrack_config_from_controls()

    # Pipeline handler methods are owned by NucleusPipelineWidget and aliased
    # onto this instance by _alias_pipeline_controls() during __init__.


    # ================================================================
    # Correction / DB browser coordination
    # ================================================================
    def _on_dims_step_changed(self, event=None) -> None:
        if not self._dims_step_refresh_pending:
            self._dims_step_refresh_pending = True
            QTimer.singleShot(0, self._refresh_after_dims_step_changed)
        if self.ultrack_db_browser_section.is_expanded:
            QTimer.singleShot(0, self._refresh_ultrack_db_browser)
        if getattr(self, "_atom_preview_active", False):
            QTimer.singleShot(0, self._refresh_atom_preview)

    def _refresh_after_dims_step_changed(self) -> None:
        self._dims_step_refresh_pending = False
        self.nucleus_correction_widget.on_dims_step_changed()


def make_nucleus_tracking_widget(napari_viewer=None):
    """napari plugin factory for the standalone nucleus tracking/correction piece.

    Patches the napari layer-controls delegate (best-effort) and returns the
    workflow widget in standalone mode, with its own working-directory picker
    and config.
    """
    try:
        from cellflow.napari._napari_compat import patch_napari_layer_delegate

        patch_napari_layer_delegate()
    except Exception:
        pass
    return NucleusWorkflowWidget(viewer=napari_viewer, standalone=True)
