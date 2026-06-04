"""Correction section widget for the nucleus workflow."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker as _thread_worker
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._widget_helpers import (
    btn as _btn,
    dslider as _dslider,
    heading as _heading,
    make_status as _make_status,
    tool_btn as _tool_btn,
)
from cellflow.napari._correction_centroids import (
    ensure_label_colormap_entries,
    refresh_label_colormap,
)
from cellflow.napari._correction_anchor import (
    anchor_correction,
    without_anchor_correction,
)
from cellflow.napari._correction_utils import (
    frame_view_2d,
    reassign_ids_stack,
    retrack_stack_direction,
)
from cellflow.napari._correction_commit import (
    prepare_committed_labels,
    remove_unvalidated_from_data,
)
from cellflow.napari._correction_layer_lifecycle import (
    LayerViewState,
    capture_layer_view_state,
    detach_higher_dim_stacks,
    hide_all_layers,
    reattach_layers,
    remove_owned_layers,
    restore_layer_view_state,
)
from cellflow.napari._correction_layer_loader import (
    add_correction_image_layer,
    add_tracked_labels_and_track_layer,
    remove_other_correction_label_layers,
)
from cellflow.napari._correction_protection import (
    protected_cell_ids_at_frame,
    protected_cell_mask,
)
from cellflow.napari._correction_validation import (
    correction_for_label_frame,
    corrections_for_label_frames,
    selected_correction_target,
)
from cellflow.napari._paths import NucleusArtifactPaths
from cellflow.napari.correction_widget import CorrectionWidget
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
    read_validated_frames,
    read_validated_tracks,
    remap_validated_tracks,
    write_corrections,
)
from cellflow.napari.ui_style import (
    add_block_checkbox_row,
    add_block_pair_row,
    block_grid,
    danger_button,
)
from cellflow.napari.validated_overlay_controller import (
    ValidatedOverlayController,
)
from cellflow.napari.track_path_controller import AllTracksController
from cellflow.napari.lineage_canvas_controller import LineageCanvasController
from cellflow.napari.candidate_gallery_controller import CandidateGalleryController
from cellflow.napari._correction_takeover import (
    hide_native_docks,
    restore_native_docks,
)
from cellflow.napari.widgets import CollapsibleSection
from cellflow.napari._correction_ui import (
    build_correction_header,
    build_correction_toolbar,
    build_shortcuts_widget,
    flatten_embedded_section,
)
from cellflow.tracking_ultrack.corrections import (
    Correction,
    corrections_from_validated_tracks,
)
from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.db_build import (
    annotate_database_from_corrections as _annotate_database_from_corrections,
)
from cellflow.tracking_ultrack.extend import extend_track_from_db as _extend_track_from_db
from cellflow.tracking_ultrack.swap_candidate import (
    SwapCandidate as _SwapCandidate,
    _SwapCursor,
    list_swap_candidates,
    cycle_index as _cycle_index,
    nearest_area_index as _nearest_area_index,
)
from cellflow.tracking_ultrack.retracker import retrack_frame_constrained

logger = logging.getLogger(__name__)

_TRACKED_LAYER = "Tracked: Nucleus"
_CORRECTION_TRACKED_LAYER = "[Correction] Nucleus Labels"
_CORRECTION_TRACK_LAYER = "[Correction] Nucleus tracks"
_CORRECTION_CELL_ZAVG_LAYER = "[Correction] Cell z-avg"
_CORRECTION_NUC_ZAVG_LAYER = "[Correction] Nucleus z-avg"
_NUCLEUS_TRACK_COLOR_SCALE = 0.65

_DEFAULT_DEPENDENCIES = {
    "annotate_database_from_corrections": _annotate_database_from_corrections,
    "extend_track_from_db": _extend_track_from_db,
    "read_corrections": lambda *args, **kwargs: read_corrections(*args, **kwargs),
    "read_validated_tracks": (
        lambda *args, **kwargs: read_validated_tracks(*args, **kwargs)
    ),
    "thread_worker": _thread_worker,
}


class NucleusCorrectionWidget(QWidget):
    """Qt controls for nucleus tracking correction workflows."""

    def __init__(
        self,
        viewer,
        *,
        edit_callback=None,
        pos_dir_provider: Callable[[], Path | None] | None = None,
        ultrack_config_provider: Callable[[], TrackingConfig] | None = None,
        dependencies: dict[str, Callable] | None = None,
        focus_takeover_callback: Callable[[bool], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir_provider = pos_dir_provider
        self._ultrack_config_provider = ultrack_config_provider
        # Called with True once correction-mode activation succeeds and False on
        # deactivate, so the host can hide its other workflow sections for a
        # full-window focus mode (the widget deliberately holds no sibling refs).
        self._focus_takeover = focus_takeover_callback or (lambda _on: None)
        self._local_pos_dir: Path | None = None
        self._dependencies = {**_DEFAULT_DEPENDENCIES, **(dependencies or {})}
        self._edit_callback = edit_callback or self._on_cells_edited
        self._correction_owned_layers: set[str] = set()
        self._correction_view_state: LayerViewState | None = None
        self._correction_dirty: bool = False
        # True only while a lineage-canvas click is driving the selection, so the
        # selection listener does not scroll the lineage canvas back onto the row
        # the user just clicked there (image-viewer clicks should recenter it;
        # lineage clicks should not yank the canvas around).
        self._navigating_from_lineage: bool = False
        self._swap_cursor: _SwapCursor | None = None
        self._native_dock_state: dict[str, bool] = {}
        # The single right-side workspace dock and the horizontal splitter it
        # hosts (candidate gallery · film strip · controls strip), built on
        # activate; both None while correction mode is off. The controls strip
        # holds the correction header + section + lineage overview, reparented
        # out of the plugin dock.
        self._workspace_dock = None
        self._workspace_splitter: QSplitter | None = None
        self._controls_container: QWidget | None = None
        # Higher-dim stacks (e.g. a raw T,Z,Y,X z-stack) removed on activate so
        # the viewer collapses to a single frame slider, kept verbatim here to
        # re-append on deactivate. The plugin dock + its pre-correction width are
        # captured so the dock, blown up to half-window for focus mode, can be
        # shrunk back when correction turns off.
        self._detached_stack_layers: list = []
        self._pre_correction_dock = None
        self._pre_correction_dock_width: int | None = None
        self._validated_overlay = ValidatedOverlayController(
            self.viewer,
            tracked_layer_provider=self._correction_tracked_layer,
            pos_dir_provider=lambda: self._pos_dir,
            owned_layers=self._correction_owned_layers,
        )
        self._setup_ui()
        self._all_tracks = AllTracksController(
            self.viewer,
            tracked_layer_provider=self._correction_tracked_layer,
            selected_label_provider=lambda: int(
                getattr(self.correction_widget, "_selected_label", 0) or 0
            ),
            enabled_provider=self.track_path_check.isChecked,
            current_t_provider=self._current_t,
            status_callback=self._correction_status,
            owned_layers=self._correction_owned_layers,
        )
        self._lineage_canvas = LineageCanvasController(
            self.viewer,
            tracked_data_provider=self._ensure_tracked_layer_data,
            tracked_layer_provider=self._correction_tracked_layer,
            intensity_layer_provider=self._film_strip_intensity_layer,
            selected_label_provider=lambda: int(
                getattr(self.correction_widget, "_selected_label", 0) or 0
            ),
            current_t_provider=self._current_t,
            on_activate=self._navigate_to_error,
            pos_dir_provider=lambda: self._pos_dir,
        )
        self._candidate_gallery = CandidateGalleryController(
            self.viewer,
            tracked_data_provider=self._ensure_tracked_layer_data,
            tracked_layer_provider=self._correction_tracked_layer,
            intensity_layer_provider=self._film_strip_intensity_layer,
            selected_label_provider=lambda: int(
                getattr(self.correction_widget, "_selected_label", 0) or 0
            ),
            current_t_provider=self._current_t,
            db_path_provider=self._ultrack_db_path,
            protected_mask_provider=self._gallery_protected_mask,
            extend_kwargs_provider=self._extend_kwargs,
            apply_swap=self._apply_swap_candidate,
            apply_extend=self._apply_extend_candidate,
            status_callback=self._correction_status,
        )

    @property
    def _pos_dir(self):
        if self._pos_dir_provider is not None:
            return self._pos_dir_provider()
        return self._local_pos_dir

    @_pos_dir.setter
    def _pos_dir(self, value) -> None:
        self._local_pos_dir = value

    def _dependency(self, name: str):
        return self._dependencies[name]

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Searching + rendering the candidate gallery is expensive, so debounce
        # it: every refresh request (re)starts this timer and the actual rebuild
        # only fires once the frame has been still for the interval — fast
        # scrubbing no longer recomputes candidates for every frame it sweeps.
        self._candidate_refresh_timer = QTimer(self)
        self._candidate_refresh_timer.setSingleShot(True)
        self._candidate_refresh_timer.setInterval(200)
        self._candidate_refresh_timer.timeout.connect(
            self._refresh_candidate_gallery_now
        )

        inner = QWidget(self)
        group_lay = QVBoxLayout(inner)
        group_lay.setContentsMargins(0, 0, 0, 0)
        group_lay.setSpacing(6)

        self.active_btn = _tool_btn(
            "⏻",
            "Activate correction mode and show correction layers and controls.",
            checkable=True,
        )
        self.active_btn.setToolTip(
            "Activate correction mode and show correction layers and controls."
        )
        self.params_btn = _tool_btn(
            "⚙", "Show correction parameters.", checkable=True
        )
        self.shortcuts_btn = _tool_btn(
            "📖", "Show correction shortcuts.", checkable=True
        )

        self.save_tracked_btn = _tool_btn(
            "💾", "Save corrected tracked nucleus labels to disk (S)."
        )
        self.extend_back_btn = _tool_btn(
            "◀", "Extend selected track one frame backward (A)."
        )
        self.extend_fwd_btn = _tool_btn(
            "▶", "Extend selected track one frame forward (D)."
        )
        self.retrack_back_btn = _tool_btn(
            "↶", "Retrack all labels backward from current frame (Q)."
        )
        self.retrack_fwd_btn = _tool_btn(
            "↷", "Retrack all labels forward from current frame (E)."
        )
        self.swap_smaller_btn = _tool_btn(
            "⮂", "Swap selected cell with the next smaller candidate fragment (Z)."
        )
        self.swap_larger_btn = _tool_btn(
            "⮀", "Swap selected cell with the next larger candidate fragment (C)."
        )
        self.reassign_ids_btn = _tool_btn(
            "#", "Reassign cell IDs to contiguous range 1-N."
        )
        self.validate_track_btn = _tool_btn(
            "✓", "Lock selected cell geometry in every frame where it appears (V)."
        )
        self.anchor_here_btn = _tool_btn(
            "⚓", "Anchor selected cell identity at the current frame (B)."
        )
        self.annotate_db_btn = _tool_btn(
            "✎", "Apply saved validations and anchors to the Ultrack database."
        )
        self.remove_unvalidated_btn = _tool_btn(
            "🗑",
            "Remove nucleus label pixels not marked validated for their frame.",
        )
        danger_button(self.remove_unvalidated_btn)

        self.commit_btn = _btn(
            "Commit",
            "Reassign cell IDs, remove unvalidated labels, and save tracked labels.",
        )

        self.status_lbl = _make_status()
        # Drop the smaller status font so the status line reads at the same
        # size as the rest of the controls column.
        self.status_lbl.setStyleSheet("")

        self.validation_counter_lbl = QLabel("")
        self.validation_counter_lbl.setWordWrap(True)

        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
            show_cleanup=False,
            # The lineage canvas is the navigation surface now — drop the
            # redundant "Inspect cell" group from the correction column.
            show_inspector=False,
        )
        self.correction_widget.set_edit_callback(self._edit_callback)
        self.correction_widget.set_protected_mask_callback(
            self._manual_correction_protected_mask
        )
        # Use an additive listener, not set_selection_callback: the workflow
        # widget owns that single slot and would otherwise clobber the comet's
        # rebuild-on-selection (so it would only build on first checkbox tick).
        self.correction_widget.add_selection_listener(
            self._on_track_selection_changed
        )
        self.correction_widget.set_spotlight_mask_provider(
            self._track_path_spotlight_mask
        )
        self.correction_widget._status.setVisible(False)

        extend_retrack_inner = QWidget(self)
        extend_retrack_lay = QVBoxLayout(extend_retrack_inner)
        extend_retrack_lay.setContentsMargins(0, 0, 0, 0)
        extend_retrack_lay.setSpacing(6)
        extend_retrack_lay.addWidget(self.correction_widget._outline_btn)

        extend_retrack_lay.addWidget(_heading("Extend"))
        g = block_grid(horizontal_spacing=12)
        # Extend follows precomputed LinkDB edges (no geometric scoring knobs);
        # only the paint behavior remains tunable.
        self.extend_greedy_overwrite_check = QCheckBox("Greedy overwrite")
        add_block_checkbox_row(g, 0, self.extend_greedy_overwrite_check)
        extend_retrack_lay.addLayout(g)

        extend_retrack_lay.addWidget(_heading("Retrack"))
        g = block_grid(horizontal_spacing=12)
        self.retrack_max_dist_spin = _dslider(0, 500, 20.0, 1.0, 1)
        # Scoring weights for the retrack frame matcher. Kept under the
        # ``extend_*`` attribute names for settings back-compat; extend no longer
        # uses them.
        self.extend_area_weight_spin = _dslider(0, 10, 1.0, 0.1, 2)
        self.extend_iou_weight_spin = _dslider(0, 10, 1.0, 0.1, 2)
        self.extend_distance_weight_spin = _dslider(0, 10, 0.05, 0.01, 3)
        add_block_pair_row(
            g,
            0,
            "Max distance:",
            self.retrack_max_dist_spin,
            "Area weight:",
            self.extend_area_weight_spin,
        )
        add_block_pair_row(
            g,
            1,
            "IoU weight:",
            self.extend_iou_weight_spin,
            "Distance weight:",
            self.extend_distance_weight_spin,
        )
        extend_retrack_lay.addLayout(g)
        self.extend_retrack_params_section = CollapsibleSection(
            "Extend / Retrack Parameters",
            extend_retrack_inner,
            expanded=False,

        )
        flatten_embedded_section(self.extend_retrack_params_section)
        self.extend_retrack_params_section.setVisible(False)
        self.extend_params_section = self.extend_retrack_params_section
        self.retrack_params_section = self.extend_retrack_params_section

        self.shortcuts_section = CollapsibleSection(
            "Correction Shortcuts",
            build_shortcuts_widget(),
            expanded=False,

        )
        flatten_embedded_section(self.shortcuts_section)
        self.shortcuts_section.setVisible(False)
        self.correction_widget.setVisible(False)
        # Extend / swap are driven by clicking into the candidate gallery now,
        # so they're dropped from the toolbar (the A/D/Z/C shortcuts still work).
        self.toolbar = build_correction_toolbar(
            self,
            [
                (self.save_tracked_btn,),
                (self.retrack_back_btn, self.retrack_fwd_btn),
                (self.validate_track_btn, self.anchor_here_btn),
                (self.annotate_db_btn,),
                (self.reassign_ids_btn, self.remove_unvalidated_btn),
            ],
        )
        self.toolbar.setVisible(False)
        self.track_path_check = QCheckBox("Track path")
        self.track_path_check.setToolTip(
            "Paint the selected track's whole trajectory as a fading comet "
            "(viridis, oldest→newest) with a frame number in each mask."
        )
        self.track_path_check.setVisible(False)
        self.lineage_canvas_check = QCheckBox("Lineage canvas")
        self.lineage_canvas_check.setToolTip(
            "Open the docked correction canvas: a swimlane overview of every "
            "track (time across, gaps flag likely ID swaps; green=validated, "
            "orange=anchored) over the selected track's "
            "per-frame film strip. Click a lane or tile to jump there."
        )
        self.lineage_canvas_check.setVisible(False)
        self.candidate_gallery_check = QCheckBox("Candidate gallery")
        self.candidate_gallery_check.setToolTip(
            "Dock three thumbnail columns — extend-backward, swap, extend-forward — "
            "of the candidate segmentations for the selected cell at this frame. "
            "Click a thumbnail to apply that extend/swap."
        )
        self.candidate_gallery_check.setVisible(False)
        self.validation_counter_lbl.setVisible(False)
        self.correction_widget._attrib_lbl.setVisible(False)

        # Lay the controls column out top→bottom: the action toolbar, the
        # collapsible params / shortcuts panels (they expand just below it),
        # the per-frame correction tools, the view-toggle checkboxes in two
        # columns, then the status line. The swimlane track renders are
        # appended below this whole section in _build_workspace_controls_dock.
        group_lay.addWidget(self.toolbar)
        group_lay.addWidget(self.extend_retrack_params_section)
        group_lay.addWidget(self.shortcuts_section)
        group_lay.addWidget(self.correction_widget)

        # One checkbox per row: a two-wide grid pinned the controls strip (and
        # so the whole right dock) to the summed label width — the dominant
        # floor on how narrow the dock can be dragged.
        self._view_toggle_grid = QGridLayout()
        self._view_toggle_grid.setContentsMargins(0, 0, 0, 0)
        self._view_toggle_grid.setVerticalSpacing(2)
        self._view_toggle_grid.addWidget(self.track_path_check, 0, 0)
        self._view_toggle_grid.addWidget(self.lineage_canvas_check, 1, 0)
        self._view_toggle_grid.addWidget(self.candidate_gallery_check, 2, 0)
        self._view_toggle_grid.setColumnStretch(0, 1)
        group_lay.addLayout(self._view_toggle_grid)

        group_lay.addWidget(self.status_lbl)
        group_lay.addWidget(self.validation_counter_lbl)
        group_lay.addWidget(self.correction_widget._attrib_lbl)

        self.header, self.header_lbl = build_correction_header(
            self,
            shortcuts_btn=self.shortcuts_btn,
            params_btn=self.params_btn,
            active_btn=self.active_btn,
        )

        self.section = CollapsibleSection(
            "Correction",
            inner,
            expanded=False,

        )
        self.section._toggle.setVisible(False)
        self.section._toggle.setEnabled(False)

        self.correction_active_btn = self.active_btn
        self.correction_shortcuts_btn = self.shortcuts_btn
        self.correction_status_lbl = self.status_lbl
        self.correction_mode_section = self.section
        self._correction_active_content_visible = False
        self._connect_signals()

    def _connect_signals(self) -> None:
        self.save_tracked_btn.clicked.connect(self._on_save_tracked)
        self.reassign_ids_btn.clicked.connect(self._on_reassign_ids)
        self.validate_track_btn.clicked.connect(self._on_validate_track)
        self.anchor_here_btn.clicked.connect(self._on_anchor_here)
        self.annotate_db_btn.clicked.connect(self._on_annotate_database)
        self.extend_back_btn.clicked.connect(self._on_extend_backward)
        self.extend_fwd_btn.clicked.connect(self._on_extend_forward)
        self.retrack_back_btn.clicked.connect(self._on_retrack_backward)
        self.retrack_fwd_btn.clicked.connect(self._on_retrack_forward)
        self.swap_smaller_btn.clicked.connect(
            lambda: self._on_swap_step(direction="smaller")
        )
        self.swap_larger_btn.clicked.connect(
            lambda: self._on_swap_step(direction="larger")
        )
        self.remove_unvalidated_btn.clicked.connect(
            self._on_remove_unvalidated_labels
        )
        self.commit_btn.clicked.connect(self._on_commit)
        self.track_path_check.toggled.connect(self._on_toggle_track_path)
        self.lineage_canvas_check.toggled.connect(self._on_toggle_lineage_canvas)
        self.candidate_gallery_check.toggled.connect(self._on_toggle_candidate_gallery)
        self.params_btn.toggled.connect(
            self._on_correction_params_button_toggled
        )
        self.shortcuts_btn.toggled.connect(
            self._on_correction_shortcuts_button_toggled
        )
        self.active_btn.toggled.connect(self._on_correction_active_button_toggled)
        self.correction_widget._activate_btn.toggled.connect(
            self._on_correction_mode_toggled
        )
        self._install_correction_shortcuts()

    @staticmethod
    def _set_checked_without_signal(button, checked: bool) -> None:
        old = button.blockSignals(True)
        try:
            button.setChecked(checked)
        finally:
            button.blockSignals(old)

    def _sync_correction_panel_visibility(self) -> None:
        show_params = self.params_btn.isChecked()
        show_shortcuts = self.shortcuts_btn.isChecked()
        show_active = self._correction_active_content_visible

        self.extend_retrack_params_section.setVisible(show_params)
        self.shortcuts_section.setVisible(show_shortcuts)
        self.correction_widget.setVisible(show_active)
        self.toolbar.setVisible(show_active)
        self.validation_counter_lbl.setVisible(show_active)
        self.correction_widget._attrib_lbl.setVisible(show_active)
        self.track_path_check.setVisible(show_active)
        self.lineage_canvas_check.setVisible(show_active)
        self.candidate_gallery_check.setVisible(show_active)

        if show_params or show_shortcuts or show_active:
            self.section.expand()
        else:
            self.section.collapse()

    def _on_correction_params_button_toggled(self, checked: bool) -> None:
        self.extend_retrack_params_section._toggle.setChecked(checked)
        if checked:
            self._set_checked_without_signal(self.shortcuts_btn, False)
            self.shortcuts_section._toggle.setChecked(False)
        self._sync_correction_panel_visibility()

    def _on_correction_shortcuts_button_toggled(self, checked: bool) -> None:
        self.shortcuts_section._toggle.setChecked(checked)
        if checked:
            self._set_checked_without_signal(self.params_btn, False)
            self.extend_retrack_params_section._toggle.setChecked(False)
        self._sync_correction_panel_visibility()

    @property
    def _paths(self) -> NucleusArtifactPaths | None:
        return NucleusArtifactPaths(self._pos_dir) if self._pos_dir else None

    def _tracked_path(self):
        return self._paths.tracked if self._paths else None

    def _foreground_path(self):
        return self._paths.nucleus_foreground if self._paths else None

    def _ultrack_workdir(self):
        return self._paths.ultrack_workdir if self._paths else None

    def _ultrack_db_path(self):
        return self._paths.ultrack_db if self._paths else None

    def _ultrack_config_from_controls(self):
        if self._ultrack_config_provider is not None:
            return self._ultrack_config_provider()
        return TrackingConfig()

    def _ensure_tracked_layer_data(self) -> np.ndarray | None:
        layer = self._correction_tracked_layer()
        if layer is not None:
            return np.asarray(layer.data)
        tracked_path = self._tracked_path()
        if tracked_path is None or not tracked_path.exists():
            return None
        labels = np.asarray(tifffile.imread(str(tracked_path)), dtype=np.uint32)
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        return labels


    def _cell_foreground_path(self):
        return self._paths.cell_foreground if self._paths else None

    def _nucleus_foreground_path(self):
        return self._paths.nucleus_foreground if self._paths else None

    def _current_t(self) -> int:
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) >= 1 else 0

    def _current_cell_ids(self, t: int) -> set[int]:
        layer = self._correction_tracked_layer()
        if layer is None:
            return set()
        frame = frame_view_2d(np.asarray(layer.data), t)
        if frame is None:
            return set()
        return set(int(value) for value in np.unique(frame)) - {0}

    def _correction_tracked_layer(self):
        if _CORRECTION_TRACKED_LAYER in self.viewer.layers:
            return self.viewer.layers[_CORRECTION_TRACKED_LAYER]
        if _TRACKED_LAYER in self.viewer.layers:
            return self.viewer.layers[_TRACKED_LAYER]
        return None

    def _capture_correction_view_state(self) -> None:
        self._correction_view_state = capture_layer_view_state(self.viewer.layers)

    def _restore_correction_view_state(self) -> None:
        restore_layer_view_state(self.viewer.layers, self._correction_view_state)
        self._correction_view_state = None

    def _remove_correction_owned_layers(self) -> None:
        remove_owned_layers(self.viewer.layers, self._correction_owned_layers)

    def _add_correction_image_layer(self, data: np.ndarray, name: str, colormap: str) -> None:
        add_correction_image_layer(
            self.viewer,
            data,
            name=name,
            colormap=colormap,
            owned_layer_names=self._correction_owned_layers,
        )

    def _correction_status(self, msg: str) -> None:
        self.status_lbl.setText(msg)
        self.status_lbl.setVisible(bool(msg))
        lowered = msg.lower()
        if "unsaved" in lowered:
            self._correction_dirty = True
        elif lowered.startswith("saved") or lowered.startswith("loaded"):
            self._correction_dirty = False
        if msg:
            logger.info(msg)

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
        self._correction_dirty = False
        self._correction_status(f"Saved {n} frame(s) to {tracked_path.name}.")

    def _refresh_tracked_layer_from_disk(self) -> None:
        """Overwrite the 'Tracked: Nucleus' layer data from the saved TIFF.

        Called when correction mode is deactivated so the pipeline widget's
        re-solve reads the latest saved state rather than stale in-memory data.
        Does nothing if the file does not exist or if the layer is absent.
        """
        tracked_path = self._tracked_path()
        if tracked_path is None or not tracked_path.exists():
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            return
        try:
            from cellflow.database.tracked import read_full_tracked_stack
            data = np.asarray(read_full_tracked_stack(tracked_path), dtype=np.uint32)
            self.viewer.layers[_TRACKED_LAYER].data = data
        except Exception:
            pass

    def _load_correction_layers_from_disk(self) -> bool:
        tracked_path = self._tracked_path()
        if tracked_path is None or not tracked_path.exists():
            self._correction_status("No tracked labels file found.")
            return False

        self._remove_correction_owned_layers()
        self._remove_other_correction_prefix_layers()
        stack = read_full_tracked_stack(tracked_path)

        for data, name, cmap in (
            (
                self._cell_foreground_path(),
                _CORRECTION_CELL_ZAVG_LAYER,
                "gray",
            ),
            (
                self._nucleus_foreground_path(),
                _CORRECTION_NUC_ZAVG_LAYER,
                "I Purple",
            ),
        ):
            if data is not None and data.exists():
                add_correction_image_layer(
                    self.viewer,
                    np.asarray(tifffile.imread(str(data)), dtype=np.float32),
                    name=name,
                    colormap=cmap,
                    owned_layer_names=self._correction_owned_layers,
                )

        add_tracked_labels_and_track_layer(
            self.viewer,
            stack,
            labels_layer_name=_CORRECTION_TRACKED_LAYER,
            owned_layer_names=self._correction_owned_layers,
            color_scale=_NUCLEUS_TRACK_COLOR_SCALE,
        )
        # Build the live all-tracks overlay from the labels just loaded (a no-op
        # while the "Track path" toggle is off).
        self._all_tracks.refresh()

        self._correction_status(f"Loaded tracked stack {stack.shape} into correction mode.")
        return True

    def _remove_other_correction_prefix_layers(self) -> None:
        remove_other_correction_label_layers(
            self.viewer,
            owned_layer_names=self._correction_owned_layers,
            label_layer_type=napari.layers.Labels,
        )

    def _on_load_tracked(self) -> None:
        self._load_correction_layers_from_disk()

    def _on_reassign_ids(self) -> None:
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        stack = np.asarray(layer.data)
        self._correction_status("Reassigning cell IDs...")

        @self._dependency("thread_worker")(connect={
            "returned": self._on_reassign_ids_done,
            "errored": self._on_correction_worker_error,
        })
        def _worker():
            return reassign_ids_stack(stack)

        _worker()

    def _on_reassign_ids_done(self, result: tuple) -> None:
        remapped, n_cells, old_to_new = result
        layer = self._correction_tracked_layer()
        if layer is not None:
            layer.data = remapped
            self._refresh_correction_label_visuals()
        if self._pos_dir is not None and old_to_new:
            remap_validated_tracks(self._pos_dir, old_to_new)
        self._refresh_lineage_canvas_if_shown()
        self._correction_status(
            f"Reassigned {n_cells} cell IDs to range 1-{n_cells}. Unsaved."
        )

    def _commit_reassign_ids(self, layer) -> int:
        remapped, n_cells, old_to_new = reassign_ids_stack(np.asarray(layer.data))
        layer.data = remapped
        self._refresh_correction_label_visuals()
        if self._pos_dir is not None and old_to_new:
            remap_validated_tracks(self._pos_dir, old_to_new)
        return int(n_cells)

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
        target = selected_correction_target(
            np.asarray(layer.data),
            cell_id=cell_id,
            frame=t,
        )
        if target is None:
            self._correction_status(f"Cell {cell_id} not present at t={t}."); return None
        return target.cell_id, target.frame, target.y, target.x

    def _validated_correction_for_frame(
        self, cell_id: int, t: int, data: np.ndarray
    ) -> Correction | None:
        return correction_for_label_frame(
            data,
            cell_id=cell_id,
            frame=t,
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
        for correction in corrections_for_label_frames(
            data,
            cell_id=cell_id,
            frames=frames,
        ):
            add_correction(self._pos_dir, correction)
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        self._refresh_lineage_canvas_if_shown()
        self._correction_status(
            f"Validated track {cell_id} across {len(frames)} frame(s)."
        )

    def _on_anchor_here(self) -> None:
        target = self._selected_correction_target()
        if target is None or self._pos_dir is None:
            return
        cell_id, t, y, x = target
        corrections = read_corrections(self._pos_dir)
        removal = without_anchor_correction(corrections, cell_id=cell_id, frame=t)
        if removal.removed:
            write_corrections(self._pos_dir, removal.remaining)
            self._refresh_validated_overlay()
            self._refresh_lineage_canvas_if_shown()
            self._correction_status(f"Unanchored cell {cell_id} at t={t}.")
            return
        layer = self._correction_tracked_layer()
        if layer is None:
            add_correction(
                self._pos_dir,
                anchor_correction(cell_id=cell_id, frame=t, y=y, x=x),
            )
            self._refresh_validated_overlay()
            self._refresh_lineage_canvas_if_shown()
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
        self._refresh_lineage_canvas_if_shown()
        suffix = f" (gap-filled {filled} frame(s))" if filled else ""
        self._correction_status(f"Anchored cell {cell_id} at t={t}.{suffix}")

    def _on_annotate_database(self) -> None:
        pos_dir = self._pos_dir
        if pos_dir is None:
            self._correction_status("No project open."); return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._correction_status("data.db not found — run DB Generation first."); return
        score_path = self._foreground_path()
        if score_path is None or not score_path.exists():
            self._correction_status(
                "Missing: nucleus_foreground.tif — build divergence maps first."
            ); return
        working_dir = self._ultrack_workdir()
        if working_dir is None:
            self._correction_status("No Ultrack working directory."); return

        read_corrections_fn = self._dependency("read_corrections")
        read_validated_tracks_fn = self._dependency("read_validated_tracks")
        corrections = list(read_corrections_fn(pos_dir))
        validated_tracks = read_validated_tracks_fn(pos_dir) or None
        tracked_labels = self._ensure_tracked_layer_data()
        if validated_tracks and tracked_labels is None:
            self._correction_status(
                "Annotation from validated tracks requires tracked_labels.tif "
                "(layer not loaded and file not on disk)."
            ); return
        if corrections and validated_tracks and tracked_labels is not None:
            existing = {
                (int(c.cell_id), int(c.t))
                for c in corrections
                if getattr(c, "kind", None) == "validated"
            }
            corrections = list(corrections) + [
                c for c in corrections_from_validated_tracks(validated_tracks, tracked_labels)
                if (int(c.cell_id), int(c.t)) not in existing
            ]
            validated_tracks = None

        cfg = self._ultrack_config_from_controls()
        self._correction_status("Annotating Ultrack database...")

        @self._dependency("thread_worker")(connect={
            "returned": self._on_annotate_database_done,
            "errored": self._on_correction_worker_error,
        })
        def _worker():
            return self._dependency("annotate_database_from_corrections")(
                working_dir=working_dir,
                cfg=cfg,
                score_signal_path=score_path,
                corrections=corrections,
                validated_tracks=validated_tracks,
                tracked_labels=tracked_labels,
            )

        _worker()

    def _on_annotate_database_done(self, report) -> None:
        fake_nodes = int(getattr(report, "fake_nodes", 0) or 0)
        anchor_nodes = int(getattr(report, "anchor_nodes", 0) or 0)
        anchor_links = int(getattr(report, "anchor_links", 0) or 0)
        scored_nodes = int(getattr(report, "scored_nodes", 0) or 0)
        inserted_nodes = int(getattr(report, "injected_homemade_anchors", 0) or 0)
        inserted_links = int(getattr(report, "anchor_incident_links_inserted", 0) or 0)
        self._correction_status(
            "Annotated DB: "
            f"{fake_nodes} FAKE, {anchor_nodes} REAL, "
            f"{anchor_links + inserted_links} link(s), "
            f"{inserted_nodes} inserted node(s), {scored_nodes} scored."
        )

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
            self._correction_status("Extend: data.db not found - run DB Generation first."); return
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

        read_validated_tracks_fn = self._dependency("read_validated_tracks")
        read_corrections_fn = self._dependency("read_corrections")
        validated_tracks = (
            read_validated_tracks_fn(self._pos_dir) if self._pos_dir is not None else {}
        )
        result = self._dependency("extend_track_from_db")(
            source_id=source_id, source_frame=t, direction=direction,
            tracked_labels=tracked, db_path=db_path,
        )

        if result is None:
            self._correction_status(
                f"No DB link to extend cell {source_id} to t={target_frame}."
            ); return

        assignments = result.assignments or ()
        if not assignments:
            assignments = (SimpleNamespace(cell_id=source_id, mask_2d=result.mask_2d),)

        frame = layer.data[result.target_frame]
        before = np.asarray(frame).copy()
        corrections = (
            read_corrections_fn(self._pos_dir) if self._pos_dir is not None else []
        )
        protected_ids_at_target = protected_cell_ids_at_frame(
            validated_tracks,
            corrections,
            frame=result.target_frame,
        )
        protected_mask = protected_cell_mask(frame, protected_ids_at_target)

        assignments = tuple(
            a for a in assignments
            if int(a.cell_id) not in protected_ids_at_target
        )
        if not assignments:
            self._correction_status(
                f"Extend skipped: cell {source_id} is protected at t={result.target_frame}."
            )
            return

        changed_ids = {int(a.cell_id) for a in assignments}
        for cid in changed_ids:
            frame[frame == cid] = 0
        if self.extend_greedy_overwrite_check.isChecked():
            for a in assignments:
                frame[a.mask_2d & ~protected_mask] = int(a.cell_id)
        else:
            for a in assignments:
                frame[a.mask_2d & (frame == 0)] = int(a.cell_id)
        changed = before != frame
        changed_ids = (
            set(int(v) for v in np.unique(before[changed]))
            | set(int(v) for v in np.unique(np.asarray(frame)[changed]))
        )
        changed_ids.discard(0)
        layer.refresh()
        self._refresh_correction_label_visuals_for_edit(
            result.target_frame,
            changed_ids,
        )

        step = list(self.viewer.dims.current_step)
        step[0] = result.target_frame
        self.viewer.dims.current_step = tuple(step)

        self._refresh_track_visuals_live()
        self._correction_status(
            f"Extended cell {source_id} -> t={result.target_frame} "
            f"(link weight={result.weight:.2f})"
        )

    def _on_swap_step(self, direction: str) -> None:
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._correction_status("data.db not found - run DB Generation first."); return
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

        if self._swap_cursor is not None and (
            self._swap_cursor.source_id != source_id
            or self._swap_cursor.frame != t
        ):
            self._swap_cursor = None

        if self._swap_cursor is None:
            from skimage.measure import regionprops as _regionprops
            props = _regionprops(source_mask.astype(np.uint8))
            if not props:
                self._correction_status("Cannot compute area for source cell."); return
            src_area = int(props[0].area)

            if self._pos_dir is not None:
                corrections = read_corrections(self._pos_dir)
            else:
                corrections = []
            protected_ids = protected_cell_ids_at_frame(
                validated_tracks,
                corrections,
                frame=t,
                exclude_cell_id=source_id,
            )
            protected_mask = protected_cell_mask(tracked[t], protected_ids)

            candidates = list_swap_candidates(
                db_path=db_path,
                frame=t,
                source_mask=source_mask,
                frame_shape=tuple(tracked.shape[1:]),
                protected_mask=protected_mask,
            )
            if not candidates:
                self._correction_status(
                    "No swap candidates in the segmentation hierarchy."
                ); return

            self._swap_cursor = _SwapCursor(
                source_id=source_id,
                frame=t,
                candidates=tuple(candidates),
                index=_nearest_area_index(candidates, src_area),
                baseline_frame=tracked[t].copy(),
            )

        cursor = self._swap_cursor
        if len(cursor.candidates) <= 1:
            self._correction_status(
                "Only one node in this lattice branch — nothing to cycle."
            ); return

        idx = _cycle_index(
            len(cursor.candidates), cursor.index, larger=(direction != "smaller")
        )

        candidate = cursor.candidates[idx]
        validated_tracks_full = (
            read_validated_tracks(self._pos_dir) if self._pos_dir is not None else {}
        )
        self._apply_swap(layer, t, source_id, candidate, validated_tracks_full)
        cursor.index = idx
        self._refresh_swap_visuals_live()
        self._correction_status(
            f"Swapped cell {source_id} -> candidate {idx + 1}/{len(cursor.candidates)}"
            f" (area={candidate.area} px)"
        )

    def _apply_swap(self, layer, t: int, source_id: int, candidate: _SwapCandidate, validated_tracks: dict) -> None:
        frame = layer.data[t]
        before = frame.copy()

        cursor = self._swap_cursor
        if (
            cursor is not None
            and cursor.baseline_frame is not None
            and cursor.source_id == source_id
            and cursor.frame == t
            and cursor.baseline_frame.shape == frame.shape
        ):
            frame[:] = cursor.baseline_frame

        if self._pos_dir is not None:
            corrections = read_corrections(self._pos_dir)
        else:
            corrections = []
        protected_ids = protected_cell_ids_at_frame(
            validated_tracks,
            corrections,
            frame=t,
            exclude_cell_id=source_id,
        )
        protected_mask = protected_cell_mask(frame, protected_ids)

        frame[frame == source_id] = 0
        paintable = candidate.mask_2d & ~protected_mask
        frame[paintable] = source_id

        self.correction_widget._record_history(layer, t, before)
        layer.refresh()

    # ── candidate gallery (clickable extend / swap thumbnails) ─────────────────

    def _extend_kwargs(self) -> dict:
        """Extra kwargs for list_extend_candidates (none: extend is LinkDB-driven)."""
        return {}

    def _gallery_protected_mask(self, t: int) -> np.ndarray | None:
        """Protected-pixel mask at ``t`` for swap candidates (excludes the source)."""
        pos_dir = self._pos_dir
        layer = self._correction_tracked_layer()
        if pos_dir is None or layer is None:
            return None
        frame = frame_view_2d(np.asarray(layer.data), t)
        if frame is None:
            return None
        source_id = int(getattr(self.correction_widget, "_selected_label", 0) or 0)
        validated_tracks = read_validated_tracks(pos_dir)
        corrections = read_corrections(pos_dir)
        protected_ids = protected_cell_ids_at_frame(
            validated_tracks,
            corrections,
            frame=t,
            exclude_cell_id=source_id,
        )
        return protected_cell_mask(frame, protected_ids)

    def _refresh_candidate_gallery_if_shown(self) -> None:
        # Debounced: (re)start the timer so a burst of frame/selection changes
        # collapses into a single rebuild once things settle (see the timer
        # set-up in _setup_ui). The fired handler re-checks visibility.
        if self.candidate_gallery_check.isChecked():
            self._candidate_refresh_timer.start()
        else:
            self._candidate_refresh_timer.stop()

    def _refresh_candidate_gallery_now(self) -> None:
        """Do the actual (expensive) gallery rebuild — only via the debounce timer."""
        if self.candidate_gallery_check.isChecked():
            self._candidate_gallery.refresh()

    def _apply_swap_candidate(self, candidate) -> None:
        """Apply a swap candidate picked from the gallery (mirrors Z/C apply)."""
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        source_id = int(getattr(self.correction_widget, "_selected_label", 0) or 0)
        if not source_id:
            self._correction_status("Swap: no cell selected (left-click first)."); return
        t = self._current_t()
        validated_tracks = (
            read_validated_tracks(self._pos_dir) if self._pos_dir is not None else {}
        )
        if source_id in validated_tracks:
            self._correction_status("Cannot swap a validated cell."); return
        # A direct gallery pick is independent of any in-flight Z/C cursor.
        self._swap_cursor = None
        self._apply_swap(layer, t, source_id, candidate, validated_tracks)
        self._refresh_swap_visuals_live()
        self._correction_status(
            f"Swapped cell {source_id} -> candidate (area={int(candidate.area)} px). Unsaved."
        )

    def _apply_extend_candidate(self, which: str, target_frame: int, assignment) -> None:
        """Apply an extend candidate picked from the gallery (mirrors A/D apply)."""
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        if target_frame < 0 or target_frame >= layer.data.shape[0]:
            self._correction_status("Extend target frame out of range."); return
        changed_ids = self._paint_extend_assignment(
            layer,
            target_frame,
            assignment,
            greedy=self.extend_greedy_overwrite_check.isChecked(),
        )
        if not changed_ids:
            self._correction_status(
                f"Extend skipped: cell {int(assignment.cell_id)} is protected "
                f"at t={target_frame}."
            )
            return
        step = list(self.viewer.dims.current_step)
        if step:
            step[0] = int(target_frame)
            self.viewer.dims.current_step = tuple(step)
        self._refresh_track_visuals_live()
        self._correction_status(
            f"Extended cell {int(assignment.cell_id)} -> t={target_frame} "
            f"(link weight={float(assignment.weight):.2f}). Unsaved."
        )

    def _paint_extend_assignment(
        self, layer, target_frame: int, assignment, *, greedy: bool
    ) -> set[int]:
        """Paint one extend assignment at ``target_frame`` honoring protection.

        Returns the set of changed cell ids (empty when the source is protected
        at the target, or nothing changed).
        """
        frame = layer.data[target_frame]
        before = np.asarray(frame).copy()
        validated_tracks = (
            read_validated_tracks(self._pos_dir) if self._pos_dir is not None else {}
        )
        corrections = (
            read_corrections(self._pos_dir) if self._pos_dir is not None else []
        )
        protected_ids = protected_cell_ids_at_frame(
            validated_tracks, corrections, frame=target_frame
        )
        cid = int(assignment.cell_id)
        if cid in protected_ids:
            return set()
        protected_mask = protected_cell_mask(frame, protected_ids)
        frame[frame == cid] = 0
        if greedy:
            frame[assignment.mask_2d & ~protected_mask] = cid
        else:
            frame[assignment.mask_2d & (frame == 0)] = cid
        changed = before != frame
        changed_ids = (
            set(int(v) for v in np.unique(before[changed]))
            | set(int(v) for v in np.unique(np.asarray(frame)[changed]))
        )
        changed_ids.discard(0)
        if not changed_ids:
            return set()
        layer.refresh()
        self._refresh_correction_label_visuals_for_edit(target_frame, changed_ids)
        return changed_ids

    def _on_retrack_forward(self) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._correction_status("Need >= 2 frames to retrack."); return
        t0 = int(self.viewer.dims.current_step[0])
        if t0 >= layer.data.shape[0] - 1:
            self._correction_status("Already at last frame."); return

        before = np.asarray(layer.data).copy()
        validated_tracks = read_validated_tracks(self._pos_dir)
        result = retrack_stack_direction(
            before,
            start_frame=t0,
            direction="forward",
            fully_validated_frames=read_validated_frames(self._pos_dir),
            validated_cells_at_frame=lambda t: {
                cid for cid, frames in validated_tracks.items() if t in frames
            },
            retrack_frame=retrack_frame_constrained,
            max_dist_px=float(self.retrack_max_dist_spin.value()),
            reserved_ids=set(validated_tracks),
            area_weight=float(self.extend_area_weight_spin.value()),
            iou_weight=float(self.extend_iou_weight_spin.value()),
            distance_weight=float(self.extend_distance_weight_spin.value()),
        )
        layer.data = result.stack
        self._refresh_correction_label_visuals_for_changed_frames(before, result.stack)
        self._refresh_track_visuals_live()
        self._correction_status(
            f"Retrackedforward from t={result.first_target_frame}: "
            f"{result.n_retracked} updated, "
            f"{result.n_skipped} validated skipped. Unsaved."
        )

    def _on_retrack_backward(self) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._correction_status("Need >= 2 frames to retrack."); return
        t0 = int(self.viewer.dims.current_step[0])
        if t0 <= 0:
            self._correction_status("Already at first frame."); return

        before = np.asarray(layer.data).copy()
        validated_tracks = read_validated_tracks(self._pos_dir)
        result = retrack_stack_direction(
            before,
            start_frame=t0,
            direction="backward",
            fully_validated_frames=read_validated_frames(self._pos_dir),
            validated_cells_at_frame=lambda t: {
                cid for cid, frames in validated_tracks.items() if t in frames
            },
            retrack_frame=retrack_frame_constrained,
            max_dist_px=float(self.retrack_max_dist_spin.value()),
            reserved_ids=set(validated_tracks),
            area_weight=float(self.extend_area_weight_spin.value()),
            iou_weight=float(self.extend_iou_weight_spin.value()),
            distance_weight=float(self.extend_distance_weight_spin.value()),
        )
        layer.data = result.stack
        self._refresh_correction_label_visuals_for_changed_frames(before, result.stack)
        self._refresh_track_visuals_live()
        self._correction_status(
            f"Retrackedbackward from t={result.first_target_frame}: "
            f"{result.n_retracked} updated, "
            f"{result.n_skipped} validated skipped. Unsaved."
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
        try:
            result = remove_unvalidated_from_data(
                data,
                validated_tracks,
            )
        except ValueError as exc:
            self._correction_status(str(exc)); return

        if not result.changed_pixels:
            self._correction_status("No unvalidated labels found."); return
        layer.refresh()
        self._refresh_correction_label_visuals()
        if self.correction_widget._selected_label:
            ct = self._current_t()
            if self.correction_widget._selected_label not in self._current_cell_ids(ct):
                self.correction_widget.select_label(ct, 0)
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        self._refresh_lineage_canvas_if_shown()
        self._correction_status(
            f"Removed unvalidated labels in {result.changed_frames} frame(s), "
            f"{result.changed_pixels} px changed. Unsaved."
        )

    def _remove_unvalidated_from_layer(self, layer) -> tuple[int, int]:
        if self._pos_dir is None:
            return 0, 0
        data = np.asarray(layer.data)
        validated_tracks = read_validated_tracks(self._pos_dir)
        result = remove_unvalidated_from_data(
            data,
            validated_tracks,
        )
        if result.changed_pixels:
            layer.refresh()
            self._refresh_correction_label_visuals()
            if self.correction_widget._selected_label:
                ct = self._current_t()
                if self.correction_widget._selected_label not in self._current_cell_ids(ct):
                    self.correction_widget.select_label(ct, 0)
            self._refresh_validated_overlay()
            self._refresh_validation_counter()
        return result.changed_frames, result.changed_pixels

    def _on_commit(self) -> None:
        tracked_path = self._tracked_path()
        if tracked_path is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer to commit."); return
        if layer.data.ndim != 3:
            self._correction_status("Tracked layer is not a 3D stack."); return
        try:
            result = prepare_committed_labels(
                np.asarray(layer.data),
                read_validated_tracks(self._pos_dir),
            )
        except Exception as exc:
            self._on_correction_worker_error(exc)
            return
        layer.data = result.stack
        if result.old_to_new:
            remap_validated_tracks(self._pos_dir, result.old_to_new)
        if result.changed_pixels:
            layer.refresh()
            self._refresh_correction_label_visuals()
            if self.correction_widget._selected_label:
                ct = self._current_t()
                if self.correction_widget._selected_label not in self._current_cell_ids(ct):
                    self.correction_widget.select_label(ct, 0)
            self._refresh_validated_overlay()
            self._refresh_validation_counter()
            self._refresh_lineage_canvas_if_shown()
        for t in range(int(layer.data.shape[0])):
            write_tracked_frame(tracked_path, t, np.asarray(layer.data[t]))
        self._correction_status(
            f"Committed {result.n_cells} cell(s); removed {result.changed_pixels} px in "
            f"{result.changed_frames} frame(s); saved to {tracked_path.name}."
        )

    def _on_correction_worker_error(self, exc: Exception) -> None:
        self._correction_status(f"Error: {exc}")
        logger.exception("Correction worker error", exc_info=exc)

    def _install_correction_shortcuts(self) -> None:
        # Correction hotkeys are bound on the active tracked Labels layer via
        # napari's keymap (see _bind_correction_keys), NOT Qt QShortcuts. The
        # single-letter keys collide with napari's built-in keybindings
        # (Z=pan_zoom, E=erase, B=preserve_labels, C=auto_contrast, S=select,
        # A=select_all, D=direct_mode). A Qt ApplicationShortcut only wins that
        # race intermittently — once it goes ambiguous (e.g. the widget is
        # recreated and its app-global QShortcuts linger) the keypress falls
        # through to napari, which flips the layer mode instead of swapping.
        # layer.bind_key(overwrite=True) is the only mechanism that
        # deterministically takes precedence. "V" is intentionally omitted: it
        # is already bound at the viewer level (NucleusWorkflowWidget) and
        # surfaces here while the tracked layer is active.
        self._correction_key_specs = [
            ("A", lambda: self._on_extend(direction="backward")),
            ("D", lambda: self._on_extend(direction="forward")),
            ("Q", self._on_retrack_backward),
            ("E", self._on_retrack_forward),
            ("B", self._on_anchor_here),
            ("S", self._on_save_tracked),
            ("Z", lambda: self._on_swap_step(direction="smaller")),
            ("C", lambda: self._on_swap_step(direction="larger")),
        ]
        self._bound_correction_keys: list[str] = []
        self._correction_keys_layer = None

    def _bind_correction_keys(self) -> None:
        """Bind correction hotkeys on the active tracked Labels layer.

        Layer-level keymap entries take precedence over both napari's built-in
        class keybindings and the viewer keymap, so this overrides the
        conflicting defaults (Z, E, B, …) only while correction is active.
        """
        self._unbind_correction_keys()
        layer = self._correction_tracked_layer()
        if layer is None:
            return
        for key, slot in self._correction_key_specs:
            def handler(_layer, _slot=slot):
                _slot()

            layer.bind_key(key, handler, overwrite=True)
            self._bound_correction_keys.append(key)
        self._correction_keys_layer = layer

    def _unbind_correction_keys(self) -> None:
        layer = self._correction_keys_layer
        for key in self._bound_correction_keys:
            try:
                if layer is not None:
                    layer.bind_key(key, None)
            except Exception:
                pass
        self._bound_correction_keys = []
        self._correction_keys_layer = None

    def _on_correction_active_button_toggled(self, active: bool) -> None:
        if active:
            # Capture the plugin dock + width now, before the focus takeover
            # reparents the header out of it, so its width can be restored later.
            self._pre_correction_dock = self._find_plugin_dock()
            self._pre_correction_dock_width = (
                self._pre_correction_dock.width()
                if self._pre_correction_dock is not None
                else None
            )
            self._capture_correction_view_state()
            hide_all_layers(self.viewer.layers)

            if not self._load_correction_layers_from_disk():
                self._restore_correction_view_state()
                old = self.active_btn.blockSignals(True)
                try:
                    self.active_btn.setChecked(False)
                finally:
                    self.active_btn.blockSignals(old)
                self.correction_widget.deactivate()
                self._correction_active_content_visible = False
                self._sync_correction_panel_visibility()
                return
            layer = self.viewer.layers[_CORRECTION_TRACKED_LAYER]
            layer.visible = True
            self.viewer.layers.selection.active = layer
            self.correction_widget.activate_layer(layer)
            # Drop any pre-loaded higher-rank stack (e.g. a raw T,Z,Y,X z-stack)
            # so the viewer collapses to one frame slider that matches the
            # correction frames; re-added verbatim on deactivate. Same-rank
            # layers are left alone so nothing essential is removed.
            self._detached_stack_layers = detach_higher_dim_stacks(
                self.viewer.layers,
                max_ndim=int(np.asarray(layer.data).ndim),
                keep_names=self._correction_owned_layers,
            )
            # Focus mode: hand the correction panels the right column by hiding
            # napari's native layer-list / layer-controls docks.
            self._native_dock_state = hide_native_docks(self.viewer)
            self._set_checked_without_signal(self.params_btn, False)
            self._set_checked_without_signal(self.shortcuts_btn, False)
            self.extend_retrack_params_section._toggle.setChecked(False)
            self.shortcuts_section._toggle.setChecked(False)
            self._correction_active_content_visible = True
            self._sync_correction_panel_visibility()
            # Open both focus-mode panels by default: the lineage canvas is the
            # headline view, the candidate galleries are the action surface.
            # Set the boxes silently and render explicitly so the splitter is
            # built once, below, rather than on every box's toggle signal.
            self._set_checked_without_signal(self.lineage_canvas_check, True)
            self._set_checked_without_signal(self.candidate_gallery_check, True)
            self._lineage_canvas.refresh()
            self._candidate_gallery.refresh()
            # Build the whole-stack validated / anchor overlays once now; they're
            # only rebuilt on validation or label changes afterwards (not per frame).
            self._refresh_validated_overlay()
            self._refresh_validation_counter()
            # Build the controls strip (header + lineage overview + controls),
            # reparenting the controls out of the plugin dock; then hand the whole
            # window to the workspace by hiding the now-empty plugin dock and lay
            # the panels out as one splitter so they resize independently.
            self._build_workspace_controls_dock()
            self._focus_takeover(True)
            self._arrange_workspace_docks()
            # Open the workspace at half the window width with its three strips
            # (candidate gallery · film strip · controls) split in equal thirds.
            # Deferred so the dock is laid out before resizeDocks runs.
            QTimer.singleShot(0, self._size_workspace_dock)
            return

        if not self._confirm_deactivate_with_unsaved_changes():
            self._set_checked_without_signal(self.active_btn, True)
            self._correction_active_content_visible = True
            self._sync_correction_panel_visibility()
            return

        self._set_checked_without_signal(self.active_btn, False)
        self._correction_active_content_visible = False
        # Reset the single-cell focus presentation before tearing layers down so
        # the show_selected_label flag and overview visibility don't leak.
        self._apply_focus_presentation(0)
        self.correction_widget.deactivate()
        self._unbind_correction_keys()
        # Refresh the main Tracked layer from disk so a subsequent re-solve
        # picks up any corrections the user saved during this session.
        self._refresh_tracked_layer_from_disk()
        self._remove_correction_owned_layers()
        # Reparent the correction header + controls back into the plugin dock and
        # show it again *before* removing the now-empty workspace strip (so those
        # widgets aren't deleted with the strip's container), then tear panels down.
        self._focus_takeover(False)
        self._teardown_workspace_controls_dock()
        self._teardown_focus_panels()
        restore_native_docks(self.viewer, self._native_dock_state)
        self._native_dock_state = {}
        # Put back any higher-dim stack removed on activate (data/contrast/
        # colormap intact) before restoring visibility/selection over it.
        reattach_layers(self.viewer.layers, self._detached_stack_layers)
        self._detached_stack_layers = []
        self._restore_correction_view_state()
        self._restore_pre_correction_dock_width()
        self._set_checked_without_signal(self.params_btn, False)
        self._set_checked_without_signal(self.shortcuts_btn, False)
        self.extend_retrack_params_section._toggle.setChecked(False)
        self.shortcuts_section._toggle.setChecked(False)
        self._sync_correction_panel_visibility()

    def _confirm_deactivate_with_unsaved_changes(self) -> bool:
        if not self._correction_dirty:
            return True
        choice = QMessageBox.question(
            self,
            "Save correction changes?",
            (
                "Correction mode has unsaved changes. "
                "Save tracked labels before turning correction mode off?"
            ),
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if choice == QMessageBox.Cancel:
            return False
        if choice == QMessageBox.Save:
            self._on_save_tracked()
            self._correction_dirty = False
        elif choice == QMessageBox.Discard:
            self._correction_dirty = False
        return True

    def _on_correction_mode_toggled(self, active: bool) -> None:
        if not active:
            self._swap_cursor = None
            self._clear_track_path_overlay()
            self._lineage_canvas.teardown()
            self._candidate_gallery.teardown()
        if active:
            self._bind_correction_keys()
        else:
            self._unbind_correction_keys()

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
            for correction in corrections_for_label_frames(
                data,
                cell_id=sel,
                frames=frames,
            ):
                add_correction(self._pos_dir, correction)
            self._correction_status(
                f"Cell {sel} validated across {len(frames)} frame(s)."
            )
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        self._refresh_lineage_canvas_if_shown()

    def on_dims_step_changed(self) -> None:
        self._swap_cursor = None
        # The validated / anchor overlays are whole-stack masks (built on
        # activation and on each validation/label change), so napari already
        # shows the right slice — no per-frame rebuild here.
        # Keep the focused track's tip cross on the cell in the new frame.
        self._all_tracks.set_current_frame(self._current_t())
        if self.lineage_canvas_check.isChecked():
            self._lineage_canvas.set_current_frame(self._current_t())
        # Candidates are frame-specific (the swap lattice + adjacent-frame extend),
        # so a frame change invalidates them — recompute for the new frame.
        self._refresh_candidate_gallery_if_shown()

    def _refresh_validated_overlay(self) -> None:
        self._validated_overlay.refresh_overlay(frame_view_2d)

    def _add_validated_overlay(self, data: np.ndarray) -> None:
        self._validated_overlay.add_overlay(data)

    def _place_validated_overlay_below_spotlight(self) -> None:
        self._validated_overlay.place_below_spotlight()

    # ── Whole-track temporal overlay ("comet") ─────────────────────────────────

    def _on_toggle_track_path(self, checked: bool) -> None:
        if checked:
            self._refresh_track_path_overlay()
        else:
            self._clear_track_path_overlay()
        self._refresh_track_path_spotlight()

    def _on_track_selection_changed(self, _t: int, _lab: int) -> None:
        """Recolour the all-tracks layer / canvas detail when selection changes."""
        self._apply_focus_presentation(_lab)
        if self.lineage_canvas_check.isChecked():
            self._lineage_canvas.set_selection(_lab)
            # Recenter the canvas on the selected track only when the selection
            # came from the image viewer; a lineage click already shows the row.
            if not self._navigating_from_lineage:
                self._lineage_canvas.center_on_track(_lab)
        self._refresh_candidate_gallery_if_shown()

    def _apply_focus_presentation(self, lab: int) -> None:
        """Focus a single cell in the all-tracks layer.

        The selected track is recoloured bright viridis-by-time and gets a
        current-frame tip cross while every other track fades to a faint grey;
        ``lab == 0`` restores the overview. The label spotlight (dimming the
        other cells' masks) is handled by the inner correction widget; this only
        drives the nucleus-side track overlay.
        """
        self._all_tracks.set_focus(int(lab or 0))

    def _refresh_track_visuals_live(self) -> None:
        """Rebuild the all-tracks overlay + film strip after an edit.

        Called when the selected track's pixels change (swap / extend / retrack)
        so both views reflect the new trajectory without reselecting the cell.
        """
        if self.track_path_check.isChecked():
            self._refresh_track_path_overlay()
            self._refresh_track_path_spotlight()
        self._refresh_lineage_canvas_if_shown()
        self._refresh_candidate_gallery_if_shown()

    def _refresh_swap_visuals_live(self) -> None:
        """Cheap post-swap refresh used while stepping candidates with Z / C.

        Updates the all-tracks overlay and the selected track's detail strip
        only. The full :meth:`_refresh_lineage_canvas_if_shown` re-runs the
        lineage build over the *whole* stack, which froze the GUI when fired on
        every swap keystroke; the overview catches up on the next full refresh.
        """
        if self.track_path_check.isChecked():
            self._refresh_track_path_overlay()
            self._refresh_track_path_spotlight()
        if self.lineage_canvas_check.isChecked():
            self._lineage_canvas.refresh_detail()
        self._refresh_candidate_gallery_if_shown()

    def _refresh_lineage_canvas_if_shown(self) -> None:
        """Rebuild the lineage canvas (overview + detail) after a label change.

        The swimlane overview is structural in the controls strip and stays
        visible the whole time focus mode is active — independent of the
        ``lineage_canvas_check`` toggle, which only shows/hides the film-strip
        *detail*. So the rebuild is gated on the workspace being docked, not on
        that checkbox; otherwise the overview goes stale after a label change
        (reassign IDs, remove unvalidated, validate, …) whenever the detail
        strip happens to be toggled off.
        """
        if self._workspace_splitter is not None:
            self._lineage_canvas.refresh()

    def _track_path_spotlight_mask(self, _t: int, lab: int, _default_mask):
        return self._all_tracks.spotlight_mask(_t, lab, _default_mask)

    def _refresh_track_path_spotlight(self) -> None:
        """Re-run the inner highlight so the spotlight mask provider is consulted."""
        cw = self.correction_widget
        lab = int(getattr(cw, "_selected_label", 0) or 0)
        if not lab:
            return
        try:
            cw._update_highlight(self._current_t(), lab, notify=False)
        except Exception:
            logger.exception("track path spotlight refresh failed")

    def _refresh_track_path_overlay(self) -> None:
        self._all_tracks.refresh()

    def _clear_track_path_overlay(self) -> None:
        self._all_tracks.clear()

    def _film_strip_intensity_layer(self):
        """Best raw layer to crop tiles from (nucleus, then cell)."""
        for name in (
            _CORRECTION_NUC_ZAVG_LAYER,
            _CORRECTION_CELL_ZAVG_LAYER,
        ):
            if name in self.viewer.layers:
                return self.viewer.layers[name]
        return None

    # -- Focus-mode panels (lineage canvas) --------------------------------

    def _on_toggle_lineage_canvas(self, checked: bool) -> None:
        # Toggle-on rebuilds the film strip; the swimlane overview is structural
        # in the controls strip and stays put. _arrange_workspace_docks slots the
        # panel into the splitter (if needed) and shows/hides it to match.
        if checked:
            self._lineage_canvas.refresh()
        self._arrange_workspace_docks()

    def _on_toggle_candidate_gallery(self, checked: bool) -> None:
        if checked:
            self._candidate_gallery.refresh()
        self._arrange_workspace_docks()

    def _navigate_to_error(self, t: int, cell_id: int) -> None:
        """Jump the viewer to frame ``t``, select ``cell_id`` and center on it.

        Called when a track is clicked in the lineage canvas: besides stepping
        to the clicked frame and selecting the cell, this pans the image-viewer
        camera so the cell sits in the middle of the canvas.
        """
        try:
            step = list(self.viewer.dims.current_step)
            if step:
                step[0] = int(t)
                self.viewer.dims.current_step = tuple(step)
        except Exception:
            logger.exception("focus-mode navigation: frame jump failed")
        # Guard so the resulting selection callback does not scroll the lineage
        # canvas off the row the user just clicked there.
        self._navigating_from_lineage = True
        try:
            self.correction_widget.select_label(int(t), int(cell_id))
        except Exception:
            logger.exception("focus-mode navigation: cell select failed")
        finally:
            self._navigating_from_lineage = False
        self._center_viewer_on_cell(int(t), int(cell_id))

    def _center_viewer_on_cell(self, t: int, cell_id: int) -> None:
        """Pan the napari camera onto cell ``cell_id`` at frame ``t``."""
        layer = self._correction_tracked_layer()
        if layer is None or not cell_id:
            return
        try:
            seg2d = frame_view_2d(np.asarray(layer.data), int(t))
            if seg2d is None:
                return
            ys, xs = np.nonzero(seg2d == int(cell_id))
            if ys.size == 0:
                return
            cy, cx = float(ys.mean()), float(xs.mean())
            data_coord = (
                (int(t), cy, cx) if np.asarray(layer.data).ndim >= 3 else (cy, cx)
            )
            world = layer.data_to_world(data_coord)
            center = list(self.viewer.camera.center)
            center[-2:] = [float(world[-2]), float(world[-1])]
            self.viewer.camera.center = tuple(center)
        except Exception:
            logger.exception("lineage navigation: camera centering failed")

    def _find_plugin_dock(self):
        """The QDockWidget hosting the correction header, walking up from it.

        Resolved from ``self.header`` (still parented into the plugin dock at
        activation start, before the focus takeover reparents it out).
        """
        from qtpy.QtWidgets import QDockWidget

        widget = self.header.parentWidget()
        while widget is not None:
            if isinstance(widget, QDockWidget):
                return widget
            widget = widget.parentWidget()
        return None

    def _main_window(self):
        return getattr(getattr(self.viewer, "window", None), "_qt_window", None)

    def _size_workspace_dock(self) -> None:
        """Open the workspace at half the window width, panels split in thirds."""
        win = self._main_window()
        dock = self._workspace_dock
        splitter = self._workspace_splitter
        if win is None or dock is None or splitter is None:
            return
        try:
            target = max(int(win.width()) // 2, 1)
            win.resizeDocks([dock], [target], Qt.Horizontal)
            third = max(target // 3, 1)
            splitter.setSizes([third, third, third])
        except Exception:
            logger.exception("could not size the correction workspace dock")

    def _restore_pre_correction_dock_width(self) -> None:
        """Shrink the plugin dock back to its pre-correction (focus-mode) width."""
        dock = self._pre_correction_dock
        width = self._pre_correction_dock_width
        self._pre_correction_dock = None
        self._pre_correction_dock_width = None
        win = self._main_window()
        if dock is None or width is None or win is None:
            return
        # Deferred: the plugin dock has only just been re-shown, so let Qt lay it
        # back out before forcing its width.
        QTimer.singleShot(
            0,
            lambda: self._apply_dock_width(win, dock, int(width)),
        )

    @staticmethod
    def _apply_dock_width(win, dock, width: int) -> None:
        try:
            win.resizeDocks([dock], [width], Qt.Horizontal)
        except Exception:
            logger.exception("could not restore the plugin dock width")

    def _build_workspace_controls_dock(self):
        """Build the controls strip: correction header + controls + lineage overview.

        Reparents ``self.header`` and ``self.section`` (the toolbar / params /
        shortcuts / correction widget / checkboxes / status) out of the plugin
        dock into a single container, with the lineage swimlane overview (the
        track renders) embedded *below* them. The reparent reuses every existing
        button + signal as-is. The container is the rightmost panel of the
        workspace splitter built by :meth:`_arrange_workspace_docks`; the plugin
        dock is hidden by the caller once this succeeds. Idempotent.
        """
        if self._controls_container is not None:
            return self._controls_container
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.header)
        lay.addWidget(self.section)
        lay.addWidget(self._lineage_canvas.overview_panel(), stretch=1)
        self._controls_container = container
        return container

    def _teardown_workspace_controls_dock(self) -> None:
        """Remove the single workspace dock (header/section already reparented out).

        Destroying the dock also deletes the splitter and every embedded panel
        (candidate gallery, film strip, controls strip); the controllers drop
        their stale references in their own ``teardown``.
        """
        if self._workspace_dock is not None:
            try:
                self.viewer.window.remove_dock_widget(self._workspace_dock)
            except Exception:
                logger.exception("could not remove the correction workspace dock")
        self._workspace_dock = None
        self._workspace_splitter = None
        self._controls_container = None

    def _arrange_workspace_docks(self) -> None:
        """Lay the right-column panels out as one splitter of side-by-side strips.

        All three panels live in a single ``QSplitter`` inside one napari dock,
        left→right: candidate gallery · film strip · controls (header + lineage
        overview). Hosting them in a splitter — rather than three separate
        napari docks — is what makes resizing intuitive: napari's dock area
        redistributes the canvas/dock boundary across *all* docks proportionally,
        whereas a splitter handle moves only the two panels it sits between, and
        the leftmost panel (the only one with a non-zero stretch factor) absorbs
        any change to the dock's outer width.

        The splitter is created on first call (panels are materialised eagerly
        as bare widgets via the controllers) and reused afterwards; each call
        also shows/hides the candidate and film panels to match their toggles.
        """
        splitter = self._ensure_workspace_splitter()
        if splitter is None:
            return
        self._candidate_gallery.widget().setVisible(
            self.candidate_gallery_check.isChecked()
        )
        self._lineage_canvas.film_widget().setVisible(
            self.lineage_canvas_check.isChecked()
        )

    def _ensure_workspace_splitter(self) -> QSplitter | None:
        """Create the workspace splitter + dock once, returning it (or None)."""
        if self._workspace_splitter is not None:
            return self._workspace_splitter
        if self._controls_container is None:
            self._build_workspace_controls_dock()
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        # Default handles are a ~3 px sliver and near-invisible on the dark
        # theme. Widen them and paint a centred grip so they're easy to grab.
        # Explicit greys (not palette roles, which napari leaves unset — they
        # resolved to white): a mid-grey grip inset against the dark dock bg.
        splitter.setHandleWidth(10)
        splitter.setStyleSheet(
            "QSplitter::handle:horizontal {"
            " background: #5a606b;"
            " border-left: 4px solid #2e3440;"
            " border-right: 4px solid #2e3440;"
            " }"
            "QSplitter::handle:horizontal:hover { background: #7a828f; }"
        )
        # left→right: candidate gallery · film strip · controls strip.
        splitter.addWidget(self._candidate_gallery.widget())
        splitter.addWidget(self._lineage_canvas.film_widget())
        splitter.addWidget(self._controls_container)
        # Only the leftmost panel grows when the dock's outer edge is dragged;
        # the others keep their width until a handle between them is moved.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 0)
        try:
            self._workspace_dock = self.viewer.window.add_dock_widget(
                splitter, name="Correction", area="right"
            )
        except Exception:
            logger.exception("could not dock the correction workspace splitter")
            self._workspace_dock = None
            return None
        self._workspace_splitter = splitter
        return splitter

    def _teardown_focus_panels(self) -> None:
        """Undock the focus-mode panels (lineage canvas, candidate gallery)."""
        self._set_checked_without_signal(self.lineage_canvas_check, False)
        self._lineage_canvas.teardown()
        self._set_checked_without_signal(self.candidate_gallery_check, False)
        self._candidate_gallery.teardown()

    def _refresh_validation_counter(self) -> None:
        self._validated_overlay.refresh_counter(self.validation_counter_lbl)

    def _on_cells_edited(self, t: int, changed_ids: set[int]) -> None:
        self._refresh_correction_label_visuals_for_edit(t, changed_ids)
        self._validated_overlay.on_cells_edited(
            t,
            changed_ids,
            frame_view_2d=frame_view_2d,
            counter_label=self.validation_counter_lbl,
        )
        # Mask edits (draw / merge / relabel / redraw / fill / split) funnel
        # through here, while retrack/extend/swap drive their own *_visuals_live
        # paths. If the focused cell's pixels changed, its track-path comet +
        # spotlight are now stale, so rebuild them here too.
        sel = int(getattr(self.correction_widget, "_selected_label", 0) or 0)
        if sel and sel in {int(v) for v in changed_ids}:
            if self.track_path_check.isChecked():
                self._refresh_track_path_overlay()
                self._refresh_track_path_spotlight()
        # Rebuild the canvas too (its overview presence + detail strip go stale
        # otherwise) — the retrack/extend paths do this separately.
        self._refresh_lineage_canvas_if_shown()
        self._refresh_candidate_gallery_if_shown()

    def _frames_with_cell(self, cell_id: int) -> list[int]:
        return self._validated_overlay.frames_with_cell(cell_id)

    def _manual_correction_protected_mask(
        self,
        t: int,
        frame: np.ndarray,
    ) -> np.ndarray:
        pos_dir = self._pos_dir
        if pos_dir is None:
            return np.zeros_like(frame, dtype=bool)
        validated_tracks = self._dependency("read_validated_tracks")(pos_dir)
        corrections = self._dependency("read_corrections")(pos_dir)
        protected_ids = protected_cell_ids_at_frame(
            validated_tracks,
            corrections,
            frame=t,
        )
        return protected_cell_mask(frame, protected_ids)

    def _refresh_correction_label_visuals_for_edit(
        self,
        t: int,
        changed_ids: set[int],
    ) -> None:
        if _CORRECTION_TRACKED_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[_CORRECTION_TRACKED_LAYER]
        ensure_label_colormap_entries(
            layer,
            changed_ids,
            color_scale=_NUCLEUS_TRACK_COLOR_SCALE,
        )
        try:
            self.viewer.layers.selection.active = layer
        except Exception:
            pass

    def _refresh_correction_label_visuals_for_changed_frames(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> None:
        before_arr = np.asarray(before)
        after_arr = np.asarray(after)
        if before_arr.shape != after_arr.shape or before_arr.ndim != 3:
            self._refresh_correction_label_visuals()
            return

        for t in range(after_arr.shape[0]):
            before_frame = before_arr[t]
            after_frame = after_arr[t]
            if np.array_equal(before_frame, after_frame):
                continue
            changed_mask = before_frame != after_frame
            changed_ids = {
                int(value)
                for value in np.unique(np.concatenate(
                    [before_frame[changed_mask], after_frame[changed_mask]]
                ))
                if int(value) != 0
            }
            self._refresh_correction_label_visuals_for_edit(t, changed_ids)

    def _refresh_correction_label_visuals(self) -> None:
        if _CORRECTION_TRACKED_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[_CORRECTION_TRACKED_LAYER]
        data = np.asarray(layer.data)
        refresh_label_colormap(
            layer,
            data,
            color_scale=_NUCLEUS_TRACK_COLOR_SCALE,
        )
        # The all-tracks overlay's geometry follows the labels — rebuild it so a
        # reassign / remove-unvalidated / retrack reshapes the trajectories too.
        self._all_tracks.refresh()
        try:
            self.viewer.layers.selection.active = layer
        except Exception:
            pass
