"""Correction section widget for the nucleus workflow.

The public class :class:`NucleusCorrectionWidget` is a coordinator over a set of
focused collaborators (controllers + stateless helper modules + the
``tracking_ultrack`` algorithms); see its docstring for the architecture map.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker as _thread_worker
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._correction_centroids import (
    apply_neutral_label_colormap,
    ensure_label_colormap_entries,
    refresh_label_colormap,
)
from cellflow.napari._correction_anchor import (
    anchor_correction,
    without_anchor_correction,
)
from cellflow.napari._correction_utils import (
    frame_view_2d,
    reassign_ids_ordered,
    retrack_stack_direction,
    track_order_by_frame_and_size,
)
from cellflow.napari._correction_commit import (
    prepare_committed_labels,
    remove_unvalidated_from_data,
)
from cellflow.napari._correction_layer_lifecycle import (
    CorrectionViewStateMixin,
    LayerViewState,
    detach_higher_dim_stacks,
    hide_all_layers,
    reattach_layers,
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
from cellflow.napari._paths import NucleusWorkspace
from cellflow.core.label_store import (
    read_full_tracked_stack,
    write_tracked_frame,
)
from cellflow.tracking_ultrack.validation_state import (
    add_anchor,
    add_correction,
    add_corrections,
    invalidate_track,
    is_track_validated,
    read_corrections,
    read_validated_frames,
    read_validated_tracks,
    remap_validated_tracks,
    write_corrections,
)
from cellflow.napari.ui_style import (
    stage_header_label,
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
from cellflow.napari._correction_ui_nucleus import build_nucleus_correction_ui
from cellflow.napari._correction_paint import paint_assignments
from cellflow.napari._correction_events import CorrectionEvents
from cellflow.napari._correction_keymap import HeldKeyRepeater
from cellflow.napari._correction_navigation import center_viewer_on_cell
from cellflow.napari._correction_playback import (
    nav_repeat_interval_ms,
    playback_loops,
)
from cellflow.napari._correction_ui import (
    CollapsiblePane,
    confirm_unsaved_before_deactivate,
    set_checked_without_signal,
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
_NUCLEUS_TRACK_COLOR_SCALE = 1.0
# Filled (by-ID) view draws cells solid; outline view keeps them semi-transparent
# so the reference images underneath stay visible through the contours.
_FILLED_LABEL_OPACITY = 1.0
_OUTLINE_LABEL_OPACITY = 0.7
# Width a collapsed workspace pane (gallery / accordion) shrinks to — must match
# the slim show-tab in CollapsiblePane.
_PANEL_STRIP_W = 24

_DEFAULT_DEPENDENCIES = {
    "annotate_database_from_corrections": _annotate_database_from_corrections,
    "extend_track_from_db": _extend_track_from_db,
    "read_corrections": lambda *args, **kwargs: read_corrections(*args, **kwargs),
    "read_validated_tracks": (
        lambda *args, **kwargs: read_validated_tracks(*args, **kwargs)
    ),
    "thread_worker": _thread_worker,
}


class NucleusCorrectionWidget(CorrectionViewStateMixin, QWidget):
    """Coordinator for the nucleus tracking-correction workflow.

    This class is intentionally a **hub, not a god object**: it owns the controls
    and orchestrates them, but the actual work lives in focused collaborators.
    Its size reflects a wide *coordination* surface (many controls, many refresh
    paths) rather than logic that belongs elsewhere — the heavy logic has been
    pushed out to the pieces below, and splitting the remaining glue further only
    relocates ``self.collaborator.…`` calls without reducing the coupling.

    Where the work actually lives
    -----------------------------
    * Drawing / contour editing — the embedded :class:`CorrectionWidget`
      (``self.correction_widget``): brush, redraw, new-shape painting, selection.
    * Display subsystems, each its own controller:
      ``self._validated_overlay`` (:class:`ValidatedOverlayController`),
      ``self._all_tracks`` (:class:`AllTracksController`, the track-path comet),
      ``self._lineage_canvas`` (:class:`LineageCanvasController`, the accordion +
      film strip) and ``self._candidate_gallery``
      (:class:`CandidateGalleryController`).
    * Stateless helpers in sibling modules: control assembly
      (:mod:`_correction_ui_nucleus`), label painting (:mod:`_correction_paint`),
      camera framing (:mod:`_correction_navigation`), playback queries
      (:mod:`_correction_playback`), held-key auto-repeat
      (:mod:`_correction_keymap`), label colormaps (:mod:`_correction_centroids`)
      and the shared view-state / owned-layer lifecycle
      (:class:`CorrectionViewStateMixin`, also used by the cell widget).
    * Tracking algorithms — :mod:`cellflow.tracking_ultrack` (extend, swap
      candidates, retrack, DB annotation, validation/commit). The widget only
      orchestrates these: read the layer, call the algorithm, paint the result,
      fan out the refreshes, set status.

    What stays here (the coordinator's own job)
    -------------------------------------------
    Building + wiring the controls, the correction-mode lifecycle
    (activate/deactivate, view-state capture/restore, owned-layer teardown, the
    focus-mode workspace dock), and translating user actions — mask edits, track
    selection, navigation, keyboard shortcuts — into calls on the collaborators
    above followed by the appropriate layer repaint + refresh fan-out.
    """

    def __init__(
        self,
        viewer,
        *,
        edit_callback=None,
        workspace_provider: Callable[[], NucleusWorkspace | None] | None = None,
        ultrack_config_provider: Callable[[], TrackingConfig] | None = None,
        dependencies: dict[str, Callable] | None = None,
        focus_takeover_callback: Callable[[bool], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        # Display collaborators subscribe to these; operations emit them instead
        # of hand-calling each controller's refresh (see _wire_events).
        self.events = CorrectionEvents()
        self._workspace_provider = workspace_provider
        self._ultrack_config_provider = ultrack_config_provider
        # Called with True once correction-mode activation succeeds and False on
        # deactivate, so the host can hide its other workflow sections for a
        # full-window focus mode (the widget deliberately holds no sibling refs).
        self._focus_takeover = focus_takeover_callback or (lambda _on: None)
        self._local_workspace: NucleusWorkspace | None = None
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
        # The single right-side workspace dock, its container (top bar + reveal
        # area + body splitter), and the horizontal body splitter it hosts
        # (toolbar · candidate gallery · accordion); all None while correction
        # mode is off. The top bar + reveal area are reparented out of the plugin
        # dock into the container on activate. ``_workspace_splitter`` doubles as
        # the "focus mode is docked" sentinel the refresh paths gate on.
        self._workspace_dock = None
        self._workspace_splitter: QSplitter | None = None
        self._workspace_container: QWidget | None = None
        # Collapsible wrappers around the candidate gallery and the accordion;
        # each carries a ✕ hide button and collapses to a slim show-tab. None
        # until the body splitter is built on activate.
        self._gallery_pane: CollapsiblePane | None = None
        self._accordion_pane: CollapsiblePane | None = None
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
            enabled_provider=self.track_path_btn.isChecked,
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
        self._wire_events()

    def _wire_events(self) -> None:
        """Subscribe the display collaborators to correction-domain events.

        A label edit fans out to the label colormap, the validated overlay, the
        focused track's comet, the lineage canvas and the candidate gallery —
        each self-gating through the providers it already holds. The emitter
        (:meth:`_on_cells_edited`) no longer needs to know any of these listeners
        exist; the subscriber map lives here, in one greppable place.

        (Prototype: only ``labels_edited`` is event-driven so far; the other
        refresh paths still fan out directly until they are migrated too.)
        """
        e = self.events
        e.labels_edited.connect(self._refresh_correction_label_visuals_for_edit)
        e.labels_edited.connect(self._apply_overlay_edit)
        e.labels_edited.connect(self._apply_track_path_edit)
        e.labels_edited.connect(lambda *_: self._refresh_lineage_canvas_if_shown())
        e.labels_edited.connect(lambda *_: self._refresh_candidate_gallery_if_shown())
        e.tracks_rebuilt.connect(self._apply_track_path_rebuilt)
        e.tracks_rebuilt.connect(self._refresh_lineage_canvas_if_shown)
        e.tracks_rebuilt.connect(self._refresh_candidate_gallery_if_shown)
        e.validation_changed.connect(self._refresh_validated_overlay)
        e.validation_changed.connect(self._refresh_validation_counter)
        e.validation_changed.connect(self._refresh_lineage_canvas_status_if_shown)
        e.swap_stepped.connect(self._apply_track_path_rebuilt)
        e.swap_stepped.connect(self._refresh_lineage_detail_if_shown)
        e.swap_stepped.connect(self._refresh_candidate_gallery_if_shown)
        e.stack_relabeled.connect(self._refresh_correction_label_visuals)
        e.stack_relabeled.connect(self._refresh_validated_overlay)
        e.stack_relabeled.connect(self._refresh_validation_counter)
        e.stack_relabeled.connect(self._refresh_lineage_canvas_if_shown)

    @property
    def _workspace(self) -> NucleusWorkspace | None:
        if self._workspace_provider is not None:
            return self._workspace_provider()
        return self._local_workspace

    @_workspace.setter
    def _workspace(self, value: NucleusWorkspace | None) -> None:
        self._local_workspace = value

    @property
    def _pos_dir(self):
        """The nucleus annotation/store directory (validation JSONs live here).

        Retained under this name because the validation API and child
        controllers thread it straight through; it now resolves from the active
        :class:`NucleusWorkspace` rather than a position directory.
        """
        ws = self._workspace
        return ws.nucleus_dir if ws is not None else None

    def _dependency(self, name: str):
        return self._dependencies[name]

    def _setup_ui(self) -> None:
        # All one-time control assembly + signal wiring lives in the builder
        # module so this widget stays focused on behaviour.
        build_nucleus_correction_ui(self)

    @staticmethod
    def _set_checked_without_signal(button, checked: bool) -> None:
        set_checked_without_signal(button, checked)

    def _sync_correction_panel_visibility(self) -> None:
        show_params = self.params_btn.isChecked()
        show_shortcuts = self.shortcuts_btn.isChecked()
        show_active = self._correction_active_content_visible

        self.extend_retrack_params_section.setVisible(show_params)
        self.shortcuts_section.setVisible(show_shortcuts)
        self.toolbar.setVisible(show_active)
        # The title is always shown: as a "Tracking Correction" stage pill next
        # to the on/off button in the inactive plugin dock, and as a full-size
        # workspace title once correction mode is active. The shortcuts / params
        # toggles and the view toggles still only appear when active.
        self.header_lbl.setVisible(True)
        self._apply_header_title_style(show_active)
        self.shortcuts_btn.setVisible(show_active)
        self.params_btn.setVisible(show_active)
        self.track_path_btn.setVisible(show_active)
        self.spotlight_btn.setVisible(show_active)
        self.filled_view_btn.setVisible(show_active)
        # The status line is only meaningful inside the active workspace; hide it
        # when inactive so it never wraps the plugin-dock title pill to two rows.
        if not show_active:
            self.status_lbl.setVisible(False)

        # Collapse the reveal area whenever both params and shortcuts are closed
        # (even while active): it only holds those two panels, so leaving it
        # expanded just pins dead vertical space and stops the body (gallery /
        # tracking overview) from growing back to full height.
        if show_params or show_shortcuts:
            self.section.expand()
        else:
            self.section.collapse()

    def _apply_header_title_style(self, active: bool) -> None:
        """Style the title as a workspace heading (active) or stage pill (idle)."""
        if active:
            self.header_lbl.setProperty("cellflow_stage_key", None)
            self.header_lbl.setStyleSheet("font-weight: bold; font-size: 12pt;")
        else:
            stage_header_label(self.header_lbl, "nucleus")

    def _on_correction_params_button_toggled(self, checked: bool) -> None:
        # 📖 and ⚙ are independent: both can be open at once — the reveal area
        # simply grows to fit. No mutual exclusion.
        self.extend_retrack_params_section._toggle.setChecked(checked)
        self._sync_correction_panel_visibility()

    def _on_correction_shortcuts_button_toggled(self, checked: bool) -> None:
        self.shortcuts_section._toggle.setChecked(checked)
        self._sync_correction_panel_visibility()

    def _tracked_path(self):
        ws = self._workspace
        return ws.tracked if ws is not None else None

    def _foreground_path(self):
        ws = self._workspace
        return ws.foreground if ws is not None else None

    def _ultrack_workdir(self):
        ws = self._workspace
        return ws.ultrack_workdir if ws is not None else None

    def _ultrack_db_path(self):
        ws = self._workspace
        return ws.ultrack_db if ws is not None else None

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
        ws = self._workspace
        return ws.cell_foreground if ws is not None else None

    def _nucleus_foreground_path(self):
        # The nucleus foreground *is* the workspace foreground; named for the
        # load site, where "nucleus" disambiguates it from the cell foreground.
        return self._foreground_path()


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


    def _add_correction_image_layer(self, data: np.ndarray, name: str, colormap: str) -> None:
        add_correction_image_layer(
            self.viewer,
            data,
            name=name,
            colormap=colormap,
            owned_layer_names=self._correction_owned_layers,
        )


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
            from cellflow.core.label_store import read_full_tracked_stack
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

        # The nucleus reads as "bop purple" on its own, but when the cell z-avg is
        # also shown (a grey channel beneath it) "I Purple" separates better.
        cell_path = self._cell_foreground_path()
        cell_present = cell_path is not None and cell_path.exists()
        nucleus_cmap = "I Purple" if cell_present else "bop purple"

        for data, name, cmap in (
            (
                cell_path,
                _CORRECTION_CELL_ZAVG_LAYER,
                "gray",
            ),
            (
                self._nucleus_foreground_path(),
                _CORRECTION_NUC_ZAVG_LAYER,
                nucleus_cmap,
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

    def _validated_track_ids(self) -> list[int]:
        """Validated track IDs, used to group reassign priority."""
        if self._pos_dir is None:
            return []
        read_validated_tracks_fn = self._dependency("read_validated_tracks")
        validated_tracks = read_validated_tracks_fn(self._pos_dir) or {}
        return sorted(int(cid) for cid in validated_tracks)

    def _on_reassign_ids(self) -> None:
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        stack = np.asarray(layer.data)
        validated = self._validated_track_ids()
        self._correction_status("Reassigning cell IDs...")

        @self._dependency("thread_worker")(connect={
            "returned": self._on_reassign_ids_done,
            "errored": self._on_correction_worker_error,
        })
        def _worker():
            order = track_order_by_frame_and_size(stack, validated)
            return reassign_ids_ordered(stack, order)

        _worker()

    def _on_reassign_ids_done(self, result: tuple) -> None:
        remapped, n_cells, old_to_new = result
        layer = self._correction_tracked_layer()
        if layer is not None:
            layer.data = remapped
        if self._pos_dir is not None and old_to_new:
            remap_validated_tracks(self._pos_dir, old_to_new)
        self.events.stack_relabeled.emit()
        self._correction_status(
            f"Reassigned {n_cells} cell IDs to range 1-{n_cells}. Unsaved."
        )

    def _commit_reassign_ids(self, layer) -> int:
        stack = np.asarray(layer.data)
        order = track_order_by_frame_and_size(stack, self._validated_track_ids())
        remapped, n_cells, old_to_new = reassign_ids_ordered(stack, order)
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
        add_corrections(
            self._pos_dir,
            corrections_for_label_frames(
                data,
                cell_id=cell_id,
                frames=frames,
            ),
        )
        self.events.validation_changed.emit()
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
            self.events.stack_relabeled.emit()
            self._correction_status(f"Unanchored cell {cell_id} at t={t}.")
            return
        layer = self._correction_tracked_layer()
        if layer is None:
            add_correction(
                self._pos_dir,
                anchor_correction(cell_id=cell_id, frame=t, y=y, x=x),
            )
            self.events.stack_relabeled.emit()
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
        self.events.stack_relabeled.emit()
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

        changed_ids = paint_assignments(
            frame, assignments, protected_mask,
            greedy=self.extend_greedy_overwrite_check.isChecked(),
        )
        layer.refresh()
        self._refresh_correction_label_visuals_for_edit(
            result.target_frame,
            changed_ids,
        )

        step = list(self.viewer.dims.current_step)
        step[0] = result.target_frame
        self.viewer.dims.current_step = tuple(step)

        self.events.tracks_rebuilt.emit()
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
        self.events.swap_stepped.emit()
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

    def _gallery_is_shown(self) -> bool:
        """True while the candidate gallery pane is built and not collapsed."""
        return self._gallery_pane is not None and not self._gallery_pane.is_collapsed()

    def _refresh_candidate_gallery_if_shown(self) -> None:
        # Debounced: (re)start the timer so a burst of frame/selection changes
        # collapses into a single rebuild once things settle (see the timer
        # set-up in _setup_ui). The fired handler re-checks visibility.
        if self._gallery_is_shown():
            self._candidate_refresh_timer.start()
        else:
            self._candidate_refresh_timer.stop()

    def _refresh_candidate_gallery_now(self) -> None:
        """Do the actual (expensive) gallery rebuild — only via the debounce timer."""
        if self._gallery_is_shown():
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
        self.events.swap_stepped.emit()
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
        self.events.tracks_rebuilt.emit()
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
        validated_tracks = (
            read_validated_tracks(self._pos_dir) if self._pos_dir is not None else {}
        )
        corrections = (
            read_corrections(self._pos_dir) if self._pos_dir is not None else []
        )
        protected_ids = protected_cell_ids_at_frame(
            validated_tracks, corrections, frame=target_frame
        )
        if int(assignment.cell_id) in protected_ids:
            return set()
        protected_mask = protected_cell_mask(frame, protected_ids)
        changed_ids = paint_assignments(
            frame, (assignment,), protected_mask, greedy=greedy
        )
        if not changed_ids:
            return set()
        layer.refresh()
        self._refresh_correction_label_visuals_for_edit(target_frame, changed_ids)
        return changed_ids

    def _on_retrack(self, direction: str) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._correction_status("Need >= 2 frames to retrack."); return
        t0 = int(self.viewer.dims.current_step[0])
        if direction == "forward" and t0 >= layer.data.shape[0] - 1:
            self._correction_status("Already at last frame."); return
        if direction == "backward" and t0 <= 0:
            self._correction_status("Already at first frame."); return

        before = np.asarray(layer.data).copy()
        validated_tracks = read_validated_tracks(self._pos_dir)
        result = retrack_stack_direction(
            before,
            start_frame=t0,
            direction=direction,
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
        self.events.tracks_rebuilt.emit()
        self._correction_status(
            f"Retracked {direction} from t={result.first_target_frame}: "
            f"{result.n_retracked} updated, "
            f"{result.n_skipped} validated skipped. Unsaved."
        )

    def _on_retrack_forward(self) -> None:
        self._on_retrack("forward")

    def _on_retrack_backward(self) -> None:
        self._on_retrack("backward")

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
        self._deselect_if_selection_gone()
        self.events.stack_relabeled.emit()
        self._correction_status(
            f"Removed unvalidated labels in {result.changed_frames} frame(s), "
            f"{result.changed_pixels} px changed. Unsaved."
        )

    def _deselect_if_selection_gone(self) -> None:
        """Clear the selection when the selected cell no longer exists at the
        current frame (a relabel / removal may have dropped it)."""
        sel = self.correction_widget._selected_label
        if sel:
            ct = self._current_t()
            if sel not in self._current_cell_ids(ct):
                self.correction_widget.select_label(ct, 0)

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
            self._deselect_if_selection_gone()
            self.events.stack_relabeled.emit()
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
        # ``(key, slot, repeat)``: ``repeat`` keys keep firing while held (the
        # navigation arrows, where holding to scrub is natural); the edit / save
        # keys fire once per press only.
        self._correction_key_specs = [
            ("A", lambda: self._on_extend(direction="backward"), False),
            ("D", lambda: self._on_extend(direction="forward"), False),
            ("Q", self._on_retrack_backward, False),
            ("E", self._on_retrack_forward, False),
            ("B", self._on_anchor_here, False),
            ("S", self._on_save_tracked, False),
            ("Z", lambda: self._on_swap_step(direction="smaller"), False),
            ("C", lambda: self._on_swap_step(direction="larger"), False),
            ("Space", self._toggle_movie_playback, False),
            # Arrow keys navigate the selected track's film strip as laid out:
            # Left/Right walk the thumbnails in reading order (columns), Up/Down
            # jump a whole wrapped row. Shift+Up/Down switch tracks instead,
            # walking the global track list and recentering like a track click.
            # All six auto-repeat while held (see HeldKeyRepeater).
            ("Left", lambda: self._step_film_frame(dx=-1), True),
            ("Right", lambda: self._step_film_frame(dx=1), True),
            ("Up", lambda: self._step_film_frame(dy=-1), True),
            ("Down", lambda: self._step_film_frame(dy=1), True),
            ("Shift-Up", lambda: self._step_track(-1), True),
            ("Shift-Down", lambda: self._step_track(1), True),
        ]
        self._bound_correction_keys: list[str] = []
        self._correction_keys_layer = None

        # Auto-repeat for held navigation keys, paced to the viewer's playback
        # fps (see HeldKeyRepeater); napari only auto-repeats bare arrows itself.
        self._key_repeater = HeldKeyRepeater(
            QTimer(self),
            interval_provider=lambda: nav_repeat_interval_ms(self.viewer),
        )

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
        for key, slot, repeat in self._correction_key_specs:
            if repeat:
                handler = self._key_repeater.key_handler(key, slot)
            else:
                def handler(_layer, _slot=slot):
                    _slot()

            layer.bind_key(key, handler, overwrite=True)
            self._bound_correction_keys.append(key)
        self._correction_keys_layer = layer

    def _unbind_correction_keys(self) -> None:
        self._key_repeater.stop()
        layer = self._correction_keys_layer
        for key in self._bound_correction_keys:
            try:
                if layer is not None:
                    layer.bind_key(key, None)
            except Exception:
                pass
        self._bound_correction_keys = []
        self._correction_keys_layer = None

    def _correction_data_available(self) -> bool:
        """True when a tracked nucleus stack exists on disk to correct."""
        tp = self._tracked_path()
        return tp is not None and tp.exists()

    def _on_correction_active_button_toggled(self, active: bool) -> None:
        if active:
            if not self._correction_data_available():
                self._correction_status("No tracked labels available to correct.")
                self._set_checked_without_signal(self.active_btn, False)
                self._sync_correction_panel_visibility()
                return
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
            # Start every session in the default outline view: neutral-coloured
            # outlines + tracks over the visible reference images.
            self._set_checked_without_signal(self.filled_view_btn, False)
            self._apply_label_view_mode(filled=False)
            # Sync the inner widget's selection indicator to the (possibly
            # persisted) spotlight toggle so they never drift apart.
            self.correction_widget.set_highlight_style(
                "spotlight" if self.spotlight_btn.isChecked() else "border"
            )
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
            # The accordion is the always-on main surface; the candidate gallery
            # is the action surface, both opened by default (each collapsible via
            # its own ✕). Render both explicitly so the body splitter is built once.
            self._lineage_canvas.refresh()
            self._candidate_gallery.refresh()
            # Build the whole-stack validated / anchor overlays once now; they're
            # only rebuilt on validation or label changes afterwards (not per frame).
            self._refresh_validated_overlay()
            self._refresh_validation_counter()
            # Hand the whole window to the workspace: reparent the top bar +
            # reveal area out of the plugin dock, hide the now-empty plugin dock,
            # and lay the body panels out as one splitter so they resize
            # independently.
            self._focus_takeover(True)
            self._arrange_workspace_docks()
            # Open the workspace at half the window width; the body splitter sizes
            # its toolbar · gallery · accordion strips. Deferred so the dock is
            # laid out before resizeDocks runs.
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
        # Stop any in-progress movie playback before the layers are torn down.
        self._stop_movie_playback()
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
        action = confirm_unsaved_before_deactivate(self, save_noun="tracked labels")
        if action == "cancel":
            return False
        if action == "save":
            self._on_save_tracked()
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
        # Flag-only change, so a light status recolour (was a full lineage
        # rebuild here — inconsistent with _on_validate_track, which the
        # event normalises to the lighter, freeze-free path).
        self.events.validation_changed.emit()

    def _toggle_movie_playback(self) -> None:
        """Space: start/stop animating the frame (time) slider like a movie."""
        try:
            qt_dims = self.viewer.window._qt_viewer.dims
        except AttributeError:
            return
        if qt_dims.is_playing:
            qt_dims.stop()
        else:
            # Time is axis 0 (see _current_t); fps=None uses napari's setting.
            qt_dims.play(axis=0)

    def _stop_movie_playback(self) -> None:
        try:
            qt_dims = self.viewer.window._qt_viewer.dims
        except AttributeError:
            return
        if qt_dims.is_playing:
            qt_dims.stop()

    def on_dims_step_changed(self) -> None:
        self._swap_cursor = None
        # The validated / anchor overlays are whole-stack masks (built on
        # activation and on each validation/label change), so napari already
        # shows the right slice — no per-frame rebuild here.
        # Keep the focused track's tip cross on the cell in the new frame.
        self._all_tracks.set_current_frame(self._current_t())
        if self._workspace_splitter is not None:
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

    def _on_toggle_spotlight(self, checked: bool) -> None:
        """Spotlight on → dim outside the selection; off → plain yellow border."""
        self.correction_widget.set_highlight_style(
            "spotlight" if checked else "border"
        )

    def _on_track_selection_changed(self, _t: int, _lab: int) -> None:
        """Recolour the all-tracks layer / accordion when selection changes."""
        self._apply_focus_presentation(_lab)
        if self._workspace_splitter is not None:
            # Selection drives the accordion: the picked track expands inline.
            self._lineage_canvas.set_selection(_lab)
            # Recenter on the selected track's film strip (the thumbnail band,
            # not the bar row) when the selection came from the image viewer; an
            # accordion click already shows the row.
            if not self._navigating_from_lineage:
                self._lineage_canvas.center_on_strip(_lab)
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

    def _apply_track_path_rebuilt(self) -> None:
        """``tracks_rebuilt`` listener: redraw the comet + spotlight when on.

        The all-tracks overlay's geometry follows the selected track, so an
        extend/retrack that reshaped it must repaint the comet (and re-run the
        spotlight so its mask provider is consulted) — but only while the
        track-path overlay is toggled on.
        """
        if self.track_path_btn.isChecked():
            self._refresh_track_path_overlay()
            self._refresh_track_path_spotlight()

    def _refresh_lineage_detail_if_shown(self) -> None:
        """``swap_stepped`` listener: refresh only the selected track's detail
        strip — not the whole-stack lineage build, which froze the GUI when fired
        on every swap keystroke (the overview catches up on the next full
        refresh)."""
        if self._workspace_splitter is not None:
            self._lineage_canvas.refresh_detail()

    def _refresh_lineage_canvas_if_shown(self) -> None:
        """Rebuild the accordion (bars + expanded band) after a label change.

        The accordion is the always-on main surface for the whole time focus
        mode is active, so the rebuild is gated on the workspace being docked;
        otherwise the bars go stale after a label change (reassign IDs, remove
        unvalidated, validate, …).
        """
        if self._workspace_splitter is not None:
            self._lineage_canvas.refresh()

    def _refresh_lineage_canvas_status_if_shown(self) -> None:
        """Lightweight lineage refresh for flag-only changes (validate/anchor).

        Validation changes per-frame status, not topology, so this recolours the
        cached lanes instead of re-running the whole-stack lineage build that
        :meth:`_refresh_lineage_canvas_if_shown` does (which froze the GUI for
        seconds when validating a long track).
        """
        if self._workspace_splitter is not None:
            self._lineage_canvas.refresh_status()

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

    def _correction_intensity_frame(self, t: int):
        """2-D nucleus frame at time *t* used to snap spawned cells to signal."""
        if _CORRECTION_NUC_ZAVG_LAYER not in self.viewer.layers:
            return None
        layer = self.viewer.layers[_CORRECTION_NUC_ZAVG_LAYER]
        return frame_view_2d(np.asarray(layer.data), int(t))

    def _film_strip_intensity_layer(self):
        """Best raw layer to crop tiles from (nucleus, then cell)."""
        for name in (
            _CORRECTION_NUC_ZAVG_LAYER,
            _CORRECTION_CELL_ZAVG_LAYER,
        ):
            if name in self.viewer.layers:
                return self.viewer.layers[name]
        return None

    def _on_workspace_panel_toggled(self, pane, collapsed: bool) -> None:
        """A pane's ✕ / show-tab flipped: keep ≥1 open and resize the splitter.

        Hiding one of the gallery / accordion panes while the other is already
        collapsed re-opens that other one, so the workspace never goes blank.
        Re-showing the gallery also repopulates it for the current selection.
        """
        gallery = self._gallery_pane
        accordion = self._accordion_pane
        if (
            collapsed
            and gallery is not None
            and accordion is not None
            and gallery.is_collapsed()
            and accordion.is_collapsed()
        ):
            other = accordion if pane is gallery else gallery
            other.set_collapsed(False)  # re-enters here with one pane open
            return
        self._apply_workspace_panel_sizes()
        if not collapsed and self._gallery_is_shown():
            self._refresh_candidate_gallery_if_shown()

    # ── label / track view mode (outline-neutral ↔ filled-by-id) ──────────────

    def _filled_view_active(self) -> bool:
        return self.filled_view_btn.isChecked()

    def _on_toggle_filled_view(self, checked: bool) -> None:
        self._apply_label_view_mode(filled=checked)

    def _apply_label_view_mode(self, *, filled: bool) -> None:
        """Switch between the outline (neutral) and filled (by-id) viewer modes.

        Outline mode shows the cell + nucleus reference images and draws the
        labels as outlines and the tracks overview in one neutral colour. Filled
        mode hides those images and draws the labels filled + opaque and the
        tracks overview coloured by ID.
        """
        for name in (_CORRECTION_CELL_ZAVG_LAYER, _CORRECTION_NUC_ZAVG_LAYER):
            if name in self.viewer.layers:
                self.viewer.layers[name].visible = not filled
        layer = self._correction_tracked_layer()
        if layer is not None:
            try:
                layer.contour = 0 if filled else 2
                layer.opacity = (
                    _FILLED_LABEL_OPACITY if filled else _OUTLINE_LABEL_OPACITY
                )
            except Exception:
                pass
            if filled:
                refresh_label_colormap(
                    layer,
                    np.asarray(layer.data),
                    color_scale=_NUCLEUS_TRACK_COLOR_SCALE,
                )
            else:
                apply_neutral_label_colormap(layer)
        # In filled-by-ID mode the green/anchor wash would hide the per-ID
        # colours, so draw validated/anchor cells as a coloured border instead.
        self._validated_overlay.set_border_mode(filled)
        self._all_tracks.set_filled_mode(filled)

    def _navigate_to_error(self, t: int, cell_id: int) -> None:
        """Lineage-canvas click handler: jump to frame ``t``, select and center.

        The clicked row is already in view, so the canvas recenter is suppressed
        (``from_lineage=True``) while the image viewer is still panned onto the
        cell.
        """
        self._navigate_to_cell(int(t), int(cell_id), from_lineage=True)

    def _navigate_to_cell(self, t: int, cell_id: int, *, from_lineage: bool) -> None:
        """Jump to frame ``t``, select ``cell_id`` and center both views on it.

        Besides stepping to the frame and selecting the cell, this pans the
        image-viewer camera so the cell sits in the middle of the canvas. With
        ``from_lineage=False`` (keyboard track stepping) the resulting selection
        callback also recenters the lineage canvas on the track; a lineage click
        passes ``from_lineage=True`` because the clicked row is already shown.
        """
        try:
            step = list(self.viewer.dims.current_step)
            if step:
                step[0] = int(t)
                self.viewer.dims.current_step = tuple(step)
        except Exception:
            logger.exception("focus-mode navigation: frame jump failed")
        # When the track is absent at the target frame (e.g. an empty placeholder
        # thumbnail in an incomplete track's film strip), still step to the frame
        # but keep the current selection: re-selecting would find no mask there
        # and clear the track. The frame jump above already moved the canvas
        # guide via on_dims_step_changed.
        if int(cell_id) and int(cell_id) not in self._current_cell_ids(int(t)):
            return
        self._navigating_from_lineage = from_lineage
        try:
            self.correction_widget.select_label(int(t), int(cell_id))
        except Exception:
            logger.exception("focus-mode navigation: cell select failed")
        finally:
            self._navigating_from_lineage = False
        self._center_viewer_on_cell(int(t), int(cell_id))

    def _step_film_frame(self, *, dx: int = 0, dy: int = 0) -> None:
        """Step the current frame across the selected track's film-strip grid.

        Bound to the bare arrow keys in focus mode: Left/Right move one thumbnail
        in reading order, Up/Down jump a wrapped row. Running off the end wraps
        back when the viewer's playback loop mode is on. Delegates to the lineage
        canvas, which owns the band geometry and the frame-jump path.
        """
        if self._workspace_splitter is None:
            return
        self._lineage_canvas.step_film_frame(
            dx=dx, dy=dy, wrap=playback_loops(self.viewer)
        )

    def _step_track(self, direction: int) -> None:
        """Select the next/previous track in the global list and recenter on it.

        Bound to Shift+Up / Shift+Down in focus mode. Unlike a frame-local
        scan, this walks the sorted list of every track ID in the stack and
        navigates to the chosen track exactly as a lineage-canvas click would —
        landing on a frame where it exists, selecting it, and recentering both
        the lineage canvas and the image viewer.
        """
        layer = self._correction_tracked_layer()
        if layer is None:
            return
        data = np.asarray(layer.data)
        all_ids = sorted(set(int(v) for v in np.unique(data)) - {0})
        if not all_ids:
            self._correction_status("No cells in any frame.")
            return
        cur = int(getattr(self.correction_widget, "_selected_label", 0) or 0)
        if cur in all_ids:
            nxt = all_ids[(all_ids.index(cur) + direction) % len(all_ids)]
        else:
            nxt = all_ids[0] if direction > 0 else all_ids[-1]
        # Stay on the current frame if the track is present there; otherwise jump
        # to the first frame that contains it so it can be selected and centered.
        t = self._current_t()
        if nxt not in self._current_cell_ids(t):
            frames = [i for i in range(data.shape[0]) if np.any(data[i] == nxt)]
            if frames:
                t = frames[0]
        self._navigate_to_cell(t, nxt, from_lineage=False)

    def _center_viewer_on_cell(self, t: int, cell_id: int) -> None:
        """Frame the selected track in the viewer (camera math lives in
        :func:`cellflow.napari._correction_navigation.center_viewer_on_cell`)."""
        center_viewer_on_cell(
            self.viewer, self._correction_tracked_layer(), t, cell_id
        )

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
        """Open the workspace at half the window width; size the body splitter.

        The dock opens at half the window; ``_apply_workspace_panel_sizes`` then
        distributes the toolbar · gallery · accordion strips (honouring any
        already-collapsed pane).
        """
        win = self._main_window()
        dock = self._workspace_dock
        splitter = self._workspace_splitter
        if win is None or dock is None or splitter is None:
            return
        try:
            target = max(int(win.width()) // 2, 1)
            win.resizeDocks([dock], [target], Qt.Horizontal)
            self._apply_workspace_panel_sizes()
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

    def _teardown_workspace_controls_dock(self) -> None:
        """Remove the workspace dock (top bar / reveal already reparented out).

        The toolbar is rescued back onto ``self`` first — it is built once and
        must survive the dock teardown to be re-used on a later activate.
        Destroying the dock then deletes the container, the body splitter and the
        embedded panels (candidate gallery, accordion); the controllers drop
        their stale references in their own ``teardown``.
        """
        self.toolbar.setParent(self)
        self.toolbar.setVisible(False)
        if self._workspace_dock is not None:
            try:
                self.viewer.window.remove_dock_widget(self._workspace_dock)
            except Exception:
                logger.exception("could not remove the correction workspace dock")
        self._workspace_dock = None
        self._workspace_splitter = None
        self._workspace_container = None
        self._gallery_pane = None
        self._accordion_pane = None

    def _arrange_workspace_docks(self) -> None:
        """Build the workspace dock once.

        The dock holds one container: the full-width top bar and reveal area
        stacked over a body row — a fixed-width toolbar column beside a
        horizontal splitter (candidate gallery · accordion). The gallery and
        accordion are each wrapped in a ``CollapsiblePane`` so they hide/show via
        their own ✕ / show-tab; the accordion is the main surface but can be
        collapsed just the same.
        """
        self._ensure_workspace_splitter()

    def _ensure_workspace_splitter(self) -> QSplitter | None:
        """Create the body splitter, the container + dock once (or return None).

        Reparents the top bar (``self.header``) and reveal area (``self.section``)
        out of the plugin dock into the container, with the body splitter beneath
        them. The reparent reuses every existing button + signal as-is.
        """
        if self._workspace_splitter is not None:
            return self._workspace_splitter

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
        # candidate gallery · accordion, each wrapped in a collapsible pane; the
        # accordion pane is the stretch panel that absorbs outer-width changes.
        self._gallery_pane = CollapsiblePane(
            self._candidate_gallery.widget(), title="Candidate gallery"
        )
        self._accordion_pane = CollapsiblePane(
            self._lineage_canvas.panel(), title="Tracking overview"
        )
        self._gallery_pane.collapsed_changed.connect(
            lambda collapsed: self._on_workspace_panel_toggled(
                self._gallery_pane, collapsed
            )
        )
        self._accordion_pane.collapsed_changed.connect(
            lambda collapsed: self._on_workspace_panel_toggled(
                self._accordion_pane, collapsed
            )
        )
        splitter.addWidget(self._gallery_pane)
        splitter.addWidget(self._accordion_pane)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # The toolbar is a fixed-width icon column, not a resizable pane: pin it
        # beside the splitter in a plain row (no drag handle) so it always hugs
        # its icons and can't be stretched.
        self.toolbar.setVisible(True)
        self.toolbar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        body = QWidget()
        body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(4)
        body_lay.addWidget(self.toolbar)
        body_lay.addWidget(splitter, stretch=1)

        container = QWidget()
        # The body row carries the stretch so the splitter (gallery · accordion)
        # grows to the full dock height; an explicit Expanding policy keeps the
        # container itself from shrinking to its hint in packaged builds.
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.header)
        lay.addWidget(self.section)
        lay.addWidget(body, stretch=1)
        self._workspace_container = container

        try:
            self._workspace_dock = self.viewer.window.add_dock_widget(
                container, name="Tracking Correction", area="right"
            )
        except Exception:
            logger.exception("could not dock the correction workspace")
            self._workspace_dock = None
            self._workspace_container = None
            self._gallery_pane = None
            self._accordion_pane = None
            return None
        self._workspace_splitter = splitter
        return splitter

    def _apply_workspace_panel_sizes(self) -> None:
        """Size the body splitter from the panes' collapsed state.

        The toolbar is a fixed column outside the splitter; here only the two
        panes are sized: a collapsed pane shrinks to its slim show-tab, and the
        expanded pane(s) share the remainder (the accordion favoured when both
        are open).
        """
        splitter = self._workspace_splitter
        if splitter is None or self._gallery_pane is None or self._accordion_pane is None:
            return
        rest = max(splitter.size().width(), 300)
        strip = _PANEL_STRIP_W
        gallery_collapsed = self._gallery_pane.is_collapsed()
        accordion_collapsed = self._accordion_pane.is_collapsed()
        if gallery_collapsed:
            gallery_w, accordion_w = strip, rest - strip
        elif accordion_collapsed:
            gallery_w, accordion_w = rest - strip, strip
        else:
            gallery_w = max(rest // 3, 1)
            accordion_w = rest - gallery_w
        splitter.setSizes([gallery_w, accordion_w])

    def _teardown_focus_panels(self) -> None:
        """Undock the focus-mode panels (accordion, candidate gallery)."""
        self._lineage_canvas.teardown()
        self._candidate_gallery.teardown()

    def _refresh_validation_counter(self) -> None:
        self._validated_overlay.refresh_counter(self.validation_counter_lbl)

    def _on_cells_edited(self, t: int, changed_ids: set[int]) -> None:
        """A hand mask edit (draw / merge / relabel / redraw / fill / split)
        landed — announce it; the wired listeners repaint themselves."""
        self.events.labels_edited.emit(t, changed_ids)

    def _apply_overlay_edit(self, t: int, changed_ids: set[int]) -> None:
        """``labels_edited`` listener: refresh the validated overlay + counter."""
        self._validated_overlay.on_cells_edited(
            t,
            changed_ids,
            frame_view_2d=frame_view_2d,
            counter_label=self.validation_counter_lbl,
        )

    def _apply_track_path_edit(self, t: int, changed_ids: set[int]) -> None:
        """``labels_edited`` listener: rebuild the comet only when the *focused*
        track's pixels changed and the track-path overlay is on."""
        sel = int(getattr(self.correction_widget, "_selected_label", 0) or 0)
        if sel and sel in {int(v) for v in changed_ids} and self.track_path_btn.isChecked():
            self._refresh_track_path_overlay()
            self._refresh_track_path_spotlight()

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
        # In the neutral (outline) view the single-colour colormap already covers
        # any new ids via its ``None`` default, so only extend the per-id map in
        # the filled (by-id) view.
        if self._filled_view_active():
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
        if self._filled_view_active():
            refresh_label_colormap(
                layer,
                data,
                color_scale=_NUCLEUS_TRACK_COLOR_SCALE,
            )
        else:
            apply_neutral_label_colormap(layer)
        # The all-tracks overlay's geometry follows the labels — rebuild it so a
        # reassign / remove-unvalidated / retrack reshapes the trajectories too.
        self._all_tracks.refresh()
        try:
            self.viewer.layers.selection.active = layer
        except Exception:
            pass
