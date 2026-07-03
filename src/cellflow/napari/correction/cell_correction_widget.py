"""Correction section widget for the cell workflow."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile
from napari.utils.colormaps import direct_colormap
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.correction.labels import (
    clean_stranded_pixels,
    fill_label_holes,
)
from cellflow.core.tiff import imwrite_grayscale
from cellflow.core.label_store import read_full_tracked_stack
from cellflow.napari.correction._correction_utils import frame_view_2d, remove_unvalidated_labels
from cellflow.napari.lineage_canvas_controller import LineageCanvasController
from cellflow.napari._widget_helpers import (
    make_status as _make_status,
    tool_btn as _tool_btn,
)
from cellflow.napari.correction._correction_layer_lifecycle import (
    CorrectionViewStateMixin,
    LayerViewState,
    hide_all_layers,
)
from cellflow.napari.correction._correction_ui import (
    build_correction_toolbar,
    confirm_unsaved_before_deactivate,
    set_checked_without_signal,
)
from cellflow.napari.correction.correction_widget import CorrectionWidget
from cellflow.napari.ui_style import (
    add_block_pair_row,
    block_grid,
    stage_header_action_button,
    stage_header_label,
)
from cellflow.napari.widgets import CollapsibleSection
from cellflow.napari._widget_helpers import (
    dslider as _dslider,
    islider as _islider,
)

logger = logging.getLogger(__name__)

# Layers created by the cell correction widget itself carry a [Correction]
# prefix so they can be cleanly removed on deactivate without clobbering the
# regular pipeline layers (Tracked: Cell, Cell z-avg, …) shown elsewhere.
_TRACKED_CELL_LAYER = "[Correction] Cell Labels"
_CELL_ZAVG_LAYER = "[Correction] Cell z-avg"
_NUC_ZAVG_LAYER = "[Correction] Nucleus z-avg"
# Validated-cell overlay (full-editing only): an opaque green border around
# every cell in every frame it's validated for. Standalone analogue of
# ValidatedOverlayController's border mode, drawn from the in-memory
# ``self._validated_tracks`` store instead of validated_cells.json.
_VALIDATED_OVERLAY_LAYER = "[Correction] Validated: Cell"
_VALIDATED_OVERLAY_COLOR = "#00ff00"
_VALIDATED_OVERLAY_CONTOUR = 2

# Pipeline-side (non-correction) layer names — used as fallbacks where the
# user may have loaded layers manually, and as the target for the on-disk
# Tracked: Cell refresh when correction mode is exited.
_PIPELINE_TRACKED_CELL_LAYER = "Tracked: Cell"


class CellCorrectionWidget(CorrectionViewStateMixin, QWidget):
    """Qt controls for cell tracking correction workflows.

    Owns the correction group-box UI, all correction-scoped button handlers,
    and the inline CorrectionWidget.  Does not hold a back-reference to the
    parent workflow widget.

    Parameters
    ----------
    viewer:
        The napari viewer instance.
    pos_dir_provider:
        Zero-argument callable that returns the current position directory
        (or None).  When None, the widget falls back to a locally stored
        ``_local_pos_dir``.
    files_widget_refresh_callback:
        Called with the current ``pos_dir`` after save operations so that the
        parent workflow can refresh its file status panel.
    labels_path_provider:
        Optional zero-argument callable returning the tracked-cell-labels file to
        load and save.  When given it overrides the default position-dir path
        (``<pos_dir>/3_cell/tracked_labels.tif``), letting a standalone tool point
        the corrector at a flat output file.  Defaults to None → unchanged app
        behaviour.
    cell_ref_path_provider, nucleus_ref_path_provider:
        Optional overrides for the cell / nucleus reference-image backdrops
        (default ``<pos_dir>/1_cellpose/*_foreground.tif``).  None → app default.
    active_labels_layer_provider:
        Optional zero-argument callable returning the napari **Labels** layer to
        correct right now (or None when the active layer is not a Labels layer).
        When given, the corrector runs in *active-layer mode*: it edits that
        layer's data in place — no on-disk label file, no backdrop loading, no
        layer hiding — and the user saves it via napari. Defaults to None →
        unchanged disk-based app behaviour.
    intensity_frame_provider:
        Optional callable ``(t) -> 2D array | None`` giving the signal frame that
        middle-click spawn snaps a new cell to (see ``add_cell``'s ``image``
        argument); ``None`` (or a returned ``None``) falls back to a plain disk.
        Defaults to None → spawn always stamps a disk (unchanged app behaviour).
    parent:
        Optional Qt parent.
    """

    def __init__(
        self,
        viewer,
        *,
        pos_dir_provider: Callable[[], Path | None] | None = None,
        files_widget_refresh_callback: Callable[[Path | None], None] | None = None,
        labels_path_provider: Callable[[], Path | None] | None = None,
        cell_ref_path_provider: Callable[[], Path | None] | None = None,
        nucleus_ref_path_provider: Callable[[], Path | None] | None = None,
        active_labels_layer_provider: Callable[[], object] | None = None,
        intensity_frame_provider: Callable[[int], np.ndarray | None] | None = None,
        full_editing: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        # Opt-in (standalone segment+track only): unlock the full DB-free editing
        # toolkit — spawn / erase / merge / swap / split (mouse + Delete), plus a
        # greedy retracker on Q/E. The integrated app never passes this, so its
        # contour-only cell corrector is byte-for-byte unchanged.
        self._full_editing = full_editing
        self._pos_dir_provider = pos_dir_provider
        self._local_pos_dir: Path | None = None
        self._files_widget_refresh = files_widget_refresh_callback or (lambda _pd: None)
        self._labels_path_provider = labels_path_provider
        self._cell_ref_path_provider = cell_ref_path_provider
        self._nucleus_ref_path_provider = nucleus_ref_path_provider
        self._active_labels_layer_provider = active_labels_layer_provider
        self._intensity_frame_provider = intensity_frame_provider
        self._active_bound_layer = None
        self._retrack_keys_layer = None
        self._correction_view_state: LayerViewState | None = None
        self._correction_owned_layers: set[str] = set()
        self._correction_dirty: bool = False
        # DB-free validation store (full-editing/standalone only): there is no
        # project directory to persist validated_cells.json into, so validation
        # lives only for the session, keyed the same as the disk-backed store
        # ({cell_id: {validated frames}}) so remove_unvalidated_labels and the
        # lineage canvas need no adaptation.
        self._validated_tracks: dict[int, set[int]] = {}
        self._lineage_canvas: LineageCanvasController | None = None
        self._setup_ui()
        self._connect_signals()
        # Active-layer mode has no on-disk target — saving is napari's job, so
        # the in-widget disk-save button is hidden.
        if self._active_labels_layer_provider is not None:
            self.save_labels_btn.setVisible(False)
        self._sync_active_btn_enabled()

    # ------------------------------------------------------------------ #
    # pos_dir property — delegate to provider if available                #
    # ------------------------------------------------------------------ #

    @property
    def _pos_dir(self) -> Path | None:
        if self._pos_dir_provider is not None:
            return self._pos_dir_provider()
        return self._local_pos_dir

    @_pos_dir.setter
    def _pos_dir(self, value: Path | None) -> None:
        self._local_pos_dir = value

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        inner = QWidget(self)
        group_lay = QVBoxLayout(inner)
        group_lay.setContentsMargins(0, 0, 0, 0)
        group_lay.setSpacing(6)

        self.active_btn = _tool_btn(
            "⏻",
            "Activate correction mode and show correction controls.",
            checkable=True,
        )
        self.active_btn.setToolTip(
            "Activate correction mode and show correction controls."
        )
        self.params_btn = _tool_btn(
            "⚙", "Show correction parameters.", checkable=True
        )
        self.shortcuts_btn = _tool_btn(
            "📖", "Show correction shortcuts.", checkable=True
        )

        # ── Action toolbar buttons (icon tool-buttons in the body) ────
        # Mirror the nucleus correction widget: the main actions read as a thin
        # vertical icon toolbar beside the editing surface (plain enlarged
        # glyphs, no stage pill) rather than a stack of text buttons.
        # ``outline_btn`` is the visible driver for the embedded correction
        # widget's (now hidden) "Show outlines only" checkbox; the rest are
        # momentary actions.
        self.outline_btn = _tool_btn(
            "◻",
            "Show outlines only — draw labels as contours so the reference "
            "images stay visible underneath.",
            checkable=True,
        )
        self.save_labels_btn = _tool_btn(
            "💾", "Save tracked cell labels to disk."
        )
        self.fill_holes_btn = _tool_btn(
            "🪣",
            "Fill background holes fully enclosed within individual labels.",
        )
        self.cleanup_btn = _tool_btn(
            "🧹",
            "Remove disconnected same-label fragments (honours the Scope selector).",
        )
        # Retracker (standalone full-editing only): re-link every later frame to
        # the current one by greedy geometric similarity (area + centroid IoU -
        # distance). DB-free — there is no validation/locking, just re-linking.
        if self._full_editing:
            self.retrack_back_btn = _tool_btn(
                "↶", "Retrack labels backward from the current frame (Q)."
            )
            self.retrack_fwd_btn = _tool_btn(
                "↷", "Retrack labels forward from the current frame (E)."
            )
            # Validation is DB-free here too: it just flags cell/frame pairs in
            # memory for this session (no on-disk project to persist into).
            self.validate_btn = _tool_btn(
                "✓",
                "Toggle validation for the selected cell across its frames (V).",
            )
            self.remove_unvalidated_btn = _tool_btn(
                "🗑",
                "Remove cell label pixels not marked validated for their frame.",
            )

        # ── Scope selector (always visible in the active panel) ───────
        # Scope drives Fill Holes / Remove Stranded Fragments, so it lives
        # inline rather than behind the ⚙ params reveal.
        self.scope_row = QWidget(self)
        scope_row = QHBoxLayout(self.scope_row)
        scope_row.setContentsMargins(0, 0, 0, 0)
        scope_lbl = QLabel("Scope:")
        scope_lbl.setToolTip(
            "Applies to Fill Holes and Remove Stranded Fragments."
        )
        scope_row.addWidget(scope_lbl)
        self.correction_scope_combo = QComboBox()
        self.correction_scope_combo.addItems(["Current frame", "All frames"])
        self.correction_scope_combo.setToolTip(
            "Applies to Fill Holes and Remove Stranded Fragments."
        )
        scope_row.addWidget(self.correction_scope_combo)
        scope_row.addStretch(1)

        # ── Correction parameters (collapsible) ───────────────────────
        params_inner = QWidget(self)
        params_lay = QVBoxLayout(params_inner)
        params_lay.setContentsMargins(0, 0, 0, 0)
        params_lay.setSpacing(6)

        g = block_grid(horizontal_spacing=12)
        self.hole_radius_spin = _islider(
            0, 999, 5,
            tooltip="Max hole size (pixels) for the Fill Holes operation.",
        )
        add_block_pair_row(g, 0, "Hole radius:", self.hole_radius_spin)
        if self._full_editing:
            # Cell radius backs middle-click spawning; max dist gates retrack
            # matches. The spawn spinbox lives inside the inner widget (created
            # below) and is relocated here once it exists.
            self.retrack_max_dist_spin = _dslider(
                0, 500, 50.0, 1.0, 1,
                tooltip="Max centroid distance (px) for a retrack (Q/E) match.",
            )
            add_block_pair_row(
                g, 1, "Retrack max dist:", self.retrack_max_dist_spin
            )
        params_lay.addLayout(g)

        self.correction_params_section = CollapsibleSection(
            "Correction Parameters",
            params_inner,
            expanded=False,

        )
        self.correction_params_section.set_header_visible(False)
        self.correction_params_section.setVisible(False)
        group_lay.addWidget(self.correction_params_section)

        # ── Inline CorrectionWidget ───────────────────────────────────
        # App default: cell labels are tied to the nuclei, so restrict to contour
        # edits only (select + Shift-left extend + Shift-right carve), border
        # highlight, no spawn-radius control.
        # ``full_editing`` (standalone segment+track only) drops ``contour_only``
        # so the whole DB-free toolkit is live — spawn / erase / merge / swap /
        # split (mouse + Delete) — and the shortcuts panel auto-lists them.
        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
            show_cleanup=False,
            show_spawn_controls=False,
            contour_only=not self._full_editing,
            highlight_style="border",
        )
        # Inline contour edits (extend / carve / split) flow through the inner
        # widget's own status label, so they never hit ``_correction_status`` and
        # its "Unsaved" dirty-tracking. Mark the panel dirty on every edit so the
        # exit confirm fires, just like the nucleus correction widget.
        self.correction_widget.set_edit_callback(self._on_correction_edit)
        if self._intensity_frame_provider is not None:
            self.correction_widget.set_intensity_frame_callback(
                self._intensity_frame_provider
            )
        # The outline view is driven from the toolbar ``outline_btn`` toggle now,
        # so hide the embedded checkbox and keep the two in sync.
        self.correction_widget._outline_btn.setVisible(False)
        if self._full_editing:
            # Relocate the inner widget's spawn-radius spinbox into our params
            # panel (it is created but unparented to a layout when
            # show_spawn_controls=False), so middle-click spawn size is tunable.
            cell_radius_grid = block_grid(horizontal_spacing=12)
            add_block_pair_row(
                cell_radius_grid, 0, "Cell radius:",
                self.correction_widget._cell_radius_spin,
            )
            params_lay.addLayout(cell_radius_grid)

        # Thin vertical icon toolbar (nucleus styling: enlarged glyphs, ruled
        # groups, no pill) shown to the left of the editing surface.
        toolbar_groups = [
            (self.outline_btn,),
            (self.save_labels_btn,),
            (self.fill_holes_btn, self.cleanup_btn),
        ]
        if self._full_editing:
            toolbar_groups.append((self.retrack_back_btn, self.retrack_fwd_btn))
            toolbar_groups.append((self.validate_btn, self.remove_unvalidated_btn))
        self.toolbar = build_correction_toolbar(self, toolbar_groups)
        self.toolbar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        self.correction_shortcuts_section = CollapsibleSection(
            "Correction Shortcuts",
            self.correction_widget.build_shortcuts_widget(),
            expanded=False,

        )
        self.correction_shortcuts_section.set_header_visible(False)
        self.correction_shortcuts_section.setVisible(False)
        group_lay.addWidget(self.correction_shortcuts_section)

        self.active_content = QWidget(self)
        active_lay = QVBoxLayout(self.active_content)
        active_lay.setContentsMargins(0, 0, 0, 0)
        active_lay.setSpacing(6)

        body_row = QHBoxLayout()
        body_row.setContentsMargins(0, 0, 0, 0)
        body_row.setSpacing(6)
        body_row.addWidget(self.toolbar)
        body_row.addWidget(self.correction_widget, stretch=1)
        active_lay.addLayout(body_row)

        self.correction_status_lbl = _make_status()
        active_lay.addWidget(self.correction_status_lbl)

        active_lay.addWidget(self.scope_row)
        active_lay.addWidget(self.correction_widget._attrib_lbl)

        # ── Track-list navigator (full-editing only) ──────────────────
        # Reuses the nucleus workflow's swimlane accordion: it needs a plain
        # (T, Y, X)-ish label stack, not an Ultrack DB, so it is directly
        # portable. There is no intensity backdrop to crop thumbnails from in
        # this standalone tool (the active layer is whatever the user bound),
        # so the film-strip band stays empty; the per-track presence/validated
        # bars and click-to-jump navigation still work fully.
        if self._full_editing:
            self._lineage_canvas = LineageCanvasController(
                self.viewer,
                tracked_data_provider=self._lineage_tracked_data,
                tracked_layer_provider=self._correction_tracked_layer,
                intensity_layer_provider=lambda: None,
                selected_label_provider=lambda: int(
                    getattr(self.correction_widget, "_selected_label", 0) or 0
                ),
                current_t_provider=self._current_t,
                on_activate=self._navigate_to_cell_from_lineage,
                validated_tracks_provider=lambda: self._validated_tracks,
            )
            panel = self._lineage_canvas.panel()
            panel.setMinimumHeight(140)
            active_lay.addWidget(panel, stretch=1)

        self.active_content.setVisible(False)
        group_lay.addWidget(self.active_content)

        self.header = self._build_correction_header()

        self.section = CollapsibleSection(
            "Correction",
            inner,
            expanded=False,

        )
        self.section.set_header_visible(False)
        self.section._toggle.setEnabled(False)
        root.addWidget(self.header)
        root.addWidget(self.section)

        self.header_lbl = self.correction_header_lbl
        self.correction_active_btn = self.active_btn
        self.correction_params_btn = self.params_btn
        self.correction_shortcuts_btn = self.shortcuts_btn
        self.correction_mode_section = self.section

    def _build_correction_header(self) -> QWidget:
        header = QWidget(self)
        row = QHBoxLayout(header)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self.correction_header_lbl = QLabel("Correction")
        stage_header_label(self.correction_header_lbl, "cell")
        for button in (self.shortcuts_btn, self.params_btn, self.active_btn):
            stage_header_action_button(button, "cell")
        row.addWidget(self.correction_header_lbl)
        row.addWidget(self.shortcuts_btn)
        row.addWidget(self.params_btn)
        row.addWidget(self.active_btn)
        row.addStretch(1)
        return header

    # ------------------------------------------------------------------ #
    # Signal wiring                                                        #
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        self.save_labels_btn.clicked.connect(self._on_save_labels)
        self.fill_holes_btn.clicked.connect(self._on_fill_holes)
        self.cleanup_btn.clicked.connect(self._on_cleanup)
        self.outline_btn.toggled.connect(self._on_outline_toggled)
        self.correction_widget._outline_btn.toggled.connect(
            self._on_embedded_outline_toggled
        )
        self.active_btn.toggled.connect(self._on_active_button_toggled)
        self.params_btn.toggled.connect(self._on_params_button_toggled)
        self.shortcuts_btn.toggled.connect(self._on_shortcuts_button_toggled)
        if self._full_editing:
            self.retrack_back_btn.clicked.connect(self._on_retrack_backward)
            self.retrack_fwd_btn.clicked.connect(self._on_retrack_forward)
            self.validate_btn.clicked.connect(self._on_toggle_validation)
            self.remove_unvalidated_btn.clicked.connect(
                self._on_remove_unvalidated_labels
            )
            self.correction_widget.add_selection_listener(
                self._on_track_selection_changed
            )
            self.viewer.dims.events.current_step.connect(
                self._on_dims_changed_for_lineage
            )
        if self._active_layer_mode():
            # Standalone tools have no project directory to poll — readiness
            # here is "is the viewer's active layer a Labels layer", which
            # only changes on a selection change.
            active_events = getattr(
                getattr(getattr(self.viewer.layers, "selection", None), "events", None),
                "active", None,
            )
            if active_events is not None:
                active_events.connect(lambda *_a, **_k: self._sync_active_btn_enabled())
            self.active_btn.toggled.connect(lambda _c: self._sync_active_btn_enabled())

    @staticmethod
    def _set_checked_without_signal(button, checked: bool) -> None:
        set_checked_without_signal(button, checked)

    def _sync_correction_panel_visibility(self) -> None:
        show_params = self.params_btn.isChecked()
        show_shortcuts = self.shortcuts_btn.isChecked()
        show_active = self.active_btn.isChecked()

        self.correction_params_section.setVisible(show_params)
        self.correction_shortcuts_section.setVisible(show_shortcuts)
        self.active_content.setVisible(show_active)

        if show_params or show_shortcuts or show_active:
            self.section.expand()
        else:
            self.section.collapse()

    def _active_layer_mode(self) -> bool:
        """True when bound to the viewer's active Labels layer (standalone)."""
        return self._active_labels_layer_provider is not None

    def _correction_data_available(self) -> bool:
        """True when there is a cell-label stack to correct.

        Active-layer mode: the provider yields a Labels layer (None otherwise).
        Disk mode: a tracked cell-label file exists on disk.
        """
        if self._active_layer_mode():
            return self._active_labels_layer_provider() is not None
        lp = self._cell_labels_path()
        return lp is not None and lp.exists()

    def _sync_active_btn_enabled(self) -> None:
        """Locally gate the activate button in active-layer mode.

        Not routed through the parent widget's ``UiGate`` — this widget is
        reused in disk-mode contexts (the app) where this precondition doesn't
        apply. Stays enabled while already checked so an active correction
        session can always be turned back off even if the viewer's active-layer
        selection has since moved off the bound Labels layer.
        """
        if not self._active_layer_mode():
            return
        available = self._correction_data_available() or self.active_btn.isChecked()
        self.active_btn.setEnabled(available)
        self.active_btn.setToolTip(
            "Activate correction mode and show correction controls."
            if available
            else "Select a Labels layer to correct — the active layer is not one."
        )

    def _toggle_active_layer_correction(self, active: bool) -> None:
        """Bind/unbind the corrector to the viewer's active Labels layer.

        Standalone mode: there is no on-disk label file — the corrector edits the
        currently-active napari Labels layer in place and the user saves it via
        napari. Activating does not capture/hide other layers or load backdrops;
        deactivating loses nothing (edits already live in the layer), so there is
        no unsaved-changes prompt.
        """
        if active:
            layer = self._active_labels_layer_provider()
            if layer is None:
                self._correction_status(
                    "Select a Labels layer to correct — the active layer is not one."
                )
                self._set_checked_without_signal(self.active_btn, False)
                self._sync_correction_panel_visibility()
                return
            self._active_bound_layer = layer
            self.viewer.layers.selection.active = layer
            self.correction_widget.activate_layer(layer)
            self._bind_full_editing_keys(layer)
            if self._full_editing:
                # A fresh binding starts a fresh validation session — stale
                # entries from a previous layer/segment run would otherwise
                # collide with unrelated cell ids by coincidence.
                self._validated_tracks = {}
                self._lineage_canvas.refresh()
                self._refresh_validated_overlay()
            shape = tuple(np.asarray(layer.data).shape)
            self._correction_status(f"Correcting '{layer.name}' {shape}.")
            self._sync_correction_panel_visibility()
            return
        self._set_checked_without_signal(self.active_btn, False)
        self._unbind_full_editing_keys()
        self.correction_widget.deactivate()
        self._active_bound_layer = None
        self._correction_dirty = False
        self._sync_correction_panel_visibility()

    def _on_active_button_toggled(self, active: bool) -> None:
        if self._active_layer_mode():
            self._toggle_active_layer_correction(active)
            return
        if active:
            if not self._correction_data_available():
                self._correction_status("No cell labels available to correct.")
                self._set_checked_without_signal(self.active_btn, False)
                self._sync_correction_panel_visibility()
                return
            self._capture_correction_view_state()
            hide_all_layers(self.viewer.layers)

            if not self._load_correction_layers_from_disk():
                self._restore_correction_view_state()
                self._set_checked_without_signal(self.active_btn, False)
                self._sync_correction_panel_visibility()
                return

            layer = self.viewer.layers[_TRACKED_CELL_LAYER]
            layer.visible = True
            for name in (_CELL_ZAVG_LAYER, _NUC_ZAVG_LAYER):
                if name in self.viewer.layers:
                    self.viewer.layers[name].visible = True
            self.viewer.layers.selection.active = layer
            self.correction_widget.activate_layer(layer)
            self._sync_correction_panel_visibility()
            return

        if not self._confirm_deactivate_with_unsaved_changes():
            self._set_checked_without_signal(self.active_btn, True)
            self._sync_correction_panel_visibility()
            return

        self._set_checked_without_signal(self.active_btn, False)
        self.correction_widget.deactivate()
        self._refresh_tracked_layer_from_disk()
        self._remove_correction_owned_layers()
        self._restore_correction_view_state()
        self._sync_correction_panel_visibility()

    def _on_correction_edit(self, t: int, changed_ids: set[int]) -> None:
        """An inline contour edit leaves the cell labels unsaved."""
        self._correction_dirty = True

    def _on_outline_toggled(self, checked: bool) -> None:
        """Drive the embedded widget's outline view from the toolbar toggle."""
        self.correction_widget._outline_btn.setChecked(checked)

    def _on_embedded_outline_toggled(self, checked: bool) -> None:
        """Mirror the embedded outline checkbox onto the toolbar toggle.

        ``activate_layer`` ticks the embedded checkbox on, so this keeps the
        toolbar toggle in step without re-entering ``_on_outline_toggled``.
        """
        self._set_checked_without_signal(self.outline_btn, checked)

    def _confirm_deactivate_with_unsaved_changes(self) -> bool:
        if not self._correction_dirty:
            return True
        action = confirm_unsaved_before_deactivate(self, save_noun="cell labels")
        if action == "cancel":
            return False
        if action == "save":
            self._on_save_labels()
        self._correction_dirty = False
        return True

    def _correction_tracked_layer(self):
        """Return the layer being corrected, or None if absent.

        Active-layer mode: the bound active Labels layer. Disk mode: the
        ``[Correction] Cell Labels`` layer loaded from disk.
        """
        if self._active_layer_mode():
            return self._active_bound_layer
        if _TRACKED_CELL_LAYER in self.viewer.layers:
            return self.viewer.layers[_TRACKED_CELL_LAYER]
        return None

    def _load_correction_layers_from_disk(self) -> bool:
        lp = self._cell_labels_path()
        if lp is None or not lp.exists():
            self._correction_status("No cell labels file found.")
            return False
        try:
            labels = read_full_tracked_stack(lp)
        except Exception as exc:
            self._correction_status(f"Error reading cell labels: {exc}")
            return False
        czp, nzp = self._cell_foreground_path(), self._nuc_foreground_path()
        cz = (
            np.asarray(tifffile.imread(str(czp)), dtype=np.float32)
            if czp and czp.exists() else None
        )
        nz = (
            np.asarray(tifffile.imread(str(nzp)), dtype=np.float32)
            if nzp and nzp.exists() else None
        )
        self._apply_loaded_layers(labels, cz, nz)
        self._correction_status(f"Loaded cell label stack {labels.shape}.")
        return True

    def _refresh_tracked_layer_from_disk(self) -> None:
        """Reload the pipeline-side Tracked: Cell layer from disk on deactivate.

        Mirrors NucleusCorrectionWidget so a subsequent re-solve picks up any
        corrections saved during the session. The in-correction layer
        (``[Correction] Cell Labels``) is torn down separately via
        ``_remove_correction_owned_layers``.
        """
        lp = self._cell_labels_path()
        if lp is None or not lp.exists():
            return
        if _PIPELINE_TRACKED_CELL_LAYER not in self.viewer.layers:
            return
        try:
            data = np.asarray(read_full_tracked_stack(lp), dtype=np.uint32)
            self.viewer.layers[_PIPELINE_TRACKED_CELL_LAYER].data = data
        except Exception:
            pass

    def _on_params_button_toggled(self, checked: bool) -> None:
        self.correction_params_section._toggle.setChecked(checked)
        if checked:
            self._set_checked_without_signal(self.shortcuts_btn, False)
            self.correction_shortcuts_section._toggle.setChecked(False)
        self._sync_correction_panel_visibility()

    def _on_shortcuts_button_toggled(self, checked: bool) -> None:
        self.correction_shortcuts_section._toggle.setChecked(checked)
        if checked:
            self._set_checked_without_signal(self.params_btn, False)
            self.correction_params_section._toggle.setChecked(False)
        self._sync_correction_panel_visibility()

    # ------------------------------------------------------------------ #
    # Status helper                                                        #
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    # Path helpers (delegate to _pos_dir)                                 #
    # ------------------------------------------------------------------ #

    def _p(self, *parts: str) -> Path | None:
        return self._pos_dir.joinpath(*parts) if self._pos_dir else None

    def _cell_labels_path(self) -> Path | None:
        if self._labels_path_provider is not None:
            return self._labels_path_provider()
        return self._p("3_cell", "tracked_labels.tif")

    def _cell_foreground_path(self) -> Path | None:
        if self._cell_ref_path_provider is not None:
            return self._cell_ref_path_provider()
        return self._p("1_cellpose", "cell_foreground.tif")

    def _nuc_foreground_path(self) -> Path | None:
        if self._nucleus_ref_path_provider is not None:
            return self._nucleus_ref_path_provider()
        return self._p("1_cellpose", "nucleus_foreground.tif")

    # ------------------------------------------------------------------ #
    # Frame helpers                                                        #
    # ------------------------------------------------------------------ #


    def _correction_frame_indices(self, layer) -> list[int]:
        """Return frame indices based on the correction scope combo."""
        if layer.data.ndim < 3:
            return [0]
        if self.correction_scope_combo.currentText() == "All frames":
            return list(range(int(layer.data.shape[0])))
        return [self._current_t()]

    # ------------------------------------------------------------------ #
    # Load labels                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _broadcast_ref(image, shape):
        if image is None:
            return None
        if image.ndim == 2 and len(shape) >= 3:
            return np.broadcast_to(image[np.newaxis], (shape[0],) + image.shape).copy()
        return image

    def _show_layer(self, name, data, kwargs, adder):
        if name in self.viewer.layers:
            try:
                self.viewer.layers[name].data = data
            except Exception:
                self.viewer.layers.remove(self.viewer.layers[name])
                adder(data, name=name, **kwargs)
        else:
            adder(data, name=name, **kwargs)

    def _apply_loaded_layers(self, labels, cz, nz) -> None:
        # Add the reference images first so the labels layer ends up at the
        # top of the layer stack (otherwise the z-avg images render over it).
        for img, name, cmap in (
            (self._broadcast_ref(cz, labels.shape), _CELL_ZAVG_LAYER, "gray"),
            (self._broadcast_ref(nz, labels.shape), _NUC_ZAVG_LAYER, "I Purple"),
        ):
            if img is None:
                continue
            if name in self.viewer.layers:
                self.viewer.layers[name].data = img
                self.viewer.layers[name].colormap = cmap
                self.viewer.layers[name].blending = "minimum"
            else:
                self.viewer.add_image(img, name=name, colormap=cmap, blending="minimum")
            self._correction_owned_layers.add(name)
        self._show_layer(_TRACKED_CELL_LAYER, labels, {}, self.viewer.add_labels)
        self._correction_owned_layers.add(_TRACKED_CELL_LAYER)

    # ------------------------------------------------------------------ #
    # Save labels                                                          #
    # ------------------------------------------------------------------ #

    def _on_save_labels(self) -> None:
        if self._active_layer_mode():
            self._correction_status(
                "Save the layer via napari (File ▸ Save Selected Layers)."
            )
            return
        lp = self._cell_labels_path()
        if lp is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No labels layer."); return
        data = np.asarray(layer.data)
        if data.ndim != 3:
            self._correction_status("Labels not 3D."); return
        lp.parent.mkdir(parents=True, exist_ok=True)
        imwrite_grayscale(lp, data.astype(np.uint32, copy=False), compression="zlib")
        self._files_widget_refresh(self._pos_dir)
        self._correction_dirty = False
        self._correction_status(f"Saved {data.shape[0]} frames → {lp.name}.")

    # ------------------------------------------------------------------ #
    # Fill Holes                                                           #
    # ------------------------------------------------------------------ #

    def _on_fill_holes(self) -> None:
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No cell labels loaded."); return
        radius = int(self.hole_radius_spin.value())
        frames = self._correction_frame_indices(layer)

        changed_frames = 0
        changed_pixels = 0
        for t in frames:
            seg2d = self.correction_widget._frame_view(layer, t)
            before = seg2d.copy()
            result = fill_label_holes(seg2d, radius=radius)
            np.copyto(seg2d, result)
            diff = int(np.sum(before != seg2d))
            if diff:
                changed_frames += 1
                changed_pixels += diff
                self.correction_widget._record_history(layer, t, before)

        if changed_pixels:
            layer.refresh()
            if self.correction_widget._selected_label:
                t_now = self._current_t()
                self.correction_widget._update_highlight(
                    t_now, self.correction_widget._selected_label,
                )
            self._correction_status(
                f"Filled holes in {changed_frames} frame(s), "
                f"{changed_pixels} px changed. Unsaved."
            )
        else:
            self._correction_status("No interior holes found.")

    # ------------------------------------------------------------------ #
    # Remove Stranded Fragments                                            #
    # ------------------------------------------------------------------ #

    def _on_cleanup(self) -> None:
        """Drop disconnected same-label fragments on the scoped frame(s).

        Cells stay tied to the nuclei, so this only removes stray pixels that
        broke off a label (e.g. after a carve); it never renumbers or resyncs.
        Per-frame so each change is undoable via the layer history.
        """
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No cell labels loaded."); return
        frames = self._correction_frame_indices(layer)

        changed_frames = 0
        changed_pixels = 0
        for t in frames:
            seg2d = self.correction_widget._frame_view(layer, t)
            before = seg2d.copy()
            clean_stranded_pixels(seg2d)  # mutates seg2d in place
            diff = int(np.sum(before != seg2d))
            if diff:
                changed_frames += 1
                changed_pixels += diff
                self.correction_widget._record_history(layer, t, before)

        if changed_pixels:
            layer.refresh()
            if self.correction_widget._selected_label:
                self.correction_widget._update_highlight(
                    self._current_t(), self.correction_widget._selected_label,
                )
            self._correction_status(
                f"Removed stranded fragments in {changed_frames} frame(s), "
                f"{changed_pixels} px changed. Unsaved."
            )
        else:
            self._correction_status("No stranded fragments found.")

    # ------------------------------------------------------------------ #
    # Retracker (standalone full-editing only) — DB-free                  #
    # ------------------------------------------------------------------ #

    def _bind_full_editing_keys(self, layer) -> None:
        """Bind the Q/E retrack + V validation hotkeys (full-editing only).

        Layer-level keymap entries take precedence over napari's defaults, so
        these reliably drive the DB-free toolkit while correction is active.
        The inner CorrectionWidget already binds Delete (erase) when not
        contour-only.
        """
        if not self._full_editing:
            return
        self._unbind_full_editing_keys()
        for key, slot in (
            ("Q", self._on_retrack_backward),
            ("E", self._on_retrack_forward),
            ("V", self._on_toggle_validation),
        ):
            def handler(_layer, _slot=slot):
                _slot()

            layer.bind_key(key, handler, overwrite=True)
        self._retrack_keys_layer = layer

    def _unbind_full_editing_keys(self) -> None:
        layer = self._retrack_keys_layer
        if layer is None:
            return
        for key in ("Q", "E", "V"):
            try:
                layer.bind_key(key, None)
            except Exception:
                pass
        self._retrack_keys_layer = None

    def _on_retrack_forward(self) -> None:
        self._on_retrack("forward")

    def _on_retrack_backward(self) -> None:
        self._on_retrack("backward")

    def _on_retrack(self, direction: str) -> None:
        """Re-link the tracked stack from the current frame outward (DB-free).

        Greedy geometric similarity to the already-retracked neighbour toward the
        current frame; no validation/locking, so it is the pure standalone case
        of the nucleus widget's Q/E retracker.
        """
        from cellflow.cellpose.retrack import retrack_stack

        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No labels layer to retrack."); return
        data = np.asarray(layer.data)
        if data.ndim != 3 or data.shape[0] < 2:
            self._correction_status(
                "Retrack needs a 3D time-first stack (>= 2 frames)."
            )
            return
        t0 = self._current_t()
        if direction == "forward" and t0 >= data.shape[0] - 1:
            self._correction_status("Already at the last frame."); return
        if direction == "backward" and t0 <= 0:
            self._correction_status("Already at the first frame."); return
        try:
            out = retrack_stack(
                data,
                start_frame=t0,
                direction=direction,
                max_dist_px=float(self.retrack_max_dist_spin.value()),
            )
        except ValueError as exc:
            self._correction_status(str(exc)); return
        layer.data = out
        layer.refresh()
        self._correction_dirty = True
        if self.correction_widget._selected_label:
            self.correction_widget._update_highlight(
                t0, self.correction_widget._selected_label,
            )
        self._lineage_canvas.refresh()
        self._refresh_validated_overlay()
        self._correction_status(f"Retracked {direction} from t={t0}. Unsaved.")

    # ------------------------------------------------------------------ #
    # Validation + track-list navigator (standalone full-editing only)    #
    # ------------------------------------------------------------------ #

    def _lineage_tracked_data(self) -> np.ndarray | None:
        layer = self._correction_tracked_layer()
        return np.asarray(layer.data) if layer is not None else None

    def _current_cell_ids(self, t: int) -> set[int]:
        layer = self._correction_tracked_layer()
        if layer is None:
            return set()
        frame = frame_view_2d(np.asarray(layer.data), t)
        if frame is None:
            return set()
        return set(int(v) for v in np.unique(frame)) - {0}

    def _frames_with_cell(self, cell_id: int) -> list[int]:
        layer = self._correction_tracked_layer()
        if layer is None:
            return []
        data = np.asarray(layer.data)
        if data.ndim < 3:
            return [0] if int(cell_id) in np.unique(data) else []
        frames = []
        for t in range(data.shape[0]):
            frame = frame_view_2d(data, t)
            if frame is not None and int(cell_id) in frame:
                frames.append(t)
        return frames

    def _deselect_if_selection_gone(self) -> None:
        """Clear the selection when the selected cell no longer exists at the
        current frame (a relabel / removal may have dropped it)."""
        sel = self.correction_widget._selected_label
        if sel:
            ct = self._current_t()
            if sel not in self._current_cell_ids(ct):
                self.correction_widget.select_label(ct, 0)

    def _on_toggle_validation(self) -> None:
        """Toggle validation for the selected cell across all its frames (V).

        DB-free: flips membership in the in-memory ``self._validated_tracks``
        store rather than writing ``validated_cells.json`` (there is no project
        directory in the standalone active-layer tool).
        """
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No labels layer."); return
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
        if sel in self._validated_tracks:
            del self._validated_tracks[sel]
            self._correction_status(
                f"Cell {sel} invalidated across {len(frames)} frame(s)."
            )
        else:
            self._validated_tracks[sel] = set(frames)
            self._correction_status(
                f"Cell {sel} validated across {len(frames)} frame(s)."
            )
        # Flag-only change (no pixels touched) — recolour without rescanning.
        self._lineage_canvas.refresh_status()
        self._refresh_validated_overlay()

    def _on_remove_unvalidated_labels(self) -> None:
        """Zero out cell label pixels not validated for their frame."""
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No cell labels loaded."); return
        data = np.asarray(layer.data)
        # remove_unvalidated_labels expects a time-first stack of 2D frames;
        # squeeze singleton extra dims the same way _frame_view tolerates.
        squeezed = data
        while squeezed.ndim > 3:
            if squeezed.shape[1] != 1:
                self._correction_status(
                    "Labels layer has a non-singleton extra dimension."
                )
                return
            squeezed = squeezed[:, 0]
        before = squeezed.copy()
        try:
            changed_frames, changed_pixels = remove_unvalidated_labels(
                squeezed, self._validated_tracks
            )
        except ValueError as exc:
            self._correction_status(str(exc)); return
        if not changed_pixels:
            self._correction_status("No unvalidated labels found."); return
        frame_range = range(squeezed.shape[0]) if squeezed.ndim >= 3 else [0]
        for t in frame_range:
            frame_before = before[t] if before.ndim >= 3 else before
            frame_after = squeezed[t] if squeezed.ndim >= 3 else squeezed
            if not np.array_equal(frame_before, frame_after):
                self.correction_widget._record_history(layer, t, frame_before)
        layer.refresh()
        self._deselect_if_selection_gone()
        self._lineage_canvas.refresh()
        self._refresh_validated_overlay()
        self._correction_status(
            f"Removed unvalidated labels in {changed_frames} frame(s), "
            f"{changed_pixels} px changed. Unsaved."
        )

    def _refresh_validated_overlay(self) -> None:
        """Draw an opaque green border around every validated cell.

        Standalone counterpart of ``ValidatedOverlayController``'s border
        mode: same visual (a napari Labels layer, contour-only, fully
        opaque), but masked from the in-memory ``self._validated_tracks``
        store instead of ``validated_cells.json`` (no project directory
        here to read one from).
        """
        layer = self._correction_tracked_layer()
        data = np.asarray(layer.data) if layer is not None else None
        if data is None or data.ndim < 3 or not self._validated_tracks:
            self._remove_validated_overlay()
            return
        mask = np.zeros(data.shape, dtype=np.uint8)
        n_frames = data.shape[0]
        for cell_id, frames in self._validated_tracks.items():
            for t in frames:
                if 0 <= t < n_frames:
                    mask[t][data[t] == int(cell_id)] = 1
        colormap = direct_colormap({None: (0, 0, 0, 0), 1: _VALIDATED_OVERLAY_COLOR})
        if _VALIDATED_OVERLAY_LAYER in self.viewer.layers:
            overlay = self.viewer.layers[_VALIDATED_OVERLAY_LAYER]
            overlay.data = mask
            overlay.colormap = colormap
        else:
            overlay = self.viewer.add_labels(
                mask, name=_VALIDATED_OVERLAY_LAYER, opacity=1.0, colormap=colormap,
            )
            self._correction_owned_layers.add(_VALIDATED_OVERLAY_LAYER)
        overlay.contour = _VALIDATED_OVERLAY_CONTOUR
        overlay.opacity = 1.0
        self.viewer.layers.selection.active = layer

    def _remove_validated_overlay(self) -> None:
        if _VALIDATED_OVERLAY_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_VALIDATED_OVERLAY_LAYER])
        self._correction_owned_layers.discard(_VALIDATED_OVERLAY_LAYER)

    def _on_track_selection_changed(self, _t: int, cell_id: int) -> None:
        """Keep the accordion's expanded track in step with the image selection."""
        self._lineage_canvas.set_selection(int(cell_id or 0))

    def _on_dims_changed_for_lineage(self, _event=None) -> None:
        self._lineage_canvas.set_current_frame(self._current_t())

    def _navigate_to_cell_from_lineage(self, t: int, cell_id: int) -> None:
        """Accordion click: jump to frame ``t`` and select ``cell_id``."""
        try:
            step = list(self.viewer.dims.current_step)
            if step:
                step[0] = int(t)
                self.viewer.dims.current_step = tuple(step)
        except Exception:
            logger.exception("track-list navigation: frame jump failed")
        if int(cell_id) and int(cell_id) not in self._current_cell_ids(int(t)):
            return
        try:
            self.correction_widget.select_label(int(t), int(cell_id))
        except Exception:
            logger.exception("track-list navigation: cell select failed")
