"""Correction section widget for the cell workflow."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile
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
from cellflow.napari._widget_helpers import (
    make_status as _make_status,
    tool_btn as _tool_btn,
)
from cellflow.napari._correction_layer_lifecycle import (
    LayerViewState,
    capture_layer_view_state,
    hide_all_layers,
    remove_owned_layers,
    restore_layer_view_state,
)
from cellflow.napari._correction_ui import (
    build_correction_toolbar,
    confirm_unsaved_before_deactivate,
    set_checked_without_signal,
)
from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.ui_style import (
    add_block_pair_row,
    block_grid,
    stage_header_action_button,
    stage_header_label,
)
from cellflow.napari.widgets import CollapsibleSection
from cellflow.napari._widget_helpers import islider as _islider

logger = logging.getLogger(__name__)

# Layers created by the cell correction widget itself carry a [Correction]
# prefix so they can be cleanly removed on deactivate without clobbering the
# regular pipeline layers (Tracked: Cell, Cell z-avg, …) shown elsewhere.
_TRACKED_CELL_LAYER = "[Correction] Cell Labels"
_CELL_ZAVG_LAYER = "[Correction] Cell z-avg"
_NUC_ZAVG_LAYER = "[Correction] Nucleus z-avg"

# Pipeline-side (non-correction) layer names — used as fallbacks where the
# user may have loaded layers manually, and as the target for the on-disk
# Tracked: Cell refresh when correction mode is exited.
_PIPELINE_TRACKED_CELL_LAYER = "Tracked: Cell"


class CellCorrectionWidget(QWidget):
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
    parent:
        Optional Qt parent.
    """

    def __init__(
        self,
        viewer,
        *,
        pos_dir_provider: Callable[[], Path | None] | None = None,
        files_widget_refresh_callback: Callable[[Path | None], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir_provider = pos_dir_provider
        self._local_pos_dir: Path | None = None
        self._files_widget_refresh = files_widget_refresh_callback or (lambda _pd: None)
        self._correction_view_state: LayerViewState | None = None
        self._correction_owned_layers: set[str] = set()
        self._correction_dirty: bool = False
        self._setup_ui()
        self._connect_signals()

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
        # Cell labels are tied to the nuclei: restrict to contour edits only
        # (select + Shift-left extend + Shift-right carve), border highlight,
        # and no spawn-radius control.
        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
            show_cleanup=False,
            show_spawn_controls=False,
            contour_only=True,
            highlight_style="border",
        )
        # Inline contour edits (extend / carve / split) flow through the inner
        # widget's own status label, so they never hit ``_correction_status`` and
        # its "Unsaved" dirty-tracking. Mark the panel dirty on every edit so the
        # exit confirm fires, just like the nucleus correction widget.
        self.correction_widget.set_edit_callback(self._on_correction_edit)
        # The outline view is driven from the toolbar ``outline_btn`` toggle now,
        # so hide the embedded checkbox and keep the two in sync.
        self.correction_widget._outline_btn.setVisible(False)

        # Thin vertical icon toolbar (nucleus styling: enlarged glyphs, ruled
        # groups, no pill) shown to the left of the editing surface.
        self.toolbar = build_correction_toolbar(
            self,
            [
                (self.outline_btn,),
                (self.save_labels_btn,),
                (self.fill_holes_btn, self.cleanup_btn),
            ],
        )
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

    def _correction_data_available(self) -> bool:
        """True when a tracked cell-label stack exists on disk to correct."""
        lp = self._cell_labels_path()
        return lp is not None and lp.exists()

    def _on_active_button_toggled(self, active: bool) -> None:
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

    def _remove_correction_owned_layers(self) -> None:
        remove_owned_layers(self.viewer.layers, self._correction_owned_layers)

    def _correction_tracked_layer(self):
        """Return the [Correction] Cell Labels layer, or None if absent."""
        if _TRACKED_CELL_LAYER in self.viewer.layers:
            return self.viewer.layers[_TRACKED_CELL_LAYER]
        return None

    # ------------------------------------------------------------------ #
    # View state capture / restore (shared with NucleusCorrectionWidget) #
    # ------------------------------------------------------------------ #

    def _capture_correction_view_state(self) -> None:
        self._correction_view_state = capture_layer_view_state(self.viewer.layers)

    def _restore_correction_view_state(self) -> None:
        restore_layer_view_state(self.viewer.layers, self._correction_view_state)
        self._correction_view_state = None

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

    def _correction_status(self, msg: str) -> None:
        self.correction_status_lbl.setText(msg)
        self.correction_status_lbl.setVisible(bool(msg))
        lowered = msg.lower()
        if "unsaved" in lowered:
            self._correction_dirty = True
        elif lowered.startswith("saved") or lowered.startswith("loaded"):
            self._correction_dirty = False
        if msg:
            logger.info(msg)

    # ------------------------------------------------------------------ #
    # Path helpers (delegate to _pos_dir)                                 #
    # ------------------------------------------------------------------ #

    def _p(self, *parts: str) -> Path | None:
        return self._pos_dir.joinpath(*parts) if self._pos_dir else None

    def _cell_labels_path(self) -> Path | None:
        return self._p("3_cell", "tracked_labels.tif")

    def _cell_foreground_path(self) -> Path | None:
        return self._p("1_cellpose", "cell_foreground.tif")

    def _nuc_foreground_path(self) -> Path | None:
        return self._p("1_cellpose", "nucleus_foreground.tif")

    # ------------------------------------------------------------------ #
    # Frame helpers                                                        #
    # ------------------------------------------------------------------ #

    def _current_t(self) -> int:
        step = getattr(getattr(self.viewer, "dims", None), "current_step", (0,))
        return int(step[0]) if len(step) >= 1 else 0

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
