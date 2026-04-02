"""
Label correction dock widget.

Select a Labels layer and click Activate.  The shortcuts below become
active whenever that layer is the active layer in the viewer.
After corrections, re-run graph extraction in the Edge Analysis tab.

Correction shortcuts
--------------------
Left-click              Select / highlight cell (click background to deselect)
Delete                  Erase selected cell
Ctrl+Left-click         Merge (if a cell is selected) or start split:
                          • cell selected + click diff cell → merge
                          • click same cell twice          → split (watershed)
Ctrl+Right-click        Swap: if a cell is selected, swaps immediately;
                          otherwise starts two-step swap (Right-click second cell)
Ctrl-z                  Undo
Shift+Right-drag        Split by drawn line (uses selected cell if set)
Shift+Left-drag         Draw cell path: extends selected cell along stroke,
                          or creates new cell if none selected
"""

import logging
import os

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QSpinBox,
)
from qtpy.QtCore import Qt
import napari
import napari.layers
from napari.utils.notifications import show_error, show_info
from skimage.measure import find_contours

log = logging.getLogger("cellflow.correction")
if os.environ.get("CELLFLOW_DEBUG"):
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[cellflow.correction] %(levelname)s %(message)s"))
        log.addHandler(_h)

from ..backend.labels import (
    erase_cell, merge_cells, split_across,
    split_draw, draw_cell_path, swap_labels,
    fix_cell_borders,
    _free_label, _label_at,
)
from .registry import get_state


def _record_history(layer, t: int, before: np.ndarray) -> None:
    """Push changed pixels in frame *t* onto napari's undo stack.

    Call *after* the in-place modification, passing the pre-modification
    snapshot as *before*.  Only pixels that actually changed are stored,
    so the undo atom is compact even for large frames.
    """
    after = layer.data[t]
    changed = np.where(before != after)
    if not changed[0].size:
        return
    indices = (np.full(changed[0].size, t, dtype=layer.data.dtype), *changed)
    layer._save_history((indices, before[changed], after[changed]))

_DRAW_LAYER      = "CorrectionDraw"
_HIGHLIGHT_LAYER = "CellHighlight"


class CorrectionWidget(QWidget):
    """Dock widget for interactive label correction."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self._state = get_state(viewer)

        self._layer: napari.layers.Labels = None

        # selection / operation state
        self._selected_label: int = 0          # currently highlighted cell
        self._selected_pos            = None   # world position of the left-click selection
        self._ctrl_click_first        = None   # first Ctrl+Left-Click position (split mode)
        self._ctrl_click_first_label: int = 0
        self._ctrl_click_first_t: int = -1     # time frame of first Ctrl+Left-Click
        self._swap_first_pos          = None   # first Ctrl+Right-Click position
        self._swap_first_t: int = -1           # time frame of first Ctrl+Right-Click

        self._drag_callbacks: list = []
        self._bound_keys: list = []

        # saved napari state (populated on activate, cleared on deactivate)
        self._saved_viewer_drag_cbs: list = []
        self._saved_layer_mode: str = "pan_zoom"
        self._saved_layer_drag_cbs: list = []

        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # activate toggle
        self._activate_btn = QPushButton("Activate")
        self._activate_btn.setCheckable(True)
        self._activate_btn.clicked.connect(self._toggle_active)
        root.addWidget(self._activate_btn)

        # mode-change warning (hidden until napari steals the tool mode)
        self._reset_mode_btn = QPushButton("⚠  Restore correction mode")
        self._reset_mode_btn.setVisible(False)
        self._reset_mode_btn.setStyleSheet(
            "QPushButton { background-color: #7a3c00; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #a05000; }"
        )
        self._reset_mode_btn.clicked.connect(self._reset_tool_mode)
        root.addWidget(self._reset_mode_btn)

        # status
        self._status = QLabel("Inactive")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color: palette(text); font-style: italic;")
        root.addWidget(self._status)

        # correction shortcuts reference (CellFlow custom — not napari native)
        lbl_ref = QGroupBox("Correction shortcuts")
        lbl_lay = QVBoxLayout(lbl_ref)
        for key, desc in [
            ("Left-click",                          "Select / highlight cell"),
            ("Delete",                              "Erase selected cell"),
            ("Ctrl+Left-click (cell selected)",     "Merge with clicked cell"),
            ("Ctrl+Left-click × 2 (same cell)",     "Split (watershed, 2 seeds)"),
            ("Ctrl+Right-click (cell selected)",      "Swap with clicked cell"),
            ("Ctrl+Right-click → Right-click",        "Swap (two-step, no selection)"),
            ("Ctrl-z",                              "Undo"),
            ("Shift+Right-drag",                    "Split by drawn line"),
            ("Shift+Left-drag",                     "Draw cell path (extends selected cell or creates new)"),
        ]:
            lbl_lay.addWidget(QLabel(f"<tt>{key}</tt>  –  {desc}"))
        root.addWidget(lbl_ref)

        # fix-borders batch operation
        fix_box = QGroupBox("Fix borders")
        fix_lay = QVBoxLayout(fix_box)

        fix_row = QHBoxLayout()
        fix_row.addWidget(QLabel("Radius (px):"))
        self._border_radius = QSpinBox()
        self._border_radius.setRange(1, 50)
        self._border_radius.setValue(2)
        fix_row.addWidget(self._border_radius)
        fix_lay.addLayout(fix_row)

        fix_desc = QLabel(
            "Dilates cells into narrow gaps between them.\n"
            "Free edges (open border) are not grown."
        )
        fix_desc.setWordWrap(True)
        fix_desc.setStyleSheet("font-size: 9pt; color: palette(text);")
        fix_lay.addWidget(fix_desc)

        self._fix_borders_btn = QPushButton("Fix borders (all frames)")
        self._fix_borders_btn.clicked.connect(self._run_fix_borders)
        fix_lay.addWidget(self._fix_borders_btn)

        root.addWidget(fix_box)

        root.addStretch()

        # attribution
        attrib = QLabel(
            'Correction tools adapted from '
            '<a href="https://github.com/Image-Analysis-Hub/Epicure">Epicure</a>.'
            '<br>If you use these tools, please cite:<br>'
            '<a href="https://doi.org/10.64898/2026.03.27.714683">'
            'doi:10.64898/2026.03.27.714683</a>'
        )
        attrib.setOpenExternalLinks(True)
        attrib.setWordWrap(True)
        attrib.setStyleSheet("color: palette(text); font-size: 9pt;")
        root.addWidget(attrib)

    # ── activation ────────────────────────────────────────────────────────

    def _toggle_active(self, checked: bool):
        if checked:
            name = self._state.tissue.labels_layer
            if not name or name not in self.viewer.layers:
                self._activate_btn.setChecked(False)
                self._set_status("Layer not found", error=True)
                return
            layer = self.viewer.layers[name]
            if not isinstance(layer, napari.layers.Labels):
                self._activate_btn.setChecked(False)
                self._set_status("Not a Labels layer", error=True)
                return
            self._activate(layer)
        else:
            self._deactivate()

    def _activate(self, layer: napari.layers.Labels):
        log.debug("activate: layer='%s' shape=%s", layer.name, layer.data.shape)
        self._layer = layer
        self._selected_label = 0
        self._selected_pos = None
        self._ctrl_click_first = None
        self._ctrl_click_first_label = 0
        self._ctrl_click_first_t = -1
        self._swap_first_pos = None
        self._swap_first_t = -1

        # ── suspend conflicting napari callbacks ──────────────────────────
        self._saved_viewer_drag_cbs = list(self.viewer.mouse_drag_callbacks)
        self.viewer.mouse_drag_callbacks.clear()

        self._saved_layer_mode = layer.mode
        self._saved_layer_drag_cbs = list(layer.mouse_drag_callbacks)
        layer.mode = "pan_zoom"
        layer.mouse_drag_callbacks.clear()

        # ── make the layer the active selection so key bindings fire ──────
        self.viewer.layers.selection.active = layer

        # Pre-create auxiliary layers while Labels is still the active layer
        self._get_draw_layer()
        self._get_highlight_layer()

        # Update highlight when the user scrubs through time
        self.viewer.dims.events.current_step.connect(self._on_dims_change)

        # Detect napari toolbar / shortcut stealing the tool mode
        layer.events.mode.connect(self._on_layer_mode_change)

        self._register_callbacks()
        self._activate_btn.setText("Deactivate")
        self._set_status(f"Active on '{layer.name}'")

    def _deactivate(self):
        log.debug("deactivate: layer='%s'", self._layer.name if self._layer else None)
        if self._layer is not None:
            self._remove_callbacks()

            try:
                self.viewer.dims.events.current_step.disconnect(self._on_dims_change)
            except Exception:
                pass

            try:
                self._layer.events.mode.disconnect(self._on_layer_mode_change)
            except Exception:
                pass

            # ── restore layer state ───────────────────────────────────────
            self._layer.mouse_drag_callbacks.clear()
            self._layer.mode = self._saved_layer_mode
            for cb in self._saved_layer_drag_cbs:
                if cb not in self._layer.mouse_drag_callbacks:
                    self._layer.mouse_drag_callbacks.append(cb)

            # ── restore viewer callbacks ──────────────────────────────────
            self.viewer.mouse_drag_callbacks.clear()
            for cb in self._saved_viewer_drag_cbs:
                self.viewer.mouse_drag_callbacks.append(cb)

        # Sync corrected labels to internal state before releasing the layer
        if self._layer is not None:
            self._state.set_tissue_labels(np.asarray(self._layer.data), self._layer.name)

        self._layer = None
        self._selected_label = 0
        self._selected_pos = None
        self._ctrl_click_first = None
        self._ctrl_click_first_label = 0
        self._ctrl_click_first_t = -1
        self._swap_first_pos = None
        self._swap_first_t = -1
        self._saved_viewer_drag_cbs = []
        self._saved_layer_drag_cbs = []
        self._activate_btn.setText("Activate")
        self._activate_btn.setChecked(False)
        self._set_status("Inactive")
        self._cleanup_draw_layer()
        self._cleanup_highlight_layer()

    def _set_status(self, msg: str, error: bool = False):
        self._status.setText(msg)
        colour = "red" if error else "palette(text)"
        self._status.setStyleSheet(f"color: {colour}; font-style: italic;")

    # ── draw layer ────────────────────────────────────────────────────────

    def _get_draw_layer(self):
        if _DRAW_LAYER in self.viewer.layers:
            return self.viewer.layers[_DRAW_LAYER]
        dl = self.viewer.add_shapes(
            name=_DRAW_LAYER,
            ndim=2,
            edge_color="yellow",
            edge_width=1,
            face_color="transparent",
        )
        dl.visible = False
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer
        return dl

    def _cleanup_draw_layer(self):
        if _DRAW_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_DRAW_LAYER])

    # ── highlight layer ───────────────────────────────────────────────────

    def _get_highlight_layer(self):
        if _HIGHLIGHT_LAYER in self.viewer.layers:
            return self.viewer.layers[_HIGHLIGHT_LAYER]
        hl = self.viewer.add_shapes(
            name=_HIGHLIGHT_LAYER,
            ndim=2,
            edge_color="cyan",
            edge_width=2,
            face_color="transparent",
        )
        hl.visible = False
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer
        return hl

    def _update_highlight(self, t: int, lab: int):
        """Redraw the cyan boundary for *lab* at time *t*.  Pass 0 to clear."""
        self._selected_label = lab
        hl = self._get_highlight_layer()
        if lab == 0 or self._layer is None:
            hl.data = []
            hl.visible = False
            return
        seg2d = self._layer.data[t]
        if not np.any(seg2d == lab):
            hl.data = []
            hl.visible = False
            return
        mask = (seg2d == lab).astype(np.uint8)
        contours = find_contours(mask, level=0.5)
        if not contours:
            hl.data = []
            hl.visible = False
            return
        contour = max(contours, key=len)
        hl.data = [contour]
        hl.shape_type = ["polygon"]
        hl.visible = True
        # Shapes layer addition can steal focus — restore active layer
        self.viewer.layers.selection.active = self._layer

    def _cleanup_highlight_layer(self):
        if _HIGHLIGHT_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_HIGHLIGHT_LAYER])

    def _on_dims_change(self, event=None):
        """Keep highlight current when the user changes the time slider."""
        if self._selected_label and self._layer is not None:
            t = int(self.viewer.dims.current_step[0])
            self._update_highlight(t, self._selected_label)

    def _on_layer_mode_change(self, event=None):
        """Show warning when napari changes the layer mode away from pan_zoom."""
        if self._layer is None:
            return
        mode = getattr(event, "value", None) or self._layer.mode
        log.debug("_on_layer_mode_change: mode=%s", mode)
        if mode != "pan_zoom":
            self._reset_mode_btn.setVisible(True)
            self._set_status("Tool mode changed — corrections disabled", error=True)
        else:
            self._reset_mode_btn.setVisible(False)
            if self._layer is not None:
                self._set_status(f"Active on '{self._layer.name}'")

    def _reset_tool_mode(self):
        """Restore pan_zoom mode so correction shortcuts work again."""
        if self._layer is not None:
            log.debug("_reset_tool_mode: restoring pan_zoom")
            self._layer.mode = "pan_zoom"
            # _on_layer_mode_change will hide the button and update status

    # ── callback registration ─────────────────────────────────────────────

    def _register_callbacks(self):
        layer = self._layer

        # ── key bindings ──────────────────────────────────────────────────

        def key_delete(_layer):
            try:
                log.debug("key_delete: selected_label=%s", self._selected_label)
                if self._selected_label == 0:
                    self._set_status("No cell selected — left-click a cell first")
                    return
                t = int(self.viewer.dims.current_step[0])
                seg2d = _layer.data[t]
                before = seg2d.copy()
                if erase_cell(seg2d, label=self._selected_label):
                    _record_history(_layer, t, before)
                    _layer.refresh()
                    self._update_highlight(t, 0)
                    self._set_status(f"Erased — Active on '{_layer.name}'")
            except Exception as exc:
                show_error(f"delete error: {exc}")

        for key, fn in [
            ("Delete", key_delete),
        ]:
            layer.bind_key(key, fn, overwrite=True)
            self._bound_keys.append(key)

        # ── mouse drag ────────────────────────────────────────────────────

        def on_drag(_layer, event):
            try:
                if event.type != "mouse_press":
                    return

                t   = int(self.viewer.dims.current_step[0])
                btn = event.button
                mods = {m.name for m in event.modifiers}

                seg2d = _layer.data[t]
                pos   = _layer.world_to_data(event.position)
                log.debug(
                    "on_drag: type=%s btn=%s mods=%s  world=%s data_pos=%s  t=%d "
                    "selected=%s ctrl_first=%s swap_first=%s",
                    event.type, btn, mods, event.position, pos, t,
                    self._selected_label, self._ctrl_click_first_label, self._swap_first_pos,
                )

                # ── Ctrl+Right-click: swap ────────────────────────────────
                if btn == 2 and mods == {"Control"}:
                    lab = _label_at(seg2d, pos)
                    log.debug("swap-ctrl-right: label_at_click=%s selected=%s", lab, self._selected_label)
                    if lab == 0:
                        self._set_status("Swap — click on a cell (not background)")
                        return
                    if (
                        self._selected_label != 0
                        and self._selected_pos is not None
                        and lab != self._selected_label
                    ):
                        # Cell already selected → swap directly
                        before = seg2d.copy()
                        ok = swap_labels(seg2d, self._selected_pos, pos)
                        log.debug("swap direct: ok=%s", ok)
                        if ok:
                            _record_history(_layer, t, before)
                            _layer.refresh()
                            self._selected_label = 0
                            self._selected_pos = None
                            self._update_highlight(t, 0)
                            self._set_status(f"Swapped — Active on '{_layer.name}'")
                        else:
                            self._set_status("Swap failed — click on two different cells")
                    else:
                        # No prior selection → enter two-step swap mode
                        self._swap_first_pos = pos
                        self._swap_first_t = t
                        log.debug("swap: two-step mode started, first_pos=%s first_t=%d", pos, t)
                        self._set_status(
                            f"Swap — label {lab} selected, right-click second cell"
                        )
                    return

                # ── Plain Right-click: complete swap ──────────────────────
                if btn == 2 and not mods:
                    log.debug("plain right-click: swap_first_pos=%s swap_first_t=%s t=%d", self._swap_first_pos, self._swap_first_t, t)
                    if self._swap_first_pos is not None:
                        if t != self._swap_first_t:
                            self._swap_first_pos = None
                            self._swap_first_t = -1
                            self._set_status("Frame changed — swap cancelled")
                        else:
                            before = seg2d.copy()
                            ok = swap_labels(seg2d, self._swap_first_pos, pos)
                            log.debug("swap two-step: ok=%s", ok)
                            if ok:
                                _record_history(_layer, t, before)
                                _layer.refresh()
                                self._swap_first_pos = None
                                self._swap_first_t = -1
                                self._set_status(f"Swapped — Active on '{_layer.name}'")
                            else:
                                self._set_status(
                                    "Swap failed — click on two different cells"
                                )
                                self._swap_first_pos = None
                                self._swap_first_t = -1
                    return

                # ── Ctrl+Left-click: merge (if cell selected) or split ────
                if btn == 1 and mods == {"Control"}:
                    lab = _label_at(seg2d, pos)
                    log.debug("ctrl-left-click: label_at_click=%s selected=%s ctrl_first=%s", lab, self._selected_label, self._ctrl_click_first_label)
                    if lab == 0:
                        self._set_status("Click on a cell, not background")
                        return

                    if self._ctrl_click_first is not None:
                        # ── already in split mode (waiting for second seed) ──
                        if t != self._ctrl_click_first_t:
                            # Frame changed — restart split mode
                            self._ctrl_click_first = pos
                            self._ctrl_click_first_label = lab
                            self._ctrl_click_first_t = t
                            self._update_highlight(t, lab)
                            self._set_status(
                                f"Frame changed — restarted: label {lab} selected"
                            )
                        elif lab == self._ctrl_click_first_label:
                            # Second seed on same cell → split
                            log.debug("split_across: first=%s second=%s label=%s", self._ctrl_click_first, pos, lab)
                            before = seg2d.copy()
                            ok = split_across(
                                seg2d, self._image_frame(t),
                                self._ctrl_click_first, pos,
                            )
                            log.debug("split_across result: ok=%s", ok)
                            self._set_status(
                                f"Split — Active on '{_layer.name}'"
                                if ok else "Split failed — seeds too close or result too small"
                            )
                            if ok:
                                _record_history(_layer, t, before)
                            _layer.refresh()
                            self._ctrl_click_first = None
                            self._ctrl_click_first_label = 0
                            self._ctrl_click_first_t = -1
                            self._update_highlight(t, _label_at(seg2d, pos))
                        else:
                            # Different cell during split mode — cancel split,
                            # fall through to merge-or-new-split logic below
                            log.debug("ctrl-left: different cell clicked during split mode — cancelling split")
                            self._ctrl_click_first = None
                            self._ctrl_click_first_label = 0
                            self._ctrl_click_first_t = -1

                    if self._ctrl_click_first is None:
                        # ── fresh click ───────────────────────────────────────
                        if (
                            self._selected_label != 0
                            and self._selected_pos is not None
                            and lab != self._selected_label
                        ):
                            # Cell already selected → merge directly
                            log.debug("merge: selected=%s clicked=%s", self._selected_label, lab)
                            before = seg2d.copy()
                            ok = merge_cells(seg2d, self._selected_pos, pos)
                            log.debug("merge result: ok=%s", ok)
                            self._set_status(
                                f"Merged — Active on '{_layer.name}'"
                                if ok else "Merge failed — labels not touching"
                            )
                            if ok:
                                _record_history(_layer, t, before)
                            _layer.refresh()
                            self._selected_label = 0
                            self._selected_pos = None
                            self._update_highlight(t, _label_at(seg2d, pos))
                        else:
                            # No prior selection (or clicking same cell) → start split
                            self._ctrl_click_first = pos
                            self._ctrl_click_first_label = lab
                            self._ctrl_click_first_t = t
                            self._update_highlight(t, lab)
                            log.debug("split mode: first seed set label=%s pos=%s", lab, pos)
                            self._set_status(
                                f"Label {lab} — Ctrl+click same cell again for second split seed"
                            )
                    return

                # ── Plain Left-click: select / highlight cell ─────────────
                if btn == 1 and not mods:
                    # cancel any in-progress multi-step operations
                    self._ctrl_click_first = None
                    self._ctrl_click_first_label = 0
                    self._ctrl_click_first_t = -1
                    self._swap_first_pos = None
                    self._swap_first_t = -1
                    lab = _label_at(seg2d, pos)
                    log.debug("left-click select: label_at_click=%s pos=%s", lab, pos)
                    self._selected_pos = pos if lab != 0 else None
                    self._update_highlight(t, lab)
                    if lab:
                        self._set_status(
                            f"Selected label {lab} — Active on '{_layer.name}'"
                        )
                    else:
                        self._set_status(f"Active on '{_layer.name}'")
                    return

                # ── Shift+Right-drag: split by drawn line ─────────────────
                if mods == {"Shift"} and btn == 2:
                    dl = self._get_draw_layer()
                    dl.data = []
                    dl.visible = True
                    pos_list = [_layer.world_to_data(event.position)]
                    yield
                    while event.type == "mouse_move":
                        pos_list.append(_layer.world_to_data(event.position))
                        if len(pos_list) % 3 == 0:
                            dl.data = [np.array([[p[-2], p[-1]] for p in pos_list])]
                            dl.shape_type = ["path"]
                        yield
                    pos_list.append(_layer.world_to_data(event.position))
                    dl.data = []
                    dl.visible = False
                    curlabel = self._selected_label if self._selected_label else None
                    log.debug("split_draw: %d positions collected, curlabel=%s", len(pos_list), curlabel)
                    before = seg2d.copy()
                    ok = split_draw(seg2d, pos_list, curlabel=curlabel)
                    log.debug("split_draw result: ok=%s", ok)
                    self._set_status(
                        f"Split — Active on '{_layer.name}'"
                        if ok else "Split draw failed — line did not divide the cell"
                    )
                    if ok:
                        _record_history(_layer, t, before)
                    _layer.refresh()
                    return

                # ── Shift+Left-drag: draw cell path ───────────────────────
                if mods == {"Shift"} and btn == 1:
                    dl = self._get_draw_layer()
                    dl.data = []
                    dl.visible = True
                    pos_list = [_layer.world_to_data(event.position)]
                    yield
                    while event.type == "mouse_move":
                        pos_list.append(_layer.world_to_data(event.position))
                        if len(pos_list) % 3 == 0:
                            dl.data = [np.array([[p[-2], p[-1]] for p in pos_list])]
                            dl.shape_type = ["path"]
                        yield
                    pos_list.append(_layer.world_to_data(event.position))
                    dl.data = []
                    dl.visible = False
                    curlabel = self._selected_label if self._selected_label else None
                    log.debug("draw_cell_path: %d positions collected, curlabel=%s", len(pos_list), curlabel)
                    before = seg2d.copy()
                    ok = draw_cell_path(seg2d, pos_list, curlabel=curlabel)
                    log.debug("draw_cell_path result: ok=%s", ok)
                    self._set_status(
                        f"Drew cell path — Active on '{_layer.name}'"
                        if ok else "Draw failed — stroke too short"
                    )
                    if ok:
                        _record_history(_layer, t, before)
                    _layer.refresh()
                    return

            except Exception as exc:
                import traceback
                show_error(f"Correction error: {exc}\n{traceback.format_exc()}")

        layer.mouse_drag_callbacks.append(on_drag)
        self._drag_callbacks.append(on_drag)

    def _remove_callbacks(self):
        layer = self._layer
        for fn in self._drag_callbacks:
            try:
                layer.mouse_drag_callbacks.remove(fn)
            except ValueError:
                pass
        self._drag_callbacks.clear()
        for key in self._bound_keys:
            try:
                layer.bind_key(key, None)
            except Exception:
                pass
        self._bound_keys.clear()

    def _run_fix_borders(self):
        """Apply fix_cell_borders to every frame of the active labels layer."""
        if self._layer is None:
            show_error("Activate the correction widget first")
            return
        radius = self._border_radius.value()
        data = self._layer.data
        n_frames = data.shape[0]
        changed_frames = 0
        for t in range(n_frames):
            frame = data[t]
            before = frame.copy()
            if fix_cell_borders(frame, radius=radius):
                _record_history(self._layer, t, before)
                changed_frames += 1
        if changed_frames:
            self._layer.refresh()
            self._set_status(
                f"Fixed borders (r={radius}) in {changed_frames}/{n_frames} frames"
                f" — Active on '{self._layer.name}'"
            )
        else:
            self._set_status(f"Fix borders: no gaps found — Active on '{self._layer.name}'")

    # ── helpers ───────────────────────────────────────────────────────────

    def _image_frame(self, t: int):
        name = self._state.tissue.image_layer
        if name and name in self.viewer.layers:
            lyr = self.viewer.layers[name]
            if isinstance(lyr, napari.layers.Image):
                d = lyr.data
                if d.ndim == 3:
                    return d[t]
                if d.ndim == 2:
                    return d
        return None
