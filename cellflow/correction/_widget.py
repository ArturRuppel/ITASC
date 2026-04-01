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
n                       Select next free label, switch to paint mode
f                       Select next free label, switch to fill mode
Ctrl-z                  Undo (native napari)
Shift+Right-drag        Split by drawn line (uses selected cell if set)
Shift+Left-drag         Redraw junction
"""

import logging
import os

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QGroupBox,
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

from ._labels import (
    erase_cell, merge_cells, split_across,
    split_draw, redraw_junction, swap_labels,
    _free_label, _label_at,
)


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

        # layer selectors
        row = QHBoxLayout()
        row.addWidget(QLabel("Labels layer:"))
        self._layer_combo = QComboBox()
        row.addWidget(self._layer_combo, stretch=1)
        root.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Image layer:"))
        self._image_combo = QComboBox()
        self._image_combo.setToolTip("Image used for watershed split (Ctrl+Left-click split)")
        row2.addWidget(self._image_combo, stretch=1)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(28)
        refresh_btn.setToolTip("Refresh layer list")
        refresh_btn.clicked.connect(self._refresh_layers)
        row2.addWidget(refresh_btn)
        root.addLayout(row2)

        self._refresh_layers()

        # activate toggle
        self._activate_btn = QPushButton("Activate")
        self._activate_btn.setCheckable(True)
        self._activate_btn.clicked.connect(self._toggle_active)
        root.addWidget(self._activate_btn)

        # status
        self._status = QLabel("Inactive")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color: palette(mid); font-style: italic;")
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
            ("n",                                   "Paint with new label"),
            ("f",                                   "Fill with new label"),
            ("Ctrl-z",                              "Undo (native napari)"),
            ("Shift+Right-drag",                    "Split by drawn line"),
            ("Shift+Left-drag",                     "Redraw junction"),
        ]:
            lbl_lay.addWidget(QLabel(f"<tt>{key}</tt>  –  {desc}"))
        root.addWidget(lbl_ref)

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

    def _refresh_layers(self):
        current_lab = self._layer_combo.currentText()
        current_img = self._image_combo.currentText()
        self._layer_combo.clear()
        self._image_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                self._layer_combo.addItem(layer.name)
            elif isinstance(layer, napari.layers.Image):
                self._image_combo.addItem(layer.name)
        for combo, prev in [(self._layer_combo, current_lab),
                            (self._image_combo, current_img)]:
            idx = combo.findText(prev)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    # ── activation ────────────────────────────────────────────────────────

    def _toggle_active(self, checked: bool):
        if checked:
            name = self._layer_combo.currentText()
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
        colour = "red" if error else "palette(mid)"
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

    # ── callback registration ─────────────────────────────────────────────

    def _register_callbacks(self):
        layer = self._layer

        # ── key bindings ──────────────────────────────────────────────────

        def key_n(_layer):
            try:
                lab = _free_label(_layer.data)
                log.debug("key_n: new label=%s", lab)
                _layer.selected_label = lab
                _layer.mode = "paint"
                self._set_status(f"Paint — new label {lab}")
            except Exception as exc:
                show_error(f"key_n error: {exc}")

        def key_f(_layer):
            try:
                lab = _free_label(_layer.data)
                log.debug("key_f: new label=%s", lab)
                _layer.selected_label = lab
                _layer.mode = "fill"
                self._set_status(f"Fill — new label {lab}")
            except Exception as exc:
                show_error(f"key_f error: {exc}")

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
            ("n",      key_n),
            ("f",      key_f),
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

                # ── Shift+Left-drag: redraw junction ──────────────────────
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
                    log.debug("redraw_junction: %d positions collected", len(pos_list))
                    before = seg2d.copy()
                    ok = redraw_junction(seg2d, pos_list)
                    log.debug("redraw_junction result: ok=%s", ok)
                    self._set_status(
                        f"Junction redrawn — Active on '{_layer.name}'"
                        if ok else "Redraw failed — could not find two adjacent cells"
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

    # ── helpers ───────────────────────────────────────────────────────────

    def _image_frame(self, t: int):
        name = self._image_combo.currentText()
        if name and name in self.viewer.layers:
            lyr = self.viewer.layers[name]
            if isinstance(lyr, napari.layers.Image):
                d = lyr.data
                if d.ndim == 3:
                    return d[t]
                if d.ndim == 2:
                    return d
        return None
