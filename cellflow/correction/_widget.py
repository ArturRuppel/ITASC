"""
Label correction dock widget.

Select a Labels layer and click Activate.  The shortcuts below become
active whenever that layer is the active layer in the viewer.
After corrections, re-run graph extraction in the Edge Analysis tab.

Label shortcuts
---------------
n                       Select next free label, switch to paint mode
Shift-n                 Select next free label, switch to fill mode
s  →  Ctrl+drag         Swap two labels (single frame)
Ctrl-z                  Undo

Right-click             Erase cell
Ctrl + Left-drag        Merge cells (drag from cell A onto touching cell B)
Ctrl + Right-drag       Split (watershed): drag two seed points on the SAME cell
Shift + Right-drag      Split by drawn line
Shift + Left-drag       Redraw junction
"""

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QGroupBox,
)
from qtpy.QtCore import Qt
import napari
import napari.layers
from napari.utils.notifications import show_error, show_info

from ._labels import (
    erase_cell, merge_cells, split_across,
    split_draw, redraw_junction, swap_labels,
    _free_label,
)

_DRAW_LAYER = "CorrectionDraw"


class CorrectionWidget(QWidget):
    """Dock widget for interactive label correction."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        self._layer: napari.layers.Labels = None
        self._swap_mode: bool = False

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
        self._image_combo.setToolTip("Image used for watershed split (Ctrl+Right-drag)")
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

        # label shortcuts reference
        lbl_ref = QGroupBox("Label shortcuts")
        lbl_lay = QVBoxLayout(lbl_ref)
        for key, desc in [
            ("n",                  "Paint with new label"),
            ("Shift-n",            "Fill with new label"),
            ("s → Ctrl+drag",      "Swap two labels"),
            ("Right-click",        "Erase cell"),
            ("Ctrl+Left-drag",     "Merge cells (A→B, touching)"),
            ("Ctrl+Right-drag",    "Split (watershed, same cell)"),
            ("Shift+Right-drag",   "Split by drawn line"),
            ("Shift+Left-drag",    "Redraw junction"),
        ]:
            lbl_lay.addWidget(QLabel(f"<tt>{key}</tt>  –  {desc}"))
        root.addWidget(lbl_ref)

        root.addStretch()

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
        self._layer = layer
        self._swap_mode = False

        # ── suspend conflicting napari callbacks ──────────────────────────
        # 1. viewer-level: drag_to_zoom intercepts all Alt+drag (and any
        #    future viewer-level drag callbacks that might interfere)
        self._saved_viewer_drag_cbs = list(self.viewer.mouse_drag_callbacks)
        self.viewer.mouse_drag_callbacks.clear()

        # 2. layer-level: force pan_zoom so napari doesn't add its own
        #    drawing callbacks (e.g. the 'draw' callback added in paint/erase
        #    mode) that would run alongside ours
        self._saved_layer_mode = layer.mode
        self._saved_layer_drag_cbs = list(layer.mouse_drag_callbacks)
        layer.mode = "pan_zoom"
        layer.mouse_drag_callbacks.clear()

        # ── make the layer the active selection so key bindings fire ──────
        self.viewer.layers.selection.active = layer

        # Pre-create the draw layer now (while Labels is the active layer) so
        # that _get_draw_layer() during a drag never triggers a layer switch.
        self._get_draw_layer()

        self._register_callbacks()
        self._activate_btn.setText("Deactivate")
        self._set_status(f"Active on '{layer.name}'")

    def _deactivate(self):
        if self._layer is not None:
            self._remove_callbacks()

            # ── restore layer state ───────────────────────────────────────
            self._layer.mouse_drag_callbacks.clear()
            self._layer.mode = self._saved_layer_mode   # re-adds mode callbacks
            # re-add any non-mode custom callbacks that were present before
            for cb in self._saved_layer_drag_cbs:
                if cb not in self._layer.mouse_drag_callbacks:
                    self._layer.mouse_drag_callbacks.append(cb)

            # ── restore viewer callbacks ──────────────────────────────────
            self.viewer.mouse_drag_callbacks.clear()
            for cb in self._saved_viewer_drag_cbs:
                self.viewer.mouse_drag_callbacks.append(cb)

        self._layer = None
        self._swap_mode = False
        self._saved_viewer_drag_cbs = []
        self._saved_layer_drag_cbs = []
        self._activate_btn.setText("Activate")
        self._activate_btn.setChecked(False)
        self._set_status("Inactive")
        self._cleanup_draw_layer()

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
        # Adding a layer makes it the active layer, which would steal mouse
        # events from our Labels layer.  Restore the active layer immediately.
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer
        return dl

    def _cleanup_draw_layer(self):
        if _DRAW_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_DRAW_LAYER])

    # ── callback registration ─────────────────────────────────────────────

    def _register_callbacks(self):
        layer = self._layer

        # ── key bindings ──────────────────────────────────────────────────

        def key_n(_layer):
            try:
                lab = _free_label(_layer.data)
                _layer.selected_label = lab
                _layer.mode = "paint"
                self._set_status(f"Paint — new label {lab}")
            except Exception as exc:
                show_error(f"key_n error: {exc}")

        def key_shift_n(_layer):
            try:
                lab = _free_label(_layer.data)
                _layer.selected_label = lab
                _layer.mode = "fill"
                self._set_status(f"Fill — new label {lab}")
            except Exception as exc:
                show_error(f"key_shift_n error: {exc}")

        def key_w(_layer):
            self._swap_mode = not self._swap_mode
            if self._swap_mode:
                self._set_status("Swap mode — Ctrl+click two cells")
            else:
                self._set_status(f"Active on '{layer.name}'")

        def key_undo(_layer):
            try:
                _layer.undo()
            except Exception as exc:
                show_error(f"undo error: {exc}")

        for key, fn in [
            ("n",         key_n),
            ("Shift-n",   key_shift_n),
            ("s",         key_w),
            ("Control-z", key_undo),
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
                # vispy Key.__str__ returns "<Key 'Control'>" — extract just the name
                def _mod(k):
                    s = str(k)
                    return s[6:-2] if s.startswith("<Key '") and s.endswith("'>") else s
                mods = {_mod(m) for m in event.modifiers}

                # ── label-edit mode ──────────────────────────────────────
                seg2d = _layer.data[t]

                # swap mode: Ctrl+Left-click drag
                if self._swap_mode:
                    if mods == {"Control"} and btn == 1:
                        pos_a = _layer.world_to_data(event.position)
                        yield
                        while event.type == "mouse_move":
                            yield
                        pos_b = _layer.world_to_data(event.position)
                        if swap_labels(seg2d, pos_a, pos_b):
                            _layer.refresh()
                        self._swap_mode = False
                        self._set_status(f"Active on '{_layer.name}'")
                    return

                # erase: Right-click, no modifiers
                if btn == 2 and not mods:
                    pos = _layer.world_to_data(event.position)
                    if erase_cell(seg2d, pos):
                        _layer.refresh()
                    return

                # merge / split across
                if mods == {"Control"} and btn in (1, 2):
                    pos_start = _layer.world_to_data(event.position)
                    yield
                    while event.type == "mouse_move":
                        yield
                    pos_end = _layer.world_to_data(event.position)
                    if btn == 1:
                        ok = merge_cells(seg2d, pos_start, pos_end)
                        self._set_status(
                            f"Merged — Active on '{_layer.name}'"
                            if ok else "Merge failed — labels not touching"
                        )
                    else:
                        ok = split_across(seg2d, self._image_frame(t), pos_start, pos_end)
                        self._set_status(
                            f"Split — Active on '{_layer.name}'"
                            if ok else "Split failed — seeds on different cells or result too small"
                        )
                    _layer.refresh()
                    return

                # split draw: Shift+Right-drag
                if mods == {"Shift"} and btn == 2:
                    dl = self._get_draw_layer()
                    dl.data = []
                    dl.visible = True
                    pos = [_layer.world_to_data(event.position)]
                    yield
                    while event.type == "mouse_move":
                        pos.append(_layer.world_to_data(event.position))
                        if len(pos) % 3 == 0:
                            dl.data = [np.array([[p[-2], p[-1]] for p in pos])]
                            dl.shape_type = ["path"]
                        yield
                    pos.append(_layer.world_to_data(event.position))
                    dl.data = []
                    dl.visible = False
                    ok = split_draw(seg2d, pos)
                    self._set_status(
                        f"Split — Active on '{_layer.name}'"
                        if ok else "Split draw failed — line did not divide the cell"
                    )
                    _layer.refresh()
                    return

                # redraw junction: Shift+Left-drag
                if mods == {"Shift"} and btn == 1:
                    dl = self._get_draw_layer()
                    dl.data = []
                    dl.visible = True
                    pos = [_layer.world_to_data(event.position)]
                    yield
                    while event.type == "mouse_move":
                        pos.append(_layer.world_to_data(event.position))
                        if len(pos) % 3 == 0:
                            dl.data = [np.array([[p[-2], p[-1]] for p in pos])]
                            dl.shape_type = ["path"]
                        yield
                    pos.append(_layer.world_to_data(event.position))
                    dl.data = []
                    dl.visible = False
                    ok = redraw_junction(seg2d, pos)
                    self._set_status(
                        f"Junction redrawn — Active on '{_layer.name}'"
                        if ok else "Redraw failed — could not find two adjacent cells"
                    )
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
