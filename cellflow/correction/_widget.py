"""
Label correction dock widget.

Select a Labels layer and click Activate.  The shortcuts below become
active whenever that layer is the active layer in the viewer.
After corrections, re-run graph extraction in the Edge Analysis tab.

Correction shortcuts
--------------------
Left-click              Select / highlight cell (click background to deselect)
Delete                  Erase selected cell
Ctrl+Left-click (×2)   Merge or split:
                          • same cell twice  → split (watershed, 2 seed clicks)
                          • two diff cells   → merge
Ctrl+Right-click        Start swap — then Right-click second cell
n                       Select next free label, switch to paint mode
f                       Select next free label, switch to fill mode
Ctrl-z                  Undo
Shift+Right-drag        Split by drawn line (uses selected cell if set)
Shift+Left-drag         Redraw junction
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
from skimage.measure import find_contours

from ._labels import (
    erase_cell, merge_cells, split_across,
    split_draw, redraw_junction, swap_labels,
    _free_label, _label_at,
)

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
        self._ctrl_click_first        = None   # first Ctrl+Left-Click position
        self._ctrl_click_first_label: int = 0
        self._swap_first_pos          = None   # first Ctrl+Right-Click position

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
            ("Left-click",                   "Select / highlight cell"),
            ("Delete",                        "Erase selected cell"),
            ("Ctrl+Left-click → same cell",   "Split (watershed, 2 seed clicks)"),
            ("Ctrl+Left-click → diff cell",   "Merge two cells"),
            ("Ctrl+Right-click → Right-click","Swap labels"),
            ("n",                             "Paint with new label"),
            ("f",                             "Fill with new label"),
            ("Ctrl-z",                        "Undo"),
            ("Shift+Right-drag",              "Split by drawn line"),
            ("Shift+Left-drag",               "Redraw junction"),
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
        self._layer = layer
        self._selected_label = 0
        self._ctrl_click_first = None
        self._ctrl_click_first_label = 0
        self._swap_first_pos = None

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
        self._ctrl_click_first = None
        self._ctrl_click_first_label = 0
        self._swap_first_pos = None
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
                _layer.selected_label = lab
                _layer.mode = "paint"
                self._set_status(f"Paint — new label {lab}")
            except Exception as exc:
                show_error(f"key_n error: {exc}")

        def key_f(_layer):
            try:
                lab = _free_label(_layer.data)
                _layer.selected_label = lab
                _layer.mode = "fill"
                self._set_status(f"Fill — new label {lab}")
            except Exception as exc:
                show_error(f"key_f error: {exc}")

        def key_delete(_layer):
            try:
                if self._selected_label == 0:
                    self._set_status("No cell selected — left-click a cell first")
                    return
                t = int(self.viewer.dims.current_step[0])
                seg2d = _layer.data[t]
                if erase_cell(seg2d, label=self._selected_label):
                    _layer.refresh()
                    self._update_highlight(t, 0)
                    self._set_status(f"Erased — Active on '{_layer.name}'")
            except Exception as exc:
                show_error(f"delete error: {exc}")

        def key_undo(_layer):
            try:
                _layer.undo()
            except Exception as exc:
                show_error(f"undo error: {exc}")

        for key, fn in [
            ("n",         key_n),
            ("f",         key_f),
            ("Delete",    key_delete),
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
                def _mod(k):
                    s = str(k)
                    return s[6:-2] if s.startswith("<Key '") and s.endswith("'>") else s
                mods = {_mod(m) for m in event.modifiers}

                seg2d = _layer.data[t]
                pos   = _layer.world_to_data(event.position)

                # ── Ctrl+Right-click: start swap ──────────────────────────
                if btn == 2 and mods == {"Control"}:
                    lab = _label_at(seg2d, pos)
                    if lab == 0:
                        self._set_status("Swap — click on a cell (not background)")
                        return
                    self._swap_first_pos = pos
                    self._set_status(
                        f"Swap — label {lab} selected, right-click second cell"
                    )
                    return

                # ── Plain Right-click: complete swap ──────────────────────
                if btn == 2 and not mods:
                    if self._swap_first_pos is not None:
                        if swap_labels(seg2d, self._swap_first_pos, pos):
                            _layer.refresh()
                            self._set_status(f"Swapped — Active on '{_layer.name}'")
                        else:
                            self._set_status(
                                "Swap failed — click on two different cells"
                            )
                        self._swap_first_pos = None
                    return

                # ── Ctrl+Left-click: merge or split (two-click) ───────────
                if btn == 1 and mods == {"Control"}:
                    lab = _label_at(seg2d, pos)
                    if lab == 0:
                        self._set_status("Click on a cell, not background")
                        return
                    if self._ctrl_click_first is None:
                        self._ctrl_click_first = pos
                        self._ctrl_click_first_label = lab
                        self._update_highlight(t, lab)
                        self._set_status(
                            f"Label {lab} — Ctrl+click same cell again to split, "
                            f"or Ctrl+click a different cell to merge"
                        )
                    else:
                        if lab == self._ctrl_click_first_label:
                            # same cell → watershed split (two seeds)
                            ok = split_across(
                                seg2d, self._image_frame(t),
                                self._ctrl_click_first, pos,
                            )
                            self._set_status(
                                f"Split — Active on '{_layer.name}'"
                                if ok else "Split failed — seeds too close or result too small"
                            )
                        else:
                            # different cell → merge
                            ok = merge_cells(seg2d, self._ctrl_click_first, pos)
                            self._set_status(
                                f"Merged — Active on '{_layer.name}'"
                                if ok else "Merge failed — labels not touching"
                            )
                        _layer.refresh()
                        self._ctrl_click_first = None
                        self._ctrl_click_first_label = 0
                        # keep highlight on the cell at current click position
                        self._update_highlight(t, _label_at(seg2d, pos))
                    return

                # ── Plain Left-click: select / highlight cell ─────────────
                if btn == 1 and not mods:
                    # cancel any in-progress multi-step operations
                    self._ctrl_click_first = None
                    self._ctrl_click_first_label = 0
                    self._swap_first_pos = None
                    lab = _label_at(seg2d, pos)
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
                    ok = split_draw(seg2d, pos_list, curlabel=curlabel)
                    self._set_status(
                        f"Split — Active on '{_layer.name}'"
                        if ok else "Split draw failed — line did not divide the cell"
                    )
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
                    ok = redraw_junction(seg2d, pos_list)
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
