"""
Label and track correction dock widget.

Select a Labels layer and click Activate.  The shortcuts below become
active whenever that layer is the active layer in the viewer.
After corrections, re-run graph extraction in the Edge Analysis tab.

Label shortcuts
---------------
n                       Select next free label, switch to paint mode
Shift-n                 Select next free label, switch to fill mode
w  →  Ctrl+click        Swap two labels (single frame)
Ctrl-z                  Undo

Right-click             Erase cell
Ctrl + Left-drag        Merge cells
Ctrl + Right-drag       Split (watershed, two seeds)
Alt  + Right-drag       Split by drawn line
Ctrl + Alt + Left-drag  Redraw junction

Track shortcuts  (press t to enter/leave track-edit mode)
---------------------------------------------------------
t                       Toggle track-edit mode
r                       Show / hide Tracks layer
l                       Re-colour labels by track (identity colourmap)

In track-edit mode
  Left-click                    Select first track for merge
  Right-click                   Complete merge (second track)
  Shift + Right-click           Split track at current frame
  Shift + Left-drag             Swap two tracks from current frame
  Ctrl + Left-click             Start manual re-link (pick source cell)
  Ctrl + Right-click            End manual re-link (pick destination cell)
  Alt  + Left-click             Interpolation: mark start frame
  Alt  + Right-click            Interpolation: mark end frame
  Ctrl + Alt + Right-click      Delete track from current frame
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
from ._tracks import (
    merge_tracks, split_track, swap_tracks,
    delete_track, reassign_cell, interpolate_track,
    _label_at as _track_label_at,
)

_DRAW_LAYER = "CorrectionDraw"


class CorrectionWidget(QWidget):
    """Dock widget for interactive label and track correction."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        self._layer: napari.layers.Labels = None
        self._swap_mode: bool = False

        # track-edit state
        self._track_mode: bool = False
        self._track_pending: str = None   # "merge" | "manual" | "interpolate"
        self._track_first: int = None     # first selected track ID
        self._interp_frame: int = None    # frame for interpolation start

        self._drag_callbacks: list = []
        self._bound_keys: list = []

        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # layer selector
        row = QHBoxLayout()
        row.addWidget(QLabel("Labels layer:"))
        self._layer_combo = QComboBox()
        self._refresh_layers()
        row.addWidget(self._layer_combo, stretch=1)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(28)
        refresh_btn.setToolTip("Refresh layer list")
        refresh_btn.clicked.connect(self._refresh_layers)
        row.addWidget(refresh_btn)
        root.addLayout(row)

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
            ("w → Ctrl+click",     "Swap two labels"),
            ("Right-click",        "Erase cell"),
            ("Ctrl+drag",          "Merge cells"),
            ("Ctrl+Right-drag",    "Split (watershed)"),
            ("Alt+Right-drag",     "Split by drawn line"),
            ("Ctrl+Alt+drag",      "Redraw junction"),
        ]:
            lbl_lay.addWidget(QLabel(f"<tt>{key}</tt>  –  {desc}"))
        root.addWidget(lbl_ref)

        # track shortcuts reference
        trk_ref = QGroupBox("Track shortcuts  (press t to toggle)")
        trk_lay = QVBoxLayout(trk_ref)
        for key, desc in [
            ("t",                        "Toggle track-edit mode"),
            ("r",                        "Show/hide Tracks layer"),
            ("l",                        "Re-colour by track ID"),
            ("Click → Right-click",      "Merge two tracks"),
            ("Shift+Right-click",        "Split track here"),
            ("Shift+drag",               "Swap two tracks"),
            ("Ctrl+click → Ctrl+Right",  "Manual re-link"),
            ("Alt+click → Alt+Right",    "Interpolate gap"),
            ("Ctrl+Alt+Right-click",     "Delete track here"),
        ]:
            trk_lay.addWidget(QLabel(f"<tt>{key}</tt>  –  {desc}"))
        root.addWidget(trk_ref)
        root.addStretch()

    def _refresh_layers(self):
        current = self._layer_combo.currentText()
        self._layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                self._layer_combo.addItem(layer.name)
        idx = self._layer_combo.findText(current)
        if idx >= 0:
            self._layer_combo.setCurrentIndex(idx)

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
        self._reset_track_state()
        self._register_callbacks()
        self._activate_btn.setText("Deactivate")
        n_drag = len(layer.mouse_drag_callbacks)
        n_keys = len(self._bound_keys)
        self._set_status(f"Active on '{layer.name}' — {n_drag} drag cb, {n_keys} keys")

    def _deactivate(self):
        if self._layer is not None:
            self._remove_callbacks()
        self._layer = None
        self._swap_mode = False
        self._reset_track_state()
        self._activate_btn.setText("Activate")
        self._activate_btn.setChecked(False)
        self._set_status("Inactive")
        self._cleanup_draw_layer()

    def _set_status(self, msg: str, error: bool = False):
        self._status.setText(msg)
        colour = "red" if error else "palette(mid)"
        self._status.setStyleSheet(f"color: {colour}; font-style: italic;")

    # ── track state ───────────────────────────────────────────────────────

    def _reset_track_state(self):
        self._track_mode = False
        self._track_pending = None
        self._track_first = None
        self._interp_frame = None

    def _track_status(self):
        return f"[TRACK] Active on '{self._layer.name}'"

    # ── draw layer ────────────────────────────────────────────────────────

    def _get_draw_layer(self):
        if _DRAW_LAYER in self.viewer.layers:
            return self.viewer.layers[_DRAW_LAYER]
        dl = self.viewer.add_shapes(
            name=_DRAW_LAYER,
            edge_color="yellow",
            edge_width=1,
            face_color="transparent",
        )
        dl.visible = False
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

        def key_t(_layer):
            self._track_mode = not self._track_mode
            self._track_pending = None
            self._track_first = None
            self._interp_frame = None
            if self._track_mode:
                self._set_status(self._track_status())
            else:
                self._set_status(f"Active on '{layer.name}'")

        def key_r(_layer):
            for lyr in self.viewer.layers:
                if isinstance(lyr, napari.layers.Tracks):
                    lyr.visible = not lyr.visible
                    return

        def key_l(_layer):
            try:
                if _layer.color_mode == "auto":
                    _layer.color_mode = "direct"
                else:
                    _layer.color_mode = "auto"
            except Exception as exc:
                show_error(f"key_l error: {exc}")

        for key, fn in [
            ("n",         key_n),
            ("Shift-n",   key_shift_n),
            ("w",         key_w),
            ("Control-z", key_undo),
            ("t",         key_t),
            ("r",         key_r),
            ("l",         key_l),
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

                # ── track-edit mode ───────────────────────────────────────
                if self._track_mode:
                    seg = _layer.data              # full (T,H,W) stack
                    pos = _layer.world_to_data(event.position)
                    frame_lab = _track_label_at(seg[t], pos)

                    # delete track: Ctrl+Alt+Right-click
                    if mods == {"Control", "Alt"} and btn == 2:
                        if frame_lab > 0:
                            delete_track(seg, frame_lab, t)
                            _layer.refresh()
                            self._set_status(
                                f"Deleted track {frame_lab} from frame {t} — "
                                + self._track_status()
                            )
                        return

                    # split track: Shift+Right-click
                    if mods == {"Shift"} and btn == 2:
                        if frame_lab > 0:
                            new_lab = split_track(seg, frame_lab, t)
                            _layer.refresh()
                            self._set_status(
                                f"Split track {frame_lab} → {new_lab} at frame {t} — "
                                + self._track_status()
                                if new_lab else "Split failed — track not present here"
                            )
                        return

                    # swap two tracks: Shift+Left-drag
                    if mods == {"Shift"} and btn == 1:
                        track_a = frame_lab
                        yield
                        while event.type == "mouse_move":
                            yield
                        pos_end = _layer.world_to_data(event.position)
                        track_b = _track_label_at(seg[t], pos_end)
                        if track_a > 0 and track_b > 0 and track_a != track_b:
                            swap_tracks(seg, track_a, track_b, t)
                            _layer.refresh()
                            self._set_status(
                                f"Swapped tracks {track_a} ↔ {track_b} from frame {t} — "
                                + self._track_status()
                            )
                        else:
                            self._set_status("Swap failed — " + self._track_status())
                        return

                    # manual re-link: Ctrl+Left-click (pick source)
                    if mods == {"Control"} and btn == 1:
                        self._track_first = frame_lab
                        self._track_pending = "manual"
                        self._set_status(
                            f"Manual re-link: source = {frame_lab} (frame {t})  "
                            "— Ctrl+Right-click destination cell"
                        )
                        return

                    # manual re-link: Ctrl+Right-click (pick destination)
                    if mods == {"Control"} and btn == 2 and self._track_pending == "manual":
                        src = self._track_first
                        if src is not None and src > 0 and frame_lab > 0:
                            ok = reassign_cell(seg, t, frame_lab, src)
                            _layer.refresh()
                            self._set_status(
                                f"Re-linked {frame_lab} → {src} from frame {t} — "
                                + self._track_status()
                                if ok else "Re-link failed — " + self._track_status()
                            )
                        self._track_pending = None
                        self._track_first = None
                        return

                    # interpolate: Alt+Left-click (mark start)
                    if mods == {"Alt"} and btn == 1:
                        self._track_first = frame_lab
                        self._interp_frame = t
                        self._track_pending = "interpolate"
                        self._set_status(
                            f"Interpolate: start = track {frame_lab} frame {t}  "
                            "— Alt+Right-click end frame"
                        )
                        return

                    # interpolate: Alt+Right-click (mark end, execute)
                    if mods == {"Alt"} and btn == 2 and self._track_pending == "interpolate":
                        track_id = self._track_first
                        f_start  = self._interp_frame
                        if track_id is not None and f_start is not None and t != f_start:
                            f0, f1 = (f_start, t) if t > f_start else (t, f_start)
                            ok = interpolate_track(seg, track_id, f0, f1)
                            _layer.refresh()
                            self._set_status(
                                f"Interpolated track {track_id} frames {f0}–{f1} — "
                                + self._track_status()
                                if ok else "Interpolation failed — " + self._track_status()
                            )
                        self._track_pending = None
                        self._track_first = None
                        self._interp_frame = None
                        return

                    # merge: Left-click (first track)
                    if not mods and btn == 1:
                        if frame_lab > 0:
                            self._track_first = frame_lab
                            self._track_pending = "merge"
                            self._set_status(
                                f"Merge: selected track {frame_lab} — Right-click second track"
                            )
                        return

                    # merge: Right-click (second track, execute)
                    if not mods and btn == 2 and self._track_pending == "merge":
                        track_a = self._track_first
                        track_b = frame_lab
                        if track_a is not None and track_a > 0 and track_b > 0 and track_a != track_b:
                            merge_tracks(seg, track_a, track_b, t)
                            _layer.refresh()
                            self._set_status(
                                f"Merged track {track_b} → {track_a} from frame {t} — "
                                + self._track_status()
                            )
                        else:
                            self._set_status("Merge failed — " + self._track_status())
                        self._track_pending = None
                        self._track_first = None
                        return

                    return

                # ── label-edit mode (default) ─────────────────────────────
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

                # split draw: Alt+Right-drag
                if mods == {"Alt"} and btn == 2:
                    dl = self._get_draw_layer()
                    dl.data = []
                    dl.visible = True
                    pos = [_layer.world_to_data(event.position)]
                    yield
                    while event.type == "mouse_move":
                        pos.append(_layer.world_to_data(event.position))
                        if len(pos) % 3 == 0:
                            dl.data = [np.array(pos)]
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

                # redraw junction: Ctrl+Alt+Left-drag
                if mods == {"Control", "Alt"} and btn == 1:
                    dl = self._get_draw_layer()
                    dl.data = []
                    dl.visible = True
                    pos = [_layer.world_to_data(event.position)]
                    yield
                    while event.type == "mouse_move":
                        pos.append(_layer.world_to_data(event.position))
                        if len(pos) % 3 == 0:
                            dl.data = [np.array(pos)]
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
        for lyr in self.viewer.layers:
            if isinstance(lyr, napari.layers.Image):
                d = lyr.data
                return d[t] if d.ndim == 3 else d if d.ndim == 2 else None
        return None
