"""Nuclear segmentation tab for CellFlow.

Two sections:

  Section A — Cellpose Nuclear Segmentation
      Runs Cellpose (nuclei model, or custom) on the selected nuclear channel
      image.  Supports Z-projection (→ 2-D labels), stitch-z-slices, and full
      3-D volumetric mode.  Output is written to the Nuclear Labels layer and
      registered in shared state.

  Section B — Flatten & Post-process
      Converts a 3-D (T, Z, H, W) nuclear label stack to a 2-D (T, H, W) one
      that the standard Correction and Tracking widgets can operate on, and
      that guided segmentation can use as watershed seeds.
      Options: projection method, hole fill, min size, split touching nuclei.

After this tab: correct the nuclear layer in the Correction tab, then track it
in the Tracking tab, then run Guided Segmentation in the Cell Seg tab.
"""

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout,
    QScrollArea, QToolButton,
    QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QLabel, QLineEdit, QFileDialog,
    QTextEdit, QProgressBar, QGroupBox,
)
from napari.qt.threading import thread_worker
import napari

from ..backend.segmentation import make_cp_model, run_cp, flatten_nuclear_labels
from .registry import get_state

_LAYER_NAME = "Nuclear Labels"

_SEG_DEFAULTS = {
    "model_type":         "nuclei",
    "diameter":           15.0,
    "auto_diameter":      True,
    "flow_threshold":     0.4,
    "cellprob_threshold": 0.0,
    "min_size":           15,
    "gpu":                True,
    "mode":               "Z-projection",
    "stitch_threshold":   0.1,
}

_FLAT_DEFAULTS = {
    "method":            "max",
    "hole_fill_radius":  0,
    "min_size":          0,
    "split_touching":    False,
}


# ── helpers ────────────────────────────────────────────────────────────

def _dspin(lo, hi, default, decimals=2, step=0.1):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(default)
    return s


def _sep(title):
    lbl = QLabel(f"<b>{title}</b>")
    lbl.setStyleSheet("color: palette(text); margin-top: 4px;")
    return lbl


# ── main widget ────────────────────────────────────────────────────────

class NuclearSegWidget(QWidget):
    """Nuclear segmentation + flatten/post-process tab."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer        = viewer
        self._state        = get_state(viewer)
        self._worker       = None
        self._is_running   = False
        self._cp_model     = None
        self._cp_model_key = None
        self._setup_ui()

    # ── UI ─────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── Section A: Cellpose Nuclear Segmentation ───────────────────
        seg_box = QGroupBox("A — Cellpose Nuclear Segmentation")
        seg_lay = QVBoxLayout(seg_box)
        seg_lay.setSpacing(6)

        # Layer picker
        picker_form = QFormLayout()
        picker_form.setSpacing(4)
        self._layer_combo = QComboBox()
        self._layer_combo.setToolTip(
            "Nuclear channel image layer.\n"
            "Expected shape: (T, Z, H, W) or (T, H, W)."
        )
        picker_form.addRow("Nuclear channel:", self._layer_combo)
        picker_w = QWidget()
        picker_w.setLayout(picker_form)
        seg_lay.addWidget(picker_w)

        # Cellpose parameters (collapsible)
        seg_toggle = QToolButton()
        seg_toggle.setText("Cellpose Parameters")
        seg_toggle.setArrowType(Qt.RightArrow)
        seg_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        seg_toggle.setCheckable(True)
        seg_toggle.setStyleSheet("QToolButton { font-weight: bold; }")
        seg_lay.addWidget(seg_toggle)

        params_inner = QWidget()
        self._params_form = QFormLayout(params_inner)
        self._params_form.setSpacing(4)
        self._build_seg_params()
        seg_scroll = QScrollArea()
        seg_scroll.setWidget(params_inner)
        seg_scroll.setWidgetResizable(True)
        seg_scroll.setFixedHeight(280)
        seg_scroll.setVisible(False)
        seg_lay.addWidget(seg_scroll)

        def _toggle_seg(checked):
            seg_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
            seg_scroll.setVisible(checked)
        seg_toggle.toggled.connect(_toggle_seg)

        # Segment buttons
        btns = QHBoxLayout()
        self._seg_frame_btn = QPushButton("Segment Frame")
        self._seg_stack_btn = QPushButton("Segment Stack")
        self._seg_frame_btn.clicked.connect(self._on_segment_frame)
        self._seg_stack_btn.clicked.connect(self._on_segment_stack)
        btns.addWidget(self._seg_frame_btn)
        btns.addWidget(self._seg_stack_btn)
        seg_lay.addLayout(btns)

        self._seg_progress = QProgressBar()
        self._seg_progress.setRange(0, 0)
        self._seg_progress.setVisible(False)
        seg_lay.addWidget(self._seg_progress)
        self._seg_cancel_btn = QPushButton("Cancel")
        self._seg_cancel_btn.setVisible(False)
        self._seg_cancel_btn.clicked.connect(self._on_cancel)
        seg_lay.addWidget(self._seg_cancel_btn)

        self._seg_log = QTextEdit()
        self._seg_log.setReadOnly(True)
        self._seg_log.setFixedHeight(90)
        self._seg_log.setPlaceholderText("Segmentation log…")
        seg_lay.addWidget(self._seg_log)

        root.addWidget(seg_box)

        # ── Section B: Flatten & Post-process ──────────────────────────
        flat_box = QGroupBox("B — Flatten & Post-process")
        flat_lay = QVBoxLayout(flat_box)
        flat_lay.setSpacing(6)

        flat_hint = QLabel(
            "Convert a 3-D (T, Z, H, W) nuclear label stack to 2-D (T, H, W) "
            "for use in Correction, Tracking, and Guided Segmentation. "
            "Skip if you used Z-projection mode above."
        )
        flat_hint.setWordWrap(True)
        flat_hint.setStyleSheet("color: palette(mid); font-size: 8pt;")
        flat_lay.addWidget(flat_hint)

        flat_picker_form = QFormLayout()
        flat_picker_form.setSpacing(4)
        self._flat_layer_combo = QComboBox()
        self._flat_layer_combo.setToolTip(
            "Nuclear Labels layer to flatten. Defaults to the output of Section A."
        )
        flat_picker_form.addRow("Source layer:", self._flat_layer_combo)
        flat_picker_w = QWidget()
        flat_picker_w.setLayout(flat_picker_form)
        flat_lay.addWidget(flat_picker_w)

        flat_form = QFormLayout()
        flat_form.setSpacing(4)

        self._flat_method = QComboBox()
        self._flat_method.addItems(["max", "mean", "sum"])
        self._flat_method.setCurrentText(_FLAT_DEFAULTS["method"])
        flat_form.addRow("Projection method:", self._flat_method)

        self._flat_hole_fill = QSpinBox()
        self._flat_hole_fill.setRange(0, 100)
        self._flat_hole_fill.setValue(_FLAT_DEFAULTS["hole_fill_radius"])
        self._flat_hole_fill.setToolTip(
            "Fill holes inside labeled regions up to this radius (px). 0 = disabled."
        )
        flat_form.addRow("Hole fill radius (px):", self._flat_hole_fill)

        self._flat_min_size = QSpinBox()
        self._flat_min_size.setRange(0, 100000)
        self._flat_min_size.setValue(_FLAT_DEFAULTS["min_size"])
        self._flat_min_size.setToolTip(
            "Remove labeled regions smaller than this area (px²). 0 = disabled."
        )
        flat_form.addRow("Min size (px²):", self._flat_min_size)

        self._flat_split = QCheckBox()
        self._flat_split.setChecked(_FLAT_DEFAULTS["split_touching"])
        self._flat_split.setToolTip(
            "Run watershed on the distance map to split merged/touching nuclei."
        )
        flat_form.addRow("Split touching:", self._flat_split)

        flat_form_w = QWidget()
        flat_form_w.setLayout(flat_form)
        flat_lay.addWidget(flat_form_w)

        self._flat_btn = QPushButton("Flatten")
        self._flat_btn.clicked.connect(self._on_flatten)
        flat_lay.addWidget(self._flat_btn)

        self._flat_progress = QProgressBar()
        self._flat_progress.setRange(0, 0)
        self._flat_progress.setVisible(False)
        flat_lay.addWidget(self._flat_progress)

        self._flat_log = QTextEdit()
        self._flat_log.setReadOnly(True)
        self._flat_log.setFixedHeight(60)
        self._flat_log.setPlaceholderText("Flatten log…")
        flat_lay.addWidget(self._flat_log)

        root.addWidget(flat_box)

        # ── Workflow hint ───────────────────────────────────────────────
        hint = QLabel(
            "<b>Next:</b> correct the <i>Nuclear Labels</i> layer in the "
            "<b>Correction</b> tab, then track it in the <b>Tracking</b> tab "
            "(select Nuclear Labels as the target), then run "
            "<b>Guided Segmentation</b> in the Cell Seg tab."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid); font-size: 8pt; margin-top: 4px;")
        root.addWidget(hint)

        root.addStretch()

        # Layer list listeners
        self._refresh_image_layers()
        self._refresh_labels_layers()
        self.viewer.layers.events.inserted.connect(self._refresh_image_layers)
        self.viewer.layers.events.inserted.connect(self._refresh_labels_layers)
        self.viewer.layers.events.removed.connect(self._refresh_image_layers)
        self.viewer.layers.events.removed.connect(self._refresh_labels_layers)

    def _build_seg_params(self):
        add = self._params_form.addRow

        add(_sep("Model"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["nuclei", "cpsam", "custom"])
        self._model_combo.setCurrentText(_SEG_DEFAULTS["model_type"])
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        add("Model:", self._model_combo)

        custom_row = QHBoxLayout()
        self._custom_path_edit = QLineEdit()
        self._custom_path_edit.setPlaceholderText("path/to/model.pt")
        self._custom_path_edit.setEnabled(False)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(60)
        browse_btn.setEnabled(False)
        browse_btn.clicked.connect(self._browse_custom)
        custom_row.addWidget(self._custom_path_edit)
        custom_row.addWidget(browse_btn)
        self._custom_browse_btn = browse_btn
        add("Custom model:", custom_row)

        add(_sep("Cell detection"))
        self._auto_diam = QCheckBox("Auto")
        self._auto_diam.setChecked(_SEG_DEFAULTS["auto_diameter"])
        self._auto_diam.stateChanged.connect(
            lambda s: self._p_diam.setEnabled(not s)
        )
        add("Auto diameter:", self._auto_diam)

        self._p_diam = _dspin(1, 1000, _SEG_DEFAULTS["diameter"], decimals=1, step=1.0)
        self._p_diam.setEnabled(not _SEG_DEFAULTS["auto_diameter"])
        add("Diameter (px):", self._p_diam)

        self._p_flow = _dspin(0, 10, _SEG_DEFAULTS["flow_threshold"])
        add("Flow threshold:", self._p_flow)

        self._p_cellprob = _dspin(-6, 6, _SEG_DEFAULTS["cellprob_threshold"])
        add("Cellprob threshold:", self._p_cellprob)

        self._p_minsize = QSpinBox()
        self._p_minsize.setRange(0, 100000)
        self._p_minsize.setValue(_SEG_DEFAULTS["min_size"])
        add("Min size (px):", self._p_minsize)

        add(_sep("Z handling"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Z-projection", "Stitch z-slices", "Full 3-D"])
        self._mode_combo.setCurrentText(_SEG_DEFAULTS["mode"])
        self._mode_combo.setToolTip(
            "Z-projection (recommended): max-project each z-stack to 2-D, then\n"
            "segment. Output is (T, H, W) — ready for Correction and Tracking.\n\n"
            "Stitch z-slices: segment per z-slice, merge overlapping masks across z.\n\n"
            "Full 3-D: volumetric Cellpose — use Flatten (Section B) afterwards."
        )
        self._mode_combo.currentTextChanged.connect(
            lambda t: self._p_stitch.setEnabled(t == "Stitch z-slices")
        )
        add("Mode:", self._mode_combo)

        self._p_stitch = _dspin(0, 1, _SEG_DEFAULTS["stitch_threshold"])
        self._p_stitch.setToolTip("IoU threshold for stitching masks across z-slices.")
        self._p_stitch.setEnabled(False)
        add("Stitch threshold:", self._p_stitch)

        add(_sep("Hardware"))
        self._p_gpu = QCheckBox()
        self._p_gpu.setChecked(_SEG_DEFAULTS["gpu"])
        add("Use GPU:", self._p_gpu)

    # ── Layer list helpers ─────────────────────────────────────────────

    def _refresh_image_layers(self, *_):
        current = self._layer_combo.currentText()
        self._layer_combo.blockSignals(True)
        self._layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                self._layer_combo.addItem(layer.name)
        # prefer the nuclear image layer stored in state
        nuc_name = self._state.tissue.image2_layer
        idx = self._layer_combo.findText(nuc_name or current)
        if idx >= 0:
            self._layer_combo.setCurrentIndex(idx)
        self._layer_combo.blockSignals(False)

    def _refresh_labels_layers(self, *_):
        current = self._flat_layer_combo.currentText()
        self._flat_layer_combo.blockSignals(True)
        self._flat_layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                self._flat_layer_combo.addItem(layer.name)
        # prefer Nuclear Labels
        idx = self._flat_layer_combo.findText(_LAYER_NAME)
        if idx < 0:
            idx = self._flat_layer_combo.findText(current)
        if idx >= 0:
            self._flat_layer_combo.setCurrentIndex(idx)
        self._flat_layer_combo.blockSignals(False)

    # ── Helpers ────────────────────────────────────────────────────────

    def _on_model_changed(self, text):
        is_custom = text == "custom"
        self._custom_path_edit.setEnabled(is_custom)
        self._custom_browse_btn.setEnabled(is_custom)

    def _browse_custom(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select model", "", "Model (*.pt *.pth *)")
        if path:
            self._custom_path_edit.setText(path)

    def _seg_log_append(self, msg):
        self._seg_log.append(msg)
        self._seg_log.verticalScrollBar().setValue(
            self._seg_log.verticalScrollBar().maximum()
        )

    def _flat_log_append(self, msg):
        self._flat_log.append(msg)
        self._flat_log.verticalScrollBar().setValue(
            self._flat_log.verticalScrollBar().maximum()
        )

    def _set_seg_running(self, running):
        self._is_running = running
        self._seg_frame_btn.setEnabled(not running)
        self._seg_stack_btn.setEnabled(not running)
        self._seg_progress.setVisible(running)
        self._seg_cancel_btn.setVisible(running)

    def _get_seg_params(self):
        model_type  = self._model_combo.currentText()
        custom_path = self._custom_path_edit.text() if model_type == "custom" else None
        diameter    = None if self._auto_diam.isChecked() else float(self._p_diam.value())
        return dict(
            model_type  = model_type,
            custom_path = custom_path,
            diameter    = diameter,
            flow_thresh = float(self._p_flow.value()),
            cellprob    = float(self._p_cellprob.value()),
            min_size    = int(self._p_minsize.value()),
            gpu         = self._p_gpu.isChecked(),
            mode        = self._mode_combo.currentText(),
            stitch      = float(self._p_stitch.value()),
        )

    def _segment_frame(self, frame_data, params, model):
        mode = params["mode"]
        img  = np.asarray(frame_data)
        cp_kwargs = dict(
            diameter           = params["diameter"] or 0,
            flow_threshold     = params["flow_thresh"],
            cellprob_threshold = params["cellprob"],
            min_size           = params["min_size"],
        )
        if mode == "Z-projection":
            if img.ndim == 3:
                img = img.max(axis=0)
            masks = run_cp(img, model, **cp_kwargs)
            return np.asarray(masks, dtype=np.uint16)
        elif mode == "Stitch z-slices":
            if img.ndim == 2:
                img = img[np.newaxis]
            masks = run_cp(img, model, **cp_kwargs, stitch_threshold=params["stitch"])
            return np.asarray(masks, dtype=np.uint16)
        else:  # Full 3-D
            if img.ndim == 2:
                img = img[np.newaxis]
            masks = run_cp(img, model, **cp_kwargs, do_3D=True)
            return np.asarray(masks, dtype=np.uint16)

    def _src_scale(self, output_ndim):
        name = self._layer_combo.currentText()
        if name and name in self.viewer.layers:
            s = tuple(self.viewer.layers[name].scale)
            return s[-output_ndim:]
        return None

    def _ensure_nuclear_layer(self, shape):
        scale = self._src_scale(len(shape))
        if _LAYER_NAME in self.viewer.layers:
            existing = np.asarray(self.viewer.layers[_LAYER_NAME].data)
            if existing.shape == shape:
                return
            self.viewer.layers.remove(_LAYER_NAME)
        data = np.zeros(shape, dtype=np.uint16)
        kw = {"scale": scale} if scale is not None else {}
        self.viewer.add_labels(data, name=_LAYER_NAME, **kw)

    def _publish_nuclear_layer(self):
        """Register the Nuclear Labels layer in shared state."""
        if _LAYER_NAME in self.viewer.layers:
            lyr = self.viewer.layers[_LAYER_NAME]
            self._state.set_tissue_nuclear_labels(np.asarray(lyr.data), _LAYER_NAME)

    # ── Segment Frame ──────────────────────────────────────────────────

    def _on_segment_frame(self):
        if self._is_running:
            return

        layer_name = self._layer_combo.currentText()
        if not layer_name or layer_name not in self.viewer.layers:
            self._seg_log_append("No layer selected.")
            return

        img = np.asarray(self.viewer.layers[layer_name].data)
        if img.ndim not in (3, 4):
            self._seg_log_append(f"Expected (T, H, W) or (T, Z, H, W), got {img.shape}.")
            return

        t = int(self.viewer.dims.current_step[0]) % img.shape[0]
        params    = self._get_seg_params()
        model_key = (params["model_type"], params["custom_path"], params["gpu"])

        self._seg_log.clear()
        self._seg_log_append(f"Segmenting frame {t}  (mode: {params['mode']})…")
        self._set_seg_running(True)

        frame_data = img[t]
        log_msgs   = []

        @thread_worker(connect={
            "returned": lambda r: self._on_frame_done(r, t, img.shape, params["mode"]),
            "errored":  self._on_error,
        })
        def _worker():
            if self._cp_model is None or self._cp_model_key != model_key:
                yield "Loading model…"
                self._cp_model     = make_cp_model(
                    params["model_type"], params["custom_path"], params["gpu"]
                )
                self._cp_model_key = model_key
            yield f"Running cellpose on frame {t}…"
            return self._segment_frame(frame_data, params, self._cp_model)

        worker = _worker()
        worker.yielded.connect(self._seg_log_append)
        self._worker = worker

    def _on_frame_done(self, masks, t, img_shape, mode):
        self._set_seg_running(False)
        T = img_shape[0]

        if masks.ndim == 2:
            H, W = masks.shape
            target_shape = (T, H, W)
        else:
            Z, H, W = masks.shape
            target_shape = (T, Z, H, W)

        self._ensure_nuclear_layer(target_shape)
        self.viewer.layers[_LAYER_NAME].data[t] = masks
        self.viewer.layers[_LAYER_NAME].refresh()
        self._publish_nuclear_layer()
        self._seg_log_append(f"Frame {t}: {int(masks.max())} nuclei detected.")

    # ── Segment Stack ──────────────────────────────────────────────────

    def _on_segment_stack(self):
        if self._is_running:
            return

        layer_name = self._layer_combo.currentText()
        if not layer_name or layer_name not in self.viewer.layers:
            self._seg_log_append("No layer selected.")
            return

        img       = np.asarray(self.viewer.layers[layer_name].data)
        if img.ndim not in (3, 4):
            self._seg_log_append(f"Expected (T, H, W) or (T, Z, H, W), got {img.shape}.")
            return

        params    = self._get_seg_params()
        model_key = (params["model_type"], params["custom_path"], params["gpu"])

        self._seg_log.clear()
        self._seg_log_append(f"Segmenting {img.shape[0]} frame(s)  (mode: {params['mode']})…")
        self._set_seg_running(True)

        @thread_worker(connect={"returned": self._on_stack_done, "errored": self._on_error})
        def _worker():
            if self._cp_model is None or self._cp_model_key != model_key:
                yield "Loading model…"
                self._cp_model     = make_cp_model(
                    params["model_type"], params["custom_path"], params["gpu"]
                )
                self._cp_model_key = model_key

            results = []
            for t in range(img.shape[0]):
                yield f"Frame {t + 1}/{img.shape[0]}…"
                frame_masks = self._segment_frame(img[t], params, self._cp_model)
                yield f"  → frame {t} shape: {frame_masks.shape}"
                results.append(frame_masks)
            return results, params["mode"]

        worker = _worker()
        worker.yielded.connect(self._seg_log_append)
        self._worker = worker

    def _on_stack_done(self, payload):
        self._set_seg_running(False)
        results, mode = payload

        shapes = [r.shape for r in results]
        if len(set(shapes)) > 1:
            self._seg_log_append(f"WARNING: inconsistent frame shapes: {set(shapes)}")
            return

        try:
            stacked = np.stack(results, axis=0)
        except ValueError as e:
            self._seg_log_append(f"Stacking failed: {e}")
            return

        scale = self._src_scale(stacked.ndim)
        if _LAYER_NAME in self.viewer.layers:
            lyr = self.viewer.layers[_LAYER_NAME]
            if lyr.data.shape != stacked.shape:
                self.viewer.layers.remove(_LAYER_NAME)
                kw = {"scale": scale} if scale is not None else {}
                self.viewer.add_labels(stacked, name=_LAYER_NAME, **kw)
            else:
                lyr.data = stacked
                if scale is not None:
                    lyr.scale = scale
        else:
            kw = {"scale": scale} if scale is not None else {}
            self.viewer.add_labels(stacked, name=_LAYER_NAME, **kw)

        self._publish_nuclear_layer()
        total = sum(int(r.max()) for r in results)
        self._seg_log_append(f"Done. {len(results)} frame(s), ~{total} total detections.")

    # ── Flatten ────────────────────────────────────────────────────────

    def _on_flatten(self):
        layer_name = self._flat_layer_combo.currentText()
        if not layer_name or layer_name not in self.viewer.layers:
            self._flat_log_append("No nuclear labels layer selected.")
            return

        data = np.asarray(self.viewer.layers[layer_name].data)
        if data.ndim not in (3, 4):
            self._flat_log_append(f"Expected (T, H, W) or (T, Z, H, W), got {data.shape}.")
            return
        if data.ndim == 3:
            self._flat_log_append("Layer is already 2D (T, H, W) — nothing to flatten.")
            return

        method        = self._flat_method.currentText()
        hole_fill_r   = int(self._flat_hole_fill.value())
        min_size      = int(self._flat_min_size.value())
        split_touching = self._flat_split.isChecked()

        self._flat_log.clear()
        self._flat_log_append(
            f"Flattening {data.shape} → (T, H, W)  method={method}…"
        )
        self._flat_btn.setEnabled(False)
        self._flat_progress.setVisible(True)

        @thread_worker(connect={"returned": self._on_flatten_done, "errored": self._on_flatten_error})
        def _worker():
            return flatten_nuclear_labels(
                data,
                method=method,
                hole_fill_radius=hole_fill_r,
                min_size=min_size,
                split_touching=split_touching,
            )

        _worker()

    def _on_flatten_done(self, flat):
        self._flat_btn.setEnabled(True)
        self._flat_progress.setVisible(False)

        scale = None
        src_name = self._flat_layer_combo.currentText()
        if src_name and src_name in self.viewer.layers:
            s = tuple(self.viewer.layers[src_name].scale)
            scale = s[-flat.ndim:]

        if _LAYER_NAME in self.viewer.layers:
            lyr = self.viewer.layers[_LAYER_NAME]
            if lyr.data.shape != flat.shape:
                self.viewer.layers.remove(_LAYER_NAME)
                kw = {"scale": scale} if scale is not None else {}
                self.viewer.add_labels(flat, name=_LAYER_NAME, **kw)
            else:
                lyr.data = flat
        else:
            kw = {"scale": scale} if scale is not None else {}
            self.viewer.add_labels(flat, name=_LAYER_NAME, **kw)

        self._publish_nuclear_layer()
        T = flat.shape[0]
        total = sum(int(np.unique(flat[t][flat[t] > 0]).shape[0]) for t in range(T))
        self._flat_log_append(f"Done. Shape: {flat.shape}, ~{total} total nuclei across {T} frames.")

    def _on_flatten_error(self, exc):
        self._flat_btn.setEnabled(True)
        self._flat_progress.setVisible(False)
        self._flat_log_append(f"Error: {exc}")

    # ── Error / cancel ─────────────────────────────────────────────────

    def _on_error(self, exc):
        self._set_seg_running(False)
        self._seg_log_append(f"Error: {exc}")

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.quit()
        self._set_seg_running(False)
        self._seg_log_append("Cancelled.")
