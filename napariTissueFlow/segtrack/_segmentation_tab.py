"""
Segmentation tab for napariSegTrack.

Supports two modes:
  - Single Channel : Cellpose on one channel (grayscale)
  - Two Channel    : Cellpose two-channel mode (cell body + nucleus)

Includes "Open in Cellpose GUI / Import from Cellpose GUI" for manual correction
of the current frame via a temp directory.

For Voronoi cell body expansion, use the dedicated "Cell Bodies" tab.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QRadioButton, QButtonGroup,
    QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QLabel, QLineEdit, QFileDialog,
    QTextEdit, QProgressBar, QScrollArea,
)
from napari.qt.threading import thread_worker
import napari


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
    lbl.setStyleSheet("color: palette(mid); margin-top: 4px;")
    return lbl


# ── defaults ───────────────────────────────────────────────────────────

SEG_DEFAULTS = {
    "model_type":         "cyto3",
    "diameter":           30.0,
    "auto_diameter":      False,
    "flow_threshold":     0.4,
    "cellprob_threshold": 0.0,
    "min_size":           15,
    "gpu":                True,
}


# ── main widget ────────────────────────────────────────────────────────

class SegmentationTab(QWidget):
    """Segmentation tab: frame-by-frame and full-stack Cellpose segmentation."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self._worker = None
        self._is_running = False  # explicit guard; avoids is_running race on worker teardown
        self._custom_model_path = None
        self._temp_dir = None
        self._cellpose_proc = None
        self._temp_frame_idx = None

        self._setup_ui()

        viewer.layers.events.inserted.connect(self._refresh_layer_combos)
        viewer.layers.events.removed.connect(self._refresh_layer_combos)

    # ── UI construction ────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Mode selector ──
        mode_box = QGroupBox("Segmentation Mode")
        mode_lay = QHBoxLayout(mode_box)
        self._rb_single = QRadioButton("Single Channel")
        self._rb_two_ch = QRadioButton("Two Channel")
        self._rb_single.setChecked(True)
        self._mode_grp = QButtonGroup(self)
        for rb in (self._rb_single, self._rb_two_ch):
            mode_lay.addWidget(rb)
            self._mode_grp.addButton(rb)
        root.addWidget(mode_box)
        self._mode_grp.buttonClicked.connect(self._on_mode_changed)

        # ── Input panels (one per mode) ──
        self._single_panel = self._build_single_panel()
        self._two_ch_panel = self._build_two_ch_panel()
        root.addWidget(self._single_panel)
        root.addWidget(self._two_ch_panel)
        self._two_ch_panel.setVisible(False)

        # ── Cellpose parameters (scrollable, collapsible) ──
        params_box = QGroupBox("Cellpose Parameters")
        params_box.setCheckable(True)
        params_box.setChecked(False)
        params_inner = QWidget()
        self._params_form = QFormLayout(params_inner)
        self._params_form.setSpacing(4)
        self._build_params()
        scroll = QScrollArea()
        scroll.setWidget(params_inner)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(240)
        pb_lay = QVBoxLayout(params_box)
        pb_lay.addWidget(scroll)
        params_box.toggled.connect(scroll.setVisible)
        scroll.setVisible(False)
        root.addWidget(params_box)

        # ── Segmentation action buttons ──
        seg_btns = QHBoxLayout()
        self._seg_frame_btn = QPushButton("Segment Frame")
        self._seg_stack_btn = QPushButton("Segment Stack")
        self._seg_frame_btn.clicked.connect(self._on_segment_frame)
        self._seg_stack_btn.clicked.connect(self._on_segment_stack)
        seg_btns.addWidget(self._seg_frame_btn)
        seg_btns.addWidget(self._seg_stack_btn)
        root.addLayout(seg_btns)

        # ── Cellpose GUI integration ──
        gui_box = QGroupBox("Cellpose GUI Correction")
        gui_lay = QVBoxLayout(gui_box)
        gui_btns = QHBoxLayout()
        self._open_gui_btn   = QPushButton("Open Frame in Cellpose GUI")
        self._import_gui_btn = QPushButton("Import from Cellpose GUI")
        self._import_gui_btn.setEnabled(False)
        self._open_gui_btn.clicked.connect(self._on_open_in_cellpose_gui)
        self._import_gui_btn.clicked.connect(self._on_import_from_cellpose_gui)
        gui_btns.addWidget(self._open_gui_btn)
        gui_btns.addWidget(self._import_gui_btn)
        gui_lay.addLayout(gui_btns)
        self._gui_status = QLabel("Export the current frame, correct it in Cellpose GUI, then import.")
        self._gui_status.setWordWrap(True)
        self._gui_status.setStyleSheet("color: palette(mid);")
        gui_lay.addWidget(self._gui_status)
        root.addWidget(gui_box)

        # ── Progress + log ──
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(120)
        self._log.setPlaceholderText("Segmentation log…")
        root.addWidget(self._log)

        root.addStretch()
        self._refresh_layer_combos()

    # ── Input panels ───────────────────────────────────────────────────

    def _build_single_panel(self):
        box = QGroupBox("Inputs – Single Channel")
        lay = QFormLayout(box)
        self._single_combo = QComboBox()
        lay.addRow("Image layer:", self._single_combo)
        return box

    def _build_two_ch_panel(self):
        box = QGroupBox("Inputs – Two Channel")
        lay = QFormLayout(box)
        self._primary_combo   = QComboBox()
        self._secondary_combo = QComboBox()
        self._primary_combo.setToolTip("Primary channel (cell body / cytoplasm)")
        self._secondary_combo.setToolTip("Secondary channel (nucleus / helper)")
        lay.addRow("Primary (cell):", self._primary_combo)
        lay.addRow("Secondary (nucleus):", self._secondary_combo)
        return box

    def _build_params(self):
        add = self._params_form.addRow

        # Model
        add(_sep("Model"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["cyto3", "nuclei", "cpsam", "custom"])
        self._model_combo.setCurrentText(SEG_DEFAULTS["model_type"])
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        add("Model:", self._model_combo)

        custom_row = QHBoxLayout()
        self._custom_path_edit = QLineEdit()
        self._custom_path_edit.setPlaceholderText("path/to/model.pt")
        self._custom_path_edit.setEnabled(False)
        self._custom_browse_btn = QPushButton("Browse")
        self._custom_browse_btn.setEnabled(False)
        self._custom_browse_btn.clicked.connect(self._browse_custom_model)
        custom_row.addWidget(self._custom_path_edit)
        custom_row.addWidget(self._custom_browse_btn)
        add("Custom model:", custom_row)

        # Parameters
        add(_sep("Parameters"))
        diam_row = QHBoxLayout()
        self._diam_spin = _dspin(1.0, 2000.0, SEG_DEFAULTS["diameter"], 1, 1.0)
        self._auto_diam_chk = QCheckBox("Auto")
        self._auto_diam_chk.setChecked(SEG_DEFAULTS["auto_diameter"])
        self._auto_diam_chk.toggled.connect(lambda c: self._diam_spin.setEnabled(not c))
        diam_row.addWidget(self._diam_spin)
        diam_row.addWidget(self._auto_diam_chk)
        add("Diameter (px):", diam_row)

        self._flow_spin    = _dspin(-10, 10, SEG_DEFAULTS["flow_threshold"],     2, 0.05)
        self._prob_spin    = _dspin(-10, 10, SEG_DEFAULTS["cellprob_threshold"], 1, 0.5)
        self._minsize_spin = QSpinBox()
        self._minsize_spin.setRange(1, 10000)
        self._minsize_spin.setValue(SEG_DEFAULTS["min_size"])
        add("Flow threshold:",     self._flow_spin)
        add("CellProb threshold:", self._prob_spin)
        add("Min size (px):",      self._minsize_spin)

        # Options
        add(_sep("Options"))
        self._gpu_chk = QCheckBox()
        self._gpu_chk.setChecked(SEG_DEFAULTS["gpu"])
        add("GPU:", self._gpu_chk)

    # ── Mode / model switching ─────────────────────────────────────────

    def _on_mode_changed(self, _=None):
        self._single_panel.setVisible(self._rb_single.isChecked())
        self._two_ch_panel.setVisible(self._rb_two_ch.isChecked())

    def _on_model_changed(self, text):
        is_custom = (text == "custom")
        self._custom_path_edit.setEnabled(is_custom)
        self._custom_browse_btn.setEnabled(is_custom)

    def _browse_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select custom Cellpose model", "", "Model files (*.pt *.pth);;All files (*)"
        )
        if path:
            self._custom_model_path = path
            self._custom_path_edit.setText(path)

    # ── Layer combo refresh ────────────────────────────────────────────

    def _refresh_layer_combos(self, *_):
        img_names = [
            lay.name for lay in self.viewer.layers
            if isinstance(lay, napari.layers.Image)
        ]
        for combo, items in [
            (self._single_combo,    img_names),
            (self._primary_combo,   img_names),
            (self._secondary_combo, img_names),
        ]:
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            idx = combo.findText(prev)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    # ── Parameter collection ───────────────────────────────────────────

    def _collect_params(self):
        return {
            "model_type":         self._model_combo.currentText(),
            "custom_model_path":  self._custom_model_path,
            "diameter":           None if self._auto_diam_chk.isChecked()
                                       else self._diam_spin.value(),
            "flow_threshold":     self._flow_spin.value(),
            "cellprob_threshold": self._prob_spin.value(),
            "min_size":           self._minsize_spin.value(),
            "gpu":                self._gpu_chk.isChecked(),
        }

    # ── Frame helpers ──────────────────────────────────────────────────

    def _current_frame_idx(self, data):
        if data.ndim == 2:
            return 0
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) > 0 else 0

    def _get_frame(self, data, t):
        return data if data.ndim == 2 else data[t]

    def _get_or_create_labels_layer(self, shape, name="Segmentation"):
        for lay in self.viewer.layers:
            if isinstance(lay, napari.layers.Labels) and lay.name == name:
                return lay
        return self.viewer.add_labels(np.zeros(shape, dtype=np.uint32), name=name)

    def _current_mode(self):
        return "single" if self._rb_single.isChecked() else "two_ch"

    def _reference_stack_shape(self):
        name = (self._single_combo.currentText() if self._rb_single.isChecked()
                else self._primary_combo.currentText())
        if not name:
            return (512, 512)
        return np.asarray(self.viewer.layers[name].data).shape

    # ── Segment Frame ──────────────────────────────────────────────────

    def _on_segment_frame(self):
        if self._is_running:
            self._log_append("Already processing.")
            return

        try:
            imgs, t = self._get_inputs_for_frame()
        except ValueError as e:
            self._log_append(f"ERROR: {e}")
            return

        params = self._collect_params()
        mode   = self._current_mode()

        self._is_running = True
        self._seg_frame_btn.setEnabled(False)
        self._seg_stack_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._log.clear()

        @thread_worker(connect={
            "yielded":  lambda m: self._log_append(m),
            "returned": lambda r: self._on_frame_done(r, t),
            "errored":  self._on_error,
        })
        def _work():
            yield f"Segmenting frame {t}…"
            masks = _run_segmentation(imgs, mode, params)
            n = len(np.unique(masks[masks > 0]))
            yield f"  {n} object(s) detected"
            return masks

        self._worker = _work()

    def _get_inputs_for_frame(self):
        mode = self._current_mode()
        if mode == "single":
            name = self._single_combo.currentText()
            if not name:
                raise ValueError("Select an image layer.")
            data = np.asarray(self.viewer.layers[name].data)
            t = self._current_frame_idx(data)
            return {"img": self._get_frame(data, t)}, t
        else:  # two_ch
            p_name = self._primary_combo.currentText()
            s_name = self._secondary_combo.currentText()
            if not p_name or not s_name:
                raise ValueError("Select both primary and secondary layers.")
            if p_name == s_name:
                raise ValueError("Primary and secondary layers must differ.")
            p_data = np.asarray(self.viewer.layers[p_name].data)
            s_data = np.asarray(self.viewer.layers[s_name].data)
            t = self._current_frame_idx(p_data)
            return {
                "primary":   self._get_frame(p_data, t),
                "secondary": self._get_frame(s_data, t),
            }, t

    def _on_frame_done(self, masks, t):
        self._is_running = False
        self._seg_frame_btn.setEnabled(True)
        self._seg_stack_btn.setEnabled(True)
        self._progress.setVisible(False)

        shape = self._reference_stack_shape()
        layer = self._get_or_create_labels_layer(shape)
        if layer.data.ndim == 2:
            layer.data = masks.astype(np.uint32)
        else:
            data = layer.data.copy()
            data[t] = masks.astype(np.uint32)
            layer.data = data
        layer.refresh()
        self._log_append("Frame segmentation complete.")

    # ── Segment Stack ──────────────────────────────────────────────────

    def _on_segment_stack(self):
        if self._is_running:
            self._log_append("Already processing.")
            return

        try:
            all_frames = self._get_all_frames()
        except ValueError as e:
            self._log_append(f"ERROR: {e}")
            return

        params   = self._collect_params()
        mode     = self._current_mode()
        n_frames = len(all_frames)

        self._is_running = True
        self._seg_frame_btn.setEnabled(False)
        self._seg_stack_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._log.clear()

        @thread_worker(connect={
            "yielded":  lambda m: self._log_append(m),
            "returned": self._on_stack_done,
            "errored":  self._on_error,
        })
        def _work():
            results = []
            for i, imgs in enumerate(all_frames):
                yield f"Frame {i+1}/{n_frames}…"
                results.append(_run_segmentation(imgs, mode, params).astype(np.uint32))
            return results

        self._worker = _work()

    def _get_all_frames(self):
        mode = self._current_mode()
        if mode == "single":
            name = self._single_combo.currentText()
            if not name:
                raise ValueError("Select an image layer.")
            data = np.asarray(self.viewer.layers[name].data)
            if data.ndim == 2:
                return [{"img": data}]
            return [{"img": data[t]} for t in range(data.shape[0])]
        else:  # two_ch
            p_name = self._primary_combo.currentText()
            s_name = self._secondary_combo.currentText()
            if not p_name or not s_name:
                raise ValueError("Select both primary and secondary layers.")
            if p_name == s_name:
                raise ValueError("Primary and secondary layers must differ.")
            p_data = np.asarray(self.viewer.layers[p_name].data)
            s_data = np.asarray(self.viewer.layers[s_name].data)
            if p_data.ndim == 2:
                return [{"primary": p_data, "secondary": s_data}]
            return [
                {"primary": p_data[t], "secondary": s_data[t]}
                for t in range(p_data.shape[0])
            ]

    def _on_stack_done(self, results):
        self._is_running = False
        self._seg_frame_btn.setEnabled(True)
        self._seg_stack_btn.setEnabled(True)
        self._progress.setVisible(False)

        stack = np.stack(results, axis=0)
        for lay in list(self.viewer.layers):
            if isinstance(lay, napari.layers.Labels) and lay.name == "Segmentation":
                if lay.data.shape != stack.shape:
                    self.viewer.layers.remove(lay)
                break
        layer = self._get_or_create_labels_layer(stack.shape)
        layer.data = stack
        layer.refresh()
        self._log_append(f"Stack segmentation complete: {stack.shape[0]} frame(s).")

    # ── Cellpose GUI integration ───────────────────────────────────────

    def _on_open_in_cellpose_gui(self):
        try:
            imgs, t = self._get_inputs_for_frame()
        except ValueError as e:
            self._log_append(f"ERROR: {e}")
            return

        # Guard: if Cellpose is open for a *different* frame, block until the user
        # imports that frame first. Without this guard, file writes below would
        # update _temp_frame_idx while Cellpose still holds the old image, causing
        # the old frame's corrections to be imported into the wrong slot.
        if self._cellpose_proc is not None and self._cellpose_proc.poll() is None:
            if t == self._temp_frame_idx:
                self._log_append(f"Cellpose GUI is already open with frame {t}.")
            else:
                self._log_append(
                    f"Cellpose GUI is open for frame {self._temp_frame_idx}. "
                    f"Import that frame first before exporting frame {t}."
                )
            return

        if self._temp_dir is None or not os.path.isdir(self._temp_dir):
            self._temp_dir = tempfile.mkdtemp(prefix="napariTissueFlow.segtrack_cellpose_")
        self._temp_frame_idx = t

        # Use frame-indexed filenames so multiple exports never overwrite each other.
        # The frame index is also embedded in the seg file so import does not rely
        # solely on the in-memory variable (which is lost on restart).
        img_path = Path(self._temp_dir) / f"frame_{t}.tif"
        seg_path = Path(self._temp_dir) / f"frame_{t}_seg.npy"

        display_img = imgs.get("img") if imgs.get("img") is not None else imgs.get("primary")

        from tifffile import imwrite
        imwrite(str(img_path), display_img.astype(np.uint16))

        existing_masks = self._get_existing_masks(t)
        if existing_masks is not None and existing_masks.max() > 0:
            try:
                from cellpose.utils import masks_to_outlines
                outlines = masks_to_outlines(existing_masks)
            except Exception:
                outlines = np.zeros_like(existing_masks, dtype=bool)

            n_cells = int(existing_masks.max())
            seg_data = {
                "img":              display_img.astype(np.uint16),
                "masks":            existing_masks.astype(np.uint16),
                "outlines":         outlines,
                "colors":           np.random.randint(0, 255, (n_cells + 1, 3), dtype=np.uint8),
                "filename":         str(img_path),
                "flows":            [],
                "chan_choose":       [0, 0],
                "ismanual":         np.zeros(n_cells + 1, dtype=bool),
                "diameter":         self._diam_spin.value(),
                "napari_frame_idx": int(t),  # ground-truth frame index for safe import
            }
            np.save(str(seg_path), seg_data)
            self._log_append(f"Exported frame {t} with {n_cells} mask(s) to Cellpose GUI.")
        else:
            if seg_path.exists():
                seg_path.unlink()
            self._log_append(f"Exported frame {t} (no prior segmentation) to Cellpose GUI.")

        launcher = Path(self._temp_dir) / "launch_cellpose.py"
        launcher.write_text(
            f"from cellpose.gui.gui import run\n"
            f"run(image={str(img_path)!r})\n"
        )
        try:
            self._cellpose_proc = subprocess.Popen(
                [sys.executable, str(launcher)],
                start_new_session=True,
            )
        except Exception as exc:
            self._log_append(f"ERROR launching Cellpose GUI: {exc}")
            return

        self._gui_status.setText(
            f"Frame {t} opened in Cellpose GUI.\n"
            "Correct and save (File → Save), then click 'Import from Cellpose GUI'."
        )
        self._import_gui_btn.setEnabled(True)

    def _on_import_from_cellpose_gui(self):
        if self._temp_dir is None or self._temp_frame_idx is None:
            self._log_append("No Cellpose GUI session found. Open a frame first.")
            return

        seg_path = Path(self._temp_dir) / f"frame_{self._temp_frame_idx}_seg.npy"
        if not seg_path.exists():
            self._log_append(
                f"No seg file found at {seg_path}.\n"
                "Save the segmentation in Cellpose GUI first (File → Save)."
            )
            return

        try:
            seg_data = np.load(str(seg_path), allow_pickle=True).item()
            masks = np.asarray(seg_data["masks"]).astype(np.uint32)
        except Exception as exc:
            self._log_append(f"ERROR reading seg file: {exc}")
            return

        # Trust the frame index embedded in the file; fall back to the in-memory
        # variable only if the file was saved by an older version without the field.
        t = int(seg_data.get("napari_frame_idx", self._temp_frame_idx))
        shape = self._reference_stack_shape()
        layer = self._get_or_create_labels_layer(shape)

        if layer.data.ndim == 2:
            layer.data = masks
        else:
            data = layer.data.copy()
            if t < data.shape[0]:
                data[t] = masks
            layer.data = data
        layer.refresh()

        n = len(np.unique(masks[masks > 0]))
        self._log_append(f"Imported {n} mask(s) into frame {t} from Cellpose GUI.")
        self._gui_status.setText(f"Imported {n} mask(s) for frame {t}.")

    def _get_existing_masks(self, t):
        for lay in self.viewer.layers:
            if isinstance(lay, napari.layers.Labels) and lay.name == "Segmentation":
                data = lay.data
                if data.ndim == 2:
                    return data
                if t < data.shape[0]:
                    return data[t]
        return None

    # ── Error handling ─────────────────────────────────────────────────

    def _on_error(self, exc):
        self._is_running = False
        self._seg_frame_btn.setEnabled(True)
        self._seg_stack_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._log_append(f"ERROR: {exc}")
        import traceback
        self._log_append(traceback.format_exc())

    def _log_append(self, msg):
        self._log.append(str(msg))
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())


# ── Segmentation dispatcher (called in background thread) ──────────────

def _run_segmentation(imgs, mode, params):
    """
    Run Cellpose segmentation for a single frame.

    imgs   : dict from _get_inputs_for_frame / _get_all_frames
    mode   : "single" | "two_ch"
    params : dict from _collect_params()

    Returns (H, W) integer mask array.
    """
    from napariTissueFlow.segtrack._pipeline import make_cp_model, run_cp, run_cp_two_channel

    model = make_cp_model(
        params["model_type"],
        custom_model_path=params.get("custom_model_path"),
        gpu=params["gpu"],
    )
    cp_kwargs = dict(
        model              = model,
        diameter           = params["diameter"],
        flow_threshold     = params["flow_threshold"],
        cellprob_threshold = params["cellprob_threshold"],
        min_size           = params["min_size"],
    )

    if mode == "single":
        return run_cp(imgs["img"], **cp_kwargs)
    else:  # two_ch
        return run_cp_two_channel(imgs["primary"], imgs["secondary"], **cp_kwargs)
