"""
Segmentation tab for napariSegTrack.

Runs Cellpose on the primary image channel. If a secondary channel (Image 2) is
set in the Project Panel, two-channel mode is used automatically.

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
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox,
    QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QLabel, QLineEdit, QFileDialog, QToolButton,
    QTextEdit, QProgressBar, QScrollArea,
)
from napari.qt.threading import thread_worker
import napari

from .registry import get_state


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


# ── defaults ───────────────────────────────────────────────────────────

SEG_DEFAULTS = {
    "model_type":         "cpsam",
    "diameter":           30.0,
    "auto_diameter":      True,
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

        # ── data manager ──
        # Input data is read directly from state (no local copy).
        # Use the _input_data / _secondary_data / _input_scale properties below.
        # Single source of truth for segmentation: the napari Labels layer.
        # _seg_layer.data IS the segmentation — no shadow copy is kept.
        self._seg_layer      = None   # napari Labels layer (single source of truth)

        # ── model cache ──
        # Persists between runs so GPU memory is not thrashed on every segment call.
        # Both fields are written only from the worker thread, but _is_running ensures
        # at most one worker is active, so there is no concurrent access.
        self._cp_model     = None   # CellposeModel instance currently on GPU
        self._cp_model_key = None   # (model_type, custom_model_path, gpu) for the cached model

        self._export_dir = None  # user-configured export directory for Cellpose GUI
        self._state = get_state(viewer)

        self._setup_ui()
        self._state.tissue_changed.connect(self._on_tissue_changed)

    # ── State-backed data properties ───────────────────────────────────

    @property
    def _input_data(self):
        """Primary image — read directly from state; no local copy."""
        img = self._state.tissue.image
        return np.asarray(img) if img is not None else None

    @property
    def _secondary_data(self):
        """Secondary image for two-channel mode — read directly from state."""
        img2 = self._state.tissue.image2
        return np.asarray(img2) if img2 is not None else None

    @property
    def _input_scale(self):
        """Spatial scale from the linked napari Image layer."""
        layer_name = self._state.tissue.image_layer
        if layer_name and layer_name in self.viewer.layers:
            return tuple(self.viewer.layers[layer_name].scale)
        return None

    # ── UI construction ────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Segmentation status ──
        self._seg_status = QLabel("No segmentation")
        root.addWidget(self._seg_status)

        # ── Cellpose parameters (scrollable, collapsible) ──
        params_toggle = QToolButton()
        params_toggle.setText("Cellpose Parameters")
        params_toggle.setArrowType(Qt.RightArrow)
        params_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        params_toggle.setCheckable(True)
        params_toggle.setChecked(False)
        params_toggle.setStyleSheet("QToolButton { font-weight: bold; }")
        root.addWidget(params_toggle)

        params_inner = QWidget()
        self._params_form = QFormLayout(params_inner)
        self._params_form.setSpacing(4)
        self._build_params()
        scroll = QScrollArea()
        scroll.setWidget(params_inner)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(240)
        scroll.setVisible(False)
        root.addWidget(scroll)

        def _toggle_params(checked):
            params_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
            scroll.setVisible(checked)
        params_toggle.toggled.connect(_toggle_params)

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

        # Directory picker
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Directory:"))
        self._export_dir_edit = QLineEdit()
        self._export_dir_edit.setPlaceholderText("Leave empty to use a temp directory")
        self._export_dir_edit.textChanged.connect(self._on_export_dir_changed)
        self._browse_export_dir_btn = QPushButton("Browse")
        self._browse_export_dir_btn.setFixedWidth(60)
        self._browse_export_dir_btn.clicked.connect(self._browse_export_dir)
        dir_row.addWidget(self._export_dir_edit)
        dir_row.addWidget(self._browse_export_dir_btn)
        gui_lay.addLayout(dir_row)

        # Frame buttons
        frame_btns = QHBoxLayout()
        self._open_gui_btn   = QPushButton("Export Frame to Cellpose GUI")
        self._import_gui_btn = QPushButton("Import Frame from Cellpose GUI")
        self._import_gui_btn.setEnabled(False)
        self._open_gui_btn.clicked.connect(self._on_open_in_cellpose_gui)
        self._import_gui_btn.clicked.connect(self._on_import_from_cellpose_gui)
        frame_btns.addWidget(self._open_gui_btn)
        frame_btns.addWidget(self._import_gui_btn)
        gui_lay.addLayout(frame_btns)

        # Stack buttons
        stack_btns = QHBoxLayout()
        self._export_stack_gui_btn = QPushButton("Export Stack to Cellpose GUI")
        self._import_stack_gui_btn = QPushButton("Import Stack from Cellpose GUI")
        self._export_stack_gui_btn.clicked.connect(self._on_export_stack_to_cellpose_gui)
        self._import_stack_gui_btn.clicked.connect(self._on_import_stack_from_cellpose_gui)
        stack_btns.addWidget(self._export_stack_gui_btn)
        stack_btns.addWidget(self._import_stack_gui_btn)
        gui_lay.addLayout(stack_btns)

        self._gui_status = QLabel("Set a directory, then export frame(s) to edit in Cellpose GUI.")
        self._gui_status.setWordWrap(True)
        self._gui_status.setStyleSheet("color: palette(text);")
        gui_lay.addWidget(self._gui_status)
        root.addWidget(gui_box)

        # ── Progress + log ──
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        root.addWidget(self._progress)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        root.addWidget(self._cancel_btn)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(120)
        self._log.setPlaceholderText("Segmentation log…")
        root.addWidget(self._log)

        root.addStretch()

        # attribution
        attrib = QLabel(
            'Segmentation powered by '
            '<a href="https://github.com/MouseLand/cellpose">Cellpose</a>.'
            '<br>If you use segmentation, please cite:<br>'
            '<a href="https://doi.org/10.1038/s41592-022-01663-4">'
            'doi:10.1038/s41592-022-01663-4</a>'
        )
        attrib.setOpenExternalLinks(True)
        attrib.setWordWrap(True)
        attrib.setStyleSheet("color: palette(text); font-size: 9pt;")
        root.addWidget(attrib)

    def _build_params(self):
        add = self._params_form.addRow

        # Model
        add(_sep("Model"))
        self._model_combo = QComboBox()
        # In cellpose ≥ 4.0, all standard models (cyto3, nuclei, …) use the
        # same cpsam weights. Only cpsam and custom are meaningfully distinct.
        self._model_combo.addItems(["cpsam", "custom"])
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

    # ── Model cache ────────────────────────────────────────────────────

    def _ensure_model(self, params):
        """Return the cached CellposeModel, reloading only when params change.

        Called from the worker thread. Safe because _is_running ensures at most
        one worker is active at a time — no concurrent reads/writes to _cp_model.
        """
        from cellflow.backend.segmentation import make_cp_model

        key = (params["model_type"], params.get("custom_model_path"), params["gpu"])
        if self._cp_model is not None and self._cp_model_key == key:
            return self._cp_model

        # Free the old model explicitly before loading a new one.
        if self._cp_model is not None:
            del self._cp_model
            self._cp_model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        self._cp_model = make_cp_model(
            params["model_type"],
            custom_model_path=params.get("custom_model_path"),
            gpu=params["gpu"],
        )
        self._cp_model_key = key
        return self._cp_model

    # ── Frame helpers ──────────────────────────────────────────────────

    def _current_frame_idx(self, data):
        if data.ndim == 2:
            return 0
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) > 0 else 0

    def _get_frame(self, data, t):
        return data if data.ndim == 2 else data[t]

    def _get_or_create_labels_layer(self, shape, name="Segmentation"):
        # Prefer tracked layer reference — avoids creating a duplicate when the
        # layer was renamed or when a new stack has a different shape.
        if self._seg_layer is not None and self._seg_layer in self.viewer.layers:
            return self._seg_layer
        # Fall back to name-based search so we can adopt an existing layer.
        for lay in self.viewer.layers:
            if isinstance(lay, napari.layers.Labels) and lay.name == name:
                self._seg_layer = lay
                return lay
        # Nothing found — create a fresh layer and track it.
        # Inherit spatial scale from the source image layer so the labels align.
        scale = self._input_scale[-len(shape):] if self._input_scale is not None else None
        lay = self.viewer.add_labels(np.zeros(shape, dtype=np.uint32), name=name,
                                     **({} if scale is None else {"scale": scale}))
        self._seg_layer = lay
        return lay

    def _current_mode(self):
        """Auto-detect mode: use two-channel if Image 2 is available."""
        return "two_ch" if self._secondary_data is not None else "single"

    def _reference_stack_shape(self):
        if self._input_data is None:
            return (512, 512)
        return self._input_data.shape

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
        self._cancel_btn.setVisible(True)
        self._log.clear()

        @thread_worker(connect={
            "yielded":  lambda m: self._log_append(m),
            "returned": lambda r: self._on_frame_done(r, t),
            "errored":  self._on_error,
        })
        def _work():
            yield f"Segmenting frame {t}…"
            model = self._ensure_model(params)
            masks = _run_segmentation(imgs, mode, params, model=model)
            n = len(np.unique(masks[masks > 0]))
            yield f"  {n} object(s) detected"
            return masks

        self._worker = _work()
        self._worker.aborted.connect(self._on_aborted)

    def _get_inputs_for_frame(self):
        mode = self._current_mode()
        if mode == "single":
            if self._input_data is None:
                raise ValueError("Capture an Image layer in the Project Panel first.")
            t = self._current_frame_idx(self._input_data)
            return {"img": self._get_frame(self._input_data, t)}, t
        else:  # two_ch
            if self._input_data is None or self._secondary_data is None:
                raise ValueError("Capture Image and Image 2 in the Project Panel first.")
            t = self._current_frame_idx(self._input_data)
            return {
                "primary":   self._get_frame(self._input_data, t),
                "secondary": self._get_frame(self._secondary_data, t),
            }, t

    def _on_tissue_changed(self):
        """Input data is read from state via properties — nothing to sync here."""

    def _sync_labels_to_state(self):
        """Mirror the current seg_layer data to internal TissueData state."""
        if self._seg_layer is not None and self._seg_layer in self.viewer.layers:
            self._state.set_tissue_labels(
                np.asarray(self._seg_layer.data), self._seg_layer.name
            )

    def _on_frame_done(self, masks, t):
        self._is_running = False
        self._seg_frame_btn.setEnabled(True)
        self._seg_stack_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._cancel_btn.setVisible(False)

        shape = self._reference_stack_shape()
        layer = self._get_or_create_labels_layer(shape)
        if layer.data.ndim == 2:
            layer.data = masks.astype(np.uint32)
        else:
            data = layer.data.copy()
            data[t] = masks.astype(np.uint32)
            layer.data = data
        layer.refresh()
        self._seg_status.setText(f"Loaded: {layer.data.shape}")
        self._log_append("Frame segmentation complete.")
        self._sync_labels_to_state()

    # ── Segment Stack ──────────────────────────────────────────────────

    def _on_segment_stack(self):
        if self._is_running:
            self._log_append("Already processing.")
            return

        params = self._collect_params()
        mode   = self._current_mode()

        try:
            all_frames = self._get_all_frames()
        except ValueError as e:
            self._log_append(f"ERROR: {e}")
            return
        n_frames = len(all_frames)

        self._is_running = True
        self._seg_frame_btn.setEnabled(False)
        self._seg_stack_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._cancel_btn.setVisible(True)
        self._log.clear()

        @thread_worker(connect={
            "yielded":  lambda m: self._log_append(m),
            "returned": self._on_stack_done,
            "errored":  self._on_error,
        })
        def _work():
            yield "Loading model…"
            model = self._ensure_model(params)
            results = []
            for i, imgs in enumerate(all_frames):
                yield f"Frame {i+1}/{n_frames}…"
                results.append(_run_segmentation(imgs, mode, params, model=model).astype(np.uint32))
            return np.stack(results, axis=0)

        self._worker = _work()
        self._worker.aborted.connect(self._on_aborted)

    def _get_all_frames(self):
        mode = self._current_mode()
        if mode == "single":
            if self._input_data is None:
                raise ValueError("Capture an Image layer in the Project Panel first.")
            data = self._input_data
            if data.ndim == 2:
                return [{"img": data}]
            return [{"img": data[t]} for t in range(data.shape[0])]
        else:  # two_ch
            if self._input_data is None or self._secondary_data is None:
                raise ValueError("Capture Image and Image 2 in the Project Panel first.")
            p_data, s_data = self._input_data, self._secondary_data
            if p_data.ndim == 2:
                return [{"primary": p_data, "secondary": s_data}]
            return [
                {"primary": p_data[t], "secondary": s_data[t]}
                for t in range(p_data.shape[0])
            ]

    def _on_stack_done(self, stack):
        self._is_running = False
        self._seg_frame_btn.setEnabled(True)
        self._seg_stack_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._cancel_btn.setVisible(False)

        layer = self._get_or_create_labels_layer(stack.shape)
        layer.data = stack
        layer.refresh()
        self._seg_status.setText(f"Loaded: {layer.data.shape}")
        self._log_append(f"Stack segmentation complete: {stack.shape}.")
        self._sync_labels_to_state()

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

        export_dir = self._get_export_dir()
        self._temp_frame_idx = t

        # Use frame-indexed filenames so multiple exports never overwrite each other.
        # The frame index is also embedded in the seg file so import does not rely
        # solely on the in-memory variable (which is lost on restart).
        img_path = Path(export_dir) / f"frame_{t}.tif"
        seg_path = Path(export_dir) / f"frame_{t}_seg.npy"

        primary_img = imgs.get("img") if imgs.get("img") is not None else imgs.get("primary")
        secondary_img = imgs.get("secondary")

        from tifffile import imwrite
        if secondary_img is not None:
            # Write a two-channel TIF (2, H, W) so Cellpose GUI can use both channels.
            imwrite(str(img_path), np.stack([primary_img, secondary_img], axis=0).astype(np.uint16))
        else:
            imwrite(str(img_path), primary_img.astype(np.uint16))
        display_img = primary_img  # used for the seg npy below

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

        # Release the cached model from GPU before spawning Cellpose GUI.
        # cpsam is ~2 GiB on GPU; without this the subprocess OOMs trying to load
        # its own copy into the same device.  The model is re-cached on the next
        # Segment Frame / Segment Stack call.
        if self._cp_model is not None:
            del self._cp_model
            self._cp_model = None
            self._cp_model_key = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        launcher = Path(export_dir) / "launch_cellpose.py"
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
        if self._temp_frame_idx is None:
            self._log_append("No Cellpose GUI session found. Export a frame first.")
            return

        seg_path = Path(self._get_export_dir()) / f"frame_{self._temp_frame_idx}_seg.npy"
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
        self._seg_status.setText(f"Loaded: {layer.data.shape}")
        self._sync_labels_to_state()

        n = len(np.unique(masks[masks > 0]))
        self._log_append(f"Imported {n} mask(s) into frame {t} from Cellpose GUI.")
        self._gui_status.setText(f"Imported {n} mask(s) for frame {t}.")

    # ── Export directory helpers ───────────────────────────────────────

    def _browse_export_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Cellpose Export Directory", "")
        if d:
            self._export_dir = d
            self._export_dir_edit.setText(d)

    def _on_export_dir_changed(self, text):
        self._export_dir = text.strip() if text.strip() else None

    def _get_export_dir(self):
        """Return the configured export dir, or a persistent temp dir as fallback."""
        if self._export_dir and os.path.isdir(self._export_dir):
            return self._export_dir
        if self._temp_dir is None or not os.path.isdir(self._temp_dir):
            self._temp_dir = tempfile.mkdtemp(prefix="cellflow.segtrack_cellpose_")
        return self._temp_dir

    # ── Stack Cellpose GUI export / import ────────────────────────────

    def _on_export_stack_to_cellpose_gui(self):
        """Export all frames as TIF + NPY to the configured directory, then open Cellpose GUI."""
        if self._is_running:
            self._log_append("Already processing.")
            return

        try:
            all_frames = self._get_all_frames()
        except ValueError as e:
            self._log_append(f"ERROR: {e}")
            return

        export_dir = Path(self._get_export_dir())
        n_frames = len(all_frames)
        t_current = (
            self._current_frame_idx(self._input_data)
            if self._input_data is not None
            else 0
        )

        # Guard: if Cellpose is open for a specific frame, warn the user.
        if self._cellpose_proc is not None and self._cellpose_proc.poll() is None:
            self._log_append(
                "Cellpose GUI is still open. Import the current frame before exporting the stack."
            )
            return

        from tifffile import imwrite

        self._log_append(f"Exporting {n_frames} frame(s) to {export_dir} …")
        for i, imgs in enumerate(all_frames):
            primary_img = imgs.get("img") if imgs.get("img") is not None else imgs.get("primary")
            secondary_img = imgs.get("secondary")
            img_path = export_dir / f"frame_{i}.tif"
            seg_path = export_dir / f"frame_{i}_seg.npy"

            if secondary_img is not None:
                imwrite(str(img_path), np.stack([primary_img, secondary_img], axis=0).astype(np.uint16))
            else:
                imwrite(str(img_path), primary_img.astype(np.uint16))
            display_img = primary_img  # used for the seg npy below

            existing_masks = self._get_existing_masks(i)
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
                    "chan_choose":      [0, 0],
                    "ismanual":         np.zeros(n_cells + 1, dtype=bool),
                    "diameter":         self._diam_spin.value(),
                    "napari_frame_idx": i,
                }
                np.save(str(seg_path), seg_data)
            else:
                if seg_path.exists():
                    seg_path.unlink()

            self._log_append(f"  Frame {i+1}/{n_frames} exported.")

        self._log_append(f"Stack export complete: {n_frames} frame(s).")

        # Release GPU memory before spawning Cellpose GUI.
        if self._cp_model is not None:
            del self._cp_model
            self._cp_model = None
            self._cp_model_key = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Open Cellpose GUI with the current frame.
        self._temp_frame_idx = t_current
        img_path = export_dir / f"frame_{t_current}.tif"
        launcher = export_dir / "launch_cellpose.py"
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

        self._import_gui_btn.setEnabled(True)
        self._gui_status.setText(
            f"Stack exported to {export_dir}.\n"
            f"Frame {t_current} opened in Cellpose GUI.\n"
            "Correct and save each frame, then click 'Import Stack from Cellpose GUI'."
        )

    def _on_import_stack_from_cellpose_gui(self):
        """Import all frame_*_seg.npy files from the configured directory into the segmentation layer."""
        import glob

        export_dir = Path(self._get_export_dir())
        seg_files = sorted(glob.glob(str(export_dir / "frame_*_seg.npy")))

        if not seg_files:
            self._log_append(f"No frame_*_seg.npy files found in {export_dir}.")
            return

        shape = self._reference_stack_shape()
        layer = self._get_or_create_labels_layer(shape)
        data = layer.data.copy()

        imported = 0
        for f in seg_files:
            fname = Path(f).name
            try:
                idx_str = fname.removeprefix("frame_").removesuffix("_seg.npy")
                default_idx = int(idx_str)
            except ValueError:
                self._log_append(f"WARNING: Cannot parse frame index from {fname}, skipping.")
                continue

            try:
                seg_data = np.load(f, allow_pickle=True).item()
                masks = np.asarray(seg_data["masks"]).astype(np.uint32)
                t = int(seg_data.get("napari_frame_idx", default_idx))
            except Exception as exc:
                self._log_append(f"ERROR reading {fname}: {exc}")
                continue

            if data.ndim == 2:
                data = masks
            elif t < data.shape[0]:
                data[t] = masks
            else:
                self._log_append(f"WARNING: Frame index {t} out of range, skipping {fname}.")
                continue

            imported += 1

        layer.data = data
        layer.refresh()
        self._seg_status.setText(f"Loaded: {layer.data.shape}")
        self._sync_labels_to_state()
        self._log_append(f"Imported {imported} frame(s) from {export_dir}.")
        self._gui_status.setText(f"Imported {imported} frame(s) from {export_dir}.")

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
        self._cancel_btn.setVisible(False)
        self._log_append(f"ERROR: {exc}")

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.quit()

    def _on_aborted(self):
        self._is_running = False
        self._seg_frame_btn.setEnabled(True)
        self._seg_stack_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._log_append("Cancelled.")
        import traceback
        self._log_append(traceback.format_exc())

    def _log_append(self, msg):
        self._log.append(str(msg))
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())


# ── Segmentation dispatcher (called in background thread) ──────────────

def _run_segmentation(imgs, mode, params, model=None):
    """
    Run Cellpose segmentation for a single frame.

    imgs   : dict from _get_inputs_for_frame / _get_all_frames
    mode   : "single" | "two_ch"
    params : dict from _collect_params()
    model  : pre-loaded CellposeModel; if None one is created (and immediately
             discarded — prefer passing the cached model from SegmentationTab)

    Returns (H, W) mask.
    """
    from cellflow.backend.segmentation import make_cp_model, run_cp, run_cp_two_channel

    if model is None:
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
