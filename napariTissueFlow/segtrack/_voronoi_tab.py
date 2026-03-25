"""
Cell body expansion tab for napariSegTrack.

Two input paths — both share the Lloyd's relaxation parameters:

  1. From Layer   : select a Labels layer in the napari layer list and click
                    "Load from Layer".  Then use "Expand Frame" / "Expand Stack"
                    to grow nuclear masks to cell bodies via Voronoi expansion.

  2. From TrackMate : load a TrackMate XML file; Voronoi tessellation is computed
                    directly from nuclear spot positions and rasterized into a
                    Labels layer.  Use "Generate Labels" to run.

Workflow:
  Segmentation tab  →  [nuclear Labels layer]  →  Cell Bodies tab  →  [cell Labels layer]
"""

import numpy as np
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QTextEdit, QProgressBar, QLabel, QFileDialog,
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


# ── defaults ───────────────────────────────────────────────────────────

VORONOI_DEFAULTS = {
    "max_expand":       30,
    "lloyd":            False,
    "lloyd_iterations": 10,
    "lloyd_tol":        0.1,
    "smooth_sigma":     4.0,
    "smooth_thresh":    0.4,
}


# ── widget ─────────────────────────────────────────────────────────────

class VoronoiTab(QWidget):
    """Cell body expansion tab: Voronoi tessellation from nuclear masks or TrackMate XML."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer   = viewer
        self._worker  = None

        # ── data state ──
        self._seg_layer      = None   # napari Labels layer (from-layer path)
        self._trackmate_data = None   # TrackMateData (from-trackmate path)

        self._setup_ui()

    # ── UI construction ────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        root.addWidget(self._build_input_panel())
        root.addWidget(self._build_expansion_params())
        root.addWidget(self._build_lloyd_panel())

        # ── Action buttons ──
        btn_row = QHBoxLayout()
        self._frame_btn    = QPushButton("Expand Frame")
        self._stack_btn    = QPushButton("Expand Stack")
        self._generate_btn = QPushButton("Generate Labels")
        self._frame_btn.setToolTip("Expand the current napari frame from the loaded Labels layer.")
        self._stack_btn.setToolTip("Expand every frame of the loaded Labels layer.")
        self._generate_btn.setToolTip("Build a Voronoi label stack from the loaded TrackMate XML.")
        self._frame_btn.clicked.connect(self._on_expand_frame)
        self._stack_btn.clicked.connect(self._on_expand_stack)
        self._generate_btn.clicked.connect(self._on_generate_labels)
        btn_row.addWidget(self._frame_btn)
        btn_row.addWidget(self._stack_btn)
        btn_row.addWidget(self._generate_btn)
        root.addLayout(btn_row)

        # ── Progress + cancel ──
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        root.addWidget(self._progress)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        root.addWidget(self._cancel_btn)

        # ── Log ──
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(120)
        self._log.setPlaceholderText("Cell body expansion log…")
        root.addWidget(self._log)

        root.addStretch()

        self._update_button_states()

    # ── Input panel ────────────────────────────────────────────────────

    def _build_input_panel(self):
        box = QGroupBox("Input")
        lay = QVBoxLayout(box)

        # ── From layer ──
        from_layer_w = QWidget()
        from_layer_lay = QHBoxLayout(from_layer_w)
        from_layer_lay.setContentsMargins(0, 0, 0, 0)

        self._load_layer_btn = QPushButton("Load from Layer")
        self._load_layer_btn.setFixedWidth(140)
        self._load_layer_btn.setFixedHeight(25)
        self._load_layer_btn.setToolTip(
            "Select a Labels layer in the napari layer list, then click this button."
        )
        self._load_layer_btn.clicked.connect(self._on_load_layer)

        self._layer_status = QLabel("Not loaded")
        self._layer_status.setWordWrap(True)

        self._clear_layer_btn = QPushButton("Clear")
        self._clear_layer_btn.setFixedWidth(50)
        self._clear_layer_btn.setFixedHeight(25)
        self._clear_layer_btn.clicked.connect(self._on_clear_layer)

        from_layer_lay.addWidget(self._load_layer_btn)
        from_layer_lay.addWidget(self._layer_status, 1)
        from_layer_lay.addWidget(self._clear_layer_btn)
        lay.addWidget(from_layer_w)

        # ── Separator ──
        sep = QLabel("─── or ───────────────────────────────")
        sep.setStyleSheet("color: palette(mid); font-size: 10px;")
        lay.addWidget(sep)

        # ── From TrackMate XML ──
        tm_w = QWidget()
        tm_lay = QHBoxLayout(tm_w)
        tm_lay.setContentsMargins(0, 0, 0, 0)

        self._load_xml_btn = QPushButton("Load TrackMate XML…")
        self._load_xml_btn.setFixedWidth(160)
        self._load_xml_btn.setFixedHeight(25)
        self._load_xml_btn.clicked.connect(self._on_load_xml)

        self._xml_status = QLabel("No file loaded")
        self._xml_status.setWordWrap(True)

        tm_lay.addWidget(self._load_xml_btn)
        tm_lay.addWidget(self._xml_status, 1)
        lay.addWidget(tm_w)

        # Image dimensions (for TrackMate path)
        dims_w = QWidget()
        dims_lay = QHBoxLayout(dims_w)
        dims_lay.setContentsMargins(0, 0, 0, 0)
        dims_lay.addWidget(QLabel("H (px):"))
        self._height_spin = QSpinBox()
        self._height_spin.setRange(1, 100000)
        self._height_spin.setValue(512)
        dims_lay.addWidget(self._height_spin)
        dims_lay.addWidget(QLabel("W (px):"))
        self._width_spin = QSpinBox()
        self._width_spin.setRange(1, 100000)
        self._width_spin.setValue(512)
        dims_lay.addWidget(self._width_spin)
        dims_lay.addStretch()
        dims_w.setToolTip(
            "Image dimensions for Voronoi rasterization.\n"
            "Auto-detected from the TrackMate XML when available."
        )
        lay.addWidget(dims_w)

        return box

    def _build_expansion_params(self):
        box = QGroupBox("Expansion Parameters  (Labels path)")
        lay = QFormLayout(box)

        self._maxexp_spin = QSpinBox()
        self._maxexp_spin.setRange(1, 500)
        self._maxexp_spin.setValue(VORONOI_DEFAULTS["max_expand"])
        self._maxexp_spin.setToolTip(
            "Maximum growth from the nucleus boundary (px). "
            "Edge cells are limited to this distance."
        )
        lay.addRow("Max expand (px):", self._maxexp_spin)

        self._smooth_sigma  = _dspin(0.1, 20, VORONOI_DEFAULTS["smooth_sigma"],  1, 0.5)
        self._smooth_thresh = _dspin(0.0,  1, VORONOI_DEFAULTS["smooth_thresh"], 2, 0.05)
        lay.addRow("Smooth sigma:",  self._smooth_sigma)
        lay.addRow("Smooth thresh:", self._smooth_thresh)

        return box

    def _build_lloyd_panel(self):
        lloyd_outer = QGroupBox("Geometry Relaxation — Lloyd's Algorithm  (shared)")
        lloyd_outer_lay = QVBoxLayout(lloyd_outer)

        self._lloyd_chk = QCheckBox("Enable Lloyd's relaxation")
        self._lloyd_chk.setChecked(VORONOI_DEFAULTS["lloyd"])
        self._lloyd_chk.setToolTip(
            "Iteratively moves Voronoi seeds to polygon centroids, producing more "
            "regular (hexagonal-like) cell shapes.\n"
            "Applied to both the Labels expansion path and the TrackMate path."
        )
        lloyd_outer_lay.addWidget(self._lloyd_chk)

        self._lloyd_params_box = QGroupBox("Lloyd's Parameters")
        lloyd_params_lay = QFormLayout(self._lloyd_params_box)
        self._lloyd_iter_spin = QSpinBox()
        self._lloyd_iter_spin.setRange(1, 200)
        self._lloyd_iter_spin.setValue(VORONOI_DEFAULTS["lloyd_iterations"])
        self._lloyd_tol_spin = _dspin(0.0, 100.0, VORONOI_DEFAULTS["lloyd_tol"], 2, 0.05)
        lloyd_params_lay.addRow("Max iterations:", self._lloyd_iter_spin)
        lloyd_params_lay.addRow("Tolerance (px):", self._lloyd_tol_spin)
        self._lloyd_params_box.setVisible(False)
        lloyd_outer_lay.addWidget(self._lloyd_params_box)

        self._lloyd_chk.toggled.connect(self._lloyd_params_box.setVisible)

        return lloyd_outer

    # ── Load from layer ────────────────────────────────────────────────

    def _on_load_layer(self):
        active = self.viewer.layers.selection.active
        if active is None:
            self._layer_status.setText("Select a layer in napari first")
            return
        if not isinstance(active, napari.layers.Labels):
            self._layer_status.setText(f"'{active.name}' is not a Labels layer")
            return
        self._seg_layer = active
        self._layer_status.setText(f"Loaded: {active.name}  {active.data.shape}")
        self._update_button_states()

    def _on_clear_layer(self):
        self._seg_layer = None
        self._layer_status.setText("Not loaded")
        self._update_button_states()

    # ── Load TrackMate XML ─────────────────────────────────────────────

    def _on_load_xml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load TrackMate XML", "", "XML files (*.xml);;All files (*)"
        )
        if not path:
            return
        try:
            from napariTissueFlow.core.trackmate import parse_trackmate_xml
            td = parse_trackmate_xml(path)
            self._trackmate_data = td
            filename = Path(path).name
            if td.image_shape is not None:
                self._height_spin.setValue(td.image_shape[0])
                self._width_spin.setValue(td.image_shape[1])
            info = (
                f"{filename}  —  {td.n_spots} spots, {td.n_tracks} tracks, "
                f"{len(td.spots_by_frame)} frames"
            )
            if td.image_shape:
                info += f"  ({td.image_shape[1]}×{td.image_shape[0]} px)"
            self._xml_status.setText(info)
        except Exception as e:
            self._xml_status.setText(f"Error: {e}")
            self._trackmate_data = None
        self._update_button_states()

    # ── Button state management ────────────────────────────────────────

    def _update_button_states(self):
        layer_ok = self._seg_layer is not None and self._seg_layer in self.viewer.layers
        tm_ok    = self._trackmate_data is not None
        self._frame_btn.setEnabled(layer_ok)
        self._stack_btn.setEnabled(layer_ok)
        self._generate_btn.setEnabled(tm_ok)

    # ── Parameter collection ───────────────────────────────────────────

    def _collect_expansion_params(self):
        return {
            "max_expand":       self._maxexp_spin.value(),
            "lloyd":            self._lloyd_chk.isChecked(),
            "lloyd_iterations": self._lloyd_iter_spin.value(),
            "lloyd_tol":        self._lloyd_tol_spin.value(),
            "smooth_sigma":     self._smooth_sigma.value(),
            "smooth_thresh":    self._smooth_thresh.value(),
        }

    def _collect_lloyd_params(self):
        return {
            "lloyd":            self._lloyd_chk.isChecked(),
            "lloyd_iterations": self._lloyd_iter_spin.value(),
            "lloyd_tol":        self._lloyd_tol_spin.value(),
        }

    # ── Helpers ────────────────────────────────────────────────────────

    def _current_frame_idx(self, data):
        if data.ndim == 2:
            return 0
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) > 0 else 0

    def _busy(self, is_busy):
        self._frame_btn.setEnabled(not is_busy)
        self._stack_btn.setEnabled(not is_busy)
        self._generate_btn.setEnabled(not is_busy)
        self._load_layer_btn.setEnabled(not is_busy)
        self._load_xml_btn.setEnabled(not is_busy)
        self._progress.setVisible(is_busy)
        self._cancel_btn.setVisible(is_busy)

    def _log_append(self, msg):
        self._log.append(str(msg))
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _on_error(self, exc):
        self._busy(False)
        self._update_button_states()
        self._log_append(f"ERROR: {exc}")
        import traceback
        self._log_append(traceback.format_exc())

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.quit()

    def _on_aborted(self):
        self._busy(False)
        self._update_button_states()
        self._log_append("Cancelled.")

    def _get_or_create_cells_layer(self, shape):
        for lay in self.viewer.layers:
            if isinstance(lay, napari.layers.Labels) and lay.name == "Cells":
                return lay
        return self.viewer.add_labels(np.zeros(shape, dtype=np.uint32), name="Cells")

    # ── Expand Frame (Labels path) ─────────────────────────────────────

    def _on_expand_frame(self):
        if self._worker is not None and self._worker.is_running:
            self._log_append("Already processing.")
            return
        if self._seg_layer is None or self._seg_layer not in self.viewer.layers:
            self._log_append("ERROR: Load a Labels layer first.")
            return

        data   = np.asarray(self._seg_layer.data)
        t      = self._current_frame_idx(data)
        frame  = data if data.ndim == 2 else data[t]
        params = self._collect_expansion_params()

        self._log.clear()
        self._busy(True)

        @thread_worker(connect={
            "yielded":  lambda m: self._log_append(m),
            "returned": lambda r: self._on_frame_done(r, data.shape, t),
            "errored":  self._on_error,
        })
        def _work():
            method = "lloyd" if params["lloyd"] else "standard"
            yield f"Expanding frame {t} (method={method})…"
            result = _run_expansion(frame, params)
            n = len(np.unique(result[result > 0]))
            yield f"  {n} cell(s)"
            return result

        self._worker = _work()
        self._worker.aborted.connect(self._on_aborted)

    def _on_frame_done(self, result, orig_shape, t):
        self._busy(False)
        self._update_button_states()
        layer = self._get_or_create_cells_layer(orig_shape)
        if layer.data.ndim == 2:
            layer.data = result.astype(np.uint32)
        else:
            data = layer.data.copy()
            data[t] = result.astype(np.uint32)
            layer.data = data
        layer.refresh()
        self._log_append("Done.")

    # ── Expand Stack (Labels path) ─────────────────────────────────────

    def _on_expand_stack(self):
        if self._worker is not None and self._worker.is_running:
            self._log_append("Already processing.")
            return
        if self._seg_layer is None or self._seg_layer not in self.viewer.layers:
            self._log_append("ERROR: Load a Labels layer first.")
            return

        data = np.asarray(self._seg_layer.data)
        if data.ndim == 2:
            frames = [data]
        else:
            frames = [data[t] for t in range(data.shape[0])]
        params   = self._collect_expansion_params()
        n_frames = len(frames)

        self._log.clear()
        self._busy(True)

        @thread_worker(connect={
            "yielded":  lambda m: self._log_append(m),
            "returned": self._on_stack_done,
            "errored":  self._on_error,
        })
        def _work():
            results = []
            for i, frame in enumerate(frames):
                yield f"Frame {i+1}/{n_frames}…"
                results.append(_run_expansion(frame, params).astype(np.uint32))
            return results

        self._worker = _work()
        self._worker.aborted.connect(self._on_aborted)

    def _on_stack_done(self, results):
        self._busy(False)
        self._update_button_states()
        stack = np.stack(results, axis=0)
        for lay in list(self.viewer.layers):
            if isinstance(lay, napari.layers.Labels) and lay.name == "Cells":
                if lay.data.shape != stack.shape:
                    self.viewer.layers.remove(lay)
                break
        layer = self._get_or_create_cells_layer(stack.shape)
        layer.data = stack
        layer.refresh()
        self._log_append(f"Done: {len(results)} frame(s).")

    # ── Generate Labels (TrackMate path) ───────────────────────────────

    def _on_generate_labels(self):
        if self._worker is not None and self._worker.is_running:
            self._log_append("Already processing.")
            return
        if self._trackmate_data is None:
            self._log_append("ERROR: Load a TrackMate XML first.")
            return

        td           = self._trackmate_data
        image_shape  = (self._height_spin.value(), self._width_spin.value())
        lloyd_params = self._collect_lloyd_params()

        self._log.clear()
        self._busy(True)

        @thread_worker(connect={
            "yielded":  lambda m: self._log_append(m),
            "returned": self._on_generate_done,
            "errored":  self._on_error,
        })
        def _work():
            from napariTissueFlow.core.voronoi import voronoi_to_labels
            from napariTissueFlow.structures import VoronoiMethod

            method = VoronoiMethod.LLOYD if lloyd_params["lloyd"] else VoronoiMethod.STANDARD
            frames     = sorted(td.spots_by_frame.keys())
            n_frames   = len(frames)
            H, W       = image_shape
            label_stack = np.zeros((n_frames, H, W), dtype=np.int32)
            track_map   = {}

            yield f"Building Voronoi labels: {n_frames} frames, {H}×{W} px…"
            for i, frame in enumerate(frames):
                yield f"  Frame {i+1}/{n_frames}…"
                spots    = td.spots_by_frame[frame]
                if not spots:
                    continue
                spot_ids  = [s[0] for s in spots]
                positions = np.array([[s[1], s[2]] for s in spots])
                labels, _ = voronoi_to_labels(
                    positions, image_shape,
                    method=method,
                    lloyd_iterations=lloyd_params["lloyd_iterations"],
                    lloyd_tol=lloyd_params["lloyd_tol"],
                )
                label_stack[i] = labels
                frame_tracks = {}
                for cell_idx, spot_id in enumerate(spot_ids):
                    track_id = td.spot_to_track.get(spot_id)
                    if track_id is not None:
                        frame_tracks[cell_idx + 1] = track_id
                track_map[i] = frame_tracks

            yield "Voronoi tessellation complete."
            return label_stack, track_map

        self._worker = _work()
        self._worker.aborted.connect(self._on_aborted)

    def _on_generate_done(self, result):
        self._busy(False)
        self._update_button_states()
        label_stack, track_map = result

        layer_name = "Voronoi Labels"
        for lay in list(self.viewer.layers):
            if lay.name == layer_name:
                self.viewer.layers.remove(lay)
                break

        layer = self.viewer.add_labels(label_stack, name=layer_name)
        layer.metadata["track_map"] = track_map
        layer.metadata["source"]    = "nuclear_tracks"

        n_frames = label_stack.shape[0]
        avg = np.mean([
            len(np.unique(label_stack[i])) - (1 if 0 in label_stack[i] else 0)
            for i in range(n_frames)
        ])
        self._log_append(
            f"Done: {n_frames} frame(s), ~{avg:.0f} cells/frame. "
            f"Layer '{layer_name}' added."
        )


# ── Expansion dispatcher (called in background thread) ─────────────────

def _run_expansion(nuclei, params):
    """Expand nuclear masks to cell bodies for a single (H, W) frame."""
    from napariTissueFlow.segtrack._pipeline import (
        expand_voronoi, expand_voronoi_lloyd, smooth_labels,
    )
    if params["lloyd"]:
        expanded = expand_voronoi_lloyd(
            nuclei,
            params["max_expand"],
            lloyd_iterations=params["lloyd_iterations"],
            lloyd_tol=params["lloyd_tol"],
        )
    else:
        expanded = expand_voronoi(nuclei, params["max_expand"])

    return smooth_labels(expanded, params["smooth_sigma"], params["smooth_thresh"])
