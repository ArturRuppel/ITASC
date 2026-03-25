"""
Cell body expansion tab for napariSegTrack.

Takes a nuclear segmentation (Labels layer) as input and expands nuclear masks
to cell bodies via Voronoi tessellation. Optionally applies Lloyd's relaxation
(centroidal Voronoi) to produce more regular, hexagonal-like cell shapes —
the same algorithm used in napariTissueFlow.

Workflow:
  Segmentation tab  →  [nuclear Labels layer]  →  Cell Bodies tab  →  [cell Labels layer]
"""

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QComboBox, QCheckBox, QDoubleSpinBox, QSpinBox,
    QPushButton, QTextEdit, QProgressBar,
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
    """Cell body expansion tab: Voronoi tessellation from nuclear masks."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer  = viewer
        self._worker = None

        self._setup_ui()

        viewer.layers.events.inserted.connect(self._refresh_layer_combo)
        viewer.layers.events.removed.connect(self._refresh_layer_combo)

    # ── UI construction ────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Input ──
        input_box = QGroupBox("Nuclear Segmentation Input")
        input_lay = QFormLayout(input_box)
        self._nuc_combo = QComboBox()
        self._nuc_combo.setToolTip(
            "Labels layer produced by the Segmentation or Tracking tab."
        )
        input_lay.addRow("Labels layer:", self._nuc_combo)
        root.addWidget(input_box)

        # ── Expansion parameters ──
        params_box = QGroupBox("Expansion Parameters")
        params_lay = QFormLayout(params_box)

        self._maxexp_spin = QSpinBox()
        self._maxexp_spin.setRange(1, 500)
        self._maxexp_spin.setValue(VORONOI_DEFAULTS["max_expand"])
        self._maxexp_spin.setToolTip(
            "Maximum growth from the nucleus boundary (px). "
            "Edge cells are limited to this distance."
        )
        params_lay.addRow("Max expand (px):", self._maxexp_spin)

        self._smooth_sigma  = _dspin(0.1, 20, VORONOI_DEFAULTS["smooth_sigma"],  1, 0.5)
        self._smooth_thresh = _dspin(0.0,  1, VORONOI_DEFAULTS["smooth_thresh"], 2, 0.05)
        params_lay.addRow("Smooth sigma:",  self._smooth_sigma)
        params_lay.addRow("Smooth thresh:", self._smooth_thresh)

        root.addWidget(params_box)

        # ── Lloyd's relaxation ──
        lloyd_outer = QGroupBox("Geometry Relaxation (Lloyd's Algorithm)")
        lloyd_outer_lay = QVBoxLayout(lloyd_outer)

        self._lloyd_chk = QCheckBox("Enable Lloyd's relaxation")
        self._lloyd_chk.setChecked(VORONOI_DEFAULTS["lloyd"])
        self._lloyd_chk.setToolTip(
            "Iteratively moves Voronoi seeds to polygon centroids, producing more "
            "regular (hexagonal-like) cell shapes. Uses the same algorithm as "
            "napariTissueFlow."
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

        root.addWidget(lloyd_outer)

        # ── Action buttons ──
        btn_row = QHBoxLayout()
        self._frame_btn = QPushButton("Expand Frame")
        self._stack_btn = QPushButton("Expand Stack")
        self._frame_btn.clicked.connect(self._on_expand_frame)
        self._stack_btn.clicked.connect(self._on_expand_stack)
        btn_row.addWidget(self._frame_btn)
        btn_row.addWidget(self._stack_btn)
        root.addLayout(btn_row)

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
        self._log.setPlaceholderText("Cell body expansion log…")
        root.addWidget(self._log)

        root.addStretch()
        self._refresh_layer_combo()

    # ── Layer combo ────────────────────────────────────────────────────

    def _refresh_layer_combo(self, *_):
        label_names = [
            lay.name for lay in self.viewer.layers
            if isinstance(lay, napari.layers.Labels)
        ]
        prev = self._nuc_combo.currentText()
        self._nuc_combo.blockSignals(True)
        self._nuc_combo.clear()
        self._nuc_combo.addItems(label_names)
        idx = self._nuc_combo.findText(prev)
        if idx >= 0:
            self._nuc_combo.setCurrentIndex(idx)
        self._nuc_combo.blockSignals(False)

    # ── Parameter collection ───────────────────────────────────────────

    def _collect_params(self):
        return {
            "max_expand":       self._maxexp_spin.value(),
            "lloyd":            self._lloyd_chk.isChecked(),
            "lloyd_iterations": self._lloyd_iter_spin.value(),
            "lloyd_tol":        self._lloyd_tol_spin.value(),
            "smooth_sigma":     self._smooth_sigma.value(),
            "smooth_thresh":    self._smooth_thresh.value(),
        }

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_nuc_layer(self):
        name = self._nuc_combo.currentText()
        if not name:
            raise ValueError("Select a Labels layer.")
        return self.viewer.layers[name]

    def _current_frame_idx(self, data):
        if data.ndim == 2:
            return 0
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) > 0 else 0

    def _busy(self, is_busy):
        self._frame_btn.setEnabled(not is_busy)
        self._stack_btn.setEnabled(not is_busy)
        self._progress.setVisible(is_busy)
        self._cancel_btn.setVisible(is_busy)

    def _log_append(self, msg):
        self._log.append(str(msg))
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _on_error(self, exc):
        self._busy(False)
        self._log_append(f"ERROR: {exc}")
        import traceback
        self._log_append(traceback.format_exc())

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.quit()

    def _on_aborted(self):
        self._busy(False)
        self._log_append("Cancelled.")

    def _get_or_create_cells_layer(self, shape):
        for lay in self.viewer.layers:
            if isinstance(lay, napari.layers.Labels) and lay.name == "Cells":
                return lay
        return self.viewer.add_labels(np.zeros(shape, dtype=np.uint32), name="Cells")

    # ── Expand Frame ───────────────────────────────────────────────────

    def _on_expand_frame(self):
        if self._worker is not None and self._worker.is_running:
            self._log_append("Already processing.")
            return
        try:
            layer = self._get_nuc_layer()
        except ValueError as e:
            self._log_append(f"ERROR: {e}")
            return

        data   = np.asarray(layer.data)
        t      = self._current_frame_idx(data)
        frame  = data if data.ndim == 2 else data[t]
        params = self._collect_params()

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
        layer = self._get_or_create_cells_layer(orig_shape)
        if layer.data.ndim == 2:
            layer.data = result.astype(np.uint32)
        else:
            data = layer.data.copy()
            data[t] = result.astype(np.uint32)
            layer.data = data
        layer.refresh()
        self._log_append("Done.")

    # ── Expand Stack ───────────────────────────────────────────────────

    def _on_expand_stack(self):
        if self._worker is not None and self._worker.is_running:
            self._log_append("Already processing.")
            return
        try:
            layer = self._get_nuc_layer()
        except ValueError as e:
            self._log_append(f"ERROR: {e}")
            return

        data = np.asarray(layer.data)
        if data.ndim == 2:
            frames = [data]
        else:
            frames = [data[t] for t in range(data.shape[0])]
        params   = self._collect_params()
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
        stack = np.stack(results, axis=0)
        # Replace existing Cells layer if shape changed
        for lay in list(self.viewer.layers):
            if isinstance(lay, napari.layers.Labels) and lay.name == "Cells":
                if lay.data.shape != stack.shape:
                    self.viewer.layers.remove(lay)
                break
        layer = self._get_or_create_cells_layer(stack.shape)
        layer.data = stack
        layer.refresh()
        self._log_append(f"Done: {len(results)} frame(s).")


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
