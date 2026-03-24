"""Standalone Forces tab widget for ForSys force inference.

Takes a dataset from the shared ViewerState registry and lets users
run tension/pressure inference tissue-by-tissue or frame-by-frame,
with napari overlay visualization.
"""
import logging
from typing import Optional

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QProgressBar,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QScrollArea,
)
from qtpy.QtCore import QThread, Qt

from ..core.forsys_adapter import forsys_available
from .registry import get_state
from .visualization import (
    build_tension_colored_junctions,
    build_pressure_colored_cells,
)
from .workers import ForceInferenceWorker

logger = logging.getLogger(__name__)

_TENSION_LAYER = "[Forces] Tensions"
_PRESSURE_LAYER = "[Forces] Pressures"


class ForcesWidget(QWidget):
    """Widget for running ForSys force inference on a dataset."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._state = get_state(napari_viewer)
        self._thread: Optional[QThread] = None
        self._worker: Optional[ForceInferenceWorker] = None

        self._build_ui()
        self._connect_signals()
        self._update_from_dataset()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        scroll.setWidget(container)

        # --- ForSys availability banner ---
        if not forsys_available():
            banner = QLabel(
                "forsys is not installed.\n"
                "Install with: pip install napariTissueFlow[forces]"
            )
            banner.setStyleSheet(
                "QLabel { color: #cc4444; padding: 8px; "
                "border: 1px solid #cc4444; border-radius: 4px; }"
            )
            banner.setWordWrap(True)
            layout.addWidget(banner)

        # --- Dataset status ---
        self.dataset_label = QLabel("No dataset loaded")
        self.dataset_label.setWordWrap(True)
        layout.addWidget(self.dataset_label)

        # --- Tissue selector ---
        tissue_group = QGroupBox("Selection")
        tg_layout = QVBoxLayout()

        tissue_row = QHBoxLayout()
        tissue_row.addWidget(QLabel("Tissue:"))
        self.tissue_spin = QSpinBox()
        self.tissue_spin.setMinimum(0)
        self.tissue_spin.setMaximum(0)
        tissue_row.addWidget(self.tissue_spin)
        self.tissue_info = QLabel("")
        tissue_row.addWidget(self.tissue_info)
        tg_layout.addLayout(tissue_row)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(0)
        frame_row.addWidget(self.frame_spin)
        self.frame_info = QLabel("")
        frame_row.addWidget(self.frame_info)
        tg_layout.addLayout(frame_row)

        tissue_group.setLayout(tg_layout)
        layout.addWidget(tissue_group)

        # --- Scope ---
        scope_group = QGroupBox("Inference Scope")
        scope_layout = QVBoxLayout()

        self.scope_all = QRadioButton("All tissues (entire dataset)")
        self.scope_tissue = QRadioButton("Current tissue (all frames)")
        self.scope_frame = QRadioButton("Current frame only")
        self.scope_tissue.setChecked(True)

        self._scope_group = QButtonGroup(self)
        self._scope_group.addButton(self.scope_all, 0)
        self._scope_group.addButton(self.scope_tissue, 1)
        self._scope_group.addButton(self.scope_frame, 2)

        scope_layout.addWidget(self.scope_all)
        scope_layout.addWidget(self.scope_tissue)
        scope_layout.addWidget(self.scope_frame)
        scope_group.setLayout(scope_layout)
        layout.addWidget(scope_group)

        # --- Parameters ---
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout()

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Endpoint cluster tol (px):"))
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setMinimum(0.5)
        self.tol_spin.setMaximum(20.0)
        self.tol_spin.setSingleStep(0.5)
        self.tol_spin.setValue(3.0)
        self.tol_spin.setToolTip(
            "Distance threshold for clustering junction endpoints "
            "into triple junctions. Increase for coarse segmentations."
        )
        tol_row.addWidget(self.tol_spin)
        param_layout.addLayout(tol_row)

        self.allow_neg_cb = QCheckBox("Allow negative tensions")
        self.allow_neg_cb.setToolTip(
            "If unchecked, ForSys constrains tensions to be non-negative "
            "(physically realistic). Check for debugging."
        )
        param_layout.addWidget(self.allow_neg_cb)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # --- Run button ---
        self.infer_btn = QPushButton("Infer Forces")
        self.infer_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }"
        )
        layout.addWidget(self.infer_btn)

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # --- Visualization ---
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()

        self.show_tensions_cb = QCheckBox("Show tensions")
        self.show_tensions_cb.setToolTip(
            "Overlay junction lines colored by inferred tension "
            "(blue = low, red = high)"
        )
        viz_layout.addWidget(self.show_tensions_cb)

        self.show_pressures_cb = QCheckBox("Show pressures")
        self.show_pressures_cb.setToolTip(
            "Overlay cell polygons colored by inferred pressure "
            "(blue = low, red = high)"
        )
        viz_layout.addWidget(self.show_pressures_cb)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # --- Results ---
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    def _connect_signals(self):
        self._state.dataset_changed.connect(self._update_from_dataset)
        self.tissue_spin.valueChanged.connect(self._on_tissue_changed)
        self.frame_spin.valueChanged.connect(self._on_frame_changed)
        self._scope_group.buttonToggled.connect(self._on_scope_changed)
        self.infer_btn.clicked.connect(self._run_inference)
        self.show_tensions_cb.toggled.connect(self._update_visualization)
        self.show_pressures_cb.toggled.connect(self._update_visualization)

    # ------------------------------------------------------------------
    # Dataset state
    # ------------------------------------------------------------------
    def _update_from_dataset(self):
        ds = self._state.dataset
        if ds is None or ds.n_tissues == 0:
            self.dataset_label.setText("No dataset loaded")
            self.tissue_spin.setMaximum(0)
            self.frame_spin.setMaximum(0)
            self.tissue_info.setText("")
            self.frame_info.setText("")
            self.infer_btn.setEnabled(False)
            self.results_label.setText("")
            return

        ids = ds.tissue_ids
        self.dataset_label.setText(
            f"Dataset: {ds.n_tissues} tissue(s), "
            f"condition={ds.condition or '(none)'}"
        )
        self.tissue_spin.setMinimum(min(ids))
        self.tissue_spin.setMaximum(max(ids))
        self.infer_btn.setEnabled(forsys_available())
        self._on_tissue_changed(self.tissue_spin.value())

    def _on_tissue_changed(self, tissue_id):
        ds = self._state.dataset
        if ds is None or tissue_id not in ds.tissues:
            self.tissue_info.setText("")
            self.frame_spin.setMaximum(0)
            return

        series = ds.tissues[tissue_id]
        fi = series.frame_indices
        self.tissue_info.setText(f"({series.num_frames} frames)")
        self.frame_spin.setMinimum(min(fi))
        self.frame_spin.setMaximum(max(fi))
        self._on_frame_changed(self.frame_spin.value())
        self._update_results_label()

        # Update visualization if showing
        if self.show_tensions_cb.isChecked() or self.show_pressures_cb.isChecked():
            self._update_visualization()

    def _on_frame_changed(self, frame_idx):
        ds = self._state.dataset
        if ds is None:
            return
        tid = self.tissue_spin.value()
        if tid not in ds.tissues:
            return

        series = ds.tissues[tid]
        if frame_idx not in series.frames:
            self.frame_info.setText("")
            return

        frame = series.frames[frame_idx]
        n_t = sum(1 for jd in frame.junctions.values() if jd.tension is not None)
        n_p = sum(1 for cd in frame.cells.values() if cd.pressure is not None)
        if n_t > 0 or n_p > 0:
            self.frame_info.setText(f"({n_t}T, {n_p}P)")
        else:
            self.frame_info.setText("(no forces)")

        if self.show_tensions_cb.isChecked() or self.show_pressures_cb.isChecked():
            self._update_visualization()

    def _on_scope_changed(self):
        self.frame_spin.setEnabled(self.scope_frame.isChecked())

    # ------------------------------------------------------------------
    # Force inference
    # ------------------------------------------------------------------
    def _run_inference(self):
        ds = self._state.dataset
        if ds is None or ds.n_tissues == 0:
            return
        if not forsys_available():
            return

        # Clean up any previous thread
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    self._thread.quit()
                    self._thread.wait()
            except RuntimeError:
                pass  # C++ object already deleted by deleteLater
            self._thread = None
            self._worker = None

        scope = self._scope_group.checkedId()
        tid = self.tissue_spin.value()

        if scope == 0:
            tissue_ids = None  # all
            frame_indices = None
        elif scope == 1:
            tissue_ids = [tid]
            frame_indices = None
        else:
            tissue_ids = [tid]
            frame_indices = [self.frame_spin.value()]

        worker = ForceInferenceWorker(
            dataset=ds,
            tissue_ids=tissue_ids,
            frame_indices=frame_indices,
            endpoint_cluster_tol=self.tol_spin.value(),
            allow_negatives=self.allow_neg_cb.isChecked(),
        )

        self._thread = QThread()
        self._worker = worker
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        worker.finished.connect(self._thread.quit)
        worker.error.connect(self._thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self.infer_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting force inference...")

        self._thread.start()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self):
        self.progress_bar.setVisible(False)
        self.infer_btn.setEnabled(True)
        self.status_label.setText("Force inference complete.")

        self._update_results_label()
        self._on_frame_changed(self.frame_spin.value())

        # Auto-show results
        self.show_tensions_cb.setChecked(True)

    def _on_error(self, exc):
        self.progress_bar.setVisible(False)
        self.infer_btn.setEnabled(True)
        self.status_label.setText(f"Error: {exc}")

    # ------------------------------------------------------------------
    # Results summary
    # ------------------------------------------------------------------
    def _update_results_label(self):
        ds = self._state.dataset
        if ds is None:
            self.results_label.setText("")
            return

        lines = []
        for tid in ds.tissue_ids:
            series = ds.tissues[tid]
            n_t = sum(
                1 for f in series.frames.values()
                for jd in f.junctions.values() if jd.tension is not None
            )
            n_p = sum(
                1 for f in series.frames.values()
                for cd in f.cells.values() if cd.pressure is not None
            )
            if n_t > 0 or n_p > 0:
                lines.append(
                    f"Tissue {tid}: {n_t} tensions, {n_p} pressures"
                )
        self.results_label.setText("\n".join(lines) if lines else "")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def _remove_layer(self, name):
        for layer in list(self.viewer.layers):
            if layer.name == name:
                self.viewer.layers.remove(layer)

    def _update_visualization(self, _=None):
        ds = self._state.dataset
        if ds is None:
            return

        tid = self.tissue_spin.value()
        if tid not in ds.tissues:
            return
        series = ds.tissues[tid]

        # --- Tensions ---
        self._remove_layer(_TENSION_LAYER)
        if self.show_tensions_cb.isChecked():
            lines, colors = build_tension_colored_junctions(series)
            if lines:
                self.viewer.add_shapes(
                    lines, shape_type="path", edge_color=colors,
                    edge_width=3, name=_TENSION_LAYER,
                )

        # --- Pressures ---
        self._remove_layer(_PRESSURE_LAYER)
        if self.show_pressures_cb.isChecked():
            polys, pcolors = build_pressure_colored_cells(series)
            if polys:
                layer = self.viewer.add_shapes(
                    polys, shape_type="polygon", face_color=pcolors,
                    edge_width=0, name=_PRESSURE_LAYER, opacity=0.4,
                )
                # Move below tension layer so junctions are visible
                try:
                    self.viewer.layers.move(
                        self.viewer.layers.index(layer), 0
                    )
                except Exception:
                    pass
