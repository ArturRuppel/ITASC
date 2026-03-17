"""Napari dock widget for napariTissueGraph — multi-tissue support."""
import logging
import os
from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QProgressBar,
    QListWidget,
    QLineEdit,
    QFileDialog,
    QGroupBox,
    QSlider,
    QSpinBox,
)
from qtpy.QtCore import Signal, QThread, QObject, Qt

from ..structures import TissueGraphDataset, TissueGraphTimeSeries, InputType
from ..core.graph import build_from_labels_4d, build_from_tracks_4d
from ..core.topology import detect_all_t1_events
from .visualization import build_all_junction_lines, build_all_centroids, build_t1_markers

logger = logging.getLogger(__name__)


class DatasetBuildWorker(QObject):
    """Worker to build a TissueGraphDataset in a background thread."""

    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(Exception)

    def __init__(self, input_type: str, label_stacks=None, track_positions=None,
                 pixel_size=None, time_interval=None, condition=""):
        super().__init__()
        self.input_type = input_type
        self.label_stacks = label_stacks
        self.track_positions = track_positions
        self.pixel_size = pixel_size
        self.time_interval = time_interval
        self.condition = condition

    def run(self):
        try:
            def _progress(frac, msg):
                self.progress.emit(int(frac * 90), msg)

            if self.input_type == "Segmentation Labels":
                dataset = build_from_labels_4d(
                    self.label_stacks,
                    pixel_size=self.pixel_size,
                    time_interval=self.time_interval,
                    condition=self.condition,
                    progress_callback=_progress,
                )
            else:
                dataset = build_from_tracks_4d(
                    self.track_positions,
                    pixel_size=self.pixel_size,
                    time_interval=self.time_interval,
                    condition=self.condition,
                    progress_callback=_progress,
                )

            self.progress.emit(92, "Detecting T1 events...")
            detect_all_t1_events(dataset)

            self.finished.emit(dataset)
        except Exception as e:
            self.error.emit(e)


class TissueGraphWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.dataset: TissueGraphDataset = None
        self._label_stacks: list[np.ndarray] = []
        self._file_names: list[str] = []
        self._junction_layer = None
        self._centroid_layer = None
        self._t1_layer = None
        self._thread = None
        self._worker = None

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- Input type ---
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Input type:"))
        self.input_type_combo = QComboBox()
        self.input_type_combo.addItems(["Segmentation Labels", "Nuclear Tracks"])
        type_row.addWidget(self.input_type_combo)
        layout.addLayout(type_row)

        # --- Layer selection (for single-layer / tracks mode) ---
        self.layer_group = QGroupBox("Layer")
        layer_layout = QHBoxLayout()
        self.layer_combo = QComboBox()
        layer_layout.addWidget(self.layer_combo)
        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedWidth(30)
        layer_layout.addWidget(self.refresh_btn)
        self.layer_group.setLayout(layer_layout)
        layout.addWidget(self.layer_group)

        # --- Multi-file loading (for segmentation labels) ---
        self.file_group = QGroupBox("Label Files")
        file_layout = QVBoxLayout()
        self.load_btn = QPushButton("Load Labels...")
        file_layout.addWidget(self.load_btn)
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(120)
        file_layout.addWidget(self.file_list)
        self.file_info_label = QLabel("")
        file_layout.addWidget(self.file_info_label)
        self.file_group.setLayout(file_layout)
        layout.addWidget(self.file_group)

        # --- Parameters ---
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout()

        px_row = QHBoxLayout()
        px_row.addWidget(QLabel("Pixel size (µm/px):"))
        self.pixel_size_edit = QLineEdit("")
        self.pixel_size_edit.setPlaceholderText("optional")
        px_row.addWidget(self.pixel_size_edit)
        param_layout.addLayout(px_row)

        dt_row = QHBoxLayout()
        dt_row.addWidget(QLabel("Time interval (s):"))
        self.time_interval_edit = QLineEdit("")
        self.time_interval_edit.setPlaceholderText("optional")
        dt_row.addWidget(self.time_interval_edit)
        param_layout.addLayout(dt_row)

        cond_row = QHBoxLayout()
        cond_row.addWidget(QLabel("Condition:"))
        self.condition_edit = QLineEdit("")
        self.condition_edit.setPlaceholderText("e.g. WT, vim_KO")
        cond_row.addWidget(self.condition_edit)
        param_layout.addLayout(cond_row)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # --- Build button ---
        self.build_btn = QPushButton("Build Graph")
        layout.addWidget(self.build_btn)

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # --- Tissue inspection ---
        self.inspect_group = QGroupBox("Tissue Inspection")
        inspect_layout = QVBoxLayout()

        tissue_row = QHBoxLayout()
        tissue_row.addWidget(QLabel("Tissue:"))
        self.tissue_spinner = QSpinBox()
        self.tissue_spinner.setMinimum(0)
        self.tissue_spinner.setMaximum(0)
        tissue_row.addWidget(self.tissue_spinner)
        self.tissue_info_label = QLabel("")
        tissue_row.addWidget(self.tissue_info_label)
        inspect_layout.addLayout(tissue_row)

        self.remove_tissue_btn = QPushButton("Remove Tissue")
        inspect_layout.addWidget(self.remove_tissue_btn)

        self.dataset_summary_label = QLabel("")
        self.dataset_summary_label.setWordWrap(True)
        inspect_layout.addWidget(self.dataset_summary_label)

        self.inspect_group.setLayout(inspect_layout)
        self.inspect_group.setVisible(False)
        layout.addWidget(self.inspect_group)

        layout.addStretch()

        # Set initial visibility
        self._update_input_mode()

    def _connect_signals(self):
        self.input_type_combo.currentIndexChanged.connect(self._update_input_mode)
        self.refresh_btn.clicked.connect(self._refresh_layers)
        self.load_btn.clicked.connect(self._load_label_files)
        self.build_btn.clicked.connect(self._build_graph)
        self.tissue_spinner.valueChanged.connect(self._on_tissue_changed)
        self.remove_tissue_btn.clicked.connect(self._remove_current_tissue)
        self._refresh_layers()

    # ------------------------------------------------------------------
    # Input mode switching
    # ------------------------------------------------------------------
    def _update_input_mode(self):
        is_labels = self.input_type_combo.currentText() == "Segmentation Labels"
        self.file_group.setVisible(is_labels)
        self.layer_group.setVisible(not is_labels)
        self._refresh_layers()

    def _refresh_layers(self):
        self.layer_combo.clear()
        import napari
        input_type = self.input_type_combo.currentText()
        for layer in self.viewer.layers:
            if input_type == "Nuclear Tracks" and isinstance(layer, napari.layers.Points):
                self.layer_combo.addItem(layer.name)

    # ------------------------------------------------------------------
    # Multi-file loading
    # ------------------------------------------------------------------
    def _load_label_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Label Files", "", "TIFF files (*.tif *.tiff);;All files (*)"
        )
        if not files:
            return

        import tifffile

        self._label_stacks = []
        self._file_names = []
        self.file_list.clear()

        for path in sorted(files):
            stack = tifffile.imread(path)
            if stack.ndim == 2:
                stack = stack[np.newaxis, ...]
            if stack.ndim != 3:
                self.status_label.setText(
                    f"Skipped {Path(path).name}: expected 2D or 3D, got {stack.ndim}D"
                )
                continue
            self._label_stacks.append(stack)
            name = Path(path).name
            self._file_names.append(name)
            self.file_list.addItem(f"{name}  ({stack.shape[0]} frames, {stack.shape[1]}x{stack.shape[2]})")

        n = len(self._label_stacks)
        frame_counts = [s.shape[0] for s in self._label_stacks]
        self.file_info_label.setText(
            f"{n} tissue(s) loaded — frames: {frame_counts}"
        )

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------
    def _build_graph(self):
        input_type = self.input_type_combo.currentText()

        # Parse parameters
        pixel_size = self._parse_float(self.pixel_size_edit.text())
        time_interval = self._parse_float(self.time_interval_edit.text())
        condition = self.condition_edit.text().strip()

        if input_type == "Segmentation Labels":
            if not self._label_stacks:
                self.status_label.setText("No label files loaded. Use 'Load Labels...' first.")
                return
            self._thread = QThread()
            self._worker = DatasetBuildWorker(
                input_type=input_type,
                label_stacks=self._label_stacks,
                pixel_size=pixel_size,
                time_interval=time_interval,
                condition=condition,
            )
        else:
            layer_name = self.layer_combo.currentText()
            if not layer_name:
                self.status_label.setText("No layer selected.")
                return
            layer = self.viewer.layers[layer_name]
            positions = layer.data
            self._thread = QThread()
            self._worker = DatasetBuildWorker(
                input_type=input_type,
                track_positions=positions,
                pixel_size=pixel_size,
                time_interval=time_interval,
                condition=condition,
            )

        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_build_finished)
        self._worker.error.connect(self._on_build_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self.build_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Building graph...")

        self._thread.start()

    def _on_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _on_build_finished(self, dataset):
        self.dataset = dataset
        self.build_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        self._update_dataset_summary()
        self._setup_tissue_inspector()

        # Show first tissue
        if dataset.tissue_ids:
            self.tissue_spinner.setValue(dataset.tissue_ids[0])
            self._show_tissue(dataset.tissue_ids[0])

    def _on_build_error(self, exc):
        self.build_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {exc}")
        logger.exception("Graph build failed", exc_info=exc)

    # ------------------------------------------------------------------
    # Dataset summary
    # ------------------------------------------------------------------
    def _update_dataset_summary(self):
        ds = self.dataset
        parts = [f"{ds.n_tissues} tissue(s)"]
        if ds.condition:
            parts.append(f"condition: {ds.condition}")
        for tid in ds.tissue_ids:
            s = ds.tissues[tid]
            n_t1 = len(s.t1_events)
            parts.append(f"  T{tid}: {s.num_frames} frames, {n_t1} T1 events")
        self.dataset_summary_label.setText("\n".join(parts))

    # ------------------------------------------------------------------
    # Tissue inspection
    # ------------------------------------------------------------------
    def _setup_tissue_inspector(self):
        self.inspect_group.setVisible(True)
        ids = self.dataset.tissue_ids
        if ids:
            self.tissue_spinner.setMinimum(min(ids))
            self.tissue_spinner.setMaximum(max(ids))

    def _on_tissue_changed(self, tissue_id):
        if self.dataset is None:
            return
        if tissue_id in self.dataset.tissue_ids:
            self._show_tissue(tissue_id)

    def _show_tissue(self, tissue_id):
        series = self.dataset.tissues[tissue_id]
        n_t1 = len(series.t1_events)
        n_trajs = len(series.edge_trajectories)
        total_cells = sum(len(f.cells) for f in series.frames.values())
        total_junctions = sum(len(f.junctions) for f in series.frames.values())
        self.tissue_info_label.setText(
            f"{series.num_frames}f, {total_cells}c, {total_junctions}j, {n_t1} T1"
        )
        self.status_label.setText(
            f"Tissue {tissue_id}: {series.num_frames} frames, "
            f"{total_cells} cells, {total_junctions} junctions, "
            f"{n_t1} T1 events, {n_trajs} edge trajectories"
        )
        self._add_layers(series)

    def _remove_current_tissue(self):
        if self.dataset is None:
            return
        tid = self.tissue_spinner.value()
        if tid not in self.dataset.tissue_ids:
            self.status_label.setText(f"Tissue {tid} does not exist.")
            return
        self.dataset.remove_tissue(tid)
        self._update_dataset_summary()

        remaining = self.dataset.tissue_ids
        if remaining:
            self.tissue_spinner.setMinimum(min(remaining))
            self.tissue_spinner.setMaximum(max(remaining))
            self.tissue_spinner.setValue(remaining[0])
            self._show_tissue(remaining[0])
        else:
            self.inspect_group.setVisible(False)
            self._remove_layers()
            self.status_label.setText("All tissues removed.")

    # ------------------------------------------------------------------
    # Napari layer management
    # ------------------------------------------------------------------
    def _remove_layers(self):
        for layer in (self._junction_layer, self._centroid_layer, self._t1_layer):
            if layer is not None and layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self._junction_layer = None
        self._centroid_layer = None
        self._t1_layer = None

    def _add_layers(self, series: TissueGraphTimeSeries):
        """Add junction, centroid, and T1 layers for a single tissue.

        Data is pre-built for all frames so napari's native dim slider
        handles frame scrubbing — no per-frame recomputation needed.
        """
        self._remove_layers()

        # Junctions as shapes with (frame, y, x) coordinates
        lines, colors = build_all_junction_lines(series)
        if lines:
            self._junction_layer = self.viewer.add_shapes(
                lines,
                shape_type="path",
                edge_color=colors,
                edge_width=2,
                name="Junctions",
            )

        # Centroids as points with (frame, y, x) coordinates
        centroids = build_all_centroids(series)
        if len(centroids) > 0:
            self._centroid_layer = self.viewer.add_points(
                centroids,
                size=5,
                face_color="yellow",
                name="Cell Centroids",
            )

        # T1 event markers with (frame, y, x) coordinates
        t1_positions = build_t1_markers(series.t1_events)
        if len(t1_positions) > 0:
            self._t1_layer = self.viewer.add_points(
                t1_positions,
                size=12,
                face_color="red",
                symbol="star",
                name="T1 Events",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_float(text: str):
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def cleanup(self):
        """Clean up background thread if running."""
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
