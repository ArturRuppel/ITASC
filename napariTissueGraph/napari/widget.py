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
    QDoubleSpinBox,
)
from qtpy.QtCore import Signal, QThread, QObject, Qt

from ..structures import TissueGraphDataset, TissueGraphTimeSeries, InputType, VoronoiMethod
from ..core.graph import (
    build_from_labels_4d,
    build_from_tracks_4d,
    build_from_both,
    build_from_trackmate,
)
from ..core.trackmate import parse_trackmate_xml, TrackMateData
from ..core.topology import detect_all_t1_events
from .visualization import build_all_junction_lines, build_all_centroids, build_t1_markers

logger = logging.getLogger(__name__)

INPUT_SEGMENTATION = "Segmentation Labels"
INPUT_TRACKS = "Nuclear Tracks"
INPUT_BOTH = "Both (Labels + Tracks)"


class DatasetBuildWorker(QObject):
    """Worker to build a TissueGraphDataset in a background thread."""

    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(Exception)

    def __init__(
        self,
        input_type: str,
        label_stacks=None,
        track_positions=None,
        trackmate_data=None,
        pixel_size=None,
        time_interval=None,
        condition="",
        voronoi_method=VoronoiMethod.STANDARD,
        lloyd_iterations=10,
        lloyd_tol=0.1,
        min_iou=0.3,
        match_threshold=10.0,
    ):
        super().__init__()
        self.input_type = input_type
        self.label_stacks = label_stacks
        self.track_positions = track_positions
        self.trackmate_data = trackmate_data
        self.pixel_size = pixel_size
        self.time_interval = time_interval
        self.condition = condition
        self.voronoi_method = voronoi_method
        self.lloyd_iterations = lloyd_iterations
        self.lloyd_tol = lloyd_tol
        self.min_iou = min_iou
        self.match_threshold = match_threshold

    def run(self):
        try:
            def _progress(frac, msg):
                self.progress.emit(int(frac * 90), msg)

            if self.input_type == INPUT_SEGMENTATION:
                dataset = build_from_labels_4d(
                    self.label_stacks,
                    pixel_size=self.pixel_size,
                    time_interval=self.time_interval,
                    condition=self.condition,
                    progress_callback=_progress,
                    min_iou=self.min_iou,
                )
            elif self.input_type == INPUT_TRACKS:
                if self.trackmate_data is not None:
                    # Single tissue from TrackMate
                    self.progress.emit(10, "Building from TrackMate data...")
                    series = build_from_trackmate(
                        self.trackmate_data,
                        pixel_size=self.pixel_size,
                        time_interval=self.time_interval,
                        method=self.voronoi_method,
                        lloyd_iterations=self.lloyd_iterations,
                        lloyd_tol=self.lloyd_tol,
                    )
                    dataset = TissueGraphDataset(
                        tissues={},
                        condition=self.condition,
                        pixel_size=self.pixel_size,
                        time_interval=self.time_interval,
                        input_type=InputType.VORONOI,
                    )
                    dataset.add_tissue(series)
                else:
                    dataset = build_from_tracks_4d(
                        self.track_positions,
                        pixel_size=self.pixel_size,
                        time_interval=self.time_interval,
                        condition=self.condition,
                        progress_callback=_progress,
                        method=self.voronoi_method,
                        lloyd_iterations=self.lloyd_iterations,
                        lloyd_tol=self.lloyd_tol,
                    )
            else:
                # Both mode
                self.progress.emit(10, "Building from labels + tracks...")
                series = build_from_both(
                    self.label_stacks[0],
                    self.trackmate_data,
                    pixel_size=self.pixel_size,
                    time_interval=self.time_interval,
                    match_threshold=self.match_threshold,
                )
                dataset = TissueGraphDataset(
                    tissues={},
                    condition=self.condition,
                    pixel_size=self.pixel_size,
                    time_interval=self.time_interval,
                    input_type=InputType.SEGMENTATION_WITH_TRACKS,
                )
                dataset.add_tissue(series)

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
        self._trackmate_data: TrackMateData = None
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
        self.input_type_combo.addItems([INPUT_SEGMENTATION, INPUT_TRACKS, INPUT_BOTH])
        type_row.addWidget(self.input_type_combo)
        layout.addLayout(type_row)

        # --- Layer selection (for tracks mode with Points layer) ---
        self.layer_group = QGroupBox("Layer")
        layer_layout = QHBoxLayout()
        self.layer_combo = QComboBox()
        layer_layout.addWidget(self.layer_combo)
        self.refresh_btn = QPushButton("\u21bb")
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

        # --- TrackMate XML loading ---
        self.trackmate_group = QGroupBox("TrackMate XML")
        tm_layout = QVBoxLayout()
        self.trackmate_load_btn = QPushButton("Load TrackMate XML...")
        tm_layout.addWidget(self.trackmate_load_btn)
        self.trackmate_info_label = QLabel("")
        self.trackmate_info_label.setWordWrap(True)
        tm_layout.addWidget(self.trackmate_info_label)
        self.trackmate_group.setLayout(tm_layout)
        layout.addWidget(self.trackmate_group)

        # --- Voronoi parameters (Nuclear Tracks mode) ---
        self.voronoi_group = QGroupBox("Voronoi Parameters")
        vor_layout = QVBoxLayout()

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.voronoi_method_combo = QComboBox()
        self.voronoi_method_combo.addItems(["Standard", "Lloyd's relaxation"])
        method_row.addWidget(self.voronoi_method_combo)
        vor_layout.addLayout(method_row)

        lloyd_row = QHBoxLayout()
        lloyd_row.addWidget(QLabel("Lloyd iterations:"))
        self.lloyd_iter_spin = QSpinBox()
        self.lloyd_iter_spin.setMinimum(0)
        self.lloyd_iter_spin.setMaximum(100)
        self.lloyd_iter_spin.setValue(10)
        lloyd_row.addWidget(self.lloyd_iter_spin)
        vor_layout.addLayout(lloyd_row)

        self.voronoi_group.setLayout(vor_layout)
        layout.addWidget(self.voronoi_group)

        # --- Segmentation tracking parameters ---
        self.tracking_group = QGroupBox("Tracking Parameters")
        track_layout = QVBoxLayout()

        iou_row = QHBoxLayout()
        iou_row.addWidget(QLabel("Min IoU:"))
        self.min_iou_spin = QDoubleSpinBox()
        self.min_iou_spin.setMinimum(0.0)
        self.min_iou_spin.setMaximum(1.0)
        self.min_iou_spin.setSingleStep(0.05)
        self.min_iou_spin.setValue(0.3)
        iou_row.addWidget(self.min_iou_spin)
        track_layout.addLayout(iou_row)

        self.tracking_group.setLayout(track_layout)
        layout.addWidget(self.tracking_group)

        # --- Both mode: matching threshold ---
        self.match_group = QGroupBox("Spot-Label Matching")
        match_layout = QVBoxLayout()

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Match threshold (px):"))
        self.match_threshold_spin = QDoubleSpinBox()
        self.match_threshold_spin.setMinimum(1.0)
        self.match_threshold_spin.setMaximum(100.0)
        self.match_threshold_spin.setSingleStep(1.0)
        self.match_threshold_spin.setValue(10.0)
        thresh_row.addWidget(self.match_threshold_spin)
        match_layout.addLayout(thresh_row)

        self.match_group.setLayout(match_layout)
        layout.addWidget(self.match_group)

        # --- Parameters ---
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout()

        px_row = QHBoxLayout()
        px_row.addWidget(QLabel("Pixel size (\u00b5m/px):"))
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
        self.trackmate_load_btn.clicked.connect(self._load_trackmate_xml)
        self.build_btn.clicked.connect(self._build_graph)
        self.tissue_spinner.valueChanged.connect(self._on_tissue_changed)
        self.remove_tissue_btn.clicked.connect(self._remove_current_tissue)
        self.voronoi_method_combo.currentIndexChanged.connect(self._update_lloyd_visibility)
        self._refresh_layers()

    # ------------------------------------------------------------------
    # Input mode switching
    # ------------------------------------------------------------------
    def _update_input_mode(self):
        mode = self.input_type_combo.currentText()
        is_seg = mode == INPUT_SEGMENTATION
        is_tracks = mode == INPUT_TRACKS
        is_both = mode == INPUT_BOTH

        self.file_group.setVisible(is_seg or is_both)
        self.layer_group.setVisible(is_tracks)
        self.trackmate_group.setVisible(is_tracks or is_both)
        self.voronoi_group.setVisible(is_tracks)
        self.tracking_group.setVisible(is_seg)
        self.match_group.setVisible(is_both)

        self._update_lloyd_visibility()
        self._refresh_layers()

    def _update_lloyd_visibility(self):
        is_lloyd = self.voronoi_method_combo.currentIndex() == 1
        self.lloyd_iter_spin.setEnabled(is_lloyd)

    def _refresh_layers(self):
        self.layer_combo.clear()
        import napari
        input_type = self.input_type_combo.currentText()
        for layer in self.viewer.layers:
            if input_type == INPUT_TRACKS and isinstance(layer, napari.layers.Points):
                self.layer_combo.addItem(layer.name)

    # ------------------------------------------------------------------
    # TrackMate XML loading
    # ------------------------------------------------------------------
    def _load_trackmate_xml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select TrackMate XML", "", "XML files (*.xml);;All files (*)"
        )
        if not path:
            return

        try:
            self._trackmate_data = parse_trackmate_xml(path)
            d = self._trackmate_data
            self.trackmate_info_label.setText(
                f"{Path(path).name}\n"
                f"{d.n_spots} spots, {d.n_tracks} tracks, "
                f"{len(d.spots_by_frame)} frames"
                + (f"\nImage: {d.image_shape[1]}x{d.image_shape[0]}" if d.image_shape else "")
            )
            # Auto-fill calibration if available
            if d.pixel_size_x is not None and not self.pixel_size_edit.text().strip():
                self.pixel_size_edit.setText(str(d.pixel_size_x))
            if d.time_interval is not None and not self.time_interval_edit.text().strip():
                self.time_interval_edit.setText(str(d.time_interval))
        except Exception as e:
            self.trackmate_info_label.setText(f"Error: {e}")
            self._trackmate_data = None

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
            f"{n} tissue(s) loaded \u2014 frames: {frame_counts}"
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

        # Voronoi parameters
        voronoi_method = (
            VoronoiMethod.LLOYD if self.voronoi_method_combo.currentIndex() == 1
            else VoronoiMethod.STANDARD
        )
        lloyd_iterations = self.lloyd_iter_spin.value()

        if input_type == INPUT_SEGMENTATION:
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
                min_iou=self.min_iou_spin.value(),
            )
        elif input_type == INPUT_TRACKS:
            if self._trackmate_data is not None:
                # Use TrackMate data
                self._thread = QThread()
                self._worker = DatasetBuildWorker(
                    input_type=input_type,
                    trackmate_data=self._trackmate_data,
                    pixel_size=pixel_size,
                    time_interval=time_interval,
                    condition=condition,
                    voronoi_method=voronoi_method,
                    lloyd_iterations=lloyd_iterations,
                )
            else:
                # Use Points layer
                layer_name = self.layer_combo.currentText()
                if not layer_name:
                    self.status_label.setText("No layer selected and no TrackMate XML loaded.")
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
                    voronoi_method=voronoi_method,
                    lloyd_iterations=lloyd_iterations,
                )
        else:
            # Both mode
            if not self._label_stacks:
                self.status_label.setText("No label files loaded.")
                return
            if self._trackmate_data is None:
                self.status_label.setText("No TrackMate XML loaded.")
                return
            self._thread = QThread()
            self._worker = DatasetBuildWorker(
                input_type=input_type,
                label_stacks=self._label_stacks,
                trackmate_data=self._trackmate_data,
                pixel_size=pixel_size,
                time_interval=time_interval,
                condition=condition,
                match_threshold=self.match_threshold_spin.value(),
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
