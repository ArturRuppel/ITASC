"""Napari dock widget for napariTissueGraph.

Supports two workflows:
- Single-tissue: staged pipeline with visual QC at each step.
- Batch: build multiple tissues at once, add all to dataset.

The dataset accumulates tissues and can be saved/loaded.
"""
import logging
from enum import auto, Enum
from pathlib import Path
from typing import Dict, List, Optional

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
    QSpinBox,
    QDoubleSpinBox,
    QScrollArea,
)
from qtpy.QtCore import Signal, QThread, QObject, Qt

from ..structures import TissueGraphDataset, TissueGraphTimeSeries, InputType, VoronoiMethod
from ..core.graph import (
    build_from_labels,
    build_from_labels_4d,
    build_from_tracks,
    build_from_tracks_4d,
    build_from_both,
    build_from_trackmate,
    extract_graphs_from_labels,
    extract_graphs_from_tracks,
    extract_graphs_from_trackmate,
    extract_graphs_from_both,
    assign_tracking_labels,
    assign_tracking_trackmate,
    has_tracking,
)
from ..core.trackmate import parse_trackmate_xml, TrackMateData
from ..core.topology import detect_t1_events, detect_all_t1_events
from ..core.io import save_dataset, load_dataset
from ..analysis.trajectories import build_edge_trajectories
from .visualization import (
    build_all_junction_lines,
    build_all_centroids,
    build_t1_markers,
    build_tracked_centroids,
    build_track_breaks,
    build_trajectory_lines,
)

logger = logging.getLogger(__name__)

INPUT_SEGMENTATION = "Segmentation Labels"
INPUT_TRACKS = "Nuclear Tracks"
INPUT_BOTH = "Both (Labels + Tracks)"


class PipelineStage(Enum):
    IDLE = auto()
    GRAPHS_BUILT = auto()
    TRACKED = auto()
    ANALYZED = auto()


# ------------------------------------------------------------------
# Workers
# ------------------------------------------------------------------

class GraphExtractWorker(QObject):
    """Stage 1: extract per-frame graphs (no tracking)."""

    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(Exception)

    def __init__(self, input_type, label_stack=None, track_positions=None,
                 trackmate_data=None, pixel_size=None, time_interval=None,
                 voronoi_method=VoronoiMethod.STANDARD,
                 lloyd_iterations=10, lloyd_tol=0.1):
        super().__init__()
        self.input_type = input_type
        self.label_stack = label_stack
        self.track_positions = track_positions
        self.trackmate_data = trackmate_data
        self.pixel_size = pixel_size
        self.time_interval = time_interval
        self.voronoi_method = voronoi_method
        self.lloyd_iterations = lloyd_iterations
        self.lloyd_tol = lloyd_tol

    def run(self):
        try:
            self.progress.emit(10, "Extracting graphs...")

            if self.input_type == INPUT_SEGMENTATION:
                series = extract_graphs_from_labels(
                    self.label_stack,
                    pixel_size=self.pixel_size,
                    time_interval=self.time_interval,
                )
            elif self.input_type == INPUT_TRACKS:
                if self.trackmate_data is not None:
                    series = extract_graphs_from_trackmate(
                        self.trackmate_data,
                        pixel_size=self.pixel_size,
                        time_interval=self.time_interval,
                        method=self.voronoi_method,
                        lloyd_iterations=self.lloyd_iterations,
                        lloyd_tol=self.lloyd_tol,
                    )
                else:
                    series = extract_graphs_from_tracks(
                        self.track_positions,
                        pixel_size=self.pixel_size,
                        time_interval=self.time_interval,
                        method=self.voronoi_method,
                        lloyd_iterations=self.lloyd_iterations,
                        lloyd_tol=self.lloyd_tol,
                    )
            else:
                series = extract_graphs_from_both(
                    self.label_stack,
                    pixel_size=self.pixel_size,
                    time_interval=self.time_interval,
                )

            self.progress.emit(100, "Graphs extracted.")
            self.finished.emit(series)
        except Exception as e:
            self.error.emit(e)


class TrackingWorker(QObject):
    """Stage 2: assign tracking IDs."""

    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(Exception)

    def __init__(self, series, input_type, label_stack=None,
                 trackmate_data=None, min_iou=0.3, match_threshold=10.0):
        super().__init__()
        self.series = series
        self.input_type = input_type
        self.label_stack = label_stack
        self.trackmate_data = trackmate_data
        self.min_iou = min_iou
        self.match_threshold = match_threshold

    def run(self):
        try:
            self.progress.emit(10, "Running tracking...")

            if self.input_type == INPUT_SEGMENTATION:
                assign_tracking_labels(self.series, self.label_stack, min_iou=self.min_iou)
            elif self.input_type == INPUT_TRACKS:
                if self.trackmate_data is not None:
                    assign_tracking_trackmate(self.series, self.trackmate_data)
                # Points-only tracks mode has no separate tracking source
            else:
                assign_tracking_trackmate(
                    self.series, self.trackmate_data,
                    match_threshold=self.match_threshold,
                )

            self.progress.emit(100, "Tracking complete.")
            self.finished.emit(self.series)
        except Exception as e:
            self.error.emit(e)


class AnalysisWorker(QObject):
    """Stage 3: T1 detection + edge trajectories."""

    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(Exception)

    def __init__(self, series):
        super().__init__()
        self.series = series

    def run(self):
        try:
            self.progress.emit(10, "Detecting T1 events...")
            events = detect_t1_events(self.series)

            self.progress.emit(60, "Building edge trajectories...")
            build_edge_trajectories(self.series, events)

            self.progress.emit(100, "Analysis complete.")
            self.finished.emit(self.series)
        except Exception as e:
            self.error.emit(e)


class BatchBuildWorker(QObject):
    """Build multiple tissues in a background thread, returns a list."""

    progress = Signal(int, str)
    finished = Signal(object)  # emits list of TissueGraphTimeSeries
    error = Signal(Exception)

    def __init__(
        self,
        input_type: str,
        label_stacks=None,
        track_positions=None,
        trackmate_data_list=None,
        pixel_size=None,
        time_interval=None,
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
        self.trackmate_data_list = trackmate_data_list
        self.pixel_size = pixel_size
        self.time_interval = time_interval
        self.voronoi_method = voronoi_method
        self.lloyd_iterations = lloyd_iterations
        self.lloyd_tol = lloyd_tol
        self.min_iou = min_iou
        self.match_threshold = match_threshold

    def run(self):
        try:
            results = []

            if self.input_type == INPUT_SEGMENTATION:
                n = len(self.label_stacks)
                for i, stack in enumerate(self.label_stacks):
                    self.progress.emit(
                        int((i / n) * 80),
                        f"Building tissue {i + 1}/{n}...",
                    )
                    series = build_from_labels(
                        stack,
                        pixel_size=self.pixel_size,
                        time_interval=self.time_interval,
                        min_iou=self.min_iou,
                    )
                    detect_t1_events(series)
                    build_edge_trajectories(series, series.t1_events)
                    results.append(series)

            elif self.input_type == INPUT_TRACKS:
                if self.trackmate_data_list:
                    n = len(self.trackmate_data_list)
                    for i, tm_data in enumerate(self.trackmate_data_list):
                        self.progress.emit(
                            int((i / n) * 80),
                            f"Building tissue {i + 1}/{n}...",
                        )
                        series = build_from_trackmate(
                            tm_data,
                            pixel_size=self.pixel_size,
                            time_interval=self.time_interval,
                            method=self.voronoi_method,
                            lloyd_iterations=self.lloyd_iterations,
                            lloyd_tol=self.lloyd_tol,
                        )
                        detect_t1_events(series)
                        build_edge_trajectories(series, series.t1_events)
                        results.append(series)
                else:
                    # Points layer — single tissue in batch mode
                    self.progress.emit(10, "Building from points...")
                    series = build_from_tracks(
                        self.track_positions,
                        pixel_size=self.pixel_size,
                        time_interval=self.time_interval,
                        method=self.voronoi_method,
                        lloyd_iterations=self.lloyd_iterations,
                        lloyd_tol=self.lloyd_tol,
                    )
                    detect_t1_events(series)
                    build_edge_trajectories(series, series.t1_events)
                    results.append(series)

            else:
                # Both mode: pair label stacks with TrackMate files
                n = len(self.label_stacks)
                for i in range(n):
                    self.progress.emit(
                        int((i / n) * 80),
                        f"Building tissue {i + 1}/{n}...",
                    )
                    series = build_from_both(
                        self.label_stacks[i],
                        self.trackmate_data_list[i],
                        pixel_size=self.pixel_size,
                        time_interval=self.time_interval,
                        match_threshold=self.match_threshold,
                    )
                    detect_t1_events(series)
                    build_edge_trajectories(series, series.t1_events)
                    results.append(series)

            self.finished.emit(results)
        except Exception as e:
            self.error.emit(e)


class IOWorker(QObject):
    """Save or load a dataset in a background thread."""

    finished = Signal(object)  # emits TissueGraphDataset (load) or None (save)
    error = Signal(Exception)

    def __init__(self, mode: str, path: str, dataset=None):
        super().__init__()
        self.mode = mode
        self.path = path
        self.dataset = dataset

    def run(self):
        try:
            if self.mode == "save":
                save_dataset(self.dataset, self.path)
                self.finished.emit(None)
            else:
                ds = load_dataset(self.path)
                self.finished.emit(ds)
        except Exception as e:
            self.error.emit(e)


# ------------------------------------------------------------------
# Widget
# ------------------------------------------------------------------

class TissueGraphWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.dataset: Optional[TissueGraphDataset] = None
        self._label_stacks: List[np.ndarray] = []
        self._file_names: List[str] = []
        self._trackmate_data_list: List[TrackMateData] = []

        # Pipeline state
        self._pipeline_stage = PipelineStage.IDLE
        self._preview_series: Optional[TissueGraphTimeSeries] = None
        self._current_label_stack: Optional[np.ndarray] = None
        self._current_trackmate_data: Optional[TrackMateData] = None
        self._stage_layers: Dict[int, list] = {1: [], 2: [], 3: []}

        # Dataset inspection layers
        self._inspect_junction_layer = None
        self._inspect_centroid_layer = None
        self._inspect_t1_layer = None
        self._thread = None
        self._worker = None

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer_layout.addWidget(scroll)

        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        scroll.setWidget(container)

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
        self.file_list.setMaximumHeight(100)
        file_layout.addWidget(self.file_list)
        self.file_info_label = QLabel("")
        file_layout.addWidget(self.file_info_label)
        self.file_group.setLayout(file_layout)
        layout.addWidget(self.file_group)

        # --- TrackMate XML loading (multi-file) ---
        self.trackmate_group = QGroupBox("TrackMate XML")
        tm_layout = QVBoxLayout()
        self.trackmate_load_btn = QPushButton("Load TrackMate XML(s)...")
        tm_layout.addWidget(self.trackmate_load_btn)
        self.trackmate_file_list = QListWidget()
        self.trackmate_file_list.setMaximumHeight(80)
        tm_layout.addWidget(self.trackmate_file_list)
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

        # --- Pipeline: Stage 1 ---
        self.stage1_group = QGroupBox("Stage 1: Graph Extraction")
        s1_layout = QVBoxLayout()
        self.extract_btn = QPushButton("Extract Graphs")
        s1_layout.addWidget(self.extract_btn)
        self.stage1_info = QLabel("")
        self.stage1_info.setWordWrap(True)
        s1_layout.addWidget(self.stage1_info)
        self.stage1_group.setLayout(s1_layout)
        layout.addWidget(self.stage1_group)

        # --- Pipeline: Stage 2 ---
        self.stage2_group = QGroupBox("Stage 2: Cell Tracking")
        s2_layout = QVBoxLayout()
        self.track_btn = QPushButton("Run Tracking")
        s2_layout.addWidget(self.track_btn)
        self.stage2_info = QLabel("")
        self.stage2_info.setWordWrap(True)
        s2_layout.addWidget(self.stage2_info)
        self.stage2_group.setLayout(s2_layout)
        layout.addWidget(self.stage2_group)

        # --- Pipeline: Stage 3 ---
        self.stage3_group = QGroupBox("Stage 3: T1 + Edge Tracking")
        s3_layout = QVBoxLayout()
        self.analyze_btn = QPushButton("Run Analysis")
        s3_layout.addWidget(self.analyze_btn)
        self.stage3_info = QLabel("")
        self.stage3_info.setWordWrap(True)
        s3_layout.addWidget(self.stage3_info)
        self.stage3_group.setLayout(s3_layout)
        layout.addWidget(self.stage3_group)

        # --- Pipeline: Add / Discard ---
        self.finalize_group = QGroupBox("Add / Discard")
        fin_layout = QHBoxLayout()
        self.add_to_dataset_btn = QPushButton("Add to Dataset")
        self.discard_btn = QPushButton("Discard")
        fin_layout.addWidget(self.add_to_dataset_btn)
        fin_layout.addWidget(self.discard_btn)
        self.finalize_group.setLayout(fin_layout)
        layout.addWidget(self.finalize_group)

        # --- Batch mode ---
        batch_group = QGroupBox("Batch")
        batch_layout = QVBoxLayout()
        self.build_batch_btn = QPushButton("Build All (Batch)")
        batch_layout.addWidget(self.build_batch_btn)
        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # --- Dataset section ---
        self.dataset_group = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout()

        self.dataset_summary_label = QLabel("No dataset")
        self.dataset_summary_label.setWordWrap(True)
        dataset_layout.addWidget(self.dataset_summary_label)

        # Tissue inspector
        tissue_row = QHBoxLayout()
        tissue_row.addWidget(QLabel("Tissue:"))
        self.tissue_spinner = QSpinBox()
        self.tissue_spinner.setMinimum(0)
        self.tissue_spinner.setMaximum(0)
        tissue_row.addWidget(self.tissue_spinner)
        self.tissue_info_label = QLabel("")
        tissue_row.addWidget(self.tissue_info_label)
        dataset_layout.addLayout(tissue_row)

        self.show_tissue_btn = QPushButton("Show Tissue")
        self.remove_tissue_btn = QPushButton("Remove Tissue")
        tissue_btn_row = QHBoxLayout()
        tissue_btn_row.addWidget(self.show_tissue_btn)
        tissue_btn_row.addWidget(self.remove_tissue_btn)
        dataset_layout.addLayout(tissue_btn_row)

        # Save/Load
        io_row = QHBoxLayout()
        self.save_btn = QPushButton("Save Dataset...")
        self.load_dataset_btn = QPushButton("Load Dataset...")
        self.new_dataset_btn = QPushButton("New")
        io_row.addWidget(self.save_btn)
        io_row.addWidget(self.load_dataset_btn)
        io_row.addWidget(self.new_dataset_btn)
        dataset_layout.addLayout(io_row)

        self.dataset_group.setLayout(dataset_layout)
        layout.addWidget(self.dataset_group)

        layout.addStretch()

        # Set initial visibility and button state
        self._update_input_mode()
        self._update_pipeline_buttons()

    def _connect_signals(self):
        self.input_type_combo.currentIndexChanged.connect(self._update_input_mode)
        self.refresh_btn.clicked.connect(self._refresh_layers)
        self.load_btn.clicked.connect(self._load_label_files)
        self.trackmate_load_btn.clicked.connect(self._load_trackmate_xml)
        self.voronoi_method_combo.currentIndexChanged.connect(self._update_lloyd_visibility)

        # Pipeline
        self.extract_btn.clicked.connect(self._run_extract)
        self.track_btn.clicked.connect(self._run_tracking)
        self.analyze_btn.clicked.connect(self._run_analysis)
        self.add_to_dataset_btn.clicked.connect(self._add_preview_to_dataset)
        self.discard_btn.clicked.connect(self._discard_pipeline)

        # Batch
        self.build_batch_btn.clicked.connect(self._build_batch)

        # Dataset
        self.show_tissue_btn.clicked.connect(self._show_selected_tissue)
        self.remove_tissue_btn.clicked.connect(self._remove_current_tissue)
        self.save_btn.clicked.connect(self._save_dataset)
        self.load_dataset_btn.clicked.connect(self._load_dataset)
        self.new_dataset_btn.clicked.connect(self._new_dataset)

        self._refresh_layers()

    # ------------------------------------------------------------------
    # Pipeline stage gating
    # ------------------------------------------------------------------
    def _update_pipeline_buttons(self):
        stage = self._pipeline_stage
        self.extract_btn.setEnabled(stage == PipelineStage.IDLE)
        self.track_btn.setEnabled(stage == PipelineStage.GRAPHS_BUILT)
        self.analyze_btn.setEnabled(stage == PipelineStage.TRACKED)
        self.add_to_dataset_btn.setEnabled(stage == PipelineStage.ANALYZED)
        self.discard_btn.setEnabled(stage != PipelineStage.IDLE)

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
    # TrackMate XML loading (multi-file)
    # ------------------------------------------------------------------
    def _load_trackmate_xml(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select TrackMate XML(s)", "", "XML files (*.xml);;All files (*)"
        )
        if not files:
            return

        self._trackmate_data_list = []
        self.trackmate_file_list.clear()
        errors = []

        for path in sorted(files):
            try:
                tm_data = parse_trackmate_xml(path)
                self._trackmate_data_list.append(tm_data)
                self.trackmate_file_list.addItem(
                    f"{Path(path).name}  ({tm_data.n_spots} spots, "
                    f"{tm_data.n_tracks} tracks, {len(tm_data.spots_by_frame)} frames)"
                )
            except Exception as e:
                errors.append(f"{Path(path).name}: {e}")

        n = len(self._trackmate_data_list)
        total_spots = sum(d.n_spots for d in self._trackmate_data_list)
        info = f"{n} file(s), {total_spots} spots total"
        if errors:
            info += f"\nErrors: {'; '.join(errors)}"
        self.trackmate_info_label.setText(info)

        # Auto-fill calibration from first file
        if self._trackmate_data_list:
            d = self._trackmate_data_list[0]
            if d.pixel_size_x is not None and not self.pixel_size_edit.text().strip():
                self.pixel_size_edit.setText(str(d.pixel_size_x))
            if d.time_interval is not None and not self.time_interval_edit.text().strip():
                self.time_interval_edit.setText(str(d.time_interval))

    # ------------------------------------------------------------------
    # Multi-file label loading
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
            self.file_list.addItem(
                f"{name}  ({stack.shape[0]} frames, {stack.shape[1]}x{stack.shape[2]})"
            )

        n = len(self._label_stacks)
        frame_counts = [s.shape[0] for s in self._label_stacks]
        self.file_info_label.setText(f"{n} tissue(s) loaded \u2014 frames: {frame_counts}")

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def _selected_label_index(self) -> int:
        row = self.file_list.currentRow()
        return row if row >= 0 else 0

    def _selected_trackmate_index(self) -> int:
        row = self.trackmate_file_list.currentRow()
        return row if row >= 0 else 0

    # ------------------------------------------------------------------
    # Stage 1: Extract graphs
    # ------------------------------------------------------------------
    def _run_extract(self):
        input_type = self.input_type_combo.currentText()
        pixel_size = self._parse_float(self.pixel_size_edit.text())
        time_interval = self._parse_float(self.time_interval_edit.text())
        voronoi_method = (
            VoronoiMethod.LLOYD if self.voronoi_method_combo.currentIndex() == 1
            else VoronoiMethod.STANDARD
        )

        kwargs = dict(
            input_type=input_type,
            pixel_size=pixel_size,
            time_interval=time_interval,
            voronoi_method=voronoi_method,
            lloyd_iterations=self.lloyd_iter_spin.value(),
        )

        # Collect inputs and validate before discarding previous pipeline
        label_stack = None
        trackmate_data = None

        if input_type == INPUT_SEGMENTATION:
            if not self._label_stacks:
                self.status_label.setText("No label files loaded.")
                return
            idx = self._selected_label_index()
            kwargs["label_stack"] = self._label_stacks[idx]
            label_stack = self._label_stacks[idx]
        elif input_type == INPUT_TRACKS:
            if self._trackmate_data_list:
                idx = self._selected_trackmate_index()
                kwargs["trackmate_data"] = self._trackmate_data_list[idx]
                trackmate_data = self._trackmate_data_list[idx]
            else:
                layer_name = self.layer_combo.currentText()
                if not layer_name:
                    self.status_label.setText("No layer selected and no TrackMate XML loaded.")
                    return
                kwargs["track_positions"] = self.viewer.layers[layer_name].data
        else:
            # Both
            if not self._label_stacks:
                self.status_label.setText("No label files loaded.")
                return
            if not self._trackmate_data_list:
                self.status_label.setText("No TrackMate XML loaded.")
                return
            kwargs["label_stack"] = self._label_stacks[self._selected_label_index()]
            kwargs["trackmate_data"] = self._trackmate_data_list[self._selected_trackmate_index()]
            label_stack = kwargs["label_stack"]
            trackmate_data = kwargs["trackmate_data"]

        # Discard previous pipeline, then set references for Stage 2
        self._discard_pipeline()
        self._current_label_stack = label_stack
        self._current_trackmate_data = trackmate_data

        self._run_worker(GraphExtractWorker(**kwargs), self._on_extract_finished)

    def _on_extract_finished(self, series):
        self._preview_series = series
        self._pipeline_stage = PipelineStage.GRAPHS_BUILT
        self._finish_worker()

        # Summary
        total_cells = sum(len(f.cells) for f in series.frames.values())
        total_junctions = sum(len(f.junctions) for f in series.frames.values())
        self.stage1_info.setText(
            f"{series.num_frames} frames, {total_cells} cells, {total_junctions} junctions"
        )
        self.status_label.setText("Stage 1 complete. Inspect junctions & centroids, then run tracking.")

        # Show junction lines (length-colored) + centroids (yellow)
        self._remove_stage_layers(1)
        lines, colors = build_all_junction_lines(series)
        if lines:
            layer = self.viewer.add_shapes(
                lines, shape_type="path", edge_color=colors,
                edge_width=2, name="[Pipeline] Junctions",
            )
            self._stage_layers[1].append(layer)

        centroids = build_all_centroids(series)
        if len(centroids) > 0:
            layer = self.viewer.add_points(
                centroids, size=5, face_color="yellow",
                name="[Pipeline] Cell Centroids",
            )
            self._stage_layers[1].append(layer)

        self._update_pipeline_buttons()

    # ------------------------------------------------------------------
    # Stage 2: Tracking
    # ------------------------------------------------------------------
    def _run_tracking(self):
        if self._preview_series is None:
            return

        input_type = self.input_type_combo.currentText()
        kwargs = dict(
            series=self._preview_series,
            input_type=input_type,
            label_stack=self._current_label_stack,
            trackmate_data=self._current_trackmate_data,
            min_iou=self.min_iou_spin.value(),
            match_threshold=self.match_threshold_spin.value(),
        )

        self._run_worker(TrackingWorker(**kwargs), self._on_tracking_finished)

    def _on_tracking_finished(self, series):
        self._preview_series = series
        self._pipeline_stage = PipelineStage.TRACKED
        self._finish_worker()

        # Count unique track_ids, births, deaths
        all_track_ids = set()
        for frame in series.frames.values():
            for cell in frame.cells.values():
                if cell.track_id is not None:
                    all_track_ids.add(cell.track_id)

        positions, types = build_track_breaks(series)
        n_births = types.count("birth")
        n_deaths = types.count("death")

        self.stage2_info.setText(
            f"{len(all_track_ids)} tracks, {n_births} births, {n_deaths} deaths"
        )
        self.status_label.setText("Stage 2 complete. Inspect track colors, then run analysis.")

        # Replace yellow centroids with track-colored centroids
        self._remove_stage_layers(2)

        tracked_pos, tracked_colors, _ = build_tracked_centroids(series)
        if len(tracked_pos) > 0:
            # Remove stage 1 centroid layer, keep junction layer
            self._stage_layers[1] = [
                l for l in self._stage_layers[1]
                if l in self.viewer.layers and "Centroid" not in l.name
            ]
            # Remove old centroid layer from viewer
            for l in list(self.viewer.layers):
                if l.name == "[Pipeline] Cell Centroids":
                    self.viewer.layers.remove(l)

            layer = self.viewer.add_points(
                tracked_pos, size=5, face_color=tracked_colors,
                name="[Pipeline] Tracked Centroids",
            )
            self._stage_layers[2].append(layer)

        # Add track break markers
        if len(positions) > 0:
            break_colors = [
                [0.0, 1.0, 0.0, 1.0] if t == "birth" else [1.0, 0.0, 0.0, 1.0]
                for t in types
            ]
            layer = self.viewer.add_points(
                positions, size=10, face_color=break_colors,
                symbol="diamond", name="[Pipeline] Track Breaks",
            )
            self._stage_layers[2].append(layer)

        self._update_pipeline_buttons()

    # ------------------------------------------------------------------
    # Stage 3: T1 + Edge trajectories
    # ------------------------------------------------------------------
    def _run_analysis(self):
        if self._preview_series is None:
            return
        self._run_worker(AnalysisWorker(self._preview_series), self._on_analysis_finished)

    def _on_analysis_finished(self, series):
        self._preview_series = series
        self._pipeline_stage = PipelineStage.ANALYZED
        self._finish_worker()

        n_t1 = len(series.t1_events)
        n_trajs = len(series.edge_trajectories)

        self.stage3_info.setText(
            f"{n_t1} T1 events, {n_trajs} edge trajectories"
        )
        self.status_label.setText("Stage 3 complete. Inspect T1 events & trajectories, then add to dataset.")

        # Replace length-colored junctions with trajectory-colored junctions
        self._remove_stage_layers(3)

        # Remove stage 1 junction layer
        for l in list(self.viewer.layers):
            if l.name == "[Pipeline] Junctions":
                self.viewer.layers.remove(l)
        self._stage_layers[1] = [
            l for l in self._stage_layers[1]
            if l in self.viewer.layers
        ]

        traj_lines, traj_colors = build_trajectory_lines(series)
        if traj_lines:
            layer = self.viewer.add_shapes(
                traj_lines, shape_type="path", edge_color=traj_colors,
                edge_width=2, name="[Pipeline] Trajectories",
            )
            self._stage_layers[3].append(layer)

        # T1 markers
        t1_positions = build_t1_markers(series.t1_events)
        if len(t1_positions) > 0:
            layer = self.viewer.add_points(
                t1_positions, size=12, face_color="red",
                symbol="star", name="[Pipeline] T1 Events",
            )
            self._stage_layers[3].append(layer)

        self._update_pipeline_buttons()

    # ------------------------------------------------------------------
    # Add / Discard
    # ------------------------------------------------------------------
    def _add_preview_to_dataset(self):
        if self._preview_series is None:
            return
        self._ensure_dataset()
        tid = self.dataset.add_tissue(self._preview_series)
        self._remove_all_stage_layers()
        self._preview_series = None
        self._current_label_stack = None
        self._current_trackmate_data = None
        self._pipeline_stage = PipelineStage.IDLE
        self._update_pipeline_buttons()
        self._clear_stage_info()
        self.status_label.setText(f"Added tissue {tid} to dataset.")
        self._update_dataset_ui()

    def _discard_pipeline(self):
        self._remove_all_stage_layers()
        self._preview_series = None
        self._current_label_stack = None
        self._current_trackmate_data = None
        self._pipeline_stage = PipelineStage.IDLE
        self._update_pipeline_buttons()
        self._clear_stage_info()

    def _clear_stage_info(self):
        self.stage1_info.setText("")
        self.stage2_info.setText("")
        self.stage3_info.setText("")

    # ------------------------------------------------------------------
    # Batch build (monolithic, no per-stage QC)
    # ------------------------------------------------------------------
    def _build_batch(self):
        input_type = self.input_type_combo.currentText()
        pixel_size = self._parse_float(self.pixel_size_edit.text())
        time_interval = self._parse_float(self.time_interval_edit.text())
        voronoi_method = (
            VoronoiMethod.LLOYD if self.voronoi_method_combo.currentIndex() == 1
            else VoronoiMethod.STANDARD
        )

        kwargs = dict(
            input_type=input_type,
            pixel_size=pixel_size,
            time_interval=time_interval,
            voronoi_method=voronoi_method,
            lloyd_iterations=self.lloyd_iter_spin.value(),
            min_iou=self.min_iou_spin.value(),
            match_threshold=self.match_threshold_spin.value(),
        )

        if input_type == INPUT_SEGMENTATION:
            if not self._label_stacks:
                self.status_label.setText("No label files loaded.")
                return
            kwargs["label_stacks"] = self._label_stacks
        elif input_type == INPUT_TRACKS:
            if self._trackmate_data_list:
                kwargs["trackmate_data_list"] = self._trackmate_data_list
            else:
                layer_name = self.layer_combo.currentText()
                if not layer_name:
                    self.status_label.setText("No layer selected and no TrackMate XML loaded.")
                    return
                kwargs["track_positions"] = self.viewer.layers[layer_name].data
        else:
            # Both mode: need matching counts
            if not self._label_stacks:
                self.status_label.setText("No label files loaded.")
                return
            if not self._trackmate_data_list:
                self.status_label.setText("No TrackMate XML loaded.")
                return
            if len(self._label_stacks) != len(self._trackmate_data_list):
                self.status_label.setText(
                    f"Mismatch: {len(self._label_stacks)} label files "
                    f"vs {len(self._trackmate_data_list)} TrackMate files."
                )
                return
            kwargs["label_stacks"] = self._label_stacks
            kwargs["trackmate_data_list"] = self._trackmate_data_list

        self._run_worker(BatchBuildWorker(**kwargs), self._on_batch_finished)

    def _on_batch_finished(self, series_list):
        self._ensure_dataset()
        for series in series_list:
            self.dataset.add_tissue(series)

        self._finish_worker()
        n = len(series_list)
        self.status_label.setText(f"Added {n} tissue(s) to dataset.")
        self._update_dataset_ui()

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------
    def _ensure_dataset(self):
        if self.dataset is None:
            self.dataset = TissueGraphDataset(
                condition=self.condition_edit.text().strip(),
                pixel_size=self._parse_float(self.pixel_size_edit.text()),
                time_interval=self._parse_float(self.time_interval_edit.text()),
            )

    def _update_dataset_ui(self):
        ds = self.dataset
        if ds is None or ds.n_tissues == 0:
            self.dataset_summary_label.setText("No dataset" if ds is None else "Dataset empty")
            self.tissue_spinner.setMinimum(0)
            self.tissue_spinner.setMaximum(0)
            return

        parts = [f"{ds.n_tissues} tissue(s)"]
        if ds.condition:
            parts.append(f"Condition: {ds.condition}")
        for tid in ds.tissue_ids:
            s = ds.tissues[tid]
            n_t1 = len(s.t1_events)
            parts.append(f"  T{tid}: {s.num_frames} frames, {n_t1} T1 events")
        self.dataset_summary_label.setText("\n".join(parts))

        ids = ds.tissue_ids
        self.tissue_spinner.setMinimum(min(ids))
        self.tissue_spinner.setMaximum(max(ids))

    def _show_selected_tissue(self):
        if self.dataset is None:
            return
        tid = self.tissue_spinner.value()
        if tid not in self.dataset.tissue_ids:
            self.status_label.setText(f"Tissue {tid} does not exist.")
            return
        self._remove_inspect_layers()
        series = self.dataset.tissues[tid]
        n_t1 = len(series.t1_events)
        n_trajs = len(series.edge_trajectories)
        total_cells = sum(len(f.cells) for f in series.frames.values())
        total_junctions = sum(len(f.junctions) for f in series.frames.values())
        self.tissue_info_label.setText(
            f"{series.num_frames}f, {total_cells}c, {total_junctions}j, {n_t1} T1"
        )
        self.status_label.setText(
            f"Tissue {tid}: {series.num_frames} frames, "
            f"{total_cells} cells, {total_junctions} junctions, "
            f"{n_t1} T1 events, {n_trajs} edge trajectories"
        )
        self._add_inspect_layers(series)

    def _remove_current_tissue(self):
        if self.dataset is None:
            return
        tid = self.tissue_spinner.value()
        if tid not in self.dataset.tissue_ids:
            self.status_label.setText(f"Tissue {tid} does not exist.")
            return
        self.dataset.remove_tissue(tid)
        self._remove_inspect_layers()
        self.status_label.setText(f"Removed tissue {tid}.")
        self._update_dataset_ui()

    def _new_dataset(self):
        self._remove_inspect_layers()
        self._discard_pipeline()
        self.dataset = None
        self._update_dataset_ui()
        self.status_label.setText("Created new dataset.")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def _save_dataset(self):
        if self.dataset is None or self.dataset.n_tissues == 0:
            self.status_label.setText("No dataset to save.")
            return
        path = QFileDialog.getExistingDirectory(self, "Save Dataset To")
        if not path:
            return
        self._run_worker(
            IOWorker("save", path, self.dataset), self._on_save_finished
        )

    def _on_save_finished(self, _):
        self._finish_worker()
        self.status_label.setText("Dataset saved.")

    def _load_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Load Dataset From")
        if not path:
            return
        self._run_worker(
            IOWorker("load", path), self._on_load_finished
        )

    def _on_load_finished(self, dataset):
        self.dataset = dataset
        self._finish_worker()

        # Populate UI from loaded metadata
        if dataset.condition:
            self.condition_edit.setText(dataset.condition)
        if dataset.pixel_size is not None:
            self.pixel_size_edit.setText(str(dataset.pixel_size))
        if dataset.time_interval is not None:
            self.time_interval_edit.setText(str(dataset.time_interval))

        self._update_dataset_ui()
        self.status_label.setText(
            f"Loaded dataset: {dataset.n_tissues} tissue(s), "
            f"condition: {dataset.condition or '(none)'}"
        )

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------
    def _run_worker(self, worker, on_finished):
        self._thread = QThread()
        self._worker = worker
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        if hasattr(self._worker, "progress"):
            self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self.extract_btn.setEnabled(False)
        self.track_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.build_batch_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Working...")

        self._thread.start()

    def _finish_worker(self):
        """Re-enable buttons and hide progress bar after worker completes."""
        self.progress_bar.setVisible(False)
        self.build_batch_btn.setEnabled(True)
        self._update_pipeline_buttons()

    def _on_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _on_error(self, exc):
        self.progress_bar.setVisible(False)
        self.build_batch_btn.setEnabled(True)
        self._update_pipeline_buttons()
        self.status_label.setText(f"Error: {exc}")
        logger.exception("Worker failed", exc_info=exc)

    # ------------------------------------------------------------------
    # Napari layer management
    # ------------------------------------------------------------------
    def _remove_stage_layers(self, stage: int):
        for layer in self._stage_layers[stage]:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self._stage_layers[stage] = []

    def _remove_all_stage_layers(self):
        for stage in (1, 2, 3):
            self._remove_stage_layers(stage)

    def _remove_inspect_layers(self):
        for layer in (
            self._inspect_junction_layer,
            self._inspect_centroid_layer,
            self._inspect_t1_layer,
        ):
            if layer is not None and layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self._inspect_junction_layer = None
        self._inspect_centroid_layer = None
        self._inspect_t1_layer = None

    def _add_inspect_layers(self, series: TissueGraphTimeSeries):
        self._remove_inspect_layers()
        j, c, t = self._make_layers(series, prefix="")
        self._inspect_junction_layer = j
        self._inspect_centroid_layer = c
        self._inspect_t1_layer = t

    def _make_layers(self, series: TissueGraphTimeSeries, prefix: str = ""):
        """Create junction, centroid, and T1 layers for a tissue."""
        junction_layer = None
        centroid_layer = None
        t1_layer = None

        lines, colors = build_all_junction_lines(series)
        if lines:
            junction_layer = self.viewer.add_shapes(
                lines,
                shape_type="path",
                edge_color=colors,
                edge_width=2,
                name=f"{prefix}Junctions",
            )

        centroids = build_all_centroids(series)
        if len(centroids) > 0:
            centroid_layer = self.viewer.add_points(
                centroids,
                size=5,
                face_color="yellow",
                name=f"{prefix}Cell Centroids",
            )

        t1_positions = build_t1_markers(series.t1_events)
        if len(t1_positions) > 0:
            t1_layer = self.viewer.add_points(
                t1_positions,
                size=12,
                face_color="red",
                symbol="star",
                name=f"{prefix}T1 Events",
            )

        return junction_layer, centroid_layer, t1_layer

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
