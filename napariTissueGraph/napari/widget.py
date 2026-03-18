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
    QListWidgetItem,
    QLineEdit,
    QFileDialog,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QScrollArea,
    QCheckBox,
)
from qtpy.QtCore import Signal, QThread, QObject, Qt, QTimer

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
    apply_track_map,
    has_tracking,
)
from ..core.label_tracking import assign_track_ids
from ..core.trackmate import parse_trackmate_xml, TrackMateData
from ..core.topology import detect_t1_events, detect_all_t1_events
from ..core.io import save_dataset, load_dataset
from ..analysis.trajectories import build_edge_trajectories, filter_trajectories
from .visualization import (
    build_all_junction_lines,
    build_all_centroids,
    build_t1_markers,
    build_tracked_centroids,
    build_tracked_labels,
    build_track_breaks,
    build_trajectory_lines,
    build_trajectory_lines_with_features,
)
from ..analysis.tagging import (
    tag_trajectory,
    untag_trajectory,
    tag_junction,
    untag_junction,
    get_all_tags,
    clear_tag,
)

logger = logging.getLogger(__name__)

INPUT_SEGMENTATION = "Segmentation Labels"
INPUT_TRACKS = "Nuclear Tracks"
INPUT_BOTH = "Both (Labels + Tracks)"


class PipelineStage(Enum):
    IDLE = auto()
    STAGE1_DONE = auto()
    STAGE2_DONE = auto()
    STAGE3_DONE = auto()


# ------------------------------------------------------------------
# Workers
# ------------------------------------------------------------------

class CellTrackingWorker(QObject):
    """Track cells via IoU matching on label stack (segmentation Stage 1)."""

    progress = Signal(int, str)
    finished = Signal(object)  # emits track_map dict
    error = Signal(Exception)

    def __init__(self, label_stack, min_iou=0.3, max_area_change=0.0):
        super().__init__()
        self.label_stack = label_stack
        self.min_iou = min_iou
        self.max_area_change = float('inf') if max_area_change == 0 else max_area_change

    def run(self):
        try:
            self.progress.emit(10, "Running cell tracking...")
            track_map = assign_track_ids(
                self.label_stack,
                min_iou=self.min_iou,
                max_area_change=self.max_area_change,
            )
            self.progress.emit(100, "Cell tracking complete.")
            self.finished.emit(track_map)
        except Exception as e:
            self.error.emit(e)


class GraphExtractWorker(QObject):
    """Extract per-frame graphs (no tracking)."""

    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(Exception)

    def __init__(self, input_type, label_stack=None, track_positions=None,
                 trackmate_data=None, pixel_size=None, time_interval=None,
                 voronoi_method=VoronoiMethod.STANDARD,
                 lloyd_iterations=10, lloyd_tol=0.1,
                 dilation_radius=1, min_overlap_pixels=5,
                 min_edge_length=0.0, filter_isolated=True):
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
        self.dilation_radius = dilation_radius
        self.min_overlap_pixels = min_overlap_pixels
        self.min_edge_length = min_edge_length
        self.filter_isolated = filter_isolated

    def run(self):
        try:
            self.progress.emit(10, "Extracting graphs...")

            if self.input_type == INPUT_SEGMENTATION:
                series = extract_graphs_from_labels(
                    self.label_stack,
                    pixel_size=self.pixel_size,
                    time_interval=self.time_interval,
                    dilation_radius=self.dilation_radius,
                    min_overlap_pixels=self.min_overlap_pixels,
                    min_edge_length=self.min_edge_length,
                    filter_isolated=self.filter_isolated,
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
                    dilation_radius=self.dilation_radius,
                    min_overlap_pixels=self.min_overlap_pixels,
                    min_edge_length=self.min_edge_length,
                    filter_isolated=self.filter_isolated,
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
                 trackmate_data=None, min_iou=0.3, match_threshold=10.0,
                 max_area_change=0.0):
        super().__init__()
        self.series = series
        self.input_type = input_type
        self.label_stack = label_stack
        self.trackmate_data = trackmate_data
        self.min_iou = min_iou
        self.match_threshold = match_threshold
        # 0 in the UI means "no limit"
        self.max_area_change = float('inf') if max_area_change == 0 else max_area_change

    def run(self):
        try:
            self.progress.emit(10, "Running tracking...")

            if self.input_type == INPUT_SEGMENTATION:
                assign_tracking_labels(
                    self.series, self.label_stack,
                    min_iou=self.min_iou, max_area_change=self.max_area_change,
                )
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

    def __init__(self, series, min_junction_length=0.0, max_t1_distance=0.0,
                 min_traj_frames=1, min_completeness=0.0, max_gap=0):
        super().__init__()
        self.series = series
        self.min_junction_length = min_junction_length
        # 0 in the UI means "no limit"
        self.max_t1_distance = float('inf') if max_t1_distance == 0 else max_t1_distance
        self.min_traj_frames = min_traj_frames
        self.min_completeness = min_completeness
        self.max_gap = max_gap

    def run(self):
        try:
            self.progress.emit(10, "Detecting T1 events...")
            events = detect_t1_events(
                self.series,
                min_junction_length=self.min_junction_length,
                max_t1_distance=self.max_t1_distance,
            )

            self.progress.emit(60, "Building edge trajectories...")
            build_edge_trajectories(self.series, events)

            # Filter trajectories if any non-default filtering requested
            if (self.min_traj_frames > 1 or self.min_completeness > 0
                    or self.max_gap > 0):
                self.progress.emit(80, "Filtering trajectories...")
                self.series.edge_trajectories = filter_trajectories(
                    self.series,
                    min_frames=self.min_traj_frames,
                    min_completeness=self.min_completeness,
                    max_gap=self.max_gap,
                )

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
        max_area_change=0.0,
        min_junction_length=0.0,
        max_t1_distance=0.0,
        min_traj_frames=1,
        min_completeness=0.0,
        max_gap=0,
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
        self.max_area_change = float('inf') if max_area_change == 0 else max_area_change
        self.min_junction_length = min_junction_length
        self.max_t1_distance = float('inf') if max_t1_distance == 0 else max_t1_distance
        self.min_traj_frames = min_traj_frames
        self.min_completeness = min_completeness
        self.max_gap = max_gap

    def _analyze_series(self, series):
        """Run T1 detection, trajectory building, and filtering on a series."""
        detect_t1_events(
            series,
            min_junction_length=self.min_junction_length,
            max_t1_distance=self.max_t1_distance,
        )
        build_edge_trajectories(series, series.t1_events)
        if (self.min_traj_frames > 1 or self.min_completeness > 0
                or self.max_gap > 0):
            series.edge_trajectories = filter_trajectories(
                series,
                min_frames=self.min_traj_frames,
                min_completeness=self.min_completeness,
                max_gap=self.max_gap,
            )

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
                        max_area_change=self.max_area_change,
                    )
                    detect_t1_events(
                        series,
                        min_junction_length=self.min_junction_length,
                        max_t1_distance=self.max_t1_distance,
                    )
                    build_edge_trajectories(series, series.t1_events)
                    if (self.min_traj_frames > 1 or self.min_completeness > 0
                            or self.max_gap > 0):
                        series.edge_trajectories = filter_trajectories(
                            series,
                            min_frames=self.min_traj_frames,
                            min_completeness=self.min_completeness,
                            max_gap=self.max_gap,
                        )
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
                        self._analyze_series(series)
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
                    self._analyze_series(series)
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
                    self._analyze_series(series)
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
        self._trackmate_data_list: List[TrackMateData] = []

        # Pipeline state
        self._pipeline_stage = PipelineStage.IDLE
        self._preview_series: Optional[TissueGraphTimeSeries] = None
        self._current_label_stack: Optional[np.ndarray] = None
        self._current_trackmate_data: Optional[TrackMateData] = None
        self._current_track_map: Optional[Dict] = None
        self._tracked_labels_layer = None
        self._stage_layers: Dict[int, list] = {1: [], 2: [], 3: []}

        # Tagging state
        self._tagging_shapes_layer = None  # the Shapes layer used for tagging
        self._tagging_series = None  # the series currently shown in the tagging layer
        self._cached_selection = []  # persists selection when layer loses focus
        self._show_only_tagged = False
        self._color_by_tags = False

        # Poll the shapes layer selection every 200ms to keep the cache fresh.
        # napari clears selected_data when the mouse leaves the canvas, and
        # event-based approaches are unreliable across napari versions.
        self._selection_timer = QTimer(self)
        self._selection_timer.setInterval(200)
        self._selection_timer.timeout.connect(self._poll_selection)
        self._selection_timer.start()

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

        # --- Labels layer dropdown (for segmentation + both modes) ---
        self.labels_layer_group = QGroupBox("Labels Layer")
        labels_layout = QHBoxLayout()
        self.labels_layer_combo = QComboBox()
        labels_layout.addWidget(self.labels_layer_combo)
        self.labels_refresh_btn = QPushButton("\u21bb")
        self.labels_refresh_btn.setFixedWidth(30)
        labels_layout.addWidget(self.labels_refresh_btn)
        self.labels_layer_group.setLayout(labels_layout)
        layout.addWidget(self.labels_layer_group)

        # --- Points layer selection (for tracks mode) ---
        self.layer_group = QGroupBox("Points Layer")
        layer_layout = QHBoxLayout()
        self.layer_combo = QComboBox()
        layer_layout.addWidget(self.layer_combo)
        self.refresh_btn = QPushButton("\u21bb")
        self.refresh_btn.setFixedWidth(30)
        layer_layout.addWidget(self.refresh_btn)
        self.layer_group.setLayout(layer_layout)
        layout.addWidget(self.layer_group)

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
        # Parameters for both possible Stage 1 operations live here.
        # _update_input_mode toggles which set is visible.
        self.stage1_group = QGroupBox("Stage 1")
        s1_layout = QVBoxLayout()

        # -- Tracking parameters (segmentation Stage 1) --
        self.s1_tracking_widget = QWidget()
        s1_track_layout = QVBoxLayout()
        s1_track_layout.setContentsMargins(0, 0, 0, 0)

        iou_row = QHBoxLayout()
        iou_row.addWidget(QLabel("Min IoU:"))
        self.min_iou_spin = QDoubleSpinBox()
        self.min_iou_spin.setMinimum(0.0)
        self.min_iou_spin.setMaximum(1.0)
        self.min_iou_spin.setSingleStep(0.05)
        self.min_iou_spin.setValue(0.3)
        iou_row.addWidget(self.min_iou_spin)
        s1_track_layout.addLayout(iou_row)

        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Max area change:"))
        self.max_area_change_spin = QDoubleSpinBox()
        self.max_area_change_spin.setMinimum(0.0)
        self.max_area_change_spin.setMaximum(100.0)
        self.max_area_change_spin.setSingleStep(0.5)
        self.max_area_change_spin.setValue(0.0)
        self.max_area_change_spin.setToolTip(
            "Max area ratio between matched labels. "
            "0 = no limit. Try 2.0 to reject segmentation errors."
        )
        area_row.addWidget(self.max_area_change_spin)
        s1_track_layout.addLayout(area_row)

        self.s1_tracking_widget.setLayout(s1_track_layout)
        s1_layout.addWidget(self.s1_tracking_widget)

        # -- Graph extraction parameters (non-seg Stage 1) --
        self.s1_extract_widget = QWidget()
        s1_ext_layout = QVBoxLayout()
        s1_ext_layout.setContentsMargins(0, 0, 0, 0)

        # Voronoi parameters (tracks mode only)
        self.voronoi_widget = QWidget()
        vor_layout = QVBoxLayout()
        vor_layout.setContentsMargins(0, 0, 0, 0)

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

        self.voronoi_widget.setLayout(vor_layout)
        s1_ext_layout.addWidget(self.voronoi_widget)

        # Label extraction parameters (both mode Stage 1)
        self.s1_label_extract_widget = QWidget()
        s1_le_layout = QVBoxLayout()
        s1_le_layout.setContentsMargins(0, 0, 0, 0)

        dr_row = QHBoxLayout()
        dr_row.addWidget(QLabel("Dilation radius:"))
        self.s1_dilation_radius_spin = QSpinBox()
        self.s1_dilation_radius_spin.setMinimum(1)
        self.s1_dilation_radius_spin.setMaximum(20)
        self.s1_dilation_radius_spin.setValue(1)
        self.s1_dilation_radius_spin.setToolTip(
            "Radius for morphological dilation when detecting cell adjacency."
        )
        dr_row.addWidget(self.s1_dilation_radius_spin)
        s1_le_layout.addLayout(dr_row)

        mo_row = QHBoxLayout()
        mo_row.addWidget(QLabel("Min overlap (px):"))
        self.s1_min_overlap_spin = QSpinBox()
        self.s1_min_overlap_spin.setMinimum(1)
        self.s1_min_overlap_spin.setMaximum(1000)
        self.s1_min_overlap_spin.setValue(5)
        self.s1_min_overlap_spin.setToolTip(
            "Minimum boundary overlap pixels to consider two cells adjacent."
        )
        mo_row.addWidget(self.s1_min_overlap_spin)
        s1_le_layout.addLayout(mo_row)

        mel_row = QHBoxLayout()
        mel_row.addWidget(QLabel("Min edge length (px):"))
        self.s1_min_edge_length_spin = QDoubleSpinBox()
        self.s1_min_edge_length_spin.setMinimum(0.0)
        self.s1_min_edge_length_spin.setMaximum(1000.0)
        self.s1_min_edge_length_spin.setSingleStep(1.0)
        self.s1_min_edge_length_spin.setValue(0.0)
        self.s1_min_edge_length_spin.setToolTip(
            "Minimum junction length to keep. Shorter junctions are discarded."
        )
        mel_row.addWidget(self.s1_min_edge_length_spin)
        s1_le_layout.addLayout(mel_row)

        self.s1_filter_isolated_cb = QCheckBox("Filter isolated edges")
        self.s1_filter_isolated_cb.setChecked(True)
        self.s1_filter_isolated_cb.setToolTip(
            "Remove edges where either cell has only one neighbor."
        )
        s1_le_layout.addWidget(self.s1_filter_isolated_cb)

        self.s1_label_extract_widget.setLayout(s1_le_layout)
        s1_ext_layout.addWidget(self.s1_label_extract_widget)

        self.s1_extract_widget.setLayout(s1_ext_layout)
        s1_layout.addWidget(self.s1_extract_widget)

        self.stage1_btn = QPushButton("Run Stage 1")
        s1_layout.addWidget(self.stage1_btn)
        self.stage1_info = QLabel("")
        self.stage1_info.setWordWrap(True)
        s1_layout.addWidget(self.stage1_info)
        self.stage1_group.setLayout(s1_layout)
        layout.addWidget(self.stage1_group)

        # --- Pipeline: Stage 2 ---
        # Parameters for both possible Stage 2 operations live here.
        self.stage2_group = QGroupBox("Stage 2")
        s2_layout = QVBoxLayout()

        # -- Graph extraction parameters (segmentation Stage 2) --
        self.s2_extract_widget = QWidget()
        s2_ext_layout = QVBoxLayout()
        s2_ext_layout.setContentsMargins(0, 0, 0, 0)

        dr_row2 = QHBoxLayout()
        dr_row2.addWidget(QLabel("Dilation radius:"))
        self.dilation_radius_spin = QSpinBox()
        self.dilation_radius_spin.setMinimum(1)
        self.dilation_radius_spin.setMaximum(20)
        self.dilation_radius_spin.setValue(1)
        self.dilation_radius_spin.setToolTip(
            "Radius for morphological dilation when detecting cell adjacency."
        )
        dr_row2.addWidget(self.dilation_radius_spin)
        s2_ext_layout.addLayout(dr_row2)

        mo_row2 = QHBoxLayout()
        mo_row2.addWidget(QLabel("Min overlap (px):"))
        self.min_overlap_spin = QSpinBox()
        self.min_overlap_spin.setMinimum(1)
        self.min_overlap_spin.setMaximum(1000)
        self.min_overlap_spin.setValue(5)
        self.min_overlap_spin.setToolTip(
            "Minimum boundary overlap pixels to consider two cells adjacent."
        )
        mo_row2.addWidget(self.min_overlap_spin)
        s2_ext_layout.addLayout(mo_row2)

        mel_row2 = QHBoxLayout()
        mel_row2.addWidget(QLabel("Min edge length (px):"))
        self.min_edge_length_spin = QDoubleSpinBox()
        self.min_edge_length_spin.setMinimum(0.0)
        self.min_edge_length_spin.setMaximum(1000.0)
        self.min_edge_length_spin.setSingleStep(1.0)
        self.min_edge_length_spin.setValue(0.0)
        self.min_edge_length_spin.setToolTip(
            "Minimum junction length to keep. Shorter junctions are discarded."
        )
        mel_row2.addWidget(self.min_edge_length_spin)
        s2_ext_layout.addLayout(mel_row2)

        self.filter_isolated_cb = QCheckBox("Filter isolated edges")
        self.filter_isolated_cb.setChecked(True)
        self.filter_isolated_cb.setToolTip(
            "Remove edges where either cell has only one neighbor."
        )
        s2_ext_layout.addWidget(self.filter_isolated_cb)

        self.s2_extract_widget.setLayout(s2_ext_layout)
        s2_layout.addWidget(self.s2_extract_widget)

        # -- Tracking parameters (non-seg Stage 2) --
        self.s2_tracking_widget = QWidget()
        s2_track_layout = QVBoxLayout()
        s2_track_layout.setContentsMargins(0, 0, 0, 0)

        # Match threshold (Both mode)
        self.match_threshold_row = QWidget()
        mt_layout = QHBoxLayout()
        mt_layout.setContentsMargins(0, 0, 0, 0)
        mt_layout.addWidget(QLabel("Match threshold (px):"))
        self.match_threshold_spin = QDoubleSpinBox()
        self.match_threshold_spin.setMinimum(1.0)
        self.match_threshold_spin.setMaximum(100.0)
        self.match_threshold_spin.setSingleStep(1.0)
        self.match_threshold_spin.setValue(10.0)
        mt_layout.addWidget(self.match_threshold_spin)
        self.match_threshold_row.setLayout(mt_layout)
        s2_track_layout.addWidget(self.match_threshold_row)

        self.s2_tracking_widget.setLayout(s2_track_layout)
        s2_layout.addWidget(self.s2_tracking_widget)

        self.stage2_btn = QPushButton("Run Stage 2")
        s2_layout.addWidget(self.stage2_btn)
        self.stage2_info = QLabel("")
        self.stage2_info.setWordWrap(True)
        s2_layout.addWidget(self.stage2_info)
        self.stage2_group.setLayout(s2_layout)
        layout.addWidget(self.stage2_group)

        # --- Pipeline: Stage 3 ---
        self.stage3_group = QGroupBox("Stage 3: T1 + Edge Tracking")
        s3_layout = QVBoxLayout()

        # T1 detection parameters
        mjl_row = QHBoxLayout()
        mjl_row.addWidget(QLabel("Min junction length (px):"))
        self.min_junction_length_spin = QDoubleSpinBox()
        self.min_junction_length_spin.setMinimum(0.0)
        self.min_junction_length_spin.setMaximum(1000.0)
        self.min_junction_length_spin.setSingleStep(1.0)
        self.min_junction_length_spin.setValue(0.0)
        self.min_junction_length_spin.setToolTip(
            "Junctions shorter than this are ignored for T1 detection. "
            "Increase if noisy short edges cause false T1s."
        )
        mjl_row.addWidget(self.min_junction_length_spin)
        s3_layout.addLayout(mjl_row)

        mtd_row = QHBoxLayout()
        mtd_row.addWidget(QLabel("Max T1 distance (px):"))
        self.max_t1_distance_spin = QDoubleSpinBox()
        self.max_t1_distance_spin.setMinimum(0.0)
        self.max_t1_distance_spin.setMaximum(10000.0)
        self.max_t1_distance_spin.setSingleStep(5.0)
        self.max_t1_distance_spin.setValue(0.0)
        self.max_t1_distance_spin.setToolTip(
            "Max distance between lost/gained edge midpoints to pair as T1. "
            "0 = no limit. Reduce to avoid pairing distant events."
        )
        mtd_row.addWidget(self.max_t1_distance_spin)
        s3_layout.addLayout(mtd_row)

        # Trajectory filtering parameters
        mtf_row = QHBoxLayout()
        mtf_row.addWidget(QLabel("Min trajectory frames:"))
        self.min_traj_frames_spin = QSpinBox()
        self.min_traj_frames_spin.setMinimum(1)
        self.min_traj_frames_spin.setMaximum(10000)
        self.min_traj_frames_spin.setValue(1)
        self.min_traj_frames_spin.setToolTip(
            "Minimum frames a junction must exist to keep its trajectory."
        )
        mtf_row.addWidget(self.min_traj_frames_spin)
        s3_layout.addLayout(mtf_row)

        mc_row = QHBoxLayout()
        mc_row.addWidget(QLabel("Min completeness:"))
        self.min_completeness_spin = QDoubleSpinBox()
        self.min_completeness_spin.setMinimum(0.0)
        self.min_completeness_spin.setMaximum(1.0)
        self.min_completeness_spin.setSingleStep(0.05)
        self.min_completeness_spin.setValue(0.0)
        self.min_completeness_spin.setToolTip(
            "Fraction of total frames a trajectory must span (0.0-1.0)."
        )
        mc_row.addWidget(self.min_completeness_spin)
        s3_layout.addLayout(mc_row)

        mg_row = QHBoxLayout()
        mg_row.addWidget(QLabel("Max gap tolerance:"))
        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setMinimum(0)
        self.max_gap_spin.setMaximum(1000)
        self.max_gap_spin.setValue(0)
        self.max_gap_spin.setToolTip(
            "Max consecutive missing frames allowed in a trajectory. "
            "0 = no gaps allowed."
        )
        mg_row.addWidget(self.max_gap_spin)
        s3_layout.addLayout(mg_row)

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

        # --- Tagging ---
        self.tagging_group = QGroupBox("Junction Tagging")
        tag_layout = QVBoxLayout()

        # Selection indicator
        self.selection_label = QLabel("No junctions selected")
        tag_layout.addWidget(self.selection_label)

        # Tag input + buttons
        tag_input_row = QHBoxLayout()
        self.tag_name_edit = QLineEdit()
        self.tag_name_edit.setPlaceholderText("Tag name (e.g. central)")
        tag_input_row.addWidget(self.tag_name_edit)
        tag_layout.addLayout(tag_input_row)

        tag_btn_row = QHBoxLayout()
        self.tag_selected_btn = QPushButton("Tag Selected")
        self.untag_selected_btn = QPushButton("Untag Selected")
        tag_btn_row.addWidget(self.tag_selected_btn)
        tag_btn_row.addWidget(self.untag_selected_btn)
        tag_layout.addLayout(tag_btn_row)

        # Checkboxes
        self.color_by_tags_cb = QCheckBox("Color by tags")
        tag_layout.addWidget(self.color_by_tags_cb)
        self.show_only_tagged_cb = QCheckBox("Show only tagged junctions")
        tag_layout.addWidget(self.show_only_tagged_cb)

        # Tag list
        tag_layout.addWidget(QLabel("Tags:"))
        self.tag_list_widget = QListWidget()
        self.tag_list_widget.setMaximumHeight(80)
        tag_layout.addWidget(self.tag_list_widget)

        # Clear tag button
        clear_tag_row = QHBoxLayout()
        self.clear_tag_btn = QPushButton("Clear Selected Tag")
        self.refresh_tags_btn = QPushButton("Refresh")
        clear_tag_row.addWidget(self.clear_tag_btn)
        clear_tag_row.addWidget(self.refresh_tags_btn)
        tag_layout.addLayout(clear_tag_row)

        self.tagging_group.setLayout(tag_layout)
        layout.addWidget(self.tagging_group)
        self.tagging_group.setVisible(False)

        # --- Batch mode ---
        self.batch_group = QGroupBox("Batch")
        batch_layout = QVBoxLayout()
        self.build_batch_btn = QPushButton("Build All (Batch)")
        batch_layout.addWidget(self.build_batch_btn)
        self.batch_group.setLayout(batch_layout)
        layout.addWidget(self.batch_group)

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
        self.labels_refresh_btn.clicked.connect(self._refresh_layers)
        self.refresh_btn.clicked.connect(self._refresh_layers)
        self.trackmate_load_btn.clicked.connect(self._load_trackmate_xml)
        self.voronoi_method_combo.currentIndexChanged.connect(self._update_lloyd_visibility)

        # Auto-sync layer dropdowns when layers are added/removed/renamed
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.layers.events.changed.connect(self._refresh_layers)

        # Pipeline
        self.stage1_btn.clicked.connect(self._run_stage1)
        self.stage2_btn.clicked.connect(self._run_stage2)
        self.analyze_btn.clicked.connect(self._run_analysis)
        self.add_to_dataset_btn.clicked.connect(self._add_preview_to_dataset)
        self.discard_btn.clicked.connect(self._discard_pipeline)

        # Tagging
        self.tag_selected_btn.clicked.connect(self._tag_selected)
        self.untag_selected_btn.clicked.connect(self._untag_selected)
        self.color_by_tags_cb.toggled.connect(self._toggle_color_by_tags)
        self.show_only_tagged_cb.toggled.connect(self._toggle_show_only_tagged)
        self.clear_tag_btn.clicked.connect(self._clear_selected_tag)
        self.refresh_tags_btn.clicked.connect(self._refresh_tagging_layer)

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
        # Stage 1 can be re-run after completion (user tweaks params)
        self.stage1_btn.setEnabled(
            stage in (PipelineStage.IDLE, PipelineStage.STAGE1_DONE)
        )
        self.stage2_btn.setEnabled(stage == PipelineStage.STAGE1_DONE)
        self.analyze_btn.setEnabled(stage == PipelineStage.STAGE2_DONE)
        self.add_to_dataset_btn.setEnabled(stage == PipelineStage.STAGE3_DONE)
        self.discard_btn.setEnabled(stage != PipelineStage.IDLE)

    # ------------------------------------------------------------------
    # Input mode switching
    # ------------------------------------------------------------------
    def _update_input_mode(self):
        mode = self.input_type_combo.currentText()
        is_seg = mode == INPUT_SEGMENTATION
        is_tracks = mode == INPUT_TRACKS
        is_both = mode == INPUT_BOTH

        # Input sources
        self.labels_layer_group.setVisible(is_seg or is_both)
        self.layer_group.setVisible(is_tracks)
        self.trackmate_group.setVisible(is_tracks or is_both)
        self.batch_group.setVisible(not is_seg)

        # Stage 1 & 2: swap params based on mode
        if is_seg:
            # Stage 1 = Tracking, Stage 2 = Graph Extraction
            self.stage1_group.setTitle("Stage 1: Cell Tracking")
            self.stage1_btn.setText("Run Tracking")
            self.s1_tracking_widget.setVisible(True)
            self.s1_extract_widget.setVisible(False)

            self.stage2_group.setTitle("Stage 2: Graph Extraction")
            self.stage2_btn.setText("Extract Graphs")
            self.s2_extract_widget.setVisible(True)
            self.s2_tracking_widget.setVisible(False)
        else:
            # Stage 1 = Graph Extraction, Stage 2 = Tracking
            self.stage1_group.setTitle("Stage 1: Graph Extraction")
            self.stage1_btn.setText("Extract Graphs")
            self.s1_tracking_widget.setVisible(False)
            self.s1_extract_widget.setVisible(True)
            self.voronoi_widget.setVisible(is_tracks)
            self.s1_label_extract_widget.setVisible(is_both)

            self.stage2_group.setTitle("Stage 2: Cell Tracking")
            self.stage2_btn.setText("Run Tracking")
            self.s2_extract_widget.setVisible(False)
            self.s2_tracking_widget.setVisible(True)
            self.match_threshold_row.setVisible(is_both)

        self._update_lloyd_visibility()
        self._refresh_layers()

    def _update_lloyd_visibility(self):
        is_lloyd = self.voronoi_method_combo.currentIndex() == 1
        self.lloyd_iter_spin.setEnabled(is_lloyd)

    def _refresh_layers(self, event=None):
        import napari

        # Populate Labels layer dropdown
        self.labels_layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                self.labels_layer_combo.addItem(layer.name)

        # Populate Points layer dropdown
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Points):
                self.layer_combo.addItem(layer.name)

        # Auto-select the active layer if it matches the expected type
        active = self.viewer.layers.selection.active
        if active is not None:
            if isinstance(active, napari.layers.Labels):
                idx = self.labels_layer_combo.findText(active.name)
                if idx >= 0:
                    self.labels_layer_combo.setCurrentIndex(idx)
            elif isinstance(active, napari.layers.Points):
                idx = self.layer_combo.findText(active.name)
                if idx >= 0:
                    self.layer_combo.setCurrentIndex(idx)

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
    # Label stack from viewer
    # ------------------------------------------------------------------
    def _get_selected_label_stack(self) -> Optional[np.ndarray]:
        """Return the data from the selected Labels layer, or None."""
        layer_name = self.labels_layer_combo.currentText()
        if not layer_name:
            self.status_label.setText("No Labels layer selected.")
            return None
        try:
            data = self.viewer.layers[layer_name].data
        except KeyError:
            self.status_label.setText(f"Layer '{layer_name}' not found.")
            return None
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if data.ndim != 3:
            self.status_label.setText(
                f"Labels layer must be 2D or 3D (T, H, W), got {data.ndim}D."
            )
            return None
        return data

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def _selected_trackmate_index(self) -> int:
        row = self.trackmate_file_list.currentRow()
        return row if row >= 0 else 0

    # ------------------------------------------------------------------
    # Stage 1: mode-dependent dispatch
    # ------------------------------------------------------------------
    def _run_stage1(self):
        input_type = self.input_type_combo.currentText()

        if input_type == INPUT_SEGMENTATION:
            self._run_seg_tracking()
        else:
            self._run_extract()

    def _run_seg_tracking(self):
        """Segmentation Stage 1: Cell tracking via IoU matching."""
        label_stack = self._get_selected_label_stack()
        if label_stack is None:
            return

        self._discard_pipeline()
        self._current_label_stack = label_stack

        worker = CellTrackingWorker(
            label_stack,
            min_iou=self.min_iou_spin.value(),
            max_area_change=self.max_area_change_spin.value(),
        )
        self._run_worker(worker, self._on_seg_tracking_done)

    def _on_seg_tracking_done(self, track_map):
        self._current_track_map = track_map
        self._pipeline_stage = PipelineStage.STAGE1_DONE
        self._finish_worker()

        # Count tracks and breaks from the track_map
        all_track_ids = set()
        for frame_tracks in track_map.values():
            all_track_ids.update(frame_tracks.values())

        # Count births/deaths from track_map
        frame_indices = sorted(track_map.keys())
        track_first = {}
        track_last = {}
        for f_idx in frame_indices:
            for tid in track_map[f_idx].values():
                if tid not in track_first:
                    track_first[tid] = f_idx
                track_last[tid] = f_idx

        n_births = sum(1 for f in track_first.values() if f > frame_indices[0]) if frame_indices else 0
        n_deaths = sum(1 for f in track_last.values() if f < frame_indices[-1]) if frame_indices else 0

        self.stage1_info.setText(
            f"{len(all_track_ids)} tracks, {n_births} births, {n_deaths} deaths"
        )
        self.status_label.setText("Stage 1 complete. Inspect tracked labels, then extract graphs.")

        # Show tracked labels as napari Labels layer
        self._remove_stage_layers(1)
        tracked = build_tracked_labels(self._current_label_stack, track_map)
        layer = self.viewer.add_labels(
            tracked, name="[Pipeline] Tracked Labels",
        )
        self._tracked_labels_layer = layer
        self._stage_layers[1].append(layer)

        self._update_pipeline_buttons()

    def _run_extract(self):
        """Non-segmentation Stage 1 / Segmentation Stage 2: graph extraction."""
        input_type = self.input_type_combo.currentText()
        pixel_size = self._parse_float(self.pixel_size_edit.text())
        time_interval = self._parse_float(self.time_interval_edit.text())
        voronoi_method = (
            VoronoiMethod.LLOYD if self.voronoi_method_combo.currentIndex() == 1
            else VoronoiMethod.STANDARD
        )

        # Read extraction params from the correct stage group
        if input_type == INPUT_SEGMENTATION:
            # Segmentation Stage 2: params in s2_extract_widget
            dilation_radius = self.dilation_radius_spin.value()
            min_overlap_pixels = self.min_overlap_spin.value()
            min_edge_length = self.min_edge_length_spin.value()
            filter_isolated = self.filter_isolated_cb.isChecked()
        else:
            # Non-seg Stage 1: params in s1_label_extract_widget
            dilation_radius = self.s1_dilation_radius_spin.value()
            min_overlap_pixels = self.s1_min_overlap_spin.value()
            min_edge_length = self.s1_min_edge_length_spin.value()
            filter_isolated = self.s1_filter_isolated_cb.isChecked()

        kwargs = dict(
            input_type=input_type,
            pixel_size=pixel_size,
            time_interval=time_interval,
            voronoi_method=voronoi_method,
            lloyd_iterations=self.lloyd_iter_spin.value(),
            dilation_radius=dilation_radius,
            min_overlap_pixels=min_overlap_pixels,
            min_edge_length=min_edge_length,
            filter_isolated=filter_isolated,
        )

        label_stack = None
        trackmate_data = None

        if input_type == INPUT_SEGMENTATION:
            # Called as Stage 2 for segmentation — label_stack already stored
            kwargs["label_stack"] = self._current_label_stack
            label_stack = self._current_label_stack
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
            label_stack = self._get_selected_label_stack()
            if label_stack is None:
                return
            if not self._trackmate_data_list:
                self.status_label.setText("No TrackMate XML loaded.")
                return
            kwargs["label_stack"] = label_stack
            trackmate_data = self._trackmate_data_list[self._selected_trackmate_index()]

        # For non-segmentation, discard previous pipeline and set references
        if input_type != INPUT_SEGMENTATION:
            self._discard_pipeline()
            self._current_label_stack = label_stack
            self._current_trackmate_data = trackmate_data

        self._run_worker(GraphExtractWorker(**kwargs), self._on_extract_finished)

    def _on_extract_finished(self, series):
        input_type = self.input_type_combo.currentText()

        if input_type == INPUT_SEGMENTATION:
            # Segmentation Stage 2: apply track_map after graph extraction
            self._on_seg_extract_done(series)
        else:
            # Non-segmentation Stage 1
            self._preview_series = series
            self._pipeline_stage = PipelineStage.STAGE1_DONE
            self._finish_worker()

            total_cells = sum(len(f.cells) for f in series.frames.values())
            total_junctions = sum(len(f.junctions) for f in series.frames.values())
            self.stage1_info.setText(
                f"{series.num_frames} frames, {total_cells} cells, {total_junctions} junctions"
            )
            self.status_label.setText("Stage 1 complete. Inspect junctions & centroids, then run tracking.")

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

    def _on_seg_extract_done(self, series):
        """Segmentation Stage 2 finish: apply track_map, show junctions + tracked centroids."""
        if self._current_track_map is not None:
            apply_track_map(series, self._current_track_map)

        self._preview_series = series
        self._pipeline_stage = PipelineStage.STAGE2_DONE
        self._finish_worker()

        total_cells = sum(len(f.cells) for f in series.frames.values())
        total_junctions = sum(len(f.junctions) for f in series.frames.values())

        # Count how many cells have tracking
        n_tracked = sum(
            1 for f in series.frames.values()
            for c in f.cells.values() if c.track_id is not None
        )
        self.stage2_info.setText(
            f"{series.num_frames} frames, {total_cells} cells ({n_tracked} tracked), "
            f"{total_junctions} junctions"
        )
        self.status_label.setText("Stage 2 complete. Inspect junctions & tracked centroids, then run analysis.")

        # Keep tracked labels layer visible underneath for context
        self._remove_stage_layers(2)

        # Show junctions
        lines, colors = build_all_junction_lines(series)
        if lines:
            layer = self.viewer.add_shapes(
                lines, shape_type="path", edge_color=colors,
                edge_width=2, name="[Pipeline] Junctions",
            )
            self._stage_layers[2].append(layer)

        # Show tracked centroids (graph cells only)
        tracked_pos, tracked_colors, _ = build_tracked_centroids(series)
        if len(tracked_pos) > 0:
            layer = self.viewer.add_points(
                tracked_pos, size=5, face_color=tracked_colors,
                name="[Pipeline] Tracked Centroids",
            )
            self._stage_layers[2].append(layer)

        self._update_pipeline_buttons()

    # ------------------------------------------------------------------
    # Stage 2: mode-dependent dispatch
    # ------------------------------------------------------------------
    def _run_stage2(self):
        input_type = self.input_type_combo.currentText()

        if input_type == INPUT_SEGMENTATION:
            self._run_extract()
        else:
            self._run_tracking()

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
            max_area_change=self.max_area_change_spin.value(),
        )

        self._run_worker(TrackingWorker(**kwargs), self._on_tracking_finished)

    def _on_tracking_finished(self, series):
        self._preview_series = series
        self._pipeline_stage = PipelineStage.STAGE2_DONE
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
        worker = AnalysisWorker(
            self._preview_series,
            min_junction_length=self.min_junction_length_spin.value(),
            max_t1_distance=self.max_t1_distance_spin.value(),
            min_traj_frames=self.min_traj_frames_spin.value(),
            min_completeness=self.min_completeness_spin.value(),
            max_gap=self.max_gap_spin.value(),
        )
        self._run_worker(worker, self._on_analysis_finished)

    def _on_analysis_finished(self, series):
        self._preview_series = series
        self._pipeline_stage = PipelineStage.STAGE3_DONE
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

        # Build junction Shapes layer with features for tagging
        self._show_tagging_for_series(series, stage_layer=True)

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
        self._current_track_map = None
        if self._tracked_labels_layer is not None:
            if self._tracked_labels_layer in self.viewer.layers:
                self.viewer.layers.remove(self._tracked_labels_layer)
            self._tracked_labels_layer = None
        self._pipeline_stage = PipelineStage.IDLE
        self._update_pipeline_buttons()
        self._clear_stage_info()
        self.status_label.setText(f"Added tissue {tid} to dataset.")
        self._update_dataset_ui()

    def _discard_pipeline(self):
        self._remove_all_stage_layers()
        self._remove_tagging_layer()
        self._tagging_series = None
        self.tagging_group.setVisible(False)
        self._preview_series = None
        self._current_label_stack = None
        self._current_trackmate_data = None
        self._current_track_map = None
        if self._tracked_labels_layer is not None:
            if self._tracked_labels_layer in self.viewer.layers:
                self.viewer.layers.remove(self._tracked_labels_layer)
            self._tracked_labels_layer = None
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
            max_area_change=self.max_area_change_spin.value(),
            min_junction_length=self.min_junction_length_spin.value(),
            max_t1_distance=self.max_t1_distance_spin.value(),
            min_traj_frames=self.min_traj_frames_spin.value(),
            min_completeness=self.min_completeness_spin.value(),
            max_gap=self.max_gap_spin.value(),
        )

        # Batch not available for segmentation mode (single tissue from viewer)
        if input_type == INPUT_SEGMENTATION:
            self.status_label.setText("Batch mode not available for segmentation. Use the staged pipeline.")
            return
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
            # Both mode: labels from viewer + TrackMate XMLs
            label_stack = self._get_selected_label_stack()
            if label_stack is None:
                return
            if not self._trackmate_data_list:
                self.status_label.setText("No TrackMate XML loaded.")
                return
            # Single label stack paired with each TrackMate file
            kwargs["label_stacks"] = [label_stack] * len(self._trackmate_data_list)
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
        if series.edge_trajectories:
            # Use the tagging-enabled Shapes layer for junctions
            self._add_inspect_layers(series, skip_junctions=True)
            self._show_tagging_for_series(series)
        else:
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

        self.stage1_btn.setEnabled(False)
        self.stage2_btn.setEnabled(False)
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
        self._remove_tagging_layer()
        self._tagging_series = None
        self.tagging_group.setVisible(False)

    def _add_inspect_layers(
        self, series: TissueGraphTimeSeries, skip_junctions: bool = False,
    ):
        self._remove_inspect_layers()
        j, c, t = self._make_layers(
            series, prefix="", skip_junctions=skip_junctions,
        )
        self._inspect_junction_layer = j
        self._inspect_centroid_layer = c
        self._inspect_t1_layer = t

    def _make_layers(
        self,
        series: TissueGraphTimeSeries,
        prefix: str = "",
        skip_junctions: bool = False,
    ):
        """Create junction, centroid, and T1 layers for a tissue."""
        junction_layer = None
        centroid_layer = None
        t1_layer = None

        if not skip_junctions:
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
    # Tagging
    # ------------------------------------------------------------------
    def _show_tagging_for_series(
        self, series: TissueGraphTimeSeries, stage_layer: bool = False,
    ):
        """Create or refresh the junction Shapes layer with features for tagging.

        Parameters
        ----------
        series : TissueGraphTimeSeries
            The series to visualize.
        stage_layer : bool
            If True, register the layer as a stage 3 layer (pipeline flow).
        """
        self._tagging_series = series
        self._remove_tagging_layer()
        self.tagging_group.setVisible(True)

        lines, colors, features = build_trajectory_lines_with_features(
            series,
            color_by_tags=self._color_by_tags,
            show_only_tagged=self._show_only_tagged,
        )
        if not lines:
            self._update_tag_list()
            return

        layer = self.viewer.add_shapes(
            lines,
            shape_type="path",
            edge_color=colors,
            edge_width=2,
            features=features,
            name="[Pipeline] Trajectories",
        )
        self._tagging_shapes_layer = layer
        self._cached_selection = []

        # Switch to SELECT mode so the user can click to select junctions.
        # In PAN_ZOOM (the default), hovering highlights shapes but clicking
        # does not populate selected_data.
        layer.mode = "select"

        if stage_layer:
            self._stage_layers[3].append(layer)

        self._update_tag_list()

    def _remove_tagging_layer(self):
        if (
            self._tagging_shapes_layer is not None
            and self._tagging_shapes_layer in self.viewer.layers
        ):
            self.viewer.layers.remove(self._tagging_shapes_layer)
            # Also remove from stage layers if present
            for stage in self._stage_layers.values():
                if self._tagging_shapes_layer in stage:
                    stage.remove(self._tagging_shapes_layer)
        self._tagging_shapes_layer = None

    def _poll_selection(self):
        """Poll the shapes layer selection and cache it.

        napari clears selected_data when the mouse leaves the canvas.
        By polling at 200ms we capture the selection while the user
        still has the mouse on the canvas, before it gets cleared.
        """
        if self._tagging_shapes_layer is None:
            return
        try:
            if self._tagging_shapes_layer not in self.viewer.layers:
                return
            current = set(self._tagging_shapes_layer.selected_data)
        except (RuntimeError, AttributeError):
            return
        if current:
            self._cached_selection = list(current)
            n = len(current)
            self.selection_label.setText(f"{n} junction(s) selected")

    def _get_selection(self):
        """Return the current or cached shape selection."""
        if self._tagging_shapes_layer is None:
            return []
        try:
            live = list(self._tagging_shapes_layer.selected_data)
        except (RuntimeError, AttributeError):
            live = []
        if live:
            return live
        return self._cached_selection

    def _refresh_tagging_layer(self):
        """Rebuild the tagging layer from the current series."""
        if self._tagging_series is not None:
            is_pipeline = self._pipeline_stage == PipelineStage.STAGE3_DONE
            self._show_tagging_for_series(
                self._tagging_series, stage_layer=is_pipeline,
            )

    def _tag_selected(self):
        """Apply a tag to all selected junctions in the Shapes layer."""
        tag_name = self.tag_name_edit.text().strip()
        if not tag_name:
            self.status_label.setText("Enter a tag name first.")
            return
        if self._tagging_shapes_layer is None or self._tagging_series is None:
            return

        selected = self._get_selection()
        if not selected:
            self.status_label.setText("Select junctions first by clicking them on the canvas.")
            return

        features = self._tagging_shapes_layer.features
        series = self._tagging_series
        count = 0

        for idx in selected:
            row = features.iloc[idx]
            traj_id = int(row["trajectory_id"])
            pair = (int(row["cell_pair_a"]), int(row["cell_pair_b"]))

            # Tag the trajectory if it exists
            if traj_id != -1:
                tag_trajectory(series, traj_id, tag_name)
                count += 1
            else:
                # Tag the junction directly in all frames where it appears
                key = frozenset(pair)
                for frame in series.frames.values():
                    if key in frame.junctions:
                        tag_junction(frame, pair, tag_name)
                count += 1

        self._cached_selection = []
        self.selection_label.setText("No junctions selected")
        self.status_label.setText(f"Tagged {count} junction(s) as '{tag_name}'.")
        self._refresh_tagging_layer()

    def _untag_selected(self):
        """Remove a tag from all selected junctions in the Shapes layer."""
        tag_name = self.tag_name_edit.text().strip()
        if not tag_name:
            self.status_label.setText("Enter a tag name first.")
            return
        if self._tagging_shapes_layer is None or self._tagging_series is None:
            return

        selected = self._get_selection()
        if not selected:
            self.status_label.setText("Select junctions first.")
            return

        features = self._tagging_shapes_layer.features
        series = self._tagging_series
        count = 0

        for idx in selected:
            row = features.iloc[idx]
            traj_id = int(row["trajectory_id"])
            pair = (int(row["cell_pair_a"]), int(row["cell_pair_b"]))

            if traj_id != -1:
                untag_trajectory(series, traj_id, tag_name)
                count += 1
            else:
                key = frozenset(pair)
                for frame in series.frames.values():
                    if key in frame.junctions:
                        untag_junction(frame, pair, tag_name)
                count += 1

        self._cached_selection = []
        self.selection_label.setText("No junctions selected")
        self.status_label.setText(f"Removed tag '{tag_name}' from {count} junction(s).")
        self._refresh_tagging_layer()

    def _toggle_color_by_tags(self, checked: bool):
        self._color_by_tags = checked
        self._refresh_tagging_layer()

    def _toggle_show_only_tagged(self, checked: bool):
        self._show_only_tagged = checked
        self._refresh_tagging_layer()

    def _clear_selected_tag(self):
        """Clear the tag selected in the tag list from all junctions/trajectories."""
        item = self.tag_list_widget.currentItem()
        if item is None or self._tagging_series is None:
            self.status_label.setText("Select a tag from the list first.")
            return

        # Extract tag name (format: "tag_name (N)")
        tag_name = item.text().rsplit(" (", 1)[0]
        count = clear_tag(self._tagging_series, tag_name)
        self.status_label.setText(f"Cleared tag '{tag_name}' from {count} item(s).")
        self._refresh_tagging_layer()

    def _update_tag_list(self):
        """Refresh the tag list widget from the current series."""
        self.tag_list_widget.clear()
        if self._tagging_series is None:
            return

        tags = get_all_tags(self._tagging_series)
        if not tags:
            return

        # Count occurrences per tag
        for tag in sorted(tags):
            count = 0
            for traj in self._tagging_series.edge_trajectories.values():
                if tag in traj.tags:
                    count += 1
            for frame in self._tagging_series.frames.values():
                for jd in frame.junctions.values():
                    if tag in jd.tags:
                        count += 1
            self.tag_list_widget.addItem(f"{tag} ({count})")

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
        """Clean up background thread and layers if running."""
        self._selection_timer.stop()
        self._remove_tagging_layer()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
