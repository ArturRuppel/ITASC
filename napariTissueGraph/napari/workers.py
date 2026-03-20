"""Background workers for napariTissueGraph pipeline stages.

All QObject-based workers that run in QThreads are defined here,
shared by both the main widget and the nuclear tracks widget.
"""
import logging
from enum import auto, Enum

import numpy as np
from qtpy.QtCore import Signal, QObject

from ..structures import TissueGraphTimeSeries, VoronoiMethod
from ..core.graph import (
    build_from_labels,
    extract_graphs_from_labels,
    assign_tracking_labels,
    apply_track_map,
)
from ..core.label_tracking import assign_track_ids
from ..core.topology import detect_t1_events
from ..core.io import save_dataset, load_dataset
from ..analysis.trajectories import build_edge_trajectories, filter_trajectories

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    IDLE = auto()
    STAGE1_DONE = auto()
    STAGE2_DONE = auto()
    STAGE3_DONE = auto()


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
    """Extract per-frame graphs from a segmentation label stack."""

    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(Exception)

    def __init__(self, label_stack, pixel_size=None, time_interval=None,
                 dilation_radius=1, min_overlap_pixels=5,
                 min_edge_length=0.0, filter_isolated=True,
                 min_border_edge_length=5.0):
        super().__init__()
        self.label_stack = label_stack
        self.pixel_size = pixel_size
        self.time_interval = time_interval
        self.dilation_radius = dilation_radius
        self.min_overlap_pixels = min_overlap_pixels
        self.min_edge_length = min_edge_length
        self.filter_isolated = filter_isolated
        self.min_border_edge_length = min_border_edge_length

    def run(self):
        try:
            self.progress.emit(10, "Extracting graphs...")
            series = extract_graphs_from_labels(
                self.label_stack,
                pixel_size=self.pixel_size,
                time_interval=self.time_interval,
                dilation_radius=self.dilation_radius,
                min_overlap_pixels=self.min_overlap_pixels,
                min_edge_length=self.min_edge_length,
                filter_isolated=self.filter_isolated,
                min_border_edge_length=self.min_border_edge_length,
            )
            self.progress.emit(100, "Graphs extracted.")
            self.finished.emit(series)
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
        label_stacks,
        pixel_size=None,
        time_interval=None,
        min_iou=0.3,
        max_area_change=0.0,
        min_junction_length=0.0,
        max_t1_distance=0.0,
        min_traj_frames=1,
        min_completeness=0.0,
        max_gap=0,
    ):
        super().__init__()
        self.label_stacks = label_stacks
        self.pixel_size = pixel_size
        self.time_interval = time_interval
        self.min_iou = min_iou
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
