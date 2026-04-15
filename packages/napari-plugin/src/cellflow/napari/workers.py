"""Background workers for cellflow.

QObject-based workers that have not yet been migrated to
napari thread_worker generators.  New code should use
``@thread_worker`` directly in the widget method.
"""
import logging

from qtpy.QtCore import Signal, QObject

from cellflow.utils.structures import TissueGraphTimeSeries
from cellflow.backend.graph import (
    build_from_labels,
    assign_tracking_labels,
    apply_track_map,
)
from cellflow.backend.tracking import assign_track_ids
from cellflow.utils.io import save_dataset, load_dataset
from cellflow.backend.trajectories import build_edge_trajectories, filter_trajectories

logger = logging.getLogger(__name__)


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
        from cellflow.backend.topology import detect_t1_events
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
