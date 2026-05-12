================================================================================
CELLFLOW — COMPLETE SOURCE CONCATENATION
================================================================================
Generated: 2026-05-12 12:49:02
Project:   cellflow v0.2.0 — Fast, interactive, hypothesis-driven cell tracking
Author:    Artur Ruppel
License:   GPL-3.0
================================================================================

OVERVIEW
--------

CellFlow is a napari plugin for interactive cell tracking. It provides a
hypothesis-driven workflow: segment cells (via CellPose or custom methods),
track them across time (using Ultrack or custom propagators), correct
tracking errors manually, and analyze / classify trajectories and artifacts.

The codebase is organized into these modules:

  cellflow/                 — package root (__init__.py, napari.yaml manifest)
  cellflow/core/            — shared utilities: logging, paths, data preparation
  cellflow/meta/            — experiment catalog / metadata management
  cellflow/segmentation/    — cell segmentation algorithms (ICM, flow-following)
  cellflow/tracking/        — custom tracking: propagator, retracker, consensus
  cellflow/tracking_ultrack/ — Ultrack integration: DB build, solve, linking
  cellflow/correction/      — label correction utilities
  cellflow/database/        — tracked object DB, hypotheses, validation
  cellflow/analysis/        — artifact detection, NLS classification
  cellflow/napari/          — napari GUI widgets (~15 widget classes)

Entry points:
  - napari plugin: cellflow.napari:CellFlowWidget (via napari.yaml manifest)
  - CLI:            cellflow-classify-nls

================================================================================
FILE TREE
================================================================================

src/
└── cellflow/
    ├── __init__.py
    ├── napari.yaml
    ├── analysis/
    │   ├── __init__.py
    │   ├── artifact_reader.py
    │   ├── nls_classification.py
    │   └── position_artifact.py
    ├── core/
    │   ├── __init__.py
    │   ├── data_prep.py
    │   ├── logging.py
    │   └── paths.py
    ├── correction/
    │   ├── __init__.py
    │   └── labels.py
    ├── database/
    │   ├── __init__.py
    │   ├── hypotheses.py
    │   ├── tracked.py
    │   └── validation.py
    ├── meta/
    │   ├── __init__.py
    │   └── catalog.py
    ├── napari/
    │   ├── __init__.py
    │   ├── _napari_compat.py
    │   ├── analysis_widget.py
    │   ├── artifact_visualization.py
    │   ├── cell_boundary_workflow_widget.py
    │   ├── cell_workflow_widget.py
    │   ├── cellpose_widget.py
    │   ├── correction_widget.py
    │   ├── data_panel_widget.py
    │   ├── data_prep_widget.py
    │   ├── hpc_cellpose_widget.py
    │   ├── main_widget.py
    │   ├── meta_widget.py
    │   ├── nls_classification_widget.py
    │   ├── nucleus_workflow_widget.py
    │   ├── ui_style.py
    │   ├── utils.py
    │   └── widgets.py
    ├── segmentation/
    │   ├── __init__.py
    │   ├── cell_label_icm.py
    │   ├── contour_filtering.py
    │   └── flow_following.py
    ├── tracking/
    │   ├── __init__.py
    │   ├── consensus_movie.py
    │   ├── frame_selector.py
    │   ├── propagator.py
    │   └── retracker.py
    └── tracking_ultrack/
        ├── __init__.py
        ├── anchor.py
        ├── anchor_diagnostics.py
        ├── cell_boundary_selection.py
        ├── config.py
        ├── db_build.py
        ├── export.py
        ├── extend.py
        ├── ingest.py
        ├── linking.py
        ├── metrics.py
        ├── reseed.py
        ├── seed_prior.py
        ├── solve.py
        └── validation_nodes.py

================================================================================
FILE CONTENTS
================================================================================


--------------------------------------------------------------------------------
FILE: src/cellflow/__init__.py
--------------------------------------------------------------------------------




--------------------------------------------------------------------------------
FILE: src/cellflow/analysis/__init__.py
--------------------------------------------------------------------------------

"""Analysis artifact generation for CellFlow."""

from .position_artifact import build_position_analysis_artifact

__all__ = ["build_position_analysis_artifact"]


--------------------------------------------------------------------------------
FILE: src/cellflow/analysis/artifact_reader.py
--------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


def _read_dataset(dataset: h5py.Dataset) -> np.ndarray:
    if h5py.check_string_dtype(dataset.dtype) is not None:
        return dataset.asstr()[:]
    return dataset[:]


def _read_table(group: h5py.Group) -> dict[str, np.ndarray]:
    return {name: _read_dataset(dataset) for name, dataset in group.items()}


@dataclass(frozen=True)
class PositionArtifactData:
    cells: dict[str, np.ndarray]
    edges: dict[str, np.ndarray]
    t1_events: dict[str, np.ndarray]
    cell_tracked_labels_path: str
    nucleus_tracked_labels_path: str
    _edge_coord_y: np.ndarray = field(repr=False)
    _edge_coord_x: np.ndarray = field(repr=False)

    @property
    def coord_y(self) -> np.ndarray:
        return self._edge_coord_y

    @property
    def coord_x(self) -> np.ndarray:
        return self._edge_coord_x

    def edge_lines(self) -> list[np.ndarray]:
        lines: list[np.ndarray] = []
        offsets = self.edges.get("coord_offset", np.asarray([], dtype=np.int64))
        counts = self.edges.get("coord_count", np.asarray([], dtype=np.int64))
        frames = self.edges.get("frame", np.asarray([], dtype=np.int64))
        for frame, offset, count in zip(frames, offsets, counts):
            start = int(offset)
            stop = start + int(count)
            y = self._edge_coord_y[start:stop]
            x = self._edge_coord_x[start:stop]
            if len(y) == 0:
                lines.append(np.empty((0, 3), dtype=float))
                continue
            frame_col = np.full(len(y), float(frame), dtype=float)
            lines.append(np.column_stack([frame_col, y.astype(float, copy=False), x.astype(float, copy=False)]))
        return lines

    def centroid_points(self) -> np.ndarray:
        frames = self.cells.get("frame", np.asarray([], dtype=np.int64))
        ys = self.cells.get("centroid_y", np.asarray([], dtype=float))
        xs = self.cells.get("centroid_x", np.asarray([], dtype=float))
        if len(frames) == 0:
            return np.empty((0, 3), dtype=float)
        return np.column_stack([frames.astype(float, copy=False), ys.astype(float, copy=False), xs.astype(float, copy=False)])


def read_position_artifact(path: str | Path) -> PositionArtifactData:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        provenance = h5["provenance"].attrs
        cell_tracked_labels_path = str(provenance["cell_tracked_labels_path"])
        nucleus_tracked_labels_path = str(provenance["nucleus_tracked_labels_path"])
        cells = _read_table(h5["cells/table"])
        edges = _read_table(h5["edges/table"])
        t1_events = _read_table(h5["t1_events/table"])
        edge_coord_y = h5["edges/coordinates/y"][:]
        edge_coord_x = h5["edges/coordinates/x"][:]
    return PositionArtifactData(
        cells=cells,
        edges=edges,
        t1_events=t1_events,
        cell_tracked_labels_path=cell_tracked_labels_path,
        nucleus_tracked_labels_path=nucleus_tracked_labels_path,
        _edge_coord_y=edge_coord_y,
        _edge_coord_x=edge_coord_x,
    )


--------------------------------------------------------------------------------
FILE: src/cellflow/analysis/nls_classification.py
--------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import tifffile
from skimage.filters import threshold_otsu


HIGH_LABEL = "ctrl"
LOW_LABEL = "vimentin_ko"
METHOD = "two_cluster_track_median"


class NLSClassificationError(ValueError):
    """Raised when NLS track classification would be ambiguous or invalid."""


@dataclass(frozen=True)
class TrackNLSMeasurement:
    median_intensity: float
    pixel_count: int
    frame_count: int


@dataclass(frozen=True)
class NLSClassificationSummary:
    h5_path: Path
    threshold: float
    track_count: int
    high_track_count: int
    low_track_count: int


def measure_track_nls_intensity(
    nls_zavg: np.ndarray,
    nucleus_labels: np.ndarray,
) -> dict[int, TrackNLSMeasurement]:
    """Measure one median NLS intensity scalar per nonzero nuclear track ID."""
    nls = _as_time_yx(np.asarray(nls_zavg), "NLS image")
    labels = _as_time_yx(np.asarray(nucleus_labels), "nucleus tracked labels")
    if nls.shape != labels.shape:
        raise NLSClassificationError(
            f"NLS image and nucleus tracked-label stack shapes do not match: {nls.shape} != {labels.shape}"
        )

    measurements: dict[int, TrackNLSMeasurement] = {}
    for track_id in sorted(np.unique(labels).astype(int)):
        if track_id == 0:
            continue
        mask = labels == track_id
        if not np.any(mask):
            continue
        values = nls[mask]
        frame_count = int(np.count_nonzero(np.any(mask, axis=(1, 2))))
        measurements[track_id] = TrackNLSMeasurement(
            median_intensity=float(np.median(values)),
            pixel_count=int(values.size),
            frame_count=frame_count,
        )
    return measurements


def split_tracks_otsu(track_medians: Mapping[int, float]) -> tuple[float, dict[int, str]]:
    """Split per-track median intensities into deterministic high/low statuses."""
    if len(track_medians) < 2:
        raise NLSClassificationError("Cannot classify fewer than two nonzero tracks with sampled pixels")

    ordered_ids = sorted(int(track_id) for track_id in track_medians)
    values = np.asarray([float(track_medians[track_id]) for track_id in ordered_ids], dtype=float)
    if np.any(~np.isfinite(values)):
        raise NLSClassificationError("Track median intensities must be finite for Otsu classification")
    if np.all(values == values[0]):
        raise NLSClassificationError("Cannot classify tracks because all track intensity scalars are identical")

    threshold = float(threshold_otsu(values))
    assignments = {
        track_id: ("high" if float(track_medians[track_id]) > threshold else "low")
        for track_id in ordered_ids
    }
    high_count = sum(status == "high" for status in assignments.values())
    low_count = sum(status == "low" for status in assignments.values())
    if high_count == 0 or low_count == 0:
        raise NLSClassificationError(
            "Otsu threshold assigned all tracks to one group; refusing to write classifications"
        )
    return threshold, assignments


def split_tracks_two_clusters(track_medians: Mapping[int, float]) -> tuple[float, dict[int, str]]:
    """Split per-track median intensities into deterministic two-Gaussian classes."""
    if len(track_medians) < 2:
        raise NLSClassificationError("Cannot classify fewer than two nonzero tracks with sampled pixels")

    ordered_items = sorted(
        ((int(track_id), float(median)) for track_id, median in track_medians.items()),
        key=lambda item: (item[1], item[0]),
    )
    ordered_ids = [track_id for track_id, _ in ordered_items]
    values = np.asarray([median for _, median in ordered_items], dtype=float)
    if np.any(~np.isfinite(values)):
        raise NLSClassificationError("Track median intensities must be finite for two-cluster classification")
    if np.all(values == values[0]):
        raise NLSClassificationError("Cannot classify tracks because all track intensity scalars are identical")

    if len(values) == 2:
        high_ids = {ordered_ids[1]}
    else:
        high_ids = _fit_high_intensity_gaussian_cluster(ordered_ids, values)

    low_values = np.asarray(
        [median for track_id, median in zip(ordered_ids, values) if track_id not in high_ids],
        dtype=float,
    )
    high_values = np.asarray(
        [median for track_id, median in zip(ordered_ids, values) if track_id in high_ids],
        dtype=float,
    )
    if low_values.size == 0 or high_values.size == 0:
        raise NLSClassificationError(
            "Two-cluster split assigned all tracks to one group; refusing to write classifications"
        )

    low_center = float(np.mean(low_values))
    high_center = float(np.mean(high_values))
    if high_center <= low_center:
        raise NLSClassificationError("Cannot classify tracks because two-cluster centers are not separated")

    threshold = (float(np.max(low_values)) + float(np.min(high_values))) / 2.0

    assignments = {
        track_id: ("high" if track_id in high_ids else "low")
        for track_id in ordered_ids
    }
    high_count = sum(status == "high" for status in assignments.values())
    low_count = sum(status == "low" for status in assignments.values())
    if high_count == 0 or low_count == 0:
        raise NLSClassificationError(
            "Two-cluster split assigned all tracks to one group; refusing to write classifications"
        )
    return threshold, assignments


def _fit_high_intensity_gaussian_cluster(ordered_ids: list[int], values: np.ndarray) -> set[int]:
    global_variance = float(np.var(values))
    min_variance = max(global_variance * 1e-6, 1e-6)
    means = np.asarray([np.percentile(values, 25), np.percentile(values, 75)], dtype=float)
    if means[0] == means[1]:
        means = np.asarray([float(np.min(values)), float(np.max(values))], dtype=float)
    variances = np.asarray([max(global_variance, min_variance), max(global_variance, min_variance)], dtype=float)
    weights = np.asarray([0.5, 0.5], dtype=float)

    for _ in range(200):
        log_prob = np.empty((values.size, 2), dtype=float)
        for component in range(2):
            variance = max(float(variances[component]), min_variance)
            weight = max(float(weights[component]), 1e-12)
            log_prob[:, component] = (
                np.log(weight)
                - 0.5 * np.log(2.0 * np.pi * variance)
                - ((values - means[component]) ** 2) / (2.0 * variance)
            )

        row_max = np.max(log_prob, axis=1, keepdims=True)
        responsibilities = np.exp(log_prob - row_max)
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

        component_weights = np.sum(responsibilities, axis=0)
        if np.any(component_weights <= 1e-9):
            raise NLSClassificationError("Cannot classify NLS tracks because a two-cluster component is empty")

        weights = component_weights / values.size
        means = np.sum(responsibilities * values[:, np.newaxis], axis=0) / component_weights
        variances = (
            np.sum(responsibilities * ((values[:, np.newaxis] - means) ** 2), axis=0)
            / component_weights
        )

    high_component = int(np.argmax(means))
    return {
        track_id
        for track_id, responsibility in zip(ordered_ids, responsibilities[:, high_component])
        if float(responsibility) >= 0.5
    }


def patch_position_artifact_nls_classes(
    h5_path: str | Path,
    nls_zavg_path: str | Path | None = None,
    nucleus_labels_path: str | Path | None = None,
    *,
    output_path: str | Path | None = None,
    overwrite_output: bool = False,
) -> NLSClassificationSummary:
    """Patch a CellFlow position analysis H5 with NLS high/low track classes."""
    source_h5_path = Path(h5_path)
    target_h5_path = Path(output_path) if output_path is not None else source_h5_path
    nls_path = _resolve_nls_path(source_h5_path, nls_zavg_path)
    labels_path = _resolve_nucleus_labels_path(source_h5_path, nucleus_labels_path)

    nls = _read_image_stack(nls_path)
    labels = _read_image_stack(labels_path)
    measurements = measure_track_nls_intensity(nls, labels)
    medians = {track_id: item.median_intensity for track_id, item in measurements.items()}
    threshold, assignments = split_tracks_two_clusters(medians)
    cell_ids = _read_cell_ids(source_h5_path)
    if not set(cell_ids).intersection(assignments):
        raise NLSClassificationError(
            "H5 cells/table/cell_id values do not overlap any classified nuclear track IDs"
        )

    if output_path is not None:
        if target_h5_path.exists() and not overwrite_output:
            raise FileExistsError(target_h5_path)
        target_h5_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_h5_path, target_h5_path)

    _write_classification_columns(
        target_h5_path,
        cell_ids=cell_ids,
        measurements=measurements,
        assignments=assignments,
        threshold=threshold,
        nls_path=nls_path,
        labels_path=labels_path,
    )

    high_count = sum(status == "high" for status in assignments.values())
    low_count = sum(status == "low" for status in assignments.values())
    return NLSClassificationSummary(
        h5_path=target_h5_path,
        threshold=threshold,
        track_count=len(assignments),
        high_track_count=high_count,
        low_track_count=low_count,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Patch a CellFlow position analysis H5 with NLS-high/NLS-low track classifications."
    )
    parser.add_argument("h5_path", type=Path, help="Path to position_analysis.h5")
    parser.add_argument("--nls-zavg", type=Path, default=None, help="Path to 0_input/NLS_zavg.tif")
    parser.add_argument(
        "--nucleus-labels",
        type=Path,
        default=None,
        help="Path to 2_nucleus/tracked_labels.tif",
    )
    parser.add_argument("--output", type=Path, default=None, help="Write a patched copy instead of patching in place")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow --output to replace an existing file",
    )
    args = parser.parse_args(argv)

    summary = patch_position_artifact_nls_classes(
        args.h5_path,
        args.nls_zavg,
        args.nucleus_labels,
        output_path=args.output,
        overwrite_output=args.overwrite_output,
    )
    print(f"tracks measured: {summary.track_count}")
    print(f"threshold: {summary.threshold:.6g}")
    print(f"high tracks: {summary.high_track_count}")
    print(f"low tracks: {summary.low_track_count}")
    print(f"H5 written: {summary.h5_path}")
    return 0


def _as_time_yx(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise NLSClassificationError(f"Expected a 2-D or 3-D {name}, got shape {arr.shape}")
    return arr


def _read_image_stack(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return _as_time_yx(np.asarray(tifffile.imread(path)), str(path))


def _read_cell_ids(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if "cells/table/cell_id" not in h5:
            raise NLSClassificationError("H5 artifact is missing cells/table/cell_id")
        return np.asarray(h5["cells/table/cell_id"][:], dtype=np.int64)


def _write_classification_columns(
    path: Path,
    *,
    cell_ids: np.ndarray,
    measurements: Mapping[int, TrackNLSMeasurement],
    assignments: Mapping[int, str],
    threshold: float,
    nls_path: Path,
    labels_path: Path,
) -> None:
    class_labels: list[str] = []
    statuses: list[str] = []
    medians = np.full(len(cell_ids), np.nan, dtype=float)
    pixel_counts = np.zeros(len(cell_ids), dtype=np.int64)
    frame_counts = np.zeros(len(cell_ids), dtype=np.int64)

    for idx, cell_id_value in enumerate(cell_ids):
        cell_id = int(cell_id_value)
        status = assignments.get(cell_id, "")
        statuses.append(status)
        class_labels.append(HIGH_LABEL if status == "high" else LOW_LABEL if status == "low" else "")
        measurement = measurements.get(cell_id)
        if measurement is not None:
            medians[idx] = measurement.median_intensity
            pixel_counts[idx] = measurement.pixel_count
            frame_counts[idx] = measurement.frame_count

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "r+") as h5:
        if "cells/table" not in h5:
            raise NLSClassificationError("H5 artifact is missing cells/table")
        cells = h5["cells/table"]
        _replace_dataset(cells, "class_label", np.asarray(class_labels, dtype=object), dtype=string_dtype)
        _replace_dataset(cells, "nls_status", np.asarray(statuses, dtype=object), dtype=string_dtype)
        _replace_dataset(cells, "nls_track_median_intensity", medians)
        _replace_dataset(cells, "nls_track_pixel_count", pixel_counts)
        _replace_dataset(cells, "nls_track_frame_count", frame_counts)

        measurements_root = h5.require_group("cells/measurements")
        if "nls_classification" in measurements_root:
            del measurements_root["nls_classification"]
        meta = measurements_root.create_group("nls_classification")
        high_count = sum(status == "high" for status in assignments.values())
        low_count = sum(status == "low" for status in assignments.values())
        meta.attrs["method"] = METHOD
        meta.attrs["threshold"] = threshold
        meta.attrs["high_label"] = HIGH_LABEL
        meta.attrs["low_label"] = LOW_LABEL
        meta.attrs["nls_zavg_path"] = str(nls_path)
        meta.attrs["nucleus_tracked_labels_path"] = str(labels_path)
        meta.attrs["classified_track_count"] = len(assignments)
        meta.attrs["high_track_count"] = high_count
        meta.attrs["low_track_count"] = low_count
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()


def _replace_dataset(group: h5py.Group, name: str, data: np.ndarray, **kwargs) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, **kwargs)


def _resolve_nucleus_labels_path(h5_path: Path, explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    with h5py.File(h5_path, "r") as h5:
        provenance = h5.get("provenance")
        if provenance is not None and "nucleus_tracked_labels_path" in provenance.attrs:
            return Path(provenance.attrs["nucleus_tracked_labels_path"])
    position_dir = h5_path.parent.parent
    return position_dir / "2_nucleus" / "tracked_labels.tif"


def _resolve_nls_path(h5_path: Path, explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    position_dir = h5_path.parent.parent
    return position_dir / "0_input" / "NLS_zavg.tif"


if __name__ == "__main__":
    raise SystemExit(main())


--------------------------------------------------------------------------------
FILE: src/cellflow/analysis/position_artifact.py
--------------------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import h5py
import numpy as np
import tifffile
from skimage.measure import regionprops


NULL_INT = -1


@dataclass(frozen=True)
class EdgeRecord:
    frame: int
    pair: tuple[int, int]
    kind: str
    edge_label: str
    length: float
    midpoint_y: float
    midpoint_x: float
    coordinates: np.ndarray


@dataclass(frozen=True)
class T1Record:
    t1_event_id: int
    frame: int
    edge_id: int
    losing_pair: tuple[int, int]
    gaining_pair: tuple[int, int]
    location_y: float
    location_x: float

    @property
    def losing_cell_a(self) -> int:
        return self.losing_pair[0]

    @property
    def losing_cell_b(self) -> int:
        return self.losing_pair[1]

    @property
    def gaining_cell_a(self) -> int:
        return self.gaining_pair[0]

    @property
    def gaining_cell_b(self) -> int:
        return self.gaining_pair[1]


def build_position_analysis_artifact(
    position_path: str | Path,
    output_path: str | Path,
    *,
    cell_tracked_labels_path: str | Path | None = None,
    nucleus_tracked_labels_path: str | Path | None = None,
    edge_extraction_params: dict | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Build the canonical per-position analysis HDF5 artifact."""
    total = 6
    position_path = Path(position_path)
    output_path = Path(output_path)
    params = dict(edge_extraction_params or {})

    cell_labels_path = Path(cell_tracked_labels_path) if cell_tracked_labels_path else (
        position_path / "cell" / "tracked_labels.tif"
    )
    nucleus_labels_path = (
        Path(nucleus_tracked_labels_path)
        if nucleus_tracked_labels_path
        else position_path / "nucleus" / "tracked_labels.tif"
    )
    cell_stack = _read_label_stack(cell_labels_path)
    nucleus_stack = _read_label_stack(nucleus_labels_path)
    _report_progress(progress_cb, 1, total, "read labels")
    _validate_cell_nucleus_identity(cell_stack, nucleus_stack)
    _report_progress(progress_cb, 2, total, "validate IDs")

    cell_columns = _extract_cell_columns(cell_stack)
    _report_progress(progress_cb, 3, total, "extract cells")
    edge_records = _extract_edges(cell_stack)
    _report_progress(progress_cb, 4, total, "extract edges")
    assignments, t1_events = _assign_ids_to_records(edge_records)
    _report_progress(progress_cb, 5, total, "assign edge IDs/T1")
    edge_columns, coord_y, coord_x = _extract_edge_columns(edge_records, assignments, t1_events)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        provenance = h5.create_group("provenance")
        provenance.attrs["source_position_path"] = str(position_path)
        provenance.attrs["cell_tracked_labels_path"] = str(cell_labels_path)
        provenance.attrs["nucleus_tracked_labels_path"] = str(nucleus_labels_path)
        provenance.attrs["edge_extraction_params_json"] = json.dumps(
            params, sort_keys=True, separators=(",", ":")
        )
        provenance.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        provenance.attrs["cellflow_version"] = _cellflow_version()

        _write_column_group(h5.create_group("cells/table", track_order=True), cell_columns)
        h5.create_group("cells/measurements")

        edges_group = h5.create_group("edges/table", track_order=True)
        _write_column_group(edges_group, edge_columns)
        edges_group["t1_event_id"].attrs["null_sentinel"] = NULL_INT

        coords = h5.create_group("edges/coordinates", track_order=True)
        coords.create_dataset("y", data=coord_y)
        coords.create_dataset("x", data=coord_x)
        h5.create_group("edges/measurements")

        _write_t1_table(h5.create_group("t1_events/table", track_order=True), t1_events)
    _report_progress(progress_cb, 6, total, "write HDF5")

    return output_path


def assign_persistent_edge_ids(
    frame_edges: list[Iterable[tuple[int, int]]],
) -> tuple[list[dict[tuple[int, int], int]], list[T1Record]]:
    """Assign deterministic edge IDs, linking losing/gaining pairs through T1s."""
    assignments: list[dict[tuple[int, int], int]] = []
    pair_to_id: dict[tuple[int, int], int] = {}
    events: list[T1Record] = []
    next_edge_id = 1

    normalized_frames = [
        {tuple(sorted((int(a), int(b)))) for a, b in pairs}
        for pairs in frame_edges
    ]

    for frame_idx, pairs in enumerate(normalized_frames):
        if frame_idx > 0:
            prev = normalized_frames[frame_idx - 1]
            for losing, gaining in _detect_t1_pairs(prev, pairs):
                edge_id = pair_to_id.get(losing)
                if edge_id is None:
                    edge_id = next_edge_id
                    next_edge_id += 1
                    pair_to_id[losing] = edge_id
                pair_to_id[gaining] = edge_id
                events.append(
                    T1Record(
                        t1_event_id=len(events) + 1,
                        frame=frame_idx - 1,
                        edge_id=edge_id,
                        losing_pair=losing,
                        gaining_pair=gaining,
                        location_y=np.nan,
                        location_x=np.nan,
                    )
                )

        frame_assignment: dict[tuple[int, int], int] = {}
        for pair in sorted(pairs):
            if pair not in pair_to_id:
                pair_to_id[pair] = next_edge_id
                next_edge_id += 1
            frame_assignment[pair] = pair_to_id[pair]
        assignments.append(frame_assignment)

    return assignments, events


def _read_label_stack(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.asarray(tifffile.imread(path))
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D tracked label TIFF at {path}, got shape {arr.shape}")
    return arr.astype(np.int64, copy=False)


def _validate_cell_nucleus_identity(cell_stack: np.ndarray, nucleus_stack: np.ndarray) -> None:
    if cell_stack.shape != nucleus_stack.shape:
        raise ValueError(
            "cell_id == nucleus_id invariant requires matching cell and nucleus stack shapes"
        )
    for frame_idx in range(cell_stack.shape[0]):
        cell_ids = set(np.unique(cell_stack[frame_idx]).astype(int))
        nucleus_ids = set(np.unique(nucleus_stack[frame_idx]).astype(int))
        cell_ids.discard(0)
        nucleus_ids.discard(0)
        if cell_ids != nucleus_ids:
            cell_only = sorted(int(label) for label in cell_ids - nucleus_ids)
            nucleus_only = sorted(int(label) for label in nucleus_ids - cell_ids)
            raise ValueError(
                "cell_id == nucleus_id invariant failed for frame "
                f"{frame_idx}: cell labels present only in cell stack: {cell_only}; "
                f"nucleus labels present only in nucleus stack: {nucleus_only}"
            )


def _extract_cell_columns(label_stack: np.ndarray) -> dict[str, np.ndarray]:
    rows: list[dict[str, int | float | str]] = []
    for frame_idx, frame in enumerate(label_stack):
        for prop in sorted(regionprops(frame), key=lambda item: item.label):
            min_y, min_x, max_y, max_x = prop.bbox
            rows.append(
                {
                    "frame": frame_idx,
                    "cell_id": int(prop.label),
                    "class_label": "",
                    "area": float(prop.area),
                    "centroid_y": float(prop.centroid[0]),
                    "centroid_x": float(prop.centroid[1]),
                    "perimeter": float(prop.perimeter),
                    "bbox_min_y": int(min_y),
                    "bbox_min_x": int(min_x),
                    "bbox_max_y": int(max_y),
                    "bbox_max_x": int(max_x),
                }
            )
    return _columns_from_rows(
        rows,
        [
            "frame",
            "cell_id",
            "class_label",
            "area",
            "centroid_y",
            "centroid_x",
            "perimeter",
            "bbox_min_y",
            "bbox_min_x",
            "bbox_max_y",
            "bbox_max_x",
        ],
    )


def _extract_edges(label_stack: np.ndarray) -> list[EdgeRecord]:
    records: list[EdgeRecord] = []
    for frame_idx, frame in enumerate(label_stack):
        records.extend(_extract_frame_cell_edges(frame, frame_idx))
        records.extend(_extract_frame_border_edges(frame, frame_idx))
    return sorted(records, key=lambda row: (row.frame, row.kind != "cell_cell", row.pair))


def _extract_frame_cell_edges(frame: np.ndarray, frame_idx: int) -> list[EdgeRecord]:
    points_by_pair: dict[tuple[int, int], list[tuple[float, float]]] = {}

    left = frame[:, :-1]
    right = frame[:, 1:]
    ys, xs = np.where((left != right) & (left > 0) & (right > 0))
    for y, x in zip(ys, xs):
        pair = tuple(sorted((int(left[y, x]), int(right[y, x]))))
        points_by_pair.setdefault(pair, []).append((float(y), float(x) + 0.5))

    top = frame[:-1, :]
    bottom = frame[1:, :]
    ys, xs = np.where((top != bottom) & (top > 0) & (bottom > 0))
    for y, x in zip(ys, xs):
        pair = tuple(sorted((int(top[y, x]), int(bottom[y, x]))))
        points_by_pair.setdefault(pair, []).append((float(y) + 0.5, float(x)))

    rows = []
    for pair in sorted(points_by_pair):
        for segment in _coordinate_segments(np.asarray(points_by_pair[pair], dtype=float)):
            coords = _order_coordinates(segment)
            rows.append(_edge_record(frame_idx, pair, "cell_cell", "", coords))
    return rows


def _extract_frame_border_edges(frame: np.ndarray, frame_idx: int) -> list[EdgeRecord]:
    points_by_cell: dict[int, list[tuple[float, float]]] = {}
    h, w = frame.shape

    for y in range(h):
        for x in range(w):
            cell_id = int(frame[y, x])
            if cell_id <= 0:
                continue
            if y == 0:
                points_by_cell.setdefault(cell_id, []).append((float(y) - 0.5, float(x)))
            if y == h - 1:
                points_by_cell.setdefault(cell_id, []).append((float(y) + 0.5, float(x)))
            if x == 0:
                points_by_cell.setdefault(cell_id, []).append((float(y), float(x) - 0.5))
            if x == w - 1:
                points_by_cell.setdefault(cell_id, []).append((float(y), float(x) + 0.5))

            if y > 0 and frame[y - 1, x] == 0:
                points_by_cell.setdefault(cell_id, []).append((float(y) - 0.5, float(x)))
            if y < h - 1 and frame[y + 1, x] == 0:
                points_by_cell.setdefault(cell_id, []).append((float(y) + 0.5, float(x)))
            if x > 0 and frame[y, x - 1] == 0:
                points_by_cell.setdefault(cell_id, []).append((float(y), float(x) - 0.5))
            if x < w - 1 and frame[y, x + 1] == 0:
                points_by_cell.setdefault(cell_id, []).append((float(y), float(x) + 0.5))

    rows = []
    for cell_id in sorted(points_by_cell):
        for segment in _coordinate_segments(np.asarray(points_by_cell[cell_id], dtype=float)):
            coords = _order_coordinates(segment)
            rows.append(_edge_record(frame_idx, (cell_id, 0), "border", "border", coords))
    return rows


def _edge_record(
    frame_idx: int,
    pair: tuple[int, int],
    kind: str,
    edge_label: str,
    coords: np.ndarray,
) -> EdgeRecord:
    midpoint = coords[len(coords) // 2] if len(coords) else np.array([np.nan, np.nan])
    return EdgeRecord(
        frame=frame_idx,
        pair=pair,
        kind=kind,
        edge_label=edge_label,
        length=_path_length(coords),
        midpoint_y=float(midpoint[0]),
        midpoint_x=float(midpoint[1]),
        coordinates=coords,
    )


def _assign_ids_to_records(
    records: list[EdgeRecord],
) -> tuple[list[dict[tuple[int, int], int]], list[T1Record]]:
    max_frame = max((record.frame for record in records), default=-1)
    frame_cell_edges: list[set[tuple[int, int]]] = []
    for frame_idx in range(max_frame + 1):
        frame_cell_edges.append(
            {
                record.pair
                for record in records
                if record.frame == frame_idx and record.kind == "cell_cell"
            }
        )
    assignments, events = assign_persistent_edge_ids(frame_cell_edges)
    border_next_id = (
        max((edge_id for frame in assignments for edge_id in frame.values()), default=0) + 1
    )
    border_ids: dict[tuple[int, int], int] = {}
    for record in records:
        if record.kind != "border" or record.pair in border_ids:
            continue
        border_ids[record.pair] = border_next_id
        border_next_id += 1
    merged = []
    for frame_idx, frame_assignment in enumerate(assignments):
        full = dict(frame_assignment)
        for pair, edge_id in border_ids.items():
            if any(record.frame == frame_idx and record.pair == pair for record in records):
                full[pair] = edge_id
        merged.append(full)

    t1_by_key = {(event.frame, event.losing_pair): event for event in events}
    records_by_key = {(record.frame, record.pair): record for record in records}
    resolved_events = []
    for event in events:
        rec = records_by_key.get((event.frame, event.losing_pair))
        if rec is None:
            resolved_events.append(event)
            continue
        resolved_events.append(
            T1Record(
                t1_event_id=event.t1_event_id,
                frame=event.frame,
                edge_id=event.edge_id,
                losing_pair=event.losing_pair,
                gaining_pair=event.gaining_pair,
                location_y=rec.midpoint_y,
                location_x=rec.midpoint_x,
            )
        )
    return merged, resolved_events


def _extract_edge_columns(
    records: list[EdgeRecord],
    assignments: list[dict[tuple[int, int], int]],
    t1_events: list[T1Record],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rows = []
    coord_y: list[float] = []
    coord_x: list[float] = []
    t1_by_key = {(event.frame, event.losing_pair): event for event in t1_events}

    for record in records:
        offset = len(coord_y)
        coords = record.coordinates
        coord_y.extend(coords[:, 0].astype(float).tolist())
        coord_x.extend(coords[:, 1].astype(float).tolist())
        event = t1_by_key.get((record.frame, record.pair))
        cell_a, cell_b = record.pair
        rows.append(
            {
                "frame": record.frame,
                "edge_id": assignments[record.frame][record.pair],
                "cell_a": int(cell_a),
                "cell_b": int(cell_b),
                "kind": record.kind,
                "edge_label": record.edge_label,
                "is_t1_frame": event is not None,
                "t1_event_id": event.t1_event_id if event is not None else NULL_INT,
                "length": record.length,
                "midpoint_y": record.midpoint_y,
                "midpoint_x": record.midpoint_x,
                "coord_offset": offset,
                "coord_count": len(coords),
            }
        )
    return (
        _columns_from_rows(
            rows,
            [
                "frame",
                "edge_id",
                "cell_a",
                "cell_b",
                "kind",
                "edge_label",
                "is_t1_frame",
                "t1_event_id",
                "length",
                "midpoint_y",
                "midpoint_x",
                "coord_offset",
                "coord_count",
            ],
        ),
        np.asarray(coord_y, dtype=float),
        np.asarray(coord_x, dtype=float),
    )


def _detect_t1_pairs(
    prev_edges: set[tuple[int, int]],
    next_edges: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    removed = sorted(prev_edges - next_edges)
    added = sorted(next_edges - prev_edges)
    events = []
    used_removed = set()
    used_added = set()
    for lost in removed:
        for gained in added:
            if lost in used_removed or gained in used_added:
                continue
            if _is_valid_t1(lost, gained, prev_edges, next_edges):
                events.append((lost, gained))
                used_removed.add(lost)
                used_added.add(gained)
    return events


def _is_valid_t1(
    lost: tuple[int, int],
    gained: tuple[int, int],
    prev_edges: set[tuple[int, int]],
    next_edges: set[tuple[int, int]],
) -> bool:
    all_cells = set(lost) | set(gained)
    if len(all_cells) != 4:
        return False
    lost_cells = tuple(sorted(lost))
    gained_cells = tuple(sorted(gained))
    connectors = [
        tuple(sorted((lost_cells[0], gained_cells[0]))),
        tuple(sorted((lost_cells[0], gained_cells[1]))),
        tuple(sorted((lost_cells[1], gained_cells[0]))),
        tuple(sorted((lost_cells[1], gained_cells[1]))),
    ]
    return all(edge in prev_edges and edge in next_edges for edge in connectors)


def _order_coordinates(coords: np.ndarray) -> np.ndarray:
    if len(coords) <= 2:
        return coords[np.lexsort((coords[:, 1], coords[:, 0]))]

    coords = np.unique(np.asarray(coords, dtype=float), axis=0)
    neighbors = _coordinate_neighbors(coords)
    endpoints = [idx for idx, adjacent in enumerate(neighbors) if len(adjacent) == 1]
    start = min(endpoints or range(len(coords)), key=lambda idx: (coords[idx, 0], coords[idx, 1]))

    ordered_indices = [start]
    visited = {start}
    prev_idx: int | None = None
    current_idx = start
    while len(visited) < len(coords):
        candidates = [idx for idx in neighbors[current_idx] if idx not in visited]
        if candidates:
            next_idx = min(
                candidates,
                key=lambda idx: (
                    _turn_cost(coords[prev_idx], coords[current_idx], coords[idx])
                    if prev_idx is not None
                    else 0.0,
                    coords[idx, 0],
                    coords[idx, 1],
                ),
            )
        else:
            remaining = [idx for idx in range(len(coords)) if idx not in visited]
            next_idx = min(
                remaining,
                key=lambda idx: (
                    float(np.linalg.norm(coords[current_idx] - coords[idx])),
                    coords[idx, 0],
                    coords[idx, 1],
                ),
            )
        ordered_indices.append(next_idx)
        visited.add(next_idx)
        prev_idx = current_idx
        current_idx = next_idx
    return coords[np.asarray(ordered_indices, dtype=np.intp)]


def _coordinate_neighbors(coords: np.ndarray) -> list[list[int]]:
    scaled = np.rint(coords * 2.0).astype(np.int64)
    point_to_idx = {tuple(point): idx for idx, point in enumerate(scaled)}
    neighbor_offsets = [
        (dy, dx)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
        if (dy or dx) and (dy * dy + dx * dx) <= 4
    ]
    neighbors: list[list[int]] = []
    for y, x in scaled:
        adjacent = []
        for dy, dx in neighbor_offsets:
            idx = point_to_idx.get((int(y + dy), int(x + dx)))
            if idx is not None:
                adjacent.append(idx)
        neighbors.append(sorted(adjacent, key=lambda idx: (coords[idx, 0], coords[idx, 1])))
    return neighbors


def _turn_cost(prev: np.ndarray, current: np.ndarray, candidate: np.ndarray) -> float:
    incoming = current - prev
    outgoing = candidate - current
    incoming_norm = float(np.linalg.norm(incoming))
    outgoing_norm = float(np.linalg.norm(outgoing))
    if incoming_norm == 0.0 or outgoing_norm == 0.0:
        return 0.0
    return float(1.0 - np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm))


def _coordinate_segments(coords: np.ndarray) -> list[np.ndarray]:
    if len(coords) <= 1:
        return [coords]

    coords = np.unique(np.asarray(coords, dtype=float), axis=0)
    neighbors = _coordinate_neighbors(coords)
    unused_edges = {
        tuple(sorted((idx, neighbor_idx)))
        for idx, adjacent in enumerate(neighbors)
        for neighbor_idx in adjacent
        if idx != neighbor_idx
    }
    segments: list[np.ndarray] = []

    while unused_edges:
        start_idx = _next_trail_start(coords, neighbors, unused_edges)
        next_idx = _next_unused_neighbor(coords, start_idx, None, neighbors, unused_edges)
        if next_idx is None:
            unused_edges = {edge for edge in unused_edges if start_idx not in edge}
            continue

        path = [start_idx]
        prev_idx = start_idx
        current_idx = next_idx
        unused_edges.remove(tuple(sorted((prev_idx, current_idx))))
        path.append(current_idx)

        while len(neighbors[current_idx]) == 2:
            next_idx = _next_unused_neighbor(
                coords, current_idx, prev_idx, neighbors, unused_edges
            )
            if next_idx is None:
                break
            unused_edges.remove(tuple(sorted((current_idx, next_idx))))
            path.append(next_idx)
            prev_idx, current_idx = current_idx, next_idx

        segments.append(coords[np.asarray(path, dtype=np.intp)])

    return sorted(segments, key=lambda segment: (segment[0, 0], segment[0, 1], len(segment)))


def _next_trail_start(
    coords: np.ndarray,
    neighbors: list[list[int]],
    unused_edges: set[tuple[int, int]],
) -> int:
    incident = {idx for edge in unused_edges for idx in edge}
    preferred = [idx for idx in incident if len(neighbors[idx]) != 2]
    return min(preferred or incident, key=lambda idx: (coords[idx, 0], coords[idx, 1]))


def _next_unused_neighbor(
    coords: np.ndarray,
    current_idx: int,
    prev_idx: int | None,
    neighbors: list[list[int]],
    unused_edges: set[tuple[int, int]],
) -> int | None:
    candidates = [
        idx
        for idx in neighbors[current_idx]
        if tuple(sorted((current_idx, idx))) in unused_edges
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda idx: (
            _turn_cost(coords[prev_idx], coords[current_idx], coords[idx])
            if prev_idx is not None
            else 0.0,
            coords[idx, 0],
            coords[idx, 1],
        ),
    )


def _path_length(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    return float(np.sqrt(np.sum(diffs * diffs, axis=1)).sum())


def _columns_from_rows(rows: list[dict], names: list[str]) -> dict[str, np.ndarray]:
    columns = {}
    for name in names:
        values = [row[name] for row in rows]
        if name in {"kind", "edge_label", "class_label"}:
            columns[name] = np.asarray(values, dtype=object)
        elif name == "is_t1_frame":
            columns[name] = np.asarray(values, dtype=bool)
        elif name in {
            "area",
            "centroid_y",
            "centroid_x",
            "perimeter",
            "length",
            "midpoint_y",
            "midpoint_x",
            "location_y",
            "location_x",
        }:
            columns[name] = np.asarray(values, dtype=float)
        else:
            columns[name] = np.asarray(values, dtype=np.int64)
    return columns


def _write_column_group(group: h5py.Group, columns: dict[str, np.ndarray]) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    for name, values in columns.items():
        if values.dtype == object:
            group.create_dataset(name, data=values, dtype=string_dtype)
        else:
            group.create_dataset(name, data=values)


def _write_t1_table(group: h5py.Group, events: list[T1Record]) -> None:
    rows = [
        {
            "t1_event_id": event.t1_event_id,
            "frame": event.frame,
            "edge_id": event.edge_id,
            "losing_cell_a": event.losing_cell_a,
            "losing_cell_b": event.losing_cell_b,
            "gaining_cell_a": event.gaining_cell_a,
            "gaining_cell_b": event.gaining_cell_b,
            "location_y": event.location_y,
            "location_x": event.location_x,
        }
        for event in events
    ]
    columns = _columns_from_rows(
        rows,
        [
            "t1_event_id",
            "frame",
            "edge_id",
            "losing_cell_a",
            "losing_cell_b",
            "gaining_cell_a",
            "gaining_cell_b",
            "location_y",
            "location_x",
        ],
    )
    _write_column_group(group, columns)


def _report_progress(
    progress_cb: Callable[[int, int, str], None] | None,
    done: int,
    total: int,
    message: str,
) -> None:
    if progress_cb is not None:
        progress_cb(done, total, message)


def _cellflow_version() -> str:
    try:
        from importlib.metadata import version

        return version("cellflow")
    except Exception:
        return "unknown"


--------------------------------------------------------------------------------
FILE: src/cellflow/core/__init__.py
--------------------------------------------------------------------------------




--------------------------------------------------------------------------------
FILE: src/cellflow/core/data_prep.py
--------------------------------------------------------------------------------

"""Backend logic for raw data import and preparation."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import tifffile
from scipy.interpolate import interp1d
from scipy.ndimage import shift as nd_shift
from scipy.optimize import least_squares
from skimage.transform import downscale_local_mean

from cellflow.core.paths import stage_dir


@dataclass
class DatasetConfig:
    """Raw data source configuration."""
    ndtiff_path: str
    root_dir: str
    positions: list[int]
    xy_downsample: int = 3
    frame_start: int = 0
    frame_end: int = -1


# Channel indices (0-based) in the NDTiff dataset
_CH_642 = 3  # CSU642  — nuclear marker
_CH_488 = 1  # CSU488  — membrane marker
_CH_561 = 2  # CSU561  — NLS-mCherry marker


def discover_metadata(ndtiff_path: str) -> dict:
    """Open an NDTiff dataset and return its metadata."""
    from ndtiff import Dataset

    ds = Dataset(ndtiff_path)
    axes = ds.axes
    positions = sorted(axes.get("position", axes.get("p", [0])))

    pixel_size_um: Optional[float] = None
    time_interval_s: Optional[float] = None
    try:
        summary = getattr(ds, "summary_metadata", None) or {}
        px = summary.get("PixelSizeUm")
        if px is not None and float(px) > 0:
            pixel_size_um = float(px)
        interval_ms = summary.get("Interval_ms")
        if interval_ms is None:
            interval_ms = summary.get("CustomIntervals_ms")
            if isinstance(interval_ms, list) and interval_ms:
                interval_ms = interval_ms[0]
        if interval_ms is not None:
            time_interval_s = float(interval_ms) / 1000.0
    except Exception:
        pass

    if pixel_size_um is None:
        try:
            coords_list = ds.get_image_coordinates_list()
            if coords_list:
                img_meta = ds.read_metadata(**coords_list[0])
                px = img_meta.get("PixelSizeUm")
                if px is not None and float(px) > 0:
                    pixel_size_um = float(px)
        except Exception:
            pass

    return {
        "positions": positions,
        "pixel_size_um": pixel_size_um,
        "time_interval_s": time_interval_s,
    }


def _raw_dir(root_dir, pos):
    return stage_dir(root_dir, pos, "raw_import")


def _nucleus_4d_path(root_dir, pos):
    return _raw_dir(root_dir, pos) / "nucleus_zavg.tif"


def _nucleus_3dt_path(root_dir, pos):
    return _raw_dir(root_dir, pos) / "nucleus_3dt.tif"


def _cell_4d_path(root_dir, pos):
    return _raw_dir(root_dir, pos) / "cell_zavg.tif"


def _cell_3dt_path(root_dir, pos):
    return _raw_dir(root_dir, pos) / "cell_3dt.tif"


def _nls_4d_path(root_dir, pos):
    return _raw_dir(root_dir, pos) / "NLS_zavg.tif"


def _nls_3dt_path(root_dir, pos):
    return _raw_dir(root_dir, pos) / "NLS_3dt.tif"


def _z_shift_csv_path(root_dir, pos):
    return _raw_dir(root_dir, pos) / "z_shift.csv"


def _read_z_stack(
    ds: Any, position: int, time: int, channel: int, z_indices: list[int]
) -> np.ndarray:
    slices = []
    for z in z_indices:
        img = ds.read_image(position=position, time=time, channel=channel, z=z)
        if img is None:
            img = np.zeros((ds.image_height, ds.image_width), dtype=np.uint16)
        slices.append(img)
    return np.stack(slices, axis=0)


def _xy_avg(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr.astype(np.uint16)
    if arr.ndim == 3:
        downsampled = downscale_local_mean(arr, (1, factor, factor))
    else:
        downsampled = downscale_local_mean(arr, (factor, factor))
    return downsampled.astype(np.uint16)


def _mean_profile(volume: np.ndarray) -> np.ndarray:
    return volume.astype(np.float32).mean(axis=(1, 2))


def _smooth_profile(profile: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if profile.size < 3 or sigma <= 0:
        return profile.astype(np.float64)
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(profile.astype(np.float64), sigma=sigma, mode="nearest")


def _fill_profile_nans(profile: np.ndarray) -> np.ndarray:
    y = profile.astype(np.float64)
    mask = np.isfinite(y)
    if mask.all():
        return y
    if not mask.any():
        return np.zeros_like(y, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64)
    y[~mask] = np.interp(x[~mask], x[mask], y[mask])
    return y


def _double_sigmoid_profile(
    z: np.ndarray,
    offset: float,
    amplitude: float,
    center: float,
    span: float,
    left_width: float,
    right_width: float,
    slope: float,
) -> np.ndarray:
    left_edge = center - 0.5 * span
    right_edge = center + 0.5 * span
    left = 1.0 / (1.0 + np.exp(-(z - left_edge) / left_width))
    right = 1.0 / (1.0 + np.exp(-(z - right_edge) / right_width))
    return offset + slope * (z - center) + amplitude * (left - right)


def _fit_double_sigmoid_profile(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = _fill_profile_nans(_smooth_profile(profile))
    z = np.arange(y.size, dtype=np.float64)

    y_min, y_max = float(np.min(y)), float(np.max(y))
    y_range = max(y_max - y_min, 1.0)

    half_level = y_min + 0.5 * y_range
    support = np.flatnonzero(y >= half_level)
    if support.size >= 2:
        center0 = float(0.5 * (support[0] + support[-1]))
        span0 = float(max(2.0, support[-1] - support[0] + 1))
    else:
        weights = np.clip(y - y_min, 0.0, None)
        center0 = float(np.average(z, weights=weights)) if weights.sum() > 0 else float(0.5 * (y.size - 1))
        span0 = float(max(2.0, min(y.size - 1.0, y.size / 3.0)))

    width0 = float(max(0.5, min(span0 / 3.0, y.size / 4.0)))
    p0 = np.array([y_min, y_range, center0, span0, width0, width0, 0.0], dtype=np.float64)

    lower = np.array([y_min - 2.0 * y_range, 0.0, 0.0, 0.5, 0.1, 0.1, -y_range], dtype=np.float64)
    upper = np.array([y_max + 2.0 * y_range, 20.0 * y_range, float(y.size - 1), float(max(1.0, y.size - 1)), float(max(1.0, y.size)), float(max(1.0, y.size)), y_range], dtype=np.float64)

    def residuals(params: np.ndarray) -> np.ndarray:
        return _double_sigmoid_profile(z, *params) - y

    result = least_squares(residuals, p0, bounds=(lower, upper), loss="soft_l1", f_scale=max(1.0, 0.1 * y_range), max_nfev=2000)
    params = result.x if result.success else p0
    return _double_sigmoid_profile(z, *params), params


def _affine_fit_mse(reference: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(reference) & np.isfinite(target)
    if mask.sum() < 2:
        return float("inf"), 1.0, 0.0
    ref, tgt = reference[mask].astype(np.float64), target[mask].astype(np.float64)
    design = np.column_stack([ref, np.ones_like(ref)])
    coeffs, _, _, _ = np.linalg.lstsq(design, tgt, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    return float(np.mean((tgt - (a * ref + b)) ** 2)), a, b


def _shift_profile(profile: np.ndarray, shift_slices: float) -> np.ndarray:
    z = np.arange(profile.size, dtype=np.float64)
    interpolator = interp1d(z, profile.astype(np.float64), kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)
    return interpolator(z - shift_slices)


def _estimate_z_shift(reference_profile: np.ndarray, target_profile: np.ndarray, max_shift_slices: float) -> tuple[float, float, float, float]:
    ref_fit, ref_params = _fit_double_sigmoid_profile(reference_profile)
    tgt_fit, tgt_params = _fit_double_sigmoid_profile(target_profile)
    shift_slices = float(np.clip(float(tgt_params[2]) - float(ref_params[2]), -max_shift_slices, max_shift_slices))
    shifted_ref = _shift_profile(ref_fit, shift_slices)
    mse, scale, offset = _affine_fit_mse(shifted_ref, tgt_fit)
    return shift_slices, scale, offset, mse


def _shift_volume(volume: np.ndarray, shift_slices: float) -> np.ndarray:
    if abs(shift_slices) < 1e-9:
        return volume
    shifted = nd_shift(volume.astype(np.float32), shift=(shift_slices, 0.0, 0.0), order=1, mode="constant", cval=0.0, prefilter=False)
    return np.clip(np.rint(shifted), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def _read_corrected_volume(
    ds: Any, position: int, time: int, channel: int, z_indices: list[int], xy_factor: int, z_shift_slices: float
) -> np.ndarray:
    volume = _read_z_stack(ds, position, time, channel, z_indices)
    volume = _xy_avg(volume, xy_factor)
    volume = _shift_volume(volume, -z_shift_slices)
    return volume


def _read_corrected_z_avg(
    ds: Any, position: int, time: int, channel: int, z_indices: list[int], xy_factor: int, z_shift_slices: float
) -> np.ndarray:
    volume = _read_corrected_volume(ds, position, time, channel, z_indices, xy_factor, z_shift_slices)
    return volume.mean(axis=0).astype(np.uint16)


def run(config: DatasetConfig, pos: int, overwrite: bool = False) -> Generator[tuple[int, int, str], None, None]:
    """Export raw NDTiff data for one position as Z-averages."""
    from ndtiff import Dataset
    ds = Dataset(config.ndtiff_path)
    axes = ds.axes
    available_positions = sorted(axes.get("position", axes.get("p", [0])))
    if pos not in available_positions:
        raise ValueError(f"Position {pos} not found.")

    available_times = sorted(axes.get("time", [0]))
    z_indices = sorted(axes.get("z", [0]))
    if not available_times:
        raise ValueError("No timepoints found.")
    start = max(0, int(config.frame_start))
    end = int(config.frame_end)
    if end < 0:
        end = len(available_times) - 1
    end = min(end, len(available_times) - 1)
    if start > end:
        raise ValueError(f"Invalid frame range: start {start} is after end {end}.")
    all_times = available_times[start:end + 1]
    max_shift_slices = max(1.0, min(8.0, (len(z_indices) - 1) / 2.0))

    out_dir = _raw_dir(config.root_dir, pos)
    out_dir.mkdir(parents=True, exist_ok=True)
    nuc_out, cell_out = _nucleus_4d_path(config.root_dir, pos), _cell_4d_path(config.root_dir, pos)
    nuc_3dt_out, cell_3dt_out = _nucleus_3dt_path(config.root_dir, pos), _cell_3dt_path(config.root_dir, pos)
    nls_out, nls_3dt_out = _nls_4d_path(config.root_dir, pos), _nls_3dt_path(config.root_dir, pos)
    shift_out, run_params_out = _z_shift_csv_path(config.root_dir, pos), out_dir / "run_params.json"

    if not overwrite and all(p.exists() for p in [nuc_out, cell_out, nls_out, nuc_3dt_out, cell_3dt_out, nls_3dt_out, shift_out, run_params_out]):
        yield (len(all_times), len(all_times), "done")
        return

    reference_profile = None
    shift_rows, z_shifts = [], {}
    for i, t in enumerate(all_times):
        profile = _mean_profile(_xy_avg(_read_z_stack(ds, pos, t, _CH_488, z_indices), config.xy_downsample))
        if reference_profile is None:
            z_shift_slices, scale, offset, mse = 0.0, 1.0, 0.0, 0.0
            reference_profile = profile
        else:
            z_shift_slices, scale, offset, mse = _estimate_z_shift(reference_profile, profile, max_shift_slices)
        z_shifts[t] = z_shift_slices
        shift_rows.append({"time": float(t), "z_shift_slices": z_shift_slices, "intensity_scale": scale, "intensity_offset": offset, "fit_mse": mse})
        yield (i + 1, len(all_times), "z-shift")

    with shift_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "z_shift_slices", "intensity_scale", "intensity_offset", "fit_mse"])
        writer.writeheader()
        writer.writerows(shift_rows)

    # Export nucleus volumes
    h = ds.image_height // config.xy_downsample
    w = ds.image_width // config.xy_downsample
    nz = len(z_indices)
    nuc_4d = np.empty((len(all_times), nz, h, w), dtype=np.uint16)
    nuc_stack = np.empty((len(all_times), h, w), dtype=np.uint16)
    for i, t in enumerate(all_times):
        vol = _read_corrected_volume(ds, pos, t, _CH_642, z_indices, config.xy_downsample, z_shifts[t])
        nuc_4d[i] = vol
        nuc_stack[i] = vol.mean(axis=0).astype(np.uint16)
        yield (i + 1, len(all_times), "nucleus")
    tifffile.imwrite(str(nuc_3dt_out), nuc_4d, compression="zlib", metadata={"axes": "TZYX"})
    tifffile.imwrite(str(nuc_out), nuc_stack, compression="zlib", metadata={"axes": "TYX"})

    # Export cell volumes
    cell_4d = np.empty((len(all_times), nz, h, w), dtype=np.uint16)
    cell_stack = np.empty((len(all_times), h, w), dtype=np.uint16)
    for i, t in enumerate(all_times):
        vol = _read_corrected_volume(ds, pos, t, _CH_488, z_indices, config.xy_downsample, z_shifts[t])
        cell_4d[i] = vol
        cell_stack[i] = vol.mean(axis=0).astype(np.uint16)
        yield (i + 1, len(all_times), "cell")
    tifffile.imwrite(str(cell_3dt_out), cell_4d, compression="zlib", metadata={"axes": "TZYX"})
    tifffile.imwrite(str(cell_out), cell_stack, compression="zlib", metadata={"axes": "TYX"})

    # Export NLS-mCherry volumes from CSU561
    nls_4d = np.empty((len(all_times), nz, h, w), dtype=np.uint16)
    nls_stack = np.empty((len(all_times), h, w), dtype=np.uint16)
    for i, t in enumerate(all_times):
        vol = _read_corrected_volume(ds, pos, t, _CH_561, z_indices, config.xy_downsample, z_shifts[t])
        nls_4d[i] = vol
        nls_stack[i] = vol.mean(axis=0).astype(np.uint16)
        yield (i + 1, len(all_times), "NLS")
    tifffile.imwrite(str(nls_3dt_out), nls_4d, compression="zlib", metadata={"axes": "TZYX"})
    tifffile.imwrite(str(nls_out), nls_stack, compression="zlib", metadata={"axes": "TYX"})

    # Metadata
    meta = discover_metadata(config.ndtiff_path)
    run_params = {
        "stage": "raw", "pos": pos, "xy_downsample": config.xy_downsample,
        "frame_start": start, "frame_end": end,
        "pixel_size_um": (meta["pixel_size_um"] or 0.1) * config.xy_downsample,
        "time_interval_s": meta["time_interval_s"] or 60.0,
    }
    run_params_out.write_text(json.dumps(run_params, indent=2), encoding="utf-8")


--------------------------------------------------------------------------------
FILE: src/cellflow/core/logging.py
--------------------------------------------------------------------------------

"""StageLogger — JSON-lines logger writing to per-position pipeline.log."""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional


class StageLogger:
    """Context manager that appends JSON-lines entries to pipeline.log."""

    def __init__(
        self,
        pipeline_log: Path,
        stage_name: str,
    ) -> None:
        self._pipeline_log = pipeline_log
        self._stage_name = stage_name

    def _write(self, level: str, message: str) -> None:
        entry = json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "stage": self._stage_name,
                "level": level,
                "message": message,
            }
        )
        self._pipeline_log.parent.mkdir(parents=True, exist_ok=True)
        with self._pipeline_log.open("a", encoding="utf-8") as fh:
            fh.write(entry + "\n")

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARNING", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def __enter__(self) -> "StageLogger":
        self.info("Stage started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.info("Stage completed successfully")
        else:
            self.error(f"Stage failed: {exc_val}")
        return False


--------------------------------------------------------------------------------
FILE: src/cellflow/core/paths.py
--------------------------------------------------------------------------------

"""Unified path resolution for the CellFlow pipeline directory layout."""
from __future__ import annotations

from pathlib import Path

# Authoritative stage-name → output-directory mapping.
STAGE_DIRS: dict[str, str] = {
    "raw_import": "0_input",
    "cellpose": "1_cellpose",
    "nucleus": "2_nucleus",
    "cell": "3_cell",
    "analysis": "4_analysis",
}


def pos_dir(root: Path | str, pos: int) -> Path:
    """Return <root>/pos<pos:02d>."""
    return Path(root) / f"pos{pos:02d}"


def stage_dir(root: Path | str, pos: int, stage: str) -> Path:
    """Return the output directory for *stage* at position *pos*."""
    dirname = STAGE_DIRS.get(stage, stage)
    return pos_dir(root, pos) / dirname


def log_path(root: Path | str, pos: int) -> Path:
    """Return the path to the pipeline log for a position."""
    return pos_dir(root, pos) / "pipeline.log"


--------------------------------------------------------------------------------
FILE: src/cellflow/correction/__init__.py
--------------------------------------------------------------------------------

from cellflow.correction.labels import (
    apply_gamma,
    clean_stranded_pixels,
    draw_cell_path,
    erase_cell,
    expand_label_to_foreground,
    fill_label_holes,
    fix_label_semiholes,
    merge_cells,
    split_across,
    split_draw,
    swap_labels,
)

__all__ = [
    "apply_gamma",
    "clean_stranded_pixels",
    "draw_cell_path",
    "erase_cell",
    "expand_label_to_foreground",
    "fill_label_holes",
    "fix_label_semiholes",
    "merge_cells",
    "split_across",
    "split_draw",
    "swap_labels",
]


--------------------------------------------------------------------------------
FILE: src/cellflow/correction/labels.py
--------------------------------------------------------------------------------

"""Label correction operations on a single (H, W) segmentation frame.

All functions accept a 2-D ``seg`` array and modify it **in-place**.
They return ``True`` on success and ``False`` when the operation is
rejected (e.g. labels don't touch, result too small, background click).
"""
from __future__ import annotations

import logging
import os

import numpy as np
from scipy.ndimage import binary_dilation, binary_closing, binary_fill_holes, label as nd_label
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon as draw_polygon
from skimage.morphology import disk
from skimage.segmentation import watershed, expand_labels

log = logging.getLogger("cellflow.correction")
if os.environ.get("CELLFLOW_DEBUG"):
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[cellflow.correction] %(levelname)s %(message)s"))
        log.addHandler(_h)

MIN_CELL_SIZE: int = 4


# ── bounding-box helpers ──────────────────────────────────────────────────────

def _bbox_of_label(seg: np.ndarray, lab: int) -> tuple[int, int, int, int]:
    rows, cols = np.where(seg == lab)
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _bbox_of_two(seg: np.ndarray, la: int, lb: int) -> tuple[int, int, int, int]:
    rows, cols = np.where(np.isin(seg, [la, lb]))
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _extend_bbox(
    bbox: tuple[int, int, int, int],
    factor: float,
    shape: tuple[int, int],
    min_pad: int = 0,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    dr = max(int((r1 - r0) * (factor - 1) / 2), min_pad)
    dc = max(int((c1 - c0) * (factor - 1) / 2), min_pad)
    return (
        max(0, r0 - dr), max(0, c0 - dc),
        min(shape[0], r1 + dr), min(shape[1], c1 + dc),
    )


def _crop(arr: np.ndarray, bbox: tuple) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0:r1, c0:c1]


def _to_local(pts: list, bbox: tuple) -> list[tuple[float, float]]:
    r0, c0 = bbox[0], bbox[1]
    return [(float(p[-2]) - r0, float(p[-1]) - c0) for p in pts]


# ── line drawing ──────────────────────────────────────────────────────────────

def _interpolate(pts: list[tuple[float, float]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in range(len(pts) - 1):
        r0, c0 = pts[i]
        r1, c1 = pts[i + 1]
        n = max(abs(int(r1) - int(r0)), abs(int(c1) - int(c0)), 1)
        for t in np.linspace(0, 1, n + 1):
            out.append((int(round(r0 + t * (r1 - r0))), int(round(c0 + t * (c1 - c0)))))
    seen: set = set()
    result = []
    for p in out:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _draw_line(shape: tuple[int, int], pts: list[tuple[int, int]]) -> np.ndarray:
    line = np.zeros(shape, dtype=np.uint8)
    for r, c in pts:
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            line[r, c] = 1
    return line


# ── misc helpers ──────────────────────────────────────────────────────────────

def _free_label(seg: np.ndarray) -> int:
    return int(seg.max()) + 1


def _touches(seg: np.ndarray, la: int, lb: int) -> bool:
    dilated_a = binary_dilation(seg == la, disk(1))
    dilated_b = binary_dilation(seg == lb, disk(1))
    return bool(np.any(dilated_a & dilated_b))


def _label_at(seg: np.ndarray, pos: tuple) -> int:
    r, c = int(round(float(pos[-2]))), int(round(float(pos[-1])))
    r = max(0, min(r, seg.shape[0] - 1))
    c = max(0, min(c, seg.shape[1] - 1))
    return int(seg[r, c])


def frame_view_2d(arr: np.ndarray, t: int) -> np.ndarray | None:
    """Return a 2D frame view from a time-indexed label stack."""
    if arr.ndim < 3 or t < 0 or t >= arr.shape[0]:
        return None
    view = arr[t]
    while view.ndim > 2:
        if view.shape[0] != 1:
            return None
        view = view[0]
    return view


def best_overlapping_label(
    target_labels: np.ndarray,
    source_labels: np.ndarray,
    t: int,
    source_label: int,
) -> int:
    """Return the non-zero target label with most overlap against source_label."""
    if source_label == 0:
        return 0
    target_frame = frame_view_2d(target_labels, t)
    source_frame = frame_view_2d(source_labels, t)
    if target_frame is None or source_frame is None or target_frame.shape != source_frame.shape:
        return 0
    source_mask = source_frame == int(source_label)
    if not np.any(source_mask):
        return 0
    overlap_values, counts = np.unique(target_frame[source_mask], return_counts=True)
    best_label = 0
    best_count = 0
    for label, count in zip(overlap_values, counts, strict=True):
        label = int(label)
        if label != 0 and int(count) > best_count:
            best_label = label
            best_count = int(count)
    return best_label


# ── public operations ─────────────────────────────────────────────────────────

def expand_label_to_foreground(
    seg: np.ndarray,
    foreground: np.ndarray,
    label: int,
    *,
    max_distance: int,
) -> int:
    """Expand ``label`` into connected foreground background pixels in-place.

    Returns the number of newly labelled pixels. A ``max_distance`` of 0 means
    no distance cap.
    """
    if foreground.shape != seg.shape:
        raise ValueError("foreground and seg must have the same shape")

    label = int(label)
    if label == 0:
        return 0
    seed = seg == label
    if not np.any(seed):
        return 0

    allowed = (foreground > 0) & ((seg == 0) | seed)
    component_labels, _num_components = nd_label(
        allowed,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    touching_ids = np.unique(component_labels[seed])
    touching_ids = touching_ids[touching_ids != 0]
    if touching_ids.size == 0:
        return 0

    touching_component = np.isin(component_labels, touching_ids)
    if max_distance > 0:
        dist = distance_transform_edt(~seed)
        touching_component &= dist <= int(max_distance)

    added = touching_component & (seg == 0)
    n_added = int(np.count_nonzero(added))
    if n_added:
        seg[added] = label
    return n_added

def erase_cell(seg: np.ndarray, pos: tuple | None = None, *, label: int | None = None) -> bool:
    """Set all pixels of the label under *pos* (or *label*) to 0."""
    if label is None:
        if pos is None:
            return False
        label = _label_at(seg, pos)
    log.debug("erase_cell: label=%s pos=%s", label, pos)
    if label == 0:
        return False
    seg[seg == label] = 0
    return True


def merge_cells(
    seg: np.ndarray,
    pos_start: tuple,
    pos_end: tuple,
    *,
    label_a: int | None = None,
    label_b: int | None = None,
) -> bool:
    """Merge the cell at *pos_start* into the cell at *pos_end*."""
    la = label_a if label_a is not None else _label_at(seg, pos_start)
    lb = label_b if label_b is not None else _label_at(seg, pos_end)
    log.debug("merge_cells: la=%s lb=%s", la, lb)
    if la == 0 or lb == 0 or la == lb:
        return False
    if not _touches(seg, la, lb):
        return False

    bbox = _bbox_of_two(seg, la, lb)
    bbox = _extend_bbox(bbox, 1.25, seg.shape)
    r0, c0, r1, c1 = bbox
    crop = _crop(seg, bbox)

    combined = np.isin(crop, [la, lb])
    closed = binary_closing(combined, disk(2))
    other_cells = (crop != 0) & ~combined
    closed = closed & ~other_cells
    seg[r0:r1, c0:c1][closed] = lb

    remaining_la = seg == la
    if remaining_la.any():
        seg[remaining_la] = lb

    clean_stranded_pixels(seg)
    return True


def split_across(
    seg: np.ndarray,
    img: np.ndarray | None,
    pos_start: tuple,
    pos_end: tuple,
    *,
    new_label: int | None = None,
) -> bool:
    """Watershed-split the cell under *pos_start* using two seeds."""
    la = _label_at(seg, pos_start)
    lb = _label_at(seg, pos_end)
    log.debug("split_across: la=%s lb=%s", la, lb)
    if la == 0 or la != lb:
        return False

    bbox = _bbox_of_label(seg, la)
    bbox = _extend_bbox(bbox, 1.25, seg.shape)
    r0, c0, r1, c1 = bbox
    crop_seg = _crop(seg, bbox)
    mask = (crop_seg == la).astype(np.uint8)
    interior = mask.astype(bool)

    rs = max(0, min(int(round(float(pos_start[-2]))) - r0, mask.shape[0] - 1))
    cs = max(0, min(int(round(float(pos_start[-1]))) - c0, mask.shape[1] - 1))
    re = max(0, min(int(round(float(pos_end[-2]))) - r0, mask.shape[0] - 1))
    ce = max(0, min(int(round(float(pos_end[-1]))) - c0, mask.shape[1] - 1))

    new_lab = int(new_label) if new_label is not None else _free_label(seg)

    for radius in range(7):
        markers = np.zeros(mask.shape, dtype=np.int32)
        if radius == 0:
            markers[rs, cs] = la
            markers[re, ce] = new_lab
        else:
            d = disk(radius)
            seed_a = np.zeros(mask.shape, dtype=bool)
            seed_a[rs, cs] = True
            seed_b = np.zeros(mask.shape, dtype=bool)
            seed_b[re, ce] = True
            markers[binary_dilation(seed_a, d) & interior] = la
            markers[binary_dilation(seed_b, d) & interior] = new_lab

        if img is not None:
            crop_img = _crop(img, bbox)
            ws = watershed(crop_img, markers=markers, mask=mask)
        else:
            dist = distance_transform_edt(mask)
            ws = watershed(-dist, markers=markers, mask=mask)

        size_a = int(np.sum(ws == la))
        size_b = int(np.sum(ws == new_lab))
        if size_a >= MIN_CELL_SIZE and size_b >= MIN_CELL_SIZE:
            seg[r0:r1, c0:c1][ws == new_lab] = new_lab
            return True

    return False


def split_draw(
    seg: np.ndarray,
    positions: list,
    *,
    curlabel: int | None = None,
    new_label: int | None = None,
) -> bool:
    """Split a cell along a manually drawn line."""
    log.debug("split_draw: %d raw positions, curlabel=%s", len(positions), curlabel)
    if curlabel is None or curlabel == 0 or not np.any(seg == curlabel):
        return False

    bbox = _bbox_of_label(seg, curlabel)
    bbox = _extend_bbox(bbox, 1.25, seg.shape)
    crop = _crop(seg, bbox).copy()
    local_pts = _to_local(positions, bbox)

    in_cell_indices = [
        i for i, p in enumerate(local_pts)
        if 0 <= int(round(p[0])) < crop.shape[0]
        and 0 <= int(round(p[1])) < crop.shape[1]
        and crop[int(round(p[0])), int(round(p[1]))] == curlabel
    ]
    if len(in_cell_indices) < 2:
        return False

    first_idx, last_idx = in_cell_indices[0], in_cell_indices[-1]
    in_cell = [local_pts[i] for i in in_cell_indices]

    ext_start = local_pts[first_idx - 1] if first_idx > 0 else in_cell[0]
    ext_end   = local_pts[last_idx + 1]  if last_idx < len(local_pts) - 1 else in_cell[-1]

    all_pts = [ext_start] + in_cell + [ext_end]
    interp = _interpolate(all_pts)
    line = _draw_line(crop.shape, interp)

    if int(np.sum(line & (crop == curlabel))) == 0:
        return False

    return _split_in_crop(seg, crop, line, bbox, curlabel, new_label=new_label)


def _split_in_crop(
    seg: np.ndarray,
    crop: np.ndarray,
    line: np.ndarray,
    bbox: tuple,
    curlabel: int,
    retry: int = 0,
    *,
    new_label: int | None = None,
) -> bool:
    if retry > 6:
        return False

    dilated = binary_dilation(line, disk(retry)) if retry > 0 else line.astype(bool)
    mask = np.zeros(crop.shape, dtype=np.uint8)
    mask[crop == curlabel] = 1
    mask[dilated] = 0

    regions, n = nd_label(mask)
    sizes = [int(np.sum(regions == i)) for i in range(1, n + 1)]
    log.debug("_split_in_crop: retry=%d n_regions=%d sizes=%s", retry, n, sizes)

    if n >= 2:
        ids_by_size = sorted(range(1, n + 1), key=lambda i: sizes[i - 1], reverse=True)
        id_a, id_b = ids_by_size[0], ids_by_size[1]
        size_a, size_b = sizes[id_a - 1], sizes[id_b - 1]
        if size_a >= MIN_CELL_SIZE and size_b >= MIN_CELL_SIZE:
            regions_2 = np.zeros_like(regions)
            regions_2[regions == id_a] = 1
            regions_2[regions == id_b] = 2
            expanded = expand_labels(regions_2, distance=max(retry + 2, 3))
            r0, c0, r1, c1 = bbox
            new_lab = int(new_label) if new_label is not None else _free_label(seg)
            orig_cell = crop == curlabel
            seg[r0:r1, c0:c1][(expanded == 2) & orig_cell] = new_lab
            return True

    return _split_in_crop(seg, crop, line, bbox, curlabel, retry + 1, new_label=new_label)


def draw_cell_path(
    seg: np.ndarray,
    positions: list,
    *,
    curlabel: int | None = None,
    new_label: int | None = None,
) -> bool:
    """Draw a closed region from the user's stroke and fill its interior."""
    log.debug("draw_cell_path: %d raw positions, curlabel=%s", len(positions), curlabel)
    if len(positions) < 2:
        return False

    local_pts = [(float(p[-2]), float(p[-1])) for p in positions]

    rows = np.array([p[0] for p in local_pts])
    cols = np.array([p[1] for p in local_pts])
    rr, cc = draw_polygon(rows, cols, seg.shape)
    log.debug("draw_cell_path: polygon fill pixels=%d", len(rr))

    if len(rr) < MIN_CELL_SIZE:
        return False

    extending = bool(curlabel) and curlabel != 0 and np.any(seg == curlabel)
    label = curlabel if extending else (
        int(new_label) if new_label is not None else _free_label(seg)
    )

    fill_mask = np.zeros(seg.shape, dtype=bool)
    fill_mask[rr, cc] = True
    if extending:
        existing_mask = seg == label
        connected_regions, _ = nd_label(existing_mask | fill_mask)
        connected_ids = np.unique(connected_regions[existing_mask])
        fill_mask &= np.isin(connected_regions, connected_ids)
    else:
        fill_mask &= (seg == 0)

    n_px = int(np.sum(fill_mask))
    if n_px < MIN_CELL_SIZE:
        return False

    seg[fill_mask] = label
    if extending:
        cell_mask = seg == label
        filled_mask = binary_fill_holes(cell_mask)
        seg[filled_mask & ~cell_mask] = label
    return True


def swap_labels(seg: np.ndarray, pos_a: tuple, pos_b: tuple) -> bool:
    """Swap the label values at the two click positions across the whole frame."""
    la = _label_at(seg, pos_a)
    lb = _label_at(seg, pos_b)
    log.debug("swap_labels: la=%s lb=%s", la, lb)
    if la == 0 or lb == 0 or la == lb:
        return False
    mask_a = seg == la
    mask_b = seg == lb
    seg[mask_a] = lb
    seg[mask_b] = la
    return True


def relabel_cell(seg: np.ndarray, pos: tuple, new_label: int) -> bool:
    """Assign *new_label* to the cell at *pos* in *seg* (in-place).

    If *new_label* already exists in the frame, the two cells are swapped so
    no label is lost.  Returns ``False`` when *pos* hits background, already
    has *new_label*, or *new_label* is 0.
    """
    old_label = _label_at(seg, pos)
    if old_label == 0 or new_label == 0 or old_label == new_label:
        return False
    conflict = seg == new_label
    seg[seg == old_label] = new_label
    if np.any(conflict):
        seg[conflict] = old_label
    return True


def fill_label_holes(labels: np.ndarray, radius: int = 5) -> np.ndarray:
    """Fill enclosed background gaps by expanding neighboring labels.

    Background connected to the image border is preserved.  Enclosed zero-valued
    components are filled only as far as labels can expand within *radius*
    pixels; use a large radius to fill all enclosed gaps.
    """
    from skimage.measure import label as _cc_label

    if radius <= 0:
        return labels

    bg = labels == 0
    if not np.any(bg):
        return labels

    bg_labeled = _cc_label(bg, connectivity=2)
    open_ids: set[int] = set()
    for edge in (
        bg_labeled[0, :], bg_labeled[-1, :],
        bg_labeled[:, 0], bg_labeled[:, -1],
    ):
        open_ids.update(int(v) for v in np.unique(edge))
    open_ids.discard(0)

    open_bg = bg & np.isin(bg_labeled, list(open_ids))
    enclosed = bg & ~open_bg
    if not np.any(enclosed):
        return labels

    sentinel = int(np.max(labels)) + 1
    work = labels.copy()
    work[open_bg] = sentinel
    expanded = expand_labels(work, distance=int(radius))
    expanded[open_bg] = 0
    expanded[expanded == sentinel] = 0
    return expanded.astype(labels.dtype, copy=False)


def fix_label_semiholes(
    labels: np.ndarray,
    radius: int = 5,
    max_opening: int = 3,
) -> np.ndarray:
    """Fill narrow border-connected background gaps by expanding labels.

    Candidate zero-valued components must touch the image border with no more
    than ``max_opening`` pixels.  Wider border-connected background regions are
    preserved as open background.
    """
    from skimage.measure import label as _cc_label

    if radius <= 0 or max_opening <= 0:
        return labels

    bg = labels == 0
    if not np.any(bg):
        return labels

    bg_labeled = _cc_label(bg, connectivity=2)
    border_mask = np.zeros(labels.shape, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    candidate = np.zeros(labels.shape, dtype=bool)
    for comp_id in np.unique(bg_labeled[border_mask & bg]):
        comp_id = int(comp_id)
        if comp_id == 0:
            continue
        comp_mask = bg_labeled == comp_id
        opening = int(np.sum(comp_mask & border_mask))
        if opening <= int(max_opening):
            candidate |= comp_mask

    if not np.any(candidate):
        return labels

    sentinel = int(np.max(labels)) + 1
    work = labels.copy()
    work[bg & ~candidate] = sentinel
    expanded = expand_labels(work, distance=int(radius))
    expanded[bg & ~candidate] = 0
    expanded[expanded == sentinel] = 0
    return expanded.astype(labels.dtype, copy=False)


def clean_stranded_pixels(seg: np.ndarray, min_size: int = MIN_CELL_SIZE) -> int:
    """Remove disconnected same-label fragments, keeping each label's largest component."""
    from skimage.measure import label as _cc_label
    cleared = 0

    for cell_id in np.unique(seg):
        if cell_id == 0:
            continue
        mask = seg == cell_id
        labeled, n_comp = _cc_label(mask, return_num=True, connectivity=2)
        if n_comp <= 1:
            continue
        comp_sizes = {cid: int(np.sum(labeled == cid)) for cid in range(1, n_comp + 1)}
        largest = max(comp_sizes, key=comp_sizes.__getitem__)
        for comp_id, n_px in comp_sizes.items():
            if comp_id == largest:
                continue
            comp_mask = labeled == comp_id
            seg[comp_mask] = 0
            filled = expand_labels(seg, distance=n_px + 2)
            seg[comp_mask] = filled[comp_mask]
            cleared += n_px

    return cleared


from cellflow.segmentation import apply_gamma  # noqa: F401 — re-exported from here


--------------------------------------------------------------------------------
FILE: src/cellflow/database/__init__.py
--------------------------------------------------------------------------------

"""I/O for CellFlow tracked label volumes (TIFF) and validation state."""
from cellflow.database.tracked import (
    read_tracked_frame,
    tracked_frame_exists,
    tracked_n_frames,
    write_tracked_frame,
)
from cellflow.database.validation import (
    invalidate_track,
    is_track_validated,
    is_validated,
    read_validated_cells_at_frame,
    read_validated_frames,
    read_validated_tracks,
    remap_validated_tracks,
    validate_track,
)

__all__ = [
    "read_tracked_frame",
    "tracked_frame_exists",
    "tracked_n_frames",
    "write_tracked_frame",
    "invalidate_track",
    "is_track_validated",
    "is_validated",
    "read_validated_cells_at_frame",
    "read_validated_frames",
    "read_validated_tracks",
    "remap_validated_tracks",
    "validate_track",
]


--------------------------------------------------------------------------------
FILE: src/cellflow/database/hypotheses.py
--------------------------------------------------------------------------------

"""HDF5 hypothesis pool for nucleus segmentation candidates.

Schema: hypotheses/t{t:03d}/p{p:03d}/labels
Each labels dataset has shape (Z, Y, X) and dtype uint32.
Parameters are stored as group attributes on each p group.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np

from cellflow.segmentation import (
    CellposeFlowHypothesisParams,
    ContourWatershedParams,
    NucleusHypothesisParams,
    SeededWatershedParams,
    compute_seeded_watershed,
)

_LABEL_DTYPE = np.uint32
_ROOT_GROUP = "hypotheses"


def _check_schema(h5: "h5py.File", path: Path) -> None:
    """Raise if the file uses the old t/z/p layout from v1."""
    layout = h5.attrs.get("layout", "")
    if layout and "z{z" in str(layout):
        raise ValueError(
            f"{path} uses the v1 t/z/p schema and cannot be read by v2. "
            "Re-generate the hypothesis database."
        )


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    """A single (t, p) label volume payload."""

    t: int
    p: int
    labels: np.ndarray  # shape (Z, Y, X), dtype uint32
    params: NucleusHypothesisParams | CellposeFlowHypothesisParams | ContourWatershedParams | SeededWatershedParams


def _values(current: float, minimum: float, maximum: float, step: float) -> list[float]:
    if step > 0 and minimum != maximum:
        vals = np.arange(float(minimum), float(maximum) + step / 2.0, step, dtype=np.float64)
        return [float(v) for v in vals]
    return [float(current)]


def _int_values(current: int, minimum: int, maximum: int, step: int) -> list[int]:
    if step > 0 and minimum != maximum:
        return list(range(int(minimum), int(maximum) + 1, max(1, int(step))))
    return [int(current)]


@dataclass(frozen=True, slots=True)
class SeededWatershedSweepSpec:
    """Parameter sweep spec for nucleus-seeded watershed cell hypothesis generation."""

    basin: str = "prob"
    foreground_threshold: float = 0.5
    foreground_threshold_min: float = 0.5
    foreground_threshold_max: float = 0.5
    foreground_threshold_step: float = 0.05
    compactness: float = 0.0
    compactness_min: float = 0.0
    compactness_max: float = 0.0
    compactness_step: float = 0.1


def build_seeded_watershed_parameter_sets(spec: SeededWatershedSweepSpec) -> list[SeededWatershedParams]:
    """Return the deterministic list of SeededWatershedParams for this sweep spec."""
    fg_vals = _values(spec.foreground_threshold, spec.foreground_threshold_min, spec.foreground_threshold_max, spec.foreground_threshold_step)
    compactness_vals = _values(spec.compactness, spec.compactness_min, spec.compactness_max, spec.compactness_step)
    return [
        SeededWatershedParams(basin=spec.basin, foreground_threshold=float(fg), compactness=float(c))
        for fg in fg_vals
        for c in compactness_vals
    ]


def normalize_seeded_watershed_dp_stack(
    dp_stack: np.ndarray,
    prob_shape: tuple[int, int, int, int],
) -> np.ndarray:
    """Return flow vectors as (T, Z, C, Y, X), accepting common Cellpose layouts."""
    dp = np.asarray(dp_stack, dtype=np.float32)
    n_t, n_z, n_y, n_x = prob_shape

    if dp.ndim == 4:
        if dp.shape == (n_z, n_y, n_x, 2) or dp.shape == (n_z, n_y, n_x, 3):
            dp = np.moveaxis(dp, -1, 1)[np.newaxis]
        elif dp.shape[0] == n_z and dp.shape[1] in (2, 3) and dp.shape[2:] == (n_y, n_x):
            dp = dp[np.newaxis]
        elif dp.shape[0] in (2, 3) and dp.shape[1:] == (n_z, n_y, n_x):
            dp = np.moveaxis(dp, 0, 1)[np.newaxis]
        else:
            raise ValueError(
                f"Expected dp stack matching prob shape {prob_shape}, got {dp.shape}"
            )
    elif dp.ndim == 5:
        if dp.shape == (n_t, n_z, n_y, n_x, 2) or dp.shape == (n_t, n_z, n_y, n_x, 3):
            dp = np.moveaxis(dp, -1, 2)
        elif dp.shape[0] == n_t and dp.shape[1] == n_z and dp.shape[2] in (2, 3) and dp.shape[3:] == (n_y, n_x):
            pass
        elif dp.shape[0] == n_t and dp.shape[1] in (2, 3) and dp.shape[2:] == (n_z, n_y, n_x):
            dp = np.moveaxis(dp, 1, 2)
        else:
            raise ValueError(
                f"Expected dp stack matching prob shape {prob_shape}, got {dp.shape}"
            )
    else:
        raise ValueError(f"Expected dp stack with 4 or 5 dimensions, got shape {dp.shape}")

    return np.asarray(dp, dtype=np.float32)


def normalize_seeded_watershed_nucleus_stack(
    nucleus_stack: np.ndarray,
    prob_shape: tuple[int, int, int, int],
) -> np.ndarray:
    """Return nucleus seed labels as (T, Z, Y, X), accepting 2D tracked labels."""
    nucleus = np.asarray(nucleus_stack)
    n_t, n_z, n_y, n_x = prob_shape

    if nucleus.ndim == 2:
        if nucleus.shape != (n_y, n_x):
            raise ValueError(
                f"Expected nucleus labels matching prob shape {prob_shape}, got {nucleus.shape}"
            )
        nucleus = np.broadcast_to(nucleus, (n_t, n_z, n_y, n_x)).copy()
    elif nucleus.ndim == 3:
        if nucleus.shape == (n_t, n_y, n_x):
            nucleus = np.broadcast_to(nucleus[:, np.newaxis], (n_t, n_z, n_y, n_x)).copy()
        elif nucleus.shape == (n_z, n_y, n_x) and n_t == 1:
            nucleus = nucleus[np.newaxis]
        else:
            raise ValueError(
                f"Expected nucleus labels matching prob shape {prob_shape}, got {nucleus.shape}"
            )
    elif nucleus.ndim == 4:
        if nucleus.shape == (n_t, n_z, n_y, n_x):
            pass
        elif nucleus.shape == (1, n_t, n_y, n_x):
            nucleus = np.broadcast_to(nucleus[0, :, np.newaxis], (n_t, n_z, n_y, n_x)).copy()
        elif nucleus.shape == (1, n_z, n_y, n_x) and n_t == 1:
            pass
        else:
            raise ValueError(
                f"Expected nucleus labels matching prob shape {prob_shape}, got {nucleus.shape}"
            )
    else:
        raise ValueError(f"Expected nucleus labels with 2-4 dimensions, got shape {nucleus.shape}")

    return np.asarray(nucleus)


def _run_seeded_watershed_task(
    args: tuple[int, int, "SeededWatershedParams", np.ndarray, np.ndarray | None, np.ndarray],
) -> "HypothesisRecord":
    t, p_idx, params, prob_t, dp_t, nuc_t = args
    n_z = prob_t.shape[0]
    slices = []
    for z in range(n_z):
        dp_2d = dp_t[z] if dp_t is not None else None
        slices.append(compute_seeded_watershed(prob_t[z], dp_2d, nuc_t[z], params))
    return HypothesisRecord(t=t, p=p_idx, labels=np.stack(slices, axis=0), params=params)


def _ordered_bounded_map(fn, inputs: Iterable, max_workers: int) -> Iterator:
    """Map fn over inputs while keeping at most max_workers submitted tasks."""
    if max_workers <= 1:
        for item in inputs:
            yield fn(item)
        return

    from concurrent.futures import ThreadPoolExecutor

    iterator = iter(inputs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = deque()
        for _ in range(max_workers):
            try:
                pending.append(executor.submit(fn, next(iterator)))
            except StopIteration:
                break

        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                pending.append(executor.submit(fn, next(iterator)))
            except StopIteration:
                pass


def iter_seeded_watershed_records(
    prob_stack: np.ndarray,
    dp_stack: np.ndarray | None,
    nucleus_stack: np.ndarray,
    spec: SeededWatershedSweepSpec,
    n_workers: int = 1,
) -> Iterator[HypothesisRecord]:
    """Yield one HypothesisRecord per (t, p) for seeded-watershed cell segmentation.

    prob_stack:    (T, Z, Y, X) float32 probability logits
    dp_stack:      (T, Z, 2, Y, X) float32 flow vectors; required when basin='flow_mag'
    nucleus_stack: (T, Z, Y, X) int32 tracked nucleus labels used as watershed seeds
    """
    params_list = build_seeded_watershed_parameter_sets(spec)
    if not params_list:
        return

    prob_stack = np.asarray(prob_stack, dtype=np.float32)
    if prob_stack.ndim == 3:
        prob_stack = prob_stack[np.newaxis]
    if dp_stack is not None:
        dp_stack = normalize_seeded_watershed_dp_stack(dp_stack, prob_stack.shape)
    nucleus_stack = normalize_seeded_watershed_nucleus_stack(nucleus_stack, prob_stack.shape)

    n_t = prob_stack.shape[0]
    def tasks():
        for t in range(n_t):
            for p_idx, params in enumerate(params_list):
                yield (t, p_idx, params, prob_stack[t], dp_stack[t] if dp_stack is not None else None, nucleus_stack[t])

    if n_workers > 1:
        yield from _ordered_bounded_map(_run_seeded_watershed_task, tasks(), n_workers)
    else:
        for a in tasks():
            yield _run_seeded_watershed_task(a)


# ---------------------------------------------------------------------------
# Legacy read helpers — still used by extend.py, ingest.py, propagator.py
# ---------------------------------------------------------------------------

def read_hypothesis_labels(path: str | Path, t: int, p: int) -> np.ndarray:
    """Read the (Z, Y, X) label volume for one (t, p) entry."""
    with h5py.File(Path(path), "r") as h5:
        _check_schema(h5, Path(path))
        return np.asarray(h5[f"{_ROOT_GROUP}/t{t:03d}/p{p:03d}/labels"], dtype=_LABEL_DTYPE)


def list_hypotheses(path: str | Path) -> tuple[int, dict[int, dict]]:
    """Return (n_p, params_by_p_index) from the first timepoint in the file.

    n_p is the number of parameter sets. params_by_p_index maps p index to
    the attribute dict stored on that group.
    """
    with h5py.File(Path(path), "r") as h5:
        _check_schema(h5, Path(path))
        root = h5[_ROOT_GROUP]
        t_keys = sorted(k for k in root.keys() if k.startswith("t"))
        if not t_keys:
            return 0, {}
        first_t = root[t_keys[0]]
        p_keys = sorted(k for k in first_t.keys() if k.startswith("p"))
        n_p = len(p_keys)
        params_by_p: dict[int, dict] = {}
        for p_name in p_keys:
            p_idx = int(p_name[1:])
            params_by_p[p_idx] = dict(first_t[p_name].attrs)
        return n_p, params_by_p




--------------------------------------------------------------------------------
FILE: src/cellflow/database/tracked.py
--------------------------------------------------------------------------------

"""TIFF storage for tracked nucleus label volumes.

Schema: single multipage TIFF — shape (T, Y, X), dtype uint32.
Frames that have not yet been tracked are stored as all-zeros.
A frame is considered "tracked" (exists) if it contains at least one non-zero label.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

_LABEL_DTYPE = np.uint32


def _load_stack(path: Path) -> np.ndarray:
    """Load the TIFF as (T, Y, X). Returns empty array if file does not exist.

    Tolerates legacy files written with a singleton Z axis: ``(T, 1, Y, X)``
    is squeezed to ``(T, Y, X)`` so the in-memory shape always matches the
    documented schema.
    """
    if not path.exists():
        return np.empty((0, 0, 0), dtype=_LABEL_DTYPE)
    stack = np.asarray(tifffile.imread(str(path)), dtype=_LABEL_DTYPE)
    if stack.ndim == 2:
        stack = stack[np.newaxis]
    elif stack.ndim == 4 and stack.shape[1] == 1:
        stack = stack[:, 0]
    return stack


def write_tracked_frame(path: str | Path, t: int, labels: np.ndarray) -> None:
    """Write a single tracked frame into tracked_labels.tif."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=_LABEL_DTYPE)
    if labels.ndim == 3:
        labels = labels.max(axis=0)  # (Z, Y, X) → (Y, X)
    H, W = labels.shape
    stack = _load_stack(path)
    if stack.size == 0:
        stack = np.zeros((t + 1, H, W), dtype=_LABEL_DTYPE)
    elif t >= stack.shape[0]:
        extra = np.zeros((t + 1 - stack.shape[0], H, W), dtype=_LABEL_DTYPE)
        stack = np.concatenate([stack, extra], axis=0)
    stack[t] = labels
    tifffile.imwrite(str(path), stack, compression="zlib")


def read_tracked_frame(path: str | Path, t: int) -> np.ndarray:
    """Read a single tracked frame, returned as (Y, X) uint32 array."""
    stack = _load_stack(Path(path))
    if t >= stack.shape[0]:
        raise KeyError(f"Frame t={t} not found in {path}")
    return stack[t]


def read_full_tracked_stack(path: str | Path) -> np.ndarray:
    """Read all tracked frames as a (T, Y, X) uint32 array."""
    return _load_stack(Path(path))


def tracked_n_frames(path: str | Path) -> int:
    """Return the number of timepoints written to tracked_labels.tif."""
    stack = _load_stack(Path(path))
    return stack.shape[0]


def tracked_frame_exists(path: str | Path, t: int) -> bool:
    """Return True if timepoint t has been written (contains non-zero labels)."""
    path = Path(path)
    if not path.exists():
        return False
    stack = _load_stack(path)
    return t < stack.shape[0] and bool(stack[t].any())


--------------------------------------------------------------------------------
FILE: src/cellflow/database/validation.py
--------------------------------------------------------------------------------

"""Persistent validation metadata for the nucleus workflow.

Frame-level validation (validated_frames.json):
    A "fully-validated" frame is one where every current (non-zero) cell ID has
    been individually validated.  The file acts as a *cache* so UI counters can
    count fully-validated frames without scanning the whole stack.
    Used by the cell workflow (3_cell). Stays untouched.

    Schema: JSON array of ints, e.g. [0, 3, 7].

Track-level validation (validated_cells.json):
    Tracks which frames have been validated for each cell (track) ID.
    Used by the nucleus workflow (2_nucleus).

    Schema: JSON object keyed by cell ID string, value is a list of frame ints,
    e.g. {"47": [10, 11, 12], "82": [3, 4, 5]}.
    Cell IDs with no validated frames are omitted entirely (sparse).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _path(pos_dir: Path) -> Path:
    return pos_dir / "2_nucleus" / "validated_frames.json"


def read_validated_frames(pos_dir: Path) -> set[int]:
    """Return the set of validated frame indices, or an empty set if none."""
    p = _path(pos_dir)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text())
        return set(int(t) for t in data)
    except Exception:
        return set()


def write_validated_frames(pos_dir: Path, frames: set[int]) -> None:
    """Persist the full set of validated frames."""
    p = _path(pos_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sorted(frames)))


def validate_frame(pos_dir: Path, t: int) -> None:
    """Mark frame t as validated."""
    frames = read_validated_frames(pos_dir)
    frames.add(t)
    write_validated_frames(pos_dir, frames)


def invalidate_frame(pos_dir: Path, t: int) -> None:
    """Remove the validated mark from frame t."""
    frames = read_validated_frames(pos_dir)
    frames.discard(t)
    write_validated_frames(pos_dir, frames)


def is_validated(pos_dir: Path, t: int) -> bool:
    """Return True if frame t is in the validated set."""
    return t in read_validated_frames(pos_dir)


# ---------------------------------------------------------------------------
# Track-level validation (nucleus workflow)
# ---------------------------------------------------------------------------

def _cells_path(pos_dir: Path) -> Path:
    return pos_dir / "2_nucleus" / "validated_cells.json"


def read_validated_tracks(pos_dir: Path) -> dict[int, set[int]]:
    """Return {cell_id: {frames}} for all validated tracks.

    Empty dict if the file is missing or corrupt.
    JSON keys are cell ID strings; values are lists of frame ints.
    """
    p = _cells_path(pos_dir)
    if not p.exists():
        return {}
    try:
        raw: dict = json.loads(p.read_text())
        return {int(k): set(int(f) for f in vs) for k, vs in raw.items() if vs}
    except Exception:
        return {}


def _write_validated_tracks(pos_dir: Path, data: dict[int, set[int]]) -> None:
    """Persist the full {cell_id: {frames}} map. Entries with empty sets are dropped."""
    p = _cells_path(pos_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        str(cell_id): sorted(frames)
        for cell_id, frames in data.items()
        if frames
    }
    p.write_text(json.dumps(serialisable))


def read_validated_cells_at_frame(pos_dir: Path, t: int) -> set[int]:
    """Return all cell IDs that have frame *t* in their validated set.

    Derived from the track-keyed store; suitable for overlay rendering.
    """
    return {cell_id for cell_id, frames in read_validated_tracks(pos_dir).items() if t in frames}


def is_track_validated(pos_dir: Path, cell_id: int) -> bool:
    """Return True if *cell_id* has any entry in the validated-tracks store."""
    return cell_id in read_validated_tracks(pos_dir)


def validate_track(pos_dir: Path, cell_id: int, frames: Iterable[int]) -> None:
    """Add the given frames to *cell_id*'s validated set (idempotent, accumulates).

    Creates an entry for *cell_id* if none exists yet.
    """
    frames_set = set(frames)
    if not frames_set:
        return
    data = read_validated_tracks(pos_dir)
    existing = data.get(cell_id, set())
    data[cell_id] = existing | frames_set
    _write_validated_tracks(pos_dir, data)


def invalidate_track(pos_dir: Path, cell_id: int) -> None:
    """Remove the entire entry for *cell_id* from the validated-tracks store.

    No-op if *cell_id* is not present.
    """
    data = read_validated_tracks(pos_dir)
    if cell_id in data:
        del data[cell_id]
        _write_validated_tracks(pos_dir, data)


def remap_validated_tracks(pos_dir: Path, old_to_new: dict[int, int]) -> None:
    """Remap cell IDs in the validated-tracks store using *old_to_new* mapping.

    IDs not present in the mapping are dropped.
    """
    data = read_validated_tracks(pos_dir)
    remapped = {
        old_to_new[cell_id]: frames
        for cell_id, frames in data.items()
        if cell_id in old_to_new
    }
    _write_validated_tracks(pos_dir, remapped)


--------------------------------------------------------------------------------
FILE: src/cellflow/meta/__init__.py
--------------------------------------------------------------------------------

"""Metadata inspection for CellFlow studies."""


--------------------------------------------------------------------------------
FILE: src/cellflow/meta/catalog.py
--------------------------------------------------------------------------------

"""Study directory discovery.

Provides ``discover_study(root)`` which scans a root directory for
condition/experiment/position trees and returns a sorted list of records
describing each position and its analysis readiness.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

ARTIFACT_REL = Path("4_analysis") / "position_analysis.h5"
NUCLEUS_LABELS_REL = Path("2_nucleus") / "tracked_labels.tif"
CELL_LABELS_REL = Path("3_cell") / "tracked_labels.tif"

STATUS_READY = "ready"
STATUS_INCOMPLETE = "incomplete"

REQUIRED_CSV_COLUMNS = ("path", "date", "condition", "id", "labels")


def discover_study(root: Path) -> list[dict]:
    """Scan *root* for ``condition/experiment/position`` trees.

    Returns a list of dicts sorted by ``condition_id``, ``experiment_id``,
    ``position_id``.  Each dict contains the keys:

    * ``condition_id``
    * ``experiment_id``
    * ``position_id``
    * ``position_path`` (:class:`~pathlib.Path`)
    * ``artifact_path`` (:class:`~pathlib.Path`)
    * ``nucleus_tracked_labels_path`` (:class:`~pathlib.Path`)
    * ``cell_tracked_labels_path`` (:class:`~pathlib.Path`)
    * ``analysis_status`` --- ``"ready"`` when all three files exist,
      otherwise ``"incomplete"``.
    """
    records: list[dict] = []

    if not root.is_dir():
        return records

    for cond_path in sorted(root.iterdir()):
        if not cond_path.is_dir():
            continue

        for exp_path in sorted(cond_path.iterdir()):
            if not exp_path.is_dir():
                continue

            for pos_path in sorted(exp_path.iterdir()):
                if not pos_path.is_dir():
                    continue

                artifact_path = pos_path / ARTIFACT_REL
                nucleus_path = pos_path / NUCLEUS_LABELS_REL
                cell_path = pos_path / CELL_LABELS_REL

                all_exist = (
                    artifact_path.is_file()
                    and nucleus_path.is_file()
                    and cell_path.is_file()
                )

                records.append({
                    "condition_id": cond_path.name,
                    "experiment_id": exp_path.name,
                    "position_id": pos_path.name,
                    "position_path": pos_path,
                    "artifact_path": artifact_path,
                    "nucleus_tracked_labels_path": nucleus_path,
                    "cell_tracked_labels_path": cell_path,
                    "analysis_status": STATUS_READY if all_exist else STATUS_INCOMPLETE,
                })

    return records


def load_meta_catalog(csv_path: Path | str) -> list[dict]:
    """Load CSV catalog records and expose meta-browser compatibility keys."""
    catalog_path = Path(csv_path)
    with catalog_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [column for column in REQUIRED_CSV_COLUMNS if column not in fieldnames]
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"Meta catalog is missing required column(s): {missing_text}")

        records = [
            _normalize_catalog_record(row, base_dir=catalog_path.parent)
            for row in reader
        ]

    return sorted(records, key=_catalog_sort_key)


def save_meta_catalog(csv_path: Path | str, records: Iterable[dict]) -> None:
    """Write catalog records to CSV using paths relative to the CSV when possible."""
    catalog_path = Path(csv_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    with catalog_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_CSV_COLUMNS))
        writer.writeheader()
        for record in records:
            normalized = _normalize_catalog_record(record, base_dir=catalog_path.parent)
            row = {
                "path": _path_for_csv(normalized["artifact_path"], catalog_path.parent),
                "date": normalized["date"],
                "condition": normalized["condition"],
                "id": normalized["id"],
                "labels": normalized["labels"],
            }
            writer.writerow(row)


def discover_h5_files(folder: Path | str, recursive: bool = True) -> list[Path]:
    """Return sorted H5 files from *folder*."""
    root = Path(folder)
    if not root.is_dir():
        return []

    patterns = ("*.h5", "*.hdf5")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern) if recursive else root.glob(pattern))
    return sorted(path for path in paths if path.is_file())


def records_from_h5_paths(
    paths: Iterable[Path | str],
    defaults: dict | None = None,
) -> list[dict]:
    """Create conservative catalog records from explicit H5 paths."""
    defaults = defaults or {}
    resolved_paths = [_resolve_path(Path(path)) for path in paths]
    ids = _ids_for_h5_paths(resolved_paths)
    records = []
    for path, source_id in zip(resolved_paths, ids):
        record = {
            "path": path,
            "date": str(defaults.get("date", "unknown_date")),
            "condition": str(defaults.get("condition", "unknown_condition")),
            "id": str(defaults.get("id", source_id)),
            "labels": str(defaults.get("labels", "")),
        }
        records.append(_normalize_catalog_record(record))
    return sorted(records, key=_catalog_sort_key)


def merge_catalog_records(existing: Iterable[dict], incoming: Iterable[dict]) -> list[dict]:
    """Append incoming records while skipping duplicate resolved H5 paths."""
    merged: list[dict] = []
    seen: set[Path] = set()

    for record in list(existing) + list(incoming):
        normalized = _normalize_catalog_record(record)
        artifact_path = normalized["artifact_path"]
        resolved = _resolve_path(artifact_path)
        if resolved in seen:
            continue
        seen.add(resolved)
        merged.append(normalized)

    return merged


def _normalize_catalog_record(record: dict, base_dir: Path | None = None) -> dict:
    """Return a record with required CSV fields and widget compatibility keys."""
    normalized = dict(record)

    raw_path = normalized.get("path", normalized.get("artifact_path", ""))
    artifact_path = Path(raw_path)
    if base_dir is not None and not artifact_path.is_absolute():
        artifact_path = base_dir / artifact_path
    artifact_path = _resolve_path(artifact_path)

    date = str(normalized.get("date", normalized.get("experiment_id", "unknown_date")))
    condition = str(
        normalized.get("condition", normalized.get("condition_id", "unknown_condition"))
    )
    source_id = str(normalized.get("id", normalized.get("position_id", artifact_path.stem)))
    labels = str(normalized.get("labels", ""))

    normalized.update({
        "path": artifact_path,
        "date": date,
        "condition": condition,
        "id": source_id,
        "labels": labels,
        "condition_id": condition,
        "experiment_id": date,
        "position_id": source_id,
        "artifact_path": artifact_path,
        "analysis_status": STATUS_READY if artifact_path.is_file() else STATUS_INCOMPLETE,
    })
    return normalized


def _resolve_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _path_for_csv(path: Path, base_dir: Path) -> str:
    try:
        return os.path.relpath(path, start=base_dir)
    except ValueError:
        return str(path)


def _catalog_sort_key(record: dict) -> tuple[str, str, str]:
    return (
        str(record.get("condition", record.get("condition_id", ""))),
        str(record.get("date", record.get("experiment_id", ""))),
        str(record.get("id", record.get("position_id", ""))),
    )


def _ids_for_h5_paths(paths: list[Path]) -> list[str]:
    stems = [path.stem for path in paths]
    duplicate_stems = {stem for stem in stems if stems.count(stem) > 1}
    ids: list[str] = []
    used: set[str] = set()
    for path in paths:
        parent_name = (
            path.parent.parent.name
            if path.parent.name == "4_analysis"
            else path.parent.name
        )
        source_id = (
            parent_name
            if path.stem in duplicate_stems or parent_name.lower().startswith("pos")
            else path.stem
        )
        unique_id = source_id
        suffix = 2
        while unique_id in used:
            unique_id = f"{source_id}-{suffix}"
            suffix += 1
        used.add(unique_id)
        ids.append(unique_id)
    return ids


--------------------------------------------------------------------------------
FILE: src/cellflow/napari.yaml
--------------------------------------------------------------------------------

name: cellflow
display_name: CellFlow
contributions:
  commands:
    - id: cellflow.main_widget
      title: "CellFlow"
      python_name: cellflow.napari:CellFlowWidget
  widgets:
    - command: cellflow.main_widget
      display_name: "CellFlow"


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/__init__.py
--------------------------------------------------------------------------------

from cellflow.napari._napari_compat import patch_napari_layer_delegate

patch_napari_layer_delegate()

from cellflow.napari.main_widget import CellFlowMainWidget as CellFlowWidget

__all__ = ["CellFlowWidget"]


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/_napari_compat.py
--------------------------------------------------------------------------------

"""Compatibility patches for napari/Qt integration edge cases."""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap


def _index_size_hint(index):
    if index is None or (hasattr(index, "isValid") and not index.isValid()):
        return None
    size_hint = index.data(Qt.ItemDataRole.SizeHintRole)
    if size_hint is None or not hasattr(size_hint, "height"):
        return None
    return size_hint


def patch_napari_layer_delegate() -> None:
    """Make napari's layer delegate tolerate transient indexes without size hints."""
    try:
        from napari._qt.containers._layer_delegate import LayerDelegate
        from napari._qt.containers.qt_layer_model import LoadedRole, ThumbnailRole
    except Exception:
        return

    if getattr(LayerDelegate, "_cellflow_missing_size_hint_patch", False):
        return

    def _paint_loading(self, painter, option, index):
        loaded = index.data(LoadedRole)
        if loaded:
            return

        size_hint = _index_size_hint(index)
        if size_hint is None:
            return

        self._load_movie.start()
        load_rect = option.rect.translated(4, 8)
        h = size_hint.height() - 16
        load_rect.setWidth(h)
        load_rect.setHeight(h)
        painter.drawPixmap(load_rect, self._load_movie.currentPixmap())

    def _paint_thumbnail(self, painter, option, index):
        loaded = index.data(LoadedRole)
        if not loaded:
            return

        size_hint = _index_size_hint(index)
        if size_hint is None:
            return

        all_loaded = index.model().sourceModel().all_loaded()
        if all_loaded:
            self._load_movie.setPaused(True)

        thumb_rect = option.rect.translated(-2, 2)
        h = size_hint.height() - 4
        thumb_rect.setWidth(h)
        thumb_rect.setHeight(h)
        image = index.data(ThumbnailRole)
        painter.drawPixmap(thumb_rect, QPixmap.fromImage(image))

    LayerDelegate._paint_loading = _paint_loading
    LayerDelegate._paint_thumbnail = _paint_thumbnail
    LayerDelegate._cellflow_missing_size_hint_patch = True



--------------------------------------------------------------------------------
FILE: src/cellflow/napari/analysis_widget.py
--------------------------------------------------------------------------------

"""Analysis widget for final processing and export in CellFlow v2."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from napari.qt.threading import thread_worker
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QCheckBox, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from cellflow.analysis import build_position_analysis_artifact
from cellflow.napari.ui_style import action_button, status_label

try:  # pragma: no cover - local branch compatibility
    from cellflow.analysis.artifact_reader import read_position_artifact
except ImportError:  # pragma: no cover - tests monkeypatch this when absent
    def read_position_artifact(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("cellflow.analysis.artifact_reader is unavailable")


try:  # pragma: no cover - local branch compatibility
    from cellflow.napari.artifact_visualization import add_artifact_layers, _nucleus_centroids_by_track
except ImportError:  # pragma: no cover - tests monkeypatch this when absent
    def add_artifact_layers(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("cellflow.napari.artifact_visualization is unavailable")

    def _nucleus_centroids_by_track(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("cellflow.napari.artifact_visualization is unavailable")


class _ProgressEmitter(QObject):
    progress = Signal(int, int, str)


class AnalysisWidget(QWidget):
    """Final analysis and export."""

    _artifact_layer_prefix = "[Artifact] "

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._build_worker = None
        self._build_completion_pending = False
        self._build_error_pending = False
        self._progress_emitter = _ProgressEmitter(self)
        self._progress_emitter.progress.connect(self._on_build_progress)
        self._cached_artifact_path: Path | None = None
        self._cached_artifact: Any = None
        self._cached_cell_labels: np.ndarray | None = None
        self._cached_nucleus_labels: np.ndarray | None = None
        self._cached_track_centroids: dict | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self.input_status_lbl = QLabel("")
        self.input_status_lbl.setWordWrap(True)
        status_label(self.input_status_lbl)
        layout.addWidget(self.input_status_lbl)

        self.artifact_path_lbl = QLabel("")
        self.artifact_path_lbl.setWordWrap(True)
        status_label(self.artifact_path_lbl)
        layout.addWidget(self.artifact_path_lbl)

        self.artifact_status_lbl = QLabel("")
        self.artifact_status_lbl.setWordWrap(True)
        status_label(self.artifact_status_lbl)
        layout.addWidget(self.artifact_status_lbl)

        self.artifact_progress_bar = QProgressBar()
        self.artifact_progress_bar.setRange(0, 100)
        self.artifact_progress_bar.setValue(0)
        self.artifact_progress_bar.setVisible(False)
        self.artifact_progress_bar.setTextVisible(True)
        layout.addWidget(self.artifact_progress_bar)

        self.build_artifact_btn = QPushButton("Build Position Artifact")
        action_button(self.build_artifact_btn, expand=True)
        layout.addWidget(self.build_artifact_btn)

        self.cancel_build_btn = QPushButton("Cancel")
        action_button(self.cancel_build_btn)
        self.cancel_build_btn.setEnabled(False)
        layout.addWidget(self.cancel_build_btn)

        self.show_artifact_btn = QPushButton("Show Artifact")
        action_button(self.show_artifact_btn, expand=True)
        layout.addWidget(self.show_artifact_btn)

        self.color_cells_by_label_cb = QCheckBox("Color cells by label")
        layout.addWidget(self.color_cells_by_label_cb)

        self.color_edges_by_id_cb = QCheckBox("Color edges by ID")
        layout.addWidget(self.color_edges_by_id_cb)

        self.color_edges_by_label_cb = QCheckBox("Color edges by label")
        layout.addWidget(self.color_edges_by_label_cb)

        self.hide_border_edges_cb = QCheckBox("Hide border edges")
        layout.addWidget(self.hide_border_edges_cb)

        self.clear_artifact_btn = QPushButton("Clear Artifact Layers")
        action_button(self.clear_artifact_btn, expand=True)
        layout.addWidget(self.clear_artifact_btn)

        layout.addStretch()

        self.build_artifact_btn.clicked.connect(self._on_build_artifact)
        self.cancel_build_btn.clicked.connect(self._on_cancel_build)
        self.show_artifact_btn.clicked.connect(self._on_show_artifact)
        self.clear_artifact_btn.clicked.connect(self._on_clear_artifact_layers)
        self.refresh(None)

    @property
    def cell_labels_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "tracked_labels.tif" if self._pos_dir else None

    @property
    def nucleus_labels_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    @property
    def artifact_out_path(self) -> Path | None:
        return self._pos_dir / "4_analysis" / "position_analysis.h5" if self._pos_dir else None

    def refresh(self, pos_dir: Path | str | None) -> None:
        new_pos_dir = Path(pos_dir) if pos_dir is not None else None
        if new_pos_dir != self._pos_dir:
            self._cached_artifact_path = None
            self._cached_artifact = None
            self._cached_cell_labels = None
            self._cached_nucleus_labels = None
            self._cached_track_centroids = None
        self._pos_dir = new_pos_dir
        self._update_status()

    def _output_path_text(self) -> str:
        if self.artifact_out_path is None:
            return "Output: no project open."
        return f"Output: {self.artifact_out_path}"

    def _update_status(self) -> None:
        self.artifact_path_lbl.setText(self._output_path_text())
        self._update_input_status()
        self._update_action_states()
        if self._pos_dir is None:
            self._set_artifact_status("Status: no project open.")
        elif not self.artifact_status_lbl.text():
            self._set_artifact_status("Status: ready.")

    def _update_input_status(self) -> None:
        if self._pos_dir is None:
            self.input_status_lbl.setText("Inputs: no project open.")
            return

        cell_ok = self.cell_labels_path is not None and self.cell_labels_path.exists()
        nucleus_ok = self.nucleus_labels_path is not None and self.nucleus_labels_path.exists()
        artifact_ok = self.artifact_out_path is not None and self.artifact_out_path.exists()
        check = "✓"
        cross = "✗"
        self.input_status_lbl.setText(
            f"Inputs: {check if cell_ok else cross} cell labels  "
            f"{check if nucleus_ok else cross} nucleus labels  "
            f"{check if artifact_ok else cross} artifact"
        )

    def _update_action_states(self) -> None:
        inputs_ready = (
            self._pos_dir is not None
            and self.cell_labels_path is not None
            and self.cell_labels_path.exists()
            and self.nucleus_labels_path is not None
            and self.nucleus_labels_path.exists()
        )
        artifact_ready = (
            self.viewer is not None
            and self.artifact_out_path is not None
            and self.artifact_out_path.exists()
        )
        running = self._build_worker is not None
        self.build_artifact_btn.setEnabled(inputs_ready and not running)
        self.cancel_build_btn.setEnabled(running)
        self.show_artifact_btn.setEnabled(artifact_ready and not running)
        self.clear_artifact_btn.setEnabled(self.viewer is not None and not running)

    def _set_build_running(self, running: bool) -> None:
        self.artifact_progress_bar.setVisible(running)
        if running:
            self.artifact_progress_bar.setRange(0, 100)
            self.artifact_progress_bar.setValue(0)
        else:
            self.artifact_progress_bar.setValue(0)
            self.artifact_progress_bar.setRange(0, 100)
        self._update_action_states()

    def _set_artifact_status(self, message: str) -> None:
        self.artifact_status_lbl.setText(message)

    def _on_build_progress(self, done: int, total: int, message: str) -> None:
        if total > 0:
            self.artifact_progress_bar.setRange(0, total)
            self.artifact_progress_bar.setValue(done)
        self._set_artifact_status(f"Status: {message}")

    def _on_build_done(self, output_path: Path) -> None:
        self._build_completion_pending = True
        self._build_worker = None
        self._set_build_running(False)
        self._set_artifact_status(f"Status: Wrote {output_path}")
        self._update_status()

    def _on_build_error(self, exc: Exception) -> None:
        self._build_error_pending = True
        self._build_worker = None
        self._set_build_running(False)
        self._set_artifact_status(f"Status: error: {exc}")
        self._update_status()

    def _on_build_artifact(self) -> None:
        if self._pos_dir is None or self.artifact_out_path is None:
            self._set_artifact_status("Status: no project open.")
            self._update_action_states()
            return
        if self.cell_labels_path is None or not self.cell_labels_path.exists():
            self._set_artifact_status("Status: missing 3_cell/tracked_labels.tif.")
            self._update_status()
            return
        if self.nucleus_labels_path is None or not self.nucleus_labels_path.exists():
            self._set_artifact_status("Status: missing 2_nucleus/tracked_labels.tif.")
            self._update_status()
            return

        self._build_completion_pending = False
        self._build_error_pending = False
        self._set_artifact_status("Status: building position artifact...")
        self._set_build_running(True)

        @thread_worker(
            connect={
                "returned": self._on_build_done,
                "errored": self._on_build_error,
            }
        )
        def _worker():
            return build_position_analysis_artifact(
                self._pos_dir,
                self.artifact_out_path,
                cell_tracked_labels_path=self.cell_labels_path,
                nucleus_tracked_labels_path=self.nucleus_labels_path,
                progress_cb=self._progress_emitter.progress.emit,
            )

        worker = _worker()
        self._build_worker = worker
        if self._build_completion_pending or self._build_error_pending:
            self._build_worker = None
            self._build_completion_pending = False
            self._build_error_pending = False
            self._update_action_states()

    def _on_cancel_build(self) -> None:
        worker = self._build_worker
        if worker is not None:
            self._build_worker = None
            worker.quit()
        self._set_build_running(False)
        self._set_artifact_status("Status: build cancelled.")
        self._update_status()

    def _on_show_artifact(self) -> None:
        if self.viewer is None:
            self._set_artifact_status("Status: no viewer available.")
            self._update_action_states()
            return
        artifact_path = self.artifact_out_path
        if artifact_path is None or not artifact_path.exists():
            self._set_artifact_status("Status: artifact file not found.")
            self._update_action_states()
            return

        # Cache artifact to avoid re-reading HDF5 on every Show click
        if self._cached_artifact_path != artifact_path:
            self._cached_artifact = read_position_artifact(artifact_path)
            self._cached_artifact_path = artifact_path
            self._cached_cell_labels = None
            self._cached_nucleus_labels = None
            self._cached_track_centroids = None

        # Cache label TIFFs — these are large files whose repeated reading blocks
        # the Qt main thread and causes freezes + ghost layer artifacts
        if self._cached_cell_labels is None:
            if self.cell_labels_path is not None and self.cell_labels_path.exists():
                try:
                    self._cached_cell_labels = np.asarray(tifffile.imread(self.cell_labels_path))
                except Exception:
                    pass
        if self._cached_nucleus_labels is None:
            if self.nucleus_labels_path is not None and self.nucleus_labels_path.exists():
                try:
                    self._cached_nucleus_labels = np.asarray(tifffile.imread(self.nucleus_labels_path))
                except Exception:
                    pass

        # Cache nucleus track centroids — O(T*W*H*N) pixel iteration, very expensive
        if self._cached_track_centroids is None and self._cached_nucleus_labels is not None:
            try:
                self._cached_track_centroids = _nucleus_centroids_by_track(self._cached_nucleus_labels)
            except Exception:
                pass

        self._clear_artifact_layers(set_status=False)
        show_kwargs: dict[str, Any] = {
            "prefix": self._artifact_layer_prefix,
            "color_cells_by_label": self.color_cells_by_label_cb.isChecked(),
            "color_edges_by_id": self.color_edges_by_id_cb.isChecked(),
            "color_edges_by_label": self.color_edges_by_label_cb.isChecked(),
            "hide_border_edges": self.hide_border_edges_cb.isChecked(),
        }
        if self._cached_cell_labels is not None:
            show_kwargs["cell_labels"] = self._cached_cell_labels
        if self._cached_nucleus_labels is not None:
            show_kwargs["nucleus_labels"] = self._cached_nucleus_labels
        if self._cached_track_centroids is not None:
            show_kwargs["nucleus_track_centroids"] = self._cached_track_centroids
        add_artifact_layers(self.viewer, self._cached_artifact, **show_kwargs)
        self._set_artifact_status(f"Status: loaded {artifact_path.name}")
        self._update_action_states()

    def _artifact_layer_names(self) -> list[str]:
        if self.viewer is None:
            return []
        names: list[str] = []
        for layer in list(self.viewer.layers):
            layer_name = getattr(layer, "name", layer)
            if isinstance(layer_name, str) and layer_name.startswith(self._artifact_layer_prefix):
                names.append(layer_name)
        return names

    def _on_clear_artifact_layers(self) -> None:
        if self.viewer is None:
            self._set_artifact_status("Status: no viewer available.")
            self._update_action_states()
            return

        self._clear_artifact_layers(set_status=True)
        self._update_action_states()

    def _clear_artifact_layers(self, *, set_status: bool) -> int:
        if self.viewer is None:
            return 0

        removed = 0
        names = self._artifact_layer_names()
        layers = self.viewer.layers
        for name in names:
            layer = None
            try:
                layer = layers[name]
            except Exception:
                layer = name
            cleanup = getattr(layer, "_cellflow_frame_shape_cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass
            # Hide before removal so napari clears the canvas visual first,
            # preventing a ghost frame from persisting in the viewport
            try:
                layer.visible = False
            except Exception:
                pass
            try:
                layers.remove(layer)
            except Exception:
                try:
                    del layers[name]
                except Exception:
                    pass
            removed += 1

        if set_status:
            if removed:
                self._set_artifact_status(f"Status: cleared {removed} artifact layers.")
            else:
                self._set_artifact_status("Status: no artifact layers to clear.")
        return removed


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/artifact_visualization.py
--------------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

__all__ = [
    "build_cell_centroid_points",
    "build_edge_shapes",
    "build_nucleus_track_shapes",
    "build_t1_edge_shapes",
    "build_t1_points",
    "add_artifact_layers",
]

_BORDER_EDGE_COLOR = np.array([0.6, 0.6, 0.6, 1.0], dtype=float)
_CELL_EDGE_COLOR = np.array([0.12156863, 0.46666667, 0.70588235, 1.0], dtype=float)
_T1_EDGE_COLOR = np.array([0.0, 1.0, 0.9, 1.0], dtype=float)
_UNLABELED_COLOR = np.array([0.7, 0.7, 0.7, 1.0], dtype=float)
_MIN_TRACK_ALPHA = 0.12


def build_cell_centroid_points(artifact: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cells = _section(artifact, "cells")
    frame = _column(cells, "frame").astype(float, copy=False)
    y = _column(cells, "centroid_y").astype(float, copy=False)
    x = _column(cells, "centroid_x").astype(float, copy=False)
    points = _stack_points(frame, y, x)
    features = {
        "frame": _column(cells, "frame"),
        "cell_id": _column(cells, "cell_id"),
        "area": _column(cells, "area").astype(float, copy=False),
        "class_label": _column(cells, "class_label"),
    }
    return points, features


def build_edge_shapes(
    artifact: Any,
    *,
    hide_border_edges: bool = False,
    color_by_id: bool = False,
    color_by_label: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    edges = _section(artifact, "edges")
    coord_y = np.asarray(_value(artifact, "coord_y"), dtype=float)
    coord_x = np.asarray(_value(artifact, "coord_x"), dtype=float)

    frame = _column(edges, "frame")
    edge_id = _column(edges, "edge_id")
    cell_a = _column(edges, "cell_a")
    cell_b = _column(edges, "cell_b")
    kind = _column(edges, "kind")
    edge_label = _column(edges, "edge_label")
    length = _column(edges, "length").astype(float, copy=False)
    is_t1_frame = _column(edges, "is_t1_frame").astype(bool, copy=False)
    coord_offset = _column(edges, "coord_offset")
    coord_count = _column(edges, "coord_count")

    lines: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    keep: list[int] = []
    for idx in range(len(frame)):
        if hide_border_edges and str(kind[idx]) == "border":
            continue
        start = int(coord_offset[idx])
        count = int(coord_count[idx])
        if count < 2:
            continue
        stop = start + count
        ys = coord_y[start:stop]
        xs = coord_x[start:stop]
        lines.append(_stack_points(np.full(len(ys), frame[idx], dtype=float), ys, xs))
        keep.append(idx)

    mask = np.asarray(keep, dtype=np.intp)
    if color_by_label:
        colors = _categorical_colors(edge_label[mask])
    elif color_by_id:
        colors = _categorical_colors(edge_id[mask])
    else:
        colors = [_edge_color_for_kind(item) for item in kind[mask]]
    features = {
        "frame": frame[mask],
        "edge_id": edge_id[mask],
        "cell_a": cell_a[mask],
        "cell_b": cell_b[mask],
        "kind": kind[mask],
        "edge_label": edge_label[mask],
        "length": length[mask],
        "is_t1_frame": is_t1_frame[mask],
        "coord_offset": coord_offset[mask],
        "coord_count": coord_count[mask],
    }
    color_array = np.asarray(colors, dtype=float)
    if color_array.size == 0:
        color_array = np.empty((0, 4), dtype=float)
    return lines, color_array, features


def build_t1_points(artifact: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    t1_events = _section(artifact, "t1_events")
    frame = _column(t1_events, "frame").astype(float, copy=False)
    y = _column(t1_events, "location_y").astype(float, copy=False)
    x = _column(t1_events, "location_x").astype(float, copy=False)
    points = _stack_points(frame, y, x)
    features = {
        "t1_event_id": _column(t1_events, "t1_event_id"),
        "frame": _column(t1_events, "frame"),
        "edge_id": _column(t1_events, "edge_id"),
        "losing_cell_a": _column(t1_events, "losing_cell_a"),
        "losing_cell_b": _column(t1_events, "losing_cell_b"),
        "gaining_cell_a": _column(t1_events, "gaining_cell_a"),
        "gaining_cell_b": _column(t1_events, "gaining_cell_b"),
        "location_y": _column(t1_events, "location_y").astype(float, copy=False),
        "location_x": _column(t1_events, "location_x").astype(float, copy=False),
    }
    return points, features


def build_t1_edge_shapes(artifact: Any) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    edges = _section(artifact, "edges")
    t1_events = _section(artifact, "t1_events")
    coord_y = np.asarray(_value(artifact, "coord_y"), dtype=float)
    coord_x = np.asarray(_value(artifact, "coord_x"), dtype=float)

    edge_frame = _column(edges, "frame")
    edge_id = _column(edges, "edge_id")
    coord_offset = _column(edges, "coord_offset")
    coord_count = _column(edges, "coord_count")

    event_ids = _column(t1_events, "t1_event_id")
    event_frame = _column(t1_events, "frame")
    event_edge_id = _column(t1_events, "edge_id")
    losing_cell_a = _column(t1_events, "losing_cell_a")
    losing_cell_b = _column(t1_events, "losing_cell_b")
    gaining_cell_a = _column(t1_events, "gaining_cell_a")
    gaining_cell_b = _column(t1_events, "gaining_cell_b")
    location_y = _column(t1_events, "location_y").astype(float, copy=False)
    location_x = _column(t1_events, "location_x").astype(float, copy=False)

    lines: list[np.ndarray] = []
    feature_rows: list[dict[str, Any]] = []
    for event_idx in range(len(event_ids)):
        transition_frame = int(event_frame[event_idx])
        transition_edge_id = int(event_edge_id[event_idx])
        for side, frame in (("before", transition_frame), ("after", transition_frame + 1)):
            row_idx = _find_edge_row(edge_frame, edge_id, frame, transition_edge_id)
            if row_idx is None:
                continue
            start = int(coord_offset[row_idx])
            count = int(coord_count[row_idx])
            if count < 2:
                continue
            stop = start + count
            ys = coord_y[start:stop]
            xs = coord_x[start:stop]
            lines.append(_stack_points(np.full(len(ys), frame, dtype=float), ys, xs))
            feature_rows.append(
                {
                    "t1_event_id": event_ids[event_idx],
                    "frame": frame,
                    "transition_frame": transition_frame,
                    "transition_side": side,
                    "edge_id": transition_edge_id,
                    "losing_cell_a": losing_cell_a[event_idx],
                    "losing_cell_b": losing_cell_b[event_idx],
                    "gaining_cell_a": gaining_cell_a[event_idx],
                    "gaining_cell_b": gaining_cell_b[event_idx],
                    "location_y": location_y[event_idx],
                    "location_x": location_x[event_idx],
                }
            )

    colors = np.tile(_T1_EDGE_COLOR, (len(lines), 1))
    return lines, colors, _feature_columns(
        feature_rows,
        [
            "t1_event_id",
            "frame",
            "transition_frame",
            "transition_side",
            "edge_id",
            "losing_cell_a",
            "losing_cell_b",
            "gaining_cell_a",
            "gaining_cell_b",
            "location_y",
            "location_x",
        ],
    )


def build_nucleus_track_shapes(
    artifact: Any,
    nucleus_labels: np.ndarray,
    *,
    current_frame: int,
    color_cells_by_label: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Return past-only nucleus centroid track segments for one viewer frame."""
    centroids = _nucleus_centroids_by_track(nucleus_labels)
    color_map = _cell_color_map(artifact, color_by_label=color_cells_by_label)
    return _build_nucleus_track_shapes_from_centroids(
        centroids,
        color_map,
        current_frame=current_frame,
    )


def _build_nucleus_track_shapes_from_centroids(
    centroids: dict[int, list[tuple[int, float, float]]],
    color_map: dict[int | None, tuple[float, float, float, float] | str],
    *,
    current_frame: int,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    current_frame = max(0, int(current_frame))

    lines: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    feature_rows: list[dict[str, Any]] = []
    max_age = max(current_frame, 1)
    for cell_id in sorted(centroids):
        rows = centroids[cell_id]
        for previous, current in zip(rows[:-1], rows[1:], strict=False):
            start_frame, start_y, start_x = previous
            end_frame, end_y, end_x = current
            if int(end_frame) > current_frame or int(end_frame) != int(start_frame) + 1:
                continue
            age = current_frame - int(end_frame)
            alpha = max(_MIN_TRACK_ALPHA, 1.0 - (age / max_age))
            color = np.asarray(color_map.get(int(cell_id), _UNLABELED_COLOR), dtype=float).copy()
            color[3] = alpha
            lines.append(
                _stack_points(
                    np.asarray([current_frame, current_frame], dtype=float),
                    np.asarray([start_y, end_y], dtype=float),
                    np.asarray([start_x, end_x], dtype=float),
                )
            )
            colors.append(color)
            feature_rows.append(
                {
                    "cell_id": int(cell_id),
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "age": int(age),
                }
            )

    color_array = np.asarray(colors, dtype=float)
    if color_array.size == 0:
        color_array = np.empty((0, 4), dtype=float)
    return lines, color_array, _feature_columns(
        feature_rows,
        ["cell_id", "start_frame", "end_frame", "age"],
    )


def add_artifact_layers(
    viewer: Any,
    artifact: Any,
    prefix: str = "[Artifact] ",
    *,
    color_cells_by_label: bool = False,
    color_edges_by_id: bool = False,
    color_edges_by_label: bool = False,
    hide_border_edges: bool = False,
    cell_labels: np.ndarray | None = None,
    nucleus_labels: np.ndarray | None = None,
    nucleus_track_centroids: dict | None = None,
) -> list[Any]:
    edge_lines, edge_colors, edge_features = build_edge_shapes(
        artifact,
        hide_border_edges=hide_border_edges,
        color_by_id=color_edges_by_id,
        color_by_label=color_edges_by_label,
    )
    edge_cache = _frame_shape_cache(edge_lines, edge_colors, edge_features)
    current_edge_lines, current_edge_colors, current_edge_features = _cached_frame_shapes(
        edge_cache,
        _current_frame(viewer),
    )
    t1_lines, t1_colors, t1_features = build_t1_edge_shapes(artifact)
    t1_cache = _frame_shape_cache(t1_lines, t1_colors, t1_features)
    current_t1_lines, current_t1_colors, current_t1_features = _cached_frame_shapes(
        t1_cache,
        _current_frame(viewer),
    )
    if cell_labels is None:
        cell_labels = _read_label_image(_artifact_label_path(artifact, "cell_tracked_labels_path"))
    if nucleus_labels is None:
        nucleus_labels = _read_label_image(_artifact_label_path(artifact, "nucleus_tracked_labels_path"))
    cell_kwargs: dict[str, Any] = {}
    nucleus_kwargs: dict[str, Any] = {}
    color_dict = _cell_color_map(artifact, color_by_label=color_cells_by_label)
    try:
        from napari.utils.colormaps import DirectLabelColormap
    except Exception:  # pragma: no cover - napari compatibility
        pass
    else:
        label_colormap = DirectLabelColormap(color_dict=color_dict)
        cell_kwargs["colormap"] = label_colormap
        nucleus_kwargs["colormap"] = label_colormap

    track_centroids = nucleus_track_centroids if nucleus_track_centroids is not None else _nucleus_centroids_by_track(nucleus_labels)
    track_color_map = _cell_color_map(artifact, color_by_label=color_cells_by_label)
    track_cache = _track_shape_cache(track_centroids, track_color_map)
    track_lines, track_colors, track_features = _cached_frame_shapes(track_cache, _current_frame(viewer))
    layers = [
        viewer.add_labels(
            cell_labels,
            name=f"{prefix}Cell labels",
            opacity=0.55,
            blending="translucent",
            **cell_kwargs,
        ),
        viewer.add_labels(
            nucleus_labels,
            name=f"{prefix}Nucleus labels",
            opacity=0.65,
            blending="translucent",
            **nucleus_kwargs,
        ),
        viewer.add_shapes(
            track_lines,
            ndim=3,
            name=f"{prefix}Nucleus tracks",
            shape_type="path",
            features=track_features,
            edge_width=2,
            face_color="transparent",
            blending="translucent",
            **_edge_color_kwargs(track_colors),
        ),
        viewer.add_shapes(
            current_edge_lines,
            ndim=3,
            name=f"{prefix}Edges",
            shape_type="path",
            features=current_edge_features,
            edge_width=1,
            face_color="transparent",
            blending="translucent",
            **_edge_color_kwargs(current_edge_colors),
        ),
        viewer.add_shapes(
            current_t1_lines,
            ndim=3,
            name=f"{prefix}T1 edges",
            shape_type="path",
            features=current_t1_features,
            edge_width=1,
            face_color="transparent",
            blending="translucent",
            **_edge_color_kwargs(current_t1_colors),
        ),
    ]
    _connect_frame_shape_layer_to_dims(viewer, layers[2], frame_cache=track_cache)
    _connect_frame_shape_layer_to_dims(viewer, layers[3], frame_cache=edge_cache)
    _connect_frame_shape_layer_to_dims(viewer, layers[4], frame_cache=t1_cache)
    return layers


def _section(artifact: Any, name: str) -> Any:
    if isinstance(artifact, Mapping):
        return artifact[name]
    return getattr(artifact, name)


def _value(artifact: Any, name: str) -> Any:
    if isinstance(artifact, Mapping):
        return artifact[name]
    return getattr(artifact, name)


def _artifact_label_path(artifact: Any, name: str) -> Path:
    return Path(_value(artifact, name))


def _read_label_image(path: Path) -> np.ndarray:
    return np.asarray(tifffile.imread(path))


def _current_frame(viewer: Any) -> int:
    step = getattr(getattr(viewer, "dims", None), "current_step", ())
    if not step:
        return 0
    try:
        return int(step[0])
    except Exception:
        return 0


def _edge_color_kwargs(colors: np.ndarray) -> dict[str, np.ndarray]:
    if len(colors) == 0:
        return {}
    return {"edge_color": colors}


def _frame_shape_cache(
    lines: list[np.ndarray],
    colors: np.ndarray,
    features: dict[str, np.ndarray],
) -> dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]]:
    frames = np.asarray(features.get("frame", np.asarray([], dtype=int))).astype(int, copy=False)
    cache: dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]] = {}
    for frame in sorted(set(frames.tolist())):
        indexes = np.flatnonzero(frames == int(frame))
        cache[int(frame)] = (
            [lines[int(idx)] for idx in indexes],
            colors[indexes] if len(colors) else np.empty((0, 4), dtype=float),
            {name: values[indexes] for name, values in features.items()},
        )
    return cache


def _empty_frame_shapes(
    frame_cache: dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]],
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    feature_names: list[str] = []
    for _lines, _colors, features in frame_cache.values():
        feature_names = list(features)
        break
    return [], np.empty((0, 4), dtype=float), {name: np.asarray([], dtype=object) for name in feature_names}


def _cached_frame_shapes(
    frame_cache: dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]],
    frame: int,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    return frame_cache.get(int(frame), _empty_frame_shapes(frame_cache))


def _track_shape_cache(
    centroids: dict[int, list[tuple[int, float, float]]],
    color_map: dict[int | None, tuple[float, float, float, float] | str],
) -> dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]]:
    max_frame = max(
        (int(frame) for rows in centroids.values() for frame, _y, _x in rows),
        default=0,
    )
    return {
        frame: _build_nucleus_track_shapes_from_centroids(
            centroids,
            color_map,
            current_frame=frame,
        )
        for frame in range(max_frame + 1)
    }


def _connect_frame_shape_layer_to_dims(
    viewer: Any,
    layer: Any,
    *,
    frame_cache: dict[int, tuple[list[np.ndarray], np.ndarray, dict[str, np.ndarray]]],
) -> None:
    dims = getattr(viewer, "dims", None)
    events = getattr(dims, "events", None)
    current_step_event = getattr(events, "current_step", None)
    connect = getattr(current_step_event, "connect", None)
    if not callable(connect):
        return
    current_disconnect = getattr(current_step_event, "disconnect", None)

    def _update(_event=None) -> None:
        viewer_layers = getattr(viewer, "layers", None)
        if viewer_layers is not None:
            try:
                if layer not in viewer_layers:
                    return
            except Exception:
                pass
        lines, colors, features = _cached_frame_shapes(frame_cache, _current_frame(viewer))
        layer.data = lines
        layer.features = features
        layer.edge_color = colors if len(colors) else "transparent"
        refresh = getattr(layer, "refresh", None)
        if callable(refresh):
            refresh()

    def _disconnect() -> None:
        if callable(current_disconnect):
            try:
                current_disconnect(_update)
            except Exception:
                pass
        if callable(removed_disconnect):
            try:
                removed_disconnect(_on_removed)
            except Exception:
                pass
        try:
            frame_cache.clear()
        except Exception:
            pass

    def _on_removed(event=None) -> None:
        removed = getattr(event, "value", None)
        if removed is layer or getattr(removed, "name", None) == getattr(layer, "name", None):
            _disconnect()

    layers = getattr(viewer, "layers", None)
    layer_events = getattr(layers, "events", None)
    removed_event = getattr(layer_events, "removed", None)
    removed_connect = getattr(removed_event, "connect", None)
    removed_disconnect = getattr(removed_event, "disconnect", None)

    connect(_update)
    if callable(removed_connect):
        removed_connect(_on_removed)
    try:
        layer._cellflow_frame_shape_update = _update
        layer._cellflow_frame_shape_cleanup = _disconnect
    except Exception:
        pass


def _nucleus_centroids_by_track(nucleus_labels: np.ndarray) -> dict[int, list[tuple[int, float, float]]]:
    labels = np.asarray(nucleus_labels)
    if labels.ndim == 2:
        labels = labels[np.newaxis, ...]
    if labels.ndim > 3:
        labels = np.squeeze(labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected time-first 2D/3D nucleus labels, got shape {nucleus_labels.shape}")

    centroids: dict[int, list[tuple[int, float, float]]] = {}
    for frame_idx, frame in enumerate(labels):
        for cell_id in sorted(np.unique(frame).astype(int)):
            if cell_id == 0:
                continue
            coords = np.argwhere(frame == cell_id)
            if coords.size == 0:
                continue
            y, x = coords.mean(axis=0)
            centroids.setdefault(int(cell_id), []).append((int(frame_idx), float(y), float(x)))
    return centroids


def _cell_color_map(
    artifact: Any,
    *,
    color_by_label: bool,
) -> dict[int | None, tuple[float, float, float, float] | str]:
    if color_by_label:
        return _cell_label_color_map(artifact)

    cells = _section(artifact, "cells")
    cell_ids = np.asarray(sorted(set(_column(cells, "cell_id").astype(int))))
    cell_colors = _categorical_colors(cell_ids)
    color_map: dict[int | None, tuple[float, float, float, float] | str] = {
        None: "transparent",
        0: "transparent",
    }
    for cell_id, color in zip(cell_ids, cell_colors, strict=True):
        color_map[int(cell_id)] = tuple(float(channel) for channel in color)
    return color_map


def _cell_label_color_map(artifact: Any) -> dict[int | None, tuple[float, float, float, float] | str]:
    cells = _section(artifact, "cells")
    cell_ids = _column(cells, "cell_id")
    class_labels = _column(cells, "class_label")
    class_colors = _categorical_colors(class_labels)
    color_map: dict[int | None, tuple[float, float, float, float] | str] = {
        None: "transparent",
        0: "transparent",
    }
    for cell_id, color in zip(cell_ids, class_colors):
        cell_id_int = int(cell_id)
        if cell_id_int not in color_map:
            color_map[cell_id_int] = tuple(float(channel) for channel in color)
    return color_map


def _column(table: Any, name: str) -> np.ndarray:
    if isinstance(table, Mapping):
        value = table[name]
    else:
        value = getattr(table, name)
    return np.asarray(value)


def _find_edge_row(frame: np.ndarray, edge_id: np.ndarray, target_frame: int, target_edge_id: int) -> int | None:
    matches = np.flatnonzero(
        (frame.astype(int, copy=False) == target_frame)
        & (edge_id.astype(int, copy=False) == target_edge_id)
    )
    if len(matches) == 0:
        return None
    return int(matches[0])


def _feature_columns(rows: list[dict[str, Any]], names: list[str]) -> dict[str, np.ndarray]:
    if not rows:
        return {name: np.asarray([], dtype=object) for name in names}
    return {name: np.asarray([row[name] for row in rows]) for name in names}


def _stack_points(frame: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(frame) == 0:
        return np.empty((0, 3), dtype=float)
    return np.column_stack((frame.astype(float, copy=False), y.astype(float, copy=False), x.astype(float, copy=False)))


def _edge_color_for_kind(kind: Any) -> np.ndarray:
    if str(kind) == "border":
        return _BORDER_EDGE_COLOR
    return _CELL_EDGE_COLOR


def _categorical_colors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if len(values) == 0:
        return np.empty((0, 4), dtype=float)

    keys = [str(value) for value in values]
    palette = {
        key: _palette_color(idx)
        for idx, key in enumerate(sorted({key for key in keys if key != ""}))
    }
    colors = np.empty((len(values), 4), dtype=float)
    for idx, key in enumerate(keys):
        colors[idx] = _UNLABELED_COLOR if key == "" else palette[key]
    return colors


def _palette_color(index: int) -> np.ndarray:
    hue = (index * 0.618033988749895) % 1.0
    return np.asarray((*_hsv_to_rgb(hue, 0.65, 0.9), 1.0), dtype=float)


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[float, float, float]:
    sector = int(hue * 6.0)
    fraction = hue * 6.0 - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    sector %= 6
    if sector == 0:
        return value, t, p
    if sector == 1:
        return q, value, p
    if sector == 2:
        return p, value, t
    if sector == 3:
        return p, q, value
    if sector == 4:
        return t, p, value
    return value, p, q


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/cell_boundary_workflow_widget.py
--------------------------------------------------------------------------------

"""Track-conditioned cell boundary selection widget for CellFlow.

Three-stage workflow:
  1. **Initialize** — compute geodesic unary costs and pairwise weights,
     build initial labels, display in the viewer.
  2. **Refine** — run N ICM sweeps on the current viewer labels (repeatable,
     interleaves with manual correction).
  3. **Commit** — write the current viewer labels to disk.
"""

from __future__ import annotations

import logging
import queue
import threading
from enum import Enum
from pathlib import Path
import os

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.napari.ui_style import (
    add_block_button_row,
    add_block_pair_row,
    block_grid,
    compact_spinbox,
    status_label,
)
from cellflow.segmentation import (
    apply_gamma,
    build_consensus_boundary_2d,
    build_consensus_boundary_flow_following,
    FlowFollowingParams,
)

logger = logging.getLogger(__name__)

_CELL_SEG_LAYER = "Cell Segmentation"
_CELL_CONTOUR_LAYER = "Contour Map: Cell"
_CELL_FOREGROUND_SCORE_LAYER = "Foreground Score: Cell"
_CELL_FOREGROUND_LAYER = "Foreground Mask: Cell"
_CONTOUR_SWEEP_WIDTH = 60


class ContourMethod(str, Enum):
    """Selectable contour-map creation strategy."""

    CELLPOSE = "Cellpose Native"
    FLOW_FOLLOWING = "Flow-Following (EDT Gravity)"


class CellBoundaryWorkflowWidget(QWidget):
    """Track-conditioned cell boundary selection workflow."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None

        # Worker references
        self._contour_worker = None
        self._initialize_worker = None
        self._refine_worker = None

        # Cached ICM state (set by Initialize, consumed by Refine)
        self._icm_state = None  # CellICMState | None

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        # ---- 1. Contour Maps ----
        self._setup_contour_section(layout)

        # ---- 2. Boundary Selection (Initialize → Refine → Commit) ----
        self._setup_boundary_selection_section(layout)

        # ---- 3. Correction ----
        self._setup_correction_section(layout)

        layout.addStretch()

    # -- Contour Maps section (unchanged from original) -------------------

    def _setup_contour_section(self, layout: QVBoxLayout) -> None:
        def _stage_files(group_label, entries):
            return PipelineFilesWidget(
                [(group_label, entries)], viewer=self.viewer
            )

        def _stage_status():
            lbl = QLabel("")
            lbl.setWordWrap(True)
            lbl.setVisible(False)
            status_label(lbl)
            return lbl

        def _stage_progress():
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setVisible(False)
            return bar

        def _spin_width(widget, width=_CONTOUR_SWEEP_WIDTH):
            widget.setMinimumWidth(width)
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
            )
            return widget

        def _param_group_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight: 600;")
            return lbl

        contour_inner = QWidget()
        contour_lay = QVBoxLayout(contour_inner)
        contour_lay.setContentsMargins(0, 0, 0, 0)
        contour_lay.setSpacing(4)
        contour_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.contour_input_files = _stage_files("Inputs", [
            ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
            ("3_cell/filtered_dp.tif", "Filtered flow vectors"),
        ])
        contour_lay.addWidget(self.contour_input_files)

        # ── Method selector ──────────────────────────────────────────────
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._contour_method_combo = QComboBox()
        for m in ContourMethod:
            self._contour_method_combo.addItem(m.value)
        self._contour_method_combo.setCurrentText(ContourMethod.CELLPOSE.value)
        self._contour_method_combo.currentTextChanged.connect(
            self._on_contour_method_changed,
        )
        method_row.addWidget(self._contour_method_combo)
        contour_lay.addLayout(method_row)

        # ── Cellpose-specific parameter container ────────────────────────
        # Wrap the EXISTING flow_threshold and niter spinboxes so they can
        # be shown/hidden as a group.
        self._cellpose_params_container = QWidget()
        cp_lay = QHBoxLayout(self._cellpose_params_container)
        cp_lay.setContentsMargins(0, 0, 0, 0)

        # Cellpose mask sweep params
        self.cp_min_spin = _spin_width(QDoubleSpinBox())
        self.cp_min_spin.setRange(-20.0, 20.0)
        self.cp_min_spin.setValue(-3.0)
        self.cp_min_spin.setDecimals(1)
        self.cp_min_spin.setSingleStep(1.0)
        self.cp_max_spin = _spin_width(QDoubleSpinBox())
        self.cp_max_spin.setRange(-20.0, 20.0)
        self.cp_max_spin.setValue(0.0)
        self.cp_max_spin.setDecimals(1)
        self.cp_max_spin.setSingleStep(1.0)
        self.cp_step_spin = _spin_width(QDoubleSpinBox())
        self.cp_step_spin.setRange(0.1, 10.0)
        self.cp_step_spin.setValue(1.0)
        self.cp_step_spin.setDecimals(1)
        self.cp_step_spin.setSingleStep(0.5)
        self.contour_flow_threshold_spin = _spin_width(QDoubleSpinBox())
        self.contour_flow_threshold_spin.setRange(0.0, 10.0)
        self.contour_flow_threshold_spin.setValue(0.0)
        self.contour_flow_threshold_spin.setDecimals(2)
        self.contour_flow_threshold_spin.setSingleStep(0.1)
        self.contour_niter_spin = _spin_width(QSpinBox())
        self.contour_niter_spin.setRange(0, 2000)
        self.contour_niter_spin.setValue(200)

        # Build cellpose parameters container
        sweep_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(sweep_grid, 0,
            "Cellprob min:", compact_spinbox(self.cp_min_spin),
            "Cellprob max:", compact_spinbox(self.cp_max_spin))
        add_block_pair_row(sweep_grid, 1,
            "Cellprob step:", compact_spinbox(self.cp_step_spin),
            "Flow threshold:", compact_spinbox(self.contour_flow_threshold_spin))
        add_block_pair_row(sweep_grid, 2,
            "Niter:", compact_spinbox(self.contour_niter_spin))

        cellpose_container = QWidget()
        cellpose_lay = QVBoxLayout(cellpose_container)
        cellpose_lay.setContentsMargins(0, 0, 0, 0)
        cellpose_lay.setSpacing(4)
        cellpose_lay.addWidget(_param_group_label("Cellpose mask sweep"))
        cellpose_lay.addLayout(sweep_grid)

        self._cellpose_params_container = cellpose_container
        contour_lay.addWidget(self._cellpose_params_container)

        # ── Flow-following parameter container ───────────────────────────
        self._ff_params_container = QWidget()
        ff_lay = QHBoxLayout(self._ff_params_container)
        ff_lay.setContentsMargins(0, 0, 0, 0)

        ff_lay.addWidget(QLabel("Flow weight:"))
        self._ff_flow_weight_spin = QDoubleSpinBox()
        self._ff_flow_weight_spin.setRange(0.0, 1.0)
        self._ff_flow_weight_spin.setSingleStep(0.05)
        self._ff_flow_weight_spin.setValue(0.5)
        self._ff_flow_weight_spin.setToolTip(
            "Blend between flow direction (1.0) and EDT gravity toward "
            "nearest nucleus (0.0)."
        )
        ff_lay.addWidget(self._ff_flow_weight_spin)

        ff_lay.addWidget(QLabel("Step scale:"))
        self._ff_step_scale_spin = QDoubleSpinBox()
        self._ff_step_scale_spin.setRange(0.01, 2.0)
        self._ff_step_scale_spin.setSingleStep(0.05)
        self._ff_step_scale_spin.setValue(0.2)
        self._ff_step_scale_spin.setToolTip("Integration step-size multiplier.")
        ff_lay.addWidget(self._ff_step_scale_spin)

        ff_lay.addWidget(QLabel("Max iter:"))
        self._ff_max_iter_spin = QSpinBox()
        self._ff_max_iter_spin.setRange(10, 2000)
        self._ff_max_iter_spin.setSingleStep(10)
        self._ff_max_iter_spin.setValue(100)
        self._ff_max_iter_spin.setToolTip(
            "Maximum integration steps per pixel before giving up."
        )
        ff_lay.addWidget(self._ff_max_iter_spin)

        ff_lay.addWidget(QLabel("Capture r:"))
        self._ff_capture_radius_spin = QDoubleSpinBox()
        self._ff_capture_radius_spin.setRange(0.0, 30.0)
        self._ff_capture_radius_spin.setSingleStep(0.5)
        self._ff_capture_radius_spin.setValue(0.0)
        self._ff_capture_radius_spin.setToolTip(
            "0 = progressive shell assignment (recommended).\n"
            "> 0 = legacy fixed-radius capture at the given distance (px)."
        )
        ff_lay.addWidget(self._ff_capture_radius_spin)

        # Starts hidden — shown only when Flow-Following is selected
        self._ff_params_container.setVisible(False)
        contour_lay.addWidget(self._ff_params_container)

        # Gamma averaging params
        self.cp_gamma_min_spin = _spin_width(QDoubleSpinBox())
        self.cp_gamma_min_spin.setRange(0.05, 5.0)
        self.cp_gamma_min_spin.setValue(1.0)
        self.cp_gamma_min_spin.setDecimals(2)
        self.cp_gamma_min_spin.setSingleStep(0.05)
        self.cp_gamma_max_spin = _spin_width(QDoubleSpinBox())
        self.cp_gamma_max_spin.setRange(0.05, 5.0)
        self.cp_gamma_max_spin.setValue(1.0)
        self.cp_gamma_max_spin.setDecimals(2)
        self.cp_gamma_max_spin.setSingleStep(0.05)
        self.cp_gamma_step_spin = _spin_width(QDoubleSpinBox())
        self.cp_gamma_step_spin.setRange(0.05, 2.0)
        self.cp_gamma_step_spin.setValue(0.25)
        self.cp_gamma_step_spin.setDecimals(2)
        self.cp_gamma_step_spin.setSingleStep(0.05)

        gamma_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(gamma_grid, 0,
            "Gamma min:", compact_spinbox(self.cp_gamma_min_spin),
            "Gamma max:", compact_spinbox(self.cp_gamma_max_spin))
        add_block_pair_row(gamma_grid, 1,
            "Gamma step:", compact_spinbox(self.cp_gamma_step_spin))
        contour_lay.addWidget(_param_group_label("Gamma averaging"))
        contour_lay.addLayout(gamma_grid)

        # Foreground output
        self.contour_fg_threshold_spin = _spin_width(QDoubleSpinBox())
        self.contour_fg_threshold_spin.setRange(0.0, 1.0)
        self.contour_fg_threshold_spin.setValue(0.5)
        self.contour_fg_threshold_spin.setDecimals(2)
        self.contour_fg_threshold_spin.setSingleStep(0.01)

        fg_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(fg_grid, 0,
            "FG threshold:", compact_spinbox(self.contour_fg_threshold_spin))
        contour_lay.addWidget(_param_group_label("Foreground output"))
        contour_lay.addLayout(fg_grid)

        # Output files
        self.contour_output_files = _stage_files("Outputs", [
            ("3_cell/contour_maps.tif", "Contour maps"),
            ("3_cell/foreground_scores.tif", "Foreground scores"),
            ("3_cell/foreground_masks.tif", "Foreground masks"),
        ])
        contour_lay.addWidget(self.contour_output_files)

        # Buttons
        contour_btn_row = block_grid(horizontal_spacing=12)
        self.preview_contour_btn = QPushButton("Preview")
        self.preview_contour_btn.setToolTip(
            "Build contour maps for the current frame only and display in napari"
        )
        self.preview_contour_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.build_contour_maps_btn = QPushButton("Build Contour Maps")
        self.build_contour_maps_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        add_block_pair_row(contour_btn_row, 0,
            "", self.preview_contour_btn,
            "", self.build_contour_maps_btn)
        contour_lay.addLayout(contour_btn_row)

        self.contour_status_lbl = _stage_status()
        contour_lay.addWidget(self.contour_status_lbl)
        self.contour_progress_bar = _stage_progress()
        contour_lay.addWidget(self.contour_progress_bar)

        self.contour_section = CollapsibleSection(
            "1. Contour Maps", contour_inner, expanded=False
        )
        layout.addWidget(self.contour_section)

    # -- Boundary Selection section (Initialize → Refine → Commit) --------

    def _setup_boundary_selection_section(self, layout: QVBoxLayout) -> None:
        def _stage_files(group_label, entries):
            return PipelineFilesWidget(
                [(group_label, entries)], viewer=self.viewer
            )

        def _stage_status():
            lbl = QLabel("")
            lbl.setWordWrap(True)
            lbl.setVisible(False)
            status_label(lbl)
            return lbl

        def _stage_progress():
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setVisible(False)
            return bar

        def _float_spin(lo, hi, val, tooltip, *, decimals=2, step=0.1):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setDecimals(decimals)
            s.setSingleStep(step)
            s.setToolTip(tooltip)
            s.setMinimumWidth(_CONTOUR_SWEEP_WIDTH)
            s.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            return s

        def _int_spin(lo, hi, val, tooltip):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setToolTip(tooltip)
            s.setMinimumWidth(_CONTOUR_SWEEP_WIDTH)
            s.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            return s

        def _param_group_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight: 600;")
            return lbl

        sel_inner = QWidget()
        sel_lay = QVBoxLayout(sel_inner)
        sel_lay.setContentsMargins(0, 0, 0, 0)
        sel_lay.setSpacing(4)
        sel_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Input files
        self.boundary_selection_input_files = _stage_files("Inputs", [
            ("2_nucleus/tracked_labels.tif", "Nucleus tracked labels"),
            ("3_cell/contour_maps.tif", "Contour maps"),
            ("3_cell/foreground_scores.tif", "Foreground scores"),
            ("3_cell/foreground_masks.tif", "Foreground masks"),
        ])
        sel_lay.addWidget(self.boundary_selection_input_files)

        # ── Initialize parameters ─────────────────────────────────────
        sel_lay.addWidget(_param_group_label("Initialize"))

        self.alpha_unary_spin = _float_spin(
            0.0, 1000.0, 4.0,
            "Contour weight in the geodesic cost field: 1 + α·contour.",
        )
        self.lambda_s_spin = _float_spin(
            0.0, 1000.0, 1.0, "Spatial pairwise Potts weight.",
        )
        self.beta_s_spin = _float_spin(
            0.0, 1000.0, 5.0,
            "Contour sensitivity in spatial pairwise: exp(-β·avg_contour).",
        )
        self.lambda_t_spin = _float_spin(
            0.0, 1000.0, 1.0, "Temporal pairwise Potts weight.",
        )
        self.gamma_unary_spin = _float_spin(
            0.0, 100.0, 0.0,
            "Weight for (1 − foreground_score) in the geodesic cost field. "
            "0 = contour-only (default).",
        )
        self.init_mode_combo = QComboBox()
        self.init_mode_combo.addItems(["nuclei", "unary", "watershed"])
        self.init_mode_combo.setCurrentText("nuclei")
        self.init_mode_combo.setToolTip(
            "nuclei: only nucleus pixels labelled at init, cells grow via ICM.\n"
            "unary: init from per-pixel argmin of geodesic cost.\n"
            "watershed: seeded watershed on geodesic elevation."
        )
        self.init_mode_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        # In _setup_boundary_selection_section, after init_mode_combo:

        self.n_workers_spin = _int_spin(
            1, max(1, os.cpu_count() or 1), min(4, os.cpu_count() or 1),
            "Parallel workers for geodesic unary computation. "
            "Uses fork-based multiprocessing (Linux).",
        )

        init_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(init_grid, 3,
            "n_workers:", compact_spinbox(self.n_workers_spin),
            field_width=92)
        add_block_pair_row(init_grid, 0,
            "alpha_unary:", compact_spinbox(self.alpha_unary_spin),
            "lambda_s:", compact_spinbox(self.lambda_s_spin),
            field_width=92)
        add_block_pair_row(init_grid, 1,
            "beta_s:", compact_spinbox(self.beta_s_spin),
            "lambda_t:", compact_spinbox(self.lambda_t_spin),
            field_width=92)
        add_block_pair_row(init_grid, 2,
            "init_mode:", self.init_mode_combo,
            "gamma_unary:", compact_spinbox(self.gamma_unary_spin),
            field_width=92)
        sel_lay.addLayout(init_grid)

        init_btn_row = block_grid(horizontal_spacing=12)
        self.initialize_btn = QPushButton("Initialize")
        self.initialize_btn.setToolTip(
            "Compute geodesic unary costs and pairwise weights, then build "
            "initial labels. This is the expensive step."
        )
        self.initialize_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        add_block_button_row(init_btn_row, 0, self.initialize_btn)
        sel_lay.addLayout(init_btn_row)

        self.initialize_status_lbl = _stage_status()
        sel_lay.addWidget(self.initialize_status_lbl)
        self.initialize_progress_bar = _stage_progress()
        sel_lay.addWidget(self.initialize_progress_bar)

        # ── Refine parameters ─────────────────────────────────────────
        sel_lay.addWidget(_param_group_label("Refine"))

        self.n_iters_spin = _int_spin(
            1, 100, 3,
            "Number of ICM Gauss-Seidel sweeps per Refine press.",
        )
        self.min_round_flips_spin = _int_spin(
            0, 1_000_000, 0,
            "Stop early if a round produces fewer flips than this.",
        )
        self.lambda_area_spin = _float_spin(
            0.0, 10.0, 0.0,
            "Per-label frame-to-frame area-change penalty. 0 = disabled.",
            decimals=4, step=0.0001,
        )

        refine_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(refine_grid, 0,
            "n_iters:", compact_spinbox(self.n_iters_spin),
            "min_round_flips:", compact_spinbox(self.min_round_flips_spin),
            field_width=92)
        add_block_pair_row(refine_grid, 1,
            "lambda_area:", compact_spinbox(self.lambda_area_spin),
            field_width=92)
        sel_lay.addLayout(refine_grid)

        refine_btn_row = block_grid(horizontal_spacing=12)
        self.refine_btn = QPushButton("Refine")
        self.refine_btn.setToolTip(
            "Run ICM sweeps on the current viewer labels. Press repeatedly "
            "for incremental refinement; hand-correct between presses."
        )
        self.refine_btn.setEnabled(False)
        self.refine_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        add_block_button_row(refine_btn_row, 0, self.refine_btn)
        sel_lay.addLayout(refine_btn_row)

        self.refine_status_lbl = _stage_status()
        sel_lay.addWidget(self.refine_status_lbl)

        # ── Commit ────────────────────────────────────────────────────
        sel_lay.addWidget(_param_group_label("Commit"))

        commit_btn_row = block_grid(horizontal_spacing=12)
        self.commit_btn = QPushButton("Commit to disk")
        self.commit_btn.setToolTip(
            "Write the current viewer labels to 3_cell/tracked_labels.tif."
        )
        self.commit_btn.setEnabled(False)
        self.commit_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        add_block_button_row(commit_btn_row, 0, self.commit_btn)
        sel_lay.addLayout(commit_btn_row)

        self.commit_status_lbl = _stage_status()
        sel_lay.addWidget(self.commit_status_lbl)

        # Output files
        self.boundary_selection_output_files = _stage_files("Outputs", [
            ("3_cell/tracked_labels.tif", "Cell labels"),
        ])
        sel_lay.addWidget(self.boundary_selection_output_files)

        self.boundary_selection_section = CollapsibleSection(
            "2. Track-Conditioned Boundary Selection",
            sel_inner,
            expanded=False,
        )
        layout.addWidget(self.boundary_selection_section)

    # -- Correction section -----------------------------------------------

    def _setup_correction_section(self, layout: QVBoxLayout) -> None:
        correction_inner = QWidget()
        correction_lay = QVBoxLayout(correction_inner)
        correction_lay.setContentsMargins(0, 0, 0, 0)
        correction_lay.setSpacing(4)
        correction_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
        )
        correction_lay.addWidget(self.correction_widget)

        self.correction_section = CollapsibleSection(
            "3. Correction", correction_inner, expanded=False
        )
        layout.addWidget(self.correction_section)

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        # Contour
        self.preview_contour_btn.clicked.connect(self._on_preview_contour_maps)
        self.build_contour_maps_btn.clicked.connect(self._on_build_contour_maps)
        # Boundary selection stages
        self.initialize_btn.clicked.connect(self._on_initialize)
        self.refine_btn.clicked.connect(self._on_refine)
        self.commit_btn.clicked.connect(self._on_commit)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _contour_maps_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "contour_maps.tif" if self._pos_dir else None

    def _foreground_scores_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "foreground_scores.tif" if self._pos_dir else None

    def _foreground_masks_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "foreground_masks.tif" if self._pos_dir else None

    def _nucleus_labels_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    def _cell_labels_output_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "tracked_labels.tif" if self._pos_dir else None

    def _prob_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_prob_3dt.tif" if self._pos_dir else None

    def _filtered_dp_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "filtered_dp.tif" if self._pos_dir else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        # Clear cached state when switching positions
        if self._icm_state is not None:
            self._icm_state = None
            self._update_stage_enabled()
        self._refresh_stage_files(pos_dir)
        if pos_dir is None:
            self.correction_widget.deactivate()

    def _refresh_stage_files(self, pos_dir: Path | None = None) -> None:
        if pos_dir is None:
            pos_dir = self._pos_dir
        for fw in (
            self.contour_input_files,
            self.contour_output_files,
            self.boundary_selection_input_files,
            self.boundary_selection_output_files,
        ):
            fw.refresh(pos_dir)

    def get_state(self) -> dict:
        return {
            # Contour params
            "cp_min": self.cp_min_spin.value(),
            "cp_max": self.cp_max_spin.value(),
            "cp_step": self.cp_step_spin.value(),
            "contour_flow_threshold": self.contour_flow_threshold_spin.value(),
            "contour_niter": self.contour_niter_spin.value(),
            "cp_gamma_min": self.cp_gamma_min_spin.value(),
            "cp_gamma_max": self.cp_gamma_max_spin.value(),
            "cp_gamma_step": self.cp_gamma_step_spin.value(),
            "contour_fg_threshold": self.contour_fg_threshold_spin.value(),
            # --- flow-following state ---
            "contour_method": self._contour_method_combo.currentText(),
            "ff_flow_weight": self._ff_flow_weight_spin.value(),
            "ff_step_scale": self._ff_step_scale_spin.value(),
            "ff_max_iter": self._ff_max_iter_spin.value(),
            "ff_capture_radius": self._ff_capture_radius_spin.value(),
            # Initialize params
            "alpha_unary": self.alpha_unary_spin.value(),
            "lambda_s": self.lambda_s_spin.value(),
            "beta_s": self.beta_s_spin.value(),
            "lambda_t": self.lambda_t_spin.value(),
            "gamma_unary": self.gamma_unary_spin.value(),
            "init_mode": self.init_mode_combo.currentText(),
            # Refine params
            "n_iters": self.n_iters_spin.value(),
            "min_round_flips": self.min_round_flips_spin.value(),
            "lambda_area": self.lambda_area_spin.value(),
            "n_workers": self.n_workers_spin.value(),

        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return

        _spin_map = {
            "cp_min": self.cp_min_spin,
            "cp_max": self.cp_max_spin,
            "cp_step": self.cp_step_spin,
            "contour_flow_threshold": self.contour_flow_threshold_spin,
            "contour_niter": self.contour_niter_spin,
            "cp_gamma_min": self.cp_gamma_min_spin,
            "cp_gamma_max": self.cp_gamma_max_spin,
            "cp_gamma_step": self.cp_gamma_step_spin,
            "contour_fg_threshold": self.contour_fg_threshold_spin,
            "alpha_unary": self.alpha_unary_spin,
            "lambda_s": self.lambda_s_spin,
            "beta_s": self.beta_s_spin,
            "lambda_t": self.lambda_t_spin,
            "gamma_unary": self.gamma_unary_spin,
            "n_iters": self.n_iters_spin,
            "min_round_flips": self.min_round_flips_spin,
            "lambda_area": self.lambda_area_spin,
            "n_workers": self.n_workers_spin.value(),

        }
        # Backward-compat: map old graphcut_* keys
        _legacy_map = {
            "graphcut_alpha_unary": "alpha_unary",
            "graphcut_lambda_s": "lambda_s",
            "graphcut_beta_s": "beta_s",
            "graphcut_lambda_t": "lambda_t",
            "graphcut_n_iters": "n_iters",
            "graphcut_min_round_flips": "min_round_flips",
            "graphcut_init_mode": "init_mode",
        }
        for old_key, new_key in _legacy_map.items():
            if old_key in state and new_key not in state:
                state[new_key] = state[old_key]

        for key, widget in _spin_map.items():
            if key in state:
                widget.setValue(state[key])
        if "init_mode" in state:
            self.init_mode_combo.setCurrentText(str(state["init_mode"]))

        # --- flow-following state ---
        if "contour_method" in state:
            self._contour_method_combo.setCurrentText(state["contour_method"])
        if "ff_flow_weight" in state:
            self._ff_flow_weight_spin.setValue(state["ff_flow_weight"])
        if "ff_step_scale" in state:
            self._ff_step_scale_spin.setValue(state["ff_step_scale"])
        if "ff_max_iter" in state:
            self._ff_max_iter_spin.setValue(state["ff_max_iter"])
        if "ff_capture_radius" in state:
            self._ff_capture_radius_spin.setValue(state["ff_capture_radius"])

    # ------------------------------------------------------------------
    # Stage enable/disable
    # ------------------------------------------------------------------
    def _update_stage_enabled(self) -> None:
        """Enable/disable Refine and Commit based on whether Initialize has run."""
        has_state = self._icm_state is not None
        has_layer = _CELL_SEG_LAYER in self.viewer.layers
        self.refine_btn.setEnabled(has_state and has_layer)
        self.commit_btn.setEnabled(has_layer)

    def _set_all_buttons_enabled(self, enabled: bool) -> None:
        """Disable all stage buttons during a long-running operation."""
        self.initialize_btn.setEnabled(enabled)
        self.refine_btn.setEnabled(enabled and self._icm_state is not None)
        self.commit_btn.setEnabled(enabled and _CELL_SEG_LAYER in self.viewer.layers)

    # ------------------------------------------------------------------
    # Status / layer helpers
    # ------------------------------------------------------------------
    def _set_contour_status(self, msg: str) -> None:
        self.contour_status_lbl.setText(msg)
        self.contour_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_initialize_status(self, msg: str) -> None:
        self.initialize_status_lbl.setText(msg)
        self.initialize_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_refine_status(self, msg: str) -> None:
        self.refine_status_lbl.setText(msg)
        self.refine_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_commit_status(self, msg: str) -> None:
        self.commit_status_lbl.setText(msg)
        self.commit_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _show_layer(self, layer_name, data, kwargs, adder):
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = data
        else:
            adder(data, name=layer_name, **kwargs)

    def _current_t(self) -> int:
        dims = getattr(self.viewer, "dims", None)
        step = getattr(dims, "current_step", (0,))
        return int(step[0]) if len(step) >= 1 else 0

    # ------------------------------------------------------------------
    # 1. Contour Maps (Preview / Build)
    # ------------------------------------------------------------------
    def _cellprob_thresholds(self) -> list[float]:
        step = self.cp_step_spin.value()
        return list(np.arange(
            self.cp_min_spin.value(),
            self.cp_max_spin.value() + step / 2,
            step,
        ))

    def _cp_gammas(self) -> list[float]:
        step = self.cp_gamma_step_spin.value()
        return list(np.arange(
            self.cp_gamma_min_spin.value(),
            self.cp_gamma_max_spin.value() + step / 2,
            step,
        ))

    def _build_consensus_boundary_averaged(
        self, prob_3d, dp_2d, thresholds, gammas,
        *, flow_threshold, niter,
    ):
        boundary_accum = foreground_accum = None
        for gamma in gammas:
            prob_2d = apply_gamma(prob_3d, gamma).mean(axis=0)
            b, fg = build_consensus_boundary_2d(
                prob_2d, dp_2d, thresholds,
                flow_threshold=flow_threshold, reduction="mean", niter=niter,
            )
            if boundary_accum is None:
                boundary_accum = b.copy()
                foreground_accum = fg.copy()
            else:
                boundary_accum += b
                foreground_accum += fg
        n = len(gammas)
        return boundary_accum / n, foreground_accum / n

    def _set_contour_buttons_running(self, running: bool) -> None:
        self.build_contour_maps_btn.setEnabled(not running)
        self.preview_contour_btn.setEnabled(not running)
        self.contour_progress_bar.setVisible(running)
        if not running:
            self.contour_progress_bar.setValue(0)

    # ------------------------------------------------------------------
    # Contour method toggle
    # ------------------------------------------------------------------

    def _on_contour_method_changed(self, text: str) -> None:
        """Show / hide parameter rows that belong to the selected method."""
        is_ff = text == ContourMethod.FLOW_FOLLOWING.value
        self._cellpose_params_container.setVisible(not is_ff)
        self._ff_params_container.setVisible(is_ff)

    # ------------------------------------------------------------------
    # Flow-following helpers
    # ------------------------------------------------------------------

    def _load_prob_frame(self, t: int) -> np.ndarray:
        """Load a single probability frame (Z, Y, X)."""
        prob_path = self._prob_path()
        if prob_path is None or not prob_path.exists():
            raise FileNotFoundError(f"Probability file not found: {prob_path}")
        prob_stack = tifffile.imread(str(prob_path))
        if prob_stack.ndim == 3:
            prob_stack = prob_stack[np.newaxis]
        return prob_stack[t].astype(np.float32)

    def _load_dp_frame(self, t: int) -> np.ndarray:
        """Load a single DP flow frame (2, Y, X)."""
        dp_path = self._filtered_dp_path()
        if dp_path is None or not dp_path.exists():
            raise FileNotFoundError(f"DP file not found: {dp_path}")
        dp_stack = tifffile.imread(str(dp_path))
        if dp_stack.ndim == 3:
            dp_stack = dp_stack[np.newaxis]
        return dp_stack[t].astype(np.float32)

    def _show_error(self, title: str, message: str) -> None:
        """Display an error dialog (simple text message for now)."""
        self._set_contour_status(f"{title}: {message}")

    def _current_ff_params(self) -> FlowFollowingParams:
        """Read the flow-following spinboxes and return a frozen dataclass."""
        return FlowFollowingParams(
            median_kernel_time=1,
            median_kernel_space=1,
            gaussian_sigma_time=0.0,
            gaussian_sigma_space=0.0,
            flow_weight=self._ff_flow_weight_spin.value(),
            flow_step_scale=self._ff_step_scale_spin.value(),
            max_iterations=self._ff_max_iter_spin.value(),
            capture_radius=self._ff_capture_radius_spin.value(),
        )

    def _build_consensus_boundary_ff_averaged(
        self,
        prob_3d: np.ndarray,
        dp_2d: np.ndarray,
        labels_yx: np.ndarray,
        thresholds: list[float],
        gammas: list[float],
        *,
        ff_params: FlowFollowingParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gamma-averaged consensus boundary via flow-following.

        Mirrors ``_build_consensus_boundary_averaged`` but dispatches to
        :func:`build_consensus_boundary_flow_following` instead of
        Cellpose's ``compute_masks``.
        """
        boundary_accum = None
        foreground_accum = None
        n = 0
        for gamma in gammas:
            prob_2d = apply_gamma(prob_3d, gamma).mean(axis=0)
            b, fg = build_consensus_boundary_flow_following(
                prob_2d,
                dp_2d,
                labels_yx,
                thresholds,
                params=ff_params,
                reduction="mean",
            )
            if boundary_accum is None:
                boundary_accum = b.copy()
                foreground_accum = fg.copy()
            else:
                boundary_accum += b
                foreground_accum += fg
            n += 1
        if n > 0:
            boundary_accum /= n
            foreground_accum /= n
        return boundary_accum, foreground_accum

    def _on_build_contour_maps(self) -> None:
        """Launch the contour-map worker, dispatching by selected method."""
        # ── shared parameter gathering ───────────────────
        thresholds = self._cellprob_thresholds()
        gammas = self._cp_gammas()
        fg_threshold = self.contour_fg_threshold_spin.value()

        method = self._contour_method_combo.currentText()

        # ── method-specific gathering ────────────────────────────────
        if method == ContourMethod.FLOW_FOLLOWING.value:
            nuc_path = self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None
            if nuc_path is None or not nuc_path.exists():
                self._show_error(
                    "Nucleus tracked labels not found",
                    f"Flow-following requires nucleus tracked labels at:\n"
                    f"{nuc_path}\n\n"
                    f"Run the Nucleus Segmentation & Tracking step first.",
                )
                return
            nuc_labels = tifffile.imread(str(nuc_path))  # (T, Y, X)
            ff_params = self._current_ff_params()
            extra_kw = dict(
                method="flow_following",
                nuc_labels=nuc_labels,
                ff_params=ff_params,
            )
        else:
            extra_kw = dict(
                method="cellpose",
                flow_threshold=self.contour_flow_threshold_spin.value(),
                niter=int(self.contour_niter_spin.value()),
            )

        # ── launch worker (napari thread worker) ─────────────────────
        prob_path = self._prob_path()
        filtered_dp_path = self._filtered_dp_path()
        contour_path = self._contour_maps_path()
        score_path = self._foreground_scores_path()
        foreground_path = self._foreground_masks_path()
        for path, name in [
            (prob_path, "cell_prob_3dt.tif"),
            (filtered_dp_path, "filtered_dp.tif"),
        ]:
            if path is None or not path.exists():
                self._set_contour_status(f"Missing: {name}")
                return
        if contour_path is None or score_path is None or foreground_path is None:
            self._set_contour_status("No project open.")
            return

        pos_dir = self._pos_dir

        def _on_done(result):
            self._contour_worker = None
            self._set_contour_buttons_running(False)
            contours, scores, foreground = result
            self._show_layer(_CELL_CONTOUR_LAYER, contours,
                             {"colormap": "magma", "visible": True}, self.viewer.add_image)
            self._show_layer(_CELL_FOREGROUND_SCORE_LAYER, scores,
                             {"colormap": "viridis", "visible": True}, self.viewer.add_image)
            self._show_layer(_CELL_FOREGROUND_LAYER, foreground,
                             {}, self.viewer.add_labels)
            self._refresh_stage_files(pos_dir)
            self._set_contour_status("Contour maps complete.")

        def _on_progress(data):
            if isinstance(data, tuple):
                done, total, msg = data
                if total > 0:
                    self.contour_progress_bar.setVisible(True)
                    self.contour_progress_bar.setRange(0, total)
                    self.contour_progress_bar.setValue(done)
                self._set_contour_status(msg)
            else:
                self._set_contour_status(str(data))

        @thread_worker(connect={
            "yielded": _on_progress,
            "returned": _on_done,
            "errored": lambda exc: self._on_contour_error(exc),
        })
        def _worker():
            prob_path = self._prob_path()
            filtered_dp_path = self._filtered_dp_path()
            contour_path = self._contour_maps_path()
            score_path = self._foreground_scores_path()
            foreground_path = self._foreground_masks_path()

            prob_stack = tifffile.imread(str(prob_path))  # (T, Z, Y, X)
            dp_stack = tifffile.imread(str(filtered_dp_path))  # (T, 2, Y, X)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis]
            if dp_stack.ndim == 3:
                dp_stack = dp_stack[np.newaxis]
            T = prob_stack.shape[0]

            contour_maps = np.zeros((T, *prob_stack.shape[2:]), dtype=np.float32)
            fg_scores = np.zeros_like(contour_maps)
            fg_masks = np.zeros_like(contour_maps, dtype=bool)

            method = extra_kw.get("method", "cellpose")
            nuc_labels = extra_kw.get("nuc_labels", None)
            ff_params = extra_kw.get("ff_params", None)
            flow_threshold = extra_kw.get("flow_threshold", 0.0)
            niter = extra_kw.get("niter", 200)

            for t in range(T):
                yield (t + 1, T, f"Building contour maps: frame {t + 1}/{T}...")
                prob_3d = prob_stack[t]  # (Z, Y, X)
                dp_2d = dp_stack[t]     # (2, Y, X)

                if method == "flow_following":
                    labels_t = nuc_labels[t]  # (Y, X)
                    b, fg = self._build_consensus_boundary_ff_averaged(
                        prob_3d,
                        dp_2d,
                        labels_t,
                        thresholds,
                        gammas,
                        ff_params=ff_params,
                    )
                else:  # "cellpose"
                    b, fg = self._build_consensus_boundary_averaged(
                        prob_3d,
                        dp_2d,
                        thresholds,
                        gammas,
                        flow_threshold=flow_threshold,
                        niter=niter,
                    )

                contour_maps[t] = b
                fg_scores[t] = fg
                fg_masks[t] = fg > fg_threshold

            contour_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(contour_path), contour_maps, compression="zlib")
            tifffile.imwrite(str(score_path), fg_scores, compression="zlib")
            tifffile.imwrite(str(foreground_path), fg_masks.astype(np.uint8), compression="zlib")
            return contour_maps, fg_scores, fg_masks.astype(np.uint8)

        self._set_contour_status(
            f"Building contour maps ({len(thresholds)} thresholds, "
            f"{len(gammas)} gamma value(s))..."
        )
        self._set_contour_buttons_running(True)
        self._contour_worker = _worker()

    def _on_preview_contour_maps(self) -> None:
        """Preview the contour map for the current frame only."""
        t = self._current_t()
        thresholds = self._cellprob_thresholds()
        gammas = self._cp_gammas()
        method = self._contour_method_combo.currentText()

        prob_3d = self._load_prob_frame(t)  # (Z, Y, X)
        dp_2d = self._load_dp_frame(t)     # (2, Y, X)

        if method == ContourMethod.FLOW_FOLLOWING.value:
            nuc_path = self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None
            if nuc_path is None or not nuc_path.exists():
                self._show_error(
                    "Nucleus tracked labels not found",
                    f"Flow-following requires:\n{nuc_path}",
                )
                return
            nuc_labels_t = tifffile.imread(str(nuc_path))[t]
            ff_params = self._current_ff_params()
            b, fg = self._build_consensus_boundary_ff_averaged(
                prob_3d,
                dp_2d,
                nuc_labels_t,
                thresholds,
                gammas,
                ff_params=ff_params,
            )
        else:
            b, fg = self._build_consensus_boundary_averaged(
                prob_3d,
                dp_2d,
                thresholds,
                gammas,
                flow_threshold=self.contour_flow_threshold_spin.value(),
                niter=int(self.contour_niter_spin.value()),
            )

        # Display preview
        foreground_threshold = self.contour_fg_threshold_spin.value()

        # Load the full stack to get dimensions
        prob_path = self._prob_path()
        prob_stack = tifffile.imread(str(prob_path))
        if prob_stack.ndim == 3:
            prob_stack = prob_stack[np.newaxis]
        n_t = prob_stack.shape[0]

        contour_data = np.zeros((n_t,) + b.shape, dtype=np.float32)
        contour_data[t] = b
        score_data = np.zeros((n_t,) + fg.shape, dtype=np.float32)
        score_data[t] = fg
        mask_data = (score_data >= foreground_threshold).astype(np.uint8)
        self._show_layer(_CELL_CONTOUR_LAYER, contour_data,
                         {"colormap": "magma", "visible": True}, self.viewer.add_image)
        self._show_layer(_CELL_FOREGROUND_SCORE_LAYER, score_data,
                         {"colormap": "viridis", "visible": True}, self.viewer.add_image)
        self._show_layer(_CELL_FOREGROUND_LAYER, mask_data, {}, self.viewer.add_labels)
        self._set_contour_status(
            f"Preview t={t} — {len(thresholds)} thresholds, "
            f"{len(gammas)} gamma(s)"
        )

    def _on_contour_error(self, exc: Exception) -> None:
        self._contour_worker = None
        self._set_contour_buttons_running(False)
        self._set_contour_status(f"Error: {exc}")
        logger.exception("Cell contour worker error", exc_info=exc)

    # ------------------------------------------------------------------
    # 2a. Initialize
    # ------------------------------------------------------------------
    def _on_initialize(self) -> None:
        if self._pos_dir is None:
            self._set_initialize_status("No project open.")
            return

        required = [
            (self._nucleus_labels_path(), "tracked_labels.tif (nucleus)"),
            (self._contour_maps_path(), "contour_maps.tif"),
            (self._foreground_masks_path(), "foreground_masks.tif"),
        ]
        for path, name in required:
            if path is None or not path.exists():
                self._set_initialize_status(f"Missing: {name}")
                return

        # Collect paths and params before entering the thread
        nuc_path = self._nucleus_labels_path()
        fg_path = self._foreground_masks_path()
        ct_path = self._contour_maps_path()
        score_path = self._foreground_scores_path()
        pos_dir = self._pos_dir

        from cellflow.segmentation.cell_label_icm import (
            CellLabelICMParams,
            initialize_icm,
        )

        params = CellLabelICMParams(
            alpha_unary=self.alpha_unary_spin.value(),
            lambda_s=self.lambda_s_spin.value(),
            beta_s=self.beta_s_spin.value(),
            lambda_t=self.lambda_t_spin.value(),
            gamma_unary=self.gamma_unary_spin.value(),
            init_mode=self.init_mode_combo.currentText(),
            n_workers=self.n_workers_spin.value(),  # ← new
        )

        def _on_done(result):
            self._initialize_worker = None
            state, init_labels = result
            self._icm_state = state
            self._show_layer(
                _CELL_SEG_LAYER, init_labels, {"visible": True},
                self.viewer.add_labels,
            )
            self._set_all_buttons_enabled(True)
            self._update_stage_enabled()
            self.initialize_progress_bar.setVisible(False)
            self._set_initialize_status(
                f"Initialized: {state.n_labels} labels, "
                f"{'×'.join(str(d) for d in state.shape)}. "
                f"Ready for refinement."
            )

        def _on_error(exc):
            self._initialize_worker = None
            self._set_all_buttons_enabled(True)
            self._update_stage_enabled()
            self.initialize_progress_bar.setVisible(False)
            self._set_initialize_status(f"Error: {exc}")
            logger.exception("Initialize error", exc_info=exc)

        def _on_yielded(msg):
            self._set_initialize_status(str(msg))

        @thread_worker(connect={
            "yielded": _on_yielded,
            "returned": _on_done,
            "errored": _on_error,
        })
        def _worker():
            from cellflow.segmentation.cell_label_icm import _load_pos_dir_inputs

            msg_q: queue.SimpleQueue = queue.SimpleQueue()
            result_holder: list = []
            exc_holder: list = []

            def _run():
                try:
                    nuc, fg, ct, fg_scores = _load_pos_dir_inputs(pos_dir)
                    s, init = initialize_icm(
                        nuc, fg, ct, params,
                        foreground_scores=fg_scores,
                        progress_cb=lambda m: msg_q.put(m),
                    )
                    result_holder.append((s, init))
                except Exception as e:
                    exc_holder.append(e)

            yield "Loading inputs..."
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            while t.is_alive() or not msg_q.empty():
                try:
                    yield msg_q.get_nowait()
                except queue.Empty:
                    t.join(timeout=0.05)
            if exc_holder:
                raise exc_holder[0]
            return result_holder[0]

        self._set_initialize_status("Initializing...")
        self.initialize_progress_bar.setRange(0, 0)  # indeterminate
        self.initialize_progress_bar.setVisible(True)
        self._set_all_buttons_enabled(False)
        self._initialize_worker = _worker()

    # ------------------------------------------------------------------
    # 2b. Refine
    # ------------------------------------------------------------------
    def _on_refine(self) -> None:
        if self._icm_state is None:
            self._set_refine_status("Run Initialize first.")
            return
        if _CELL_SEG_LAYER not in self.viewer.layers:
            self._set_refine_status("No label layer — run Initialize first.")
            return

        current_labels = np.asarray(
            self.viewer.layers[_CELL_SEG_LAYER].data, dtype=np.uint32,
        )
        state = self._icm_state
        n_iters = self.n_iters_spin.value()
        min_flips = self.min_round_flips_spin.value()
        lambda_area = self.lambda_area_spin.value()

        from cellflow.segmentation.cell_label_icm import refine_icm

        def _on_done(result):
            self._refine_worker = None
            new_labels, energy_log = result
            self.viewer.layers[_CELL_SEG_LAYER].data = new_labels
            self._set_all_buttons_enabled(True)
            self._update_stage_enabled()

            # Summarise
            total_flips = sum(e["flips"] for e in energy_log)
            rounds = len(energy_log)
            detail = ", ".join(
                f"r{e['iteration']}={e['flips']}" for e in energy_log
            )
            self._set_refine_status(
                f"{rounds} round(s), {total_flips} total flips. [{detail}]"
            )

        def _on_error(exc):
            self._refine_worker = None
            self._set_all_buttons_enabled(True)
            self._update_stage_enabled()
            self._set_refine_status(f"Error: {exc}")
            logger.exception("Refine error", exc_info=exc)

        def _on_yielded(msg):
            self._set_refine_status(str(msg))

        @thread_worker(connect={
            "yielded": _on_yielded,
            "returned": _on_done,
            "errored": _on_error,
        })
        def _worker():
            msg_q: queue.SimpleQueue = queue.SimpleQueue()
            result_holder: list = []
            exc_holder: list = []

            def _run():
                try:
                    result_holder.append(
                        refine_icm(
                            state, current_labels,
                            n_iters=n_iters,
                            min_round_flips=min_flips,
                            lambda_area=lambda_area,
                            progress_cb=lambda m: msg_q.put(m),
                        )
                    )
                except Exception as e:
                    exc_holder.append(e)

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            while t.is_alive() or not msg_q.empty():
                try:
                    yield msg_q.get_nowait()
                except queue.Empty:
                    t.join(timeout=0.05)
            if exc_holder:
                raise exc_holder[0]
            return result_holder[0]

        self._set_refine_status(f"Refining ({n_iters} iterations)...")
        self._set_all_buttons_enabled(False)
        self._refine_worker = _worker()

    # ------------------------------------------------------------------
    # 2c. Commit
    # ------------------------------------------------------------------
    def _on_commit(self) -> None:
        if _CELL_SEG_LAYER not in self.viewer.layers:
            self._set_commit_status("No label layer to save.")
            return
        output_path = self._cell_labels_output_path()
        if output_path is None:
            self._set_commit_status("No project open.")
            return

        from cellflow.segmentation.cell_label_icm import commit_labels

        labels = np.asarray(self.viewer.layers[_CELL_SEG_LAYER].data)
        commit_labels(labels, output_path)
        self._refresh_stage_files(self._pos_dir)
        self._set_commit_status(f"Saved to {output_path.name}.")


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/cell_workflow_widget.py
--------------------------------------------------------------------------------

"""Cell segmentation widget for CellFlow — Flow-Following Segmentation."""
from __future__ import annotations

import logging
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QSpinBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.correction.labels import best_overlapping_label, expand_label_to_foreground
from cellflow.database.tracked import read_full_tracked_stack
from cellflow.napari.cell_boundary_workflow_widget import CellBoundaryWorkflowWidget
from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.napari.ui_style import (
    add_block_button_row,
    add_parameter_grid_row,
    block_grid,
    status_label,
)

logger = logging.getLogger(__name__)

_FILTERED_FLOW_LAYER = "Filtered Flow Magnitude"
_FOREGROUND_MASK_LAYER = "Foreground Mask"
_TRACKED_CELL_LAYER = "Tracked: Cell"
_CELL_ZAVG_LAYER = "Cell z-avg"
_NUC_ZAVG_LAYER = "Nucleus z-avg"
_FF_SPIN_WIDTH = 80
_FF_SPIN_MIN_WIDTH = int(_FF_SPIN_WIDTH * 0.9)


class CellWorkflowWidget(QWidget):
    """Cell segmentation — Flow-Following Segmentation."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._ff_worker = None
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        def _stage_files(group_label: str, entries: list[tuple[str, str]]) -> PipelineFilesWidget:
            return PipelineFilesWidget([(group_label, entries)], viewer=self.viewer)

        def _stage_status() -> QLabel:
            label = QLabel("")
            label.setWordWrap(True)
            label.setVisible(False)
            status_label(label)
            return label

        def _stage_progress() -> QProgressBar:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setVisible(False)
            return bar

        def _param_grid():
            grid = block_grid(horizontal_spacing=12, vertical_spacing=4)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)
            return grid

        def _dspin(lo, hi, val, step, decimals=1):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
            s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return s

        def _ispin(lo, hi, val, step=1):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
            s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return s

        self.filtered_flow_params_widget = QWidget()
        filter_lay = QVBoxLayout(self.filtered_flow_params_widget)
        filter_lay.setContentsMargins(0, 0, 0, 0)
        filter_lay.setSpacing(4)
        filter_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.filtered_flow_input_files = _stage_files("Inputs", [
            ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
            ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
        ])
        filter_lay.addWidget(self.filtered_flow_input_files)
        filter_grid = _param_grid()
        self.ff_median_time_spin   = _ispin(1, 15, 3)
        self.ff_median_space_spin  = _ispin(1, 15, 5)
        self.ff_gauss_time_spin    = _dspin(0.0, 10.0, 0.0, 0.1)
        self.ff_gauss_space_spin   = _dspin(0.0, 10.0, 0.0, 0.1)
        add_parameter_grid_row(filter_grid, 0, 0, "Median t kernel:", self.ff_median_time_spin)
        add_parameter_grid_row(filter_grid, 0, 1, "Median xy kernel:", self.ff_median_space_spin)
        add_parameter_grid_row(filter_grid, 1, 0, "Gaussian t sigma:", self.ff_gauss_time_spin)
        add_parameter_grid_row(filter_grid, 1, 1, "Gaussian xy sigma:", self.ff_gauss_space_spin)
        filter_lay.addLayout(filter_grid)

        self.ff_flow_mag_btn = QPushButton("Create filtered_dp")
        self.ff_flow_mag_btn.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
        self.ff_flow_mag_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        filter_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(filter_btn_row, 0, self.ff_flow_mag_btn)
        filter_lay.addLayout(filter_btn_row)
        self.filtered_flow_status_lbl = _stage_status()
        filter_lay.addWidget(self.filtered_flow_status_lbl)
        self.filtered_flow_progress_bar = _stage_progress()
        filter_lay.addWidget(self.filtered_flow_progress_bar)
        self.filtered_flow_output_files = _stage_files("Outputs", [
            ("3_cell/filtered_dp.tif", "Filtered flow vectors"),
            ("3_cell/filtered_flow_mag.tif", "Filtered flow magnitude"),
        ])
        filter_lay.addWidget(self.filtered_flow_output_files)

        self.filtered_flow_section = CollapsibleSection(
            "Flow Filtering", self.filtered_flow_params_widget, expanded=False
        )
        layout.addWidget(self.filtered_flow_section)

        # ---- Contour Maps + Segmentation (embedded from CellBoundaryWorkflowWidget) ----
        self._seg_widget = CellBoundaryWorkflowWidget(self.viewer, parent=None)
        self._seg_widget.correction_section.setVisible(False)
        self._seg_widget.boundary_selection_section.set_title("Segmentation")
        self._seg_widget.contour_section.set_title("Contour Maps")
        # Remove the internal stretch so it doesn't create extra dead space
        seg_layout = self._seg_widget.layout()
        last_item = seg_layout.itemAt(seg_layout.count() - 1)
        if last_item and last_item.spacerItem():
            seg_layout.removeItem(last_item)
        layout.addWidget(self._seg_widget)

        self.correction_params_widget = QWidget()
        correction_lay = QVBoxLayout(self.correction_params_widget)
        correction_lay.setContentsMargins(0, 0, 0, 0)
        correction_lay.setSpacing(4)
        correction_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.correction_input_files = _stage_files("Inputs", [
            ("3_cell/tracked_labels.tif", "Cell labels"),
            ("0_input/cell_zavg.tif", "Cell z-avg"),
            ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
        ])
        correction_lay.addWidget(self.correction_input_files)

        self.load_cell_correction_btn = QPushButton("Load Cell Labels")
        self.save_cell_correction_btn = QPushButton("Save Cell Labels")
        self.reassign_cell_ids_btn = QPushButton("Reassign IDs")
        self.expand_selected_cell_btn = QPushButton("Expand Selected Cell")
        for button in (
            self.load_cell_correction_btn,
            self.save_cell_correction_btn,
            self.reassign_cell_ids_btn,
            self.expand_selected_cell_btn,
        ):
            button.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        correction_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(
            correction_btn_row,
            0,
            self.load_cell_correction_btn,
            self.save_cell_correction_btn,
        )
        add_block_button_row(correction_btn_row, 1, self.reassign_cell_ids_btn)
        correction_lay.addLayout(correction_btn_row)

        expand_grid = _param_grid()
        self.expand_cell_max_px_spin = _ispin(0, 999, 25)
        add_parameter_grid_row(
            expand_grid,
            0,
            0,
            "Max expansion px:",
            self.expand_cell_max_px_spin,
        )
        correction_lay.addLayout(expand_grid)
        expand_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(expand_btn_row, 0, self.expand_selected_cell_btn)
        correction_lay.addLayout(expand_btn_row)

        self.correction_status_lbl = _stage_status()
        correction_lay.addWidget(self.correction_status_lbl)
        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
        )
        correction_lay.addWidget(self.correction_widget)
        self.correction_shortcuts_section = CollapsibleSection(
            "Correction Shortcuts",
            self.correction_widget.build_shortcuts_widget(),
            expanded=False,
        )
        correction_lay.addWidget(self.correction_shortcuts_section)
        self.correction_section = CollapsibleSection(
            "Correction", self.correction_params_widget, expanded=False
        )
        layout.addWidget(self.correction_section)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.ff_flow_mag_btn.clicked.connect(self._on_create_flow_mag)
        self.load_cell_correction_btn.clicked.connect(self._on_load_cell_correction)
        self.save_cell_correction_btn.clicked.connect(self._on_save_cell_correction)
        self.reassign_cell_ids_btn.clicked.connect(self._on_reassign_cell_ids)
        self.expand_selected_cell_btn.clicked.connect(self._on_expand_selected_cell)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _prob_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_prob_3dt.tif" if self._pos_dir else None

    def _dp_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_dp_3dt.tif" if self._pos_dir else None

    def _foreground_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "foreground_masks.tif" if self._pos_dir else None

    def _nucleus_labels_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    def _flow_mag_out_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "filtered_flow_mag.tif" if self._pos_dir else None

    def _filtered_dp_out_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "filtered_dp.tif" if self._pos_dir else None

    def _cell_labels_out_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "tracked_labels.tif" if self._pos_dir else None

    def _cell_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "cell_zavg.tif" if self._pos_dir else None

    def _nucleus_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "nucleus_zavg.tif" if self._pos_dir else None

    # ------------------------------------------------------------------
    # State + status
    # ------------------------------------------------------------------
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._refresh_stage_files(pos_dir)
        self._seg_widget.refresh(pos_dir)

    def _refresh_stage_files(self, pos_dir: Path | None = None) -> None:
        if pos_dir is None:
            pos_dir = self._pos_dir
        for files_widget in (
            self.filtered_flow_input_files,
            self.filtered_flow_output_files,
            self.correction_input_files,
        ):
            files_widget.refresh(pos_dir)

    def _set_correction_status(self, msg: str) -> None:
        self.correction_status_lbl.setText(msg)
        self.correction_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def get_state(self) -> dict:
        return {
            "flow_following": {
                "median_time":  self.ff_median_time_spin.value(),
                "median_space": self.ff_median_space_spin.value(),
                "gauss_time":   self.ff_gauss_time_spin.value(),
                "gauss_space":  self.ff_gauss_space_spin.value(),
            },
            "segmentation": self._seg_widget.get_state(),
        }

    def set_state(self, state: dict) -> None:
        if "flow_following" in state:
            ff = state["flow_following"]
            if "median_time"  in ff: self.ff_median_time_spin.setValue(ff["median_time"])
            if "median_space" in ff: self.ff_median_space_spin.setValue(ff["median_space"])
            if "gauss_time"   in ff: self.ff_gauss_time_spin.setValue(ff["gauss_time"])
            if "gauss_space"  in ff: self.ff_gauss_space_spin.setValue(ff["gauss_space"])
            if "segmentation" in state: 
                self._seg_widget.set_state(state["segmentation"])


    def _set_stage_status(self, stage: str, msg: str) -> None:
        label = self._stage_status_label(stage)
        label.setText(msg)
        label.setVisible(bool(msg))
        logger.info(msg)

    def _stage_status_label(self, stage: str) -> QLabel:
        return {
            "filtered_flow": self.filtered_flow_status_lbl,
        }[stage]

    def _stage_progress_bar(self, stage: str) -> QProgressBar:
        return {
            "filtered_flow": self.filtered_flow_progress_bar,
        }[stage]

    def _set_ff_buttons_running(self, running: bool) -> None:
        self.ff_flow_mag_btn.setEnabled(not running)
        if not running:
            self.filtered_flow_progress_bar.setValue(0)
            self.filtered_flow_progress_bar.setVisible(False)

    def _show_layer(self, layer_name: str, data: np.ndarray, kwargs: dict, adder) -> None:
        if layer_name in self.viewer.layers:
            try:
                self.viewer.layers[layer_name].data = data
            except Exception:
                self.viewer.layers.remove(self.viewer.layers[layer_name])
                adder(data, name=layer_name, **kwargs)
        else:
            adder(data, name=layer_name, **kwargs)

    def _on_stage_progress(self, stage: str, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            bar = self._stage_progress_bar(stage)
            if total > 0:
                bar.setVisible(True)
                bar.setRange(0, total)
                bar.setValue(done)
            self._set_stage_status(stage, msg)
        else:
            self._set_stage_status(stage, str(data))

    def _on_stage_worker_error(self, stage: str, exc: Exception) -> None:
        if self._ff_worker is None:
            return
        self._ff_worker = None
        self._set_ff_buttons_running(False)
        self._set_stage_status(stage, f"Error: {exc}")
        logger.exception("Cell workflow worker error", exc_info=exc)

    # ------------------------------------------------------------------
    # Manual correction
    # ------------------------------------------------------------------
    @staticmethod
    def _broadcast_reference_image(image: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray | None:
        if image is None:
            return None
        if image.ndim == 2 and len(shape) >= 3:
            return np.broadcast_to(image[np.newaxis], (shape[0],) + image.shape).copy()
        return image

    def _on_load_cell_correction(self) -> None:
        labels_path = self._cell_labels_out_path()
        cell_zavg_path = self._cell_zavg_path()
        nuc_zavg_path = self._nucleus_zavg_path()
        if labels_path is None or not labels_path.exists():
            self._set_correction_status("No cell labels file found.")
            return
        self._set_correction_status("Loading cell labels...")

        @thread_worker(connect={
            "returned": self._on_load_cell_correction_done,
            "errored": lambda exc: self._set_correction_status(f"Error: {exc}"),
        })
        def _worker():
            labels = read_full_tracked_stack(labels_path)
            cell_zavg = (
                np.asarray(tifffile.imread(str(cell_zavg_path)), dtype=np.float32)
                if cell_zavg_path and cell_zavg_path.exists() else None
            )
            nuc_zavg = (
                np.asarray(tifffile.imread(str(nuc_zavg_path)), dtype=np.float32)
                if nuc_zavg_path and nuc_zavg_path.exists() else None
            )
            return labels, cell_zavg, nuc_zavg

        _worker()

    def _on_load_cell_correction_done(self, result: tuple) -> None:
        labels, cell_zavg, nuc_zavg = result
        if _TRACKED_CELL_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_CELL_LAYER].data = labels
        else:
            self.viewer.add_labels(labels, name=_TRACKED_CELL_LAYER)

        for image, layer_name, cmap in (
            (self._broadcast_reference_image(cell_zavg, labels.shape), _CELL_ZAVG_LAYER, "gray"),
            (self._broadcast_reference_image(nuc_zavg, labels.shape), _NUC_ZAVG_LAYER, "bop orange"),
        ):
            if image is None:
                continue
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = image
            else:
                self.viewer.add_image(image, name=layer_name, colormap=cmap, blending="additive")

        self._set_correction_status(f"Loaded cell label stack {labels.shape} into napari.")
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
        self.correction_widget.activate_layer(layer)
        self.correction_section.expand()

    def set_selection_callback(self, fn) -> None:
        """Register a callback for cell correction label selection changes."""
        self.correction_widget.set_selection_callback(fn)

    def select_matching_cell_label(
        self,
        t: int,
        source_label: int,
        *,
        source_labels: np.ndarray | None = None,
    ) -> None:
        """Highlight the cell label that best overlaps a selected nucleus label."""
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            return
        if source_labels is None:
            if "Tracked: Nucleus" not in self.viewer.layers:
                return
            source_labels = np.asarray(self.viewer.layers["Tracked: Nucleus"].data)
        target_labels = np.asarray(self.viewer.layers[_TRACKED_CELL_LAYER].data)
        matched_label = best_overlapping_label(target_labels, source_labels, t, source_label)
        self.correction_widget.select_label(t, matched_label, notify=False)

    def _on_save_cell_correction(self) -> None:
        labels_path = self._cell_labels_out_path()
        if labels_path is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._set_correction_status("No cell labels layer to save.")
            return
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
        data = np.asarray(layer.data)
        if data.ndim != 3:
            self._set_correction_status("Cell labels layer is not a 3D stack.")
            return
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            str(labels_path),
            data.astype(np.uint32, copy=False),
            compression="zlib",
        )
        self._refresh_stage_files(self._pos_dir)
        self._set_correction_status(f"Saved {data.shape[0]} frame(s) to {labels_path.name}.")

    def _on_reassign_cell_ids(self) -> None:
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._set_correction_status("No cell labels layer loaded.")
            return
        stack = np.asarray(self.viewer.layers[_TRACKED_CELL_LAYER].data)
        unique_ids = np.unique(stack)
        unique_ids = unique_ids[unique_ids != 0]
        if unique_ids.size == 0:
            self._set_correction_status("No cell IDs to reassign.")
            return
        lut = np.zeros(int(unique_ids.max()) + 1, dtype=np.uint32)
        for new_id, old_id in enumerate(unique_ids, start=1):
            lut[int(old_id)] = new_id
        self.viewer.layers[_TRACKED_CELL_LAYER].data = lut[stack]
        self._set_correction_status(
            f"Reassigned {len(unique_ids)} cell IDs to contiguous range 1-{len(unique_ids)}. Unsaved."
        )

    def _foreground_stack_for_expansion(self) -> np.ndarray | None:
        if _FOREGROUND_MASK_LAYER in self.viewer.layers:
            return np.asarray(self.viewer.layers[_FOREGROUND_MASK_LAYER].data)
        fg_path = self._foreground_path()
        if fg_path is None or not fg_path.exists():
            return None
        foreground = np.asarray(tifffile.imread(str(fg_path)))
        self._show_layer(_FOREGROUND_MASK_LAYER, foreground, {}, self.viewer.add_labels)
        return foreground

    def _on_expand_selected_cell(self) -> None:
        if self._pos_dir is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked cell labels layer loaded.")
            return
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
        if self.correction_widget._layer is not layer:
            self._set_correction_status("No active tracked cell labels layer.")
            return
        label_id = int(self.correction_widget._selected_label)
        if label_id == 0:
            self._set_correction_status("No cell selected.")
            return

        labels = np.asarray(layer.data)
        if labels.ndim < 3:
            self._set_correction_status("Tracked cell labels layer is not a 3D stack.")
            return
        t = self._current_time_index(labels.shape[0])
        seg2d = self.correction_widget._frame_view(layer, t)
        if not np.any(seg2d == label_id):
            self._set_correction_status(f"Cell {label_id} not present at t={t}.")
            return

        foreground = self._foreground_stack_for_expansion()
        if foreground is None:
            self._set_correction_status("Foreground mask not found.")
            return
        if foreground.shape != labels.shape:
            self._set_correction_status(
                f"Foreground mask shape {foreground.shape} does not match labels shape {labels.shape}."
            )
            return
        foreground2d = foreground[t]
        while foreground2d.ndim > 2:
            if foreground2d.shape[0] != 1:
                self._set_correction_status(
                    f"Foreground mask frame has unsupported shape {foreground2d.shape}."
                )
                return
            foreground2d = foreground2d[0]

        before = seg2d.copy()
        try:
            added = expand_label_to_foreground(
                seg2d,
                foreground2d,
                label_id,
                max_distance=int(self.expand_cell_max_px_spin.value()),
            )
        except ValueError as exc:
            self._set_correction_status(str(exc))
            return
        if added == 0:
            seed_touches_foreground = bool(np.any((foreground2d > 0) & (before == label_id)))
            if not seed_touches_foreground:
                self._set_correction_status(
                    f"Cell {label_id} does not touch foreground at t={t}."
                )
            else:
                self._set_correction_status(f"Expansion added no pixels for cell {label_id} at t={t}.")
            return

        self.correction_widget._record_history(layer, t, before)
        layer.refresh()
        self.correction_widget._update_highlight(t, label_id)
        self._set_correction_status(
            f"Expanded cell {label_id} at t={t} by {added} px. Unsaved."
        )

    # ------------------------------------------------------------------
    # Run / Cancel
    # ------------------------------------------------------------------
    def _read_dp_tcyx(self, prob_path: Path, dp_path: Path) -> np.ndarray:
        from cellflow.database.hypotheses import normalize_seeded_watershed_dp_stack

        prob = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
        if prob.ndim == 3:
            prob = prob[np.newaxis]
        dp_raw = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
        dp_full = normalize_seeded_watershed_dp_stack(dp_raw, prob.shape)
        return dp_full[:, :, :2].mean(axis=1).astype(np.float32)

    def _on_create_flow_mag(self) -> None:
        if self._pos_dir is None:
            self._set_stage_status("filtered_flow", "No project open.")
            return

        prob_path = self._prob_path()
        dp_path = self._dp_path()
        filtered_dp_path = self._filtered_dp_out_path()
        flow_mag_path = self._flow_mag_out_path()

        for path, name in [
            (prob_path, "cell_prob_3dt.tif"),
            (dp_path,   "cell_dp_3dt.tif"),
        ]:
            if path is None or not path.exists():
                self._set_stage_status("filtered_flow", f"Missing: {name}")
                return
        if filtered_dp_path is None or flow_mag_path is None:
            self._set_stage_status("filtered_flow", "No project open.")
            return

        params_snapshot = self._params_from_ui()
        pos_dir = self._pos_dir

        def _on_done(result):
            self._ff_worker = None
            self._set_ff_buttons_running(False)
            filtered_mag = result
            self._show_layer(
                _FILTERED_FLOW_LAYER,
                filtered_mag,
                {"colormap": "inferno", "blending": "additive"},
                self.viewer.add_image,
            )
            self._refresh_stage_files(pos_dir)
            self._set_stage_status("filtered_flow", "Flow magnitude complete.")

        @thread_worker(connect={
            "yielded":  lambda data: self._on_stage_progress("filtered_flow", data),
            "returned": _on_done,
            "errored":  lambda exc: self._on_stage_worker_error("filtered_flow", exc),
        })
        def _worker():
            from cellflow.segmentation import compute_filtered_flow_vectors

            yield (0, 4, "Loading flow inputs...")
            dp_tcyx = self._read_dp_tcyx(prob_path, dp_path)

            yield (1, 4, "Filtering flow vectors...")
            filtered_dp = compute_filtered_flow_vectors(dp_tcyx, params_snapshot)

            yield (2, 4, "Creating flow magnitude...")
            filtered_mag = np.sqrt(
                filtered_dp[:, 0] ** 2 + filtered_dp[:, 1] ** 2
            ).astype(np.float32)

            yield (3, 4, "Saving flow magnitude...")
            filtered_dp_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(filtered_dp_path), filtered_dp, compression="zlib")
            tifffile.imwrite(str(flow_mag_path), filtered_mag, compression="zlib")
            return filtered_mag

        self._set_stage_status("filtered_flow", "Creating flow magnitude...")
        self._set_ff_buttons_running(True)
        self._ff_worker = _worker()

    def _current_time_index(self, max_t: int) -> int:
        step = getattr(getattr(self.viewer, "dims", None), "current_step", ())
        if not step:
            return 0
        return min(max(int(step[0]), 0), max(max_t - 1, 0))


    def _params_from_ui(self):
        from cellflow.segmentation import FlowFollowingParams
        return FlowFollowingParams(
            median_kernel_time=int(self.ff_median_time_spin.value()),
            median_kernel_space=int(self.ff_median_space_spin.value()),
            gaussian_sigma_time=float(self.ff_gauss_time_spin.value()),
            gaussian_sigma_space=float(self.ff_gauss_space_spin.value()),
        )


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/cellpose_widget.py
--------------------------------------------------------------------------------

"""Informational panel for external Cellpose output."""
from __future__ import annotations

from pathlib import Path

import napari
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from cellflow.napari.hpc_cellpose_widget import HpcCellposeWidget
from cellflow.napari.ui_style import muted_label
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget


class CellposeWidget(QWidget):
    """Informational panel for external Cellpose output."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        description = QLabel(
            "Cellpose runs externally on the cluster. This panel only documents "
            "the expected input/output files and loads them into napari."
        )
        description.setWordWrap(True)
        muted_label(description)
        layout.addWidget(description)

        self.input_files_tracker = PipelineFilesWidget([
            ("Inputs", [
                ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
                ("0_input/cell_zavg.tif", "Cell z-avg"),
                ("0_input/nucleus_3dt.tif", "Nucleus 3D+t"),
                ("0_input/cell_3dt.tif", "Cell 3D+t"),
            ]),
        ], viewer=self.viewer)
        layout.addWidget(self.input_files_tracker)

        self.hpc_cellpose_widget = HpcCellposeWidget(self.viewer)
        self.hpc_cellpose_section = CollapsibleSection(
            "HPC Cellpose", self.hpc_cellpose_widget, expanded=False
        )
        layout.addWidget(self.hpc_cellpose_section)

        self.output_files_tracker = PipelineFilesWidget([
            ("Outputs", [
                ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
                ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
                ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
                ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
                ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
                ("1_cellpose/cell_prob_zavg.tif", "Cell prob z-avg"),
            ]),
        ], viewer=self.viewer)
        layout.addWidget(self.output_files_tracker)

        self._pos_dir: Path | None = None

    def refresh(self, pos_dir: Path | None) -> None:
        """Update file status display."""
        self._pos_dir = pos_dir
        self.input_files_tracker.refresh(pos_dir)
        self.hpc_cellpose_widget.refresh(pos_dir)
        self.output_files_tracker.refresh(pos_dir)


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/correction_widget.py
--------------------------------------------------------------------------------

"""Label correction widget for CellFlow v2."""
from __future__ import annotations

import logging
import os
from typing import Callable

import napari
import napari.layers
import numpy as np
from napari.utils.notifications import show_error
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours

from cellflow.correction.labels import (
    _label_at,
    draw_cell_path,
    erase_cell,
    clean_stranded_pixels,
    fill_label_holes,
    fix_label_semiholes,
    merge_cells,
    relabel_cell,
    split_across,
    split_draw,
    swap_labels,
)
from cellflow.napari.ui_style import (
    action_button,
    checked_success_button,
    danger_button,
    muted_label,
    status_label,
)

log = logging.getLogger("cellflow.correction")
if os.environ.get("CELLFLOW_DEBUG"):
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[cellflow.correction] %(levelname)s %(message)s"))
        log.addHandler(_h)

_DRAW_LAYER      = "CorrectionDraw"
_HIGHLIGHT_LAYER = "CellHighlight"
_SPOTLIGHT_LAYER = "CellSpotlight"
_SPOTLIGHT_OPACITY = 0.7
_SPOTLIGHT_SCALE = 3.0


class CorrectionWidget(QWidget):
    """Dock widget for interactive label correction."""

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: QWidget | None = None,
        *,
        show_activate_btn: bool = True,
        show_shortcuts: bool = True,
        inspector_first: bool = False,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._show_activate_btn = show_activate_btn
        self._show_shortcuts = show_shortcuts
        self._inspector_first = inspector_first

        self._layer: napari.layers.Labels | None = None

        self._selected_label: int = 0
        self._selected_pos = None
        self._selected_t: int = -1
        self._ctrl_click_first = None
        self._ctrl_click_first_label: int = 0
        self._ctrl_click_first_t: int = -1
        self._swap_first_pos = None
        self._swap_first_t: int = -1

        self._drag_callbacks: list = []
        self._bound_keys: list = []

        self._in_deactivate: bool = False

        self._saved_viewer_drag_cbs: list = []
        self._saved_layer_mode: str = "pan_zoom"
        self._saved_layer_contour: int = 0

        self._edit_callback: Callable[[int, set[int]], None] | None = None
        self._selection_callback: Callable[[int, int], None] | None = None

        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)

        self._activate_btn = QPushButton("Activate on selected layer")
        self._activate_btn.setCheckable(True)
        self._activate_btn.setToolTip(
            "Enable interactive mouse callbacks for merging/splitting."
        )
        action_button(self._activate_btn, expand=True)
        checked_success_button(self._activate_btn)
        self._activate_btn.clicked.connect(self._toggle_active)
        if self._show_activate_btn:
            root.addWidget(self._activate_btn)

        self._outline_btn = QPushButton("Show outlines only")
        self._outline_btn.setCheckable(True)
        self._outline_btn.setEnabled(False)
        action_button(self._outline_btn, expand=True)
        self._outline_btn.clicked.connect(self._toggle_outline)
        root.addWidget(self._outline_btn)

        self._reset_mode_btn = QPushButton("⚠  Restore correction mode")
        self._reset_mode_btn.setVisible(False)
        action_button(self._reset_mode_btn, expand=True)
        danger_button(self._reset_mode_btn)
        self._reset_mode_btn.clicked.connect(self._reset_tool_mode)
        root.addWidget(self._reset_mode_btn)

        cleanup_label = QLabel("Artifact cleanup")
        muted_label(cleanup_label, size_pt=9)
        root.addWidget(cleanup_label)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope:"))
        self._cleanup_scope_combo = QComboBox()
        self._cleanup_scope_combo.addItems(["Current frame", "All frames"])
        self._cleanup_scope_combo.setToolTip(
            "Choose whether cleanup applies to the visible frame or the full label stack."
        )
        scope_row.addWidget(self._cleanup_scope_combo)
        root.addLayout(scope_row)

        hole_row = QHBoxLayout()
        hole_row.addWidget(QLabel("Hole radius:"))
        self._hole_radius_spin = QSpinBox()
        self._hole_radius_spin.setRange(0, 999)
        self._hole_radius_spin.setValue(5)
        self._hole_radius_spin.setToolTip(
            "Maximum pixel distance for filling enclosed background gaps. Set to 0 to skip gap filling."
        )
        hole_row.addWidget(self._hole_radius_spin)
        root.addLayout(hole_row)

        semihole_row = QHBoxLayout()
        semihole_row.addWidget(QLabel("Max opening:"))
        self._semihole_opening_spin = QSpinBox()
        self._semihole_opening_spin.setRange(0, 999)
        self._semihole_opening_spin.setValue(3)
        self._semihole_opening_spin.setToolTip(
            "Maximum border contact, in pixels, for semihole repair. Set to 0 to skip semihole repair."
        )
        semihole_row.addWidget(self._semihole_opening_spin)
        root.addLayout(semihole_row)

        self._fill_holes_btn = QPushButton("Fill Holes")
        self._fill_holes_btn.setEnabled(False)
        self._fill_holes_btn.setToolTip("Fill enclosed background gaps using the configured hole radius.")
        action_button(self._fill_holes_btn, expand=True)
        self._fill_holes_btn.clicked.connect(self._fill_holes)
        root.addWidget(self._fill_holes_btn)

        self._fix_semiholes_btn = QPushButton("Fix Semiholes")
        self._fix_semiholes_btn.setEnabled(False)
        self._fix_semiholes_btn.setToolTip(
            "Repair narrow border-connected gaps using the radius and max opening controls."
        )
        action_button(self._fix_semiholes_btn, expand=True)
        self._fix_semiholes_btn.clicked.connect(self._fix_semiholes)
        root.addWidget(self._fix_semiholes_btn)

        self._clean_fragments_btn = QPushButton("Clean Fragments")
        self._clean_fragments_btn.setEnabled(False)
        self._clean_fragments_btn.setToolTip("Remove disconnected same-label fragments without filling background holes.")
        action_button(self._clean_fragments_btn, expand=True)
        self._clean_fragments_btn.clicked.connect(self._clean_fragments)
        root.addWidget(self._clean_fragments_btn)

        self._status = QLabel("Inactive")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_label(self._status, italic=True, muted=True)
        root.addWidget(self._status)

        inspect_group = QGroupBox("Inspect cell")
        inspect_lay = QVBoxLayout(inspect_group)

        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Cell ID:"))
        self._goto_cell_id = QSpinBox()
        self._goto_cell_id.setRange(0, 999_999)
        self._goto_cell_id.setValue(0)
        self._goto_cell_id.setSpecialValueText("—")
        id_row.addWidget(self._goto_cell_id)
        self._goto_btn = QPushButton("Go")
        self._goto_btn.setEnabled(False)
        action_button(self._goto_btn, expand=True)
        self._goto_btn.clicked.connect(self._goto_cell)
        id_row.addWidget(self._goto_btn)
        inspect_lay.addLayout(id_row)

        self._inspect_frames_label = QLabel("")
        self._inspect_frames_label.setWordWrap(True)
        muted_label(self._inspect_frames_label, size_pt=9)
        inspect_lay.addWidget(self._inspect_frames_label)

        ref_group = self.build_shortcuts_widget()

        if self._inspector_first:
            root.addWidget(inspect_group)
            if self._show_shortcuts:
                root.addWidget(ref_group)
        else:
            if self._show_shortcuts:
                root.addWidget(ref_group)
            root.addWidget(inspect_group)

        root.addStretch()

        attrib = QLabel(
            "Correction tools adapted from "
            '<a href="https://github.com/Image-Analysis-Hub/Epicure">Epicure</a>.'
            "<br>If you use these tools, please cite:<br>"
            '<a href="https://doi.org/10.64898/2026.03.27.714683">'
            "doi:10.64898/2026.03.27.714683</a>"
        )
        attrib.setOpenExternalLinks(True)
        attrib.setWordWrap(True)
        muted_label(attrib, size_pt=9)
        root.addWidget(attrib)

    def build_shortcuts_widget(self) -> QWidget:
        group = QGroupBox("Correction shortcuts")
        lay = QVBoxLayout(group)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(6)
        for key, desc in [
            ("Left-click",                         "Select / highlight cell"),
            ("Middle-click",                       "Erase clicked cell"),
            ("Delete",                             "Erase selected cell"),
            ("Ctrl+Left-click (cell selected)",    "Merge with clicked cell"),
            ("Ctrl+Left-click × 2 (same cell)",    "Split (watershed, 2 seeds)"),
            ("Right-click (cell selected)",         "Swap with clicked cell (same or other frame)"),
            ("Ctrl+Right-click (cell selected)",   "Swap with clicked cell (same frame)"),
            ("Ctrl+Right-click → Right-click",     "Swap (two-step, no selection)"),
            ("Ctrl-z",                             "Undo"),
            ("Shift+Left / Shift+Right",           "Previous / next cell across all frames"),
            ("Shift+Right-drag",                   "Split by drawn line"),
            ("Shift+Left-drag",                    "Draw cell path (extends or creates)"),
        ]:
            row = QWidget()
            row_lay = QVBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(1)

            key_lbl = QLabel(f"<tt>{key}</tt>")
            key_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

            row_lay.addWidget(key_lbl)
            row_lay.addWidget(desc_lbl)
            lay.addWidget(row)
        return group

    # ── activation ────────────────────────────────────────────────────────────

    def _toggle_active(self, checked: bool) -> None:
        if checked:
            layer = self.viewer.layers.selection.active
            if layer is None:
                self._activate_btn.setChecked(False)
                self._set_status("Select a Labels layer first", error=True)
                return
            if not isinstance(layer, napari.layers.Labels):
                self._activate_btn.setChecked(False)
                self._set_status("Not a Labels layer", error=True)
                return
            self._activate(layer)
        else:
            self._deactivate()

    def _activate(self, layer: napari.layers.Labels) -> None:
        log.debug("activate: layer='%s' shape=%s", layer.name, layer.data.shape)
        self._layer = layer
        self._selected_label = 0
        self._selected_pos = None
        self._selected_t = -1
        self._ctrl_click_first = None
        self._ctrl_click_first_label = 0
        self._ctrl_click_first_t = -1
        self._swap_first_pos = None
        self._swap_first_t = -1

        if hasattr(self.viewer, "mouse_drag_callbacks"):
            self._saved_viewer_drag_cbs = list(self.viewer.mouse_drag_callbacks)
            self.viewer.mouse_drag_callbacks.clear()
        else:
            self._saved_viewer_drag_cbs = []

        self._saved_layer_mode = layer.mode
        self._saved_layer_contour = int(layer.contour)
        layer.mode = "pan_zoom"

        self.viewer.layers.selection.active = layer
        self._get_draw_layer()
        self._get_spotlight_layer()
        self._get_highlight_layer()

        self.viewer.dims.events.current_step.connect(self._on_dims_change)
        layer.events.data.connect(self._on_layer_data_changed)
        layer.events.paint.connect(self._on_layer_data_changed)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        layer.events.mode.connect(self._on_layer_mode_change)

        self._register_callbacks()
        self._activate_btn.setText("Deactivate")
        self._outline_btn.setEnabled(True)
        self._set_cleanup_enabled(True)
        self._outline_btn.setChecked(True)
        self._toggle_outline(True)
        self._goto_btn.setEnabled(True)
        self._set_status(f"Active on '{layer.name}'")

    def _deactivate(self) -> None:
        if self._in_deactivate:
            return
        self._in_deactivate = True
        try:
            self._deactivate_impl()
        finally:
            self._in_deactivate = False

    def _deactivate_impl(self) -> None:
        log.debug("deactivate: layer='%s'", self._layer.name if self._layer else None)
        if self._layer is not None:
            self._remove_callbacks()

            for disconnect in [
                lambda: self.viewer.dims.events.current_step.disconnect(self._on_dims_change),
                lambda: self.viewer.layers.events.removed.disconnect(self._on_layer_removed),
                lambda: self._layer.events.data.disconnect(self._on_layer_data_changed),
                lambda: self._layer.events.paint.disconnect(self._on_layer_data_changed),
                lambda: self._layer.events.mode.disconnect(self._on_layer_mode_change),
            ]:
                try:
                    disconnect()
                except Exception:
                    pass

            try:
                self._layer.mode = self._saved_layer_mode
            except Exception:
                pass
            try:
                self._layer.contour = self._saved_layer_contour
            except Exception:
                pass

            if hasattr(self.viewer, "mouse_drag_callbacks"):
                self.viewer.mouse_drag_callbacks.clear()
                for cb in self._saved_viewer_drag_cbs:
                    self.viewer.mouse_drag_callbacks.append(cb)

        self._layer = None
        self._selected_label = 0
        self._selected_pos = None
        self._selected_t = -1
        self._ctrl_click_first = None
        self._ctrl_click_first_label = 0
        self._ctrl_click_first_t = -1
        self._swap_first_pos = None
        self._swap_first_t = -1
        self._saved_viewer_drag_cbs = []

        self._activate_btn.setText("Activate on selected layer")
        self._activate_btn.setChecked(False)
        self._outline_btn.setChecked(False)
        self._outline_btn.setEnabled(False)
        self._set_cleanup_enabled(False)
        self._goto_btn.setEnabled(False)
        self._goto_cell_id.setValue(0)
        self._inspect_frames_label.setText("")
        self._set_status("Inactive")
        self._cleanup_draw_layer()
        self._cleanup_highlight_layer()
        self._cleanup_spotlight_layer()

    def activate_layer(self, layer: napari.layers.Labels) -> None:
        """Activate correction on a specific Labels layer (bypasses the UI button)."""
        if self._layer is not None:
            self._deactivate()
        self._activate(layer)
        self._activate_btn.setChecked(True)

    def deactivate(self) -> None:
        """Deactivate correction (public API)."""
        self._deactivate()

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status.setText(msg)
        if error:
            self._status.setStyleSheet("color: #b00020; font-style: italic;")
        else:
            status_label(self._status, italic=True, muted=True)

    def _set_cleanup_enabled(self, enabled: bool) -> None:
        for button in (
            self._fill_holes_btn,
            self._fix_semiholes_btn,
            self._clean_fragments_btn,
        ):
            button.setEnabled(enabled)

    def set_edit_callback(self, fn: Callable[[int, set[int]], None] | None) -> None:
        """Register a callback fired after every successful edit.
        Signature: fn(t: int, changed_ids: set[int]) -> None.
        Pass None to clear."""
        self._edit_callback = fn

    def set_selection_callback(self, fn: Callable[[int, int], None] | None) -> None:
        """Register a callback fired when the selected label changes.

        Signature: fn(t: int, label: int) -> None.  ``label`` is 0 when the
        selection is cleared.
        """
        self._selection_callback = fn

    def select_label(self, t: int, label: int, *, notify: bool = True) -> None:
        """Select and highlight *label* at frame *t*."""
        self._update_highlight(t, label, notify=notify)

    def _cleanup_frame_indices(self) -> list[int]:
        if self._layer is None:
            return []
        if self._layer.data.ndim < 3:
            return [0]
        if self._cleanup_scope_combo.currentText() == "All frames":
            return list(range(int(self._layer.data.shape[0])))
        return [int(self.viewer.dims.current_step[0])]

    def _run_artifact_cleanup(
        self,
        operation_name: str,
        no_change_message: str,
        operation: Callable[[np.ndarray], None],
    ) -> None:
        if self._layer is None:
            self._set_status("No active labels layer", error=True)
            return
        try:
            changed_frames = 0
            changed_pixels = 0
            for t in self._cleanup_frame_indices():
                seg2d = self._frame_view(self._layer, t)
                before = seg2d.copy()
                operation(seg2d)
                changed = int(np.sum(before != seg2d))
                if not changed:
                    continue
                changed_frames += 1
                changed_pixels += changed
                self._record_history(self._layer, t, before)

            if changed_pixels:
                self._layer.refresh()
                current_t = (
                    int(self.viewer.dims.current_step[0])
                    if self._layer.data.ndim >= 3
                    else 0
                )
                if self._selected_label:
                    self._update_highlight(current_t, self._selected_label)
                self._set_status(
                    f"{operation_name} in {changed_frames} frame(s), {changed_pixels} px changed. Unsaved."
                )
            else:
                self._set_status(no_change_message)
        except Exception as exc:
            show_error(f"cleanup error: {exc}")

    def _fill_holes(self) -> None:
        radius = int(self._hole_radius_spin.value())
        self._run_artifact_cleanup(
            "Filled holes",
            "No holes found",
            lambda seg2d: np.copyto(seg2d, fill_label_holes(seg2d, radius=radius)),
        )

    def _fix_semiholes(self) -> None:
        radius = int(self._hole_radius_spin.value())
        max_opening = int(self._semihole_opening_spin.value())
        self._run_artifact_cleanup(
            "Fixed semiholes",
            "No semiholes found",
            lambda seg2d: np.copyto(
                seg2d,
                fix_label_semiholes(seg2d, radius=radius, max_opening=max_opening),
            ),
        )

    def _clean_fragments(self) -> None:
        self._run_artifact_cleanup(
            "Cleaned fragments",
            "No fragments found",
            lambda seg2d: clean_stranded_pixels(seg2d),
        )

    @staticmethod
    def _frame_view(layer, t: int) -> np.ndarray:
        """Return a 2D writable view of frame *t* (squeezes singleton leading dims)."""
        if layer.data.ndim == 2:
            return layer.data
        v = layer.data[t]
        while v.ndim > 2:
            if v.shape[0] != 1:
                raise ValueError(f"non-singleton dim in frame slice: shape={v.shape}")
            v = v[0]
        return v

    def _next_free_label(self) -> int:
        """Return the next unused label across the full active stack."""
        if self._layer is None:
            return 1
        return int(np.max(self._layer.data)) + 1

    def _record_history(self, layer, t: int, before: np.ndarray) -> None:
        """Push changed pixels in frame *t* onto napari's undo stack and fire edit callback.

        ``before`` is a 2D snapshot of the frame; supports 3D and 4D underlying layers.
        """
        after = self._frame_view(layer, t)
        changed = np.where(before != after)
        if not changed[0].size:
            return
        n = changed[0].size
        # Build undo indices matching layer.data.ndim. Prepend t, fill any
        # extra leading dims (e.g. Z=1) with zeros, then the 2D (y, x) coords.
        extra = layer.data.ndim - 1 - 2
        parts = [np.full(n, t, dtype=layer.data.dtype)]
        parts.extend(np.zeros(n, dtype=layer.data.dtype) for _ in range(extra))
        parts.extend(changed)
        layer._save_history((tuple(parts), before[changed], after[changed]))
        if self._edit_callback is not None:
            ids = set(int(v) for v in before[changed]) | set(int(v) for v in after[changed])
            ids.discard(0)
            if ids:
                try:
                    self._edit_callback(t, ids)
                except Exception:
                    import logging as _logging
                    _logging.getLogger("cellflow.correction").exception("edit_callback failed")

    # ── draw layer ────────────────────────────────────────────────────────────

    def _get_draw_layer(self):
        if _DRAW_LAYER in self.viewer.layers:
            return self.viewer.layers[_DRAW_LAYER]
        dl = self.viewer.add_shapes(
            name=_DRAW_LAYER,
            ndim=2,
            edge_color="yellow",
            edge_width=1,
            face_color="transparent",
        )
        dl.visible = False
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer
        return dl

    def _cleanup_draw_layer(self) -> None:
        if _DRAW_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_DRAW_LAYER])

    # ── highlight layer ───────────────────────────────────────────────────────

    def _get_highlight_layer(self):
        if _HIGHLIGHT_LAYER in self.viewer.layers:
            return self.viewer.layers[_HIGHLIGHT_LAYER]
        hl = self.viewer.add_shapes(
            name=_HIGHLIGHT_LAYER,
            ndim=2,
            edge_color="cyan",
            edge_width=2,
            face_color="transparent",
        )
        hl.visible = False
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer
        return hl

    def _get_spotlight_layer(self):
        if _SPOTLIGHT_LAYER in self.viewer.layers:
            return self.viewer.layers[_SPOTLIGHT_LAYER]
        spotlight = self.viewer.add_image(
            np.zeros((1, 1, 4), dtype=np.float32),
            name=_SPOTLIGHT_LAYER,
            rgb=True,
            blending="translucent",
        )
        spotlight.visible = False
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer
        return spotlight

    def _notify_selection_changed(self, t: int, lab: int, previous_label: int) -> None:
        if lab == previous_label or self._selection_callback is None:
            return
        try:
            self._selection_callback(t, lab)
        except Exception:
            import logging as _logging
            _logging.getLogger("cellflow.correction").exception("selection_callback failed")

    def _update_highlight(self, t: int, lab: int, *, notify: bool = True) -> None:
        """Redraw the cyan boundary for *lab* at time *t*. Pass 0 to clear."""
        previous_label = self._selected_label
        self._selected_label = lab
        self._selected_t = t if lab != 0 else -1
        hl = self._get_highlight_layer()
        if lab == 0 or self._layer is None:
            hl.data = []
            hl.visible = False
            self._clear_spotlight()
            if notify:
                self._notify_selection_changed(t, lab, previous_label)
            return
        seg2d = self._frame_view(self._layer, t)
        if not np.any(seg2d == lab):
            self._selected_label = 0
            self._selected_t = -1
            hl.data = []
            hl.visible = False
            self._clear_spotlight()
            if notify:
                self._notify_selection_changed(t, 0, previous_label)
            return
        mask = (seg2d == lab).astype(np.uint8)
        contours = find_contours(mask, level=0.5)
        if not contours:
            self._selected_label = 0
            self._selected_t = -1
            hl.data = []
            hl.visible = False
            self._clear_spotlight()
            if notify:
                self._notify_selection_changed(t, 0, previous_label)
            return
        self._update_spotlight(mask.astype(bool))
        contour = max(contours, key=len)
        hl.data = [contour]
        hl.shape_type = ["polygon"]
        hl.visible = True
        self.viewer.layers.selection.active = self._layer
        if notify:
            self._notify_selection_changed(t, lab, previous_label)

    def _cleanup_highlight_layer(self) -> None:
        if _HIGHLIGHT_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_HIGHLIGHT_LAYER])

    def _update_spotlight(self, mask: np.ndarray) -> None:
        spotlight = self._get_spotlight_layer()
        outer_mask = self._scaled_mask(mask, scale=_SPOTLIGHT_SCALE)
        ring = outer_mask & ~mask
        alpha = np.full(mask.shape, _SPOTLIGHT_OPACITY, dtype=np.float32)
        if np.any(ring):
            inner_dist = distance_transform_edt(~mask)
            outer_dist = distance_transform_edt(outer_mask)
            denom = inner_dist + outer_dist
            ramp = np.divide(
                inner_dist,
                denom,
                out=np.zeros_like(inner_dist, dtype=np.float64),
                where=denom > 0,
            )
            alpha[ring] = (ramp[ring] * _SPOTLIGHT_OPACITY).astype(np.float32)
        alpha[mask] = 0.0
        data = np.zeros(mask.shape + (4,), dtype=np.float32)
        data[..., 3] = alpha
        spotlight.data = data
        spotlight.visible = True
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer

    @staticmethod
    def _scaled_mask(mask: np.ndarray, *, scale: float) -> np.ndarray:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return np.zeros_like(mask, dtype=bool)
        center = coords.mean(axis=0)
        yy, xx = np.indices(mask.shape)
        src_y = np.rint(center[0] + (yy - center[0]) / scale).astype(int)
        src_x = np.rint(center[1] + (xx - center[1]) / scale).astype(int)
        np.clip(src_y, 0, mask.shape[0] - 1, out=src_y)
        np.clip(src_x, 0, mask.shape[1] - 1, out=src_x)
        return mask[src_y, src_x]

    def _clear_spotlight(self) -> None:
        if _SPOTLIGHT_LAYER in self.viewer.layers:
            spotlight = self.viewer.layers[_SPOTLIGHT_LAYER]
            spotlight.data = np.zeros((1, 1, 4), dtype=np.float32)
            spotlight.visible = False

    def _cleanup_spotlight_layer(self) -> None:
        if _SPOTLIGHT_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_SPOTLIGHT_LAYER])

    def _on_dims_change(self, event=None) -> None:
        if not (self._selected_label and self._layer is not None):
            return
        step = self.viewer.dims.current_step
        if self._layer.data.ndim < 3 or len(step) < self._layer.data.ndim:
            return
        t = int(step[0])
        if t >= self._layer.data.shape[0]:
            return
        selected_label = self._selected_label
        selected_pos = self._selected_pos
        selected_t = self._selected_t
        self._update_highlight(t, selected_label, notify=False)
        self._selected_label = selected_label
        self._selected_pos = selected_pos
        self._selected_t = selected_t

    def _on_layer_data_changed(self, event=None) -> None:
        if not (self._selected_label and self._layer is not None):
            return
        step = self.viewer.dims.current_step
        if self._layer.data.ndim < 3 or len(step) < self._layer.data.ndim:
            return
        t = int(step[0])
        if t >= self._layer.data.shape[0]:
            return
        self._update_highlight(t, self._selected_label)

    def _on_layer_mode_change(self, event=None) -> None:
        if self._layer is None:
            return
        mode = getattr(event, "value", None) or self._layer.mode
        log.debug("_on_layer_mode_change: mode=%s", mode)
        if mode != "pan_zoom":
            self._reset_mode_btn.setVisible(True)
            self._set_status("Tool mode changed — corrections disabled", error=True)
        else:
            self._reset_mode_btn.setVisible(False)
            if self._layer is not None:
                self._set_status(f"Active on '{self._layer.name}'")

    def _on_layer_removed(self, event=None) -> None:
        removed = getattr(event, "value", None)
        removed_name = getattr(removed, "name", None)
        if removed is self._layer or removed_name in (_DRAW_LAYER, _HIGHLIGHT_LAYER, _SPOTLIGHT_LAYER):
            log.debug("_on_layer_removed: '%s' removed, deactivating", removed_name)
            self._deactivate()

    def _reset_tool_mode(self) -> None:
        if self._layer is not None:
            self._layer.mode = "pan_zoom"

    # ── inspect cell ──────────────────────────────────────────────────────────

    def _goto_cell(self) -> None:
        lab = self._goto_cell_id.value()
        if lab == 0:
            step = self.viewer.dims.current_step
            t = int(step[0]) if (self._layer is not None and self._layer.data.ndim >= 3 and len(step) >= 1) else 0
            self._update_highlight(t, 0)
            self._inspect_frames_label.setText("")
            return
        if self._layer is None:
            return
        data = self._layer.data
        frames = [i for i in range(data.shape[0]) if np.any(data[i] == lab)]
        if not frames:
            self._inspect_frames_label.setText(f"Cell {lab} not found in any frame.")
            step = self.viewer.dims.current_step
            t = int(step[0]) if len(step) >= 1 else 0
            self._update_highlight(t, 0)
            return
        _MAX = 20
        if len(frames) <= _MAX:
            frames_str = ", ".join(str(f) for f in frames)
        else:
            shown = ", ".join(str(f) for f in frames[:_MAX])
            frames_str = f"{shown}, … ({len(frames)} frames total)"
        self._inspect_frames_label.setText(f"Frames: {frames_str}")
        step = self.viewer.dims.current_step
        t = int(step[0]) if len(step) >= 1 else 0
        self._update_highlight(t, lab)

    def _step_cell(self, direction: int) -> None:
        if self._layer is None:
            return
        data = self._layer.data
        ids = sorted(set(int(v) for v in np.unique(data)) - {0})
        if not ids:
            self._set_status("No cells in any frame")
            return
        cur = self._selected_label
        if direction > 0:
            nxt = next((i for i in ids if i > cur), ids[0])
        else:
            nxt = next((i for i in reversed(ids) if i < cur), ids[-1])
        frames = [i for i in range(data.shape[0]) if np.any(data[i] == nxt)]
        if frames:
            step = list(self.viewer.dims.current_step)
            step[0] = frames[0]
            self.viewer.dims.current_step = tuple(step)
        self._goto_cell_id.setValue(nxt)
        self._goto_cell()

    # ── callback registration ─────────────────────────────────────────────────

    def _register_callbacks(self) -> None:
        layer = self._layer

        def key_delete(_layer):
            try:
                if self._selected_label == 0:
                    self._set_status("No cell selected — left-click a cell first")
                    return
                t = int(self.viewer.dims.current_step[0])
                seg2d = self._frame_view(_layer, t)
                before = seg2d.copy()
                if erase_cell(seg2d, label=self._selected_label):
                    self._record_history(_layer, t, before)
                    _layer.refresh()
                    self._update_highlight(t, 0)
                    self._set_status(f"Erased — Active on '{_layer.name}'")
            except Exception as exc:
                show_error(f"delete error: {exc}")

        def key_prev_cell(_layer):
            self._step_cell(-1)

        def key_next_cell(_layer):
            self._step_cell(1)

        for key, fn in [
            ("Delete", key_delete),
            ("Shift-Left", key_prev_cell),
            ("Shift-Right", key_next_cell),
        ]:
            layer.bind_key(key, fn, overwrite=True)
            self._bound_keys.append(key)

        def on_drag(_layer, event):
            try:
                if event.type != "mouse_press":
                    return

                t   = int(self.viewer.dims.current_step[0])
                btn = event.button
                mods = {m.name for m in event.modifiers}

                seg2d = self._frame_view(_layer, t)
                pos   = _layer.world_to_data(event.position)
                log.debug(
                    "on_drag: btn=%s mods=%s t=%d selected=%s",
                    btn, mods, t, self._selected_label,
                )

                # Middle-click: erase clicked cell
                if btn == 3 and not mods:
                    lab = _label_at(seg2d, pos)
                    if lab == 0:
                        return
                    before = seg2d.copy()
                    if erase_cell(seg2d, label=lab):
                        self._record_history(_layer, t, before)
                        _layer.refresh()
                        if lab == self._selected_label:
                            self._update_highlight(t, 0)
                        self._set_status(f"Erased — Active on '{_layer.name}'")
                    return

                # Ctrl+Right-click: swap
                if btn == 2 and mods == {"Control"}:
                    lab = _label_at(seg2d, pos)
                    if lab == 0:
                        self._set_status("Swap — click on a cell (not background)")
                        return
                    if (
                        self._selected_label != 0
                        and self._selected_pos is not None
                        and lab != self._selected_label
                    ):
                        before = seg2d.copy()
                        ok = swap_labels(seg2d, self._selected_pos, pos)
                        if ok:
                            self._record_history(_layer, t, before)
                            _layer.refresh()
                            self._selected_label = 0
                            self._selected_pos = None
                            self._selected_t = -1
                            self._update_highlight(t, 0)
                            self._set_status(f"Swapped — Active on '{_layer.name}'")
                        else:
                            self._set_status("Swap failed — click on two different cells")
                    else:
                        self._swap_first_pos = pos
                        self._swap_first_t = t
                        self._set_status(f"Swap — label {lab} selected, right-click second cell")
                    return

                # Plain Right-click: complete two-step swap, or pass label across frames
                if btn == 2 and not mods:
                    if self._swap_first_pos is not None:
                        if t != self._swap_first_t:
                            self._swap_first_pos = None
                            self._swap_first_t = -1
                            self._set_status("Frame changed — swap cancelled")
                        else:
                            before = seg2d.copy()
                            ok = swap_labels(seg2d, self._swap_first_pos, pos)
                            if ok:
                                self._record_history(_layer, t, before)
                                _layer.refresh()
                                self._swap_first_pos = None
                                self._swap_first_t = -1
                                self._set_status(f"Swapped — Active on '{_layer.name}'")
                            else:
                                self._set_status("Swap failed — click on two different cells")
                                self._swap_first_pos = None
                                self._swap_first_t = -1
                    elif self._selected_label != 0 and self._selected_t != -1:
                        before = seg2d.copy()
                        if t != self._selected_t:
                            # Pass selected label to a cell in a different frame
                            ok = relabel_cell(seg2d, pos, self._selected_label)
                            msg_ok  = f"Relabelled → {self._selected_label} — Active on '{_layer.name}'"
                            msg_err = "Relabel failed — click on a different cell"
                        else:
                            # Swap selected cell with right-clicked cell in same frame
                            ok = swap_labels(seg2d, self._selected_pos, pos)
                            msg_ok  = f"Swapped — Active on '{_layer.name}'"
                            msg_err = "Swap failed — click on a different cell"
                        if ok:
                            self._record_history(_layer, t, before)
                            _layer.refresh()
                            self._set_status(msg_ok)
                        else:
                            self._set_status(msg_err)
                    return

                # Ctrl+Left-click: merge or split
                if btn == 1 and mods == {"Control"}:
                    lab = _label_at(seg2d, pos)
                    if lab == 0:
                        self._set_status("Click on a cell, not background")
                        return

                    if self._ctrl_click_first is not None:
                        if t != self._ctrl_click_first_t:
                            self._ctrl_click_first = pos
                            self._ctrl_click_first_label = lab
                            self._ctrl_click_first_t = t
                            self._update_highlight(t, lab)
                            self._set_status(f"Frame changed — restarted: label {lab} selected")
                        elif lab == self._ctrl_click_first_label:
                            before = seg2d.copy()
                            ok = split_across(
                                seg2d, self._image_frame(t),
                                self._ctrl_click_first, pos,
                                new_label=self._next_free_label(),
                            )
                            self._set_status(
                                f"Split — Active on '{_layer.name}'"
                                if ok else "Split failed — seeds too close or result too small"
                            )
                            if ok:
                                self._record_history(_layer, t, before)
                            _layer.refresh()
                            self._ctrl_click_first = None
                            self._ctrl_click_first_label = 0
                            self._ctrl_click_first_t = -1
                            self._update_highlight(t, _label_at(seg2d, pos))
                        else:
                            self._ctrl_click_first = None
                            self._ctrl_click_first_label = 0
                            self._ctrl_click_first_t = -1

                    if self._ctrl_click_first is None:
                        if (
                            self._selected_label != 0
                            and lab != self._selected_label
                            and np.any(seg2d == self._selected_label)
                        ):
                            before = seg2d.copy()
                            ok = merge_cells(
                                seg2d, pos, pos,
                                label_a=lab, label_b=self._selected_label,
                            )
                            self._set_status(
                                f"Merged — Active on '{_layer.name}'"
                                if ok else "Merge failed — labels not touching"
                            )
                            if ok:
                                self._record_history(_layer, t, before)
                            _layer.refresh()
                            self._selected_label = 0
                            self._selected_pos = None
                            self._selected_t = -1
                            self._update_highlight(t, _label_at(seg2d, pos))
                        else:
                            self._ctrl_click_first = pos
                            self._ctrl_click_first_label = lab
                            self._ctrl_click_first_t = t
                            self._update_highlight(t, lab)
                            self._set_status(
                                f"Label {lab} — Ctrl+click same cell again for second split seed"
                            )
                    return

                # Plain Left-click: select / highlight cell
                if btn == 1 and not mods:
                    self._ctrl_click_first = None
                    self._ctrl_click_first_label = 0
                    self._ctrl_click_first_t = -1
                    self._swap_first_pos = None
                    self._swap_first_t = -1
                    lab = _label_at(seg2d, pos)
                    self._selected_pos = pos if lab != 0 else None
                    self._selected_t = t if lab != 0 else -1
                    self._update_highlight(t, lab)
                    if lab:
                        self._set_status(f"Selected label {lab} — Active on '{_layer.name}'")
                    else:
                        self._set_status(f"Active on '{_layer.name}'")
                    return

                # Shift+Right-drag: split by drawn line
                if mods == {"Shift"} and btn == 2:
                    dl = self._get_draw_layer()
                    dl.data = []
                    dl.visible = True
                    pos_list = [_layer.world_to_data(event.position)]
                    yield
                    while event.type == "mouse_move":
                        pos_list.append(_layer.world_to_data(event.position))
                        if len(pos_list) % 3 == 0:
                            dl.data = [np.array([[p[-2], p[-1]] for p in pos_list])]
                            dl.shape_type = ["path"]
                        yield
                    pos_list.append(_layer.world_to_data(event.position))
                    dl.data = []
                    dl.visible = False
                    self.viewer.layers.selection.active = _layer
                    curlabel = self._selected_label if self._selected_label else None
                    before = seg2d.copy()
                    ok = split_draw(
                        seg2d,
                        pos_list,
                        curlabel=curlabel,
                        new_label=self._next_free_label(),
                    )
                    self._set_status(
                        f"Split — Active on '{_layer.name}'"
                        if ok else "Split draw failed — line did not divide the cell"
                    )
                    if ok:
                        self._record_history(_layer, t, before)
                    _layer.refresh()
                    self._update_highlight(t, self._selected_label)
                    return

                # Shift+Left-drag: draw cell path
                if mods == {"Shift"} and btn == 1:
                    dl = self._get_draw_layer()
                    dl.data = []
                    dl.visible = True
                    pos_list = [_layer.world_to_data(event.position)]
                    yield
                    while event.type == "mouse_move":
                        pos_list.append(_layer.world_to_data(event.position))
                        if len(pos_list) % 3 == 0:
                            dl.data = [np.array([[p[-2], p[-1]] for p in pos_list])]
                            dl.shape_type = ["path"]
                        yield
                    pos_list.append(_layer.world_to_data(event.position))
                    dl.data = []
                    dl.visible = False
                    self.viewer.layers.selection.active = _layer
                    curlabel = self._selected_label if self._selected_label else None
                    before = seg2d.copy()
                    ok = draw_cell_path(
                        seg2d,
                        pos_list,
                        curlabel=curlabel,
                        new_label=self._next_free_label(),
                    )
                    self._set_status(
                        f"Drew cell path — Active on '{_layer.name}'"
                        if ok else "Draw failed — stroke too short"
                    )
                    if ok:
                        self._record_history(_layer, t, before)
                    _layer.refresh()
                    self._update_highlight(t, self._selected_label)
                    return

            except Exception as exc:
                import traceback
                show_error(f"Correction error: {exc}\n{traceback.format_exc()}")

        layer.mouse_drag_callbacks.append(on_drag)
        self._drag_callbacks.append(on_drag)

    def _remove_callbacks(self) -> None:
        layer = self._layer
        for fn in self._drag_callbacks:
            try:
                layer.mouse_drag_callbacks.remove(fn)
            except ValueError:
                pass
        self._drag_callbacks.clear()
        for key in self._bound_keys:
            try:
                layer.bind_key(key, None)
            except Exception:
                pass
        self._bound_keys.clear()

    def _toggle_outline(self, checked: bool) -> None:
        if self._layer is None:
            self._outline_btn.setChecked(False)
            return
        self._layer.contour = 1 if checked else 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _image_frame(self, t: int) -> np.ndarray | None:
        """Return the intensity image at frame *t* from the first Image layer found.

        Squeezes singleton leading dims so the result is always 2D."""
        for lyr in self.viewer.layers:
            if getattr(lyr, "name", None) == _SPOTLIGHT_LAYER:
                continue
            if isinstance(lyr, napari.layers.Image):
                d = lyr.data
                if d.ndim == 2:
                    return d
                v = d[t] if d.ndim >= 3 else d
                while v.ndim > 2:
                    if v.shape[0] != 1:
                        return None
                    v = v[0]
                return v
        return None


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/data_panel_widget.py
--------------------------------------------------------------------------------

"""Data panel widget for project and metadata management in CellFlow v2."""
from __future__ import annotations

from pathlib import Path
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.widgets import PipelineFilesWidget

# Define tracked files for the project-wide status view
_TRACKED_FILE_GROUPS = [
    ("Input Data", [
        ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
        ("0_input/cell_zavg.tif", "Cell z-avg"),
        ("0_input/NLS_zavg.tif", "NLS z-avg"),
        ("0_input/nucleus_3dt.tif", "Nucleus 3D+t"),
        ("0_input/cell_3dt.tif", "Cell 3D+t"),
        ("0_input/NLS_3dt.tif", "NLS 3D+t"),
    ]),
    ("Cellpose", [
        ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
        ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
        ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
        ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
        ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
        ("1_cellpose/cell_prob_zavg.tif", "Cell prob z-avg"),
    ]),
    ("Nucleus Workflow", [
        ("2_nucleus/contour_maps.tif", "Contour maps"),
        ("2_nucleus/foreground_masks.tif", "Foreground masks (external)"),
        ("2_nucleus/ultrack_workdir/data.db", "Ultrack DB"),
        ("2_nucleus/tracked_labels.tif", "Tracked labels"),
    ]),
    ("Cell Workflow", [
        ("3_cell/foreground_masks.tif", "Foreground masks (external)"),
        ("3_cell/filtered_dp.tif", "Filtered flow vectors"),
        ("3_cell/filtered_flow_mag.tif", "Filtered flow magnitude"),
        ("3_cell/tracked_labels.tif", "Tracked labels"),
    ]),
]


class ProjectStatusPanel(QWidget):
    """Widget for viewing project file status."""

    def __init__(self, viewer=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # ── File Tracker (Scrollable) ─────────
        self.file_tracker = PipelineFilesWidget(_TRACKED_FILE_GROUPS, viewer=viewer)
        
        scroll = QScrollArea()
        scroll.setWidget(self.file_tracker)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(150)
        scroll.setFrameShape(QFrame.NoFrame)
        layout.addWidget(scroll)

    def refresh(self, pos_dir: Path | None) -> None:
        """Update file status display."""
        self.file_tracker.refresh(pos_dir)


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/data_prep_widget.py
--------------------------------------------------------------------------------

"""Widget for exporting and preparing raw data."""
from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
import shlex

import napari
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QCheckBox,
    QProgressBar,
    QSpinBox,
)
from qtpy.QtCore import Qt, QTimer
from napari.qt.threading import thread_worker

from cellflow.napari.widgets import PipelineFilesWidget
from cellflow.napari.ui_style import muted_label, status_label
from cellflow.core.data_prep import DatasetConfig, discover_metadata, run as run_prep
from cellflow.napari.utils import launch_in_terminal

if TYPE_CHECKING:
    from cellflow.napari.main_widget import CellFlowMainWidget


class DataPrepWidget(QWidget):
    """Widget for exporting and preparing raw data."""

    def __init__(self, viewer: napari.Viewer, main_widget: CellFlowMainWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.main_widget = main_widget
        self._worker = None
        self._meta_worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # NDTiff path
        layout.addWidget(QLabel("NDTiff Directory:"))
        row = QHBoxLayout()
        self.ndtiff_edit = QLineEdit()
        self.ndtiff_edit.setPlaceholderText("/path/to/ndtiff_dataset")
        row.addWidget(self.ndtiff_edit)
        self.browse_btn = QPushButton("Browse...")
        row.addWidget(self.browse_btn)
        self.pull_btn = QPushButton("Pull Metadata")
        row.addWidget(self.pull_btn)
        layout.addLayout(row)

        # Metadata display (placeholders)
        meta_row = QHBoxLayout()
        self.px_label = QLabel("Pixel size: —")
        muted_label(self.px_label)
        meta_row.addWidget(self.px_label)
        self.dt_label = QLabel("Interval: —")
        muted_label(self.dt_label)
        meta_row.addWidget(self.dt_label)
        meta_row.addStretch()
        layout.addLayout(meta_row)

        # Positions
        layout.addWidget(QLabel("Positions (e.g. 0,1,2):"))
        self.pos_edit = QLineEdit("0")
        layout.addWidget(self.pos_edit)

        # XY downsample
        ds_row = QHBoxLayout()
        ds_row.addWidget(QLabel("XY Downsample:"))
        self.ds_spin = QSpinBox()
        self.ds_spin.setRange(1, 16)
        self.ds_spin.setValue(2)
        ds_row.addWidget(self.ds_spin)
        layout.addLayout(ds_row)

        # Frame range
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frames:"))
        frame_row.addWidget(QLabel("Start"))
        self.frame_start_spin = QSpinBox()
        self.frame_start_spin.setRange(0, 999999)
        self.frame_start_spin.setValue(0)
        frame_row.addWidget(self.frame_start_spin)
        frame_row.addWidget(QLabel("End"))
        self.frame_end_spin = QSpinBox()
        self.frame_end_spin.setRange(-1, 999999)
        self.frame_end_spin.setValue(-1)
        self.frame_end_spin.setSpecialValueText("last")
        frame_row.addWidget(self.frame_end_spin)
        layout.addLayout(frame_row)

        # Overwrite
        self.overwrite_check = QCheckBox("Overwrite existing files")
        layout.addWidget(self.overwrite_check)

        # Run buttons
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Export")
        self.term_btn = QPushButton("Run in Terminal")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.term_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        # Progress & Status
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status_label = QLabel("")
        status_label(self.status_label)
        layout.addWidget(self.status_label)

        # ── Project file status ──────────────────────────────────────────
        self.files_tracker = PipelineFilesWidget([
            ("Outputs", [
                ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
                ("0_input/cell_zavg.tif", "Cell z-avg"),
                ("0_input/NLS_zavg.tif", "NLS z-avg"),
                ("0_input/nucleus_3dt.tif", "Nucleus 3D+t"),
                ("0_input/cell_3dt.tif", "Cell 3D+t"),
                ("0_input/NLS_3dt.tif", "NLS 3D+t"),
                ("0_input/z_shift.csv", "Z shift CSV"),
            ]),
        ], viewer=self.viewer)
        layout.addWidget(self.files_tracker)

        # Connect signals
        self.browse_btn.clicked.connect(self._on_browse)
        self.pull_btn.clicked.connect(self._on_pull_metadata)
        self.run_btn.clicked.connect(self._on_run)
        self.term_btn.clicked.connect(self._on_run_in_terminal)
        self.cancel_btn.clicked.connect(self._on_cancel)
        
        # Debounce timer for auto-pull
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(500)
        self._debounce_timer.timeout.connect(self._on_pull_metadata)
        self.ndtiff_edit.textChanged.connect(self._debounce_timer.start)

        # Auto-refresh on project change
        self.main_widget.refresh_requested.connect(self.refresh)

    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select NDTiff Directory")
        if path:
            self.ndtiff_edit.setText(path)

    def _on_pull_metadata(self) -> None:
        path = self.ndtiff_edit.text().strip()
        if not path or not Path(path).exists():
            return

        if self._meta_worker is not None:
            return

        self.status_label.setText("Reading metadata...")

        @thread_worker(connect={"returned": self._on_metadata_returned, "errored": self._on_error})
        def _work():
            return discover_metadata(path)
        
        self._meta_worker = _work()

    def _on_metadata_returned(self, result: dict) -> None:
        self._meta_worker = None
        positions = result.get("positions", [])
        px = result.get("pixel_size_um")
        dt = result.get("time_interval_s")

        self.pos_edit.setText(",".join(str(p) for p in positions))
        if px:
            self.px_label.setText(f"Pixel size: {px:.4g} µm")
            self.main_widget.px_edit.setText(f"{px:.4g}")
        if dt:
            self.dt_label.setText(f"Interval: {dt/60:.4g} min")
            self.main_widget.dt_edit.setText(f"{dt/60:.4g}")
        
        self.status_label.setText(f"Metadata: {len(positions)} positions found.")

    def _on_run(self) -> None:
        root_dir = self.main_widget.path_label.text()
        if not root_dir or root_dir == "[no project]":
            self.status_label.setText("Error: No project open.")
            return

        ndtiff_path = self.ndtiff_edit.text().strip()
        if not ndtiff_path:
            self.status_label.setText("Error: No NDTiff directory.")
            return

        try:
            positions = [int(p.strip()) for p in self.pos_edit.text().split(",") if p.strip()]
        except ValueError:
            self.status_label.setText("Error: Invalid positions.")
            return

        config = DatasetConfig(
            ndtiff_path=ndtiff_path,
            root_dir=root_dir,
            positions=positions,
            xy_downsample=self.ds_spin.value(),
            frame_start=self.frame_start_spin.value(),
            frame_end=self.frame_end_spin.value(),
        )

        self.run_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Starting export...")

        overwrite = self.overwrite_check.isChecked()

        @thread_worker(connect={"yielded": self._on_progress, "finished": self._on_finished, "errored": self._on_error})
        def _work():
            for pos in positions:
                for done, total, label in run_prep(config, pos, overwrite=overwrite):
                    yield (pos, done, total, label)

        self._worker = _work()
        self._worker.aborted.connect(self._on_aborted)

    def _on_progress(self, update: tuple) -> None:
        pos, done, total, label = update
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.status_label.setText(f"pos{pos:02d} — {label} [{done}/{total}]")

    def _on_finished(self) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress.setVisible(False)
        self.status_label.setText("Export finished.")
        self._worker = None
        self.refresh(self.main_widget.path_label.text())

    def _on_error(self, exc: Exception) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress.setVisible(False)
        self.status_label.setText(f"Error: {exc}")
        self._worker = None
        self._meta_worker = None

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.quit()

    def _on_aborted(self) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress.setVisible(False)
        self.status_label.setText("Cancelled.")
        self._worker = None

    def _on_run_in_terminal(self) -> None:
        root_dir = self.main_widget.path_label.text()
        if not root_dir or root_dir == "[no project]":
            self.status_label.setText("Error: No project open.")
            return

        ndtiff_path = self.ndtiff_edit.text().strip()
        overwrite = self.overwrite_check.isChecked()
        ds = self.ds_spin.value()
        positions = self.pos_edit.text().strip()
        frame_start = self.frame_start_spin.value()
        frame_end = self.frame_end_spin.value()

        python_code = (
            "from cellflow.core.data_prep import DatasetConfig, run\n"
            f"config = DatasetConfig(ndtiff_path={ndtiff_path!r}, root_dir={root_dir!r}, "
            f"positions=[{positions}], xy_downsample={ds}, "
            f"frame_start={frame_start}, frame_end={frame_end})\n"
            "for pos in config.positions:\n"
            "    print(f'--- pos{pos} ---', flush=True)\n"
            f"    for d, t, l in run(config, pos, overwrite={overwrite}):\n"
            "        print(f'  {l} [{d}/{t}]', flush=True)"
        )
        cmd = f"python -c {shlex.quote(python_code)}"
        
        try:
            launch_in_terminal(cmd)
            self.status_label.setText("Launched in terminal.")
        except Exception as e:
            self.status_label.setText(f"Terminal error: {e}")

    def get_state(self) -> dict:
        """Return the current UI state as a dictionary."""
        return {
            "ndtiff_path": self.ndtiff_edit.text(),
            "positions": self.pos_edit.text(),
            "xy_downsample": self.ds_spin.value(),
            "frame_start": self.frame_start_spin.value(),
            "frame_end": self.frame_end_spin.value(),
            "overwrite": self.overwrite_check.isChecked(),
        }

    def set_state(self, state: dict) -> None:
        """Update the UI state from a dictionary."""
        if "ndtiff_path" in state:
            self.ndtiff_edit.setText(state["ndtiff_path"])
        if "positions" in state:
            self.pos_edit.setText(state["positions"])
        if "xy_downsample" in state:
            self.ds_spin.setValue(state["xy_downsample"])
        if "frame_start" in state:
            self.frame_start_spin.setValue(state["frame_start"])
        if "frame_end" in state:
            self.frame_end_spin.setValue(state["frame_end"])
        if "overwrite" in state:
            self.overwrite_check.setChecked(state["overwrite"])

    def refresh(self, pos_dir: Path | str | None) -> None:
        """Update file status display."""
        if pos_dir is None or str(pos_dir) == "[no project]":
            self.files_tracker.refresh(None)
            return
        
        p = Path(pos_dir)
        if not p.name.startswith("pos"):
            pos = self.main_widget.pos_spin.value()
            p = p / f"pos{pos:02d}"
        
        self.files_tracker.refresh(p)


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/hpc_cellpose_widget.py
--------------------------------------------------------------------------------

"""Standalone widget for launching the HPC Cellpose pipeline."""
from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path
from typing import Any

from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.ui_style import action_button, compact_spinbox, status_label
from cellflow.napari.utils import launch_in_terminal


DEFAULT_PIPELINE_SCRIPT = Path(
    "/home/aruppel/Projects/HPC/cellpose_full/run_pipeline.sh"
)
DEFAULT_CONFIG_PATH = Path(
    "/home/aruppel/Projects/HPC/cellpose_full/cellpose_full.json"
)


class HpcCellposeWidget(QWidget):
    """Widget for configuring and launching the external Cellpose pipeline."""

    def __init__(self, viewer: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.pipeline_script_path = DEFAULT_PIPELINE_SCRIPT

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        form = QFormLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)
        layout.addLayout(form)

        self.input_dir_edit = QLineEdit()
        self.input_dir_browse_btn = QPushButton("Browse...")
        form.addRow("Input dir:", self._path_row(self.input_dir_edit, self.input_dir_browse_btn))

        self.output_dir_edit = QLineEdit()
        self.output_dir_browse_btn = QPushButton("Browse...")
        form.addRow(
            "Output dir:",
            self._path_row(self.output_dir_edit, self.output_dir_browse_btn),
        )

        self.config_path_edit = QLineEdit(str(DEFAULT_CONFIG_PATH))
        self.config_path_browse_btn = QPushButton("Browse...")
        form.addRow(
            "Config:",
            self._path_row(self.config_path_edit, self.config_path_browse_btn),
        )

        self.nuclei_input_edit = QLineEdit("nucleus_3dt.tif")
        form.addRow("Nuclei input:", self.nuclei_input_edit)

        self.cells_input_edit = QLineEdit("cell_3dt.tif")
        form.addRow("Cells input:", self.cells_input_edit)

        self.frames_edit = QLineEdit("all")
        form.addRow("Frames:", self.frames_edit)

        self.nuclei_do_3d_check = QCheckBox("Nuclei 3D")
        form.addRow("", self.nuclei_do_3d_check)

        self.nuclei_anisotropy_spin = self._double_spin(0.01, 100.0, 1.5, 2)
        form.addRow("Nuclei anisotropy:", self.nuclei_anisotropy_spin)

        self.nuclei_diameter_spin = self._int_spin(0, 10000, 25)
        form.addRow("Nuclei diameter:", self.nuclei_diameter_spin)

        self.nuclei_size_spin = self._int_spin(0, 1000000, 0)
        form.addRow("Nuclei size:", self.nuclei_size_spin)

        self.nuclei_gamma_spin = self._double_spin(0.01, 100.0, 1.0, 2)
        form.addRow("Nuclei gamma:", self.nuclei_gamma_spin)

        self.cells_size_spin = self._int_spin(0, 1000000, 0)
        form.addRow("Cells size:", self.cells_size_spin)

        self.cells_gamma_spin = self._double_spin(0.01, 100.0, 1.0, 2)
        form.addRow("Cells gamma:", self.cells_gamma_spin)

        self.max_concurrent_jobs_spin = self._int_spin(1, 1000, 4)
        form.addRow("Max concurrent jobs:", self.max_concurrent_jobs_spin)

        self.remote_user_edit = QLineEdit("aruppel")
        form.addRow("Remote user:", self.remote_user_edit)

        self.remote_host_edit = QLineEdit("maestro.pasteur.fr")
        form.addRow("Remote host:", self.remote_host_edit)

        self.input_status_lbl = QLabel("")
        status_label(self.input_status_lbl, muted=True)
        layout.addWidget(self.input_status_lbl)

        self.run_btn = QPushButton("Run in Terminal")
        action_button(self.run_btn, expand=True)
        layout.addWidget(self.run_btn)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        status_label(self.status_lbl)
        layout.addWidget(self.status_lbl)

        self.input_dir_browse_btn.clicked.connect(self._browse_input_dir)
        self.output_dir_browse_btn.clicked.connect(self._browse_output_dir)
        self.config_path_browse_btn.clicked.connect(self._browse_config_path)
        self.run_btn.clicked.connect(self._on_run_terminal)

        for field in (
            self.input_dir_edit,
            self.nuclei_input_edit,
            self.cells_input_edit,
        ):
            field.textChanged.connect(self._update_input_status)

        self._update_input_status()

    def refresh(self, pos_dir: Path | str | None) -> None:
        """Derive default local pipeline paths from a selected position directory."""
        if pos_dir is None or str(pos_dir) == "[no project]":
            self.input_dir_edit.clear()
            self.output_dir_edit.clear()
            self._set_status("No project open.")
            self._update_input_status()
            return

        pos_path = Path(pos_dir)
        self.input_dir_edit.setText(str(pos_path / "0_input"))
        self.output_dir_edit.setText(str(pos_path / "1_cellpose"))
        self._set_status("")
        self._update_input_status()

    def get_state(self) -> dict[str, Any]:
        """Return the current control values."""
        return {
            "input_dir": self.input_dir_edit.text(),
            "output_dir": self.output_dir_edit.text(),
            "config_path": self.config_path_edit.text(),
            "nuclei_input": self.nuclei_input_edit.text(),
            "cells_input": self.cells_input_edit.text(),
            "frames": self.frames_edit.text(),
            "nuclei_do_3d": self.nuclei_do_3d_check.isChecked(),
            "nuclei_anisotropy": self.nuclei_anisotropy_spin.value(),
            "nuclei_diameter": self.nuclei_diameter_spin.value(),
            "nuclei_size": self.nuclei_size_spin.value(),
            "nuclei_gamma": self.nuclei_gamma_spin.value(),
            "cells_size": self.cells_size_spin.value(),
            "cells_gamma": self.cells_gamma_spin.value(),
            "max_concurrent_jobs": self.max_concurrent_jobs_spin.value(),
            "remote_user": self.remote_user_edit.text(),
            "remote_host": self.remote_host_edit.text(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Update controls from a previously saved state dictionary."""
        if "input_dir" in state:
            self.input_dir_edit.setText(str(state["input_dir"]))
        if "output_dir" in state:
            self.output_dir_edit.setText(str(state["output_dir"]))
        if "config_path" in state:
            self.config_path_edit.setText(str(state["config_path"]))
        if "nuclei_input" in state:
            self.nuclei_input_edit.setText(str(state["nuclei_input"]))
        if "cells_input" in state:
            self.cells_input_edit.setText(str(state["cells_input"]))
        if "frames" in state:
            self.frames_edit.setText(str(state["frames"]))
        if "nuclei_do_3d" in state:
            self.nuclei_do_3d_check.setChecked(bool(state["nuclei_do_3d"]))
        if "nuclei_anisotropy" in state:
            self.nuclei_anisotropy_spin.setValue(float(state["nuclei_anisotropy"]))
        if "nuclei_diameter" in state:
            self.nuclei_diameter_spin.setValue(int(state["nuclei_diameter"]))
        if "nuclei_size" in state:
            self.nuclei_size_spin.setValue(int(state["nuclei_size"]))
        if "nuclei_gamma" in state:
            self.nuclei_gamma_spin.setValue(float(state["nuclei_gamma"]))
        if "cells_size" in state:
            self.cells_size_spin.setValue(int(state["cells_size"]))
        if "cells_gamma" in state:
            self.cells_gamma_spin.setValue(float(state["cells_gamma"]))
        if "max_concurrent_jobs" in state:
            self.max_concurrent_jobs_spin.setValue(int(state["max_concurrent_jobs"]))
        if "remote_user" in state:
            self.remote_user_edit.setText(str(state["remote_user"]))
        if "remote_host" in state:
            self.remote_host_edit.setText(str(state["remote_host"]))
        self._update_input_status()

    def build_runtime_config(self) -> dict[str, Any]:
        """Build the temporary JSON payload consumed by the pipeline script."""
        return {
            "input_dir": self.input_dir_edit.text().strip(),
            "frames": self.frames_edit.text().strip() or "all",
            "nuclei": {
                "input": self.nuclei_input_edit.text().strip(),
                "do_3d": self.nuclei_do_3d_check.isChecked(),
                "anisotropy": self.nuclei_anisotropy_spin.value(),
                "diameter": self.nuclei_diameter_spin.value(),
                "size": self.nuclei_size_spin.value(),
                "gamma": self.nuclei_gamma_spin.value(),
            },
            "cells": {
                "input": self.cells_input_edit.text().strip(),
                "size": self.cells_size_spin.value(),
                "gamma": self.cells_gamma_spin.value(),
            },
        }

    def build_command(self, config_path: Path | str) -> str:
        """Build the shell command used for terminal launch."""
        parts = [
            "bash",
            str(self.pipeline_script_path),
            "--input-dir",
            self.input_dir_edit.text().strip(),
            "--output-dir",
            self.output_dir_edit.text().strip(),
            "--config",
            str(config_path),
            "--nuclei-input",
            self.nuclei_input_edit.text().strip(),
            "--cells-input",
            self.cells_input_edit.text().strip(),
            "--max-concurrent-jobs",
            str(self.max_concurrent_jobs_spin.value()),
            "--remote-user",
            self.remote_user_edit.text().strip(),
            "--remote-host",
            self.remote_host_edit.text().strip(),
        ]
        return " ".join(shlex.quote(part) for part in parts)

    def _on_run_terminal(self) -> None:
        error = self._validation_error()
        if error:
            self._set_status(error)
            return

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            suffix=".json",
            prefix="cellflow_hpc_cellpose_",
        ) as tmp:
            json.dump(self.build_runtime_config(), tmp, indent=2, sort_keys=True)
            tmp_path = Path(tmp.name)

        command = self.build_command(tmp_path)
        try:
            launch_in_terminal(command)
            self._set_status("HPC Cellpose command launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(command)
            self._set_status("Terminal launch failed; command copied to clipboard.")

    def _validation_error(self) -> str | None:
        script_path = Path(self.pipeline_script_path)
        config_path = Path(self.config_path_edit.text().strip())
        input_dir = Path(self.input_dir_edit.text().strip())
        nuclei_input = self.nuclei_input_edit.text().strip()
        cells_input = self.cells_input_edit.text().strip()

        if not str(input_dir):
            return "Error: No project open."
        if not script_path.is_file():
            return f"Error: Pipeline script missing: {script_path}"
        if not config_path.is_file():
            return f"Error: Config file missing: {config_path}"
        if not input_dir.is_dir():
            return f"Error: Input directory missing: {input_dir}"
        if self.max_concurrent_jobs_spin.value() < 1:
            return "Error: Invalid max concurrent jobs."
        if not nuclei_input or not (input_dir / nuclei_input).is_file():
            return f"Error: Missing nuclei input: {input_dir / nuclei_input}"
        if not cells_input or not (input_dir / cells_input).is_file():
            return f"Error: Missing cells input: {input_dir / cells_input}"
        return None

    def _update_input_status(self) -> None:
        input_dir_text = self.input_dir_edit.text().strip()
        if not input_dir_text:
            self.input_status_lbl.setText("Inputs: no input directory selected.")
            return

        input_dir = Path(input_dir_text)
        nuclei_path = input_dir / self.nuclei_input_edit.text().strip()
        cells_path = input_dir / self.cells_input_edit.text().strip()
        nuclei_status = "ok" if nuclei_path.is_file() else "missing"
        cells_status = "ok" if cells_path.is_file() else "missing"
        self.input_status_lbl.setText(
            f"Inputs: nuclei {nuclei_status}; cells {cells_status}."
        )

    def _browse_input_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if path:
            self.input_dir_edit.setText(path)

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_edit.setText(path)

    def _browse_config_path(self) -> None:
        path = QFileDialog.getOpenFileName(
            self,
            "Select Pipeline Config",
            self.config_path_edit.text(),
            "JSON (*.json);;All Files (*)",
        )[0]
        if path:
            self.config_path_edit.setText(path)

    def _set_status(self, message: str) -> None:
        self.status_lbl.setText(message)
        self.status_lbl.setVisible(bool(message))

    @staticmethod
    def _path_row(line_edit: QLineEdit, button: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return row

    @staticmethod
    def _int_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return compact_spinbox(spin)

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.1)
        spin.setValue(value)
        return compact_spinbox(spin)


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/main_widget.py
--------------------------------------------------------------------------------

"""Main widget for the CellFlow napari plugin."""
from __future__ import annotations

import json
from pathlib import Path

import napari
from qtpy.QtCore import Qt, QSize, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.analysis_widget import AnalysisWidget
from cellflow.napari.cell_workflow_widget import CellWorkflowWidget
from cellflow.napari.cellpose_widget import CellposeWidget
from cellflow.napari.data_panel_widget import ProjectStatusPanel
from cellflow.napari.data_prep_widget import DataPrepWidget
from cellflow.napari.meta_widget import MetaSourceBrowserWidget
from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget
from cellflow.napari.nls_classification_widget import NLSClassificationWidget
from cellflow.napari.widgets import CollapsibleSection
from cellflow.napari.ui_style import icon_button, muted_label, tiny_button


class CellFlowMainWidget(QWidget):
    """The unified workflow-based UI for CellFlow."""

    refresh_requested = Signal(object)  # emits pos_dir: Path | None

    def __init__(self, napari_viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = napari_viewer

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── Project Info (Top Level) ──────────────────────────────────
        self._setup_project_ui(main_layout)

        # Main scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(2, 2, 2, 2)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.scroll_widget)

        main_layout.addWidget(self.scroll)

        # Add sections
        self.data_panel = ProjectStatusPanel(self.viewer)
        self.data_section = CollapsibleSection(
            "Project Status", self.data_panel, expanded=False, title_color="#ADD8E6"
        )

        self._data_prep_widget = DataPrepWidget(self.viewer, self)
        self.prep_section = CollapsibleSection(
            "1. Data Preparation", self._data_prep_widget, expanded=False
        )

        self._cellpose_widget = CellposeWidget(self.viewer)
        self.cellpose_section = CollapsibleSection(
            "2. Cellpose", self._cellpose_widget, expanded=False
        )
        self.hpc_cellpose_widget = self._cellpose_widget.hpc_cellpose_widget

        self.nucleus_workflow_widget = NucleusWorkflowWidget(self.viewer)
        self.nucleus_section = CollapsibleSection(
            "3. Nucleus Segmentation & Tracking", self.nucleus_workflow_widget, expanded=False
        )

        self.cell_workflow_widget = CellWorkflowWidget(self.viewer)
        self.cell_section = CollapsibleSection(
            "4. Cell Segmentation", self.cell_workflow_widget, expanded=False
        )
        self._connect_label_selection_sync()

        self.analysis_widget = AnalysisWidget(self.viewer)
        self.analysis_section = CollapsibleSection(
            "5. Analysis", self.analysis_widget, expanded=False
        )

        self.nls_classification_widget = NLSClassificationWidget(self.viewer)
        self.nls_classification_section = CollapsibleSection(
            "5b. NLS Classification", self.nls_classification_widget, expanded=False
        )

        self.meta_source_browser = MetaSourceBrowserWidget(self.viewer)
        self.meta_section = CollapsibleSection(
            "6. Meta Analyzer", self.meta_source_browser, expanded=False
        )

        self.scroll_layout.addWidget(self.data_section)
        self.scroll_layout.addWidget(self.prep_section)
        self.scroll_layout.addWidget(self.cellpose_section)
        self.scroll_layout.addWidget(self.nucleus_section)
        self.scroll_layout.addWidget(self.cell_section)
        self.scroll_layout.addWidget(self.analysis_section)
        self.scroll_layout.addWidget(self.nls_classification_section)
        self.scroll_layout.addWidget(self.meta_section)

        # Add stretch at the end
        self.scroll_layout.addStretch()

        # Connect signals
        self.project_btn.clicked.connect(lambda: self._on_set_project_directory())
        self.save_btn.clicked.connect(lambda: self._on_save_config())
        self.save_as_btn.clicked.connect(lambda: self._on_save_config_as())
        self.load_btn.clicked.connect(lambda: self._on_load_config())
        self.load_from_btn.clicked.connect(lambda: self._on_load_config_from())
        
        self.refresh_btn.clicked.connect(lambda: self._refresh_all())
        self.pos_spin.valueChanged.connect(lambda: self._refresh_all())

    def _connect_label_selection_sync(self) -> None:
        """Synchronize selected cell/nucleus IDs across correction widgets."""
        if hasattr(self.nucleus_workflow_widget, "set_selection_callback"):
            self.nucleus_workflow_widget.set_selection_callback(
                lambda t, label: self.cell_workflow_widget.select_matching_cell_label(t, label)
            )
        if hasattr(self.cell_workflow_widget, "set_selection_callback"):
            self.cell_workflow_widget.set_selection_callback(
                lambda t, label: self.nucleus_workflow_widget.select_matching_nucleus_label(t, label)
            )

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(int(hint.width() * 1.5), hint.height())

    def _setup_project_ui(self, layout: QVBoxLayout) -> None:
        """Create the top-level project metadata and buttons."""
        proj_widget = QWidget()
        proj_lay = QVBoxLayout(proj_widget)
        proj_lay.setContentsMargins(0, 0, 0, 0)
        proj_lay.setSpacing(4)

        # Row 1: Metadata
        meta_row = QHBoxLayout()
        meta_row.setSpacing(4)
        
        meta_row.addWidget(QLabel("px:"))
        self.px_edit = QLineEdit()
        self.px_edit.setFixedWidth(40)
        meta_row.addWidget(self.px_edit)

        meta_row.addWidget(QLabel("dt:"))
        self.dt_edit = QLineEdit()
        self.dt_edit.setFixedWidth(40)
        meta_row.addWidget(self.dt_edit)

        meta_row.addWidget(QLabel("C:"))
        self.cond_edit = QLineEdit()
        meta_row.addWidget(self.cond_edit)

        meta_row.addWidget(QLabel("P:"))
        self.pos_spin = QSpinBox()
        self.pos_spin.setRange(0, 99)
        self.pos_spin.setFixedWidth(40)
        meta_row.addWidget(self.pos_spin)
        
        self.refresh_btn = QPushButton("↺")
        icon_button(self.refresh_btn)
        self.refresh_btn.setToolTip("Refresh all status")
        meta_row.addWidget(self.refresh_btn)
        
        proj_lay.addLayout(meta_row)

        # Row 2: Project Actions
        project_row = QHBoxLayout()
        project_row.setSpacing(4)
        self.project_btn = QPushButton("Project Directory...")
        tiny_button(self.project_btn)
        project_row.addWidget(self.project_btn)
        proj_lay.addLayout(project_row)

        # Row 3: Config Actions
        config_row = QHBoxLayout()
        config_row.setSpacing(4)
        self.save_btn = QPushButton("Save Config")
        self.save_as_btn = QPushButton("Save Config As...")
        self.load_btn = QPushButton("Load Config")
        self.load_from_btn = QPushButton("Load Config From...")
        
        for btn in (self.save_btn, self.save_as_btn, self.load_btn, self.load_from_btn):
            tiny_button(btn)
            config_row.addWidget(btn)
        proj_lay.addLayout(config_row)

        # Row 4: Path Label
        self.path_label = QLabel("[no project]")
        muted_label(self.path_label)
        self.path_label.setWordWrap(True)
        proj_lay.addWidget(self.path_label)

        layout.addWidget(proj_widget)

    def _on_set_project_directory(self) -> None:
        """Set the project directory and load config if present."""
        path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if path:
            p = Path(path)
            self.path_label.setText(str(p))
            self.path_label.setToolTip(str(p))
            
            # Look for config file
            config_path = p / "cellflow_config.json"
            if config_path.exists():
                self._load_config(str(config_path))
            
            self._refresh_all()

    def get_state(self) -> dict:
        """Return the current UI state as a dictionary."""
        return {
            "metadata": {
                "pixel_size_um": self.px_edit.text(),
                "time_interval_s": self.dt_edit.text(),
                "condition": self.cond_edit.text(),
                "position": self.pos_spin.value(),
            },
            "data_prep": self._data_prep_widget.get_state(),
            "hpc_cellpose": self.hpc_cellpose_widget.get_state(),
            "nucleus": self.nucleus_workflow_widget.get_state(),
            "cell": self.cell_workflow_widget.get_state(),
        }

    def set_state(self, state: dict) -> None:
        """Update the UI state from a dictionary."""
        if "metadata" in state:
            m = state["metadata"]
            if "pixel_size_um" in m: self.px_edit.setText(str(m["pixel_size_um"]))
            if "time_interval_s" in m: self.dt_edit.setText(str(m["time_interval_s"]))
            if "condition" in m: self.cond_edit.setText(str(m["condition"]))
            if "position" in m: self.pos_spin.setValue(int(m["position"]))

        if "data_prep" in state:
            self._data_prep_widget.set_state(state["data_prep"])

        if "hpc_cellpose" in state:
            self.hpc_cellpose_widget.set_state(state["hpc_cellpose"])
        
        if "nucleus" in state:
            self.nucleus_workflow_widget.set_state(state["nucleus"])
        
        if "cell" in state:
            self.cell_workflow_widget.set_state(state["cell"])

    def _on_save_config(self) -> None:
        """Save current configuration to project directory."""
        path_text = self.path_label.text()
        if not path_text or path_text == "[no project]":
            return
        
        config_path = Path(path_text) / "cellflow_config.json"
        self._save_config(str(config_path))

    def _on_save_config_as(self) -> None:
        """Save current configuration to a specific file."""
        path = QFileDialog.getSaveFileName(self, "Save Config As", filter="JSON (*.json)")[0]
        if path:
            self._save_config(path)

    def _on_load_config(self) -> None:
        """Load configuration from project directory."""
        path_text = self.path_label.text()
        if not path_text or path_text == "[no project]":
            return
        
        config_path = Path(path_text) / "cellflow_config.json"
        if config_path.exists():
            self._load_config(str(config_path))
        else:
            print(f"Config not found: {config_path}")

    def _on_load_config_from(self) -> None:
        """Load configuration from a specific file."""
        path = QFileDialog.getOpenFileName(self, "Load Config From", filter="JSON (*.json)")[0]
        if path:
            self._load_config(path)

    def _save_config(self, path: str) -> None:
        """Save state to a JSON file."""
        state = self.get_state()
        try:
            with open(path, "w") as f:
                json.dump(state, f, indent=4)
            print(f"Config saved to {path}")
        except Exception as e:
            print(f"Error saving config: {e}")

    def _load_config(self, path: str) -> None:
        """Load state from a JSON file."""
        try:
            with open(path, "r") as f:
                state = json.load(f)
            self.set_state(state)
            print(f"Config loaded from {path}")
        except Exception as e:
            print(f"Error loading config: {e}")

    def _refresh_all(self) -> None:
        """Refresh file status in all child widgets."""
        path_text = self.path_label.text()
        if path_text and path_text != "[no project]":
            pos = self.pos_spin.value()
            pos_dir = Path(path_text) / f"pos{pos:02d}"
        else:
            pos_dir = None

        self.data_panel.refresh(pos_dir)
        self._data_prep_widget.refresh(pos_dir)
        self._cellpose_widget.refresh(pos_dir)
        self.nucleus_workflow_widget.refresh(pos_dir)
        self.cell_workflow_widget.refresh(pos_dir)
        self.analysis_widget.refresh(pos_dir)
        self.nls_classification_widget.refresh(pos_dir)
        project_root = Path(path_text) if path_text and path_text != "[no project]" else None
        self.meta_source_browser.refresh(project_root)
        # Emit signal for other widgets
        self.refresh_requested.emit(pos_dir)


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/meta_widget.py
--------------------------------------------------------------------------------

"""Meta source browser widget for browsing and loading CellFlow meta-study positions."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cellflow.meta.catalog import (
    discover_h5_files,
    discover_study,
    load_meta_catalog,
    merge_catalog_records,
    records_from_h5_paths,
    save_meta_catalog,
)

try:  # pragma: no cover - local branch compatibility
    from cellflow.analysis.artifact_reader import read_position_artifact
except ImportError:  # pragma: no cover - tests monkeypatch this when absent
    def read_position_artifact(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("cellflow.analysis.artifact_reader is unavailable")

try:  # pragma: no cover - local branch compatibility
    from cellflow.napari.artifact_visualization import add_artifact_layers
except ImportError:  # pragma: no cover - tests monkeypatch this when absent
    def add_artifact_layers(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("cellflow.napari.artifact_visualization is unavailable")


class MetaSourceBrowserWidget(QWidget):
    """Browse and load positions from a CellFlow meta-study directory.

    Scans a root directory for ``condition/experiment/position`` trees,
    populates cascading combo boxes, and allows loading ready positions
    into the napari viewer.
    """

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._records: list[dict] = []
        self._csv_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # --- Catalog actions ---
        catalog_row = QHBoxLayout()
        self.open_catalog_btn = QPushButton("Open catalog")
        self.save_catalog_btn = QPushButton("Save catalog")
        catalog_row.addWidget(self.open_catalog_btn)
        catalog_row.addWidget(self.save_catalog_btn)
        layout.addLayout(catalog_row)

        source_row = QHBoxLayout()
        self.add_h5_btn = QPushButton("Add H5")
        self.autodiscover_folder_btn = QPushButton("Autodiscover folder")
        source_row.addWidget(self.add_h5_btn)
        source_row.addWidget(self.autodiscover_folder_btn)
        layout.addLayout(source_row)

        metadata_layout = QVBoxLayout()
        metadata_layout.setSpacing(2)
        self.condition_edit = QLineEdit()
        self.condition_edit.setPlaceholderText("Condition")
        self.experiment_edit = QLineEdit()
        self.experiment_edit.setPlaceholderText("Experiment")
        self.position_edit = QLineEdit()
        self.position_edit.setPlaceholderText("Position")
        self.labels_edit = QLineEdit()
        self.labels_edit.setPlaceholderText("Optional labels")
        for label, line_edit in (
            ("Condition:", self.condition_edit),
            ("Experiment:", self.experiment_edit),
            ("Position:", self.position_edit),
            ("Labels:", self.labels_edit),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(line_edit, 1)
            metadata_layout.addLayout(row)
        layout.addLayout(metadata_layout)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.catalog_tree = QTreeWidget()
        self.catalog_tree.setHeaderHidden(True)
        self.catalog_tree.setMinimumHeight(140)
        layout.addWidget(self.catalog_tree)

        # Internal selectors keep the previous public API and selection logic stable.
        self.condition_combo = QComboBox()
        self.experiment_combo = QComboBox()
        self.position_combo = QComboBox()

        # --- Load button ---
        self.load_source_btn = QPushButton("Load Source")
        layout.addWidget(self.load_source_btn)

        layout.addStretch()

        # --- Wire signals ---
        self.condition_combo.currentTextChanged.connect(self._on_condition_changed)
        self.experiment_combo.currentTextChanged.connect(self._on_experiment_changed)
        self.position_combo.currentTextChanged.connect(self._on_position_changed)
        self.catalog_tree.currentItemChanged.connect(self._on_tree_current_item_changed)
        self.open_catalog_btn.clicked.connect(self._on_open_catalog)
        self.save_catalog_btn.clicked.connect(self._on_save_catalog)
        self.add_h5_btn.clicked.connect(self._on_add_h5)
        self.autodiscover_folder_btn.clicked.connect(self._on_autodiscover_folder)
        self.load_source_btn.clicked.connect(self._on_load_source)

        self._update_load_button()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def refresh(self, root: Path | str | None) -> None:
        """Rescan *root* and repopulate the cascading selectors."""
        self._csv_path = None
        self._set_records([] if root is None else discover_study(Path(root)))

    def _set_records(self, records: list[dict]) -> None:
        """Replace records and repopulate the cascading selectors."""
        self._records = records

        # Block signals so repopulating doesn't fire cascading updates prematurely.
        self.condition_combo.blockSignals(True)
        self.experiment_combo.blockSignals(True)
        self.position_combo.blockSignals(True)

        self.condition_combo.clear()
        self.experiment_combo.clear()
        self.position_combo.clear()
        self.catalog_tree.clear()

        conditions = sorted({r["condition_id"] for r in self._records})
        self.condition_combo.addItems(conditions)

        self.condition_combo.blockSignals(False)
        self.experiment_combo.blockSignals(False)
        self.position_combo.blockSignals(False)

        if conditions:
            self._populate_catalog_tree()
            self._on_condition_changed(conditions[0])
        else:
            self._update_load_button()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _current_record(self) -> dict | None:
        """Return the record matching the three combo selections, if any."""
        cond = self.condition_combo.currentText()
        exp = self.experiment_combo.currentText()
        pos = self.position_combo.currentText()
        if not cond or not exp or not pos:
            return None
        for r in self._records:
            if (
                r["condition_id"] == cond
                and r["experiment_id"] == exp
                and r["position_id"] == pos
            ):
                return r
        return None

    def _update_load_button(self) -> None:
        """Enable *Load Source* only when the selected record is ready."""
        record = self._current_record()
        self.load_source_btn.setEnabled(
            record is not None and record.get("analysis_status") == "ready"
        )

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _catalog_defaults(self, *, include_position: bool) -> dict[str, str]:
        defaults = {
            "date": self.experiment_edit.text().strip() or "unknown_date",
            "condition": self.condition_edit.text().strip() or "unknown_condition",
            "labels": self.labels_edit.text().strip(),
        }
        position = self.position_edit.text().strip()
        if include_position and position:
            defaults["id"] = position
        return defaults

    def _populate_catalog_tree(self) -> None:
        self.catalog_tree.blockSignals(True)
        self.catalog_tree.clear()

        by_condition: dict[str, dict[str, list[dict]]] = {}
        for record in sorted(self._records, key=lambda r: (
            str(r["condition_id"]),
            str(r["experiment_id"]),
            str(r["position_id"]),
        )):
            by_condition.setdefault(record["condition_id"], {}).setdefault(
                record["experiment_id"], []
            ).append(record)

        first_position_item: QTreeWidgetItem | None = None
        for condition, experiments in by_condition.items():
            condition_item = QTreeWidgetItem([condition])
            condition_item.setData(0, Qt.UserRole, ("condition", condition, "", ""))
            self.catalog_tree.addTopLevelItem(condition_item)
            condition_item.setExpanded(True)

            for experiment, records in experiments.items():
                experiment_item = QTreeWidgetItem([experiment])
                experiment_item.setData(
                    0, Qt.UserRole, ("experiment", condition, experiment, "")
                )
                condition_item.addChild(experiment_item)
                experiment_item.setExpanded(True)

                for record in records:
                    position = record["position_id"]
                    position_item = QTreeWidgetItem([position])
                    position_item.setData(
                        0,
                        Qt.UserRole,
                        ("position", condition, experiment, position),
                    )
                    experiment_item.addChild(position_item)
                    if first_position_item is None:
                        first_position_item = position_item

        if first_position_item is not None:
            self.catalog_tree.setCurrentItem(first_position_item)
        self.catalog_tree.blockSignals(False)

    def _select_record_keys(self, condition: str, experiment: str, position: str) -> None:
        self.condition_combo.setCurrentText(condition)
        self.experiment_combo.setCurrentText(experiment)
        self.position_combo.setCurrentText(position)
        self._update_load_button()

    # ------------------------------------------------------------------
    # signal handlers
    # ------------------------------------------------------------------

    def _on_condition_changed(self, text: str) -> None:
        if not text:
            self.experiment_combo.clear()
            self.position_combo.clear()
            self._update_load_button()
            return

        experiments = sorted(
            {r["experiment_id"] for r in self._records if r["condition_id"] == text}
        )

        self.experiment_combo.blockSignals(True)
        self.experiment_combo.clear()
        self.experiment_combo.addItems(experiments)
        self.experiment_combo.blockSignals(False)

        if experiments:
            self._on_experiment_changed(experiments[0])
        else:
            self.position_combo.clear()
            self._update_load_button()

    def _on_experiment_changed(self, text: str) -> None:
        if not text:
            self.position_combo.clear()
            self._update_load_button()
            return

        cond = self.condition_combo.currentText()
        positions = sorted(
            {
                r["position_id"]
                for r in self._records
                if r["condition_id"] == cond and r["experiment_id"] == text
            }
        )

        self.position_combo.blockSignals(True)
        self.position_combo.clear()
        self.position_combo.addItems(positions)
        self.position_combo.blockSignals(False)

        if positions:
            self.position_combo.setCurrentIndex(0)
        self._update_load_button()

    def _on_position_changed(self, _text: str) -> None:
        self._update_load_button()

    def _on_tree_current_item_changed(
        self,
        current: QTreeWidgetItem | None,
        _previous: QTreeWidgetItem | None,
    ) -> None:
        if current is None:
            self._update_load_button()
            return
        data = current.data(0, Qt.UserRole)
        if not data:
            self._update_load_button()
            return
        level, condition, experiment, position = data
        if level == "condition":
            experiments = sorted(
                {
                    r["experiment_id"]
                    for r in self._records
                    if r["condition_id"] == condition
                }
            )
            experiment = experiments[0] if experiments else ""
        if level in {"condition", "experiment"}:
            positions = sorted(
                {
                    r["position_id"]
                    for r in self._records
                    if r["condition_id"] == condition and r["experiment_id"] == experiment
                }
            )
            position = positions[0] if positions else ""
        if condition and experiment and position:
            self._select_record_keys(condition, experiment, position)
        else:
            self._update_load_button()

    def _on_open_catalog(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Open meta catalog",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not selected:
            return

        csv_path = Path(selected)
        try:
            records = load_meta_catalog(csv_path)
        except (OSError, ValueError) as exc:
            self._set_status(str(exc))
            return

        self._csv_path = csv_path
        self._set_records(records)
        self._set_status(f"Loaded {len(records)} catalog row(s).")

    def _on_save_catalog(self) -> None:
        csv_path = self._csv_path
        if csv_path is None:
            selected, _filter = QFileDialog.getSaveFileName(
                self,
                "Save meta catalog",
                "",
                "CSV Files (*.csv);;All Files (*)",
            )
            if not selected:
                return
            csv_path = Path(selected)

        try:
            save_meta_catalog(csv_path, self._records)
        except OSError as exc:
            self._set_status(str(exc))
            return

        self._csv_path = csv_path
        self._set_status(f"Saved {len(self._records)} catalog row(s).")

    def _on_add_h5(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Add H5 source",
            "",
            "H5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if not selected:
            return

        incoming = records_from_h5_paths(
            [Path(selected)],
            defaults=self._catalog_defaults(include_position=True),
        )
        self._set_records(merge_catalog_records(self._records, incoming))
        self._set_status(f"Catalog contains {len(self._records)} source(s).")

    def _on_autodiscover_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Autodiscover H5 sources",
            "",
        )
        if not selected:
            return

        h5_paths = discover_h5_files(Path(selected), recursive=True)
        if not h5_paths:
            self._set_status("No H5 files found.")
            return

        incoming = records_from_h5_paths(
            h5_paths,
            defaults=self._catalog_defaults(include_position=False),
        )
        self._set_records(merge_catalog_records(self._records, incoming))
        self._set_status(f"Catalog contains {len(self._records)} source(s).")

    def _on_load_source(self) -> None:
        record = self._current_record()
        if record is None or record.get("analysis_status") != "ready":
            return
        if self.viewer is None:
            return

        artifact = read_position_artifact(record["artifact_path"])
        add_artifact_layers(self.viewer, artifact, prefix="[Meta] ")


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/nls_classification_widget.py
--------------------------------------------------------------------------------

"""Napari widget for patching position artifacts with NLS classifications."""
from __future__ import annotations

from pathlib import Path

from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from cellflow.analysis.nls_classification import (
    NLSClassificationSummary,
    patch_position_artifact_nls_classes,
)
from cellflow.napari.ui_style import action_button, status_label


class NLSClassificationWidget(QWidget):
    """Run NLS-high/NLS-low classification for the active position artifact."""

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._classify_worker = None
        self._classify_completion_pending = False
        self._classify_error_pending = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        status_label(self.status_lbl)
        layout.addWidget(self.status_lbl)

        self.classify_btn = QPushButton("Classify NLS Tracks")
        action_button(self.classify_btn, expand=True)
        layout.addWidget(self.classify_btn)

        layout.addStretch()

        self.classify_btn.clicked.connect(self._on_classify)
        self.refresh(None)

    @property
    def nls_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "NLS_zavg.tif" if self._pos_dir else None

    @property
    def nucleus_labels_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    @property
    def artifact_path(self) -> Path | None:
        return self._pos_dir / "4_analysis" / "position_analysis.h5" if self._pos_dir else None

    def refresh(self, pos_dir: Path | str | None) -> None:
        self._pos_dir = Path(pos_dir) if pos_dir is not None else None
        self._update_status()

    def _inputs_ready(self) -> bool:
        return (
            self.nls_zavg_path is not None
            and self.nls_zavg_path.exists()
            and self.nucleus_labels_path is not None
            and self.nucleus_labels_path.exists()
            and self.artifact_path is not None
            and self.artifact_path.exists()
        )

    def _update_status(self) -> None:
        self._update_action_states()
        if self._pos_dir is None:
            self._set_status("Status: no project open.")
            return
        if not self._inputs_ready():
            missing = []
            if self.nls_zavg_path is None or not self.nls_zavg_path.exists():
                missing.append("NLS image")
            if self.nucleus_labels_path is None or not self.nucleus_labels_path.exists():
                missing.append("nucleus labels")
            if self.artifact_path is None or not self.artifact_path.exists():
                missing.append("artifact")
            self._set_status(f"Status: missing {', '.join(missing)}.")
            return
        if not self.status_lbl.text() or self.status_lbl.text().startswith("Status: missing"):
            self._set_status("Status: ready.")

    def _update_action_states(self) -> None:
        running = self._classify_worker is not None
        self.classify_btn.setEnabled(self._inputs_ready() and not running)

    def _set_classify_running(self, running: bool) -> None:
        self._update_action_states()
        if not running:
            self._update_action_states()

    def _set_status(self, message: str) -> None:
        self.status_lbl.setText(message)

    def _on_classify_done(self, summary: NLSClassificationSummary) -> None:
        self._classify_completion_pending = True
        self._classify_worker = None
        self._set_classify_running(False)
        self._set_status(
            "Status: classified "
            f"{summary.track_count} tracks "
            f"(high={summary.high_track_count}, low={summary.low_track_count}) "
            f"into {summary.h5_path.name}"
        )
        self._update_action_states()

    def _on_classify_error(self, exc: Exception) -> None:
        self._classify_error_pending = True
        self._classify_worker = None
        self._set_classify_running(False)
        self._set_status(f"Status: error: {exc}")
        self._update_action_states()

    def _on_classify(self) -> None:
        if not self._inputs_ready():
            self._update_status()
            return

        artifact_path = self.artifact_path
        nls_zavg_path = self.nls_zavg_path
        nucleus_labels_path = self.nucleus_labels_path
        if artifact_path is None or nls_zavg_path is None or nucleus_labels_path is None:
            self._update_status()
            return

        self._classify_completion_pending = False
        self._classify_error_pending = False
        self._set_status("Status: classifying NLS tracks...")
        self._classify_worker = object()
        self._set_classify_running(True)

        @thread_worker(
            connect={
                "returned": self._on_classify_done,
                "errored": self._on_classify_error,
            }
        )
        def _worker():
            return patch_position_artifact_nls_classes(
                artifact_path,
                nls_zavg_path,
                nucleus_labels_path,
            )

        worker = _worker()
        self._classify_worker = worker
        if self._classify_completion_pending or self._classify_error_pending:
            self._classify_worker = None
            self._classify_completion_pending = False
            self._classify_error_pending = False
            self._update_action_states()


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/nucleus_workflow_widget.py
--------------------------------------------------------------------------------

"""Nucleus workflow widget for hypothesis generation and tracking in CellFlow v2."""
from __future__ import annotations

import logging
import os
import pickle
import shlex
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from napari.utils.colormaps import direct_colormap
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cellflow.correction.labels import best_overlapping_label
from cellflow.database.tracked import (
    read_full_tracked_stack,
    write_tracked_frame,
)
from cellflow.database.validation import (
    invalidate_track,
    is_track_validated,
    is_validated,
    read_validated_cells_at_frame,
    read_validated_frames,
    read_validated_tracks,
    remap_validated_tracks,
    validate_track,
)
from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.napari.ui_style import (
    action_button,
    add_block_button_row,
    add_block_checkbox_row,
    add_block_pair_row,
    add_parameter_grid_row,
    block_grid,
    compact_spinbox,
    danger_button,
    muted_label,
    status_label,
)
from cellflow.segmentation import ContourWatershedParams, compute_contour_watershed
from cellflow.tracking.retracker import retrack_frame_constrained
from cellflow.tracking_ultrack.config import TrackingConfig as UltrackConfig
from cellflow.tracking_ultrack.db_build import build_ultrack_database
from cellflow.tracking_ultrack.export import export_tracked_labels
from cellflow.tracking_ultrack.ingest import _select_solver
from cellflow.tracking_ultrack.extend import extend_track, extend_track_from_db
from cellflow.tracking_ultrack.solve import database_has_annotations, run_solve

logger = logging.getLogger(__name__)

try:
    from ultrack.core.segmentation.processing import segment as _ultrack_segment
except ImportError:
    _ultrack_segment = None  # type: ignore[assignment]

_PREVIEW_LAYER = "Preview: Nucleus"
_HYP_LAYER = "Hypothesis: Nucleus"
_TRACKED_LAYER = "Tracked: Nucleus"
_VALIDATED_OVERLAY = "Validated: Nucleus"
_SPOTLIGHT_LAYER = "CellSpotlight"
_VALIDATED_OVERLAY_OPACITY = 0.4
_CONTOUR_LAYER = "Contour Map: Nucleus"
_CELLPROB_LAYER = "Cellprob Map: Nucleus"
_FOREGROUND_SCORE_LAYER = "Foreground Score: Nucleus"
_FOREGROUND_MASK_LAYER = "Foreground Mask: Nucleus"
_CELL_ZAVG_LAYER = "Cell z-avg"
_NUC_ZAVG_LAYER = "Nucleus z-avg"
_ULTRACK_DB_PREVIEW_LAYER = "Ultrack DB Preview"
_ULTRACK_DB_SELECTION_LAYER = "Ultrack DB Selection"
_ULTRACK_DB_ANNOTATION_LAYER = "Ultrack DB Annotations"
_CONTOUR_SWEEP_WIDTH = 60
_CONTOUR_SWEEP_MIN_WIDTH = int(_CONTOUR_SWEEP_WIDTH * 0.9)


@dataclass(frozen=True)
class _HierarchyCutState:
    node_ids: tuple[int, ...]
    height: float | None


class NucleusWorkflowWidget(QWidget):
    """Nucleus hypothesis generation and tracking management."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._stop_flag: bool = False
        self._build_worker = None
        self._sweep_worker = None
        self._ultrack_db_preview_cache: dict[
            tuple,
            tuple[np.ndarray, str]
            | tuple[np.ndarray, str, dict[int, float]]
            | tuple[
                np.ndarray,
                str,
                dict[int, float],
                dict[int, int],
                dict[int, int],
            ],
        ] = {}
        self._ultrack_db_height_values_cache: dict[tuple, tuple[float, ...]] = {}
        self._ultrack_db_cut_state_cache: dict[tuple, tuple[_HierarchyCutState, ...]] = {}
        self._ultrack_db_browser_active: bool = False
        self._ultrack_db_frame_initialized: bool = False
        self._ultrack_db_selected_node_id: int | None = None
        self._ultrack_db_selected_frame: int | None = None
        self._ultrack_db_label_to_node_id: dict[int, int] = {}
        self._ultrack_db_node_id_to_label: dict[int, int] = {}
        self._ultrack_db_node_annotations: dict[int, str] = {}
        self._ultrack_db_preview_labels: np.ndarray | None = None
        self._ultrack_db_preview_mouse_callback = None
        self._setup_ui()
        self._connect_signals()

    # ──────────────────────────────────────────────────────────────────────────
    # UI setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        # ── Compact layout helpers ────────────────────────────────────────
        SPIN_MAX_W = 70

        def _compact(spin, w=SPIN_MAX_W):
            return compact_spinbox(spin, w)

        def _stage_files(group_label: str, entries: list[tuple[str, str]]) -> PipelineFilesWidget:
            return PipelineFilesWidget([(group_label, entries)], viewer=self.viewer)

        def _stage_status() -> QLabel:
            label = QLabel("")
            label.setWordWrap(True)
            label.setVisible(False)
            status_label(label)
            return label

        def _stage_progress() -> QProgressBar:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setVisible(False)
            return bar

        def _param_grid():
            grid = block_grid(horizontal_spacing=12, vertical_spacing=4)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)
            return grid

        def _param_group_label(text: str) -> QLabel:
            label = QLabel(text)
            label.setStyleSheet("font-weight: 600;")
            return label

        # ── 1. Contour Maps ───────────────────────────────────────────────
        _contour_inner = QWidget()
        contour_lay = QVBoxLayout(_contour_inner)
        contour_lay.setContentsMargins(4, 4, 4, 4)
        contour_lay.setSpacing(4)
        contour_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        cp_params_scroll = QScrollArea()
        cp_params_scroll.setWidgetResizable(True)
        cp_params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        cp_params_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        cp_params_scroll.setFrameShape(QFrame.NoFrame)
        cp_params_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        cp_params_widget = QWidget()
        cp_params_widget.setMinimumWidth(520)
        cp_params_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        cp_params_lay = QVBoxLayout(cp_params_widget)
        cp_params_lay.setContentsMargins(0, 0, 0, 0)
        cp_params_lay.setSpacing(4)
        cp_params_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.contour_input_files = _stage_files("Inputs", [
            ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
            ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
        ])
        cp_params_lay.addWidget(self.contour_input_files)

        self.cp_min_spin = QDoubleSpinBox()
        self.cp_min_spin.setRange(-20.0, 20.0)
        self.cp_min_spin.setValue(-3.0)
        self.cp_min_spin.setDecimals(1)
        self.cp_min_spin.setSingleStep(1.0)
        self.cp_min_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_min_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.cp_max_spin = QDoubleSpinBox()
        self.cp_max_spin.setRange(-20.0, 20.0)
        self.cp_max_spin.setValue(0.0)
        self.cp_max_spin.setDecimals(1)
        self.cp_max_spin.setSingleStep(1.0)
        self.cp_max_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_max_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.cp_step_spin = QDoubleSpinBox()
        self.cp_step_spin.setRange(0.1, 10.0)
        self.cp_step_spin.setValue(1.0)
        self.cp_step_spin.setDecimals(1)
        self.cp_step_spin.setSingleStep(0.5)
        self.cp_step_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_step_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.contour_flow_threshold_spin = QDoubleSpinBox()
        self.contour_flow_threshold_spin.setRange(0.0, 10.0)
        self.contour_flow_threshold_spin.setValue(0.0)
        self.contour_flow_threshold_spin.setDecimals(2)
        self.contour_flow_threshold_spin.setSingleStep(0.1)
        self.contour_flow_threshold_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.contour_flow_threshold_spin.setToolTip(
            "Cellpose flow error threshold passed to compute_masks. 0 disables filtering."
        )
        self.contour_flow_threshold_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self.cp_gamma_min_spin = QDoubleSpinBox()
        self.cp_gamma_min_spin.setRange(0.05, 5.0)
        self.cp_gamma_min_spin.setValue(1.0)
        self.cp_gamma_min_spin.setDecimals(2)
        self.cp_gamma_min_spin.setSingleStep(0.05)
        self.cp_gamma_min_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_gamma_min_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.cp_gamma_max_spin = QDoubleSpinBox()
        self.cp_gamma_max_spin.setRange(0.05, 5.0)
        self.cp_gamma_max_spin.setValue(1.0)
        self.cp_gamma_max_spin.setDecimals(2)
        self.cp_gamma_max_spin.setSingleStep(0.05)
        self.cp_gamma_max_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_gamma_max_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.cp_gamma_step_spin = QDoubleSpinBox()
        self.cp_gamma_step_spin.setRange(0.05, 2.0)
        self.cp_gamma_step_spin.setValue(0.25)
        self.cp_gamma_step_spin.setDecimals(2)
        self.cp_gamma_step_spin.setSingleStep(0.05)
        self.cp_gamma_step_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_gamma_step_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        _gamma_tip = (
            "Gamma correction on Cellpose probability logits before boundary building. "
            "<1 boosts dim signals; >1 suppresses them. 1.0 = no correction. "
            "Contour maps are averaged over all gamma values in [min, max]."
        )
        for _w in (self.cp_gamma_min_spin, self.cp_gamma_max_spin, self.cp_gamma_step_spin):
            _w.setToolTip(_gamma_tip)
        self.contour_fg_threshold_spin = QDoubleSpinBox()
        self.contour_fg_threshold_spin.setRange(0.0, 1.0)
        self.contour_fg_threshold_spin.setValue(0.5)
        self.contour_fg_threshold_spin.setDecimals(2)
        self.contour_fg_threshold_spin.setSingleStep(0.01)
        self.contour_fg_threshold_spin.setToolTip(
            "Threshold applied to the fuzzy foreground score written by Contour Maps"
        )
        self.save_source_check = QCheckBox("Save label images")
        self.save_source_check.setToolTip("Save all label images used for contour building in 2_nucleus/source_labels/")
        self.save_source_check.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        contour_sweep_grid = _param_grid()
        add_parameter_grid_row(contour_sweep_grid, 0, 0, "Cellprob min:", self.cp_min_spin)
        add_parameter_grid_row(contour_sweep_grid, 0, 1, "Cellprob max:", self.cp_max_spin)
        add_parameter_grid_row(contour_sweep_grid, 1, 0, "Cellprob step:", self.cp_step_spin)
        add_parameter_grid_row(contour_sweep_grid, 1, 1, "Flow threshold:", self.contour_flow_threshold_spin)
        cp_params_lay.addWidget(_param_group_label("Cellpose mask sweep"))
        cp_params_lay.addLayout(contour_sweep_grid)

        contour_gamma_grid = _param_grid()
        add_parameter_grid_row(contour_gamma_grid, 0, 0, "Gamma min:", self.cp_gamma_min_spin)
        add_parameter_grid_row(contour_gamma_grid, 0, 1, "Gamma max:", self.cp_gamma_max_spin)
        add_parameter_grid_row(contour_gamma_grid, 1, 0, "Gamma step:", self.cp_gamma_step_spin)
        cp_params_lay.addWidget(_param_group_label("Gamma averaging"))
        cp_params_lay.addLayout(contour_gamma_grid)

        contour_output_grid = _param_grid()
        add_parameter_grid_row(contour_output_grid, 0, 0, "FG threshold:", self.contour_fg_threshold_spin)
        contour_output_grid.addWidget(
            self.save_source_check,
            0,
            2,
            1,
            2,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        cp_params_lay.addWidget(_param_group_label("Foreground output"))
        cp_params_lay.addLayout(contour_output_grid)
        for spin in (
            self.cp_min_spin,
            self.cp_max_spin,
            self.cp_step_spin,
            self.contour_flow_threshold_spin,
            self.cp_gamma_min_spin,
            self.cp_gamma_max_spin,
            self.cp_gamma_step_spin,
            self.contour_fg_threshold_spin,
        ):
            spin.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )

        self.preview_contour_btn = QPushButton("Preview")
        self.preview_contour_btn.setToolTip(
            "Build contour maps for the current frame only and display in napari"
        )
        self.build_btn = QPushButton("Build")
        self.contour_terminal_btn = QPushButton("Run in Terminal")
        self.cancel_build_btn = QPushButton("Cancel")
        self.cancel_build_btn.setEnabled(False)

        for button in (
            self.preview_contour_btn,
            self.build_btn,
            self.contour_terminal_btn,
            self.cancel_build_btn,
        ):
            button.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )

        contour_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(
            contour_btn_row,
            0,
            self.preview_contour_btn,
            self.build_btn,
            self.contour_terminal_btn,
            self.cancel_build_btn,
        )
        cp_params_lay.addLayout(contour_btn_row)

        contour_filter_grid = _param_grid()
        self.contour_filter_median_time_spin = QSpinBox()
        self.contour_filter_median_time_spin.setRange(1, 15)
        self.contour_filter_median_time_spin.setValue(1)
        self.contour_filter_median_time_spin.setSingleStep(1)
        self.contour_filter_median_space_spin = QSpinBox()
        self.contour_filter_median_space_spin.setRange(1, 15)
        self.contour_filter_median_space_spin.setValue(1)
        self.contour_filter_median_space_spin.setSingleStep(1)
        self.contour_filter_gauss_time_spin = QDoubleSpinBox()
        self.contour_filter_gauss_time_spin.setRange(0.0, 10.0)
        self.contour_filter_gauss_time_spin.setValue(0.0)
        self.contour_filter_gauss_time_spin.setDecimals(1)
        self.contour_filter_gauss_time_spin.setSingleStep(0.1)
        self.contour_filter_gauss_space_spin = QDoubleSpinBox()
        self.contour_filter_gauss_space_spin.setRange(0.0, 10.0)
        self.contour_filter_gauss_space_spin.setValue(0.0)
        self.contour_filter_gauss_space_spin.setDecimals(1)
        self.contour_filter_gauss_space_spin.setSingleStep(0.1)
        for spin in (
            self.contour_filter_median_time_spin,
            self.contour_filter_median_space_spin,
            self.contour_filter_gauss_time_spin,
            self.contour_filter_gauss_space_spin,
        ):
            spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            spin.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        add_parameter_grid_row(contour_filter_grid, 0, 0, "Median t kernel:", self.contour_filter_median_time_spin)
        add_parameter_grid_row(contour_filter_grid, 0, 1, "Median xy kernel:", self.contour_filter_median_space_spin)
        add_parameter_grid_row(contour_filter_grid, 1, 0, "Gaussian t sigma:", self.contour_filter_gauss_time_spin)
        add_parameter_grid_row(contour_filter_grid, 1, 1, "Gaussian xy sigma:", self.contour_filter_gauss_space_spin)
        for spin in (
            self.contour_filter_median_time_spin,
            self.contour_filter_median_space_spin,
            self.contour_filter_gauss_time_spin,
            self.contour_filter_gauss_space_spin,
        ):
            spin.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        cp_params_lay.addWidget(_param_group_label("Post-filter contour maps"))
        cp_params_lay.addLayout(contour_filter_grid)

        self.preview_contour_filter_btn = QPushButton("Preview Filter")
        self.preview_contour_filter_btn.setToolTip(
            "Preview filtered contour_maps.tif in napari without overwriting it"
        )
        self.run_contour_filter_btn = QPushButton("Run Filter")
        self.run_contour_filter_btn.setToolTip(
            "Filter contour_maps.tif and overwrite contour_maps.tif"
        )
        for button in (self.preview_contour_filter_btn, self.run_contour_filter_btn):
            button.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        contour_filter_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(
            contour_filter_btn_row,
            0,
            self.preview_contour_filter_btn,
            self.run_contour_filter_btn,
        )
        cp_params_lay.addLayout(contour_filter_btn_row)

        self.contour_status_lbl = _stage_status()
        cp_params_lay.addWidget(self.contour_status_lbl)

        self.build_progress_bar = _stage_progress()
        self.contour_output_files = _stage_files("Outputs", [
            ("2_nucleus/contour_maps.tif", "Contour maps"),
            ("2_nucleus/foreground_scores.tif", "Foreground scores"),
            ("2_nucleus/foreground_masks.tif", "Foreground masks"),
        ])
        cp_params_lay.addWidget(self.build_progress_bar)
        cp_params_lay.addWidget(self.contour_output_files)

        cp_params_scroll.setWidget(cp_params_widget)
        contour_lay.addWidget(cp_params_scroll)
        self.contour_section = CollapsibleSection(
            "1. Contour Maps", _contour_inner, expanded=False
        )
        layout.addWidget(self.contour_section)

        # ── 2. Ultrack Database Generation ────────────────────────────────
        _db_gen_inner = QWidget()
        db_gen_lay = QVBoxLayout(_db_gen_inner)
        db_gen_lay.setContentsMargins(0, 0, 0, 0)
        db_gen_lay.setSpacing(4)
        db_gen_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.db_gen_input_files = _stage_files("Inputs", [
            ("2_nucleus/contour_maps.tif", "Contour maps"),
            ("2_nucleus/foreground_masks.tif", "Foreground masks"),
            ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
        ])
        db_gen_lay.addWidget(self.db_gen_input_files)

        self.db_gen_min_area_spin = QSpinBox()
        self.db_gen_min_area_spin.setRange(0, 1_000_000)
        self.db_gen_min_area_spin.setValue(300)

        self.db_gen_max_area_spin = QSpinBox()
        self.db_gen_max_area_spin.setRange(0, 10_000_000)
        self.db_gen_max_area_spin.setValue(100_000)

        self.db_gen_fg_thr_spin = QDoubleSpinBox()
        self.db_gen_fg_thr_spin.setRange(-5.0, 1.0)
        self.db_gen_fg_thr_spin.setValue(0.5)
        self.db_gen_fg_thr_spin.setDecimals(2)
        self.db_gen_fg_thr_spin.setSingleStep(0.05)
        self.db_gen_fg_thr_spin.setToolTip(
            "Pixel-level foreground threshold for ultrack segmentation (threshold in segmentation_config)"
        )

        self.db_gen_min_frontier_spin = QDoubleSpinBox()
        self.db_gen_min_frontier_spin.setRange(0.0, 1.0)
        self.db_gen_min_frontier_spin.setValue(0.0)
        self.db_gen_min_frontier_spin.setDecimals(3)
        self.db_gen_min_frontier_spin.setSingleStep(0.01)
        self.db_gen_min_frontier_spin.setToolTip(
            "Minimum boundary fraction to keep a candidate (min_frontier in segmentation_config)"
        )

        self.db_gen_ws_hierarchy_combo = QComboBox()
        self.db_gen_ws_hierarchy_combo.addItems(["area", "dynamics", "volume"])

        self.db_gen_n_workers_spin = QSpinBox()
        self.db_gen_n_workers_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.db_gen_n_workers_spin.setValue(1)
        self.db_gen_n_workers_spin.setToolTip("Parallel workers for segmentation")

        self.db_gen_max_dist_spin = QDoubleSpinBox()
        self.db_gen_max_dist_spin.setRange(0.0, 500.0)
        self.db_gen_max_dist_spin.setValue(15.0)
        self.db_gen_max_dist_spin.setDecimals(1)

        self.db_gen_max_neighbors_spin = QSpinBox()
        self.db_gen_max_neighbors_spin.setRange(1, 50)
        self.db_gen_max_neighbors_spin.setValue(5)

        self.db_gen_linking_mode_combo = QComboBox()
        self.db_gen_linking_mode_combo.addItems(["default", "iou"])

        self.db_gen_iou_weight_spin = QDoubleSpinBox()
        self.db_gen_iou_weight_spin.setRange(0.0, 1.0)
        self.db_gen_iou_weight_spin.setValue(1.0)
        self.db_gen_iou_weight_spin.setDecimals(2)
        self.db_gen_iou_weight_spin.setEnabled(False)

        self.db_gen_quality_weight_spin = QDoubleSpinBox()
        self.db_gen_quality_weight_spin.setRange(0.0, 10.0)
        self.db_gen_quality_weight_spin.setValue(1.0)
        self.db_gen_quality_weight_spin.setDecimals(2)
        self.db_gen_quality_weight_spin.setSingleStep(0.05)
        self.db_gen_quality_weight_spin.setToolTip(
            "Weight applied to signal-based segmentation quality before storing node_prob"
        )

        self.db_gen_quality_exp_spin = QDoubleSpinBox()
        self.db_gen_quality_exp_spin.setRange(0.1, 50.0)
        self.db_gen_quality_exp_spin.setValue(8.0)
        self.db_gen_quality_exp_spin.setDecimals(2)
        self.db_gen_quality_exp_spin.setToolTip(
            "Raises signal-based quality before storing as node_prob"
        )

        self.db_gen_circularity_weight_spin = QDoubleSpinBox()
        self.db_gen_circularity_weight_spin.setRange(0.0, 10.0)
        self.db_gen_circularity_weight_spin.setValue(0.25)
        self.db_gen_circularity_weight_spin.setDecimals(2)
        self.db_gen_circularity_weight_spin.setSingleStep(0.05)
        self.db_gen_circularity_weight_spin.setToolTip(
            "Weight applied to shape circularity before storing node_prob"
        )

        self.db_gen_power_spin = QDoubleSpinBox()
        self.db_gen_power_spin.setRange(0.1, 20.0)
        self.db_gen_power_spin.setValue(4.0)
        self.db_gen_power_spin.setDecimals(2)
        self.db_gen_power_spin.setToolTip(
            "Deprecated duplicate of the solver power control; solver transform for stored weights"
        )
        self.db_gen_power_spin.setVisible(False)

        self.ultrack_seed_weight_spin = QDoubleSpinBox()
        self.ultrack_seed_weight_spin.setRange(0.0, 10.0)
        self.ultrack_seed_weight_spin.setValue(0.5)
        self.ultrack_seed_weight_spin.setSingleStep(0.1)
        self.ultrack_seed_weight_spin.setDecimals(2)
        self.ultrack_seed_weight_spin.setToolTip(
            "Additive reward for candidates similar to nearby validated cells. "
            "Zero disables the seed-local bonus."
        )

        self.ultrack_seed_space_spin = QDoubleSpinBox()
        self.ultrack_seed_space_spin.setRange(1.0, 500.0)
        self.ultrack_seed_space_spin.setValue(25.0)
        self.ultrack_seed_space_spin.setSingleStep(5.0)
        self.ultrack_seed_space_spin.setDecimals(1)
        self.ultrack_seed_space_spin.setToolTip(
            "Spatial decay scale for seed proximity. Larger values let validated cells influence candidates farther away."
        )

        self.ultrack_seed_time_spin = QDoubleSpinBox()
        self.ultrack_seed_time_spin.setRange(0.1, 50.0)
        self.ultrack_seed_time_spin.setValue(2.0)
        self.ultrack_seed_time_spin.setSingleStep(0.5)
        self.ultrack_seed_time_spin.setDecimals(1)
        self.ultrack_seed_time_spin.setToolTip(
            "Temporal decay scale in frames. Larger values let validated cells influence more distant frames within the seed window."
        )

        self.ultrack_seed_window_spin = QSpinBox()
        self.ultrack_seed_window_spin.setRange(0, 100)
        self.ultrack_seed_window_spin.setValue(5)
        self.ultrack_seed_window_spin.setToolTip(
            "Maximum frame distance from a validated cell used for seed affinity."
        )

        db_candidate_grid = block_grid(horizontal_spacing=12)
        db_candidate_grid.setContentsMargins(0, 0, 0, 0)
        add_block_pair_row(db_candidate_grid, 0, "Min Area (px):", _compact(self.db_gen_min_area_spin), "Max Area (px):", _compact(self.db_gen_max_area_spin))
        add_block_pair_row(db_candidate_grid, 1, "FG Threshold:", _compact(self.db_gen_fg_thr_spin), "Min Frontier:", _compact(self.db_gen_min_frontier_spin))
        add_block_pair_row(db_candidate_grid, 2, "WS Hierarchy:", self.db_gen_ws_hierarchy_combo, "N Workers:", _compact(self.db_gen_n_workers_spin))
        db_gen_lay.addWidget(muted_label(QLabel("Candidate extraction")))
        db_gen_lay.addLayout(db_candidate_grid)

        db_linking_grid = block_grid(horizontal_spacing=12)
        db_linking_grid.setContentsMargins(0, 0, 0, 0)
        add_block_pair_row(db_linking_grid, 0, "Max Distance (px):", _compact(self.db_gen_max_dist_spin), "Max Neighbors:", _compact(self.db_gen_max_neighbors_spin))
        add_block_pair_row(db_linking_grid, 1, "Linking Mode:", self.db_gen_linking_mode_combo, "IoU Weight:", _compact(self.db_gen_iou_weight_spin))
        db_gen_lay.addWidget(muted_label(QLabel("Candidate linking")))
        db_gen_lay.addLayout(db_linking_grid)

        db_scoring_grid = block_grid(horizontal_spacing=12)
        db_scoring_grid.setContentsMargins(0, 0, 0, 0)
        add_block_pair_row(db_scoring_grid, 0, "Quality Weight:", _compact(self.db_gen_quality_weight_spin), "Quality Exp:", _compact(self.db_gen_quality_exp_spin))
        add_block_pair_row(db_scoring_grid, 1, "Circularity Weight:", _compact(self.db_gen_circularity_weight_spin), "", QWidget())
        db_gen_lay.addWidget(muted_label(QLabel("Node scoring")))
        db_gen_lay.addLayout(db_scoring_grid)

        self.db_gen_use_validated_check = QCheckBox("Use validated corrections")
        db_gen_validated_grid = block_grid(horizontal_spacing=12)
        add_block_checkbox_row(db_gen_validated_grid, 0, self.db_gen_use_validated_check)
        db_gen_lay.addLayout(db_gen_validated_grid)

        db_seed_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(
            db_seed_grid,
            0,
            "Seed Weight:",
            _compact(self.ultrack_seed_weight_spin, 80),
            "Seed Space (px):",
            _compact(self.ultrack_seed_space_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            db_seed_grid,
            1,
            "Seed Time:",
            _compact(self.ultrack_seed_time_spin, 80),
            "Seed Window:",
            _compact(self.ultrack_seed_window_spin, 80),
            field_width=80,
        )
        db_gen_lay.addWidget(muted_label(QLabel("Validated seed prior")))
        db_gen_lay.addLayout(db_seed_grid)

        db_gen_run_row = block_grid(horizontal_spacing=12)
        self.run_db_gen_btn = QPushButton("Run DB Generation")
        self.run_db_gen_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.db_gen_terminal_btn = QPushButton("Run in Terminal")
        self.db_gen_terminal_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        add_block_button_row(db_gen_run_row, 0, self.run_db_gen_btn, self.db_gen_terminal_btn)
        db_gen_lay.addLayout(db_gen_run_row)

        self.db_gen_status_lbl = _stage_status()
        db_gen_lay.addWidget(self.db_gen_status_lbl)

        self.db_gen_progress_bar = _stage_progress()
        db_gen_lay.addWidget(self.db_gen_progress_bar)

        self.db_gen_output_files = _stage_files("Outputs", [
            ("2_nucleus/ultrack_workdir/data.db", "Ultrack database"),
        ])
        db_gen_lay.addWidget(self.db_gen_output_files)

        self.db_gen_section = CollapsibleSection(
            "2. Ultrack Database Generation", _db_gen_inner, expanded=False
        )
        layout.addWidget(self.db_gen_section)

        # ── Optional Ultrack Database Browser ──────────────────────────────
        _ultrack_db_browser_inner = QWidget()
        ultrack_db_browser_lay = QVBoxLayout(_ultrack_db_browser_inner)
        ultrack_db_browser_lay.setContentsMargins(0, 0, 0, 0)
        ultrack_db_browser_lay.setSpacing(4)

        from qtpy.QtGui import QIcon
        from qtpy.QtCore import Qt as _Qt
        self.ultrack_db_info_lbl = QLabel("—")
        self.ultrack_db_info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ultrack_db_info_lbl.setWordWrap(True)
        self.ultrack_db_info_lbl.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        ultrack_db_browser_lay.addWidget(self.ultrack_db_info_lbl)

        ultrack_db_grid = block_grid(horizontal_spacing=12)
        ultrack_db_grid.setContentsMargins(0, 0, 0, 0)
        self.ultrack_db_hierarchy_slider = QSlider(_Qt.Horizontal)
        self.ultrack_db_hierarchy_slider.setRange(0, 100)
        self.ultrack_db_hierarchy_slider.setValue(50)
        self.ultrack_db_hierarchy_slider.setToolTip(
            "Hierarchy cut level: 0 = most split, 1 = most merged"
        )
        self.ultrack_db_height_lbl = QLabel("0.50")
        self.ultrack_db_height_lbl.setFixedWidth(48)
        self._ultrack_db_slider_row = QWidget()
        _slider_lay = QHBoxLayout(self._ultrack_db_slider_row)
        _slider_lay.setContentsMargins(0, 0, 0, 0)
        _slider_lay.addWidget(self.ultrack_db_hierarchy_slider)
        _slider_lay.addWidget(self.ultrack_db_height_lbl)
        ultrack_db_grid.addWidget(self._ultrack_db_slider_row, 0, 0, 1, 4)
        self._ultrack_db_slider_row.setVisible(True)

        ultrack_db_browser_lay.addLayout(ultrack_db_grid)

        _db_btn_row = QWidget()
        _db_btn_lay = QHBoxLayout(_db_btn_row)
        _db_btn_lay.setContentsMargins(0, 0, 0, 0)
        _db_btn_lay.setSpacing(4)
        self.ultrack_db_active_btn = QPushButton("Activate")
        self.ultrack_db_active_btn.setCheckable(True)
        self.ultrack_db_active_btn.setChecked(False)
        self.ultrack_db_active_btn.setToolTip("Load contour maps and foreground masks into viewer and enable DB preview")
        self.ultrack_db_refresh_btn = QPushButton()
        self.ultrack_db_refresh_btn.setToolTip("Refresh Ultrack database browser")
        self.ultrack_db_refresh_btn.setIcon(QIcon.fromTheme("view-refresh"))
        self.ultrack_db_refresh_btn.setEnabled(False)
        _db_btn_lay.addWidget(self.ultrack_db_active_btn)
        _db_btn_lay.addWidget(self.ultrack_db_refresh_btn)
        ultrack_db_browser_lay.addWidget(_db_btn_row)
        self.ultrack_db_hierarchy_slider.setEnabled(False)
        self.ultrack_db_prob_alpha_check = QCheckBox("Node prob transparency")
        self.ultrack_db_prob_alpha_check.setToolTip("Modulate label opacity by node probability (higher quality = more opaque)")
        self.ultrack_db_prob_alpha_check.setEnabled(False)
        self.ultrack_db_connected_focus_check = QCheckBox("Connected focus")
        self.ultrack_db_connected_focus_check.setToolTip(
            "Focus the DB preview on a selected node and its temporal neighbors"
        )
        self.ultrack_db_connected_focus_check.setEnabled(False)
        self.ultrack_db_edge_alpha_check = QCheckBox("Edge weight transparency")
        self.ultrack_db_edge_alpha_check.setToolTip(
            "Modulate connected-neighbor opacity by link weight"
        )
        self.ultrack_db_edge_alpha_check.setEnabled(False)
        self.ultrack_db_show_validated_check = QCheckBox("Show validated nodes")
        self.ultrack_db_show_validated_check.setChecked(True)
        self.ultrack_db_show_validated_check.setEnabled(False)
        self.ultrack_db_show_fake_check = QCheckBox("Show fake nodes")
        self.ultrack_db_show_fake_check.setChecked(False)
        self.ultrack_db_show_fake_check.setEnabled(False)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_prob_alpha_check)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_connected_focus_check)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_edge_alpha_check)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_show_validated_check)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_show_fake_check)

        self.ultrack_db_section_status_lbl = QLabel("")
        self.ultrack_db_section_status_lbl.setWordWrap(True)
        self.ultrack_db_section_status_lbl.setVisible(False)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_section_status_lbl)

        self.ultrack_db_browser_section = CollapsibleSection(
            "Ultrack Database Browser", _ultrack_db_browser_inner, expanded=False
        )

        # ── 4. Ultrack Tracking ───────────────────────────────────────────

        _ultrack_inner = QWidget()
        ultrack_lay = QVBoxLayout(_ultrack_inner)
        ultrack_lay.setContentsMargins(0, 0, 0, 0)
        ultrack_lay.setSpacing(4)
        ultrack_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.ultrack_input_files = _stage_files("Inputs", [
            ("2_nucleus/ultrack_workdir/data.db", "Ultrack database"),
        ])
        ultrack_lay.addWidget(self.ultrack_input_files)

        self.ultrack_min_area_spin = QSpinBox()
        self.ultrack_min_area_spin.setRange(0, 100000)
        self.ultrack_min_area_spin.setValue(300)
        self.ultrack_min_area_spin.setSingleStep(50)

        self.ultrack_max_partitions_spin = QSpinBox()
        self.ultrack_max_partitions_spin.setRange(0, 1000)
        self.ultrack_max_partitions_spin.setValue(30)
        self.ultrack_max_partitions_spin.setToolTip("0 = use all partitions")

        self.ultrack_n_frames_spin = QSpinBox()
        self.ultrack_n_frames_spin.setRange(0, 10000)
        self.ultrack_n_frames_spin.setValue(0)
        self.ultrack_n_frames_spin.setToolTip("0 = process all frames")

        self.ultrack_linking_mode_combo = QComboBox()
        self.ultrack_linking_mode_combo.addItems(["default", "iou"])

        self.ultrack_max_dist_spin = QDoubleSpinBox()
        self.ultrack_max_dist_spin.setRange(0.0, 500.0)
        self.ultrack_max_dist_spin.setValue(15.0)
        self.ultrack_max_dist_spin.setSingleStep(1.0)
        self.ultrack_max_dist_spin.setDecimals(1)

        self.ultrack_iou_weight_spin = QDoubleSpinBox()
        self.ultrack_iou_weight_spin.setRange(0.0, 1.0)
        self.ultrack_iou_weight_spin.setValue(1.0)
        self.ultrack_iou_weight_spin.setSingleStep(0.05)
        self.ultrack_iou_weight_spin.setDecimals(2)

        self.ultrack_appear_spin = QDoubleSpinBox()
        self.ultrack_appear_spin.setRange(-10.0, 0.0)
        self.ultrack_appear_spin.setValue(-0.1)
        self.ultrack_appear_spin.setSingleStep(0.05)
        self.ultrack_appear_spin.setDecimals(3)

        self.ultrack_disappear_spin = QDoubleSpinBox()
        self.ultrack_disappear_spin.setRange(-10.0, 0.0)
        self.ultrack_disappear_spin.setValue(-0.1)
        self.ultrack_disappear_spin.setSingleStep(0.05)
        self.ultrack_disappear_spin.setDecimals(3)

        self.ultrack_division_spin = QDoubleSpinBox()
        self.ultrack_division_spin.setRange(-10.0, 0.0)
        self.ultrack_division_spin.setValue(-0.001)
        self.ultrack_division_spin.setSingleStep(0.05)
        self.ultrack_division_spin.setDecimals(3)
        self.ultrack_division_spin.setToolTip(
            "ILP penalty for cell division events. More negative = fewer divisions allowed."
        )

        self.ultrack_max_neighbors_spin = QSpinBox()
        self.ultrack_max_neighbors_spin.setRange(1, 50)
        self.ultrack_max_neighbors_spin.setValue(5)
        self.ultrack_max_neighbors_spin.setToolTip(
            "Maximum number of candidate predecessor nodes considered during linking."
        )

        self.ultrack_power_spin = QDoubleSpinBox()
        self.ultrack_power_spin.setRange(0.1, 20.0)
        self.ultrack_power_spin.setValue(4.0)
        self.ultrack_power_spin.setSingleStep(0.5)
        self.ultrack_power_spin.setDecimals(2)
        self.ultrack_power_spin.setToolTip(
            "Ultrack's solver transform for node_prob and link weights. "
            "With link_function=power, stored weights are raised to this power during solving."
        )

        self.ultrack_quality_exp_spin = QDoubleSpinBox()
        self.ultrack_quality_exp_spin.setRange(0.1, 50.0)
        self.ultrack_quality_exp_spin.setValue(8.0)
        self.ultrack_quality_exp_spin.setSingleStep(0.5)
        self.ultrack_quality_exp_spin.setDecimals(2)
        self.ultrack_quality_exp_spin.setToolTip(
            "Raises the signal-based segmentation quality before storing it as node_prob. "
            "Higher values favor high-confidence whole-object candidates over fragments."
        )

        self.ultrack_solver_lbl = QLabel("—")
        track_scope_grid = block_grid(horizontal_spacing=12)
        track_scope_grid.setContentsMargins(0, 0, 0, 0)
        add_block_pair_row(
            track_scope_grid,
            0,
            "Max Partitions/frame:",
            _compact(self.ultrack_max_partitions_spin, 80),
            "First N frames:",
            _compact(self.ultrack_n_frames_spin, 80),
            field_width=80,
        )
        ultrack_lay.addWidget(muted_label(QLabel("Track scope")))
        ultrack_lay.addLayout(track_scope_grid)

        event_penalty_grid = block_grid(horizontal_spacing=12)
        event_penalty_grid.setContentsMargins(0, 0, 0, 0)
        add_block_pair_row(
            event_penalty_grid,
            0,
            "Appear Penalty:",
            _compact(self.ultrack_appear_spin, 80),
            "Disappear Penalty:",
            _compact(self.ultrack_disappear_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            event_penalty_grid,
            1,
            "Division Penalty:",
            _compact(self.ultrack_division_spin, 80),
            field_width=80,
        )
        ultrack_lay.addWidget(muted_label(QLabel("Event penalties")))
        ultrack_lay.addLayout(event_penalty_grid)

        solver_grid = block_grid(horizontal_spacing=12)
        solver_grid.setContentsMargins(0, 0, 0, 0)
        add_block_pair_row(
            solver_grid,
            0,
            "Ultrack Power:",
            _compact(self.ultrack_power_spin, 80),
            "Solver:",
            self.ultrack_solver_lbl,
            field_width=None,
        )
        ultrack_lay.addWidget(muted_label(QLabel("Solver scoring")))
        ultrack_lay.addLayout(solver_grid)

        ultrack_run_row = block_grid(horizontal_spacing=12)
        self.run_ultrack_btn = QPushButton("Run Ultrack Tracking")
        self.ultrack_terminal_btn = QPushButton("Run in Terminal")
        add_block_button_row(ultrack_run_row, 0, self.run_ultrack_btn, self.ultrack_terminal_btn)
        ultrack_lay.addLayout(ultrack_run_row)

        self.ultrack_status_lbl = _stage_status()
        ultrack_lay.addWidget(self.ultrack_status_lbl)

        self.ultrack_progress_bar = _stage_progress()
        ultrack_lay.addWidget(self.ultrack_progress_bar)

        self.ultrack_output_files = _stage_files("Outputs", [
            ("2_nucleus/tracked_labels.tif", "Tracked labels"),
        ])
        ultrack_lay.addWidget(self.ultrack_output_files)

        ultrack_attrib = QLabel(
            "Ultrack tracking is powered by the "
            '<a href="https://github.com/royerlab/ultrack">Ultrack</a> project.'
        )
        ultrack_attrib.setOpenExternalLinks(True)
        ultrack_attrib.setWordWrap(True)
        muted_label(ultrack_attrib, size_pt=9)
        ultrack_lay.addWidget(ultrack_attrib)

        self.ultrack_section = CollapsibleSection(
            "4. Ultrack Tracking", _ultrack_inner, expanded=False
        )
        layout.addWidget(self.ultrack_section)

        _corr_inner = QWidget()
        _corr_inner_lay = QVBoxLayout(_corr_inner)
        _corr_inner_lay.setContentsMargins(0, 0, 0, 0)
        _corr_inner_lay.setSpacing(4)

        extend_row = block_grid(horizontal_spacing=12)
        self.extend_back_btn = QPushButton("◀ Extend (A)")
        self.extend_fwd_btn = QPushButton("Extend (D) ▶")
        add_block_button_row(extend_row, 0, self.extend_back_btn, self.extend_fwd_btn)
        _corr_inner_lay.addLayout(extend_row)

        retrack_row = block_grid(horizontal_spacing=12)
        self.retrack_back_btn = QPushButton("◀ Retrack (Q)")
        self.retrack_fwd_btn = QPushButton("Retrack (E) ▶")
        add_block_button_row(retrack_row, 0, self.retrack_back_btn, self.retrack_fwd_btn)
        _corr_inner_lay.addLayout(retrack_row)

        save_load_row = block_grid(horizontal_spacing=12)
        self.save_tracked_btn = QPushButton("Save Tracked Labels")
        self.load_tracked_btn = QPushButton("Load Tracked Labels")
        add_block_button_row(save_load_row, 0, self.save_tracked_btn, self.load_tracked_btn)
        _corr_inner_lay.addLayout(save_load_row)

        reassign_row = block_grid(horizontal_spacing=12)
        self.reassign_ids_btn = QPushButton("Reassign IDs")
        add_block_button_row(reassign_row, 0, self.reassign_ids_btn)
        _corr_inner_lay.addLayout(reassign_row)

        extend_params_inner = QWidget()
        extend_params_lay = QVBoxLayout(extend_params_inner)
        extend_params_lay.setContentsMargins(0, 0, 0, 0)
        extend_params_lay.setSpacing(4)
        extend_params_form = block_grid(horizontal_spacing=12)
        self.extend_max_dist_spin = QDoubleSpinBox()
        self.extend_max_dist_spin.setRange(0.0, 500.0)
        self.extend_max_dist_spin.setValue(40.0)
        self.extend_max_dist_spin.setSingleStep(1.0)
        self.extend_max_dist_spin.setDecimals(1)
        self.extend_area_weight_spin = QDoubleSpinBox()
        self.extend_area_weight_spin.setRange(0.0, 10.0)
        self.extend_area_weight_spin.setValue(1.0)
        self.extend_area_weight_spin.setSingleStep(0.1)
        self.extend_area_weight_spin.setDecimals(2)
        self.extend_iou_weight_spin = QDoubleSpinBox()
        self.extend_iou_weight_spin.setRange(0.0, 10.0)
        self.extend_iou_weight_spin.setValue(1.0)
        self.extend_iou_weight_spin.setSingleStep(0.1)
        self.extend_iou_weight_spin.setDecimals(2)
        self.extend_distance_weight_spin = QDoubleSpinBox()
        self.extend_distance_weight_spin.setRange(0.0, 10.0)
        self.extend_distance_weight_spin.setValue(0.25)
        self.extend_distance_weight_spin.setSingleStep(0.05)
        self.extend_distance_weight_spin.setDecimals(2)
        self.extend_overlap_penalty_spin = QDoubleSpinBox()
        self.extend_overlap_penalty_spin.setRange(0.0, 10.0)
        self.extend_overlap_penalty_spin.setValue(1.0)
        self.extend_overlap_penalty_spin.setSingleStep(0.1)
        self.extend_overlap_penalty_spin.setDecimals(2)
        self.extend_greedy_overwrite_check = QCheckBox("Greedy overwrite")
        add_block_pair_row(
            extend_params_form,
            0,
            "Max Distance (px):",
            _compact(self.extend_max_dist_spin, 80),
            "Area Weight:",
            _compact(self.extend_area_weight_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            extend_params_form,
            1,
            "IoU Weight:",
            _compact(self.extend_iou_weight_spin, 80),
            "Distance Weight:",
            _compact(self.extend_distance_weight_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            extend_params_form,
            2,
            "Overlap Penalty:",
            _compact(self.extend_overlap_penalty_spin, 80),
            field_width=80,
        )
        add_block_checkbox_row(extend_params_form, 3, self.extend_greedy_overwrite_check)
        extend_params_lay.addLayout(extend_params_form)
        self.extend_params_section = CollapsibleSection(
            "Extend Parameters", extend_params_inner, expanded=False
        )
        _corr_inner_lay.addWidget(self.extend_params_section)

        retrack_params_inner = QWidget()
        retrack_params_lay = QVBoxLayout(retrack_params_inner)
        retrack_params_lay.setContentsMargins(0, 0, 0, 0)
        retrack_params_lay.setSpacing(4)
        retrack_params_form = block_grid(horizontal_spacing=12)
        self.retrack_max_dist_spin = QDoubleSpinBox()
        self.retrack_max_dist_spin.setRange(0.0, 500.0)
        self.retrack_max_dist_spin.setValue(20.0)
        self.retrack_max_dist_spin.setSingleStep(1.0)
        self.retrack_max_dist_spin.setDecimals(1)
        add_block_pair_row(
            retrack_params_form,
            0,
            "Max Distance (px):",
            _compact(self.retrack_max_dist_spin, 80),
            field_width=80,
        )
        retrack_params_lay.addLayout(retrack_params_form)
        self.retrack_params_section = CollapsibleSection(
            "Retrack Parameters", retrack_params_inner, expanded=False
        )
        _corr_inner_lay.addWidget(self.retrack_params_section)

        self.validation_counter_lbl = QLabel("")
        self.validation_counter_lbl.setWordWrap(True)
        _corr_inner_lay.addWidget(self.validation_counter_lbl)

        self.remove_unvalidated_btn = QPushButton("Remove Unvalidated Labels")
        self.remove_unvalidated_btn.setToolTip(
            "Remove nucleus label pixels that are not marked validated for their frame."
        )
        action_button(self.remove_unvalidated_btn, expand=True)
        danger_button(self.remove_unvalidated_btn)
        _corr_inner_lay.addWidget(self.remove_unvalidated_btn)

        self.correction_status_lbl = QLabel("")
        self.correction_status_lbl.setWordWrap(True)
        self.correction_status_lbl.setVisible(False)
        _corr_inner_lay.addWidget(self.correction_status_lbl)

        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
        )
        self.correction_widget.set_edit_callback(self._on_cells_edited)
        _corr_inner_lay.addWidget(self.correction_widget)
        self.correction_shortcuts_section = CollapsibleSection(
            "Correction Shortcuts",
            self.correction_widget.build_shortcuts_widget(),
            expanded=False,
        )
        _corr_inner_lay.addWidget(self.correction_shortcuts_section)

        self.correction_section = CollapsibleSection(
            "5. Correction", _corr_inner, expanded=False
        )
        layout.addWidget(self.correction_section)
        layout.addWidget(self.ultrack_db_browser_section)

    # ──────────────────────────────────────────────────────────────────────────
    # Signal wiring
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.build_btn.clicked.connect(self._on_build_contour_maps)
        self.preview_contour_btn.clicked.connect(self._on_preview_contour_maps)
        self.contour_terminal_btn.clicked.connect(self._on_run_contour_terminal)
        self.preview_contour_filter_btn.clicked.connect(self._on_preview_contour_filter)
        self.run_contour_filter_btn.clicked.connect(self._on_run_contour_filter)
        self.cancel_build_btn.clicked.connect(self._on_cancel_build)
        self.run_db_gen_btn.clicked.connect(self._on_run_db_generation)
        self.db_gen_terminal_btn.clicked.connect(self._on_db_gen_terminal)
        self.db_gen_linking_mode_combo.currentTextChanged.connect(self._on_db_gen_mode_changed)
        self.db_gen_use_validated_check.toggled.connect(self._set_resolve_prior_controls_enabled)
        self.ultrack_db_active_btn.toggled.connect(self._on_ultrack_db_activate)
        self.ultrack_db_refresh_btn.clicked.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_hierarchy_slider.valueChanged.connect(self._on_ultrack_db_slider_changed)
        self.ultrack_db_prob_alpha_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_connected_focus_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_edge_alpha_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_show_validated_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_show_fake_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.run_ultrack_btn.clicked.connect(self._on_run_ultrack)
        self.ultrack_terminal_btn.clicked.connect(self._on_ultrack_terminal)
        self.save_tracked_btn.clicked.connect(self._on_save_tracked)
        self.load_tracked_btn.clicked.connect(self._on_load_tracked)
        self.reassign_ids_btn.clicked.connect(self._on_reassign_ids)
        self.ultrack_linking_mode_combo.currentTextChanged.connect(self._on_ultrack_mode_changed)
        self.retrack_back_btn.clicked.connect(self._on_retrack_backward)
        self.retrack_fwd_btn.clicked.connect(self._on_retrack_forward)
        self.extend_back_btn.clicked.connect(self._on_extend_backward)
        self.extend_fwd_btn.clicked.connect(self._on_extend_forward)
        self.remove_unvalidated_btn.clicked.connect(self._on_remove_unvalidated_labels)
        self.viewer.dims.events.current_step.connect(self._on_dims_step_changed)
        self.viewer.bind_key("V", self._kb_toggle_cell_validation, overwrite=True)
        self._install_correction_shortcuts()
        self.correction_widget._activate_btn.toggled.connect(self._on_correction_mode_toggled)
        # Set initial state for solver label and IoU weight enablement
        solver = _select_solver()
        solver_display = "Gurobi (licensed)" if solver == "GUROBI" else "CBC"
        self.ultrack_solver_lbl.setText(solver_display)
        self._on_ultrack_mode_changed(self.ultrack_linking_mode_combo.currentText())
        self._set_resolve_prior_controls_enabled()

    # ──────────────────────────────────────────────────────────────────────────
    # Public refresh
    # ──────────────────────────────────────────────────────────────────────────

    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._refresh_stage_files(pos_dir)
        if pos_dir is None:
            self.correction_widget.deactivate()
            return
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _refresh_stage_files(self, pos_dir: Path | None = None) -> None:
        if pos_dir is None:
            pos_dir = self._pos_dir
        for files_widget in (
            self.contour_input_files,
            self.contour_output_files,
            self.db_gen_input_files,
            self.db_gen_output_files,
            self.ultrack_input_files,
            self.ultrack_output_files,
        ):
            files_widget.refresh(pos_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Path helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _tracked_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    def _prob_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif" if self._pos_dir else None

    def _dp_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif" if self._pos_dir else None

    def _contour_maps_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "contour_maps.tif" if self._pos_dir else None

    def _foreground_scores_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "foreground_scores.tif" if self._pos_dir else None

    def _cell_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "cell_zavg.tif" if self._pos_dir else None

    def _nucleus_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "nucleus_zavg.tif" if self._pos_dir else None

    def _ultrack_workdir(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "ultrack_workdir" if self._pos_dir else None

    def _ultrack_db_path(self) -> Path | None:
        workdir = self._ultrack_workdir()
        return workdir / "data.db" if workdir else None

    def _foreground_masks_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "foreground_masks.tif" if self._pos_dir else None

    def _nucleus_prob_zavg_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif" if self._pos_dir else None

    # ── DB Generation section ─────────────────────────────────────────────────

    def _db_gen_config_from_controls(self) -> UltrackConfig:
        return UltrackConfig(
            seg_min_area=self.db_gen_min_area_spin.value(),
            seg_max_area=self.db_gen_max_area_spin.value(),
            seg_foreground_threshold=self.db_gen_fg_thr_spin.value(),
            seg_min_frontier=self.db_gen_min_frontier_spin.value(),
            seg_ws_hierarchy=self.db_gen_ws_hierarchy_combo.currentText(),
            seg_n_workers=self.db_gen_n_workers_spin.value(),
            max_distance=self.db_gen_max_dist_spin.value(),
            max_neighbors=self.db_gen_max_neighbors_spin.value(),
            linking_mode=self.db_gen_linking_mode_combo.currentText(),
            iou_weight=self.db_gen_iou_weight_spin.value(),
            quality_weight=self.db_gen_quality_weight_spin.value(),
            quality_exponent=self.db_gen_quality_exp_spin.value(),
            circularity_weight=self.db_gen_circularity_weight_spin.value(),
            link_n_workers=self.db_gen_n_workers_spin.value(),
            seed_weight=self.ultrack_seed_weight_spin.value(),
            seed_sigma_space=self.ultrack_seed_space_spin.value(),
            seed_tau_time=self.ultrack_seed_time_spin.value(),
            seed_max_dt=self.ultrack_seed_window_spin.value(),
        )

    def _on_run_db_generation(self) -> None:
        if self._pos_dir is None:
            self._set_db_gen_status("No project open.")
            return
        contour_path = self._contour_maps_path()
        fg_path = self._foreground_masks_path()
        nuc_zavg_path = self._nucleus_prob_zavg_path()
        if contour_path is None or not contour_path.exists():
            self._set_db_gen_status("Missing: contour_maps.tif — run Contour Maps first.")
            return
        if fg_path is None or not fg_path.exists():
            self._set_db_gen_status(
                "Missing: foreground_masks.tif (foreground mask) — run Contour Maps first."
            )
            return
        if nuc_zavg_path is None or not nuc_zavg_path.exists():
            self._set_db_gen_status("Missing: nucleus_prob_zavg.tif — run Cellpose first.")
            return
        if _ultrack_segment is None:
            self._set_db_gen_status("ultrack not installed — activate the cellflow conda environment.")
            return

        cfg = self._db_gen_config_from_controls()
        working_dir = self._ultrack_workdir()
        pos_dir = self._pos_dir
        use_validated = self.db_gen_use_validated_check.isChecked()
        validated_tracks: dict[int, set[int]] | None = None
        tracked_labels: np.ndarray | None = None
        if use_validated:
            validated_tracks = read_validated_tracks(pos_dir)
            if not validated_tracks:
                self._set_db_gen_status("No validated tracks found — validate some cells first (press V).")
                return
            if _TRACKED_LAYER not in self.viewer.layers:
                self._set_db_gen_status("No tracked layer loaded for validated DB generation.")
                return
            tracked_labels = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)

        self.db_gen_progress_bar.setRange(0, 0)
        self.db_gen_progress_bar.setVisible(True)
        self._set_db_gen_status("Starting DB generation…")
        self.run_db_gen_btn.setEnabled(False)
        self.db_gen_terminal_btn.setEnabled(False)

        @thread_worker(connect={
            "yielded": self._on_db_gen_progress,
            "returned": self._on_db_gen_done,
            "errored": self._on_db_gen_worker_error,
        })
        def _worker():
            import queue as _queue
            import threading

            msg_queue: _queue.SimpleQueue = _queue.SimpleQueue()
            result_holder: list = []
            exc_holder: list = []

            def _progress(msg: str) -> None:
                msg_queue.put(msg)

            def _run() -> None:
                try:
                    result_holder.append(
                        build_ultrack_database(
                            contour_maps_path=contour_path,
                            foreground_masks_path=fg_path,
                            nucleus_prob_zavg_path=nuc_zavg_path,
                            working_dir=working_dir,
                            cfg=cfg,
                            validated_tracks=validated_tracks,
                            tracked_labels=tracked_labels,
                            use_validated=use_validated,
                            progress_cb=_progress,
                        )
                    )
                except Exception as e:
                    exc_holder.append(e)

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            while t.is_alive() or not msg_queue.empty():
                try:
                    yield msg_queue.get_nowait()
                except _queue.Empty:
                    t.join(timeout=0.05)
            if exc_holder:
                raise exc_holder[0]
            return pos_dir

        _worker()

    def _on_db_gen_terminal(self) -> None:
        import sys
        import tempfile

        if self._pos_dir is None:
            self._set_db_gen_status("No project open.")
            return
        contour_path = self._contour_maps_path()
        fg_path = self._foreground_masks_path()
        nuc_zavg_path = self._nucleus_prob_zavg_path()
        if contour_path is None or not contour_path.exists():
            self._set_db_gen_status("Missing: contour_maps.tif")
            return
        if fg_path is None or not fg_path.exists():
            self._set_db_gen_status("Missing: foreground_masks.tif (foreground mask) — run Contour Maps first.")
            return
        if nuc_zavg_path is None or not nuc_zavg_path.exists():
            self._set_db_gen_status("Missing: nucleus_prob_zavg.tif")
            return

        cfg = self._db_gen_config_from_controls()
        working_dir = self._ultrack_workdir()
        use_validated = self.db_gen_use_validated_check.isChecked()
        tracked_path = self._tracked_path()
        if use_validated:
            validated_tracks = read_validated_tracks(self._pos_dir)
            if not validated_tracks:
                self._set_db_gen_status("No validated tracks found — validate some cells first (press V).")
                return
            if tracked_path is None or not tracked_path.exists():
                self._set_db_gen_status("Tracked labels not found for validated DB generation.")
                return

        python_code = (
            "import pathlib, sys\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))\n"
            "from cellflow.database.tracked import read_full_tracked_stack\n"
            "from cellflow.database.validation import read_validated_tracks\n"
            "from cellflow.tracking_ultrack.config import TrackingConfig\n"
            "from cellflow.tracking_ultrack.db_build import build_ultrack_database\n"
            "\n"
            "if __name__ == '__main__':\n"
            f"    pos_dir = pathlib.Path({str(self._pos_dir)!r})\n"
            f"    contour_path = pathlib.Path({str(contour_path)!r})\n"
            f"    foreground_masks_path = pathlib.Path({str(fg_path)!r})\n"
            f"    nucleus_prob_zavg_path = pathlib.Path({str(nuc_zavg_path)!r})\n"
            f"    working_dir = pathlib.Path({str(working_dir)!r})\n"
            f"    tracked_path = pathlib.Path({str(tracked_path)!r})\n"
            f"    use_validated = {bool(use_validated)!r}\n"
            "    cfg = TrackingConfig(\n"
            f"        seg_min_area={cfg.seg_min_area},\n"
            f"        seg_max_area={cfg.seg_max_area},\n"
            f"        seg_foreground_threshold={cfg.seg_foreground_threshold},\n"
            f"        seg_min_frontier={cfg.seg_min_frontier},\n"
            f"        seg_ws_hierarchy={cfg.seg_ws_hierarchy!r},\n"
            f"        seg_n_workers={cfg.seg_n_workers},\n"
            f"        max_distance={cfg.max_distance},\n"
            f"        max_neighbors={cfg.max_neighbors},\n"
            f"        linking_mode={cfg.linking_mode!r},\n"
            f"        iou_weight={cfg.iou_weight},\n"
            f"        quality_weight={cfg.quality_weight},\n"
            f"        quality_exponent={cfg.quality_exponent},\n"
            f"        circularity_weight={cfg.circularity_weight},\n"
            f"        link_n_workers={cfg.link_n_workers},\n"
            f"        seed_weight={cfg.seed_weight},\n"
            f"        seed_sigma_space={cfg.seed_sigma_space},\n"
            f"        seed_tau_time={cfg.seed_tau_time},\n"
            f"        seed_max_dt={cfg.seed_max_dt},\n"
            "    )\n"
            "    validated_tracks = read_validated_tracks(pos_dir) if use_validated else None\n"
            "    tracked_labels = read_full_tracked_stack(tracked_path) if use_validated else None\n"
            "    report = build_ultrack_database(\n"
            "        contour_maps_path=contour_path,\n"
            "        foreground_masks_path=foreground_masks_path,\n"
            "        nucleus_prob_zavg_path=nucleus_prob_zavg_path,\n"
            "        working_dir=working_dir,\n"
            "        cfg=cfg,\n"
            "        validated_tracks=validated_tracks,\n"
            "        tracked_labels=tracked_labels,\n"
            "        use_validated=use_validated,\n"
            "        progress_cb=lambda msg: print(msg, flush=True),\n"
            "    )\n"
            "    print(f'Done. {report}', flush=True)\n"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix="cellflow_db_gen_", delete=False) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path)}"
        try:
            from cellflow.napari.utils import launch_in_terminal
            launch_in_terminal(cmd)
            self._set_db_gen_status("DB generation launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(cmd)
            self._set_db_gen_status("Copied DB generation command to clipboard.")

    def _on_db_gen_mode_changed(self, mode: str) -> None:
        self.db_gen_iou_weight_spin.setEnabled(mode == "iou")

    def _on_db_gen_progress(self, msg: str) -> None:
        self._set_db_gen_status(msg)

    def _on_db_gen_done(self, pos_dir: Path) -> None:
        self.db_gen_progress_bar.setVisible(False)
        self.run_db_gen_btn.setEnabled(True)
        self.db_gen_terminal_btn.setEnabled(True)
        self._set_db_gen_status("DB generation complete.")
        self._refresh_stage_files(pos_dir)
        self._refresh_ultrack_db_browser()

    def _on_db_gen_worker_error(self, exc: Exception) -> None:
        self.db_gen_progress_bar.setVisible(False)
        self.run_db_gen_btn.setEnabled(True)
        self.db_gen_terminal_btn.setEnabled(True)
        self._set_db_gen_status(f"Error: {exc}")
        logger.exception("DB generation worker error", exc_info=exc)

    def _set_db_gen_status(self, msg: str) -> None:
        self.db_gen_status_lbl.setText(msg)
        self.db_gen_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    # ── Ultrack DB Browser section ────────────────────────────────────────────

    def _set_ultrack_db_status(self, msg: str) -> None:
        self.ultrack_db_section_status_lbl.setText(msg)
        self.ultrack_db_section_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _on_ultrack_db_browser_param_changed(self, *_args) -> None:
        self._ultrack_db_preview_cache.clear()

    def _on_ultrack_db_slider_changed(self, value: int) -> None:
        if not self._ultrack_db_browser_active:
            return
        db_path = self._ultrack_db_path()
        if db_path is not None and db_path.exists():
            try:
                mtime_ns = db_path.stat().st_mtime_ns
                heights = self._query_distinct_heights(db_path, mtime_ns)
                index = min(max(int(value), 0), max(len(heights) - 1, 0))
                if heights:
                    self._set_ultrack_db_height_label(index, heights[index], len(heights))
                else:
                    self.ultrack_db_height_lbl.setText("—")
            except Exception:
                self.ultrack_db_height_lbl.setText(str(value))
        else:
            self.ultrack_db_height_lbl.setText(str(value))
        self._ultrack_db_preview_cache.clear()
        from qtpy.QtCore import QTimer
        QTimer.singleShot(150, self._refresh_ultrack_db_browser)

    def _on_ultrack_db_activate(self, checked: bool) -> None:
        self._ultrack_db_browser_active = checked
        self.ultrack_db_active_btn.setText("Deactivate" if checked else "Activate")
        self.ultrack_db_refresh_btn.setEnabled(checked)
        self.ultrack_db_hierarchy_slider.setEnabled(checked)
        self.ultrack_db_prob_alpha_check.setEnabled(checked)
        self.ultrack_db_connected_focus_check.setEnabled(checked)
        self.ultrack_db_edge_alpha_check.setEnabled(checked)
        self.ultrack_db_show_validated_check.setEnabled(checked)
        self.ultrack_db_show_fake_check.setEnabled(checked)
        if checked:
            self._ultrack_db_frame_initialized = False
            self._refresh_ultrack_db_browser()
        else:
            self._remove_ultrack_db_browser_layers()

    def _remove_ultrack_db_browser_layers(self) -> None:
        self._remove_ultrack_db_preview_selector()
        for name in (
            _ULTRACK_DB_PREVIEW_LAYER,
            _ULTRACK_DB_ANNOTATION_LAYER,
        ):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        if _ULTRACK_DB_SELECTION_LAYER in self.viewer.layers:
            self.viewer.layers.remove(_ULTRACK_DB_SELECTION_LAYER)
        self.ultrack_db_info_lbl.setText("—")
        self._set_ultrack_db_status("")

    def _ultrack_db_middle_frame(self, db_path: Path) -> int | None:
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                frames = sorted(
                    int(r[0]) for r in session.query(NodeDB.t).distinct().all()
                )
        except Exception:
            return None
        finally:
            engine.dispose()
        if not frames:
            return None
        return frames[len(frames) // 2]

    def _refresh_ultrack_db_browser(self) -> None:
        if not self._ultrack_db_browser_active:
            return
        self.ultrack_db_info_lbl.setText("—")
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._set_ultrack_db_status("data.db not found — run DB generation first.")
            return
        frame = self._current_t()
        if not self._ultrack_db_frame_initialized:
            self._ultrack_db_frame_initialized = True
            if frame == 0:
                mid = self._ultrack_db_middle_frame(db_path)
                if mid is not None and mid > 0:
                    frame = mid
                    self._set_viewer_frame(frame)
        try:
            self.ultrack_db_info_lbl.setText(self._ultrack_db_summary_text(db_path, frame))
            mtime_ns = db_path.stat().st_mtime_ns
            states = self._configure_ultrack_db_hierarchy_slider(db_path, mtime_ns, frame)
            if not states:
                labels = self._empty_ultrack_db_preview()
                self._update_layer(_ULTRACK_DB_PREVIEW_LAYER, labels)
                self._set_ultrack_db_status(f"No hierarchy states for frame {frame}.")
                return

            slider_int = int(self.ultrack_db_hierarchy_slider.value())
            state = states[slider_int]
            key = (
                str(db_path.resolve()),
                mtime_ns,
                frame,
                slider_int,
                state,
                self.ultrack_db_show_validated_check.isChecked(),
                self.ultrack_db_show_fake_check.isChecked(),
            )
            cached = self._ultrack_db_preview_cache.get(key)
            if cached is None:
                cached = self._render_hierarchy_cut_state(db_path, frame, state)
                self._ultrack_db_preview_cache[key] = cached
            labels, status, prob_dict, label_to_node_id, node_id_to_label, node_annotations = (
                self._normalize_ultrack_db_preview(cached)
            )
            self._ultrack_db_label_to_node_id = label_to_node_id
            self._ultrack_db_node_id_to_label = node_id_to_label
            self._ultrack_db_node_annotations = node_annotations
            alpha_dict: dict[int, float] = {}
            if self.ultrack_db_connected_focus_check.isChecked():
                labels, status, alpha_dict = self._render_ultrack_db_connected_focus(
                    db_path,
                    frame,
                    labels,
                    status,
                    prob_dict,
                    label_to_node_id,
                    node_id_to_label,
                )
            self._ultrack_db_preview_labels = labels.astype(np.uint32, copy=False)
            self._update_ultrack_db_preview_layer(
                self._ultrack_db_preview_labels, prob_dict, alpha_dict
            )
            self._update_ultrack_db_annotation_layer(
                self._ultrack_db_preview_labels,
                label_to_node_id,
                node_annotations,
            )
            self._install_ultrack_db_preview_selector()
            if not self.ultrack_db_connected_focus_check.isChecked():
                status = self._refresh_ultrack_db_selection_highlight(
                    self._ultrack_db_preview_labels,
                    status,
                    node_id_to_label,
                    frame,
                )
            self._set_ultrack_db_status(status)
        except Exception as e:
            self._set_ultrack_db_status(f"DB read error: {e}")
            logger.warning("DB browser error: %s", e)

    @staticmethod
    def _normalize_ultrack_db_preview(
        cached: tuple[np.ndarray, str]
        | tuple[np.ndarray, str, dict[int, float]]
        | tuple[
            np.ndarray,
            str,
            dict[int, float],
            dict[int, int],
            dict[int, int],
            dict[int, str],
        ],
    ) -> tuple[np.ndarray, str, dict[int, float], dict[int, int], dict[int, int], dict[int, str]]:
        if len(cached) == 2:
            labels, status = cached
            return labels, status, {}, {}, {}, {}
        if len(cached) == 3:
            labels, status, prob_dict = cached
            return labels, status, prob_dict, {}, {}, {}
        if len(cached) == 5:
            labels, status, prob_dict, label_to_node_id, node_id_to_label = cached
            return labels, status, prob_dict, label_to_node_id, node_id_to_label, {}
        labels, status, prob_dict, label_to_node_id, node_id_to_label, node_annotations = cached
        return labels, status, prob_dict, label_to_node_id, node_id_to_label, node_annotations

    def _update_ultrack_db_preview_layer(
        self,
        labels: np.ndarray,
        prob_dict: dict[int, float],
        alpha_dict: dict[int, float] | None = None,
    ) -> None:
        if alpha_dict:
            data = self._ultrack_db_alpha_rgba(labels, alpha_dict)
            self._update_image_layer(_ULTRACK_DB_PREVIEW_LAYER, data, rgb=True)
            return
        if self.ultrack_db_prob_alpha_check.isChecked() and prob_dict:
            data = self._ultrack_db_probability_rgba(labels, prob_dict)
            self._update_image_layer(_ULTRACK_DB_PREVIEW_LAYER, data, rgb=True)
            return
        self._update_labels_layer(_ULTRACK_DB_PREVIEW_LAYER, labels)

    def _update_ultrack_db_annotation_layer(
        self,
        labels: np.ndarray,
        label_to_node_id: dict[int, int],
        node_annotations: dict[int, str],
    ) -> None:
        overlay = np.zeros_like(labels, dtype=np.uint8)
        for label_id, node_id in label_to_node_id.items():
            annot = node_annotations.get(int(node_id), "UNKNOWN")
            if annot == "REAL":
                overlay[labels == int(label_id)] = 1
            elif annot == "FAKE":
                overlay[labels == int(label_id)] = 2
        if not np.any(overlay):
            if _ULTRACK_DB_ANNOTATION_LAYER in self.viewer.layers:
                self.viewer.layers.remove(_ULTRACK_DB_ANNOTATION_LAYER)
            return
        self._update_labels_layer(_ULTRACK_DB_ANNOTATION_LAYER, overlay)

    def _update_labels_layer(self, name: str, data: np.ndarray) -> None:
        from napari.layers import Labels

        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Labels):
            self.viewer.layers[name].data = data
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        self.viewer.add_labels(data, name=name)

    def _update_image_layer(self, name: str, data: np.ndarray, *, rgb: bool = False) -> None:
        from napari.layers import Image

        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Image):
            self.viewer.layers[name].data = data
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        self.viewer.add_image(data, name=name, rgb=rgb, blending="translucent")

    @staticmethod
    def _ultrack_db_probability_rgba(
        labels: np.ndarray, prob_dict: dict[int, float]
    ) -> np.ndarray:
        from napari.utils.colormaps import label_colormap

        rgba = np.zeros(labels.shape + (4,), dtype=np.float32)
        if labels.size == 0 or not prob_dict:
            return rgba

        probs = [float(v) for v in prob_dict.values()]
        min_p = min(probs)
        max_p = max(probs)
        denom = max(max_p - min_p, 1e-9)
        cmap = label_colormap(max(prob_dict.keys()) + 1)
        for label_id, prob in prob_dict.items():
            label_mask = labels == int(label_id)
            if not np.any(label_mask):
                continue
            color = np.asarray(cmap.map(int(label_id)), dtype=np.float32)
            alpha = 0.15 + 0.85 * (float(prob) - min_p) / denom
            color[3] = float(np.clip(alpha, 0.15, 1.0))
            rgba[label_mask] = color
        return rgba

    @staticmethod
    def _ultrack_db_alpha_rgba(
        labels: np.ndarray, alpha_dict: dict[int, float]
    ) -> np.ndarray:
        from napari.utils.colormaps import label_colormap

        rgba = np.zeros(labels.shape + (4,), dtype=np.float32)
        if labels.size == 0 or not alpha_dict:
            return rgba

        cmap = label_colormap(max(alpha_dict.keys()) + 1)
        for label_id, alpha in alpha_dict.items():
            label_mask = labels == int(label_id)
            if not np.any(label_mask):
                continue
            color = np.asarray(cmap.map(int(label_id)), dtype=np.float32)
            color[3] = float(np.clip(alpha, 0.0, 1.0))
            rgba[label_mask] = color
        return rgba

    def _install_ultrack_db_preview_selector(self) -> None:
        if _ULTRACK_DB_PREVIEW_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[_ULTRACK_DB_PREVIEW_LAYER]
        self._remove_ultrack_db_preview_selector()

        def _on_drag(_layer, event):
            if getattr(event, "type", None) != "mouse_press":
                return
            if getattr(event, "button", None) != 1:
                return
            if getattr(event, "modifiers", set()):
                return
            labels = self._ultrack_db_preview_labels
            if labels is None or labels.size == 0:
                return
            pos = _layer.world_to_data(event.position)
            y = int(round(float(pos[-2])))
            x = int(round(float(pos[-1])))
            if y < 0 or x < 0 or y >= labels.shape[-2] or x >= labels.shape[-1]:
                return
            display_label = int(labels[y, x])
            if display_label == 0:
                return
            self._select_ultrack_db_preview_label(display_label, frame=self._current_t())
            yield

        layer.mouse_drag_callbacks.append(_on_drag)
        self._ultrack_db_preview_mouse_callback = _on_drag

    def _remove_ultrack_db_preview_selector(self) -> None:
        callback = self._ultrack_db_preview_mouse_callback
        if callback is None or _ULTRACK_DB_PREVIEW_LAYER not in self.viewer.layers:
            self._ultrack_db_preview_mouse_callback = None
            return
        layer = self.viewer.layers[_ULTRACK_DB_PREVIEW_LAYER]
        try:
            layer.mouse_drag_callbacks.remove(callback)
        except ValueError:
            pass
        self._ultrack_db_preview_mouse_callback = None

    def _select_ultrack_db_preview_label(
        self, display_label: int, *, frame: int | None = None
    ) -> None:
        node_id = self._ultrack_db_label_to_node_id.get(int(display_label))
        if node_id is None:
            self._set_ultrack_db_status(f"No DB node mapped to label {display_label}.")
            self._clear_ultrack_db_highlight()
            return
        selected_frame = self._current_t() if frame is None else int(frame)
        self._ultrack_db_selected_node_id = int(node_id)
        self._ultrack_db_selected_frame = selected_frame
        self._update_ultrack_db_highlight(self._ultrack_db_preview_labels, int(display_label))
        annot = self._ultrack_db_node_annotations.get(int(node_id), "UNKNOWN")
        annot_suffix = "" if annot == "UNKNOWN" else f" [{annot}]"
        self._set_ultrack_db_status(
            f"Selected node {node_id}{annot_suffix} at t={selected_frame}."
        )
        if self.ultrack_db_connected_focus_check.isChecked():
            self._refresh_ultrack_db_browser()

    def _refresh_ultrack_db_selection_highlight(
        self,
        labels: np.ndarray,
        status: str,
        node_id_to_label: dict[int, int],
        frame: int,
    ) -> str:
        selected_node_id = self._ultrack_db_selected_node_id
        if selected_node_id is None:
            self._clear_ultrack_db_highlight()
            return status
        display_label = node_id_to_label.get(int(selected_node_id))
        if display_label is None:
            self._clear_ultrack_db_highlight()
            annot = self._query_ultrack_db_node_annotation_for_status(
                node_id_to_label, selected_node_id
            )
            if annot in {"REAL", "FAKE"}:
                return (
                    f"{status} Selected node {selected_node_id} [{annot}] is hidden "
                    f"by annotation filter at frame {frame}."
                )
            return (
                f"{status} Selected node {selected_node_id} is hidden "
                f"at frame {frame} and the current hierarchy threshold."
            )
        self._update_ultrack_db_highlight(labels, int(display_label))
        return status

    def _query_ultrack_db_node_annotation_for_status(
        self, node_id_to_label: dict[int, int], selected_node_id: int
    ) -> str:
        return self._ultrack_db_node_annotations.get(int(selected_node_id), "UNKNOWN")

    def _get_ultrack_db_highlight_layer(self):
        if _ULTRACK_DB_SELECTION_LAYER in self.viewer.layers:
            return self.viewer.layers[_ULTRACK_DB_SELECTION_LAYER]
        layer = self.viewer.add_shapes(
            name=_ULTRACK_DB_SELECTION_LAYER,
            ndim=2,
            edge_color="cyan",
            edge_width=2,
            face_color="transparent",
        )
        layer.visible = False
        return layer

    def _update_ultrack_db_highlight(
        self, labels: np.ndarray | None, display_label: int
    ) -> None:
        layer = self._get_ultrack_db_highlight_layer()
        if labels is None or display_label == 0:
            layer.data = []
            layer.visible = False
            return
        mask = (labels == int(display_label)).astype(np.uint8)
        if not np.any(mask):
            layer.data = []
            layer.visible = False
            return
        from skimage.measure import find_contours

        contours = find_contours(mask, level=0.5)
        if not contours:
            layer.data = []
            layer.visible = False
            return
        layer.data = [max(contours, key=len)]
        layer.shape_type = ["polygon"]
        layer.visible = True

    def _clear_ultrack_db_highlight(self) -> None:
        if _ULTRACK_DB_SELECTION_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[_ULTRACK_DB_SELECTION_LAYER]
        layer.data = []
        layer.visible = False

    def _query_ultrack_db_connected_nodes(
        self, db_path: Path, selected_node_id: int
    ) -> tuple[dict[int, float], dict[int, float]]:
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import LinkDB

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        predecessors: dict[int, float] = {}
        successors: dict[int, float] = {}
        try:
            with Session(engine) as session:
                rows = (
                    session.query(LinkDB.source_id, LinkDB.target_id, LinkDB.weight)
                    .filter(
                        (LinkDB.source_id == int(selected_node_id))
                        | (LinkDB.target_id == int(selected_node_id))
                    )
                    .all()
                )
                for source_id, target_id, weight in rows:
                    weight_f = float(weight if weight is not None else 1.0)
                    if int(target_id) == int(selected_node_id):
                        source_i = int(source_id)
                        predecessors[source_i] = predecessors.get(source_i, 1.0) * weight_f
                    if int(source_id) == int(selected_node_id):
                        target_i = int(target_id)
                        successors[target_i] = successors.get(target_i, 1.0) * weight_f
        finally:
            engine.dispose()
        return predecessors, successors

    def _render_ultrack_db_connected_focus(
        self,
        db_path: Path,
        frame: int,
        labels: np.ndarray,
        status: str,
        prob_dict: dict[int, float],
        label_to_node_id: dict[int, int],
        node_id_to_label: dict[int, int],
    ) -> tuple[np.ndarray, str, dict[int, float]]:
        selected_node_id = self._ultrack_db_selected_node_id
        selected_frame = self._ultrack_db_selected_frame
        if selected_node_id is None or selected_frame is None:
            self._clear_ultrack_db_highlight()
            return labels, f"{status} Click a DB preview node to focus links.", {}

        predecessors, successors = self._query_ultrack_db_connected_nodes(
            db_path, selected_node_id
        )
        if frame == selected_frame:
            relation = "selected"
            allowed_weights = {selected_node_id: 1.0}
            if int(selected_node_id) not in node_id_to_label:
                self._clear_ultrack_db_highlight()
                empty = np.zeros_like(labels, dtype=np.uint32)
                annot = self._ultrack_db_node_annotations.get(int(selected_node_id), "UNKNOWN")
                annot_suffix = "" if annot == "UNKNOWN" else f" [{annot}]"
                return (
                    empty,
                    f"Selected node {selected_node_id}{annot_suffix} at t={selected_frame} is "
                    "hidden by the current threshold or annotation filter.",
                    {},
                )
        elif frame == selected_frame - 1:
            relation = "t-1"
            allowed_weights = predecessors
        elif frame == selected_frame + 1:
            relation = "t+1"
            allowed_weights = successors
        else:
            empty = np.zeros_like(labels, dtype=np.uint32)
            self._clear_ultrack_db_highlight()
            return (
                empty,
                f"Selected node {selected_node_id} at t={selected_frame} | "
                f"frame {frame}: outside connected focus.",
                {},
            )

        focused = np.zeros_like(labels, dtype=np.uint32)
        alpha_dict: dict[int, float] = {}
        for label_id, node_id in label_to_node_id.items():
            label_i = int(label_id)
            node_i = int(node_id)
            if node_i not in allowed_weights:
                continue
            focused[labels == label_i] = label_i
            alpha_enabled = (
                self.ultrack_db_edge_alpha_check.isChecked()
                or self.ultrack_db_prob_alpha_check.isChecked()
            )
            if alpha_enabled:
                if node_i == selected_node_id:
                    alpha_dict[label_i] = 1.0
                else:
                    alpha_dict[label_i] = self._ultrack_db_connected_alpha(
                        label_i,
                        float(allowed_weights[node_i]),
                        prob_dict,
                    )

        selected_label = node_id_to_label.get(int(selected_node_id))
        if frame == selected_frame and selected_label is not None:
            self._update_ultrack_db_highlight(focused, int(selected_label))
        else:
            self._clear_ultrack_db_highlight()

        edge_values = [
            float(v)
            for node_id, v in allowed_weights.items()
            if node_id in node_id_to_label and node_id != selected_node_id
        ]
        if edge_values:
            edge_summary = (
                f" | edge product range {min(edge_values):.2f}-{max(edge_values):.2f}"
            )
        else:
            edge_summary = ""
        count = int(np.unique(focused[focused != 0]).size)
        annot = self._ultrack_db_node_annotations.get(int(selected_node_id), "UNKNOWN")
        annot_suffix = "" if annot == "UNKNOWN" else f" [{annot}]"
        return (
            focused,
            f"Selected node {selected_node_id}{annot_suffix} at t={selected_frame} | "
            f"{relation}: {count} connected node(s){edge_summary}",
            alpha_dict,
        )

    def _ultrack_db_connected_alpha(
        self,
        label_id: int,
        edge_weight: float,
        prob_dict: dict[int, float],
    ) -> float:
        alpha = 1.0
        if self.ultrack_db_edge_alpha_check.isChecked():
            alpha *= float(edge_weight)
        if self.ultrack_db_prob_alpha_check.isChecked() and prob_dict:
            probs = [float(v) for v in prob_dict.values()]
            min_p = min(probs)
            max_p = max(probs)
            denom = max(max_p - min_p, 1e-9)
            prob = float(prob_dict.get(int(label_id), 1.0))
            alpha *= 0.15 + 0.85 * (prob - min_p) / denom
        return float(np.clip(alpha, 0.05, 1.0))

    def _ultrack_db_summary_text(self, db_path: Path, frame: int) -> str:
        import sqlalchemy as sqla
        from sqlalchemy import func
        from sqlalchemy.orm import Session
        from ultrack.core.database import LinkDB, NodeDB, VarAnnotation

        try:
            from ultrack.core.database import OverlapDB
        except Exception:
            OverlapDB = None

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                n_nodes = int(session.query(func.count(NodeDB.id)).scalar() or 0)
                n_links = int(session.query(func.count(LinkDB.source_id)).scalar() or 0)
                n_real = int(
                    session.query(func.count(NodeDB.id))
                    .filter(NodeDB.node_annot == VarAnnotation.REAL)
                    .scalar() or 0
                )
                n_fake = int(
                    session.query(func.count(NodeDB.id))
                    .filter(NodeDB.node_annot == VarAnnotation.FAKE)
                    .scalar() or 0
                )
                frame_nodes = session.query(NodeDB).filter(NodeDB.t == frame).all()
                selected = sum(1 for n in frame_nodes if getattr(n, "selected", False))
                node_ids = [int(n.id) for n in frame_nodes]
                outgoing = incoming = overlaps = 0
                if node_ids:
                    outgoing = int(
                        session.query(func.count(LinkDB.source_id))
                        .filter(LinkDB.source_id.in_(node_ids))
                        .scalar() or 0
                    )
                    incoming = int(
                        session.query(func.count(LinkDB.target_id))
                        .filter(LinkDB.target_id.in_(node_ids))
                        .scalar() or 0
                    )
                    if OverlapDB is not None:
                        try:
                            overlaps = int(
                                session.query(func.count(OverlapDB.node_id))
                                .filter(
                                    OverlapDB.node_id.in_(node_ids)
                                    | OverlapDB.ancestor_id.in_(node_ids)
                                )
                                .scalar() or 0
                            )
                        except Exception:
                            overlaps = 0
            return (
                f"{n_nodes} nodes | {n_links} links | REAL {n_real} | FAKE {n_fake} | frame {frame}: "
                f"{len(node_ids)} nodes, {selected} selected, "
                f"{incoming} in/{outgoing} out links, {overlaps} overlaps"
            )
        finally:
            engine.dispose()

    def _query_distinct_heights(self, db_path: Path, mtime_ns: int) -> tuple[float, ...]:
        key = (str(db_path.resolve()), mtime_ns)
        cached = self._ultrack_db_height_values_cache.get(key)
        if cached is not None:
            return cached
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                heights = tuple(
                    float(row[0])
                    for row in session.query(NodeDB.height)
                    .distinct()
                    .order_by(NodeDB.height)
                    .all()
                    if row[0] is not None
                )
        finally:
            engine.dispose()
        self._ultrack_db_height_values_cache[key] = heights
        return heights

    def _query_hierarchy_cut_states(
        self, db_path: Path, mtime_ns: int, frame: int
    ) -> tuple[_HierarchyCutState, ...]:
        key = (str(db_path.resolve()), mtime_ns, frame)
        cached = self._ultrack_db_cut_state_cache.get(key)
        if cached is not None:
            return cached

        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        from ultrack.utils.constants import NO_PARENT

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                rows = [
                    (int(node_id), int(parent_id), float(height))
                    for node_id, parent_id, height in session.query(
                        NodeDB.id, NodeDB.hier_parent_id, NodeDB.height
                    )
                    .filter(NodeDB.t == frame)
                    .order_by(NodeDB.height, NodeDB.id)
                    .all()
                    if height is not None
                ]
        except Exception:
            heights = self._query_distinct_heights(db_path, mtime_ns)
            return tuple(_HierarchyCutState((), float(height)) for height in heights)
        finally:
            engine.dispose()

        if not rows:
            self._ultrack_db_cut_state_cache[key] = ()
            return ()

        node_ids = {node_id for node_id, _parent_id, _height in rows}
        heights_by_id = {node_id: height for node_id, _parent_id, height in rows}
        parent_by_id = {
            node_id: parent_id
            for node_id, parent_id, _height in rows
            if parent_id != NO_PARENT and parent_id in node_ids
        }
        children_by_parent: dict[int, set[int]] = {}
        for child_id, parent_id in parent_by_id.items():
            children_by_parent.setdefault(parent_id, set()).add(child_id)

        active = {
            node_id for node_id, _parent_id, _height in rows
            if node_id not in children_by_parent
        }
        if not active:
            active = set(node_ids)

        states: list[_HierarchyCutState] = []
        seen_states: set[tuple[int, ...]] = set()

        def _append_state() -> None:
            ordered = tuple(
                sorted(active, key=lambda node_id: (heights_by_id[node_id], node_id))
            )
            if ordered in seen_states:
                return
            seen_states.add(ordered)
            height = max((heights_by_id[node_id] for node_id in ordered), default=None)
            states.append(_HierarchyCutState(ordered, height))

        _append_state()
        while True:
            promotable = [
                parent_id
                for parent_id, child_ids in children_by_parent.items()
                if parent_id not in active and child_ids and child_ids.issubset(active)
            ]
            if not promotable:
                break
            min_height = min(heights_by_id[parent_id] for parent_id in promotable)
            promote_now = [
                parent_id
                for parent_id in promotable
                if heights_by_id[parent_id] == min_height
            ]
            for parent_id in sorted(promote_now):
                active.difference_update(children_by_parent[parent_id])
                active.add(parent_id)
            _append_state()

        result = tuple(states)
        self._ultrack_db_cut_state_cache[key] = result
        return result

    def _configure_ultrack_db_hierarchy_slider(
        self, db_path: Path, mtime_ns: int, frame: int
    ) -> tuple[_HierarchyCutState, ...]:
        states = self._query_hierarchy_cut_states(db_path, mtime_ns, frame)
        maximum = max(len(states) - 1, 0)
        value = min(max(int(self.ultrack_db_hierarchy_slider.value()), 0), maximum)

        old_blocked = self.ultrack_db_hierarchy_slider.blockSignals(True)
        try:
            self.ultrack_db_hierarchy_slider.setRange(0, maximum)
            self.ultrack_db_hierarchy_slider.setValue(value)
        finally:
            self.ultrack_db_hierarchy_slider.blockSignals(old_blocked)

        if states:
            self._set_ultrack_db_height_label(value, states[value].height, len(states))
        else:
            self.ultrack_db_height_lbl.setText("—")
        return states

    def _set_ultrack_db_height_label(
        self, index: int, height: float | None, total: int
    ) -> None:
        height_text = "—" if height is None else f"{height:.2f}"
        self.ultrack_db_height_lbl.setText(
            f"i={index} h={height_text} ({index + 1}/{total})"
        )

    def _render_hierarchy_cut(
        self, db_path: Path, frame: int, h_actual: float
    ) -> tuple[
        np.ndarray,
        str,
        dict[int, float],
        dict[int, int],
        dict[int, int],
        dict[int, str],
    ]:
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session, aliased
        from ultrack.core.database import NodeDB
        from ultrack.utils.constants import NO_PARENT

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                P = aliased(NodeDB)
                C = aliased(NodeDB)
                same_height_child_exists = (
                    session.query(C.id)
                    .where(C.hier_parent_id == NodeDB.id)
                    .where(C.height == NodeDB.height)
                    .where(NodeDB.height == h_actual)
                    .exists()
                )
                nodes = (
                    session.query(NodeDB)
                    .outerjoin(P, NodeDB.hier_parent_id == P.id)
                    .where(NodeDB.t == frame)
                    .where(NodeDB.height <= h_actual)
                    .where(
                        (NodeDB.hier_parent_id == NO_PARENT)
                        | ((NodeDB.height < h_actual) & (P.height > h_actual))
                        | ((NodeDB.height == h_actual) & (P.height >= h_actual))
                    )
                    .where(~same_height_child_exists)
                    .all()
                )
        finally:
            engine.dispose()

        return self._finalize_hierarchy_nodes(
            nodes,
            frame,
            empty_msg=f"No segments at this threshold for frame {frame}.",
            status_suffix=f"at h={h_actual:.2f}",
        )

    def _render_hierarchy_cut_state(
        self, db_path: Path, frame: int, state: _HierarchyCutState
    ) -> tuple[
        np.ndarray,
        str,
        dict[int, float],
        dict[int, int],
        dict[int, int],
        dict[int, str],
    ]:
        if not state.node_ids:
            return self._render_hierarchy_cut(db_path, frame, float(state.height or 0.0))

        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                rows = (
                    session.query(NodeDB)
                    .where(NodeDB.t == frame)
                    .where(NodeDB.id.in_(state.node_ids))
                    .all()
                )
        finally:
            engine.dispose()

        nodes_by_id = {int(node.id): node for node in rows}
        nodes = [
            nodes_by_id[node_id]
            for node_id in state.node_ids
            if node_id in nodes_by_id
        ]
        height_text = "—" if state.height is None else f"{state.height:.2f}"
        return self._finalize_hierarchy_nodes(
            nodes,
            frame,
            empty_msg=f"No hierarchy state segments for frame {frame}.",
            status_suffix=f"at cut state h={height_text}",
        )

    def _finalize_hierarchy_nodes(
        self,
        nodes: list,
        frame: int,
        *,
        empty_msg: str,
        status_suffix: str,
    ) -> tuple[
        np.ndarray,
        str,
        dict[int, float],
        dict[int, int],
        dict[int, int],
        dict[int, str],
    ]:
        if not nodes:
            return self._empty_ultrack_db_preview(), empty_msg, {}, {}, {}, {}
        show_validated = self.ultrack_db_show_validated_check.isChecked()
        show_fake = self.ultrack_db_show_fake_check.isChecked()
        filtered_nodes = []
        hidden_real = hidden_fake = 0
        for node in nodes:
            annot = self._ultrack_db_annotation_name(getattr(node, "node_annot", None))
            if annot == "REAL" and not show_validated:
                hidden_real += 1
                continue
            if annot == "FAKE" and not show_fake:
                hidden_fake += 1
                continue
            filtered_nodes.append(node)
        if not filtered_nodes:
            return self._empty_ultrack_db_preview(), (
                f"Frame {frame}: annotation filters hid all {len(nodes)} segment(s)."
            ), {}, {}, {}, {}
        labels = self._paint_ultrack_db_nodes(filtered_nodes)
        prob_dict, label_to_node_id, node_id_to_label = (
            self._ultrack_db_node_preview_metadata(filtered_nodes)
        )
        node_annotations = self._ultrack_db_node_annotation_metadata(filtered_nodes)
        hidden_summary = ""
        if hidden_real or hidden_fake:
            hidden_summary = f" Hidden by annotation filter: REAL {hidden_real}, FAKE {hidden_fake}."
        return labels, (
            f"Frame {frame}: {len(filtered_nodes)} segment(s) {status_suffix}."
            f"{hidden_summary}"
        ), prob_dict, label_to_node_id, node_id_to_label, node_annotations

    @staticmethod
    def _ultrack_db_annotation_name(value) -> str:
        if value is None:
            return "UNKNOWN"
        raw = getattr(value, "value", value)
        if raw is None:
            return "UNKNOWN"
        name = str(raw).split(".")[-1].upper()
        if name in {"REAL", "FAKE"}:
            return name
        return "UNKNOWN"

    @staticmethod
    def _ultrack_db_node_preview_metadata(
        nodes: list,
    ) -> tuple[dict[int, float], dict[int, int], dict[int, int]]:
        prob_dict: dict[int, float] = {}
        label_to_node_id: dict[int, int] = {}
        node_id_to_label: dict[int, int] = {}
        for label, node in enumerate(nodes, start=1):
            try:
                prob = float(node.node_prob if node.node_prob is not None else 1.0)
            except (TypeError, ValueError):
                prob = 1.0
            prob_dict[label] = prob
            try:
                node_id = int(node.id)
            except (TypeError, ValueError):
                continue
            label_to_node_id[label] = node_id
            node_id_to_label[node_id] = label
        return prob_dict, label_to_node_id, node_id_to_label

    @staticmethod
    def _ultrack_db_node_annotation_metadata(nodes: list) -> dict[int, str]:
        node_annotations: dict[int, str] = {}
        for node in nodes:
            try:
                node_id = int(node.id)
            except (TypeError, ValueError):
                continue
            node_annotations[node_id] = NucleusWorkflowWidget._ultrack_db_annotation_name(
                getattr(node, "node_annot", None)
            )
        return node_annotations

    def _empty_ultrack_db_preview(self) -> np.ndarray:
        shape = self._viewer_plane_shape()
        return np.zeros(shape, dtype=np.uint32)

    def _viewer_plane_shape(self) -> tuple[int, int]:
        for layer in self.viewer.layers:
            data = getattr(layer, "data", None)
            if isinstance(data, np.ndarray) and data.ndim >= 2:
                return tuple(int(v) for v in data.shape[-2:])
        return (1, 1)

    def _paint_ultrack_db_nodes(self, nodes: list) -> np.ndarray:
        masks: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
        max_y = max_x = 0
        for label, node in enumerate(nodes, start=1):
            parsed = self._node_mask_and_bbox(node)
            if parsed is None:
                continue
            bbox, mask = parsed
            y0, x0, y1, x1 = bbox
            max_y = max(max_y, y1)
            max_x = max(max_x, x1)
            masks.append((label, bbox, mask))

        base_y, base_x = self._viewer_plane_shape()
        labels = np.zeros((max(base_y, max_y, 1), max(base_x, max_x, 1)), dtype=np.uint32)
        for label, (y0, x0, y1, x1), mask in masks:
            target = labels[y0:y1, x0:x1]
            if target.shape != mask.shape:
                continue
            target[mask.astype(bool)] = label
        return labels

    @staticmethod
    def _node_mask_and_bbox(node) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        try:
            # MaybePickleType already unpickles on read; only call pickle.loads if raw bytes
            node_obj = node.pickle
            if isinstance(node_obj, (bytes, memoryview)):
                node_obj = pickle.loads(bytes(node_obj))
            if node_obj is None:
                return None
        except Exception:
            return None

        if isinstance(node_obj, dict):
            bbox = node_obj.get("bbox")
            mask = node_obj.get("mask")
        elif isinstance(node_obj, tuple) and len(node_obj) >= 2:
            bbox, mask = node_obj[0], node_obj[1]
        else:
            bbox = getattr(node_obj, "bbox", None)
            mask = getattr(node_obj, "mask", None)

        if bbox is None or mask is None:
            return None
        bbox_arr = np.asarray(bbox, dtype=int).ravel()
        if bbox_arr.size >= 6:
            y0, x0, y1, x1 = int(bbox_arr[1]), int(bbox_arr[2]), int(bbox_arr[4]), int(bbox_arr[5])
        elif bbox_arr.size >= 4:
            y0, x0, y1, x1 = (int(v) for v in bbox_arr[:4])
        else:
            return None

        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
            mask_arr = mask_arr[0]
        elif mask_arr.ndim > 2:
            mask_arr = np.squeeze(mask_arr)
        if mask_arr.ndim != 2:
            return None
        if mask_arr.shape != (y1 - y0, x1 - x0):
            return None
        return (y0, x0, y1, x1), mask_arr.astype(bool, copy=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _current_t(self) -> int:
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) >= 1 else 0

    def _update_tracked_display(
        self,
        labels: np.ndarray,
        t: int | None = None,
    ) -> None:
        if _TRACKED_LAYER in self.viewer.layers and t is not None:
            layer = self.viewer.layers[_TRACKED_LAYER]
            if layer.data.ndim == 3:
                if t < layer.data.shape[0]:
                    new_data = layer.data.copy()
                    new_data[t] = labels
                    layer.data = new_data
                    return
                # Extend the in-memory stack rather than reloading from disk.
                new_data = np.concatenate(
                    [layer.data, labels[np.newaxis].astype(layer.data.dtype)], axis=0
                )
                layer.data = new_data
                return
        display = labels[np.newaxis].copy() if labels.ndim == 2 else labels
        self._update_layer(_TRACKED_LAYER, display)

    def _update_layer(self, name: str, data: np.ndarray) -> None:
        self._update_labels_layer(name, data)

    def _set_viewer_frame(self, t: int) -> None:
        step = list(self.viewer.dims.current_step)
        if not step:
            return
        step[0] = int(t)
        self.viewer.dims.current_step = tuple(step)

    @staticmethod
    def _sigmoid_zavg(stack: np.ndarray) -> np.ndarray:
        zavg_logits = np.asarray(stack, dtype=np.float32).mean(axis=1)
        return (1.0 / (1.0 + np.exp(-zavg_logits))).astype(np.float32)

    def _set_contour_status(self, msg: str) -> None:
        self.contour_status_lbl.setText(msg)
        self.contour_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_correction_status(self, msg: str) -> None:
        self.correction_status_lbl.setText(msg)
        self.correction_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_ultrack_status(self, msg: str) -> None:
        self.ultrack_status_lbl.setText(msg)
        self.ultrack_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _on_contour_worker_error(self, exc: Exception) -> None:
        self.build_progress_bar.setVisible(False)
        self._set_contour_status(f"Error: {exc}")
        logger.exception("Contour worker error", exc_info=exc)

    def _on_correction_worker_error(self, exc: Exception) -> None:
        self._set_correction_status(f"Error: {exc}")
        logger.exception("Correction worker error", exc_info=exc)

    def _on_ultrack_worker_error(self, exc: Exception) -> None:
        self.ultrack_progress_bar.setVisible(False)
        self.ultrack_progress_bar.setRange(0, 100)
        self.run_ultrack_btn.setEnabled(True)
        self.ultrack_terminal_btn.setEnabled(True)
        self._set_ultrack_status(f"Error: {exc}")
        logger.exception("Ultrack worker error", exc_info=exc)

    def _cp_gammas(self) -> list[float]:
        """Gamma values to iterate during consensus boundary building."""
        gmin  = self.cp_gamma_min_spin.value()
        gmax  = self.cp_gamma_max_spin.value()
        gstep = self.cp_gamma_step_spin.value()
        return list(np.arange(gmin, gmax + gstep / 2, gstep))


    # ──────────────────────────────────────────────────────────────────────────
    # 1. Contour map build
    # ──────────────────────────────────────────────────────────────────────────

    def _build_consensus_boundary_averaged(
        self,
        prob_3d: np.ndarray,
        dp_3d: np.ndarray,
        thresholds: list[float],
        gammas: list[float],
        *,
        flow_threshold: float = 0.0,
        mask_callback=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from cellflow.segmentation import build_consensus_boundary

        boundary_sum  = None
        foreground_sum = None
        for g_idx, g in enumerate(gammas):
            cb = None
            if mask_callback is not None:
                def cb(masks, i_thresh, *, _gi=g_idx):
                    mask_callback(masks, _gi, i_thresh)
            b, fg = build_consensus_boundary(
                prob_3d,
                dp_3d,
                thresholds,
                gamma=g,
                flow_threshold=flow_threshold,
                mask_callback=cb,
            )
            if boundary_sum is None:
                boundary_sum  = b.copy()
                foreground_sum = fg.copy()
            else:
                boundary_sum  += b
                foreground_sum += fg
        n = len(gammas)
        return boundary_sum / n, foreground_sum / n

    def _on_build_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path   = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return

        thresholds      = list(np.arange(self.cp_min_spin.value(), self.cp_max_spin.value() + self.cp_step_spin.value() / 2, self.cp_step_spin.value()))
        gammas          = self._cp_gammas()
        contour_path    = self._contour_maps_path()
        score_path      = self._foreground_scores_path()
        mask_path       = self._foreground_masks_path()
        foreground_threshold = self.contour_fg_threshold_spin.value()
        flow_threshold = self.contour_flow_threshold_spin.value()
        save_source     = self.save_source_check.isChecked()
        pos_dir         = self._pos_dir
        build_fn        = self._build_consensus_boundary_averaged
        if contour_path is None or score_path is None or mask_path is None:
            self._set_contour_status("No project open.")
            return

        @thread_worker(connect={
            "yielded":   self._on_build_progress,
            "returned":  self._on_build_done,
            "errored":   self._on_contour_worker_error,
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            dp_stack   = np.asarray(tifffile.imread(str(dp_path)),   dtype=np.float32)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis]
            if dp_stack.ndim == 4:
                dp_stack = dp_stack[np.newaxis]

            n_t = prob_stack.shape[0]
            contour_frames:    list[np.ndarray] = []
            foreground_score_frames: list[np.ndarray] = []
            foreground_mask_frames: list[np.ndarray] = []
            source_dir = pos_dir / "2_nucleus/source_labels"

            for t in range(n_t):
                yield (t + 1, n_t, f"Building contour maps and foreground masks: frame {t + 1}/{n_t}…")
                mask_cb = None
                if save_source:
                    source_dir.mkdir(parents=True, exist_ok=True)
                    def mask_cb(masks, g_idx, thresh_idx, *, _t=t):
                        tifffile.imwrite(
                            source_dir / f"masks_t{_t:04d}_g{g_idx:02d}_thr{thresh_idx:02d}.tif",
                            masks, compression="zlib",
                        )
                boundary, foreground_score = build_fn(
                    prob_stack[t],
                    dp_stack[t],
                    thresholds,
                    gammas,
                    flow_threshold=flow_threshold,
                    mask_callback=mask_cb,
                )
                contour_frames.append(boundary.astype(np.float32, copy=False))
                foreground_score = foreground_score.astype(np.float32, copy=False)
                foreground_score_frames.append(foreground_score)
                foreground_mask_frames.append(
                    (foreground_score >= foreground_threshold).astype(np.uint8)
                )

            contour_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(contour_path), np.stack(contour_frames), compression="zlib")
            tifffile.imwrite(str(score_path), np.stack(foreground_score_frames), compression="zlib")
            tifffile.imwrite(str(mask_path), np.stack(foreground_mask_frames), compression="zlib")
            return pos_dir

        gamma_desc = f"γ={gammas[0]:.2f}" if len(gammas) == 1 else f"γ={gammas[0]:.2f}–{gammas[-1]:.2f} ({len(gammas)} steps)"
        self._set_contour_status(f"Building contour maps and foreground masks ({len(thresholds)} cellprob thresholds, {gamma_desc})…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _on_build_done(self, pos_dir: Path) -> None:
        self._build_worker = None
        self._set_build_buttons_running(False)
        self._refresh_stage_files(pos_dir)
        self._set_contour_status("Contour maps and foreground masks built.")

    def _on_cancel_build(self) -> None:
        if self._build_worker is not None:
            self._build_worker.quit()
        self._build_worker = None
        self._set_build_buttons_running(False)
        self._set_contour_status("Build cancelled.")

    def _set_build_buttons_running(self, running: bool) -> None:
        self.build_btn.setEnabled(not running)
        self.preview_contour_btn.setEnabled(not running)
        self.contour_terminal_btn.setEnabled(not running)
        self.preview_contour_filter_btn.setEnabled(not running)
        self.run_contour_filter_btn.setEnabled(not running)
        self.cancel_build_btn.setEnabled(running)
        self.build_progress_bar.setVisible(running)
        if not running:
            self.build_progress_bar.setValue(0)

    def _on_build_progress(self, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            if total > 0:
                self.build_progress_bar.setRange(0, total)
                self.build_progress_bar.setValue(done)
            self._set_contour_status(msg)
        else:
            self._set_contour_status(str(data))

    def _on_preview_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path   = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return

        t_frame    = self._current_t()
        thresholds = list(np.arange(self.cp_min_spin.value(), self.cp_max_spin.value() + self.cp_step_spin.value() / 2, self.cp_step_spin.value()))
        gammas     = self._cp_gammas()
        flow_threshold = self.contour_flow_threshold_spin.value()
        build_fn   = self._build_consensus_boundary_averaged

        def _on_preview_done(result):
            self._build_worker = None
            self._set_build_buttons_running(False)
            boundary, foreground, cellprob_zavg, t_idx = result
            data = np.zeros((cellprob_zavg.shape[0],) + boundary.shape, dtype=boundary.dtype)
            data[t_idx] = boundary
            foreground_score_data = np.zeros(
                (cellprob_zavg.shape[0],) + foreground.shape, dtype=np.float32
            )
            foreground_score_data[t_idx] = foreground
            foreground_mask_data = (
                foreground_score_data >= self.contour_fg_threshold_spin.value()
            ).astype(np.uint8)
            if _CELLPROB_LAYER in self.viewer.layers:
                self.viewer.layers[_CELLPROB_LAYER].data = cellprob_zavg
            else:
                self.viewer.add_image(
                    cellprob_zavg,
                    name=_CELLPROB_LAYER,
                    colormap="inferno",
                    blending="additive",
                    visible=True,
                )
            if _CONTOUR_LAYER in self.viewer.layers:
                self.viewer.layers[_CONTOUR_LAYER].data = data
            else:
                self.viewer.add_image(data, name=_CONTOUR_LAYER, colormap="magma", visible=True)
            if _FOREGROUND_SCORE_LAYER in self.viewer.layers:
                self.viewer.layers[_FOREGROUND_SCORE_LAYER].data = foreground_score_data
            else:
                self.viewer.add_image(
                    foreground_score_data,
                    name=_FOREGROUND_SCORE_LAYER,
                    colormap="viridis",
                    visible=True,
                )
            self._update_layer(_FOREGROUND_MASK_LAYER, foreground_mask_data)
            self._set_viewer_frame(t_idx)
            self._set_contour_status(
                f"Preview contour map and foreground mask t={t_idx} — "
                f"{len(thresholds)} cellprob thresholds, "
                f"{len(gammas)} gamma value(s)"
            )

        @thread_worker(connect={
            "returned": _on_preview_done,
            "errored":  self._on_contour_worker_error,
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            dp_stack   = np.asarray(tifffile.imread(str(dp_path)),   dtype=np.float32)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis]
            if dp_stack.ndim == 4:
                dp_stack = dp_stack[np.newaxis]
            n_t = min(prob_stack.shape[0], dp_stack.shape[0])
            t_idx = min(max(t_frame, 0), n_t - 1)
            boundary, foreground = build_fn(
                prob_stack[t_idx],
                dp_stack[t_idx],
                thresholds,
                gammas,
                flow_threshold=flow_threshold,
            )
            return boundary, foreground, self._sigmoid_zavg(prob_stack), t_idx

        self._set_contour_status(f"Previewing contour map for frame t={t_frame}…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _contour_filter_params_from_ui(self):
        from cellflow.segmentation import ContourFilterParams

        return ContourFilterParams(
            median_kernel_time=int(self.contour_filter_median_time_spin.value()),
            median_kernel_space=int(self.contour_filter_median_space_spin.value()),
            gaussian_sigma_time=float(self.contour_filter_gauss_time_spin.value()),
            gaussian_sigma_space=float(self.contour_filter_gauss_space_spin.value()),
        )

    def _update_contour_image_layer(self, data: np.ndarray) -> None:
        if _CONTOUR_LAYER in self.viewer.layers:
            self.viewer.layers[_CONTOUR_LAYER].data = data
        else:
            self.viewer.add_image(
                data,
                name=_CONTOUR_LAYER,
                colormap="magma",
                visible=True,
            )

    def _on_preview_contour_filter(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        contour_path = self._contour_maps_path()
        if contour_path is None or not contour_path.exists():
            self._set_contour_status(
                "Missing: contour_maps.tif — run Contour Maps first."
            )
            return

        params = self._contour_filter_params_from_ui()

        def _on_preview_done(result):
            self._build_worker = None
            self._set_build_buttons_running(False)
            filtered = result
            self._update_contour_image_layer(filtered)
            self._set_contour_status("Previewed filtered contour maps.")

        @thread_worker(connect={
            "returned": _on_preview_done,
            "errored":  self._on_contour_worker_error,
        })
        def _worker():
            from cellflow.segmentation import compute_filtered_contour_maps

            contours = np.asarray(tifffile.imread(str(contour_path)), dtype=np.float32)
            return compute_filtered_contour_maps(contours, params)

        self._set_contour_status("Previewing filtered contour maps…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _on_run_contour_filter(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        contour_path = self._contour_maps_path()
        if contour_path is None or not contour_path.exists():
            self._set_contour_status(
                "Missing: contour_maps.tif — run Contour Maps first."
            )
            return

        params = self._contour_filter_params_from_ui()
        pos_dir = self._pos_dir

        def _on_filter_done(result):
            self._build_worker = None
            self._set_build_buttons_running(False)
            pos_dir, filtered = result
            self._refresh_stage_files(pos_dir)
            self._update_contour_image_layer(filtered)
            self._set_contour_status("Filtered contour maps written to contour_maps.tif.")

        @thread_worker(connect={
            "returned": _on_filter_done,
            "errored":  self._on_contour_worker_error,
        })
        def _worker():
            from cellflow.segmentation import compute_filtered_contour_maps

            contours = np.asarray(tifffile.imread(str(contour_path)), dtype=np.float32)
            filtered = compute_filtered_contour_maps(contours, params)
            tifffile.imwrite(
                str(contour_path),
                filtered.astype(np.float32, copy=False),
                compression="zlib",
                photometric="minisblack",
            )
            return pos_dir, filtered

        self._set_contour_status("Filtering contour_maps.tif…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _on_run_contour_terminal(self) -> None:
        import sys
        import tempfile

        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path = self._dp_path()
        contour_path = self._contour_maps_path()
        score_path = self._foreground_scores_path()
        mask_path = self._foreground_masks_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return
        if contour_path is None or score_path is None or mask_path is None:
            self._set_contour_status("No project open.")
            return

        thresholds = list(
            np.arange(
                self.cp_min_spin.value(),
                self.cp_max_spin.value() + self.cp_step_spin.value() / 2,
                self.cp_step_spin.value(),
            )
        )
        gammas = self._cp_gammas()
        foreground_threshold = self.contour_fg_threshold_spin.value()
        flow_threshold = self.contour_flow_threshold_spin.value()
        save_source = self.save_source_check.isChecked()
        pos_dir = self._pos_dir

        python_code = (
            "import pathlib\n"
            "import numpy as np\n"
            "import tifffile\n"
            "from cellflow.segmentation import build_consensus_boundary\n"
            f"prob_path = pathlib.Path({str(prob_path)!r})\n"
            f"dp_path = pathlib.Path({str(dp_path)!r})\n"
            f"contour_path = pathlib.Path({str(contour_path)!r})\n"
            f"score_path = pathlib.Path({str(score_path)!r})\n"
            f"mask_path = pathlib.Path({str(mask_path)!r})\n"
            f"save_source = {save_source!r}\n"
            f"source_dir = pathlib.Path({str(pos_dir / '2_nucleus/source_labels')!r})\n"
            f"thresholds = {thresholds!r}\n"
            f"gammas = {gammas!r}\n"
            f"foreground_threshold = {foreground_threshold!r}\n"
            f"flow_threshold = {flow_threshold!r}\n"
            "def build_consensus_boundary_averaged(prob_3d, dp_3d, thresholds, gammas, flow_threshold=0.0, mask_callback=None):\n"
            "    boundary_sum = None\n"
            "    foreground_sum = None\n"
            "    for g_idx, g in enumerate(gammas):\n"
            "        cb = None\n"
            "        if mask_callback is not None:\n"
            "            def cb(masks, i_thresh, *, _gi=g_idx):\n"
            "                mask_callback(masks, _gi, i_thresh)\n"
            "        boundary, foreground = build_consensus_boundary(\n"
            "            prob_3d,\n"
            "            dp_3d,\n"
            "            thresholds,\n"
            "            gamma=g,\n"
            "            flow_threshold=flow_threshold,\n"
            "            mask_callback=cb,\n"
            "        )\n"
            "        if boundary_sum is None:\n"
            "            boundary_sum = boundary.copy()\n"
            "            foreground_sum = foreground.copy()\n"
            "        else:\n"
            "            boundary_sum += boundary\n"
            "            foreground_sum += foreground\n"
            "    n = len(gammas)\n"
            "    return boundary_sum / n, foreground_sum / n\n"
            "prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)\n"
            "dp_stack = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)\n"
            "if prob_stack.ndim == 3:\n"
            "    prob_stack = prob_stack[np.newaxis]\n"
            "if dp_stack.ndim == 4:\n"
            "    dp_stack = dp_stack[np.newaxis]\n"
            "n_t = prob_stack.shape[0]\n"
            "contour_frames = []\n"
            "foreground_score_frames = []\n"
            "foreground_mask_frames = []\n"
            "for t in range(n_t):\n"
            "    print(f'Building contour maps and foreground masks: frame {t + 1}/{n_t}...', flush=True)\n"
            "    mask_cb = None\n"
            "    if save_source:\n"
            "        source_dir.mkdir(parents=True, exist_ok=True)\n"
            "        def mask_cb(masks, g_idx, thresh_idx, *, _t=t):\n"
            "            tifffile.imwrite(\n"
            "                source_dir / f'masks_t{_t:04d}_g{g_idx:02d}_thr{thresh_idx:02d}.tif',\n"
            "                masks,\n"
            "                compression='zlib',\n"
            "            )\n"
            "    boundary, foreground = build_consensus_boundary_averaged(\n"
            "        prob_stack[t],\n"
            "        dp_stack[t],\n"
            "        thresholds,\n"
            "        gammas,\n"
            "        flow_threshold=flow_threshold,\n"
            "        mask_callback=mask_cb,\n"
            "    )\n"
            "    contour_frames.append(boundary.astype(np.float32, copy=False))\n"
            "    foreground = foreground.astype(np.float32, copy=False)\n"
            "    foreground_score_frames.append(foreground)\n"
            "    foreground_mask_frames.append((foreground >= foreground_threshold).astype(np.uint8))\n"
            "contour_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "print('Writing contour maps and foreground masks...', flush=True)\n"
            "tifffile.imwrite(str(contour_path), np.stack(contour_frames), compression='zlib')\n"
            "tifffile.imwrite(str(score_path), np.stack(foreground_score_frames), compression='zlib')\n"
            "tifffile.imwrite(str(mask_path), np.stack(foreground_mask_frames), compression='zlib')\n"
            "print('Done.')\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="cellflow_contour_build_", delete=False
        ) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path)}"
        try:
            from cellflow.napari.utils import launch_in_terminal
            launch_in_terminal(cmd)
            self._set_contour_status("Contour build command launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(cmd)
            self._set_contour_status(
                "Copied contour build command to clipboard (terminal launch unavailable)."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Automated search / propagation
    # ──────────────────────────────────────────────────────────────────────────

    def _on_save_tracked(self) -> None:
        tracked_path = self._tracked_path()
        if tracked_path is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer to save.")
            return
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3:
            self._set_correction_status("Tracked layer is not a 3D stack.")
            return
        n = layer.data.shape[0]
        for t in range(n):
            write_tracked_frame(tracked_path, t, np.asarray(layer.data[t]))
        self._set_correction_status(f"Saved {n} frame(s) to {tracked_path.name}.")

    def _on_load_tracked(self) -> None:
        tracked_path   = self._tracked_path()
        cell_zavg_path = self._cell_zavg_path()
        nuc_zavg_path  = self._nucleus_zavg_path()
        if tracked_path is None or not tracked_path.exists():
            self._set_correction_status("No tracked labels file found.")
            return
        self._set_correction_status("Loading tracked labels…")

        @thread_worker(connect={"returned": self._on_load_tracked_done, "errored": self._on_correction_worker_error})
        def _worker():
            stack = read_full_tracked_stack(tracked_path)
            cell_zavg = (
                np.asarray(tifffile.imread(str(cell_zavg_path)), dtype=np.float32)
                if cell_zavg_path and cell_zavg_path.exists() else None
            )
            nuc_zavg = (
                np.asarray(tifffile.imread(str(nuc_zavg_path)), dtype=np.float32)
                if nuc_zavg_path and nuc_zavg_path.exists() else None
            )
            return stack, cell_zavg, nuc_zavg

        _worker()

    def _on_load_tracked_done(self, result: tuple) -> None:
        stack, cell_zavg, nuc_zavg = result
        nt = stack.shape[0]
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = stack
        else:
            self.viewer.add_labels(stack, name=_TRACKED_LAYER)

        for zavg_data, layer_name, cmap in (
            (cell_zavg, _CELL_ZAVG_LAYER, "gray"),
            (nuc_zavg,  _NUC_ZAVG_LAYER,  "bop orange"),
        ):
            if zavg_data is None:
                continue
            if zavg_data.ndim == 2:
                broadcast_zavg = np.broadcast_to(zavg_data[np.newaxis], (nt,) + zavg_data.shape).copy()
            else:
                broadcast_zavg = zavg_data
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = broadcast_zavg
            else:
                self.viewer.add_image(broadcast_zavg, name=layer_name, colormap=cmap, blending="additive")

        self._set_correction_status(f"Loaded tracked stack {stack.shape} into napari.")
        layer = self.viewer.layers[_TRACKED_LAYER]
        self.correction_widget.activate_layer(layer)
        self.correction_section.expand()

    def set_selection_callback(self, fn) -> None:
        """Register a callback for nucleus correction label selection changes."""
        self.correction_widget.set_selection_callback(fn)

    def select_matching_nucleus_label(
        self,
        t: int,
        source_label: int,
        *,
        source_labels: np.ndarray | None = None,
    ) -> None:
        """Highlight the nucleus label that best overlaps a selected cell label."""
        if _TRACKED_LAYER not in self.viewer.layers:
            return
        if source_labels is None:
            if "Tracked: Cell" not in self.viewer.layers:
                return
            source_labels = np.asarray(self.viewer.layers["Tracked: Cell"].data)
        target_labels = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)
        matched_label = best_overlapping_label(target_labels, source_labels, t, source_label)
        self.correction_widget.select_label(t, matched_label, notify=False)

    def _on_reassign_ids(self) -> None:
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer loaded.")
            return
        stack = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)
        self._set_correction_status("Reassigning cell IDs to contiguous range…")

        @thread_worker(connect={"returned": self._on_reassign_ids_done, "errored": self._on_correction_worker_error})
        def _worker():
            unique_ids = np.unique(stack)
            unique_ids = unique_ids[unique_ids != 0]
            if unique_ids.size == 0:
                return stack, 0, {}
            lut = np.zeros(int(unique_ids.max()) + 1, dtype=np.uint32)
            old_to_new: dict[int, int] = {}
            for new_id, old_id in enumerate(unique_ids, start=1):
                lut[old_id] = new_id
                old_to_new[int(old_id)] = new_id
            return lut[stack], len(unique_ids), old_to_new

        _worker()

    def _on_reassign_ids_done(self, result: tuple) -> None:
        remapped, n_cells, old_to_new = result
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = remapped
        if self._pos_dir is not None and old_to_new:
            remap_validated_tracks(self._pos_dir, old_to_new)
        self._set_correction_status(f"Reassigned {n_cells} cell IDs to contiguous range 1–{n_cells}. Unsaved.")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Tracking & Correction
    # ──────────────────────────────────────────────────────────────────────────

    def _on_ultrack_mode_changed(self, mode: str) -> None:
        self.ultrack_iou_weight_spin.setEnabled(mode == "iou")

    def _set_resolve_prior_controls_enabled(self, _checked: bool | None = None) -> None:
        enabled = self.db_gen_use_validated_check.isChecked()
        for control in (
            self.ultrack_seed_weight_spin,
            self.ultrack_seed_space_spin,
            self.ultrack_seed_time_spin,
            self.ultrack_seed_window_spin,
        ):
            control.setEnabled(enabled)

    def _ultrack_config_from_controls(self) -> UltrackConfig:
        return UltrackConfig(
            seg_min_area=self.db_gen_min_area_spin.value(),
            seg_max_area=self.db_gen_max_area_spin.value(),
            seg_foreground_threshold=self.db_gen_fg_thr_spin.value(),
            seg_min_frontier=self.db_gen_min_frontier_spin.value(),
            seg_ws_hierarchy=self.db_gen_ws_hierarchy_combo.currentText(),
            seg_n_workers=self.db_gen_n_workers_spin.value(),
            max_distance=self.db_gen_max_dist_spin.value(),
            max_neighbors=self.db_gen_max_neighbors_spin.value(),
            linking_mode=self.db_gen_linking_mode_combo.currentText(),
            iou_weight=self.db_gen_iou_weight_spin.value(),
            quality_weight=self.db_gen_quality_weight_spin.value(),
            quality_exponent=self.db_gen_quality_exp_spin.value(),
            circularity_weight=self.db_gen_circularity_weight_spin.value(),
            power=self.ultrack_power_spin.value(),
            appear_weight=self.ultrack_appear_spin.value(),
            disappear_weight=self.ultrack_disappear_spin.value(),
            division_weight=self.ultrack_division_spin.value(),
        )

    def _on_run_ultrack(self) -> None:
        if self._pos_dir is None:
            self._set_ultrack_status("No project open.")
            return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._set_ultrack_status("data.db not found — run DB generation first.")
            return
        working_dir = self._ultrack_workdir()
        tracked_path = self._tracked_path()

        cfg = self._ultrack_config_from_controls()
        needs_validated_export = database_has_annotations(working_dir)
        validated_tracks = None
        tracked_labels = None
        if needs_validated_export:
            validated_tracks = read_validated_tracks(self._pos_dir)
            if not validated_tracks:
                self._set_ultrack_status(
                    "Annotated data.db requires validated tracks for ID-preserving export."
                )
                return
            if _TRACKED_LAYER not in self.viewer.layers:
                self._set_ultrack_status(
                    "Annotated data.db requires the current tracked layer for ID-preserving export."
                )
                return
            tracked_labels = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)

        self.ultrack_progress_bar.setRange(0, 100)
        self.ultrack_progress_bar.setVisible(True)
        self.ultrack_progress_bar.setValue(0)
        self._set_ultrack_status("Starting Ultrack solve…")
        self.run_ultrack_btn.setEnabled(False)
        self.ultrack_terminal_btn.setEnabled(False)

        @thread_worker(connect={
            "yielded":  self._on_ultrack_progress,
            "returned": self._on_run_ultrack_done,
            "errored":  self._on_ultrack_worker_error,
        })
        def _worker():
            for step, total, label in run_solve(working_dir, cfg, overwrite=True):
                yield ("solve", step, total, label)
            yield ("export", 0, 1, "Exporting tracked labels…")
            return export_tracked_labels(
                working_dir,
                cfg,
                tracked_path,
                validated_tracks=validated_tracks,
                tracked_labels=tracked_labels,
            )

        _worker()

    def _on_ultrack_progress(self, payload: tuple) -> None:
        stage, step, total, label = payload
        self._set_ultrack_status(f"[{stage}] {label}")
        if total > 0:
            self.ultrack_progress_bar.setValue(int(100 * step / total))

    def _on_run_ultrack_done(self, labels: np.ndarray | None) -> None:
        self.ultrack_progress_bar.setVisible(False)
        self.run_ultrack_btn.setEnabled(True)
        self.ultrack_terminal_btn.setEnabled(True)
        if labels is None:
            self._set_ultrack_status("Ultrack tracking failed (no output).")
            return
        # Normalize (T, 1, Y, X) → (T, Y, X)
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        nt = labels.shape[0]
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = labels
        else:
            self.viewer.add_labels(labels, name=_TRACKED_LAYER)
        layer = self.viewer.layers[_TRACKED_LAYER]
        self.correction_widget.activate_layer(layer)
        self._refresh_stage_files()
        self._set_ultrack_status(f"Tracking done: {nt} frame(s). Unsaved.")

    def _on_ultrack_terminal(self) -> None:
        import sys
        import tempfile

        if self._pos_dir is None:
            self._set_ultrack_status("No project open.")
            return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._set_ultrack_status("data.db not found — run DB generation first.")
            return
        working_dir = self._ultrack_workdir()
        tracked_path = self._tracked_path()

        cfg = self._ultrack_config_from_controls()
        needs_validated_export = database_has_annotations(working_dir)
        if needs_validated_export:
            validated_tracks = read_validated_tracks(self._pos_dir)
            if not validated_tracks:
                self._set_ultrack_status(
                    "Annotated data.db requires validated tracks for ID-preserving export."
                )
                return
            if tracked_path is None or not tracked_path.exists():
                self._set_ultrack_status(
                    "Annotated data.db requires current tracked labels for ID-preserving export."
                )
                return

        # NOTE: body must live under `if __name__ == "__main__":` because
        # Ultrack's linker uses spawn-based multiprocessing, which re-executes
        # this script in each child via runpy with run_name="__mp_main__".
        # Without the guard, every worker re-runs the full pipeline and races
        # the parent on the SQLite DB.
        python_code = (
            "import sys, pathlib\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))\n"
            "from cellflow.tracking_ultrack.config import TrackingConfig\n"
            "from cellflow.tracking_ultrack.solve import run_solve\n"
            "from cellflow.tracking_ultrack.export import export_tracked_labels\n"
            "from cellflow.database.tracked import read_full_tracked_stack\n"
            "from cellflow.database.validation import read_validated_tracks\n"
            "\n"
            "if __name__ == '__main__':\n"
            f"    pos_dir = pathlib.Path({str(self._pos_dir)!r})\n"
            f"    working_dir = pathlib.Path({str(working_dir)!r})\n"
            f"    tracked_path= pathlib.Path({str(tracked_path)!r})\n"
            f"    needs_validated_export = {bool(needs_validated_export)!r}\n"
            f"    cfg = TrackingConfig(\n"
            f"        power={cfg.power},\n"
            f"        appear_weight={cfg.appear_weight},\n"
            f"        disappear_weight={cfg.disappear_weight},\n"
            f"        division_weight={cfg.division_weight},\n"
            f"        solution_gap={cfg.solution_gap},\n"
            f"        time_limit={cfg.time_limit},\n"
            f"        window_size={cfg.window_size},\n"
            f"    )\n"
            "    print('[1/2] Solving ILP…', flush=True)\n"
            "    for step, total, label in run_solve(working_dir, cfg, overwrite=True):\n"
            "        print(f'  [{step}/{total}] {label}', flush=True)\n"
            "    print('[2/2] Exporting…', flush=True)\n"
            "    validated_tracks = read_validated_tracks(pos_dir) if needs_validated_export else None\n"
            "    tracked_labels = read_full_tracked_stack(tracked_path) if needs_validated_export else None\n"
            "    labels = export_tracked_labels(\n"
            "        working_dir,\n"
            "        cfg,\n"
            "        tracked_path,\n"
            "        validated_tracks=validated_tracks,\n"
            "        tracked_labels=tracked_labels,\n"
            "    )\n"
            f"    print(f'Done — {{labels.shape}} written to {{tracked_path}}', flush=True)\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="cellflow_ultrack_", delete=False
        ) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path)}"
        try:
            from cellflow.napari.utils import launch_in_terminal
            launch_in_terminal(cmd)
            self._set_ultrack_status("Ultrack command launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(cmd)
            self._set_ultrack_status("Copied Ultrack command to clipboard (terminal launch unavailable).")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Manual correction
    # ──────────────────────────────────────────────────────────────────────────

    def _on_dims_step_changed(self, event=None) -> None:
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        if self.ultrack_db_browser_section.is_expanded:
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, self._refresh_ultrack_db_browser)

    @staticmethod
    def _frame_view_2d(arr: np.ndarray, t: int) -> np.ndarray | None:
        """Return a 2D (Y, X) view of frame t from a (T, Y, X) or (T, 1, Y, X) stack."""
        if arr.ndim < 3 or t < 0 or t >= arr.shape[0]:
            return None
        v = arr[t]
        while v.ndim > 2:
            if v.shape[0] != 1:
                return None
            v = v[0]
        return v

    def _current_cell_ids(self, t: int) -> set[int]:
        """Return the set of non-zero cell IDs in the tracked layer at frame t."""
        if _TRACKED_LAYER not in self.viewer.layers:
            return set()
        layer = self.viewer.layers[_TRACKED_LAYER]
        frame = self._frame_view_2d(layer.data, t)
        if frame is None:
            return set()
        return set(int(v) for v in np.unique(frame)) - {0}

    def _refresh_validated_overlay(self) -> None:
        """Rebuild the green overlay layer from current frame's validated cells."""
        if self._pos_dir is None or _TRACKED_LAYER not in self.viewer.layers:
            if _VALIDATED_OVERLAY in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[_VALIDATED_OVERLAY])
            return
        tracked = self.viewer.layers[_TRACKED_LAYER]
        if tracked.data.ndim < 3:
            return
        t = self._current_t()
        if t >= tracked.data.shape[0]:
            return
        frame = self._frame_view_2d(tracked.data, t)
        if frame is None:
            return
        validated_ids = read_validated_cells_at_frame(self._pos_dir, t)
        overlay_exists = _VALIDATED_OVERLAY in self.viewer.layers
        if not validated_ids and not overlay_exists:
            # Nothing to draw and no overlay yet — skip creating one. This avoids
            # adding a layer during napari's own layer-insertion event chain
            # (which would re-enter and crash vispy's _reorder_layers).
            return
        if validated_ids:
            mask2d = np.isin(frame, list(validated_ids)).astype(np.uint8)
        else:
            mask2d = np.zeros(frame.shape, dtype=np.uint8)
        full = np.zeros(tracked.data.shape, dtype=np.uint8)
        full[t] = mask2d
        if overlay_exists:
            self.viewer.layers[_VALIDATED_OVERLAY].data = full
        else:
            from qtpy.QtCore import QTimer
            # Defer the add so we don't run inside napari's insert-event chain.
            QTimer.singleShot(0, lambda data=full: self._add_validated_overlay(data))

    def _add_validated_overlay(self, data: np.ndarray) -> None:
        if _VALIDATED_OVERLAY in self.viewer.layers:
            layer = self.viewer.layers[_VALIDATED_OVERLAY]
            layer.data = data
            layer.opacity = _VALIDATED_OVERLAY_OPACITY
            self._place_validated_overlay_below_spotlight()
            return
        ov = self.viewer.add_labels(
            data,
            name=_VALIDATED_OVERLAY,
            opacity=_VALIDATED_OVERLAY_OPACITY,
            colormap=direct_colormap({None: (0, 0, 0, 0), 1: "#00ff00"}),
        )
        self._place_validated_overlay_below_spotlight()
        # Send the active layer back to tracked so corrections still target it.
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[_TRACKED_LAYER]

    def _place_validated_overlay_below_spotlight(self) -> None:
        if _VALIDATED_OVERLAY not in self.viewer.layers or _SPOTLIGHT_LAYER not in self.viewer.layers:
            return
        validated_index = self.viewer.layers.index(_VALIDATED_OVERLAY)
        spotlight_index = self.viewer.layers.index(_SPOTLIGHT_LAYER)
        if validated_index > spotlight_index:
            self.viewer.layers.move(validated_index, spotlight_index)

    def _refresh_validation_counter(self) -> None:
        """Update 'N tracks validated, M cell-frames covered' label."""
        if self._pos_dir is None or _TRACKED_LAYER not in self.viewer.layers:
            self.validation_counter_lbl.setText("")
            return
        validated_tracks = read_validated_tracks(self._pos_dir)
        n_tracks = len(validated_tracks)
        n_cellframes = sum(len(frames) for frames in validated_tracks.values())
        self.validation_counter_lbl.setText(
            f"{n_tracks} track(s) validated, {n_cellframes} cell-frame(s) covered"
        )

    def _on_cells_edited(self, t: int, changed_ids: set[int]) -> None:
        """Callback registered with CorrectionWidget. Invalidate any edited cell IDs."""
        if self._pos_dir is None:
            return
        for cell_id in changed_ids:
            invalidate_track(self._pos_dir, cell_id)
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _frames_with_cell(self, cell_id: int) -> list[int]:
        """Return sorted list of frame indices where cell_id is present in the tracked layer."""
        if cell_id == 0 or _TRACKED_LAYER not in self.viewer.layers:
            return []
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim < 3:
            return []
        # Compare on the whole stack at once — np.any over the spatial axes is cheap.
        nt = layer.data.shape[0]
        spatial_axes = tuple(range(1, layer.data.ndim))
        present = np.any(layer.data == cell_id, axis=spatial_axes)
        return [int(t) for t in np.where(present)[0]]

    def _install_correction_shortcuts(self) -> None:
        specs = [
            ("A", lambda: self._on_extend(direction="backward")),
            ("D", lambda: self._on_extend(direction="forward")),
            ("Q", self._on_retrack_backward),
            ("E", self._on_retrack_forward),
        ]
        self._correction_shortcuts: list[QShortcut] = []
        for key, slot in specs:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.setEnabled(False)
            sc.activated.connect(slot)
            self._correction_shortcuts.append(sc)

    def _on_correction_mode_toggled(self, active: bool) -> None:
        for sc in self._correction_shortcuts:
            sc.setEnabled(active)

    def _kb_toggle_cell_validation(self, _viewer) -> None:
        if self._pos_dir is None:
            return
        sel = self.correction_widget._selected_label
        if not sel:
            self._set_correction_status("Validation toggle: no cell selected (left-click a cell first).")
            return
        t = self._current_t()
        if sel not in self._current_cell_ids(t):
            self._set_correction_status(f"Cell {sel} not present at t={t}.")
            return
        frames = self._frames_with_cell(sel)
        if not frames:
            return
        currently_validated = is_track_validated(self._pos_dir, sel)
        if currently_validated:
            invalidate_track(self._pos_dir, sel)
            self._set_correction_status(f"Cell {sel} invalidated across {len(frames)} frame(s).")
        else:
            validate_track(self._pos_dir, sel, frames)
            self._set_correction_status(f"Cell {sel} validated across {len(frames)} frame(s).")
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _on_remove_unvalidated_labels(self) -> None:
        if self._pos_dir is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer loaded.")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        data = np.asarray(layer.data)
        if data.ndim < 2:
            self._set_correction_status("Tracked layer has no image data.")
            return

        validated_tracks = read_validated_tracks(self._pos_dir)
        frame_count = int(data.shape[0]) if data.ndim >= 3 else 1
        changed_pixels = 0
        changed_frames = 0

        for t in range(frame_count):
            frame = self._frame_view_2d(data, t) if data.ndim >= 3 else data
            if frame is None:
                self._set_correction_status("Tracked layer must be a time-first 2D/3D stack.")
                return
            validated_ids = {
                cell_id
                for cell_id, frames in validated_tracks.items()
                if t in frames
            }
            remove_mask = frame != 0
            if validated_ids:
                remove_mask &= ~np.isin(frame, list(validated_ids))
            n_remove = int(np.count_nonzero(remove_mask))
            if not n_remove:
                continue
            frame[remove_mask] = 0
            changed_pixels += n_remove
            changed_frames += 1

        if not changed_pixels:
            self._set_correction_status("No unvalidated labels found.")
            return

        layer.refresh()
        if self.correction_widget._selected_label:
            current_t = self._current_t()
            if self.correction_widget._selected_label not in self._current_cell_ids(current_t):
                self.correction_widget.select_label(current_t, 0)
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        self._set_correction_status(
            f"Removed unvalidated labels in {changed_frames} frame(s), "
            f"{changed_pixels} px changed. Unsaved."
        )

    def _on_extend_backward(self) -> None:
        self._on_extend(direction="backward")

    def _on_extend_forward(self) -> None:
        self._on_extend(direction="forward")

    def _on_extend(self, direction: str) -> None:
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer loaded.")
            return

        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._set_correction_status(
                "Extend: data.db not found — run DB generation first."
            )
            return

        source_id = self.correction_widget._selected_label
        if not source_id:
            self._set_correction_status("Extend: no cell selected (left-click a cell first).")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        t = self._current_t()
        tracked = np.asarray(layer.data)
        T = tracked.shape[0]

        target_frame = t + (1 if direction == "forward" else -1)
        if direction == "forward" and t >= T - 1:
            self._set_correction_status("Already at last frame")
            return
        if direction == "backward" and t <= 0:
            self._set_correction_status("Already at first frame")
            return

        if not np.any(tracked[t] == source_id):
            self._set_correction_status(f"Cell {source_id} not present at t={t}")
            return

        validated_tracks = read_validated_tracks(self._pos_dir) if self._pos_dir is not None else {}
        result = extend_track_from_db(
            source_id=source_id,
            source_frame=t,
            direction=direction,
            tracked_labels=tracked,
            db_path=db_path,
            d_max=float(self.extend_max_dist_spin.value()),
            area_weight=float(self.extend_area_weight_spin.value()),
            iou_weight=float(self.extend_iou_weight_spin.value()),
            distance_weight=float(self.extend_distance_weight_spin.value()),
            overlap_penalty=float(self.extend_overlap_penalty_spin.value()),
            greedy_overwrite=self.extend_greedy_overwrite_check.isChecked(),
            validated_tracks=validated_tracks,
        )

        if result is None:
            self._set_correction_status(
                f"No candidate within {self.extend_max_dist_spin.value():g}px at t={target_frame}"
            )
            return

        assignments = result.assignments or ()
        if not assignments:
            assignments = (
                SimpleNamespace(cell_id=source_id, mask_2d=result.mask_2d),
            )

        frame = layer.data[result.target_frame]
        changed_ids = {int(assignment.cell_id) for assignment in assignments}
        for cell_id in changed_ids:
            frame[frame == cell_id] = 0

        if self.extend_greedy_overwrite_check.isChecked():
            for assignment in assignments:
                frame[assignment.mask_2d] = int(assignment.cell_id)
        else:
            for assignment in assignments:
                paintable = assignment.mask_2d & (frame == 0)
                frame[paintable] = int(assignment.cell_id)
        layer.refresh()

        step = list(self.viewer.dims.current_step)
        step[0] = result.target_frame
        self.viewer.dims.current_step = tuple(step)

        moved_text = (
            f", reassigned {len(changed_ids) - 1} conflict(s)"
            if len(changed_ids) > 1 else ""
        )
        self._set_correction_status(
            f"Extended cell {source_id} → t={result.target_frame}{moved_text} "
            f"(dist={result.centroid_distance:.1f}px, area={result.area_ratio:.2f}, "
            f"iou={result.centroid_corrected_iou:.2f}, overlap={result.existing_overlap:.2f})"
        )

    def _on_retrack_forward(self) -> None:
        if self._pos_dir is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer loaded.")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._set_correction_status("Tracked layer must be a stack of at least 2 frames.")
            return

        t0 = int(self.viewer.dims.current_step[0])
        if t0 >= layer.data.shape[0] - 1:
            self._set_correction_status("Already at last frame — nothing to retrack forward.")
            return

        T = layer.data.shape[0]
        stack = layer.data.copy()
        fully_validated = read_validated_frames(self._pos_dir)
        reserved_ids = set(read_validated_tracks(self._pos_dir))

        n_retracked = 0
        n_skipped = 0
        for t in range(t0 + 1, T):
            if t in fully_validated:
                n_skipped += 1
                continue
            ref = stack[t - 1]
            tgt = stack[t]
            locked = read_validated_cells_at_frame(self._pos_dir, t)
            stack[t] = retrack_frame_constrained(
                ref,
                tgt,
                locked,
                max_dist_px=float(self.retrack_max_dist_spin.value()),
                reserved_ids=reserved_ids,
            )
            n_retracked += 1

        layer.data = stack
        self._set_correction_status(
            f"Retracked forward from t={t0 + 1}: {n_retracked} frame(s) updated, "
            f"{n_skipped} fully-validated frame(s) skipped. Unsaved."
        )

    def _on_retrack_backward(self) -> None:
        if self._pos_dir is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer loaded.")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._set_correction_status("Tracked layer must be a stack of at least 2 frames.")
            return

        t0 = int(self.viewer.dims.current_step[0])
        if t0 <= 0:
            self._set_correction_status("Already at first frame — nothing to retrack backward.")
            return

        stack = layer.data.copy()
        fully_validated = read_validated_frames(self._pos_dir)
        reserved_ids = set(read_validated_tracks(self._pos_dir))

        n_retracked = 0
        n_skipped = 0
        for t in range(t0 - 1, -1, -1):
            if t in fully_validated:
                n_skipped += 1
                continue
            ref = stack[t + 1]
            tgt = stack[t]
            locked = read_validated_cells_at_frame(self._pos_dir, t)
            stack[t] = retrack_frame_constrained(
                ref,
                tgt,
                locked,
                max_dist_px=float(self.retrack_max_dist_spin.value()),
                reserved_ids=reserved_ids,
            )
            n_retracked += 1

        layer.data = stack
        self._set_correction_status(
            f"Retracked backward from t={t0 - 1}: {n_retracked} frame(s) updated, "
            f"{n_skipped} fully-validated frame(s) skipped. Unsaved."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # State persistence
    # ──────────────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "save_source":      self.save_source_check.isChecked(),
            "cellprob": {
                "min":       self.cp_min_spin.value(),
                "max":       self.cp_max_spin.value(),
                "step":      self.cp_step_spin.value(),
                "gamma_min": self.cp_gamma_min_spin.value(),
                "gamma_max": self.cp_gamma_max_spin.value(),
                "gamma_step": self.cp_gamma_step_spin.value(),
                "foreground_threshold": self.contour_fg_threshold_spin.value(),
                "flow_threshold": self.contour_flow_threshold_spin.value(),
            },
            "contour_filter": {
                "median_time": self.contour_filter_median_time_spin.value(),
                "median_space": self.contour_filter_median_space_spin.value(),
                "gauss_time": self.contour_filter_gauss_time_spin.value(),
                "gauss_space": self.contour_filter_gauss_space_spin.value(),
            },
            "db_generation": {
                "min_area":         self.db_gen_min_area_spin.value(),
                "max_area":         self.db_gen_max_area_spin.value(),
                "fg_threshold":     self.db_gen_fg_thr_spin.value(),
                "min_frontier":     self.db_gen_min_frontier_spin.value(),
                "ws_hierarchy":     self.db_gen_ws_hierarchy_combo.currentText(),
                "max_distance":     self.db_gen_max_dist_spin.value(),
                "max_neighbors":    self.db_gen_max_neighbors_spin.value(),
                "linking_mode":     self.db_gen_linking_mode_combo.currentText(),
                "iou_weight":       self.db_gen_iou_weight_spin.value(),
                "quality_weight":   self.db_gen_quality_weight_spin.value(),
                "quality_exponent": self.db_gen_quality_exp_spin.value(),
                "circularity_weight": self.db_gen_circularity_weight_spin.value(),
                "power":            self.db_gen_power_spin.value(),
                "n_workers":        self.db_gen_n_workers_spin.value(),
                "use_validated":    self.db_gen_use_validated_check.isChecked(),
                "seed_weight":      self.ultrack_seed_weight_spin.value(),
                "seed_sigma_space": self.ultrack_seed_space_spin.value(),
                "seed_tau_time":    self.ultrack_seed_time_spin.value(),
                "seed_max_dt":      self.ultrack_seed_window_spin.value(),
            },
            "extend": {
                "max_distance":     self.extend_max_dist_spin.value(),
                "area_weight":      self.extend_area_weight_spin.value(),
                "iou_weight":       self.extend_iou_weight_spin.value(),
                "distance_weight":  self.extend_distance_weight_spin.value(),
                "overlap_penalty":  self.extend_overlap_penalty_spin.value(),
                "greedy_overwrite": self.extend_greedy_overwrite_check.isChecked(),
            },
            "ultrack": {
                "max_partitions":   self.ultrack_max_partitions_spin.value(),
                "n_frames":         self.ultrack_n_frames_spin.value(),
                "appear_weight":    self.ultrack_appear_spin.value(),
                "disappear_weight": self.ultrack_disappear_spin.value(),
                "division_weight":  self.ultrack_division_spin.value(),
                "power":            self.ultrack_power_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if "save_source" in state:
            self.save_source_check.setChecked(state["save_source"])
        if "cellprob" in state:
            cp = state["cellprob"]
            if "min"        in cp: self.cp_min_spin.setValue(cp["min"])
            if "max"        in cp: self.cp_max_spin.setValue(cp["max"])
            if "step"       in cp: self.cp_step_spin.setValue(cp["step"])
            if "gamma_min"  in cp: self.cp_gamma_min_spin.setValue(cp["gamma_min"])
            if "gamma_max"  in cp: self.cp_gamma_max_spin.setValue(cp["gamma_max"])
            if "gamma_step" in cp: self.cp_gamma_step_spin.setValue(cp["gamma_step"])
            if "flow_threshold" in cp:
                self.contour_flow_threshold_spin.setValue(cp["flow_threshold"])
            if "foreground_threshold" in cp:
                self.contour_fg_threshold_spin.setValue(cp["foreground_threshold"])
        if "contour_filter" in state:
            cf = state["contour_filter"]
            if "median_time" in cf:
                self.contour_filter_median_time_spin.setValue(cf["median_time"])
            if "median_space" in cf:
                self.contour_filter_median_space_spin.setValue(cf["median_space"])
            if "gauss_time" in cf:
                self.contour_filter_gauss_time_spin.setValue(cf["gauss_time"])
            if "gauss_space" in cf:
                self.contour_filter_gauss_space_spin.setValue(cf["gauss_space"])
        if "db_generation" in state:
            dbg = state["db_generation"]
            if "min_area"         in dbg: self.db_gen_min_area_spin.setValue(dbg["min_area"])
            if "max_area"         in dbg: self.db_gen_max_area_spin.setValue(dbg["max_area"])
            if "fg_threshold"     in dbg: self.db_gen_fg_thr_spin.setValue(dbg["fg_threshold"])
            if "min_frontier"     in dbg: self.db_gen_min_frontier_spin.setValue(dbg["min_frontier"])
            if "ws_hierarchy"     in dbg:
                idx = self.db_gen_ws_hierarchy_combo.findText(dbg["ws_hierarchy"])
                if idx >= 0:
                    self.db_gen_ws_hierarchy_combo.setCurrentIndex(idx)
            if "max_distance"     in dbg: self.db_gen_max_dist_spin.setValue(dbg["max_distance"])
            if "max_neighbors"    in dbg: self.db_gen_max_neighbors_spin.setValue(dbg["max_neighbors"])
            if "linking_mode"     in dbg:
                idx = self.db_gen_linking_mode_combo.findText(dbg["linking_mode"])
                if idx >= 0:
                    self.db_gen_linking_mode_combo.setCurrentIndex(idx)
            if "iou_weight"       in dbg: self.db_gen_iou_weight_spin.setValue(dbg["iou_weight"])
            if "quality_weight"   in dbg: self.db_gen_quality_weight_spin.setValue(dbg["quality_weight"])
            if "quality_exponent" in dbg: self.db_gen_quality_exp_spin.setValue(dbg["quality_exponent"])
            if "circularity_weight" in dbg: self.db_gen_circularity_weight_spin.setValue(dbg["circularity_weight"])
            if "power"            in dbg: self.db_gen_power_spin.setValue(dbg["power"])
            if "n_workers"        in dbg: self.db_gen_n_workers_spin.setValue(dbg["n_workers"])
            if "use_validated"    in dbg: self.db_gen_use_validated_check.setChecked(dbg["use_validated"])
            if "seed_weight"      in dbg: self.ultrack_seed_weight_spin.setValue(dbg["seed_weight"])
            if "seed_sigma_space" in dbg: self.ultrack_seed_space_spin.setValue(dbg["seed_sigma_space"])
            if "seed_tau_time"    in dbg: self.ultrack_seed_time_spin.setValue(dbg["seed_tau_time"])
            if "seed_max_dt"      in dbg: self.ultrack_seed_window_spin.setValue(dbg["seed_max_dt"])
        if "extend" in state:
            ext = state["extend"]
            if "max_distance"    in ext: self.extend_max_dist_spin.setValue(ext["max_distance"])
            if "area_weight"     in ext: self.extend_area_weight_spin.setValue(ext["area_weight"])
            if "iou_weight"      in ext: self.extend_iou_weight_spin.setValue(ext["iou_weight"])
            if "distance_weight" in ext: self.extend_distance_weight_spin.setValue(ext["distance_weight"])
            if "overlap_penalty" in ext: self.extend_overlap_penalty_spin.setValue(ext["overlap_penalty"])
            if "greedy_overwrite" in ext: self.extend_greedy_overwrite_check.setChecked(ext["greedy_overwrite"])
        if "search" in state:
            pass  # Old propagator state — silently skip
        if "search_v2" in state:
            pass  # Old propagator v2 state — silently skip
        if "ultrack" in state:
            ul = state["ultrack"]
            if "min_area" in ul and (
                "db_generation" not in state or "min_area" not in state["db_generation"]
            ):
                self.db_gen_min_area_spin.setValue(ul["min_area"])
            if "max_partitions"   in ul: self.ultrack_max_partitions_spin.setValue(ul["max_partitions"])
            if "n_frames"         in ul: self.ultrack_n_frames_spin.setValue(ul["n_frames"])
            if "max_distance" in ul and (
                "db_generation" not in state or "max_distance" not in state["db_generation"]
            ):
                self.db_gen_max_dist_spin.setValue(ul["max_distance"])
            if "linking_mode" in ul and (
                "db_generation" not in state or "linking_mode" not in state["db_generation"]
            ):
                idx = self.db_gen_linking_mode_combo.findText(ul["linking_mode"])
                if idx >= 0:
                    self.db_gen_linking_mode_combo.setCurrentIndex(idx)
            if "iou_weight" in ul and (
                "db_generation" not in state or "iou_weight" not in state["db_generation"]
            ):
                self.db_gen_iou_weight_spin.setValue(ul["iou_weight"])
            if "appear_weight"    in ul: self.ultrack_appear_spin.setValue(ul["appear_weight"])
            if "disappear_weight" in ul: self.ultrack_disappear_spin.setValue(ul["disappear_weight"])
            if "division_weight"  in ul: self.ultrack_division_spin.setValue(ul["division_weight"])
            if "max_neighbors" in ul and (
                "db_generation" not in state or "max_neighbors" not in state["db_generation"]
            ):
                self.db_gen_max_neighbors_spin.setValue(ul["max_neighbors"])
            if "power"            in ul: self.ultrack_power_spin.setValue(ul["power"])
            if "resolve_from_validated" in ul and (
                "db_generation" not in state or "use_validated" not in state["db_generation"]
            ):
                self.db_gen_use_validated_check.setChecked(ul["resolve_from_validated"])
            if "quality_exponent" in ul and (
                "db_generation" not in state
                or "quality_exponent" not in state["db_generation"]
            ):
                self.db_gen_quality_exp_spin.setValue(ul["quality_exponent"])
            if "seed_weight" in ul and (
                "db_generation" not in state or "seed_weight" not in state["db_generation"]
            ):
                self.ultrack_seed_weight_spin.setValue(ul["seed_weight"])
            if "seed_sigma_space" in ul and (
                "db_generation" not in state or "seed_sigma_space" not in state["db_generation"]
            ):
                self.ultrack_seed_space_spin.setValue(ul["seed_sigma_space"])
            if "seed_tau_time" in ul and (
                "db_generation" not in state or "seed_tau_time" not in state["db_generation"]
            ):
                self.ultrack_seed_time_spin.setValue(ul["seed_tau_time"])
            if "seed_max_dt" in ul and (
                "db_generation" not in state or "seed_max_dt" not in state["db_generation"]
            ):
                self.ultrack_seed_window_spin.setValue(ul["seed_max_dt"])


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/ui_style.py
--------------------------------------------------------------------------------

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFormLayout, QGridLayout, QLabel, QSizePolicy

TINY_MARGIN = 2
SECTION_MARGIN = 4
TIGHT_SPACING = 4
DEFAULT_SPIN_WIDTH = 70
DEFAULT_FIELD_SPACING = 8
DEFAULT_ROW_SPACING = 4
DEFAULT_SWEEP_SPIN_WIDTH = 62
BLOCK_GRID_COLUMNS = 4


def _fixed_widget(widget, width=None):
    if width is not None:
        widget.setMaximumWidth(width)
    widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return widget


def compact_spinbox(widget, width=DEFAULT_SPIN_WIDTH):
    return _fixed_widget(widget, width)


def action_button(button, expand=False):
    horizontal_policy = (
        QSizePolicy.Policy.Expanding if expand else QSizePolicy.Policy.Fixed
    )
    button.setSizePolicy(horizontal_policy, QSizePolicy.Policy.Fixed)
    return button


def tiny_button(button):
    button.setStyleSheet("font-size: 8pt; padding: 1px 4px;")
    button.setSizePolicy(
        button.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Fixed
    )
    return button


def icon_button(button, width=24, height=None):
    button.setFixedWidth(width)
    if height is not None:
        button.setFixedHeight(height)
    return button


def muted_label(label, size_pt=8):
    label.setStyleSheet(f"color: palette(mid); font-size: {size_pt}pt;")
    return label


def status_label(label, size_pt=8, italic=False, muted=False):
    style = f"font-size: {size_pt}pt;"
    if muted:
        style += " color: palette(mid);"
    if italic:
        style += " font-style: italic;"
    label.setStyleSheet(style)
    return label


def danger_button(button):
    button.setStyleSheet(
        """
        QPushButton {
            background-color: #b00020;
            color: white;
        }
        QPushButton:hover {
            background-color: #c62828;
        }
        """
    )
    return button


def checked_success_button(button):
    button.setStyleSheet(
        """
        QPushButton:checked {
            background-color: #2e7d32;
            color: white;
            font-weight: bold;
        }
        """
    )
    return button


def compact_form_layout():
    layout = QFormLayout()
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
    layout.setHorizontalSpacing(DEFAULT_FIELD_SPACING)
    layout.setVerticalSpacing(DEFAULT_ROW_SPACING)
    return layout


def block_grid(horizontal_spacing=8, vertical_spacing=4):
    layout = QGridLayout()
    layout.setHorizontalSpacing(horizontal_spacing)
    layout.setVerticalSpacing(vertical_spacing)
    for col in range(BLOCK_GRID_COLUMNS):
        layout.setColumnStretch(col, 0)
    return layout


def two_column_parameter_grid(horizontal_spacing=12, vertical_spacing=4):
    return block_grid(horizontal_spacing, vertical_spacing)


def _block_label(text):
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return label


def _add_block_cell(grid, row, column, widget, span=1, alignment=None):
    if alignment is None:
        grid.addWidget(widget, row, column, 1, span)
    else:
        grid.addWidget(widget, row, column, 1, span, alignment)
    return widget


def add_block_pair_row(
    grid,
    row,
    left_label,
    left_widget,
    right_label=None,
    right_widget=None,
    field_width=70,
):
    left_label_widget = _block_label(left_label)
    _add_block_cell(grid, row, 0, left_label_widget)
    _add_block_cell(grid, row, 1, _fixed_widget(left_widget, field_width))

    right_label_widget = None
    if right_widget is not None:
        right_label_widget = _block_label(right_label or "")
        _add_block_cell(grid, row, 2, right_label_widget)
        _add_block_cell(grid, row, 3, _fixed_widget(right_widget, field_width))

    return left_label_widget, left_widget, right_label_widget, right_widget


def add_block_checkbox_row(grid, row, checkbox):
    _add_block_cell(
        grid,
        row,
        0,
        checkbox,
        span=BLOCK_GRID_COLUMNS,
        alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
    )
    return checkbox


def add_block_button_row(grid, row, *buttons):
    count = len(buttons)
    if count == 0:
        return ()
    if count == 1:
        placements = ((0, 4),)
    elif count == 2:
        placements = ((0, 2), (2, 2))
    elif count == 3:
        placements = ((0, 1), (1, 1), (2, 2))
    elif count == 4:
        placements = ((0, 1), (1, 1), (2, 1), (3, 1))
    else:
        raise ValueError("add_block_button_row supports at most four buttons")

    for button, (column, span) in zip(buttons, placements):
        action_button(button, expand=True)
        _add_block_cell(
            grid,
            row,
            column,
            button,
            span=span,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
    return buttons


def add_parameter_grid_row(grid, row, column, label_text, field):
    base_col = column * 2
    label = _block_label(label_text)
    _add_block_cell(grid, row, base_col, label)
    _add_block_cell(grid, row, base_col + 1, _fixed_widget(field))
    return label, field


def sweep_parameter_grid(
    horizontal_spacing=8,
    vertical_spacing=4,
    spin_width=DEFAULT_SWEEP_SPIN_WIDTH,
):
    layout = block_grid(horizontal_spacing, vertical_spacing)
    layout.setColumnMinimumWidth(1, spin_width)
    layout.setColumnMinimumWidth(2, spin_width)
    layout.setColumnMinimumWidth(3, spin_width)

    layout.addWidget(QLabel(""), 0, 0)
    for col, text in enumerate(("min", "max", "step"), start=1):
        header = QLabel(text)
        header.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(header, 0, col)
    return layout


def add_sweep_parameter_row(
    grid,
    row,
    label_text,
    min_widget,
    max_widget,
    step_widget,
    spin_width=DEFAULT_SWEEP_SPIN_WIDTH,
):
    label = _block_label(label_text)
    _add_block_cell(grid, row, 0, label)
    _add_block_cell(grid, row, 1, compact_spinbox(min_widget, spin_width))
    _add_block_cell(grid, row, 2, compact_spinbox(max_widget, spin_width))
    _add_block_cell(grid, row, 3, compact_spinbox(step_widget, spin_width))
    return label, min_widget, max_widget, step_widget


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/utils.py
--------------------------------------------------------------------------------

"""UI utility functions for the CellFlow napari plugin."""
from __future__ import annotations

import platform
import subprocess


def launch_in_terminal(command: str) -> None:
    """Open a new OS terminal and run *command* inside it."""
    system = platform.system()
    if system == "Linux":
        # Using a more generic terminal if kitty is not available might be better,
        # but for now let's stick to what was there or try a few common ones.
        try:
            subprocess.Popen(["kitty", "--", "bash", "-c", f"{command}; exec bash"])
        except FileNotFoundError:
            try:
                subprocess.Popen(["gnome-terminal", "--", "bash", "-c", f"{command}; exec bash"])
            except FileNotFoundError:
                subprocess.Popen(["xterm", "-e", f"bash -c '{command}; exec bash'"])
    elif system == "Darwin":
        escaped = command.replace("'", "'\\''")
        apple_script = f'tell application "Terminal" to do script "{escaped}"'
        subprocess.Popen(["osascript", "-e", apple_script])
    elif system == "Windows":
        subprocess.Popen(f'start cmd /k "{command}"', shell=True)
    else:
        raise RuntimeError(f"Unsupported platform '{system}'")


--------------------------------------------------------------------------------
FILE: src/cellflow/napari/widgets.py
--------------------------------------------------------------------------------

"""Shared reusable Qt widgets for the CellFlow napari plugin."""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .ui_style import (
    SECTION_MARGIN,
    TIGHT_SPACING,
    TINY_MARGIN,
    icon_button,
    muted_label,
    status_label,
)


class CollapsibleSection(QWidget):
    """A labelled section with a toggle button that shows/hides its inner widget."""

    def __init__(
        self,
        title: str,
        inner: QWidget,
        expanded: bool = False,
        parent: QWidget | None = None,
        title_color: str = "white",
    ) -> None:
        super().__init__(parent)
        self._inner = inner
        self._base_title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, TINY_MARGIN, 0, TINY_MARGIN)
        layout.setSpacing(0)

        # Header toggle button
        self._toggle = QToolButton()
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setText(self._qt_display_text(title))
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.setStyleSheet(
            f"QToolButton {{ font-weight: bold; font-size: 10pt; border: none; "
            f"padding: 2px; color: {title_color}; }}"
        )
        self._toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self._toggle)

        # White-bordered frame that wraps inner content when expanded
        self._content_frame = QFrame()
        self._content_frame.setObjectName("collapsible_content")
        self._content_frame.setFrameShape(QFrame.NoFrame)
        self._content_frame.setStyleSheet(
            "QFrame#collapsible_content { border: 1px solid #666666; "
            "border-radius: 4px; margin: 0px 2px 2px 2px; }"
        )
        frame_layout = QVBoxLayout(self._content_frame)
        frame_layout.setContentsMargins(
            SECTION_MARGIN, SECTION_MARGIN, SECTION_MARGIN, SECTION_MARGIN
        )
        frame_layout.setSpacing(TINY_MARGIN)
        frame_layout.addWidget(inner)

        self._content_frame.setVisible(expanded)
        layout.addWidget(self._content_frame)

        # Always Preferred policy — height is driven by scroll area's minimumHeight
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        if expanded:
            QTimer.singleShot(0, self._notify_layout_change)

    def set_title(self, title: str) -> None:
        """Update the header text."""
        self._base_title = title
        self._toggle.setText(self._qt_display_text(title))

    @property
    def title(self) -> str:
        return self._base_title

    @property
    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def expand(self) -> None:
        self._toggle.setChecked(True)

    def collapse(self) -> None:
        self._toggle.setChecked(False)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content_frame.setVisible(checked)
        QTimer.singleShot(0, self._notify_layout_change)

    @staticmethod
    def _qt_display_text(title: str) -> str:
        """Escape mnemonic markers so literal ampersands render correctly."""
        return title.replace("&", "&&")

    def _notify_layout_change(self) -> None:
        """Propagate geometry changes up the nested collapsible chain."""
        self.updateGeometry()
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, CollapsibleSection) and parent.is_expanded:
                parent.updateGeometry()
                QTimer.singleShot(0, parent._notify_layout_change)
                return
            parent.updateGeometry()
            parent = parent.parent()


# ---------------------------------------------------------------------------
# Pipeline file status rows
# ---------------------------------------------------------------------------

class _PipelineFileRow(QWidget):
    """One pipeline file status row: icon | rel-path | info | [load btn]"""

    def __init__(
        self,
        rel_path: str,
        display_name: str,
        loadable: str | None = None,
        viewer=None,
    ):
        super().__init__()
        self._rel_path = rel_path
        self._loadable = loadable or self._infer_load_kind(rel_path)
        self._full_path: "Path | None" = None
        self._viewer = viewer

        lay = QHBoxLayout(self)
        lay.setContentsMargins(
            SECTION_MARGIN, TINY_MARGIN, SECTION_MARGIN, TINY_MARGIN
        )
        lay.setSpacing(TIGHT_SPACING)

        self._icon_lbl = QLabel("○")
        self._icon_lbl.setFixedWidth(14)
        self._icon_lbl.setAlignment(Qt.AlignCenter)
        muted_label(self._icon_lbl, size_pt=9)
        lay.addWidget(self._icon_lbl)

        name_lbl = QLabel(rel_path)
        name_lbl.setFixedWidth(200)
        status_label(name_lbl)
        name_lbl.setToolTip(display_name)
        lay.addWidget(name_lbl)

        self._info_lbl = QLabel("—")
        self._info_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        status_label(self._info_lbl)
        lay.addWidget(self._info_lbl)

        self._load_btn = QPushButton("↑")
        icon_button(self._load_btn, width=18, height=18)
        self._load_btn.clicked.connect(self._on_load_clicked)
        self._load_btn.setEnabled(False)
        self._load_btn.setToolTip(self._load_tooltip())
        # Hide the button entirely when no viewer is wired in or file is not napari-loadable.
        self._load_btn.setVisible(viewer is not None and self._loadable is not None)
        lay.addWidget(self._load_btn)

    def set_present(self, info_text: str) -> None:
        self._icon_lbl.setText("✓")
        self._icon_lbl.setStyleSheet("font-size: 9pt; font-weight: bold; color: #4CAF50;")
        self._info_lbl.setText(info_text)
        status_label(self._info_lbl)
        self._update_load_button()

    def set_missing(self) -> None:
        self._icon_lbl.setText("✗")
        muted_label(self._icon_lbl, size_pt=9)
        self._info_lbl.setText("missing")
        muted_label(self._info_lbl)
        self._full_path = None
        self._load_btn.setEnabled(False)
        self._load_btn.setToolTip(self._load_tooltip(missing=True))

    def set_no_project(self) -> None:
        self._icon_lbl.setText("○")
        muted_label(self._icon_lbl, size_pt=9)
        self._info_lbl.setText("—")
        muted_label(self._info_lbl)
        self._full_path = None
        self._load_btn.setEnabled(False)
        self._load_btn.setToolTip(self._load_tooltip(no_project=True))

    def _update_load_button(self) -> None:
        if self._full_path is None or not self._full_path.exists():
            self._load_btn.setEnabled(False)
            self._load_btn.setToolTip(self._load_tooltip(missing=True))
            return

        if self._loadable in {"tracked", "labels", "tiff"}:
            self._load_btn.setEnabled(True)
            self._load_btn.setToolTip(self._load_tooltip())
            return

        self._load_btn.setEnabled(False)
        self._load_btn.setToolTip("No direct napari load action for this file.")

    def _on_load_clicked(self) -> None:
        if self._full_path is None or not self._full_path.exists():
            return

        self._load_file_into_viewer()

    def _load_file_into_viewer(self) -> None:
        viewer = self._viewer if self._viewer is not None else self._find_viewer()
        if viewer is None:
            return

        import tifffile

        data = tifffile.imread(str(self._full_path))
        layer_name = self._layer_name()
        use_labels = self._loadable in {"tracked", "labels"}

        if use_labels:
            if layer_name in viewer.layers:
                try:
                    viewer.layers[layer_name].data = data
                    return
                except Exception:
                    viewer.layers.remove(viewer.layers[layer_name])
            viewer.add_labels(data, name=layer_name)
        else:
            colormap = self._pick_colormap()
            if layer_name in viewer.layers:
                try:
                    viewer.layers[layer_name].data = data
                    return
                except Exception:
                    viewer.layers.remove(viewer.layers[layer_name])
            viewer.add_image(data, name=layer_name, colormap=colormap)

    def _pick_colormap(self) -> str:
        rel = self._rel_path
        name = Path(rel).name
        if rel.startswith("0_input/") or name.endswith(("_zavg.tif", "_3dt.tif")):
            return "gray"
        if (
            rel.startswith("1_cellpose/")
            or rel in (
                "2_nucleus/contour_maps.tif",
                "3_cell/filtered_flow_mag.tif",
            )
            or (rel.startswith("2_nucleus/") and name.startswith("foreground_"))
            or (rel.startswith("3_cell/") and name.startswith("foreground_"))
        ):
            return "inferno"
        return "gray"

    def _find_viewer(self):
        widget = self.parentWidget()
        while widget is not None:
            viewer = getattr(widget, "viewer", None)
            if viewer is not None and hasattr(viewer, "add_image") and hasattr(viewer, "add_labels"):
                return viewer
            widget = widget.parentWidget()
        return None

    def _layer_name(self) -> str:
        return Path(self._rel_path).with_suffix("").as_posix().replace("/", "_")

    def _load_tooltip(self, *, missing: bool = False, no_project: bool = False) -> str:
        if no_project:
            return "No project open."
        if missing:
            return "File is missing."
        if self._loadable in {"tracked", "labels"}:
            return "Load labels into napari."
        if self._loadable == "tiff":
            return "Load into napari viewer."
        return "No direct napari load action for this file."

    @staticmethod
    def _infer_load_kind(rel_path: str) -> str | None:
        name = Path(rel_path).name
        if name == "tracked_labels.tif":
            return "tracked"
        if name.endswith("_labels.tif") or ("labels" in name and name.endswith((".tif", ".tiff"))):
            return "labels"
        if name.endswith((".tif", ".tiff")):
            return "tiff"
        return None


def _file_info(path: "Path") -> str:
    """Return a concise shape/dtype string for a pipeline output file."""
    if path.is_dir():
        return "Directory"
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(path) as tif:
                shape = tif.series[0].shape if tif.series else None
            if shape:
                return "×".join(str(d) for d in shape)
        except Exception:
            pass
        return "TIFF"
    if suffix in (".h5", ".hdf5"):
        try:
            import h5py
            shapes = []
            def _collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    shapes.append(f"{name}: " + "×".join(str(d) for d in obj.shape))
            with h5py.File(path, "r") as f:
                f.visititems(_collect)
            if shapes:
                return "; ".join(shapes[:2]) + ("…" if len(shapes) > 2 else "")
        except Exception:
            pass
        kb = path.stat().st_size // 1024
        return f"{kb} KB"
    return f"{path.stat().st_size // 1024} KB"


class PipelineFilesWidget(QWidget):
    """Compact file-status display for pipeline-stage widgets."""

    def __init__(
        self,
        groups: list[tuple[str, list[tuple[str, str]]]],
        parent: QWidget | None = None,
        viewer=None,
    ) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._rows: list[_PipelineFileRow] = []

        for group_label, entries in groups:
            if group_label:
                hdr = QLabel(group_label)
                hdr.setStyleSheet(
                    "font-size: 7pt; font-weight: bold; padding: 1px 4px;"
                    " background: palette(alternateBase); color: palette(mid);"
                )
                lay.addWidget(hdr)
            for rel_path, display_name in entries:
                row = _PipelineFileRow(rel_path, display_name, loadable=None, viewer=viewer)
                self._rows.append(row)
                lay.addWidget(row)

    def refresh(self, pos_dir: "Path" | None) -> None:
        """Update all rows to reflect current on-disk state."""
        if pos_dir is None:
            for row in self._rows:
                row.set_no_project()
            return
        for row in self._rows:
            full_path = pos_dir / row._rel_path
            if full_path.exists():
                row._full_path = full_path
                row.set_present(_file_info(full_path))
            else:
                row.set_missing()


--------------------------------------------------------------------------------
FILE: src/cellflow/segmentation/__init__.py
--------------------------------------------------------------------------------

"""Nucleus segmentation via watershed on Cellpose probability maps."""
from __future__ import annotations

from cellflow.segmentation.flow_following import (
    FlowFollowingParams,
    compute_filtered_flow_vectors,
    compute_flow_following_movie,
    compute_flow_following_frame,
    build_consensus_boundary_flow_following,
)

from cellflow.segmentation.contour_filtering import (
    ContourFilterParams,
    compute_filtered_contour_maps,
)

from cellflow.segmentation.cell_label_icm import (
    CellICMState,
    CellLabelICMParams,
    commit_labels,
    initialize_icm,
    refine_icm,
)

import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

_LABEL_DTYPE = np.uint32


def apply_gamma(logits: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma-correct Cellpose probability logits: sigmoid → power → logit."""
    if gamma == 1.0:
        return logits
    probs = 1.0 / (1.0 + np.exp(-logits))
    probs = np.clip(np.power(probs, gamma), 1e-7, 1 - 1e-7)
    return np.log(probs / (1.0 - probs))


def _validate_foreground_gamma(gamma: float) -> float:
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    return gamma


def _validate_foreground_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return threshold


def _apply_post_average_gamma(score: np.ndarray, gamma: float) -> np.ndarray:
    score = np.clip(score, 0.0, 1.0).astype(np.float32, copy=False)
    if gamma == 1.0:
        return score
    return np.power(score, gamma).astype(np.float32)


def _normalize_foreground_score(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    lo = float(np.min(score))
    hi = float(np.max(score))
    if hi <= lo:
        return np.zeros_like(score, dtype=np.float32)
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _flow_dp_magnitude_stack(data: np.ndarray) -> tuple[np.ndarray, bool]:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 3:
        return np.abs(data), False
    if data.ndim == 4:
        if data.shape[-1] in (2, 3):
            return np.sqrt(np.sum(data * data, axis=-1)).astype(np.float32), False
        if data.shape[1] in (2, 3):
            return np.sqrt(np.sum(data * data, axis=1)).astype(np.float32), False
        return np.abs(data), True
    if data.ndim == 5:
        if data.shape[2] in (2, 3):
            axis = 2
        elif data.shape[-1] in (2, 3):
            axis = -1
        else:
            raise ValueError(f"Unsupported flow_dp shape {data.shape}")
        return np.sqrt(np.sum(data * data, axis=axis)).astype(np.float32), True
    raise ValueError(f"Unsupported flow_dp shape {data.shape}")


def foreground_score_stack(data, source: str, gamma: float = 1.0) -> np.ndarray:
    """Return a foreground score image or time stack from probability or flow-DP data."""
    gamma = _validate_foreground_gamma(gamma)
    source_key = str(source).lower()
    arr = np.asarray(data, dtype=np.float32)

    if source_key == "probability":
        if arr.ndim == 3:
            score = 1.0 / (1.0 + np.exp(-arr.mean(axis=0)))
        elif arr.ndim == 4:
            score = 1.0 / (1.0 + np.exp(-arr.mean(axis=1)))
        else:
            raise ValueError(f"Unsupported probability shape {arr.shape}")
        return _apply_post_average_gamma(score, gamma)

    if source_key == "flow_dp":
        magnitude, has_time_axis = _flow_dp_magnitude_stack(arr)
        if has_time_axis:
            score = magnitude.mean(axis=1)
            normalized = np.empty_like(score, dtype=np.float32)
            for t in range(score.shape[0]):
                normalized[t] = _normalize_foreground_score(score[t])
        else:
            normalized = _normalize_foreground_score(magnitude.mean(axis=0))
        return _apply_post_average_gamma(normalized, gamma)

    raise ValueError(f"Unsupported foreground source {source!r}")


def foreground_mask_stack(
    data,
    source: str,
    threshold: float = 0.5,
    gamma: float = 1.0,
) -> np.ndarray:
    """Return a uint8 foreground mask with values 0/1."""
    threshold = _validate_foreground_threshold(threshold)
    score = foreground_score_stack(data, source, gamma=gamma)
    return (score >= threshold).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class ContourWatershedParams:
    """Parameters for contour-map watershed hypothesis generation."""

    seed_distance: int = 10
    foreground_threshold: float = 0.5
    ridge_threshold: float = 0.5
    min_size: int = 0
    min_circularity: float = 0.0
    noise_scale: float = 0.0
    noise_blur_sigma: float = 0.0
    run_index: int = 0

    def to_dict(self) -> dict[str, object]:
        return {"method": "contour_watershed", **asdict(self)}


@dataclass(frozen=True, slots=True)
class CellposeFlowHypothesisParams:
    """Parameters for native Cellpose flow-based mask generation (no sweep)."""

    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.0   # 0 = disabled; >0 removes masks with high flow error
    min_size: int = 15
    niter: int = 200

    def to_dict(self) -> dict[str, object]:
        return {"method": "cellpose_flow", **asdict(self)}


@dataclass(frozen=True, slots=True)
class NucleusHypothesisParams:
    """One parameter set for nucleus hypothesis generation."""

    basin: str = "prob"
    threshold_pct: float = 30.0
    compactness: float = 0.0
    smooth_sigma: float = 0.5
    seed_source: str = "auto"
    seed_distance: int = 5
    min_size: int = 0
    min_circularity: float = 0.0
    z_slice: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_01(arr: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if lo is None:
        lo = float(np.min(arr))
    if hi is None:
        hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, np.nextafter(np.float32(1.0), np.float32(0.0)))
    return scaled.astype(np.float32)


def _flow_magnitude(dp: np.ndarray) -> np.ndarray:
    """Compute L2 magnitude from a DP stack."""
    dp = np.asarray(dp, dtype=np.float32)
    if dp.ndim == 2:
        return np.abs(dp)
    if dp.ndim == 3:
        if dp.shape[0] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=0)).astype(np.float32)
        if dp.shape[-1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=-1)).astype(np.float32)
        return np.abs(dp).astype(np.float32)
    if dp.ndim >= 4:
        if dp.shape[1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=1)).astype(np.float32)
        if dp.shape[-1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=-1)).astype(np.float32)
    raise ValueError(f"Unsupported DP shape for magnitude: {dp.shape}")


def _remove_small_labels(labels: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return labels
    ids, counts = np.unique(labels, return_counts=True)
    small = ids[(ids > 0) & (counts < min_size)]
    if small.size == 0:
        return labels
    out = labels.copy()
    out[np.isin(labels, small)] = 0
    return out


def _remove_low_circularity_labels(labels: np.ndarray, min_circularity: float) -> np.ndarray:
    """Remove labels whose 4π·area/perimeter² is below min_circularity (0 = keep all)."""
    if min_circularity <= 0.0:
        return labels
    from skimage.measure import regionprops

    # Work on a 2D projection if labels is 3D with a single Z
    squeezed = labels.squeeze() if labels.ndim == 3 and labels.shape[0] == 1 else labels
    if squeezed.ndim != 2:
        return labels  # can't compute perimeter on >2D, skip

    import math
    remove = []
    for prop in regionprops(squeezed.astype(np.int32)):
        perimeter = prop.perimeter
        if perimeter < 1e-6:
            remove.append(prop.label)
            continue
        circularity = 4.0 * math.pi * prop.area / (perimeter ** 2)
        if circularity < min_circularity:
            remove.append(prop.label)

    if not remove:
        return labels
    out = labels.copy()
    out[np.isin(labels, remove)] = 0
    return out


def _fill_and_close_labels(labels: np.ndarray) -> np.ndarray:
    """Fill interior holes per label."""
    from scipy.ndimage import binary_fill_holes

    out = np.zeros_like(labels)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        coords = np.nonzero(labels == label_id)
        if not coords or coords[0].size == 0:
            continue
        slices = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coords)
        filled = binary_fill_holes(labels[slices] == label_id)
        out_view = out[slices]
        out_view[filled] = label_id
    return out


def _centroid_markers_2d(labels: np.ndarray) -> np.ndarray:
    """Place one marker pixel at the centroid of each 2D label."""
    labels = np.asarray(labels)
    out = np.zeros_like(labels)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        coords = np.argwhere(labels == label_id)
        centroid = coords.mean(axis=0)
        seed_yx = np.rint(centroid).astype(np.int64)
        if (
            seed_yx[0] < 0
            or seed_yx[0] >= labels.shape[0]
            or seed_yx[1] < 0
            or seed_yx[1] >= labels.shape[1]
            or labels[seed_yx[0], seed_yx[1]] != label_id
        ):
            distances = np.sum((coords - centroid) ** 2, axis=1)
            seed_yx = coords[int(np.argmin(distances))]
        out[int(seed_yx[0]), int(seed_yx[1])] = label_id
    return out


def centroid_markers_from_labels(labels: np.ndarray) -> np.ndarray:
    """Return one centroid seed pixel per non-zero label.

    For a 2D label image, each label is replaced by a single marker pixel at
    its rounded centroid. If the rounded centroid falls outside the label, the
    closest pixel belonging to that label is used instead. For a 3D stack, the
    operation is applied independently to each first-axis plane, matching
    time-first ``(T, Y, X)`` tracked nuclear labels.
    """
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return _centroid_markers_2d(labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected 2D labels or time-first 3D stack, got {labels.shape}")
    out = np.zeros_like(labels)
    for t in range(labels.shape[0]):
        out[t] = _centroid_markers_2d(labels[t])
    return out


def _peak_local_max_markers(basin: np.ndarray, min_distance: int) -> np.ndarray:
    from scipy.ndimage import label as nd_label
    from skimage.feature import peak_local_max

    coords = peak_local_max(basin, min_distance=max(1, min_distance), exclude_border=False)
    mask = np.zeros(basin.shape, dtype=bool)
    if coords.size:
        mask[coords[:, 0], coords[:, 1]] = True
    markers, _ = nd_label(mask)
    return markers.astype(np.int32)


def compute_hypothesis_labels(
    prob: np.ndarray,
    dp: np.ndarray | None,
    markers: np.ndarray | None,
    params: NucleusHypothesisParams,
    *,
    global_lo: float | None = None,
    global_hi: float | None = None,
) -> np.ndarray:
    """Compute a single nucleus hypothesis label image for one 2D slice.

    global_lo/global_hi: min/max of the basin computed over the full 3D volume,
    so threshold_pct is a fraction of the whole-frame dynamic range, not per-slice.
    """
    from skimage.segmentation import watershed

    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"Expected 2D probability slice, got shape {prob.shape}")

    if params.basin == "prob":
        basin = 1.0 / (1.0 + np.exp(-prob))  # logits → probabilities
    elif params.basin == "flow_mag":
        if dp is None:
            raise ValueError("flow_mag basin requested but no DP array provided")
        basin = _flow_magnitude(dp)
        if basin.ndim != 2:
            raise ValueError(f"Expected 2D flow magnitude slice, got shape {basin.shape}")
    else:
        raise ValueError(f"Unknown basin={params.basin!r}; expected 'prob' or 'flow_mag'")

    basin = _normalize_01(basin, lo=global_lo, hi=global_hi)
    if params.smooth_sigma > 0:
        basin = gaussian_filter(basin, sigma=float(params.smooth_sigma))

    if markers is None:
        markers = _peak_local_max_markers(basin, params.seed_distance)
    else:
        markers = np.asarray(markers, dtype=np.int32)
        if markers.shape != basin.shape:
            raise ValueError(
                f"Markers shape {markers.shape} does not match basin shape {basin.shape}"
            )

    from scipy.ndimage import binary_fill_holes

    threshold = float(params.threshold_pct) / 100.0
    mask = binary_fill_holes((basin >= threshold) | (markers > 0))

    labels = watershed(
        -basin,
        markers=markers,
        mask=mask,
        compactness=float(params.compactness),
        watershed_line=False,
    )
    result = _fill_and_close_labels(np.asarray(labels, dtype=_LABEL_DTYPE))
    result = _remove_small_labels(result, params.min_size)
    return _remove_low_circularity_labels(result, params.min_circularity)


def compute_cellpose_flow_hypothesis(
    prob_3d: np.ndarray,
    dp_3d: np.ndarray,
    params: CellposeFlowHypothesisParams,
) -> np.ndarray:
    """Run Cellpose native mask generation independently per z-slice.

    prob_3d: (Z, Y, X) logits from Cellpose (flows[2])
    dp_3d:   (Z, 2, Y, X) flow fields from Cellpose (flows[1])
    Returns: (Z, Y, X) uint32
    """
    try:
        import torch
        from cellpose.dynamics import compute_masks
    except ImportError as exc:
        raise ImportError(
            "cellpose and torch must be installed to use flow-based hypothesis generation"
        ) from exc

    prob_3d = np.asarray(prob_3d, dtype=np.float32)
    dp_3d = np.asarray(dp_3d, dtype=np.float32)
    if prob_3d.ndim != 3:
        raise ValueError(f"Expected (Z, Y, X) prob, got {prob_3d.shape}")
    if dp_3d.ndim != 4 or dp_3d.shape[1] != 2:
        raise ValueError(f"Expected (Z, 2, Y, X) dp, got {dp_3d.shape}")

    n_foreground = int(np.sum(prob_3d > params.cellprob_threshold))
    if n_foreground == 0:
        raise RuntimeError(
            f"No foreground pixels found: all prob values <= cellprob_threshold={params.cellprob_threshold}. "
            f"Prob range: [{float(prob_3d.min()):.2f}, {float(prob_3d.max()):.2f}]. "
            "Try lowering cellprob_threshold."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_z = prob_3d.shape[0]
    out = np.zeros_like(prob_3d, dtype=_LABEL_DTYPE)
    cp_min_size = params.min_size if params.min_size > 0 else -1
    for z in range(n_z):
        result = compute_masks(
            dp_3d[z],
            prob_3d[z],
            cellprob_threshold=params.cellprob_threshold,
            flow_threshold=params.flow_threshold,
            min_size=cp_min_size,
            niter=params.niter,
            do_3D=False,
            device=device,
        )
        # Cellpose ≥3.x returns just the mask array; older versions return (mask, p, tr).
        masks = result[0] if isinstance(result, tuple) else result
        out[z] = np.asarray(masks, dtype=_LABEL_DTYPE)
    return out


def build_consensus_boundary(
    prob_3d: np.ndarray,
    dp_3d: np.ndarray,
    cellprob_thresholds: list[float],
    gamma: float = 1.0,
    flow_threshold: float = 0.0,
    reduction: str = "mean",
    *,
    mask_callback: Callable[[np.ndarray, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce mask boundaries and occupancy over (threshold × z-slice).

    prob_3d: (Z, Y, X) logits  dp_3d: (Z, 2, Y, X)
    reduction: "mean" averages across all (threshold × z-slice) combinations;
               "max" takes the per-pixel maximum instead.
    mask_callback: optional sink called as mask_callback(masks_zyx, thresh_idx) after each threshold.
    Returns: (boundary, foreground) both (Y, X) float32.
    """
    try:
        import torch
        from cellpose.dynamics import compute_masks
        from skimage.segmentation import find_boundaries
    except ImportError as exc:
        raise ImportError("cellpose, torch, and scikit-image required") from exc

    prob_3d = apply_gamma(np.asarray(prob_3d, dtype=np.float32), gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_z = prob_3d.shape[0]
    accum = np.zeros(prob_3d.shape[1:], dtype=np.float32)
    foreground_accum = np.zeros(prob_3d.shape[1:], dtype=np.float32)
    n_total = 0

    for i_thresh, thresh in enumerate(cellprob_thresholds):
        z_masks: list[np.ndarray] = []
        for z in range(n_z):
            result = compute_masks(
                dp_3d[z], prob_3d[z],
                cellprob_threshold=float(thresh),
                flow_threshold=float(flow_threshold),
                niter=200,
                do_3D=False,
                device=device,
            )
            masks = result[0] if isinstance(result, tuple) else result
            masks_arr = np.asarray(masks)
            boundary_slice = find_boundaries(masks_arr, mode="inner").astype(np.float32)
            fg_slice = (masks_arr > 0).astype(np.float32)
            if reduction == "max":
                np.maximum(accum, boundary_slice, out=accum)
                np.maximum(foreground_accum, fg_slice, out=foreground_accum)
            else:
                accum += boundary_slice
                foreground_accum += fg_slice
            n_total += 1
            if mask_callback is not None:
                z_masks.append(np.asarray(masks_arr, dtype=np.uint32))
        if mask_callback is not None:
            mask_callback(np.stack(z_masks), i_thresh)

    if reduction == "max":
        return accum, foreground_accum
    boundary = accum / n_total if n_total > 0 else accum
    foreground = foreground_accum / n_total if n_total > 0 else foreground_accum
    return boundary, foreground


def build_consensus_boundary_2d(
    prob_yx: np.ndarray,
    dp_cyx: np.ndarray,
    cellprob_thresholds: list[float],
    flow_threshold: float = 0.0,
    reduction: str = "mean",
    niter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Build consensus boundary from a Z-averaged probability map and 2D flow vectors.

    prob_yx:  (Y, X) Cellpose probability logits — already Z-projected and gamma-corrected.
    dp_cyx:   (2, Y, X) flow vectors (e.g. from filtered_dp).
    Returns:  (boundary, foreground) both (Y, X) float32.
    """
    try:
        import torch
        from cellpose.dynamics import compute_masks
        from skimage.segmentation import find_boundaries
    except ImportError as exc:
        raise ImportError("cellpose, torch, and scikit-image required") from exc

    prob_yx = np.asarray(prob_yx, dtype=np.float32)
    dp_cyx = np.asarray(dp_cyx, dtype=np.float32)
    if prob_yx.ndim != 2:
        raise ValueError(f"Expected (Y, X) prob, got {prob_yx.shape}")
    if dp_cyx.ndim != 3 or dp_cyx.shape[0] != 2:
        raise ValueError(f"Expected (2, Y, X) dp, got {dp_cyx.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accum = np.zeros(prob_yx.shape, dtype=np.float32)
    foreground_accum = np.zeros(prob_yx.shape, dtype=np.float32)

    for thresh in cellprob_thresholds:
        result = compute_masks(
            dp_cyx,
            prob_yx,
            cellprob_threshold=float(thresh),
            flow_threshold=float(flow_threshold),
            niter=int(niter),
            do_3D=False,
            device=device,
        )
        masks = result[0] if isinstance(result, tuple) else result
        masks_arr = np.asarray(masks)
        boundary_slice = find_boundaries(masks_arr, mode="inner").astype(np.float32)
        fg_slice = (masks_arr > 0).astype(np.float32)
        if reduction == "max":
            np.maximum(accum, boundary_slice, out=accum)
            np.maximum(foreground_accum, fg_slice, out=foreground_accum)
        else:
            accum += boundary_slice
            foreground_accum += fg_slice

    n = len(cellprob_thresholds)
    if reduction != "max" and n > 0:
        accum /= n
        foreground_accum /= n

    return accum, foreground_accum


def compute_masks_for_threshold(
    dp_3d: np.ndarray, prob_3d: np.ndarray, threshold: float
) -> np.ndarray:
    """Run Cellpose mask generation for a specific threshold across all z-slices."""
    try:
        import torch
        from cellpose.dynamics import compute_masks
    except ImportError as exc:
        raise ImportError("cellpose and torch required") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_z = prob_3d.shape[0]
    out = np.zeros(prob_3d.shape, dtype=_LABEL_DTYPE)
    for z in range(n_z):
        result = compute_masks(
            dp_3d[z],
            prob_3d[z],
            cellprob_threshold=float(threshold),
            flow_threshold=0.0,
            niter=200,
            do_3D=False,
            device=device,
        )
        masks = result[0] if isinstance(result, tuple) else result
        out[z] = np.asarray(masks, dtype=_LABEL_DTYPE)
    return out


def compute_cellpose_foreground_masks(
    prob_tzyx: np.ndarray,
    filtered_dp_tcyx: np.ndarray,
    *,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.0,
    min_size: int = 15,
    niter: int = 200,
    progress_cb: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Generate binary cell foreground masks with Cellpose dynamics.

    prob_tzyx is Cellpose probability logits shaped (T, Z, Y, X), or a single
    volume shaped (Z, Y, X). filtered_dp_tcyx must be the filtered flow stack
    produced by the cell workflow, shaped (T, 2, Y, X).
    """
    prob = np.asarray(prob_tzyx, dtype=np.float32)
    if prob.ndim == 3:
        prob = prob[np.newaxis]
    if prob.ndim != 4:
        raise ValueError(
            f"Expected probability shape (T, Z, Y, X) or (Z, Y, X), got {prob.shape}"
        )

    filtered_dp = np.asarray(filtered_dp_tcyx, dtype=np.float32)
    if filtered_dp.ndim != 4 or filtered_dp.shape[1] != 2:
        raise ValueError(
            f"Expected filtered flow shape (T, 2, Y, X), got {filtered_dp.shape}"
        )
    if prob.shape[0] != filtered_dp.shape[0] or prob.shape[2:] != filtered_dp.shape[2:]:
        raise ValueError(
            "Cellpose probability and filtered flow shapes do not match: "
            f"probability {prob.shape}, filtered flow {filtered_dp.shape}"
        )

    try:
        import torch
        from cellpose.dynamics import compute_masks
    except ImportError as exc:
        raise ImportError(
            "cellpose and torch must be installed to generate Cellpose foreground masks"
        ) from exc

    prob_tyx = prob.mean(axis=1).astype(np.float32, copy=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = np.zeros(prob_tyx.shape, dtype=np.uint8)

    for t in range(prob_tyx.shape[0]):
        result = compute_masks(
            filtered_dp[t],
            prob_tyx[t],
            cellprob_threshold=float(cellprob_threshold),
            flow_threshold=float(flow_threshold),
            min_size=int(min_size),
            niter=int(niter),
            do_3D=False,
            device=device,
        )
        masks = result[0] if isinstance(result, tuple) else result
        out[t] = (np.asarray(masks) > 0).astype(np.uint8)
        if progress_cb is not None:
            progress_cb(t + 1, prob_tyx.shape[0])

    return out


@dataclass(frozen=True, slots=True)
class SeededWatershedParams:
    """Parameters for nucleus-seeded watershed cell hypothesis generation."""

    basin: str = "prob"
    foreground_threshold: float = 0.5
    compactness: float = 0.0

    def __post_init__(self) -> None:
        warnings.warn(
            "SeededWatershedParams is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

    def to_dict(self) -> dict[str, object]:
        return {"method": "seeded_watershed", **asdict(self)}


def compute_seeded_watershed(
    prob_2d: np.ndarray,
    dp_2d: np.ndarray | None,
    seeds_2d: np.ndarray,
    params: SeededWatershedParams,
) -> np.ndarray:
    """Seeded watershed using nucleus labels as markers for one 2D z-slice.

    Foreground mask is always derived from sigmoid(prob_2d). Seeds whose
    centroid falls outside the mask are silently dropped by the watershed.
    """
    warnings.warn(
        "compute_seeded_watershed is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    from scipy.ndimage import binary_fill_holes
    from skimage.segmentation import watershed

    prob_2d = np.asarray(prob_2d, dtype=np.float32)
    seeds_2d = np.asarray(seeds_2d, dtype=np.int32)

    sigmoid_prob = 1.0 / (1.0 + np.exp(-prob_2d))
    fg_mask = binary_fill_holes(sigmoid_prob > params.foreground_threshold)

    if params.basin == "prob":
        basin = sigmoid_prob
    elif params.basin == "flow_mag":
        if dp_2d is None:
            raise ValueError("flow_mag basin requires a dp array")
        basin = _flow_magnitude(dp_2d)
        if basin.ndim != 2:
            raise ValueError(f"Expected 2D flow magnitude slice, got shape {basin.shape}")
    else:
        raise ValueError(f"Unknown basin={params.basin!r}; expected 'prob' or 'flow_mag'")

    labels = watershed(
        -basin,
        markers=seeds_2d,
        mask=fg_mask,
        compactness=float(params.compactness),
        watershed_line=False,
    )
    return np.asarray(labels, dtype=_LABEL_DTYPE)


def compute_contour_watershed(
    boundary: np.ndarray,
    foreground_mask: np.ndarray,
    params: ContourWatershedParams,
) -> np.ndarray:
    """Run seeded watershed on a consensus boundary image and binary foreground mask.

    Seeds are placed at EDT maxima of fg_mask & (boundary < ridge_threshold),
    so contour ridges separating touching cells drive seed placement rather than
    foreground intensity peaks.

    boundary:   (Y, X) float32 — high at cell borders
    foreground_mask: (Y, X) binary — nonzero pixels are allowed segmentation area
    Returns:    (Y, X) uint32 label image
    """
    from scipy.ndimage import label as nd_label
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    boundary = np.asarray(boundary, dtype=np.float32)
    foreground_mask = np.asarray(foreground_mask)
    if foreground_mask.shape != boundary.shape:
        raise ValueError(
            f"Foreground mask shape {foreground_mask.shape} does not match boundary shape {boundary.shape}"
        )
    fg_mask = foreground_mask > 0

    boundary_pre = np.asarray(boundary, dtype=np.float32).copy()

    # Apply correlated noise perturbation
    if params.noise_scale > 0:
        noise = np.random.normal(0, params.noise_scale, boundary_pre.shape)
        if params.noise_blur_sigma > 0:
            noise = gaussian_filter(noise, sigma=params.noise_blur_sigma)
        boundary_pre = np.clip(boundary_pre + noise, 0, 1)

    boundary_pre[boundary_pre < params.foreground_threshold] = 0

    from scipy.ndimage import distance_transform_edt

    # Carve strong contour ridges out of the mask so touching cells become
    # separate connected components before seeding.
    core = fg_mask & (boundary_pre < params.ridge_threshold)
    edt = distance_transform_edt(core)

    coords = peak_local_max(
        edt,
        min_distance=max(1, int(params.seed_distance)),
        threshold_abs=1.0,
        exclude_border=False,
    )
    marker_mask = np.zeros(boundary_pre.shape, dtype=bool)
    if coords.size:
        marker_mask[coords[:, 0], coords[:, 1]] = True
    markers, _ = nd_label(marker_mask)

    # Watershed floods fg_mask (not core) so basins fill back over carved ridges.
    labels = watershed(boundary_pre, markers=markers, mask=fg_mask, watershed_line=False)
    result = _fill_and_close_labels(np.asarray(labels, dtype=_LABEL_DTYPE))
    result = _remove_small_labels(result, params.min_size)
    return _remove_low_circularity_labels(result, params.min_circularity)


--------------------------------------------------------------------------------
FILE: src/cellflow/segmentation/cell_label_icm.py
--------------------------------------------------------------------------------

"""Cell label ICM solver with staged API (Initialize → Refine → Commit).

Provides a decomposed pipeline for cell boundary optimization:

- ``initialize_icm``: compute geodesic unary costs and spatial/temporal
  pairwise weights, then build initial labels (nucleus-only, argmin, or
  watershed).  Returns a :class:`CellICMState` that caches all
  energy-landscape data, plus the initial label array.

- ``refine_icm``: run *N* Iterated Conditional Modes (ICM) sweeps on an
  existing label array using a previously computed ``CellICMState``.
  Can be called repeatedly for incremental refinement, interleaved with
  manual corrections in the napari viewer.

- ``commit_labels``: write the current label array to a TIFF file.

Backward-compatible monolithic entry points are preserved:

- ``segment_cells_icm``: run the full pipeline on in-memory arrays.
- ``run_cell_icm_from_pos_dir``: load TIFFs from disk, run full pipeline.

The ICM solver uses a sequential Gauss-Seidel raster sweep (Numba JIT)
with 8-connected spatial neighbours and spatiotemporal face-diagonal edges.
"""
from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

import h5py
import numba
import numpy as np
import tifffile
from skimage.graph import MCP_Geometric

__all__ = [
    "CellLabelICMParams",
    "CellICMState",
    "initialize_icm",
    "refine_icm",
    "commit_labels",
    "segment_cells_icm",
    "run_cell_icm_from_pos_dir",
]

# ── Constants ────────────────────────────────────────────────────────────────

_INF: float = 1e9
_DIAG_SCALE: float = 1.0 / math.sqrt(2.0)


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class CellLabelICMParams:
    """Parameters for the cell label ICM pipeline."""

    alpha_unary: float = 4.0
    """Contour weight in the geodesic cost field: ``1 + alpha_unary * contour``."""

    lambda_s: float = 1.0
    """Spatial pairwise Potts weight."""

    beta_s: float = 5.0
    """Contour sensitivity in spatial pairwise: ``exp(-beta_s * avg_contour)``."""

    lambda_t: float = 1.0
    """Temporal pairwise Potts weight."""

    n_iters: int = 3
    """Number of ICM sweeps."""

    min_round_flips: int = 0
    """Stop early if a round has fewer flips than this."""

    lambda_area: float = 0.0
    """Weight for per-label frame-to-frame area-change penalty."""

    gamma_unary: float = 0.0
    """Weight for ``(1 - foreground_score)`` added to the geodesic cost field."""

    init_mode: str = "nuclei"
    """Label initialisation: ``"nuclei"`` | ``"unary"`` | ``"watershed"``."""

    n_workers: int = 1
    """Parallel worker processes for geodesic unary computation.
    1 = sequential.  Values > 1 use fork-based multiprocessing
    to compute frames in parallel."""


@dataclass
class CellICMState:
    """Cached energy-landscape data for incremental ICM refinement.

    Created by :func:`initialize_icm` and consumed by :func:`refine_icm`.
    All arrays are stored as their solver-ready dtypes (float32 / uint32 /
    bool).  The state is **read-only** once created — it describes the energy
    landscape, not the current solution.
    """

    fg_mask: np.ndarray = field(repr=False)
    """(T, Y, X) bool — foreground mask (includes nucleus pixels)."""

    nuc_tracks: np.ndarray = field(repr=False)
    """(T, Y, X) uint32 — nucleus track IDs (0 = no nucleus)."""

    label_ids: np.ndarray = field(repr=False)
    """(K,) uint32 — sorted global set of label (track) IDs."""

    unary_dense: np.ndarray = field(repr=False)
    """(T, Y, X, K) float32 — dense unary cost array.  Dead / background
    entries are ``_INF``."""

    # Spatial pairwise weights — all (T, Y, X) float32
    h: np.ndarray = field(repr=False)
    v: np.ndarray = field(repr=False)
    dr: np.ndarray = field(repr=False)
    dl: np.ndarray = field(repr=False)

    # Temporal pairwise weights — all (T, Y, X) float32
    tw: np.ndarray = field(repr=False)
    tw_ty_dn: np.ndarray = field(repr=False)
    tw_ty_up: np.ndarray = field(repr=False)
    tw_tx_r: np.ndarray = field(repr=False)
    tw_tx_l: np.ndarray = field(repr=False)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.fg_mask.shape  # type: ignore[return-value]

    @property
    def n_labels(self) -> int:
        return len(self.label_ids)


# ── Internal: pairwise weights ───────────────────────────────────────────────

def _compute_pairwise_weights(
    fg_mask: np.ndarray,
    boundary_signal: np.ndarray,
    lambda_s: float,
    beta_s: float,
    lambda_t: float,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Compute per-pixel spatial + temporal Potts pairwise weights.

    Returns ``(h, v, dr, dl, tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l)``
    — nine ``(T, Y, X)`` float32 arrays.
    """
    T, Y, X = fg_mask.shape
    fg = fg_mask.astype(bool)
    c = boundary_signal

    h = np.zeros((T, Y, X), dtype=np.float32)
    h[:, :, :-1] = (
        lambda_s * np.exp(-beta_s * 0.5 * (c[:, :, :-1] + c[:, :, 1:]))
        * (fg[:, :, :-1] & fg[:, :, 1:])
    ).astype(np.float32)

    v = np.zeros((T, Y, X), dtype=np.float32)
    v[:, :-1, :] = (
        lambda_s * np.exp(-beta_s * 0.5 * (c[:, :-1, :] + c[:, 1:, :]))
        * (fg[:, :-1, :] & fg[:, 1:, :])
    ).astype(np.float32)

    dr = np.zeros((T, Y, X), dtype=np.float32)
    dr[:, :-1, :-1] = (
        _DIAG_SCALE * lambda_s
        * np.exp(-beta_s * 0.5 * (c[:, :-1, :-1] + c[:, 1:, 1:]))
        * (fg[:, :-1, :-1] & fg[:, 1:, 1:])
    ).astype(np.float32)

    dl = np.zeros((T, Y, X), dtype=np.float32)
    dl[:, :-1, 1:] = (
        _DIAG_SCALE * lambda_s
        * np.exp(-beta_s * 0.5 * (c[:, :-1, 1:] + c[:, 1:, :-1]))
        * (fg[:, :-1, 1:] & fg[:, 1:, :-1])
    ).astype(np.float32)

    tw = np.zeros((T, Y, X), dtype=np.float32)
    if T > 1:
        tw[:-1, :, :] = (lambda_t * (fg[:-1] & fg[1:])).astype(np.float32)

    tw_ty_dn = np.zeros((T, Y, X), dtype=np.float32)
    tw_ty_up = np.zeros((T, Y, X), dtype=np.float32)
    tw_tx_r = np.zeros((T, Y, X), dtype=np.float32)
    tw_tx_l = np.zeros((T, Y, X), dtype=np.float32)
    if T > 1:
        tw_ty_dn[:-1, :-1, :] = (
            _DIAG_SCALE * lambda_t * (fg[:-1, :-1, :] & fg[1:, 1:, :])
        ).astype(np.float32)
        tw_ty_up[:-1, 1:, :] = (
            _DIAG_SCALE * lambda_t * (fg[:-1, 1:, :] & fg[1:, :-1, :])
        ).astype(np.float32)
        tw_tx_r[:-1, :, :-1] = (
            _DIAG_SCALE * lambda_t * (fg[:-1, :, :-1] & fg[1:, :, 1:])
        ).astype(np.float32)
        tw_tx_l[:-1, :, 1:] = (
            _DIAG_SCALE * lambda_t * (fg[:-1, :, 1:] & fg[1:, :, :-1])
        ).astype(np.float32)

    return h, v, dr, dl, tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l


# ── Internal: geodesic unaries ───────────────────────────────────────────────

def _compute_frame_geodesic(
    contours_t: np.ndarray,
    fg_t: np.ndarray,
    nuc_t: np.ndarray,
    label_ids: np.ndarray,
    alpha_unary: float,
    fg_scores_t: np.ndarray | None = None,
    gamma_unary: float = 0.0,
) -> dict[int, np.ndarray]:
    """Compute normalised geodesic unaries for all alive labels in one frame.

    Returns ``{k: (Y, X) float32}`` — normalised geodesic distance per label.
    Dead / background entries are ``_INF``.

    The MCP object is created once and reused for all labels in the frame
    (the cost field depends only on the contour map, not the label).
    """
    Y, X = fg_t.shape

    # Build cost field — shared across all labels
    cost_field = np.full((Y, X), np.inf, dtype=np.float32)
    c = 1.0 + alpha_unary * contours_t[fg_t]
    if gamma_unary != 0.0 and fg_scores_t is not None:
        c = c + gamma_unary * (1.0 - np.clip(fg_scores_t[fg_t], 0.0, 1.0))
    cost_field[fg_t] = c

    alive = [int(k) for k in label_ids if np.any(nuc_t == k)]
    if not alive:
        return {}

    # Single MCP object reused for all labels in this frame
    mcp = MCP_Geometric(cost_field, fully_connected=True)

    raw: dict[int, np.ndarray] = {}
    for k in alive:
        starts = [
            tuple(int(v) for v in coord)
            for coord in np.argwhere(nuc_t == k)
        ]
        cum, _ = mcp.find_costs(starts)
        d = cum.astype(np.float32)
        d[~fg_t] = np.inf
        raw[k] = d

    # Per-frame median normalisation
    all_finite = np.concatenate([d[np.isfinite(d)] for d in raw.values()])
    med = float(np.median(all_finite)) if all_finite.size > 0 else 1.0
    if med <= 0.0:
        med = 1.0

    result: dict[int, np.ndarray] = {}
    for k, d in raw.items():
        nd = d / med
        nd[~np.isfinite(nd)] = _INF
        result[k] = nd

    # Hard nucleus anchors within this frame
    for k in alive:
        k_pix = nuc_t == k
        if not k_pix.any():
            continue
        for j in alive:
            if j != k and j in result:
                result[j][k_pix] = _INF

    return result


# ── Parallel worker globals (fork-inherited, no pickling) ────────────────────

_MP_CONTOURS: np.ndarray | None = None
_MP_FG_MASK: np.ndarray | None = None
_MP_NUC_TRACKS: np.ndarray | None = None
_MP_LABEL_IDS: np.ndarray | None = None
_MP_ALPHA_UNARY: float = 0.0
_MP_GAMMA_UNARY: float = 0.0
_MP_FG_SCORES: np.ndarray | None = None


def _geodesic_frame_worker(t: int) -> tuple[int, dict[int, np.ndarray]]:
    """Multiprocessing worker: compute geodesic unaries for frame *t*.

    Reads from module-level globals set before ``Pool`` creation —
    inherited via fork-COW on Linux, zero pickling overhead.
    """
    result = _compute_frame_geodesic(
        _MP_CONTOURS[t],
        _MP_FG_MASK[t],
        _MP_NUC_TRACKS[t],
        _MP_LABEL_IDS,
        _MP_ALPHA_UNARY,
        fg_scores_t=_MP_FG_SCORES[t] if _MP_FG_SCORES is not None else None,
        gamma_unary=_MP_GAMMA_UNARY,
    )
    return t, result


def _compute_geodesic_unaries(
    nuc_tracks: np.ndarray,
    fg_mask: np.ndarray,
    contours: np.ndarray,
    label_ids: np.ndarray,
    alpha_unary: float,
    *,
    foreground_scores: np.ndarray | None = None,
    gamma_unary: float = 0.0,
    n_workers: int = 1,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Compute normalised geodesic unary costs for each alive (frame, label).

    When ``n_workers > 1``, frames are computed in parallel using
    fork-based multiprocessing (Linux).  Each worker inherits the input
    arrays via copy-on-write — only the frame index is sent through the
    pipe per task.
    """
    T = fg_mask.shape[0]
    _report = progress_cb or (lambda msg: None)

    if n_workers > 1:
        return _compute_geodesic_unaries_parallel(
            nuc_tracks, fg_mask, contours, label_ids, alpha_unary,
            foreground_scores=foreground_scores,
            gamma_unary=gamma_unary,
            n_workers=n_workers,
            progress_cb=progress_cb,
        )

    # ── Sequential path ──────────────────────────────────────────────
    unary: dict[tuple[int, int], np.ndarray] = {}
    for t in range(T):
        frame_result = _compute_frame_geodesic(
            contours[t], fg_mask[t], nuc_tracks[t], label_ids,
            alpha_unary,
            fg_scores_t=(
                foreground_scores[t] if foreground_scores is not None else None
            ),
            gamma_unary=gamma_unary,
        )
        for k, d in frame_result.items():
            unary[(t, k)] = d
        if progress_cb and ((t + 1) % 10 == 0 or t + 1 == T):
            alive = len(frame_result)
            _report(f"Geodesic unaries: frame {t + 1}/{T}, {alive} alive")

    return unary


def _compute_geodesic_unaries_parallel(
    nuc_tracks: np.ndarray,
    fg_mask: np.ndarray,
    contours: np.ndarray,
    label_ids: np.ndarray,
    alpha_unary: float,
    *,
    foreground_scores: np.ndarray | None = None,
    gamma_unary: float = 0.0,
    n_workers: int = 4,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Parallel geodesic unary computation across frames.

    Uses fork-based multiprocessing: input arrays are set as module-level
    globals and inherited by worker processes via COW.  Only the frame
    index (a single int) is sent per task; results (sparse dicts of
    float32 arrays) are returned through the pipe.
    """
    global _MP_CONTOURS, _MP_FG_MASK, _MP_NUC_TRACKS, _MP_LABEL_IDS
    global _MP_ALPHA_UNARY, _MP_GAMMA_UNARY, _MP_FG_SCORES

    T = fg_mask.shape[0]
    _report = progress_cb or (lambda msg: None)
    n_workers = min(n_workers, T, os.cpu_count() or 1)

    # Set globals before fork — workers inherit via COW
    _MP_CONTOURS = contours
    _MP_FG_MASK = fg_mask
    _MP_NUC_TRACKS = nuc_tracks
    _MP_LABEL_IDS = label_ids
    _MP_ALPHA_UNARY = alpha_unary
    _MP_GAMMA_UNARY = gamma_unary
    _MP_FG_SCORES = foreground_scores

    _report(f"Computing geodesic unaries ({n_workers} workers, {T} frames)...")

    unary: dict[tuple[int, int], np.ndarray] = {}
    done = 0

    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for t, frame_result in pool.imap_unordered(
                _geodesic_frame_worker, range(T)
            ):
                for k, d in frame_result.items():
                    unary[(t, k)] = d
                done += 1
                if progress_cb and (done % 10 == 0 or done == T):
                    _report(
                        f"Geodesic unaries: {done}/{T} frames "
                        f"({len(frame_result)} labels in frame {t})"
                    )
    finally:
        # Clear globals — don't keep references to large arrays
        _MP_CONTOURS = None
        _MP_FG_MASK = None
        _MP_NUC_TRACKS = None
        _MP_LABEL_IDS = None
        _MP_FG_SCORES = None

    return unary


def _apply_nucleus_anchors(
    unary: dict[tuple[int, int], np.ndarray],
    nuc_tracks: np.ndarray,
    label_ids: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Re-apply hard nucleus anchors: cost=0 for own label, INF for others."""
    T = nuc_tracks.shape[0]
    label_list = [int(k) for k in label_ids]
    cached_by_t: dict[int, list[int]] = {}
    for (t, j) in unary:
        cached_by_t.setdefault(int(t), []).append(int(j))

    for t in range(T):
        alive = [k for k in label_list if int((nuc_tracks[t] == k).sum()) > 0]
        cached = cached_by_t.get(t, [])
        for k in alive:
            k_pix = nuc_tracks[t] == k
            if (t, k) in unary:
                unary[(t, k)][k_pix] = 0.0
            for j in cached:
                if j != k:
                    unary[(t, j)][k_pix] = _INF
    return unary


def _dict_to_dense_unary(
    unary: dict[tuple[int, int], np.ndarray],
    fg_mask: np.ndarray,
    label_ids: np.ndarray,
) -> np.ndarray:
    """Convert sparse ``{(t, k): (Y, X)}`` to ``(T, Y, X, K)`` float32."""
    T, Y, X = fg_mask.shape
    K = len(label_ids)
    dense = np.full((T, Y, X, K), _INF, dtype=np.float32)
    for ki, k in enumerate(label_ids):
        for t in range(T):
            u = unary.get((t, int(k)))
            if u is not None:
                dense[t, :, :, ki] = u
    return dense


# ── Internal: HDF5 unary cache ───────────────────────────────────────────────

def _unary_cache_key(
    shape: tuple[int, int, int],
    alpha_unary: float,
    gamma_unary: float,
) -> str:
    raw = f"{shape[0]}x{shape[1]}x{shape[2]}_a{alpha_unary:g}_g{gamma_unary:g}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"unary_{digest}"


def _unary_cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.h5"


def _read_unary_cache(
    cache_dir: Path,
    key: str,
) -> dict[tuple[int, int], np.ndarray] | None:
    path = _unary_cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        unary: dict[tuple[int, int], np.ndarray] = {}
        with h5py.File(path, "r") as f:
            grp = f["unaries"]
            for name in grp:
                t_s, k_s = name.split("_", 1)
                unary[(int(t_s), int(k_s))] = grp[name][...].astype(
                    np.float32, copy=False
                )
        return unary
    except Exception:
        return None


def _write_unary_cache(
    cache_dir: Path,
    key: str,
    unary: dict[tuple[int, int], np.ndarray],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _unary_cache_path(cache_dir, key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with h5py.File(tmp, "w") as f:
            grp = f.create_group("unaries")
            grp.attrs["cache_key"] = key
            for (t, k), arr in unary.items():
                grp.create_dataset(
                    f"{int(t)}_{int(k)}",
                    data=np.asarray(arr, dtype=np.float32),
                    compression="lzf",
                )
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


# ── Internal: initialisation helpers ─────────────────────────────────────────

def _unary_elevation_from_dense(
    unary_dense: np.ndarray,
    fg_mask: np.ndarray,
) -> np.ndarray:
    T, Y, X = fg_mask.shape
    elevation = np.min(unary_dense, axis=3)
    for t in range(T):
        finite_fg = fg_mask[t] & (elevation[t] < _INF)
        if finite_fg.any():
            cap = float(np.max(elevation[t][finite_fg]))
            inf_fg = fg_mask[t] & (elevation[t] >= _INF)
            elevation[t][inf_fg] = cap
    return elevation


def _watershed_init(
    fg_mask: np.ndarray,
    nuc_tracks: np.ndarray,
    elevation: np.ndarray,
) -> np.ndarray:
    from skimage.segmentation import watershed

    T, Y, X = fg_mask.shape
    labels = np.zeros((T, Y, X), dtype=np.uint32)
    for t in range(T):
        markers = nuc_tracks[t].astype(np.int32)
        labels[t] = watershed(
            elevation[t].astype(np.float32),
            markers=markers,
            mask=fg_mask[t],
        ).astype(np.uint32)
    return labels


def _argmin_init(
    unary_dense: np.ndarray,
    fg_mask: np.ndarray,
    label_ids: np.ndarray,
) -> np.ndarray:
    best_ki = np.argmin(unary_dense, axis=3)
    return np.where(fg_mask, label_ids[best_ki], 0).astype(np.uint32)


# ── Internal: Numba ICM kernel ───────────────────────────────────────────────

@numba.njit(cache=True)
def _nb_icm_round(
    labels: np.ndarray,
    unary_dense: np.ndarray,
    h_w: np.ndarray,
    v_w: np.ndarray,
    dr_w: np.ndarray,
    dl_w: np.ndarray,
    tw_w: np.ndarray,
    tw_ty_dn_w: np.ndarray,
    tw_ty_up_w: np.ndarray,
    tw_tx_r_w: np.ndarray,
    tw_tx_l_w: np.ndarray,
    fg_mask: np.ndarray,
    label_ids: np.ndarray,
    anchor_label: np.ndarray,
    areas: np.ndarray,
    lambda_area: np.float32,
) -> int:
    """One sequential Gauss-Seidel ICM sweep — raster scan, in-place."""
    T, Y, X = fg_mask.shape
    K = len(label_ids)
    n_flips = 0

    for t in range(T):
        for y in range(Y):
            for x in range(X):
                if anchor_label[t, y, x] > np.uint32(0):
                    continue
                if not fg_mask[t, y, x]:
                    continue

                old_label = labels[t, y, x]

                ji = np.int32(-1)
                for kk in range(K):
                    if label_ids[kk] == old_label:
                        ji = np.int32(kk)
                        break

                best_cost = np.float32(1e30)
                best_k = old_label
                best_ki = np.int32(-1)

                for ki in range(K):
                    u = unary_dense[t, y, x, ki]
                    if u >= np.float32(1e8):
                        continue
                    k = label_ids[ki]

                    if k != old_label:
                        adj = False
                        if x + 1 < X and labels[t, y, x + 1] == k:
                            adj = True
                        elif x > 0 and labels[t, y, x - 1] == k:
                            adj = True
                        elif y + 1 < Y and labels[t, y + 1, x] == k:
                            adj = True
                        elif y > 0 and labels[t, y - 1, x] == k:
                            adj = True
                        elif y + 1 < Y and x + 1 < X and labels[t, y + 1, x + 1] == k:
                            adj = True
                        elif y > 0 and x > 0 and labels[t, y - 1, x - 1] == k:
                            adj = True
                        elif y + 1 < Y and x > 0 and labels[t, y + 1, x - 1] == k:
                            adj = True
                        elif y > 0 and x + 1 < X and labels[t, y - 1, x + 1] == k:
                            adj = True
                        if not adj:
                            continue

                    cost = u

                    if x + 1 < X and fg_mask[t, y, x + 1] and labels[t, y, x + 1] != k:
                        cost += h_w[t, y, x]
                    if x > 0 and fg_mask[t, y, x - 1] and labels[t, y, x - 1] != k:
                        cost += h_w[t, y, x - 1]
                    if y + 1 < Y and fg_mask[t, y + 1, x] and labels[t, y + 1, x] != k:
                        cost += v_w[t, y, x]
                    if y > 0 and fg_mask[t, y - 1, x] and labels[t, y - 1, x] != k:
                        cost += v_w[t, y - 1, x]

                    if y + 1 < Y and x + 1 < X and fg_mask[t, y + 1, x + 1] and labels[t, y + 1, x + 1] != k:
                        cost += dr_w[t, y, x]
                    if y > 0 and x > 0 and fg_mask[t, y - 1, x - 1] and labels[t, y - 1, x - 1] != k:
                        cost += dr_w[t, y - 1, x - 1]
                    if y + 1 < Y and x > 0 and fg_mask[t, y + 1, x - 1] and labels[t, y + 1, x - 1] != k:
                        cost += dl_w[t, y, x]
                    if y > 0 and x + 1 < X and fg_mask[t, y - 1, x + 1] and labels[t, y - 1, x + 1] != k:
                        cost += dl_w[t, y - 1, x + 1]

                    if t + 1 < T and fg_mask[t + 1, y, x] and labels[t + 1, y, x] != k:
                        cost += tw_w[t, y, x]
                    if t > 0 and fg_mask[t - 1, y, x] and labels[t - 1, y, x] != k:
                        cost += tw_w[t - 1, y, x]

                    if t + 1 < T:
                        if y + 1 < Y and fg_mask[t + 1, y + 1, x] and labels[t + 1, y + 1, x] != k:
                            cost += tw_ty_dn_w[t, y, x]
                        if y > 0 and fg_mask[t + 1, y - 1, x] and labels[t + 1, y - 1, x] != k:
                            cost += tw_ty_up_w[t, y, x]
                        if x + 1 < X and fg_mask[t + 1, y, x + 1] and labels[t + 1, y, x + 1] != k:
                            cost += tw_tx_r_w[t, y, x]
                        if x > 0 and fg_mask[t + 1, y, x - 1] and labels[t + 1, y, x - 1] != k:
                            cost += tw_tx_l_w[t, y, x]

                    if t > 0:
                        if y + 1 < Y and fg_mask[t - 1, y + 1, x] and labels[t - 1, y + 1, x] != k:
                            cost += tw_ty_up_w[t - 1, y + 1, x]
                        if y > 0 and fg_mask[t - 1, y - 1, x] and labels[t - 1, y - 1, x] != k:
                            cost += tw_ty_dn_w[t - 1, y - 1, x]
                        if x + 1 < X and fg_mask[t - 1, y, x + 1] and labels[t - 1, y, x + 1] != k:
                            cost += tw_tx_l_w[t - 1, y, x + 1]
                        if x > 0 and fg_mask[t - 1, y, x - 1] and labels[t - 1, y, x - 1] != k:
                            cost += tw_tx_r_w[t - 1, y, x - 1]

                    if lambda_area > np.float32(0.0) and ki != ji:
                        ak = areas[ki, t]
                        aj = areas[ji, t] if ji >= np.int32(0) else np.int32(0)
                        area_delta = np.float32(0.0)
                        if t > 0:
                            area_delta += np.float32(2 * (ak - areas[ki, t - 1]) + 1)
                            if ji >= np.int32(0):
                                area_delta += np.float32(1 - 2 * (aj - areas[ji, t - 1]))
                        if t + 1 < T:
                            area_delta += np.float32(2 * (ak - areas[ki, t + 1]) + 1)
                            if ji >= np.int32(0):
                                area_delta += np.float32(1 - 2 * (aj - areas[ji, t + 1]))
                        cost += lambda_area * area_delta

                    if cost < best_cost:
                        best_cost = cost
                        best_k = k
                        best_ki = np.int32(ki)

                labels[t, y, x] = best_k
                if best_k != old_label:
                    n_flips += 1
                    if best_ki >= np.int32(0):
                        areas[best_ki, t] += np.int32(1)
                    if ji >= np.int32(0):
                        areas[ji, t] -= np.int32(1)

    return n_flips


def _build_areas(labels: np.ndarray, label_ids: np.ndarray) -> np.ndarray:
    K = len(label_ids)
    T = labels.shape[0]
    areas = np.zeros((K, T), dtype=np.int32)
    for ki, k in enumerate(label_ids):
        for t in range(T):
            areas[ki, t] = int(np.count_nonzero(labels[t] == k))
    return areas


# ══════════════════════════════════════════════════════════════════════════════
# Public API — Staged
# ══════════════════════════════════════════════════════════════════════════════

def initialize_icm(
    nuc_tracks: np.ndarray,
    fg_mask: np.ndarray,
    contours: np.ndarray,
    params: CellLabelICMParams,
    *,
    foreground_scores: np.ndarray | None = None,
    cache_dir: Path | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[CellICMState, np.ndarray]:
    """Compute energy terms and build initial labels.

    Parameters
    ----------
    nuc_tracks : (T, Y, X) uint32
    fg_mask : (T, Y, X) bool
    contours : (T, Y, X) float32
    params : CellLabelICMParams
    foreground_scores : (T, Y, X) float32, optional
    cache_dir : Path, optional
        HDF5 unary cache directory.
    progress_cb : callable, optional

    Returns
    -------
    state : CellICMState
    init_labels : (T, Y, X) uint32
    """
    _report = progress_cb or (lambda msg: None)

    fg_mask = fg_mask | (nuc_tracks > 0)

    label_ids = np.array(
        sorted(int(k) for k in np.unique(nuc_tracks) if k > 0),
        dtype=np.uint32,
    )
    T, Y, X = fg_mask.shape
    _report(
        f"Label set: {len(label_ids)} track IDs, "
        f"shape {T}×{Y}×{X}, "
        f"fg_voxels={int(np.count_nonzero(fg_mask))}"
    )

    # ── Pairwise weights (cheap — always recompute) ───────────────────
    _report("Computing pairwise weights...")
    boundary_signal = np.clip(contours, 0.0, 1.0).astype(np.float32, copy=False)
    h, v, dr, dl, tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l = (
        _compute_pairwise_weights(
            fg_mask, boundary_signal,
            params.lambda_s, params.beta_s, params.lambda_t,
        )
    )

    # ── Geodesic unaries (expensive — cache + parallel) ───────────────
    cache_key = _unary_cache_key((T, Y, X), params.alpha_unary, params.gamma_unary)
    unary_dict: dict[tuple[int, int], np.ndarray] | None = None

    if cache_dir is not None:
        _report(f"Checking unary cache: {cache_key}")
        unary_dict = _read_unary_cache(cache_dir, cache_key)
        if unary_dict is not None:
            _report(f"Cache hit: {len(unary_dict)} entries loaded.")

    if unary_dict is None:
        t0 = perf_counter()
        unary_dict = _compute_geodesic_unaries(
            nuc_tracks, fg_mask, contours, label_ids, params.alpha_unary,
            foreground_scores=foreground_scores,
            gamma_unary=params.gamma_unary,
            n_workers=params.n_workers,
            progress_cb=progress_cb,
        )
        elapsed = perf_counter() - t0
        _report(f"Geodesic unaries: {len(unary_dict)} entries in {elapsed:.1f}s")
        if cache_dir is not None:
            _report("Writing unary cache...")
            _write_unary_cache(cache_dir, cache_key, unary_dict)

    _apply_nucleus_anchors(unary_dict, nuc_tracks, label_ids)

    # ── Dense unary ───────────────────────────────────────────────────
    _report("Building dense unary array...")
    unary_dense = _dict_to_dense_unary(unary_dict, fg_mask, label_ids)
    del unary_dict

    # ── Initial labels ────────────────────────────────────────────────
    mode = params.init_mode
    if mode == "unary":
        _report("Initialising labels from unary argmin...")
        init_labels = _argmin_init(unary_dense, fg_mask, label_ids)
    elif mode == "watershed":
        _report("Initialising labels via seeded watershed...")
        elevation = _unary_elevation_from_dense(unary_dense, fg_mask)
        init_labels = _watershed_init(fg_mask, nuc_tracks, elevation)
    else:
        _report("Initialising labels from nucleus pixels only...")
        init_labels = np.zeros((T, Y, X), dtype=np.uint32)

    nuc_mask = nuc_tracks > 0
    init_labels[nuc_mask] = nuc_tracks[nuc_mask].astype(np.uint32)

    state = CellICMState(
        fg_mask=fg_mask.astype(bool, copy=False),
        nuc_tracks=nuc_tracks.astype(np.uint32, copy=False),
        label_ids=label_ids,
        unary_dense=unary_dense,
        h=h, v=v, dr=dr, dl=dl,
        tw=tw, tw_ty_dn=tw_ty_dn, tw_ty_up=tw_ty_up,
        tw_tx_r=tw_tx_r, tw_tx_l=tw_tx_l,
    )

    _report("Initialisation complete.")
    return state, init_labels


def refine_icm(
    state: CellICMState,
    labels: np.ndarray,
    n_iters: int = 1,
    *,
    min_round_flips: int = 0,
    lambda_area: float = 0.0,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Run *n_iters* ICM sweeps on *labels*.

    Input ``labels`` is **not** modified — a copy is returned.
    """
    _report = progress_cb or (lambda msg: None)

    labels = labels.copy().astype(np.uint32)
    nuc_mask = state.nuc_tracks > 0
    labels[nuc_mask] = state.nuc_tracks[nuc_mask].astype(np.uint32)

    anchor_label = state.nuc_tracks.astype(np.uint32)
    h32 = state.h.astype(np.float32, copy=False)
    v32 = state.v.astype(np.float32, copy=False)
    dr32 = state.dr.astype(np.float32, copy=False)
    dl32 = state.dl.astype(np.float32, copy=False)
    tw32 = state.tw.astype(np.float32, copy=False)
    tw_ty_dn32 = state.tw_ty_dn.astype(np.float32, copy=False)
    tw_ty_up32 = state.tw_ty_up.astype(np.float32, copy=False)
    tw_tx_r32 = state.tw_tx_r.astype(np.float32, copy=False)
    tw_tx_l32 = state.tw_tx_l.astype(np.float32, copy=False)
    lids32 = state.label_ids.astype(np.uint32, copy=False)
    lambda_area32 = np.float32(lambda_area)

    areas = _build_areas(labels, state.label_ids)

    energy_log: list[dict] = []
    for iteration in range(n_iters):
        _report(f"ICM round {iteration + 1}/{n_iters}...")
        t0 = perf_counter()
        n_flips = _nb_icm_round(
            labels, state.unary_dense,
            h32, v32, dr32, dl32, tw32,
            tw_ty_dn32, tw_ty_up32, tw_tx_r32, tw_tx_l32,
            state.fg_mask, lids32, anchor_label, areas, lambda_area32,
        )
        elapsed = perf_counter() - t0
        _report(f"ICM round {iteration + 1}/{n_iters}: {n_flips} flips, {elapsed:.1f}s")
        energy_log.append({
            "iteration": iteration + 1,
            "flips": int(n_flips),
            "elapsed_s": round(elapsed, 2),
        })
        if n_flips == 0:
            _report(f"Converged after round {iteration + 1}.")
            break
        if min_round_flips > 0 and n_flips < min_round_flips:
            _report(f"Stopping: {n_flips} < min_round_flips={min_round_flips}")
            break

    return labels, energy_log


def commit_labels(labels: np.ndarray, output_path: Path | str) -> None:
    """Write label array to TIFF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(output_path),
        labels.astype(np.uint16, copy=False),
        compression="zlib",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Public API — Legacy monolithic wrappers
# ══════════════════════════════════════════════════════════════════════════════

def segment_cells_icm(
    nuc_tracks: np.ndarray,
    fg_mask: np.ndarray,
    contours: np.ndarray,
    params: CellLabelICMParams,
    *,
    foreground_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Run the full pipeline on in-memory arrays."""
    state, init_labels = initialize_icm(
        nuc_tracks, fg_mask, contours, params,
        foreground_scores=foreground_scores,
        progress_cb=lambda msg: print(msg, flush=True),
    )
    labels, _ = refine_icm(
        state, init_labels,
        n_iters=params.n_iters,
        min_round_flips=params.min_round_flips,
        lambda_area=params.lambda_area,
        progress_cb=lambda msg: print(msg, flush=True),
    )
    return labels


def run_cell_icm_from_pos_dir(
    pos_dir: Path | str,
    params: CellLabelICMParams,
) -> np.ndarray:
    """Load TIFFs from disk, run full pipeline."""
    pos_dir = Path(pos_dir)
    nuc_tracks, fg_mask, contours, foreground_scores = _load_pos_dir_inputs(pos_dir)
    return segment_cells_icm(
        nuc_tracks, fg_mask, contours, params,
        foreground_scores=foreground_scores,
    )


def _read_tiff(path: Path, dtype) -> np.ndarray:
    a = np.asarray(tifffile.imread(str(path)), dtype=dtype)
    if a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0]
    return a


def _load_pos_dir_inputs(
    pos_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    nuc = _read_tiff(pos_dir / "2_nucleus" / "tracked_labels.tif", np.uint32)
    fg = _read_tiff(pos_dir / "3_cell" / "foreground_masks.tif", np.uint8) > 0
    ct = _read_tiff(pos_dir / "3_cell" / "contour_maps.tif", np.float32)
    fg = fg | (nuc > 0)
    fg_score_path = pos_dir / "3_cell" / "foreground_scores.tif"
    fg_scores = (
        _read_tiff(fg_score_path, np.float32) if fg_score_path.exists() else None
    )
    return nuc, fg, ct, fg_scores


--------------------------------------------------------------------------------
FILE: src/cellflow/segmentation/contour_filtering.py
--------------------------------------------------------------------------------

"""Filtering helpers for nucleus contour-map stacks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


@dataclass(frozen=True, slots=True)
class ContourFilterParams:
    """Parameters for spatial and temporal contour-map filtering."""

    median_kernel_time: int = 1
    median_kernel_space: int = 1
    gaussian_sigma_time: float = 0.0
    gaussian_sigma_space: float = 0.0


def _normalize_contour_stack(contours: np.ndarray) -> tuple[np.ndarray, str]:
    arr = np.asarray(contours, dtype=np.float32)
    if arr.ndim == 2:
        return arr[np.newaxis], "yx"
    if arr.ndim == 3:
        return arr, "tyx"
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0], "tcyx"
    raise ValueError(f"Unsupported contour maps shape {arr.shape}")


def _restore_contour_stack(contours_tyx: np.ndarray, layout: str) -> np.ndarray:
    if layout == "yx":
        return contours_tyx[0]
    if layout == "tcyx":
        return contours_tyx[:, np.newaxis]
    return contours_tyx


def compute_filtered_contour_maps(
    contours: np.ndarray,
    params: ContourFilterParams,
) -> np.ndarray:
    """Return contour maps after median and Gaussian filtering."""
    filtered, layout = _normalize_contour_stack(contours)
    if params.median_kernel_time > 1 or params.median_kernel_space > 1:
        filtered = median_filter(
            filtered,
            size=(
                int(params.median_kernel_time),
                int(params.median_kernel_space),
                int(params.median_kernel_space),
            ),
        )
    if params.gaussian_sigma_time > 0.0 or params.gaussian_sigma_space > 0.0:
        filtered = gaussian_filter(
            filtered,
            sigma=(
                float(params.gaussian_sigma_time),
                float(params.gaussian_sigma_space),
                float(params.gaussian_sigma_space),
            ),
        )
    return _restore_contour_stack(
        np.asarray(filtered, dtype=np.float32),
        layout,
    )


--------------------------------------------------------------------------------
FILE: src/cellflow/segmentation/flow_following.py
--------------------------------------------------------------------------------

"""Flow-following cell segmentation: per-frame Euler integration of the
Cellpose flow field with an EDT-direction gravity blend toward tracked nuclei."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numba
import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    median_filter,
)


@dataclass(frozen=True, slots=True)
class FlowFollowingParams:
    """Parameters for `compute_flow_following_movie`."""

    median_kernel_time: int = 3
    median_kernel_space: int = 5
    gaussian_sigma_time: float = 0.0
    gaussian_sigma_space: float = 0.0
    flow_weight: float = 0.5
    flow_step_scale: float = 0.2
    max_iterations: int = 100
    capture_radius: float = 3.0


# Progressive shell assignment defaults (not user-facing)
_SHELL_WIDTH: float = 5.0
_MAX_SHELLS: int = 50


def compute_filtered_flow_vectors(
    dp_tcyx: np.ndarray,
    params: FlowFollowingParams,
) -> np.ndarray:
    """Return flow vectors after the configured median and Gaussian filters."""
    filtered = np.asarray(dp_tcyx, dtype=np.float32)
    if params.median_kernel_time > 1 or params.median_kernel_space > 1:
        filtered = median_filter(
            filtered,
            size=(
                1,
                int(params.median_kernel_time),
                int(params.median_kernel_space),
                int(params.median_kernel_space),
            ),
        )
    if params.gaussian_sigma_time > 0.0 or params.gaussian_sigma_space > 0.0:
        filtered = gaussian_filter(
            filtered,
            sigma=(
                0.0,
                float(params.gaussian_sigma_time),
                float(params.gaussian_sigma_space),
                float(params.gaussian_sigma_space),
            ),
        )
    return np.asarray(filtered, dtype=np.float32)


@numba.njit(parallel=True, cache=True)
def _flow_integrate(
    nuclear_labels: np.ndarray,    # (H, W) int32
    flow: np.ndarray,              # (H, W, 2) float32 — channel 0 = dY, channel 1 = dX
    grav_y: np.ndarray,            # (H, W) float32 — EDT-direction unit vector y
    grav_x: np.ndarray,            # (H, W) float32 — EDT-direction unit vector x
    dist_to_nucleus: np.ndarray,   # (H, W) float32 — EDT distance to nearest nuclear pixel
    nearest_y: np.ndarray,         # (H, W) int32 — y-index of nearest nuclear pixel
    nearest_x: np.ndarray,         # (H, W) int32
    prob_mask: np.ndarray,         # (H, W) bool — foreground mask
    n_steps: int,
    flow_step_scale: float,
    flow_weight: float,
    capture_radius: float,
) -> np.ndarray:
    H, W = nuclear_labels.shape
    result = nuclear_labels.copy()

    for i in numba.prange(H):
        for j in range(W):
            if result[i, j] > 0:
                continue
            if not prob_mask[i, j]:
                continue

            py = float(i)
            px = float(j)
            label = 0

            for _ in range(n_steps):
                iy0 = int(py)
                ix0 = int(px)
                iy0 = max(0, min(H - 2, iy0))
                ix0 = max(0, min(W - 2, ix0))

                fy = py - float(iy0)
                fx = px - float(ix0)

                flow_y = (flow[iy0,     ix0,     0] * (1.0 - fy) * (1.0 - fx) +
                          flow[iy0 + 1, ix0,     0] * fy          * (1.0 - fx) +
                          flow[iy0,     ix0 + 1, 0] * (1.0 - fy) * fx          +
                          flow[iy0 + 1, ix0 + 1, 0] * fy          * fx)

                flow_x = (flow[iy0,     ix0,     1] * (1.0 - fy) * (1.0 - fx) +
                          flow[iy0 + 1, ix0,     1] * fy          * (1.0 - fx) +
                          flow[iy0,     ix0 + 1, 1] * (1.0 - fy) * fx          +
                          flow[iy0 + 1, ix0 + 1, 1] * fy          * fx)

                w = flow_weight

                iy_nn = max(0, min(H - 1, int(py + 0.5)))
                ix_nn = max(0, min(W - 1, int(px + 0.5)))

                step_y = w * flow_y + (1.0 - w) * grav_y[iy_nn, ix_nn]
                step_x = w * flow_x + (1.0 - w) * grav_x[iy_nn, ix_nn]

                py = max(0.0, min(float(H - 1), py + step_y * flow_step_scale))
                px = max(0.0, min(float(W - 1), px + step_x * flow_step_scale))

                iy = max(0, min(H - 1, int(py + 0.5)))
                ix = max(0, min(W - 1, int(px + 0.5)))

                if dist_to_nucleus[iy, ix] <= capture_radius:
                    L = nuclear_labels[nearest_y[iy, ix], nearest_x[iy, ix]]
                    if L > 0:
                        label = L
                        break

            result[i, j] = label

    return result


def compute_flow_following_movie(
    foreground_tyx: np.ndarray,    # (T, Y, X) bool
    dp_tcyx: np.ndarray,           # (T, 2, Y, X) float32
    labels_tyx: np.ndarray,        # (T, Y, X) int32
    params: FlowFollowingParams,
    progress_cb: Callable[[int, int], None] | None = None,
    *,
    filter_vectors: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame flow-following segmentation with pre-integration filtering.

    Returns
    -------
    filtered_dp_tcyx : (T, 2, Y, X) float32 — flow stack after median+Gaussian.
    cell_labels_tyx  : (T, Y, X) int32      — same labelling as input nuclei.
    """
    foreground = np.asarray(foreground_tyx, dtype=bool)
    dp = np.asarray(dp_tcyx, dtype=np.float32)
    labels = np.asarray(labels_tyx, dtype=np.int32)

    T = dp.shape[0]

    filtered = compute_filtered_flow_vectors(dp, params) if filter_vectors else dp

    out_labels = np.zeros_like(labels, dtype=np.int32)

    for t in range(T):
        prob_mask = foreground[t]
        nuclear_labels = labels[t]

        if not prob_mask.any() or not (nuclear_labels > 0).any():
            if progress_cb is not None:
                progress_cb(t + 1, T)
            continue

        flow_yx2 = np.stack(
            [filtered[t, 0], filtered[t, 1]], axis=-1
        ).astype(np.float32)
        mags = np.hypot(flow_yx2[..., 0], flow_yx2[..., 1])
        mean_mag = float(mags[prob_mask].mean()) if prob_mask.any() else 0.0
        if mean_mag > 1e-6:
            flow_yx2 = (flow_yx2 / mean_mag).astype(np.float32)

        dist, (ny, nx) = distance_transform_edt(
            nuclear_labels == 0, return_indices=True
        )
        H, W = nuclear_labels.shape
        yi, xi = np.indices((H, W))
        dy = (ny - yi).astype(np.float32)
        dx = (nx - xi).astype(np.float32)
        norm = np.hypot(dy, dx)
        safe = np.where(norm > 0, norm, 1.0)
        grav_y = (dy / safe).astype(np.float32)
        grav_x = (dx / safe).astype(np.float32)
        inside = nuclear_labels > 0
        grav_y[inside] = 0.0
        grav_x[inside] = 0.0

        integrated = _flow_integrate(
            nuclear_labels.astype(np.int32),
            np.ascontiguousarray(flow_yx2, dtype=np.float32),
            grav_y, grav_x,
            dist.astype(np.float32),
            ny.astype(np.int32), nx.astype(np.int32),
            prob_mask,
            int(params.max_iterations),
            float(params.flow_step_scale),
            float(params.flow_weight),
            float(params.capture_radius),
        )

        out_labels[t] = integrated

        if progress_cb is not None:
            progress_cb(t + 1, T)

    return filtered, out_labels


# ---------------------------------------------------------------------------
# Single-frame & consensus helpers (for contour-map alternative path)
# ---------------------------------------------------------------------------


# ===================================================================
# Two-phase flow integration (capture_radius == 0 path)
# ===================================================================


@numba.njit(parallel=True, cache=True)
def _flow_integrate_to_positions(
    nuclear_labels,   # (H, W) int32
    flow,             # (2, H, W) float32
    grav_y,           # (H, W) float32
    grav_x,           # (H, W) float32
    prob_mask,        # (H, W) bool
    n_steps,          # int
    flow_step_scale,  # float
    flow_weight,      # float
):
    """Phase 1: integrate each foreground pixel along flow + gravity.

    Returns
    -------
    result : (H, W) int32
        Label map.  Nucleus pixels keep their label.  Foreground pixels
        that land directly on a nucleus pixel during integration receive
        that label.  Everything else remains 0.
    final_y, final_x : (H, W) float32
        Displaced position of every pixel after integration.
        Non-foreground and nucleus pixels store their original coords.
    """
    H = nuclear_labels.shape[0]
    W = nuclear_labels.shape[1]
    result = nuclear_labels.copy()
    final_y = np.empty((H, W), dtype=np.float32)
    final_x = np.empty((H, W), dtype=np.float32)

    for i in numba.prange(H):
        for j in range(W):
            # Default: pixel stays at its original position
            final_y[i, j] = np.float32(i)
            final_x[i, j] = np.float32(j)

            # Skip nucleus pixels and non-foreground
            if nuclear_labels[i, j] > 0 or not prob_mask[i, j]:
                continue

            py = np.float32(i)
            px = np.float32(j)

            for _ in range(n_steps):
                iy = min(max(int(py), 0), H - 1)
                ix = min(max(int(px), 0), W - 1)

                fy = (
                    flow_weight * flow[0, iy, ix]
                    + (np.float32(1.0) - flow_weight) * grav_y[iy, ix]
                )
                fx = (
                    flow_weight * flow[1, iy, ix]
                    + (np.float32(1.0) - flow_weight) * grav_x[iy, ix]
                )

                py = py + fy * flow_step_scale
                px = px + fx * flow_step_scale
                py = min(max(py, np.float32(0.0)), np.float32(H - 1))
                px = min(max(px, np.float32(0.0)), np.float32(W - 1))

                # Early stop: landed on a nucleus pixel
                iy2 = int(py)
                ix2 = int(px)
                if nuclear_labels[iy2, ix2] > 0:
                    result[i, j] = nuclear_labels[iy2, ix2]
                    break

            final_y[i, j] = py
            final_x[i, j] = px

    return result, final_y, final_x


def _progressive_shell_assign(
    labels,       # (H, W) int32  — from phase 1
    final_y,      # (H, W) float32
    final_x,      # (H, W) float32
    foreground,   # (H, W) bool
    shell_width=_SHELL_WIDTH,
    max_shells=_MAX_SHELLS,
):
    """Phase 2: grow labels outward in shells through displaced positions.

    Each iteration:
      1. Compute EDT from all currently-labelled pixels.
      2. For each unassigned foreground pixel, look up the distance at
         its *final* (flow-displaced) position.
      3. If within ``shell_width`` → assign the nearest label.
      4. Newly labelled pixels (at their *original* positions) seed the
         next EDT — this is how labels propagate through the
         displaced-position topology.

    A final fallback assigns any remaining pixels to their nearest label
    (no distance limit) so that every foreground pixel is guaranteed a
    label.

    Parameters
    ----------
    labels : ndarray (H, W) int32
        Partially assigned label map from phase 1.
    final_y, final_x : ndarray (H, W) float32
        Displaced positions from phase 1.
    foreground : ndarray (H, W) bool
        Foreground mask (True = should be labelled).
    shell_width : float
        Maximum capture distance per iteration (pixels).
    max_shells : int
        Safety cap on the number of growth iterations.

    Returns
    -------
    result : ndarray (H, W) int32
        Fully assigned label map.
    """
    from scipy.ndimage import distance_transform_edt

    H, W = labels.shape
    result = labels.copy()
    unassigned = foreground & (result == 0)

    # Integer final positions, clipped to image bounds
    fy = np.clip(np.round(final_y).astype(np.intp), 0, H - 1)
    fx = np.clip(np.round(final_x).astype(np.intp), 0, W - 1)

    for _ in range(max_shells):
        if not unassigned.any():
            break

        # EDT from every unlabelled pixel to the nearest labelled pixel
        unlabelled_mask = result == 0
        if not unlabelled_mask.any():
            break
        dist, indices = distance_transform_edt(
            unlabelled_mask, return_indices=True,
        )
        ind_y = indices[0]
        ind_x = indices[1]

        # Query at each unassigned pixel's FINAL (displaced) position
        d = dist[fy, fx]
        ny = ind_y[fy, fx]
        nx = ind_x[fy, fx]
        nearest_label = result[ny, nx]

        can_assign = unassigned & (d <= shell_width) & (nearest_label > 0)
        if not can_assign.any():
            break  # stalled — no pixel's displaced position is close enough

        result[can_assign] = nearest_label[can_assign]
        unassigned &= ~can_assign

    # ------------------------------------------------------------------
    # Fallback: assign any remaining pixels with no distance limit
    # ------------------------------------------------------------------
    if unassigned.any() and (result > 0).any():
        unlabelled_mask = result == 0
        if unlabelled_mask.any():
            _dist, indices = distance_transform_edt(
                unlabelled_mask, return_indices=True,
            )
            ny = indices[0][fy, fx]
            nx = indices[1][fy, fx]
            nearest_label = result[ny, nx]
            still_open = unassigned & (nearest_label > 0)
            result[still_open] = nearest_label[still_open]

    return result


def compute_flow_following_frame(
    foreground_yx: np.ndarray,
    dp_cyx: np.ndarray,
    labels_yx: np.ndarray,
    params: FlowFollowingParams,
) -> np.ndarray:
    """Run flow-following segmentation on a **single frame**.

    Dispatches between two strategies based on ``params.capture_radius``:

    * ``capture_radius > 0`` — legacy behaviour: wraps
      :func:`compute_flow_following_movie` with ``T=1``.  A pixel is
      captured when it enters the fixed radius around a nucleus.
    * ``capture_radius == 0`` — new two-phase algorithm:
        1. Integrate every foreground pixel along the flow field
           (blended with EDT gravity).  If a pixel lands directly on a
           nucleus, assign immediately.
        2. Grow labels outward in progressive shells: each iteration
           assigns unassigned pixels whose *displaced* positions are
           within a shell width of an already-labelled pixel's *original*
           position.  This lets labels chain-propagate through the
           displaced-position topology.

    Parameters
    ----------
    foreground_yx : ndarray (Y, X), bool
    dp_cyx : ndarray (2, Y, X), float32  — pre-filtered flow vectors
    labels_yx : ndarray (Y, X), int32  — nucleus tracked labels
    params : FlowFollowingParams

    Returns
    -------
    cell_labels : ndarray (Y, X), int32
    """
    from scipy.ndimage import distance_transform_edt

    if not foreground_yx.any():
        return np.zeros(foreground_yx.shape, dtype=np.int32)
    if labels_yx.max() == 0:
        return np.zeros(foreground_yx.shape, dtype=np.int32)

    # ------------------------------------------------------------------
    # Legacy path (capture_radius > 0): delegate to existing movie fn
    # ------------------------------------------------------------------
    if params.capture_radius > 0:
        fg_tyx = np.ascontiguousarray(foreground_yx[np.newaxis], dtype=bool)
        dp_tcyx = np.ascontiguousarray(dp_cyx[np.newaxis], dtype=np.float32)
        lab_tyx = np.ascontiguousarray(labels_yx[np.newaxis], dtype=np.int32)
        _, cell_labels_tyx = compute_flow_following_movie(
            fg_tyx, dp_tcyx, lab_tyx, params,
            progress_cb=None, filter_vectors=False,
        )
        return cell_labels_tyx[0]

    # ------------------------------------------------------------------
    # New path (capture_radius == 0): two-phase progressive assignment
    # ------------------------------------------------------------------
    H, W = foreground_yx.shape

    # ---- normalise flow by mean foreground magnitude ----
    flow = np.ascontiguousarray(dp_cyx, dtype=np.float32).copy()
    mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
    mean_mag = float(mag[foreground_yx].mean()) if foreground_yx.any() else 1.0
    if mean_mag > 0:
        flow /= np.float32(mean_mag)

    # ---- EDT from nucleus pixels → gravity directions ----
    nucleus_mask = labels_yx > 0
    dist, indices = distance_transform_edt(
        ~nucleus_mask, return_indices=True,
    )
    near_y = indices[0]
    near_x = indices[1]

    yy, xx = np.mgrid[:H, :W]
    dy = (near_y - yy).astype(np.float64)
    dx = (near_x - xx).astype(np.float64)
    norm = np.sqrt(dy ** 2 + dx ** 2)
    norm[norm == 0] = 1.0
    grav_y = (dy / norm).astype(np.float32)
    grav_x = (dx / norm).astype(np.float32)

    # ---- Phase 1: flow integration ----
    lab32 = np.ascontiguousarray(labels_yx, dtype=np.int32)
    fg_bool = np.ascontiguousarray(foreground_yx, dtype=np.bool_)
    grav_y = np.ascontiguousarray(grav_y)
    grav_x = np.ascontiguousarray(grav_x)

    result, final_y, final_x = _flow_integrate_to_positions(
        lab32, flow, grav_y, grav_x,
        fg_bool,
        int(params.max_iterations),
        np.float32(params.flow_step_scale),
        np.float32(params.flow_weight),
    )

    # ---- Phase 2: progressive shell assignment ----
    result = _progressive_shell_assign(
        result, final_y, final_x, fg_bool,
    )

    return result


def build_consensus_boundary_flow_following(
    prob_yx: np.ndarray,
    dp_cyx: np.ndarray,
    labels_yx: np.ndarray,
    cellprob_thresholds: list[float],
    params: FlowFollowingParams,
    reduction: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Build a consensus contour map using flow-following + EDT gravity.

    This mirrors :func:`build_consensus_boundary_2d` but replaces Cellpose's
    ``compute_masks`` with :func:`compute_flow_following_frame`.  For every
    *cellprob_threshold* the probability map is binarised into a foreground
    mask; flow-following assigns each foreground pixel to the nearest nucleus;
    inner boundaries are extracted from the resulting label map and
    accumulated across thresholds.

    Parameters
    ----------
    prob_yx : ndarray, shape (Y, X), dtype float32
        Z-averaged, gamma-corrected probability logits.
    dp_cyx : ndarray, shape (2, Y, X), dtype float32
        Pre-filtered 2-D flow vectors.
    labels_yx : ndarray, shape (Y, X), dtype int32
        Nucleus tracked labels for this frame.
    cellprob_thresholds : list[float]
        Thresholds swept for the consensus boundary.
    params : FlowFollowingParams
        Integration hyper-parameters.
    reduction : {"mean", "sum"}
        How to reduce across thresholds.

    Returns
    -------
    boundary : ndarray, shape (Y, X), dtype float32
        Accumulated boundary confidence in [0, 1] (if *reduction* = "mean").
    foreground : ndarray, shape (Y, X), dtype float32
        Accumulated foreground score in [0, 1] (if *reduction* = "mean").
    """
    from skimage.segmentation import find_boundaries

    H, W = prob_yx.shape
    boundary_accum = np.zeros((H, W), dtype=np.float32)
    foreground_accum = np.zeros((H, W), dtype=np.float32)
    n = 0

    for thresh in cellprob_thresholds:
        fg_mask = prob_yx > thresh
        if not fg_mask.any():
            continue

        cell_labels = compute_flow_following_frame(
            fg_mask, dp_cyx, labels_yx, params,
        )

        boundary_accum += find_boundaries(
            cell_labels, mode="inner",
        ).astype(np.float32)
        foreground_accum += (cell_labels > 0).astype(np.float32)
        n += 1

    if n > 0 and reduction == "mean":
        boundary_accum /= n
        foreground_accum /= n

    return boundary_accum, foreground_accum


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking/__init__.py
--------------------------------------------------------------------------------

"""Greedy IoU propagator for nucleus tracking."""
from cellflow.tracking.propagator import find_best_hypothesis, propagate_one_frame

__all__ = ["find_best_hypothesis", "propagate_one_frame"]


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking/consensus_movie.py
--------------------------------------------------------------------------------

"""Consensus-label movie helpers for cell-boundary hypothesis sweeps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True, slots=True)
class ConsensusMember:
    p: int
    compactness: float
    foreground_threshold: float
    basin: str


@dataclass(frozen=True, slots=True)
class CompactnessGroup:
    compactness: float
    members: tuple[ConsensusMember, ...]


@dataclass(frozen=True, slots=True)
class ConsensusMovie:
    labels: np.ndarray
    support: np.ndarray
    thresholds: np.ndarray


def vote_consensus_labels(
    labels: np.ndarray,
    *,
    vote_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pixel majority labels and their vote support.

    ``labels`` is an integer array with shape ``(n_votes, y, x)``. Label IDs are
    treated as categories, not numeric intensities.
    """
    stack = np.asarray(labels)
    if stack.ndim != 3:
        raise ValueError(f"Expected labels with shape (n_votes, y, x), got {stack.shape}")
    if stack.shape[0] == 0:
        raise ValueError("Expected at least one vote plane")
    if not 0.0 <= vote_threshold <= 1.0:
        raise ValueError("vote_threshold must be between 0 and 1")

    sorted_stack = np.sort(stack, axis=0)
    best_label = sorted_stack[0].copy()
    current_label = sorted_stack[0].copy()
    best_count = np.ones(sorted_stack.shape[1:], dtype=np.uint16)
    current_count = np.ones(sorted_stack.shape[1:], dtype=np.uint16)

    for plane in sorted_stack[1:]:
        same_label = plane == current_label
        current_count = np.where(same_label, current_count + 1, 1)
        current_label = np.where(same_label, current_label, plane)
        better = current_count > best_count
        best_count = np.where(better, current_count, best_count)
        best_label = np.where(better, current_label, best_label)

    support = (best_count.astype(np.float32) / float(stack.shape[0])).astype(np.float32)
    consensus = np.where(support >= vote_threshold, best_label, 0).astype(stack.dtype, copy=False)
    return consensus, support


def collapse_z_by_label_presence(labels: np.ndarray) -> np.ndarray:
    """Collapse a z-stack by non-background label presence.

    Background z-slices do not vote unless every z-slice is background. If
    multiple non-background labels occur at a pixel, the most frequent label
    wins with the same deterministic tie behavior as ``vote_consensus_labels``.
    """
    stack = np.asarray(labels)
    if stack.ndim != 3:
        raise ValueError(f"Expected labels with shape (z, y, x), got {stack.shape}")
    if stack.shape[0] == 0:
        raise ValueError("Expected at least one z-slice")

    sentinel = np.iinfo(stack.dtype).max if np.issubdtype(stack.dtype, np.integer) else -1
    foreground_votes = np.where(stack > 0, stack, sentinel)
    collapsed, _support = vote_consensus_labels(foreground_votes, vote_threshold=0.0)
    return np.where(collapsed == sentinel, 0, collapsed).astype(stack.dtype, copy=False)


def vote_label_footprints(
    labels: np.ndarray,
    *,
    vote_threshold: float = 0.0,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vote across 2D label footprints while ignoring background as a competitor."""
    stack = np.asarray(labels)
    if stack.ndim != 3:
        raise ValueError(f"Expected labels with shape (n_votes, y, x), got {stack.shape}")
    if stack.shape[0] == 0:
        raise ValueError("Expected at least one footprint vote")
    if not 0.0 <= vote_threshold <= 1.0:
        raise ValueError("vote_threshold must be between 0 and 1")
    if weights is None:
        vote_weights = np.ones(stack.shape[0], dtype=np.float32)
    else:
        vote_weights = np.asarray(weights, dtype=np.float32)
        if vote_weights.shape != (stack.shape[0],):
            raise ValueError(f"Expected {stack.shape[0]} weights, got shape {vote_weights.shape}")
        if np.any(vote_weights < 0):
            raise ValueError("weights must be non-negative")
        if not np.any(vote_weights > 0):
            raise ValueError("At least one weight must be positive")

    labels_out = np.zeros(stack.shape[1:], dtype=stack.dtype)
    best_count = np.zeros(stack.shape[1:], dtype=np.float32)
    for label in np.unique(stack):
        if label == 0:
            continue
        count = np.sum(
            (stack == label) * vote_weights[:, np.newaxis, np.newaxis],
            axis=0,
            dtype=np.float32,
        )
        better = count > best_count
        labels_out = np.where(better, label, labels_out)
        best_count = np.where(better, count, best_count)

    support_out = (best_count.astype(np.float32) / float(np.sum(vote_weights))).astype(
        np.float32,
        copy=False,
    )
    labels_out = np.where(support_out >= vote_threshold, labels_out, 0).astype(
        stack.dtype,
        copy=False,
    )
    return labels_out, support_out


def apply_vote_thresholds(
    labels: np.ndarray,
    support: np.ndarray,
    thresholds: float | np.ndarray,
) -> np.ndarray:
    """Set labels below scalar or per-frame support thresholds to background."""
    label_stack = np.asarray(labels)
    support_stack = np.asarray(support, dtype=np.float32)
    if label_stack.shape != support_stack.shape:
        raise ValueError(f"Expected support shape {label_stack.shape}, got {support_stack.shape}")

    threshold_arr = np.asarray(thresholds, dtype=np.float32)
    if threshold_arr.ndim == 0:
        threshold_view = threshold_arr
    elif threshold_arr.ndim == 1 and label_stack.ndim == 3:
        if threshold_arr.shape[0] != label_stack.shape[0]:
            raise ValueError(
                f"Expected {label_stack.shape[0]} frame thresholds, got {threshold_arr.shape[0]}"
            )
        threshold_view = threshold_arr[:, np.newaxis, np.newaxis]
    elif threshold_arr.shape == label_stack.shape:
        threshold_view = threshold_arr
    else:
        raise ValueError(
            "thresholds must be scalar, one value per frame, or match the label shape"
        )
    return np.where(support_stack >= threshold_view, label_stack, 0).astype(
        label_stack.dtype,
        copy=False,
    )


def resolve_vote_thresholds(
    support: np.ndarray,
    labels: np.ndarray,
    *,
    mode: str = "fixed",
    vote_threshold: float = 0.5,
    percentile: float = 60.0,
    min_threshold: float = 0.35,
    max_threshold: float = 0.65,
) -> np.ndarray:
    """Return one support threshold per frame."""
    support_stack = np.asarray(support, dtype=np.float32)
    label_stack = np.asarray(labels)
    if support_stack.ndim != 3:
        raise ValueError(f"Expected support with shape (t, y, x), got {support_stack.shape}")
    if label_stack.shape != support_stack.shape:
        raise ValueError(f"Expected labels shape {support_stack.shape}, got {label_stack.shape}")
    if mode == "fixed":
        _validate_threshold(vote_threshold, "vote_threshold")
        return np.full(support_stack.shape[0], vote_threshold, dtype=np.float32)
    if mode != "percentile":
        raise ValueError(f"Unknown threshold mode: {mode}")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    _validate_threshold(min_threshold, "min_threshold")
    _validate_threshold(max_threshold, "max_threshold")
    if min_threshold > max_threshold:
        raise ValueError("min_threshold must be <= max_threshold")

    thresholds = np.zeros(support_stack.shape[0], dtype=np.float32)
    for t in range(support_stack.shape[0]):
        values = support_stack[t][label_stack[t] > 0]
        if values.size == 0:
            threshold = vote_threshold
        else:
            threshold = float(np.percentile(values, percentile))
        thresholds[t] = np.float32(np.clip(threshold, min_threshold, max_threshold))
    return thresholds


def smooth_consensus_labels(
    labels: np.ndarray,
    support: np.ndarray,
    *,
    vote_threshold: float = 0.5,
    weights: tuple[float, float, float] = (0.25, 0.5, 0.25),
) -> tuple[np.ndarray, np.ndarray]:
    """Temporally smooth top-voted consensus labels with a 3-frame window."""
    label_stack = np.asarray(labels)
    support_stack = np.asarray(support, dtype=np.float32)
    if label_stack.ndim != 3:
        raise ValueError(f"Expected labels with shape (t, y, x), got {label_stack.shape}")
    if support_stack.shape != label_stack.shape:
        raise ValueError(
            f"Expected support shape {label_stack.shape}, got {support_stack.shape}"
        )
    if len(weights) != 3:
        raise ValueError("weights must contain previous/current/next weights")
    if not 0.0 <= vote_threshold <= 1.0:
        raise ValueError("vote_threshold must be between 0 and 1")

    output = np.zeros_like(label_stack)
    output_support = np.zeros(label_stack.shape, dtype=np.float32)
    base_weights = np.asarray(weights, dtype=np.float32)

    for t in range(label_stack.shape[0]):
        window_labels = []
        window_scores = []
        active_weights = []
        for offset, weight in zip((-1, 0, 1), base_weights):
            idx = t + offset
            if 0 <= idx < label_stack.shape[0]:
                window_labels.append(label_stack[idx])
                window_scores.append(support_stack[idx])
                active_weights.append(float(weight))

        norm = float(sum(active_weights))
        candidate_labels = np.stack(window_labels, axis=0)
        candidate_scores = (
            np.asarray(active_weights, dtype=np.float32)[:, np.newaxis, np.newaxis]
            * np.stack(window_scores, axis=0)
            / norm
        )

        best_label = candidate_labels[0].copy()
        best_score = _score_candidate_label(candidate_labels, candidate_scores, best_label)
        for candidate in candidate_labels[1:]:
            score = _score_candidate_label(candidate_labels, candidate_scores, candidate)
            better = score > best_score
            best_label = np.where(better, candidate, best_label)
            best_score = np.where(better, score, best_score)

        output_support[t] = best_score.astype(np.float32, copy=False)
        output[t] = np.where(best_score >= vote_threshold, best_label, 0)

    return output, output_support


def load_compactness_groups(path: str | Path) -> list[CompactnessGroup]:
    """Return hypothesis parameter groups keyed by watershed compactness."""
    groups: dict[float, list[ConsensusMember]] = {}
    with h5py.File(Path(path), "r") as h5:
        root = h5["hypotheses"]
        first_t = sorted(k for k in root.keys() if k.startswith("t"))[0]
        for p_key in sorted(k for k in root[first_t].keys() if k.startswith("p")):
            p = int(p_key[1:])
            attrs = root[first_t][p_key].attrs
            compactness = float(attrs.get("compactness", 0.0))
            member = ConsensusMember(
                p=p,
                compactness=compactness,
                foreground_threshold=float(attrs.get("foreground_threshold", 0.0)),
                basin=str(attrs.get("basin", "")),
            )
            groups.setdefault(compactness, []).append(member)

    return [
        CompactnessGroup(
            compactness=compactness,
            members=tuple(
                sorted(
                    members,
                    key=lambda member: (member.foreground_threshold, member.basin, member.p),
                )
            ),
        )
        for compactness, members in sorted(groups.items())
    ]


def build_consensus_movie(
    path: str | Path,
    group: CompactnessGroup,
    *,
    vote_threshold: float = 0.5,
    smooth_temporally: bool = True,
    temporal_weights: tuple[float, float, float] = (0.25, 0.5, 0.25),
    threshold_mode: str = "fixed",
    threshold_percentile: float = 60.0,
    min_vote_threshold: float = 0.35,
    max_vote_threshold: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one consensus label movie for a compactness group."""
    movie = build_consensus_movie_with_thresholds(
        path,
        group,
        vote_threshold=vote_threshold,
        smooth_temporally=smooth_temporally,
        temporal_weights=temporal_weights,
        threshold_mode=threshold_mode,
        threshold_percentile=threshold_percentile,
        min_vote_threshold=min_vote_threshold,
        max_vote_threshold=max_vote_threshold,
    )
    return movie.labels, movie.support


def build_consensus_movie_with_thresholds(
    path: str | Path,
    group: CompactnessGroup,
    *,
    vote_threshold: float = 0.5,
    smooth_temporally: bool = True,
    temporal_weights: tuple[float, float, float] = (0.25, 0.5, 0.25),
    threshold_mode: str = "fixed",
    threshold_percentile: float = 60.0,
    min_vote_threshold: float = 0.35,
    max_vote_threshold: float = 0.65,
) -> ConsensusMovie:
    """Build one consensus label movie and return per-frame thresholds."""
    raw_labels = []
    raw_support = []
    with h5py.File(Path(path), "r") as h5:
        root = h5["hypotheses"]
        for t_key in sorted(k for k in root.keys() if k.startswith("t")):
            vote_planes = []
            for member in group.members:
                labels = root[t_key][f"p{member.p:03d}"]["labels"][:]
                vote_planes.extend(labels[z] for z in range(labels.shape[0]))
            consensus, support = vote_consensus_labels(
                np.stack(vote_planes, axis=0),
                vote_threshold=0.0,
            )
            raw_labels.append(consensus)
            raw_support.append(support)

    label_movie = np.stack(raw_labels, axis=0)
    support_movie = np.stack(raw_support, axis=0)
    if smooth_temporally:
        label_movie, support_movie = smooth_consensus_labels(
            label_movie,
            support_movie,
            vote_threshold=0.0,
            weights=temporal_weights,
        )
    thresholds = resolve_vote_thresholds(
        support_movie,
        label_movie,
        mode=threshold_mode,
        vote_threshold=vote_threshold,
        percentile=threshold_percentile,
        min_threshold=min_vote_threshold,
        max_threshold=max_vote_threshold,
    )
    thresholded_labels = apply_vote_thresholds(label_movie, support_movie, thresholds)
    return ConsensusMovie(
        labels=thresholded_labels,
        support=support_movie,
        thresholds=thresholds,
    )


def _score_candidate_label(
    candidate_labels: np.ndarray,
    candidate_scores: np.ndarray,
    label: np.ndarray,
) -> np.ndarray:
    return np.sum(np.where(candidate_labels == label, candidate_scores, 0.0), axis=0)


def _validate_threshold(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking/frame_selector.py
--------------------------------------------------------------------------------

"""Full-frame hypothesis selector for cell-boundary sweeps.

This module ranks complete per-frame hypotheses. It assumes cell positions and
IDs are already anchored by nucleus-derived seeds, so temporal coherence is
measured from same-ID boundary statistics rather than centroid search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True, slots=True)
class SelectorWeights:
    area: float = 1.0
    shape: float = 1.0
    missing: float = 5.0
    extra: float = 2.0
    parameter_switch: float = 0.05


@dataclass(frozen=True, slots=True)
class FrameStats:
    t: int
    p: int
    z: int
    ids: tuple[int, ...]
    areas: np.ndarray
    compactness: np.ndarray
    foreground_area: int
    id_array: np.ndarray | None = field(default=None, compare=False, repr=False)
    id_set: frozenset[int] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class TransitionScore:
    total: float
    area_cost: float
    shape_cost: float
    missing_count: int
    extra_count: int
    switch_cost: float


@dataclass(frozen=True, slots=True)
class RankedPath:
    score: float
    states: tuple[FrameStats, ...]
    transitions: tuple[TransitionScore, ...]
    state_key: tuple[tuple[int, int], ...] = field(default=(), compare=False, repr=False)


def compute_frame_stats(labels: np.ndarray, *, t: int, p: int, z: int = 0) -> FrameStats:
    """Return compact per-label statistics for one 2D frame hypothesis."""
    arr = np.asarray(labels)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(
                f"Expected a 2D label image or single-slice volume, got shape {arr.shape}"
            )
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label image, got shape {arr.shape}")

    areas = np.bincount(arr.ravel().astype(np.int64))
    if areas.size == 0:
        areas = np.zeros(1, dtype=np.int64)
    id_array = np.flatnonzero(areas).astype(np.int64, copy=False)
    id_array = id_array[id_array != 0]
    ids = tuple(int(i) for i in id_array)
    compactness = _label_compactness(arr, areas)
    foreground_area = int(areas[1:].sum()) if areas.size > 1 else 0
    return FrameStats(
        t=int(t),
        p=int(p),
        z=int(z),
        ids=ids,
        areas=areas,
        compactness=compactness,
        foreground_area=foreground_area,
        id_array=id_array,
        id_set=frozenset(ids),
    )


def _label_compactness(labels: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Return per-label 2D compactness."""
    arr = np.asarray(labels)
    if arr.ndim == 2:
        arr = arr[np.newaxis]

    max_id = len(areas) - 1
    perimeter = np.zeros(len(areas), dtype=np.float64)
    for left, right in ((arr[:, :, :-1], arr[:, :, 1:]), (arr[:, :-1, :], arr[:, 1:, :])):
        diff = left != right
        if not np.any(diff):
            continue
        left_ids = left[diff].astype(np.int64)
        right_ids = right[diff].astype(np.int64)
        if left_ids.size:
            perimeter += np.bincount(left_ids[left_ids != 0], minlength=max_id + 1)[:max_id + 1]
        if right_ids.size:
            perimeter += np.bincount(right_ids[right_ids != 0], minlength=max_id + 1)[:max_id + 1]

    compactness = np.zeros(len(areas), dtype=np.float64)
    valid = (areas > 0) & (perimeter > 0)
    compactness[valid] = np.minimum(1.0, (4.0 * np.pi * areas[valid]) / (perimeter[valid] ** 2))
    return compactness


def score_transition(
    previous: FrameStats,
    current: FrameStats,
    weights: SelectorWeights = SelectorWeights(),
) -> TransitionScore:
    """Score how coherent it is to move from one full frame to the next."""
    if previous.ids == current.ids:
        common = _id_array(previous)
        missing_count = 0
        extra_count = 0
    else:
        prev_ids = _id_set(previous)
        cur_ids = _id_set(current)
        common_set = prev_ids & cur_ids
        common = np.fromiter(sorted(common_set), dtype=np.int64, count=len(common_set))
        missing_count = len(prev_ids - cur_ids)
        extra_count = len(cur_ids - prev_ids)

    area_cost = 0.0
    shape_cost = 0.0
    if common.size:
        prev_area = previous.areas[common].astype(np.float64, copy=False)
        cur_area = current.areas[common].astype(np.float64, copy=False)
        area_cost = float(np.mean(np.abs(np.log((cur_area + 1.0) / (prev_area + 1.0)))))
        prev_shape = previous.compactness[common].astype(np.float64, copy=False)
        cur_shape = current.compactness[common].astype(np.float64, copy=False)
        shape_cost = float(np.mean(np.abs(cur_shape - prev_shape)))

    switch_cost = weights.parameter_switch if previous.p != current.p else 0.0
    total = (
        weights.area * area_cost
        + weights.shape * shape_cost
        + weights.missing * missing_count
        + weights.extra * extra_count
        + switch_cost
    )
    return TransitionScore(
        total=float(total),
        area_cost=area_cost,
        shape_cost=shape_cost,
        missing_count=missing_count,
        extra_count=extra_count,
        switch_cost=float(switch_cost),
    )


def select_top_k_paths(
    candidates_by_t: list[list[FrameStats]],
    *,
    k: int = 5,
    beam_width: int = 200,
    weights: SelectorWeights = SelectorWeights(),
) -> list[RankedPath]:
    """Return low-cost 2D frame paths through candidates_by_t using beam search."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if not candidates_by_t:
        return []
    if any(not candidates for candidates in candidates_by_t):
        raise ValueError("Each timepoint must contain at least one candidate")

    active_paths = [
        RankedPath(
            score=0.0,
            states=(state,),
            transitions=(),
            state_key=((state.p, state.z),),
        )
        for state in candidates_by_t[0]
    ]
    active_paths.sort(key=lambda path: (path.score, _path_state_key(path)))
    active_paths = active_paths[:beam_width]

    for candidates in candidates_by_t[1:]:
        expanded = []
        transition_cache: dict[tuple[int, int], TransitionScore] = {}
        for state in candidates:
            for path in active_paths:
                previous = path.states[-1]
                cache_key = (id(previous), id(state))
                transition = transition_cache.get(cache_key)
                if transition is None:
                    transition = score_transition(previous, state, weights)
                    transition_cache[cache_key] = transition
                expanded.append(
                    RankedPath(
                        score=path.score + transition.total,
                        states=path.states + (state,),
                        transitions=path.transitions + (transition,),
                        state_key=_path_state_key(path) + ((state.p, state.z),),
                    )
                )
        expanded.sort(key=lambda path: (path.score, _path_state_key(path)))
        active_paths = expanded[:beam_width]

    return active_paths[:k]


def _id_array(stats: FrameStats) -> np.ndarray:
    if stats.id_array is not None:
        return stats.id_array
    return np.fromiter(stats.ids, dtype=np.int64, count=len(stats.ids))


def _id_set(stats: FrameStats) -> frozenset[int]:
    if stats.id_set is not None:
        return stats.id_set
    return frozenset(stats.ids)


def _path_state_key(path: RankedPath) -> tuple[tuple[int, int], ...]:
    if path.state_key:
        return path.state_key
    return tuple((state.p, state.z) for state in path.states)


def load_hypothesis_frame_stats(path: str | Path) -> list[list[FrameStats]]:
    """Load per-candidate stats from a CellFlow hypotheses.h5 file."""
    grouped: list[list[FrameStats]] = []
    with h5py.File(Path(path), "r") as h5:
        root = h5["hypotheses"]
        for t_key in sorted(k for k in root.keys() if k.startswith("t")):
            t = int(t_key[1:])
            states = []
            for p_key in sorted(k for k in root[t_key].keys() if k.startswith("p")):
                p = int(p_key[1:])
                labels = root[t_key][p_key]["labels"][:]
                for z in range(labels.shape[0]):
                    states.append(compute_frame_stats(labels[z], t=t, p=p, z=z))
            grouped.append(states)
    return grouped


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking/propagator.py
--------------------------------------------------------------------------------

"""Best-match propagator for nucleus tracking.

For each nucleus in the current tracked frame, gates all candidate nuclei from
all hypotheses for the next timepoint by distance, scores the survivors using
additive shape-quality metrics, and picks the single best match greedily.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.spatial import KDTree, ConvexHull

from cellflow.database.hypotheses import read_hypothesis_labels, list_hypotheses


def _label_stats(labels: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Return (areas, centroids) without building per-label boolean masks."""
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return np.zeros(1, dtype=np.int64), {}
    areas = np.bincount(labels.ravel())
    coms = center_of_mass(np.ones_like(labels), labels, ids.tolist())
    if len(ids) == 1:
        coms = [coms]
    centroids = {int(lid): np.array(com).ravel() for lid, com in zip(ids, coms)}
    return areas, centroids


def _nucleus_pixels(labels: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return {label_id: (ys, xs)} pixel coordinate arrays for all non-zero labels."""
    ys_all, xs_all = np.nonzero(labels)
    vals = labels[ys_all, xs_all]
    result = {}
    for lid in np.unique(vals):
        mask = vals == lid
        result[int(lid)] = (ys_all[mask], xs_all[mask])
    return result


def _circularity(ys: np.ndarray, xs: np.ndarray, area: int) -> float:
    """4π·area / perimeter² using 4-connected boundary edge count."""
    if area == 0:
        return 0.0
    y_min, x_min = int(ys.min()), int(xs.min())
    h = int(ys.max()) - y_min + 3
    w = int(xs.max()) - x_min + 3
    img = np.zeros((h, w), dtype=bool)
    img[ys - y_min + 1, xs - x_min + 1] = True
    perimeter = (
        int(np.sum(img[:-1, :] != img[1:, :]))
        + int(np.sum(img[:, :-1] != img[:, 1:]))
    )
    if perimeter == 0:
        return 1.0
    return min(1.0, (4.0 * np.pi * area) / (perimeter ** 2))


def _solidity(ys: np.ndarray, xs: np.ndarray, area: int) -> float:
    """area / convex_hull_area; penalises holes and concave indentations."""
    pts = np.column_stack([ys, xs])
    if len(pts) < 3:
        return 1.0
    try:
        hull = ConvexHull(pts)
        return min(1.0, area / hull.volume)  # hull.volume == area in 2-D
    except Exception:
        return 1.0


def _iou_position_corrected(
    cur_ys: np.ndarray, cur_xs: np.ndarray, cur_area: int,
    cand_ys: np.ndarray, cand_xs: np.ndarray, cand_area: int,
) -> float:
    """IoU after aligning candidate centroid to current centroid (pure shape similarity)."""
    dy = int(round(float(cur_ys.mean()) - float(cand_ys.mean())))
    dx = int(round(float(cur_xs.mean()) - float(cand_xs.mean())))
    cur_set = set(zip(cur_ys.tolist(), cur_xs.tolist()))
    cand_set = set(zip((cand_ys + dy).tolist(), (cand_xs + dx).tolist()))
    inter = len(cur_set & cand_set)
    union = cur_area + cand_area - inter
    return inter / union if union > 0 else 0.0


def find_best_hypothesis(
    current_labels: np.ndarray,
    candidates: list[np.ndarray],
    max_dist_px: float = 50.0,
    predicted_centroids: dict[int, np.ndarray] | None = None,
    area_weight: float = 1.0,
    iou_weight: float = 1.0,
    circularity_weight: float = 1.0,
    solidity_weight: float = 1.0,
    dedup_radius_px: float = 0.0,  # no longer used; kept for API compatibility
) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Return (relabeled_next_frame, winning_entry_index) or (None, None).

    Candidates are gated by distance from the predicted position (or current
    centroid when no prediction is available), then ranked by an additive score:

        score = area_weight * area_ratio
              + iou_weight  * position_corrected_iou
              + circularity_weight * circularity_ratio
              + solidity_weight    * solidity
    """
    if not candidates:
        return None, None

    H, W = current_labels.shape
    cur_areas, cur_centroids = _label_stats(current_labels)
    cur_ids = sorted(cur_centroids.keys())
    if not cur_ids:
        return None, None

    cur_pixels = _nucleus_pixels(current_labels)

    # Build flat list of all (entry_idx, cand_id, centroid, area, flat_pixel_idx, ys, xs).
    flat_cands: list[tuple[int, int, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for entry_idx, cand in enumerate(candidates):
        c_areas, c_centroids = _label_stats(cand)
        c_pixels = _nucleus_pixels(cand)
        for cid, centroid in c_centroids.items():
            cys, cxs = c_pixels[cid]
            flat_idx = cys * W + cxs
            flat_cands.append((entry_idx, int(cid), centroid, int(c_areas[cid]), flat_idx, cys, cxs))

    if not flat_cands:
        return None, None

    cand_centroids_arr = np.vstack([c[2] for c in flat_cands])
    tree = KDTree(cand_centroids_arr)

    next_frame = np.zeros_like(current_labels)
    matched_entry_indices: list[int] = []

    for cur_id in cur_ids:
        cur_centroid = cur_centroids[cur_id]
        cur_area = int(cur_areas[cur_id])
        cur_ys, cur_xs = cur_pixels[cur_id]
        pred_centroid = (predicted_centroids or {}).get(cur_id)

        gate_centroid = pred_centroid if pred_centroid is not None else cur_centroid
        nearby_ks = tree.query_ball_point(gate_centroid, max_dist_px)
        if not nearby_ks:
            continue

        cur_circ = _circularity(cur_ys, cur_xs, cur_area)

        best_score = -1.0
        best_k = -1

        for k in nearby_ks:
            entry_idx, cand_id, cand_centroid, cand_area, cand_flat_idx, cand_ys, cand_xs = flat_cands[k]

            area_ratio = (
                min(cur_area, cand_area) / max(cur_area, cand_area)
                if max(cur_area, cand_area) > 0 else 1.0
            )
            pos_iou = _iou_position_corrected(cur_ys, cur_xs, cur_area, cand_ys, cand_xs, cand_area)
            cand_circ = _circularity(cand_ys, cand_xs, cand_area)
            circ_ratio = (
                min(cur_circ, cand_circ) / max(cur_circ, cand_circ)
                if max(cur_circ, cand_circ) > 0 else 1.0
            )
            sol = _solidity(cand_ys, cand_xs, cand_area)

            score = (
                area_weight        * area_ratio
                + iou_weight       * pos_iou
                + circularity_weight * circ_ratio
                + solidity_weight  * sol
            )

            if score > best_score:
                best_score = score
                best_k = k

        if best_k >= 0:
            entry_idx, cand_id, cand_centroid, cand_area, cand_flat_idx, cand_ys, cand_xs = flat_cands[best_k]
            next_frame[candidates[entry_idx] == cand_id] = cur_id
            matched_entry_indices.append(entry_idx)

    if not matched_entry_indices:
        return None, None

    winning_entry = Counter(matched_entry_indices).most_common(1)[0][0]
    return next_frame, winning_entry


def propagate_one_frame(
    hypotheses_h5: str | Path,
    current_labels: np.ndarray,
    t_next: int,
    prev_labels: np.ndarray | None = None,
    max_dist_px: float = 50.0,
    area_weight: float = 1.0,
    iou_weight: float = 1.0,
    circularity_weight: float = 1.0,
    solidity_weight: float = 1.0,
) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Propagate tracking to t_next using current_labels as the source frame.

    Searches all hypotheses in the hypothesis database for t_next, matches each
    tracked nucleus to its best candidate via greedy per-nucleus scoring.

    Returns (relabeled_next_frame, winning_p) or (None, None) if no matches.
    Does not read from or write to any tracked labels file.
    """
    hypotheses_h5 = Path(hypotheses_h5)

    n_p, params_by_p = list_hypotheses(hypotheses_h5)
    if n_p == 0:
        return None, None

    # Derive per-nucleus velocity from the previous frame if provided.
    predicted_centroids: dict[int, np.ndarray] | None = None
    if prev_labels is not None:
        try:
            _, prev_centroids = _label_stats(prev_labels)
            _, cur_centroids = _label_stats(current_labels)
            predicted_centroids = {
                lid: cur_centroids[lid] + (cur_centroids[lid] - prev_centroids[lid])
                for lid in cur_centroids
                if lid in prev_centroids
            }
        except Exception:
            pass

    entries: list[tuple[int, int, np.ndarray]] = []
    for p in params_by_p.keys():
        try:
            volume = read_hypothesis_labels(hypotheses_h5, t_next, p)  # (Z, Y, X) with Z=1
        except (KeyError, ValueError):
            continue
        for z in range(volume.shape[0]):
            entries.append((p, z, volume[z]))

    if not entries:
        return None, None

    candidates = [e[2] for e in entries]
    next_frame, winner_idx = find_best_hypothesis(
        current_labels, candidates, max_dist_px,
        predicted_centroids=predicted_centroids,
        area_weight=area_weight,
        iou_weight=iou_weight,
        circularity_weight=circularity_weight,
        solidity_weight=solidity_weight,
    )
    if next_frame is None or winner_idx is None:
        return None, None

    p_win, _z_win, _slice = entries[winner_idx]
    return next_frame, p_win


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking/retracker.py
--------------------------------------------------------------------------------

"""Centroid-distance LAP retracker for relabelling corrected frames.

Given a reference frame (whose IDs are trusted) and a target frame (which may
contain arbitrarily-assigned IDs after manual correction), this module remaps
the target IDs so that cells matching the reference keep the reference ID.

Unmatched target cells (new appearances) receive fresh IDs that do not collide
with any ID already present in either frame.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.optimize import linear_sum_assignment


def _centroids(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Return {label_id: centroid_yx} for all non-zero labels."""
    ids = [int(i) for i in np.unique(labels) if i != 0]
    if not ids:
        return {}
    coms = center_of_mass(np.ones_like(labels), labels, ids)
    if len(ids) == 1:
        coms = [coms]
    return {lid: np.array(com).ravel() for lid, com in zip(ids, coms)}


def retrack_frame(
    ref_labels: np.ndarray,
    target_labels: np.ndarray,
    max_dist_px: float = 50.0,
) -> np.ndarray:
    """Remap cell IDs in *target_labels* to match *ref_labels* by centroid proximity.

    Each target cell is matched to the nearest reference cell within max_dist_px
    using the Hungarian algorithm.  Matched target cells receive the reference ID.
    Unmatched target cells (new appearances) receive IDs above the current maximum
    to avoid collisions.

    Parameters
    ----------
    ref_labels:
        (Y, X) uint32 — the trusted reference frame.
    target_labels:
        (Y, X) uint32 — the frame to relabel (e.g. after manual correction).
    max_dist_px:
        Maximum centroid distance for a valid match.  Pairs further apart are
        treated as unmatched.

    Returns
    -------
    (Y, X) uint32 array with IDs remapped to match ref_labels where possible.
    """
    ref_centroids = _centroids(ref_labels)
    tgt_centroids = _centroids(target_labels)

    if not tgt_centroids:
        return target_labels.copy()

    result = np.zeros_like(target_labels)

    if not ref_centroids:
        # No reference cells — assign fresh sequential IDs to all target cells.
        next_id = 1
        for tid in tgt_centroids:
            result[target_labels == tid] = next_id
            next_id += 1
        return result

    ref_ids = list(ref_centroids.keys())
    tgt_ids = list(tgt_centroids.keys())
    ref_pts = np.array([ref_centroids[i] for i in ref_ids])
    tgt_pts = np.array([tgt_centroids[i] for i in tgt_ids])

    n_ref, n_tgt = len(ref_ids), len(tgt_ids)

    # Cost matrix: rows = target cells, cols = reference cells.
    cost = np.full((n_tgt, n_ref), fill_value=np.inf)
    for ti, tp in enumerate(tgt_pts):
        for ri, rp in enumerate(ref_pts):
            d = float(np.linalg.norm(tp - rp))
            if d <= max_dist_px:
                cost[ti, ri] = d

    # Solve assignment (minimise cost).  Pairs where cost is inf are blocked by
    # replacing inf with a large finite sentinel so scipy doesn't reject the matrix.
    sentinel = max_dist_px * 10 * (n_tgt + n_ref + 1)
    finite_cost = np.where(np.isinf(cost), sentinel, cost)
    row_ind, col_ind = linear_sum_assignment(finite_cost)

    # Build remapping: target_id -> new_id
    remap: dict[int, int] = {}
    used_ref_ids: set[int] = set()
    for ti, ri in zip(row_ind, col_ind):
        if cost[ti, ri] <= max_dist_px:
            remap[tgt_ids[ti]] = ref_ids[ri]
            used_ref_ids.add(ref_ids[ri])

    # Assign fresh IDs to unmatched target cells, above the current max.
    max_existing = max(
        int(ref_labels.max()) if ref_labels.max() > 0 else 0,
        int(target_labels.max()) if target_labels.max() > 0 else 0,
    )
    next_id = max_existing + 1
    for tid in tgt_ids:
        if tid not in remap:
            remap[tid] = next_id
            next_id += 1

    # Apply remapping.
    for tid, new_id in remap.items():
        result[target_labels == tid] = new_id

    return result


def retrack_frame_constrained(
    ref_labels: np.ndarray,
    target_labels: np.ndarray,
    locked_target_ids: set[int],
    max_dist_px: float = 50.0,
    reserved_ids: set[int] | None = None,
) -> np.ndarray:
    """Like retrack_frame, but target cells whose ID is in locked_target_ids
    keep their existing IDs unchanged. The IDs they hold are also reserved —
    no other (non-locked) target cell may be remapped onto them.

    Locked target cells are copied straight to the output and excluded from the
    LAP entirely. Reference cells that share an ID with a locked target cell are
    also excluded from the LAP — their ID is already occupied, so letting the
    LAP assign it to an unlocked cell would create a collision.

    Additional reserved_ids are protected from assignment to unlocked targets
    even if they are present in the reference frame.
    """
    locked_target_ids = set(locked_target_ids)  # defensive copy
    reserved_ids = set(reserved_ids or set())
    blocked_ids = locked_target_ids | reserved_ids

    ref_centroids = _centroids(ref_labels)
    tgt_centroids = _centroids(target_labels)

    result = np.zeros_like(target_labels)

    # Copy locked cells directly to output first; they participate in nothing else.
    for lid in locked_target_ids:
        if lid in tgt_centroids:
            result[target_labels == lid] = lid

    unlocked_tgt_ids = [tid for tid in tgt_centroids if tid not in locked_target_ids]

    if not unlocked_tgt_ids:
        return result

    # Protected IDs are excluded so the LAP cannot assign them to an unlocked
    # target cell.
    available_ref_ids = [rid for rid in ref_centroids if rid not in blocked_ids]

    if not available_ref_ids:
        # No usable reference cells — assign fresh IDs to all unlocked targets.
        max_existing = max(
            int(ref_labels.max()) if ref_labels.max() > 0 else 0,
            int(target_labels.max()) if target_labels.max() > 0 else 0,
        )
        next_id = max_existing + 1
        for tid in unlocked_tgt_ids:
            while next_id in blocked_ids:
                next_id += 1
            result[target_labels == tid] = next_id
            next_id += 1
        return result

    ref_pts = np.array([ref_centroids[i] for i in available_ref_ids])
    tgt_pts = np.array([tgt_centroids[i] for i in unlocked_tgt_ids])

    n_ref, n_tgt = len(available_ref_ids), len(unlocked_tgt_ids)

    cost = np.full((n_tgt, n_ref), fill_value=np.inf)
    for ti, tp in enumerate(tgt_pts):
        for ri, rp in enumerate(ref_pts):
            d = float(np.linalg.norm(tp - rp))
            if d <= max_dist_px:
                cost[ti, ri] = d

    sentinel = max_dist_px * 10 * (n_tgt + n_ref + 1)
    finite_cost = np.where(np.isinf(cost), sentinel, cost)
    row_ind, col_ind = linear_sum_assignment(finite_cost)

    remap: dict[int, int] = {}
    for ti, ri in zip(row_ind, col_ind):
        if cost[ti, ri] <= max_dist_px:
            remap[unlocked_tgt_ids[ti]] = available_ref_ids[ri]

    max_existing = max(
        int(ref_labels.max()) if ref_labels.max() > 0 else 0,
        int(target_labels.max()) if target_labels.max() > 0 else 0,
    )
    next_id = max_existing + 1
    for tid in unlocked_tgt_ids:
        if tid not in remap:
            while next_id in blocked_ids:
                next_id += 1
            remap[tid] = next_id
            next_id += 1

    for tid, new_id in remap.items():
        result[target_labels == tid] = new_id

    return result


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/__init__.py
--------------------------------------------------------------------------------

"""Ultrack-based ILP tracker for CellFlow v2 hypotheses."""
from __future__ import annotations

__all__ = ["TrackingConfig", "ingest_hypotheses_to_db"]


def __getattr__(name: str):
    if name == "TrackingConfig":
        from cellflow.tracking_ultrack.config import TrackingConfig

        return TrackingConfig
    if name == "ingest_hypotheses_to_db":
        from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db

        return ingest_hypotheses_to_db
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/anchor.py
--------------------------------------------------------------------------------

"""Anchor-frame constraints for Ultrack solves."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlalchemy as sqla
from skimage.measure import regionprops
from sqlalchemy.orm import Session


@dataclass(frozen=True)
class AnchorReport:
    frame_index: int
    n_gt_labels: int
    n_matched: int
    n_unmatched: int
    matched_node_ids: list[int]
    unmatched_labels: list[int]
    mean_matched_iou: float
    min_matched_iou: float


@dataclass(frozen=True)
class AnchorSuppressionReport:
    frame_index: int
    neighbor_offsets: tuple[int, ...]
    suppressed_node_ids: list[int]
    by_frame: dict[int, int]


@dataclass(frozen=True)
class _MaskRecord:
    label_id: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray


def _frame_2d(labels: np.ndarray, frame_index: int) -> np.ndarray:
    arr = np.asarray(labels)
    frame = arr[frame_index]
    if frame.ndim == 3:
        if frame.shape[0] == 1:
            return frame[0]
        raise NotImplementedError("Anchor matching for true 3D labels is not implemented")
    if frame.ndim == 2:
        return frame
    raise ValueError(f"Expected frame to be 2D or singleton-Z 3D, got {frame.shape}")


def _gt_masks(labels: np.ndarray, frame_index: int) -> list[_MaskRecord]:
    frame = _frame_2d(labels, frame_index)
    masks: list[_MaskRecord] = []
    for prop in regionprops(frame):
        y0, x0, y1, x1 = prop.bbox
        mask = np.ascontiguousarray(frame[y0:y1, x0:x1] == prop.label, dtype=bool)
        masks.append(_MaskRecord(int(prop.label), (int(y0), int(x0), int(y1), int(x1)), mask))
    return masks


def _node_mask_record(node_id: int, node) -> _MaskRecord:
    bbox = np.asarray(node.bbox)
    ndim = len(bbox) // 2
    if ndim == 3:
        y0, x0, y1, x1 = int(bbox[1]), int(bbox[2]), int(bbox[4]), int(bbox[5])
    elif ndim == 2:
        y0, x0, y1, x1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    else:
        raise ValueError(f"Unexpected node bbox shape for node {node_id}: {bbox}")

    mask = np.asarray(node.mask, dtype=bool)
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            mask = mask.any(axis=0)
    if mask.ndim != 2:
        raise ValueError(f"Unexpected node mask shape for node {node_id}: {mask.shape}")
    return _MaskRecord(int(node_id), (y0, x0, y1, x1), np.ascontiguousarray(mask, dtype=bool))


def _mask_iou(lhs: _MaskRecord, rhs: _MaskRecord) -> float:
    intersection = _mask_intersection(lhs, rhs)
    lhs_area = int(lhs.mask.sum())
    rhs_area = int(rhs.mask.sum())
    if lhs_area == 0 and rhs_area == 0:
        return 1.0
    if lhs_area == 0 or rhs_area == 0:
        return 0.0
    union = lhs_area + rhs_area - intersection
    return float(intersection / union) if union else 0.0


def _mask_intersection(lhs: _MaskRecord, rhs: _MaskRecord) -> int:
    ly0, lx0, ly1, lx1 = lhs.bbox
    ry0, rx0, ry1, rx1 = rhs.bbox
    oy0, ox0 = max(ly0, ry0), max(lx0, rx0)
    oy1, ox1 = min(ly1, ry1), min(lx1, rx1)

    intersection = 0
    if oy0 < oy1 and ox0 < ox1:
        lhs_crop = lhs.mask[oy0 - ly0: oy1 - ly0, ox0 - lx0: ox1 - lx0]
        rhs_crop = rhs.mask[oy0 - ry0: oy1 - ry0, ox0 - rx0: ox1 - rx0]
        intersection = int(np.logical_and(lhs_crop, rhs_crop).sum())
    return intersection


def _node_containment_in_anchor(anchor: _MaskRecord, node: _MaskRecord) -> float:
    node_area = int(node.mask.sum())
    if node_area == 0:
        return 0.0
    return float(_mask_intersection(anchor, node) / node_area)


def annotate_anchor_frame(
    working_dir: str | Path,
    anchor_labels: np.ndarray,
    *,
    frame_index: int,
    min_iou: float = 0.95,
) -> AnchorReport:
    """Pin the solver's selected nodes at one frame to match an anchor labelmap.

    Matched nodes at ``frame_index`` are marked ``VarAnnotation.REAL``. Every
    other node in the same frame is marked ``VarAnnotation.FAKE`` so the anchor
    frame cannot contain extra non-GT cells. Other frames are left unannotated.
    """
    from ultrack.core.database import NodeDB, VarAnnotation

    working_dir = Path(working_dir)
    db_url = f"sqlite:///{working_dir / 'data.db'}"
    gt_records = _gt_masks(anchor_labels, frame_index)
    matched_node_ids: list[int] = []
    matched_ious: list[float] = []
    unmatched_labels: list[int] = []

    engine = sqla.create_engine(db_url)
    with Session(engine) as session:
        node_rows = (
            session.query(NodeDB.id, NodeDB.pickle)
            .where(NodeDB.t == frame_index)
            .all()
        )
        node_records = [
            _node_mask_record(int(node_id), node)
            for node_id, node in node_rows
        ]

        available = {rec.label_id for rec in node_records}
        by_id = {rec.label_id: rec for rec in node_records}

        candidates: list[tuple[float, int, int]] = []
        for gt in gt_records:
            for node in node_records:
                iou = _mask_iou(gt, node)
                if iou >= min_iou:
                    candidates.append((iou, gt.label_id, node.label_id))

        matched_gt: set[int] = set()
        for iou, gt_label, node_id in sorted(candidates, reverse=True):
            if gt_label in matched_gt or node_id not in available:
                continue
            matched_gt.add(gt_label)
            available.remove(node_id)
            matched_node_ids.append(node_id)
            matched_ious.append(iou)

        for gt in gt_records:
            if gt.label_id not in matched_gt:
                unmatched_labels.append(gt.label_id)

        session.query(NodeDB).where(NodeDB.t == frame_index).update(
            {NodeDB.node_annot: VarAnnotation.FAKE},
            synchronize_session=False,
        )
        if matched_node_ids:
            session.query(NodeDB).where(NodeDB.id.in_(matched_node_ids)).update(
                {NodeDB.node_annot: VarAnnotation.REAL},
                synchronize_session=False,
            )
        session.commit()

    matched_node_ids.sort()
    unmatched_labels.sort()
    return AnchorReport(
        frame_index=int(frame_index),
        n_gt_labels=len(gt_records),
        n_matched=len(matched_node_ids),
        n_unmatched=len(unmatched_labels),
        matched_node_ids=matched_node_ids,
        unmatched_labels=unmatched_labels,
        mean_matched_iou=float(np.mean(matched_ious)) if matched_ious else 0.0,
        min_matched_iou=float(np.min(matched_ious)) if matched_ious else 0.0,
    )


def suppress_anchor_adjacent_fragments(
    working_dir: str | Path,
    anchor_labels: np.ndarray,
    *,
    frame_index: int,
    neighbor_offsets: tuple[int, ...] = (-1, 1),
    min_best_iou: float = 0.60,
    fragment_max_iou_fraction: float = 0.80,
    min_fragment_containment: float = 0.90,
) -> AnchorSuppressionReport:
    """Suppress obvious fragment alternatives next to an anchored frame.

    For each anchored object, each neighboring frame keeps its best-overlapping
    candidate unconstrained. Other candidates are marked FAKE only when they are
    mostly contained inside the anchored object and have substantially lower IoU
    than the best candidate. Ambiguous candidates are left untouched.
    """
    from ultrack.core.database import NodeDB, VarAnnotation

    working_dir = Path(working_dir)
    db_url = f"sqlite:///{working_dir / 'data.db'}"
    anchor_records = _gt_masks(anchor_labels, frame_index)
    suppressed: set[int] = set()
    by_frame: dict[int, int] = {}

    engine = sqla.create_engine(db_url)
    with Session(engine) as session:
        for offset in neighbor_offsets:
            t = int(frame_index + offset)
            if t < 0:
                continue
            node_rows = (
                session.query(NodeDB.id, NodeDB.pickle, NodeDB.node_annot)
                .where(NodeDB.t == t)
                .all()
            )
            node_records = [
                _node_mask_record(int(node_id), node)
                for node_id, node, _annot in node_rows
            ]
            node_annots = {int(node_id): annot for node_id, _node, annot in node_rows}
            frame_suppressed: set[int] = set()

            for anchor in anchor_records:
                scored: list[tuple[float, float, int, _MaskRecord]] = []
                for node in node_records:
                    iou = _mask_iou(anchor, node)
                    containment = _node_containment_in_anchor(anchor, node)
                    if iou > 0.0 or containment > 0.0:
                        scored.append((iou, containment, node.label_id, node))
                if not scored:
                    continue

                best_iou, _best_containment, best_id, best_node = max(
                    scored,
                    key=lambda item: (item[0], item[1]),
                )
                if best_iou < min_best_iou:
                    continue
                best_area = int(best_node.mask.sum())
                iou_cutoff = best_iou * fragment_max_iou_fraction

                for iou, containment, node_id, node in scored:
                    if node_id == best_id:
                        continue
                    if node_annots.get(node_id) == VarAnnotation.REAL:
                        continue
                    node_area = int(node.mask.sum())
                    if (
                        containment >= min_fragment_containment
                        and iou <= iou_cutoff
                        and node_area < best_area
                    ):
                        frame_suppressed.add(node_id)

            if frame_suppressed:
                session.query(NodeDB).where(NodeDB.id.in_(frame_suppressed)).update(
                    {NodeDB.node_annot: VarAnnotation.FAKE},
                    synchronize_session=False,
                )
                suppressed.update(frame_suppressed)
                by_frame[t] = len(frame_suppressed)

        session.commit()

    return AnchorSuppressionReport(
        frame_index=int(frame_index),
        neighbor_offsets=tuple(int(offset) for offset in neighbor_offsets),
        suppressed_node_ids=sorted(suppressed),
        by_frame=dict(sorted(by_frame.items())),
    )


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/anchor_diagnostics.py
--------------------------------------------------------------------------------

"""Diagnostics for anchored Ultrack solves."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from cellflow.tracking_ultrack.anchor import _gt_masks, _mask_iou, _node_mask_record


@dataclass(frozen=True)
class SelectedOverlap:
    node_id: int
    iou: float
    area: int


@dataclass(frozen=True)
class AnchorCandidateMatch:
    gt_label: int
    best_node_id: int | None
    best_iou: float
    selected: bool
    node_annot: Any
    incoming_link_count: int = 0
    best_incoming_weight: float | None = None
    outgoing_link_count: int = 0
    best_outgoing_weight: float | None = None
    selected_overlaps: list[SelectedOverlap] | None = None


@dataclass(frozen=True)
class AnchorCandidateDiagnostics:
    frame_index: int
    matches: list[AnchorCandidateMatch]


def diagnose_anchor_frame_candidates(
    working_dir: str | Path,
    labels: np.ndarray,
    *,
    frame_index: int,
) -> AnchorCandidateDiagnostics:
    """Report each GT object's best matching NodeDB candidate at one frame."""
    from ultrack.core.database import LinkDB, NodeDB, OverlapDB

    db_url = f"sqlite:///{Path(working_dir) / 'data.db'}"
    gt_records = _gt_masks(labels, frame_index)

    engine = sqla.create_engine(db_url)
    with Session(engine) as session:
        rows = (
            session.query(NodeDB.id, NodeDB.pickle, NodeDB.selected, NodeDB.node_annot)
            .where(NodeDB.t == frame_index)
            .all()
        )
        node_ids = [int(node_id) for node_id, _node, _selected, _annot in rows]
        link_rows = []
        overlap_rows = []
        if node_ids:
            link_rows = (
                session.query(LinkDB.source_id, LinkDB.target_id, LinkDB.weight)
                .where(
                    sqla.or_(
                        LinkDB.source_id.in_(node_ids),
                        LinkDB.target_id.in_(node_ids),
                    )
                )
                .all()
            )
            overlap_rows = (
                session.query(OverlapDB.node_id, OverlapDB.ancestor_id)
                .where(
                    sqla.or_(
                        OverlapDB.node_id.in_(node_ids),
                        OverlapDB.ancestor_id.in_(node_ids),
                    )
                )
                .all()
            )

    node_records = [
        (
            _node_mask_record(int(node_id), node),
            bool(selected),
            node_annot,
        )
        for node_id, node, selected, node_annot in rows
    ]
    node_by_id = {node.label_id: (node, selected, node_annot) for node, selected, node_annot in node_records}

    incoming_weights: dict[int, list[float]] = {node_id: [] for node_id in node_by_id}
    outgoing_weights: dict[int, list[float]] = {node_id: [] for node_id in node_by_id}
    for source_id, target_id, weight in link_rows:
        source_id = int(source_id)
        target_id = int(target_id)
        if target_id in incoming_weights:
            incoming_weights[target_id].append(float(weight))
        if source_id in outgoing_weights:
            outgoing_weights[source_id].append(float(weight))

    selected_overlaps_by_id: dict[int, list[SelectedOverlap]] = {node_id: [] for node_id in node_by_id}
    for node_id, ancestor_id in overlap_rows:
        lhs_id = int(node_id)
        rhs_id = int(ancestor_id)
        for current_id, other_id in ((lhs_id, rhs_id), (rhs_id, lhs_id)):
            current = node_by_id.get(current_id)
            other = node_by_id.get(other_id)
            if current is None or other is None:
                continue
            other_node, other_selected, _other_annot = other
            if not other_selected:
                continue
            current_node, _current_selected, _current_annot = current
            selected_overlaps_by_id[current_id].append(
                SelectedOverlap(
                    node_id=other_id,
                    iou=float(_mask_iou(current_node, other_node)),
                    area=int(other_node.mask.sum()),
                )
            )

    matches: list[AnchorCandidateMatch] = []
    for gt in gt_records:
        best: tuple[float, int, bool, Any] | None = None
        for node, selected, node_annot in node_records:
            iou = _mask_iou(gt, node)
            candidate = (iou, node.label_id, selected, node_annot)
            if best is None or candidate[:2] > best[:2]:
                best = candidate

        if best is None:
            matches.append(
                AnchorCandidateMatch(
                    gt_label=gt.label_id,
                    best_node_id=None,
                    best_iou=0.0,
                    selected=False,
                    node_annot=None,
                    selected_overlaps=[],
                )
            )
            continue

        best_iou, best_node_id, selected, node_annot = best
        incoming = incoming_weights.get(best_node_id, [])
        outgoing = outgoing_weights.get(best_node_id, [])
        matches.append(
            AnchorCandidateMatch(
                gt_label=gt.label_id,
                best_node_id=best_node_id,
                best_iou=float(best_iou),
                selected=selected,
                node_annot=node_annot,
                incoming_link_count=len(incoming),
                best_incoming_weight=max(incoming) if incoming else None,
                outgoing_link_count=len(outgoing),
                best_outgoing_weight=max(outgoing) if outgoing else None,
                selected_overlaps=sorted(
                    selected_overlaps_by_id.get(best_node_id, []),
                    key=lambda overlap: overlap.node_id,
                ),
            )
        )

    return AnchorCandidateDiagnostics(frame_index=int(frame_index), matches=matches)


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/cell_boundary_selection.py
--------------------------------------------------------------------------------

"""Track-conditioned cell boundary selection utilities.

This module implements the phase-1 pure-Python backend from the design: given
per-frame candidate masks and existing nucleus track labels, choose one
candidate per known track-frame with a small dynamic-programming solver.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import tifffile


@dataclass(frozen=True, slots=True)
class BoundaryCandidate:
    """One cell-boundary candidate mask for a single frame."""

    node_id: int
    t: int
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class BoundarySelectionParams:
    """Weights and thresholds for per-track boundary dynamic programming."""

    min_nucleus_fraction: float = 0.0
    anchor_weight: float = 2.0
    conflict_weight: float = 0.25
    centroid_jump_weight: float = 0.25
    area_change_weight: float = 0.5
    iou_loss_weight: float = 1.0
    missing_penalty: float = 10.0


@dataclass(frozen=True, slots=True)
class AnchorScore:
    track_id: int
    nucleus_pixels: int
    covered_pixels: int
    fraction: float
    other_nucleus_pixels: int


@dataclass(frozen=True, slots=True)
class BoundarySelectionResult:
    track_id: int
    selected_node_ids: dict[int, int | None]
    total_score: float

    @property
    def missing_frames(self) -> set[int]:
        return {t for t, node_id in self.selected_node_ids.items() if node_id is None}


@dataclass(frozen=True, slots=True)
class OverlapConflict:
    t: int
    track_ids: tuple[int, int]
    node_ids: tuple[int, int]
    overlap_pixels: int


def _normalize_tyx_array(array: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr[np.newaxis]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(
                f"{name} has unsupported non-singleton Z/channel axis: {arr.shape}"
            )
        return arr[:, 0]
    raise ValueError(f"{name} has unsupported ndim {arr.ndim}: {arr.shape}")


def _read_required_tiff(path: str | Path, *, name: str) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return np.asarray(tifffile.imread(str(path)))


def validate_cell_boundary_inputs(
    contour_maps_path: str | Path,
    foreground_masks_path: str | Path,
    nucleus_tracked_labels_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate boundary-selection TIFF inputs as ``(T, Y, X)`` arrays."""
    contours = _normalize_tyx_array(
        _read_required_tiff(contour_maps_path, name="cell contour maps"),
        name="cell contour maps",
    ).astype(np.float32, copy=False)
    foreground = _normalize_tyx_array(
        _read_required_tiff(foreground_masks_path, name="cell foreground masks"),
        name="cell foreground masks",
    ).astype(bool, copy=False)
    nuclei = _normalize_tyx_array(
        _read_required_tiff(nucleus_tracked_labels_path, name="nucleus tracked labels"),
        name="nucleus tracked labels",
    ).astype(np.uint32, copy=False)

    if contours.shape != foreground.shape:
        raise ValueError(
            "cell contour maps and foreground masks shape mismatch: "
            f"{contours.shape} != {foreground.shape}"
        )
    if contours.shape != nuclei.shape:
        raise ValueError(
            "cell boundary inputs and nucleus tracked labels shape mismatch: "
            f"{contours.shape} != {nuclei.shape}"
        )
    return contours, foreground, nuclei


def _data_db_path(working_dir: str | Path) -> Path:
    path = Path(working_dir)
    if path.name == "data.db":
        return path
    return path / "data.db"


def load_candidates_from_db(working_dir: str | Path) -> list[BoundaryCandidate]:
    """Read Ultrack ``NodeDB`` rows from ``working_dir/data.db`` as candidates."""
    db_path = _data_db_path(working_dir)
    if not db_path.exists():
        raise FileNotFoundError(f"Ultrack data.db not found: {db_path}")

    try:
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
    except ImportError as exc:
        raise ImportError(
            "sqlalchemy and ultrack must be installed to load candidates from data.db"
        ) from exc

    from cellflow.tracking_ultrack.validation_nodes import _node_bbox_and_mask

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            rows = (
                session.query(NodeDB.id, NodeDB.t, NodeDB.pickle, NodeDB.node_prob)
                .order_by(NodeDB.t, NodeDB.id)
                .all()
            )
    finally:
        engine.dispose()

    candidates: list[BoundaryCandidate] = []
    for node_id, t, node_pickle, node_prob in rows:
        bbox, mask = _node_bbox_and_mask(int(node_id), node_pickle)
        raw_score = None if node_prob is None else float(node_prob)
        score = 0.0 if raw_score is None or raw_score < 0.0 else raw_score
        candidates.append(
            BoundaryCandidate(
                node_id=int(node_id),
                t=int(t),
                mask=np.ascontiguousarray(mask, dtype=bool),
                bbox=tuple(int(v) for v in bbox),
                score=score,
            )
        )
    return candidates


def _bbox_slices(bbox: tuple[int, int, int, int]) -> tuple[slice, slice]:
    y0, x0, y1, x1 = bbox
    return slice(int(y0), int(y1)), slice(int(x0), int(x1))


def _candidate_full_mask(candidate: BoundaryCandidate, shape: tuple[int, int]) -> np.ndarray:
    full = np.zeros(shape, dtype=bool)
    y_slice, x_slice = _bbox_slices(candidate.bbox)
    crop = np.asarray(candidate.mask, dtype=bool)
    expected_shape = _validate_candidate_geometry(candidate, shape)
    if crop.shape != expected_shape:
        raise ValueError(
            f"candidate {candidate.node_id} mask shape {crop.shape} does not match "
            f"bbox shape {expected_shape}"
        )
    full[y_slice, x_slice] = crop
    return full


def _validate_candidate_geometry(
    candidate: BoundaryCandidate,
    frame_shape: tuple[int, int],
) -> tuple[int, int]:
    y0, x0, y1, x1 = candidate.bbox
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if y0 < 0 or x0 < 0 or y1 > height or x1 > width or y0 >= y1 or x0 >= x1:
        raise ValueError(
            f"candidate {candidate.node_id} bbox {candidate.bbox} is outside "
            f"output shape {frame_shape}"
        )
    expected_shape = (int(y1 - y0), int(x1 - x0))
    crop = np.asarray(candidate.mask, dtype=bool)
    if crop.shape != expected_shape:
        raise ValueError(
            f"candidate {candidate.node_id} mask shape {crop.shape} does not match "
            f"bbox shape {expected_shape}"
        )
    return expected_shape


def _candidate_centroid(candidate: BoundaryCandidate) -> tuple[float, float]:
    mask = np.asarray(candidate.mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        y0, x0, y1, x1 = candidate.bbox
        return ((y0 + y1) / 2.0, (x0 + x1) / 2.0)
    return (
        float(candidate.bbox[0] + ys.mean()),
        float(candidate.bbox[1] + xs.mean()),
    )


def _candidate_area(candidate: BoundaryCandidate) -> int:
    return int(np.asarray(candidate.mask, dtype=bool).sum())


def score_candidate_anchor(
    candidate: BoundaryCandidate,
    nucleus_frame: np.ndarray,
    track_id: int,
) -> AnchorScore:
    """Return nucleus anchor coverage for ``candidate`` and ``track_id``."""
    frame = np.asarray(nucleus_frame)
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D nucleus frame, got shape {frame.shape}")

    nucleus = frame == int(track_id)
    nucleus_pixels = int(nucleus.sum())
    if nucleus_pixels == 0:
        return AnchorScore(int(track_id), 0, 0, 0.0, 0)

    full_mask = _candidate_full_mask(candidate, frame.shape)
    covered = int(np.logical_and(full_mask, nucleus).sum())
    other_nucleus_pixels = int(np.logical_and(full_mask, (frame > 0) & ~nucleus).sum())
    return AnchorScore(
        track_id=int(track_id),
        nucleus_pixels=nucleus_pixels,
        covered_pixels=covered,
        fraction=float(covered) / float(nucleus_pixels),
        other_nucleus_pixels=other_nucleus_pixels,
    )


def is_candidate_eligible_for_track(
    candidate: BoundaryCandidate,
    nucleus_frame: np.ndarray,
    track_id: int,
    *,
    min_nucleus_fraction: float = 0.0,
) -> bool:
    """Return true when the candidate satisfies the hard-anchor rule."""
    anchor = score_candidate_anchor(candidate, nucleus_frame, track_id)
    if anchor.covered_pixels <= 0:
        return False
    return anchor.fraction >= float(min_nucleus_fraction)


def _candidate_iou(lhs: BoundaryCandidate, rhs: BoundaryCandidate) -> float:
    ly0, lx0, ly1, lx1 = lhs.bbox
    ry0, rx0, ry1, rx1 = rhs.bbox
    oy0, ox0 = max(ly0, ry0), max(lx0, rx0)
    oy1, ox1 = min(ly1, ry1), min(lx1, rx1)
    intersection = 0
    if oy0 < oy1 and ox0 < ox1:
        lhs_crop = np.asarray(lhs.mask, dtype=bool)[oy0 - ly0: oy1 - ly0, ox0 - lx0: ox1 - lx0]
        rhs_crop = np.asarray(rhs.mask, dtype=bool)[oy0 - ry0: oy1 - ry0, ox0 - rx0: ox1 - rx0]
        intersection = int(np.logical_and(lhs_crop, rhs_crop).sum())
    union = _candidate_area(lhs) + _candidate_area(rhs) - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def _transition_score(
    previous: BoundaryCandidate | None,
    current: BoundaryCandidate | None,
    params: BoundarySelectionParams,
) -> float:
    if previous is None or current is None:
        return 0.0
    py, px = _candidate_centroid(previous)
    cy, cx = _candidate_centroid(current)
    centroid_penalty = float(np.hypot(cy - py, cx - px)) * params.centroid_jump_weight

    prev_area = max(_candidate_area(previous), 1)
    curr_area = max(_candidate_area(current), 1)
    area_ratio = abs(np.log(float(curr_area) / float(prev_area)))
    area_penalty = area_ratio * params.area_change_weight

    iou_penalty = (1.0 - _candidate_iou(previous, current)) * params.iou_loss_weight
    return -(centroid_penalty + area_penalty + iou_penalty)


def _unary_score(
    candidate: BoundaryCandidate | None,
    nucleus_frame: np.ndarray,
    track_id: int,
    params: BoundarySelectionParams,
) -> float:
    if candidate is None:
        return -float(params.missing_penalty)
    anchor = score_candidate_anchor(candidate, nucleus_frame, track_id)
    conflict_penalty = anchor.other_nucleus_pixels * params.conflict_weight
    return float(candidate.score) + anchor.fraction * params.anchor_weight - conflict_penalty


def select_track_boundaries_dp(
    candidates: list[BoundaryCandidate],
    nucleus_labels: np.ndarray,
    track_id: int,
    params: BoundarySelectionParams | None = None,
) -> BoundarySelectionResult:
    """Select one candidate or missing state for each frame of a known track."""
    params = params or BoundarySelectionParams()
    labels = np.asarray(nucleus_labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected nucleus labels shaped (T, Y, X), got {labels.shape}")

    frames = [
        int(t)
        for t in range(labels.shape[0])
        if np.any(labels[t] == int(track_id))
    ]
    by_frame: dict[int, list[BoundaryCandidate | None]] = {}
    for t in frames:
        eligible = [
            candidate
            for candidate in candidates
            if int(candidate.t) == t
            and is_candidate_eligible_for_track(
                candidate,
                labels[t],
                track_id,
                min_nucleus_fraction=params.min_nucleus_fraction,
            )
        ]
        by_frame[t] = [None] + sorted(eligible, key=lambda candidate: int(candidate.node_id))

    if not frames:
        return BoundarySelectionResult(int(track_id), {}, 0.0)

    scores: dict[tuple[int, int], float] = {}
    back: dict[tuple[int, int], int | None] = {}
    first_t = frames[0]
    for idx, state in enumerate(by_frame[first_t]):
        scores[(first_t, idx)] = _unary_score(state, labels[first_t], track_id, params)
        back[(first_t, idx)] = None

    for t_prev, t_cur in zip(frames, frames[1:]):
        for cur_idx, cur_state in enumerate(by_frame[t_cur]):
            best_score = -np.inf
            best_prev_idx: int | None = None
            for prev_idx, prev_state in enumerate(by_frame[t_prev]):
                candidate_score = (
                    scores[(t_prev, prev_idx)]
                    + _transition_score(prev_state, cur_state, params)
                    + _unary_score(cur_state, labels[t_cur], track_id, params)
                )
                if candidate_score > best_score:
                    best_score = float(candidate_score)
                    best_prev_idx = prev_idx
            scores[(t_cur, cur_idx)] = best_score
            back[(t_cur, cur_idx)] = best_prev_idx

    last_t = frames[-1]
    last_idx = max(range(len(by_frame[last_t])), key=lambda idx: scores[(last_t, idx)])
    total_score = scores[(last_t, last_idx)]

    selected: dict[int, int | None] = {}
    idx: int | None = last_idx
    for frame_index in range(len(frames) - 1, -1, -1):
        t = frames[frame_index]
        state = by_frame[t][int(idx)]
        selected[t] = None if state is None else int(state.node_id)
        idx = back[(t, int(idx))]
    selected = dict(sorted(selected.items()))
    return BoundarySelectionResult(int(track_id), selected, total_score)


def select_all_track_boundaries(
    candidates: list[BoundaryCandidate],
    nucleus_labels: np.ndarray,
    *,
    track_ids: list[int] | None = None,
    params: BoundarySelectionParams | None = None,
) -> dict[int, BoundarySelectionResult]:
    """Run per-track boundary selection for each known nucleus track ID."""
    labels = np.asarray(nucleus_labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected nucleus labels shaped (T, Y, X), got {labels.shape}")
    if track_ids is None:
        track_ids = [
            int(track_id)
            for track_id in sorted(np.unique(labels))
            if int(track_id) != 0
        ]
    return {
        int(track_id): select_track_boundaries_dp(
            candidates,
            labels,
            int(track_id),
            params=params,
        )
        for track_id in track_ids
    }


def _selected_items(
    selections: dict[int, dict[int, int | None]],
    candidates: dict[int, BoundaryCandidate],
    t: int,
) -> list[tuple[int, BoundaryCandidate]]:
    items: list[tuple[int, BoundaryCandidate]] = []
    for track_id, by_frame in selections.items():
        node_id = by_frame.get(t)
        if node_id is None:
            continue
        items.append((int(track_id), candidates[int(node_id)]))
    return items


def _overlap_pixels(lhs: BoundaryCandidate, rhs: BoundaryCandidate) -> int:
    ly0, lx0, ly1, lx1 = lhs.bbox
    ry0, rx0, ry1, rx1 = rhs.bbox
    oy0, ox0 = max(ly0, ry0), max(lx0, rx0)
    oy1, ox1 = min(ly1, ry1), min(lx1, rx1)
    if oy0 >= oy1 or ox0 >= ox1:
        return 0
    lhs_crop = np.asarray(lhs.mask, dtype=bool)[oy0 - ly0: oy1 - ly0, ox0 - lx0: ox1 - lx0]
    rhs_crop = np.asarray(rhs.mask, dtype=bool)[oy0 - ry0: oy1 - ry0, ox0 - rx0: ox1 - rx0]
    return int(np.logical_and(lhs_crop, rhs_crop).sum())


def detect_overlap_conflicts(
    selections: dict[int, dict[int, int | None]],
    candidates: dict[int, BoundaryCandidate],
    *,
    min_overlap_pixels: int = 1,
) -> list[OverlapConflict]:
    """Return selected candidate overlaps between different tracks."""
    frames = sorted({t for by_frame in selections.values() for t in by_frame})
    conflicts: list[OverlapConflict] = []
    for t in frames:
        for (lhs_track, lhs), (rhs_track, rhs) in combinations(
            _selected_items(selections, candidates, t),
            2,
        ):
            overlap = _overlap_pixels(lhs, rhs)
            if overlap >= int(min_overlap_pixels):
                conflicts.append(
                    OverlapConflict(
                        t=int(t),
                        track_ids=(lhs_track, rhs_track),
                        node_ids=(int(lhs.node_id), int(rhs.node_id)),
                        overlap_pixels=overlap,
                    )
                )
    return conflicts


def export_selected_boundaries(
    selections: dict[int, dict[int, int | None]],
    candidates: dict[int, BoundaryCandidate],
    *,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Rasterize selected candidates with the original nucleus track IDs."""
    if len(shape) != 3:
        raise ValueError(f"Expected output shape (T, Y, X), got {shape}")
    labels = np.zeros(shape, dtype=np.uint32)
    for track_id in sorted(selections):
        for t, node_id in sorted(selections[track_id].items()):
            if node_id is None:
                continue
            candidate = candidates[int(node_id)]
            if int(candidate.t) != int(t):
                raise ValueError(
                    f"candidate {candidate.node_id} belongs to t={candidate.t}, "
                    f"not selected frame t={t}"
                )
            if int(t) < 0 or int(t) >= shape[0]:
                raise ValueError(
                    f"selected frame t={t} is outside output shape {shape}"
                )
            _validate_candidate_geometry(candidate, shape[1:])
            y_slice, x_slice = _bbox_slices(candidate.bbox)
            labels[int(t), y_slice, x_slice][np.asarray(candidate.mask, dtype=bool)] = int(track_id)
    return labels


@dataclass(frozen=True, slots=True)
class BoundarySelectionRunResult:
    """Result of a ``run_track_conditioned_boundary_selection`` invocation."""

    output_path: Path
    diagnostics_path: Path
    candidate_count: int
    missing_frames_by_track: dict[int, list[int]]
    overlap_conflicts: list[OverlapConflict]


def run_track_conditioned_boundary_selection(
    pos_dir: str | Path,
    *,
    candidate_loader: Callable[[str | Path], list[BoundaryCandidate]] | None = None,
    params: BoundarySelectionParams | None = None,
) -> BoundarySelectionRunResult:
    """Run the full track-conditioned boundary selection pipeline for a position.

    Resolves default input and output paths relative to *pos_dir*:

    * ``3_cell/contour_maps.tif``
    * ``3_cell/foreground_masks.tif``
    * ``2_nucleus/tracked_labels.tif``
    * ``3_cell/ultrack_workdir`` (Ultrack working directory)
    * ``3_cell/tracked_labels.tif`` (output)
    * ``3_cell/boundary_selection_diagnostics.json`` (diagnostics)

    Parameters
    ----------
    pos_dir:
        Root position directory containing ``3_cell`` and ``2_nucleus`` subdirs.
    candidate_loader:
        Callable that receives the working directory path and returns a list of
        ``BoundaryCandidate`` objects.  When *None* (default), the standard
        ``load_candidates_from_db`` loader is used.
    params:
        Optional :class:`BoundarySelectionParams` controlling the per-track DP.

    Returns
    -------
    BoundarySelectionRunResult
    """
    pos_dir = Path(pos_dir)
    cell_dir = pos_dir / "3_cell"
    nucleus_dir = pos_dir / "2_nucleus"

    contour_maps_path = cell_dir / "contour_maps.tif"
    foreground_masks_path = cell_dir / "foreground_masks.tif"
    nucleus_tracked_labels_path = nucleus_dir / "tracked_labels.tif"
    working_dir = cell_dir / "ultrack_workdir"
    output_path = cell_dir / "tracked_labels.tif"
    diagnostics_path = cell_dir / "boundary_selection_diagnostics.json"

    # --- validate inputs ---
    _, _, nuclei = validate_cell_boundary_inputs(
        contour_maps_path,
        foreground_masks_path,
        nucleus_tracked_labels_path,
    )

    # --- load candidates ---
    loader = candidate_loader if candidate_loader is not None else load_candidates_from_db
    candidates = loader(working_dir)

    # --- per-track DP ---
    results = select_all_track_boundaries(candidates, nuclei, params=params)

    selections: dict[int, dict[int, int | None]] = {
        track_id: result.selected_node_ids
        for track_id, result in results.items()
    }
    candidates_by_id: dict[int, BoundaryCandidate] = {
        candidate.node_id: candidate for candidate in candidates
    }

    # --- rasterize & write output ---
    labels = export_selected_boundaries(selections, candidates_by_id, shape=nuclei.shape)
    tifffile.imwrite(str(output_path), labels, compression="zlib")

    # --- overlap diagnostics ---
    conflicts = detect_overlap_conflicts(selections, candidates_by_id)

    missing_frames_by_track: dict[int, list[int]] = {
        track_id: sorted(result.missing_frames)
        for track_id, result in results.items()
    }

    diagnostics: dict = {
        "candidate_count": len(candidates),
        "track_count": len(results),
        "conflicts": [
            {
                "t": c.t,
                "track_ids": list(c.track_ids),
                "node_ids": list(c.node_ids),
                "overlap_pixels": c.overlap_pixels,
            }
            for c in conflicts
        ],
        "missing_frames_by_track": {
            str(k): v for k, v in missing_frames_by_track.items()
        },
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))

    return BoundarySelectionRunResult(
        output_path=output_path,
        diagnostics_path=diagnostics_path,
        candidate_count=len(candidates),
        missing_frames_by_track=missing_frames_by_track,
        overlap_conflicts=conflicts,
    )


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/config.py
--------------------------------------------------------------------------------

"""Pydantic configuration model for the Ultrack-based tracking stage."""
from __future__ import annotations

from multiprocessing import cpu_count

from pydantic import BaseModel


class TrackingConfig(BaseModel):
    # Node area filters (applied before NodeDB insert)
    min_area: int = 100
    max_area: int = 1_000_000

    # ID scheme — must match ultrack._generate_id
    max_segments_per_time: int = 1_000_000

    # Linking
    # Ultrack's multiprocessing_apply activates a Pool only when n_workers > 1,
    # and batch_index_range uses n_workers as a window stride (0 would div-zero).
    # Default to min(cpu_count(), 8) for automatic parallelism up to 8 threads.
    max_distance: float = 15.0
    max_neighbors: int = 5
    distance_weight: float = 0.0
    link_n_workers: int = min(cpu_count(), 8)
    linking_mode: str = "default"  # "default" or "iou"
    iou_weight: float = 1.0
    min_link_iou: float = 0.1

    # Solver / ILP
    appear_weight: float = -0.001
    disappear_weight: float = -0.001
    division_weight: float = -0.001
    link_function: str = "power"
    power: float = 4.0
    bias: float = 0.0
    solution_gap: float = 0.001
    time_limit: int = 36000
    window_size: int = 0  # 0 = solve all at once

    # Segmentation (ultrack.segment / ultrack.core.segmentation.processing.segment)
    seg_min_area: int = 300
    seg_max_area: int = 100_000
    seg_foreground_threshold: float = 0.5
    seg_min_frontier: float = 0.0
    seg_ws_hierarchy: str = "area"    # "area", "dynamics", or "volume"
    seg_n_workers: int = 1

    # Resolve-from-validated node prior
    quality_weight: float = 1.0
    quality_exponent: float = 8.0
    circularity_weight: float = 0.25
    seed_weight: float = 0.5
    seed_sigma_space: float = 25.0
    seed_tau_time: float = 2.0
    seed_max_dt: int = 5
    seed_sigma_area: float = 0.5


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/db_build.py
--------------------------------------------------------------------------------

"""Shared Ultrack database construction pipeline."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.ingest import _build_ultrack_config
from cellflow.tracking_ultrack.linking import run_linking
from cellflow.tracking_ultrack.seed_prior import (
    boost_validated_edges,
    write_seed_prior_node_probs,
)
from cellflow.tracking_ultrack.validation_nodes import inject_validated_nodes


@dataclass(frozen=True)
class UltrackDatabaseBuildReport:
    real_nodes: int = 0
    skipped_validated: int = 0
    fake_nodes: int = 0
    overlaps_added: int = 0
    scored_nodes: int = 0
    seed_nodes: int = 0
    boosted_edges: int = 0


def _notify(progress_cb: Callable[[str], None] | None, message: str) -> None:
    if progress_cb is not None:
        progress_cb(message)


def _load_ultrack_inputs(
    contour_maps_path: str | Path,
    foreground_masks_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    contours = np.asarray(tifffile.imread(str(contour_maps_path)), dtype=np.float32)
    foreground = np.asarray(tifffile.imread(str(foreground_masks_path)), dtype=np.float32)
    if contours.ndim == 4 and contours.shape[1] == 1:
        contours = contours[:, 0]
    if foreground.ndim == 4 and foreground.shape[1] == 1:
        foreground = foreground[:, 0]
    return contours, foreground


def _run_ultrack_segment(
    foreground: np.ndarray,
    contours: np.ndarray,
    ultrack_cfg,
    cfg: TrackingConfig,
) -> None:
    try:
        from ultrack.core.segmentation.processing import segment as ultrack_segment
    except ImportError as exc:
        raise ImportError(
            "ultrack must be installed (conda env cellflow) to build data.db"
        ) from exc

    ultrack_segment(
        foreground,
        contours,
        ultrack_cfg,
        max_segments_per_time=cfg.max_segments_per_time,
        overwrite=True,
    )


def build_ultrack_database(
    contour_maps_path: str | Path,
    foreground_masks_path: str | Path,
    nucleus_prob_zavg_path: str | Path,
    working_dir: str | Path,
    cfg: TrackingConfig,
    validated_tracks: dict[int, set[int]] | None = None,
    tracked_labels: np.ndarray | None = None,
    use_validated: bool = False,
    progress_cb: Callable[[str], None] | None = None,
) -> UltrackDatabaseBuildReport:
    """Build ``data.db`` from canonical Ultrack segmentation and linking.

    When ``use_validated`` is true, validated masks are injected as REAL nodes,
    conflicting canonical candidates are marked FAKE, node probabilities include
    seed affinity, and links incident to REAL nodes are boosted after linking.
    """
    if use_validated and (not validated_tracks or tracked_labels is None):
        raise ValueError(
            "Validated-aware DB generation requires validated tracks and tracked labels."
        )

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_cb, "Loading contour maps and foreground masks...")
    contours, foreground = _load_ultrack_inputs(contour_maps_path, foreground_masks_path)
    ultrack_cfg = _build_ultrack_config(cfg, working_dir)

    _notify(progress_cb, "Segmenting candidates (ultrack hierarchy)...")
    _run_ultrack_segment(foreground, contours, ultrack_cfg, cfg)

    real_nodes = skipped_validated = fake_nodes = overlaps_added = 0
    if use_validated:
        _notify(progress_cb, "Injecting validated nodes...")
        injection = inject_validated_nodes(
            working_dir=working_dir,
            validated_tracks=validated_tracks or {},
            tracked_labels=np.asarray(tracked_labels, dtype=np.uint32),
            cfg=cfg,
        )
        real_nodes = int(injection.inserted)
        skipped_validated = int(injection.skipped_missing)
        fake_nodes = int(injection.faked)
        overlaps_added = int(injection.overlaps_added)
        _notify(
            progress_cb,
            (
                f"Inserted {real_nodes} REAL node(s), marked {fake_nodes} FAKE "
                f"candidate(s), skipped {skipped_validated} validated cell-frame(s)."
            ),
        )
        if real_nodes == 0:
            raise ValueError("No validated masks could be injected; DB build aborted.")

    _notify(progress_cb, "Scoring node probabilities...")
    score_report = write_seed_prior_node_probs(working_dir, nucleus_prob_zavg_path, cfg)
    scored_nodes = int(getattr(score_report, "scored", 0))
    seed_nodes = int(getattr(score_report, "seeds", 0))
    _notify(progress_cb, f"Scored {scored_nodes} node(s) using {seed_nodes} seed node(s).")

    _notify(progress_cb, "Linking candidates...")
    for step, total, label in run_linking(working_dir, cfg):
        _notify(progress_cb, f"[link {step}/{total}] {label}")

    boosted_edges = 0
    if use_validated:
        _notify(progress_cb, "Boosting edges incident to validated nodes...")
        boost_report = boost_validated_edges(working_dir, cfg)
        boosted_edges = int(getattr(boost_report, "boosted", 0))
        _notify(progress_cb, f"Boosted {boosted_edges} link(s) incident to REAL nodes.")

    return UltrackDatabaseBuildReport(
        real_nodes=real_nodes,
        skipped_validated=skipped_validated,
        fake_nodes=fake_nodes,
        overlaps_added=overlaps_added,
        scored_nodes=scored_nodes,
        seed_nodes=seed_nodes,
        boosted_edges=boosted_edges,
    )


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/export.py
--------------------------------------------------------------------------------

"""Export selected NodeDB nodes to a (T, Z, Y, X) tracked labelmap."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.solve import database_has_annotations


def _build_export_config(cfg: TrackingConfig, working_dir: Path):
    from cellflow.tracking_ultrack.ingest import _build_ultrack_config

    return _build_ultrack_config(cfg, working_dir)


def _materialize_labels(labels) -> np.ndarray:
    if hasattr(labels, "compute"):
        labels = labels.compute()
    labels = np.asarray(labels, dtype=np.uint32)
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    return labels


def export_tracked_labels(
    working_dir: str | Path,
    cfg: TrackingConfig,
    output_path: str | Path,
    *,
    validated_tracks: dict[int, set[int]] | None = None,
    tracked_labels: np.ndarray | None = None,
    preserve_validated_ids: bool | None = None,
) -> np.ndarray:
    """Write ``tracked_labels.tif`` and return the (T, [Z,] Y, X) array."""
    wd = Path(working_dir)
    output_path = Path(output_path)
    annotated_db = database_has_annotations(wd)
    if preserve_validated_ids is None:
        preserve_validated_ids = annotated_db
    if preserve_validated_ids and (not validated_tracks or tracked_labels is None):
        raise ValueError(
            "Validated-aware export requires validated tracks and tracked labels."
        )

    labels = _export_tracked_labels_raw(wd, cfg, output_path)
    if preserve_validated_ids:
        from cellflow.tracking_ultrack.reseed import merge_validated_into_export

        labels, _id_map = merge_validated_into_export(
            labels,
            validated_tracks or {},
            np.asarray(tracked_labels, dtype=np.uint32),
        )
        tifffile.imwrite(str(output_path), labels, compression="zlib")
    return labels


def _export_tracked_labels_raw(
    working_dir: str | Path,
    cfg: TrackingConfig,
    output_path: str | Path,
) -> np.ndarray:
    wd = Path(working_dir)
    output_path = Path(output_path)
    ultrack_cfg = _build_export_config(cfg, wd)

    # Prefer public track export: tracks_to_zarr rasterizes each segment with
    # its track_id, while label-export helpers may expose per-frame segment IDs.
    try:
        from ultrack.core.export import to_tracks_layer, tracks_to_zarr  # type: ignore[import]

        tracks_df, _graph = to_tracks_layer(ultrack_cfg)
        labels = _materialize_labels(
            tracks_to_zarr(ultrack_cfg, tracks_df, overwrite=True)
        )
        tifffile.imwrite(str(output_path), labels, compression="zlib")
        return labels
    except Exception:
        pass

    # Try the modern to_labels API next (returns dask or numpy)
    try:
        from ultrack.core.export.labels import to_labels  # type: ignore[import]

        labels = _materialize_labels(to_labels(ultrack_cfg))
        tifffile.imwrite(str(output_path), labels, compression="zlib")
        return labels
    except Exception:
        pass

    # Fallback: CTC export → stack TIFFs
    tmpdir = Path(tempfile.mkdtemp(prefix="ultrack_ctc_"))
    try:
        from ultrack.core.export.ctc import to_ctc  # type: ignore[import]

        to_ctc(tmpdir, ultrack_cfg, overwrite=True)
        mask_files = sorted(tmpdir.rglob("mask*.tif"))
        if not mask_files:
            mask_files = sorted(tmpdir.rglob("man_track*.tif"))
        if not mask_files:
            mask_files = sorted(tmpdir.rglob("*.tif"))
        if not mask_files:
            raise RuntimeError("CTC export produced no mask files.")
        frames = [tifffile.imread(str(f)) for f in mask_files]
        labels = _materialize_labels(np.stack(frames, axis=0))
        tifffile.imwrite(str(output_path), labels, compression="zlib")
        return labels
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/extend.py
--------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
from skimage.measure import regionprops

from cellflow.database.hypotheses import list_hypotheses, read_hypothesis_labels

_D_MAX_DEFAULT = 40.0
_GREEDY_CANDIDATE_LIMIT = 5


@dataclass(frozen=True)
class ExtendAssignment:
    cell_id: int
    candidate_label: int
    candidate_partition: int
    mask_2d: np.ndarray       # bool, full-frame (Y, X)
    bbox: tuple[int, int, int, int]
    centroid_distance: float
    area_ratio: float
    centroid_corrected_iou: float
    existing_overlap: float
    score: float


@dataclass
class ExtendResult:
    target_frame: int
    candidate_label: int
    candidate_partition: int
    mask_2d: np.ndarray       # bool, full-frame (Y, X)
    bbox: tuple[int, int, int, int]  # (y0, x0, y1, x1)
    centroid_distance: float
    area_ratio: float         # ∈ [0, 1]; 1.0 = identical area
    centroid_corrected_iou: float
    existing_overlap: float   # candidate ∩ (other cells at target) / candidate_area
    assignments: tuple[ExtendAssignment, ...] = ()


def _centroid_corrected_iou(source_mask: np.ndarray, candidate_mask: np.ndarray) -> float:
    """IoU after translating candidate pixels to the source centroid."""
    src_y, src_x = np.nonzero(source_mask)
    cand_y, cand_x = np.nonzero(candidate_mask)
    if len(src_y) == 0 or len(cand_y) == 0:
        return 0.0

    dy = int(round(float(src_y.mean()) - float(cand_y.mean())))
    dx = int(round(float(src_x.mean()) - float(cand_x.mean())))
    src_pixels = set(zip(src_y.tolist(), src_x.tolist()))
    cand_pixels = set(zip((cand_y + dy).tolist(), (cand_x + dx).tolist()))
    intersection = len(src_pixels & cand_pixels)
    union = len(src_pixels) + len(cand_pixels) - intersection
    return intersection / union if union > 0 else 0.0


def _extend_score(
    *,
    area_ratio: float,
    centroid_corrected_iou: float,
    centroid_distance: float,
    d_max: float,
    existing_overlap: float,
    area_weight: float,
    iou_weight: float,
    distance_weight: float,
    overlap_penalty: float,
) -> tuple[float, float]:
    distance_score = 1.0 if d_max <= 0 else max(0.0, 1.0 - centroid_distance / d_max)
    weighted_score = (
        area_weight * area_ratio
        + iou_weight * centroid_corrected_iou
        + distance_weight * distance_score
        - overlap_penalty * existing_overlap
    )
    return (weighted_score, -centroid_distance)


@dataclass(frozen=True)
class _DbCandidate:
    candidate_label: int
    candidate_partition: int
    mask_2d: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]


def _mask_centroid_area(mask: np.ndarray) -> tuple[float, float, float] | None:
    props = regionprops(mask.astype(np.uint8))
    if not props:
        return None
    cy, cx = props[0].centroid
    return float(cy), float(cx), float(props[0].area)


def _assignment_for_candidate(
    *,
    cell_id: int,
    reference_mask: np.ndarray,
    target_frame_labels: np.ndarray,
    candidate: _DbCandidate,
    d_max: float,
    area_weight: float,
    iou_weight: float,
    distance_weight: float,
    overlap_penalty: float,
) -> ExtendAssignment | None:
    stats = _mask_centroid_area(reference_mask)
    if stats is None:
        return None
    src_cy, src_cx, src_area = stats
    cand_cy, cand_cx = candidate.centroid
    dist = float(np.hypot(cand_cy - src_cy, cand_cx - src_cx))
    if dist > d_max:
        return None

    cand_area = float(candidate.mask_2d.sum())
    if cand_area == 0:
        return None
    other_cells = (target_frame_labels != 0) & (target_frame_labels != cell_id)
    existing_overlap = float((candidate.mask_2d & other_cells).sum()) / cand_area
    area_ratio = min(src_area, cand_area) / max(src_area, cand_area)
    shape_iou = _centroid_corrected_iou(reference_mask, candidate.mask_2d)
    score = _extend_score(
        area_ratio=area_ratio,
        centroid_corrected_iou=shape_iou,
        centroid_distance=dist,
        d_max=d_max,
        existing_overlap=existing_overlap,
        area_weight=area_weight,
        iou_weight=iou_weight,
        distance_weight=distance_weight,
        overlap_penalty=overlap_penalty,
    )[0]
    return ExtendAssignment(
        cell_id=cell_id,
        candidate_label=candidate.candidate_label,
        candidate_partition=candidate.candidate_partition,
        mask_2d=candidate.mask_2d,
        bbox=candidate.bbox,
        centroid_distance=dist,
        area_ratio=area_ratio,
        centroid_corrected_iou=shape_iou,
        existing_overlap=existing_overlap,
        score=score,
    )


def _result_from_assignment(
    assignment: ExtendAssignment,
    target_frame: int,
    assignments: tuple[ExtendAssignment, ...] | None = None,
) -> ExtendResult:
    return ExtendResult(
        target_frame=target_frame,
        candidate_label=assignment.candidate_label,
        candidate_partition=assignment.candidate_partition,
        mask_2d=assignment.mask_2d,
        bbox=assignment.bbox,
        centroid_distance=assignment.centroid_distance,
        area_ratio=assignment.area_ratio,
        centroid_corrected_iou=assignment.centroid_corrected_iou,
        existing_overlap=assignment.existing_overlap,
        assignments=assignments or (assignment,),
    )


def _masks_are_disjoint(assignments: tuple[ExtendAssignment, ...]) -> bool:
    occupied: np.ndarray | None = None
    for assignment in assignments:
        if occupied is None:
            occupied = assignment.mask_2d.copy()
            continue
        if np.any(occupied & assignment.mask_2d):
            return False
        occupied |= assignment.mask_2d
    return True


def _plan_overwrites_only_assigned_cells(
    assignments: tuple[ExtendAssignment, ...],
    target_frame_labels: np.ndarray,
    protected_ids: set[int],
) -> bool:
    assigned_ids = {int(assignment.cell_id) for assignment in assignments}
    overwritten = np.zeros_like(target_frame_labels, dtype=bool)
    for assignment in assignments:
        overwritten |= assignment.mask_2d
    overwritten_ids = {
        int(v)
        for v in np.unique(target_frame_labels[overwritten])
        if int(v) != 0
    }
    return (overwritten_ids & protected_ids).issubset(assigned_ids)


def _plan_preserves_locked_target_cells(
    assignments: tuple[ExtendAssignment, ...],
    target_frame_labels: np.ndarray,
    locked_target_ids: set[int],
) -> bool:
    if not locked_target_ids:
        return True
    overwritten = np.zeros_like(target_frame_labels, dtype=bool)
    for assignment in assignments:
        overwritten |= assignment.mask_2d
    overwritten_ids = {
        int(v)
        for v in np.unique(target_frame_labels[overwritten])
        if int(v) != 0
    }
    return not (overwritten_ids & locked_target_ids)


def _top_assignments_for_cell(
    *,
    cell_id: int,
    reference_mask: np.ndarray,
    target_frame_labels: np.ndarray,
    candidates: list[_DbCandidate],
    d_max: float,
    area_weight: float,
    iou_weight: float,
    distance_weight: float,
    overlap_penalty: float,
    limit: int = _GREEDY_CANDIDATE_LIMIT,
) -> list[ExtendAssignment]:
    assignments = []
    for candidate in candidates:
        assignment = _assignment_for_candidate(
            cell_id=cell_id,
            reference_mask=reference_mask,
            target_frame_labels=target_frame_labels,
            candidate=candidate,
            d_max=d_max,
            area_weight=area_weight,
            iou_weight=iou_weight,
            distance_weight=distance_weight,
            overlap_penalty=overlap_penalty,
        )
        if assignment is not None:
            assignments.append(assignment)
    assignments.sort(key=lambda item: (item.score, -item.centroid_distance), reverse=True)
    return assignments[:limit]


def _best_greedy_overwrite_plan(
    *,
    source_id: int,
    source_mask: np.ndarray,
    source_frame_labels: np.ndarray,
    target_frame_labels: np.ndarray,
    candidates: list[_DbCandidate],
    d_max: float,
    area_weight: float,
    iou_weight: float,
    distance_weight: float,
    overlap_penalty: float,
    locked_target_ids: set[int] | None = None,
) -> tuple[ExtendAssignment, ...] | None:
    locked_target_ids = set(locked_target_ids or set())
    if source_id in locked_target_ids:
        return None

    source_assignments = _top_assignments_for_cell(
        cell_id=source_id,
        reference_mask=source_mask,
        target_frame_labels=target_frame_labels,
        candidates=candidates,
        d_max=d_max,
        area_weight=area_weight,
        iou_weight=iou_weight,
        distance_weight=distance_weight,
        overlap_penalty=overlap_penalty,
    )
    if not source_assignments:
        return None

    best_plan: tuple[ExtendAssignment, ...] | None = None
    best_score: tuple[float, float] | None = None
    protected_ids = {int(v) for v in np.unique(source_frame_labels) if int(v) != 0}
    protected_ids.discard(source_id)

    for source_assignment in source_assignments:
        if not _plan_preserves_locked_target_cells(
            (source_assignment,),
            target_frame_labels,
            locked_target_ids,
        ):
            continue
        conflict_labels = target_frame_labels[
            source_assignment.mask_2d
            & (target_frame_labels != 0)
            & (target_frame_labels != source_id)
        ]
        conflicted_ids = sorted(
            int(v)
            for v in np.unique(conflict_labels)
            if int(v) != 0 and int(v) in protected_ids
        )
        if not conflicted_ids:
            plan = (source_assignment,)
            score = (source_assignment.score, -source_assignment.centroid_distance)
            if best_score is None or score > best_score:
                best_plan = plan
                best_score = score
            continue

        choices: list[list[ExtendAssignment]] = []
        for cell_id in conflicted_ids:
            reference_mask = source_frame_labels == cell_id
            cell_choices = _top_assignments_for_cell(
                cell_id=cell_id,
                reference_mask=reference_mask,
                target_frame_labels=target_frame_labels,
                candidates=candidates,
                d_max=d_max,
                area_weight=area_weight,
                iou_weight=iou_weight,
                distance_weight=distance_weight,
                overlap_penalty=overlap_penalty,
            )
            if not cell_choices:
                choices = []
                break
            choices.append(cell_choices)
        if not choices:
            continue

        for combo in product(*choices):
            plan = (source_assignment, *combo)
            if not _masks_are_disjoint(plan):
                continue
            if not _plan_preserves_locked_target_cells(
                plan,
                target_frame_labels,
                locked_target_ids,
            ):
                continue
            if not _plan_overwrites_only_assigned_cells(
                plan,
                target_frame_labels,
                protected_ids,
            ):
                continue
            total_score = sum(item.score for item in plan)
            total_distance = sum(item.centroid_distance for item in plan)
            score = (total_score, -total_distance)
            if best_score is None or score > best_score:
                best_plan = plan
                best_score = score

    return best_plan


def extend_track(
    *,
    source_id: int,
    source_frame: int,
    direction: Literal["forward", "backward"],
    tracked_labels: np.ndarray,   # (T, Y, X) uint32
    hypotheses_path: Path,
    d_max: float = _D_MAX_DEFAULT,
    area_weight: float = 1.0,
    iou_weight: float = 1.0,
    distance_weight: float = 0.25,
    overlap_penalty: float = 1.0,
) -> ExtendResult | None:
    T = tracked_labels.shape[0]
    target_frame = source_frame + (1 if direction == "forward" else -1)

    if target_frame < 0 or target_frame >= T:
        return None

    source_mask = tracked_labels[source_frame] == source_id
    if not source_mask.any():
        return None

    props = regionprops(source_mask.astype(np.uint8))
    src_cy, src_cx = props[0].centroid
    src_area = float(props[0].area)

    target_frame_labels = tracked_labels[target_frame]
    other_cells = (target_frame_labels != 0) & (target_frame_labels != source_id)

    n_p, _ = list_hypotheses(hypotheses_path)
    if n_p == 0:
        return None

    best: ExtendResult | None = None
    best_score: tuple[float, float] | None = None

    for p in range(n_p):
        labels_raw = read_hypothesis_labels(hypotheses_path, target_frame, p)
        labels_2d: np.ndarray
        if labels_raw.ndim == 3 and labels_raw.shape[0] == 1:
            labels_2d = labels_raw[0]
        elif labels_raw.ndim == 2:
            labels_2d = labels_raw
        else:
            continue

        for rp in regionprops(labels_2d.astype(np.int32)):
            cy, cx = rp.centroid
            dist = float(np.hypot(cy - src_cy, cx - src_cx))
            if dist > d_max:
                continue

            cand_area = float(rp.area)
            cand_mask = labels_2d == rp.label
            existing_overlap = float((cand_mask & other_cells).sum()) / cand_area
            area_ratio = min(src_area, cand_area) / max(src_area, cand_area)
            shape_iou = _centroid_corrected_iou(source_mask, cand_mask)
            score = _extend_score(
                area_ratio=area_ratio,
                centroid_corrected_iou=shape_iou,
                centroid_distance=dist,
                d_max=d_max,
                existing_overlap=existing_overlap,
                area_weight=area_weight,
                iou_weight=iou_weight,
                distance_weight=distance_weight,
                overlap_penalty=overlap_penalty,
            )

            if best_score is None or score > best_score:
                y0, x0, y1, x1 = rp.bbox
                assignment = ExtendAssignment(
                    cell_id=source_id,
                    candidate_label=int(rp.label),
                    candidate_partition=p,
                    mask_2d=cand_mask,
                    bbox=(y0, x0, y1, x1),
                    centroid_distance=dist,
                    area_ratio=area_ratio,
                    centroid_corrected_iou=shape_iou,
                    existing_overlap=existing_overlap,
                    score=score[0],
                )
                best = _result_from_assignment(assignment, target_frame)
                best_score = score

    return best


def extend_track_from_db(
    *,
    source_id: int,
    source_frame: int,
    direction: Literal["forward", "backward"],
    tracked_labels: np.ndarray,   # (T, Y, X) uint32
    db_path: Path,
    d_max: float = _D_MAX_DEFAULT,
    area_weight: float = 1.0,
    iou_weight: float = 1.0,
    distance_weight: float = 0.25,
    overlap_penalty: float = 1.0,
    greedy_overwrite: bool = False,
    validated_tracks: dict[int, set[int]] | None = None,
) -> ExtendResult | None:
    """Extend a track using candidates from ultrack_workdir/data.db.

    Returns None if the DB is missing, target frame is out of range, or no
    candidate within d_max is found.  Widget caller should show a local status
    message on None.
    """
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB
    from cellflow.tracking_ultrack.validation_nodes import _node_bbox_and_mask

    if not db_path.exists():
        return None

    T = tracked_labels.shape[0]
    target_frame = source_frame + (1 if direction == "forward" else -1)
    if target_frame < 0 or target_frame >= T:
        return None

    source_mask = tracked_labels[source_frame] == source_id
    if not source_mask.any():
        return None

    source_frame_labels = tracked_labels[source_frame]
    target_frame_labels = tracked_labels[target_frame]
    locked_target_ids = {
        int(cell_id)
        for cell_id, frames in (validated_tracks or {}).items()
        if target_frame in frames
    }

    engine = sqla.create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    candidates: list[_DbCandidate] = []

    with Session(engine) as session:
        rows = session.query(NodeDB).filter(NodeDB.t == target_frame).all()
        for node in rows:
            try:
                (y0, x0, y1, x1), mask_2d = _node_bbox_and_mask(int(node.id), node.pickle)
            except Exception:
                continue
            if mask_2d.shape != (y1 - y0, x1 - x0):
                continue
            full_mask = np.zeros(tracked_labels.shape[1:], dtype=bool)
            full_mask[y0:y1, x0:x1] = mask_2d
            if not full_mask.any():
                continue
            candidates.append(
                _DbCandidate(
                    candidate_label=int(node.id),
                    candidate_partition=0,
                    mask_2d=full_mask,
                    bbox=(y0, x0, y1, x1),
                    centroid=(float(node.y), float(node.x)),
                )
            )
    engine.dispose()

    if not candidates:
        return None

    if greedy_overwrite:
        plan = _best_greedy_overwrite_plan(
            source_id=source_id,
            source_mask=source_mask,
            source_frame_labels=source_frame_labels,
            target_frame_labels=target_frame_labels,
            candidates=candidates,
            d_max=d_max,
            area_weight=area_weight,
            iou_weight=iou_weight,
            distance_weight=distance_weight,
            overlap_penalty=overlap_penalty,
            locked_target_ids=locked_target_ids,
        )
        if not plan:
            return None
        return _result_from_assignment(plan[0], target_frame, plan)

    assignments = _top_assignments_for_cell(
        cell_id=source_id,
        reference_mask=source_mask,
        target_frame_labels=target_frame_labels,
        candidates=candidates,
        d_max=d_max,
        area_weight=area_weight,
        iou_weight=iou_weight,
        distance_weight=distance_weight,
        overlap_penalty=overlap_penalty,
        limit=1,
    )
    if not assignments:
        return None
    return _result_from_assignment(assignments[0], target_frame)


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/ingest.py
--------------------------------------------------------------------------------

"""Ingest v2 hypothesis HDF5 directly into Ultrack's NodeDB + OverlapDB.

Bypasses ultrack.segment() — each (t, p, label_id) becomes one NodeDB row;
cross-p mask overlaps at the same t become OverlapDB pairs.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import threading
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import sqlalchemy as sqla
from skimage.measure import regionprops
from sqlalchemy.orm import Session

from cellflow.database.hypotheses import read_hypothesis_labels
from cellflow.tracking_ultrack.config import TrackingConfig

LOG = logging.getLogger(__name__)


def _canonical_hash(labelmap_2d: np.ndarray) -> bytes:
    """Return an 8-byte hash that identifies the partition structure of a 2D labelmap.

    Two labelmaps are considered duplicates if they describe the same set of cell
    regions regardless of how the labels are numbered.  We canonicalize by
    relabelling in raster-scan order of first pixel occurrence (so the first
    non-zero label encountered becomes 1, the second becomes 2, etc.) then hashing
    the resulting byte sequence with BLAKE2b.

    All-zero maps hash identically to each other and correctly to a single entry.
    No division-by-zero or empty-array edge cases: the max-value check handles both.
    """
    flat = labelmap_2d.ravel()
    max_val = int(flat.max()) if flat.size > 0 else 0
    if max_val == 0:
        # All background — hash the zero array directly (shape is already fixed
        # per frame, so all-zero maps of the same shape produce the same hash).
        return hashlib.blake2b(flat.tobytes(), digest_size=8).digest()

    # np.unique with return_index gives the first occurrence (in raster order)
    # for every label value.
    unique_labels, first_indices = np.unique(flat, return_index=True)
    # Drop background (0) — it keeps its value (0) in the canonical map.
    nonzero_mask = unique_labels > 0
    unique_labels = unique_labels[nonzero_mask]
    first_indices = first_indices[nonzero_mask]

    # Sort non-zero labels by their first-occurrence position → canonical 1, 2, 3 ...
    sort_order = np.argsort(first_indices)
    sorted_labels = unique_labels[sort_order]

    # Build a lookup table: old_label_value → canonical_label_value.
    lookup = np.zeros(max_val + 1, dtype=np.uint32)
    lookup[sorted_labels] = np.arange(1, len(sorted_labels) + 1, dtype=np.uint32)

    canonical = lookup[flat]
    return hashlib.blake2b(canonical.tobytes(), digest_size=8).digest()


def _cell_mask_hash(bbox: np.ndarray, mask: np.ndarray) -> bytes:
    """Return a hash for one cell mask in frame coordinates."""
    bbox_arr = np.asarray(bbox, dtype=np.int32)
    mask_arr = np.ascontiguousarray(mask, dtype=bool)
    h = hashlib.blake2b(digest_size=16)
    h.update(bbox_arr.tobytes())
    h.update(mask_arr.tobytes())
    return h.digest()


def _generate_id(index: int, time: int, max_segments: int) -> int:
    return index + (time + 1) * max_segments


def _build_ultrack_config(cfg: TrackingConfig, working_dir: Path):
    from ultrack.config import MainConfig
    from ultrack.config.segmentationconfig import NAME_TO_WS_HIER

    ultrack_cfg = MainConfig(
        data={"working_dir": str(working_dir)},
        linking={
            "max_distance": cfg.max_distance,
            "max_neighbors": cfg.max_neighbors,
            "distance_weight": cfg.distance_weight,
            "n_workers": cfg.link_n_workers,
        },
        tracking={
            "solver_name": _select_solver(),
            "appear_weight": cfg.appear_weight,
            "disappear_weight": cfg.disappear_weight,
            "division_weight": cfg.division_weight,
            "link_function": cfg.link_function,
            "power": cfg.power,
            "bias": cfg.bias,
            "solution_gap": cfg.solution_gap,
            "time_limit": cfg.time_limit,
            "window_size": cfg.window_size if cfg.window_size > 0 else None,
        },
    )
    sc = ultrack_cfg.segmentation_config
    sc.min_area = cfg.seg_min_area
    sc.max_area = cfg.seg_max_area
    sc.threshold = cfg.seg_foreground_threshold
    sc.min_frontier = cfg.seg_min_frontier
    sc.ws_hierarchy = NAME_TO_WS_HIER[cfg.seg_ws_hierarchy]
    sc.n_workers = cfg.seg_n_workers
    return ultrack_cfg


def _select_solver() -> str:
    try:
        import gurobipy  # noqa: F401
        return "GUROBI"
    except ImportError:
        return "CBC"


@dataclass
class _CellRecord:
    node_id: int
    label_val: int          # original label value in the labelmap
    p: int
    bbox: np.ndarray        # (min_y, min_x, max_y, max_x) exclusive-max
    mask: np.ndarray        # bool crop (h, w)
    area: int
    y: float
    x: float


def _extract_cell_records_2d(
    labelmap_2d: np.ndarray,
    t: int,
    p: int,
    index_start: int,
    max_segments: int,
    min_area: int | None,
    max_area: int | None,
    forbidden_mask: np.ndarray | None = None,
    seen_cell_hashes: set[bytes] | None = None,
) -> tuple[list[_CellRecord], int]:
    """Extract CellRecords from one 2D labelmap. Returns (records, next_index).

    If ``forbidden_mask`` is given (Y×X bool), any region whose pixels intersect
    a True pixel in the mask is silently skipped — used to keep validated cells
    out of the hypothesis DB at ingest time.
    """
    records: list[_CellRecord] = []
    idx = index_start

    for prop in regionprops(labelmap_2d):
        area = int(prop.area)
        if min_area is not None and area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        min_r, min_c, max_r, max_c = prop.bbox
        bbox = np.array([min_r, min_c, max_r, max_c], dtype=np.int32)
        mask = (labelmap_2d[min_r:max_r, min_c:max_c] == prop.label).astype(bool)

        if forbidden_mask is not None:
            forbidden_crop = forbidden_mask[min_r:max_r, min_c:max_c]
            if np.any(mask & forbidden_crop):
                continue

        if seen_cell_hashes is not None:
            cell_hash = _cell_mask_hash(bbox, mask)
            if cell_hash in seen_cell_hashes:
                continue
            seen_cell_hashes.add(cell_hash)

        node_id = _generate_id(idx, t, max_segments)
        cy, cx = prop.centroid

        records.append(_CellRecord(
            node_id=node_id,
            label_val=int(prop.label),
            p=p,
            bbox=bbox,
            mask=mask,
            area=area,
            y=float(cy),
            x=float(cx),
        ))
        idx += 1

    return records, idx


def _build_nid_labelmap(
    labelmap_2d: np.ndarray,
    records: list[_CellRecord],
) -> np.ndarray:
    """Return a (Y, X) int64 array with node_ids where cells were kept, 0 elsewhere.

    Uses a vectorized lookup table instead of per-cell masking.
    """
    max_label = int(labelmap_2d.max())
    if max_label == 0 or not records:
        return np.zeros(labelmap_2d.shape, dtype=np.int64)
    lookup = np.zeros(max_label + 1, dtype=np.int64)
    for rec in records:
        if rec.label_val <= max_label:
            lookup[rec.label_val] = rec.node_id
    return lookup[labelmap_2d]


def _compute_overlaps_vectorized(
    nid_lms: list[np.ndarray],
) -> list[tuple[int, int]]:
    """Find cross-partition overlapping node pairs using vectorized labelmap ANDs.

    Encodes each (hi, lo) node-id pair as a single int64 (hi*MAX_ID + lo) so
    that deduplication uses a fast 1D sort rather than 2D column_stack+unique.
    All encoded pairs are concatenated and deduplicated in a single numpy call
    at the end, avoiding per-pair Python overhead.
    """
    n = len(nid_lms)
    if n == 0:
        return []

    # Pre-flatten to 1D and pre-compute nonzero masks once
    flat = [lm.ravel() for lm in nid_lms]
    nz = [f > 0 for f in flat]

    # MAX_ID must exceed every node id present in this frame so that
    # encoding hi*MAX_ID+lo is injective.  Using a fixed constant like 1e7 is
    # wrong at t>=9 (ids start at 10_000_001).  Derive it from the actual data.
    max_id_in_frame = max((int(lm.max()) for lm in nid_lms if lm.size), default=0)
    MAX_ID = max_id_in_frame + 1
    encoded_chunks: list[np.ndarray] = []

    for i in range(n):
        if not nz[i].any():
            continue
        for j in range(i + 1, n):
            combined = nz[i] & nz[j]
            if not combined.any():
                continue
            ni = flat[i][combined]
            nj = flat[j][combined]
            lo = np.minimum(ni, nj)
            hi = np.maximum(ni, nj)
            # Per-pair unique reduces the chunk size before the final dedup
            encoded_chunks.append(np.unique(hi * MAX_ID + lo))

    if not encoded_chunks:
        return []

    all_encoded = np.unique(np.concatenate(encoded_chunks))
    return [(int(e // MAX_ID), int(e % MAX_ID)) for e in all_encoded]


def _list_timepoints(hypotheses_h5: Path) -> list[int]:
    import h5py
    with h5py.File(hypotheses_h5, "r") as f:
        root = f["hypotheses"]
        return sorted(int(k[1:]) for k in root.keys() if k.startswith("t"))


def _list_partitions(hypotheses_h5: Path, t: int) -> list[int]:
    import h5py
    with h5py.File(hypotheses_h5, "r") as f:
        grp = f[f"hypotheses/t{t:03d}"]
        return sorted(int(k[1:]) for k in grp.keys() if k.startswith("p"))


def _resolve_ingest_worker_count(n_total: int, n_workers: int | None) -> int:
    """Return a worker count that avoids fork-based pools from background threads."""
    if n_workers is not None:
        return max(1, int(n_workers))
    if threading.current_thread() is not threading.main_thread():
        return 1
    return max(1, min(cpu_count(), n_total, 8))


def _make_node_pickle(t: int, mask_2d: np.ndarray, bbox: np.ndarray, node_id: int) -> bytes:
    from ultrack.core.segmentation.node import Node
    # Lift to 3D (1, h, w) so paint_buffer works against a (Z, Y, X) export buffer
    mask_3d = mask_2d[np.newaxis]
    min_y, min_x, max_y, max_x = bbox
    bbox_3d = np.array([0, int(min_y), int(min_x), 1, int(max_y), int(max_x)], dtype=np.int64)
    node = Node.from_mask(time=t, mask=mask_3d, bbox=bbox_3d, node_id=node_id)
    return pickle.dumps(node)


# ---------------------------------------------------------------------------
# Worker function for parallel ingest (must be module-level for pickling)
# ---------------------------------------------------------------------------

def _ingest_frame_worker(args: tuple) -> None:
    """Process one timepoint and write results to a per-frame temp SQLite DB.

    Called in a subprocess via multiprocessing.Pool.map.  Each worker opens its
    own HDF5 handle (read-only) and writes a standalone SQLite DB that the main
    process later bulk-merges into data.db via ATTACH DATABASE.

    Node IDs are timepoint-scoped: _generate_id(index, t, max_segments) =
    index + (t+1)*max_segments.  Since every frame uses a distinct t, IDs
    cannot collide across temp DBs.
    """
    (
        t,
        hypotheses_h5_path,
        max_segments,
        eff_min_area,
        eff_max_area,
        max_partitions,
        tmp_db_path_str,
        forbidden_mask,
    ) = args

    import sqlalchemy as _sqla
    from sqlalchemy.orm import Session as _Session
    from ultrack.core.database import Base as _Base, NodeDB as _NodeDB, OverlapDB as _OverlapDB
    import pandas as _pd

    hypotheses_h5 = Path(hypotheses_h5_path)
    tmp_db_path = Path(tmp_db_path_str)

    # Create isolated temp DB for this frame
    tmp_engine = _sqla.create_engine(f"sqlite:///{tmp_db_path}")
    _Base.metadata.create_all(tmp_engine)

    all_partitions = _list_partitions(hypotheses_h5, t)
    n_raw = len(all_partitions)

    all_records: list[_CellRecord] = []
    nid_lms: list[np.ndarray] = []
    index = 1

    # Deduplication — same logic as serial path
    seen_hashes: set[bytes] = set()
    seen_cell_hashes: set[bytes] = set()
    n_kept = 0
    n_dropped = 0

    for p in all_partitions:
        labels = read_hypothesis_labels(hypotheses_h5, t, p)
        if labels.ndim == 3 and labels.shape[0] == 1:
            labels_2d = labels[0]
        elif labels.ndim == 2:
            labels_2d = labels
        else:
            raise NotImplementedError(f"3D ingestion not yet supported (shape {labels.shape})")

        h = _canonical_hash(labels_2d)
        if h in seen_hashes:
            n_dropped += 1
            continue
        seen_hashes.add(h)

        if max_partitions is not None and n_kept >= max_partitions:
            n_dropped += 1
            continue

        n_kept += 1
        recs, index = _extract_cell_records_2d(
            labels_2d, t, p, index, max_segments, eff_min_area, eff_max_area,
            forbidden_mask=forbidden_mask,
            seen_cell_hashes=seen_cell_hashes,
        )
        nid_lm = _build_nid_labelmap(labels_2d, recs)
        all_records.extend(recs)
        nid_lms.append(nid_lm)

    overlap_pairs = _compute_overlaps_vectorized(nid_lms)

    # Write Nodes
    with _Session(tmp_engine) as session:
        session.bulk_save_objects([
            _NodeDB(
                id=rec.node_id,
                t=t,
                t_node_id=rec.node_id - (t + 1) * max_segments,
                t_hier_id=rec.p + 1,
                z=0,
                y=rec.y,
                x=rec.x,
                area=rec.area,
                pickle=_make_node_pickle(t, rec.mask, rec.bbox, rec.node_id),
            )
            for rec in all_records
        ])
        session.commit()

    # Write Overlaps
    if overlap_pairs:
        df_overlaps = _pd.DataFrame(overlap_pairs, columns=["node_id", "ancestor_id"])
        with tmp_engine.begin() as conn:
            df_overlaps.to_sql(
                _OverlapDB.__tablename__, conn,
                if_exists="append", index=False,
                chunksize=50_000, method="multi",
            )

    tmp_engine.dispose()
    return (t, n_raw, n_kept, n_dropped, len(all_records), len(overlap_pairs))


def ingest_hypotheses_to_db(
    hypotheses_h5: Path,
    working_dir: Path,
    cfg: TrackingConfig,
    *,
    overwrite: bool = True,
    min_area: int | None = None,
    max_area: int | None = None,
    max_partitions: int | None = None,
    n_frames: int | None = None,
    n_workers: int | None = None,
    forbidden_masks: dict[int, np.ndarray] | None = None,
) -> None:
    """Write v2 hypothesis HDF5 into Ultrack's NodeDB + OverlapDB.

    Parameters
    ----------
    hypotheses_h5:
        Path to v2 ``hypotheses.h5`` file.
    working_dir:
        Directory for Ultrack's ``data.db`` SQLite file.
    cfg:
        Tracking configuration (area filters, ILP parameters).
    overwrite:
        Clear existing DB before ingestion.
    min_area, max_area:
        Optional pixel-area filters applied before insert.
    max_partitions:
        Cap the number of partitions used per frame. Useful for large sweeps.
        None = use all partitions.
    n_frames:
        Limit ingestion to the first ``n_frames`` timepoints from the HDF5.
        None (default) = all timepoints.
    n_workers:
        Number of worker processes for parallel frame ingest.
        None (default) = min(cpu_count(), n_frames, 8) on the main thread,
        or serial when called from a background thread.
        1 = serial (no subprocess overhead).
    forbidden_masks:
        Optional ``{t: bool_array (Y, X)}`` map.  Any hypothesis cell whose
        pixels intersect ``forbidden_masks[t]`` at frame ``t`` is silently
        skipped — used by the validate-and-resolve flow to keep validated
        cells out of the hypothesis DB without a separate prune pass.
    """
    import multiprocessing as _mp
    import time as _time

    from ultrack.core.database import Base, NodeDB, OverlapDB, clear_all_data  # noqa: F401

    hypotheses_h5 = Path(hypotheses_h5)
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    ultrack_cfg = _build_ultrack_config(cfg, working_dir)
    db_path = ultrack_cfg.data_config.database_path

    engine = sqla.create_engine(db_path)
    if overwrite:
        Base.metadata.create_all(engine)  # ensure tables exist so drop_all doesn't fail on fresh DB
        clear_all_data(db_path)
    Base.metadata.create_all(engine)

    eff_min_area = min_area if min_area is not None else cfg.min_area
    eff_max_area = max_area if max_area is not None else cfg.max_area
    max_segments = cfg.max_segments_per_time

    all_timepoints = _list_timepoints(hypotheses_h5)
    if n_frames is not None:
        timepoints = all_timepoints[:n_frames]
    else:
        timepoints = all_timepoints
    n_total = len(timepoints)
    LOG.info(f"Ingesting {n_total} timepoints from {hypotheses_h5}")

    # Write shape metadata required by the solver: (T, Z, Y, X)
    # Done in main process (cheap + synchronous) before workers start.
    first_t = timepoints[0]
    first_p = _list_partitions(hypotheses_h5, first_t)[0]
    sample = read_hypothesis_labels(hypotheses_h5, first_t, first_p)
    frame_shape = sample.shape  # (Z, Y, X)
    full_shape = (n_total,) + frame_shape
    ultrack_cfg.data_config.metadata_add({"shape": list(full_shape), "properties": []})

    n_workers = _resolve_ingest_worker_count(n_total, n_workers)

    # Temp DB directory — clean up any leftover DBs from a previous crashed run
    tmp_dir = working_dir / "_tmp_frame_dbs"
    tmp_dir.mkdir(exist_ok=True)
    for t in timepoints:
        stale = tmp_dir / f"frame_{t:04d}.db"
        stale.unlink(missing_ok=True)

    worker_args = [
        (
            t,
            str(hypotheses_h5),
            max_segments,
            eff_min_area,
            eff_max_area,
            max_partitions,
            str(tmp_dir / f"frame_{t:04d}.db"),
            (forbidden_masks.get(t) if forbidden_masks else None),
        )
        for t in timepoints
    ]

    t_start_all = _time.monotonic()

    if n_workers > 1:
        print(f"  Parallel ingest: {n_workers} workers for {n_total} frames …", flush=True)
        engine.dispose()  # close inherited connections before fork so children don't share file descriptors
        ctx = _mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_ingest_frame_worker, worker_args)
    else:
        print(f"  Serial ingest: {n_total} frames …", flush=True)
        results = [_ingest_frame_worker(a) for a in worker_args]

    t_parallel_done = _time.monotonic()

    # Print per-frame summary (results arrive in submission order from Pool.map)
    for res in results:
        t_val, n_raw, n_kept, n_dropped, n_nodes, n_overlaps = res
        print(
            f"  t={t_val}: {n_kept}/{n_raw} unique partitions "
            f"(dropped {n_dropped}), {n_nodes} nodes, {n_overlaps} overlaps",
            flush=True,
        )

    # --- Merge temp DBs into main data.db via ATTACH DATABASE ----------------
    print(f"  Merging {n_total} temp DBs into data.db …", flush=True)
    t_merge_start = _time.monotonic()

    # Use raw SQLite with isolation_level=None (autocommit) so that explicit
    # BEGIN/COMMIT transactions do not hold a cross-database lock that would
    # prevent DETACH after the INSERT.
    import sqlite3 as _sqlite3
    main_db_file = str(db_path).replace("sqlite:///", "")
    conn = _sqlite3.connect(main_db_file, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        for t in timepoints:
            frame_db = str(tmp_dir / f"frame_{t:04d}.db")
            conn.execute(f"ATTACH DATABASE '{frame_db}' AS frame")
            conn.execute("BEGIN")
            conn.execute(f"INSERT INTO {NodeDB.__tablename__} SELECT * FROM frame.{NodeDB.__tablename__}")
            # OverlapDB has an auto-increment rowid 'id' — exclude it so SQLite
            # assigns fresh unique rowids in the main DB.
            conn.execute(
                f"INSERT INTO {OverlapDB.__tablename__} (node_id, ancestor_id) "
                f"SELECT node_id, ancestor_id FROM frame.{OverlapDB.__tablename__}"
            )
            conn.execute("COMMIT")
            conn.execute("DETACH DATABASE frame")
    finally:
        conn.close()

    t_merge_done = _time.monotonic()

    # Clean up temp DBs
    for t in timepoints:
        frame_db = tmp_dir / f"frame_{t:04d}.db"
        frame_db.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass  # not empty — leave it

    total = _time.monotonic() - t_start_all
    parallel_s = t_parallel_done - t_start_all
    merge_s = t_merge_done - t_parallel_done
    print(
        f"  Ingest complete — {n_total} frames in {total/60:.1f}min "
        f"(parallel={parallel_s:.1f}s, merge={merge_s:.1f}s)",
        flush=True,
    )
    LOG.info("Ingestion complete.")


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/linking.py
--------------------------------------------------------------------------------

"""Linking step: wire NodeDB into LinkDB.

Default mode uses Ultrack's built-in linker.
IoU mode uses the custom IoU-weighted linker lifted from v1.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Generator

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.ingest import _build_ultrack_config


def run_linking(
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    overwrite: bool = True,
) -> Generator[tuple[int, int, str], None, None]:
    """Run the linking step, yielding (step, total, label) progress tuples."""
    if cfg.linking_mode == "iou":
        yield from _run_iou_linking(working_dir, cfg, overwrite=overwrite)
        return
    if cfg.linking_mode != "default":
        raise ValueError(f"Unknown linking_mode={cfg.linking_mode!r}")

    total = 3
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)

    from ultrack.core.linking.processing import link
    from ultrack.core.linking.utils import clear_linking_data

    if overwrite:
        yield (0, total, "Clearing existing links…")
        clear_linking_data(ultrack_cfg.data_config.database_path)
    else:
        yield (0, total, "Skipping link clear (overwrite=False)…")

    yield (1, total, "Running Ultrack linker…")
    link(ultrack_cfg, overwrite=False)

    yield (total, total, "Linking done.")


# ---------------------------------------------------------------------------
# IoU-aware linking (lifted from archive/v1/…/ultrack/linking.py)
# ---------------------------------------------------------------------------

def _node_mask(node) -> np.ndarray | None:
    mask = getattr(node, "mask", None)
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    return mask if mask.ndim > 0 else None


def _node_origin(node, ndim: int) -> np.ndarray | None:
    for attr in ("origin", "offset", "bbox_start", "bbox_min", "start"):
        value = getattr(node, attr, None)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= ndim:
            return arr[-ndim:]
    bbox = getattr(node, "bbox", None)
    if bbox is None:
        return None
    if isinstance(bbox, tuple) and bbox and all(hasattr(item, "start") for item in bbox):
        starts = [0.0 if item.start is None else float(item.start) for item in bbox]
        arr = np.asarray(starts, dtype=np.float32).reshape(-1)
        if arr.size >= ndim:
            return arr[-ndim:]
    arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
    if arr.size >= ndim:
        return arr[-ndim:]
    return None


def _centroid_tail(centroid: np.ndarray, ndim: int) -> np.ndarray:
    centroid = np.asarray(centroid, dtype=np.float32).reshape(-1)
    return centroid[-ndim:]


def _aligned_mask_iou(source, target) -> float:
    source_mask = _node_mask(source)
    target_mask = _node_mask(target)
    if source_mask is None or target_mask is None:
        return float(source.IoU(target))
    if source_mask.ndim != target_mask.ndim or source_mask.ndim == 0:
        return float(source.IoU(target))
    if not source_mask.any() or not target_mask.any():
        return 0.0

    ndim = int(source_mask.ndim)
    source_origin = _node_origin(source, ndim)
    target_origin = _node_origin(target, ndim)
    if source_origin is None or target_origin is None:
        return float(source.IoU(target))

    source_coords = np.argwhere(source_mask).astype(np.float32) + source_origin
    target_coords = np.argwhere(target_mask).astype(np.float32) + target_origin
    centroid_shift = _centroid_tail(source.centroid, ndim) - _centroid_tail(target.centroid, ndim)
    target_coords = target_coords + centroid_shift

    all_coords = np.vstack([source_coords, target_coords])
    mins = np.floor(all_coords.min(axis=0)).astype(int) - 1
    maxs = np.ceil(all_coords.max(axis=0)).astype(int) + 1
    shape = tuple((maxs - mins + 1).tolist())
    if any(dim <= 0 for dim in shape):
        return 0.0

    def _rasterize(coords: np.ndarray) -> np.ndarray:
        canvas = np.zeros(shape, dtype=bool)
        idx = np.rint(coords - mins).astype(int)
        valid = np.ones(len(idx), dtype=bool)
        for axis, size in enumerate(shape):
            valid &= (idx[:, axis] >= 0) & (idx[:, axis] < size)
        idx = idx[valid]
        if idx.size:
            canvas[tuple(idx.T)] = True
        return canvas

    source_canvas = _rasterize(source_coords)
    target_canvas = _rasterize(target_coords)
    union = np.logical_or(source_canvas, target_canvas).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(source_canvas, target_canvas).sum() / union)


def _blend_score(iou: float, distance: float, max_distance: float, iou_weight: float) -> float:
    iou_weight = float(np.clip(iou_weight, 0.0, 1.0))
    distance_score = max(0.0, 1.0 - float(distance) / float(max_distance))
    return (1.0 - iou_weight) * distance_score + float(np.clip(iou, 0.0, 1.0)) * iou_weight


def _run_iou_linking(
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    overwrite: bool = True,
) -> Generator[tuple[int, int, str], None, None]:
    total = 4
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)

    from ultrack.core.database import NodeDB, maximum_time_from_database
    from ultrack.core.linking.processing import add_links
    from ultrack.core.linking.utils import clear_linking_data
    from scipy.spatial import KDTree
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session

    if overwrite:
        yield (0, total, "Clearing existing links…")
        clear_linking_data(ultrack_cfg.data_config.database_path)
    else:
        yield (0, total, "Skipping link clear (overwrite=False)…")

    engine = sqla.create_engine(ultrack_cfg.data_config.database_path)
    max_t = int(maximum_time_from_database(ultrack_cfg.data_config))
    if max_t <= 0:
        yield (total, total, "No frames; skipping IoU linking.")
        return

    yield (1, total, "Computing IoU-weighted links…")
    total_links = 0

    with Session(engine) as session:
        for time in range(max_t):
            source_nodes = [n for (n,) in session.query(NodeDB.pickle).where(NodeDB.t == time)]
            target_rows = list(
                session.query(NodeDB.pickle, NodeDB.z_shift, NodeDB.y_shift, NodeDB.x_shift)
                .where(NodeDB.t == time + 1)
            )
            if not source_nodes or not target_rows:
                continue
            target_nodes = [r[0] for r in target_rows]

            source_pos = np.array([n.centroid for n in source_nodes], dtype=np.float32)
            target_pos = np.array([n.centroid for n in target_nodes], dtype=np.float32)
            # apply shift if non-zero
            for i, row in enumerate(target_rows):
                shift = np.asarray(row[1:], dtype=np.float32)
                target_pos[i] += shift[-target_pos.shape[1]:]

            tree = KDTree(source_pos)
            k = min(len(source_nodes), max(1, 2 * cfg.max_neighbors))
            dists, neigh_idx = tree.query(target_pos, k=k, distance_upper_bound=cfg.max_distance)
            if dists.ndim == 1:
                dists, neigh_idx = dists[:, None], neigh_idx[:, None]

            src_ids, tgt_ids, weights = [], [], []
            for ti, (dist_row, ni_row) in enumerate(zip(dists, neigh_idx)):
                target = target_nodes[ti]
                candidates = []
                for dist, si in zip(dist_row, ni_row):
                    if si >= len(source_nodes) or not np.isfinite(dist):
                        continue
                    source = source_nodes[si]
                    iou = _aligned_mask_iou(source, target)
                    if iou < cfg.min_link_iou:
                        continue
                    w = _blend_score(iou, dist, cfg.max_distance, cfg.iou_weight)
                    if w > 0:
                        candidates.append((w, int(source.id), int(target.id)))
                candidates.sort(reverse=True)
                for w, sid, tid in candidates[:cfg.max_neighbors]:
                    src_ids.append(sid)
                    tgt_ids.append(tid)
                    weights.append(w)

            if src_ids:
                add_links(ultrack_cfg, src_ids, tgt_ids, weights)
                total_links += len(src_ids)

            yield (
                1 + int(math.floor((time + 1) / max(max_t, 1) * 2)),
                total,
                f"Linked t={time + 1}/{max_t} ({len(src_ids)} edges)",
            )

    yield (total, total, f"IoU linking done ({total_links} total edges).")


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/metrics.py
--------------------------------------------------------------------------------

"""Comparison metrics for tracked labelmaps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackSummary:
    n_tracks: int
    average_length: float
    track_lengths: dict[int, int]


def tracked_label_summary(labels: np.ndarray) -> TrackSummary:
    """Return track count and frame-presence lengths for a tracked labelmap."""
    arr = np.asarray(labels)
    if arr.ndim < 2:
        raise ValueError(f"Expected labels with time as axis 0, got shape {arr.shape}")

    track_lengths: dict[int, int] = {}
    for t in range(arr.shape[0]):
        ids = np.unique(arr[t])
        for label_id in ids:
            label_int = int(label_id)
            if label_int == 0:
                continue
            track_lengths[label_int] = track_lengths.get(label_int, 0) + 1

    lengths = list(track_lengths.values())
    average_length = float(np.mean(lengths)) if lengths else 0.0
    return TrackSummary(
        n_tracks=len(track_lengths),
        average_length=average_length,
        track_lengths=track_lengths,
    )


def binary_labelmap_iou(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Return foreground IoU after binarizing two labelmaps with ``label > 0``."""
    left = np.asarray(lhs) > 0
    right = np.asarray(rhs) > 0
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} != {right.shape}")

    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(left, right).sum()
    return float(intersection / union)


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/reseed.py
--------------------------------------------------------------------------------

"""Re-solve with validated tracks using annotated Ultrack seed nodes.

The validate-and-resolve loop lets the user lock in confirmed cell tracks
(validated_tracks), then re-solve the rest of the stack — possibly with
tweaked parameters — without losing the validated work.

The contract for `validated_tracks` is `dict[int, set[int]]` where keys are
cell IDs and values are the set of frames at which that cell is validated.
This is independent of the on-disk JSON shape (which is managed separately by
`cellflow.database.validation`).

`prune_validated_overlaps` and `_build_frame_masks` are kept for existing
unit tests and ad hoc use.  `merge_validated_into_export` is called at the
end of `resolve_with_canonical_segment` to paste validated IDs back.
"""
from __future__ import annotations

from collections.abc import Callable
import logging
import pickle
import tempfile
from pathlib import Path

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.db_build import build_ultrack_database
from cellflow.tracking_ultrack.export import export_tracked_labels
from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db
from cellflow.tracking_ultrack.linking import run_linking
from cellflow.tracking_ultrack.seed_prior import boost_validated_edges, write_seed_prior_node_probs
from cellflow.tracking_ultrack.solve import run_solve
from cellflow.tracking_ultrack.validation_nodes import inject_validated_nodes

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: build per-frame validated-mask index
# ---------------------------------------------------------------------------

def _build_frame_masks(
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
) -> dict[int, list[tuple[int, np.ndarray, tuple[int, int, int, int]]]]:
    """Build a per-frame index of validated masks.

    Returns
    -------
    dict mapping frame index ``t`` to a list of
    ``(cell_id, bool_mask_crop, (y0, x0, y1, x1))`` tuples, one per validated
    cell at that frame.  Both ``bool_mask_crop`` and the bbox refer to the
    cropped region (exclusive-max convention).

    Only frames where at least one cell is validated are included.
    The 2D spatial slice is taken from ``tracked_labels[t]`` (shape Y×X).
    For a 4-D input ``(T, Z, Y, X)`` the z-axis is max-projected to give a
    2D footprint for intersection tests (conservative: any z-slice counts).
    """
    frame_index: dict[int, list[tuple[int, np.ndarray, tuple[int, int, int, int]]]] = {}

    for cell_id, frames in validated_tracks.items():
        for t in frames:
            t = int(t)
            frame_vol = tracked_labels[t]  # (Y, X) or (Z, Y, X)

            # Project to 2D for intersection (max over Z if 3D)
            if frame_vol.ndim == 3:
                frame_2d = (frame_vol == cell_id).any(axis=0)
            else:
                frame_2d = (frame_vol == cell_id)

            if not frame_2d.any():
                # Cell absent at this frame — skip silently
                continue

            # Tight bbox around the mask
            rows = np.where(frame_2d.any(axis=1))[0]
            cols = np.where(frame_2d.any(axis=0))[0]
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            x0, x1 = int(cols[0]), int(cols[-1]) + 1
            mask_crop = frame_2d[y0:y1, x0:x1]

            if t not in frame_index:
                frame_index[t] = []
            frame_index[t].append((cell_id, mask_crop, (y0, x0, y1, x1)))

    return frame_index


def _build_frame_forbidden_masks(
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Return ``{t: bool_array (Y, X)}`` — union of all validated cell footprints per frame.

    Frames with no validated cells are omitted.  For 4-D ``(T, Z, Y, X)`` input
    the z-axis is max-projected to give a conservative 2D footprint.
    """
    if not validated_tracks:
        return {}

    n_frames = tracked_labels.shape[0]
    if tracked_labels.ndim == 4:
        Y, X = tracked_labels.shape[2], tracked_labels.shape[3]
    else:
        Y, X = tracked_labels.shape[1], tracked_labels.shape[2]

    # Group cell IDs by frame so we touch each frame slice once.
    cells_by_frame: dict[int, list[int]] = {}
    for cell_id, frames in validated_tracks.items():
        for t in frames:
            cells_by_frame.setdefault(int(t), []).append(int(cell_id))

    out: dict[int, np.ndarray] = {}
    for t, cell_ids in cells_by_frame.items():
        if t < 0 or t >= n_frames:
            continue
        frame_vol = tracked_labels[t]
        if frame_vol.ndim == 3:
            frame_2d = np.isin(frame_vol, cell_ids).any(axis=0)
        else:
            frame_2d = np.isin(frame_vol, cell_ids)
        if frame_2d.any():
            out[t] = np.ascontiguousarray(frame_2d, dtype=bool)
    return out


# ---------------------------------------------------------------------------
# 1. prune_validated_overlaps
# ---------------------------------------------------------------------------

def prune_validated_overlaps(
    working_dir: str | Path,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
) -> int:
    """Delete NodeDB rows that overlap any validated mask and cascade to OverlapDB.

    Parameters
    ----------
    working_dir:
        Ultrack working directory containing ``data.db``.
    validated_tracks:
        ``{cell_id: {frames}}`` — which frames each validated cell spans.
    tracked_labels:
        Current corrected labelmap, shape ``(T, Y, X)`` or ``(T, Z, Y, X)``.

    Returns
    -------
    int
        Number of NodeDB rows deleted.

    Notes
    -----
    Conflict criterion: **any pixel intersection** between a hypothesis node's
    2D footprint and any validated cell mask at the same frame counts as a
    conflict.  The node (and all OverlapDB rows referencing it) is deleted.

    The 3D node masks stored in NodeDB are stored as ``(1, h, w)`` crops
    (the ingest convention in ``ingest.py``).  We squeeze the Z=1 dimension
    for the 2D intersection test.  True 3D (Z > 1) is currently not supported;
    the bbox test gracefully falls back to a bounding-box-only check in that
    case.

    Both NodeDB and referencing OverlapDB rows are deleted in a single
    SQLite transaction to preserve DB consistency.
    """
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, OverlapDB

    if not validated_tracks:
        return 0

    working_dir = Path(working_dir)
    db_url = f"sqlite:///{working_dir / 'data.db'}"
    engine = sqla.create_engine(db_url)

    frame_masks = _build_frame_masks(validated_tracks, tracked_labels)
    if not frame_masks:
        return 0

    total_deleted = 0

    with Session(engine) as session:
        for t, val_cells in frame_masks.items():
            # Query all nodes at this frame (id + pickle blob)
            rows = (
                session.query(NodeDB.id, NodeDB.pickle)
                .where(NodeDB.t == t)
                .all()
            )
            if not rows:
                continue

            conflict_ids: list[int] = []

            for node_id_val, node_blob in rows:
                # Deserialize; NodeDB.pickle is a MaybePickleType that
                # returns a Node object on read.
                if isinstance(node_blob, (bytes, memoryview)):
                    try:
                        node = pickle.loads(bytes(node_blob))
                    except Exception:
                        LOG.warning(
                            "Could not unpickle node id=%s at t=%s — skipping",
                            node_id_val, t,
                        )
                        continue
                else:
                    # SQLAlchemy MaybePickleType already deserialized it
                    node = node_blob

                bbox = np.asarray(node.bbox)
                ndim = len(bbox) // 2
                if ndim == 3:
                    # bbox = [z0, y0, x0, z1, y1, x1]
                    ny0, nx0, ny1, nx1 = int(bbox[1]), int(bbox[2]), int(bbox[4]), int(bbox[5])
                    node_mask = getattr(node, "mask", None)
                    if node_mask is not None:
                        node_mask = np.asarray(node_mask, dtype=bool)
                        if node_mask.shape[0] == 1:
                            node_mask_2d = node_mask[0]  # (h, w)
                        else:
                            # Multi-Z: project
                            node_mask_2d = node_mask.any(axis=0)
                    else:
                        node_mask_2d = None
                elif ndim == 2:
                    # bbox = [y0, x0, y1, x1]
                    ny0, nx0, ny1, nx1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    node_mask = getattr(node, "mask", None)
                    if node_mask is not None:
                        node_mask_2d = np.asarray(node_mask, dtype=bool)
                        if node_mask_2d.ndim == 3 and node_mask_2d.shape[0] == 1:
                            node_mask_2d = node_mask_2d[0]
                    else:
                        node_mask_2d = None
                else:
                    LOG.warning(
                        "Unexpected bbox ndim=%s for node id=%s — skipping",
                        ndim, node_id_val,
                    )
                    continue

                # Check against every validated cell at this frame
                is_conflict = False
                for _cell_id, val_crop, (vy0, vx0, vy1, vx1) in val_cells:
                    # Fast bbox intersection test first
                    if ny1 <= vy0 or vy1 <= ny0 or nx1 <= vx0 or vx1 <= nx0:
                        continue  # bboxes don't overlap → no conflict

                    # Compute the overlap region in image coordinates
                    oy0, oy1 = max(ny0, vy0), min(ny1, vy1)
                    ox0, ox1 = max(nx0, vx0), min(nx1, vx1)

                    if node_mask_2d is None:
                        # No mask data — treat bbox overlap as conflict
                        is_conflict = True
                        break

                    # Crop both masks to the overlap region
                    nm_crop = node_mask_2d[
                        oy0 - ny0: oy1 - ny0,
                        ox0 - nx0: ox1 - nx0,
                    ]
                    vm_crop = val_crop[
                        oy0 - vy0: oy1 - vy0,
                        ox0 - vx0: ox1 - vx0,
                    ]
                    if nm_crop.shape != vm_crop.shape:
                        # Shape mismatch due to off-by-one — treat as conflict
                        is_conflict = True
                        break
                    if np.any(nm_crop & vm_crop):
                        is_conflict = True
                        break

                if is_conflict:
                    conflict_ids.append(int(node_id_val))

            if not conflict_ids:
                continue

            # Delete OverlapDB rows referencing any conflict node, then nodes —
            # all in one transaction (the session is autocommit=False by default).
            (
                session.query(OverlapDB)
                .where(
                    sqla.or_(
                        OverlapDB.node_id.in_(conflict_ids),
                        OverlapDB.ancestor_id.in_(conflict_ids),
                    )
                )
                .delete(synchronize_session=False)
            )
            (
                session.query(NodeDB)
                .where(NodeDB.id.in_(conflict_ids))
                .delete(synchronize_session=False)
            )
            total_deleted += len(conflict_ids)

        session.commit()

    LOG.info("prune_validated_overlaps: deleted %d NodeDB rows", total_deleted)
    return total_deleted


# ---------------------------------------------------------------------------
# 2. merge_validated_into_export
# ---------------------------------------------------------------------------

def merge_validated_into_export(
    exported_labels: np.ndarray,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    """Paste validated masks back onto the solver's exported labelmap.

    Validated ``cell_id`` values are reserved and pasted back with their
    original IDs so the validation store remains stable across resolve runs.
    If the solver independently used a reserved validated ID elsewhere, those
    solver pixels are moved to a fresh ID before the validated mask is pasted.
    Validated pixels overwrite any conflicting content in the exported labelmap.

    Parameters
    ----------
    exported_labels:
        Labelmap produced by Ultrack export, shape ``(T, Y, X)`` or
        ``(T, Z, Y, X)``.  Modified **in place** and returned.
    validated_tracks:
        ``{cell_id: {frames}}`` from the validation store.
    tracked_labels:
        Current corrected labelmap (same spatial shape as ``exported_labels``).
        The source of ground-truth masks.

    Returns
    -------
    tuple[np.ndarray, dict[int, int]]
        ``(merged_labelmap, id_map)`` where ``id_map`` maps original validated
        IDs to changed output IDs. The normal path preserves validated IDs, so
        ``id_map`` is usually empty. The labelmap is the same array as
        ``exported_labels``, modified in place.
    """
    if not validated_tracks:
        return exported_labels, {}

    id_map: dict[int, int] = {}
    reserved_ids = {int(cell_id) for cell_id, frames in validated_tracks.items() if frames}
    used_fresh_ids = set(int(v) for v in np.unique(exported_labels))

    def _next_fresh_id() -> int:
        next_id = int(exported_labels.max()) + 1
        while next_id in reserved_ids or next_id in used_fresh_ids:
            next_id += 1
        used_fresh_ids.add(next_id)
        return next_id

    # Build validated_masks: per cell_id, list of (t, mask) pairs present in tracked_labels
    validated_masks: dict[int, list[tuple[int, np.ndarray]]] = {}
    for cell_id, frames in validated_tracks.items():
        cell_id = int(cell_id)
        if not frames:
            continue
        present: list[tuple[int, np.ndarray]] = []
        for t in sorted(frames):
            t = int(t)
            if t < 0 or t >= tracked_labels.shape[0] or t >= exported_labels.shape[0]:
                continue
            frame_src = tracked_labels[t]
            mask = np.asarray(frame_src == cell_id)
            if mask.any():
                present.append((t, mask))
        if present:
            validated_masks[cell_id] = present

    # Build solver_track_remap: solver_id -> validated cell_id
    # Inspect each validated frame to find the dominant solver track covering that mask region,
    # then propagate that solver track's ID to the validated cell_id everywhere in the export.
    solver_track_remap: dict[int, int] = {}
    for cell_id in sorted(validated_masks.keys()):
        present_masks = validated_masks[cell_id]
        dominant_solver_id: int | None = None
        best_count = 0
        for t, mask in present_masks:
            if exported_labels.ndim == 4:
                # (T, Z, Y, X): project mask to 3D for indexing
                frame_exp = exported_labels[t]
                if mask.ndim == 2:
                    solver_pixels = frame_exp[:, mask]
                elif mask.ndim == 3:
                    solver_pixels = frame_exp[mask]
                else:
                    continue
            else:
                frame_exp = exported_labels[t]
                if mask.ndim == frame_exp.ndim:
                    solver_pixels = frame_exp[mask]
                elif mask.ndim == 2 and frame_exp.ndim == 3:
                    solver_pixels = frame_exp[:, mask]
                elif mask.ndim == 3 and frame_exp.ndim == 2:
                    solver_pixels = frame_exp[mask.any(axis=0)]
                else:
                    continue
            nonzero = solver_pixels[solver_pixels != 0]
            if nonzero.size == 0:
                continue
            unique_ids, counts = np.unique(nonzero, return_counts=True)
            t_best_idx = int(np.argmax(counts))
            t_best_id = int(unique_ids[t_best_idx])
            t_best_count = int(counts[t_best_idx])
            if t_best_count > best_count:
                best_count = t_best_count
                dominant_solver_id = t_best_id

        if dominant_solver_id is None or dominant_solver_id == 0:
            continue
        if dominant_solver_id == cell_id:
            # Solver already used the correct ID — no remap needed, but still handle
            # collision below where the solver used cell_id for unrelated pixels
            pass
        elif dominant_solver_id in solver_track_remap:
            LOG.warning(
                "Solver track %d claimed by multiple validated cells; "
                "keeping first claim (cell %d), skipping cell %d",
                dominant_solver_id,
                solver_track_remap[dominant_solver_id],
                cell_id,
            )
            continue
        else:
            solver_track_remap[dominant_solver_id] = cell_id

    # Resolve collisions: if any existing pixels have value == target cell_id
    # but DON'T belong to the source solver track, move them to a fresh ID.
    # This is the same as the original collision handling, now generalized to
    # cover both the remap targets and direct-match cases.
    for solver_id, target_cell_id in list(solver_track_remap.items()):
        if solver_id == target_cell_id:
            continue
        # Pixels currently carrying target_cell_id that are NOT from this solver track
        collision_mask = np.asarray(exported_labels == target_cell_id)
        # Remove pixels belonging to the source solver track from collision set
        src_mask = np.asarray(exported_labels == solver_id)
        # Exclude src_mask locations — those will become target_cell_id after remap anyway
        collision_mask = collision_mask & ~src_mask
        if collision_mask.any():
            exported_labels[collision_mask] = _next_fresh_id()

    # Handle cells that are not remapped (dominant_solver_id == cell_id or no dominant found)
    # — original collision logic for reserved IDs used by solver for unrelated pixels.
    remapped_sources = set(solver_track_remap.keys())
    remapped_targets = set(solver_track_remap.values())
    for cell_id, present_masks in validated_masks.items():
        if cell_id in remapped_targets and cell_id not in remapped_sources:
            # Already handled in collision resolution above
            continue
        if cell_id in remapped_sources:
            # This cell IS the source solver track being remapped — no separate collision needed
            continue
        # No remap found: solver either didn't have this cell or already used correct ID.
        # Run original collision resolution for the reserved cell_id.
        solver_collision = np.asarray(exported_labels == cell_id)
        for t, mask in present_masks:
            frame_collision = solver_collision[t]
            if mask.ndim == frame_collision.ndim:
                frame_collision[mask] = False
            elif mask.ndim == 2 and frame_collision.ndim == 3:
                frame_collision[:, mask] = False
            elif mask.ndim == 3 and frame_collision.ndim == 2:
                frame_collision[mask.any(axis=0)] = False
            else:
                raise ValueError(
                    "Validated mask shape is incompatible with exported labels: "
                    f"mask={mask.shape}, exported={frame_collision.shape}"
                )
        if solver_collision.any():
            exported_labels[solver_collision] = _next_fresh_id()

    # Apply solver track remap: rename solver track IDs to validated cell IDs
    for solver_id, target_cell_id in solver_track_remap.items():
        if solver_id == target_cell_id:
            continue
        exported_labels[exported_labels == solver_id] = target_cell_id

    # Paste validated masks (overrides geometry at validated frames)
    for cell_id, present_masks in validated_masks.items():
        for t, mask in present_masks:
            frame_out = exported_labels[t]
            if mask.ndim == frame_out.ndim:
                frame_out[mask] = cell_id
            elif mask.ndim == 2 and frame_out.ndim == 3:
                frame_out[:, mask] = cell_id
            elif mask.ndim == 3 and frame_out.ndim == 2:
                frame_out[mask.any(axis=0)] = cell_id
            else:
                raise ValueError(
                    "Validated mask shape is incompatible with exported labels: "
                    f"mask={mask.shape}, exported={frame_out.shape}"
                )

    return exported_labels, id_map


def resolve_with_validation(
    hypotheses_path: str | Path,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
    cfg: TrackingConfig,
    progress_cb: Callable[[str], None] | None = None,
    *,
    intensity_image_path: str | Path,
) -> tuple[np.ndarray, dict[int, int]]:
    """Re-ingest hypotheses, inject validated nodes, solve with annotations."""
    def _notify(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    hypotheses_path = Path(hypotheses_path)
    if not validated_tracks:
        return np.asarray(tracked_labels, dtype=np.uint32).copy(), {}

    with tempfile.TemporaryDirectory(prefix="cellflow_resolve_workdir_") as tmp_workdir:
        working_dir = Path(tmp_workdir)

        _notify("Ingesting hypotheses…")
        ingest_hypotheses_to_db(hypotheses_path, working_dir, cfg, overwrite=True)

        _notify("Injecting validated masks…")
        injection = inject_validated_nodes(working_dir, validated_tracks, tracked_labels, cfg)
        if injection.skipped_missing:
            _notify(
                f"Skipped {injection.skipped_missing} validated cell-frame(s) "
                "missing from tracked labels."
            )
        if injection.inserted == 0:
            raise ValueError("No validated masks could be injected; resolve aborted before solve.")

        _notify("Scoring node probabilities…")
        write_seed_prior_node_probs(working_dir, intensity_image_path, cfg)

        _notify("Linking hypotheses…")
        for _step, _total, label in run_linking(working_dir, cfg):
            _notify(label)

        _notify("Solving ILP with annotations…")
        for _step, _total, label in run_solve(
            working_dir,
            cfg,
            overwrite=True,
            use_annotations=True,
        ):
            _notify(label)

        _notify("Exporting tracks…")
        with tempfile.TemporaryDirectory(prefix="cellflow_resolve_export_") as tmpdir:
            out_path = Path(tmpdir) / "tracked_labels.tif"
            exported = export_tracked_labels(
                working_dir,
                cfg,
                out_path,
                validated_tracks=validated_tracks,
                tracked_labels=tracked_labels,
            )

    exported = np.asarray(exported, dtype=np.uint32)
    exported, id_map = merge_validated_into_export(
        exported,
        validated_tracks,
        tracked_labels,
    )
    return exported, id_map


def resolve_with_canonical_segment(
    contour_maps_path: str | Path,
    foreground_masks_path: str | Path,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
    cfg: "TrackingConfig",
    progress_cb: Callable[[str], None] | None = None,
    *,
    intensity_image_path: str | Path,
) -> tuple[np.ndarray, dict[int, int]]:
    """Re-solve using canonical ultrack.segment instead of hypotheses.h5.

    Replaces the old resolve_with_validation() call chain:
      old: hypotheses.h5 → ingest_hypotheses_to_db → inject → score → link → solve
      new: foreground_masks + contour_maps → ultrack.segment → inject → score → link → solve
    """
    def _notify(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    contour_maps_path = Path(contour_maps_path)
    foreground_masks_path = Path(foreground_masks_path)

    if not validated_tracks:
        return np.asarray(tracked_labels, dtype=np.uint32).copy(), {}

    with tempfile.TemporaryDirectory(prefix="cellflow_resolve_") as tmp_dir:
        working_dir = Path(tmp_dir)
        build_ultrack_database(
            contour_maps_path=contour_maps_path,
            foreground_masks_path=foreground_masks_path,
            nucleus_prob_zavg_path=intensity_image_path,
            working_dir=working_dir,
            cfg=cfg,
            validated_tracks=validated_tracks,
            tracked_labels=np.asarray(tracked_labels, dtype=np.uint32),
            use_validated=True,
            progress_cb=_notify,
        )

        _notify("Solving ILP…")
        for _step, _total, label in run_solve(working_dir, cfg, overwrite=True):
            _notify(f"[solve] {label}")

        _notify("Exporting tracked labels…")
        tmp_out = working_dir / "tracked_labels_resolve.tif"
        export_tracked_labels(
            working_dir,
            cfg,
            tmp_out,
            validated_tracks=validated_tracks,
            tracked_labels=np.asarray(tracked_labels, dtype=np.uint32),
        )
        import tifffile
        new_labels = np.asarray(tifffile.imread(str(tmp_out)), dtype=np.uint32)
        if new_labels.ndim == 4 and new_labels.shape[1] == 1:
            new_labels = new_labels[:, 0]

    _notify("Pasting validated IDs back into export…")
    new_labels, id_map = merge_validated_into_export(
        new_labels,
        validated_tracks,
        np.asarray(tracked_labels, dtype=np.uint32),
    )

    return new_labels, id_map


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/seed_prior.py
--------------------------------------------------------------------------------

"""Resolve-time node probability scoring from image quality and validated seeds."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import tifffile

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.validation_nodes import _node_bbox_and_mask as _node_mask_record


@dataclass(frozen=True)
class SeedPriorReport:
    scored: int
    seeds: int


@dataclass(frozen=True)
class EdgeBoostReport:
    boosted: int
    seeds: int


@dataclass(frozen=True)
class _NodeScoreRecord:
    node_id: int
    t: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    area: int
    y: float
    x: float


def _load_signal_stack(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Nucleus zavg image not found: {path}")

    arr = np.asarray(tifffile.imread(path), dtype=np.float32)
    if arr.ndim == 2:
        return arr[np.newaxis]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError(
        f"Expected nucleus zavg image to be 2D, 3D, or singleton-Z 4D, got {arr.shape}"
    )


def _binary_dilation_2d(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, mode="constant")
    h, w = mask.shape
    dilated = np.zeros((h, w), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            dilated |= padded[dy:dy + h, dx:dx + w]
    return dilated


def compute_drop_frac(frame: np.ndarray, bbox: tuple[int, int, int, int], mask: np.ndarray) -> float:
    y0, x0, y1, x1 = bbox
    frame = np.asarray(frame, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    inside = frame[y0:y1, x0:x1][mask]
    if inside.size == 0:
        return 0.0
    inside_median = float(np.median(inside))

    pad_y0 = max(0, y0 - 1)
    pad_x0 = max(0, x0 - 1)
    pad_y1 = min(frame.shape[0], y1 + 1)
    pad_x1 = min(frame.shape[1], x1 + 1)

    expanded = np.zeros((pad_y1 - pad_y0, pad_x1 - pad_x0), dtype=bool)
    inner_y0 = y0 - pad_y0
    inner_x0 = x0 - pad_x0
    expanded[
        inner_y0:inner_y0 + mask.shape[0],
        inner_x0:inner_x0 + mask.shape[1],
    ] = mask
    ring = _binary_dilation_2d(expanded) & ~expanded
    if not ring.any():
        return 0.0

    ring_values = frame[pad_y0:pad_y1, pad_x0:pad_x1][ring]
    return float(np.mean(ring_values < inside_median))


def compute_mask_circularity(mask: np.ndarray) -> float:
    from skimage.measure import perimeter

    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if area == 0:
        return 0.0

    perimeter_px = float(perimeter(mask, neighborhood=4))
    if perimeter_px <= 0.0:
        return 0.0

    circularity = 4.0 * math.pi * float(area) / (perimeter_px * perimeter_px)
    return float(np.clip(circularity, 0.0, 1.0))


def _affinity(node: _NodeScoreRecord, seed: _NodeScoreRecord, cfg: TrackingConfig) -> float:
    if node.area <= 0 or seed.area <= 0:
        return 0.0
    dt = abs(int(node.t) - int(seed.t))
    if dt > cfg.seed_max_dt:
        return 0.0

    area_ratio = float(node.area) / float(seed.area)
    size_similarity = np.exp(-abs(np.log(area_ratio)) / cfg.seed_sigma_area)
    dist = float(np.hypot(node.y - seed.y, node.x - seed.x))
    spatial_decay = np.exp(-((dist / cfg.seed_sigma_space) ** 2))
    temporal_decay = np.exp(-(dt / cfg.seed_tau_time))
    return float(size_similarity * spatial_decay * temporal_decay)


def boost_validated_edges(
    working_dir: str | Path,
    cfg: TrackingConfig,
) -> EdgeBoostReport:
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, NodeDB, VarAnnotation

    engine = sqla.create_engine(f"sqlite:///{Path(working_dir) / 'data.db'}")

    with Session(engine) as session:
        real_rows = session.query(
            NodeDB.id,
            NodeDB.t,
            NodeDB.y,
            NodeDB.x,
            NodeDB.area,
        ).where(NodeDB.node_annot == VarAnnotation.REAL).all()

        if not real_rows:
            return EdgeBoostReport(boosted=0, seeds=0)

        seed_by_id: dict[int, _NodeScoreRecord] = {}
        for node_id, t, y, x, area in real_rows:
            seed_by_id[int(node_id)] = _NodeScoreRecord(
                node_id=int(node_id),
                t=int(t),
                bbox=(0, 0, 1, 1),
                mask=np.zeros((1, 1), dtype=bool),
                area=int(area),
                y=float(y),
                x=float(x),
            )

        real_ids = list(seed_by_id.keys())

        link_rows = session.query(
            LinkDB.id,
            LinkDB.source_id,
            LinkDB.target_id,
            LinkDB.weight,
        ).where(
            sqla.or_(
                LinkDB.source_id.in_(real_ids),
                LinkDB.target_id.in_(real_ids),
            )
        ).all()

        if not link_rows:
            return EdgeBoostReport(boosted=0, seeds=len(real_ids))

        # Gather candidate (non-REAL-endpoint) node IDs for batched lookup
        candidate_ids: set[int] = set()
        for _link_id, source_id, target_id, _weight in link_rows:
            source_id, target_id = int(source_id), int(target_id)
            if source_id not in seed_by_id:
                candidate_ids.add(source_id)
            if target_id not in seed_by_id:
                candidate_ids.add(target_id)

        candidate_by_id: dict[int, _NodeScoreRecord] = {}
        if candidate_ids:
            cand_rows = session.query(
                NodeDB.id,
                NodeDB.t,
                NodeDB.y,
                NodeDB.x,
                NodeDB.area,
            ).where(NodeDB.id.in_(list(candidate_ids))).all()
            for node_id, t, y, x, area in cand_rows:
                candidate_by_id[int(node_id)] = _NodeScoreRecord(
                    node_id=int(node_id),
                    t=int(t),
                    bbox=(0, 0, 1, 1),
                    mask=np.zeros((1, 1), dtype=bool),
                    area=int(area),
                    y=float(y),
                    x=float(x),
                )

        boosted = 0
        for link_id, source_id, target_id, weight in link_rows:
            link_id = int(link_id)
            source_id, target_id = int(source_id), int(target_id)

            # Determine which endpoint is the seed (REAL) and which is the candidate
            if source_id in seed_by_id:
                seed = seed_by_id[source_id]
                candidate = seed_by_id.get(target_id) or candidate_by_id.get(target_id)
            else:
                seed = seed_by_id[target_id]
                candidate = candidate_by_id.get(source_id)

            if candidate is None:
                continue

            aff = _affinity(candidate, seed, cfg)
            if aff == 0.0:
                continue

            new_weight = float(weight) + cfg.seed_weight * aff
            session.query(LinkDB).where(LinkDB.id == link_id).update(
                {LinkDB.weight: new_weight},
                synchronize_session=False,
            )
            boosted += 1

        session.commit()

    engine.dispose()
    return EdgeBoostReport(boosted=boosted, seeds=len(real_ids))


def write_seed_prior_node_probs(
    working_dir: str | Path,
    intensity_image_path: str | Path,
    cfg: TrackingConfig,
) -> SeedPriorReport:
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, VarAnnotation

    signal = _load_signal_stack(intensity_image_path)
    engine = sqla.create_engine(f"sqlite:///{Path(working_dir) / 'data.db'}")

    with Session(engine) as session:
        rows = session.query(
            NodeDB.id,
            NodeDB.t,
            NodeDB.pickle,
            NodeDB.area,
            NodeDB.y,
            NodeDB.x,
            NodeDB.node_annot,
        ).all()

        records: list[_NodeScoreRecord] = []
        seed_records: list[_NodeScoreRecord] = []
        for node_id, t, node, area, y, x, annot in rows:
            bbox, mask = _node_mask_record(int(node_id), node)
            record = _NodeScoreRecord(
                node_id=int(node_id),
                t=int(t),
                bbox=bbox,
                mask=mask,
                area=int(area),
                y=float(y),
                x=float(x),
            )
            if annot == VarAnnotation.REAL:
                seed_records.append(record)
            else:
                records.append(record)

        scored = 0
        for record in records:
            if record.t >= signal.shape[0]:
                raise ValueError(
                    f"Nucleus zavg image has {signal.shape[0]} frame(s), "
                    f"cannot score node at t={record.t}"
                )
            drop_frac = compute_drop_frac(signal[record.t], record.bbox, record.mask)
            best_affinity = max(
                (_affinity(record, seed, cfg) for seed in seed_records),
                default=0.0,
            )
            circularity = compute_mask_circularity(record.mask)
            node_prob = float(
                cfg.quality_weight * (drop_frac ** cfg.quality_exponent)
                + cfg.circularity_weight * circularity
                + cfg.seed_weight * best_affinity
            )
            session.query(NodeDB).where(NodeDB.id == record.node_id).update(
                {NodeDB.node_prob: node_prob},
                synchronize_session=False,
            )
            scored += 1

        for seed in seed_records:
            session.query(NodeDB).where(NodeDB.id == seed.node_id).update(
                {NodeDB.node_prob: 1.0},
                synchronize_session=False,
            )
        session.commit()

    engine.dispose()
    return SeedPriorReport(scored=scored, seeds=len(seed_records))


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/solve.py
--------------------------------------------------------------------------------

"""Thin wrapper around ultrack.core.solve.processing.solve."""
from __future__ import annotations

from pathlib import Path
from typing import Generator

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.ingest import _build_ultrack_config


def database_has_annotations(working_dir: str | Path) -> bool:
    """Return whether ``data.db`` contains REAL or FAKE node annotations."""
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, VarAnnotation

    db_path = Path(working_dir) / "data.db"
    if not db_path.exists():
        return False

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            return bool(
                session.query(NodeDB.id)
                .where(NodeDB.node_annot.in_([VarAnnotation.REAL, VarAnnotation.FAKE]))
                .limit(1)
                .first()
            )
    finally:
        engine.dispose()


def run_solve(
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    overwrite: bool = True,
    use_annotations: bool | None = None,
) -> Generator[tuple[int, int, str], None, None]:
    """Run the ILP solver, yielding (step, total, label) progress tuples."""
    from ultrack.core.solve.processing import solve

    total = 2
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)
    if use_annotations is None:
        use_annotations = database_has_annotations(wd)

    yield (0, total, "Running ILP solver…")
    solve(ultrack_cfg, overwrite=overwrite, use_annotations=use_annotations)
    yield (total, total, "Solve done.")


--------------------------------------------------------------------------------
FILE: src/cellflow/tracking_ultrack/validation_nodes.py
--------------------------------------------------------------------------------

"""Inject validated tracked labels as annotated Ultrack nodes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig


@dataclass(frozen=True)
class ValidationInjectionReport:
    inserted: int
    skipped_missing: int
    skipped: list[tuple[int, int]]
    faked: int
    overlaps_added: int


@dataclass(frozen=True)
class _MaskRecord:
    cell_id: int
    t: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    area: int
    y: float
    x: float


def _frame_mask_for_cell(tracked_labels: np.ndarray, t: int, cell_id: int) -> np.ndarray:
    frame = np.asarray(tracked_labels)[t]
    if frame.ndim == 2:
        return np.asarray(frame == cell_id)
    if frame.ndim == 3:
        return np.asarray(frame == cell_id).any(axis=0)
    raise ValueError(f"Expected tracked frame to be 2D or 3D, got shape {frame.shape}")


def _validated_mask_records(
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
) -> tuple[list[_MaskRecord], list[tuple[int, int]]]:
    records: list[_MaskRecord] = []
    skipped: list[tuple[int, int]] = []
    labels = np.asarray(tracked_labels)
    n_frames = int(labels.shape[0])

    for cell_id, frames in sorted(validated_tracks.items()):
        for raw_t in sorted(frames):
            t = int(raw_t)
            if t < 0 or t >= n_frames:
                skipped.append((int(cell_id), t))
                continue

            mask_2d = _frame_mask_for_cell(labels, t, int(cell_id))
            if not mask_2d.any():
                skipped.append((int(cell_id), t))
                continue

            rows = np.flatnonzero(mask_2d.any(axis=1))
            cols = np.flatnonzero(mask_2d.any(axis=0))
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            x0, x1 = int(cols[0]), int(cols[-1]) + 1
            crop = np.ascontiguousarray(mask_2d[y0:y1, x0:x1], dtype=bool)
            ys, xs = np.nonzero(crop)
            records.append(
                _MaskRecord(
                    cell_id=int(cell_id),
                    t=t,
                    bbox=(y0, x0, y1, x1),
                    mask=crop,
                    area=int(crop.sum()),
                    y=float(y0 + ys.mean()),
                    x=float(x0 + xs.mean()),
                )
            )

    return records, skipped


def _node_bbox_and_mask(node_id: int, node) -> tuple[tuple[int, int, int, int], np.ndarray]:
    if isinstance(node, (bytes, memoryview)):
        node = pickle.loads(bytes(node))

    bbox = np.asarray(node.bbox)
    ndim = len(bbox) // 2
    if ndim == 3:
        y0, x0 = int(bbox[1]), int(bbox[2])
        y1, x1 = int(bbox[4]), int(bbox[5])
    elif ndim == 2:
        y0, x0 = int(bbox[0]), int(bbox[1])
        y1, x1 = int(bbox[2]), int(bbox[3])
    else:
        raise ValueError(f"Unexpected bbox for node {node_id}: {bbox}")

    mask = np.asarray(node.mask, dtype=bool)
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] == 1 else mask.any(axis=0)
    elif mask.ndim != 2:
        raise ValueError(f"Unexpected mask for node {node_id}: shape {mask.shape}")

    return (y0, x0, y1, x1), np.ascontiguousarray(mask, dtype=bool)


def _intersects(
    lhs_bbox: tuple[int, int, int, int],
    lhs_mask: np.ndarray,
    rhs_bbox: tuple[int, int, int, int],
    rhs_mask: np.ndarray,
) -> bool:
    ly0, lx0, ly1, lx1 = lhs_bbox
    ry0, rx0, ry1, rx1 = rhs_bbox
    oy0, ox0 = max(ly0, ry0), max(lx0, rx0)
    oy1, ox1 = min(ly1, ry1), min(lx1, rx1)
    if oy0 >= oy1 or ox0 >= ox1:
        return False

    lhs_crop = lhs_mask[oy0 - ly0: oy1 - ly0, ox0 - lx0: ox1 - lx0]
    rhs_crop = rhs_mask[oy0 - ry0: oy1 - ry0, ox0 - rx0: ox1 - rx0]
    return bool(np.logical_and(lhs_crop, rhs_crop).any())


def _raw_iou(
    lhs_bbox: tuple[int, int, int, int],
    lhs_mask: np.ndarray,
    rhs_bbox: tuple[int, int, int, int],
    rhs_mask: np.ndarray,
) -> float:
    ly0, lx0, ly1, lx1 = lhs_bbox
    ry0, rx0, ry1, rx1 = rhs_bbox
    oy0, ox0 = max(ly0, ry0), max(lx0, rx0)
    oy1, ox1 = min(ly1, ry1), min(lx1, rx1)
    intersection = 0
    if oy0 < oy1 and ox0 < ox1:
        lhs_crop = lhs_mask[oy0 - ly0: oy1 - ly0, ox0 - lx0: ox1 - lx0]
        rhs_crop = rhs_mask[oy0 - ry0: oy1 - ry0, ox0 - rx0: ox1 - rx0]
        intersection = int(np.logical_and(lhs_crop, rhs_crop).sum())

    union = int(lhs_mask.sum()) + int(rhs_mask.sum()) - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def _overlap_pair(lhs_id: int, rhs_id: int) -> tuple[int, int]:
    return (max(lhs_id, rhs_id), min(lhs_id, rhs_id))


def _generate_node_id(index: int, time: int, max_segments: int) -> int:
    return index + (time + 1) * max_segments


def _node_pickle_ndim(node) -> int:
    if isinstance(node, (bytes, memoryview)):
        node = pickle.loads(bytes(node))
    bbox = np.asarray(node.bbox)
    return len(bbox) // 2


def _make_node_pickle(
    t: int,
    mask_2d: np.ndarray,
    bbox: np.ndarray,
    node_id: int,
    *,
    ndim: int = 2,
) -> bytes:
    from ultrack.core.segmentation.node import Node

    min_y, min_x, max_y, max_x = bbox
    if ndim == 3:
        bbox_arr = np.array(
            [0, int(min_y), int(min_x), 1, int(max_y), int(max_x)],
            dtype=np.int64,
        )
        mask = np.asarray(mask_2d, dtype=bool)[np.newaxis]
    else:
        bbox_arr = np.array(
            [int(min_y), int(min_x), int(max_y), int(max_x)],
            dtype=np.int64,
        )
        mask = np.asarray(mask_2d, dtype=bool)
    node = Node.from_mask(time=t, mask=mask, bbox=bbox_arr, node_id=node_id)
    return pickle.dumps(node)


def _best_iou_assignments(
    records: list[_MaskRecord],
    candidates: dict[int, tuple[tuple[int, int, int, int], np.ndarray, int]],
) -> dict[int, int]:
    pairs: list[tuple[float, int, int, int, int]] = []
    for record_index, record in enumerate(records):
        for candidate_id, (candidate_bbox, candidate_mask, _ndim) in candidates.items():
            iou = _raw_iou(record.bbox, record.mask, candidate_bbox, candidate_mask)
            pairs.append((-iou, record_index, int(candidate_id), record.cell_id, record.t))

    pairs.sort()
    assigned_records: set[int] = set()
    assigned_candidates: set[int] = set()
    assignments: dict[int, int] = {}
    for _neg_iou, record_index, candidate_id, _cell_id, _t in pairs:
        if record_index in assigned_records or candidate_id in assigned_candidates:
            continue
        assignments[record_index] = candidate_id
        assigned_records.add(record_index)
        assigned_candidates.add(candidate_id)
        if len(assigned_records) == min(len(records), len(candidates)):
            break
    return assignments


def inject_validated_nodes(
    working_dir: str | Path,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
    cfg: TrackingConfig,
) -> ValidationInjectionReport:
    """Replace best-matching candidates with validated masks as fixed REAL nodes.

    The best same-frame candidate by raw IoU is updated in place so its
    hierarchy placement and temporal links are preserved. If no candidate is
    available for a validated mask, a reserved REAL node is inserted instead.
    Other candidates in the same frame that overlap a validated mask are marked
    FAKE and paired with the REAL node in OverlapDB.
    """
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, OverlapDB, VarAnnotation

    records, skipped = _validated_mask_records(validated_tracks, tracked_labels)
    if not records:
        return ValidationInjectionReport(
            inserted=0,
            skipped_missing=len(skipped),
            skipped=skipped,
            faked=0,
            overlaps_added=0,
        )

    working_dir = Path(working_dir)
    engine = sqla.create_engine(f"sqlite:///{working_dir / 'data.db'}")
    inserted = 0
    faked_ids: set[int] = set()
    overlap_pairs: set[tuple[int, int]] = set()
    real_node_ids: set[int] = set()

    with Session(engine) as session:
        sample_node = session.query(NodeDB.pickle).limit(1).scalar()
        fallback_ndim = _node_pickle_ndim(sample_node) if sample_node is not None else 2
        next_t_node_id: dict[int, int] = {}
        for t in {record.t for record in records}:
            max_t_node_id = (
                session.query(sqla.func.max(NodeDB.t_node_id))
                .where(NodeDB.t == t)
                .scalar()
            )
            next_t_node_id[t] = int(max_t_node_id or 0) + 1

        records_by_t: dict[int, list[tuple[int, _MaskRecord]]] = {}
        for index, record in enumerate(records):
            records_by_t.setdefault(record.t, []).append((index, record))

        for t, indexed_records in records_by_t.items():
            candidate_rows = (
                session.query(NodeDB.id, NodeDB.pickle)
                .where(NodeDB.t == t)
                .where(NodeDB.t_hier_id != 0)
                .all()
            )
            candidates: dict[int, tuple[tuple[int, int, int, int], np.ndarray, int]] = {}
            for candidate_id, candidate_node in candidate_rows:
                candidate_bbox, candidate_mask = _node_bbox_and_mask(
                    int(candidate_id), candidate_node
                )
                candidates[int(candidate_id)] = (
                    candidate_bbox,
                    candidate_mask,
                    _node_pickle_ndim(candidate_node),
                )

            frame_records = [record for _index, record in indexed_records]
            local_assignments = _best_iou_assignments(frame_records, candidates)
            real_node_by_local_index: dict[int, int] = {}
            matched_candidate_ids: set[int] = set()

            for local_index, record in enumerate(frame_records):
                bbox_arr = np.asarray(record.bbox, dtype=np.int32)
                candidate_id = local_assignments.get(local_index)
                if candidate_id is None:
                    node_ndim = next(
                        (ndim for _bbox, _mask, ndim in candidates.values()),
                        fallback_ndim,
                    )
                    t_node_id = next_t_node_id[record.t]
                    next_t_node_id[record.t] += 1
                    node_id = _generate_node_id(
                        t_node_id, record.t, cfg.max_segments_per_time
                    )
                    session.add(
                        NodeDB(
                            id=node_id,
                            t=record.t,
                            t_node_id=t_node_id,
                            t_hier_id=0,
                            z=0,
                            y=record.y,
                            x=record.x,
                            area=record.area,
                            pickle=_make_node_pickle(
                                record.t,
                                record.mask,
                                bbox_arr,
                                node_id,
                                ndim=node_ndim,
                            ),
                            node_prob=1.0,
                            node_annot=VarAnnotation.REAL,
                        )
                    )
                    inserted += 1
                    real_node_ids.add(node_id)
                    real_node_by_local_index[local_index] = node_id
                    continue

                node_ndim = candidates[candidate_id][2]
                session.query(NodeDB).where(NodeDB.id == candidate_id).update(
                    {
                        NodeDB.y: record.y,
                        NodeDB.x: record.x,
                        NodeDB.area: record.area,
                        NodeDB.pickle: _make_node_pickle(
                            record.t,
                            record.mask,
                            bbox_arr,
                            candidate_id,
                            ndim=node_ndim,
                        ),
                        NodeDB.node_prob: 1.0,
                        NodeDB.node_annot: VarAnnotation.REAL,
                    },
                    synchronize_session=False,
                )
                inserted += 1
                real_node_ids.add(candidate_id)
                matched_candidate_ids.add(candidate_id)
                real_node_by_local_index[local_index] = candidate_id

            for local_index, record in enumerate(frame_records):
                real_node_id = real_node_by_local_index[local_index]
                for candidate_id, (
                    candidate_bbox,
                    candidate_mask,
                    _ndim,
                ) in candidates.items():
                    if candidate_id in matched_candidate_ids:
                        continue
                    if not _intersects(
                        record.bbox, record.mask, candidate_bbox, candidate_mask
                    ):
                        continue
                    faked_ids.add(candidate_id)
                    overlap_pairs.add(_overlap_pair(real_node_id, candidate_id))

        if faked_ids:
            session.query(NodeDB).where(NodeDB.id.in_(faked_ids)).update(
                {NodeDB.node_annot: VarAnnotation.FAKE},
                synchronize_session=False,
            )

        existing_pairs: set[tuple[int, int]] = set()
        if real_node_ids:
            session.query(OverlapDB).where(
                sqla.or_(
                    OverlapDB.node_id.in_(real_node_ids),
                    OverlapDB.ancestor_id.in_(real_node_ids),
                )
            ).delete(synchronize_session=False)
        if overlap_pairs:
            node_ids = {pair[0] for pair in overlap_pairs} | {pair[1] for pair in overlap_pairs}
            existing_pairs = {
                (int(row.node_id), int(row.ancestor_id))
                for row in session.query(OverlapDB)
                .where(
                    sqla.or_(
                        OverlapDB.node_id.in_(node_ids),
                        OverlapDB.ancestor_id.in_(node_ids),
                    )
                )
                .all()
            }
            for node_id, ancestor_id in sorted(overlap_pairs - existing_pairs):
                session.add(OverlapDB(node_id=node_id, ancestor_id=ancestor_id))

        session.commit()

    engine.dispose()
    added_pairs = len(overlap_pairs - existing_pairs)
    return ValidationInjectionReport(
        inserted=inserted,
        skipped_missing=len(skipped),
        skipped=skipped,
        faked=len(faked_ids),
        overlaps_added=added_pairs,
    )


--------------------------------------------------------------------------------
FILE: pyproject.toml (project root)
--------------------------------------------------------------------------------

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "cellflow"
version = "0.2.0"
description = "Fast, interactive, hypothesis-driven cell tracking"
requires-python = ">=3.9"
license = {text = "GPL-3.0"}
authors = [{name = "Artur Ruppel"}]
dependencies = [
    "napari[all]>=0.4.18",
    "qtpy>=2.3.0",
    "h5py",
    "numpy",
    "scipy",
    "scikit-image",
    "pandas",
    "tifffile",
    "numba",
    "matplotlib>=3.9.4",
    "pymaxflow>=1.3.2",
]

[project.entry-points."napari.manifest"]
cellflow = "cellflow:napari.yaml"

[project.scripts]
cellflow-classify-nls = "cellflow.analysis.nls_classification:main"

[tool.hatch.build.targets.wheel]
packages = ["src/cellflow"]

[tool.hatch.build.targets.wheel.shared-data]
"src/cellflow/napari.yaml" = "cellflow/napari.yaml"
