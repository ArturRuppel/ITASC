"""Classify nuclear tracks into a labelled subpopulation by marker intensity.

Some experiments image a nucleus-localised marker (e.g. an NLS reporter) for only
a subpopulation of cells. Aggregating that marker per nuclear *track* yields one
median-intensity scalar per track; thresholding those scalars splits the tracks
into a **positive** subpopulation (above the threshold) and the **negative** rest.

This module is the headless core: it measures per-track medians
(:func:`measure_track_nls_intensity`), proposes a starting threshold
(:func:`auto_threshold`, built on the deterministic two-cluster / Otsu splitters),
classifies against a (possibly hand-tuned) threshold
(:func:`classify_by_threshold`), and patches the columns into a contact-analysis
``.h5`` (:func:`write_nls_classification`). The interactive tuning lives in the
``NLS Classification`` analysis plugin
(:mod:`cellflow.napari.aggregate_quantification.plugins.nls_classification`).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import tifffile
from skimage.filters import threshold_otsu

#: Default audit method recorded in the H5 when a threshold is applied.
METHOD = "threshold_track_median"
#: Status strings written to ``cells/table/nls_status``.
POSITIVE = "positive"
NEGATIVE = "negative"


class NLSClassificationError(ValueError):
    """Raised when track classification would be ambiguous or invalid."""


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
    positive_track_count: int
    negative_track_count: int


def measure_track_nls_intensity(
    nls_zavg: np.ndarray,
    nucleus_labels: np.ndarray,
) -> dict[int, TrackNLSMeasurement]:
    """Measure one median marker-intensity scalar per nonzero nuclear track ID."""
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


def auto_threshold(track_medians: Mapping[int, float]) -> float:
    """Propose a starting threshold separating low- from high-intensity tracks.

    Prefers the two-Gaussian boundary (:func:`split_tracks_two_clusters`); falls
    back to Otsu when the two-cluster fit is degenerate. Raises
    :class:`NLSClassificationError` when no split is possible.
    """
    try:
        threshold, _ = split_tracks_two_clusters(track_medians)
        return threshold
    except NLSClassificationError:
        threshold, _ = split_tracks_otsu(track_medians)
        return threshold


def classify_by_threshold(
    measurements: Mapping[int, TrackNLSMeasurement],
    threshold: float,
) -> dict[int, str]:
    """Split tracks at *threshold*: ``positive`` if ``median > threshold``."""
    return {
        int(track_id): (POSITIVE if item.median_intensity > threshold else NEGATIVE)
        for track_id, item in measurements.items()
    }


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


def patch_position_contact_analysis_nls_classes(
    h5_path: str | Path,
    nls_zavg_path: str | Path,
    nucleus_labels_path: str | Path,
    *,
    threshold: float | None = None,
    positive_label: str = POSITIVE,
    negative_label: str = NEGATIVE,
) -> NLSClassificationSummary:
    """Measure, classify, and patch NLS subpopulation columns into a position H5.

    *threshold* defaults to :func:`auto_threshold`; pass a value (e.g. the
    hand-tuned threshold from the plugin) to override it.
    """
    h5_path = Path(h5_path)
    nls_path = Path(nls_zavg_path)
    labels_path = Path(nucleus_labels_path)

    nls = _read_image_stack(nls_path)
    labels = _read_image_stack(labels_path)
    measurements = measure_track_nls_intensity(nls, labels)
    medians = {track_id: item.median_intensity for track_id, item in measurements.items()}
    resolved_threshold = float(threshold) if threshold is not None else auto_threshold(medians)
    assignments = classify_by_threshold(measurements, resolved_threshold)

    cell_ids = _read_cell_ids(h5_path)
    if not set(int(cid) for cid in cell_ids).intersection(assignments):
        raise NLSClassificationError(
            "H5 cells/table/cell_id values do not overlap any classified nuclear track IDs"
        )

    write_nls_classification(
        h5_path,
        cell_ids=cell_ids,
        measurements=measurements,
        assignments=assignments,
        threshold=resolved_threshold,
        positive_label=positive_label,
        negative_label=negative_label,
        nls_path=nls_path,
        labels_path=labels_path,
    )

    positive = sum(status == POSITIVE for status in assignments.values())
    negative = sum(status == NEGATIVE for status in assignments.values())
    return NLSClassificationSummary(
        h5_path=h5_path,
        threshold=resolved_threshold,
        track_count=len(assignments),
        positive_track_count=positive,
        negative_track_count=negative,
    )


def write_nls_classification(
    path: str | Path,
    *,
    cell_ids: np.ndarray,
    measurements: Mapping[int, TrackNLSMeasurement],
    assignments: Mapping[int, str],
    threshold: float,
    positive_label: str = POSITIVE,
    negative_label: str = NEGATIVE,
    nls_path: str | Path,
    labels_path: str | Path,
) -> None:
    """Write NLS subpopulation columns + audit metadata into ``cells`` of an H5.

    Validates inputs before mutating: shapes and the ``cells/table`` group must be
    present, so a failure leaves the file untouched.
    """
    path = Path(path)
    cell_ids = np.asarray(cell_ids, dtype=np.int64)

    class_labels: list[str] = []
    statuses: list[str] = []
    medians = np.full(len(cell_ids), np.nan, dtype=float)
    pixel_counts = np.zeros(len(cell_ids), dtype=np.int64)
    frame_counts = np.zeros(len(cell_ids), dtype=np.int64)

    for idx, cell_id_value in enumerate(cell_ids):
        cell_id = int(cell_id_value)
        status = assignments.get(cell_id, "")
        statuses.append(status)
        class_labels.append(
            positive_label if status == POSITIVE else negative_label if status == NEGATIVE else ""
        )
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
        positive_count = sum(status == POSITIVE for status in assignments.values())
        negative_count = sum(status == NEGATIVE for status in assignments.values())
        meta.attrs["method"] = METHOD
        meta.attrs["threshold"] = float(threshold)
        meta.attrs["positive_label"] = positive_label
        meta.attrs["negative_label"] = negative_label
        meta.attrs["nls_zavg_path"] = str(nls_path)
        meta.attrs["nucleus_tracked_labels_path"] = str(labels_path)
        meta.attrs["classified_track_count"] = len(assignments)
        meta.attrs["positive_track_count"] = positive_count
        meta.attrs["negative_track_count"] = negative_count
        meta.attrs["created_at"] = datetime.now(timezone.utc).isoformat()


def read_position_cell_ids(h5_path: str | Path) -> np.ndarray:
    """Public accessor for ``cells/table/cell_id`` (used by the plugin on Apply)."""
    return _read_cell_ids(Path(h5_path))


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


def _replace_dataset(group: h5py.Group, name: str, data: np.ndarray, **kwargs) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, **kwargs)
