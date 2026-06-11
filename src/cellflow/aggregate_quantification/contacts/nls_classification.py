"""Classify nuclear tracks into a labelled subpopulation by marker intensity.

Some experiments image a nucleus-localised marker (e.g. an NLS reporter) for only
a subpopulation of cells. Aggregating that marker per nuclear *track* yields one
intensity scalar per track; thresholding those scalars splits the tracks
into a **positive** subpopulation (above the threshold) and the **negative** rest.

The per-track scalar is the **90th percentile of the per-frame median intensities**
(one median over each frame's nucleus pixels). The per-frame median ignores
brightness on only part of a nucleus, so a negative cell brushing past a positive
one is not misread as positive; the cross-frame 90th percentile keeps the dim
frames where a nucleus is dividing/dispersed from dragging a genuinely positive
track down.

This module is the headless core: it measures per-track intensities
(:func:`measure_track_nls_intensity`), proposes a starting threshold
(:func:`auto_threshold`, built on the deterministic two-cluster / Otsu splitters),
classifies against a (possibly hand-tuned) threshold
(:func:`classify_by_threshold`), and writes the result as a two-column
``id,label`` **sidecar CSV** (:func:`write_nls_classification_csv`) — it never
touches the contact-analysis ``.h5``, which stays a pure, regenerable build
artifact. The CSV is the single source of truth for the subpopulation label;
consumers join it by ``cell_id`` via :func:`read_nls_classification_csv`. The
interactive tuning lives in the ``NLS Classification`` analysis plugin
(:mod:`cellflow.napari.aggregate_quantification.plugins.nls_classification`).
"""
from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import threshold_otsu

from cellflow.aggregate_quantification.quantifier import OUTPUT_SUBDIR

#: Percentile (over per-frame median intensities) used to summarise each track.
INTENSITY_PERCENTILE = 90.0
#: Internal status strings produced by :func:`classify_by_threshold`; mapped to
#: the caller's positive/negative label strings when the sidecar CSV is written.
POSITIVE = "positive"
NEGATIVE = "negative"
#: Sidecar CSV file name, written into each position's :data:`OUTPUT_SUBDIR`.
CSV_NAME = "nls_classification.csv"


class NLSClassificationError(ValueError):
    """Raised when track classification would be ambiguous or invalid."""


@dataclass(frozen=True)
class TrackNLSMeasurement:
    #: 90th percentile of the track's per-frame median nucleus intensities.
    intensity: float
    pixel_count: int
    frame_count: int


@dataclass(frozen=True)
class NLSClassificationSummary:
    csv_path: Path
    threshold: float
    track_count: int
    positive_track_count: int
    negative_track_count: int


def measure_track_nls_intensity(
    nls_zavg: np.ndarray,
    nucleus_labels: np.ndarray,
    *,
    percentile: float = INTENSITY_PERCENTILE,
) -> dict[int, TrackNLSMeasurement]:
    """Measure one marker-intensity scalar per nonzero nuclear track ID.

    For each track the scalar is the *percentile* (default the 90th) of its
    per-frame **median** nucleus intensities — one median over the nucleus pixels
    in each frame the track appears in. Two robustness properties, one per axis:

    * the **per-frame median** ignores brightness on only part of a nucleus, so a
      negative cell that brushes past a positive one (PSF bleed / a few of the
      neighbour's pixels caught in the mask) does not register as positive;
    * the **cross-frame 90th percentile** keeps the dim frames where a nucleus is
      dividing/dispersed from pulling a genuinely positive track down.
    """
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
        frames_present = np.nonzero(np.any(mask, axis=(1, 2)))[0]
        if frames_present.size == 0:
            continue
        per_frame_medians = np.asarray(
            [float(np.median(nls[frame][mask[frame]])) for frame in frames_present],
            dtype=float,
        )
        measurements[track_id] = TrackNLSMeasurement(
            intensity=float(np.percentile(per_frame_medians, percentile)),
            pixel_count=int(mask.sum()),
            frame_count=int(frames_present.size),
        )
    return measurements


def auto_threshold(track_intensities: Mapping[int, float]) -> float:
    """Propose a starting threshold separating low- from high-intensity tracks.

    Prefers the two-Gaussian boundary (:func:`split_tracks_two_clusters`); falls
    back to Otsu when the two-cluster fit is degenerate. Raises
    :class:`NLSClassificationError` when no split is possible.
    """
    try:
        threshold, _ = split_tracks_two_clusters(track_intensities)
        return threshold
    except NLSClassificationError:
        threshold, _ = split_tracks_otsu(track_intensities)
        return threshold


def classify_by_threshold(
    measurements: Mapping[int, TrackNLSMeasurement],
    threshold: float,
) -> dict[int, str]:
    """Split tracks at *threshold*: ``positive`` if ``median > threshold``."""
    return {
        int(track_id): (POSITIVE if item.intensity > threshold else NEGATIVE)
        for track_id, item in measurements.items()
    }


def split_tracks_otsu(track_intensities: Mapping[int, float]) -> tuple[float, dict[int, str]]:
    """Split per-track intensities into deterministic high/low statuses."""
    if len(track_intensities) < 2:
        raise NLSClassificationError("Cannot classify fewer than two nonzero tracks with sampled pixels")

    ordered_ids = sorted(int(track_id) for track_id in track_intensities)
    values = np.asarray([float(track_intensities[track_id]) for track_id in ordered_ids], dtype=float)
    if np.any(~np.isfinite(values)):
        raise NLSClassificationError("Track median intensities must be finite for Otsu classification")
    if np.all(values == values[0]):
        raise NLSClassificationError("Cannot classify tracks because all track intensity scalars are identical")

    threshold = float(threshold_otsu(values))
    assignments = {
        track_id: ("high" if float(track_intensities[track_id]) > threshold else "low")
        for track_id in ordered_ids
    }
    high_count = sum(status == "high" for status in assignments.values())
    low_count = sum(status == "low" for status in assignments.values())
    if high_count == 0 or low_count == 0:
        raise NLSClassificationError(
            "Otsu threshold assigned all tracks to one group; refusing to write classifications"
        )
    return threshold, assignments


def split_tracks_two_clusters(track_intensities: Mapping[int, float]) -> tuple[float, dict[int, str]]:
    """Split per-track intensities into deterministic two-Gaussian classes."""
    if len(track_intensities) < 2:
        raise NLSClassificationError("Cannot classify fewer than two nonzero tracks with sampled pixels")

    ordered_items = sorted(
        ((int(track_id), float(intensity)) for track_id, intensity in track_intensities.items()),
        key=lambda item: (item[1], item[0]),
    )
    ordered_ids = [track_id for track_id, _ in ordered_items]
    values = np.asarray([intensity for _, intensity in ordered_items], dtype=float)
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


def nls_classification_csv_path(position_path: str | Path) -> Path:
    """The position's NLS sidecar CSV path.

    ``<position_path>/<OUTPUT_SUBDIR>/nls_classification.csv`` — the single home
    used by the plugin (write), the shape plugin (join), and the contact
    visualizer (colour-by-label).
    """
    return Path(position_path) / OUTPUT_SUBDIR / CSV_NAME


def classify_position_nls_to_csv(
    csv_path: str | Path,
    nls_zavg_path: str | Path,
    nucleus_labels_path: str | Path,
    *,
    threshold: float | None = None,
    positive_label: str = POSITIVE,
    negative_label: str = NEGATIVE,
) -> NLSClassificationSummary:
    """Measure, classify, and write the NLS sidecar CSV for one position.

    Reads only the NLS image + nucleus labels — never an ``.h5``. *threshold*
    defaults to :func:`auto_threshold`; pass a value (e.g. the hand-tuned
    threshold from the plugin) to override it.
    """
    csv_path = Path(csv_path)
    nls = _read_image_stack(Path(nls_zavg_path))
    labels = _read_image_stack(Path(nucleus_labels_path))
    measurements = measure_track_nls_intensity(nls, labels)
    intensities = {track_id: item.intensity for track_id, item in measurements.items()}
    resolved_threshold = float(threshold) if threshold is not None else auto_threshold(intensities)
    assignments = classify_by_threshold(measurements, resolved_threshold)

    write_nls_classification_csv(
        csv_path,
        assignments,
        positive_label=positive_label,
        negative_label=negative_label,
    )

    positive = sum(status == POSITIVE for status in assignments.values())
    negative = sum(status == NEGATIVE for status in assignments.values())
    return NLSClassificationSummary(
        csv_path=csv_path,
        threshold=resolved_threshold,
        track_count=len(assignments),
        positive_track_count=positive,
        negative_track_count=negative,
    )


def write_nls_classification_csv(
    path: str | Path,
    assignments: Mapping[int, str],
    *,
    positive_label: str = POSITIVE,
    negative_label: str = NEGATIVE,
) -> None:
    """Write the two-column ``id,label`` sidecar CSV — one row per classified track.

    *assignments* maps each track id to a :data:`POSITIVE` / :data:`NEGATIVE`
    status; the row's ``label`` is the caller's *positive_label* / *negative_label*
    string. The parent folder is created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        for track_id in sorted(assignments, key=int):
            label = positive_label if assignments[track_id] == POSITIVE else negative_label
            writer.writerow([int(track_id), label])


def read_nls_classification_csv(path: str | Path) -> dict[int, str]:
    """Read an ``id,label`` sidecar CSV back into a ``cell_id -> label`` map."""
    path = Path(path)
    labels: dict[int, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels[int(row["id"])] = str(row["label"])
    return labels


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
