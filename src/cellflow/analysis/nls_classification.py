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
METHOD = "otsu_track_median"


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
    threshold, assignments = split_tracks_otsu(medians)
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
