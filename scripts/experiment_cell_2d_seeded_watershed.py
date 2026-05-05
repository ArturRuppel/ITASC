"""Run a frame-by-frame 2D seeded watershed baseline for tracked cell labels.

Each timepoint is segmented independently. Curated 2D tracked nuclear labels
are hard markers, the binary foreground mask is the domain, and cell contour
maps are the watershed height image. No compactness regularization is used.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import tifffile
from cellflow.segmentation import centroid_markers_from_labels
from skimage.segmentation import watershed


DEFAULT_POS_DIR = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/"
    "analysis/pos00"
)
DEFAULT_CONTOUR_DIR = (
    DEFAULT_POS_DIR
    / "3_cell"
    / "contour_experiment"
    / "20260504-contours-meanz-thr-m5-to-5"
)
DEFAULT_FOREGROUND_MASK = (
    DEFAULT_POS_DIR
    / "3_cell"
    / "contour_experiment"
    / "20260503-232245-thr-8-to-0-maxfg"
    / "foreground_masks.tif"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frame-by-frame 2D seeded watershed baseline for tracked cell segmentation."
    )
    parser.add_argument("--pos-dir", type=Path, default=DEFAULT_POS_DIR)
    parser.add_argument(
        "--contours",
        type=Path,
        default=DEFAULT_CONTOUR_DIR / "contours.tif",
        help="Contour probability/cost stack, expected shape (T, Y, X).",
    )
    parser.add_argument(
        "--foreground-mask",
        type=Path,
        default=DEFAULT_FOREGROUND_MASK,
        help="Binary foreground domain mask, expected shape (T, Y, X).",
    )
    parser.add_argument(
        "--markers",
        type=Path,
        default=DEFAULT_POS_DIR / "2_nucleus" / "tracked_labels.tif",
        help="Curated tracked nuclear labels, expected shape (T, Y, X).",
    )
    parser.add_argument(
        "--timestamp",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Checkpoint directory name under 3_cell/watershed2d_experiment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing timestamped checkpoint directory.",
    )
    parser.add_argument(
        "--no-force-marker-domain",
        action="store_true",
        help="Do not add marker pixels to the foreground domain.",
    )
    parser.add_argument(
        "--seed-mode",
        choices=("centroid", "label"),
        default="centroid",
        help="Use one centroid pixel per nucleus label, or the full label mask, as watershed seeds.",
    )
    return parser.parse_args()


def _load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contours = np.asarray(tifffile.imread(args.contours), dtype=np.float32)
    foreground = np.asarray(tifffile.imread(args.foreground_mask))
    markers = np.asarray(tifffile.imread(args.markers), dtype=np.uint32)

    if contours.ndim == 4 and contours.shape[1] == 1:
        contours = contours[:, 0]
    if foreground.ndim == 4 and foreground.shape[1] == 1:
        foreground = foreground[:, 0]
    if markers.ndim == 4 and markers.shape[1] == 1:
        markers = markers[:, 0]

    if contours.ndim != 3:
        raise ValueError(f"Expected contours as (T, Y, X), got {contours.shape}")
    if foreground.shape != contours.shape:
        raise ValueError(
            f"Foreground mask shape {foreground.shape} does not match contours {contours.shape}"
        )
    if markers.shape != contours.shape:
        raise ValueError(
            f"Marker shape {markers.shape} does not match contours {contours.shape}"
        )
    return contours, foreground, markers


def _summarize_labels(labels: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    marker_ids = np.unique(markers)
    marker_ids = marker_ids[marker_ids != 0]
    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids != 0]
    missing_ids = np.setdiff1d(marker_ids, label_ids)
    extra_ids = np.setdiff1d(label_ids, marker_ids)
    unlabeled_foreground = mask & (labels == 0)
    return {
        "n_marker_ids": int(marker_ids.size),
        "n_output_ids": int(label_ids.size),
        "missing_marker_ids": [int(v) for v in missing_ids[:50]],
        "n_missing_marker_ids": int(missing_ids.size),
        "extra_output_ids": [int(v) for v in extra_ids[:50]],
        "n_extra_output_ids": int(extra_ids.size),
        "foreground_voxels": int(np.count_nonzero(mask)),
        "labeled_voxels": int(np.count_nonzero(labels)),
        "unlabeled_foreground_voxels": int(np.count_nonzero(unlabeled_foreground)),
        "max_label": int(labels.max()) if labels.size else 0,
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.pos_dir / "3_cell" / "watershed2d_experiment" / args.timestamp
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite or use a new --timestamp")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading inputs...", flush=True)
    contours, foreground_raw, full_markers = _load_inputs(args)
    markers = (
        centroid_markers_from_labels(full_markers)
        if args.seed_mode == "centroid"
        else full_markers
    )
    marker_mask = markers > 0
    foreground = foreground_raw > 0
    added_marker_voxels = int(np.count_nonzero(marker_mask & ~foreground))
    effective_mask = foreground | marker_mask
    if args.no_force_marker_domain:
        effective_mask = foreground

    params = {
        "script": str(Path(__file__).resolve()),
        "pos_dir": args.pos_dir,
        "contours": args.contours,
        "foreground_mask": args.foreground_mask,
        "markers": args.markers,
        "shape": tuple(int(v) for v in contours.shape),
        "contours_dtype": str(contours.dtype),
        "foreground_dtype": str(foreground_raw.dtype),
        "markers_dtype": str(full_markers.dtype),
        "seed_markers_dtype": str(markers.dtype),
        "seed_mode": args.seed_mode,
        "full_marker_voxels": int(np.count_nonzero(full_markers)),
        "seed_marker_voxels": int(np.count_nonzero(markers)),
        "foreground_binary_values": [int(v) for v in np.unique(foreground_raw)[:16]],
        "force_marker_domain": not args.no_force_marker_domain,
        "added_marker_voxels": added_marker_voxels,
        "compactness": 0.0,
        "watershed_line": False,
        "mode": "frame_by_frame_2d",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": output_dir,
    }
    _write_json(output_dir / "parameters.json", params)

    tifffile.imwrite(
        output_dir / "effective_foreground_mask.tif",
        effective_mask.astype(np.uint8),
        compression="zlib",
    )
    tifffile.imwrite(output_dir / "seed_markers.tif", markers, compression="zlib")

    labels = np.zeros_like(markers, dtype=np.uint32)
    frame_summaries = []
    print(f"Running 2D watershed for {contours.shape[0]} frames...", flush=True)
    t0_all = perf_counter()
    for t in range(contours.shape[0]):
        t0 = perf_counter()
        frame_labels = watershed(
            contours[t],
            markers=markers[t].astype(np.int32, copy=False),
            mask=effective_mask[t],
            compactness=0.0,
            watershed_line=False,
        )
        labels[t] = np.asarray(frame_labels, dtype=np.uint32)
        elapsed_s = perf_counter() - t0
        frame_summary = {
            "t": int(t),
            "elapsed_s": round(elapsed_s, 3),
            **_summarize_labels(labels[t], markers[t], effective_mask[t]),
        }
        frame_summaries.append(frame_summary)
        print(
            f"  frame {t + 1}/{contours.shape[0]}: "
            f"{frame_summary['n_output_ids']} IDs, "
            f"{frame_summary['n_missing_marker_ids']} missing marker IDs, "
            f"{elapsed_s:.2f}s",
            flush=True,
        )

    label_path = output_dir / "tracked_labels.tif"
    tifffile.imwrite(label_path, labels, compression="zlib")
    summary = {
        "elapsed_s": round(perf_counter() - t0_all, 3),
        "path": label_path,
        **_summarize_labels(labels, markers, effective_mask),
    }
    params["finished_at"] = datetime.now().isoformat(timespec="seconds")
    params["summary"] = summary
    _write_json(output_dir / "parameters.json", params)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "frame_summaries.json", {"frames": frame_summaries})
    print(
        f"wrote {label_path.name}: {summary['n_output_ids']} IDs, "
        f"{summary['n_missing_marker_ids']} missing marker IDs, "
        f"{summary['unlabeled_foreground_voxels']} unlabeled fg voxels",
        flush=True,
    )
    print("Done.", flush=True)
    print(f"Output directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
