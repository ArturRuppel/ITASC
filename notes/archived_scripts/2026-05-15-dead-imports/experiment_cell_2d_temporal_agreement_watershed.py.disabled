"""Run a 2D seeded watershed on temporally agreement-filtered contour maps.

Instead of blending or smoothing across time, this experiment uses neighbor
frames to modulate contour confidence at each pixel:

- **max_agreement**: output = (1-α)·contour_t + α·max(contour_{t-1}, contour_{t+1})
  A boundary present in either neighbor is boosted in the current frame; a
  spurious boundary absent from both neighbors is suppressed.

- **asymmetric**: output = contour_t + boost·(neighbor_max - contour_t)_+
                              - suppress·(contour_t - neighbor_max)_+
  Allows independent control of how aggressively missing boundaries are
  recovered vs spurious boundaries removed.

The filtered contour stack is checkpointed alongside the watershed output.
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

PARAM_SETS: list[dict[str, Any]] = [
    # Baseline: no filtering (equivalent to the plain 2D watershed baseline)
    {"method": "max_agreement", "alpha": 0.0},
    # Light neighbour influence — mostly original, slight flicker reduction
    {"method": "max_agreement", "alpha": 0.2},
    {"method": "max_agreement", "alpha": 0.35},
    # Equal weight to current and neighbours
    {"method": "max_agreement", "alpha": 0.5},
    # Heavy neighbour influence — neighbours dominate
    {"method": "max_agreement", "alpha": 0.65},
    {"method": "max_agreement", "alpha": 0.8},
    # Pure neighbour max — current frame ignored entirely
    {"method": "max_agreement", "alpha": 1.0},
    # Asymmetric: recover missing boundaries more than suppress spurious ones
    {"method": "asymmetric", "boost": 0.5, "suppress": 0.2},
    {"method": "asymmetric", "boost": 0.8, "suppress": 0.3},
    {"method": "asymmetric", "boost": 0.3, "suppress": 0.5},
    {"method": "asymmetric", "boost": 0.5, "suppress": 0.5},
]


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
        description="Temporal agreement-filtered 2D seeded watershed sweep."
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
        help="Checkpoint directory name prefix.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing timestamped checkpoint directories.",
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


def _summarize_labels(
    labels: np.ndarray, markers: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
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


def _apply_max_agreement(contours: np.ndarray, alpha: float) -> np.ndarray:
    """Replace each frame with (1-α)·contour_t + α·max(contour_{t-1}, contour_{t+1}).

    Edge frames (t=0, t=n-1) use the single available neighbour.
    """
    n_frames = contours.shape[0]
    filtered = contours.copy()
    for t in range(n_frames):
        prev = contours[max(0, t - 1)]
        next_ = contours[min(n_frames - 1, t + 1)]
        neighbour_max = np.maximum(prev, next_)
        filtered[t] = (1.0 - alpha) * contours[t] + alpha * neighbour_max
    return np.clip(filtered, 0.0, 1.0).astype(np.float32)


def _apply_asymmetric(
    contours: np.ndarray, boost: float, suppress: float
) -> np.ndarray:
    """Boost boundaries present in neighbours but missing in current.
    Suppress boundaries present in current but absent from neighbours.

    output = contour_t + boost·(neighbour_max - contour_t)_+
                      - suppress·(contour_t - neighbour_max)_+
    """
    n_frames = contours.shape[0]
    filtered = contours.copy()
    for t in range(n_frames):
        prev = contours[max(0, t - 1)]
        next_ = contours[min(n_frames - 1, t + 1)]
        neighbour_max = np.maximum(prev, next_)
        boost_amount = boost * np.maximum(0, neighbour_max - contours[t])
        suppress_amount = suppress * np.maximum(0, contours[t] - neighbour_max)
        filtered[t] = contours[t] + boost_amount - suppress_amount
    return np.clip(filtered, 0.0, 1.0).astype(np.float32)


def main() -> None:
    args = _parse_args()
    base_dir = args.pos_dir / "3_cell" / "temporal_agreement_watershed_experiment"

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

    n_frames = contours.shape[0]
    all_run_summaries: list[dict[str, Any]] = []

    for idx, ps in enumerate(PARAM_SETS, start=1):
        method = ps["method"]
        if method == "max_agreement":
            alpha = ps["alpha"]
            label = f"max_agreement-alpha{alpha}"
            run_dir = base_dir / f"{args.timestamp}-{label}"
        else:
            boost = ps["boost"]
            suppress = ps["suppress"]
            label = f"asymmetric-b{boost}-s{suppress}"
            run_dir = base_dir / f"{args.timestamp}-{label}"

        if run_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"{run_dir} exists; pass --overwrite or use a new --timestamp"
            )
        run_dir.mkdir(parents=True, exist_ok=True)

        header = f"=== [{idx}/{len(PARAM_SETS)}] {label} ==="
        print(header, flush=True)

        t0_run = perf_counter()

        print("  applying temporal agreement filter...", flush=True)
        if method == "max_agreement":
            filtered = _apply_max_agreement(contours, float(alpha))
        else:
            filtered = _apply_asymmetric(
                contours, float(boost), float(suppress)
            )

        tifffile.imwrite(
            run_dir / "filtered_contours.tif", filtered, compression="zlib"
        )

        labels = np.zeros_like(markers, dtype=np.uint32)
        frame_summaries = []
        for t in range(n_frames):
            t0 = perf_counter()
            frame_labels = watershed(
                filtered[t],
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
                f"  frame {t + 1}/{n_frames}: "
                f"{frame_summary['n_output_ids']} IDs, "
                f"{frame_summary['n_missing_marker_ids']} missing marker IDs, "
                f"{elapsed_s:.2f}s",
                flush=True,
            )

        label_path = run_dir / "tracked_labels.tif"
        tifffile.imwrite(label_path, labels, compression="zlib")

        elapsed_run = round(perf_counter() - t0_run, 3)
        summary = {
            "elapsed_s": elapsed_run,
            "path": label_path,
            **_summarize_labels(labels, markers, effective_mask),
        }

        params: dict[str, Any] = {
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
            "force_marker_domain": True,
            "added_marker_voxels": added_marker_voxels,
            "compactness": 0.0,
            "watershed_line": False,
            "mode": "frame_by_frame_2d",
            "filter_method": method,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": run_dir,
            "filtered_contours_path": str(run_dir / "filtered_contours.tif"),
        }
        if method == "max_agreement":
            params["alpha"] = float(alpha)
        else:
            params["boost"] = float(boost)
            params["suppress"] = float(suppress)

        params["finished_at"] = datetime.now().isoformat(timespec="seconds")
        params["summary"] = summary
        _write_json(run_dir / "parameters.json", params)
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "frame_summaries.json", {"frames": frame_summaries})

        run_info = {
            "label": label,
            "n_output_ids": summary["n_output_ids"],
            "n_missing_marker_ids": summary["n_missing_marker_ids"],
            "unlabeled_fg_voxels": summary["unlabeled_foreground_voxels"],
            "elapsed_s": elapsed_run,
        }
        all_run_summaries.append(run_info)

    print()
    print(
        f"{'label':>32s}  {'n_out':>6s}  {'n_miss':>6s}  "
        f"{'unlab_fg':>9s}  {'elap_s':>8s}"
    )
    print("-" * 72)
    for r in all_run_summaries:
        print(
            f"{r['label']:>32s}  "
            f"{r['n_output_ids']:>6d}  {r['n_missing_marker_ids']:>6d}  "
            f"{r['unlabeled_fg_voxels']:>9d}  {r['elapsed_s']:>8.1f}"
        )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
