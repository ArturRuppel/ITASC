"""Build experimental cell contour maps from Cellpose probability and flow files.

Default target:
  /home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00

Outputs are checkpointed under:
  3_cell/contour_experiment/<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cellflow.database.hypotheses import normalize_seeded_watershed_dp_stack
from cellflow.segmentation import build_consensus_boundary


DEFAULT_POS_DIR = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/"
    "analysis/pos00"
)
DEFAULT_THRESHOLDS = [float(v) for v in range(-8, 1)]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _shape_from_tiff(path: Path) -> tuple[tuple[int, ...], str]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return tuple(int(v) for v in series.shape), str(series.dtype)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build checkpointed experimental CellFlow cell contour maps."
    )
    parser.add_argument("--pos-dir", type=Path, default=DEFAULT_POS_DIR)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Cellpose cellprob thresholds to average.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional bounded test run. Default processes every timepoint.",
    )
    parser.add_argument(
        "--timestamp",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Checkpoint directory name under 3_cell/contour_experiment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing timestamped checkpoint directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pos_dir = args.pos_dir
    cellpose_dir = pos_dir / "1_cellpose"
    prob_path = cellpose_dir / "cell_prob_3dt.tif"
    dp_path = cellpose_dir / "cell_dp_3dt.tif"
    output_dir = pos_dir / "3_cell" / "contour_experiment" / args.timestamp
    frame_dir = output_dir / "frames"
    contour_path = output_dir / "contours.tif"
    foreground_path = output_dir / "foreground.tif"
    params_path = output_dir / "parameters.json"
    summary_path = output_dir / "frame_summaries.jsonl"

    if not prob_path.exists():
        raise FileNotFoundError(prob_path)
    if not dp_path.exists():
        raise FileNotFoundError(dp_path)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite or use a new --timestamp")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    prob_shape_meta, prob_dtype_meta = _shape_from_tiff(prob_path)
    dp_shape_meta, dp_dtype_meta = _shape_from_tiff(dp_path)
    params = {
        "script": str(Path(__file__).resolve()),
        "pos_dir": pos_dir,
        "prob_path": prob_path,
        "dp_path": dp_path,
        "prob_shape_metadata": prob_shape_meta,
        "prob_dtype_metadata": prob_dtype_meta,
        "dp_shape_metadata": dp_shape_meta,
        "dp_dtype_metadata": dp_dtype_meta,
        "cellprob_thresholds": [float(v) for v in args.thresholds],
        "gamma": 1.0,
        "flow_threshold": 0.0,
        "niter": 200,
        "do_3D": False,
        "z_slice_policy": "all z-slices",
        "foreground_projection": "max_z_sigmoid_cellprob",
        "max_frames": args.max_frames,
        "output_dir": output_dir,
        "contours_path": contour_path,
        "foreground_path": foreground_path,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(params_path, params)

    print(f"Loading probability stack: {prob_path}", flush=True)
    prob_stack = np.asarray(tifffile.imread(prob_path), dtype=np.float32)
    if prob_stack.ndim == 3:
        prob_stack = prob_stack[np.newaxis]
    if prob_stack.ndim != 4:
        raise ValueError(f"Expected prob stack as (T, Z, Y, X), got {prob_stack.shape}")

    print(f"Loading flow stack: {dp_path}", flush=True)
    dp_raw = np.asarray(tifffile.imread(dp_path), dtype=np.float32)
    dp_stack = normalize_seeded_watershed_dp_stack(dp_raw, prob_stack.shape)
    if dp_stack.shape[2] < 2:
        raise ValueError(f"Expected at least two flow channels after normalization, got {dp_stack.shape}")
    dp_stack = dp_stack[:, :, :2]

    n_t, n_z, n_y, n_x = prob_stack.shape
    n_frames = n_t if args.max_frames is None else min(n_t, int(args.max_frames))
    if n_frames <= 0:
        raise ValueError(f"Nothing to process: n_t={n_t}, max_frames={args.max_frames}")

    params.update(
        {
            "prob_shape_loaded": tuple(int(v) for v in prob_stack.shape),
            "dp_shape_loaded": tuple(int(v) for v in dp_stack.shape),
            "processed_timepoints": n_frames,
            "total_timepoints": n_t,
            "processed_z_slices_per_timepoint": n_z,
            "image_shape_yx": (n_y, n_x),
        }
    )
    _write_json(params_path, params)

    contour_frames: list[np.ndarray] = []
    foreground_frames: list[np.ndarray] = []
    print(
        f"Building contours for {n_frames}/{n_t} timepoints, {n_z} z-slices, "
        f"thresholds={params['cellprob_thresholds']}",
        flush=True,
    )

    with summary_path.open("w", encoding="utf-8") as summary_file:
        for t in range(n_frames):
            t0 = perf_counter()
            threshold_summaries: list[dict[str, Any]] = []

            def mask_callback(masks_zyx: np.ndarray, thresh_idx: int) -> None:
                labels_per_z = [int(np.max(masks_zyx[z])) for z in range(masks_zyx.shape[0])]
                threshold_summaries.append(
                    {
                        "threshold": float(args.thresholds[thresh_idx]),
                        "labels_per_z": labels_per_z,
                        "labels_total_across_z": int(sum(labels_per_z)),
                    }
                )

            print(f"Frame {t + 1}/{n_frames}: building consensus boundary...", flush=True)
            boundary, _foreground_average = build_consensus_boundary(
                prob_stack[t],
                dp_stack[t],
                [float(v) for v in args.thresholds],
                gamma=1.0,
                mask_callback=mask_callback,
            )
            boundary = np.asarray(boundary, dtype=np.float32)
            foreground = np.max(
                1.0 / (1.0 + np.exp(-prob_stack[t].astype(np.float32))),
                axis=0,
            )
            foreground = np.asarray(foreground, dtype=np.float32)

            frame_contour_path = frame_dir / f"contours_t{t:04d}.tif"
            frame_foreground_path = frame_dir / f"foreground_t{t:04d}.tif"
            tifffile.imwrite(frame_contour_path, boundary, compression="zlib")
            tifffile.imwrite(frame_foreground_path, foreground, compression="zlib")
            contour_frames.append(boundary)
            foreground_frames.append(foreground)

            elapsed_s = perf_counter() - t0
            summary = {
                "t": t,
                "elapsed_s": round(elapsed_s, 3),
                "boundary_min": float(boundary.min()),
                "boundary_max": float(boundary.max()),
                "boundary_mean": float(boundary.mean()),
                "foreground_min": float(foreground.min()),
                "foreground_max": float(foreground.max()),
                "foreground_mean": float(foreground.mean()),
                "threshold_summaries": threshold_summaries,
                "contour_checkpoint": frame_contour_path,
                "foreground_checkpoint": frame_foreground_path,
            }
            summary_file.write(json.dumps(summary, default=_json_default) + "\n")
            summary_file.flush()
            print(
                f"Frame {t + 1}/{n_frames}: boundary range "
                f"[{summary['boundary_min']:.4f}, {summary['boundary_max']:.4f}] "
                f"in {elapsed_s:.1f}s",
                flush=True,
            )

    print(f"Writing final contour stack: {contour_path}", flush=True)
    tifffile.imwrite(contour_path, np.stack(contour_frames), compression="zlib")
    print(f"Writing foreground diagnostic stack: {foreground_path}", flush=True)
    tifffile.imwrite(foreground_path, np.stack(foreground_frames), compression="zlib")

    params["finished_at"] = datetime.now().isoformat(timespec="seconds")
    params["contours_shape"] = tuple(int(v) for v in np.stack(contour_frames).shape)
    params["foreground_shape"] = tuple(int(v) for v in np.stack(foreground_frames).shape)
    _write_json(params_path, params)
    print("Done.", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Contours: {contour_path}", flush=True)


if __name__ == "__main__":
    main()
