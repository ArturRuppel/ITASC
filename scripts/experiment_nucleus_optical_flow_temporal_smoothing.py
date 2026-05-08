"""Optical-flow temporal smoothing for nucleus contour and foreground maps.

This is an experiment script, not production pipeline code. It estimates dense
Farneback optical flow from `1_cellpose/nucleus_prob_zavg.tif`, then warps the
neighboring `2_nucleus/foreground_scores.tif` and `2_nucleus/contour_maps.tif`
into each frame before blending them with the native frame.

Outputs are written to a timestamped directory under:

    <pos_dir>/2_nucleus/optical_flow_temporal_experiment/

The original contour and foreground files are not modified.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import tifffile


DEFAULT_POS_DIR = Path(
    "/home/aruppel/Data/"
    "2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/"
    "analysis/pos02"
)

FLOW_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 63,
    "iterations": 3,
    "poly_n": 7,
    "poly_sigma": 1.5,
    "flags": 0,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _read_stack(path: Path, dtype: np.dtype | type) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.asarray(tifffile.imread(path), dtype=dtype)
    if arr.ndim != 3:
        raise ValueError(f"Expected a (T, Y, X) stack at {path}, got {arr.shape}")
    return arr


def _frame_to_u8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    lo = float(np.nanmin(frame))
    hi = float(np.nanmax(frame))
    if hi <= lo:
        return np.zeros(frame.shape, dtype=np.uint8)
    scaled = (frame - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def compute_forward_flows(prob_zavg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return dense flow and magnitude for consecutive frame pairs.

    `flows[t]` maps coordinates from frame `t` to frame `t + 1`, with channels
    `(dx, dy)`.
    """
    n_frames, height, width = prob_zavg.shape
    flows = np.zeros((max(n_frames - 1, 0), height, width, 2), dtype=np.float32)
    magnitudes = np.zeros((max(n_frames - 1, 0), height, width), dtype=np.float32)

    for t in range(n_frames - 1):
        prev = _frame_to_u8(prob_zavg[t])
        nxt = _frame_to_u8(prob_zavg[t + 1])
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, **FLOW_PARAMS)
        flows[t] = flow.astype(np.float32, copy=False)
        magnitudes[t] = cv2.cartToPolar(flow[..., 0], flow[..., 1])[0]
        print(
            f"  flow {t + 1:03d}/{n_frames - 1:03d}: "
            f"mean_mag={float(np.mean(magnitudes[t])):.3f}, "
            f"p95_mag={float(np.percentile(magnitudes[t], 95)):.3f}",
            flush=True,
        )

    return flows, magnitudes


def _warp_previous_to_current(
    previous: np.ndarray,
    flow_prev_to_current: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    dx = flow_prev_to_current[..., 0]
    dy = flow_prev_to_current[..., 1]
    return cv2.remap(
        previous.astype(np.float32, copy=False),
        x_grid - dx,
        y_grid - dy,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _warp_next_to_current(
    next_frame: np.ndarray,
    flow_current_to_next: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    dx = flow_current_to_next[..., 0]
    dy = flow_current_to_next[..., 1]
    return cv2.remap(
        next_frame.astype(np.float32, copy=False),
        x_grid + dx,
        y_grid + dy,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def smooth_stack_bidirectional(
    stack: np.ndarray,
    flows: np.ndarray,
    *,
    native_weight: float,
    prev_weight: float,
    next_weight: float,
) -> np.ndarray:
    """Blend each frame with motion-compensated previous and next frames."""
    n_frames, height, width = stack.shape
    y_grid, x_grid = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    out = np.zeros_like(stack, dtype=np.float32)

    for t in range(n_frames):
        blended = native_weight * stack[t].astype(np.float32, copy=False)
        total_weight = native_weight

        if t > 0:
            warped_prev = _warp_previous_to_current(
                stack[t - 1], flows[t - 1], x_grid, y_grid
            )
            blended += prev_weight * warped_prev
            total_weight += prev_weight

        if t < n_frames - 1:
            warped_next = _warp_next_to_current(stack[t + 1], flows[t], x_grid, y_grid)
            blended += next_weight * warped_next
            total_weight += next_weight

        out[t] = blended / total_weight
        print(f"  smoothed frame {t + 1:03d}/{n_frames:03d}", flush=True)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _summarize_stack(name: str, original: np.ndarray, smoothed: np.ndarray) -> dict[str, Any]:
    delta = smoothed.astype(np.float32) - original.astype(np.float32)
    return {
        "name": name,
        "original_min": float(np.min(original)),
        "original_max": float(np.max(original)),
        "smoothed_min": float(np.min(smoothed)),
        "smoothed_max": float(np.max(smoothed)),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
        "p95_abs_delta": float(np.percentile(np.abs(delta), 95)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos-dir", type=Path, default=DEFAULT_POS_DIR)
    parser.add_argument("--native-weight", type=float, default=0.6)
    parser.add_argument("--prev-weight", type=float, default=0.2)
    parser.add_argument("--next-weight", type=float, default=0.2)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument(
        "--write-flow",
        action="store_true",
        help="Also write dense flow_xy.tif. This is large but useful for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pos_dir = args.pos_dir
    nucleus_dir = pos_dir / "2_nucleus"

    prob_path = pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif"
    contour_path = nucleus_dir / "contour_maps.tif"
    foreground_score_path = nucleus_dir / "foreground_scores.tif"

    print(f"Loading {pos_dir}", flush=True)
    prob = _read_stack(prob_path, np.float32)
    contours = _read_stack(contour_path, np.float32)
    foreground_scores = _read_stack(foreground_score_path, np.float32)

    if not (prob.shape == contours.shape == foreground_scores.shape):
        raise ValueError(
            "Input stack shapes do not match: "
            f"prob={prob.shape}, contours={contours.shape}, "
            f"foreground_scores={foreground_scores.shape}"
        )

    if args.native_weight <= 0:
        raise ValueError("--native-weight must be > 0")
    if args.prev_weight < 0 or args.next_weight < 0:
        raise ValueError("--prev-weight and --next-weight must be >= 0")
    if not (0.0 <= args.foreground_threshold <= 1.0):
        raise ValueError("--foreground-threshold must be in [0, 1]")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = nucleus_dir / "optical_flow_temporal_experiment" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = perf_counter()
    print("Computing optical flow from nucleus z-avg probability...", flush=True)
    flows, flow_magnitude = compute_forward_flows(prob)

    print("Smoothing contour maps...", flush=True)
    smoothed_contours = smooth_stack_bidirectional(
        contours,
        flows,
        native_weight=args.native_weight,
        prev_weight=args.prev_weight,
        next_weight=args.next_weight,
    )

    print("Smoothing foreground scores...", flush=True)
    smoothed_foreground_scores = smooth_stack_bidirectional(
        foreground_scores,
        flows,
        native_weight=args.native_weight,
        prev_weight=args.prev_weight,
        next_weight=args.next_weight,
    )
    smoothed_foreground_masks = (
        smoothed_foreground_scores >= args.foreground_threshold
    ).astype(np.uint8)

    print(f"Writing outputs to {out_dir}", flush=True)
    tifffile.imwrite(
        out_dir / "smoothed_contour_maps.tif",
        smoothed_contours,
        compression="zlib",
    )
    tifffile.imwrite(
        out_dir / "smoothed_foreground_scores.tif",
        smoothed_foreground_scores,
        compression="zlib",
    )
    tifffile.imwrite(
        out_dir / "smoothed_foreground_masks.tif",
        smoothed_foreground_masks,
        compression="zlib",
    )
    tifffile.imwrite(out_dir / "flow_magnitude.tif", flow_magnitude, compression="zlib")
    if args.write_flow:
        tifffile.imwrite(out_dir / "flow_xy.tif", flows, compression="zlib")

    elapsed = perf_counter() - t0
    summary = {
        "script": Path(__file__).resolve(),
        "pos_dir": pos_dir,
        "input_paths": {
            "prob": prob_path,
            "contours": contour_path,
            "foreground_scores": foreground_score_path,
        },
        "output_dir": out_dir,
        "shape": prob.shape,
        "flow_params": FLOW_PARAMS,
        "weights": {
            "native": args.native_weight,
            "previous": args.prev_weight,
            "next": args.next_weight,
        },
        "foreground_threshold": args.foreground_threshold,
        "elapsed_s": round(elapsed, 3),
        "flow_magnitude": {
            "mean": float(np.mean(flow_magnitude)) if flow_magnitude.size else 0.0,
            "p95": float(np.percentile(flow_magnitude, 95)) if flow_magnitude.size else 0.0,
            "max": float(np.max(flow_magnitude)) if flow_magnitude.size else 0.0,
        },
        "outputs": {
            "smoothed_contour_maps": out_dir / "smoothed_contour_maps.tif",
            "smoothed_foreground_scores": out_dir / "smoothed_foreground_scores.tif",
            "smoothed_foreground_masks": out_dir / "smoothed_foreground_masks.tif",
            "flow_magnitude": out_dir / "flow_magnitude.tif",
            "flow_xy": out_dir / "flow_xy.tif" if args.write_flow else None,
        },
        "stack_summaries": [
            _summarize_stack("contours", contours, smoothed_contours),
            _summarize_stack(
                "foreground_scores", foreground_scores, smoothed_foreground_scores
            ),
        ],
    }
    _write_json(out_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
