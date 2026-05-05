"""Seeded 2D watershed using optical-flow-blended contour maps.

For each frame t, contours[t] is blended with a motion-compensated prediction
warped from contours[t-1] using dense Farneback optical flow computed on the
Cellpose probability maps.  A sweep over the blend ratio alpha reveals how
much temporal consistency the flow-based prediction contributes to segmentation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import tifffile
from cellflow.segmentation import centroid_markers_from_labels
from skimage.segmentation import find_boundaries, watershed


BASE = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/"
    "analysis/pos00"
)
PROB_PATH = BASE / "1_cellpose" / "cell_prob_zavg.tif"
CONTOURS_PATH = (
    BASE / "3_cell" / "contour_experiment"
    / "20260504-contours-meanz-thr-m5-to-5" / "contours.tif"
)
FOREGROUND_PATH = (
    BASE / "3_cell" / "contour_experiment"
    / "20260503-232245-thr-8-to-0-maxfg" / "foreground.tif"
)
NUCLEUS_PATH = BASE / "2_nucleus" / "tracked_labels.tif"
OUT_BASE = BASE / "3_cell" / "optical_flow_experiment"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

FLOW_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 63,
    "iterations": 3,
    "poly_n": 7,
    "poly_sigma": 1.5,
    "flags": 0,
}

H, W = 512, 512
N_FRAMES = 50


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def compute_flows(prob: np.ndarray) -> list[np.ndarray]:
    """Compute dense Farneback optical flow for all 49 consecutive frame pairs.

    Each prob frame is normalized to uint8 before flow computation.
    """
    flows: list[np.ndarray] = []
    for t in range(N_FRAMES - 1):
        f0 = cv2.normalize(prob[t], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        f1 = cv2.normalize(prob[t + 1], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            f0, f1, None,
            pyr_scale=FLOW_PARAMS["pyr_scale"],
            levels=FLOW_PARAMS["levels"],
            winsize=FLOW_PARAMS["winsize"],
            iterations=FLOW_PARAMS["iterations"],
            poly_n=FLOW_PARAMS["poly_n"],
            poly_sigma=FLOW_PARAMS["poly_sigma"],
            flags=FLOW_PARAMS["flags"],
        )
        flows.append(flow)
    return flows


def build_blended_contours(
    contours: np.ndarray, flows: list[np.ndarray], alpha: float,
) -> np.ndarray:
    """Blend each frame's contours with the previous frame warped by optical flow.

    blended[0] = contours[0]
    For t >= 1: blended[t] = alpha * contours[t] + (1-alpha) * warp(blended[t-1])
    """
    blended = np.zeros((N_FRAMES, H, W), dtype=np.float32)
    blended[0] = contours[0]

    y_grid, x_grid = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )

    for t in range(1, N_FRAMES):
        dx = flows[t - 1][..., 0]
        dy = flows[t - 1][..., 1]
        map_x = x_grid - dx
        map_y = y_grid - dy
        predicted = cv2.remap(
            blended[t - 1], map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        blended[t] = alpha * contours[t] + (1.0 - alpha) * predicted

    return blended


def _per_frame_metrics(
    labels: np.ndarray,
    blended: np.ndarray,
    foreground: np.ndarray,
    markers: np.ndarray,
) -> dict[str, Any]:
    marker_ids = np.unique(markers)
    marker_ids = marker_ids[marker_ids != 0]
    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids != 0]
    missing = np.setdiff1d(marker_ids, label_ids)

    fg_mask = foreground > 0
    unlabeled_fg = np.sum(fg_mask & (labels == 0))

    boundaries = find_boundaries(labels, mode="inner")
    interior = (labels > 0) & ~boundaries
    boundary_align = 0.0
    if boundaries.any():
        b_mean = float(np.mean(blended[boundaries]))
        i_mean = float(np.mean(blended[interior])) if interior.any() else 1.0
        boundary_align = b_mean / i_mean if i_mean > 0 else 0.0

    return {
        "n_output_ids": int(label_ids.size),
        "n_missing_marker_ids": int(missing.size),
        "unlabeled_foreground_voxels": int(unlabeled_fg),
        "boundary_alignment_score": round(boundary_align, 6),
    }



def main() -> None:
    print("Loading inputs...", flush=True)
    prob = np.asarray(tifffile.imread(PROB_PATH), dtype=np.float32)
    contours = np.asarray(tifffile.imread(CONTOURS_PATH), dtype=np.float32)
    foreground_raw = np.asarray(tifffile.imread(FOREGROUND_PATH))
    nucleus_labels = np.asarray(tifffile.imread(NUCLEUS_PATH), dtype=np.uint32)

    markers = centroid_markers_from_labels(nucleus_labels)

    print("Computing optical flow (49 frame pairs)...", flush=True)
    t_flow_start = perf_counter()
    flows = compute_flows(prob)
    print(f"  Flow done in {perf_counter() - t_flow_start:.1f}s", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_run_summaries: list[dict[str, Any]] = []

    for alpha in ALPHAS:
        run_dir = OUT_BASE / f"{timestamp}-alpha-{alpha}"
        run_dir.mkdir(parents=True, exist_ok=True)

        header = f"=== alpha={alpha} ==="
        print(header, flush=True)
        t0 = perf_counter()

        blended = build_blended_contours(contours, flows, alpha)

        labels = np.zeros((N_FRAMES, H, W), dtype=np.uint32)
        frame_summaries: list[dict[str, Any]] = []

        for t in range(N_FRAMES):
            t_frame = perf_counter()
            height = 1.0 - blended[t].astype(np.float64)
            frame_labels = watershed(
                height,
                markers=markers[t].astype(np.int32, copy=False),
                mask=(foreground_raw[t] > 0),
                compactness=0.0,
                watershed_line=False,
            )
            labels[t] = np.asarray(frame_labels, dtype=np.uint32)
            pf = _per_frame_metrics(
                labels[t], blended[t], foreground_raw[t], markers[t],
            )
            pf["t"] = t
            pf["elapsed_s"] = round(perf_counter() - t_frame, 3)
            frame_summaries.append(pf)
            print(
                f"  frame {t + 1}/{N_FRAMES}: "
                f"{pf['n_output_ids']} IDs, "
                f"{pf['n_missing_marker_ids']} missing, "
                f"{pf['unlabeled_foreground_voxels']} unlabeled fg, "
                f"align={pf['boundary_alignment_score']:.4f}, "
                f"{pf['elapsed_s']:.2f}s",
                flush=True,
            )

        # Aggregate metrics
        agg = {
            key: round(float(np.mean([f[key] for f in frame_summaries])), 6)
            for key in ("n_output_ids", "n_missing_marker_ids",
                         "unlabeled_foreground_voxels", "boundary_alignment_score")
        }

        # Write outputs
        tifffile.imwrite(run_dir / "labels.tif", labels, compression="zlib")
        tifffile.imwrite(run_dir / "blended_contours.tif", blended, compression="zlib")

        elapsed_run = round(perf_counter() - t0, 3)

        params = {
            "script": str(Path(__file__).resolve()),
            "alpha": alpha,
            "flow_params": FLOW_PARAMS,
            "prob_path": PROB_PATH,
            "contours_path": CONTOURS_PATH,
            "foreground_path": FOREGROUND_PATH,
            "nucleus_path": NUCLEUS_PATH,
            "shape": (N_FRAMES, H, W),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": run_dir,
        }
        params["finished_at"] = datetime.now().isoformat(timespec="seconds")
        params["metrics"] = agg
        _write_json(run_dir / "parameters.json", params)

        with open(run_dir / "frame_summaries.jsonl", "w") as fh:
            for fs in frame_summaries:
                fh.write(json.dumps(fs, default=_json_default) + "\n")

        run_info = {
            "alpha": alpha,
            "n_out": agg["n_output_ids"],
            "n_miss": agg["n_missing_marker_ids"],
            "unlab_fg": agg["unlabeled_foreground_voxels"],
            "bnd_align": agg["boundary_alignment_score"],
            "elap_s": elapsed_run,
        }
        all_run_summaries.append(run_info)
        print(f"  Done in {elapsed_run:.1f}s", flush=True)

    print()
    print(
        f"{'alpha':>6s}  {'n_out':>7s}  {'n_miss':>7s}  "
        f"{'unlab_fg':>9s}  {'bnd_align':>10s}  {'elap_s':>8s}"
    )
    print("-" * 60)
    for r in all_run_summaries:
        print(
            f"{r['alpha']:>6.2f}  {r['n_out']:>7.1f}  {r['n_miss']:>7.1f}  "
            f"{r['unlab_fg']:>9.1f}  {r['bnd_align']:>10.4f}  {r['elap_s']:>8.1f}"
        )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
