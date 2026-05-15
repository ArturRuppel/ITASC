"""Run a 2D per-frame α-expansion graph cut experiment for cell labels (fast variant).

Uses vectorized add_grid_edges / add_grid_tedges / get_grid_segments on the full
(H, W) pixel grid instead of a compact foreground-only node set with a Python
edge loop. Background pixels are pinned to the sink via infinite t-link capacity.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import maxflow
import numpy as np
import tifffile
from cellflow.segmentation import centroid_markers_from_labels
from scipy.ndimage import distance_transform_edt


DEFAULT_POS_DIR = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/"
    "analysis/pos00"
)
DEFAULT_CONTOUR_DIR = (
    DEFAULT_POS_DIR
    / "3_cell"
    / "contour_experiment"
    / "20260503-232245-thr-8-to-0-maxfg"
)
DEFAULT_SMOOTHNESS_WEIGHT = [5.0, 20.0, 50.0, 100.0]
_INF = 1e10

_H_STRUCT = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
_V_STRUCT = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _format_float(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text or "0"


def _precompute_edge_weights(
    contours: np.ndarray,
    foreground: np.ndarray,
    smoothness_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute per-pixel edge weights for add_grid_edges.

    Returns h_weights (H, W) and v_weights (H, W). Each value is the edge weight
    from pixel (y, x) to its right / down neighbor. Edges crossing the
    foreground-background boundary are zeroed out.
    """
    H, W = contours.shape
    c = np.asarray(contours, dtype=np.float32)
    fg = foreground.astype(bool)

    h_weights = np.zeros((H, W), dtype=np.float64)
    h_weights[:, :-1] = (
        smoothness_weight
        * np.maximum(0.0, 1.0 - 0.5 * (c[:, :-1] + c[:, 1:]))
        * (fg[:, :-1] & fg[:, 1:])
    )

    v_weights = np.zeros((H, W), dtype=np.float64)
    v_weights[:-1, :] = (
        smoothness_weight
        * np.maximum(0.0, 1.0 - 0.5 * (c[:-1, :] + c[1:, :]))
        * (fg[:-1, :] & fg[1:, :])
    )

    return h_weights, v_weights


def _run_alpha_expansion_fast(
    contours: np.ndarray,
    foreground: np.ndarray,
    seeds: np.ndarray,
    smoothness_weight: float,
    max_rounds: int = 5,
) -> np.ndarray:
    """2D α-expansion graph cut on the full pixel grid using vectorized edge ops.

    contours:  (Y, X) float32 in [0, 1]; high = likely boundary
    foreground: (Y, X) bool; pixels outside are fixed to label 0
    seeds:     (Y, X) uint32; nonzero pixels are hard-pinned to their label
    Returns:   (Y, X) uint32; background pixels are 0
    """
    H, W = contours.shape
    n_nodes = H * W
    fg = np.asarray(foreground, dtype=bool)
    flat_fg = fg.ravel()
    flat_seeds = seeds.ravel().astype(np.uint32)

    # Node ID grid — all pixels are nodes (background pinned via t-links)
    nodeids = np.arange(n_nodes, dtype=np.int32).reshape(H, W)
    flat_nodeids = nodeids.ravel()

    # Precompute edge weights once (reused across all GC builds)
    h_weights, v_weights = _precompute_edge_weights(contours, fg, smoothness_weight)

    # Initialize: each foreground pixel gets nearest-seed label
    seed_pixels = seeds > 0
    if seed_pixels.any():
        _, nearest = distance_transform_edt(~seed_pixels, return_indices=True)
        init_labels = seeds[nearest[0], nearest[1]]
    else:
        init_labels = np.zeros((H, W), dtype=np.uint32)
    current_labels = np.where(fg, init_labels, 0).astype(np.uint32)

    label_ids = np.unique(seeds)
    label_ids = label_ids[label_ids != 0]

    for _round in range(max_rounds):
        changed = False
        for alpha in label_ids:
            g = maxflow.Graph[float](n_nodes, 2 * n_nodes)
            g.add_nodes(n_nodes)

            # Data term (all vectorized)
            sourcecaps = np.zeros(n_nodes, dtype=np.float64)
            sinkcaps = np.zeros(n_nodes, dtype=np.float64)
            sinkcaps[~flat_fg] = _INF                              # pin background to sink
            sourcecaps[flat_seeds == alpha] = _INF                 # pin alpha seeds to source
            sinkcaps[flat_seeds == alpha] = 0.0
            other_seeds = (flat_seeds != 0) & (flat_seeds != alpha)
            sinkcaps[other_seeds] = _INF                           # pin other seeds to sink
            sourcecaps[other_seeds] = 0.0
            g.add_grid_tedges(flat_nodeids, sourcecaps, sinkcaps)

            # Smoothness term (vectorized — no Python edge loop)
            g.add_grid_edges(nodeids, h_weights, _H_STRUCT, symmetric=True)
            g.add_grid_edges(nodeids, v_weights, _V_STRUCT, symmetric=True)

            g.maxflow()

            # Update (vectorized)
            in_sink = g.get_grid_segments(flat_nodeids).reshape(H, W)
            new_alpha = ~in_sink & fg
            prev = current_labels[new_alpha]
            if np.any(prev != alpha):
                changed = True
            current_labels[new_alpha] = alpha

        if not changed:
            break

    return current_labels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast 2D α-expansion graph cut (vectorized edges) for cell segmentation."
    )
    parser.add_argument("--pos-dir", type=Path, default=DEFAULT_POS_DIR)
    parser.add_argument(
        "--contours",
        type=Path,
        default=DEFAULT_CONTOUR_DIR / "contours.tif",
    )
    parser.add_argument(
        "--foreground-mask",
        type=Path,
        default=DEFAULT_CONTOUR_DIR / "foreground_masks.tif",
    )
    parser.add_argument(
        "--markers",
        type=Path,
        default=DEFAULT_POS_DIR / "2_nucleus" / "tracked_labels.tif",
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        nargs="+",
        default=DEFAULT_SMOOTHNESS_WEIGHT,
    )
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument(
        "--timestamp",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--crop",
        type=int,
        nargs=6,
        metavar=("T0", "T1", "Y0", "Y1", "X0", "X1"),
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
        raise ValueError(f"Foreground shape {foreground.shape} != contours {contours.shape}")
    if markers.shape != contours.shape:
        raise ValueError(f"Markers shape {markers.shape} != contours {contours.shape}")

    if args.crop is not None:
        t0, t1, y0, y1, x0, x1 = args.crop
        crop = (slice(t0, t1), slice(y0, y1), slice(x0, x1))
        contours, foreground, markers = contours[crop], foreground[crop], markers[crop]

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
    unlabeled_fg = mask & (labels == 0)
    return {
        "n_marker_ids": int(marker_ids.size),
        "n_output_ids": int(label_ids.size),
        "missing_marker_ids": [int(v) for v in missing_ids[:50]],
        "n_missing_marker_ids": int(missing_ids.size),
        "extra_output_ids": [int(v) for v in extra_ids[:50]],
        "n_extra_output_ids": int(extra_ids.size),
        "foreground_voxels": int(np.count_nonzero(mask)),
        "labeled_voxels": int(np.count_nonzero(labels)),
        "unlabeled_foreground_voxels": int(np.count_nonzero(unlabeled_fg)),
        "max_label": int(labels.max()) if labels.size else 0,
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.pos_dir / "3_cell" / "graphcut_fast_experiment" / args.timestamp
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite or use a new --timestamp")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading inputs...", flush=True)
    contours, foreground_raw, full_markers = _load_inputs(args)
    seeds = centroid_markers_from_labels(full_markers)
    fg_mask = foreground_raw > 0

    params: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "pos_dir": args.pos_dir,
        "contours": args.contours,
        "foreground_mask": args.foreground_mask,
        "markers": args.markers,
        "smoothness_weight": [float(v) for v in args.smoothness_weight],
        "max_rounds": int(args.max_rounds),
        "shape": tuple(int(v) for v in contours.shape),
        "crop": args.crop,
        "seed_mode": "centroid",
        "variant": "fast_grid_edges",
        "seed_marker_voxels": int(np.count_nonzero(seeds)),
        "foreground_voxels": int(np.count_nonzero(fg_mask)),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": output_dir,
    }
    _write_json(output_dir / "parameters.json", params)
    tifffile.imwrite(output_dir / "seed_markers.tif", seeds, compression="zlib")

    n_frames = contours.shape[0]
    summaries: list[dict[str, Any]] = []

    for smoothness_weight in args.smoothness_weight:
        suffix = f"sw_{_format_float(float(smoothness_weight))}"
        label_path = output_dir / f"tracked_labels_{suffix}.tif"
        print(f"Running fast graph cut {suffix} ({n_frames} frames)...", flush=True)
        t0 = perf_counter()

        all_frames: list[np.ndarray] = []
        for t in range(n_frames):
            frame_labels = _run_alpha_expansion_fast(
                contours[t],
                fg_mask[t],
                seeds[t],
                smoothness_weight=float(smoothness_weight),
                max_rounds=int(args.max_rounds),
            )
            all_frames.append(frame_labels)
            if (t + 1) % 10 == 0 or t + 1 == n_frames:
                print(f"  frame {t + 1}/{n_frames}", flush=True)

        labels = np.stack(all_frames, axis=0).astype(np.uint32)
        elapsed_s = perf_counter() - t0

        tifffile.imwrite(label_path, labels, compression="zlib")
        summary = {
            "smoothness_weight": float(smoothness_weight),
            "elapsed_s": round(elapsed_s, 3),
            "path": label_path,
            **_summarize_labels(labels, seeds, fg_mask),
        }
        summaries.append(summary)
        _write_json(output_dir / f"summary_{suffix}.json", summary)
        print(
            f"  wrote {label_path.name}: {summary['n_output_ids']} IDs, "
            f"{summary['n_missing_marker_ids']} missing, "
            f"{summary['unlabeled_foreground_voxels']} unlabeled fg, "
            f"{elapsed_s:.1f}s",
            flush=True,
        )

    params["finished_at"] = datetime.now().isoformat(timespec="seconds")
    params["summaries"] = summaries
    _write_json(output_dir / "parameters.json", params)
    _write_json(output_dir / "summaries.json", {"summaries": summaries})
    print("Done.", flush=True)
    print(f"Output directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
