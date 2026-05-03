"""Run a 3D weighted geodesic seeded Voronoi experiment for cell labels.

This experiment treats the time axis as the third image axis. Curated 2D
tracked nuclear labels are hard markers, the binary foreground mask is the
domain, and contour probability raises the cost of crossing putative borders.
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
from numba import njit
from skimage.graph import MCP_Geometric


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
DEFAULT_BOUNDARY_WEIGHT = [4.0, 16.0]
DEFAULT_TIME_SPACING = [5.0]
DEFAULT_BOUNDARY_ALPHA = [1.0]


@njit(cache=True)
def _propagate_labels_in_cost_order(
    order: np.ndarray,
    cumulative_costs: np.ndarray,
    traceback: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    stride_t: int,
    stride_y: int,
    stride_x: int,
) -> int:
    assigned = 0
    for idx in order:
        if not np.isfinite(cumulative_costs[idx]):
            continue
        if labels[idx] != 0:
            continue
        code = traceback[idx]
        if code < 0:
            continue
        predecessor_idx = (
            idx
            - int(offsets[code, 0]) * stride_t
            - int(offsets[code, 1]) * stride_y
            - int(offsets[code, 2]) * stride_x
        )
        predecessor_label = labels[predecessor_idx]
        if predecessor_label != 0:
            labels[idx] = predecessor_label
            assigned += 1
    return assigned


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D weighted geodesic seeded Voronoi for tracked cell segmentation."
    )
    parser.add_argument("--pos-dir", type=Path, default=DEFAULT_POS_DIR)
    parser.add_argument(
        "--contours",
        type=Path,
        default=DEFAULT_CONTOUR_DIR / "contours.tif",
        help="Contour probability/cost volume, expected shape (T, Y, X).",
    )
    parser.add_argument(
        "--foreground-mask",
        type=Path,
        default=DEFAULT_CONTOUR_DIR / "foreground_masks.tif",
        help="Binary foreground domain mask, expected shape (T, Y, X).",
    )
    parser.add_argument(
        "--markers",
        type=Path,
        default=DEFAULT_POS_DIR / "2_nucleus" / "tracked_labels.tif",
        help="Curated tracked nuclear labels, expected shape (T, Y, X).",
    )
    parser.add_argument(
        "--boundary-weight",
        type=float,
        nargs="+",
        default=DEFAULT_BOUNDARY_WEIGHT,
        help="Multiplier for contour-derived travel cost.",
    )
    parser.add_argument(
        "--boundary-alpha",
        type=float,
        nargs="+",
        default=DEFAULT_BOUNDARY_ALPHA,
        help="Exponent applied to normalized contour probabilities before weighting.",
    )
    parser.add_argument(
        "--time-spacing",
        type=float,
        nargs="+",
        default=DEFAULT_TIME_SPACING,
        help="Sampling distance for the time axis; y/x spacing stays 1.",
    )
    parser.add_argument(
        "--timestamp",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Checkpoint directory name under 3_cell/geodesic_voronoi_experiment.",
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
        "--fully-connected",
        action="store_true",
        help="Use 26-connected moves instead of axial 6-connected moves.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=6,
        metavar=("T0", "T1", "Y0", "Y1", "X0", "X1"),
        help="Optional crop for pilot runs, using half-open ranges.",
    )
    parser.add_argument(
        "--write-costs",
        action="store_true",
        help="Write each finite travel-cost volume as an intermediate checkpoint.",
    )
    parser.add_argument(
        "--write-cumulative-costs",
        action="store_true",
        help="Write each cumulative geodesic distance volume as an intermediate checkpoint.",
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

    if args.crop is not None:
        t0, t1, y0, y1, x0, x1 = args.crop
        crop = (slice(t0, t1), slice(y0, y1), slice(x0, x1))
        contours = contours[crop]
        foreground = foreground[crop]
        markers = markers[crop]

    return contours, foreground, markers


def _build_costs(
    contours: np.ndarray,
    mask: np.ndarray,
    boundary_weight: float,
    boundary_alpha: float,
) -> np.ndarray:
    normalized = np.clip(contours, 0.0, 1.0)
    costs = 1.0 + boundary_weight * np.power(normalized, boundary_alpha, dtype=np.float32)
    costs = np.asarray(costs, dtype=np.float32)
    costs[~mask] = np.inf
    return costs


def _run_geodesic_assignment(
    costs: np.ndarray,
    markers: np.ndarray,
    time_spacing: float,
    fully_connected: bool,
) -> tuple[np.ndarray, np.ndarray]:
    starts = [tuple(int(v) for v in coord) for coord in np.argwhere(markers > 0)]
    if not starts:
        raise ValueError("No positive marker pixels found.")

    mcp = MCP_Geometric(
        costs,
        fully_connected=fully_connected,
        sampling=(float(time_spacing), 1.0, 1.0),
    )
    cumulative_costs, traceback = mcp.find_costs(starts)

    labels = np.asarray(markers, dtype=np.uint32).reshape(-1).copy()
    order = np.argsort(cumulative_costs.reshape(-1), kind="stable")
    offsets = np.asarray(mcp.offsets, dtype=np.int64)
    stride_t = int(costs.shape[1] * costs.shape[2])
    stride_y = int(costs.shape[2])
    stride_x = 1
    _propagate_labels_in_cost_order(
        order.astype(np.int64, copy=False),
        cumulative_costs.reshape(-1),
        traceback.reshape(-1).astype(np.int64, copy=False),
        labels,
        offsets,
        stride_t,
        stride_y,
        stride_x,
    )
    return labels.reshape(costs.shape), np.asarray(cumulative_costs, dtype=np.float32)


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
    output_dir = args.pos_dir / "3_cell" / "geodesic_voronoi_experiment" / args.timestamp
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite or use a new --timestamp")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading inputs...", flush=True)
    contours, foreground_raw, markers = _load_inputs(args)
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
        "boundary_weight": [float(v) for v in args.boundary_weight],
        "boundary_alpha": [float(v) for v in args.boundary_alpha],
        "time_spacing": [float(v) for v in args.time_spacing],
        "shape": tuple(int(v) for v in contours.shape),
        "crop": args.crop,
        "contours_dtype": str(contours.dtype),
        "foreground_dtype": str(foreground_raw.dtype),
        "markers_dtype": str(markers.dtype),
        "foreground_binary_values": [int(v) for v in np.unique(foreground_raw)[:16]],
        "force_marker_domain": not args.no_force_marker_domain,
        "added_marker_voxels": added_marker_voxels,
        "fully_connected": bool(args.fully_connected),
        "write_costs": bool(args.write_costs),
        "write_cumulative_costs": bool(args.write_cumulative_costs),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": output_dir,
    }
    _write_json(output_dir / "parameters.json", params)

    tifffile.imwrite(
        output_dir / "effective_foreground_mask.tif",
        effective_mask.astype(np.uint8),
        compression="zlib",
    )

    summaries = []
    for boundary_alpha in args.boundary_alpha:
        for boundary_weight in args.boundary_weight:
            print(
                f"Building costs boundary_weight={boundary_weight:g}, "
                f"boundary_alpha={boundary_alpha:g}...",
                flush=True,
            )
            costs = _build_costs(
                contours,
                effective_mask,
                float(boundary_weight),
                float(boundary_alpha),
            )
            cost_suffix = (
                f"bw_{_format_float(float(boundary_weight))}"
                f"_alpha_{_format_float(float(boundary_alpha))}"
            )
            if args.write_costs:
                tifffile.imwrite(
                    output_dir / f"travel_costs_{cost_suffix}.tif",
                    costs,
                    compression="zlib",
                )

            for time_spacing in args.time_spacing:
                suffix = f"{cost_suffix}_tspace_{_format_float(float(time_spacing))}"
                label_path = output_dir / f"tracked_labels_{suffix}.tif"
                print(f"Running geodesic Voronoi {suffix}...", flush=True)
                t0 = perf_counter()
                labels, cumulative_costs = _run_geodesic_assignment(
                    costs,
                    markers,
                    float(time_spacing),
                    bool(args.fully_connected),
                )
                elapsed_s = perf_counter() - t0
                tifffile.imwrite(label_path, labels.astype(np.uint32, copy=False), compression="zlib")
                if args.write_cumulative_costs:
                    tifffile.imwrite(
                        output_dir / f"cumulative_costs_{suffix}.tif",
                        cumulative_costs,
                        compression="zlib",
                    )
                summary = {
                    "boundary_weight": float(boundary_weight),
                    "boundary_alpha": float(boundary_alpha),
                    "time_spacing": float(time_spacing),
                    "elapsed_s": round(elapsed_s, 3),
                    "path": label_path,
                    **_summarize_labels(labels, markers, effective_mask),
                }
                summaries.append(summary)
                _write_json(output_dir / f"summary_{suffix}.json", summary)
                print(
                    f"  wrote {label_path.name}: {summary['n_output_ids']} IDs, "
                    f"{summary['n_missing_marker_ids']} missing marker IDs, "
                    f"{summary['unlabeled_foreground_voxels']} unlabeled fg voxels, "
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
