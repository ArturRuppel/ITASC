"""Run a frame-by-frame 2D seeded watershed with Voronoi-fused contour maps.

Cellpose contour maps are temporally inconsistent — spurious boundaries in some
frames, missing boundaries in others.  This experiment fuses the Cellpose
contour map with a per-frame Euclidean Voronoi boundary derived from tracked
nuclear labels.  The Voronoi boundary is perfectly temporally consistent (it
follows the tracks), so blending it with the Cellpose signal fills in missing
contours and dilutes spurious ones.

Eight parameter combinations are swept in a single invocation.
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
from scipy.ndimage import gaussian_filter
from skimage.segmentation import expand_labels, find_boundaries, watershed


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
    {"alpha": 0.0, "voronoi_sigma": 2.0},
    {"alpha": 0.1, "voronoi_sigma": 2.0},
    {"alpha": 0.25, "voronoi_sigma": 2.0},
    {"alpha": 0.5, "voronoi_sigma": 2.0},
    {"alpha": 0.75, "voronoi_sigma": 2.0},
    {"alpha": 0.9, "voronoi_sigma": 2.0},
    {"alpha": 1.0, "voronoi_sigma": 2.0},
    {"alpha": 0.5, "voronoi_sigma": 5.0},
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
        description="Voronoi-fusion 2D seeded watershed sweep."
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


def _build_voronoi_boundary(
    nuc_labels: np.ndarray,
    fg_mask: np.ndarray,
    voronoi_sigma: float,
) -> np.ndarray:
    """Compute a soft Voronoi boundary from nuclear labels restricted to foreground.

    Parameters
    ----------
    nuc_labels : (Y, X) uint32
        Dense nuclear label map for one frame.
    fg_mask : (Y, X) bool
        Binary foreground domain mask.
    voronoi_sigma : float
        Gaussian sigma for spatial softening of the 1px Voronoi boundary.

    Returns
    -------
    voronoi_boundary : (Y, X) float32 in [0, 1]
    """
    max_dim = max(nuc_labels.shape)
    restricted = nuc_labels.copy()
    restricted[~fg_mask] = 0
    voronoi_cells = expand_labels(restricted, distance=max_dim)
    voronoi_cells[~fg_mask] = 0
    boundary_binary = find_boundaries(voronoi_cells, mode="inner")
    boundary_soft = gaussian_filter(
        boundary_binary.astype(np.float32), sigma=float(voronoi_sigma)
    )
    boundary_soft = np.clip(boundary_soft, 0.0, 1.0)
    return boundary_soft.astype(np.float32)


def main() -> None:
    args = _parse_args()
    base_dir = args.pos_dir / "3_cell" / "voronoi_fusion_watershed_experiment"

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
        alpha = ps["alpha"]
        voronoi_sigma = ps["voronoi_sigma"]
        run_dir = base_dir / f"{args.timestamp}-alpha{alpha}-vs{voronoi_sigma}"
        if run_dir.exists() and not args.overwrite:
            raise FileExistsError(
                f"{run_dir} exists; pass --overwrite or use a new --timestamp"
            )
        run_dir.mkdir(parents=True, exist_ok=True)

        header = f"=== [{idx}/{len(PARAM_SETS)}] alpha={alpha} voronoi_sigma={voronoi_sigma} ==="
        print(header, flush=True)

        t0_run = perf_counter()

        labels = np.zeros_like(markers, dtype=np.uint32)
        fused_stack = np.zeros(contours.shape, dtype=np.float32)
        voronoi_stack = np.zeros(contours.shape, dtype=np.float32)
        frame_summaries = []

        for t in range(n_frames):
            t0 = perf_counter()
            voronoi_boundary = _build_voronoi_boundary(
                full_markers[t], effective_mask[t], voronoi_sigma
            )
            voronoi_stack[t] = voronoi_boundary
            fused = alpha * contours[t] + (1.0 - alpha) * voronoi_boundary
            fused_stack[t] = fused
            frame_labels = watershed(
                fused,
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
        tifffile.imwrite(run_dir / "fused_contours.tif", fused_stack, compression="zlib")
        if idx == 1:
            tifffile.imwrite(
                run_dir / "voronoi_boundary.tif", voronoi_stack, compression="zlib"
            )

        elapsed_run = round(perf_counter() - t0_run, 3)
        summary = {
            "elapsed_s": elapsed_run,
            "path": label_path,
            **_summarize_labels(labels, markers, effective_mask),
        }

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
            "force_marker_domain": True,
            "added_marker_voxels": added_marker_voxels,
            "compactness": 0.0,
            "watershed_line": False,
            "mode": "frame_by_frame_2d",
            "alpha": alpha,
            "voronoi_sigma": voronoi_sigma,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": run_dir,
        }

        params["finished_at"] = datetime.now().isoformat(timespec="seconds")
        params["summary"] = summary
        _write_json(run_dir / "parameters.json", params)
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "frame_summaries.json", {"frames": frame_summaries})

        run_info = {
            "filter": "fusion",
            "alpha": alpha,
            "vs": voronoi_sigma,
            "n_out": summary["n_output_ids"],
            "n_miss": summary["n_missing_marker_ids"],
            "unlab_fg": summary["unlabeled_foreground_voxels"],
            "elap_s": elapsed_run,
        }
        all_run_summaries.append(run_info)

    print()
    print(
        f"{'filter':>8s}  {'alpha':>6s}  {'vs':>5s}  {'n_out':>6s}  "
        f"{'n_miss':>6s}  {'unlab_fg':>9s}  {'elap_s':>8s}"
    )
    print("-" * 65)
    for r in all_run_summaries:
        print(
            f"{r['filter']:>8s}  {r['alpha']:>6.2f}  {r['vs']:>5.1f}  "
            f"{r['n_out']:>6d}  {r['n_miss']:>6d}  "
            f"{r['unlab_fg']:>9d}  {r['elap_s']:>8.1f}"
        )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
