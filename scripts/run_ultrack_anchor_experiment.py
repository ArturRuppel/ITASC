#!/usr/bin/env python
"""Run the standard Ultrack pipeline and compare it against curated GT labels.

This is the baseline for anchor/constraint experiments:

    hypotheses.h5 -> Ultrack DB -> linking -> ILP solve -> tracked_labels.tif

Usage from the repository root:

    python scripts/run_ultrack_anchor_experiment.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.anchor import (
    annotate_anchor_frame,
    suppress_anchor_adjacent_fragments,
)
from cellflow.tracking_ultrack.export import export_tracked_labels
from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db
from cellflow.tracking_ultrack.linking import run_linking
from cellflow.tracking_ultrack.metrics import binary_labelmap_iou, tracked_label_summary
from cellflow.tracking_ultrack.solve import run_solve


DEFAULT_NUCLEUS_DIR = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/analysis/pos00/2_nucleus"
)


def _labels_2d(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected labels shaped (T, Y, X) or (T, 1, Y, X), got {arr.shape}")


def _time_stage(name: str, timings: dict[str, float]):
    class _Timer:
        def __enter__(self):
            self.start = time.monotonic()
            print(f"\n[{len(timings) + 1}] {name} ...", flush=True)
            return self

        def __exit__(self, exc_type, exc, tb):
            timings[name] = time.monotonic() - self.start
            print(f"    {name} done in {timings[name]:.1f}s", flush=True)

    return _Timer()


def _summary_dict(labels: np.ndarray) -> dict[str, Any]:
    summary = tracked_label_summary(labels)
    lengths = np.array(list(summary.track_lengths.values()), dtype=np.float32)
    return {
        "n_tracks": summary.n_tracks,
        "average_length": summary.average_length,
        "median_length": float(np.median(lengths)) if lengths.size else 0.0,
        "min_length": int(lengths.min()) if lengths.size else 0,
        "max_length": int(lengths.max()) if lengths.size else 0,
    }


def _compare(result_labels: np.ndarray, gt_labels: np.ndarray) -> dict[str, Any]:
    result = _labels_2d(result_labels)
    gt = _labels_2d(gt_labels)
    n_frames = min(result.shape[0], gt.shape[0])
    result = result[:n_frames]
    gt = gt[:n_frames]
    if result.shape != gt.shape:
        raise ValueError(f"Shape mismatch after frame trim: result {result.shape}, GT {gt.shape}")

    frame_ious = [binary_labelmap_iou(result[t], gt[t]) for t in range(n_frames)]
    return {
        "frames_compared": n_frames,
        "result": _summary_dict(result),
        "ground_truth": _summary_dict(gt),
        "binary_iou": {
            "global": binary_labelmap_iou(result, gt),
            "mean_per_frame": float(np.mean(frame_ious)) if frame_ious else 0.0,
            "min_per_frame": float(np.min(frame_ious)) if frame_ious else 0.0,
            "max_per_frame": float(np.max(frame_ious)) if frame_ious else 0.0,
        },
    }


def _print_comparison(comparison: dict[str, Any]) -> None:
    gt = comparison["ground_truth"]
    result = comparison["result"]
    iou = comparison["binary_iou"]
    print("\n=== Standard Ultrack vs GT ===")
    print(f"Frames compared: {comparison['frames_compared']}")
    print(
        "GT tracks: "
        f"n={gt['n_tracks']}, avg_len={gt['average_length']:.2f}, "
        f"median={gt['median_length']:.2f}, min={gt['min_length']}, max={gt['max_length']}"
    )
    print(
        "Ultrack tracks: "
        f"n={result['n_tracks']}, avg_len={result['average_length']:.2f}, "
        f"median={result['median_length']:.2f}, min={result['min_length']}, max={result['max_length']}"
    )
    print(
        "Binary labelmap IoU: "
        f"global={iou['global']:.4f}, mean_frame={iou['mean_per_frame']:.4f}, "
        f"min_frame={iou['min_per_frame']:.4f}, max_frame={iou['max_per_frame']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nucleus-dir", type=Path, default=DEFAULT_NUCLEUS_DIR)
    parser.add_argument("--working-dir", type=Path, default=None)
    parser.add_argument("--n-frames", type=int, default=None)
    parser.add_argument(
        "--max-partitions",
        type=int,
        default=0,
        help="0 means use every unique partition in hypotheses.h5.",
    )
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--max-area", type=int, default=1_000_000)
    parser.add_argument("--max-distance", type=float, default=15.0)
    parser.add_argument("--max-neighbors", type=int, default=5)
    parser.add_argument("--linking-mode", choices=["default", "iou"], default="default")
    parser.add_argument("--iou-weight", type=float, default=1.0)
    parser.add_argument("--min-link-iou", type=float, default=0.1)
    parser.add_argument("--link-n-workers", type=int, default=None)
    parser.add_argument("--time-limit", type=int, default=36_000)
    parser.add_argument("--appear-weight", type=float, default=-0.001)
    parser.add_argument("--disappear-weight", type=float, default=-0.001)
    parser.add_argument("--division-weight", type=float, default=-0.001)
    parser.add_argument("--power", type=float, default=4.0)
    parser.add_argument("--solution-gap", type=float, default=0.001)
    parser.add_argument(
        "--anchor-frame",
        type=int,
        default=None,
        help="Frame index to pin to the GT labelmap. Default: no anchor.",
    )
    parser.add_argument(
        "--anchor-middle-frame",
        action="store_true",
        help="Pin the middle GT frame before solving.",
    )
    parser.add_argument("--anchor-min-iou", type=float, default=0.95)
    parser.add_argument(
        "--suppress-anchor-adjacent-fragments",
        action="store_true",
        help="Mark obvious fragment alternatives in frames adjacent to the anchor as FAKE.",
    )
    parser.add_argument("--suppression-min-best-iou", type=float, default=0.60)
    parser.add_argument("--suppression-fragment-max-iou-fraction", type=float, default=0.80)
    parser.add_argument("--suppression-min-fragment-containment", type=float, default=0.90)
    args = parser.parse_args()

    nucleus_dir = args.nucleus_dir
    hypotheses_h5 = nucleus_dir / "hypotheses.h5"
    gt_tif = nucleus_dir / "tracked_labels.tif"
    working_dir = args.working_dir or (nucleus_dir / "ultrack_anchor_experiment" / "standard")
    output_tif = working_dir / "tracked_labels.tif"
    report_json = working_dir / "comparison_standard.json"

    if not hypotheses_h5.exists():
        raise FileNotFoundError(hypotheses_h5)
    if not gt_tif.exists():
        raise FileNotFoundError(gt_tif)

    working_dir.mkdir(parents=True, exist_ok=True)
    max_partitions = args.max_partitions if args.max_partitions > 0 else None

    cfg_kwargs: dict[str, Any] = {
        "min_area": args.min_area,
        "max_area": args.max_area,
        "max_distance": args.max_distance,
        "max_neighbors": args.max_neighbors,
        "linking_mode": args.linking_mode,
        "iou_weight": args.iou_weight,
        "min_link_iou": args.min_link_iou,
        "time_limit": args.time_limit,
        "appear_weight": args.appear_weight,
        "disappear_weight": args.disappear_weight,
        "division_weight": args.division_weight,
        "power": args.power,
        "solution_gap": args.solution_gap,
    }
    if args.link_n_workers is not None:
        cfg_kwargs["link_n_workers"] = args.link_n_workers
    cfg = TrackingConfig(**cfg_kwargs)

    print("=== Ultrack Anchor Experiment: standard baseline ===")
    print(f"Nucleus dir: {nucleus_dir}")
    print(f"Working dir: {working_dir}")
    print(f"n_frames: {args.n_frames if args.n_frames is not None else 'all'}")
    print(f"max_partitions: {max_partitions if max_partitions is not None else 'all'}")
    print(f"config: {cfg.model_dump()}")

    timings: dict[str, float] = {}
    with _time_stage("ingest", timings):
        ingest_hypotheses_to_db(
            hypotheses_h5,
            working_dir,
            cfg,
            overwrite=True,
            max_partitions=max_partitions,
            n_frames=args.n_frames,
        )

    with _time_stage("link", timings):
        for step, total, label in run_linking(working_dir, cfg, overwrite=True):
            print(f"    [{step}/{total}] {label}", flush=True)

    anchor_frame = args.anchor_frame
    anchor_report = None
    gt_labels = tifffile.imread(gt_tif)
    if args.n_frames is not None:
        gt_labels = gt_labels[: args.n_frames]
    if args.anchor_middle_frame:
        anchor_frame = int(gt_labels.shape[0] // 2)

    use_annotations = anchor_frame is not None
    if anchor_frame is not None:
        with _time_stage("anchor", timings):
            report = annotate_anchor_frame(
                working_dir,
                gt_labels,
                frame_index=int(anchor_frame),
                min_iou=args.anchor_min_iou,
            )
            anchor_report = report
            print(
                "    anchor report: "
                f"frame={report.frame_index}, gt={report.n_gt_labels}, "
                f"matched={report.n_matched}, unmatched={report.n_unmatched}, "
                f"mean_iou={report.mean_matched_iou:.4f}, "
                f"min_iou={report.min_matched_iou:.4f}",
                flush=True,
            )
            if report.unmatched_labels:
                preview = report.unmatched_labels[:10]
                suffix = "..." if len(report.unmatched_labels) > len(preview) else ""
                print(f"    unmatched GT labels: {preview}{suffix}", flush=True)

            if args.suppress_anchor_adjacent_fragments:
                suppression_report = suppress_anchor_adjacent_fragments(
                    working_dir,
                    gt_labels,
                    frame_index=int(anchor_frame),
                    min_best_iou=args.suppression_min_best_iou,
                    fragment_max_iou_fraction=args.suppression_fragment_max_iou_fraction,
                    min_fragment_containment=args.suppression_min_fragment_containment,
                )
                anchor_report = (report, suppression_report)
                print(
                    "    adjacent suppression: "
                    f"suppressed={len(suppression_report.suppressed_node_ids)}, "
                    f"by_frame={suppression_report.by_frame}",
                    flush=True,
                )

    with _time_stage("solve", timings):
        for step, total, label in run_solve(
            working_dir,
            cfg,
            overwrite=True,
            use_annotations=use_annotations,
        ):
            print(f"    [{step}/{total}] {label}", flush=True)

    with _time_stage("export", timings):
        result_labels = export_tracked_labels(working_dir, cfg, output_tif)

    comparison = _compare(result_labels, gt_labels)
    if anchor_report is not None:
        suppression_report = None
        if isinstance(anchor_report, tuple):
            anchor_report, suppression_report = anchor_report
        comparison["anchor"] = {
            "frame_index": anchor_report.frame_index,
            "min_iou_threshold": args.anchor_min_iou,
            "n_gt_labels": anchor_report.n_gt_labels,
            "n_matched": anchor_report.n_matched,
            "n_unmatched": anchor_report.n_unmatched,
            "mean_matched_iou": anchor_report.mean_matched_iou,
            "min_matched_iou": anchor_report.min_matched_iou,
            "unmatched_labels": anchor_report.unmatched_labels,
        }
        if suppression_report is not None:
            comparison["anchor"]["adjacent_suppression"] = {
                "neighbor_offsets": list(suppression_report.neighbor_offsets),
                "min_best_iou": args.suppression_min_best_iou,
                "fragment_max_iou_fraction": args.suppression_fragment_max_iou_fraction,
                "min_fragment_containment": args.suppression_min_fragment_containment,
                "n_suppressed": len(suppression_report.suppressed_node_ids),
                "suppressed_node_ids": suppression_report.suppressed_node_ids,
                "by_frame": suppression_report.by_frame,
            }
    comparison["timings_seconds"] = timings
    comparison["output_tif"] = str(output_tif)
    comparison["working_dir"] = str(working_dir)
    comparison["hypotheses_h5"] = str(hypotheses_h5)
    comparison["ground_truth_tif"] = str(gt_tif)

    _print_comparison(comparison)
    report_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nWrote result labels: {output_tif}")
    print(f"Wrote comparison report: {report_json}")


if __name__ == "__main__":
    main()
