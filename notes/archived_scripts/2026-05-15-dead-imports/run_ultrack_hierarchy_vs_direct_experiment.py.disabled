#!/usr/bin/env python
"""Compare direct H5 Ultrack ingest against canonical Ultrack hierarchy segmentation.

The experiment uses one CellFlow nucleus folder as a fixed test ground:

  hypotheses.h5 + contour/foreground maps -> two Ultrack candidate databases

Both branches then receive the same CellFlow node-probability scoring, linking,
ILP solving, export, and benchmark against curated tracked_labels.tif.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

NUCLEUS_DIR = Path(
    "/home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/analysis/pos00/2_nucleus"
)


class Tee:
    def __init__(self, stream, log_path: Path):
        self._stream = stream
        self._log = log_path.open("w", buffering=1)

    def write(self, data: str) -> None:
        self._stream.write(data)
        self._log.write(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def close(self) -> None:
        self._log.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _load_labels(path: Path, n_frames: int | None = None) -> np.ndarray:
    labels = np.asarray(tifffile.imread(str(path)), dtype=np.uint32)
    if n_frames is not None:
        labels = labels[:n_frames]
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    return labels


def _load_float_stack(path: Path, n_frames: int | None = None) -> np.ndarray:
    arr = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    if n_frames is not None:
        arr = arr[:n_frames]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D TYX stack at {path}, got {arr.shape}")
    return arr


def _cell_counts(labels: np.ndarray) -> list[int]:
    counts: list[int] = []
    for frame in labels:
        ids = np.unique(frame)
        counts.append(int(np.count_nonzero(ids[ids != 0])))
    return counts


def _track_lengths(labels: np.ndarray) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for t in range(labels.shape[0]):
        for label_id in np.unique(labels[t]):
            label_int = int(label_id)
            if label_int == 0:
                continue
            lengths[label_int] = lengths.get(label_int, 0) + 1
    return lengths


def _binary_iou(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left = np.asarray(lhs) > 0
    right = np.asarray(rhs) > 0
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum() / union)


def _best_iou_stats(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids != 0]
    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids != 0]
    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return {
            "gt_mean_best_iou": 1.0,
            "pred_mean_best_iou": 1.0,
            "gt_recall_iou_0_5": 1.0,
            "pred_precision_iou_0_5": 1.0,
        }
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return {
            "gt_mean_best_iou": 0.0,
            "pred_mean_best_iou": 0.0,
            "gt_recall_iou_0_5": 0.0,
            "pred_precision_iou_0_5": 0.0,
        }

    pred_area = np.bincount(pred.ravel())
    gt_area = np.bincount(gt.ravel())
    encoded = gt.ravel() * (int(pred.max()) + 1) + pred.ravel()
    pairs, inter = np.unique(encoded, return_counts=True)
    gt_pair = pairs // (int(pred.max()) + 1)
    pred_pair = pairs % (int(pred.max()) + 1)

    gt_best = {int(i): 0.0 for i in gt_ids}
    pred_best = {int(i): 0.0 for i in pred_ids}
    for g, p, intersection in zip(gt_pair, pred_pair, inter, strict=False):
        if g == 0 or p == 0:
            continue
        union = int(gt_area[g]) + int(pred_area[p]) - int(intersection)
        iou = float(intersection / union) if union else 0.0
        gt_best[int(g)] = max(gt_best[int(g)], iou)
        pred_best[int(p)] = max(pred_best[int(p)], iou)

    gt_vals = np.asarray(list(gt_best.values()), dtype=np.float32)
    pred_vals = np.asarray(list(pred_best.values()), dtype=np.float32)
    return {
        "gt_mean_best_iou": float(gt_vals.mean()) if gt_vals.size else 0.0,
        "pred_mean_best_iou": float(pred_vals.mean()) if pred_vals.size else 0.0,
        "gt_recall_iou_0_5": float((gt_vals >= 0.5).mean()) if gt_vals.size else 0.0,
        "pred_precision_iou_0_5": float((pred_vals >= 0.5).mean()) if pred_vals.size else 0.0,
    }


def _db_stats(working_dir: Path) -> dict[str, Any]:
    import sqlalchemy as sqla
    from sqlalchemy import func
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, NodeDB, OverlapDB

    engine = sqla.create_engine(f"sqlite:///{working_dir / 'data.db'}")
    with Session(engine) as session:
        node_count = int(session.query(NodeDB).count())
        overlap_count = int(session.query(OverlapDB).count())
        link_count = int(session.query(LinkDB).count())
        selected_count = int(session.query(NodeDB).where(NodeDB.selected).count())
        hierarchy_parent_count = int(session.query(NodeDB).where(NodeDB.hier_parent_id > 0).count())
        hierarchy_count_by_t = {
            int(t): int(n)
            for t, n in session.query(NodeDB.t, func.count(func.distinct(NodeDB.t_hier_id)))
            .group_by(NodeDB.t)
            .all()
        }
        nodes_by_t = {
            int(t): int(n)
            for t, n in session.query(NodeDB.t, func.count(NodeDB.id)).group_by(NodeDB.t).all()
        }
        selected_by_t = {
            int(t): int(n)
            for t, n in session.query(NodeDB.t, func.count(NodeDB.id))
            .where(NodeDB.selected)
            .group_by(NodeDB.t)
            .all()
        }
        area_stats = session.query(
            func.min(NodeDB.area),
            func.avg(NodeDB.area),
            func.max(NodeDB.area),
        ).one()

    engine.dispose()
    return {
        "nodes": node_count,
        "overlaps": overlap_count,
        "links": link_count,
        "selected_nodes": selected_count,
        "nodes_with_hier_parent": hierarchy_parent_count,
        "hierarchies_by_t": hierarchy_count_by_t,
        "nodes_by_t": nodes_by_t,
        "selected_by_t": selected_by_t,
        "area_min_mean_max": [float(v) if v is not None else None for v in area_stats],
    }


def _metrics_for_labels(labels: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    n = min(labels.shape[0], gt.shape[0])
    labels = labels[:n]
    gt = gt[:n]

    per_frame = []
    for t in range(n):
        stats = _best_iou_stats(labels[t], gt[t])
        stats.update(
            {
                "t": t,
                "binary_iou": _binary_iou(labels[t], gt[t]),
                "pred_count": _cell_counts(labels[t:t + 1])[0],
                "gt_count": _cell_counts(gt[t:t + 1])[0],
            }
        )
        per_frame.append(stats)

    return {
        "shape": list(labels.shape),
        "binary_iou_global": _binary_iou(labels, gt),
        "binary_iou_mean_frame": float(np.mean([f["binary_iou"] for f in per_frame])),
        "gt_mean_best_iou_mean_frame": float(np.mean([f["gt_mean_best_iou"] for f in per_frame])),
        "pred_mean_best_iou_mean_frame": float(np.mean([f["pred_mean_best_iou"] for f in per_frame])),
        "gt_recall_iou_0_5_mean_frame": float(np.mean([f["gt_recall_iou_0_5"] for f in per_frame])),
        "pred_precision_iou_0_5_mean_frame": float(np.mean([f["pred_precision_iou_0_5"] for f in per_frame])),
        "cell_counts": _cell_counts(labels),
        "track_lengths": _track_lengths(labels),
        "n_tracks": len(_track_lengths(labels)),
        "mean_track_length": float(np.mean(list(_track_lengths(labels).values()))) if _track_lengths(labels) else 0.0,
        "per_frame": per_frame,
    }


def _make_cfg(args: argparse.Namespace):
    from cellflow.tracking_ultrack.config import TrackingConfig

    return TrackingConfig(
        min_area=args.min_area,
        max_area=args.max_area,
        max_distance=args.max_distance,
        max_neighbors=args.max_neighbors,
        link_n_workers=args.link_n_workers,
        linking_mode=args.linking_mode,
        iou_weight=args.iou_weight,
        appear_weight=args.appear_weight,
        disappear_weight=args.disappear_weight,
        division_weight=args.division_weight,
        power=args.power,
        quality_exponent=args.quality_exponent,
        time_limit=args.time_limit,
    )


def _run_direct(
    run_dir: Path,
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> tuple[np.ndarray, dict[str, Any]]:
    from cellflow.tracking_ultrack.export import export_tracked_labels
    from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db
    from cellflow.tracking_ultrack.linking import run_linking
    from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs
    from cellflow.tracking_ultrack.solve import run_solve

    cfg = _make_cfg(args)
    max_partitions = None if args.max_partitions == 0 else args.max_partitions
    timings: dict[str, float] = {}

    t0 = time.monotonic()
    ingest_hypotheses_to_db(
        paths["hypotheses"],
        run_dir,
        cfg,
        overwrite=True,
        max_partitions=max_partitions,
        n_frames=args.n_frames,
    )
    timings["build_db"] = time.monotonic() - t0

    t0 = time.monotonic()
    score_report = write_seed_prior_node_probs(run_dir, paths["signal"], cfg)
    timings["score"] = time.monotonic() - t0

    t0 = time.monotonic()
    for step, total, label in run_linking(run_dir, cfg):
        print(f"      [{step}/{total}] {label}", flush=True)
    timings["link"] = time.monotonic() - t0

    t0 = time.monotonic()
    for step, total, label in run_solve(run_dir, cfg, overwrite=True):
        print(f"      [{step}/{total}] {label}", flush=True)
    timings["solve"] = time.monotonic() - t0

    t0 = time.monotonic()
    labels = export_tracked_labels(run_dir, cfg, run_dir / "tracked_labels.tif")
    timings["export"] = time.monotonic() - t0

    meta = {
        "branch": "direct_h5_ingest",
        "timings_sec": timings,
        "score_report": score_report.__dict__,
        "db_stats": _db_stats(run_dir),
    }
    return labels, meta


def _run_hierarchy(
    run_dir: Path,
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> tuple[np.ndarray, dict[str, Any]]:
    from cellflow.tracking_ultrack.export import export_tracked_labels
    from cellflow.tracking_ultrack.ingest import _build_ultrack_config
    from cellflow.tracking_ultrack.linking import run_linking
    from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs
    from cellflow.tracking_ultrack.solve import run_solve
    from ultrack.config.segmentationconfig import NAME_TO_WS_HIER
    from ultrack.core.segmentation.processing import segment

    cfg = _make_cfg(args)
    ultrack_cfg = _build_ultrack_config(cfg, run_dir)
    ultrack_cfg.segmentation_config.min_area = args.min_area
    ultrack_cfg.segmentation_config.max_area = args.max_area
    ultrack_cfg.segmentation_config.threshold = args.foreground_threshold
    ultrack_cfg.segmentation_config.min_frontier = args.min_frontier
    ultrack_cfg.segmentation_config.n_workers = args.seg_n_workers
    ultrack_cfg.segmentation_config.ws_hierarchy = NAME_TO_WS_HIER[args.ws_hierarchy]

    foreground = _load_float_stack(paths["foreground"], args.n_frames)
    contours = _load_float_stack(paths["contours"], args.n_frames)
    timings: dict[str, float] = {}

    t0 = time.monotonic()
    segment(
        foreground,
        contours,
        ultrack_cfg,
        max_segments_per_time=cfg.max_segments_per_time,
        overwrite=True,
    )
    timings["build_db"] = time.monotonic() - t0

    t0 = time.monotonic()
    score_report = write_seed_prior_node_probs(run_dir, paths["signal"], cfg)
    timings["score"] = time.monotonic() - t0

    t0 = time.monotonic()
    for step, total, label in run_linking(run_dir, cfg):
        print(f"      [{step}/{total}] {label}", flush=True)
    timings["link"] = time.monotonic() - t0

    t0 = time.monotonic()
    for step, total, label in run_solve(run_dir, cfg, overwrite=True):
        print(f"      [{step}/{total}] {label}", flush=True)
    timings["solve"] = time.monotonic() - t0

    t0 = time.monotonic()
    labels = export_tracked_labels(run_dir, cfg, run_dir / "tracked_labels.tif")
    timings["export"] = time.monotonic() - t0

    meta = {
        "branch": "canonical_hierarchy_segment",
        "timings_sec": timings,
        "score_report": score_report.__dict__,
        "db_stats": _db_stats(run_dir),
        "segmentation_config": ultrack_cfg.segmentation_config.dict(),
    }
    return labels, meta


def _write_text_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "Ultrack Hierarchy vs Direct H5 Experiment",
        "=" * 44,
        f"results_dir: {report['results_dir']}",
        f"n_frames: {report['args']['n_frames']}",
        f"linking_mode: {report['args']['linking_mode']}",
        f"node_prob_scoring: enabled for both branches",
        "",
    ]
    for name in ("direct_h5_ingest", "canonical_hierarchy_segment"):
        branch = report["branches"][name]
        metrics = branch["metrics"]
        db = branch["db_stats"]
        lines += [
            name,
            "-" * len(name),
            f"  output: {branch['output_tif']}",
            f"  nodes={db['nodes']} overlaps={db['overlaps']} links={db['links']} selected={db['selected_nodes']}",
            f"  nodes_with_hier_parent={db['nodes_with_hier_parent']}",
            f"  global_binary_iou={metrics['binary_iou_global']:.4f}",
            f"  mean_frame_binary_iou={metrics['binary_iou_mean_frame']:.4f}",
            f"  mean_gt_best_iou={metrics['gt_mean_best_iou_mean_frame']:.4f}",
            f"  gt_recall_iou_0_5={metrics['gt_recall_iou_0_5_mean_frame']:.4f}",
            f"  pred_precision_iou_0_5={metrics['pred_precision_iou_0_5_mean_frame']:.4f}",
            f"  n_tracks={metrics['n_tracks']} mean_track_length={metrics['mean_track_length']:.2f}",
            f"  timings_sec={branch['timings_sec']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=50)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--linking-mode", choices=["default", "shape"], default="default")
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--max-area", type=int, default=1_000_000)
    parser.add_argument("--max-partitions", type=int, default=30, help="Direct ingest cap; 0 means all partitions.")
    parser.add_argument("--max-distance", type=float, default=15.0)
    parser.add_argument("--max-neighbors", type=int, default=5)
    parser.add_argument("--link-n-workers", type=int, default=8)
    parser.add_argument("--seg-n-workers", type=int, default=1)
    parser.add_argument("--iou-weight", type=float, default=1.0)
    parser.add_argument("--appear-weight", type=float, default=-0.1)
    parser.add_argument("--disappear-weight", type=float, default=-0.1)
    parser.add_argument("--division-weight", type=float, default=-0.001)
    parser.add_argument("--power", type=float, default=4.0)
    parser.add_argument("--quality-exponent", type=float, default=8.0)
    parser.add_argument("--time-limit", type=int, default=36000)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--min-frontier", type=float, default=0.0)
    parser.add_argument("--ws-hierarchy", choices=["area", "dynamics", "volume"], default="area")
    args = parser.parse_args()

    pos_dir = NUCLEUS_DIR.parent
    paths = {
        "hypotheses": NUCLEUS_DIR / "hypotheses.h5",
        "contours": NUCLEUS_DIR / "contour_maps.tif",
        "foreground": NUCLEUS_DIR / "foreground_maps.tif",
        "gt": NUCLEUS_DIR / "tracked_labels.tif",
        "signal": pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif",
    }
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {key}: {path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir or (NUCLEUS_DIR / "ultrack_hierarchy_vs_direct_experiment" / timestamp)
    if results_dir.exists() and args.overwrite_results:
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=False)

    log_path = results_dir / "run.log"
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    try:
        print(f"Results dir: {results_dir}", flush=True)
        print(f"Using signal scoring image: {paths['signal']}", flush=True)
        print(f"Arguments: {vars(args)}", flush=True)

        gt = _load_labels(paths["gt"], args.n_frames)
        report: dict[str, Any] = {
            "results_dir": str(results_dir),
            "inputs": {k: str(v) for k, v in paths.items()},
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "ground_truth": _metrics_for_labels(gt, gt),
            "branches": {},
        }

        direct_dir = results_dir / "direct_h5_ingest"
        direct_dir.mkdir()
        print("\n[direct_h5_ingest] running", flush=True)
        direct_labels, direct_meta = _run_direct(direct_dir, args, paths)
        direct_metrics = _metrics_for_labels(direct_labels, gt)
        report["branches"]["direct_h5_ingest"] = {
            **direct_meta,
            "metrics": direct_metrics,
            "output_tif": str(direct_dir / "tracked_labels.tif"),
        }
        (direct_dir / "metrics.json").write_text(json.dumps(report["branches"]["direct_h5_ingest"], indent=2), encoding="utf-8")

        hierarchy_dir = results_dir / "canonical_hierarchy_segment"
        hierarchy_dir.mkdir()
        print("\n[canonical_hierarchy_segment] running", flush=True)
        hierarchy_labels, hierarchy_meta = _run_hierarchy(hierarchy_dir, args, paths)
        hierarchy_metrics = _metrics_for_labels(hierarchy_labels, gt)
        report["branches"]["canonical_hierarchy_segment"] = {
            **hierarchy_meta,
            "metrics": hierarchy_metrics,
            "output_tif": str(hierarchy_dir / "tracked_labels.tif"),
        }
        (hierarchy_dir / "metrics.json").write_text(
            json.dumps(report["branches"]["canonical_hierarchy_segment"], indent=2),
            encoding="utf-8",
        )

        (results_dir / "experiment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_text_summary(results_dir / "experiment_summary.txt", report)
        print(f"\nSummary written to {results_dir / 'experiment_summary.txt'}", flush=True)
        print((results_dir / "experiment_summary.txt").read_text(encoding="utf-8"), flush=True)
    finally:
        sys.stdout = tee._stream
        tee.close()


if __name__ == "__main__":
    main()
