#!/usr/bin/env python
"""Run the v2 progressive Ultrack pipeline.

This generates continuous foreground scores plus contour maps, builds one
Ultrack database, and writes a compact JSON report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.progressive_merge import (
    build_progressive_ultrack_database,
    write_progressive_inputs,
)
from cellflow.tracking_ultrack.solve import run_solve

DEFAULT_POS_DIR = Path(
    "/home/aruppel/Data/"
    "2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/analysis/pos00"
)


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


def _default_masks_path(pos_dir: Path) -> Path:
    candidates = [
        pos_dir / "1_cellpose" / "nucleus_masks.tif",
        pos_dir / "1_cellpose" / "masks.tif",
        pos_dir / "2_nucleus" / "masks.tif",
        pos_dir / "2_nucleus" / "tracked_labels.tif",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not locate Cellpose masks automatically; pass --masks-path. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def _count_database(db_path: Path) -> dict[str, int | None]:
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, OverlapDB

    if not db_path.exists():
        return {
            "node_count": None,
            "hierarchy_count": None,
            "overlap_count": None,
            "track_count": None,
        }

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            return {
                "node_count": int(session.query(NodeDB).count()),
                "hierarchy_count": int(session.query(NodeDB.t_hier_id).distinct().count()),
                "overlap_count": int(session.query(OverlapDB).count()),
                "track_count": None,
            }
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run continuous foreground / progressive hierarchy Ultrack experiment"
    )
    parser.add_argument("--pos-dir", type=Path, default=DEFAULT_POS_DIR)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Defaults to POS_DIR / 2_nucleus / ultrack_progressive_sweep_experiment",
    )
    parser.add_argument("--prob-path", type=Path, default=None)
    parser.add_argument("--masks-path", type=Path, default=None)
    parser.add_argument("--working-dir", type=Path, default=None)
    parser.add_argument("--seg-foreground-threshold", type=float, default=0.3)
    parser.add_argument("--bias", type=float, default=-0.3)
    parser.add_argument("--max-segments-per-time", type=int, default=None)
    parser.add_argument("--no-solve", action="store_true")
    parser.add_argument("--seg-n-workers", type=int, default=1)
    parser.add_argument("--seg-min-area", type=int, default=300)
    parser.add_argument("--seg-max-area", type=int, default=100_000)
    parser.add_argument("--max-distance", type=float, default=15.0)
    parser.add_argument("--max-neighbors", type=int, default=5)
    parser.add_argument("--link-n-workers", type=int, default=None)
    parser.add_argument("--time-limit", type=int, default=36_000)
    parser.add_argument("--solution-gap", type=float, default=0.001)

    args = parser.parse_args()

    pos_dir: Path = args.pos_dir
    experiment_dir = (
        args.experiment_dir
        or pos_dir / "2_nucleus" / "ultrack_progressive_sweep_experiment"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    working_dir = args.working_dir or experiment_dir / "ultrack_workdir"

    prob_path = args.prob_path or pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif"
    masks_path = args.masks_path or _default_masks_path(pos_dir)
    if not prob_path.exists():
        raise FileNotFoundError(prob_path)
    if not masks_path.exists():
        raise FileNotFoundError(masks_path)

    cfg_kwargs: dict[str, Any] = {
        "seg_foreground_threshold": args.seg_foreground_threshold,
        "bias": args.bias,
        "seg_n_workers": args.seg_n_workers,
        "seg_min_area": args.seg_min_area,
        "seg_max_area": args.seg_max_area,
        "max_distance": args.max_distance,
        "max_neighbors": args.max_neighbors,
        "time_limit": args.time_limit,
        "solution_gap": args.solution_gap,
    }
    if args.max_segments_per_time is not None:
        cfg_kwargs["max_segments_per_time"] = args.max_segments_per_time
    if args.link_n_workers is not None:
        cfg_kwargs["link_n_workers"] = args.link_n_workers
    cfg = TrackingConfig(**cfg_kwargs)

    print("=" * 60)
    print("Ultrack Progressive Sweep Experiment")
    print("=" * 60)
    print(f"pos_dir:    {pos_dir}")
    print(f"experiment: {experiment_dir}")
    print(f"prob_path:  {prob_path}")
    print(f"masks_path: {masks_path}")
    print(f"workdir:    {working_dir}")
    print(f"config:     {cfg.model_dump()}")

    timings: dict[str, float] = {}

    with _time_stage("generate_inputs", timings):
        foreground_path, contour_path = write_progressive_inputs(
            prob_path, masks_path, experiment_dir / "inputs"
        )

    with _time_stage("build_database", timings):
        build_report = build_progressive_ultrack_database(
            foreground_scores_path=foreground_path,
            contour_maps_path=contour_path,
            nucleus_prob_zavg_path=foreground_path,
            working_dir=working_dir,
            cfg=cfg,
            progress_cb=lambda msg: print(f"    {msg}", flush=True),
        )

    solve_runtime = None
    if not args.no_solve:
        with _time_stage("solve", timings):
            for step, total, label in run_solve(working_dir, cfg, overwrite=True):
                print(f"    [{step}/{total}] {label}", flush=True)
        solve_runtime = timings["solve"]

    db_path = working_dir / "data.db"
    counts = _count_database(db_path)
    report = {
        "pos_dir": str(pos_dir),
        "experiment_dir": str(experiment_dir),
        "working_dir": str(working_dir),
        "input_paths": {
            "prob_3dt": str(prob_path),
            "masks": str(masks_path),
            "foreground_scores": str(foreground_path),
            "contour_maps": str(contour_path),
        },
        "config": cfg.model_dump(),
        "timings_seconds": timings,
        "solve_runtime_seconds": solve_runtime,
        "build_report": {
            "real_nodes": build_report.real_nodes,
            "skipped_validated": build_report.skipped_validated,
            "fake_nodes": build_report.fake_nodes,
            "overlaps_added": build_report.overlaps_added,
            "scored_nodes": build_report.scored_nodes,
            "seed_nodes": build_report.seed_nodes,
            "boosted_edges": build_report.boosted_edges,
        },
        **counts,
    }

    report_path = experiment_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote report: {report_path}")
    print(f"DB:           {db_path}")


if __name__ == "__main__":
    main()
