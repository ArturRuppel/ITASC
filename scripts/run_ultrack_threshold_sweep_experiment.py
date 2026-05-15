#!/usr/bin/env python
"""Ultrack foreground / contour threshold sweep experiment.

Generates variant foreground and contour inputs from Cellpose probability maps,
runs Ultrack segmentation for all combinations, merges the candidate databases,
and runs downstream scoring and linking.

Usage (from repo root):

    python scripts/run_ultrack_threshold_sweep_experiment.py
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
from cellflow.tracking_ultrack.db_build import _build_ultrack_config, _run_ultrack_segment
from cellflow.tracking_ultrack.linking import run_linking
from cellflow.tracking_ultrack.multi_threshold import merge_ultrack_databases
from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_POS_DIR = Path(
    "/home/aruppel/Data/"
    "2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/analysis/pos00"
)

DEFAULT_XY_SHAPE = (512, 512)

FOREGROUND_THRESHOLDS = [0.1, 0.3, 0.5]
CONTOUR_THRESHOLDS = [None, 0.5]  # None == raw

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _suffix(thr: float | None) -> str:
    return "raw" if thr is None else f"{thr:.1f}"


def _variant_name(fg_thr: float, contour_thr: float | None) -> str:
    return f"fg_{fg_thr:.1f}_contour_{_suffix(contour_thr)}"


def _generate_inputs(
    pos_dir: Path,
    experiment_dir: Path,
    sigmoid_k: float,
    sigmoid_midpoint: float,
    fg_thresholds: list[float],
    contour_thresholds: list[float | None],
    normalize_contours: bool,
    n_frames: int | None,
) -> dict[str, Any]:
    """Create ``inputs/`` directory with transformed foregrounds and contours."""
    inputs_dir = experiment_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    nucleus_prob_path = pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif"
    contour_src_path = pos_dir / "2_nucleus" / "contour_maps.tif"

    if not nucleus_prob_path.exists():
        raise FileNotFoundError(nucleus_prob_path)
    if not contour_src_path.exists():
        raise FileNotFoundError(contour_src_path)

    # --- nucleus probability → sigmoid z-average --------------------------------
    sigmoid_zavg_path = inputs_dir / "nucleus_prob_sigmoid_zavg.tif"
    if sigmoid_zavg_path.exists():
        print(f"  Using existing {sigmoid_zavg_path.name}")
        sigmoid_zavg = np.asarray(tifffile.imread(sigmoid_zavg_path), dtype=np.float32)
    else:
        print(f"  Loading {nucleus_prob_path} ...")
        prob = np.asarray(tifffile.imread(nucleus_prob_path), dtype=np.float32)
        print(f"    prob shape={prob.shape}, range=[{prob.min():.3f}, {prob.max():.3f}]")
        if prob.ndim == 4:
            sigmoid = 1.0 / (1.0 + np.exp(-sigmoid_k * (prob - sigmoid_midpoint)))
            sigmoid_zavg = sigmoid.mean(axis=1)
        elif prob.ndim == 3:
            sigmoid_zavg = 1.0 / (
                1.0 + np.exp(-sigmoid_k * (prob - sigmoid_midpoint))
            )
        else:
            raise ValueError(f"Expected 3D or 4D prob map, got {prob.shape}")
        print(
            f"    sigmoid z-avg shape={sigmoid_zavg.shape}, "
            f"range=[{sigmoid_zavg.min():.4f}, {sigmoid_zavg.max():.4f}]",
        )
        tifffile.imwrite(sigmoid_zavg_path, sigmoid_zavg)
        print(f"    Saved {sigmoid_zavg_path.name}")

    if n_frames is not None:
        sigmoid_zavg = sigmoid_zavg[:n_frames]

    # --- foreground variants ----------------------------------------------------
    expected_shape = sigmoid_zavg.shape
    fg_paths: dict[float, Path] = {}
    for thr in fg_thresholds:
        fg_path = inputs_dir / f"foreground_thr_{thr:.1f}.tif"
        fg_paths[thr] = fg_path
        if fg_path.exists():
            existing = tifffile.imread(fg_path)
            if existing.shape == expected_shape:
                print(f"  Using existing {fg_path.name}")
                continue
            print(
                f"  Regenerating {fg_path.name} — shape mismatch "
                f"({existing.shape} vs {expected_shape})"
            )
        fg = (sigmoid_zavg >= thr).astype(np.uint8)
        print(f"  Saving {fg_path.name} — threshold={thr}, n_fg_pixels={int(fg.sum())}")
        tifffile.imwrite(fg_path, fg)

    # --- contour variants -------------------------------------------------------
    print(f"  Loading {contour_src_path} ...")
    contours = np.asarray(tifffile.imread(contour_src_path), dtype=np.float32)
    if contours.ndim == 4 and contours.shape[1] == 1:
        contours = contours[:, 0]
    print(f"    contours shape={contours.shape}, range=[{contours.min():.4f}, {contours.max():.4f}]")
    if n_frames is not None:
        contours = contours[:n_frames]

    if normalize_contours:
        cmin, cmax = contours.min(), contours.max()
        if cmax > cmin:
            contours = (contours - cmin) / (cmax - cmin)
            print(f"    normalized contours range=[{contours.min():.4f}, {contours.max():.4f}]")
        else:
            print("    WARNING: contour map has zero dynamic range; skipping normalization")

    contour_paths: dict[float | None, Path] = {}
    contour_paths[None] = inputs_dir / "contours_raw.tif"
    if contour_paths[None].exists():
        existing = tifffile.imread(contour_paths[None])
        if existing.shape == contours.shape:
            print(f"  Using existing {contour_paths[None].name}")
        else:
            print(
                f"  Regenerating {contour_paths[None].name} — shape mismatch "
                f"({existing.shape} vs {contours.shape})"
            )
            tifffile.imwrite(contour_paths[None], contours.astype(np.float32))
    else:
        tifffile.imwrite(contour_paths[None], contours.astype(np.float32))
        print(f"  Saved {contour_paths[None].name}")

    for thr in contour_thresholds:
        if thr is None:
            continue
        c_path = inputs_dir / f"contours_thr_{thr:.1f}.tif"
        contour_paths[thr] = c_path
        if c_path.exists():
            existing = tifffile.imread(c_path)
            if existing.shape == contours.shape:
                print(f"  Using existing {c_path.name}")
                continue
            print(
                f"  Regenerating {c_path.name} — shape mismatch "
                f"({existing.shape} vs {contours.shape})"
            )
        c_thr = np.where(contours < thr, 0.0, contours).astype(np.float32)
        print(f"  Saving {c_path.name} — threshold={thr}, n_nonzero={int((c_thr > 0).sum())}")
        tifffile.imwrite(c_path, c_thr)

    # --- manifest ---------------------------------------------------------------
    manifest = {
        "pos_dir": str(pos_dir),
        "nucleus_prob_3dt": str(nucleus_prob_path),
        "contour_maps": str(contour_src_path),
        "sigmoid": {"k": sigmoid_k, "midpoint": sigmoid_midpoint},
        "foreground_thresholds": fg_thresholds,
        "contour_thresholds": [thr if thr is None else float(thr) for thr in contour_thresholds],
        "normalize_contours": normalize_contours,
        "foreground_paths": {f"{k:.1f}": str(v) for k, v in fg_paths.items()},
        "contour_paths": {"raw" if k is None else f"{k:.1f}": str(v) for k, v in contour_paths.items()},
    }
    manifest_path = inputs_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Saved manifest.json")

    return {
        "sigmoid_zavg": sigmoid_zavg,
        "fg_paths": fg_paths,
        "contour_paths": contour_paths,
        "frame_shape": (int(sigmoid_zavg.shape[1]), int(sigmoid_zavg.shape[2])),
    }


def _build_variant_db(
    fg_path: Path,
    contour_path: Path,
    variant_dir: Path,
    cfg: TrackingConfig,
) -> Path:
    """Run Ultrack segmentation for one variant.  Returns path to ``data.db``."""
    variant_dir.mkdir(parents=True, exist_ok=True)
    db_path = variant_dir / "data.db"
    fg = np.asarray(tifffile.imread(fg_path), dtype=np.float32)
    contours = np.asarray(tifffile.imread(contour_path), dtype=np.float32)

    if db_path.exists():
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB

        engine = sqla.create_engine(f"sqlite:///{db_path}")
        try:
            with Session(engine) as session:
                max_t = session.query(sqla.func.max(NodeDB.t)).scalar()
        finally:
            engine.dispose()
        expected_t = fg.shape[0] - 1
        if max_t == expected_t:
            print(f"    Skipping — {db_path} already exists")
            return db_path
        print(f"    Rebuilding — frame count mismatch (DB t={max_t} vs input {expected_t})")

    ultrack_cfg = _build_ultrack_config(cfg, variant_dir)
    _run_ultrack_segment(fg, contours, ultrack_cfg, cfg)

    if not db_path.exists():
        raise RuntimeError(f"Ultrack segment did not create {db_path}")
    return db_path


def _count_nodes_and_overlaps(db_path: Path) -> dict[str, int]:
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, OverlapDB

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            node_count = session.query(NodeDB).count()
            overlap_count = session.query(OverlapDB).count()
    finally:
        engine.dispose()
    return {"node_count": node_count, "overlap_count": overlap_count}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ultrack foreground/contour threshold sweep experiment"
    )
    parser.add_argument("--pos-dir", type=Path, default=DEFAULT_POS_DIR)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Defaults to POS_DIR / 2_nucleus / ultrack_threshold_sweep_experiment",
    )
    parser.add_argument("--n-frames", type=int, default=None)
    parser.add_argument(
        "--fg-thresholds",
        type=float,
        nargs="+",
        default=FOREGROUND_THRESHOLDS,
    )
    parser.add_argument(
        "--contour-thresholds",
        type=str,
        nargs="+",
        default=["raw", "0.3"],
        help='Use "raw" for unthresholded; otherwise a float (e.g. 0.3)',
    )
    parser.add_argument("--sigmoid-k", type=float, default=1.0)
    parser.add_argument("--sigmoid-midpoint", type=float, default=0.0)
    parser.add_argument(
        "--normalize-contours",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Min-max normalise contour maps before thresholding (default: True)",
    )
    parser.add_argument(
        "--score-image",
        type=Path,
        default=None,
        help="Intensity image for node scoring. Defaults to inputs/nucleus_prob_sigmoid_zavg.tif",
    )

    # TrackingConfig overrides
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--max-area", type=int, default=1_000_000)
    parser.add_argument("--max-distance", type=float, default=15.0)
    parser.add_argument("--max-neighbors", type=int, default=5)
    parser.add_argument("--linking-mode", choices=["default", "shape"], default="default")
    parser.add_argument("--iou-weight", type=float, default=1.0)
    parser.add_argument("--min-link-iou", type=float, default=0.1)
    parser.add_argument("--link-n-workers", type=int, default=None)
    parser.add_argument("--time-limit", type=int, default=36_000)
    parser.add_argument("--appear-weight", type=float, default=-0.001)
    parser.add_argument("--disappear-weight", type=float, default=-0.001)
    parser.add_argument("--division-weight", type=float, default=-0.001)
    parser.add_argument("--power", type=float, default=4.0)
    parser.add_argument("--solution-gap", type=float, default=0.001)
    parser.add_argument("--seg-foreground-threshold", type=float, default=0.5)
    parser.add_argument("--seg-n-workers", type=int, default=1)
    parser.add_argument("--seg-min-area", type=int, default=300)
    parser.add_argument("--seg-max-area", type=int, default=100_000)
    parser.add_argument(
        "--skip-variants",
        action="store_true",
        help="Skip segmentation if all variant DBs already exist",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merge if merged DB already exists",
    )
    parser.add_argument(
        "--skip-score-link",
        action="store_true",
        help="Skip scoring and linking if already present",
    )

    args = parser.parse_args()

    pos_dir: Path = args.pos_dir
    experiment_dir: Path = (
        args.experiment_dir or pos_dir / "2_nucleus" / "ultrack_threshold_sweep_experiment"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # --- parse contour thresholds -----------------------------------------------
    contour_thresholds: list[float | None] = []
    for s in args.contour_thresholds:
        if s.lower() == "raw":
            contour_thresholds.append(None)
        else:
            contour_thresholds.append(float(s))

    fg_thresholds = [float(v) for v in args.fg_thresholds]

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
        "seg_foreground_threshold": args.seg_foreground_threshold,
        "seg_n_workers": args.seg_n_workers,
        "seg_min_area": args.seg_min_area,
        "seg_max_area": args.seg_max_area,
    }
    if args.link_n_workers is not None:
        cfg_kwargs["link_n_workers"] = args.link_n_workers
    cfg = TrackingConfig(**cfg_kwargs)

    print("=" * 60)
    print("Ultrack Foreground/Contour Threshold Sweep Experiment")
    print("=" * 60)
    print(f"pos_dir:      {pos_dir}")
    print(f"experiment:   {experiment_dir}")
    print(f"n_frames:     {args.n_frames if args.n_frames is not None else 'all'}")
    print(f"fg_thrs:      {fg_thresholds}")
    print(f"contour_thrs: {contour_thresholds}")
    print(f"normalize_contours: {args.normalize_contours}")
    print(f"TrackingConfig: {cfg.model_dump()}")
    print()

    timings: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Generate inputs
    # ------------------------------------------------------------------
    with _time_stage("generate_inputs", timings):
        input_info = _generate_inputs(
            pos_dir=pos_dir,
            experiment_dir=experiment_dir,
            sigmoid_k=args.sigmoid_k,
            sigmoid_midpoint=args.sigmoid_midpoint,
            fg_thresholds=fg_thresholds,
            contour_thresholds=contour_thresholds,
            normalize_contours=args.normalize_contours,
            n_frames=args.n_frames,
        )

    frame_shape = input_info["frame_shape"]
    print(f"Frame shape detected: {frame_shape}")

    # ------------------------------------------------------------------
    # 2. Variant segmentation
    # ------------------------------------------------------------------
    variants_dir = experiment_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    variant_reports: list[dict[str, Any]] = []
    variant_db_paths: list[Path] = []
    all_exist = True

    for fg_thr in fg_thresholds:
        for ct in contour_thresholds:
            name = _variant_name(fg_thr, ct)
            vdir = variants_dir / name
            v_db = vdir / "data.db"
            if not v_db.exists():
                all_exist = False

    if args.skip_variants and all_exist:
        print("\n[*] Skipping variant segmentation (--skip-variants and all DBs exist)")
        for fg_thr in fg_thresholds:
            for ct in contour_thresholds:
                name = _variant_name(fg_thr, ct)
                vdir = variants_dir / name
                v_db = vdir / "data.db"
                variant_db_paths.append(v_db)
                counts = _count_nodes_and_overlaps(v_db)
                variant_reports.append(
                    {
                        "name": name,
                        "foreground_path": str(input_info["fg_paths"][fg_thr]),
                        "contour_path": str(input_info["contour_paths"][ct]),
                        "db_path": str(v_db),
                        **counts,
                    }
                )
    else:
        with _time_stage("variant_segmentation", timings):
            for fg_thr in fg_thresholds:
                for ct in contour_thresholds:
                    name = _variant_name(fg_thr, ct)
                    vdir = variants_dir / name
                    print(f"\n  Variant: {name}")
                    v_db = _build_variant_db(
                        input_info["fg_paths"][fg_thr],
                        input_info["contour_paths"][ct],
                        vdir,
                        cfg,
                    )
                    variant_db_paths.append(v_db)
                    counts = _count_nodes_and_overlaps(v_db)
                    print(f"    nodes={counts['node_count']} overlaps={counts['overlap_count']}")
                    variant_reports.append(
                        {
                            "name": name,
                            "foreground_path": str(input_info["fg_paths"][fg_thr]),
                            "contour_path": str(input_info["contour_paths"][ct]),
                            "db_path": str(v_db),
                            **counts,
                        }
                    )

    # ------------------------------------------------------------------
    # 3. Merge
    # ------------------------------------------------------------------
    merged_dir = experiment_dir / "merged_ultrack_workdir"
    merged_db = merged_dir / "data.db"

    if args.skip_merge and merged_db.exists():
        print("\n[*] Skipping merge (--skip-merge and merged DB exists)")
        merge_report = None
    else:
        with _time_stage("merge_databases", timings):
            merge_report = merge_ultrack_databases(
                source_db_paths=variant_db_paths,
                output_db_path=merged_db,
                frame_shape=frame_shape,
                progress_cb=lambda msg: print(f"    {msg}", flush=True),
            )
            print(
                f"\n    Merged {merge_report.source_count} sources, "
                f"{merge_report.total_nodes} total nodes, "
                f"cross-source overlaps={merge_report.cross_source_overlaps}"
            )

    # ------------------------------------------------------------------
    # 4. Score node probabilities
    # ------------------------------------------------------------------
    score_image = args.score_image or (experiment_dir / "inputs" / "nucleus_prob_sigmoid_zavg.tif")
    if not score_image.exists():
        raise FileNotFoundError(f"Score image not found: {score_image}")

    score_report = None

    if args.skip_score_link and merged_db.exists():
        # Check whether LinkDB already has rows — simple heuristic
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import LinkDB

        engine = sqla.create_engine(f"sqlite:///{merged_db}")
        try:
            with Session(engine) as session:
                has_links = session.query(LinkDB).first() is not None
        finally:
            engine.dispose()

        if has_links:
            print("\n[*] Skipping score+link (--skip-score-link and links present)")
        else:
            has_links = False
    else:
        has_links = False

    if not (args.skip_score_link and has_links):
        with _time_stage("score_nodes", timings):
            score_report = write_seed_prior_node_probs(merged_dir, score_image, cfg)
            print(
                f"    Scored {score_report.scored} node(s) using {score_report.seeds} seed node(s)",
            )

        # ------------------------------------------------------------------
        # 5. Linking
        # ------------------------------------------------------------------
        with _time_stage("linking", timings):
            for step, total, label in run_linking(merged_dir, cfg, overwrite=True):
                print(f"    [{step}/{total}] {label}", flush=True)

    # ------------------------------------------------------------------
    # 6. Reports
    # ------------------------------------------------------------------
    report = {
        "pos_dir": str(pos_dir),
        "experiment_dir": str(experiment_dir),
        "foreground_thresholds": fg_thresholds,
        "contour_thresholds": [
            thr if thr is None else float(thr) for thr in contour_thresholds
        ],
        "normalize_contours": args.normalize_contours,
        "n_frames": args.n_frames,
        "config": cfg.model_dump(),
        "timings_seconds": timings,
        "variants": variant_reports,
        "merged_db_path": str(merged_db),
    }

    if merge_report is not None:
        report["merged_node_count"] = merge_report.total_nodes
        report["merged_nodes_per_source"] = merge_report.nodes_per_source
        report["merged_within_source_overlaps"] = merge_report.within_source_overlaps
        report["merged_cross_source_overlaps"] = merge_report.cross_source_overlaps
    else:
        report["merged_node_count"] = None
        report["merged_nodes_per_source"] = None
        report["merged_within_source_overlaps"] = None
        report["merged_cross_source_overlaps"] = None

    if score_report is not None:
        report["score_report"] = {
            "scored": score_report.scored,
            "seeds": score_report.seeds,
        }
    else:
        report["score_report"] = None

    report_path = experiment_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote report: {report_path}")
    print(f"Merged DB:    {merged_db}")


if __name__ == "__main__":
    main()
