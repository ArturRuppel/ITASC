#!/usr/bin/env python
"""Phase 2 benchmark: end-to-end Ultrack ILP tracking on real data.

Runs ingest → link → solve → export on the first N frames of a real
hypotheses.h5, then compares against a ground-truth tracked_labels.tif.

Usage (from repo root, inside cellflow env):
    python scripts/benchmark_ultrack_phase2.py [--n-frames 10] [--linking-mode default|iou]

Outputs:
    <working_dir>/tracked_labels.tif       — new ILP-tracked labelmap
    <working_dir>/benchmark_report.txt     — per-frame + track-length comparison
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

HYPOTHESES_H5 = Path(
    "/home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/v2/pos00/2_nucleus/hypotheses.h5"
)
GROUND_TRUTH_TIF = Path(
    "/home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/v2/pos00/2_nucleus/tracked_labels.tif"
)
WORKING_DIR = Path(
    "/home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/v2/pos00/2_nucleus/ultrack_phase2"
)


def _crop_h5_to_n_frames(src: Path, dst: Path, n_frames: int) -> None:
    """Write a copy of src with only the first n_frames timepoints."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as f_in, h5py.File(dst, "w") as f_out:
        for attr_key in f_in.attrs:
            f_out.attrs[attr_key] = f_in.attrs[attr_key]

        root_in = f_in["hypotheses"]
        root_out = f_out.require_group("hypotheses")
        t_keys = sorted(k for k in root_in.keys() if k.startswith("t"))[:n_frames]
        for t_key in t_keys:
            t_grp_in = root_in[t_key]
            t_grp_out = root_out.require_group(t_key)
            for p_key in t_grp_in.keys():
                p_grp_in = t_grp_in[p_key]
                p_grp_out = t_grp_out.require_group(p_key)
                p_grp_out.create_dataset("labels", data=p_grp_in["labels"][:], compression="gzip", compression_opts=4)
                for attr_key in p_grp_in.attrs:
                    p_grp_out.attrs[attr_key] = p_grp_in.attrs[attr_key]


def _cell_count_per_frame(labels: np.ndarray) -> list[int]:
    """Return number of unique non-zero labels per frame (axis 0)."""
    counts = []
    for t in range(labels.shape[0]):
        frame = labels[t]
        counts.append(int(np.count_nonzero(np.unique(frame)[1:]) if frame.max() > 0 else 0))
    return counts


def _track_lengths(labels: np.ndarray) -> dict[int, int]:
    """Return {track_id: n_frames_present} for all non-zero IDs."""
    from collections import defaultdict
    lengths: dict[int, int] = defaultdict(int)
    for t in range(labels.shape[0]):
        for uid in np.unique(labels[t]):
            if uid != 0:
                lengths[int(uid)] += 1
    return dict(lengths)


def _distribution_summary(lengths: dict[int, int]) -> str:
    if not lengths:
        return "  (no tracks)"
    vals = list(lengths.values())
    return (
        f"  n_tracks={len(vals)}, "
        f"mean={np.mean(vals):.1f}, "
        f"median={np.median(vals):.1f}, "
        f"min={min(vals)}, "
        f"max={max(vals)}, "
        f"full_length={sum(v == max(vals) for v in vals)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=10)
    parser.add_argument("--linking-mode", choices=["default", "iou"], default="default")
    parser.add_argument("--min-area", type=int, default=300)
    parser.add_argument("--max-partitions", type=int, default=30,
                        help="Cap partitions per frame (default: 30; use 0 for all 176)")
    parser.add_argument("--appear-weight", type=float, default=-0.1,
                        help="ILP penalty per track appearance (default: -0.1)")
    parser.add_argument("--disappear-weight", type=float, default=-0.1,
                        help="ILP penalty per track disappearance (default: -0.1)")
    args = parser.parse_args()

    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db
    from cellflow.tracking_ultrack.linking import run_linking
    from cellflow.tracking_ultrack.solve import run_solve
    from cellflow.tracking_ultrack.export import export_tracked_labels

    n = args.n_frames
    mode = args.linking_mode
    max_partitions = args.max_partitions or None
    p_tag = f"p{max_partitions}" if max_partitions else "pall"
    wd = WORKING_DIR / f"n{n}_{mode}_{p_tag}_a{args.appear_weight}_mn{args.min_area}"
    wd.mkdir(parents=True, exist_ok=True)

    cropped_h5 = wd / "hypotheses_crop.h5"
    output_tif = wd / "tracked_labels.tif"
    report_path = wd / "benchmark_report.txt"

    print(f"=== Phase 2 Benchmark  (n_frames={n}, linking_mode={mode}) ===")
    print(f"Working dir: {wd}")

    # ---- Crop hypotheses -----------------------------------------------
    print(f"\n[1/5] Cropping hypotheses to first {n} frames …")
    t0 = time.time()
    _crop_h5_to_n_frames(HYPOTHESES_H5, cropped_h5, n)
    print(f"      done in {time.time()-t0:.1f}s  →  {cropped_h5}")

    # ---- Ingest -----------------------------------------------------------
    print("\n[2/5] Ingesting into Ultrack NodeDB …")
    cfg = TrackingConfig(
        linking_mode=mode,
        min_area=args.min_area,
        appear_weight=args.appear_weight,
        disappear_weight=args.disappear_weight,
    )
    t0 = time.time()
    ingest_hypotheses_to_db(cropped_h5, wd, cfg, overwrite=True, max_partitions=max_partitions)
    print(f"      done in {time.time()-t0:.1f}s")

    # ---- Link ------------------------------------------------------------
    print("\n[3/5] Linking …")
    t0 = time.time()
    for step, total, label in run_linking(wd, cfg):
        print(f"      [{step}/{total}] {label}")
    print(f"      done in {time.time()-t0:.1f}s")

    # ---- Solve -----------------------------------------------------------
    print("\n[4/5] Solving ILP …")
    t0 = time.time()
    for step, total, label in run_solve(wd, cfg):
        print(f"      [{step}/{total}] {label}")
    print(f"      done in {time.time()-t0:.1f}s")

    # ---- Export ----------------------------------------------------------
    print("\n[5/5] Exporting tracked labels …")
    t0 = time.time()
    new_labels = export_tracked_labels(wd, cfg, output_tif)
    print(f"      done in {time.time()-t0:.1f}s  →  {output_tif}")

    # ---- Ground truth ----------------------------------------------------
    gt_full = tifffile.imread(str(GROUND_TRUTH_TIF))
    gt_labels = gt_full[:n]  # first n frames
    # Handle (T, Z, Y, X) vs (T, Y, X)
    if gt_labels.ndim == 4 and gt_labels.shape[1] == 1:
        gt_labels = gt_labels[:, 0]
    new_labels_2d = new_labels
    if new_labels_2d.ndim == 4 and new_labels_2d.shape[1] == 1:
        new_labels_2d = new_labels_2d[:, 0]

    # ---- Compare ---------------------------------------------------------
    gt_counts = _cell_count_per_frame(gt_labels)
    new_counts = _cell_count_per_frame(new_labels_2d)

    gt_lengths = _track_lengths(gt_labels)
    new_lengths = _track_lengths(new_labels_2d)

    # ILP-selected node counts directly from DB (before export relabeling)
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session as _Session
    from ultrack.core.database import NodeDB as _NodeDB
    _engine = sqla.create_engine(f"sqlite:///{wd}/data.db")
    ilp_selected = {}
    with _Session(_engine) as _s:
        for _t in range(n):
            ilp_selected[_t] = _s.query(_NodeDB).filter(
                _NodeDB.t == _t, _NodeDB.selected == True
            ).count()

    lines = [
        f"Phase 2 Benchmark  n_frames={n}  linking_mode={mode}",
        f"  appear_weight={cfg.appear_weight}  disappear_weight={cfg.disappear_weight}",
        f"  min_area={cfg.min_area}  max_partitions={max_partitions}",
        "=" * 60,
        "",
        "Per-frame cell count (ILP selected vs exported vs GT):",
        f"  {'t':>4}  {'gt':>6}  {'ilp':>6}  {'exp':>6}  {'diff':>6}",
        f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}",
    ]
    for t in range(n):
        gt_c  = gt_counts[t] if t < len(gt_counts) else 0
        new_c = new_counts[t] if t < len(new_counts) else 0
        ilp_c = ilp_selected.get(t, 0)
        lines.append(f"  {t:>4}  {gt_c:>6}  {ilp_c:>6}  {new_c:>6}  {new_c-gt_c:>+6}")

    mean_gt  = np.mean(gt_counts)  if gt_counts  else 0
    mean_new = np.mean(new_counts) if new_counts else 0
    mean_ilp = np.mean(list(ilp_selected.values())) if ilp_selected else 0
    lines += [
        f"  {'mean':>4}  {mean_gt:>6.1f}  {mean_ilp:>6.1f}  {mean_new:>6.1f}  {mean_new-mean_gt:>+6.1f}",
        "",
        "Track length distribution:",
        "  Ground truth:",
        _distribution_summary(gt_lengths),
        "  New (Ultrack ILP):",
        _distribution_summary(new_lengths),
        "",
        f"Output: {output_tif}",
    ]

    report = "\n".join(lines)
    print("\n" + report)
    report_path.write_text(report)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
