#!/usr/bin/env python
"""Experiment: define nucleus foreground from raw Cellpose flow magnitude.

The contour image is a derived scalar made from flow divergence. This script
tests a simpler scalar field from the raw flow tensor:

    magnitude = sqrt(dy**2 + dx**2)

Per-z magnitude is reduced to a 2-D movie with mean or max reduction, normalized
to [0, 1], thresholded, and evaluated against the same recall/bloat metrics used
by the contour foreground experiments.

Usage
-----
    python scripts/experiment_flow_magnitude_fg.py
    python scripts/experiment_flow_magnitude_fg.py --gui --reduction max
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi


DATA = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/pos00"
)
DP_PATH = DATA / "1_cellpose" / "nucleus_dp_3dt.tif"
FG_PATH = DATA / "1_cellpose" / "nucleus_foreground.tif"
GT_PATH = DATA / "2_nucleus" / "tracked_labels.tif"

COVERAGE_HIT = 0.5
COVERAGE_MISS = 0.1
FAINT_PCT = 15
BRIGHT_PCT = 70
RING_PX = 4


def normalize01(stack):
    arr = np.asarray(stack, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.float32)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32, copy=False)


def flow_magnitude_scalar(dp, reduction):
    """Return a T,Y,X scalar movie from Cellpose dp shaped T,Z,2,Y,X."""
    arr = np.asarray(dp, dtype=np.float32)
    if arr.ndim != 5 or arr.shape[2] != 2:
        raise ValueError(f"Expected dp shape T,Z,2,Y,X; got {arr.shape}")
    mag = np.sqrt(np.sum(arr * arr, axis=2, dtype=np.float32))
    if reduction == "mean":
        return mag.mean(axis=1, dtype=np.float32)
    if reduction == "max":
        return mag.max(axis=1)
    raise ValueError(f"reduction must be 'mean' or 'max', got {reduction!r}")


def build_gt_index(gt, score):
    nuclei = []
    for t in range(gt.shape[0]):
        lab = gt[t]
        s = score[t]
        for rid in np.unique(lab):
            if rid == 0:
                continue
            m = lab == rid
            nuclei.append({
                "t": t,
                "rid": int(rid),
                "mask": m,
                "area": int(m.sum()),
                "mean_score": float(s[m].mean()),
            })
    mean_scores = np.array([n["mean_score"] for n in nuclei])
    faint_cut = np.percentile(mean_scores, FAINT_PCT)
    bright_cut = np.percentile(mean_scores, BRIGHT_PCT)
    for n in nuclei:
        n["faint"] = n["mean_score"] <= faint_cut
        n["bright"] = n["mean_score"] >= bright_cut
    return nuclei


def precompute_bloat_rings(gt):
    rings = {}
    struct = ndi.generate_binary_structure(2, 2)
    for t in range(gt.shape[0]):
        lab = gt[t]
        any_nucleus = lab > 0
        for rid in np.unique(lab):
            if rid == 0:
                continue
            m = lab == rid
            dil = ndi.binary_dilation(m, structure=struct, iterations=RING_PX)
            ring = dil & ~m & ~(any_nucleus & ~m)
            rings[(t, int(rid))] = ring
    return rings


def evaluate(mask, nuclei, rings):
    cov_all, cov_faint, missed = [], [], 0
    for n in nuclei:
        m = n["mask"]
        cov = (mask[n["t"]] & m).sum() / n["area"]
        cov_all.append(cov >= COVERAGE_HIT)
        if cov < COVERAGE_MISS:
            missed += 1
        if n["faint"]:
            cov_faint.append(cov >= COVERAGE_HIT)
    bloat = []
    for n in nuclei:
        if not n["bright"]:
            continue
        ring = rings[(n["t"], n["rid"])]
        if ring.sum() == 0:
            continue
        bloat.append((mask[n["t"]] & ring).sum() / n["area"])
    return {
        "all_rec": float(np.mean(cov_all)),
        "faint_rec": float(np.mean(cov_faint)) if cov_faint else float("nan"),
        "missed": missed,
        "bloat": float(np.mean(bloat)) if bloat else float("nan"),
    }


def print_metrics(prefix, score, thresholds, nuclei, rings):
    rows = []
    for thr in thresholds:
        r = evaluate(score >= thr, nuclei, rings)
        rows.append((thr, r))
        print(f"  {prefix}>={thr:.3f}  all={r['all_rec']:.3f}  "
              f"faint={r['faint_rec']:.3f}  miss={r['missed']:>4}  "
              f"bloat={r['bloat']:.4f}")
    viable = [row for row in rows if row[1]["all_rec"] >= 0.95]
    if viable:
        thr, r = min(viable, key=lambda row: (row[1]["bloat"], -row[1]["faint_rec"]))
        label = "tradeoff"
    else:
        thr, r = max(rows, key=lambda row: (row[1]["all_rec"], row[1]["faint_rec"]))
        label = "highest-recall"
    print(f"  {label} {prefix}>={thr:.3f}  all={r['all_rec']:.3f}  "
          f"faint={r['faint_rec']:.3f}  miss={r['missed']:>4}  "
          f"bloat={r['bloat']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--reduction", choices=("mean", "max", "both"), default="both")
    ap.add_argument("--frames", type=int, default=None,
                    help="limit frames for faster iteration")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80])
    args = ap.parse_args()

    dp = tifffile.imread(DP_PATH).astype(np.float32)
    fg = tifffile.imread(FG_PATH).astype(np.float32)
    gt = tifffile.imread(GT_PATH)
    if args.frames is not None:
        dp = dp[:args.frames]
        fg = fg[:args.frames]
        gt = gt[:args.frames]

    print(f"dp {dp.shape}  fg {fg.shape}  gt {gt.shape}")
    nuclei = build_gt_index(gt, fg)
    rings = precompute_bloat_rings(gt)
    n_faint = sum(n["faint"] for n in nuclei)
    n_bright = sum(n["bright"] for n in nuclei)
    print(f"GT nuclei={len(nuclei)}  faint={n_faint}  bright={n_bright}")

    print("\n# baseline: foreground score")
    print_metrics("fg", fg, args.thresholds, nuclei, rings)

    reductions = ("mean", "max") if args.reduction == "both" else (args.reduction,)
    scalar_layers = {}
    for reduction in reductions:
        raw = flow_magnitude_scalar(dp, reduction)
        score = normalize01(raw)
        scalar_layers[reduction] = score
        qs = np.percentile(raw, [50, 75, 90, 95, 99])
        print(f"\n# flow magnitude {reduction} z-reduction")
        print("  raw percentiles: " + " ".join(f"p{p}={q:.3f}" for p, q in zip([50, 75, 90, 95, 99], qs)))
        print_metrics(f"mag_{reduction}", score, args.thresholds, nuclei, rings)

    if not args.gui:
        print("\n(compute OK; pass --gui to open napari)")
        return

    import napari
    viewer = napari.Viewer()
    viewer.add_image(fg, name="foreground", colormap="gray")
    for reduction, score in scalar_layers.items():
        viewer.add_image(score, name=f"flow_magnitude_{reduction}", colormap="inferno")
        for thr in args.thresholds:
            viewer.add_labels((score >= thr).astype(np.uint8),
                              name=f"flow_magnitude_{reduction}>={thr:.3f}",
                              opacity=0.45, visible=False)
    viewer.add_labels(gt, name="GT", opacity=0.35, visible=False)
    napari.run()


if __name__ == "__main__":
    main()
