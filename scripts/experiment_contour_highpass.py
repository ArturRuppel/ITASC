#!/usr/bin/env python
"""Explore binary edges from a highpass of the contour map (full stack).

Idea
----
The contour map encodes nucleus boundaries as *thick* bright ridges. A highpass
filter (image minus a blurred copy) is positive only where the signal rises above
its local surroundings -- i.e. on the ridge -- and negative/zero on the flat core
and flat background. Thresholding the highpass at zero therefore yields a binary
"edge" band sitting on the ridge, independent of the absolute contour level (so
faint-cell ridges and bright-cell ridges are treated alike).

The inner edge of that band approximates the *inner rim* of the thick ridge.

Layers shown for inspection (full stack):
  foreground, contour            -- raw inputs
  highpass                       -- contour minus blur (divergent)
  edges_unsharp                  -- highpass > 0
  edges_dog                      -- difference-of-gaussians > 0
  regions_unsharp                -- connected components NOT on an edge
                                    (tests whether the edges actually close cells)

Usage
-----
    python scripts/experiment_contour_highpass.py            # compute + stats
    python scripts/experiment_contour_highpass.py --gui      # also open napari
    python scripts/experiment_contour_highpass.py --gui --sigma 3 --dog 1 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import gaussian

DATA = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/pos00"
)
CONTOUR_PATH = DATA / "1_cellpose" / "nucleus_contours.tif"
FG_PATH = DATA / "1_cellpose" / "nucleus_foreground.tif"
GT_PATH = DATA / "2_nucleus" / "tracked_labels.tif"

COVERAGE_HIT = 0.5
COVERAGE_MISS = 0.1
FAINT_PCT = 15
BRIGHT_PCT = 70
RING_PX = 4


def unsharp_highpass(contour, sigma):
    """contour minus a Gaussian-blurred copy, per frame."""
    hp = np.empty_like(contour)
    for t in range(contour.shape[0]):
        hp[t] = contour[t] - gaussian(contour[t], sigma=sigma, preserve_range=True)
    return hp


def dog_highpass(contour, s1, s2):
    """Difference of Gaussians (s1<s2): band-pass that isolates ridge structure."""
    out = np.empty_like(contour)
    for t in range(contour.shape[0]):
        out[t] = (gaussian(contour[t], sigma=s1, preserve_range=True)
                  - gaussian(contour[t], sigma=s2, preserve_range=True))
    return out


def ridge_meijering(contour, sigma, floor):
    """Hessian ridge detector (Meijering): local maxima perpendicular to the
    ridge -> connected crest lines. Gated by an absolute contour floor so only
    real boundary ridges survive (background ridges are low-valued)."""
    from skimage.filters import meijering
    out = np.zeros(contour.shape, dtype=bool)
    for t in range(contour.shape[0]):
        cs = gaussian(contour[t], sigma=sigma, preserve_range=True)
        resp = meijering(cs, sigmas=[1.5], black_ridges=False)
        out[t] = (resp > 0.5 * resp.max()) & (cs > floor)
    return out


def ridge_skeleton(contour, sigma, floor):
    """Skeleton of the thresholded ridge band -> 1px connected centerline."""
    from skimage.morphology import skeletonize
    out = np.zeros(contour.shape, dtype=bool)
    for t in range(contour.shape[0]):
        cs = gaussian(contour[t], sigma=sigma, preserve_range=True)
        out[t] = skeletonize(cs > floor)
    return out


def ridge_local_max(contour, sigma, size, floor):
    """Binary ridge crests: pixels that are a local maximum AND above an absolute
    floor. The local-max test thins thick ridges to their crest; the absolute
    floor rejects background/interior texture that is locally peaked but low."""
    out = np.zeros(contour.shape, dtype=bool)
    for t in range(contour.shape[0]):
        cs = gaussian(contour[t], sigma=sigma, preserve_range=True)
        mx = ndi.maximum_filter(cs, size=size)
        out[t] = (cs >= mx) & (cs > floor)
    return out


def label_regions(edges):
    """Connected components of the non-edge area, per frame (0 = edge)."""
    out = np.zeros(edges.shape, dtype=np.int32)
    struct = ndi.generate_binary_structure(2, 1)
    for t in range(edges.shape[0]):
        lab, _ = ndi.label(~edges[t], structure=struct)
        out[t] = lab
    return out


def foreground_from_region_labels(regions, fg, keep_thr, min_area=0):
    """Keep non-edge regions whose mean foreground score is high enough."""
    out = np.zeros(regions.shape, dtype=bool)
    for t in range(regions.shape[0]):
        lab = regions[t]
        n = int(lab.max())
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        vals = ndi.mean(fg[t], lab, index=idx)
        areas = ndi.sum(np.ones(lab.shape, dtype=np.uint8), lab, index=idx)
        keep = idx[(vals >= keep_thr) & (areas >= min_area)]
        out[t] = np.isin(lab, keep)
    return out


def foreground_from_skeleton_regions(edges, fg, keep_thr, min_area=0):
    """Foreground = skeleton-bounded regions classified by mean foreground."""
    return foreground_from_region_labels(
        label_regions(edges),
        fg,
        keep_thr=keep_thr,
        min_area=min_area,
    )


def build_gt_index(gt, fg):
    nuclei = []
    for t in range(gt.shape[0]):
        lab = gt[t]
        fgt = fg[t]
        for rid in np.unique(lab):
            if rid == 0:
                continue
            m = lab == rid
            nuclei.append({
                "t": t,
                "rid": int(rid),
                "mask": m,
                "area": int(m.sum()),
                "mean_fg": float(fgt[m].mean()),
            })
    mean_fgs = np.array([n["mean_fg"] for n in nuclei])
    faint_cut = np.percentile(mean_fgs, FAINT_PCT)
    bright_cut = np.percentile(mean_fgs, BRIGHT_PCT)
    for n in nuclei:
        n["faint"] = n["mean_fg"] <= faint_cut
        n["bright"] = n["mean_fg"] >= bright_cut
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


def pick_best(results):
    """Prefer low bloat among configs that keep strong recall."""
    viable = [r for r in results if r[1]["all_rec"] >= 0.98]
    if viable:
        return min(viable, key=lambda r: (r[1]["bloat"], -r[1]["faint_rec"]))
    return max(results, key=lambda r: (r[1]["all_rec"], r[1]["faint_rec"], -r[1]["bloat"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--sigma", type=float, default=1.0, help="pre-smoothing sigma")
    ap.add_argument("--size", type=int, default=3, help="local-max window (px)")
    ap.add_argument("--floor", type=float, default=None,
                    help="single ridge-band floor (default: sweep 0.01/0.02/0.037)")
    ap.add_argument("--fg-thr", type=float, nargs="+",
                    default=[0.05, 0.08, 0.12, 0.20, 0.30],
                    help="region mean-foreground thresholds to evaluate")
    ap.add_argument("--min-area", type=int, default=60,
                    help="minimum classified region area in pixels")
    ap.add_argument("--detail", action="store_true",
                    help="print every floor x foreground-threshold metric")
    args = ap.parse_args()

    contour = tifffile.imread(CONTOUR_PATH).astype(np.float32)
    fg = tifffile.imread(FG_PATH).astype(np.float32)
    print(f"stack {contour.shape}  contour max={contour.max():.3f}")

    pcts = [50, 75, 90, 95, 99]
    qs = np.percentile(contour[contour > 0], pcts)
    print("contour positive-value percentiles (for picking the floor):")
    for p, q in zip(pcts, qs):
        print(f"  {p:>2}th = {q:.3f}")

    if args.floor is not None:
        floors = [args.floor]
    else:
        floors = [0.005, 0.010, 0.020, 0.037, 0.075, 0.100]

    gt = tifffile.imread(GT_PATH) if GT_PATH.exists() else None
    nuclei = rings = None
    if gt is not None:
        n_eval = min(contour.shape[0], fg.shape[0], gt.shape[0])
        nuclei = build_gt_index(gt[:n_eval], fg[:n_eval])
        rings = precompute_bloat_rings(gt[:n_eval])
        n_faint = sum(n["faint"] for n in nuclei)
        n_bright = sum(n["bright"] for n in nuclei)
        print(f"GT nuclei={len(nuclei)}  faint={n_faint}  bright={n_bright}")

        print("\n# baseline: global foreground threshold")
        for thr in args.fg_thr:
            r = evaluate(fg[:n_eval] > thr, nuclei, rings)
            print(f"  fg>{thr:.3f}  all={r['all_rec']:.3f}  "
                  f"faint={r['faint_rec']:.3f}  miss={r['missed']:>4}  "
                  f"bloat={r['bloat']:.4f}")

    print(f"\nridge skeleton at varying floor (sigma={args.sigma}):")
    edge_layers = {}
    mask_layers = {}
    for fl in floors:
        e = ridge_skeleton(contour, args.sigma, fl)
        edge_layers[f"skeleton_floor{fl:.3f}"] = e
        regions = label_regions(e)
        n_reg = np.array([int(regions[t].max()) for t in range(regions.shape[0])])
        print(f"  floor={fl:.3f}  edge-frac={e.mean():.3f}  "
              f"regions/frame mean={n_reg.mean():.0f}")
        if nuclei is not None and rings is not None:
            n_eval = len({n["t"] for n in nuclei})
            scored = []
            for keep_thr in args.fg_thr:
                mask = foreground_from_region_labels(
                    regions[:n_eval],
                    fg[:n_eval],
                    keep_thr=keep_thr,
                    min_area=args.min_area,
                )
                r = evaluate(mask, nuclei, rings)
                scored.append((keep_thr, r, mask))
                if args.detail:
                    print(f"    meanfg>={keep_thr:.3f}  all={r['all_rec']:.3f}  "
                          f"faint={r['faint_rec']:.3f}  miss={r['missed']:>4}  "
                          f"bloat={r['bloat']:.4f}")
            best_thr, best_r, best_mask = pick_best(scored)
            mask_layers[f"regions_floor{fl:.3f}_fg{best_thr:.3f}"] = best_mask
            print(f"    best meanfg>={best_thr:.3f}  all={best_r['all_rec']:.3f}  "
                  f"faint={best_r['faint_rec']:.3f}  miss={best_r['missed']:>4}  "
                  f"bloat={best_r['bloat']:.4f}")

    if not args.gui:
        print("\n(compute OK; pass --gui to open napari)")
        return

    import napari
    v = napari.Viewer()
    v.add_image(fg, name="foreground", colormap="gray")
    v.add_image(contour, name="contour", colormap="inferno", blending="additive")
    for i, (name, e) in enumerate(edge_layers.items()):
        v.add_labels(e.astype(np.uint8), name=name, opacity=0.7, visible=(i == 0))
    for i, (name, m) in enumerate(mask_layers.items()):
        v.add_labels(m.astype(np.uint8), name=name, opacity=0.45, visible=False)
    napari.run()


if __name__ == "__main__":
    main()
