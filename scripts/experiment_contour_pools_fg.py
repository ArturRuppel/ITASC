#!/usr/bin/env python
"""Experiment: define foreground from contour-bounded pools, classified by fg.

Hypothesis
----------
A global foreground threshold couples two things we want to decouple: lowering
it to reveal faint nuclei simultaneously GROWS bright nuclei into their halo
(bloat). The boundary should instead be set by the CONTOUR RIDGE, not by an fg
isocontour.

Crossing a thick contour ridge from a nucleus core outward, the contour profile
is: low (core) -> rises -> peak -> falls -> low (halo). Connected components of
``contour < L`` therefore split into pools whose boundary sits where the contour
starts to rise -- the *inner rim* of the ridge. The core is one pool; the halo,
on the far side of the ridge, is a *separate* pool. Foreground score is then used
ONLY to classify pools (high fg -> real nucleus, low fg -> halo/background),
never to set the boundary. This should keep faint cells (their pool passes the fg
test) while not bloating bright nuclei (their pool stops at the inner rim).

Compared in the same harness: the global-threshold baseline, so metric
definitions are identical across methods.

    python scripts/experiment_contour_pools_fg.py            # sweep + print
    python scripts/experiment_contour_pools_fg.py --gui L S T # napari one config
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

N_FRAMES = 10
SMOOTH_SIGMA = 1.0          # light smoothing of contour before thresholding
COVERAGE_HIT = 0.5          # GT nucleus counts as recalled if >=50% covered
COVERAGE_MISS = 0.1         # GT nucleus counts as missed if <10% covered
FAINT_PCT = 15              # bottom 15% of nuclei by mean fg = "faint"
BRIGHT_PCT = 70             # top 30% (>=70th pct) by mean fg = "bright" (bloat target)
RING_PX = 4                 # halo ring width for bloat measurement


# --------------------------------------------------------------------------- #
# GT bookkeeping
# --------------------------------------------------------------------------- #
def build_gt_index(gt, fg):
    """Per (t, label): mask bbox + mean fg. Returns list of dicts and brightness cuts."""
    nuclei = []
    for t in range(gt.shape[0]):
        lab = gt[t]
        fgt = fg[t]
        for rid in np.unique(lab):
            if rid == 0:
                continue
            m = lab == rid
            nuclei.append({
                "t": t, "rid": int(rid), "mask": m,
                "area": int(m.sum()), "mean_fg": float(fgt[m].mean()),
            })
    mean_fgs = np.array([n["mean_fg"] for n in nuclei])
    faint_cut = np.percentile(mean_fgs, FAINT_PCT)
    bright_cut = np.percentile(mean_fgs, BRIGHT_PCT)
    for n in nuclei:
        n["faint"] = n["mean_fg"] <= faint_cut
        n["bright"] = n["mean_fg"] >= bright_cut
    return nuclei


def precompute_bloat_rings(gt):
    """For each bright nucleus, the 4px outer ring excluding any other GT nucleus."""
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
            ring = dil & ~m & ~(any_nucleus & ~m)  # exclude self and all other nuclei
            rings[(t, int(rid))] = ring
    return rings


# --------------------------------------------------------------------------- #
# Foreground builders
# --------------------------------------------------------------------------- #
def fg_global(contour, fg, thr):
    return fg > thr


def fg_marker_watershed(contour, fg, core_floor, outside_ceil, rim_trim=None):
    """Two-class marker watershed: core seeds vs an 'outside' (halo+bg) seed.

    The contour map is the elevation. Core markers sit in high-fg cores; the
    'outside' marker covers the dim skirt (halo) and background. Watershed lines
    land on the contour ridge crest *between* a core and the outside, so a bright
    nucleus basin stops at its rim instead of flooding into the halo. Foreground
    = pixels assigned to any core basin.
    """
    from skimage.segmentation import watershed
    out = np.zeros(fg.shape, dtype=bool)
    struct = ndi.generate_binary_structure(2, 2)
    for t in range(fg.shape[0]):
        cs = gaussian(contour[t], sigma=SMOOTH_SIGMA, preserve_range=True)
        fgt = fg[t]
        core = fgt >= core_floor
        outside = fgt <= outside_ceil
        markers = np.zeros(fgt.shape, dtype=np.int32)
        markers[outside] = 1                       # single shared 'outside' basin
        core_lbl, n = ndi.label(core, structure=struct)
        markers[core] = core_lbl[core] + 1         # core basins = labels 2..n+1
        ws = watershed(cs, markers=markers)
        basin = ws >= 2
        if rim_trim is not None:
            # pull the boundary from the ridge crest back to the inner rim by
            # dropping the high-contour outer half of the ridge from each basin
            basin &= cs < rim_trim
        out[t] = basin
    return out


def fg_progressive_grow(contour, fg, seed_floor, levels, rim_offset=0, min_area=60):
    """Grow each seed down through thresholds; freeze at peak boundary contour.

    Seeds = connected components of fg>seed_floor (one per nucleus core, bright
    AND faint). Lowering the threshold through ``levels`` (all < seed_floor) grows
    each seed's component. We track each seed's boundary contour (mean raw contour
    on its 1px ring) and keep the extent at the level where it PEAKS -- i.e. where
    the boundary sits on the ridge, before it spills into the halo. When two seeds
    merge into one component we lock them at their last single-seed extent (no
    over-growth across a shared ridge). Foreground = union of frozen extents.

    No global score gate: every seed yields a region (faint cells survive),
    background has no seed (excluded). rim_offset erodes toward the inner rim.
    """
    from collections import defaultdict
    struct = ndi.generate_binary_structure(2, 2)
    out = np.zeros(fg.shape, dtype=bool)

    def boundary_score(submask, csub):
        ring = submask & ~ndi.binary_erosion(submask)
        bvals = csub[ring]
        return float(bvals.mean()) if bvals.size else -1.0

    for t in range(fg.shape[0]):
        ct = contour[t]
        fgt = fg[t]
        seed_lbl, n_seeds = ndi.label(fgt > seed_floor, structure=struct)
        if n_seeds == 0:
            continue
        seed_pos = seed_lbl > 0
        sids_at_pos = seed_lbl[seed_pos]

        # initialise each seed's best extent at the seed level itself
        best = {}  # sid -> (score, slice, submask)
        objs0 = ndi.find_objects(seed_lbl)
        for sid in range(1, n_seeds + 1):
            sl = objs0[sid - 1]
            if sl is None:
                continue
            sub = seed_lbl[sl] == sid
            best[sid] = (boundary_score(sub, ct[sl]), sl, sub)
        locked = set()

        for lvl in levels:
            lab, n = ndi.label(fgt > lvl, structure=struct)
            if n == 0:
                continue
            comp_at = lab[seed_pos]
            comp_seeds = defaultdict(set)
            for c, s in zip(comp_at, sids_at_pos):
                if c > 0:
                    comp_seeds[c].add(int(s))
            objs = ndi.find_objects(lab)
            for c, sset in comp_seeds.items():
                live = sset - locked
                if len(sset) > 1:
                    locked |= sset           # merged: lock all involved
                    continue
                if not live:
                    continue
                sid = next(iter(live))
                sl = objs[c - 1]
                if sl is None:
                    continue
                sub = lab[sl] == c
                if sub.sum() < min_area:
                    continue
                score = boundary_score(sub, ct[sl])
                if score > best[sid][0]:
                    best[sid] = (score, sl, sub)

        frame = np.zeros(fgt.shape, dtype=bool)
        for sid, (score, sl, sub) in best.items():
            mask = ndi.binary_erosion(sub, iterations=rim_offset) if rim_offset else sub
            frame[sl] |= mask
        out[t] = frame
    return out


def fg_progressive_boundary(contour, fg, levels, min_score, overlap_max,
                            min_area=80, rim_offset=0):
    """Progressive thresholding; each region freezes where its boundary best
    aligns with the contour ridge.

    For every threshold level we label fg>level into components and score each by
    the mean *raw* contour along its 1px boundary ring (how well the boundary sits
    on a ridge). Across all levels this yields many candidate masks per nucleus
    (tight at high level, bloated at low level). We then greedily accept
    candidates in descending score order, skipping any that overlaps an
    already-accepted region by more than ``overlap_max`` of its own area. So each
    nucleus is represented by the single level-set whose boundary hugs its ridge
    best -- bright nuclei freeze tight, faint nuclei freeze low, none bloat. A
    ``min_score`` gate rejects background/halo blobs whose boundary is not a ridge.

    rim_offset>0 erodes each accepted mask by that many px to move the boundary
    from the ridge crest inward toward the inner rim.
    """
    out = np.zeros(fg.shape, dtype=bool)
    struct = ndi.generate_binary_structure(2, 2)
    for t in range(fg.shape[0]):
        ct = contour[t]
        fgt = fg[t]
        candidates = []  # (score, area, mask)
        for lvl in levels:
            lab, n = ndi.label(fgt > lvl, structure=struct)
            if n == 0:
                continue
            objs = ndi.find_objects(lab)
            for i in range(1, n + 1):
                sl = objs[i - 1]
                if sl is None:
                    continue
                sub = lab[sl] == i
                area = int(sub.sum())
                if area < min_area:
                    continue
                ring = sub & ~ndi.binary_erosion(sub)
                bvals = ct[sl][ring]
                if bvals.size == 0:
                    continue
                score = float(bvals.mean())
                if score < min_score:
                    continue
                candidates.append((score, area, sl, sub))
        candidates.sort(key=lambda c: -c[0])
        claimed = np.zeros(fgt.shape, dtype=bool)
        frame = np.zeros(fgt.shape, dtype=bool)
        for score, area, sl, sub in candidates:
            overlap = (claimed[sl] & sub).sum()
            if overlap > overlap_max * area:
                continue
            mask = sub
            if rim_offset > 0:
                mask = ndi.binary_erosion(sub, iterations=rim_offset)
            frame[sl] |= mask
            claimed[sl] |= sub
        out[t] = frame
    return out


def fg_contour_pools(contour, fg, L, stat, keep_thr):
    """Foreground = union of low-contour pools whose fg statistic passes keep_thr."""
    out = np.zeros(fg.shape, dtype=bool)
    for t in range(fg.shape[0]):
        cs = gaussian(contour[t], sigma=SMOOTH_SIGMA, preserve_range=True)
        pools, n = ndi.label(cs < L)
        if n == 0:
            continue
        fgt = fg[t]
        # vectorized per-pool stat
        if stat == "mean":
            sums = ndi.sum(fgt, pools, index=np.arange(1, n + 1))
            counts = ndi.sum(np.ones_like(fgt), pools, index=np.arange(1, n + 1))
            vals = sums / np.maximum(counts, 1)
        elif stat == "median":
            vals = ndi.median(fgt, pools, index=np.arange(1, n + 1))
        else:
            raise ValueError(stat)
        keep = np.flatnonzero(vals >= keep_thr) + 1
        out[t] = np.isin(pools, keep)
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true",
                    help="open napari for the progressive-boundary config")
    ap.add_argument("--seedfloor", type=float, default=0.30)
    ap.add_argument("--rimoff", type=int, default=0)
    args = ap.parse_args()

    contour = tifffile.imread(CONTOUR_PATH)[:N_FRAMES].astype(np.float32)
    fg = tifffile.imread(FG_PATH)[:N_FRAMES].astype(np.float32)
    gt = tifffile.imread(GT_PATH)[:N_FRAMES]

    print(f"frames={N_FRAMES} shape={fg.shape[1:]}  "
          f"contour[min,max]=[{contour.min():.3f},{contour.max():.3f}]  "
          f"fg[min,max]=[{fg.min():.3f},{fg.max():.3f}]")

    nuclei = build_gt_index(gt, fg)
    rings = precompute_bloat_rings(gt)
    n_faint = sum(n["faint"] for n in nuclei)
    n_bright = sum(n["bright"] for n in nuclei)
    print(f"GT nuclei={len(nuclei)}  faint={n_faint}  bright={n_bright}\n")

    if args.gui:
        grow = np.round(np.arange(args.seedfloor - 0.02, 0.03, -0.02), 3)
        mask = fg_progressive_grow(contour, fg, args.seedfloor, grow,
                                   rim_offset=args.rimoff)
        base = fg_global(contour, fg, 0.12)
        print(f"progressive seedfloor={args.seedfloor} rimoff={args.rimoff}:",
              evaluate(mask, nuclei, rings))
        print("global thr=0.12:", evaluate(base, nuclei, rings))
        import napari
        v = napari.Viewer()
        v.add_image(fg, name="foreground", colormap="gray")
        v.add_image(contour, name="contour", colormap="inferno", blending="additive")
        v.add_labels(gt, name="GT", opacity=0.4)
        v.add_labels(base.astype(np.uint8), name="global_thr0.12", opacity=0.5, visible=False)
        v.add_labels(mask.astype(np.uint8), name="pred_fg_progressive", opacity=0.5)
        napari.run()
        return

    hdr = f"{'method':<26} | {'all':>5} | {'faint':>5} | {'miss':>4} | {'bloat':>6}"
    print(hdr); print("-" * len(hdr))

    print("# baseline: global fg threshold")
    for thr in (0.08, 0.12, 0.20, 0.30, 0.50):
        r = evaluate(fg_global(contour, fg, thr), nuclei, rings)
        print(f"{'global thr=%.2f' % thr:<26} | {r['all_rec']:.3f} | "
              f"{r['faint_rec']:.3f} | {r['missed']:>4} | {r['bloat']:.4f}")

    print("\n# progressive grow-from-seeds (freeze each seed at peak boundary contour)")
    for seed_floor in (0.20, 0.30, 0.40):
        grow = np.round(np.arange(seed_floor - 0.02, 0.03, -0.02), 3)
        for rimoff in (0, 1):
            r = evaluate(
                fg_progressive_grow(contour, fg, seed_floor, grow, rim_offset=rimoff),
                nuclei, rings)
            tag = f"seed>={seed_floor:.2f} rimoff={rimoff}"
            print(f"{tag:<32} | {r['all_rec']:.3f} | {r['faint_rec']:.3f} | "
                  f"{r['missed']:>4} | {r['bloat']:.4f}")


if __name__ == "__main__":
    main()
