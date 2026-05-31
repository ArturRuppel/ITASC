#!/usr/bin/env python
"""Grow blob seeds into a foreground mask that stops at contour ridges.

Context
-------
``experiment_foreground_particle_seeds.py`` showed that blob detectors find even
very faint nuclei from the foreground score image without firing on background.
This script takes those seeds and grows each one into a region that stops at the
contour ridge surrounding it.

Requirements driving the design
-------------------------------
1. Growth must stop at the contour ridges (the boundary band).
2. Robust to *holes* in the ridge -- faint nuclei sometimes have gaps in their
   ridge, and naive flood-fill would leak through them.
3. A seed whose grown region never runs into a ridge is spurious and is dropped.

Method: marker-controlled watershed on the contour
---------------------------------------------------
* Seeds become watershed markers (label 1..K).
* A *background* marker (label K+1) is placed at every pixel farther than
  ``--max-dist`` from any seed. This caps how far a basin can flood, so a seed
  in open background cannot run away -- it gets boxed in by the background basin.
* The flooding elevation is the (smoothed) contour. To survive ridge holes the
  thresholded ridge is morphologically *closed* and then burned into the
  elevation as a tall wall, sealing small gaps before flooding.
* Watershed naturally places the boundary between two adjacent seed basins at the
  ridge crest between them, so an inter-nucleus ridge hole does not merge them --
  the two floods meet and stop.
* Pruning (requirement 3): for each seed basin we measure the fraction of its
  perimeter that sits on a ridge. A genuine nucleus is ringed by ridge (or by a
  neighbouring basin, whose shared border is also on a ridge), giving high
  contact. A runaway seed only touched the background cap -> low ridge contact
  -> discarded.

The final foreground is the union of the surviving basins. Metrics (recall,
faint recall, bright-nucleus bloat) are reported against the global-threshold
baseline using the same harness as ``experiment_contour_highpass.py``.

Usage
-----
    python scripts/experiment_grow_seeds.py
    python scripts/experiment_grow_seeds.py --gui
    python scripts/experiment_grow_seeds.py --gui --blob dog --blob-thr 0.010 \
        --floor 0.037 --close 2 --max-dist 45 --contact 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.feature import blob_dog, blob_log
from skimage.filters import gaussian
from skimage.morphology import disk
from skimage.segmentation import watershed

from experiment_contour_highpass import (
    FG_PATH,
    CONTOUR_PATH,
    GT_PATH,
    build_gt_index,
    precompute_bloat_rings,
    evaluate,
)


def detect_seeds(fg, *, method, threshold, min_sigma=1.5, max_sigma=8.0):
    """Return a list (per frame) of (y, x) seed coordinates."""
    seeds = []
    for t in range(fg.shape[0]):
        img = fg[t]
        if method == "log":
            blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma,
                             num_sigma=6, threshold=threshold, overlap=0.75,
                             exclude_border=False)
        elif method == "dog":
            blobs = blob_dog(img, min_sigma=min_sigma, max_sigma=max_sigma,
                             sigma_ratio=1.6, threshold=threshold, overlap=0.75,
                             exclude_border=False)
        else:
            raise ValueError(method)
        seeds.append(blobs[:, :2] if len(blobs) else np.empty((0, 2)))
    return seeds


def grow_frame(contour_t, seed_yx, *, sigma, floor, close_radius, max_dist,
               contact_frac, min_area):
    """Grow one frame's seeds; return (foreground bool, watershed labels,
    n_seeds, n_kept).

    The contour/divergence map is *high inside* a nucleus and peaks on the
    surrounding rim. Flooding the contour from a seed marker therefore grows the
    basin outward across the interior and the watershed line settles on the rim
    *peak* between this nucleus and its neighbour (or the background). So the
    grown basin reaches the rim peak with no explicit clipping.
    """
    cs = gaussian(contour_t, sigma=sigma, preserve_range=True)
    # Seal holes in the rim: grayscale closing bridges dark notches in the
    # bright rim up to rim height, so a gap in a faint ridge does not let the
    # flood spill into the neighbour/background.
    if close_radius > 0:
        cs = ndi.grey_closing(cs, footprint=disk(close_radius))
    H, W = cs.shape
    n_seeds = len(seed_yx)
    if n_seeds == 0:
        return np.zeros((H, W), bool), np.zeros((H, W), np.int32), 0, 0

    markers = np.zeros((H, W), dtype=np.int32)
    seed_mask = np.zeros((H, W), dtype=bool)
    for i, (y, x) in enumerate(seed_yx, start=1):
        yy = int(np.clip(round(y), 0, H - 1))
        xx = int(np.clip(round(x), 0, W - 1))
        markers[yy, xx] = i
        seed_mask[yy, xx] = True

    # Background marker caps runaway flooding: a seed sitting in open background
    # gets boxed in once its basin reaches max_dist, instead of filling a frame.
    dist = ndi.distance_transform_edt(~seed_mask)
    bg_label = n_seeds + 1
    markers[(dist > max_dist) & (markers == 0)] = bg_label

    ws = watershed(cs, markers)

    out = np.zeros((H, W), dtype=bool)
    kept = 0
    for i in range(1, n_seeds + 1):
        basin = ws == i
        area = int(basin.sum())
        if area < min_area:
            continue
        border = basin & ~ndi.binary_erosion(basin)
        nb = int(border.sum())
        if nb == 0:
            continue
        # Fraction of the basin perimeter sitting on the rim crest. A genuine
        # nucleus is ringed by rim (its own, or the shared crest with a
        # neighbour); a background seed's basin is bounded by the low-contour
        # distance cap -> low contact -> dropped.
        contact = float((cs[border] > floor).mean())
        if contact < contact_frac:
            continue
        out |= basin
        kept += 1
    return out, ws.astype(np.int32), n_seeds, kept


def grow_stack(contour, seeds, **kw):
    fgmask = np.zeros(contour.shape, dtype=bool)
    wslab = np.zeros(contour.shape, dtype=np.int32)
    tot_seeds = tot_kept = 0
    for t in range(contour.shape[0]):
        m, ws, ns, nk = grow_frame(contour[t], seeds[t], **kw)
        fgmask[t] = m
        wslab[t] = ws
        tot_seeds += ns
        tot_kept += nk
    return fgmask, wslab, tot_seeds, tot_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--blob", choices=["log", "dog"], default="log")
    ap.add_argument("--blob-thr", type=float, default=0.010)
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="contour pre-smoothing sigma")
    ap.add_argument("--floor", type=float, default=0.30,
                    help="rim-crest level: contour above this counts as rim "
                         "(separates rim from the high-ish nucleus interior)")
    ap.add_argument("--close", type=int, default=2,
                    help="disk radius for closing ridge holes (0 disables)")
    ap.add_argument("--max-dist", type=float, default=45.0,
                    help="max flood distance from a seed (background cap, px)")
    ap.add_argument("--contact", type=float, default=0.5,
                    help="min fraction of basin perimeter on a ridge to keep")
    ap.add_argument("--min-area", type=int, default=30)
    args = ap.parse_args()

    contour = tifffile.imread(CONTOUR_PATH).astype(np.float32)
    fg = tifffile.imread(FG_PATH).astype(np.float32)
    gt = tifffile.imread(GT_PATH) if GT_PATH.exists() else None
    if args.frames is not None:
        contour = contour[:args.frames]
        fg = fg[:args.frames]
        if gt is not None:
            gt = gt[:args.frames]
    print(f"stack {contour.shape}  contour max={contour.max():.3f}")

    seeds = detect_seeds(fg, method=args.blob, threshold=args.blob_thr)
    seed_counts = np.array([len(s) for s in seeds])
    print(f"seeds ({args.blob}, thr={args.blob_thr}): "
          f"mean={seed_counts.mean():.1f}/frame  total={seed_counts.sum()}")

    fgmask, wslab, tot_seeds, tot_kept = grow_stack(
        contour, seeds,
        sigma=args.sigma, floor=args.floor, close_radius=args.close,
        max_dist=args.max_dist, contact_frac=args.contact,
        min_area=args.min_area,
    )
    print(f"grown: kept {tot_kept}/{tot_seeds} seeds "
          f"({100 * tot_kept / max(tot_seeds, 1):.0f}%)  "
          f"fg-frac={fgmask.mean():.3f}")

    if gt is not None:
        n_eval = min(contour.shape[0], fg.shape[0], gt.shape[0])
        nuclei = build_gt_index(gt[:n_eval], fg[:n_eval])
        rings = precompute_bloat_rings(gt[:n_eval])
        n_faint = sum(n["faint"] for n in nuclei)
        n_bright = sum(n["bright"] for n in nuclei)
        print(f"\nGT nuclei={len(nuclei)}  faint={n_faint}  bright={n_bright}")

        print("\n# baseline: global foreground threshold")
        for thr in [0.05, 0.08, 0.12, 0.20]:
            r = evaluate(fg[:n_eval] > thr, nuclei, rings)
            print(f"  fg>{thr:.3f}  all={r['all_rec']:.3f}  "
                  f"faint={r['faint_rec']:.3f}  miss={r['missed']:>4}  "
                  f"bloat={r['bloat']:.4f}")

        r = evaluate(fgmask[:n_eval], nuclei, rings)
        print("\n# grown seeds")
        print(f"  all={r['all_rec']:.3f}  faint={r['faint_rec']:.3f}  "
              f"miss={r['missed']:>4}  bloat={r['bloat']:.4f}")

    if not args.gui:
        print("\n(compute OK; pass --gui to open napari)")
        return

    # Build a points layer of the (raw) seeds for inspection.
    seed_pts = []
    for t, s in enumerate(seeds):
        for y, x in s:
            seed_pts.append((t, y, x))
    seed_pts = np.asarray(seed_pts, dtype=np.float32) if seed_pts else None

    import napari
    v = napari.Viewer()
    v.add_image(fg, name="foreground", colormap="gray")
    v.add_image(contour, name="contour", colormap="inferno", blending="additive")
    v.add_labels(wslab, name="watershed_basins", opacity=0.4, visible=False)
    v.add_labels(fgmask.astype(np.uint8), name="grown_foreground", opacity=0.5)
    if gt is not None:
        v.add_labels(gt, name="GT", opacity=0.35, visible=False)
    if seed_pts is not None:
        v.add_points(seed_pts, name="seeds", size=5, face_color="cyan",
                     opacity=0.9)
    napari.run()


if __name__ == "__main__":
    main()
