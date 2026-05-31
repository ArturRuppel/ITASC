#!/usr/bin/env python
"""Compare background-subtraction methods on the cellpose fg/contour maps.

The diagnostic showed nuclei separate near-perfectly from *empty* background but
the background *floor varies spatially*, so a single threshold that catches a
faint nucleus also catches the elevated floor elsewhere. This is a background-
subtraction problem: estimate a spatially-varying floor, subtract it, threshold
the flat residual.

Methods compared (each applied to the foreground map; top-hat also to contour):
  raw            baseline
  tophat         white top-hat = img - grayscale_opening(disk R>nucleus).
                 The opening cannot see nuclei (smaller than the SE) so it is the
                 true local floor even in crowded regions. Literal bg subtraction.
  localadapt     img - local Gaussian mean (threshold_local). Cheap, but the
                 local mean is contaminated by neighbouring nuclei in dense areas.
  tv_cartoon     TV (Chambolle) cartoon: keeps piecewise-smooth nuclei, drops
                 oscillatory background texture.

For each residual we also show a binary preview (global Otsu on the residual).

Separability is reported as AUC of inside-nucleus vs background pixels for two
background pools: FAR (>=margin px from any nucleus, = empty space) and NEAR (a
ring close to nuclei, = halo / structured background, the real adversary).

Usage
-----
    python scripts/experiment_bg_subtract.py            # compute + stats
    python scripts/experiment_bg_subtract.py --gui      # also open napari
    python scripts/experiment_bg_subtract.py --gui --ball 15 --block 51 --tv 0.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_local, threshold_otsu
from skimage.morphology import disk, white_tophat
from skimage.restoration import denoise_tv_chambolle

DATA = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/pos00"
)
CONTOUR_PATH = DATA / "1_cellpose" / "nucleus_contours.tif"
FG_PATH = DATA / "1_cellpose" / "nucleus_foreground.tif"
GT_PATH = DATA / "2_nucleus" / "tracked_labels.tif"

FAINT_PCT = 15


# --------------------------------------------------------------------------- #
# background-subtraction methods (per frame, stacked)
# --------------------------------------------------------------------------- #
def tophat(stack, ball):
    se = disk(ball)
    out = np.empty_like(stack)
    for t in range(stack.shape[0]):
        out[t] = white_tophat(stack[t], se)
    return out


def localadapt(stack, block):
    out = np.empty_like(stack)
    for t in range(stack.shape[0]):
        thr = threshold_local(stack[t], block_size=block, method="gaussian")
        out[t] = np.clip(stack[t] - thr, 0, None)
    return out


def tv_cartoon(stack, weight):
    out = np.empty_like(stack)
    for t in range(stack.shape[0]):
        out[t] = denoise_tv_chambolle(stack[t], weight=weight)
    return out


def otsu_binary(stack):
    vals = stack[stack > 0]
    if vals.size == 0:
        return np.zeros(stack.shape, bool)
    thr = threshold_otsu(vals)
    return stack > thr


# --------------------------------------------------------------------------- #
# separability
# --------------------------------------------------------------------------- #
def auc(pos, neg, max_n=150000):
    rng = np.random.default_rng(0)
    if len(pos) > max_n:
        pos = rng.choice(pos, max_n, replace=False)
    if len(neg) > max_n:
        neg = rng.choice(neg, max_n, replace=False)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def build_faint_flags(gt, fg):
    means, keys = [], []
    for t in range(gt.shape[0]):
        lab = gt[t]
        for rid in np.unique(lab):
            if rid == 0:
                continue
            means.append(float(fg[t][lab == rid].mean()))
            keys.append((t, int(rid)))
    cut = np.percentile(means, FAINT_PCT)
    return {k: (m <= cut) for k, m in zip(keys, means)}


def separability(feat, gt, faint_flags, margin=4, near_lo=2, near_hi=8):
    struct = ndi.generate_binary_structure(2, 2)
    pos, posf, far, near = [], [], [], []
    for t in range(feat.shape[0]):
        lab = gt[t]
        any_nuc = lab > 0
        inside = ndi.binary_erosion(any_nuc, struct, iterations=1)
        far_bg = ~ndi.binary_dilation(any_nuc, struct, iterations=margin)
        ring_out = ndi.binary_dilation(any_nuc, struct, iterations=near_hi)
        ring_in = ndi.binary_dilation(any_nuc, struct, iterations=near_lo)
        near_bg = ring_out & ~ring_in & ~any_nuc
        faint_mask = np.zeros_like(any_nuc)
        for rid in np.unique(lab):
            if rid != 0 and faint_flags.get((t, int(rid)), False):
                faint_mask |= lab == rid
        pos.append(feat[t][inside])
        posf.append(feat[t][faint_mask])
        far.append(feat[t][far_bg])
        near.append(feat[t][near_bg])
    pos = np.concatenate(pos); posf = np.concatenate(posf)
    far = np.concatenate(far); near = np.concatenate(near)
    return {
        "all_far": auc(pos, far),
        "all_near": auc(pos, near),
        "faint_far": auc(posf, far) if len(posf) else float("nan"),
        "faint_near": auc(posf, near) if len(posf) else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--ball", type=int, default=15, help="top-hat SE radius (px)")
    ap.add_argument("--block", type=int, default=51, help="local-adapt window (odd px)")
    ap.add_argument("--tv", type=float, default=0.1, help="TV weight")
    args = ap.parse_args()

    contour = tifffile.imread(CONTOUR_PATH).astype(np.float32)
    fg = tifffile.imread(FG_PATH).astype(np.float32)
    gt = tifffile.imread(GT_PATH)
    if args.frames is not None:
        contour, fg, gt = contour[:args.frames], fg[:args.frames], gt[:args.frames]
    print(f"frames={fg.shape[0]}  ball={args.ball} block={args.block} tv={args.tv}")

    faint_flags = build_faint_flags(gt, fg)
    print(f"GT nuclei={len(faint_flags)}  faint={sum(faint_flags.values())}")

    feats = {
        "fg_raw": fg,
        "fg_tophat": tophat(fg, args.ball),
        "fg_localadapt": localadapt(fg, args.block),
        "fg_tv_cartoon": tv_cartoon(fg, args.tv),
        "contour_raw": contour,
        "contour_tophat": tophat(contour, args.ball),
    }

    print("\nseparability AUC (inside-nucleus vs background)")
    print(f"{'feature':<20}{'all/far':>9}{'all/near':>9}{'fnt/far':>9}{'fnt/near':>9}")
    for name, feat in feats.items():
        s = separability(feat, gt, faint_flags)
        print(f"{name:<20}{s['all_far']:>9.3f}{s['all_near']:>9.3f}"
              f"{s['faint_far']:>9.3f}{s['faint_near']:>9.3f}")

    if not args.gui:
        print("\n(pass --gui to inspect in napari)")
        return

    import napari
    v = napari.Viewer()
    v.add_image(fg, name="fg_raw", colormap="gray")
    v.add_image(contour, name="contour_raw", colormap="inferno",
                blending="additive", visible=False)
    for name in ["fg_tophat", "fg_localadapt", "fg_tv_cartoon", "contour_tophat"]:
        feat = feats[name]
        v.add_image(feat, name=name, colormap="magma",
                    blending="additive", visible=False)
        v.add_labels(otsu_binary(feat).astype(np.uint8),
                     name=f"{name}_otsu", opacity=0.45, visible=False)
    v.add_labels(gt, name="GT", opacity=0.3, visible=False)
    napari.run()


if __name__ == "__main__":
    main()
