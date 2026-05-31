#!/usr/bin/env python
"""Adaptive-threshold foreground from the cellpose fg map.

The bg-subtraction comparison showed that subtracting a *local* mean from the
foreground map flattens the per-nucleus offset that defeats a single global
threshold: faint and bright nuclei both stand out relative to their own
surroundings. This script develops that into a tunable foreground mask.

Two adaptive schemes, both robust to a spatially-varying background floor:
  gaussian   fg > local_gaussian_mean(window) + offset
  sauvola    fg > local_mean * (1 + k*(local_std/R - 1))   [Sauvola]
             -- adds the local std term, the standard for uneven illumination.

The window must be larger than a nucleus (or the local mean rises inside big
nuclei and erodes them) but not so large it reverts to a global threshold.

For each setting we report faint-nucleus pixel recall and the near-background
false-positive rate (fraction of the 2-8px ring around nuclei flagged) -- the
two quantities the global threshold could not satisfy at once -- and show the
binary masks in napari against GT.

Usage
-----
    python scripts/experiment_localadapt_fg.py
    python scripts/experiment_localadapt_fg.py --gui
    python scripts/experiment_localadapt_fg.py --gui --method gaussian \
        --window 31 51 81 --offset 0.0
    python scripts/experiment_localadapt_fg.py --gui --method sauvola \
        --window 51 --k 0.1 0.2 0.3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_local, threshold_otsu, threshold_sauvola

DATA = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/pos00"
)
FG_PATH = DATA / "1_cellpose" / "nucleus_foreground.tif"
CONTOUR_PATH = DATA / "1_cellpose" / "nucleus_contours.tif"
GT_PATH = DATA / "2_nucleus" / "tracked_labels.tif"

FAINT_PCT = 15


def residual_gaussian(stack, window):
    """Local-mean-subtracted residual: clip(fg - local_gaussian_mean, 0).

    Flattens the per-nucleus offset (faint and bright nuclei both rise above
    their own surroundings) while staying ~0 in flat background -- so a single
    global threshold on the residual works everywhere and does not light up
    flat-area noise (the failure of the bare `fg > local_mean` rule)."""
    out = np.zeros(stack.shape, np.float32)
    for t in range(stack.shape[0]):
        local_mean = threshold_local(stack[t], block_size=window, method="gaussian")
        out[t] = np.clip(stack[t] - local_mean, 0, None)
    return out


def adaptive_sauvola(stack, window, k):
    out = np.zeros(stack.shape, bool)
    for t in range(stack.shape[0]):
        thr = threshold_sauvola(stack[t], window_size=window, k=k)
        out[t] = stack[t] > thr
    return out


def build_gt(gt, fg):
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


def metrics(mask, gt, faint_flags, near_lo=2, near_hi=8):
    struct = ndi.generate_binary_structure(2, 2)
    faint_hit = faint_tot = 0
    near_fp = near_tot = 0
    for t in range(mask.shape[0]):
        lab = gt[t]
        any_nuc = lab > 0
        faint_mask = np.zeros_like(any_nuc)
        for rid in np.unique(lab):
            if rid != 0 and faint_flags.get((t, int(rid)), False):
                faint_mask |= lab == rid
        faint_hit += int((mask[t] & faint_mask).sum())
        faint_tot += int(faint_mask.sum())
        ring = (ndi.binary_dilation(any_nuc, struct, iterations=near_hi)
                & ~ndi.binary_dilation(any_nuc, struct, iterations=near_lo)
                & ~any_nuc)
        near_fp += int((mask[t] & ring).sum())
        near_tot += int(ring.sum())
    return (faint_hit / max(faint_tot, 1), near_fp / max(near_tot, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--method", choices=["residual", "sauvola"], default="residual")
    ap.add_argument("--window", type=int, nargs="+", default=[51])
    ap.add_argument("--resid-thr", type=float, nargs="+",
                    default=[0.0, 0.02, 0.05, 0.10],
                    help="global threshold on the local-mean-subtracted residual "
                         "(0.0 => Otsu picks it per setting)")
    ap.add_argument("--k", type=float, nargs="+", default=[0.1, 0.2])
    args = ap.parse_args()

    fg = tifffile.imread(FG_PATH).astype(np.float32)
    contour = tifffile.imread(CONTOUR_PATH).astype(np.float32)
    gt = tifffile.imread(GT_PATH)
    if args.frames is not None:
        fg, contour, gt = fg[:args.frames], contour[:args.frames], gt[:args.frames]
    print(f"frames={fg.shape[0]}  method={args.method}")

    faint_flags = build_gt(gt, fg)
    print(f"GT nuclei={len(faint_flags)}  faint={sum(faint_flags.values())}")

    masks = {}
    extra_images = {}
    if args.method == "residual":
        for w in args.window:
            w = w | 1  # force odd
            resid = residual_gaussian(fg, w)
            extra_images[f"residual_w{w}"] = resid
            pos = resid[resid > 0]
            for rt in args.resid_thr:
                thr = threshold_otsu(pos) if rt == 0.0 and pos.size else rt
                label = "otsu" if rt == 0.0 else f"{rt:g}"
                masks[f"resid_w{w}_t{label}"] = resid > thr
    else:
        for w in args.window:
            w = w | 1
            for k in args.k:
                masks[f"sauvola_w{w}_k{k:g}"] = adaptive_sauvola(fg, w, k)

    print(f"\n{'setting':<22}{'faint_recall':>14}{'near_fp':>10}{'fg_frac':>9}")
    for name, m in masks.items():
        fr, nfp = metrics(m, gt, faint_flags)
        print(f"{name:<22}{fr:>14.3f}{nfp:>10.3f}{m.mean():>9.3f}")

    if not args.gui:
        print("\n(pass --gui to inspect in napari)")
        return

    import napari
    v = napari.Viewer()
    v.add_image(fg, name="fg_raw", colormap="gray")
    v.add_image(contour, name="contour_raw", colormap="inferno",
                blending="additive", visible=False)
    for name, img in extra_images.items():
        v.add_image(img, name=name, colormap="magma", blending="additive",
                    visible=False)
    for i, (name, m) in enumerate(masks.items()):
        v.add_labels(m.astype(np.uint8), name=name, opacity=0.45, visible=(i == 0))
    v.add_labels(gt, name="GT", opacity=0.3, visible=False)
    napari.run()


if __name__ == "__main__":
    main()
