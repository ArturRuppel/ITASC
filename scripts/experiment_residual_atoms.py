#!/usr/bin/env python
"""Atom extraction via residual (local-mean-subtracted) contour ridges.

The local-mean-subtraction that flattened the foreground map (so a single
threshold works everywhere) is applied to the *contour* map here. The contour is
high inside nuclei (~0.15) and peaks on the rim; subtracting the local mean
removes that interior offset and leaves the rim as a clean positive ridge, ~0
elsewhere. A low-intensity cutoff then drops noise ridges so we don't oversegment
against speckle.

Atoms = territory regions separated by the cleaned ridges. We close small ridge
gaps with a watershed on the residual ridge map (markers = ridge-free cores,
flooded up to the rim crest), so a broken faint ridge does not merge two nuclei
through a gap as easily as plain connected components would.

Pipeline per frame
-------------------
1. territory  = (clip(fg - localmean(fg, W), 0) > fg_thr)          [locked fg]
2. ridge_resid= clip(contour - localmean(contour, Wc), 0)
3. ridge      = ridge_resid > floor                                 [noise cutoff]
4. cores      = label(territory & ~ridge)                           [seeds]
5. atoms      = watershed(ridge_resid, cores, mask=territory)       [close gaps]
   then drop atoms below min_area.

Usage
-----
    python scripts/experiment_residual_atoms.py
    python scripts/experiment_residual_atoms.py --gui
    python scripts/experiment_residual_atoms.py --gui --wc 51 --floor 0.01 0.02 0.04
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_local
from skimage.segmentation import watershed

DATA = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/pos00"
)
CONTOUR_PATH = DATA / "1_cellpose" / "nucleus_contours.tif"
FG_PATH = DATA / "1_cellpose" / "nucleus_foreground.tif"
GT_PATH = DATA / "2_nucleus" / "tracked_labels.tif"


def residual(stack, window):
    """clip(img - local_gaussian_mean(img, window), 0), per frame."""
    out = np.zeros(stack.shape, np.float32)
    for t in range(stack.shape[0]):
        lm = threshold_local(stack[t], block_size=window, method="gaussian")
        out[t] = np.clip(stack[t] - lm, 0, None)
    return out


def extract_atoms_frame(ridge_resid, territory, floor, min_area):
    ridge = ridge_resid > floor
    cores = territory & ~ridge
    markers, _ = ndi.label(cores)
    atoms = watershed(ridge_resid, markers=markers, mask=territory)
    # Merge atoms below min_area into their neighbours instead of deleting them,
    # so the whole territory stays covered (no holes, no specks). We drop the
    # small atoms' markers and re-flood from the surviving ones.
    if min_area > 0:
        ids, counts = np.unique(atoms, return_counts=True)
        small = set(ids[(counts < min_area) & (ids != 0)].tolist())
        if small:
            keep_markers = np.where(np.isin(atoms, list(small)), 0, atoms)
            atoms = watershed(ridge_resid, markers=keep_markers, mask=territory)
    return atoms.astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--w", type=int, default=51, help="fg local-mean window")
    ap.add_argument("--wc", type=int, default=51, help="contour local-mean window")
    ap.add_argument("--fg-thr", type=float, nargs="+",
                    default=[0.001, 0.002, 0.005, 0.01],
                    help="residual territory cut(s) to sweep")
    ap.add_argument("--floor", type=float, default=0.01,
                    help="ridge-residual noise cutoff (fixed)")
    ap.add_argument("--min-area", type=int, default=100)
    args = ap.parse_args()

    fg = tifffile.imread(FG_PATH).astype(np.float32)[:args.frames]
    contour = tifffile.imread(CONTOUR_PATH).astype(np.float32)[:args.frames]
    gt = tifffile.imread(GT_PATH)[:args.frames] if GT_PATH.exists() else None
    print(f"frames={fg.shape[0]}  w={args.w} wc={args.wc} fg_thr={args.fg_thr}")

    fg_resid = residual(fg, args.w | 1)
    ridge_resid = residual(contour, args.wc | 1)
    print(f"ridge_resid max={ridge_resid.max():.3f}")

    gt_per_frame = None
    if gt is not None:
        gt_per_frame = np.array([len(np.unique(g)) - 1 for g in gt])

    gt_mask = (gt > 0) if gt is not None else None
    atom_layers = {}
    terr_layers = {}
    print(f"\nterritory = fg_resid > thr")
    print(f"{'fg_thr':>8}{'terr_frac':>11}{'gt_cov':>9}{'atoms/frame':>13}{'gt/frame':>10}")
    for thr in args.fg_thr:
        territory = fg_resid > thr
        atoms = np.zeros(fg.shape, np.int32)
        counts = []
        for t in range(fg.shape[0]):
            a, _ = extract_atoms_frame(ridge_resid[t], territory[t], args.floor,
                                       args.min_area)
            atoms[t] = a
            counts.append(int(a.max()))
        atom_layers[f"atoms_t{thr:g}"] = atoms
        terr_layers[f"terr_t{thr:g}"] = territory.astype(np.uint8)
        gtm = gt_per_frame.mean() if gt_per_frame is not None else float("nan")
        gtcov = ((territory & gt_mask).sum() / gt_mask.sum()) if gt_mask is not None else float("nan")
        print(f"{thr:>8.4f}{territory.mean():>11.3f}{gtcov:>9.3f}"
              f"{np.mean(counts):>13.1f}{gtm:>10.1f}")

    if not args.gui:
        print("\n(pass --gui to inspect in napari)")
        return

    import napari
    v = napari.Viewer()
    v.add_image(fg, name="fg", colormap="gray")
    v.add_image(contour, name="contour", colormap="inferno", blending="additive",
                visible=False)
    v.add_image(ridge_resid, name="ridge_resid", colormap="magma",
                blending="additive", visible=False)
    for name, terr in terr_layers.items():
        v.add_labels(terr, name=name, opacity=0.25, visible=False)
    for i, (name, a) in enumerate(atom_layers.items()):
        v.add_labels(a, name=name, opacity=0.55, visible=(i == len(atom_layers) - 1))
    if gt is not None:
        v.add_labels(gt, name="GT", opacity=0.3, visible=False)
    napari.run()


if __name__ == "__main__":
    main()
