#!/usr/bin/env python
"""Inspect binarized divergence contour maps.

The useful edge signal is already in ``nucleus_contours.tif``: it is positive
flow divergence reduced to a scalar contour/ridge map. This script thresholds
that scalar directly and shows the resulting binary contour bands, optionally
with skeletonized centerlines.

Usage
-----
    python scripts/experiment_binary_contours.py
    python scripts/experiment_binary_contours.py --gui
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.morphology import skeletonize


DATA = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/pos00"
)
CONTOUR_PATH = DATA / "1_cellpose" / "nucleus_contours.tif"
FG_PATH = DATA / "1_cellpose" / "nucleus_foreground.tif"
GT_PATH = DATA / "2_nucleus" / "tracked_labels.tif"


def binary_contours(contour, floor, sigma):
    out = np.zeros(contour.shape, dtype=bool)
    for t in range(contour.shape[0]):
        frame = contour[t]
        if sigma > 0:
            frame = gaussian(frame, sigma=sigma, preserve_range=True)
        out[t] = frame >= floor
    return out


def skeleton_contours(binary):
    out = np.zeros(binary.shape, dtype=bool)
    for t in range(binary.shape[0]):
        out[t] = skeletonize(binary[t])
    return out


def region_counts(edges):
    return np.array([ndi.label(~edges[t])[1] for t in range(edges.shape[0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="Gaussian smoothing before thresholding")
    ap.add_argument("--floors", type=float, nargs="+",
                    default=[0.005, 0.010, 0.020, 0.037, 0.075, 0.100, 0.200])
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--no-skeleton", action="store_true")
    args = ap.parse_args()

    contour = tifffile.imread(CONTOUR_PATH).astype(np.float32)
    fg = tifffile.imread(FG_PATH).astype(np.float32)
    gt = tifffile.imread(GT_PATH) if GT_PATH.exists() else None
    if args.frames is not None:
        contour = contour[:args.frames]
        fg = fg[:args.frames]
        gt = gt[:args.frames] if gt is not None else None

    print(f"contour {contour.shape}  max={contour.max():.3f}")
    qs = np.percentile(contour[contour > 0], [50, 75, 90, 95, 99])
    print("positive contour percentiles: "
          + " ".join(f"p{p}={q:.3f}" for p, q in zip([50, 75, 90, 95, 99], qs)))

    binary_layers = {}
    skeleton_layers = {}
    for floor in args.floors:
        band = binary_contours(contour, floor=floor, sigma=args.sigma)
        binary_layers[f"binary_contour>={floor:.3f}"] = band
        print(f"floor={floor:.3f}  band-frac={band.mean():.3f}", end="")
        if not args.no_skeleton:
            skel = skeleton_contours(band)
            skeleton_layers[f"skeleton_contour>={floor:.3f}"] = skel
            counts = region_counts(skel)
            print(f"  skeleton-frac={skel.mean():.3f}  "
                  f"skeleton-regions/frame={counts.mean():.0f}")
        else:
            print()

    if not args.gui:
        print("\n(compute OK; pass --gui to open napari)")
        return

    import napari
    viewer = napari.Viewer()
    viewer.add_image(fg, name="foreground", colormap="gray")
    viewer.add_image(contour, name="contour_divergence", colormap="inferno",
                     blending="additive")
    for i, (name, band) in enumerate(binary_layers.items()):
        viewer.add_labels(band.astype(np.uint8), name=name, opacity=0.45,
                          visible=(i == 0))
    for name, skel in skeleton_layers.items():
        viewer.add_labels(skel.astype(np.uint8), name=name, opacity=0.75,
                          visible=False)
    if gt is not None:
        viewer.add_labels(gt, name="GT", opacity=0.35, visible=False)
    napari.run()


if __name__ == "__main__":
    main()
