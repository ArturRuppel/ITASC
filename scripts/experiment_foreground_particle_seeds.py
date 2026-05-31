#!/usr/bin/env python
"""Inspect overpermissive particle seeds from the foreground score image.

This does not try to make final masks. It asks a simpler question: can the
foreground image provide useful nucleus seeds? Several particle/blob detectors
are run with permissive parameters and displayed as napari point layers.

Usage
-----
    python scripts/experiment_foreground_particle_seeds.py
    python scripts/experiment_foreground_particle_seeds.py --gui
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.feature import blob_dog, blob_doh, blob_log, peak_local_max
from skimage.filters import gaussian
from skimage.morphology import h_maxima


DATA = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/pos00"
)
FG_PATH = DATA / "1_cellpose" / "nucleus_foreground.tif"
CONTOUR_PATH = DATA / "1_cellpose" / "nucleus_contours.tif"
GT_PATH = DATA / "2_nucleus" / "tracked_labels.tif"


def _points_from_yx(t, yx):
    if len(yx) == 0:
        return np.empty((0, 3), dtype=np.float32)
    pts = np.asarray(yx, dtype=np.float32)
    return np.column_stack([
        np.full(pts.shape[0], float(t), dtype=np.float32),
        pts[:, 0],
        pts[:, 1],
    ])


def _points_from_blobs(t, blobs):
    if len(blobs) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    arr = np.asarray(blobs, dtype=np.float32)
    pts = np.column_stack([
        np.full(arr.shape[0], float(t), dtype=np.float32),
        arr[:, 0],
        arr[:, 1],
    ])
    sizes = np.maximum(3.0, arr[:, 2] * 2.5).astype(np.float32)
    return pts, sizes


def local_max_points(fg, *, sigma, threshold_abs, min_distance):
    frames = []
    for t in range(fg.shape[0]):
        img = fg[t]
        if sigma > 0:
            img = gaussian(img, sigma=sigma, preserve_range=True)
        yx = peak_local_max(
            img,
            threshold_abs=threshold_abs,
            min_distance=min_distance,
            exclude_border=False,
        )
        frames.append(_points_from_yx(t, yx))
    return np.concatenate(frames, axis=0)


def hmax_points(fg, *, h, sigma, min_area):
    frames = []
    struct = ndi.generate_binary_structure(2, 2)
    for t in range(fg.shape[0]):
        img = fg[t]
        if sigma > 0:
            img = gaussian(img, sigma=sigma, preserve_range=True)
        maxima = h_maxima(img, h)
        labels, n = ndi.label(maxima, structure=struct)
        if n == 0:
            frames.append(np.empty((0, 3), dtype=np.float32))
            continue
        centers = []
        for lab in range(1, n + 1):
            mask = labels == lab
            if int(mask.sum()) < min_area:
                continue
            centers.append(ndi.center_of_mass(mask))
        frames.append(_points_from_yx(t, centers))
    return np.concatenate(frames, axis=0)


def blob_points(fg, *, method, threshold, min_sigma, max_sigma, sigma_ratio=1.6):
    points_by_frame = []
    sizes_by_frame = []
    for t in range(fg.shape[0]):
        img = fg[t]
        if method == "log":
            blobs = blob_log(
                img,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=6,
                threshold=threshold,
                overlap=0.75,
                exclude_border=False,
            )
        elif method == "dog":
            blobs = blob_dog(
                img,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                sigma_ratio=sigma_ratio,
                threshold=threshold,
                overlap=0.75,
                exclude_border=False,
            )
        elif method == "doh":
            blobs = blob_doh(
                img,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=6,
                threshold=threshold,
                overlap=0.75,
            )
        else:
            raise ValueError(method)
        pts, sizes = _points_from_blobs(t, blobs)
        points_by_frame.append(pts)
        sizes_by_frame.append(sizes)
    return np.concatenate(points_by_frame, axis=0), np.concatenate(sizes_by_frame)


def gt_counts(gt):
    return np.array([max(len(np.unique(frame)) - 1, 0) for frame in gt], dtype=int)


def print_counts(name, points, n_frames, gt_per_frame):
    counts = np.bincount(points[:, 0].astype(int), minlength=n_frames) if len(points) else np.zeros(n_frames, dtype=int)
    print(f"{name:<32} mean={counts.mean():6.1f}  "
          f"min={counts.min():4d}  max={counts.max():4d}  "
          f"gt_mean={gt_per_frame.mean():6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--include-doh", action="store_true",
                    help="Also run DoH. It is slower and often less useful here.")
    args = ap.parse_args()

    fg = tifffile.imread(FG_PATH).astype(np.float32)
    contour = tifffile.imread(CONTOUR_PATH).astype(np.float32)
    gt = tifffile.imread(GT_PATH)
    if args.frames is not None:
        fg = fg[:args.frames]
        contour = contour[:args.frames]
        gt = gt[:args.frames]

    n_frames = fg.shape[0]
    gt_per_frame = gt_counts(gt)
    print(f"foreground {fg.shape}  range=[{fg.min():.3f},{fg.max():.3f}]")
    print(f"GT nuclei/frame mean={gt_per_frame.mean():.1f}  "
          f"min={gt_per_frame.min()}  max={gt_per_frame.max()}\n")

    layers: list[tuple[str, np.ndarray, float | np.ndarray, str]] = []

    for sigma, min_distance, thr in [
        (0.0, 4, 0.020),
        (1.0, 4, 0.020),
        (1.0, 6, 0.020),
        (1.0, 6, 0.050),
        (2.0, 8, 0.030),
        (2.0, 10, 0.050),
    ]:
        name = f"local_s{sigma:g}_d{min_distance}_t{thr:.3f}"
        pts = local_max_points(
            fg,
            sigma=sigma,
            threshold_abs=thr,
            min_distance=min_distance,
        )
        print_counts(name, pts, n_frames, gt_per_frame)
        layers.append((name, pts, 6.0, "cyan"))

    for h, sigma in [
        (0.003, 1.0),
        (0.005, 1.0),
        (0.010, 1.0),
        (0.020, 1.5),
    ]:
        name = f"hmax_h{h:.3f}_s{sigma:g}"
        pts = hmax_points(fg, h=h, sigma=sigma, min_area=1)
        print_counts(name, pts, n_frames, gt_per_frame)
        layers.append((name, pts, 6.0, "magenta"))

    for method, threshold in [
        ("log", 0.005),
        ("log", 0.010),
        ("log", 0.020),
        ("dog", 0.005),
        ("dog", 0.010),
        ("dog", 0.020),
    ]:
        name = f"blob_{method}_t{threshold:.3f}"
        pts, sizes = blob_points(
            fg,
            method=method,
            threshold=threshold,
            min_sigma=1.5,
            max_sigma=8.0,
        )
        print_counts(name, pts, n_frames, gt_per_frame)
        layers.append((name, pts, sizes, "yellow" if method == "log" else "lime"))

    if args.include_doh:
        for threshold in [0.001, 0.003, 0.005]:
            name = f"blob_doh_t{threshold:.3f}"
            pts, sizes = blob_points(
                fg,
                method="doh",
                threshold=threshold,
                min_sigma=2.0,
                max_sigma=10.0,
            )
            print_counts(name, pts, n_frames, gt_per_frame)
            layers.append((name, pts, sizes, "orange"))

    if not args.gui:
        print("\n(compute OK; pass --gui to open napari)")
        return

    import napari
    viewer = napari.Viewer()
    viewer.add_image(fg, name="foreground", colormap="gray")
    viewer.add_image(contour, name="contour_divergence", colormap="inferno",
                     blending="additive", visible=False)
    viewer.add_labels(gt, name="GT", opacity=0.35, visible=False)
    for i, (name, pts, size, color) in enumerate(layers):
        if len(pts) == 0:
            continue
        viewer.add_points(
            pts,
            name=name,
            size=size,
            face_color=color,
            opacity=0.8,
            visible=(i < 3),
        )
    napari.run()


if __name__ == "__main__":
    main()
