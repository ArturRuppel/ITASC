#!/usr/bin/env python
"""Experiment (Stage 1): rim-ridge atom extraction from contour + foreground maps.

Goal
----
Eyeball whether a *relative*, mostly parameter-free "rim-ridge" test can cleanly
separate real nuclei (including faint ones) from halo, before investing in any
database / permutation pipeline.

Idea
----
A good atom is a region whose border, crossed from inside to outside, shows a
contour profile of low -> high -> low (a ridge sits all the way around it) while
its interior has only low contour. Halo regions fail this: their *outer* rim is a
smooth intensity falloff with no contour ridge, so part of their boundary is not
elevated relative to their interior.

Pipeline per frame
-------------------
1. Admit territory: permissive background removal on the foreground map via
   Triangle thresholding (keeps faint signal; clear background removed).
2. Watershed the (lightly smoothed) contour map within the territory, seeded from
   contour regional minima -> candidate basins (low-contour pools).
3. Rim-ridge test: keep a basin as an atom only if a sufficient fraction of its
   1-px boundary ring has contour *above* the basin's interior contour level
   (boundary is a ridge relative to the interior). Halo basins, whose outer rim
   is ~flat/low, are dropped.

Outputs four napari layers for inspection: foreground, contour, all basins
(pre-drop), and kept atoms (post rim-ridge test). No database is written and
nothing in the data directory is modified.

Usage
-----
    python scripts/experiment_rim_ridge_atoms.py            # compute + print stats
    python scripts/experiment_rim_ridge_atoms.py --gui      # also open napari
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_triangle
from skimage.filters import gaussian
from skimage.morphology import h_minima
from skimage.segmentation import watershed

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
DATA_DIR = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/"
    "pos00/1_cellpose"
)
CONTOUR_PATH = DATA_DIR / "nucleus_contours.tif"
FOREGROUND_PATH = DATA_DIR / "nucleus_foreground.tif"

N_FRAMES = 10

# --------------------------------------------------------------------------- #
# Tunables (we want to drive these toward parameter-free as we iterate)
# --------------------------------------------------------------------------- #
SMOOTH_SIGMA = 1.5       # contour smoothing before minima / watershed
H_MINIMA = 0.2           # min depth (dynamics) of a contour minimum to seed a basin;
                         # suppresses shallow interior-texture dips that over-segment
MIN_BASIN_AREA = 100     # drop specks (px); median nucleus area in this data is ~490
RIM_RIDGE_FRACTION = 0.5  # min fraction of boundary that must be ridge-like to keep


def extract_atoms(contour: np.ndarray, foreground: np.ndarray):
    """Return (basins, atoms) label images for one 2-D frame.

    basins: all watershed basins inside the admitted territory.
    atoms:  basins that passed the rim-ridge test (halo dropped), relabelled 1..K.
    """
    contour = np.asarray(contour, dtype=np.float32)
    foreground = np.asarray(foreground, dtype=np.float32)

    # 1. territory ---------------------------------------------------------- #
    thr = float(threshold_triangle(foreground))
    territory = foreground > thr

    # 2. contour watershed -------------------------------------------------- #
    contour_s = gaussian(contour, sigma=SMOOTH_SIGMA, preserve_range=True)
    seeds = (h_minima(contour_s, H_MINIMA) > 0) & territory
    markers, _ = ndi.label(seeds)
    basins = watershed(contour_s, markers=markers, mask=territory)

    # 3. rim-ridge test ----------------------------------------------------- #
    atoms = np.zeros_like(basins, dtype=np.int32)
    keep_id = 1
    labels = [lbl for lbl in np.unique(basins) if lbl != 0]
    for lbl in labels:
        mask = basins == lbl
        area = int(mask.sum())
        if area < MIN_BASIN_AREA:
            continue
        eroded = ndi.binary_erosion(mask)
        boundary = mask & ~eroded
        interior = eroded if eroded.any() else mask
        interior_level = float(np.median(contour[interior]))
        boundary_vals = contour[boundary]
        if boundary_vals.size == 0:
            continue
        ridge_frac = float(np.mean(boundary_vals > interior_level))
        if ridge_frac >= RIM_RIDGE_FRACTION:
            atoms[mask] = keep_id
            keep_id += 1

    return basins.astype(np.int32), atoms


def main(show_gui: bool) -> None:
    contour_stack = tifffile.imread(str(CONTOUR_PATH))[:N_FRAMES].astype(np.float32)
    foreground_stack = tifffile.imread(str(FOREGROUND_PATH))[:N_FRAMES].astype(np.float32)
    assert contour_stack.shape == foreground_stack.shape, (
        contour_stack.shape, foreground_stack.shape
    )

    basins_stack = np.zeros(contour_stack.shape, dtype=np.int32)
    atoms_stack = np.zeros(contour_stack.shape, dtype=np.int32)

    print(f"frames={contour_stack.shape[0]}  shape={contour_stack.shape[1:]}")
    print(f"{'t':>3} {'basins':>7} {'atoms':>6} {'dropped':>8}")
    for t in range(contour_stack.shape[0]):
        basins, atoms = extract_atoms(contour_stack[t], foreground_stack[t])
        basins_stack[t] = basins
        atoms_stack[t] = atoms
        n_basins = int(basins.max())
        n_atoms = int(atoms.max())
        print(f"{t:>3} {n_basins:>7} {n_atoms:>6} {n_basins - n_atoms:>8}")

    if not show_gui:
        print("\n(compute OK; pass --gui to open napari)")
        return

    import napari

    viewer = napari.Viewer()
    viewer.add_image(foreground_stack, name="foreground", colormap="gray")
    viewer.add_image(
        contour_stack, name="contour", colormap="inferno", blending="additive"
    )
    viewer.add_labels(basins_stack, name="basins_all", opacity=0.4)
    viewer.add_labels(atoms_stack, name="atoms", opacity=0.6)
    napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true", help="open napari viewer")
    args = parser.parse_args()
    main(show_gui=args.gui)
