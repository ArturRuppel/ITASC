#!/usr/bin/env python
"""Open the atom-DB tracking result against foreground + GT (no recompute)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

from experiment_residual_atoms import FG_PATH, CONTOUR_PATH, GT_PATH, DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--workdir", type=str,
                    default=str(DATA / "2_nucleus" / "atoms_ultrack_workdir"))
    args = ap.parse_args()

    labels = tifffile.imread(Path(args.workdir) / "atoms_tracked_labels.tif")
    n = min(args.frames, labels.shape[0])
    fg = tifffile.imread(FG_PATH).astype(np.float32)[:n]
    contour = tifffile.imread(CONTOUR_PATH).astype(np.float32)[:n]
    gt = tifffile.imread(GT_PATH)[:n] if GT_PATH.exists() else None

    import napari
    v = napari.Viewer()
    v.add_image(fg, name="fg", colormap="gray")
    v.add_image(contour, name="contour", colormap="inferno",
                blending="additive", visible=False)
    v.add_labels(labels[:n].astype(np.int32), name="atoms_tracked", opacity=0.6)
    if gt is not None:
        v.add_labels(gt, name="GT", opacity=0.4, visible=False)
    napari.run()


if __name__ == "__main__":
    main()
