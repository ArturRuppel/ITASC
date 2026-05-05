"""Compare Cellpose mask output at two niter values for Z-mean dp projection.

Frame T=10, cellprob_threshold=-10, flow_threshold=0.
niter=200 (current) vs niter=1000 (5x).

Saves masks as TIFs and prints comparison statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROB = Path("/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/1_cellpose/cell_prob_3dt.tif")
DP   = Path("/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/1_cellpose/cell_dp_3dt.tif")

T = 10
CELLPROB_THRESHOLD = -10.0
FLOW_THRESHOLD = 0.0
NITERS = [200, 1000]

OUT_DIR = REPO_ROOT / "scripts" / "niter_sweep_out"
OUT_DIR.mkdir(exist_ok=True)


def mask_stats(masks: np.ndarray, label: str) -> None:
    ids = np.unique(masks)
    ids = ids[ids > 0]
    n = len(ids)
    if n == 0:
        print(f"  {label}: 0 cells")
        return
    areas = np.array([np.sum(masks == i) for i in ids])
    print(
        f"  {label}: n={n:4d}  "
        f"mean={areas.mean():.0f}  median={np.median(areas):.0f}  "
        f"max={areas.max():.0f}  min={areas.min():.0f} px²"
    )


def run() -> None:
    from cellpose.dynamics import compute_masks

    print(f"Loading T={T}...")
    prob_zyx = tifffile.imread(PROB)[T].astype(np.float32)   # (Z, Y, X)
    dp_zcyx  = tifffile.imread(DP)[T].astype(np.float32)     # (Z, C, Y, X)

    print(f"  prob shape: {prob_zyx.shape}, dp shape: {dp_zcyx.shape}")

    projected_prob = prob_zyx.mean(axis=0)    # (Y, X)
    projected_dp   = dp_zcyx.mean(axis=0)    # (C, Y, X)

    above = (projected_prob > CELLPROB_THRESHOLD).sum()
    print(f"  Pixels above cellprob_threshold={CELLPROB_THRESHOLD}: {above} / {projected_prob.size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    results: dict[int, np.ndarray] = {}

    for niter in NITERS:
        print(f"Running compute_masks niter={niter}...")
        masks = compute_masks(
            projected_dp,
            projected_prob,
            cellprob_threshold=CELLPROB_THRESHOLD,
            flow_threshold=FLOW_THRESHOLD,
            niter=niter,
            do_3D=False,
            device=device,
        )
        if isinstance(masks, tuple):
            masks = masks[0]
        masks = np.asarray(masks, dtype=np.uint32)
        results[niter] = masks

        out_path = OUT_DIR / f"masks_T{T:02d}_niter{niter}.tif"
        tifffile.imwrite(out_path, masks.astype(np.uint16))
        print(f"  Saved: {out_path}")
        mask_stats(masks, f"niter={niter}")
        print()

    # Overlap analysis
    m200  = results[NITERS[0]]
    m1000 = results[NITERS[1]]
    agree = (m200 > 0) == (m1000 > 0)
    print(f"Foreground agreement (both masked or both background): {agree.mean()*100:.1f}%")
    print(f"  niter={NITERS[0]} fg: {(m200>0).sum()} px")
    print(f"  niter={NITERS[1]} fg: {(m1000>0).sum()} px")


if __name__ == "__main__":
    run()
