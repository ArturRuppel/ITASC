"""Test script: Flow-aligned priority-queue seeded region growing on frame t=0.

Expands nuclear seeds using a Dijkstra-style priority queue where priority is
determined by how well each pixel's flow vector aligns with the outward radial
direction from the candidate label's centroid.  No Poisson solve required —
the flow field guides assignment everywhere, including at weak/invisible edges.

Loads the old stochastic result for side-by-side comparison in napari.

Usage:
    conda run -n cellflow python test_poisson_watershed.py
"""

from __future__ import annotations

import heapq
import math
import time
import numpy as np
import tifffile

DATA_ROOT = (
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/analysis/pos00"
)

FLOW_SMOOTHING_SIGMA = 0.5
FLOW_SCALE = 3.0
UNIFORM_GROWTH_RATE = 0.2

DIRS = [(di, dj) for di in range(-1, 2) for dj in range(-1, 2) if not (di == 0 and dj == 0)]


# ── Core: flow-aligned priority-queue expansion ───────────────────────────────

def _compute_centroids(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_label = int(labels.max())
    centroid_y = np.zeros(max_label + 1, dtype=np.float32)
    centroid_x = np.zeros(max_label + 1, dtype=np.float32)
    for lid in range(1, max_label + 1):
        pts = np.argwhere(labels == lid)
        if len(pts):
            centroid_y[lid] = pts[:, 0].mean()
            centroid_x[lid] = pts[:, 1].mean()
    return centroid_y, centroid_x


def _score(i, j, L, cy, cx, flow, flow_scale, ugr):
    ry = float(i) - cy[L]
    rx = float(j) - cx[L]
    norm = math.sqrt(ry * ry + rx * rx)
    if norm > 1e-6:
        ry /= norm
        rx /= norm
    align = ry * (-flow[i, j, 0]) + rx * (-flow[i, j, 1])
    boost = min(max(align * flow_scale, 0.0), 1.0)
    return ugr + (1.0 - ugr) * boost


def flow_priority_watershed(
    nuclear_labels: np.ndarray,   # (H, W) int32
    flow_field: np.ndarray,       # (H, W, 2) float32
    flow_scale: float = FLOW_SCALE,
    uniform_growth_rate: float = UNIFORM_GROWTH_RATE,
) -> np.ndarray:
    """Dijkstra-style seeded region growing guided by flow alignment.

    Each unlabeled pixel is assigned to the label whose centroid best aligns
    with the pixel's flow vector.  Priority queue ensures the highest-confidence
    assignment is always made first.
    """
    H, W = nuclear_labels.shape
    result = nuclear_labels.copy().astype(np.int32)
    cy, cx = _compute_centroids(result)

    heap: list = []  # (-score, i, j, label)

    # Seed: all unlabeled pixels adjacent to a seed
    seed_ys, seed_xs = np.where(result > 0)
    for si, sj in zip(seed_ys.tolist(), seed_xs.tolist()):
        L = result[si, sj]
        for di, dj in DIRS:
            ni, nj = si + di, sj + dj
            if 0 <= ni < H and 0 <= nj < W and result[ni, nj] == 0:
                s = _score(ni, nj, L, cy, cx, flow_field, flow_scale, uniform_growth_rate)
                heapq.heappush(heap, (-s, ni, nj, L))

    assigned = 0
    while heap:
        neg_s, i, j, L = heapq.heappop(heap)
        if result[i, j] != 0:
            continue
        result[i, j] = L
        assigned += 1
        for di, dj in DIRS:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and result[ni, nj] == 0:
                s = _score(ni, nj, L, cy, cx, flow_field, flow_scale, uniform_growth_rate)
                heapq.heappush(heap, (-s, ni, nj, L))

    print(f"  assigned {assigned} pixels", flush=True)
    return result


# ── Load data ─────────────────────────────────────────────────────────────────

def load_frame0():
    print("Loading inputs…")
    nuc_stack  = tifffile.imread(f"{DATA_ROOT}/3_correction/nuclear_labels_corrected.tif")
    dp_stack   = tifffile.imread(f"{DATA_ROOT}/1_cellpose/cell/cell_dp.tif")
    prob_stack = tifffile.imread(f"{DATA_ROOT}/1_cellpose/cell/cell_prob.tif")

    print(f"  nuclear_labels: {nuc_stack.shape} {nuc_stack.dtype}")
    print(f"  cell_dp:        {dp_stack.shape}  range [{dp_stack.min():.2f}, {dp_stack.max():.2f}]")
    print(f"  cell_prob:      {prob_stack.shape} range [{prob_stack.min():.2f}, {prob_stack.max():.2f}]")

    nuc  = nuc_stack[0].astype(np.int32)
    dp   = dp_stack[0]                        # (2, H, W)
    prob = prob_stack[0]                      # (H, W) raw logits

    # (2, H, W) → (H, W, 2)
    flow = np.transpose(dp, (1, 2, 0)).astype(np.float32)

    if FLOW_SMOOTHING_SIGMA > 0:
        from scipy.ndimage import gaussian_filter
        flow = np.stack([
            gaussian_filter(flow[..., 0], sigma=FLOW_SMOOTHING_SIGMA),
            gaussian_filter(flow[..., 1], sigma=FLOW_SMOOTHING_SIGMA),
        ], axis=-1).astype(np.float32)

    # Sigmoid probability for display / optional masking
    prob_sigmoid = (1.0 / (1.0 + np.exp(-prob))).astype(np.float32)

    return nuc, flow, prob_sigmoid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    nuc, flow, prob = load_frame0()

    # ── Flow-aligned priority-queue expansion ─────────────────────────────────
    print("\nRunning flow-aligned priority-queue watershed…", flush=True)
    t0 = time.perf_counter()
    result = flow_priority_watershed(nuc, flow)
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s  |  unique cells: {np.unique(result[result > 0]).size}")

    # ── Load old result for comparison ────────────────────────────────────────
    old_path = f"{DATA_ROOT}/4_cell_segmentation/cell_labels_raw.tif"
    print(f"\nLoading old result from {old_path}…")
    old_result = tifffile.imread(old_path)[0].astype(np.int32)

    # ── Launch napari ─────────────────────────────────────────────────────────
    print("\nLaunching napari…")
    import napari

    viewer = napari.Viewer(title="Flow priority watershed — frame t=0")
    viewer.add_image(prob,       name="cell_prob (sigmoid)",        colormap="magma",  opacity=0.5)
    viewer.add_labels(nuc,       name="nuclear seeds",              opacity=0.8)
    viewer.add_labels(old_result, name="OLD (stochastic iterative)", opacity=0.6)
    viewer.add_labels(result,    name="NEW (flow priority queue)",  opacity=0.6)

    napari.run()


if __name__ == "__main__":
    main()
