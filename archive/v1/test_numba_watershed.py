"""Test script: Numba per-pixel flow-guided watershed on frame t=0.

Loads inputs from pos00, runs the new deterministic Numba watershed on the
first frame, then opens napari with the inputs and result for visual comparison.

Usage:
    conda run -n cellflow python test_numba_watershed.py
"""

from __future__ import annotations

import math
import time

import numba
import numpy as np
import tifffile

DATA_ROOT = (
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/analysis/pos00"
)

# ── Parameters (matching previous run_params.json) ───────────────────────────
FLOW_SCALE = 3.0
FLOW_SMOOTHING_SIGMA = 0.5
PROB_THRESHOLD = 0.0        # applied to sigmoid(cell_prob); 0.0 = no mask
UNIFORM_GROWTH_RATE = 0.2   # also used as expansion threshold
MAX_ITERATIONS = 80


# ── Numba kernel ─────────────────────────────────────────────────────────────

@numba.njit(parallel=True, cache=True)
def _watershed_step(
    result: np.ndarray,       # (H, W) int32 — read-only snapshot
    next_result: np.ndarray,  # (H, W) int32 — write output here
    flow: np.ndarray,         # (H, W, 2) float32
    centroid_y: np.ndarray,   # (max_label+1,) float32
    centroid_x: np.ndarray,   # (max_label+1,) float32
    prob_mask: np.ndarray,    # (H, W) bool
    flow_scale: float,
    uniform_growth_rate: float,
) -> int:
    """Single expansion step. Returns number of newly assigned pixels."""
    H, W = result.shape
    changed = 0

    for i in numba.prange(H):
        for j in range(W):
            cur = result[i, j]
            if cur != 0:
                next_result[i, j] = cur
                continue
            if not prob_mask[i, j]:
                continue

            best_label = 0
            best_prob = -1.0  # below min possible prob; flow alignment decides *which* label wins, not *whether*

            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    if ni < 0 or ni >= H or nj < 0 or nj >= W:
                        continue
                    L = result[ni, nj]
                    if L <= 0:
                        continue

                    # Radial direction from centroid to current pixel
                    ry = float(i) - centroid_y[L]
                    rx = float(j) - centroid_x[L]
                    norm = math.sqrt(ry * ry + rx * rx)
                    if norm > 1e-6:
                        ry /= norm
                        rx /= norm

                    # Flow points toward cell center → negate for outward direction
                    align = ry * (-flow[i, j, 0]) + rx * (-flow[i, j, 1])
                    boost = min(max(align * flow_scale, 0.0), 1.0)
                    prob = uniform_growth_rate + (1.0 - uniform_growth_rate) * boost

                    if prob > best_prob:
                        best_prob = prob
                        best_label = L

            if best_label > 0:
                next_result[i, j] = best_label
                changed += 1

    return changed


def _compute_centroids(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centroid_y, centroid_x arrays indexed by label id."""
    max_label = int(labels.max())
    centroid_y = np.zeros(max_label + 1, dtype=np.float32)
    centroid_x = np.zeros(max_label + 1, dtype=np.float32)
    for lid in range(1, max_label + 1):
        pts = np.argwhere(labels == lid)
        if len(pts):
            centroid_y[lid] = pts[:, 0].mean()
            centroid_x[lid] = pts[:, 1].mean()
    return centroid_y, centroid_x


def flow_watershed_numba(
    nuclear_labels: np.ndarray,   # (H, W) int32
    flow_field: np.ndarray,       # (H, W, 2) float32
    prob_mask: np.ndarray,        # (H, W) bool
    flow_scale: float = FLOW_SCALE,
    uniform_growth_rate: float = UNIFORM_GROWTH_RATE,
    max_iterations: int = MAX_ITERATIONS,
) -> np.ndarray:
    """Deterministic Numba per-pixel flow-guided watershed."""
    result = nuclear_labels.copy().astype(np.int32)
    next_result = np.zeros_like(result)
    centroid_y, centroid_x = _compute_centroids(result)

    for it in range(max_iterations):
        next_result[:] = 0
        changed = _watershed_step(
            result, next_result, flow_field,
            centroid_y, centroid_x, prob_mask,
            flow_scale, uniform_growth_rate,
        )
        result, next_result = next_result, result  # swap buffers
        print(f"  iter {it+1:3d}: {changed:6d} pixels assigned", flush=True)
        if changed == 0:
            print(f"  converged at iteration {it+1}", flush=True)
            break

    return result


# ── Load data ─────────────────────────────────────────────────────────────────

def load_frame0():
    print("Loading inputs…")
    nuc_stack = tifffile.imread(f"{DATA_ROOT}/3_correction/nuclear_labels_corrected.tif")
    dp_stack  = tifffile.imread(f"{DATA_ROOT}/1_cellpose/cell/cell_dp.tif")
    prob_stack = tifffile.imread(f"{DATA_ROOT}/1_cellpose/cell/cell_prob.tif")

    print(f"  nuclear_labels: {nuc_stack.shape} {nuc_stack.dtype}")
    print(f"  cell_dp:        {dp_stack.shape}  {dp_stack.dtype}  range [{dp_stack.min():.2f}, {dp_stack.max():.2f}]")
    print(f"  cell_prob:      {prob_stack.shape} {prob_stack.dtype} range [{prob_stack.min():.2f}, {prob_stack.max():.2f}]")

    # Frame 0
    nuc   = nuc_stack[0].astype(np.int32)        # (H, W)
    dp    = dp_stack[0]                           # (2, H, W)
    prob  = prob_stack[0]                         # (H, W) — raw logits

    # (2, H, W) → (H, W, 2)
    flow = np.transpose(dp, (1, 2, 0)).astype(np.float32)

    # cellpose prob is raw logits → convert to [0,1] with sigmoid
    prob_sigmoid = (1.0 / (1.0 + np.exp(-prob))).astype(np.float32)
    print(f"  prob_sigmoid range: [{prob_sigmoid.min():.3f}, {prob_sigmoid.max():.3f}]")

    if FLOW_SMOOTHING_SIGMA > 0:
        from scipy.ndimage import gaussian_filter
        flow = np.stack([
            gaussian_filter(flow[..., 0], sigma=FLOW_SMOOTHING_SIGMA),
            gaussian_filter(flow[..., 1], sigma=FLOW_SMOOTHING_SIGMA),
        ], axis=-1).astype(np.float32)
        print(f"  flow smoothed (sigma={FLOW_SMOOTHING_SIGMA})")

    prob_mask = prob_sigmoid >= PROB_THRESHOLD
    print(f"  {prob_mask.sum()} / {prob_mask.size} pixels in foreground mask")
    print(f"  {(nuc > 0).sum()} seed pixels, {np.unique(nuc[nuc>0]).size} cells")

    return nuc, flow, prob_sigmoid, prob_mask


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    nuc, flow, prob, prob_mask = load_frame0()

    # Warm-up JIT compilation on a tiny dummy array
    print("\nWarming up Numba JIT…", flush=True)
    _dummy_labels = np.zeros((4, 4), dtype=np.int32)
    _dummy_labels[1, 1] = 1
    _dummy_flow = np.zeros((4, 4, 2), dtype=np.float32)
    _dummy_mask = np.ones((4, 4), dtype=np.bool_)
    _dummy_cy = np.array([0.0, 1.0], dtype=np.float32)
    _dummy_cx = np.array([0.0, 1.0], dtype=np.float32)
    _dummy_next = np.zeros((4, 4), dtype=np.int32)
    _watershed_step(_dummy_labels, _dummy_next, _dummy_flow, _dummy_cy, _dummy_cx, _dummy_mask, 1.0, 0.2)
    print("  JIT ready.", flush=True)

    print("\nRunning Numba flow watershed on frame t=0…", flush=True)
    t0 = time.perf_counter()
    result = flow_watershed_numba(nuc, flow, prob_mask)
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s.  Unique cells: {np.unique(result[result>0]).size}")

    # ── Load old result (frame 0) ─────────────────────────────────────────────
    old_path = f"{DATA_ROOT}/4_cell_segmentation/cell_labels_raw.tif"
    print(f"\nLoading old result from {old_path}…")
    old_stack = tifffile.imread(old_path)
    old_result = old_stack[0].astype(np.int32)
    print(f"  old result: {np.unique(old_result[old_result>0]).size} cells")

    # ── Launch napari ─────────────────────────────────────────────────────────
    print("\nLaunching napari…")
    import napari

    viewer = napari.Viewer(title="Flow Watershed — frame t=0: old vs new")
    viewer.add_image(prob, name="cell_prob (sigmoid)", colormap="magma", opacity=0.5)
    viewer.add_labels(nuc, name="nuclear seeds", opacity=0.8)
    viewer.add_labels(old_result, name="OLD (stochastic iterative)", opacity=0.6)
    viewer.add_labels(result, name="NEW (numba deterministic)", opacity=0.6)

    napari.run()


if __name__ == "__main__":
    main()
