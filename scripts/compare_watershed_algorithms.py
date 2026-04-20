"""Compare old stochastic vs new deterministic flow-guided watershed on pos02 frame 10.

Usage:
    conda run -n cellflow python scripts/compare_watershed_algorithms.py
"""

from __future__ import annotations

import math
import time

import numba
import numpy as np
import tifffile
from scipy import ndimage

DATA_ROOT = (
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/analysis/pos02"
)
FRAME = 10

# Parameters
PROB_THRESHOLD = 0.0
# Stochastic legacy params
FLOW_SCALE_STOCH = 1.0
UNIFORM_GROWTH_RATE = 0.2
MAX_ITERATIONS_STOCH = 50
# New Euler integration params
FLOW_STEP_SCALE = 0.2
MAX_STEPS = 100
CAPTURE_RADIUS = 3.0
FLOW_WEIGHT = 0.5
GRAVITY_FALLOFF = 2.0


# ── OLD: stochastic iterative (from ultrack_wrapper origin, commit 8c0620c) ──

def _stochastic_watershed(
    nuclear_labels: np.ndarray,
    flow_field: np.ndarray,
    cellpose_prob: np.ndarray | None = None,
    flow_scale: float = FLOW_SCALE_STOCH,
    cellpose_prob_threshold: float = PROB_THRESHOLD,
    max_iterations: int = MAX_ITERATIONS_STOCH,
    uniform_growth_rate: float = UNIFORM_GROWTH_RATE,
) -> np.ndarray:
    """Original stochastic iterative expansion (per-label, numpy, random sampling)."""
    H, W = nuclear_labels.shape

    centroids = {}
    for label_id in np.unique(nuclear_labels):
        if label_id > 0:
            pts = np.argwhere(nuclear_labels == label_id)
            if len(pts) > 0:
                centroids[label_id] = pts.mean(axis=0)

    prob_mask = None
    if cellpose_prob is not None:
        prob_mask = cellpose_prob >= cellpose_prob_threshold

    result = nuclear_labels.copy().astype(np.int32)

    for iteration in range(max_iterations):
        unassigned = result == 0
        if not unassigned.any():
            break

        for label_id in np.unique(result[result > 0]):
            if label_id not in centroids:
                continue

            labeled = result == label_id
            dilated = ndimage.binary_dilation(labeled)
            boundary_candidates = dilated & unassigned
            if not boundary_candidates.any():
                continue

            center = centroids[label_id]
            boundary_points = np.argwhere(boundary_candidates)

            radial_dirs = boundary_points - center
            radial_norms = np.linalg.norm(radial_dirs, axis=1, keepdims=True)
            radial_dirs = np.where(radial_norms > 1e-6, radial_dirs / radial_norms, 0)

            neg_flow = -flow_field[boundary_points[:, 0], boundary_points[:, 1]]
            flow_alignment = np.sum(radial_dirs * neg_flow, axis=1)

            flow_boost = np.clip(flow_alignment * flow_scale, 0, 1)
            expand_prob = uniform_growth_rate + (1.0 - uniform_growth_rate) * flow_boost

            if prob_mask is not None:
                expand_prob = np.where(
                    prob_mask[boundary_points[:, 0], boundary_points[:, 1]],
                    expand_prob, 0.0,
                )

            expand_mask = np.random.rand(len(boundary_points)) < expand_prob
            expand_coords = boundary_points[expand_mask]
            if len(expand_coords) > 0:
                result[expand_coords[:, 0], expand_coords[:, 1]] = label_id

    return result


# ── NEW: Euler flow integration (cellpose dynamics approach) ──────────────────

@numba.njit(parallel=True, cache=True)
def _gravity_from_centroids(
    centroids_y: np.ndarray,
    centroids_x: np.ndarray,
    H: int,
    W: int,
    falloff: float,
) -> tuple:
    N = len(centroids_y)
    grav_y = np.zeros((H, W), dtype=np.float32)
    grav_x = np.zeros((H, W), dtype=np.float32)
    for i in numba.prange(H):
        for j in range(W):
            gy = 0.0
            gx = 0.0
            for k in range(N):
                dy = centroids_y[k] - float(i)
                dx = centroids_x[k] - float(j)
                dist = math.sqrt(dy * dy + dx * dx)
                if dist > 1e-6:
                    w = 1.0 / math.pow(dist, falloff + 1.0)
                    gy += dy * w
                    gx += dx * w
            mag = math.sqrt(gy * gy + gx * gx)
            if mag > 1e-6:
                grav_y[i, j] = gy / mag
                grav_x[i, j] = gx / mag
    return grav_y, grav_x

@numba.njit(parallel=True, cache=True)
def _flow_integrate(
    nuclear_labels: np.ndarray,
    flow: np.ndarray,
    grav_y: np.ndarray,
    grav_x: np.ndarray,
    dist_to_nucleus: np.ndarray,
    nearest_y: np.ndarray,
    nearest_x: np.ndarray,
    prob_mask: np.ndarray,
    n_steps: int,
    flow_step_scale: float,
    flow_weight: float,
    capture_radius: float,
) -> np.ndarray:
    H, W = nuclear_labels.shape
    result = nuclear_labels.copy()

    for i in numba.prange(H):
        for j in range(W):
            if result[i, j] > 0:
                continue
            if not prob_mask[i, j]:
                continue

            py = float(i)
            px = float(j)
            label = 0

            for _ in range(n_steps):
                iy0 = int(py)
                ix0 = int(px)
                iy0 = max(0, min(H - 2, iy0))
                ix0 = max(0, min(W - 2, ix0))

                fy = py - float(iy0)
                fx = px - float(ix0)

                flow_y = (flow[iy0,     ix0,     0] * (1.0 - fy) * (1.0 - fx) +
                          flow[iy0 + 1, ix0,     0] * fy          * (1.0 - fx) +
                          flow[iy0,     ix0 + 1, 0] * (1.0 - fy) * fx          +
                          flow[iy0 + 1, ix0 + 1, 0] * fy          * fx)

                flow_x = (flow[iy0,     ix0,     1] * (1.0 - fy) * (1.0 - fx) +
                          flow[iy0 + 1, ix0,     1] * fy          * (1.0 - fx) +
                          flow[iy0,     ix0 + 1, 1] * (1.0 - fy) * fx          +
                          flow[iy0 + 1, ix0 + 1, 1] * fy          * fx)

                w = flow_weight

                iy_nn = max(0, min(H - 1, int(py + 0.5)))
                ix_nn = max(0, min(W - 1, int(px + 0.5)))

                step_y = w * flow_y + (1.0 - w) * grav_y[iy_nn, ix_nn]
                step_x = w * flow_x + (1.0 - w) * grav_x[iy_nn, ix_nn]

                py = max(0.0, min(float(H - 1), py + step_y * flow_step_scale))
                px = max(0.0, min(float(W - 1), px + step_x * flow_step_scale))

                iy = max(0, min(H - 1, int(py + 0.5)))
                ix = max(0, min(W - 1, int(px + 0.5)))

                if dist_to_nucleus[iy, ix] <= capture_radius:
                    L = nuclear_labels[nearest_y[iy, ix], nearest_x[iy, ix]]
                    if L > 0:
                        label = L
                        break

            result[i, j] = label

    return result


def _euler_flow_segmentation(
    nuclear_labels: np.ndarray,
    flow_field: np.ndarray,
    cellpose_prob: np.ndarray | None = None,
    flow_step_scale: float = FLOW_STEP_SCALE,
    cellpose_prob_threshold: float = PROB_THRESHOLD,
    n_steps: int = MAX_STEPS,
    capture_radius: float = CAPTURE_RADIUS,
    flow_weight: float = FLOW_WEIGHT,
    gravity_falloff: float = GRAVITY_FALLOFF,
) -> np.ndarray:
    """Euler integration blending cellpose flow with N-body gravitational field."""
    from scipy.ndimage import distance_transform_edt

    if cellpose_prob is not None:
        prob_mask = (cellpose_prob >= cellpose_prob_threshold)
    else:
        prob_mask = np.ones(nuclear_labels.shape[:2], dtype=np.bool_)
    prob_mask = prob_mask | (nuclear_labels > 0)

    flow_mags = np.sqrt((flow_field ** 2).sum(axis=-1))
    mean_mag = float(flow_mags[prob_mask].mean()) if prob_mask.any() else 1.0
    if mean_mag > 1e-6:
        flow_field = (flow_field / mean_mag).astype(np.float32)

    dist_to_nucleus, (nearest_y, nearest_x) = distance_transform_edt(
        nuclear_labels == 0, return_indices=True
    )
    dist_to_nucleus = dist_to_nucleus.astype(np.float32)
    nearest_y = nearest_y.astype(np.int32)
    nearest_x = nearest_x.astype(np.int32)

    gy, gx = compute_gravity(nuclear_labels, falloff=gravity_falloff)

    result = _flow_integrate(
        nuclear_labels.astype(np.int32),
        flow_field.astype(np.float32),
        gy.astype(np.float32),
        gx.astype(np.float32),
        dist_to_nucleus,
        nearest_y,
        nearest_x,
        prob_mask.astype(np.bool_),
        int(n_steps),
        float(flow_step_scale),
        float(flow_weight),
        float(capture_radius),
    )

    unfilled = (result == 0) & prob_mask
    if unfilled.any():
        filled = result.copy()
        _, (yi, xi) = distance_transform_edt(filled == 0, return_indices=True)
        filled[unfilled] = result[yi[unfilled], xi[unfilled]]
        result = filled

    return result


# ── Field helpers ─────────────────────────────────────────────────────────────

def field_angle(fy: np.ndarray, fx: np.ndarray) -> np.ndarray:
    """Direction of a 2D vector field as degrees in [0, 360), for HSV display."""
    return (np.degrees(np.arctan2(fy, fx)) % 360).astype(np.float32)


def field_mag(fy: np.ndarray, fx: np.ndarray) -> np.ndarray:
    return np.sqrt(fy ** 2 + fx ** 2).astype(np.float32)


def compute_gravity(nuc: np.ndarray, falloff: float = GRAVITY_FALLOFF):
    """N-body gravity field: sum inverse-power contributions from all nuclear centroids."""
    from scipy.ndimage import center_of_mass
    label_ids = np.unique(nuc[nuc > 0])
    coords = center_of_mass(nuc > 0, nuc, label_ids)
    cy = np.array([c[0] for c in coords], dtype=np.float32)
    cx = np.array([c[1] for c in coords], dtype=np.float32)
    H, W = nuc.shape
    gy, gx = _gravity_from_centroids(cy, cx, H, W, float(falloff))
    gy[nuc > 0] = 0.0
    gx[nuc > 0] = 0.0
    return gy, gx


def blend_fields(flow_y, flow_x, grav_y, grav_x, alpha: float):
    """alpha = fraction of cellpose flow (0 = pure gravity, 1 = pure flow)."""
    return alpha * flow_y + (1 - alpha) * grav_y, alpha * flow_x + (1 - alpha) * grav_x


# ── Data loading ──────────────────────────────────────────────────────────────

def load_frame(frame: int):
    print(f"Loading frame {frame} from {DATA_ROOT}…")
    nuc_stack  = tifffile.imread(f"{DATA_ROOT}/3_correction/nuclear_labels_corrected.tif")
    dp_stack   = tifffile.imread(f"{DATA_ROOT}/1_cellpose/cell/cell_dp.tif")
    prob_stack = tifffile.imread(f"{DATA_ROOT}/1_cellpose/cell/cell_prob.tif")

    print(f"  nuclear_labels_corrected: {nuc_stack.shape} {nuc_stack.dtype}")
    print(f"  cell_dp:  {dp_stack.shape}  range [{dp_stack.min():.2f}, {dp_stack.max():.2f}]")
    print(f"  cell_prob: {prob_stack.shape} range [{prob_stack.min():.2f}, {prob_stack.max():.2f}]")

    nuc  = nuc_stack[frame].astype(np.int32)
    dp   = dp_stack[frame]   # (2, H, W)
    prob = prob_stack[frame]  # (H, W) raw logits

    flow = np.transpose(dp, (1, 2, 0)).astype(np.float32)
    prob_sigmoid = (1.0 / (1.0 + np.exp(-prob))).astype(np.float32)

    return nuc, flow, prob_sigmoid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    nuc, flow, prob_sigmoid = load_frame(FRAME)

    print(f"\nRunning OLD stochastic watershed (frame {FRAME})…", flush=True)
    t0 = time.perf_counter()
    result_stochastic = _stochastic_watershed(nuc, flow, prob_sigmoid,
                                              flow_scale=FLOW_SCALE_STOCH,
                                              uniform_growth_rate=UNIFORM_GROWTH_RATE,
                                              max_iterations=MAX_ITERATIONS_STOCH)
    print(f"  done in {time.perf_counter() - t0:.1f}s  |  cells: {np.unique(result_stochastic[result_stochastic > 0]).size}")

    print(f"\nRunning NEW Euler flow integration (frame {FRAME})…", flush=True)
    t0 = time.perf_counter()
    result_euler = _euler_flow_segmentation(nuc, flow, prob_sigmoid)
    print(f"  done in {time.perf_counter() - t0:.1f}s  |  cells: {np.unique(result_euler[result_euler > 0]).size}")

    # ── Field visualisations ──────────────────────────────────────────────────
    fy, fx = flow[..., 0], flow[..., 1]
    gy, gx = compute_gravity(nuc)

    b25y,  b25x  = blend_fields(fy, fx, gy, gx, 0.25)
    b50y,  b50x  = blend_fields(fy, fx, gy, gx, 0.50)
    b75y,  b75x  = blend_fields(fy, fx, gy, gx, 0.75)

    print("\nLaunching napari…")
    import napari

    viewer = napari.Viewer(title=f"Flow segmentation comparison — pos02 frame {FRAME}")

    # Segmentation results
    viewer.add_labels(nuc,               name="nuclear seeds",               opacity=0.7)
    viewer.add_labels(result_stochastic, name="OLD stochastic iterative",    opacity=0.5, visible=False)
    viewer.add_labels(result_euler,      name="NEW Euler integration",       opacity=0.5)

    # Cellpose flow
    viewer.add_image(field_mag(fy, fx),    name="cellpose flow — magnitude", colormap="magma",  opacity=0.8, visible=True)
    viewer.add_image(field_angle(fy, fx),  name="cellpose flow — direction", colormap="hsv",    opacity=0.7, visible=False)

    # Gravity field
    viewer.add_image(field_angle(gy, gx),  name="gravity — direction",       colormap="hsv",    opacity=0.7, visible=False)

    # Blended fields (magnitude + direction)
    for alpha, by, bx in [(0.25, b25y, b25x), (0.50, b50y, b50x), (0.75, b75y, b75x)]:
        label = f"blend {int(alpha*100)}% flow"
        viewer.add_image(field_mag(by, bx),   name=f"{label} — magnitude",  colormap="magma",  opacity=0.8, visible=False)
        viewer.add_image(field_angle(by, bx),  name=f"{label} — direction",  colormap="hsv",    opacity=0.7, visible=False)

    # Probability mask
    viewer.add_image(prob_sigmoid, name="cell prob (sigmoid)", colormap="gray", opacity=0.4, visible=False)

    napari.run()


if __name__ == "__main__":
    main()
