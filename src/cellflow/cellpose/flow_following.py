"""Flow-following cell segmentation, Qt-free, for the standalone distro.

When the standalone tool has **both** a nucleus and a cell channel, it can do
better than independent native masks: it assigns each cell-foreground pixel to a
*nucleus*, giving one cell per nucleus with the cell sharing the nucleus' id.
The cell's *extent* is decided by Cellpose's own flow field rather than by naive
distance.

This is the proven two-phase algorithm (recovered from the pre-publication
``segmentation/flow_following.py``):

1. **Integrate.** Each foreground pixel is Euler-integrated along a blend of the
   Cellpose flow vector and an EDT *gravity* vector pointing toward the nearest
   nucleus (``flow_weight`` mixes the two). A pixel that lands on a nucleus pixel
   during integration inherits that nucleus' label immediately.
2. **Grow.** Remaining foreground pixels are assigned by growing the labelled
   region outward in shells through the *displaced* (post-integration) positions,
   so labels chain-propagate along the flow topology. Pixels whose displaced
   position has no labelled neighbour within ``max_assign_radius`` stay
   background (orphans are dropped, not force-assigned).

The numba kernel is lazily JIT-compiled when numba is importable (it ships with
the Cellpose stack) and falls back to the identical pure-Python function
otherwise, so importing this module never requires numba.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

__all__ = [
    "FlowFollowingParams",
    "flow_follow_frame",
    "flow_follow_movie",
]


@dataclass(frozen=True)
class FlowFollowingParams:
    """Parameters for flow-following cell assignment."""

    fg_threshold: float = 0.5
    """Cell-foreground cutoff on the sigmoid probability (``sigmoid(prob) >
    fg_threshold``). Exposed knob — pixels below it are not assigned."""

    flow_weight: float = 0.5
    """Blend between the Cellpose flow vector (``1``) and the EDT gravity vector
    toward the nearest nucleus (``0``). ``0.5`` mixes both equally."""

    flow_step_scale: float = 0.2
    """Euler step size (multiplies the unit blended vector each iteration)."""

    max_iterations: int = 100
    """Maximum integration steps per pixel."""

    shell_width: float = 5.0
    """Per-iteration capture distance when growing labels in shells (px)."""

    max_assign_radius: float = 30.0
    """A displaced foreground pixel with no labelled pixel within this radius
    stays background — orphan foreground is dropped, not force-assigned."""


# ── Phase 1: Euler integration (lazy-JIT kernel + pure-Python twin) ──────────

def _integrate_positions_py(
    nuclear_labels,   # (H, W) int32
    flow,             # (2, H, W) float32 — flow[0]=dy, flow[1]=dx
    grav_y,           # (H, W) float32
    grav_x,           # (H, W) float32
    prob_mask,        # (H, W) bool
    n_steps,          # int
    flow_step_scale,  # float
    flow_weight,      # float
):
    """Integrate each foreground pixel along ``flow_weight*flow + (1-w)*gravity``.

    Returns ``(result, final_y, final_x)``: a label map where nucleus pixels keep
    their label and foreground pixels that land directly on a nucleus inherit it,
    plus the displaced position of every pixel after integration.
    """
    H = nuclear_labels.shape[0]
    W = nuclear_labels.shape[1]
    result = nuclear_labels.copy()
    final_y = np.empty((H, W), dtype=np.float32)
    final_x = np.empty((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            final_y[i, j] = np.float32(i)
            final_x[i, j] = np.float32(j)

            if nuclear_labels[i, j] > 0 or not prob_mask[i, j]:
                continue

            py = np.float32(i)
            px = np.float32(j)
            for _ in range(n_steps):
                iy = min(max(int(py), 0), H - 1)
                ix = min(max(int(px), 0), W - 1)
                fy = (
                    flow_weight * flow[0, iy, ix]
                    + (np.float32(1.0) - flow_weight) * grav_y[iy, ix]
                )
                fx = (
                    flow_weight * flow[1, iy, ix]
                    + (np.float32(1.0) - flow_weight) * grav_x[iy, ix]
                )
                py = py + fy * flow_step_scale
                px = px + fx * flow_step_scale
                py = min(max(py, np.float32(0.0)), np.float32(H - 1))
                px = min(max(px, np.float32(0.0)), np.float32(W - 1))
                if nuclear_labels[int(py), int(px)] > 0:
                    result[i, j] = nuclear_labels[int(py), int(px)]
                    break

            final_y[i, j] = py
            final_x[i, j] = px

    return result, final_y, final_x


def _integrate_positions(*args):
    """Dispatch to the numba-compiled kernel when available, else pure Python."""
    global _INTEGRATE
    if _INTEGRATE is None:
        try:  # numba ships with the cellpose stack; optional otherwise.
            import numba

            _INTEGRATE = numba.njit(cache=True)(_integrate_positions_py)
        except Exception:  # pragma: no cover - exercised only without numba
            _INTEGRATE = _integrate_positions_py
    return _INTEGRATE(*args)


_INTEGRATE = None


# ── Phase 2: progressive shell growth (orphan-bounded) ───────────────────────

def _progressive_shell_assign(
    labels: np.ndarray,
    final_y: np.ndarray,
    final_x: np.ndarray,
    foreground: np.ndarray,
    *,
    shell_width: float,
    max_assign_radius: float,
) -> np.ndarray:
    """Grow labels outward through displaced positions; drop distant orphans.

    Each iteration computes the EDT from currently-labelled pixels and assigns any
    unassigned foreground pixel whose *displaced* position is within
    ``shell_width`` of a labelled pixel. Newly labelled pixels seed the next EDT,
    so labels chain-propagate along the flow topology. The growth stops once no
    displaced position is within ``max_assign_radius`` of a label — remaining
    pixels stay background.
    """
    H, W = labels.shape
    result = labels.copy()
    unassigned = foreground & (result == 0)

    fy = np.clip(np.round(final_y).astype(np.intp), 0, H - 1)
    fx = np.clip(np.round(final_x).astype(np.intp), 0, W - 1)

    while unassigned.any():
        unlabelled = result == 0
        if not unlabelled.any():
            break
        dist, indices = distance_transform_edt(unlabelled, return_indices=True)
        d = dist[fy, fx]
        nearest = result[indices[0][fy, fx], indices[1][fy, fx]]
        can_assign = unassigned & (d <= shell_width) & (nearest > 0)
        if not can_assign.any():
            break
        result[can_assign] = nearest[can_assign]
        unassigned &= ~can_assign

    # Bounded fallback (orphan drop): assign any remaining pixel only if a label
    # sits within max_assign_radius of its displaced position; else leave it 0.
    if unassigned.any() and (result > 0).any():
        unlabelled = result == 0
        if unlabelled.any():
            dist, indices = distance_transform_edt(unlabelled, return_indices=True)
            d = dist[fy, fx]
            nearest = result[indices[0][fy, fx], indices[1][fy, fx]]
            close = unassigned & (d <= max_assign_radius) & (nearest > 0)
            result[close] = nearest[close]

    return result


# ── Gravity field ────────────────────────────────────────────────────────────

def _gravity_toward_nuclei(labels_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors at every pixel pointing toward the nearest nucleus pixel."""
    nucleus = labels_yx > 0
    _dist, indices = distance_transform_edt(~nucleus, return_indices=True)
    H, W = labels_yx.shape
    yy, xx = np.mgrid[:H, :W]
    dy = (indices[0] - yy).astype(np.float32)
    dx = (indices[1] - xx).astype(np.float32)
    norm = np.hypot(dy, dx).astype(np.float32)
    norm[norm == 0.0] = 1.0
    return (dy / norm).astype(np.float32), (dx / norm).astype(np.float32)


# ── Public: single frame + movie ─────────────────────────────────────────────

def flow_follow_frame(
    foreground_yx: np.ndarray,
    dp_cyx: np.ndarray,
    labels_yx: np.ndarray,
    params: FlowFollowingParams,
) -> np.ndarray:
    """Assign one frame's cell foreground to nuclei via flow-following.

    ``foreground_yx`` is a bool mask, ``dp_cyx`` the Cellpose cell flow
    ``(2, Y, X)`` (``dy, dx``), ``labels_yx`` the tracked nucleus labels. Returns
    ``(Y, X)`` int32 cell labels carrying the nucleus ids (background ``0``).
    """
    foreground_yx = np.ascontiguousarray(foreground_yx, dtype=np.bool_)
    labels_yx = np.ascontiguousarray(labels_yx, dtype=np.int32)
    if not foreground_yx.any() or labels_yx.max() == 0:
        return np.zeros(foreground_yx.shape, dtype=np.int32)

    # Normalise flow by mean foreground magnitude so the step scale is comparable
    # across frames/datasets.
    flow = np.ascontiguousarray(dp_cyx, dtype=np.float32).copy()
    mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
    mean_mag = float(mag[foreground_yx].mean()) if foreground_yx.any() else 1.0
    if mean_mag > 0:
        flow /= np.float32(mean_mag)

    grav_y, grav_x = _gravity_toward_nuclei(labels_yx)
    result, final_y, final_x = _integrate_positions(
        labels_yx,
        flow,
        np.ascontiguousarray(grav_y),
        np.ascontiguousarray(grav_x),
        foreground_yx,
        int(params.max_iterations),
        np.float32(params.flow_step_scale),
        np.float32(params.flow_weight),
    )
    return _progressive_shell_assign(
        result, final_y, final_x, foreground_yx,
        shell_width=float(params.shell_width),
        max_assign_radius=float(params.max_assign_radius),
    ).astype(np.int32)


def flow_follow_movie(
    foreground_tyx: np.ndarray,
    dp_tcyx: np.ndarray,
    labels_tyx: np.ndarray,
    params: FlowFollowingParams,
    *,
    progress_cb=None,
) -> np.ndarray:
    """Run :func:`flow_follow_frame` over a ``(T, Y, X)`` series.

    ``dp_tcyx`` is ``(T, 2, Y, X)``. Cell labels inherit the (already tracked)
    nucleus ids per frame, so tracking the nuclei tracks the cells. Returns
    ``(T, Y, X)`` int32.
    """
    foreground = np.asarray(foreground_tyx, dtype=bool)
    labels = np.asarray(labels_tyx, dtype=np.int32)
    flow = np.asarray(dp_tcyx, dtype=np.float32)
    if foreground.ndim != 3 or labels.shape != foreground.shape:
        raise ValueError("foreground and labels must share shape (T, Y, X)")
    T, Y, X = foreground.shape
    if flow.shape != (T, 2, Y, X):
        raise ValueError(f"dp_tcyx must be (T, 2, Y, X); got {flow.shape}")

    out = np.zeros((T, Y, X), dtype=np.int32)
    for t in range(T):
        out[t] = flow_follow_frame(foreground[t], flow[t], labels[t], params)
        if progress_cb is not None:
            progress_cb(t + 1, T, f"Flow-following: frame {t + 1}/{T}...")
    return out
