"""Run a 2D+T multi-label α-expansion graph-cut experiment for cell labels.

Nodes: every voxel in the full (T, Y, X) grid (background pinned to sink).
Labels: nucleus track IDs, global set; dead labels get INF unary cost.
Unary: geodesic distance from nucleus k in frame t (cost field = 1 + alpha_unary*contour);
       normalized per frame by median reachable distance.
Spatial pairwise: truncated Potts — lambda_s * exp(-beta_s * avg_contour).
Temporal pairwise: constant Potts — lambda_t.
Solver: PyMaxflow (maxflow) alpha-expansion, binary graph-cut per label per round.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import math
import multiprocessing as mp

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import maxflow
import numpy as np
import numba
import tifffile
from scipy.ndimage import distance_transform_edt, label as ndi_label
from skimage.graph import MCP_Geometric

DEFAULT_POS_DIR = Path(
    "/home/aruppel/Data/"
    "2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/"
    "analysis/pos00"
)
_INF = 1e9
_COST_SCALE = 100        # multiply float costs → int32
_INT_INF = 1_000_000_000  # terminal capacity for Graph[int] anchors (fits in int32)
_NUC_MAX_PX = 256        # max nucleus pixels sampled per (label, frame) for Numba unary

# 2-D add_grid_edges structures (center at [1,1] in 3x3)
_H_STRUCT = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)   # right
_V_STRUCT = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)   # down
_DR_STRUCT = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=bool)  # down-right diagonal
_DL_STRUCT = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=bool)  # down-left diagonal
_DIAG_SCALE = float(1.0 / np.sqrt(2.0))
# For temporal: reshape node grid as (T, Y*X) and connect row+1 → same pixel next frame
_T_STRUCT = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=bool)   # next row, same col
_CC_STRUCT = np.ones((3, 3), dtype=bool)


# ── Parallel worker globals ─────────────────────────────────────────────────
# Set in the parent process before Pool.map; inherited by forked workers (Linux
# COW — no per-task pickling of the large arrays).
_W_FG: np.ndarray | None = None
_W_FG_NODE_ID: np.ndarray | None = None  # (T,Y,X) int32; bg pixels → _W_N_FG (dummy)
_W_N_FG: int = 0                          # number of foreground nodes
_W_CC: np.ndarray | None = None   # int32 current_cost snapshot (full T,Y,X)
_W_CL: np.ndarray | None = None   # uint32 current_labels snapshot
_W_UNARY_INT: dict | None = None  # quantized unary dict
_W_SHAPE: tuple | None = None
_W_N_EDGES: int = 0
# Precomputed foreground-only edge lists (src_flat, dst_flat, weight_int32) per direction.
# Spatial: list of T tuples, one per frame. Temporal: single tuple spanning all T-1 frame pairs.
_W_FG_EDGES_H: list | None = None
_W_FG_EDGES_V: list | None = None
_W_FG_EDGES_DR: list | None = None
_W_FG_EDGES_DL: list | None = None
_W_FG_EDGES_TW: tuple | None = None


def _quantize(arr: np.ndarray) -> np.ndarray:
    """Scale float cost array to int32, capping float INF at _INT_INF."""
    return np.minimum(arr * _COST_SCALE, _INT_INF).round().astype(np.int32)


def _candidate_mask(current_labels: np.ndarray, fg_mask: np.ndarray, alpha: int) -> np.ndarray:
    """Return fg pixels that are currently alpha OR 8-connected to alpha within the same frame.

    Restricting to within-frame connectivity prevents the expansion from claiming pixels
    that are spatially isolated from the current alpha region in their own frame just
    because the same (y, x) is alpha at the previous/next frame.  Temporal coherence is
    still enforced through the temporal Potts pairwise edges between candidate pixels
    and their frozen non-candidate temporal neighbors.
    """
    seed = (current_labels == alpha) & fg_mask
    cand = seed.copy()
    # 8-connected spatial dilation (within each frame only).
    cand[:, :, 1:]    |= seed[:, :, :-1]
    cand[:, :, :-1]   |= seed[:, :, 1:]
    cand[:, 1:, :]    |= seed[:, :-1, :]
    cand[:, :-1, :]   |= seed[:, 1:, :]
    cand[:, 1:, 1:]   |= seed[:, :-1, :-1]
    cand[:, :-1, :-1] |= seed[:, 1:, 1:]
    cand[:, 1:, :-1]  |= seed[:, :-1, 1:]
    cand[:, :-1, 1:]  |= seed[:, 1:, :-1]
    return cand & fg_mask


def _component_count(mask: np.ndarray) -> int:
    """Return 8-connected component count for a single 2D label mask."""
    if not mask.any():
        return 0
    _, n = ndi_label(mask, structure=_CC_STRUCT)
    return int(n)


def _filter_connectivity_preserving_flips(
    current_labels: np.ndarray,
    flip_mask: np.ndarray,
    alpha: int,
) -> np.ndarray:
    """Reject flips that would split the source label in any frame.

    The candidate mask guarantees that newly claimed pixels touch the alpha
    region, but it says nothing about the label losing those pixels.  Removing
    an articulation pixel from the source label can create fragmented cells.
    This filter is intentionally conservative and processes proposed removals
    one pixel at a time, accepting a flip only when the source label's
    8-connected component count does not increase.
    """
    filtered = flip_mask.copy()
    T = current_labels.shape[0]
    for t in range(T):
        changed_t = filtered[t] & (current_labels[t] != alpha) & (current_labels[t] > 0)
        if not changed_t.any():
            continue
        for source_label in np.unique(current_labels[t][changed_t]):
            source_label = int(source_label)
            source_mask = current_labels[t] == source_label
            source_changes = np.argwhere(changed_t & source_mask)
            if source_changes.size == 0:
                continue
            n_components = _component_count(source_mask)
            for y, x in source_changes:
                trial = source_mask.copy()
                trial[int(y), int(x)] = False
                trial_components = _component_count(trial)
                if trial_components <= n_components:
                    source_mask = trial
                    n_components = trial_components
                else:
                    filtered[t, int(y), int(x)] = False
    return filtered


def _precompute_fg_edge_lists(
    fg_mask: np.ndarray,
    h_int: np.ndarray,
    v_int: np.ndarray,
    dr_int: np.ndarray,
    dl_int: np.ndarray,
    tw_int: np.ndarray,
) -> tuple[list, list, list, list, tuple]:
    """Pre-compute foreground-only edge index lists for use with Graph.add_edges.

    Returns (e_h, e_v, e_dr, e_dl, e_tw) where:
    - e_h[t] = (src_flat, dst_flat, w_int32): horizontal fg-fg pairs in frame t
    - e_v, e_dr, e_dl: same for vertical / down-right / down-left
    - e_tw = (src_global, dst_global, w_int32): temporal fg-fg pairs, all frames concatenated

    src/dst_flat are flat (Y*X) indices within a single frame.
    src/dst_global are flat (T*Y*X) indices across all frames.
    Weights are int32 (already quantized).
    """
    T, Y, X = fg_mask.shape
    e_h, e_v, e_dr, e_dl = [], [], [], []

    for t in range(T):
        fg_t = fg_mask[t]

        # Horizontal: (y,x) → (y,x+1)
        valid = fg_t[:, :-1] & fg_t[:, 1:]
        yx = np.argwhere(valid)
        src = (yx[:, 0] * X + yx[:, 1]).astype(np.int32)
        dst = src + np.int32(1)
        e_h.append((src, dst, h_int[t].ravel()[src]))

        # Vertical: (y,x) → (y+1,x)
        valid = fg_t[:-1, :] & fg_t[1:, :]
        yx = np.argwhere(valid)
        src = (yx[:, 0] * X + yx[:, 1]).astype(np.int32)
        dst = src + np.int32(X)
        e_v.append((src, dst, v_int[t].ravel()[src]))

        # Down-right: (y,x) → (y+1,x+1)
        valid = fg_t[:-1, :-1] & fg_t[1:, 1:]
        yx = np.argwhere(valid)
        src = (yx[:, 0] * X + yx[:, 1]).astype(np.int32)
        dst = src + np.int32(X + 1)
        e_dr.append((src, dst, dr_int[t].ravel()[src]))

        # Down-left: (y,x) → (y+1,x-1); source at column ≥ 1
        valid = fg_t[:-1, 1:] & fg_t[1:, :-1]
        yx = np.argwhere(valid)
        src = (yx[:, 0] * X + yx[:, 1] + 1).astype(np.int32)  # x = subgrid_col + 1
        dst = src + np.int32(X - 1)
        e_dl.append((src, dst, dl_int[t].ravel()[src]))

    # Temporal: (t,y,x) → (t+1,y,x) for all fg-fg pairs across frame boundaries
    tw_src_all, tw_dst_all, tw_w_all = [], [], []
    for t in range(T - 1):
        shared = fg_mask[t] & fg_mask[t + 1]
        yx = np.argwhere(shared)
        yx_flat = (yx[:, 0] * X + yx[:, 1]).astype(np.int32)
        src_g = (t * Y * X + yx_flat.astype(np.int64)).astype(np.int32)
        dst_g = ((t + 1) * Y * X + yx_flat.astype(np.int64)).astype(np.int32)
        tw_src_all.append(src_g)
        tw_dst_all.append(dst_g)
        tw_w_all.append(tw_int[t].ravel()[yx_flat])

    if tw_src_all:
        e_tw = (
            np.concatenate(tw_src_all),
            np.concatenate(tw_dst_all),
            np.concatenate(tw_w_all),
        )
    else:
        empty = np.array([], dtype=np.int32)
        e_tw = (empty, empty, empty)

    return e_h, e_v, e_dr, e_dl, e_tw


def _add_fg_edges(
    g: maxflow.Graph,
    cand_node_id_flat: np.ndarray,
    edge_list: tuple,
    n_cand: int,
    frame_offset: int = 0,
) -> None:
    """Add precomputed fg edge pairs to graph g, skipping frozen-frozen pairs."""
    src_flat, dst_flat, w = edge_list
    src_nodes = cand_node_id_flat[frame_offset + src_flat]
    dst_nodes = cand_node_id_flat[frame_offset + dst_flat]
    keep = (src_nodes != n_cand) | (dst_nodes != n_cand)
    if keep.any():
        g.add_edges(src_nodes[keep], dst_nodes[keep], w[keep], w[keep])


def _alpha_cut(alpha: int) -> tuple[int, np.ndarray, int]:
    """Worker: binary Graph[int] cut for one α label (reads module-level globals).

    Only candidate pixels (currently α or fg-adjacent to α) enter the graph.
    Non-candidate pixels map to a dummy SINK node; edges from candidate pixels to
    the dummy correctly charge pairwise costs against frozen non-α neighbors.
    """
    T, Y, X = _W_SHAPE

    cand = _candidate_mask(_W_CL, _W_FG, alpha)
    n_cand = int(np.count_nonzero(cand))
    if n_cand == 0:
        return alpha, np.zeros((T, Y, X), dtype=bool), 0

    # cand pixels → 0..n_cand-1; everything else (bg + frozen fg) → n_cand (dummy SINK).
    cand_node_id = np.full((T, Y, X), n_cand, dtype=np.int32)
    cand_node_id[cand] = np.arange(n_cand, dtype=np.int32)
    cand_flat = np.arange(n_cand, dtype=np.int32)
    cand_node_id_flat = cand_node_id.ravel()  # (T*Y*X,) for temporal indexing

    alpha_cost = np.full(n_cand, _INT_INF, dtype=np.int32)
    for t in range(T):
        if (t, alpha) in _W_UNARY_INT:
            m = cand[t]
            alpha_cost[cand_node_id[t][m]] = _W_UNARY_INT[(t, alpha)][m]

    # SOURCE anchor: currently-α candidate pixels forced to SOURCE via INT_INF source cap.
    cc_cand = _W_CC[cand].copy()
    cc_cand[(_W_CL[cand] == alpha)] = _INT_INF

    n_edges_est = n_cand * 10
    g = maxflow.Graph[int](n_cand + 1, n_edges_est)
    g.add_nodes(n_cand + 1)
    g.add_tedge(n_cand, 0, _INT_INF)  # dummy always in SINK
    g.add_grid_tedges(cand_flat, cc_cand, alpha_cost)

    # Spatial edges: foreground-only precomputed pairs, skip frozen-frozen.
    YX = Y * X
    for t in range(T):
        offset = t * YX
        _add_fg_edges(g, cand_node_id_flat, _W_FG_EDGES_H[t], n_cand, offset)
        _add_fg_edges(g, cand_node_id_flat, _W_FG_EDGES_V[t], n_cand, offset)
        _add_fg_edges(g, cand_node_id_flat, _W_FG_EDGES_DR[t], n_cand, offset)
        _add_fg_edges(g, cand_node_id_flat, _W_FG_EDGES_DL[t], n_cand, offset)

    # Temporal edges: stored as (T*Y*X) global flat indices.
    src_gl, dst_gl, tw_w = _W_FG_EDGES_TW
    src_nodes = cand_node_id_flat[src_gl]
    dst_nodes = cand_node_id_flat[dst_gl]
    keep = (src_nodes != n_cand) | (dst_nodes != n_cand)
    if keep.any():
        g.add_edges(src_nodes[keep], dst_nodes[keep], tw_w[keep], tw_w[keep])

    g.maxflow()

    in_sink_cand = g.get_grid_segments(cand_flat)
    in_sink = np.ones((T, Y, X), dtype=bool)
    in_sink[cand] = in_sink_cand
    flip_mask = (~in_sink) & _W_FG
    n_changed = int(np.count_nonzero(_W_CL[flip_mask] != alpha))
    return alpha, flip_mask, n_changed


# ── Numba ICM kernels ───────────────────────────────────────────────────────

def _build_nucleus_pixels(
    nuc_tracks: np.ndarray,
    label_ids: np.ndarray,
    max_px: int = _NUC_MAX_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect nucleus pixel coords per (label, frame), capped at max_px.

    Returns:
        nuc_px:  (K, T, max_px, 2) int16 — y, x coords (zero-padded)
        nuc_cnt: (K, T) int32         — actual pixel count per (label, frame)
    """
    K = len(label_ids)
    T = nuc_tracks.shape[0]
    nuc_px = np.zeros((K, T, max_px, 2), dtype=np.int16)
    nuc_cnt = np.zeros((K, T), dtype=np.int32)
    rng = np.random.default_rng(0)
    for ki, k in enumerate(label_ids):
        for t in range(T):
            yx = np.argwhere(nuc_tracks[t] == k)
            n = len(yx)
            if n == 0:
                continue
            if n > max_px:
                yx = yx[rng.choice(n, max_px, replace=False)]
                n = max_px
            nuc_px[ki, t, :n] = yx.astype(np.int16)
            nuc_cnt[ki, t] = n
    return nuc_px, nuc_cnt


@numba.njit(parallel=True, cache=True)
def _nb_integrate_flow(
    flow_field: np.ndarray,  # (T, 2, Y, X) float32
    n_steps: int,
    step_scale: float,
) -> np.ndarray:
    """Walk n_steps along the flow field for every pixel.

    Returns (T, Y, X, 2) float32 endpoint coordinates.
    """
    T, _, Y, X = flow_field.shape
    endpoints = np.empty((T, Y, X, 2), dtype=np.float32)
    for t in numba.prange(T):
        dy_f = flow_field[t, 0]
        dx_f = flow_field[t, 1]
        for y in range(Y):
            for x in range(X):
                py = np.float32(y)
                px = np.float32(x)
                for _ in range(n_steps):
                    py_c = min(max(py, np.float32(0.0)), np.float32(Y - 1.0001))
                    px_c = min(max(px, np.float32(0.0)), np.float32(X - 1.0001))
                    iy0 = int(py_c)
                    ix0 = int(px_c)
                    fy = py_c - np.float32(iy0)
                    fx = px_c - np.float32(ix0)
                    iy1 = min(iy0 + 1, Y - 1)
                    ix1 = min(ix0 + 1, X - 1)
                    w00 = (np.float32(1.0) - fy) * (np.float32(1.0) - fx)
                    w10 = fy * (np.float32(1.0) - fx)
                    w01 = (np.float32(1.0) - fy) * fx
                    w11 = fy * fx
                    dy = (dy_f[iy0, ix0] * w00 + dy_f[iy1, ix0] * w10
                          + dy_f[iy0, ix1] * w01 + dy_f[iy1, ix1] * w11)
                    dx = (dx_f[iy0, ix0] * w00 + dx_f[iy1, ix0] * w10
                          + dx_f[iy0, ix1] * w01 + dx_f[iy1, ix1] * w11)
                    py = min(max(py + np.float32(step_scale) * dy,
                                 np.float32(0.0)), np.float32(Y - 1))
                    px = min(max(px + np.float32(step_scale) * dx,
                                 np.float32(0.0)), np.float32(X - 1))
                endpoints[t, y, x, 0] = py
                endpoints[t, y, x, 1] = px
    return endpoints


@numba.njit(parallel=True, cache=True)
def _nb_flow_unary_raw(
    endpoints: np.ndarray,  # (T, Y, X, 2) float32
    nuc_px: np.ndarray,     # (K, T, max_px, 2) int16
    nuc_cnt: np.ndarray,    # (K, T) int32
    fg_mask: np.ndarray,    # (T, Y, X) bool
    cap_px: float,
) -> np.ndarray:
    """Min distance from each pixel's flow endpoint to each label's nucleus pixels.

    Returns (T, Y, X, K) float32.  Dead labels and background → cap_px + 1 (sentinel).
    """
    T, Y, X = fg_mask.shape
    K = nuc_px.shape[0]
    sentinel = np.float32(cap_px + 1.0)
    raw = np.full((T, Y, X, K), sentinel, dtype=np.float32)
    for t in numba.prange(T):
        for y in range(Y):
            for x in range(X):
                if not fg_mask[t, y, x]:
                    continue
                eq_y = endpoints[t, y, x, 0]
                eq_x = endpoints[t, y, x, 1]
                for ki in range(K):
                    n = nuc_cnt[ki, t]
                    if n == 0:
                        continue
                    min_d = np.float32(1e18)
                    for pi in range(n):
                        dy = eq_y - np.float32(nuc_px[ki, t, pi, 0])
                        dx = eq_x - np.float32(nuc_px[ki, t, pi, 1])
                        d = np.float32(math.sqrt(dy * dy + dx * dx))
                        if d < min_d:
                            min_d = d
                    raw[t, y, x, ki] = min(min_d, np.float32(cap_px))
    return raw


def _normalize_flow_unary(
    raw: np.ndarray,       # (T, Y, X, K) float32 from _nb_flow_unary_raw
    fg_mask: np.ndarray,   # (T, Y, X) bool
    nuc_cnt: np.ndarray,   # (K, T) int32
    cap_px: float = 100.0,
) -> np.ndarray:
    """Per-frame median normalization. Dead / unreachable entries → _INF."""
    T, Y, X, K = raw.shape
    sentinel = cap_px + 0.5  # values above this are dead/background
    result = np.full((T, Y, X, K), _INF, dtype=np.float32)
    for t in range(T):
        alive = nuc_cnt[:, t] > 0  # (K,) bool
        if not alive.any():
            continue
        fg_t = fg_mask[t]
        raw_t = raw[t]  # (Y, X, K)
        # Collect reachable distances for alive labels in foreground
        reachable_mask = fg_t[:, :, None] & alive[None, None, :] & (raw_t < sentinel)
        vals = raw_t[reachable_mask]
        med = float(np.median(vals)) if vals.size > 0 else 1.0
        if med <= 0.0:
            med = 1.0
        for ki in range(K):
            if not alive[ki]:
                continue
            r = raw_t[:, :, ki]
            ok = fg_t & (r < sentinel)
            result[t, :, :, ki][ok] = r[ok] / med
    return result


def _dict_to_dense_unary(
    unary: dict[tuple[int, int], np.ndarray],
    fg_mask: np.ndarray,
    label_ids: np.ndarray,
) -> np.ndarray:
    """Convert sparse unary dict to dense (T, Y, X, K) float32 array for ICM."""
    T, Y, X = fg_mask.shape
    K = len(label_ids)
    dense = np.full((T, Y, X, K), _INF, dtype=np.float32)
    for ki, k in enumerate(label_ids):
        for t in range(T):
            u = unary.get((t, int(k)))
            if u is not None:
                dense[t, :, :, ki] = u
    return dense


@numba.njit(cache=True)
def _nb_icm_round(
    labels: np.ndarray,           # (T, Y, X) uint32 — in-place read + write
    unary_dense: np.ndarray,      # (T, Y, X, K) float32
    h_w: np.ndarray,              # (T, Y, X) float32
    v_w: np.ndarray,              # (T, Y, X) float32
    dr_w: np.ndarray,             # (T, Y, X) float32 — down-right diagonal
    dl_w: np.ndarray,             # (T, Y, X) float32 — down-left diagonal
    tw_w: np.ndarray,             # (T, Y, X) float32 — pure temporal
    tw_ty_dn_w: np.ndarray,       # (T, Y, X) float32 — (t,y,x)↔(t+1,y+1,x)
    tw_ty_up_w: np.ndarray,       # (T, Y, X) float32 — (t,y,x)↔(t+1,y-1,x)
    tw_tx_r_w: np.ndarray,        # (T, Y, X) float32 — (t,y,x)↔(t+1,y,x+1)
    tw_tx_l_w: np.ndarray,        # (T, Y, X) float32 — (t,y,x)↔(t+1,y,x-1)
    fg_mask: np.ndarray,          # (T, Y, X) bool
    label_ids: np.ndarray,        # (K,) uint32
    anchor_label: np.ndarray,     # (T, Y, X) uint32 — 0=free, k=pinned to k
    areas: np.ndarray,            # (K, T) int32 — per-label pixel counts, updated in-place
    lambda_area: np.float32,      # weight for frame-to-frame area change penalty
) -> int:
    """One sequential (Gauss-Seidel) ICM sweep — raster scan, in-place.

    Each pixel immediately sees its neighbors' latest decisions, so changes
    propagate locally rather than as coordinated mass swaps.  Guaranteed to
    converge for non-negative Potts pairwise weights.
    Returns number of label changes.
    """
    T, Y, X = fg_mask.shape
    K = len(label_ids)
    n_flips = 0
    for t in range(T):
        for y in range(Y):
            for x in range(X):
                if anchor_label[t, y, x] > np.uint32(0):
                    continue
                if not fg_mask[t, y, x]:
                    continue
                old_label = labels[t, y, x]

                # Find index of old_label in label_ids for area delta computation.
                ji = np.int32(-1)
                for kk in range(K):
                    if label_ids[kk] == old_label:
                        ji = np.int32(kk)
                        break

                best_cost = np.float32(1e30)
                best_k = old_label
                best_ki = np.int32(-1)
                for ki in range(K):
                    u = unary_dense[t, y, x, ki]
                    if u >= np.float32(1e8):
                        continue
                    k = label_ids[ki]
                    # 8-connected lateral-adjacency gate (within frame): a
                    # pixel may flip to label k only if it currently is k or
                    # has an in-frame 8-neighbor already carrying k.
                    if k != old_label:
                        adj = False
                        if x + 1 < X and labels[t, y, x + 1] == k:
                            adj = True
                        elif x > 0 and labels[t, y, x - 1] == k:
                            adj = True
                        elif y + 1 < Y and labels[t, y + 1, x] == k:
                            adj = True
                        elif y > 0 and labels[t, y - 1, x] == k:
                            adj = True
                        elif y + 1 < Y and x + 1 < X and labels[t, y + 1, x + 1] == k:
                            adj = True
                        elif y > 0 and x > 0 and labels[t, y - 1, x - 1] == k:
                            adj = True
                        elif y + 1 < Y and x > 0 and labels[t, y + 1, x - 1] == k:
                            adj = True
                        elif y > 0 and x + 1 < X and labels[t, y - 1, x + 1] == k:
                            adj = True
                        if not adj:
                            continue
                    cost = u
                    # Axis-aligned
                    if x + 1 < X and fg_mask[t, y, x + 1] and labels[t, y, x + 1] != k:
                        cost += h_w[t, y, x]
                    if x > 0 and fg_mask[t, y, x - 1] and labels[t, y, x - 1] != k:
                        cost += h_w[t, y, x - 1]
                    if y + 1 < Y and fg_mask[t, y + 1, x] and labels[t, y + 1, x] != k:
                        cost += v_w[t, y, x]
                    if y > 0 and fg_mask[t, y - 1, x] and labels[t, y - 1, x] != k:
                        cost += v_w[t, y - 1, x]
                    # Diagonals: dr[t,y,x] is the (y,x)→(y+1,x+1) edge weight
                    if y + 1 < Y and x + 1 < X and fg_mask[t, y + 1, x + 1] and labels[t, y + 1, x + 1] != k:
                        cost += dr_w[t, y, x]
                    if y > 0 and x > 0 and fg_mask[t, y - 1, x - 1] and labels[t, y - 1, x - 1] != k:
                        cost += dr_w[t, y - 1, x - 1]
                    # dl[t,y,x] is the (y,x)→(y+1,x-1) edge weight
                    if y + 1 < Y and x > 0 and fg_mask[t, y + 1, x - 1] and labels[t, y + 1, x - 1] != k:
                        cost += dl_w[t, y, x]
                    if y > 0 and x + 1 < X and fg_mask[t, y - 1, x + 1] and labels[t, y - 1, x + 1] != k:
                        cost += dl_w[t, y - 1, x + 1]
                    # Temporal (same pixel, adjacent frame)
                    if t + 1 < T and fg_mask[t + 1, y, x] and labels[t + 1, y, x] != k:
                        cost += tw_w[t, y, x]
                    if t > 0 and fg_mask[t - 1, y, x] and labels[t - 1, y, x] != k:
                        cost += tw_w[t - 1, y, x]
                    # Spatiotemporal face-diagonals (forward: to t+1)
                    if t + 1 < T:
                        if y + 1 < Y and fg_mask[t + 1, y + 1, x] and labels[t + 1, y + 1, x] != k:
                            cost += tw_ty_dn_w[t, y, x]
                        if y > 0 and fg_mask[t + 1, y - 1, x] and labels[t + 1, y - 1, x] != k:
                            cost += tw_ty_up_w[t, y, x]
                        if x + 1 < X and fg_mask[t + 1, y, x + 1] and labels[t + 1, y, x + 1] != k:
                            cost += tw_tx_r_w[t, y, x]
                        if x > 0 and fg_mask[t + 1, y, x - 1] and labels[t + 1, y, x - 1] != k:
                            cost += tw_tx_l_w[t, y, x]
                    # Spatiotemporal face-diagonals (reverse: from t-1; weight stored at source)
                    if t > 0:
                        if y + 1 < Y and fg_mask[t - 1, y + 1, x] and labels[t - 1, y + 1, x] != k:
                            cost += tw_ty_up_w[t - 1, y + 1, x]
                        if y > 0 and fg_mask[t - 1, y - 1, x] and labels[t - 1, y - 1, x] != k:
                            cost += tw_ty_dn_w[t - 1, y - 1, x]
                        if x + 1 < X and fg_mask[t - 1, y, x + 1] and labels[t - 1, y, x + 1] != k:
                            cost += tw_tx_l_w[t - 1, y, x + 1]
                        if x > 0 and fg_mask[t - 1, y, x - 1] and labels[t - 1, y, x - 1] != k:
                            cost += tw_tx_r_w[t - 1, y, x - 1]
                    # Area change penalty: quadratic delta of sum_t (A_k(t) - A_k(t±1))^2.
                    # Gaining k at t: (ak+1-ref)^2 - (ak-ref)^2 = 2*(ak-ref) + 1
                    # Losing  j at t: (aj-1-ref)^2 - (aj-ref)^2 = 1 - 2*(aj-ref)
                    # Scales with actual deviation, so larger errors get stronger signal.
                    if lambda_area > np.float32(0.0) and ki != ji:
                        ak = areas[ki, t]
                        aj = areas[ji, t] if ji >= np.int32(0) else np.int32(0)
                        area_delta = np.float32(0.0)
                        if t > 0:
                            area_delta += np.float32(2 * (ak - areas[ki, t - 1]) + 1)
                            if ji >= np.int32(0):
                                area_delta += np.float32(1 - 2 * (aj - areas[ji, t - 1]))
                        if t + 1 < T:
                            area_delta += np.float32(2 * (ak - areas[ki, t + 1]) + 1)
                            if ji >= np.int32(0):
                                area_delta += np.float32(1 - 2 * (aj - areas[ji, t + 1]))
                        cost += lambda_area * area_delta
                    if cost < best_cost:
                        best_cost = cost
                        best_k = k
                        best_ki = np.int32(ki)
                labels[t, y, x] = best_k
                if best_k != old_label:
                    n_flips += 1
                    if best_ki >= np.int32(0):
                        areas[best_ki, t] += np.int32(1)
                    if ji >= np.int32(0):
                        areas[ji, t] -= np.int32(1)
    return n_flips


def _unary_elevation(
    unary: dict[tuple[int, int], np.ndarray],
    fg_mask: np.ndarray,
    label_ids: np.ndarray,
) -> np.ndarray:
    """Build (T, Y, X) elevation from any unary dict for seeded watershed.

    Each pixel's elevation is the minimum unary cost across all alive labels —
    low near any cell nucleus, higher at cell-cell boundaries.
    Works with flow, geodesic, or any other dict-format unary.
    """
    T, Y, X = fg_mask.shape
    elevation = np.full((T, Y, X), _INF, dtype=np.float32)
    for t in range(T):
        for k in label_ids:
            u = unary.get((t, int(k)))
            if u is None:
                continue
            finite = fg_mask[t] & (u < _INF)
            np.minimum(elevation[t], np.where(finite, u, _INF), out=elevation[t])
        # Replace remaining INF pixels inside fg with local max so watershed
        # can still flood through them (they sit at the highest cost).
        finite_fg = fg_mask[t] & (elevation[t] < _INF)
        if finite_fg.any():
            cap = float(np.max(elevation[t][finite_fg]))
            inf_fg = fg_mask[t] & (elevation[t] >= _INF)
            elevation[t][inf_fg] = cap
    return elevation


def _watershed_init(
    fg_mask: np.ndarray,
    nuc_tracks: np.ndarray,
    foreground_scores: np.ndarray | None = None,
    contours: np.ndarray | None = None,
    elevation: np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame seeded watershed from nucleus pixels into the foreground mask.

    Priority order for the elevation map: explicit ``elevation`` array, then
    ``1 - foreground_scores``, then ``contours``, then flat (all zeros).
    Returns (T, Y, X) uint32 label array.
    """
    from skimage.segmentation import watershed
    T, Y, X = fg_mask.shape
    labels = np.zeros((T, Y, X), dtype=np.uint32)
    for t in range(T):
        if elevation is not None:
            elev_t = elevation[t].astype(np.float32)
        elif foreground_scores is not None:
            elev_t = (1.0 - foreground_scores[t]).astype(np.float32)
        elif contours is not None:
            elev_t = contours[t].astype(np.float32)
        else:
            elev_t = np.zeros((Y, X), dtype=np.float32)
        markers = nuc_tracks[t].astype(np.int32)
        labels[t] = watershed(elev_t, markers=markers, mask=fg_mask[t]).astype(np.uint32)
        if (t + 1) % 10 == 0 or t + 1 == T:
            print(f"  watershed init: frame {t + 1}/{T}", flush=True)
    return labels


def _run_icm(
    fg_mask: np.ndarray,
    unary_dense: np.ndarray,   # (T, Y, X, K) float32
    label_ids: np.ndarray,
    h: np.ndarray,
    v: np.ndarray,
    dr: np.ndarray,
    dl: np.ndarray,
    tw: np.ndarray,
    tw_ty_dn: np.ndarray,
    tw_ty_up: np.ndarray,
    tw_tx_r: np.ndarray,
    tw_tx_l: np.ndarray,
    nuc_tracks: np.ndarray,
    n_iters: int = 10,
    min_round_flips: int = 0,
    init_labels: np.ndarray | None = None,
    lambda_area: float = 0.0,
) -> tuple[np.ndarray, list[dict]]:
    T, Y, X = fg_mask.shape

    if init_labels is not None:
        labels = init_labels.copy().astype(np.uint32)
    else:
        print("Initializing labels from unary argmin...", flush=True)
        best_ki = np.argmin(unary_dense, axis=3)
        labels = np.where(fg_mask, label_ids[best_ki], 0).astype(np.uint32)

    # Nucleus pixels are anchored — enforce at init regardless of source.
    nuc_mask = nuc_tracks > 0
    labels[nuc_mask] = nuc_tracks[nuc_mask].astype(np.uint32)

    anchor_label = nuc_tracks.astype(np.uint32)
    h32 = h.astype(np.float32)
    v32 = v.astype(np.float32)
    dr32 = dr.astype(np.float32)
    dl32 = dl.astype(np.float32)
    tw32 = tw.astype(np.float32)
    tw_ty_dn32 = tw_ty_dn.astype(np.float32)
    tw_ty_up32 = tw_ty_up.astype(np.float32)
    tw_tx_r32  = tw_tx_r.astype(np.float32)
    tw_tx_l32  = tw_tx_l.astype(np.float32)
    lids32 = label_ids.astype(np.uint32)
    lambda_area32 = np.float32(lambda_area)

    # Per-label pixel counts per frame — updated in-place by _nb_icm_round.
    K = len(label_ids)
    areas = np.zeros((K, T), dtype=np.int32)
    for ki, k in enumerate(label_ids):
        for t in range(T):
            areas[ki, t] = int(np.count_nonzero(labels[t] == k))

    energy_log: list[dict] = []
    for iteration in range(n_iters):
        print(f"\n=== ICM Round {iteration + 1}/{n_iters} ===", flush=True)
        t0 = perf_counter()
        n_flips = _nb_icm_round(
            labels, unary_dense, h32, v32, dr32, dl32, tw32,
            tw_ty_dn32, tw_ty_up32, tw_tx_r32, tw_tx_l32,
            fg_mask, lids32, anchor_label, areas, lambda_area32,
        )
        elapsed = perf_counter() - t0
        print(f"  {elapsed:.1f}s  flips={n_flips}", flush=True)
        energy_log.append({"iteration": iteration + 1, "flips": int(n_flips)})
        if n_flips == 0:
            print(f"  Converged after round {iteration + 1}.", flush=True)
            break
        if _should_stop_after_round(n_flips, min_round_flips):
            print(
                f"  Stopping after round {iteration + 1}: "
                f"flips={n_flips} < min_round_flips={min_round_flips}.",
                flush=True,
            )
            break

    return labels, energy_log


def _json_default(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _save_unary_images(
    unary: dict[tuple[int, int], np.ndarray],
    fg_mask: np.ndarray,
    label_ids: np.ndarray,
    output_dir: Path,
    cap: float = 10.0,
    cost_field: np.ndarray | None = None,
) -> None:
    """Save per-label (T,Y,X) unary cost TIFFs, an argmin label map, and optionally
    the MCP cost field (resistance surface) used to compute geodesic distances.

    Values are clipped to [0, cap] (INF → cap) so the TIFFs are displayable.
    Background pixels are written as 0.
    """
    T, Y, X = fg_mask.shape
    K = len(label_ids)
    img_dir = output_dir / "unary_images"
    img_dir.mkdir(exist_ok=True)

    # Build dense (T, K, Y, X) array for argmin; fill with cap for dead/bg.
    dense = np.full((T, K, Y, X), cap, dtype=np.float32)
    for ki, k in enumerate(label_ids):
        for t in range(T):
            u = unary.get((t, int(k)))
            if u is None:
                continue
            clipped = np.where(fg_mask[t], np.clip(u, 0.0, cap), 0.0).astype(np.float32)
            dense[t, ki] = clipped

    # Per-label TIFFs: (T, Y, X) float32 — normalized geodesic distance to label k
    for ki, k in enumerate(label_ids):
        tifffile.imwrite(
            img_dir / f"label_{int(k):03d}.tif",
            dense[:, ki],  # (T, Y, X)
            compression="zlib",
        )

    # Argmin map: for each fg pixel, which label has the lowest unary cost?
    argmin_ki = np.argmin(dense, axis=1)  # (T, Y, X) int index into label_ids
    argmin_labels = label_ids[argmin_ki].astype(np.uint16)
    argmin_labels[~fg_mask] = 0
    tifffile.imwrite(img_dir / "argmin.tif", argmin_labels, compression="zlib")

    # Minimum geodesic distance across all labels
    min_geodesic = np.min(dense, axis=1)  # (T, Y, X)
    min_geodesic[~fg_mask] = 0.0
    tifffile.imwrite(img_dir / "min_geodesic.tif", min_geodesic, compression="zlib")

    # MCP cost field: 1 + alpha_unary * contour (the resistance surface MCP traverses)
    if cost_field is not None:
        cf = np.where(fg_mask, cost_field, 0.0).astype(np.float32)
        tifffile.imwrite(img_dir / "costfield.tif", cf, compression="zlib")

    saved = f"{K} labels + argmin + min_geodesic" + (" + costfield" if cost_field is not None else "")
    print(f"  Saved unary images ({saved}) → {img_dir}", flush=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2D+T multi-label alpha-expansion graph-cut for cell segmentation."
    )
    p.add_argument("--pos-dir", type=Path, default=DEFAULT_POS_DIR)
    p.add_argument("--alpha-unary", type=float, default=4.0,
                   help="Contour weight in geodesic cost field.")
    p.add_argument("--lambda-s", type=float, default=1.0,
                   help="Spatial pairwise weight.")
    p.add_argument("--beta-s", type=float, default=5.0,
                   help="Contour sensitivity in spatial pairwise.")
    p.add_argument("--lambda-t", type=float, default=1.0,
                   help="Temporal pairwise weight.")
    p.add_argument("--boundary-mode", choices=["contour", "foreground_inverse", "combined"],
                   default="contour",
                   help="Image signal used for spatial pairwise costs. "
                        "contour uses the contour map; foreground_inverse uses 1 - fg score; "
                        "combined uses clip(contour + (1 - fg)).")
    p.add_argument("--foreground-score-path", type=Path, default=None,
                   help="Foreground score TIFF for --boundary-mode foreground_inverse. "
                        "Defaults to 3_cell/foreground_scores.tif.")
    p.add_argument("--flow-field-path", type=Path, default=None,
                   help="Flow field TIFF for flow/geodesic_flow unaries. "
                        "Defaults to 3_cell/filtered_dp.tif.")
    p.add_argument("--n-iters", type=int, default=3,
                   help="Alpha-expansion rounds.")
    p.add_argument("--unary-mode", choices=["geodesic", "euclidean", "flow", "geodesic_flow", "costfield"],
                   default="geodesic",
                   help="geodesic: contour-weighted MCP; euclidean: centroid Euclidean; "
                        "flow: flow-endpoint distance to nucleus mask; "
                        "geodesic_flow: geodesic + lambda_flow * flow.")
    p.add_argument("--init-mode", choices=["unary", "euclidean", "geodesic", "nuclei", "watershed_flow", "watershed_geodesic"], default="nuclei",
                   help="nuclei (default): only nucleus pixels are labeled at init — "
                        "all other fg pixels start at 0 and grow naturally via "
                        "alpha-expansion; "
                        "unary: init from argmin of the selected unary; "
                        "euclidean: always init from Euclidean centroid argmin; "
                        "geodesic: always init from geodesic argmin.")
    p.add_argument("--lambda-flow", type=float, default=1.0,
                   help="Weight for flow endpoint unary in geodesic_flow mode.")
    p.add_argument("--gamma-unary", type=float, default=0.0,
                   help="Weight for (1 - foreground_score) added to the geodesic cost "
                        "field. 0 (default) keeps the legacy contour-only field; positive "
                        "values penalise paths through low-foreground regions.")
    p.add_argument("--lambda-geodesic", type=float, default=1.0,
                   help="Weight for geodesic unary in geodesic_flow mode.")
    p.add_argument("--min-round-flips", type=int, default=0,
                   help="Stop alpha-expansion after a round with fewer than this many flips.")
    p.add_argument("--unary-cache-dir", type=Path, default=None,
                   help="Directory for reusable HDF5 unary caches. Defaults under 4_cell_graphcut.")
    p.add_argument("--no-unary-cache", action="store_true",
                   help="Disable persistent unary cache reads/writes.")
    p.add_argument("--solver", choices=["graphcut", "icm"], default="graphcut",
                   help="graphcut: PyMaxflow α-expansion (original); "
                        "icm: Numba parallel ICM with Numba flow unary.")
    p.add_argument("--lambda-area", type=float, default=0.0,
                   help="ICM only: weight for per-label frame-to-frame area change penalty "
                        "((A_k(t) - A_k(t-1))^2 summed over all labels and frames). "
                        "Quadratic: delta scales with actual deviation, so larger area "
                        "changes get a stronger signal. Start around 1e-4 to 1e-3 "
                        "(cells with ~5000px and 5%% variation give deltas ~500). "
                        "0 = disabled (default).")
    p.add_argument("--lambda-contour", type=float, default=0.0,
                   help="Add lambda_contour * median2_contour to the pairwise boundary "
                        "signal (after --boundary-mode is applied). Both signals are "
                        "high at ridges so polarities match. 0 = disabled (default).")
    p.add_argument("--n-workers", type=int, default=1,
                   help="Number of parallel worker processes for α-expansion "
                        "(uses fork; each worker builds one binary graph). "
                        "Default 1 (sequential). Try 4 on a 32 GiB machine.")
    p.add_argument("--preserve-connectivity", action="store_true",
                   help="Experiment: reject alpha-expansion flips that would split "
                        "the source label in a 2D frame.")
    p.add_argument("--save-unary-images", action="store_true",
                   help="Save per-label unary cost TIFFs and an argmin map into "
                        "unary_images/ inside the output directory.")
    p.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--crop", type=int, nargs=6,
                   metavar=("T0", "T1", "Y0", "Y1", "X0", "X1"))
    return p.parse_args()


def _load_inputs(
    pos_dir: Path, crop: list[int] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    def _read(path: Path, dtype) -> np.ndarray:
        a = np.asarray(tifffile.imread(path), dtype=dtype)
        if a.ndim == 4 and a.shape[1] == 1:
            a = a[:, 0]
        return a

    nuc = _read(pos_dir / "2_nucleus/tracked_labels.tif", np.uint32)
    fg = _read(pos_dir / "3_cell/foreground_masks.tif", np.uint8) > 0
    ct = _read(pos_dir / "3_cell/contour_maps.tif", np.float32)
    gt_path = pos_dir / "3_cell/tracked_labels.tif"
    gt = _read(gt_path, np.uint32) if gt_path.exists() else None

    required = [(nuc, "nuc"), (fg, "fg"), (ct, "contours")]
    if gt is not None:
        required.append((gt, "gt"))
    for arr, name in required:
        if arr.ndim != 3:
            raise ValueError(f"{name}: expected (T,Y,X), got {arr.shape}")

    shapes = {nuc.shape, fg.shape, ct.shape}
    if gt is not None:
        shapes.add(gt.shape)
    if len(shapes) > 1:
        raise ValueError(f"Shape mismatch: {shapes}")

    if crop is not None:
        t0, t1, y0, y1, x0, x1 = crop
        s = (slice(t0, t1), slice(y0, y1), slice(x0, x1))
        nuc, fg, ct = nuc[s], fg[s], ct[s]
        if gt is not None:
            gt = gt[s]

    # Always include tracked nucleus pixels in the cell foreground so they are
    # part of the graph and can be anchored to their track ID — see also the
    # `_apply_nucleus_anchors` call after the unary cache is loaded.
    fg = fg | (nuc > 0)

    return nuc, fg, ct, gt


def _read_stack(path: Path, dtype) -> np.ndarray:
    a = np.asarray(tifffile.imread(path), dtype=dtype)
    if a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0]
    return a


def _load_foreground_scores(path: Path, crop: list[int] | None) -> np.ndarray:
    scores = _read_stack(path, np.float32)
    if scores.ndim != 3:
        raise ValueError(f"foreground_scores: expected (T,Y,X), got {scores.shape}")
    if crop is not None:
        t0, t1, y0, y1, x0, x1 = crop
        scores = scores[t0:t1, y0:y1, x0:x1]
    return scores


def _prepare_boundary_signal(
    contours: np.ndarray,
    foreground_scores: np.ndarray | None,
    mode: str,
) -> np.ndarray:
    if mode == "contour":
        return np.clip(contours, 0.0, 1.0).astype(np.float32, copy=False)
    if mode == "foreground_inverse":
        if foreground_scores is None:
            raise ValueError("foreground_scores are required for foreground_inverse boundary mode")
        if foreground_scores.shape != contours.shape:
            raise ValueError(
                f"foreground_scores shape {foreground_scores.shape} does not match "
                f"contours shape {contours.shape}"
            )
        return (1.0 - np.clip(foreground_scores, 0.0, 1.0)).astype(np.float32)
    if mode == "combined":
        if foreground_scores is None:
            raise ValueError("foreground_scores are required for combined boundary mode")
        if foreground_scores.shape != contours.shape:
            raise ValueError(
                f"foreground_scores shape {foreground_scores.shape} does not match "
                f"contours shape {contours.shape}"
            )
        c = np.clip(contours, 0.0, 1.0).astype(np.float32, copy=False)
        f = (1.0 - np.clip(foreground_scores, 0.0, 1.0)).astype(np.float32)
        return np.clip(c + f, 0.0, 1.0)
    raise ValueError(f"Unsupported boundary mode {mode!r}")


def _compute_geodesic_unaries(
    nuc_tracks: np.ndarray,
    fg_mask: np.ndarray,
    contours: np.ndarray,
    label_ids: np.ndarray,
    alpha_unary: float,
    foreground_scores: np.ndarray | None = None,
    gamma_unary: float = 0.0,
) -> dict[tuple[int, int], np.ndarray]:
    """Compute normalized geodesic unary costs for each alive (frame, label) pair.

    Cost field per pixel: 1 + alpha_unary * contour + gamma_unary * (1 - fg).
    The MCP shortest-path then naturally bends around high-cost regions
    (high contour and/or low foreground score).
    """
    T, Y, X = fg_mask.shape
    unary: dict[tuple[int, int], np.ndarray] = {}

    use_fg = gamma_unary != 0.0 and foreground_scores is not None
    for t in range(T):
        fg_t = fg_mask[t]
        cost_field = np.full((Y, X), np.inf, dtype=np.float32)
        c = 1.0 + alpha_unary * contours[t][fg_t]
        if use_fg:
            c = c + gamma_unary * (1.0 - np.clip(foreground_scores[t][fg_t], 0.0, 1.0))
        cost_field[fg_t] = c

        alive = [int(k) for k in label_ids if np.any(nuc_tracks[t] == k)]
        if not alive:
            continue

        raw: dict[int, np.ndarray] = {}
        for k in alive:
            starts = [tuple(int(v) for v in c) for c in np.argwhere(nuc_tracks[t] == k)]
            mcp = MCP_Geometric(cost_field, fully_connected=True)
            cum, _ = mcp.find_costs(starts)
            d = cum.astype(np.float32)
            d[~fg_t] = np.inf
            raw[k] = d

        # Normalize per frame
        all_finite = np.concatenate([d[np.isfinite(d)] for d in raw.values()])
        med = float(np.median(all_finite)) if all_finite.size > 0 else 1.0
        if med <= 0.0:
            med = 1.0

        for k, d in raw.items():
            nd = d / med
            nd[~np.isfinite(nd)] = _INF
            unary[(t, k)] = nd

        # Hard nucleus anchors: set INF for wrong labels at nucleus pixels
        for k in alive:
            k_pix = nuc_tracks[t] == k
            if not k_pix.any():
                continue
            for j in alive:
                if j != k and (t, j) in unary:
                    unary[(t, j)][k_pix] = _INF

        if (t + 1) % 10 == 0 or t + 1 == T:
            print(f"  geodesic unaries: frame {t+1}/{T}, {len(alive)} alive", flush=True)

    return unary


def _combine_unaries(
    base: dict[tuple[int, int], np.ndarray],
    extra: dict[tuple[int, int], np.ndarray],
    lambda_geodesic: float,
    lambda_flow: float,
) -> dict[tuple[int, int], np.ndarray]:
    """Return lambda_geodesic * base + lambda_flow * extra, preserving INF constraints."""
    combined: dict[tuple[int, int], np.ndarray] = {}
    for key, base_cost in base.items():
        if key not in extra:
            merged = base_cost.astype(np.float32, copy=True)
            finite = base_cost < _INF
            merged[finite] = float(lambda_geodesic) * base_cost[finite]
            merged[~finite] = _INF
            combined[key] = merged
            continue
        extra_cost = extra[key]
        merged = base_cost.astype(np.float32, copy=True)
        finite = (base_cost < _INF) & (extra_cost < _INF)
        merged[finite] = (
            float(lambda_geodesic) * base_cost[finite]
            + float(lambda_flow) * extra_cost[finite]
        )
        merged[~finite] = _INF
        combined[key] = merged
    return combined


def _should_stop_after_round(total_flips: int, min_flips: int) -> bool:
    return int(min_flips) > 0 and int(total_flips) < int(min_flips)


def _round_energy_is_worse(
    current_energy: float,
    best_energy: float,
    tolerance: float = 1e-6,
) -> bool:
    return float(current_energy) > float(best_energy) + float(tolerance)


def _cache_suffix(
    mode: str,
    shape: tuple[int, int, int],
    crop: list[int] | None,
    alpha_unary: float,
) -> str:
    crop_s = "full" if crop is None else "_".join(str(v) for v in crop)
    return f"{mode}_shape-{shape[0]}x{shape[1]}x{shape[2]}_crop-{crop_s}_alpha-{alpha_unary:g}.h5"


def _read_unary_cache(path: Path) -> dict[tuple[int, int], np.ndarray] | None:
    if not path.exists():
        return None
    unary: dict[tuple[int, int], np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for name, ds in f["unaries"].items():
            t_s, k_s = name.split("_", 1)
            unary[(int(t_s), int(k_s))] = ds[...].astype(np.float32, copy=False)
    print(f"  Loaded unary cache: {path}", flush=True)
    return unary


def _write_unary_cache(path: Path, unary: dict[tuple[int, int], np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as f:
        g = f.create_group("unaries")
        for (t, k), arr in unary.items():
            g.create_dataset(
                f"{int(t)}_{int(k)}",
                data=np.asarray(arr, dtype=np.float32),
                compression="lzf",
            )
    tmp.replace(path)
    print(f"  Wrote unary cache: {path}", flush=True)


def _get_cached_unaries(
    cache_dir: Path | None,
    mode: str,
    shape: tuple[int, int, int],
    crop: list[int] | None,
    alpha_unary: float,
    compute,
) -> dict[tuple[int, int], np.ndarray]:
    if cache_dir is None:
        return compute()
    path = cache_dir / _cache_suffix(mode, shape, crop, alpha_unary)
    cached = _read_unary_cache(path)
    if cached is not None:
        return cached
    unary = compute()
    _write_unary_cache(path, unary)
    return unary


def _apply_nucleus_anchors(
    unary: dict[tuple[int, int], np.ndarray],
    nuc_tracks: np.ndarray,
    label_ids: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Re-apply hard nucleus anchors after loading a (possibly stale) cache.

    For each frame t and each track k alive at t:
      * `unary[(t, k)]` at nucleus pixels of k is forced to 0 so the pixel can be
        claimed by k even if the cached value was INF (e.g. the pixel was outside
        the cell foreground when the cache was first written).
      * `unary[(t, j)]` at nucleus pixels of k is forced to INF for every other
        cached track j, so no other label can steal the nucleus pixel.

    The unary dict is mutated in place and also returned for convenience.
    """
    T = nuc_tracks.shape[0]
    label_ids = [int(k) for k in label_ids]
    # Group cached labels by frame so we can also anchor against tracks whose
    # cache entries linger from a previous (stale) state.
    cached_labels_by_t: dict[int, list[int]] = {}
    for (t, j) in unary:
        cached_labels_by_t.setdefault(int(t), []).append(int(j))

    for t in range(T):
        alive = [k for k in label_ids if int((nuc_tracks[t] == k).sum()) > 0]
        cached = cached_labels_by_t.get(t, [])
        for k in alive:
            k_pix = nuc_tracks[t] == k
            if (t, k) in unary:
                unary[(t, k)][k_pix] = 0.0
            for j in cached:
                if j != k:
                    unary[(t, j)][k_pix] = _INF
    return unary


def _compute_costfield_unaries(
    nuc_tracks: np.ndarray,
    fg_mask: np.ndarray,
    contours: np.ndarray,
    label_ids: np.ndarray,
    alpha_unary: float,
    foreground_scores: np.ndarray | None = None,
    gamma_unary: float = 0.0,
) -> dict[tuple[int, int], np.ndarray]:
    """Use the MCP cost field directly as the unary, bypassing geodesic path distances.

    Every alive (t, k) pair gets unary(p) = 1 + alpha_unary * contour(p)
    (+ gamma_unary * (1 - fg_score) if requested) at every foreground pixel.
    The value is label-independent, so the solver is driven purely by pairwise
    boundary costs. Hard nucleus anchors are applied as usual.
    """
    T, Y, X = fg_mask.shape
    unary: dict[tuple[int, int], np.ndarray] = {}

    use_fg = gamma_unary != 0.0 and foreground_scores is not None
    for t in range(T):
        cf = (1.0 + alpha_unary * np.clip(contours[t], 0.0, 1.0)).astype(np.float32)
        if use_fg:
            cf = cf + gamma_unary * (1.0 - np.clip(foreground_scores[t], 0.0, 1.0))
        alive = [int(k) for k in label_ids if np.any(nuc_tracks[t] == k)]
        for k in alive:
            u = np.where(fg_mask[t], cf, _INF).astype(np.float32)
            unary[(t, k)] = u

        # Hard nucleus anchors
        for k in alive:
            k_pix = nuc_tracks[t] == k
            if not k_pix.any():
                continue
            unary[(t, k)][k_pix] = 0.0
            for j in alive:
                if j != k:
                    unary[(t, j)][k_pix] = _INF

    return unary


def _compute_euclidean_unaries(
    nuc_tracks: np.ndarray,
    fg_mask: np.ndarray,
    label_ids: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Euclidean distance from each pixel to the centroid of nucleus k at frame t.

    No contour information — pairwise is the sole contour-aware term.
    Normalised per frame by median reachable distance. Hard nucleus anchors same as geodesic mode.
    """
    T, Y, X = fg_mask.shape
    yy, xx = np.mgrid[:Y, :X]
    unary: dict[tuple[int, int], np.ndarray] = {}

    for t in range(T):
        fg_t = fg_mask[t]
        alive = [int(k) for k in label_ids if np.any(nuc_tracks[t] == k)]
        if not alive:
            continue

        raw: dict[int, np.ndarray] = {}
        for k in alive:
            pix = np.argwhere(nuc_tracks[t] == k)
            cy, cx = pix[:, 0].mean(), pix[:, 1].mean()
            d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
            d[~fg_t] = np.inf
            raw[k] = d

        all_finite = np.concatenate([d[np.isfinite(d)] for d in raw.values()])
        med = float(np.median(all_finite)) if all_finite.size > 0 else 1.0
        if med <= 0.0:
            med = 1.0

        for k, d in raw.items():
            nd = d / med
            nd[~np.isfinite(nd)] = _INF
            unary[(t, k)] = nd

        # Hard nucleus anchors
        for k in alive:
            k_pix = nuc_tracks[t] == k
            if not k_pix.any():
                continue
            for j in alive:
                if j != k and (t, j) in unary:
                    unary[(t, j)][k_pix] = _INF

        if (t + 1) % 10 == 0 or t + 1 == T:
            print(f"  euclidean unaries: frame {t+1}/{T}, {len(alive)} alive", flush=True)

    return unary


def _integrate_flow_endpoints(
    flow_field: np.ndarray,  # (T, 2, Y, X) float32 — channel 0=dY, 1=dX
    fg_mask: np.ndarray,     # (T, Y, X) bool
    K: int = 50,
    step_scale: float = 0.5,
) -> np.ndarray:
    """Walk K steps per fg pixel along flow field. Returns (T,Y,X,2) float32 endpoints."""
    T, _, Y, X = flow_field.shape
    endpoints = np.zeros((T, Y, X, 2), dtype=np.float32)
    yy = np.arange(Y, dtype=np.float32)[:, None] * np.ones((1, X), dtype=np.float32)
    xx = np.arange(X, dtype=np.float32)[None, :] * np.ones((Y, 1), dtype=np.float32)
    for t in range(T):
        flow_dy = flow_field[t, 0]  # (Y, X)
        flow_dx = flow_field[t, 1]  # (Y, X)
        py = yy.copy()
        px = xx.copy()
        for _ in range(K):
            py_c = np.clip(py, 0, Y - 1.0001)
            px_c = np.clip(px, 0, X - 1.0001)
            iy0 = py_c.astype(np.int32)
            ix0 = px_c.astype(np.int32)
            fy = py_c - iy0
            fx = px_c - ix0
            iy1 = np.minimum(iy0 + 1, Y - 1)
            ix1 = np.minimum(ix0 + 1, X - 1)
            w00 = (1 - fy) * (1 - fx)
            w10 = fy * (1 - fx)
            w01 = (1 - fy) * fx
            w11 = fy * fx
            dy = (flow_dy[iy0, ix0] * w00 + flow_dy[iy1, ix0] * w10 +
                  flow_dy[iy0, ix1] * w01 + flow_dy[iy1, ix1] * w11)
            dx = (flow_dx[iy0, ix0] * w00 + flow_dx[iy1, ix0] * w10 +
                  flow_dx[iy0, ix1] * w01 + flow_dx[iy1, ix1] * w11)
            py = np.clip(py + step_scale * dy, 0, Y - 1)
            px = np.clip(px + step_scale * dx, 0, X - 1)
        endpoints[t, :, :, 0] = py
        endpoints[t, :, :, 1] = px
        if (t + 1) % 10 == 0 or t + 1 == T:
            print(f"  flow integration: frame {t+1}/{T}", flush=True)
    return endpoints


def _compute_flow_unaries(
    nuc_tracks: np.ndarray,
    fg_mask: np.ndarray,
    flow_field: np.ndarray,  # (T, 2, Y, X)
    label_ids: np.ndarray,
    K: int = 50,
    step_scale: float = 0.5,
    cap_px: float = 100.0,
) -> dict[tuple[int, int], np.ndarray]:
    """Flow-endpoint unary: distance from flow endpoint to nearest nucleus mask pixel.

    Walk K steps along flow field to get endpoint q for each fg pixel.
    unary(p, k) = clip(dist(q_p, M_k(t)), 0, cap_px). Normalized per frame by median.
    Hard nucleus anchors same as other modes.
    """
    T, Y, X = fg_mask.shape
    print("  Integrating flow endpoints...", flush=True)
    endpoints = _integrate_flow_endpoints(flow_field, fg_mask, K=K, step_scale=step_scale)

    unary: dict[tuple[int, int], np.ndarray] = {}

    for t in range(T):
        fg_t = fg_mask[t]
        alive = [int(k) for k in label_ids if np.any(nuc_tracks[t] == k)]
        if not alive:
            continue

        eq_y = endpoints[t, :, :, 0]  # (Y, X) float32 endpoint y-coords
        eq_x = endpoints[t, :, :, 1]

        raw: dict[int, np.ndarray] = {}
        for k in alive:
            nuc_mask_k = nuc_tracks[t] == k
            edt = distance_transform_edt(~nuc_mask_k).astype(np.float32)

            # Bilinear lookup of EDT at endpoint positions
            ey0 = np.clip(eq_y.astype(np.int32), 0, Y - 2)
            ex0 = np.clip(eq_x.astype(np.int32), 0, X - 2)
            fy = np.clip(eq_y - ey0, 0.0, 1.0)
            fx = np.clip(eq_x - ex0, 0.0, 1.0)
            ey1 = ey0 + 1
            ex1 = ex0 + 1
            d_endpoint = (
                edt[ey0, ex0] * (1 - fy) * (1 - fx) +
                edt[ey1, ex0] * fy * (1 - fx) +
                edt[ey0, ex1] * (1 - fy) * fx +
                edt[ey1, ex1] * fy * fx
            )
            d = np.clip(d_endpoint, 0.0, cap_px).astype(np.float32)
            d[~fg_t] = np.inf
            raw[k] = d

        all_finite = np.concatenate([d[np.isfinite(d)] for d in raw.values()])
        med = float(np.median(all_finite)) if all_finite.size > 0 else 1.0
        if med <= 0.0:
            med = 1.0

        for k, d in raw.items():
            nd = d / med
            nd[~np.isfinite(nd)] = _INF
            unary[(t, k)] = nd

        # Hard nucleus anchors
        for k in alive:
            k_pix = nuc_tracks[t] == k
            if not k_pix.any():
                continue
            for j in alive:
                if j != k and (t, j) in unary:
                    unary[(t, j)][k_pix] = _INF

        if (t + 1) % 10 == 0 or t + 1 == T:
            print(f"  flow unaries: frame {t+1}/{T}, {len(alive)} alive", flush=True)

    return unary


def _compute_pairwise_weights(
    fg_mask: np.ndarray,
    contours: np.ndarray,
    lambda_s: float,
    beta_s: float,
    lambda_t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute (T,Y,X) n-link weight arrays for spatial (h,v,dr,dl) and temporal edges.

    Diagonal edges (dr=down-right, dl=down-left) are scaled by 1/sqrt(2) relative to
    axis-aligned edges so the total boundary penalty is approximately isotropic.
    """
    T, Y, X = fg_mask.shape
    fg = fg_mask.astype(bool)
    c = contours

    h = np.zeros((T, Y, X), dtype=np.float32)
    h[:, :, :-1] = (
        lambda_s * np.exp(-beta_s * 0.5 * (c[:, :, :-1] + c[:, :, 1:]))
        * (fg[:, :, :-1] & fg[:, :, 1:])
    ).astype(np.float32)

    v = np.zeros((T, Y, X), dtype=np.float32)
    v[:, :-1, :] = (
        lambda_s * np.exp(-beta_s * 0.5 * (c[:, :-1, :] + c[:, 1:, :]))
        * (fg[:, :-1, :] & fg[:, 1:, :])
    ).astype(np.float32)

    dr = np.zeros((T, Y, X), dtype=np.float32)
    dr[:, :-1, :-1] = (
        _DIAG_SCALE * lambda_s
        * np.exp(-beta_s * 0.5 * (c[:, :-1, :-1] + c[:, 1:, 1:]))
        * (fg[:, :-1, :-1] & fg[:, 1:, 1:])
    ).astype(np.float32)

    dl = np.zeros((T, Y, X), dtype=np.float32)
    dl[:, :-1, 1:] = (
        _DIAG_SCALE * lambda_s
        * np.exp(-beta_s * 0.5 * (c[:, :-1, 1:] + c[:, 1:, :-1]))
        * (fg[:, :-1, 1:] & fg[:, 1:, :-1])
    ).astype(np.float32)

    tw = np.zeros((T, Y, X), dtype=np.float32)
    if T > 1:
        tw[:-1, :, :] = (lambda_t * (fg[:-1] & fg[1:])).astype(np.float32)

    # Spatiotemporal face-diagonal edges: connect (t,y,x) to (t+1,y±1,x) and (t+1,y,x±1).
    # Scaled by 1/√2 to account for the larger spacetime distance.
    # tw_ty_dn[t,y,x] = weight for (t,y,x)↔(t+1,y+1,x)
    # tw_ty_up[t,y,x] = weight for (t,y,x)↔(t+1,y-1,x)
    # tw_tx_r [t,y,x] = weight for (t,y,x)↔(t+1,y,x+1)
    # tw_tx_l [t,y,x] = weight for (t,y,x)↔(t+1,y,x-1)
    tw_ty_dn = np.zeros((T, Y, X), dtype=np.float32)
    tw_ty_up = np.zeros((T, Y, X), dtype=np.float32)
    tw_tx_r  = np.zeros((T, Y, X), dtype=np.float32)
    tw_tx_l  = np.zeros((T, Y, X), dtype=np.float32)
    if T > 1:
        tw_ty_dn[:-1, :-1, :] = (_DIAG_SCALE * lambda_t * (fg[:-1, :-1, :] & fg[1:, 1:, :])).astype(np.float32)
        tw_ty_up[:-1, 1:,  :] = (_DIAG_SCALE * lambda_t * (fg[:-1, 1:,  :] & fg[1:, :-1, :])).astype(np.float32)
        tw_tx_r [:-1, :, :-1] = (_DIAG_SCALE * lambda_t * (fg[:-1, :, :-1] & fg[1:, :, 1:])).astype(np.float32)
        tw_tx_l [:-1, :,  1:] = (_DIAG_SCALE * lambda_t * (fg[:-1, :,  1:] & fg[1:, :, :-1])).astype(np.float32)

    return h, v, dr, dl, tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l


def _build_current_cost(
    current_labels: np.ndarray,
    fg_mask: np.ndarray,
    unary: dict[tuple[int, int], np.ndarray],
    label_ids: np.ndarray,
) -> np.ndarray:
    """Build (T,Y,X) array of unary cost for each pixel's current label."""
    T, Y, X = fg_mask.shape
    cc = np.zeros((T, Y, X), dtype=np.float32)
    for t in range(T):
        cl = current_labels[t]
        for k in label_ids:
            m = fg_mask[t] & (cl == k)
            if m.any() and (t, int(k)) in unary:
                cc[t][m] = unary[(t, int(k))][m]
            elif m.any():
                cc[t][m] = _INF
    # background source cap = 0 (background always stays in sink)
    cc[~fg_mask] = 0.0
    return cc


def _compute_energy(
    current_labels: np.ndarray,
    fg_mask: np.ndarray,
    unary: dict[tuple[int, int], np.ndarray],
    label_ids: np.ndarray,
    h: np.ndarray,
    v: np.ndarray,
    dr: np.ndarray,
    dl: np.ndarray,
    tw: np.ndarray,
) -> float:
    T, Y, X = fg_mask.shape
    e_u = 0.0
    for t in range(T):
        cl = current_labels[t]
        for k in label_ids:
            m = fg_mask[t] & (cl == k)
            if m.any() and (t, int(k)) in unary:
                e_u += float(np.sum(unary[(t, int(k))][m]))
            elif m.any():
                e_u += float(m.sum()) * _INF

    lbl = current_labels
    diff_h = (lbl[:, :, :-1] != lbl[:, :, 1:]) & fg_mask[:, :, :-1] & fg_mask[:, :, 1:]
    e_s = float(np.sum(h[:, :, :-1] * diff_h))
    diff_v = (lbl[:, :-1, :] != lbl[:, 1:, :]) & fg_mask[:, :-1, :] & fg_mask[:, 1:, :]
    e_s += float(np.sum(v[:, :-1, :] * diff_v))
    diff_dr = (lbl[:, :-1, :-1] != lbl[:, 1:, 1:]) & fg_mask[:, :-1, :-1] & fg_mask[:, 1:, 1:]
    e_s += float(np.sum(dr[:, :-1, :-1] * diff_dr))
    diff_dl = (lbl[:, :-1, 1:] != lbl[:, 1:, :-1]) & fg_mask[:, :-1, 1:] & fg_mask[:, 1:, :-1]
    e_s += float(np.sum(dl[:, :-1, 1:] * diff_dl))
    diff_t = (lbl[:-1] != lbl[1:]) & fg_mask[:-1] & fg_mask[1:]
    e_t = float(np.sum(tw[:-1] * diff_t))
    return e_u + e_s + e_t


def _run_alpha_expansion(
    fg_mask: np.ndarray,
    unary: dict[tuple[int, int], np.ndarray],
    label_ids: np.ndarray,
    h: np.ndarray,
    v: np.ndarray,
    dr: np.ndarray,
    dl: np.ndarray,
    tw: np.ndarray,
    n_iters: int,
    nuc_tracks: np.ndarray | None = None,
    init_unary: dict[tuple[int, int], np.ndarray] | None = None,
    init_labels: np.ndarray | None = None,
    nuclei_only_init: bool = False,
    min_round_flips: int = 0,
    n_workers: int = 1,
    preserve_connectivity: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    T, Y, X = fg_mask.shape
    n_fg = int(np.count_nonzero(fg_mask))
    # Compact node IDs: foreground pixels → 0..n_fg-1; background → n_fg (dummy SINK).
    fg_node_id = np.full((T, Y, X), n_fg, dtype=np.int32)
    fg_node_id[fg_mask] = np.arange(n_fg, dtype=np.int32)
    fg_flat = np.arange(n_fg, dtype=np.int32)

    # Pre-quantize all cost arrays once (float → int32) for Graph[int].
    unary_int: dict[tuple[int, int], np.ndarray] = {
        k: _quantize(v_arr) for k, v_arr in unary.items()
    }
    h_int = _quantize(h)
    v_int = _quantize(v)
    dr_int = _quantize(dr)
    dl_int = _quantize(dl)
    tw_int = _quantize(tw)

    print("Precomputing foreground-only edge lists...", flush=True)
    e_h, e_v, e_dr, e_dl, e_tw = _precompute_fg_edge_lists(
        fg_mask, h_int, v_int, dr_int, dl_int, tw_int
    )

    current_labels = np.zeros((T, Y, X), dtype=np.uint32)
    if init_labels is not None:
        print("Initializing labels from provided watershed...", flush=True)
        current_labels = init_labels.copy().astype(np.uint32)
    elif nuclei_only_init:
        if nuc_tracks is None:
            raise ValueError("nuclei_only_init requires nuc_tracks")
        print(
            "Initializing labels from nucleus pixels only "
            "(other fg pixels start at 0 and grow via alpha-expansion)...",
            flush=True,
        )
    else:
        # Initialize: for each foreground pixel, pick cheapest alive label.
        init_u = init_unary if init_unary is not None else unary
        print("Initializing labels from minimum unary cost...", flush=True)
        for t in range(T):
            alive_t = [k for k in label_ids if (t, int(k)) in init_u]
            if not alive_t:
                continue
            min_cost = np.full((Y, X), _INF, dtype=np.float32)
            best = np.zeros((Y, X), dtype=np.uint32)
            for k in alive_t:
                u = init_u[(t, int(k))]
                better = fg_mask[t] & (u < min_cost)
                min_cost[better] = u[better]
                best[better] = k
            current_labels[t] = np.where(fg_mask[t], best, 0)

    # Force-paint nucleus pixels to their track ID — guards against the unary
    # init missing a label (e.g. a stale cache marking the pixel unreachable),
    # and provides the seed for the nuclei_only_init path.
    if nuc_tracks is not None:
        nuc_mask = nuc_tracks > 0
        current_labels[nuc_mask] = nuc_tracks[nuc_mask].astype(np.uint32)

    print("Building initial current_cost array...", flush=True)
    current_cost = _build_current_cost(current_labels, fg_mask, unary, label_ids)

    energy_log: list[dict] = []
    best_labels = current_labels.copy()
    best_energy = _compute_energy(current_labels, fg_mask, unary, label_ids, h, v, dr, dl, tw)
    # Edge estimate: fg-fg pairs only (bg-touching edges have weight 0, still allocated).
    n_edges_est = (
        T * Y * (X - 1)           # horizontal
        + T * (Y - 1) * X         # vertical
        + T * (Y - 1) * (X - 1)   # down-right diagonal
        + T * (Y - 1) * (X - 1)   # down-left diagonal
        + (T - 1) * Y * X         # temporal
    )

    use_parallel = n_workers > 1

    for iteration in range(n_iters):
        print(f"\n=== Round {iteration + 1}/{n_iters} ===", flush=True)
        round_total_flips = 0
        changed_any = False

        # Quantize current_cost snapshot for t-links.
        cc_snap = _quantize(current_cost)
        cl_snap = current_labels.copy()

        if use_parallel:
            # Set module-level globals before forking so workers inherit them
            # via Linux COW — the large arrays are not re-pickled per task.
            global _W_FG, _W_FG_NODE_ID, _W_N_FG
            global _W_CC, _W_CL, _W_UNARY_INT, _W_SHAPE, _W_N_EDGES
            global _W_FG_EDGES_H, _W_FG_EDGES_V, _W_FG_EDGES_DR, _W_FG_EDGES_DL, _W_FG_EDGES_TW
            _W_FG = fg_mask
            _W_FG_NODE_ID = fg_node_id
            _W_N_FG = n_fg
            _W_CC = cc_snap
            _W_CL = cl_snap
            _W_UNARY_INT = unary_int
            _W_SHAPE = (T, Y, X)
            _W_N_EDGES = n_edges_est
            _W_FG_EDGES_H = e_h
            _W_FG_EDGES_V = e_v
            _W_FG_EDGES_DR = e_dr
            _W_FG_EDGES_DL = e_dl
            _W_FG_EDGES_TW = e_tw

            t_round = perf_counter()
            with mp.get_context("fork").Pool(n_workers) as pool:
                results = pool.map(_alpha_cut, [int(a) for a in label_ids])

            # Apply all flip masks (snapshot-based parallel α-expansion;
            # conflicts resolved by label order — last writer wins).
            for alpha, flip_mask, _n_changed in results:
                if preserve_connectivity:
                    flip_mask = _filter_connectivity_preserving_flips(
                        current_labels=current_labels,
                        flip_mask=flip_mask,
                        alpha=int(alpha),
                    )
                n_changed = int(np.count_nonzero(current_labels[flip_mask] != alpha))
                if n_changed > 0:
                    current_labels[flip_mask] = alpha
                    changed_any = True
                round_total_flips += n_changed

            elapsed_round = perf_counter() - t_round
            print(
                f"  Round {iteration + 1}: {elapsed_round:.1f}s  "
                f"total_flips={round_total_flips}",
                flush=True,
            )
            for alpha, _, n_changed in results:
                print(f"    α={alpha:3d}: changed={n_changed:7d}", flush=True)

            # Rebuild current_cost from updated labels (parallel path doesn't
            # track incremental flips, so rebuild from scratch).
            current_cost = _build_current_cost(current_labels, fg_mask, unary, label_ids)

        else:
            # Sequential path — Graph[int], border-only candidate subgraph per alpha.
            for alpha in label_ids:
                alpha = int(alpha)
                t0 = perf_counter()

                cand = _candidate_mask(current_labels, fg_mask, alpha)
                n_cand = int(np.count_nonzero(cand))
                if n_cand == 0:
                    print(f"  α={alpha:3d}:  0.0s  changed=      0  (no candidate pixels)", flush=True)
                    continue

                # cand pixels → 0..n_cand-1; everything else → n_cand (dummy SINK).
                cand_node_id = np.full((T, Y, X), n_cand, dtype=np.int32)
                cand_node_id[cand] = np.arange(n_cand, dtype=np.int32)
                cand_flat = np.arange(n_cand, dtype=np.int32)

                alpha_cost_cand = np.full(n_cand, _INT_INF, dtype=np.int32)
                for t in range(T):
                    if (t, alpha) in unary_int:
                        m = cand[t]
                        alpha_cost_cand[cand_node_id[t][m]] = unary_int[(t, alpha)][m]

                # SOURCE anchor: currently-α candidates get INT_INF source cap (temporary copy).
                cc_cand = cc_snap[cand].copy()
                cc_cand[current_labels[cand] == alpha] = _INT_INF

                cand_node_id_flat = cand_node_id.ravel()
                n_edges_est_cand = n_cand * 10
                g = maxflow.Graph[int](n_cand + 1, n_edges_est_cand)
                g.add_nodes(n_cand + 1)
                g.add_tedge(n_cand, 0, _INT_INF)  # dummy always in SINK
                g.add_grid_tedges(cand_flat, cc_cand, alpha_cost_cand)
                YX = Y * X
                for t in range(T):
                    offset = t * YX
                    _add_fg_edges(g, cand_node_id_flat, e_h[t], n_cand, offset)
                    _add_fg_edges(g, cand_node_id_flat, e_v[t], n_cand, offset)
                    _add_fg_edges(g, cand_node_id_flat, e_dr[t], n_cand, offset)
                    _add_fg_edges(g, cand_node_id_flat, e_dl[t], n_cand, offset)
                src_gl, dst_gl, tw_w = e_tw
                src_nodes_tw = cand_node_id_flat[src_gl]
                dst_nodes_tw = cand_node_id_flat[dst_gl]
                keep_tw = (src_nodes_tw != n_cand) | (dst_nodes_tw != n_cand)
                if keep_tw.any():
                    g.add_edges(src_nodes_tw[keep_tw], dst_nodes_tw[keep_tw], tw_w[keep_tw], tw_w[keep_tw])
                g.maxflow()

                in_sink_cand = g.get_grid_segments(cand_flat)
                in_sink = np.ones((T, Y, X), dtype=bool)
                in_sink[cand] = in_sink_cand
                flip_mask = (~in_sink) & fg_mask
                if preserve_connectivity:
                    flip_mask = _filter_connectivity_preserving_flips(
                        current_labels=current_labels,
                        flip_mask=flip_mask,
                        alpha=alpha,
                    )
                prev = current_labels[flip_mask]
                n_changed = int(np.count_nonzero(prev != alpha))
                if n_changed > 0:
                    changed_any = True
                round_total_flips += n_changed

                current_labels[flip_mask] = alpha
                # Update current_cost and cc_snap incrementally for next alpha.
                for t in range(T):
                    ft = flip_mask[t]
                    if ft.any():
                        cost_val = unary[(t, alpha)][ft] if (t, alpha) in unary else _INF
                        current_cost[t][ft] = cost_val
                        cc_snap[t][ft] = unary_int[(t, alpha)][ft] if (t, alpha) in unary_int else _INT_INF

                elapsed = perf_counter() - t0
                print(f"  α={alpha:3d}: {elapsed:5.1f}s  changed={n_changed:7d}", flush=True)

        E = _compute_energy(current_labels, fg_mask, unary, label_ids, h, v, dr, dl, tw)
        reverted = False
        if _round_energy_is_worse(E, best_energy):
            print(
                f"  Round worsened energy ({E:.4f} > {best_energy:.4f}); "
                "reverting to best previous labeling.",
                flush=True,
            )
            current_labels = best_labels.copy()
            current_cost = _build_current_cost(current_labels, fg_mask, unary, label_ids)
            E = best_energy
            reverted = True
        else:
            best_energy = E
            best_labels = current_labels.copy()

        energy_log.append({
            "iteration": iteration + 1,
            "energy": round(E, 4),
            "total_flips": round_total_flips,
            "reverted": reverted,
        })
        print(
            f"  Energy after round {iteration + 1}: {E:.4f}  total_flips={round_total_flips}",
            flush=True,
        )

        if reverted:
            print(f"  Stopping after reverted round {iteration + 1}.", flush=True)
            break
        if not changed_any:
            print(f"  Converged after round {iteration + 1}.", flush=True)
            break
        if _should_stop_after_round(round_total_flips, min_round_flips):
            print(
                f"  Stopping after round {iteration + 1}: "
                f"total_flips={round_total_flips} < min_round_flips={min_round_flips}.",
                flush=True,
            )
            break

    return best_labels, energy_log


def _evaluate(
    pred: np.ndarray,
    gt: np.ndarray,
    fg_mask: np.ndarray,
    contours: np.ndarray,
    label_ids: np.ndarray,
) -> dict[str, Any]:
    T, Y, X = pred.shape
    metrics: dict[str, Any] = {}

    # 1. Coverage
    fg_voxels = int(np.count_nonzero(fg_mask))
    labeled_voxels = int(np.count_nonzero(pred[fg_mask]))
    metrics["coverage"] = labeled_voxels / fg_voxels if fg_voxels > 0 else 0.0
    metrics["fg_voxels"] = fg_voxels
    metrics["labeled_voxels"] = labeled_voxels

    # 2. Per-track temporal IoU
    all_ids = np.union1d(
        np.unique(pred[pred > 0]),
        np.unique(gt[gt > 0]),
    )
    per_track_iou: dict[int, float] = {}
    for k in all_ids:
        k = int(k)
        ious = []
        for t in range(T):
            pm = pred[t] == k
            gm = gt[t] == k
            if not pm.any() and not gm.any():
                continue
            inter = int((pm & gm).sum())
            union_v = int((pm | gm).sum())
            if union_v > 0:
                ious.append(inter / union_v)
        per_track_iou[k] = float(np.mean(ious)) if ious else 0.0
    metrics["per_track_iou"] = per_track_iou
    iou_vals = list(per_track_iou.values())
    metrics["mean_temporal_iou"] = float(np.mean(iou_vals)) if iou_vals else 0.0
    metrics["median_temporal_iou"] = float(np.median(iou_vals)) if iou_vals else 0.0
    metrics["p25_temporal_iou"] = float(np.percentile(iou_vals, 25)) if iou_vals else 0.0

    # 3. ID consistency (purity and completeness)
    gt_ids = [int(k) for k in np.unique(gt) if k > 0]
    pred_ids = [int(k) for k in np.unique(pred) if k > 0]

    purities = []
    for pk in pred_ids:
        pm = pred == pk
        total = int(pm.sum())
        if total == 0:
            continue
        best = max(
            (int((pm & (gt == gk)).sum()) for gk in gt_ids), default=0
        )
        purities.append(best / total)
    metrics["mean_purity"] = float(np.mean(purities)) if purities else 0.0

    completenesses = []
    for gk in gt_ids:
        gm = gt == gk
        total = int(gm.sum())
        if total == 0:
            continue
        best = max(
            (int((gm & (pred == pk)).sum()) for pk in pred_ids), default=0
        )
        completenesses.append(best / total)
    metrics["mean_completeness"] = float(np.mean(completenesses)) if completenesses else 0.0

    # 4. Label flicker rate (per track, then aggregate)
    def _flicker(vol: np.ndarray) -> float:
        if T < 2:
            return 0.0
        rates = []
        for k in [int(i) for i in np.unique(vol) if i > 0]:
            pm_t = vol[:-1] == k
            pm_t1 = vol[1:] == k
            both = pm_t | pm_t1
            n_both = int(both.sum())
            if n_both == 0:
                continue
            diff = (pm_t != pm_t1) & both
            rates.append(int(diff.sum()) / n_both)
        return float(np.mean(rates)) if rates else 0.0

    metrics["pred_flicker_rate"] = _flicker(pred)
    metrics["gt_flicker_rate"] = _flicker(gt)

    # 5. Boundary alignment: contour intensity at pred boundaries vs interiors
    pred_boundary = np.zeros((T, Y, X), dtype=bool)
    pred_boundary[:, :, :-1] |= (pred[:, :, :-1] != pred[:, :, 1:]) & fg_mask[:, :, :-1] & fg_mask[:, :, 1:]
    pred_boundary[:, :, 1:] |= (pred[:, :, :-1] != pred[:, :, 1:]) & fg_mask[:, :, :-1] & fg_mask[:, :, 1:]
    pred_boundary[:, :-1, :] |= (pred[:, :-1, :] != pred[:, 1:, :]) & fg_mask[:, :-1, :] & fg_mask[:, 1:, :]
    pred_boundary[:, 1:, :] |= (pred[:, :-1, :] != pred[:, 1:, :]) & fg_mask[:, :-1, :] & fg_mask[:, 1:, :]
    pred_interior = fg_mask & ~pred_boundary & (pred > 0)

    bnd_vals = contours[pred_boundary & fg_mask]
    int_vals = contours[pred_interior]
    if bnd_vals.size > 0 and int_vals.size > 0:
        metrics["boundary_contour_mean"] = float(bnd_vals.mean())
        metrics["interior_contour_mean"] = float(int_vals.mean())
        metrics["boundary_alignment_ratio"] = float(bnd_vals.mean() / max(int_vals.mean(), 1e-9))
    else:
        metrics["boundary_contour_mean"] = 0.0
        metrics["interior_contour_mean"] = 0.0
        metrics["boundary_alignment_ratio"] = 0.0

    return metrics


def _make_comparison_figure(
    pred: np.ndarray,
    gt: np.ndarray,
    contours: np.ndarray,
    out_path: Path,
) -> None:
    T = pred.shape[0]
    mid = T // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(contours[mid], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Contour map (frame {mid})")
    axes[1].imshow(pred[mid], cmap="tab20", interpolation="nearest")
    axes[1].set_title(f"Predicted labels (frame {mid})")
    axes[2].imshow(gt[mid], cmap="tab20", interpolation="nearest")
    axes[2].set_title(f"GT labels (frame {mid})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_flow_field(pos_dir: Path, crop: list[int] | None, flow_field_path: Path | None = None) -> np.ndarray:
    path = flow_field_path or (pos_dir / "3_cell/filtered_dp.tif")
    flow_field = np.asarray(tifffile.imread(path), dtype=np.float32)
    if flow_field.ndim == 5 and flow_field.shape[2] == 2:
        # Raw Cellpose cell_dp_3dt.tif: (T, Z, 2, Y, X). Project Z without
        # applying the downstream median/Gaussian filtering used for filtered_dp.
        flow_field = flow_field.mean(axis=1)
    if flow_field.ndim == 4 and flow_field.shape[1] == 1:
        flow_field = flow_field[:, 0]
    if flow_field.ndim == 4 and flow_field.shape[1] == 2:
        pass  # (T, 2, Y, X) as expected
    elif flow_field.ndim == 4 and flow_field.shape[3] == 2:
        flow_field = flow_field.transpose(0, 3, 1, 2)  # (T,Y,X,2) → (T,2,Y,X)
    else:
        raise ValueError(f"Expected flow field as (T,2,Y,X) or (T,Y,X,2), got {flow_field.shape}")
    if crop is not None:
        t0c, t1c, y0c, y1c, x0c, x1c = crop
        flow_field = flow_field[t0c:t1c, :, y0c:y1c, x0c:x1c]
    return flow_field


def main() -> None:
    args = _parse_args()
    output_dir = args.pos_dir / "4_cell_graphcut" / args.timestamp
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite or use a new --timestamp")
    output_dir.mkdir(parents=True, exist_ok=True)

    params: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "pos_dir": str(args.pos_dir),
        "unary_mode": args.unary_mode,
        "init_mode": args.init_mode,
        "alpha_unary": args.alpha_unary,
        "gamma_unary": args.gamma_unary,
        "lambda_s": args.lambda_s,
        "beta_s": args.beta_s,
        "lambda_t": args.lambda_t,
        "lambda_area": args.lambda_area,
        "boundary_mode": args.boundary_mode,
        "foreground_score_path": str(args.foreground_score_path) if args.foreground_score_path else None,
        "flow_field_path": str(args.flow_field_path) if args.flow_field_path else None,
        "lambda_geodesic": args.lambda_geodesic,
        "lambda_flow": args.lambda_flow,
        "solver": args.solver,
        "lambda_contour": args.lambda_contour,
        "min_round_flips": args.min_round_flips,
        "preserve_connectivity": args.preserve_connectivity,
        "n_iters": args.n_iters,
        "n_workers": args.n_workers,
        "crop": args.crop,
        "INF": _INF,
        "graph_mode": "foreground_only_edge_list",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "params.json", params)

    print("Loading inputs...", flush=True)
    nuc_tracks, fg_mask, contours, gt_labels = _load_inputs(args.pos_dir, args.crop)
    T, Y, X = fg_mask.shape
    params["shape"] = (T, Y, X)
    print(f"  Shape: {T}×{Y}×{X}, fg_voxels={np.count_nonzero(fg_mask)}", flush=True)

    label_ids = np.array(sorted(int(k) for k in np.unique(nuc_tracks) if k > 0), dtype=np.uint32)
    print(f"  Global label set: {len(label_ids)} track IDs: {label_ids.tolist()}", flush=True)
    params["label_ids"] = label_ids.tolist()

    foreground_scores = None
    foreground_score_path = args.foreground_score_path
    needs_fg = (
        args.boundary_mode in ("foreground_inverse", "combined")
        or args.gamma_unary != 0.0
        or args.init_mode == "watershed_geodesic"
    )
    if needs_fg:
        foreground_score_path = foreground_score_path or (args.pos_dir / "3_cell" / "foreground_scores.tif")
        print(f"  Loading foreground scores: {foreground_score_path}", flush=True)
        foreground_scores = _load_foreground_scores(foreground_score_path, args.crop)
        if foreground_scores.shape != contours.shape:
            raise ValueError(
                f"foreground_scores shape {foreground_scores.shape} does not match "
                f"input shape {contours.shape}"
            )
        params["foreground_score_path"] = str(foreground_score_path)

    print("\nComputing pairwise weights...", flush=True)
    boundary_signal = _prepare_boundary_signal(contours, foreground_scores, args.boundary_mode)
    if args.lambda_contour > 0.0:
        from scipy.ndimage import median_filter
        print(f"  Blending median-filtered contours into boundary signal "
              f"(lambda_contour={args.lambda_contour})...", flush=True)
        contours_smooth = np.stack(
            [median_filter(contours[t], size=2) for t in range(T)],
            axis=0,
        ).astype(np.float32)
        boundary_signal = boundary_signal + args.lambda_contour * contours_smooth
    h, v, dr, dl, tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l = _compute_pairwise_weights(
        fg_mask, boundary_signal, args.lambda_s, args.beta_s, args.lambda_t
    )

    if args.solver == "icm":
        # ── Numba ICM path ──────────────────────────────────────────────────
        cache_dir = None
        if not args.no_unary_cache:
            cache_dir = args.unary_cache_dir or (args.pos_dir / "4_cell_graphcut" / "unary_cache")
        params["unary_cache_dir"] = str(cache_dir) if cache_dir is not None else None

        geodesic_cache_mode = (
            "geodesic" if args.gamma_unary == 0.0
            else f"geodesic_g{args.gamma_unary:g}"
        )

        print(f"\nComputing {args.unary_mode} unaries...", flush=True)
        t0 = perf_counter()
        if args.unary_mode == "geodesic":
            unary = _get_cached_unaries(
                cache_dir, geodesic_cache_mode, (T, Y, X), args.crop, args.alpha_unary,
                lambda: _compute_geodesic_unaries(
                    nuc_tracks, fg_mask, contours, label_ids, args.alpha_unary,
                    foreground_scores=foreground_scores, gamma_unary=args.gamma_unary,
                ),
            )
        elif args.unary_mode == "costfield":
            unary = _compute_costfield_unaries(
                nuc_tracks, fg_mask, contours, label_ids, args.alpha_unary,
                foreground_scores=foreground_scores, gamma_unary=args.gamma_unary,
            )
        elif args.unary_mode == "euclidean":
            unary = _get_cached_unaries(
                cache_dir, "euclidean", (T, Y, X), args.crop, 0.0,
                lambda: _compute_euclidean_unaries(nuc_tracks, fg_mask, label_ids),
            )
        elif args.unary_mode == "flow":
            flow_cache_mode = "flow_raw_cellpose" if args.flow_field_path else "flow"
            unary = _get_cached_unaries(
                cache_dir, flow_cache_mode, (T, Y, X), args.crop, 0.0,
                lambda: _compute_flow_unaries(
                    nuc_tracks, fg_mask,
                    _load_flow_field(args.pos_dir, args.crop, args.flow_field_path),
                    label_ids,
                ),
            )
        else:  # geodesic_flow
            geodesic_unary = None
            if args.lambda_geodesic != 0.0:
                geodesic_unary = _get_cached_unaries(
                    cache_dir, geodesic_cache_mode, (T, Y, X), args.crop, args.alpha_unary,
                    lambda: _compute_geodesic_unaries(
                        nuc_tracks, fg_mask, contours, label_ids, args.alpha_unary,
                        foreground_scores=foreground_scores, gamma_unary=args.gamma_unary,
                    ),
                )
                _apply_nucleus_anchors(geodesic_unary, nuc_tracks, label_ids)
            flow_cache_mode = "flow_raw_cellpose" if args.flow_field_path else "flow"
            flow_unary = _get_cached_unaries(
                cache_dir, flow_cache_mode, (T, Y, X), args.crop, 0.0,
                lambda: _compute_flow_unaries(
                    nuc_tracks, fg_mask,
                    _load_flow_field(args.pos_dir, args.crop, args.flow_field_path),
                    label_ids,
                ),
            )
            _apply_nucleus_anchors(flow_unary, nuc_tracks, label_ids)
            base_unary = geodesic_unary if geodesic_unary is not None else flow_unary
            base_weight = args.lambda_geodesic if geodesic_unary is not None else 0.0
            unary = _combine_unaries(
                base_unary, flow_unary, lambda_geodesic=base_weight, lambda_flow=args.lambda_flow,
            )
        _apply_nucleus_anchors(unary, nuc_tracks, label_ids)
        t_unary = perf_counter() - t0
        print(f"  Done in {t_unary:.1f}s, {len(unary)} (frame, label) pairs.", flush=True)

        print("\nConverting unary to dense array...", flush=True)
        unary_dense = _dict_to_dense_unary(unary, fg_mask, label_ids)
        del unary

        if args.init_mode == "nuclei":
            print("\nInitializing labels from nucleus pixels only...", flush=True)
            init_labels = np.zeros(fg_mask.shape, dtype=np.uint32)
        elif args.init_mode == "unary":
            init_labels = None  # _run_icm uses unary argmin when init_labels is None
        else:
            print("\nInitializing labels via seeded watershed...", flush=True)
            init_labels = _watershed_init(fg_mask, nuc_tracks, foreground_scores, contours)

        print("\nRunning ICM...", flush=True)
        t0 = perf_counter()
        pred_labels, energy_log = _run_icm(
            fg_mask, unary_dense, label_ids, h, v, dr, dl, tw,
            tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l, nuc_tracks,
            n_iters=args.n_iters,
            min_round_flips=args.min_round_flips,
            init_labels=init_labels,
            lambda_area=args.lambda_area,
        )
        t_solver = perf_counter() - t0
        print(f"\nICM done in {t_solver:.1f}s.", flush=True)
        solver_label = f"icm/{args.unary_mode}"

    else:
        # ── Graph-cut path (original) ────────────────────────────────────────
        cache_dir = None
        if not args.no_unary_cache:
            cache_dir = args.unary_cache_dir or (args.pos_dir / "4_cell_graphcut" / "unary_cache")
        params["unary_cache_dir"] = str(cache_dir) if cache_dir is not None else None

        geodesic_cache_mode = (
            "geodesic" if args.gamma_unary == 0.0
            else f"geodesic_g{args.gamma_unary:g}"
        )
        print(f"\nComputing {args.unary_mode} unaries...", flush=True)
        t0 = perf_counter()
        if args.unary_mode == "geodesic":
            unary = _get_cached_unaries(
                cache_dir, geodesic_cache_mode, (T, Y, X), args.crop, args.alpha_unary,
                lambda: _compute_geodesic_unaries(
                    nuc_tracks, fg_mask, contours, label_ids, args.alpha_unary,
                    foreground_scores=foreground_scores, gamma_unary=args.gamma_unary,
                ),
            )
        elif args.unary_mode == "costfield":
            unary = _compute_costfield_unaries(
                nuc_tracks, fg_mask, contours, label_ids, args.alpha_unary,
                foreground_scores=foreground_scores, gamma_unary=args.gamma_unary,
            )
        elif args.unary_mode == "euclidean":
            unary = _get_cached_unaries(
                cache_dir, "euclidean", (T, Y, X), args.crop, 0.0,
                lambda: _compute_euclidean_unaries(nuc_tracks, fg_mask, label_ids),
            )
        elif args.unary_mode == "flow":
            flow_cache_mode = "flow_raw_cellpose" if args.flow_field_path else "flow"
            unary = _get_cached_unaries(
                cache_dir, flow_cache_mode, (T, Y, X), args.crop, 0.0,
                lambda: _compute_flow_unaries(
                    nuc_tracks, fg_mask, _load_flow_field(args.pos_dir, args.crop, args.flow_field_path), label_ids
                ),
            )
        else:  # geodesic_flow
            geodesic_unary = None
            if args.lambda_geodesic != 0.0 or args.init_mode == "geodesic":
                geodesic_unary = _get_cached_unaries(
                    cache_dir, geodesic_cache_mode, (T, Y, X), args.crop, args.alpha_unary,
                    lambda: _compute_geodesic_unaries(
                        nuc_tracks, fg_mask, contours, label_ids, args.alpha_unary,
                        foreground_scores=foreground_scores, gamma_unary=args.gamma_unary,
                    ),
                )
                _apply_nucleus_anchors(geodesic_unary, nuc_tracks, label_ids)
            flow_cache_mode = "flow_raw_cellpose" if args.flow_field_path else "flow"
            flow_unary = _get_cached_unaries(
                cache_dir, flow_cache_mode, (T, Y, X), args.crop, 0.0,
                lambda: _compute_flow_unaries(
                    nuc_tracks, fg_mask, _load_flow_field(args.pos_dir, args.crop, args.flow_field_path), label_ids
                ),
            )
            _apply_nucleus_anchors(flow_unary, nuc_tracks, label_ids)
            base_unary = geodesic_unary if geodesic_unary is not None else flow_unary
            base_weight = args.lambda_geodesic if geodesic_unary is not None else 0.0
            unary = _combine_unaries(
                base_unary, flow_unary,
                lambda_geodesic=base_weight, lambda_flow=args.lambda_flow,
            )
        # Re-apply hard nucleus anchors against the current nuc/fg state so a
        # stale unary cache (e.g. computed before the cell foreground mask was
        # rebuilt) cannot mislabel nucleus pixels at init.
        _apply_nucleus_anchors(unary, nuc_tracks, label_ids)
        t_unary = perf_counter() - t0
        print(f"  Done in {t_unary:.1f}s, {len(unary)} (frame, label) pairs.", flush=True)

        if args.save_unary_images:
            print("\nSaving unary images...", flush=True)
            # Cost field: 1 + alpha_unary * contour [+ gamma_unary * (1 - fg)]
            # Only meaningful for geodesic/geodesic_flow modes; pass None otherwise.
            cost_field_img: np.ndarray | None = None
            if args.unary_mode in ("geodesic", "geodesic_flow", "costfield"):
                cost_field_img = (
                    1.0 + args.alpha_unary * np.clip(contours, 0.0, 1.0)
                ).astype(np.float32)
                if args.gamma_unary != 0.0 and foreground_scores is not None:
                    cost_field_img += args.gamma_unary * (1.0 - np.clip(foreground_scores, 0.0, 1.0))
            _save_unary_images(unary, fg_mask, label_ids, output_dir, cost_field=cost_field_img)

        init_unary = None
        init_labels_ws = None
        if args.init_mode == "euclidean" and args.unary_mode != "euclidean":
            print("\nComputing euclidean unaries for initialization...", flush=True)
            init_unary = _get_cached_unaries(
                cache_dir, "euclidean", (T, Y, X), args.crop, 0.0,
                lambda: _compute_euclidean_unaries(nuc_tracks, fg_mask, label_ids),
            )
            _apply_nucleus_anchors(init_unary, nuc_tracks, label_ids)
        elif args.init_mode == "geodesic" and args.unary_mode != "geodesic":
            if args.unary_mode == "geodesic_flow" and geodesic_unary is not None:
                init_unary = geodesic_unary
            else:
                print("\nComputing geodesic unaries for initialization...", flush=True)
                init_unary = _get_cached_unaries(
                    cache_dir, geodesic_cache_mode, (T, Y, X), args.crop, args.alpha_unary,
                    lambda: _compute_geodesic_unaries(
                        nuc_tracks, fg_mask, contours, label_ids, args.alpha_unary,
                        foreground_scores=foreground_scores, gamma_unary=args.gamma_unary,
                    ),
                )
                _apply_nucleus_anchors(init_unary, nuc_tracks, label_ids)
        elif args.init_mode == "watershed_flow":
            print("\nBuilding flow elevation for watershed init...", flush=True)
            elevation = _unary_elevation(unary, fg_mask, label_ids)
            print("  Running seeded watershed on flow elevation...", flush=True)
            init_labels_ws = _watershed_init(fg_mask, nuc_tracks, elevation=elevation)
        elif args.init_mode == "watershed_geodesic":
            print("\nBuilding geodesic elevation for watershed init...", flush=True)
            if args.unary_mode == "geodesic":
                geo_unary_for_ws = unary
            elif args.unary_mode == "geodesic_flow" and geodesic_unary is not None:
                geo_unary_for_ws = geodesic_unary
            else:
                print("  Computing geodesic unaries for watershed elevation...", flush=True)
                geo_unary_for_ws = _get_cached_unaries(
                    cache_dir, geodesic_cache_mode, (T, Y, X), args.crop, args.alpha_unary,
                    lambda: _compute_geodesic_unaries(
                        nuc_tracks, fg_mask, contours, label_ids, args.alpha_unary,
                        foreground_scores=foreground_scores, gamma_unary=args.gamma_unary,
                    ),
                )
            elevation = _unary_elevation(geo_unary_for_ws, fg_mask, label_ids)
            print("  Running seeded watershed on geodesic elevation...", flush=True)
            init_labels_ws = _watershed_init(fg_mask, nuc_tracks, elevation=elevation)

        print("\nRunning alpha-expansion...", flush=True)
        t0 = perf_counter()
        pred_labels, energy_log = _run_alpha_expansion(
            fg_mask, unary, label_ids, h, v, dr, dl, tw, args.n_iters,
            nuc_tracks=nuc_tracks,
            init_unary=init_unary,
            init_labels=init_labels_ws,
            nuclei_only_init=(args.init_mode == "nuclei"),
            min_round_flips=args.min_round_flips,
            n_workers=args.n_workers,
            preserve_connectivity=args.preserve_connectivity,
        )
        t_solver = perf_counter() - t0
        print(f"\nAlpha-expansion done in {t_solver:.1f}s.", flush=True)
        solver_label = f"graphcut/{args.unary_mode}"

    print("\nSaving cell_labels.tif...", flush=True)
    tifffile.imwrite(
        output_dir / "cell_labels.tif",
        pred_labels.astype(np.uint16),
        compression="zlib",
    )

    if gt_labels is not None:
        print("Evaluating...", flush=True)
        metrics = _evaluate(pred_labels, gt_labels, fg_mask, contours, label_ids)
    else:
        print("Skipping evaluation: no 3_cell/tracked_labels.tif ground truth found.", flush=True)
        metrics = {}
    metrics["unary_time_s"] = round(t_unary, 1)
    metrics["solver_time_s"] = round(t_solver, 1)
    metrics["total_time_s"] = round(t_unary + t_solver, 1)
    metrics["solver"] = solver_label
    metrics["energy_log"] = energy_log
    _write_json(output_dir / "metrics.json", metrics)

    if gt_labels is not None:
        print("Making comparison figure...", flush=True)
        _make_comparison_figure(pred_labels, gt_labels, contours, output_dir / "comparison_frame.png")

    params["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(output_dir / "params.json", params)

    print("\n=== Summary ===")
    print(f"  Output: {output_dir}")
    if gt_labels is not None:
        print(f"  Coverage: {metrics['coverage']:.3f}")
        print(f"  Mean temporal IoU: {metrics['mean_temporal_iou']:.3f}")
        print(f"  Median temporal IoU: {metrics['median_temporal_iou']:.3f}")
        print(f"  Purity: {metrics['mean_purity']:.3f}  Completeness: {metrics['mean_completeness']:.3f}")
        print(f"  Pred flicker: {metrics['pred_flicker_rate']:.4f}  GT flicker: {metrics['gt_flicker_rate']:.4f}")
        print(f"  Boundary alignment ratio: {metrics['boundary_alignment_ratio']:.3f}")
    print(f"  Solver: {solver_label}  Unary: {t_unary:.1f}s  Solver: {t_solver:.1f}s")


if __name__ == "__main__":
    main()
"""Unified path resolution for the CellFlow pipeline directory layout."""
from __future__ import annotations

from pathlib import Path

# Authoritative stage-name → output-directory mapping.
STAGE_DIRS: dict[str, str] = {
    "raw_import": "0_input",
    "cellpose": "1_cellpose",
    "nucleus": "2_nucleus",
    "cell": "3_cell",
    "analysis": "4_analysis",
}


def pos_dir(root: Path | str, pos: int) -> Path:
    """Return <root>/pos<pos:02d>."""
    return Path(root) / f"pos{pos:02d}"


def stage_dir(root: Path | str, pos: int, stage: str) -> Path:
    """Return the output directory for *stage* at position *pos*."""
    dirname = STAGE_DIRS.get(stage, stage)
    return pos_dir(root, pos) / dirname


def log_path(root: Path | str, pos: int) -> Path:
    """Return the path to the pipeline log for a position."""
    return pos_dir(root, pos) / "pipeline.log"
from cellflow.correction.labels import (
    apply_gamma,
    clean_stranded_pixels,
    draw_cell_path,
    erase_cell,
    expand_label_to_foreground,
    fill_label_holes,
    fix_label_semiholes,
    merge_cells,
    split_across,
    split_draw,
    swap_labels,
)

__all__ = [
    "apply_gamma",
    "clean_stranded_pixels",
    "draw_cell_path",
    "erase_cell",
    "expand_label_to_foreground",
    "fill_label_holes",
    "fix_label_semiholes",
    "merge_cells",
    "split_across",
    "split_draw",
    "swap_labels",
]
"""Label correction operations on a single (H, W) segmentation frame.

All functions accept a 2-D ``seg`` array and modify it **in-place**.
They return ``True`` on success and ``False`` when the operation is
rejected (e.g. labels don't touch, result too small, background click).
"""
from __future__ import annotations

import logging
import os

import numpy as np
from scipy.ndimage import binary_dilation, binary_closing, binary_fill_holes, label as nd_label
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon as draw_polygon
from skimage.morphology import disk
from skimage.segmentation import watershed, expand_labels

log = logging.getLogger("cellflow.correction")
if os.environ.get("CELLFLOW_DEBUG"):
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[cellflow.correction] %(levelname)s %(message)s"))
        log.addHandler(_h)

MIN_CELL_SIZE: int = 4


# ── bounding-box helpers ──────────────────────────────────────────────────────

def _bbox_of_label(seg: np.ndarray, lab: int) -> tuple[int, int, int, int]:
    rows, cols = np.where(seg == lab)
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _bbox_of_two(seg: np.ndarray, la: int, lb: int) -> tuple[int, int, int, int]:
    rows, cols = np.where(np.isin(seg, [la, lb]))
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _extend_bbox(
    bbox: tuple[int, int, int, int],
    factor: float,
    shape: tuple[int, int],
    min_pad: int = 0,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    dr = max(int((r1 - r0) * (factor - 1) / 2), min_pad)
    dc = max(int((c1 - c0) * (factor - 1) / 2), min_pad)
    return (
        max(0, r0 - dr), max(0, c0 - dc),
        min(shape[0], r1 + dr), min(shape[1], c1 + dc),
    )


def _crop(arr: np.ndarray, bbox: tuple) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0:r1, c0:c1]


def _to_local(pts: list, bbox: tuple) -> list[tuple[float, float]]:
    r0, c0 = bbox[0], bbox[1]
    return [(float(p[-2]) - r0, float(p[-1]) - c0) for p in pts]


# ── line drawing ──────────────────────────────────────────────────────────────

def _interpolate(pts: list[tuple[float, float]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in range(len(pts) - 1):
        r0, c0 = pts[i]
        r1, c1 = pts[i + 1]
        n = max(abs(int(r1) - int(r0)), abs(int(c1) - int(c0)), 1)
        for t in np.linspace(0, 1, n + 1):
            out.append((int(round(r0 + t * (r1 - r0))), int(round(c0 + t * (c1 - c0)))))
    seen: set = set()
    result = []
    for p in out:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _draw_line(shape: tuple[int, int], pts: list[tuple[int, int]]) -> np.ndarray:
    line = np.zeros(shape, dtype=np.uint8)
    for r, c in pts:
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            line[r, c] = 1
    return line


# ── misc helpers ──────────────────────────────────────────────────────────────

def _free_label(seg: np.ndarray) -> int:
    return int(seg.max()) + 1


def _touches(seg: np.ndarray, la: int, lb: int) -> bool:
    dilated_a = binary_dilation(seg == la, disk(1))
    dilated_b = binary_dilation(seg == lb, disk(1))
    return bool(np.any(dilated_a & dilated_b))


def _label_at(seg: np.ndarray, pos: tuple) -> int:
    r, c = int(round(float(pos[-2]))), int(round(float(pos[-1])))
    r = max(0, min(r, seg.shape[0] - 1))
    c = max(0, min(c, seg.shape[1] - 1))
    return int(seg[r, c])


def frame_view_2d(arr: np.ndarray, t: int) -> np.ndarray | None:
    """Return a 2D frame view from a time-indexed label stack."""
    if arr.ndim < 3 or t < 0 or t >= arr.shape[0]:
        return None
    view = arr[t]
    while view.ndim > 2:
        if view.shape[0] != 1:
            return None
        view = view[0]
    return view


def best_overlapping_label(
    target_labels: np.ndarray,
    source_labels: np.ndarray,
    t: int,
    source_label: int,
) -> int:
    """Return the non-zero target label with most overlap against source_label."""
    if source_label == 0:
        return 0
    target_frame = frame_view_2d(target_labels, t)
    source_frame = frame_view_2d(source_labels, t)
    if target_frame is None or source_frame is None or target_frame.shape != source_frame.shape:
        return 0
    source_mask = source_frame == int(source_label)
    if not np.any(source_mask):
        return 0
    overlap_values, counts = np.unique(target_frame[source_mask], return_counts=True)
    best_label = 0
    best_count = 0
    for label, count in zip(overlap_values, counts, strict=True):
        label = int(label)
        if label != 0 and int(count) > best_count:
            best_label = label
            best_count = int(count)
    return best_label


# ── public operations ─────────────────────────────────────────────────────────

def expand_label_to_foreground(
    seg: np.ndarray,
    foreground: np.ndarray,
    label: int,
    *,
    max_distance: int,
) -> int:
    """Expand ``label`` into connected foreground background pixels in-place.

    Returns the number of newly labelled pixels. A ``max_distance`` of 0 means
    no distance cap.
    """
    if foreground.shape != seg.shape:
        raise ValueError("foreground and seg must have the same shape")

    label = int(label)
    if label == 0:
        return 0
    seed = seg == label
    if not np.any(seed):
        return 0

    allowed = (foreground > 0) & ((seg == 0) | seed)
    component_labels, _num_components = nd_label(
        allowed,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    touching_ids = np.unique(component_labels[seed])
    touching_ids = touching_ids[touching_ids != 0]
    if touching_ids.size == 0:
        return 0

    touching_component = np.isin(component_labels, touching_ids)
    if max_distance > 0:
        dist = distance_transform_edt(~seed)
        touching_component &= dist <= int(max_distance)

    added = touching_component & (seg == 0)
    n_added = int(np.count_nonzero(added))
    if n_added:
        seg[added] = label
    return n_added

def erase_cell(seg: np.ndarray, pos: tuple | None = None, *, label: int | None = None) -> bool:
    """Set all pixels of the label under *pos* (or *label*) to 0."""
    if label is None:
        if pos is None:
            return False
        label = _label_at(seg, pos)
    log.debug("erase_cell: label=%s pos=%s", label, pos)
    if label == 0:
        return False
    seg[seg == label] = 0
    return True


def merge_cells(
    seg: np.ndarray,
    pos_start: tuple,
    pos_end: tuple,
    *,
    label_a: int | None = None,
    label_b: int | None = None,
) -> bool:
    """Merge the cell at *pos_start* into the cell at *pos_end*."""
    la = label_a if label_a is not None else _label_at(seg, pos_start)
    lb = label_b if label_b is not None else _label_at(seg, pos_end)
    log.debug("merge_cells: la=%s lb=%s", la, lb)
    if la == 0 or lb == 0 or la == lb:
        return False
    if not _touches(seg, la, lb):
        return False

    bbox = _bbox_of_two(seg, la, lb)
    bbox = _extend_bbox(bbox, 1.25, seg.shape)
    r0, c0, r1, c1 = bbox
    crop = _crop(seg, bbox)

    combined = np.isin(crop, [la, lb])
    closed = binary_closing(combined, disk(2))
    other_cells = (crop != 0) & ~combined
    closed = closed & ~other_cells
    seg[r0:r1, c0:c1][closed] = lb

    remaining_la = seg == la
    if remaining_la.any():
        seg[remaining_la] = lb

    clean_stranded_pixels(seg)
    return True


def split_across(
    seg: np.ndarray,
    img: np.ndarray | None,
    pos_start: tuple,
    pos_end: tuple,
    *,
    new_label: int | None = None,
) -> bool:
    """Watershed-split the cell under *pos_start* using two seeds."""
    la = _label_at(seg, pos_start)
    lb = _label_at(seg, pos_end)
    log.debug("split_across: la=%s lb=%s", la, lb)
    if la == 0 or la != lb:
        return False

    bbox = _bbox_of_label(seg, la)
    bbox = _extend_bbox(bbox, 1.25, seg.shape)
    r0, c0, r1, c1 = bbox
    crop_seg = _crop(seg, bbox)
    mask = (crop_seg == la).astype(np.uint8)
    interior = mask.astype(bool)

    rs = max(0, min(int(round(float(pos_start[-2]))) - r0, mask.shape[0] - 1))
    cs = max(0, min(int(round(float(pos_start[-1]))) - c0, mask.shape[1] - 1))
    re = max(0, min(int(round(float(pos_end[-2]))) - r0, mask.shape[0] - 1))
    ce = max(0, min(int(round(float(pos_end[-1]))) - c0, mask.shape[1] - 1))

    new_lab = int(new_label) if new_label is not None else _free_label(seg)

    for radius in range(7):
        markers = np.zeros(mask.shape, dtype=np.int32)
        if radius == 0:
            markers[rs, cs] = la
            markers[re, ce] = new_lab
        else:
            d = disk(radius)
            seed_a = np.zeros(mask.shape, dtype=bool)
            seed_a[rs, cs] = True
            seed_b = np.zeros(mask.shape, dtype=bool)
            seed_b[re, ce] = True
            markers[binary_dilation(seed_a, d) & interior] = la
            markers[binary_dilation(seed_b, d) & interior] = new_lab

        if img is not None:
            crop_img = _crop(img, bbox)
            ws = watershed(crop_img, markers=markers, mask=mask)
        else:
            dist = distance_transform_edt(mask)
            ws = watershed(-dist, markers=markers, mask=mask)

        size_a = int(np.sum(ws == la))
        size_b = int(np.sum(ws == new_lab))
        if size_a >= MIN_CELL_SIZE and size_b >= MIN_CELL_SIZE:
            seg[r0:r1, c0:c1][ws == new_lab] = new_lab
            return True

    return False


def split_draw(
    seg: np.ndarray,
    positions: list,
    *,
    curlabel: int | None = None,
    new_label: int | None = None,
) -> bool:
    """Split a cell along a manually drawn line."""
    log.debug("split_draw: %d raw positions, curlabel=%s", len(positions), curlabel)
    if curlabel is None or curlabel == 0 or not np.any(seg == curlabel):
        return False

    bbox = _bbox_of_label(seg, curlabel)
    bbox = _extend_bbox(bbox, 1.25, seg.shape)
    crop = _crop(seg, bbox).copy()
    local_pts = _to_local(positions, bbox)

    in_cell_indices = [
        i for i, p in enumerate(local_pts)
        if 0 <= int(round(p[0])) < crop.shape[0]
        and 0 <= int(round(p[1])) < crop.shape[1]
        and crop[int(round(p[0])), int(round(p[1]))] == curlabel
    ]
    if len(in_cell_indices) < 2:
        return False

    first_idx, last_idx = in_cell_indices[0], in_cell_indices[-1]
    in_cell = [local_pts[i] for i in in_cell_indices]

    ext_start = local_pts[first_idx - 1] if first_idx > 0 else in_cell[0]
    ext_end   = local_pts[last_idx + 1]  if last_idx < len(local_pts) - 1 else in_cell[-1]

    all_pts = [ext_start] + in_cell + [ext_end]
    interp = _interpolate(all_pts)
    line = _draw_line(crop.shape, interp)

    if int(np.sum(line & (crop == curlabel))) == 0:
        return False

    return _split_in_crop(seg, crop, line, bbox, curlabel, new_label=new_label)


def _split_in_crop(
    seg: np.ndarray,
    crop: np.ndarray,
    line: np.ndarray,
    bbox: tuple,
    curlabel: int,
    retry: int = 0,
    *,
    new_label: int | None = None,
) -> bool:
    if retry > 6:
        return False

    dilated = binary_dilation(line, disk(retry)) if retry > 0 else line.astype(bool)
    mask = np.zeros(crop.shape, dtype=np.uint8)
    mask[crop == curlabel] = 1
    mask[dilated] = 0

    regions, n = nd_label(mask)
    sizes = [int(np.sum(regions == i)) for i in range(1, n + 1)]
    log.debug("_split_in_crop: retry=%d n_regions=%d sizes=%s", retry, n, sizes)

    if n >= 2:
        ids_by_size = sorted(range(1, n + 1), key=lambda i: sizes[i - 1], reverse=True)
        id_a, id_b = ids_by_size[0], ids_by_size[1]
        size_a, size_b = sizes[id_a - 1], sizes[id_b - 1]
        if size_a >= MIN_CELL_SIZE and size_b >= MIN_CELL_SIZE:
            regions_2 = np.zeros_like(regions)
            regions_2[regions == id_a] = 1
            regions_2[regions == id_b] = 2
            expanded = expand_labels(regions_2, distance=max(retry + 2, 3))
            r0, c0, r1, c1 = bbox
            new_lab = int(new_label) if new_label is not None else _free_label(seg)
            orig_cell = crop == curlabel
            seg[r0:r1, c0:c1][(expanded == 2) & orig_cell] = new_lab
            return True

    return _split_in_crop(seg, crop, line, bbox, curlabel, retry + 1, new_label=new_label)


def draw_cell_path(
    seg: np.ndarray,
    positions: list,
    *,
    curlabel: int | None = None,
    new_label: int | None = None,
) -> bool:
    """Draw a closed region from the user's stroke and fill its interior."""
    log.debug("draw_cell_path: %d raw positions, curlabel=%s", len(positions), curlabel)
    if len(positions) < 2:
        return False

    local_pts = [(float(p[-2]), float(p[-1])) for p in positions]

    rows = np.array([p[0] for p in local_pts])
    cols = np.array([p[1] for p in local_pts])
    rr, cc = draw_polygon(rows, cols, seg.shape)
    log.debug("draw_cell_path: polygon fill pixels=%d", len(rr))

    if len(rr) < MIN_CELL_SIZE:
        return False

    extending = bool(curlabel) and curlabel != 0 and np.any(seg == curlabel)
    label = curlabel if extending else (
        int(new_label) if new_label is not None else _free_label(seg)
    )

    fill_mask = np.zeros(seg.shape, dtype=bool)
    fill_mask[rr, cc] = True
    if extending:
        existing_mask = seg == label
        connected_regions, _ = nd_label(existing_mask | fill_mask)
        connected_ids = np.unique(connected_regions[existing_mask])
        fill_mask &= np.isin(connected_regions, connected_ids)
    else:
        fill_mask &= (seg == 0)

    n_px = int(np.sum(fill_mask))
    if n_px < MIN_CELL_SIZE:
        return False

    seg[fill_mask] = label
    if extending:
        cell_mask = seg == label
        filled_mask = binary_fill_holes(cell_mask)
        seg[filled_mask & ~cell_mask] = label
    return True


def swap_labels(seg: np.ndarray, pos_a: tuple, pos_b: tuple) -> bool:
    """Swap the label values at the two click positions across the whole frame."""
    la = _label_at(seg, pos_a)
    lb = _label_at(seg, pos_b)
    log.debug("swap_labels: la=%s lb=%s", la, lb)
    if la == 0 or lb == 0 or la == lb:
        return False
    mask_a = seg == la
    mask_b = seg == lb
    seg[mask_a] = lb
    seg[mask_b] = la
    return True


def relabel_cell(seg: np.ndarray, pos: tuple, new_label: int) -> bool:
    """Assign *new_label* to the cell at *pos* in *seg* (in-place).

    If *new_label* already exists in the frame, the two cells are swapped so
    no label is lost.  Returns ``False`` when *pos* hits background, already
    has *new_label*, or *new_label* is 0.
    """
    old_label = _label_at(seg, pos)
    if old_label == 0 or new_label == 0 or old_label == new_label:
        return False
    conflict = seg == new_label
    seg[seg == old_label] = new_label
    if np.any(conflict):
        seg[conflict] = old_label
    return True


def fill_label_holes(labels: np.ndarray, radius: int = 5) -> np.ndarray:
    """Fill enclosed background gaps by expanding neighboring labels.

    Background connected to the image border is preserved.  Enclosed zero-valued
    components are filled only as far as labels can expand within *radius*
    pixels; use a large radius to fill all enclosed gaps.
    """
    from skimage.measure import label as _cc_label

    if radius <= 0:
        return labels

    bg = labels == 0
    if not np.any(bg):
        return labels

    bg_labeled = _cc_label(bg, connectivity=2)
    open_ids: set[int] = set()
    for edge in (
        bg_labeled[0, :], bg_labeled[-1, :],
        bg_labeled[:, 0], bg_labeled[:, -1],
    ):
        open_ids.update(int(v) for v in np.unique(edge))
    open_ids.discard(0)

    open_bg = bg & np.isin(bg_labeled, list(open_ids))
    enclosed = bg & ~open_bg
    if not np.any(enclosed):
        return labels

    sentinel = int(np.max(labels)) + 1
    work = labels.copy()
    work[open_bg] = sentinel
    expanded = expand_labels(work, distance=int(radius))
    expanded[open_bg] = 0
    expanded[expanded == sentinel] = 0
    return expanded.astype(labels.dtype, copy=False)


def fix_label_semiholes(
    labels: np.ndarray,
    radius: int = 5,
    max_opening: int = 3,
) -> np.ndarray:
    """Fill narrow border-connected background gaps by expanding labels.

    Candidate zero-valued components must touch the image border with no more
    than ``max_opening`` pixels.  Wider border-connected background regions are
    preserved as open background.
    """
    from skimage.measure import label as _cc_label

    if radius <= 0 or max_opening <= 0:
        return labels

    bg = labels == 0
    if not np.any(bg):
        return labels

    bg_labeled = _cc_label(bg, connectivity=2)
    border_mask = np.zeros(labels.shape, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    candidate = np.zeros(labels.shape, dtype=bool)
    for comp_id in np.unique(bg_labeled[border_mask & bg]):
        comp_id = int(comp_id)
        if comp_id == 0:
            continue
        comp_mask = bg_labeled == comp_id
        opening = int(np.sum(comp_mask & border_mask))
        if opening <= int(max_opening):
            candidate |= comp_mask

    if not np.any(candidate):
        return labels

    sentinel = int(np.max(labels)) + 1
    work = labels.copy()
    work[bg & ~candidate] = sentinel
    expanded = expand_labels(work, distance=int(radius))
    expanded[bg & ~candidate] = 0
    expanded[expanded == sentinel] = 0
    return expanded.astype(labels.dtype, copy=False)


def clean_stranded_pixels(seg: np.ndarray, min_size: int = MIN_CELL_SIZE) -> int:
    """Remove disconnected same-label fragments, keeping each label's largest component."""
    from skimage.measure import label as _cc_label
    cleared = 0

    for cell_id in np.unique(seg):
        if cell_id == 0:
            continue
        mask = seg == cell_id
        labeled, n_comp = _cc_label(mask, return_num=True, connectivity=2)
        if n_comp <= 1:
            continue
        comp_sizes = {cid: int(np.sum(labeled == cid)) for cid in range(1, n_comp + 1)}
        largest = max(comp_sizes, key=comp_sizes.__getitem__)
        for comp_id, n_px in comp_sizes.items():
            if comp_id == largest:
                continue
            comp_mask = labeled == comp_id
            seg[comp_mask] = 0
            filled = expand_labels(seg, distance=n_px + 2)
            seg[comp_mask] = filled[comp_mask]
            cleared += n_px

    return cleared


from cellflow.segmentation import apply_gamma  # noqa: F401 — re-exported from here
"""TIFF storage for tracked nucleus label volumes.

Schema: single multipage TIFF — shape (T, Y, X), dtype uint32.
Frames that have not yet been tracked are stored as all-zeros.
A frame is considered "tracked" (exists) if it contains at least one non-zero label.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

_LABEL_DTYPE = np.uint32


def _load_stack(path: Path) -> np.ndarray:
    """Load the TIFF as (T, Y, X). Returns empty array if file does not exist.

    Tolerates legacy files written with a singleton Z axis: ``(T, 1, Y, X)``
    is squeezed to ``(T, Y, X)`` so the in-memory shape always matches the
    documented schema.
    """
    if not path.exists():
        return np.empty((0, 0, 0), dtype=_LABEL_DTYPE)
    stack = np.asarray(tifffile.imread(str(path)), dtype=_LABEL_DTYPE)
    if stack.ndim == 2:
        stack = stack[np.newaxis]
    elif stack.ndim == 4 and stack.shape[1] == 1:
        stack = stack[:, 0]
    return stack


def write_tracked_frame(path: str | Path, t: int, labels: np.ndarray) -> None:
    """Write a single tracked frame into tracked_labels.tif."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=_LABEL_DTYPE)
    if labels.ndim == 3:
        labels = labels.max(axis=0)  # (Z, Y, X) → (Y, X)
    H, W = labels.shape
    stack = _load_stack(path)
    if stack.size == 0:
        stack = np.zeros((t + 1, H, W), dtype=_LABEL_DTYPE)
    elif t >= stack.shape[0]:
        extra = np.zeros((t + 1 - stack.shape[0], H, W), dtype=_LABEL_DTYPE)
        stack = np.concatenate([stack, extra], axis=0)
    stack[t] = labels
    tifffile.imwrite(str(path), stack, compression="zlib")


def read_tracked_frame(path: str | Path, t: int) -> np.ndarray:
    """Read a single tracked frame, returned as (Y, X) uint32 array."""
    stack = _load_stack(Path(path))
    if t >= stack.shape[0]:
        raise KeyError(f"Frame t={t} not found in {path}")
    return stack[t]


def read_full_tracked_stack(path: str | Path) -> np.ndarray:
    """Read all tracked frames as a (T, Y, X) uint32 array."""
    return _load_stack(Path(path))


def tracked_n_frames(path: str | Path) -> int:
    """Return the number of timepoints written to tracked_labels.tif."""
    stack = _load_stack(Path(path))
    return stack.shape[0]


def tracked_frame_exists(path: str | Path, t: int) -> bool:
    """Return True if timepoint t has been written (contains non-zero labels)."""
    path = Path(path)
    if not path.exists():
        return False
    stack = _load_stack(path)
    return t < stack.shape[0] and bool(stack[t].any())
from cellflow.napari._napari_compat import patch_napari_layer_delegate

patch_napari_layer_delegate()

from cellflow.napari.main_widget import CellFlowMainWidget as CellFlowWidget

__all__ = ["CellFlowWidget"]
"""Track-conditioned cell boundary selection widget for CellFlow."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.napari.ui_style import (
    add_block_button_row,
    add_block_pair_row,
    block_grid,
    compact_spinbox,
    status_label,
)
from cellflow.segmentation import apply_gamma, build_consensus_boundary_2d

logger = logging.getLogger(__name__)

_CELL_LABELS_LAYER = "Cell Labels"
_TRACKED_CELL_LAYER = "Tracked: Cell"
_CELL_CONTOUR_LAYER = "Contour Map: Cell"
_CELL_FOREGROUND_SCORE_LAYER = "Foreground Score: Cell"
_CELL_FOREGROUND_LAYER = "Foreground Mask: Cell"
_GRAPHCUT_CELL_LABELS_LAYER = "cell_labels_graphcut"
_CONTOUR_SWEEP_WIDTH = 60


class CellBoundaryWorkflowWidget(QWidget):
    """Track-conditioned cell boundary selection workflow."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._contour_worker = None
        self._boundary_selection_worker = None
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        def _stage_files(
            group_label: str, entries: list[tuple[str, str]]
        ) -> PipelineFilesWidget:
            return PipelineFilesWidget(
                [(group_label, entries)], viewer=self.viewer
            )

        def _stage_status() -> QLabel:
            label = QLabel("")
            label.setWordWrap(True)
            label.setVisible(False)
            status_label(label)
            return label

        def _stage_progress() -> QProgressBar:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setVisible(False)
            return bar

        def _spin_width(widget, width=_CONTOUR_SWEEP_WIDTH):
            widget.setMinimumWidth(width)
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            return widget

        def _combo(items: list[str], current: str, tooltip: str) -> QComboBox:
            combo = QComboBox()
            combo.addItems(items)
            combo.setCurrentText(current)
            combo.setToolTip(tooltip)
            combo.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            return combo

        def _int_spin(
            minimum: int,
            maximum: int,
            value: int,
            tooltip: str,
        ) -> QSpinBox:
            spin = _spin_width(QSpinBox())
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            spin.setToolTip(tooltip)
            return spin

        def _float_spin(
            minimum: float,
            maximum: float,
            value: float,
            tooltip: str,
            *,
            decimals: int = 2,
            step: float = 0.1,
        ) -> QDoubleSpinBox:
            spin = _spin_width(QDoubleSpinBox())
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setToolTip(tooltip)
            return spin

        def _param_group_label(text: str) -> QLabel:
            label = QLabel(text)
            label.setStyleSheet("font-weight: 600;")
            return label

        self._icm_disabled_widgets: list[QWidget] = []

        # ---- 1. Contour Maps ----
        contour_inner = QWidget()
        contour_lay = QVBoxLayout(contour_inner)
        contour_lay.setContentsMargins(0, 0, 0, 0)
        contour_lay.setSpacing(4)
        contour_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.contour_input_files = _stage_files("Inputs", [
            ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
            ("3_cell/filtered_dp.tif", "Filtered flow vectors"),
        ])
        contour_lay.addWidget(self.contour_input_files)

        self.cp_min_spin = _spin_width(QDoubleSpinBox())
        self.cp_min_spin.setRange(-20.0, 20.0)
        self.cp_min_spin.setValue(-3.0)
        self.cp_min_spin.setDecimals(1)
        self.cp_min_spin.setSingleStep(1.0)
        self.cp_max_spin = _spin_width(QDoubleSpinBox())
        self.cp_max_spin.setRange(-20.0, 20.0)
        self.cp_max_spin.setValue(0.0)
        self.cp_max_spin.setDecimals(1)
        self.cp_max_spin.setSingleStep(1.0)
        self.cp_step_spin = _spin_width(QDoubleSpinBox())
        self.cp_step_spin.setRange(0.1, 10.0)
        self.cp_step_spin.setValue(1.0)
        self.cp_step_spin.setDecimals(1)
        self.cp_step_spin.setSingleStep(0.5)
        self.contour_flow_threshold_spin = _spin_width(QDoubleSpinBox())
        self.contour_flow_threshold_spin.setRange(0.0, 10.0)
        self.contour_flow_threshold_spin.setValue(0.0)
        self.contour_flow_threshold_spin.setDecimals(2)
        self.contour_flow_threshold_spin.setSingleStep(0.1)
        self.contour_niter_spin = _spin_width(QSpinBox())
        self.contour_niter_spin.setRange(0, 2000)
        self.contour_niter_spin.setValue(200)
        self.contour_niter_spin.setToolTip("Cellpose flow ODE integration steps.")

        contour_sweep_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(
            contour_sweep_grid,
            0,
            "Cellprob min:",
            compact_spinbox(self.cp_min_spin),
            "Cellprob max:",
            compact_spinbox(self.cp_max_spin),
        )
        add_block_pair_row(
            contour_sweep_grid,
            1,
            "Cellprob step:",
            compact_spinbox(self.cp_step_spin),
            "Flow threshold:",
            compact_spinbox(self.contour_flow_threshold_spin),
        )
        add_block_pair_row(
            contour_sweep_grid,
            2,
            "Niter:",
            compact_spinbox(self.contour_niter_spin),
        )
        contour_lay.addWidget(_param_group_label("Cellpose mask sweep"))
        contour_lay.addLayout(contour_sweep_grid)

        self.cp_gamma_min_spin = _spin_width(QDoubleSpinBox())
        self.cp_gamma_min_spin.setRange(0.05, 5.0)
        self.cp_gamma_min_spin.setValue(1.0)
        self.cp_gamma_min_spin.setDecimals(2)
        self.cp_gamma_min_spin.setSingleStep(0.05)
        self.cp_gamma_max_spin = _spin_width(QDoubleSpinBox())
        self.cp_gamma_max_spin.setRange(0.05, 5.0)
        self.cp_gamma_max_spin.setValue(1.0)
        self.cp_gamma_max_spin.setDecimals(2)
        self.cp_gamma_max_spin.setSingleStep(0.05)
        self.cp_gamma_step_spin = _spin_width(QDoubleSpinBox())
        self.cp_gamma_step_spin.setRange(0.05, 2.0)
        self.cp_gamma_step_spin.setValue(0.25)
        self.cp_gamma_step_spin.setDecimals(2)
        self.cp_gamma_step_spin.setSingleStep(0.05)
        gamma_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(
            gamma_grid,
            0,
            "Gamma min:",
            compact_spinbox(self.cp_gamma_min_spin),
            "Gamma max:",
            compact_spinbox(self.cp_gamma_max_spin),
        )
        add_block_pair_row(
            gamma_grid,
            1,
            "Gamma step:",
            compact_spinbox(self.cp_gamma_step_spin),
        )
        contour_lay.addWidget(_param_group_label("Gamma averaging"))
        contour_lay.addLayout(gamma_grid)

        self.contour_fg_threshold_spin = _spin_width(QDoubleSpinBox())
        self.contour_fg_threshold_spin.setRange(0.0, 1.0)
        self.contour_fg_threshold_spin.setValue(0.5)
        self.contour_fg_threshold_spin.setDecimals(2)
        self.contour_fg_threshold_spin.setSingleStep(0.01)
        fg_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(
            fg_grid,
            0,
            "FG threshold:",
            compact_spinbox(self.contour_fg_threshold_spin),
        )
        contour_lay.addWidget(_param_group_label("Foreground output"))
        contour_lay.addLayout(fg_grid)

        self.contour_output_files = _stage_files("Outputs", [
            ("3_cell/contour_maps.tif", "Contour maps"),
            ("3_cell/foreground_scores.tif", "Foreground scores"),
            ("3_cell/foreground_masks.tif", "Foreground masks"),
        ])
        contour_lay.addWidget(self.contour_output_files)

        contour_btn_row = block_grid(horizontal_spacing=12)
        self.preview_contour_btn = QPushButton("Preview")
        self.preview_contour_btn.setToolTip(
            "Build contour maps for the current frame only and display in napari"
        )
        self.preview_contour_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.build_contour_maps_btn = QPushButton("Build Contour Maps")
        self.build_contour_maps_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        add_block_pair_row(
            contour_btn_row,
            0,
            "",
            self.preview_contour_btn,
            "",
            self.build_contour_maps_btn,
        )
        contour_lay.addLayout(contour_btn_row)
        self.contour_status_lbl = _stage_status()
        contour_lay.addWidget(self.contour_status_lbl)
        self.contour_progress_bar = _stage_progress()
        contour_lay.addWidget(self.contour_progress_bar)

        self.contour_section = CollapsibleSection(
            "1. Contour Maps", contour_inner, expanded=False
        )
        layout.addWidget(self.contour_section)

        # ---- 2. Track-Conditioned Boundary Selection ----
        selection_inner = QWidget()
        selection_lay = QVBoxLayout(selection_inner)
        selection_lay.setContentsMargins(0, 0, 0, 0)
        selection_lay.setSpacing(4)
        selection_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.boundary_selection_input_files = _stage_files("Inputs", [
            ("2_nucleus/tracked_labels.tif", "Nucleus tracked labels"),
            ("3_cell/contour_maps.tif", "Contour maps"),
            ("3_cell/foreground_scores.tif", "Foreground scores"),
            ("3_cell/foreground_masks.tif", "Foreground masks"),
            ("1_cellpose/cell_dp_3dt.tif", "Raw Cellpose cell flow"),
        ])
        selection_lay.addWidget(self.boundary_selection_input_files)

        self.graphcut_solver_combo = _combo(
            ["graphcut", "icm"],
            "graphcut",
            "Solver used by the graphcut experiment script.",
        )
        self.graphcut_unary_mode_combo = _combo(
            ["flow", "geodesic_flow", "geodesic", "euclidean"],
            "flow",
            "Unary term. flow uses raw Cellpose flows from 1_cellpose/cell_dp_3dt.tif.",
        )
        self.graphcut_boundary_mode_combo = _combo(
            ["contour", "foreground_inverse"],
            "contour",
            "Boundary signal for pairwise costs.",
        )
        self.graphcut_n_iters_spin = _int_spin(
            1, 100, 1, "Number of solver iterations or alpha-expansion rounds."
        )
        self.graphcut_n_workers_spin = _int_spin(
            1, 128, 1, "Parallel worker processes for graphcut expansion moves."
        )

        core_grid = block_grid(horizontal_spacing=12)
        _, _, graphcut_unary_label, _ = add_block_pair_row(
            core_grid,
            0,
            "Solver:",
            self.graphcut_solver_combo,
            "Unary:",
            self.graphcut_unary_mode_combo,
            field_width=110,
        )
        boundary_label, _, _, _ = add_block_pair_row(
            core_grid,
            1,
            "Boundary:",
            self.graphcut_boundary_mode_combo,
            "Iters:",
            compact_spinbox(self.graphcut_n_iters_spin),
            field_width=110,
        )
        workers_label, _, _, _ = add_block_pair_row(
            core_grid,
            2,
            "Workers:",
            compact_spinbox(self.graphcut_n_workers_spin),
            field_width=110,
        )
        self._icm_disabled_widgets.extend([
            graphcut_unary_label,
            self.graphcut_unary_mode_combo,
            boundary_label,
            self.graphcut_boundary_mode_combo,
            workers_label,
            self.graphcut_n_workers_spin,
        ])
        selection_lay.addWidget(_param_group_label("Graphcut run"))
        selection_lay.addLayout(core_grid)

        advanced_inner = QWidget()
        advanced_lay = QVBoxLayout(advanced_inner)
        advanced_lay.setContentsMargins(0, 0, 0, 0)
        advanced_lay.setSpacing(4)

        self.graphcut_alpha_unary_spin = _float_spin(
            0.0, 1000.0, 4.0, "Contour weight in the geodesic unary cost field."
        )
        self.graphcut_lambda_geodesic_spin = _float_spin(
            0.0, 1000.0, 1.0, "Geodesic unary weight for geodesic_flow mode."
        )
        self.graphcut_lambda_flow_spin = _float_spin(
            0.0, 1000.0, 1.0, "Flow endpoint unary weight for geodesic_flow mode."
        )
        unary_grid = block_grid(horizontal_spacing=12)
        _, _, lambda_geodesic_label, _ = add_block_pair_row(
            unary_grid,
            0,
            "alpha_unary:",
            compact_spinbox(self.graphcut_alpha_unary_spin),
            "lambda_geodesic:",
            compact_spinbox(self.graphcut_lambda_geodesic_spin),
            field_width=92,
        )
        lambda_flow_label, _, _, _ = add_block_pair_row(
            unary_grid,
            1,
            "lambda_flow:",
            compact_spinbox(self.graphcut_lambda_flow_spin),
            field_width=92,
        )
        self._icm_disabled_widgets.extend([
            lambda_geodesic_label,
            self.graphcut_lambda_geodesic_spin,
            lambda_flow_label,
            self.graphcut_lambda_flow_spin,
        ])
        advanced_lay.addWidget(_param_group_label("Unary"))
        advanced_lay.addLayout(unary_grid)

        self.graphcut_lambda_s_spin = _float_spin(
            0.0, 1000.0, 1.0, "Spatial pairwise weight."
        )
        self.graphcut_beta_s_spin = _float_spin(
            0.0, 1000.0, 5.0, "Contour sensitivity for spatial pairwise costs."
        )
        self.graphcut_lambda_contour_spin = _float_spin(
            0.0, 1000.0, 0.0, "Extra contour-weighted pairwise term."
        )
        spatial_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(
            spatial_grid,
            0,
            "lambda_s:",
            compact_spinbox(self.graphcut_lambda_s_spin),
            "beta_s:",
            compact_spinbox(self.graphcut_beta_s_spin),
            field_width=92,
        )
        lambda_contour_label, _, _, _ = add_block_pair_row(
            spatial_grid,
            1,
            "lambda_contour:",
            compact_spinbox(self.graphcut_lambda_contour_spin),
            field_width=92,
        )
        self._icm_disabled_widgets.extend([
            lambda_contour_label,
            self.graphcut_lambda_contour_spin,
        ])
        advanced_lay.addWidget(_param_group_label("Spatial Pairwise"))
        advanced_lay.addLayout(spatial_grid)

        self.graphcut_lambda_t_spin = _float_spin(
            0.0, 1000.0, 1.0, "Temporal pairwise weight."
        )
        temporal_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(
            temporal_grid,
            0,
            "lambda_t:",
            compact_spinbox(self.graphcut_lambda_t_spin),
            field_width=92,
        )
        advanced_lay.addWidget(_param_group_label("Temporal"))
        advanced_lay.addLayout(temporal_grid)

        self.graphcut_init_mode_combo = _combo(
            ["nuclei", "unary", "euclidean", "geodesic"],
            "nuclei",
            "Initialization used before iterative optimization. "
            "'nuclei' (default) labels only nucleus pixels and lets the "
            "algorithm grow each cell naturally from its nucleus.",
        )
        self.graphcut_min_round_flips_spin = _int_spin(
            0, 1_000_000, 0, "Stop after a round with fewer flips than this value."
        )
        solver_grid = block_grid(horizontal_spacing=12)
        init_mode_label, _, _, _ = add_block_pair_row(
            solver_grid,
            0,
            "init_mode:",
            self.graphcut_init_mode_combo,
            "min_round_flips:",
            compact_spinbox(self.graphcut_min_round_flips_spin),
            field_width=92,
        )
        self._icm_disabled_widgets.extend([
            init_mode_label,
            self.graphcut_init_mode_combo,
        ])
        advanced_lay.addWidget(_param_group_label("Solver"))
        advanced_lay.addLayout(solver_grid)

        self.graphcut_advanced_section = CollapsibleSection(
            "Advanced Graphcut Parameters", advanced_inner, expanded=True
        )
        selection_lay.addWidget(self.graphcut_advanced_section)

        selection_run_row = block_grid(horizontal_spacing=12)
        self.run_boundary_selection_btn = QPushButton(
            "Run Track-Conditioned Boundary Selection"
        )
        self.run_boundary_selection_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        add_block_button_row(
            selection_run_row, 0, self.run_boundary_selection_btn
        )
        selection_lay.addLayout(selection_run_row)

        self.boundary_selection_status_lbl = _stage_status()
        selection_lay.addWidget(self.boundary_selection_status_lbl)

        self.boundary_selection_progress_bar = _stage_progress()
        selection_lay.addWidget(self.boundary_selection_progress_bar)

        self.boundary_selection_output_files = _stage_files("Outputs", [
            ("3_cell/tracked_labels.tif", "Cell labels"),
        ])
        selection_lay.addWidget(self.boundary_selection_output_files)

        self.boundary_selection_section = CollapsibleSection(
            "2. Track-Conditioned Boundary Selection",
            selection_inner,
            expanded=False,
        )
        layout.addWidget(self.boundary_selection_section)

        # ---- 3. Correction ----
        correction_inner = QWidget()
        correction_lay = QVBoxLayout(correction_inner)
        correction_lay.setContentsMargins(0, 0, 0, 0)
        correction_lay.setSpacing(4)
        correction_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
        )
        correction_lay.addWidget(self.correction_widget)

        self.correction_section = CollapsibleSection(
            "3. Correction", correction_inner, expanded=False
        )
        layout.addWidget(self.correction_section)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.preview_contour_btn.clicked.connect(self._on_preview_contour_maps)
        self.build_contour_maps_btn.clicked.connect(self._on_build_contour_maps)
        self.run_boundary_selection_btn.clicked.connect(
            self._on_run_boundary_selection
        )
        self.graphcut_solver_combo.currentTextChanged.connect(
            self._on_solver_changed
        )
        self._on_solver_changed(self.graphcut_solver_combo.currentText())

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _contour_maps_path(self) -> Path | None:
        return (
            self._pos_dir / "3_cell" / "contour_maps.tif"
            if self._pos_dir
            else None
        )

    def _foreground_scores_path(self) -> Path | None:
        return (
            self._pos_dir / "3_cell" / "foreground_scores.tif"
            if self._pos_dir
            else None
        )

    def _foreground_masks_path(self) -> Path | None:
        return (
            self._pos_dir / "3_cell" / "foreground_masks.tif"
            if self._pos_dir
            else None
        )

    def _nucleus_labels_path(self) -> Path | None:
        return (
            self._pos_dir / "2_nucleus" / "tracked_labels.tif"
            if self._pos_dir
            else None
        )

    def _cell_labels_output_path(self) -> Path | None:
        return (
            self._pos_dir / "3_cell" / "tracked_labels.tif"
            if self._pos_dir
            else None
        )

    def _prob_path(self) -> Path | None:
        return (
            self._pos_dir / "1_cellpose" / "cell_prob_3dt.tif"
            if self._pos_dir
            else None
        )

    def _dp_path(self) -> Path | None:
        return (
            self._pos_dir / "1_cellpose" / "cell_dp_3dt.tif"
            if self._pos_dir
            else None
        )

    def _filtered_dp_path(self) -> Path | None:
        return (
            self._pos_dir / "3_cell" / "filtered_dp.tif"
            if self._pos_dir
            else None
        )

    def _graphcut_script_path(self) -> Path:
        return (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "experiment_cell_2d_t_multilabel_graphcut.py"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._refresh_stage_files(pos_dir)
        if pos_dir is None:
            self.correction_widget.deactivate()

    def _refresh_stage_files(self, pos_dir: Path | None = None) -> None:
        if pos_dir is None:
            pos_dir = self._pos_dir
        for files_widget in (
            self.contour_input_files,
            self.contour_output_files,
            self.boundary_selection_input_files,
            self.boundary_selection_output_files,
        ):
            files_widget.refresh(pos_dir)

    def get_state(self) -> dict:
        return {
            "cp_min": self.cp_min_spin.value(),
            "cp_max": self.cp_max_spin.value(),
            "cp_step": self.cp_step_spin.value(),
            "contour_flow_threshold": self.contour_flow_threshold_spin.value(),
            "contour_niter": self.contour_niter_spin.value(),
            "cp_gamma_min": self.cp_gamma_min_spin.value(),
            "cp_gamma_max": self.cp_gamma_max_spin.value(),
            "cp_gamma_step": self.cp_gamma_step_spin.value(),
            "contour_fg_threshold": self.contour_fg_threshold_spin.value(),
            "graphcut_solver": self.graphcut_solver_combo.currentText(),
            "graphcut_unary_mode": self.graphcut_unary_mode_combo.currentText(),
            "graphcut_boundary_mode": self.graphcut_boundary_mode_combo.currentText(),
            "graphcut_n_iters": self.graphcut_n_iters_spin.value(),
            "graphcut_n_workers": self.graphcut_n_workers_spin.value(),
            "graphcut_alpha_unary": self.graphcut_alpha_unary_spin.value(),
            "graphcut_lambda_geodesic": self.graphcut_lambda_geodesic_spin.value(),
            "graphcut_lambda_flow": self.graphcut_lambda_flow_spin.value(),
            "graphcut_lambda_s": self.graphcut_lambda_s_spin.value(),
            "graphcut_beta_s": self.graphcut_beta_s_spin.value(),
            "graphcut_lambda_contour": self.graphcut_lambda_contour_spin.value(),
            "graphcut_lambda_t": self.graphcut_lambda_t_spin.value(),
            "graphcut_init_mode": self.graphcut_init_mode_combo.currentText(),
            "graphcut_min_round_flips": self.graphcut_min_round_flips_spin.value(),
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        spin_values = {
            "cp_min": self.cp_min_spin,
            "cp_max": self.cp_max_spin,
            "cp_step": self.cp_step_spin,
            "contour_flow_threshold": self.contour_flow_threshold_spin,
            "contour_niter": self.contour_niter_spin,
            "cp_gamma_min": self.cp_gamma_min_spin,
            "cp_gamma_max": self.cp_gamma_max_spin,
            "cp_gamma_step": self.cp_gamma_step_spin,
            "contour_fg_threshold": self.contour_fg_threshold_spin,
        }
        for key, widget in spin_values.items():
            if key in state:
                widget.setValue(state[key])

        combo_values = {
            "graphcut_solver": self.graphcut_solver_combo,
            "graphcut_unary_mode": self.graphcut_unary_mode_combo,
            "graphcut_boundary_mode": self.graphcut_boundary_mode_combo,
            "graphcut_init_mode": self.graphcut_init_mode_combo,
        }
        for key, widget in combo_values.items():
            if key in state:
                widget.setCurrentText(str(state[key]))

        graphcut_spin_values = {
            "graphcut_n_iters": self.graphcut_n_iters_spin,
            "graphcut_n_workers": self.graphcut_n_workers_spin,
            "graphcut_alpha_unary": self.graphcut_alpha_unary_spin,
            "graphcut_lambda_geodesic": self.graphcut_lambda_geodesic_spin,
            "graphcut_lambda_flow": self.graphcut_lambda_flow_spin,
            "graphcut_lambda_s": self.graphcut_lambda_s_spin,
            "graphcut_beta_s": self.graphcut_beta_s_spin,
            "graphcut_lambda_contour": self.graphcut_lambda_contour_spin,
            "graphcut_lambda_t": self.graphcut_lambda_t_spin,
            "graphcut_min_round_flips": self.graphcut_min_round_flips_spin,
        }
        for key, widget in graphcut_spin_values.items():
            if key in state:
                widget.setValue(state[key])

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def _set_contour_status(self, msg: str) -> None:
        self.contour_status_lbl.setText(msg)
        self.contour_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_selection_status(self, msg: str) -> None:
        self.boundary_selection_status_lbl.setText(msg)
        self.boundary_selection_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _on_stage_progress(self, bar: QProgressBar, set_status, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            if total > 0:
                bar.setVisible(True)
                bar.setRange(0, total)
                bar.setValue(done)
            set_status(msg)
        else:
            set_status(str(data))

    def _show_layer(self, layer_name: str, data: np.ndarray, kwargs: dict, adder) -> None:
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = data
        else:
            adder(data, name=layer_name, **kwargs)

    def _current_t(self) -> int:
        dims = getattr(self.viewer, "dims", None)
        step = getattr(dims, "current_step", (0,))
        return int(step[0]) if len(step) >= 1 else 0

    # ------------------------------------------------------------------
    # 1. Contour Maps
    # ------------------------------------------------------------------
    def _cellprob_thresholds(self) -> list[float]:
        step = self.cp_step_spin.value()
        return list(
            np.arange(
                self.cp_min_spin.value(),
                self.cp_max_spin.value() + step / 2,
                step,
            )
        )

    def _cp_gammas(self) -> list[float]:
        step = self.cp_gamma_step_spin.value()
        return list(
            np.arange(
                self.cp_gamma_min_spin.value(),
                self.cp_gamma_max_spin.value() + step / 2,
                step,
            )
        )

    def _build_consensus_boundary_averaged(
        self,
        prob_3d: np.ndarray,
        dp_2d: np.ndarray,
        thresholds: list[float],
        gammas: list[float],
        *,
        flow_threshold: float,
        niter: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        boundary_accum = None
        foreground_accum = None
        for gamma in gammas:
            prob_2d = apply_gamma(prob_3d, gamma).mean(axis=0)
            b, fg = build_consensus_boundary_2d(
                prob_2d,
                dp_2d,
                thresholds,
                flow_threshold=flow_threshold,
                reduction="mean",
                niter=niter,
            )
            if boundary_accum is None:
                boundary_accum = b.copy()
                foreground_accum = fg.copy()
            else:
                boundary_accum += b
                foreground_accum += fg
        n = len(gammas)
        return boundary_accum / n, foreground_accum / n

    def _set_contour_buttons_running(self, running: bool) -> None:
        self.build_contour_maps_btn.setEnabled(not running)
        self.preview_contour_btn.setEnabled(not running)
        self.contour_progress_bar.setVisible(running)
        if not running:
            self.contour_progress_bar.setValue(0)

    def _on_build_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return

        prob_path = self._prob_path()
        filtered_dp_path = self._filtered_dp_path()
        contour_path = self._contour_maps_path()
        score_path = self._foreground_scores_path()
        foreground_path = self._foreground_masks_path()
        for path, name in [
            (prob_path, "cell_prob_3dt.tif"),
            (filtered_dp_path, "filtered_dp.tif"),
        ]:
            if path is None or not path.exists():
                self._set_contour_status(f"Missing: {name}")
                return
        if contour_path is None or score_path is None or foreground_path is None:
            self._set_contour_status("No project open.")
            return

        pos_dir = self._pos_dir
        thresholds = self._cellprob_thresholds()
        gammas = self._cp_gammas()
        flow_threshold = self.contour_flow_threshold_spin.value()
        niter = self.contour_niter_spin.value()
        foreground_threshold = self.contour_fg_threshold_spin.value()
        build_fn = self._build_consensus_boundary_averaged

        def _on_done(result):
            self._contour_worker = None
            self._set_contour_buttons_running(False)
            contours, scores, foreground = result
            self._show_layer(
                _CELL_CONTOUR_LAYER,
                contours,
                {"colormap": "magma", "visible": True},
                self.viewer.add_image,
            )
            self._show_layer(
                _CELL_FOREGROUND_SCORE_LAYER,
                scores,
                {"colormap": "viridis", "visible": True},
                self.viewer.add_image,
            )
            self._show_layer(
                _CELL_FOREGROUND_LAYER,
                foreground,
                {},
                self.viewer.add_labels,
            )
            self._refresh_stage_files(pos_dir)
            self._set_contour_status("Contour maps complete.")

        @thread_worker(connect={
            "yielded": lambda data: self._on_stage_progress(
                self.contour_progress_bar, self._set_contour_status, data
            ),
            "returned": _on_done,
            "errored": lambda exc: self._on_contour_error(exc),
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            filtered_dp_stack = np.asarray(tifffile.imread(str(filtered_dp_path)), dtype=np.float32)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis]
            if filtered_dp_stack.ndim == 3:
                filtered_dp_stack = filtered_dp_stack[np.newaxis]

            n_t = min(prob_stack.shape[0], filtered_dp_stack.shape[0])
            contour_frames: list[np.ndarray] = []
            score_frames: list[np.ndarray] = []
            foreground_frames: list[np.ndarray] = []
            for t in range(n_t):
                yield (t + 1, n_t, f"Building contour maps: frame {t + 1}/{n_t}...")
                contour, foreground_score = build_fn(
                    prob_stack[t],          # (Z, Y, X)
                    filtered_dp_stack[t],   # (2, Y, X)
                    thresholds,
                    gammas,
                    flow_threshold=flow_threshold,
                    niter=niter,
                )
                contour_frames.append(contour.astype(np.float32, copy=False))
                foreground_score = foreground_score.astype(np.float32, copy=False)
                score_frames.append(foreground_score)
                foreground_frames.append(
                    (foreground_score >= foreground_threshold).astype(np.uint8)
                )

            contour_arr = np.stack(contour_frames)
            score_arr = np.stack(score_frames)
            foreground_arr = np.stack(foreground_frames)
            contour_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(contour_path), contour_arr, compression="zlib")
            tifffile.imwrite(str(score_path), score_arr, compression="zlib")
            tifffile.imwrite(str(foreground_path), foreground_arr, compression="zlib")
            return contour_arr, score_arr, foreground_arr

        self._set_contour_status(
            f"Building contour maps ({len(thresholds)} thresholds, {len(gammas)} gamma value(s))..."
        )
        self._set_contour_buttons_running(True)
        self._contour_worker = _worker()

    def _on_preview_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return

        prob_path = self._prob_path()
        filtered_dp_path = self._filtered_dp_path()
        for path, name in [
            (prob_path, "cell_prob_3dt.tif"),
            (filtered_dp_path, "filtered_dp.tif"),
        ]:
            if path is None or not path.exists():
                self._set_contour_status(f"Missing: {name}")
                return

        t_frame = self._current_t()
        thresholds = self._cellprob_thresholds()
        gammas = self._cp_gammas()
        flow_threshold = self.contour_flow_threshold_spin.value()
        niter = self.contour_niter_spin.value()
        foreground_threshold = self.contour_fg_threshold_spin.value()
        build_fn = self._build_consensus_boundary_averaged

        def _on_preview_done(result):
            self._contour_worker = None
            self._set_contour_buttons_running(False)
            contour, foreground_score, n_t, t_idx = result
            contour_data = np.zeros((n_t,) + contour.shape, dtype=np.float32)
            contour_data[t_idx] = contour
            score_data = np.zeros((n_t,) + foreground_score.shape, dtype=np.float32)
            score_data[t_idx] = foreground_score
            mask_data = (score_data >= foreground_threshold).astype(np.uint8)
            self._show_layer(
                _CELL_CONTOUR_LAYER,
                contour_data,
                {"colormap": "magma", "visible": True},
                self.viewer.add_image,
            )
            self._show_layer(
                _CELL_FOREGROUND_SCORE_LAYER,
                score_data,
                {"colormap": "viridis", "visible": True},
                self.viewer.add_image,
            )
            self._show_layer(
                _CELL_FOREGROUND_LAYER,
                mask_data,
                {},
                self.viewer.add_labels,
            )
            self._set_contour_status(
                f"Preview t={t_idx} — {len(thresholds)} thresholds, {len(gammas)} gamma value(s)"
            )

        @thread_worker(connect={
            "returned": _on_preview_done,
            "errored": lambda exc: self._on_contour_error(exc),
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            filtered_dp_stack = np.asarray(tifffile.imread(str(filtered_dp_path)), dtype=np.float32)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis]
            if filtered_dp_stack.ndim == 3:
                filtered_dp_stack = filtered_dp_stack[np.newaxis]
            n_t = min(prob_stack.shape[0], filtered_dp_stack.shape[0])
            t_idx = min(max(t_frame, 0), n_t - 1)
            contour, foreground_score = build_fn(
                prob_stack[t_idx],          # (Z, Y, X)
                filtered_dp_stack[t_idx],   # (2, Y, X)
                thresholds,
                gammas,
                flow_threshold=flow_threshold,
                niter=niter,
            )
            return (
                contour.astype(np.float32, copy=False),
                foreground_score.astype(np.float32, copy=False),
                n_t,
                t_idx,
            )

        self._set_contour_status(f"Previewing contour map for frame t={t_frame}...")
        self._set_contour_buttons_running(True)
        self._contour_worker = _worker()

    def _on_contour_error(self, exc: Exception) -> None:
        self._contour_worker = None
        self._set_contour_buttons_running(False)
        self._set_contour_status(f"Error: {exc}")
        logger.exception("Cell contour worker error", exc_info=exc)

    # ------------------------------------------------------------------
    # 2. Track-Conditioned Boundary Selection
    # ------------------------------------------------------------------
    def _set_boundary_selection_running(self, running: bool) -> None:
        self.run_boundary_selection_btn.setEnabled(not running)
        self.boundary_selection_progress_bar.setVisible(running)
        if running:
            self.boundary_selection_progress_bar.setRange(0, 0)
        else:
            self.boundary_selection_progress_bar.setRange(0, 100)
            self.boundary_selection_progress_bar.setValue(0)

    def _on_solver_changed(self, solver: str) -> None:
        enabled = solver != "icm"
        for widget in self._icm_disabled_widgets:
            if widget is not None:
                widget.setEnabled(enabled)

    def _build_graphcut_command(self, timestamp: str) -> tuple[list[str], Path]:
        if self._pos_dir is None:
            raise RuntimeError("No project open.")
        output_dir = self._pos_dir / "4_cell_graphcut" / timestamp
        cmd = [
            sys.executable,
            str(self._graphcut_script_path()),
            "--pos-dir",
            str(self._pos_dir),
            "--solver",
            self.graphcut_solver_combo.currentText(),
            "--unary-mode",
            self.graphcut_unary_mode_combo.currentText(),
            "--flow-field-path",
            str(self._dp_path()),
            "--boundary-mode",
            self.graphcut_boundary_mode_combo.currentText(),
            "--n-iters",
            str(self.graphcut_n_iters_spin.value()),
            "--n-workers",
            str(self.graphcut_n_workers_spin.value()),
            "--alpha-unary",
            str(self.graphcut_alpha_unary_spin.value()),
            "--lambda-geodesic",
            str(self.graphcut_lambda_geodesic_spin.value()),
            "--lambda-flow",
            str(self.graphcut_lambda_flow_spin.value()),
            "--lambda-s",
            str(self.graphcut_lambda_s_spin.value()),
            "--beta-s",
            str(self.graphcut_beta_s_spin.value()),
            "--lambda-contour",
            str(self.graphcut_lambda_contour_spin.value()),
            "--lambda-t",
            str(self.graphcut_lambda_t_spin.value()),
            "--init-mode",
            self.graphcut_init_mode_combo.currentText(),
            "--min-round-flips",
            str(self.graphcut_min_round_flips_spin.value()),
            "--timestamp",
            timestamp,
            "--overwrite",
        ]
        if self.graphcut_boundary_mode_combo.currentText() == "foreground_inverse":
            cmd.extend([
                "--foreground-score-path",
                str(self._foreground_scores_path()),
            ])
        return cmd, output_dir

    def _on_run_boundary_selection(self) -> None:
        if self._pos_dir is None:
            self._set_selection_status("No project open.")
            return

        solver = self.graphcut_solver_combo.currentText()
        required_files = [
            (self._nucleus_labels_path(), "tracked_labels.tif (nucleus)"),
            (self._contour_maps_path(), "contour_maps.tif"),
            (self._foreground_scores_path(), "foreground_scores.tif"),
            (self._foreground_masks_path(), "foreground_masks.tif"),
        ]
        if solver != "icm":
            required_files.append((self._dp_path(), "cell_dp_3dt.tif"))
        for path, name in required_files:
            if path is None or not path.exists():
                self._set_selection_status(f"Missing: {name}")
                return

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if solver == "icm":
            self._run_boundary_selection_icm(timestamp)
        else:
            self._run_boundary_selection_graphcut(timestamp)

    def _run_boundary_selection_graphcut(self, timestamp: str) -> None:
        cmd, output_dir = self._build_graphcut_command(timestamp)
        canonical_output = self._cell_labels_output_path()
        pos_dir = self._pos_dir

        def _on_done(labels: np.ndarray) -> None:
            self._boundary_selection_worker = None
            self._set_boundary_selection_running(False)
            self._show_layer(
                _GRAPHCUT_CELL_LABELS_LAYER,
                labels,
                {"visible": True},
                self.viewer.add_labels,
            )
            self._refresh_stage_files(pos_dir)
            self._set_selection_status(
                f"Graphcut boundary selection complete: {canonical_output}"
            )

        @thread_worker(connect={
            "yielded": lambda data: self._on_stage_progress(
                self.boundary_selection_progress_bar,
                self._set_selection_status,
                data,
            ),
            "returned": _on_done,
            "errored": lambda exc: self._on_boundary_selection_error(exc),
        })
        def _worker():
            yield "Running graphcut boundary selection..."
            completed = subprocess.run(
                cmd,
                cwd=str(self._graphcut_script_path().parents[1]),
                capture_output=True,
                text=True,
            )
            if completed.stdout:
                yield completed.stdout.strip().splitlines()[-1]
            if completed.returncode != 0:
                msg = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(msg or f"Graphcut command failed with exit code {completed.returncode}")

            graphcut_output = output_dir / "cell_labels.tif"
            if not graphcut_output.exists():
                raise FileNotFoundError(graphcut_output)
            if canonical_output is None:
                raise RuntimeError("No project open.")
            canonical_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(graphcut_output, canonical_output)
            labels = np.asarray(tifffile.imread(str(canonical_output)))
            return labels

        self._set_selection_status("Running graphcut boundary selection...")
        self._set_boundary_selection_running(True)
        self._boundary_selection_worker = _worker()

    def _run_boundary_selection_icm(self, timestamp: str) -> None:
        if self._pos_dir is None:
            raise RuntimeError("No project open.")
        from cellflow.segmentation import CellLabelICMParams

        output_dir = self._pos_dir / "4_cell_graphcut" / timestamp
        canonical_output = self._cell_labels_output_path()
        pos_dir = self._pos_dir
        params = CellLabelICMParams(
            alpha_unary=self.graphcut_alpha_unary_spin.value(),
            lambda_s=self.graphcut_lambda_s_spin.value(),
            beta_s=self.graphcut_beta_s_spin.value(),
            lambda_t=self.graphcut_lambda_t_spin.value(),
            n_iters=self.graphcut_n_iters_spin.value(),
            min_round_flips=self.graphcut_min_round_flips_spin.value(),
        )

        def _on_done(labels: np.ndarray) -> None:
            self._boundary_selection_worker = None
            self._set_boundary_selection_running(False)
            self._show_layer(
                _GRAPHCUT_CELL_LABELS_LAYER,
                labels,
                {"visible": True},
                self.viewer.add_labels,
            )
            self._refresh_stage_files(pos_dir)
            self._set_selection_status(
                f"Graphcut boundary selection complete: {canonical_output}"
            )

        @thread_worker(connect={
            "yielded": lambda data: self._on_stage_progress(
                self.boundary_selection_progress_bar,
                self._set_selection_status,
                data,
            ),
            "returned": _on_done,
            "errored": lambda exc: self._on_boundary_selection_error(exc),
        })
        def _worker():
            from cellflow.segmentation.cell_label_icm import run_cell_icm_from_pos_dir

            yield "Running ICM boundary selection..."
            labels = run_cell_icm_from_pos_dir(pos_dir, params)
            graphcut_output = output_dir / "cell_labels.tif"
            graphcut_output.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(graphcut_output), labels, compression="zlib")
            if canonical_output is None:
                raise RuntimeError("No project open.")
            canonical_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(graphcut_output, canonical_output)
            labels = np.asarray(tifffile.imread(str(canonical_output)))
            return labels

        self._set_selection_status("Running ICM boundary selection...")
        self._set_boundary_selection_running(True)
        self._boundary_selection_worker = _worker()

    def _on_boundary_selection_error(self, exc: Exception) -> None:
        self._boundary_selection_worker = None
        self._set_boundary_selection_running(False)
        self._set_selection_status(f"Error: {exc}")
        logger.exception(
            "Boundary selection worker error", exc_info=exc
        )
"""Cell segmentation widget for CellFlow — Flow-Following Segmentation."""
from __future__ import annotations

import logging
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QSpinBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.correction.labels import best_overlapping_label, expand_label_to_foreground
from cellflow.database.tracked import read_full_tracked_stack
from cellflow.napari.cell_boundary_workflow_widget import CellBoundaryWorkflowWidget
from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.napari.ui_style import (
    add_block_button_row,
    add_parameter_grid_row,
    block_grid,
    status_label,
)

logger = logging.getLogger(__name__)

_FILTERED_FLOW_LAYER = "Filtered Flow Magnitude"
_FOREGROUND_MASK_LAYER = "Foreground Mask"
_TRACKED_CELL_LAYER = "Tracked: Cell"
_CELL_ZAVG_LAYER = "Cell z-avg"
_NUC_ZAVG_LAYER = "Nucleus z-avg"
_FF_SPIN_WIDTH = 80
_FF_SPIN_MIN_WIDTH = int(_FF_SPIN_WIDTH * 0.9)


class CellWorkflowWidget(QWidget):
    """Cell segmentation — Flow-Following Segmentation."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._ff_worker = None
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        def _stage_files(group_label: str, entries: list[tuple[str, str]]) -> PipelineFilesWidget:
            return PipelineFilesWidget([(group_label, entries)], viewer=self.viewer)

        def _stage_status() -> QLabel:
            label = QLabel("")
            label.setWordWrap(True)
            label.setVisible(False)
            status_label(label)
            return label

        def _stage_progress() -> QProgressBar:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setVisible(False)
            return bar

        def _param_grid():
            grid = block_grid(horizontal_spacing=12, vertical_spacing=4)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)
            return grid

        def _dspin(lo, hi, val, step, decimals=1):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
            s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return s

        def _ispin(lo, hi, val, step=1):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
            s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return s

        self.filtered_flow_params_widget = QWidget()
        filter_lay = QVBoxLayout(self.filtered_flow_params_widget)
        filter_lay.setContentsMargins(0, 0, 0, 0)
        filter_lay.setSpacing(4)
        filter_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.filtered_flow_input_files = _stage_files("Inputs", [
            ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
            ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
        ])
        filter_lay.addWidget(self.filtered_flow_input_files)
        filter_grid = _param_grid()
        self.ff_median_time_spin   = _ispin(1, 15, 3)
        self.ff_median_space_spin  = _ispin(1, 15, 5)
        self.ff_gauss_time_spin    = _dspin(0.0, 10.0, 0.0, 0.1)
        self.ff_gauss_space_spin   = _dspin(0.0, 10.0, 0.0, 0.1)
        add_parameter_grid_row(filter_grid, 0, 0, "Median t kernel:", self.ff_median_time_spin)
        add_parameter_grid_row(filter_grid, 0, 1, "Median xy kernel:", self.ff_median_space_spin)
        add_parameter_grid_row(filter_grid, 1, 0, "Gaussian t sigma:", self.ff_gauss_time_spin)
        add_parameter_grid_row(filter_grid, 1, 1, "Gaussian xy sigma:", self.ff_gauss_space_spin)
        filter_lay.addLayout(filter_grid)

        self.ff_flow_mag_btn = QPushButton("Create filtered_dp")
        self.ff_flow_mag_btn.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
        self.ff_flow_mag_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        filter_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(filter_btn_row, 0, self.ff_flow_mag_btn)
        filter_lay.addLayout(filter_btn_row)
        self.filtered_flow_status_lbl = _stage_status()
        filter_lay.addWidget(self.filtered_flow_status_lbl)
        self.filtered_flow_progress_bar = _stage_progress()
        filter_lay.addWidget(self.filtered_flow_progress_bar)
        self.filtered_flow_output_files = _stage_files("Outputs", [
            ("3_cell/filtered_dp.tif", "Filtered flow vectors"),
            ("3_cell/filtered_flow_mag.tif", "Filtered flow magnitude"),
        ])
        filter_lay.addWidget(self.filtered_flow_output_files)

        self.filtered_flow_section = CollapsibleSection(
            "Flow Filtering", self.filtered_flow_params_widget, expanded=False
        )
        layout.addWidget(self.filtered_flow_section)

        # ---- Contour Maps + Segmentation (embedded from CellBoundaryWorkflowWidget) ----
        self._seg_widget = CellBoundaryWorkflowWidget(self.viewer, parent=None)
        self._seg_widget.correction_section.setVisible(False)
        self._seg_widget.boundary_selection_section.set_title("Segmentation")
        self._seg_widget.contour_section.set_title("Contour Maps")
        # Remove the internal stretch so it doesn't create extra dead space
        seg_layout = self._seg_widget.layout()
        last_item = seg_layout.itemAt(seg_layout.count() - 1)
        if last_item and last_item.spacerItem():
            seg_layout.removeItem(last_item)
        layout.addWidget(self._seg_widget)

        self.correction_params_widget = QWidget()
        correction_lay = QVBoxLayout(self.correction_params_widget)
        correction_lay.setContentsMargins(0, 0, 0, 0)
        correction_lay.setSpacing(4)
        correction_lay.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.correction_input_files = _stage_files("Inputs", [
            ("3_cell/tracked_labels.tif", "Cell labels"),
            ("0_input/cell_zavg.tif", "Cell z-avg"),
            ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
        ])
        correction_lay.addWidget(self.correction_input_files)

        self.load_cell_correction_btn = QPushButton("Load Cell Labels")
        self.save_cell_correction_btn = QPushButton("Save Cell Labels")
        self.reassign_cell_ids_btn = QPushButton("Reassign IDs")
        self.expand_selected_cell_btn = QPushButton("Expand Selected Cell")
        for button in (
            self.load_cell_correction_btn,
            self.save_cell_correction_btn,
            self.reassign_cell_ids_btn,
            self.expand_selected_cell_btn,
        ):
            button.setMinimumWidth(_FF_SPIN_MIN_WIDTH)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        correction_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(
            correction_btn_row,
            0,
            self.load_cell_correction_btn,
            self.save_cell_correction_btn,
        )
        add_block_button_row(correction_btn_row, 1, self.reassign_cell_ids_btn)
        correction_lay.addLayout(correction_btn_row)

        expand_grid = _param_grid()
        self.expand_cell_max_px_spin = _ispin(0, 999, 25)
        add_parameter_grid_row(
            expand_grid,
            0,
            0,
            "Max expansion px:",
            self.expand_cell_max_px_spin,
        )
        correction_lay.addLayout(expand_grid)
        expand_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(expand_btn_row, 0, self.expand_selected_cell_btn)
        correction_lay.addLayout(expand_btn_row)

        self.correction_status_lbl = _stage_status()
        correction_lay.addWidget(self.correction_status_lbl)
        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
        )
        correction_lay.addWidget(self.correction_widget)
        self.correction_shortcuts_section = CollapsibleSection(
            "Correction Shortcuts",
            self.correction_widget.build_shortcuts_widget(),
            expanded=False,
        )
        correction_lay.addWidget(self.correction_shortcuts_section)
        self.correction_section = CollapsibleSection(
            "Correction", self.correction_params_widget, expanded=False
        )
        layout.addWidget(self.correction_section)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.ff_flow_mag_btn.clicked.connect(self._on_create_flow_mag)
        self.load_cell_correction_btn.clicked.connect(self._on_load_cell_correction)
        self.save_cell_correction_btn.clicked.connect(self._on_save_cell_correction)
        self.reassign_cell_ids_btn.clicked.connect(self._on_reassign_cell_ids)
        self.expand_selected_cell_btn.clicked.connect(self._on_expand_selected_cell)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _prob_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_prob_3dt.tif" if self._pos_dir else None

    def _dp_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_dp_3dt.tif" if self._pos_dir else None

    def _foreground_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "foreground_masks.tif" if self._pos_dir else None

    def _nucleus_labels_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    def _flow_mag_out_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "filtered_flow_mag.tif" if self._pos_dir else None

    def _filtered_dp_out_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "filtered_dp.tif" if self._pos_dir else None

    def _cell_labels_out_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "tracked_labels.tif" if self._pos_dir else None

    def _cell_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "cell_zavg.tif" if self._pos_dir else None

    def _nucleus_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "nucleus_zavg.tif" if self._pos_dir else None

    # ------------------------------------------------------------------
    # State + status
    # ------------------------------------------------------------------
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._refresh_stage_files(pos_dir)
        self._seg_widget.refresh(pos_dir)

    def _refresh_stage_files(self, pos_dir: Path | None = None) -> None:
        if pos_dir is None:
            pos_dir = self._pos_dir
        for files_widget in (
            self.filtered_flow_input_files,
            self.filtered_flow_output_files,
            self.correction_input_files,
        ):
            files_widget.refresh(pos_dir)

    def _set_correction_status(self, msg: str) -> None:
        self.correction_status_lbl.setText(msg)
        self.correction_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def get_state(self) -> dict:
        return {
            "flow_following": {
                "median_time":  self.ff_median_time_spin.value(),
                "median_space": self.ff_median_space_spin.value(),
                "gauss_time":   self.ff_gauss_time_spin.value(),
                "gauss_space":  self.ff_gauss_space_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if "flow_following" in state:
            ff = state["flow_following"]
            if "median_time"  in ff: self.ff_median_time_spin.setValue(ff["median_time"])
            if "median_space" in ff: self.ff_median_space_spin.setValue(ff["median_space"])
            if "gauss_time"   in ff: self.ff_gauss_time_spin.setValue(ff["gauss_time"])
            if "gauss_space"  in ff: self.ff_gauss_space_spin.setValue(ff["gauss_space"])

    def _set_stage_status(self, stage: str, msg: str) -> None:
        label = self._stage_status_label(stage)
        label.setText(msg)
        label.setVisible(bool(msg))
        logger.info(msg)

    def _stage_status_label(self, stage: str) -> QLabel:
        return {
            "filtered_flow": self.filtered_flow_status_lbl,
        }[stage]

    def _stage_progress_bar(self, stage: str) -> QProgressBar:
        return {
            "filtered_flow": self.filtered_flow_progress_bar,
        }[stage]

    def _set_ff_buttons_running(self, running: bool) -> None:
        self.ff_flow_mag_btn.setEnabled(not running)
        if not running:
            self.filtered_flow_progress_bar.setValue(0)
            self.filtered_flow_progress_bar.setVisible(False)

    def _show_layer(self, layer_name: str, data: np.ndarray, kwargs: dict, adder) -> None:
        if layer_name in self.viewer.layers:
            try:
                self.viewer.layers[layer_name].data = data
            except Exception:
                self.viewer.layers.remove(self.viewer.layers[layer_name])
                adder(data, name=layer_name, **kwargs)
        else:
            adder(data, name=layer_name, **kwargs)

    def _on_stage_progress(self, stage: str, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            bar = self._stage_progress_bar(stage)
            if total > 0:
                bar.setVisible(True)
                bar.setRange(0, total)
                bar.setValue(done)
            self._set_stage_status(stage, msg)
        else:
            self._set_stage_status(stage, str(data))

    def _on_stage_worker_error(self, stage: str, exc: Exception) -> None:
        if self._ff_worker is None:
            return
        self._ff_worker = None
        self._set_ff_buttons_running(False)
        self._set_stage_status(stage, f"Error: {exc}")
        logger.exception("Cell workflow worker error", exc_info=exc)

    # ------------------------------------------------------------------
    # Manual correction
    # ------------------------------------------------------------------
    @staticmethod
    def _broadcast_reference_image(image: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray | None:
        if image is None:
            return None
        if image.ndim == 2 and len(shape) >= 3:
            return np.broadcast_to(image[np.newaxis], (shape[0],) + image.shape).copy()
        return image

    def _on_load_cell_correction(self) -> None:
        labels_path = self._cell_labels_out_path()
        cell_zavg_path = self._cell_zavg_path()
        nuc_zavg_path = self._nucleus_zavg_path()
        if labels_path is None or not labels_path.exists():
            self._set_correction_status("No cell labels file found.")
            return
        self._set_correction_status("Loading cell labels...")

        @thread_worker(connect={
            "returned": self._on_load_cell_correction_done,
            "errored": lambda exc: self._set_correction_status(f"Error: {exc}"),
        })
        def _worker():
            labels = read_full_tracked_stack(labels_path)
            cell_zavg = (
                np.asarray(tifffile.imread(str(cell_zavg_path)), dtype=np.float32)
                if cell_zavg_path and cell_zavg_path.exists() else None
            )
            nuc_zavg = (
                np.asarray(tifffile.imread(str(nuc_zavg_path)), dtype=np.float32)
                if nuc_zavg_path and nuc_zavg_path.exists() else None
            )
            return labels, cell_zavg, nuc_zavg

        _worker()

    def _on_load_cell_correction_done(self, result: tuple) -> None:
        labels, cell_zavg, nuc_zavg = result
        if _TRACKED_CELL_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_CELL_LAYER].data = labels
        else:
            self.viewer.add_labels(labels, name=_TRACKED_CELL_LAYER)

        for image, layer_name, cmap in (
            (self._broadcast_reference_image(cell_zavg, labels.shape), _CELL_ZAVG_LAYER, "gray"),
            (self._broadcast_reference_image(nuc_zavg, labels.shape), _NUC_ZAVG_LAYER, "bop orange"),
        ):
            if image is None:
                continue
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = image
            else:
                self.viewer.add_image(image, name=layer_name, colormap=cmap, blending="additive")

        self._set_correction_status(f"Loaded cell label stack {labels.shape} into napari.")
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
        self.correction_widget.activate_layer(layer)
        self.correction_section.expand()

    def set_selection_callback(self, fn) -> None:
        """Register a callback for cell correction label selection changes."""
        self.correction_widget.set_selection_callback(fn)

    def select_matching_cell_label(
        self,
        t: int,
        source_label: int,
        *,
        source_labels: np.ndarray | None = None,
    ) -> None:
        """Highlight the cell label that best overlaps a selected nucleus label."""
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            return
        if source_labels is None:
            if "Tracked: Nucleus" not in self.viewer.layers:
                return
            source_labels = np.asarray(self.viewer.layers["Tracked: Nucleus"].data)
        target_labels = np.asarray(self.viewer.layers[_TRACKED_CELL_LAYER].data)
        matched_label = best_overlapping_label(target_labels, source_labels, t, source_label)
        self.correction_widget.select_label(t, matched_label, notify=False)

    def _on_save_cell_correction(self) -> None:
        labels_path = self._cell_labels_out_path()
        if labels_path is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._set_correction_status("No cell labels layer to save.")
            return
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
        data = np.asarray(layer.data)
        if data.ndim != 3:
            self._set_correction_status("Cell labels layer is not a 3D stack.")
            return
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            str(labels_path),
            data.astype(np.uint32, copy=False),
            compression="zlib",
        )
        self._refresh_stage_files(self._pos_dir)
        self._set_correction_status(f"Saved {data.shape[0]} frame(s) to {labels_path.name}.")

    def _on_reassign_cell_ids(self) -> None:
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._set_correction_status("No cell labels layer loaded.")
            return
        stack = np.asarray(self.viewer.layers[_TRACKED_CELL_LAYER].data)
        unique_ids = np.unique(stack)
        unique_ids = unique_ids[unique_ids != 0]
        if unique_ids.size == 0:
            self._set_correction_status("No cell IDs to reassign.")
            return
        lut = np.zeros(int(unique_ids.max()) + 1, dtype=np.uint32)
        for new_id, old_id in enumerate(unique_ids, start=1):
            lut[int(old_id)] = new_id
        self.viewer.layers[_TRACKED_CELL_LAYER].data = lut[stack]
        self._set_correction_status(
            f"Reassigned {len(unique_ids)} cell IDs to contiguous range 1-{len(unique_ids)}. Unsaved."
        )

    def _foreground_stack_for_expansion(self) -> np.ndarray | None:
        if _FOREGROUND_MASK_LAYER in self.viewer.layers:
            return np.asarray(self.viewer.layers[_FOREGROUND_MASK_LAYER].data)
        fg_path = self._foreground_path()
        if fg_path is None or not fg_path.exists():
            return None
        foreground = np.asarray(tifffile.imread(str(fg_path)))
        self._show_layer(_FOREGROUND_MASK_LAYER, foreground, {}, self.viewer.add_labels)
        return foreground

    def _on_expand_selected_cell(self) -> None:
        if self._pos_dir is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked cell labels layer loaded.")
            return
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
        if self.correction_widget._layer is not layer:
            self._set_correction_status("No active tracked cell labels layer.")
            return
        label_id = int(self.correction_widget._selected_label)
        if label_id == 0:
            self._set_correction_status("No cell selected.")
            return

        labels = np.asarray(layer.data)
        if labels.ndim < 3:
            self._set_correction_status("Tracked cell labels layer is not a 3D stack.")
            return
        t = self._current_time_index(labels.shape[0])
        seg2d = self.correction_widget._frame_view(layer, t)
        if not np.any(seg2d == label_id):
            self._set_correction_status(f"Cell {label_id} not present at t={t}.")
            return

        foreground = self._foreground_stack_for_expansion()
        if foreground is None:
            self._set_correction_status("Foreground mask not found.")
            return
        if foreground.shape != labels.shape:
            self._set_correction_status(
                f"Foreground mask shape {foreground.shape} does not match labels shape {labels.shape}."
            )
            return
        foreground2d = foreground[t]
        while foreground2d.ndim > 2:
            if foreground2d.shape[0] != 1:
                self._set_correction_status(
                    f"Foreground mask frame has unsupported shape {foreground2d.shape}."
                )
                return
            foreground2d = foreground2d[0]

        before = seg2d.copy()
        try:
            added = expand_label_to_foreground(
                seg2d,
                foreground2d,
                label_id,
                max_distance=int(self.expand_cell_max_px_spin.value()),
            )
        except ValueError as exc:
            self._set_correction_status(str(exc))
            return
        if added == 0:
            seed_touches_foreground = bool(np.any((foreground2d > 0) & (before == label_id)))
            if not seed_touches_foreground:
                self._set_correction_status(
                    f"Cell {label_id} does not touch foreground at t={t}."
                )
            else:
                self._set_correction_status(f"Expansion added no pixels for cell {label_id} at t={t}.")
            return

        self.correction_widget._record_history(layer, t, before)
        layer.refresh()
        self.correction_widget._update_highlight(t, label_id)
        self._set_correction_status(
            f"Expanded cell {label_id} at t={t} by {added} px. Unsaved."
        )

    # ------------------------------------------------------------------
    # Run / Cancel
    # ------------------------------------------------------------------
    def _read_dp_tcyx(self, prob_path: Path, dp_path: Path) -> np.ndarray:
        from cellflow.database.hypotheses import normalize_seeded_watershed_dp_stack

        prob = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
        if prob.ndim == 3:
            prob = prob[np.newaxis]
        dp_raw = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
        dp_full = normalize_seeded_watershed_dp_stack(dp_raw, prob.shape)
        return dp_full[:, :, :2].mean(axis=1).astype(np.float32)

    def _on_create_flow_mag(self) -> None:
        if self._pos_dir is None:
            self._set_stage_status("filtered_flow", "No project open.")
            return

        prob_path = self._prob_path()
        dp_path = self._dp_path()
        filtered_dp_path = self._filtered_dp_out_path()
        flow_mag_path = self._flow_mag_out_path()

        for path, name in [
            (prob_path, "cell_prob_3dt.tif"),
            (dp_path,   "cell_dp_3dt.tif"),
        ]:
            if path is None or not path.exists():
                self._set_stage_status("filtered_flow", f"Missing: {name}")
                return
        if filtered_dp_path is None or flow_mag_path is None:
            self._set_stage_status("filtered_flow", "No project open.")
            return

        params_snapshot = self._params_from_ui()
        pos_dir = self._pos_dir

        def _on_done(result):
            self._ff_worker = None
            self._set_ff_buttons_running(False)
            filtered_mag = result
            self._show_layer(
                _FILTERED_FLOW_LAYER,
                filtered_mag,
                {"colormap": "inferno", "blending": "additive"},
                self.viewer.add_image,
            )
            self._refresh_stage_files(pos_dir)
            self._set_stage_status("filtered_flow", "Flow magnitude complete.")

        @thread_worker(connect={
            "yielded":  lambda data: self._on_stage_progress("filtered_flow", data),
            "returned": _on_done,
            "errored":  lambda exc: self._on_stage_worker_error("filtered_flow", exc),
        })
        def _worker():
            from cellflow.segmentation import compute_filtered_flow_vectors

            yield (0, 4, "Loading flow inputs...")
            dp_tcyx = self._read_dp_tcyx(prob_path, dp_path)

            yield (1, 4, "Filtering flow vectors...")
            filtered_dp = compute_filtered_flow_vectors(dp_tcyx, params_snapshot)

            yield (2, 4, "Creating flow magnitude...")
            filtered_mag = np.sqrt(
                filtered_dp[:, 0] ** 2 + filtered_dp[:, 1] ** 2
            ).astype(np.float32)

            yield (3, 4, "Saving flow magnitude...")
            filtered_dp_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(filtered_dp_path), filtered_dp, compression="zlib")
            tifffile.imwrite(str(flow_mag_path), filtered_mag, compression="zlib")
            return filtered_mag

        self._set_stage_status("filtered_flow", "Creating flow magnitude...")
        self._set_ff_buttons_running(True)
        self._ff_worker = _worker()

    def _current_time_index(self, max_t: int) -> int:
        step = getattr(getattr(self.viewer, "dims", None), "current_step", ())
        if not step:
            return 0
        return min(max(int(step[0]), 0), max(max_t - 1, 0))


    def _params_from_ui(self):
        from cellflow.segmentation import FlowFollowingParams
        return FlowFollowingParams(
            median_kernel_time=int(self.ff_median_time_spin.value()),
            median_kernel_space=int(self.ff_median_space_spin.value()),
            gaussian_sigma_time=float(self.ff_gauss_time_spin.value()),
            gaussian_sigma_space=float(self.ff_gauss_space_spin.value()),
        )
"""Label correction widget for CellFlow v2."""
from __future__ import annotations

import logging
import os
from typing import Callable

import napari
import napari.layers
import numpy as np
from napari.utils.notifications import show_error
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours

from cellflow.correction.labels import (
    _label_at,
    draw_cell_path,
    erase_cell,
    clean_stranded_pixels,
    fill_label_holes,
    fix_label_semiholes,
    merge_cells,
    relabel_cell,
    split_across,
    split_draw,
    swap_labels,
)
from cellflow.napari.ui_style import (
    action_button,
    checked_success_button,
    danger_button,
    muted_label,
    status_label,
)

log = logging.getLogger("cellflow.correction")
if os.environ.get("CELLFLOW_DEBUG"):
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[cellflow.correction] %(levelname)s %(message)s"))
        log.addHandler(_h)

_DRAW_LAYER      = "CorrectionDraw"
_HIGHLIGHT_LAYER = "CellHighlight"
_SPOTLIGHT_LAYER = "CellSpotlight"
_SPOTLIGHT_OPACITY = 0.7
_SPOTLIGHT_SCALE = 3.0


class CorrectionWidget(QWidget):
    """Dock widget for interactive label correction."""

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: QWidget | None = None,
        *,
        show_activate_btn: bool = True,
        show_shortcuts: bool = True,
        inspector_first: bool = False,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._show_activate_btn = show_activate_btn
        self._show_shortcuts = show_shortcuts
        self._inspector_first = inspector_first

        self._layer: napari.layers.Labels | None = None

        self._selected_label: int = 0
        self._selected_pos = None
        self._selected_t: int = -1
        self._ctrl_click_first = None
        self._ctrl_click_first_label: int = 0
        self._ctrl_click_first_t: int = -1
        self._swap_first_pos = None
        self._swap_first_t: int = -1

        self._drag_callbacks: list = []
        self._bound_keys: list = []

        self._in_deactivate: bool = False

        self._saved_viewer_drag_cbs: list = []
        self._saved_layer_mode: str = "pan_zoom"
        self._saved_layer_contour: int = 0

        self._edit_callback: Callable[[int, set[int]], None] | None = None
        self._selection_callback: Callable[[int, int], None] | None = None

        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)

        self._activate_btn = QPushButton("Activate on selected layer")
        self._activate_btn.setCheckable(True)
        self._activate_btn.setToolTip(
            "Enable interactive mouse callbacks for merging/splitting."
        )
        action_button(self._activate_btn, expand=True)
        checked_success_button(self._activate_btn)
        self._activate_btn.clicked.connect(self._toggle_active)
        if self._show_activate_btn:
            root.addWidget(self._activate_btn)

        self._outline_btn = QPushButton("Show outlines only")
        self._outline_btn.setCheckable(True)
        self._outline_btn.setEnabled(False)
        action_button(self._outline_btn, expand=True)
        self._outline_btn.clicked.connect(self._toggle_outline)
        root.addWidget(self._outline_btn)

        self._reset_mode_btn = QPushButton("⚠  Restore correction mode")
        self._reset_mode_btn.setVisible(False)
        action_button(self._reset_mode_btn, expand=True)
        danger_button(self._reset_mode_btn)
        self._reset_mode_btn.clicked.connect(self._reset_tool_mode)
        root.addWidget(self._reset_mode_btn)

        cleanup_label = QLabel("Artifact cleanup")
        muted_label(cleanup_label, size_pt=9)
        root.addWidget(cleanup_label)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope:"))
        self._cleanup_scope_combo = QComboBox()
        self._cleanup_scope_combo.addItems(["Current frame", "All frames"])
        self._cleanup_scope_combo.setToolTip(
            "Choose whether cleanup applies to the visible frame or the full label stack."
        )
        scope_row.addWidget(self._cleanup_scope_combo)
        root.addLayout(scope_row)

        hole_row = QHBoxLayout()
        hole_row.addWidget(QLabel("Hole radius:"))
        self._hole_radius_spin = QSpinBox()
        self._hole_radius_spin.setRange(0, 999)
        self._hole_radius_spin.setValue(5)
        self._hole_radius_spin.setToolTip(
            "Maximum pixel distance for filling enclosed background gaps. Set to 0 to skip gap filling."
        )
        hole_row.addWidget(self._hole_radius_spin)
        root.addLayout(hole_row)

        semihole_row = QHBoxLayout()
        semihole_row.addWidget(QLabel("Max opening:"))
        self._semihole_opening_spin = QSpinBox()
        self._semihole_opening_spin.setRange(0, 999)
        self._semihole_opening_spin.setValue(3)
        self._semihole_opening_spin.setToolTip(
            "Maximum border contact, in pixels, for semihole repair. Set to 0 to skip semihole repair."
        )
        semihole_row.addWidget(self._semihole_opening_spin)
        root.addLayout(semihole_row)

        self._fill_holes_btn = QPushButton("Fill Holes")
        self._fill_holes_btn.setEnabled(False)
        self._fill_holes_btn.setToolTip("Fill enclosed background gaps using the configured hole radius.")
        action_button(self._fill_holes_btn, expand=True)
        self._fill_holes_btn.clicked.connect(self._fill_holes)
        root.addWidget(self._fill_holes_btn)

        self._fix_semiholes_btn = QPushButton("Fix Semiholes")
        self._fix_semiholes_btn.setEnabled(False)
        self._fix_semiholes_btn.setToolTip(
            "Repair narrow border-connected gaps using the radius and max opening controls."
        )
        action_button(self._fix_semiholes_btn, expand=True)
        self._fix_semiholes_btn.clicked.connect(self._fix_semiholes)
        root.addWidget(self._fix_semiholes_btn)

        self._clean_fragments_btn = QPushButton("Clean Fragments")
        self._clean_fragments_btn.setEnabled(False)
        self._clean_fragments_btn.setToolTip("Remove disconnected same-label fragments without filling background holes.")
        action_button(self._clean_fragments_btn, expand=True)
        self._clean_fragments_btn.clicked.connect(self._clean_fragments)
        root.addWidget(self._clean_fragments_btn)

        self._status = QLabel("Inactive")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_label(self._status, italic=True, muted=True)
        root.addWidget(self._status)

        inspect_group = QGroupBox("Inspect cell")
        inspect_lay = QVBoxLayout(inspect_group)

        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Cell ID:"))
        self._goto_cell_id = QSpinBox()
        self._goto_cell_id.setRange(0, 999_999)
        self._goto_cell_id.setValue(0)
        self._goto_cell_id.setSpecialValueText("—")
        id_row.addWidget(self._goto_cell_id)
        self._goto_btn = QPushButton("Go")
        self._goto_btn.setEnabled(False)
        action_button(self._goto_btn, expand=True)
        self._goto_btn.clicked.connect(self._goto_cell)
        id_row.addWidget(self._goto_btn)
        inspect_lay.addLayout(id_row)

        self._inspect_frames_label = QLabel("")
        self._inspect_frames_label.setWordWrap(True)
        muted_label(self._inspect_frames_label, size_pt=9)
        inspect_lay.addWidget(self._inspect_frames_label)

        ref_group = self.build_shortcuts_widget()

        if self._inspector_first:
            root.addWidget(inspect_group)
            if self._show_shortcuts:
                root.addWidget(ref_group)
        else:
            if self._show_shortcuts:
                root.addWidget(ref_group)
            root.addWidget(inspect_group)

        root.addStretch()

        attrib = QLabel(
            "Correction tools adapted from "
            '<a href="https://github.com/Image-Analysis-Hub/Epicure">Epicure</a>.'
            "<br>If you use these tools, please cite:<br>"
            '<a href="https://doi.org/10.64898/2026.03.27.714683">'
            "doi:10.64898/2026.03.27.714683</a>"
        )
        attrib.setOpenExternalLinks(True)
        attrib.setWordWrap(True)
        muted_label(attrib, size_pt=9)
        root.addWidget(attrib)

    def build_shortcuts_widget(self) -> QWidget:
        group = QGroupBox("Correction shortcuts")
        lay = QVBoxLayout(group)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(6)
        for key, desc in [
            ("Left-click",                         "Select / highlight cell"),
            ("Middle-click",                       "Erase clicked cell"),
            ("Delete",                             "Erase selected cell"),
            ("Ctrl+Left-click (cell selected)",    "Merge with clicked cell"),
            ("Ctrl+Left-click × 2 (same cell)",    "Split (watershed, 2 seeds)"),
            ("Right-click (cell selected)",         "Swap with clicked cell (same or other frame)"),
            ("Ctrl+Right-click (cell selected)",   "Swap with clicked cell (same frame)"),
            ("Ctrl+Right-click → Right-click",     "Swap (two-step, no selection)"),
            ("Ctrl-z",                             "Undo"),
            ("Shift+Left / Shift+Right",           "Previous / next cell across all frames"),
            ("Shift+Right-drag",                   "Split by drawn line"),
            ("Shift+Left-drag",                    "Draw cell path (extends or creates)"),
        ]:
            row = QWidget()
            row_lay = QVBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(1)

            key_lbl = QLabel(f"<tt>{key}</tt>")
            key_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

            row_lay.addWidget(key_lbl)
            row_lay.addWidget(desc_lbl)
            lay.addWidget(row)
        return group

    # ── activation ────────────────────────────────────────────────────────────

    def _toggle_active(self, checked: bool) -> None:
        if checked:
            layer = self.viewer.layers.selection.active
            if layer is None:
                self._activate_btn.setChecked(False)
                self._set_status("Select a Labels layer first", error=True)
                return
            if not isinstance(layer, napari.layers.Labels):
                self._activate_btn.setChecked(False)
                self._set_status("Not a Labels layer", error=True)
                return
            self._activate(layer)
        else:
            self._deactivate()

    def _activate(self, layer: napari.layers.Labels) -> None:
        log.debug("activate: layer='%s' shape=%s", layer.name, layer.data.shape)
        self._layer = layer
        self._selected_label = 0
        self._selected_pos = None
        self._selected_t = -1
        self._ctrl_click_first = None
        self._ctrl_click_first_label = 0
        self._ctrl_click_first_t = -1
        self._swap_first_pos = None
        self._swap_first_t = -1

        if hasattr(self.viewer, "mouse_drag_callbacks"):
            self._saved_viewer_drag_cbs = list(self.viewer.mouse_drag_callbacks)
            self.viewer.mouse_drag_callbacks.clear()
        else:
            self._saved_viewer_drag_cbs = []

        self._saved_layer_mode = layer.mode
        self._saved_layer_contour = int(layer.contour)
        layer.mode = "pan_zoom"

        self.viewer.layers.selection.active = layer
        self._get_draw_layer()
        self._get_spotlight_layer()
        self._get_highlight_layer()

        self.viewer.dims.events.current_step.connect(self._on_dims_change)
        layer.events.data.connect(self._on_layer_data_changed)
        layer.events.paint.connect(self._on_layer_data_changed)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        layer.events.mode.connect(self._on_layer_mode_change)

        self._register_callbacks()
        self._activate_btn.setText("Deactivate")
        self._outline_btn.setEnabled(True)
        self._set_cleanup_enabled(True)
        self._outline_btn.setChecked(True)
        self._toggle_outline(True)
        self._goto_btn.setEnabled(True)
        self._set_status(f"Active on '{layer.name}'")

    def _deactivate(self) -> None:
        if self._in_deactivate:
            return
        self._in_deactivate = True
        try:
            self._deactivate_impl()
        finally:
            self._in_deactivate = False

    def _deactivate_impl(self) -> None:
        log.debug("deactivate: layer='%s'", self._layer.name if self._layer else None)
        if self._layer is not None:
            self._remove_callbacks()

            for disconnect in [
                lambda: self.viewer.dims.events.current_step.disconnect(self._on_dims_change),
                lambda: self.viewer.layers.events.removed.disconnect(self._on_layer_removed),
                lambda: self._layer.events.data.disconnect(self._on_layer_data_changed),
                lambda: self._layer.events.paint.disconnect(self._on_layer_data_changed),
                lambda: self._layer.events.mode.disconnect(self._on_layer_mode_change),
            ]:
                try:
                    disconnect()
                except Exception:
                    pass

            try:
                self._layer.mode = self._saved_layer_mode
            except Exception:
                pass
            try:
                self._layer.contour = self._saved_layer_contour
            except Exception:
                pass

            if hasattr(self.viewer, "mouse_drag_callbacks"):
                self.viewer.mouse_drag_callbacks.clear()
                for cb in self._saved_viewer_drag_cbs:
                    self.viewer.mouse_drag_callbacks.append(cb)

        self._layer = None
        self._selected_label = 0
        self._selected_pos = None
        self._selected_t = -1
        self._ctrl_click_first = None
        self._ctrl_click_first_label = 0
        self._ctrl_click_first_t = -1
        self._swap_first_pos = None
        self._swap_first_t = -1
        self._saved_viewer_drag_cbs = []

        self._activate_btn.setText("Activate on selected layer")
        self._activate_btn.setChecked(False)
        self._outline_btn.setChecked(False)
        self._outline_btn.setEnabled(False)
        self._set_cleanup_enabled(False)
        self._goto_btn.setEnabled(False)
        self._goto_cell_id.setValue(0)
        self._inspect_frames_label.setText("")
        self._set_status("Inactive")
        self._cleanup_draw_layer()
        self._cleanup_highlight_layer()
        self._cleanup_spotlight_layer()

    def activate_layer(self, layer: napari.layers.Labels) -> None:
        """Activate correction on a specific Labels layer (bypasses the UI button)."""
        if self._layer is not None:
            self._deactivate()
        self._activate(layer)
        self._activate_btn.setChecked(True)

    def deactivate(self) -> None:
        """Deactivate correction (public API)."""
        self._deactivate()

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status.setText(msg)
        if error:
            self._status.setStyleSheet("color: #b00020; font-style: italic;")
        else:
            status_label(self._status, italic=True, muted=True)

    def _set_cleanup_enabled(self, enabled: bool) -> None:
        for button in (
            self._fill_holes_btn,
            self._fix_semiholes_btn,
            self._clean_fragments_btn,
        ):
            button.setEnabled(enabled)

    def set_edit_callback(self, fn: Callable[[int, set[int]], None] | None) -> None:
        """Register a callback fired after every successful edit.
        Signature: fn(t: int, changed_ids: set[int]) -> None.
        Pass None to clear."""
        self._edit_callback = fn

    def set_selection_callback(self, fn: Callable[[int, int], None] | None) -> None:
        """Register a callback fired when the selected label changes.

        Signature: fn(t: int, label: int) -> None.  ``label`` is 0 when the
        selection is cleared.
        """
        self._selection_callback = fn

    def select_label(self, t: int, label: int, *, notify: bool = True) -> None:
        """Select and highlight *label* at frame *t*."""
        self._update_highlight(t, label, notify=notify)

    def _cleanup_frame_indices(self) -> list[int]:
        if self._layer is None:
            return []
        if self._layer.data.ndim < 3:
            return [0]
        if self._cleanup_scope_combo.currentText() == "All frames":
            return list(range(int(self._layer.data.shape[0])))
        return [int(self.viewer.dims.current_step[0])]

    def _run_artifact_cleanup(
        self,
        operation_name: str,
        no_change_message: str,
        operation: Callable[[np.ndarray], None],
    ) -> None:
        if self._layer is None:
            self._set_status("No active labels layer", error=True)
            return
        try:
            changed_frames = 0
            changed_pixels = 0
            for t in self._cleanup_frame_indices():
                seg2d = self._frame_view(self._layer, t)
                before = seg2d.copy()
                operation(seg2d)
                changed = int(np.sum(before != seg2d))
                if not changed:
                    continue
                changed_frames += 1
                changed_pixels += changed
                self._record_history(self._layer, t, before)

            if changed_pixels:
                self._layer.refresh()
                current_t = (
                    int(self.viewer.dims.current_step[0])
                    if self._layer.data.ndim >= 3
                    else 0
                )
                if self._selected_label:
                    self._update_highlight(current_t, self._selected_label)
                self._set_status(
                    f"{operation_name} in {changed_frames} frame(s), {changed_pixels} px changed. Unsaved."
                )
            else:
                self._set_status(no_change_message)
        except Exception as exc:
            show_error(f"cleanup error: {exc}")

    def _fill_holes(self) -> None:
        radius = int(self._hole_radius_spin.value())
        self._run_artifact_cleanup(
            "Filled holes",
            "No holes found",
            lambda seg2d: np.copyto(seg2d, fill_label_holes(seg2d, radius=radius)),
        )

    def _fix_semiholes(self) -> None:
        radius = int(self._hole_radius_spin.value())
        max_opening = int(self._semihole_opening_spin.value())
        self._run_artifact_cleanup(
            "Fixed semiholes",
            "No semiholes found",
            lambda seg2d: np.copyto(
                seg2d,
                fix_label_semiholes(seg2d, radius=radius, max_opening=max_opening),
            ),
        )

    def _clean_fragments(self) -> None:
        self._run_artifact_cleanup(
            "Cleaned fragments",
            "No fragments found",
            lambda seg2d: clean_stranded_pixels(seg2d),
        )

    @staticmethod
    def _frame_view(layer, t: int) -> np.ndarray:
        """Return a 2D writable view of frame *t* (squeezes singleton leading dims)."""
        if layer.data.ndim == 2:
            return layer.data
        v = layer.data[t]
        while v.ndim > 2:
            if v.shape[0] != 1:
                raise ValueError(f"non-singleton dim in frame slice: shape={v.shape}")
            v = v[0]
        return v

    def _next_free_label(self) -> int:
        """Return the next unused label across the full active stack."""
        if self._layer is None:
            return 1
        return int(np.max(self._layer.data)) + 1

    def _record_history(self, layer, t: int, before: np.ndarray) -> None:
        """Push changed pixels in frame *t* onto napari's undo stack and fire edit callback.

        ``before`` is a 2D snapshot of the frame; supports 3D and 4D underlying layers.
        """
        after = self._frame_view(layer, t)
        changed = np.where(before != after)
        if not changed[0].size:
            return
        n = changed[0].size
        # Build undo indices matching layer.data.ndim. Prepend t, fill any
        # extra leading dims (e.g. Z=1) with zeros, then the 2D (y, x) coords.
        extra = layer.data.ndim - 1 - 2
        parts = [np.full(n, t, dtype=layer.data.dtype)]
        parts.extend(np.zeros(n, dtype=layer.data.dtype) for _ in range(extra))
        parts.extend(changed)
        layer._save_history((tuple(parts), before[changed], after[changed]))
        if self._edit_callback is not None:
            ids = set(int(v) for v in before[changed]) | set(int(v) for v in after[changed])
            ids.discard(0)
            if ids:
                try:
                    self._edit_callback(t, ids)
                except Exception:
                    import logging as _logging
                    _logging.getLogger("cellflow.correction").exception("edit_callback failed")

    # ── draw layer ────────────────────────────────────────────────────────────

    def _get_draw_layer(self):
        if _DRAW_LAYER in self.viewer.layers:
            return self.viewer.layers[_DRAW_LAYER]
        dl = self.viewer.add_shapes(
            name=_DRAW_LAYER,
            ndim=2,
            edge_color="yellow",
            edge_width=1,
            face_color="transparent",
        )
        dl.visible = False
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer
        return dl

    def _cleanup_draw_layer(self) -> None:
        if _DRAW_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_DRAW_LAYER])

    # ── highlight layer ───────────────────────────────────────────────────────

    def _get_highlight_layer(self):
        if _HIGHLIGHT_LAYER in self.viewer.layers:
            return self.viewer.layers[_HIGHLIGHT_LAYER]
        hl = self.viewer.add_shapes(
            name=_HIGHLIGHT_LAYER,
            ndim=2,
            edge_color="cyan",
            edge_width=2,
            face_color="transparent",
        )
        hl.visible = False
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer
        return hl

    def _get_spotlight_layer(self):
        if _SPOTLIGHT_LAYER in self.viewer.layers:
            return self.viewer.layers[_SPOTLIGHT_LAYER]
        spotlight = self.viewer.add_image(
            np.zeros((1, 1, 4), dtype=np.float32),
            name=_SPOTLIGHT_LAYER,
            rgb=True,
            blending="translucent",
        )
        spotlight.visible = False
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer
        return spotlight

    def _notify_selection_changed(self, t: int, lab: int, previous_label: int) -> None:
        if lab == previous_label or self._selection_callback is None:
            return
        try:
            self._selection_callback(t, lab)
        except Exception:
            import logging as _logging
            _logging.getLogger("cellflow.correction").exception("selection_callback failed")

    def _update_highlight(self, t: int, lab: int, *, notify: bool = True) -> None:
        """Redraw the cyan boundary for *lab* at time *t*. Pass 0 to clear."""
        previous_label = self._selected_label
        self._selected_label = lab
        self._selected_t = t if lab != 0 else -1
        hl = self._get_highlight_layer()
        if lab == 0 or self._layer is None:
            hl.data = []
            hl.visible = False
            self._clear_spotlight()
            if notify:
                self._notify_selection_changed(t, lab, previous_label)
            return
        seg2d = self._frame_view(self._layer, t)
        if not np.any(seg2d == lab):
            self._selected_label = 0
            self._selected_t = -1
            hl.data = []
            hl.visible = False
            self._clear_spotlight()
            if notify:
                self._notify_selection_changed(t, 0, previous_label)
            return
        mask = (seg2d == lab).astype(np.uint8)
        contours = find_contours(mask, level=0.5)
        if not contours:
            self._selected_label = 0
            self._selected_t = -1
            hl.data = []
            hl.visible = False
            self._clear_spotlight()
            if notify:
                self._notify_selection_changed(t, 0, previous_label)
            return
        self._update_spotlight(mask.astype(bool))
        contour = max(contours, key=len)
        hl.data = [contour]
        hl.shape_type = ["polygon"]
        hl.visible = True
        self.viewer.layers.selection.active = self._layer
        if notify:
            self._notify_selection_changed(t, lab, previous_label)

    def _cleanup_highlight_layer(self) -> None:
        if _HIGHLIGHT_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_HIGHLIGHT_LAYER])

    def _update_spotlight(self, mask: np.ndarray) -> None:
        spotlight = self._get_spotlight_layer()
        outer_mask = self._scaled_mask(mask, scale=_SPOTLIGHT_SCALE)
        ring = outer_mask & ~mask
        alpha = np.full(mask.shape, _SPOTLIGHT_OPACITY, dtype=np.float32)
        if np.any(ring):
            inner_dist = distance_transform_edt(~mask)
            outer_dist = distance_transform_edt(outer_mask)
            denom = inner_dist + outer_dist
            ramp = np.divide(
                inner_dist,
                denom,
                out=np.zeros_like(inner_dist, dtype=np.float64),
                where=denom > 0,
            )
            alpha[ring] = (ramp[ring] * _SPOTLIGHT_OPACITY).astype(np.float32)
        alpha[mask] = 0.0
        data = np.zeros(mask.shape + (4,), dtype=np.float32)
        data[..., 3] = alpha
        spotlight.data = data
        spotlight.visible = True
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer

    @staticmethod
    def _scaled_mask(mask: np.ndarray, *, scale: float) -> np.ndarray:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return np.zeros_like(mask, dtype=bool)
        center = coords.mean(axis=0)
        yy, xx = np.indices(mask.shape)
        src_y = np.rint(center[0] + (yy - center[0]) / scale).astype(int)
        src_x = np.rint(center[1] + (xx - center[1]) / scale).astype(int)
        np.clip(src_y, 0, mask.shape[0] - 1, out=src_y)
        np.clip(src_x, 0, mask.shape[1] - 1, out=src_x)
        return mask[src_y, src_x]

    def _clear_spotlight(self) -> None:
        if _SPOTLIGHT_LAYER in self.viewer.layers:
            spotlight = self.viewer.layers[_SPOTLIGHT_LAYER]
            spotlight.data = np.zeros((1, 1, 4), dtype=np.float32)
            spotlight.visible = False

    def _cleanup_spotlight_layer(self) -> None:
        if _SPOTLIGHT_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_SPOTLIGHT_LAYER])

    def _on_dims_change(self, event=None) -> None:
        if not (self._selected_label and self._layer is not None):
            return
        step = self.viewer.dims.current_step
        if self._layer.data.ndim < 3 or len(step) < self._layer.data.ndim:
            return
        t = int(step[0])
        if t >= self._layer.data.shape[0]:
            return
        selected_label = self._selected_label
        selected_pos = self._selected_pos
        selected_t = self._selected_t
        self._update_highlight(t, selected_label, notify=False)
        self._selected_label = selected_label
        self._selected_pos = selected_pos
        self._selected_t = selected_t

    def _on_layer_data_changed(self, event=None) -> None:
        if not (self._selected_label and self._layer is not None):
            return
        step = self.viewer.dims.current_step
        if self._layer.data.ndim < 3 or len(step) < self._layer.data.ndim:
            return
        t = int(step[0])
        if t >= self._layer.data.shape[0]:
            return
        self._update_highlight(t, self._selected_label)

    def _on_layer_mode_change(self, event=None) -> None:
        if self._layer is None:
            return
        mode = getattr(event, "value", None) or self._layer.mode
        log.debug("_on_layer_mode_change: mode=%s", mode)
        if mode != "pan_zoom":
            self._reset_mode_btn.setVisible(True)
            self._set_status("Tool mode changed — corrections disabled", error=True)
        else:
            self._reset_mode_btn.setVisible(False)
            if self._layer is not None:
                self._set_status(f"Active on '{self._layer.name}'")

    def _on_layer_removed(self, event=None) -> None:
        removed = getattr(event, "value", None)
        removed_name = getattr(removed, "name", None)
        if removed is self._layer or removed_name in (_DRAW_LAYER, _HIGHLIGHT_LAYER, _SPOTLIGHT_LAYER):
            log.debug("_on_layer_removed: '%s' removed, deactivating", removed_name)
            self._deactivate()

    def _reset_tool_mode(self) -> None:
        if self._layer is not None:
            self._layer.mode = "pan_zoom"

    # ── inspect cell ──────────────────────────────────────────────────────────

    def _goto_cell(self) -> None:
        lab = self._goto_cell_id.value()
        if lab == 0:
            step = self.viewer.dims.current_step
            t = int(step[0]) if (self._layer is not None and self._layer.data.ndim >= 3 and len(step) >= 1) else 0
            self._update_highlight(t, 0)
            self._inspect_frames_label.setText("")
            return
        if self._layer is None:
            return
        data = self._layer.data
        frames = [i for i in range(data.shape[0]) if np.any(data[i] == lab)]
        if not frames:
            self._inspect_frames_label.setText(f"Cell {lab} not found in any frame.")
            step = self.viewer.dims.current_step
            t = int(step[0]) if len(step) >= 1 else 0
            self._update_highlight(t, 0)
            return
        _MAX = 20
        if len(frames) <= _MAX:
            frames_str = ", ".join(str(f) for f in frames)
        else:
            shown = ", ".join(str(f) for f in frames[:_MAX])
            frames_str = f"{shown}, … ({len(frames)} frames total)"
        self._inspect_frames_label.setText(f"Frames: {frames_str}")
        step = self.viewer.dims.current_step
        t = int(step[0]) if len(step) >= 1 else 0
        self._update_highlight(t, lab)

    def _step_cell(self, direction: int) -> None:
        if self._layer is None:
            return
        data = self._layer.data
        ids = sorted(set(int(v) for v in np.unique(data)) - {0})
        if not ids:
            self._set_status("No cells in any frame")
            return
        cur = self._selected_label
        if direction > 0:
            nxt = next((i for i in ids if i > cur), ids[0])
        else:
            nxt = next((i for i in reversed(ids) if i < cur), ids[-1])
        frames = [i for i in range(data.shape[0]) if np.any(data[i] == nxt)]
        if frames:
            step = list(self.viewer.dims.current_step)
            step[0] = frames[0]
            self.viewer.dims.current_step = tuple(step)
        self._goto_cell_id.setValue(nxt)
        self._goto_cell()

    # ── callback registration ─────────────────────────────────────────────────

    def _register_callbacks(self) -> None:
        layer = self._layer

        def key_delete(_layer):
            try:
                if self._selected_label == 0:
                    self._set_status("No cell selected — left-click a cell first")
                    return
                t = int(self.viewer.dims.current_step[0])
                seg2d = self._frame_view(_layer, t)
                before = seg2d.copy()
                if erase_cell(seg2d, label=self._selected_label):
                    self._record_history(_layer, t, before)
                    _layer.refresh()
                    self._update_highlight(t, 0)
                    self._set_status(f"Erased — Active on '{_layer.name}'")
            except Exception as exc:
                show_error(f"delete error: {exc}")

        def key_prev_cell(_layer):
            self._step_cell(-1)

        def key_next_cell(_layer):
            self._step_cell(1)

        for key, fn in [
            ("Delete", key_delete),
            ("Shift-Left", key_prev_cell),
            ("Shift-Right", key_next_cell),
        ]:
            layer.bind_key(key, fn, overwrite=True)
            self._bound_keys.append(key)

        def on_drag(_layer, event):
            try:
                if event.type != "mouse_press":
                    return

                t   = int(self.viewer.dims.current_step[0])
                btn = event.button
                mods = {m.name for m in event.modifiers}

                seg2d = self._frame_view(_layer, t)
                pos   = _layer.world_to_data(event.position)
                log.debug(
                    "on_drag: btn=%s mods=%s t=%d selected=%s",
                    btn, mods, t, self._selected_label,
                )

                # Middle-click: erase clicked cell
                if btn == 3 and not mods:
                    lab = _label_at(seg2d, pos)
                    if lab == 0:
                        return
                    before = seg2d.copy()
                    if erase_cell(seg2d, label=lab):
                        self._record_history(_layer, t, before)
                        _layer.refresh()
                        if lab == self._selected_label:
                            self._update_highlight(t, 0)
                        self._set_status(f"Erased — Active on '{_layer.name}'")
                    return

                # Ctrl+Right-click: swap
                if btn == 2 and mods == {"Control"}:
                    lab = _label_at(seg2d, pos)
                    if lab == 0:
                        self._set_status("Swap — click on a cell (not background)")
                        return
                    if (
                        self._selected_label != 0
                        and self._selected_pos is not None
                        and lab != self._selected_label
                    ):
                        before = seg2d.copy()
                        ok = swap_labels(seg2d, self._selected_pos, pos)
                        if ok:
                            self._record_history(_layer, t, before)
                            _layer.refresh()
                            self._selected_label = 0
                            self._selected_pos = None
                            self._selected_t = -1
                            self._update_highlight(t, 0)
                            self._set_status(f"Swapped — Active on '{_layer.name}'")
                        else:
                            self._set_status("Swap failed — click on two different cells")
                    else:
                        self._swap_first_pos = pos
                        self._swap_first_t = t
                        self._set_status(f"Swap — label {lab} selected, right-click second cell")
                    return

                # Plain Right-click: complete two-step swap, or pass label across frames
                if btn == 2 and not mods:
                    if self._swap_first_pos is not None:
                        if t != self._swap_first_t:
                            self._swap_first_pos = None
                            self._swap_first_t = -1
                            self._set_status("Frame changed — swap cancelled")
                        else:
                            before = seg2d.copy()
                            ok = swap_labels(seg2d, self._swap_first_pos, pos)
                            if ok:
                                self._record_history(_layer, t, before)
                                _layer.refresh()
                                self._swap_first_pos = None
                                self._swap_first_t = -1
                                self._set_status(f"Swapped — Active on '{_layer.name}'")
                            else:
                                self._set_status("Swap failed — click on two different cells")
                                self._swap_first_pos = None
                                self._swap_first_t = -1
                    elif self._selected_label != 0 and self._selected_t != -1:
                        before = seg2d.copy()
                        if t != self._selected_t:
                            # Pass selected label to a cell in a different frame
                            ok = relabel_cell(seg2d, pos, self._selected_label)
                            msg_ok  = f"Relabelled → {self._selected_label} — Active on '{_layer.name}'"
                            msg_err = "Relabel failed — click on a different cell"
                        else:
                            # Swap selected cell with right-clicked cell in same frame
                            ok = swap_labels(seg2d, self._selected_pos, pos)
                            msg_ok  = f"Swapped — Active on '{_layer.name}'"
                            msg_err = "Swap failed — click on a different cell"
                        if ok:
                            self._record_history(_layer, t, before)
                            _layer.refresh()
                            self._set_status(msg_ok)
                        else:
                            self._set_status(msg_err)
                    return

                # Ctrl+Left-click: merge or split
                if btn == 1 and mods == {"Control"}:
                    lab = _label_at(seg2d, pos)
                    if lab == 0:
                        self._set_status("Click on a cell, not background")
                        return

                    if self._ctrl_click_first is not None:
                        if t != self._ctrl_click_first_t:
                            self._ctrl_click_first = pos
                            self._ctrl_click_first_label = lab
                            self._ctrl_click_first_t = t
                            self._update_highlight(t, lab)
                            self._set_status(f"Frame changed — restarted: label {lab} selected")
                        elif lab == self._ctrl_click_first_label:
                            before = seg2d.copy()
                            ok = split_across(
                                seg2d, self._image_frame(t),
                                self._ctrl_click_first, pos,
                                new_label=self._next_free_label(),
                            )
                            self._set_status(
                                f"Split — Active on '{_layer.name}'"
                                if ok else "Split failed — seeds too close or result too small"
                            )
                            if ok:
                                self._record_history(_layer, t, before)
                            _layer.refresh()
                            self._ctrl_click_first = None
                            self._ctrl_click_first_label = 0
                            self._ctrl_click_first_t = -1
                            self._update_highlight(t, _label_at(seg2d, pos))
                        else:
                            self._ctrl_click_first = None
                            self._ctrl_click_first_label = 0
                            self._ctrl_click_first_t = -1

                    if self._ctrl_click_first is None:
                        if (
                            self._selected_label != 0
                            and lab != self._selected_label
                            and np.any(seg2d == self._selected_label)
                        ):
                            before = seg2d.copy()
                            ok = merge_cells(
                                seg2d, pos, pos,
                                label_a=lab, label_b=self._selected_label,
                            )
                            self._set_status(
                                f"Merged — Active on '{_layer.name}'"
                                if ok else "Merge failed — labels not touching"
                            )
                            if ok:
                                self._record_history(_layer, t, before)
                            _layer.refresh()
                            self._selected_label = 0
                            self._selected_pos = None
                            self._selected_t = -1
                            self._update_highlight(t, _label_at(seg2d, pos))
                        else:
                            self._ctrl_click_first = pos
                            self._ctrl_click_first_label = lab
                            self._ctrl_click_first_t = t
                            self._update_highlight(t, lab)
                            self._set_status(
                                f"Label {lab} — Ctrl+click same cell again for second split seed"
                            )
                    return

                # Plain Left-click: select / highlight cell
                if btn == 1 and not mods:
                    self._ctrl_click_first = None
                    self._ctrl_click_first_label = 0
                    self._ctrl_click_first_t = -1
                    self._swap_first_pos = None
                    self._swap_first_t = -1
                    lab = _label_at(seg2d, pos)
                    self._selected_pos = pos if lab != 0 else None
                    self._selected_t = t if lab != 0 else -1
                    self._update_highlight(t, lab)
                    if lab:
                        self._set_status(f"Selected label {lab} — Active on '{_layer.name}'")
                    else:
                        self._set_status(f"Active on '{_layer.name}'")
                    return

                # Shift+Right-drag: split by drawn line
                if mods == {"Shift"} and btn == 2:
                    dl = self._get_draw_layer()
                    dl.data = []
                    dl.visible = True
                    pos_list = [_layer.world_to_data(event.position)]
                    yield
                    while event.type == "mouse_move":
                        pos_list.append(_layer.world_to_data(event.position))
                        if len(pos_list) % 3 == 0:
                            dl.data = [np.array([[p[-2], p[-1]] for p in pos_list])]
                            dl.shape_type = ["path"]
                        yield
                    pos_list.append(_layer.world_to_data(event.position))
                    dl.data = []
                    dl.visible = False
                    self.viewer.layers.selection.active = _layer
                    curlabel = self._selected_label if self._selected_label else None
                    before = seg2d.copy()
                    ok = split_draw(
                        seg2d,
                        pos_list,
                        curlabel=curlabel,
                        new_label=self._next_free_label(),
                    )
                    self._set_status(
                        f"Split — Active on '{_layer.name}'"
                        if ok else "Split draw failed — line did not divide the cell"
                    )
                    if ok:
                        self._record_history(_layer, t, before)
                    _layer.refresh()
                    self._update_highlight(t, self._selected_label)
                    return

                # Shift+Left-drag: draw cell path
                if mods == {"Shift"} and btn == 1:
                    dl = self._get_draw_layer()
                    dl.data = []
                    dl.visible = True
                    pos_list = [_layer.world_to_data(event.position)]
                    yield
                    while event.type == "mouse_move":
                        pos_list.append(_layer.world_to_data(event.position))
                        if len(pos_list) % 3 == 0:
                            dl.data = [np.array([[p[-2], p[-1]] for p in pos_list])]
                            dl.shape_type = ["path"]
                        yield
                    pos_list.append(_layer.world_to_data(event.position))
                    dl.data = []
                    dl.visible = False
                    self.viewer.layers.selection.active = _layer
                    curlabel = self._selected_label if self._selected_label else None
                    before = seg2d.copy()
                    ok = draw_cell_path(
                        seg2d,
                        pos_list,
                        curlabel=curlabel,
                        new_label=self._next_free_label(),
                    )
                    self._set_status(
                        f"Drew cell path — Active on '{_layer.name}'"
                        if ok else "Draw failed — stroke too short"
                    )
                    if ok:
                        self._record_history(_layer, t, before)
                    _layer.refresh()
                    self._update_highlight(t, self._selected_label)
                    return

            except Exception as exc:
                import traceback
                show_error(f"Correction error: {exc}\n{traceback.format_exc()}")

        layer.mouse_drag_callbacks.append(on_drag)
        self._drag_callbacks.append(on_drag)

    def _remove_callbacks(self) -> None:
        layer = self._layer
        for fn in self._drag_callbacks:
            try:
                layer.mouse_drag_callbacks.remove(fn)
            except ValueError:
                pass
        self._drag_callbacks.clear()
        for key in self._bound_keys:
            try:
                layer.bind_key(key, None)
            except Exception:
                pass
        self._bound_keys.clear()

    def _toggle_outline(self, checked: bool) -> None:
        if self._layer is None:
            self._outline_btn.setChecked(False)
            return
        self._layer.contour = 1 if checked else 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _image_frame(self, t: int) -> np.ndarray | None:
        """Return the intensity image at frame *t* from the first Image layer found.

        Squeezes singleton leading dims so the result is always 2D."""
        for lyr in self.viewer.layers:
            if getattr(lyr, "name", None) == _SPOTLIGHT_LAYER:
                continue
            if isinstance(lyr, napari.layers.Image):
                d = lyr.data
                if d.ndim == 2:
                    return d
                v = d[t] if d.ndim >= 3 else d
                while v.ndim > 2:
                    if v.shape[0] != 1:
                        return None
                    v = v[0]
                return v
        return None
"""Main widget for the CellFlow napari plugin."""
from __future__ import annotations

import json
from pathlib import Path

import napari
from qtpy.QtCore import Qt, QSize, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.analysis_widget import AnalysisWidget
from cellflow.napari.cell_workflow_widget import CellWorkflowWidget
from cellflow.napari.cellpose_widget import CellposeWidget
from cellflow.napari.data_panel_widget import ProjectStatusPanel
from cellflow.napari.data_prep_widget import DataPrepWidget
from cellflow.napari.meta_widget import MetaSourceBrowserWidget
from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget
from cellflow.napari.nls_classification_widget import NLSClassificationWidget
from cellflow.napari.widgets import CollapsibleSection
from cellflow.napari.ui_style import icon_button, muted_label, tiny_button


class CellFlowMainWidget(QWidget):
    """The unified workflow-based UI for CellFlow."""

    refresh_requested = Signal(object)  # emits pos_dir: Path | None

    def __init__(self, napari_viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = napari_viewer

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── Project Info (Top Level) ──────────────────────────────────
        self._setup_project_ui(main_layout)

        # Main scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(2, 2, 2, 2)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.scroll_widget)

        main_layout.addWidget(self.scroll)

        # Add sections
        self.data_panel = ProjectStatusPanel(self.viewer)
        self.data_section = CollapsibleSection(
            "Project Status", self.data_panel, expanded=False, title_color="#ADD8E6"
        )

        self._data_prep_widget = DataPrepWidget(self.viewer, self)
        self.prep_section = CollapsibleSection(
            "1. Data Preparation", self._data_prep_widget, expanded=False
        )

        self._cellpose_widget = CellposeWidget(self.viewer)
        self.cellpose_section = CollapsibleSection(
            "2. Cellpose", self._cellpose_widget, expanded=False
        )
        self.hpc_cellpose_widget = self._cellpose_widget.hpc_cellpose_widget

        self.nucleus_workflow_widget = NucleusWorkflowWidget(self.viewer)
        self.nucleus_section = CollapsibleSection(
            "3. Nucleus Segmentation & Tracking", self.nucleus_workflow_widget, expanded=False
        )

        self.cell_workflow_widget = CellWorkflowWidget(self.viewer)
        self.cell_section = CollapsibleSection(
            "4. Cell Segmentation", self.cell_workflow_widget, expanded=False
        )
        self._connect_label_selection_sync()

        self.analysis_widget = AnalysisWidget(self.viewer)
        self.analysis_section = CollapsibleSection(
            "5. Analysis", self.analysis_widget, expanded=False
        )

        self.nls_classification_widget = NLSClassificationWidget(self.viewer)
        self.nls_classification_section = CollapsibleSection(
            "5b. NLS Classification", self.nls_classification_widget, expanded=False
        )

        self.meta_source_browser = MetaSourceBrowserWidget(self.viewer)
        self.meta_section = CollapsibleSection(
            "6. Meta Analyzer", self.meta_source_browser, expanded=False
        )

        self.scroll_layout.addWidget(self.data_section)
        self.scroll_layout.addWidget(self.prep_section)
        self.scroll_layout.addWidget(self.cellpose_section)
        self.scroll_layout.addWidget(self.nucleus_section)
        self.scroll_layout.addWidget(self.cell_section)
        self.scroll_layout.addWidget(self.analysis_section)
        self.scroll_layout.addWidget(self.nls_classification_section)
        self.scroll_layout.addWidget(self.meta_section)

        # Add stretch at the end
        self.scroll_layout.addStretch()

        # Connect signals
        self.project_btn.clicked.connect(lambda: self._on_set_project_directory())
        self.save_btn.clicked.connect(lambda: self._on_save_config())
        self.save_as_btn.clicked.connect(lambda: self._on_save_config_as())
        self.load_btn.clicked.connect(lambda: self._on_load_config())
        self.load_from_btn.clicked.connect(lambda: self._on_load_config_from())
        
        self.refresh_btn.clicked.connect(lambda: self._refresh_all())
        self.pos_spin.valueChanged.connect(lambda: self._refresh_all())

    def _connect_label_selection_sync(self) -> None:
        """Synchronize selected cell/nucleus IDs across correction widgets."""
        if hasattr(self.nucleus_workflow_widget, "set_selection_callback"):
            self.nucleus_workflow_widget.set_selection_callback(
                lambda t, label: self.cell_workflow_widget.select_matching_cell_label(t, label)
            )
        if hasattr(self.cell_workflow_widget, "set_selection_callback"):
            self.cell_workflow_widget.set_selection_callback(
                lambda t, label: self.nucleus_workflow_widget.select_matching_nucleus_label(t, label)
            )

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(int(hint.width() * 1.5), hint.height())

    def _setup_project_ui(self, layout: QVBoxLayout) -> None:
        """Create the top-level project metadata and buttons."""
        proj_widget = QWidget()
        proj_lay = QVBoxLayout(proj_widget)
        proj_lay.setContentsMargins(0, 0, 0, 0)
        proj_lay.setSpacing(4)

        # Row 1: Metadata
        meta_row = QHBoxLayout()
        meta_row.setSpacing(4)
        
        meta_row.addWidget(QLabel("px:"))
        self.px_edit = QLineEdit()
        self.px_edit.setFixedWidth(40)
        meta_row.addWidget(self.px_edit)

        meta_row.addWidget(QLabel("dt:"))
        self.dt_edit = QLineEdit()
        self.dt_edit.setFixedWidth(40)
        meta_row.addWidget(self.dt_edit)

        meta_row.addWidget(QLabel("C:"))
        self.cond_edit = QLineEdit()
        meta_row.addWidget(self.cond_edit)

        meta_row.addWidget(QLabel("P:"))
        self.pos_spin = QSpinBox()
        self.pos_spin.setRange(0, 99)
        self.pos_spin.setFixedWidth(40)
        meta_row.addWidget(self.pos_spin)
        
        self.refresh_btn = QPushButton("↺")
        icon_button(self.refresh_btn)
        self.refresh_btn.setToolTip("Refresh all status")
        meta_row.addWidget(self.refresh_btn)
        
        proj_lay.addLayout(meta_row)

        # Row 2: Project Actions
        project_row = QHBoxLayout()
        project_row.setSpacing(4)
        self.project_btn = QPushButton("Project Directory...")
        tiny_button(self.project_btn)
        project_row.addWidget(self.project_btn)
        proj_lay.addLayout(project_row)

        # Row 3: Config Actions
        config_row = QHBoxLayout()
        config_row.setSpacing(4)
        self.save_btn = QPushButton("Save Config")
        self.save_as_btn = QPushButton("Save Config As...")
        self.load_btn = QPushButton("Load Config")
        self.load_from_btn = QPushButton("Load Config From...")
        
        for btn in (self.save_btn, self.save_as_btn, self.load_btn, self.load_from_btn):
            tiny_button(btn)
            config_row.addWidget(btn)
        proj_lay.addLayout(config_row)

        # Row 4: Path Label
        self.path_label = QLabel("[no project]")
        muted_label(self.path_label)
        self.path_label.setWordWrap(True)
        proj_lay.addWidget(self.path_label)

        layout.addWidget(proj_widget)

    def _on_set_project_directory(self) -> None:
        """Set the project directory and load config if present."""
        path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if path:
            p = Path(path)
            self.path_label.setText(str(p))
            self.path_label.setToolTip(str(p))
            
            # Look for config file
            config_path = p / "cellflow_config.json"
            if config_path.exists():
                self._load_config(str(config_path))
            
            self._refresh_all()

    def get_state(self) -> dict:
        """Return the current UI state as a dictionary."""
        return {
            "metadata": {
                "pixel_size_um": self.px_edit.text(),
                "time_interval_s": self.dt_edit.text(),
                "condition": self.cond_edit.text(),
                "position": self.pos_spin.value(),
            },
            "data_prep": self._data_prep_widget.get_state(),
            "hpc_cellpose": self.hpc_cellpose_widget.get_state(),
            "nucleus": self.nucleus_workflow_widget.get_state(),
            "cell": self.cell_workflow_widget.get_state(),
        }

    def set_state(self, state: dict) -> None:
        """Update the UI state from a dictionary."""
        if "metadata" in state:
            m = state["metadata"]
            if "pixel_size_um" in m: self.px_edit.setText(str(m["pixel_size_um"]))
            if "time_interval_s" in m: self.dt_edit.setText(str(m["time_interval_s"]))
            if "condition" in m: self.cond_edit.setText(str(m["condition"]))
            if "position" in m: self.pos_spin.setValue(int(m["position"]))

        if "data_prep" in state:
            self._data_prep_widget.set_state(state["data_prep"])

        if "hpc_cellpose" in state:
            self.hpc_cellpose_widget.set_state(state["hpc_cellpose"])
        
        if "nucleus" in state:
            self.nucleus_workflow_widget.set_state(state["nucleus"])
        
        if "cell" in state:
            self.cell_workflow_widget.set_state(state["cell"])

    def _on_save_config(self) -> None:
        """Save current configuration to project directory."""
        path_text = self.path_label.text()
        if not path_text or path_text == "[no project]":
            return
        
        config_path = Path(path_text) / "cellflow_config.json"
        self._save_config(str(config_path))

    def _on_save_config_as(self) -> None:
        """Save current configuration to a specific file."""
        path = QFileDialog.getSaveFileName(self, "Save Config As", filter="JSON (*.json)")[0]
        if path:
            self._save_config(path)

    def _on_load_config(self) -> None:
        """Load configuration from project directory."""
        path_text = self.path_label.text()
        if not path_text or path_text == "[no project]":
            return
        
        config_path = Path(path_text) / "cellflow_config.json"
        if config_path.exists():
            self._load_config(str(config_path))
        else:
            print(f"Config not found: {config_path}")

    def _on_load_config_from(self) -> None:
        """Load configuration from a specific file."""
        path = QFileDialog.getOpenFileName(self, "Load Config From", filter="JSON (*.json)")[0]
        if path:
            self._load_config(path)

    def _save_config(self, path: str) -> None:
        """Save state to a JSON file."""
        state = self.get_state()
        try:
            with open(path, "w") as f:
                json.dump(state, f, indent=4)
            print(f"Config saved to {path}")
        except Exception as e:
            print(f"Error saving config: {e}")

    def _load_config(self, path: str) -> None:
        """Load state from a JSON file."""
        try:
            with open(path, "r") as f:
                state = json.load(f)
            self.set_state(state)
            print(f"Config loaded from {path}")
        except Exception as e:
            print(f"Error loading config: {e}")

    def _refresh_all(self) -> None:
        """Refresh file status in all child widgets."""
        path_text = self.path_label.text()
        if path_text and path_text != "[no project]":
            pos = self.pos_spin.value()
            pos_dir = Path(path_text) / f"pos{pos:02d}"
        else:
            pos_dir = None

        self.data_panel.refresh(pos_dir)
        self._data_prep_widget.refresh(pos_dir)
        self._cellpose_widget.refresh(pos_dir)
        self.nucleus_workflow_widget.refresh(pos_dir)
        self.cell_workflow_widget.refresh(pos_dir)
        self.analysis_widget.refresh(pos_dir)
        self.nls_classification_widget.refresh(pos_dir)
        project_root = Path(path_text) if path_text and path_text != "[no project]" else None
        self.meta_source_browser.refresh(project_root)
        # Emit signal for other widgets
        self.refresh_requested.emit(pos_dir)
"""Nucleus workflow widget for hypothesis generation and tracking in CellFlow v2."""
from __future__ import annotations

import logging
import os
import pickle
import shlex
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from napari.utils.colormaps import direct_colormap
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cellflow.correction.labels import best_overlapping_label
from cellflow.database.tracked import (
    read_full_tracked_stack,
    write_tracked_frame,
)
from cellflow.database.validation import (
    invalidate_track,
    is_track_validated,
    is_validated,
    read_validated_cells_at_frame,
    read_validated_frames,
    read_validated_tracks,
    remap_validated_tracks,
    validate_track,
)
from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.napari.ui_style import (
    action_button,
    add_block_button_row,
    add_block_checkbox_row,
    add_block_pair_row,
    add_parameter_grid_row,
    block_grid,
    compact_spinbox,
    danger_button,
    muted_label,
    status_label,
)
from cellflow.segmentation import ContourWatershedParams, compute_contour_watershed
from cellflow.tracking.retracker import retrack_frame_constrained
from cellflow.tracking_ultrack.config import TrackingConfig as UltrackConfig
from cellflow.tracking_ultrack.db_build import build_ultrack_database
from cellflow.tracking_ultrack.export import export_tracked_labels
from cellflow.tracking_ultrack.ingest import _select_solver
from cellflow.tracking_ultrack.extend import extend_track, extend_track_from_db
from cellflow.tracking_ultrack.solve import database_has_annotations, run_solve

logger = logging.getLogger(__name__)

try:
    from ultrack.core.segmentation.processing import segment as _ultrack_segment
except ImportError:
    _ultrack_segment = None  # type: ignore[assignment]

_PREVIEW_LAYER = "Preview: Nucleus"
_HYP_LAYER = "Hypothesis: Nucleus"
_TRACKED_LAYER = "Tracked: Nucleus"
_VALIDATED_OVERLAY = "Validated: Nucleus"
_SPOTLIGHT_LAYER = "CellSpotlight"
_VALIDATED_OVERLAY_OPACITY = 0.4
_CONTOUR_LAYER = "Contour Map: Nucleus"
_CELLPROB_LAYER = "Cellprob Map: Nucleus"
_FOREGROUND_SCORE_LAYER = "Foreground Score: Nucleus"
_FOREGROUND_MASK_LAYER = "Foreground Mask: Nucleus"
_CELL_ZAVG_LAYER = "Cell z-avg"
_NUC_ZAVG_LAYER = "Nucleus z-avg"
_ULTRACK_DB_PREVIEW_LAYER = "Ultrack DB Preview"
_ULTRACK_DB_SELECTION_LAYER = "Ultrack DB Selection"
_ULTRACK_DB_ANNOTATION_LAYER = "Ultrack DB Annotations"
_CONTOUR_SWEEP_WIDTH = 60
_CONTOUR_SWEEP_MIN_WIDTH = int(_CONTOUR_SWEEP_WIDTH * 0.9)


@dataclass(frozen=True)
class _HierarchyCutState:
    node_ids: tuple[int, ...]
    height: float | None


class NucleusWorkflowWidget(QWidget):
    """Nucleus hypothesis generation and tracking management."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._stop_flag: bool = False
        self._build_worker = None
        self._sweep_worker = None
        self._ultrack_db_preview_cache: dict[
            tuple,
            tuple[np.ndarray, str]
            | tuple[np.ndarray, str, dict[int, float]]
            | tuple[
                np.ndarray,
                str,
                dict[int, float],
                dict[int, int],
                dict[int, int],
            ],
        ] = {}
        self._ultrack_db_height_values_cache: dict[tuple, tuple[float, ...]] = {}
        self._ultrack_db_cut_state_cache: dict[tuple, tuple[_HierarchyCutState, ...]] = {}
        self._ultrack_db_browser_active: bool = False
        self._ultrack_db_frame_initialized: bool = False
        self._ultrack_db_selected_node_id: int | None = None
        self._ultrack_db_selected_frame: int | None = None
        self._ultrack_db_label_to_node_id: dict[int, int] = {}
        self._ultrack_db_node_id_to_label: dict[int, int] = {}
        self._ultrack_db_node_annotations: dict[int, str] = {}
        self._ultrack_db_preview_labels: np.ndarray | None = None
        self._ultrack_db_preview_mouse_callback = None
        self._setup_ui()
        self._connect_signals()

    # ──────────────────────────────────────────────────────────────────────────
    # UI setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        # ── Compact layout helpers ────────────────────────────────────────
        SPIN_MAX_W = 70

        def _compact(spin, w=SPIN_MAX_W):
            return compact_spinbox(spin, w)

        def _stage_files(group_label: str, entries: list[tuple[str, str]]) -> PipelineFilesWidget:
            return PipelineFilesWidget([(group_label, entries)], viewer=self.viewer)

        def _stage_status() -> QLabel:
            label = QLabel("")
            label.setWordWrap(True)
            label.setVisible(False)
            status_label(label)
            return label

        def _stage_progress() -> QProgressBar:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setVisible(False)
            return bar

        def _param_grid():
            grid = block_grid(horizontal_spacing=12, vertical_spacing=4)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)
            return grid

        def _param_group_label(text: str) -> QLabel:
            label = QLabel(text)
            label.setStyleSheet("font-weight: 600;")
            return label

        # ── 1. Contour Maps ───────────────────────────────────────────────
        _contour_inner = QWidget()
        contour_lay = QVBoxLayout(_contour_inner)
        contour_lay.setContentsMargins(4, 4, 4, 4)
        contour_lay.setSpacing(4)
        contour_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        cp_params_scroll = QScrollArea()
        cp_params_scroll.setWidgetResizable(True)
        cp_params_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        cp_params_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        cp_params_scroll.setFrameShape(QFrame.NoFrame)
        cp_params_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        cp_params_widget = QWidget()
        cp_params_widget.setMinimumWidth(520)
        cp_params_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        cp_params_lay = QVBoxLayout(cp_params_widget)
        cp_params_lay.setContentsMargins(0, 0, 0, 0)
        cp_params_lay.setSpacing(4)
        cp_params_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.contour_input_files = _stage_files("Inputs", [
            ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
            ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
        ])
        cp_params_lay.addWidget(self.contour_input_files)

        self.cp_min_spin = QDoubleSpinBox()
        self.cp_min_spin.setRange(-20.0, 20.0)
        self.cp_min_spin.setValue(-3.0)
        self.cp_min_spin.setDecimals(1)
        self.cp_min_spin.setSingleStep(1.0)
        self.cp_min_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_min_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.cp_max_spin = QDoubleSpinBox()
        self.cp_max_spin.setRange(-20.0, 20.0)
        self.cp_max_spin.setValue(0.0)
        self.cp_max_spin.setDecimals(1)
        self.cp_max_spin.setSingleStep(1.0)
        self.cp_max_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_max_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.cp_step_spin = QDoubleSpinBox()
        self.cp_step_spin.setRange(0.1, 10.0)
        self.cp_step_spin.setValue(1.0)
        self.cp_step_spin.setDecimals(1)
        self.cp_step_spin.setSingleStep(0.5)
        self.cp_step_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_step_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.contour_flow_threshold_spin = QDoubleSpinBox()
        self.contour_flow_threshold_spin.setRange(0.0, 10.0)
        self.contour_flow_threshold_spin.setValue(0.0)
        self.contour_flow_threshold_spin.setDecimals(2)
        self.contour_flow_threshold_spin.setSingleStep(0.1)
        self.contour_flow_threshold_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.contour_flow_threshold_spin.setToolTip(
            "Cellpose flow error threshold passed to compute_masks. 0 disables filtering."
        )
        self.contour_flow_threshold_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self.cp_gamma_min_spin = QDoubleSpinBox()
        self.cp_gamma_min_spin.setRange(0.05, 5.0)
        self.cp_gamma_min_spin.setValue(1.0)
        self.cp_gamma_min_spin.setDecimals(2)
        self.cp_gamma_min_spin.setSingleStep(0.05)
        self.cp_gamma_min_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_gamma_min_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.cp_gamma_max_spin = QDoubleSpinBox()
        self.cp_gamma_max_spin.setRange(0.05, 5.0)
        self.cp_gamma_max_spin.setValue(1.0)
        self.cp_gamma_max_spin.setDecimals(2)
        self.cp_gamma_max_spin.setSingleStep(0.05)
        self.cp_gamma_max_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_gamma_max_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.cp_gamma_step_spin = QDoubleSpinBox()
        self.cp_gamma_step_spin.setRange(0.05, 2.0)
        self.cp_gamma_step_spin.setValue(0.25)
        self.cp_gamma_step_spin.setDecimals(2)
        self.cp_gamma_step_spin.setSingleStep(0.05)
        self.cp_gamma_step_spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
        self.cp_gamma_step_spin.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        _gamma_tip = (
            "Gamma correction on Cellpose probability logits before boundary building. "
            "<1 boosts dim signals; >1 suppresses them. 1.0 = no correction. "
            "Contour maps are averaged over all gamma values in [min, max]."
        )
        for _w in (self.cp_gamma_min_spin, self.cp_gamma_max_spin, self.cp_gamma_step_spin):
            _w.setToolTip(_gamma_tip)
        self.contour_fg_threshold_spin = QDoubleSpinBox()
        self.contour_fg_threshold_spin.setRange(0.0, 1.0)
        self.contour_fg_threshold_spin.setValue(0.5)
        self.contour_fg_threshold_spin.setDecimals(2)
        self.contour_fg_threshold_spin.setSingleStep(0.01)
        self.contour_fg_threshold_spin.setToolTip(
            "Threshold applied to the fuzzy foreground score written by Contour Maps"
        )
        self.save_source_check = QCheckBox("Save label images")
        self.save_source_check.setToolTip("Save all label images used for contour building in 2_nucleus/source_labels/")
        self.save_source_check.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        contour_sweep_grid = _param_grid()
        add_parameter_grid_row(contour_sweep_grid, 0, 0, "Cellprob min:", self.cp_min_spin)
        add_parameter_grid_row(contour_sweep_grid, 0, 1, "Cellprob max:", self.cp_max_spin)
        add_parameter_grid_row(contour_sweep_grid, 1, 0, "Cellprob step:", self.cp_step_spin)
        add_parameter_grid_row(contour_sweep_grid, 1, 1, "Flow threshold:", self.contour_flow_threshold_spin)
        cp_params_lay.addWidget(_param_group_label("Cellpose mask sweep"))
        cp_params_lay.addLayout(contour_sweep_grid)

        contour_gamma_grid = _param_grid()
        add_parameter_grid_row(contour_gamma_grid, 0, 0, "Gamma min:", self.cp_gamma_min_spin)
        add_parameter_grid_row(contour_gamma_grid, 0, 1, "Gamma max:", self.cp_gamma_max_spin)
        add_parameter_grid_row(contour_gamma_grid, 1, 0, "Gamma step:", self.cp_gamma_step_spin)
        cp_params_lay.addWidget(_param_group_label("Gamma averaging"))
        cp_params_lay.addLayout(contour_gamma_grid)

        contour_output_grid = _param_grid()
        add_parameter_grid_row(contour_output_grid, 0, 0, "FG threshold:", self.contour_fg_threshold_spin)
        contour_output_grid.addWidget(
            self.save_source_check,
            0,
            2,
            1,
            2,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        cp_params_lay.addWidget(_param_group_label("Foreground output"))
        cp_params_lay.addLayout(contour_output_grid)
        for spin in (
            self.cp_min_spin,
            self.cp_max_spin,
            self.cp_step_spin,
            self.contour_flow_threshold_spin,
            self.cp_gamma_min_spin,
            self.cp_gamma_max_spin,
            self.cp_gamma_step_spin,
            self.contour_fg_threshold_spin,
        ):
            spin.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )

        self.preview_contour_btn = QPushButton("Preview")
        self.preview_contour_btn.setToolTip(
            "Build contour maps for the current frame only and display in napari"
        )
        self.build_btn = QPushButton("Build")
        self.contour_terminal_btn = QPushButton("Run in Terminal")
        self.cancel_build_btn = QPushButton("Cancel")
        self.cancel_build_btn.setEnabled(False)

        for button in (
            self.preview_contour_btn,
            self.build_btn,
            self.contour_terminal_btn,
            self.cancel_build_btn,
        ):
            button.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )

        contour_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(
            contour_btn_row,
            0,
            self.preview_contour_btn,
            self.build_btn,
            self.contour_terminal_btn,
            self.cancel_build_btn,
        )
        cp_params_lay.addLayout(contour_btn_row)

        contour_filter_grid = _param_grid()
        self.contour_filter_median_time_spin = QSpinBox()
        self.contour_filter_median_time_spin.setRange(1, 15)
        self.contour_filter_median_time_spin.setValue(1)
        self.contour_filter_median_time_spin.setSingleStep(1)
        self.contour_filter_median_space_spin = QSpinBox()
        self.contour_filter_median_space_spin.setRange(1, 15)
        self.contour_filter_median_space_spin.setValue(1)
        self.contour_filter_median_space_spin.setSingleStep(1)
        self.contour_filter_gauss_time_spin = QDoubleSpinBox()
        self.contour_filter_gauss_time_spin.setRange(0.0, 10.0)
        self.contour_filter_gauss_time_spin.setValue(0.0)
        self.contour_filter_gauss_time_spin.setDecimals(1)
        self.contour_filter_gauss_time_spin.setSingleStep(0.1)
        self.contour_filter_gauss_space_spin = QDoubleSpinBox()
        self.contour_filter_gauss_space_spin.setRange(0.0, 10.0)
        self.contour_filter_gauss_space_spin.setValue(0.0)
        self.contour_filter_gauss_space_spin.setDecimals(1)
        self.contour_filter_gauss_space_spin.setSingleStep(0.1)
        for spin in (
            self.contour_filter_median_time_spin,
            self.contour_filter_median_space_spin,
            self.contour_filter_gauss_time_spin,
            self.contour_filter_gauss_space_spin,
        ):
            spin.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            spin.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        add_parameter_grid_row(contour_filter_grid, 0, 0, "Median t kernel:", self.contour_filter_median_time_spin)
        add_parameter_grid_row(contour_filter_grid, 0, 1, "Median xy kernel:", self.contour_filter_median_space_spin)
        add_parameter_grid_row(contour_filter_grid, 1, 0, "Gaussian t sigma:", self.contour_filter_gauss_time_spin)
        add_parameter_grid_row(contour_filter_grid, 1, 1, "Gaussian xy sigma:", self.contour_filter_gauss_space_spin)
        for spin in (
            self.contour_filter_median_time_spin,
            self.contour_filter_median_space_spin,
            self.contour_filter_gauss_time_spin,
            self.contour_filter_gauss_space_spin,
        ):
            spin.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        cp_params_lay.addWidget(_param_group_label("Post-filter contour maps"))
        cp_params_lay.addLayout(contour_filter_grid)

        self.preview_contour_filter_btn = QPushButton("Preview Filter")
        self.preview_contour_filter_btn.setToolTip(
            "Preview filtered contour_maps.tif in napari without overwriting it"
        )
        self.run_contour_filter_btn = QPushButton("Run Filter")
        self.run_contour_filter_btn.setToolTip(
            "Filter contour_maps.tif and overwrite contour_maps.tif"
        )
        for button in (self.preview_contour_filter_btn, self.run_contour_filter_btn):
            button.setMinimumWidth(_CONTOUR_SWEEP_MIN_WIDTH)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        contour_filter_btn_row = block_grid(horizontal_spacing=12)
        add_block_button_row(
            contour_filter_btn_row,
            0,
            self.preview_contour_filter_btn,
            self.run_contour_filter_btn,
        )
        cp_params_lay.addLayout(contour_filter_btn_row)

        self.contour_status_lbl = _stage_status()
        cp_params_lay.addWidget(self.contour_status_lbl)

        self.build_progress_bar = _stage_progress()
        self.contour_output_files = _stage_files("Outputs", [
            ("2_nucleus/contour_maps.tif", "Contour maps"),
            ("2_nucleus/foreground_scores.tif", "Foreground scores"),
            ("2_nucleus/foreground_masks.tif", "Foreground masks"),
        ])
        cp_params_lay.addWidget(self.build_progress_bar)
        cp_params_lay.addWidget(self.contour_output_files)

        cp_params_scroll.setWidget(cp_params_widget)
        contour_lay.addWidget(cp_params_scroll)
        self.contour_section = CollapsibleSection(
            "1. Contour Maps", _contour_inner, expanded=False
        )
        layout.addWidget(self.contour_section)

        # ── 2. Ultrack Database Generation ────────────────────────────────
        _db_gen_inner = QWidget()
        db_gen_lay = QVBoxLayout(_db_gen_inner)
        db_gen_lay.setContentsMargins(0, 0, 0, 0)
        db_gen_lay.setSpacing(4)
        db_gen_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.db_gen_input_files = _stage_files("Inputs", [
            ("2_nucleus/contour_maps.tif", "Contour maps"),
            ("2_nucleus/foreground_masks.tif", "Foreground masks"),
            ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
        ])
        db_gen_lay.addWidget(self.db_gen_input_files)

        self.db_gen_min_area_spin = QSpinBox()
        self.db_gen_min_area_spin.setRange(0, 1_000_000)
        self.db_gen_min_area_spin.setValue(300)

        self.db_gen_max_area_spin = QSpinBox()
        self.db_gen_max_area_spin.setRange(0, 10_000_000)
        self.db_gen_max_area_spin.setValue(100_000)

        self.db_gen_fg_thr_spin = QDoubleSpinBox()
        self.db_gen_fg_thr_spin.setRange(-5.0, 1.0)
        self.db_gen_fg_thr_spin.setValue(0.5)
        self.db_gen_fg_thr_spin.setDecimals(2)
        self.db_gen_fg_thr_spin.setSingleStep(0.05)
        self.db_gen_fg_thr_spin.setToolTip(
            "Pixel-level foreground threshold for ultrack segmentation (threshold in segmentation_config)"
        )

        self.db_gen_min_frontier_spin = QDoubleSpinBox()
        self.db_gen_min_frontier_spin.setRange(0.0, 1.0)
        self.db_gen_min_frontier_spin.setValue(0.0)
        self.db_gen_min_frontier_spin.setDecimals(3)
        self.db_gen_min_frontier_spin.setSingleStep(0.01)
        self.db_gen_min_frontier_spin.setToolTip(
            "Minimum boundary fraction to keep a candidate (min_frontier in segmentation_config)"
        )

        self.db_gen_ws_hierarchy_combo = QComboBox()
        self.db_gen_ws_hierarchy_combo.addItems(["area", "dynamics", "volume"])

        self.db_gen_n_workers_spin = QSpinBox()
        self.db_gen_n_workers_spin.setRange(1, max(1, os.cpu_count() or 1))
        self.db_gen_n_workers_spin.setValue(1)
        self.db_gen_n_workers_spin.setToolTip("Parallel workers for segmentation")

        self.db_gen_max_dist_spin = QDoubleSpinBox()
        self.db_gen_max_dist_spin.setRange(0.0, 500.0)
        self.db_gen_max_dist_spin.setValue(15.0)
        self.db_gen_max_dist_spin.setDecimals(1)

        self.db_gen_max_neighbors_spin = QSpinBox()
        self.db_gen_max_neighbors_spin.setRange(1, 50)
        self.db_gen_max_neighbors_spin.setValue(5)

        self.db_gen_linking_mode_combo = QComboBox()
        self.db_gen_linking_mode_combo.addItems(["default", "iou"])

        self.db_gen_iou_weight_spin = QDoubleSpinBox()
        self.db_gen_iou_weight_spin.setRange(0.0, 1.0)
        self.db_gen_iou_weight_spin.setValue(1.0)
        self.db_gen_iou_weight_spin.setDecimals(2)
        self.db_gen_iou_weight_spin.setEnabled(False)

        self.db_gen_quality_weight_spin = QDoubleSpinBox()
        self.db_gen_quality_weight_spin.setRange(0.0, 10.0)
        self.db_gen_quality_weight_spin.setValue(1.0)
        self.db_gen_quality_weight_spin.setDecimals(2)
        self.db_gen_quality_weight_spin.setSingleStep(0.05)
        self.db_gen_quality_weight_spin.setToolTip(
            "Weight applied to signal-based segmentation quality before storing node_prob"
        )

        self.db_gen_quality_exp_spin = QDoubleSpinBox()
        self.db_gen_quality_exp_spin.setRange(0.1, 50.0)
        self.db_gen_quality_exp_spin.setValue(8.0)
        self.db_gen_quality_exp_spin.setDecimals(2)
        self.db_gen_quality_exp_spin.setToolTip(
            "Raises signal-based quality before storing as node_prob"
        )

        self.db_gen_circularity_weight_spin = QDoubleSpinBox()
        self.db_gen_circularity_weight_spin.setRange(0.0, 10.0)
        self.db_gen_circularity_weight_spin.setValue(0.25)
        self.db_gen_circularity_weight_spin.setDecimals(2)
        self.db_gen_circularity_weight_spin.setSingleStep(0.05)
        self.db_gen_circularity_weight_spin.setToolTip(
            "Weight applied to shape circularity before storing node_prob"
        )

        self.db_gen_power_spin = QDoubleSpinBox()
        self.db_gen_power_spin.setRange(0.1, 20.0)
        self.db_gen_power_spin.setValue(4.0)
        self.db_gen_power_spin.setDecimals(2)
        self.db_gen_power_spin.setToolTip(
            "Deprecated duplicate of the solver power control; solver transform for stored weights"
        )
        self.db_gen_power_spin.setVisible(False)

        self.ultrack_seed_weight_spin = QDoubleSpinBox()
        self.ultrack_seed_weight_spin.setRange(0.0, 10.0)
        self.ultrack_seed_weight_spin.setValue(0.5)
        self.ultrack_seed_weight_spin.setSingleStep(0.1)
        self.ultrack_seed_weight_spin.setDecimals(2)
        self.ultrack_seed_weight_spin.setToolTip(
            "Additive reward for candidates similar to nearby validated cells. "
            "Zero disables the seed-local bonus."
        )

        self.ultrack_seed_space_spin = QDoubleSpinBox()
        self.ultrack_seed_space_spin.setRange(1.0, 500.0)
        self.ultrack_seed_space_spin.setValue(25.0)
        self.ultrack_seed_space_spin.setSingleStep(5.0)
        self.ultrack_seed_space_spin.setDecimals(1)
        self.ultrack_seed_space_spin.setToolTip(
            "Spatial decay scale for seed proximity. Larger values let validated cells influence candidates farther away."
        )

        self.ultrack_seed_time_spin = QDoubleSpinBox()
        self.ultrack_seed_time_spin.setRange(0.1, 50.0)
        self.ultrack_seed_time_spin.setValue(2.0)
        self.ultrack_seed_time_spin.setSingleStep(0.5)
        self.ultrack_seed_time_spin.setDecimals(1)
        self.ultrack_seed_time_spin.setToolTip(
            "Temporal decay scale in frames. Larger values let validated cells influence more distant frames within the seed window."
        )

        self.ultrack_seed_window_spin = QSpinBox()
        self.ultrack_seed_window_spin.setRange(0, 100)
        self.ultrack_seed_window_spin.setValue(5)
        self.ultrack_seed_window_spin.setToolTip(
            "Maximum frame distance from a validated cell used for seed affinity."
        )

        db_candidate_grid = block_grid(horizontal_spacing=12)
        db_candidate_grid.setContentsMargins(0, 0, 0, 0)
        add_block_pair_row(db_candidate_grid, 0, "Min Area (px):", _compact(self.db_gen_min_area_spin), "Max Area (px):", _compact(self.db_gen_max_area_spin))
        add_block_pair_row(db_candidate_grid, 1, "FG Threshold:", _compact(self.db_gen_fg_thr_spin), "Min Frontier:", _compact(self.db_gen_min_frontier_spin))
        add_block_pair_row(db_candidate_grid, 2, "WS Hierarchy:", self.db_gen_ws_hierarchy_combo, "N Workers:", _compact(self.db_gen_n_workers_spin))
        db_gen_lay.addWidget(muted_label(QLabel("Candidate extraction")))
        db_gen_lay.addLayout(db_candidate_grid)

        db_linking_grid = block_grid(horizontal_spacing=12)
        db_linking_grid.setContentsMargins(0, 0, 0, 0)
        add_block_pair_row(db_linking_grid, 0, "Max Distance (px):", _compact(self.db_gen_max_dist_spin), "Max Neighbors:", _compact(self.db_gen_max_neighbors_spin))
        add_block_pair_row(db_linking_grid, 1, "Linking Mode:", self.db_gen_linking_mode_combo, "IoU Weight:", _compact(self.db_gen_iou_weight_spin))
        db_gen_lay.addWidget(muted_label(QLabel("Candidate linking")))
        db_gen_lay.addLayout(db_linking_grid)

        db_scoring_grid = block_grid(horizontal_spacing=12)
        db_scoring_grid.setContentsMargins(0, 0, 0, 0)
        add_block_pair_row(db_scoring_grid, 0, "Quality Weight:", _compact(self.db_gen_quality_weight_spin), "Quality Exp:", _compact(self.db_gen_quality_exp_spin))
        add_block_pair_row(db_scoring_grid, 1, "Circularity Weight:", _compact(self.db_gen_circularity_weight_spin), "", QWidget())
        db_gen_lay.addWidget(muted_label(QLabel("Node scoring")))
        db_gen_lay.addLayout(db_scoring_grid)

        self.db_gen_use_validated_check = QCheckBox("Use validated corrections")
        db_gen_validated_grid = block_grid(horizontal_spacing=12)
        add_block_checkbox_row(db_gen_validated_grid, 0, self.db_gen_use_validated_check)
        db_gen_lay.addLayout(db_gen_validated_grid)

        db_seed_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(
            db_seed_grid,
            0,
            "Seed Weight:",
            _compact(self.ultrack_seed_weight_spin, 80),
            "Seed Space (px):",
            _compact(self.ultrack_seed_space_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            db_seed_grid,
            1,
            "Seed Time:",
            _compact(self.ultrack_seed_time_spin, 80),
            "Seed Window:",
            _compact(self.ultrack_seed_window_spin, 80),
            field_width=80,
        )
        db_gen_lay.addWidget(muted_label(QLabel("Validated seed prior")))
        db_gen_lay.addLayout(db_seed_grid)

        db_gen_run_row = block_grid(horizontal_spacing=12)
        self.run_db_gen_btn = QPushButton("Run DB Generation")
        self.run_db_gen_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.db_gen_terminal_btn = QPushButton("Run in Terminal")
        self.db_gen_terminal_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        add_block_button_row(db_gen_run_row, 0, self.run_db_gen_btn, self.db_gen_terminal_btn)
        db_gen_lay.addLayout(db_gen_run_row)

        self.db_gen_status_lbl = _stage_status()
        db_gen_lay.addWidget(self.db_gen_status_lbl)

        self.db_gen_progress_bar = _stage_progress()
        db_gen_lay.addWidget(self.db_gen_progress_bar)

        self.db_gen_output_files = _stage_files("Outputs", [
            ("2_nucleus/ultrack_workdir/data.db", "Ultrack database"),
        ])
        db_gen_lay.addWidget(self.db_gen_output_files)

        self.db_gen_section = CollapsibleSection(
            "2. Ultrack Database Generation", _db_gen_inner, expanded=False
        )
        layout.addWidget(self.db_gen_section)

        # ── Optional Ultrack Database Browser ──────────────────────────────
        _ultrack_db_browser_inner = QWidget()
        ultrack_db_browser_lay = QVBoxLayout(_ultrack_db_browser_inner)
        ultrack_db_browser_lay.setContentsMargins(0, 0, 0, 0)
        ultrack_db_browser_lay.setSpacing(4)

        from qtpy.QtGui import QIcon
        from qtpy.QtCore import Qt as _Qt
        self.ultrack_db_info_lbl = QLabel("—")
        self.ultrack_db_info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ultrack_db_info_lbl.setWordWrap(True)
        self.ultrack_db_info_lbl.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        ultrack_db_browser_lay.addWidget(self.ultrack_db_info_lbl)

        ultrack_db_grid = block_grid(horizontal_spacing=12)
        ultrack_db_grid.setContentsMargins(0, 0, 0, 0)
        self.ultrack_db_hierarchy_slider = QSlider(_Qt.Horizontal)
        self.ultrack_db_hierarchy_slider.setRange(0, 100)
        self.ultrack_db_hierarchy_slider.setValue(50)
        self.ultrack_db_hierarchy_slider.setToolTip(
            "Hierarchy cut level: 0 = most split, 1 = most merged"
        )
        self.ultrack_db_height_lbl = QLabel("0.50")
        self.ultrack_db_height_lbl.setFixedWidth(48)
        self._ultrack_db_slider_row = QWidget()
        _slider_lay = QHBoxLayout(self._ultrack_db_slider_row)
        _slider_lay.setContentsMargins(0, 0, 0, 0)
        _slider_lay.addWidget(self.ultrack_db_hierarchy_slider)
        _slider_lay.addWidget(self.ultrack_db_height_lbl)
        ultrack_db_grid.addWidget(self._ultrack_db_slider_row, 0, 0, 1, 4)
        self._ultrack_db_slider_row.setVisible(True)

        ultrack_db_browser_lay.addLayout(ultrack_db_grid)

        _db_btn_row = QWidget()
        _db_btn_lay = QHBoxLayout(_db_btn_row)
        _db_btn_lay.setContentsMargins(0, 0, 0, 0)
        _db_btn_lay.setSpacing(4)
        self.ultrack_db_active_btn = QPushButton("Activate")
        self.ultrack_db_active_btn.setCheckable(True)
        self.ultrack_db_active_btn.setChecked(False)
        self.ultrack_db_active_btn.setToolTip("Load contour maps and foreground masks into viewer and enable DB preview")
        self.ultrack_db_refresh_btn = QPushButton()
        self.ultrack_db_refresh_btn.setToolTip("Refresh Ultrack database browser")
        self.ultrack_db_refresh_btn.setIcon(QIcon.fromTheme("view-refresh"))
        self.ultrack_db_refresh_btn.setEnabled(False)
        _db_btn_lay.addWidget(self.ultrack_db_active_btn)
        _db_btn_lay.addWidget(self.ultrack_db_refresh_btn)
        ultrack_db_browser_lay.addWidget(_db_btn_row)
        self.ultrack_db_hierarchy_slider.setEnabled(False)
        self.ultrack_db_prob_alpha_check = QCheckBox("Node prob transparency")
        self.ultrack_db_prob_alpha_check.setToolTip("Modulate label opacity by node probability (higher quality = more opaque)")
        self.ultrack_db_prob_alpha_check.setEnabled(False)
        self.ultrack_db_connected_focus_check = QCheckBox("Connected focus")
        self.ultrack_db_connected_focus_check.setToolTip(
            "Focus the DB preview on a selected node and its temporal neighbors"
        )
        self.ultrack_db_connected_focus_check.setEnabled(False)
        self.ultrack_db_edge_alpha_check = QCheckBox("Edge weight transparency")
        self.ultrack_db_edge_alpha_check.setToolTip(
            "Modulate connected-neighbor opacity by link weight"
        )
        self.ultrack_db_edge_alpha_check.setEnabled(False)
        self.ultrack_db_show_validated_check = QCheckBox("Show validated nodes")
        self.ultrack_db_show_validated_check.setChecked(True)
        self.ultrack_db_show_validated_check.setEnabled(False)
        self.ultrack_db_show_fake_check = QCheckBox("Show fake nodes")
        self.ultrack_db_show_fake_check.setChecked(False)
        self.ultrack_db_show_fake_check.setEnabled(False)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_prob_alpha_check)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_connected_focus_check)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_edge_alpha_check)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_show_validated_check)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_show_fake_check)

        self.ultrack_db_section_status_lbl = QLabel("")
        self.ultrack_db_section_status_lbl.setWordWrap(True)
        self.ultrack_db_section_status_lbl.setVisible(False)
        ultrack_db_browser_lay.addWidget(self.ultrack_db_section_status_lbl)

        self.ultrack_db_browser_section = CollapsibleSection(
            "Ultrack Database Browser", _ultrack_db_browser_inner, expanded=False
        )

        # ── 4. Ultrack Tracking ───────────────────────────────────────────

        _ultrack_inner = QWidget()
        ultrack_lay = QVBoxLayout(_ultrack_inner)
        ultrack_lay.setContentsMargins(0, 0, 0, 0)
        ultrack_lay.setSpacing(4)
        ultrack_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.ultrack_input_files = _stage_files("Inputs", [
            ("2_nucleus/ultrack_workdir/data.db", "Ultrack database"),
        ])
        ultrack_lay.addWidget(self.ultrack_input_files)

        self.ultrack_min_area_spin = QSpinBox()
        self.ultrack_min_area_spin.setRange(0, 100000)
        self.ultrack_min_area_spin.setValue(300)
        self.ultrack_min_area_spin.setSingleStep(50)

        self.ultrack_max_partitions_spin = QSpinBox()
        self.ultrack_max_partitions_spin.setRange(0, 1000)
        self.ultrack_max_partitions_spin.setValue(30)
        self.ultrack_max_partitions_spin.setToolTip("0 = use all partitions")

        self.ultrack_n_frames_spin = QSpinBox()
        self.ultrack_n_frames_spin.setRange(0, 10000)
        self.ultrack_n_frames_spin.setValue(0)
        self.ultrack_n_frames_spin.setToolTip("0 = process all frames")

        self.ultrack_linking_mode_combo = QComboBox()
        self.ultrack_linking_mode_combo.addItems(["default", "iou"])

        self.ultrack_max_dist_spin = QDoubleSpinBox()
        self.ultrack_max_dist_spin.setRange(0.0, 500.0)
        self.ultrack_max_dist_spin.setValue(15.0)
        self.ultrack_max_dist_spin.setSingleStep(1.0)
        self.ultrack_max_dist_spin.setDecimals(1)

        self.ultrack_iou_weight_spin = QDoubleSpinBox()
        self.ultrack_iou_weight_spin.setRange(0.0, 1.0)
        self.ultrack_iou_weight_spin.setValue(1.0)
        self.ultrack_iou_weight_spin.setSingleStep(0.05)
        self.ultrack_iou_weight_spin.setDecimals(2)

        self.ultrack_appear_spin = QDoubleSpinBox()
        self.ultrack_appear_spin.setRange(-10.0, 0.0)
        self.ultrack_appear_spin.setValue(-0.1)
        self.ultrack_appear_spin.setSingleStep(0.05)
        self.ultrack_appear_spin.setDecimals(3)

        self.ultrack_disappear_spin = QDoubleSpinBox()
        self.ultrack_disappear_spin.setRange(-10.0, 0.0)
        self.ultrack_disappear_spin.setValue(-0.1)
        self.ultrack_disappear_spin.setSingleStep(0.05)
        self.ultrack_disappear_spin.setDecimals(3)

        self.ultrack_division_spin = QDoubleSpinBox()
        self.ultrack_division_spin.setRange(-10.0, 0.0)
        self.ultrack_division_spin.setValue(-0.001)
        self.ultrack_division_spin.setSingleStep(0.05)
        self.ultrack_division_spin.setDecimals(3)
        self.ultrack_division_spin.setToolTip(
            "ILP penalty for cell division events. More negative = fewer divisions allowed."
        )

        self.ultrack_max_neighbors_spin = QSpinBox()
        self.ultrack_max_neighbors_spin.setRange(1, 50)
        self.ultrack_max_neighbors_spin.setValue(5)
        self.ultrack_max_neighbors_spin.setToolTip(
            "Maximum number of candidate predecessor nodes considered during linking."
        )

        self.ultrack_power_spin = QDoubleSpinBox()
        self.ultrack_power_spin.setRange(0.1, 20.0)
        self.ultrack_power_spin.setValue(4.0)
        self.ultrack_power_spin.setSingleStep(0.5)
        self.ultrack_power_spin.setDecimals(2)
        self.ultrack_power_spin.setToolTip(
            "Ultrack's solver transform for node_prob and link weights. "
            "With link_function=power, stored weights are raised to this power during solving."
        )

        self.ultrack_quality_exp_spin = QDoubleSpinBox()
        self.ultrack_quality_exp_spin.setRange(0.1, 50.0)
        self.ultrack_quality_exp_spin.setValue(8.0)
        self.ultrack_quality_exp_spin.setSingleStep(0.5)
        self.ultrack_quality_exp_spin.setDecimals(2)
        self.ultrack_quality_exp_spin.setToolTip(
            "Raises the signal-based segmentation quality before storing it as node_prob. "
            "Higher values favor high-confidence whole-object candidates over fragments."
        )

        self.ultrack_solver_lbl = QLabel("—")
        track_scope_grid = block_grid(horizontal_spacing=12)
        track_scope_grid.setContentsMargins(0, 0, 0, 0)
        add_block_pair_row(
            track_scope_grid,
            0,
            "Max Partitions/frame:",
            _compact(self.ultrack_max_partitions_spin, 80),
            "First N frames:",
            _compact(self.ultrack_n_frames_spin, 80),
            field_width=80,
        )
        ultrack_lay.addWidget(muted_label(QLabel("Track scope")))
        ultrack_lay.addLayout(track_scope_grid)

        event_penalty_grid = block_grid(horizontal_spacing=12)
        event_penalty_grid.setContentsMargins(0, 0, 0, 0)
        add_block_pair_row(
            event_penalty_grid,
            0,
            "Appear Penalty:",
            _compact(self.ultrack_appear_spin, 80),
            "Disappear Penalty:",
            _compact(self.ultrack_disappear_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            event_penalty_grid,
            1,
            "Division Penalty:",
            _compact(self.ultrack_division_spin, 80),
            field_width=80,
        )
        ultrack_lay.addWidget(muted_label(QLabel("Event penalties")))
        ultrack_lay.addLayout(event_penalty_grid)

        solver_grid = block_grid(horizontal_spacing=12)
        solver_grid.setContentsMargins(0, 0, 0, 0)
        add_block_pair_row(
            solver_grid,
            0,
            "Ultrack Power:",
            _compact(self.ultrack_power_spin, 80),
            "Solver:",
            self.ultrack_solver_lbl,
            field_width=None,
        )
        ultrack_lay.addWidget(muted_label(QLabel("Solver scoring")))
        ultrack_lay.addLayout(solver_grid)

        ultrack_run_row = block_grid(horizontal_spacing=12)
        self.run_ultrack_btn = QPushButton("Run Ultrack Tracking")
        self.ultrack_terminal_btn = QPushButton("Run in Terminal")
        add_block_button_row(ultrack_run_row, 0, self.run_ultrack_btn, self.ultrack_terminal_btn)
        ultrack_lay.addLayout(ultrack_run_row)

        self.ultrack_status_lbl = _stage_status()
        ultrack_lay.addWidget(self.ultrack_status_lbl)

        self.ultrack_progress_bar = _stage_progress()
        ultrack_lay.addWidget(self.ultrack_progress_bar)

        self.ultrack_output_files = _stage_files("Outputs", [
            ("2_nucleus/tracked_labels.tif", "Tracked labels"),
        ])
        ultrack_lay.addWidget(self.ultrack_output_files)

        ultrack_attrib = QLabel(
            "Ultrack tracking is powered by the "
            '<a href="https://github.com/royerlab/ultrack">Ultrack</a> project.'
        )
        ultrack_attrib.setOpenExternalLinks(True)
        ultrack_attrib.setWordWrap(True)
        muted_label(ultrack_attrib, size_pt=9)
        ultrack_lay.addWidget(ultrack_attrib)

        self.ultrack_section = CollapsibleSection(
            "4. Ultrack Tracking", _ultrack_inner, expanded=False
        )
        layout.addWidget(self.ultrack_section)

        _corr_inner = QWidget()
        _corr_inner_lay = QVBoxLayout(_corr_inner)
        _corr_inner_lay.setContentsMargins(0, 0, 0, 0)
        _corr_inner_lay.setSpacing(4)

        extend_row = block_grid(horizontal_spacing=12)
        self.extend_back_btn = QPushButton("◀ Extend (A)")
        self.extend_fwd_btn = QPushButton("Extend (D) ▶")
        add_block_button_row(extend_row, 0, self.extend_back_btn, self.extend_fwd_btn)
        _corr_inner_lay.addLayout(extend_row)

        retrack_row = block_grid(horizontal_spacing=12)
        self.retrack_back_btn = QPushButton("◀ Retrack (Q)")
        self.retrack_fwd_btn = QPushButton("Retrack (E) ▶")
        add_block_button_row(retrack_row, 0, self.retrack_back_btn, self.retrack_fwd_btn)
        _corr_inner_lay.addLayout(retrack_row)

        save_load_row = block_grid(horizontal_spacing=12)
        self.save_tracked_btn = QPushButton("Save Tracked Labels")
        self.load_tracked_btn = QPushButton("Load Tracked Labels")
        add_block_button_row(save_load_row, 0, self.save_tracked_btn, self.load_tracked_btn)
        _corr_inner_lay.addLayout(save_load_row)

        reassign_row = block_grid(horizontal_spacing=12)
        self.reassign_ids_btn = QPushButton("Reassign IDs")
        add_block_button_row(reassign_row, 0, self.reassign_ids_btn)
        _corr_inner_lay.addLayout(reassign_row)

        extend_params_inner = QWidget()
        extend_params_lay = QVBoxLayout(extend_params_inner)
        extend_params_lay.setContentsMargins(0, 0, 0, 0)
        extend_params_lay.setSpacing(4)
        extend_params_form = block_grid(horizontal_spacing=12)
        self.extend_max_dist_spin = QDoubleSpinBox()
        self.extend_max_dist_spin.setRange(0.0, 500.0)
        self.extend_max_dist_spin.setValue(40.0)
        self.extend_max_dist_spin.setSingleStep(1.0)
        self.extend_max_dist_spin.setDecimals(1)
        self.extend_area_weight_spin = QDoubleSpinBox()
        self.extend_area_weight_spin.setRange(0.0, 10.0)
        self.extend_area_weight_spin.setValue(1.0)
        self.extend_area_weight_spin.setSingleStep(0.1)
        self.extend_area_weight_spin.setDecimals(2)
        self.extend_iou_weight_spin = QDoubleSpinBox()
        self.extend_iou_weight_spin.setRange(0.0, 10.0)
        self.extend_iou_weight_spin.setValue(1.0)
        self.extend_iou_weight_spin.setSingleStep(0.1)
        self.extend_iou_weight_spin.setDecimals(2)
        self.extend_distance_weight_spin = QDoubleSpinBox()
        self.extend_distance_weight_spin.setRange(0.0, 10.0)
        self.extend_distance_weight_spin.setValue(0.25)
        self.extend_distance_weight_spin.setSingleStep(0.05)
        self.extend_distance_weight_spin.setDecimals(2)
        self.extend_overlap_penalty_spin = QDoubleSpinBox()
        self.extend_overlap_penalty_spin.setRange(0.0, 10.0)
        self.extend_overlap_penalty_spin.setValue(1.0)
        self.extend_overlap_penalty_spin.setSingleStep(0.1)
        self.extend_overlap_penalty_spin.setDecimals(2)
        self.extend_greedy_overwrite_check = QCheckBox("Greedy overwrite")
        add_block_pair_row(
            extend_params_form,
            0,
            "Max Distance (px):",
            _compact(self.extend_max_dist_spin, 80),
            "Area Weight:",
            _compact(self.extend_area_weight_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            extend_params_form,
            1,
            "IoU Weight:",
            _compact(self.extend_iou_weight_spin, 80),
            "Distance Weight:",
            _compact(self.extend_distance_weight_spin, 80),
            field_width=80,
        )
        add_block_pair_row(
            extend_params_form,
            2,
            "Overlap Penalty:",
            _compact(self.extend_overlap_penalty_spin, 80),
            field_width=80,
        )
        add_block_checkbox_row(extend_params_form, 3, self.extend_greedy_overwrite_check)
        extend_params_lay.addLayout(extend_params_form)
        self.extend_params_section = CollapsibleSection(
            "Extend Parameters", extend_params_inner, expanded=False
        )
        _corr_inner_lay.addWidget(self.extend_params_section)

        retrack_params_inner = QWidget()
        retrack_params_lay = QVBoxLayout(retrack_params_inner)
        retrack_params_lay.setContentsMargins(0, 0, 0, 0)
        retrack_params_lay.setSpacing(4)
        retrack_params_form = block_grid(horizontal_spacing=12)
        self.retrack_max_dist_spin = QDoubleSpinBox()
        self.retrack_max_dist_spin.setRange(0.0, 500.0)
        self.retrack_max_dist_spin.setValue(20.0)
        self.retrack_max_dist_spin.setSingleStep(1.0)
        self.retrack_max_dist_spin.setDecimals(1)
        add_block_pair_row(
            retrack_params_form,
            0,
            "Max Distance (px):",
            _compact(self.retrack_max_dist_spin, 80),
            field_width=80,
        )
        retrack_params_lay.addLayout(retrack_params_form)
        self.retrack_params_section = CollapsibleSection(
            "Retrack Parameters", retrack_params_inner, expanded=False
        )
        _corr_inner_lay.addWidget(self.retrack_params_section)

        self.validation_counter_lbl = QLabel("")
        self.validation_counter_lbl.setWordWrap(True)
        _corr_inner_lay.addWidget(self.validation_counter_lbl)

        self.remove_unvalidated_btn = QPushButton("Remove Unvalidated Labels")
        self.remove_unvalidated_btn.setToolTip(
            "Remove nucleus label pixels that are not marked validated for their frame."
        )
        action_button(self.remove_unvalidated_btn, expand=True)
        danger_button(self.remove_unvalidated_btn)
        _corr_inner_lay.addWidget(self.remove_unvalidated_btn)

        self.correction_status_lbl = QLabel("")
        self.correction_status_lbl.setWordWrap(True)
        self.correction_status_lbl.setVisible(False)
        _corr_inner_lay.addWidget(self.correction_status_lbl)

        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
        )
        self.correction_widget.set_edit_callback(self._on_cells_edited)
        _corr_inner_lay.addWidget(self.correction_widget)
        self.correction_shortcuts_section = CollapsibleSection(
            "Correction Shortcuts",
            self.correction_widget.build_shortcuts_widget(),
            expanded=False,
        )
        _corr_inner_lay.addWidget(self.correction_shortcuts_section)

        self.correction_section = CollapsibleSection(
            "5. Correction", _corr_inner, expanded=False
        )
        layout.addWidget(self.correction_section)
        layout.addWidget(self.ultrack_db_browser_section)

    # ──────────────────────────────────────────────────────────────────────────
    # Signal wiring
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.build_btn.clicked.connect(self._on_build_contour_maps)
        self.preview_contour_btn.clicked.connect(self._on_preview_contour_maps)
        self.contour_terminal_btn.clicked.connect(self._on_run_contour_terminal)
        self.preview_contour_filter_btn.clicked.connect(self._on_preview_contour_filter)
        self.run_contour_filter_btn.clicked.connect(self._on_run_contour_filter)
        self.cancel_build_btn.clicked.connect(self._on_cancel_build)
        self.run_db_gen_btn.clicked.connect(self._on_run_db_generation)
        self.db_gen_terminal_btn.clicked.connect(self._on_db_gen_terminal)
        self.db_gen_linking_mode_combo.currentTextChanged.connect(self._on_db_gen_mode_changed)
        self.db_gen_use_validated_check.toggled.connect(self._set_resolve_prior_controls_enabled)
        self.ultrack_db_active_btn.toggled.connect(self._on_ultrack_db_activate)
        self.ultrack_db_refresh_btn.clicked.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_hierarchy_slider.valueChanged.connect(self._on_ultrack_db_slider_changed)
        self.ultrack_db_prob_alpha_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_connected_focus_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_edge_alpha_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_show_validated_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.ultrack_db_show_fake_check.toggled.connect(self._refresh_ultrack_db_browser)
        self.run_ultrack_btn.clicked.connect(self._on_run_ultrack)
        self.ultrack_terminal_btn.clicked.connect(self._on_ultrack_terminal)
        self.save_tracked_btn.clicked.connect(self._on_save_tracked)
        self.load_tracked_btn.clicked.connect(self._on_load_tracked)
        self.reassign_ids_btn.clicked.connect(self._on_reassign_ids)
        self.ultrack_linking_mode_combo.currentTextChanged.connect(self._on_ultrack_mode_changed)
        self.retrack_back_btn.clicked.connect(self._on_retrack_backward)
        self.retrack_fwd_btn.clicked.connect(self._on_retrack_forward)
        self.extend_back_btn.clicked.connect(self._on_extend_backward)
        self.extend_fwd_btn.clicked.connect(self._on_extend_forward)
        self.remove_unvalidated_btn.clicked.connect(self._on_remove_unvalidated_labels)
        self.viewer.dims.events.current_step.connect(self._on_dims_step_changed)
        self.viewer.bind_key("V", self._kb_toggle_cell_validation, overwrite=True)
        self._install_correction_shortcuts()
        self.correction_widget._activate_btn.toggled.connect(self._on_correction_mode_toggled)
        # Set initial state for solver label and IoU weight enablement
        solver = _select_solver()
        solver_display = "Gurobi (licensed)" if solver == "GUROBI" else "CBC"
        self.ultrack_solver_lbl.setText(solver_display)
        self._on_ultrack_mode_changed(self.ultrack_linking_mode_combo.currentText())
        self._set_resolve_prior_controls_enabled()

    # ──────────────────────────────────────────────────────────────────────────
    # Public refresh
    # ──────────────────────────────────────────────────────────────────────────

    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._refresh_stage_files(pos_dir)
        if pos_dir is None:
            self.correction_widget.deactivate()
            return
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _refresh_stage_files(self, pos_dir: Path | None = None) -> None:
        if pos_dir is None:
            pos_dir = self._pos_dir
        for files_widget in (
            self.contour_input_files,
            self.contour_output_files,
            self.db_gen_input_files,
            self.db_gen_output_files,
            self.ultrack_input_files,
            self.ultrack_output_files,
        ):
            files_widget.refresh(pos_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Path helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _tracked_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    def _prob_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif" if self._pos_dir else None

    def _dp_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif" if self._pos_dir else None

    def _contour_maps_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "contour_maps.tif" if self._pos_dir else None

    def _foreground_scores_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "foreground_scores.tif" if self._pos_dir else None

    def _cell_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "cell_zavg.tif" if self._pos_dir else None

    def _nucleus_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "nucleus_zavg.tif" if self._pos_dir else None

    def _ultrack_workdir(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "ultrack_workdir" if self._pos_dir else None

    def _ultrack_db_path(self) -> Path | None:
        workdir = self._ultrack_workdir()
        return workdir / "data.db" if workdir else None

    def _foreground_masks_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "foreground_masks.tif" if self._pos_dir else None

    def _nucleus_prob_zavg_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif" if self._pos_dir else None

    # ── DB Generation section ─────────────────────────────────────────────────

    def _db_gen_config_from_controls(self) -> UltrackConfig:
        return UltrackConfig(
            seg_min_area=self.db_gen_min_area_spin.value(),
            seg_max_area=self.db_gen_max_area_spin.value(),
            seg_foreground_threshold=self.db_gen_fg_thr_spin.value(),
            seg_min_frontier=self.db_gen_min_frontier_spin.value(),
            seg_ws_hierarchy=self.db_gen_ws_hierarchy_combo.currentText(),
            seg_n_workers=self.db_gen_n_workers_spin.value(),
            max_distance=self.db_gen_max_dist_spin.value(),
            max_neighbors=self.db_gen_max_neighbors_spin.value(),
            linking_mode=self.db_gen_linking_mode_combo.currentText(),
            iou_weight=self.db_gen_iou_weight_spin.value(),
            quality_weight=self.db_gen_quality_weight_spin.value(),
            quality_exponent=self.db_gen_quality_exp_spin.value(),
            circularity_weight=self.db_gen_circularity_weight_spin.value(),
            link_n_workers=self.db_gen_n_workers_spin.value(),
            seed_weight=self.ultrack_seed_weight_spin.value(),
            seed_sigma_space=self.ultrack_seed_space_spin.value(),
            seed_tau_time=self.ultrack_seed_time_spin.value(),
            seed_max_dt=self.ultrack_seed_window_spin.value(),
        )

    def _on_run_db_generation(self) -> None:
        if self._pos_dir is None:
            self._set_db_gen_status("No project open.")
            return
        contour_path = self._contour_maps_path()
        fg_path = self._foreground_masks_path()
        nuc_zavg_path = self._nucleus_prob_zavg_path()
        if contour_path is None or not contour_path.exists():
            self._set_db_gen_status("Missing: contour_maps.tif — run Contour Maps first.")
            return
        if fg_path is None or not fg_path.exists():
            self._set_db_gen_status(
                "Missing: foreground_masks.tif (foreground mask) — run Contour Maps first."
            )
            return
        if nuc_zavg_path is None or not nuc_zavg_path.exists():
            self._set_db_gen_status("Missing: nucleus_prob_zavg.tif — run Cellpose first.")
            return
        if _ultrack_segment is None:
            self._set_db_gen_status("ultrack not installed — activate the cellflow conda environment.")
            return

        cfg = self._db_gen_config_from_controls()
        working_dir = self._ultrack_workdir()
        pos_dir = self._pos_dir
        use_validated = self.db_gen_use_validated_check.isChecked()
        validated_tracks: dict[int, set[int]] | None = None
        tracked_labels: np.ndarray | None = None
        if use_validated:
            validated_tracks = read_validated_tracks(pos_dir)
            if not validated_tracks:
                self._set_db_gen_status("No validated tracks found — validate some cells first (press V).")
                return
            if _TRACKED_LAYER not in self.viewer.layers:
                self._set_db_gen_status("No tracked layer loaded for validated DB generation.")
                return
            tracked_labels = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)

        self.db_gen_progress_bar.setRange(0, 0)
        self.db_gen_progress_bar.setVisible(True)
        self._set_db_gen_status("Starting DB generation…")
        self.run_db_gen_btn.setEnabled(False)
        self.db_gen_terminal_btn.setEnabled(False)

        @thread_worker(connect={
            "yielded": self._on_db_gen_progress,
            "returned": self._on_db_gen_done,
            "errored": self._on_db_gen_worker_error,
        })
        def _worker():
            import queue as _queue
            import threading

            msg_queue: _queue.SimpleQueue = _queue.SimpleQueue()
            result_holder: list = []
            exc_holder: list = []

            def _progress(msg: str) -> None:
                msg_queue.put(msg)

            def _run() -> None:
                try:
                    result_holder.append(
                        build_ultrack_database(
                            contour_maps_path=contour_path,
                            foreground_masks_path=fg_path,
                            nucleus_prob_zavg_path=nuc_zavg_path,
                            working_dir=working_dir,
                            cfg=cfg,
                            validated_tracks=validated_tracks,
                            tracked_labels=tracked_labels,
                            use_validated=use_validated,
                            progress_cb=_progress,
                        )
                    )
                except Exception as e:
                    exc_holder.append(e)

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            while t.is_alive() or not msg_queue.empty():
                try:
                    yield msg_queue.get_nowait()
                except _queue.Empty:
                    t.join(timeout=0.05)
            if exc_holder:
                raise exc_holder[0]
            return pos_dir

        _worker()

    def _on_db_gen_terminal(self) -> None:
        import sys
        import tempfile

        if self._pos_dir is None:
            self._set_db_gen_status("No project open.")
            return
        contour_path = self._contour_maps_path()
        fg_path = self._foreground_masks_path()
        nuc_zavg_path = self._nucleus_prob_zavg_path()
        if contour_path is None or not contour_path.exists():
            self._set_db_gen_status("Missing: contour_maps.tif")
            return
        if fg_path is None or not fg_path.exists():
            self._set_db_gen_status("Missing: foreground_masks.tif (foreground mask) — run Contour Maps first.")
            return
        if nuc_zavg_path is None or not nuc_zavg_path.exists():
            self._set_db_gen_status("Missing: nucleus_prob_zavg.tif")
            return

        cfg = self._db_gen_config_from_controls()
        working_dir = self._ultrack_workdir()
        use_validated = self.db_gen_use_validated_check.isChecked()
        tracked_path = self._tracked_path()
        if use_validated:
            validated_tracks = read_validated_tracks(self._pos_dir)
            if not validated_tracks:
                self._set_db_gen_status("No validated tracks found — validate some cells first (press V).")
                return
            if tracked_path is None or not tracked_path.exists():
                self._set_db_gen_status("Tracked labels not found for validated DB generation.")
                return

        python_code = (
            "import pathlib, sys\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))\n"
            "from cellflow.database.tracked import read_full_tracked_stack\n"
            "from cellflow.database.validation import read_validated_tracks\n"
            "from cellflow.tracking_ultrack.config import TrackingConfig\n"
            "from cellflow.tracking_ultrack.db_build import build_ultrack_database\n"
            "\n"
            "if __name__ == '__main__':\n"
            f"    pos_dir = pathlib.Path({str(self._pos_dir)!r})\n"
            f"    contour_path = pathlib.Path({str(contour_path)!r})\n"
            f"    foreground_masks_path = pathlib.Path({str(fg_path)!r})\n"
            f"    nucleus_prob_zavg_path = pathlib.Path({str(nuc_zavg_path)!r})\n"
            f"    working_dir = pathlib.Path({str(working_dir)!r})\n"
            f"    tracked_path = pathlib.Path({str(tracked_path)!r})\n"
            f"    use_validated = {bool(use_validated)!r}\n"
            "    cfg = TrackingConfig(\n"
            f"        seg_min_area={cfg.seg_min_area},\n"
            f"        seg_max_area={cfg.seg_max_area},\n"
            f"        seg_foreground_threshold={cfg.seg_foreground_threshold},\n"
            f"        seg_min_frontier={cfg.seg_min_frontier},\n"
            f"        seg_ws_hierarchy={cfg.seg_ws_hierarchy!r},\n"
            f"        seg_n_workers={cfg.seg_n_workers},\n"
            f"        max_distance={cfg.max_distance},\n"
            f"        max_neighbors={cfg.max_neighbors},\n"
            f"        linking_mode={cfg.linking_mode!r},\n"
            f"        iou_weight={cfg.iou_weight},\n"
            f"        quality_weight={cfg.quality_weight},\n"
            f"        quality_exponent={cfg.quality_exponent},\n"
            f"        circularity_weight={cfg.circularity_weight},\n"
            f"        link_n_workers={cfg.link_n_workers},\n"
            f"        seed_weight={cfg.seed_weight},\n"
            f"        seed_sigma_space={cfg.seed_sigma_space},\n"
            f"        seed_tau_time={cfg.seed_tau_time},\n"
            f"        seed_max_dt={cfg.seed_max_dt},\n"
            "    )\n"
            "    validated_tracks = read_validated_tracks(pos_dir) if use_validated else None\n"
            "    tracked_labels = read_full_tracked_stack(tracked_path) if use_validated else None\n"
            "    report = build_ultrack_database(\n"
            "        contour_maps_path=contour_path,\n"
            "        foreground_masks_path=foreground_masks_path,\n"
            "        nucleus_prob_zavg_path=nucleus_prob_zavg_path,\n"
            "        working_dir=working_dir,\n"
            "        cfg=cfg,\n"
            "        validated_tracks=validated_tracks,\n"
            "        tracked_labels=tracked_labels,\n"
            "        use_validated=use_validated,\n"
            "        progress_cb=lambda msg: print(msg, flush=True),\n"
            "    )\n"
            "    print(f'Done. {report}', flush=True)\n"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix="cellflow_db_gen_", delete=False) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path)}"
        try:
            from cellflow.napari.utils import launch_in_terminal
            launch_in_terminal(cmd)
            self._set_db_gen_status("DB generation launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(cmd)
            self._set_db_gen_status("Copied DB generation command to clipboard.")

    def _on_db_gen_mode_changed(self, mode: str) -> None:
        self.db_gen_iou_weight_spin.setEnabled(mode == "iou")

    def _on_db_gen_progress(self, msg: str) -> None:
        self._set_db_gen_status(msg)

    def _on_db_gen_done(self, pos_dir: Path) -> None:
        self.db_gen_progress_bar.setVisible(False)
        self.run_db_gen_btn.setEnabled(True)
        self.db_gen_terminal_btn.setEnabled(True)
        self._set_db_gen_status("DB generation complete.")
        self._refresh_stage_files(pos_dir)
        self._refresh_ultrack_db_browser()

    def _on_db_gen_worker_error(self, exc: Exception) -> None:
        self.db_gen_progress_bar.setVisible(False)
        self.run_db_gen_btn.setEnabled(True)
        self.db_gen_terminal_btn.setEnabled(True)
        self._set_db_gen_status(f"Error: {exc}")
        logger.exception("DB generation worker error", exc_info=exc)

    def _set_db_gen_status(self, msg: str) -> None:
        self.db_gen_status_lbl.setText(msg)
        self.db_gen_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    # ── Ultrack DB Browser section ────────────────────────────────────────────

    def _set_ultrack_db_status(self, msg: str) -> None:
        self.ultrack_db_section_status_lbl.setText(msg)
        self.ultrack_db_section_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _on_ultrack_db_browser_param_changed(self, *_args) -> None:
        self._ultrack_db_preview_cache.clear()

    def _on_ultrack_db_slider_changed(self, value: int) -> None:
        if not self._ultrack_db_browser_active:
            return
        db_path = self._ultrack_db_path()
        if db_path is not None and db_path.exists():
            try:
                mtime_ns = db_path.stat().st_mtime_ns
                heights = self._query_distinct_heights(db_path, mtime_ns)
                index = min(max(int(value), 0), max(len(heights) - 1, 0))
                if heights:
                    self._set_ultrack_db_height_label(index, heights[index], len(heights))
                else:
                    self.ultrack_db_height_lbl.setText("—")
            except Exception:
                self.ultrack_db_height_lbl.setText(str(value))
        else:
            self.ultrack_db_height_lbl.setText(str(value))
        self._ultrack_db_preview_cache.clear()
        from qtpy.QtCore import QTimer
        QTimer.singleShot(150, self._refresh_ultrack_db_browser)

    def _on_ultrack_db_activate(self, checked: bool) -> None:
        self._ultrack_db_browser_active = checked
        self.ultrack_db_active_btn.setText("Deactivate" if checked else "Activate")
        self.ultrack_db_refresh_btn.setEnabled(checked)
        self.ultrack_db_hierarchy_slider.setEnabled(checked)
        self.ultrack_db_prob_alpha_check.setEnabled(checked)
        self.ultrack_db_connected_focus_check.setEnabled(checked)
        self.ultrack_db_edge_alpha_check.setEnabled(checked)
        self.ultrack_db_show_validated_check.setEnabled(checked)
        self.ultrack_db_show_fake_check.setEnabled(checked)
        if checked:
            self._ultrack_db_frame_initialized = False
            self._refresh_ultrack_db_browser()
        else:
            self._remove_ultrack_db_browser_layers()

    def _remove_ultrack_db_browser_layers(self) -> None:
        self._remove_ultrack_db_preview_selector()
        for name in (
            _ULTRACK_DB_PREVIEW_LAYER,
            _ULTRACK_DB_ANNOTATION_LAYER,
        ):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        if _ULTRACK_DB_SELECTION_LAYER in self.viewer.layers:
            self.viewer.layers.remove(_ULTRACK_DB_SELECTION_LAYER)
        self.ultrack_db_info_lbl.setText("—")
        self._set_ultrack_db_status("")

    def _ultrack_db_middle_frame(self, db_path: Path) -> int | None:
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                frames = sorted(
                    int(r[0]) for r in session.query(NodeDB.t).distinct().all()
                )
        except Exception:
            return None
        finally:
            engine.dispose()
        if not frames:
            return None
        return frames[len(frames) // 2]

    def _refresh_ultrack_db_browser(self) -> None:
        if not self._ultrack_db_browser_active:
            return
        self.ultrack_db_info_lbl.setText("—")
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._set_ultrack_db_status("data.db not found — run DB generation first.")
            return
        frame = self._current_t()
        if not self._ultrack_db_frame_initialized:
            self._ultrack_db_frame_initialized = True
            if frame == 0:
                mid = self._ultrack_db_middle_frame(db_path)
                if mid is not None and mid > 0:
                    frame = mid
                    self._set_viewer_frame(frame)
        try:
            self.ultrack_db_info_lbl.setText(self._ultrack_db_summary_text(db_path, frame))
            mtime_ns = db_path.stat().st_mtime_ns
            states = self._configure_ultrack_db_hierarchy_slider(db_path, mtime_ns, frame)
            if not states:
                labels = self._empty_ultrack_db_preview()
                self._update_layer(_ULTRACK_DB_PREVIEW_LAYER, labels)
                self._set_ultrack_db_status(f"No hierarchy states for frame {frame}.")
                return

            slider_int = int(self.ultrack_db_hierarchy_slider.value())
            state = states[slider_int]
            key = (
                str(db_path.resolve()),
                mtime_ns,
                frame,
                slider_int,
                state,
                self.ultrack_db_show_validated_check.isChecked(),
                self.ultrack_db_show_fake_check.isChecked(),
            )
            cached = self._ultrack_db_preview_cache.get(key)
            if cached is None:
                cached = self._render_hierarchy_cut_state(db_path, frame, state)
                self._ultrack_db_preview_cache[key] = cached
            labels, status, prob_dict, label_to_node_id, node_id_to_label, node_annotations = (
                self._normalize_ultrack_db_preview(cached)
            )
            self._ultrack_db_label_to_node_id = label_to_node_id
            self._ultrack_db_node_id_to_label = node_id_to_label
            self._ultrack_db_node_annotations = node_annotations
            alpha_dict: dict[int, float] = {}
            if self.ultrack_db_connected_focus_check.isChecked():
                labels, status, alpha_dict = self._render_ultrack_db_connected_focus(
                    db_path,
                    frame,
                    labels,
                    status,
                    prob_dict,
                    label_to_node_id,
                    node_id_to_label,
                )
            self._ultrack_db_preview_labels = labels.astype(np.uint32, copy=False)
            self._update_ultrack_db_preview_layer(
                self._ultrack_db_preview_labels, prob_dict, alpha_dict
            )
            self._update_ultrack_db_annotation_layer(
                self._ultrack_db_preview_labels,
                label_to_node_id,
                node_annotations,
            )
            self._install_ultrack_db_preview_selector()
            if not self.ultrack_db_connected_focus_check.isChecked():
                status = self._refresh_ultrack_db_selection_highlight(
                    self._ultrack_db_preview_labels,
                    status,
                    node_id_to_label,
                    frame,
                )
            self._set_ultrack_db_status(status)
        except Exception as e:
            self._set_ultrack_db_status(f"DB read error: {e}")
            logger.warning("DB browser error: %s", e)

    @staticmethod
    def _normalize_ultrack_db_preview(
        cached: tuple[np.ndarray, str]
        | tuple[np.ndarray, str, dict[int, float]]
        | tuple[
            np.ndarray,
            str,
            dict[int, float],
            dict[int, int],
            dict[int, int],
            dict[int, str],
        ],
    ) -> tuple[np.ndarray, str, dict[int, float], dict[int, int], dict[int, int], dict[int, str]]:
        if len(cached) == 2:
            labels, status = cached
            return labels, status, {}, {}, {}, {}
        if len(cached) == 3:
            labels, status, prob_dict = cached
            return labels, status, prob_dict, {}, {}, {}
        if len(cached) == 5:
            labels, status, prob_dict, label_to_node_id, node_id_to_label = cached
            return labels, status, prob_dict, label_to_node_id, node_id_to_label, {}
        labels, status, prob_dict, label_to_node_id, node_id_to_label, node_annotations = cached
        return labels, status, prob_dict, label_to_node_id, node_id_to_label, node_annotations

    def _update_ultrack_db_preview_layer(
        self,
        labels: np.ndarray,
        prob_dict: dict[int, float],
        alpha_dict: dict[int, float] | None = None,
    ) -> None:
        if alpha_dict:
            data = self._ultrack_db_alpha_rgba(labels, alpha_dict)
            self._update_image_layer(_ULTRACK_DB_PREVIEW_LAYER, data, rgb=True)
            return
        if self.ultrack_db_prob_alpha_check.isChecked() and prob_dict:
            data = self._ultrack_db_probability_rgba(labels, prob_dict)
            self._update_image_layer(_ULTRACK_DB_PREVIEW_LAYER, data, rgb=True)
            return
        self._update_labels_layer(_ULTRACK_DB_PREVIEW_LAYER, labels)

    def _update_ultrack_db_annotation_layer(
        self,
        labels: np.ndarray,
        label_to_node_id: dict[int, int],
        node_annotations: dict[int, str],
    ) -> None:
        overlay = np.zeros_like(labels, dtype=np.uint8)
        for label_id, node_id in label_to_node_id.items():
            annot = node_annotations.get(int(node_id), "UNKNOWN")
            if annot == "REAL":
                overlay[labels == int(label_id)] = 1
            elif annot == "FAKE":
                overlay[labels == int(label_id)] = 2
        if not np.any(overlay):
            if _ULTRACK_DB_ANNOTATION_LAYER in self.viewer.layers:
                self.viewer.layers.remove(_ULTRACK_DB_ANNOTATION_LAYER)
            return
        self._update_labels_layer(_ULTRACK_DB_ANNOTATION_LAYER, overlay)

    def _update_labels_layer(self, name: str, data: np.ndarray) -> None:
        from napari.layers import Labels

        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Labels):
            self.viewer.layers[name].data = data
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        self.viewer.add_labels(data, name=name)

    def _update_image_layer(self, name: str, data: np.ndarray, *, rgb: bool = False) -> None:
        from napari.layers import Image

        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Image):
            self.viewer.layers[name].data = data
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        self.viewer.add_image(data, name=name, rgb=rgb, blending="translucent")

    @staticmethod
    def _ultrack_db_probability_rgba(
        labels: np.ndarray, prob_dict: dict[int, float]
    ) -> np.ndarray:
        from napari.utils.colormaps import label_colormap

        rgba = np.zeros(labels.shape + (4,), dtype=np.float32)
        if labels.size == 0 or not prob_dict:
            return rgba

        probs = [float(v) for v in prob_dict.values()]
        min_p = min(probs)
        max_p = max(probs)
        denom = max(max_p - min_p, 1e-9)
        cmap = label_colormap(max(prob_dict.keys()) + 1)
        for label_id, prob in prob_dict.items():
            label_mask = labels == int(label_id)
            if not np.any(label_mask):
                continue
            color = np.asarray(cmap.map(int(label_id)), dtype=np.float32)
            alpha = 0.15 + 0.85 * (float(prob) - min_p) / denom
            color[3] = float(np.clip(alpha, 0.15, 1.0))
            rgba[label_mask] = color
        return rgba

    @staticmethod
    def _ultrack_db_alpha_rgba(
        labels: np.ndarray, alpha_dict: dict[int, float]
    ) -> np.ndarray:
        from napari.utils.colormaps import label_colormap

        rgba = np.zeros(labels.shape + (4,), dtype=np.float32)
        if labels.size == 0 or not alpha_dict:
            return rgba

        cmap = label_colormap(max(alpha_dict.keys()) + 1)
        for label_id, alpha in alpha_dict.items():
            label_mask = labels == int(label_id)
            if not np.any(label_mask):
                continue
            color = np.asarray(cmap.map(int(label_id)), dtype=np.float32)
            color[3] = float(np.clip(alpha, 0.0, 1.0))
            rgba[label_mask] = color
        return rgba

    def _install_ultrack_db_preview_selector(self) -> None:
        if _ULTRACK_DB_PREVIEW_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[_ULTRACK_DB_PREVIEW_LAYER]
        self._remove_ultrack_db_preview_selector()

        def _on_drag(_layer, event):
            if getattr(event, "type", None) != "mouse_press":
                return
            if getattr(event, "button", None) != 1:
                return
            if getattr(event, "modifiers", set()):
                return
            labels = self._ultrack_db_preview_labels
            if labels is None or labels.size == 0:
                return
            pos = _layer.world_to_data(event.position)
            y = int(round(float(pos[-2])))
            x = int(round(float(pos[-1])))
            if y < 0 or x < 0 or y >= labels.shape[-2] or x >= labels.shape[-1]:
                return
            display_label = int(labels[y, x])
            if display_label == 0:
                return
            self._select_ultrack_db_preview_label(display_label, frame=self._current_t())
            yield

        layer.mouse_drag_callbacks.append(_on_drag)
        self._ultrack_db_preview_mouse_callback = _on_drag

    def _remove_ultrack_db_preview_selector(self) -> None:
        callback = self._ultrack_db_preview_mouse_callback
        if callback is None or _ULTRACK_DB_PREVIEW_LAYER not in self.viewer.layers:
            self._ultrack_db_preview_mouse_callback = None
            return
        layer = self.viewer.layers[_ULTRACK_DB_PREVIEW_LAYER]
        try:
            layer.mouse_drag_callbacks.remove(callback)
        except ValueError:
            pass
        self._ultrack_db_preview_mouse_callback = None

    def _select_ultrack_db_preview_label(
        self, display_label: int, *, frame: int | None = None
    ) -> None:
        node_id = self._ultrack_db_label_to_node_id.get(int(display_label))
        if node_id is None:
            self._set_ultrack_db_status(f"No DB node mapped to label {display_label}.")
            self._clear_ultrack_db_highlight()
            return
        selected_frame = self._current_t() if frame is None else int(frame)
        self._ultrack_db_selected_node_id = int(node_id)
        self._ultrack_db_selected_frame = selected_frame
        self._update_ultrack_db_highlight(self._ultrack_db_preview_labels, int(display_label))
        annot = self._ultrack_db_node_annotations.get(int(node_id), "UNKNOWN")
        annot_suffix = "" if annot == "UNKNOWN" else f" [{annot}]"
        self._set_ultrack_db_status(
            f"Selected node {node_id}{annot_suffix} at t={selected_frame}."
        )
        if self.ultrack_db_connected_focus_check.isChecked():
            self._refresh_ultrack_db_browser()

    def _refresh_ultrack_db_selection_highlight(
        self,
        labels: np.ndarray,
        status: str,
        node_id_to_label: dict[int, int],
        frame: int,
    ) -> str:
        selected_node_id = self._ultrack_db_selected_node_id
        if selected_node_id is None:
            self._clear_ultrack_db_highlight()
            return status
        display_label = node_id_to_label.get(int(selected_node_id))
        if display_label is None:
            self._clear_ultrack_db_highlight()
            annot = self._query_ultrack_db_node_annotation_for_status(
                node_id_to_label, selected_node_id
            )
            if annot in {"REAL", "FAKE"}:
                return (
                    f"{status} Selected node {selected_node_id} [{annot}] is hidden "
                    f"by annotation filter at frame {frame}."
                )
            return (
                f"{status} Selected node {selected_node_id} is hidden "
                f"at frame {frame} and the current hierarchy threshold."
            )
        self._update_ultrack_db_highlight(labels, int(display_label))
        return status

    def _query_ultrack_db_node_annotation_for_status(
        self, node_id_to_label: dict[int, int], selected_node_id: int
    ) -> str:
        return self._ultrack_db_node_annotations.get(int(selected_node_id), "UNKNOWN")

    def _get_ultrack_db_highlight_layer(self):
        if _ULTRACK_DB_SELECTION_LAYER in self.viewer.layers:
            return self.viewer.layers[_ULTRACK_DB_SELECTION_LAYER]
        layer = self.viewer.add_shapes(
            name=_ULTRACK_DB_SELECTION_LAYER,
            ndim=2,
            edge_color="cyan",
            edge_width=2,
            face_color="transparent",
        )
        layer.visible = False
        return layer

    def _update_ultrack_db_highlight(
        self, labels: np.ndarray | None, display_label: int
    ) -> None:
        layer = self._get_ultrack_db_highlight_layer()
        if labels is None or display_label == 0:
            layer.data = []
            layer.visible = False
            return
        mask = (labels == int(display_label)).astype(np.uint8)
        if not np.any(mask):
            layer.data = []
            layer.visible = False
            return
        from skimage.measure import find_contours

        contours = find_contours(mask, level=0.5)
        if not contours:
            layer.data = []
            layer.visible = False
            return
        layer.data = [max(contours, key=len)]
        layer.shape_type = ["polygon"]
        layer.visible = True

    def _clear_ultrack_db_highlight(self) -> None:
        if _ULTRACK_DB_SELECTION_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[_ULTRACK_DB_SELECTION_LAYER]
        layer.data = []
        layer.visible = False

    def _query_ultrack_db_connected_nodes(
        self, db_path: Path, selected_node_id: int
    ) -> tuple[dict[int, float], dict[int, float]]:
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import LinkDB

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        predecessors: dict[int, float] = {}
        successors: dict[int, float] = {}
        try:
            with Session(engine) as session:
                rows = (
                    session.query(LinkDB.source_id, LinkDB.target_id, LinkDB.weight)
                    .filter(
                        (LinkDB.source_id == int(selected_node_id))
                        | (LinkDB.target_id == int(selected_node_id))
                    )
                    .all()
                )
                for source_id, target_id, weight in rows:
                    weight_f = float(weight if weight is not None else 1.0)
                    if int(target_id) == int(selected_node_id):
                        source_i = int(source_id)
                        predecessors[source_i] = predecessors.get(source_i, 1.0) * weight_f
                    if int(source_id) == int(selected_node_id):
                        target_i = int(target_id)
                        successors[target_i] = successors.get(target_i, 1.0) * weight_f
        finally:
            engine.dispose()
        return predecessors, successors

    def _render_ultrack_db_connected_focus(
        self,
        db_path: Path,
        frame: int,
        labels: np.ndarray,
        status: str,
        prob_dict: dict[int, float],
        label_to_node_id: dict[int, int],
        node_id_to_label: dict[int, int],
    ) -> tuple[np.ndarray, str, dict[int, float]]:
        selected_node_id = self._ultrack_db_selected_node_id
        selected_frame = self._ultrack_db_selected_frame
        if selected_node_id is None or selected_frame is None:
            self._clear_ultrack_db_highlight()
            return labels, f"{status} Click a DB preview node to focus links.", {}

        predecessors, successors = self._query_ultrack_db_connected_nodes(
            db_path, selected_node_id
        )
        if frame == selected_frame:
            relation = "selected"
            allowed_weights = {selected_node_id: 1.0}
            if int(selected_node_id) not in node_id_to_label:
                self._clear_ultrack_db_highlight()
                empty = np.zeros_like(labels, dtype=np.uint32)
                annot = self._ultrack_db_node_annotations.get(int(selected_node_id), "UNKNOWN")
                annot_suffix = "" if annot == "UNKNOWN" else f" [{annot}]"
                return (
                    empty,
                    f"Selected node {selected_node_id}{annot_suffix} at t={selected_frame} is "
                    "hidden by the current threshold or annotation filter.",
                    {},
                )
        elif frame == selected_frame - 1:
            relation = "t-1"
            allowed_weights = predecessors
        elif frame == selected_frame + 1:
            relation = "t+1"
            allowed_weights = successors
        else:
            empty = np.zeros_like(labels, dtype=np.uint32)
            self._clear_ultrack_db_highlight()
            return (
                empty,
                f"Selected node {selected_node_id} at t={selected_frame} | "
                f"frame {frame}: outside connected focus.",
                {},
            )

        focused = np.zeros_like(labels, dtype=np.uint32)
        alpha_dict: dict[int, float] = {}
        for label_id, node_id in label_to_node_id.items():
            label_i = int(label_id)
            node_i = int(node_id)
            if node_i not in allowed_weights:
                continue
            focused[labels == label_i] = label_i
            alpha_enabled = (
                self.ultrack_db_edge_alpha_check.isChecked()
                or self.ultrack_db_prob_alpha_check.isChecked()
            )
            if alpha_enabled:
                if node_i == selected_node_id:
                    alpha_dict[label_i] = 1.0
                else:
                    alpha_dict[label_i] = self._ultrack_db_connected_alpha(
                        label_i,
                        float(allowed_weights[node_i]),
                        prob_dict,
                    )

        selected_label = node_id_to_label.get(int(selected_node_id))
        if frame == selected_frame and selected_label is not None:
            self._update_ultrack_db_highlight(focused, int(selected_label))
        else:
            self._clear_ultrack_db_highlight()

        edge_values = [
            float(v)
            for node_id, v in allowed_weights.items()
            if node_id in node_id_to_label and node_id != selected_node_id
        ]
        if edge_values:
            edge_summary = (
                f" | edge product range {min(edge_values):.2f}-{max(edge_values):.2f}"
            )
        else:
            edge_summary = ""
        count = int(np.unique(focused[focused != 0]).size)
        annot = self._ultrack_db_node_annotations.get(int(selected_node_id), "UNKNOWN")
        annot_suffix = "" if annot == "UNKNOWN" else f" [{annot}]"
        return (
            focused,
            f"Selected node {selected_node_id}{annot_suffix} at t={selected_frame} | "
            f"{relation}: {count} connected node(s){edge_summary}",
            alpha_dict,
        )

    def _ultrack_db_connected_alpha(
        self,
        label_id: int,
        edge_weight: float,
        prob_dict: dict[int, float],
    ) -> float:
        alpha = 1.0
        if self.ultrack_db_edge_alpha_check.isChecked():
            alpha *= float(edge_weight)
        if self.ultrack_db_prob_alpha_check.isChecked() and prob_dict:
            probs = [float(v) for v in prob_dict.values()]
            min_p = min(probs)
            max_p = max(probs)
            denom = max(max_p - min_p, 1e-9)
            prob = float(prob_dict.get(int(label_id), 1.0))
            alpha *= 0.15 + 0.85 * (prob - min_p) / denom
        return float(np.clip(alpha, 0.05, 1.0))

    def _ultrack_db_summary_text(self, db_path: Path, frame: int) -> str:
        import sqlalchemy as sqla
        from sqlalchemy import func
        from sqlalchemy.orm import Session
        from ultrack.core.database import LinkDB, NodeDB, VarAnnotation

        try:
            from ultrack.core.database import OverlapDB
        except Exception:
            OverlapDB = None

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                n_nodes = int(session.query(func.count(NodeDB.id)).scalar() or 0)
                n_links = int(session.query(func.count(LinkDB.source_id)).scalar() or 0)
                n_real = int(
                    session.query(func.count(NodeDB.id))
                    .filter(NodeDB.node_annot == VarAnnotation.REAL)
                    .scalar() or 0
                )
                n_fake = int(
                    session.query(func.count(NodeDB.id))
                    .filter(NodeDB.node_annot == VarAnnotation.FAKE)
                    .scalar() or 0
                )
                frame_nodes = session.query(NodeDB).filter(NodeDB.t == frame).all()
                selected = sum(1 for n in frame_nodes if getattr(n, "selected", False))
                node_ids = [int(n.id) for n in frame_nodes]
                outgoing = incoming = overlaps = 0
                if node_ids:
                    outgoing = int(
                        session.query(func.count(LinkDB.source_id))
                        .filter(LinkDB.source_id.in_(node_ids))
                        .scalar() or 0
                    )
                    incoming = int(
                        session.query(func.count(LinkDB.target_id))
                        .filter(LinkDB.target_id.in_(node_ids))
                        .scalar() or 0
                    )
                    if OverlapDB is not None:
                        try:
                            overlaps = int(
                                session.query(func.count(OverlapDB.node_id))
                                .filter(
                                    OverlapDB.node_id.in_(node_ids)
                                    | OverlapDB.ancestor_id.in_(node_ids)
                                )
                                .scalar() or 0
                            )
                        except Exception:
                            overlaps = 0
            return (
                f"{n_nodes} nodes | {n_links} links | REAL {n_real} | FAKE {n_fake} | frame {frame}: "
                f"{len(node_ids)} nodes, {selected} selected, "
                f"{incoming} in/{outgoing} out links, {overlaps} overlaps"
            )
        finally:
            engine.dispose()

    def _query_distinct_heights(self, db_path: Path, mtime_ns: int) -> tuple[float, ...]:
        key = (str(db_path.resolve()), mtime_ns)
        cached = self._ultrack_db_height_values_cache.get(key)
        if cached is not None:
            return cached
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                heights = tuple(
                    float(row[0])
                    for row in session.query(NodeDB.height)
                    .distinct()
                    .order_by(NodeDB.height)
                    .all()
                    if row[0] is not None
                )
        finally:
            engine.dispose()
        self._ultrack_db_height_values_cache[key] = heights
        return heights

    def _query_hierarchy_cut_states(
        self, db_path: Path, mtime_ns: int, frame: int
    ) -> tuple[_HierarchyCutState, ...]:
        key = (str(db_path.resolve()), mtime_ns, frame)
        cached = self._ultrack_db_cut_state_cache.get(key)
        if cached is not None:
            return cached

        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
        from ultrack.utils.constants import NO_PARENT

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                rows = [
                    (int(node_id), int(parent_id), float(height))
                    for node_id, parent_id, height in session.query(
                        NodeDB.id, NodeDB.hier_parent_id, NodeDB.height
                    )
                    .filter(NodeDB.t == frame)
                    .order_by(NodeDB.height, NodeDB.id)
                    .all()
                    if height is not None
                ]
        except Exception:
            heights = self._query_distinct_heights(db_path, mtime_ns)
            return tuple(_HierarchyCutState((), float(height)) for height in heights)
        finally:
            engine.dispose()

        if not rows:
            self._ultrack_db_cut_state_cache[key] = ()
            return ()

        node_ids = {node_id for node_id, _parent_id, _height in rows}
        heights_by_id = {node_id: height for node_id, _parent_id, height in rows}
        parent_by_id = {
            node_id: parent_id
            for node_id, parent_id, _height in rows
            if parent_id != NO_PARENT and parent_id in node_ids
        }
        children_by_parent: dict[int, set[int]] = {}
        for child_id, parent_id in parent_by_id.items():
            children_by_parent.setdefault(parent_id, set()).add(child_id)

        active = {
            node_id for node_id, _parent_id, _height in rows
            if node_id not in children_by_parent
        }
        if not active:
            active = set(node_ids)

        states: list[_HierarchyCutState] = []
        seen_states: set[tuple[int, ...]] = set()

        def _append_state() -> None:
            ordered = tuple(
                sorted(active, key=lambda node_id: (heights_by_id[node_id], node_id))
            )
            if ordered in seen_states:
                return
            seen_states.add(ordered)
            height = max((heights_by_id[node_id] for node_id in ordered), default=None)
            states.append(_HierarchyCutState(ordered, height))

        _append_state()
        while True:
            promotable = [
                parent_id
                for parent_id, child_ids in children_by_parent.items()
                if parent_id not in active and child_ids and child_ids.issubset(active)
            ]
            if not promotable:
                break
            min_height = min(heights_by_id[parent_id] for parent_id in promotable)
            promote_now = [
                parent_id
                for parent_id in promotable
                if heights_by_id[parent_id] == min_height
            ]
            for parent_id in sorted(promote_now):
                active.difference_update(children_by_parent[parent_id])
                active.add(parent_id)
            _append_state()

        result = tuple(states)
        self._ultrack_db_cut_state_cache[key] = result
        return result

    def _configure_ultrack_db_hierarchy_slider(
        self, db_path: Path, mtime_ns: int, frame: int
    ) -> tuple[_HierarchyCutState, ...]:
        states = self._query_hierarchy_cut_states(db_path, mtime_ns, frame)
        maximum = max(len(states) - 1, 0)
        value = min(max(int(self.ultrack_db_hierarchy_slider.value()), 0), maximum)

        old_blocked = self.ultrack_db_hierarchy_slider.blockSignals(True)
        try:
            self.ultrack_db_hierarchy_slider.setRange(0, maximum)
            self.ultrack_db_hierarchy_slider.setValue(value)
        finally:
            self.ultrack_db_hierarchy_slider.blockSignals(old_blocked)

        if states:
            self._set_ultrack_db_height_label(value, states[value].height, len(states))
        else:
            self.ultrack_db_height_lbl.setText("—")
        return states

    def _set_ultrack_db_height_label(
        self, index: int, height: float | None, total: int
    ) -> None:
        height_text = "—" if height is None else f"{height:.2f}"
        self.ultrack_db_height_lbl.setText(
            f"i={index} h={height_text} ({index + 1}/{total})"
        )

    def _render_hierarchy_cut(
        self, db_path: Path, frame: int, h_actual: float
    ) -> tuple[
        np.ndarray,
        str,
        dict[int, float],
        dict[int, int],
        dict[int, int],
        dict[int, str],
    ]:
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session, aliased
        from ultrack.core.database import NodeDB
        from ultrack.utils.constants import NO_PARENT

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                P = aliased(NodeDB)
                C = aliased(NodeDB)
                same_height_child_exists = (
                    session.query(C.id)
                    .where(C.hier_parent_id == NodeDB.id)
                    .where(C.height == NodeDB.height)
                    .where(NodeDB.height == h_actual)
                    .exists()
                )
                nodes = (
                    session.query(NodeDB)
                    .outerjoin(P, NodeDB.hier_parent_id == P.id)
                    .where(NodeDB.t == frame)
                    .where(NodeDB.height <= h_actual)
                    .where(
                        (NodeDB.hier_parent_id == NO_PARENT)
                        | ((NodeDB.height < h_actual) & (P.height > h_actual))
                        | ((NodeDB.height == h_actual) & (P.height >= h_actual))
                    )
                    .where(~same_height_child_exists)
                    .all()
                )
        finally:
            engine.dispose()

        return self._finalize_hierarchy_nodes(
            nodes,
            frame,
            empty_msg=f"No segments at this threshold for frame {frame}.",
            status_suffix=f"at h={h_actual:.2f}",
        )

    def _render_hierarchy_cut_state(
        self, db_path: Path, frame: int, state: _HierarchyCutState
    ) -> tuple[
        np.ndarray,
        str,
        dict[int, float],
        dict[int, int],
        dict[int, int],
        dict[int, str],
    ]:
        if not state.node_ids:
            return self._render_hierarchy_cut(db_path, frame, float(state.height or 0.0))

        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB

        engine = sqla.create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        try:
            with Session(engine) as session:
                rows = (
                    session.query(NodeDB)
                    .where(NodeDB.t == frame)
                    .where(NodeDB.id.in_(state.node_ids))
                    .all()
                )
        finally:
            engine.dispose()

        nodes_by_id = {int(node.id): node for node in rows}
        nodes = [
            nodes_by_id[node_id]
            for node_id in state.node_ids
            if node_id in nodes_by_id
        ]
        height_text = "—" if state.height is None else f"{state.height:.2f}"
        return self._finalize_hierarchy_nodes(
            nodes,
            frame,
            empty_msg=f"No hierarchy state segments for frame {frame}.",
            status_suffix=f"at cut state h={height_text}",
        )

    def _finalize_hierarchy_nodes(
        self,
        nodes: list,
        frame: int,
        *,
        empty_msg: str,
        status_suffix: str,
    ) -> tuple[
        np.ndarray,
        str,
        dict[int, float],
        dict[int, int],
        dict[int, int],
        dict[int, str],
    ]:
        if not nodes:
            return self._empty_ultrack_db_preview(), empty_msg, {}, {}, {}, {}
        show_validated = self.ultrack_db_show_validated_check.isChecked()
        show_fake = self.ultrack_db_show_fake_check.isChecked()
        filtered_nodes = []
        hidden_real = hidden_fake = 0
        for node in nodes:
            annot = self._ultrack_db_annotation_name(getattr(node, "node_annot", None))
            if annot == "REAL" and not show_validated:
                hidden_real += 1
                continue
            if annot == "FAKE" and not show_fake:
                hidden_fake += 1
                continue
            filtered_nodes.append(node)
        if not filtered_nodes:
            return self._empty_ultrack_db_preview(), (
                f"Frame {frame}: annotation filters hid all {len(nodes)} segment(s)."
            ), {}, {}, {}, {}
        labels = self._paint_ultrack_db_nodes(filtered_nodes)
        prob_dict, label_to_node_id, node_id_to_label = (
            self._ultrack_db_node_preview_metadata(filtered_nodes)
        )
        node_annotations = self._ultrack_db_node_annotation_metadata(filtered_nodes)
        hidden_summary = ""
        if hidden_real or hidden_fake:
            hidden_summary = f" Hidden by annotation filter: REAL {hidden_real}, FAKE {hidden_fake}."
        return labels, (
            f"Frame {frame}: {len(filtered_nodes)} segment(s) {status_suffix}."
            f"{hidden_summary}"
        ), prob_dict, label_to_node_id, node_id_to_label, node_annotations

    @staticmethod
    def _ultrack_db_annotation_name(value) -> str:
        if value is None:
            return "UNKNOWN"
        raw = getattr(value, "value", value)
        if raw is None:
            return "UNKNOWN"
        name = str(raw).split(".")[-1].upper()
        if name in {"REAL", "FAKE"}:
            return name
        return "UNKNOWN"

    @staticmethod
    def _ultrack_db_node_preview_metadata(
        nodes: list,
    ) -> tuple[dict[int, float], dict[int, int], dict[int, int]]:
        prob_dict: dict[int, float] = {}
        label_to_node_id: dict[int, int] = {}
        node_id_to_label: dict[int, int] = {}
        for label, node in enumerate(nodes, start=1):
            try:
                prob = float(node.node_prob if node.node_prob is not None else 1.0)
            except (TypeError, ValueError):
                prob = 1.0
            prob_dict[label] = prob
            try:
                node_id = int(node.id)
            except (TypeError, ValueError):
                continue
            label_to_node_id[label] = node_id
            node_id_to_label[node_id] = label
        return prob_dict, label_to_node_id, node_id_to_label

    @staticmethod
    def _ultrack_db_node_annotation_metadata(nodes: list) -> dict[int, str]:
        node_annotations: dict[int, str] = {}
        for node in nodes:
            try:
                node_id = int(node.id)
            except (TypeError, ValueError):
                continue
            node_annotations[node_id] = NucleusWorkflowWidget._ultrack_db_annotation_name(
                getattr(node, "node_annot", None)
            )
        return node_annotations

    def _empty_ultrack_db_preview(self) -> np.ndarray:
        shape = self._viewer_plane_shape()
        return np.zeros(shape, dtype=np.uint32)

    def _viewer_plane_shape(self) -> tuple[int, int]:
        for layer in self.viewer.layers:
            data = getattr(layer, "data", None)
            if isinstance(data, np.ndarray) and data.ndim >= 2:
                return tuple(int(v) for v in data.shape[-2:])
        return (1, 1)

    def _paint_ultrack_db_nodes(self, nodes: list) -> np.ndarray:
        masks: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
        max_y = max_x = 0
        for label, node in enumerate(nodes, start=1):
            parsed = self._node_mask_and_bbox(node)
            if parsed is None:
                continue
            bbox, mask = parsed
            y0, x0, y1, x1 = bbox
            max_y = max(max_y, y1)
            max_x = max(max_x, x1)
            masks.append((label, bbox, mask))

        base_y, base_x = self._viewer_plane_shape()
        labels = np.zeros((max(base_y, max_y, 1), max(base_x, max_x, 1)), dtype=np.uint32)
        for label, (y0, x0, y1, x1), mask in masks:
            target = labels[y0:y1, x0:x1]
            if target.shape != mask.shape:
                continue
            target[mask.astype(bool)] = label
        return labels

    @staticmethod
    def _node_mask_and_bbox(node) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        try:
            # MaybePickleType already unpickles on read; only call pickle.loads if raw bytes
            node_obj = node.pickle
            if isinstance(node_obj, (bytes, memoryview)):
                node_obj = pickle.loads(bytes(node_obj))
            if node_obj is None:
                return None
        except Exception:
            return None

        if isinstance(node_obj, dict):
            bbox = node_obj.get("bbox")
            mask = node_obj.get("mask")
        elif isinstance(node_obj, tuple) and len(node_obj) >= 2:
            bbox, mask = node_obj[0], node_obj[1]
        else:
            bbox = getattr(node_obj, "bbox", None)
            mask = getattr(node_obj, "mask", None)

        if bbox is None or mask is None:
            return None
        bbox_arr = np.asarray(bbox, dtype=int).ravel()
        if bbox_arr.size >= 6:
            y0, x0, y1, x1 = int(bbox_arr[1]), int(bbox_arr[2]), int(bbox_arr[4]), int(bbox_arr[5])
        elif bbox_arr.size >= 4:
            y0, x0, y1, x1 = (int(v) for v in bbox_arr[:4])
        else:
            return None

        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
            mask_arr = mask_arr[0]
        elif mask_arr.ndim > 2:
            mask_arr = np.squeeze(mask_arr)
        if mask_arr.ndim != 2:
            return None
        if mask_arr.shape != (y1 - y0, x1 - x0):
            return None
        return (y0, x0, y1, x1), mask_arr.astype(bool, copy=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _current_t(self) -> int:
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) >= 1 else 0

    def _update_tracked_display(
        self,
        labels: np.ndarray,
        t: int | None = None,
    ) -> None:
        if _TRACKED_LAYER in self.viewer.layers and t is not None:
            layer = self.viewer.layers[_TRACKED_LAYER]
            if layer.data.ndim == 3:
                if t < layer.data.shape[0]:
                    new_data = layer.data.copy()
                    new_data[t] = labels
                    layer.data = new_data
                    return
                # Extend the in-memory stack rather than reloading from disk.
                new_data = np.concatenate(
                    [layer.data, labels[np.newaxis].astype(layer.data.dtype)], axis=0
                )
                layer.data = new_data
                return
        display = labels[np.newaxis].copy() if labels.ndim == 2 else labels
        self._update_layer(_TRACKED_LAYER, display)

    def _update_layer(self, name: str, data: np.ndarray) -> None:
        self._update_labels_layer(name, data)

    def _set_viewer_frame(self, t: int) -> None:
        step = list(self.viewer.dims.current_step)
        if not step:
            return
        step[0] = int(t)
        self.viewer.dims.current_step = tuple(step)

    @staticmethod
    def _sigmoid_zavg(stack: np.ndarray) -> np.ndarray:
        zavg_logits = np.asarray(stack, dtype=np.float32).mean(axis=1)
        return (1.0 / (1.0 + np.exp(-zavg_logits))).astype(np.float32)

    def _set_contour_status(self, msg: str) -> None:
        self.contour_status_lbl.setText(msg)
        self.contour_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_correction_status(self, msg: str) -> None:
        self.correction_status_lbl.setText(msg)
        self.correction_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _set_ultrack_status(self, msg: str) -> None:
        self.ultrack_status_lbl.setText(msg)
        self.ultrack_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _on_contour_worker_error(self, exc: Exception) -> None:
        self.build_progress_bar.setVisible(False)
        self._set_contour_status(f"Error: {exc}")
        logger.exception("Contour worker error", exc_info=exc)

    def _on_correction_worker_error(self, exc: Exception) -> None:
        self._set_correction_status(f"Error: {exc}")
        logger.exception("Correction worker error", exc_info=exc)

    def _on_ultrack_worker_error(self, exc: Exception) -> None:
        self.ultrack_progress_bar.setVisible(False)
        self.ultrack_progress_bar.setRange(0, 100)
        self.run_ultrack_btn.setEnabled(True)
        self.ultrack_terminal_btn.setEnabled(True)
        self._set_ultrack_status(f"Error: {exc}")
        logger.exception("Ultrack worker error", exc_info=exc)

    def _cp_gammas(self) -> list[float]:
        """Gamma values to iterate during consensus boundary building."""
        gmin  = self.cp_gamma_min_spin.value()
        gmax  = self.cp_gamma_max_spin.value()
        gstep = self.cp_gamma_step_spin.value()
        return list(np.arange(gmin, gmax + gstep / 2, gstep))


    # ──────────────────────────────────────────────────────────────────────────
    # 1. Contour map build
    # ──────────────────────────────────────────────────────────────────────────

    def _build_consensus_boundary_averaged(
        self,
        prob_3d: np.ndarray,
        dp_3d: np.ndarray,
        thresholds: list[float],
        gammas: list[float],
        *,
        flow_threshold: float = 0.0,
        mask_callback=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from cellflow.segmentation import build_consensus_boundary

        boundary_sum  = None
        foreground_sum = None
        for g_idx, g in enumerate(gammas):
            cb = None
            if mask_callback is not None:
                def cb(masks, i_thresh, *, _gi=g_idx):
                    mask_callback(masks, _gi, i_thresh)
            b, fg = build_consensus_boundary(
                prob_3d,
                dp_3d,
                thresholds,
                gamma=g,
                flow_threshold=flow_threshold,
                mask_callback=cb,
            )
            if boundary_sum is None:
                boundary_sum  = b.copy()
                foreground_sum = fg.copy()
            else:
                boundary_sum  += b
                foreground_sum += fg
        n = len(gammas)
        return boundary_sum / n, foreground_sum / n

    def _on_build_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path   = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return

        thresholds      = list(np.arange(self.cp_min_spin.value(), self.cp_max_spin.value() + self.cp_step_spin.value() / 2, self.cp_step_spin.value()))
        gammas          = self._cp_gammas()
        contour_path    = self._contour_maps_path()
        score_path      = self._foreground_scores_path()
        mask_path       = self._foreground_masks_path()
        foreground_threshold = self.contour_fg_threshold_spin.value()
        flow_threshold = self.contour_flow_threshold_spin.value()
        save_source     = self.save_source_check.isChecked()
        pos_dir         = self._pos_dir
        build_fn        = self._build_consensus_boundary_averaged
        if contour_path is None or score_path is None or mask_path is None:
            self._set_contour_status("No project open.")
            return

        @thread_worker(connect={
            "yielded":   self._on_build_progress,
            "returned":  self._on_build_done,
            "errored":   self._on_contour_worker_error,
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            dp_stack   = np.asarray(tifffile.imread(str(dp_path)),   dtype=np.float32)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis]
            if dp_stack.ndim == 4:
                dp_stack = dp_stack[np.newaxis]

            n_t = prob_stack.shape[0]
            contour_frames:    list[np.ndarray] = []
            foreground_score_frames: list[np.ndarray] = []
            foreground_mask_frames: list[np.ndarray] = []
            source_dir = pos_dir / "2_nucleus/source_labels"

            for t in range(n_t):
                yield (t + 1, n_t, f"Building contour maps and foreground masks: frame {t + 1}/{n_t}…")
                mask_cb = None
                if save_source:
                    source_dir.mkdir(parents=True, exist_ok=True)
                    def mask_cb(masks, g_idx, thresh_idx, *, _t=t):
                        tifffile.imwrite(
                            source_dir / f"masks_t{_t:04d}_g{g_idx:02d}_thr{thresh_idx:02d}.tif",
                            masks, compression="zlib",
                        )
                boundary, foreground_score = build_fn(
                    prob_stack[t],
                    dp_stack[t],
                    thresholds,
                    gammas,
                    flow_threshold=flow_threshold,
                    mask_callback=mask_cb,
                )
                contour_frames.append(boundary.astype(np.float32, copy=False))
                foreground_score = foreground_score.astype(np.float32, copy=False)
                foreground_score_frames.append(foreground_score)
                foreground_mask_frames.append(
                    (foreground_score >= foreground_threshold).astype(np.uint8)
                )

            contour_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(contour_path), np.stack(contour_frames), compression="zlib")
            tifffile.imwrite(str(score_path), np.stack(foreground_score_frames), compression="zlib")
            tifffile.imwrite(str(mask_path), np.stack(foreground_mask_frames), compression="zlib")
            return pos_dir

        gamma_desc = f"γ={gammas[0]:.2f}" if len(gammas) == 1 else f"γ={gammas[0]:.2f}–{gammas[-1]:.2f} ({len(gammas)} steps)"
        self._set_contour_status(f"Building contour maps and foreground masks ({len(thresholds)} cellprob thresholds, {gamma_desc})…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _on_build_done(self, pos_dir: Path) -> None:
        self._build_worker = None
        self._set_build_buttons_running(False)
        self._refresh_stage_files(pos_dir)
        self._set_contour_status("Contour maps and foreground masks built.")

    def _on_cancel_build(self) -> None:
        if self._build_worker is not None:
            self._build_worker.quit()
        self._build_worker = None
        self._set_build_buttons_running(False)
        self._set_contour_status("Build cancelled.")

    def _set_build_buttons_running(self, running: bool) -> None:
        self.build_btn.setEnabled(not running)
        self.preview_contour_btn.setEnabled(not running)
        self.contour_terminal_btn.setEnabled(not running)
        self.preview_contour_filter_btn.setEnabled(not running)
        self.run_contour_filter_btn.setEnabled(not running)
        self.cancel_build_btn.setEnabled(running)
        self.build_progress_bar.setVisible(running)
        if not running:
            self.build_progress_bar.setValue(0)

    def _on_build_progress(self, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            if total > 0:
                self.build_progress_bar.setRange(0, total)
                self.build_progress_bar.setValue(done)
            self._set_contour_status(msg)
        else:
            self._set_contour_status(str(data))

    def _on_preview_contour_maps(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path   = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return

        t_frame    = self._current_t()
        thresholds = list(np.arange(self.cp_min_spin.value(), self.cp_max_spin.value() + self.cp_step_spin.value() / 2, self.cp_step_spin.value()))
        gammas     = self._cp_gammas()
        flow_threshold = self.contour_flow_threshold_spin.value()
        build_fn   = self._build_consensus_boundary_averaged

        def _on_preview_done(result):
            self._build_worker = None
            self._set_build_buttons_running(False)
            boundary, foreground, cellprob_zavg, t_idx = result
            data = np.zeros((cellprob_zavg.shape[0],) + boundary.shape, dtype=boundary.dtype)
            data[t_idx] = boundary
            foreground_score_data = np.zeros(
                (cellprob_zavg.shape[0],) + foreground.shape, dtype=np.float32
            )
            foreground_score_data[t_idx] = foreground
            foreground_mask_data = (
                foreground_score_data >= self.contour_fg_threshold_spin.value()
            ).astype(np.uint8)
            if _CELLPROB_LAYER in self.viewer.layers:
                self.viewer.layers[_CELLPROB_LAYER].data = cellprob_zavg
            else:
                self.viewer.add_image(
                    cellprob_zavg,
                    name=_CELLPROB_LAYER,
                    colormap="inferno",
                    blending="additive",
                    visible=True,
                )
            if _CONTOUR_LAYER in self.viewer.layers:
                self.viewer.layers[_CONTOUR_LAYER].data = data
            else:
                self.viewer.add_image(data, name=_CONTOUR_LAYER, colormap="magma", visible=True)
            if _FOREGROUND_SCORE_LAYER in self.viewer.layers:
                self.viewer.layers[_FOREGROUND_SCORE_LAYER].data = foreground_score_data
            else:
                self.viewer.add_image(
                    foreground_score_data,
                    name=_FOREGROUND_SCORE_LAYER,
                    colormap="viridis",
                    visible=True,
                )
            self._update_layer(_FOREGROUND_MASK_LAYER, foreground_mask_data)
            self._set_viewer_frame(t_idx)
            self._set_contour_status(
                f"Preview contour map and foreground mask t={t_idx} — "
                f"{len(thresholds)} cellprob thresholds, "
                f"{len(gammas)} gamma value(s)"
            )

        @thread_worker(connect={
            "returned": _on_preview_done,
            "errored":  self._on_contour_worker_error,
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            dp_stack   = np.asarray(tifffile.imread(str(dp_path)),   dtype=np.float32)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis]
            if dp_stack.ndim == 4:
                dp_stack = dp_stack[np.newaxis]
            n_t = min(prob_stack.shape[0], dp_stack.shape[0])
            t_idx = min(max(t_frame, 0), n_t - 1)
            boundary, foreground = build_fn(
                prob_stack[t_idx],
                dp_stack[t_idx],
                thresholds,
                gammas,
                flow_threshold=flow_threshold,
            )
            return boundary, foreground, self._sigmoid_zavg(prob_stack), t_idx

        self._set_contour_status(f"Previewing contour map for frame t={t_frame}…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _contour_filter_params_from_ui(self):
        from cellflow.segmentation import ContourFilterParams

        return ContourFilterParams(
            median_kernel_time=int(self.contour_filter_median_time_spin.value()),
            median_kernel_space=int(self.contour_filter_median_space_spin.value()),
            gaussian_sigma_time=float(self.contour_filter_gauss_time_spin.value()),
            gaussian_sigma_space=float(self.contour_filter_gauss_space_spin.value()),
        )

    def _update_contour_image_layer(self, data: np.ndarray) -> None:
        if _CONTOUR_LAYER in self.viewer.layers:
            self.viewer.layers[_CONTOUR_LAYER].data = data
        else:
            self.viewer.add_image(
                data,
                name=_CONTOUR_LAYER,
                colormap="magma",
                visible=True,
            )

    def _on_preview_contour_filter(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        contour_path = self._contour_maps_path()
        if contour_path is None or not contour_path.exists():
            self._set_contour_status(
                "Missing: contour_maps.tif — run Contour Maps first."
            )
            return

        params = self._contour_filter_params_from_ui()

        def _on_preview_done(result):
            self._build_worker = None
            self._set_build_buttons_running(False)
            filtered = result
            self._update_contour_image_layer(filtered)
            self._set_contour_status("Previewed filtered contour maps.")

        @thread_worker(connect={
            "returned": _on_preview_done,
            "errored":  self._on_contour_worker_error,
        })
        def _worker():
            from cellflow.segmentation import compute_filtered_contour_maps

            contours = np.asarray(tifffile.imread(str(contour_path)), dtype=np.float32)
            return compute_filtered_contour_maps(contours, params)

        self._set_contour_status("Previewing filtered contour maps…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _on_run_contour_filter(self) -> None:
        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        contour_path = self._contour_maps_path()
        if contour_path is None or not contour_path.exists():
            self._set_contour_status(
                "Missing: contour_maps.tif — run Contour Maps first."
            )
            return

        params = self._contour_filter_params_from_ui()
        pos_dir = self._pos_dir

        def _on_filter_done(result):
            self._build_worker = None
            self._set_build_buttons_running(False)
            pos_dir, filtered = result
            self._refresh_stage_files(pos_dir)
            self._update_contour_image_layer(filtered)
            self._set_contour_status("Filtered contour maps written to contour_maps.tif.")

        @thread_worker(connect={
            "returned": _on_filter_done,
            "errored":  self._on_contour_worker_error,
        })
        def _worker():
            from cellflow.segmentation import compute_filtered_contour_maps

            contours = np.asarray(tifffile.imread(str(contour_path)), dtype=np.float32)
            filtered = compute_filtered_contour_maps(contours, params)
            tifffile.imwrite(
                str(contour_path),
                filtered.astype(np.float32, copy=False),
                compression="zlib",
                photometric="minisblack",
            )
            return pos_dir, filtered

        self._set_contour_status("Filtering contour_maps.tif…")
        self._set_build_buttons_running(True)
        self._build_worker = _worker()

    def _on_run_contour_terminal(self) -> None:
        import sys
        import tempfile

        if self._pos_dir is None:
            self._set_contour_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path = self._dp_path()
        contour_path = self._contour_maps_path()
        score_path = self._foreground_scores_path()
        mask_path = self._foreground_masks_path()
        if prob_path is None or not prob_path.exists():
            self._set_contour_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_contour_status(f"Missing: {dp_path}")
            return
        if contour_path is None or score_path is None or mask_path is None:
            self._set_contour_status("No project open.")
            return

        thresholds = list(
            np.arange(
                self.cp_min_spin.value(),
                self.cp_max_spin.value() + self.cp_step_spin.value() / 2,
                self.cp_step_spin.value(),
            )
        )
        gammas = self._cp_gammas()
        foreground_threshold = self.contour_fg_threshold_spin.value()
        flow_threshold = self.contour_flow_threshold_spin.value()
        save_source = self.save_source_check.isChecked()
        pos_dir = self._pos_dir

        python_code = (
            "import pathlib\n"
            "import numpy as np\n"
            "import tifffile\n"
            "from cellflow.segmentation import build_consensus_boundary\n"
            f"prob_path = pathlib.Path({str(prob_path)!r})\n"
            f"dp_path = pathlib.Path({str(dp_path)!r})\n"
            f"contour_path = pathlib.Path({str(contour_path)!r})\n"
            f"score_path = pathlib.Path({str(score_path)!r})\n"
            f"mask_path = pathlib.Path({str(mask_path)!r})\n"
            f"save_source = {save_source!r}\n"
            f"source_dir = pathlib.Path({str(pos_dir / '2_nucleus/source_labels')!r})\n"
            f"thresholds = {thresholds!r}\n"
            f"gammas = {gammas!r}\n"
            f"foreground_threshold = {foreground_threshold!r}\n"
            f"flow_threshold = {flow_threshold!r}\n"
            "def build_consensus_boundary_averaged(prob_3d, dp_3d, thresholds, gammas, flow_threshold=0.0, mask_callback=None):\n"
            "    boundary_sum = None\n"
            "    foreground_sum = None\n"
            "    for g_idx, g in enumerate(gammas):\n"
            "        cb = None\n"
            "        if mask_callback is not None:\n"
            "            def cb(masks, i_thresh, *, _gi=g_idx):\n"
            "                mask_callback(masks, _gi, i_thresh)\n"
            "        boundary, foreground = build_consensus_boundary(\n"
            "            prob_3d,\n"
            "            dp_3d,\n"
            "            thresholds,\n"
            "            gamma=g,\n"
            "            flow_threshold=flow_threshold,\n"
            "            mask_callback=cb,\n"
            "        )\n"
            "        if boundary_sum is None:\n"
            "            boundary_sum = boundary.copy()\n"
            "            foreground_sum = foreground.copy()\n"
            "        else:\n"
            "            boundary_sum += boundary\n"
            "            foreground_sum += foreground\n"
            "    n = len(gammas)\n"
            "    return boundary_sum / n, foreground_sum / n\n"
            "prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)\n"
            "dp_stack = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)\n"
            "if prob_stack.ndim == 3:\n"
            "    prob_stack = prob_stack[np.newaxis]\n"
            "if dp_stack.ndim == 4:\n"
            "    dp_stack = dp_stack[np.newaxis]\n"
            "n_t = prob_stack.shape[0]\n"
            "contour_frames = []\n"
            "foreground_score_frames = []\n"
            "foreground_mask_frames = []\n"
            "for t in range(n_t):\n"
            "    print(f'Building contour maps and foreground masks: frame {t + 1}/{n_t}...', flush=True)\n"
            "    mask_cb = None\n"
            "    if save_source:\n"
            "        source_dir.mkdir(parents=True, exist_ok=True)\n"
            "        def mask_cb(masks, g_idx, thresh_idx, *, _t=t):\n"
            "            tifffile.imwrite(\n"
            "                source_dir / f'masks_t{_t:04d}_g{g_idx:02d}_thr{thresh_idx:02d}.tif',\n"
            "                masks,\n"
            "                compression='zlib',\n"
            "            )\n"
            "    boundary, foreground = build_consensus_boundary_averaged(\n"
            "        prob_stack[t],\n"
            "        dp_stack[t],\n"
            "        thresholds,\n"
            "        gammas,\n"
            "        flow_threshold=flow_threshold,\n"
            "        mask_callback=mask_cb,\n"
            "    )\n"
            "    contour_frames.append(boundary.astype(np.float32, copy=False))\n"
            "    foreground = foreground.astype(np.float32, copy=False)\n"
            "    foreground_score_frames.append(foreground)\n"
            "    foreground_mask_frames.append((foreground >= foreground_threshold).astype(np.uint8))\n"
            "contour_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "print('Writing contour maps and foreground masks...', flush=True)\n"
            "tifffile.imwrite(str(contour_path), np.stack(contour_frames), compression='zlib')\n"
            "tifffile.imwrite(str(score_path), np.stack(foreground_score_frames), compression='zlib')\n"
            "tifffile.imwrite(str(mask_path), np.stack(foreground_mask_frames), compression='zlib')\n"
            "print('Done.')\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="cellflow_contour_build_", delete=False
        ) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path)}"
        try:
            from cellflow.napari.utils import launch_in_terminal
            launch_in_terminal(cmd)
            self._set_contour_status("Contour build command launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(cmd)
            self._set_contour_status(
                "Copied contour build command to clipboard (terminal launch unavailable)."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Automated search / propagation
    # ──────────────────────────────────────────────────────────────────────────

    def _on_save_tracked(self) -> None:
        tracked_path = self._tracked_path()
        if tracked_path is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer to save.")
            return
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3:
            self._set_correction_status("Tracked layer is not a 3D stack.")
            return
        n = layer.data.shape[0]
        for t in range(n):
            write_tracked_frame(tracked_path, t, np.asarray(layer.data[t]))
        self._set_correction_status(f"Saved {n} frame(s) to {tracked_path.name}.")

    def _on_load_tracked(self) -> None:
        tracked_path   = self._tracked_path()
        cell_zavg_path = self._cell_zavg_path()
        nuc_zavg_path  = self._nucleus_zavg_path()
        if tracked_path is None or not tracked_path.exists():
            self._set_correction_status("No tracked labels file found.")
            return
        self._set_correction_status("Loading tracked labels…")

        @thread_worker(connect={"returned": self._on_load_tracked_done, "errored": self._on_correction_worker_error})
        def _worker():
            stack = read_full_tracked_stack(tracked_path)
            cell_zavg = (
                np.asarray(tifffile.imread(str(cell_zavg_path)), dtype=np.float32)
                if cell_zavg_path and cell_zavg_path.exists() else None
            )
            nuc_zavg = (
                np.asarray(tifffile.imread(str(nuc_zavg_path)), dtype=np.float32)
                if nuc_zavg_path and nuc_zavg_path.exists() else None
            )
            return stack, cell_zavg, nuc_zavg

        _worker()

    def _on_load_tracked_done(self, result: tuple) -> None:
        stack, cell_zavg, nuc_zavg = result
        nt = stack.shape[0]
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = stack
        else:
            self.viewer.add_labels(stack, name=_TRACKED_LAYER)

        for zavg_data, layer_name, cmap in (
            (cell_zavg, _CELL_ZAVG_LAYER, "gray"),
            (nuc_zavg,  _NUC_ZAVG_LAYER,  "bop orange"),
        ):
            if zavg_data is None:
                continue
            if zavg_data.ndim == 2:
                broadcast_zavg = np.broadcast_to(zavg_data[np.newaxis], (nt,) + zavg_data.shape).copy()
            else:
                broadcast_zavg = zavg_data
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = broadcast_zavg
            else:
                self.viewer.add_image(broadcast_zavg, name=layer_name, colormap=cmap, blending="additive")

        self._set_correction_status(f"Loaded tracked stack {stack.shape} into napari.")
        layer = self.viewer.layers[_TRACKED_LAYER]
        self.correction_widget.activate_layer(layer)
        self.correction_section.expand()

    def set_selection_callback(self, fn) -> None:
        """Register a callback for nucleus correction label selection changes."""
        self.correction_widget.set_selection_callback(fn)

    def select_matching_nucleus_label(
        self,
        t: int,
        source_label: int,
        *,
        source_labels: np.ndarray | None = None,
    ) -> None:
        """Highlight the nucleus label that best overlaps a selected cell label."""
        if _TRACKED_LAYER not in self.viewer.layers:
            return
        if source_labels is None:
            if "Tracked: Cell" not in self.viewer.layers:
                return
            source_labels = np.asarray(self.viewer.layers["Tracked: Cell"].data)
        target_labels = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)
        matched_label = best_overlapping_label(target_labels, source_labels, t, source_label)
        self.correction_widget.select_label(t, matched_label, notify=False)

    def _on_reassign_ids(self) -> None:
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer loaded.")
            return
        stack = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)
        self._set_correction_status("Reassigning cell IDs to contiguous range…")

        @thread_worker(connect={"returned": self._on_reassign_ids_done, "errored": self._on_correction_worker_error})
        def _worker():
            unique_ids = np.unique(stack)
            unique_ids = unique_ids[unique_ids != 0]
            if unique_ids.size == 0:
                return stack, 0, {}
            lut = np.zeros(int(unique_ids.max()) + 1, dtype=np.uint32)
            old_to_new: dict[int, int] = {}
            for new_id, old_id in enumerate(unique_ids, start=1):
                lut[old_id] = new_id
                old_to_new[int(old_id)] = new_id
            return lut[stack], len(unique_ids), old_to_new

        _worker()

    def _on_reassign_ids_done(self, result: tuple) -> None:
        remapped, n_cells, old_to_new = result
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = remapped
        if self._pos_dir is not None and old_to_new:
            remap_validated_tracks(self._pos_dir, old_to_new)
        self._set_correction_status(f"Reassigned {n_cells} cell IDs to contiguous range 1–{n_cells}. Unsaved.")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Tracking & Correction
    # ──────────────────────────────────────────────────────────────────────────

    def _on_ultrack_mode_changed(self, mode: str) -> None:
        self.ultrack_iou_weight_spin.setEnabled(mode == "iou")

    def _set_resolve_prior_controls_enabled(self, _checked: bool | None = None) -> None:
        enabled = self.db_gen_use_validated_check.isChecked()
        for control in (
            self.ultrack_seed_weight_spin,
            self.ultrack_seed_space_spin,
            self.ultrack_seed_time_spin,
            self.ultrack_seed_window_spin,
        ):
            control.setEnabled(enabled)

    def _ultrack_config_from_controls(self) -> UltrackConfig:
        return UltrackConfig(
            seg_min_area=self.db_gen_min_area_spin.value(),
            seg_max_area=self.db_gen_max_area_spin.value(),
            seg_foreground_threshold=self.db_gen_fg_thr_spin.value(),
            seg_min_frontier=self.db_gen_min_frontier_spin.value(),
            seg_ws_hierarchy=self.db_gen_ws_hierarchy_combo.currentText(),
            seg_n_workers=self.db_gen_n_workers_spin.value(),
            max_distance=self.db_gen_max_dist_spin.value(),
            max_neighbors=self.db_gen_max_neighbors_spin.value(),
            linking_mode=self.db_gen_linking_mode_combo.currentText(),
            iou_weight=self.db_gen_iou_weight_spin.value(),
            quality_weight=self.db_gen_quality_weight_spin.value(),
            quality_exponent=self.db_gen_quality_exp_spin.value(),
            circularity_weight=self.db_gen_circularity_weight_spin.value(),
            power=self.ultrack_power_spin.value(),
            appear_weight=self.ultrack_appear_spin.value(),
            disappear_weight=self.ultrack_disappear_spin.value(),
            division_weight=self.ultrack_division_spin.value(),
        )

    def _on_run_ultrack(self) -> None:
        if self._pos_dir is None:
            self._set_ultrack_status("No project open.")
            return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._set_ultrack_status("data.db not found — run DB generation first.")
            return
        working_dir = self._ultrack_workdir()
        tracked_path = self._tracked_path()

        cfg = self._ultrack_config_from_controls()
        needs_validated_export = database_has_annotations(working_dir)
        validated_tracks = None
        tracked_labels = None
        if needs_validated_export:
            validated_tracks = read_validated_tracks(self._pos_dir)
            if not validated_tracks:
                self._set_ultrack_status(
                    "Annotated data.db requires validated tracks for ID-preserving export."
                )
                return
            if _TRACKED_LAYER not in self.viewer.layers:
                self._set_ultrack_status(
                    "Annotated data.db requires the current tracked layer for ID-preserving export."
                )
                return
            tracked_labels = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)

        self.ultrack_progress_bar.setRange(0, 100)
        self.ultrack_progress_bar.setVisible(True)
        self.ultrack_progress_bar.setValue(0)
        self._set_ultrack_status("Starting Ultrack solve…")
        self.run_ultrack_btn.setEnabled(False)
        self.ultrack_terminal_btn.setEnabled(False)

        @thread_worker(connect={
            "yielded":  self._on_ultrack_progress,
            "returned": self._on_run_ultrack_done,
            "errored":  self._on_ultrack_worker_error,
        })
        def _worker():
            for step, total, label in run_solve(working_dir, cfg, overwrite=True):
                yield ("solve", step, total, label)
            yield ("export", 0, 1, "Exporting tracked labels…")
            return export_tracked_labels(
                working_dir,
                cfg,
                tracked_path,
                validated_tracks=validated_tracks,
                tracked_labels=tracked_labels,
            )

        _worker()

    def _on_ultrack_progress(self, payload: tuple) -> None:
        stage, step, total, label = payload
        self._set_ultrack_status(f"[{stage}] {label}")
        if total > 0:
            self.ultrack_progress_bar.setValue(int(100 * step / total))

    def _on_run_ultrack_done(self, labels: np.ndarray | None) -> None:
        self.ultrack_progress_bar.setVisible(False)
        self.run_ultrack_btn.setEnabled(True)
        self.ultrack_terminal_btn.setEnabled(True)
        if labels is None:
            self._set_ultrack_status("Ultrack tracking failed (no output).")
            return
        # Normalize (T, 1, Y, X) → (T, Y, X)
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        nt = labels.shape[0]
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = labels
        else:
            self.viewer.add_labels(labels, name=_TRACKED_LAYER)
        layer = self.viewer.layers[_TRACKED_LAYER]
        self.correction_widget.activate_layer(layer)
        self._refresh_stage_files()
        self._set_ultrack_status(f"Tracking done: {nt} frame(s). Unsaved.")

    def _on_ultrack_terminal(self) -> None:
        import sys
        import tempfile

        if self._pos_dir is None:
            self._set_ultrack_status("No project open.")
            return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._set_ultrack_status("data.db not found — run DB generation first.")
            return
        working_dir = self._ultrack_workdir()
        tracked_path = self._tracked_path()

        cfg = self._ultrack_config_from_controls()
        needs_validated_export = database_has_annotations(working_dir)
        if needs_validated_export:
            validated_tracks = read_validated_tracks(self._pos_dir)
            if not validated_tracks:
                self._set_ultrack_status(
                    "Annotated data.db requires validated tracks for ID-preserving export."
                )
                return
            if tracked_path is None or not tracked_path.exists():
                self._set_ultrack_status(
                    "Annotated data.db requires current tracked labels for ID-preserving export."
                )
                return

        # NOTE: body must live under `if __name__ == "__main__":` because
        # Ultrack's linker uses spawn-based multiprocessing, which re-executes
        # this script in each child via runpy with run_name="__mp_main__".
        # Without the guard, every worker re-runs the full pipeline and races
        # the parent on the SQLite DB.
        python_code = (
            "import sys, pathlib\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'src'))\n"
            "from cellflow.tracking_ultrack.config import TrackingConfig\n"
            "from cellflow.tracking_ultrack.solve import run_solve\n"
            "from cellflow.tracking_ultrack.export import export_tracked_labels\n"
            "from cellflow.database.tracked import read_full_tracked_stack\n"
            "from cellflow.database.validation import read_validated_tracks\n"
            "\n"
            "if __name__ == '__main__':\n"
            f"    pos_dir = pathlib.Path({str(self._pos_dir)!r})\n"
            f"    working_dir = pathlib.Path({str(working_dir)!r})\n"
            f"    tracked_path= pathlib.Path({str(tracked_path)!r})\n"
            f"    needs_validated_export = {bool(needs_validated_export)!r}\n"
            f"    cfg = TrackingConfig(\n"
            f"        power={cfg.power},\n"
            f"        appear_weight={cfg.appear_weight},\n"
            f"        disappear_weight={cfg.disappear_weight},\n"
            f"        division_weight={cfg.division_weight},\n"
            f"        solution_gap={cfg.solution_gap},\n"
            f"        time_limit={cfg.time_limit},\n"
            f"        window_size={cfg.window_size},\n"
            f"    )\n"
            "    print('[1/2] Solving ILP…', flush=True)\n"
            "    for step, total, label in run_solve(working_dir, cfg, overwrite=True):\n"
            "        print(f'  [{step}/{total}] {label}', flush=True)\n"
            "    print('[2/2] Exporting…', flush=True)\n"
            "    validated_tracks = read_validated_tracks(pos_dir) if needs_validated_export else None\n"
            "    tracked_labels = read_full_tracked_stack(tracked_path) if needs_validated_export else None\n"
            "    labels = export_tracked_labels(\n"
            "        working_dir,\n"
            "        cfg,\n"
            "        tracked_path,\n"
            "        validated_tracks=validated_tracks,\n"
            "        tracked_labels=tracked_labels,\n"
            "    )\n"
            f"    print(f'Done — {{labels.shape}} written to {{tracked_path}}', flush=True)\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="cellflow_ultrack_", delete=False
        ) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        cmd = f"{shlex.quote(sys.executable)} {shlex.quote(tmp_path)}"
        try:
            from cellflow.napari.utils import launch_in_terminal
            launch_in_terminal(cmd)
            self._set_ultrack_status("Ultrack command launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(cmd)
            self._set_ultrack_status("Copied Ultrack command to clipboard (terminal launch unavailable).")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Manual correction
    # ──────────────────────────────────────────────────────────────────────────

    def _on_dims_step_changed(self, event=None) -> None:
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        if self.ultrack_db_browser_section.is_expanded:
            from qtpy.QtCore import QTimer
            QTimer.singleShot(0, self._refresh_ultrack_db_browser)

    @staticmethod
    def _frame_view_2d(arr: np.ndarray, t: int) -> np.ndarray | None:
        """Return a 2D (Y, X) view of frame t from a (T, Y, X) or (T, 1, Y, X) stack."""
        if arr.ndim < 3 or t < 0 or t >= arr.shape[0]:
            return None
        v = arr[t]
        while v.ndim > 2:
            if v.shape[0] != 1:
                return None
            v = v[0]
        return v

    def _current_cell_ids(self, t: int) -> set[int]:
        """Return the set of non-zero cell IDs in the tracked layer at frame t."""
        if _TRACKED_LAYER not in self.viewer.layers:
            return set()
        layer = self.viewer.layers[_TRACKED_LAYER]
        frame = self._frame_view_2d(layer.data, t)
        if frame is None:
            return set()
        return set(int(v) for v in np.unique(frame)) - {0}

    def _refresh_validated_overlay(self) -> None:
        """Rebuild the green overlay layer from current frame's validated cells."""
        if self._pos_dir is None or _TRACKED_LAYER not in self.viewer.layers:
            if _VALIDATED_OVERLAY in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[_VALIDATED_OVERLAY])
            return
        tracked = self.viewer.layers[_TRACKED_LAYER]
        if tracked.data.ndim < 3:
            return
        t = self._current_t()
        if t >= tracked.data.shape[0]:
            return
        frame = self._frame_view_2d(tracked.data, t)
        if frame is None:
            return
        validated_ids = read_validated_cells_at_frame(self._pos_dir, t)
        overlay_exists = _VALIDATED_OVERLAY in self.viewer.layers
        if not validated_ids and not overlay_exists:
            # Nothing to draw and no overlay yet — skip creating one. This avoids
            # adding a layer during napari's own layer-insertion event chain
            # (which would re-enter and crash vispy's _reorder_layers).
            return
        if validated_ids:
            mask2d = np.isin(frame, list(validated_ids)).astype(np.uint8)
        else:
            mask2d = np.zeros(frame.shape, dtype=np.uint8)
        full = np.zeros(tracked.data.shape, dtype=np.uint8)
        full[t] = mask2d
        if overlay_exists:
            self.viewer.layers[_VALIDATED_OVERLAY].data = full
        else:
            from qtpy.QtCore import QTimer
            # Defer the add so we don't run inside napari's insert-event chain.
            QTimer.singleShot(0, lambda data=full: self._add_validated_overlay(data))

    def _add_validated_overlay(self, data: np.ndarray) -> None:
        if _VALIDATED_OVERLAY in self.viewer.layers:
            layer = self.viewer.layers[_VALIDATED_OVERLAY]
            layer.data = data
            layer.opacity = _VALIDATED_OVERLAY_OPACITY
            self._place_validated_overlay_below_spotlight()
            return
        ov = self.viewer.add_labels(
            data,
            name=_VALIDATED_OVERLAY,
            opacity=_VALIDATED_OVERLAY_OPACITY,
            colormap=direct_colormap({None: (0, 0, 0, 0), 1: "#00ff00"}),
        )
        self._place_validated_overlay_below_spotlight()
        # Send the active layer back to tracked so corrections still target it.
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[_TRACKED_LAYER]

    def _place_validated_overlay_below_spotlight(self) -> None:
        if _VALIDATED_OVERLAY not in self.viewer.layers or _SPOTLIGHT_LAYER not in self.viewer.layers:
            return
        validated_index = self.viewer.layers.index(_VALIDATED_OVERLAY)
        spotlight_index = self.viewer.layers.index(_SPOTLIGHT_LAYER)
        if validated_index > spotlight_index:
            self.viewer.layers.move(validated_index, spotlight_index)

    def _refresh_validation_counter(self) -> None:
        """Update 'N tracks validated, M cell-frames covered' label."""
        if self._pos_dir is None or _TRACKED_LAYER not in self.viewer.layers:
            self.validation_counter_lbl.setText("")
            return
        validated_tracks = read_validated_tracks(self._pos_dir)
        n_tracks = len(validated_tracks)
        n_cellframes = sum(len(frames) for frames in validated_tracks.values())
        self.validation_counter_lbl.setText(
            f"{n_tracks} track(s) validated, {n_cellframes} cell-frame(s) covered"
        )

    def _on_cells_edited(self, t: int, changed_ids: set[int]) -> None:
        """Callback registered with CorrectionWidget. Invalidate any edited cell IDs."""
        if self._pos_dir is None:
            return
        for cell_id in changed_ids:
            invalidate_track(self._pos_dir, cell_id)
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _frames_with_cell(self, cell_id: int) -> list[int]:
        """Return sorted list of frame indices where cell_id is present in the tracked layer."""
        if cell_id == 0 or _TRACKED_LAYER not in self.viewer.layers:
            return []
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim < 3:
            return []
        # Compare on the whole stack at once — np.any over the spatial axes is cheap.
        nt = layer.data.shape[0]
        spatial_axes = tuple(range(1, layer.data.ndim))
        present = np.any(layer.data == cell_id, axis=spatial_axes)
        return [int(t) for t in np.where(present)[0]]

    def _install_correction_shortcuts(self) -> None:
        specs = [
            ("A", lambda: self._on_extend(direction="backward")),
            ("D", lambda: self._on_extend(direction="forward")),
            ("Q", self._on_retrack_backward),
            ("E", self._on_retrack_forward),
        ]
        self._correction_shortcuts: list[QShortcut] = []
        for key, slot in specs:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.setEnabled(False)
            sc.activated.connect(slot)
            self._correction_shortcuts.append(sc)

    def _on_correction_mode_toggled(self, active: bool) -> None:
        for sc in self._correction_shortcuts:
            sc.setEnabled(active)

    def _kb_toggle_cell_validation(self, _viewer) -> None:
        if self._pos_dir is None:
            return
        sel = self.correction_widget._selected_label
        if not sel:
            self._set_correction_status("Validation toggle: no cell selected (left-click a cell first).")
            return
        t = self._current_t()
        if sel not in self._current_cell_ids(t):
            self._set_correction_status(f"Cell {sel} not present at t={t}.")
            return
        frames = self._frames_with_cell(sel)
        if not frames:
            return
        currently_validated = is_track_validated(self._pos_dir, sel)
        if currently_validated:
            invalidate_track(self._pos_dir, sel)
            self._set_correction_status(f"Cell {sel} invalidated across {len(frames)} frame(s).")
        else:
            validate_track(self._pos_dir, sel, frames)
            self._set_correction_status(f"Cell {sel} validated across {len(frames)} frame(s).")
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _on_remove_unvalidated_labels(self) -> None:
        if self._pos_dir is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer loaded.")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        data = np.asarray(layer.data)
        if data.ndim < 2:
            self._set_correction_status("Tracked layer has no image data.")
            return

        validated_tracks = read_validated_tracks(self._pos_dir)
        frame_count = int(data.shape[0]) if data.ndim >= 3 else 1
        changed_pixels = 0
        changed_frames = 0

        for t in range(frame_count):
            frame = self._frame_view_2d(data, t) if data.ndim >= 3 else data
            if frame is None:
                self._set_correction_status("Tracked layer must be a time-first 2D/3D stack.")
                return
            validated_ids = {
                cell_id
                for cell_id, frames in validated_tracks.items()
                if t in frames
            }
            remove_mask = frame != 0
            if validated_ids:
                remove_mask &= ~np.isin(frame, list(validated_ids))
            n_remove = int(np.count_nonzero(remove_mask))
            if not n_remove:
                continue
            frame[remove_mask] = 0
            changed_pixels += n_remove
            changed_frames += 1

        if not changed_pixels:
            self._set_correction_status("No unvalidated labels found.")
            return

        layer.refresh()
        if self.correction_widget._selected_label:
            current_t = self._current_t()
            if self.correction_widget._selected_label not in self._current_cell_ids(current_t):
                self.correction_widget.select_label(current_t, 0)
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        self._set_correction_status(
            f"Removed unvalidated labels in {changed_frames} frame(s), "
            f"{changed_pixels} px changed. Unsaved."
        )

    def _on_extend_backward(self) -> None:
        self._on_extend(direction="backward")

    def _on_extend_forward(self) -> None:
        self._on_extend(direction="forward")

    def _on_extend(self, direction: str) -> None:
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer loaded.")
            return

        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._set_correction_status(
                "Extend: data.db not found — run DB generation first."
            )
            return

        source_id = self.correction_widget._selected_label
        if not source_id:
            self._set_correction_status("Extend: no cell selected (left-click a cell first).")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        t = self._current_t()
        tracked = np.asarray(layer.data)
        T = tracked.shape[0]

        target_frame = t + (1 if direction == "forward" else -1)
        if direction == "forward" and t >= T - 1:
            self._set_correction_status("Already at last frame")
            return
        if direction == "backward" and t <= 0:
            self._set_correction_status("Already at first frame")
            return

        if not np.any(tracked[t] == source_id):
            self._set_correction_status(f"Cell {source_id} not present at t={t}")
            return

        validated_tracks = read_validated_tracks(self._pos_dir) if self._pos_dir is not None else {}
        result = extend_track_from_db(
            source_id=source_id,
            source_frame=t,
            direction=direction,
            tracked_labels=tracked,
            db_path=db_path,
            d_max=float(self.extend_max_dist_spin.value()),
            area_weight=float(self.extend_area_weight_spin.value()),
            iou_weight=float(self.extend_iou_weight_spin.value()),
            distance_weight=float(self.extend_distance_weight_spin.value()),
            overlap_penalty=float(self.extend_overlap_penalty_spin.value()),
            greedy_overwrite=self.extend_greedy_overwrite_check.isChecked(),
            validated_tracks=validated_tracks,
        )

        if result is None:
            self._set_correction_status(
                f"No candidate within {self.extend_max_dist_spin.value():g}px at t={target_frame}"
            )
            return

        assignments = result.assignments or ()
        if not assignments:
            assignments = (
                SimpleNamespace(cell_id=source_id, mask_2d=result.mask_2d),
            )

        frame = layer.data[result.target_frame]
        changed_ids = {int(assignment.cell_id) for assignment in assignments}
        for cell_id in changed_ids:
            frame[frame == cell_id] = 0

        if self.extend_greedy_overwrite_check.isChecked():
            for assignment in assignments:
                frame[assignment.mask_2d] = int(assignment.cell_id)
        else:
            for assignment in assignments:
                paintable = assignment.mask_2d & (frame == 0)
                frame[paintable] = int(assignment.cell_id)
        layer.refresh()

        step = list(self.viewer.dims.current_step)
        step[0] = result.target_frame
        self.viewer.dims.current_step = tuple(step)

        moved_text = (
            f", reassigned {len(changed_ids) - 1} conflict(s)"
            if len(changed_ids) > 1 else ""
        )
        self._set_correction_status(
            f"Extended cell {source_id} → t={result.target_frame}{moved_text} "
            f"(dist={result.centroid_distance:.1f}px, area={result.area_ratio:.2f}, "
            f"iou={result.centroid_corrected_iou:.2f}, overlap={result.existing_overlap:.2f})"
        )

    def _on_retrack_forward(self) -> None:
        if self._pos_dir is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer loaded.")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._set_correction_status("Tracked layer must be a stack of at least 2 frames.")
            return

        t0 = int(self.viewer.dims.current_step[0])
        if t0 >= layer.data.shape[0] - 1:
            self._set_correction_status("Already at last frame — nothing to retrack forward.")
            return

        T = layer.data.shape[0]
        stack = layer.data.copy()
        fully_validated = read_validated_frames(self._pos_dir)
        reserved_ids = set(read_validated_tracks(self._pos_dir))

        n_retracked = 0
        n_skipped = 0
        for t in range(t0 + 1, T):
            if t in fully_validated:
                n_skipped += 1
                continue
            ref = stack[t - 1]
            tgt = stack[t]
            locked = read_validated_cells_at_frame(self._pos_dir, t)
            stack[t] = retrack_frame_constrained(
                ref,
                tgt,
                locked,
                max_dist_px=float(self.retrack_max_dist_spin.value()),
                reserved_ids=reserved_ids,
            )
            n_retracked += 1

        layer.data = stack
        self._set_correction_status(
            f"Retracked forward from t={t0 + 1}: {n_retracked} frame(s) updated, "
            f"{n_skipped} fully-validated frame(s) skipped. Unsaved."
        )

    def _on_retrack_backward(self) -> None:
        if self._pos_dir is None:
            self._set_correction_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_correction_status("No tracked layer loaded.")
            return

        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._set_correction_status("Tracked layer must be a stack of at least 2 frames.")
            return

        t0 = int(self.viewer.dims.current_step[0])
        if t0 <= 0:
            self._set_correction_status("Already at first frame — nothing to retrack backward.")
            return

        stack = layer.data.copy()
        fully_validated = read_validated_frames(self._pos_dir)
        reserved_ids = set(read_validated_tracks(self._pos_dir))

        n_retracked = 0
        n_skipped = 0
        for t in range(t0 - 1, -1, -1):
            if t in fully_validated:
                n_skipped += 1
                continue
            ref = stack[t + 1]
            tgt = stack[t]
            locked = read_validated_cells_at_frame(self._pos_dir, t)
            stack[t] = retrack_frame_constrained(
                ref,
                tgt,
                locked,
                max_dist_px=float(self.retrack_max_dist_spin.value()),
                reserved_ids=reserved_ids,
            )
            n_retracked += 1

        layer.data = stack
        self._set_correction_status(
            f"Retracked backward from t={t0 - 1}: {n_retracked} frame(s) updated, "
            f"{n_skipped} fully-validated frame(s) skipped. Unsaved."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # State persistence
    # ──────────────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "save_source":      self.save_source_check.isChecked(),
            "cellprob": {
                "min":       self.cp_min_spin.value(),
                "max":       self.cp_max_spin.value(),
                "step":      self.cp_step_spin.value(),
                "gamma_min": self.cp_gamma_min_spin.value(),
                "gamma_max": self.cp_gamma_max_spin.value(),
                "gamma_step": self.cp_gamma_step_spin.value(),
                "foreground_threshold": self.contour_fg_threshold_spin.value(),
                "flow_threshold": self.contour_flow_threshold_spin.value(),
            },
            "contour_filter": {
                "median_time": self.contour_filter_median_time_spin.value(),
                "median_space": self.contour_filter_median_space_spin.value(),
                "gauss_time": self.contour_filter_gauss_time_spin.value(),
                "gauss_space": self.contour_filter_gauss_space_spin.value(),
            },
            "db_generation": {
                "min_area":         self.db_gen_min_area_spin.value(),
                "max_area":         self.db_gen_max_area_spin.value(),
                "fg_threshold":     self.db_gen_fg_thr_spin.value(),
                "min_frontier":     self.db_gen_min_frontier_spin.value(),
                "ws_hierarchy":     self.db_gen_ws_hierarchy_combo.currentText(),
                "max_distance":     self.db_gen_max_dist_spin.value(),
                "max_neighbors":    self.db_gen_max_neighbors_spin.value(),
                "linking_mode":     self.db_gen_linking_mode_combo.currentText(),
                "iou_weight":       self.db_gen_iou_weight_spin.value(),
                "quality_weight":   self.db_gen_quality_weight_spin.value(),
                "quality_exponent": self.db_gen_quality_exp_spin.value(),
                "circularity_weight": self.db_gen_circularity_weight_spin.value(),
                "power":            self.db_gen_power_spin.value(),
                "n_workers":        self.db_gen_n_workers_spin.value(),
                "use_validated":    self.db_gen_use_validated_check.isChecked(),
                "seed_weight":      self.ultrack_seed_weight_spin.value(),
                "seed_sigma_space": self.ultrack_seed_space_spin.value(),
                "seed_tau_time":    self.ultrack_seed_time_spin.value(),
                "seed_max_dt":      self.ultrack_seed_window_spin.value(),
            },
            "extend": {
                "max_distance":     self.extend_max_dist_spin.value(),
                "area_weight":      self.extend_area_weight_spin.value(),
                "iou_weight":       self.extend_iou_weight_spin.value(),
                "distance_weight":  self.extend_distance_weight_spin.value(),
                "overlap_penalty":  self.extend_overlap_penalty_spin.value(),
                "greedy_overwrite": self.extend_greedy_overwrite_check.isChecked(),
            },
            "ultrack": {
                "max_partitions":   self.ultrack_max_partitions_spin.value(),
                "n_frames":         self.ultrack_n_frames_spin.value(),
                "appear_weight":    self.ultrack_appear_spin.value(),
                "disappear_weight": self.ultrack_disappear_spin.value(),
                "division_weight":  self.ultrack_division_spin.value(),
                "power":            self.ultrack_power_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if "save_source" in state:
            self.save_source_check.setChecked(state["save_source"])
        if "cellprob" in state:
            cp = state["cellprob"]
            if "min"        in cp: self.cp_min_spin.setValue(cp["min"])
            if "max"        in cp: self.cp_max_spin.setValue(cp["max"])
            if "step"       in cp: self.cp_step_spin.setValue(cp["step"])
            if "gamma_min"  in cp: self.cp_gamma_min_spin.setValue(cp["gamma_min"])
            if "gamma_max"  in cp: self.cp_gamma_max_spin.setValue(cp["gamma_max"])
            if "gamma_step" in cp: self.cp_gamma_step_spin.setValue(cp["gamma_step"])
            if "flow_threshold" in cp:
                self.contour_flow_threshold_spin.setValue(cp["flow_threshold"])
            if "foreground_threshold" in cp:
                self.contour_fg_threshold_spin.setValue(cp["foreground_threshold"])
        if "contour_filter" in state:
            cf = state["contour_filter"]
            if "median_time" in cf:
                self.contour_filter_median_time_spin.setValue(cf["median_time"])
            if "median_space" in cf:
                self.contour_filter_median_space_spin.setValue(cf["median_space"])
            if "gauss_time" in cf:
                self.contour_filter_gauss_time_spin.setValue(cf["gauss_time"])
            if "gauss_space" in cf:
                self.contour_filter_gauss_space_spin.setValue(cf["gauss_space"])
        if "db_generation" in state:
            dbg = state["db_generation"]
            if "min_area"         in dbg: self.db_gen_min_area_spin.setValue(dbg["min_area"])
            if "max_area"         in dbg: self.db_gen_max_area_spin.setValue(dbg["max_area"])
            if "fg_threshold"     in dbg: self.db_gen_fg_thr_spin.setValue(dbg["fg_threshold"])
            if "min_frontier"     in dbg: self.db_gen_min_frontier_spin.setValue(dbg["min_frontier"])
            if "ws_hierarchy"     in dbg:
                idx = self.db_gen_ws_hierarchy_combo.findText(dbg["ws_hierarchy"])
                if idx >= 0:
                    self.db_gen_ws_hierarchy_combo.setCurrentIndex(idx)
            if "max_distance"     in dbg: self.db_gen_max_dist_spin.setValue(dbg["max_distance"])
            if "max_neighbors"    in dbg: self.db_gen_max_neighbors_spin.setValue(dbg["max_neighbors"])
            if "linking_mode"     in dbg:
                idx = self.db_gen_linking_mode_combo.findText(dbg["linking_mode"])
                if idx >= 0:
                    self.db_gen_linking_mode_combo.setCurrentIndex(idx)
            if "iou_weight"       in dbg: self.db_gen_iou_weight_spin.setValue(dbg["iou_weight"])
            if "quality_weight"   in dbg: self.db_gen_quality_weight_spin.setValue(dbg["quality_weight"])
            if "quality_exponent" in dbg: self.db_gen_quality_exp_spin.setValue(dbg["quality_exponent"])
            if "circularity_weight" in dbg: self.db_gen_circularity_weight_spin.setValue(dbg["circularity_weight"])
            if "power"            in dbg: self.db_gen_power_spin.setValue(dbg["power"])
            if "n_workers"        in dbg: self.db_gen_n_workers_spin.setValue(dbg["n_workers"])
            if "use_validated"    in dbg: self.db_gen_use_validated_check.setChecked(dbg["use_validated"])
            if "seed_weight"      in dbg: self.ultrack_seed_weight_spin.setValue(dbg["seed_weight"])
            if "seed_sigma_space" in dbg: self.ultrack_seed_space_spin.setValue(dbg["seed_sigma_space"])
            if "seed_tau_time"    in dbg: self.ultrack_seed_time_spin.setValue(dbg["seed_tau_time"])
            if "seed_max_dt"      in dbg: self.ultrack_seed_window_spin.setValue(dbg["seed_max_dt"])
        if "extend" in state:
            ext = state["extend"]
            if "max_distance"    in ext: self.extend_max_dist_spin.setValue(ext["max_distance"])
            if "area_weight"     in ext: self.extend_area_weight_spin.setValue(ext["area_weight"])
            if "iou_weight"      in ext: self.extend_iou_weight_spin.setValue(ext["iou_weight"])
            if "distance_weight" in ext: self.extend_distance_weight_spin.setValue(ext["distance_weight"])
            if "overlap_penalty" in ext: self.extend_overlap_penalty_spin.setValue(ext["overlap_penalty"])
            if "greedy_overwrite" in ext: self.extend_greedy_overwrite_check.setChecked(ext["greedy_overwrite"])
        if "search" in state:
            pass  # Old propagator state — silently skip
        if "search_v2" in state:
            pass  # Old propagator v2 state — silently skip
        if "ultrack" in state:
            ul = state["ultrack"]
            if "min_area" in ul and (
                "db_generation" not in state or "min_area" not in state["db_generation"]
            ):
                self.db_gen_min_area_spin.setValue(ul["min_area"])
            if "max_partitions"   in ul: self.ultrack_max_partitions_spin.setValue(ul["max_partitions"])
            if "n_frames"         in ul: self.ultrack_n_frames_spin.setValue(ul["n_frames"])
            if "max_distance" in ul and (
                "db_generation" not in state or "max_distance" not in state["db_generation"]
            ):
                self.db_gen_max_dist_spin.setValue(ul["max_distance"])
            if "linking_mode" in ul and (
                "db_generation" not in state or "linking_mode" not in state["db_generation"]
            ):
                idx = self.db_gen_linking_mode_combo.findText(ul["linking_mode"])
                if idx >= 0:
                    self.db_gen_linking_mode_combo.setCurrentIndex(idx)
            if "iou_weight" in ul and (
                "db_generation" not in state or "iou_weight" not in state["db_generation"]
            ):
                self.db_gen_iou_weight_spin.setValue(ul["iou_weight"])
            if "appear_weight"    in ul: self.ultrack_appear_spin.setValue(ul["appear_weight"])
            if "disappear_weight" in ul: self.ultrack_disappear_spin.setValue(ul["disappear_weight"])
            if "division_weight"  in ul: self.ultrack_division_spin.setValue(ul["division_weight"])
            if "max_neighbors" in ul and (
                "db_generation" not in state or "max_neighbors" not in state["db_generation"]
            ):
                self.db_gen_max_neighbors_spin.setValue(ul["max_neighbors"])
            if "power"            in ul: self.ultrack_power_spin.setValue(ul["power"])
            if "resolve_from_validated" in ul and (
                "db_generation" not in state or "use_validated" not in state["db_generation"]
            ):
                self.db_gen_use_validated_check.setChecked(ul["resolve_from_validated"])
            if "quality_exponent" in ul and (
                "db_generation" not in state
                or "quality_exponent" not in state["db_generation"]
            ):
                self.db_gen_quality_exp_spin.setValue(ul["quality_exponent"])
            if "seed_weight" in ul and (
                "db_generation" not in state or "seed_weight" not in state["db_generation"]
            ):
                self.ultrack_seed_weight_spin.setValue(ul["seed_weight"])
            if "seed_sigma_space" in ul and (
                "db_generation" not in state or "seed_sigma_space" not in state["db_generation"]
            ):
                self.ultrack_seed_space_spin.setValue(ul["seed_sigma_space"])
            if "seed_tau_time" in ul and (
                "db_generation" not in state or "seed_tau_time" not in state["db_generation"]
            ):
                self.ultrack_seed_time_spin.setValue(ul["seed_tau_time"])
            if "seed_max_dt" in ul and (
                "db_generation" not in state or "seed_max_dt" not in state["db_generation"]
            ):
                self.ultrack_seed_window_spin.setValue(ul["seed_max_dt"])
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFormLayout, QGridLayout, QLabel, QSizePolicy

TINY_MARGIN = 2
SECTION_MARGIN = 4
TIGHT_SPACING = 4
DEFAULT_SPIN_WIDTH = 70
DEFAULT_FIELD_SPACING = 8
DEFAULT_ROW_SPACING = 4
DEFAULT_SWEEP_SPIN_WIDTH = 62
BLOCK_GRID_COLUMNS = 4


def _fixed_widget(widget, width=None):
    if width is not None:
        widget.setMaximumWidth(width)
    widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return widget


def compact_spinbox(widget, width=DEFAULT_SPIN_WIDTH):
    return _fixed_widget(widget, width)


def action_button(button, expand=False):
    horizontal_policy = (
        QSizePolicy.Policy.Expanding if expand else QSizePolicy.Policy.Fixed
    )
    button.setSizePolicy(horizontal_policy, QSizePolicy.Policy.Fixed)
    return button


def tiny_button(button):
    button.setStyleSheet("font-size: 8pt; padding: 1px 4px;")
    button.setSizePolicy(
        button.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Fixed
    )
    return button


def icon_button(button, width=24, height=None):
    button.setFixedWidth(width)
    if height is not None:
        button.setFixedHeight(height)
    return button


def muted_label(label, size_pt=8):
    label.setStyleSheet(f"color: palette(mid); font-size: {size_pt}pt;")
    return label


def status_label(label, size_pt=8, italic=False, muted=False):
    style = f"font-size: {size_pt}pt;"
    if muted:
        style += " color: palette(mid);"
    if italic:
        style += " font-style: italic;"
    label.setStyleSheet(style)
    return label


def danger_button(button):
    button.setStyleSheet(
        """
        QPushButton {
            background-color: #b00020;
            color: white;
        }
        QPushButton:hover {
            background-color: #c62828;
        }
        """
    )
    return button


def checked_success_button(button):
    button.setStyleSheet(
        """
        QPushButton:checked {
            background-color: #2e7d32;
            color: white;
            font-weight: bold;
        }
        """
    )
    return button


def compact_form_layout():
    layout = QFormLayout()
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
    layout.setHorizontalSpacing(DEFAULT_FIELD_SPACING)
    layout.setVerticalSpacing(DEFAULT_ROW_SPACING)
    return layout


def block_grid(horizontal_spacing=8, vertical_spacing=4):
    layout = QGridLayout()
    layout.setHorizontalSpacing(horizontal_spacing)
    layout.setVerticalSpacing(vertical_spacing)
    for col in range(BLOCK_GRID_COLUMNS):
        layout.setColumnStretch(col, 0)
    return layout


def two_column_parameter_grid(horizontal_spacing=12, vertical_spacing=4):
    return block_grid(horizontal_spacing, vertical_spacing)


def _block_label(text):
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return label


def _add_block_cell(grid, row, column, widget, span=1, alignment=None):
    if alignment is None:
        grid.addWidget(widget, row, column, 1, span)
    else:
        grid.addWidget(widget, row, column, 1, span, alignment)
    return widget


def add_block_pair_row(
    grid,
    row,
    left_label,
    left_widget,
    right_label=None,
    right_widget=None,
    field_width=70,
):
    left_label_widget = _block_label(left_label)
    _add_block_cell(grid, row, 0, left_label_widget)
    _add_block_cell(grid, row, 1, _fixed_widget(left_widget, field_width))

    right_label_widget = None
    if right_widget is not None:
        right_label_widget = _block_label(right_label or "")
        _add_block_cell(grid, row, 2, right_label_widget)
        _add_block_cell(grid, row, 3, _fixed_widget(right_widget, field_width))

    return left_label_widget, left_widget, right_label_widget, right_widget


def add_block_checkbox_row(grid, row, checkbox):
    _add_block_cell(
        grid,
        row,
        0,
        checkbox,
        span=BLOCK_GRID_COLUMNS,
        alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
    )
    return checkbox


def add_block_button_row(grid, row, *buttons):
    count = len(buttons)
    if count == 0:
        return ()
    if count == 1:
        placements = ((0, 4),)
    elif count == 2:
        placements = ((0, 2), (2, 2))
    elif count == 3:
        placements = ((0, 1), (1, 1), (2, 2))
    elif count == 4:
        placements = ((0, 1), (1, 1), (2, 1), (3, 1))
    else:
        raise ValueError("add_block_button_row supports at most four buttons")

    for button, (column, span) in zip(buttons, placements):
        action_button(button, expand=True)
        _add_block_cell(
            grid,
            row,
            column,
            button,
            span=span,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
    return buttons


def add_parameter_grid_row(grid, row, column, label_text, field):
    base_col = column * 2
    label = _block_label(label_text)
    _add_block_cell(grid, row, base_col, label)
    _add_block_cell(grid, row, base_col + 1, _fixed_widget(field))
    return label, field


def sweep_parameter_grid(
    horizontal_spacing=8,
    vertical_spacing=4,
    spin_width=DEFAULT_SWEEP_SPIN_WIDTH,
):
    layout = block_grid(horizontal_spacing, vertical_spacing)
    layout.setColumnMinimumWidth(1, spin_width)
    layout.setColumnMinimumWidth(2, spin_width)
    layout.setColumnMinimumWidth(3, spin_width)

    layout.addWidget(QLabel(""), 0, 0)
    for col, text in enumerate(("min", "max", "step"), start=1):
        header = QLabel(text)
        header.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(header, 0, col)
    return layout


def add_sweep_parameter_row(
    grid,
    row,
    label_text,
    min_widget,
    max_widget,
    step_widget,
    spin_width=DEFAULT_SWEEP_SPIN_WIDTH,
):
    label = _block_label(label_text)
    _add_block_cell(grid, row, 0, label)
    _add_block_cell(grid, row, 1, compact_spinbox(min_widget, spin_width))
    _add_block_cell(grid, row, 2, compact_spinbox(max_widget, spin_width))
    _add_block_cell(grid, row, 3, compact_spinbox(step_widget, spin_width))
    return label, min_widget, max_widget, step_widget
"""Shared reusable Qt widgets for the CellFlow napari plugin."""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .ui_style import (
    SECTION_MARGIN,
    TIGHT_SPACING,
    TINY_MARGIN,
    icon_button,
    muted_label,
    status_label,
)


class CollapsibleSection(QWidget):
    """A labelled section with a toggle button that shows/hides its inner widget."""

    def __init__(
        self,
        title: str,
        inner: QWidget,
        expanded: bool = False,
        parent: QWidget | None = None,
        title_color: str = "white",
    ) -> None:
        super().__init__(parent)
        self._inner = inner
        self._base_title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, TINY_MARGIN, 0, TINY_MARGIN)
        layout.setSpacing(0)

        # Header toggle button
        self._toggle = QToolButton()
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setText(self._qt_display_text(title))
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.setStyleSheet(
            f"QToolButton {{ font-weight: bold; font-size: 10pt; border: none; "
            f"padding: 2px; color: {title_color}; }}"
        )
        self._toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self._toggle)

        # White-bordered frame that wraps inner content when expanded
        self._content_frame = QFrame()
        self._content_frame.setObjectName("collapsible_content")
        self._content_frame.setFrameShape(QFrame.NoFrame)
        self._content_frame.setStyleSheet(
            "QFrame#collapsible_content { border: 1px solid #666666; "
            "border-radius: 4px; margin: 0px 2px 2px 2px; }"
        )
        frame_layout = QVBoxLayout(self._content_frame)
        frame_layout.setContentsMargins(
            SECTION_MARGIN, SECTION_MARGIN, SECTION_MARGIN, SECTION_MARGIN
        )
        frame_layout.setSpacing(TINY_MARGIN)
        frame_layout.addWidget(inner)

        self._content_frame.setVisible(expanded)
        layout.addWidget(self._content_frame)

        # Always Preferred policy — height is driven by scroll area's minimumHeight
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        if expanded:
            QTimer.singleShot(0, self._notify_layout_change)

    def set_title(self, title: str) -> None:
        """Update the header text."""
        self._base_title = title
        self._toggle.setText(self._qt_display_text(title))

    @property
    def title(self) -> str:
        return self._base_title

    @property
    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def expand(self) -> None:
        self._toggle.setChecked(True)

    def collapse(self) -> None:
        self._toggle.setChecked(False)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content_frame.setVisible(checked)
        QTimer.singleShot(0, self._notify_layout_change)

    @staticmethod
    def _qt_display_text(title: str) -> str:
        """Escape mnemonic markers so literal ampersands render correctly."""
        return title.replace("&", "&&")

    def _notify_layout_change(self) -> None:
        """Propagate geometry changes up the nested collapsible chain."""
        self.updateGeometry()
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, CollapsibleSection) and parent.is_expanded:
                parent.updateGeometry()
                QTimer.singleShot(0, parent._notify_layout_change)
                return
            parent.updateGeometry()
            parent = parent.parent()


# ---------------------------------------------------------------------------
# Pipeline file status rows
# ---------------------------------------------------------------------------

class _PipelineFileRow(QWidget):
    """One pipeline file status row: icon | rel-path | info | [load btn]"""

    def __init__(
        self,
        rel_path: str,
        display_name: str,
        loadable: str | None = None,
        viewer=None,
    ):
        super().__init__()
        self._rel_path = rel_path
        self._loadable = loadable or self._infer_load_kind(rel_path)
        self._full_path: "Path | None" = None
        self._viewer = viewer

        lay = QHBoxLayout(self)
        lay.setContentsMargins(
            SECTION_MARGIN, TINY_MARGIN, SECTION_MARGIN, TINY_MARGIN
        )
        lay.setSpacing(TIGHT_SPACING)

        self._icon_lbl = QLabel("○")
        self._icon_lbl.setFixedWidth(14)
        self._icon_lbl.setAlignment(Qt.AlignCenter)
        muted_label(self._icon_lbl, size_pt=9)
        lay.addWidget(self._icon_lbl)

        name_lbl = QLabel(rel_path)
        name_lbl.setFixedWidth(200)
        status_label(name_lbl)
        name_lbl.setToolTip(display_name)
        lay.addWidget(name_lbl)

        self._info_lbl = QLabel("—")
        self._info_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        status_label(self._info_lbl)
        lay.addWidget(self._info_lbl)

        self._load_btn = QPushButton("↑")
        icon_button(self._load_btn, width=18, height=18)
        self._load_btn.clicked.connect(self._on_load_clicked)
        self._load_btn.setEnabled(False)
        self._load_btn.setToolTip(self._load_tooltip())
        # Hide the button entirely when no viewer is wired in or file is not napari-loadable.
        self._load_btn.setVisible(viewer is not None and self._loadable is not None)
        lay.addWidget(self._load_btn)

    def set_present(self, info_text: str) -> None:
        self._icon_lbl.setText("✓")
        self._icon_lbl.setStyleSheet("font-size: 9pt; font-weight: bold; color: #4CAF50;")
        self._info_lbl.setText(info_text)
        status_label(self._info_lbl)
        self._update_load_button()

    def set_missing(self) -> None:
        self._icon_lbl.setText("✗")
        muted_label(self._icon_lbl, size_pt=9)
        self._info_lbl.setText("missing")
        muted_label(self._info_lbl)
        self._full_path = None
        self._load_btn.setEnabled(False)
        self._load_btn.setToolTip(self._load_tooltip(missing=True))

    def set_no_project(self) -> None:
        self._icon_lbl.setText("○")
        muted_label(self._icon_lbl, size_pt=9)
        self._info_lbl.setText("—")
        muted_label(self._info_lbl)
        self._full_path = None
        self._load_btn.setEnabled(False)
        self._load_btn.setToolTip(self._load_tooltip(no_project=True))

    def _update_load_button(self) -> None:
        if self._full_path is None or not self._full_path.exists():
            self._load_btn.setEnabled(False)
            self._load_btn.setToolTip(self._load_tooltip(missing=True))
            return

        if self._loadable in {"tracked", "labels", "tiff"}:
            self._load_btn.setEnabled(True)
            self._load_btn.setToolTip(self._load_tooltip())
            return

        self._load_btn.setEnabled(False)
        self._load_btn.setToolTip("No direct napari load action for this file.")

    def _on_load_clicked(self) -> None:
        if self._full_path is None or not self._full_path.exists():
            return

        self._load_file_into_viewer()

    def _load_file_into_viewer(self) -> None:
        viewer = self._viewer if self._viewer is not None else self._find_viewer()
        if viewer is None:
            return

        import tifffile

        data = tifffile.imread(str(self._full_path))
        layer_name = self._layer_name()
        use_labels = self._loadable in {"tracked", "labels"}

        if use_labels:
            if layer_name in viewer.layers:
                try:
                    viewer.layers[layer_name].data = data
                    return
                except Exception:
                    viewer.layers.remove(viewer.layers[layer_name])
            viewer.add_labels(data, name=layer_name)
        else:
            colormap = self._pick_colormap()
            if layer_name in viewer.layers:
                try:
                    viewer.layers[layer_name].data = data
                    return
                except Exception:
                    viewer.layers.remove(viewer.layers[layer_name])
            viewer.add_image(data, name=layer_name, colormap=colormap)

    def _pick_colormap(self) -> str:
        rel = self._rel_path
        name = Path(rel).name
        if rel.startswith("0_input/") or name.endswith(("_zavg.tif", "_3dt.tif")):
            return "gray"
        if (
            rel.startswith("1_cellpose/")
            or rel in (
                "2_nucleus/contour_maps.tif",
                "3_cell/filtered_flow_mag.tif",
            )
            or (rel.startswith("2_nucleus/") and name.startswith("foreground_"))
            or (rel.startswith("3_cell/") and name.startswith("foreground_"))
        ):
            return "inferno"
        return "gray"

    def _find_viewer(self):
        widget = self.parentWidget()
        while widget is not None:
            viewer = getattr(widget, "viewer", None)
            if viewer is not None and hasattr(viewer, "add_image") and hasattr(viewer, "add_labels"):
                return viewer
            widget = widget.parentWidget()
        return None

    def _layer_name(self) -> str:
        return Path(self._rel_path).with_suffix("").as_posix().replace("/", "_")

    def _load_tooltip(self, *, missing: bool = False, no_project: bool = False) -> str:
        if no_project:
            return "No project open."
        if missing:
            return "File is missing."
        if self._loadable in {"tracked", "labels"}:
            return "Load labels into napari."
        if self._loadable == "tiff":
            return "Load into napari viewer."
        return "No direct napari load action for this file."

    @staticmethod
    def _infer_load_kind(rel_path: str) -> str | None:
        name = Path(rel_path).name
        if name == "tracked_labels.tif":
            return "tracked"
        if name.endswith("_labels.tif") or ("labels" in name and name.endswith((".tif", ".tiff"))):
            return "labels"
        if name.endswith((".tif", ".tiff")):
            return "tiff"
        return None


def _file_info(path: "Path") -> str:
    """Return a concise shape/dtype string for a pipeline output file."""
    if path.is_dir():
        return "Directory"
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(path) as tif:
                shape = tif.series[0].shape if tif.series else None
            if shape:
                return "×".join(str(d) for d in shape)
        except Exception:
            pass
        return "TIFF"
    if suffix in (".h5", ".hdf5"):
        try:
            import h5py
            shapes = []
            def _collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    shapes.append(f"{name}: " + "×".join(str(d) for d in obj.shape))
            with h5py.File(path, "r") as f:
                f.visititems(_collect)
            if shapes:
                return "; ".join(shapes[:2]) + ("…" if len(shapes) > 2 else "")
        except Exception:
            pass
        kb = path.stat().st_size // 1024
        return f"{kb} KB"
    return f"{path.stat().st_size // 1024} KB"


class PipelineFilesWidget(QWidget):
    """Compact file-status display for pipeline-stage widgets."""

    def __init__(
        self,
        groups: list[tuple[str, list[tuple[str, str]]]],
        parent: QWidget | None = None,
        viewer=None,
    ) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._rows: list[_PipelineFileRow] = []

        for group_label, entries in groups:
            if group_label:
                hdr = QLabel(group_label)
                hdr.setStyleSheet(
                    "font-size: 7pt; font-weight: bold; padding: 1px 4px;"
                    " background: palette(alternateBase); color: palette(mid);"
                )
                lay.addWidget(hdr)
            for rel_path, display_name in entries:
                row = _PipelineFileRow(rel_path, display_name, loadable=None, viewer=viewer)
                self._rows.append(row)
                lay.addWidget(row)

    def refresh(self, pos_dir: "Path" | None) -> None:
        """Update all rows to reflect current on-disk state."""
        if pos_dir is None:
            for row in self._rows:
                row.set_no_project()
            return
        for row in self._rows:
            full_path = pos_dir / row._rel_path
            if full_path.exists():
                row._full_path = full_path
                row.set_present(_file_info(full_path))
            else:
                row.set_missing()
"""Nucleus segmentation via watershed on Cellpose probability maps."""
from __future__ import annotations

from cellflow.segmentation.flow_following import (
    FlowFollowingParams,
    compute_filtered_flow_vectors,
    compute_flow_following_movie,
)
from cellflow.segmentation.cell_label_icm import (
    CellLabelICMParams,
    segment_cells_icm,
)
from cellflow.segmentation.contour_filtering import (
    ContourFilterParams,
    compute_filtered_contour_maps,
)

import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

_LABEL_DTYPE = np.uint32


def apply_gamma(logits: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma-correct Cellpose probability logits: sigmoid → power → logit."""
    if gamma == 1.0:
        return logits
    probs = 1.0 / (1.0 + np.exp(-logits))
    probs = np.clip(np.power(probs, gamma), 1e-7, 1 - 1e-7)
    return np.log(probs / (1.0 - probs))


def _validate_foreground_gamma(gamma: float) -> float:
    gamma = float(gamma)
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    return gamma


def _validate_foreground_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    return threshold


def _apply_post_average_gamma(score: np.ndarray, gamma: float) -> np.ndarray:
    score = np.clip(score, 0.0, 1.0).astype(np.float32, copy=False)
    if gamma == 1.0:
        return score
    return np.power(score, gamma).astype(np.float32)


def _normalize_foreground_score(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    lo = float(np.min(score))
    hi = float(np.max(score))
    if hi <= lo:
        return np.zeros_like(score, dtype=np.float32)
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _flow_dp_magnitude_stack(data: np.ndarray) -> tuple[np.ndarray, bool]:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 3:
        return np.abs(data), False
    if data.ndim == 4:
        if data.shape[-1] in (2, 3):
            return np.sqrt(np.sum(data * data, axis=-1)).astype(np.float32), False
        if data.shape[1] in (2, 3):
            return np.sqrt(np.sum(data * data, axis=1)).astype(np.float32), False
        return np.abs(data), True
    if data.ndim == 5:
        if data.shape[2] in (2, 3):
            axis = 2
        elif data.shape[-1] in (2, 3):
            axis = -1
        else:
            raise ValueError(f"Unsupported flow_dp shape {data.shape}")
        return np.sqrt(np.sum(data * data, axis=axis)).astype(np.float32), True
    raise ValueError(f"Unsupported flow_dp shape {data.shape}")


def foreground_score_stack(data, source: str, gamma: float = 1.0) -> np.ndarray:
    """Return a foreground score image or time stack from probability or flow-DP data."""
    gamma = _validate_foreground_gamma(gamma)
    source_key = str(source).lower()
    arr = np.asarray(data, dtype=np.float32)

    if source_key == "probability":
        if arr.ndim == 3:
            score = 1.0 / (1.0 + np.exp(-arr.mean(axis=0)))
        elif arr.ndim == 4:
            score = 1.0 / (1.0 + np.exp(-arr.mean(axis=1)))
        else:
            raise ValueError(f"Unsupported probability shape {arr.shape}")
        return _apply_post_average_gamma(score, gamma)

    if source_key == "flow_dp":
        magnitude, has_time_axis = _flow_dp_magnitude_stack(arr)
        if has_time_axis:
            score = magnitude.mean(axis=1)
            normalized = np.empty_like(score, dtype=np.float32)
            for t in range(score.shape[0]):
                normalized[t] = _normalize_foreground_score(score[t])
        else:
            normalized = _normalize_foreground_score(magnitude.mean(axis=0))
        return _apply_post_average_gamma(normalized, gamma)

    raise ValueError(f"Unsupported foreground source {source!r}")


def foreground_mask_stack(
    data,
    source: str,
    threshold: float = 0.5,
    gamma: float = 1.0,
) -> np.ndarray:
    """Return a uint8 foreground mask with values 0/1."""
    threshold = _validate_foreground_threshold(threshold)
    score = foreground_score_stack(data, source, gamma=gamma)
    return (score >= threshold).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class ContourWatershedParams:
    """Parameters for contour-map watershed hypothesis generation."""

    seed_distance: int = 10
    foreground_threshold: float = 0.5
    ridge_threshold: float = 0.5
    min_size: int = 0
    min_circularity: float = 0.0
    noise_scale: float = 0.0
    noise_blur_sigma: float = 0.0
    run_index: int = 0

    def to_dict(self) -> dict[str, object]:
        return {"method": "contour_watershed", **asdict(self)}


@dataclass(frozen=True, slots=True)
class CellposeFlowHypothesisParams:
    """Parameters for native Cellpose flow-based mask generation (no sweep)."""

    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.0   # 0 = disabled; >0 removes masks with high flow error
    min_size: int = 15
    niter: int = 200

    def to_dict(self) -> dict[str, object]:
        return {"method": "cellpose_flow", **asdict(self)}


@dataclass(frozen=True, slots=True)
class NucleusHypothesisParams:
    """One parameter set for nucleus hypothesis generation."""

    basin: str = "prob"
    threshold_pct: float = 30.0
    compactness: float = 0.0
    smooth_sigma: float = 0.5
    seed_source: str = "auto"
    seed_distance: int = 5
    min_size: int = 0
    min_circularity: float = 0.0
    z_slice: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_01(arr: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if lo is None:
        lo = float(np.min(arr))
    if hi is None:
        hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, np.nextafter(np.float32(1.0), np.float32(0.0)))
    return scaled.astype(np.float32)


def _flow_magnitude(dp: np.ndarray) -> np.ndarray:
    """Compute L2 magnitude from a DP stack."""
    dp = np.asarray(dp, dtype=np.float32)
    if dp.ndim == 2:
        return np.abs(dp)
    if dp.ndim == 3:
        if dp.shape[0] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=0)).astype(np.float32)
        if dp.shape[-1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=-1)).astype(np.float32)
        return np.abs(dp).astype(np.float32)
    if dp.ndim >= 4:
        if dp.shape[1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=1)).astype(np.float32)
        if dp.shape[-1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=-1)).astype(np.float32)
    raise ValueError(f"Unsupported DP shape for magnitude: {dp.shape}")


def _remove_small_labels(labels: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return labels
    ids, counts = np.unique(labels, return_counts=True)
    small = ids[(ids > 0) & (counts < min_size)]
    if small.size == 0:
        return labels
    out = labels.copy()
    out[np.isin(labels, small)] = 0
    return out


def _remove_low_circularity_labels(labels: np.ndarray, min_circularity: float) -> np.ndarray:
    """Remove labels whose 4π·area/perimeter² is below min_circularity (0 = keep all)."""
    if min_circularity <= 0.0:
        return labels
    from skimage.measure import regionprops

    # Work on a 2D projection if labels is 3D with a single Z
    squeezed = labels.squeeze() if labels.ndim == 3 and labels.shape[0] == 1 else labels
    if squeezed.ndim != 2:
        return labels  # can't compute perimeter on >2D, skip

    import math
    remove = []
    for prop in regionprops(squeezed.astype(np.int32)):
        perimeter = prop.perimeter
        if perimeter < 1e-6:
            remove.append(prop.label)
            continue
        circularity = 4.0 * math.pi * prop.area / (perimeter ** 2)
        if circularity < min_circularity:
            remove.append(prop.label)

    if not remove:
        return labels
    out = labels.copy()
    out[np.isin(labels, remove)] = 0
    return out


def _fill_and_close_labels(labels: np.ndarray) -> np.ndarray:
    """Fill interior holes per label."""
    from scipy.ndimage import binary_fill_holes

    out = np.zeros_like(labels)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        coords = np.nonzero(labels == label_id)
        if not coords or coords[0].size == 0:
            continue
        slices = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coords)
        filled = binary_fill_holes(labels[slices] == label_id)
        out_view = out[slices]
        out_view[filled] = label_id
    return out


def _centroid_markers_2d(labels: np.ndarray) -> np.ndarray:
    """Place one marker pixel at the centroid of each 2D label."""
    labels = np.asarray(labels)
    out = np.zeros_like(labels)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        coords = np.argwhere(labels == label_id)
        centroid = coords.mean(axis=0)
        seed_yx = np.rint(centroid).astype(np.int64)
        if (
            seed_yx[0] < 0
            or seed_yx[0] >= labels.shape[0]
            or seed_yx[1] < 0
            or seed_yx[1] >= labels.shape[1]
            or labels[seed_yx[0], seed_yx[1]] != label_id
        ):
            distances = np.sum((coords - centroid) ** 2, axis=1)
            seed_yx = coords[int(np.argmin(distances))]
        out[int(seed_yx[0]), int(seed_yx[1])] = label_id
    return out


def centroid_markers_from_labels(labels: np.ndarray) -> np.ndarray:
    """Return one centroid seed pixel per non-zero label.

    For a 2D label image, each label is replaced by a single marker pixel at
    its rounded centroid. If the rounded centroid falls outside the label, the
    closest pixel belonging to that label is used instead. For a 3D stack, the
    operation is applied independently to each first-axis plane, matching
    time-first ``(T, Y, X)`` tracked nuclear labels.
    """
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return _centroid_markers_2d(labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected 2D labels or time-first 3D stack, got {labels.shape}")
    out = np.zeros_like(labels)
    for t in range(labels.shape[0]):
        out[t] = _centroid_markers_2d(labels[t])
    return out


def _peak_local_max_markers(basin: np.ndarray, min_distance: int) -> np.ndarray:
    from scipy.ndimage import label as nd_label
    from skimage.feature import peak_local_max

    coords = peak_local_max(basin, min_distance=max(1, min_distance), exclude_border=False)
    mask = np.zeros(basin.shape, dtype=bool)
    if coords.size:
        mask[coords[:, 0], coords[:, 1]] = True
    markers, _ = nd_label(mask)
    return markers.astype(np.int32)


def compute_hypothesis_labels(
    prob: np.ndarray,
    dp: np.ndarray | None,
    markers: np.ndarray | None,
    params: NucleusHypothesisParams,
    *,
    global_lo: float | None = None,
    global_hi: float | None = None,
) -> np.ndarray:
    """Compute a single nucleus hypothesis label image for one 2D slice.

    global_lo/global_hi: min/max of the basin computed over the full 3D volume,
    so threshold_pct is a fraction of the whole-frame dynamic range, not per-slice.
    """
    from skimage.segmentation import watershed

    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"Expected 2D probability slice, got shape {prob.shape}")

    if params.basin == "prob":
        basin = 1.0 / (1.0 + np.exp(-prob))  # logits → probabilities
    elif params.basin == "flow_mag":
        if dp is None:
            raise ValueError("flow_mag basin requested but no DP array provided")
        basin = _flow_magnitude(dp)
        if basin.ndim != 2:
            raise ValueError(f"Expected 2D flow magnitude slice, got shape {basin.shape}")
    else:
        raise ValueError(f"Unknown basin={params.basin!r}; expected 'prob' or 'flow_mag'")

    basin = _normalize_01(basin, lo=global_lo, hi=global_hi)
    if params.smooth_sigma > 0:
        basin = gaussian_filter(basin, sigma=float(params.smooth_sigma))

    if markers is None:
        markers = _peak_local_max_markers(basin, params.seed_distance)
    else:
        markers = np.asarray(markers, dtype=np.int32)
        if markers.shape != basin.shape:
            raise ValueError(
                f"Markers shape {markers.shape} does not match basin shape {basin.shape}"
            )

    from scipy.ndimage import binary_fill_holes

    threshold = float(params.threshold_pct) / 100.0
    mask = binary_fill_holes((basin >= threshold) | (markers > 0))

    labels = watershed(
        -basin,
        markers=markers,
        mask=mask,
        compactness=float(params.compactness),
        watershed_line=False,
    )
    result = _fill_and_close_labels(np.asarray(labels, dtype=_LABEL_DTYPE))
    result = _remove_small_labels(result, params.min_size)
    return _remove_low_circularity_labels(result, params.min_circularity)


def compute_cellpose_flow_hypothesis(
    prob_3d: np.ndarray,
    dp_3d: np.ndarray,
    params: CellposeFlowHypothesisParams,
) -> np.ndarray:
    """Run Cellpose native mask generation independently per z-slice.

    prob_3d: (Z, Y, X) logits from Cellpose (flows[2])
    dp_3d:   (Z, 2, Y, X) flow fields from Cellpose (flows[1])
    Returns: (Z, Y, X) uint32
    """
    try:
        import torch
        from cellpose.dynamics import compute_masks
    except ImportError as exc:
        raise ImportError(
            "cellpose and torch must be installed to use flow-based hypothesis generation"
        ) from exc

    prob_3d = np.asarray(prob_3d, dtype=np.float32)
    dp_3d = np.asarray(dp_3d, dtype=np.float32)
    if prob_3d.ndim != 3:
        raise ValueError(f"Expected (Z, Y, X) prob, got {prob_3d.shape}")
    if dp_3d.ndim != 4 or dp_3d.shape[1] != 2:
        raise ValueError(f"Expected (Z, 2, Y, X) dp, got {dp_3d.shape}")

    n_foreground = int(np.sum(prob_3d > params.cellprob_threshold))
    if n_foreground == 0:
        raise RuntimeError(
            f"No foreground pixels found: all prob values <= cellprob_threshold={params.cellprob_threshold}. "
            f"Prob range: [{float(prob_3d.min()):.2f}, {float(prob_3d.max()):.2f}]. "
            "Try lowering cellprob_threshold."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_z = prob_3d.shape[0]
    out = np.zeros_like(prob_3d, dtype=_LABEL_DTYPE)
    cp_min_size = params.min_size if params.min_size > 0 else -1
    for z in range(n_z):
        result = compute_masks(
            dp_3d[z],
            prob_3d[z],
            cellprob_threshold=params.cellprob_threshold,
            flow_threshold=params.flow_threshold,
            min_size=cp_min_size,
            niter=params.niter,
            do_3D=False,
            device=device,
        )
        # Cellpose ≥3.x returns just the mask array; older versions return (mask, p, tr).
        masks = result[0] if isinstance(result, tuple) else result
        out[z] = np.asarray(masks, dtype=_LABEL_DTYPE)
    return out


def build_consensus_boundary(
    prob_3d: np.ndarray,
    dp_3d: np.ndarray,
    cellprob_thresholds: list[float],
    gamma: float = 1.0,
    flow_threshold: float = 0.0,
    reduction: str = "mean",
    *,
    mask_callback: Callable[[np.ndarray, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce mask boundaries and occupancy over (threshold × z-slice).

    prob_3d: (Z, Y, X) logits  dp_3d: (Z, 2, Y, X)
    reduction: "mean" averages across all (threshold × z-slice) combinations;
               "max" takes the per-pixel maximum instead.
    mask_callback: optional sink called as mask_callback(masks_zyx, thresh_idx) after each threshold.
    Returns: (boundary, foreground) both (Y, X) float32.
    """
    try:
        import torch
        from cellpose.dynamics import compute_masks
        from skimage.segmentation import find_boundaries
    except ImportError as exc:
        raise ImportError("cellpose, torch, and scikit-image required") from exc

    prob_3d = apply_gamma(np.asarray(prob_3d, dtype=np.float32), gamma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_z = prob_3d.shape[0]
    accum = np.zeros(prob_3d.shape[1:], dtype=np.float32)
    foreground_accum = np.zeros(prob_3d.shape[1:], dtype=np.float32)
    n_total = 0

    for i_thresh, thresh in enumerate(cellprob_thresholds):
        z_masks: list[np.ndarray] = []
        for z in range(n_z):
            result = compute_masks(
                dp_3d[z], prob_3d[z],
                cellprob_threshold=float(thresh),
                flow_threshold=float(flow_threshold),
                niter=200,
                do_3D=False,
                device=device,
            )
            masks = result[0] if isinstance(result, tuple) else result
            masks_arr = np.asarray(masks)
            boundary_slice = find_boundaries(masks_arr, mode="inner").astype(np.float32)
            fg_slice = (masks_arr > 0).astype(np.float32)
            if reduction == "max":
                np.maximum(accum, boundary_slice, out=accum)
                np.maximum(foreground_accum, fg_slice, out=foreground_accum)
            else:
                accum += boundary_slice
                foreground_accum += fg_slice
            n_total += 1
            if mask_callback is not None:
                z_masks.append(np.asarray(masks_arr, dtype=np.uint32))
        if mask_callback is not None:
            mask_callback(np.stack(z_masks), i_thresh)

    if reduction == "max":
        return accum, foreground_accum
    boundary = accum / n_total if n_total > 0 else accum
    foreground = foreground_accum / n_total if n_total > 0 else foreground_accum
    return boundary, foreground


def build_consensus_boundary_2d(
    prob_yx: np.ndarray,
    dp_cyx: np.ndarray,
    cellprob_thresholds: list[float],
    flow_threshold: float = 0.0,
    reduction: str = "mean",
    niter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Build consensus boundary from a Z-averaged probability map and 2D flow vectors.

    prob_yx:  (Y, X) Cellpose probability logits — already Z-projected and gamma-corrected.
    dp_cyx:   (2, Y, X) flow vectors (e.g. from filtered_dp).
    Returns:  (boundary, foreground) both (Y, X) float32.
    """
    try:
        import torch
        from cellpose.dynamics import compute_masks
        from skimage.segmentation import find_boundaries
    except ImportError as exc:
        raise ImportError("cellpose, torch, and scikit-image required") from exc

    prob_yx = np.asarray(prob_yx, dtype=np.float32)
    dp_cyx = np.asarray(dp_cyx, dtype=np.float32)
    if prob_yx.ndim != 2:
        raise ValueError(f"Expected (Y, X) prob, got {prob_yx.shape}")
    if dp_cyx.ndim != 3 or dp_cyx.shape[0] != 2:
        raise ValueError(f"Expected (2, Y, X) dp, got {dp_cyx.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accum = np.zeros(prob_yx.shape, dtype=np.float32)
    foreground_accum = np.zeros(prob_yx.shape, dtype=np.float32)

    for thresh in cellprob_thresholds:
        result = compute_masks(
            dp_cyx,
            prob_yx,
            cellprob_threshold=float(thresh),
            flow_threshold=float(flow_threshold),
            niter=int(niter),
            do_3D=False,
            device=device,
        )
        masks = result[0] if isinstance(result, tuple) else result
        masks_arr = np.asarray(masks)
        boundary_slice = find_boundaries(masks_arr, mode="inner").astype(np.float32)
        fg_slice = (masks_arr > 0).astype(np.float32)
        if reduction == "max":
            np.maximum(accum, boundary_slice, out=accum)
            np.maximum(foreground_accum, fg_slice, out=foreground_accum)
        else:
            accum += boundary_slice
            foreground_accum += fg_slice

    n = len(cellprob_thresholds)
    if reduction != "max" and n > 0:
        accum /= n
        foreground_accum /= n

    return accum, foreground_accum


def compute_masks_for_threshold(
    dp_3d: np.ndarray, prob_3d: np.ndarray, threshold: float
) -> np.ndarray:
    """Run Cellpose mask generation for a specific threshold across all z-slices."""
    try:
        import torch
        from cellpose.dynamics import compute_masks
    except ImportError as exc:
        raise ImportError("cellpose and torch required") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_z = prob_3d.shape[0]
    out = np.zeros(prob_3d.shape, dtype=_LABEL_DTYPE)
    for z in range(n_z):
        result = compute_masks(
            dp_3d[z],
            prob_3d[z],
            cellprob_threshold=float(threshold),
            flow_threshold=0.0,
            niter=200,
            do_3D=False,
            device=device,
        )
        masks = result[0] if isinstance(result, tuple) else result
        out[z] = np.asarray(masks, dtype=_LABEL_DTYPE)
    return out


def compute_cellpose_foreground_masks(
    prob_tzyx: np.ndarray,
    filtered_dp_tcyx: np.ndarray,
    *,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.0,
    min_size: int = 15,
    niter: int = 200,
    progress_cb: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Generate binary cell foreground masks with Cellpose dynamics.

    prob_tzyx is Cellpose probability logits shaped (T, Z, Y, X), or a single
    volume shaped (Z, Y, X). filtered_dp_tcyx must be the filtered flow stack
    produced by the cell workflow, shaped (T, 2, Y, X).
    """
    prob = np.asarray(prob_tzyx, dtype=np.float32)
    if prob.ndim == 3:
        prob = prob[np.newaxis]
    if prob.ndim != 4:
        raise ValueError(
            f"Expected probability shape (T, Z, Y, X) or (Z, Y, X), got {prob.shape}"
        )

    filtered_dp = np.asarray(filtered_dp_tcyx, dtype=np.float32)
    if filtered_dp.ndim != 4 or filtered_dp.shape[1] != 2:
        raise ValueError(
            f"Expected filtered flow shape (T, 2, Y, X), got {filtered_dp.shape}"
        )
    if prob.shape[0] != filtered_dp.shape[0] or prob.shape[2:] != filtered_dp.shape[2:]:
        raise ValueError(
            "Cellpose probability and filtered flow shapes do not match: "
            f"probability {prob.shape}, filtered flow {filtered_dp.shape}"
        )

    try:
        import torch
        from cellpose.dynamics import compute_masks
    except ImportError as exc:
        raise ImportError(
            "cellpose and torch must be installed to generate Cellpose foreground masks"
        ) from exc

    prob_tyx = prob.mean(axis=1).astype(np.float32, copy=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = np.zeros(prob_tyx.shape, dtype=np.uint8)

    for t in range(prob_tyx.shape[0]):
        result = compute_masks(
            filtered_dp[t],
            prob_tyx[t],
            cellprob_threshold=float(cellprob_threshold),
            flow_threshold=float(flow_threshold),
            min_size=int(min_size),
            niter=int(niter),
            do_3D=False,
            device=device,
        )
        masks = result[0] if isinstance(result, tuple) else result
        out[t] = (np.asarray(masks) > 0).astype(np.uint8)
        if progress_cb is not None:
            progress_cb(t + 1, prob_tyx.shape[0])

    return out


@dataclass(frozen=True, slots=True)
class SeededWatershedParams:
    """Parameters for nucleus-seeded watershed cell hypothesis generation."""

    basin: str = "prob"
    foreground_threshold: float = 0.5
    compactness: float = 0.0

    def __post_init__(self) -> None:
        warnings.warn(
            "SeededWatershedParams is deprecated and will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

    def to_dict(self) -> dict[str, object]:
        return {"method": "seeded_watershed", **asdict(self)}


def compute_seeded_watershed(
    prob_2d: np.ndarray,
    dp_2d: np.ndarray | None,
    seeds_2d: np.ndarray,
    params: SeededWatershedParams,
) -> np.ndarray:
    """Seeded watershed using nucleus labels as markers for one 2D z-slice.

    Foreground mask is always derived from sigmoid(prob_2d). Seeds whose
    centroid falls outside the mask are silently dropped by the watershed.
    """
    warnings.warn(
        "compute_seeded_watershed is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    from scipy.ndimage import binary_fill_holes
    from skimage.segmentation import watershed

    prob_2d = np.asarray(prob_2d, dtype=np.float32)
    seeds_2d = np.asarray(seeds_2d, dtype=np.int32)

    sigmoid_prob = 1.0 / (1.0 + np.exp(-prob_2d))
    fg_mask = binary_fill_holes(sigmoid_prob > params.foreground_threshold)

    if params.basin == "prob":
        basin = sigmoid_prob
    elif params.basin == "flow_mag":
        if dp_2d is None:
            raise ValueError("flow_mag basin requires a dp array")
        basin = _flow_magnitude(dp_2d)
        if basin.ndim != 2:
            raise ValueError(f"Expected 2D flow magnitude slice, got shape {basin.shape}")
    else:
        raise ValueError(f"Unknown basin={params.basin!r}; expected 'prob' or 'flow_mag'")

    labels = watershed(
        -basin,
        markers=seeds_2d,
        mask=fg_mask,
        compactness=float(params.compactness),
        watershed_line=False,
    )
    return np.asarray(labels, dtype=_LABEL_DTYPE)


def compute_contour_watershed(
    boundary: np.ndarray,
    foreground_mask: np.ndarray,
    params: ContourWatershedParams,
) -> np.ndarray:
    """Run seeded watershed on a consensus boundary image and binary foreground mask.

    Seeds are placed at EDT maxima of fg_mask & (boundary < ridge_threshold),
    so contour ridges separating touching cells drive seed placement rather than
    foreground intensity peaks.

    boundary:   (Y, X) float32 — high at cell borders
    foreground_mask: (Y, X) binary — nonzero pixels are allowed segmentation area
    Returns:    (Y, X) uint32 label image
    """
    from scipy.ndimage import label as nd_label
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    boundary = np.asarray(boundary, dtype=np.float32)
    foreground_mask = np.asarray(foreground_mask)
    if foreground_mask.shape != boundary.shape:
        raise ValueError(
            f"Foreground mask shape {foreground_mask.shape} does not match boundary shape {boundary.shape}"
        )
    fg_mask = foreground_mask > 0

    boundary_pre = np.asarray(boundary, dtype=np.float32).copy()

    # Apply correlated noise perturbation
    if params.noise_scale > 0:
        noise = np.random.normal(0, params.noise_scale, boundary_pre.shape)
        if params.noise_blur_sigma > 0:
            noise = gaussian_filter(noise, sigma=params.noise_blur_sigma)
        boundary_pre = np.clip(boundary_pre + noise, 0, 1)

    boundary_pre[boundary_pre < params.foreground_threshold] = 0

    from scipy.ndimage import distance_transform_edt

    # Carve strong contour ridges out of the mask so touching cells become
    # separate connected components before seeding.
    core = fg_mask & (boundary_pre < params.ridge_threshold)
    edt = distance_transform_edt(core)

    coords = peak_local_max(
        edt,
        min_distance=max(1, int(params.seed_distance)),
        threshold_abs=1.0,
        exclude_border=False,
    )
    marker_mask = np.zeros(boundary_pre.shape, dtype=bool)
    if coords.size:
        marker_mask[coords[:, 0], coords[:, 1]] = True
    markers, _ = nd_label(marker_mask)

    # Watershed floods fg_mask (not core) so basins fill back over carved ridges.
    labels = watershed(boundary_pre, markers=markers, mask=fg_mask, watershed_line=False)
    result = _fill_and_close_labels(np.asarray(labels, dtype=_LABEL_DTYPE))
    result = _remove_small_labels(result, params.min_size)
    return _remove_low_circularity_labels(result, params.min_circularity)
"""ICM + geodesic-unary cell labelling pipeline.

Extracted from ``scripts/experiment_cell_2d_t_multilabel_graphcut.py`` and
packaged as a reusable API.  Only the ICM solver with geodesic unaries and
unary-argmin initialisation is included; alpha-expansion, all other unary
modes, and the lambda-area cost term are dropped.

Public surface
--------------
``CellLabelICMParams`` — frozen dataclass with solver hyper-parameters.
``segment_cells_icm`` — run the pipeline on in-memory arrays.
``run_cell_icm_from_pos_dir`` — load standard TIFFs from a position directory
                                and delegates to ``segment_cells_icm``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numba
import numpy as np
import tifffile
from skimage.graph import MCP_Geometric

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INF: float = 1e9
_DIAG_SCALE: float = float(1.0 / np.sqrt(2.0))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _should_stop_after_round(total_flips: int, min_flips: int) -> bool:
    return int(min_flips) > 0 and int(total_flips) < int(min_flips)


def _compute_pairwise_weights(
    fg_mask: np.ndarray,
    contours: np.ndarray,
    lambda_s: float,
    beta_s: float,
    lambda_t: float,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Pre-compute (T,Y,X) n-link weight arrays for spatial and temporal edges.

    Boundary signal is always the raw contour map.  Diagonal edges (dr, dl) and
    spatiotemporal face-diagonals are scaled by 1/√2 relative to axis-aligned
    edges so the total boundary penalty is approximately isotropic.

    Returns (h, v, dr, dl, tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l).
    """
    T, Y, X = fg_mask.shape
    fg = fg_mask.astype(bool)
    c = contours

    h = np.zeros((T, Y, X), dtype=np.float32)
    h[:, :, :-1] = (
        lambda_s * np.exp(-beta_s * 0.5 * (c[:, :, :-1] + c[:, :, 1:]))
        * (fg[:, :, :-1] & fg[:, :, 1:])
    ).astype(np.float32)

    v = np.zeros((T, Y, X), dtype=np.float32)
    v[:, :-1, :] = (
        lambda_s * np.exp(-beta_s * 0.5 * (c[:, :-1, :] + c[:, 1:, :]))
        * (fg[:, :-1, :] & fg[:, 1:, :])
    ).astype(np.float32)

    dr = np.zeros((T, Y, X), dtype=np.float32)
    dr[:, :-1, :-1] = (
        _DIAG_SCALE * lambda_s
        * np.exp(-beta_s * 0.5 * (c[:, :-1, :-1] + c[:, 1:, 1:]))
        * (fg[:, :-1, :-1] & fg[:, 1:, 1:])
    ).astype(np.float32)

    dl = np.zeros((T, Y, X), dtype=np.float32)
    dl[:, :-1, 1:] = (
        _DIAG_SCALE * lambda_s
        * np.exp(-beta_s * 0.5 * (c[:, :-1, 1:] + c[:, 1:, :-1]))
        * (fg[:, :-1, 1:] & fg[:, 1:, :-1])
    ).astype(np.float32)

    tw = np.zeros((T, Y, X), dtype=np.float32)
    if T > 1:
        tw[:-1, :, :] = (lambda_t * (fg[:-1] & fg[1:])).astype(np.float32)

    # Spatiotemporal face-diagonal edges: connect (t,y,x) to (t+1,y±1,x)
    # and (t+1,y,x±1).  Scaled by 1/√2.
    tw_ty_dn = np.zeros((T, Y, X), dtype=np.float32)
    tw_ty_up = np.zeros((T, Y, X), dtype=np.float32)
    tw_tx_r  = np.zeros((T, Y, X), dtype=np.float32)
    tw_tx_l  = np.zeros((T, Y, X), dtype=np.float32)
    if T > 1:
        tw_ty_dn[:-1, :-1, :] = (
            _DIAG_SCALE * lambda_t * (fg[:-1, :-1, :] & fg[1:, 1:, :])
        ).astype(np.float32)
        tw_ty_up[:-1, 1:, :] = (
            _DIAG_SCALE * lambda_t * (fg[:-1, 1:, :] & fg[1:, :-1, :])
        ).astype(np.float32)
        tw_tx_r [:-1, :, :-1] = (
            _DIAG_SCALE * lambda_t * (fg[:-1, :, :-1] & fg[1:, :, 1:])
        ).astype(np.float32)
        tw_tx_l [:-1, :,  1:] = (
            _DIAG_SCALE * lambda_t * (fg[:-1, :,  1:] & fg[1:, :, :-1])
        ).astype(np.float32)

    return h, v, dr, dl, tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l


# ---------------------------------------------------------------------------
# Geodesic unary computation
# ---------------------------------------------------------------------------

def _compute_geodesic_unaries(
    nuc_tracks: np.ndarray,
    fg_mask: np.ndarray,
    contours: np.ndarray,
    label_ids: np.ndarray,
    alpha_unary: float,
) -> dict[tuple[int, int], np.ndarray]:
    """Compute normalized geodesic unary costs for each alive (frame, label) pair.

    Cost field per pixel: 1 + alpha_unary * contour.
    The MCP shortest-path naturally bends around high-cost (high-contour) regions.
    """
    T, Y, X = fg_mask.shape
    unary: dict[tuple[int, int], np.ndarray] = {}

    for t in range(T):
        fg_t = fg_mask[t]
        cost_field = np.full((Y, X), np.inf, dtype=np.float32)
        cost_field[fg_t] = 1.0 + alpha_unary * contours[t][fg_t]

        alive = [int(k) for k in label_ids if np.any(nuc_tracks[t] == k)]
        if not alive:
            continue

        raw: dict[int, np.ndarray] = {}
        for k in alive:
            starts = [
                tuple(int(v) for v in c)
                for c in np.argwhere(nuc_tracks[t] == k)
            ]
            mcp = MCP_Geometric(cost_field, fully_connected=True)
            cum, _ = mcp.find_costs(starts)
            d = cum.astype(np.float32)
            d[~fg_t] = np.inf
            raw[k] = d

        # Normalize per frame
        all_finite = np.concatenate(
            [d[np.isfinite(d)] for d in raw.values()]
        )
        med = float(np.median(all_finite)) if all_finite.size > 0 else 1.0
        if med <= 0.0:
            med = 1.0

        for k, d in raw.items():
            nd = d / med
            nd[~np.isfinite(nd)] = _INF
            unary[(t, k)] = nd

        # Hard nucleus anchors: set INF for wrong labels at nucleus pixels
        for k in alive:
            k_pix = nuc_tracks[t] == k
            if not k_pix.any():
                continue
            for j in alive:
                if j != k and (t, j) in unary:
                    unary[(t, j)][k_pix] = _INF

        if (t + 1) % 10 == 0 or t + 1 == T:
            print(
                f"  geodesic unaries: frame {t + 1}/{T}, {len(alive)} alive",
                flush=True,
            )

    return unary


# ---------------------------------------------------------------------------
# Nucleus anchors and unary densification
# ---------------------------------------------------------------------------

def _apply_nucleus_anchors(
    unary: dict[tuple[int, int], np.ndarray],
    nuc_tracks: np.ndarray,
    label_ids: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Apply hard nucleus anchors to a unary dict (mutates and returns it).

    For each frame t and each track k alive at t:
      * ``unary[(t, k)]`` at nucleus pixels of k is forced to 0.
      * ``unary[(t, j)]`` at nucleus pixels of k is forced to INF for all j≠k.
    """
    T = nuc_tracks.shape[0]
    label_ids = [int(k) for k in label_ids]
    cached_labels_by_t: dict[int, list[int]] = {}
    for (t, j) in unary:
        cached_labels_by_t.setdefault(int(t), []).append(int(j))

    for t in range(T):
        alive = [k for k in label_ids if int((nuc_tracks[t] == k).sum()) > 0]
        cached = cached_labels_by_t.get(t, [])
        for k in alive:
            k_pix = nuc_tracks[t] == k
            if (t, k) in unary:
                unary[(t, k)][k_pix] = 0.0
            for j in cached:
                if j != k:
                    unary[(t, j)][k_pix] = _INF
    return unary


def _dict_to_dense_unary(
    unary: dict[tuple[int, int], np.ndarray],
    fg_mask: np.ndarray,
    label_ids: np.ndarray,
) -> np.ndarray:
    """Convert sparse unary dict to dense (T, Y, X, K) float32 array for ICM."""
    T, Y, X = fg_mask.shape
    K = len(label_ids)
    dense = np.full((T, Y, X, K), _INF, dtype=np.float32)
    for ki, k in enumerate(label_ids):
        for t in range(T):
            u = unary.get((t, int(k)))
            if u is not None:
                dense[t, :, :, ki] = u
    return dense


# ---------------------------------------------------------------------------
# Numba ICM kernel
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _nb_icm_round(
    labels: np.ndarray,           # (T, Y, X) uint32 — in-place read + write
    unary_dense: np.ndarray,      # (T, Y, X, K) float32
    h_w: np.ndarray,              # (T, Y, X) float32
    v_w: np.ndarray,              # (T, Y, X) float32
    dr_w: np.ndarray,             # (T, Y, X) float32 — down-right diagonal
    dl_w: np.ndarray,             # (T, Y, X) float32 — down-left diagonal
    tw_w: np.ndarray,             # (T, Y, X) float32 — pure temporal
    tw_ty_dn_w: np.ndarray,       # (T, Y, X) float32 — (t,y,x)↔(t+1,y+1,x)
    tw_ty_up_w: np.ndarray,       # (T, Y, X) float32 — (t,y,x)↔(t+1,y-1,x)
    tw_tx_r_w: np.ndarray,        # (T, Y, X) float32 — (t,y,x)↔(t+1,y,x+1)
    tw_tx_l_w: np.ndarray,        # (T, Y, X) float32 — (t,y,x)↔(t+1,y,x-1)
    fg_mask: np.ndarray,          # (T, Y, X) bool
    label_ids: np.ndarray,        # (K,) uint32
    anchor_label: np.ndarray,     # (T, Y, X) uint32 — 0=free, k=pinned to k
    areas: np.ndarray,            # (K, T) int32 — per-label pixel counts, updated in-place
    lambda_area: np.float32,      # weight for frame-to-frame area change penalty
) -> int:
    """One sequential (Gauss-Seidel) ICM sweep — raster scan, in-place.

    Each pixel immediately sees its neighbors' latest decisions, so changes
    propagate locally rather than as coordinated mass swaps.  Guaranteed to
    converge for non-negative Potts pairwise weights.
    Returns number of label changes.
    """
    T, Y, X = fg_mask.shape
    K = len(label_ids)
    n_flips = 0
    for t in range(T):
        for y in range(Y):
            for x in range(X):
                if anchor_label[t, y, x] > np.uint32(0):
                    continue
                if not fg_mask[t, y, x]:
                    continue
                old_label = labels[t, y, x]

                # Find index of old_label in label_ids for area delta computation.
                ji = np.int32(-1)
                for kk in range(K):
                    if label_ids[kk] == old_label:
                        ji = np.int32(kk)
                        break

                best_cost = np.float32(1e30)
                best_k = old_label
                best_ki = np.int32(-1)
                for ki in range(K):
                    u = unary_dense[t, y, x, ki]
                    if u >= np.float32(1e8):
                        continue
                    k = label_ids[ki]
                    # 8-connected lateral-adjacency gate (within frame): a
                    # pixel may flip to label k only if it currently is k or
                    # has an in-frame 8-neighbor already carrying k.
                    if k != old_label:
                        adj = False
                        if x + 1 < X and labels[t, y, x + 1] == k:
                            adj = True
                        elif x > 0 and labels[t, y, x - 1] == k:
                            adj = True
                        elif y + 1 < Y and labels[t, y + 1, x] == k:
                            adj = True
                        elif y > 0 and labels[t, y - 1, x] == k:
                            adj = True
                        elif (y + 1 < Y and x + 1 < X
                              and labels[t, y + 1, x + 1] == k):
                            adj = True
                        elif (y > 0 and x > 0
                              and labels[t, y - 1, x - 1] == k):
                            adj = True
                        elif (y + 1 < Y and x > 0
                              and labels[t, y + 1, x - 1] == k):
                            adj = True
                        elif (y > 0 and x + 1 < X
                              and labels[t, y - 1, x + 1] == k):
                            adj = True
                        if not adj:
                            continue
                    cost = u
                    # Axis-aligned
                    if (x + 1 < X and fg_mask[t, y, x + 1]
                            and labels[t, y, x + 1] != k):
                        cost += h_w[t, y, x]
                    if (x > 0 and fg_mask[t, y, x - 1]
                            and labels[t, y, x - 1] != k):
                        cost += h_w[t, y, x - 1]
                    if (y + 1 < Y and fg_mask[t, y + 1, x]
                            and labels[t, y + 1, x] != k):
                        cost += v_w[t, y, x]
                    if (y > 0 and fg_mask[t, y - 1, x]
                            and labels[t, y - 1, x] != k):
                        cost += v_w[t, y - 1, x]
                    # Diagonals: dr[t,y,x] is the (y,x)→(y+1,x+1) edge weight
                    if (y + 1 < Y and x + 1 < X and fg_mask[t, y + 1, x + 1]
                            and labels[t, y + 1, x + 1] != k):
                        cost += dr_w[t, y, x]
                    if (y > 0 and x > 0 and fg_mask[t, y - 1, x - 1]
                            and labels[t, y - 1, x - 1] != k):
                        cost += dr_w[t, y - 1, x - 1]
                    # dl[t,y,x] is the (y,x)→(y+1,x-1) edge weight
                    if (y + 1 < Y and x > 0 and fg_mask[t, y + 1, x - 1]
                            and labels[t, y + 1, x - 1] != k):
                        cost += dl_w[t, y, x]
                    if (y > 0 and x + 1 < X and fg_mask[t, y - 1, x + 1]
                            and labels[t, y - 1, x + 1] != k):
                        cost += dl_w[t, y - 1, x + 1]
                    # Temporal (same pixel, adjacent frame)
                    if (t + 1 < T and fg_mask[t + 1, y, x]
                            and labels[t + 1, y, x] != k):
                        cost += tw_w[t, y, x]
                    if (t > 0 and fg_mask[t - 1, y, x]
                            and labels[t - 1, y, x] != k):
                        cost += tw_w[t - 1, y, x]
                    # Spatiotemporal face-diagonals (forward: to t+1)
                    if t + 1 < T:
                        if (y + 1 < Y and fg_mask[t + 1, y + 1, x]
                                and labels[t + 1, y + 1, x] != k):
                            cost += tw_ty_dn_w[t, y, x]
                        if (y > 0 and fg_mask[t + 1, y - 1, x]
                                and labels[t + 1, y - 1, x] != k):
                            cost += tw_ty_up_w[t, y, x]
                        if (x + 1 < X and fg_mask[t + 1, y, x + 1]
                                and labels[t + 1, y, x + 1] != k):
                            cost += tw_tx_r_w[t, y, x]
                        if (x > 0 and fg_mask[t + 1, y, x - 1]
                                and labels[t + 1, y, x - 1] != k):
                            cost += tw_tx_l_w[t, y, x]
                    # Spatiotemporal face-diagonals (reverse: from t-1)
                    if t > 0:
                        if (y + 1 < Y and fg_mask[t - 1, y + 1, x]
                                and labels[t - 1, y + 1, x] != k):
                            cost += tw_ty_up_w[t - 1, y + 1, x]
                        if (y > 0 and fg_mask[t - 1, y - 1, x]
                                and labels[t - 1, y - 1, x] != k):
                            cost += tw_ty_dn_w[t - 1, y - 1, x]
                        if (x + 1 < X and fg_mask[t - 1, y, x + 1]
                                and labels[t - 1, y, x + 1] != k):
                            cost += tw_tx_l_w[t - 1, y, x + 1]
                        if (x > 0 and fg_mask[t - 1, y, x - 1]
                                and labels[t - 1, y, x - 1] != k):
                            cost += tw_tx_r_w[t - 1, y, x - 1]
                    # Area change penalty: quadratic delta of
                    # sum_t (A_k(t) - A_k(t±1))^2.
                    # Gaining k at t: (ak+1-ref)^2 - (ak-ref)^2 = 2*(ak-ref)+1
                    # Losing  j at t: (aj-1-ref)^2 - (aj-ref)^2 = 1-2*(aj-ref)
                    if lambda_area > np.float32(0.0) and ki != ji:
                        ak = areas[ki, t]
                        aj = areas[ji, t] if ji >= np.int32(0) else np.int32(0)
                        area_delta = np.float32(0.0)
                        if t > 0:
                            area_delta += np.float32(
                                2 * (ak - areas[ki, t - 1]) + 1
                            )
                            if ji >= np.int32(0):
                                area_delta += np.float32(
                                    1 - 2 * (aj - areas[ji, t - 1])
                                )
                        if t + 1 < T:
                            area_delta += np.float32(
                                2 * (ak - areas[ki, t + 1]) + 1
                            )
                            if ji >= np.int32(0):
                                area_delta += np.float32(
                                    1 - 2 * (aj - areas[ji, t + 1])
                                )
                        cost += lambda_area * area_delta
                    if cost < best_cost:
                        best_cost = cost
                        best_k = k
                        best_ki = np.int32(ki)
                labels[t, y, x] = best_k
                if best_k != old_label:
                    n_flips += 1
                    if best_ki >= np.int32(0):
                        areas[best_ki, t] += np.int32(1)
                    if ji >= np.int32(0):
                        areas[ji, t] -= np.int32(1)
    return n_flips


# ---------------------------------------------------------------------------
# ICM runner (simplified: lambda_area=0, always unary-argmin init)
# ---------------------------------------------------------------------------

def _run_icm(
    fg_mask: np.ndarray,
    unary_dense: np.ndarray,   # (T, Y, X, K) float32
    label_ids: np.ndarray,
    h: np.ndarray,
    v: np.ndarray,
    dr: np.ndarray,
    dl: np.ndarray,
    tw: np.ndarray,
    tw_ty_dn: np.ndarray,
    tw_ty_up: np.ndarray,
    tw_tx_r: np.ndarray,
    tw_tx_l: np.ndarray,
    nuc_tracks: np.ndarray,
    n_iters: int = 10,
    min_round_flips: int = 0,
) -> tuple[np.ndarray, list[dict]]:
    """Run ICM with geodesic unaries, unary-argmin init, and no area penalty.

    ``lambda_area`` is hardwired to 0.0.  Init is always the per-pixel argmin
    of the dense unary cost volume.  Returns ``(labels, energy_log)``.
    """
    T, Y, X = fg_mask.shape

    print("Initializing labels from unary argmin...", flush=True)
    best_ki = np.argmin(unary_dense, axis=3)
    labels = np.where(fg_mask, label_ids[best_ki], 0).astype(np.uint32)

    # Nucleus pixels are anchored — enforce at init.
    nuc_mask = nuc_tracks > 0
    labels[nuc_mask] = nuc_tracks[nuc_mask].astype(np.uint32)

    anchor_label = nuc_tracks.astype(np.uint32)
    h32 = h.astype(np.float32)
    v32 = v.astype(np.float32)
    dr32 = dr.astype(np.float32)
    dl32 = dl.astype(np.float32)
    tw32 = tw.astype(np.float32)
    tw_ty_dn32 = tw_ty_dn.astype(np.float32)
    tw_ty_up32 = tw_ty_up.astype(np.float32)
    tw_tx_r32  = tw_tx_r.astype(np.float32)
    tw_tx_l32  = tw_tx_l.astype(np.float32)
    lids32 = label_ids.astype(np.uint32)
    lambda_area32 = np.float32(0.0)

    # Per-label pixel counts per frame — updated in-place by _nb_icm_round.
    K = len(label_ids)
    areas = np.zeros((K, T), dtype=np.int32)
    for ki, k in enumerate(label_ids):
        for t in range(T):
            areas[ki, t] = int(np.count_nonzero(labels[t] == k))

    energy_log: list[dict] = []
    for iteration in range(n_iters):
        print(
            f"\n=== ICM Round {iteration + 1}/{n_iters} ===", flush=True,
        )
        t0 = perf_counter()
        n_flips = _nb_icm_round(
            labels, unary_dense, h32, v32, dr32, dl32, tw32,
            tw_ty_dn32, tw_ty_up32, tw_tx_r32, tw_tx_l32,
            fg_mask, lids32, anchor_label, areas, lambda_area32,
        )
        elapsed = perf_counter() - t0
        print(f"  {elapsed:.1f}s  flips={n_flips}", flush=True)
        energy_log.append({
            "iteration": iteration + 1,
            "flips": int(n_flips),
        })
        if n_flips == 0:
            print(
                f"  Converged after round {iteration + 1}.", flush=True,
            )
            break
        if _should_stop_after_round(n_flips, min_round_flips):
            print(
                f"  Stopping after round {iteration + 1}: "
                f"flips={n_flips} < min_round_flips={min_round_flips}.",
                flush=True,
            )
            break

    return labels, energy_log


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CellLabelICMParams:
    """Hyper-parameters for the ICM + geodesic-unary cell labelling pipeline.

    Attributes
    ----------
    alpha_unary:
        Contour weight in the geodesic cost field
        ``1 + alpha_unary * contour``.
    lambda_s:
        Spatial pairwise Potts weight.
    beta_s:
        Contour-sensitivity exponent in spatial pairwise:
        ``lambda_s * exp(-beta_s * avg_contour)``.
    lambda_t:
        Temporal pairwise Potts weight.
    n_iters:
        Maximum number of ICM rounds.
    min_round_flips:
        Early-stop threshold: stop if a round flips fewer than this many pixels
        (0 = disabled).
    """

    alpha_unary: float = 200.0
    lambda_s: float = 1.0
    beta_s: float = 5.0
    lambda_t: float = 0.1
    n_iters: int = 25
    min_round_flips: int = 0


def segment_cells_icm(
    nuc_tracks: np.ndarray,
    fg_mask: np.ndarray,
    contours: np.ndarray,
    params: CellLabelICMParams,
) -> np.ndarray:
    """Run the ICM + geodesic-unary cell labelling pipeline.

    Parameters
    ----------
    nuc_tracks:
        (T, Y, X) uint32 nucleus tracked labels.  Non-zero values are track
        IDs; zero is background.
    fg_mask:
        (T, Y, X) bool foreground mask.  Only True voxels take part in the
        graph and receive a cell label.
    contours:
        (T, Y, X) float32 contour map.  Values should be in [0, 1]; higher
        values indicate stronger cell-boundary evidence.
    params:
        Solver hyper-parameters.

    Returns
    -------
    (T, Y, X) uint32
        Predicted cell labels.  Zero where ``fg_mask`` is False.
        Every foreground pixel is assigned a label.

    Raises
    ------
    ValueError
        If the three input arrays do not share the same ``(T, Y, X)`` shape.
    """
    # ── Validate shapes ─────────────────────────────────────────────────
    shape = nuc_tracks.shape
    if fg_mask.shape != shape:
        raise ValueError(
            f"Shape mismatch: nuc_tracks {shape} vs fg_mask {fg_mask.shape}"
        )
    if contours.shape != shape:
        raise ValueError(
            f"Shape mismatch: nuc_tracks {shape} vs contours {contours.shape}"
        )

    T, Y, X = shape

    # ── Global label set ─────────────────────────────────────────────────
    label_ids = np.array(
        sorted(int(k) for k in np.unique(nuc_tracks) if k > 0),
        dtype=np.uint32,
    )

    # ── Pairwise weights ─────────────────────────────────────────────────
    print("Computing pairwise weights...", flush=True)
    h, v, dr, dl, tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l = (
        _compute_pairwise_weights(
            fg_mask, contours,
            params.lambda_s, params.beta_s, params.lambda_t,
        )
    )

    # ── Geodesic unaries ─────────────────────────────────────────────────
    print("Computing geodesic unaries...", flush=True)
    unary = _compute_geodesic_unaries(
        nuc_tracks, fg_mask, contours, label_ids, params.alpha_unary,
    )

    # ── Nucleus anchors ──────────────────────────────────────────────────
    _apply_nucleus_anchors(unary, nuc_tracks, label_ids)

    # ── Densify ──────────────────────────────────────────────────────────
    print("Converting unary to dense array...", flush=True)
    unary_dense = _dict_to_dense_unary(unary, fg_mask, label_ids)

    # ── ICM ──────────────────────────────────────────────────────────────
    print("Running ICM...", flush=True)
    labels, _energy_log = _run_icm(
        fg_mask, unary_dense, label_ids,
        h, v, dr, dl, tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l,
        nuc_tracks,
        n_iters=params.n_iters,
        min_round_flips=params.min_round_flips,
    )

    return labels


def run_cell_icm_from_pos_dir(
    pos_dir: Path,
    params: CellLabelICMParams,
    *,
    crop: tuple[int, int, int, int, int, int] | None = None,
) -> np.ndarray:
    """Load standard TIFFs from a position directory and run ``segment_cells_icm``.

    Expected files (raises ``FileNotFoundError`` if missing):

    - ``2_nucleus/tracked_labels.tif`` → ``nuc_tracks`` (uint32)
    - ``3_cell/foreground_masks.tif`` → ``fg_mask`` (uint8 → bool)
    - ``3_cell/contour_maps.tif`` → ``contours`` (float32)

    Nucleus pixels are unioned into the foreground mask (they are always part
    of the cell graph).  An optional ``crop`` ``(T0, T1, Y0, Y1, X0, X1)``
    is applied after loading.

    Parameters
    ----------
    pos_dir:
        Position directory containing the ``2_nucleus/`` and ``3_cell/``
        sub-directories.
    params:
        Solver hyper-parameters forwarded to ``segment_cells_icm``.
    crop:
        Optional ``(T0, T1, Y0, Y1, X0, X1)`` slice.

    Returns
    -------
    (T, Y, X) uint32
        Predicted cell labels.
    """
    pos_dir = Path(pos_dir)

    def _read(path: Path, dtype) -> np.ndarray:
        full = pos_dir / path
        if not full.exists():
            raise FileNotFoundError(str(full))
        a = np.asarray(tifffile.imread(full), dtype=dtype)
        if a.ndim == 4 and a.shape[1] == 1:
            a = a[:, 0]
        return a

    nuc_tracks = _read("2_nucleus/tracked_labels.tif", np.uint32)
    fg_mask = _read("3_cell/foreground_masks.tif", np.uint8) > 0
    contours = _read("3_cell/contour_maps.tif", np.float32)

    if crop is not None:
        t0, t1, y0, y1, x0, x1 = crop
        s = (slice(t0, t1), slice(y0, y1), slice(x0, x1))
        nuc_tracks = nuc_tracks[s]
        fg_mask = fg_mask[s]
        contours = contours[s]

    # Nucleus pixels are always part of the cell foreground.
    fg_mask = fg_mask | (nuc_tracks > 0)

    return segment_cells_icm(nuc_tracks, fg_mask, contours, params)
"""Filtering helpers for nucleus contour-map stacks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


@dataclass(frozen=True, slots=True)
class ContourFilterParams:
    """Parameters for spatial and temporal contour-map filtering."""

    median_kernel_time: int = 1
    median_kernel_space: int = 1
    gaussian_sigma_time: float = 0.0
    gaussian_sigma_space: float = 0.0


def _normalize_contour_stack(contours: np.ndarray) -> tuple[np.ndarray, str]:
    arr = np.asarray(contours, dtype=np.float32)
    if arr.ndim == 2:
        return arr[np.newaxis], "yx"
    if arr.ndim == 3:
        return arr, "tyx"
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0], "tcyx"
    raise ValueError(f"Unsupported contour maps shape {arr.shape}")


def _restore_contour_stack(contours_tyx: np.ndarray, layout: str) -> np.ndarray:
    if layout == "yx":
        return contours_tyx[0]
    if layout == "tcyx":
        return contours_tyx[:, np.newaxis]
    return contours_tyx


def compute_filtered_contour_maps(
    contours: np.ndarray,
    params: ContourFilterParams,
) -> np.ndarray:
    """Return contour maps after median and Gaussian filtering."""
    filtered, layout = _normalize_contour_stack(contours)
    if params.median_kernel_time > 1 or params.median_kernel_space > 1:
        filtered = median_filter(
            filtered,
            size=(
                int(params.median_kernel_time),
                int(params.median_kernel_space),
                int(params.median_kernel_space),
            ),
        )
    if params.gaussian_sigma_time > 0.0 or params.gaussian_sigma_space > 0.0:
        filtered = gaussian_filter(
            filtered,
            sigma=(
                float(params.gaussian_sigma_time),
                float(params.gaussian_sigma_space),
                float(params.gaussian_sigma_space),
            ),
        )
    return _restore_contour_stack(
        np.asarray(filtered, dtype=np.float32),
        layout,
    )
"""Flow-following cell segmentation: per-frame Euler integration of the
Cellpose flow field with an EDT-direction gravity blend toward tracked nuclei."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numba
import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    median_filter,
)


@dataclass(frozen=True, slots=True)
class FlowFollowingParams:
    """Parameters for `compute_flow_following_movie`."""

    median_kernel_time: int = 3
    median_kernel_space: int = 5
    gaussian_sigma_time: float = 0.0
    gaussian_sigma_space: float = 0.0
    flow_weight: float = 0.5
    flow_step_scale: float = 0.2
    max_iterations: int = 100
    capture_radius: float = 3.0


def compute_filtered_flow_vectors(
    dp_tcyx: np.ndarray,
    params: FlowFollowingParams,
) -> np.ndarray:
    """Return flow vectors after the configured median and Gaussian filters."""
    filtered = np.asarray(dp_tcyx, dtype=np.float32)
    if params.median_kernel_time > 1 or params.median_kernel_space > 1:
        filtered = median_filter(
            filtered,
            size=(
                1,
                int(params.median_kernel_time),
                int(params.median_kernel_space),
                int(params.median_kernel_space),
            ),
        )
    if params.gaussian_sigma_time > 0.0 or params.gaussian_sigma_space > 0.0:
        filtered = gaussian_filter(
            filtered,
            sigma=(
                0.0,
                float(params.gaussian_sigma_time),
                float(params.gaussian_sigma_space),
                float(params.gaussian_sigma_space),
            ),
        )
    return np.asarray(filtered, dtype=np.float32)


@numba.njit(parallel=True, cache=True)
def _flow_integrate(
    nuclear_labels: np.ndarray,    # (H, W) int32
    flow: np.ndarray,              # (H, W, 2) float32 — channel 0 = dY, channel 1 = dX
    grav_y: np.ndarray,            # (H, W) float32 — EDT-direction unit vector y
    grav_x: np.ndarray,            # (H, W) float32 — EDT-direction unit vector x
    dist_to_nucleus: np.ndarray,   # (H, W) float32 — EDT distance to nearest nuclear pixel
    nearest_y: np.ndarray,         # (H, W) int32 — y-index of nearest nuclear pixel
    nearest_x: np.ndarray,         # (H, W) int32
    prob_mask: np.ndarray,         # (H, W) bool — foreground mask
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


def compute_flow_following_movie(
    foreground_tyx: np.ndarray,    # (T, Y, X) bool
    dp_tcyx: np.ndarray,           # (T, 2, Y, X) float32
    labels_tyx: np.ndarray,        # (T, Y, X) int32
    params: FlowFollowingParams,
    progress_cb: Callable[[int, int], None] | None = None,
    *,
    filter_vectors: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame flow-following segmentation with pre-integration filtering.

    Returns
    -------
    filtered_dp_tcyx : (T, 2, Y, X) float32 — flow stack after median+Gaussian.
    cell_labels_tyx  : (T, Y, X) int32      — same labelling as input nuclei.
    """
    foreground = np.asarray(foreground_tyx, dtype=bool)
    dp = np.asarray(dp_tcyx, dtype=np.float32)
    labels = np.asarray(labels_tyx, dtype=np.int32)

    T = dp.shape[0]

    filtered = compute_filtered_flow_vectors(dp, params) if filter_vectors else dp

    out_labels = np.zeros_like(labels, dtype=np.int32)

    for t in range(T):
        prob_mask = foreground[t]
        nuclear_labels = labels[t]

        if not prob_mask.any() or not (nuclear_labels > 0).any():
            if progress_cb is not None:
                progress_cb(t + 1, T)
            continue

        flow_yx2 = np.stack(
            [filtered[t, 0], filtered[t, 1]], axis=-1
        ).astype(np.float32)
        mags = np.hypot(flow_yx2[..., 0], flow_yx2[..., 1])
        mean_mag = float(mags[prob_mask].mean()) if prob_mask.any() else 0.0
        if mean_mag > 1e-6:
            flow_yx2 = (flow_yx2 / mean_mag).astype(np.float32)

        dist, (ny, nx) = distance_transform_edt(
            nuclear_labels == 0, return_indices=True
        )
        H, W = nuclear_labels.shape
        yi, xi = np.indices((H, W))
        dy = (ny - yi).astype(np.float32)
        dx = (nx - xi).astype(np.float32)
        norm = np.hypot(dy, dx)
        safe = np.where(norm > 0, norm, 1.0)
        grav_y = (dy / safe).astype(np.float32)
        grav_x = (dx / safe).astype(np.float32)
        inside = nuclear_labels > 0
        grav_y[inside] = 0.0
        grav_x[inside] = 0.0

        integrated = _flow_integrate(
            nuclear_labels.astype(np.int32),
            np.ascontiguousarray(flow_yx2, dtype=np.float32),
            grav_y, grav_x,
            dist.astype(np.float32),
            ny.astype(np.int32), nx.astype(np.int32),
            prob_mask,
            int(params.max_iterations),
            float(params.flow_step_scale),
            float(params.flow_weight),
            float(params.capture_radius),
        )

        out_labels[t] = integrated

        if progress_cb is not None:
            progress_cb(t + 1, T)

    return filtered, out_labels
