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
