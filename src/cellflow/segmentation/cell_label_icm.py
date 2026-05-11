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
