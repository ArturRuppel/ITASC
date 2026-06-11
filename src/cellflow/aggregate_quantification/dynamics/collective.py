"""Tissue-scale collective motion: alignment, velocity correlation, length scale.

Consumes the per-frame instantaneous table (positions + velocities, the single
source of truth) and, for each frame, measures how coherently neighbours move:

* **order parameter** ``φ = |⟨v_i/|v_i|⟩|`` ∈ [0,1] — 0 random, 1 fully aligned
  (uses raw velocities).
* **velocity correlation** of fluctuations ``δv_i = v_i − ⟨v⟩`` (the per-frame
  drift removed): ``C(r) = ⟨δv_i·δv_j⟩_{|r_i−r_j|≈r} / ⟨δv_i·δv_i⟩``, normalised
  so ``C → 1`` as ``r → 0``.
* **correlation length** ``ξ`` = the separation where ``C(r)`` decays to ``1/e``
  (linear interpolation between bin centres); NaN when it never does.
* **nearest-neighbour distance** (median, per frame) — a natural length scale and
  the default correlation bin width.

Pairs are binned by separation with width *corr_bin_um* (defaults to the global
median NN distance). The per-frame ``C(r)`` curves are also pooled across frames
into a single dataset-level curve. Pure NumPy; no I/O.
"""
from __future__ import annotations

import numpy as np

#: Per-frame collective columns.
COLLECTIVE_COLUMNS = ("frame", "n_cells", "order_param", "corr_length_um", "nn_distance_um")
#: Pooled correlation-curve columns.
CORR_CURVE_COLUMNS = ("separation_um", "corr", "n_pairs")

_INV_E = float(np.exp(-1.0))


def collective_tables(
    instantaneous: dict[str, np.ndarray],
    *,
    corr_bin_um: float | None = None,
    min_cells: int = 5,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return ``(collective_table, corr_curve_table)`` from the instantaneous table.

    *min_cells* is the minimum velocity-bearing cells a frame needs for its
    correlation/alignment metrics (fewer → NaN, but the frame's NN distance is
    still recorded). *corr_bin_um* defaults to the global median NN distance.
    """
    frames_all = instantaneous["frame"]
    x = instantaneous["x_um"]
    y = instantaneous["y_um"]
    vx = instantaneous["vx_um_per_s"]
    vy = instantaneous["vy_um_per_s"]

    per_frame = _per_frame_views(frames_all, x, y, vx, vy)
    bin_width = _resolve_bin_width(corr_bin_um, per_frame)

    frames: list[int] = []
    n_cells: list[int] = []
    order: list[float] = []
    corr_len: list[float] = []
    nn_dist: list[float] = []

    # Pooled-curve accumulators, keyed by integer bin index.
    pooled_dot: dict[int, float] = {}
    pooled_count: dict[int, int] = {}
    pooled_var_sum = 0.0
    pooled_var_count = 0

    for frame, (pos_all, pos_v, vel) in per_frame.items():
        frames.append(frame)
        n_cells.append(pos_v.shape[0])
        nn_dist.append(_median_nn_distance(pos_all))
        if pos_v.shape[0] < int(min_cells):
            order.append(np.nan)
            corr_len.append(np.nan)
            continue
        order.append(_order_parameter(vel))
        dV = vel - vel.mean(axis=0)
        var = float(np.mean(np.einsum("ij,ij->i", dV, dV)))
        centers, C, counts, dot_by_bin = _velocity_correlation(pos_v, dV, bin_width, var)
        corr_len.append(_one_over_e_length(centers, C))
        # Pool raw dot sums + the variance normaliser (a single global C(0)).
        for b, d_sum in dot_by_bin.items():
            pooled_dot[b] = pooled_dot.get(b, 0.0) + d_sum
            pooled_count[b] = pooled_count.get(b, 0) + counts[b]
        pooled_var_sum += var * pos_v.shape[0]
        pooled_var_count += pos_v.shape[0]

    collective = {
        "frame": np.asarray(frames, dtype=np.int64),
        "n_cells": np.asarray(n_cells, dtype=np.int64),
        "order_param": np.asarray(order, dtype=float),
        "corr_length_um": np.asarray(corr_len, dtype=float),
        "nn_distance_um": np.asarray(nn_dist, dtype=float),
    }
    curve = _pooled_curve(pooled_dot, pooled_count, pooled_var_sum, pooled_var_count, bin_width)
    return collective, curve


# --------------------------------------------------------------------- helpers
def _per_frame_views(frames_all, x, y, vx, vy):
    """``frame -> (positions_all, positions_with_velocity, velocities)`` arrays."""
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for frame in np.unique(frames_all):
        m = frames_all == frame
        pos_all = np.column_stack((x[m], y[m]))
        has_v = np.isfinite(vx[m]) & np.isfinite(vy[m])
        pos_v = pos_all[has_v]
        vel = np.column_stack((vx[m][has_v], vy[m][has_v]))
        out[int(frame)] = (pos_all, pos_v, vel)
    return out


def _resolve_bin_width(corr_bin_um, per_frame) -> float:
    if corr_bin_um is not None and float(corr_bin_um) > 0:
        return float(corr_bin_um)
    nn = [_median_nn_distance(pos_all) for pos_all, _, _ in per_frame.values()]
    nn = [v for v in nn if np.isfinite(v) and v > 0]
    return float(np.median(nn)) if nn else 1.0


def _order_parameter(vel: np.ndarray) -> float:
    speed = np.hypot(vel[:, 0], vel[:, 1])
    moving = speed > 0
    if not moving.any():
        return float("nan")
    units = vel[moving] / speed[moving, None]
    return float(np.hypot(*units.mean(axis=0)))


def _velocity_correlation(pos: np.ndarray, dV: np.ndarray, bin_width: float, var: float):
    """Per-frame ``C(r)``: bin centres, normalised correlation, pair counts, dot sums."""
    n = pos.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    diff = pos[iu] - pos[ju]
    r = np.hypot(diff[:, 0], diff[:, 1])
    dots = np.einsum("ij,ij->i", dV[iu], dV[ju])
    bins = np.floor(r / bin_width).astype(np.int64)

    dot_by_bin: dict[int, float] = {}
    count_by_bin: dict[int, int] = {}
    for b, d in zip(bins.tolist(), dots.tolist()):
        dot_by_bin[b] = dot_by_bin.get(b, 0.0) + d
        count_by_bin[b] = count_by_bin.get(b, 0) + 1

    order = sorted(dot_by_bin)
    centers = np.asarray([(b + 0.5) * bin_width for b in order], dtype=float)
    denom = var if var > 0 else np.nan
    C = np.asarray([dot_by_bin[b] / count_by_bin[b] / denom for b in order], dtype=float)
    counts = {b: count_by_bin[b] for b in order}
    return centers, C, counts, dot_by_bin


def pooled_corr_length(corr_curve: dict[str, np.ndarray]) -> float:
    """Single per-tissue correlation length ``ξ`` from the pooled ``C(r)`` curve.

    Applies the same ``1/e`` crossing rule as the per-frame ``corr_length_um``
    column to the dataset-level pooled curve; returns ``NaN`` when the curve is
    empty or never decays to ``1/e``.
    """
    centers = np.asarray(corr_curve.get("separation_um", []), dtype=float)
    C = np.asarray(corr_curve.get("corr", []), dtype=float)
    if centers.size == 0:
        return float("nan")
    return _one_over_e_length(centers, C)


def _one_over_e_length(centers: np.ndarray, C: np.ndarray) -> float:
    """Separation where *C* first decays to ``1/e``, by linear interpolation."""
    if centers.size == 0:
        return float("nan")
    for i in range(centers.size):
        if C[i] <= _INV_E:
            if i == 0:
                return float(centers[0])
            c0, c1 = C[i - 1], C[i]
            if c0 == c1:
                return float(centers[i])
            t = (c0 - _INV_E) / (c0 - c1)
            return float(centers[i - 1] + t * (centers[i] - centers[i - 1]))
    return float("nan")  # never decays to 1/e within the field


def _pooled_curve(dot_by_bin, count_by_bin, var_sum, var_count, bin_width):
    if not count_by_bin or var_count == 0:
        return {name: np.asarray([], dtype=float) for name in CORR_CURVE_COLUMNS}
    var = var_sum / var_count
    order = sorted(count_by_bin)
    sep = np.asarray([(b + 0.5) * bin_width for b in order], dtype=float)
    corr = np.asarray(
        [dot_by_bin[b] / count_by_bin[b] / var if var > 0 else np.nan for b in order],
        dtype=float,
    )
    n_pairs = np.asarray([count_by_bin[b] for b in order], dtype=np.int64)
    return {"separation_um": sep, "corr": corr, "n_pairs": n_pairs}


def _median_nn_distance(pos: np.ndarray) -> float:
    n = pos.shape[0]
    if n < 2:
        return float("nan")
    # Pairwise distances; small per-frame cell counts make this cheap and exact.
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    np.fill_diagonal(dist, np.inf)
    return float(np.median(dist.min(axis=1)))
