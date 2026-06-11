"""Ensemble time-averaged mean-square displacement and its power-law fit.

For each integer frame lag ``n``, the MSD is averaged over **all tracks** and
**all present overlapping origin pairs** ``(k, k+n)`` whose frames are separated
by exactly ``n`` (a gap inside a pair breaks it, so only genuine ``n``-frame
displacements count):

    MSD(n) = mean over tracks, origins of |r(f+n) − r(f)|²        (µm²)

Lags run ``1 … n_max`` with ``n_max = max(1, ⌊¼ · longest track span⌋)`` — the
usual reliability cap, since large-lag estimates rest on few, strongly
overlapping samples. The curve is fit in log-log to ``MSD = 2·d·D·τ^α`` (``d =
2``): ``α`` is the anomalous exponent (≈1 diffusive, →2 ballistic/persistent, <1
confined) and ``D`` the (generalised) diffusion coefficient in µm²/s. Pure
NumPy; no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trajectories import Trajectory

#: Spatial dimensionality of the trajectories (2-D centroids everywhere here).
_DIMS = 2

#: MSD curve columns (the store prepends nothing; these are the full set).
MSD_COLUMNS = ("lag_s", "msd_um2", "n_samples", "sem")

#: Per-track MSD-fit value columns, merged into the per-track summary table.
MSD_TRACK_COLUMNS = ("msd_D_um2_per_s", "msd_alpha", "msd_r2")
#: Default fixed lag window (frames) for the per-track MSD fit. A fixed window
#: makes every track's ``D``/``α`` comparable and excludes the noisy long-lag tail.
DEFAULT_MSD_TRACK_WINDOW = 8


@dataclass(frozen=True)
class MsdFit:
    """Power-law fit of the MSD curve: ``MSD = 2·d·D·τ^α``."""

    D_um2_per_s: float
    alpha: float
    r2: float


def ensemble_msd(
    trajectories: list[Trajectory],
    *,
    time_interval_s: float,
    n_max: int | None = None,
) -> dict[str, np.ndarray]:
    """Column-major ``MSD_COLUMNS`` table (``lag_s, msd_um2, n_samples, sem``)."""
    dt = float(time_interval_s)
    if n_max is None:
        n_max = _default_n_max(trajectories)
    # lag -> running sum of squared displacements and a list for the SEM.
    sq_by_lag: dict[int, list[float]] = {}
    for traj in trajectories:
        _accumulate_track_sq(traj, n_max, sq_by_lag)

    lags = np.asarray(sorted(sq_by_lag), dtype=np.int64)
    msd = np.empty(lags.size, dtype=float)
    n_samples = np.empty(lags.size, dtype=np.int64)
    sem = np.empty(lags.size, dtype=float)
    for i, n in enumerate(lags):
        vals = np.asarray(sq_by_lag[int(n)], dtype=float)
        msd[i] = float(vals.mean())
        n_samples[i] = vals.size
        sem[i] = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan
    return {
        "lag_s": lags.astype(float) * dt,
        "msd_um2": msd,
        "n_samples": n_samples,
        "sem": sem,
    }


def fit_msd_power_law(
    lag_s: np.ndarray, msd_um2: np.ndarray, *, min_samples_mask: np.ndarray | None = None
) -> MsdFit:
    """Log-log fit ``log MSD = log(2dD) + α·log τ`` → :class:`MsdFit`.

    *min_samples_mask* optionally restricts the fit to well-sampled lags. NaNs in
    ``D``/``α``/``r2`` when fewer than two usable positive points remain.
    """
    lag_s = np.asarray(lag_s, dtype=float)
    msd_um2 = np.asarray(msd_um2, dtype=float)
    valid = (lag_s > 0) & (msd_um2 > 0) & np.isfinite(lag_s) & np.isfinite(msd_um2)
    if min_samples_mask is not None:
        valid &= np.asarray(min_samples_mask, dtype=bool)
    if int(valid.sum()) < 2:
        return MsdFit(float("nan"), float("nan"), float("nan"))
    x = np.log(lag_s[valid])
    y = np.log(msd_um2[valid])
    slope, intercept = np.polyfit(x, y, 1)
    alpha = float(slope)
    D = float(np.exp(intercept) / (2.0 * _DIMS))
    r2 = _r_squared(x, y, slope, intercept)
    return MsdFit(D_um2_per_s=D, alpha=alpha, r2=r2)


def per_track_msd_fit(
    trajectories: list[Trajectory],
    *,
    time_interval_s: float,
    window_frames: int = DEFAULT_MSD_TRACK_WINDOW,
) -> dict[int, MsdFit]:
    """Fit each track's own MSD over a shared fixed lag window → ``track_id → MsdFit``.

    For each track the MSD is averaged over its present ``(k, k+n)`` origin pairs
    for lags ``1 … window_frames`` (the same rule as :func:`ensemble_msd`, capped
    at the window), then fit in log-log. A track that cannot fill at least two
    distinct lags within the window yields ``MsdFit(nan, nan, nan)``, so its
    ``D``/``α``/``r2`` are NaN rather than a spurious single-point slope.
    """
    dt = float(time_interval_s)
    window = int(window_frames)
    fits: dict[int, MsdFit] = {}
    for traj in trajectories:
        sq_by_lag: dict[int, list[float]] = {}
        _accumulate_track_sq(traj, window, sq_by_lag)
        if len(sq_by_lag) < 2:
            fits[traj.track_id] = MsdFit(float("nan"), float("nan"), float("nan"))
            continue
        lags = np.asarray(sorted(sq_by_lag), dtype=np.int64)
        msd = np.asarray([float(np.mean(sq_by_lag[int(n)])) for n in lags], dtype=float)
        fits[traj.track_id] = fit_msd_power_law(lags.astype(float) * dt, msd)
    return fits


def _accumulate_track_sq(
    traj: Trajectory, n_max: int, sq_by_lag: dict[int, list[float]]
) -> None:
    frames = traj.frames
    xy = traj.xy
    # frame -> position index, so an origin pair is "present" only when both
    # endpoints exist and are exactly n frames apart.
    pos = {int(f): i for i, f in enumerate(frames)}
    for f, i in pos.items():
        for n in range(1, n_max + 1):
            j = pos.get(f + n)
            if j is None:
                continue
            d = xy[j] - xy[i]
            sq_by_lag.setdefault(n, []).append(float(d[0] * d[0] + d[1] * d[1]))


def _default_n_max(trajectories: list[Trajectory]) -> int:
    spans = [int(t.frames[-1] - t.frames[0]) for t in trajectories if t.n_frames >= 2]
    longest = max(spans) if spans else 1
    return max(1, longest // 4)


def _r_squared(x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot
