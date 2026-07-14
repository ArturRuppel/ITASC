"""Per-frame velocities, per-track motility summaries, and persistence times.

Pure NumPy over :class:`~.trajectories.Trajectory` lists; no I/O, no Qt. Three
public pieces:

* :func:`instantaneous_table` — one row per ``(frame, cell_id)``: position,
  velocity, speed, and net displacement from the track's start. Forward
  differences between consecutive **present** frames, divided by the *real*
  elapsed time ``Δf · dt`` so gaps don't inflate speed. A track's last present
  frame has no forward step → NaN velocity/speed.
* :func:`track_summary_table` — one row per track: path length, net displacement,
  curvilinear / net speed, directionality (confinement) ratio, and the
  directional-autocorrelation persistence time.
* :func:`directional_autocorrelation` / :func:`fit_persistence_time` — the
  shared DAC machinery, reused for the ensemble curve in :mod:`.msd`-adjacent
  code (the dynamics store) and for each track's own persistence time.

``step directions`` use only **real** single-frame steps (``Δf == 1``); a step
spanning a gap is dropped from the direction statistics (its heading over the
gap is unreliable), though it still contributes to path length / speed.
"""
from __future__ import annotations

import numpy as np

from .trajectories import Trajectory

#: Instantaneous per-(frame, cell) value columns (keys ``frame``/``cell_id`` are
#: prepended by the store). Order is the on-disk order.
INSTANTANEOUS_COLUMNS = (
    "x_um",
    "y_um",
    "vx_um_per_s",
    "vy_um_per_s",
    "speed_um_per_s",
    "net_disp_um",
)

#: Ensemble directional-autocorrelation curve columns.
DAC_COLUMNS = ("lag_s", "dac", "n_samples", "sem")

#: Per-track summary value columns (key ``cell_id`` prepended by the store).
TRACK_COLUMNS = (
    "n_frames",
    "n_gaps",
    "frame_start",
    "frame_end",
    "duration_s",
    "path_length_um",
    "net_displacement_um",
    "curvilinear_speed_um_per_s",
    "net_speed_um_per_s",
    "directionality_ratio",
    "persistence_time_s",
)


def instantaneous_table(
    trajectories: list[Trajectory], *, time_interval_s: float
) -> dict[str, np.ndarray]:
    """Column-major ``(frame, cell_id, …INSTANTANEOUS_COLUMNS)`` over all tracks."""
    dt = float(time_interval_s)
    frame: list[int] = []
    cell_id: list[int] = []
    cols: dict[str, list[float]] = {name: [] for name in INSTANTANEOUS_COLUMNS}
    for traj in trajectories:
        xy = traj.xy
        frames = traj.frames
        n = traj.n_frames
        r0 = xy[0]
        for k in range(n):
            frame.append(int(frames[k]))
            cell_id.append(traj.track_id)
            cols["x_um"].append(float(xy[k, 0]))
            cols["y_um"].append(float(xy[k, 1]))
            cols["net_disp_um"].append(float(np.hypot(*(xy[k] - r0))))
            if k < n - 1:
                dt_real = float(frames[k + 1] - frames[k]) * dt
                step = xy[k + 1] - xy[k]
                vx, vy = step / dt_real
                cols["vx_um_per_s"].append(float(vx))
                cols["vy_um_per_s"].append(float(vy))
                cols["speed_um_per_s"].append(float(np.hypot(vx, vy)))
            else:
                cols["vx_um_per_s"].append(np.nan)
                cols["vy_um_per_s"].append(np.nan)
                cols["speed_um_per_s"].append(np.nan)
    out: dict[str, np.ndarray] = {
        "frame": np.asarray(frame, dtype=np.int64),
        "cell_id": np.asarray(cell_id, dtype=np.int64),
    }
    for name in INSTANTANEOUS_COLUMNS:
        out[name] = np.asarray(cols[name], dtype=float)
    return out


def track_summary_table(
    trajectories: list[Trajectory],
    *,
    time_interval_s: float,
    min_track_frames: int = 3,
) -> dict[str, np.ndarray]:
    """Column-major per-track summary; tracks shorter than *min_track_frames* drop."""
    dt = float(time_interval_s)
    cell_id: list[int] = []
    rows: dict[str, list[float]] = {name: [] for name in TRACK_COLUMNS}
    for traj in trajectories:
        if traj.n_frames < int(min_track_frames):
            continue
        xy = traj.xy
        frames = traj.frames
        steps = np.diff(xy, axis=0)
        step_len = np.hypot(steps[:, 0], steps[:, 1])
        path_length = float(step_len.sum())
        net_disp = float(np.hypot(*(xy[-1] - xy[0])))
        duration = float(frames[-1] - frames[0]) * dt
        curv_speed = path_length / duration if duration > 0 else np.nan
        net_speed = net_disp / duration if duration > 0 else np.nan
        dir_ratio = net_disp / path_length if path_length > 0 else np.nan
        persistence = _track_persistence_time(traj, time_interval_s=dt)

        cell_id.append(traj.track_id)
        rows["n_frames"].append(traj.n_frames)
        rows["n_gaps"].append(traj.n_gaps)
        rows["frame_start"].append(int(frames[0]))
        rows["frame_end"].append(int(frames[-1]))
        rows["duration_s"].append(duration)
        rows["path_length_um"].append(path_length)
        rows["net_displacement_um"].append(net_disp)
        rows["curvilinear_speed_um_per_s"].append(curv_speed)
        rows["net_speed_um_per_s"].append(net_speed)
        rows["directionality_ratio"].append(dir_ratio)
        rows["persistence_time_s"].append(persistence)

    out: dict[str, np.ndarray] = {"cell_id": np.asarray(cell_id, dtype=np.int64)}
    # Integer-valued columns stay integers on disk; the rest are floats (NaN-safe).
    int_cols = {"n_frames", "n_gaps", "frame_start", "frame_end"}
    for name in TRACK_COLUMNS:
        dtype = np.int64 if name in int_cols else float
        out[name] = np.asarray(rows[name], dtype=dtype)
    return out


# --------------------------------------------------------- directional autocorr
def step_directions_by_frame(traj: Trajectory) -> dict[int, np.ndarray]:
    """``start_frame -> unit step direction`` for each **real** (``Δf == 1``) step.

    Zero-length steps (a stationary cell) have no defined direction and are
    skipped.
    """
    out: dict[int, np.ndarray] = {}
    frames = traj.frames
    xy = traj.xy
    for k in range(traj.n_frames - 1):
        if int(frames[k + 1] - frames[k]) != 1:
            continue
        step = xy[k + 1] - xy[k]
        norm = float(np.hypot(step[0], step[1]))
        if norm > 0:
            out[int(frames[k])] = step / norm
    return out


def accumulate_dac(
    dirs_by_frame: dict[int, np.ndarray],
    max_lag: int,
    sums: dict[int, float],
    counts: dict[int, int],
    sumsq: dict[int, float] | None = None,
) -> None:
    """Add ``û(f)·û(f+n)`` dot products into *sums* / *counts*, keyed by lag *n*.

    Reused for a single track (fresh accumulators → that track's persistence
    time) and for the ensemble (shared accumulators across all tracks). Pass
    *sumsq* to also accumulate squared dot products (for the ensemble SEM).
    """
    if not dirs_by_frame:
        return
    for f in sorted(dirs_by_frame):
        u = dirs_by_frame[f]
        for n in range(1, max_lag + 1):
            v = dirs_by_frame.get(f + n)
            if v is None:  # f+n absent (gap / track end) — no pair at this lag
                continue
            dot = float(u[0] * v[0] + u[1] * v[1])
            sums[n] = sums.get(n, 0.0) + dot
            counts[n] = counts.get(n, 0) + 1
            if sumsq is not None:
                sumsq[n] = sumsq.get(n, 0.0) + dot * dot


def ensemble_dac(
    trajectories: list[Trajectory], *, time_interval_s: float
) -> tuple[dict[str, np.ndarray], float]:
    """Pooled DAC curve over all tracks → ``(DAC_COLUMNS table, persistence_s)``.

    Each track contributes its real-step direction pairs up to its own span; the
    fitted ensemble persistence time uses the same recipe as the per-track one.
    """
    dt = float(time_interval_s)
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    sumsq: dict[int, float] = {}
    for traj in trajectories:
        dirs = step_directions_by_frame(traj)
        if len(dirs) < 2:
            continue
        accumulate_dac(dirs, max(dirs) - min(dirs), sums, counts, sumsq)

    lags, dac, n_samples = directional_autocorrelation(sums, counts)
    lag_s = lags.astype(float) * dt
    sem = np.asarray(
        [_dac_sem(sums[int(n)], sumsq[int(n)], counts[int(n)]) for n in lags],
        dtype=float,
    )
    table = {"lag_s": lag_s, "dac": dac, "n_samples": n_samples, "sem": sem}
    return table, fit_persistence_time(lag_s, dac)


def _dac_sem(s: float, ss: float, count: int) -> float:
    if count < 2:
        return float("nan")
    var = (ss - s * s / count) / (count - 1)
    return float(np.sqrt(max(var, 0.0) / count))


def directional_autocorrelation(
    sums: dict[int, float], counts: dict[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn accumulated sums/counts into ``(lags, C, n_samples)`` arrays.

    ``lags`` are integer frame lags ``1 … max`` present in *counts*, sorted.
    """
    lags = np.asarray(sorted(counts), dtype=np.int64)
    C = np.asarray([sums[n] / counts[n] for n in lags], dtype=float)
    n_samples = np.asarray([counts[n] for n in lags], dtype=np.int64)
    return lags, C, n_samples


def fit_persistence_time(lags_s: np.ndarray, C: np.ndarray) -> float:
    """Fit ``C = exp(-τ/P)`` over the leading positive run; return ``P`` (s).

    Linear fit of ``ln C`` vs ``τ`` across the contiguous ``C > 0`` prefix
    (autocorrelation decays from 1, so the leading run is the signal). ``P =
    -1/slope``. NaN when fewer than two positive points or a non-negative slope
    (no decay).
    """
    lags_s = np.asarray(lags_s, dtype=float)
    C = np.asarray(C, dtype=float)
    if lags_s.size == 0:
        return float("nan")
    # Leading contiguous positive run.
    positive = C > 0
    if not positive[0]:
        return float("nan")
    cut = int(np.argmin(positive)) if not positive.all() else positive.size
    if cut < 2:
        return float("nan")
    x = lags_s[:cut]
    y = np.log(C[:cut])
    slope = float(np.polyfit(x, y, 1)[0])
    if slope >= 0:
        return float("nan")
    return -1.0 / slope


def _track_persistence_time(traj: Trajectory, *, time_interval_s: float) -> float:
    dirs = step_directions_by_frame(traj)
    if len(dirs) < 2:
        return float("nan")
    max_lag = max(dirs) - min(dirs)
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    accumulate_dac(dirs, max_lag, sums, counts)
    if not counts:
        return float("nan")
    lags, C, _ = directional_autocorrelation(sums, counts)
    return fit_persistence_time(lags.astype(float) * float(time_interval_s), C)
