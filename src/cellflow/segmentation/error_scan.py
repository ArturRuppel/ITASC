"""Rank ``(frame, cell)`` pairs by how likely they are to be a correction error.

This is the data half of the correction *error worklist*: a pure function over
data the correction widget already has on hand — the tracked label stack and the
divergence ``contours`` map (positive flow divergence, high where Cellpose put a
strong/uncertain boundary). No model work, no new on-disk artifact.

Three error classes are surfaced, each tied to a concrete ``(t, cell_id)`` the
viewer can jump to:

* **high boundary divergence** — the cell mask sits on a lot of contour signal,
  i.e. the boundary the tracker committed to disagrees with the flow field.
* **area jump** — the cell's area changes sharply between adjacent frames
  (a merge swallowing a neighbour, or an over-split halving it).
* **gap / orphan** — the track disappears for a frame then returns (likely an
  ID swap or missed link), or exists for only a frame or two (likely spurious).

Scores are normalised to ``[0, 1]`` so the worklist can colour-grade and sort a
single column regardless of which reason produced the entry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # SciPy is a hard dep of the segmentation package; guard only for typing.
    from scipy.ndimage import mean as _ndi_mean
except Exception:  # pragma: no cover - exercised only if SciPy is unavailable
    _ndi_mean = None


@dataclass(frozen=True, slots=True)
class CellError:
    """One flagged ``(frame, cell)`` pair for the worklist.

    ``score`` is in ``[0, 1]`` (higher = more suspicious); ``reasons`` is a
    short, human-readable tuple explaining why it was flagged.
    """

    t: int
    cell_id: int
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Entry:
    """Mutable-by-replacement accumulator for one ``(t, cell_id)`` key."""

    score: float
    reasons: tuple[str, ...]


def _as_tyx(tracked: np.ndarray) -> np.ndarray:
    """Coerce a tracked label stack to ``(T, Y, X)`` (squeezing a singleton Z)."""
    arr = np.asarray(tracked)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"tracked must be (T, Y, X); got shape {arr.shape}")
    return arr


def _frame_divergence_means(
    frame: np.ndarray, contour_frame: np.ndarray | None, ids: np.ndarray,
) -> dict[int, float]:
    """Mean contour value under each labelled cell in one frame."""
    if contour_frame is None or ids.size == 0:
        return {}
    if _ndi_mean is not None:
        means = _ndi_mean(contour_frame, labels=frame, index=ids)
        return {int(i): float(m) for i, m in zip(ids, np.atleast_1d(means))}
    # Fallback: per-id masking (only hit if SciPy is missing).
    return {int(i): float(contour_frame[frame == int(i)].mean()) for i in ids}


def _divergence_bounds(values: np.ndarray, k: float) -> tuple[float, float]:
    """Robust ``(threshold, high)`` for the divergence reason.

    A cell is flagged only above ``threshold = median + k·MAD`` — a robust
    outlier test that adapts to each run's contrast instead of always flagging a
    fixed top-decile — and the score is anchored by ``high`` (the 99th
    percentile) so a just-over-threshold cell reads faint and the strongest
    divergence reads bold. When the spread is degenerate (more than half the
    values identical, e.g. lots of zeros) it falls back to the midpoint between
    the median and the peak: a lone clear outlier above a sea of zeros is still
    caught, while a genuinely uniform field flags nothing (median == peak leaves
    the band empty).
    """
    if values.size == 0:
        return float("inf"), 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    peak = float(values.max())
    threshold = med + k * 1.4826 * mad if mad > 0.0 else 0.5 * (med + peak)
    high = float(np.percentile(values, 99.0))
    if high <= threshold:
        high = peak
    return threshold, high


def scan_errors(
    tracked: np.ndarray,
    contours: np.ndarray | None = None,
    *,
    max_results: int = 200,
    orphan_max_frames: int = 2,
    area_jump_ratio: float = 2.0,
    area_floor: int = 5,
    divergence_k: float = 3.0,
) -> list[CellError]:
    """Rank likely correction errors in a tracked label stack.

    ``tracked`` is the ``(T, Y, X)`` (or ``(T, 1, Y, X)``) integer label stack;
    ``contours`` is the matching ``(T, Y, X)`` positive-divergence map, or
    ``None`` to skip the divergence reason. Returns up to ``max_results``
    :class:`CellError` entries sorted by descending score (all scores in
    ``[0, 1]`` so they grade comparably across reasons).

    Tuning knobs: ``divergence_k`` sets the robust outlier threshold for the
    divergence reason (median + k·MAD); ``area_floor`` is the smallest mask area
    (px) an area jump must reach to count, so sub-resolution wobble is ignored;
    short tracks touching the first/last frame of a multi-frame stack are treated
    as field-of-view entry/exit and not flagged.
    """
    arr = _as_tyx(tracked)
    n_t = arr.shape[0]
    contour_arr = None if contours is None else _as_tyx(contours)
    if contour_arr is not None and contour_arr.shape != arr.shape:
        raise ValueError(
            f"contours shape {contour_arr.shape} != tracked shape {arr.shape}"
        )

    # Pass 1: per-(t, id) area + divergence mean, and per-id frame presence.
    area: dict[int, dict[int, int]] = {}
    div_mean: dict[int, dict[int, float]] = {}
    frames_of: dict[int, list[int]] = {}
    all_div: list[float] = []
    for t in range(n_t):
        frame = arr[t]
        ids, counts = np.unique(frame, return_counts=True)
        keep = ids != 0
        ids, counts = ids[keep], counts[keep]
        if ids.size == 0:
            continue
        cmeans = _frame_divergence_means(
            frame, None if contour_arr is None else contour_arr[t], ids,
        )
        area_t = area.setdefault(t, {})
        div_t = div_mean.setdefault(t, {})
        for cell_id, count in zip(ids.tolist(), counts.tolist()):
            area_t[cell_id] = int(count)
            frames_of.setdefault(cell_id, []).append(t)
            if cell_id in cmeans:
                div_t[cell_id] = cmeans[cell_id]
                all_div.append(cmeans[cell_id])

    div_values = np.asarray(all_div, dtype=float)
    div_threshold, div_high = _divergence_bounds(div_values, divergence_k)

    entries: dict[tuple[int, int], _Entry] = {}

    def _flag(t: int, cell_id: int, score: float, reason: str) -> None:
        key = (int(t), int(cell_id))
        prev = entries.get(key)
        if prev is None:
            entries[key] = _Entry(score=score, reasons=(reason,))
        else:
            entries[key] = _Entry(
                score=max(prev.score, score), reasons=prev.reasons + (reason,)
            )

    # Reason: high boundary divergence — a robust outlier above the frame's own
    # spread, scored by how far past the threshold it sits (not a fixed decile).
    if div_high > div_threshold:
        div_span = div_high - div_threshold
        for t, by_id in div_mean.items():
            for cell_id, value in by_id.items():
                if value > div_threshold:
                    _flag(
                        t, cell_id, min((value - div_threshold) / div_span, 1.0),
                        "high boundary divergence",
                    )

    # Reasons derived from per-track frame spans (gaps, orphans, area jumps).
    for cell_id, frames in frames_of.items():
        frames.sort()
        span = frames[-1] - frames[0] + 1
        # Orphan: a very short track — unless it touches the first/last frame of
        # a multi-frame stack, where it is probably a cell entering or leaving
        # the field of view rather than a tracking error.
        touches_bound = frames[0] == 0 or frames[-1] == n_t - 1
        if len(frames) <= orphan_max_frames and not (n_t > 1 and touches_bound):
            _flag(
                frames[0], cell_id,
                min(1.0, (orphan_max_frames - len(frames) + 1) / orphan_max_frames),
                f"short track ({len(frames)} frame{'s' if len(frames) != 1 else ''})",
            )
        # Gap: track is absent for some frame(s) between appearances.
        if span > len(frames):
            present = set(frames)
            for t in frames[1:]:
                if (t - 1) not in present:
                    gap = 1
                    probe = t - 2
                    while probe >= frames[0] and probe not in present:
                        gap += 1
                        probe -= 1
                    _flag(
                        t, cell_id, min(1.0, gap / 3.0),
                        f"reappears after {gap}-frame gap",
                    )
        # Area jump: sharp area change between consecutive present frames.
        for prev_t, t in zip(frames, frames[1:]):
            if t - prev_t != 1:
                continue
            a0 = area[prev_t][cell_id]
            a1 = area[t][cell_id]
            lo, hi = sorted((a0, a1))
            # Ignore jumps among sub-resolution masks (a 1->3 px wobble is noise,
            # not a merge/split): require the larger mask to clear ``area_floor``.
            if lo > 0 and hi >= area_floor and hi / lo >= area_jump_ratio:
                ratio = hi / lo
                _flag(
                    t, cell_id, min(1.0, ratio / (area_jump_ratio * 2.0)),
                    f"area ×{ratio:.1f} vs prev frame",
                )

    result = [
        CellError(t=t, cell_id=cell_id, score=entry.score, reasons=entry.reasons)
        for (t, cell_id), entry in entries.items()
    ]
    result.sort(key=lambda e: (-e.score, e.t, e.cell_id))
    return result[:max_results]


__all__ = ["CellError", "scan_errors"]
