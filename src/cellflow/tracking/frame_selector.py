"""Full-frame hypothesis selector for cell-boundary sweeps.

This module ranks complete per-frame hypotheses. It assumes cell positions and
IDs are already anchored by nucleus-derived seeds, so temporal coherence is
measured from same-ID boundary statistics rather than centroid search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True, slots=True)
class SelectorWeights:
    area: float = 1.0
    shape: float = 1.0
    missing: float = 5.0
    extra: float = 2.0
    parameter_switch: float = 0.05


@dataclass(frozen=True, slots=True)
class FrameStats:
    t: int
    p: int
    z: int
    ids: tuple[int, ...]
    areas: np.ndarray
    compactness: np.ndarray
    foreground_area: int
    id_array: np.ndarray | None = field(default=None, compare=False, repr=False)
    id_set: frozenset[int] | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class TransitionScore:
    total: float
    area_cost: float
    shape_cost: float
    missing_count: int
    extra_count: int
    switch_cost: float


@dataclass(frozen=True, slots=True)
class RankedPath:
    score: float
    states: tuple[FrameStats, ...]
    transitions: tuple[TransitionScore, ...]
    state_key: tuple[tuple[int, int], ...] = field(default=(), compare=False, repr=False)


def compute_frame_stats(labels: np.ndarray, *, t: int, p: int, z: int = 0) -> FrameStats:
    """Return compact per-label statistics for one 2D frame hypothesis."""
    arr = np.asarray(labels)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(
                f"Expected a 2D label image or single-slice volume, got shape {arr.shape}"
            )
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label image, got shape {arr.shape}")

    areas = np.bincount(arr.ravel().astype(np.int64))
    if areas.size == 0:
        areas = np.zeros(1, dtype=np.int64)
    id_array = np.flatnonzero(areas).astype(np.int64, copy=False)
    id_array = id_array[id_array != 0]
    ids = tuple(int(i) for i in id_array)
    compactness = _label_compactness(arr, areas)
    foreground_area = int(areas[1:].sum()) if areas.size > 1 else 0
    return FrameStats(
        t=int(t),
        p=int(p),
        z=int(z),
        ids=ids,
        areas=areas,
        compactness=compactness,
        foreground_area=foreground_area,
        id_array=id_array,
        id_set=frozenset(ids),
    )


def _label_compactness(labels: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Return per-label 2D compactness."""
    arr = np.asarray(labels)
    if arr.ndim == 2:
        arr = arr[np.newaxis]

    max_id = len(areas) - 1
    perimeter = np.zeros(len(areas), dtype=np.float64)
    for left, right in ((arr[:, :, :-1], arr[:, :, 1:]), (arr[:, :-1, :], arr[:, 1:, :])):
        diff = left != right
        if not np.any(diff):
            continue
        left_ids = left[diff].astype(np.int64)
        right_ids = right[diff].astype(np.int64)
        if left_ids.size:
            perimeter += np.bincount(left_ids[left_ids != 0], minlength=max_id + 1)[:max_id + 1]
        if right_ids.size:
            perimeter += np.bincount(right_ids[right_ids != 0], minlength=max_id + 1)[:max_id + 1]

    compactness = np.zeros(len(areas), dtype=np.float64)
    valid = (areas > 0) & (perimeter > 0)
    compactness[valid] = np.minimum(1.0, (4.0 * np.pi * areas[valid]) / (perimeter[valid] ** 2))
    return compactness


def score_transition(
    previous: FrameStats,
    current: FrameStats,
    weights: SelectorWeights = SelectorWeights(),
) -> TransitionScore:
    """Score how coherent it is to move from one full frame to the next."""
    if previous.ids == current.ids:
        common = _id_array(previous)
        missing_count = 0
        extra_count = 0
    else:
        prev_ids = _id_set(previous)
        cur_ids = _id_set(current)
        common_set = prev_ids & cur_ids
        common = np.fromiter(sorted(common_set), dtype=np.int64, count=len(common_set))
        missing_count = len(prev_ids - cur_ids)
        extra_count = len(cur_ids - prev_ids)

    area_cost = 0.0
    shape_cost = 0.0
    if common.size:
        prev_area = previous.areas[common].astype(np.float64, copy=False)
        cur_area = current.areas[common].astype(np.float64, copy=False)
        area_cost = float(np.mean(np.abs(np.log((cur_area + 1.0) / (prev_area + 1.0)))))
        prev_shape = previous.compactness[common].astype(np.float64, copy=False)
        cur_shape = current.compactness[common].astype(np.float64, copy=False)
        shape_cost = float(np.mean(np.abs(cur_shape - prev_shape)))

    switch_cost = weights.parameter_switch if previous.p != current.p else 0.0
    total = (
        weights.area * area_cost
        + weights.shape * shape_cost
        + weights.missing * missing_count
        + weights.extra * extra_count
        + switch_cost
    )
    return TransitionScore(
        total=float(total),
        area_cost=area_cost,
        shape_cost=shape_cost,
        missing_count=missing_count,
        extra_count=extra_count,
        switch_cost=float(switch_cost),
    )


def select_top_k_paths(
    candidates_by_t: list[list[FrameStats]],
    *,
    k: int = 5,
    beam_width: int = 200,
    weights: SelectorWeights = SelectorWeights(),
) -> list[RankedPath]:
    """Return low-cost 2D frame paths through candidates_by_t using beam search."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if not candidates_by_t:
        return []
    if any(not candidates for candidates in candidates_by_t):
        raise ValueError("Each timepoint must contain at least one candidate")

    active_paths = [
        RankedPath(
            score=0.0,
            states=(state,),
            transitions=(),
            state_key=((state.p, state.z),),
        )
        for state in candidates_by_t[0]
    ]
    active_paths.sort(key=lambda path: (path.score, _path_state_key(path)))
    active_paths = active_paths[:beam_width]

    for candidates in candidates_by_t[1:]:
        expanded = []
        transition_cache: dict[tuple[int, int], TransitionScore] = {}
        for state in candidates:
            for path in active_paths:
                previous = path.states[-1]
                cache_key = (id(previous), id(state))
                transition = transition_cache.get(cache_key)
                if transition is None:
                    transition = score_transition(previous, state, weights)
                    transition_cache[cache_key] = transition
                expanded.append(
                    RankedPath(
                        score=path.score + transition.total,
                        states=path.states + (state,),
                        transitions=path.transitions + (transition,),
                        state_key=_path_state_key(path) + ((state.p, state.z),),
                    )
                )
        expanded.sort(key=lambda path: (path.score, _path_state_key(path)))
        active_paths = expanded[:beam_width]

    return active_paths[:k]


def _id_array(stats: FrameStats) -> np.ndarray:
    if stats.id_array is not None:
        return stats.id_array
    return np.fromiter(stats.ids, dtype=np.int64, count=len(stats.ids))


def _id_set(stats: FrameStats) -> frozenset[int]:
    if stats.id_set is not None:
        return stats.id_set
    return frozenset(stats.ids)


def _path_state_key(path: RankedPath) -> tuple[tuple[int, int], ...]:
    if path.state_key:
        return path.state_key
    return tuple((state.p, state.z) for state in path.states)


def load_hypothesis_frame_stats(path: str | Path) -> list[list[FrameStats]]:
    """Load per-candidate stats from a CellFlow hypotheses.h5 file."""
    grouped: list[list[FrameStats]] = []
    with h5py.File(Path(path), "r") as h5:
        root = h5["hypotheses"]
        for t_key in sorted(k for k in root.keys() if k.startswith("t")):
            t = int(t_key[1:])
            states = []
            for p_key in sorted(k for k in root[t_key].keys() if k.startswith("p")):
                p = int(p_key[1:])
                labels = root[t_key][p_key]["labels"][:]
                for z in range(labels.shape[0]):
                    states.append(compute_frame_stats(labels[z], t=t, p=p, z=z))
            grouped.append(states)
    return grouped
