"""HDF5 hypothesis pool for nucleus segmentation candidates.

Schema: hypotheses/t{t:03d}/p{p:03d}/labels
Each labels dataset has shape (Z, Y, X) and dtype uint32.
Parameters are stored as group attributes on each p group.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np

from cellflow.segmentation import (
    CellposeFlowHypothesisParams,
    ContourWatershedParams,
    NucleusHypothesisParams,
    SeededWatershedParams,
    compute_seeded_watershed,
)

_LABEL_DTYPE = np.uint32
_ROOT_GROUP = "hypotheses"


def _check_schema(h5: "h5py.File", path: Path) -> None:
    """Raise if the file uses the old t/z/p layout from v1."""
    layout = h5.attrs.get("layout", "")
    if layout and "z{z" in str(layout):
        raise ValueError(
            f"{path} uses the v1 t/z/p schema and cannot be read by v2. "
            "Re-generate the hypothesis database."
        )


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    """A single (t, p) label volume payload."""

    t: int
    p: int
    labels: np.ndarray  # shape (Z, Y, X), dtype uint32
    params: NucleusHypothesisParams | CellposeFlowHypothesisParams | ContourWatershedParams | SeededWatershedParams


def _values(current: float, minimum: float, maximum: float, step: float) -> list[float]:
    if step > 0 and minimum != maximum:
        vals = np.arange(float(minimum), float(maximum) + step / 2.0, step, dtype=np.float64)
        return [float(v) for v in vals]
    return [float(current)]


def _int_values(current: int, minimum: int, maximum: int, step: int) -> list[int]:
    if step > 0 and minimum != maximum:
        return list(range(int(minimum), int(maximum) + 1, max(1, int(step))))
    return [int(current)]


@dataclass(frozen=True, slots=True)
class SeededWatershedSweepSpec:
    """Parameter sweep spec for nucleus-seeded watershed cell hypothesis generation."""

    basin: str = "prob"
    foreground_threshold: float = 0.5
    foreground_threshold_min: float = 0.5
    foreground_threshold_max: float = 0.5
    foreground_threshold_step: float = 0.05
    compactness: float = 0.0
    compactness_min: float = 0.0
    compactness_max: float = 0.0
    compactness_step: float = 0.1


def build_seeded_watershed_parameter_sets(spec: SeededWatershedSweepSpec) -> list[SeededWatershedParams]:
    """Return the deterministic list of SeededWatershedParams for this sweep spec."""
    fg_vals = _values(spec.foreground_threshold, spec.foreground_threshold_min, spec.foreground_threshold_max, spec.foreground_threshold_step)
    compactness_vals = _values(spec.compactness, spec.compactness_min, spec.compactness_max, spec.compactness_step)
    return [
        SeededWatershedParams(basin=spec.basin, foreground_threshold=float(fg), compactness=float(c))
        for fg in fg_vals
        for c in compactness_vals
    ]


def normalize_seeded_watershed_dp_stack(
    dp_stack: np.ndarray,
    prob_shape: tuple[int, int, int, int],
) -> np.ndarray:
    """Return flow vectors as (T, Z, C, Y, X), accepting common Cellpose layouts."""
    dp = np.asarray(dp_stack, dtype=np.float32)
    n_t, n_z, n_y, n_x = prob_shape

    if dp.ndim == 4:
        if dp.shape == (n_z, n_y, n_x, 2) or dp.shape == (n_z, n_y, n_x, 3):
            dp = np.moveaxis(dp, -1, 1)[np.newaxis]
        elif dp.shape[0] == n_z and dp.shape[1] in (2, 3) and dp.shape[2:] == (n_y, n_x):
            dp = dp[np.newaxis]
        elif dp.shape[0] in (2, 3) and dp.shape[1:] == (n_z, n_y, n_x):
            dp = np.moveaxis(dp, 0, 1)[np.newaxis]
        else:
            raise ValueError(
                f"Expected dp stack matching prob shape {prob_shape}, got {dp.shape}"
            )
    elif dp.ndim == 5:
        if dp.shape == (n_t, n_z, n_y, n_x, 2) or dp.shape == (n_t, n_z, n_y, n_x, 3):
            dp = np.moveaxis(dp, -1, 2)
        elif dp.shape[0] == n_t and dp.shape[1] == n_z and dp.shape[2] in (2, 3) and dp.shape[3:] == (n_y, n_x):
            pass
        elif dp.shape[0] == n_t and dp.shape[1] in (2, 3) and dp.shape[2:] == (n_z, n_y, n_x):
            dp = np.moveaxis(dp, 1, 2)
        else:
            raise ValueError(
                f"Expected dp stack matching prob shape {prob_shape}, got {dp.shape}"
            )
    else:
        raise ValueError(f"Expected dp stack with 4 or 5 dimensions, got shape {dp.shape}")

    return np.asarray(dp, dtype=np.float32)


def normalize_seeded_watershed_nucleus_stack(
    nucleus_stack: np.ndarray,
    prob_shape: tuple[int, int, int, int],
) -> np.ndarray:
    """Return nucleus seed labels as (T, Z, Y, X), accepting 2D tracked labels."""
    nucleus = np.asarray(nucleus_stack)
    n_t, n_z, n_y, n_x = prob_shape

    if nucleus.ndim == 2:
        if nucleus.shape != (n_y, n_x):
            raise ValueError(
                f"Expected nucleus labels matching prob shape {prob_shape}, got {nucleus.shape}"
            )
        nucleus = np.broadcast_to(nucleus, (n_t, n_z, n_y, n_x)).copy()
    elif nucleus.ndim == 3:
        if nucleus.shape == (n_t, n_y, n_x):
            nucleus = np.broadcast_to(nucleus[:, np.newaxis], (n_t, n_z, n_y, n_x)).copy()
        elif nucleus.shape == (n_z, n_y, n_x) and n_t == 1:
            nucleus = nucleus[np.newaxis]
        else:
            raise ValueError(
                f"Expected nucleus labels matching prob shape {prob_shape}, got {nucleus.shape}"
            )
    elif nucleus.ndim == 4:
        if nucleus.shape == (n_t, n_z, n_y, n_x):
            pass
        elif nucleus.shape == (1, n_t, n_y, n_x):
            nucleus = np.broadcast_to(nucleus[0, :, np.newaxis], (n_t, n_z, n_y, n_x)).copy()
        elif nucleus.shape == (1, n_z, n_y, n_x) and n_t == 1:
            pass
        else:
            raise ValueError(
                f"Expected nucleus labels matching prob shape {prob_shape}, got {nucleus.shape}"
            )
    else:
        raise ValueError(f"Expected nucleus labels with 2-4 dimensions, got shape {nucleus.shape}")

    return np.asarray(nucleus)


def _run_seeded_watershed_task(
    args: tuple[int, int, "SeededWatershedParams", np.ndarray, np.ndarray | None, np.ndarray],
) -> "HypothesisRecord":
    t, p_idx, params, prob_t, dp_t, nuc_t = args
    n_z = prob_t.shape[0]
    slices = []
    for z in range(n_z):
        dp_2d = dp_t[z] if dp_t is not None else None
        slices.append(compute_seeded_watershed(prob_t[z], dp_2d, nuc_t[z], params))
    return HypothesisRecord(t=t, p=p_idx, labels=np.stack(slices, axis=0), params=params)


def _ordered_bounded_map(fn, inputs: Iterable, max_workers: int) -> Iterator:
    """Map fn over inputs while keeping at most max_workers submitted tasks."""
    if max_workers <= 1:
        for item in inputs:
            yield fn(item)
        return

    from concurrent.futures import ThreadPoolExecutor

    iterator = iter(inputs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = deque()
        for _ in range(max_workers):
            try:
                pending.append(executor.submit(fn, next(iterator)))
            except StopIteration:
                break

        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                pending.append(executor.submit(fn, next(iterator)))
            except StopIteration:
                pass


def iter_seeded_watershed_records(
    prob_stack: np.ndarray,
    dp_stack: np.ndarray | None,
    nucleus_stack: np.ndarray,
    spec: SeededWatershedSweepSpec,
    n_workers: int = 1,
) -> Iterator[HypothesisRecord]:
    """Yield one HypothesisRecord per (t, p) for seeded-watershed cell segmentation.

    prob_stack:    (T, Z, Y, X) float32 probability logits
    dp_stack:      (T, Z, 2, Y, X) float32 flow vectors; required when basin='flow_mag'
    nucleus_stack: (T, Z, Y, X) int32 tracked nucleus labels used as watershed seeds
    """
    params_list = build_seeded_watershed_parameter_sets(spec)
    if not params_list:
        return

    prob_stack = np.asarray(prob_stack, dtype=np.float32)
    if prob_stack.ndim == 3:
        prob_stack = prob_stack[np.newaxis]
    if dp_stack is not None:
        dp_stack = normalize_seeded_watershed_dp_stack(dp_stack, prob_stack.shape)
    nucleus_stack = normalize_seeded_watershed_nucleus_stack(nucleus_stack, prob_stack.shape)

    n_t = prob_stack.shape[0]
    def tasks():
        for t in range(n_t):
            for p_idx, params in enumerate(params_list):
                yield (t, p_idx, params, prob_stack[t], dp_stack[t] if dp_stack is not None else None, nucleus_stack[t])

    if n_workers > 1:
        yield from _ordered_bounded_map(_run_seeded_watershed_task, tasks(), n_workers)
    else:
        for a in tasks():
            yield _run_seeded_watershed_task(a)


# ---------------------------------------------------------------------------
# Legacy read helpers — still used by extend.py, ingest.py, propagator.py
# ---------------------------------------------------------------------------

def read_hypothesis_labels(path: str | Path, t: int, p: int) -> np.ndarray:
    """Read the (Z, Y, X) label volume for one (t, p) entry."""
    with h5py.File(Path(path), "r") as h5:
        _check_schema(h5, Path(path))
        return np.asarray(h5[f"{_ROOT_GROUP}/t{t:03d}/p{p:03d}/labels"], dtype=_LABEL_DTYPE)


def list_hypotheses(path: str | Path) -> tuple[int, dict[int, dict]]:
    """Return (n_p, params_by_p_index) from the first timepoint in the file.

    n_p is the number of parameter sets. params_by_p_index maps p index to
    the attribute dict stored on that group.
    """
    with h5py.File(Path(path), "r") as h5:
        _check_schema(h5, Path(path))
        root = h5[_ROOT_GROUP]
        t_keys = sorted(k for k in root.keys() if k.startswith("t"))
        if not t_keys:
            return 0, {}
        first_t = root[t_keys[0]]
        p_keys = sorted(k for k in first_t.keys() if k.startswith("p"))
        n_p = len(p_keys)
        params_by_p: dict[int, dict] = {}
        for p_name in p_keys:
            p_idx = int(p_name[1:])
            params_by_p[p_idx] = dict(first_t[p_name].attrs)
        return n_p, params_by_p


