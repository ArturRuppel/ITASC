"""HDF5 hypothesis pool for nucleus segmentation candidates.

Schema: hypotheses/t{t:03d}/p{p:03d}/labels
Each labels dataset has shape (Z, Y, X) and dtype uint32.
Parameters are stored as group attributes on each p group.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from cellflow.segmentation._array_utils import normalize_seeded_watershed_dp_stack
from cellflow.segmentation.nucleus_segmentation import (
    ContourWatershedParams,
    compute_contour_watershed,
)

_LABEL_DTYPE = np.uint32
_ROOT_GROUP = "hypotheses"


@dataclass(frozen=True, slots=True)
class SeededWatershedParams:
    """Parameters for Cellpose-flow-seeded watershed hypotheses."""

    basin: str = "probability"
    foreground_threshold: float = 0.5
    compactness: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {"method": "seeded_watershed", **asdict(self)}


@dataclass(frozen=True, slots=True)
class SeededWatershedSweepSpec:
    """Compact sweep specification for seeded watershed hypotheses."""

    basin: str = "probability"
    foreground_threshold: float = 0.5
    foreground_threshold_min: float = 0.5
    foreground_threshold_max: float = 0.5
    compactness: float = 0.0
    compactness_min: float = 0.0
    compactness_max: float = 0.0


@dataclass(frozen=True, slots=True)
class ContourWatershedSweepSpec:
    """Compact sweep specification for contour watershed hypotheses."""

    seed_distance: int = 10
    foreground_threshold: float = 0.5
    ridge_threshold: float = 0.5


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    """One hypothesis label volume for a time point and parameter index."""

    t: int
    p: int
    labels: np.ndarray
    params: object


def _param_attrs(params: object) -> dict[str, object]:
    if hasattr(params, "to_dict"):
        return dict(params.to_dict())  # type: ignore[no-any-return]
    if hasattr(params, "__dataclass_fields__"):
        return asdict(params)  # type: ignore[arg-type]
    return {}


def _seeded_params_from_spec(spec: SeededWatershedSweepSpec) -> list[SeededWatershedParams]:
    return [
        SeededWatershedParams(
            basin=spec.basin,
            foreground_threshold=spec.foreground_threshold,
            compactness=spec.compactness,
        )
    ]


def _contour_params_from_spec(spec: ContourWatershedSweepSpec) -> list[ContourWatershedParams]:
    return [
        ContourWatershedParams(
            seed_distance=spec.seed_distance,
            foreground_threshold=spec.foreground_threshold,
            ridge_threshold=spec.ridge_threshold,
        )
    ]


def _normalize_stack_4d(stack: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(stack)
    if arr.ndim == 3:
        return arr[:, np.newaxis]
    if arr.ndim != 4:
        raise ValueError(f"Expected {name} as (T, Z, Y, X), got {arr.shape}")
    return arr


def _normalize_nucleus_stack(nucleus: np.ndarray, prob_shape: tuple[int, int, int, int]) -> np.ndarray:
    arr = np.asarray(nucleus, dtype=np.uint32)
    n_t, n_z, n_y, n_x = prob_shape
    if arr.ndim == 3:
        arr = arr[:, np.newaxis]
    elif arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == n_t:
        arr = np.moveaxis(arr, 0, 1)
    elif arr.ndim != 4:
        raise ValueError(f"Expected nucleus labels as 3D or 4D, got {arr.shape}")

    if arr.shape[0] != n_t or arr.shape[2:] != (n_y, n_x):
        raise ValueError(f"Nucleus labels shape {arr.shape} does not match probability shape {prob_shape}")
    if arr.shape[1] == 1 and n_z != 1:
        arr = np.broadcast_to(arr, (n_t, n_z, n_y, n_x)).copy()
    if arr.shape[1] != n_z:
        raise ValueError(f"Nucleus labels shape {arr.shape} does not match probability shape {prob_shape}")
    return arr


def compute_seeded_watershed(
    prob_2d: np.ndarray,
    dp_2d: np.ndarray | None,
    seeds_2d: np.ndarray,
    params: SeededWatershedParams,
) -> np.ndarray:
    """Run a simple seeded watershed for one 2D plane."""
    from skimage.segmentation import watershed

    prob = np.asarray(prob_2d, dtype=np.float32)
    seeds = np.asarray(seeds_2d, dtype=np.uint32)
    mask = (prob >= params.foreground_threshold) | (seeds > 0)
    if params.basin == "flow_mag" and dp_2d is not None:
        basin = -np.linalg.norm(np.asarray(dp_2d, dtype=np.float32), axis=0)
    else:
        basin = -prob
    labels = watershed(
        basin,
        markers=seeds.astype(np.int32, copy=False),
        mask=mask,
        compactness=params.compactness,
        watershed_line=False,
    )
    return np.asarray(labels, dtype=_LABEL_DTYPE)


def iter_seeded_watershed_records(
    prob: np.ndarray,
    dp: np.ndarray | None,
    nucleus: np.ndarray,
    spec: SeededWatershedSweepSpec,
    *,
    params_list: Iterable[SeededWatershedParams] | None = None,
) -> Iterator[HypothesisRecord]:
    """Yield seeded watershed records in deterministic ``t`` then ``p`` order."""
    prob4 = _normalize_stack_4d(prob, name="probability").astype(np.float32, copy=False)
    nucleus4 = _normalize_nucleus_stack(nucleus, prob4.shape)
    dp5 = normalize_seeded_watershed_dp_stack(dp, prob4.shape) if dp is not None else None
    params = list(params_list) if params_list is not None else _seeded_params_from_spec(spec)

    n_t, n_z, n_y, n_x = prob4.shape
    for t in range(n_t):
        for p_idx, param in enumerate(params):
            labels = np.zeros((n_z, n_y, n_x), dtype=_LABEL_DTYPE)
            for z in range(n_z):
                dp_plane = None if dp5 is None else dp5[t, z]
                labels[z] = compute_seeded_watershed(
                    prob4[t, z],
                    dp_plane,
                    nucleus4[t, z],
                    param,
                )
            yield HypothesisRecord(t=t, p=p_idx, labels=labels, params=param)


def _ordered_bounded_map(
    fn: Callable[[object], object],
    items: Iterable[object],
    *,
    max_workers: int,
) -> Iterator[object]:
    """Map while preserving order and limiting in-flight input consumption."""
    if max_workers <= 1:
        for item in items:
            yield fn(item)
        return

    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending: deque[Future] = deque()
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


def _prepare_contour_watershed_frame(
    boundary: np.ndarray,
    foreground_frame: np.ndarray,
    params: ContourWatershedParams,
) -> np.ndarray:
    return np.asarray(foreground_frame, dtype=np.float32)


def _run_watershed_task(args: tuple[int, int, ContourWatershedParams, np.ndarray, np.ndarray]) -> HypothesisRecord:
    t, p_idx, param, contour_frame, fg_frame = args
    labels = compute_contour_watershed(contour_frame, fg_frame, param)
    return HypothesisRecord(
        t=t,
        p=p_idx,
        labels=labels[np.newaxis].astype(_LABEL_DTYPE, copy=False),
        params=param,
    )


def _run_cached_watershed_task(args: tuple[int, int, ContourWatershedParams, np.ndarray, np.ndarray]) -> HypothesisRecord:
    t, p_idx, param, boundary, prepared_foreground = args
    labels = compute_contour_watershed(boundary, prepared_foreground, param)
    return HypothesisRecord(
        t=t,
        p=p_idx,
        labels=labels[np.newaxis].astype(_LABEL_DTYPE, copy=False),
        params=param,
    )


def iter_contour_watershed_records(
    contour: np.ndarray,
    foreground: np.ndarray,
    spec: ContourWatershedSweepSpec,
    *,
    n_workers: int = 1,
    params_list: Iterable[ContourWatershedParams] | None = None,
) -> Iterator[HypothesisRecord]:
    """Yield contour watershed records in deterministic ``t`` then ``p`` order."""
    contour3 = np.asarray(contour, dtype=np.float32)
    foreground3 = np.asarray(foreground, dtype=np.float32)
    if contour3.ndim == 2:
        contour3 = contour3[np.newaxis]
    if foreground3.ndim == 2:
        foreground3 = foreground3[np.newaxis]
    if contour3.shape != foreground3.shape:
        raise ValueError(f"Contour shape {contour3.shape} does not match foreground shape {foreground3.shape}")

    params = list(params_list) if params_list is not None else _contour_params_from_spec(spec)
    deterministic = all(
        param.noise_scale == 0.0 and param.noise_blur_sigma == 0.0
        for param in params
    )

    if deterministic:
        tasks = []
        cache: dict[tuple[int, float, float], np.ndarray] = {}
        for t in range(contour3.shape[0]):
            for p_idx, param in enumerate(params):
                key = (t, param.foreground_threshold, param.ridge_threshold)
                if key not in cache:
                    cache[key] = _prepare_contour_watershed_frame(
                        contour3[t],
                        foreground3[t],
                        param,
                    )
                tasks.append((t, p_idx, param, contour3[t], cache[key]))
        yield from _ordered_bounded_map(
            _run_cached_watershed_task,
            tasks,
            max_workers=max(1, n_workers),
        )
        return

    tasks = (
        (t, p_idx, param, contour3[t], foreground3[t])
        for t in range(contour3.shape[0])
        for p_idx, param in enumerate(params)
    )
    yield from _ordered_bounded_map(
        _run_watershed_task,
        tasks,
        max_workers=max(1, n_workers),
    )


def iter_write_hypothesis_sweep_h5(
    path: str | Path,
    records: Iterable[HypothesisRecord],
    *,
    overwrite: bool = False,
    compression: str | None = "gzip",
) -> Iterator[int]:
    """Stream hypothesis records to HDF5 and yield the count written so far."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    count = 0
    with h5py.File(path, mode) as h5:
        h5.attrs["layout"] = "hypotheses/t{t:03d}/p{p:03d}/labels"
        root = h5.require_group(_ROOT_GROUP)
        for record in records:
            group = root.require_group(f"t{record.t:03d}").require_group(f"p{record.p:03d}")
            for key, value in _param_attrs(record.params).items():
                group.attrs[key] = value
            group.create_dataset(
                "labels",
                data=np.asarray(record.labels, dtype=_LABEL_DTYPE),
                compression=compression,
            )
            count += 1
            yield count


def _check_schema(h5: h5py.File, path: Path) -> None:
    """Raise if the file uses the old t/z/p layout from v1."""
    layout = h5.attrs.get("layout", "")
    if layout and "z{z" in str(layout):
        raise ValueError(
            f"{path} uses the v1 t/z/p schema and cannot be read by v2. "
            "Re-generate the hypothesis database."
        )

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

