"""HDF5 hypothesis pool for nucleus segmentation candidates.

Schema: hypotheses/t{t:03d}/p{p:03d}/labels
Each labels dataset has shape (Z, Y, X) and dtype uint32.
Parameters are stored as group attributes on each p group.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import h5py
import numpy as np

from cellflow.segmentation import NucleusHypothesisParams, compute_hypothesis_labels

_LABEL_DTYPE = np.uint32
_ROOT_GROUP = "hypotheses"
_SCHEMA_VERSION = 2
_LAYOUT = "hypotheses/t{t:03d}/p{p:03d}/labels"


@dataclass(frozen=True, slots=True)
class NucleusHypothesisSweepSpec:
    """Defines the parameter space for a hypothesis sweep."""

    threshold: float = 30.0
    threshold_min: float = 0.0
    threshold_max: float = 0.0
    threshold_step: float = 1.0
    compactness: float = 0.0
    compactness_min: float = 0.0
    compactness_max: float = 0.0
    compactness_step: float = 0.01
    smooth_sigma: float = 0.5
    smooth_min: float = 0.0
    smooth_max: float = 0.0
    smooth_step: float = 0.25
    seed_source: str = "auto"
    seed_distance: int = 5
    seed_distance_min: int = 5
    seed_distance_max: int = 5
    seed_distance_step: int = 1
    min_size: int = 0


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    """A single (t, p) label volume payload."""

    t: int
    p: int
    labels: np.ndarray  # shape (Z, Y, X), dtype uint32
    params: NucleusHypothesisParams


def _values(current: float, minimum: float, maximum: float, step: float) -> list[float]:
    if step > 0 and minimum != maximum:
        vals = np.arange(float(minimum), float(maximum) + step / 2.0, step, dtype=np.float64)
        return [float(v) for v in vals]
    return [float(current)]


def _int_values(current: int, minimum: int, maximum: int, step: int) -> list[int]:
    if step > 0 and minimum != maximum:
        return list(range(int(minimum), int(maximum) + 1, max(1, int(step))))
    return [int(current)]


def build_parameter_sets(spec: NucleusHypothesisSweepSpec) -> list[NucleusHypothesisParams]:
    """Return the deterministic list of parameter sets for this sweep spec."""
    threshold_vals = _values(spec.threshold, spec.threshold_min, spec.threshold_max, spec.threshold_step)
    compactness_vals = _values(spec.compactness, spec.compactness_min, spec.compactness_max, spec.compactness_step)
    smooth_vals = _values(spec.smooth_sigma, spec.smooth_min, spec.smooth_max, spec.smooth_step)
    seed_dist_vals = _int_values(spec.seed_distance, spec.seed_distance_min, spec.seed_distance_max, spec.seed_distance_step)

    params: list[NucleusHypothesisParams] = []
    for threshold_pct in threshold_vals:
        for compactness in compactness_vals:
            for smooth_sigma in smooth_vals:
                for seed_distance in seed_dist_vals:
                    params.append(NucleusHypothesisParams(
                        basin="prob",
                        threshold_pct=float(threshold_pct),
                        compactness=float(compactness),
                        smooth_sigma=float(smooth_sigma),
                        seed_source=spec.seed_source,
                        seed_distance=int(seed_distance),
                        min_size=int(spec.min_size),
                    ))
    return params


def _check_schema(h5: h5py.File, path: Path) -> None:
    """Raise if the file uses the old t/z/p layout from v1."""
    layout = h5.attrs.get("layout", "")
    if layout and "z{z" in str(layout):
        raise ValueError(
            f"{path} uses the v1 t/z/p schema and cannot be read by v2. "
            "Re-generate the hypothesis database."
        )


def _write_root_metadata(h5: h5py.File, *, n_t: int | None, n_p: int | None) -> None:
    attrs = h5.attrs
    attrs.setdefault("version", _SCHEMA_VERSION)
    attrs.setdefault("stage", "nucleus_hypotheses")
    attrs.setdefault("layout", _LAYOUT)
    if n_t is not None:
        attrs["n_t"] = int(n_t)
    if n_p is not None:
        attrs["n_p"] = int(n_p)


def write_hypothesis_record(h5: h5py.File, record: HypothesisRecord) -> None:
    """Write one (t, p) record into the open HDF5 file."""
    root = h5.require_group(_ROOT_GROUP)
    t_grp = root.require_group(f"t{record.t:03d}")
    p_grp = t_grp.require_group(f"p{record.p:03d}")

    labels = np.asarray(record.labels, dtype=_LABEL_DTYPE)
    if "labels" in p_grp:
        del p_grp["labels"]
    p_grp.create_dataset("labels", data=labels, compression="gzip", compression_opts=4, shuffle=True)

    params = record.params.to_dict()
    p_grp.attrs["parameter_index"] = int(record.p)
    p_grp.attrs["parameter_json"] = json.dumps(params, sort_keys=True)
    for key, value in params.items():
        p_grp.attrs[key] = value
    p_grp.attrs["label_shape"] = np.asarray(labels.shape, dtype=np.int64)
    p_grp.attrs["label_dtype"] = str(labels.dtype)


def write_hypothesis_sweep_h5(
    output_path: str | Path,
    records: Iterable[HypothesisRecord],
    *,
    overwrite: bool = True,
    n_t: int | None = None,
    n_p: int | None = None,
) -> Path:
    """Write a full hypothesis sweep to a single HDF5 file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite or not path.exists() else "a"
    with h5py.File(path, mode) as h5:
        _write_root_metadata(h5, n_t=n_t, n_p=n_p)
        for record in records:
            write_hypothesis_record(h5, record)
    return path


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


def iter_hypothesis_records(path: str | Path) -> Iterator[HypothesisRecord]:
    """Yield all (t, p) records from a hypotheses HDF5 file in sorted order."""
    with h5py.File(Path(path), "r") as h5:
        _check_schema(h5, Path(path))
        root = h5[_ROOT_GROUP]
        for t_name in sorted(k for k in root.keys() if k.startswith("t")):
            t_idx = int(t_name[1:])
            t_grp = root[t_name]
            for p_name in sorted(k for k in t_grp.keys() if k.startswith("p")):
                p_idx = int(p_name[1:])
                p_grp = t_grp[p_name]
                labels = np.asarray(p_grp["labels"][:], dtype=_LABEL_DTYPE)
                params = NucleusHypothesisParams(
                    basin=str(p_grp.attrs.get("basin", "prob")),
                    threshold_pct=float(p_grp.attrs["threshold_pct"]),
                    compactness=float(p_grp.attrs["compactness"]),
                    smooth_sigma=float(p_grp.attrs["smooth_sigma"]),
                    seed_source=str(p_grp.attrs.get("seed_source", "auto")),
                    seed_distance=int(p_grp.attrs.get("seed_distance", 5)),
                    min_size=int(p_grp.attrs.get("min_size", 0)),
                )
                yield HypothesisRecord(t=t_idx, p=p_idx, labels=labels, params=params)


def iter_hypothesis_records_from_stacks(
    prob_stack: np.ndarray,
    dp_stack: np.ndarray | None,
    seed_stack: np.ndarray | None,
    spec: NucleusHypothesisSweepSpec,
) -> Iterator[HypothesisRecord]:
    """Compute and yield HypothesisRecords from in-memory stacks.

    prob_stack shape: (T, Z, Y, X) or (Z, Y, X) for single-frame.
    Yields one record per (t, p) with labels shape (Z, Y, X).
    """
    prob_stack = np.asarray(prob_stack)
    if prob_stack.ndim == 3:
        # Single frame — add T axis
        prob_stack = prob_stack[np.newaxis]
    if prob_stack.ndim != 4:
        raise ValueError(f"Expected (T, Z, Y, X) or (Z, Y, X), got shape {prob_stack.shape}")

    n_t, n_z = prob_stack.shape[0], prob_stack.shape[1]

    dp_stack = None if dp_stack is None else np.asarray(dp_stack)
    seed_stack = None if seed_stack is None else np.asarray(seed_stack)

    params_list = build_parameter_sets(spec)
    if not params_list:
        return

    for t in range(n_t):
        prob_t = prob_stack[t]  # (Z, Y, X)
        dp_t = dp_stack[t] if dp_stack is not None else None
        seed_t = seed_stack[t] if seed_stack is not None else None

        for p_idx, params in enumerate(params_list):
            z_slices: list[np.ndarray] = []
            for z in range(n_z):
                prob_2d = prob_t[z]
                dp_2d = dp_t[z] if dp_t is not None else None
                seed_2d = seed_t[z] if seed_t is not None else None
                labels_2d = compute_hypothesis_labels(prob_2d, dp_2d, seed_2d, params)
                z_slices.append(labels_2d)

            labels_3d = np.stack(z_slices, axis=0)  # (Z, Y, X)
            yield HypothesisRecord(t=t, p=p_idx, labels=labels_3d, params=params)
