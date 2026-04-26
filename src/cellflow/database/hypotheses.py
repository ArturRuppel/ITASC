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

from cellflow.segmentation import (
    CellposeFlowHypothesisParams,
    ContourWatershedParams,
    NucleusHypothesisParams,
    build_consensus_boundary,
    compute_cellpose_flow_hypothesis,
    compute_contour_watershed,
    compute_hypothesis_labels,
)

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
    min_circularity: float = 0.0
    z_slice: int = 0
    z_slice_min: int = 0
    z_slice_max: int = 0
    z_slice_step: int = 1


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    """A single (t, p) label volume payload."""

    t: int
    p: int
    labels: np.ndarray  # shape (Z, Y, X), dtype uint32
    params: NucleusHypothesisParams | CellposeFlowHypothesisParams


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
    z_slice_vals = _int_values(spec.z_slice, spec.z_slice_min, spec.z_slice_max, spec.z_slice_step)

    params: list[NucleusHypothesisParams] = []
    for z_slice in z_slice_vals:
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
                            min_circularity=float(spec.min_circularity),
                            z_slice=int(z_slice),
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


def _read_append_state(path: Path) -> tuple[int, set[str]]:
    """Return (next_p, existing_param_jsons) from an existing HDF5 file."""
    with h5py.File(path, "r") as h5:
        if _ROOT_GROUP not in h5:
            return 0, set()
        root = h5[_ROOT_GROUP]
        t_keys = [k for k in root.keys() if k.startswith("t")]
        if not t_keys:
            return 0, set()
        first_t = root[t_keys[0]]
        p_keys = [k for k in first_t.keys() if k.startswith("p")]
        if not p_keys:
            return 0, set()
        p_indices = [int(k[1:]) for k in p_keys]
        existing_jsons = {
            first_t[pk].attrs["parameter_json"]
            for pk in p_keys
            if "parameter_json" in first_t[pk].attrs
        }
        return max(p_indices) + 1, existing_jsons


def write_hypothesis_sweep_h5(
    output_path: str | Path,
    records: Iterable[HypothesisRecord],
    *,
    overwrite: bool = True,
    n_t: int | None = None,
    n_p: int | None = None,
) -> Path:
    """Write a full hypothesis sweep to a single HDF5 file.

    When overwrite=False and the file exists, records whose parameter set is
    already present are skipped; new parameter sets are appended after the
    highest existing p index.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    appending = not overwrite and path.exists()
    if appending:
        next_p, existing_jsons = _read_append_state(path)
    else:
        next_p, existing_jsons = 0, set()
    # Maps param_json -> assigned p index for param sets added in this call.
    new_param_p: dict[str, int] = {}
    mode = "w" if not appending else "a"
    with h5py.File(path, mode) as h5:
        _write_root_metadata(h5, n_t=n_t, n_p=n_p)
        for record in records:
            param_json = json.dumps(record.params.to_dict(), sort_keys=True)
            if param_json in existing_jsons:
                continue
            if param_json not in new_param_p:
                new_param_p[param_json] = next_p
                next_p += 1
            shifted = HypothesisRecord(t=record.t, p=new_param_p[param_json], labels=record.labels, params=record.params)
            write_hypothesis_record(h5, shifted)
    return path


def read_hypothesis_labels(path: str | Path, t: int, p: int) -> np.ndarray:
    """Read the (Z, Y, X) label volume for one (t, p) entry."""
    with h5py.File(Path(path), "r") as h5:
        _check_schema(h5, Path(path))
        return np.asarray(h5[f"{_ROOT_GROUP}/t{t:03d}/p{p:03d}/labels"], dtype=_LABEL_DTYPE)


def read_full_hypothesis_stack(path: str | Path, p: int) -> np.ndarray:
    """Read the (T, Z, Y, X) label volume for one parameter choice p across all timepoints."""
    with h5py.File(Path(path), "r") as h5:
        _check_schema(h5, Path(path))
        root = h5[_ROOT_GROUP]
        t_keys = sorted(k for k in root.keys() if k.startswith("t"))
        if not t_keys:
            return np.empty((0, 0, 0, 0), dtype=_LABEL_DTYPE)

        # Get shape from first available t for parameter p
        first_t = root[t_keys[0]]
        p_name = f"p{p:03d}"
        if p_name not in first_t:
            raise ValueError(f"Parameter index {p_name} not found in {path}")

        ds = first_t[p_name]["labels"]
        z, y, x = ds.shape
        n_t = len(t_keys)

        stack = np.empty((n_t, z, y, x), dtype=_LABEL_DTYPE)
        for i, t_key in enumerate(t_keys):
            stack[i] = root[f"{t_key}/{p_name}/labels"][:]
        return stack


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


def zero_hypothesis_slice(path: str | Path, z: int, p: int) -> None:
    """Zero out z-plane z across all timepoints for parameter p, keeping array shape intact."""
    with h5py.File(Path(path), "r+") as h5:
        _check_schema(h5, Path(path))
        root = h5[_ROOT_GROUP]
        p_name = f"p{p:03d}"
        for t_key in sorted(k for k in root.keys() if k.startswith("t")):
            ds_key = f"{_ROOT_GROUP}/{t_key}/{p_name}/labels"
            if ds_key not in h5:
                continue
            data = h5[ds_key][:]
            if z >= data.shape[0]:
                continue
            data[z] = 0
            p_grp = root[t_key][p_name]
            del p_grp["labels"]
            p_grp.create_dataset("labels", data=data, compression="gzip", compression_opts=4, shuffle=True)


def delete_hypothesis_parameter(path: str | Path, p: int) -> None:
    """Remove every t-group's entry for parameter p from the hypothesis database."""
    with h5py.File(Path(path), "r+") as h5:
        _check_schema(h5, Path(path))
        root = h5[_ROOT_GROUP]
        p_name = f"p{p:03d}"
        for t_key in sorted(k for k in root.keys() if k.startswith("t")):
            t_grp = root[t_key]
            if p_name in t_grp:
                del t_grp[p_name]
            if len(t_grp) == 0:
                del root[t_key]
        remaining_t = [k for k in root.keys() if k.startswith("t")]
        if remaining_t:
            n_p = len([k for k in root[remaining_t[0]].keys() if k.startswith("p")])
        else:
            n_p = 0
        h5.attrs["n_p"] = n_p


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
                method = str(p_grp.attrs.get("method", "watershed"))
                if method == "cellpose_flow":
                    params: NucleusHypothesisParams | CellposeFlowHypothesisParams | ContourWatershedParams = CellposeFlowHypothesisParams(
                        cellprob_threshold=float(p_grp.attrs.get("cellprob_threshold", 0.0)),
                        flow_threshold=float(p_grp.attrs.get("flow_threshold", 0.4)),
                        min_size=int(p_grp.attrs.get("min_size", 15)),
                        niter=int(p_grp.attrs.get("niter", 200)),
                    )
                elif method == "contour_watershed":
                    params = ContourWatershedParams(
                        seed_distance=int(p_grp.attrs.get("seed_distance", 10)),
                        foreground_threshold=float(p_grp.attrs.get("foreground_threshold", 0.5)),
                        min_size=int(p_grp.attrs.get("min_size", 0)),
                        min_circularity=float(p_grp.attrs.get("min_circularity", 0.0)),
                        noise_scale=float(p_grp.attrs.get("noise_scale", 0.0)),
                        noise_blur_sigma=float(p_grp.attrs.get("noise_blur_sigma", 0.0)),
                        run_index=int(p_grp.attrs.get("run_index", 0)),
                    )
                else:
                    params = NucleusHypothesisParams(
                        basin=str(p_grp.attrs.get("basin", "prob")),
                        threshold_pct=float(p_grp.attrs.get("threshold_pct", 30.0)),
                        compactness=float(p_grp.attrs.get("compactness", 0.0)),
                        smooth_sigma=float(p_grp.attrs.get("smooth_sigma", 0.5)),
                        seed_source=str(p_grp.attrs.get("seed_source", "auto")),
                        seed_distance=int(p_grp.attrs.get("seed_distance", 5)),
                        min_size=int(p_grp.attrs.get("min_size", 0)),
                        min_circularity=float(p_grp.attrs.get("min_circularity", 0.0)),
                        z_slice=int(p_grp.attrs.get("z_slice", 0)),
                    )
                yield HypothesisRecord(t=t_idx, p=p_idx, labels=labels, params=params)


def iter_cellpose_flow_records_from_stacks(
    prob_stack: np.ndarray,
    dp_stack: np.ndarray,
    params: CellposeFlowHypothesisParams,
) -> Iterator[HypothesisRecord]:
    """Yield one HypothesisRecord per timepoint using Cellpose native flow segmentation.

    prob_stack: (T, Z, Y, X) logits
    dp_stack:   (T, Z, 2, Y, X) flow fields
    Each z-slice is segmented independently. p is set to 0; write_hypothesis_sweep_h5
    assigns the actual index during deduplication.
    """
    prob_stack = np.asarray(prob_stack)
    dp_stack = np.asarray(dp_stack)
    if prob_stack.ndim == 3:
        prob_stack = prob_stack[np.newaxis]
    if dp_stack.ndim == 4:
        dp_stack = dp_stack[np.newaxis]
    n_t = prob_stack.shape[0]
    for t in range(n_t):
        labels_3d = compute_cellpose_flow_hypothesis(prob_stack[t], dp_stack[t], params)
        yield HypothesisRecord(t=t, p=0, labels=labels_3d, params=params)


def iter_hypothesis_records_from_stacks(
    prob_stack: np.ndarray,
    dp_stack: np.ndarray | None,
    seed_stack: np.ndarray | None,
    spec: NucleusHypothesisSweepSpec,
    params_list: list | None = None,
) -> Iterator[HypothesisRecord]:
    """Compute and yield HypothesisRecords from in-memory stacks.

    prob_stack shape: (T, Z, Y, X) or (Z, Y, X) for single-frame.
    Yields one record per (t, p) with labels shape (Z, Y, X).

    Pass params_list to skip building it from spec (e.g. to compute only a
    filtered subset without running build_parameter_sets again).
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

    if params_list is None:
        params_list = build_parameter_sets(spec)
    if not params_list:
        return

    for t in range(n_t):
        prob_t = prob_stack[t]  # (Z, Y, X)
        dp_t = dp_stack[t] if dp_stack is not None else None
        seed_t = seed_stack[t] if seed_stack is not None else None

        # Global normalization reference from all z-slices so threshold_pct is
        # consistent across slices rather than relative to each slice's own range.
        basin_3d = 1.0 / (1.0 + np.exp(-prob_t.astype(np.float32)))
        global_lo = float(np.min(basin_3d))
        global_hi = float(np.max(basin_3d))

        for p_idx, params in enumerate(params_list):
            z = min(params.z_slice, n_z - 1)
            prob_2d = prob_t[z]
            dp_2d = dp_t[z] if dp_t is not None else None
            seed_2d = seed_t[z] if seed_t is not None else None
            g_lo = global_lo if params.basin == "prob" else None
            g_hi = global_hi if params.basin == "prob" else None
            labels_2d = compute_hypothesis_labels(prob_2d, dp_2d, seed_2d, params, global_lo=g_lo, global_hi=g_hi)
            labels_3d = labels_2d[np.newaxis]  # (1, Y, X)
            yield HypothesisRecord(t=t, p=p_idx, labels=labels_3d, params=params)


@dataclass(frozen=True, slots=True)
class ContourWatershedSweepSpec:
    """Parameter sweep spec for contour-map watershed hypothesis generation."""

    seed_distance: int = 10
    seed_distance_min: int = 10
    seed_distance_max: int = 10
    seed_distance_step: int = 2
    foreground_threshold: float = 0.5
    foreground_threshold_min: float = 0.5
    foreground_threshold_max: float = 0.5
    foreground_threshold_step: float = 0.05
    min_size: int = 0
    min_circularity: float = 0.0
    noise_scale: float = 0.0
    noise_blur_sigma: float = 0.0
    n_runs: int = 1


def build_contour_watershed_parameter_sets(spec: ContourWatershedSweepSpec) -> list[ContourWatershedParams]:
    """Return the deterministic list of ContourWatershedParams for this sweep spec.

    With n_runs > 1 and noise_scale > 0, each run gets a unique run_index so
    stochastic repetitions are stored as distinct hypotheses in the DB.
    """
    seed_dist_vals = _int_values(spec.seed_distance, spec.seed_distance_min, spec.seed_distance_max, spec.seed_distance_step)
    fg_vals        = _values(spec.foreground_threshold, spec.foreground_threshold_min, spec.foreground_threshold_max, spec.foreground_threshold_step)
    return [
        ContourWatershedParams(
            seed_distance=int(d),
            foreground_threshold=float(fg),
            min_size=int(spec.min_size),
            min_circularity=float(spec.min_circularity),
            noise_scale=float(spec.noise_scale),
            noise_blur_sigma=float(spec.noise_blur_sigma),
            run_index=run_idx,
        )
        for run_idx in range(max(1, spec.n_runs))
        for d in seed_dist_vals
        for fg in fg_vals
    ]


def _run_watershed_task(
    args: tuple[int, int, "ContourWatershedParams", np.ndarray, np.ndarray],
) -> "HypothesisRecord":
    t, p_idx, params, contour_frame, fg_frame = args
    labels_2d = compute_contour_watershed(contour_frame, fg_frame, params)
    return HypothesisRecord(t=t, p=p_idx, labels=labels_2d[np.newaxis], params=params)


def iter_contour_watershed_records(
    contour_stack: np.ndarray,
    foreground_stack: np.ndarray,
    spec: ContourWatershedSweepSpec,
    n_workers: int = 1,
) -> Iterator[HypothesisRecord]:
    """Yield HypothesisRecords from cached contour and foreground maps (single gamma).

    contour_stack:    (T, Y, X) float32 consensus boundary maps
    foreground_stack: (T, Y, X) float32 foreground probability maps
    n_workers:        number of threads (scipy/skimage release the GIL)
    """
    params_list = build_contour_watershed_parameter_sets(spec)
    n_t = contour_stack.shape[0]
    args = [
        (t, p_idx, params, contour_stack[t], foreground_stack[t])
        for t in range(n_t)
        for p_idx, params in enumerate(params_list)
    ]
    if n_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            yield from executor.map(_run_watershed_task, args)
    else:
        for a in args:
            yield _run_watershed_task(a)


def iter_contour_watershed_records_from_raw(
    prob_stack: np.ndarray,
    dp_stack: np.ndarray,
    cellprob_thresholds: list[float],
    spec: ContourWatershedSweepSpec,
    gamma: float = 1.0,
) -> Iterator[HypothesisRecord]:
    """Build contour maps from raw prob/dp and yield HypothesisRecords.

    prob_stack: (T, Z, Y, X)  dp_stack: (T, Z, 2, Y, X)
    Contour maps are built once per frame; gamma is passed through to build_consensus_boundary.
    """
    params_list = build_contour_watershed_parameter_sets(spec)
    if not params_list:
        return

    prob_stack = np.asarray(prob_stack, dtype=np.float32)
    dp_stack   = np.asarray(dp_stack,   dtype=np.float32)
    if prob_stack.ndim == 3:
        prob_stack = prob_stack[np.newaxis]
    if dp_stack.ndim == 4:
        dp_stack = dp_stack[np.newaxis]
    n_t = prob_stack.shape[0]

    for t in range(n_t):
        boundary, foreground = build_consensus_boundary(
            prob_stack[t], dp_stack[t], cellprob_thresholds, gamma=gamma
        )
        for p_idx, params in enumerate(params_list):
            labels_2d = compute_contour_watershed(boundary, foreground, params)
            yield HypothesisRecord(t=t, p=p_idx, labels=labels_2d[np.newaxis], params=params)
