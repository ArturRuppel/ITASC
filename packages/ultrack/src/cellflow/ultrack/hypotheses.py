"""HDF5 helpers for nucleus hypothesis sweeps.

The sweep layout is intentionally hierarchical and stable:

    /hypotheses/t000/z000/p000/labels
    /hypotheses/t000/z000/p001/labels
    /hypotheses/t001/z000/p000/labels

The `t` group is the outermost axis, followed by `z`, then `p` for the
parameter-set index.  Each `p` group stores the parameter metadata needed to
reconstruct the sweep without a sidecar manifest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

_LABEL_DTYPE = np.uint32
_ROOT_GROUP = "hypotheses"
_AXES = "TZP"


@dataclass(frozen=True, slots=True)
class NucleusHypothesisParams:
    """One parameter set within the sweep."""

    basin: str
    threshold_pct: float
    compactness: float
    smooth_sigma: float
    seed_source: str = "auto"
    seed_distance: int = 5
    min_size: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NucleusHypothesisSweepSpec:
    """Sweep definition for the nucleus hypothesis writer."""

    basins: tuple[str, ...] = ("prob", "flow_mag")
    threshold: float = 0.0
    threshold_min: float = 0.0
    threshold_max: float = 0.0
    threshold_step: float = 1.0
    compactness: float = 0.0
    compactness_min: float = 0.0
    compactness_max: float = 0.0
    compactness_step: float = 0.01
    smooth_sigma: float = 0.0
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
    """A single `(t, z, p)` labelmap payload."""

    t: int
    z: int
    p: int
    labels: np.ndarray
    params: NucleusHypothesisParams


def _values(current: float, minimum: float, maximum: float, step: float) -> list[float]:
    if step > 0 and minimum != maximum:
        lo = float(minimum)
        hi = float(maximum)
        vals = np.arange(lo, hi + step / 2.0, step, dtype=np.float64)
        return [float(v) for v in vals]
    return [float(current)]


def _int_values(current: int, minimum: int, maximum: int, step: int) -> list[int]:
    if step > 0 and minimum != maximum:
        vals = list(range(int(minimum), int(maximum) + 1, max(1, int(step))))
        return vals
    return [int(current)]


def build_parameter_sets(spec: NucleusHypothesisSweepSpec) -> list[NucleusHypothesisParams]:
    """Return a deterministic list of parameter sets to sweep."""
    if not spec.basins:
        raise ValueError("spec.basins must contain at least one basin source")

    threshold_vals = _values(spec.threshold, spec.threshold_min, spec.threshold_max, spec.threshold_step)
    compactness_vals = _values(spec.compactness, spec.compactness_min, spec.compactness_max, spec.compactness_step)
    smooth_vals = _values(spec.smooth_sigma, spec.smooth_min, spec.smooth_max, spec.smooth_step)
    seed_dist_vals = _int_values(spec.seed_distance, spec.seed_distance_min, spec.seed_distance_max, spec.seed_distance_step)

    params: list[NucleusHypothesisParams] = []
    for basin in spec.basins:
        for threshold_pct in threshold_vals:
            for compactness in compactness_vals:
                for smooth_sigma in smooth_vals:
                    for seed_distance in seed_dist_vals:
                        params.append(
                            NucleusHypothesisParams(
                                basin=str(basin),
                                threshold_pct=float(threshold_pct),
                                compactness=float(compactness),
                                smooth_sigma=float(smooth_sigma),
                                seed_source=spec.seed_source,
                                seed_distance=int(seed_distance),
                                min_size=int(spec.min_size),
                            )
                        )
    return params


def _time_slice(arr: np.ndarray, t: int, n_frames: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim <= 2:
        return arr
    if arr.ndim in (4, 5):
        return arr[t]
    if arr.ndim == 3 and n_frames > 1 and arr.shape[0] == n_frames:
        return arr[t]
    return arr


def _as_2d_slice(arr: np.ndarray, z: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if z < 0 or z >= arr.shape[0]:
            raise IndexError(f"z={z} out of range for shape {arr.shape}")
        return arr[z]
    raise ValueError(f"Expected a 2D or 3D array, got shape {arr.shape}")


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    scaled = (arr - lo) / (hi - lo)
    scaled = np.minimum(scaled, np.nextafter(np.float32(1.0), np.float32(0.0)))
    return scaled.astype(np.float32)


def _flow_magnitude(dp: np.ndarray) -> np.ndarray:
    """Compute an L2 magnitude image from a DP stack."""
    dp = np.asarray(dp, dtype=np.float32)

    if dp.ndim == 2:
        return np.abs(dp)

    if dp.ndim == 3:
        if dp.shape[0] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=0)).astype(np.float32)
        if dp.shape[-1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=-1)).astype(np.float32)
        return np.abs(dp).astype(np.float32)

    if dp.ndim == 4:
        if dp.shape[1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=1)).astype(np.float32)
        if dp.shape[-1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=-1)).astype(np.float32)

    if dp.ndim == 5:
        if dp.shape[1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=1)).astype(np.float32)
        if dp.shape[-1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=-1)).astype(np.float32)

    if dp.ndim >= 3 and dp.shape[0] in (2, 3):
        return np.sqrt(np.sum(dp * dp, axis=0)).astype(np.float32)

    if dp.ndim >= 3 and dp.shape[-1] in (2, 3):
        return np.sqrt(np.sum(dp * dp, axis=-1)).astype(np.float32)

    raise ValueError(f"Unsupported DP shape for magnitude computation: {dp.shape}")


def _remove_small_labels(labels: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return labels
    ids, counts = np.unique(labels, return_counts=True)
    small = ids[(ids > 0) & (counts < min_size)]
    if small.size == 0:
        return labels
    out = labels.copy()
    out[np.isin(labels, small)] = 0
    return out


def _peak_local_max_markers(basin: np.ndarray, min_distance: int) -> np.ndarray:
    from scipy.ndimage import label as nd_label
    from skimage.feature import peak_local_max

    coords = peak_local_max(basin, min_distance=max(1, min_distance), exclude_border=False)
    mask = np.zeros(basin.shape, dtype=bool)
    if coords.size:
        mask[coords[:, 0], coords[:, 1]] = True
    markers, _ = nd_label(mask)
    return markers.astype(np.int32)


def compute_hypothesis_labels(
    prob: np.ndarray,
    dp: np.ndarray | None,
    markers: np.ndarray | None,
    params: NucleusHypothesisParams,
) -> np.ndarray:
    """Compute a single nucleus hypothesis label image for one parameter set."""
    from skimage.segmentation import watershed

    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"Expected a 2D probability slice, got shape {prob.shape}")

    if params.basin == "prob":
        basin = prob
    elif params.basin == "flow_mag":
        if dp is None:
            raise ValueError("flow_mag basin requested but no DP stack was provided")
        basin = _flow_magnitude(dp)
        if basin.ndim != 2:
            raise ValueError(f"Expected a 2D flow magnitude slice, got shape {basin.shape}")
    else:
        raise ValueError(f"Unknown basin={params.basin!r}; expected 'prob' or 'flow_mag'")

    basin = _normalize_01(basin)
    if params.smooth_sigma > 0:
        basin = gaussian_filter(basin, sigma=float(params.smooth_sigma))
        basin = _normalize_01(basin)

    if markers is None:
        markers = _peak_local_max_markers(basin, params.seed_distance)
    else:
        markers = np.asarray(markers, dtype=np.int32)
        if markers.shape != basin.shape:
            raise ValueError(
                f"Markers shape {markers.shape} does not match basin shape {basin.shape}"
            )

    threshold = float(params.threshold_pct) / 100.0
    mask = (basin >= threshold) | (markers > 0)

    labels = watershed(
        -basin,
        markers=markers,
        mask=mask,
        compactness=float(params.compactness),
        watershed_line=False,
    )
    labels = np.asarray(labels, dtype=_LABEL_DTYPE)
    return _remove_small_labels(labels, params.min_size)


def iter_hypothesis_records_from_stacks(
    prob_stack: np.ndarray,
    dp_stack: np.ndarray | None,
    seed_stack: np.ndarray | None,
    spec: NucleusHypothesisSweepSpec,
) -> Iterator[HypothesisRecord]:
    """Yield hypothesis records for all `(t, z, p)` combinations in the sweep.

    When *seed_stack* is ``None`` the seeds are generated per-slice from the
    basin image using ``peak_local_max`` with each parameter set's
    ``seed_distance``.
    """
    prob_stack = np.asarray(prob_stack)
    dp_stack = None if dp_stack is None else np.asarray(dp_stack)
    seed_stack = None if seed_stack is None else np.asarray(seed_stack)

    n_frames = 1
    if prob_stack.ndim in (4, 5):
        n_frames = int(prob_stack.shape[0])
    elif dp_stack is not None and dp_stack.ndim in (4, 5):
        n_frames = int(dp_stack.shape[0])
    elif (
        prob_stack.ndim == 3
        and dp_stack is not None
        and dp_stack.ndim == 3
        and prob_stack.shape[0] == dp_stack.shape[0]
        and prob_stack.shape[0] > 1
    ):
        n_frames = int(prob_stack.shape[0])

    params = build_parameter_sets(spec)
    if not params:
        return

    sample_prob = _time_slice(prob_stack, 0, n_frames)
    n_z = sample_prob.shape[0] if sample_prob.ndim == 3 else 1

    for t in range(n_frames):
        prob_t = _time_slice(prob_stack, t, n_frames)
        dp_t = _time_slice(dp_stack, t, n_frames) if dp_stack is not None else None
        seed_t = _time_slice(seed_stack, t, n_frames) if seed_stack is not None else None
        flow_t = _flow_magnitude(dp_t) if dp_t is not None else None

        for z in range(n_z):
            prob_2d = _as_2d_slice(prob_t, z) if prob_t.ndim == 3 else prob_t
            flow_2d = _as_2d_slice(flow_t, z) if flow_t is not None and flow_t.ndim == 3 else flow_t
            seed_2d = (_as_2d_slice(seed_t, z) if seed_t.ndim == 3 else seed_t) if seed_t is not None else None

            for p_idx, params_i in enumerate(params):
                labels = compute_hypothesis_labels(prob_2d, flow_2d, seed_2d, params_i)
                yield HypothesisRecord(
                    t=t,
                    z=z,
                    p=p_idx,
                    labels=labels,
                    params=params_i,
                )


def _group_name(prefix: str, index: int) -> str:
    return f"{prefix}{index:03d}"


def _ensure_root_metadata(
    h5: h5py.File,
    *,
    stage: str = "nucleus_hypotheses",
    source: str | None = None,
    n_t: int | None = None,
    n_z: int | None = None,
    n_p: int | None = None,
    layout: str = "hypotheses/t{t:03d}/z{z:03d}/p{p:03d}/labels",
) -> None:
    attrs = h5.attrs
    if "version" not in attrs:
        attrs["version"] = 1
    if "stage" not in attrs:
        attrs["stage"] = stage
    if source is not None and "source" not in attrs:
        attrs["source"] = source
    if "axes" not in attrs:
        attrs["axes"] = _AXES
    if "layout" not in attrs:
        attrs["layout"] = layout
    if n_t is not None and "n_t" not in attrs:
        attrs["n_t"] = int(n_t)
    if n_z is not None and "n_z" not in attrs:
        attrs["n_z"] = int(n_z)
    if n_p is not None and "n_p" not in attrs:
        attrs["n_p"] = int(n_p)


def write_hypothesis_record(
    h5: h5py.File,
    record: HypothesisRecord,
) -> None:
    """Write one hypothesis record into the `t/z/p` hierarchy."""
    root = h5.require_group(_ROOT_GROUP)
    t_grp = root.require_group(_group_name("t", record.t))
    z_grp = t_grp.require_group(_group_name("z", record.z))
    p_grp = z_grp.require_group(_group_name("p", record.p))

    labels = np.asarray(record.labels, dtype=_LABEL_DTYPE)
    if "labels" in p_grp:
        del p_grp["labels"]
    p_grp.create_dataset(
        "labels",
        data=labels,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )

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
    stage: str = "nucleus_hypotheses",
    source: str | None = None,
    n_t: int | None = None,
    n_z: int | None = None,
    n_p: int | None = None,
) -> Path:
    """Write a full hypothesis sweep into a single HDF5 file.

    The caller is expected to provide records in deterministic `(t, z, p)` order.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite or not path.exists() else "a"

    with h5py.File(path, mode) as h5:
        _ensure_root_metadata(h5, stage=stage, source=source, n_t=n_t, n_z=n_z, n_p=n_p)
        for record in records:
            write_hypothesis_record(h5, record)
    return path


def iter_hypothesis_records(path: str | Path) -> Iterator[HypothesisRecord]:
    """Yield hypothesis records from a sweep HDF5 file in sorted order."""
    with h5py.File(Path(path), "r") as h5:
        root = h5[_ROOT_GROUP]
        for t_name in sorted(root.keys()):
            if not t_name.startswith("t"):
                continue
            t_idx = int(t_name[1:])
            t_grp = root[t_name]
            for z_name in sorted(t_grp.keys()):
                if not z_name.startswith("z"):
                    continue
                z_idx = int(z_name[1:])
                z_grp = t_grp[z_name]
                for p_name in sorted(z_grp.keys()):
                    if not p_name.startswith("p"):
                        continue
                    p_idx = int(p_name[1:])
                    p_grp = z_grp[p_name]
                    labels = np.asarray(p_grp["labels"][:], dtype=_LABEL_DTYPE)
                    params = NucleusHypothesisParams(
                        basin=str(p_grp.attrs["basin"]),
                        threshold_pct=float(p_grp.attrs["threshold_pct"]),
                        compactness=float(p_grp.attrs["compactness"]),
                        smooth_sigma=float(p_grp.attrs["smooth_sigma"]),
                        seed_source=str(p_grp.attrs.get("seed_source", "auto")),
                        seed_distance=int(p_grp.attrs.get("seed_distance", 5)),
                        min_size=int(p_grp.attrs.get("min_size", 0)),
                    )
                    yield HypothesisRecord(
                        t=t_idx,
                        z=z_idx,
                        p=p_idx,
                        labels=labels,
                        params=params,
                    )


def _read_hypothesis_labels(path: str, t: int, z: int, p: int) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        dataset = h5[f"hypotheses/t{t:03d}/z{z:03d}/p{p:03d}/labels"]
        return np.asarray(dataset[:], dtype=np.uint32)


def compute_medoid_stack(records: list[HypothesisRecord], n_t: int) -> np.ndarray:
    """Return the medoid label image for each t across all (z, p) combinations.

    Flattens z and p into a single pool per t, then finds the image closest to
    the element-wise median (approximate medoid via L1 distance).

    Returns shape ``(y, x, t)`` — time axis last.
    """
    by_t: dict[int, list[np.ndarray]] = {}
    for rec in records:
        by_t.setdefault(rec.t, []).append(rec.labels)

    spatial_shape = records[0].labels.shape  # (y, x)
    medoid_stack = np.zeros((*spatial_shape, n_t), dtype=_LABEL_DTYPE)

    for t in range(n_t):
        imgs = by_t.get(t, [])
        if not imgs:
            continue
        if len(imgs) == 1:
            medoid_stack[..., t] = imgs[0]
            continue
        mat = np.stack([img.ravel().astype(np.float32) for img in imgs], axis=0)
        median_vec = np.median(mat, axis=0)
        dists = np.sum(np.abs(mat - median_vec[np.newaxis, :]), axis=1)
        medoid_idx = int(np.argmin(dists))
        medoid_stack[..., t] = imgs[medoid_idx]

    return medoid_stack


def write_medoid_stack(h5: h5py.File, medoid_stack: np.ndarray) -> None:
    """Write medoid stack with shape ``(y, x, t)`` to the HDF5 root."""
    if "medoid_stack" in h5:
        del h5["medoid_stack"]
    ds = h5.create_dataset(
        "medoid_stack",
        data=np.asarray(medoid_stack, dtype=_LABEL_DTYPE),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )
    ds.attrs["axes"] = "YXT"


def load_medoid_stack(path: str | Path) -> np.ndarray:
    """Load the medoid stack ``(y, x, t)`` from a hypothesis HDF5 file."""
    with h5py.File(Path(path), "r") as h5:
        return np.asarray(h5["medoid_stack"][:])


def load_hypotheses_h5_lazy(path: str | Path):
    """Return a lazily-loaded dask array with axes ``(t, z, param, y, x)``."""
    try:
        import dask.array as da
        from dask import delayed
    except Exception as exc:
        raise RuntimeError("dask is required to lazily load hypotheses.h5") from exc

    path = Path(path)
    with h5py.File(path, "r") as h5:
        root = h5["hypotheses"]
        t_keys = [k for k in root.keys() if k.startswith("t")]
        n_t = int(h5.attrs.get("n_t", len(t_keys)))
        n_z = int(h5.attrs.get("n_z", 1))
        n_p = int(h5.attrs.get("n_p", 1))

        first = None
        for t_name in sorted(t_keys):
            for z_name in sorted(k for k in root[t_name].keys() if k.startswith("z")):
                for p_name in sorted(k for k in root[t_name][z_name].keys() if k.startswith("p")):
                    first = np.asarray(root[t_name][z_name][p_name]["labels"])
                    break
                if first is not None:
                    break
            if first is not None:
                break

        if first is None:
            raise ValueError(f"No labels datasets found in {path}")

        spatial_shape = tuple(first.shape)
        if n_z == 1 and t_keys:
            first_z_keys = [k for k in root[t_keys[0]].keys() if k.startswith("z")]
            n_z = len(first_z_keys) or n_z
            if first_z_keys:
                first_p_keys = [k for k in root[t_keys[0]][first_z_keys[0]].keys() if k.startswith("p")]
                n_p = len(first_p_keys) or n_p

    rows = []
    for t in range(n_t):
        z_rows = []
        for z in range(n_z):
            p_rows = []
            for p in range(n_p):
                delayed_arr = delayed(_read_hypothesis_labels)(str(path), t, z, p)
                p_rows.append(da.from_delayed(delayed_arr, shape=spatial_shape, dtype=np.uint32))
            z_rows.append(da.stack(p_rows, axis=0))
        rows.append(da.stack(z_rows, axis=0))

    return da.stack(rows, axis=0)
