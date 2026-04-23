"""Save and load CellFlow projects to/from a single HDF5 file.

A project file (.h5) bundles:

* ``/labels``    — segmentation label stack (T, H, W), optional
* ``/metadata``  — pixel_size, time_interval, condition (group attributes)
* ``/analysis``  — TissueGraphDataset (per-tissue frames, trajectories, T1s)

Usage::

    from cellflow.utils.io import save_project, load_project

    save_project("experiment.h5", labels=label_stack, dataset=ds,
                 pixel_size=0.13, time_interval=5.0, condition="WT")

    result = load_project("experiment.h5")
    labels   = result["labels"]    # numpy array or None
    dataset  = result["dataset"]   # TissueGraphDataset or None
    px       = result["pixel_size"]
    dt       = result["time_interval"]
    cond     = result["condition"]
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import networkx as nx

from .structures import (
    CellData,
    EdgeTrajectory,
    InputType,
    JunctionData,
    T1Event,
    TissueGraphDataset,
    TissueGraphFrame,
    TissueGraphTimeSeries,
)

logger = logging.getLogger(__name__)

_FORMAT_VERSION = "2.0"
_MANIFEST_VERSION = "1.0"
_CATALOG_VERSION = "3.0"
_DATASET_VERSION = "3.0"
_STR_DTYPE = h5py.string_dtype()


# ------------------------------------------------------------------
# Multi-file project manifest (legacy — kept for backward compat)
# ------------------------------------------------------------------

@dataclass
class ProjectEntry:
    """One .h5 file within a project manifest."""

    path: str
    display_name: str = ""
    note: str = ""


@dataclass
class ProjectManifest:
    """Multi-file project: an ordered list of .h5 files with one active."""

    entries: List[ProjectEntry] = field(default_factory=list)
    active_index: int = 0

    @property
    def active_entry(self) -> Optional[ProjectEntry]:
        if 0 <= self.active_index < len(self.entries):
            return self.entries[self.active_index]
        return None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _set_opt_attr(group, name: str, value) -> None:
    """Write an attribute only if value is not None."""
    if value is not None:
        group.attrs[name] = value


def _compute_label_stats(
    frame: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ids, sizes, centroids) for all non-zero labels in a 2-D frame.

    Returns
    -------
    ids : int64 array, shape (N,)
    sizes : int64 array, shape (N,)  — pixel count per label
    centroids : float64 array, shape (N, 2) — (row, col) centroid per label
    """
    from skimage.measure import regionprops

    props = regionprops(frame.astype(np.int32, copy=False))
    if not props:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.float64),
        )
    ids = np.array([p.label for p in props], dtype=np.int64)
    sizes = np.array([p.area for p in props], dtype=np.int64)
    centroids = np.array([p.centroid for p in props], dtype=np.float64)
    return ids, sizes, centroids


def _save_label_metadata(f: h5py.File, key: str, arr: np.ndarray) -> None:
    """Write per-frame label stats (ids, sizes, centroids) under ``/key``."""
    grp = f.create_group(key)
    for t in range(arr.shape[0]):
        ids, sizes, centroids = _compute_label_stats(arr[t])
        fg = grp.create_group(str(t))
        fg.create_dataset("ids", data=ids)
        fg.create_dataset("sizes", data=sizes)
        fg.create_dataset("centroids", data=centroids)


def _get_opt_attr(group, name: str, default=None):
    return group.attrs[name] if name in group.attrs else default


def _serialize_ragged(
    arrays: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten a list of variable-length arrays into ``(flat, offsets)``.

    Each input array is cast to ``float64`` and reshaped to 2-D
    (rows × cols).  The returned *flat* array is the row-wise concatenation;
    *offsets* is an ``int64`` array of length ``len(arrays) + 1`` where
    ``flat[offsets[i]:offsets[i+1]]`` reconstructs ``arrays[i]``.
    """
    if not arrays:
        return np.empty((0, 2), dtype=np.float64), np.zeros(1, dtype=np.int64)
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    parts: List[np.ndarray] = []
    for i, a in enumerate(arrays):
        a = np.asarray(a, dtype=np.float64)
        offsets[i + 1] = offsets[i] + len(a)
        if len(a):
            parts.append(a.reshape(len(a), -1))
    flat = np.concatenate(parts, axis=0) if parts else np.empty((0, 2), dtype=np.float64)
    return flat, offsets


def _deserialize_ragged(
    flat: np.ndarray, offsets: np.ndarray
) -> List[np.ndarray]:
    """Reconstruct a list of arrays from the ``(flat, offsets)`` pair
    produced by :func:`_serialize_ragged`."""
    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def _save_ragged(group, name: str, arrays: List[np.ndarray]) -> None:
    """Save a list of variable-length 2-D arrays as flat+offsets datasets."""
    flat, offsets = _serialize_ragged(arrays)
    group.create_dataset(f"{name}_flat", data=flat, compression="gzip", compression_opts=4)
    group.create_dataset(f"{name}_offsets", data=offsets)


def _load_ragged(group, name: str) -> List[np.ndarray]:
    flat = group[f"{name}_flat"][:]
    offsets = group[f"{name}_offsets"][:]
    return _deserialize_ragged(flat, offsets)


def _serialize_ragged_1d(
    arrays: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Like :func:`_serialize_ragged` but keeps arrays 1-D (int64)."""
    if not arrays:
        return np.empty(0, dtype=np.int64), np.zeros(1, dtype=np.int64)
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    parts: List[np.ndarray] = []
    for i, a in enumerate(arrays):
        a = np.asarray(a, dtype=np.int64).ravel()
        offsets[i + 1] = offsets[i] + len(a)
        if len(a):
            parts.append(a)
    flat = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    return flat, offsets


def _save_ragged_1d(group, name: str, arrays: List[np.ndarray]) -> None:
    """Save a list of variable-length 1-D arrays as flat+offsets datasets."""
    flat, offsets = _serialize_ragged_1d(arrays)
    group.create_dataset(f"{name}_flat", data=flat)
    group.create_dataset(f"{name}_offsets", data=offsets)


def _load_ragged_1d(group, name: str) -> List[np.ndarray]:
    flat = group[f"{name}_flat"][:]
    offsets = group[f"{name}_offsets"][:]
    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


# ------------------------------------------------------------------
# Tissue serialization
# ------------------------------------------------------------------

def _save_tissue(grp: h5py.Group, series: TissueGraphTimeSeries) -> None:
    """Write a TissueGraphTimeSeries into an HDF5 group."""
    grp.attrs["pixel_size"] = series.pixel_size if series.pixel_size is not None else ""
    grp.attrs["time_interval"] = series.time_interval if series.time_interval is not None else ""
    grp.attrs["input_type"] = series.input_type.value if series.input_type else ""
    grp.attrs["metadata_json"] = json.dumps(series.metadata or {})

    # ── T1 events ────────────────────────────────────────────────────────────
    t1_grp = grp.create_group("t1_events")
    n_t1 = len(series.t1_events)
    if n_t1:
        t1_grp.create_dataset("frames",
            data=np.array([e.frame for e in series.t1_events], dtype=np.int64))
        t1_grp.create_dataset("losing_pairs",
            data=np.array([e.losing_pair for e in series.t1_events], dtype=np.int64).reshape(n_t1, 2))
        t1_grp.create_dataset("gaining_pairs",
            data=np.array([e.gaining_pair for e in series.t1_events], dtype=np.int64).reshape(n_t1, 2))
        t1_grp.create_dataset("locations",
            data=np.array([e.location for e in series.t1_events], dtype=np.float64).reshape(n_t1, 2))
        all_cells_arrays = [np.array(sorted(e.all_cells), dtype=np.int64) for e in series.t1_events]
        _save_ragged_1d(t1_grp, "all_cells", all_cells_arrays)

    # ── Frames ───────────────────────────────────────────────────────────────
    frames_grp = grp.create_group("frames")
    for fi in series.frame_indices:
        frame = series.frames[fi]
        fg = frames_grp.create_group(str(fi))
        fg.attrs["input_type"] = frame.input_type.value

        # Cells
        cg = fg.create_group("cells")
        cell_ids = sorted(frame.cells.keys())
        n = len(cell_ids)
        cg.create_dataset("ids", data=np.array(cell_ids, dtype=np.int64))
        cg.create_dataset("positions",
            data=np.array([frame.cells[c].position for c in cell_ids], dtype=np.float64).reshape(n, 2))
        cg.create_dataset("areas",
            data=np.array([frame.cells[c].area for c in cell_ids], dtype=np.float64))
        cg.create_dataset("perimeters",
            data=np.array([frame.cells[c].perimeter for c in cell_ids], dtype=np.float64))
        cg.create_dataset("shape_indices",
            data=np.array([frame.cells[c].shape_index for c in cell_ids], dtype=np.float64))
        cg.create_dataset("num_neighbors",
            data=np.array([frame.cells[c].num_neighbors for c in cell_ids], dtype=np.int64))
        cg.create_dataset("track_ids",
            data=np.array([c2 if (c2 := frame.cells[c].track_id) is not None else -1
                           for c in cell_ids], dtype=np.int64))
        cg.create_dataset("pressures",
            data=np.array([frame.cells[c].pressure if frame.cells[c].pressure is not None else np.nan
                           for c in cell_ids], dtype=np.float64))
        cg.create_dataset("is_border",
            data=np.array([frame.cells[c].is_border for c in cell_ids], dtype=np.bool_))

        # Cell vertices (ragged)
        verts = [frame.cells[c].vertices if frame.cells[c].vertices is not None
                 else np.empty((0, 2), dtype=np.float64) for c in cell_ids]
        _save_ragged(cg, "vertices", verts)

        # Junctions
        jg = fg.create_group("junctions")
        junc_keys = sorted(frame.junctions.keys(), key=lambda fs: tuple(sorted(fs)))
        nj = len(junc_keys)
        if nj:
            jg.create_dataset("pairs",
                data=np.array([tuple(sorted(k)) for k in junc_keys], dtype=np.int64).reshape(nj, 2))
            jg.create_dataset("lengths",
                data=np.array([frame.junctions[k].length for k in junc_keys], dtype=np.float64))
            jg.create_dataset("midpoints",
                data=np.array([frame.junctions[k].midpoint for k in junc_keys], dtype=np.float64).reshape(nj, 2))
            jg.create_dataset("tensions",
                data=np.array([frame.junctions[k].tension if frame.junctions[k].tension is not None else np.nan
                               for k in junc_keys], dtype=np.float64))
            jg.create_dataset("stresses",
                data=np.array([frame.junctions[k].normal_stress if frame.junctions[k].normal_stress is not None else np.nan
                               for k in junc_keys], dtype=np.float64))
            jg.create_dataset("tags",
                data=np.array([",".join(sorted(frame.junctions[k].tags)) for k in junc_keys],
                              dtype=_STR_DTYPE))
            _save_ragged(jg, "coords", [frame.junctions[k].coordinates for k in junc_keys])
        else:
            jg.create_dataset("pairs", data=np.empty((0, 2), dtype=np.int64))

        # Graph edges
        eg = fg.create_group("edges")
        graph_edges = list(frame.graph.edges(data=True))
        if graph_edges:
            eg.create_dataset("pairs",
                data=np.array([(u, v) for u, v, _ in graph_edges], dtype=np.int64))
            eg.create_dataset("weights",
                data=np.array([d.get("length", 0.0) for _, _, d in graph_edges], dtype=np.float64))
        else:
            eg.create_dataset("pairs", data=np.empty((0, 2), dtype=np.int64))
            eg.create_dataset("weights", data=np.empty(0, dtype=np.float64))

    # ── Trajectories ─────────────────────────────────────────────────────────
    traj_grp = grp.create_group("trajectories")
    for tid, traj in series.edge_trajectories.items():
        tg = traj_grp.create_group(str(tid))
        tg.attrs["trajectory_id"] = tid
        tg.attrs["name"] = traj.name or ""
        tg.attrs["tags"] = ",".join(sorted(traj.tags)) if traj.tags else ""

        tg.create_dataset("frames", data=np.array(traj.frames, dtype=np.int64))
        tg.create_dataset("signed_lengths", data=np.array(traj.signed_lengths, dtype=np.float64))
        pairs = np.array(traj.cell_pairs, dtype=np.int64).reshape(-1, 2) if traj.cell_pairs else np.empty((0, 2), dtype=np.int64)
        tg.create_dataset("cell_pairs", data=pairs)

        # Coordinates (ragged list of arrays)
        _save_ragged(tg, "coords",
                     traj.coordinates if traj.coordinates else [])

        # T1 event indices (references into series.t1_events)
        t1_idxs = []
        for evt in traj.t1_events:
            for i, se in enumerate(series.t1_events):
                if se is evt:
                    t1_idxs.append(i)
                    break
        tg.create_dataset("t1_indices", data=np.array(t1_idxs, dtype=np.int64))


def _load_tissue(grp: h5py.Group) -> TissueGraphTimeSeries:
    """Reconstruct a TissueGraphTimeSeries from an HDF5 group."""
    pixel_size_raw = grp.attrs.get("pixel_size", "")
    time_interval_raw = grp.attrs.get("time_interval", "")
    pixel_size = float(pixel_size_raw) if pixel_size_raw != "" else None
    time_interval = float(time_interval_raw) if time_interval_raw != "" else None
    input_type_str = grp.attrs.get("input_type", "")
    series_input_type = InputType(input_type_str) if input_type_str else None
    metadata = json.loads(grp.attrs.get("metadata_json", "{}"))

    # ── T1 events ────────────────────────────────────────────────────────────
    t1_events: List[T1Event] = []
    t1_grp = grp["t1_events"]
    if "frames" in t1_grp:
        t1_frames = t1_grp["frames"][:]
        t1_losing = t1_grp["losing_pairs"][:]
        t1_gaining = t1_grp["gaining_pairs"][:]
        t1_locs = t1_grp["locations"][:]
        all_cells_list = _load_ragged_1d(t1_grp, "all_cells")
        for i in range(len(t1_frames)):
            t1_events.append(T1Event(
                frame=int(t1_frames[i]),
                losing_pair=tuple(t1_losing[i].tolist()),
                gaining_pair=tuple(t1_gaining[i].tolist()),
                location=t1_locs[i].copy(),
                all_cells=set(int(x) for x in all_cells_list[i]),
            ))

    # ── Frames ───────────────────────────────────────────────────────────────
    frames: Dict[int, TissueGraphFrame] = {}
    for fi_str, fg in grp["frames"].items():
        fi = int(fi_str)
        input_type = InputType(fg.attrs["input_type"])

        cg = fg["cells"]
        cell_ids = cg["ids"][:].astype(np.int64)
        positions = cg["positions"][:]
        areas = cg["areas"][:]
        perimeters = cg["perimeters"][:]
        shape_indices = cg["shape_indices"][:]
        num_neighbors = cg["num_neighbors"][:]
        track_ids = cg["track_ids"][:]
        pressures = cg["pressures"][:]
        is_border = cg["is_border"][:]
        vert_list = _load_ragged(cg, "vertices")

        cells: Dict[int, CellData] = {}
        for i, cid in enumerate(cell_ids):
            cid = int(cid)
            verts = vert_list[i] if len(vert_list[i]) > 0 else None
            pressure = float(pressures[i]) if not np.isnan(pressures[i]) else None
            tid = int(track_ids[i])
            cells[cid] = CellData(
                cell_id=cid,
                position=positions[i].copy(),
                area=float(areas[i]),
                perimeter=float(perimeters[i]),
                shape_index=float(shape_indices[i]),
                num_neighbors=int(num_neighbors[i]),
                track_id=tid if tid != -1 else None,
                vertices=verts,
                pressure=pressure,
                is_border=bool(is_border[i]),
            )

        jg = fg["junctions"]
        junctions: Dict = {}
        if "pairs" in jg and len(jg["pairs"]) > 0:
            pairs = jg["pairs"][:]
            lengths = jg["lengths"][:]
            midpoints = jg["midpoints"][:]
            tensions = jg["tensions"][:]
            stresses = jg["stresses"][:]
            tag_strs = jg["tags"][:]
            coord_list = _load_ragged(jg, "coords")
            for j in range(len(pairs)):
                pair = (int(pairs[j, 0]), int(pairs[j, 1]))
                key = frozenset(pair)
                tension = float(tensions[j]) if not np.isnan(tensions[j]) else None
                stress = float(stresses[j]) if not np.isnan(stresses[j]) else None
                tag_str = tag_strs[j].decode() if isinstance(tag_strs[j], bytes) else str(tag_strs[j])
                tags = set(tag_str.split(",")) - {""} if tag_str else set()
                junctions[key] = JunctionData(
                    cell_pair=pair,
                    length=float(lengths[j]),
                    coordinates=coord_list[j].copy(),
                    midpoint=midpoints[j].copy(),
                    tension=tension,
                    normal_stress=stress,
                    tags=tags,
                )

        eg = fg["edges"]
        graph = nx.Graph()
        for cid in cell_ids:
            graph.add_node(int(cid))
        edge_pairs = eg["pairs"][:]
        edge_weights = eg["weights"][:]
        for j in range(len(edge_pairs)):
            graph.add_edge(int(edge_pairs[j, 0]), int(edge_pairs[j, 1]),
                           length=float(edge_weights[j]))

        frames[fi] = TissueGraphFrame(
            frame=fi, graph=graph, cells=cells,
            junctions=junctions, input_type=input_type,
        )

    # ── Trajectories ─────────────────────────────────────────────────────────
    edge_trajectories: Dict[int, EdgeTrajectory] = {}
    for tid_str, tg in grp["trajectories"].items():
        tid = int(tg.attrs["trajectory_id"])
        name_raw = tg.attrs.get("name", "")
        name = name_raw if name_raw else None
        tags_raw = tg.attrs.get("tags", "")
        tags = set(tags_raw.split(",")) - {""} if tags_raw else set()

        traj_frames = tg["frames"][:].astype(np.int64).tolist()
        signed_lengths = tg["signed_lengths"][:].astype(np.float64).tolist()
        pairs_arr = tg["cell_pairs"][:]
        cell_pairs = [(int(pairs_arr[j, 0]), int(pairs_arr[j, 1]))
                      for j in range(len(pairs_arr))]
        coord_list = _load_ragged(tg, "coords")
        t1_idxs = tg["t1_indices"][:].astype(np.int64).tolist()
        traj_t1 = [t1_events[idx] for idx in t1_idxs if idx < len(t1_events)]

        edge_trajectories[tid] = EdgeTrajectory(
            trajectory_id=tid,
            frames=traj_frames,
            cell_pairs=cell_pairs,
            signed_lengths=signed_lengths,
            coordinates=coord_list,
            t1_events=traj_t1,
            tags=tags,
            name=name,
        )

    return TissueGraphTimeSeries(
        frames=frames,
        edge_trajectories=edge_trajectories,
        t1_events=t1_events,
        pixel_size=pixel_size,
        time_interval=time_interval,
        input_type=series_input_type,
        metadata=metadata,
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def save_project(
    path: Union[str, Path],
    *,
    labels: Optional[np.ndarray] = None,
    nuclear_labels: Optional[np.ndarray] = None,
    dataset: Optional[TissueGraphDataset] = None,
    pixel_size: Optional[float] = None,
    time_interval: Optional[float] = None,
    condition: str = "",
    provenance: Optional[Dict] = None,
) -> None:
    """Save a CellFlow project to a single HDF5 file.

    Parameters
    ----------
    path:
        Output file path (e.g. ``"experiment.h5"``).
    labels:
        Optional label stack (T, H, W) or (H, W).
    nuclear_labels:
        Optional nuclear label stack (T, H, W) or (H, W).
    dataset:
        Optional TissueGraphDataset with analysis results.
    pixel_size, time_interval, condition:
        Project-level metadata.
    """
    path = Path(path)
    with h5py.File(path, "w") as f:
        f.attrs["version"] = _FORMAT_VERSION
        f.attrs["created"] = datetime.now().isoformat()

        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["pixel_size"] = pixel_size if pixel_size is not None else ""
        meta.attrs["time_interval"] = time_interval if time_interval is not None else ""
        meta.attrs["condition"] = condition or ""

        # Labels
        if labels is not None:
            arr = np.asarray(labels)
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            chunks = (1, min(512, arr.shape[1]), min(512, arr.shape[2]))
            f.create_dataset("labels", data=arr, chunks=chunks,
                             compression="gzip", compression_opts=4)
            _save_label_metadata(f, "label_metadata", arr)

        # Nuclear Labels
        if nuclear_labels is not None:
            arr = np.asarray(nuclear_labels)
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            chunks = (1, min(512, arr.shape[1]), min(512, arr.shape[2]))
            f.create_dataset("nuclear_labels", data=arr, chunks=chunks,
                             compression="gzip", compression_opts=4)
            _save_label_metadata(f, "nuclear_label_metadata", arr)

        # Analysis
        if dataset is not None:
            ana = f.create_group("analysis")
            ana.attrs["condition"] = dataset.condition or ""
            ana.attrs["pixel_size"] = dataset.pixel_size if dataset.pixel_size is not None else ""
            ana.attrs["time_interval"] = dataset.time_interval if dataset.time_interval is not None else ""
            ana.attrs["input_type"] = dataset.input_type.value if dataset.input_type else ""
            ana.attrs["metadata_json"] = json.dumps(dataset.metadata or {})

            for tid in dataset.tissue_ids:
                tg = ana.create_group(f"tissue_{tid:03d}")
                tg.attrs["tissue_id"] = tid
                _save_tissue(tg, dataset.tissues[tid])

        # Provenance (optional — pipeline manifest + config hashes)
        if provenance is not None:
            prov = f.create_group("provenance")
            prov.attrs["saved_at"] = datetime.now().isoformat()
            prov.attrs["provenance_json"] = json.dumps(provenance)

    logger.info("Saved project to %s", path)


def load_project(path: Union[str, Path]) -> dict:
    """Load a CellFlow project from an HDF5 file.

    Returns
    -------
    dict with keys:
        ``labels`` (ndarray or None), ``dataset`` (TissueGraphDataset or None),
        ``pixel_size`` (float or None), ``time_interval`` (float or None),
        ``condition`` (str)
    """
    path = Path(path)
    result: dict = {
        "labels": None,
        "nuclear_labels": None,
        "dataset": None,
        "pixel_size": None,
        "time_interval": None,
        "condition": "",
    }

    with h5py.File(path, "r") as f:
        # Metadata
        if "metadata" in f:
            meta = f["metadata"]
            px = meta.attrs.get("pixel_size", "")
            dt = meta.attrs.get("time_interval", "")
            result["pixel_size"] = float(px) if px != "" else None
            result["time_interval"] = float(dt) if dt != "" else None
            result["condition"] = meta.attrs.get("condition", "")

        # Labels
        if "labels" in f:
            result["labels"] = f["labels"][:]

        # Nuclear Labels
        if "nuclear_labels" in f:
            result["nuclear_labels"] = f["nuclear_labels"][:]

        # Analysis
        if "analysis" in f:
            ana = f["analysis"]
            px_raw = ana.attrs.get("pixel_size", "")
            dt_raw = ana.attrs.get("time_interval", "")
            ds = TissueGraphDataset(
                condition=ana.attrs.get("condition", ""),
                pixel_size=float(px_raw) if px_raw != "" else None,
                time_interval=float(dt_raw) if dt_raw != "" else None,
                input_type=InputType(ana.attrs["input_type"]) if ana.attrs.get("input_type") else None,
                metadata=json.loads(ana.attrs.get("metadata_json", "{}")),
            )
            for key in sorted(ana.keys()):
                if not key.startswith("tissue_"):
                    continue
                tg = ana[key]
                tid = int(tg.attrs["tissue_id"])
                series = _load_tissue(tg)
                ds.tissues[tid] = series
            result["dataset"] = ds

            # Prefer metadata group values for top-level fields
            if result["pixel_size"] is None and ds.pixel_size is not None:
                result["pixel_size"] = ds.pixel_size
            if result["time_interval"] is None and ds.time_interval is not None:
                result["time_interval"] = ds.time_interval
            if not result["condition"] and ds.condition:
                result["condition"] = ds.condition

    logger.info("Loaded project from %s", path)
    return result


# ------------------------------------------------------------------
# Single-tissue IO  (new API)
# ------------------------------------------------------------------

def save_tissue(path: Union[str, Path], tissue) -> None:
    """Write a TissueData to a single-tissue HDF5 file.

    *tissue* is a :class:`~cellflow.frontend.registry.TissueData` instance.
    Any of its fields (image, labels, series) may be None and will be skipped.
    """
    from cellflow.napari.registry import TissueData  # avoid circular at module level

    path = Path(path)
    dataset: Optional[TissueGraphDataset] = None
    if tissue.series is not None:
        dataset = TissueGraphDataset(
            condition="",
            pixel_size=tissue.series.pixel_size,
            time_interval=tissue.series.time_interval,
        )
        dataset.tissues[0] = tissue.series

    save_project(
        path,
        labels=tissue.labels,
        nuclear_labels=tissue.nuclear_labels,
        dataset=dataset,
        pixel_size=tissue.pixel_size,
        time_interval=tissue.time_interval,
        condition=tissue.condition,
    )


def load_tissue(path: Union[str, Path]):
    """Load a single-tissue HDF5 file into a TissueData.

    Returns a :class:`~cellflow.frontend.registry.TissueData` instance.
    """
    from cellflow.napari.registry import TissueData  # avoid circular at module level

    result = load_project(path)
    ds = result.get("dataset")
    series: Optional[TissueGraphTimeSeries] = None
    if ds is not None and ds.tissues:
        # take the first tissue (single-tissue convention)
        series = ds.tissues[min(ds.tissues.keys())]

    return TissueData(
        labels=result.get("labels"),
        nuclear_labels=result.get("nuclear_labels"),
        series=series,
        path=str(path),
        pixel_size=result.get("pixel_size"),
        time_interval=result.get("time_interval"),
        condition=result.get("condition", ""),
    )


def read_tissue_summary(path: Union[str, Path]) -> dict:
    """Read only the metadata needed for a catalog summary without loading arrays.

    Returns a dict with keys: n_frames, avg_cells, n_t1_events, n_trajectories, condition.
    Opens the HDF5 file but does not load any large datasets.
    """
    path = Path(path)
    summary: dict = {}
    with h5py.File(path, "r") as f:
        # Read condition from metadata group first, fall back to analysis group
        condition = ""
        if "metadata" in f:
            condition = f["metadata"].attrs.get("condition", "")
        if not condition and "analysis" in f:
            condition = f["analysis"].attrs.get("condition", "")
        summary["condition"] = condition

        if "analysis" not in f:
            return summary
        ana = f["analysis"]
        tissue_keys = sorted(k for k in ana.keys() if k.startswith("tissue_"))
        if not tissue_keys:
            return summary

        # use the first tissue group
        tg = ana[tissue_keys[0]]
        frames_grp = tg.get("frames", {})
        n_frames = len(frames_grp)
        summary["n_frames"] = n_frames

        # count cells per frame for avg
        total_cells = 0
        for fi_str in frames_grp:
            fg = frames_grp[fi_str]
            cells_grp = fg.get("cells", {})
            if "ids" in cells_grp:
                total_cells += len(cells_grp["ids"])
        summary["avg_cells"] = round(total_cells / n_frames, 1) if n_frames else 0

        t1_grp = tg.get("t1_events", {})
        summary["n_t1_events"] = len(t1_grp.get("frames", [])) if "frames" in t1_grp else 0

        traj_grp = tg.get("trajectories", {})
        summary["n_trajectories"] = len(traj_grp)

    return summary


# ------------------------------------------------------------------
# Dataset catalog IO  (new v2.0 cfproj format)
# ------------------------------------------------------------------

def save_catalog(path: Union[str, Path], catalog) -> None:
    """Write a DatasetCatalog to a v3.0 JSON .cfproj file.

    *catalog* is a :class:`~cellflow.napari.registry.DatasetCatalog`.
    """
    path = Path(path)
    data = {
        "version": _CATALOG_VERSION,
        "pixel_size": catalog.pixel_size,
        "time_interval": catalog.time_interval,
        "condition": catalog.condition or "",
        "condition_groups": getattr(catalog, "condition_groups", {}),
        "entries": [
            {
                "path": e.path,
                "display_name": e.display_name,
                "condition": e.condition,
                "summary": e.summary,
            }
            for e in catalog.entries
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved catalog to %s", path)


def load_catalog(path: Union[str, Path]):
    """Load a .cfproj file into a DatasetCatalog (v1.0 / v2.0 / v3.0).

    Returns a :class:`~cellflow.napari.registry.DatasetCatalog`.
    """
    from cellflow.napari.registry import CatalogEntry, DatasetCatalog  # avoid circular

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    entries = []
    for e in data.get("entries", []):
        entries.append(CatalogEntry(
            path=e["path"],
            display_name=e.get("display_name", ""),
            condition=e.get("condition", e.get("note", "")),  # "note" is legacy v1.0 field
            summary=e.get("summary", {}),
        ))

    # v1.0 cfproj used a "files" key instead of "entries"
    if not entries and "files" in data:
        for f in data["files"]:
            entries.append(CatalogEntry(
                path=f["path"],
                display_name=f.get("display_name", ""),
                condition=f.get("condition", f.get("note", "")),
            ))

    catalog = DatasetCatalog(
        entries=entries,
        path=str(path),
        pixel_size=data.get("pixel_size"),
        time_interval=data.get("time_interval"),
        condition=data.get("condition", ""),
        condition_groups=data.get("condition_groups", {}),
    )
    logger.info("Loaded catalog from %s", path)
    return catalog


# ------------------------------------------------------------------
# Manifest IO  (legacy)
# ------------------------------------------------------------------

def save_manifest(path: Union[str, Path], manifest: ProjectManifest) -> None:
    """Write a :class:`ProjectManifest` to a JSON file (``.cfproj``).

    File paths inside the manifest are stored as-is (relative paths
    are resolved at load time against the manifest directory).
    """
    path = Path(path)
    data = {
        "version": _MANIFEST_VERSION,
        "active_index": manifest.active_index,
        "files": [
            {"path": e.path, "display_name": e.display_name, "note": e.note}
            for e in manifest.entries
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved manifest to %s", path)


def load_manifest(path: Union[str, Path]) -> ProjectManifest:
    """Load a :class:`ProjectManifest` from a ``.cfproj`` JSON file."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = [
        ProjectEntry(
            path=f["path"],
            display_name=f.get("display_name", ""),
            note=f.get("note", ""),
        )
        for f in data.get("files", [])
    ]
    active_index = data.get("active_index", 0)
    if entries:
        active_index = max(0, min(active_index, len(entries) - 1))
    return ProjectManifest(entries=entries, active_index=active_index)


# ------------------------------------------------------------------
# Directory-format dataset IO  (v3.0 — metadata.json + tissue_NNN.npz)
# ------------------------------------------------------------------

def _tissue_to_arrays(series: TissueGraphTimeSeries) -> dict:
    """Serialize a TissueGraphTimeSeries to a flat dict of numpy arrays for npz."""
    a: dict = {}

    # Scalar metadata
    a['pixel_size'] = np.array([series.pixel_size if series.pixel_size is not None else np.nan])
    a['time_interval'] = np.array([series.time_interval if series.time_interval is not None else np.nan])
    a['input_type'] = np.array([series.input_type.value if series.input_type else ''])
    a['metadata_json'] = np.array([json.dumps(series.metadata or {})])

    frame_indices = sorted(series.frames.keys())
    n_frames = len(frame_indices)
    a['frame_indices'] = np.array(frame_indices, dtype=np.int64)
    a['frame_input_types'] = np.array([series.frames[fi].input_type.value for fi in frame_indices])

    # T1 events
    t1_events = series.t1_events
    n_t1 = len(t1_events)
    if n_t1:
        a['t1_frames'] = np.array([e.frame for e in t1_events], dtype=np.int64)
        a['t1_losing_pairs'] = np.array([list(e.losing_pair) for e in t1_events], dtype=np.int64)
        a['t1_gaining_pairs'] = np.array([list(e.gaining_pair) for e in t1_events], dtype=np.int64)
        a['t1_locations'] = np.array([e.location for e in t1_events], dtype=np.float64)
        all_cells = [np.array(sorted(e.all_cells), dtype=np.int64) for e in t1_events]
        t1_flat, t1_offsets = _serialize_ragged_1d(all_cells)
        a['t1_all_cells_flat'] = t1_flat
        a['t1_all_cells_offsets'] = t1_offsets
    else:
        a['t1_frames'] = np.empty(0, dtype=np.int64)
        a['t1_losing_pairs'] = np.empty((0, 2), dtype=np.int64)
        a['t1_gaining_pairs'] = np.empty((0, 2), dtype=np.int64)
        a['t1_locations'] = np.empty((0, 2), dtype=np.float64)
        a['t1_all_cells_flat'] = np.empty(0, dtype=np.int64)
        a['t1_all_cells_offsets'] = np.zeros(1, dtype=np.int64)

    # Per-frame cells (concatenated across all frames)
    cells_offsets = np.zeros(n_frames + 1, dtype=np.int64)
    cids_all: List = []
    pos_all: List = []
    area_all: List = []
    peri_all: List = []
    si_all: List = []
    nn_all: List = []
    tid_all: List = []
    pres_all: List = []
    bord_all: List = []
    verts_all: List = []

    for fi_idx, fi in enumerate(frame_indices):
        frame = series.frames[fi]
        cids = sorted(frame.cells.keys())
        cells_offsets[fi_idx + 1] = cells_offsets[fi_idx] + len(cids)
        for cid in cids:
            c = frame.cells[cid]
            cids_all.append(cid)
            pos_all.append(c.position)
            area_all.append(c.area)
            peri_all.append(c.perimeter)
            si_all.append(c.shape_index)
            nn_all.append(c.num_neighbors)
            tid_all.append(c.track_id if c.track_id is not None else -1)
            pres_all.append(c.pressure)  # keep None; converted to nan only when writing
            bord_all.append(c.is_border)
            verts_all.append(c.vertices if c.vertices is not None else np.empty((0, 2), dtype=np.float64))

    tc = int(cells_offsets[-1])
    a['cells_offsets'] = cells_offsets
    a['cells_ids'] = np.array(cids_all, dtype=np.int64) if tc else np.empty(0, dtype=np.int64)
    a['cells_positions'] = np.array(pos_all, dtype=np.float64).reshape(tc, 2) if tc else np.empty((0, 2), dtype=np.float64)
    a['cells_areas'] = np.array(area_all, dtype=np.float64) if tc else np.empty(0, dtype=np.float64)
    a['cells_perimeters'] = np.array(peri_all, dtype=np.float64) if tc else np.empty(0, dtype=np.float64)
    a['cells_shape_indices'] = np.array(si_all, dtype=np.float64) if tc else np.empty(0, dtype=np.float64)
    a['cells_num_neighbors'] = np.array(nn_all, dtype=np.int64) if tc else np.empty(0, dtype=np.int64)
    a['cells_track_ids'] = np.array(tid_all, dtype=np.int64) if tc else np.empty(0, dtype=np.int64)
    if tc and any(p is not None for p in pres_all):
        a['cells_pressures'] = np.array(
            [p if p is not None else np.nan for p in pres_all], dtype=np.float64
        )
    a['cells_is_border'] = np.array(bord_all, dtype=bool) if tc else np.empty(0, dtype=bool)
    vf, vo = _serialize_ragged(verts_all) if verts_all else (np.empty((0, 2), dtype=np.float64), np.zeros(1, dtype=np.int64))
    a['cells_verts_flat'] = vf
    a['cells_verts_offsets'] = vo

    # Per-frame junctions (concatenated)
    juncs_offsets = np.zeros(n_frames + 1, dtype=np.int64)
    jp_all: List = []
    jl_all: List = []
    jm_all: List = []
    jt_all: List = []
    js_all: List = []
    jtags_all: List = []
    jcoords_all: List = []

    for fi_idx, fi in enumerate(frame_indices):
        frame = series.frames[fi]
        jkeys = sorted(frame.junctions.keys(), key=lambda fs: tuple(sorted(fs)))
        juncs_offsets[fi_idx + 1] = juncs_offsets[fi_idx] + len(jkeys)
        for k in jkeys:
            j = frame.junctions[k]
            jp_all.append(tuple(sorted(k)))
            jl_all.append(j.length)
            jm_all.append(j.midpoint)
            jt_all.append(j.tension if j.tension is not None else np.nan)
            js_all.append(j.normal_stress if j.normal_stress is not None else np.nan)
            jtags_all.append(",".join(sorted(j.tags)))
            jcoords_all.append(j.coordinates)

    tj = int(juncs_offsets[-1])
    a['juncs_offsets'] = juncs_offsets
    a['juncs_pairs'] = np.array(jp_all, dtype=np.int64).reshape(tj, 2) if tj else np.empty((0, 2), dtype=np.int64)
    a['juncs_lengths'] = np.array(jl_all, dtype=np.float64) if tj else np.empty(0, dtype=np.float64)
    a['juncs_midpoints'] = np.array(jm_all, dtype=np.float64).reshape(tj, 2) if tj else np.empty((0, 2), dtype=np.float64)
    a['juncs_tensions'] = np.array(jt_all, dtype=np.float64) if tj else np.empty(0, dtype=np.float64)
    a['juncs_stresses'] = np.array(js_all, dtype=np.float64) if tj else np.empty(0, dtype=np.float64)
    a['juncs_tags'] = np.array(jtags_all) if tj else np.empty(0, dtype='U1')
    jcf, jco = _serialize_ragged(jcoords_all) if jcoords_all else (np.empty((0, 2), dtype=np.float64), np.zeros(1, dtype=np.int64))
    a['juncs_coords_flat'] = jcf
    a['juncs_coords_offsets'] = jco

    # Per-frame graph edges (concatenated)
    edges_offsets = np.zeros(n_frames + 1, dtype=np.int64)
    ep_all: List = []
    ew_all: List = []

    for fi_idx, fi in enumerate(frame_indices):
        frame = series.frames[fi]
        graph_edges = list(frame.graph.edges(data=True))
        edges_offsets[fi_idx + 1] = edges_offsets[fi_idx] + len(graph_edges)
        for u, v, d in graph_edges:
            ep_all.append((u, v))
            ew_all.append(d.get("length", 0.0))

    te = int(edges_offsets[-1])
    a['edges_offsets'] = edges_offsets
    a['edges_pairs'] = np.array(ep_all, dtype=np.int64).reshape(te, 2) if te else np.empty((0, 2), dtype=np.int64)
    a['edges_weights'] = np.array(ew_all, dtype=np.float64) if te else np.empty(0, dtype=np.float64)

    # Trajectories
    traj_items = sorted(series.edge_trajectories.items())
    n_traj = len(traj_items)

    if n_traj:
        traj_ids_arr = np.array([tid for tid, _ in traj_items], dtype=np.int64)
        traj_names_arr = np.array([t.name or "" for _, t in traj_items])
        traj_tags_arr = np.array([",".join(sorted(t.tags)) if t.tags else "" for _, t in traj_items])

        tf_lists = [np.array(t.frames, dtype=np.int64) for _, t in traj_items]
        tl_lists = [np.array(t.signed_lengths, dtype=np.float64) for _, t in traj_items]
        tp_lists = [
            np.array(t.cell_pairs, dtype=np.int64).reshape(-1, 2) if t.cell_pairs else np.empty((0, 2), dtype=np.int64)
            for _, t in traj_items
        ]

        traj_frames_offsets = np.zeros(n_traj + 1, dtype=np.int64)
        for i, arr in enumerate(tf_lists):
            traj_frames_offsets[i + 1] = traj_frames_offsets[i] + len(arr)
        traj_frames_flat = np.concatenate([a2 for a2 in tf_lists if len(a2)]) if any(len(a2) for a2 in tf_lists) else np.empty(0, dtype=np.int64)
        traj_lengths_flat = np.concatenate([a2 for a2 in tl_lists if len(a2)]) if any(len(a2) for a2 in tl_lists) else np.empty(0, dtype=np.float64)
        traj_pairs_flat = np.concatenate([a2 for a2 in tp_lists if len(a2)], axis=0) if any(len(a2) for a2 in tp_lists) else np.empty((0, 2), dtype=np.int64)

        # Coordinates: ragged of ragged
        traj_coord_seg_offsets = np.zeros(n_traj + 1, dtype=np.int64)
        all_coord_arrs: List = []
        for i, (_, t) in enumerate(traj_items):
            traj_coord_seg_offsets[i + 1] = traj_coord_seg_offsets[i] + len(t.coordinates)
            all_coord_arrs.extend(t.coordinates or [])
        tcf, tco = _serialize_ragged(all_coord_arrs) if all_coord_arrs else (np.empty((0, 2), dtype=np.float64), np.zeros(1, dtype=np.int64))

        # T1 back-references
        traj_t1_offsets = np.zeros(n_traj + 1, dtype=np.int64)
        traj_t1_parts: List = []
        for i, (_, t) in enumerate(traj_items):
            idxs = []
            for evt in t.t1_events:
                for i_t1, se in enumerate(t1_events):
                    if se is evt:
                        idxs.append(i_t1)
                        break
            arr_t1 = np.array(idxs, dtype=np.int64)
            traj_t1_offsets[i + 1] = traj_t1_offsets[i] + len(arr_t1)
            if len(arr_t1):
                traj_t1_parts.append(arr_t1)
        traj_t1_flat = np.concatenate(traj_t1_parts) if traj_t1_parts else np.empty(0, dtype=np.int64)

        a['traj_ids'] = traj_ids_arr
        a['traj_names'] = traj_names_arr
        a['traj_tags'] = traj_tags_arr
        a['traj_frames_flat'] = traj_frames_flat
        a['traj_frames_offsets'] = traj_frames_offsets
        a['traj_lengths_flat'] = traj_lengths_flat
        a['traj_pairs_flat'] = traj_pairs_flat
        a['traj_coord_seg_offsets'] = traj_coord_seg_offsets
        a['traj_coords_flat'] = tcf
        a['traj_coords_offsets'] = tco
        a['traj_t1_flat'] = traj_t1_flat
        a['traj_t1_offsets'] = traj_t1_offsets
    else:
        a['traj_ids'] = np.empty(0, dtype=np.int64)
        a['traj_names'] = np.empty(0, dtype='U1')
        a['traj_tags'] = np.empty(0, dtype='U1')
        a['traj_frames_flat'] = np.empty(0, dtype=np.int64)
        a['traj_frames_offsets'] = np.zeros(1, dtype=np.int64)
        a['traj_lengths_flat'] = np.empty(0, dtype=np.float64)
        a['traj_pairs_flat'] = np.empty((0, 2), dtype=np.int64)
        a['traj_coord_seg_offsets'] = np.zeros(1, dtype=np.int64)
        a['traj_coords_flat'] = np.empty((0, 2), dtype=np.float64)
        a['traj_coords_offsets'] = np.zeros(1, dtype=np.int64)
        a['traj_t1_flat'] = np.empty(0, dtype=np.int64)
        a['traj_t1_offsets'] = np.zeros(1, dtype=np.int64)

    return a


def _arrays_to_tissue(a: dict) -> TissueGraphTimeSeries:
    """Reconstruct a TissueGraphTimeSeries from the dict produced by _tissue_to_arrays."""
    from .structures import CellData, EdgeTrajectory, JunctionData, T1Event  # local to avoid circular

    px_raw = float(a['pixel_size'][0])
    pixel_size = None if np.isnan(px_raw) else px_raw
    dt_raw = float(a['time_interval'][0])
    time_interval = None if np.isnan(dt_raw) else dt_raw
    input_type_str = str(a['input_type'][0])
    series_input_type = InputType(input_type_str) if input_type_str else None
    metadata = json.loads(str(a['metadata_json'][0]))

    frame_indices = a['frame_indices'].astype(np.int64).tolist()
    frame_input_types = [str(s) for s in a['frame_input_types']]

    # T1 events
    t1_events: List[T1Event] = []
    t1_frames_arr = a['t1_frames']
    if len(t1_frames_arr):
        t1_losing = a['t1_losing_pairs']
        t1_gaining = a['t1_gaining_pairs']
        t1_locs = a['t1_locations']
        t1_flat = a['t1_all_cells_flat']
        t1_off = a['t1_all_cells_offsets']
        for i in range(len(t1_frames_arr)):
            cells_slice = t1_flat[int(t1_off[i]):int(t1_off[i + 1])]
            t1_events.append(T1Event(
                frame=int(t1_frames_arr[i]),
                losing_pair=tuple(int(x) for x in t1_losing[i]),
                gaining_pair=tuple(int(x) for x in t1_gaining[i]),
                location=t1_locs[i].copy(),
                all_cells={int(x) for x in cells_slice},
            ))

    # Load cell arrays once
    cells_offsets = a['cells_offsets']
    cells_ids = a['cells_ids']
    cells_positions = a['cells_positions']
    cells_areas = a['cells_areas']
    cells_perimeters = a['cells_perimeters']
    cells_si = a['cells_shape_indices']
    cells_nn = a['cells_num_neighbors']
    cells_track = a['cells_track_ids']
    cells_pres = a.get('cells_pressures')  # omitted when all pressures are None
    cells_border = a['cells_is_border']
    verts_list = _deserialize_ragged(a['cells_verts_flat'], a['cells_verts_offsets'])

    # Load junction arrays once
    juncs_offsets = a['juncs_offsets']
    juncs_pairs = a['juncs_pairs']
    juncs_lengths = a['juncs_lengths']
    juncs_midpoints = a['juncs_midpoints']
    juncs_tensions = a['juncs_tensions']
    juncs_stresses = a['juncs_stresses']
    juncs_tags = a['juncs_tags']
    jcoords_list = _deserialize_ragged(a['juncs_coords_flat'], a['juncs_coords_offsets'])

    # Load edge arrays once
    edges_offsets = a['edges_offsets']
    edges_pairs = a['edges_pairs']
    edges_weights = a['edges_weights']

    # Reconstruct frames
    frames: Dict[int, TissueGraphFrame] = {}
    for fi_idx, fi in enumerate(frame_indices):
        c_s, c_e = int(cells_offsets[fi_idx]), int(cells_offsets[fi_idx + 1])
        cells: Dict = {}
        for k in range(c_s, c_e):
            cid = int(cells_ids[k])
            verts = verts_list[k] if len(verts_list[k]) > 0 else None
            pres = (
                None if cells_pres is None
                else (None if np.isnan(cells_pres[k]) else float(cells_pres[k]))
            )
            tid_val = int(cells_track[k])
            cells[cid] = CellData(
                cell_id=cid,
                position=cells_positions[k].copy(),
                area=float(cells_areas[k]),
                perimeter=float(cells_perimeters[k]),
                shape_index=float(cells_si[k]),
                num_neighbors=int(cells_nn[k]),
                track_id=tid_val if tid_val != -1 else None,
                vertices=verts,
                pressure=pres,
                is_border=bool(cells_border[k]),
            )

        j_s, j_e = int(juncs_offsets[fi_idx]), int(juncs_offsets[fi_idx + 1])
        junctions: Dict = {}
        for k in range(j_s, j_e):
            pair = (int(juncs_pairs[k, 0]), int(juncs_pairs[k, 1]))
            key = frozenset(pair)
            tension = float(juncs_tensions[k])
            tension = None if np.isnan(tension) else tension
            stress = float(juncs_stresses[k])
            stress = None if np.isnan(stress) else stress
            tag_raw = juncs_tags[k]
            tag_str = tag_raw.decode() if isinstance(tag_raw, bytes) else str(tag_raw)
            tags = set(tag_str.split(",")) - {""} if tag_str else set()
            junctions[key] = JunctionData(
                cell_pair=pair,
                length=float(juncs_lengths[k]),
                coordinates=jcoords_list[k].copy(),
                midpoint=juncs_midpoints[k].copy(),
                tension=tension,
                normal_stress=stress,
                tags=tags,
            )

        e_s, e_e = int(edges_offsets[fi_idx]), int(edges_offsets[fi_idx + 1])
        graph = nx.Graph()
        for cid in cells:
            graph.add_node(cid)
        for k in range(e_s, e_e):
            graph.add_edge(int(edges_pairs[k, 0]), int(edges_pairs[k, 1]),
                           length=float(edges_weights[k]))

        frames[fi] = TissueGraphFrame(
            frame=fi,
            graph=graph,
            cells=cells,
            junctions=junctions,
            input_type=InputType(frame_input_types[fi_idx]),
        )

    # Trajectories
    edge_trajectories: Dict[int, EdgeTrajectory] = {}
    traj_ids_arr = a['traj_ids']
    if len(traj_ids_arr):
        traj_names_arr = a['traj_names']
        traj_tags_arr = a['traj_tags']
        traj_frames_flat = a['traj_frames_flat']
        traj_frames_offsets = a['traj_frames_offsets']
        traj_lengths_flat = a['traj_lengths_flat']
        traj_pairs_flat = a['traj_pairs_flat']
        traj_coord_seg_offsets = a['traj_coord_seg_offsets']
        all_coord_arrs = _deserialize_ragged(a['traj_coords_flat'], a['traj_coords_offsets'])
        traj_t1_flat = a['traj_t1_flat']
        traj_t1_offsets = a['traj_t1_offsets']

        for ti in range(len(traj_ids_arr)):
            tid = int(traj_ids_arr[ti])
            name_raw = str(traj_names_arr[ti])
            name = name_raw if name_raw else None
            tags_raw = str(traj_tags_arr[ti])
            tags = set(tags_raw.split(",")) - {""} if tags_raw else set()

            f_s, f_e = int(traj_frames_offsets[ti]), int(traj_frames_offsets[ti + 1])
            traj_frames = traj_frames_flat[f_s:f_e].astype(np.int64).tolist()
            traj_lengths = traj_lengths_flat[f_s:f_e].astype(np.float64).tolist()
            if len(traj_pairs_flat) and f_e > f_s:
                pairs = [(int(traj_pairs_flat[k, 0]), int(traj_pairs_flat[k, 1]))
                         for k in range(f_s, f_e)]
            else:
                pairs = []

            cs_s, cs_e = int(traj_coord_seg_offsets[ti]), int(traj_coord_seg_offsets[ti + 1])
            coords = [all_coord_arrs[k].copy() for k in range(cs_s, cs_e) if k < len(all_coord_arrs)]

            t1_s, t1_e = int(traj_t1_offsets[ti]), int(traj_t1_offsets[ti + 1])
            t1_idxs = traj_t1_flat[t1_s:t1_e].astype(np.int64).tolist()
            traj_t1 = [t1_events[idx] for idx in t1_idxs if idx < len(t1_events)]

            edge_trajectories[tid] = EdgeTrajectory(
                trajectory_id=tid,
                frames=traj_frames,
                cell_pairs=pairs,
                signed_lengths=traj_lengths,
                coordinates=coords,
                t1_events=traj_t1,
                tags=tags,
                name=name,
            )

    return TissueGraphTimeSeries(
        frames=frames,
        edge_trajectories=edge_trajectories,
        t1_events=t1_events,
        pixel_size=pixel_size,
        time_interval=time_interval,
        input_type=series_input_type,
        metadata=metadata,
    )


def save_dataset(dataset: TissueGraphDataset, path: Union[str, Path]) -> None:
    """Save a TissueGraphDataset to a directory (``metadata.json`` + ``tissue_NNN.npz``).

    Parameters
    ----------
    dataset:
        The dataset to persist.
    path:
        Target directory path (created if absent).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    meta = {
        "version": _DATASET_VERSION,
        "condition": dataset.condition or "",
        "pixel_size": dataset.pixel_size,
        "time_interval": dataset.time_interval,
        "input_type": dataset.input_type.value if dataset.input_type else "",
        "metadata": dataset.metadata or {},
        "n_tissues": dataset.n_tissues,
        "tissue_ids": dataset.tissue_ids,
    }
    (path / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for tid in dataset.tissue_ids:
        arrays = _tissue_to_arrays(dataset.tissues[tid])
        np.savez(path / f"tissue_{tid:03d}.npz", **arrays)

    logger.info("Saved dataset to %s", path)


def load_dataset(path: Union[str, Path]) -> TissueGraphDataset:
    """Load a TissueGraphDataset from a directory written by :func:`save_dataset`.

    Returns
    -------
    TissueGraphDataset
    """
    path = Path(path)
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        raise ValueError(f"No metadata.json found in {path}; not a v3 dataset directory")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    input_type_str = meta.get("input_type", "")
    dataset = TissueGraphDataset(
        condition=meta.get("condition", ""),
        pixel_size=meta.get("pixel_size"),
        time_interval=meta.get("time_interval"),
        input_type=InputType(input_type_str) if input_type_str else None,
        metadata=meta.get("metadata", {}),
    )

    for tid in meta.get("tissue_ids", list(range(meta.get("n_tissues", 0)))):
        npz_path = path / f"tissue_{tid:03d}.npz"
        if npz_path.exists():
            arrays = dict(np.load(npz_path, allow_pickle=False))
            dataset.tissues[tid] = _arrays_to_tissue(arrays)

    logger.info("Loaded dataset from %s", path)
    return dataset


def load_multiple_datasets(paths) -> Dict[str, TissueGraphDataset]:
    """Load several datasets at once, keyed by condition (or directory name as fallback).

    Parameters
    ----------
    paths:
        Iterable of directory paths, each written by :func:`save_dataset`.

    Returns
    -------
    Dict[str, TissueGraphDataset]
        Keys are the ``condition`` field from each dataset's metadata, falling
        back to the directory basename when the condition is empty.
    """
    result: Dict[str, TissueGraphDataset] = {}
    for p in paths:
        ds = load_dataset(p)
        key = ds.condition or Path(p).name
        result[key] = ds
    return result
