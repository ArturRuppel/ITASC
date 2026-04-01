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
_STR_DTYPE = h5py.string_dtype()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _set_opt_attr(group, name: str, value) -> None:
    """Write an attribute only if value is not None."""
    if value is not None:
        group.attrs[name] = value


def _get_opt_attr(group, name: str, default=None):
    return group.attrs[name] if name in group.attrs else default


def _save_ragged(group, name: str, arrays: List[np.ndarray]) -> None:
    """Save a list of variable-length 2-D arrays as flat+offsets datasets."""
    if not arrays:
        group.create_dataset(f"{name}_flat", data=np.empty((0, 2), dtype=np.float64))
        group.create_dataset(f"{name}_offsets", data=np.zeros(1, dtype=np.int64))
        return
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    parts = []
    for i, a in enumerate(arrays):
        a = np.asarray(a, dtype=np.float64)
        offsets[i + 1] = offsets[i] + len(a)
        if len(a):
            parts.append(a.reshape(len(a), -1))
    flat = np.concatenate(parts, axis=0) if parts else np.empty((0, 2), dtype=np.float64)
    group.create_dataset(f"{name}_flat", data=flat, compression="gzip", compression_opts=4)
    group.create_dataset(f"{name}_offsets", data=offsets)


def _load_ragged(group, name: str) -> List[np.ndarray]:
    flat = group[f"{name}_flat"][:]
    offsets = group[f"{name}_offsets"][:]
    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def _save_ragged_1d(group, name: str, arrays: List[np.ndarray]) -> None:
    """Save a list of variable-length 1-D arrays as flat+offsets datasets."""
    if not arrays:
        group.create_dataset(f"{name}_flat", data=np.empty(0, dtype=np.int64))
        group.create_dataset(f"{name}_offsets", data=np.zeros(1, dtype=np.int64))
        return
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    parts = []
    for i, a in enumerate(arrays):
        a = np.asarray(a, dtype=np.int64).ravel()
        offsets[i + 1] = offsets[i] + len(a)
        if len(a):
            parts.append(a)
    flat = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
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
    dataset: Optional[TissueGraphDataset] = None,
    pixel_size: Optional[float] = None,
    time_interval: Optional[float] = None,
    condition: str = "",
) -> None:
    """Save a CellFlow project to a single HDF5 file.

    Parameters
    ----------
    path:
        Output file path (e.g. ``"experiment.h5"``).
    labels:
        Optional label stack (T, H, W) or (H, W).
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
# Convenience aliases used by the dashboard / old callers
# ------------------------------------------------------------------

def save_dataset(dataset: TissueGraphDataset, path: Union[str, Path]) -> None:
    """Save just the analysis dataset (no labels) to an HDF5 file."""
    save_project(
        path,
        dataset=dataset,
        pixel_size=dataset.pixel_size,
        time_interval=dataset.time_interval,
        condition=dataset.condition or "",
    )


def load_dataset(path: Union[str, Path]) -> TissueGraphDataset:
    """Load a TissueGraphDataset from an HDF5 project file."""
    result = load_project(path)
    ds = result.get("dataset")
    if ds is None:
        raise ValueError(f"No analysis data found in {path}")
    return ds
