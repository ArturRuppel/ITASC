"""Save and load TissueGraphDataset to/from disk.

Format: a directory containing one NPZ file per tissue plus a metadata.json.

    my_dataset/
      metadata.json
      tissue_000.npz
      tissue_001.npz
      ...

NetworkX graphs are stored as edge lists (no pickle).
Variable-length arrays use a flat-data + offsets pattern.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import networkx as nx

from ..structures import (
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

_FORMAT_VERSION = "1.0"


# ------------------------------------------------------------------
# Ragged array helpers
# ------------------------------------------------------------------

def _serialize_ragged(arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Pack a list of variable-length arrays into (flat_data, offsets).

    Parameters
    ----------
    arrays : List[np.ndarray]
        Variable-length arrays to concatenate.

    Returns
    -------
    flat_data : np.ndarray
        Concatenated data.
    offsets : np.ndarray
        1D int64 array of length len(arrays)+1. Slice i is
        flat_data[offsets[i]:offsets[i+1]].
    """
    if not arrays:
        return np.empty((0,), dtype=np.float64), np.zeros(1, dtype=np.int64)

    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    for i, arr in enumerate(arrays):
        offsets[i + 1] = offsets[i] + len(arr)

    # Filter out empty arrays, then concatenate
    parts = []
    for a in arrays:
        a = np.asarray(a)
        if len(a) == 0:
            continue
        parts.append(a.reshape(len(a), -1))
    if not parts:
        return np.empty((0,), dtype=np.float64), offsets
    flat = np.concatenate(parts, axis=0)
    return flat, offsets


def _deserialize_ragged(
    flat_data: np.ndarray, offsets: np.ndarray
) -> List[np.ndarray]:
    """Unpack (flat_data, offsets) into a list of arrays."""
    result = []
    for i in range(len(offsets) - 1):
        result.append(flat_data[offsets[i]:offsets[i + 1]])
    return result


# ------------------------------------------------------------------
# Serialize / deserialize a single tissue
# ------------------------------------------------------------------

def _serialize_tissue(series: TissueGraphTimeSeries) -> Dict[str, np.ndarray]:
    """Convert a TissueGraphTimeSeries to a dict of arrays for np.savez."""
    data = {}

    frame_indices = np.array(series.frame_indices, dtype=np.int64)
    data["frame_indices"] = frame_indices

    # Per-frame data
    for f in frame_indices:
        frame = series.frames[int(f)]
        prefix = f"f{f}"

        # Graph edges
        edges = np.array(
            [tuple(sorted(e)) for e in frame.graph.edges()], dtype=np.int64
        )
        data[f"{prefix}_edges"] = edges if len(edges) > 0 else np.empty((0, 2), dtype=np.int64)

        edge_weights = np.array(
            [frame.graph.edges[e].get("length", 0.0) for e in frame.graph.edges()],
            dtype=np.float64,
        )
        data[f"{prefix}_edge_weights"] = edge_weights

        # Cells
        cell_ids = sorted(frame.cells.keys())
        n_cells = len(cell_ids)
        data[f"{prefix}_cell_ids"] = np.array(cell_ids, dtype=np.int64)
        data[f"{prefix}_cell_positions"] = np.array(
            [frame.cells[c].position for c in cell_ids], dtype=np.float64
        ).reshape(n_cells, 2)
        data[f"{prefix}_cell_areas"] = np.array(
            [frame.cells[c].area for c in cell_ids], dtype=np.float64
        )
        data[f"{prefix}_cell_perimeters"] = np.array(
            [frame.cells[c].perimeter for c in cell_ids], dtype=np.float64
        )
        data[f"{prefix}_cell_shape_indices"] = np.array(
            [frame.cells[c].shape_index for c in cell_ids], dtype=np.float64
        )
        data[f"{prefix}_cell_num_neighbors"] = np.array(
            [frame.cells[c].num_neighbors for c in cell_ids], dtype=np.int64
        )
        data[f"{prefix}_cell_track_ids"] = np.array(
            [frame.cells[c].track_id if frame.cells[c].track_id is not None else -1
             for c in cell_ids],
            dtype=np.int64,
        )

        # Cell vertices (ragged)
        vert_arrays = []
        has_vertices = False
        for c in cell_ids:
            v = frame.cells[c].vertices
            if v is not None:
                vert_arrays.append(v)
                has_vertices = True
            else:
                vert_arrays.append(np.empty((0, 2), dtype=np.float64))
        if has_vertices:
            flat, offsets = _serialize_ragged(vert_arrays)
            data[f"{prefix}_cell_vert_flat"] = flat
            data[f"{prefix}_cell_vert_offsets"] = offsets

        # Junctions
        junction_keys = sorted(frame.junctions.keys(), key=lambda fs: tuple(sorted(fs)))
        n_junctions = len(junction_keys)
        data[f"{prefix}_junc_pairs"] = np.array(
            [tuple(sorted(k)) for k in junction_keys], dtype=np.int64
        ).reshape(n_junctions, 2) if n_junctions > 0 else np.empty((0, 2), dtype=np.int64)
        data[f"{prefix}_junc_lengths"] = np.array(
            [frame.junctions[k].length for k in junction_keys], dtype=np.float64
        )
        data[f"{prefix}_junc_midpoints"] = np.array(
            [frame.junctions[k].midpoint for k in junction_keys], dtype=np.float64
        ).reshape(n_junctions, 2) if n_junctions > 0 else np.empty((0, 2), dtype=np.float64)

        # Junction coordinates (ragged)
        junc_coord_arrays = [frame.junctions[k].coordinates for k in junction_keys]
        if junc_coord_arrays:
            flat, offsets = _serialize_ragged(junc_coord_arrays)
            data[f"{prefix}_junc_coord_flat"] = flat
            data[f"{prefix}_junc_coord_offsets"] = offsets

        # Junction optional fields
        tensions = np.array(
            [frame.junctions[k].tension if frame.junctions[k].tension is not None else np.nan
             for k in junction_keys],
            dtype=np.float64,
        )
        if not np.all(np.isnan(tensions)):
            data[f"{prefix}_junc_tensions"] = tensions

        stresses = np.array(
            [frame.junctions[k].normal_stress if frame.junctions[k].normal_stress is not None else np.nan
             for k in junction_keys],
            dtype=np.float64,
        )
        if not np.all(np.isnan(stresses)):
            data[f"{prefix}_junc_stresses"] = stresses

        # Input type
        data[f"{prefix}_input_type"] = np.array(
            [frame.input_type.value], dtype="U32"
        )

    # T1 events
    n_t1 = len(series.t1_events)
    if n_t1 > 0:
        data["t1_frames"] = np.array(
            [e.frame for e in series.t1_events], dtype=np.int64
        )
        data["t1_losing_pairs"] = np.array(
            [e.losing_pair for e in series.t1_events], dtype=np.int64
        ).reshape(n_t1, 2)
        data["t1_gaining_pairs"] = np.array(
            [e.gaining_pair for e in series.t1_events], dtype=np.int64
        ).reshape(n_t1, 2)
        data["t1_locations"] = np.array(
            [e.location for e in series.t1_events], dtype=np.float64
        ).reshape(n_t1, 2)
        # all_cells (ragged sets)
        all_cells_arrays = [
            np.array(sorted(e.all_cells), dtype=np.int64) for e in series.t1_events
        ]
        flat, offsets = _serialize_ragged(all_cells_arrays)
        data["t1_all_cells_flat"] = flat
        data["t1_all_cells_offsets"] = offsets

    # Edge trajectories
    n_traj = len(series.edge_trajectories)
    if n_traj > 0:
        traj_ids = sorted(series.edge_trajectories.keys())
        data["traj_ids"] = np.array(traj_ids, dtype=np.int64)

        # Per-trajectory: frames, signed_lengths (ragged), cell_pairs (ragged), coordinates (ragged)
        traj_frame_arrays = []
        traj_length_arrays = []
        traj_pair_arrays = []
        traj_coord_arrays = []
        traj_t1_indices = []  # which T1 events belong to which trajectory

        for tid in traj_ids:
            traj = series.edge_trajectories[tid]
            traj_frame_arrays.append(np.array(traj.frames, dtype=np.int64))
            traj_length_arrays.append(np.array(traj.signed_lengths, dtype=np.float64))
            traj_pair_arrays.append(
                np.array(traj.cell_pairs, dtype=np.int64).reshape(-1, 2)
                if traj.cell_pairs else np.empty((0, 2), dtype=np.int64)
            )
            # Coordinates are list of Nx2 arrays — double-ragged. Flatten all into one.
            if traj.coordinates:
                flat_coords, coord_offsets = _serialize_ragged(traj.coordinates)
                traj_coord_arrays.append((flat_coords, coord_offsets))
            else:
                traj_coord_arrays.append((
                    np.empty((0, 2), dtype=np.float64),
                    np.zeros(1, dtype=np.int64),
                ))

            # T1 event indices for this trajectory (indices into series.t1_events)
            t1_idxs = []
            for evt in traj.t1_events:
                for i, se in enumerate(series.t1_events):
                    if se is evt:
                        t1_idxs.append(i)
                        break
            traj_t1_indices.append(np.array(t1_idxs, dtype=np.int64))

        # Serialize ragged trajectory data
        flat, offsets = _serialize_ragged(traj_frame_arrays)
        data["traj_frames_flat"] = flat
        data["traj_frames_offsets"] = offsets

        flat, offsets = _serialize_ragged(traj_length_arrays)
        data["traj_lengths_flat"] = flat
        data["traj_lengths_offsets"] = offsets

        flat, offsets = _serialize_ragged(traj_pair_arrays)
        data["traj_pairs_flat"] = flat
        data["traj_pairs_offsets"] = offsets

        # Trajectory coordinates: store per-trajectory coord flat+offsets
        for i, tid in enumerate(traj_ids):
            cf, co = traj_coord_arrays[i]
            data[f"traj_{tid}_coord_flat"] = cf
            data[f"traj_{tid}_coord_offsets"] = co

        # T1 event references per trajectory
        flat, offsets = _serialize_ragged(traj_t1_indices)
        data["traj_t1_refs_flat"] = flat
        data["traj_t1_refs_offsets"] = offsets

    return data


def _deserialize_tissue(npz) -> TissueGraphTimeSeries:
    """Reconstruct a TissueGraphTimeSeries from NPZ data."""
    frame_indices = npz["frame_indices"].astype(np.int64)

    # Reconstruct T1 events first (trajectories reference them)
    t1_events = []
    if "t1_frames" in npz:
        t1_frames = npz["t1_frames"]
        t1_losing = npz["t1_losing_pairs"]
        t1_gaining = npz["t1_gaining_pairs"]
        t1_locs = npz["t1_locations"]
        all_cells_list = _deserialize_ragged(
            npz["t1_all_cells_flat"], npz["t1_all_cells_offsets"]
        )
        for i in range(len(t1_frames)):
            t1_events.append(T1Event(
                frame=int(t1_frames[i]),
                losing_pair=tuple(t1_losing[i].tolist()),
                gaining_pair=tuple(t1_gaining[i].tolist()),
                location=t1_locs[i].copy(),
                all_cells=set(all_cells_list[i].astype(np.int64).ravel().tolist()),
            ))

    # Reconstruct frames
    frames = {}
    for f in frame_indices:
        f = int(f)
        prefix = f"f{f}"

        # Graph
        edges = npz[f"{prefix}_edges"]
        edge_weights = npz[f"{prefix}_edge_weights"]
        graph = nx.Graph()

        cell_ids = npz[f"{prefix}_cell_ids"].astype(np.int64)
        for cid in cell_ids:
            graph.add_node(int(cid))

        for j in range(len(edges)):
            u, v = int(edges[j, 0]), int(edges[j, 1])
            graph.add_edge(u, v, length=float(edge_weights[j]))

        # Cells
        positions = npz[f"{prefix}_cell_positions"]
        areas = npz[f"{prefix}_cell_areas"]
        perimeters = npz[f"{prefix}_cell_perimeters"]
        shape_indices = npz[f"{prefix}_cell_shape_indices"]
        num_neighbors = npz[f"{prefix}_cell_num_neighbors"]
        track_ids = npz[f"{prefix}_cell_track_ids"]

        # Cell vertices
        has_verts = f"{prefix}_cell_vert_flat" in npz
        if has_verts:
            vert_list = _deserialize_ragged(
                npz[f"{prefix}_cell_vert_flat"],
                npz[f"{prefix}_cell_vert_offsets"],
            )
        else:
            vert_list = [None] * len(cell_ids)

        cells = {}
        for i, cid in enumerate(cell_ids):
            cid = int(cid)
            tid = int(track_ids[i])
            verts = vert_list[i] if has_verts else None
            if verts is not None and len(verts) == 0:
                verts = None
            cells[cid] = CellData(
                cell_id=cid,
                position=positions[i].copy(),
                area=float(areas[i]),
                perimeter=float(perimeters[i]),
                shape_index=float(shape_indices[i]),
                num_neighbors=int(num_neighbors[i]),
                track_id=tid if tid != -1 else None,
                vertices=verts,
            )

        # Junctions
        junc_pairs = npz[f"{prefix}_junc_pairs"]
        junc_lengths = npz[f"{prefix}_junc_lengths"]
        junc_midpoints = npz[f"{prefix}_junc_midpoints"]

        has_junc_coords = f"{prefix}_junc_coord_flat" in npz
        if has_junc_coords:
            junc_coord_list = _deserialize_ragged(
                npz[f"{prefix}_junc_coord_flat"],
                npz[f"{prefix}_junc_coord_offsets"],
            )
        else:
            junc_coord_list = [np.empty((0, 2))] * len(junc_pairs)

        # Optional junction fields
        has_tensions = f"{prefix}_junc_tensions" in npz
        has_stresses = f"{prefix}_junc_stresses" in npz
        tensions = npz[f"{prefix}_junc_tensions"] if has_tensions else None
        stresses = npz[f"{prefix}_junc_stresses"] if has_stresses else None

        junctions = {}
        for j in range(len(junc_pairs)):
            pair = (int(junc_pairs[j, 0]), int(junc_pairs[j, 1]))
            key = frozenset(pair)
            tension = float(tensions[j]) if has_tensions and not np.isnan(tensions[j]) else None
            stress = float(stresses[j]) if has_stresses and not np.isnan(stresses[j]) else None
            junctions[key] = JunctionData(
                cell_pair=pair,
                length=float(junc_lengths[j]),
                coordinates=junc_coord_list[j].copy(),
                midpoint=junc_midpoints[j].copy(),
                tension=tension,
                normal_stress=stress,
            )

        input_type_str = str(npz[f"{prefix}_input_type"][0])
        input_type = InputType(input_type_str)

        frames[f] = TissueGraphFrame(
            frame=f,
            graph=graph,
            cells=cells,
            junctions=junctions,
            input_type=input_type,
        )

    # Reconstruct edge trajectories
    edge_trajectories = {}
    if "traj_ids" in npz:
        traj_ids = npz["traj_ids"].astype(np.int64)
        traj_frames_list = _deserialize_ragged(
            npz["traj_frames_flat"], npz["traj_frames_offsets"]
        )
        traj_lengths_list = _deserialize_ragged(
            npz["traj_lengths_flat"], npz["traj_lengths_offsets"]
        )
        traj_pairs_list = _deserialize_ragged(
            npz["traj_pairs_flat"], npz["traj_pairs_offsets"]
        )
        traj_t1_refs_list = _deserialize_ragged(
            npz["traj_t1_refs_flat"], npz["traj_t1_refs_offsets"]
        )

        for i, tid in enumerate(traj_ids):
            tid = int(tid)

            # Coordinates
            coord_flat_key = f"traj_{tid}_coord_flat"
            coord_off_key = f"traj_{tid}_coord_offsets"
            if coord_flat_key in npz:
                coord_list = _deserialize_ragged(
                    npz[coord_flat_key], npz[coord_off_key]
                )
            else:
                coord_list = []

            # Cell pairs
            pairs_arr = traj_pairs_list[i]
            cell_pairs = [
                (int(pairs_arr[j, 0]), int(pairs_arr[j, 1]))
                for j in range(len(pairs_arr))
            ] if len(pairs_arr) > 0 else []

            # T1 event references
            t1_refs = traj_t1_refs_list[i].astype(np.int64).ravel()
            traj_t1 = [t1_events[int(idx)] for idx in t1_refs]

            edge_trajectories[tid] = EdgeTrajectory(
                trajectory_id=tid,
                frames=traj_frames_list[i].astype(np.int64).ravel().tolist(),
                cell_pairs=cell_pairs,
                signed_lengths=traj_lengths_list[i].astype(np.float64).ravel().tolist(),
                coordinates=coord_list,
                t1_events=traj_t1,
            )

    # Determine series-level input type from first frame
    series_input_type = None
    if frames:
        first_frame = frames[sorted(frames.keys())[0]]
        series_input_type = first_frame.input_type

    return TissueGraphTimeSeries(
        frames=frames,
        edge_trajectories=edge_trajectories,
        t1_events=t1_events,
        input_type=series_input_type,
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def save_dataset(dataset: TissueGraphDataset, path: Union[str, Path]) -> None:
    """Save a TissueGraphDataset to a directory.

    Creates a directory containing metadata.json and one NPZ file per tissue.

    Parameters
    ----------
    dataset : TissueGraphDataset
        The dataset to save.
    path : str or Path
        Directory path to save to. Created if it does not exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "version": _FORMAT_VERSION,
        "condition": dataset.condition,
        "pixel_size": dataset.pixel_size,
        "time_interval": dataset.time_interval,
        "input_type": dataset.input_type.value if dataset.input_type else None,
        "n_tissues": dataset.n_tissues,
        "tissue_ids": dataset.tissue_ids,
        "metadata": dataset.metadata,
        "created": datetime.now().isoformat(),
    }
    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save each tissue
    for tid in dataset.tissue_ids:
        series = dataset.tissues[tid]
        arrays = _serialize_tissue(series)
        np.savez_compressed(path / f"tissue_{tid:03d}.npz", **arrays)

    logger.info(f"Saved dataset with {dataset.n_tissues} tissues to {path}")


def load_dataset(path: Union[str, Path]) -> TissueGraphDataset:
    """Load a TissueGraphDataset from a directory.

    Parameters
    ----------
    path : str or Path
        Directory containing metadata.json and tissue NPZ files.

    Returns
    -------
    TissueGraphDataset
        The loaded dataset.
    """
    path = Path(path)

    with open(path / "metadata.json") as f:
        meta = json.load(f)

    dataset = TissueGraphDataset(
        condition=meta.get("condition", ""),
        pixel_size=meta.get("pixel_size"),
        time_interval=meta.get("time_interval"),
        input_type=InputType(meta["input_type"]) if meta.get("input_type") else None,
        metadata=meta.get("metadata", {}),
    )

    for tid in meta["tissue_ids"]:
        npz_path = path / f"tissue_{tid:03d}.npz"
        with np.load(npz_path, allow_pickle=False) as npz:
            series = _deserialize_tissue(npz)
            series.pixel_size = meta.get("pixel_size")
            series.time_interval = meta.get("time_interval")
        dataset.tissues[tid] = series

    logger.info(f"Loaded dataset with {dataset.n_tissues} tissues from {path}")
    return dataset


def load_multiple_datasets(
    paths: List[Union[str, Path]],
) -> Dict[str, TissueGraphDataset]:
    """Load multiple datasets, keyed by condition name or directory name.

    Parameters
    ----------
    paths : List[str or Path]
        List of dataset directory paths.

    Returns
    -------
    Dict[str, TissueGraphDataset]
        Mapping of condition/directory name to dataset.
    """
    datasets = {}
    for p in paths:
        p = Path(p)
        ds = load_dataset(p)
        key = ds.condition if ds.condition else p.name
        datasets[key] = ds
    return datasets
