"""Data query API for TissueGraph data.

Provides a clean Python interface for querying tissue graph data,
returning pandas DataFrames. Accepts TissueGraphTimeSeries,
TissueGraphDataset, or a path (str/Path) to a saved dataset.
"""
import bisect
from collections import defaultdict
from pathlib import Path
from typing import Collection, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..structures import (
    TissueGraphDataset,
    TissueGraphTimeSeries,
)

Source = Union[TissueGraphTimeSeries, TissueGraphDataset, str, Path]


def _resolve_source(
    source: Source,
) -> Tuple[List[Tuple[int, TissueGraphTimeSeries]], bool]:
    """Resolve *source* to a list of (tissue_id, series) pairs.

    Returns
    -------
    items : list of (tissue_id, TissueGraphTimeSeries)
    multi : bool
        True when source contains multiple tissues (dataset).
    """
    if isinstance(source, TissueGraphTimeSeries):
        return [(0, source)], False
    if isinstance(source, TissueGraphDataset):
        items = [(tid, source.tissues[tid]) for tid in source.tissue_ids]
        return items, len(items) > 1
    # str / Path -> load from disk
    from .io import load_dataset

    ds = load_dataset(Path(source))
    items = [(tid, ds.tissues[tid]) for tid in ds.tissue_ids]
    return items, len(items) > 1


# ------------------------------------------------------------------
# get_cells
# ------------------------------------------------------------------

def get_cells(
    source: Source,
    *,
    frames: Optional[Collection[int]] = None,
    min_neighbors: Optional[int] = None,
    max_neighbors: Optional[int] = None,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
    track_ids: Optional[Collection[int]] = None,
    has_tracking: Optional[bool] = None,
) -> pd.DataFrame:
    """Return a DataFrame of cell properties.

    Columns: [tissue_id], frame, cell_id, track_id, y, x, area,
    perimeter, shape_index, num_neighbors, pressure, speed
    """
    items, multi = _resolve_source(source)
    track_set = set(track_ids) if track_ids is not None else None
    rows: list = []

    for tid, series in items:
        for fi in series.frame_indices:
            if frames is not None and fi not in frames:
                continue
            frame = series.frames[fi]
            for cell in frame.cells.values():
                if min_neighbors is not None and cell.num_neighbors < min_neighbors:
                    continue
                if max_neighbors is not None and cell.num_neighbors > max_neighbors:
                    continue
                if min_area is not None and cell.area < min_area:
                    continue
                if max_area is not None and cell.area > max_area:
                    continue
                if track_set is not None and cell.track_id not in track_set:
                    continue
                if has_tracking is True and cell.track_id is None:
                    continue
                if has_tracking is False and cell.track_id is not None:
                    continue
                row = {
                    "frame": fi,
                    "cell_id": cell.cell_id,
                    "track_id": cell.track_id,
                    "y": cell.position[0],
                    "x": cell.position[1],
                    "area": cell.area,
                    "perimeter": cell.perimeter,
                    "shape_index": cell.shape_index,
                    "num_neighbors": cell.num_neighbors,
                    "pressure": cell.pressure,
                    "speed": cell.instantaneous_speed,
                }
                if multi:
                    row["tissue_id"] = tid
                rows.append(row)

    df = pd.DataFrame(rows)
    if multi and not df.empty:
        cols = ["tissue_id"] + [c for c in df.columns if c != "tissue_id"]
        df = df[cols]
    return df


# ------------------------------------------------------------------
# get_junctions
# ------------------------------------------------------------------

def get_junctions(
    source: Source,
    *,
    frames: Optional[Collection[int]] = None,
    tags: Optional[Collection[str]] = None,
    exclude_tags: Optional[Collection[str]] = None,
    min_length: Optional[float] = None,
    max_length: Optional[float] = None,
    cell_ids: Optional[Collection[int]] = None,
) -> pd.DataFrame:
    """Return a DataFrame of junction properties.

    Columns: [tissue_id], frame, cell_a, cell_b, length,
    midpoint_y, midpoint_x, tension, normal_stress, tags
    """
    items, multi = _resolve_source(source)
    tag_set = set(tags) if tags is not None else None
    excl_set = set(exclude_tags) if exclude_tags is not None else None
    cell_set = set(cell_ids) if cell_ids is not None else None
    rows: list = []

    for tid, series in items:
        for fi in series.frame_indices:
            if frames is not None and fi not in frames:
                continue
            frame = series.frames[fi]
            for junc in frame.junctions.values():
                if min_length is not None and junc.length < min_length:
                    continue
                if max_length is not None and junc.length > max_length:
                    continue
                if tag_set is not None and not (junc.tags & tag_set):
                    continue
                if excl_set is not None and (junc.tags & excl_set):
                    continue
                if cell_set is not None and not (set(junc.cell_pair) & cell_set):
                    continue
                row = {
                    "frame": fi,
                    "cell_a": junc.cell_pair[0],
                    "cell_b": junc.cell_pair[1],
                    "length": junc.length,
                    "midpoint_y": junc.midpoint[0],
                    "midpoint_x": junc.midpoint[1],
                    "tension": junc.tension,
                    "normal_stress": junc.normal_stress,
                    "tags": ",".join(sorted(junc.tags)) if junc.tags else "",
                }
                if multi:
                    row["tissue_id"] = tid
                rows.append(row)

    df = pd.DataFrame(rows)
    if multi and not df.empty:
        cols = ["tissue_id"] + [c for c in df.columns if c != "tissue_id"]
        df = df[cols]
    return df


# ------------------------------------------------------------------
# get_trajectories
# ------------------------------------------------------------------

def get_trajectories(
    source: Source,
    *,
    tags: Optional[Collection[str]] = None,
    exclude_tags: Optional[Collection[str]] = None,
    min_frames: Optional[int] = None,
    min_completeness: Optional[float] = None,
    has_t1: Optional[bool] = None,
    trajectory_ids: Optional[Collection[int]] = None,
) -> pd.DataFrame:
    """Return one row per trajectory-frame (long format).

    Columns: [tissue_id], trajectory_id, frame, cell_a, cell_b,
    signed_length, abs_length, tags, name, n_t1_events
    """
    items, multi = _resolve_source(source)
    tag_set = set(tags) if tags is not None else None
    excl_set = set(exclude_tags) if exclude_tags is not None else None
    traj_set = set(trajectory_ids) if trajectory_ids is not None else None
    rows: list = []

    for tid, series in items:
        n_series_frames = series.num_frames
        for traj in series.edge_trajectories.values():
            if traj_set is not None and traj.trajectory_id not in traj_set:
                continue
            if tag_set is not None and not (traj.tags & tag_set):
                continue
            if excl_set is not None and (traj.tags & excl_set):
                continue
            if min_frames is not None and len(traj.frames) < min_frames:
                continue
            if min_completeness is not None and n_series_frames > 0:
                completeness = len(traj.frames) / n_series_frames
                if completeness < min_completeness:
                    continue
            if has_t1 is True and len(traj.t1_events) == 0:
                continue
            if has_t1 is False and len(traj.t1_events) > 0:
                continue

            n_t1 = len(traj.t1_events)
            tag_str = ",".join(sorted(traj.tags)) if traj.tags else ""
            for i, frame_idx in enumerate(traj.frames):
                pair = traj.cell_pairs[i]
                sl = traj.signed_lengths[i]
                row = {
                    "trajectory_id": traj.trajectory_id,
                    "frame": frame_idx,
                    "cell_a": pair[0],
                    "cell_b": pair[1],
                    "signed_length": sl,
                    "abs_length": abs(sl),
                    "tags": tag_str,
                    "name": traj.name or "",
                    "n_t1_events": n_t1,
                }
                if multi:
                    row["tissue_id"] = tid
                rows.append(row)

    df = pd.DataFrame(rows)
    if multi and not df.empty:
        cols = ["tissue_id"] + [c for c in df.columns if c != "tissue_id"]
        df = df[cols]
    return df


# ------------------------------------------------------------------
# get_trajectory_summary
# ------------------------------------------------------------------

def get_trajectory_summary(
    source: Source,
    *,
    tags: Optional[Collection[str]] = None,
    exclude_tags: Optional[Collection[str]] = None,
    min_frames: Optional[int] = None,
    has_t1: Optional[bool] = None,
) -> pd.DataFrame:
    """Return one row per trajectory (wide format).

    Columns: [tissue_id], trajectory_id, n_frames, first_frame,
    last_frame, mean_abs_length, n_t1_events, tags, name
    """
    items, multi = _resolve_source(source)
    tag_set = set(tags) if tags is not None else None
    excl_set = set(exclude_tags) if exclude_tags is not None else None
    rows: list = []

    for tid, series in items:
        for traj in series.edge_trajectories.values():
            if tag_set is not None and not (traj.tags & tag_set):
                continue
            if excl_set is not None and (traj.tags & excl_set):
                continue
            if min_frames is not None and len(traj.frames) < min_frames:
                continue
            n_t1 = len(traj.t1_events)
            if has_t1 is True and n_t1 == 0:
                continue
            if has_t1 is False and n_t1 > 0:
                continue

            abs_lengths = [abs(sl) for sl in traj.signed_lengths]
            row = {
                "trajectory_id": traj.trajectory_id,
                "n_frames": len(traj.frames),
                "first_frame": min(traj.frames),
                "last_frame": max(traj.frames),
                "mean_abs_length": float(np.mean(abs_lengths)) if abs_lengths else 0.0,
                "n_t1_events": n_t1,
                "tags": ",".join(sorted(traj.tags)) if traj.tags else "",
                "name": traj.name or "",
            }
            if multi:
                row["tissue_id"] = tid
            rows.append(row)

    df = pd.DataFrame(rows)
    if multi and not df.empty:
        cols = ["tissue_id"] + [c for c in df.columns if c != "tissue_id"]
        df = df[cols]
    return df


# ------------------------------------------------------------------
# get_t1_events
# ------------------------------------------------------------------

def get_t1_events(
    source: Source,
    *,
    frames: Optional[Collection[int]] = None,
    cell_ids: Optional[Collection[int]] = None,
) -> pd.DataFrame:
    """Return a DataFrame of T1 transition events.

    Columns: [tissue_id], frame, losing_a, losing_b, gaining_a,
    gaining_b, y, x
    """
    items, multi = _resolve_source(source)
    cell_set = set(cell_ids) if cell_ids is not None else None
    rows: list = []

    for tid, series in items:
        for evt in series.t1_events:
            if frames is not None and evt.frame not in frames:
                continue
            if cell_set is not None and not (evt.all_cells & cell_set):
                continue
            row = {
                "frame": evt.frame,
                "losing_a": evt.losing_pair[0],
                "losing_b": evt.losing_pair[1],
                "gaining_a": evt.gaining_pair[0],
                "gaining_b": evt.gaining_pair[1],
                "y": evt.location[0],
                "x": evt.location[1],
            }
            if multi:
                row["tissue_id"] = tid
            rows.append(row)

    df = pd.DataFrame(rows)
    if multi and not df.empty:
        cols = ["tissue_id"] + [c for c in df.columns if c != "tissue_id"]
        df = df[cols]
    return df


# ------------------------------------------------------------------
# get_time_since_last_t1
# ------------------------------------------------------------------

def get_time_since_last_t1(
    source: Source,
    *,
    frames: Optional[Collection[int]] = None,
    cell_ids: Optional[Collection[int]] = None,
) -> pd.DataFrame:
    """Return time since last T1 event for each cell at each frame.

    Columns: [tissue_id], frame, cell_id, frames_since_last_t1,
    time_since_last_t1
    """
    items, multi = _resolve_source(source)
    cell_set = set(cell_ids) if cell_ids is not None else None
    rows: list = []

    for tid, series in items:
        # Pre-build {cell_id: [sorted T1 frames]}
        cell_t1_frames: dict[int, list[int]] = defaultdict(list)
        for evt in series.t1_events:
            for cid in evt.all_cells:
                cell_t1_frames[cid].append(evt.frame)
        for cid in cell_t1_frames:
            cell_t1_frames[cid].sort()

        dt = series.time_interval

        for fi in series.frame_indices:
            if frames is not None and fi not in frames:
                continue
            frame = series.frames[fi]
            for cell in frame.cells.values():
                cid = cell.cell_id
                if cell_set is not None and cid not in cell_set:
                    continue

                t1_list = cell_t1_frames.get(cid)
                if not t1_list:
                    frames_since = None
                    time_since = None
                else:
                    # Binary search for most recent T1 at or before fi
                    idx = bisect.bisect_right(t1_list, fi) - 1
                    if idx < 0:
                        frames_since = None
                        time_since = None
                    else:
                        frames_since = fi - t1_list[idx]
                        time_since = frames_since * dt if dt is not None else None

                row = {
                    "frame": fi,
                    "cell_id": cid,
                    "frames_since_last_t1": frames_since,
                    "time_since_last_t1": time_since,
                }
                if multi:
                    row["tissue_id"] = tid
                rows.append(row)

    df = pd.DataFrame(rows)
    if multi and not df.empty:
        cols = ["tissue_id"] + [c for c in df.columns if c != "tissue_id"]
        df = df[cols]
    return df


# ------------------------------------------------------------------
# get_cell_history
# ------------------------------------------------------------------

def get_cell_history(
    source: Source,
    track_id: int,
    *,
    tissue_id: int = 0,
) -> pd.DataFrame:
    """Convenience: get_cells filtered to one track, sorted by frame."""
    items, multi = _resolve_source(source)

    # Find the right series
    series = None
    for tid, s in items:
        if tid == tissue_id:
            series = s
            break
    if series is None:
        return pd.DataFrame()

    df = get_cells(series, track_ids={track_id})
    if not df.empty:
        df = df.sort_values("frame").reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# get_neighbor_history
# ------------------------------------------------------------------

def get_neighbor_history(
    source: Source,
    cell_id: int,
    *,
    tissue_id: int = 0,
    frames: Optional[Collection[int]] = None,
) -> pd.DataFrame:
    """Return neighbors of a cell over time from the networkx graph.

    Columns: frame, cell_id, neighbor_id
    """
    items, _ = _resolve_source(source)

    series = None
    for tid, s in items:
        if tid == tissue_id:
            series = s
            break
    if series is None:
        return pd.DataFrame()

    rows: list = []
    for fi in series.frame_indices:
        if frames is not None and fi not in frames:
            continue
        graph = series.frames[fi].graph
        if cell_id not in graph:
            continue
        for nbr in graph.neighbors(cell_id):
            rows.append({"frame": fi, "cell_id": cell_id, "neighbor_id": nbr})

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# get_t1_rate
# ------------------------------------------------------------------

def get_t1_rate(
    source: Source,
    *,
    window: int = 1,
) -> pd.DataFrame:
    """Return the T1 event rate per frame.

    Columns: [tissue_id], frame, n_t1_events, t1_rate
    """
    items, multi = _resolve_source(source)
    rows: list = []

    for tid, series in items:
        # Count T1 events per frame
        t1_counts: dict[int, int] = defaultdict(int)
        for evt in series.t1_events:
            t1_counts[evt.frame] += 1

        for fi in series.frame_indices:
            # Sum counts over the window [fi - window + 1, fi]
            total = sum(t1_counts.get(fi - w, 0) for w in range(window))
            rate = total / window
            row = {
                "frame": fi,
                "n_t1_events": t1_counts.get(fi, 0),
                "t1_rate": rate,
            }
            if multi:
                row["tissue_id"] = tid
            rows.append(row)

    df = pd.DataFrame(rows)
    if multi and not df.empty:
        cols = ["tissue_id"] + [c for c in df.columns if c != "tissue_id"]
        df = df[cols]
    return df
