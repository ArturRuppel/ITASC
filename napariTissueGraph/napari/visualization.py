"""Napari layer rendering for tissue graphs."""
import numpy as np
from typing import Dict, List, Optional, Tuple

from ..structures import T1Event, TissueGraphFrame, TissueGraphTimeSeries


def build_all_junction_lines(series: TissueGraphTimeSeries) -> Tuple[List[np.ndarray], np.ndarray]:
    """Build junction line data for all frames at once.

    Each line gets a frame coordinate prepended so napari's dim slider works natively.

    Returns:
        (lines, colors) where lines is a list of Nx3 arrays (frame, y, x)
        and colors is an Nx4 RGBA array.
    """
    lines = []
    colors = []

    # Collect all junction lengths across all frames for global normalization
    all_lengths = []
    for frame in series.frames.values():
        for jd in frame.junctions.values():
            if len(jd.coordinates) >= 2:
                all_lengths.append(jd.length)

    if not all_lengths:
        return [], np.empty((0, 4))

    vmin, vmax = min(all_lengths), max(all_lengths)

    for frame_idx in series.frame_indices:
        frame = series.frames[frame_idx]
        for jd in frame.junctions.values():
            if len(jd.coordinates) < 2:
                continue

            # Prepend frame index to coordinates: (y, x) -> (frame, y, x)
            coords = jd.coordinates.astype(float)
            frame_col = np.full((len(coords), 1), float(frame_idx))
            lines.append(np.hstack([frame_col, coords]))

            # Color by length (globally normalized)
            if vmax > vmin:
                normed = (jd.length - vmin) / (vmax - vmin)
            else:
                normed = 0.5
            colors.append([normed, 0.0, 1.0 - normed, 1.0])

    return lines, np.array(colors)


def build_all_centroids(series: TissueGraphTimeSeries) -> np.ndarray:
    """Build centroid positions for all frames at once.

    Returns Nx3 array of (frame, y, x) positions.
    """
    positions = []
    for frame_idx in series.frame_indices:
        frame = series.frames[frame_idx]
        for cd in frame.cells.values():
            positions.append([float(frame_idx), cd.position[0], cd.position[1]])

    if not positions:
        return np.empty((0, 3))
    return np.array(positions)


def build_t1_markers(t1_events: List[T1Event]) -> np.ndarray:
    """Build T1 event marker positions for all frames at once.

    Returns Nx3 array of (frame, y, x) positions so napari's
    dim slider handles frame filtering natively.
    """
    if not t1_events:
        return np.empty((0, 3))

    positions = []
    for event in t1_events:
        positions.append([
            float(event.frame),
            event.location[0],
            event.location[1],
        ])
    return np.array(positions)


# ------------------------------------------------------------------
# Stage 2 QC: Tracking visualization
# ------------------------------------------------------------------

_TRACK_COLORMAP = [
    [0.12, 0.47, 0.71, 1.0],  # blue
    [1.00, 0.50, 0.05, 1.0],  # orange
    [0.17, 0.63, 0.17, 1.0],  # green
    [0.84, 0.15, 0.16, 1.0],  # red
    [0.58, 0.40, 0.74, 1.0],  # purple
    [0.55, 0.34, 0.29, 1.0],  # brown
    [0.89, 0.47, 0.76, 1.0],  # pink
    [0.74, 0.74, 0.13, 1.0],  # olive
    [0.09, 0.75, 0.81, 1.0],  # cyan
    [0.50, 0.50, 0.50, 1.0],  # gray
]

_UNTRACKED_COLOR = np.array([0.5, 0.5, 0.5, 0.5])


def build_tracked_centroids(
    series: TissueGraphTimeSeries,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """Build centroid positions colored by track_id.

    Returns:
        (positions, colors, track_color_map) where:
        - positions: Nx3 (frame, y, x)
        - colors: Nx4 RGBA
        - track_color_map: dict track_id -> RGBA color
    """
    positions = []
    track_ids_per_point = []

    for frame_idx in series.frame_indices:
        frame = series.frames[frame_idx]
        for cd in frame.cells.values():
            positions.append([float(frame_idx), cd.position[0], cd.position[1]])
            track_ids_per_point.append(cd.track_id)

    if not positions:
        return np.empty((0, 3)), np.empty((0, 4)), {}

    positions = np.array(positions)

    # Build color map: unique track_ids -> cycled colors
    unique_tracks = sorted(set(t for t in track_ids_per_point if t is not None))
    track_color_map = {}
    for i, tid in enumerate(unique_tracks):
        track_color_map[tid] = np.array(_TRACK_COLORMAP[i % len(_TRACK_COLORMAP)])

    colors = np.empty((len(positions), 4))
    for i, tid in enumerate(track_ids_per_point):
        if tid is None:
            colors[i] = _UNTRACKED_COLOR
        else:
            colors[i] = track_color_map[tid]

    return positions, colors, track_color_map


def build_track_breaks(
    series: TissueGraphTimeSeries,
) -> Tuple[np.ndarray, List[str]]:
    """Find tracks that start after first frame or end before last frame.

    Returns:
        (positions, types) where:
        - positions: Nx3 (frame, y, x)
        - types: list of "birth" or "death" strings
    """
    if series.num_frames < 2:
        return np.empty((0, 3)), []

    frame_indices = series.frame_indices
    first_frame = frame_indices[0]
    last_frame = frame_indices[-1]

    # Collect first and last appearance of each track
    track_first: Dict[int, Tuple[int, np.ndarray]] = {}  # track_id -> (frame, position)
    track_last: Dict[int, Tuple[int, np.ndarray]] = {}

    for frame_idx in frame_indices:
        frame = series.frames[frame_idx]
        for cd in frame.cells.values():
            if cd.track_id is None:
                continue
            if cd.track_id not in track_first or frame_idx < track_first[cd.track_id][0]:
                track_first[cd.track_id] = (frame_idx, cd.position)
            if cd.track_id not in track_last or frame_idx > track_last[cd.track_id][0]:
                track_last[cd.track_id] = (frame_idx, cd.position)

    positions = []
    types = []

    for tid, (f, pos) in track_first.items():
        if f > first_frame:
            positions.append([float(f), pos[0], pos[1]])
            types.append("birth")

    for tid, (f, pos) in track_last.items():
        if f < last_frame:
            positions.append([float(f), pos[0], pos[1]])
            types.append("death")

    if not positions:
        return np.empty((0, 3)), []

    return np.array(positions), types


# ------------------------------------------------------------------
# Stage 3 QC: Trajectory visualization
# ------------------------------------------------------------------

def build_trajectory_lines(
    series: TissueGraphTimeSeries,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Build junction lines colored by trajectory_id.

    Junctions that belong to a trajectory share a color across frames.
    Junctions not in any trajectory get gray.

    Returns:
        (lines, colors) same format as build_all_junction_lines.
    """
    # Build a lookup: (frame, frozenset(cell_pair)) -> trajectory_id
    traj_lookup: Dict[Tuple[int, frozenset], int] = {}
    for traj in series.edge_trajectories.values():
        for i, frame_idx in enumerate(traj.frames):
            key = (frame_idx, frozenset(traj.cell_pairs[i]))
            traj_lookup[key] = traj.trajectory_id

    # Assign colors to trajectories
    unique_traj_ids = sorted(set(traj_lookup.values()))
    traj_colors = {}
    for i, tid in enumerate(unique_traj_ids):
        traj_colors[tid] = np.array(_TRACK_COLORMAP[i % len(_TRACK_COLORMAP)])

    gray = np.array([0.5, 0.5, 0.5, 0.5])

    lines = []
    colors = []

    for frame_idx in series.frame_indices:
        frame = series.frames[frame_idx]
        for edge_key, jd in frame.junctions.items():
            if len(jd.coordinates) < 2:
                continue

            coords = jd.coordinates.astype(float)
            frame_col = np.full((len(coords), 1), float(frame_idx))
            lines.append(np.hstack([frame_col, coords]))

            lookup_key = (frame_idx, frozenset(jd.cell_pair))
            if lookup_key in traj_lookup:
                colors.append(traj_colors[traj_lookup[lookup_key]])
            else:
                colors.append(gray)

    if not colors:
        return [], np.empty((0, 4))

    return lines, np.array(colors)
