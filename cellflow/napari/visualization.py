"""Napari layer rendering for tissue graphs."""
import numpy as np
import pandas as pd
from matplotlib import colormaps as _cm
from typing import Dict, List, Optional, Set, Tuple

from ..structures import T1Event, TissueGraphFrame, TissueGraphTimeSeries

_MAGMA = _cm["magma"]
_CIVIDIS = _cm["cividis"]


def build_tracked_labels(
    label_stack: np.ndarray,
    track_map: Dict[int, Dict[int, int]],
) -> np.ndarray:
    """Create a label array where pixel values are track IDs instead of cell labels.

    Untracked cells get unique IDs above max_track_id so they remain visible
    but are distinguishable from tracked cells.  Border cells (touching the
    image edge) are set to 0 (transparent) so they don't appear as
    coloured background.

    Args:
        label_stack: Shape (T, H, W) integer labels, 0 = background.
        track_map: Dict of {frame_idx: {cell_id: track_id}}.

    Returns:
        Array of same shape as label_stack with track IDs as pixel values.
    """
    from ..core.labels import find_border_cells

    result = np.zeros_like(label_stack)

    # Find max track_id across all frames
    max_track_id = 0
    for frame_tracks in track_map.values():
        if frame_tracks:
            max_track_id = max(max_track_id, max(frame_tracks.values()))

    next_id = max_track_id + 1

    for frame_idx in range(len(label_stack)):
        frame = label_stack[frame_idx]
        frame_tracks = track_map.get(frame_idx, {})
        border_ids = find_border_cells(frame)

        # Map to assign untracked cells unique IDs (per-frame)
        untracked_ids: Dict[int, int] = {}

        for cell_id in np.unique(frame):
            if cell_id == 0 or cell_id in border_ids:
                continue
            mask = frame == cell_id
            if cell_id in frame_tracks:
                result[frame_idx][mask] = frame_tracks[cell_id]
            else:
                if cell_id not in untracked_ids:
                    untracked_ids[cell_id] = next_id
                    next_id += 1
                result[frame_idx][mask] = untracked_ids[cell_id]

    return result


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


def build_tension_colored_junctions(
    series: TissueGraphTimeSeries,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Build junction line data colored by inferred tension.

    Junctions without tension values are shown in gray.  Internal junctions
    are colored using the magma colormap.

    Returns:
        (lines, colors) same format as build_all_junction_lines.
    """
    lines = []
    colors = []

    # Collect all non-None tensions for global normalization
    all_tensions = []
    for frame in series.frames.values():
        for jd in frame.junctions.values():
            if jd.tension is not None and len(jd.coordinates) >= 2:
                all_tensions.append(jd.tension)

    if not all_tensions:
        return [], np.empty((0, 4))

    vmin, vmax = min(all_tensions), max(all_tensions)

    gray = [0.5, 0.5, 0.5, 0.4]

    for frame_idx in series.frame_indices:
        frame = series.frames[frame_idx]
        for jd in frame.junctions.values():
            if len(jd.coordinates) < 2:
                continue

            coords = jd.coordinates.astype(float)
            frame_col = np.full((len(coords), 1), float(frame_idx))
            lines.append(np.hstack([frame_col, coords]))

            if jd.tension is not None and vmax > vmin:
                normed = (jd.tension - vmin) / (vmax - vmin)
                colors.append(_MAGMA(normed))
            elif jd.tension is not None:
                colors.append(_MAGMA(0.5))
            else:
                colors.append(gray)

    return lines, np.array(colors)


def build_pressure_colored_cells(
    series: TissueGraphTimeSeries,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Build cell polygon data colored by inferred pressure.

    Only cells with both vertices and pressure values are included.
    Colored using the cividis colormap.

    Returns:
        (polygons, colors) where polygons is a list of Nx3 arrays
        (frame, y, x) and colors is an Nx4 RGBA array.
    """
    polygons = []
    colors = []

    # Collect all pressures for global normalization (skip border cells)
    all_pressures = []
    for frame in series.frames.values():
        for cd in frame.cells.values():
            if cd.is_border:
                continue
            if cd.pressure is not None and cd.vertices is not None:
                all_pressures.append(cd.pressure)

    if not all_pressures:
        return [], np.empty((0, 4))

    vmin, vmax = min(all_pressures), max(all_pressures)

    for frame_idx in series.frame_indices:
        frame = series.frames[frame_idx]
        for cd in frame.cells.values():
            if cd.is_border:
                continue
            if cd.vertices is None or cd.pressure is None:
                continue
            if len(cd.vertices) < 3:
                continue

            coords = cd.vertices.astype(float)
            frame_col = np.full((len(coords), 1), float(frame_idx))
            polygons.append(np.hstack([frame_col, coords]))

            if vmax > vmin:
                normed = (cd.pressure - vmin) / (vmax - vmin)
            else:
                normed = 0.5
            rgba = list(_CIVIDIS(normed))
            rgba[3] = 0.5
            colors.append(rgba)

    return polygons, np.array(colors)


def build_all_centroids(series: TissueGraphTimeSeries) -> np.ndarray:
    """Build centroid positions for all frames at once.

    Border cells are excluded so that tissue-edge cells (which often
    look like "background") don't get markers.

    Returns Nx3 array of (frame, y, x) positions.
    """
    positions = []
    for frame_idx in series.frame_indices:
        frame = series.frames[frame_idx]
        for cd in frame.cells.values():
            if cd.is_border:
                continue
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
            if cd.is_border:
                continue
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
    lines, colors, _ = build_trajectory_lines_with_features(series)
    return lines, colors


def build_trajectory_lines_with_features(
    series: TissueGraphTimeSeries,
    color_by_tags: bool = False,
    show_only_tagged: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray, pd.DataFrame]:
    """Build junction lines with per-shape features for selection and tagging.

    Each shape (line) gets a row in the features DataFrame containing
    trajectory_id, cell_pair_a, cell_pair_b, tags, and name. This enables
    napari's built-in shape selection for interactive tagging.

    Parameters
    ----------
    series : TissueGraphTimeSeries
        The time series with edge trajectories.
    color_by_tags : bool
        If True, color by tag (first tag wins). If False, color by trajectory_id.
    show_only_tagged : bool
        If True, only include junctions that have at least one tag.

    Returns
    -------
    lines : List[np.ndarray]
        List of Nx3 arrays (frame, y, x) — one per junction.
    colors : np.ndarray
        Nx4 RGBA color array — one per junction.
    features : pd.DataFrame
        Per-shape features with columns: trajectory_id, cell_pair_a,
        cell_pair_b, frame, tags, name.
    """
    # Build a lookup: (frame, frozenset(cell_pair)) -> trajectory_id
    traj_lookup: Dict[Tuple[int, frozenset], int] = {}
    for traj in series.edge_trajectories.values():
        for i, frame_idx in enumerate(traj.frames):
            key = (frame_idx, frozenset(traj.cell_pairs[i]))
            traj_lookup[key] = traj.trajectory_id

    # Collect all tags for color assignment
    all_tags: Set[str] = set()
    if color_by_tags:
        for traj in series.edge_trajectories.values():
            all_tags.update(traj.tags)
        for frame in series.frames.values():
            for jd in frame.junctions.values():
                all_tags.update(jd.tags)
    sorted_tags = sorted(all_tags)
    tag_color_map = {
        tag: np.array(_TAG_COLORS[i % len(_TAG_COLORS)])
        for i, tag in enumerate(sorted_tags)
    }

    # Assign trajectory colors (for non-tag mode)
    unique_traj_ids = sorted(set(traj_lookup.values()))
    traj_colors = {}
    for i, tid in enumerate(unique_traj_ids):
        traj_colors[tid] = np.array(_TRACK_COLORMAP[i % len(_TRACK_COLORMAP)])

    gray = np.array([0.5, 0.5, 0.5, 0.5])

    lines = []
    colors = []
    feat_rows = []

    for frame_idx in series.frame_indices:
        frame = series.frames[frame_idx]
        for edge_key, jd in frame.junctions.items():
            if len(jd.coordinates) < 2:
                continue

            lookup_key = (frame_idx, frozenset(jd.cell_pair))
            traj_id = traj_lookup.get(lookup_key, -1)

            # Merge tags from junction and trajectory
            merged_tags = set(jd.tags)
            if traj_id != -1:
                merged_tags |= series.edge_trajectories[traj_id].tags
            tags_str = ",".join(sorted(merged_tags)) if merged_tags else ""

            if show_only_tagged and not merged_tags:
                continue

            # Get trajectory name
            traj_name = ""
            if traj_id != -1:
                traj = series.edge_trajectories[traj_id]
                traj_name = traj.name or ""

            coords = jd.coordinates.astype(float)
            frame_col = np.full((len(coords), 1), float(frame_idx))
            lines.append(np.hstack([frame_col, coords]))

            # Color logic
            if color_by_tags and merged_tags:
                first_tag = sorted(merged_tags)[0]
                colors.append(tag_color_map[first_tag])
            elif traj_id != -1:
                colors.append(traj_colors[traj_id])
            else:
                colors.append(gray)

            feat_rows.append({
                "trajectory_id": traj_id,
                "cell_pair_a": jd.cell_pair[0],
                "cell_pair_b": jd.cell_pair[1],
                "frame": frame_idx,
                "tags": tags_str,
                "name": traj_name,
            })

    if not colors:
        empty_df = pd.DataFrame(columns=[
            "trajectory_id", "cell_pair_a", "cell_pair_b", "frame", "tags", "name",
        ])
        return [], np.empty((0, 4)), empty_df

    return lines, np.array(colors), pd.DataFrame(feat_rows)


_TAG_COLORS = [
    [0.90, 0.10, 0.10, 1.0],  # red
    [0.10, 0.70, 0.10, 1.0],  # green
    [0.10, 0.10, 0.90, 1.0],  # blue
    [0.90, 0.60, 0.10, 1.0],  # orange
    [0.70, 0.10, 0.70, 1.0],  # magenta
    [0.10, 0.70, 0.70, 1.0],  # teal
    [0.90, 0.90, 0.10, 1.0],  # yellow
]


def build_tag_text_annotations(
    series: TissueGraphTimeSeries,
) -> Tuple[np.ndarray, List[str], np.ndarray, pd.DataFrame]:
    """Build text annotation data for tagged junctions.

    Returns one point per tagged junction at its midpoint, with the
    tag string as the text label plus a features DataFrame for mapping
    back to the underlying junction/trajectory.

    Parameters
    ----------
    series : TissueGraphTimeSeries
        The time series with edge trajectories and junction tags.

    Returns
    -------
    positions : np.ndarray
        Nx3 array of (frame, y, x) positions for text anchors.
    texts : list of str
        One label string per point (comma-separated tags).
    colors : np.ndarray
        Nx4 RGBA array — color matches the first tag.
    features : pd.DataFrame
        Per-point features: trajectory_id, cell_pair_a, cell_pair_b, frame, tags.
    """
    traj_lookup: Dict[Tuple[int, frozenset], int] = {}
    for traj in series.edge_trajectories.values():
        for i, frame_idx in enumerate(traj.frames):
            key = (frame_idx, frozenset(traj.cell_pairs[i]))
            traj_lookup[key] = traj.trajectory_id

    # Build tag -> color map
    all_tags: Set[str] = set()
    for traj in series.edge_trajectories.values():
        all_tags.update(traj.tags)
    for frame in series.frames.values():
        for jd in frame.junctions.values():
            all_tags.update(jd.tags)
    sorted_tags = sorted(all_tags)
    tag_color_map = {
        tag: np.array(_TAG_COLORS[i % len(_TAG_COLORS)])
        for i, tag in enumerate(sorted_tags)
    }

    positions = []
    texts = []
    colors = []
    feat_rows = []

    for frame_idx in series.frame_indices:
        frame = series.frames[frame_idx]
        for jd in frame.junctions.values():
            lookup_key = (frame_idx, frozenset(jd.cell_pair))
            traj_id = traj_lookup.get(lookup_key, -1)

            merged_tags = set(jd.tags)
            if traj_id != -1:
                merged_tags |= series.edge_trajectories[traj_id].tags

            if not merged_tags:
                continue

            label = ", ".join(sorted(merged_tags))
            first_tag = sorted(merged_tags)[0]

            # Place text at the junction midpoint, offset slightly upward
            y, x = jd.midpoint
            positions.append([float(frame_idx), y - 3.0, x])
            texts.append(label)
            colors.append(tag_color_map[first_tag])
            feat_rows.append({
                "trajectory_id": traj_id,
                "cell_pair_a": jd.cell_pair[0],
                "cell_pair_b": jd.cell_pair[1],
                "frame": frame_idx,
                "tags": label,
            })

    if not positions:
        empty_df = pd.DataFrame(
            columns=["trajectory_id", "cell_pair_a", "cell_pair_b", "frame", "tags"],
        )
        return np.empty((0, 3)), [], np.empty((0, 4)), empty_df

    return np.array(positions), texts, np.array(colors), pd.DataFrame(feat_rows)
