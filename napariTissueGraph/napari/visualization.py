"""Napari layer rendering for tissue graphs."""
import numpy as np
from typing import List, Tuple

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
