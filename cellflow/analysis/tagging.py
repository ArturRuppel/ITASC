"""Tagging API for junctions and edge trajectories.

Provides functions to tag, untag, and query junctions and trajectories
by user-defined string tags. Tags persist through save/load cycles and
can be used to filter downstream analysis (e.g., focus on central junctions).
"""
from typing import FrozenSet, List, Optional, Set, Tuple

import numpy as np

from ..structures import (
    EdgeTrajectory,
    TissueGraphFrame,
    TissueGraphTimeSeries,
)


# ------------------------------------------------------------------
# Trajectory tagging
# ------------------------------------------------------------------

def tag_trajectory(
    series: TissueGraphTimeSeries,
    trajectory_id: int,
    tag: str,
) -> None:
    """Add a tag to a trajectory."""
    series.edge_trajectories[trajectory_id].tags.add(tag)


def untag_trajectory(
    series: TissueGraphTimeSeries,
    trajectory_id: int,
    tag: str,
) -> None:
    """Remove a tag from a trajectory."""
    series.edge_trajectories[trajectory_id].tags.discard(tag)


def name_trajectory(
    series: TissueGraphTimeSeries,
    trajectory_id: int,
    name: Optional[str],
) -> None:
    """Set a user-assigned name for a trajectory."""
    series.edge_trajectories[trajectory_id].name = name


def get_trajectories_by_tag(
    series: TissueGraphTimeSeries,
    tag: str,
) -> List[EdgeTrajectory]:
    """Return all trajectories that carry a given tag."""
    return [
        t for t in series.edge_trajectories.values()
        if tag in t.tags
    ]


def get_trajectory_by_name(
    series: TissueGraphTimeSeries,
    name: str,
) -> Optional[EdgeTrajectory]:
    """Return the trajectory with a given name, or None."""
    for t in series.edge_trajectories.values():
        if t.name == name:
            return t
    return None


# ------------------------------------------------------------------
# Junction tagging (per-frame)
# ------------------------------------------------------------------

def tag_junction(
    frame: TissueGraphFrame,
    cell_pair: Tuple[int, int],
    tag: str,
) -> None:
    """Add a tag to a junction in a specific frame."""
    key = frozenset(cell_pair)
    frame.junctions[key].tags.add(tag)


def untag_junction(
    frame: TissueGraphFrame,
    cell_pair: Tuple[int, int],
    tag: str,
) -> None:
    """Remove a tag from a junction in a specific frame."""
    key = frozenset(cell_pair)
    frame.junctions[key].tags.discard(tag)


def get_junctions_by_tag(
    frame: TissueGraphFrame,
    tag: str,
) -> List[FrozenSet[int]]:
    """Return cell-pair keys of all junctions carrying a given tag."""
    return [
        key for key, jd in frame.junctions.items()
        if tag in jd.tags
    ]


# ------------------------------------------------------------------
# Bulk tagging
# ------------------------------------------------------------------

def tag_trajectories_near(
    series: TissueGraphTimeSeries,
    location: np.ndarray,
    radius: float,
    tag: str,
) -> List[int]:
    """Tag all trajectories whose midpoint falls within radius of location.

    Uses the midpoint from the first frame of each trajectory.

    Parameters
    ----------
    series : TissueGraphTimeSeries
        The time series to search.
    location : np.ndarray
        (y, x) center point.
    radius : float
        Search radius in pixels.
    tag : str
        Tag to apply.

    Returns
    -------
    List[int]
        IDs of trajectories that were tagged.
    """
    location = np.asarray(location, dtype=float)
    tagged_ids = []

    for traj in series.edge_trajectories.values():
        if not traj.frames:
            continue
        # Get midpoint from first frame
        first_frame_idx = traj.frames[0]
        first_pair = traj.cell_pairs[0]
        key = frozenset(first_pair)
        frame = series.frames.get(first_frame_idx)
        if frame is None:
            continue
        jd = frame.junctions.get(key)
        if jd is None:
            continue
        dist = np.linalg.norm(jd.midpoint - location)
        if dist <= radius:
            traj.tags.add(tag)
            tagged_ids.append(traj.trajectory_id)

    return tagged_ids


# ------------------------------------------------------------------
# Tag queries
# ------------------------------------------------------------------

def get_all_tags(series: TissueGraphTimeSeries) -> Set[str]:
    """Collect all unique tags across trajectories and junctions."""
    tags = set()
    for traj in series.edge_trajectories.values():
        tags.update(traj.tags)
    for frame in series.frames.values():
        for jd in frame.junctions.values():
            tags.update(jd.tags)
    return tags


def clear_tag(series: TissueGraphTimeSeries, tag: str) -> int:
    """Remove a tag from all trajectories and junctions. Returns count removed."""
    count = 0
    for traj in series.edge_trajectories.values():
        if tag in traj.tags:
            traj.tags.discard(tag)
            count += 1
    for frame in series.frames.values():
        for jd in frame.junctions.values():
            if tag in jd.tags:
                jd.tags.discard(tag)
                count += 1
    return count
