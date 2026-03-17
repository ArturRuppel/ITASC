"""Edge trajectory construction with sign convention.

Tracks junction identity through time and links junctions across
T1 events. The sign convention is: junction length is positive before
a T1 (collapsing edge) and negative after (new edge growing).
"""
import logging
from typing import Dict, List, Tuple, FrozenSet

import numpy as np

from ..structures import (
    EdgeTrajectory,
    T1Event,
    TissueGraphTimeSeries,
)

logger = logging.getLogger(__name__)


def build_edge_trajectories(
    series: TissueGraphTimeSeries,
    t1_events: List[T1Event],
) -> Dict[int, EdgeTrajectory]:
    """Build edge trajectories, linking junctions across T1 events.

    Uses a two-pass algorithm:
    1. Process intercalation events to merge losing/gaining edges
       into shared trajectories.
    2. Process all junctions frame-by-frame, assigning each to a
       trajectory and applying the sign convention.

    Parameters
    ----------
    series : TissueGraphTimeSeries
        The tissue graph time series.
    t1_events : List[T1Event]
        Detected T1 events (from detect_t1_events).

    Returns
    -------
    Dict[int, EdgeTrajectory]
        Mapping of trajectory_id to EdgeTrajectory.
    """
    sorted_events = sorted(t1_events, key=lambda e: e.frame)

    trajectories: Dict[int, EdgeTrajectory] = {}
    pair_to_traj: Dict[FrozenSet[int], int] = {}
    next_id = 1

    # --- Pass 1: link losing/gaining edges through T1 events ---
    for event in sorted_events:
        losing = frozenset(event.losing_pair)
        gaining = frozenset(event.gaining_pair)

        traj_id = pair_to_traj.get(losing)

        if traj_id is None:
            traj_id = next_id
            next_id += 1
            trajectories[traj_id] = EdgeTrajectory(
                trajectory_id=traj_id,
                frames=[],
                cell_pairs=[],
                signed_lengths=[],
                coordinates=[],
                t1_events=[event],
            )
            pair_to_traj[losing] = traj_id
        else:
            trajectories[traj_id].t1_events.append(event)

        # Map gaining edge to the same trajectory
        pair_to_traj[gaining] = traj_id

    # --- Pass 2: fill in junction data frame by frame ---
    for frame_idx in series.frame_indices:
        frame = series.frames[frame_idx]

        for cell_pair_fs, jdata in frame.junctions.items():
            # Get or create trajectory
            if cell_pair_fs not in pair_to_traj:
                traj_id = next_id
                next_id += 1
                pair_to_traj[cell_pair_fs] = traj_id
                trajectories[traj_id] = EdgeTrajectory(
                    trajectory_id=traj_id,
                    frames=[],
                    cell_pairs=[],
                    signed_lengths=[],
                    coordinates=[],
                    t1_events=[],
                )

            traj_id = pair_to_traj[cell_pair_fs]
            traj = trajectories[traj_id]

            # Determine sign: starts positive, flips at each T1
            sign = 1.0
            for evt in traj.t1_events:
                if frame_idx > evt.frame:
                    sign *= -1.0

            traj.frames.append(frame_idx)
            traj.cell_pairs.append(jdata.cell_pair)
            traj.signed_lengths.append(sign * jdata.length)
            traj.coordinates.append(jdata.coordinates)

    # Store on series
    series.edge_trajectories = trajectories

    n_t1 = sum(1 for t in trajectories.values() if t.t1_events)
    logger.info(
        f"Built {len(trajectories)} edge trajectories "
        f"({n_t1} involved in T1 events)"
    )
    return trajectories


def get_t1_trajectories(
    series: TissueGraphTimeSeries,
) -> List[EdgeTrajectory]:
    """Filter to trajectories that contain T1 events.

    Parameters
    ----------
    series : TissueGraphTimeSeries
        Must have edge_trajectories populated.

    Returns
    -------
    List[EdgeTrajectory]
        Trajectories with at least one T1 event.
    """
    return [t for t in series.edge_trajectories.values() if t.t1_events]


def get_stable_trajectories(
    series: TissueGraphTimeSeries,
    min_frames: int = 1,
) -> List[EdgeTrajectory]:
    """Filter to long-lived junctions without T1 events.

    Parameters
    ----------
    series : TissueGraphTimeSeries
        Must have edge_trajectories populated.
    min_frames : int
        Minimum number of frames the trajectory must span.

    Returns
    -------
    List[EdgeTrajectory]
        Stable trajectories meeting the criteria.
    """
    return [
        t
        for t in series.edge_trajectories.values()
        if not t.t1_events and len(t.frames) >= min_frames
    ]
