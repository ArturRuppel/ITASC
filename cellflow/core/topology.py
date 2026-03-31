"""T1 transition detection from TissueGraphTimeSeries.

Compares graph topology between consecutive frames to identify
neighbor exchange events (T1 transitions / intercalations).
"""
import logging
from typing import List, Tuple, FrozenSet, Set

import numpy as np
import networkx as nx

from ..structures import T1Event, TissueGraphDataset, TissueGraphFrame, TissueGraphTimeSeries
from ..analysis.trajectories import build_edge_trajectories

logger = logging.getLogger(__name__)


def _validate_t1_transition(
    lost_edge: FrozenSet[int],
    gained_edge: FrozenSet[int],
    edges_prev: Set[Tuple[int, int]],
    edges_next: Set[Tuple[int, int]],
) -> bool:
    """Validate that a lost/gained edge pair represents a true T1 transition.

    A valid T1 requires:
    - 4 unique cells involved (2 from lost edge, 2 from gained edge)
    - The 4 connecting edges (between lost-pair cells and gained-pair cells)
      exist in both frames

    Parameters
    ----------
    lost_edge : FrozenSet[int]
        The cell pair that lost contact.
    gained_edge : FrozenSet[int]
        The cell pair that gained contact.
    edges_prev : Set[Tuple[int, int]]
        All edges (sorted tuples) in the previous frame.
    edges_next : Set[Tuple[int, int]]
        All edges (sorted tuples) in the next frame.

    Returns
    -------
    bool
        True if this is a valid T1 transition.
    """
    all_cells = lost_edge | gained_edge
    if len(all_cells) != 4:
        return False

    lost_cells = tuple(sorted(lost_edge))
    gained_cells = tuple(sorted(gained_edge))

    # The 4 connecting edges between the losing pair and the gaining pair
    connecting = [
        tuple(sorted((lost_cells[0], gained_cells[0]))),
        tuple(sorted((lost_cells[0], gained_cells[1]))),
        tuple(sorted((lost_cells[1], gained_cells[0]))),
        tuple(sorted((lost_cells[1], gained_cells[1]))),
    ]

    # All connecting edges (excluding the lost/gained edges themselves)
    # must exist in both frames
    for e in connecting:
        if e == lost_cells or e == gained_cells:
            continue
        if e not in edges_prev or e not in edges_next:
            return False

    return True


def _detect_t1_between_frames(
    frame_prev: TissueGraphFrame,
    frame_next: TissueGraphFrame,
    min_junction_length: float = 0.0,
    max_t1_distance: float = float('inf'),
) -> List[T1Event]:
    """Detect T1 transitions between two consecutive frames.

    Parameters
    ----------
    frame_prev : TissueGraphFrame
        The earlier frame.
    frame_next : TissueGraphFrame
        The later frame.
    min_junction_length : float
        Junctions shorter than this are excluded from the removed/added
        sets. Default 0 (no filtering).
    max_t1_distance : float
        Maximum distance between lost and gained edge midpoints to
        pair them as a T1. Default inf (no limit).

    Returns
    -------
    List[T1Event]
        Validated T1 events occurring between these frames.
    """
    edges_prev = set(tuple(sorted(e)) for e in frame_prev.graph.edges())
    edges_next = set(tuple(sorted(e)) for e in frame_next.graph.edges())

    removed = edges_prev - edges_next
    added = edges_next - edges_prev

    # Filter out short junctions
    if min_junction_length > 0:
        removed = {
            e for e in removed
            if frozenset(e) in frame_prev.junctions
            and frame_prev.junctions[frozenset(e)].length >= min_junction_length
        }
        added = {
            e for e in added
            if frozenset(e) in frame_next.junctions
            and frame_next.junctions[frozenset(e)].length >= min_junction_length
        }

    events = []
    used_removed = set()
    used_added = set()

    for rem in removed:
        for add in added:
            if rem in used_removed or add in used_added:
                continue

            lost = frozenset(rem)
            gained = frozenset(add)

            if _validate_t1_transition(lost, gained, edges_prev, edges_next):
                # Location: midpoint of the lost junction
                jdata = frame_prev.junctions.get(lost)
                if jdata is not None:
                    location = jdata.midpoint.copy()
                    # Check spatial proximity constraint
                    if max_t1_distance < float('inf'):
                        jdata_gained = frame_next.junctions.get(gained)
                        if jdata_gained is not None:
                            dist = np.linalg.norm(location - jdata_gained.midpoint)
                            if dist > max_t1_distance:
                                continue
                else:
                    # Fallback: average position of the two losing cells
                    positions = []
                    for cid in rem:
                        if cid in frame_prev.cells:
                            positions.append(frame_prev.cells[cid].position)
                    location = np.mean(positions, axis=0) if positions else np.zeros(2)

                all_cells = set(rem) | set(add)
                event = T1Event(
                    frame=frame_prev.frame,
                    losing_pair=rem,
                    gaining_pair=add,
                    location=location,
                    all_cells=all_cells,
                )
                events.append(event)
                used_removed.add(rem)
                used_added.add(add)

    return events


def detect_t1_events(
    series: TissueGraphTimeSeries,
    min_junction_length: float = 0.0,
    max_t1_distance: float = float('inf'),
) -> List[T1Event]:
    """Detect all T1 transitions in a time series.

    Compares graph topology between consecutive frames to find
    neighbor exchange events. Results are also stored on the
    series object.

    Parameters
    ----------
    series : TissueGraphTimeSeries
        The tissue graph time series to analyze.
    min_junction_length : float
        Junctions shorter than this (px) are excluded from T1 detection.
    max_t1_distance : float
        Max distance between lost/gained edge midpoints to pair as T1.

    Returns
    -------
    List[T1Event]
        All validated T1 events, sorted by frame.
    """
    all_events = []
    indices = series.frame_indices

    for i in range(len(indices) - 1):
        f1 = indices[i]
        f2 = indices[i + 1]
        events = _detect_t1_between_frames(
            series.frames[f1], series.frames[f2],
            min_junction_length=min_junction_length,
            max_t1_distance=max_t1_distance,
        )
        all_events.extend(events)

    all_events.sort(key=lambda e: e.frame)
    series.t1_events = all_events

    logger.info(f"Detected {len(all_events)} T1 events across {len(indices)} frames")
    return all_events


def detect_all_t1_events(
    dataset: TissueGraphDataset,
    progress_callback=None,
    min_junction_length: float = 0.0,
    max_t1_distance: float = float('inf'),
) -> None:
    """Run T1 detection and edge trajectory construction on all tissues.

    Modifies the dataset in place, populating t1_events and
    edge_trajectories on each TissueGraphTimeSeries.

    Parameters
    ----------
    dataset : TissueGraphDataset
        The dataset to analyze.
    progress_callback : callable, optional
        Optional callback(progress_fraction, message).
    min_junction_length : float
        Junctions shorter than this (px) are excluded from T1 detection.
    max_t1_distance : float
        Max distance between lost/gained edge midpoints to pair as T1.
    """
    for idx, (tid, series) in enumerate(dataset.tissues.items()):
        if progress_callback:
            progress_callback(
                idx / dataset.n_tissues,
                f"T1 detection: tissue {idx + 1}/{dataset.n_tissues}",
            )
        events = detect_t1_events(
            series,
            min_junction_length=min_junction_length,
            max_t1_distance=max_t1_distance,
        )
        series.t1_events = events
        trajectories = build_edge_trajectories(series, events)
        series.edge_trajectories = trajectories

    total = sum(len(s.t1_events) for s in dataset.tissues.values())
    logger.info(
        f"Detected {total} T1 events across {dataset.n_tissues} tissues"
    )
