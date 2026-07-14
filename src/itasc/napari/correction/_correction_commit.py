from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from itasc.napari.correction._correction_utils import (
    reassign_ids_stack,
    remove_unvalidated_labels,
)


@dataclass(frozen=True)
class RemoveUnvalidatedResult:
    changed_frames: int
    changed_pixels: int


@dataclass(frozen=True)
class CommitLabelsResult:
    stack: np.ndarray
    n_cells: int
    old_to_new: dict[int, int]
    validated_tracks: dict[int, set[int]]
    changed_frames: int
    changed_pixels: int


def remove_unvalidated_from_data(
    data: np.ndarray,
    validated_tracks: dict[int, set[int]],
) -> RemoveUnvalidatedResult:
    """Remove unvalidated labels from *data* in-place."""
    changed_frames, changed_pixels = remove_unvalidated_labels(
        data,
        validated_tracks,
    )
    return RemoveUnvalidatedResult(
        changed_frames=changed_frames,
        changed_pixels=changed_pixels,
    )


def prepare_committed_labels(
    data: np.ndarray,
    validated_tracks: dict[int, set[int]],
) -> CommitLabelsResult:
    """Return committed labels after ID compaction and validation filtering."""
    remapped, n_cells, old_to_new = reassign_ids_stack(np.asarray(data))
    remapped_validated = remap_validated_track_ids(validated_tracks, old_to_new)
    removal = remove_unvalidated_from_data(remapped, remapped_validated)
    return CommitLabelsResult(
        stack=remapped,
        n_cells=int(n_cells),
        old_to_new=old_to_new,
        validated_tracks=remapped_validated,
        changed_frames=removal.changed_frames,
        changed_pixels=removal.changed_pixels,
    )


def remap_validated_track_ids(
    validated_tracks: dict[int, set[int]],
    old_to_new: dict[int, int],
) -> dict[int, set[int]]:
    """Remap validated-track IDs using the same drop-missing policy as the DB store."""
    return {
        int(old_to_new[int(cell_id)]): set(frames)
        for cell_id, frames in validated_tracks.items()
        if int(cell_id) in old_to_new
    }
