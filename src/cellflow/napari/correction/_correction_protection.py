from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from cellflow.tracking_ultrack.corrections import Correction


def protected_cell_ids_at_frame(
    validated_tracks: Mapping[int, Iterable[int]],
    corrections: Iterable[Correction],
    *,
    frame: int,
    exclude_cell_id: int | None = None,
) -> set[int]:
    """Return cell IDs that should not be overwritten at *frame*."""
    target_frame = int(frame)
    excluded = int(exclude_cell_id) if exclude_cell_id is not None else None
    protected: set[int] = set()

    for raw_cell_id, frames in validated_tracks.items():
        cell_id = int(raw_cell_id)
        if excluded is not None and cell_id == excluded:
            continue
        if target_frame in {int(raw_frame) for raw_frame in frames}:
            protected.add(cell_id)

    for correction in corrections:
        cell_id = int(correction.cell_id)
        if excluded is not None and cell_id == excluded:
            continue
        if correction.kind == "anchor" and int(correction.t) == target_frame:
            protected.add(cell_id)

    return protected


def protected_cell_mask(frame: np.ndarray, protected_ids: Iterable[int]) -> np.ndarray:
    """Return a boolean mask of protected labels in *frame*."""
    ids = [int(cell_id) for cell_id in protected_ids]
    if not ids:
        return np.zeros_like(frame, dtype=bool)
    return np.isin(frame, ids)
