from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cellflow.napari.correction._correction_utils import frame_view_2d
from cellflow.tracking_ultrack.corrections import Correction


@dataclass(frozen=True)
class SelectedCorrectionTarget:
    cell_id: int
    frame: int
    y: float
    x: float


def selected_correction_target(
    data: np.ndarray,
    *,
    cell_id: int,
    frame: int,
) -> SelectedCorrectionTarget | None:
    """Return the selected cell centroid at *frame* when present."""
    frame_view = _label_frame(data, frame)
    if frame_view is None or not np.any(frame_view == cell_id):
        return None
    y, x = _label_centroid(frame_view, cell_id)
    return SelectedCorrectionTarget(
        cell_id=int(cell_id),
        frame=int(frame),
        y=y,
        x=x,
    )


def correction_for_label_frame(
    data: np.ndarray,
    *,
    cell_id: int,
    frame: int,
) -> Correction | None:
    """Build a validated correction for *cell_id* at *frame* when present."""
    target = selected_correction_target(data, cell_id=cell_id, frame=frame)
    if target is None:
        return None
    return Correction(
        cell_id=target.cell_id,
        t=target.frame,
        kind="validated",
        y=target.y,
        x=target.x,
    )


def corrections_for_label_frames(
    data: np.ndarray,
    *,
    cell_id: int,
    frames: list[int] | tuple[int, ...],
) -> list[Correction]:
    """Build validated corrections for frames where *cell_id* is present."""
    corrections: list[Correction] = []
    for frame in frames:
        correction = correction_for_label_frame(
            data,
            cell_id=cell_id,
            frame=int(frame),
        )
        if correction is not None:
            corrections.append(correction)
    return corrections


def _label_frame(data: np.ndarray, frame: int) -> np.ndarray | None:
    arr = np.asarray(data)
    return frame_view_2d(arr, frame) if arr.ndim >= 3 else arr


def _label_centroid(frame: np.ndarray, cell_id: int) -> tuple[float, float]:
    yy, xx = np.nonzero(frame == cell_id)
    return float(np.mean(yy)), float(np.mean(xx))
