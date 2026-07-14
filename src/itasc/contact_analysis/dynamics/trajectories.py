"""Tracked label stack → per-track centroid trajectories (µm), gaps preserved.

The shared front end of every dynamics calculation. Runs
:func:`skimage.measure.regionprops` over each frame of a tracked label stack
(``label == track_id``, constant across frames — the same per-frame loop the
shape core uses) and groups centroids into one trajectory per track id. Frames
where a track's label is absent are simply missing from its series, so a
:class:`Trajectory` carries an explicit ``frames`` array and downstream code can
detect gaps (``diff(frames) > 1``).

Centroids are returned in **µm** (``(x, y)`` = ``(col, row) · pixel_size_um``);
``x`` is the column axis and ``y`` the row axis, matching the shape core's
``centroid_x_um`` / ``centroid_y_um``. Backend-only (no Qt / napari).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.measure import regionprops

from ..shape.core import read_label_stack


@dataclass(frozen=True)
class Trajectory:
    """One track's centroid path in µm, in frame order.

    ``frames`` is strictly increasing (one entry per frame the track is present);
    ``xy`` is the matching ``(N, 2)`` array of ``(x_um, y_um)`` centroids. A jump
    of more than 1 between consecutive ``frames`` is a gap.
    """

    track_id: int
    frames: np.ndarray  # (N,) int64, strictly increasing
    xy: np.ndarray      # (N, 2) float, columns (x_um, y_um)

    @property
    def n_frames(self) -> int:
        return int(self.frames.size)

    @property
    def n_gaps(self) -> int:
        if self.frames.size < 2:
            return 0
        return int(np.count_nonzero(np.diff(self.frames) > 1))


def extract_trajectories(
    label_path: str | Path,
    *,
    pixel_size_um: float,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> list[Trajectory]:
    """Read a tracked label TIFF and return one :class:`Trajectory` per track id.

    Centroids come from ``regionprops(frame).centroid`` (row, col) scaled by
    *pixel_size_um* into ``(x_um, y_um)``. Trajectories are sorted by track id;
    each is internally sorted by frame.
    """
    pixel_size_um = float(pixel_size_um)
    if not pixel_size_um > 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um!r}")
    label_stack = read_label_stack(Path(label_path))
    return trajectories_from_stack(
        label_stack, pixel_size_um=pixel_size_um, progress_cb=progress_cb
    )


def trajectories_from_stack(
    label_stack: np.ndarray,
    *,
    pixel_size_um: float,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> list[Trajectory]:
    """:func:`extract_trajectories` over an in-memory ``T×Y×X`` label stack."""
    total = int(label_stack.shape[0])
    # track_id -> (list[frame], list[(x_um, y_um)])
    by_track: dict[int, tuple[list[int], list[tuple[float, float]]]] = {}
    for frame_idx, frame in enumerate(label_stack):
        for prop in regionprops(frame):
            row, col = prop.centroid
            frames, coords = by_track.setdefault(int(prop.label), ([], []))
            frames.append(frame_idx)
            coords.append((col * pixel_size_um, row * pixel_size_um))
        if progress_cb is not None:
            progress_cb(frame_idx + 1, total, "extract centroids")

    trajectories: list[Trajectory] = []
    for track_id in sorted(by_track):
        frames, coords = by_track[track_id]
        frame_arr = np.asarray(frames, dtype=np.int64)
        xy = np.asarray(coords, dtype=float)
        order = np.argsort(frame_arr, kind="stable")
        trajectories.append(
            Trajectory(track_id=track_id, frames=frame_arr[order], xy=xy[order])
        )
    return trajectories
