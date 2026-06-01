"""Build the whole-track temporal "comet" overlay for correction mode.

Given a tracked label stack and one track id, outline that track's mask from
every frame it appears in onto a single plane, colored start->finish with
viridis (earliest frame dark, latest yellow). Only the mask boundary is painted
so every frame's footprint stays visible instead of later frames burying earlier
ones; where outlines overlap, the newest frame is drawn on top. Also return the
boolean union of all the track's (filled) masks (used to enlarge the correction
spotlight to the whole trajectory) and the per-frame centroids (used to place a
frame number inside each mask).

Pure module: no Qt, no napari, so it is unit-testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackPathOverlay:
    """Rendered comet for one track.

    ``frames`` lists the occupied frame indices in ascending (oldest-first)
    order; ``colors``, ``centroids`` are aligned with it row-for-row.
    """

    frames: tuple[int, ...]      # occupied frame indices, ascending
    colors: np.ndarray           # (N, 4) RGBA, frames[0] dark -> frames[-1] yellow
    overlay: np.ndarray          # (H, W, 4) RGBA float, mask outlines, newest on top
    union_mask: np.ndarray       # (H, W) bool, union of all the track's filled masks
    centroids: np.ndarray        # (N, 2) (y, x) centroid per occupied frame

    def is_empty(self) -> bool:
        return len(self.frames) == 0

    def frame_number_labels(self) -> list[str]:
        """Text labels (the frame numbers) aligned with :attr:`centroids`."""
        return [str(f) for f in self.frames]


def _viridis_colors(n: int) -> np.ndarray:
    """``n`` RGBA viridis samples from dark (0.0) to yellow (1.0).

    A single frame maps to the dark end so the mapping stays deterministic.
    """
    if n <= 0:
        return np.empty((0, 4), dtype=float)
    from matplotlib import colormaps

    positions = np.linspace(0.0, 1.0, n)
    return np.asarray(colormaps["viridis"](positions), dtype=float)


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of ``mask``: in the mask with a non-mask 4-neighbor.

    Pixels on the array edge count as boundary (they have an off-image
    neighbor), so a mask touching the border still gets a closed outline.
    """
    if not mask.any():
        return mask
    interior = np.ones_like(mask)
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    return mask & ~interior


def build_track_path_overlay(
    tracked_stack: np.ndarray, track_id: int
) -> TrackPathOverlay:
    """Paint the comet for ``track_id`` across all frames of ``tracked_stack``.

    ``tracked_stack`` is a ``(T, H, W)`` label array (a bare ``(H, W)`` plane is
    treated as a single frame). The painted label value equals ``track_id``,
    matching the exported tracked stack convention.
    """
    stack = np.asarray(tracked_stack)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3:
        raise ValueError(
            f"tracked_stack must be 2D or 3D, got {stack.ndim}D"
        )

    track_id = int(track_id)
    height, width = stack.shape[1], stack.shape[2]

    occupied: list[int] = []
    masks: list[np.ndarray] = []
    centroids: list[tuple[float, float]] = []
    for t in range(stack.shape[0]):
        mask = stack[t] == track_id
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        occupied.append(t)
        masks.append(mask)
        centroids.append((float(ys.mean()), float(xs.mean())))

    overlay = np.zeros((height, width, 4), dtype=float)
    union_mask = np.zeros((height, width), dtype=bool)
    if not occupied:
        return TrackPathOverlay(
            frames=(),
            colors=np.empty((0, 4), dtype=float),
            overlay=overlay,
            union_mask=union_mask,
            centroids=np.empty((0, 2), dtype=float),
        )

    colors = _viridis_colors(len(occupied))
    # Oldest-first so larger t overwrites where outlines overlap (newest on top).
    # union_mask stays the filled union (spotlight covers the whole trajectory).
    for mask, color in zip(masks, colors, strict=True):
        overlay[_mask_outline(mask)] = color
        union_mask |= mask

    return TrackPathOverlay(
        frames=tuple(occupied),
        colors=colors,
        overlay=overlay,
        union_mask=union_mask,
        centroids=np.asarray(centroids, dtype=float),
    )
