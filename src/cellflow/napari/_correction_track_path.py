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


@dataclass(frozen=True)
class FilmStripTile:
    """One frame's panel in the film strip: an RGB crop with the mask outlined.

    ``validated`` / ``anchored`` flag whether this frame is a validated or an
    anchored frame for the track; the view draws a coloured marker strip for
    each. They are view metadata, not baked into ``rgb``.
    """

    frame: int            # source frame index
    rgb: np.ndarray       # (h, w, 3) uint8, raw crop with the mask edge drawn on
    validated: bool = False
    anchored: bool = False

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]


@dataclass(frozen=True)
class TrackFilmStrip:
    """Per-frame crops of one track, ready for a Qt dock to blit side by side.

    Every tile is a fixed-size square window *centered on that frame's nucleus
    centroid*, so the nucleus stays put in the middle of every tile and you read
    the surroundings sweeping past. Tiles are ordered oldest-first.
    """

    tiles: tuple[FilmStripTile, ...]

    def is_empty(self) -> bool:
        return len(self.tiles) == 0

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(tile.frame for tile in self.tiles)


def _binary_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    """8-connected binary dilation by ``iterations`` (no scipy dependency)."""
    out = mask
    for _ in range(max(iterations, 0)):
        d = out.copy()
        d[1:, :] |= out[:-1, :]
        d[:-1, :] |= out[1:, :]
        d[:, 1:] |= out[:, :-1]
        d[:, :-1] |= out[:, 1:]
        d[1:, 1:] |= out[:-1, :-1]
        d[1:, :-1] |= out[:-1, 1:]
        d[:-1, 1:] |= out[1:, :-1]
        d[:-1, :-1] |= out[1:, 1:]
        out = d
    return out


def _centered_crop(arr: np.ndarray, cy: int, cx: int, size: int) -> np.ndarray:
    """``size``x``size`` crop of ``arr`` centered on (cy, cx), zero-padded at edges."""
    out = np.zeros((size, size), dtype=arr.dtype)
    half = size // 2
    top, left = cy - half, cx - half
    h, w = arr.shape
    y0, y1 = max(top, 0), min(top + size, h)
    x0, x1 = max(left, 0), min(left + size, w)
    if y0 < y1 and x0 < x1:
        out[y0 - top : y1 - top, x0 - left : x1 - left] = arr[y0:y1, x0:x1]
    return out


def _apply_colormap(normalized: np.ndarray, colormap) -> np.ndarray:
    """Map a (h, w) array in [0, 1] to (h, w, 3) RGB float; grayscale if None."""
    if colormap is None:
        return np.repeat(normalized[:, :, np.newaxis], 3, axis=2)
    mapped = np.asarray(colormap(normalized), dtype=float)
    # ascontiguousarray guarantees a writable copy (the builder mutates rgb).
    return np.ascontiguousarray(mapped[..., :3], dtype=float)


def build_track_film_strip(
    tracked_stack: np.ndarray,
    intensity_stack: np.ndarray,
    track_id: int,
    *,
    margin: int = 6,
    colormap=None,
    outline_width: int = 2,
    outline_color: tuple[float, float, float] | None = None,
    spotlight_dim: float = 0.35,
    spotlight_dilation: int = 2,
    validated_frames: set[int] | None = None,
    anchored_frames: set[int] | None = None,
) -> TrackFilmStrip:
    """Build per-frame, nucleus-centered intensity crops for ``track_id``.

    ``tracked_stack`` and ``intensity_stack`` are matching ``(T, H, W)`` arrays
    (bare ``(H, W)`` planes are treated as a single frame). For each occupied
    frame the intensity is cropped to a fixed square window centered on the
    nucleus, contrast-stretched against the track's own nucleus pixels, colored
    through ``colormap`` (e.g. the layer's "I Purple"; grayscale if ``None``),
    and dimmed outside the nucleus by ``spotlight_dim`` for a spotlight effect.

    A ``outline_width``-thick border is drawn at the *inner edge of the bright
    spotlight region*, so the coloured contour coincides with the bright/dim
    boundary instead of leaving a bright ring outside it. The border uses
    ``outline_color`` (the label layer's colour for this track, RGB in 0..1);
    when ``None`` it falls back to the frame's viridis time colour.

    ``validated_frames`` / ``anchored_frames`` (sets of frame indices) flag each
    tile so the view can mark validated/anchored frames. ``margin`` pads the
    window around the largest nucleus.
    """
    tracked = np.asarray(tracked_stack)
    intensity = np.asarray(intensity_stack)
    if tracked.ndim == 2:
        tracked = tracked[np.newaxis, ...]
    if intensity.ndim == 2:
        intensity = intensity[np.newaxis, ...]
    if tracked.ndim != 3 or intensity.ndim != 3:
        raise ValueError("tracked_stack and intensity_stack must be 2D or 3D")
    if tracked.shape != intensity.shape:
        raise ValueError(
            f"shape mismatch: tracked {tracked.shape} vs intensity {intensity.shape}"
        )

    track_id = int(track_id)

    occupied: list[int] = []
    masks: list[np.ndarray] = []
    centroids: list[tuple[int, int]] = []
    extents: list[int] = []
    for t in range(tracked.shape[0]):
        mask = tracked[t] == track_id
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        occupied.append(t)
        masks.append(mask)
        centroids.append((int(round(ys.mean())), int(round(xs.mean()))))
        extents.append(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
    if not occupied:
        return TrackFilmStrip(tiles=())

    # One square window big enough for the largest nucleus plus margin, used for
    # every tile so the strip is uniform and the nucleus is always centered.
    size = int(max(extents)) + 2 * margin

    # Contrast-stretch against the track's own nucleus pixels (good cell contrast
    # regardless of background), shared across tiles for comparability.
    nucleus_values = np.concatenate(
        [intensity[t][mask] for t, mask in zip(occupied, masks)]
    ).astype(float)
    if nucleus_values.size:
        lo = float(np.percentile(nucleus_values, 2.0))
        hi = float(np.percentile(nucleus_values, 98.0))
    else:  # pragma: no cover - occupied implies non-empty masks
        lo, hi = 0.0, 1.0

    validated = {int(f) for f in (validated_frames or set())}
    anchored = {int(f) for f in (anchored_frames or set())}

    colors = _viridis_colors(len(occupied))
    tiles: list[FilmStripTile] = []
    for t, mask, (cy, cx), color in zip(
        occupied, masks, centroids, colors, strict=True
    ):
        crop = _centered_crop(intensity[t], cy, cx, size).astype(float)
        norm = np.zeros_like(crop) if hi <= lo else np.clip((crop - lo) / (hi - lo), 0, 1)
        rgb = _apply_colormap(norm, colormap)

        mask_crop = _centered_crop(mask, cy, cx, size)
        spotlight = _binary_dilate(mask_crop, spotlight_dilation)
        rgb[~spotlight] *= spotlight_dim

        # Border on the inner edge of the bright spotlight, so the coloured
        # contour lands exactly on the bright/dim boundary (no bright ring left
        # outside it). Grown inward from the boundary so it stays robust even
        # when the spotlight fills the whole crop.
        border = _binary_dilate(_mask_outline(spotlight), outline_width - 1) & spotlight
        rgb[border] = outline_color if outline_color is not None else color[:3]

        tiles.append(
            FilmStripTile(
                frame=t,
                rgb=(np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8),
                validated=t in validated,
                anchored=t in anchored,
            )
        )

    return TrackFilmStrip(tiles=tuple(tiles))
