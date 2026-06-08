"""Pure geometry/colour helpers for correction-mode track rendering.

Two concerns live here, both Qt/napari-free so they unit-test on their own:

* :func:`build_all_tracks_data` feeds a single napari ``Tracks`` layer that draws
  *every* track as the overview; focus mode then slices out the selected track's
  vertices (by ``row_index``) and colours them by time.
* :func:`build_track_film_strip` and the crop helpers build the per-frame film
  strip used by the lineage canvas / candidate gallery.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cellflow.napari._track_render import (
    _nucleus_centroids_by_track,
)


def build_all_tracks_data(
    tracked_stack: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[int, np.ndarray]]:
    """Build napari ``Tracks`` ``data`` + ``properties`` for *every* track.

    ``tracked_stack`` is a ``(T, H, W)`` label array (a bare ``(H, W)`` plane is
    treated as a single frame). Returns ``(data, properties, row_index)``:

    * ``data`` — ``(N, 4)`` float rows ``[track_id, t, y, x]`` through each
      track's per-frame nucleus centroids, grouped by track and time-ascending.
    * ``properties`` — ``track_id`` (per vertex) for the overview colouring and
      ``time`` (per-track normalised 0→1 oldest→newest) for the focused track's
      viridis time gradient.
    * ``row_index`` — ``{track_id: row positions into data}`` so focus can slice
      out a single track's vertices without rescanning the stack.

    Every returned array shares one row order, so ``row_index`` indexes both
    ``data`` and the property arrays.
    """
    centroids = _nucleus_centroids_by_track(tracked_stack)

    rows: list[tuple[float, float, float, float]] = []
    track_ids: list[float] = []
    times: list[float] = []
    row_index: dict[int, np.ndarray] = {}

    cursor = 0
    for track_id in sorted(centroids):
        points = sorted(centroids[track_id])  # by frame (already, but be explicit)
        n = len(points)
        row_index[int(track_id)] = np.arange(cursor, cursor + n)
        cursor += n

        frames = np.asarray([p[0] for p in points], dtype=float)
        if n > 1 and frames.max() > frames.min():
            norm = (frames - frames.min()) / (frames.max() - frames.min())
        else:
            norm = np.zeros(n, dtype=float)

        for (t, y, x), nt in zip(points, norm):
            rows.append((float(track_id), float(t), float(y), float(x)))
            track_ids.append(float(track_id))
            times.append(float(nt))

    if not rows:
        empty = np.empty(0, dtype=float)
        return (
            np.empty((0, 4), dtype=float),
            {"track_id": empty, "time": empty},
            {},
        )

    properties = {
        "track_id": np.asarray(track_ids, dtype=float),
        "time": np.asarray(times, dtype=float),
    }
    return np.asarray(rows, dtype=float), properties, row_index


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


@dataclass(frozen=True)
class FilmStripTile:
    """One frame's panel in the film strip: an RGB crop with the mask outlined.

    ``validated`` / ``anchored`` flag whether this frame is a validated or an
    anchored frame for the track; the view draws a coloured marker strip for
    each. They are view metadata, not baked into ``rgb``.

    ``placeholder`` marks a frame the track does *not* occupy: an empty (blank)
    tile emitted only to keep an incomplete track's strip aligned to the movie
    timeline, so the gaps read as missing frames rather than a shorter strip.
    """

    frame: int            # source frame index
    rgb: np.ndarray       # (h, w, 3) uint8, raw crop with the mask edge drawn on
    validated: bool = False
    anchored: bool = False
    placeholder: bool = False

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


def render_crop_tile(
    intensity_2d: np.ndarray,
    mask: np.ndarray,
    cy: int,
    cx: int,
    size: int,
    *,
    lo: float,
    hi: float,
    colormap=None,
    outline_color: tuple[float, float, float],
    outline_width: int = 2,
    spotlight_dim: float = 0.35,
    spotlight_dilation: int = 2,
) -> np.ndarray:
    """Render one spotlighted, outlined crop centered on ``(cy, cx)``.

    Crops ``intensity_2d`` to a ``size``x``size`` window, contrast-stretches it
    against the shared ``[lo, hi]`` range, colors it through ``colormap``
    (grayscale if ``None``), dims everything outside ``mask`` by ``spotlight_dim``,
    and draws a ``outline_width``-thick ``outline_color`` border on the inner edge
    of the bright region. Returns ``(size, size, 3)`` uint8.

    Shared by the per-frame film strip and the per-candidate gallery so both read
    identically; callers own the window ``size``, contrast range, and outline
    colour (e.g. the film strip falls back to a per-frame viridis colour).
    """
    crop = _centered_crop(intensity_2d, cy, cx, size).astype(float)
    norm = np.zeros_like(crop) if hi <= lo else np.clip((crop - lo) / (hi - lo), 0, 1)
    rgb = _apply_colormap(norm, colormap)

    mask_crop = _centered_crop(mask, cy, cx, size)
    spotlight = _binary_dilate(mask_crop, spotlight_dilation)
    rgb[~spotlight] *= spotlight_dim

    # Border on the inner edge of the bright spotlight, so the coloured contour
    # lands exactly on the bright/dim boundary (no bright ring left outside it).
    border = _binary_dilate(_mask_outline(spotlight), outline_width - 1) & spotlight
    rgb[border] = outline_color
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


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
    frames: Sequence[int] | None = None,
    total_frames: int | None = None,
) -> TrackFilmStrip:
    """Build per-frame, nucleus-centered intensity crops for ``track_id``.

    ``tracked_stack`` and ``intensity_stack`` are matching ``(T, H, W)`` arrays
    (bare ``(H, W)`` planes are treated as a single frame). For each occupied
    frame the intensity is cropped to a fixed square window centered on the
    nucleus, contrast-stretched against the track's own nucleus pixels, colored
    through ``colormap`` (e.g. the layer's "bop purple"; grayscale if ``None``),
    and dimmed outside the nucleus by ``spotlight_dim`` for a spotlight effect.

    A ``outline_width``-thick border is drawn at the *inner edge of the bright
    spotlight region*, so the coloured contour coincides with the bright/dim
    boundary instead of leaving a bright ring outside it. The border uses
    ``outline_color`` (the label layer's colour for this track, RGB in 0..1);
    when ``None`` it falls back to the frame's viridis time colour.

    ``validated_frames`` / ``anchored_frames`` (sets of frame indices) flag each
    tile so the view can mark validated/anchored frames. ``margin`` pads the
    window around the largest nucleus.

    ``frames`` optionally restricts the scan to a known set of occupied frame
    indices (e.g. supplied by the lineage graph), so callers building strips for
    many tracks at once avoid re-scanning every empty frame per track.

    ``total_frames``, when given, pads the strip to one tile per movie frame in
    ``range(total_frames)``: frames the track does not occupy get a blank
    ``placeholder`` tile, so an *incomplete* track's strip stays aligned to the
    timeline with its missing frames shown as empty thumbnails.
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

    if frames is None:
        scan = range(tracked.shape[0])
    else:
        scan = [int(f) for f in frames if 0 <= int(f) < tracked.shape[0]]

    occupied: list[int] = []
    masks: list[np.ndarray] = []
    centroids: list[tuple[int, int]] = []
    extents: list[int] = []
    for t in scan:
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
    rendered: dict[int, FilmStripTile] = {}
    for t, mask, (cy, cx), color in zip(
        occupied, masks, centroids, colors, strict=True
    ):
        # Per-frame outline colour falls back to the viridis time colour; the
        # bright/dim spotlight and inner-edge border are handled by the renderer.
        rgb = render_crop_tile(
            intensity[t],
            mask,
            cy,
            cx,
            size,
            lo=lo,
            hi=hi,
            colormap=colormap,
            outline_color=outline_color if outline_color is not None else tuple(color[:3]),
            outline_width=outline_width,
            spotlight_dim=spotlight_dim,
            spotlight_dilation=spotlight_dilation,
        )
        rendered[t] = FilmStripTile(
            frame=t,
            rgb=rgb,
            validated=t in validated,
            anchored=t in anchored,
        )

    # Without ``total_frames`` the strip is just the occupied tiles (oldest-first);
    # with it, every movie frame gets a tile so missing frames show as empty
    # placeholders and the strip stays aligned to the timeline.
    if total_frames is not None and int(total_frames) > 0:
        display = range(int(total_frames))
    else:
        display = occupied
    tiles = [
        rendered[t]
        if t in rendered
        else FilmStripTile(
            frame=int(t),
            rgb=np.zeros((size, size, 3), dtype=np.uint8),
            placeholder=True,
        )
        for t in display
    ]

    return TrackFilmStrip(tiles=tuple(tiles))
