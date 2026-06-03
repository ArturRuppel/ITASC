"""Build thumbnail strips for extend / swap *candidates* at a single frame.

The correction workspace shows three side-by-side galleries — extend-backward,
swap, extend-forward — each a column of clickable thumbnails of the alternative
segmentations on offer. Where :func:`build_track_film_strip` crops one track
*across frames*, this crops a set of *candidate masks on one frame*: each
candidate is a full-frame boolean mask (from ``list_swap_candidates`` or
``list_extend_candidates``), rendered through the shared
:func:`~cellflow.napari._correction_track_path.render_crop_tile` so a candidate
thumbnail reads identically to a film-strip tile.

Pure module: no Qt, no napari, so it is unit-testable on its own. The view half
(a clickable column) and the controller that maps a click back to an apply live
elsewhere.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from cellflow.napari._correction_track_path import render_crop_tile

# Neutral outline when the caller does not pass the selected track's colour.
_DEFAULT_OUTLINE = (0.75, 0.75, 0.75)


@dataclass(frozen=True)
class CandidateSpec:
    """One candidate to render: a routing ``key`` and its full-frame mask.

    ``key`` is whatever the controller needs to apply the candidate later (the
    node id for swap, the candidate label for extend); ``caption`` overrides the
    default ``"<area> px"`` label when the caller has something better to show.
    """

    key: int
    mask: np.ndarray            # (H, W) bool, full frame
    caption: str = ""


@dataclass(frozen=True)
class CandidateTile:
    """One rendered candidate thumbnail plus the metadata the view shows."""

    key: int
    rgb: np.ndarray             # (size, size, 3) uint8
    area: int
    caption: str

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        return self.rgb.shape[1]


@dataclass(frozen=True)
class CandidateStrip:
    """The rendered candidate tiles for one gallery column, in input order."""

    tiles: tuple[CandidateTile, ...] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        return len(self.tiles) == 0

    @property
    def keys(self) -> tuple[int, ...]:
        return tuple(tile.key for tile in self.tiles)


def _mask_centroid_extent(mask: np.ndarray) -> tuple[int, int, int] | None:
    """``(cy, cx, extent)`` of ``mask``; ``None`` for an empty mask."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    cy = int(round(float(ys.mean())))
    cx = int(round(float(xs.mean())))
    extent = max(int(ys.max() - ys.min()) + 1, int(xs.max() - xs.min()) + 1)
    return cy, cx, extent


def build_candidate_strip(
    intensity_2d: np.ndarray,
    specs: Sequence[CandidateSpec],
    *,
    margin: int = 6,
    colormap=None,
    outline_color: tuple[float, float, float] | None = None,
    outline_width: int = 2,
    spotlight_dim: float = 0.35,
    spotlight_dilation: int = 2,
) -> CandidateStrip:
    """Render ``specs`` as uniform, mask-centered thumbnails over ``intensity_2d``.

    Every tile is a fixed square window (sized to the largest candidate plus
    ``margin``) centered on that candidate's centroid, so the cell stays put in
    the middle of each thumbnail. Contrast is stretched once against the union of
    all candidate pixels (so the gallery is comparable tile-to-tile) and the mask
    is spotlighted with an inner-edge ``outline_color`` border (a neutral grey if
    ``None``). Empty masks are skipped; an empty ``specs`` yields an empty strip.
    """
    intensity = np.asarray(intensity_2d)
    if intensity.ndim != 2:
        raise ValueError(f"intensity_2d must be 2D, got {intensity.ndim}D")
    shape = intensity.shape

    kept: list[tuple[CandidateSpec, np.ndarray, int, int]] = []
    extents: list[int] = []
    for spec in specs:
        mask = np.asarray(spec.mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(
                f"candidate mask shape {mask.shape} != intensity shape {shape}"
            )
        ce = _mask_centroid_extent(mask)
        if ce is None:
            continue
        cy, cx, extent = ce
        kept.append((spec, mask, cy, cx))
        extents.append(extent)
    if not kept:
        return CandidateStrip(tiles=())

    size = int(max(extents)) + 2 * margin

    # Shared contrast over every candidate's pixels, so brighter/dimmer fragments
    # are judged on the same scale rather than each self-normalising.
    pooled = np.concatenate([intensity[mask] for _, mask, _, _ in kept]).astype(float)
    if pooled.size:
        lo = float(np.percentile(pooled, 2.0))
        hi = float(np.percentile(pooled, 98.0))
    else:  # pragma: no cover - kept implies non-empty masks
        lo, hi = 0.0, 1.0

    color = outline_color if outline_color is not None else _DEFAULT_OUTLINE

    tiles: list[CandidateTile] = []
    for spec, mask, cy, cx in kept:
        rgb = render_crop_tile(
            intensity,
            mask,
            cy,
            cx,
            size,
            lo=lo,
            hi=hi,
            colormap=colormap,
            outline_color=color,
            outline_width=outline_width,
            spotlight_dim=spotlight_dim,
            spotlight_dilation=spotlight_dilation,
        )
        area = int(mask.sum())
        tiles.append(
            CandidateTile(
                key=int(spec.key),
                rgb=rgb,
                area=area,
                caption=spec.caption or f"{area} px",
            )
        )
    return CandidateStrip(tiles=tuple(tiles))


__all__ = ["CandidateSpec", "CandidateTile", "CandidateStrip", "build_candidate_strip"]
