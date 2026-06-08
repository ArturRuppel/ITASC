"""Shared napari track-rendering helpers.

Pure numpy / napari colormap maths reused by both the contact-analysis
visualization (``cellflow-contact``) and the nucleus-correction track overview
(``cellflow-tracking``). It lives in ``cellflow-core`` so neither piece has to
depend on the other.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_UNLABELED_COLOR = np.array([0.7, 0.7, 0.7, 1.0], dtype=float)


def _track_label_color_styling(
    track_ids: np.ndarray,
    color_map: dict[int | None, tuple[float, float, float, float] | str],
) -> tuple[Any, np.ndarray] | None:
    """Colour each track exactly like its cell label.

    A ``Tracks`` layer colours by mapping a property through a colormap. When a
    custom colormap is supplied via ``colormaps_dict`` napari feeds it the *raw*
    property values (no normalisation), so we build a per-vertex ``label_pos``
    property in ``[0, 1]`` plus a step (``interpolation="zero"``) colormap whose
    bins reproduce ``color_map`` for each track. Returns ``(colormap, label_pos)``
    or ``None`` when there are no tracks / napari is unavailable.
    """
    ids = sorted({int(i) for i in np.asarray(track_ids).tolist()})
    if not ids:
        return None
    try:
        from napari.utils.colormaps import Colormap
    except Exception:  # pragma: no cover - napari compatibility
        return None

    def _color(cell_id: int) -> tuple[float, float, float, float]:
        raw = color_map.get(int(cell_id))
        # ``color_map`` may come straight from a caller (tuple/list values) or be
        # read back off a napari ``DirectLabelColormap``, which normalises every
        # entry to an ``np.ndarray``. Accept any length-4 numeric sequence so the
        # latter doesn't silently fall through to the grey unlabeled colour.
        if isinstance(raw, (tuple, list, np.ndarray)) and len(raw) == 4:
            return tuple(float(c) for c in raw)
        return tuple(float(c) for c in _UNLABELED_COLOR)

    colors = [_color(i) for i in ids]
    if len(ids) == 1:
        cmap = Colormap([colors[0], colors[0]], controls=[0.0, 1.0])
        pos_by_id = {ids[0]: 0.0}
    else:
        vals = [k / (len(ids) - 1) for k in range(len(ids))]
        pos_by_id = {ids[k]: vals[k] for k in range(len(ids))}
        edges = (
            [0.0]
            + [(vals[k - 1] + vals[k]) / 2 for k in range(1, len(ids))]
            + [1.0]
        )
        cmap = Colormap(colors, controls=edges, interpolation="zero")
    label_pos = np.array(
        [pos_by_id[int(i)] for i in np.asarray(track_ids).tolist()], dtype=float
    )
    return cmap, label_pos


def _nucleus_centroids_by_track(
    nucleus_labels: np.ndarray,
) -> dict[int, list[tuple[int, float, float]]]:
    labels = np.asarray(nucleus_labels)
    if labels.ndim == 2:
        labels = labels[np.newaxis, ...]
    if labels.ndim > 3:
        labels = np.squeeze(labels)
    if labels.ndim != 3:
        raise ValueError(
            f"Expected time-first 2D/3D nucleus labels, got shape {nucleus_labels.shape}"
        )

    centroids: dict[int, list[tuple[int, float, float]]] = {}
    for frame_idx in range(labels.shape[0]):
        frame = labels[frame_idx]
        flat = frame.ravel()
        order = np.argsort(flat, kind="stable")
        sorted_ids = flat[order]

        change = np.empty(len(sorted_ids), dtype=bool)
        change[0] = True
        np.not_equal(sorted_ids[1:], sorted_ids[:-1], out=change[1:])
        boundaries = np.flatnonzero(change)

        rows_all, cols_all = np.divmod(order, frame.shape[1])

        ends = np.empty_like(boundaries)
        ends[:-1] = boundaries[1:]
        ends[-1] = len(sorted_ids)

        for bi in range(len(boundaries)):
            cell_id = int(sorted_ids[boundaries[bi]])
            if cell_id == 0:
                continue
            s, e = int(boundaries[bi]), int(ends[bi])
            y = float(rows_all[s:e].mean())
            x = float(cols_all[s:e].mean())
            centroids.setdefault(cell_id, []).append((frame_idx, y, x))

    return centroids


__all__ = [
    "_UNLABELED_COLOR",
    "_nucleus_centroids_by_track",
    "_track_label_color_styling",
]
