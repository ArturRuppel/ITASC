"""Whole-track overlay for the nucleus correction workflow.

This owns the in-canvas overlay: a napari Tracks layer that draws the selected
track's trajectory as a vector polyline through the per-frame nucleus centroids,
coloured by time with viridis (earliest frame dark, latest yellow), plus the
spotlight mask the inner correction widget consults to highlight the union of
every frame's mask (full brightness inside that footprint, darkened outside,
with a sharp boundary). The geometry comes from the pure
:func:`build_track_path_overlay` helper; this controller is the layer-lifecycle
glue the correction widget used to carry inline.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from cellflow.napari._correction_track_path import build_track_path_overlay

TRACK_PATH_LAYER = "[Correction] Track Path"
TRACK_PATH_NUMBERS_LAYER = "[Correction] Track Path Numbers"
TRACK_PATH_OPACITY = 1.0
TRACK_PATH_TAIL_WIDTH = 4


class TrackPathController:
    """Own the comet overlay layer and its spotlight mask for one track."""

    def __init__(
        self,
        viewer,
        *,
        tracked_layer_provider: Callable[[], object | None],
        selected_label_provider: Callable[[], int],
        enabled_provider: Callable[[], bool],
        status_callback: Callable[[str], None],
        owned_layers: set[str],
    ) -> None:
        self.viewer = viewer
        self._tracked_layer_provider = tracked_layer_provider
        self._selected_label_provider = selected_label_provider
        self._enabled_provider = enabled_provider
        self._status = status_callback
        self._owned_layers = owned_layers

    def refresh(self) -> None:
        """Rebuild the comet for the selected track (clears it if off/empty)."""
        lab = int(self._selected_label_provider() or 0)
        if not self._enabled_provider() or not lab:
            self.clear()
            return
        layer = self._tracked_layer_provider()
        if layer is None:
            self.clear()
            return
        data = np.asarray(layer.data)
        overlay = build_track_path_overlay(data, lab)
        if overlay.is_empty():
            self.clear()
            return
        n_frames = int(data.shape[0]) if data.ndim == 3 else 1
        self._update_layers(overlay, lab, n_frames)
        self._status(
            f"Track path: cell {lab} across {len(overlay.frames)} frame(s)."
        )

    def clear(self) -> None:
        """Remove the comet layers from the viewer."""
        for name in (TRACK_PATH_LAYER, TRACK_PATH_NUMBERS_LAYER):
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])
            self._owned_layers.discard(name)

    def spotlight_mask(self, _t: int, lab: int, _default_mask):
        """Spotlight the union of the selected track's masks while the comet is on."""
        if not self._enabled_provider() or not lab:
            return None
        layer = self._tracked_layer_provider()
        if layer is None:
            return None
        data = np.asarray(layer.data)
        if data.ndim != 3:
            return None
        union = np.any(data == int(lab), axis=0)
        return union if union.any() else None

    def _update_layers(self, overlay, lab: int, n_frames: int) -> None:
        from napari.layers import Tracks

        data, properties = self._tracks_data(overlay, lab)
        # Long tail + head so the whole trajectory stays visible on every frame,
        # rather than growing/shrinking as the time slider moves.
        span = max(int(n_frames), 1)

        name = TRACK_PATH_LAYER
        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Tracks):
            layer = self.viewer.layers[name]
            # Park color_by on the always-present 'track_id' before swapping data:
            # assigning ``data`` resets features to the default, so leaving it on
            # 'time' makes napari warn about a missing key and fall back. Restore
            # 'time' once the new properties carry it again.
            layer.color_by = "track_id"
            layer.data = data
            layer.properties = properties
            layer.color_by = "time"
            layer.colormap = "viridis"
            layer.tail_length = span
            layer.head_length = span
        else:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
            self.viewer.add_tracks(
                data,
                name=name,
                properties=properties,
                color_by="time",
                colormap="viridis",
                tail_width=TRACK_PATH_TAIL_WIDTH,
                tail_length=span,
                head_length=span,
                blending="translucent",
                opacity=TRACK_PATH_OPACITY,
            )
        self._owned_layers.add(name)

        # The per-frame number labels are intentionally not drawn anymore; drop
        # any layer left over from an earlier session.
        nname = TRACK_PATH_NUMBERS_LAYER
        if nname in self.viewer.layers:
            self.viewer.layers.remove(nname)
        self._owned_layers.discard(nname)

        try:
            self.viewer.layers.selection.active = self._tracked_layer_provider()
        except Exception:
            pass

    @staticmethod
    def _tracks_data(overlay, lab: int):
        """Build napari Tracks ``data`` + ``properties`` from the overlay.

        ``data`` rows are ``[track_id, t, y, x]`` (one per occupied frame) and
        ``properties['time']`` carries the frame index so the layer can colour
        the polyline by time with viridis.
        """
        frames = np.asarray(overlay.frames, dtype=float)
        centroids = np.asarray(overlay.centroids, dtype=float).reshape((-1, 2))
        track_ids = np.full(len(frames), float(int(lab)))
        data = np.column_stack(
            [track_ids, frames, centroids[:, 0], centroids[:, 1]]
        )
        return data, {"time": frames}
