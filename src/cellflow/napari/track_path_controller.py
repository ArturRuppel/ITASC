"""Whole-track temporal overlay ("comet") for the nucleus correction workflow.

This owns the in-canvas comet: the fading viridis image layer that paints the
selected track's entire trajectory (oldest→newest), plus the spotlight mask the
inner correction widget consults to highlight that track's union footprint. The
pixels come from the pure :func:`build_track_path_overlay` helper; this
controller is the layer-lifecycle glue the correction widget used to carry
inline.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from cellflow.napari._correction_track_path import build_track_path_overlay

TRACK_PATH_LAYER = "[Correction] Track Path"
TRACK_PATH_NUMBERS_LAYER = "[Correction] Track Path Numbers"
TRACK_PATH_OPACITY = 1.0


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
        overlay = build_track_path_overlay(np.asarray(layer.data), lab)
        if overlay.is_empty():
            self.clear()
            return
        self._update_layers(overlay)
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

    def _update_layers(self, overlay) -> None:
        from napari.layers import Image

        name = TRACK_PATH_LAYER
        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Image):
            layer = self.viewer.layers[name]
            layer.data = overlay.overlay
            layer.opacity = TRACK_PATH_OPACITY
        else:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
            self.viewer.add_image(
                overlay.overlay,
                name=name,
                rgb=True,
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
