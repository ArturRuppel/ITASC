"""Validated-track overlay state for the nucleus correction workflow."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

import numpy as np
from napari.utils.colormaps import direct_colormap

from itasc.tracking_ultrack.validation_state import (
    invalidate_track,
    read_corrections,
    read_validated_tracks,
)

VALIDATED_OVERLAY = "[Correction] Validated: Nucleus"
LEGACY_VALIDATED_OVERLAY = "Validated: Nucleus"
ANCHOR_OVERLAY = "[Correction] Anchors: Nucleus"
LEGACY_ANCHOR_OVERLAY = "Anchors: Nucleus"
SPOTLIGHT_LAYER = "[Correction] CellSpotlight"
VALIDATED_OVERLAY_OPACITY = 0.4
VALIDATED_OVERLAY_COLOR = "#00ff00"
ANCHOR_OVERLAY_COLOR = "#b39400"
# In the filled-by-ID view (images hidden) a translucent wash would hide the
# per-ID colours, so the overlay is drawn as an opaque border of this thickness.
VALIDATED_OVERLAY_CONTOUR = 2


class ValidatedOverlayController:
    """Own validated-cell overlay rendering and validation counter updates."""

    def __init__(
        self,
        viewer,
        *,
        tracked_layer_provider: Callable[[], object | None],
        pos_dir_provider: Callable[[], Path | None],
        owned_layers: set[str],
    ) -> None:
        self.viewer = viewer
        self._tracked_layer_provider = tracked_layer_provider
        self._pos_dir_provider = pos_dir_provider
        self._owned_layers = owned_layers
        # When True, overlays render as an opaque border instead of a wash.
        self._border_mode = False

    def set_border_mode(self, enabled: bool) -> None:
        """Switch overlays between translucent wash and opaque border.

        Used by the filled-by-ID viewer mode: a wash would obscure the per-ID
        label colours, so validated/anchor cells are shown as a coloured border.
        Applies to existing overlay layers immediately and is remembered so
        later overlay rebuilds keep the chosen style.
        """
        self._border_mode = bool(enabled)
        for name in (VALIDATED_OVERLAY, ANCHOR_OVERLAY):
            if name in self.viewer.layers:
                self._apply_overlay_style(self.viewer.layers[name])

    def _apply_overlay_style(self, layer) -> None:
        try:
            if self._border_mode:
                layer.contour = VALIDATED_OVERLAY_CONTOUR
                layer.opacity = 1.0
            else:
                layer.contour = 0
                layer.opacity = VALIDATED_OVERLAY_OPACITY
        except Exception:
            pass

    def refresh_overlay(self, frame_view_2d=None) -> None:
        # ``frame_view_2d`` is accepted (and ignored) for call-site compatibility:
        # the overlays are whole-stack masks now, so napari slices the current
        # frame for us — no per-frame rebuild needed. See refresh_validated_overlay.
        self.refresh_validated_overlay()
        self.refresh_anchor_overlay()

    def refresh_validated_overlay(self, frame_view_2d=None) -> None:
        """Paint validated cells across *every* frame they're validated in.

        Builds the whole ``(T, H, W)`` mask once from the validated-tracks store
        rather than just the current frame, so scrubbing the time slider does not
        trigger a rebuild — napari shows the right slice itself. Re-run only when
        the validations or the labels actually change.
        """
        tracked = self._tracked_layer_provider()
        pos_dir = self._pos_dir_provider()
        if pos_dir is None or tracked is None:
            self.remove_overlay_layers()
            return
        data = getattr(tracked, "data", None)
        if data is None or data.ndim < 3:
            return
        validated_tracks = read_validated_tracks(pos_dir)
        overlay_exists = VALIDATED_OVERLAY in self.viewer.layers
        if not validated_tracks and not overlay_exists:
            return
        full = self._mask_stack(np.asarray(data), validated_tracks)
        if overlay_exists:
            self.viewer.layers[VALIDATED_OVERLAY].data = full
        else:
            self.add_overlay(full)

    def refresh_anchor_overlay(self, frame_view_2d=None) -> None:
        """Paint anchored cells across every frame they're anchored in (whole stack)."""
        tracked = self._tracked_layer_provider()
        pos_dir = self._pos_dir_provider()
        if pos_dir is None or tracked is None:
            self.remove_overlay_layers()
            return
        data = getattr(tracked, "data", None)
        if data is None or data.ndim < 3:
            return
        anchor_tracks: dict[int, set[int]] = {}
        for correction in read_corrections(pos_dir):
            if correction.kind == "anchor":
                anchor_tracks.setdefault(int(correction.cell_id), set()).add(
                    int(correction.t)
                )
        overlay_exists = ANCHOR_OVERLAY in self.viewer.layers
        if not anchor_tracks and not overlay_exists:
            return
        full = self._mask_stack(np.asarray(data), anchor_tracks)
        if overlay_exists:
            self.viewer.layers[ANCHOR_OVERLAY].data = full
        else:
            self.add_anchor_overlay(full)

    @staticmethod
    def _mask_stack(data: np.ndarray, tracks: dict[int, set[int]]) -> np.ndarray:
        """A ``(T, H, W)`` uint8 mask: 1 where ``cell_id`` lives in its frames."""
        full = np.zeros(data.shape, dtype=np.uint8)
        n_frames = data.shape[0]
        for cell_id, frames in tracks.items():
            for frame in frames:
                frame = int(frame)
                if 0 <= frame < n_frames:
                    full[frame][data[frame] == int(cell_id)] = 1
        return full

    def add_overlay(self, data: np.ndarray) -> None:
        if VALIDATED_OVERLAY in self.viewer.layers:
            layer = self.viewer.layers[VALIDATED_OVERLAY]
            layer.data = data
            layer.colormap = direct_colormap(
                {None: (0, 0, 0, 0), 1: VALIDATED_OVERLAY_COLOR}
            )
            self._apply_overlay_style(layer)
            self._owned_layers.add(VALIDATED_OVERLAY)
            self.place_below_spotlight()
            return
        if LEGACY_VALIDATED_OVERLAY in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[LEGACY_VALIDATED_OVERLAY])
        layer = self.viewer.add_labels(
            data,
            name=VALIDATED_OVERLAY,
            opacity=VALIDATED_OVERLAY_OPACITY,
            colormap=direct_colormap(
                {None: (0, 0, 0, 0), 1: VALIDATED_OVERLAY_COLOR}
            ),
        )
        self._apply_overlay_style(layer)
        self._owned_layers.add(VALIDATED_OVERLAY)
        self.place_below_spotlight()
        tracked = self._tracked_layer_provider()
        if tracked is not None:
            self.viewer.layers.selection.active = tracked

    def add_anchor_overlay(self, data: np.ndarray) -> None:
        if ANCHOR_OVERLAY in self.viewer.layers:
            layer = self.viewer.layers[ANCHOR_OVERLAY]
            layer.data = data
            layer.colormap = direct_colormap(
                {None: (0, 0, 0, 0), 1: ANCHOR_OVERLAY_COLOR}
            )
            self._apply_overlay_style(layer)
            self._owned_layers.add(ANCHOR_OVERLAY)
            self.place_below_spotlight()
            return
        if LEGACY_ANCHOR_OVERLAY in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[LEGACY_ANCHOR_OVERLAY])
        layer = self.viewer.add_labels(
            data,
            name=ANCHOR_OVERLAY,
            opacity=VALIDATED_OVERLAY_OPACITY,
            colormap=direct_colormap({None: (0, 0, 0, 0), 1: ANCHOR_OVERLAY_COLOR}),
        )
        self._apply_overlay_style(layer)
        self._owned_layers.add(ANCHOR_OVERLAY)
        self.place_below_spotlight()
        tracked = self._tracked_layer_provider()
        if tracked is not None:
            self.viewer.layers.selection.active = tracked

    def place_below_spotlight(self) -> None:
        if SPOTLIGHT_LAYER not in self.viewer.layers:
            return
        for name in (VALIDATED_OVERLAY, ANCHOR_OVERLAY):
            if name not in self.viewer.layers:
                continue
            overlay_index = self.viewer.layers.index(name)
            spotlight_index = self.viewer.layers.index(SPOTLIGHT_LAYER)
            if overlay_index > spotlight_index:
                self.viewer.layers.move(overlay_index, spotlight_index)

    def remove_overlay_layers(self) -> None:
        for name in (
            VALIDATED_OVERLAY,
            LEGACY_VALIDATED_OVERLAY,
            ANCHOR_OVERLAY,
            LEGACY_ANCHOR_OVERLAY,
        ):
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])
                self._owned_layers.discard(name)

    def refresh_counter(self, label) -> None:
        pos_dir = self._pos_dir_provider()
        if pos_dir is None or self._tracked_layer_provider() is None:
            label.setText("")
            return
        validated_tracks = read_validated_tracks(pos_dir)
        n_tracks = len(validated_tracks)
        n_cell_frames = sum(len(frames) for frames in validated_tracks.values())
        label.setText(
            f"{n_tracks} track(s) validated, {n_cell_frames} cell-frame(s) covered"
        )

    def on_cells_edited(
        self,
        t: int,
        changed_ids: set[int],
        *,
        frame_view_2d: Callable[[np.ndarray, int], np.ndarray | None],
        counter_label,
    ) -> None:
        pos_dir = self._pos_dir_provider()
        if pos_dir is None:
            return
        for cell_id in changed_ids:
            invalidate_track(pos_dir, cell_id)
        self.refresh_overlay(frame_view_2d)
        self.refresh_counter(counter_label)

    def frames_with_cell(self, cell_id: int) -> list[int]:
        layer = self._tracked_layer_provider()
        if cell_id == 0 or layer is None:
            return []
        data = getattr(layer, "data", None)
        if data is None or data.ndim < 3:
            return []
        spatial_axes = tuple(range(1, data.ndim))
        present = np.any(data == cell_id, axis=spatial_axes)
        return [int(t) for t in np.where(present)[0]]
