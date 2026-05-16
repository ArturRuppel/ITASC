"""Validated-track overlay state for the nucleus correction workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from napari.utils.colormaps import direct_colormap

from cellflow.database.validation import (
    invalidate_track,
    read_corrections,
    read_validated_cells_at_frame,
    read_validated_tracks,
)

VALIDATED_OVERLAY = "[Correction] Validated: Nucleus"
LEGACY_VALIDATED_OVERLAY = "Validated: Nucleus"
ANCHOR_OVERLAY = "[Correction] Anchors: Nucleus"
LEGACY_ANCHOR_OVERLAY = "Anchors: Nucleus"
SPOTLIGHT_LAYER = "[Correction] CellSpotlight"
VALIDATED_OVERLAY_OPACITY = 0.75
VALIDATED_OVERLAY_COLOR = "#007300"
ANCHOR_OVERLAY_COLOR = "#b39400"


class ValidatedOverlayController:
    """Own validated-cell overlay rendering and validation counter updates."""

    def __init__(
        self,
        viewer,
        *,
        tracked_layer_provider: Callable[[], object | None],
        pos_dir_provider: Callable[[], Path | None],
        current_t_provider: Callable[[], int],
        owned_layers: set[str],
    ) -> None:
        self.viewer = viewer
        self._tracked_layer_provider = tracked_layer_provider
        self._pos_dir_provider = pos_dir_provider
        self._current_t_provider = current_t_provider
        self._owned_layers = owned_layers

    def refresh_overlay(self, frame_view_2d: Callable[[np.ndarray, int], np.ndarray | None]) -> None:
        self.refresh_validated_overlay(frame_view_2d)
        self.refresh_anchor_overlay(frame_view_2d)

    def refresh_validated_overlay(self, frame_view_2d: Callable[[np.ndarray, int], np.ndarray | None]) -> None:
        tracked = self._tracked_layer_provider()
        pos_dir = self._pos_dir_provider()
        if pos_dir is None or tracked is None:
            self.remove_overlay_layers()
            return
        data = getattr(tracked, "data", None)
        if data is None or data.ndim < 3:
            return
        t = self._current_t_provider()
        if t >= data.shape[0]:
            return
        frame = frame_view_2d(data, t)
        if frame is None:
            return
        validated_ids = read_validated_cells_at_frame(pos_dir, t)
        overlay_exists = VALIDATED_OVERLAY in self.viewer.layers
        if not validated_ids and not overlay_exists:
            return
        mask2d = (
            np.isin(frame, list(validated_ids)).astype(np.uint8)
            if validated_ids
            else np.zeros(frame.shape, dtype=np.uint8)
        )
        full = np.zeros(data.shape, dtype=np.uint8)
        full[t] = mask2d
        if overlay_exists:
            self.viewer.layers[VALIDATED_OVERLAY].data = full
        else:
            from qtpy.QtCore import QTimer

            QTimer.singleShot(0, lambda data=full: self.add_overlay(data))

    def refresh_anchor_overlay(self, frame_view_2d: Callable[[np.ndarray, int], np.ndarray | None]) -> None:
        tracked = self._tracked_layer_provider()
        pos_dir = self._pos_dir_provider()
        if pos_dir is None or tracked is None:
            self.remove_overlay_layers()
            return
        data = getattr(tracked, "data", None)
        if data is None or data.ndim < 3:
            return
        t = self._current_t_provider()
        if t >= data.shape[0]:
            return
        frame = frame_view_2d(data, t)
        if frame is None:
            return
        anchor_ids = {
            int(correction.cell_id)
            for correction in read_corrections(pos_dir)
            if correction.kind == "anchor" and int(correction.t) == int(t)
        }
        overlay_exists = ANCHOR_OVERLAY in self.viewer.layers
        if not anchor_ids and not overlay_exists:
            return
        mask2d = (
            np.isin(frame, list(anchor_ids)).astype(np.uint8)
            if anchor_ids
            else np.zeros(frame.shape, dtype=np.uint8)
        )
        full = np.zeros(data.shape, dtype=np.uint8)
        full[t] = mask2d
        if overlay_exists:
            self.viewer.layers[ANCHOR_OVERLAY].data = full
        else:
            from qtpy.QtCore import QTimer

            QTimer.singleShot(0, lambda data=full: self.add_anchor_overlay(data))

    def add_overlay(self, data: np.ndarray) -> None:
        if VALIDATED_OVERLAY in self.viewer.layers:
            layer = self.viewer.layers[VALIDATED_OVERLAY]
            layer.data = data
            layer.opacity = VALIDATED_OVERLAY_OPACITY
            layer.colormap = direct_colormap(
                {None: (0, 0, 0, 0), 1: VALIDATED_OVERLAY_COLOR}
            )
            self._owned_layers.add(VALIDATED_OVERLAY)
            self.place_below_spotlight()
            return
        if LEGACY_VALIDATED_OVERLAY in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[LEGACY_VALIDATED_OVERLAY])
        self.viewer.add_labels(
            data,
            name=VALIDATED_OVERLAY,
            opacity=VALIDATED_OVERLAY_OPACITY,
            colormap=direct_colormap(
                {None: (0, 0, 0, 0), 1: VALIDATED_OVERLAY_COLOR}
            ),
        )
        self._owned_layers.add(VALIDATED_OVERLAY)
        self.place_below_spotlight()
        tracked = self._tracked_layer_provider()
        if tracked is not None:
            self.viewer.layers.selection.active = tracked

    def add_anchor_overlay(self, data: np.ndarray) -> None:
        if ANCHOR_OVERLAY in self.viewer.layers:
            layer = self.viewer.layers[ANCHOR_OVERLAY]
            layer.data = data
            layer.opacity = VALIDATED_OVERLAY_OPACITY
            layer.colormap = direct_colormap(
                {None: (0, 0, 0, 0), 1: ANCHOR_OVERLAY_COLOR}
            )
            self._owned_layers.add(ANCHOR_OVERLAY)
            self.place_below_spotlight()
            return
        if LEGACY_ANCHOR_OVERLAY in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[LEGACY_ANCHOR_OVERLAY])
        self.viewer.add_labels(
            data,
            name=ANCHOR_OVERLAY,
            opacity=VALIDATED_OVERLAY_OPACITY,
            colormap=direct_colormap({None: (0, 0, 0, 0), 1: ANCHOR_OVERLAY_COLOR}),
        )
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
