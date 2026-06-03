"""Docked track film-strip state for the nucleus correction workflow.

This owns the *docked panel* half of the film strip: building the per-frame
crop strip for the selected track, docking/undocking the panel, keeping the
current-frame highlight in sync, and adapting tile size. The pixels themselves
come from the pure :func:`build_track_film_strip` helper and are laid out by
:class:`TrackFilmStripPanel`; this controller is the glue the correction widget
used to carry inline.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

from cellflow.database.validation import read_corrections, read_validated_tracks
from cellflow.napari._correction_film_strip import TrackFilmStripPanel
from cellflow.napari._correction_track_path import (
    TrackFilmStrip,
    build_track_film_strip,
)

logger = logging.getLogger(__name__)


class FilmStripController:
    """Own the docked per-frame film strip for the selected correction track."""

    def __init__(
        self,
        viewer,
        *,
        tracked_layer_provider: Callable[[], object | None],
        intensity_layer_provider: Callable[[], object | None],
        pos_dir_provider: Callable[[], Path | None],
        current_t_provider: Callable[[], int],
        selected_label_provider: Callable[[], int],
        tile_px: int = 96,
    ) -> None:
        self.viewer = viewer
        self._tracked_layer_provider = tracked_layer_provider
        self._intensity_layer_provider = intensity_layer_provider
        self._pos_dir_provider = pos_dir_provider
        self._current_t_provider = current_t_provider
        self._selected_label_provider = selected_label_provider
        self._tile_px = int(tile_px)
        self._panel: TrackFilmStripPanel | None = None
        self._dock = None

    def refresh(self) -> None:
        """Rebuild the strip for the selected track (clears it if none/loadable)."""
        lab = int(self._selected_label_provider() or 0)
        tracked = self._tracked_layer_provider()
        intensity = self._intensity_layer_provider()
        if not lab or tracked is None or intensity is None:
            if self._panel is not None:
                self._panel.set_strip(
                    TrackFilmStrip(tiles=()), title="No track selected"
                )
            return
        validated_frames, anchored_frames = self._validated_anchored_frames(lab)
        strip = build_track_film_strip(
            np.asarray(tracked.data),
            np.asarray(intensity.data),
            lab,
            colormap=self._film_strip_colormap(intensity),
            outline_color=self._track_outline_color(lab),
            validated_frames=validated_frames,
            anchored_frames=anchored_frames,
        )
        panel = self._ensure_panel()
        panel.set_strip(strip, title=f"Track {lab} — {len(strip.frames)} frame(s)")
        panel.set_current_frame(self._current_t_provider())

    def set_tile_size(self, value: int) -> None:
        """Change the on-screen tile size; applies live if a panel is docked."""
        self._tile_px = int(value)
        if self._panel is not None:
            self._panel.set_tile_size(self._tile_px)

    def set_current_frame(self, frame: int) -> None:
        """Move the current-frame highlight without rebuilding the strip."""
        if self._panel is not None:
            self._panel.set_current_frame(frame)

    def teardown(self) -> None:
        """Undock and forget the panel (next refresh re-creates it)."""
        if self._dock is not None:
            try:
                self.viewer.window.remove_dock_widget(self._dock)
            except Exception:
                logger.exception("could not remove the track film strip dock")
        self._dock = None
        self._panel = None

    def _ensure_panel(self) -> TrackFilmStripPanel:
        if self._panel is not None:
            return self._panel
        panel = TrackFilmStripPanel(tile_px=self._tile_px)
        panel.frame_clicked.connect(self._on_frame_clicked)
        self._panel = panel
        try:
            self._dock = self.viewer.window.add_dock_widget(
                panel, name="Track film strip", area="bottom"
            )
        except Exception:
            logger.exception("could not dock the track film strip")
            self._dock = None
        return panel

    def _track_outline_color(self, lab: int):
        """RGB (0..1) the tracked labels layer paints cell ``lab`` with, or None."""
        layer = self._tracked_layer_provider()
        color_dict = getattr(getattr(layer, "colormap", None), "color_dict", None)
        try:
            raw = color_dict.get(int(lab)) if color_dict is not None else None
        except Exception:
            raw = None
        if raw is None or isinstance(raw, str):
            return None
        rgba = np.asarray(raw, dtype=float).ravel()
        if rgba.size < 3:
            return None
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]))

    def _validated_anchored_frames(self, lab: int) -> tuple[set[int], set[int]]:
        """Frames where cell ``lab`` is validated / anchored (empty if no project)."""
        pos_dir = self._pos_dir_provider()
        if pos_dir is None:
            return set(), set()
        validated = {
            int(f) for f in read_validated_tracks(pos_dir).get(int(lab), set())
        }
        anchored = {
            int(correction.t)
            for correction in read_corrections(pos_dir)
            if correction.kind == "anchor" and int(correction.cell_id) == int(lab)
        }
        return validated, anchored

    @staticmethod
    def _film_strip_colormap(layer):
        """Adapt the intensity layer's colormap (e.g. 'I Purple') to a (h,w)->RGB map."""
        cmap = getattr(layer, "colormap", None)
        if cmap is None or not hasattr(cmap, "map"):
            return None

        def _map(values: np.ndarray) -> np.ndarray:
            flat = np.asarray(values, dtype=float).ravel()
            mapped = np.asarray(cmap.map(flat), dtype=float)
            return mapped.reshape(values.shape + (mapped.shape[-1],))

        return _map

    def _on_frame_clicked(self, frame: int) -> None:
        try:
            step = list(self.viewer.dims.current_step)
            step[0] = int(frame)
            self.viewer.dims.current_step = tuple(step)
        except Exception:
            logger.exception("film strip frame jump failed")
