"""Docked combined-canvas state — assembles the overview + focus detail.

Owns the bottom-docked correction canvas: it builds the per-track swimlane
overview (present runs from :func:`~cellflow.segmentation.lineage.build_lineage`,
status frames from the project's validation records, flagged frames from
:func:`~cellflow.segmentation.error_scan.scan_errors`) and, for the *selected*
track only, the film-strip detail band (via
:func:`~cellflow.napari._correction_track_path.build_track_film_strip`). Building
crops for a single track keeps refresh cheap no matter how many tracks exist.

The overview, lane assembly and crops are pure/testable; this is the glue that
reads layers, docks the panel, and turns a lane/tile click into a viewer jump +
cell selection via ``on_activate``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np

from cellflow.database.validation import read_corrections, read_validated_tracks
from cellflow.napari._correction_lineage_canvas import LaneView, LineageCanvasPanel
from cellflow.napari._correction_track_path import (
    TrackFilmStrip,
    build_track_film_strip,
)
from cellflow.segmentation.error_scan import scan_errors
from cellflow.segmentation.lineage import build_lineage

logger = logging.getLogger(__name__)

_NODE_OUTLINE = (0.75, 0.75, 0.75)  # fallback neutral cell outline for the detail


class LineageCanvasController:
    """Own the docked overview + focus-detail canvas for a correction session."""

    def __init__(
        self,
        viewer,
        *,
        tracked_data_provider: Callable[[], np.ndarray | None],
        tracked_layer_provider: Callable[[], object | None],
        intensity_layer_provider: Callable[[], object | None],
        selected_label_provider: Callable[[], int],
        current_t_provider: Callable[[], int],
        on_activate: Callable[[int, int], None],
        pos_dir_provider: Callable[[], Path | None] | None = None,
        contours_provider: Callable[[], np.ndarray | None] | None = None,
    ) -> None:
        self.viewer = viewer
        self._tracked_data_provider = tracked_data_provider
        self._tracked_layer_provider = tracked_layer_provider
        self._intensity_layer_provider = intensity_layer_provider
        self._selected_label_provider = selected_label_provider
        self._current_t_provider = current_t_provider
        self._on_activate = on_activate
        self._pos_dir_provider = pos_dir_provider
        self._contours_provider = contours_provider
        self._panel: LineageCanvasPanel | None = None
        self._dock = None
        # Cached from the last refresh so a selection can rebuild only the detail.
        self._occupied: dict[int, list[int]] = {}
        self._validated_map: dict[int, set[int]] = {}
        self._anchored_map: dict[int, set[int]] = {}

    def refresh(self) -> None:
        """Rebuild the overview and the detail strip for the current selection."""
        tracked = self._tracked_data_provider()
        if tracked is None:
            if self._panel is not None:
                self._panel.set_overview([], n_frames=0)
            return
        try:
            lanes, n_frames = self._assemble(np.asarray(tracked))
        except Exception:
            logger.exception("lineage overview assembly failed")
            return
        panel = self._ensure_panel()
        panel.set_overview(
            lanes, n_frames=n_frames, title=f"{len(lanes)} track(s)",
        )
        panel.set_current_frame(self._current_t_provider())
        self.set_selection(int(self._selected_label_provider() or 0))

    def _assemble(self, tracked: np.ndarray) -> tuple[list[LaneView], int]:
        model = build_lineage(tracked)
        self._validated_map, self._anchored_map = self._validated_anchored_maps()
        errors_map = self._error_map(tracked)
        self._occupied = {}
        lanes: list[LaneView] = []
        for column, lane in enumerate(model.lanes):
            cid = int(lane.cell_id)
            segments = tuple((int(s.start), int(s.end)) for s in lane.segments)
            self._occupied[cid] = [
                f for s, e in segments for f in range(s, e + 1)
            ]
            lanes.append(LaneView(
                cell_id=cid,
                column=column,
                segments=segments,
                validated=frozenset(self._validated_map.get(cid, ())),
                anchored=frozenset(self._anchored_map.get(cid, ())),
                errors=frozenset(errors_map.get(cid, ())),
            ))
        return lanes, model.n_frames

    def set_selection(self, cell_id: int) -> None:
        """Highlight the selected lane and rebuild only its detail film strip."""
        cell_id = int(cell_id or 0)
        if self._panel is None:
            return
        self._panel.set_selection(cell_id)
        self._panel.set_detail(self._build_detail(cell_id), title=self._detail_title(cell_id))

    def set_current_frame(self, frame: int) -> None:
        """Move the shared frame guide without rebuilding the canvas."""
        if self._panel is not None:
            self._panel.set_current_frame(int(frame))

    def teardown(self) -> None:
        """Undock and forget the panel (next refresh re-creates it)."""
        if self._dock is not None:
            try:
                self.viewer.window.remove_dock_widget(self._dock)
            except Exception:
                logger.exception("could not remove the lineage canvas dock")
        self._dock = None
        self._panel = None

    # -- assembly helpers ---------------------------------------------------
    def _validated_anchored_maps(
        self,
    ) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
        """Per-track validated / anchored frame sets, read once per refresh."""
        pos_dir = self._pos_dir_provider() if self._pos_dir_provider else None
        if pos_dir is None:
            return {}, {}
        try:
            validated = {
                int(cell_id): {int(f) for f in frames}
                for cell_id, frames in (read_validated_tracks(pos_dir) or {}).items()
            }
            anchored: dict[int, set[int]] = {}
            for corr in read_corrections(pos_dir):
                if getattr(corr, "kind", None) == "anchor":
                    anchored.setdefault(int(corr.cell_id), set()).add(int(corr.t))
        except Exception:
            logger.exception("could not read validated/anchored frames for the canvas")
            return {}, {}
        return validated, anchored

    def _error_map(self, tracked: np.ndarray) -> dict[int, set[int]]:
        """Per-track flagged frames from the error scan (empty if it fails)."""
        contours = self._contours_provider() if self._contours_provider else None
        try:
            errors: dict[int, set[int]] = {}
            for err in scan_errors(tracked, contours):
                errors.setdefault(int(err.cell_id), set()).add(int(err.t))
        except Exception:
            logger.exception("error scan for the canvas failed")
            return {}
        return errors

    def _build_detail(self, cell_id: int) -> TrackFilmStrip:
        tracked = self._tracked_data_provider()
        intensity_layer = self._intensity_layer_provider()
        if not cell_id or tracked is None or intensity_layer is None:
            return TrackFilmStrip(tiles=())
        try:
            return build_track_film_strip(
                np.asarray(tracked),
                np.asarray(intensity_layer.data),
                cell_id,
                colormap=self._intensity_colormap(intensity_layer),
                outline_color=self._track_outline_color(cell_id),
                frames=self._occupied.get(cell_id),
                validated_frames=self._validated_map.get(cell_id),
                anchored_frames=self._anchored_map.get(cell_id),
            )
        except Exception:
            logger.exception("detail film strip build failed")
            return TrackFilmStrip(tiles=())

    def _detail_title(self, cell_id: int) -> str:
        if not cell_id:
            return "No track selected"
        return f"Track {cell_id} — {len(self._occupied.get(cell_id, ()))} frame(s)"

    def _track_outline_color(self, cell_id: int):
        """RGB (0..1) the tracked labels layer paints ``cell_id`` with, or None."""
        layer = self._tracked_layer_provider()
        color_dict = getattr(getattr(layer, "colormap", None), "color_dict", None)
        try:
            raw = color_dict.get(int(cell_id)) if color_dict is not None else None
        except Exception:
            raw = None
        if raw is None or isinstance(raw, str):
            return _NODE_OUTLINE
        rgba = np.asarray(raw, dtype=float).ravel()
        if rgba.size < 3:
            return _NODE_OUTLINE
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]))

    @staticmethod
    def _intensity_colormap(layer):
        """Adapt the intensity layer's colormap (e.g. 'I Purple') to (h,w)->RGB."""
        cmap = getattr(layer, "colormap", None)
        if cmap is None or not hasattr(cmap, "map"):
            return None

        def _map(values: np.ndarray) -> np.ndarray:
            flat = np.asarray(values, dtype=float).ravel()
            mapped = np.asarray(cmap.map(flat), dtype=float)
            return mapped.reshape(values.shape + (mapped.shape[-1],))

        return _map

    def _ensure_panel(self) -> LineageCanvasPanel:
        if self._panel is not None:
            return self._panel
        panel = LineageCanvasPanel()
        panel.node_activated.connect(self._on_node_activated)
        self._panel = panel
        try:
            self._dock = self.viewer.window.add_dock_widget(
                panel, name="Lineage canvas", area="left"
            )
        except Exception:
            logger.exception("could not dock the lineage canvas")
            self._dock = None
        return panel

    def _on_node_activated(self, frame: int, cell_id: int) -> None:
        try:
            self._on_activate(int(frame), int(cell_id))
        except Exception:
            logger.exception("lineage canvas navigation failed")


__all__ = ["LineageCanvasController"]
