"""Docked combined-canvas state — assembles the overview + focus detail.

Owns the bottom-docked correction canvas: it builds the per-track swimlane
overview (present runs from :func:`~cellflow.segmentation.lineage.build_lineage`,
status frames from the project's validation records) and, for the *selected*
track only, the film-strip detail band (via
:func:`~cellflow.napari._correction_track_path.build_track_film_strip`). Building
crops for a single track keeps refresh cheap no matter how many tracks exist.

The overview, lane assembly and crops are pure/testable; this is the glue that
reads layers, owns the overview + film-strip widgets (embedded by the host into
its workspace splitter), and turns a lane/tile click into a viewer jump + cell
selection via ``on_activate``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Callable

import numpy as np

from cellflow.tracking_ultrack.validation_state import read_corrections, read_validated_tracks
from cellflow.napari._correction_film_strip import TrackFilmStripPanel
from cellflow.napari._correction_lineage_canvas import LaneView, LineageCanvasPanel
from cellflow.napari._correction_track_path import (
    TrackFilmStrip,
    build_track_film_strip,
)
from cellflow.core.lineage import build_lineage

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
    ) -> None:
        self.viewer = viewer
        self._tracked_data_provider = tracked_data_provider
        self._tracked_layer_provider = tracked_layer_provider
        self._intensity_layer_provider = intensity_layer_provider
        self._selected_label_provider = selected_label_provider
        self._current_t_provider = current_t_provider
        self._on_activate = on_activate
        self._pos_dir_provider = pos_dir_provider
        # Both the swimlane overview and the per-track film strip are embedded as
        # bare widgets into the host's workspace splitter; the controller no
        # longer owns napari docks.
        self._overview_panel: LineageCanvasPanel | None = None
        self._film_panel: TrackFilmStripPanel | None = None
        # Cached from the last refresh so a selection can rebuild only the detail.
        self._occupied: dict[int, list[int]] = {}
        self._validated_map: dict[int, set[int]] = {}
        self._anchored_map: dict[int, set[int]] = {}
        # Structural lanes (cell_id, column, segments) from the last full assemble,
        # so a flag-only change can recolour without re-running build_lineage.
        self._lane_structure: list[tuple[int, int, tuple[tuple[int, int], ...]]] | None = None
        self._n_frames: int = 0

    def refresh(self) -> None:
        """Rebuild the overview and the detail strip for the current selection."""
        tracked = self._tracked_data_provider()
        if tracked is None:
            if self._overview_panel is not None:
                self._overview_panel.set_overview([], n_frames=0)
            return
        try:
            lanes, n_frames = self._assemble(np.asarray(tracked))
        except Exception:
            logger.exception("lineage overview assembly failed")
            return
        self._ensure_panels()
        self._overview_panel.set_overview(
            lanes, n_frames=n_frames, title=f"{len(lanes)} track(s)",
        )
        self.set_current_frame(self._current_t_provider())
        self.set_selection(int(self._selected_label_provider() or 0))

    def _assemble(self, tracked: np.ndarray) -> tuple[list[LaneView], int]:
        model = build_lineage(tracked)
        self._validated_map, self._anchored_map = self._validated_anchored_maps()
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
            ))
        self._lane_structure = [
            (ln.cell_id, ln.column, ln.segments) for ln in lanes
        ]
        self._n_frames = model.n_frames
        return lanes, model.n_frames

    def set_selection(self, cell_id: int) -> None:
        """Highlight the selected lane and rebuild only its detail film strip."""
        cell_id = int(cell_id or 0)
        if self._overview_panel is None:
            return
        self._overview_panel.set_selection(cell_id)
        if self._film_panel is None:
            return
        self._film_panel.set_strip(
            self._build_detail(cell_id), title=self._detail_title(cell_id)
        )
        self._film_panel.set_current_frame(self._current_t_provider())

    def set_current_frame(self, frame: int) -> None:
        """Move the shared frame guide / film highlight without rebuilding."""
        frame = int(frame)
        if self._overview_panel is not None:
            self._overview_panel.set_current_frame(frame)
        if self._film_panel is not None:
            self._film_panel.set_current_frame(frame)

    def center_on_track(self, cell_id: int) -> None:
        """Scroll the overview so ``cell_id``'s row is vertically centered."""
        if self._overview_panel is not None:
            self._overview_panel.center_on_track(int(cell_id or 0))

    def refresh_status(self) -> None:
        """Recolour validated/anchored flags without rescanning the stack.

        Validation and anchoring change only per-frame *status*, never track
        topology, so the expensive whole-stack ``build_lineage`` in
        :meth:`refresh` is unnecessary. This re-reads the validation records and
        re-applies them to the lanes cached by the last full refresh, keeping
        the GUI responsive on long tracks (the per-frame ``build_lineage`` froze
        it for seconds). Falls back to a full :meth:`refresh` when no structure
        has been cached yet.
        """
        if self._overview_panel is None:
            return
        if self._lane_structure is None:
            self.refresh()
            return
        self._validated_map, self._anchored_map = self._validated_anchored_maps()
        lanes = [
            LaneView(
                cell_id=cid,
                column=column,
                segments=segments,
                validated=frozenset(self._validated_map.get(cid, ())),
                anchored=frozenset(self._anchored_map.get(cid, ())),
            )
            for cid, column, segments in self._lane_structure
        ]
        self._overview_panel.set_overview(
            lanes, n_frames=self._n_frames, title=f"{len(lanes)} track(s)",
        )
        self.set_current_frame(self._current_t_provider())
        self.set_selection(int(self._selected_label_provider() or 0))

    def refresh_detail(self) -> None:
        """Rebuild only the selected track's detail strip (no overview rescan).

        Used after a rapid live edit (stepping swap candidates with Z / C): it
        re-crops just the selected track from the cached frame set, so the strip
        reflects the new pixels *without* re-running the whole-stack lineage
        build that the full :meth:`refresh` does (that froze the GUI when fired
        on every keystroke). The overview is left as-is until the next
        full refresh (selection change, validate/anchor, reload).
        """
        if self._overview_panel is None:
            return
        self.set_selection(int(self._selected_label_provider() or 0))

    def overview_panel(self) -> LineageCanvasPanel:
        """The swimlane overview widget (created on first access) to embed."""
        self._ensure_panels()
        return self._overview_panel

    def film_widget(self) -> TrackFilmStripPanel:
        """The per-track film strip widget (created on first access) to embed."""
        self._ensure_panels()
        return self._film_panel

    def teardown(self) -> None:
        """Drop references to both panels for deactivate.

        Both panels are embedded as bare widgets in the host's workspace
        splitter and are deleted when that dock is torn down; here we just drop
        our references so a later re-activate recreates them.
        """
        self._film_panel = None
        self._overview_panel = None
        self._lane_structure = None

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

    def _ensure_panels(self) -> None:
        """Create the overview + film panels as bare widgets (idempotent)."""
        if self._overview_panel is None:
            overview = LineageCanvasPanel()
            overview.node_activated.connect(self._on_node_activated)
            self._overview_panel = overview
        if self._film_panel is None:
            film = TrackFilmStripPanel(tile_px=72)
            film.frame_clicked.connect(self._on_film_frame_clicked)
            self._film_panel = film

    def _on_node_activated(self, frame: int, cell_id: int) -> None:
        try:
            self._on_activate(int(frame), int(cell_id))
        except Exception:
            logger.exception("lineage canvas navigation failed")

    def _on_film_frame_clicked(self, frame: int) -> None:
        cell_id = int(self._selected_label_provider() or 0)
        if not cell_id:
            return
        self._on_node_activated(int(frame), cell_id)


__all__ = ["LineageCanvasController"]
