"""Docked lineage-canvas state — assembles the unified correction view.

Owns the *docked canvas* half of the unified visualization: it builds the pure
:class:`~cellflow.segmentation.lineage_graph.LineageGraph` (per-track nodes +
edges), crops each node's thumbnail with the same
:func:`~cellflow.napari._correction_track_path.build_track_film_strip` the film
strip uses, lays tracks out in columns with time running downward, and hands the
ready-to-blit :class:`NodeView` / :class:`EdgeView` structs to the panel. Node
clicks turn into a viewer jump + cell selection via ``on_activate``. The graph,
cropping and layout are all pure/testable; this is the glue, mirroring
:class:`~cellflow.napari.film_strip_controller`.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from cellflow.napari._correction_lineage_canvas import (
    EdgeView,
    LineageCanvasPanel,
    NodeView,
)
from cellflow.napari._correction_track_path import build_track_film_strip
from cellflow.segmentation.lineage_graph import assign_columns, build_lineage_graph

logger = logging.getLogger(__name__)

_COL_GAP = 24  # horizontal padding between track columns (scene px)
_ROW_GAP = 16  # vertical padding between frames (scene px)
_NODE_OUTLINE = (0.75, 0.75, 0.75)  # neutral cell outline (not time/confidence)


class LineageCanvasController:
    """Own the docked, pannable lineage canvas for the correction session."""

    def __init__(
        self,
        viewer,
        *,
        tracked_data_provider: Callable[[], np.ndarray | None],
        intensity_layer_provider: Callable[[], object | None],
        selected_label_provider: Callable[[], int],
        current_t_provider: Callable[[], int],
        on_activate: Callable[[int, int], None],
    ) -> None:
        self.viewer = viewer
        self._tracked_data_provider = tracked_data_provider
        self._intensity_layer_provider = intensity_layer_provider
        self._selected_label_provider = selected_label_provider
        self._current_t_provider = current_t_provider
        self._on_activate = on_activate
        self._panel: LineageCanvasPanel | None = None
        self._dock = None

    def refresh(self) -> None:
        """Rebuild the graph, crop node thumbnails, and re-render the canvas."""
        tracked = self._tracked_data_provider()
        intensity_layer = self._intensity_layer_provider()
        if tracked is None or intensity_layer is None:
            if self._panel is not None:
                self._panel.set_scene([], [], row_height=1.0, scene_width=0.0)
            return
        try:
            nodes, edges, row_h, scene_w = self._assemble(
                tracked, np.asarray(intensity_layer.data),
            )
        except Exception:
            logger.exception("lineage canvas assembly failed")
            return
        panel = self._ensure_panel()
        panel.set_scene(
            nodes, edges, row_height=row_h, scene_width=scene_w,
            title=f"{len({n.cell_id for n in nodes})} track(s), {len(nodes)} node(s)",
        )
        panel.set_selection(int(self._selected_label_provider() or 0))
        panel.set_current_frame(self._current_t_provider())

    def _assemble(
        self, tracked: np.ndarray, intensity: np.ndarray,
    ) -> tuple[list[NodeView], list[EdgeView], float, float]:
        graph = build_lineage_graph(tracked)
        columns = assign_columns(graph)
        # Crop every track's per-frame thumbnails once (the film-strip tiles).
        rgb_of: dict[tuple[int, int], np.ndarray] = {}
        max_w = max_h = 1
        for cell_id in columns:
            strip = build_track_film_strip(
                tracked, intensity, cell_id, outline_color=_NODE_OUTLINE,
            )
            for tile in strip.tiles:
                rgb_of[(cell_id, tile.frame)] = tile.rgb
                max_w = max(max_w, tile.width)
                max_h = max(max_h, tile.height)
        col_spacing = max_w + _COL_GAP
        row_height = max_h + _ROW_GAP

        def _center(cell_id: int, t: int) -> tuple[float, float]:
            return (
                columns[cell_id] * col_spacing + col_spacing / 2.0,
                t * row_height + row_height / 2.0,
            )

        nodes: list[NodeView] = []
        for node in graph.nodes:
            rgb = rgb_of.get((node.cell_id, node.t))
            if rgb is None:
                continue
            x, y = _center(node.cell_id, node.t)
            nodes.append(NodeView(cell_id=node.cell_id, t=node.t, x=x, y=y, rgb=rgb))

        edges: list[EdgeView] = []
        for edge in graph.edges:
            x0, y0 = _center(edge.cell_id, edge.t0)
            x1, y1 = _center(edge.cell_id, edge.t1)
            edges.append(EdgeView(cell_id=edge.cell_id, x0=x0, y0=y0, x1=x1, y1=y1))

        scene_width = len(columns) * col_spacing
        return nodes, edges, row_height, scene_width

    def set_selection(self, cell_id: int) -> None:
        """Highlight ``cell_id``'s nodes without rebuilding the canvas."""
        if self._panel is not None:
            self._panel.set_selection(int(cell_id or 0))

    def set_current_frame(self, frame: int) -> None:
        """Move the current-frame cursor without rebuilding the canvas."""
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
