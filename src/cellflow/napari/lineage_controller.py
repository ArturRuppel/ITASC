"""Docked lineage-swimlane state for the nucleus correction workflow.

Owns the *docked panel* half of the lineage graph: building the per-track
presence model from the tracked stack, docking/undocking the panel, keeping the
selected lane and current-frame cursor in sync, and turning a lane click into a
viewer jump + cell selection via the host-supplied ``on_activate`` callback. The
model itself is the pure :func:`~cellflow.segmentation.lineage.build_lineage`;
this controller is the glue, mirroring
:class:`~cellflow.napari.film_strip_controller`.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from cellflow.napari._correction_lineage_panel import LineagePanel
from cellflow.segmentation.lineage import build_lineage

logger = logging.getLogger(__name__)


class LineageController:
    """Own the docked lineage swimlanes for the correction session."""

    def __init__(
        self,
        viewer,
        *,
        tracked_data_provider: Callable[[], np.ndarray | None],
        selected_label_provider: Callable[[], int],
        current_t_provider: Callable[[], int],
        on_activate: Callable[[int, int], None],
    ) -> None:
        self.viewer = viewer
        self._tracked_data_provider = tracked_data_provider
        self._selected_label_provider = selected_label_provider
        self._current_t_provider = current_t_provider
        self._on_activate = on_activate
        self._panel: LineagePanel | None = None
        self._dock = None

    def refresh(self) -> None:
        """Rebuild the lineage model and re-render the lanes."""
        tracked = self._tracked_data_provider()
        if tracked is None:
            if self._panel is not None:
                self._panel.set_model(None)
            return
        try:
            model = build_lineage(tracked)
        except Exception:
            logger.exception("lineage build failed")
            return
        panel = self._ensure_panel()
        panel.set_model(model)
        panel.set_selection(int(self._selected_label_provider() or 0))
        panel.set_current_frame(self._current_t_provider())

    def set_selection(self, cell_id: int) -> None:
        """Highlight ``cell_id``'s lane without rebuilding the model."""
        if self._panel is not None:
            self._panel.set_selection(int(cell_id or 0))

    def set_current_frame(self, frame: int) -> None:
        """Move the current-frame cursor without rebuilding the model."""
        if self._panel is not None:
            self._panel.set_current_frame(int(frame))

    def teardown(self) -> None:
        """Undock and forget the panel (next refresh re-creates it)."""
        if self._dock is not None:
            try:
                self.viewer.window.remove_dock_widget(self._dock)
            except Exception:
                logger.exception("could not remove the lineage dock")
        self._dock = None
        self._panel = None

    def _ensure_panel(self) -> LineagePanel:
        if self._panel is not None:
            return self._panel
        panel = LineagePanel()
        panel.track_activated.connect(self._on_track_activated)
        self._panel = panel
        try:
            self._dock = self.viewer.window.add_dock_widget(
                panel, name="Lineage", area="right"
            )
        except Exception:
            logger.exception("could not dock the lineage panel")
            self._dock = None
        return panel

    def _on_track_activated(self, frame: int, cell_id: int) -> None:
        try:
            self._on_activate(int(frame), int(cell_id))
        except Exception:
            logger.exception("lineage navigation failed")


__all__ = ["LineageController"]
