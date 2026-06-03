"""Docked error-worklist state for the nucleus correction workflow.

Owns the *docked panel* half of the error worklist: scanning the tracked stack
(crossed with the divergence ``contours`` map) for likely errors, docking and
undocking the panel, and turning a row activation into a viewer jump + cell
selection via the host-supplied ``on_activate`` callback. The scoring itself is
the pure :func:`~cellflow.segmentation.error_scan.scan_errors`; this controller
is the glue, mirroring :class:`~cellflow.napari.film_strip_controller`.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from cellflow.napari._correction_worklist import ErrorWorklistPanel
from cellflow.segmentation.error_scan import scan_errors

logger = logging.getLogger(__name__)


class WorklistController:
    """Own the docked, ranked error worklist for the correction session."""

    def __init__(
        self,
        viewer,
        *,
        tracked_data_provider: Callable[[], np.ndarray | None],
        contours_provider: Callable[[], np.ndarray | None],
        on_activate: Callable[[int, int], None],
    ) -> None:
        self.viewer = viewer
        self._tracked_data_provider = tracked_data_provider
        self._contours_provider = contours_provider
        self._on_activate = on_activate
        self._panel: ErrorWorklistPanel | None = None
        self._dock = None

    def refresh(self) -> None:
        """Re-scan the tracked stack and repopulate the worklist."""
        tracked = self._tracked_data_provider()
        if tracked is None:
            if self._panel is not None:
                self._panel.clear()
            return
        try:
            contours = self._contours_provider()
        except Exception:
            logger.exception("could not load contours for the worklist")
            contours = None
        try:
            entries = scan_errors(tracked, contours)
        except Exception:
            logger.exception("error scan failed")
            entries = []
        self._ensure_panel().set_entries(entries)

    def mark_resolved(self, frame: int, cell_id: int) -> None:
        """Strike out a row once its cell has been edited (best-effort)."""
        if self._panel is not None:
            self._panel.mark_resolved(frame, cell_id)

    def teardown(self) -> None:
        """Undock and forget the panel (next refresh re-creates it)."""
        if self._dock is not None:
            try:
                self.viewer.window.remove_dock_widget(self._dock)
            except Exception:
                logger.exception("could not remove the worklist dock")
        self._dock = None
        self._panel = None

    def _ensure_panel(self) -> ErrorWorklistPanel:
        if self._panel is not None:
            return self._panel
        panel = ErrorWorklistPanel()
        panel.entry_activated.connect(self._on_entry_activated)
        self._panel = panel
        try:
            self._dock = self.viewer.window.add_dock_widget(
                panel, name="Correction worklist", area="right"
            )
        except Exception:
            logger.exception("could not dock the worklist")
            self._dock = None
        return panel

    def _on_entry_activated(self, frame: int, cell_id: int) -> None:
        try:
            self._on_activate(int(frame), int(cell_id))
        except Exception:
            logger.exception("worklist navigation failed")


__all__ = ["WorklistController"]
