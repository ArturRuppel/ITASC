"""Per-position contact visualizer, as an analysis plugin.

This was a hardwired "Visualize position" section inside the studio's Catalogue
region. It is now a regular :class:`AnalysisPlugin` so it lives in the same list
as every other analysis (one plugin = one collapsible), instead of being a
special case the studio assembles by hand.

It wraps the embedded :class:`AggregateQuantificationWidget` and drives it from
the catalogue scope: a single in-scope position is visualized; zero or several
are ambiguous for one viewer, so the view is cleared.
"""
from __future__ import annotations

from qtpy.QtWidgets import QVBoxLayout, QWidget

from cellflow.napari.aggregate_quantification.plugins import AnalysisContext, AnalysisPlugin


class VisualizeContactsPlugin(AnalysisPlugin):
    """Visualize (and compute-if-missing) one selected position's contacts."""

    plugin_id = "visualize_contacts"
    display_name = "Visualize Contacts"
    # Deliberately no ``requires``: a row may carry only a loose ``.h5`` (show an
    # existing result, no labels to gate on) or carry labels (compute on demand).
    # Gating on ``cell_labels_path`` would wrongly hide it for visualize-only
    # catalogues; the plugin self-handles missing inputs instead.

    def __init__(self, viewer=None, parent: QWidget | None = None) -> None:
        super().__init__(viewer=viewer, parent=parent)
        # Imported lazily: keep the plugin module import cheap and avoid an
        # import cycle with the widget/studio modules.
        from cellflow.napari.aggregate_quantification_widget import (
            AggregateQuantificationWidget,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._view = AggregateQuantificationWidget(viewer=viewer, standalone=False)
        # The embedded widget's own "Pipeline Files" panel is an orchestrator
        # concept; here the catalogue table is the position source instead.
        self._view.pipeline_files_header.setVisible(False)
        self._view._pipeline_files_section.setVisible(False)
        layout.addWidget(self._view)

    def set_context(self, ctx: AnalysisContext) -> None:
        """Point the embedded view at the single in-scope position, else clear.

        Catalogue rows from a study scan carry the cell/nucleus label paths; rows
        from a loose ``.h5`` add or a reloaded CSV carry only the ``.h5``, so
        those support showing an existing result but not compute-on-demand.
        """
        records = list(ctx.records)
        if len(records) != 1:
            # Zero or several positions is ambiguous for a single viewer.
            self._view.set_context(cell_labels=None, nucleus_labels=None, out_path=None)
            return
        record = records[0]
        self._view.set_context(
            cell_labels=record.get("cell_tracked_labels_path"),
            nucleus_labels=record.get("nucleus_tracked_labels_path"),
            out_path=record.get("contact_analysis_path"),
            status_root=record.get("position_path"),
        )
