"""Trivial analysis plugin: summarise the in-scope catalog.

This plugin does no per-position analysis; it exists to prove the base ↔ plugin
seam end-to-end. It reports how many positions are in scope, how they break down
by condition, and how many are contact-analysis ``ready`` vs ``incomplete``.
"""
from __future__ import annotations

from collections import Counter

from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from itasc.contact_analysis.catalog import STATUS_READY
from itasc.napari.contact_analysis.plugins import AnalysisContext, AnalysisPlugin


class CatalogSummaryPlugin(AnalysisPlugin):
    """Show counts for the currently-selected catalog records."""

    plugin_id = "catalog_summary"
    display_name = "Catalog Summary"

    def __init__(self, viewer=None, parent: QWidget | None = None) -> None:
        super().__init__(viewer=viewer, parent=parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        self._summary_lbl = QLabel("No positions in scope.")
        self._summary_lbl.setWordWrap(True)
        layout.addWidget(self._summary_lbl)
        layout.addStretch()

    def set_context(self, ctx: AnalysisContext) -> None:
        records = list(ctx.records)
        if not records:
            self._summary_lbl.setText("No positions in scope.")
            return

        total = len(records)
        ready = sum(
            1
            for record in records
            if record.get("contact_analysis_status") == STATUS_READY
        )
        by_condition = Counter(
            str(record.get("condition", "unknown_condition")) for record in records
        )
        condition_lines = "\n".join(
            f"  • {condition}: {count}"
            for condition, count in sorted(by_condition.items())
        )
        self._summary_lbl.setText(
            f"{total} position(s) in scope\n"
            f"  ready: {ready}    incomplete: {total - ready}\n"
            f"by condition:\n{condition_lines}"
        )
