"""The Aggregate Quantification studio's Aggregate area.

A thin region between Build and Plot. Where Build runs *producers* over the
in-scope positions (writing per-position artifacts), Aggregate pools **what is
built on disk** for those positions into the small set of index-keyed shape
tables (:mod:`cellflow.aggregate_quantification.shape_tables`), written once at
``<catalogue>/aggregate_quantification/``. The Plot area then reads those tables
rather than pooling per-position live.

It offers a single **Run Aggregate** button (scope = the catalogue selection,
empty = the whole catalogue) and a small status list of the shape tables (built /
empty, row count, last-written). The studio also fires it automatically after a
Build Run finishes, so the common path never goes stale; the button is for
re-aggregating without rebuilding. Reading ``what is built`` (not the Build
checkboxes — those mean *what to compute*) keeps the two concerns from
overloading one control and risking partial tables.

The actual pooling runs off the GUI thread, delegated to the studio's
*aggregate_callback*; this widget owns only the controls and the on-disk status
read.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from qtpy.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.shape_tables import (
    catalogue_root,
    read_table,
    shape_table_registry,
    table_path,
)
from cellflow.napari.ui_style import action_button, parameter_heading, status_label

#: Signature of the studio callback Run Aggregate invokes: ``(in_scope_records)``.
AggregateCallback = Callable[[list[dict]], None]


class AggregateArea(QWidget):
    """Run Aggregate button + a per-table built/empty · rows · last-written list."""

    def __init__(
        self,
        aggregate_callback: AggregateCallback,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._aggregate_callback = aggregate_callback
        self._records: list[dict] = []
        #: table name → (status label, detail label)
        self._rows: dict[str, tuple[QLabel, QLabel]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        intro = QLabel(
            "Pool what is built for the in-scope positions into the aggregated "
            "tables the plots read. Runs automatically after a build."
        )
        intro.setWordWrap(True)
        status_label(intro, muted=True)
        layout.addWidget(intro)

        heading = QLabel("TABLES")
        parameter_heading(heading)
        layout.addWidget(heading)

        grid = QGridLayout()
        grid.setContentsMargins(12, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(1)
        grid.setColumnStretch(2, 1)
        for i, name in enumerate(sorted(shape_table_registry())):
            name_lbl = QLabel(name)
            status_lbl = QLabel("")
            detail_lbl = QLabel("")
            status_label(detail_lbl, muted=True)
            grid.addWidget(name_lbl, i, 0)
            grid.addWidget(status_lbl, i, 1)
            grid.addWidget(detail_lbl, i, 2)
            self._rows[name] = (status_lbl, detail_lbl)
        layout.addLayout(grid)

        self._run_btn = QPushButton("Run Aggregate")
        self._run_btn.setToolTip(
            "Pool every built product for the in-scope positions into the "
            "aggregated tables (overwriting them)."
        )
        action_button(self._run_btn, expand=True)
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        status_label(self._status, muted=True)
        layout.addWidget(self._status)

        self.refresh_status()

    def set_context(self, ctx: object) -> None:
        self._records = list(getattr(ctx, "records", []))
        self.refresh_status()

    def set_status(self, message: str) -> None:
        self._status.setText(message)

    def refresh_status(self) -> None:
        """Re-read each shape table's CSV from the current catalogue root and
        update its built/empty · row-count · last-written line."""
        root = catalogue_root(self._records) if self._records else None
        for name, (status_lbl, detail_lbl) in self._rows.items():
            path = table_path(root, name) if root is not None else None
            if path is not None and path.is_file():
                status_lbl.setText("built")
                detail_lbl.setText(self._detail(path))
            else:
                status_lbl.setText("—")
                detail_lbl.setText("not aggregated")
        self._run_btn.setEnabled(bool(self._records))

    @staticmethod
    def _detail(path: Path) -> str:
        try:
            rows = len(read_table(path))
        except Exception:  # noqa: BLE001 - a malformed table reads as "?" rows
            rows = None
        when = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        count = f"{rows:,} rows" if rows is not None else "? rows"
        return f"{count} · {when}"

    def _on_run(self) -> None:
        if not self._records:
            self._status.setText("Add positions to the catalogue first.")
            return
        self._status.setText("Aggregating…")
        self._run_btn.setEnabled(False)
        self._aggregate_callback(list(self._records))
