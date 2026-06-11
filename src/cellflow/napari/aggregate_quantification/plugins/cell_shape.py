"""Cell Shape group plugin — unifies *compute* and *plot* for one logical group.

The first plugin to surface a quantity's build action and its plots together (the
"one plugin per logical group" shape). Two sections:

* **Compute** — a Build button that computes ``cell_shape.h5`` for the in-scope
  positions, delegating to the studio's centralized (threaded) build path via
  :meth:`set_build_callback`. Because this plugin *owns* the ``cell_shape``
  quantity (``owns_quantities``), the studio suppresses the generic auto-builder
  for it, so the quantity is offered exactly once.
* **Plot** — pools the in-scope positions' shape tables (optionally joined to the
  contacts subpopulation ``class_label``) and renders histograms / box / violin /
  grouped-bar / line-over-frame via the headless
  :mod:`cellflow.aggregate_quantification.plotting` backend, embedded in a
  matplotlib canvas. Exports the pooled / per-position / aggregated table to CSV
  and the figure to PNG/SVG.

All compute and plotting logic lives in the backend layers; this module is the
thin Qt shell that wires controls → :class:`PlotSpec` → backend → canvas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.cell_shape import DESCRIPTOR_COLUMNS
from cellflow.aggregate_quantification.plotting import (
    PlotSpec,
    PositionSource,
    aggregate,
    build_figure,
    pool_object_tables,
    write_csv,
)
from cellflow.aggregate_quantification.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.aggregate_quantification.quantifiers.contacts import ContactsQuantifier
from cellflow.napari.aggregate_quantification.plugins import AnalysisContext, AnalysisPlugin
from cellflow.napari.ui_style import action_button, status_label
from cellflow.napari.widgets import CollapsibleSection

# matplotlib's Qt canvas needs a running QApplication; guard the import so a
# headless / display-less environment degrades to a hint instead of breaking
# plugin discovery.
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    _HAS_MPL_QT = True
except Exception:  # pragma: no cover - exercised only without a Qt matplotlib
    FigureCanvasQTAgg = None
    _HAS_MPL_QT = False

#: The catalogue-metadata axes a plot can group/facet by, plus ``frame``.
_METADATA_GROUPS = ("condition", "date", "position_id")
#: Subpopulation column joined from the contacts artifact when split is on.
_CLASS_COLUMN = "class_label"


class CellShapePlugin(AnalysisPlugin):
    """Compute + plot per-cell shape descriptors for the in-scope positions."""

    plugin_id = "cell_shape"
    display_name = "Cell shape"
    requires = ("cell_labels_path",)
    owns_quantities = ("cell_shape",)

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(viewer=viewer, parent=parent)
        self._quantifier = CellShapeQuantifier()
        self._contacts = ContactsQuantifier()
        self._records: list[dict] = []
        self._build_callback = None
        #: Pooled table cache + the (scope, split) signature it was built for.
        self._pooled: pd.DataFrame | None = None
        self._pool_signature: tuple | None = None
        self._pool_worker = None
        self._canvas: QWidget | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)
        layout.addWidget(CollapsibleSection("Compute", self._build_compute(), expanded=True))
        layout.addWidget(CollapsibleSection("Plot", self._build_plot(), expanded=True))
        self._update_enabled()

    # --------------------------------------------------------------- compute UI
    def _build_compute(self) -> QWidget:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        self._compute_status = QLabel("No positions in scope.")
        self._compute_status.setWordWrap(True)
        status_label(self._compute_status)
        col.addWidget(self._compute_status)

        self._overwrite_cb = QCheckBox("Recompute (overwrite existing)")
        col.addWidget(self._overwrite_cb)

        self._build_btn = QPushButton("Build cell shape")
        action_button(self._build_btn, expand=True)
        self._build_btn.clicked.connect(self._on_build)
        col.addWidget(self._build_btn)
        return body

    # ------------------------------------------------------------------ plot UI
    def _build_plot(self) -> QWidget:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        self._value_combo = self._combo(DESCRIPTOR_COLUMNS, default="area")
        col.addLayout(self._labelled("Value:", self._value_combo))

        self._level_combo = QComboBox()
        self._level_combo.addItem("Per cell (pooled)", "cell")
        self._level_combo.addItem("Per position", "position")
        col.addLayout(self._labelled("Level:", self._level_combo))

        self._plot_combo = self._combo(("hist", "box", "violin", "bar", "line"))
        col.addLayout(self._labelled("Plot:", self._plot_combo))

        self._stat_combo = self._combo(("mean", "median", "count"))
        self._error_combo = self._combo(("sd", "sem", "none"))
        stat_row = QHBoxLayout()
        stat_row.setContentsMargins(0, 0, 0, 0)
        stat_row.addWidget(QLabel("Stat:"))
        stat_row.addWidget(self._stat_combo, 1)
        stat_row.addWidget(QLabel("Error:"))
        stat_row.addWidget(self._error_combo, 1)
        col.addLayout(stat_row)

        # Group / facet axes + the subpopulation split.
        self._split_cb = QCheckBox("Split by subpopulation (contacts class)")
        self._split_cb.toggled.connect(self._on_split_toggled)
        col.addWidget(self._split_cb)

        self._group_checks: dict[str, QCheckBox] = {}
        group_row = QHBoxLayout()
        group_row.setContentsMargins(0, 0, 0, 0)
        group_row.addWidget(QLabel("Group by:"))
        for name in (*_METADATA_GROUPS, _CLASS_COLUMN, "frame"):
            check = QCheckBox(name)
            self._group_checks[name] = check
            group_row.addWidget(check)
        group_row.addStretch(1)
        col.addLayout(group_row)

        self._plot_btn = QPushButton("Update plot")
        action_button(self._plot_btn, expand=True)
        self._plot_btn.clicked.connect(self._on_plot)
        col.addWidget(self._plot_btn)

        # Re-render (cheap, from the pooled cache) when a plot control changes.
        for combo in (self._value_combo, self._level_combo, self._plot_combo,
                      self._stat_combo, self._error_combo):
            combo.currentIndexChanged.connect(self._on_control_changed)
        for check in self._group_checks.values():
            check.toggled.connect(self._on_control_changed)

        self._canvas_holder = QVBoxLayout()
        col.addLayout(self._canvas_holder, 1)
        if not _HAS_MPL_QT:  # pragma: no cover - only without a Qt matplotlib
            hint = QLabel("Plotting unavailable (matplotlib Qt backend not usable).")
            status_label(hint, muted=True)
            self._canvas_holder.addWidget(hint)

        export_row = QHBoxLayout()
        export_row.setContentsMargins(0, 0, 0, 0)
        self._export_pooled_btn = self._export_button("Pooled CSV", self._export_pooled)
        self._export_position_btn = self._export_button("Per-position CSV", self._export_position)
        self._export_agg_btn = self._export_button("Aggregated CSV", self._export_aggregated)
        self._export_fig_btn = self._export_button("Figure", self._export_figure)
        for btn in (self._export_pooled_btn, self._export_position_btn,
                    self._export_agg_btn, self._export_fig_btn):
            export_row.addWidget(btn)
        col.addLayout(export_row)

        self._plot_status = QLabel("")
        self._plot_status.setWordWrap(True)
        status_label(self._plot_status, muted=True)
        col.addWidget(self._plot_status)
        return body

    # --------------------------------------------------------------- UI helpers
    @staticmethod
    def _combo(items, default: str | None = None) -> QComboBox:
        combo = QComboBox()
        for item in items:
            combo.addItem(item, item)
        if default is not None and default in items:
            combo.setCurrentText(default)
        return combo

    @staticmethod
    def _labelled(label: str, widget: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setFixedWidth(70)
        row.addWidget(lbl)
        row.addWidget(widget, 1)
        return row

    def _export_button(self, text: str, slot) -> QPushButton:
        btn = QPushButton(text)
        action_button(btn)
        btn.clicked.connect(slot)
        return btn

    # ------------------------------------------------------- studio integration
    def set_build_callback(self, callback) -> None:
        """Receive the studio's ``(quantifier, records, overwrite)`` build hook."""
        self._build_callback = callback
        self._update_enabled()

    def set_context(self, ctx: AnalysisContext) -> None:
        if ctx.viewer is not None:
            self.viewer = ctx.viewer
        self._records = list(ctx.records)
        # Scope changed → the pooled cache no longer matches; require a re-pool.
        self._invalidate_pool()
        self._update_enabled()

    def _update_enabled(self) -> None:
        records = self._records
        buildable = [r for r in records if r.get("cell_tracked_labels_path")]
        built = [r for r in records if self._is_built(self._quantifier, r)]
        if not records:
            self._compute_status.setText("No positions in scope.")
        else:
            self._compute_status.setText(
                f"{len(buildable)} of {len(records)} in-scope position(s) have cell "
                f"labels; {len(built)} already built."
            )
        self._build_btn.setEnabled(bool(buildable) and self._build_callback is not None)
        self._plot_btn.setEnabled(bool(built) and _HAS_MPL_QT and self._pool_worker is None)
        self._group_checks[_CLASS_COLUMN].setEnabled(self._split_cb.isChecked())
        has_plot = self._pooled is not None and not self._pooled.empty
        self._export_pooled_btn.setEnabled(has_plot)
        self._export_agg_btn.setEnabled(has_plot)
        self._export_fig_btn.setEnabled(has_plot)
        self._export_position_btn.setEnabled(len(built) == 1)

    # ----------------------------------------------------------------- building
    def _on_build(self) -> None:
        if self._build_callback is None:
            return
        self._build_callback(self._quantifier, list(self._records), self._overwrite_cb.isChecked())

    # ------------------------------------------------------------------ pooling
    def _scope_signature(self) -> tuple:
        return (
            tuple(str(r.get("contact_analysis_path", r.get("id", ""))) for r in self._records),
            self._split_cb.isChecked(),
        )

    def _invalidate_pool(self) -> None:
        self._pooled = None
        self._pool_signature = None

    def _on_split_toggled(self) -> None:
        self._invalidate_pool()  # the join changes which rows/columns are pooled
        self._update_enabled()

    def _on_control_changed(self) -> None:
        # Re-render from the cached pool; control changes never re-read artifacts.
        if self._pooled is not None:
            self._render()

    def _on_plot(self) -> None:
        if self._pooled is not None and self._pool_signature == self._scope_signature():
            self._render()
            return
        self._pool_async()

    def _pool_async(self) -> None:
        records = list(self._records)
        split = self._split_cb.isChecked()
        signature = self._scope_signature()
        self._plot_status.setText("Reading shape tables…")
        self._pool_worker = object()
        self._update_enabled()

        @thread_worker(
            connect={"returned": self._on_pool_done, "errored": self._on_pool_error}
        )
        def _worker():
            return signature, _pool_records(self._quantifier, self._contacts, records, split)

        self._pool_worker = _worker()

    def _on_pool_error(self, exc: Exception) -> None:
        self._pool_worker = None
        self._plot_status.setText(f"Plot error: {exc}")
        self._update_enabled()

    def _on_pool_done(self, result: tuple) -> None:
        self._pool_worker = None
        signature, pooled = result
        self._pooled = pooled
        self._pool_signature = signature
        if pooled.empty:
            self._plot_status.setText("No built cell-shape tables in scope.")
        else:
            self._plot_status.setText(f"Pooled {len(pooled)} cell-rows.")
            self._render()
        self._update_enabled()

    # ----------------------------------------------------------------- spec/render
    def _current_spec(self) -> PlotSpec:
        group_by = tuple(
            name for name, check in self._group_checks.items()
            if check.isChecked() and (name != _CLASS_COLUMN or self._split_cb.isChecked())
        )
        return PlotSpec(
            value=self._value_combo.currentData(),
            group_by=group_by,
            level=self._level_combo.currentData(),
            plot=self._plot_combo.currentData(),
            stat=self._stat_combo.currentData(),
            error=self._error_combo.currentData(),
        )

    def _render(self) -> None:
        if self._pooled is None or not _HAS_MPL_QT:
            return
        fig = build_figure(self._pooled, self._current_spec())
        canvas = FigureCanvasQTAgg(fig)
        if self._canvas is not None:
            self._canvas_holder.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas.deleteLater()
        self._canvas_holder.addWidget(canvas, 1)
        self._canvas = canvas

    # ------------------------------------------------------------------ exports
    def _save_path(self, caption: str, filt: str) -> Path | None:
        path, _ = QFileDialog.getSaveFileName(self, caption, filter=filt)
        return Path(path) if path else None

    def _export_pooled(self) -> None:
        if self._pooled is None:
            return
        path = self._save_path("Export pooled cell table", "CSV files (*.csv)")
        if path:
            self._plot_status.setText(f"Wrote {write_csv(self._pooled, path).name}.")

    def _export_aggregated(self) -> None:
        if self._pooled is None:
            return
        path = self._save_path("Export aggregated table", "CSV files (*.csv)")
        if path:
            summary = aggregate(self._pooled, self._current_spec())
            self._plot_status.setText(f"Wrote {write_csv(summary, path).name}.")

    def _export_position(self) -> None:
        built = [r for r in self._records if self._is_built(self._quantifier, r)]
        if len(built) != 1:
            return
        path = self._save_path("Export this position's table", "CSV files (*.csv)")
        if not path:
            return
        table = self._quantifier.object_table(self._output(self._quantifier, built[0]))
        self._plot_status.setText(f"Wrote {write_csv(pd.DataFrame(dict(table)), path).name}.")

    def _export_figure(self) -> None:
        if self._canvas is None:
            return
        path = self._save_path("Export figure", "Images (*.png *.svg)")
        if path:
            self._canvas.figure.savefig(path)
            self._plot_status.setText(f"Wrote {path.name}.")

    # ----------------------------------------------------------- path resolution
    def _output(self, quantifier, record: dict) -> Path:
        # Lazy import: studio_plugins lives in the Qt layer above this package; a
        # module-level import would risk a discovery-time cycle.
        from cellflow.napari.studio_plugins import output_for_record

        return output_for_record(quantifier, record)

    def _is_built(self, quantifier, record: dict) -> bool:
        return quantifier.is_built(self._output(quantifier, record))


def _pool_records(quantifier, contacts, records, split: bool) -> pd.DataFrame:
    """Read each built position's shape table (optionally joined to the contacts
    subpopulation class) and pool them. Runs off the GUI thread."""
    from cellflow.napari.studio_plugins import output_for_record

    sources: list[PositionSource] = []
    for record in records:
        path = output_for_record(quantifier, record)
        if not quantifier.is_built(path):
            continue
        join_table = None
        join_columns: tuple[str, ...] = ()
        if split:
            contacts_path = output_for_record(contacts, record)
            if contacts.is_built(contacts_path):
                join_table = contacts.object_table(contacts_path)
            join_columns = (_CLASS_COLUMN,)
        sources.append(
            PositionSource(
                metadata=_metadata(record),
                table=quantifier.object_table(path),
                join_table=join_table,
                join_columns=join_columns,
            )
        )
    return pool_object_tables(sources)


def _metadata(record: dict) -> dict[str, Any]:
    return {
        "condition": str(record.get("condition", "")),
        "date": str(record.get("date", "")),
        "position_id": str(record.get("id", "")),
    }
