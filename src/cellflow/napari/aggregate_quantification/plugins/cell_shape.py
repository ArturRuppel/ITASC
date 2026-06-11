"""Cell Shape group plugin — unifies *compute* and *launch-plot* for one group.

The first plugin to surface a quantity's build action together with its plots
(the "one plugin per logical group" shape). Two sections:

* **Compute** — a Build button that computes ``cell_shape.h5`` for the in-scope
  positions, delegating to the studio's centralized (threaded) build path via
  :meth:`set_build_callback`. Because this plugin *owns* the ``cell_shape``
  quantity (``owns_quantities``), the studio suppresses the generic auto-builder
  for it, so the quantity is offered exactly once.
* **Plot** — a single **"Plot…"** button. A click snapshots the in-scope scope,
  pools the shape tables off-thread (always left-joining the contacts
  ``class_label`` when a position has it), and opens a detached, floatable
  :class:`~cellflow.napari.aggregate_quantification.plot_panel.PlotPanel` bound to
  that snapshot via ``add_dock_widget``. Each click → an independent dock; two can
  be floated side-by-side to compare. The panel owns every plotting/styling
  control and never listens back to the studio.

All compute and plotting logic lives in the backend / panel layers; this module
is the thin Qt shell that pools a snapshot and launches a panel.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.cell_shape import DESCRIPTOR_COLUMNS
from cellflow.aggregate_quantification.plotting import PositionSource, pool_object_tables
from cellflow.aggregate_quantification.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.aggregate_quantification.quantifiers.contacts import ContactsQuantifier
from cellflow.napari.aggregate_quantification.plugins import AnalysisContext, AnalysisPlugin
from cellflow.napari.ui_style import action_button, status_label
from cellflow.napari.widgets import CollapsibleSection

# matplotlib's Qt canvas needs a running QApplication; probe it so a headless /
# display-less environment degrades to a disabled button instead of breaking
# plugin discovery. The PlotPanel itself is imported lazily at click time.
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: F401

    _HAS_MPL_QT = True
except Exception:  # pragma: no cover - exercised only without a Qt matplotlib
    _HAS_MPL_QT = False

#: The catalogue-metadata axes a plot can group/facet by.
_METADATA_GROUPS = ("condition", "date", "position_id")
#: Subpopulation column always joined from the contacts artifact at pool time;
#: becomes just another group-by column in the panel.
_CLASS_COLUMN = "class_label"
#: Column roles handed to the (quantity-agnostic) PlotPanel.
_GROUP_COLUMNS = (*_METADATA_GROUPS, _CLASS_COLUMN, "frame")


class CellShapePlugin(AnalysisPlugin):
    """Compute per-cell shape descriptors and launch detached plot panels."""

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
        self._pool_worker = None
        #: Increments per launched dock so each gets a distinct name.
        self._plot_count = 0

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

        self._plot_btn = QPushButton("Plot…")
        action_button(self._plot_btn, expand=True)
        self._plot_btn.clicked.connect(self._on_plot)
        col.addWidget(self._plot_btn)

        self._plot_status = QLabel("")
        self._plot_status.setWordWrap(True)
        status_label(self._plot_status, muted=True)
        if not _HAS_MPL_QT:  # pragma: no cover - only without a Qt matplotlib
            self._plot_status.setText("Plotting unavailable (matplotlib Qt backend not usable).")
        col.addWidget(self._plot_status)
        return body

    # ------------------------------------------------------- studio integration
    def set_build_callback(self, callback) -> None:
        """Receive the studio's ``(quantifier, records, overwrite)`` build hook."""
        self._build_callback = callback
        self._update_enabled()

    def set_context(self, ctx: AnalysisContext) -> None:
        if ctx.viewer is not None:
            self.viewer = ctx.viewer
        self._records = list(ctx.records)
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
        self._plot_btn.setEnabled(
            bool(built) and _HAS_MPL_QT and self.viewer is not None and self._pool_worker is None
        )

    # ----------------------------------------------------------------- building
    def _on_build(self) -> None:
        if self._build_callback is None:
            return
        self._build_callback(self._quantifier, list(self._records), self._overwrite_cb.isChecked())

    # ------------------------------------------------------- pooling + launching
    def _on_plot(self) -> None:
        """Snapshot the current scope, pool off-thread, then open a dock."""
        records = list(self._records)
        self._plot_status.setText("Reading shape tables…")
        self._pool_worker = object()
        self._update_enabled()

        @thread_worker(connect={"returned": self._on_pool_done, "errored": self._on_pool_error})
        def _worker():
            return _pool_records(self._quantifier, self._contacts, records)

        self._pool_worker = _worker()

    def _on_pool_error(self, exc: Exception) -> None:
        self._pool_worker = None
        self._plot_status.setText(f"Plot error: {exc}")
        self._update_enabled()

    def _on_pool_done(self, pooled: pd.DataFrame) -> None:
        self._pool_worker = None
        if pooled.empty:
            self._plot_status.setText("No built cell-shape tables in scope.")
            self._update_enabled()
            return
        self._open_panel(pooled)
        self._update_enabled()

    def _open_panel(self, pooled: pd.DataFrame) -> None:
        # Lazy import: keeps the Qt matplotlib backend off the plugin-discovery
        # path (guarded by _HAS_MPL_QT above).
        from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel

        panel = PlotPanel(pooled, value_columns=DESCRIPTOR_COLUMNS, group_columns=_GROUP_COLUMNS)
        self._plot_count += 1
        name = f"Cell shape plot {self._plot_count}"
        if self.viewer is not None:
            self.viewer.window.add_dock_widget(panel, area="right", name=name)
        self._plot_status.setText(f"Opened {name} ({len(pooled)} cell-rows).")

    # ----------------------------------------------------------- path resolution
    def _output(self, quantifier, record: dict) -> Path:
        # Lazy import: studio_plugins lives in the Qt layer above this package; a
        # module-level import would risk a discovery-time cycle.
        from cellflow.napari.studio_plugins import output_for_record

        return output_for_record(quantifier, record)

    def _is_built(self, quantifier, record: dict) -> bool:
        return quantifier.is_built(self._output(quantifier, record))


def _pool_records(quantifier, contacts, records) -> pd.DataFrame:
    """Read each built position's shape table, always left-joining the contacts
    subpopulation ``class_label`` when present, and pool them. Runs off the GUI
    thread. A position without a contacts artifact contributes ``unclassified``."""
    from cellflow.napari.studio_plugins import output_for_record

    sources: list[PositionSource] = []
    for record in records:
        path = output_for_record(quantifier, record)
        if not quantifier.is_built(path):
            continue
        join_table = None
        contacts_path = output_for_record(contacts, record)
        if contacts.is_built(contacts_path):
            join_table = contacts.object_table(contacts_path)
        sources.append(
            PositionSource(
                metadata=_metadata(record),
                table=quantifier.object_table(path),
                join_table=join_table,
                join_columns=(_CLASS_COLUMN,),
            )
        )
    return pool_object_tables(sources)


def _metadata(record: dict) -> dict[str, Any]:
    return {
        "condition": str(record.get("condition", "")),
        "date": str(record.get("date", "")),
        "position_id": str(record.get("id", "")),
    }
