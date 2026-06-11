"""Shape group plugin — scope-aware compute + launch-plot for one group.

The "one plugin per logical group" shape, generalized from cell-only to a
**cell / nucleus / both** scope. A scope dropdown selects which registered
quantifier the Compute and Plot sections drive:

* ``cell`` / ``nucleus`` build & plot the per-object shape table for that label
  source (:data:`DESCRIPTOR_COLUMNS`).
* ``both`` builds & plots the relational nucleus-vs-cell table
  (:data:`RELATIONAL_COLUMNS`) — strictly the relational quantities, not the raw
  per-source descriptors.

Two sections:

* **Compute** — a Build button that computes the scope-selected quantity for the
  in-scope positions, delegating to the studio's centralized (threaded) build
  path via :meth:`set_build_callback`. Because this plugin *owns* all three shape
  quantities (``owns_quantities``), the studio suppresses their generic
  auto-builders, so each is offered exactly once.
* **Plot** — a single **"Plot…"** button. A click snapshots the in-scope scope,
  pools the scope-selected tables off-thread (left-joining the NLS sidecar CSV's
  ``class_label`` by ``cell_id`` when a position has one), and opens a detached, dockable
  :class:`~cellflow.napari.aggregate_quantification.plot_panel.PlotPanel` bound to
  that snapshot via ``add_dock_widget(area="right")``.

All compute and plotting logic lives in the backend / panel layers; this module
is the thin Qt shell that pools a snapshot and launches a panel.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.pixel_size import resolve_pixel_size_um
from cellflow.aggregate_quantification.plotting import PositionSource, pool_object_tables
from cellflow.aggregate_quantification.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.aggregate_quantification.quantifiers.nucleus_shape import NucleusShapeQuantifier
from cellflow.aggregate_quantification.quantifiers.shape_relational import (
    ShapeRelationalQuantifier,
)
from cellflow.aggregate_quantification.shape import DESCRIPTOR_COLUMNS, RELATIONAL_COLUMNS
from cellflow.napari.aggregate_quantification.plugins import AnalysisContext, AnalysisPlugin
from cellflow.napari.aggregate_quantification.plugins._plot_dock import PlotDockTabs
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
#: Subpopulation column joined from the NLS sidecar CSV (by ``cell_id``) at pool
#: time; becomes just another group-by column in the panel.
_CLASS_COLUMN = "class_label"
#: Column roles handed to the (quantity-agnostic) PlotPanel.
_GROUP_COLUMNS = (*_METADATA_GROUPS, _CLASS_COLUMN, "frame")

#: scope -> (quantifier-id, record label-path field(s), plot value columns).
_CELL_FIELD = "cell_tracked_labels_path"
_NUCLEUS_FIELD = "nucleus_tracked_labels_path"


class ShapePlugin(AnalysisPlugin):
    """Compute per-object / relational shape and launch detached plot panels."""

    plugin_id = "shape"
    display_name = "Shape"
    # cell OR nucleus suffices, which a static AND-gated ``requires`` cannot
    # express — so the plugin is always offered and gates per-scope internally.
    requires = ()
    owns_quantities = ("cell_shape", "nucleus_shape", "shape_relational")

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(viewer=viewer, parent=parent)
        self._quantifiers = {
            "cell": CellShapeQuantifier(),
            "nucleus": NucleusShapeQuantifier(),
            "both": ShapeRelationalQuantifier(),
        }
        self._records: list[dict] = []
        self._build_callback = None
        self._pool_worker = None
        #: True once the user picks a scope by hand; until then the scope tracks
        #: the data-derived default as the in-scope catalogue changes.
        self._scope_user_set = False
        #: Increments per plot so each tab gets a distinct title.
        self._plot_count = 0
        #: All plots share one dock as tabs (constant size) — see ``_plot_dock.py``.
        self._plot_tabs = PlotDockTabs(self, dock_name="Shape plots")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)
        layout.addWidget(self._build_scope_row())
        layout.addWidget(CollapsibleSection("Compute", self._build_compute(), expanded=True))
        layout.addWidget(CollapsibleSection("Plot", self._build_plot(), expanded=True))
        self._update_enabled()

    # ------------------------------------------------------------------ scope UI
    def _build_scope_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(QLabel("Scope:"))
        self._scope_combo = QComboBox()
        self._scope_combo.addItem("Cell", "cell")
        self._scope_combo.addItem("Nucleus", "nucleus")
        self._scope_combo.addItem("Both (nucleus vs cell)", "both")
        self._scope_combo.setToolTip(
            "Cell / nucleus measure per-object shape on that label source; "
            "Both pairs each nucleus with its cell and measures relational "
            "quantities (area ratio, centroid offset, …)."
        )
        self._scope_combo.currentIndexChanged.connect(lambda _=None: self._on_scope_changed())
        self._scope_combo.activated.connect(lambda _=None: self._mark_scope_user_set())
        layout.addWidget(self._scope_combo, 1)
        return row

    @property
    def _scope(self) -> str:
        return self._scope_combo.currentData() or "cell"

    @property
    def _quantifier(self):
        return self._quantifiers[self._scope]

    def _mark_scope_user_set(self) -> None:
        self._scope_user_set = True

    def _on_scope_changed(self) -> None:
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

        px_row = QHBoxLayout()
        px_row.setContentsMargins(0, 0, 0, 0)
        px_label = QLabel("Pixel size (µm/px):")
        px_row.addWidget(px_label)
        self._pixel_size_edit = QLineEdit()
        self._pixel_size_edit.setPlaceholderText("auto")
        self._pixel_size_edit.setToolTip(
            "µm per pixel. Leave blank to auto-resolve per position from its "
            "cellflow_config.json or the label TIFF; a value here applies to all "
            "in-scope positions."
        )
        self._pixel_size_edit.textChanged.connect(lambda _=None: self._update_enabled())
        px_row.addWidget(self._pixel_size_edit, 1)
        col.addLayout(px_row)

        self._overwrite_cb = QCheckBox("Recompute (overwrite existing)")
        col.addWidget(self._overwrite_cb)

        self._build_btn = QPushButton("Build shape")
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
        if not self._scope_user_set:
            self._apply_default_scope()
        self._update_enabled()

    # --------------------------------------------------------------- scope rules
    def _apply_default_scope(self) -> None:
        """Default to ``both`` when ≥1 in-scope position has both labels; else the
        single source present across the scope (cell preferred)."""
        scope = self._default_scope(self._records)
        index = self._scope_combo.findData(scope)
        if index >= 0:
            blocked = self._scope_combo.blockSignals(True)
            self._scope_combo.setCurrentIndex(index)
            self._scope_combo.blockSignals(blocked)

    @staticmethod
    def _default_scope(records: list[dict]) -> str:
        has_cell = any(r.get(_CELL_FIELD) for r in records)
        has_nucleus = any(r.get(_NUCLEUS_FIELD) for r in records)
        has_both = any(r.get(_CELL_FIELD) and r.get(_NUCLEUS_FIELD) for r in records)
        if has_both:
            return "both"
        if has_cell:
            return "cell"
        if has_nucleus:
            return "nucleus"
        return "cell"

    def _has_inputs(self, record: dict, scope: str) -> bool:
        if scope == "cell":
            return bool(record.get(_CELL_FIELD))
        if scope == "nucleus":
            return bool(record.get(_NUCLEUS_FIELD))
        return bool(record.get(_CELL_FIELD) and record.get(_NUCLEUS_FIELD))

    def _label_path_for(self, record: dict, scope: str) -> Any:
        """The label TIFF used to auto-resolve pixel size for *scope* (cell
        preferred; nucleus when cell-less)."""
        if scope == "nucleus":
            return record.get(_NUCLEUS_FIELD)
        return record.get(_CELL_FIELD) or record.get(_NUCLEUS_FIELD)

    def _update_enabled(self) -> None:
        records = self._records
        scope = self._scope
        noun = {"cell": "cell", "nucleus": "nucleus", "both": "cell+nucleus"}[scope]
        has_inputs = [r for r in records if self._has_inputs(r, scope)]
        buildable = [r for r in has_inputs if self._pixel_size_for(r) is not None]
        missing_px = len(has_inputs) - len(buildable)
        built = [r for r in records if self._is_built(self._quantifier, r)]
        if not records:
            self._compute_status.setText("No positions in scope.")
        else:
            status = (
                f"{len(has_inputs)} of {len(records)} in-scope position(s) have "
                f"{noun} labels; {len(built)} already built."
            )
            if missing_px:
                status += (
                    f" {missing_px} need a pixel size — enter one above to enable them."
                )
            self._compute_status.setText(status)
        self._build_btn.setEnabled(bool(buildable) and self._build_callback is not None)
        self._plot_btn.setEnabled(
            bool(built) and _HAS_MPL_QT and self.viewer is not None and self._pool_worker is None
        )

    # ----------------------------------------------------------------- building
    def _manual_pixel_size(self) -> float | None:
        """The typed override (µm/px), or ``None`` when blank/invalid."""
        text = self._pixel_size_edit.text().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if value > 0 else None

    def _pixel_size_for(self, record: dict) -> float | None:
        """Effective µm/px for *record*: the manual override else auto-resolved."""
        manual = self._manual_pixel_size()
        if manual is not None:
            return manual
        return resolve_pixel_size_um(
            record.get("position_path"), self._label_path_for(record, self._scope)
        )

    def _stamped(self, record: dict) -> dict:
        """A record copy carrying the manual pixel-size override when set."""
        manual = self._manual_pixel_size()
        if manual is None:
            return dict(record)
        return {**record, "pixel_size_um": manual}

    def _on_build(self) -> None:
        if self._build_callback is None:
            return
        records = [self._stamped(r) for r in self._records]
        self._build_callback(self._quantifier, records, self._overwrite_cb.isChecked())

    # ------------------------------------------------------- pooling + launching
    def _on_plot(self) -> None:
        """Snapshot the current scope, pool off-thread, then open a dock."""
        records = list(self._records)
        quantifier = self._quantifier
        self._plot_status.setText("Reading shape tables…")
        self._pool_worker = object()
        self._update_enabled()

        @thread_worker(connect={"returned": self._on_pool_done, "errored": self._on_pool_error})
        def _worker():
            return _pool_records(quantifier, records)

        self._pool_worker = _worker()

    def _on_pool_error(self, exc: Exception) -> None:
        self._pool_worker = None
        self._plot_status.setText(f"Plot error: {exc}")
        self._update_enabled()

    def _on_pool_done(self, pooled: pd.DataFrame) -> None:
        self._pool_worker = None
        if pooled.empty:
            self._plot_status.setText("No built shape tables in scope.")
            self._update_enabled()
            return
        self._open_panel(pooled)
        self._update_enabled()

    def _value_columns(self) -> tuple[str, ...]:
        """Relational columns for ``both``; per-object descriptors otherwise."""
        return RELATIONAL_COLUMNS if self._scope == "both" else DESCRIPTOR_COLUMNS

    def _open_panel(self, pooled: pd.DataFrame) -> None:
        # Lazy import: keeps the Qt matplotlib backend off the plugin-discovery
        # path (guarded by _HAS_MPL_QT above).
        from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel

        panel = PlotPanel(
            pooled, value_columns=self._value_columns(), group_columns=_GROUP_COLUMNS
        )
        self._plot_count += 1
        name = f"Plot {self._plot_count}"
        self._plot_tabs.add(panel, name)
        self._plot_status.setText(f"Opened {name} ({len(pooled)} object-rows).")

    # ----------------------------------------------------------- path resolution
    def _is_built(self, quantifier, record: dict) -> bool:
        # Lazy import: studio_plugins lives in the Qt layer above this package; a
        # module-level import would risk a discovery-time cycle.
        from cellflow.napari.studio_plugins import output_for_record

        return quantifier.is_built(output_for_record(quantifier, record))


def _pool_records(quantifier, records) -> pd.DataFrame:
    """Read each built position's shape table, left-joining the NLS sidecar CSV's
    subpopulation ``class_label`` (by ``cell_id``) when present, and pool them.
    Runs off the GUI thread. A position without an NLS CSV contributes
    ``unclassified``."""
    from cellflow.napari.studio_plugins import output_for_record

    sources: list[PositionSource] = []
    for record in records:
        path = output_for_record(quantifier, record)
        if not quantifier.is_built(path):
            continue
        sources.append(
            PositionSource(
                metadata=_metadata(record),
                table=quantifier.object_table(path),
                join_table=_nls_join_table(record),
                join_columns=(_CLASS_COLUMN,),
            )
        )
    return pool_object_tables(sources)


def _nls_join_table(record) -> dict[str, np.ndarray] | None:
    """A ``{cell_id, class_label}`` join table from the record's NLS sidecar CSV,
    or ``None`` when the position has no CSV (→ ``unclassified`` at pool time)."""
    from cellflow.aggregate_quantification.contacts.nls_classification import (
        nls_classification_csv_path,
        read_nls_classification_csv,
    )

    position_path = record.get("position_path")
    if not position_path:
        return None
    csv_path = nls_classification_csv_path(position_path)
    if not csv_path.is_file():
        return None
    labels = read_nls_classification_csv(csv_path)
    if not labels:
        return None
    return {
        "cell_id": np.asarray(list(labels), dtype=np.int64),
        _CLASS_COLUMN: np.asarray(list(labels.values()), dtype=object),
    }


def _metadata(record: dict) -> dict[str, Any]:
    return {
        "condition": str(record.get("condition", "")),
        "date": str(record.get("date", "")),
        "position_id": str(record.get("id", "")),
    }
