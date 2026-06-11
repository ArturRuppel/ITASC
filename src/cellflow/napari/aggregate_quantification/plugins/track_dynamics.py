"""Track Dynamics group plugin — scope-aware compute + launch-plot for motion.

The "one plugin per logical group" shape (see :mod:`.shape`), for the dynamics
quantities. A scope dropdown selects which registered quantifier drives Compute
and Plot:

* ``cell`` / ``nucleus`` build & plot motion from that label source.

Two sections:

* **Compute** — a Build button computing the scope-selected ``*_dynamics.h5`` for
  the in-scope positions, delegating to the studio's centralized (threaded)
  build path. Because this plugin *owns* both dynamics quantities
  (``owns_quantities``), the studio suppresses their generic auto-builders.
  Dynamics needs both a **pixel size** (µm/px) and a **frame interval** (s/frame);
  either can be overridden here or auto-resolved per position.
* **Plot** — a *View* selector and a **"Plot…"** button. *Per-frame* and
  *Per-track* pool tidy tables off-thread and open the generic
  :class:`~cellflow.napari.aggregate_quantification.plot_panel.PlotPanel`;
  *Curves* opens the bespoke
  :class:`~cellflow.napari.aggregate_quantification.dynamics_curves_panel.DynamicsCurvesPanel`
  (MSD / DAC / velocity correlation).

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

from cellflow.aggregate_quantification.dynamics import read_track_dynamics
from cellflow.aggregate_quantification.frame_interval import resolve_time_interval_s
from cellflow.aggregate_quantification.pixel_size import resolve_pixel_size_um
from cellflow.aggregate_quantification.plotting import PositionSource, pool_object_tables
from cellflow.aggregate_quantification.quantifiers.cell_dynamics import (
    CellDynamicsQuantifier,
)
from cellflow.aggregate_quantification.quantifiers.nucleus_dynamics import (
    NucleusDynamicsQuantifier,
)
from cellflow.napari.aggregate_quantification.plugins import AnalysisContext, AnalysisPlugin
from cellflow.napari.aggregate_quantification.plugins._plot_dock import PlotDockTabs
from cellflow.napari.ui_style import action_button, status_label
from cellflow.napari.widgets import CollapsibleSection

# matplotlib's Qt canvas needs a running QApplication; probe it so a headless
# environment degrades to a disabled button instead of breaking discovery. The
# panels are imported lazily at click time.
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: F401

    _HAS_MPL_QT = True
except Exception:  # pragma: no cover - exercised only without a Qt matplotlib
    _HAS_MPL_QT = False

_CELL_FIELD = "cell_tracked_labels_path"
_NUCLEUS_FIELD = "nucleus_tracked_labels_path"

#: Catalogue-metadata axes a plot can group/facet by.
_METADATA_GROUPS = ("condition", "date", "position_id")
_CLASS_COLUMN = "class_label"

#: Per-frame (instantaneous) plottable value columns + the group axes it offers.
_FRAME_VALUES = ("speed_um_per_s", "vx_um_per_s", "vy_um_per_s", "net_disp_um")
_FRAME_GROUPS = (*_METADATA_GROUPS, _CLASS_COLUMN, "frame")
#: Per-track (summary) plottable value columns + group axes (no frame axis).
_TRACK_VALUES = (
    "curvilinear_speed_um_per_s",
    "net_speed_um_per_s",
    "directionality_ratio",
    "persistence_time_s",
    "path_length_um",
    "net_displacement_um",
    "duration_s",
)
_TRACK_GROUPS = (*_METADATA_GROUPS, _CLASS_COLUMN)


class TrackDynamicsPlugin(AnalysisPlugin):
    """Compute track motion and launch detached distribution / curve panels."""

    plugin_id = "track_dynamics"
    display_name = "Track dynamics"
    # cell OR nucleus suffices, which a static AND-gated ``requires`` cannot
    # express — so the plugin is always offered and gates per-scope internally.
    requires = ()
    owns_quantities = ("cell_dynamics", "nucleus_dynamics")

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(viewer=viewer, parent=parent)
        self._quantifiers = {
            "cell": CellDynamicsQuantifier(),
            "nucleus": NucleusDynamicsQuantifier(),
        }
        self._records: list[dict] = []
        self._build_callback = None
        self._pool_worker = None
        self._scope_user_set = False
        self._plot_count = 0
        #: All plots share one dock as tabs (constant size) — see ``_plot_dock.py``.
        self._plot_tabs = PlotDockTabs(self, dock_name="Dynamics plots")

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
        self._scope_combo.setToolTip(
            "Measure centroid motion from the cell or nucleus tracked labels. "
            "Nuclei are compact, point-like centroids — the robust motility default."
        )
        self._scope_combo.currentIndexChanged.connect(lambda _=None: self._update_enabled())
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

        self._pixel_size_edit = self._labelled_edit(
            col, "Pixel size (µm/px):",
            "µm per pixel. Leave blank to auto-resolve per position from its "
            "cellflow_config.json or the label TIFF; a value here applies to all.",
        )
        self._interval_edit = self._labelled_edit(
            col, "Frame interval (s):",
            "Seconds per frame. Leave blank to auto-resolve per position from its "
            "cellflow_config.json or the label TIFF; a value here applies to all.",
        )

        self._overwrite_cb = QCheckBox("Recompute (overwrite existing)")
        col.addWidget(self._overwrite_cb)

        self._build_btn = QPushButton("Build dynamics")
        action_button(self._build_btn, expand=True)
        self._build_btn.clicked.connect(self._on_build)
        col.addWidget(self._build_btn)
        return body

    def _labelled_edit(self, col: QVBoxLayout, label: str, tooltip: str) -> QLineEdit:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label))
        edit = QLineEdit()
        edit.setPlaceholderText("auto")
        edit.setToolTip(tooltip)
        edit.textChanged.connect(lambda _=None: self._update_enabled())
        row.addWidget(edit, 1)
        col.addLayout(row)
        return edit

    # ------------------------------------------------------------------ plot UI
    def _build_plot(self) -> QWidget:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        self._view_combo = QComboBox()
        self._view_combo.addItem("Per-frame (speed, velocity…)", "frame")
        self._view_combo.addItem("Per-track (persistence, directionality…)", "track")
        self._view_combo.addItem("Curves (MSD / DAC / C(r))", "curves")
        col.addWidget(self._view_combo)

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

    @property
    def _view(self) -> str:
        return self._view_combo.currentData() or "frame"

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
        scope = self._default_scope(self._records)
        index = self._scope_combo.findData(scope)
        if index >= 0:
            blocked = self._scope_combo.blockSignals(True)
            self._scope_combo.setCurrentIndex(index)
            self._scope_combo.blockSignals(blocked)

    @staticmethod
    def _default_scope(records: list[dict]) -> str:
        if any(r.get(_CELL_FIELD) for r in records):
            return "cell"
        if any(r.get(_NUCLEUS_FIELD) for r in records):
            return "nucleus"
        return "cell"

    def _label_field(self, scope: str | None = None) -> str:
        return _NUCLEUS_FIELD if (scope or self._scope) == "nucleus" else _CELL_FIELD

    def _has_inputs(self, record: dict) -> bool:
        return bool(record.get(self._label_field()))

    def _update_enabled(self) -> None:
        records = self._records
        noun = self._scope
        has_inputs = [r for r in records if self._has_inputs(r)]
        buildable = [r for r in has_inputs if self._is_buildable(r)]
        missing = len(has_inputs) - len(buildable)
        built = [r for r in records if self._is_built(self._quantifier, r)]
        if not records:
            self._compute_status.setText("No positions in scope.")
        else:
            status = (
                f"{len(has_inputs)} of {len(records)} in-scope position(s) have "
                f"{noun} labels; {len(built)} already built."
            )
            if missing:
                status += (
                    f" {missing} need a pixel size and/or frame interval — "
                    "enter them above to enable them."
                )
            self._compute_status.setText(status)
        self._build_btn.setEnabled(bool(buildable) and self._build_callback is not None)
        self._plot_btn.setEnabled(
            bool(built) and _HAS_MPL_QT and self.viewer is not None and self._pool_worker is None
        )

    # ----------------------------------------------------------------- building
    def _manual(self, edit: QLineEdit) -> float | None:
        text = edit.text().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if value > 0 else None

    def _pixel_size_for(self, record: dict) -> float | None:
        manual = self._manual(self._pixel_size_edit)
        if manual is not None:
            return manual
        return resolve_pixel_size_um(record.get("position_path"), record.get(self._label_field()))

    def _interval_for(self, record: dict) -> float | None:
        manual = self._manual(self._interval_edit)
        if manual is not None:
            return manual
        return resolve_time_interval_s(
            record.get("position_path"), record.get(self._label_field())
        )

    def _is_buildable(self, record: dict) -> bool:
        return self._pixel_size_for(record) is not None and self._interval_for(record) is not None

    def _stamped(self, record: dict) -> dict:
        """A record copy carrying the manual pixel-size / interval overrides."""
        out = dict(record)
        px = self._manual(self._pixel_size_edit)
        dt = self._manual(self._interval_edit)
        if px is not None:
            out["pixel_size_um"] = px
        if dt is not None:
            out["time_interval_s"] = dt
        return out

    def _on_build(self) -> None:
        if self._build_callback is None:
            return
        records = [self._stamped(r) for r in self._records]
        self._build_callback(self._quantifier, records, self._overwrite_cb.isChecked())

    # ------------------------------------------------------- pooling + launching
    def _on_plot(self) -> None:
        records = list(self._records)
        quantifier = self._quantifier
        view = self._view
        self._plot_status.setText("Reading dynamics tables…")
        self._pool_worker = object()
        self._update_enabled()

        @thread_worker(connect={"returned": self._on_pool_done, "errored": self._on_pool_error})
        def _worker():
            if view == "curves":
                return ("curves", _curve_records(quantifier, records))
            return (view, _pool_records(quantifier, records, view))

        self._pool_worker = _worker()

    def _on_pool_error(self, exc: Exception) -> None:
        self._pool_worker = None
        self._plot_status.setText(f"Plot error: {exc}")
        self._update_enabled()

    def _on_pool_done(self, result: tuple[str, Any]) -> None:
        self._pool_worker = None
        view, payload = result
        if view == "curves":
            self._open_curves(payload)
        else:
            self._open_distribution(view, payload)
        self._update_enabled()

    def _open_distribution(self, view: str, pooled: pd.DataFrame) -> None:
        if pooled.empty:
            self._plot_status.setText("No built dynamics tables in scope.")
            return
        from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel

        values, groups = (
            (_FRAME_VALUES, _FRAME_GROUPS) if view == "frame" else (_TRACK_VALUES, _TRACK_GROUPS)
        )
        panel = PlotPanel(pooled, value_columns=values, group_columns=groups)
        name = self._dock_name()
        self._add_dock(panel, name)
        self._plot_status.setText(f"Opened {name} ({len(pooled)} rows).")

    def _open_curves(self, curves: list) -> None:
        if not curves:
            self._plot_status.setText("No built dynamics tables in scope.")
            return
        from cellflow.napari.aggregate_quantification.dynamics_curves_panel import (
            DynamicsCurvesPanel,
        )

        panel = DynamicsCurvesPanel(curves)
        name = self._dock_name()
        self._add_dock(panel, name)
        self._plot_status.setText(f"Opened {name} ({len(curves)} position curves).")

    def _dock_name(self) -> str:
        self._plot_count += 1
        return f"Plot {self._plot_count}"

    def _add_dock(self, panel: QWidget, name: str) -> None:
        self._plot_tabs.add(panel, name)

    # ----------------------------------------------------------- path resolution
    def _is_built(self, quantifier, record: dict) -> bool:
        from cellflow.napari.studio_plugins import output_for_record

        return quantifier.is_built(output_for_record(quantifier, record))


# ----------------------------------------------------------------- pooling (off-thread)
def _pool_records(quantifier, records, view: str) -> pd.DataFrame:
    """Pool the per-frame (instantaneous) or per-track table across built positions.

    Both left-join the NLS sidecar CSV's subpopulation ``class_label`` (by
    ``cell_id``); a position without a CSV contributes ``unclassified``. Runs off
    the GUI thread.
    """
    from cellflow.napari.studio_plugins import output_for_record

    sources: list[PositionSource] = []
    for record in records:
        path = output_for_record(quantifier, record)
        if not quantifier.is_built(path):
            continue
        table = (
            quantifier.object_table(path)
            if view == "frame"
            else read_track_dynamics(path).tracks
        )
        sources.append(
            PositionSource(
                metadata=_metadata(record),
                table=table,
                join_table=_nls_join_table(record),
                join_columns=(_CLASS_COLUMN,),
            )
        )
    return pool_object_tables(sources)


def _curve_records(quantifier, records) -> list:
    """Read each built position's curve set (MSD / DAC / C(r) + fits), off-thread."""
    from cellflow.napari.aggregate_quantification.dynamics_curves_panel import CurveSet
    from cellflow.napari.studio_plugins import output_for_record

    curves: list[CurveSet] = []
    for record in records:
        path = output_for_record(quantifier, record)
        if not quantifier.is_built(path):
            continue
        dyn = read_track_dynamics(path)
        curves.append(
            CurveSet(
                group=str(record.get("condition", "") or record.get("id", "")),
                msd_lag_s=dyn.msd["lag_s"],
                msd_um2=dyn.msd["msd_um2"],
                msd_D_um2_per_s=dyn.msd_D_um2_per_s,
                msd_alpha=dyn.msd_alpha,
                dac_lag_s=dyn.dac["lag_s"],
                dac=dyn.dac["dac"],
                dac_persistence_time_s=dyn.dac_persistence_time_s,
                corr_separation_um=dyn.corr_curve.get("separation_um", np.asarray([])),
                corr=dyn.corr_curve.get("corr", np.asarray([])),
            )
        )
    return curves


def _nls_join_table(record) -> dict[str, np.ndarray] | None:
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
