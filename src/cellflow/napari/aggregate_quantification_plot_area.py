"""The Aggregate Quantification studio's Plot area.

The counterpart of the Build area: where the Build area runs *producers*
(:class:`~cellflow.aggregate_quantification.quantifier.Quantifier`) over the
in-scope positions, the Plot area lets you *consume* whatever they built. It
offers **one button per render type** — Distribution, Bar, Potential landscape,
Curves — rather than one per product. Clicking a render-type button pools every
available product of that type and opens a single panel whose **value picker
spans them all, grouped by source** (the family header), so e.g. one
"Distribution" button covers cell/nucleus shape, dynamics, and neighbor counts.

A render-type button is enabled only when at least one product feeding it is
built for the scope. Reading happens off the GUI thread (:meth:`Plot.prepare`);
the panel is then docked as a tab in one shared, constant-size dock.
"""
from __future__ import annotations

from typing import Any

from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel, ValueSource
from cellflow.napari.aggregate_quantification.plots import (
    Plot,
    PlotParams,
    available_plots,
)
from cellflow.napari.aggregate_quantification.plugins._plot_dock import PlotDockTabs
from cellflow.napari.studio_plugins import built_quantity_ids
from cellflow.napari.ui_style import action_button, status_label

# matplotlib's Qt canvas needs a running QApplication; probe it so a headless
# environment degrades to disabled buttons instead of breaking import.
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: F401

    _HAS_MPL_QT = True
except Exception:  # pragma: no cover - exercised only without a Qt matplotlib
    _HAS_MPL_QT = False

#: Catalog render types → (button label, PlotPanel default plot, adaptive bins).
#: These open one multi-source PlotPanel whose value picker spans the type's
#: products. Order here is the button order.
_CATALOG_TYPES: dict[str, tuple[str, str, bool]] = {
    "distribution": ("Distribution (box / violin / strip / hist)", "box", False),
    "bar": ("Bar charts", "bar", False),
    "potential": ("Potential landscape", "potential", True),
}
#: The bespoke curve render type opens its own panel (no value picker).
_CURVE_TYPE = "curve"
_CURVE_LABEL = "Curves (MSD / DAC / C(r))"


class PlotAreaWidget(QWidget):
    """Render-type launcher with availability gating + a spanning value picker."""

    def __init__(
        self,
        viewer: object | None = None,
        parent: QWidget | None = None,
        *,
        params_provider=None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        #: Supplies the shared plot-time params at launch; the studio wires this to
        #: its one SharedParamsWidget. ``None`` → defaults (standalone / tests).
        self._params_provider = params_provider
        self._records: list[dict] = []
        self._built: frozenset[str] = frozenset()
        self._loader = None
        self._pool_worker = None
        self._plot_count = 0
        #: One instance per registered plot, kept for availability + pooling.
        self._plots: list[Plot] = [cls() for cls in available_plots()]
        #: button -> render_type.
        self._buttons: dict[QPushButton, str] = {}
        #: All plots share one dock as tabs (constant size) — see ``_plot_dock.py``.
        self._plot_tabs = PlotDockTabs(self, dock_name="Aggregate plots")

        col = QVBoxLayout(self)
        col.setContentsMargins(2, 2, 2, 2)
        col.setSpacing(4)

        intro = QLabel(
            "One button per plot type. The value picker inside spans every "
            "available quantity of that type, grouped by source; a button is "
            "enabled once a product feeding it is built for an in-scope position."
        )
        intro.setWordWrap(True)
        status_label(intro, muted=True)
        col.addWidget(intro)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        status_label(self._status, muted=True)
        if not _HAS_MPL_QT:  # pragma: no cover - only without a Qt matplotlib
            self._status.setText("Plotting unavailable (matplotlib Qt backend not usable).")
        col.addWidget(self._status)

        for render_type in self._render_types():
            label = (
                _CATALOG_TYPES[render_type][0]
                if render_type in _CATALOG_TYPES
                else _CURVE_LABEL
            )
            col.addWidget(self._render_button(render_type, label))
        col.addStretch(1)
        self._refresh_availability()

    # ------------------------------------------------------------------ rows
    def _render_types(self) -> list[str]:
        """Render types that have at least one registered plot, in button order."""
        present = {plot.render_type for plot in self._plots}
        ordered = [t for t in _CATALOG_TYPES if t in present]
        if _CURVE_TYPE in present:
            ordered.append(_CURVE_TYPE)
        return ordered

    def _render_button(self, render_type: str, label: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        button = QPushButton(label)
        action_button(button, expand=True)
        button.clicked.connect(lambda _=False, t=render_type: self._launch(t))
        self._buttons[button] = render_type
        layout.addWidget(button, 1)
        return row

    def _plots_of_type(self, render_type: str) -> list[Plot]:
        return [p for p in self._plots if p.render_type == render_type]

    def _available_of_type(self, render_type: str) -> list[Plot]:
        return [p for p in self._plots_of_type(render_type) if p.is_available(self._built)]

    # -------------------------------------------------------- studio integration
    def _params(self) -> PlotParams:
        return self._params_provider() if self._params_provider else PlotParams()

    def set_context(self, ctx: Any) -> None:
        """Receive the catalogue scope (an ``AnalysisContext``-shaped object)."""
        if getattr(ctx, "viewer", None) is not None:
            self.viewer = ctx.viewer
        self._records = list(getattr(ctx, "records", []))
        self._loader = getattr(ctx, "loader", None)
        self._built = built_quantity_ids(self._records)
        self._refresh_availability()

    def _refresh_availability(self) -> None:
        idle = self._pool_worker is None and _HAS_MPL_QT and self.viewer is not None
        for button, render_type in self._buttons.items():
            available = bool(self._available_of_type(render_type))
            button.setEnabled(available and idle)
            button.setToolTip(
                ""
                if available
                else "No product feeding this plot type is built for an in-scope "
                "position yet."
            )

    # --------------------------------------------------------- launching (threaded)
    def _launch(self, render_type: str) -> None:
        if self._pool_worker is not None:
            return
        plots = self._available_of_type(render_type)
        if not plots:
            return
        records = list(self._records)
        params = self._params()
        self._status.setText("Reading data…")
        self._pool_worker = object()
        self._refresh_availability()

        @thread_worker(connect={"returned": self._on_prepared, "errored": self._on_error})
        def _worker():
            # Scope the dynamics read cache here (on the worker thread, where the
            # reads run) so the four dynamics views pooled by one button share a
            # single read per position instead of re-parsing each .h5 ~6×.
            from cellflow.napari.aggregate_quantification.plots.dynamics import (
                dynamics_read_cache,
            )

            with dynamics_read_cache():
                prepared = [(plot, plot.prepare(records, params)) for plot in plots]
            return render_type, prepared, records

        self._pool_worker = _worker()

    def _on_error(self, exc: Exception) -> None:
        self._pool_worker = None
        self._status.setText(f"Plot error: {exc}")
        self._refresh_availability()

    def _on_prepared(self, result: tuple) -> None:
        self._pool_worker = None
        render_type, prepared, records = result
        if render_type == _CURVE_TYPE:
            panel = self._build_curve_panel(prepared, records)
        else:
            panel = self._build_catalog_panel(render_type, prepared, records)
        if panel is None:
            self._status.setText("No data in scope for this plot type.")
            self._refresh_availability()
            return
        self._plot_count += 1
        name = self._dock_name(render_type)
        self._plot_tabs.add(panel, name)
        self._status.setText(f"Opened {name}.")
        self._refresh_availability()

    def _build_catalog_panel(
        self, render_type: str, prepared: list, records: list[dict]
    ) -> QWidget | None:
        """Assemble the spanning value catalog and open one multi-source panel.

        A single :class:`ClickToLoad` controller backs the panel; each product's
        values carry a resolver against that product's label field (``None`` when
        the product has no label source), so picking a point loads the right input
        regardless of which value is shown.
        """
        from cellflow.napari.aggregate_quantification.plugins._click_to_load import (
            ClickToLoad,
        )

        controller = ClickToLoad(self.viewer)
        catalog: list[ValueSource] = []
        for plot, df in prepared:
            if df is None or getattr(df, "empty", True):
                continue
            label_field = getattr(plot, "label_field", None)
            resolver = (
                controller.resolver(records, label_field) if label_field else None
            )
            for value in plot.value_columns:
                if value in df.columns:
                    catalog.append(
                        ValueSource(
                            df=df,
                            value=value,
                            group_columns=tuple(plot.group_columns),
                            label=f"{plot.display_name}: {value}",
                            source=plot.family,
                            target_resolver=resolver,
                        )
                    )
        if not catalog:
            return None
        _label, default_plot, adaptive = _CATALOG_TYPES[render_type]
        # loader is the shared controller's load; held strongly by the panel so
        # the controller (and thus every source's resolver) stays alive.
        return PlotPanel(
            value_catalog=catalog,
            default_plot=default_plot,
            default_adaptive_bins=adaptive,
            loader=controller.load,
        )

    def _build_curve_panel(self, prepared: list, records: list[dict]) -> QWidget | None:
        """Combine every available position's curve sets into one bespoke panel."""
        curves: list = []
        for _plot, prepared_curves in prepared:
            curves.extend(prepared_curves or [])
        if not curves:
            return None
        from cellflow.napari.aggregate_quantification.dynamics_curves_panel import (
            DynamicsCurvesPanel,
        )

        return DynamicsCurvesPanel(curves)

    def _dock_name(self, render_type: str) -> str:
        base = (
            _CATALOG_TYPES[render_type][0]
            if render_type in _CATALOG_TYPES
            else _CURVE_LABEL
        )
        return f"{base.split(' (')[0]} {self._plot_count}"
