"""The Aggregate Quantification studio's Plot area.

The counterpart of the Build area: where the Build area runs *producers*
(:class:`~cellflow.aggregate_quantification.quantifier.Quantifier`) over the
in-scope positions, the Plot area lists every *consumer*
(:class:`~cellflow.napari.aggregate_quantification.plots.Plot`), grouped by
product **family**, and lights each one only when the products it ``consumes``
are built for the scope. A disabled plot says exactly which product it needs, so
the analysis↔plot relationship is visible without reading code.

Clicking an available plot snapshots the scope, reads off the GUI thread
(:meth:`Plot.prepare`), builds the panel (:meth:`Plot.create_panel`), and docks
it as a tab in one shared, constant-size dock (:class:`PlotDockTabs`).
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import groupby
from typing import Any

from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.aggregate_quantification.plots import (
    Plot,
    PlotContext,
    PlotParams,
    available_plots,
)
from cellflow.napari.aggregate_quantification.plugins._plot_dock import PlotDockTabs
from cellflow.napari.studio_plugins import built_quantity_ids
from cellflow.napari.ui_style import action_button, status_label
from cellflow.napari.widgets import CollapsibleSection

# matplotlib's Qt canvas needs a running QApplication; probe it so a headless
# environment degrades to disabled buttons instead of breaking import.
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: F401

    _HAS_MPL_QT = True
except Exception:  # pragma: no cover - exercised only without a Qt matplotlib
    _HAS_MPL_QT = False


class PlotAreaWidget(QWidget):
    """Family-grouped, availability-gated launcher for every registered plot."""

    def __init__(
        self,
        viewer: object | None = None,
        parent: QWidget | None = None,
        *,
        params_provider: Callable[[], PlotParams] | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        #: Supplies the shared plot-time params at launch; the studio wires this to
        #: its one SharedParamsWidget. ``None`` → defaults (standalone / tests).
        self._params_provider = params_provider
        self._records: list[dict] = []
        self._built: frozenset[str] = frozenset()
        self._pool_worker = None
        self._plot_count = 0
        #: button -> plot instance, for availability refresh + launch.
        self._buttons: dict[QPushButton, Plot] = {}
        #: All plots share one dock as tabs (constant size) — see ``_plot_dock.py``.
        self._plot_tabs = PlotDockTabs(self, dock_name="Aggregate plots")

        self._col = QVBoxLayout(self)
        self._col.setContentsMargins(2, 2, 2, 2)
        self._col.setSpacing(4)

        intro = QLabel(
            "Plots are grouped by the input data they read. A plot is enabled once "
            "the product it needs is built for an in-scope position; otherwise it "
            "names the missing product."
        )
        intro.setWordWrap(True)
        status_label(intro, muted=True)
        self._col.addWidget(intro)

        self._col.addWidget(self._build_params_row())

        self._status = QLabel("")
        self._status.setWordWrap(True)
        status_label(self._status, muted=True)
        if not _HAS_MPL_QT:  # pragma: no cover - only without a Qt matplotlib
            self._status.setText("Plotting unavailable (matplotlib Qt backend not usable).")
        self._col.addWidget(self._status)

        self._build_rows()
        self._refresh_availability()

    # -------------------------------------------------------------- shared params
    def _build_params_row(self) -> QWidget:
        """One shared set of plot-time fields applied to whichever plot is launched.

        Each field is "auto" by default; a plot reads only the ones it needs
        (pixel size → potential landscape; FOV area → density; shuffles →
        contact-type z-score) and ignores the rest.
        """
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        self._pixel_size_edit = self._param_field(
            col,
            "Pixel size (µm/px):",
            placeholder="auto",
            tip="µm per pixel for physical-unit axes (potential landscape). Blank "
            "auto-resolves per position from its config / label TIFF.",
        )
        self._fov_edit = self._param_field(
            col,
            "FOV area (mm²):",
            placeholder="auto",
            tip="Field-of-view area for the Density view, applied to all positions. "
            "Blank uses each position's full image area.",
        )
        self._shuffles_edit = self._param_field(
            col,
            "Shuffles:",
            placeholder=str(PlotParams().shuffles),
            tip="Label permutations for the contact-type z-score null.",
        )
        return CollapsibleSection("Plot parameters", body, expanded=False)

    def _param_field(self, layout, label: str, *, placeholder: str, tip: str) -> QLineEdit:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        lbl = QLabel(label)
        lbl.setFixedWidth(120)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setToolTip(tip)
        row.addWidget(lbl)
        row.addWidget(edit, 1)
        layout.addLayout(row)
        return edit

    def _current_params(self) -> PlotParams:
        """Build :class:`PlotParams` from the shared fields (blank/invalid → auto)."""
        shuffles = _parse_int(self._shuffles_edit.text())
        return PlotParams(
            pixel_size_um=_parse_positive(self._pixel_size_edit.text()),
            fov_area_mm2=_parse_positive(self._fov_edit.text()),
            shuffles=shuffles if shuffles and shuffles > 0 else PlotParams().shuffles,
        )

    # ------------------------------------------------------------------ build rows
    def _build_rows(self) -> None:
        """One collapsible per family; one button per plot, built once."""
        plots = available_plots()
        for family, group in groupby(plots, key=lambda cls: cls.family):
            body = QWidget()
            inner = QVBoxLayout(body)
            inner.setContentsMargins(0, 0, 0, 0)
            inner.setSpacing(2)
            for plot_cls in group:
                inner.addWidget(self._plot_row(plot_cls()))
            self._col.addWidget(CollapsibleSection(family or "Other", body, expanded=True))

    def _plot_row(self, plot: Plot) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        button = QPushButton(plot.display_name)
        action_button(button, expand=True)
        button.clicked.connect(lambda _=False, p=plot: self._launch(p))
        self._buttons[button] = plot
        layout.addWidget(button, 1)
        return row

    # -------------------------------------------------------- studio integration
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
        for button, plot in self._buttons.items():
            available = plot.is_available(self._built)
            button.setEnabled(available and idle)
            if not available:
                button.setToolTip(
                    "Needs " + ", ".join(plot.missing(self._built))
                    + " — not built for any in-scope position."
                )
            else:
                button.setToolTip("")

    # --------------------------------------------------------- launching (threaded)
    def _launch(self, plot: Plot) -> None:
        if self._pool_worker is not None:
            return
        records = list(self._records)
        viewer = self.viewer
        loader = getattr(self, "_loader", None)
        params = self._current_params()
        self._status.setText(f"Reading data for {plot.display_name}…")
        self._pool_worker = object()
        self._refresh_availability()

        @thread_worker(
            connect={"returned": self._on_prepared, "errored": self._on_error}
        )
        def _worker():
            return plot, plot.prepare(records, params), records, viewer, loader

        self._pool_worker = _worker()

    def _on_error(self, exc: Exception) -> None:
        self._pool_worker = None
        self._status.setText(f"Plot error: {exc}")
        self._refresh_availability()

    def _on_prepared(self, result: tuple) -> None:
        self._pool_worker = None
        plot, prepared, records, viewer, loader = result
        if _is_empty(prepared):
            self._status.setText(f"No data in scope for {plot.display_name}.")
            self._refresh_availability()
            return
        ctx = PlotContext(
            records=records, viewer=viewer, built=self._built, loader=loader
        )
        panel = plot.create_panel(ctx, prepared=prepared)
        self._plot_count += 1
        name = f"{plot.display_name} {self._plot_count}"
        self._plot_tabs.add(panel, name)
        self._status.setText(f"Opened {name}.")
        self._refresh_availability()


def _parse_positive(text: str) -> float | None:
    """A positive float from *text*, or ``None`` when blank / invalid (→ auto)."""
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _parse_int(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _is_empty(prepared: Any) -> bool:
    """True when a prepared payload carries nothing to plot (empty frame / list)."""
    if prepared is None:
        return True
    empty_attr = getattr(prepared, "empty", None)
    if empty_attr is not None:
        return bool(empty_attr)
    try:
        return len(prepared) == 0
    except TypeError:  # pragma: no cover - non-sized payloads are treated as present
        return False
