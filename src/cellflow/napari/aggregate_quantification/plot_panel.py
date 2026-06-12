"""Generic, detached plotting surface for Aggregate Quantification.

:class:`PlotPanel` is the whole plotting experience: analytical controls that
build a :class:`~cellflow.aggregate_quantification.plotting.PlotSpec`, styling
controls that build a
:class:`~cellflow.aggregate_quantification.plotting.StyleSpec`, an embedded
matplotlib canvas with its native navigation toolbar, and CSV/figure export.

It is constructed from a *snapshot* — ``(dataframe, value_columns,
group_columns)`` — the only quantity-specific knowledge, supplied by the caller.
The panel holds that one tidy DataFrame for its whole life: every control change
is a pure, cheap **re-render**; it never re-pools and never listens back to the
studio. A host hands it to ``viewer.window.add_dock_widget`` to get a floatable,
drag-resizable napari dock.

**Imports qtpy + matplotlib + the headless backend only — no napari**, so it
stays unit-testable without a viewer and reusable by any future quantity that
hands it a different DataFrame + column roles.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.plotting import (
    PlotSpec,
    StyleSpec,
    aggregate,
    build_figure,
    pickable_points,
    potential_table,
    write_csv,
)
from cellflow.napari.ui_style import action_button, status_label
from cellflow.napari.widgets import CollapsibleSection

#: Identity columns carried by a pooled table; the selection payload is these for
#: the picked rows (``frame_start`` lets per-track points report their start
#: frame). Only the ones present in the snapshot are used.
_IDENTITY_COLUMNS = ("position_id", "frame", "frame_start", "cell_id")


@dataclass(frozen=True)
class LoadTarget:
    """A picked point's input data + where to look in it (napari-free).

    Produced by a plugin-supplied resolver, handed to the plugin-supplied
    ``loader`` (and echoed on ``load_requested``) when the user clicks Load;
    the loader turns it into viewer layers.
    """

    path: Path
    kind: str  # "labels"
    frame: int | None
    cell_id: int | None
    identity: dict


@dataclass(frozen=True)
class ValueSource:
    """One value option in a multi-source panel's value picker.

    A render-type panel (e.g. "Distribution") spans values from several pooled
    products; each is a ``ValueSource`` carrying its own pooled ``df``, the
    ``value`` column to plot from it, the ``group_columns`` that df offers, a
    picker ``label`` and a ``source`` header for visible grouping. Selecting one
    swaps the panel onto its df — so products with different rows / group axes
    coexist in one picker without colliding.
    """

    df: pd.DataFrame
    value: str
    group_columns: tuple[str, ...]
    label: str
    source: str  # group header shown in the picker (the product family / view)
    #: Resolves a picked point's identity dict to its input ``LoadTarget`` for
    #: click-to-load — per source, since products carry different label fields.
    #: ``None`` disables loading for this source.
    target_resolver: Callable[[dict], "LoadTarget | None"] | None = None


_PLOT_TYPES = ("hist", "box", "violin", "strip", "swarm", "bar", "line", "potential")
#: Named qualitative palettes offered for the group colors (seaborn names).
_PALETTES = ("tab10", "Set1", "Set2", "Dark2", "Paired", "colorblind", "muted", "deep")
_LEGEND_LOCS = ("best", "upper right", "upper left", "lower right", "lower left", "center")
#: Style sheets offered, filtered to those the installed matplotlib provides.
_STYLE_CANDIDATES = (
    "default", "ggplot", "bmh", "fivethirtyeight", "grayscale",
    "seaborn-v0_8", "seaborn-v0_8-darkgrid", "seaborn-v0_8-whitegrid",
)


class PlotPanel(QWidget):
    """A self-contained plotting dock bound to one snapshot DataFrame."""

    #: Emits the identity dict of a picked point (plain Python scalars). Wired to
    #: the napari highlight via a plugin-supplied resolver; harmless when unwired.
    selection_changed = Signal(object)
    #: Emits a :class:`LoadTarget` when the user clicks the Load button.
    load_requested = Signal(object)

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        value_columns: tuple[str, ...] = (),
        group_columns: tuple[str, ...] = (),
        target_resolver: Callable[[dict], LoadTarget | None] | None = None,
        loader: Callable[[LoadTarget], None] | None = None,
        default_plot: str = "",
        default_adaptive_bins: bool = False,
        value_catalog: list[ValueSource] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        #: Multi-source mode: the value picker spans these products, swapping the
        #: active df on selection. ``None`` → the classic single-snapshot panel.
        self._catalog = value_catalog
        #: combo-index → ValueSource, only in catalog mode (headers map to None).
        self._source_by_index: dict[int, ValueSource] = {}
        if value_catalog:
            dataframe = value_catalog[0].df
            group_columns = value_catalog[0].group_columns
        if dataframe is None:
            raise ValueError("PlotPanel needs a dataframe or a non-empty value_catalog")
        self._df = dataframe
        # Only offer values the snapshot actually carries (symmetric with the
        # identity-column filter below). A stale ``.h5`` built before a column
        # existed — e.g. the per-track ``msd_*`` fit — would otherwise advertise
        # it and crash the render with a cryptic KeyError deep in the backend.
        self._value_columns = tuple(c for c in value_columns if c in dataframe.columns)
        self._group_columns = tuple(group_columns)
        self._identity_columns = tuple(c for c in _IDENTITY_COLUMNS if c in dataframe.columns)
        # In catalog mode the resolver follows the active source (each product
        # has its own label field); it is re-pointed on every value change.
        self._target_resolver = (
            value_catalog[0].target_resolver if value_catalog else target_resolver
        )
        # Held strongly on purpose: a Qt signal connection to a bound method
        # keeps only a weak reference, so a loader owned by nothing but the
        # connection would be GC'd and the Load click would silently no-op.
        self._loader = loader
        self._selected_target: LoadTarget | None = None
        self._canvas: FigureCanvasQTAgg | None = None
        self._toolbar: NavigationToolbar2QT | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        layout.addWidget(CollapsibleSection("Plot", self._build_analytical(), expanded=True))
        layout.addWidget(CollapsibleSection("Styling", self._build_styling(), expanded=False))

        self._canvas_holder = QVBoxLayout()
        layout.addLayout(self._canvas_holder, 1)

        self._path_label = QLabel("Click a point to select its input data.")
        status_label(self._path_label, muted=True)
        self._path_label.setWordWrap(True)
        layout.addWidget(self._path_label)
        self._load_btn = QPushButton("Load in viewer")
        action_button(self._load_btn, expand=True)
        _shrinkable(self._load_btn)
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._on_load_clicked)
        layout.addWidget(self._load_btn)

        layout.addLayout(self._build_exports())
        self._status = QLabel(f"{len(self._df)} row(s) in snapshot.")
        status_label(self._status, muted=True)
        layout.addWidget(self._status)

        # Open on a caller-chosen plot type (e.g. the Potential landscape plugin
        # launches straight into "potential"); signals stay blocked so the single
        # explicit render below does the one draw.
        if default_plot:
            index = self._plot_combo.findData(default_plot)
            if index >= 0:
                blocked = self._plot_combo.blockSignals(True)
                self._plot_combo.setCurrentIndex(index)
                self._plot_combo.blockSignals(blocked)
        if default_adaptive_bins:
            blocked = self._adaptive_bins_cb.blockSignals(True)
            self._adaptive_bins_cb.setChecked(True)
            self._adaptive_bins_cb.blockSignals(blocked)

        self._render()

    # ----------------------------------------------------------- analytical UI
    def _build_analytical(self) -> QWidget:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        self._value_combo = _shrinkable(QComboBox())
        self._populate_value_combo()
        col.addLayout(_labelled("Value:", self._value_combo))

        self._level_combo = _shrinkable(QComboBox())
        # The independent unit for error bars / comparisons. "Per cell" collapses
        # each track's frames to one value first (no frame is its own datapoint);
        # the coarser levels climb to the field of view and the biological replicate.
        self._level_combo.addItem("Per cell (track)", "cell")
        self._level_combo.addItem("Per position", "position")
        self._level_combo.addItem("Per date (replicate)", "date")
        col.addLayout(_labelled("Level:", self._level_combo))

        self._plot_combo = _combo(_PLOT_TYPES)
        col.addLayout(_labelled("Plot:", self._plot_combo))

        self._stat_combo = _combo(("mean", "median", "count"))
        self._error_combo = _combo(("sd", "sem", "none"))
        stat_row = QHBoxLayout()
        stat_row.setContentsMargins(0, 0, 0, 0)
        stat_row.addWidget(QLabel("Stat:"))
        stat_row.addWidget(self._stat_combo, 1)
        stat_row.addWidget(QLabel("Error:"))
        stat_row.addWidget(self._error_combo, 1)
        col.addLayout(stat_row)

        self._bins_spin = _shrinkable(QSpinBox())
        self._bins_spin.setRange(2, 200)
        self._bins_spin.setValue(30)
        col.addLayout(_labelled("Bins:", self._bins_spin))

        # Potential view only: sinh-spaced bins, narrowest at x=0 (like Bins, it
        # stays visible but only bites when Plot=potential).
        self._adaptive_bins_cb = QCheckBox("Tighter bins near 0")
        self._adaptive_bins_cb.setToolTip(
            "Potential view only: sinh-spaced bins, narrowest at x=0, to resolve "
            "the barrier at the transition state (junction length → 0)."
        )
        col.addWidget(self._adaptive_bins_cb)

        # One group-by checkbox per supplied group column (class_label included).
        # "Group by:" sits on its own line and the checkboxes wrap across a
        # two-column grid, so a long column name never sets the panel's min width.
        # The grid is rebuilt when a catalog value swaps to a product with
        # different group axes (see ``_rebuild_group_checks``).
        self._group_checks: dict[str, QCheckBox] = {}
        col.addWidget(QLabel("Group by:"))
        self._group_grid = QGridLayout()
        self._group_grid.setContentsMargins(0, 0, 0, 0)
        self._group_grid.setSpacing(4)
        self._group_grid.setColumnStretch(2, 1)
        col.addLayout(self._group_grid)
        self._rebuild_group_checks()

        # The value combo may swap the active product (catalog mode), so it routes
        # through a dedicated handler; the rest are pure re-renders.
        self._value_combo.currentIndexChanged.connect(self._on_value_changed)
        for combo in (self._level_combo, self._plot_combo,
                      self._stat_combo, self._error_combo):
            combo.currentIndexChanged.connect(self._render)
        self._bins_spin.valueChanged.connect(self._render)
        self._adaptive_bins_cb.toggled.connect(self._render)
        return body

    # ---------------------------------------------------------- value catalog
    def _populate_value_combo(self) -> None:
        """Fill the value picker — flat columns (single mode) or source-grouped
        entries with disabled headers (catalog mode)."""
        combo = self._value_combo
        blocked = combo.blockSignals(True)
        combo.clear()
        self._source_by_index = {}
        if self._catalog is None:
            for name in self._value_columns:
                combo.addItem(name, name)
        else:
            last_source = None
            for src in self._catalog:
                if src.source != last_source:
                    combo.addItem(f"── {src.source} ──", None)
                    item = combo.model().item(combo.count() - 1)
                    item.setEnabled(False)
                    last_source = src.source
                combo.addItem(f"  {src.label}", src.value)
                self._source_by_index[combo.count() - 1] = src
            # Start on the first real (non-header) value.
            for index in range(combo.count()):
                if combo.itemData(index) is not None:
                    combo.setCurrentIndex(index)
                    break
        combo.blockSignals(blocked)

    def _rebuild_group_checks(self) -> None:
        """Repopulate the group-by checkboxes for the current ``group_columns``."""
        while self._group_grid.count():
            item = self._group_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._group_checks = {}
        for i, name in enumerate(self._group_columns):
            check = QCheckBox(name)
            self._group_checks[name] = check
            check.toggled.connect(self._render)
            self._group_grid.addWidget(check, i // 2, i % 2)

    def _on_value_changed(self) -> None:
        """Render; in catalog mode first swap onto the selected value's product."""
        if self._catalog is not None:
            src = self._source_by_index.get(self._value_combo.currentIndex())
            if src is None:  # a header row — ignore
                return
            # Click-to-load follows the selected value's product.
            self._target_resolver = src.target_resolver
            if src.df is not self._df or tuple(src.group_columns) != self._group_columns:
                self._df = src.df
                self._group_columns = tuple(src.group_columns)
                self._identity_columns = tuple(
                    c for c in _IDENTITY_COLUMNS if c in self._df.columns
                )
                self._rebuild_group_checks()
        self._render()

    # -------------------------------------------------------------- styling UI
    def _build_styling(self) -> QWidget:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        self._palette_combo = _combo(_PALETTES)
        col.addLayout(_labelled("Colors:", self._palette_combo))

        styles = [s for s in _STYLE_CANDIDATES if s == "default" or s in mpl.style.available]
        self._style_combo = _combo(styles)
        col.addLayout(_labelled("Style:", self._style_combo))

        self._title_edit = _shrinkable(QLineEdit())
        self._title_edit.setPlaceholderText("auto")
        col.addLayout(_labelled("Title:", self._title_edit))
        self._xlabel_edit = _shrinkable(QLineEdit())
        self._xlabel_edit.setPlaceholderText("auto")
        col.addLayout(_labelled("X label:", self._xlabel_edit))
        self._ylabel_edit = _shrinkable(QLineEdit())
        self._ylabel_edit.setPlaceholderText("auto")
        col.addLayout(_labelled("Y label:", self._ylabel_edit))

        self._width_spin = _double_spin(2.0, 30.0, 6.0, step=0.5)
        self._height_spin = _double_spin(2.0, 30.0, 4.0, step=0.5)
        dim_row = QHBoxLayout()
        dim_row.setContentsMargins(0, 0, 0, 0)
        dim_row.addWidget(QLabel("W×H (in):"))
        dim_row.addWidget(self._width_spin, 1)
        dim_row.addWidget(self._height_spin, 1)
        col.addLayout(dim_row)

        self._font_spin = _double_spin(4.0, 40.0, 10.0, step=1.0)
        col.addLayout(_labelled("Font:", self._font_spin))

        self._grid_cb = QCheckBox("Grid")
        self._legend_cb = QCheckBox("Legend")
        self._legend_cb.setChecked(True)
        self._legend_loc_combo = _combo(_LEGEND_LOCS)
        col.addLayout(_labelled("Legend loc:", self._legend_loc_combo))

        # Box-plot knobs — they only bite when Plot=box, but stay visible (like
        # Bins for hist) rather than appearing and disappearing on plot change.
        self._box_whis_spin = _double_spin(0.0, 100.0, 1.5, step=0.5)
        self._box_fliers_cb = QCheckBox("Outliers")
        self._box_fliers_cb.setChecked(True)
        self._box_notch_cb = QCheckBox("Notch (CI)")
        whis_row = QHBoxLayout()
        whis_row.setContentsMargins(0, 0, 0, 0)
        whis_row.addWidget(QLabel("Box whis ×IQR:"))
        whis_row.addWidget(self._box_whis_spin, 1)
        col.addLayout(whis_row)

        # The four toggles share a 2×2 grid so no single checkbox row sets the
        # panel's minimum width — it can shrink to roughly one toggle wide.
        checks = QGridLayout()
        checks.setContentsMargins(0, 0, 0, 0)
        checks.setSpacing(4)
        checks.addWidget(self._grid_cb, 0, 0)
        checks.addWidget(self._legend_cb, 0, 1)
        checks.addWidget(self._box_fliers_cb, 1, 0)
        checks.addWidget(self._box_notch_cb, 1, 1)
        checks.setColumnStretch(2, 1)
        col.addLayout(checks)

        # Axis-range overrides; blank ("auto") keeps matplotlib's autoscale.
        self._xmin_edit = _range_edit()
        self._xmax_edit = _range_edit()
        self._ymin_edit = _range_edit()
        self._ymax_edit = _range_edit()
        xr = QHBoxLayout()
        xr.setContentsMargins(0, 0, 0, 0)
        xr.addWidget(QLabel("X range:"))
        xr.addWidget(self._xmin_edit, 1)
        xr.addWidget(self._xmax_edit, 1)
        col.addLayout(xr)
        yr = QHBoxLayout()
        yr.setContentsMargins(0, 0, 0, 0)
        yr.addWidget(QLabel("Y range:"))
        yr.addWidget(self._ymin_edit, 1)
        yr.addWidget(self._ymax_edit, 1)
        col.addLayout(yr)
        for edit in (self._xmin_edit, self._xmax_edit, self._ymin_edit, self._ymax_edit):
            edit.editingFinished.connect(self._render)

        for combo in (self._palette_combo, self._style_combo, self._legend_loc_combo):
            combo.currentIndexChanged.connect(self._render)
        for edit in (self._title_edit, self._xlabel_edit, self._ylabel_edit):
            edit.editingFinished.connect(self._render)
        for spin in (self._width_spin, self._height_spin, self._font_spin, self._box_whis_spin):
            spin.valueChanged.connect(self._render)
        for check in (self._grid_cb, self._legend_cb, self._box_fliers_cb, self._box_notch_cb):
            check.toggled.connect(self._render)
        return body

    # --------------------------------------------------------------- export UI
    def _build_exports(self) -> QGridLayout:
        # Equal-width export buttons on a "Export:" row: tidy columns that share
        # the width evenly and collapse together as the panel narrows.
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        grid.addWidget(QLabel("Export:"), 0, 0)
        self._export_pooled_btn = _export_button("Pooled CSV", self._export_pooled)
        self._export_agg_btn = _export_button("Aggregated CSV", self._export_aggregated)
        self._export_fig_btn = _export_button("Figure", self._export_figure)
        buttons = (self._export_pooled_btn, self._export_agg_btn, self._export_fig_btn)
        for col, btn in enumerate(buttons, start=1):
            grid.addWidget(btn, 0, col)
            grid.setColumnStretch(col, 1)
            btn.setEnabled(not self._df.empty)
        return grid

    # ----------------------------------------------------------------- specs
    def current_spec(self) -> PlotSpec:
        group_by = tuple(name for name, check in self._group_checks.items() if check.isChecked())
        return PlotSpec(
            value=self._value_combo.currentData(),
            group_by=group_by,
            level=self._level_combo.currentData(),
            plot=self._plot_combo.currentData(),
            stat=self._stat_combo.currentData(),
            error=self._error_combo.currentData(),
            bins=self._bins_spin.value(),
            bin_mode="adaptive" if self._adaptive_bins_cb.isChecked() else "uniform",
        )

    def current_style(self) -> StyleSpec:
        return StyleSpec(
            palette=self._palette_combo.currentData(),
            title=self._title_edit.text().strip(),
            xlabel=self._xlabel_edit.text().strip(),
            ylabel=self._ylabel_edit.text().strip(),
            style=self._style_combo.currentData(),
            width=self._width_spin.value(),
            height=self._height_spin.value(),
            grid=self._grid_cb.isChecked(),
            legend=self._legend_cb.isChecked(),
            legend_loc=self._legend_loc_combo.currentData(),
            font_size=self._font_spin.value(),
            box_whis=self._box_whis_spin.value(),
            box_showfliers=self._box_fliers_cb.isChecked(),
            box_notch=self._box_notch_cb.isChecked(),
            xmin=_parse_float(self._xmin_edit.text()),
            xmax=_parse_float(self._xmax_edit.text()),
            ymin=_parse_float(self._ymin_edit.text()),
            ymax=_parse_float(self._ymax_edit.text()),
        )

    # ---------------------------------------------------------------- render
    def _render(self) -> None:
        """Pure re-render from the held snapshot — never re-pools."""
        fig = build_figure(self._df, self.current_spec(), self.current_style())
        canvas = FigureCanvasQTAgg(fig)
        _shrinkable(canvas)
        toolbar = NavigationToolbar2QT(canvas, self)
        # As a QToolBar it spills overflow tools into a "»" menu when squeezed,
        # so it stops being the panel's hard minimum width.
        _shrinkable(toolbar)
        if self._canvas is not None:
            self._canvas_holder.removeWidget(self._canvas)
            self._canvas.setParent(None)
            self._canvas.deleteLater()
        if self._toolbar is not None:
            self._canvas_holder.removeWidget(self._toolbar)
            self._toolbar.setParent(None)
            self._toolbar.deleteLater()
        self._canvas_holder.addWidget(toolbar)
        self._canvas_holder.addWidget(canvas, 1)
        self._canvas, self._toolbar = canvas, toolbar
        canvas.mpl_connect("button_press_event", self._on_pick)
        self._clear_selection()

    # -------------------------------------------------------------- click-to-load
    def _category_x(self) -> dict[str, float]:
        """Map each drawn x-axis category label to its x position."""
        if self._canvas is None:
            return {}
        ax = self._canvas.figure.axes[0]
        ticks = ax.get_xticks()
        labels = [t.get_text() for t in ax.get_xticklabels()]
        return {lab: float(x) for x, lab in zip(ticks, labels) if lab}

    def _nearest_row_index(self, xdata: float, ydata: float) -> int | None:
        """Row whose plotted point is nearest the click: snap to the x-category,
        then the closest value within it. Returns None when nothing is pickable."""
        pts = pickable_points(self._df, self.current_spec(), self.current_style())
        if not pts:
            return None
        cat_x = self._category_x()
        if cat_x:
            cat = min(cat_x, key=lambda c: abs(cat_x[c] - xdata))  # snap to category
            candidates = [p for p in pts if p.category == cat]
        else:
            candidates = list(pts)  # single, ungrouped bucket
        if not candidates:
            return None
        return min(candidates, key=lambda p: abs(p.value - ydata)).row_index

    def _on_pick(self, event) -> None:
        if self._target_resolver is None or event.inaxes is None or event.ydata is None:
            return
        row = self._nearest_row_index(event.xdata, event.ydata)
        if row is not None:
            self._select_row(row)

    def _select_row(self, row: int) -> None:
        record = self._df.iloc[row]
        identity = {c: _py(record[c]) for c in self._identity_columns}
        self.selection_changed.emit(identity)
        target = self._target_resolver(identity) if self._target_resolver else None
        self._selected_target = target
        if target is None:
            self._path_label.setText("No input data found for this point.")
            self._load_btn.setEnabled(False)
        else:
            self._path_label.setText(str(target.path))
            self._load_btn.setEnabled(True)

    def _clear_selection(self) -> None:
        self._selected_target = None
        if hasattr(self, "_load_btn"):
            self._load_btn.setEnabled(False)
            self._path_label.setText("Click a point to select its input data.")

    def _on_load_clicked(self) -> None:
        if self._selected_target is None:
            return
        self.load_requested.emit(self._selected_target)
        if self._loader is not None:
            self._loader(self._selected_target)

    # ---------------------------------------------------------------- exports
    def _save_path(self, caption: str, filt: str) -> Path | None:
        path, _ = QFileDialog.getSaveFileName(self, caption, filter=filt)
        return Path(path) if path else None

    def _export_pooled(self) -> None:
        path = self._save_path("Export pooled table", "CSV files (*.csv)")
        if path:
            self._status.setText(f"Wrote {write_csv(self._df, path).name}.")

    def _export_aggregated(self) -> None:
        path = self._save_path("Export aggregated table", "CSV files (*.csv)")
        if path:
            spec = self.current_spec()
            # For the potential view the "aggregated" table is the plotted curve
            # (group · center · U · counts · ΔE_eff), not a per-unit summary, so
            # the exported numbers match what is drawn.
            summary = (
                potential_table(self._df, spec)
                if spec.plot == "potential"
                else aggregate(self._df, spec)
            )
            self._status.setText(f"Wrote {write_csv(summary, path).name}.")

    def _export_figure(self) -> None:
        if self._canvas is None:
            return
        path = self._save_path("Export figure", "Images (*.png *.svg)")
        if path:
            self._canvas.figure.savefig(path)
            self._status.setText(f"Wrote {path.name}.")


# --------------------------------------------------------------- UI factories
def _shrinkable(widget: QWidget) -> QWidget:
    """Let *widget* shrink below its content width so the panel can narrow.

    ``Ignored`` drops the widget's width hint as a floor, so a field fills its
    row but also collapses when the dock is dragged narrow — the panel shrinks
    in place instead of forcing a scrollbar or a hard minimum."""
    widget.setMinimumWidth(0)
    policy = widget.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Ignored)
    widget.setSizePolicy(policy)
    return widget


def _combo(items) -> QComboBox:
    combo = QComboBox()
    for item in items:
        combo.addItem(item, item)
    return _shrinkable(combo)


def _double_spin(lo: float, hi: float, value: float, step: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setValue(value)
    return _shrinkable(spin)


def _labelled(label: str, widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label)
    lbl.setFixedWidth(70)
    row.addWidget(lbl)
    row.addWidget(widget, 1)
    return row


def _export_button(text: str, slot) -> QPushButton:
    btn = QPushButton(text)
    action_button(btn, expand=True)
    _shrinkable(btn)  # Ignored policy: collapse with the panel, don't floor it
    btn.clicked.connect(slot)
    return btn


def _range_edit() -> QLineEdit:
    edit = QLineEdit()
    edit.setPlaceholderText("auto")
    return _shrinkable(edit)


def _parse_float(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _py(value):
    try:
        return value.item()  # numpy scalar -> Python scalar
    except AttributeError:
        return value
