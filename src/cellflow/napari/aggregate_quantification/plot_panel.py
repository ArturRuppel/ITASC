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
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
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

    Produced by a plugin-supplied resolver, carried out of the panel via
    ``load_requested``; the plugin's loader turns it into viewer layers.
    """

    path: Path
    kind: str  # "labels"
    frame: int | None
    cell_id: int | None
    identity: dict

_PLOT_TYPES = ("hist", "box", "violin", "strip", "swarm", "bar", "line")
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
        dataframe: pd.DataFrame,
        value_columns: tuple[str, ...],
        group_columns: tuple[str, ...],
        target_resolver: Callable[[dict], LoadTarget | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._df = dataframe
        self._value_columns = tuple(value_columns)
        self._group_columns = tuple(group_columns)
        self._identity_columns = tuple(c for c in _IDENTITY_COLUMNS if c in dataframe.columns)
        self._target_resolver = target_resolver
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
        action_button(self._load_btn)
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._on_load_clicked)
        layout.addWidget(self._load_btn)

        layout.addLayout(self._build_exports())
        self._status = QLabel(f"{len(self._df)} row(s) in snapshot.")
        status_label(self._status, muted=True)
        layout.addWidget(self._status)

        self._render()

    # ----------------------------------------------------------- analytical UI
    def _build_analytical(self) -> QWidget:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        self._value_combo = _combo(self._value_columns)
        col.addLayout(_labelled("Value:", self._value_combo))

        self._level_combo = QComboBox()
        self._level_combo.addItem("Per cell (pooled)", "cell")
        self._level_combo.addItem("Per position", "position")
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

        self._bins_spin = QSpinBox()
        self._bins_spin.setRange(2, 200)
        self._bins_spin.setValue(30)
        col.addLayout(_labelled("Bins:", self._bins_spin))

        # One group-by checkbox per supplied group column (class_label included).
        self._group_checks: dict[str, QCheckBox] = {}
        group_row = QHBoxLayout()
        group_row.setContentsMargins(0, 0, 0, 0)
        group_row.addWidget(QLabel("Group by:"))
        for name in self._group_columns:
            check = QCheckBox(name)
            self._group_checks[name] = check
            group_row.addWidget(check)
        group_row.addStretch(1)
        col.addLayout(group_row)

        for combo in (self._value_combo, self._level_combo, self._plot_combo,
                      self._stat_combo, self._error_combo):
            combo.currentIndexChanged.connect(self._render)
        self._bins_spin.valueChanged.connect(self._render)
        for check in self._group_checks.values():
            check.toggled.connect(self._render)
        return body

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

        self._title_edit = QLineEdit()
        self._title_edit.setPlaceholderText("auto")
        col.addLayout(_labelled("Title:", self._title_edit))
        self._xlabel_edit = QLineEdit()
        self._xlabel_edit.setPlaceholderText("auto")
        col.addLayout(_labelled("X label:", self._xlabel_edit))
        self._ylabel_edit = QLineEdit()
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
        toggles = QHBoxLayout()
        toggles.setContentsMargins(0, 0, 0, 0)
        toggles.addWidget(self._grid_cb)
        toggles.addWidget(self._legend_cb)
        toggles.addWidget(self._legend_loc_combo, 1)
        col.addLayout(toggles)

        # Box-plot knobs — they only bite when Plot=box, but stay visible (like
        # Bins for hist) rather than appearing and disappearing on plot change.
        self._box_whis_spin = _double_spin(0.0, 100.0, 1.5, step=0.5)
        self._box_fliers_cb = QCheckBox("Outliers")
        self._box_fliers_cb.setChecked(True)
        self._box_notch_cb = QCheckBox("Notch (CI)")
        box_row = QHBoxLayout()
        box_row.setContentsMargins(0, 0, 0, 0)
        box_row.addWidget(QLabel("Box whis ×IQR:"))
        box_row.addWidget(self._box_whis_spin, 1)
        box_row.addWidget(self._box_fliers_cb)
        box_row.addWidget(self._box_notch_cb)
        col.addLayout(box_row)

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
    def _build_exports(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._export_pooled_btn = _export_button("Pooled CSV", self._export_pooled)
        self._export_agg_btn = _export_button("Aggregated CSV", self._export_aggregated)
        self._export_fig_btn = _export_button("Figure", self._export_figure)
        for btn in (self._export_pooled_btn, self._export_agg_btn, self._export_fig_btn):
            row.addWidget(btn)
        enabled = not self._df.empty
        for btn in (self._export_pooled_btn, self._export_agg_btn, self._export_fig_btn):
            btn.setEnabled(enabled)
        return row

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
        toolbar = NavigationToolbar2QT(canvas, self)
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
        if self._selected_target is not None:
            self.load_requested.emit(self._selected_target)

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
            summary = aggregate(self._df, self.current_spec())
            self._status.setText(f"Wrote {write_csv(summary, path).name}.")

    def _export_figure(self) -> None:
        if self._canvas is None:
            return
        path = self._save_path("Export figure", "Images (*.png *.svg)")
        if path:
            self._canvas.figure.savefig(path)
            self._status.setText(f"Wrote {path.name}.")


# --------------------------------------------------------------- UI factories
def _combo(items) -> QComboBox:
    combo = QComboBox()
    for item in items:
        combo.addItem(item, item)
    return combo


def _double_spin(lo: float, hi: float, value: float, step: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setValue(value)
    return spin


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
    action_button(btn)
    btn.clicked.connect(slot)
    return btn


def _range_edit() -> QLineEdit:
    edit = QLineEdit()
    edit.setPlaceholderText("auto")
    return edit


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
