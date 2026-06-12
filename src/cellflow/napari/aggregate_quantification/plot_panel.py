"""Generic, detached plotting surface for Aggregate Quantification.

:class:`PlotPanel` is the whole plotting experience: analytical controls that
build a :class:`~cellflow.aggregate_quantification.plotting.PlotSpec`, styling
controls that build a
:class:`~cellflow.aggregate_quantification.plotting.StyleSpec`, an embedded
matplotlib canvas with its native navigation toolbar, and CSV / figure export.

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

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.plotting import (
    _PICK_ROWS_ATTR,
    PlotSpec,
    StyleSpec,
    build_figure,
    plot_options,
    plotted_table,
    style_from_dict,
    style_to_dict,
    summary_table,
    write_csv,
)
from cellflow.napari.aggregate_quantification._mpl_toolbar import theme_toolbar_icons
from cellflow.napari.ui_style import action_button, status_label
from cellflow.napari.widgets import CollapsibleSection

#: Identity columns carried by a pooled table; the selection payload is these for
#: the picked rows (``frame_start`` lets per-track points report their start
#: frame). Only the ones present in the snapshot are used.
#:
#: ``date`` is part of the payload because ``position_id`` (the catalogue ``id``)
#: is reused across experiments — ``pos00`` exists on every date — so it is *not*
#: a unique record key. ``(date, position_id)`` is, and click-to-load keys on the
#: pair (see ``ClickToLoad.resolver``); without ``date`` a picked point resolves
#: to whatever same-named position landed last in the resolver's dict, loading the
#: wrong experiment's movie.
_IDENTITY_COLUMNS = ("date", "position_id", "frame", "frame_start", "cell_id")

#: Press→release pixel travel under which a click counts as a *selection* rather
#: than a navigation drag (the toolbar's zoom/pan rubber-band moves much further),
#: so the user can zoom and still click points without toggling the zoom tool off.
_CLICK_SLOP_PX = 4
#: Max pixel distance from the click to the nearest plotted point for it to count
#: as a hit. Generous on purpose: a click *near* a marker selects the nearest one
#: (no dead points), and the search runs in live display coordinates so it stays
#: exact at any zoom.
_PICK_TOLERANCE_PX = 24


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
    target_resolver: Callable[[dict], LoadTarget | None] | None = None
    #: Plot type this value renders most naturally (``box`` / ``bar`` /
    #: ``potential`` / …). When the picker swaps to a *different product*, the
    #: panel jumps to this rendering; ``""`` keeps the current plot type.
    suggested_plot: str = ""
    #: Open this source with adaptive (sinh-spaced) bins — the potential
    #: landscape's default so sparse tails do not read as spurious wells.
    adaptive: bool = False


#: level key → the pooled column whose presence makes that level meaningful, and
#: a short human tag. A level is offered only when its entity column exists (you
#: cannot aggregate "per cell" a table that carries no ``cell_id``). The tags label
#: the Level selector and the stats read-out — the *only* place the level appears
#: now (it used to also be baked into every value name, which duplicated it).
_LEVEL_ENTITY = {"cell": "cell_id", "position": "position_id", "date": "date"}
_LEVEL_LABELS = {"cell": "per cell", "position": "per position"}
#: Skip building filter checkboxes for a column with more distinct values than
#: this — a long checkbox wall (e.g. hundreds of position ids) helps no one.
_FILTER_MAX_VALUES = 40

_PLOT_TYPES = ("hist", "box", "violin", "strip", "swarm", "bar", "line", "potential")
#: Named qualitative palettes offered for the group colors (seaborn names).
_PALETTES = ("tab10", "Set1", "Set2", "Dark2", "Paired", "colorblind", "muted", "deep")
_LEGEND_LOCS = ("best", "upper right", "upper left", "lower right", "lower left", "center")
#: Style sheets offered, filtered to those the installed matplotlib provides.
_STYLE_CANDIDATES = (
    "default", "ggplot", "bmh", "fivethirtyeight", "grayscale",
    "seaborn-v0_8", "seaborn-v0_8-darkgrid", "seaborn-v0_8-whitegrid",
)
#: Font families offered; ``default`` keeps the style sheet's own (→ ``""``).
_FONT_FAMILIES = ("default", "sans-serif", "serif", "monospace")
_TICK_DIRECTIONS = ("out", "in", "inout")
_GRID_AXES = ("both", "x", "y")
_LINESTYLES = ("-", "--", ":", "-.")
_VIOLIN_INNERS = ("box", "quartile", "point", "stick", "none")
_HIST_ELEMENTS = ("bars", "step", "poly")
#: The four axis spines, in (checkbox label, matplotlib side) order.
_SPINE_SIDES = ("left", "bottom", "top", "right")


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
            first = value_catalog[0]
            dataframe = first.df
            group_columns = first.group_columns
            # No single button-level default any more: the opening plot type /
            # adaptive-bins come from the first value's own suggestion.
            default_plot = default_plot or first.suggested_plot
            default_adaptive_bins = default_adaptive_bins or first.adaptive
        if dataframe is None:
            raise ValueError("PlotPanel needs a dataframe or a non-empty value_catalog")
        self._df = dataframe
        #: The (filter-narrowed) DataFrame the *current* figure was drawn from —
        #: what picking, stats, and CSV export read so they always match the plot.
        #: Reset on every render; equals ``_df`` when no filter is active.
        self._plot_df = dataframe
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
        #: Pixel (x, y) where the current left-press began, or None. Lets the
        #: release handler tell a select-click from a zoom/pan drag.
        self._press_xy: tuple[float, float] | None = None
        #: Marker artist ringing the currently picked point, or None when nothing
        #: is selected. Removed and redrawn on each pick; cleared on re-render.
        self._pick_marker = None
        #: Latched True once a swarm plot overflowed at the *displayed* canvas size
        #: (seaborn can't place every marker); the panel then renders it as a strip
        #: so no point is dropped. Reset on any control change (each re-render).
        self._swarm_overflowed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        layout.addWidget(CollapsibleSection("Plot", self._build_analytical(), expanded=True))
        layout.addWidget(CollapsibleSection("Filter", self._build_filters(), expanded=False))
        layout.addWidget(CollapsibleSection("Styling", self._build_styling(), expanded=False))

        # The canvas + its toolbar live in their own container so the whole plot
        # can pop out into a floating window — once every styling tab is open the
        # controls crowd the plot. ``_canvas_holder`` is the container's layout, so
        # a re-render targets it whether the container is docked here or floating.
        self._main_layout = layout
        self._canvas_holder = QVBoxLayout()
        self._canvas_holder.setContentsMargins(0, 0, 0, 0)
        self._plot_container = QWidget()
        container_col = QVBoxLayout(self._plot_container)
        container_col.setContentsMargins(0, 0, 0, 0)
        container_col.setSpacing(2)
        self._detach_btn = QPushButton("⧉ Detach plot")
        self._detach_btn.setToolTip(
            "Pop the plot out into its own window to free up room for the controls."
        )
        # Keep its natural width (no ``_shrinkable`` here): it is narrow, so it
        # never floors the panel width, and an Ignored policy next to the row's
        # stretch would collapse it to nothing.
        self._detach_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._detach_btn.clicked.connect(self._toggle_detach)
        detach_row = QHBoxLayout()
        detach_row.setContentsMargins(0, 0, 0, 0)
        detach_row.addStretch(1)
        detach_row.addWidget(self._detach_btn)
        container_col.addLayout(detach_row)
        container_col.addLayout(self._canvas_holder, 1)
        layout.addWidget(self._plot_container, 1)
        self._plot_index = layout.indexOf(self._plot_container)
        # State for the popped-out window; the placeholder fills the docked slot
        # while the plot floats so the controls don't jump.
        self._detached_window: QWidget | None = None
        self._detach_placeholder = QLabel(
            "Plot detached into its own window. Close that window or click "
            "“⧉ Re-attach plot” to dock it again."
        )
        self._detach_placeholder.setWordWrap(True)
        self._detach_placeholder.setAlignment(Qt.AlignCenter)
        status_label(self._detach_placeholder, muted=True)
        self._detach_placeholder.hide()

        # Numeric summary of exactly what the figure draws (n / mean / median / sd /
        # sem / min / max per group), so the headline numbers are legible without
        # eyeballing the plot. Rebuilt on every render from the same ``_plot_df``.
        self._stats_label = QLabel()
        self._stats_label.setTextFormat(Qt.RichText)
        self._stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._stats_label.setWordWrap(True)
        status_label(self._stats_label, muted=True)
        layout.addWidget(self._stats_label)

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

        # Show the option set for the (possibly caller-chosen) plot type, then draw.
        self._sync_plot_options()
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
        col.addLayout(_labelled("Stat:", self._stat_combo))

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

        # Plot-specific controls (bins, box knobs, markers …) live in one box that
        # shows only the options the chosen plot supports — see the capability map
        # ``plot_options`` and ``_sync_plot_options``.
        col.addWidget(self._build_plot_options())

        # The value combo may swap the active product (catalog mode), and the plot
        # combo re-syncs which options show; the rest are pure re-renders.
        self._value_combo.currentIndexChanged.connect(self._on_value_changed)
        self._plot_combo.currentIndexChanged.connect(self._on_plot_changed)
        for combo in (self._level_combo, self._stat_combo):
            combo.currentIndexChanged.connect(self._render)
        # Offer only the levels the active table can actually aggregate to.
        self._sync_levels()
        return body

    # ------------------------------------------------------------ plot options
    def _build_plot_options(self) -> QWidget:
        """Every plot-specific control, built once; ``_sync_plot_options`` shows
        exactly those the current plot type supports and hides the rest."""
        box = QWidget()
        col = QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        #: option key → its row widget, toggled visible per plot type.
        self._opt_rows: dict[str, QWidget] = {}

        self._bins_spin = _shrinkable(QSpinBox())
        self._bins_spin.setRange(2, 200)
        self._bins_spin.setValue(30)
        self._add_opt(col, "bins", _labelled("Bins:", self._bins_spin))

        self._adaptive_bins_cb = QCheckBox("Tighter bins near 0")
        self._adaptive_bins_cb.setToolTip(
            "Potential view only: sinh-spaced bins, narrowest at x=0, to resolve "
            "the barrier at the transition state (junction length → 0)."
        )
        self._add_opt(col, "adaptive_bins", self._adaptive_bins_cb)

        self._error_combo = _combo(("sd", "sem", "none"))
        self._add_opt(col, "error", _labelled("Error:", self._error_combo))

        self._box_whis_spin = _double_spin(0.0, 100.0, 1.5, step=0.5)
        self._add_opt(col, "box_whis", _labelled("Box whis ×IQR:", self._box_whis_spin))
        self._box_fliers_cb = QCheckBox("Outliers")
        self._box_fliers_cb.setChecked(True)
        self._add_opt(col, "box_showfliers", self._box_fliers_cb)
        self._box_notch_cb = QCheckBox("Notch (CI)")
        self._add_opt(col, "box_notch", self._box_notch_cb)

        self._violin_inner_combo = _combo(_VIOLIN_INNERS)
        self._add_opt(col, "violin_inner", _labelled("Violin inner:", self._violin_inner_combo))

        self._hist_element_combo = _combo(_HIST_ELEMENTS)
        self._add_opt(col, "hist_element", _labelled("Hist element:", self._hist_element_combo))
        self._hist_cumulative_cb = QCheckBox("Cumulative")
        self._add_opt(col, "hist_cumulative", self._hist_cumulative_cb)

        self._markers_cb = QCheckBox("Markers")
        self._markers_cb.setChecked(True)
        self._add_opt(col, "markers", self._markers_cb)
        self._marker_size_spin = _auto_double_spin(0.0, 30.0, step=1.0)
        self._add_opt(col, "marker_size", _labelled("Marker size:", self._marker_size_spin))

        self._bins_spin.valueChanged.connect(self._render)
        self._error_combo.currentIndexChanged.connect(self._render)
        self._box_whis_spin.valueChanged.connect(self._render)
        self._marker_size_spin.valueChanged.connect(self._render)
        for cb in (self._adaptive_bins_cb, self._box_fliers_cb, self._box_notch_cb,
                   self._hist_cumulative_cb, self._markers_cb):
            cb.toggled.connect(self._render)
        for combo in (self._violin_inner_combo, self._hist_element_combo):
            combo.currentIndexChanged.connect(self._render)
        self._plot_opts_box = box
        return box

    def _add_opt(self, parent: QVBoxLayout, key: str, item) -> None:
        """Wrap *item* (a widget or a layout) in a holder, register it under *key*
        for show/hide, and add it to *parent*."""
        holder = QWidget()
        if isinstance(item, QWidget):
            inner = QVBoxLayout(holder)
            inner.setContentsMargins(0, 0, 0, 0)
            inner.addWidget(item)
        else:
            holder.setLayout(item)
        parent.addWidget(holder)
        self._opt_rows[key] = holder

    def _sync_plot_options(self) -> None:
        """Show the options the current plot supports; hide the box when none."""
        active = set(plot_options(self._plot_combo.currentData()))
        for key, row in self._opt_rows.items():
            row.setVisible(key in active)
        self._plot_opts_box.setVisible(bool(active))

    def _on_plot_changed(self) -> None:
        self._sync_plot_options()
        self._render()

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
        """Repopulate the group-by checkboxes for the current ``group_columns``.

        Changing the Value picker (catalog mode) swaps the active product and
        rebuilds these checkboxes; carry over which columns were ticked so the
        user's grouping isn't reset just because they switched quantity — a
        column the new product still offers stays checked."""
        previously_checked = {
            name for name, check in self._group_checks.items() if check.isChecked()
        }
        while self._group_grid.count():
            item = self._group_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._group_checks = {}
        for i, name in enumerate(self._group_columns):
            check = QCheckBox(name)
            if name in previously_checked:
                check.setChecked(True)
            self._group_checks[name] = check
            check.toggled.connect(self._on_group_changed)
            self._group_grid.addWidget(check, i // 2, i % 2)

    def _on_group_changed(self) -> None:
        """A group-by toggle changes which series exist — refresh the per-group
        colour swatches before re-rendering."""
        self._rebuild_color_overrides()
        self._render()

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
                self._rebuild_filters()
                self._sync_levels()
                # Jump to the new product's natural rendering (only on a product
                # switch — flipping between a product's own values keeps your plot).
                self._apply_suggested_plot(src)
        self._rebuild_color_overrides()
        self._render()

    def _apply_suggested_plot(self, src: ValueSource) -> None:
        """Set the plot type + adaptive bins to *src*'s suggestion, silently (the
        single ``_render`` in the caller does the one redraw)."""
        if src.suggested_plot:
            index = self._plot_combo.findData(src.suggested_plot)
            if index >= 0:
                blocked = self._plot_combo.blockSignals(True)
                self._plot_combo.setCurrentIndex(index)
                self._plot_combo.blockSignals(blocked)
                self._sync_plot_options()
        blocked = self._adaptive_bins_cb.blockSignals(True)
        self._adaptive_bins_cb.setChecked(src.adaptive)
        self._adaptive_bins_cb.blockSignals(blocked)

    # ---------------------------------------------------------------- filtering
    def _build_filters(self) -> QWidget:
        """The Filter section: a checkbox per distinct value of each categorical
        group axis, so the user can restrict the plot to specific catalogue
        elements (a condition, a date, a position, a subpopulation). All ticked =
        no filter; unticking a value drops its rows from a pure re-render."""
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        hint = QLabel("Show only the ticked catalogue elements. All ticked = no filter.")
        hint.setWordWrap(True)
        status_label(hint, muted=True)
        col.addWidget(hint)
        #: column -> {value(str) -> checkbox}. Rebuilt when the product swaps.
        self._filter_checks: dict[str, dict[str, QCheckBox]] = {}
        self._filter_body = QVBoxLayout()
        self._filter_body.setContentsMargins(0, 0, 0, 0)
        self._filter_body.setSpacing(4)
        col.addLayout(self._filter_body)
        self._rebuild_filters()
        return body

    def _rebuild_filters(self) -> None:
        """Repopulate the filter checkboxes for the active table, carrying over
        which values were *un*ticked for a column the new product still offers."""
        if not hasattr(self, "_filter_body"):
            return  # filter section not built yet
        unticked = {
            (col, val)
            for col, checks in self._filter_checks.items()
            for val, cb in checks.items()
            if not cb.isChecked()
        }
        while self._filter_body.count():
            item = self._filter_body.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                _clear_layout(item.layout())
        self._filter_checks = {}
        for col in self._group_columns:
            if col == "frame" or col not in self._df.columns:
                continue  # ``frame`` is a continuous axis, not a categorical filter
            values = sorted(str(v) for v in pd.unique(self._df[col]))
            if len(values) < 2 or len(values) > _FILTER_MAX_VALUES:
                continue  # nothing to filter, or too many to show as checkboxes
            self._filter_body.addWidget(QLabel(f"{col}:"))
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(4)
            grid.setColumnStretch(2, 1)
            checks: dict[str, QCheckBox] = {}
            for i, val in enumerate(values):
                cb = QCheckBox(val)
                cb.setChecked((col, val) not in unticked)
                cb.toggled.connect(self._render)
                checks[val] = cb
                grid.addWidget(cb, i // 2, i % 2)
            self._filter_body.addLayout(grid)
            self._filter_checks[col] = checks
        if not self._filter_checks:
            none = QLabel("No filterable columns for this quantity.")
            status_label(none, muted=True)
            self._filter_body.addWidget(none)

    def _render_df(self) -> pd.DataFrame:
        """The snapshot narrowed to the ticked filter values — what the current
        figure draws. A fully-ticked column does not filter; the result is
        re-indexed so the positional pick-row stamping stays aligned."""
        masks = []
        for col, checks in getattr(self, "_filter_checks", {}).items():
            allowed = {val for val, cb in checks.items() if cb.isChecked()}
            if len(allowed) == len(checks):
                continue  # all ticked → no constraint from this column
            masks.append(self._df[col].astype(str).isin(allowed))
        if not masks:
            return self._df
        mask = masks[0]
        for extra in masks[1:]:
            mask &= extra
        return self._df[mask].reset_index(drop=True)

    # ------------------------------------------------------------- data levels
    def _sync_levels(self) -> None:
        """Enable only the Level options the active table supports, and bump the
        current selection off a now-invalid one.

        A level needs its entity column (``cell`` → ``cell_id`` …): a per-tissue
        table with no ``cell_id`` can't be aggregated "per cell". The current
        choice, if disabled, drops to the first enabled (finest) level."""
        columns = set(self._df.columns)
        model = self._level_combo.model()
        first_enabled = -1
        for i in range(self._level_combo.count()):
            level = self._level_combo.itemData(i)
            enabled = _LEVEL_ENTITY[level] in columns
            model.item(i).setEnabled(enabled)
            if enabled and first_enabled < 0:
                first_enabled = i
        current = self._level_combo.currentIndex()
        if first_enabled >= 0 and not model.item(current).isEnabled():
            blocked = self._level_combo.blockSignals(True)
            self._level_combo.setCurrentIndex(first_enabled)
            self._level_combo.blockSignals(blocked)

    # -------------------------------------------------------------- styling UI
    def _build_styling(self) -> QWidget:
        """The Style editor: a tab per concern (Figure / Axes / Colors / Legend /
        Grid) over the one :class:`StyleSpec`. Every general style knob applies to
        every plot type, so these tabs never gate on the plot (unlike the
        plot-specific options in the Plot section)."""
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        tabs = QTabWidget()
        _shrinkable(tabs)
        tabs.addTab(self._build_figure_tab(), "Figure")
        tabs.addTab(self._build_axes_tab(), "Axes")
        tabs.addTab(self._build_colors_tab(), "Colors")
        tabs.addTab(self._build_legend_tab(), "Legend")
        tabs.addTab(self._build_grid_tab(), "Grid")
        outer.addWidget(tabs)

        # Save / load the whole Style as a reusable theme (JSON) — a house style a
        # lab can share and re-apply to any plot.
        save_btn = QPushButton("Save style…")
        save_btn.setToolTip("Save the current styling as a reusable theme (JSON).")
        save_btn.clicked.connect(self._save_style)
        load_btn = QPushButton("Load style…")
        load_btn.setToolTip("Load a styling theme (JSON) and apply it to this plot.")
        load_btn.clicked.connect(self._load_style)
        for btn in (save_btn, load_btn):
            _shrinkable(btn)
        outer.addLayout(_checkrow(save_btn, load_btn))
        return container

    @staticmethod
    def _tab_body() -> tuple[QWidget, QVBoxLayout]:
        body = QWidget()
        col = QVBoxLayout(body)
        col.setContentsMargins(6, 6, 6, 6)
        col.setSpacing(4)
        return body, col

    def _build_figure_tab(self) -> QWidget:
        body, col = self._tab_body()
        styles = [s for s in _STYLE_CANDIDATES if s == "default" or s in mpl.style.available]
        self._style_combo = _combo(styles)
        col.addLayout(_labelled("Style:", self._style_combo))
        self._title_edit = _line_edit("auto")
        col.addLayout(_labelled("Title:", self._title_edit))
        self._width_spin = _double_spin(2.0, 30.0, 6.0, step=0.5)
        self._height_spin = _double_spin(2.0, 30.0, 4.0, step=0.5)
        col.addLayout(_pair_row("W×H (in):", self._width_spin, self._height_spin))
        self._dpi_spin = _shrinkable(QSpinBox())
        self._dpi_spin.setRange(50, 600)
        self._dpi_spin.setValue(100)
        col.addLayout(_labelled("DPI:", self._dpi_spin))
        self._font_family_combo = _combo(_FONT_FAMILIES)
        self._font_spin = _double_spin(4.0, 40.0, 10.0, step=1.0)
        col.addLayout(_pair_row("Font:", self._font_family_combo, self._font_spin))
        self._facecolor_edit = _line_edit("auto")
        col.addLayout(_labelled("Background:", self._facecolor_edit))
        col.addStretch(1)

        self._style_combo.currentIndexChanged.connect(self._render)
        self._font_family_combo.currentIndexChanged.connect(self._render)
        for spin in (self._width_spin, self._height_spin, self._font_spin, self._dpi_spin):
            spin.valueChanged.connect(self._render)
        for edit in (self._title_edit, self._facecolor_edit):
            edit.editingFinished.connect(self._render)
        return body

    def _build_axes_tab(self) -> QWidget:
        body, col = self._tab_body()
        self._xlabel_edit = _line_edit("auto")
        col.addLayout(_labelled("X label:", self._xlabel_edit))
        self._ylabel_edit = _line_edit("auto")
        col.addLayout(_labelled("Y label:", self._ylabel_edit))
        # Axis-range overrides; blank ("auto") keeps matplotlib's autoscale.
        self._xmin_edit = _range_edit()
        self._xmax_edit = _range_edit()
        col.addLayout(_pair_row("X range:", self._xmin_edit, self._xmax_edit))
        self._ymin_edit = _range_edit()
        self._ymax_edit = _range_edit()
        col.addLayout(_pair_row("Y range:", self._ymin_edit, self._ymax_edit))
        self._xlog_cb = QCheckBox("Log X")
        self._ylog_cb = QCheckBox("Log Y")
        col.addLayout(_checkrow(self._xlog_cb, self._ylog_cb))
        self._tick_size_spin = _auto_double_spin(0.0, 40.0, step=1.0)
        col.addLayout(_labelled("Tick size:", self._tick_size_spin))
        self._tick_length_spin = _double_spin(0.0, 20.0, 3.5, step=0.5)
        col.addLayout(_labelled("Tick len:", self._tick_length_spin))
        self._tick_dir_combo = _combo(_TICK_DIRECTIONS)
        col.addLayout(_labelled("Tick dir:", self._tick_dir_combo))
        # -1 → "auto" (each plot keeps its own default x-label rotation).
        self._xrot_spin = _shrinkable(QSpinBox())
        self._xrot_spin.setRange(-1, 90)
        self._xrot_spin.setValue(-1)
        self._xrot_spin.setSpecialValueText("auto")
        col.addLayout(_labelled("X rotation:", self._xrot_spin))

        col.addWidget(QLabel("Borders:"))
        self._spine_checks: dict[str, QCheckBox] = {}
        spine_grid = QGridLayout()
        spine_grid.setContentsMargins(0, 0, 0, 0)
        spine_grid.setSpacing(4)
        spine_grid.setColumnStretch(2, 1)
        for i, side in enumerate(_SPINE_SIDES):
            cb = QCheckBox(side)
            cb.setChecked(True)
            cb.toggled.connect(self._render)
            self._spine_checks[side] = cb
            spine_grid.addWidget(cb, i // 2, i % 2)
        col.addLayout(spine_grid)
        self._spine_width_spin = _double_spin(0.0, 6.0, 0.8, step=0.1)
        col.addLayout(_labelled("Border w:", self._spine_width_spin))
        col.addStretch(1)

        self._tick_dir_combo.currentIndexChanged.connect(self._render)
        for cb in (self._xlog_cb, self._ylog_cb):
            cb.toggled.connect(self._render)
        for edit in (self._xlabel_edit, self._ylabel_edit, self._xmin_edit,
                     self._xmax_edit, self._ymin_edit, self._ymax_edit):
            edit.editingFinished.connect(self._render)
        for spin in (self._tick_size_spin, self._tick_length_spin,
                     self._xrot_spin, self._spine_width_spin):
            spin.valueChanged.connect(self._render)
        return body

    def _build_colors_tab(self) -> QWidget:
        body, col = self._tab_body()
        self._palette_combo = _combo(_PALETTES)
        col.addLayout(_labelled("Palette:", self._palette_combo))
        self._alpha_spin = _auto_double_spin(0.0, 1.0, step=0.05)
        col.addLayout(_labelled("Opacity:", self._alpha_spin))
        self._edge_edit = _line_edit("auto")
        col.addLayout(_labelled("Edge:", self._edge_edit))
        self._line_width_spin = _auto_double_spin(0.0, 10.0, step=0.5)
        col.addLayout(_labelled("Line w:", self._line_width_spin))

        col.addWidget(QLabel("Group colours:"))
        self._overrides_grid = QGridLayout()
        self._overrides_grid.setContentsMargins(0, 0, 0, 0)
        self._overrides_grid.setSpacing(4)
        self._overrides_grid.setColumnStretch(0, 1)
        self._override_buttons: dict[str, _ColorButton] = {}
        col.addLayout(self._overrides_grid)
        reset = QPushButton("Reset group colours")
        _shrinkable(reset)
        reset.clicked.connect(self._reset_overrides)
        col.addWidget(reset)
        self._rebuild_color_overrides()
        col.addStretch(1)

        self._palette_combo.currentIndexChanged.connect(self._render)
        self._alpha_spin.valueChanged.connect(self._render)
        self._line_width_spin.valueChanged.connect(self._render)
        self._edge_edit.editingFinished.connect(self._render)
        return body

    def _build_legend_tab(self) -> QWidget:
        body, col = self._tab_body()
        self._legend_cb = QCheckBox("Show legend")
        self._legend_cb.setChecked(True)
        col.addWidget(self._legend_cb)
        self._legend_loc_combo = _combo(_LEGEND_LOCS)
        col.addLayout(_labelled("Location:", self._legend_loc_combo))
        self._legend_title_edit = _line_edit("auto")
        col.addLayout(_labelled("Title:", self._legend_title_edit))
        self._legend_frame_cb = QCheckBox("Frame")
        self._legend_frame_cb.setChecked(True)
        col.addWidget(self._legend_frame_cb)
        self._legend_ncol_spin = _shrinkable(QSpinBox())
        self._legend_ncol_spin.setRange(1, 6)
        col.addLayout(_labelled("Columns:", self._legend_ncol_spin))
        col.addStretch(1)

        for cb in (self._legend_cb, self._legend_frame_cb):
            cb.toggled.connect(self._render)
        self._legend_loc_combo.currentIndexChanged.connect(self._render)
        self._legend_title_edit.editingFinished.connect(self._render)
        self._legend_ncol_spin.valueChanged.connect(self._render)
        return body

    def _build_grid_tab(self) -> QWidget:
        body, col = self._tab_body()
        self._grid_cb = QCheckBox("Show grid")
        col.addWidget(self._grid_cb)
        self._grid_axis_combo = _combo(_GRID_AXES)
        col.addLayout(_labelled("Axis:", self._grid_axis_combo))
        self._grid_alpha_spin = _double_spin(0.0, 1.0, 1.0, step=0.05)
        col.addLayout(_labelled("Opacity:", self._grid_alpha_spin))
        self._grid_ls_combo = _combo(_LINESTYLES)
        col.addLayout(_labelled("Line:", self._grid_ls_combo))
        col.addStretch(1)

        self._grid_cb.toggled.connect(self._render)
        self._grid_axis_combo.currentIndexChanged.connect(self._render)
        self._grid_ls_combo.currentIndexChanged.connect(self._render)
        self._grid_alpha_spin.valueChanged.connect(self._render)
        return body

    # ----------------------------------------------------- per-group colours
    def _current_group_labels(self) -> list[str]:
        """The group labels the current plot would draw (``" · "``-joined), so the
        Colours tab offers one swatch per series. Empty with no group-by."""
        group = [name for name, cb in self._group_checks.items() if cb.isChecked()]
        present = [g for g in group if g in self._df.columns]
        if not present or self._df.empty:
            return []
        combos = self._df[present].drop_duplicates()
        return [" · ".join(str(row[g]) for g in present) for _, row in combos.iterrows()]

    def _rebuild_color_overrides(self) -> None:
        """Repopulate per-group colour swatches for the current grouping, carrying
        over colours already chosen for labels that still exist."""
        if not hasattr(self, "_overrides_grid"):
            return  # styling tab not built yet (first call during construction)
        kept = {lbl: btn.color() for lbl, btn in self._override_buttons.items()}
        while self._overrides_grid.count():
            item = self._overrides_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._override_buttons = {}
        labels = self._current_group_labels()
        if not labels:
            hint = QLabel("Add a Group-by to colour series.")
            status_label(hint, muted=True)
            self._overrides_grid.addWidget(hint, 0, 0, 1, 2)
            return
        for i, label in enumerate(labels):
            name = QLabel(label)
            name.setWordWrap(True)
            button = _ColorButton()
            if kept.get(label):
                button.set_color(kept[label])  # before connect → no spurious render
            button.changed.connect(self._render)
            self._override_buttons[label] = button
            self._overrides_grid.addWidget(name, i, 0)
            self._overrides_grid.addWidget(button, i, 1)

    def _reset_overrides(self) -> None:
        for button in self._override_buttons.values():
            blocked = button.blockSignals(True)
            button.set_color("")
            button.blockSignals(blocked)
        self._render()

    # --------------------------------------------------------------- export UI
    def _build_exports(self) -> QGridLayout:
        # One CSV export that writes exactly the data the current plot shows
        # (so the file reproduces the figure with no further filtering) plus a
        # figure export. Equal-width buttons on an "Export:" row that collapse
        # together as the panel narrows.
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        grid.addWidget(QLabel("Export:"), 0, 0)
        self._export_csv_btn = _export_button("Plot data (CSV)", self._export_csv)
        self._export_csv_btn.setToolTip(
            "Write the exact data drawn in the current plot to CSV — the points / "
            "curve / bars shown, ready to re-plot elsewhere."
        )
        self._export_fig_btn = _export_button("Figure", self._export_figure)
        self._export_fig_btn.setToolTip(
            "Write the current figure to a raster (PNG) or vector (SVG/PDF) file, "
            "styled exactly as shown — vector formats stay editable in Illustrator/"
            "Inkscape for publication."
        )
        buttons = (self._export_csv_btn, self._export_fig_btn)
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
            dpi=self._dpi_spin.value(),
            font_family=_combo_opt(self._font_family_combo, "default"),
            facecolor=self._facecolor_edit.text().strip(),
            xlog=self._xlog_cb.isChecked(),
            ylog=self._ylog_cb.isChecked(),
            tick_label_size=_auto_value(self._tick_size_spin),
            tick_length=self._tick_length_spin.value(),
            tick_direction=self._tick_dir_combo.currentData(),
            xtick_rotation=(None if self._xrot_spin.value() < 0
                            else float(self._xrot_spin.value())),
            spines=tuple(s for s in _SPINE_SIDES if self._spine_checks[s].isChecked()),
            spine_width=self._spine_width_spin.value(),
            color_overrides=tuple(
                (lbl, btn.color()) for lbl, btn in self._override_buttons.items() if btn.color()
            ),
            alpha=_auto_value(self._alpha_spin),
            edge_color=self._edge_edit.text().strip(),
            line_width=_auto_value(self._line_width_spin),
            legend_title=self._legend_title_edit.text().strip(),
            legend_frame=self._legend_frame_cb.isChecked(),
            legend_ncol=self._legend_ncol_spin.value(),
            grid_axis=self._grid_axis_combo.currentData(),
            grid_alpha=self._grid_alpha_spin.value(),
            grid_linestyle=self._grid_ls_combo.currentData(),
            markers=self._markers_cb.isChecked(),
            marker_size=_auto_value(self._marker_size_spin),
            violin_inner=self._violin_inner_combo.currentData(),
            hist_element=self._hist_element_combo.currentData(),
            hist_cumulative=self._hist_cumulative_cb.isChecked(),
        )

    # ----------------------------------------------------------- style themes
    def _save_style(self) -> None:
        path = self._save_path("Save style theme", "JSON files (*.json)")
        if not path:
            return
        if path.suffix.lower() != ".json":
            path = path.with_name(path.name + ".json")
        path.write_text(json.dumps(style_to_dict(self.current_style()), indent=2))
        self._status.setText(f"Saved style to {path.name}.")

    def _load_style(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load style theme", filter="JSON files (*.json)"
        )
        if not path:
            return
        try:
            style = style_from_dict(json.loads(Path(path).read_text()))
        except (OSError, ValueError) as exc:
            self._status.setText(f"Could not load style: {exc}")
            return
        self._apply_style_to_widgets(style)
        self._status.setText(f"Loaded style from {Path(path).name}.")

    def _apply_style_to_widgets(self, style: StyleSpec) -> None:
        """Drive every styling control from *style* (the inverse of
        :meth:`current_style`), then re-render once.

        Suppresses the per-widget re-render while the batch runs so the dozens of
        ``setValue``/``setChecked`` calls don't each redraw the canvas."""
        self._suppress_render = True
        try:
            _set_combo(self._palette_combo, style.palette)
            _set_combo(self._style_combo, style.style)
            self._title_edit.setText(style.title)
            self._xlabel_edit.setText(style.xlabel)
            self._ylabel_edit.setText(style.ylabel)
            self._width_spin.setValue(style.width)
            self._height_spin.setValue(style.height)
            self._dpi_spin.setValue(style.dpi)
            _set_combo(self._font_family_combo, style.font_family or "default")
            self._font_spin.setValue(style.font_size)
            self._facecolor_edit.setText(style.facecolor)
            _set_range_edit(self._xmin_edit, style.xmin)
            _set_range_edit(self._xmax_edit, style.xmax)
            _set_range_edit(self._ymin_edit, style.ymin)
            _set_range_edit(self._ymax_edit, style.ymax)
            self._xlog_cb.setChecked(style.xlog)
            self._ylog_cb.setChecked(style.ylog)
            _set_auto_spin(self._tick_size_spin, style.tick_label_size)
            self._tick_length_spin.setValue(style.tick_length)
            _set_combo(self._tick_dir_combo, style.tick_direction)
            self._xrot_spin.setValue(-1 if style.xtick_rotation is None
                                     else int(style.xtick_rotation))
            for side, check in self._spine_checks.items():
                check.setChecked(side in style.spines)
            self._spine_width_spin.setValue(style.spine_width)
            _set_auto_spin(self._alpha_spin, style.alpha)
            self._edge_edit.setText(style.edge_color)
            _set_auto_spin(self._line_width_spin, style.line_width)
            self._grid_cb.setChecked(style.grid)
            _set_combo(self._grid_axis_combo, style.grid_axis)
            self._grid_alpha_spin.setValue(style.grid_alpha)
            _set_combo(self._grid_ls_combo, style.grid_linestyle)
            self._legend_cb.setChecked(style.legend)
            _set_combo(self._legend_loc_combo, style.legend_loc)
            self._legend_title_edit.setText(style.legend_title)
            self._legend_frame_cb.setChecked(style.legend_frame)
            self._legend_ncol_spin.setValue(style.legend_ncol)
            self._box_whis_spin.setValue(style.box_whis)
            self._box_fliers_cb.setChecked(style.box_showfliers)
            self._box_notch_cb.setChecked(style.box_notch)
            self._markers_cb.setChecked(style.markers)
            _set_auto_spin(self._marker_size_spin, style.marker_size)
            _set_combo(self._violin_inner_combo, style.violin_inner)
            _set_combo(self._hist_element_combo, style.hist_element)
            self._hist_cumulative_cb.setChecked(style.hist_cumulative)
            # Colours: rebuild swatches for the current grouping, then apply any
            # saved colours whose label still exists.
            self._rebuild_color_overrides()
            saved = dict(style.color_overrides)
            for label, button in self._override_buttons.items():
                if label in saved:
                    button.set_color(saved[label])
        finally:
            self._suppress_render = False
        self._render()

    # ---------------------------------------------------------------- render
    def _render(self, *, _retry_after_overflow: bool = False) -> None:
        """Pure re-render from the held snapshot — never re-pools.

        Any control change re-enters here and clears the swarm-overflow latch, so
        a fresh configuration gets a fresh swarm attempt; only the internal retry
        after an overflow keeps the latch (and renders the swarm as a strip)."""
        # Loading a theme drives dozens of widgets in one go; suppress the per-widget
        # re-render and draw once when the batch finishes (see ``_apply_style``).
        if getattr(self, "_suppress_render", False):
            return
        if not _retry_after_overflow:
            self._swarm_overflowed = False
        spec = self.current_spec()
        # Draw (and pick / stat / export) from the filter-narrowed snapshot, so the
        # figure, the numbers under it, and the CSV all describe the same rows.
        self._plot_df = self._render_df()
        # A swarm that overflowed at the displayed size is drawn as a strip so no
        # point is dropped; everything else renders as selected.
        render_spec = (
            replace(spec, plot="strip")
            if spec.plot == "swarm" and self._swarm_overflowed
            else spec
        )
        fig = build_figure(self._plot_df, render_spec, self.current_style())
        self._render_stats(spec)
        canvas = FigureCanvasQTAgg(fig)
        _shrinkable(canvas)
        toolbar = NavigationToolbar2QT(canvas, self)
        # As a QToolBar it spills overflow tools into a "»" menu when squeezed,
        # so it stops being the panel's hard minimum width.
        _shrinkable(toolbar)
        # Lighten the glyphs on napari's dark themes (matplotlib leaves them black,
        # which all but vanishes on the dark toolbar). No-op on light themes.
        theme_toolbar_icons(toolbar)
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
        # Own the click instead of matplotlib's pick_event: pick_event fires only
        # within a few pixels of a marker (dense plots leave most points dead),
        # returns an arbitrary marker among those in range (not the nearest, so a
        # click on the tall point often resolved to a shorter neighbour), and is
        # suppressed entirely while the toolbar holds the widget-lock to zoom/pan.
        # A press/release pair with nearest-in-pixels hit-testing sidesteps all
        # three and stays exact after zooming.
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("button_release_event", self._on_release)
        self._press_xy = None
        self._clear_selection()
        # seaborn only reports a swarm overflow at draw time, and the Qt canvas
        # resizes the figure to the dock — so detect it once the canvas has its
        # real size (next event-loop tick), then fall back to a strip if needed.
        if spec.plot == "swarm" and not self._swarm_overflowed:
            QTimer.singleShot(0, self._check_swarm_overflow)

    def _check_swarm_overflow(self) -> None:
        """Force a draw at the displayed size; if the swarm couldn't place every
        marker, latch the overflow and re-render it as a strip."""
        if self._canvas is None or self._swarm_overflowed:
            return
        if self._plot_combo.currentData() != "swarm":
            return
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._canvas.draw()
        if any("cannot be placed" in str(w.message) for w in caught):
            self._swarm_overflowed = True
            self._render(_retry_after_overflow=True)

    # --------------------------------------------------------------- detach plot
    def _toggle_detach(self) -> None:
        if self._detached_window is None:
            self._detach_plot()
        else:
            self._attach_plot()

    def _detach_plot(self) -> None:
        """Move the plot container into a floating top-level window, leaving the
        placeholder in its docked slot so the controls keep their layout."""
        window = _DetachedPlotWindow(self._on_detached_closed, self)
        window.setWindowTitle(f"Plot — {self._value_combo.currentText().strip()}")
        col = QVBoxLayout(window)
        col.setContentsMargins(6, 6, 6, 6)
        self._main_layout.removeWidget(self._plot_container)
        col.addWidget(self._plot_container)
        self._main_layout.insertWidget(self._plot_index, self._detach_placeholder)
        self._detach_placeholder.show()
        self._detach_btn.setText("⧉ Re-attach plot")
        self._detached_window = window
        window.resize(680, 560)
        window.show()

    def _attach_plot(self) -> None:
        """Dock the plot container back in place and dispose of the float window."""
        window = self._detached_window
        if window is None:
            return
        # Clear the handle first so the window's own closeEvent doesn't re-enter.
        self._detached_window = None
        self._main_layout.removeWidget(self._detach_placeholder)
        self._detach_placeholder.hide()
        self._detach_placeholder.setParent(self)
        self._main_layout.insertWidget(self._plot_index, self._plot_container, 1)
        self._detach_btn.setText("⧉ Detach plot")
        window.close()
        window.deleteLater()

    def _on_detached_closed(self) -> None:
        """The user closed the floating plot window directly — re-dock the plot."""
        if self._detached_window is not None:
            self._attach_plot()

    # ----------------------------------------------------------------- stats
    def _render_stats(self, spec: PlotSpec) -> None:
        """Refresh the numeric summary under the plot from the current draw."""
        if not hasattr(self, "_stats_label"):
            return
        self._stats_label.setText(self._stats_html(summary_table(self._plot_df, spec), spec))

    def _stats_html(self, table: pd.DataFrame, spec: PlotSpec) -> str:
        """A compact HTML summary table (one row per group) plus a header naming
        the value, the unit level, and how many rows are in scope after filtering."""
        level = _LEVEL_LABELS.get(spec.level, spec.level)
        shown, total = len(self._plot_df), len(self._df)
        scope = f"{shown} of {total} rows" if shown != total else f"{total} rows"
        header = f"<b>Summary</b> — {spec.value} · {level} · {scope}"
        if table.empty:
            return f"{header}<br><i>no numeric data</i>"
        group = [c for c in spec.group_by if c in table.columns]
        stat_cols = ["n", "mean", "median", "sd", "sem", "min", "max"]
        head_cells = "".join(f"<th align='right'>{c}</th>" for c in (*group, *stat_cols))
        rows = []
        for _, r in table.iterrows():
            cells = "".join(f"<td align='left'>{r[g]}</td>" for g in group)
            cells += f"<td align='right'>{int(r['n'])}</td>"
            cells += "".join(f"<td align='right'>{_fmt(r[c])}</td>" for c in stat_cols[1:])
            rows.append(f"<tr>{cells}</tr>")
        table_html = (
            "<table cellspacing='6' cellpadding='0'>"
            f"<tr>{head_cells}</tr>{''.join(rows)}</table>"
        )
        return f"{header}<br>{table_html}"

    # -------------------------------------------------------------- click-to-load
    def _on_press(self, event) -> None:
        """Remember where a left-press began, so the release can tell a selecting
        click from a zoom/pan drag."""
        self._press_xy = (
            (event.x, event.y)
            if getattr(event, "button", None) == 1 and event.x is not None
            else None
        )

    def _on_release(self, event) -> None:
        """On a left-click that didn't drag (a drag is the toolbar zooming or
        panning), select the plotted point nearest the cursor."""
        press = self._press_xy
        self._press_xy = None
        if (
            press is None
            or getattr(event, "button", None) != 1
            or event.x is None
            or abs(event.x - press[0]) > _CLICK_SLOP_PX
            or abs(event.y - press[1]) > _CLICK_SLOP_PX
        ):
            return
        self._pick_at(float(event.x), float(event.y))

    def _pick_at(self, px: float, py: float) -> int | None:
        """Select the plotted point nearest the pixel ``(px, py)`` and ring it.

        Returns the chosen source row, or None when nothing pickable lies within
        :data:`_PICK_TOLERANCE_PX`. The whole resolution runs in *live* display
        coordinates, so it is exact whatever the zoom/pan."""
        if self._target_resolver is None:
            return None
        hit = self._nearest_pickable(px, py)
        if hit is None:
            return None
        row, point_xy = hit
        self._highlight_at(point_xy)
        self._select_row(row)
        return row

    def _nearest_pickable(self, px: float, py: float):
        """``(row, data_xy)`` of the plotted point nearest ``(px, py)`` in pixels,
        or None when none lies within tolerance.

        Each drawn point collection carries its per-marker source rows
        (``_PICK_ROWS_ATTR``, stamped by the plotting backend) aligned to its
        offsets; we project every collection's offsets through its *current*
        offset transform and take the global nearest — no pick radius to miss, no
        arbitrary tie-break, and correct after the user zooms or pans."""
        if self._canvas is None:
            return None
        best = None
        best_d2 = float(_PICK_TOLERANCE_PX) ** 2
        for ax in self._canvas.figure.axes:
            for coll in ax.collections:
                rows = getattr(coll, _PICK_ROWS_ATTR, None)
                if rows is None:
                    continue
                offsets = np.asarray(coll.get_offsets(), dtype=float)
                if not len(offsets):
                    continue
                disp = coll.get_offset_transform().transform(offsets)
                d2 = (disp[:, 0] - px) ** 2 + (disp[:, 1] - py) ** 2
                i = int(np.argmin(d2))
                if d2[i] < best_d2 and i < len(rows):
                    best_d2 = float(d2[i])
                    best = (int(rows[i]), (float(offsets[i, 0]), float(offsets[i, 1])))
        return best

    def _highlight_at(self, point_xy) -> None:
        """Ring the picked marker at its exact drawn position. The marker is
        removed and redrawn on each pick."""
        if self._canvas is None:
            return
        ax = self._canvas.figure.axes[0]
        if self._pick_marker is not None:
            self._pick_marker.remove()
        (self._pick_marker,) = ax.plot(
            [float(point_xy[0])],
            [float(point_xy[1])],
            marker="o",
            markersize=14,
            markerfacecolor="none",
            markeredgecolor="yellow",
            markeredgewidth=2.5,
            zorder=10,
        )
        self._canvas.draw_idle()

    def _select_row(self, row: int) -> None:
        record = self._plot_df.iloc[row]
        identity = {c: _py(record[c]) for c in self._identity_columns}
        level = self._level_combo.currentData()
        # A per-date point pools whole positions, so no single movie is "it" —
        # report the pick but offer nothing to load.
        if level == "date":
            self.selection_changed.emit(identity)
            self._selected_target = None
            self._path_label.setText(
                "Per-date points pool multiple positions — nothing to load."
            )
            self._load_btn.setEnabled(False)
            return
        # A position-wide point identifies a position, not a cell: drop the cell
        # (and its frame) so loading shows the whole position movie with no cell
        # spotlight, and the status line reads as the position only.
        if level == "position":
            for key in ("cell_id", "frame", "frame_start"):
                identity.pop(key, None)
        self.selection_changed.emit(identity)
        target = self._target_resolver(identity) if self._target_resolver else None
        self._selected_target = target
        if target is None:
            self._path_label.setText("No input data found for this point.")
            self._load_btn.setEnabled(False)
        else:
            self._path_label.setText(self._describe_target(target))
            self._load_btn.setEnabled(True)

    def _describe_target(self, target: LoadTarget) -> str:
        """Status-line text for a picked point: its provenance (position / cell /
        frame, whichever apply at the current level) above the input path."""
        ident = target.identity or {}
        bits: list[str] = []
        position = ident.get("position_id")
        if position is not None:
            date = ident.get("date")
            bits.append(f"{date} · {position}" if date is not None else str(position))
        if target.cell_id is not None:
            bits.append(f"cell {target.cell_id}")
        if target.frame is not None:
            bits.append(f"frame {target.frame}")
        header = " · ".join(bits)
        return f"{header}\n{target.path}" if header else str(target.path)

    def _clear_selection(self) -> None:
        self._selected_target = None
        # The marker lives on the old axes; a re-render replaces the canvas, so
        # just drop our reference (removing it from a stale figure is needless).
        self._pick_marker = None
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

    def _export_csv(self) -> None:
        path = self._save_path("Export plot data", "CSV files (*.csv)")
        if path:
            # Exactly the data the current plot draws (points / curve / bars) after
            # filtering, so the CSV reproduces the figure with no further processing.
            table = plotted_table(self._plot_df, self.current_spec())
            self._status.setText(f"Wrote {write_csv(table, path).name}.")

    def _export_figure(self) -> None:
        if self._canvas is None:
            return
        path = self._save_path("Export figure", "Images (*.png *.svg *.pdf)")
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


def _auto_double_spin(lo: float, hi: float, step: float) -> QDoubleSpinBox:
    """A spin whose minimum reads ``auto`` and maps to ``None`` in the StyleSpec —
    the way an optional knob (alpha, line width, tick / marker size) opts out."""
    spin = _double_spin(lo, hi, lo, step)
    spin.setSpecialValueText("auto")
    return spin


def _auto_value(spin: QDoubleSpinBox) -> float | None:
    """The spin's value, or ``None`` when it sits at its ``auto`` minimum."""
    return None if spin.value() <= spin.minimum() else spin.value()


def _line_edit(placeholder: str) -> QLineEdit:
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    return _shrinkable(edit)


def _set_combo(combo: QComboBox, value) -> None:
    """Select the entry whose data is *value*; leave the combo as-is if absent."""
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def _set_auto_spin(spin: QDoubleSpinBox, value: float | None) -> None:
    """Set *spin* to *value*, or to its ``auto`` minimum when *value* is None."""
    spin.setValue(spin.minimum() if value is None else value)


def _set_range_edit(edit: QLineEdit, value: float | None) -> None:
    edit.setText("" if value is None else str(value))


def _combo_opt(combo: QComboBox, sentinel: str) -> str:
    """The combo's value, or ``""`` when it is the *sentinel* (``default``)."""
    value = combo.currentData()
    return "" if value == sentinel else value


def _pair_row(label: str, left: QWidget, right: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label)
    lbl.setFixedWidth(70)
    row.addWidget(lbl)
    row.addWidget(left, 1)
    row.addWidget(right, 1)
    return row


def _checkrow(*checks: QCheckBox) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    for check in checks:
        row.addWidget(check)
    row.addStretch(1)
    return row


class _ColorButton(QPushButton):
    """A colour swatch button: click to pick a colour; blank means ``auto``.

    :meth:`color` is the chosen ``#rrggbb`` or ``""`` when unset; ``changed``
    fires on any change so the panel re-renders."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hex = ""
        _shrinkable(self)
        self.clicked.connect(self._pick)
        self._refresh()

    def color(self) -> str:
        return self._hex

    def set_color(self, value: str) -> None:
        new = value or ""
        if new == self._hex:
            return
        self._hex = new
        self._refresh()
        self.changed.emit()

    def _pick(self) -> None:
        initial = QColor(self._hex) if self._hex else QColor("#888888")
        chosen = QColorDialog.getColor(initial, self, "Pick group colour")
        if chosen.isValid():
            self.set_color(chosen.name())

    def _refresh(self) -> None:
        if self._hex:
            self.setText(self._hex)
            self.setStyleSheet(f"background-color: {self._hex}; color: white;")
        else:
            self.setText("auto")
            self.setStyleSheet("")


class _DetachedPlotWindow(QWidget):
    """Top-level window holding the popped-out plot. Tells the panel to re-dock
    when the user closes it directly (rather than via the Re-attach button)."""

    def __init__(self, on_close: Callable[[], None], parent: QWidget | None = None) -> None:
        # Parented so closing the panel's tab destroys this window too, but flagged
        # as its own window so it floats free of the dock.
        super().__init__(parent)
        self.setWindowFlags(Qt.Window)
        self._on_close = on_close

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._on_close()
        super().closeEvent(event)


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


def _fmt(value: float) -> str:
    """A statistic for the summary table: 4 significant figures, or ``–`` for NaN."""
    return "–" if value != value else f"{value:.4g}"  # noqa: PLR0124 - NaN test


def _clear_layout(layout) -> None:
    """Recursively delete every widget / nested layout in *layout*."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        elif item.layout() is not None:
            _clear_layout(item.layout())
