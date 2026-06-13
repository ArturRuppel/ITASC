from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from qtpy.QtWidgets import QApplication

from cellflow.aggregate_quantification.reduce import Collapse, Filter
from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel, ValueSource


def _app():
    return QApplication.instance() or QApplication([])


def _set_level(panel, level: str) -> None:
    """Drive the panel's Shape editor to the collapse equivalent of the old Level
    dropdown (one collapse to the named unit), then re-render."""
    by_for = {
        "cell": ("condition", "date", "position_id", "cell_id"),
        "position": ("condition", "date", "position_id"),
        "date": ("condition", "date"),
    }[level]
    by = tuple(c for c in by_for if c in panel._df.columns)
    panel._shape_editor.set_columns(
        panel._shape_columns(), (Collapse(by=by, stat="mean"),), categorical={}
    )
    panel._render()


def _df():
    rng = np.random.default_rng(0)
    n = 12
    return pd.DataFrame({
        "condition": ["A"] * 6 + ["B"] * 6,
        "date": ["d1"] * 12,
        "position_id": (["p1"] * 3 + ["p2"] * 3) * 2,
        "class_label": (["pos", "neg", "pos"] * 4),
        "frame": list(range(3)) * 4,
        "cell_id": list(range(12)),
        "area": rng.normal(50, 5, n),
    })


def _panel():
    return PlotPanel(
        _df(),
        value_columns=("area",),
        group_columns=("condition", "date", "position_id", "class_label", "frame"),
    )


def _pixel_of(coll, i):
    """Display-pixel (x, y) of marker *i* in *coll* — where a click lands to hit it."""
    off = np.asarray(coll.get_offsets(), dtype=float)
    return tuple(coll.get_offset_transform().transform(off)[int(i)])


def test_catalog_mode_spans_sources_and_swaps_on_selection():
    app = _app()
    # Two products with different columns AND different group axes.
    cells = pd.DataFrame({
        "condition": ["A", "A", "B"],
        "frame": [0, 1, 0],
        "cell_id": [1, 1, 2],
        "area": [10.0, 11.0, 12.0],
    })
    tracks = pd.DataFrame({
        "condition": ["A", "B"],
        "track_id": [1, 2],
        "speed": [0.5, 0.7],
    })
    cell_resolver = lambda identity: f"cell:{identity}"  # noqa: E731 - test stub
    catalog = [
        ValueSource(
            cells, "area", ("condition", "frame"), "Cell shape: area", "Shape",
            target_resolver=cell_resolver,
        ),
        ValueSource(tracks, "speed", ("condition",), "Dynamics: speed", "Dynamics"),
    ]
    panel = PlotPanel(value_catalog=catalog, default_plot="box")
    try:
        # Starts on the first source (cells / area) with its group axes + resolver.
        assert panel.current_spec().value == "area"
        assert panel._target_resolver is cell_resolver
        assert "frame" in panel._group_checks
        # Disabled family headers appear in the picker for grouping.
        texts = [panel._value_combo.itemText(i) for i in range(panel._value_combo.count())]
        assert any("── Shape ──" in t for t in texts)
        assert any("── Dynamics ──" in t for t in texts)
        # Select the dynamics value → df + group axes swap, no 'frame' axis now.
        index = next(
            i
            for i in range(panel._value_combo.count())
            if panel._value_combo.itemData(i) == "speed"
        )
        panel._value_combo.setCurrentIndex(index)
        assert panel.current_spec().value == "speed"
        assert panel._df is tracks
        assert "frame" not in panel._group_checks
        assert "condition" in panel._group_checks
        # Resolver follows the active source (dynamics source has none).
        assert panel._target_resolver is None
    finally:
        panel.deleteLater()
        app.processEvents()


def test_shape_filter_step_narrows_the_plot():
    from cellflow.aggregate_quantification.plotting import reduce_to_units
    app = _app()
    panel = _panel()
    # A filter step (condition == A) ahead of the default collapse narrows the plot
    # through the one Shape pipeline, not a separate filter pass.
    panel._shape_editor.set_columns(
        panel._shape_columns(),
        (
            Filter("condition", "==", "A"),
            Collapse(by=tuple(panel._collapse_columns()), stat="mean"),
        ),
        categorical={"condition": ["A", "B"]},
    )
    panel._render()
    spec = panel.current_spec()
    # The filter rides at the head of the pipeline, before the collapse.
    assert spec.collapse[0] == Filter("condition", "==", "A")
    units = reduce_to_units(panel._plot_df, spec)
    assert set(units["condition"]) == {"A"}  # the B rows were filtered out
    assert len(units) == 6  # 6 A cells survive (one row each after the collapse)
    panel.deleteLater(); app.processEvents()


def test_shape_offers_categorical_filter_values():
    app = _app()
    panel = _panel()
    cat = panel._shape_categorical(panel._shape_columns())
    # ``condition`` / ``class_label`` are categorical (value dropdowns); the numeric
    # value column ``area`` and the single-date ``date`` are not multi-value picks.
    assert set(cat["condition"]) == {"A", "B"}
    assert "pos" in cat["class_label"]
    assert "area" not in cat  # numeric → free entry, not a dropdown
    panel.deleteLater(); app.processEvents()


def test_filter_n_drops_undersampled_units_after_collapse():
    from cellflow.aggregate_quantification.plotting import reduce_to_units
    app = _app()
    panel = _panel()
    # Collapse to per (condition, position) — each position pools 3 cells → n == 3 —
    # then filter n >= 4 to drop them all (the pseudoreplication payoff).
    panel._shape_editor.set_columns(
        panel._shape_columns(),
        (
            Collapse(by=("condition", "position_id"), stat="mean"),
            Filter("n", ">=", "4"),
        ),
        categorical={},
    )
    panel._render()
    spec = panel.current_spec()
    units = reduce_to_units(panel._plot_df, spec)
    assert units.empty  # every position had only 3 cells, all dropped by n >= 4
    panel.deleteLater(); app.processEvents()


def test_collapse_columns_exclude_absent_entities():
    app = _app()
    # A per-position table (no cell_id) must not offer "cell_id" as a collapse axis,
    # and its effective unit can't be "cell".
    tissue = pd.DataFrame({
        "condition": ["A", "B"],
        "date": ["d1", "d2"],
        "position_id": ["p1", "p2"],
        "order_param": [0.4, 0.6],
    })
    panel = PlotPanel(tissue, value_columns=("order_param",),
                      group_columns=("condition", "date", "position_id"))
    columns = panel._collapse_columns()
    assert "cell_id" not in columns
    assert "position_id" in columns
    # The default pipeline collapses to the finest unit present — position, not cell.
    assert panel._effective_level() != "cell"
    panel.deleteLater(); app.processEvents()


def test_value_picker_tags_native_grain():
    # The native grain is shown once, as a bracketed tag in the value picker — the
    # data here is recorded per frame (frame + cell_id columns).
    app = _app()
    panel = _panel()  # carries frame + cell_id
    texts = [panel._value_combo.itemText(i) for i in range(panel._value_combo.count())]
    assert texts == ["area  ·  [per frame]"]
    # The aggregation unit is a separate concern, in the dedicated Reduce editor:
    # the table carries both per-cell and per-position nesting to collapse over.
    columns = panel._collapse_columns()
    assert "cell_id" in columns and "position_id" in columns


def test_native_grain_tag_tracks_the_table():
    # A per-position table (no cell_id / frame) tags as "per position".
    app = _app()
    tissue = pd.DataFrame({
        "condition": ["A", "B"], "date": ["d1", "d2"], "position_id": ["p1", "p2"],
        "order_param": [0.4, 0.6],
    })
    panel = PlotPanel(tissue, value_columns=("order_param",),
                      group_columns=("condition", "date", "position_id"))
    assert panel._value_combo.itemText(0) == "order_param  ·  [per position]"
    panel.deleteLater(); app.processEvents()


def test_stats_readout_reflects_current_value_and_level():
    app = _app()
    panel = _panel()
    panel._group_checks["condition"].setChecked(True)
    panel._render()
    html = panel._stats_label.text()
    assert "Summary" in html and "area" in html
    # The per-cell unit count (12 distinct cells here) and the stat columns show.
    for col in ("mean", "median", "sd", "sem"):
        assert col in html
    panel.deleteLater(); app.processEvents()


def test_value_swap_jumps_to_suggested_plot():
    app = _app()
    box_src = ValueSource(
        pd.DataFrame({"condition": ["A", "B"], "cell_id": [1, 2], "n_neighbors": [3, 5]}),
        "n_neighbors", ("condition",), "Neighbor count: n_neighbors", "Contacts",
        suggested_plot="box",
    )
    bar_src = ValueSource(
        pd.DataFrame({"condition": ["A", "B"], "z_score": [1.0, -1.0]}),
        "z_score", ("condition",), "Contact-type z-score: z_score", "Contacts",
        suggested_plot="bar",
    )
    panel = PlotPanel(value_catalog=[box_src, bar_src])
    try:
        assert panel.current_spec().plot == "box"  # first source's suggestion
        index = next(
            i for i in range(panel._value_combo.count())
            if panel._value_combo.itemData(i) == "z_score"
        )
        panel._value_combo.setCurrentIndex(index)
        # Switching to the z-score product jumped to its natural bar rendering.
        assert panel.current_spec().plot == "bar"
    finally:
        panel.deleteLater(); app.processEvents()


def test_construct_renders_a_canvas():
    app = _app()
    panel = _panel()
    assert panel._canvas is not None
    assert panel._toolbar is not None
    panel.deleteLater()
    app.processEvents()


def test_detach_pops_plot_into_window_and_reattach_docks_it():
    app = _app()
    panel = _panel()
    # Docked: the container sits in the panel layout, the placeholder is hidden.
    assert panel._main_layout.indexOf(panel._plot_container) == panel._plot_index
    assert panel._detached_window is None

    panel._toggle_detach()
    app.processEvents()
    # Detached: container reparented into the float window, placeholder takes its
    # docked slot, and a re-render still targets the (now floating) canvas holder.
    assert panel._detached_window is not None
    assert panel._plot_container.window() is panel._detached_window
    assert panel._main_layout.indexOf(panel._detach_placeholder) == panel._plot_index
    panel._render()
    assert panel._canvas is not None

    panel._toggle_detach()
    app.processEvents()
    # Re-attached: container back in the panel, window gone.
    assert panel._detached_window is None
    assert panel._main_layout.indexOf(panel._plot_container) == panel._plot_index
    panel.deleteLater()
    app.processEvents()


def test_closing_detached_window_redocks_plot():
    app = _app()
    panel = _panel()
    panel._toggle_detach()
    app.processEvents()
    window = panel._detached_window
    assert window is not None
    # Closing the float window directly (not via the button) re-docks the plot.
    window.close()
    app.processEvents()
    assert panel._detached_window is None
    assert panel._main_layout.indexOf(panel._plot_container) == panel._plot_index
    panel.deleteLater()
    app.processEvents()


def test_collapse_pipeline_drives_the_unit():
    """The Reduce editor's collapse pipeline — not a Level dropdown — sets the
    independent unit: per-cell counts every cell, per-position collapses each
    position to one unit (climbing to a coarser unit cuts the unit count)."""
    from cellflow.aggregate_quantification.plotting import summary_table
    app = _app()
    # 12 cells; position_id p1/p2 each appear under both conditions → 4 positions.
    panel = _panel()
    panel._group_checks["condition"].setChecked(True)
    panel._plot_combo.setCurrentText("box")

    _set_level(panel, "cell")
    cell_n = int(summary_table(panel._plot_df, panel.current_spec())["n"].sum())

    _set_level(panel, "position")
    pos_n = int(summary_table(panel._plot_df, panel.current_spec())["n"].sum())

    assert cell_n == 12  # one unit per cell
    assert pos_n == 4    # one unit per (condition, position)
    assert pos_n < cell_n  # climbing to a coarser unit reduces the datapoint count
    panel.deleteLater(); app.processEvents()


def test_each_control_rerenders_without_error():
    app = _app()
    panel = _panel()

    # Every analytical plot type re-renders from the held snapshot.
    for plot in ("hist", "box", "violin", "strip", "swarm", "bar", "line"):
        panel._plot_combo.setCurrentText(plot)
        assert panel._canvas is not None

    _set_level(panel, "position")  # collapse to per position
    panel._stat_combo.setCurrentText("count")
    panel._group_checks["condition"].setChecked(True)
    panel._group_checks["class_label"].setChecked(True)
    panel._bins_spin.setValue(15)

    # Styling controls re-render too.
    panel._palette_combo.setCurrentText("Set2")
    panel._title_edit.setText("My title")
    panel._render()
    assert panel._canvas.figure.axes[0].get_title() == "My title"
    panel._font_spin.setValue(16.0)
    panel._grid_cb.setChecked(True)
    panel._legend_cb.setChecked(False)
    panel._render()

    panel.deleteLater()
    app.processEvents()


def test_specs_reflect_controls():
    app = _app()
    panel = _panel()
    panel._group_checks["condition"].setChecked(True)
    panel._plot_combo.setCurrentText("violin")
    spec = panel.current_spec()
    assert spec.value == "area"
    assert spec.plot == "violin"
    assert spec.group_by == ("condition",)

    panel._palette_combo.setCurrentText("Dark2")
    panel._grid_cb.setChecked(True)
    panel._box_whis_spin.setValue(3.0)
    panel._box_fliers_cb.setChecked(False)
    panel._box_notch_cb.setChecked(True)
    style = panel.current_style()
    assert style.palette == "Dark2"
    assert style.grid is True
    assert style.box_whis == 3.0
    assert style.box_showfliers is False
    assert style.box_notch is True
    panel.deleteLater()
    app.processEvents()


def test_export_csv_and_figure(tmp_path):
    app = _app()
    panel = _panel()

    from cellflow.aggregate_quantification.plotting import plotted_table, write_csv

    # The single CSV export writes exactly the data the current plot draws.
    csv_path = write_csv(plotted_table(panel._df, panel.current_spec()), tmp_path / "plot")
    assert csv_path.exists()

    fig_path = tmp_path / "fig.png"
    panel._canvas.figure.savefig(fig_path)
    assert fig_path.exists() and fig_path.stat().st_size > 0

    assert panel._export_csv_btn.isEnabled() is True
    assert panel._export_fig_btn.isEnabled() is True
    panel.deleteLater()
    app.processEvents()


def test_selection_changed_signal_exists_and_emits():
    app = _app()
    panel = _panel()
    received: list = []
    panel.selection_changed.connect(received.append)
    rows = [{"position_id": "p1", "frame": 0, "cell_id": 0}]
    panel.selection_changed.emit(rows)
    assert received == [rows]
    # ``date`` rides along with ``position_id`` so click-to-load can disambiguate
    # the same position id reused across experiments (see ClickToLoad.resolver).
    assert panel._identity_columns == ("date", "position_id", "frame", "cell_id")
    panel.deleteLater()
    app.processEvents()


def test_empty_snapshot_disables_exports():
    app = _app()
    panel = PlotPanel(pd.DataFrame(), value_columns=("area",), group_columns=("condition",))
    assert panel._export_csv_btn.isEnabled() is False
    panel.deleteLater()
    app.processEvents()


def test_absent_value_columns_are_not_offered():
    app = _app()
    # ``msd_D_um2_per_s`` isn't in the snapshot (a stale build lacking the
    # per-track fit); it must be dropped so it can't be selected and crash render.
    panel = PlotPanel(
        _df(),
        value_columns=("area", "msd_D_um2_per_s"),
        group_columns=("condition",),
    )
    assert panel._value_columns == ("area",)
    offered = [panel._value_combo.itemData(i) for i in range(panel._value_combo.count())]
    assert "msd_D_um2_per_s" not in offered
    panel.deleteLater()
    app.processEvents()


def test_axis_range_fields_feed_style_and_render():
    app = _app()
    panel = _panel()
    panel._ymin_edit.setText("0")
    panel._ymax_edit.setText("100")
    style = panel.current_style()
    assert style.ymin == 0.0 and style.ymax == 100.0
    panel._render()
    assert panel._canvas.figure.axes[0].get_ylim() == (0.0, 100.0)
    panel.deleteLater(); app.processEvents()


def test_pick_resolves_identity_and_enables_load():
    from pathlib import Path
    from cellflow.napari.aggregate_quantification.plot_panel import LoadTarget, PlotPanel
    from cellflow.aggregate_quantification.plotting import _PICK_ROWS_ATTR
    app = _app()
    seen = {}

    def resolver(identity):
        seen.update(identity)
        return LoadTarget(path=Path("/tmp/labels.tif"), kind="labels",
                          frame=identity.get("frame"), cell_id=identity.get("cell_id"),
                          identity=identity)

    panel = PlotPanel(_df(), value_columns=("area",),
                      group_columns=("condition", "date", "position_id", "class_label", "frame"),
                      target_resolver=resolver)
    panel._plot_combo.setCurrentText("strip")
    panel._render()
    ax = panel._canvas.figure.axes[0]
    # Clicking a marker's pixel selects it: the collection carries its exact source
    # rows aligned to its offsets, and the panel takes the nearest in pixel space.
    artist = next(c for c in ax.collections if hasattr(c, _PICK_ROWS_ATTR))
    rows = getattr(artist, _PICK_ROWS_ATTR)
    offsets = np.asarray(artist.get_offsets(), dtype=float)
    px, py = _pixel_of(artist, 0)
    assert panel._pick_at(px, py) == int(rows[0])
    expected_row = int(rows[0])
    # The ring lands exactly on the drawn marker (its offset), not a column centre.
    assert panel._pick_marker is not None
    assert panel._pick_marker.get_xdata()[0] == offsets[0][0]
    assert panel._pick_marker.get_ydata()[0] == offsets[0][1]
    assert panel._load_btn.isEnabled()
    assert "/tmp/labels.tif" in panel._path_label.text()
    assert seen["cell_id"] == int(panel._df.iloc[expected_row]["cell_id"])
    emitted = []
    panel.load_requested.connect(emitted.append)
    panel._load_btn.click()
    assert emitted and emitted[0].path == Path("/tmp/labels.tif")
    # Re-rendering drops the marker reference (it lived on the replaced canvas).
    panel._render()
    assert panel._pick_marker is None
    panel.deleteLater(); app.processEvents()


def test_level_shapes_load_target_and_status():
    """The picked unit's level controls what loads: a cell (cell+frame, spotlit),
    a whole position (no cell), or — per date — nothing at all."""
    from pathlib import Path
    from cellflow.napari.aggregate_quantification.plot_panel import LoadTarget, PlotPanel
    from cellflow.aggregate_quantification.plotting import _PICK_ROWS_ATTR
    app = _app()

    def resolver(identity):
        return LoadTarget(path=Path("/tmp/labels.tif"), kind="labels",
                          frame=identity.get("frame"), cell_id=identity.get("cell_id"),
                          identity=identity)

    def pick_top(level):
        panel = PlotPanel(_df(), value_columns=("area",),
                          group_columns=("condition", "date", "position_id", "class_label", "frame"),
                          target_resolver=resolver, default_plot="strip")
        _set_level(panel, level)
        artist = next(c for c in panel._canvas.figure.axes[0].collections
                      if hasattr(c, _PICK_ROWS_ATTR))
        offsets = np.asarray(artist.get_offsets(), dtype=float)
        top = int(np.argmax(offsets[:, 1]))
        panel._pick_at(*_pixel_of(artist, top))
        return panel

    # Cell level: cell + frame on the target and written into the status line.
    cell = pick_top("cell")
    assert cell._selected_target.cell_id is not None
    assert cell._selected_target.frame is not None
    assert cell._load_btn.isEnabled()
    assert "cell " in cell._path_label.text() and "frame " in cell._path_label.text()
    cell.deleteLater()

    # Position level: position only — no cell (and no frame) on the target.
    pos = pick_top("position")
    assert pos._selected_target.cell_id is None
    assert pos._selected_target.frame is None
    assert pos._load_btn.isEnabled()
    assert "cell " not in pos._path_label.text()
    pos.deleteLater()

    # Date level: nothing to load — button greyed out, no target.
    date = pick_top("date")
    assert date._selected_target is None
    assert date._load_btn.isEnabled() is False
    date.deleteLater()
    app.processEvents()


def test_pick_distinguishes_equal_valued_points():
    # Two points with the *same* value must resolve to their own rows. Stripplot
    # jitters them to distinct x, and nearest-in-pixels reads each marker's own
    # stamped row, so they never collide.
    from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel
    from cellflow.aggregate_quantification.plotting import _PICK_ROWS_ATTR
    app = _app()
    selected: list[int] = []
    df = pd.DataFrame({
        "condition": ["A", "A"],
        "cell_id": [101, 202],
        "area": [42.0, 42.0],   # identical value, different cells
    })
    panel = PlotPanel(df, value_columns=("area",), group_columns=("condition",),
                      target_resolver=lambda identity: None)
    panel.selection_changed.connect(lambda ident: selected.append(int(ident["cell_id"])))
    panel._plot_combo.setCurrentText("strip")
    panel._render()
    ax = panel._canvas.figure.axes[0]
    artist = next(c for c in ax.collections if hasattr(c, _PICK_ROWS_ATTR))
    rows = getattr(artist, _PICK_ROWS_ATTR)
    assert len(rows) == 2
    by_cell = {int(panel._df.iloc[int(r)]["cell_id"]): i for i, r in enumerate(rows)}
    for cell in (101, 202):
        panel._pick_at(*_pixel_of(artist, by_cell[cell]))
    assert selected == [101, 202]
    panel.deleteLater(); app.processEvents()


def test_pick_selects_nearest_not_arbitrary_overlap():
    # The wrong-cell bug: in a dense plot many markers fall within a few pixels of
    # the cursor. Picking must resolve to the marker *nearest* the click — the tall
    # point when the tall point is clicked — not an arbitrary one in range.
    from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel
    from cellflow.aggregate_quantification.plotting import _PICK_ROWS_ATTR
    app = _app()
    selected: list[int] = []
    # One conspicuously tall cell among a tight cluster of small ones.
    df = pd.DataFrame({
        "condition": ["A"] * 9,
        "cell_id": list(range(9)),
        "area": [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.1, 90.0],
    })
    panel = PlotPanel(df, value_columns=("area",), group_columns=("condition",),
                      target_resolver=lambda identity: None)
    panel.selection_changed.connect(lambda ident: selected.append(int(ident["cell_id"])))
    panel._plot_combo.setCurrentText("strip")
    panel._render()
    ax = panel._canvas.figure.axes[0]
    artist = next(c for c in ax.collections if hasattr(c, _PICK_ROWS_ATTR))
    rows = getattr(artist, _PICK_ROWS_ATTR)
    tall = next(i for i, r in enumerate(rows) if int(panel._df.iloc[int(r)]["cell_id"]) == 8)
    panel._pick_at(*_pixel_of(artist, tall))
    assert selected == [8]  # the tall cell, not a clustered neighbour
    panel.deleteLater(); app.processEvents()


def test_pick_survives_zoom():
    # Zooming changes data→pixel scaling; picking runs in *live* display coords, so
    # it still resolves the right point (the old pick_event was suppressed by the
    # toolbar's widget-lock while zoomed — nothing was clickable at all).
    from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel
    from cellflow.aggregate_quantification.plotting import _PICK_ROWS_ATTR
    app = _app()
    selected: list[int] = []
    panel = PlotPanel(_df(), value_columns=("area",), group_columns=("condition",),
                      target_resolver=lambda identity: None)
    panel.selection_changed.connect(lambda ident: selected.append(int(ident["cell_id"])))
    panel._plot_combo.setCurrentText("strip")
    panel._render()
    ax = panel._canvas.figure.axes[0]
    # Zoom into the upper half of the data, then recompute pixel positions.
    lo, hi = ax.get_ylim()
    ax.set_ylim((lo + hi) / 2, hi)
    panel._canvas.draw()
    artist = next(c for c in ax.collections if hasattr(c, _PICK_ROWS_ATTR))
    rows = getattr(artist, _PICK_ROWS_ATTR)
    # A marker that stayed in view after the zoom.
    ymid = (lo + hi) / 2
    offs = np.asarray(artist.get_offsets(), dtype=float)
    visible = next(i for i in range(len(offs)) if offs[i, 1] > ymid)
    panel._pick_at(*_pixel_of(artist, visible))
    assert selected == [int(panel._df.iloc[int(rows[visible])]["cell_id"])]
    panel.deleteLater(); app.processEvents()


def test_drag_does_not_select():
    # A press→release that travels (the toolbar's zoom/pan rubber-band) is never a
    # selection, so the user can zoom with the tool active and still click points.
    from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel
    from cellflow.aggregate_quantification.plotting import _PICK_ROWS_ATTR
    from types import SimpleNamespace
    app = _app()
    selected: list = []
    panel = PlotPanel(_df(), value_columns=("area",), group_columns=("condition",),
                      target_resolver=lambda identity: None)
    panel.selection_changed.connect(selected.append)
    panel._plot_combo.setCurrentText("strip")
    panel._render()
    ax = panel._canvas.figure.axes[0]
    artist = next(c for c in ax.collections if hasattr(c, _PICK_ROWS_ATTR))
    px, py = _pixel_of(artist, 0)
    panel._on_press(SimpleNamespace(button=1, x=px, y=py))
    panel._on_release(SimpleNamespace(button=1, x=px + 60, y=py + 40))  # dragged away
    assert selected == []
    assert panel._pick_marker is None
    # A press→release in place at the same marker does select.
    panel._on_press(SimpleNamespace(button=1, x=px, y=py))
    panel._on_release(SimpleNamespace(button=1, x=px, y=py))
    assert len(selected) == 1
    panel.deleteLater(); app.processEvents()


def test_no_resolver_means_no_load_ui():
    app = _app()
    panel = _panel()                       # no target_resolver
    assert not panel._load_btn.isEnabled()
    panel.deleteLater(); app.processEvents()


def test_group_by_survives_value_change_when_column_still_present():
    # Bug 5: changing the Value picker must not silently reset the grouping for a
    # group column the newly-selected product still offers.
    app = _app()
    cells = pd.DataFrame({"condition": ["A", "B"], "cell_id": [1, 2], "area": [10.0, 12.0]})
    tracks = pd.DataFrame({"condition": ["A", "B"], "track_id": [1, 2], "speed": [0.5, 0.7]})
    catalog = [
        ValueSource(cells, "area", ("condition",), "Cell shape: area", "Shape"),
        ValueSource(tracks, "speed", ("condition",), "Dynamics: speed", "Dynamics"),
    ]
    panel = PlotPanel(value_catalog=catalog, default_plot="strip")
    try:
        panel._group_checks["condition"].setChecked(True)
        assert panel.current_spec().group_by == ("condition",)
        # Swap to the other product (also grouped by condition).
        index = next(
            i for i in range(panel._value_combo.count())
            if panel._value_combo.itemData(i) == "speed"
        )
        panel._value_combo.setCurrentIndex(index)
        assert panel.current_spec().value == "speed"
        # The grouping carried over rather than resetting.
        assert panel._group_checks["condition"].isChecked()
        assert panel.current_spec().group_by == ("condition",)
    finally:
        panel.deleteLater(); app.processEvents()


def test_no_resolver_means_no_pick_select():
    # With no resolver the panel must not act on clicks (no selection, no ring),
    # matching how loading stays disabled.
    from cellflow.aggregate_quantification.plotting import _PICK_ROWS_ATTR
    app = _app()
    emitted: list = []
    panel = _panel()                       # no target_resolver
    panel.selection_changed.connect(emitted.append)
    panel._plot_combo.setCurrentText("strip")
    panel._render()
    ax = panel._canvas.figure.axes[0]
    artist = next(c for c in ax.collections if hasattr(c, _PICK_ROWS_ATTR))
    assert panel._pick_at(*_pixel_of(artist, 0)) is None
    assert emitted == []
    assert panel._pick_marker is None
    panel.deleteLater(); app.processEvents()


def _shown_opts(panel) -> set:
    """Plot-option keys currently shown (``isHidden`` reflects the explicit
    ``setVisible`` even when the panel was never shown on screen)."""
    return {key for key, row in panel._opt_rows.items() if not row.isHidden()}


def test_plot_options_visibility_follows_plot_type():
    app = _app()
    panel = _panel()
    panel._plot_combo.setCurrentText("box")
    assert _shown_opts(panel) == {"box_whis", "box_showfliers", "box_notch"}
    panel._plot_combo.setCurrentText("hist")
    assert _shown_opts(panel) == {"bins", "hist_element", "hist_cumulative"}
    panel._plot_combo.setCurrentText("line")
    assert _shown_opts(panel) == {"markers", "marker_size"}
    # swarm has no plot-specific options → the whole box collapses.
    panel._plot_combo.setCurrentText("swarm")
    assert _shown_opts(panel) == set()
    assert panel._plot_opts_box.isHidden() is True
    panel.deleteLater(); app.processEvents()


def test_group_colour_swatches_track_grouping_and_feed_style():
    app = _app()
    panel = _panel()
    # No group-by → no swatches.
    assert panel._override_buttons == {}
    panel._group_checks["condition"].setChecked(True)
    assert set(panel._override_buttons) == {"A", "B"}
    panel._override_buttons["A"].set_color("#ff0000")
    assert panel.current_style().color_overrides == (("A", "#ff0000"),)
    # Dropping the group-by clears the swatches again.
    panel._group_checks["condition"].setChecked(False)
    assert panel._override_buttons == {}
    assert panel.current_style().color_overrides == ()
    panel.deleteLater(); app.processEvents()


def test_style_theme_save_then_load_restores_widgets(tmp_path, monkeypatch):
    app = _app()
    panel = _panel()
    panel._group_checks["condition"].setChecked(True)
    # A distinctive style across several tabs.
    panel._dpi_spin.setValue(175)
    panel._alpha_spin.setValue(0.4)
    panel._xlog_cb.setChecked(True)
    panel._spine_checks["top"].setChecked(False)
    panel._xrot_spin.setValue(60)
    panel._title_edit.setText("My Title")
    panel._override_buttons["A"].set_color("#abcdef")
    saved = panel.current_style()

    out = tmp_path / "theme"   # no suffix — the slot must add ``.json``
    monkeypatch.setattr(panel, "_save_path", lambda *a, **k: out)
    panel._save_style()
    written = out.with_name(out.name + ".json")
    assert written.exists()

    # Perturb every changed control, then load the theme back.
    panel._dpi_spin.setValue(100)
    panel._alpha_spin.setValue(panel._alpha_spin.minimum())
    panel._xlog_cb.setChecked(False)
    panel._spine_checks["top"].setChecked(True)
    panel._title_edit.setText("")
    panel._override_buttons["A"].set_color("")
    monkeypatch.setattr(
        "qtpy.QtWidgets.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(written), ""),
    )
    panel._load_style()
    assert panel.current_style() == saved
    panel.deleteLater(); app.processEvents()
