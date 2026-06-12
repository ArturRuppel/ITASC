from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from qtpy.QtWidgets import QApplication

from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel, ValueSource


def _app():
    return QApplication.instance() or QApplication([])


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


def test_construct_renders_a_canvas():
    app = _app()
    panel = _panel()
    assert panel._canvas is not None
    assert panel._toolbar is not None
    panel.deleteLater()
    app.processEvents()


def test_each_control_rerenders_without_error():
    app = _app()
    panel = _panel()

    # Every analytical plot type re-renders from the held snapshot.
    for plot in ("hist", "box", "violin", "strip", "swarm", "bar", "line"):
        panel._plot_combo.setCurrentText(plot)
        assert panel._canvas is not None

    panel._level_combo.setCurrentIndex(1)  # per position
    panel._stat_combo.setCurrentText("count")
    panel._group_checks["condition"].setChecked(True)
    panel._group_checks["class_label"].setChecked(True)
    panel._bins_spin.setValue(15)

    # Styling controls re-render too.
    panel._palette_combo.setCurrentText("Set2")
    panel._title_edit.setText("My title")
    panel._render()
    assert panel._canvas.figure.axes[0].get_title() == "My title"
    panel._width_spin.setValue(8.0)
    panel._height_spin.setValue(5.0)
    panel._render()
    assert panel._canvas.figure.get_size_inches().tolist() == [8.0, 5.0]
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
    assert panel._identity_columns == ("position_id", "frame", "cell_id")
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
    from types import SimpleNamespace
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
    # A native matplotlib pick: the drawn artist carries its exact source rows, so
    # ``event.ind`` indexes straight to the row — no value-proximity guessing.
    artist = next(c for c in ax.collections if hasattr(c, _PICK_ROWS_ATTR))
    rows = getattr(artist, _PICK_ROWS_ATTR)
    offsets = np.asarray(artist.get_offsets(), dtype=float)
    panel._on_pick(SimpleNamespace(artist=artist, ind=[0]))
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


def test_pick_distinguishes_equal_valued_points():
    # The core fix: two points with the *same* value must resolve to their own
    # rows. The old value-proximity reconstruction collided them; native picking
    # reads each marker's stamped row, so ``ind`` 0 and 1 stay distinct.
    from types import SimpleNamespace
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
    assert len(getattr(artist, _PICK_ROWS_ATTR)) == 2
    panel._on_pick(SimpleNamespace(artist=artist, ind=[0]))
    panel._on_pick(SimpleNamespace(artist=artist, ind=[1]))
    assert selected == [101, 202]
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
    # With no resolver the panel must not act on picks (no selection, no ring),
    # matching how loading stays disabled.
    from types import SimpleNamespace
    from cellflow.aggregate_quantification.plotting import _PICK_ROWS_ATTR
    app = _app()
    emitted: list = []
    panel = _panel()                       # no target_resolver
    panel.selection_changed.connect(emitted.append)
    panel._plot_combo.setCurrentText("strip")
    panel._render()
    ax = panel._canvas.figure.axes[0]
    artist = next(c for c in ax.collections if hasattr(c, _PICK_ROWS_ATTR))
    panel._on_pick(SimpleNamespace(artist=artist, ind=[0]))
    assert emitted == []
    assert panel._pick_marker is None
    panel.deleteLater(); app.processEvents()


def test_export_html_writes_interactive_file(tmp_path, monkeypatch):
    app = _app()
    panel = _panel()
    out = tmp_path / "plot"   # no suffix — the slot must add ``.html``
    monkeypatch.setattr(panel, "_save_path", lambda *a, **k: out)
    panel._export_html()
    written = out.with_name(out.name + ".html")
    assert written.exists() and written.stat().st_size > 0
    assert "plotly" in written.read_text().lower()
    assert panel._export_html_btn.isEnabled() is True
    panel.deleteLater(); app.processEvents()
