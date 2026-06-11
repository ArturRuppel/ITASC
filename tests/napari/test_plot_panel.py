from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from qtpy.QtWidgets import QApplication

from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel


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

    pooled = tmp_path / "pooled.csv"
    panel._df.to_csv(pooled, index=False)  # sanity: snapshot is tidy
    from cellflow.aggregate_quantification.plotting import aggregate, write_csv

    agg_path = write_csv(aggregate(panel._df, panel.current_spec()), tmp_path / "agg")
    assert agg_path.exists()

    fig_path = tmp_path / "fig.png"
    panel._canvas.figure.savefig(fig_path)
    assert fig_path.exists() and fig_path.stat().st_size > 0

    assert panel._export_pooled_btn.isEnabled() is True
    assert panel._export_agg_btn.isEnabled() is True
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
    assert panel._export_pooled_btn.isEnabled() is False
    panel.deleteLater()
    app.processEvents()
