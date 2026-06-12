"""Dynamics-family Plot consumers: registration + each view's prepare/panel."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import tifffile
from qtpy.QtWidgets import QApplication
from skimage.draw import disk

from cellflow.aggregate_quantification.quantifier import PositionInputs
from cellflow.aggregate_quantification.quantifiers.cell_dynamics import (
    CellDynamicsQuantifier,
)
from cellflow.napari.aggregate_quantification.plots import PlotContext, available_plots
from cellflow.napari.aggregate_quantification.plots.dynamics import (
    CellCurvesDynamicsPlot,
    CellFrameDynamicsPlot,
    CellTissueDynamicsPlot,
    CellTrackDynamicsPlot,
    NucleusFrameDynamicsPlot,
)


def _app():
    return QApplication.instance() or QApplication([])


def _moving_disk_stack(centers, shape=(80, 80), radius=6, label=1):
    frames = []
    for row, col in centers:
        frame = np.zeros(shape, dtype=np.uint16)
        rr, cc = disk((row, col), radius, shape=shape)
        frame[rr, cc] = label
        frames.append(frame)
    return np.stack(frames)


def _built_cell_dynamics(tmp_path, name="p1", condition="A"):
    pos = tmp_path / name
    pos.mkdir()
    labels = pos / "cells.tif"
    centers = [(40, 10 + 2 * i) for i in range(16)]
    tifffile.imwrite(labels, _moving_disk_stack(centers))
    q = CellDynamicsQuantifier()
    inputs = PositionInputs(
        position_dir=pos,
        cell_labels_path=labels,
        pixel_size_um=1.0,
        time_interval_s=1.0,
    )
    q.build(inputs, q.default_output(inputs))
    return {
        "position_path": pos,
        "cell_tracked_labels_path": labels,
        "condition": condition,
        "date": "d1",
        "id": name,
    }


def test_dynamics_plots_register_under_dynamics_family():
    plots = {p.plot_id: p for p in available_plots()}
    expected = {
        "cell_dynamics_frame",
        "cell_dynamics_track",
        "cell_dynamics_tissue",
        "cell_dynamics_curves",
        "nucleus_dynamics_frame",
        "nucleus_dynamics_track",
        "nucleus_dynamics_tissue",
        "nucleus_dynamics_curves",
    }
    assert expected <= set(plots)
    for plot_id in expected:
        assert plots[plot_id].family == "Dynamics"
    # Cell views consume the cell product; nucleus views the nucleus product.
    assert CellFrameDynamicsPlot.consumes == ("cell_dynamics",)
    assert NucleusFrameDynamicsPlot.consumes == ("nucleus_dynamics",)


def test_frame_view_pools_per_frame_table(tmp_path):
    record = _built_cell_dynamics(tmp_path)
    df = CellFrameDynamicsPlot().prepare([record])
    assert not df.empty
    assert "frame" in df.columns
    assert set(CellFrameDynamicsPlot.value_columns) & set(df.columns)
    # Per-frame view carries the NLS class_label join (default unclassified).
    assert (df["class_label"] == "unclassified").all()


def test_launch_cache_reads_each_h5_once(tmp_path, monkeypatch):
    """All four dynamics views share one read per position inside a launch cache."""
    from cellflow.napari.aggregate_quantification.plots import dynamics as dyn

    record = _built_cell_dynamics(tmp_path)
    calls: list[str] = []
    real = dyn.read_track_dynamics
    monkeypatch.setattr(
        dyn, "read_track_dynamics", lambda path: (calls.append(str(path)), real(path))[1]
    )
    views = [
        CellFrameDynamicsPlot(),
        CellTrackDynamicsPlot(),
        CellTissueDynamicsPlot(),
        CellCurvesDynamicsPlot(),
    ]
    with dyn.dynamics_read_cache():
        for view in views:
            view.prepare([record])
    # Without the cache each view re-reads; with it, one parse for the position.
    assert len(calls) == 1


def test_track_view_pools_per_track_summary(tmp_path):
    record = _built_cell_dynamics(tmp_path)
    df = CellTrackDynamicsPlot().prepare([record])
    assert not df.empty
    # Per-track summary has no per-frame axis.
    assert "frame" not in CellTrackDynamicsPlot.group_columns
    assert "msd_alpha" in df.columns


def test_tissue_view_is_one_row_per_position_without_class_join(tmp_path):
    records = [
        _built_cell_dynamics(tmp_path, "p1", "A"),
        _built_cell_dynamics(tmp_path, "p2", "B"),
    ]
    df = CellTissueDynamicsPlot().prepare(records)
    assert len(df) == 2
    assert "class_label" not in df.columns
    assert {"corr_length_um", "order_param"} <= set(df.columns)
    assert set(df["position_id"]) == {"p1", "p2"}


def test_curves_view_prepares_curve_sets_and_builds_bespoke_panel(tmp_path):
    from cellflow.napari.aggregate_quantification.dynamics_curves_panel import (
        CurveSet,
        DynamicsCurvesPanel,
    )

    app = _app()
    record = _built_cell_dynamics(tmp_path)
    plot = CellCurvesDynamicsPlot()
    curves = plot.prepare([record])
    assert curves and isinstance(curves[0], CurveSet)
    panel = plot.create_panel(PlotContext(records=[record]), prepared=curves)
    try:
        assert isinstance(panel, DynamicsCurvesPanel)
    finally:
        panel.deleteLater()
        app.processEvents()


def test_distribution_views_build_plot_panels(tmp_path):
    from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel

    app = _app()
    record = _built_cell_dynamics(tmp_path)
    ctx = PlotContext(records=[record], viewer=None)
    for cls in (CellFrameDynamicsPlot, CellTrackDynamicsPlot, CellTissueDynamicsPlot):
        panel = cls().create_panel(ctx)
        try:
            assert isinstance(panel, PlotPanel)
        finally:
            panel.deleteLater()
            app.processEvents()
