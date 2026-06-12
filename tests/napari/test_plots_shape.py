"""Shape-family Plot consumers: registration, pooling, and panel construction."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import tifffile
from qtpy.QtWidgets import QApplication

from cellflow.aggregate_quantification.quantifier import PositionInputs
from cellflow.aggregate_quantification.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.aggregate_quantification.shape import DESCRIPTOR_COLUMNS
from cellflow.napari.aggregate_quantification.plots import available_plots
from cellflow.napari.aggregate_quantification.plots._pooling import pool_quantity
from cellflow.napari.aggregate_quantification.plots.shape import (
    CellShapePlot,
    NucleusShapePlot,
    ShapeRelationalPlot,
)


def _app():
    return QApplication.instance() or QApplication([])


def _split_frame() -> np.ndarray:
    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    return frame


def _built_cell_position(tmp_path, name, condition):
    pos = tmp_path / name
    pos.mkdir()
    cell_path = pos / "cells.tif"
    tifffile.imwrite(cell_path, np.stack([_split_frame(), _split_frame()]))
    q = CellShapeQuantifier()
    inputs = PositionInputs(position_dir=pos, cell_labels_path=cell_path, pixel_size_um=1.0)
    q.build(inputs, q.default_output(inputs))
    return {
        "position_path": pos,
        "cell_tracked_labels_path": cell_path,
        "condition": condition,
        "date": "d1",
        "id": name,
    }


def test_shape_plots_register_under_shape_family():
    plots = {p.plot_id: p for p in available_plots()}
    for cls in (CellShapePlot, NucleusShapePlot, ShapeRelationalPlot):
        assert plots.get(cls.plot_id) is cls
        assert cls.family == "Shape"
    # Each shape plot consumes exactly its own product.
    assert CellShapePlot.consumes == ("cell_shape",)
    assert NucleusShapePlot.consumes == ("nucleus_shape",)
    assert ShapeRelationalPlot.consumes == ("shape_relational",)


def test_pool_quantity_pools_built_tables_with_metadata(tmp_path):
    records = [
        _built_cell_position(tmp_path, "p1", "A"),
        _built_cell_position(tmp_path, "p2", "B"),
    ]
    df = pool_quantity("cell_shape", records)
    assert not df.empty
    # Metadata stamped, NLS class_label defaulted, descriptor columns present.
    assert {"condition", "date", "position_id", "class_label"} <= set(df.columns)
    assert set(df["condition"]) == {"A", "B"}
    assert (df["class_label"] == "unclassified").all()
    assert set(DESCRIPTOR_COLUMNS) & set(df.columns)


def test_pool_quantity_skips_unbuilt_positions(tmp_path):
    built = _built_cell_position(tmp_path, "p1", "A")
    unbuilt = {"position_path": tmp_path / "empty", "id": "p2", "condition": "B", "date": "d1"}
    (tmp_path / "empty").mkdir()
    df = pool_quantity("cell_shape", [built, unbuilt])
    assert set(df["position_id"]) == {"p1"}


def test_pool_plot_exposes_single_quantity():
    assert CellShapePlot().quantity_id == "cell_shape"


def test_create_panel_builds_a_plot_panel(tmp_path):
    from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel
    from cellflow.napari.aggregate_quantification.plots import PlotContext

    app = _app()
    record = _built_cell_position(tmp_path, "p1", "A")
    ctx = PlotContext(records=[record], viewer=None)
    panel = CellShapePlot().create_panel(ctx)
    try:
        assert isinstance(panel, PlotPanel)
    finally:
        panel.deleteLater()
        app.processEvents()


def test_create_panel_accepts_prepooled_frame(tmp_path):
    app = _app()
    record = _built_cell_position(tmp_path, "p1", "A")
    from cellflow.napari.aggregate_quantification.plots import PlotContext

    plot = CellShapePlot()
    pooled = plot.prepare([record])
    ctx = PlotContext(records=[record], viewer=None)
    panel = plot.create_panel(ctx, prepared=pooled)
    try:
        assert panel is not None
    finally:
        panel.deleteLater()
        app.processEvents()
