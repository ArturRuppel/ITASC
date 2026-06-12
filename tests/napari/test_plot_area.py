"""The Plot area: render-type buttons, availability gating, spanning catalog."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import tifffile
from qtpy.QtWidgets import QApplication

from cellflow.aggregate_quantification.quantifier import PositionInputs
from cellflow.aggregate_quantification.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.aggregate_quantification.quantifiers.nucleus_shape import (
    NucleusShapeQuantifier,
)
from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel
from cellflow.napari.aggregate_quantification.plots import PlotParams
from cellflow.napari.aggregate_quantification.plugins import AnalysisContext
from cellflow.napari.aggregate_quantification_plot_area import PlotAreaWidget


def _app():
    return QApplication.instance() or QApplication([])


def _button_for(area: PlotAreaWidget, key: str):
    for button, button_key in area._buttons.items():
        if button_key == key:
            return button
    raise KeyError(key)


def _split_frame() -> np.ndarray:
    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    return frame


def _built_shape_record(tmp_path, *, nucleus=False):
    pos = tmp_path / ("nuc" if nucleus else "cell")
    pos.mkdir()
    path = pos / "labels.tif"
    tifffile.imwrite(path, np.stack([_split_frame(), _split_frame()]))
    q = NucleusShapeQuantifier() if nucleus else CellShapeQuantifier()
    field = "nucleus_labels_path" if nucleus else "cell_labels_path"
    inputs = PositionInputs(position_dir=pos, pixel_size_um=1.0, **{field: path})
    q.build(inputs, q.default_output(inputs))
    record_field = "nucleus_tracked_labels_path" if nucleus else "cell_tracked_labels_path"
    return {
        "position_path": pos,
        record_field: path,
        "condition": "A",
        "date": "d1",
        "id": pos.name,
    }


def test_plot_and_curve_buttons_present():
    app = _app()
    area = PlotAreaWidget(viewer=object())
    try:
        # Distribution / bar / potential collapsed into one "plot" button; curves
        # stays its own button.
        for key in ("plot", "curve"):
            assert _button_for(area, key) is not None
        assert len(area._buttons) == 2
    finally:
        area.deleteLater()
        app.processEvents()


def test_no_built_products_disable_all_buttons():
    app = _app()
    area = PlotAreaWidget(viewer=object())
    try:
        area.set_context(AnalysisContext(records=[], viewer=object()))
        for button in area._buttons:
            assert button.isEnabled() is False
            assert "built" in button.toolTip()
    finally:
        area.deleteLater()
        app.processEvents()


def test_built_shape_enables_plot_only(tmp_path):
    app = _app()
    area = PlotAreaWidget(viewer=object())
    try:
        record = _built_shape_record(tmp_path)
        area.set_context(AnalysisContext(records=[record], viewer=object()))
        # Shape is a tidy product → the Plot button; no dynamics → Curves stays off.
        assert _button_for(area, "plot").isEnabled() is True
        assert _button_for(area, "curve").isEnabled() is False
    finally:
        area.deleteLater()
        app.processEvents()


def test_catalog_panel_spans_values_grouped_by_source(tmp_path):
    app = _app()
    area = PlotAreaWidget(viewer=object())
    try:
        cell = _built_shape_record(tmp_path)
        nucleus = _built_shape_record(tmp_path, nucleus=True)
        area.set_context(AnalysisContext(records=[cell, nucleus], viewer=object()))
        plots = area._available_for_button("plot")
        prepared = [(p, p.prepare([cell, nucleus])) for p in plots]
        panel = area._build_catalog_panel(prepared, [cell, nucleus])
        assert isinstance(panel, PlotPanel)
        # The value picker carries entries from both cell and nucleus shape, with
        # the family header ("Shape") shown as a disabled separator.
        labels = [
            panel._value_combo.itemText(i) for i in range(panel._value_combo.count())
        ]
        assert any("── Shape ──" in t for t in labels)
        assert any("Cell shape" in t for t in labels)
        assert any("Nucleus shape" in t for t in labels)
        # Click-to-load is wired: the active shape source has a resolver.
        assert panel._target_resolver is not None
        panel.deleteLater()
    finally:
        area.deleteLater()
        app.processEvents()


def test_plot_area_uses_injected_params_provider():
    app = _app()
    sentinel = PlotParams(pixel_size_um=0.5, shuffles=42)
    area = PlotAreaWidget(viewer=object(), params_provider=lambda: sentinel)
    try:
        assert area._params() is sentinel
    finally:
        area.deleteLater()
        app.processEvents()


def test_plot_area_defaults_params_without_a_provider():
    app = _app()
    area = PlotAreaWidget(viewer=object())
    try:
        assert area._params() == PlotParams()
    finally:
        area.deleteLater()
        app.processEvents()
