"""The Plot area: family grouping + product-availability gating."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import tifffile
from qtpy.QtWidgets import QApplication

from cellflow.aggregate_quantification.quantifier import PositionInputs
from cellflow.aggregate_quantification.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.napari.aggregate_quantification.plots import PlotParams
from cellflow.napari.aggregate_quantification.plugins import AnalysisContext
from cellflow.napari.aggregate_quantification_plot_area import PlotAreaWidget


def _app():
    return QApplication.instance() or QApplication([])


def _button_for(area: PlotAreaWidget, plot_id: str):
    for button, plot in area._buttons.items():
        if plot.plot_id == plot_id:
            return button
    raise KeyError(plot_id)


def _built_cell_shape_record(tmp_path):
    pos = tmp_path / "p1"
    pos.mkdir()
    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    cell_path = pos / "cells.tif"
    tifffile.imwrite(cell_path, np.stack([frame, frame]))
    q = CellShapeQuantifier()
    inputs = PositionInputs(position_dir=pos, cell_labels_path=cell_path, pixel_size_um=1.0)
    q.build(inputs, q.default_output(inputs))
    return {
        "position_path": pos,
        "cell_tracked_labels_path": cell_path,
        "condition": "A",
        "date": "d1",
        "id": "p1",
    }


def test_plot_area_lists_every_registered_plot():
    app = _app()
    area = PlotAreaWidget(viewer=object())
    try:
        from cellflow.napari.aggregate_quantification.plots import available_plots

        assert len(area._buttons) == len(available_plots())
        # Families used as group headers cover shape / dynamics / contacts.
        families = {plot.family for plot in area._buttons.values()}
        assert {"Shape", "Dynamics", "Contacts"} <= families
    finally:
        area.deleteLater()
        app.processEvents()


def test_unbuilt_products_disable_their_plots_with_a_reason():
    app = _app()
    area = PlotAreaWidget(viewer=object())
    try:
        area.set_context(AnalysisContext(records=[], viewer=object()))
        button = _button_for(area, "cell_shape")
        assert button.isEnabled() is False
        assert "cell_shape" in button.toolTip()
    finally:
        area.deleteLater()
        app.processEvents()


def test_shared_param_fields_build_plot_params():
    app = _app()
    area = PlotAreaWidget(viewer=object())
    try:
        # Blank fields → all auto / default.
        assert area._current_params() == PlotParams()
        area._pixel_size_edit.setText("0.25")
        area._fov_edit.setText("1.5")
        area._shuffles_edit.setText("200")
        params = area._current_params()
        assert params.pixel_size_um == 0.25
        assert params.fov_area_mm2 == 1.5
        assert params.shuffles == 200
        # Invalid entries fall back to auto / default rather than raising.
        area._pixel_size_edit.setText("nope")
        area._shuffles_edit.setText("0")
        params = area._current_params()
        assert params.pixel_size_um is None
        assert params.shuffles == PlotParams().shuffles
    finally:
        area.deleteLater()
        app.processEvents()


def test_building_a_product_enables_its_plot(tmp_path):
    app = _app()
    area = PlotAreaWidget(viewer=object())
    try:
        record = _built_cell_shape_record(tmp_path)
        area.set_context(AnalysisContext(records=[record], viewer=object()))
        # The cell_shape plot lights up; nucleus_shape (not built) stays disabled.
        assert _button_for(area, "cell_shape").isEnabled() is True
        nucleus_btn = _button_for(area, "nucleus_shape")
        assert nucleus_btn.isEnabled() is False
        assert "nucleus_shape" in nucleus_btn.toolTip()
    finally:
        area.deleteLater()
        app.processEvents()
