from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import tifffile
from qtpy.QtWidgets import QApplication

from cellflow.aggregate_quantification.quantifier import PositionInputs
from cellflow.aggregate_quantification.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.aggregate_quantification.quantifiers.contacts import ContactsQuantifier
from cellflow.napari.aggregate_quantification.plugins import AnalysisContext
from cellflow.napari.aggregate_quantification.plugins.cell_shape import (
    CellShapePlugin,
    _pool_records,
)


def _app():
    return QApplication.instance() or QApplication([])


class _FakeWindow:
    def __init__(self) -> None:
        self.docks: list[tuple] = []

    def add_dock_widget(self, widget, *, area=None, name=None):  # noqa: D401
        self.docks.append((widget, area, name))
        return object()


class _FakeViewer:
    def __init__(self) -> None:
        self.window = _FakeWindow()


def _built_position(tmp_path, name, condition):
    """Build cell_shape.h5 in a position dir and return its catalogue record."""
    pos = tmp_path / name
    pos.mkdir()
    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    cell_path = pos / "cells.tif"
    tifffile.imwrite(cell_path, np.stack([frame, frame]))

    q = CellShapeQuantifier()
    inputs = PositionInputs(
        position_dir=pos, cell_labels_path=cell_path, pixel_size_um=1.0
    )
    q.build(inputs, q.default_output(inputs))
    return {
        "position_path": pos,
        "cell_tracked_labels_path": cell_path,
        "condition": condition,
        "date": "d1",
        "id": name,
    }


def test_build_button_forwards_to_studio_callback(tmp_path):
    app = _app()
    captured: list = []
    plugin = CellShapePlugin()
    plugin.set_build_callback(lambda q, recs, ow: captured.append((q, recs, ow)))

    record = {"id": "p1", "cell_tracked_labels_path": tmp_path / "c.tif", "position_path": tmp_path}
    plugin.set_context(AnalysisContext(records=[record]))
    # No pixel size resolvable (no config / TIFF tags) -> blocked until set.
    assert plugin._build_btn.isEnabled() is False
    plugin._pixel_size_edit.setText("0.5")
    assert plugin._build_btn.isEnabled() is True

    plugin._overwrite_cb.setChecked(True)
    plugin._on_build()
    assert len(captured) == 1
    quantifier, recs, overwrite = captured[0]
    assert quantifier.quantity_id == "cell_shape"
    assert [r["id"] for r in recs] == ["p1"]
    # The typed override is stamped onto the records handed to the build callback.
    assert recs[0]["pixel_size_um"] == 0.5
    assert overwrite is True

    plugin.deleteLater()
    app.processEvents()


def test_build_button_disabled_without_callback_or_labels(tmp_path):
    app = _app()
    plugin = CellShapePlugin()
    # No build callback yet -> disabled even with a buildable record.
    plugin.set_context(
        AnalysisContext(records=[{"id": "p1", "cell_tracked_labels_path": tmp_path / "c.tif"}])
    )
    assert plugin._build_btn.isEnabled() is False

    plugin.set_build_callback(lambda *a: None)
    # Callback present but no position has cell labels -> still disabled.
    plugin.set_context(AnalysisContext(records=[{"id": "p2"}]))
    assert plugin._build_btn.isEnabled() is False
    plugin.deleteLater()
    app.processEvents()


def test_pool_records_always_joins_class_label(tmp_path):
    """Pooling now always carries a ``class_label`` column; positions without a
    contacts artifact fall into the ``unclassified`` bucket."""
    records = [
        _built_position(tmp_path, "p1", "A"),
        _built_position(tmp_path, "p2", "B"),
    ]
    pooled = _pool_records(CellShapeQuantifier(), ContactsQuantifier(), records)

    assert len(pooled) == 8  # 2 cells x 2 frames x 2 positions
    assert set(pooled["condition"]) == {"A", "B"}
    assert {"area_um2", "circularity", "position_id", "frame", "cell_id", "class_label"} <= set(
        pooled.columns
    )
    assert set(pooled["class_label"]) == {"unclassified"}


def test_pool_records_skips_unbuilt_positions(tmp_path):
    built = _built_position(tmp_path, "p1", "A")
    missing = {"position_path": tmp_path / "nope", "id": "p2", "condition": "A", "date": "d1"}
    pooled = _pool_records(CellShapeQuantifier(), ContactsQuantifier(), [built, missing])
    assert set(pooled["position_id"]) == {"p1"}


def test_plot_button_enabled_only_when_built_and_viewer(tmp_path):
    app = _app()
    records = [_built_position(tmp_path, "p1", "A")]
    plugin = CellShapePlugin()
    # No viewer -> can't dock -> disabled even with a built position.
    plugin.set_context(AnalysisContext(records=records))
    assert plugin._plot_btn.isEnabled() is False

    plugin.set_context(AnalysisContext(records=records, viewer=_FakeViewer()))
    assert plugin._plot_btn.isEnabled() is True

    plugin.deleteLater()
    app.processEvents()


def test_plot_opens_a_new_dock_per_click(tmp_path):
    app = _app()
    records = [_built_position(tmp_path, "p1", "A")]
    viewer = _FakeViewer()
    plugin = CellShapePlugin()
    plugin.set_context(AnalysisContext(records=records, viewer=viewer))

    pooled = _pool_records(CellShapeQuantifier(), ContactsQuantifier(), records)
    # Drive the post-pool path directly (skip the worker thread).
    plugin._on_pool_done(pooled)
    plugin._on_pool_done(pooled)

    assert len(viewer.window.docks) == 2
    names = [name for _, _, name in viewer.window.docks]
    assert names == ["Cell shape plot 1", "Cell shape plot 2"]
    areas = {area for _, area, _ in viewer.window.docks}
    assert areas == {"right"}

    plugin.deleteLater()
    app.processEvents()


def test_plot_empty_pool_reports_and_opens_no_dock(tmp_path):
    app = _app()
    viewer = _FakeViewer()
    plugin = CellShapePlugin()
    plugin.set_context(AnalysisContext(records=[], viewer=viewer))

    import pandas as pd

    plugin._on_pool_done(pd.DataFrame())
    assert viewer.window.docks == []
    assert "No built" in plugin._plot_status.text()

    plugin.deleteLater()
    app.processEvents()
