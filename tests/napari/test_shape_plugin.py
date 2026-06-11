from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import tifffile
from qtpy.QtWidgets import QApplication

from cellflow.aggregate_quantification.quantifier import PositionInputs
from cellflow.aggregate_quantification.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.aggregate_quantification.quantifiers.shape_relational import (
    ShapeRelationalQuantifier,
)
from cellflow.aggregate_quantification.shape import DESCRIPTOR_COLUMNS, RELATIONAL_COLUMNS
from cellflow.napari.aggregate_quantification.plugins import AnalysisContext
from cellflow.napari.aggregate_quantification.plugins.shape import (
    ShapePlugin,
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


class _FakeLabelsLayer:
    def __init__(self, data, name) -> None:
        self.data, self.name = data, name
        self.selected_label = None
        self.show_selected_label = False


class _FakeDims:
    def set_current_step(self, axis, value) -> None:
        pass


class _FakeCamera:
    center = None


class _FakeViewer:
    def __init__(self) -> None:
        self.window = _FakeWindow()
        self.layers: list = []
        self.dims = _FakeDims()
        self.camera = _FakeCamera()

    def add_labels(self, data, name=None):
        layer = _FakeLabelsLayer(data, name)
        self.layers.append(layer)
        return layer


def _split_frame() -> np.ndarray:
    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    return frame


def _built_cell_position(tmp_path, name, condition):
    """Build a cell_shape.csv under a position dir and return its record."""
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


def _both_labels_position(tmp_path, name):
    pos = tmp_path / name
    pos.mkdir()
    cell_path = pos / "cells.tif"
    nuc_path = pos / "nuclei.tif"
    tifffile.imwrite(cell_path, _split_frame())
    tifffile.imwrite(nuc_path, _split_frame())
    return {
        "position_path": pos,
        "cell_tracked_labels_path": cell_path,
        "nucleus_tracked_labels_path": nuc_path,
        "condition": "A",
        "date": "d1",
        "id": name,
    }


# --------------------------------------------------------------------- scope rule
def test_default_scope_prefers_both_when_a_position_has_both_labels(tmp_path):
    app = _app()
    plugin = ShapePlugin()
    plugin.set_context(AnalysisContext(records=[_both_labels_position(tmp_path, "p1")]))
    assert plugin._scope == "both"
    plugin.deleteLater()
    app.processEvents()


def test_default_scope_falls_back_to_single_present_source(tmp_path):
    app = _app()
    plugin = ShapePlugin()
    plugin.set_context(
        AnalysisContext(records=[{"id": "p1", "nucleus_tracked_labels_path": tmp_path / "n.tif"}])
    )
    assert plugin._scope == "nucleus"

    plugin2 = ShapePlugin()
    plugin2.set_context(
        AnalysisContext(records=[{"id": "p1", "cell_tracked_labels_path": tmp_path / "c.tif"}])
    )
    assert plugin2._scope == "cell"
    plugin.deleteLater()
    plugin2.deleteLater()
    app.processEvents()


# ----------------------------------------------------------------------- compute
def test_build_button_forwards_scope_selected_quantifier(tmp_path):
    app = _app()
    captured: list = []
    plugin = ShapePlugin()
    plugin.set_build_callback(lambda q, recs, ow: captured.append((q, recs, ow)))

    record = {"id": "p1", "cell_tracked_labels_path": tmp_path / "c.tif", "position_path": tmp_path}
    plugin.set_context(AnalysisContext(records=[record]))
    # No pixel size resolvable -> blocked until set.
    assert plugin._build_btn.isEnabled() is False
    plugin._pixel_size_edit.setText("0.5")
    assert plugin._build_btn.isEnabled() is True

    plugin._on_build()
    assert captured[-1][0].quantity_id == "cell_shape"
    assert captured[-1][1][0]["pixel_size_um"] == 0.5  # typed override stamped

    plugin.deleteLater()
    app.processEvents()


def test_build_button_uses_nucleus_quantifier_when_scope_is_nucleus(tmp_path):
    app = _app()
    captured: list = []
    plugin = ShapePlugin()
    plugin.set_build_callback(lambda q, recs, ow: captured.append((q, recs, ow)))
    record = {"id": "p1", "nucleus_tracked_labels_path": tmp_path / "n.tif", "position_path": tmp_path}
    plugin.set_context(AnalysisContext(records=[record]))  # defaults to nucleus
    plugin._pixel_size_edit.setText("0.5")
    assert plugin._scope == "nucleus"
    plugin._on_build()
    assert captured[-1][0].quantity_id == "nucleus_shape"

    plugin.deleteLater()
    app.processEvents()


def test_build_button_disabled_without_callback_or_labels(tmp_path):
    app = _app()
    plugin = ShapePlugin()
    plugin.set_context(
        AnalysisContext(records=[{"id": "p1", "cell_tracked_labels_path": tmp_path / "c.tif"}])
    )
    assert plugin._build_btn.isEnabled() is False  # no callback yet

    plugin.set_build_callback(lambda *a: None)
    plugin.set_context(AnalysisContext(records=[{"id": "p2"}]))  # no labels of any kind
    assert plugin._build_btn.isEnabled() is False
    plugin.deleteLater()
    app.processEvents()


# -------------------------------------------------------------------- pooling
def test_pool_records_always_joins_class_label(tmp_path):
    records = [
        _built_cell_position(tmp_path, "p1", "A"),
        _built_cell_position(tmp_path, "p2", "B"),
    ]
    pooled = _pool_records(CellShapeQuantifier(), records)

    assert len(pooled) == 8  # 2 cells x 2 frames x 2 positions
    assert set(pooled["condition"]) == {"A", "B"}
    assert {"area_um2", "circularity", "position_id", "frame", "cell_id", "class_label"} <= set(
        pooled.columns
    )
    assert set(pooled["class_label"]) == {"unclassified"}


def test_pool_records_joins_class_label_from_nls_csv(tmp_path):
    from cellflow.aggregate_quantification.contacts.nls_classification import (
        nls_classification_csv_path,
        write_nls_classification_csv,
    )

    record = _built_cell_position(tmp_path, "p1", "A")
    write_nls_classification_csv(
        nls_classification_csv_path(record["position_path"]),
        {1: "positive", 2: "negative"},
        positive_label="GFP+",
        negative_label="GFP-",
    )
    pooled = _pool_records(CellShapeQuantifier(), [record])

    # Each track's label is broadcast across its frames, keyed by cell_id.
    labels = pooled.groupby("cell_id")["class_label"].agg(set).to_dict()
    assert labels[1] == {"GFP+"}
    assert labels[2] == {"GFP-"}


def test_pool_records_skips_unbuilt_positions(tmp_path):
    built = _built_cell_position(tmp_path, "p1", "A")
    missing = {"position_path": tmp_path / "nope", "id": "p2", "condition": "A", "date": "d1"}
    pooled = _pool_records(CellShapeQuantifier(), [built, missing])
    assert set(pooled["position_id"]) == {"p1"}


# ----------------------------------------------------------------------- plotting
def test_plot_button_enabled_only_when_built_and_viewer(tmp_path):
    app = _app()
    records = [_built_cell_position(tmp_path, "p1", "A")]
    plugin = ShapePlugin()
    plugin.set_context(AnalysisContext(records=records))  # no viewer
    assert plugin._plot_btn.isEnabled() is False

    plugin.set_context(AnalysisContext(records=records, viewer=_FakeViewer()))
    assert plugin._plot_btn.isEnabled() is True
    plugin.deleteLater()
    app.processEvents()


def test_plots_share_one_dock_as_tabs(tmp_path):
    from qtpy.QtWidgets import QTabWidget

    app = _app()
    records = [_built_cell_position(tmp_path, "p1", "A")]
    viewer = _FakeViewer()
    plugin = ShapePlugin()
    plugin.set_context(AnalysisContext(records=records, viewer=viewer))

    pooled = _pool_records(CellShapeQuantifier(), records)
    plugin._on_pool_done(pooled)
    plugin._on_pool_done(pooled)

    # One dock holding a tab widget — each plot is a tab, not its own dock (which
    # made napari split the area and shrink every plot).
    assert len(viewer.window.docks) == 1
    widget, area, name = viewer.window.docks[0]
    assert area == "right"
    assert name == "Shape plots"
    assert isinstance(widget, QTabWidget)
    assert [widget.tabText(i) for i in range(widget.count())] == ["Plot 1", "Plot 2"]
    plugin.deleteLater()
    app.processEvents()


def test_plot_value_columns_follow_scope(tmp_path):
    app = _app()
    plugin = ShapePlugin()
    # default cell scope -> per-object descriptors.
    plugin.set_context(
        AnalysisContext(records=[{"id": "p1", "cell_tracked_labels_path": tmp_path / "c.tif"}])
    )
    assert plugin._value_columns() == DESCRIPTOR_COLUMNS
    # both scope -> strictly relational columns.
    plugin.set_context(AnalysisContext(records=[_both_labels_position(tmp_path, "p1")]))
    assert plugin._scope == "both"
    assert plugin._value_columns() == RELATIONAL_COLUMNS
    plugin.deleteLater()
    app.processEvents()


def test_relational_scope_pools_relational_table(tmp_path):
    record = _both_labels_position(tmp_path, "p1")
    q = ShapeRelationalQuantifier()
    inputs = PositionInputs(
        position_dir=record["position_path"],
        cell_labels_path=record["cell_tracked_labels_path"],
        nucleus_labels_path=record["nucleus_tracked_labels_path"],
        pixel_size_um=1.0,
    )
    q.build(inputs, q.default_output(inputs))

    pooled = _pool_records(q, [record])
    assert not pooled.empty
    assert {"nc_area_ratio", "centroid_offset_um", "cell_id", "frame"} <= set(pooled.columns)


def test_plot_empty_pool_reports_and_opens_no_dock(tmp_path):
    app = _app()
    viewer = _FakeViewer()
    plugin = ShapePlugin()
    plugin.set_context(AnalysisContext(records=[], viewer=viewer))

    import pandas as pd

    plugin._on_pool_done(pd.DataFrame())
    assert viewer.window.docks == []
    assert "No built" in plugin._plot_status.text()
    plugin.deleteLater()
    app.processEvents()


def test_shape_panel_gets_resolver_and_load_is_wired(tmp_path):
    from pathlib import Path

    app = _app()
    records = [_built_cell_position(tmp_path, "p1", "A")]
    viewer = _FakeViewer()
    plugin = ShapePlugin()
    plugin.set_context(AnalysisContext(records=records, viewer=viewer))

    pooled = _pool_records(CellShapeQuantifier(), records)
    plugin._on_pool_done(pooled)

    panel = plugin._panel
    assert panel._target_resolver is not None
    # The resolver maps a {position_id, cell_id} identity to the cell labels TIFF.
    target = panel._target_resolver({"position_id": "p1", "frame": 0, "cell_id": 1})
    assert target is not None
    assert target.path == Path(records[0]["cell_tracked_labels_path"])
    plugin.deleteLater()
    app.processEvents()


def test_load_button_loads_labels_into_viewer(tmp_path):
    """Clicking a point then "Load in viewer" must add the labels layer — even
    after a GC pass. Regression: the ClickToLoad controller was a local in
    _open_panel and PyQt's bound-method connections hold the receiver weakly,
    so it was collected and the Load click silently did nothing."""
    import gc

    app = _app()
    records = [_built_cell_position(tmp_path, "p1", "A")]
    viewer = _FakeViewer()
    plugin = ShapePlugin()
    plugin.set_context(AnalysisContext(records=records, viewer=viewer))

    plugin._on_pool_done(_pool_records(CellShapeQuantifier(), records))
    panel = plugin._panel
    gc.collect()

    panel._select_row(0)
    assert panel._load_btn.isEnabled()
    panel._load_btn.click()
    assert len(viewer.layers) == 1
    assert np.asarray(viewer.layers[0].data).max() == 2  # the labels, not a blank
    plugin.deleteLater()
    app.processEvents()
