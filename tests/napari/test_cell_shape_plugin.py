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
    inputs = PositionInputs(position_dir=pos, cell_labels_path=cell_path)
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
    assert plugin._build_btn.isEnabled() is True

    plugin._overwrite_cb.setChecked(True)
    plugin._on_build()
    assert len(captured) == 1
    quantifier, recs, overwrite = captured[0]
    assert quantifier.quantity_id == "cell_shape"
    assert [r["id"] for r in recs] == ["p1"]
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


def test_pool_records_reads_built_positions(tmp_path):
    records = [
        _built_position(tmp_path, "p1", "A"),
        _built_position(tmp_path, "p2", "B"),
    ]
    pooled = _pool_records(CellShapeQuantifier(), ContactsQuantifier(), records, split=False)

    assert len(pooled) == 8  # 2 cells x 2 frames x 2 positions
    assert set(pooled["condition"]) == {"A", "B"}
    assert {"area", "circularity", "position_id", "frame", "cell_id"} <= set(pooled.columns)


def test_pool_records_skips_unbuilt_positions(tmp_path):
    built = _built_position(tmp_path, "p1", "A")
    missing = {"position_path": tmp_path / "nope", "id": "p2", "condition": "A", "date": "d1"}
    pooled = _pool_records(CellShapeQuantifier(), ContactsQuantifier(), [built, missing], split=False)
    assert set(pooled["position_id"]) == {"p1"}


def test_render_embeds_a_canvas_and_enables_exports(tmp_path):
    app = _app()
    records = [_built_position(tmp_path, "p1", "A")]
    plugin = CellShapePlugin()
    plugin.set_context(AnalysisContext(records=records))

    # Drive the pool synchronously (skip the worker) then render.
    plugin._pooled = _pool_records(
        CellShapeQuantifier(), ContactsQuantifier(), records, split=False
    )
    plugin._pool_signature = plugin._scope_signature()
    plugin._render()
    plugin._update_enabled()

    assert plugin._canvas is not None
    assert plugin._export_pooled_btn.isEnabled() is True
    assert plugin._export_agg_btn.isEnabled() is True
    # One position in scope -> per-position export is available.
    assert plugin._export_position_btn.isEnabled() is True

    plugin.deleteLater()
    app.processEvents()


def test_set_context_invalidates_pool_cache(tmp_path):
    app = _app()
    records = [_built_position(tmp_path, "p1", "A")]
    plugin = CellShapePlugin()
    plugin._pooled = _pool_records(CellShapeQuantifier(), ContactsQuantifier(), records, split=False)
    plugin._pool_signature = plugin._scope_signature()

    plugin.set_context(AnalysisContext(records=records))
    assert plugin._pooled is None  # scope (re)assignment forces a re-pool
    plugin.deleteLater()
    app.processEvents()
