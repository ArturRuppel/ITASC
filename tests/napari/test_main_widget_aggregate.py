"""The aggregate capstone is wired into the full CellFlow app."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari.qt import get_qapp
from cellflow.napari.main_widget import CellFlowMainWidget


def _fake_viewer():
    class _Sel:
        active = None

    class _Layers(dict):
        selection = _Sel()
        events = SimpleNamespace(removed=SimpleNamespace(connect=lambda cb: None))

        def remove(self, layer):
            self.pop(layer.name, None)

    viewer = SimpleNamespace()
    viewer.layers = _Layers()
    viewer.dims = SimpleNamespace(
        current_step=(0, 0),
        events=SimpleNamespace(current_step=SimpleNamespace(connect=lambda cb: None)),
    )
    viewer.add_image = MagicMock()
    viewer.add_labels = MagicMock()
    viewer.add_shapes = MagicMock()
    viewer.bind_key = MagicMock()
    return viewer


def test_app_has_aggregate_section_after_results():
    get_qapp()
    w = CellFlowMainWidget(_fake_viewer())
    assert hasattr(w, "aggregate_widget")
    assert hasattr(w, "aggregate_section")
    # The capstone is the last section in the stage stack.
    layout = w.scroll_layout
    order = [layout.itemAt(i).widget() for i in range(layout.count())]
    assert order.index(w.aggregate_section) > order.index(w.contact_analysis_section)


def test_catalog_records_helper_feeds_aggregate(tmp_path):
    get_qapp()
    w = CellFlowMainWidget(_fake_viewer())
    records = w._catalog_records_for_panel([
        {"position_path": tmp_path / "posA", "columns": {"condition": "ctrl", "id": "posA"}},
    ])
    assert records[0]["contact_analysis_path"].name == "contact_analysis.h5"
