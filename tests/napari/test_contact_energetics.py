from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QApplication

from cellflow.napari.aggregate_quantification.plugins import AnalysisContext
from cellflow.napari.aggregate_quantification.plugins.contact_energetics import (
    ContactEnergeticsPlugin,
    _is_built,
    _pool_energetics,
)


def _app():
    return QApplication.instance() or QApplication([])


class _FakeWindow:
    def __init__(self) -> None:
        self.docks: list[tuple] = []

    def add_dock_widget(self, widget, *, area=None, name=None):
        self.docks.append((widget, area, name))
        return object()


class _FakeViewer:
    def __init__(self) -> None:
        self.window = _FakeWindow()
        self.layers: list = []


def _write_contacts_h5(path, edges_rows, *, losing=(1, 2), gaining=(3, 4)) -> None:
    """A minimal contact_analysis.h5 matching the reader schema."""
    frame, a, b, length = zip(*edges_rows) if edges_rows else ((), (), (), ())
    with h5py.File(path, "w") as h5:
        prov = h5.create_group("provenance")
        prov.attrs["cell_tracked_labels_path"] = "cells.tif"
        prov.attrs["nucleus_tracked_labels_path"] = "nuclei.tif"
        h5.create_group("cells/table")
        edges = h5.create_group("edges/table")
        edges.create_dataset("frame", data=np.asarray(frame, dtype=np.int64))
        edges.create_dataset("cell_a", data=np.asarray(a, dtype=np.int64))
        edges.create_dataset("cell_b", data=np.asarray(b, dtype=np.int64))
        edges.create_dataset("length", data=np.asarray(length, dtype=float))
        events = h5.create_group("t1_events/table")
        events.create_dataset("t1_event_id", data=np.array([7], dtype=np.int64))
        events.create_dataset("losing_cell_a", data=np.array([losing[0]], dtype=np.int64))
        events.create_dataset("losing_cell_b", data=np.array([losing[1]], dtype=np.int64))
        events.create_dataset("gaining_cell_a", data=np.array([gaining[0]], dtype=np.int64))
        events.create_dataset("gaining_cell_b", data=np.array([gaining[1]], dtype=np.int64))
        h5.create_dataset("edges/coordinates/y", data=np.empty(0))
        h5.create_dataset("edges/coordinates/x", data=np.empty(0))


def _record(tmp_path, name, condition):
    pos = tmp_path / name
    pos.mkdir()
    h5_path = pos / "contact_analysis.h5"
    _write_contacts_h5(
        h5_path,
        [(0, 1, 2, 5.0), (1, 1, 2, 2.0), (2, 3, 4, 2.0), (3, 3, 4, 5.0)],
    )
    return {
        "position_path": pos,
        "contact_analysis_path": h5_path,
        "condition": condition,
        "date": "d1",
        "id": name,
    }


def test_pool_signs_and_pools_across_positions(tmp_path):
    records = [_record(tmp_path, "p1", "A"), _record(tmp_path, "p2", "B")]
    pooled = _pool_energetics(records, None)
    assert {"signed_length", "condition", "date", "position_id"} <= set(pooled.columns)
    assert set(pooled["condition"]) == {"A", "B"}
    # Losing edge frames are negative, gaining frames positive.
    losing = pooled[pooled["role"] == "losing"]["signed_length"]
    gaining = pooled[pooled["role"] == "gaining"]["signed_length"]
    assert (losing < 0).all() and (gaining > 0).all()
    # Two positions × four samples each.
    assert len(pooled) == 8


def test_pixel_override_scales_signed_length(tmp_path):
    records = [_record(tmp_path, "p1", "A")]
    pooled = _pool_energetics(records, 0.5)
    # length 5.0 at the longest losing frame → −2.5 µm under 0.5 µm/px.
    assert pooled["signed_length"].min() == -2.5
    assert pooled["signed_length"].max() == 2.5


def test_is_built_false_for_missing_artifact(tmp_path):
    assert _is_built({"contact_analysis_path": tmp_path / "nope.h5"}) is False
    assert _is_built({}) is False


def test_plot_enabled_only_when_built_and_viewer(tmp_path):
    app = _app()
    records = [_record(tmp_path, "p1", "A")]
    plugin = ContactEnergeticsPlugin()
    plugin.set_context(AnalysisContext(records=records))
    assert plugin._plot_btn.isEnabled() is False  # no viewer
    plugin.set_context(AnalysisContext(records=records, viewer=_FakeViewer()))
    assert plugin._plot_btn.isEnabled() is True
    plugin.deleteLater()
    app.processEvents()


def test_on_pool_done_opens_potential_panel(tmp_path):
    app = _app()
    records = [_record(tmp_path, "p1", "A")]
    viewer = _FakeViewer()
    plugin = ContactEnergeticsPlugin()
    plugin.set_context(AnalysisContext(records=records, viewer=viewer))

    pooled = _pool_energetics(records, None)
    plugin._on_pool_done(pooled)

    assert len(viewer.window.docks) == 1
    panel = plugin._panel
    # The panel opens straight into the potential view over the signed coordinate.
    assert panel._plot_combo.currentData() == "potential"
    assert "signed_length" in panel._value_columns
    plugin.deleteLater()
    app.processEvents()


def test_empty_pool_opens_no_dock(tmp_path):
    app = _app()
    viewer = _FakeViewer()
    plugin = ContactEnergeticsPlugin()
    plugin.set_context(AnalysisContext(records=[], viewer=viewer))
    plugin._on_pool_done(pd.DataFrame())
    assert viewer.window.docks == []
    assert "No T1 junction lengths" in plugin._plot_status.text()
    plugin.deleteLater()
    app.processEvents()
