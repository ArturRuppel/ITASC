"""The aggregate capstone is wired into the full ITASC app."""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari.qt import get_qapp
from itasc.napari.main_widget import ITASCMainWidget


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
    w = ITASCMainWidget(_fake_viewer())
    assert hasattr(w, "aggregate_widget")
    assert hasattr(w, "aggregate_section")
    # The capstone is the last section in the stage stack.
    layout = w.scroll_layout
    order = [layout.itemAt(i).widget() for i in range(layout.count())]
    assert order.index(w.aggregate_section) > order.index(w.contact_analysis_section)


def test_catalog_records_helper_feeds_aggregate(tmp_path):
    get_qapp()
    w = ITASCMainWidget(_fake_viewer())
    records = w._catalog_records_for_panel([
        {"position_path": tmp_path / "posA", "columns": {"condition": "ctrl", "id": "posA"}},
    ])
    assert records[0]["contact_analysis_path"].name == "contact_analysis.h5"


def test_catalog_records_stamp_setup_calibration(tmp_path):
    """Setup calibration (pixel size / frame length) rides onto the aggregate
    records, so the pooled shape/dynamics quantities can compute in physical units
    instead of staying greyed."""
    get_qapp()
    w = ITASCMainWidget(_fake_viewer())
    # Frame length is entered in minutes; the record carries backend seconds.
    w._positions_panel.set_calibration_values(
        {"pixel_size_um": "0.25", "time_interval_min": "2.0"}
    )
    records = w._catalog_records_for_panel([
        {"position_path": tmp_path / "posA", "columns": {"condition": "ctrl", "id": "posA"}},
    ])
    assert records[0]["pixel_size_um"] == 0.25
    assert records[0]["time_interval_s"] == 120.0


def test_catalog_records_omit_blank_calibration(tmp_path):
    """A blank calibration field contributes no key (not a zero that would fail the
    positive-value gate anyway)."""
    get_qapp()
    w = ITASCMainWidget(_fake_viewer())
    w._positions_panel.set_calibration_values({"pixel_size_um": "", "time_interval_min": ""})
    records = w._catalog_records_for_panel([
        {"position_path": tmp_path / "posA", "columns": {"id": "posA"}},
    ])
    assert "pixel_size_um" not in records[0]
    assert "time_interval_s" not in records[0]


def test_aggregate_scope_band_tracks_section_visibility(tmp_path):
    """The 'All positions' scope band re-parents Aggregate to the catalog scope.
    It exists, starts hidden with the section, and appears alongside it once
    positions are added — so Aggregate never reads as a fifth per-position stage."""
    get_qapp()
    w = ITASCMainWidget(_fake_viewer())
    assert hasattr(w, "aggregate_scope_band")
    # Hidden until positions exist (mirrors the aggregate section).
    assert not w.aggregate_scope_band.isVisibleTo(w)
    # The de-staged Aggregate section carries no explicit stage accent.
    assert w.aggregate_section._explicit_accent is None
    w._positions_panel.set_records([
        {"key": str(tmp_path / "posA"),
         "columns": {"id": "posA"},
         "payload": {"position_path": str(tmp_path / "posA"), "id": "posA"}},
    ])
    w._refresh_aggregate()
    assert w.aggregate_scope_band.isVisibleTo(w)
    assert w.aggregate_section.isVisibleTo(w)


def test_calibration_edit_refreshes_aggregate():
    """Editing a calibration field re-stamps the aggregate records, so quantities
    gated on a param filled in *after* folders were added stop being greyed."""
    get_qapp()
    w = ITASCMainWidget(_fake_viewer())
    calls = []
    w._refresh_aggregate = lambda: calls.append(1)  # lambda in __init__ looks this up at emit
    w._positions_panel.calibration_changed.emit("pixel_size_um", "0.5")
    assert calls == [1]


def test_frame_length_minutes_persist_as_backend_seconds():
    """The Setup frame-length field is minutes; the config stores seconds and a
    seconds config loads back into the field as minutes."""
    get_qapp()
    w = ITASCMainWidget(_fake_viewer())
    w._positions_panel.set_calibration_values(
        {"pixel_size_um": "0.3", "time_interval_min": "5"}
    )
    assert w.get_state()["metadata"]["time_interval_s"] == "300"

    w2 = ITASCMainWidget(_fake_viewer())
    w2.set_state({"metadata": {"pixel_size_um": "0.3", "time_interval_s": "300"}})
    assert w2._positions_panel.calibration_values()["time_interval_min"] == "5"
