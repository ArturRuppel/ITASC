"""The studio-level shared parameter bar: build params + build stamping."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from cellflow.napari.aggregate_quantification_params import SharedParamsWidget


def _app():
    return QApplication.instance() or QApplication([])


def test_build_params_reads_fov_and_shuffles_with_default():
    app = _app()
    try:
        w = SharedParamsWidget()
        # Blank shuffles -> the default (1000); blank fov -> None (unset, for gating).
        params = w.build_params()
        assert params["shuffles"] == 1000
        assert params["fov_area_mm2"] is None

        w._fov_edit.setText("1.5")
        w._shuffles_edit.setText("200")
        params = w.build_params()
        assert params["fov_area_mm2"] == 1.5
        assert params["shuffles"] == 200
    finally:
        w.deleteLater()
        app.processEvents()


def test_stamp_is_noop_when_build_fields_blank():
    app = _app()
    w = SharedParamsWidget()
    try:
        records = [{"id": "p1"}, {"id": "p2"}]
        stamped = w.stamp(records)
        assert stamped == records
        assert all("pixel_size_um" not in r for r in stamped)
    finally:
        w.deleteLater()
        app.processEvents()


def test_stamp_writes_pixel_size_and_frame_interval_without_mutating():
    app = _app()
    w = SharedParamsWidget()
    try:
        w._pixel_size_edit.setText("0.3")
        w._frame_interval_edit.setText("2.0")
        original = {"id": "p1"}
        (stamped,) = w.stamp([original])
        assert stamped["pixel_size_um"] == 0.3
        assert stamped["time_interval_s"] == 2.0
        # Original record untouched (copies, not in-place edits).
        assert "pixel_size_um" not in original
    finally:
        w.deleteLater()
        app.processEvents()


def test_changed_signal_fires_on_edit():
    app = _app()
    w = SharedParamsWidget()
    fired = []
    w.changed.connect(lambda: fired.append(True))
    try:
        w._pixel_size_edit.setText("0.5")
        assert fired
    finally:
        w.deleteLater()
        app.processEvents()
