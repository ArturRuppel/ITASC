from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from itasc.tracking_ultrack.config import TrackingConfig
from itasc.tracking_ultrack.export import export_tracked_labels


def test_export_prefers_public_track_id_zarr_export(monkeypatch, tmp_path):
    import itasc.tracking_ultrack.export as export_module

    expected = np.zeros((2, 8, 8), dtype=np.uint32)
    expected[:, 1:4, 1:4] = 12
    bad_detection_labels = expected.copy()
    bad_detection_labels[1, 1:4, 1:4] = 99
    calls: list[str] = []

    monkeypatch.setattr(export_module, "_build_export_config", lambda cfg, wd: object())
    monkeypatch.setattr(export_module, "database_has_annotations", lambda wd: False)

    export_pkg = types.ModuleType("ultrack.core.export")

    def to_tracks_layer(config):
        calls.append("to_tracks_layer")
        return "tracks_df", {}

    def tracks_to_zarr(config, tracks_df, store_or_path=None, chunks=None, overwrite=False):
        calls.append(f"tracks_to_zarr:{tracks_df}")
        return expected

    export_pkg.to_tracks_layer = to_tracks_layer
    export_pkg.tracks_to_zarr = tracks_to_zarr

    labels_mod = types.ModuleType("ultrack.core.export.labels")
    labels_mod.to_labels = lambda config: bad_detection_labels

    monkeypatch.setitem(sys.modules, "ultrack", types.ModuleType("ultrack"))
    monkeypatch.setitem(sys.modules, "ultrack.core", types.ModuleType("ultrack.core"))
    monkeypatch.setitem(sys.modules, "ultrack.core.export", export_pkg)
    monkeypatch.setitem(sys.modules, "ultrack.core.export.labels", labels_mod)

    output_path = tmp_path / "tracked_labels.tif"
    result = export_tracked_labels(tmp_path, TrackingConfig(), output_path)

    assert calls == ["to_tracks_layer", "tracks_to_zarr:tracks_df"]
    np.testing.assert_array_equal(result, expected)


def test_export_propagates_runtime_error_from_track_export(monkeypatch, tmp_path):
    """A genuine runtime failure (e.g. corrupt DB) must propagate, not silently
    fall through to the degraded CTC/to_labels path and emit wrong labels."""
    import itasc.tracking_ultrack.export as export_module

    fallback = np.zeros((2, 8, 8), dtype=np.uint32)
    fallback[:, 2:5, 2:5] = 7
    to_labels_called: list[str] = []

    monkeypatch.setattr(export_module, "_build_export_config", lambda cfg, wd: object())
    monkeypatch.setattr(export_module, "database_has_annotations", lambda wd: False)

    export_pkg = types.ModuleType("ultrack.core.export")

    def to_tracks_layer(config):
        raise RuntimeError("corrupt NodeDB")

    export_pkg.to_tracks_layer = to_tracks_layer
    export_pkg.tracks_to_zarr = lambda *a, **k: fallback

    labels_mod = types.ModuleType("ultrack.core.export.labels")

    def to_labels(config):
        to_labels_called.append("to_labels")
        return fallback

    labels_mod.to_labels = to_labels

    monkeypatch.setitem(sys.modules, "ultrack", types.ModuleType("ultrack"))
    monkeypatch.setitem(sys.modules, "ultrack.core", types.ModuleType("ultrack.core"))
    monkeypatch.setitem(sys.modules, "ultrack.core.export", export_pkg)
    monkeypatch.setitem(sys.modules, "ultrack.core.export.labels", labels_mod)

    with pytest.raises(RuntimeError, match="corrupt NodeDB"):
        export_tracked_labels(tmp_path, TrackingConfig(), tmp_path / "out.tif")
    assert to_labels_called == []  # must not have degraded to the fallback


def test_export_falls_through_when_track_api_absent(monkeypatch, tmp_path):
    """An older ultrack lacking the public track-export API (ImportError) should
    fall through to the to_labels strategy — capability detection, not failure."""
    import itasc.tracking_ultrack.export as export_module

    expected = np.zeros((2, 8, 8), dtype=np.uint32)
    expected[:, 1:4, 1:4] = 3

    monkeypatch.setattr(export_module, "_build_export_config", lambda cfg, wd: object())
    monkeypatch.setattr(export_module, "database_has_annotations", lambda wd: False)

    # export package exists but lacks to_tracks_layer/tracks_to_zarr → ImportError
    export_pkg = types.ModuleType("ultrack.core.export")
    labels_mod = types.ModuleType("ultrack.core.export.labels")
    labels_mod.to_labels = lambda config: expected

    monkeypatch.setitem(sys.modules, "ultrack", types.ModuleType("ultrack"))
    monkeypatch.setitem(sys.modules, "ultrack.core", types.ModuleType("ultrack.core"))
    monkeypatch.setitem(sys.modules, "ultrack.core.export", export_pkg)
    monkeypatch.setitem(sys.modules, "ultrack.core.export.labels", labels_mod)

    result = export_tracked_labels(tmp_path, TrackingConfig(), tmp_path / "out.tif")
    np.testing.assert_array_equal(result, expected)
