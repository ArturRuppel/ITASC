from __future__ import annotations

import sys
import types

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.export import export_tracked_labels


def test_export_prefers_public_track_id_zarr_export(monkeypatch, tmp_path):
    import cellflow.tracking_ultrack.export as export_module

    expected = np.zeros((2, 8, 8), dtype=np.uint32)
    expected[:, 1:4, 1:4] = 12
    bad_detection_labels = expected.copy()
    bad_detection_labels[1, 1:4, 1:4] = 99
    calls: list[str] = []

    monkeypatch.setattr(export_module, "_build_export_config", lambda cfg, wd: object())

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
