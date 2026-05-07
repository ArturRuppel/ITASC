from __future__ import annotations

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig


def test_export_preserves_validated_ids_when_database_has_annotations(
    monkeypatch, tmp_path
):
    from cellflow.tracking_ultrack import export as export_mod

    tracked = np.zeros((1, 6, 6), dtype=np.uint32)
    tracked[0, 1:3, 1:3] = 7
    raw_export = np.zeros_like(tracked)
    raw_export[0, 1:3, 1:3] = 99

    monkeypatch.setattr(export_mod, "database_has_annotations", lambda _wd: True)
    monkeypatch.setattr(export_mod, "_export_tracked_labels_raw", lambda *_args: raw_export.copy())

    labels = export_mod.export_tracked_labels(
        tmp_path,
        TrackingConfig(),
        tmp_path / "tracked.tif",
        validated_tracks={7: {0}},
        tracked_labels=tracked,
    )

    assert np.all(labels[0, 1:3, 1:3] == 7)


def test_export_fails_when_annotated_database_lacks_validated_masks(monkeypatch, tmp_path):
    from cellflow.tracking_ultrack import export as export_mod

    monkeypatch.setattr(export_mod, "database_has_annotations", lambda _wd: True)
    monkeypatch.setattr(
        export_mod,
        "_export_tracked_labels_raw",
        lambda *_args: np.zeros((1, 6, 6), dtype=np.uint32),
    )

    try:
        export_mod.export_tracked_labels(
            tmp_path,
            TrackingConfig(),
            tmp_path / "tracked.tif",
        )
    except ValueError as exc:
        assert "validated tracks and tracked labels" in str(exc)
    else:
        raise AssertionError("annotated export should require validated masks")
