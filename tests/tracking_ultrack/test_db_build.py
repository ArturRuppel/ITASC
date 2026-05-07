from __future__ import annotations

from pathlib import Path

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig


def test_plain_db_build_segments_scores_and_links_without_validated_steps(
    monkeypatch, tmp_path
):
    from cellflow.tracking_ultrack import db_build

    calls: list[str] = []
    contour_path = tmp_path / "contours.tif"
    foreground_path = tmp_path / "foreground.tif"
    intensity_path = tmp_path / "nucleus.tif"

    monkeypatch.setattr(
        db_build,
        "_load_ultrack_inputs",
        lambda *_args: (
            calls.append("load"),
            (np.ones((1, 8, 8), dtype=np.float32), np.ones((1, 8, 8), dtype=np.float32)),
        )[1],
    )
    monkeypatch.setattr(db_build, "_build_ultrack_config", lambda *_args: object())
    monkeypatch.setattr(
        db_build,
        "_run_ultrack_segment",
        lambda *_args, **_kwargs: calls.append("segment"),
    )
    monkeypatch.setattr(
        db_build,
        "inject_validated_nodes",
        lambda *_args, **_kwargs: calls.append("inject"),
    )
    monkeypatch.setattr(
        db_build,
        "write_seed_prior_node_probs",
        lambda *_args, **_kwargs: calls.append("score")
        or type("ScoreReport", (), {"scored": 3, "seeds": 0})(),
    )
    monkeypatch.setattr(
        db_build,
        "run_linking",
        lambda *_args, **_kwargs: calls.append("link") or iter([(1, 1, "linked")]),
    )
    monkeypatch.setattr(
        db_build,
        "boost_validated_edges",
        lambda *_args, **_kwargs: calls.append("boost"),
    )

    report = db_build.build_ultrack_database(
        contour_path,
        foreground_path,
        intensity_path,
        tmp_path / "work",
        TrackingConfig(),
        use_validated=False,
    )

    assert calls == ["load", "segment", "score", "link"]
    assert report.scored_nodes == 3
    assert report.real_nodes == 0
    assert report.fake_nodes == 0
    assert report.boosted_edges == 0


def test_validated_db_build_injects_scores_links_and_boosts_in_order(
    monkeypatch, tmp_path
):
    from cellflow.tracking_ultrack import db_build

    calls: list[str] = []
    tracked = np.zeros((1, 8, 8), dtype=np.uint32)
    tracked[0, 1:4, 1:4] = 7

    monkeypatch.setattr(
        db_build,
        "_load_ultrack_inputs",
        lambda *_args: (
            calls.append("load"),
            (np.ones((1, 8, 8), dtype=np.float32), np.ones((1, 8, 8), dtype=np.float32)),
        )[1],
    )
    monkeypatch.setattr(db_build, "_build_ultrack_config", lambda *_args: object())
    monkeypatch.setattr(
        db_build,
        "_run_ultrack_segment",
        lambda *_args, **_kwargs: calls.append("segment"),
    )
    monkeypatch.setattr(
        db_build,
        "inject_validated_nodes",
        lambda *_args, **_kwargs: calls.append("inject")
        or type(
            "InjectionReport",
            (),
            {"inserted": 1, "skipped_missing": 2, "faked": 3, "overlaps_added": 4},
        )(),
    )
    monkeypatch.setattr(
        db_build,
        "write_seed_prior_node_probs",
        lambda *_args, **_kwargs: calls.append("score")
        or type("ScoreReport", (), {"scored": 5, "seeds": 1})(),
    )
    monkeypatch.setattr(
        db_build,
        "run_linking",
        lambda *_args, **_kwargs: calls.append("link") or iter([(1, 1, "linked")]),
    )
    monkeypatch.setattr(
        db_build,
        "boost_validated_edges",
        lambda *_args, **_kwargs: calls.append("boost")
        or type("BoostReport", (), {"boosted": 6, "seeds": 1})(),
    )

    report = db_build.build_ultrack_database(
        tmp_path / "contours.tif",
        tmp_path / "foreground.tif",
        tmp_path / "nucleus.tif",
        tmp_path / "work",
        TrackingConfig(),
        validated_tracks={7: {0}},
        tracked_labels=tracked,
        use_validated=True,
    )

    assert calls == ["load", "segment", "inject", "score", "link", "boost"]
    assert report.real_nodes == 1
    assert report.skipped_validated == 2
    assert report.fake_nodes == 3
    assert report.overlaps_added == 4
    assert report.scored_nodes == 5
    assert report.boosted_edges == 6


def test_validated_db_build_requires_validated_tracks_and_labels(tmp_path):
    from cellflow.tracking_ultrack.db_build import build_ultrack_database

    try:
        build_ultrack_database(
            tmp_path / "contours.tif",
            tmp_path / "foreground.tif",
            tmp_path / "nucleus.tif",
            tmp_path / "work",
            TrackingConfig(),
            use_validated=True,
        )
    except ValueError as exc:
        assert "validated tracks and tracked labels" in str(exc)
    else:
        raise AssertionError("validated DB build should require validation inputs")
