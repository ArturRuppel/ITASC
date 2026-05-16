from __future__ import annotations

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig


def test_db_build_segments_then_links(monkeypatch, tmp_path):
    """Candidate-only build: segment + link, no annotations, no scoring."""
    from cellflow.tracking_ultrack import db_build

    calls: list[str] = []

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
        "run_linking",
        lambda *_args, **_kwargs: calls.append("link") or iter([(1, 1, "linked")]),
    )

    db_build.build_ultrack_database(
        tmp_path / "contours.tif",
        tmp_path / "foreground.tif",
        tmp_path / "work",
        TrackingConfig(),
    )

    assert calls == ["load", "segment", "link"]


def test_apply_annotations_and_score_resets_then_applies_in_order(monkeypatch, tmp_path):
    """apply_annotations_and_score: reset → pre-link annotate → score → post-link annotate."""
    from cellflow.tracking_ultrack import db_build

    calls: list[str] = []
    tracked = np.zeros((1, 8, 8), dtype=np.uint32)
    tracked[0, 1:4, 1:4] = 7

    monkeypatch.setattr(db_build, "_reset_annotations", lambda *_a: calls.append("reset"))
    monkeypatch.setattr(
        db_build,
        "apply_corrections_to_database",
        lambda *_args, annotate_anchor_links=True, **_kwargs: calls.append(
            "post_link" if annotate_anchor_links else "pre_link"
        )
        or type(
            "CorrectionReport",
            (),
            {"fake_nodes": 3, "anchor_nodes": 1, "anchor_links": 6, "unmatched_anchors": ()},
        )(),
    )
    monkeypatch.setattr(
        db_build,
        "inject_unmatched_anchor_nodes",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    monkeypatch.setattr(
        db_build,
        "write_seed_prior_node_probs",
        lambda *_a, **_kw: calls.append("score")
        or type("ScoreReport", (), {"scored": 5, "seeds": 1})(),
    )
    monkeypatch.setattr(
        db_build,
        "ensure_anchor_incident_links",
        lambda *_a, **_kw: calls.append("incident_links")
        or type("IncidentReport", (), {"inserted": 4, "anchors_processed": 2})(),
    )

    report = db_build.apply_annotations_and_score(
        working_dir=tmp_path / "work",
        cfg=TrackingConfig(),
        score_signal_path=tmp_path / "foreground.tif",
        validated_tracks={7: {0}},
        tracked_labels=tracked,
    )

    assert calls == ["reset", "pre_link", "score", "post_link", "incident_links"]
    assert report.fake_nodes == 3
    assert report.anchor_nodes == 1
    assert report.anchor_links == 6
    assert report.scored_nodes == 5
    assert report.seed_nodes == 1
    assert report.anchor_incident_links_inserted == 4


def test_apply_annotations_and_score_without_corrections_just_resets_and_scores(
    monkeypatch, tmp_path
):
    """No corrections + no validated_tracks: reset and score, but skip annotation passes."""
    from cellflow.tracking_ultrack import db_build

    calls: list[str] = []

    monkeypatch.setattr(db_build, "_reset_annotations", lambda *_a: calls.append("reset"))
    monkeypatch.setattr(
        db_build,
        "apply_corrections_to_database",
        lambda *_args, **_kwargs: calls.append("unexpected_correction_call") or None,
    )
    monkeypatch.setattr(
        db_build,
        "write_seed_prior_node_probs",
        lambda *_a, **_kw: calls.append("score")
        or type("ScoreReport", (), {"scored": 0, "seeds": 0})(),
    )
    monkeypatch.setattr(
        db_build,
        "ensure_anchor_incident_links",
        lambda *_a, **_kw: calls.append("incident_links")
        or type("IncidentReport", (), {"inserted": 0, "anchors_processed": 0})(),
    )

    db_build.apply_annotations_and_score(
        working_dir=tmp_path / "work",
        cfg=TrackingConfig(),
        score_signal_path=tmp_path / "foreground.tif",
    )

    assert calls == ["reset", "score", "incident_links"]


def test_apply_annotations_and_score_validated_without_tracked_labels_raises(tmp_path):
    from cellflow.tracking_ultrack.db_build import apply_annotations_and_score

    try:
        apply_annotations_and_score(
            working_dir=tmp_path / "work",
            cfg=TrackingConfig(),
            score_signal_path=tmp_path / "foreground.tif",
            validated_tracks={7: {0}},
        )
    except ValueError as exc:
        assert "tracked_labels" in str(exc)
    else:
        raise AssertionError("validated_tracks without tracked_labels should raise")
