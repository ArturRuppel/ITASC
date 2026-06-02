from __future__ import annotations

import sys
import types

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig


class _FakeNodeDB:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeOverlapDB:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeNode:
    """Module-level so ``pickle.dumps(node)`` inside the builder succeeds."""

    def __init__(self):
        self.id = 0
        self.time = 0
        self.area = 1
        self.centroid = (0.0, 0.0)

    @classmethod
    def from_mask(cls, t, mask, bbox=None):
        node = cls()
        node.time = t
        mask = np.asarray(mask, dtype=bool)
        node.area = int(mask.sum())
        ys, xs = np.nonzero(mask)
        node.centroid = (float(ys.mean()), float(xs.mean()))
        return node


def _install_ultrack_stubs(monkeypatch):
    """Inject minimal ``ultrack.core.*`` stand-ins so build_atom_union_database
    can run without the real (unimportable) ultrack package.

    Returns a dict capturing what the function did against the fakes.
    """
    captured: dict = {"cleared": [], "created_all": 0}

    class _FakeBase:
        class metadata:
            @staticmethod
            def create_all(_engine):
                captured["created_all"] += 1

    def _clear_all_data(db_path):
        captured["cleared"].append(db_path)

    database_mod = types.ModuleType("ultrack.core.database")
    database_mod.Base = _FakeBase
    database_mod.NodeDB = _FakeNodeDB
    database_mod.OverlapDB = _FakeOverlapDB
    database_mod.clear_all_data = _clear_all_data

    node_mod = types.ModuleType("ultrack.core.segmentation.node")
    node_mod.Node = _FakeNode

    core_mod = types.ModuleType("ultrack.core")
    seg_mod = types.ModuleType("ultrack.core.segmentation")
    root_mod = types.ModuleType("ultrack")

    for name, mod in {
        "ultrack": root_mod,
        "ultrack.core": core_mod,
        "ultrack.core.database": database_mod,
        "ultrack.core.segmentation": seg_mod,
        "ultrack.core.segmentation.node": node_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return captured


def test_build_atom_union_database_segments_then_links(monkeypatch, tmp_path):
    """Atom-union build: read atoms -> config -> clear/create DB -> link, in order."""
    import tifffile

    from cellflow.tracking_ultrack import db_build

    _install_ultrack_stubs(monkeypatch)

    calls: list[str] = []

    # A tiny 2-frame atoms stack with two adjacent atoms per frame.
    atoms = np.zeros((2, 4, 6), dtype=np.int32)
    atoms[:, :, :3] = 1
    atoms[:, :, 3:] = 2

    monkeypatch.setattr(
        tifffile,
        "imread",
        lambda *_args, **_kwargs: calls.append("imread") or atoms,
    )

    class _FakeDataConfig:
        database_path = f"sqlite:///{tmp_path / 'data.db'}"

        def metadata_add(self, _meta):
            calls.append("metadata_add")

    class _FakeUltrackConfig:
        data_config = _FakeDataConfig()

    monkeypatch.setattr(
        db_build,
        "_build_ultrack_config",
        lambda *_args: calls.append("config") or _FakeUltrackConfig(),
    )

    # Stub sqlalchemy engine + Session: the fake NodeDB/OverlapDB rows are not
    # mapped ORM classes, so collect them in plain lists instead of writing SQL.
    import sqlalchemy
    import sqlalchemy.orm

    added: list = []

    class _FakeEngine:
        def dispose(self):
            pass

    monkeypatch.setattr(
        sqlalchemy, "create_engine", lambda *_a, **_kw: _FakeEngine()
    )

    class _FakeSession:
        def __init__(self, _engine):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def add_all(self, rows):
            added.extend(rows)

        def commit(self):
            calls.append("commit")

    monkeypatch.setattr(sqlalchemy.orm, "Session", _FakeSession)

    monkeypatch.setattr(
        db_build,
        "run_linking",
        lambda *_args, **_kwargs: calls.append("link") or iter([(1, 1, "linked")]),
    )

    report = db_build.build_atom_union_database(
        tmp_path / "atoms.tif",
        tmp_path / "work",
        TrackingConfig(),
    )

    # imread happens before config; linking is the last step.
    assert calls[0] == "imread"
    assert "config" in calls
    assert calls.index("config") < calls.index("link")
    assert calls[-1] == "link"

    assert isinstance(report, db_build.AtomUnionDatabaseBuildReport)
    assert report.n_frames == 2
    assert report.total_nodes > 0


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
    monkeypatch.setattr(
        db_build,
        "annotate_anchor_tail_links",
        lambda *_a, **_kw: calls.append("tail_links")
        or type("TailReport", (), {"annotated": 2})(),
    )

    report = db_build.apply_annotations_and_score(
        working_dir=tmp_path / "work",
        cfg=TrackingConfig(),
        score_signal_path=tmp_path / "foreground.tif",
        validated_tracks={7: {0}},
        tracked_labels=tracked,
    )

    assert calls == [
        "reset",
        "pre_link",
        "score",
        "post_link",
        "incident_links",
        "tail_links",
    ]
    assert report.fake_nodes == 3
    assert report.anchor_nodes == 1
    assert report.anchor_links == 6
    assert report.scored_nodes == 5
    assert report.seed_nodes == 1
    assert report.anchor_incident_links_inserted == 4
    assert report.anchor_tail_links_annotated == 2


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
