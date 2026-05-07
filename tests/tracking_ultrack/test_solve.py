from __future__ import annotations

import sys
import types

from cellflow.tracking_ultrack.config import TrackingConfig


def test_run_solve_auto_enables_annotations_when_db_contains_annotations(
    monkeypatch, tmp_path
):
    from cellflow.tracking_ultrack import solve as solve_mod

    calls: list[dict] = []

    monkeypatch.setattr(solve_mod, "_build_ultrack_config", lambda *_args: object())
    monkeypatch.setattr(solve_mod, "database_has_annotations", lambda _wd: True)

    def fake_solve(_cfg, **kwargs):
        calls.append(kwargs)

    processing = types.ModuleType("ultrack.core.solve.processing")
    processing.solve = fake_solve
    monkeypatch.setitem(sys.modules, "ultrack.core.solve.processing", processing)

    list(solve_mod.run_solve(tmp_path, TrackingConfig(), overwrite=True))

    assert calls == [{"overwrite": True, "use_annotations": True}]


def test_run_solve_keeps_plain_database_unannotated(monkeypatch, tmp_path):
    from cellflow.tracking_ultrack import solve as solve_mod

    calls: list[dict] = []

    monkeypatch.setattr(solve_mod, "_build_ultrack_config", lambda *_args: object())
    monkeypatch.setattr(solve_mod, "database_has_annotations", lambda _wd: False)

    def fake_solve(_cfg, **kwargs):
        calls.append(kwargs)

    processing = types.ModuleType("ultrack.core.solve.processing")
    processing.solve = fake_solve
    monkeypatch.setitem(sys.modules, "ultrack.core.solve.processing", processing)

    list(solve_mod.run_solve(tmp_path, TrackingConfig(), overwrite=True))

    assert calls == [{"overwrite": True, "use_annotations": False}]
