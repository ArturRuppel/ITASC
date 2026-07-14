from __future__ import annotations

import sys
import types

from itasc.tracking_ultrack.config import TrackingConfig


def test_run_solve_passes_annotations_true_by_default(
    monkeypatch, tmp_path
):
    from itasc.tracking_ultrack import solve as solve_mod

    calls: list[dict] = []

    monkeypatch.setattr(solve_mod, "_build_ultrack_config", lambda *_args: object())
    monkeypatch.setattr(
        solve_mod,
        "database_has_annotations",
        lambda _wd: (_ for _ in ()).throw(AssertionError("should not inspect DB")),
    )

    def fake_solve(_cfg, **kwargs):
        calls.append(kwargs)

    processing = types.ModuleType("ultrack.core.solve.processing")
    processing.solve = fake_solve
    monkeypatch.setitem(sys.modules, "ultrack.core.solve.processing", processing)

    list(solve_mod.run_solve(tmp_path, TrackingConfig(), overwrite=True))

    assert calls == [{"overwrite": True, "use_annotations": True}]


def test_run_solve_allows_annotations_override(monkeypatch, tmp_path):
    from itasc.tracking_ultrack import solve as solve_mod

    calls: list[dict] = []

    monkeypatch.setattr(solve_mod, "_build_ultrack_config", lambda *_args: object())
    monkeypatch.setattr(
        solve_mod,
        "database_has_annotations",
        lambda _wd: (_ for _ in ()).throw(AssertionError("should not inspect DB")),
    )

    def fake_solve(_cfg, **kwargs):
        calls.append(kwargs)

    processing = types.ModuleType("ultrack.core.solve.processing")
    processing.solve = fake_solve
    monkeypatch.setitem(sys.modules, "ultrack.core.solve.processing", processing)

    list(solve_mod.run_solve(tmp_path, TrackingConfig(), overwrite=True, use_annotations=False))

    assert calls == [{"overwrite": True, "use_annotations": False}]


def test_run_solve_builds_ultrack_config_from_supplied_cfg(monkeypatch, tmp_path):
    from itasc.tracking_ultrack import solve as solve_mod

    cfg = TrackingConfig(bias=-0.5)
    built_cfg = object()
    calls = {}

    def fake_build_ultrack_config(arg_cfg, arg_wd):
        calls["build_cfg"] = arg_cfg
        calls["build_wd"] = arg_wd
        return built_cfg

    def fake_solve(arg_cfg, **kwargs):
        calls["solve_cfg"] = arg_cfg
        calls["solve_kwargs"] = kwargs

    monkeypatch.setattr(solve_mod, "_build_ultrack_config", fake_build_ultrack_config)
    monkeypatch.setattr(
        solve_mod,
        "database_has_annotations",
        lambda _wd: (_ for _ in ()).throw(AssertionError("should not inspect DB")),
    )

    processing = types.ModuleType("ultrack.core.solve.processing")
    processing.solve = fake_solve
    monkeypatch.setitem(sys.modules, "ultrack.core.solve.processing", processing)

    progress = list(solve_mod.run_solve(tmp_path, cfg, overwrite=False))

    assert calls["build_cfg"] is cfg
    assert calls["build_wd"] == tmp_path
    assert calls["solve_cfg"] is built_cfg
    assert calls["solve_kwargs"] == {
        "overwrite": False,
        "use_annotations": True,
    }
    assert progress == [
        (0, 2, "Running ILP solver…"),
        (2, 2, "Solve done."),
    ]
