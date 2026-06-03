"""Policy-table tests for :mod:`cellflow.napari.ui_gate`.

These exercise the gate with lightweight fake controls (no Qt widgets, no GL),
so they run anywhere.
"""
from __future__ import annotations

import importlib

import pytest

ui_gate = importlib.import_module("cellflow.napari.ui_gate")
ControlClass = ui_gate.ControlClass
UiGate = ui_gate.UiGate


class FakeControl:
    def __init__(self) -> None:
        self.enabled = True
        self._tooltip = "idle tip"

    def setEnabled(self, value: bool) -> None:
        self.enabled = bool(value)

    def setToolTip(self, text: str) -> None:
        self._tooltip = text

    def toolTip(self) -> str:
        return self._tooltip


@pytest.fixture
def gate():
    return UiGate()


def test_viewer_owners_are_mutually_exclusive(gate):
    a, b = FakeControl(), FakeControl()
    gate.register(a, ControlClass.VIEWER_OWNER, owner_token="A")
    gate.register(b, ControlClass.VIEWER_OWNER, owner_token="B")
    gate.recompute()
    assert a.enabled and b.enabled

    gate.claim_viewer("A")
    assert a.enabled  # active owner stays enabled so it can be turned off
    assert not b.enabled  # the other owner is locked out

    gate.release_viewer("A")
    assert a.enabled and b.enabled


def test_run_viewer_blocked_while_any_owner_active(gate):
    run = FakeControl()
    gate.register(run, ControlClass.RUN_VIEWER)
    gate.recompute()
    assert run.enabled

    gate.claim_viewer("correction")
    assert not run.enabled
    gate.release_viewer("correction")
    assert run.enabled


def test_run_headless_independent_of_ownership(gate):
    run = FakeControl()
    gate.register(run, ControlClass.RUN_HEADLESS)
    gate.claim_viewer("correction")
    gate.recompute()
    assert run.enabled  # headless jobs run during a mode


def test_harmless_and_context_always_enabled(gate):
    params = FakeControl()
    load = FakeControl()
    gate.register(params, ControlClass.HARMLESS)
    gate.register(load, ControlClass.CONTEXT_CHANGING)
    gate.claim_viewer("correction")
    gate.recompute()
    assert params.enabled and load.enabled


def test_when_predicate_gates_run(gate):
    ready = {"ok": False}
    run = FakeControl()
    gate.register(run, ControlClass.RUN_HEADLESS, when=lambda: ready["ok"])
    gate.recompute()
    assert not run.enabled
    ready["ok"] = True
    gate.recompute()
    assert run.enabled


def test_mode_local_enabled_only_inside_owner(gate):
    btn = FakeControl()
    gate.register(btn, ControlClass.MODE_LOCAL, owner_token="correction")
    gate.recompute()
    assert not btn.enabled
    gate.claim_viewer("correction")
    assert btn.enabled
    gate.claim_viewer("db_browser")  # different owner takes over
    assert not btn.enabled


def test_tooltip_snapshot_and_restore(gate):
    run = FakeControl()
    gate.register_owner("correction", "correction mode", exit_fn=lambda: None)
    gate.register(run, ControlClass.RUN_VIEWER)
    gate.recompute()
    assert run.toolTip() == "idle tip"

    gate.claim_viewer("correction")
    assert "correction mode" in run.toolTip()

    gate.release_viewer("correction")
    assert run.toolTip() == "idle tip"


def test_confirm_context_change_runs_action_when_idle(gate):
    calls = []
    gate.confirm_handler = lambda parent, label: calls.append("asked") or True
    gate.confirm_context_change(None, lambda: calls.append("action"))
    assert calls == ["action"]  # no prompt when nothing owns the viewer


def test_confirm_context_change_exits_owner_then_acts(gate):
    order = []
    gate.register_owner(
        "correction", "correction mode", exit_fn=lambda: order.append("exit")
    )
    gate.claim_viewer("correction")
    gate.confirm_handler = lambda parent, label: True
    ran = gate.confirm_context_change(None, lambda: order.append("action"))
    assert ran is True
    assert order == ["exit", "action"]


def test_confirm_context_change_declined_blocks_action(gate):
    order = []
    gate.register_owner(
        "correction", "correction mode", exit_fn=lambda: order.append("exit")
    )
    gate.claim_viewer("correction")
    gate.confirm_handler = lambda parent, label: False
    ran = gate.confirm_context_change(None, lambda: order.append("action"))
    assert ran is False
    assert order == []  # neither exit nor action ran


def test_can_change_context_reflects_owner(gate):
    assert gate.can_change_context() is True
    gate.claim_viewer("correction")
    assert gate.can_change_context() is False
    gate.release_viewer("correction")
    assert gate.can_change_context() is True
