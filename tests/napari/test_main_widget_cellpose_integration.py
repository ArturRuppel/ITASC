"""Integration test: CellFlowMainWidget uses the new CellposeWidget."""
from __future__ import annotations

import importlib
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def _mock_cellpose(monkeypatch):
    fake_cellpose = types.ModuleType("cellpose")
    fake_models = types.ModuleType("cellpose.models")

    class _FakeModel:
        def __init__(self, *_, **__): pass

        def eval(self, img, **_kwargs):
            arr = np.asarray(img, dtype=np.float32)
            ndim = 3 if arr.ndim == 3 else 2
            chans = 3 if ndim == 3 else 2
            dp = np.zeros((chans, *arr.shape), dtype=np.float32)
            prob = np.zeros(arr.shape, dtype=np.float32)
            return None, (None, dp, prob), None

    fake_models.CellposeModel = _FakeModel
    fake_cellpose.models = fake_models
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)
    monkeypatch.delitem(sys.modules, "cellflow.cellpose.cellpose_runner", raising=False)
    monkeypatch.delitem(sys.modules, "cellflow.napari.cellpose_widget", raising=False)
    monkeypatch.delitem(sys.modules, "cellflow.napari.main_widget", raising=False)


def _fake_viewer():
    class _Sel:
        active = None

    class _Layers(dict):
        selection = _Sel()
        events = SimpleNamespace(removed=SimpleNamespace(connect=lambda cb: None))

        def remove(self, layer):
            self.pop(layer.name, None)

    viewer = SimpleNamespace()
    viewer.layers = _Layers()
    viewer.dims = SimpleNamespace(
        current_step=(0, 0),
        events=SimpleNamespace(current_step=SimpleNamespace(connect=lambda cb: None)),
    )
    viewer.add_image = MagicMock()
    viewer.add_labels = MagicMock()
    viewer.add_shapes = MagicMock()
    viewer.bind_key = MagicMock()
    return viewer


def test_main_widget_constructs_new_cellpose_widget():
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    cellpose_mod = importlib.import_module("cellflow.napari.cellpose_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())
    assert isinstance(w._cellpose_widget, cellpose_mod.CellposeWidget)
    # Old placeholder class no longer exists.
    assert not hasattr(main_mod, "_CellposePanel")
    w.deleteLater()


def test_position_change_during_correction_prompts_and_can_be_declined():
    """The position spinbox stays enabled during correction; a change there is
    guarded by a confirm-and-exit prompt and reverts when declined."""
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())

    # Position is always usable now (context-changing controls stay enabled).
    assert w.pos_spin.isEnabled()
    w._committed_pos = w.pos_spin.value()

    # A viewer owner becomes active.
    w.gate.claim_viewer("correction:nucleus")
    assert w.pos_spin.isEnabled()
    assert not w.gate.can_change_context()

    # Decline the prompt → the spinbox reverts and the owner is untouched.
    w.gate.confirm_handler = lambda parent, label: False
    w.pos_spin.setValue(w.pos_spin.value() + 1)
    assert w.pos_spin.value() == w._committed_pos
    assert w.gate.owner == "correction:nucleus"

    w.deleteLater()


def test_load_config_is_refused_while_owner_active_and_declined(tmp_path):
    """Load Config stays clickable during a mode, but a declined prompt must
    not mutate state underneath the active viewer owner."""
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())

    cfg = tmp_path / "cellflow_config.json"
    cfg.write_text('{"metadata": {"position": 7}}')
    w.path_label.setText(str(tmp_path))
    w._committed_pos = w.pos_spin.value()

    w.gate.register_owner("stub_owner", "stub mode", exit_fn=lambda: None)
    w.gate.claim_viewer("stub_owner")

    prompted = []
    w.gate.confirm_handler = lambda parent, label: prompted.append(label) or False
    w._on_load_config()

    assert prompted == ["stub mode"]  # the user was asked
    assert w.pos_spin.value() == w._committed_pos  # but nothing was loaded
    assert w.gate.owner == "stub_owner"

    w.deleteLater()


def test_position_change_during_correction_exits_owner_when_confirmed():
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())
    w._committed_pos = w.pos_spin.value()

    # Stub owner with a lightweight exit (the real correction teardown needs a
    # live viewer); the point under test is the main-widget confirm wiring.
    exited = []

    def _exit():
        exited.append(True)
        w.gate.release_viewer("stub_owner")

    w.gate.register_owner("stub_owner", "stub mode", exit_fn=_exit)
    w.gate.claim_viewer("stub_owner")
    assert not w.gate.can_change_context()

    # Confirm the prompt → the owner is exited and the new position commits.
    w.gate.confirm_handler = lambda parent, label: True
    target = w.pos_spin.value() + 1
    w.pos_spin.setValue(target)
    assert exited == [True]
    assert w.pos_spin.value() == target
    assert w._committed_pos == target
    assert w.gate.can_change_context()

    w.deleteLater()


def test_main_widget_theme_picker_restyles_stage_subheaders():
    app = QApplication.instance() or QApplication([])
    ui_style = importlib.import_module("cellflow.napari.ui_style")
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    ui_style.set_active_theme("Cividis")
    w = main_mod.CellFlowMainWidget(_fake_viewer())

    w._on_theme_selected("Sunset")

    assert (
        f"color: {ui_style.muted_stage_accent('cellpose')};"
        in w._cellpose_widget.pipeline_files_header_lbl.styleSheet()
    )
    assert (
        f"background-color: {ui_style.stage_header_pill_background('cellpose')};"
        in w._cellpose_widget.pipeline_files_header_lbl.styleSheet()
    )
    # The nucleus correction title was redesigned into a plain workspace title
    # (a bold heading, not a stage pill), so it is intentionally outside the
    # theme-restyled stage-subheader set. The cell correction header still pills.
    assert "font-weight: bold" in (
        w.nucleus_workflow_widget.correction_header_lbl.styleSheet()
    )
    assert (
        f"color: {ui_style.muted_stage_accent('cell')};"
        in w.cell_workflow_widget.correction_header_lbl.styleSheet()
    )
    assert (
        f"background-color: {ui_style.stage_header_pill_background('cell')};"
        in w.cell_workflow_widget.correction_header_lbl.styleSheet()
    )
    assert (
        f"color: {ui_style.muted_stage_accent('contact_analysis')};"
        in w.contact_analysis_widget.pipeline_files_header_lbl.styleSheet()
    )
    assert (
        f"background-color: {ui_style.stage_header_pill_background('contact_analysis')};"
        in w.contact_analysis_widget.pipeline_files_header_lbl.styleSheet()
    )
    w.deleteLater()


def test_main_widget_keeps_divergence_maps_inside_cellpose(tmp_path):
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    divergence_mod = importlib.import_module("cellflow.napari.divergence_maps_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())

    assert not hasattr(w, "_divergence_maps_widget")
    assert not hasattr(w, "divergence_maps_section")
    assert isinstance(
        w._cellpose_widget.divergence_maps_widget,
        divergence_mod.DivergenceMapsWidget,
    )
    assert not hasattr(w._cellpose_widget.divergence_maps_widget, "output_files_tracker")
    assert not hasattr(w._cellpose_widget.divergence_maps_widget, "pipeline_files_header")

    state = {
        "nucleus": {"smoothing_sigma": 2.5, "median_radius": 2},
        "cell": {"foreground_z_reduction": "max"},
    }
    w.set_state({"cellpose": {"divergence_maps": state}})
    got = w.get_state()
    assert got["cellpose"]["divergence_maps"]["nucleus"]["smoothing_sigma"] == pytest.approx(2.5)
    assert got["cellpose"]["divergence_maps"]["nucleus"]["median_radius"] == 2
    assert got["cellpose"]["divergence_maps"]["cell"]["foreground_z_reduction"] == "max"

    w.path_label.setText(str(tmp_path))
    w.pos_spin.setValue(0)
    w._refresh_all()
    assert w._cellpose_widget.divergence_maps_widget._pos_dir == tmp_path / "pos00"
    w.deleteLater()


def test_main_widget_state_round_trips_cellpose():
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())
    cellpose_state = {
        "nucleus": {
            "layout": "3D+t",
            "do_3d": False,
            "anisotropy": 1.25,
            "diameter": 33.0,
            "min_size": 9,
            "gamma": 1.2,
        },
        "cell": {"layout": "3D+t", "diameter": 17.0, "min_size": 4, "gamma": 0.9},
    }
    w.set_state({"cellpose": cellpose_state})
    got = w.get_state()
    assert "cellpose" in got
    assert got["cellpose"]["nucleus"] == cellpose_state["nucleus"]
    assert got["cellpose"]["cell"] == cellpose_state["cell"]
    assert "divergence_maps" in got["cellpose"]
    w.deleteLater()


def test_main_widget_pipeline_status_uses_output_files_tracker(tmp_path):
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())
    # Should reach pipeline_status_from_files without error (tracker exists).
    assert w._cellpose_widget.output_files_tracker is not None
    w._update_section_statuses()
    w.deleteLater()
