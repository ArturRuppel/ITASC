"""Integration test: CellFlowMainWidget uses the new CellposeWidget."""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
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


def test_positions_panel_discovers_and_drives_pos_dir(tmp_path):
    """Find data folders adds rows directly; activating a row sets ``_pos_dir``."""
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    for cond, pos in (("WT", "p1"), ("WT", "p2"), ("KO", "p1")):
        raw = tmp_path / cond / pos / "0_input"
        raw.mkdir(parents=True)
        (raw / "nucleus_3dt.tif").touch()  # the discovery input file

    w = main_mod.CellFlowMainWidget(_fake_viewer())
    found = w._positions_panel.discover(str(tmp_path))
    assert len(found) == 3
    keys = w._positions_panel.keys()
    assert len(keys) == 3  # additive Find commits directly, no staging step

    target = str(tmp_path / "WT" / "p2")
    w._positions_panel.set_active(target)
    assert w._pos_dir == (tmp_path / "WT" / "p2")
    assert w.path_label.text() == target
    w.deleteLater()


def test_discover_positions_handles_root_inside_a_position(tmp_path):
    """Picking a position's own ``0_input`` folder finds the position, not a crash.

    The relative input name is ``0_input/nucleus_3dt.tif``, so the derived position
    is the parent of the chosen root. It sits above root and has no nesting under
    it: added plainly, identified by its own folder name.
    """
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    pos = tmp_path / "processed" / "pos11"
    (pos / "0_input").mkdir(parents=True)
    (pos / "0_input" / "nucleus_3dt.tif").touch()

    entries = main_mod._discover_positions(
        str(pos / "0_input"), {"nucleus": "0_input/nucleus_3dt.tif"}
    )
    assert [Path(e["key"]) for e in entries] == [pos]
    assert entries[0]["columns"]["position_id"] == "pos11"


def test_main_widget_constructs_new_cellpose_widget():
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    cellpose_mod = importlib.import_module("cellflow.napari.cellpose_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())
    assert isinstance(w._cellpose_widget, cellpose_mod.CellposeWidget)
    # Old placeholder class no longer exists.
    assert not hasattr(main_mod, "_CellposePanel")
    w.deleteLater()


def test_load_config_is_refused_while_owner_active_and_declined(monkeypatch, tmp_path):
    """Load Config stays clickable during a mode, but a declined prompt must
    not mutate state underneath the active viewer owner."""
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())

    cfg = tmp_path / "cellflow_config.json"
    cfg.write_text('{"metadata": {"pixel_size_um": "0.5"}}')
    w._pos_dir = tmp_path
    # Calibration now lives in the positions panel's Setup section.
    w._positions_panel.set_calibration_values({"pixel_size_um": "0.123"})

    w.gate.register_owner("stub_owner", "stub mode", exit_fn=lambda: None)
    w.gate.claim_viewer("stub_owner")

    prompted = []
    w.gate.confirm_handler = lambda parent, label: prompted.append(label) or False
    # Per-folder config load/save buttons are gone (config now autosaves on
    # every run); ``_on_load_config_from`` is the remaining ``_change_context``
    # gated config-load path.
    monkeypatch.setattr(main_mod.QFileDialog, "getOpenFileName", lambda *a, **k: (str(cfg), ""))
    w._on_load_config_from()

    assert prompted == ["stub mode"]  # the user was asked
    # Nothing was loaded — the panel's pixel size is untouched.
    assert w._positions_panel.calibration_values()["pixel_size_um"] == "0.123"
    assert w.gate.owner == "stub_owner"

    w.deleteLater()


def test_config_load_failure_surfaces_via_notification(monkeypatch, tmp_path):
    """A corrupt config must report through a GUI notification, not a bare
    print to the console the user never sees."""
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    errors: list[str] = []
    monkeypatch.setattr(main_mod, "show_error", lambda m: errors.append(m))

    w = main_mod.CellFlowMainWidget(_fake_viewer())
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    w._load_config(str(bad))

    assert errors and "Error loading config" in errors[0]
    w.deleteLater()


def test_config_save_failure_surfaces_via_notification(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    errors: list[str] = []
    monkeypatch.setattr(main_mod, "show_error", lambda m: errors.append(m))

    w = main_mod.CellFlowMainWidget(_fake_viewer())
    # A directory path can't be opened for writing → save fails.
    w._save_config(str(tmp_path))

    assert errors and "Error saving config" in errors[0]
    w.deleteLater()


def test_context_change_during_correction_exits_owner_when_confirmed(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    w = main_mod.CellFlowMainWidget(_fake_viewer())

    cfg = tmp_path / "cellflow_config.json"
    cfg.write_text('{"metadata": {"pixel_size_um": "0.5"}}')
    w._pos_dir = tmp_path
    # Calibration now lives in the positions panel's Setup section.
    w._positions_panel.set_calibration_values({"pixel_size_um": "0.123"})

    # Stub owner with a lightweight exit (the real correction teardown needs a
    # live viewer); the point under test is the main-widget confirm wiring.
    exited = []

    def _exit():
        exited.append(True)
        w.gate.release_viewer("stub_owner")

    w.gate.register_owner("stub_owner", "stub mode", exit_fn=_exit)
    w.gate.claim_viewer("stub_owner")
    assert not w.gate.can_change_context()

    # Confirm the prompt → the owner is exited and the gated config load commits.
    monkeypatch.setattr(main_mod.QFileDialog, "getOpenFileName", lambda *a, **k: (str(cfg), ""))
    w.gate.confirm_handler = lambda parent, label: True
    w._on_load_config_from()
    assert exited == [True]
    assert w._positions_panel.calibration_values()["pixel_size_um"] == "0.5"
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
    # While correction is inactive, the nucleus title is its plugin-dock stage
    # pill next to the on/off button, so it restyles with the other subheaders.
    # (Once correction is active it swaps to a plain bold workspace title.)
    assert (
        f"color: {ui_style.muted_stage_accent('nucleus')};"
        in w.nucleus_workflow_widget.correction_header_lbl.styleSheet()
    )
    assert (
        f"background-color: {ui_style.stage_header_pill_background('nucleus')};"
        in w.nucleus_workflow_widget.correction_header_lbl.styleSheet()
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

    w._pos_dir = tmp_path
    w._refresh_all()
    assert w._cellpose_widget.divergence_maps_widget._pos_dir == tmp_path
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


def test_catalog_rail_updates_when_a_stage_output_appears(tmp_path):
    """A completed stage refreshes its tracker → the catalog rail repaints.

    Regression: committing a label or running a contact analysis used to leave
    the row's status circles frozen until a manual Refresh. Every stage widget
    refreshes its ``PipelineFilesWidget`` when its output changes, and the main
    widget listens for that to re-read on-disk status per row.
    """
    app = QApplication.instance() or QApplication([])
    main_mod = importlib.import_module("cellflow.napari.main_widget")
    from cellflow.contact_analysis.catalog import CONTACT_ANALYSIS_RELPATH

    pos = tmp_path / "WT" / "p1"
    (pos / "0_input").mkdir(parents=True)
    (pos / "0_input" / "nucleus_3dt.tif").touch()

    w = main_mod.CellFlowMainWidget(_fake_viewer())
    w._positions_panel.discover(str(tmp_path))
    (row,) = w._positions_panel._rows
    # Nothing on disk yet → no stage reads as done.
    assert not any(dot.state == "done" for dot in row.rail.dots)

    # Simulate a finished contact analysis: its output lands, then the widget
    # refreshes its tracker (as the real run does).
    h5 = pos / CONTACT_ANALYSIS_RELPATH
    h5.parent.mkdir(parents=True, exist_ok=True)
    h5.touch()
    w.contact_analysis_widget._files_widget.refresh(pos)

    # The rail repainted from disk without any manual Refresh.
    assert any(dot.state == "done" for dot in row.rail.dots)
    w.deleteLater()
