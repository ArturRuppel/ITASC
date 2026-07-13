"""Config autosave + project-catalog CSV behavior in the full CellFlow app."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari.qt import get_qapp
from cellflow.napari._stage_status import STAGE_CELL, STAGE_CONTACTS
from cellflow.napari.main_widget import CellFlowMainWidget


def _fake_viewer():
    # Mirrors ``tests/napari/test_main_widget_cellpose_integration.py``: this repo
    # constructs ``CellFlowMainWidget`` against a lightweight stand-in rather than
    # a real ``napari.Viewer`` (no pytest-qt / ``make_napari_viewer`` fixture here).
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


@pytest.fixture
def widget():
    get_qapp()
    return CellFlowMainWidget(_fake_viewer())


def test_save_config_quiet_suppresses_toast(widget, tmp_path):
    cfg = tmp_path / "cellflow_config.json"
    with patch("cellflow.napari.main_widget.show_info") as info:
        widget._save_config(str(cfg), quiet=True)
    assert cfg.exists()
    info.assert_not_called()
    # And the loud path still notifies.
    with patch("cellflow.napari.main_widget.show_info") as info:
        widget._save_config(str(cfg))
    info.assert_called_once()


RUN_BUTTONS = [
    ("_cellpose_widget", "nucleus_run_btn"),
    ("_cellpose_widget", "cell_run_btn"),
    ("nucleus_workflow_widget", "seg_run_btn"),
    ("nucleus_workflow_widget", "db_run_btn"),
    ("nucleus_workflow_widget", "solve_run_btn"),
    ("cell_workflow_widget", "run_btn"),
]


@pytest.mark.parametrize("widget_attr,btn_attr", RUN_BUTTONS)
def test_run_button_attr_exists(widget, widget_attr, btn_attr):
    # The __init__ wiring loop getattr's each of these; a rename would already have
    # crashed the fixture. This pins the exact names the autosave loop depends on.
    assert hasattr(getattr(widget, widget_attr), btn_attr)


def test_autosave_writes_config_into_pos_dir(widget, tmp_path):
    widget._pos_dir = tmp_path
    widget._autosave_config()
    assert (tmp_path / "cellflow_config.json").exists()


def test_autosave_noop_without_pos_dir(widget):
    widget._pos_dir = None
    widget._autosave_config()  # must not raise


from cellflow.contact_analysis.catalog import (
    CONTACT_ANALYSIS_RELPATH,
    REQUIRED_CSV_COLUMNS,
    load_catalog,
)


def test_catalog_record_stamps_default_paths(widget, tmp_path):
    pos = tmp_path / "WT" / "pos01"
    pos.mkdir(parents=True)
    rec = widget._catalog_record_for_position(
        pos, {"condition": "WT", "position_id": "pos01"}
    )
    assert Path(rec["position_path"]) == pos
    assert rec["contact_analysis_path"] == pos / CONTACT_ANALYSIS_RELPATH
    assert rec["cell_tracked_labels_path"] == pos / "cell_labels.tif"
    assert rec["nucleus_tracked_labels_path"] == pos / "nucleus_labels.tif"
    assert rec["columns"]["condition"] == "WT"


def test_catalog_record_stamps_committed_label_paths(widget, tmp_path):
    pos = tmp_path / "pos00"
    rec = widget._catalog_record_for_position(pos, {"condition": "ctrl", "position_id": "pos00"})

    assert rec["cell_tracked_labels_path"] == pos / "cell_labels.tif"
    assert rec["nucleus_tracked_labels_path"] == pos / "nucleus_labels.tif"
    assert rec["contact_analysis_path"] == pos / "contact_analysis.h5"


def _seed_two_rows(widget, tmp_path):
    a = tmp_path / "WT" / "pos01"
    b = tmp_path / "KO" / "pos01"
    for p in (a, b):
        p.mkdir(parents=True)
    widget._positions_panel.set_records([
        {"key": str(a), "columns": {"condition": "WT", "position_id": "pos01"},
         "payload": {"position_path": a}},
        {"key": str(b), "columns": {"condition": "KO", "position_id": "pos01"},
         "payload": {"position_path": b}},
    ])
    return a, b


def test_save_project_writes_loadable_catalog(widget, tmp_path):
    _seed_two_rows(widget, tmp_path)
    csv_path = tmp_path / "catalog.csv"
    with patch(
        "cellflow.napari.main_widget.QFileDialog.getSaveFileName",
        return_value=(str(csv_path), ""),
    ):
        widget._on_save_project()
    assert csv_path.exists()
    loaded = load_catalog(csv_path)
    assert len(loaded) == 2
    for col in REQUIRED_CSV_COLUMNS:
        assert col in loaded[0]
    # The stamped default h5 path survives the round-trip.
    assert loaded[0]["contact_analysis_path"].name == "contact_analysis.h5"


def test_save_project_appends_csv_suffix(widget, tmp_path):
    _seed_two_rows(widget, tmp_path)
    chosen = tmp_path / "myproject"  # no extension
    with patch(
        "cellflow.napari.main_widget.QFileDialog.getSaveFileName",
        return_value=(str(chosen), ""),
    ):
        widget._on_save_project()
    assert (tmp_path / "myproject.csv").exists()


def test_load_project_repopulates_panel(widget, tmp_path):
    a, b = _seed_two_rows(widget, tmp_path)
    csv_path = tmp_path / "catalog.csv"
    with patch(
        "cellflow.napari.main_widget.QFileDialog.getSaveFileName",
        return_value=(str(csv_path), ""),
    ):
        widget._on_save_project()
    widget._positions_panel.set_records([])
    assert widget._positions_panel.keys() == []
    with patch(
        "cellflow.napari.main_widget.QFileDialog.getOpenFileName",
        return_value=(str(csv_path), ""),
    ):
        widget._on_load_project()
    assert len(widget._positions_panel.keys()) == 2


def test_contacts_dot_visualizes_after_retargeting(widget, tmp_path):
    # The 4th rail dot (contacts) has no raw-image output; clicking it retargets
    # the detail pane to the row's position and fires the Visualize action, exactly
    # as the "Visualize Contact Analysis" button does.
    pos = tmp_path / "posA"
    pos.mkdir()
    payload = {"position_path": pos}

    widget.contact_analysis_widget._on_visualize = MagicMock()
    with patch.object(widget, "_retarget_to_position") as retarget:
        widget._on_position_stage_load(payload, STAGE_CONTACTS)

    retarget.assert_called_once_with(pos)
    widget.contact_analysis_widget._on_visualize.assert_called_once_with(overwrite=False)


def test_contacts_dot_skips_retarget_when_already_selected(widget, tmp_path):
    pos = tmp_path / "posA"
    pos.mkdir()
    widget._pos_dir = pos

    widget.contact_analysis_widget._on_visualize = MagicMock()
    with patch.object(widget, "_retarget_to_position") as retarget:
        widget._on_position_stage_load({"position_path": pos}, STAGE_CONTACTS)

    retarget.assert_not_called()
    widget.contact_analysis_widget._on_visualize.assert_called_once_with(overwrite=False)


def test_non_contacts_dot_loads_stage_into_viewer(widget, tmp_path):
    pos = tmp_path / "posA"
    pos.mkdir()
    with patch("cellflow.napari.main_widget.load_stage") as load_stage:
        widget._on_position_stage_load({"position_path": pos}, STAGE_CELL)
    load_stage.assert_called_once_with(widget.viewer, pos, STAGE_CELL)


def test_toolbar_has_project_and_params_pairs(widget):
    # Project group: select folder + load/save PROJECT (catalog CSV).
    assert hasattr(widget, "load_project_btn")
    assert hasattr(widget, "save_project_btn")
    # Params group: load/save CONFIG file.
    assert hasattr(widget, "load_from_btn")
    assert hasattr(widget, "save_as_btn")
    # The per-folder config buttons/handlers are gone.
    assert not hasattr(widget, "save_btn")
    assert not hasattr(widget, "load_btn")
    assert not hasattr(widget, "_on_save_config")
    assert not hasattr(widget, "_on_load_config")
