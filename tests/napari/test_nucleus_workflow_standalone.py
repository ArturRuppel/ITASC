"""Standalone-mode seam for the nucleus tracking/correction piece.

The full orchestrator drives the widget through ``refresh(pos_dir)`` (staged
``<pos>/2_nucleus`` layout); the independently-installable ``cellflow-tracking``
plugin drives it through ``set_context(work_dir=...)`` (flat working directory).
"""
from __future__ import annotations

import napari
import pytest

from cellflow.napari import nucleus_workflow_widget as mod
from cellflow.napari._paths import NucleusWorkspace


@pytest.fixture()
def viewer():
    v = napari.Viewer(show=False)
    try:
        yield v
    finally:
        v.close()


def test_factory_builds_standalone_widget_with_workdir_picker(viewer):
    widget = mod.make_nucleus_tracking_widget(viewer)
    assert widget._standalone is True
    # Standalone shows its own working-directory picker and hides the staged
    # pipeline-files panel (which lists non-existent 1_cellpose/2_nucleus paths).
    assert widget._workdir_container.isVisibleTo(widget)
    assert not widget._pipeline_files_section.isVisibleTo(widget)


def test_orchestrated_widget_hides_standalone_picker(viewer):
    widget = mod.NucleusWorkflowWidget(viewer)
    assert widget._standalone is False
    assert not widget._workdir_container.isVisibleTo(widget)
    # Orchestrated refresh with no project is a no-op that clears the workspace.
    widget.refresh(None)
    assert widget._workspace is None


def test_set_context_builds_flat_workspace(viewer, tmp_path):
    widget = mod.make_nucleus_tracking_widget(viewer)
    widget.set_context(work_dir=tmp_path)
    ws = widget._workspace
    assert isinstance(ws, NucleusWorkspace)
    assert ws.nucleus_dir == tmp_path
    assert ws.foreground == tmp_path / "foreground.tif"
    assert ws.contours == tmp_path / "contours.tif"
    # The annotation store lives directly in the working directory (no 2_nucleus).
    assert ws.ultrack_db == tmp_path / "ultrack_workdir" / "data.db"
    # _pos_dir (the nucleus store dir threaded to the validation API) follows it.
    assert widget._pos_dir is None
    assert widget.nucleus_correction_widget._pos_dir == tmp_path


def test_set_context_persists_work_dir_across_instances(viewer, tmp_path, monkeypatch):
    # Isolate QSettings so the test never touches the real user config.
    from qtpy.QtCore import QSettings

    monkeypatch.setattr(
        QSettings,
        "fileName",
        lambda self: str(tmp_path / "settings.ini"),
    )
    widget = mod.make_nucleus_tracking_widget(viewer)
    widget.set_context(work_dir=tmp_path)
    assert widget._settings().value("work_dir", "", type=str) == str(tmp_path)
