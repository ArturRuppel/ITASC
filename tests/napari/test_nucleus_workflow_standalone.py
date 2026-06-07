"""Standalone-mode seam for the nucleus tracking/correction piece.

The full orchestrator drives the widget through ``refresh(pos_dir)`` (staged
``<pos>/2_nucleus`` layout); the independently-installable ``cellflow-tracking``
plugin drives it through ``set_context(foreground=..., contours=...,
output_dir=...)`` — three explicit path fields in the widget.
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


def test_factory_builds_standalone_widget_with_path_pickers(viewer):
    widget = mod.make_nucleus_tracking_widget(viewer)
    assert widget._standalone is True
    # Standalone shows its own input/output path pickers and hides the staged
    # pipeline-files panel (which lists non-existent 1_cellpose/2_nucleus paths).
    assert widget._paths_container.isVisibleTo(widget)
    assert not widget._pipeline_files_section.isVisibleTo(widget)
    # Three explicit fields: two input files + an output directory.
    assert widget._foreground_edit is not None
    assert widget._contours_edit is not None
    assert widget._output_dir_edit is not None


def test_orchestrated_widget_hides_standalone_picker(viewer):
    widget = mod.NucleusWorkflowWidget(viewer)
    assert widget._standalone is False
    assert not widget._paths_container.isVisibleTo(widget)
    # Orchestrated refresh with no project is a no-op that clears the workspace.
    widget.refresh(None)
    assert widget._workspace is None


def test_set_context_builds_files_workspace(viewer, tmp_path):
    widget = mod.make_nucleus_tracking_widget(viewer)
    foreground = tmp_path / "inputs" / "nuc_fg.tif"
    contours = tmp_path / "inputs" / "nuc_contours.tif"
    out_dir = tmp_path / "out"
    widget.set_context(
        foreground=foreground, contours=contours, output_dir=out_dir
    )
    ws = widget._workspace
    assert isinstance(ws, NucleusWorkspace)
    # Inputs keep their explicit names/locations.
    assert ws.foreground == foreground
    assert ws.contours == contours
    # Every output is written under the chosen output directory.
    assert ws.nucleus_dir == out_dir
    assert ws.ultrack_db == out_dir / "ultrack_workdir" / "data.db"
    assert ws.tracked == out_dir / "tracked_labels.tif"
    # _pos_dir (the nucleus store dir threaded to the validation API) follows it.
    assert widget._pos_dir is None
    assert widget.nucleus_correction_widget._pos_dir == out_dir
    # The fields reflect the active workspace.
    assert widget._foreground_edit.text() == str(foreground)
    assert widget._contours_edit.text() == str(contours)
    assert widget._output_dir_edit.text() == str(out_dir)


def test_atom_controls_enabled_only_when_inputs_exist_on_disk(viewer, tmp_path):
    # The atom-extraction live-preview/run buttons read the foreground/contour
    # TIFFs, so they gate on those files *existing* — not merely on a workspace
    # being set. A workspace whose inputs are absent (e.g. a stale pair from
    # QSettings) must keep them disabled; clicking otherwise crashes in tifffile.
    import numpy as np
    import tifffile

    widget = mod.NucleusWorkflowWidget(viewer)
    w = widget.atom_extraction_widget
    assert not w.active_btn.isEnabled()
    assert not w.run_btn.isEnabled()

    # Workspace set but inputs missing → still disabled, gate recomputed.
    cellpose = tmp_path / "1_cellpose"
    cellpose.mkdir()
    widget.refresh(tmp_path)
    assert not w.active_btn.isEnabled()
    assert not w.run_btn.isEnabled()

    # Inputs now present on disk → controls enable.
    frame = np.zeros((2, 4, 4), dtype=np.float32)
    tifffile.imwrite(cellpose / "nucleus_foreground.tif", frame)
    tifffile.imwrite(cellpose / "nucleus_contours.tif", frame)
    widget.refresh(tmp_path)
    assert w.active_btn.isEnabled()
    assert w.run_btn.isEnabled()


def test_standalone_widget_is_top_anchored(viewer):
    # Standalone docks the widget directly (no AlignTop scroll wrapper), so it
    # must fill the dock vertically and pin its content to the top — otherwise
    # napari centres it once a section collapse shrinks its sizeHint.
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QSizePolicy

    widget = mod.make_nucleus_tracking_widget(viewer)
    assert widget.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Preferred
    assert widget.layout().alignment() & Qt.AlignTop
    # A trailing stretch absorbs slack at the bottom.
    last = widget.layout().itemAt(widget.layout().count() - 1)
    assert last.spacerItem() is not None


def test_embedded_widget_keeps_maximum_policy(viewer):
    # Embedded, main_widget's AlignTop scroll layout handles top-alignment, so
    # the widget stays compact (Maximum) and adds no trailing stretch.
    from qtpy.QtWidgets import QSizePolicy

    widget = mod.NucleusWorkflowWidget(viewer)
    assert widget.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Maximum
    last = widget.layout().itemAt(widget.layout().count() - 1)
    assert last.spacerItem() is None


def test_correction_takeover_roundtrip_keeps_header_above_stretch(viewer):
    # Standalone adds a trailing stretch to top-anchor the sections. When
    # correction focus-takeover deactivates, the header + section are re-inserted
    # into the layout; they must land *above* that stretch, not below it (which
    # would shove the correction pill to the bottom).
    widget = mod.make_nucleus_tracking_widget(viewer)
    layout = widget.layout()

    # Simulate the takeover round-trip: reparent the controls out, then back.
    widget._set_correction_focus_takeover(True)
    for w in (widget.correction_header, widget.correction_mode_section):
        layout.removeWidget(w)
    widget._set_correction_focus_takeover(False)

    # The trailing item is still the stretch, and the header/section sit above it.
    last = layout.itemAt(layout.count() - 1)
    assert last.spacerItem() is not None
    indices = {
        layout.indexOf(widget.correction_header),
        layout.indexOf(widget.correction_mode_section),
    }
    assert all(i < layout.count() - 1 for i in indices)


def test_set_context_defaults_output_dir_to_foreground_folder(viewer, tmp_path):
    widget = mod.make_nucleus_tracking_widget(viewer)
    foreground = tmp_path / "inputs" / "nuc_fg.tif"
    contours = tmp_path / "elsewhere" / "nuc_contours.tif"
    widget.set_context(foreground=foreground, contours=contours)
    ws = widget._workspace
    assert ws.nucleus_dir == foreground.parent


def test_set_context_work_dir_form_still_builds_flat_workspace(viewer, tmp_path):
    # The convenience flat-directory form is retained.
    widget = mod.make_nucleus_tracking_widget(viewer)
    widget.set_context(work_dir=tmp_path)
    ws = widget._workspace
    assert ws.nucleus_dir == tmp_path
    assert ws.foreground == tmp_path / "foreground.tif"
    assert ws.contours == tmp_path / "contours.tif"


def test_set_context_persists_paths_across_instances(viewer, tmp_path, monkeypatch):
    # Isolate QSettings so the test never touches the real user config.
    from qtpy.QtCore import QSettings

    monkeypatch.setattr(
        QSettings,
        "fileName",
        lambda self: str(tmp_path / "settings.ini"),
    )
    foreground = tmp_path / "nuc_fg.tif"
    contours = tmp_path / "nuc_contours.tif"
    out_dir = tmp_path / "out"
    widget = mod.make_nucleus_tracking_widget(viewer)
    widget.set_context(foreground=foreground, contours=contours, output_dir=out_dir)
    s = widget._settings()
    assert s.value("foreground", "", type=str) == str(foreground)
    assert s.value("contours", "", type=str) == str(contours)
    assert s.value("output_dir", "", type=str) == str(out_dir)
