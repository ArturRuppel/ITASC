"""Standalone-mode seam for the nucleus tracking/correction piece.

The full orchestrator drives the widget through ``refresh(pos_dir)`` (staged
``<pos>/2_nucleus`` layout); the independently-installable ``itasc-tracking``
plugin drives it through ``set_context(foreground=..., contours=...,
output_dir=...)`` — three explicit path fields in the widget.
"""
from __future__ import annotations

from types import SimpleNamespace

import napari
import pytest

from itasc.napari import nucleus_workflow_widget as mod
from itasc.napari._paths import NucleusWorkspace


@pytest.fixture()
def viewer():
    v = napari.Viewer(show=False)
    try:
        yield v
    finally:
        v.close()


class _LayerCollection(dict):
    """Stand-in for napari's ``LayerList``, keyed by layer name.

    Covers exactly the operations the atom-preview path uses — ``name in
    layers``, ``layers[name]`` and ``layers.remove(name)`` — so the widget can
    manage layers without a GL-backed viewer.
    """

    def remove(self, item) -> None:
        # The widget removes by name; tolerate a layer object too.
        name = item if isinstance(item, str) else getattr(item, "name", item)
        self.pop(name, None)


class _HeadlessViewer:
    """A viewer that builds genuine napari layer *models* but never a canvas.

    The atom live-preview only needs to add/remove layers and read the current
    frame index. Constructing real ``Image``/``Labels`` keeps the widget's
    ``isinstance``/``.visible``/``.contrast_limits``/``.refresh()`` logic honest
    while sidestepping the vispy GL context a real ``napari.Viewer`` demands —
    which a headless CI box (no xvfb / software GL) can't provide.
    """

    def __init__(self) -> None:
        self.layers = _LayerCollection()
        self.dims = SimpleNamespace(current_step=(0, 0))

    def add_image(self, data, *, name, **kwargs):
        from napari.layers import Image

        layer = Image(data, name=name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_labels(self, data, *, name, **kwargs):
        from napari.layers import Labels

        layer = Labels(data, name=name, **kwargs)
        self.layers[name] = layer
        return layer


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


def _sync_thread_worker():
    """A drop-in for ``thread_worker`` that runs the body inline (no Qt thread)."""
    def fake(connect=None):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:  # pragma: no cover - defensive
                    if connect and "errored" in connect:
                        connect["errored"](exc)
                    return None
                if connect and "returned" in connect:
                    connect["returned"](result)
                return type("W", (), {"quit": staticmethod(lambda: None)})()
            return wrapper
        return decorator
    return fake


def test_atom_compute_checkboxes_gate_which_layers_compute(viewer, tmp_path, monkeypatch):
    # The atom-extraction live preview computes only the ticked Compute stages
    # (Foreground, Contour) — an unticked stage's layers are not created at all,
    # rather than computed-then-hidden.
    import numpy as np
    import tifffile

    from itasc.napari import nucleus_atom_extraction_widget as atom_mod

    monkeypatch.setattr(atom_mod, "thread_worker", _sync_thread_worker())

    cellpose = tmp_path / "1_cellpose"
    cellpose.mkdir()
    rng = np.random.default_rng(0)
    fg = np.clip(rng.normal(0.6, 0.1, (2, 16, 16)), 0, 1).astype(np.float32)
    contours = np.abs(rng.normal(0, 1, (2, 16, 16))).astype(np.float32)
    tifffile.imwrite(cellpose / "nucleus_foreground.tif", fg)
    tifffile.imwrite(cellpose / "nucleus_contours.tif", contours)

    widget = mod.NucleusWorkflowWidget(viewer)
    widget.refresh(tmp_path)
    # The preview path reads ``self.viewer`` dynamically, so swap in a headless
    # viewer for it: adding real Image/Labels layers to a live napari canvas
    # needs a GL context that CI hasn't got, and the preview doesn't render —
    # it only manages layer membership and reads the current frame.
    widget.viewer = _HeadlessViewer()
    layers = widget.viewer.layers
    w = widget.atom_extraction_widget

    # Default: only Foreground is ticked. Activating the preview creates the FG
    # stage's layers and computes nothing for the (unticked) Contour stage.
    assert w.fg_check.isChecked() and not w.contour_check.isChecked()
    w.active_btn.setChecked(True)
    for name in atom_mod._ATOM_FG_GROUP_LAYERS:
        assert name in layers, f"missing foreground layer {name}"
    for name in atom_mod._ATOM_CONTOUR_GROUP_LAYERS:
        assert name not in layers

    # Ticking Contour upgrades the compute and creates its layers (residual,
    # ridge, atoms).
    widget._atom_preview_worker = None  # settle the synchronous worker
    w.contour_check.setChecked(True)
    for name in atom_mod._ATOM_CONTOUR_GROUP_LAYERS:
        assert name in layers, f"missing contour layer {name}"

    # Unticking Foreground drops its layers immediately.
    widget._atom_preview_worker = None
    w.fg_check.setChecked(False)
    for name in atom_mod._ATOM_FG_GROUP_LAYERS:
        assert name not in layers

    # Deactivation tears every atom layer down.
    w.active_btn.setChecked(False)
    for name in atom_mod._ATOM_LAYERS:
        assert name not in layers


def test_atom_run_refreshes_pipeline_files(viewer, tmp_path):
    """Running atom extraction writes atoms.tif and refreshes the files panel.

    Regression: the run wrote ``2_nucleus/atoms.tif`` but never refreshed the
    tracker, so its ``refreshed`` signal never fired and the section dot /
    catalog rail stayed stale until a manual Refresh.
    """
    import numpy as np
    import tifffile

    cellpose = tmp_path / "1_cellpose"
    cellpose.mkdir()
    rng = np.random.default_rng(0)
    fg = np.clip(rng.normal(0.6, 0.1, (2, 16, 16)), 0, 1).astype(np.float32)
    contours = np.abs(rng.normal(0, 1, (2, 16, 16))).astype(np.float32)
    tifffile.imwrite(cellpose / "nucleus_foreground.tif", fg)
    tifffile.imwrite(cellpose / "nucleus_contours.tif", contours)

    widget = mod.NucleusWorkflowWidget(viewer)
    widget.refresh(tmp_path)
    # The run adds atom layers; swap in a headless viewer (see the compute-
    # checkbox test for why a live canvas can't start on CI).
    widget.viewer = _HeadlessViewer()

    atoms_row = widget._files_widget._rows_by_group["Intermediates"][0]
    assert atoms_row._full_path is None  # atoms.tif absent to start

    widget._run_atom_extraction()

    atoms = tmp_path / "2_nucleus" / "atoms.tif"
    assert atoms.is_file()
    # The tracker refreshed, so the intermediate row now sees the file on disk.
    assert atoms_row._full_path == atoms


def test_correction_save_refreshes_pipeline_files(viewer, tmp_path):
    """Saving corrected nucleus labels refreshes the host's Pipeline Files.

    The working ``tracked_labels.tif`` changes on save; without a refresh the
    section dot / catalog rail stay stale. The cell correction widget already
    refreshed on save — the nucleus one was never wired.
    """
    import numpy as np

    widget = mod.NucleusWorkflowWidget(viewer)
    widget.refresh(tmp_path)  # staged workspace at <tmp>/2_nucleus
    widget.viewer = _HeadlessViewer()
    cw = widget.nucleus_correction_widget
    cw.viewer = widget.viewer
    cw.viewer.add_labels(
        np.zeros((2, 4, 4), dtype=np.uint32), name="Tracked: Nucleus"
    )

    out_row = widget._files_widget._rows_by_group["Output"][0]
    assert out_row._full_path is None  # tracked_labels.tif absent to start

    cw._on_save_tracked()

    tracked = tmp_path / "2_nucleus" / "tracked_labels.tif"
    assert tracked.is_file()
    # The host callback refreshed the tracker, surfacing the saved file.
    assert out_row._full_path == tracked


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
    # Standalone adds a trailing stretch to top-anchor the sections, AND napari's
    # add_dock_widget appends its own — so a docked standalone widget carries two
    # trailing spacers. When correction focus-takeover deactivates, the header +
    # section are re-inserted; they must land above *both* spacers, not stranded
    # between them (which shoves the correction pill to the bottom).
    widget = mod.make_nucleus_tracking_widget(viewer)
    # Dock it so napari contributes its extra trailing stretch (the regression
    # only reproduces with more than one spacer at the tail).
    viewer.window.add_dock_widget(widget, name="Nucleus", area="right")
    layout = widget.layout()

    trailing_spacers = sum(
        1
        for i in range(layout.count())
        if layout.itemAt(i).spacerItem() is not None
    )
    assert trailing_spacers >= 2

    # Simulate the takeover round-trip: reparent the controls out, then back.
    widget._set_correction_focus_takeover(True)
    for w in (widget.correction_header, widget.correction_mode_section):
        layout.removeWidget(w)
    widget._set_correction_focus_takeover(False)

    # Every trailing spacer is still last, and the header/section sit above them
    # all (so the pill keeps its position rather than dropping between spacers).
    first_trailing_spacer = layout.count()
    while (
        first_trailing_spacer > 0
        and layout.itemAt(first_trailing_spacer - 1).spacerItem() is not None
    ):
        first_trailing_spacer -= 1
    assert first_trailing_spacer < layout.count()  # at least one trailing spacer
    indices = {
        layout.indexOf(widget.correction_header),
        layout.indexOf(widget.correction_mode_section),
    }
    assert all(i < first_trailing_spacer for i in indices)


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


def test_finalize_promotes_nucleus_tracked_labels_to_base_folder(viewer, tmp_path):
    import numpy as np
    import tifffile

    widget = mod.NucleusWorkflowWidget(viewer)

    # Staged position, no tracked labels yet → Commit disabled.
    widget.refresh(tmp_path)
    assert widget.finalize_btn.isEnabled() is False

    # A working 2_nucleus/tracked_labels.tif → Commit enabled.
    working = tmp_path / "2_nucleus" / "tracked_labels.tif"
    working.parent.mkdir(parents=True)
    arr = np.zeros((4, 4), dtype=np.uint16)
    arr[1:3, 1:3] = 9
    tifffile.imwrite(str(working), arr)
    widget.refresh(tmp_path)
    assert widget.finalize_btn.isEnabled() is True

    # Commit promotes it to the base-folder final output.
    widget._on_finalize()
    committed = tmp_path / "nucleus_labels.tif"
    assert committed.is_file()
    np.testing.assert_array_equal(tifffile.imread(str(committed)), arr)
