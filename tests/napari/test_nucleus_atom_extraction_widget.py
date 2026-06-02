# tests/napari/test_nucleus_atom_extraction_widget.py
"""Tests for the Atom Extraction widget (controls + behavior mixin)."""
from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import napari
import numpy as np
import tifffile
from qtpy.QtWidgets import QApplication, QWidget

from cellflow.napari.nucleus_atom_extraction_widget import (
    NucleusAtomExtractionWidget,
)

_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def test_controls_have_spec_defaults():
    _app()
    w = NucleusAtomExtractionWidget()
    assert w.fg_window_spin.value() == 51
    assert abs(w.fg_cutoff_spin.value() - 0.002) < 1e-9
    assert abs(w.fg_strength_spin.value() - 1.0) < 1e-9
    assert w.contour_window_spin.value() == 51
    assert abs(w.contour_floor_spin.value() - 0.01) < 1e-9
    assert abs(w.contour_strength_spin.value() - 1.0) < 1e-9
    assert w.atom_min_area_spin.value() == 100


def test_controls_have_stage_row_buttons():
    _app()
    w = NucleusAtomExtractionWidget()
    assert hasattr(w, "params_btn") and w.params_btn.isCheckable()
    assert not w.params_btn.isChecked()
    assert hasattr(w, "active_btn") and w.active_btn.isCheckable()
    assert not w.active_btn.isChecked()
    assert hasattr(w, "run_btn") and not w.run_btn.isCheckable()
    assert not hasattr(w, "territory_overlay_check")
    assert not hasattr(w, "residual_overlay_check")
    assert not hasattr(w, "compute_btn")


def test_params_btn_toggles_section():
    _app()
    w = NucleusAtomExtractionWidget()
    from cellflow.napari.widgets import CollapsibleSection
    assert isinstance(w.section, CollapsibleSection)
    assert not w.section.is_expanded
    w.params_btn.setChecked(True)
    assert w.section.is_expanded
    w.params_btn.setChecked(False)
    assert not w.section.is_expanded


from cellflow.napari.nucleus_atom_extraction_widget import (
    NucleusAtomExtractionMixin,
    _ATOM_PREVIEW_LAYER,
)
from cellflow.tracking_ultrack.atoms import AtomParams


class _Host(NucleusAtomExtractionMixin, QWidget):
    """Minimal host exposing the surface the mixin needs.

    Subclasses QWidget so the mixin's ``QTimer(self)`` has a valid QObject
    parent — exactly as the real workflow widget (a QWidget) provides.
    """

    def __init__(self, viewer, fg, contour, out_path):
        super().__init__()
        self.viewer = viewer
        self._fg = fg
        self._contour = contour
        self._out_path = out_path
        self._init_atom_extraction_state()
        self.atom_extraction_widget = NucleusAtomExtractionWidget()
        self._alias_atom_extraction_controls()

    def _current_t(self):
        dims = self.viewer.dims
        return int(dims.current_step[0]) if dims.ndim else 0

    def _atom_fg_path(self):
        return self._fg

    def _atom_contour_path(self):
        return self._contour

    def _atom_output_path(self):
        return self._out_path


def _drain_atom_preview(h, timeout=10.0):
    """Pump the Qt loop until the (async) preview worker has finished."""
    deadline = time.time() + timeout
    while getattr(h, "_atom_preview_worker", None) is not None and time.time() < deadline:
        _app().processEvents()
        time.sleep(0.005)
    _app().processEvents()


def _host(tmp_path):
    _app()
    fg = tmp_path / "fg.tif"
    contour = tmp_path / "contour.tif"
    rng = np.random.default_rng(0)
    # Write page-per-frame TIFFs matching production format (imwrite_grayscale).
    # photometric='minisblack' causes tifffile to store each leading-axis slice
    # as a separate IFD page, so tifffile.imread(path, key=t) returns a 2-D frame.
    tifffile.imwrite(str(fg), rng.random((3, 40, 40)).astype(np.float32), photometric="minisblack")
    tifffile.imwrite(str(contour), rng.random((3, 40, 40)).astype(np.float32), photometric="minisblack")
    viewer = napari.Viewer(show=False)
    viewer.add_image(np.asarray(tifffile.imread(fg)), name="fg")
    return _Host(viewer, fg, contour, tmp_path / "atoms.tif"), viewer


def test_atom_params_reads_controls():
    _app()
    h = _Host(napari.Viewer(show=False),
              fg=None, contour=None, out_path=None)  # noqa: viewer unused here
    h.atom_extraction_widget.fg_cutoff_spin.setValue(0.01)
    h.atom_extraction_widget.atom_min_area_spin.setValue(250)
    p = h._atom_params()
    assert isinstance(p, AtomParams)
    assert p.fg_cutoff == 0.01
    assert p.atom_min_area == 250
    napari.Viewer.close_all()


def test_activate_adds_five_layers_then_deactivate_removes(tmp_path):
    from cellflow.napari.nucleus_atom_extraction_widget import (
        _ATOM_TERRITORY_LAYER,
        _ATOM_FG_RESIDUAL_LAYER,
        _ATOM_CONTOUR_RESIDUAL_LAYER,
        _ATOM_RIDGE_LAYER,
    )
    names = (_ATOM_PREVIEW_LAYER, _ATOM_TERRITORY_LAYER,
             _ATOM_FG_RESIDUAL_LAYER, _ATOM_CONTOUR_RESIDUAL_LAYER,
             _ATOM_RIDGE_LAYER)
    h, viewer = _host(tmp_path)
    h._on_atom_activate(True)
    _drain_atom_preview(h)
    for name in names:
        assert name in viewer.layers
        # Preview layers carry the full time axis (T, Y, X) so the viewer has a
        # frame slider even without an open movie.
        assert viewer.layers[name].data.shape == (3, 40, 40)
    h._on_atom_activate(False)
    for name in names:
        assert name not in viewer.layers
    napari.Viewer.close_all()


def test_atoms_layer_is_named_atoms():
    # the atoms label layer is "[Atoms] atoms" (was "[Atoms] preview").
    assert _ATOM_PREVIEW_LAYER == "[Atoms] atoms"


def test_default_visibility_on_activate_foreground_only(tmp_path):
    from cellflow.napari.nucleus_atom_extraction_widget import (
        _ATOM_FG_GROUP_LAYERS,
        _ATOM_CONTOUR_GROUP_LAYERS,
    )
    h, viewer = _host(tmp_path)
    h._on_atom_activate(True)
    _drain_atom_preview(h)
    # Foreground pair visible, Contour group hidden — and the checkboxes match.
    assert h.atom_extraction_widget.fg_visible_check.isChecked()
    assert not h.atom_extraction_widget.contour_visible_check.isChecked()
    for name in _ATOM_FG_GROUP_LAYERS:
        assert viewer.layers[name].visible
    for name in _ATOM_CONTOUR_GROUP_LAYERS:
        assert not viewer.layers[name].visible
    napari.Viewer.close_all()


def test_group_checkboxes_flip_their_layers_together(tmp_path):
    from cellflow.napari.nucleus_atom_extraction_widget import (
        _ATOM_FG_GROUP_LAYERS,
        _ATOM_CONTOUR_GROUP_LAYERS,
    )
    h, viewer = _host(tmp_path)
    h._on_atom_activate(True)
    _drain_atom_preview(h)
    w = h.atom_extraction_widget

    # Contour checkbox on → all three contour layers visible together.
    w.contour_visible_check.setChecked(True)
    for name in _ATOM_CONTOUR_GROUP_LAYERS:
        assert viewer.layers[name].visible
    # Foreground checkbox off → both foreground layers hidden together.
    w.fg_visible_check.setChecked(False)
    for name in _ATOM_FG_GROUP_LAYERS:
        assert not viewer.layers[name].visible
    # Independent: contour stays visible while foreground is hidden.
    for name in _ATOM_CONTOUR_GROUP_LAYERS:
        assert viewer.layers[name].visible
    napari.Viewer.close_all()


def test_ridge_layer_is_exactly_the_core_return(tmp_path):
    from cellflow.napari.nucleus_atom_extraction_widget import _ATOM_RIDGE_LAYER
    from cellflow.tracking_ultrack.atoms import residual, extract_atoms_frame

    h, viewer = _host(tmp_path)
    h._on_atom_activate(True)
    _drain_atom_preview(h)
    t = h._current_t()
    params = h._atom_params()
    fg = h._read_frame(h._atom_fg_path(), t)
    contour = h._read_frame(h._atom_contour_path(), t)
    rf = residual(fg, params.fg_window, params.fg_strength)
    rc = residual(contour, params.contour_window, params.contour_strength)
    _atoms, ridge = extract_atoms_frame(
        rc, rf > params.fg_cutoff, params.contour_floor, params.atom_min_area
    )
    # the displayed ridge slice equals the core's returned mask, not a re-derivation.
    assert np.array_equal(viewer.layers[_ATOM_RIDGE_LAYER].data[t], ridge.astype(np.int32))
    napari.Viewer.close_all()


def test_contrast_limits_preserved_across_refresh(tmp_path):
    from cellflow.napari.nucleus_atom_extraction_widget import (
        _ATOM_FG_RESIDUAL_LAYER,
        _ATOM_CONTOUR_RESIDUAL_LAYER,
    )
    h, viewer = _host(tmp_path)
    h._on_atom_activate(True)
    _drain_atom_preview(h)
    # user dials in a non-default contrast on both residual image layers
    viewer.layers[_ATOM_FG_RESIDUAL_LAYER].contrast_limits = (0.1, 0.2)
    viewer.layers[_ATOM_CONTOUR_RESIDUAL_LAYER].contrast_limits = (0.3, 0.4)
    h.atom_extraction_widget.contour_floor_spin.setValue(0.02)
    h._refresh_atom_preview()
    _drain_atom_preview(h)
    assert tuple(viewer.layers[_ATOM_FG_RESIDUAL_LAYER].contrast_limits) == (0.1, 0.2)
    assert tuple(viewer.layers[_ATOM_CONTOUR_RESIDUAL_LAYER].contrast_limits) == (0.3, 0.4)
    napari.Viewer.close_all()


from cellflow.tracking_ultrack.atoms import read_atoms_params, params_fingerprint


def test_run_atom_extraction_writes_tif_and_shows_layers(tmp_path):
    from cellflow.napari.nucleus_atom_extraction_widget import (
        _ATOM_TERRITORY_LAYER,
        _ATOM_FG_RESIDUAL_LAYER,
        _ATOM_CONTOUR_RESIDUAL_LAYER,
        _ATOM_RIDGE_LAYER,
    )
    h, viewer = _host(tmp_path)
    h.atom_extraction_widget.fg_window_spin.setValue(11)
    h.atom_extraction_widget.contour_window_spin.setValue(11)
    h._run_atom_extraction()
    out = tmp_path / "atoms.tif"
    assert out.exists()
    atoms = tifffile.imread(out)
    assert atoms.shape == (3, 40, 40)
    stored_params, stored_fp = read_atoms_params(out)
    assert stored_fp == params_fingerprint(h._atom_params())
    for name in (_ATOM_PREVIEW_LAYER, _ATOM_TERRITORY_LAYER,
                 _ATOM_FG_RESIDUAL_LAYER, _ATOM_CONTOUR_RESIDUAL_LAYER,
                 _ATOM_RIDGE_LAYER):
        assert name in viewer.layers
    assert viewer.layers[_ATOM_PREVIEW_LAYER].data.ndim == 3
    napari.Viewer.close_all()


# ── Task 10: workflow widget wiring ──────────────────────────────────────────


def test_workflow_widget_builds_atom_extraction_section():
    _app()
    from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget
    w = NucleusWorkflowWidget(napari.Viewer(show=False))
    assert hasattr(w, "atom_extraction_widget")
    assert w.atom_extraction_section is not None
    # path hooks resolve to None without a loaded position (no crash)
    assert w._atom_output_path() is None or str(w._atom_output_path()).endswith("atoms.tif")
    napari.Viewer.close_all()


# ── Task 11: state round-trip ─────────────────────────────────────────────────


def test_state_round_trip_for_atom_params():
    _app()
    from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget
    from cellflow.napari._state import dump_state, load_state

    w = NucleusWorkflowWidget(napari.Viewer(show=False))
    w.atom_extraction_widget.fg_cutoff_spin.setValue(0.01)
    w.atom_extraction_widget.fg_strength_spin.setValue(0.5)
    w.atom_extraction_widget.contour_strength_spin.setValue(0.25)
    w.atom_extraction_widget.atom_min_area_spin.setValue(300)
    state = dump_state(w)
    assert state["atom_extraction"]["fg_cutoff"] == 0.01
    assert state["atom_extraction"]["fg_strength"] == 0.5
    assert state["atom_extraction"]["contour_strength"] == 0.25
    assert state["atom_extraction"]["atom_min_area"] == 300

    w2 = NucleusWorkflowWidget(napari.Viewer(show=False))
    load_state(w2, state)
    assert w2.atom_extraction_widget.fg_cutoff_spin.value() == 0.01
    assert w2.atom_extraction_widget.fg_strength_spin.value() == 0.5
    assert w2.atom_extraction_widget.contour_strength_spin.value() == 0.25
    assert w2.atom_extraction_widget.atom_min_area_spin.value() == 300
    napari.Viewer.close_all()
