# tests/napari/test_nucleus_atom_extraction_widget.py
"""Tests for the Atom Extraction widget (controls + behavior mixin)."""
from __future__ import annotations

import os

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
    assert w.contour_window_spin.value() == 51
    assert abs(w.contour_floor_spin.value() - 0.01) < 1e-9
    assert w.atom_min_area_spin.value() == 100


def test_controls_have_activate_and_compute_and_overlays():
    _app()
    w = NucleusAtomExtractionWidget()
    assert w.active_btn.isCheckable()
    assert not w.active_btn.isChecked()
    assert w.compute_btn is not None
    assert w.territory_overlay_check is not None
    assert w.residual_overlay_check is not None


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


def _host(tmp_path):
    _app()
    fg = tmp_path / "fg.tif"
    contour = tmp_path / "contour.tif"
    rng = np.random.default_rng(0)
    tifffile.imwrite(fg, rng.random((3, 40, 40)).astype(np.float32))
    tifffile.imwrite(contour, rng.random((3, 40, 40)).astype(np.float32))
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


def test_activate_adds_preview_layer_then_deactivate_removes(tmp_path):
    h, viewer = _host(tmp_path)
    h._on_atom_activate(True)
    assert _ATOM_PREVIEW_LAYER in viewer.layers
    h._on_atom_activate(False)
    assert _ATOM_PREVIEW_LAYER not in viewer.layers
    napari.Viewer.close_all()


from cellflow.tracking_ultrack.atoms import read_atoms_params, params_fingerprint


def test_compute_atoms_full_stack_writes_tif_with_fingerprint(tmp_path):
    h, viewer = _host(tmp_path)
    h.atom_extraction_widget.fg_window_spin.setValue(11)
    h.atom_extraction_widget.contour_window_spin.setValue(11)
    h._compute_atoms_full_stack()
    out = tmp_path / "atoms.tif"
    assert out.exists()
    atoms = tifffile.imread(out)
    assert atoms.shape == (3, 40, 40)
    stored_params, stored_fp = read_atoms_params(out)
    assert stored_fp == params_fingerprint(h._atom_params())
    napari.Viewer.close_all()
