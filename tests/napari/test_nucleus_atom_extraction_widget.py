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
