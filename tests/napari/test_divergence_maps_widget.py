"""Smoke tests for DivergenceMapsWidget."""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication


class _FakeViewer:
    def __init__(self):
        self.layers = {}
        self.dims = SimpleNamespace(current_step=(0,))

    def add_image(self, *a, **kw): pass
    def add_labels(self, *a, **kw): pass


def _load_widget(monkeypatch):
    """Bypass cellflow.napari.__init__ (which imports main_widget) so the
    widget module can be loaded standalone in tests.
    """
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.divergence_maps_widget", None)
    return importlib.import_module("cellflow.napari.divergence_maps_widget")


def test_widget_constructs_and_exposes_public_api(monkeypatch):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)

    w = mod.DivergenceMapsWidget(_FakeViewer())

    # Per-channel rows
    assert w.nucleus_run_btn.isEnabled() in (True, False)
    assert w.cell_run_btn.isEnabled() in (True, False)
    assert w.nucleus_params_btn.isCheckable()
    assert w.cell_params_btn.isCheckable()

    # Per-channel parameter spinners exist with default values from the spec.
    assert w.nuc_smoothing_spin.value() == pytest.approx(1.0)
    assert w.nuc_median_spin.value() == 0
    assert w.nuc_fg_reduction.currentText() == "mean"
    assert w.nuc_contour_reduction.currentText() == "mean"
    assert w.cell_smoothing_spin.value() == pytest.approx(1.0)
    assert w.cell_median_spin.value() == 0

    # Public API used by main_widget.
    assert hasattr(w, "refresh")
    assert hasattr(w, "get_state")
    assert hasattr(w, "set_state")
    assert hasattr(w, "output_files_tracker")
    w.deleteLater()


def test_widget_state_roundtrip(monkeypatch):
    QApplication.instance() or QApplication([])
    mod = _load_widget(monkeypatch)

    w = mod.DivergenceMapsWidget(_FakeViewer())
    w.nuc_smoothing_spin.setValue(0.5)
    w.nuc_median_spin.setValue(3)
    w.cell_fg_reduction.setCurrentText("max")
    state = w.get_state()
    assert state["nucleus"]["smoothing_sigma"] == pytest.approx(0.5)
    assert state["nucleus"]["median_radius"] == 3
    assert state["cell"]["foreground_z_reduction"] == "max"

    w2 = mod.DivergenceMapsWidget(_FakeViewer())
    w2.set_state(state)
    assert w2.nuc_smoothing_spin.value() == pytest.approx(0.5)
    assert w2.nuc_median_spin.value() == 3
    assert w2.cell_fg_reduction.currentText() == "max"
    w.deleteLater()
    w2.deleteLater()
