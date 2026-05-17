"""Tests for the Cellpose prob z-avg compute widget."""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication


def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.cellpose_zavg_viz_widget", None)
    return importlib.import_module("cellflow.napari.cellpose_zavg_viz_widget")


def _make_widget(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellposeZavgVizWidget()
    return app, mod, widget


def _make_pos(tmp_path: Path) -> Path:
    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    return pos_dir


def _write_prob_3dt(pos_dir: Path, channel: str, data: np.ndarray) -> Path:
    path = pos_dir / "1_cellpose" / f"{channel}_prob_3dt.tif"
    tifffile.imwrite(str(path), data)
    return path


def test_button_disabled_when_no_project(monkeypatch):
    app, _mod, widget = _make_widget(monkeypatch)

    widget.refresh(None)

    assert not widget.compute_btn.isEnabled()
    assert "no project" in widget.status_lbl.text().lower()

    widget.deleteLater()
    app.processEvents()


def test_button_disabled_when_3dt_files_missing(monkeypatch, tmp_path):
    app, _mod, widget = _make_widget(monkeypatch)
    pos_dir = _make_pos(tmp_path)

    widget.refresh(pos_dir)

    assert not widget.compute_btn.isEnabled()
    assert "missing" in widget.status_lbl.text().lower()
    assert "nucleus_prob_3dt.tif" in widget.status_lbl.text()
    assert "cell_prob_3dt.tif" in widget.status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_button_enabled_when_both_3dt_files_present(monkeypatch, tmp_path):
    app, _mod, widget = _make_widget(monkeypatch)
    pos_dir = _make_pos(tmp_path)
    _write_prob_3dt(pos_dir, "nucleus", np.zeros((1, 2, 4, 4), dtype=np.float32))
    _write_prob_3dt(pos_dir, "cell", np.zeros((1, 2, 4, 4), dtype=np.float32))

    widget.refresh(pos_dir)

    assert widget.compute_btn.isEnabled()
    assert widget.status_lbl.text() == ""

    widget.deleteLater()
    app.processEvents()


def test_compute_writes_zavg_files(monkeypatch, tmp_path):
    app, _mod, widget = _make_widget(monkeypatch)
    pos_dir = _make_pos(tmp_path)
    T, Z, Y, X = 2, 3, 4, 4
    rng = np.random.default_rng(0)
    nucleus = (rng.random((T, Z, Y, X)).astype(np.float32) - 0.5) * 4.0
    cell = (rng.random((T, Z, Y, X)).astype(np.float32) - 0.5) * 4.0
    _write_prob_3dt(pos_dir, "nucleus", nucleus)
    _write_prob_3dt(pos_dir, "cell", cell)

    widget.refresh(pos_dir)
    widget._on_compute()

    nuc_zavg_path = pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif"
    cell_zavg_path = pos_dir / "1_cellpose" / "cell_prob_zavg.tif"
    assert nuc_zavg_path.is_file()
    assert cell_zavg_path.is_file()

    expected_nuc = (1.0 / (1.0 + np.exp(-nucleus))).mean(axis=1).astype(np.float32)
    actual_nuc = tifffile.imread(str(nuc_zavg_path))
    assert actual_nuc.shape == (T, Y, X)
    assert np.allclose(actual_nuc, expected_nuc, atol=1e-5)

    assert "wrote" in widget.status_lbl.text().lower()

    widget.deleteLater()
    app.processEvents()


def test_refresh_updates_state_when_pos_dir_changes(monkeypatch, tmp_path):
    app, _mod, widget = _make_widget(monkeypatch)
    pos_a = tmp_path / "pos00"
    (pos_a / "1_cellpose").mkdir(parents=True)
    pos_b = tmp_path / "pos01"
    (pos_b / "1_cellpose").mkdir(parents=True)
    _write_prob_3dt(pos_b, "nucleus", np.zeros((1, 2, 4, 4), dtype=np.float32))
    _write_prob_3dt(pos_b, "cell", np.zeros((1, 2, 4, 4), dtype=np.float32))

    widget.refresh(pos_a)
    assert not widget.compute_btn.isEnabled()

    widget.refresh(pos_b)
    assert widget.compute_btn.isEnabled()

    widget.deleteLater()
    app.processEvents()
