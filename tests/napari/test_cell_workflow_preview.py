"""Tests for the cell Contour Maps widget (CellWorkflowWidget)."""
from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QWidget


class _LayerCollection(dict):
    def remove(self, layer):
        self.pop(layer.name, None)


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _LayerCollection()
        self.dims = SimpleNamespace(
            current_step=(0,),
            events=SimpleNamespace(
                current_step=SimpleNamespace(connect=lambda cb: None)
            ),
        )

    def add_image(self, data, *, name, **kwargs):
        layer = SimpleNamespace(data=np.asarray(data), name=name, **kwargs)
        self.layers[name] = layer
        return layer


def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.cell_workflow_widget", None)
    mod = importlib.import_module("cellflow.napari.cell_workflow_widget")
    monkeypatch.setitem(sys.modules, "cellflow.napari.cell_workflow_widget", mod)
    return mod


def _make_sync_thread_worker():
    """Return a replacement for napari thread_worker that runs synchronously."""

    def fake_thread_worker(connect=None):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    if connect and "errored" in connect:
                        connect["errored"](exc)
                    return None

                if inspect.isgenerator(result):
                    return_value = None
                    while True:
                        try:
                            yielded = next(result)
                        except StopIteration as exc:
                            return_value = exc.value
                            break
                        if connect and "yielded" in connect:
                            connect["yielded"](yielded)
                    if connect and "returned" in connect:
                        connect["returned"](return_value)
                else:
                    if connect and "returned" in connect:
                        connect["returned"](result)
                return None

            return wrapper

        return decorator

    return fake_thread_worker


# ── get_state / set_state round-trip ─────────────────────────────────────────

def test_get_set_state_round_trips_cellprob_params(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    state = {
        "cellprob": {
            "min": -5.0, "max": 1.0, "step": 2.0,
            "gamma_min": 0.8, "gamma_max": 1.2, "gamma_step": 0.1,
        }
    }
    widget.set_state(state)
    got = widget.get_state()

    assert got["cellprob"]["min"]        == -5.0
    assert got["cellprob"]["max"]        ==  1.0
    assert got["cellprob"]["step"]       ==  2.0
    assert got["cellprob"]["gamma_min"]  ==  0.8
    assert got["cellprob"]["gamma_max"]  ==  1.2
    assert got["cellprob"]["gamma_step"] ==  0.1

    widget.deleteLater()
    app.processEvents()


# ── status labels ─────────────────────────────────────────────────────────────

def test_refresh_none_shows_no_project_message(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    widget.refresh(None)

    assert "no project" in widget.contour_input_lbl.text().lower()
    assert "no project" in widget.contour_output_lbl.text().lower()

    widget.deleteLater()
    app.processEvents()


def test_refresh_existing_files_shows_checkmarks(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif", np.zeros((1, 2, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_dp_3dt.tif",   np.zeros((1, 2, 2, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "3_cell" / "contour_maps.tif",      np.zeros((1, 4, 4), dtype=np.float32))

    widget.refresh(pos_dir)

    assert "✓" in widget.contour_input_lbl.text()
    assert "✓" in widget.contour_output_lbl.text()

    widget.deleteLater()
    app.processEvents()


# ── _thresholds / _cp_gammas helpers ──────────────────────────────────────────

def test_thresholds_returns_correct_values(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    widget.cp_min_spin.setValue(-2.0)
    widget.cp_max_spin.setValue(0.0)
    widget.cp_step_spin.setValue(1.0)

    thr = widget._thresholds()

    assert len(thr) == 3
    np.testing.assert_allclose(thr, [-2.0, -1.0, 0.0], atol=1e-6)

    widget.deleteLater()
    app.processEvents()


def test_cp_gammas_single_value_when_min_equals_max(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    widget.cp_gamma_min_spin.setValue(1.0)
    widget.cp_gamma_max_spin.setValue(1.0)
    widget.cp_gamma_step_spin.setValue(0.25)

    gammas = widget._cp_gammas()

    assert gammas == [1.0]

    widget.deleteLater()
    app.processEvents()


# ── preview worker ────────────────────────────────────────────────────────────

def test_on_preview_contour_maps_calls_build_mean_z_consensus_boundary(monkeypatch, tmp_path):
    """Preview should call build_mean_z_consensus_boundary and add napari layers."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    # Force thread_worker to run synchronously so patches apply inside the worker.
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)

    n_t, n_z, n_y, n_x = 2, 3, 8, 8
    prob = np.zeros((n_t, n_z, n_y, n_x), dtype=np.float32)
    dp   = np.zeros((n_t, n_z, 2, n_y, n_x), dtype=np.float32)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_dp_3dt.tif",   dp)

    boundary_result = np.full((n_y, n_x), 0.5, dtype=np.float32)
    fg_result       = np.full((n_y, n_x), 0.4, dtype=np.float32)

    viewer = _FakeViewer()
    viewer.dims.current_step = (1,)
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    with patch(
        "cellflow.segmentation.build_mean_z_consensus_boundary",
        return_value=(boundary_result, fg_result),
    ) as mock_fn:
        widget._on_preview_contour_maps()

    mock_fn.assert_called_once()
    call_args = mock_fn.call_args
    # First positional arg is prob[t_idx]: shape (n_z, n_y, n_x)
    assert call_args[0][0].shape == (n_z, n_y, n_x)
    # Second positional arg is dp[t_idx]: shape (n_z, 2, n_y, n_x)
    assert call_args[0][1].shape == (n_z, 2, n_y, n_x)

    # Both napari layers should have been added
    assert mod._CONTOUR_LAYER in viewer.layers
    assert mod._CELLPROB_LAYER in viewer.layers

    # Contour layer is a T-stack; boundary appears at t=1
    contour_data = viewer.layers[mod._CONTOUR_LAYER].data
    assert contour_data.shape == (n_t, n_y, n_x)
    np.testing.assert_array_equal(contour_data[1], boundary_result)
    np.testing.assert_array_equal(contour_data[0], 0.0)

    widget.deleteLater()
    app.processEvents()


# ── build worker writes output file ───────────────────────────────────────────

def test_on_build_contour_maps_writes_tif_file(monkeypatch, tmp_path):
    """Build should stack per-frame boundaries and write contour_maps.tif."""
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    # Force thread_worker to run synchronously so patches apply inside the worker.
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)

    n_t, n_z, n_y, n_x = 3, 2, 6, 6
    prob = np.zeros((n_t, n_z, n_y, n_x), dtype=np.float32)
    dp   = np.zeros((n_t, n_z, 2, n_y, n_x), dtype=np.float32)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_dp_3dt.tif",   dp)

    call_count = 0

    def fake_build(prob_t, dp_t, thresholds, gammas, **kwargs):
        nonlocal call_count
        call_count += 1
        boundary = np.full((n_y, n_x), float(call_count) * 0.1, dtype=np.float32)
        fg       = np.zeros((n_y, n_x), dtype=np.float32)
        return boundary, fg

    widget = mod.CellWorkflowWidget(_FakeViewer())
    widget.refresh(pos_dir)

    with patch("cellflow.segmentation.build_mean_z_consensus_boundary", fake_build):
        widget._on_build_contour_maps()

    contour_path = pos_dir / "3_cell" / "contour_maps.tif"
    assert contour_path.exists(), "contour_maps.tif was not written"
    result = tifffile.imread(str(contour_path))
    assert result.shape == (n_t, n_y, n_x)
    assert call_count == n_t
    np.testing.assert_allclose(result[0], 0.1, atol=1e-5)
    np.testing.assert_allclose(result[2], 0.3, atol=1e-5)

    widget.deleteLater()
    app.processEvents()
