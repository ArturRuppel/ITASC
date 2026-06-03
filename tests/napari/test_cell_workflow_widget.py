"""Tests for the simplified divergence-based cell workflow widget."""
from __future__ import annotations

import importlib
import inspect
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QToolButton,
)


class _LayerCollection(dict):
    def remove(self, layer):
        name = layer if isinstance(layer, str) else layer.name
        self.pop(name, None)


class _FakeEvent:
    def connect(self, cb):
        pass

    def disconnect(self, cb):
        pass


class _FakeEvents:
    def __init__(self) -> None:
        self.data = _FakeEvent()
        self.paint = _FakeEvent()
        self.mode = _FakeEvent()
        self.removed = _FakeEvent()


class _FakeSelection:
    def __init__(self) -> None:
        self.active = None


class _FakeLayer:
    def __init__(self, data, name, **kwargs) -> None:
        self.data = np.asarray(data)
        self.name = name
        self.mode = "pan_zoom"
        self.contour = 0
        self.visible = True
        self.opacity = kwargs.get("opacity", 1.0)
        self.colormap = kwargs.get("colormap")
        self.contrast_limits = (0.0, 1.0)
        self.events = _FakeEvents()
        self.mouse_drag_callbacks = []
        self.kwargs = kwargs

    def bind_key(self, key, fn, overwrite=False):
        pass

    def refresh(self):
        pass

    def _save_history(self, data):
        pass


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _LayerCollection()
        self.layers.selection = _FakeSelection()
        self.layers.events = _FakeEvents()
        self.mouse_drag_callbacks = []
        self.dims = SimpleNamespace(
            current_step=(0,),
            events=SimpleNamespace(
                current_step=SimpleNamespace(connect=lambda cb: None)
            ),
        )

    def add_image(self, data, *, name, **kwargs):
        layer = _FakeLayer(data, name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_labels(self, data, *, name, **kwargs):
        layer = _FakeLayer(data, name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_shapes(self, *, name, **kwargs):
        layer = _FakeLayer([], name, **kwargs)
        self.layers[name] = layer
        return layer


def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.cell_workflow_widget", None)
    return importlib.import_module("cellflow.napari.cell_workflow_widget")


def _make_sync_thread_worker():
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
                return SimpleNamespace(quit=lambda: None)
            return wrapper
        return decorator
    return fake_thread_worker


def _write_inputs(pos_dir: Path, *, T=2, Y=24, X=24) -> None:
    """Write the cached divergence maps + nucleus seeds the widget consumes."""
    rng = np.random.default_rng(0)
    (pos_dir / "1_cellpose").mkdir(parents=True, exist_ok=True)
    (pos_dir / "2_nucleus").mkdir(parents=True, exist_ok=True)
    fg = np.clip(rng.normal(0.6, 0.1, (T, Y, X)), 0, 1).astype(np.float32)
    contours = np.abs(rng.normal(0, 1, (T, Y, X))).astype(np.float32)
    nuc = np.zeros((T, Y, X), np.uint32)
    nuc[:, 5:8, 5:8] = 1
    nuc[:, 16:19, 16:19] = 2
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_foreground.tif", fg)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_contours.tif", contours)
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", nuc)


# ── construction / layout ─────────────────────────────────────────────────────

def test_widget_exposes_single_run_and_preview_path(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    for name in ("params_btn", "active_btn", "run_btn"):
        assert isinstance(getattr(widget, name), QToolButton)

    # Single shared status/progress, hidden until used.
    assert isinstance(widget.pipeline_status_lbl, QLabel)
    assert isinstance(widget.pipeline_progress_bar, QProgressBar)
    assert widget.pipeline_progress_bar.isVisible() is False

    # Primary knobs carry the spec defaults.
    assert widget.fg_strength_spin.value() == 0.0
    assert widget.fg_threshold_spin.value() == 0.1
    assert widget.memory_tau_spin.value() == 0.0
    assert widget.balance_spin.value() == 0.98
    assert widget.feature_strength_spin.value() == 100.0

    # Every knob lives in one flat panel — no separate Advanced block.
    assert not hasattr(widget, "advanced_section")

    widget.deleteLater()
    app.processEvents()


def test_params_button_toggles_section(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    assert widget.params_section.is_expanded is False
    widget.params_btn.setChecked(True)
    assert widget.params_section.is_expanded is True
    widget.params_btn.setChecked(False)
    assert widget.params_section.is_expanded is False

    widget.deleteLater()
    app.processEvents()


def test_pipeline_files_list_divergence_inputs_and_output(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    source = Path(mod.__file__).read_text()
    assert "1_cellpose/cell_contours.tif" in source
    assert "1_cellpose/cell_foreground.tif" in source
    assert "2_nucleus/tracked_labels.tif" in source
    assert "3_cell/tracked_labels.tif" in source

    widget.deleteLater()
    app.processEvents()


def test_correction_aliases_point_into_child_widget(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    c = widget.cell_correction_widget
    assert widget.correction_widget is c.correction_widget
    assert widget.load_labels_btn is c.load_labels_btn
    assert widget.expand_cell_btn is c.expand_cell_btn
    assert widget.correction_active_btn is c.active_btn
    assert widget.hole_radius_spin is c.hole_radius_spin

    widget.deleteLater()
    app.processEvents()


# ── state round-trip ──────────────────────────────────────────────────────────

def test_get_set_state_roundtrip(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellWorkflowWidget(_FakeViewer())

    widget.set_state({
        "cleanup": {"fg_strength": 0.3, "fg_threshold": 0.2, "contour_window": 71},
        "temporal": {"memory_tau": 0.05},
        "segmentation": {"balance": 0.5, "feature_strength": 250.0, "n_workers": 1},
        "correction": {"hole_radius": 4, "scope": "All frames"},
    })
    state = widget.get_state()
    assert state["cleanup"]["fg_strength"] == 0.3
    assert state["cleanup"]["fg_threshold"] == 0.2
    assert state["cleanup"]["contour_window"] == 71
    assert state["temporal"]["memory_tau"] == 0.05
    assert state["segmentation"]["balance"] == 0.5
    assert state["segmentation"]["feature_strength"] == 250.0
    assert state["correction"]["hole_radius"] == 4
    assert state["correction"]["scope"] == "All frames"

    widget.deleteLater()
    app.processEvents()


# ── live preview ──────────────────────────────────────────────────────────────

def test_preview_activation_populates_all_intermediate_layers(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    _write_inputs(pos_dir)

    viewer = _FakeViewer()
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    widget.active_btn.setChecked(True)  # triggers _on_activate → _refresh_preview

    for name in mod._PREVIEW_LAYERS:
        assert name in viewer.layers, f"missing preview layer {name}"

    # Preview stops before the expensive geodesic walk: no cell-labels layer,
    # but the diagnostic intermediates (incl. the weighted cost field) are shown.
    # Preview layers are full (T, Y, X) stacks (so the viewer keeps a time slider
    # even with no movie open), painted one frame at a time.
    assert mod._LABELS_LAYER not in viewer.layers
    cost_layer = viewer.layers[mod._COST_LAYER]
    assert cost_layer.data.ndim == 3
    assert cost_layer.data.shape[0] == 2  # T from the input maps
    assert np.isfinite(cost_layer.data[0]).any()  # current frame painted

    # Deactivation tears the preview layers down.
    widget.active_btn.setChecked(False)
    for name in mod._PREVIEW_LAYERS:
        assert name not in viewer.layers

    widget.deleteLater()
    app.processEvents()


def test_preview_reports_missing_maps(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir(parents=True)

    widget = mod.CellWorkflowWidget(_FakeViewer())
    widget.refresh(pos_dir)
    widget.active_btn.setChecked(True)

    assert "Divergence Maps first" in widget.pipeline_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


# ── on-demand single-frame labels ───────────────────────────────────────────

def test_labels_button_enabled_only_during_preview(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    _write_inputs(pos_dir)
    viewer = _FakeViewer()
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)

    assert widget.labels_btn.isEnabled() is False
    widget.active_btn.setChecked(True)
    assert widget.labels_btn.isEnabled() is True
    widget.active_btn.setChecked(False)
    assert widget.labels_btn.isEnabled() is False

    widget.deleteLater()
    app.processEvents()


def test_labels_button_fills_cell_labels_for_current_frame(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    _write_inputs(pos_dir)
    viewer = _FakeViewer()
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)
    widget.set_state({"segmentation": {"n_workers": 1}})

    widget.active_btn.setChecked(True)
    # Activation alone never creates the (slow) labels layer.
    assert mod._LABELS_LAYER not in viewer.layers

    widget.labels_btn.click()  # explicit on-demand geodesic for the current frame
    assert mod._LABELS_LAYER in viewer.layers
    labels = viewer.layers[mod._LABELS_LAYER].data
    assert labels.ndim == 3  # full (T, Y, X) stack, current frame painted
    assert set(np.unique(labels).tolist()) <= {0, 1, 2}
    assert labels[0].max() > 0
    assert "cell labels" in widget.pipeline_status_lbl.text()

    # Deactivation tears the labels layer down alongside the intermediates.
    widget.active_btn.setChecked(False)
    assert mod._LABELS_LAYER not in viewer.layers

    widget.deleteLater()
    app.processEvents()


# ── temporal smoothing in preview (cached stack) ─────────────────────────────

def test_preview_smoothing_caches_and_reuses_stack(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    # Count whole-movie smoothing passes so we can prove the cache is reused.
    calls = {"n": 0}
    real = mod.clean_and_smooth_contours

    def _counting(contours, params):
        calls["n"] += 1
        return real(contours, params)

    monkeypatch.setattr(mod, "clean_and_smooth_contours", _counting)

    pos_dir = tmp_path / "pos00"
    _write_inputs(pos_dir, T=4)
    viewer = _FakeViewer()
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)
    widget.memory_tau_spin.setValue(0.1)

    # The synchronous test thread_worker calls _on_preview_done (which clears
    # _preview_worker) *before* the `self._preview_worker = _worker()`
    # assignment lands, so after each pass _preview_worker holds a stale handle.
    # In real async use it is None once the pass settles; mimic that so a forced
    # refresh recomputes rather than coalescing.
    def _settled_refresh():
        widget._preview_worker = None
        widget._refresh_preview()

    widget.active_btn.setChecked(True)  # tau > 0 → one smoothing pass, cached
    assert calls["n"] == 1
    assert widget._smoothed_stack is not None
    assert widget._smoothed_stack.shape[0] == 4

    # Editing a non-smoothing knob (balance) reuses the cached stack.
    widget.balance_spin.setValue(0.5)
    _settled_refresh()
    assert calls["n"] == 1

    # Editing a contour-cleanup knob invalidates the cache → recompute.
    widget.contour_strength_spin.setValue(0.5)
    _settled_refresh()
    assert calls["n"] == 2

    # Turning smoothing off releases the resident stack.
    widget.memory_tau_spin.setValue(0.0)
    _settled_refresh()
    assert widget._smoothed_stack is None

    widget.deleteLater()
    app.processEvents()


def test_preview_deactivation_drops_smoothed_stack(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    _write_inputs(pos_dir, T=4)
    viewer = _FakeViewer()
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)
    widget.memory_tau_spin.setValue(0.1)

    widget.active_btn.setChecked(True)
    assert widget._smoothed_stack is not None
    widget.active_btn.setChecked(False)
    assert widget._smoothed_stack is None

    widget.deleteLater()
    app.processEvents()


# ── full run ──────────────────────────────────────────────────────────────────

def test_full_run_writes_tracked_labels(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    _write_inputs(pos_dir)

    viewer = _FakeViewer()
    widget = mod.CellWorkflowWidget(viewer)
    widget.refresh(pos_dir)
    widget.set_state({"segmentation": {"n_workers": 1}})

    widget.run_btn.click()

    out = pos_dir / "3_cell" / "tracked_labels.tif"
    assert out.exists()
    labels = tifffile.imread(str(out))
    assert set(np.unique(labels).tolist()) <= {0, 1, 2}
    assert labels.max() > 0
    assert mod._TRACKED_CELL_LAYER in viewer.layers
    assert "complete" in widget.pipeline_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_full_run_refuses_without_divergence_maps(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    nuc = np.zeros((2, 8, 8), np.uint32)
    nuc[:, 2, 2] = 1
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", nuc)

    widget = mod.CellWorkflowWidget(_FakeViewer())
    widget.refresh(pos_dir)
    widget.run_btn.click()

    assert "Divergence Maps first" in widget.pipeline_status_lbl.text()
    assert not (pos_dir / "3_cell" / "tracked_labels.tif").exists()

    widget.deleteLater()
    app.processEvents()
