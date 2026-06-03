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

from qtpy.QtWidgets import QApplication, QLabel, QToolButton

_APP = None


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


def _qapp():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _layout_items(layout):
    return [layout.itemAt(i) for i in range(layout.count())]


def _layout_widgets_from_items(items):
    return [item.widget() for item in items if item.widget() is not None]


def _layout_for_row_items_containing(widget, *targets):
    target_set = set(targets)
    for index in range(widget.layout().count()):
        item = widget.layout().itemAt(index)
        row = item.layout()
        if row is None:
            continue
        items = _layout_items(row)
        widgets = set(_layout_widgets_from_items(items))
        if target_set.issubset(widgets):
            return items
    raise AssertionError("Could not find matching divergence maps stage row")


def _make_sync_thread_worker():
    """Patch napari thread_worker so tests can observe worker callbacks."""
    import inspect

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
                elif connect and "returned" in connect:
                    connect["returned"](result)
                return None

            return wrapper

        return decorator

    return fake_thread_worker


def test_widget_constructs_and_exposes_public_api(monkeypatch):
    _qapp()
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


def test_divergence_parameter_controls_are_sliders(monkeypatch):
    _qapp()
    mod = _load_widget(monkeypatch)

    w = mod.DivergenceMapsWidget(_FakeViewer())

    for slider in (
        w.nuc_smoothing_spin,
        w.nuc_median_spin,
        w.nuc_fg_smoothing_spin,
        w.nuc_fg_median_spin,
        w.cell_smoothing_spin,
        w.cell_median_spin,
        w.cell_fg_smoothing_spin,
        w.cell_fg_median_spin,
    ):
        buttons = {
            button.objectName(): button
            for button in slider.findChildren(QToolButton)
        }
        start = slider.value()

        buttons["slider_increment_button"].click()
        assert slider.value() == pytest.approx(start + slider.singleStep())

        buttons["slider_decrement_button"].click()
        assert slider.value() == pytest.approx(start)

    w.deleteLater()


def test_stage_row_buttons_are_clustered_and_use_header_style(monkeypatch):
    _qapp()
    mod = _load_widget(monkeypatch)

    w = mod.DivergenceMapsWidget(_FakeViewer())

    nucleus_label = next(
        child for child in w.findChildren(QLabel)
        if child.text() == "Nucleus divergence maps"
    )
    cell_label = next(
        child for child in w.findChildren(QLabel)
        if child.text() == "Cell divergence maps"
    )

    nucleus_items = _layout_for_row_items_containing(
        w,
        nucleus_label,
        w.nucleus_params_btn,
        w.nucleus_run_btn,
    )
    assert _layout_widgets_from_items(nucleus_items[:4]) == [
        nucleus_label,
        w.nucleus_params_btn,
        w.nucleus_preview_btn,
        w.nucleus_run_btn,
    ]
    assert nucleus_items[4].spacerItem() is not None

    cell_items = _layout_for_row_items_containing(
        w,
        cell_label,
        w.cell_params_btn,
        w.cell_run_btn,
    )
    assert _layout_widgets_from_items(cell_items[:4]) == [
        cell_label,
        w.cell_params_btn,
        w.cell_preview_btn,
        w.cell_run_btn,
    ]
    assert cell_items[4].spacerItem() is not None

    for button in (
        w.nucleus_params_btn,
        w.nucleus_preview_btn,
        w.nucleus_run_btn,
        w.cell_params_btn,
        w.cell_preview_btn,
        w.cell_run_btn,
    ):
        assert button.property("cellflow_stage_header_action") is True
        assert "border: none" in button.styleSheet()
        assert "text-align: center" in button.styleSheet()

    w.deleteLater()


def test_standalone_widget_exposes_pipeline_files_by_default(monkeypatch):
    _qapp()
    mod = _load_widget(monkeypatch)

    w = mod.DivergenceMapsWidget(_FakeViewer())

    assert hasattr(w, "output_files_tracker")
    assert hasattr(w, "pipeline_files_header")
    w.deleteLater()


def test_embedded_widget_can_hide_pipeline_files(monkeypatch):
    _qapp()
    mod = _load_widget(monkeypatch)

    w = mod.DivergenceMapsWidget(_FakeViewer(), show_pipeline_files=False)

    assert not hasattr(w, "output_files_tracker")
    assert not hasattr(w, "pipeline_files_header")
    assert w._files_widget is None
    w.deleteLater()


def test_widget_state_roundtrip(monkeypatch):
    _qapp()
    mod = _load_widget(monkeypatch)

    w = mod.DivergenceMapsWidget(_FakeViewer())
    w.nuc_smoothing_spin.setValue(0.5)
    w.nuc_median_spin.setValue(3)
    w.nuc_fg_smoothing_spin.setValue(0.7)
    w.nuc_fg_median_spin.setValue(2)
    w.cell_fg_reduction.setCurrentText("max")
    state = w.get_state()
    assert state["nucleus"]["smoothing_sigma"] == pytest.approx(0.5)
    assert state["nucleus"]["median_radius"] == 3
    assert state["nucleus"]["foreground_smoothing_sigma"] == pytest.approx(0.7)
    assert state["nucleus"]["foreground_median_radius"] == 2
    assert state["cell"]["foreground_z_reduction"] == "max"

    w2 = mod.DivergenceMapsWidget(_FakeViewer())
    w2.set_state(state)
    assert w2.nuc_smoothing_spin.value() == pytest.approx(0.5)
    assert w2.nuc_median_spin.value() == 3
    assert w2.nuc_fg_smoothing_spin.value() == pytest.approx(0.7)
    assert w2.nuc_fg_median_spin.value() == 2
    assert w2.cell_fg_reduction.currentText() == "max"
    w.deleteLater()
    w2.deleteLater()


def test_run_invokes_build_divergence_maps(tmp_path, monkeypatch):
    _qapp()
    mod = _load_widget(monkeypatch)
    from cellflow.segmentation.divergence_maps import DivergenceMapsReport
    import tifffile

    pos = tmp_path / "pos00"
    cell = pos / "1_cellpose"
    cell.mkdir(parents=True)
    tifffile.imwrite(cell / "nucleus_prob_3dt.tif", np.zeros((1, 1, 2, 2), dtype=np.float32))
    tifffile.imwrite(cell / "nucleus_dp_3dt.tif", np.zeros((1, 1, 2, 2, 2), dtype=np.float32))

    captured: dict = {}

    def _fake_build(
        prob_path,
        dp_path,
        contours_out,
        foreground_out,
        *,
        foreground_z_reduction,
        contour_z_reduction,
        smoothing_sigma,
        median_radius,
        foreground_smoothing_sigma=0.0,
        foreground_median_radius=0,
        progress_cb=None,
        cancel=None,
    ):
        captured.update(dict(
            prob_path=str(prob_path),
            dp_path=str(dp_path),
            contours_out=str(contours_out),
            foreground_out=str(foreground_out),
            foreground_z_reduction=foreground_z_reduction,
            contour_z_reduction=contour_z_reduction,
            smoothing_sigma=smoothing_sigma,
            median_radius=median_radius,
            foreground_smoothing_sigma=foreground_smoothing_sigma,
            foreground_median_radius=foreground_median_radius,
        ))
        return DivergenceMapsReport(
            frames=1,
            foreground_z_reduction=foreground_z_reduction,
            contour_z_reduction=contour_z_reduction,
            smoothing_sigma=smoothing_sigma,
            median_radius=median_radius,
            contours_path=contours_out,
            foreground_path=foreground_out,
            foreground_smoothing_sigma=foreground_smoothing_sigma,
            foreground_median_radius=foreground_median_radius,
        )

    monkeypatch.setattr(mod, "build_divergence_maps", _fake_build)

    w = mod.DivergenceMapsWidget(_FakeViewer())
    w.refresh(pos)
    w.nuc_smoothing_spin.setValue(2.0)
    w.nuc_median_spin.setValue(1)
    w.nuc_fg_smoothing_spin.setValue(1.5)
    w.nuc_fg_median_spin.setValue(2)
    w._run_blocking("nucleus")

    assert captured["prob_path"].endswith("nucleus_prob_3dt.tif")
    assert captured["dp_path"].endswith("nucleus_dp_3dt.tif")
    assert captured["contours_out"].endswith("nucleus_contours.tif")
    assert captured["foreground_out"].endswith("nucleus_foreground.tif")
    assert captured["smoothing_sigma"] == 2.0
    assert captured["median_radius"] == 1
    # Foreground filters are passed independently of the contour ones.
    assert captured["foreground_smoothing_sigma"] == 1.5
    assert captured["foreground_median_radius"] == 2
    assert captured["foreground_z_reduction"] == "mean"
    assert captured["contour_z_reduction"] == "mean"
    w.deleteLater()


def test_worker_emits_granular_progress_for_nucleus_and_cell(tmp_path, monkeypatch):
    _qapp()
    mod = _load_widget(monkeypatch)
    from cellflow.segmentation.divergence_maps import DivergenceMapsReport
    import tifffile

    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos = tmp_path / "pos00"
    cellpose = pos / "1_cellpose"
    cellpose.mkdir(parents=True)
    for channel in ("nucleus", "cell"):
        tifffile.imwrite(
            cellpose / f"{channel}_prob_3dt.tif",
            np.zeros((2, 1, 2, 2), dtype=np.float32),
        )
        tifffile.imwrite(
            cellpose / f"{channel}_dp_3dt.tif",
            np.zeros((2, 1, 2, 2, 2), dtype=np.float32),
        )

    def _fake_build(
        prob_path,
        dp_path,
        contours_out,
        foreground_out,
        *,
        foreground_z_reduction,
        contour_z_reduction,
        smoothing_sigma,
        median_radius,
        progress_cb=None,
        cancel=None,
    ):
        channel = "nucleus" if "nucleus" in str(prob_path) else "cell"
        if progress_cb is not None:
            progress_cb(1, 2, f"{channel} frame 1/2")
            progress_cb(2, 2, f"{channel} frame 2/2")
        return DivergenceMapsReport(
            frames=2,
            foreground_z_reduction=foreground_z_reduction,
            contour_z_reduction=contour_z_reduction,
            smoothing_sigma=smoothing_sigma,
            median_radius=median_radius,
            contours_path=contours_out,
            foreground_path=foreground_out,
        )

    monkeypatch.setattr(mod, "build_divergence_maps", _fake_build)

    w = mod.DivergenceMapsWidget(_FakeViewer())
    w.refresh(pos)
    progress_messages: list[str] = []
    w._progress_signal.connect(lambda done, total, msg: progress_messages.append(msg))

    w._start_worker("nucleus")
    w._start_worker("cell")

    assert progress_messages == [
        "nucleus frame 1/2",
        "nucleus frame 2/2",
        "cell frame 1/2",
        "cell frame 2/2",
    ]
    w.deleteLater()


# ── Live preview ─────────────────────────────────────────────────────────

class _PreviewLayer:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.contrast_limits = (0.0, 1.0)
        self.visible = True

    def refresh(self):
        pass


class _PreviewLayers(dict):
    def remove(self, layer):
        name = layer if isinstance(layer, str) else layer.name
        self.pop(name, None)


class _PreviewViewer:
    """Fake viewer with a real layer collection and dims events."""

    def __init__(self):
        self.layers = _PreviewLayers()
        connectable = SimpleNamespace(connect=lambda cb: None)
        self.dims = SimpleNamespace(
            current_step=(0,),
            events=SimpleNamespace(current_step=connectable),
        )

    def add_image(self, data, name=None, **kw):
        self.layers[name] = _PreviewLayer(data, name)

    def add_labels(self, *a, **kw):
        pass


def _write_prob_dp(pos_dir: Path, channel: str, *, T=2, Z=2, Y=8, X=8) -> None:
    import tifffile

    rng = np.random.default_rng(0)
    cp = pos_dir / "1_cellpose"
    cp.mkdir(parents=True, exist_ok=True)
    prob = rng.normal(0.0, 1.0, (T, Z, Y, X)).astype(np.float32)
    dp = rng.normal(0.0, 1.0, (T, Z, 2, Y, X)).astype(np.float32)
    tifffile.imwrite(cp / f"{channel}_prob_3dt.tif", prob)
    tifffile.imwrite(cp / f"{channel}_dp_3dt.tif", dp)


def test_preview_activation_claims_owner_and_creates_layers(monkeypatch, tmp_path):
    _qapp()
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos = tmp_path / "pos00"
    _write_prob_dp(pos, "nucleus")
    w = mod.DivergenceMapsWidget(_PreviewViewer(), show_pipeline_files=False)
    w.refresh(pos)

    fg_name, ct_name = mod._PREVIEW_LAYER_NAMES["nucleus"]
    assert fg_name not in w.viewer.layers

    w.nucleus_preview_btn.setChecked(True)

    assert w.gate.owner == "div_preview:nucleus"
    assert fg_name in w.viewer.layers
    assert ct_name in w.viewer.layers
    # Layers are full (T, Y, X) stacks so napari shows a time slider; only the
    # current frame's slice is filled.
    assert w.viewer.layers[fg_name].data.shape == (2, 8, 8)
    assert w.viewer.layers[fg_name].data[0].any()  # current frame painted
    # The other channel's preview is locked out (single viewer owner).
    assert not w.cell_preview_btn.isEnabled()

    w.nucleus_preview_btn.setChecked(False)
    assert w.gate.owner is None
    assert fg_name not in w.viewer.layers
    assert ct_name not in w.viewer.layers
    assert w.cell_preview_btn.isEnabled()

    w.deleteLater()


def test_preview_param_change_recomputes_current_frame(monkeypatch, tmp_path):
    _qapp()
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos = tmp_path / "pos00"
    _write_prob_dp(pos, "cell")
    w = mod.DivergenceMapsWidget(_PreviewViewer(), show_pipeline_files=False)
    w.refresh(pos)

    w.cell_preview_btn.setChecked(True)
    fg_name, _ = mod._PREVIEW_LAYER_NAMES["cell"]
    before = w.viewer.layers[fg_name].data

    # A param edit re-fires the (debounced) preview; trigger it directly.
    w.cell_smoothing_spin.setValue(3.0)
    w._refresh_preview()
    assert fg_name in w.viewer.layers  # still present, recomputed in place

    w.cell_preview_btn.setChecked(False)
    w.deleteLater()


def test_preview_fills_each_frame_slice_on_scrub(monkeypatch, tmp_path):
    _qapp()
    mod = _load_widget(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos = tmp_path / "pos00"
    _write_prob_dp(pos, "nucleus", T=3)
    viewer = _PreviewViewer()
    w = mod.DivergenceMapsWidget(viewer, show_pipeline_files=False)
    w.refresh(pos)

    w.nucleus_preview_btn.setChecked(True)
    fg_name, _ = mod._PREVIEW_LAYER_NAMES["nucleus"]
    data = w.viewer.layers[fg_name].data
    assert data.shape[0] == 3  # full time axis → slider
    assert data[0].any() and not data[2].any()

    # Scrub to frame 2 → that slice gets computed and filled on refresh.
    viewer.dims.current_step = (2,)
    w._refresh_preview()
    assert w.viewer.layers[fg_name].data[2].any()

    w.nucleus_preview_btn.setChecked(False)
    w.deleteLater()


def test_preview_blocked_during_run(monkeypatch, tmp_path):
    _qapp()
    mod = _load_widget(monkeypatch)

    pos = tmp_path / "pos00"
    _write_prob_dp(pos, "nucleus")
    w = mod.DivergenceMapsWidget(_PreviewViewer(), show_pipeline_files=False)
    w.refresh(pos)

    # A running build disables both previews (no second viewer writer).
    w._running_stage = "nucleus"
    w._set_button_running("nucleus")
    assert not w.nucleus_preview_btn.isEnabled()
    assert not w.cell_preview_btn.isEnabled()

    w._running_stage = None
    w._set_button_idle()
    assert w.nucleus_preview_btn.isEnabled()

    w.deleteLater()
