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

from qtpy.QtWidgets import QApplication, QLabel

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
    assert _layout_widgets_from_items(nucleus_items[:3]) == [
        nucleus_label,
        w.nucleus_params_btn,
        w.nucleus_run_btn,
    ]
    assert nucleus_items[3].spacerItem() is not None

    cell_items = _layout_for_row_items_containing(
        w,
        cell_label,
        w.cell_params_btn,
        w.cell_run_btn,
    )
    assert _layout_widgets_from_items(cell_items[:3]) == [
        cell_label,
        w.cell_params_btn,
        w.cell_run_btn,
    ]
    assert cell_items[3].spacerItem() is not None

    for button in (
        w.nucleus_params_btn,
        w.nucleus_run_btn,
        w.cell_params_btn,
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
        ))
        return DivergenceMapsReport(
            frames=1,
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
    w.nuc_smoothing_spin.setValue(2.0)
    w.nuc_median_spin.setValue(1)
    w._run_blocking("nucleus")

    assert captured["prob_path"].endswith("nucleus_prob_3dt.tif")
    assert captured["dp_path"].endswith("nucleus_dp_3dt.tif")
    assert captured["contours_out"].endswith("nucleus_contours.tif")
    assert captured["foreground_out"].endswith("nucleus_foreground.tif")
    assert captured["smoothing_sigma"] == 2.0
    assert captured["median_radius"] == 1
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
