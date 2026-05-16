"""Tests for CellCorrectionWidget — cell correction section."""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import napari
from qtpy.QtWidgets import QApplication


def _make_viewer():
    app = QApplication.instance() or QApplication([])
    viewer = napari.Viewer(show=False)
    return app, viewer


def _install_stubs() -> None:
    src_root = Path(__file__).resolve().parents[2] / "src" / "cellflow"
    package_root = src_root / "napari"

    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    sys.modules.setdefault("cellflow.napari", napari_pkg)


def _load_widget_class():
    _install_stubs()
    module = importlib.import_module("cellflow.napari.cell_correction_widget")
    return module.CellCorrectionWidget, module


def _make_widget(viewer, pos_dir: Path | None = None):
    widget_class, module = _load_widget_class()
    widget = widget_class(viewer, pos_dir_provider=lambda: pos_dir)
    return widget, module


# ── Constructor shape ────────────────────────────────────────────────────────


def test_cell_correction_widget_takes_explicit_pos_dir_provider():
    _app, viewer = _make_viewer()
    widget_class, _module = _load_widget_class()
    state = {"pos_dir": Path("/tmp/pos00")}
    widget = widget_class(viewer, pos_dir_provider=lambda: state["pos_dir"])

    assert widget._pos_dir == Path("/tmp/pos00")
    state["pos_dir"] = Path("/tmp/pos01")
    assert widget._pos_dir == Path("/tmp/pos01")

    with pytest.raises(AttributeError):
        _ = widget._missing_workflow_back_ref_probe

    widget.deleteLater()
    viewer.close()


def test_cell_correction_widget_files_widget_refresh_callback_is_called_on_save(tmp_path):
    _app, viewer = _make_viewer()
    widget_class, _module = _load_widget_class()
    pos_dir = tmp_path / "pos00"
    (pos_dir / "3_cell").mkdir(parents=True)
    labels = np.zeros((2, 4, 4), dtype=np.uint32)
    labels[:, 1:3, 1:3] = 7
    viewer.add_labels(labels, name="Tracked: Cell")

    refresh_calls = []
    widget = widget_class(
        viewer,
        pos_dir_provider=lambda: pos_dir,
        files_widget_refresh_callback=lambda pd: refresh_calls.append(pd),
    )
    widget.save_labels_btn.click()

    assert refresh_calls == [pos_dir]

    widget.deleteLater()
    viewer.close()


# ── UI structure ─────────────────────────────────────────────────────────────


def test_cell_correction_widget_exposes_expected_buttons_and_controls():
    from qtpy.QtWidgets import QComboBox, QPushButton, QSpinBox

    _app, viewer = _make_viewer()
    widget, _module = _make_widget(viewer)

    button_texts = {btn.text() for btn in widget.findChildren(QPushButton)}
    assert "Load Labels" in button_texts
    assert "Save Labels" in button_texts
    assert "Fill Holes" in button_texts
    assert "Fix Semi Holes" in button_texts
    assert "Clean Up" in button_texts
    assert "Expand Cell" in button_texts

    assert isinstance(widget.correction_scope_combo, QComboBox)
    assert [widget.correction_scope_combo.itemText(i)
            for i in range(widget.correction_scope_combo.count())] == [
        "Current frame", "All frames",
    ]
    assert isinstance(widget.hole_radius_spin, QSpinBox)
    assert widget.hole_radius_spin.value() == 5
    assert isinstance(widget.semihole_opening_spin, QSpinBox)
    assert widget.semihole_opening_spin.value() == 3
    assert isinstance(widget.expand_max_px_spin, QSpinBox)
    assert widget.expand_max_px_spin.value() == 25
    assert widget.expand_max_px_spin.maximum() == 999

    widget.deleteLater()
    viewer.close()


def test_cell_correction_widget_has_correction_shortcuts_section():
    _app, viewer = _make_viewer()
    widget, _module = _make_widget(viewer)

    assert hasattr(widget, "correction_shortcuts_section")
    assert widget.correction_shortcuts_section.title == "Correction Shortcuts"
    assert widget.correction_shortcuts_section.is_expanded is False

    widget.deleteLater()
    viewer.close()


# ── Load labels ───────────────────────────────────────────────────────────────


def test_load_labels_shows_error_when_file_missing(tmp_path):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget, _module = _make_widget(viewer, pos_dir)

    widget.load_labels_btn.click()

    assert "No cell labels file found" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_load_labels_loads_tracked_cell_layer_and_precomputed_probability_zavgs(
    tmp_path, monkeypatch
):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    (pos_dir / "0_input").mkdir(parents=True)
    (pos_dir / "1_cellpose").mkdir()
    (pos_dir / "3_cell").mkdir()

    labels = np.zeros((2, 4, 4), dtype=np.uint32)
    labels[:, 1:3, 1:3] = 7
    raw_cell_zavg = np.ones((4, 4), dtype=np.float32)
    raw_nuc_zavg = np.full((4, 4), 2.0, dtype=np.float32)
    cell_prob_zavg = np.full((4, 4), 0.25, dtype=np.float32)
    nuc_prob_zavg = np.full((4, 4), 0.75, dtype=np.float32)
    tifffile.imwrite(pos_dir / "3_cell" / "tracked_labels.tif", labels)
    tifffile.imwrite(pos_dir / "0_input" / "cell_zavg.tif", raw_cell_zavg)
    tifffile.imwrite(pos_dir / "0_input" / "nucleus_zavg.tif", raw_nuc_zavg)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_zavg.tif", cell_prob_zavg)
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif", nuc_prob_zavg)

    widget_class, module = _load_widget_class()

    def _sync_thread_worker(connect=None):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    if connect and "errored" in connect:
                        connect["errored"](exc)
                    return None
                if connect and "returned" in connect:
                    connect["returned"](result)
                return None
            return wrapper
        return decorator

    monkeypatch.setattr(module, "_thread_worker", _sync_thread_worker)
    widget = widget_class(viewer, pos_dir_provider=lambda: pos_dir)

    widget.load_labels_btn.click()

    assert "Tracked: Cell" in viewer.layers
    assert "Cell z-avg" in viewer.layers
    assert "Nucleus z-avg" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["Tracked: Cell"].data, labels)
    np.testing.assert_array_equal(
        viewer.layers["Cell z-avg"].data,
        np.broadcast_to(cell_prob_zavg[np.newaxis], labels.shape),
    )
    np.testing.assert_array_equal(
        viewer.layers["Nucleus z-avg"].data,
        np.broadcast_to(nuc_prob_zavg[np.newaxis], labels.shape),
    )
    assert viewer.layers["Cell z-avg"].blending == "minimum"
    assert viewer.layers["Nucleus z-avg"].blending == "minimum"
    assert viewer.layers["Nucleus z-avg"].colormap.name == "I Orange"
    assert widget.correction_widget._layer is viewer.layers["Tracked: Cell"]
    assert "Loaded cell label stack" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()


# ── Save labels ───────────────────────────────────────────────────────────────


def test_save_labels_writes_tracked_cell_labels_to_disk(tmp_path):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    (pos_dir / "3_cell").mkdir(parents=True)
    initial = np.zeros((2, 4, 4), dtype=np.uint32)
    tifffile.imwrite(pos_dir / "3_cell" / "tracked_labels.tif", initial)

    edited = initial.copy()
    edited[0, 1:3, 1:3] = 4
    viewer.add_labels(edited, name="Tracked: Cell")
    widget, _module = _make_widget(viewer, pos_dir)

    widget.save_labels_btn.click()

    saved = tifffile.imread(pos_dir / "3_cell" / "tracked_labels.tif")
    np.testing.assert_array_equal(saved, edited)
    assert saved.dtype == np.uint32
    assert "Saved 2 frame" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_save_labels_shows_error_when_no_layer_present(tmp_path):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget, _module = _make_widget(viewer, pos_dir)

    widget.save_labels_btn.click()

    assert "No labels layer" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()


# ── Fill holes ────────────────────────────────────────────────────────────────


def test_fill_holes_modifies_enclosed_background_pixels(tmp_path):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()

    # 5x5 label with a 1-pixel hole in the middle
    labels = np.zeros((1, 7, 7), dtype=np.uint32)
    labels[0, 1:6, 1:6] = 3
    labels[0, 3, 3] = 0  # hole
    viewer.add_labels(labels.copy(), name="Tracked: Cell")
    widget, _module = _make_widget(viewer, pos_dir)
    widget.correction_widget.activate_layer(viewer.layers["Tracked: Cell"])
    widget.correction_scope_combo.setCurrentText("Current frame")
    widget.hole_radius_spin.setValue(5)

    widget.fill_holes_btn.click()

    result = np.asarray(viewer.layers["Tracked: Cell"].data)
    assert result[0, 3, 3] == 3
    assert "Filled holes" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_fill_holes_reports_no_holes_when_none_found(tmp_path):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()

    labels = np.zeros((1, 4, 4), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 5
    viewer.add_labels(labels.copy(), name="Tracked: Cell")
    widget, _module = _make_widget(viewer, pos_dir)
    widget.correction_widget.activate_layer(viewer.layers["Tracked: Cell"])

    widget.fill_holes_btn.click()

    assert "No interior holes found" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()


# ── Clean Up ──────────────────────────────────────────────────────────────────


def test_cleanup_requires_nuclear_labels(tmp_path):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()

    labels = np.zeros((1, 4, 4), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 2
    viewer.add_labels(labels.copy(), name="Tracked: Cell")
    widget, _module = _make_widget(viewer, pos_dir)

    widget.cleanup_btn.click()

    assert "nuclear labels not found" in widget.correction_status_lbl.text().lower()

    widget.deleteLater()
    viewer.close()


def test_cleanup_uses_nuclear_labels_from_viewer_layer(tmp_path):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()

    cell_labels = np.zeros((1, 5, 5), dtype=np.uint32)
    cell_labels[0, 1:3, 1:3] = 2
    nuc_labels = np.zeros((1, 5, 5), dtype=np.uint32)
    nuc_labels[0, 1:3, 1:3] = 1
    viewer.add_labels(cell_labels.copy(), name="Tracked: Cell")
    viewer.add_labels(nuc_labels.copy(), name="Tracked: Nucleus")
    widget, _module = _make_widget(viewer, pos_dir)

    widget.cleanup_btn.click()

    # The cleanup ran without error (status message set)
    status = widget.correction_status_lbl.text()
    assert "Cleanup" in status

    widget.deleteLater()
    viewer.close()


# ── Expand Cell ───────────────────────────────────────────────────────────────


def test_expand_cell_shows_error_when_no_cell_selected(tmp_path):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()

    labels = np.zeros((1, 5, 5), dtype=np.uint32)
    labels[0, 2, 2] = 5
    viewer.add_labels(labels, name="Tracked: Cell")
    foreground = np.ones((1, 5, 5), dtype=np.uint8)
    viewer.add_labels(foreground, name="Foreground Mask: Cell")
    widget, _module = _make_widget(viewer, pos_dir)
    widget.correction_widget.activate_layer(viewer.layers["Tracked: Cell"])

    widget.expand_cell_btn.click()

    assert "No cell selected" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_expand_cell_shows_error_when_foreground_missing(tmp_path):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()

    labels = np.zeros((1, 5, 5), dtype=np.uint32)
    labels[0, 2, 2] = 5
    viewer.add_labels(labels, name="Tracked: Cell")
    widget, _module = _make_widget(viewer, pos_dir)
    widget.correction_widget.activate_layer(viewer.layers["Tracked: Cell"])
    widget.correction_widget.select_label(0, 5)

    widget.expand_cell_btn.click()

    assert "Foreground mask not found" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_expand_cell_expands_label_into_foreground_pixels(tmp_path):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()

    labels = np.zeros((1, 7, 7), dtype=np.uint32)
    labels[0, 3, 3] = 4
    foreground = np.zeros((1, 7, 7), dtype=np.uint8)
    foreground[0, 2:5, 2:5] = 1
    viewer.add_labels(labels.copy(), name="Tracked: Cell")
    viewer.add_labels(foreground, name="Foreground Mask: Cell")
    widget, _module = _make_widget(viewer, pos_dir)
    widget.correction_widget.activate_layer(viewer.layers["Tracked: Cell"])
    widget.correction_widget.select_label(0, 4)
    widget.expand_max_px_spin.setValue(10)

    widget.expand_cell_btn.click()

    result = np.asarray(viewer.layers["Tracked: Cell"].data)
    assert int(np.sum(result[0] == 4)) > 1
    assert "Expanded cell 4" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()


# ── Seam / delegation test ────────────────────────────────────────────────────


def test_cell_workflow_widget_delegates_correction_to_child_widget(monkeypatch):
    """CellWorkflowWidget.cell_correction_widget owns the correction controls.

    This test imports the real widgets and ensures the seam (alias) wiring is correct.
    monkeypatch is used to ensure any modules cached by this import are cleaned up
    after the test so they don't interfere with other tests that use fake napari modules.
    """
    from qtpy.QtWidgets import QPushButton

    # Register key modules via monkeypatch so they get cleaned up after the test
    _workflow_mod_key = "cellflow.napari.cell_workflow_widget"
    _correction_mod_key = "cellflow.napari.cell_correction_widget"
    if _workflow_mod_key not in sys.modules:
        monkeypatch.delitem(sys.modules, _workflow_mod_key, raising=False)
    if _correction_mod_key not in sys.modules:
        monkeypatch.delitem(sys.modules, _correction_mod_key, raising=False)

    _app, viewer = _make_viewer()

    from cellflow.napari.cell_workflow_widget import CellWorkflowWidget
    from cellflow.napari.cell_correction_widget import CellCorrectionWidget

    # Register loaded modules with monkeypatch so teardown removes them if they
    # were not present before the test
    monkeypatch.setitem(sys.modules, _workflow_mod_key,
                        sys.modules[_workflow_mod_key])
    monkeypatch.setitem(sys.modules, _correction_mod_key,
                        sys.modules[_correction_mod_key])

    widget = CellWorkflowWidget(viewer)

    assert hasattr(widget, "cell_correction_widget")
    assert isinstance(widget.cell_correction_widget, CellCorrectionWidget)

    # Aliases must point into the child
    assert widget.correction_widget is widget.cell_correction_widget.correction_widget
    assert widget.correction_status_lbl is widget.cell_correction_widget.correction_status_lbl
    assert widget.load_labels_btn is widget.cell_correction_widget.load_labels_btn
    assert widget.save_labels_btn is widget.cell_correction_widget.save_labels_btn
    assert widget.fill_holes_btn is widget.cell_correction_widget.fill_holes_btn
    assert widget.fix_semiholes_btn is widget.cell_correction_widget.fix_semiholes_btn
    assert widget.cleanup_btn is widget.cell_correction_widget.cleanup_btn
    assert widget.expand_cell_btn is widget.cell_correction_widget.expand_cell_btn
    assert widget.hole_radius_spin is widget.cell_correction_widget.hole_radius_spin
    assert widget.semihole_opening_spin is widget.cell_correction_widget.semihole_opening_spin
    assert widget.expand_max_px_spin is widget.cell_correction_widget.expand_max_px_spin
    assert widget.correction_scope_combo is widget.cell_correction_widget.correction_scope_combo

    child_buttons = set(widget.cell_correction_widget.findChildren(QPushButton))
    assert widget.load_labels_btn in child_buttons
    assert widget.save_labels_btn in child_buttons
    assert widget.expand_cell_btn in child_buttons

    widget.deleteLater()
    viewer.close()
