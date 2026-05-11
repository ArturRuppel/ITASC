from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QComboBox, QLabel, QProgressBar, QSpinBox


class _LayerCollection(dict):
    def remove(self, layer):
        self.pop(layer.name, None)


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _LayerCollection()
        self.dims = types.SimpleNamespace(current_step=(0,))

    def add_image(self, data, *, name, **kwargs):
        layer = types.SimpleNamespace(data=np.asarray(data), name=name, kwargs=kwargs)
        self.layers[name] = layer
        return layer

    def add_labels(self, data, *, name, **kwargs):
        layer = types.SimpleNamespace(data=np.asarray(data), name=name, kwargs=kwargs)
        self.layers[name] = layer
        return layer


def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.cell_boundary_workflow_widget", None)
    return importlib.import_module("cellflow.napari.cell_boundary_workflow_widget")


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
                if hasattr(result, "__next__"):
                    return_value = None
                    try:
                        while True:
                            try:
                                yielded = next(result)
                            except StopIteration as exc:
                                return_value = exc.value
                                break
                            if connect and "yielded" in connect:
                                connect["yielded"](yielded)
                    except Exception as exc:
                        if connect and "errored" in connect:
                            connect["errored"](exc)
                        return None
                    if connect and "returned" in connect:
                        connect["returned"](return_value)
                elif connect and "returned" in connect:
                    connect["returned"](result)
                return None
            return wrapper
        return decorator
    return fake_thread_worker


def _label_texts(widget):
    return [child.text() for child in widget.findChildren(QLabel)]


def test_cell_boundary_widget_exposes_stage_files(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellBoundaryWorkflowWidget(_FakeViewer())

    assert widget.contour_section.title == "1. Contour Maps"
    assert widget.boundary_selection_section.title == "2. Track-Conditioned Boundary Selection"
    assert widget.correction_section.title == "3. Correction"

    contour_input = " ".join(label.text() for label in widget.contour_input_files.findChildren(QLabel))
    contour_output = " ".join(label.text() for label in widget.contour_output_files.findChildren(QLabel))
    selection_input = " ".join(label.text() for label in widget.boundary_selection_input_files.findChildren(QLabel))
    selection_output = " ".join(label.text() for label in widget.boundary_selection_output_files.findChildren(QLabel))

    assert "1_cellpose/cell_prob_3dt.tif" in contour_input
    assert "3_cell/filtered_dp.tif" in contour_input
    assert "3_cell/contour_maps.tif" in contour_output
    assert "3_cell/foreground_scores.tif" in contour_output
    assert "3_cell/foreground_masks.tif" in contour_output
    assert "2_nucleus/tracked_labels.tif" in selection_input
    assert "3_cell/contour_maps.tif" in selection_input
    assert "3_cell/foreground_scores.tif" in selection_input
    assert "3_cell/foreground_masks.tif" in selection_input
    assert "1_cellpose/cell_dp_3dt.tif" in selection_input
    assert "3_cell/tracked_labels.tif" in selection_output

    for bar in widget.findChildren(QProgressBar):
        assert bar.isVisible() is False

    assert widget.cp_min_spin.value() == -3.0
    assert widget.cp_max_spin.value() == 0.0
    assert widget.cp_step_spin.value() == 1.0
    assert widget.cp_gamma_min_spin.value() == 1.0
    assert widget.cp_gamma_max_spin.value() == 1.0
    assert widget.cp_gamma_step_spin.value() == 0.25
    assert widget.contour_fg_threshold_spin.value() == 0.5
    assert isinstance(widget.graphcut_solver_combo, QComboBox)
    assert widget.graphcut_solver_combo.currentText() == "graphcut"
    assert widget.graphcut_unary_mode_combo.currentText() == "flow"
    assert widget.graphcut_boundary_mode_combo.currentText() == "contour"
    assert isinstance(widget.graphcut_n_iters_spin, QSpinBox)
    assert widget.graphcut_n_iters_spin.value() == 1
    assert widget.graphcut_n_workers_spin.value() == 1
    assert widget.graphcut_alpha_unary_spin.value() == 4.0
    assert widget.graphcut_lambda_geodesic_spin.value() == 1.0
    assert widget.graphcut_lambda_flow_spin.value() == 1.0
    assert widget.graphcut_lambda_s_spin.value() == 1.0
    assert widget.graphcut_beta_s_spin.value() == 5.0
    assert widget.graphcut_lambda_contour_spin.value() == 0.0
    assert widget.graphcut_lambda_t_spin.value() == 1.0
    assert widget.graphcut_init_mode_combo.currentText() == "nuclei"
    assert widget.graphcut_min_round_flips_spin.value() == 0
    assert "1_cellpose/cell_dp_3dt.tif" in widget.graphcut_unary_mode_combo.toolTip()

    widget.deleteLater()
    app.processEvents()


def test_cell_boundary_widget_persists_parameter_state(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellBoundaryWorkflowWidget(_FakeViewer())

    widget.cp_min_spin.setValue(-4.0)
    widget.cp_max_spin.setValue(-1.0)
    widget.cp_gamma_min_spin.setValue(0.75)
    widget.contour_fg_threshold_spin.setValue(0.65)
    widget.graphcut_solver_combo.setCurrentText("icm")
    widget.graphcut_unary_mode_combo.setCurrentText("geodesic_flow")
    widget.graphcut_boundary_mode_combo.setCurrentText("foreground_inverse")
    widget.graphcut_n_iters_spin.setValue(3)
    widget.graphcut_n_workers_spin.setValue(4)
    widget.graphcut_alpha_unary_spin.setValue(2.5)
    widget.graphcut_lambda_geodesic_spin.setValue(0.6)
    widget.graphcut_lambda_flow_spin.setValue(1.7)
    widget.graphcut_lambda_s_spin.setValue(0.8)
    widget.graphcut_beta_s_spin.setValue(6.5)
    widget.graphcut_lambda_contour_spin.setValue(0.2)
    widget.graphcut_lambda_t_spin.setValue(1.3)
    widget.graphcut_init_mode_combo.setCurrentText("geodesic")
    widget.graphcut_min_round_flips_spin.setValue(5)

    state = widget.get_state()

    restored = mod.CellBoundaryWorkflowWidget(_FakeViewer())
    restored.set_state(state)
    assert restored.cp_min_spin.value() == -4.0
    assert restored.cp_max_spin.value() == -1.0
    assert restored.cp_gamma_min_spin.value() == 0.75
    assert restored.contour_fg_threshold_spin.value() == 0.65
    assert restored.graphcut_solver_combo.currentText() == "icm"
    assert restored.graphcut_unary_mode_combo.currentText() == "geodesic_flow"
    assert restored.graphcut_boundary_mode_combo.currentText() == "foreground_inverse"
    assert restored.graphcut_n_iters_spin.value() == 3
    assert restored.graphcut_n_workers_spin.value() == 4
    assert restored.graphcut_alpha_unary_spin.value() == 2.5
    assert restored.graphcut_lambda_geodesic_spin.value() == 0.6
    assert restored.graphcut_lambda_flow_spin.value() == 1.7
    assert restored.graphcut_lambda_s_spin.value() == 0.8
    assert restored.graphcut_beta_s_spin.value() == 6.5
    assert restored.graphcut_lambda_contour_spin.value() == 0.2
    assert restored.graphcut_lambda_t_spin.value() == 1.3
    assert restored.graphcut_init_mode_combo.currentText() == "geodesic"
    assert restored.graphcut_min_round_flips_spin.value() == 5

    widget.deleteLater()
    restored.deleteLater()
    app.processEvents()


def test_cell_boundary_widget_builds_contour_maps_and_saves_scores(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    prob = np.zeros((2, 1, 4, 4), dtype=np.float32)
    filtered_dp = np.zeros((2, 2, 4, 4), dtype=np.float32)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "3_cell" / "filtered_dp.tif", filtered_dp)

    call_params: list[tuple[tuple[float, ...], float]] = []

    def fake_boundary_2d(prob_yx, dp_cyx, thresholds, flow_threshold=0.0, reduction="mean", niter=200):
        call_params.append((tuple(float(v) for v in thresholds), float(flow_threshold), int(niter)))
        return (
            np.full((4, 4), 1.0, dtype=np.float32),
            np.full((4, 4), 0.75, dtype=np.float32),
        )

    monkeypatch.setattr(mod, "build_consensus_boundary_2d", fake_boundary_2d)

    viewer = _FakeViewer()
    widget = mod.CellBoundaryWorkflowWidget(viewer)
    widget.refresh(pos_dir)
    widget.cp_min_spin.setValue(-2.0)
    widget.cp_max_spin.setValue(0.0)
    widget.cp_step_spin.setValue(1.0)
    widget.cp_gamma_min_spin.setValue(0.5)
    widget.cp_gamma_max_spin.setValue(1.0)
    widget.cp_gamma_step_spin.setValue(0.5)
    widget.contour_flow_threshold_spin.setValue(0.2)
    widget.contour_fg_threshold_spin.setValue(0.8)
    widget._on_build_contour_maps()

    # 2 gammas × 2 timepoints = 4 calls, each with the same thresholds and flow_threshold
    assert len(call_params) == 4
    for params in call_params:
        assert params == ((-2.0, -1.0, 0.0), 0.2, 200)
    contours = tifffile.imread(pos_dir / "3_cell" / "contour_maps.tif")
    scores = tifffile.imread(pos_dir / "3_cell" / "foreground_scores.tif")
    masks = tifffile.imread(pos_dir / "3_cell" / "foreground_masks.tif")
    assert contours.shape == (2, 4, 4)
    assert scores.shape == (2, 4, 4)
    assert masks.shape == (2, 4, 4)
    # score=0.75 < threshold=0.8 → all masked out
    np.testing.assert_array_equal(masks, np.zeros((2, 4, 4), dtype=np.uint8))
    assert "Contour Map: Cell" in viewer.layers
    assert "Foreground Score: Cell" in viewer.layers
    assert "Foreground Mask: Cell" in viewer.layers
    assert "Contour maps complete." in widget.contour_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_cell_boundary_widget_preview_builds_single_frame(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    prob = np.zeros((3, 1, 4, 4), dtype=np.float32)
    filtered_dp = np.zeros((3, 2, 4, 4), dtype=np.float32)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_3dt.tif", prob)
    tifffile.imwrite(pos_dir / "3_cell" / "filtered_dp.tif", filtered_dp)

    call_count = {"n": 0}

    def fake_boundary_2d(prob_yx, dp_cyx, thresholds, flow_threshold=0.0, reduction="mean", niter=200):
        call_count["n"] += 1
        return (
            np.full((4, 4), 0.5, dtype=np.float32),
            np.full((4, 4), 0.9, dtype=np.float32),
        )

    monkeypatch.setattr(mod, "build_consensus_boundary_2d", fake_boundary_2d)

    viewer = _FakeViewer()
    viewer.dims = types.SimpleNamespace(current_step=(1,))
    widget = mod.CellBoundaryWorkflowWidget(viewer)
    widget.refresh(pos_dir)
    widget.contour_fg_threshold_spin.setValue(0.5)
    widget._on_preview_contour_maps()

    # one gamma value → one call for the single preview frame
    assert call_count["n"] == 1
    assert "Contour Map: Cell" in viewer.layers
    assert "Foreground Score: Cell" in viewer.layers
    assert "Foreground Mask: Cell" in viewer.layers
    # contour/score data for non-preview frames should be zero
    contour_data = viewer.layers["Contour Map: Cell"].data
    assert contour_data.shape[0] == 3
    assert contour_data[0].sum() == 0.0
    assert contour_data[1].sum() > 0.0  # frame t=1
    assert "t=1" in widget.contour_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_cell_boundary_widget_boundary_selection_runs_graphcut_and_loads_result(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    graphcut_script = Path(__file__).resolve().parents[2] / "scripts" / "experiment_cell_2d_t_multilabel_graphcut.py"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "1_cellpose").mkdir()
    (pos_dir / "3_cell").mkdir()
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", np.ones((1, 4, 4), dtype=np.uint32))
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_dp_3dt.tif", np.zeros((1, 1, 2, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "3_cell" / "contour_maps.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_scores.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif", np.ones((1, 4, 4), dtype=np.uint8))

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        timestamp = cmd[cmd.index("--timestamp") + 1]
        output_dir = pos_dir / "4_cell_graphcut" / timestamp
        output_dir.mkdir(parents=True)
        tifffile.imwrite(output_dir / "cell_labels.tif", np.full((1, 4, 4), 7, dtype=np.uint16))
        return types.SimpleNamespace(returncode=0, stdout="frame 1\nSaving cell_labels.tif\n", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    viewer = _FakeViewer()
    widget = mod.CellBoundaryWorkflowWidget(viewer)
    widget.refresh(pos_dir)
    widget.graphcut_boundary_mode_combo.setCurrentText("foreground_inverse")
    widget._on_run_boundary_selection()

    assert len(calls) == 1
    cmd = calls[0][0]
    assert cmd[:2] == [sys.executable, str(graphcut_script)]
    assert cmd[cmd.index("--pos-dir") + 1] == str(pos_dir)
    assert cmd[cmd.index("--solver") + 1] == "graphcut"
    assert cmd[cmd.index("--unary-mode") + 1] == "flow"
    assert cmd[cmd.index("--flow-field-path") + 1] == str(pos_dir / "1_cellpose" / "cell_dp_3dt.tif")
    assert cmd[cmd.index("--boundary-mode") + 1] == "foreground_inverse"
    assert cmd[cmd.index("--foreground-score-path") + 1] == str(pos_dir / "3_cell" / "foreground_scores.tif")
    assert cmd[cmd.index("--n-iters") + 1] == "1"
    assert cmd[cmd.index("--n-workers") + 1] == "1"
    assert "--overwrite" in cmd
    np.testing.assert_array_equal(
        tifffile.imread(pos_dir / "3_cell" / "tracked_labels.tif"),
        np.full((1, 4, 4), 7, dtype=np.uint16),
    )
    assert "cell_labels_graphcut" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["cell_labels_graphcut"].data, np.full((1, 4, 4), 7, dtype=np.uint16))
    assert "complete" in widget.boundary_selection_status_lbl.text().lower()

    widget.deleteLater()
    app.processEvents()


def test_cell_boundary_widget_disables_graphcut_only_params_for_icm(monkeypatch):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    widget = mod.CellBoundaryWorkflowWidget(_FakeViewer())

    icm_unused_widgets = [
        widget.graphcut_unary_mode_combo,
        widget.graphcut_boundary_mode_combo,
        widget.graphcut_n_workers_spin,
        widget.graphcut_lambda_geodesic_spin,
        widget.graphcut_lambda_flow_spin,
        widget.graphcut_lambda_contour_spin,
        widget.graphcut_init_mode_combo,
    ]

    assert all(w.isEnabled() for w in icm_unused_widgets)
    widget.graphcut_solver_combo.setCurrentText("icm")
    assert all(not w.isEnabled() for w in icm_unused_widgets)
    assert all(not w.isEnabled() for w in widget._icm_disabled_widgets)

    widget.graphcut_solver_combo.setCurrentText("graphcut")
    assert all(w.isEnabled() for w in icm_unused_widgets)
    assert all(w.isEnabled() for w in widget._icm_disabled_widgets)

    widget.deleteLater()
    app.processEvents()


def test_cell_boundary_widget_boundary_selection_runs_icm_with_widget_params(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "3_cell").mkdir()
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", np.ones((1, 4, 4), dtype=np.uint32))
    tifffile.imwrite(pos_dir / "3_cell" / "contour_maps.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_scores.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif", np.ones((1, 4, 4), dtype=np.uint8))

    calls = []
    expected_labels = np.full((1, 4, 4), 9, dtype=np.uint32)

    def fake_run_cell_icm_from_pos_dir(call_pos_dir, params):
        calls.append((call_pos_dir, params))
        return expected_labels

    icm_mod = importlib.import_module("cellflow.segmentation.cell_label_icm")
    monkeypatch.setattr(icm_mod, "run_cell_icm_from_pos_dir", fake_run_cell_icm_from_pos_dir)

    viewer = _FakeViewer()
    widget = mod.CellBoundaryWorkflowWidget(viewer)
    widget.refresh(pos_dir)
    widget.graphcut_solver_combo.setCurrentText("icm")
    widget.graphcut_alpha_unary_spin.setValue(2.5)
    widget.graphcut_lambda_s_spin.setValue(0.8)
    widget.graphcut_beta_s_spin.setValue(6.5)
    widget.graphcut_lambda_t_spin.setValue(1.3)
    widget.graphcut_n_iters_spin.setValue(7)
    widget.graphcut_min_round_flips_spin.setValue(5)

    widget._on_run_boundary_selection()

    assert len(calls) == 1
    call_pos_dir, params = calls[0]
    assert call_pos_dir == pos_dir
    assert params == icm_mod.CellLabelICMParams(
        alpha_unary=2.5,
        lambda_s=0.8,
        beta_s=6.5,
        lambda_t=1.3,
        n_iters=7,
        min_round_flips=5,
    )
    np.testing.assert_array_equal(
        tifffile.imread(pos_dir / "3_cell" / "tracked_labels.tif"),
        expected_labels,
    )
    assert "cell_labels_graphcut" in viewer.layers
    np.testing.assert_array_equal(viewer.layers["cell_labels_graphcut"].data, expected_labels)

    widget.deleteLater()
    app.processEvents()


def test_cell_boundary_widget_boundary_selection_reports_missing_inputs(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "3_cell").mkdir(parents=True)

    viewer = _FakeViewer()
    widget = mod.CellBoundaryWorkflowWidget(viewer)
    widget.refresh(pos_dir)
    widget._on_run_boundary_selection()

    assert "Missing" in widget.boundary_selection_status_lbl.text()

    widget.deleteLater()
    app.processEvents()


def test_cell_boundary_widget_boundary_selection_failure_does_not_overwrite(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "thread_worker", _make_sync_thread_worker())

    pos_dir = tmp_path / "pos00"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "1_cellpose").mkdir()
    (pos_dir / "3_cell").mkdir()
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", np.ones((1, 4, 4), dtype=np.uint32))
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_dp_3dt.tif", np.zeros((1, 1, 2, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "3_cell" / "contour_maps.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_scores.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "3_cell" / "foreground_masks.tif", np.ones((1, 4, 4), dtype=np.uint8))
    existing = np.full((1, 4, 4), 3, dtype=np.uint16)
    tifffile.imwrite(pos_dir / "3_cell" / "tracked_labels.tif", existing)

    def fake_run(cmd, **kwargs):
        return types.SimpleNamespace(returncode=2, stdout="starting\n", stderr="graphcut failed")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    viewer = _FakeViewer()
    widget = mod.CellBoundaryWorkflowWidget(viewer)
    widget.refresh(pos_dir)
    widget._on_run_boundary_selection()

    np.testing.assert_array_equal(tifffile.imread(pos_dir / "3_cell" / "tracked_labels.tif"), existing)
    assert "graphcut failed" in widget.boundary_selection_status_lbl.text()
    assert "cell_labels_graphcut" not in viewer.layers

    widget.deleteLater()
    app.processEvents()
