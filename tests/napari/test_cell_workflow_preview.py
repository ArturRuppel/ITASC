from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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
            current_step=(0, 1),
            events=SimpleNamespace(
                current_step=SimpleNamespace(connect=lambda callback: None)
            ),
        )

    def add_labels(self, data, *, name):
        layer = SimpleNamespace(data=np.asarray(data), name=name)
        self.layers[name] = layer
        return layer

    def add_image(self, data, *, name, colormap="gray", **kwargs):
        layer = SimpleNamespace(
            data=np.asarray(data),
            name=name,
            colormap=colormap,
            **kwargs,
        )
        self.layers[name] = layer
        return layer


def _load_cell_widget_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)

    correction_module = types.ModuleType("cellflow.napari.correction_widget")

    class _StubCorrectionWidget(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def deactivate(self):
            pass

    correction_module.CorrectionWidget = _StubCorrectionWidget
    monkeypatch.setitem(
        sys.modules, "cellflow.napari.correction_widget", correction_module
    )
    sys.modules.pop("cellflow.napari.cell_workflow_widget", None)
    module = importlib.import_module("cellflow.napari.cell_workflow_widget")
    monkeypatch.setitem(sys.modules, "cellflow.napari.cell_workflow_widget", module)
    return module


def test_cell_preview_runs_seeded_watershed_for_all_z_slices(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    module = _load_cell_widget_module(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()

    prob = np.zeros((1, 3, 4, 5), dtype=np.float32)
    nucleus = np.zeros((1, 3, 4, 5), dtype=np.uint32)
    for z in range(3):
        nucleus[0, z, 1:3, 2:4] = z + 1

    tifffile.imwrite(
        pos_dir / "1_cellpose" / "cell_prob_3dt.tif",
        prob,
        photometric="minisblack",
    )
    tifffile.imwrite(
        pos_dir / "2_nucleus" / "tracked_labels.tif",
        nucleus,
        photometric="minisblack",
    )

    calls = []

    def fake_seeded_watershed(prob_2d, dp_2d, seeds_2d, params):
        calls.append((prob_2d.copy(), dp_2d, seeds_2d.copy()))
        return np.full(prob_2d.shape, int(seeds_2d.max()), dtype=np.uint32)

    monkeypatch.setattr(module, "compute_seeded_watershed", fake_seeded_watershed)

    widget = module.CellWorkflowWidget(_FakeViewer())
    widget.refresh(pos_dir)

    widget._on_preview()

    preview = widget.viewer.layers[module._PREVIEW_LAYER].data
    assert preview.shape == (3, 4, 5)
    assert [int(slice_.max()) for slice_ in preview] == [1, 2, 3]
    assert len(calls) == 3

    widget.deleteLater()
    app.processEvents()


def test_cell_preview_flow_magnitude_loads_channel_first_dp(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    module = _load_cell_widget_module(monkeypatch)

    pos_dir = tmp_path / "pos00"
    (pos_dir / "1_cellpose").mkdir(parents=True)
    (pos_dir / "2_nucleus").mkdir()

    prob = np.zeros((1, 3, 4, 5), dtype=np.float32)
    nucleus = np.zeros((1, 3, 4, 5), dtype=np.uint32)
    nucleus[:, :, 1:3, 2:4] = 1

    # External Cellpose output may store vector channels before Z: (T, C, Z, Y, X).
    dp = np.zeros((1, 2, 3, 4, 5), dtype=np.float32)
    for z in range(3):
        dp[0, 0, z] = z + 3
        dp[0, 1, z] = 4

    tifffile.imwrite(
        pos_dir / "1_cellpose" / "cell_prob_3dt.tif",
        prob,
        photometric="minisblack",
    )
    tifffile.imwrite(
        pos_dir / "1_cellpose" / "cell_dp_3dt.tif",
        dp,
        photometric="minisblack",
    )
    tifffile.imwrite(
        pos_dir / "2_nucleus" / "tracked_labels.tif",
        nucleus,
        photometric="minisblack",
    )

    seen_dp_shapes = []

    def fake_seeded_watershed(prob_2d, dp_2d, seeds_2d, params):
        seen_dp_shapes.append(dp_2d.shape)
        return np.full(prob_2d.shape, len(seen_dp_shapes), dtype=np.uint32)

    monkeypatch.setattr(module, "compute_seeded_watershed", fake_seeded_watershed)

    widget = module.CellWorkflowWidget(_FakeViewer())
    widget.refresh(pos_dir)
    widget.basin_combo.setCurrentText("Flow Magnitude")

    widget._on_preview()

    preview = widget.viewer.layers[module._PREVIEW_LAYER].data
    basin = widget.viewer.layers[module._PREVIEW_BASIN_LAYER].data
    assert preview.shape == (3, 4, 5)
    assert basin.shape == (3, 4, 5)
    assert seen_dp_shapes == [(2, 4, 5), (2, 4, 5), (2, 4, 5)]
    np.testing.assert_allclose(basin[:, 0, 0], [5.0, np.sqrt(32.0), np.sqrt(41.0)])

    widget.deleteLater()
    app.processEvents()
