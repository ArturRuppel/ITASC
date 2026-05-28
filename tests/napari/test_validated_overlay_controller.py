from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import napari
from qtpy.QtWidgets import QApplication


def _make_viewer():
    app = QApplication.instance() or QApplication([])
    viewer = napari.Viewer(show=False)
    return app, viewer


def test_controller_adds_validated_overlay_below_spotlight():
    from cellflow.napari.validated_overlay_controller import (
        VALIDATED_OVERLAY_OPACITY,
        ValidatedOverlayController,
    )

    _app, viewer = _make_viewer()
    owned_layers: set[str] = set()
    tracked = viewer.add_labels(
        np.array([[[0, 1], [1, 0]]], dtype=np.uint8),
        name="[Correction] Tracked: Nucleus",
    )
    viewer.add_image(
        np.zeros((2, 2, 4), dtype=np.float32),
        name="[Correction] CellSpotlight",
        rgb=True,
    )
    controller = ValidatedOverlayController(
        viewer,
        tracked_layer_provider=lambda: tracked,
        pos_dir_provider=lambda: None,
        current_t_provider=lambda: 0,
        owned_layers=owned_layers,
    )

    controller.add_overlay(np.array([[[0, 1], [0, 0]]], dtype=np.uint8))

    layer = viewer.layers["[Correction] Validated: Nucleus"]
    color = layer.get_color(1)

    assert layer.opacity == VALIDATED_OVERLAY_OPACITY
    assert np.allclose(color[:3], [0.0, 1.0, 0.0], atol=1e-6)
    assert color[3] == 1.0
    assert "[Correction] Validated: Nucleus" in owned_layers
    assert viewer.layers.index("[Correction] Validated: Nucleus") < viewer.layers.index(
        "[Correction] CellSpotlight"
    )

    viewer.close()


def test_controller_refreshes_anchor_overlay_from_corrections(tmp_path):
    from cellflow.database.validation import add_correction
    from cellflow.napari.validated_overlay_controller import (
        VALIDATED_OVERLAY_OPACITY,
        ValidatedOverlayController,
    )
    from cellflow.tracking_ultrack.corrections import Correction

    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    tracked_data = np.zeros((2, 5, 5), dtype=np.uint8)
    tracked_data[1, 2:4, 1:4] = 7
    tracked = viewer.add_labels(tracked_data, name="[Correction] Tracked: Nucleus")
    viewer.add_image(
        np.zeros((5, 5, 4), dtype=np.float32),
        name="[Correction] CellSpotlight",
        rgb=True,
    )
    add_correction(pos_dir, Correction(cell_id=7, t=1, kind="anchor", y=2.5, x=2.0))
    owned_layers: set[str] = set()
    controller = ValidatedOverlayController(
        viewer,
        tracked_layer_provider=lambda: tracked,
        pos_dir_provider=lambda: pos_dir,
        current_t_provider=lambda: 1,
        owned_layers=owned_layers,
    )

    controller.refresh_anchor_overlay(lambda data, t: data[t])

    layer = viewer.layers["[Correction] Anchors: Nucleus"]
    color = layer.get_color(1)
    expected = np.zeros_like(tracked_data, dtype=np.uint8)
    expected[1, 2:4, 1:4] = 1

    np.testing.assert_array_equal(layer.data, expected)
    assert layer.opacity == VALIDATED_OVERLAY_OPACITY
    assert np.allclose(color[:3], [179 / 255, 148 / 255, 0.0], atol=1e-6)
    assert color[3] == 1.0
    assert "[Correction] Anchors: Nucleus" in owned_layers
    assert viewer.layers.index("[Correction] Anchors: Nucleus") < viewer.layers.index(
        "[Correction] CellSpotlight"
    )

    viewer.close()
