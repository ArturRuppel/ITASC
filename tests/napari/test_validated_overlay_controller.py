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
    from cellflow.napari.validated_overlay_controller import ValidatedOverlayController

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
    assert layer.opacity == 0.4
    assert "[Correction] Validated: Nucleus" in owned_layers
    assert viewer.layers.index("[Correction] Validated: Nucleus") < viewer.layers.index(
        "[Correction] CellSpotlight"
    )

    viewer.close()
