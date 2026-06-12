from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import tifffile

from cellflow.napari.aggregate_quantification.plot_panel import LoadTarget
from cellflow.napari.aggregate_quantification.plugins._click_to_load import ClickToLoad


class _FakeLayer:
    def __init__(self, data, name):
        self.data, self.name = data, name
        self.selected_label = None
        self.show_selected_label = False
        self.visible = True


class _FakeDims:
    def __init__(self):
        self.steps = []
        self.current_step = (0,)

    def set_current_step(self, axis, value):
        self.steps.append((axis, value))
        self.current_step = (value,)


class _FakeCamera:
    def __init__(self):
        self.center = None


class _FakeViewer:
    def __init__(self):
        self.layers = []
        self.dims = _FakeDims()
        self.camera = _FakeCamera()

    def add_labels(self, data, name=None):
        layer = _FakeLayer(data, name)
        self.layers.append(layer)
        return layer

    def add_image(self, data, name=None, rgb=False, blending=None):
        layer = _FakeLayer(data, name)
        layer.rgb, layer.blending = rgb, blending
        self.layers.append(layer)
        return layer


def test_resolver_maps_identity_to_input_path():
    rec = {"id": "p1", "cell_tracked_labels_path": "/data/p1/cells.tif"}
    ctl = ClickToLoad(_FakeViewer())
    resolve = ctl.resolver([rec], "cell_tracked_labels_path")
    target = resolve({"position_id": "p1", "frame": 4, "cell_id": 7})
    assert target.path == Path("/data/p1/cells.tif")
    assert target.frame == 4 and target.cell_id == 7


def test_resolver_uses_frame_start_when_no_frame():
    rec = {"id": "p1", "cell_tracked_labels_path": "/data/p1/cells.tif"}
    resolve = ClickToLoad(_FakeViewer()).resolver([rec], "cell_tracked_labels_path")
    target = resolve({"position_id": "p1", "frame_start": 9, "cell_id": 3})
    assert target.frame == 9


def test_resolver_none_when_position_missing_or_no_labels():
    ctl = ClickToLoad(_FakeViewer())
    assert ctl.resolver([], "cell_tracked_labels_path")({"position_id": "x", "cell_id": 1}) is None
    rec = {"id": "p1", "cell_tracked_labels_path": None}
    assert ctl.resolver([rec], "cell_tracked_labels_path")({"position_id": "p1", "cell_id": 1}) is None


def test_load_shows_full_labels_with_spotlight_and_centers(tmp_path):
    # 3-frame stack; cell 7 is a block in frame 2, with other cells around it.
    stack = np.zeros((3, 10, 10), dtype=np.uint16)
    stack[2, 4:6, 6:8] = 7
    stack[2, 0:2, 0:2] = 3        # a neighbour that must stay visible
    path = tmp_path / "cells.tif"
    tifffile.imwrite(path, stack)
    viewer = _FakeViewer()
    ctl = ClickToLoad(viewer)
    target = LoadTarget(path=path, kind="labels", frame=2, cell_id=7,
                        identity={"position_id": "p1", "frame": 2, "cell_id": 7})
    ctl.load(target)
    # Full labels layer plus the spotlight overlay (the rest of the segmentation
    # is kept, not hidden via show_selected_label).
    labels_layer, spotlight_layer = viewer.layers
    assert labels_layer.show_selected_label is False
    assert (labels_layer.data == stack).all()
    assert viewer.dims.steps[-1] == (0, 2)
    assert viewer.camera.center[-2:] == (4.5, 6.5)   # centroid of the 4:6 x 6:8 block
    # Spotlight: RGBA overlay over the picked frame, cell 7 left bright (alpha 0)
    # while its surroundings are dimmed, and a yellow border drawn around it.
    assert spotlight_layer.rgb is True
    assert spotlight_layer.visible is True
    overlay = spotlight_layer.data
    assert overlay.shape == (10, 10, 4)
    assert overlay[4, 6, 3] == 0.0                   # inside cell 7: fully bright
    assert overlay[0, 0, 3] > 0.0                    # outside: dimmed
    assert (overlay[3, 6] == (1.0, 1.0, 0.0, 1.0)).all()  # yellow border ring

    ctl.load(target)                                 # second load replaces both layers
    assert len(viewer.layers) == 2
