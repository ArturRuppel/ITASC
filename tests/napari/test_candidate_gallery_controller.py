"""Routing + population coverage for the candidate gallery controller.

Injects fake swap/extend enumerators (the real ones need an ultrack DB) and a
fake viewer, then checks the three columns populate from a selection and a
thumbnail click is routed to the matching apply callback.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402

from cellflow.napari.candidate_gallery_controller import (  # noqa: E402
    CandidateGalleryController,
)


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _FakeWindow:
    def __init__(self):
        self.docked = []

    def add_dock_widget(self, widget, name=None, area=None):
        dock = SimpleNamespace(widget=widget, name=name, area=area)
        self.docked.append(dock)
        return dock

    def remove_dock_widget(self, dock):
        self.docked.remove(dock)


class _FakeViewer:
    def __init__(self):
        self.window = _FakeWindow()


def _square(shape, y0, x0, side):
    m = np.zeros(shape, dtype=bool)
    m[y0 : y0 + side, x0 : x0 + side] = True
    return m


def _make_controller(applied, *, selected=7, t=1):
    shape = (30, 30)
    tracked = np.zeros((3, *shape), dtype=np.uint32)
    tracked[t, 5:10, 5:10] = selected
    intensity = SimpleNamespace(
        data=np.random.default_rng(0).random((3, *shape)).astype(np.float32),
        colormap=None,
    )
    tracked_layer = SimpleNamespace(colormap=SimpleNamespace(color_dict={}))

    def fake_swap(**kw):
        return [
            SimpleNamespace(node_id=201, mask_2d=_square(shape, 5, 5, 4), area=16),
            SimpleNamespace(node_id=202, mask_2d=_square(shape, 5, 5, 6), area=36),
        ]

    def fake_extend(*, source_id, source_frame, direction, tracked_labels, db_path, **kw):
        tf = source_frame + (1 if direction == "forward" else -1)
        if tf < 0 or tf >= tracked_labels.shape[0]:
            return SimpleNamespace(target_frame=tf, assignments=())
        key = 301 if direction == "backward" else 401
        a = SimpleNamespace(
            candidate_label=key,
            mask_2d=_square(shape, 6, 6, 4),
            centroid_corrected_iou=0.8,
            centroid_distance=3.0,
        )
        return SimpleNamespace(target_frame=tf, assignments=(a,))

    controller = CandidateGalleryController(
        _FakeViewer(),
        tracked_data_provider=lambda: tracked,
        tracked_layer_provider=lambda: tracked_layer,
        intensity_layer_provider=lambda: intensity,
        selected_label_provider=lambda: selected,
        current_t_provider=lambda: t,
        db_path_provider=lambda: Path("/nonexistent/data.db"),
        protected_mask_provider=lambda _t: None,
        extend_kwargs_provider=lambda: {},
        apply_swap=lambda cand: applied.__setitem__("swap", int(cand.node_id)),
        apply_extend=lambda which, tf, a: applied.__setitem__(
            which, (int(tf), int(a.candidate_label))
        ),
        list_swap=fake_swap,
        list_extend=fake_extend,
    )
    return controller


def test_refresh_populates_three_columns(_app):
    controller = _make_controller({})
    controller.refresh()
    panel = controller._panel
    assert [t.key for t in panel.column(panel.SWAP).tiles()] == [201, 202]
    assert [t.key for t in panel.column(panel.EXTEND_BACKWARD).tiles()] == [301]
    assert [t.key for t in panel.column(panel.EXTEND_FORWARD).tiles()] == [401]


def test_clicking_swap_tile_applies_that_candidate(_app):
    applied: dict = {}
    controller = _make_controller(applied)
    controller.refresh()
    panel = controller._panel
    panel.column(panel.SWAP).tiles()[1].clicked.emit(202)
    assert applied["swap"] == 202


def test_clicking_extend_tile_applies_direction_and_target_frame(_app):
    applied: dict = {}
    controller = _make_controller(applied)
    controller.refresh()
    panel = controller._panel
    panel.column(panel.EXTEND_FORWARD).tiles()[0].clicked.emit(401)
    assert applied["extend_forward"] == (2, 401)  # target frame 1+1, key 401


def test_extend_backward_out_of_range_is_empty(_app):
    controller = _make_controller({}, t=0)  # backward target = -1
    controller.refresh()
    panel = controller._panel
    assert panel.column(panel.EXTEND_BACKWARD).tiles() == []
    assert [t.key for t in panel.column(panel.EXTEND_FORWARD).tiles()] == [401]


def test_no_selection_clears_without_docking(_app):
    controller = _make_controller({}, selected=0)
    controller.refresh()
    assert controller._panel is None  # nothing docked when nothing is selected
