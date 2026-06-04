"""Unit tests for the single-cell *focus* presentation.

Selecting a cell in correction mode spotlights it: the whole-track comet and
per-frame centroid overlays are hidden, only the focused label is rendered in
the mask layer, and the redundant highlight-border contour is dropped (the
spotlight is the selection indicator). Deselecting restores the overview.

We exercise the pure layer/flag logic by binding the unbound methods to minimal
stand-ins, without building the full Qt widgets.
"""

from __future__ import annotations

import types

import numpy as np

from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari._correction_centroids import (
    centroid_focus_colors,
    _viridis_colors,
)
from cellflow.napari.nucleus_correction_widget import (
    NucleusCorrectionWidget,
    _CORRECTION_CENTROID_LAYER,
    _CORRECTION_TRACK_LAYER,
)


class _FakeLayer:
    def __init__(self, features=None):
        self.visible = True
        self.show_selected_label = False
        self.selected_label = 0
        # points-layer fields (centroid crosses)
        self.features = features
        self.shown = None
        self.border_color = None
        self.face_color = None
        # highlight-layer fields
        self.data = None
        self.shape_type = None


class _FakeLayers(dict):
    """Dict that also exposes a ``selection`` with a settable ``active``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection = types.SimpleNamespace(active=None)


# --------------------------------------------------------------------------- #
# _apply_focus_presentation (nucleus widget)                                  #
# --------------------------------------------------------------------------- #


def _nucleus_stub():
    track = _FakeLayer()
    # three centroid crosses: cell 7 on frames 0 and 1, cell 8 on frame 0
    centroids = _FakeLayer(
        features={"label_id": [7, 8, 7], "frame": [0, 0, 1]}
    )
    labels = _FakeLayer()
    layers = _FakeLayers(
        {
            _CORRECTION_TRACK_LAYER: track,
            _CORRECTION_CENTROID_LAYER: centroids,
        }
    )
    obj = types.SimpleNamespace(
        viewer=types.SimpleNamespace(layers=layers),
        _correction_tracked_layer=lambda: labels,
        _apply_centroid_focus=NucleusCorrectionWidget._apply_centroid_focus,
    )
    return obj, track, centroids, labels


def test_focus_hides_comet_keeps_selected_cross():
    obj, track, centroids, labels = _nucleus_stub()
    NucleusCorrectionWidget._apply_focus_presentation(obj, 7)
    assert track.visible is False
    # centroid layer stays visible; only cell 7's crosses are shown
    assert centroids.visible is True
    np.testing.assert_array_equal(centroids.shown, [True, False, True])
    # cell 7's two crosses (frames 0, 1) get the viridis time gradient
    viridis = _viridis_colors(2)
    np.testing.assert_allclose(centroids.face_color[0], viridis[0])
    np.testing.assert_allclose(centroids.face_color[2], viridis[1])
    np.testing.assert_array_equal(centroids.border_color, centroids.face_color)
    # other cells' labels stay visible: the mask layer is left untouched
    assert labels.show_selected_label is False


def test_deselect_restores_overview():
    obj, track, centroids, labels = _nucleus_stub()
    NucleusCorrectionWidget._apply_focus_presentation(obj, 7)
    NucleusCorrectionWidget._apply_focus_presentation(obj, 0)
    assert track.visible is True
    assert centroids.visible is True
    np.testing.assert_array_equal(centroids.shown, [True, True, True])


def test_missing_overlay_layers_are_tolerated():
    obj = types.SimpleNamespace(
        viewer=types.SimpleNamespace(layers=_FakeLayers()),
    )
    # No track/centroid layers present; should not raise.
    NucleusCorrectionWidget._apply_focus_presentation(obj, 3)


def test_centroid_focus_tolerates_missing_features():
    layer = _FakeLayer(features=None)
    NucleusCorrectionWidget._apply_centroid_focus(layer, 7)
    assert layer.shown is None  # nothing set, no raise


# --------------------------------------------------------------------------- #
# _update_highlight border suppression (shared correction widget)             #
# --------------------------------------------------------------------------- #


def _highlight_stub(*, spotlight: bool):
    seg2d = np.zeros((6, 6), dtype=np.uint8)
    seg2d[1:4, 1:4] = 5  # a block so find_contours yields a contour
    hl = _FakeLayer()
    layer = object()
    spotlight_calls: list = []
    goto = types.SimpleNamespace(
        blockSignals=lambda flag: False,
        setValue=lambda value: None,
    )
    obj = types.SimpleNamespace(
        _selected_label=0,
        _selected_t=-1,
        _goto_cell_id=goto,
        _layer=layer,
        _spotlight=spotlight,
        _get_highlight_layer=lambda: hl,
        _frame_view=lambda lyr, t: seg2d,
        _resolve_spotlight_mask=lambda t, lab, m: m,
        _update_spotlight=lambda m: spotlight_calls.append(m),
        _notify_selection_changed=lambda t, lab, prev: None,
        viewer=types.SimpleNamespace(layers=_FakeLayers()),
    )
    return obj, hl, spotlight_calls


def test_spotlight_suppresses_border():
    obj, hl, spotlight_calls = _highlight_stub(spotlight=True)
    CorrectionWidget._update_highlight(obj, 0, 5)
    assert hl.visible is False
    assert hl.data == []
    assert len(spotlight_calls) == 1  # spotlight still rendered


def test_no_spotlight_draws_border():
    obj, hl, spotlight_calls = _highlight_stub(spotlight=False)
    CorrectionWidget._update_highlight(obj, 0, 5)
    assert hl.visible is True
    assert hl.data and len(hl.data) == 1
    assert spotlight_calls == []


# --------------------------------------------------------------------------- #
# centroid_focus_colors (pure)                                                #
# --------------------------------------------------------------------------- #


def test_focused_cell_gets_viridis_gradient_by_frame():
    # cell 7 occupies frames 2 and 5 (out of order in the input)
    colors = centroid_focus_colors([7, 8, 7], [5, 0, 2], 7)
    viridis = _viridis_colors(2)
    # earliest frame (2, at index 2) -> viridis[0]; latest (5, index 0) -> viridis[1]
    np.testing.assert_allclose(colors[2], viridis[0])
    np.testing.assert_allclose(colors[0], viridis[1])
    # the non-focused cell keeps its per-label color (not a viridis sample)
    assert not np.allclose(colors[1], viridis[0])
    assert not np.allclose(colors[1], viridis[1])


def test_no_focus_gives_all_label_colors():
    focused = centroid_focus_colors([7, 8, 7], [5, 0, 2], 7)
    overview = centroid_focus_colors([7, 8, 7], [5, 0, 2], 0)
    # cell 8's color is unaffected by focus; cell 7's reverts away from viridis
    np.testing.assert_allclose(overview[1], focused[1])
    assert not np.allclose(overview[0], focused[0])
