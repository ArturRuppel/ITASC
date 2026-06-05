"""The validated/anchor overlay renders as a coloured border in filled-by-ID view.

In the filled (colour-by-ID) viewer mode the reference images are hidden and the
labels are drawn filled; a translucent green/anchor wash would then obscure the
per-ID colours. ``set_border_mode`` switches the overlays to an opaque contour
instead, and the choice must survive overlay rebuilds.
"""

from __future__ import annotations

from types import SimpleNamespace

from cellflow.napari.validated_overlay_controller import (
    ANCHOR_OVERLAY,
    SPOTLIGHT_LAYER,
    VALIDATED_OVERLAY,
    VALIDATED_OVERLAY_CONTOUR,
    VALIDATED_OVERLAY_OPACITY,
    ValidatedOverlayController,
)


def _layer():
    return SimpleNamespace(contour=0, opacity=VALIDATED_OVERLAY_OPACITY)


class _FakeLayers(dict):
    def index(self, name):  # pragma: no cover - unused in these tests
        return list(self).index(name)


def _controller(layers=None):
    viewer = SimpleNamespace(layers=_FakeLayers(layers or {}))
    return ValidatedOverlayController(
        viewer,
        tracked_layer_provider=lambda: None,
        pos_dir_provider=lambda: None,
        owned_layers=set(),
    )


def test_border_mode_draws_existing_overlays_as_opaque_contour():
    val, anc = _layer(), _layer()
    c = _controller({VALIDATED_OVERLAY: val, ANCHOR_OVERLAY: anc})

    c.set_border_mode(True)

    for layer in (val, anc):
        assert layer.contour == VALIDATED_OVERLAY_CONTOUR
        assert layer.opacity == 1.0


def test_border_mode_off_restores_translucent_wash():
    val = _layer()
    c = _controller({VALIDATED_OVERLAY: val})

    c.set_border_mode(True)
    c.set_border_mode(False)

    assert val.contour == 0
    assert val.opacity == VALIDATED_OVERLAY_OPACITY


def test_rebuilt_overlay_keeps_border_style():
    # No overlay layer exists yet: set border mode, then let the controller
    # create one via add_overlay and confirm it is styled as a border.
    created = {}

    def add_labels(data, *, name, opacity, colormap):
        layer = SimpleNamespace(
            data=data, name=name, opacity=opacity, colormap=colormap, contour=0
        )
        created[name] = layer
        viewer.layers[name] = layer
        return layer

    viewer = SimpleNamespace(layers=_FakeLayers(), add_labels=add_labels)
    c = ValidatedOverlayController(
        viewer,
        tracked_layer_provider=lambda: None,
        pos_dir_provider=lambda: None,
        owned_layers=set(),
    )

    c.set_border_mode(True)
    c.add_overlay([[0, 1], [1, 0]])

    layer = created[VALIDATED_OVERLAY]
    assert layer.contour == VALIDATED_OVERLAY_CONTOUR
    assert layer.opacity == 1.0
    assert SPOTLIGHT_LAYER not in viewer.layers  # place_below_spotlight is a no-op
