from __future__ import annotations

from cellflow.napari._correction_layer_lifecycle import (
    capture_layer_view_state,
    hide_all_layers,
    remove_owned_layers,
    restore_layer_view_state,
)


class _Layer:
    def __init__(self, name: str, visible: bool = True) -> None:
        self.name = name
        self.visible = visible


class _Selection:
    def __init__(self) -> None:
        self._items: list[_Layer] = []
        self.active: _Layer | None = None

    def __iter__(self):
        return iter(self._items)

    def clear(self) -> None:
        self._items.clear()
        self.active = None

    def add(self, layer: _Layer) -> None:
        if layer not in self._items:
            self._items.append(layer)


class _Layers:
    def __init__(self, layers: list[_Layer]) -> None:
        self._layers = list(layers)
        self.selection = _Selection()

    def __iter__(self):
        return iter(self._layers)

    def __contains__(self, name: str) -> bool:
        return any(layer.name == name for layer in self._layers)

    def __getitem__(self, name: str) -> _Layer:
        for layer in self._layers:
            if layer.name == name:
                return layer
        raise KeyError(name)

    def remove(self, layer: _Layer) -> None:
        self._layers.remove(layer)

    @property
    def names(self) -> list[str]:
        return [layer.name for layer in self._layers]


def test_capture_hide_and_restore_layer_view_state() -> None:
    image = _Layer("image", visible=True)
    labels = _Layer("labels", visible=False)
    extra = _Layer("extra", visible=True)
    layers = _Layers([image, labels, extra])
    layers.selection.add(image)
    layers.selection.add(extra)
    layers.selection.active = extra

    state = capture_layer_view_state(layers)
    hide_all_layers(layers)

    assert [layer.visible for layer in layers] == [False, False, False]

    restore_layer_view_state(layers, state)

    assert image.visible is True
    assert labels.visible is False
    assert extra.visible is True
    assert list(layers.selection) == [image, extra]
    assert layers.selection.active is extra


def test_restore_layer_view_state_ignores_layers_removed_during_mode() -> None:
    image = _Layer("image", visible=True)
    removed = _Layer("removed", visible=True)
    layers = _Layers([image, removed])
    layers.selection.add(removed)
    layers.selection.active = removed
    state = capture_layer_view_state(layers)
    layers.remove(removed)

    restore_layer_view_state(layers, state)

    assert image.visible is True
    assert list(layers.selection) == []
    assert layers.selection.active is None


def test_remove_owned_layers_removes_only_registered_names_and_clears_set() -> None:
    owned = _Layer("[Correction] owned")
    unowned = _Layer("[Correction] user")
    image = _Layer("image")
    layers = _Layers([owned, unowned, image])
    owned_names = {"[Correction] owned", "[Correction] missing"}

    remove_owned_layers(layers, owned_names)

    assert layers.names == ["[Correction] user", "image"]
    assert owned_names == set()
