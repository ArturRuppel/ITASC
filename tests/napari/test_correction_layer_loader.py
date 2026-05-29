from __future__ import annotations

import numpy as np

from cellflow.napari._correction_layer_loader import (
    add_correction_image_layer,
    add_tracked_labels_and_track_layer,
    remove_other_correction_label_layers,
)


class _ImageLayer:
    def __init__(self, data, name: str, **kwargs) -> None:
        self.data = data
        self.name = name
        self.kwargs = kwargs


class _LabelsLayer:
    def __init__(self, data, name: str, **kwargs) -> None:
        self.data = data
        self.name = name
        self.kwargs = kwargs
        self.blending = kwargs.get("blending")
        self.colormap = None


class _Layers:
    def __init__(self) -> None:
        self._layers = []

    def __iter__(self):
        return iter(self._layers)

    def __contains__(self, name: str) -> bool:
        return any(layer.name == name for layer in self._layers)

    def __getitem__(self, name: str):
        for layer in self._layers:
            if layer.name == name:
                return layer
        raise KeyError(name)

    def append(self, layer) -> None:
        self._layers.append(layer)

    def remove(self, layer) -> None:
        self._layers.remove(layer)

    @property
    def names(self) -> list[str]:
        return [layer.name for layer in self._layers]


class _Viewer:
    def __init__(self) -> None:
        self.layers = _Layers()

    def add_image(self, data, **kwargs):
        layer = _ImageLayer(data, kwargs["name"], **{k: v for k, v in kwargs.items() if k != "name"})
        self.layers.append(layer)
        return layer

    def add_labels(self, data, **kwargs):
        layer = _LabelsLayer(data, kwargs["name"], **{k: v for k, v in kwargs.items() if k != "name"})
        self.layers.append(layer)
        return layer


def test_add_correction_image_layer_normalizes_data_and_records_ownership() -> None:
    viewer = _Viewer()
    owned = set()

    layer = add_correction_image_layer(
        viewer,
        np.array([[1, 2]], dtype=np.uint16),
        name="[Correction] image",
        colormap="gray",
        owned_layer_names=owned,
    )

    assert layer.name == "[Correction] image"
    assert layer.data.dtype == np.float32
    assert layer.kwargs["colormap"] == "gray"
    assert layer.kwargs["blending"] == "minimum"
    assert owned == {"[Correction] image"}


def test_add_tracked_labels_and_track_layer_adds_labels_track_and_ownership() -> None:
    viewer = _Viewer()
    owned = set()
    labels = np.zeros((2, 4, 4), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 7
    labels[1, 2:4, 2:4] = 7

    result = add_tracked_labels_and_track_layer(
        viewer,
        labels,
        labels_layer_name="[Correction] Nucleus Labels",
        track_layer_name="[Correction] Nucleus tracks",
        owned_layer_names=owned,
    )

    assert result.labels_layer is viewer.layers["[Correction] Nucleus Labels"]
    assert result.labels_layer.blending == "additive"
    assert viewer.layers["[Correction] Nucleus tracks"].kwargs["rgb"] is True
    assert viewer.layers["[Correction] Nucleus tracks"].kwargs["opacity"] == 0.9
    assert viewer.layers["[Correction] Nucleus tracks"].kwargs["blending"] == "additive"
    assert viewer.layers["[Correction] Nucleus tracks"].data.shape == (2, 4, 4, 4)
    assert np.count_nonzero(viewer.layers["[Correction] Nucleus tracks"].data[..., 3]) > 0
    assert result.color_map[None] == "transparent"
    assert result.color_map[0] == "transparent"
    assert 7 in result.color_map
    assert owned == {
        "[Correction] Nucleus Labels",
        "[Correction] Nucleus tracks",
    }


def test_remove_other_correction_label_layers_keeps_owned_and_non_label_layers() -> None:
    viewer = _Viewer()
    owned = _LabelsLayer(np.zeros((1, 1)), "[Correction] owned")
    stale = _LabelsLayer(np.zeros((1, 1)), "[Correction] stale")
    image = _ImageLayer(np.zeros((1, 1)), "[Correction] image")
    normal = _LabelsLayer(np.zeros((1, 1)), "normal")
    for layer in (owned, stale, image, normal):
        viewer.layers.append(layer)

    remove_other_correction_label_layers(
        viewer,
        owned_layer_names={"[Correction] owned"},
        label_layer_type=_LabelsLayer,
    )

    assert viewer.layers.names == [
        "[Correction] owned",
        "[Correction] image",
        "normal",
    ]
