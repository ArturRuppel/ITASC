from __future__ import annotations

import numpy as np

from cellflow.napari._correction_centroids import (
    build_centroid_points,
    correction_label_color_map,
    refresh_centroid_cross_layer,
)


class _PointsLayer:
    def __init__(self, data, name: str, **kwargs) -> None:
        self.data = np.asarray(data)
        self.name = name
        self.kwargs = kwargs
        self.border_color = kwargs.get("border_color")
        self.face_color = kwargs.get("face_color")
        self.features = kwargs.get("features")
        self.symbol = kwargs.get("symbol")
        self.size = kwargs.get("size")


class _Selection:
    def __init__(self) -> None:
        self.active = None


class _Layers:
    def __init__(self) -> None:
        self._layers = []
        self.selection = _Selection()

    def __contains__(self, name: str) -> bool:
        return any(layer.name == name for layer in self._layers)

    def __getitem__(self, name: str):
        for layer in self._layers:
            if layer.name == name:
                return layer
        raise KeyError(name)

    def append(self, layer) -> None:
        self._layers.append(layer)


class _Viewer:
    def __init__(self) -> None:
        self.layers = _Layers()

    def add_points(self, data, **kwargs):
        layer = _PointsLayer(data, kwargs["name"], **{k: v for k, v in kwargs.items() if k != "name"})
        self.layers.append(layer)
        return layer


def test_build_centroid_points_returns_cross_positions_and_matching_colors() -> None:
    labels = np.zeros((2, 6, 7), dtype=np.uint32)
    labels[0, 1:3, 2:4] = 4
    labels[1, 3:6, 1:4] = 9
    color_map = correction_label_color_map(labels)

    payload = build_centroid_points(labels, color_map)

    np.testing.assert_allclose(payload.data, [[0, 1.5, 2.5], [1, 4.0, 2.0]])
    assert payload.features["label_id"] == [4, 9]
    np.testing.assert_allclose(payload.border_color[0], color_map[4])
    np.testing.assert_allclose(payload.border_color[1], color_map[9])


def test_correction_label_color_map_gives_high_new_labels_non_black_colors() -> None:
    labels = np.array([[[0, 1, 4096]]], dtype=np.uint32)

    color_map = correction_label_color_map(labels)

    assert 4096 in color_map
    assert not np.allclose(np.asarray(color_map[4096])[:3], [0.0, 0.0, 0.0])
    assert color_map[None] == "transparent"
    assert color_map[0] == "transparent"


def test_refresh_centroid_cross_layer_adds_and_updates_owned_points_layer() -> None:
    viewer = _Viewer()
    owned = set()
    labels = np.zeros((1, 5, 5), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 3
    color_map = correction_label_color_map(labels)

    layer = refresh_centroid_cross_layer(
        viewer,
        labels,
        color_map=color_map,
        name="[Correction] Nucleus Centroids",
        owned_layer_names=owned,
    )

    assert layer is viewer.layers["[Correction] Nucleus Centroids"]
    assert layer.symbol == "cross"
    assert layer.size == 7
    assert owned == {"[Correction] Nucleus Centroids"}

    labels[0, 3:5, 3:5] = 8
    refreshed_color_map = correction_label_color_map(labels)
    refreshed = refresh_centroid_cross_layer(
        viewer,
        labels,
        color_map=refreshed_color_map,
        name="[Correction] Nucleus Centroids",
        owned_layer_names=owned,
    )

    assert refreshed is layer
    assert refreshed.data.shape == (2, 3)
    assert refreshed.features["label_id"] == [3, 8]
