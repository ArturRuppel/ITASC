from __future__ import annotations

import numpy as np

import cellflow.napari._correction_centroids as centroid_module
from cellflow.napari._correction_centroids import (
    build_centroid_points,
    correction_label_color_map,
    refresh_centroid_cross_layer,
    update_centroid_cross_layer_for_edit,
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


def test_build_centroid_points_avoids_per_label_full_frame_scans(monkeypatch) -> None:
    labels = np.zeros((2, 12, 12), dtype=np.uint32)
    for label_id in range(1, 7):
        labels[0, label_id : label_id + 1, 1:3] = label_id
        labels[1, label_id : label_id + 1, 5:7] = label_id
    color_map = correction_label_color_map(labels)
    nonzero_calls = 0
    original_nonzero = np.nonzero

    def counting_nonzero(*args, **kwargs):
        nonlocal nonzero_calls
        nonzero_calls += 1
        return original_nonzero(*args, **kwargs)

    monkeypatch.setattr(centroid_module.np, "nonzero", counting_nonzero)

    payload = build_centroid_points(labels, color_map)

    assert payload.data.shape == (12, 3)
    assert nonzero_calls <= labels.shape[0]


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


def test_update_centroid_cross_layer_for_edit_replaces_only_changed_frame_ids() -> None:
    viewer = _Viewer()
    owned = set()
    labels = np.zeros((2, 10, 10), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 3
    labels[0, 7:9, 7:9] = 4
    labels[1, 2:4, 6:8] = 3
    color_map = correction_label_color_map(labels)
    layer = refresh_centroid_cross_layer(
        viewer,
        labels,
        color_map=color_map,
        name="[Correction] Nucleus Centroids",
        owned_layer_names=owned,
    )

    labels[0, 4:6, 4:6] = 3
    labels[0, 1:3, 1:3] = 0
    updated = update_centroid_cross_layer_for_edit(
        viewer,
        labels,
        color_map=color_map,
        name="[Correction] Nucleus Centroids",
        owned_layer_names=owned,
        frame=0,
        changed_ids={3},
    )

    assert updated is layer
    assert updated.data.shape == (3, 3)
    frame_label_pairs = list(
        zip(updated.features["frame"], updated.features["label_id"], strict=True)
    )
    assert frame_label_pairs.count((0, 3)) == 1
    assert (0, 4) in frame_label_pairs
    assert (1, 3) in frame_label_pairs
    row = frame_label_pairs.index((0, 3))
    np.testing.assert_allclose(updated.data[row], [0, 4.5, 4.5])
