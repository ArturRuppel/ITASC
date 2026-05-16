from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _LayerCollection(dict):
    def __init__(self) -> None:
        super().__init__()
        self.events = SimpleNamespace(removed=_FakeEvent())

    def remove(self, layer):
        if isinstance(layer, str):
            layer = self[layer]
        self.pop(layer.name, None)
        self.events.removed.emit(layer)


class _FakeEvent:
    def __init__(self) -> None:
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def disconnect(self, callback):
        self.callbacks.remove(callback)

    def emit(self, value=None):
        for callback in list(self.callbacks):
            try:
                callback(SimpleNamespace(value=value))
            except TypeError:
                callback()


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _LayerCollection()
        self.calls = []
        self.current_step_event = _FakeEvent()
        self.dims = SimpleNamespace(
            current_step=(0,),
            events=SimpleNamespace(current_step=self.current_step_event),
        )

    def add_points(self, data, *, name, **kwargs):
        layer = SimpleNamespace(data=np.asarray(data), name=name, **kwargs)
        self.layers[name] = layer
        self.calls.append(("points", name, np.asarray(data), kwargs))
        return layer

    def add_labels(self, data, *, name, **kwargs):
        layer = SimpleNamespace(data=np.asarray(data), name=name, **kwargs)
        self.layers[name] = layer
        self.calls.append(("labels", name, np.asarray(data), kwargs))
        return layer

    def add_image(self, data, *, name, **kwargs):
        layer = SimpleNamespace(data=np.asarray(data), name=name, **kwargs)
        self.layers[name] = layer
        self.calls.append(("image", name, np.asarray(data), kwargs))
        return layer

    def add_shapes(self, data, *, name, shape_type, **kwargs):
        layer = SimpleNamespace(
            data=list(data),
            name=name,
            shape_type=shape_type,
            refresh=lambda: None,
            **kwargs,
        )
        self.layers[name] = layer
        self.calls.append(("shapes", name, list(data), {"shape_type": shape_type, **kwargs}))
        return layer


def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.contact_analysis_visualization", None)
    return importlib.import_module("cellflow.napari.contact_analysis_visualization")


def _make_contact_analysis() -> dict[str, dict[str, np.ndarray]]:
    return {
        "cells": {
            "frame": np.asarray([0, 1], dtype=int),
            "cell_id": np.asarray([11, 12], dtype=int),
            "area": np.asarray([25.5, 32.0], dtype=float),
            "centroid_y": np.asarray([1.5, 3.25], dtype=float),
            "centroid_x": np.asarray([4.5, 5.75], dtype=float),
            "class_label": np.asarray(["A", ""], dtype=object),
        },
        "edges": {
            "frame": np.asarray([0, 1], dtype=int),
            "edge_id": np.asarray([101, 102], dtype=int),
            "cell_a": np.asarray([11, 12], dtype=int),
            "cell_b": np.asarray([13, 0], dtype=int),
            "kind": np.asarray(["cell_cell", "border"], dtype=object),
            "edge_label": np.asarray(["junction", "border"], dtype=object),
            "length": np.asarray([2.5, 3.5], dtype=float),
            "is_t1_frame": np.asarray([False, False], dtype=bool),
            "coord_offset": np.asarray([0, 3], dtype=int),
            "coord_count": np.asarray([3, 3], dtype=int),
        },
        "coord_y": np.asarray([1.0, 1.0, 2.0, 0.0, 0.0, 0.0], dtype=float),
        "coord_x": np.asarray([4.0, 5.0, 5.0, 0.0, 1.0, 2.0], dtype=float),
        "t1_events": {
            "t1_event_id": np.asarray([7], dtype=int),
            "frame": np.asarray([4], dtype=int),
            "edge_id": np.asarray([102], dtype=int),
            "losing_cell_a": np.asarray([11], dtype=int),
            "losing_cell_b": np.asarray([12], dtype=int),
            "gaining_cell_a": np.asarray([11], dtype=int),
            "gaining_cell_b": np.asarray([13], dtype=int),
            "location_y": np.asarray([9.5], dtype=float),
            "location_x": np.asarray([8.5], dtype=float),
        },
    }


def _make_contact_analysis_with_label_paths(tmp_path) -> dict[str, object]:
    contact_analysis = _make_contact_analysis()
    cell_labels = np.zeros((2, 4, 4), dtype=np.uint16)
    cell_labels[0, 1:3, 1:3] = 11
    cell_labels[1, 2:4, 2:4] = 12
    nucleus_labels = np.zeros((2, 4, 4), dtype=np.uint16)
    nucleus_labels[0, 1, 1] = 11
    nucleus_labels[1, 2, 2] = 12

    import tifffile

    cell_path = tmp_path / "cell_labels.tif"
    nucleus_path = tmp_path / "nucleus_labels.tif"
    tifffile.imwrite(cell_path, cell_labels)
    tifffile.imwrite(nucleus_path, nucleus_labels)
    contact_analysis["cell_tracked_labels_path"] = str(cell_path)
    contact_analysis["nucleus_tracked_labels_path"] = str(nucleus_path)
    return contact_analysis


def _make_track_contact_analysis_with_label_paths(tmp_path) -> dict[str, object]:
    contact_analysis = _make_contact_analysis()
    contact_analysis["cells"] = {
        "frame": np.asarray([0, 1, 2, 0, 1, 2], dtype=int),
        "cell_id": np.asarray([11, 11, 11, 12, 12, 12], dtype=int),
        "area": np.asarray([4, 4, 4, 4, 4, 4], dtype=float),
        "centroid_y": np.asarray([1, 2, 3, 4, 5, 6], dtype=float),
        "centroid_x": np.asarray([1, 2, 3, 4, 5, 6], dtype=float),
        "class_label": np.asarray(["A", "A", "A", "B", "B", "B"], dtype=object),
    }
    cell_labels = np.zeros((3, 8, 8), dtype=np.uint16)
    nucleus_labels = np.zeros((3, 8, 8), dtype=np.uint16)
    for frame, offset in enumerate([1, 2, 3]):
        cell_labels[frame, offset:offset + 2, offset:offset + 2] = 11
        nucleus_labels[frame, offset, offset] = 11
        cell_labels[frame, offset + 3:offset + 5, offset + 3:offset + 5] = 12
        nucleus_labels[frame, offset + 3, offset + 3] = 12

    import tifffile

    cell_path = tmp_path / "track_cell_labels.tif"
    nucleus_path = tmp_path / "track_nucleus_labels.tif"
    tifffile.imwrite(cell_path, cell_labels)
    tifffile.imwrite(nucleus_path, nucleus_labels)
    contact_analysis["cell_tracked_labels_path"] = str(cell_path)
    contact_analysis["nucleus_tracked_labels_path"] = str(nucleus_path)
    return contact_analysis


def _add_t1_edge_pair(contact_analysis: dict[str, object]) -> None:
    edges = contact_analysis["edges"]
    assert isinstance(edges, dict)
    edges["frame"] = np.asarray([0, 1, 4, 5], dtype=int)
    edges["edge_id"] = np.asarray([101, 103, 102, 102], dtype=int)
    edges["cell_a"] = np.asarray([11, 12, 11, 11], dtype=int)
    edges["cell_b"] = np.asarray([13, 0, 12, 13], dtype=int)
    edges["kind"] = np.asarray(["cell_cell", "border", "cell_cell", "cell_cell"], dtype=object)
    edges["edge_label"] = np.asarray(["junction", "border", "before_t1", "after_t1"], dtype=object)
    edges["length"] = np.asarray([2.5, 3.5, 4.0, 4.5], dtype=float)
    edges["is_t1_frame"] = np.asarray([False, False, True, False], dtype=bool)
    edges["coord_offset"] = np.asarray([0, 3, 6, 9], dtype=int)
    edges["coord_count"] = np.asarray([3, 3, 3, 3], dtype=int)
    contact_analysis["coord_y"] = np.asarray(
        [1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 9.0, 9.0, 10.0, 8.0, 8.0, 9.0],
        dtype=float,
    )
    contact_analysis["coord_x"] = np.asarray(
        [4.0, 5.0, 5.0, 0.0, 1.0, 2.0, 8.0, 9.0, 9.0, 8.5, 9.5, 9.5],
        dtype=float,
    )


def test_build_cell_centroid_points_returns_frame_prefixed_points_and_features(monkeypatch):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis()

    points, features = mod.build_cell_centroid_points(contact_analysis)

    assert points.shape == (2, 3)
    np.testing.assert_allclose(points[:, 0], [0.0, 1.0])
    np.testing.assert_allclose(points[:, 1:], [[1.5, 4.5], [3.25, 5.75]])
    assert set(features) == {"frame", "cell_id", "area", "class_label"}
    np.testing.assert_array_equal(features["cell_id"], [11, 12])
    np.testing.assert_array_equal(features["class_label"], ["A", ""])


def test_build_edge_shapes_returns_line_payloads_and_colored_border_edges(monkeypatch):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis()

    lines, edge_colors, features = mod.build_edge_shapes(contact_analysis)

    assert len(lines) == 2
    assert lines[0].shape == (3, 3)
    np.testing.assert_allclose(lines[0], [[0.0, 1.0, 4.0], [0.0, 1.0, 5.0], [0.0, 2.0, 5.0]])
    np.testing.assert_allclose(lines[1], [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 2.0]])
    assert edge_colors.shape == (2, 4)
    np.testing.assert_allclose(edge_colors[1], [0.6, 0.6, 0.6, 1.0])
    assert not np.allclose(edge_colors[0], edge_colors[1])
    assert set(features) == {
        "frame",
        "edge_id",
        "cell_a",
        "cell_b",
        "kind",
        "edge_label",
        "length",
        "is_t1_frame",
        "coord_offset",
        "coord_count",
    }
    np.testing.assert_array_equal(features["coord_count"], [3, 3])
    np.testing.assert_array_equal(features["kind"], ["cell_cell", "border"])
    np.testing.assert_array_equal(features["edge_label"], ["junction", "border"])


def test_build_edge_shapes_can_hide_border_edges(monkeypatch):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis()

    lines, edge_colors, features = mod.build_edge_shapes(contact_analysis, hide_border_edges=True)

    assert len(lines) == 1
    assert edge_colors.shape == (1, 4)
    np.testing.assert_array_equal(features["edge_id"], [101])
    np.testing.assert_array_equal(features["kind"], ["cell_cell"])


def test_build_edge_shapes_can_color_by_edge_id(monkeypatch):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis()

    _lines, edge_colors, features = mod.build_edge_shapes(contact_analysis, color_by_id=True)

    assert edge_colors.shape == (2, 4)
    assert not np.allclose(edge_colors[0], edge_colors[1])
    np.testing.assert_array_equal(features["edge_id"], [101, 102])


def test_build_edge_shapes_can_color_by_edge_label(monkeypatch):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis()
    contact_analysis["edges"]["edge_label"] = np.asarray(["same_label", "same_label"], dtype=object)

    _lines, edge_colors, features = mod.build_edge_shapes(contact_analysis, color_by_label=True, color_by_id=True)

    assert edge_colors.shape == (2, 4)
    np.testing.assert_allclose(edge_colors[0], edge_colors[1])
    np.testing.assert_array_equal(features["edge_label"], ["same_label", "same_label"])


def test_add_contact_analysis_layers_can_color_cells_by_class_label(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis_with_label_paths(tmp_path)
    contact_analysis["cells"]["class_label"] = np.asarray(["epithelial", "epithelial"], dtype=object)
    viewer = _FakeViewer()

    mod.add_contact_analysis_layers(viewer, contact_analysis, color_cells_by_label=True)

    cell_call = viewer.calls[0]
    color_map = cell_call[3]["colormap"].color_dict
    np.testing.assert_allclose(color_map[11], color_map[12])
    np.testing.assert_allclose(color_map[None], [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(color_map[0], [0.0, 0.0, 0.0, 0.0])
    nucleus_call = viewer.calls[1]
    nucleus_color_map = nucleus_call[3]["colormap"].color_dict
    np.testing.assert_allclose(nucleus_color_map[11], color_map[11])
    np.testing.assert_allclose(nucleus_color_map[12], color_map[12])


def test_build_nucleus_track_shapes_follows_centroids_and_hides_future(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_track_contact_analysis_with_label_paths(tmp_path)
    nucleus_labels = mod._read_label_image(Path(contact_analysis["nucleus_tracked_labels_path"]))

    lines, colors, features = mod.build_nucleus_track_shapes(
        contact_analysis,
        nucleus_labels,
        current_frame=1,
    )

    assert len(lines) == 2
    np.testing.assert_allclose(lines[0], [[1.0, 1.0, 1.0], [1.0, 2.0, 2.0]])
    np.testing.assert_allclose(lines[1], [[1.0, 4.0, 4.0], [1.0, 5.0, 5.0]])
    assert colors.shape == (2, 4)
    assert np.all(colors[:, :3] > 0.0)
    np.testing.assert_allclose(colors[:, 3], [1.0, 1.0])
    np.testing.assert_array_equal(features["cell_id"], [11, 12])
    np.testing.assert_array_equal(features["start_frame"], [0, 0])
    np.testing.assert_array_equal(features["end_frame"], [1, 1])


def test_build_nucleus_track_shapes_fades_more_distant_past_segments(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_track_contact_analysis_with_label_paths(tmp_path)
    nucleus_labels = mod._read_label_image(Path(contact_analysis["nucleus_tracked_labels_path"]))

    _lines, colors, features = mod.build_nucleus_track_shapes(
        contact_analysis,
        nucleus_labels,
        current_frame=2,
    )

    assert colors.shape == (4, 4)
    for line in _lines:
        np.testing.assert_allclose(line[:, 0], [2.0, 2.0])
    older = colors[features["end_frame"] == 1]
    current = colors[features["end_frame"] == 2]
    assert np.all(older[:, 3] < current[:, 3])
    assert np.all(older[:, 3] >= 0.12)


def test_build_nucleus_track_shapes_uses_cell_class_colors_when_requested(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_track_contact_analysis_with_label_paths(tmp_path)
    contact_analysis["cells"]["class_label"] = np.asarray(["same"] * 6, dtype=object)
    nucleus_labels = mod._read_label_image(Path(contact_analysis["nucleus_tracked_labels_path"]))

    _lines, colors, _features = mod.build_nucleus_track_shapes(
        contact_analysis,
        nucleus_labels,
        current_frame=1,
        color_cells_by_label=True,
    )

    np.testing.assert_allclose(colors[0], colors[1])


def test_build_edge_shapes_filters_edges_with_fewer_than_two_coords(monkeypatch):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis()
    # Inject a degenerate edge with coord_count=1 before the valid edges
    contact_analysis["edges"]["frame"] = np.concatenate([[0], contact_analysis["edges"]["frame"]])
    contact_analysis["edges"]["edge_id"] = np.concatenate([[999], contact_analysis["edges"]["edge_id"]])
    contact_analysis["edges"]["cell_a"] = np.concatenate([[1], contact_analysis["edges"]["cell_a"]])
    contact_analysis["edges"]["cell_b"] = np.concatenate([[2], contact_analysis["edges"]["cell_b"]])
    contact_analysis["edges"]["kind"] = np.concatenate([["cell_cell"], contact_analysis["edges"]["kind"]])
    contact_analysis["edges"]["edge_label"] = np.concatenate([[""], contact_analysis["edges"]["edge_label"]])
    contact_analysis["edges"]["length"] = np.concatenate([[0.5], contact_analysis["edges"]["length"]])
    contact_analysis["edges"]["is_t1_frame"] = np.concatenate([[False], contact_analysis["edges"]["is_t1_frame"]])
    # Prepend a single degenerate coord and shift existing offsets by 1
    contact_analysis["edges"]["coord_offset"] = np.concatenate([[0], contact_analysis["edges"]["coord_offset"] + 1])
    contact_analysis["edges"]["coord_count"] = np.concatenate([[1], contact_analysis["edges"]["coord_count"]])
    contact_analysis["coord_y"] = np.concatenate([[99.0], contact_analysis["coord_y"]])
    contact_analysis["coord_x"] = np.concatenate([[99.0], contact_analysis["coord_x"]])

    lines, edge_colors, features = mod.build_edge_shapes(contact_analysis)

    assert len(lines) == 2  # degenerate edge filtered out
    assert edge_colors.shape == (2, 4)
    np.testing.assert_array_equal(features["edge_id"], [101, 102])


def test_build_t1_edge_shapes_returns_before_and_after_transition_edges(monkeypatch):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis()
    _add_t1_edge_pair(contact_analysis)

    lines, colors, features = mod.build_t1_edge_shapes(contact_analysis)

    assert len(lines) == 2
    np.testing.assert_allclose(lines[0], [[4.0, 9.0, 8.0], [4.0, 9.0, 9.0], [4.0, 10.0, 9.0]])
    np.testing.assert_allclose(lines[1], [[5.0, 8.0, 8.5], [5.0, 8.0, 9.5], [5.0, 9.0, 9.5]])
    assert colors.shape == (2, 4)
    np.testing.assert_allclose(colors, [[0.0, 1.0, 0.9, 1.0], [0.0, 1.0, 0.9, 1.0]])
    assert set(features) == {
        "t1_event_id",
        "frame",
        "transition_frame",
        "transition_side",
        "edge_id",
        "losing_cell_a",
        "losing_cell_b",
        "gaining_cell_a",
        "gaining_cell_b",
        "location_y",
        "location_x",
    }
    np.testing.assert_array_equal(features["t1_event_id"], [7, 7])
    np.testing.assert_array_equal(features["frame"], [4, 5])
    np.testing.assert_array_equal(features["transition_frame"], [4, 4])
    np.testing.assert_array_equal(features["transition_side"], ["before", "after"])
    np.testing.assert_array_equal(features["edge_id"], [102, 102])


def test_add_contact_analysis_layers_uses_fake_viewer_and_styles_t1_edges(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis_with_label_paths(tmp_path)
    viewer = _FakeViewer()

    layers = mod.add_contact_analysis_layers(viewer, contact_analysis, prefix="[Contact Analysis] ")

    assert [layer.name for layer in layers] == [
        "[Contact Analysis] Cell labels",
        "[Contact Analysis] Nucleus labels",
        "[Contact Analysis] Nucleus tracks",
        "[Contact Analysis] Edges",
        "[Contact Analysis] T1 edges",
    ]
    assert [call[0] for call in viewer.calls] == ["labels", "labels", "image", "labels", "labels"]
    cell_call = viewer.calls[0]
    nucleus_call = viewer.calls[1]
    assert cell_call[2].shape == (2, 4, 4)
    assert nucleus_call[2].shape == (2, 4, 4)
    track_call = viewer.calls[2]
    assert track_call[3]["rgb"] is True
    assert track_call[2].shape == (2, 4, 4, 4)
    edge_call = viewer.calls[3]
    assert edge_call[2].shape == (2, 4, 4)
    assert set(np.unique(edge_call[2])) == {0, 102}
    assert viewer.calls[4][2].shape == (2, 4, 4)
    assert "[Contact Analysis] Cell labels" in viewer.layers
    assert "[Contact Analysis] Nucleus labels" in viewer.layers
    assert "[Contact Analysis] Nucleus tracks" in viewer.layers
    assert "[Contact Analysis] Edges" in viewer.layers
    assert "[Contact Analysis] T1 edges" in viewer.layers


def test_add_contact_analysis_layers_adds_rasterized_nucleus_track_layer(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_track_contact_analysis_with_label_paths(tmp_path)
    viewer = _FakeViewer()

    layers = mod.add_contact_analysis_layers(viewer, contact_analysis, prefix="[Contact Analysis] ")

    assert [layer.name for layer in layers] == [
        "[Contact Analysis] Cell labels",
        "[Contact Analysis] Nucleus labels",
        "[Contact Analysis] Nucleus tracks",
        "[Contact Analysis] Edges",
        "[Contact Analysis] T1 edges",
    ]
    track_call = viewer.calls[2]
    assert track_call[0] == "image"
    assert track_call[1] == "[Contact Analysis] Nucleus tracks"
    assert track_call[3]["rgb"] is True
    assert track_call[2].shape == (3, 8, 8, 4)
    assert np.count_nonzero(track_call[2][0]) == 0
    assert np.count_nonzero(track_call[2][1]) > 0
    assert np.count_nonzero(track_call[2][2]) > 0


def test_add_contact_analysis_layers_adds_rasterized_edge_layer(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis_with_label_paths(tmp_path)
    viewer = _FakeViewer()

    layers = mod.add_contact_analysis_layers(viewer, contact_analysis, prefix="[Contact Analysis] ")
    edge_layer = layers[3]

    assert edge_layer.data.shape == (2, 4, 4)
    assert 102 in edge_layer.data[1]
    assert 102 not in edge_layer.data[0]


def test_edge_layer_is_rasterized_once_without_frame_callbacks(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis_with_label_paths(tmp_path)
    viewer = _FakeViewer()
    calls = 0
    original = mod._rasterize_edge_labels

    def _counting_edge_labels(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, "_rasterize_edge_labels", _counting_edge_labels)

    mod.add_contact_analysis_layers(viewer, contact_analysis, prefix="[Contact Analysis] ")
    viewer.dims.current_step = (1, 0, 0)
    viewer.current_step_event.emit()

    assert calls == 1
    assert viewer.current_step_event.callbacks == []


def test_add_contact_analysis_layers_adds_rasterized_t1_edge_layer(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis_with_label_paths(tmp_path)
    _add_t1_edge_pair(contact_analysis)
    cell_labels = np.zeros((6, 12, 12), dtype=np.uint16)
    nucleus_labels = np.zeros((6, 12, 12), dtype=np.uint16)
    import tifffile

    tifffile.imwrite(contact_analysis["cell_tracked_labels_path"], cell_labels)
    tifffile.imwrite(contact_analysis["nucleus_tracked_labels_path"], nucleus_labels)
    viewer = _FakeViewer()

    layers = mod.add_contact_analysis_layers(viewer, contact_analysis, prefix="[Contact Analysis] ")
    t1_layer = layers[4]

    assert t1_layer.data.shape == (6, 12, 12)
    assert 7 in t1_layer.data[4]
    assert 7 in t1_layer.data[5]
    assert 7 not in t1_layer.data[3]


def test_t1_edge_layer_is_rasterized_once_without_frame_callbacks(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis_with_label_paths(tmp_path)
    _add_t1_edge_pair(contact_analysis)
    viewer = _FakeViewer()
    calls = 0
    original = mod._rasterize_t1_edge_labels

    def _counting_t1_labels(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, "_rasterize_t1_edge_labels", _counting_t1_labels)

    mod.add_contact_analysis_layers(viewer, contact_analysis, prefix="[Contact Analysis] ")
    viewer.dims.current_step = (5, 0, 0)
    viewer.current_step_event.emit()

    assert calls == 1
    assert viewer.current_step_event.callbacks == []


def test_add_contact_analysis_layers_omits_empty_rasterized_edge_labels(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis_with_label_paths(tmp_path)
    contact_analysis["edges"]["coord_count"] = np.asarray([1, 1], dtype=int)
    viewer = _FakeViewer()

    mod.add_contact_analysis_layers(viewer, contact_analysis, prefix="[Contact Analysis] ")

    edge_layer = viewer.layers["[Contact Analysis] Edges"]
    assert np.count_nonzero(edge_layer.data) == 0


def test_track_layer_reuses_centroids_for_rasterized_stack(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_track_contact_analysis_with_label_paths(tmp_path)
    viewer = _FakeViewer()
    calls = 0
    original = mod._nucleus_centroids_by_track

    def _counting_centroids(labels):
        nonlocal calls
        calls += 1
        return original(labels)

    monkeypatch.setattr(mod, "_nucleus_centroids_by_track", _counting_centroids)

    mod.add_contact_analysis_layers(viewer, contact_analysis, prefix="[Contact Analysis] ")

    assert calls == 1


def test_track_layer_uses_precomputed_centroids_for_rasterized_stack(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_track_contact_analysis_with_label_paths(tmp_path)
    viewer = _FakeViewer()
    calls = 0
    original = mod._nucleus_centroids_by_track

    def _counting_centroids(labels):
        nonlocal calls
        calls += 1
        return original(labels)

    monkeypatch.setattr(mod, "_nucleus_centroids_by_track", _counting_centroids)
    precomputed = original(np.asarray(__import__("tifffile").imread(contact_analysis["nucleus_tracked_labels_path"])))

    mod.add_contact_analysis_layers(
        viewer,
        contact_analysis,
        prefix="[Contact Analysis] ",
        nucleus_track_centroids=precomputed,
    )

    assert calls == 0


def test_track_layer_disconnects_when_removed(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_track_contact_analysis_with_label_paths(tmp_path)
    viewer = _FakeViewer()
    viewer.dims.current_step = (1, 0, 0)
    layers = mod.add_contact_analysis_layers(viewer, contact_analysis, prefix="[Contact Analysis] ")
    track_layer = layers[2]

    viewer.layers.remove(track_layer)
    viewer.dims.current_step = (2, 0, 0)
    viewer.current_step_event.emit()

    assert viewer.current_step_event.callbacks == []
    assert track_layer.name not in viewer.layers


def test_edge_layer_disconnects_when_removed(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis_with_label_paths(tmp_path)
    viewer = _FakeViewer()
    viewer.dims.current_step = (0, 0, 0)
    layers = mod.add_contact_analysis_layers(viewer, contact_analysis, prefix="[Contact Analysis] ")
    edge_layer = layers[3]

    viewer.layers.remove(edge_layer)
    viewer.dims.current_step = (1, 0, 0)
    viewer.current_step_event.emit()

    assert viewer.current_step_event.callbacks == []
    assert edge_layer.name not in viewer.layers


def test_t1_edge_layer_disconnects_when_removed(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_contact_analysis_with_label_paths(tmp_path)
    _add_t1_edge_pair(contact_analysis)
    viewer = _FakeViewer()
    viewer.dims.current_step = (4, 0, 0)
    layers = mod.add_contact_analysis_layers(viewer, contact_analysis, prefix="[Contact Analysis] ")
    t1_layer = layers[4]

    viewer.layers.remove(t1_layer)
    viewer.dims.current_step = (5, 0, 0)
    viewer.current_step_event.emit()

    assert viewer.current_step_event.callbacks == []
    assert t1_layer.name not in viewer.layers


def test_nucleus_track_layer_contains_full_rasterized_time_stack(monkeypatch, tmp_path):
    mod = _load_module(monkeypatch)
    contact_analysis = _make_track_contact_analysis_with_label_paths(tmp_path)
    viewer = _FakeViewer()

    layers = mod.add_contact_analysis_layers(viewer, contact_analysis, prefix="[Contact Analysis] ")
    track_layer = layers[2]

    assert track_layer.data.shape == (3, 8, 8, 4)
    assert np.count_nonzero(track_layer.data[0]) == 0
    assert np.count_nonzero(track_layer.data[1]) > 0
    assert np.count_nonzero(track_layer.data[2]) > 0
    assert track_layer.data[2, :, :, 3].max() == 255
