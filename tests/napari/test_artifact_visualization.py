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
    def remove(self, layer):
        self.pop(layer.name, None)


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _LayerCollection()
        self.calls = []

    def add_points(self, data, *, name, **kwargs):
        layer = SimpleNamespace(data=np.asarray(data), name=name, **kwargs)
        self.layers[name] = layer
        self.calls.append(("points", name, np.asarray(data), kwargs))
        return layer

    def add_shapes(self, data, *, name, shape_type, **kwargs):
        layer = SimpleNamespace(data=list(data), name=name, shape_type=shape_type, **kwargs)
        self.layers[name] = layer
        self.calls.append(("shapes", name, list(data), {"shape_type": shape_type, **kwargs}))
        return layer


def _load_module(monkeypatch):
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "cellflow.napari", napari_pkg)
    sys.modules.pop("cellflow.napari.artifact_visualization", None)
    return importlib.import_module("cellflow.napari.artifact_visualization")


def _make_artifact() -> dict[str, dict[str, np.ndarray]]:
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


def test_build_cell_centroid_points_returns_frame_prefixed_points_and_features(monkeypatch):
    mod = _load_module(monkeypatch)
    artifact = _make_artifact()

    points, features = mod.build_cell_centroid_points(artifact)

    assert points.shape == (2, 3)
    np.testing.assert_allclose(points[:, 0], [0.0, 1.0])
    np.testing.assert_allclose(points[:, 1:], [[1.5, 4.5], [3.25, 5.75]])
    assert set(features) == {"frame", "cell_id", "area", "class_label"}
    np.testing.assert_array_equal(features["cell_id"], [11, 12])
    np.testing.assert_array_equal(features["class_label"], ["A", ""])


def test_build_edge_shapes_returns_line_payloads_and_colored_border_edges(monkeypatch):
    mod = _load_module(monkeypatch)
    artifact = _make_artifact()

    lines, edge_colors, features = mod.build_edge_shapes(artifact)

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
        "length",
        "is_t1_frame",
        "coord_offset",
        "coord_count",
    }
    np.testing.assert_array_equal(features["coord_count"], [3, 3])
    np.testing.assert_array_equal(features["kind"], ["cell_cell", "border"])


def test_build_edge_shapes_filters_edges_with_fewer_than_two_coords(monkeypatch):
    mod = _load_module(monkeypatch)
    artifact = _make_artifact()
    # Inject a degenerate edge with coord_count=1 before the valid edges
    artifact["edges"]["frame"] = np.concatenate([[0], artifact["edges"]["frame"]])
    artifact["edges"]["edge_id"] = np.concatenate([[999], artifact["edges"]["edge_id"]])
    artifact["edges"]["cell_a"] = np.concatenate([[1], artifact["edges"]["cell_a"]])
    artifact["edges"]["cell_b"] = np.concatenate([[2], artifact["edges"]["cell_b"]])
    artifact["edges"]["kind"] = np.concatenate([["cell_cell"], artifact["edges"]["kind"]])
    artifact["edges"]["length"] = np.concatenate([[0.5], artifact["edges"]["length"]])
    artifact["edges"]["is_t1_frame"] = np.concatenate([[False], artifact["edges"]["is_t1_frame"]])
    # Prepend a single degenerate coord and shift existing offsets by 1
    artifact["edges"]["coord_offset"] = np.concatenate([[0], artifact["edges"]["coord_offset"] + 1])
    artifact["edges"]["coord_count"] = np.concatenate([[1], artifact["edges"]["coord_count"]])
    artifact["coord_y"] = np.concatenate([[99.0], artifact["coord_y"]])
    artifact["coord_x"] = np.concatenate([[99.0], artifact["coord_x"]])

    lines, edge_colors, features = mod.build_edge_shapes(artifact)

    assert len(lines) == 2  # degenerate edge filtered out
    assert edge_colors.shape == (2, 4)
    np.testing.assert_array_equal(features["edge_id"], [101, 102])


def test_build_t1_points_returns_frame_prefixed_points_and_features(monkeypatch):
    mod = _load_module(monkeypatch)
    artifact = _make_artifact()

    points, features = mod.build_t1_points(artifact)

    assert points.shape == (1, 3)
    np.testing.assert_allclose(points[0], [4.0, 9.5, 8.5])
    assert set(features) == {
        "t1_event_id",
        "frame",
        "edge_id",
        "losing_cell_a",
        "losing_cell_b",
        "gaining_cell_a",
        "gaining_cell_b",
        "location_y",
        "location_x",
    }
    np.testing.assert_array_equal(features["t1_event_id"], [7])
    np.testing.assert_array_equal(features["edge_id"], [102])


def test_add_artifact_layers_uses_fake_viewer_and_styles_t1_points(monkeypatch):
    mod = _load_module(monkeypatch)
    artifact = _make_artifact()
    viewer = _FakeViewer()

    layers = mod.add_artifact_layers(viewer, artifact, prefix="[Artifact] ")

    assert [layer.name for layer in layers] == [
        "[Artifact] Cell centroids",
        "[Artifact] Edges",
        "[Artifact] T1 events",
    ]
    assert [call[0] for call in viewer.calls] == ["points", "shapes", "points"]
    t1_call = viewer.calls[2]
    assert t1_call[3]["symbol"] == "star"
    assert t1_call[3]["face_color"] == "red"
    assert t1_call[3]["border_color"] == "red"
    edge_call = viewer.calls[1]
    assert edge_call[3]["shape_type"] == "path"
    assert edge_call[3]["face_color"] == "transparent"
    np.testing.assert_allclose(np.asarray(edge_call[3]["edge_color"])[1], [0.6, 0.6, 0.6, 1.0])
    assert "[Artifact] Cell centroids" in viewer.layers
    assert "[Artifact] Edges" in viewer.layers
    assert "[Artifact] T1 events" in viewer.layers
