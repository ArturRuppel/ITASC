"""Tests for per-stage click-to-load (Qt-free; a stub viewer stands in)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from cellflow.napari._stage_loader import (
    load_stage,
    stage_load_targets,
)
from cellflow.napari._stage_status import (
    STAGE_CELL,
    STAGE_CELLPOSE,
    STAGE_CONTACTS,
    STAGE_NUCLEUS,
)


class _StubLayer:
    def __init__(self, data):
        self.data = data


class _StubLayers:
    def __init__(self):
        self._by_name: dict[str, _StubLayer] = {}

    def __contains__(self, name):
        return name in self._by_name

    def __getitem__(self, name):
        return self._by_name[name]

    def remove(self, layer):
        for name, existing in list(self._by_name.items()):
            if existing is layer:
                del self._by_name[name]

    def add(self, name, data):
        self._by_name[name] = _StubLayer(data)


class _StubViewer:
    def __init__(self):
        self.layers = _StubLayers()
        self.image_calls: list[tuple[str, str]] = []
        self.labels_calls: list[str] = []

    def add_image(self, data, name, colormap="gray"):
        self.layers.add(name, data)
        self.image_calls.append((name, colormap))

    def add_labels(self, data, name):
        self.layers.add(name, data)
        self.labels_calls.append(name)


def _write(path: Path, dtype=np.uint16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.zeros((4, 4), dtype=dtype))


def test_nucleus_prefers_committed_over_working(tmp_path: Path):
    _write(tmp_path / "2_nucleus" / "tracked_labels.tif")
    # Working only → the working file.
    (target,) = stage_load_targets(tmp_path, STAGE_NUCLEUS)
    assert target.path == tmp_path / "2_nucleus" / "tracked_labels.tif"
    assert target.as_labels

    _write(tmp_path / "nucleus_labels.tif")
    (target,) = stage_load_targets(tmp_path, STAGE_NUCLEUS)
    assert target.path == tmp_path / "nucleus_labels.tif"


def test_cell_prefers_committed_over_working(tmp_path: Path):
    _write(tmp_path / "3_cell" / "tracked_labels.tif")
    (target,) = stage_load_targets(tmp_path, STAGE_CELL)
    assert target.path == tmp_path / "3_cell" / "tracked_labels.tif"
    _write(tmp_path / "cell_labels.tif")
    (target,) = stage_load_targets(tmp_path, STAGE_CELL)
    assert target.path == tmp_path / "cell_labels.tif"


def test_contacts_has_no_raw_load_target(tmp_path: Path):
    assert stage_load_targets(tmp_path, STAGE_CONTACTS) == []


def test_load_stage_adds_labels_layer(tmp_path: Path):
    _write(tmp_path / "nucleus_labels.tif")
    viewer = _StubViewer()
    loaded = load_stage(viewer, tmp_path, STAGE_NUCLEUS)
    assert loaded == [f"{tmp_path.name}:nucleus_labels"]
    assert viewer.labels_calls == [f"{tmp_path.name}:nucleus_labels"]


def test_load_stage_cellpose_loads_existing_maps_only(tmp_path: Path):
    _write(tmp_path / "1_cellpose" / "nucleus_foreground.tif")
    _write(tmp_path / "1_cellpose" / "nucleus_contours.tif")
    viewer = _StubViewer()
    loaded = load_stage(viewer, tmp_path, STAGE_CELLPOSE)
    # Only the two present maps load; the absent cell-channel maps are skipped.
    assert loaded == [f"{tmp_path.name}:nucleus_foreground", f"{tmp_path.name}:nucleus_contours"]
    assert dict(viewer.image_calls)[f"{tmp_path.name}:nucleus_contours"] == "magma"


def test_load_stage_replaces_existing_layer(tmp_path: Path):
    _write(tmp_path / "nucleus_labels.tif")
    viewer = _StubViewer()
    load_stage(viewer, tmp_path, STAGE_NUCLEUS)
    load_stage(viewer, tmp_path, STAGE_NUCLEUS)  # second click
    # Replaced in place, not duplicated.
    assert viewer.labels_calls.count(f"{tmp_path.name}:nucleus_labels") == 1


def test_load_stage_noop_without_pos_dir():
    viewer = _StubViewer()
    assert load_stage(viewer, None, STAGE_NUCLEUS) == []
