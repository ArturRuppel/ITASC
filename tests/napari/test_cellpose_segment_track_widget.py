"""Tests for the standalone Cellpose segment+track widget + its compute steps."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QToolButton

# A GUI QApplication must exist before napari (imported by the widget module) is
# pulled in, or QWidget construction aborts headless. Create it up front.
_APP = QApplication.instance() or QApplication([])

import cellflow.cellpose.cellpose_runner as cellpose_runner
from cellflow.napari import cellpose_segment_track_widget as stw


# ── fakes ──────────────────────────────────────────────────────────────────
class _Ev:
    def connect(self, *a, **k):
        pass

    def disconnect(self, *a, **k):
        pass


class _Sel:
    def __init__(self) -> None:
        self.active = None

    def connect(self, *a, **k):
        pass


class _LayerCollection(dict):
    def __init__(self) -> None:
        super().__init__()
        self.selection = _Sel()
        self.events = SimpleNamespace(inserted=_Ev(), removed=_Ev(), changed=_Ev())

    def remove(self, layer):
        self.pop(layer.name, None)


class _FakeViewer:
    def __init__(self) -> None:
        self.layers = _LayerCollection()
        self.dims = SimpleNamespace(
            current_step=(0, 0),
            events=SimpleNamespace(current_step=SimpleNamespace(connect=lambda cb: None)),
        )

    def add_labels(self, data, *, name, **kwargs):
        self.layers[name] = SimpleNamespace(data=np.asarray(data), name=name)
        return self.layers[name]

    add_image = add_labels


@pytest.fixture
def _model(monkeypatch):
    """Cache a fake Cellpose model returning a full-frame labelled mask."""
    class _Recorder:
        def eval(self, img, **kwargs):
            arr = np.asarray(img, dtype=np.float32)
            return np.ones(arr.shape, dtype=np.int32), (None, None, None), None

    monkeypatch.setattr(cellpose_runner, "_MODEL", _Recorder())


# ── compute steps (Qt-free) ─────────────────────────────────────────────────
def test_segment_channel_writes_flat_masks(tmp_path: Path, _model):
    raw = tmp_path / "nuc.tif"
    tifffile.imwrite(str(raw), np.zeros((2, 3, 8, 8), dtype=np.float32))
    out = tmp_path / "out"
    params = cellpose_runner.NucleusParams(
        do_3d=True, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0
    )
    path = stw.segment_channel(raw, out, "nucleus", params, "3D+t")
    assert path == out / "nucleus_masks.tif"
    masks = stw._normalize_tzyx_labels(tifffile.imread(str(path)))
    assert masks.shape == (2, 3, 8, 8)


def test_track_channel_writes_flat_tracked(tmp_path: Path, monkeypatch):
    masks = np.zeros((2, 1, 6, 6), dtype=np.int32)
    masks[0, 0, 2:4, 0:2] = 1
    masks[1, 0, 2:4, 2:4] = 1
    masks_path = tmp_path / "cell_masks.tif"
    tifffile.imwrite(str(masks_path), masks, metadata={"axes": "TZYX"})

    import cellflow.cellpose.track_laptrack as tl

    def _fake_run(df, *, max_distance, max_frame_gap):
        df = df.copy()
        df["track_id"] = 0
        return df

    monkeypatch.setattr(tl, "_run_laptrack", _fake_run)
    path = stw.track_channel(masks_path, tmp_path, "cell", max_distance=10.0, max_frame_gap=0)
    assert path == tmp_path / "cell_tracked.tif"
    tracked = stw._normalize_tzyx_labels(tifffile.imread(str(path)))
    assert set(np.unique(tracked)) == {0, 1}


def test_normalize_pads_to_tzyx():
    assert stw._normalize_tzyx_labels(np.zeros((6, 6), np.int32)).shape == (1, 1, 6, 6)
    assert stw._normalize_tzyx_labels(np.zeros((3, 6, 6), np.int32)).shape == (1, 3, 6, 6)


# ── widget construction ─────────────────────────────────────────────────────
def test_widget_exposes_two_rows_and_actions():
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    for name in (
        "nucleus_params_btn", "nucleus_seg_btn", "nucleus_track_btn",
        "cell_params_btn", "cell_seg_btn", "cell_track_btn",
    ):
        assert isinstance(getattr(w, name), QToolButton), name
    # explicit file pickers exist (no staged layout).
    assert w._nucleus_edit is not None and w._output_dir_edit is not None
    assert w.nucleus_seg_btn.text() == "▶"
    w.deleteLater()


def test_set_running_swaps_glyph_to_cancel():
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    w._set_running("nucleus_seg")
    assert w.nucleus_seg_btn.text() == "✕"
    w._set_running(None)
    assert w.nucleus_seg_btn.text() == "▶"
    w.deleteLater()


# ── source/file contract ────────────────────────────────────────────────────
def test_widget_uses_flat_output_contract():
    src = Path(stw.__file__).read_text()
    # outputs are built flat from the channel name, in the chosen output dir.
    assert "{channel}_masks.tif" in src
    assert "{channel}_tracked.tif" in src
    # the standalone tool must NOT enforce the staged pipeline layout
    # (no staged-path string literals are built anywhere in the module).
    assert '"0_input"' not in src and "'0_input'" not in src
    assert '"1_cellpose"' not in src and "'1_cellpose'" not in src


def test_factory_returns_widget():
    QApplication.instance() or QApplication([])
    w = stw.make_cellpose_segment_track_widget(_FakeViewer())
    assert isinstance(w, stw.CellposeSegmentTrackWidget)
    w.deleteLater()
