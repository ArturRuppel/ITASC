"""Tests for the standalone Cellpose segment+track widget + its compute steps.

The standalone tool is layer-based: no output directory, results are napari
layers tagged ``[Nucleus]`` / ``[Cell]``, and the embedded corrector binds to the
active Labels layer.
"""
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

import napari

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


# ── shape helpers ────────────────────────────────────────────────────────────
def test_to_tzyx_and_squeeze_z_roundtrip():
    assert stw._to_tzyx(np.zeros((6, 6), np.int32)).shape == (1, 1, 6, 6)
    assert stw._to_tzyx(np.zeros((3, 6, 6), np.int32)).shape == (3, 1, 6, 6)
    assert stw._to_tzyx(np.zeros((2, 4, 6, 6), np.int32)).shape == (2, 4, 6, 6)
    # squeeze drops only a singleton Z
    assert stw._squeeze_z(np.zeros((2, 1, 6, 6), np.int32)).shape == (2, 6, 6)
    assert stw._squeeze_z(np.zeros((2, 3, 6, 6), np.int32)).shape == (2, 3, 6, 6)
    # (T, Y, X) -> (T, 1, Y, X) -> (T, Y, X) is exact
    a = np.arange(2 * 6 * 6).reshape(2, 6, 6)
    assert np.array_equal(stw._squeeze_z(stw._to_tzyx(a)), a)


def test_layer_name_tags_channel():
    assert stw._layer_name("nucleus", "masks") == "[Nucleus] masks"
    assert stw._layer_name("cell", "tracked") == "[Cell] tracked"
    assert stw._layer_name("cell", "preview") == "[Cell] preview"


# ── compute steps (Qt-free) ─────────────────────────────────────────────────
def test_segment_channel_returns_masks_array(tmp_path: Path, _model):
    raw = tmp_path / "nuc.tif"
    tifffile.imwrite(str(raw), np.zeros((2, 3, 8, 8), dtype=np.float32))
    params = cellpose_runner.NucleusParams(
        do_3d=True, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0
    )
    masks = stw.segment_channel(raw, "nucleus", params, "3D+t")
    assert masks.shape == (2, 3, 8, 8)
    assert masks.dtype == np.int32


def test_track_channel_links_in_memory_masks(monkeypatch):
    masks = np.zeros((2, 1, 6, 6), dtype=np.int32)
    masks[0, 0, 2:4, 0:2] = 1
    masks[1, 0, 2:4, 2:4] = 1

    import cellflow.cellpose.track_laptrack as tl

    def _fake_run(df, *, max_distance, max_frame_gap):
        df = df.copy()
        df["track_id"] = 0
        return df

    monkeypatch.setattr(tl, "_run_laptrack", _fake_run)
    tracked = stw.track_channel(masks, max_distance=10.0, max_frame_gap=0)
    assert tracked.shape == (2, 1, 6, 6)
    assert set(np.unique(tracked)) == {0, 1}


def test_preview_channel_masks_populates_only_current_frame(_model):
    stack = np.zeros((3, 1, 6, 6), dtype=np.float32)
    params = cellpose_runner.CellParams(diameter=0.0, min_size=0, gamma=1.0)
    out = stw.preview_channel_masks(stack, "cell", params, t=1, z=0)
    assert out.shape == (3, 1, 6, 6)
    assert out[1].any()                       # current frame populated
    assert not out[0].any() and not out[2].any()  # others stay background


# ── widget construction ─────────────────────────────────────────────────────
def test_widget_exposes_rows_actions_and_no_output_dir():
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    for name in (
        "nucleus_params_btn", "nucleus_preview_btn", "nucleus_seg_btn", "nucleus_track_btn",
        "cell_params_btn", "cell_preview_btn", "cell_seg_btn", "cell_track_btn",
    ):
        assert isinstance(getattr(w, name), QToolButton), name
    # only input pickers — the output dir is gone (results are layers).
    assert w._nucleus_edit is not None and w._cell_edit is not None
    assert not hasattr(w, "_output_dir_edit")
    assert "output_dir" not in w._standalone_fields()
    assert w.nucleus_seg_btn.text() == "▶" and w.nucleus_preview_btn.text() == "▷"
    w.deleteLater()


def test_widget_exposes_joint_action_and_params():
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    for name in ("joint_params_btn", "joint_run_btn"):
        assert isinstance(getattr(w, name), QToolButton), name
    assert w.joint_run_btn.text() == "⧉"
    # The joint flow-following knobs are exposed (decision: fg_threshold is a knob).
    params = w._build_flow_params()
    assert params.fg_threshold == 0.5
    assert params.flow_weight == 0.5
    assert params.max_assign_radius == 30.0
    w.deleteLater()


def test_joint_button_disabled_until_both_inputs(tmp_path: Path):
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    # Clear any paths restored from persisted QSettings so the test is deterministic.
    w._nucleus_edit.setText("")
    w._cell_edit.setText("")
    w._apply_paths()
    # No inputs → joint disabled.
    assert w._both_inputs() is False
    assert w.joint_run_btn.isEnabled() is False
    # Only one input → still disabled.
    nuc = tmp_path / "nuc.tif"
    tifffile.imwrite(str(nuc), np.zeros((2, 4, 4), dtype=np.float32))
    w._nucleus_edit.setText(str(nuc))
    w._apply_paths()
    assert w._both_inputs() is False
    assert w.joint_run_btn.isEnabled() is False
    # Both inputs → enabled.
    cell = tmp_path / "cell.tif"
    tifffile.imwrite(str(cell), np.zeros((2, 4, 4), dtype=np.float32))
    w._cell_edit.setText(str(cell))
    w._apply_paths()
    assert w._both_inputs() is True
    assert w.joint_run_btn.isEnabled() is True
    w.deleteLater()


def test_segment_track_joint_returns_paired_stacks(tmp_path: Path, monkeypatch):
    """The widget's joint compute fn loads both stacks and pairs cell→nucleus ids."""
    nuc = tmp_path / "nuc.tif"
    cell = tmp_path / "cell.tif"
    tifffile.imwrite(str(nuc), np.zeros((2, 8, 8), dtype=np.float32))
    tifffile.imwrite(str(cell), np.zeros((2, 8, 8), dtype=np.float32))

    captured = {}

    def _fake_joint(nuc_stack, cell_stack, *a, **k):
        captured["nuc_shape"] = nuc_stack.shape
        captured["cell_shape"] = cell_stack.shape
        tracked = np.zeros((2, 1, 8, 8), dtype=np.int32)
        tracked[:, 0, 4, 4] = 6
        return tracked, tracked.copy()

    monkeypatch.setattr(stw.joint_mod, "joint_segment_track", _fake_joint)
    nuc_out, cell_out = stw.segment_track_joint(
        nuc, cell, "2D+t", "2D+t",
        cellpose_runner.NucleusParams(
            do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0
        ),
        cellpose_runner.CellParams(diameter=0.0, min_size=0, gamma=1.0),
        stw.FlowFollowingParams(),
        max_distance=15.0, max_frame_gap=0,
    )
    # 2D+t input was canonicalised to (T, Z, Y, X) before joint compute.
    assert captured["nuc_shape"] == (2, 1, 8, 8)
    assert set(np.unique(cell_out)) == set(np.unique(nuc_out)) == {0, 6}


def test_set_running_joint_swaps_glyph():
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    w._set_running("joint")
    assert w.joint_run_btn.text() == "✕"
    w._set_running(None)
    assert w.joint_run_btn.text() == "⧉"
    w.deleteLater()


def test_widget_has_no_output_dir_or_file_contract_in_source():
    src = Path(stw.__file__).read_text()
    # results are layers, so there is no flat-file output contract anymore.
    assert "_output_dir" not in src
    assert "{channel}_masks.tif" not in src and "{channel}_tracked.tif" not in src
    # layers are channel-tagged.
    assert "[{channel.title()}] {kind}" in src


def test_set_running_swaps_glyph_to_cancel_for_each_action():
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    for key, btn, glyph in (
        ("nucleus_preview", w.nucleus_preview_btn, "▷"),
        ("cell_seg", w.cell_seg_btn, "▶"),
        ("cell_track", w.cell_track_btn, "⊳"),
    ):
        w._set_running(key)
        assert btn.text() == "✕"
        w._set_running(None)
        assert btn.text() == glyph
    w.deleteLater()


def test_factory_returns_widget():
    QApplication.instance() or QApplication([])
    w = stw.make_cellpose_segment_track_widget(_FakeViewer())
    assert isinstance(w, stw.CellposeSegmentTrackWidget)
    w.deleteLater()


# ── layer output ─────────────────────────────────────────────────────────────
def test_add_labels_squeezes_singleton_z_and_tags():
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    name = stw._layer_name("nucleus", "masks")
    w._add_labels(name, np.zeros((2, 1, 5, 5), dtype=np.int32))
    assert name in w.viewer.layers
    assert w.viewer.layers[name].data.shape == (2, 5, 5)  # Z=1 squeezed for napari
    # re-adding updates in place rather than duplicating.
    w._add_labels(name, np.ones((2, 1, 5, 5), dtype=np.int32))
    assert w.viewer.layers[name].data.max() == 1
    w.deleteLater()


# ── embedded cell correction (active-layer scope) ────────────────────────────
def test_corrector_binds_to_active_labels_layer_only():
    """The provider yields the active layer iff it is a Labels layer."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)

    # nothing active → no labels available to correct.
    assert w._active_labels_layer() is None
    assert w.cell_correction._active_layer_mode() is True
    assert w.cell_correction._correction_data_available() is False

    # a non-Labels active layer (e.g. an image) is rejected by the scope guard.
    viewer.layers.selection.active = SimpleNamespace(name="[Cell] image")
    assert w._active_labels_layer() is None
    assert w.cell_correction._correction_data_available() is False

    # a real Labels layer becomes the correction target.
    labels = napari.layers.Labels(np.zeros((2, 5, 5), dtype=np.int32), name="[Cell] tracked")
    viewer.layers.selection.active = labels
    assert w._active_labels_layer() is labels
    assert w.cell_correction._correction_data_available() is True
    w.deleteLater()


def test_corrector_activation_noops_without_labels_layer():
    """Toggling active with no Labels layer selected stays inactive and warns."""
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    w.cell_correction.active_btn.setChecked(True)  # fires the toggle handler
    assert w.cell_correction.active_btn.isChecked() is False
    w.deleteLater()


def test_corrector_save_button_hidden_in_active_layer_mode():
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    # no on-disk target in active-layer mode → the disk-save button is hidden.
    assert w.cell_correction.save_labels_btn.isHidden() is True
    w.deleteLater()


def test_cell_corrector_default_paths_unchanged():
    """Without overrides the corrector keeps the app's staged pos-dir paths."""
    from cellflow.napari.cell_correction_widget import CellCorrectionWidget

    QApplication.instance() or QApplication([])
    c = CellCorrectionWidget(_FakeViewer(), pos_dir_provider=lambda: Path("/proj/pos1"))
    assert c._active_layer_mode() is False
    assert c._cell_labels_path() == Path("/proj/pos1/3_cell/tracked_labels.tif")
    assert c._cell_foreground_path() == Path("/proj/pos1/1_cellpose/cell_foreground.tif")
    assert c._nuc_foreground_path() == Path("/proj/pos1/1_cellpose/nucleus_foreground.tif")
    c.deleteLater()
