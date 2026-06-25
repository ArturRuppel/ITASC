"""Tests for the standalone Cellpose segment+track widget + its compute steps.

The standalone tool is layer-based: no output directory, results are napari
layers tagged ``[Channel 1]`` / ``[Channel 2]``, and the embedded corrector binds
to the active Labels layer. Channel 1 is the anchor (segment + track); a Channel
2 makes the run joint (the only mode with a second channel).
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
    assert stw._layer_name("Channel 1", "masks") == "[Channel 1] masks"
    assert stw._layer_name("Channel 2", "tracked") == "[Channel 2] tracked"
    assert stw._layer_name("Channel 1", "preview") == "[Channel 1] preview"


# ── compute steps (Qt-free) ─────────────────────────────────────────────────
def test_segment_channel_returns_masks_array(tmp_path: Path, _model):
    # Layout-free: a (T, Y, X) stack canonicalises to (T, 1, Y, X) with no layout
    # arg, and is segmented per-plane (do_3d is always off in the standalone).
    raw = tmp_path / "nuc.tif"
    tifffile.imwrite(str(raw), np.zeros((2, 8, 8), dtype=np.float32))
    params = cellpose_runner.NucleusParams(
        do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0
    )
    masks = stw.segment_channel(raw, "nucleus", params)
    assert masks.shape == (2, 1, 8, 8)
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
    # Channel 1 (anchor) has the full segment + track row; Channel 2 has its
    # params + a joint preview + run (no independent segmentation).
    for name in (
        "ch1_params_btn", "ch1_preview_btn", "ch1_seg_btn", "ch1_track_btn",
        "ch2_params_btn", "ch2_preview_btn", "ch2_run_btn",
    ):
        assert isinstance(getattr(w, name), QToolButton), name
    # each channel row has a folder + load-from-layer button.
    for name in ("ch1_folder_btn", "ch1_layer_btn", "ch2_folder_btn", "ch2_layer_btn"):
        assert isinstance(getattr(w, name), QToolButton), name
    # the nucleus/cell vocabulary is gone — no independent Channel-2 seg/track, and
    # the old single ⧉ joint button has been split into preview + run.
    for gone in (
        "cell_seg_btn", "cell_track_btn", "cell_preview_btn", "nucleus_seg_btn",
        "joint_run_btn",
    ):
        assert not hasattr(w, gone), gone
    # no path text field and no output dir — a channel's source is its two pills,
    # which are checkable so the active source lights up.
    assert not hasattr(w, "_ch1_edit") and not hasattr(w, "_ch2_edit")
    assert not hasattr(w, "_output_dir_edit")
    assert not hasattr(w, "_standalone_fields")
    assert w.ch1_folder_btn.isCheckable() and w.ch1_layer_btn.isCheckable()
    assert w.ch1_seg_btn.text() == "▶" and w.ch1_preview_btn.text() == "▷"
    assert w.ch2_run_btn.text() == "▶" and w.ch2_preview_btn.text() == "▷"
    w.deleteLater()


def test_widget_exposes_joint_action_and_params():
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    # Channel 2's params button + its joint preview/run (joint is Channel 2's only
    # mode — there is no standalone "joint_params_btn" anymore).
    for name in ("ch2_params_btn", "ch2_preview_btn", "ch2_run_btn"):
        assert isinstance(getattr(w, name), QToolButton), name
    assert not hasattr(w, "joint_params_btn")
    assert w.ch2_preview_btn.text() == "▷" and w.ch2_run_btn.text() == "▶"
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
    w._set_channel_path(1, None)
    w._set_channel_path(2, None)
    # No inputs → joint preview + run disabled.
    assert w._both_inputs() is False
    assert w.ch2_run_btn.isEnabled() is False
    assert w.ch2_preview_btn.isEnabled() is False
    # Only Channel 1 → still disabled (a second channel is required for joint).
    ch1 = tmp_path / "ch1.tif"
    tifffile.imwrite(str(ch1), np.zeros((2, 4, 4), dtype=np.float32))
    w._set_channel_path(1, ch1)
    assert w.ch1_folder_btn.isChecked() is True  # browse pill lit for the file source
    assert w._both_inputs() is False
    assert w.ch2_run_btn.isEnabled() is False
    assert w.ch2_preview_btn.isEnabled() is False
    # Both channels → enabled.
    ch2 = tmp_path / "ch2.tif"
    tifffile.imwrite(str(ch2), np.zeros((2, 4, 4), dtype=np.float32))
    w._set_channel_path(2, ch2)
    assert w._both_inputs() is True
    assert w.ch2_run_btn.isEnabled() is True
    assert w.ch2_preview_btn.isEnabled() is True
    w.deleteLater()


def test_load_channel_from_layer_sources_in_memory_stack():
    """Load-from-layer sets a canonicalised in-memory stack and drops file mode."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    w._set_channel_path(1, None)
    w._set_channel_path(2, None)
    viewer.add_image(np.zeros((2, 8, 8), dtype=np.float32), name="raw nuclei")
    w._use_layer_as_channel(1, "raw nuclei")
    assert w._ch1_stack is not None and w._ch1_stack.shape == (2, 1, 8, 8)
    assert w._ch1_path is None
    assert w._ch1_layer_name == "raw nuclei"
    assert w._channel_present(1) is True
    # the layer pill is lit (active source), the browse pill is not.
    assert w.ch1_layer_btn.isChecked() is True
    assert w.ch1_folder_btn.isChecked() is False
    # choosing a file switches the channel back out of layer mode.
    w._set_channel_path(1, Path("/some/file.tif"))
    assert w._ch1_stack is None
    assert w.ch1_folder_btn.isChecked() is True
    assert w.ch1_layer_btn.isChecked() is False
    w.deleteLater()


def test_joint_enabled_with_layer_sourced_channels():
    """Both channels sourced from layers (no files) still enable joint."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    w._set_channel_path(1, None)
    w._set_channel_path(2, None)
    viewer.add_image(np.zeros((2, 8, 8), dtype=np.float32), name="n")
    viewer.add_image(np.zeros((2, 8, 8), dtype=np.float32), name="c")
    w._use_layer_as_channel(1, "n")
    w._use_layer_as_channel(2, "c")
    assert w._both_inputs() is True
    assert w.ch2_run_btn.isEnabled() is True
    assert w.ch2_preview_btn.isEnabled() is True
    w.deleteLater()


def test_segment_track_joint_returns_paired_stacks(tmp_path: Path, monkeypatch):
    """The widget's joint compute fn loads both stacks and pairs ch2→ch1 ids."""
    ch1 = tmp_path / "ch1.tif"
    ch2 = tmp_path / "ch2.tif"
    tifffile.imwrite(str(ch1), np.zeros((2, 8, 8), dtype=np.float32))
    tifffile.imwrite(str(ch2), np.zeros((2, 8, 8), dtype=np.float32))

    captured = {}

    def _fake_joint(ch1_stack, ch2_stack, *a, **k):
        captured["ch1_shape"] = ch1_stack.shape
        captured["ch2_shape"] = ch2_stack.shape
        tracked = np.zeros((2, 1, 8, 8), dtype=np.int32)
        tracked[:, 0, 4, 4] = 6
        return tracked, tracked.copy()

    monkeypatch.setattr(stw.joint_mod, "joint_segment_track", _fake_joint)
    ch1_out, ch2_out = stw.segment_track_joint(
        ch1, ch2,
        cellpose_runner.NucleusParams(
            do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0
        ),
        cellpose_runner.CellParams(diameter=0.0, min_size=0, gamma=1.0),
        stw.FlowFollowingParams(),
        max_distance=15.0, max_frame_gap=0,
    )
    # 2D+t input was canonicalised to (T, Z, Y, X) before joint compute.
    assert captured["ch1_shape"] == (2, 1, 8, 8)
    assert set(np.unique(ch2_out)) == set(np.unique(ch1_out)) == {0, 6}


def test_segment_channel_accepts_in_memory_array(_model):
    """A channel can be sourced from a viewer layer's array, not just a .tif."""
    arr = np.zeros((2, 8, 8), dtype=np.float32)  # e.g. a napari image layer's data
    params = cellpose_runner.NucleusParams(
        do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0
    )
    masks = stw.segment_channel(arr, "nucleus", params)
    assert masks.shape == (2, 1, 8, 8)


def test_preview_joint_embeds_only_current_frame(monkeypatch):
    """preview_joint runs the joint on one frame and embeds it full-size."""
    ch1 = np.zeros((3, 1, 6, 6), dtype=np.float32)
    ch2 = np.zeros((3, 1, 6, 6), dtype=np.float32)

    def _fake_joint(ch1_s, ch2_s, *a, **k):
        assert ch1_s.shape[0] == 1 and ch2_s.shape[0] == 1  # single frame fed in
        lab = np.zeros((1, 1, 6, 6), dtype=np.int32)
        lab[0, 0, 2, 2] = 5
        return lab, lab.copy()

    monkeypatch.setattr(stw.joint_mod, "joint_segment_track", _fake_joint)
    out = stw.preview_joint(
        ch1, ch2,
        cellpose_runner.NucleusParams(
            do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0
        ),
        cellpose_runner.CellParams(diameter=0.0, min_size=0, gamma=1.0),
        stw.FlowFollowingParams(), t=1,
        max_distance=15.0, max_frame_gap=0,
    )
    assert out.shape == (3, 1, 6, 6)
    assert out[1].any()                       # current frame populated
    assert not out[0].any() and not out[2].any()  # others stay background


def test_set_running_ch2_actions_swap_glyph():
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    for key, btn, glyph in (
        ("ch2_preview", w.ch2_preview_btn, "▷"),
        ("ch2_run", w.ch2_run_btn, "▶"),
    ):
        w._set_running(key)
        assert btn.text() == "✕"
        w._set_running(None)
        assert btn.text() == glyph
    w.deleteLater()


def test_widget_has_no_layout_or_3d_mode_controls():
    """P3.5: the input-layout picker, 3D mode and anisotropy are gone."""
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    for attr in (
        "nuc_layout_combo", "cell_layout_combo", "nuc_3d_chk", "nuc_anisotropy_spin",
    ):
        assert not hasattr(w, attr), attr
    # The anchor (Channel 1) is always segmented per-plane in the standalone.
    assert w._build_ch1_params().do_3d is False
    w.deleteLater()


def test_widget_has_no_output_dir_or_file_contract_in_source():
    src = Path(stw.__file__).read_text()
    # results are layers, so there is no flat-file output contract anymore.
    assert "_output_dir" not in src
    assert "{channel}_masks.tif" not in src and "{channel}_tracked.tif" not in src
    # layers are channel-tagged via the labelled helper.
    assert 'f"[{channel_label}] {kind}"' in src
    # the 4-way layout picker is gone (segmentation is layout-free per-plane).
    assert "_layout_combo" not in src
    assert "_LAYOUT_OPTIONS" not in src
    assert "currentText()" not in src  # no layout combo to read


def test_set_running_swaps_glyph_to_cancel_for_each_action():
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    for key, btn, glyph in (
        ("ch1_preview", w.ch1_preview_btn, "▷"),
        ("ch1_seg", w.ch1_seg_btn, "▶"),
        ("ch1_track", w.ch1_track_btn, "⊳"),
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
    name = stw._layer_name("Channel 1", "masks")
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


class _FakeLabelsLayer:
    """Minimal stand-in for a napari Labels layer for retrack tests."""

    def __init__(self, data, name="cells"):
        self.data = np.asarray(data)
        self.name = name

    def refresh(self):
        pass


def test_full_editing_unlocks_toolkit_and_retracker():
    """The standalone corrector runs full DB-free editing + the Q/E retracker."""
    from cellflow.napari.cell_correction_widget import CellCorrectionWidget

    QApplication.instance() or QApplication([])
    w = CellCorrectionWidget(
        _FakeViewer(), active_labels_layer_provider=lambda: None, full_editing=True
    )
    # contour_only dropped → spawn / erase / merge / swap / split + Delete are live.
    assert w.correction_widget._contour_only is False
    assert isinstance(w.retrack_back_btn, QToolButton)
    assert isinstance(w.retrack_fwd_btn, QToolButton)
    assert hasattr(w, "retrack_max_dist_spin")
    w.deleteLater()


def test_app_corrector_stays_contour_only_without_full_editing():
    """The integrated app's corrector (no full_editing) is unchanged."""
    from cellflow.napari.cell_correction_widget import CellCorrectionWidget

    QApplication.instance() or QApplication([])
    w = CellCorrectionWidget(_FakeViewer(), pos_dir_provider=lambda: None)
    assert w.correction_widget._contour_only is True
    assert not hasattr(w, "retrack_back_btn")
    assert not hasattr(w, "retrack_max_dist_spin")
    w.deleteLater()


def test_full_editing_retrack_propagates_ids_on_bound_layer():
    """Q/E retrack re-links the bound layer's later frames to the current one."""
    from cellflow.napari.cell_correction_widget import CellCorrectionWidget

    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = CellCorrectionWidget(
        viewer, active_labels_layer_provider=lambda: None, full_editing=True
    )
    # One cell drifting +2 x per frame, with garbled per-frame ids.
    stack = np.zeros((3, 30, 30), dtype=np.int32)
    stack[0, 10:14, 2:6] = 1
    stack[1, 10:14, 4:8] = 50
    stack[2, 10:14, 6:10] = 77
    layer = _FakeLabelsLayer(stack)
    w._active_bound_layer = layer
    viewer.dims.current_step = (0, 0)
    w._on_retrack("forward")
    out = layer.data
    assert int(out[0, 11, 3]) == 1   # anchor frame kept
    assert int(out[1, 11, 5]) == 1   # propagated
    assert int(out[2, 11, 7]) == 1
    assert w._correction_dirty is True
    w.deleteLater()


def test_full_editing_retrack_needs_multiframe_stack():
    """Retrack refuses a single 2-D frame (nothing to link)."""
    from cellflow.napari.cell_correction_widget import CellCorrectionWidget

    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = CellCorrectionWidget(
        viewer, active_labels_layer_provider=lambda: None, full_editing=True
    )
    w._active_bound_layer = _FakeLabelsLayer(np.zeros((8, 8), dtype=np.int32))
    viewer.dims.current_step = (0, 0)
    w._on_retrack("forward")  # no raise; reports via status
    assert "time-first" in w.correction_status_lbl.text().lower()
    w.deleteLater()
