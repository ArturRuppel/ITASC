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

import napari  # noqa: E402

import itasc.cellpose.cellpose_runner as cellpose_runner  # noqa: E402
from itasc.napari import cellpose_segment_track_widget as stw  # noqa: E402


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
    """Cache a fake Cellpose model returning a full-frame mask + flows.

    Flows (dp at index 1, cellprob at index 2) are emitted so the masks+prob+flow
    capture path runs; the cellprob is left at 0 so its sigmoid is a flat 0.5.
    """
    class _Recorder:
        def eval(self, img, **kwargs):
            arr = np.asarray(img, dtype=np.float32)
            masks = np.ones(arr.shape, dtype=np.int32)
            n = 3 if arr.ndim == 3 else 2
            flows = (
                None,
                np.zeros((n, *arr.shape), np.float32),
                np.zeros(arr.shape, np.float32),
            )
            return masks, flows, None

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


def test_segment_done_status_names_next_step():
    msg = stw._segment_done_status("[Channel 1] masks", "[Channel 1] prob", "[Channel 1] flow")
    assert "'[Channel 1] masks'" in msg and "'[Channel 1] flow'" in msg
    assert "Track to link masks across time." in msg


def test_track_done_status_names_next_step():
    msg = stw._track_done_status("[Channel 1] masks")
    assert "'[Channel 1] masks'" in msg
    assert "updated in place" in msg
    assert "Select it below to correct." in msg


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

    import itasc.cellpose.track_laptrack as tl

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


def test_prob_threshold_is_inverse_sigmoid_of_prob_map():
    # The [0, 1] cutoff is read in the prob-map image's own space (sigmoid(cellprob))
    # and reversed by the inverse sigmoid (logit) to the raw cellprob Cellpose
    # thresholds. 0.5 -> 0.0 (Cellpose's default); exact reverse of the display.
    import math

    assert stw._prob_to_cellprob(0.5) == 0.0
    for p in (0.2, 0.7, 0.9):
        assert abs(stw._prob_to_cellprob(p) - math.log(p / (1 - p))) < 1e-6
    # round-trips through the prob-map sigmoid; symmetric about 0.5; finite at ends.
    from itasc.cellpose.native_masks import _sigmoid

    assert abs(float(_sigmoid(stw._prob_to_cellprob(0.3))) - 0.3) < 1e-6
    assert abs(stw._prob_to_cellprob(0.1) + stw._prob_to_cellprob(0.9)) < 1e-6
    assert math.isfinite(stw._prob_to_cellprob(0.0)) and math.isfinite(stw._prob_to_cellprob(1.0))


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
    # each channel row has a single load-from-active-layer pill (no file button).
    for name in ("ch1_layer_btn", "ch2_layer_btn"):
        assert isinstance(getattr(w, name), QToolButton), name
    # the nucleus/cell vocabulary is gone — no independent Channel-2 seg/track, and
    # the old single ⧉ joint button has been split into preview + run.
    for gone in (
        "cell_seg_btn", "cell_track_btn", "cell_preview_btn", "nucleus_seg_btn",
        "joint_run_btn",
    ):
        assert not hasattr(w, gone), gone
    # file loading is gone: no browse pill, no path text field, no output dir —
    # a channel's source is its single layer pill, which lights to show its binding.
    assert not hasattr(w, "ch1_folder_btn") and not hasattr(w, "ch2_folder_btn")
    assert not hasattr(w, "_ch1_edit") and not hasattr(w, "_ch2_edit")
    assert not hasattr(w, "_output_dir_edit")
    assert not hasattr(w, "_standalone_fields")
    assert w.ch1_layer_btn.isCheckable() and w.ch2_layer_btn.isCheckable()
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


def test_joint_button_disabled_until_both_inputs():
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    # No inputs → joint preview + run disabled.
    assert w._both_inputs() is False
    assert w.ch2_run_btn.isEnabled() is False
    assert w.ch2_preview_btn.isEnabled() is False
    # Only Channel 1 → still disabled (a second channel is required for joint).
    ch1 = viewer.add_image(np.zeros((2, 4, 4), dtype=np.float32), name="ch1")
    w._set_channel_layer(1, ch1)
    assert w.ch1_layer_btn.isChecked() is True  # pill lit for the bound layer
    assert w._both_inputs() is False
    assert w.ch2_run_btn.isEnabled() is False
    assert w.ch2_preview_btn.isEnabled() is False
    # Both channels → enabled.
    ch2 = viewer.add_image(np.zeros((2, 4, 4), dtype=np.float32), name="ch2")
    w._set_channel_layer(2, ch2)
    assert w._both_inputs() is True
    assert w.ch2_run_btn.isEnabled() is True
    assert w.ch2_preview_btn.isEnabled() is True
    w.deleteLater()


def test_ch1_preview_and_seg_buttons_disabled_until_bound():
    """Arrival UX: Channel 1's preview/segment are inert until a source is bound."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    assert w.ch1_preview_btn.isEnabled() is False
    assert w.ch1_seg_btn.isEnabled() is False
    assert "Bind Channel 1" in w.ch1_preview_btn.toolTip()
    assert "Bind Channel 1" in w.ch1_seg_btn.toolTip()
    ch1 = viewer.add_image(np.zeros((2, 4, 4), dtype=np.float32), name="ch1")
    w._set_channel_layer(1, ch1)
    assert w.ch1_preview_btn.isEnabled() is True
    assert w.ch1_seg_btn.isEnabled() is True
    w.deleteLater()


def test_ch1_track_button_reason_tracks_bind_then_segment_state():
    """Track's disabled reason distinguishes 'unbound' from 'bound but not segmented'."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    assert w.ch1_track_btn.isEnabled() is False
    assert "Bind Channel 1" in w.ch1_track_btn.toolTip()
    ch1 = viewer.add_image(np.zeros((2, 4, 4), dtype=np.float32), name="ch1")
    w._set_channel_layer(1, ch1)
    assert w.ch1_track_btn.isEnabled() is False
    assert "Segment Channel 1 first" in w.ch1_track_btn.toolTip()
    # a masks layer appearing (as Segment would produce) enables Track.
    viewer.add_labels(np.zeros((2, 4, 4), dtype=np.int32), name="[Channel 1] masks")
    w._on_layers_changed()
    assert w.ch1_track_btn.isEnabled() is True
    w.deleteLater()


def test_ch2_buttons_reason_names_the_missing_input():
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    assert "Bind Channel 1" in w.ch2_run_btn.toolTip()
    assert "Bind Channel 1" in w.ch2_preview_btn.toolTip()
    ch1 = viewer.add_image(np.zeros((2, 4, 4), dtype=np.float32), name="ch1")
    w._set_channel_layer(1, ch1)
    assert w.ch2_run_btn.isEnabled() is False
    assert "Bind Channel 2" in w.ch2_run_btn.toolTip()
    assert "Bind Channel 2" in w.ch2_preview_btn.toolTip()
    w.deleteLater()


def test_bind_channel_to_layer_sources_canonical_stack():
    """Binding a channel reads the layer's data as a canonical (T, Z, Y, X) stack."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    raw = viewer.add_image(np.zeros((2, 8, 8), dtype=np.float32), name="raw nuclei")
    w._set_channel_layer(1, raw)
    assert w._ch1_layer is raw
    assert w._channel_present(1) is True
    assert w._channel_source(1).shape == (2, 1, 8, 8)  # canonicalised on read
    # the pill is lit while the layer is bound + present.
    assert w.ch1_layer_btn.isChecked() is True
    w.deleteLater()


def test_status_label_shows_idle_hint_on_construction():
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    assert "Bind Channel 1 to begin" in w.status_lbl.text()
    assert w.status_lbl.isHidden() is False
    w.deleteLater()


def test_status_label_names_next_step_after_binding_channel_1():
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    ch1 = viewer.add_image(np.zeros((2, 4, 4), dtype=np.float32), name="ch1")
    w._set_channel_layer(1, ch1)
    assert "Preview a frame or Segment the full stack." in w.status_lbl.text()
    w.deleteLater()


def test_ch1_params_expose_prob_threshold_default_is_cellpose_default():
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    assert isinstance(w.ch1_prob_thr_spin, QToolButton) is False  # it is a slider
    assert w.ch1_prob_thr_spin.value() == 0.5
    # default 0.5 -> logit 0.0 keeps Channel-1 segmentation at Cellpose's default.
    assert w._build_ch1_params().cellprob_threshold == 0.0
    assert "prob_threshold" in w.get_state()["channel1"]
    w.deleteLater()


def test_ch1_params_expose_flow_threshold_and_niter():
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    # defaults match Cellpose: flow_threshold 0.4, niter 0 (== auto).
    assert w.ch1_flow_thr_spin.value() == 0.4
    assert w.ch1_niter_spin.value() == 0
    params = w._build_ch1_params()
    assert params.flow_threshold == 0.4
    assert params.niter == 0
    state = w.get_state()["channel1"]
    assert "flow_threshold" in state and "niter" in state
    w.deleteLater()


def test_segment_streaming_fills_masks_prob_flow_layers():
    """The streaming handler writes each frame into three [Channel 1] layers."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    T, Z, Y, X = 2, 1, 5, 5
    w._stream = {
        "masks": np.zeros((T, Z, Y, X), np.int32),
        "prob": np.zeros((T, Z, Y, X), np.float32),
        "flow": np.zeros((T, Z, Y, X, 3), np.uint8),
        "masks_name": "[Channel 1] masks",
        "prob_name": "[Channel 1] prob",
        "flow_name": "[Channel 1] flow",
    }
    # stream frame 0 only → its slice is filled, frame 1 stays background.
    masks0 = np.ones((Z, Y, X), np.int32)
    prob0 = np.full((Z, Y, X), 0.5, np.float32)
    flow0 = np.full((Z, Y, X, 3), 7, np.uint8)
    w._on_seg_frame((0, masks0, prob0, flow0))
    for name in ("[Channel 1] masks", "[Channel 1] prob", "[Channel 1] flow"):
        assert name in viewer.layers, name
    # singleton Z squeezed for display: labels/prob -> (T, Y, X), flow -> (T, Y, X, 3).
    assert viewer.layers["[Channel 1] masks"].data.shape == (T, Y, X)
    assert viewer.layers["[Channel 1] flow"].data.shape == (T, Y, X, 3)
    assert viewer.layers["[Channel 1] masks"].data[0].any()
    assert not viewer.layers["[Channel 1] masks"].data[1].any()
    w.deleteLater()


def test_segment_streaming_advances_frame_slider():
    """Each streamed frame drives the viewer's time slider to that frame."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    viewer.dims.current_step = (0, 0)
    w = stw.CellposeSegmentTrackWidget(viewer)
    T, Z, Y, X = 3, 1, 5, 5
    w._stream = {
        "masks": np.zeros((T, Z, Y, X), np.int32),
        "prob": np.zeros((T, Z, Y, X), np.float32),
        "flow": np.zeros((T, Z, Y, X, 3), np.uint8),
        "masks_name": "[Channel 1] masks",
        "prob_name": "[Channel 1] prob",
        "flow_name": "[Channel 1] flow",
    }
    frame = lambda t: (t, np.ones((Z, Y, X), np.int32),
                       np.zeros((Z, Y, X), np.float32),
                       np.zeros((Z, Y, X, 3), np.uint8))
    w._on_seg_frame(frame(2))
    assert viewer.dims.current_step[0] == 2
    # other slider axes are preserved
    assert viewer.dims.current_step[1] == 0
    w._on_seg_frame(frame(1))
    assert viewer.dims.current_step[0] == 1
    w.deleteLater()


def test_init_stream_matches_the_real_segment_layers():
    """_preview reuses _init_stream, so it fills the very same layers Segment does
    (no separate ``… preview`` layers) when lazily initializing on first use."""
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    w._init_stream(np.zeros((3, 1, 6, 6), dtype=np.float32))
    st = w._stream
    assert st["masks"].shape == (3, 1, 6, 6) and st["masks"].dtype == np.int32
    assert st["prob"].shape == (3, 1, 6, 6) and st["prob"].dtype == np.float32
    assert st["flow"].shape == (3, 1, 6, 6, 3) and st["flow"].dtype == np.uint8
    assert st["masks_name"] == "[Channel 1] masks"
    assert st["prob_name"] == "[Channel 1] prob"
    assert st["flow_name"] == "[Channel 1] flow"
    w.deleteLater()


def test_preview_writes_one_slice_without_clobbering_other_frames():
    """The preview _done callback (exercised directly, no thread) writes only the
    current frame's slice into the shared stream, leaving other frames alone."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    T, Z, Y, X = 3, 1, 5, 5
    w._init_stream(np.zeros((T, Z, Y, X), dtype=np.float32))
    st = w._stream
    st["masks"][0, 0] = 9  # pretend frame 0 was already segmented
    t, z = 1, 0
    masks, prob, flow = (
        np.ones((Y, X), np.int32), np.full((Y, X), 0.5, np.float32),
        np.full((Y, X, 3), 7, np.uint8),
    )
    st["masks"][t, z] = masks
    st["prob"][t, z] = prob
    st["flow"][t, z] = flow
    w._push_stream_layers()
    assert viewer.layers["[Channel 1] masks"].data[0].any()   # frame 0 preserved
    assert viewer.layers["[Channel 1] masks"].data[1].any()   # frame 1 now filled
    assert not viewer.layers["[Channel 1] masks"].data[2].any()  # frame 2 untouched
    assert "[Channel 1] preview" not in viewer.layers  # no throwaway preview layer
    w.deleteLater()


def test_init_stream_after_a_full_run_reuses_layer_data_not_zeros():
    """Regression: after a full Segment run completes, self._stream is cleared
    but the masks/prob/flow layers hold the real result. A later "segment this
    frame" (_preview) re-inits the stream via _init_stream — it must seed from
    that existing layer data, not zero it out, or the very first
    _push_stream_layers() call blanks every already-segmented frame."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    T, Z, Y, X = 3, 1, 5, 5
    masks = np.zeros((T, Y, X), np.int32)
    masks[:] = 1  # every frame "segmented" by a prior full run
    prob = np.full((T, Y, X), 0.7, np.float32)
    flow = np.full((T, Y, X, 3), 9, np.uint8)
    viewer.add_labels(masks, name="[Channel 1] masks")
    viewer.add_image(prob, name="[Channel 1] prob")
    viewer.add_image(flow, name="[Channel 1] flow")
    w._stream = None  # as left by _run_segment's _done callback

    w._init_stream(np.zeros((T, Z, Y, X), dtype=np.float32))

    st = w._stream
    assert st["masks"].shape == (T, Z, Y, X)
    assert (st["masks"] == 1).all()  # reused, not zeroed
    assert (st["prob"] == 0.7).all()
    assert (st["flow"] == 9).all()
    # pushing the rehydrated stream must not clobber the viewer layers either.
    w._push_stream_layers()
    assert (viewer.layers["[Channel 1] masks"].data == 1).all()
    w.deleteLater()


def test_init_stream_falls_back_to_zeros_when_layer_shape_differs():
    """A stale/mismatched layer (e.g. a different stack was loaded) must not be
    reused — fall back to a clean zero-filled accumulator instead."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    viewer.add_labels(np.ones((2, 5, 5), np.int32), name="[Channel 1] masks")

    w._init_stream(np.zeros((3, 1, 5, 5), dtype=np.float32))

    assert w._stream["masks"].shape == (3, 1, 5, 5)
    assert not w._stream["masks"].any()
    w.deleteLater()


def test_source_pill_darkens_when_bound_layer_removed():
    """The pill is a status light: it unlits + releases the channel on removal."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    raw = viewer.add_image(np.zeros((2, 8, 8), dtype=np.float32), name="raw")
    w._set_channel_layer(1, raw)
    assert w.ch1_layer_btn.isChecked() is True and w._channel_present(1) is True
    # remove the layer from the viewer → the pill goes dark, the binding is dropped.
    viewer.layers.remove(raw)
    w._on_layers_changed()
    assert w.ch1_layer_btn.isChecked() is False
    assert w._ch1_layer is None
    assert w._channel_present(1) is False
    w.deleteLater()


def test_bind_active_layer_rejects_non_image():
    """Binding pulls the active layer and rejects anything that is not an Image."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    # active layer is a Labels (not an image) → channel stays empty + a status hint.
    viewer.layers.selection.active = napari.layers.Labels(
        np.zeros((2, 5, 5), dtype=np.int32), name="lab"
    )
    w._bind_active_layer(1)
    assert w._channel_present(1) is False
    assert "image layer" in w.status_lbl.text().lower()
    # active layer is an Image → bound.
    img = napari.layers.Image(np.zeros((2, 8, 8), dtype=np.float32), name="img")
    viewer.layers["img"] = img
    viewer.layers.selection.active = img
    w._bind_active_layer(1)
    assert w._ch1_layer is img and w._channel_present(1) is True
    w.deleteLater()


def test_joint_enabled_with_layer_sourced_channels():
    """Both channels sourced from layers still enable joint."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    n = viewer.add_image(np.zeros((2, 8, 8), dtype=np.float32), name="n")
    c = viewer.add_image(np.zeros((2, 8, 8), dtype=np.float32), name="c")
    w._set_channel_layer(1, n)
    w._set_channel_layer(2, c)
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


def _fake_joint_cell_frame(cell_frame, nucleus_frame, z, params):
    """Stand-in for run_cell_frame_joint: distinct prob logits + flow per frame."""
    yx = cell_frame[z].shape
    prob_logits = np.zeros(yx, dtype=np.float32)  # sigmoid(0) == 0.5
    dp = np.ones((2, *yx), dtype=np.float32)
    return prob_logits, dp


def test_preview_joint_embeds_only_current_frame(monkeypatch):
    """preview_joint runs the joint on one frame and embeds assignment/prob/flow full-size."""
    ch1 = np.zeros((3, 1, 6, 6), dtype=np.float32)
    ch2 = np.zeros((3, 1, 6, 6), dtype=np.float32)

    def _fake_joint(ch1_s, ch2_s, *a, **k):
        assert ch1_s.shape[0] == 1 and ch2_s.shape[0] == 1  # single frame fed in
        lab = np.zeros((1, 1, 6, 6), dtype=np.int32)
        lab[0, 0, 2, 2] = 5
        return lab, lab.copy()

    monkeypatch.setattr(stw.joint_mod, "joint_segment_track", _fake_joint)
    monkeypatch.setattr(
        stw.cellpose_runner, "run_cell_frame_joint", _fake_joint_cell_frame
    )
    assignment, prob, flow = stw.preview_joint(
        ch1, ch2,
        cellpose_runner.NucleusParams(
            do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0
        ),
        cellpose_runner.CellParams(diameter=0.0, min_size=0, gamma=1.0),
        stw.FlowFollowingParams(), t=1,
        max_distance=15.0, max_frame_gap=0,
    )
    # Assignment: only the current frame populated.
    assert assignment.shape == (3, 1, 6, 6)
    assert assignment[1].any()
    assert not assignment[0].any() and not assignment[2].any()
    # Prob/flow maps come from the joint cell pass, embedded on the same frame only.
    assert prob.shape == (3, 1, 6, 6)
    assert flow.shape == (3, 1, 6, 6, 3)
    np.testing.assert_allclose(prob[1], 0.5)   # sigmoid(0) from the fake joint pass
    assert prob[0].sum() == 0 and prob[2].sum() == 0
    assert flow[1].any() and not flow[0].any() and not flow[2].any()


def test_preview_joint_sources_maps_from_joint_two_channel_pass(monkeypatch):
    """The prob/flow shown come from run_cell_frame_joint fed BOTH channels."""
    ch1 = np.full((2, 1, 5, 5), 7.0, dtype=np.float32)   # nucleus channel
    ch2 = np.full((2, 1, 5, 5), 3.0, dtype=np.float32)   # cell channel

    def _fake_joint(ch1_s, ch2_s, *a, **k):
        lab = np.zeros((1, 1, 5, 5), dtype=np.int32)
        return lab, lab.copy()

    seen = {}

    def _record_cell_frame(cell_frame, nucleus_frame, z, params):
        seen["cell"] = np.asarray(cell_frame).copy()
        seen["nucleus"] = np.asarray(nucleus_frame).copy()
        yx = cell_frame[z].shape
        return np.zeros(yx, dtype=np.float32), np.ones((2, *yx), dtype=np.float32)

    monkeypatch.setattr(stw.joint_mod, "joint_segment_track", _fake_joint)
    monkeypatch.setattr(stw.cellpose_runner, "run_cell_frame_joint", _record_cell_frame)
    stw.preview_joint(
        ch1, ch2,
        cellpose_runner.NucleusParams(
            do_3d=False, anisotropy=1.0, diameter=0.0, min_size=0, gamma=1.0
        ),
        cellpose_runner.CellParams(diameter=0.0, min_size=0, gamma=1.0),
        stw.FlowFollowingParams(), t=0,
        max_distance=15.0, max_frame_gap=0,
    )
    # Cell channel (ch2) is the primary; nucleus channel (ch1) is the guide.
    np.testing.assert_array_equal(seen["cell"], ch2[0])
    np.testing.assert_array_equal(seen["nucleus"], ch1[0])


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


def test_activate_button_disabled_until_labels_layer_selected():
    """Arrival UX: the ⏻ activate button is inert until a Labels layer is active."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    cc = w.cell_correction
    assert cc.active_btn.isEnabled() is False
    assert "Select a Labels layer" in cc.active_btn.toolTip()
    labels = napari.layers.Labels(np.zeros((2, 5, 5), dtype=np.int32), name="[Channel 1] tracked")
    viewer.layers.selection.active = labels
    # the fake viewer's selection has no real events to fire (see
    # test_source_pill_darkens_when_bound_layer_removed for the same pattern),
    # so drive the recompute directly.
    cc._sync_active_btn_enabled()
    assert cc.active_btn.isEnabled() is True
    assert cc.active_btn.toolTip() == "Activate correction mode and show correction controls."
    w.deleteLater()


def test_activate_button_stays_enabled_once_checked_even_if_selection_moves_away():
    """A live correction session can always be turned back off."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    cc = w.cell_correction
    labels = napari.layers.Labels(np.zeros((2, 5, 5), dtype=np.int32), name="[Channel 1] tracked")
    viewer.layers.selection.active = labels
    cc.active_btn.setChecked(True)  # activates correction on the bound layer
    assert cc.active_btn.isChecked() is True
    # selection moves to something that is not a Labels layer mid-session.
    viewer.layers.selection.active = SimpleNamespace(name="[Channel 1] image")
    cc._sync_active_btn_enabled()
    assert cc.active_btn.isEnabled() is True  # still togglable off
    w.deleteLater()


def test_spawn_intensity_frame_uses_prob_layer_sharing_the_tag():
    """Spawn snaps to the prob layer tagged like the layer under correction."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    prob = np.arange(2 * 5 * 5, dtype=np.float32).reshape(2, 5, 5)
    viewer.layers["[Channel 1] prob"] = SimpleNamespace(data=prob, name="[Channel 1] prob")
    labels = napari.layers.Labels(np.zeros((2, 5, 5), dtype=np.int32), name="[Channel 1] masks")
    viewer.layers.selection.active = labels
    w.cell_correction.active_btn.setChecked(True)  # binds _active_bound_layer

    frame = w._spawn_intensity_frame(1)

    assert np.array_equal(frame, prob[1])
    w.deleteLater()


def test_spawn_intensity_frame_none_without_matching_prob_layer():
    """No same-tag prob layer (e.g. Channel 2, which is never segmented on its
    own) falls back to None → plain-disk spawn."""
    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = stw.CellposeSegmentTrackWidget(viewer)
    labels = napari.layers.Labels(np.zeros((2, 5, 5), dtype=np.int32), name="[Channel 2] tracked")
    viewer.layers.selection.active = labels
    w.cell_correction.active_btn.setChecked(True)

    assert w._spawn_intensity_frame(0) is None
    w.deleteLater()


def test_cell_corrector_default_paths_unchanged():
    """Without overrides the corrector keeps the app's staged pos-dir paths."""
    from itasc.napari.correction.cell_correction_widget import CellCorrectionWidget

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

    def _save_history(self, entry):
        pass


def test_full_editing_unlocks_toolkit_and_retracker():
    """The standalone corrector runs full DB-free editing + the Q/E retracker."""
    from itasc.napari.correction.cell_correction_widget import CellCorrectionWidget

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
    from itasc.napari.correction.cell_correction_widget import CellCorrectionWidget

    QApplication.instance() or QApplication([])
    w = CellCorrectionWidget(_FakeViewer(), pos_dir_provider=lambda: None)
    assert w.correction_widget._contour_only is True
    assert not hasattr(w, "retrack_back_btn")
    assert not hasattr(w, "retrack_max_dist_spin")
    w.deleteLater()


def test_full_editing_retrack_propagates_ids_on_bound_layer():
    """Q/E retrack re-links the bound layer's later frames to the current one."""
    from itasc.napari.correction.cell_correction_widget import CellCorrectionWidget

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
    from itasc.napari.correction.cell_correction_widget import CellCorrectionWidget

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


def _make_three_frame_two_cell_stack():
    """Cell 1 present in every frame; cell 2 only in frame 0."""
    stack = np.zeros((3, 20, 20), dtype=np.int32)
    stack[0, 2:5, 2:5] = 1
    stack[1, 2:5, 2:5] = 1
    stack[2, 2:5, 2:5] = 1
    stack[0, 10:13, 10:13] = 2
    return stack


def test_toggle_validation_flags_and_unflags_across_a_cells_frames():
    """V / the ✓ button validates a cell across every frame it appears in, then
    invalidates it back — a DB-free in-memory analogue of the nucleus tool's
    validated_cells.json (no project directory exists in standalone mode)."""
    from itasc.napari.correction.cell_correction_widget import CellCorrectionWidget

    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = CellCorrectionWidget(
        viewer, active_labels_layer_provider=lambda: None, full_editing=True
    )
    w._active_bound_layer = _FakeLabelsLayer(_make_three_frame_two_cell_stack())
    viewer.dims.current_step = (0, 0)
    w.correction_widget._selected_label = 1

    w._on_toggle_validation()
    assert w._validated_tracks == {1: {0, 1, 2}}
    assert "validated" in w.correction_status_lbl.text().lower()

    w._on_toggle_validation()
    assert w._validated_tracks == {}
    assert "invalidated" in w.correction_status_lbl.text().lower()
    w.deleteLater()


def test_toggle_validation_requires_a_selected_cell_present_at_the_frame():
    from itasc.napari.correction.cell_correction_widget import CellCorrectionWidget

    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = CellCorrectionWidget(
        viewer, active_labels_layer_provider=lambda: None, full_editing=True
    )
    w._active_bound_layer = _FakeLabelsLayer(_make_three_frame_two_cell_stack())
    viewer.dims.current_step = (1, 0)  # cell 2 is absent here
    w.correction_widget._selected_label = 2

    w._on_toggle_validation()
    assert w._validated_tracks == {}
    assert "not present" in w.correction_status_lbl.text().lower()
    w.deleteLater()


def test_remove_unvalidated_labels_clears_everything_but_the_validated_cell():
    """The 🗑 clear-not-validated action zeroes every cell not flagged validated,
    reusing itasc.napari.correction._correction_utils.remove_unvalidated_labels verbatim
    (the same pure function the nucleus tool's DB-free clear action calls)."""
    from itasc.napari.correction.cell_correction_widget import CellCorrectionWidget

    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = CellCorrectionWidget(
        viewer, active_labels_layer_provider=lambda: None, full_editing=True
    )
    layer = _FakeLabelsLayer(_make_three_frame_two_cell_stack())
    w._active_bound_layer = layer
    viewer.dims.current_step = (0, 0)
    w._validated_tracks = {1: {0, 1, 2}}  # cell 1 validated everywhere; cell 2 not

    w._on_remove_unvalidated_labels()

    out = layer.data
    assert (out == 1).any()          # validated cell survives
    assert not (out == 2).any()      # unvalidated cell removed
    assert "removed unvalidated" in w.correction_status_lbl.text().lower()
    w.deleteLater()


def test_remove_unvalidated_labels_noop_status_when_all_validated():
    from itasc.napari.correction.cell_correction_widget import CellCorrectionWidget

    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = CellCorrectionWidget(
        viewer, active_labels_layer_provider=lambda: None, full_editing=True
    )
    stack = np.zeros((2, 10, 10), dtype=np.int32)
    stack[:, 2:5, 2:5] = 1
    layer = _FakeLabelsLayer(stack)
    w._active_bound_layer = layer
    w._validated_tracks = {1: {0, 1}}

    w._on_remove_unvalidated_labels()

    assert np.array_equal(layer.data, stack)  # nothing changed
    assert "no unvalidated labels" in w.correction_status_lbl.text().lower()
    w.deleteLater()


def test_track_list_navigator_present_only_in_full_editing():
    """The swimlane track-list navigator is part of the DB-free full-editing
    toolkit only — the integrated app's contour-only corrector is unchanged."""
    from itasc.napari.correction.cell_correction_widget import CellCorrectionWidget
    from itasc.napari.correction._correction_track_accordion import TrackAccordionPanel

    QApplication.instance() or QApplication([])
    full = CellCorrectionWidget(
        _FakeViewer(), active_labels_layer_provider=lambda: None, full_editing=True
    )
    assert isinstance(full._lineage_canvas.panel(), TrackAccordionPanel)
    full.deleteLater()

    app_widget = CellCorrectionWidget(_FakeViewer(), pos_dir_provider=lambda: None)
    assert app_widget._lineage_canvas is None
    assert not hasattr(app_widget, "validate_btn")
    assert not hasattr(app_widget, "remove_unvalidated_btn")
    app_widget.deleteLater()


def test_track_list_navigator_click_jumps_frame_and_selects_cell():
    """A bar click on the accordion (node_activated) jumps the viewer's time
    slider and selects that cell — the same navigation the nucleus tool's
    lineage canvas drives, adapted to the standalone active-layer corrector."""
    from itasc.napari.correction.cell_correction_widget import CellCorrectionWidget

    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = CellCorrectionWidget(
        viewer, active_labels_layer_provider=lambda: None, full_editing=True
    )
    w._active_bound_layer = _FakeLabelsLayer(_make_three_frame_two_cell_stack())
    viewer.dims.current_step = (0, 0)

    w._navigate_to_cell_from_lineage(2, 1)

    assert viewer.dims.current_step[0] == 2
    assert w.correction_widget._selected_label == 1
    w.deleteLater()


def test_validating_a_cell_draws_green_border_overlay_layer():
    """Validating a cell adds a new, contour-only Labels layer masking exactly
    the frames it's validated for — the standalone stand-in for
    ValidatedOverlayController's opaque green border."""
    from itasc.napari.correction.cell_correction_widget import (
        CellCorrectionWidget,
        _VALIDATED_OVERLAY_CONTOUR,
        _VALIDATED_OVERLAY_LAYER,
    )

    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = CellCorrectionWidget(
        viewer, active_labels_layer_provider=lambda: None, full_editing=True
    )
    stack = _make_three_frame_two_cell_stack()
    w._active_bound_layer = _FakeLabelsLayer(stack)
    viewer.dims.current_step = (0, 0)
    w.correction_widget._selected_label = 1

    assert _VALIDATED_OVERLAY_LAYER not in viewer.layers
    w._on_toggle_validation()  # validates cell 1 across frames 0, 1, 2

    assert _VALIDATED_OVERLAY_LAYER in viewer.layers
    overlay = viewer.layers[_VALIDATED_OVERLAY_LAYER]
    assert overlay.contour == _VALIDATED_OVERLAY_CONTOUR
    assert overlay.opacity == 1.0
    expected = (stack == 1).astype(np.uint8)  # cell 1 present in every frame
    assert np.array_equal(overlay.data, expected)
    assert not np.array_equal(overlay.data, (stack == 2).astype(np.uint8))
    w.deleteLater()


def test_invalidating_the_last_validated_cell_removes_the_overlay_layer():
    from itasc.napari.correction.cell_correction_widget import (
        CellCorrectionWidget,
        _VALIDATED_OVERLAY_LAYER,
    )

    QApplication.instance() or QApplication([])
    viewer = _FakeViewer()
    w = CellCorrectionWidget(
        viewer, active_labels_layer_provider=lambda: None, full_editing=True
    )
    w._active_bound_layer = _FakeLabelsLayer(_make_three_frame_two_cell_stack())
    viewer.dims.current_step = (0, 0)
    w.correction_widget._selected_label = 1

    w._on_toggle_validation()  # validate
    assert _VALIDATED_OVERLAY_LAYER in viewer.layers
    w._on_toggle_validation()  # invalidate
    assert _VALIDATED_OVERLAY_LAYER not in viewer.layers
    w.deleteLater()


def test_aborted_returns_ui_to_idle_after_cancel():
    """Cancel quits the worker, and napari answers a quit with the `aborted`
    signal — not `returned`/`errored`. The handler must reset the run buttons,
    progress bar, stream and status just like a normal finish. Regression:
    `aborted` used to be unconnected, so Cancel left the UI stuck."""
    QApplication.instance() or QApplication([])
    w = stw.CellposeSegmentTrackWidget(_FakeViewer())
    # A full-stack Segment run in progress: worker live, stream allocated,
    # buttons showing the Cancel glyph, progress bar advancing.
    w._worker = object()
    w._stream = {"masks_name": "[Channel 1] masks"}
    w._set_running("ch1_seg")
    w._progress(3, 10, "Segmenting Channel 1...")
    assert w.ch1_seg_btn.text() == "✕"

    w._aborted()  # what napari's `aborted` signal now invokes

    assert w._running is None
    assert w._worker is None
    assert w._stream is None
    assert w.progress_bar.value() == 0
    assert w.ch1_seg_btn.text() == "▶"  # glyph restored → button back to idle
    assert w.status_lbl.text() == "Cancelled."
    w.deleteLater()


def test_every_cancellable_worker_wires_aborted():
    """Every worker that handles `errored` must also handle `aborted`, or Cancel
    (which calls worker.quit() → napari `aborted`) leaves the run buttons
    disabled and the progress bar frozen. Guards the specific wiring bug."""
    src = Path(stw.__file__).read_text()
    assert src.count('"aborted": self._aborted') == src.count('"errored": self._errored')
    assert src.count('"aborted": self._aborted') == 5
