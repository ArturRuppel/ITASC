"""Standalone Cellpose *segment + track* widget for the ``cellflow-cellpose`` tool.

This is the independently-shipped distribution's own surface — distinct from the
app's :class:`~cellflow.napari.cellpose_widget.CellposeWidget`, which stays
untouched and keeps emitting divergence maps for the integrated pipeline.

The tool works on **one or two channels**, with no nucleus/cell vocabulary:

* **Channel 1** is the *anchor*. On its own it is segmented (Cellpose **native
  masks**, :mod:`cellflow.cellpose.native_masks`) and tracked across time with
  **laptrack** (:mod:`cellflow.cellpose.track_laptrack`) — the single-channel
  segment + track product.
* **Channel 2** is optional. When it is present the tool runs **joint** mode and
  *only* joint mode: Channel 1 is segmented + tracked, then each Channel 2
  foreground pixel is flowed along Cellpose's flow field onto the nearest
  Channel-1 object (:mod:`cellflow.cellpose.flow_following`). You get one
  Channel-2 object per Channel-1 object, sharing its track id. Channel 2 is never
  segmented independently — there is no separate-masks alternative.

Conventionally Channel 1 is the nucleus (a clean, separable anchor) and Channel 2
the cell, but nothing here assumes that.

Each channel's input is picked through its two pills — a multi-dimensional
``.tif`` (browse pill) or an image already open in the viewer (load-from-layer
pill); there is no path text field, and the active pill lights up to show which
source is in play. **There is no output directory**: every result is added straight to
the napari viewer as a layer (tagged ``[Channel 1]`` / ``[Channel 2]``), and the
user saves whichever layers they want via napari's own *Save Selected Layers*.
The embedded corrector edits whichever Labels layer is active, with the full
DB-free toolkit — select / spawn / erase / merge / swap / split (mouse + Delete),
fill-holes / fragment cleanup, and a greedy retracker on Q/E.
"""
from __future__ import annotations

import logging
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from qtpy.QtCore import QSettings, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._widget_helpers import (
    dslider as _dslider,
    islider as _islider,
    tool_btn as _tool_btn,
)
from cellflow.napari.ui_gate import ControlClass, UiGate
from cellflow.napari.ui_style import (
    add_section_pair_row,
    section_grid,
    stage_header_action_button,
    stage_header_label,
    status_label,
)
from cellflow.napari.widgets import CollapsibleSection
from cellflow.napari.cell_correction_widget import CellCorrectionWidget
from cellflow.cellpose import cellpose_runner, native_masks, track_laptrack
from cellflow.cellpose import joint as joint_mod
from cellflow.cellpose.flow_following import FlowFollowingParams
from cellflow.cellpose.shape import describe_axes, to_canonical_tzyx

logger = logging.getLogger(__name__)

# User-facing channel labels (no nucleus/cell vocabulary). Channel 1 is the
# anchor that is segmented + tracked; Channel 2 (optional) is flowed onto it.
_CH1_LABEL = "Channel 1"
_CH2_LABEL = "Channel 2"

# File dialog filter for the .tif inputs each channel can take.
_IMAGE_FILTER = "Images (*.tif *.tiff);;All files (*)"


# ---------------------------------------------------------------------------
# Array-shape helpers (canonical compute shape is (T, Z, Y, X))
# ---------------------------------------------------------------------------
def _to_tzyx(arr: np.ndarray) -> np.ndarray:
    """Coerce a label/image array to canonical ``(T, Z, Y, X)``.

    A 3-D array is read as ``(T, Y, X)`` and gains a singleton Z (the inverse of
    :func:`_squeeze_z`, so a round-trip through a squeezed napari layer is exact);
    a 2-D array becomes a single ``(1, 1, Y, X)`` frame.
    """
    arr = np.asarray(arr)
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[:, np.newaxis]
    if arr.ndim == 2:
        return arr[np.newaxis, np.newaxis]
    raise ValueError(f"expected a 2-D..4-D array, got shape {arr.shape}")


def _squeeze_z(arr: np.ndarray) -> np.ndarray:
    """Drop a singleton Z from ``(T, 1, Y, X)`` → ``(T, Y, X)`` for napari/corrector.

    2D+t data (Z=1) displays without a spurious slider and matches the basic
    corrector's 3-D ``(T, Y, X)`` expectation. True 3-D+t (Z>1) is left as-is.
    """
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr


def _layer_name(channel_label: str, kind: str) -> str:
    """Channel-tagged layer name, e.g. ``[Channel 1] masks`` / ``[Channel 2] tracked``."""
    return f"[{channel_label}] {kind}"


def _coerce_stack(source) -> np.ndarray:
    """Canonical ``(T, Z, Y, X)`` from either source a channel can have.

    ``source`` is an in-memory array (a napari image layer's ``data``) or a
    ``.tif`` path — so a channel reads identically whether it came from disk or
    from a layer already open in the viewer.
    """
    if isinstance(source, np.ndarray):
        return to_canonical_tzyx(source)
    return to_canonical_tzyx(np.asarray(tifffile.imread(str(source))))


# ---------------------------------------------------------------------------
# Qt-free compute steps (callable directly in tests; the worker just wraps them)
# ---------------------------------------------------------------------------
def segment_channel(
    source,
    channel: str,
    params,
    *,
    progress_cb=None,
    cancel_cb=None,
) -> np.ndarray:
    """Load a raw stack and return per-plane native masks ``(T, Z, Y, X)``.

    ``source`` is a ``.tif`` path or an in-memory array (a viewer layer). It is
    canonicalised layout-free (:func:`to_canonical_tzyx`) — no 2D/2D+t/3D/3D+t
    declaration — and every plane is segmented individually. ``channel`` selects
    the backend mask routine (the anchor uses ``"nucleus"``).
    """
    stack = _coerce_stack(source)
    if channel == "nucleus":
        return native_masks.run_nucleus_masks_stack(
            stack, params, progress_cb=progress_cb, cancel_cb=cancel_cb
        )
    return native_masks.run_cell_masks_stack(
        stack, params, progress_cb=progress_cb, cancel_cb=cancel_cb
    )


def track_channel(
    masks_tzyx: np.ndarray,
    *,
    max_distance: float,
    max_frame_gap: int,
) -> np.ndarray:
    """Axis-by-axis linking of an in-memory mask stack: stitch z, then track t.

    See :func:`track_laptrack.track_axiswise`. Single-slice input reduces to plain
    time tracking.
    """
    masks = _to_tzyx(masks_tzyx)
    return track_laptrack.track_axiswise(
        masks, max_distance=max_distance, max_frame_gap=max_frame_gap
    )


def preview_channel_masks(
    stack_tzyx: np.ndarray,
    channel: str,
    params,
    t: int,
    z: int,
) -> np.ndarray:
    """Native masks for a single current frame, embedded in a full ``(T, Z, Y, X)``.

    All other frames are background, so the preview overlays exactly the frame
    the user is looking at while keeping the layer's dims aligned with the input.
    """
    stack = _to_tzyx(stack_tzyx)
    out = np.zeros(stack.shape, dtype=np.int32)
    frame = stack[t]  # (Z, Y, X)
    if channel == "nucleus" and getattr(params, "do_3d", False):
        out[t] = native_masks.run_nucleus_masks_frame(frame, z=None, params=params)
    elif channel == "nucleus":
        out[t, z] = native_masks.run_nucleus_masks_frame(frame, z=z, params=params)
    else:
        out[t, z] = native_masks.run_cell_masks_frame(frame, z=z, params=params)
    return out


def segment_track_joint(
    ch1_source,
    ch2_source,
    ch1_params,
    ch2_params,
    flow_params: FlowFollowingParams,
    *,
    max_distance: float,
    max_frame_gap: int,
    progress_cb=None,
    cancel_cb=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load both channels (layout-free) and run the joint anchor path.

    Each ``*_source`` is a ``.tif`` path or an in-memory array (a viewer layer).
    Channel 1 is the anchor (segmented + tracked); Channel 2 is flowed onto it.
    Returns ``(ch1_tracked, ch2_tracked)`` as ``(T, Z, Y, X)`` int32 stacks that
    share label ids (one Channel-2 object per Channel-1 object). The Channel-2
    stack is tracked by inheriting the Channel-1 tracks, so it needs no tracker.
    """
    ch1_stack = _coerce_stack(ch1_source)
    ch2_stack = _coerce_stack(ch2_source)
    return joint_mod.joint_segment_track(
        ch1_stack, ch2_stack, ch1_params, ch2_params, flow_params,
        max_distance=max_distance, max_frame_gap=max_frame_gap,
        progress_cb=progress_cb, cancel_cb=cancel_cb,
    )


def preview_joint(
    ch1_source,
    ch2_source,
    ch1_params,
    ch2_params,
    flow_params: FlowFollowingParams,
    t: int,
    *,
    max_distance: float,
    max_frame_gap: int,
    progress_cb=None,
    cancel_cb=None,
) -> np.ndarray:
    """Joint Channel-2 result for a single current frame, embedded full-size.

    Runs the joint anchor path on just frame ``t`` of both channels (tracking one
    frame is a no-op) and embeds the Channel-2 result into an otherwise-background
    ``(T, Z, Y, X)`` stack — so the preview overlays exactly the frame on screen,
    letting the Channel-2 params be tuned before committing to the whole stack.
    """
    ch1 = _coerce_stack(ch1_source)
    ch2 = _coerce_stack(ch2_source)
    _, ch2_frame = joint_mod.joint_segment_track(
        ch1[t:t + 1], ch2[t:t + 1], ch1_params, ch2_params, flow_params,
        max_distance=max_distance, max_frame_gap=max_frame_gap,
        progress_cb=progress_cb, cancel_cb=cancel_cb,
    )
    out = np.zeros(ch1.shape, dtype=np.int32)
    out[t] = ch2_frame[0]
    return out


def _make_status() -> QLabel:
    lbl = QLabel("")
    lbl.setWordWrap(True)
    lbl.setVisible(False)
    status_label(lbl)
    return lbl


def _make_progress() -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(True)
    bar.setVisible(False)
    return bar


class CellposeSegmentTrackWidget(QWidget):
    """Standalone segment+track: Channel 1 anchors; a Channel 2 makes it joint."""

    _progress_signal = Signal(int, int, str)
    _SETTINGS_APP = "cellflow_cellpose_segment_track"

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.gate = UiGate(self)
        self._ch1_path: Path | None = None
        self._ch2_path: Path | None = None
        # Set when a channel is sourced from a viewer image layer instead of a file.
        # The stack holds the layer's data; the name is kept only for the tooltip.
        self._ch1_stack: np.ndarray | None = None
        self._ch2_stack: np.ndarray | None = None
        self._ch1_layer_name: str | None = None
        self._ch2_layer_name: str | None = None
        self._running: str | None = None  # e.g. "ch1_seg", "ch1_track", "ch2_run"
        self._worker = None
        self._cancel_requested = False

        self._setup_ui()
        self._connect_signals()
        self._register_gate_controls()
        self._progress_signal.connect(self._progress)
        self._load_standalone_settings()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # ── Channel 1 row (anchor: segment + track) ──
        # One row: the "Channel 1" pill, then its source pickers (browse / from
        # layer) and its action buttons (params, preview, segment, track). There is
        # no path text field — a channel's source is picked entirely through its two
        # source pills, and the active one lights up.
        self.ch1_folder_btn, self.ch1_layer_btn = self._make_source_buttons(
            _CH1_LABEL, self._on_browse_ch1, self._on_load_layer_ch1,
        )
        self.ch1_params_btn = _tool_btn("⚙", "Channel 1 parameters.", checkable=True)
        self.ch1_preview_btn = _tool_btn("▷", "Preview Channel 1 on the current frame.")
        self.ch1_seg_btn = _tool_btn("▶", "Segment Channel 1 (native masks).")
        self.ch1_track_btn = _tool_btn("⊳", "Track Channel 1 masks (laptrack).")
        for b in (
            self.ch1_params_btn, self.ch1_preview_btn,
            self.ch1_seg_btn, self.ch1_track_btn,
        ):
            stage_header_action_button(b, "cellpose")
        self.ch1_section = self._build_ch1_params_section()
        self.ch1_section.set_header_visible(False)
        self.ch1_section.collapse()
        self.ch1_params_btn.toggled.connect(
            lambda checked: self.ch1_section._toggle.setChecked(checked)
        )
        ch1_label = self._stage_label("Channel 1")
        ch1_label.setToolTip(
            "Channel 1 — the anchor: segmented and tracked. Typically the nucleus."
        )
        root.addLayout(self._stage_row(
            ch1_label,
            self.ch1_folder_btn, self.ch1_layer_btn,
            self.ch1_params_btn, self.ch1_preview_btn,
            self.ch1_seg_btn, self.ch1_track_btn,
        ))
        root.addWidget(self.ch1_section)

        # ── Channel 2 row (joint — the only mode for a second channel) ──
        # Same single-row shape as Channel 1: the "Channel 2" pill, its source
        # pickers, then its action buttons. A second channel is never segmented on
        # its own — it is always flowed onto the tracked Channel 1 (one object per
        # Channel-1 object, sharing its id). Preview (▷) runs that joint assignment
        # on the current frame so the Channel-2 params can be tuned; Run (▶) commits
        # it over the whole stack. Both need both inputs; Channel 1's own segment +
        # track path is untouched.
        self.ch2_folder_btn, self.ch2_layer_btn = self._make_source_buttons(
            _CH2_LABEL, self._on_browse_ch2, self._on_load_layer_ch2,
        )
        self.ch2_params_btn = _tool_btn("⚙", "Channel 2 parameters.", checkable=True)
        self.ch2_preview_btn = _tool_btn(
            "▷",
            "Preview the joint Channel-2 result on the current frame "
            "(needs both channels).",
        )
        self.ch2_run_btn = _tool_btn(
            "▶",
            "Run joint: segment + track Channel 1, then flow Channel 2 onto it "
            "(needs both channels).",
        )
        for b in (self.ch2_params_btn, self.ch2_preview_btn, self.ch2_run_btn):
            stage_header_action_button(b, "cellpose")
        self.ch2_section = self._build_ch2_params_section()
        self.ch2_section.set_header_visible(False)
        self.ch2_section.collapse()
        self.ch2_params_btn.toggled.connect(
            lambda checked: self.ch2_section._toggle.setChecked(checked)
        )
        ch2_label = self._stage_label("Channel 2")
        ch2_label.setToolTip(
            "Channel 2 — optional. When set, runs joint mode: this channel is "
            "flowed onto Channel 1 (never segmented on its own). Typically the cell."
        )
        root.addLayout(self._stage_row(
            ch2_label,
            self.ch2_folder_btn, self.ch2_layer_btn,
            self.ch2_params_btn, self.ch2_preview_btn, self.ch2_run_btn,
        ))
        root.addWidget(self.ch2_section)

        # ── Cell correction ──
        # Reuse the app's cell corrector — the ultrack/OverlapDB-free one — bound
        # to whatever Labels layer is active, so segment → track → correct is one
        # surface with no on-disk handoff. ``full_editing`` unlocks the complete
        # DB-free toolkit (spawn / erase / merge / swap / split + Q/E retrack) that
        # the app keeps contour-only. The widget brings its own "Correction"
        # header + ⏻ activate button; it edits the active layer in place and the
        # user saves it via napari.
        self.cell_correction = CellCorrectionWidget(
            self.viewer,
            active_labels_layer_provider=self._active_labels_layer,
            full_editing=True,
            parent=self,
        )
        root.addWidget(self.cell_correction)

        # ── Status + progress ──
        self.status_lbl = _make_status()
        root.addWidget(self.status_lbl)
        self.progress_bar = _make_progress()
        root.addWidget(self.progress_bar)

    def _make_source_buttons(self, label, on_browse, on_load_layer):
        """The two source pills for a channel → ``(folder_btn, layer_btn)``.

        There is no path text field: a channel's source is picked entirely through
        these two pills — browse (``…``) for a ``.tif`` file, or load-from-layer
        (``⧉``) for an image already open in the viewer. They sit in the channel's
        own stage row, right after its name pill. Both are checkable so the *active*
        source lights up (and its tooltip names the file/layer); the glyphs match
        the stage buttons' thin geometric family.
        """
        folder_btn = _tool_btn("…", f"Load {label} from a .tif file.", checkable=True)
        layer_btn = _tool_btn(
            "⧉", f"Load {label} from a napari image layer.", checkable=True
        )
        for b in (folder_btn, layer_btn):
            stage_header_action_button(b, "cellpose")
        folder_btn.clicked.connect(on_browse)
        layer_btn.clicked.connect(on_load_layer)
        return folder_btn, layer_btn

    def _build_ch1_params_section(self) -> CollapsibleSection:
        # No input-layout / 3D-mode / anisotropy: segmentation is layout-free and
        # per-plane (the shorter leading axis is treated as z for tracking).
        body = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(8, 4, 4, 4)
        body.setLayout(grid)
        self.ch1_diameter_spin = _dslider(0.0, 500.0, 25.0, 1.0, 1)
        self.ch1_min_size_spin = _islider(0, 100000, 15)
        self.ch1_gamma_spin = _dslider(0.1, 5.0, 1.0, 0.1, 2)
        add_section_pair_row(
            grid, 0, "Diameter:", self.ch1_diameter_spin, "Min size:", self.ch1_min_size_spin
        )
        add_section_pair_row(grid, 1, "Gamma:", self.ch1_gamma_spin)
        return CollapsibleSection("Channel 1 parameters", body, expanded=False)

    def _build_ch2_params_section(self) -> CollapsibleSection:
        # Channel 2 has no independent segmentation, but its Cellpose flow field
        # (diameter / gamma) and the flow-following knobs both shape how its
        # foreground is assigned to Channel-1 objects, so both live here. The
        # tracking knobs live here too: they tune the Channel-1 tracker that the
        # joint anchor (and Channel 1's own Track action) runs.
        body = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(8, 4, 4, 4)
        body.setLayout(grid)
        self.ch2_diameter_spin = _dslider(0.0, 500.0, 0.0, 1.0, 1)
        self.ch2_min_size_spin = _islider(0, 100000, 0)
        self.ch2_gamma_spin = _dslider(0.1, 5.0, 1.0, 0.1, 2)
        self.ch2_fg_thr_spin = _dslider(0.0, 1.0, 0.5, 0.05, 2)
        self.ch2_flow_weight_spin = _dslider(0.0, 1.0, 0.5, 0.05, 2)
        self.ch2_radius_spin = _dslider(1.0, 200.0, 30.0, 1.0, 1)
        self.track_max_dist_spin = _dslider(1.0, 200.0, 15.0, 1.0, 1)
        self.track_gap_spin = _islider(0, 10, 0)
        add_section_pair_row(
            grid, 0, "Diameter:", self.ch2_diameter_spin, "Min size:", self.ch2_min_size_spin
        )
        add_section_pair_row(grid, 1, "Gamma:", self.ch2_gamma_spin)
        add_section_pair_row(
            grid, 2,
            "FG threshold:", self.ch2_fg_thr_spin,
            "Flow weight:", self.ch2_flow_weight_spin,
        )
        add_section_pair_row(grid, 3, "Max assign radius:", self.ch2_radius_spin)
        add_section_pair_row(
            grid, 4,
            "Max distance:", self.track_max_dist_spin,
            "Max frame gap:", self.track_gap_spin,
        )
        return CollapsibleSection("Channel 2 & tracking parameters", body, expanded=False)

    @staticmethod
    def _stage_label(text: str) -> QLabel:
        return stage_header_label(QLabel(text), "cellpose")

    @staticmethod
    def _stage_row(label: QLabel, *trailing: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(label)
        for w in trailing:
            row.addWidget(w)
        row.addStretch(1)
        return row

    # -------------------------------------------------------------- signals
    def _connect_signals(self) -> None:
        self.ch1_preview_btn.clicked.connect(lambda: self._on_ch1("preview"))
        self.ch1_seg_btn.clicked.connect(lambda: self._on_ch1("seg"))
        self.ch1_track_btn.clicked.connect(lambda: self._on_ch1("track"))
        self.ch2_preview_btn.clicked.connect(lambda: self._on_ch2("preview"))
        self.ch2_run_btn.clicked.connect(lambda: self._on_ch2("run"))

    def _on_ch1(self, kind: str) -> None:
        if self._running is not None:
            self._on_cancel()
            return
        if kind == "seg":
            self._run_segment()
        elif kind == "track":
            self._run_track()
        else:
            self._preview()

    def _on_ch2(self, kind: str) -> None:
        if self._running is not None:
            self._on_cancel()
            return
        if kind == "preview":
            self._preview_joint()
        else:
            self._run_joint()

    def _on_cancel(self) -> None:
        self._cancel_requested = True
        worker = self._worker
        if worker is not None and hasattr(worker, "quit"):
            worker.quit()

    # ---------------------------------------------------------- path helpers
    def _on_browse_ch1(self) -> None:
        self._browse_channel(1)

    def _on_browse_ch2(self) -> None:
        self._browse_channel(2)

    def _on_load_layer_ch1(self) -> None:
        self._load_channel_from_layer(1)

    def _on_load_layer_ch2(self) -> None:
        self._load_channel_from_layer(2)

    def _browse_channel(self, which: int) -> None:
        """Pick a ``.tif`` as this channel's source (the browse pill)."""
        label = _CH1_LABEL if which == 1 else _CH2_LABEL
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {label}", filter=_IMAGE_FILTER
        )
        if path:
            self._set_channel_path(which, Path(path))
            self._status(f"{label} ← file '{Path(path).name}'.")
        else:
            # Dialog cancelled — undo the pill's auto-toggle, keep prior source.
            self._refresh_source_buttons(which)

    def _load_channel_from_layer(self, which: int) -> None:
        """Pick a viewer image layer as this channel's source (no file needed)."""
        image_layers = [ly for ly in self.viewer.layers if isinstance(ly, Image)]
        if not image_layers:
            self._status("No image layers in the viewer to load from.")
            self._refresh_source_buttons(which)
            return
        names = [ly.name for ly in image_layers]
        active = getattr(getattr(self.viewer.layers, "selection", None), "active", None)
        default = names.index(active.name) if active in image_layers else 0
        label = _CH1_LABEL if which == 1 else _CH2_LABEL
        name, ok = QInputDialog.getItem(
            self, f"Load {label} from layer", "Image layer:", names, default, False,
        )
        if ok and name:
            self._use_layer_as_channel(which, name)
        else:
            self._refresh_source_buttons(which)

    def _use_layer_as_channel(self, which: int, layer_name: str) -> None:
        """Source a channel from a named viewer layer's data (testable seam)."""
        try:
            layer = self.viewer.layers[layer_name]
            stack = to_canonical_tzyx(np.asarray(layer.data))
        except Exception as exc:
            self._status(f"Cannot use layer '{layer_name}': {exc}")
            self._refresh_source_buttons(which)
            return
        self._set_channel_stack(which, stack, layer_name)
        label = _CH1_LABEL if which == 1 else _CH2_LABEL
        self._status(f"{label} ← layer '{layer_name}'.")

    def _set_channel_path(self, which: int, path: Path | None) -> None:
        """Make a ``.tif`` path this channel's source (clears any layer source).

        ``None`` clears the channel entirely. This is the single mutation point
        for file-sourced channels, so it refreshes the pills, persists and regates.
        """
        if which == 1:
            self._ch1_path = path
            self._ch1_stack = None
            self._ch1_layer_name = None
        else:
            self._ch2_path = path
            self._ch2_stack = None
            self._ch2_layer_name = None
        self._after_source_change(which)

    def _set_channel_stack(self, which: int, stack, layer_name: str | None = None) -> None:
        """Make a viewer layer's data this channel's source (clears any file path)."""
        if which == 1:
            self._ch1_stack = stack
            self._ch1_layer_name = layer_name
            if stack is not None:
                self._ch1_path = None
        else:
            self._ch2_stack = stack
            self._ch2_layer_name = layer_name
            if stack is not None:
                self._ch2_path = None
        self._after_source_change(which)

    def _after_source_change(self, which: int) -> None:
        self._refresh_source_buttons(which)
        self._save_standalone_settings()
        self.gate.recompute()

    def _refresh_source_buttons(self, which: int) -> None:
        """Mirror the channel's active source into its two pills (checked + tooltip)."""
        if which == 1:
            folder_btn, layer_btn = self.ch1_folder_btn, self.ch1_layer_btn
            path, stack, lname = self._ch1_path, self._ch1_stack, self._ch1_layer_name
            label = _CH1_LABEL
        else:
            folder_btn, layer_btn = self.ch2_folder_btn, self.ch2_layer_btn
            path, stack, lname = self._ch2_path, self._ch2_stack, self._ch2_layer_name
            label = _CH2_LABEL
        folder_btn.setChecked(path is not None)
        layer_btn.setChecked(stack is not None)
        folder_btn.setToolTip(
            f"{label} ← file: {path.name}" if path is not None
            else f"Load {label} from a .tif file."
        )
        layer_btn.setToolTip(
            f"{label} ← layer: {lname}" if stack is not None
            else f"Load {label} from a napari image layer."
        )

    def _channel_source(self, which: int):
        """A channel's active input: an in-memory array (layer) or a Path, else None."""
        stack = self._ch1_stack if which == 1 else self._ch2_stack
        if stack is not None:
            return stack
        path = self._ch1_path if which == 1 else self._ch2_path
        if path is not None and path.exists():
            return path
        return None

    def _channel_present(self, which: int) -> bool:
        return self._channel_source(which) is not None

    def _load_standalone_settings(self) -> None:
        # Restore only real, still-present file paths. Layer sources are session-
        # bound (never persisted), so a channel either comes back as its saved
        # .tif or starts empty.
        s = QSettings("cellflow", self._SETTINGS_APP)
        for which, key in ((1, "channel1"), (2, "channel2")):
            value = s.value(key, "", type=str)
            if value and Path(value).exists():
                if which == 1:
                    self._ch1_path = Path(value)
                else:
                    self._ch2_path = Path(value)
            self._refresh_source_buttons(which)
        self.gate.recompute()

    def _save_standalone_settings(self) -> None:
        # Layer-sourced channels are session-bound, so persist only real file paths
        # (a layer would reload as a meaningless, backing-less source).
        s = QSettings("cellflow", self._SETTINGS_APP)
        s.setValue("channel1", str(self._ch1_path) if self._ch1_path else "")
        s.setValue("channel2", str(self._ch2_path) if self._ch2_path else "")

    # ------------------------------------------------------------- params
    def _build_ch1_params(self) -> cellpose_runner.NucleusParams:
        # The anchor uses the nucleus mask routine (clean, separable objects).
        # Standalone segments every plane individually: do_3d is always off
        # (anisotropy then unused) — true-3D segmentation is the app's domain.
        return cellpose_runner.NucleusParams(
            do_3d=False,
            anisotropy=1.0,
            diameter=float(self.ch1_diameter_spin.value()),
            min_size=int(self.ch1_min_size_spin.value()),
            gamma=float(self.ch1_gamma_spin.value()),
        )

    def _build_ch2_params(self) -> cellpose_runner.CellParams:
        return cellpose_runner.CellParams(
            diameter=float(self.ch2_diameter_spin.value()),
            min_size=int(self.ch2_min_size_spin.value()),
            gamma=float(self.ch2_gamma_spin.value()),
        )

    def _build_flow_params(self) -> FlowFollowingParams:
        return FlowFollowingParams(
            fg_threshold=float(self.ch2_fg_thr_spin.value()),
            flow_weight=float(self.ch2_flow_weight_spin.value()),
            max_assign_radius=float(self.ch2_radius_spin.value()),
        )

    def _both_inputs(self) -> bool:
        return self._channel_present(1) and self._channel_present(2)

    # --------------------------------------------------------------- run: seg
    def _run_segment(self) -> None:
        source = self._channel_source(1)
        if source is None:
            self._status("Missing Channel 1 input.")
            return
        params = self._build_ch1_params()
        self._cancel_requested = False
        progress_signal = self._progress_signal
        masks_name = _layer_name(_CH1_LABEL, "masks")

        def _done(result):
            self._worker = None
            self._set_running(None)
            self._clear_progress()
            self._add_labels(masks_name, result)
            self._status(f"Channel 1 masks → '{masks_name}'. Save from the layer.")

        @thread_worker(connect={
            "yielded": self._on_progress, "returned": _done, "errored": self._errored,
        })
        def _worker():
            yield (0, 1, "Loading input...")
            return segment_channel(
                source, "nucleus", params,
                progress_cb=lambda d, t, m: progress_signal.emit(int(d), int(t), str(m)),
                cancel_cb=lambda: self._cancel_requested,
            )

        self._set_running("ch1_seg")
        self._status(
            f"Loading Cellpose-SAM on {cellpose_runner.device_label()}..."
            if not cellpose_runner.is_model_loaded()
            else "Segmenting Channel 1..."
        )
        self._worker = _worker()

    # ------------------------------------------------------------- run: track
    def _run_track(self) -> None:
        masks_name = _layer_name(_CH1_LABEL, "masks")
        if masks_name not in self.viewer.layers:
            self._status(f"No '{masks_name}' layer — segment first.")
            return
        masks = _to_tzyx(np.asarray(self.viewer.layers[masks_name].data))
        max_distance = float(self.track_max_dist_spin.value())
        max_frame_gap = int(self.track_gap_spin.value())
        tracked_name = _layer_name(_CH1_LABEL, "tracked")

        def _done(result):
            self._worker = None
            self._set_running(None)
            self._clear_progress()
            self._add_labels(tracked_name, result)
            self._status(f"Channel 1 tracked → '{tracked_name}'. Save from the layer.")

        @thread_worker(connect={
            "yielded": self._on_progress, "returned": _done, "errored": self._errored,
        })
        def _worker():
            yield (0, 1, "Tracking Channel 1 masks...")
            return track_channel(
                masks, max_distance=max_distance, max_frame_gap=max_frame_gap,
            )

        self._set_running("ch1_track")
        self._status("Tracking Channel 1 masks (laptrack)...")
        self._worker = _worker()

    # --------------------------------------------------------------- run: joint
    def _run_joint(self) -> None:
        if not self._both_inputs():
            self._status("Joint mode needs both Channel 1 and Channel 2 inputs.")
            return
        ch1_source = self._channel_source(1)
        ch2_source = self._channel_source(2)
        ch1_params = self._build_ch1_params()
        ch2_params = self._build_ch2_params()
        flow_params = self._build_flow_params()
        max_distance = float(self.track_max_dist_spin.value())
        max_frame_gap = int(self.track_gap_spin.value())
        self._cancel_requested = False
        progress_signal = self._progress_signal
        ch1_name = _layer_name(_CH1_LABEL, "tracked")
        ch2_name = _layer_name(_CH2_LABEL, "tracked")

        def _done(result):
            ch1_tracked, ch2_tracked = result
            self._worker = None
            self._set_running(None)
            self._clear_progress()
            self._add_labels(ch1_name, ch1_tracked)
            self._add_labels(ch2_name, ch2_tracked)
            self._status(
                f"Joint → '{ch1_name}' + '{ch2_name}' (paired ids). Save from the layers."
            )

        @thread_worker(connect={
            "yielded": self._on_progress, "returned": _done, "errored": self._errored,
        })
        def _worker():
            yield (0, 4, "Loading inputs...")
            return segment_track_joint(
                ch1_source, ch2_source,
                ch1_params, ch2_params, flow_params,
                max_distance=max_distance, max_frame_gap=max_frame_gap,
                progress_cb=lambda d, t, m: progress_signal.emit(int(d), int(t), str(m)),
                cancel_cb=lambda: self._cancel_requested,
            )

        self._set_running("ch2_run")
        self._status(
            f"Loading Cellpose-SAM on {cellpose_runner.device_label()}..."
            if not cellpose_runner.is_model_loaded()
            else "Running joint segmentation..."
        )
        self._worker = _worker()

    # ------------------------------------------------------- preview: joint
    def _preview_joint(self) -> None:
        if not self._both_inputs():
            self._status("Joint preview needs both Channel 1 and Channel 2 inputs.")
            return
        ch1_source = self._channel_source(1)
        ch2_source = self._channel_source(2)
        ch1_params = self._build_ch1_params()
        ch2_params = self._build_ch2_params()
        flow_params = self._build_flow_params()
        max_distance = float(self.track_max_dist_spin.value())
        max_frame_gap = int(self.track_gap_spin.value())
        self._cancel_requested = False
        progress_signal = self._progress_signal
        try:
            ch1_stack = to_canonical_tzyx(
                ch1_source if isinstance(ch1_source, np.ndarray)
                else np.asarray(tifffile.imread(str(ch1_source)))
            )
            ch2_stack = to_canonical_tzyx(
                ch2_source if isinstance(ch2_source, np.ndarray)
                else np.asarray(tifffile.imread(str(ch2_source)))
            )
        except Exception as exc:
            self._status(f"Error: {exc}")
            logger.exception("joint preview load error", exc_info=exc)
            return
        # Show Channel 2 underneath so the previewed assignment is visible.
        self._add_image(_layer_name(_CH2_LABEL, "image"), ch2_stack)
        t, _z = self._current_tz(int(ch1_stack.shape[0]), int(ch1_stack.shape[1]))
        preview_name = _layer_name(_CH2_LABEL, "preview")

        def _done(result):
            self._worker = None
            self._set_running(None)
            self._clear_progress()
            self._add_labels(preview_name, result)
            self._status(
                f"Channel 2 joint preview (frame t={t}) → '{preview_name}'."
            )

        @thread_worker(connect={
            "yielded": self._on_progress, "returned": _done, "errored": self._errored,
        })
        def _worker():
            yield (0, 4, "Previewing joint...")
            return preview_joint(
                ch1_stack, ch2_stack, ch1_params, ch2_params, flow_params, t,
                max_distance=max_distance, max_frame_gap=max_frame_gap,
                progress_cb=lambda d, tot, m: progress_signal.emit(int(d), int(tot), str(m)),
                cancel_cb=lambda: self._cancel_requested,
            )

        self._set_running("ch2_preview")
        self._status(
            f"Loading Cellpose-SAM on {cellpose_runner.device_label()}..."
            if not cellpose_runner.is_model_loaded()
            else "Previewing joint..."
        )
        self._worker = _worker()

    # ----------------------------------------------------------- run: preview
    def _preview(self) -> None:
        source = self._channel_source(1)
        if source is None:
            self._status("Missing Channel 1 input.")
            return
        params = self._build_ch1_params()
        self._cancel_requested = False
        try:
            raw = source if isinstance(source, np.ndarray) else np.asarray(
                tifffile.imread(str(source))
            )
            stack = to_canonical_tzyx(raw)
        except Exception as exc:
            self._status(f"Error: {exc}")
            logger.exception("preview load error", exc_info=exc)
            return
        # Show the input so the previewed frame is visible underneath.
        self._add_image(_layer_name(_CH1_LABEL, "image"), stack)
        t, z = self._current_tz(int(stack.shape[0]), int(stack.shape[1]))
        preview_name = _layer_name(_CH1_LABEL, "preview")
        axes_desc = describe_axes(tuple(int(s) for s in raw.shape))

        def _done(result):
            self._worker = None
            self._set_running(None)
            self._clear_progress()
            self._add_labels(preview_name, result)
            self._status(
                f"Channel 1 preview (frame t={t}; {axes_desc}) → '{preview_name}'."
            )

        @thread_worker(connect={
            "yielded": self._on_progress, "returned": _done, "errored": self._errored,
        })
        def _worker():
            yield (0, 0, f"Previewing Channel 1 on {cellpose_runner.device_label()}...")
            return preview_channel_masks(stack, "nucleus", params, t, z)

        self._set_running("ch1_preview")
        self._status(
            f"Loading Cellpose-SAM on {cellpose_runner.device_label()}..."
            if not cellpose_runner.is_model_loaded()
            else "Previewing Channel 1..."
        )
        self._worker = _worker()

    def _current_tz(self, n_t: int, n_z: int) -> tuple[int, int]:
        """Current (t, z) from the viewer, clamped; z is 0 for single-slice data."""
        step = getattr(getattr(self.viewer, "dims", None), "current_step", (0, 0))
        t = int(step[0]) if len(step) >= 1 else 0
        z = 0 if n_z <= 1 else (int(step[1]) if len(step) >= 2 else 0)
        return min(max(t, 0), n_t - 1), min(max(z, 0), n_z - 1)

    def _active_labels_layer(self):
        """The viewer's active layer iff it is a Labels layer, else None.

        The embedded corrector binds to this; returning None when the active
        layer is not labels is what enforces the "must be a Labels layer" scope.
        """
        layer = getattr(getattr(self.viewer.layers, "selection", None), "active", None)
        return layer if isinstance(layer, Labels) else None

    def _errored(self, exc) -> None:
        self._worker = None
        self._set_running(None)
        self._clear_progress()
        if isinstance(exc, cellpose_runner.CancelledError):
            self._status("Cancelled.")
        else:
            self._status(f"Error: {exc}")
            logger.exception("segment/track error", exc_info=exc)

    # ------------------------------------------------------ layer output
    def _add_labels(self, name: str, data) -> None:
        arr = _squeeze_z(np.asarray(data)).astype(np.int32, copy=False)
        self._show_in_viewer(name, arr, self.viewer.add_labels)

    def _add_image(self, name: str, data) -> None:
        arr = _squeeze_z(np.asarray(data))
        self._show_in_viewer(name, arr, self.viewer.add_image)

    def _show_in_viewer(self, name: str, data, adder) -> None:
        if name in self.viewer.layers:
            try:
                self.viewer.layers[name].data = data
                return
            except Exception:
                self.viewer.layers.remove(self.viewer.layers[name])
        adder(data, name=name)

    # ---------------------------------------------------------- public API
    def get_state(self) -> dict:
        return {
            "channel1": {
                "diameter": self.ch1_diameter_spin.value(),
                "min_size": self.ch1_min_size_spin.value(),
                "gamma": self.ch1_gamma_spin.value(),
            },
            "channel2": {
                "diameter": self.ch2_diameter_spin.value(),
                "min_size": self.ch2_min_size_spin.value(),
                "gamma": self.ch2_gamma_spin.value(),
                "fg_threshold": self.ch2_fg_thr_spin.value(),
                "flow_weight": self.ch2_flow_weight_spin.value(),
                "max_assign_radius": self.ch2_radius_spin.value(),
            },
            "tracking": {
                "max_distance": self.track_max_dist_spin.value(),
                "max_frame_gap": self.track_gap_spin.value(),
            },
        }

    # -------------------------------------------------------- state helpers
    def _status(self, msg: str) -> None:
        self.status_lbl.setText(msg)
        self.status_lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    def _progress(self, done: int, total: int, msg: str) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self._status(msg)

    def _on_progress(self, data) -> None:
        if isinstance(data, tuple):
            self._progress(*data)
        else:
            self._status(str(data))

    def _clear_progress(self) -> None:
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

    def _action_buttons(self):
        return (self.ch1_preview_btn, self.ch1_seg_btn, self.ch1_track_btn)

    def _joint_buttons(self):
        return (self.ch2_preview_btn, self.ch2_run_btn)

    def _register_gate_controls(self) -> None:
        g = self.gate
        g.register(self.ch1_params_btn, ControlClass.HARMLESS)
        g.register(self.ch2_params_btn, ControlClass.HARMLESS)
        for btn in self._action_buttons():
            own = lambda b=btn: self._running is None or self._active_btn() is b
            g.register(btn, ControlClass.RUN_VIEWER, when=own)
        # Channel 2's actions (preview + run) are joint-only: both require both
        # inputs to be present.
        for btn in self._joint_buttons():
            g.register(
                btn, ControlClass.RUN_VIEWER,
                when=lambda b=btn: (
                    (self._running is None or self._active_btn() is b)
                    and self._both_inputs()
                ),
            )
        g.recompute()

    def _btn_for_key(self):
        return {
            "ch1_preview": self.ch1_preview_btn,
            "ch1_seg": self.ch1_seg_btn,
            "ch1_track": self.ch1_track_btn,
            "ch2_preview": self.ch2_preview_btn,
            "ch2_run": self.ch2_run_btn,
        }

    def _active_btn(self):
        return self._btn_for_key().get(self._running)

    _DEFAULT_GLYPHS = {
        "ch1_preview": "▷", "ch1_seg": "▶", "ch1_track": "⊳",
        "ch2_preview": "▷", "ch2_run": "▶",
    }

    def _set_running(self, key: str | None) -> None:
        # restore all glyphs first
        for k, btn in self._btn_for_key().items():
            btn.setText(self._DEFAULT_GLYPHS[k])
        self._running = key
        if key is None:
            self._cancel_requested = False
        else:
            btn = self._active_btn()
            if btn is not None:
                btn.setText("✕")
                btn.setToolTip("Cancel.")
        self.gate.recompute()


def make_cellpose_segment_track_widget(napari_viewer=None):
    """napari plugin factory for the standalone Cellpose segment+track tool."""
    try:
        from cellflow.napari._napari_compat import patch_napari_layer_delegate

        patch_napari_layer_delegate()
    except Exception:
        pass
    if napari_viewer is None:
        napari_viewer = napari.current_viewer()
    return CellposeSegmentTrackWidget(napari_viewer)
