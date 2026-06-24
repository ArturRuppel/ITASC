"""Standalone Cellpose *segment + track* widget for the ``cellflow-cellpose`` tool.

This is the independently-shipped distribution's own surface — distinct from the
app's :class:`~cellflow.napari.cellpose_widget.CellposeWidget`, which stays
untouched and keeps emitting divergence maps for the integrated pipeline. Here
the distro is repurposed into a self-contained product: pick raw nucleus/cell
stacks, run Cellpose **native masks** (:mod:`cellflow.cellpose.native_masks`),
then link them across time with **laptrack** (:mod:`cellflow.cellpose.track_laptrack`).

Input is by explicit file picker — point directly at a multi-dimensional ``.tif``
per channel. **There is no output directory**: every result is added straight to
the napari viewer as a layer (tagged ``[Nucleus]`` / ``[Cell]``), and the user
saves whichever layers they want via napari's own *Save Selected Layers*. The
embedded basic corrector edits whichever Labels layer is active.
"""
from __future__ import annotations

import logging
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.layers import Labels
from napari.qt.threading import thread_worker
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._standalone_paths import StandalonePathsMixin
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


def _layer_name(channel: str, kind: str) -> str:
    """Channel-tagged layer name, e.g. ``[Nucleus] masks`` / ``[Cell] tracked``."""
    return f"[{channel.title()}] {kind}"


# ---------------------------------------------------------------------------
# Qt-free compute steps (callable directly in tests; the worker just wraps them)
# ---------------------------------------------------------------------------
def segment_channel(
    in_path: Path,
    channel: str,
    params,
    *,
    progress_cb=None,
    cancel_cb=None,
) -> np.ndarray:
    """Load a raw stack and return per-plane native masks ``(T, Z, Y, X)``.

    The input is canonicalised layout-free (:func:`to_canonical_tzyx`) — no
    2D/2D+t/3D/3D+t declaration — and every plane is segmented individually.
    """
    stack = to_canonical_tzyx(np.asarray(tifffile.imread(str(in_path))))
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
    nuc_path: Path,
    cell_path: Path,
    nuc_params,
    cell_params,
    flow_params: FlowFollowingParams,
    *,
    max_distance: float,
    max_frame_gap: int,
    progress_cb=None,
    cancel_cb=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load both channels (layout-free) and run the joint nucleus-anchored path.

    Returns ``(nucleus_tracked, cell_tracked)`` as ``(T, Z, Y, X)`` int32 stacks
    that share label ids (one cell per nucleus). The cell stack is tracked by
    inheriting the nucleus tracks, so it needs no separate tracker.
    """
    nuc_stack = to_canonical_tzyx(np.asarray(tifffile.imread(str(nuc_path))))
    cell_stack = to_canonical_tzyx(np.asarray(tifffile.imread(str(cell_path))))
    return joint_mod.joint_segment_track(
        nuc_stack, cell_stack, nuc_params, cell_params, flow_params,
        max_distance=max_distance, max_frame_gap=max_frame_gap,
        progress_cb=progress_cb, cancel_cb=cancel_cb,
    )


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


class CellposeSegmentTrackWidget(StandalonePathsMixin, QWidget):
    """Standalone segment+track: two channel rows, explicit file pickers, layers out."""

    _progress_signal = Signal(int, int, str)
    _SETTINGS_APP = "cellflow_cellpose_segment_track"

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.gate = UiGate(self)
        self._nucleus_path: Path | None = None
        self._cell_path: Path | None = None
        self._running: str | None = None  # e.g. "nucleus_seg", "cell_track"
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

        # ── Explicit input pickers (no output dir — results are layers) ──
        paths_col = QVBoxLayout()
        paths_col.setContentsMargins(0, 0, 0, 0)
        self._nucleus_edit = self._add_path_row(
            paths_col, "Nucleus channel", "raw nucleus stack (.tif)",
            self._on_browse_nucleus, self._apply_paths,
        )
        self._cell_edit = self._add_path_row(
            paths_col, "Cell channel", "raw cell stack (.tif)",
            self._on_browse_cell, self._apply_paths,
        )
        root.addLayout(paths_col)

        # ── Nucleus row ──
        self.nucleus_params_btn = _tool_btn("⚙", "Nucleus parameters.", checkable=True)
        self.nucleus_preview_btn = _tool_btn("▷", "Preview nucleus on the current frame.")
        self.nucleus_seg_btn = _tool_btn("▶", "Segment nucleus (native masks).")
        self.nucleus_track_btn = _tool_btn("⊳", "Track nucleus masks (laptrack).")
        for b in (
            self.nucleus_params_btn, self.nucleus_preview_btn,
            self.nucleus_seg_btn, self.nucleus_track_btn,
        ):
            stage_header_action_button(b, "cellpose")
        self.nucleus_section = self._build_nucleus_params_section()
        self.nucleus_section.set_header_visible(False)
        self.nucleus_section.collapse()
        self.nucleus_params_btn.toggled.connect(
            lambda checked: self.nucleus_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Nucleus"),
            self.nucleus_params_btn, self.nucleus_preview_btn,
            self.nucleus_seg_btn, self.nucleus_track_btn,
        ))
        root.addWidget(self.nucleus_section)

        # ── Cell row ──
        self.cell_params_btn = _tool_btn("⚙", "Cell parameters.", checkable=True)
        self.cell_preview_btn = _tool_btn("▷", "Preview cell on the current frame/z-slice.")
        self.cell_seg_btn = _tool_btn("▶", "Segment cell (native masks).")
        self.cell_track_btn = _tool_btn("⊳", "Track cell masks (laptrack).")
        for b in (
            self.cell_params_btn, self.cell_preview_btn,
            self.cell_seg_btn, self.cell_track_btn,
        ):
            stage_header_action_button(b, "cellpose")
        self.cell_section = self._build_cell_params_section()
        self.cell_section.set_header_visible(False)
        self.cell_section.collapse()
        self.cell_params_btn.toggled.connect(
            lambda checked: self.cell_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Cell"),
            self.cell_params_btn, self.cell_preview_btn,
            self.cell_seg_btn, self.cell_track_btn,
        ))
        root.addWidget(self.cell_section)

        # ── Tracking params (shared) ──
        self.tracking_section = self._build_tracking_params_section()
        root.addWidget(self.tracking_section)

        # ── Joint row (nucleus-anchored cell; needs both inputs) ──
        # When both channels are present, the cell is segmented by flowing its
        # foreground along the Cellpose flow field onto the tracked nuclei (one
        # cell per nucleus, sharing its id). This is an explicit action — the
        # single-channel native-masks + laptrack path above is left untouched.
        self.joint_params_btn = _tool_btn("⚙", "Joint parameters.", checkable=True)
        self.joint_run_btn = _tool_btn(
            "⧉", "Joint nucleus-anchored cell segmentation + tracking (needs both inputs)."
        )
        for b in (self.joint_params_btn, self.joint_run_btn):
            stage_header_action_button(b, "cellpose")
        self.joint_section = self._build_joint_params_section()
        self.joint_section.set_header_visible(False)
        self.joint_section.collapse()
        self.joint_params_btn.toggled.connect(
            lambda checked: self.joint_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Joint"), self.joint_params_btn, self.joint_run_btn,
        ))
        root.addWidget(self.joint_section)

        # ── Cell correction ──
        # Reuse the app's *basic* cell corrector — the ultrack/OverlapDB-free one
        # — bound to whatever Labels layer is active, so segment → track → correct
        # is one surface with no on-disk handoff. The widget brings its own
        # "Correction" header + ⏻ activate button; it edits the active layer in
        # place and the user saves it via napari. Nucleus correction (the
        # candidate-DB workflow) is intentionally not shipped in this distro.
        self.cell_correction = CellCorrectionWidget(
            self.viewer,
            active_labels_layer_provider=self._active_labels_layer,
            parent=self,
        )
        root.addWidget(self.cell_correction)

        # ── Status + progress ──
        self.status_lbl = _make_status()
        root.addWidget(self.status_lbl)
        self.progress_bar = _make_progress()
        root.addWidget(self.progress_bar)

    def _build_nucleus_params_section(self) -> CollapsibleSection:
        # No input-layout / 3D-mode / anisotropy: segmentation is layout-free and
        # per-plane (the shorter leading axis is treated as z for tracking).
        body = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(8, 4, 4, 4)
        body.setLayout(grid)
        self.nuc_diameter_spin = _dslider(0.0, 500.0, 25.0, 1.0, 1)
        self.nuc_min_size_spin = _islider(0, 100000, 15)
        self.nuc_gamma_spin = _dslider(0.1, 5.0, 1.0, 0.1, 2)
        add_section_pair_row(
            grid, 0, "Diameter:", self.nuc_diameter_spin, "Min size:", self.nuc_min_size_spin
        )
        add_section_pair_row(grid, 1, "Gamma:", self.nuc_gamma_spin)
        return CollapsibleSection("Nucleus parameters", body, expanded=False)

    def _build_cell_params_section(self) -> CollapsibleSection:
        body = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(8, 4, 4, 4)
        body.setLayout(grid)
        self.cell_diameter_spin = _dslider(0.0, 500.0, 0.0, 1.0, 1)
        self.cell_min_size_spin = _islider(0, 100000, 0)
        self.cell_gamma_spin = _dslider(0.1, 5.0, 1.0, 0.1, 2)
        add_section_pair_row(
            grid, 0, "Diameter:", self.cell_diameter_spin, "Min size:", self.cell_min_size_spin
        )
        add_section_pair_row(grid, 1, "Gamma:", self.cell_gamma_spin)
        return CollapsibleSection("Cell parameters", body, expanded=False)

    def _build_joint_params_section(self) -> CollapsibleSection:
        body = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(8, 4, 4, 4)
        body.setLayout(grid)
        self.joint_fg_thr_spin = _dslider(0.0, 1.0, 0.5, 0.05, 2)
        self.joint_flow_weight_spin = _dslider(0.0, 1.0, 0.5, 0.05, 2)
        self.joint_radius_spin = _dslider(1.0, 200.0, 30.0, 1.0, 1)
        row = 0
        add_section_pair_row(
            grid, row,
            "FG threshold:", self.joint_fg_thr_spin,
            "Flow weight:", self.joint_flow_weight_spin,
        ); row += 1
        add_section_pair_row(grid, row, "Max assign radius:", self.joint_radius_spin)
        return CollapsibleSection("Joint parameters", body, expanded=False)

    def _build_tracking_params_section(self) -> CollapsibleSection:
        body = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(8, 4, 4, 4)
        body.setLayout(grid)
        self.track_max_dist_spin = _dslider(1.0, 200.0, 15.0, 1.0, 1)
        self.track_gap_spin = _islider(0, 10, 0)
        add_section_pair_row(
            grid, 0,
            "Max distance:", self.track_max_dist_spin,
            "Max frame gap:", self.track_gap_spin,
        )
        return CollapsibleSection("Tracking parameters", body, expanded=False)

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
        self.nucleus_preview_btn.clicked.connect(lambda: self._on_action("nucleus", "preview"))
        self.cell_preview_btn.clicked.connect(lambda: self._on_action("cell", "preview"))
        self.nucleus_seg_btn.clicked.connect(lambda: self._on_action("nucleus", "seg"))
        self.cell_seg_btn.clicked.connect(lambda: self._on_action("cell", "seg"))
        self.nucleus_track_btn.clicked.connect(lambda: self._on_action("nucleus", "track"))
        self.cell_track_btn.clicked.connect(lambda: self._on_action("cell", "track"))
        self.joint_run_btn.clicked.connect(self._on_joint_action)

    def _on_action(self, channel: str, kind: str) -> None:
        if self._running is not None:
            self._on_cancel()
            return
        if kind == "seg":
            self._run_segment(channel)
        elif kind == "track":
            self._run_track(channel)
        else:
            self._preview_channel(channel)

    def _on_joint_action(self) -> None:
        if self._running is not None:
            self._on_cancel()
            return
        self._run_joint()

    def _on_cancel(self) -> None:
        self._cancel_requested = True
        worker = self._worker
        if worker is not None and hasattr(worker, "quit"):
            worker.quit()

    # ---------------------------------------------------------- path helpers
    def _apply_paths(self) -> None:
        nuc = self._nucleus_edit.text().strip()
        cel = self._cell_edit.text().strip()
        self._nucleus_path = Path(nuc) if nuc else None
        self._cell_path = Path(cel) if cel else None
        self._save_standalone_settings()
        self.gate.recompute()

    def _on_browse_nucleus(self) -> None:
        self._browse_file_into(self._nucleus_edit, "Select nucleus channel", self._apply_paths)

    def _on_browse_cell(self) -> None:
        self._browse_file_into(self._cell_edit, "Select cell channel", self._apply_paths)

    def _standalone_fields(self) -> dict:
        return {
            "nucleus": self._nucleus_edit,
            "cell": self._cell_edit,
        }

    def _load_standalone_settings(self) -> None:
        self._load_path_settings(self._SETTINGS_APP, self._standalone_fields())
        if any(e.text().strip() for e in self._standalone_fields().values()):
            self._apply_paths()

    def _save_standalone_settings(self) -> None:
        self._save_path_settings(self._SETTINGS_APP, self._standalone_fields())

    # ------------------------------------------------------------- params
    def _build_nucleus_params(self) -> cellpose_runner.NucleusParams:
        # Standalone segments every plane individually: do_3d is always off
        # (anisotropy is then unused), so true-3D nucleus segmentation is the
        # app's domain, not the standalone's.
        return cellpose_runner.NucleusParams(
            do_3d=False,
            anisotropy=1.0,
            diameter=float(self.nuc_diameter_spin.value()),
            min_size=int(self.nuc_min_size_spin.value()),
            gamma=float(self.nuc_gamma_spin.value()),
        )

    def _build_cell_params(self) -> cellpose_runner.CellParams:
        return cellpose_runner.CellParams(
            diameter=float(self.cell_diameter_spin.value()),
            min_size=int(self.cell_min_size_spin.value()),
            gamma=float(self.cell_gamma_spin.value()),
        )

    def _build_flow_params(self) -> FlowFollowingParams:
        return FlowFollowingParams(
            fg_threshold=float(self.joint_fg_thr_spin.value()),
            flow_weight=float(self.joint_flow_weight_spin.value()),
            max_assign_radius=float(self.joint_radius_spin.value()),
        )

    def _both_inputs(self) -> bool:
        return (
            self._nucleus_path is not None and self._nucleus_path.exists()
            and self._cell_path is not None and self._cell_path.exists()
        )

    def _channel_input(self, channel: str) -> Path | None:
        return self._nucleus_path if channel == "nucleus" else self._cell_path

    def _channel_params(self, channel: str):
        return (
            self._build_nucleus_params() if channel == "nucleus"
            else self._build_cell_params()
        )

    # --------------------------------------------------------------- run: seg
    def _run_segment(self, channel: str) -> None:
        in_path = self._channel_input(channel)
        if in_path is None or not in_path.exists():
            self._status(f"Missing input for {channel}.")
            return
        params = self._channel_params(channel)
        self._cancel_requested = False
        progress_signal = self._progress_signal
        masks_name = _layer_name(channel, "masks")

        def _done(result):
            self._worker = None
            self._set_running(None)
            self._clear_progress()
            self._add_labels(masks_name, result)
            self._status(f"{channel.title()} masks → '{masks_name}'. Save from the layer.")

        @thread_worker(connect={
            "yielded": self._on_progress, "returned": _done, "errored": self._errored,
        })
        def _worker():
            yield (0, 1, "Loading input...")
            return segment_channel(
                in_path, channel, params,
                progress_cb=lambda d, t, m: progress_signal.emit(int(d), int(t), str(m)),
                cancel_cb=lambda: self._cancel_requested,
            )

        self._set_running(f"{channel}_seg")
        self._status(
            f"Loading Cellpose-SAM on {cellpose_runner.device_label()}..."
            if not cellpose_runner.is_model_loaded()
            else f"Segmenting {channel}..."
        )
        self._worker = _worker()

    # ------------------------------------------------------------- run: track
    def _run_track(self, channel: str) -> None:
        masks_name = _layer_name(channel, "masks")
        if masks_name not in self.viewer.layers:
            self._status(f"No '{masks_name}' layer — segment first.")
            return
        masks = _to_tzyx(np.asarray(self.viewer.layers[masks_name].data))
        max_distance = float(self.track_max_dist_spin.value())
        max_frame_gap = int(self.track_gap_spin.value())
        tracked_name = _layer_name(channel, "tracked")

        def _done(result):
            self._worker = None
            self._set_running(None)
            self._clear_progress()
            self._add_labels(tracked_name, result)
            self._status(f"{channel.title()} tracked → '{tracked_name}'. Save from the layer.")

        @thread_worker(connect={
            "yielded": self._on_progress, "returned": _done, "errored": self._errored,
        })
        def _worker():
            yield (0, 1, f"Tracking {channel} masks...")
            return track_channel(
                masks, max_distance=max_distance, max_frame_gap=max_frame_gap,
            )

        self._set_running(f"{channel}_track")
        self._status(f"Tracking {channel} masks (laptrack)...")
        self._worker = _worker()

    # --------------------------------------------------------------- run: joint
    def _run_joint(self) -> None:
        if not self._both_inputs():
            self._status("Joint mode needs both a nucleus and a cell input.")
            return
        nuc_path = self._nucleus_path
        cell_path = self._cell_path
        nuc_params = self._build_nucleus_params()
        cell_params = self._build_cell_params()
        flow_params = self._build_flow_params()
        max_distance = float(self.track_max_dist_spin.value())
        max_frame_gap = int(self.track_gap_spin.value())
        self._cancel_requested = False
        progress_signal = self._progress_signal
        nuc_name = _layer_name("nucleus", "tracked")
        cell_name = _layer_name("cell", "tracked")

        def _done(result):
            nuc_tracked, cell_tracked = result
            self._worker = None
            self._set_running(None)
            self._clear_progress()
            self._add_labels(nuc_name, nuc_tracked)
            self._add_labels(cell_name, cell_tracked)
            self._status(
                f"Joint → '{nuc_name}' + '{cell_name}' (paired ids). Save from the layers."
            )

        @thread_worker(connect={
            "yielded": self._on_progress, "returned": _done, "errored": self._errored,
        })
        def _worker():
            yield (0, 4, "Loading inputs...")
            return segment_track_joint(
                nuc_path, cell_path,
                nuc_params, cell_params, flow_params,
                max_distance=max_distance, max_frame_gap=max_frame_gap,
                progress_cb=lambda d, t, m: progress_signal.emit(int(d), int(t), str(m)),
                cancel_cb=lambda: self._cancel_requested,
            )

        self._set_running("joint")
        self._status(
            f"Loading Cellpose-SAM on {cellpose_runner.device_label()}..."
            if not cellpose_runner.is_model_loaded()
            else "Running joint segmentation..."
        )
        self._worker = _worker()

    # ----------------------------------------------------------- run: preview
    def _preview_channel(self, channel: str) -> None:
        in_path = self._channel_input(channel)
        if in_path is None or not in_path.exists():
            self._status(f"Missing input for {channel}.")
            return
        params = self._channel_params(channel)
        self._cancel_requested = False
        progress_signal = self._progress_signal
        try:
            raw = np.asarray(tifffile.imread(str(in_path)))
            stack = to_canonical_tzyx(raw)
        except Exception as exc:
            self._status(f"Error: {exc}")
            logger.exception("preview load error", exc_info=exc)
            return
        # Show the input so the previewed frame is visible underneath.
        self._add_image(_layer_name(channel, "image"), stack)
        t, z = self._current_tz(int(stack.shape[0]), int(stack.shape[1]))
        preview_name = _layer_name(channel, "preview")
        axes_desc = describe_axes(tuple(int(s) for s in raw.shape))

        def _done(result):
            self._worker = None
            self._set_running(None)
            self._clear_progress()
            self._add_labels(preview_name, result)
            self._status(
                f"{channel.title()} preview (frame t={t}; {axes_desc}) → '{preview_name}'."
            )

        @thread_worker(connect={
            "yielded": self._on_progress, "returned": _done, "errored": self._errored,
        })
        def _worker():
            yield (0, 0, f"Previewing {channel} on {cellpose_runner.device_label()}...")
            return preview_channel_masks(stack, channel, params, t, z)

        self._set_running(f"{channel}_preview")
        self._status(
            f"Loading Cellpose-SAM on {cellpose_runner.device_label()}..."
            if not cellpose_runner.is_model_loaded()
            else f"Previewing {channel}..."
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
            "nucleus": {
                "diameter": self.nuc_diameter_spin.value(),
                "min_size": self.nuc_min_size_spin.value(),
                "gamma": self.nuc_gamma_spin.value(),
            },
            "cell": {
                "diameter": self.cell_diameter_spin.value(),
                "min_size": self.cell_min_size_spin.value(),
                "gamma": self.cell_gamma_spin.value(),
            },
            "tracking": {
                "max_distance": self.track_max_dist_spin.value(),
                "max_frame_gap": self.track_gap_spin.value(),
            },
            "joint": {
                "fg_threshold": self.joint_fg_thr_spin.value(),
                "flow_weight": self.joint_flow_weight_spin.value(),
                "max_assign_radius": self.joint_radius_spin.value(),
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
        return (
            self.nucleus_preview_btn, self.nucleus_seg_btn, self.nucleus_track_btn,
            self.cell_preview_btn, self.cell_seg_btn, self.cell_track_btn,
        )

    def _register_gate_controls(self) -> None:
        g = self.gate
        g.register(self.nucleus_params_btn, ControlClass.HARMLESS)
        g.register(self.cell_params_btn, ControlClass.HARMLESS)
        g.register(self.joint_params_btn, ControlClass.HARMLESS)
        for btn in self._action_buttons():
            own = lambda b=btn: self._running is None or self._active_btn() is b
            g.register(btn, ControlClass.RUN_VIEWER, when=own)
        # Joint also requires both inputs to be present.
        g.register(
            self.joint_run_btn, ControlClass.RUN_VIEWER,
            when=lambda: (
                (self._running is None or self._active_btn() is self.joint_run_btn)
                and self._both_inputs()
            ),
        )
        g.recompute()

    def _btn_for_key(self):
        return {
            "nucleus_preview": self.nucleus_preview_btn,
            "nucleus_seg": self.nucleus_seg_btn,
            "nucleus_track": self.nucleus_track_btn,
            "cell_preview": self.cell_preview_btn,
            "cell_seg": self.cell_seg_btn,
            "cell_track": self.cell_track_btn,
            "joint": self.joint_run_btn,
        }

    def _active_btn(self):
        return self._btn_for_key().get(self._running)

    _DEFAULT_GLYPHS = {
        "nucleus_preview": "▷", "nucleus_seg": "▶", "nucleus_track": "⊳",
        "cell_preview": "▷", "cell_seg": "▶", "cell_track": "⊳",
        "joint": "⧉",
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
