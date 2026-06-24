"""Standalone Cellpose *segment + track* widget for the ``cellflow-cellpose`` tool.

This is the independently-shipped distribution's own surface — distinct from the
app's :class:`~cellflow.napari.cellpose_widget.CellposeWidget`, which stays
untouched and keeps emitting divergence maps for the integrated pipeline. Here
the distro is repurposed into a self-contained product: pick raw nucleus/cell
stacks, run Cellpose **native masks** (:mod:`cellflow.cellpose.native_masks`),
then link them across time with **laptrack** (:mod:`cellflow.cellpose.track_laptrack`).

Input is by explicit file picker — point directly at a multi-dimensional ``.tif``
per channel and a flat output directory; there is no ``0_input/1_cellpose``
layout to honour. Outputs are written flat as ``{channel}_masks.tif`` and
``{channel}_tracked.tif``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.core.tiff import imwrite_grayscale
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
from cellflow.cellpose import cellpose_runner, native_masks, track_laptrack

logger = logging.getLogger(__name__)

_LAYOUT_OPTIONS = ["2D", "2D+t", "3D", "3D+t"]
_DEFAULT_LAYOUT = "3D+t"


# ---------------------------------------------------------------------------
# Qt-free compute steps (callable directly in tests; the worker just wraps them)
# ---------------------------------------------------------------------------
def _normalize_tzyx_labels(arr: np.ndarray) -> np.ndarray:
    """Left-pad a read-back label array to canonical ``(T, Z, Y, X)``."""
    arr = np.asarray(arr)
    if arr.ndim > 4:
        raise ValueError(f"label stack must be <=4-D, got shape {arr.shape}")
    while arr.ndim < 4:
        arr = arr[np.newaxis]
    return arr


def segment_channel(
    in_path: Path,
    out_dir: Path,
    channel: str,
    params,
    layout: str,
    *,
    progress_cb=None,
    cancel_cb=None,
) -> Path:
    """Load a raw stack, run native masks for ``channel``, write ``{channel}_masks.tif``."""
    stack = cellpose_runner.to_tzyx(
        np.asarray(tifffile.imread(str(in_path))), layout
    )
    if channel == "nucleus":
        masks = native_masks.run_nucleus_masks_stack(
            stack, params, progress_cb=progress_cb, cancel_cb=cancel_cb
        )
    else:
        masks = native_masks.run_cell_masks_stack(
            stack, params, progress_cb=progress_cb, cancel_cb=cancel_cb
        )
    return native_masks.write_masks(masks, out_dir, channel)


def track_channel(
    masks_path: Path,
    out_dir: Path,
    channel: str,
    *,
    max_distance: float,
    max_frame_gap: int,
) -> Path:
    """Link ``{channel}_masks.tif`` across time, write ``{channel}_tracked.tif``."""
    masks = _normalize_tzyx_labels(tifffile.imread(str(masks_path)))
    tracked = track_laptrack.track_masks(
        masks, max_distance=max_distance, max_frame_gap=max_frame_gap
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{channel}_tracked.tif"
    imwrite_grayscale(
        path, tracked.astype(np.int32),
        compression="zlib", metadata={"axes": "TZYX"},
    )
    return path


def _layout_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItems(_LAYOUT_OPTIONS)
    combo.setCurrentText(_DEFAULT_LAYOUT)
    return combo


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
    """Standalone segment+track: two channel rows, explicit file pickers."""

    _progress_signal = Signal(int, int, str)
    _SETTINGS_APP = "cellflow_cellpose_segment_track"

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.gate = UiGate(self)
        self._nucleus_path: Path | None = None
        self._cell_path: Path | None = None
        self._output_dir: Path | None = None
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

        # ── Explicit input/output pickers (always shown — standalone only) ──
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
        self._output_dir_edit = self._add_path_row(
            paths_col, "Output dir", "directory for masks + tracked labels",
            self._on_browse_output_dir, self._apply_paths,
        )
        root.addLayout(paths_col)

        # ── Nucleus row ──
        self.nucleus_params_btn = _tool_btn("⚙", "Nucleus parameters.", checkable=True)
        self.nucleus_seg_btn = _tool_btn("▶", "Segment nucleus (native masks).")
        self.nucleus_track_btn = _tool_btn("⊳", "Track nucleus masks (laptrack).")
        for b in (self.nucleus_params_btn, self.nucleus_seg_btn, self.nucleus_track_btn):
            stage_header_action_button(b, "cellpose")
        self.nucleus_section = self._build_nucleus_params_section()
        self.nucleus_section.set_header_visible(False)
        self.nucleus_section.collapse()
        self.nucleus_params_btn.toggled.connect(
            lambda checked: self.nucleus_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Nucleus"),
            self.nucleus_params_btn, self.nucleus_seg_btn, self.nucleus_track_btn,
        ))
        root.addWidget(self.nucleus_section)

        # ── Cell row ──
        self.cell_params_btn = _tool_btn("⚙", "Cell parameters.", checkable=True)
        self.cell_seg_btn = _tool_btn("▶", "Segment cell (native masks).")
        self.cell_track_btn = _tool_btn("⊳", "Track cell masks (laptrack).")
        for b in (self.cell_params_btn, self.cell_seg_btn, self.cell_track_btn):
            stage_header_action_button(b, "cellpose")
        self.cell_section = self._build_cell_params_section()
        self.cell_section.set_header_visible(False)
        self.cell_section.collapse()
        self.cell_params_btn.toggled.connect(
            lambda checked: self.cell_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Cell"),
            self.cell_params_btn, self.cell_seg_btn, self.cell_track_btn,
        ))
        root.addWidget(self.cell_section)

        # ── Tracking params (shared) ──
        self.tracking_section = self._build_tracking_params_section()
        root.addWidget(self.tracking_section)

        # ── Status + progress ──
        self.status_lbl = _make_status()
        root.addWidget(self.status_lbl)
        self.progress_bar = _make_progress()
        root.addWidget(self.progress_bar)

    def _build_nucleus_params_section(self) -> CollapsibleSection:
        body = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(8, 4, 4, 4)
        body.setLayout(grid)
        self.nuc_layout_combo = _layout_combo()
        self.nuc_3d_chk = QCheckBox("3D mode")
        self.nuc_3d_chk.setChecked(True)
        self.nuc_anisotropy_spin = _dslider(0.1, 20.0, 1.5, 0.1, 2)
        self.nuc_diameter_spin = _dslider(0.0, 500.0, 25.0, 1.0, 1)
        self.nuc_min_size_spin = _islider(0, 100000, 15)
        self.nuc_gamma_spin = _dslider(0.1, 5.0, 1.0, 0.1, 2)
        row = 0
        add_section_pair_row(grid, row, "Input layout:", self.nuc_layout_combo); row += 1
        add_section_pair_row(
            grid, row, "3D mode:", self.nuc_3d_chk, "Anisotropy:", self.nuc_anisotropy_spin
        ); row += 1
        add_section_pair_row(
            grid, row, "Diameter:", self.nuc_diameter_spin, "Min size:", self.nuc_min_size_spin
        ); row += 1
        add_section_pair_row(grid, row, "Gamma:", self.nuc_gamma_spin)
        self.nuc_layout_combo.currentTextChanged.connect(self._sync_nucleus_3d_enabled)
        self._sync_nucleus_3d_enabled(self.nuc_layout_combo.currentText())
        return CollapsibleSection("Nucleus parameters", body, expanded=False)

    def _sync_nucleus_3d_enabled(self, layout: str) -> None:
        has_z = cellpose_runner.layout_has_z(layout)
        self.nuc_3d_chk.setEnabled(has_z)
        self.nuc_anisotropy_spin.setEnabled(has_z)

    def _build_cell_params_section(self) -> CollapsibleSection:
        body = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(8, 4, 4, 4)
        body.setLayout(grid)
        self.cell_layout_combo = _layout_combo()
        self.cell_diameter_spin = _dslider(0.0, 500.0, 0.0, 1.0, 1)
        self.cell_min_size_spin = _islider(0, 100000, 0)
        self.cell_gamma_spin = _dslider(0.1, 5.0, 1.0, 0.1, 2)
        row = 0
        add_section_pair_row(grid, row, "Input layout:", self.cell_layout_combo); row += 1
        add_section_pair_row(
            grid, row, "Diameter:", self.cell_diameter_spin, "Min size:", self.cell_min_size_spin
        ); row += 1
        add_section_pair_row(grid, row, "Gamma:", self.cell_gamma_spin)
        return CollapsibleSection("Cell parameters", body, expanded=False)

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
        self.nucleus_seg_btn.clicked.connect(lambda: self._on_action("nucleus", "seg"))
        self.cell_seg_btn.clicked.connect(lambda: self._on_action("cell", "seg"))
        self.nucleus_track_btn.clicked.connect(lambda: self._on_action("nucleus", "track"))
        self.cell_track_btn.clicked.connect(lambda: self._on_action("cell", "track"))

    def _on_action(self, channel: str, kind: str) -> None:
        if self._running is not None:
            self._on_cancel()
            return
        if kind == "seg":
            self._run_segment(channel)
        else:
            self._run_track(channel)

    def _on_cancel(self) -> None:
        self._cancel_requested = True
        worker = self._worker
        if worker is not None and hasattr(worker, "quit"):
            worker.quit()

    # ---------------------------------------------------------- path helpers
    def _apply_paths(self) -> None:
        nuc = self._nucleus_edit.text().strip()
        cel = self._cell_edit.text().strip()
        out = self._output_dir_edit.text().strip()
        self._nucleus_path = Path(nuc) if nuc else None
        self._cell_path = Path(cel) if cel else None
        self._output_dir = Path(out) if out else None
        self._save_standalone_settings()
        self.gate.recompute()

    def _on_browse_nucleus(self) -> None:
        self._browse_file_into(self._nucleus_edit, "Select nucleus channel", self._apply_paths)

    def _on_browse_cell(self) -> None:
        self._browse_file_into(self._cell_edit, "Select cell channel", self._apply_paths)

    def _on_browse_output_dir(self) -> None:
        self._browse_dir_into(self._output_dir_edit, "Select output directory", self._apply_paths)

    def _standalone_fields(self) -> dict:
        return {
            "nucleus": self._nucleus_edit,
            "cell": self._cell_edit,
            "output_dir": self._output_dir_edit,
        }

    def _load_standalone_settings(self) -> None:
        self._load_path_settings(self._SETTINGS_APP, self._standalone_fields())
        if any(e.text().strip() for e in self._standalone_fields().values()):
            self._apply_paths()

    def _save_standalone_settings(self) -> None:
        self._save_path_settings(self._SETTINGS_APP, self._standalone_fields())

    # ------------------------------------------------------------- params
    def _channel_layout(self, channel: str) -> str:
        combo = self.nuc_layout_combo if channel == "nucleus" else self.cell_layout_combo
        return combo.currentText()

    def _build_nucleus_params(self) -> cellpose_runner.NucleusParams:
        do_3d = self.nuc_3d_chk.isChecked() and cellpose_runner.layout_has_z(
            self._channel_layout("nucleus")
        )
        return cellpose_runner.NucleusParams(
            do_3d=do_3d,
            anisotropy=float(self.nuc_anisotropy_spin.value()),
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

    # --------------------------------------------------------------- run: seg
    def _run_segment(self, channel: str) -> None:
        in_path = self._nucleus_path if channel == "nucleus" else self._cell_path
        out_dir = self._output_dir
        if in_path is None or not in_path.exists():
            self._status(f"Missing input for {channel}.")
            return
        if out_dir is None:
            self._status("No output directory selected.")
            return
        params = (
            self._build_nucleus_params() if channel == "nucleus"
            else self._build_cell_params()
        )
        layout = self._channel_layout(channel)
        self._cancel_requested = False
        progress_signal = self._progress_signal

        def _done(result):
            self._worker = None
            self._set_running(None)
            self._clear_progress()
            self._load_labels(f"{channel.title()} masks", result)
            self._status(f"{channel.title()} masks written → {Path(result).name}")

        @thread_worker(connect={
            "yielded": self._on_progress, "returned": _done, "errored": self._errored,
        })
        def _worker():
            yield (0, 1, "Loading input...")
            return segment_channel(
                in_path, out_dir, channel, params, layout,
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
        out_dir = self._output_dir
        if out_dir is None:
            self._status("No output directory selected.")
            return
        masks_path = out_dir / f"{channel}_masks.tif"
        if not masks_path.exists():
            self._status(f"No {channel}_masks.tif — segment first.")
            return
        max_distance = float(self.track_max_dist_spin.value())
        max_frame_gap = int(self.track_gap_spin.value())

        def _done(result):
            self._worker = None
            self._set_running(None)
            self._clear_progress()
            self._load_labels(f"{channel.title()} tracked", result)
            self._status(f"{channel.title()} tracked → {Path(result).name}")

        @thread_worker(connect={
            "yielded": self._on_progress, "returned": _done, "errored": self._errored,
        })
        def _worker():
            yield (0, 1, f"Tracking {channel} masks...")
            return track_channel(
                masks_path, out_dir, channel,
                max_distance=max_distance, max_frame_gap=max_frame_gap,
            )

        self._set_running(f"{channel}_track")
        self._status(f"Tracking {channel} masks (laptrack)...")
        self._worker = _worker()

    def _errored(self, exc) -> None:
        self._worker = None
        self._set_running(None)
        self._clear_progress()
        if isinstance(exc, cellpose_runner.CancelledError):
            self._status("Cancelled.")
        else:
            self._status(f"Error: {exc}")
            logger.exception("segment/track error", exc_info=exc)

    def _load_labels(self, name: str, path) -> None:
        try:
            data = _normalize_tzyx_labels(tifffile.imread(str(path)))
        except Exception:
            return
        if name in self.viewer.layers:
            try:
                self.viewer.layers[name].data = data
                return
            except Exception:
                self.viewer.layers.remove(self.viewer.layers[name])
        self.viewer.add_labels(data.astype(np.int32), name=name)

    # ---------------------------------------------------------- public API
    def get_state(self) -> dict:
        return {
            "nucleus": {
                "layout": self.nuc_layout_combo.currentText(),
                "do_3d": self.nuc_3d_chk.isChecked(),
                "anisotropy": self.nuc_anisotropy_spin.value(),
                "diameter": self.nuc_diameter_spin.value(),
                "min_size": self.nuc_min_size_spin.value(),
                "gamma": self.nuc_gamma_spin.value(),
            },
            "cell": {
                "layout": self.cell_layout_combo.currentText(),
                "diameter": self.cell_diameter_spin.value(),
                "min_size": self.cell_min_size_spin.value(),
                "gamma": self.cell_gamma_spin.value(),
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
        return (
            self.nucleus_seg_btn, self.nucleus_track_btn,
            self.cell_seg_btn, self.cell_track_btn,
        )

    def _register_gate_controls(self) -> None:
        g = self.gate
        idle = lambda: self._running is None  # noqa: E731
        g.register(self.nucleus_params_btn, ControlClass.HARMLESS)
        g.register(self.cell_params_btn, ControlClass.HARMLESS)
        for btn in self._action_buttons():
            own = lambda b=btn: self._running is None or self._active_btn() is b
            g.register(btn, ControlClass.RUN_VIEWER, when=own)
        g.recompute()

    _GLYPH = {"seg": "▶", "track": "⊳"}

    def _active_btn(self):
        return {
            "nucleus_seg": self.nucleus_seg_btn,
            "nucleus_track": self.nucleus_track_btn,
            "cell_seg": self.cell_seg_btn,
            "cell_track": self.cell_track_btn,
        }.get(self._running)

    def _set_running(self, key: str | None) -> None:
        # restore all glyphs first
        self.nucleus_seg_btn.setText("▶")
        self.nucleus_track_btn.setText("⊳")
        self.cell_seg_btn.setText("▶")
        self.cell_track_btn.setText("⊳")
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
