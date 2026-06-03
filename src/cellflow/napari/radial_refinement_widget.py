"""Standalone widget for radial contour-registration refinement.

Designed to drop into ``NucleusWorkflowWidget`` between the database browser
section and the correction section. Operates on the active project's
``2_nucleus/`` directory provided by ``pos_dir_provider``.

The widget is fully self-contained: it owns its parameters, its worker, its
status label, and the napari layers it creates. Action enablement is governed
by the shared :class:`~cellflow.napari.ui_gate.UiGate` (its refine actions are
blocked while any viewer owner — correction or a live preview — is active); the
host wires it in via ``set_on_promoted_callback`` (to refresh the correction
tracked layer after a promote).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.database.validation import (
    read_validated_frames,
    read_validated_tracks,
)
from cellflow.napari.ui_style import (
    action_button,
    add_block_pair_row,
    add_block_checkbox_row,
    block_grid,
    compact_spinbox,
    parameter_heading,
    status_label,
)
from cellflow.napari.ui_gate import ControlClass, UiGate
from cellflow.napari.widgets import CollapsibleSection
from cellflow.tracking_ultrack.radial_refine import (
    PRESETS,
    RadialRefineConfig,
    config_label,
    preset_name,
    promote_refinement_to_tracked,
    refine_frame,
    refine_stack,
    write_refinement_outputs,
)

logger = logging.getLogger(__name__)


# Layer name constants — kept here so they can't collide with workflow layers
_REFINE_PREVIEW_LAYER = "Refined Preview: Nucleus"
_REFINE_FULL_LAYER = "Refined: Nucleus"


# ── Local UI helpers (mirroring the workflow widget's private helpers) ────────


def _dspin(lo, hi, val, step, decimals, tooltip=""):
    from qtpy.QtWidgets import QDoubleSpinBox

    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setValue(val)
    if tooltip:
        s.setToolTip(tooltip)
    return s


def _ispin(lo, hi, val, tooltip=""):
    from qtpy.QtWidgets import QSpinBox

    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    if tooltip:
        s.setToolTip(tooltip)
    return s


def _btn(text: str, tooltip: str = "") -> QPushButton:
    b = QPushButton(text)
    if tooltip:
        b.setToolTip(tooltip)
    action_button(b, expand=True)
    return b


def _heading(text: str) -> QLabel:
    return parameter_heading(QLabel(text))


def _make_status() -> QLabel:
    lbl = QLabel("")
    lbl.setWordWrap(True)
    lbl.setVisible(False)
    return status_label(lbl)


def _make_progress() -> QProgressBar:
    p = QProgressBar()
    p.setRange(0, 1)
    p.setValue(0)
    p.setVisible(False)
    p.setTextVisible(True)
    return p


# ── Widget ────────────────────────────────────────────────────────────────────


class RadialRefinementWidget(QWidget):
    """Radial contour-registration refinement of tracked nucleus labels."""

    def __init__(
        self,
        viewer: napari.Viewer,
        pos_dir_provider: Callable[[], Path | None],
        parent: QWidget | None = None,
        gate: UiGate | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir_provider = pos_dir_provider
        #: App-wide UI gate; a private one is created for standalone use.
        self.gate = gate if gate is not None else UiGate(self)
        self._on_promoted_callback: Callable[[], None] | None = None

        self._worker = None
        self._last_refined_path: Path | None = None
        self._stop_flag: bool = False
        self._suppress_preset_change: bool = False

        self._setup_ui()
        self._connect_signals()
        self._apply_preset(self.preset_combo.currentText())
        self._register_gate_controls()

    # -- Public wiring -----------------------------------------------------

    def _register_gate_controls(self) -> None:
        """Register refinement actions with the app-wide UI gate.

        Refinement rewrites tracked labels and repaints the viewer, so its
        actions are blocked while any viewer owner (correction / live preview)
        is active — replacing the old correction-only lock with the general
        rule. Cancel stays available whenever a refinement worker is running.
        """
        g = self.gate
        g.register(self.preview_btn, ControlClass.RUN_VIEWER, when=self._can_refine)
        g.register(self.refine_all_btn, ControlClass.RUN_VIEWER, when=self._can_refine)
        g.register(
            self.promote_btn,
            ControlClass.RUN_VIEWER,
            when=lambda: (
                self._can_refine()
                and self._last_refined_path is not None
                and self._last_refined_path.exists()
            ),
        )
        g.register(
            self.cancel_btn,
            ControlClass.RUN_HEADLESS,
            when=lambda: self._worker is not None,
        )
        g.recompute()

    def _can_refine(self) -> bool:
        return self._pos_dir() is not None and self._worker is None

    def set_on_promoted_callback(self, fn: Callable[[], None]) -> None:
        """Host supplies a callable invoked after a successful promote."""
        self._on_promoted_callback = fn

    def refresh(self) -> None:
        """Re-evaluate enabled/disabled state — call this from host signals."""
        self._refresh_button_states()

    # -- UI ----------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        # Parameters --------------------------------------------------------
        params_inner = QWidget()
        params_lay = QVBoxLayout(params_inner)
        params_lay.setContentsMargins(0, 0, 0, 0)
        params_lay.setSpacing(6)

        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(8)
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        for name in PRESETS.keys():
            self.preset_combo.addItem(name)
        self.preset_combo.addItem("custom")
        self.preset_combo.setCurrentText("conservative")
        preset_row.addWidget(self.preset_combo)
        preset_row.addStretch()
        params_lay.addLayout(preset_row)

        params_lay.addWidget(_heading("Search window"))
        g = block_grid(horizontal_spacing=12)
        self.n_rays_spin = _ispin(
            8, 512, 64, tooltip="Number of radial rays per object."
        )
        self.max_outward_spin = _ispin(
            0, 100, 10, tooltip="Maximum outward radial movement (px)."
        )
        self.max_inward_spin = _ispin(
            0, 100, 10, tooltip="Maximum inward radial movement (px)."
        )
        add_block_pair_row(
            g, 0,
            "Rays:", compact_spinbox(self.n_rays_spin),
            "Max outward:", compact_spinbox(self.max_outward_spin),
        )
        add_block_pair_row(
            g, 1,
            "Max inward:", compact_spinbox(self.max_inward_spin),
        )
        params_lay.addLayout(g)

        params_lay.addWidget(_heading("Score weights"))
        g = block_grid(horizontal_spacing=12)
        self.wc_spin = _dspin(
            0, 100, 4.0, 0.5, 2, "Contour attraction weight."
        )
        self.wi_spin = _dspin(
            0, 100, 2.0, 0.5, 2, "Just-inside foreground support weight."
        )
        self.we_spin = _dspin(
            0, 100, 2.0, 0.5, 2, "Boundary foreground support weight."
        )
        self.wd_spin = _dspin(
            0, 100, 2.0, 0.5, 2, "Deformation penalty from original radius."
        )
        add_block_pair_row(
            g, 0,
            "Contour (wc):", compact_spinbox(self.wc_spin),
            "Just-inside (wi):", compact_spinbox(self.wi_spin),
        )
        add_block_pair_row(
            g, 1,
            "Boundary (we):", compact_spinbox(self.we_spin),
            "Deformation (wd):", compact_spinbox(self.wd_spin),
        )
        params_lay.addLayout(g)

        params_lay.addWidget(_heading("Smoothing"))
        g = block_grid(horizontal_spacing=12)
        self.smooth_spin = _ispin(
            0, 20, 3, tooltip="Circular radius-smoothing passes."
        )
        self.orig_pull_spin = _dspin(
            0, 1, 0.30, 0.05, 2,
            "Pullback toward original radius during smoothing.",
        )
        add_block_pair_row(
            g, 0,
            "Smooth passes:", compact_spinbox(self.smooth_spin),
            "Original pull:", compact_spinbox(self.orig_pull_spin),
        )
        params_lay.addLayout(g)

        params_lay.addWidget(_heading("Validated cells"))
        g = block_grid(horizontal_spacing=12)
        self.respect_validated_check = QCheckBox(
            "Respect validated frames and tracks"
        )
        self.respect_validated_check.setChecked(True)
        self.respect_validated_check.setToolTip(
            "Skip refinement on validated frames; copy validated tracks through "
            "unchanged in every frame they appear."
        )
        add_block_checkbox_row(g, 0, self.respect_validated_check)
        params_lay.addLayout(g)

        self.params_section = CollapsibleSection(
            "Refinement Parameters",
            params_inner,
            expanded=True,
        )
        outer.addWidget(self.params_section)

        # Actions -----------------------------------------------------------
        actions_inner = QWidget()
        actions_lay = QVBoxLayout(actions_inner)
        actions_lay.setContentsMargins(0, 0, 0, 0)
        actions_lay.setSpacing(6)

        self.preview_btn = _btn(
            "Preview frame",
            "Refine the current frame only and display as a transient layer.",
        )
        self.refine_all_btn = _btn(
            "Refine all frames",
            "Refine every frame in tracked_labels.tif and write a candidate "
            "TIFF to 2_nucleus/refinement/.",
        )
        self.promote_btn = _btn(
            "Promote to tracked_labels.tif",
            "Replace tracked_labels.tif with the most recent refinement "
            "(previous file backed up to tracked_labels.prev.tif).",
        )
        self.cancel_btn = _btn(
            "Cancel", "Stop the current refinement run."
        )

        row1 = QHBoxLayout()
        row1.addWidget(self.preview_btn)
        row1.addWidget(self.refine_all_btn)
        actions_lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(self.promote_btn)
        row2.addWidget(self.cancel_btn)
        actions_lay.addLayout(row2)

        self.refined_file_lbl = QLabel("No refinement computed yet.")
        self.refined_file_lbl.setWordWrap(True)
        self.refined_file_lbl.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred
        )
        actions_lay.addWidget(self.refined_file_lbl)

        self.status_lbl = _make_status()
        actions_lay.addWidget(self.status_lbl)
        self.progress_bar = _make_progress()
        actions_lay.addWidget(self.progress_bar)

        outer.addWidget(actions_inner)

    # -- Signal wiring -----------------------------------------------------

    def _connect_signals(self) -> None:
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        for spin in (
            self.n_rays_spin,
            self.max_outward_spin,
            self.max_inward_spin,
            self.smooth_spin,
        ):
            spin.valueChanged.connect(self._mark_custom_if_user_change)
        for dspin in (
            self.wc_spin,
            self.wi_spin,
            self.we_spin,
            self.wd_spin,
            self.orig_pull_spin,
        ):
            dspin.valueChanged.connect(self._mark_custom_if_user_change)

        self.preview_btn.clicked.connect(self._on_preview)
        self.refine_all_btn.clicked.connect(self._on_refine_all)
        self.promote_btn.clicked.connect(self._on_promote)
        self.cancel_btn.clicked.connect(self._on_cancel)

    # -- Preset / config plumbing -----------------------------------------

    def _apply_preset(self, name: str) -> None:
        if name == "custom":
            return
        preset = PRESETS.get(name)
        if preset is None:
            return
        self._suppress_preset_change = True
        try:
            self.n_rays_spin.setValue(int(preset.n_rays))
            self.max_outward_spin.setValue(int(preset.max_outward))
            self.max_inward_spin.setValue(int(preset.max_inward))
            self.wc_spin.setValue(float(preset.wc))
            self.wi_spin.setValue(float(preset.wi))
            self.we_spin.setValue(float(preset.we))
            self.wd_spin.setValue(float(preset.wd))
            self.smooth_spin.setValue(int(preset.smooth))
            self.orig_pull_spin.setValue(float(preset.orig_pull))
        finally:
            self._suppress_preset_change = False

    def _on_preset_changed(self, name: str) -> None:
        if self._suppress_preset_change:
            return
        if name != "custom":
            self._apply_preset(name)

    def _mark_custom_if_user_change(self, *_args) -> None:
        if self._suppress_preset_change:
            return
        cfg = self._config_from_controls()
        match = preset_name(cfg)
        target = match if match is not None else "custom"
        if self.preset_combo.currentText() != target:
            self._suppress_preset_change = True
            try:
                self.preset_combo.setCurrentText(target)
            finally:
                self._suppress_preset_change = False

    def _config_from_controls(self) -> RadialRefineConfig:
        return RadialRefineConfig(
            n_rays=int(self.n_rays_spin.value()),
            max_outward=int(self.max_outward_spin.value()),
            max_inward=int(self.max_inward_spin.value()),
            wc=float(self.wc_spin.value()),
            wi=float(self.wi_spin.value()),
            we=float(self.we_spin.value()),
            wd=float(self.wd_spin.value()),
            smooth=int(self.smooth_spin.value()),
            orig_pull=float(self.orig_pull_spin.value()),
        )

    # -- Path helpers ------------------------------------------------------

    def _pos_dir(self) -> Path | None:
        try:
            d = self._pos_dir_provider()
        except Exception:
            return None
        return d if d is not None else None

    def _tracked_path(self) -> Path | None:
        d = self._pos_dir()
        return d / "2_nucleus" / "tracked_labels.tif" if d else None

    def _contours_path(self) -> Path | None:
        d = self._pos_dir()
        return d / "1_cellpose" / "nucleus_contours.tif" if d else None

    def _fg_path(self) -> Path | None:
        d = self._pos_dir()
        return d / "1_cellpose" / "nucleus_foreground.tif" if d else None

    def _refinement_dir(self) -> Path | None:
        d = self._pos_dir()
        return d / "2_nucleus" / "refinement" if d else None

    # -- Status / progress / button state ---------------------------------

    def _status(self, msg: str) -> None:
        self.status_lbl.setText(msg)
        self.status_lbl.setVisible(bool(msg))

    def _on_progress(self, payload) -> None:
        try:
            done, total, msg = payload
        except Exception:
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, max(1, int(total)))
        self.progress_bar.setValue(int(done))
        if msg:
            self._status(msg)

    def _clear_progress(self) -> None:
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)

    def _refresh_button_states(self) -> None:
        """Re-apply enablement via the gate, and surface a blocking hint."""
        self.gate.recompute()
        owner_label = self.gate.owner_label()
        if owner_label is not None and self._pos_dir() is not None:
            self._status(f"Exit {owner_label} to refine.")

    # -- Validated-mask lookup --------------------------------------------

    def _read_frozen(
        self, t_range: int
    ) -> tuple[set[int], set[int]]:
        if not self.respect_validated_check.isChecked():
            return set(), set()
        d = self._pos_dir()
        if d is None:
            return set(), set()
        try:
            frames = read_validated_frames(d)
        except Exception as exc:
            logger.warning("read_validated_frames failed: %s", exc)
            frames = set()
        try:
            tracks = read_validated_tracks(d)
        except Exception as exc:
            logger.warning("read_validated_tracks failed: %s", exc)
            tracks = {}
        return set(int(t) for t in frames), set(int(k) for k in tracks.keys())

    # -- Layer helpers -----------------------------------------------------

    def _put_layer(
        self, name: str, data: np.ndarray, *, opacity: float = 0.6
    ) -> None:
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            self.viewer.add_labels(data, name=name, opacity=opacity)

    # -- Inputs loading ----------------------------------------------------

    def _load_inputs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        tracked = self._tracked_path()
        contours = self._contours_path()
        fg = self._fg_path()
        for p in (tracked, contours, fg):
            if p is None or not p.exists():
                self._status(f"Missing: {p}")
                return None
        labels = tifffile.imread(str(tracked))
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        contours_arr = tifffile.imread(str(contours)).astype(np.float32)
        fg_arr = tifffile.imread(str(fg)).astype(np.float32)
        if labels.shape != contours_arr.shape or labels.shape != fg_arr.shape:
            self._status(
                "Input shape mismatch: "
                f"tracked={labels.shape} contours={contours_arr.shape} "
                f"foreground={fg_arr.shape}"
            )
            return None
        return labels.astype(np.uint32, copy=False), contours_arr, fg_arr

    # -- Actions: preview --------------------------------------------------

    def _current_t(self) -> int:
        try:
            return int(self.viewer.dims.current_step[0])
        except Exception:
            return 0

    def _on_preview(self) -> None:
        if self.gate.owner is not None:
            self._status(f"Exit {self.gate.owner_label()} to refine.")
            return
        loaded = self._load_inputs()
        if loaded is None:
            return
        labels, contours, fg = loaded
        T = labels.shape[0]
        t = max(0, min(self._current_t(), T - 1))
        cfg = self._config_from_controls()

        frozen_frames, frozen_labels = self._read_frozen(T)
        if t in frozen_frames:
            self._status(
                f"Frame {t} is validated; preview shows unchanged labels."
            )
            preview = labels[t].astype(np.uint32, copy=True)
        else:
            out_t, _ = refine_frame(
                labels[t], contours[t], fg[t], cfg,
                frozen_labels=frozen_labels,
            )
            preview = out_t
            self._status(
                f"Preview frame {t} ({config_label(cfg)})."
            )

        full = np.zeros_like(labels, dtype=np.uint32)
        full[t] = preview
        self._put_layer(_REFINE_PREVIEW_LAYER, full, opacity=0.6)

    # -- Actions: refine all ----------------------------------------------

    def _on_refine_all(self) -> None:
        if self.gate.owner is not None:
            self._status(f"Exit {self.gate.owner_label()} to refine.")
            return
        loaded = self._load_inputs()
        if loaded is None:
            return
        labels, contours, fg = loaded
        cfg = self._config_from_controls()
        out_dir = self._refinement_dir()
        if out_dir is None:
            self._status("No project open.")
            return

        frozen_frames, frozen_labels = self._read_frozen(labels.shape[0])
        self._stop_flag = False

        def _done(result):
            try:
                refined, per_obj, summary, tif_path = result
            except Exception:
                self._worker = None
                self._clear_progress()
                self._refresh_button_states()
                return
            self._worker = None
            self._clear_progress()
            self._last_refined_path = tif_path
            self.refined_file_lbl.setText(f"Last refinement: {tif_path.name}")
            self._put_layer(_REFINE_FULL_LAYER, refined, opacity=0.6)
            self._status(
                f"Refined ({summary.name}): "
                f"median_ratio={summary.median_ratio_vs_original:.3f}, "
                f"holes={summary.hole_pixels}, "
                f"fragmented={summary.fragmented_label_frames}, "
                f"missing_seed={summary.missing_seed_label_frames}"
            )
            self._refresh_button_states()

        def _errored(exc):
            self._worker = None
            self._clear_progress()
            self._status(f"Refinement failed: {exc}")
            self._refresh_button_states()
            logger.exception("Radial refinement failed", exc_info=exc)

        should_stop = lambda: self._stop_flag

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": _errored,
        })
        def _worker():
            import queue as _queue
            import threading

            msg_queue: _queue.SimpleQueue = _queue.SimpleQueue()
            result_holder: list = []
            exc_holder: list = []

            def _progress_cb(done: int, total: int, msg: str) -> None:
                msg_queue.put((done, total, msg))

            def _run() -> None:
                try:
                    refined, per_obj, summary = refine_stack(
                        labels, contours, fg, cfg,
                        frozen_frames=frozen_frames,
                        frozen_labels=frozen_labels,
                        progress_cb=_progress_cb,
                        should_stop=should_stop,
                    )
                    tif_path = write_refinement_outputs(
                        out_dir, cfg, refined, per_obj, summary
                    )
                    result_holder.append((refined, per_obj, summary, tif_path))
                except Exception as e:
                    exc_holder.append(e)

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            yield (0, max(1, labels.shape[0]), "Starting refinement...")
            while t.is_alive() or not msg_queue.empty():
                try:
                    yield msg_queue.get_nowait()
                except _queue.Empty:
                    t.join(timeout=0.05)
            if exc_holder:
                raise exc_holder[0]
            return result_holder[0]

        self._status(f"Refining all frames ({config_label(cfg)})...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, max(1, labels.shape[0]))
        self.progress_bar.setValue(0)
        self._worker = _worker()
        self._refresh_button_states()

    # -- Actions: promote / cancel ----------------------------------------

    def _on_promote(self) -> None:
        if self.gate.owner is not None:
            self._status(f"Exit {self.gate.owner_label()} before promoting.")
            return
        if self._last_refined_path is None or not self._last_refined_path.exists():
            self._status("No refinement to promote.")
            return
        tracked = self._tracked_path()
        if tracked is None:
            self._status("No project open.")
            return
        try:
            backup = promote_refinement_to_tracked(
                self._last_refined_path, tracked
            )
        except Exception as exc:
            self._status(f"Promote failed: {exc}")
            logger.exception("Promote failed")
            return
        backup_msg = f" (backup: {backup.name})" if backup else ""
        self._status(f"Promoted {self._last_refined_path.name}{backup_msg}.")
        if self._on_promoted_callback is not None:
            try:
                self._on_promoted_callback()
            except Exception as exc:
                logger.warning("on_promoted_callback failed: %s", exc)

    def _on_cancel(self) -> None:
        if self._worker is None:
            return
        self._stop_flag = True
        self._status("Cancelling...")
