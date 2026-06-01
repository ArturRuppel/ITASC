"""Cell segmentation workflow widget — simplified divergence pipeline.

A single run path and a single preview path (no per-stage run buttons). The
widget consumes the cached divergence maps produced upstream by
``DivergenceMapsWidget`` (``1_cellpose/cell_contours.tif`` +
``1_cellpose/cell_foreground.tif``) plus the tracked nucleus seeds, and runs
the unary-only geodesic-Voronoi pipeline from
:func:`cellflow.segmentation.segment_cells_divergence`:

    1. Map cleanup (foreground + contours) — local-mean residual + threshold.
    2. Temporal contour smoothing (full run only).
    3. Foreground mask.
    4. Unary-only segmentation → ``3_cell/tracked_labels.tif``.

Live preview recomputes the current frame off the GUI thread on any param edit
or time scrub and surfaces every *cheap* intermediate as a napari layer — up to
and including the weighted cost field the geodesic walk would traverse. It
deliberately stops short of the geodesic label assignment (by far the slowest
step) on every edit, so tuning stays responsive; the cost field already explains
where the labels would land. A separate on-demand button runs the geodesic walk
for just the current frame when the user wants to see the actual labels.

Temporal smoothing (``memory_tau > 0``) needs the whole movie, so the preview
computes the cleaned-and-smoothed contour stack once, caches it keyed on the
contour/temporal knobs, and slices the current frame from it for the cost field
and labels — reusing the cache across frame scrubs and edits to non-smoothing
knobs. The previewed frame then matches the full run exactly. The full run
processes all frames in memory, runs the geodesic walk, and persists only the
labels. Correction is delegated to :class:`CellCorrectionWidget`, unchanged.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import QTimer, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.correction.labels import best_overlapping_label
from cellflow.napari._widget_helpers import (
    dslider as _dslider,
    heading as _heading,
    islider as _islider,
    tool_btn as _tool_btn,
)
from cellflow.napari.cell_correction_widget import CellCorrectionWidget
from cellflow.napari.ui_style import (
    add_section_header,
    add_section_pair_row,
    section_grid,
    stage_header_action_button,
    stage_header_label,
    status_label,
)
from cellflow.napari.widgets import (
    CollapsibleSection,
    PipelineFilesWidget,
    make_pipeline_files_header,
)
from cellflow.segmentation import CancelledError
from cellflow.segmentation.cell_divergence_segmentation import (
    CellDivergenceParams,
    clean_and_smooth_contours,
    segment_cells_divergence,
)
from cellflow.segmentation.cell_label_icm import commit_labels

logger = logging.getLogger(__name__)

# ── Preview layer names (pipeline order) ──────────────────────────────────────
_PREFIX = "[Cell]"
_FG_RAW_LAYER = f"{_PREFIX} foreground (sigmoid)"
_FG_CLEAN_LAYER = f"{_PREFIX} foreground cleaned"
_CT_RAW_LAYER = f"{_PREFIX} contours raw"
_CT_CLEAN_LAYER = f"{_PREFIX} contours cleaned"
_FG_MASK_LAYER = f"{_PREFIX} foreground mask"
_COST_LAYER = f"{_PREFIX} weighted cost field"
# The geodesic label assignment is the pipeline's slowest step, so the live
# preview stops at the cost field and never creates this layer on activation or
# on a param/time edit. It is filled only when the user explicitly clicks the
# on-demand labels button (single current frame), and by the full run (as
# ``_TRACKED_CELL_LAYER``).
_LABELS_LAYER = f"{_PREFIX} cell labels"

# Tracked pipeline layer the correction widget syncs against.
_TRACKED_CELL_LAYER = "Tracked: Cell"

# (layer_name, kind, colormap) in pipeline order. Image layers carry a
# colormap; label layers ignore it.
_PREVIEW_IMAGE_LAYERS = (
    (_FG_RAW_LAYER, "gray"),
    (_FG_CLEAN_LAYER, "gray"),
    (_CT_RAW_LAYER, "magma"),
    (_CT_CLEAN_LAYER, "magma"),
    (_COST_LAYER, "turbo"),
)
_PREVIEW_LABEL_LAYERS = (_FG_MASK_LAYER,)
_PREVIEW_LAYERS = (
    _FG_RAW_LAYER, _FG_CLEAN_LAYER, _CT_RAW_LAYER, _CT_CLEAN_LAYER,
    _FG_MASK_LAYER, _COST_LAYER,
)
# Everything removed when preview deactivates: the always-on intermediates plus
# the on-demand labels layer (created only if the user asked for labels).
_PREVIEW_TEARDOWN_LAYERS = _PREVIEW_LAYERS + (_LABELS_LAYER,)


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


# Canonical input/output paths (kept as literals so the file-contract test and
# the PipelineFilesWidget stay in sync):
#   1_cellpose/cell_contours.tif, 1_cellpose/cell_foreground.tif,
#   2_nucleus/tracked_labels.tif → 3_cell/tracked_labels.tif


class CellWorkflowWidget(QWidget):
    """Simplified divergence-based cell segmentation widget."""

    _run_progress = Signal(str)

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None

        # Live preview state — a compute is in flight (None when idle); rapid
        # edits while one runs set _preview_pending so exactly one fresh pass
        # fires when it returns.
        self._preview_active = False
        self._preview_worker = None
        self._preview_pending = False
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._refresh_preview)

        # Cached cleaned+smoothed contour stack for the preview when
        # ``memory_tau > 0`` (the smoothing needs the whole movie). Keyed on the
        # contour/temporal knobs so it survives frame scrubs and edits to the
        # non-smoothing knobs (fg_*, alpha, gamma) and only recomputes when a
        # knob that actually changes the smoothing is touched. Dropped on
        # deactivate to free the (T, Y, X) array.
        self._smoothed_stack = None
        self._smoothed_key = None

        # On-demand single-frame labels worker (explicit, one-shot — never
        # fired by param edits or time scrubs).
        self._labels_worker = None

        # Full-run worker state.
        self._run_worker = None
        self._running = False

        self._setup_ui()
        self._connect_signals()
        self._run_progress.connect(self._set_status)

    # ================================================================
    # UI
    # ================================================================
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # ── Pipeline files ────────────────────────────────────────────
        self._files_widget = PipelineFilesWidget(
            [
                ("Inputs", [
                    ("1_cellpose/cell_contours.tif", "Cell contours (divergence)"),
                    ("1_cellpose/cell_foreground.tif", "Cell foreground (sigmoid)"),
                    ("2_nucleus/tracked_labels.tif", "Nucleus tracked labels"),
                ]),
                ("Output", [
                    ("3_cell/tracked_labels.tif", "Cell tracked labels"),
                ]),
            ],
            viewer=self.viewer,
        )
        self._pipeline_files_section = CollapsibleSection(
            "Pipeline Files", self._files_widget, expanded=False,
        )
        (
            self.pipeline_files_header,
            self.pipeline_files_header_lbl,
            self.pipeline_files_toggle_btn,
        ) = make_pipeline_files_header(
            self._pipeline_files_section, stage_key="cell", parent=self,
        )
        root.addWidget(self.pipeline_files_header)
        root.addWidget(self._pipeline_files_section)

        # ── Stage row: ⚙ params / ◉ live preview / ▶ run ──────────────
        self.params_btn = _tool_btn(
            "⚙", "Toggle segmentation parameters.", checkable=True
        )
        self.active_btn = _tool_btn(
            "◉", "Live preview (tune against the current frame).", checkable=True
        )
        self.labels_btn = _tool_btn(
            "▦",
            "Compute cell labels for the current frame only (the slow geodesic "
            "step). Available while live preview is active.",
        )
        self.labels_btn.setEnabled(False)
        self.run_btn = _tool_btn(
            "▶", "Run the full pipeline over all frames and write tracked_labels.tif."
        )
        for button in (self.params_btn, self.active_btn, self.labels_btn, self.run_btn):
            stage_header_action_button(button, "cell")

        self.params_section = self._build_params_section()
        self.params_section.set_header_visible(False)
        self.params_section.collapse()
        self.params_btn.toggled.connect(
            lambda checked: self.params_section._toggle.setChecked(checked)
        )

        root.addLayout(self._stage_row(
            self._stage_label("Segmentation"),
            self.params_btn, self.active_btn, self.labels_btn, self.run_btn,
        ))
        root.addWidget(self.params_section)

        self.pipeline_status_lbl = _make_status()
        root.addWidget(self.pipeline_status_lbl)
        self.pipeline_progress_bar = _make_progress()
        root.addWidget(self.pipeline_progress_bar)

        # ── Correction (delegated, unchanged) ─────────────────────────
        self.cell_correction_widget = CellCorrectionWidget(
            self.viewer,
            pos_dir_provider=lambda: self._pos_dir,
            files_widget_refresh_callback=lambda pd: self._files_widget.refresh(pd),
        )
        self._install_correction_aliases()
        self.cell_correction_widget.hide()
        root.addWidget(self.correction_header)
        root.addWidget(self.correction_mode_section)

    def _build_params_section(self) -> CollapsibleSection:
        """One flat panel: every knob shown once, grouped by stage."""
        body = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(8, 4, 4, 4)
        body.setLayout(grid)

        max_workers = max(1, os.cpu_count() or 1)

        # ── Map cleanup: foreground ──────────────────────────────────
        self.fg_strength_spin = _dslider(
            0, 1, 0.0, 0.05, 2,
            "0 = raw sigmoid, 1 = full local-mean background subtraction.",
        )
        self.fg_threshold_spin = _dslider(
            0, 1, 0.1, 0.01, 2,
            "Cleaned-foreground cutoff → fill mask (sigmoid scale).",
        )
        self.fg_window_spin = _islider(
            3, 201, 51, tooltip="Local-mean window for foreground residual (odd px)."
        )
        # ── Map cleanup: contours ────────────────────────────────────
        self.contour_strength_spin = _dslider(
            0, 1, 1.0, 0.05, 2, "0 = raw, 1 = full local-mean subtraction."
        )
        self.contour_threshold_spin = _dslider(
            0, 1, 0.0, 0.001, 3, "Noise floor on normalized contour; below → 0."
        )
        self.contour_norm_pct_spin = _dslider(
            90, 100, 99.0, 0.5, 1,
            "Percentile mapped to 1.0 in the contour [0,1] normalize.",
        )
        self.contour_window_spin = _islider(
            3, 201, 51, tooltip="Local-mean window for contour residual (odd px)."
        )
        # ── Temporal smoothing ───────────────────────────────────────
        self.memory_tau_spin = _dslider(
            0, 1, 0.0, 0.01, 3,
            "Temporal EMA crossover (~the contour value you call weak). 0 = off.",
        )
        self.memory_floor_spin = _dslider(
            0.001, 0.5, 0.01, 0.001, 3,
            "Min per-frame alpha; ghost half-life (~69 frames @ 0.01).",
        )
        # ── Segmentation ─────────────────────────────────────────────
        self.alpha_spin = _dslider(
            0, 1000, 100.0, 1.0, 1,
            "Contour weight: cost = 1 + α·contour — boundary snap strength.",
        )
        self.gamma_spin = _dslider(
            0, 100, 2.0, 0.5, 1,
            "Foreground weight: + γ·(1 − fg_score) — pull toward confident fg.",
        )
        self.n_workers_spin = _islider(
            1, max_workers, min(4, max_workers),
            tooltip="Parallel workers for geodesic computation (compute only).",
        )

        row = 0
        add_section_header(grid, row, _heading("Map cleanup")); row += 1
        add_section_header(grid, row, _heading("Foreground")); row += 1
        add_section_pair_row(
            grid, row,
            "Strength:", self.fg_strength_spin,
            "Threshold:", self.fg_threshold_spin,
        ); row += 1
        add_section_pair_row(grid, row, "Window:", self.fg_window_spin); row += 1
        add_section_header(grid, row, _heading("Contours")); row += 1
        add_section_pair_row(
            grid, row,
            "Strength:", self.contour_strength_spin,
            "Floor:", self.contour_threshold_spin,
        ); row += 1
        add_section_pair_row(
            grid, row,
            "Norm %:", self.contour_norm_pct_spin,
            "Window:", self.contour_window_spin,
        ); row += 1
        add_section_header(grid, row, _heading("Temporal smoothing")); row += 1
        add_section_pair_row(
            grid, row,
            "Memory τ:", self.memory_tau_spin,
            "Memory floor:", self.memory_floor_spin,
        ); row += 1
        add_section_header(grid, row, _heading("Segmentation")); row += 1
        add_section_pair_row(
            grid, row,
            "α (contour):", self.alpha_spin,
            "γ (foreground):", self.gamma_spin,
        ); row += 1
        add_section_pair_row(grid, row, "Workers:", self.n_workers_spin); row += 1

        return CollapsibleSection("Segmentation Parameters", body, expanded=False)

    @staticmethod
    def _stage_label(text: str) -> QLabel:
        return stage_header_label(QLabel(text), "cell")

    @staticmethod
    def _stage_row(label: QLabel, *trailing: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(label)
        for widget in trailing:
            row.addWidget(widget)
        row.addStretch(1)
        return row

    def _install_correction_aliases(self) -> None:
        """Install compatibility aliases for controls owned by CellCorrectionWidget."""
        c = self.cell_correction_widget
        self.correction_header = c.header
        self.correction_header_lbl = c.header_lbl
        self.correction_shortcuts_btn = c.shortcuts_btn
        self.correction_params_btn = c.params_btn
        self.correction_active_btn = c.active_btn
        self.correction_mode_section = c.section
        self.correction_widget = c.correction_widget
        self.correction_status_lbl = c.correction_status_lbl
        self.correction_shortcuts_section = c.correction_shortcuts_section
        self.load_labels_btn = c.load_labels_btn
        self.save_labels_btn = c.save_labels_btn
        self.fill_holes_btn = c.fill_holes_btn
        self.fix_semiholes_btn = c.fix_semiholes_btn
        self.cleanup_btn = c.cleanup_btn
        self.expand_cell_btn = c.expand_cell_btn
        self.correction_scope_combo = c.correction_scope_combo
        self.hole_radius_spin = c.hole_radius_spin
        self.semihole_opening_spin = c.semihole_opening_spin
        self.expand_max_px_spin = c.expand_max_px_spin

    # ================================================================
    # Signals
    # ================================================================
    def _connect_signals(self) -> None:
        for spin in (
            self.fg_strength_spin, self.fg_threshold_spin, self.memory_tau_spin,
            self.alpha_spin, self.gamma_spin, self.fg_window_spin,
            self.contour_window_spin, self.contour_strength_spin,
            self.contour_threshold_spin, self.contour_norm_pct_spin,
            self.memory_floor_spin, self.n_workers_spin,
        ):
            spin.valueChanged.connect(self._on_param_changed)
        self.active_btn.toggled.connect(self._on_activate)
        self.labels_btn.clicked.connect(self._on_compute_labels)
        self.run_btn.clicked.connect(self._on_run_clicked)
        if hasattr(self.viewer, "dims") and hasattr(self.viewer.dims, "events"):
            try:
                self.viewer.dims.events.current_step.connect(self._on_time_changed)
            except Exception:
                pass

    # ================================================================
    # Params / paths
    # ================================================================
    def _params(self) -> CellDivergenceParams:
        return CellDivergenceParams(
            fg_window=int(self.fg_window_spin.value()),
            fg_strength=float(self.fg_strength_spin.value()),
            fg_threshold=float(self.fg_threshold_spin.value()),
            contour_window=int(self.contour_window_spin.value()),
            contour_strength=float(self.contour_strength_spin.value()),
            contour_threshold=float(self.contour_threshold_spin.value()),
            contour_norm_pct=float(self.contour_norm_pct_spin.value()),
            memory_tau=float(self.memory_tau_spin.value()),
            memory_floor=float(self.memory_floor_spin.value()),
            alpha=float(self.alpha_spin.value()),
            gamma=float(self.gamma_spin.value()),
            n_workers=int(self.n_workers_spin.value()),
        )

    def _p(self, *parts: str) -> Path | None:
        return self._pos_dir.joinpath(*parts) if self._pos_dir else None

    def _contours_path(self):   return self._p("1_cellpose", "cell_contours.tif")
    def _foreground_path(self): return self._p("1_cellpose", "cell_foreground.tif")
    def _nuc_path(self):        return self._p("2_nucleus", "tracked_labels.tif")
    def _output_path(self):     return self._p("3_cell", "tracked_labels.tif")

    def _maps_present(self) -> bool:
        ct, fg = self._contours_path(), self._foreground_path()
        return ct is not None and ct.exists() and fg is not None and fg.exists()

    # ================================================================
    # Public API (consumed by the main widget)
    # ================================================================
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._files_widget.refresh(pos_dir)
        if pos_dir is None:
            self.correction_widget.deactivate()

    def get_state(self) -> dict:
        return {
            "cleanup": {
                "fg_window": self.fg_window_spin.value(),
                "fg_strength": self.fg_strength_spin.value(),
                "fg_threshold": self.fg_threshold_spin.value(),
                "contour_window": self.contour_window_spin.value(),
                "contour_strength": self.contour_strength_spin.value(),
                "contour_threshold": self.contour_threshold_spin.value(),
                "contour_norm_pct": self.contour_norm_pct_spin.value(),
            },
            "temporal": {
                "memory_tau": self.memory_tau_spin.value(),
                "memory_floor": self.memory_floor_spin.value(),
            },
            "segmentation": {
                "alpha": self.alpha_spin.value(),
                "gamma": self.gamma_spin.value(),
                "n_workers": self.n_workers_spin.value(),
            },
            "correction": {
                "expand_max_px": self.expand_max_px_spin.value(),
                "hole_radius": self.hole_radius_spin.value(),
                "semihole_opening": self.semihole_opening_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        _map = {
            "cleanup": {
                "fg_window": self.fg_window_spin,
                "fg_strength": self.fg_strength_spin,
                "fg_threshold": self.fg_threshold_spin,
                "contour_window": self.contour_window_spin,
                "contour_strength": self.contour_strength_spin,
                "contour_threshold": self.contour_threshold_spin,
                "contour_norm_pct": self.contour_norm_pct_spin,
            },
            "temporal": {
                "memory_tau": self.memory_tau_spin,
                "memory_floor": self.memory_floor_spin,
            },
            "segmentation": {
                "alpha": self.alpha_spin,
                "gamma": self.gamma_spin,
                "n_workers": self.n_workers_spin,
            },
            "correction": {
                "expand_max_px": self.expand_max_px_spin,
                "hole_radius": self.hole_radius_spin,
                "semihole_opening": self.semihole_opening_spin,
            },
        }
        for group_key, widgets in _map.items():
            group = state.get(group_key, {})
            if not isinstance(group, dict):
                continue
            for key, spin in widgets.items():
                if key in group:
                    spin.setValue(group[key])

    def set_selection_callback(self, fn) -> None:
        self.correction_widget.set_selection_callback(fn)

    def select_matching_cell_label(
        self, t: int, source_label: int,
        *, source_labels: np.ndarray | None = None,
    ) -> None:
        if "[Correction] Cell Labels" in self.viewer.layers:
            target_layer = self.viewer.layers["[Correction] Cell Labels"]
        elif _TRACKED_CELL_LAYER in self.viewer.layers:
            target_layer = self.viewer.layers[_TRACKED_CELL_LAYER]
        else:
            return
        if source_labels is None:
            if "Tracked: Nucleus" not in self.viewer.layers:
                return
            source_labels = np.asarray(self.viewer.layers["Tracked: Nucleus"].data)
        target = np.asarray(target_layer.data)
        matched = best_overlapping_label(target, source_labels, t, source_label)
        self.correction_widget.select_label(t, matched, notify=False)

    # ================================================================
    # Status / layer helpers
    # ================================================================
    def _set_status(self, msg: str) -> None:
        self.pipeline_status_lbl.setText(msg)
        self.pipeline_status_lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    def _show_layer(self, name, data, kwargs, adder):
        if name in self.viewer.layers:
            try:
                self.viewer.layers[name].data = data
            except Exception:
                self.viewer.layers.remove(self.viewer.layers[name])
                adder(data, name=name, **kwargs)
        else:
            adder(data, name=name, **kwargs)

    def _current_t(self) -> int:
        step = getattr(getattr(self.viewer, "dims", None), "current_step", (0,))
        return int(step[0]) if len(step) >= 1 else 0

    def _read_frame(self, path, t: int, dtype=np.float32) -> np.ndarray:
        return np.asarray(tifffile.imread(str(path), key=t), dtype=dtype)

    def _map_shape(self):
        """(T, Y, X) from the contours TIFF header (no pixel load)."""
        ct = self._contours_path()
        if ct is None or not ct.exists():
            return None
        with tifffile.TiffFile(str(ct)) as tf:
            n_frames = len(tf.pages)
            y, x = tf.pages[0].shape[-2], tf.pages[0].shape[-1]
        return int(n_frames), int(y), int(x)

    # ================================================================
    # Live preview (single frame, all intermediates)
    # ================================================================
    def _on_param_changed(self, *_args) -> None:
        if self._preview_active:
            self._preview_timer.start()

    def _on_time_changed(self, *_args) -> None:
        if self._preview_active:
            self._preview_timer.start()

    def _on_activate(self, checked: bool) -> None:
        self._preview_active = bool(checked)
        # The on-demand labels button only makes sense against a live preview.
        self.labels_btn.setEnabled(checked and not self._running)
        if checked:
            self._refresh_preview()
        else:
            self._preview_pending = False
            for name in _PREVIEW_TEARDOWN_LAYERS:
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)
            # Free the resident smoothed stack; the preview session is over.
            self._smoothed_stack = None
            self._smoothed_key = None
            self._set_status("")

    @staticmethod
    def _smooth_key(params: CellDivergenceParams) -> tuple:
        """Cache key for the smoothed contour stack: the knobs that determine it.

        The fg_*, alpha and gamma knobs are deliberately absent — they do not
        change the smoothed contours, so editing them reuses the cached stack.
        """
        return (
            params.contour_window, params.contour_strength,
            params.contour_threshold, params.contour_norm_pct,
            params.memory_tau, params.memory_floor,
        )

    def _cached_stack_for(self, params: CellDivergenceParams):
        """Return the resident smoothed stack iff it matches ``params``, else None."""
        if (
            self._smoothed_stack is not None
            and self._smoothed_key == self._smooth_key(params)
        ):
            return self._smoothed_stack
        return None

    def _preview_inputs(self):
        """Validate the inputs a single-frame compute needs.

        Sets a status message and returns ``None`` on any failure; otherwise
        returns ``(shape, contours_path, foreground_path, nuc_path)``.
        """
        if not self._preview_active:
            return None
        if not self._maps_present():
            self._set_status("Cell divergence maps not found — run Divergence Maps first.")
            return None
        nuc_path = self._nuc_path()
        if nuc_path is None or not nuc_path.exists():
            self._set_status("Nucleus tracked_labels.tif not found.")
            return None
        shape = self._map_shape()
        if shape is None:
            self._set_status("Cell divergence maps not found — run Divergence Maps first.")
            return None
        return shape, self._contours_path(), self._foreground_path(), nuc_path

    def _compute_frame(
        self, *, t, params, contours_path, foreground_path, nuc_path,
        cached_stack, with_labels,
    ):
        """Worker-thread body shared by the live preview and the labels button.

        Reads frame ``t`` of each map. When ``memory_tau > 0`` the cleaned +
        smoothed contour stack (whole movie) feeds the frame through
        ``contours_clean_override`` so the result matches the full run; the
        stack is taken from ``cached_stack`` when valid, otherwise computed here
        and returned for the caller to cache. Returns ``(t, result, new_stack)``
        where ``new_stack`` is non-``None`` only when it was (re)computed.
        """
        fg = self._read_frame(foreground_path, t)[np.newaxis]
        nuc = self._read_frame(nuc_path, t, dtype=np.uint32)[np.newaxis]
        contour = self._read_frame(contours_path, t)[np.newaxis]

        new_stack = None
        override = None
        if params.memory_tau > 0.0:
            stack = cached_stack
            if stack is None:
                full = np.asarray(
                    tifffile.imread(str(contours_path)), dtype=np.float32
                )
                stack = clean_and_smooth_contours(full, params)
                new_stack = stack
            override = stack[t]

        result = segment_cells_divergence(
            contour, fg, nuc, params, frame=0, with_labels=with_labels,
            contours_clean_override=override,
        )
        return t, result, new_stack

    def _refresh_preview(self):
        """Recompute the current frame's preview off the GUI thread.

        Mirrors the atom widget: while a pass is in flight, further edits arm
        ``_preview_pending`` so one fresh pass (latest params/frame) fires when
        the current one returns. Returns the started worker (or ``None``).
        """
        info = self._preview_inputs()
        if info is None:
            return None
        shape, contours_path, foreground_path, nuc_path = info
        self._ensure_preview_layers(shape[-2:])
        if self._preview_worker is not None:
            self._preview_pending = True
            return self._preview_worker

        params = self._params()
        n_frames = shape[0]
        t = max(0, min(self._current_t(), n_frames - 1))
        smooth = params.memory_tau > 0.0
        cached_stack = self._cached_stack_for(params) if smooth else None
        if not smooth:
            # Smoothing turned off — release the resident stack.
            self._smoothed_stack = None
            self._smoothed_key = None
        if smooth and cached_stack is None:
            self._set_status(f"Temporal smoothing over {n_frames} frames…")
        else:
            self._set_status(f"Computing cell preview for frame {t}…")

        @thread_worker(connect={
            "returned": self._on_preview_done,
            "errored": self._on_preview_error,
        })
        def _worker():
            t_, result, new_stack = self._compute_frame(
                t=t, params=params, contours_path=contours_path,
                foreground_path=foreground_path, nuc_path=nuc_path,
                cached_stack=cached_stack, with_labels=False,
            )
            return t_, result, params, new_stack

        self._preview_worker = _worker()
        return self._preview_worker

    def _on_preview_done(self, payload) -> None:
        self._preview_worker = None
        t, result, params, new_stack = payload
        self._cache_stack(params, new_stack)
        if self._preview_active:
            self._apply_intermediates(result)
            coverage = 100.0 * float(result.foreground_mask.mean())
            self._set_status(
                f"Frame {t}: {coverage:.0f}% fill coverage "
                f"(labels on ▦ / Run)."
            )
        if self._preview_pending and self._preview_active:
            self._preview_pending = False
            self._refresh_preview()
        else:
            self._preview_pending = False

    def _on_preview_error(self, exc: Exception) -> None:
        self._preview_worker = None
        self._preview_pending = False
        self._set_status(f"Cell preview failed: {exc}")
        logger.exception("Cell preview worker error", exc_info=exc)

    # ── On-demand single-frame labels (the slow geodesic step, explicit) ──────
    def _on_compute_labels(self) -> None:
        """Run the geodesic Voronoi for the current frame only, on request.

        An explicit action: it never fires on param edits or time scrubs, so
        tuning stays responsive. Reuses (or fills) the smoothed-stack cache so
        the previewed labels match the full run for this frame.
        """
        if not self._preview_active or self._labels_worker is not None:
            return
        info = self._preview_inputs()
        if info is None:
            return
        shape, contours_path, foreground_path, nuc_path = info
        self._ensure_preview_layers(shape[-2:])
        self._ensure_labels_layer(_LABELS_LAYER, shape[-2:])

        params = self._params()
        n_frames = shape[0]
        t = max(0, min(self._current_t(), n_frames - 1))
        smooth = params.memory_tau > 0.0
        cached_stack = self._cached_stack_for(params) if smooth else None
        self.labels_btn.setEnabled(False)
        self._set_status(f"Computing cell labels for frame {t}…")

        @thread_worker(connect={
            "returned": self._on_labels_done,
            "errored": self._on_labels_error,
        })
        def _worker():
            t_, result, new_stack = self._compute_frame(
                t=t, params=params, contours_path=contours_path,
                foreground_path=foreground_path, nuc_path=nuc_path,
                cached_stack=cached_stack, with_labels=True,
            )
            return t_, result, params, new_stack

        self._labels_worker = _worker()

    def _on_labels_done(self, payload) -> None:
        self._labels_worker = None
        t, result, params, new_stack = payload
        self._cache_stack(params, new_stack)
        if self._preview_active:
            self._apply_intermediates(result)
            n_labels = (
                int(np.unique(result.labels[result.labels > 0]).size)
                if result.labels is not None else 0
            )
            if result.labels is not None:
                self._fill_labels_layer(
                    _LABELS_LAYER, result.labels.astype(np.int32)
                )
            coverage = 100.0 * float(result.foreground_mask.mean())
            self._set_status(
                f"Frame {t}: {coverage:.0f}% fill, {n_labels} cell labels."
            )
        self.labels_btn.setEnabled(self._preview_active and not self._running)

    def _on_labels_error(self, exc: Exception) -> None:
        self._labels_worker = None
        self.labels_btn.setEnabled(self._preview_active and not self._running)
        self._set_status(f"Cell labels failed: {exc}")
        logger.exception("Cell labels worker error", exc_info=exc)

    def _cache_stack(self, params: CellDivergenceParams, new_stack) -> None:
        """Hold a freshly computed smoothed stack resident for later frames/edits."""
        if new_stack is not None:
            self._smoothed_stack = new_stack
            self._smoothed_key = self._smooth_key(params)

    def _apply_intermediates(self, result) -> None:
        """Fill the six always-on preview layers (everything but cell labels)."""
        self._fill_image_layer(_FG_RAW_LAYER, result.foreground_raw)
        self._fill_image_layer(_FG_CLEAN_LAYER, result.foreground_clean)
        self._fill_image_layer(_CT_RAW_LAYER, result.contours_raw)
        self._fill_image_layer(_CT_CLEAN_LAYER, result.contours_clean)
        self._fill_labels_layer(
            _FG_MASK_LAYER, result.foreground_mask.astype(np.uint8)
        )
        self._fill_image_layer(_COST_LAYER, self._cost_for_display(result.cost_field))

    @staticmethod
    def _cost_for_display(cost: np.ndarray) -> np.ndarray:
        """Mask the geodesic-cost background (inf) to NaN for the colormap."""
        return np.where(np.isfinite(cost), cost, np.nan).astype(np.float32)

    # ── preview layers (one 2-D current-frame layer per intermediate) ─────
    # The preview only ever computes the current frame, and a time scrub
    # recomputes it (``_on_time_changed``), so the layers are 2-D (Y, X) rather
    # than full (T, Y, X) stacks — napari broadcasts them across the time axis.
    # This keeps activation cheap regardless of movie length.
    def _ensure_preview_layers(self, yx) -> None:
        for name, colormap in _PREVIEW_IMAGE_LAYERS:
            self._ensure_image_layer(name, yx, colormap)
        for name in _PREVIEW_LABEL_LAYERS:
            self._ensure_labels_layer(name, yx)

    def _ensure_image_layer(self, name: str, yx, colormap: str) -> None:
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if tuple(layer.data.shape) == tuple(yx):
                return
            was_visible = layer.visible
            self.viewer.layers.remove(name)
        else:
            was_visible = True
        new_layer = self.viewer.add_image(
            np.full(yx, np.nan, dtype=np.float32), name=name, colormap=colormap,
        )
        new_layer.visible = was_visible

    def _ensure_labels_layer(self, name: str, yx) -> None:
        from napari.layers import Labels
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if isinstance(layer, Labels) and tuple(layer.data.shape) == tuple(yx):
                return
            was_visible = layer.visible
            self.viewer.layers.remove(name)
        else:
            was_visible = True
        new_layer = self.viewer.add_labels(
            np.zeros(yx, dtype=np.int32), name=name, opacity=0.55
        )
        new_layer.visible = was_visible

    def _fill_image_layer(self, name: str, frame: np.ndarray) -> None:
        if name not in self.viewer.layers:
            return
        layer = self.viewer.layers[name]
        layer.data = np.asarray(frame, dtype=np.float32)
        finite = frame[np.isfinite(frame)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            if hi > lo:
                layer.contrast_limits = (lo, hi)
        layer.refresh()

    def _fill_labels_layer(self, name: str, frame: np.ndarray) -> None:
        if name not in self.viewer.layers:
            return
        layer = self.viewer.layers[name]
        layer.data = np.asarray(frame, dtype=np.int32)
        layer.refresh()

    # ================================================================
    # Full run (final output only)
    # ================================================================
    def _on_run_clicked(self) -> None:
        if self._running:
            self._on_cancel()
        else:
            self._on_run()

    def _on_cancel(self) -> None:
        if self._run_worker is not None and hasattr(self._run_worker, "quit"):
            self._run_worker.quit()
        self._run_worker = None
        self._running = False
        self._set_run_idle()
        self._clear_progress()
        self._set_status("Cancelled.")

    def _on_run(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        if not self._maps_present():
            self._set_status("Cell divergence maps not found — run Divergence Maps first.")
            return
        nuc_path = self._nuc_path()
        if nuc_path is None or not nuc_path.exists():
            self._set_status("Nucleus tracked_labels.tif not found.")
            return

        params = self._params()
        contours_path = self._contours_path()
        foreground_path = self._foreground_path()
        output_path = self._output_path()
        pos_dir = self._pos_dir

        def _done(result):
            self._run_worker = None
            self._running = False
            self._set_run_idle()
            self._clear_progress()
            labels, n_labels = result
            self._show_layer(
                _TRACKED_CELL_LAYER, labels, {"visible": True}, self.viewer.add_labels
            )
            self._files_widget.refresh(pos_dir)
            self._set_status(
                f"Segmentation complete — {n_labels} labels, "
                f"saved to {output_path.name}."
            )

        def _error(exc):
            self._run_worker = None
            self._running = False
            self._set_run_idle()
            self._clear_progress()
            if isinstance(exc, CancelledError):
                self._set_status("Cancelled.")
                return
            self._set_status(f"Error: {exc}")
            logger.exception("Cell segmentation run error", exc_info=exc)

        @thread_worker(connect={"returned": _done, "errored": _error})
        def _worker():
            progress = self._run_progress

            def _cb(msg: str) -> None:
                progress.emit(str(msg))

            contours = tifffile.imread(str(contours_path))
            foreground = tifffile.imread(str(foreground_path))
            nuc = tifffile.imread(str(nuc_path))
            result = segment_cells_divergence(
                contours, foreground, nuc, params, progress_cb=_cb,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            commit_labels(result.labels, output_path)
            n_labels = int(np.unique(result.labels[result.labels > 0]).size)
            return result.labels, n_labels

        self._set_status("Segmenting all frames…")
        self.pipeline_progress_bar.setRange(0, 0)
        self.pipeline_progress_bar.setVisible(True)
        self._running = True
        self._set_run_running()
        self._run_worker = _worker()

    def _set_run_running(self) -> None:
        self.run_btn.setText("✕")
        self.run_btn.setToolTip("Cancel.")
        self.active_btn.setEnabled(False)
        self.labels_btn.setEnabled(False)

    def _set_run_idle(self) -> None:
        self.run_btn.setText("▶")
        self.run_btn.setToolTip(
            "Run the full pipeline over all frames and write tracked_labels.tif."
        )
        self.active_btn.setEnabled(True)
        self.labels_btn.setEnabled(self._preview_active)

    def _clear_progress(self) -> None:
        self.pipeline_progress_bar.setRange(0, 100)
        self.pipeline_progress_bar.setValue(0)
        self.pipeline_progress_bar.setVisible(False)
