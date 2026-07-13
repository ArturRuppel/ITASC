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
or time scrub. A row of *Compute* checkboxes — Foreground, Contours, Cost,
Labels — selects which layer groups the preview computes (not merely shows):
only the ticked groups are calculated, so leaving the slow geodesic *Labels* box
off (the default) keeps tuning responsive while the cheaper foreground / contour
/ cost intermediates still update. Each computed frame is cached in memory keyed
on the params, so scrubbing back to an already-computed frame repaints instantly;
any param edit clears the cache.

Temporal smoothing (``memory_tau > 0``) needs the whole movie, so the preview
computes the cleaned-and-smoothed contour stack once, caches it keyed on the
contour/temporal knobs, and slices the current frame from it for the cost field
and labels — reusing the cache across frame scrubs and edits to non-smoothing
knobs. The previewed frame then matches the full run exactly. The full run
processes all frames in memory (deriving the foreground fill mask in-process),
runs the geodesic walk, and persists only the labels. Correction is delegated to
:class:`CellCorrectionWidget`, unchanged.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
from cellflow.napari._standalone_paths import StandalonePathsMixin
from cellflow.napari.correction.cell_correction_widget import CellCorrectionWidget
from cellflow.napari.ui_gate import ControlClass, UiGate
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
    compute_cell_foreground,
    segment_cells_divergence,
)
from cellflow.segmentation.cell_label_icm import commit_labels
from cellflow.core.commit import commit_state, promote_labels
from cellflow.napari._paths import NucleusArtifactPaths

logger = logging.getLogger(__name__)

# ── Preview layer names (pipeline order) ──────────────────────────────────────
_PREFIX = "[Cell]"
_FG_RAW_LAYER = f"{_PREFIX} foreground (sigmoid)"
_FG_CLEAN_LAYER = f"{_PREFIX} foreground cleaned"
_FG_MASK_LAYER = f"{_PREFIX} foreground mask"
_CT_RAW_LAYER = f"{_PREFIX} contours raw"
_CT_CLEAN_LAYER = f"{_PREFIX} contours cleaned"
_COST_LAYER = f"{_PREFIX} weighted cost field"
_LABELS_LAYER = f"{_PREFIX} cell labels"

# Tracked pipeline layer the correction widget syncs against.
_TRACKED_CELL_LAYER = "Tracked: Cell"

# ── Compute groups ────────────────────────────────────────────────────────────
# Each checkbox owns one group of preview layers and gates whether that group is
# *computed at all* (not merely shown). The live preview computes only the
# groups whose box is ticked and caches the per-frame result so scrubbing back
# to a frame repaints instantly.
_FG_GROUP_LAYERS = (_FG_RAW_LAYER, _FG_CLEAN_LAYER, _FG_MASK_LAYER)
_CONTOUR_GROUP_LAYERS = (_CT_RAW_LAYER, _CT_CLEAN_LAYER)
_COST_GROUP_LAYERS = (_COST_LAYER,)
_LABELS_GROUP_LAYERS = (_LABELS_LAYER,)
_ALL_PREVIEW_LAYERS = (
    _FG_GROUP_LAYERS + _CONTOUR_GROUP_LAYERS
    + _COST_GROUP_LAYERS + _LABELS_GROUP_LAYERS
)

# Colormap per image layer; the rest are napari Labels layers.
_IMAGE_COLORMAPS = {
    _FG_RAW_LAYER: "gray",
    _FG_CLEAN_LAYER: "gray",
    _CT_RAW_LAYER: "magma",
    _CT_CLEAN_LAYER: "magma",
    _COST_LAYER: "turbo",
}
_LABELS_KIND = {_FG_MASK_LAYER, _LABELS_LAYER}

# How much a single-frame compute produces, in increasing cost. The desired
# level is the max over the ticked checkboxes; a cached frame is reused only when
# its stored level already covers the desired one.
_LEVEL_NONE = 0
_LEVEL_FG = 1       # foreground raw/clean/mask — compute_cell_foreground (cheap)
_LEVEL_SEG = 2      # + contours raw/clean + cost field (with_labels=False)
_LEVEL_LABELS = 3   # + the geodesic cell labels (with_labels=True, slow)


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
#   nucleus_labels.tif (committed) → 3_cell/tracked_labels.tif


class CellWorkflowWidget(StandalonePathsMixin, QWidget):
    """Simplified divergence-based cell segmentation widget."""

    _run_progress = Signal(str)

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: QWidget | None = None,
        gate: UiGate | None = None,
        standalone: bool = False,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        #: App-wide UI gate; a private one is created for standalone use.
        self.gate = gate if gate is not None else UiGate(self)
        #: When True the piece runs on its own, with its own input/output path
        #: pickers; the orchestrator drives the embedded widget via refresh().
        self._standalone = standalone
        self._pos_dir: Path | None = None
        #: Standalone explicit inputs/output (mirrors the orchestrator's staged
        #: 1_cellpose/2_nucleus/3_cell layout with arbitrary file locations).
        self._sa_foreground: Path | None = None
        self._sa_contours: Path | None = None
        self._sa_nucleus: Path | None = None
        self._sa_output_dir: Path | None = None

        # Live preview state — a compute is in flight (None when idle); rapid
        # edits while one runs set _preview_pending so exactly one fresh pass
        # fires when it returns.
        self._preview_active = False
        self._preview_worker = None
        self._preview_pending = False
        # Image preview layers whose contrast has not yet been auto-set. We seed
        # contrast once, from the first real frame a freshly created layer
        # receives, then leave it alone so a scrub to another frame (or a manual
        # contrast tweak) is not clobbered on every refresh.
        self._image_needs_autocontrast: set[str] = set()
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(150)
        self._preview_timer.timeout.connect(self._refresh_preview)

        # Per-frame result cache: frame t → (level, result). Holds whatever the
        # last compute produced for each visited frame so scrubbing back to an
        # already-computed frame repaints instantly. Keyed (``_frame_cache_key``)
        # on the full param tuple — any param edit clears it.
        self._frame_cache: dict[int, tuple[int, object]] = {}
        self._frame_cache_key: tuple | None = None

        # Cached cleaned+smoothed contour stack for the preview when
        # ``memory_tau > 0`` (the smoothing needs the whole movie). Keyed on the
        # contour/temporal knobs so it survives frame scrubs and edits to the
        # non-smoothing knobs (fg_*, balance, feature_strength) and only
        # recomputes when a
        # knob that actually changes the smoothing is touched. Dropped on
        # deactivate to free the (T, Y, X) array.
        self._smoothed_stack = None
        self._smoothed_key = None

        # Full-run worker state.
        self._run_worker = None
        self._running = False

        self._setup_ui()
        self._connect_signals()
        self._register_gate_controls()
        self._run_progress.connect(self._set_status)

        if self._standalone:
            self._load_standalone_settings()

    # ================================================================
    # UI
    # ================================================================
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)
        # Embedded, the widget lives in main_widget's AlignTop scroll layout and
        # a Maximum policy keeps each section compact. Standalone, napari docks it
        # directly, so fill the dock (Preferred) and pin content to the top with a
        # trailing stretch (mirrors NucleusWorkflowWidget).
        if self._standalone:
            root.setAlignment(Qt.AlignTop)
            self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        else:
            self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # ── Standalone input/output path pickers ──────────────────────
        # Only shown when the piece runs on its own; the orchestrator drives the
        # workspace through refresh() instead. The cell segmentation consumes the
        # Cellpose-produced foreground/contours plus the tracked nucleus seeds.
        self._paths_container = QWidget()
        paths_col = QVBoxLayout(self._paths_container)
        paths_col.setContentsMargins(0, 0, 0, 0)
        paths_col.setSpacing(2)
        apply = self._apply_standalone_paths
        self._foreground_edit = self._add_path_row(
            paths_col, "Foreground:", "Cell foreground .tif",
            lambda: self._browse_file_into(self._foreground_edit, "Select cell foreground image", apply),
            apply,
        )
        self._contours_edit = self._add_path_row(
            paths_col, "Contours:", "Cell contours .tif",
            lambda: self._browse_file_into(self._contours_edit, "Select cell contours image", apply),
            apply,
        )
        self._nucleus_edit = self._add_path_row(
            paths_col, "Nucleus:", "Tracked nucleus labels .tif",
            lambda: self._browse_file_into(self._nucleus_edit, "Select tracked nucleus labels", apply),
            apply,
        )
        self._output_dir_edit = self._add_path_row(
            paths_col, "Output dir:", "Folder for 3_cell/tracked_labels.tif",
            lambda: self._browse_dir_into(self._output_dir_edit, "Select output directory", apply),
            apply,
        )
        root.addWidget(self._paths_container)
        self._paths_container.setVisible(self._standalone)

        # ── Pipeline files ────────────────────────────────────────────
        self._files_widget = PipelineFilesWidget(
            [
                ("Inputs", [
                    ("1_cellpose/cell_contours.tif", "Cell contours (divergence)"),
                    ("1_cellpose/cell_foreground.tif", "Cell foreground (sigmoid)"),
                    ("nucleus_labels.tif", "Nucleus labels (committed)"),
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
        # The staged-files panel lists 1_cellpose/2_nucleus/3_cell paths that
        # don't exist in the standalone flat layout; the path pickers cover it.
        self.pipeline_files_header.setVisible(not self._standalone)
        self._pipeline_files_section.setVisible(not self._standalone)

        # ── Segmentation stage row: ⚙ params / ◉ live preview / ▶ run ──
        self.params_btn = _tool_btn(
            "⚙", "Toggle segmentation parameters.", checkable=True
        )
        self.active_btn = _tool_btn(
            "◉", "Live preview (tune against the current frame).", checkable=True
        )
        self.run_btn = _tool_btn(
            "▶", "Run the full pipeline over all frames and write tracked_labels.tif."
        )
        # Two Finalize buttons promoting the working tracked labels to
        # cell_labels.tif: one at the tail of the correction toolbar (visible in
        # active correction mode) and one beside the correction on/off toggle
        # (always visible). Both are placed by the correction widget below; the
        # handler stays here. Finalize is not a segmentation-stage action, so it
        # is out of the stage row and skips the stage-pill styling loop (the
        # header button is styled by the header builder, the toolbar one plain).
        _finalize_tip = (
            "Finalize: promote tracked labels to cell_labels.tif "
            "(final, downstream-stable output)."
        )
        self.finalize_btn = _tool_btn("✔", _finalize_tip)
        self.finalize_header_btn = _tool_btn("✔", _finalize_tip)
        for button in (self.finalize_btn, self.finalize_header_btn):
            button.clicked.connect(self._on_finalize)
        for button in (self.params_btn, self.active_btn, self.run_btn):
            stage_header_action_button(button, "cell")

        self.params_section = self._build_params_section()
        self.params_section.set_header_visible(False)
        self.params_section.collapse()
        self.params_btn.toggled.connect(
            lambda checked: self.params_section._toggle.setChecked(checked)
        )

        root.addLayout(self._stage_row(
            self._stage_label("Segmentation"),
            self.params_btn, self.active_btn, self.run_btn,
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
            finalize_btn=self.finalize_btn,
            finalize_header_btn=self.finalize_header_btn,
        )
        self._install_correction_aliases()
        self.cell_correction_widget.hide()
        root.addWidget(self.correction_header)
        root.addWidget(self.correction_mode_section)

    def _build_compute_row(self) -> QWidget:
        """The "Compute:" checkbox row — one box per preview layer group.

        Each box gates whether its group is *computed* by the live preview (not
        just shown). ``_compute_groups`` ties each box to its layers and the
        single-frame compute level it requires; Labels (the slow geodesic walk)
        is off by default.
        """
        self.fg_check = QCheckBox("Foreground")
        self.fg_check.setChecked(True)
        self.fg_check.setToolTip(
            "Compute + show the foreground layers (raw, cleaned, fill mask)."
        )
        self.contour_check = QCheckBox("Contours")
        self.contour_check.setChecked(True)
        self.contour_check.setToolTip(
            "Compute + show the contour layers (raw, cleaned)."
        )
        self.cost_check = QCheckBox("Cost")
        self.cost_check.setChecked(True)
        self.cost_check.setToolTip("Compute + show the weighted cost field.")
        self.labels_check = QCheckBox("Labels")
        self.labels_check.setChecked(False)
        self.labels_check.setToolTip(
            "Compute the cell labels (the slow geodesic step) for the current "
            "frame."
        )
        # (checkbox, owned layers, single-frame compute level it requires).
        self._compute_groups = (
            (self.fg_check, _FG_GROUP_LAYERS, _LEVEL_FG),
            (self.contour_check, _CONTOUR_GROUP_LAYERS, _LEVEL_SEG),
            (self.cost_check, _COST_GROUP_LAYERS, _LEVEL_SEG),
            (self.labels_check, _LABELS_GROUP_LAYERS, _LEVEL_LABELS),
        )

        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(8, 0, 4, 0)
        lay.setSpacing(8)
        lay.addWidget(QLabel("Compute:"))
        for check, _layers, _level in self._compute_groups:
            lay.addWidget(check)
        lay.addStretch(1)
        return row

    def _build_params_section(self) -> CollapsibleSection:
        """One flat panel: the Compute checkboxes on top, then every knob.

        The Compute row gates which layer groups the live preview computes; the
        grid below holds the cleanup / smoothing / segmentation knobs.
        """
        body = QWidget(self)
        outer = QVBoxLayout(body)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        outer.addWidget(self._build_compute_row())

        grid_widget = QWidget()
        grid = section_grid()
        grid.setContentsMargins(8, 4, 4, 4)
        grid_widget.setLayout(grid)
        outer.addWidget(grid_widget)

        max_workers = max(1, os.cpu_count() or 1)

        # ── Map cleanup: foreground ──────────────────────────────────
        self.fg_strength_spin = _dslider(
            0, 1, 0.0, 0.05, 2,
            "0 = raw sigmoid, 1 = full local-mean background subtraction.",
        )
        self.fg_threshold_spin = _dslider(
            0, 1, 0.1, 0.001, 3,
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
        self.balance_spin = _dslider(
            0.0, 1.0, 0.98, 0.01, 2,
            "Contour↔foreground split r: 1 = pure contour, 0 = pure "
            "foreground. cost = 1 + s·[r·contour + (1−r)·(1−fg)].",
        )
        self.feature_strength_spin = _dslider(
            0, 1000, 100.0, 1.0, 1,
            "Feature strength s: how hard contour/foreground bend the walk vs "
            "a plain distance Voronoi. 0 = pure distance.",
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
            "Balance (r):", self.balance_spin,
            "Strength (s):", self.feature_strength_spin,
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
        self.save_labels_btn = c.save_labels_btn
        self.fill_holes_btn = c.fill_holes_btn
        self.cleanup_btn = c.cleanup_btn
        self.correction_scope_combo = c.correction_scope_combo
        self.hole_radius_spin = c.hole_radius_spin

    # ================================================================
    # Signals
    # ================================================================
    def _connect_signals(self) -> None:
        for spin in (
            self.fg_strength_spin, self.fg_threshold_spin, self.memory_tau_spin,
            self.balance_spin, self.feature_strength_spin, self.fg_window_spin,
            self.contour_window_spin, self.contour_strength_spin,
            self.contour_threshold_spin, self.contour_norm_pct_spin,
            self.memory_floor_spin, self.n_workers_spin,
        ):
            spin.valueChanged.connect(self._on_param_changed)
        self.active_btn.toggled.connect(self._on_activate)
        self.run_btn.clicked.connect(self._on_run_clicked)
        for check, layers, _level in self._compute_groups:
            check.toggled.connect(
                lambda checked, lyrs=layers: self._on_compute_toggled(lyrs, checked)
            )
        if hasattr(self.viewer, "dims") and hasattr(self.viewer.dims, "events"):
            try:
                self.viewer.dims.events.current_step.connect(self._on_time_changed)
            except Exception:
                pass

    def _register_gate_controls(self) -> None:
        """Register this section's controls with the app-wide UI gate.

        Live preview and cell correction are mutually-exclusive viewer owners
        (across sections too, since the gate is shared). The full-run button
        rebuilds the data those owners view, so it is blocked while either is
        active.
        """
        g = self.gate
        g.register_owner(
            "cell_preview",
            "cell live preview",
            exit_fn=lambda: self.active_btn.setChecked(False),
        )
        g.register_owner(
            "correction:cell",
            "cell correction mode",
            exit_fn=lambda: self.correction_active_btn.setChecked(False),
        )
        g.register(
            self.active_btn,
            ControlClass.VIEWER_OWNER,
            owner_token="cell_preview",
            when=lambda: not self._running,
        )
        g.register(
            self.correction_active_btn,
            ControlClass.VIEWER_OWNER,
            owner_token="correction:cell",
            when=lambda: not self._running,
        )
        g.register(self.run_btn, ControlClass.RUN_VIEWER)
        self.correction_active_btn.toggled.connect(self._on_cell_correction_gate)
        g.recompute()

    def _on_cell_correction_gate(self, checked: bool) -> None:
        if checked:
            self.gate.claim_viewer("correction:cell")
        else:
            self.gate.release_viewer("correction:cell")

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
            balance=float(self.balance_spin.value()),
            feature_strength=float(self.feature_strength_spin.value()),
            n_workers=int(self.n_workers_spin.value()),
        )

    def _p(self, *parts: str) -> Path | None:
        return self._pos_dir.joinpath(*parts) if self._pos_dir else None

    def _contours_path(self):
        return self._sa_contours if self._standalone else self._p("1_cellpose", "cell_contours.tif")

    def _foreground_path(self):
        return self._sa_foreground if self._standalone else self._p("1_cellpose", "cell_foreground.tif")

    def _nuc_path(self):
        # Staged: consume the *committed* nucleus labels in the position base
        # folder, not the pre-commit 2_nucleus working labels (the commit gate
        # means cell segmentation runs off the validated nucleus tracking).
        return self._sa_nucleus if self._standalone else self._p("nucleus_labels.tif")

    def _nuc_missing_msg(self) -> str:
        if self._standalone:
            return "Nucleus labels not found."
        # Staged use reads the committed base-folder labels, so the fix is to
        # commit the nucleus stage rather than just re-run its tracking.
        return "Committed nucleus_labels.tif not found — commit the nucleus stage first."

    def _output_path(self):
        if self._standalone:
            return self._sa_output_dir / "3_cell" / "tracked_labels.tif" if self._sa_output_dir else None
        return self._p("3_cell", "tracked_labels.tif")

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
        self._refresh_finalize_btn()

    # ── Commit contract: promote working labels → cell_labels.tif ──────
    def _committed_labels_path(self) -> Path | None:
        if self._standalone:
            return self._sa_output_dir / "cell_labels.tif" if self._sa_output_dir else None
        return NucleusArtifactPaths(self._pos_dir).cell_labels if self._pos_dir else None

    def _on_finalize(self) -> None:
        working = self._output_path()
        committed = self._committed_labels_path()
        if working is None or committed is None or not working.exists():
            self._set_status("No tracked labels to commit — run the pipeline first.")
            return
        try:
            promote_labels(working, committed)
        except (FileNotFoundError, ValueError) as exc:
            self._set_status(f"Commit failed: {exc}")
            return
        self._set_status(f"Committed cell labels → {committed.name}.")
        self._refresh_finalize_btn()
        # Surface the committed file (and let the host repaint the stage dots).
        self._files_widget.refresh(self._pos_dir)

    def _finalize_btns(self) -> tuple:
        return tuple(
            b
            for b in (
                getattr(self, "finalize_btn", None),
                getattr(self, "finalize_header_btn", None),
            )
            if b is not None
        )

    def _refresh_finalize_btn(self) -> None:
        buttons = self._finalize_btns()
        if not buttons:
            return
        working = self._output_path()
        committed = self._committed_labels_path()
        if working is None or committed is None:
            for button in buttons:
                button.setEnabled(False)
            return
        state = commit_state(working, committed)
        tooltip = (
            "Working labels are newer than the committed cell_labels.tif — "
            "re-commit to update."
            if state == "stale"
            else "Finalize: promote tracked labels to cell_labels.tif (final, downstream-stable output)."
        )
        for button in buttons:
            button.setEnabled(state != "missing")
            button.setToolTip(tooltip)

    # ================================================================
    # Standalone input/output pickers (only built/used when standalone)
    # ================================================================
    # Row building, browse handlers, and QSettings persistence come from
    # StandalonePathsMixin; this widget only declares its fields and the
    # piece-specific step that pushes them into the workspace.
    _SETTINGS_APP = "cellflow_segmentation"

    def _standalone_fields(self) -> dict[str, QLineEdit]:
        return {
            "foreground": self._foreground_edit,
            "contours": self._contours_edit,
            "nucleus": self._nucleus_edit,
            "output_dir": self._output_dir_edit,
        }

    def _apply_standalone_paths(self) -> None:
        """Push the picker fields into the standalone inputs and re-wire paths.

        ``_pos_dir`` is set to the output dir so the ``_pos_dir is None`` guards
        (run, preview, correction) pass once an output folder is chosen; the
        input/output path methods resolve to the explicit files in standalone.
        """
        def _val(edit: QLineEdit) -> Path | None:
            text = edit.text().strip()
            return Path(text) if text else None

        self._sa_foreground = _val(self._foreground_edit)
        self._sa_contours = _val(self._contours_edit)
        self._sa_nucleus = _val(self._nucleus_edit)
        self._sa_output_dir = _val(self._output_dir_edit)
        self._save_standalone_settings()
        self.refresh(self._sa_output_dir)

    def _load_standalone_settings(self) -> None:
        self._load_path_settings(self._SETTINGS_APP, self._standalone_fields())
        self._apply_standalone_paths()

    def _save_standalone_settings(self) -> None:
        self._save_path_settings(self._SETTINGS_APP, self._standalone_fields())

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
                "balance": self.balance_spin.value(),
                "feature_strength": self.feature_strength_spin.value(),
                "n_workers": self.n_workers_spin.value(),
            },
            "preview": {
                "foreground": self.fg_check.isChecked(),
                "contours": self.contour_check.isChecked(),
                "cost": self.cost_check.isChecked(),
                "labels": self.labels_check.isChecked(),
            },
            "correction": {
                "hole_radius": self.hole_radius_spin.value(),
                "scope": self.correction_scope_combo.currentText(),
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
                "balance": self.balance_spin,
                "feature_strength": self.feature_strength_spin,
                "n_workers": self.n_workers_spin,
            },
            "correction": {
                "hole_radius": self.hole_radius_spin,
            },
        }
        for group_key, widgets in _map.items():
            group = state.get(group_key, {})
            if not isinstance(group, dict):
                continue
            for key, spin in widgets.items():
                if key in group:
                    spin.setValue(group[key])
        preview = state.get("preview", {})
        if isinstance(preview, dict):
            for key, check in (
                ("foreground", self.fg_check),
                ("contours", self.contour_check),
                ("cost", self.cost_check),
                ("labels", self.labels_check),
            ):
                if key in preview:
                    check.setChecked(bool(preview[key]))
        correction = state.get("correction", {})
        if isinstance(correction, dict) and "scope" in correction:
            idx = self.correction_scope_combo.findText(correction["scope"])
            if idx >= 0:
                self.correction_scope_combo.setCurrentIndex(idx)

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

    @staticmethod
    def _shape_of(path):
        """(T, Y, X) from a TIFF header (no pixel load); None if absent."""
        if path is None or not path.exists():
            return None
        with tifffile.TiffFile(str(path)) as tf:
            n_frames = len(tf.pages)
            y, x = tf.pages[0].shape[-2], tf.pages[0].shape[-1]
        return int(n_frames), int(y), int(x)

    def _map_shape(self):
        """(T, Y, X) from the contours TIFF header (no pixel load)."""
        return self._shape_of(self._contours_path())

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
        # The live preview owns the viewer; the gate derives cross-section
        # exclusivity from this claim.
        if checked:
            self.gate.claim_viewer("cell_preview")
            self._refresh_preview()
        else:
            self.gate.release_viewer("cell_preview")
            self._preview_pending = False
            for name in _ALL_PREVIEW_LAYERS:
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)
            self._image_needs_autocontrast.clear()
            # Free the resident caches; the preview session is over.
            self._smoothed_stack = None
            self._smoothed_key = None
            self._frame_cache = {}
            self._frame_cache_key = None
            self._set_status("")

    def _on_compute_toggled(self, group_layers, checked: bool) -> None:
        """Handle a Compute-checkbox toggle: drop hidden groups, refresh shown.

        Unticking a group removes its layers immediately. Then, while the
        preview is active, a refresh creates/paints any newly-ticked group —
        computing only if the cached frame doesn't already cover the new level,
        so re-ticking a group that was just computed repaints instantly.
        """
        if not checked:
            for name in group_layers:
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)
                self._image_needs_autocontrast.discard(name)
        if self._preview_active:
            self._refresh_preview()

    @staticmethod
    def _smooth_key(params: CellDivergenceParams) -> tuple:
        """Cache key for the smoothed contour stack: the knobs that determine it.

        The fg_*, balance and feature_strength knobs are deliberately absent —
        they do not change the smoothed contours, so editing them reuses the
        cached stack.
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

    def _desired_level(self) -> int:
        """The single-frame compute level needed to fill every ticked group."""
        level = _LEVEL_NONE
        for check, _layers, need in self._compute_groups:
            if check.isChecked():
                level = max(level, need)
        return level

    def _checked_layer_names(self) -> list[str]:
        """Layer names owned by the currently-ticked Compute checkboxes."""
        names: list[str] = []
        for check, layers, _need in self._compute_groups:
            if check.isChecked():
                names.extend(layers)
        return names

    def _preview_inputs(self):
        """Validate the inputs a single-frame compute needs.

        Sets a status message and returns ``None`` on any failure; otherwise
        returns ``(shape, contours_path, foreground_path, nuc_path)``. The
        foreground fill mask is derived in-process (no disk prerequisite).
        """
        if not self._preview_active:
            return None
        if not self._maps_present():
            self._set_status("Cell divergence maps not found — run Divergence Maps first.")
            return None
        nuc_path = self._nuc_path()
        if nuc_path is None or not nuc_path.exists():
            self._set_status(self._nuc_missing_msg())
            return None
        shape = self._map_shape()
        if shape is None:
            self._set_status("Cell divergence maps not found — run Divergence Maps first.")
            return None
        return shape, self._contours_path(), self._foreground_path(), nuc_path

    def _compute_frame(
        self, *, t, level, params, contours_path, foreground_path, nuc_path,
        cached_stack,
    ):
        """Worker-thread body: compute frame ``t`` up to ``level`` only.

        ``_LEVEL_FG`` runs the cheap foreground-only path (no contour read);
        ``_LEVEL_SEG``/``_LEVEL_LABELS`` run the full divergence pipeline,
        deriving the fill mask in-process and (for the geodesic) the labels.
        When ``memory_tau > 0`` the cleaned + smoothed contour stack (whole
        movie) feeds the frame through ``contours_clean_override`` so the result
        matches the full run; the stack is taken from ``cached_stack`` when valid,
        otherwise computed here and returned for the caller to cache. Returns
        ``(t, result, new_stack)`` where ``new_stack`` is non-``None`` only when
        it was (re)computed.
        """
        fg = self._read_frame(foreground_path, t)[np.newaxis]
        nuc = self._read_frame(nuc_path, t, dtype=np.uint32)[np.newaxis]

        if level <= _LEVEL_FG:
            result = compute_cell_foreground(fg, params, nuc, frame=0)
            return t, result, None

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
            contour, fg, nuc, params, frame=0,
            with_labels=level >= _LEVEL_LABELS,
            contours_clean_override=override,
        )
        return t, result, new_stack

    def _refresh_preview(self):
        """Repaint the current frame's preview, computing only what's missing.

        Clears the per-frame cache when the params changed, then either repaints
        the current frame from the cache (when it already covers every ticked
        group's level) or starts a worker for the shortfall. While a pass is in
        flight, further edits arm ``_preview_pending`` so one fresh pass (latest
        params/frame) fires when the current one returns. Returns the started
        worker, or ``None`` when nothing needed computing.
        """
        info = self._preview_inputs()
        if info is None:
            return None
        shape, contours_path, foreground_path, nuc_path = info

        params = self._params()
        key = dataclasses.astuple(params)
        if key != self._frame_cache_key:
            self._frame_cache = {}
            self._frame_cache_key = key
        if params.memory_tau <= 0.0:
            # Smoothing off — release the resident whole-movie stack.
            self._smoothed_stack = None
            self._smoothed_key = None

        level = self._desired_level()
        self._ensure_preview_layers(shape, self._checked_layer_names())
        if level == _LEVEL_NONE:
            self._set_status("")
            return None

        n_frames = shape[0]
        t = max(0, min(self._current_t(), n_frames - 1))

        cached = self._frame_cache.get(t)
        if cached is not None and cached[0] >= level:
            # Already computed to (at least) the needed level — instant repaint.
            self._apply_result(t, cached[1])
            self._set_status(self._frame_status(t, cached[1]))
            return None

        if self._preview_worker is not None:
            self._preview_pending = True
            return self._preview_worker

        need_stack = params.memory_tau > 0.0 and level >= _LEVEL_SEG
        cached_stack = self._cached_stack_for(params) if need_stack else None
        if need_stack and cached_stack is None:
            self._set_status(f"Temporal smoothing over {n_frames} frames…")
        else:
            self._set_status(f"Computing cell preview for frame {t}…")

        @thread_worker(connect={
            "returned": self._on_preview_done,
            "errored": self._on_preview_error,
        })
        def _worker():
            t_, result, new_stack = self._compute_frame(
                t=t, level=level, params=params, contours_path=contours_path,
                foreground_path=foreground_path, nuc_path=nuc_path,
                cached_stack=cached_stack,
            )
            return t_, result, params, new_stack, level

        self._preview_worker = _worker()
        return self._preview_worker

    def _on_preview_done(self, payload) -> None:
        self._preview_worker = None
        t, result, params, new_stack, level = payload
        self._cache_stack(params, new_stack)
        # Only keep the result if the params still match (an edit mid-flight
        # cleared and re-keyed the cache; a stale result must not poison it).
        if dataclasses.astuple(params) == self._frame_cache_key:
            self._frame_cache[t] = (level, result)
        if self._preview_active:
            self._apply_result(t, result)
            self._set_status(self._frame_status(t, result))
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

    def _cache_stack(self, params: CellDivergenceParams, new_stack) -> None:
        """Hold a freshly computed smoothed stack resident for later frames/edits."""
        if new_stack is not None:
            self._smoothed_stack = new_stack
            self._smoothed_key = self._smooth_key(params)

    def _apply_result(self, t: int, result) -> None:
        """Paint frame ``t``'s slice for every ticked group the result supplies.

        Works for both a ``CellForegroundResult`` (foreground only) and a full
        ``CellDivergenceResult``; missing fields (e.g. contours on a foreground
        result) are simply skipped — they are never requested unless the deeper
        compute that produces them ran.
        """
        if self.fg_check.isChecked():
            self._fill_image_layer(_FG_RAW_LAYER, t, result.foreground_raw)
            self._fill_image_layer(_FG_CLEAN_LAYER, t, result.foreground_clean)
            self._fill_labels_layer(
                _FG_MASK_LAYER, t, result.foreground_mask.astype(np.uint8)
            )
        if self.contour_check.isChecked() and hasattr(result, "contours_raw"):
            self._fill_image_layer(_CT_RAW_LAYER, t, result.contours_raw)
            self._fill_image_layer(_CT_CLEAN_LAYER, t, result.contours_clean)
        if self.cost_check.isChecked() and hasattr(result, "cost_field"):
            self._fill_image_layer(
                _COST_LAYER, t, self._cost_for_display(result.cost_field)
            )
        if self.labels_check.isChecked() and getattr(result, "labels", None) is not None:
            self._fill_labels_layer(_LABELS_LAYER, t, result.labels.astype(np.int32))

    @staticmethod
    def _frame_status(t: int, result) -> str:
        coverage = 100.0 * float(result.foreground_mask.mean())
        msg = f"Frame {t}: {coverage:.0f}% fill coverage"
        labels = getattr(result, "labels", None)
        if labels is not None:
            n_labels = int(np.unique(labels[labels > 0]).size)
            msg += f", {n_labels} cell labels"
        return msg

    @staticmethod
    def _cost_for_display(cost: np.ndarray) -> np.ndarray:
        """Mask the geodesic-cost background (inf) to NaN for the colormap."""
        return np.where(np.isfinite(cost), cost, np.nan).astype(np.float32)

    # ── preview layers (one full (T, Y, X) stack per intermediate) ────────
    # The preview only ever computes the current frame, but the layers are sized
    # to the whole input movie ``(T, Y, X)`` and painted one frame at a time.
    # Carrying the time axis is what gives the viewer a frame slider even when no
    # movie layer is open — otherwise ``current_step`` has no temporal entry and
    # the preview is stuck on (and mislabels) frame 0. A time scrub recomputes
    # the newly shown frame (``_on_time_changed``) and paints it into its slice;
    # previously computed frames stay painted in theirs. Only the layers for the
    # ticked Compute checkboxes are created.
    def _ensure_preview_layers(self, shape, names) -> None:
        for name in names:
            if name in _LABELS_KIND:
                self._ensure_labels_layer(name, shape)
            else:
                self._ensure_image_layer(name, shape, _IMAGE_COLORMAPS[name])

    def _ensure_image_layer(self, name: str, shape, colormap: str) -> None:
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if tuple(layer.data.shape) == tuple(shape):
                return
            was_visible = layer.visible
            self.viewer.layers.remove(name)
        else:
            was_visible = True
        new_layer = self.viewer.add_image(
            np.zeros(shape, dtype=np.float32), name=name, colormap=colormap,
        )
        new_layer.visible = was_visible
        # Seed this layer's contrast from the first real frame it receives.
        self._image_needs_autocontrast.add(name)

    def _ensure_labels_layer(self, name: str, shape) -> None:
        from napari.layers import Labels
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if isinstance(layer, Labels) and tuple(layer.data.shape) == tuple(shape):
                return
            was_visible = layer.visible
            self.viewer.layers.remove(name)
        else:
            was_visible = True
        new_layer = self.viewer.add_labels(
            np.zeros(shape, dtype=np.int32), name=name, opacity=0.55
        )
        new_layer.visible = was_visible

    def _fill_image_layer(self, name: str, t: int, frame: np.ndarray) -> None:
        if name not in self.viewer.layers:
            return
        layer = self.viewer.layers[name]
        if layer.data.ndim != 3 or not 0 <= t < layer.data.shape[0]:
            return
        layer.data[t] = np.asarray(frame, dtype=layer.data.dtype)
        if name in self._image_needs_autocontrast:
            finite = frame[np.isfinite(frame)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                if hi > lo:
                    layer.contrast_limits = (lo, hi)
                    self._image_needs_autocontrast.discard(name)
        layer.refresh()

    def _fill_labels_layer(self, name: str, t: int, frame: np.ndarray) -> None:
        if name not in self.viewer.layers:
            return
        layer = self.viewer.layers[name]
        if layer.data.ndim != 3 or not 0 <= t < layer.data.shape[0]:
            return
        layer.data[t] = np.asarray(frame, dtype=layer.data.dtype)
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
            self._set_status(self._nuc_missing_msg())
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
            self._refresh_finalize_btn()
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
        # ``self._running`` is set by the caller before this runs; the gate
        # derives active_btn enablement from it.
        self.gate.set_task("cell_run", True)

    def _set_run_idle(self) -> None:
        self.run_btn.setText("▶")
        self.run_btn.setToolTip(
            "Run the full pipeline over all frames and write tracked_labels.tif."
        )
        self.gate.set_task("cell_run", False)

    def _clear_progress(self) -> None:
        self.pipeline_progress_bar.setRange(0, 100)
        self.pipeline_progress_bar.setValue(0)
        self.pipeline_progress_bar.setVisible(False)


def make_cell_segmentation_widget(napari_viewer=None):
    """napari plugin factory for the standalone cell-segmentation piece.

    Used by the ``cellflow-segmentation`` distribution's manifest. Patches the
    napari layer-controls delegate (best-effort, normally done by the
    orchestrator) and returns the workflow widget in standalone mode, with its
    own foreground/contours/nucleus input pickers and output-dir picker.
    """
    try:
        from cellflow.napari._napari_compat import patch_napari_layer_delegate

        patch_napari_layer_delegate()
    except Exception:  # pragma: no cover - patch is best-effort
        pass
    # napari does not inject the viewer into function-based widget factories
    # (only into class-based callables / magicgui types), so ``napari_viewer``
    # arrives as ``None``. The widget needs a live viewer; fall back to the
    # active one.
    if napari_viewer is None:
        napari_viewer = napari.current_viewer()
    return CellWorkflowWidget(viewer=napari_viewer, standalone=True)
