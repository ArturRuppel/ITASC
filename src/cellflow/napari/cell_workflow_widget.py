"""Cell segmentation workflow widget for CellFlow.

Flat layout — action buttons in a two-column grid at top, one collapsible
parameter panel, correction section at bottom.

Stages:
  1. Flow Filtering → ``filtered_dp.tif``
  2. Foreground Masks → ``foreground_masks.tif``
  3. Contour Maps → ``contour_maps.tif``, ``foreground_scores.tif``
  4. Segmentation → ``tracked_labels.tif`` (initialize + auto-commit)
  5. Correction (load / save / fill holes / fix semiholes / cleanup / expand)
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,                                                     # ← NEW
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,                                                     # ← NEW
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.correction.labels import (
    best_overlapping_label,
    cleanup_movie,                                                 # ← NEW
    expand_label_to_foreground,
    fill_label_holes,                                              # ← NEW
    fix_label_semiholes,                                           # ← NEW
)
from cellflow.database.tracked import read_full_tracked_stack
from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.ui_style import (
    action_button,
    add_block_pair_row,
    add_parameter_grid_row,
    block_grid,
    compact_spinbox,
    status_label,
)
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.segmentation import (
    FlowFollowingParams,
    apply_gamma,
    build_consensus_boundary_flow_following,
)
from cellflow.segmentation.contour_filtering import contour_memory_filter

logger = logging.getLogger(__name__)

# ── Layer name constants ──────────────────────────────────────────────────────
_FILTERED_FLOW_LAYER = "Filtered Flow Magnitude"
_CELL_FOREGROUND_LAYER = "Foreground Mask: Cell"
_CELL_FOREGROUND_SCORE_LAYER = "Foreground Score: Cell"
_CELL_CONTOUR_LAYER = "Contour Map: Cell"
_CELL_SEG_LAYER = "Cell Segmentation"
_TRACKED_CELL_LAYER = "Tracked: Cell"
_CELL_ZAVG_LAYER = "Cell z-avg"
_NUC_ZAVG_LAYER = "Nucleus z-avg"


# ── Tiny helpers ──────────────────────────────────────────────────────────────

def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #555;")
    return line


def _heading(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: 600;")
    return lbl


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


def _dspin(lo, hi, val, step=0.1, decimals=2, tooltip=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
    s.setDecimals(decimals); s.setToolTip(tooltip)
    return s


def _ispin(lo, hi, val, step=1, tooltip=""):
    s = QSpinBox()
    s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
    s.setToolTip(tooltip)
    return s


def _btn(text, tooltip=""):
    b = QPushButton(text)
    b.setToolTip(tooltip)
    action_button(b, expand=True)
    return b


def _button_grid(*rows: tuple[QPushButton, ...]) -> QGridLayout:
    grid = QGridLayout()
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(4)
    for r, buttons in enumerate(rows):
        for c, btn in enumerate(buttons):
            span = 2 - c if c == len(buttons) - 1 and len(buttons) == 1 else 1
            grid.addWidget(btn, r, c, 1, span)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)
    return grid


# ══════════════════════════════════════════════════════════════════════════════


class CellWorkflowWidget(QWidget):
    """Cell segmentation pipeline — flat action-button layout."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None

        self._ff_worker = None
        self._foreground_worker = None
        self._contour_worker = None
        self._initialize_worker = None

        self._icm_state = None

        self._setup_ui()
        self._connect_signals()

    # ================================================================
    # UI
    # ================================================================
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)

        # ── Pipeline files (single deduplicated panel) ────────────────
        self._files_widget = PipelineFilesWidget(
            [
                ("Inputs", [
                    ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
                    ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
                    ("2_nucleus/tracked_labels.tif", "Nucleus tracked labels"),
                    ("0_input/cell_zavg.tif", "Cell z-avg"),
                    ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
                ]),
                ("Intermediates", [
                    ("3_cell/filtered_dp.tif", "Filtered flow vectors"),
                    ("3_cell/filtered_flow_mag.tif", "Filtered flow magnitude"),
                    ("3_cell/foreground_masks.tif", "Foreground masks"),
                    ("3_cell/contour_maps.tif", "Contour maps"),
                    ("3_cell/foreground_scores.tif", "Foreground scores"),
                ]),
                ("Output", [
                    ("3_cell/tracked_labels.tif", "Cell tracked labels"),
                ]),
            ],
            viewer=self.viewer,
        )
        root.addWidget(
            CollapsibleSection("Pipeline Files", self._files_widget, expanded=False)
        )

        # ── Pipeline action buttons (2-column grid) ──────────────────
        self.filter_flow_btn = _btn(
            "Filter Flow",
            "Apply median + Gaussian filtering to raw Cellpose flow vectors.",
        )
        self.build_foreground_btn = _btn(
            "Build Foreground",
            "Generate binary cell foreground masks with Cellpose dynamics.",
        )
        self.preview_contour_btn = _btn(
            "Preview Contours",
            "Build contour map for the current frame only (no temporal filter).",
        )
        self.build_contour_btn = _btn(
            "Build Contours",
            "Build consensus contour maps for all frames.",
        )
        self.segment_btn = _btn(
            "Segment",
            "Initialize geodesic ICM, solve, and save tracked_labels.tif.",
        )
        root.addLayout(_button_grid(
            (self.filter_flow_btn, self.build_foreground_btn),
            (self.preview_contour_btn, self.build_contour_btn),
            (self.segment_btn,),
        ))

        self.pipeline_status_lbl = _make_status()
        root.addWidget(self.pipeline_status_lbl)
        self.pipeline_progress_bar = _make_progress()
        root.addWidget(self.pipeline_progress_bar)

        # ── Single collapsible parameter panel ───────────────────────
        self._build_params_section(root)

        # ── Correction (group box for visual separation) ──────────── # ← CHANGED
        self._build_correction_section(root)

        root.addStretch()

    # -- Parameters --------------------------------------------------------

    def _build_params_section(self, root: QVBoxLayout) -> None:
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        # Flow filtering
        lay.addWidget(_heading("Flow Filtering"))
        g = block_grid(horizontal_spacing=12)
        self.ff_median_time_spin = _ispin(1, 15, 3)
        self.ff_median_space_spin = _ispin(1, 15, 5)
        self.ff_gauss_time_spin = _dspin(0, 10, 0, 0.1, 1)
        self.ff_gauss_space_spin = _dspin(0, 10, 0, 0.1, 1)
        add_block_pair_row(g, 0,
            "Median t:", compact_spinbox(self.ff_median_time_spin),
            "Median xy:", compact_spinbox(self.ff_median_space_spin))
        add_block_pair_row(g, 1,
            "Gauss t σ:", compact_spinbox(self.ff_gauss_time_spin),
            "Gauss xy σ:", compact_spinbox(self.ff_gauss_space_spin))
        lay.addLayout(g)

        # Foreground
        lay.addWidget(_heading("Foreground"))
        g = block_grid(horizontal_spacing=12)
        self.fg_cellprob_threshold_spin = _dspin(
            0, 1, 0.5, 0.01, 2,
            "Cellpose probability threshold (sigmoid space).",
        )
        add_block_pair_row(g, 0,
            "Cellprob thr:", compact_spinbox(self.fg_cellprob_threshold_spin))
        lay.addLayout(g)

        # Contour maps — consensus sweep
        lay.addWidget(_heading("Contour — Cellprob Sweep"))
        g = block_grid(horizontal_spacing=12)
        self.cp_min_spin = _dspin(0, 1, 0.05, 0.05)
        self.cp_max_spin = _dspin(0, 1, 0.50, 0.05)
        self.cp_step_spin = _dspin(0.01, 1, 0.05, 0.01)
        add_block_pair_row(g, 0,
            "Min:", compact_spinbox(self.cp_min_spin),
            "Max:", compact_spinbox(self.cp_max_spin))
        add_block_pair_row(g, 1,
            "Step:", compact_spinbox(self.cp_step_spin))
        lay.addLayout(g)

        # Contour — flow-following
        lay.addWidget(_heading("Contour — Flow-Following"))
        g = block_grid(horizontal_spacing=12)
        self.ff_flow_weight_spin = _dspin(
            0, 1, 0.5, 0.05, 2,
            "Blend: flow direction (1) vs EDT gravity (0).",
        )
        self.ff_step_scale_spin = _dspin(0.01, 2, 0.2, 0.05, 2, "Step-size multiplier.")
        self.ff_max_iter_spin = _ispin(10, 2000, 100, 10, "Max integration steps.")
        add_block_pair_row(g, 0,
            "Flow weight:", compact_spinbox(self.ff_flow_weight_spin),
            "Step scale:", compact_spinbox(self.ff_step_scale_spin))
        add_block_pair_row(g, 1,
            "Max iter:", compact_spinbox(self.ff_max_iter_spin))
        lay.addLayout(g)

        # Contour — gamma averaging
        lay.addWidget(_heading("Contour — Gamma Averaging"))
        g = block_grid(horizontal_spacing=12)
        self.gamma_min_spin = _dspin(0.05, 5, 1.0, 0.05)
        self.gamma_max_spin = _dspin(0.05, 5, 1.0, 0.05)
        self.gamma_step_spin = _dspin(0.05, 2, 0.25, 0.05)
        add_block_pair_row(g, 0,
            "Min:", compact_spinbox(self.gamma_min_spin),
            "Max:", compact_spinbox(self.gamma_max_spin))
        add_block_pair_row(g, 1,
            "Step:", compact_spinbox(self.gamma_step_spin))
        lay.addLayout(g)

        # Contour — temporal stabilization
        lay.addWidget(_heading("Contour — Temporal Stabilization"))
        g = block_grid(horizontal_spacing=12)
        self.memory_tau_spin = _dspin(
            0, 1, 0, 0.01, 3,
            "Contour memory τ. 0 = disabled.",
        )
        self.memory_floor_spin = _dspin(
            0.001, 0.5, 0.01, 0.005, 3,
            "Minimum alpha — prevents permanent ghosting.",
        )
        add_block_pair_row(g, 0,
            "Memory τ:", compact_spinbox(self.memory_tau_spin),
            "Floor:", compact_spinbox(self.memory_floor_spin))
        lay.addLayout(g)

        # Segmentation — ICM
        lay.addWidget(_heading("Segmentation"))
        g = block_grid(horizontal_spacing=12)
        self.alpha_unary_spin = _dspin(
            0, 1000, 4.0, 0.1, 2,
            "Contour weight: 1 + α·contour.",
        )
        self.lambda_s_spin = _dspin(0, 1000, 1.0, 0.1, 2, "Spatial Potts weight.")
        self.beta_s_spin = _dspin(
            0, 1000, 5.0, 0.1, 2,
            "Contour sensitivity: exp(-β·avg_contour).",
        )
        self.lambda_t_spin = _dspin(0, 1000, 1.0, 0.1, 2, "Temporal Potts weight.")
        self.gamma_unary_spin = _dspin(
            0, 100, 0, 0.1, 2,
            "(1 − foreground_score) weight. 0 = contour-only.",
        )
        self.n_workers_spin = _ispin(
            1, max(1, os.cpu_count() or 1),
            min(4, os.cpu_count() or 1),
            tooltip="Parallel workers for geodesic computation.",
        )
        add_block_pair_row(g, 0,
            "α unary:", compact_spinbox(self.alpha_unary_spin),
            "λ spatial:", compact_spinbox(self.lambda_s_spin),
            field_width=92)
        add_block_pair_row(g, 1,
            "β spatial:", compact_spinbox(self.beta_s_spin),
            "λ temporal:", compact_spinbox(self.lambda_t_spin),
            field_width=92)
        add_block_pair_row(g, 2,
            "γ unary:", compact_spinbox(self.gamma_unary_spin),
            "Workers:", compact_spinbox(self.n_workers_spin),
            field_width=92)
        lay.addLayout(g)

        # ── NOTE: "Correction" params removed from here ──────────── # ← CHANGED

        root.addWidget(CollapsibleSection("Parameters", inner, expanded=False))

    # -- Correction --------------------------------------------------------  # ← REWRITTEN

    def _build_correction_section(self, root: QVBoxLayout) -> None:
        group = QGroupBox("Correction")
        group.setStyleSheet(
            "QGroupBox { font-weight: 600; margin-top: 8px; padding-top: 14px; }"
        )
        group_lay = QVBoxLayout(group)
        group_lay.setContentsMargins(8, 16, 8, 8)
        group_lay.setSpacing(6)

        # ── Action buttons — 2-column grid ────────────────────────
        self.load_labels_btn = _btn(
            "Load Labels", "Load tracked cell labels from disk.")
        self.save_labels_btn = _btn(
            "Save Labels", "Save tracked cell labels to disk.")
        self.fill_holes_btn = _btn(
            "Fill Holes",
            "Fill background holes fully enclosed within individual labels.",
        )
        self.fix_semiholes_btn = _btn(
            "Fix Semi Holes",
            "Bridge narrow channels in label boundaries and fill the pockets.",
        )
        self.cleanup_btn = _btn(
            "Clean Up",
            "All frames: clean fragments → resync to nuclear labels → remove orphans.",
        )
        self.expand_cell_btn = _btn(
            "Expand Cell",
            "Expand selected cell into adjacent foreground mask pixels.",
        )
        group_lay.addLayout(_button_grid(
            (self.load_labels_btn, self.save_labels_btn),
            (self.fill_holes_btn, self.fix_semiholes_btn),
            (self.cleanup_btn, self.expand_cell_btn),
        ))

        self.correction_status_lbl = _make_status()
        group_lay.addWidget(self.correction_status_lbl)

        # ── Correction parameters ─────────────────────────────────
        scope_row = QHBoxLayout()
        scope_lbl = QLabel("Scope:")
        scope_lbl.setToolTip("Applies to Fill Holes and Fix Semi Holes.")
        scope_row.addWidget(scope_lbl)
        self.correction_scope_combo = QComboBox()
        self.correction_scope_combo.addItems(["Current frame", "All frames"])
        self.correction_scope_combo.setToolTip(
            "Applies to Fill Holes and Fix Semi Holes. Clean Up always processes all frames."
        )
        scope_row.addWidget(self.correction_scope_combo)
        group_lay.addLayout(scope_row)

        g = block_grid(horizontal_spacing=12)
        self.hole_radius_spin = _ispin(
            0, 999, 5,
            tooltip="Max hole size (pixels) for fill / fix operations.",
        )
        self.semihole_opening_spin = _ispin(
            0, 999, 3,
            tooltip="Max channel width for semi-hole bridging.",
        )
        self.expand_max_px_spin = _ispin(
            0, 999, 25,
            tooltip="Max expansion distance in pixels.",
        )
        add_block_pair_row(g, 0,
            "Hole radius:", compact_spinbox(self.hole_radius_spin),
            "Max opening:", compact_spinbox(self.semihole_opening_spin))
        add_block_pair_row(g, 1,
            "Max expand px:", compact_spinbox(self.expand_max_px_spin))
        group_lay.addLayout(g)

        # ── Inline CorrectionWidget (no cleanup, no spotlight) ────
        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
            spotlight=False,                                       # ← NEW
            show_cleanup=False,                                    # ← NEW
        )
        group_lay.addWidget(self.correction_widget)

        group_lay.addWidget(CollapsibleSection(
            "Correction Shortcuts",
            self.correction_widget.build_shortcuts_widget(),
            expanded=False,
        ))

        root.addWidget(group)

    # ================================================================
    # Signals
    # ================================================================
    def _connect_signals(self) -> None:
        self.filter_flow_btn.clicked.connect(self._on_filter_flow)
        self.build_foreground_btn.clicked.connect(self._on_build_foreground)
        self.preview_contour_btn.clicked.connect(self._on_preview_contours)
        self.build_contour_btn.clicked.connect(self._on_build_contours)
        self.segment_btn.clicked.connect(self._on_segment)
        self.load_labels_btn.clicked.connect(self._on_load_labels)
        self.save_labels_btn.clicked.connect(self._on_save_labels)
        self.fill_holes_btn.clicked.connect(self._on_fill_holes)               # ← NEW
        self.fix_semiholes_btn.clicked.connect(self._on_fix_semiholes)         # ← NEW
        self.cleanup_btn.clicked.connect(self._on_cleanup)                     # ← NEW
        self.expand_cell_btn.clicked.connect(self._on_expand_cell)
        # NOTE: reassign_ids_btn removed                                       # ← CHANGED

    # ================================================================
    # Path helpers
    # ================================================================
    def _p(self, *parts: str) -> Path | None:
        return self._pos_dir.joinpath(*parts) if self._pos_dir else None

    def _prob_path(self):          return self._p("1_cellpose", "cell_prob_3dt.tif")
    def _dp_path(self):            return self._p("1_cellpose", "cell_dp_3dt.tif")
    def _filtered_dp_path(self):   return self._p("3_cell", "filtered_dp.tif")
    def _flow_mag_path(self):      return self._p("3_cell", "filtered_flow_mag.tif")
    def _foreground_path(self):    return self._p("3_cell", "foreground_masks.tif")
    def _contour_path(self):       return self._p("3_cell", "contour_maps.tif")
    def _fg_scores_path(self):     return self._p("3_cell", "foreground_scores.tif")
    def _nuc_labels_path(self):    return self._p("2_nucleus", "tracked_labels.tif")
    def _cell_labels_path(self):   return self._p("3_cell", "tracked_labels.tif")
    def _cell_zavg_path(self):     return self._p("0_input", "cell_zavg.tif")
    def _nuc_zavg_path(self):      return self._p("0_input", "nucleus_zavg.tif")

    def _require(self, *pairs: tuple[Path | None, str]) -> bool:
        for path, name in pairs:
            if path is None or not path.exists():
                self._status(f"Missing: {name}")
                return False
        return True

    # ================================================================
    # Public API
    # ================================================================
    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._icm_state = None
        self._files_widget.refresh(pos_dir)
        if pos_dir is None:
            self.correction_widget.deactivate()

    def get_state(self) -> dict:
        return {
            "flow_filtering": {
                "median_time": self.ff_median_time_spin.value(),
                "median_space": self.ff_median_space_spin.value(),
                "gauss_time": self.ff_gauss_time_spin.value(),
                "gauss_space": self.ff_gauss_space_spin.value(),
            },
            "foreground": {
                "cellprob_threshold": self.fg_cellprob_threshold_spin.value(),
            },
            "contour": {
                "cp_min": self.cp_min_spin.value(),
                "cp_max": self.cp_max_spin.value(),
                "cp_step": self.cp_step_spin.value(),
                "gamma_min": self.gamma_min_spin.value(),
                "gamma_max": self.gamma_max_spin.value(),
                "gamma_step": self.gamma_step_spin.value(),
                "ff_flow_weight": self.ff_flow_weight_spin.value(),
                "ff_step_scale": self.ff_step_scale_spin.value(),
                "ff_max_iter": self.ff_max_iter_spin.value(),
                "memory_tau": self.memory_tau_spin.value(),
                "memory_floor": self.memory_floor_spin.value(),
            },
            "segmentation": {
                "alpha_unary": self.alpha_unary_spin.value(),
                "lambda_s": self.lambda_s_spin.value(),
                "beta_s": self.beta_s_spin.value(),
                "lambda_t": self.lambda_t_spin.value(),
                "gamma_unary": self.gamma_unary_spin.value(),
                "n_workers": self.n_workers_spin.value(),
            },
            "correction": {                                        # ← CHANGED
                "expand_max_px": self.expand_max_px_spin.value(),
                "hole_radius": self.hole_radius_spin.value(),
                "semihole_opening": self.semihole_opening_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if "flow_following" in state and "foreground" not in state:
            state = self._migrate_legacy_state(state)
        _map = {
            "flow_filtering": {
                "median_time": self.ff_median_time_spin,
                "median_space": self.ff_median_space_spin,
                "gauss_time": self.ff_gauss_time_spin,
                "gauss_space": self.ff_gauss_space_spin,
            },
            "foreground": {
                "cellprob_threshold": self.fg_cellprob_threshold_spin,
            },
            "contour": {
                "cp_min": self.cp_min_spin,
                "cp_max": self.cp_max_spin,
                "cp_step": self.cp_step_spin,
                "gamma_min": self.gamma_min_spin,
                "gamma_max": self.gamma_max_spin,
                "gamma_step": self.gamma_step_spin,
                "ff_flow_weight": self.ff_flow_weight_spin,
                "ff_step_scale": self.ff_step_scale_spin,
                "ff_max_iter": self.ff_max_iter_spin,
                "memory_tau": self.memory_tau_spin,
                "memory_floor": self.memory_floor_spin,
            },
            "segmentation": {
                "alpha_unary": self.alpha_unary_spin,
                "lambda_s": self.lambda_s_spin,
                "beta_s": self.beta_s_spin,
                "lambda_t": self.lambda_t_spin,
                "gamma_unary": self.gamma_unary_spin,
                "n_workers": self.n_workers_spin,
            },
            "correction": {                                        # ← CHANGED
                "expand_max_px": self.expand_max_px_spin,
                "hole_radius": self.hole_radius_spin,
                "semihole_opening": self.semihole_opening_spin,
            },
        }
        for group_key, widgets in _map.items():
            group = state.get(group_key, {})
            if not isinstance(group, dict):
                continue
            for k, w in widgets.items():
                if k in group:
                    w.setValue(group[k])

    @staticmethod
    def _migrate_legacy_state(state: dict) -> dict:
        new: dict = {}
        ff = state.get("flow_following", {})
        if ff:
            new["flow_filtering"] = dict(ff)
        seg = state.get("segmentation", {})
        if not seg:
            return new
        new["foreground"] = {}
        for k in ("fg_cellprob_threshold", "cellprob_threshold"):
            if k in seg:
                new["foreground"]["cellprob_threshold"] = seg[k]
        new["contour"] = {
            k: v for k, v in seg.items()
            if k.startswith(("cp_", "ff_", "memory_"))
        }
        for old, new_k in [
            ("cp_gamma_min", "gamma_min"),
            ("cp_gamma_max", "gamma_max"),
            ("cp_gamma_step", "gamma_step"),
        ]:
            if old in new["contour"]:
                new["contour"][new_k] = new["contour"].pop(old)
        new["segmentation"] = {
            k: v for k, v in seg.items()
            if k in {"alpha_unary", "lambda_s", "beta_s",
                      "lambda_t", "gamma_unary", "n_workers"}
        }
        return new

    def set_selection_callback(self, fn) -> None:
        self.correction_widget.set_selection_callback(fn)

    def select_matching_cell_label(
        self, t: int, source_label: int,
        *, source_labels: np.ndarray | None = None,
    ) -> None:
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            return
        if source_labels is None:
            if "Tracked: Nucleus" not in self.viewer.layers:
                return
            source_labels = np.asarray(self.viewer.layers["Tracked: Nucleus"].data)
        target = np.asarray(self.viewer.layers[_TRACKED_CELL_LAYER].data)
        matched = best_overlapping_label(target, source_labels, t, source_label)
        self.correction_widget.select_label(t, matched, notify=False)

    # ================================================================
    # Status / layer helpers
    # ================================================================
    def _status(self, msg: str) -> None:
        self.pipeline_status_lbl.setText(msg)
        self.pipeline_status_lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    def _correction_status(self, msg: str) -> None:
        self.correction_status_lbl.setText(msg)
        self.correction_status_lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    def _progress(self, done: int, total: int, msg: str) -> None:
        self.pipeline_progress_bar.setVisible(True)
        self.pipeline_progress_bar.setRange(0, total)
        self.pipeline_progress_bar.setValue(done)
        self._status(msg)

    def _on_progress(self, data) -> None:
        if isinstance(data, tuple):
            self._progress(*data)
        else:
            self._status(str(data))

    def _clear_progress(self) -> None:
        self.pipeline_progress_bar.setValue(0)
        self.pipeline_progress_bar.setVisible(False)

    def _set_pipeline_buttons_enabled(self, enabled: bool) -> None:
        for btn in (
            self.filter_flow_btn,
            self.build_foreground_btn,
            self.preview_contour_btn,
            self.build_contour_btn,
            self.segment_btn,
        ):
            btn.setEnabled(enabled)

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

    def _current_time_index(self, max_t: int) -> int:
        return min(max(self._current_t(), 0), max(max_t - 1, 0))

    # ================================================================
    # Correction — frame index helper
    # ================================================================
    def _correction_frame_indices(self, layer) -> list[int]:          # ← NEW
        """Return frame indices based on the correction scope combo."""
        if layer.data.ndim < 3:
            return [0]
        if self.correction_scope_combo.currentText() == "All frames":
            return list(range(int(layer.data.shape[0])))
        return [self._current_t()]

    # ================================================================
    # 1. Flow Filtering
    # ================================================================
    def _flow_filter_params(self) -> FlowFollowingParams:
        return FlowFollowingParams(
            median_kernel_time=int(self.ff_median_time_spin.value()),
            median_kernel_space=int(self.ff_median_space_spin.value()),
            gaussian_sigma_time=float(self.ff_gauss_time_spin.value()),
            gaussian_sigma_space=float(self.ff_gauss_space_spin.value()),
        )

    def _read_dp_tcyx(self, prob_path: Path, dp_path: Path) -> np.ndarray:
        from cellflow.segmentation._array_utils import normalize_seeded_watershed_dp_stack
        prob = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
        if prob.ndim == 3:
            prob = prob[np.newaxis]
        dp_raw = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
        dp_full = normalize_seeded_watershed_dp_stack(dp_raw, prob.shape)
        return dp_full[:, :, :2].mean(axis=1).astype(np.float32)

    def _on_filter_flow(self) -> None:
        if self._pos_dir is None:
            self._status("No project open."); return
        prob_path, dp_path = self._prob_path(), self._dp_path()
        fdp, fmag = self._filtered_dp_path(), self._flow_mag_path()
        if not self._require(
            (prob_path, "cell_prob_3dt.tif"),
            (dp_path, "cell_dp_3dt.tif"),
        ):
            return

        params = self._flow_filter_params()
        pos_dir = self._pos_dir

        def _done(result):
            self._ff_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            self._show_layer(
                _FILTERED_FLOW_LAYER, result,
                {"colormap": "inferno", "blending": "additive"},
                self.viewer.add_image,
            )
            self._files_widget.refresh(pos_dir)
            self._status("Flow filtering complete.")

        def _error(exc):
            self._ff_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            self._status(f"Error: {exc}")
            logger.exception("Flow filter error", exc_info=exc)

        @thread_worker(connect={"yielded": self._on_progress, "returned": _done, "errored": _error})
        def _worker():
            from cellflow.segmentation import compute_filtered_flow_vectors
            yield (0, 4, "Loading flow inputs...")
            dp_tcyx = self._read_dp_tcyx(prob_path, dp_path)
            yield (1, 4, "Filtering...")
            filtered_dp = compute_filtered_flow_vectors(dp_tcyx, params)
            yield (2, 4, "Computing magnitude...")
            mag = np.sqrt(filtered_dp[:, 0]**2 + filtered_dp[:, 1]**2).astype(np.float32)
            yield (3, 4, "Saving...")
            fdp.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(fdp), filtered_dp, compression="zlib")
            tifffile.imwrite(str(fmag), mag, compression="zlib")
            return mag

        self._status("Filtering flow...")
        self._set_pipeline_buttons_enabled(False)
        self._ff_worker = _worker()

    # ================================================================
    # 2. Foreground Masks
    # ================================================================
    def _on_build_foreground(self) -> None:
        if self._pos_dir is None:
            self._status("No project open."); return
        prob_path, fdp = self._prob_path(), self._filtered_dp_path()
        fg_path = self._foreground_path()
        if not self._require(
            (prob_path, "cell_prob_3dt.tif"),
            (fdp, "filtered_dp.tif"),
        ):
            return

        thr = self.fg_cellprob_threshold_spin.value()
        pos_dir = self._pos_dir

        def _done(result):
            self._foreground_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            self._show_layer(_CELL_FOREGROUND_LAYER, result, {}, self.viewer.add_labels)
            self._files_widget.refresh(pos_dir)
            self._status("Foreground masks complete.")

        def _error(exc):
            self._foreground_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            self._status(f"Error: {exc}")
            logger.exception("Foreground error", exc_info=exc)

        @thread_worker(connect={"yielded": self._on_progress, "returned": _done, "errored": _error})
        def _worker():
            from cellflow.segmentation.cell_foreground import compute_cellpose_foreground_masks
            yield (0, 1, "Loading inputs...")
            prob = tifffile.imread(str(prob_path))
            dp = tifffile.imread(str(fdp))
            if prob.ndim == 3: prob = prob[np.newaxis]
            if dp.ndim == 3: dp = dp[np.newaxis]
            T = prob.shape[0]
            yield (0, T, f"Building foreground (T={T})...")
            masks = compute_cellpose_foreground_masks(
                prob, dp, cellprob_threshold=thr,
                flow_threshold=0.0, min_size=15, niter=200,
                progress_cb=lambda d, t: None,
            )
            fg_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(fg_path), masks, compression="zlib")
            return masks

        self._status("Building foreground...")
        self._set_pipeline_buttons_enabled(False)
        self._foreground_worker = _worker()

    # ================================================================
    # 3. Contour Maps
    # ================================================================
    def _cellprob_thresholds(self) -> list[float]:
        step = self.cp_step_spin.value()
        return list(np.arange(self.cp_min_spin.value(), self.cp_max_spin.value() + step / 2, step))

    def _gammas(self) -> list[float]:
        step = self.gamma_step_spin.value()
        return list(np.arange(self.gamma_min_spin.value(), self.gamma_max_spin.value() + step / 2, step))

    def _contour_ff_params(self) -> FlowFollowingParams:
        return FlowFollowingParams(
            median_kernel_time=1, median_kernel_space=1,
            gaussian_sigma_time=0.0, gaussian_sigma_space=0.0,
            flow_weight=self.ff_flow_weight_spin.value(),
            flow_step_scale=self.ff_step_scale_spin.value(),
            max_iterations=self.ff_max_iter_spin.value(),
        )

    def _consensus_boundary_averaged(
        self, prob_3d, dp_2d, labels_yx, thresholds, gammas, *, ff_params,
    ) -> tuple[np.ndarray, np.ndarray]:
        b_acc = fg_acc = None
        n = 0
        for gamma in gammas:
            logits = apply_gamma(prob_3d, gamma)
            probs = 1.0 / (1.0 + np.exp(-logits))
            prob_2d = probs.mean(axis=0).astype(np.float32)
            b, fg = build_consensus_boundary_flow_following(
                prob_2d, dp_2d, labels_yx, thresholds,
                params=ff_params, reduction="mean",
            )
            if b_acc is None:
                b_acc, fg_acc = b.copy(), fg.copy()
            else:
                b_acc += b; fg_acc += fg
            n += 1
        if n > 0:
            b_acc /= n; fg_acc /= n
        return b_acc, fg_acc

    def _on_build_contours(self) -> None:
        if self._pos_dir is None:
            self._status("No project open."); return
        prob_path, fdp = self._prob_path(), self._filtered_dp_path()
        nuc_path = self._nuc_labels_path()
        ct_path, sc_path = self._contour_path(), self._fg_scores_path()
        if not self._require(
            (prob_path, "cell_prob_3dt.tif"),
            (fdp, "filtered_dp.tif"),
            (nuc_path, "tracked_labels.tif (nucleus)"),
        ):
            return

        thresholds = self._cellprob_thresholds()
        gammas = self._gammas()
        tau = self.memory_tau_spin.value()
        floor = self.memory_floor_spin.value()
        ff_params = self._contour_ff_params()
        nuc_labels = tifffile.imread(str(nuc_path))
        pos_dir = self._pos_dir

        def _done(result):
            self._contour_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            contours, scores = result
            self._show_layer(_CELL_CONTOUR_LAYER, contours,
                             {"colormap": "magma", "visible": True}, self.viewer.add_image)
            self._show_layer(_CELL_FOREGROUND_SCORE_LAYER, scores,
                             {"colormap": "viridis", "visible": True}, self.viewer.add_image)
            self._files_widget.refresh(pos_dir)
            self._status("Contour maps complete.")

        def _error(exc):
            self._contour_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            self._status(f"Error: {exc}")
            logger.exception("Contour error", exc_info=exc)

        @thread_worker(connect={"yielded": self._on_progress, "returned": _done, "errored": _error})
        def _worker():
            prob_stack = tifffile.imread(str(prob_path))
            dp_stack = tifffile.imread(str(fdp))
            if prob_stack.ndim == 3: prob_stack = prob_stack[np.newaxis]
            if dp_stack.ndim == 3: dp_stack = dp_stack[np.newaxis]
            T = prob_stack.shape[0]
            cm = np.zeros((T, *prob_stack.shape[2:]), dtype=np.float32)
            fs = np.zeros_like(cm)
            for t in range(T):
                yield (t + 1, T, f"Contour maps: frame {t+1}/{T}...")
                b, fg = self._consensus_boundary_averaged(
                    prob_stack[t], dp_stack[t], nuc_labels[t],
                    thresholds, gammas, ff_params=ff_params,
                )
                cm[t], fs[t] = b, fg
            if tau > 0 and T > 1:
                yield (T, T, f"Memory filter (τ={tau})...")
                cm = contour_memory_filter(cm, tau=tau, floor=floor)
            ct_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(ct_path), cm, compression="zlib")
            tifffile.imwrite(str(sc_path), fs, compression="zlib")
            return cm, fs

        tau_msg = f", τ={tau}" if tau > 0 else ""
        self._status(f"Building contours ({len(thresholds)} thr, {len(gammas)} γ{tau_msg})...")
        self._set_pipeline_buttons_enabled(False)
        self._contour_worker = _worker()

    def _on_preview_contours(self) -> None:
        t = self._current_t()
        prob_path, fdp = self._prob_path(), self._filtered_dp_path()
        nuc_path = self._nuc_labels_path()
        if not self._require(
            (prob_path, "cell_prob_3dt.tif"),
            (fdp, "filtered_dp.tif"),
            (nuc_path, "tracked_labels.tif (nucleus)"),
        ):
            return

        prob_stack = tifffile.imread(str(prob_path))
        if prob_stack.ndim == 3: prob_stack = prob_stack[np.newaxis]
        dp_stack = tifffile.imread(str(fdp))
        if dp_stack.ndim == 3: dp_stack = dp_stack[np.newaxis]
        nuc_t = tifffile.imread(str(nuc_path))[t]

        b, fg = self._consensus_boundary_averaged(
            prob_stack[t].astype(np.float32),
            dp_stack[t].astype(np.float32),
            nuc_t,
            self._cellprob_thresholds(), self._gammas(),
            ff_params=self._contour_ff_params(),
        )
        n_t = prob_stack.shape[0]
        cd = np.zeros((n_t,) + b.shape, dtype=np.float32); cd[t] = b
        sd = np.zeros((n_t,) + fg.shape, dtype=np.float32); sd[t] = fg
        self._show_layer(_CELL_CONTOUR_LAYER, cd,
                         {"colormap": "magma", "visible": True}, self.viewer.add_image)
        self._show_layer(_CELL_FOREGROUND_SCORE_LAYER, sd,
                         {"colormap": "viridis", "visible": True}, self.viewer.add_image)
        mem = " (memory filter on full build only)" if self.memory_tau_spin.value() > 0 else ""
        self._status(f"Preview t={t}{mem}")

    # ================================================================
    # 4. Segment (Initialize + auto-commit)
    # ================================================================
    def _on_segment(self) -> None:
        if self._pos_dir is None:
            self._status("No project open."); return
        if not self._require(
            (self._nuc_labels_path(), "tracked_labels.tif (nucleus)"),
            (self._contour_path(), "contour_maps.tif"),
            (self._foreground_path(), "foreground_masks.tif"),
        ):
            return

        pos_dir = self._pos_dir
        output_path = self._cell_labels_path()

        from cellflow.segmentation.cell_label_icm import (
            CellLabelICMParams, initialize_icm, commit_labels,
        )
        params = CellLabelICMParams(
            alpha_unary=self.alpha_unary_spin.value(),
            lambda_s=self.lambda_s_spin.value(),
            beta_s=self.beta_s_spin.value(),
            lambda_t=self.lambda_t_spin.value(),
            gamma_unary=self.gamma_unary_spin.value(),
            n_workers=self.n_workers_spin.value(),
        )

        def _done(result):
            self._initialize_worker = None
            state, labels = result
            self._icm_state = state
            self._show_layer(_CELL_SEG_LAYER, labels, {"visible": True}, self.viewer.add_labels)
            commit_labels(labels, output_path)
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            self._files_widget.refresh(pos_dir)
            self._status(
                f"Segmentation complete — {state.n_labels} labels, "
                f"saved to {output_path.name}."
            )

        def _error(exc):
            self._initialize_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            self._status(f"Error: {exc}")
            logger.exception("Segment error", exc_info=exc)

        @thread_worker(connect={
            "yielded": self._on_progress, "returned": _done, "errored": _error,
        })
        def _worker():
            from cellflow.segmentation.cell_label_icm import _load_pos_dir_inputs
            msg_q: queue.SimpleQueue = queue.SimpleQueue()
            result_holder, exc_holder = [], []

            def _run():
                try:
                    nuc, fg, ct, fg_scores = _load_pos_dir_inputs(pos_dir)
                    s, init = initialize_icm(
                        nuc, fg, ct, params,
                        foreground_scores=fg_scores,
                        progress_cb=lambda m: msg_q.put(m),
                    )
                    result_holder.append((s, init))
                except Exception as e:
                    exc_holder.append(e)

            yield "Loading inputs..."
            t = threading.Thread(target=_run, daemon=True)
            t.start()
            while t.is_alive() or not msg_q.empty():
                try:
                    yield msg_q.get_nowait()
                except queue.Empty:
                    t.join(timeout=0.05)
            if exc_holder:
                raise exc_holder[0]
            return result_holder[0]

        self._status("Segmenting...")
        self.pipeline_progress_bar.setRange(0, 0)
        self.pipeline_progress_bar.setVisible(True)
        self._set_pipeline_buttons_enabled(False)
        self._initialize_worker = _worker()

    # ================================================================
    # 5. Correction
    # ================================================================
    @staticmethod
    def _broadcast_ref(image, shape):
        if image is None:
            return None
        if image.ndim == 2 and len(shape) >= 3:
            return np.broadcast_to(image[np.newaxis], (shape[0],) + image.shape).copy()
        return image

    def _on_load_labels(self) -> None:
        lp = self._cell_labels_path()
        if lp is None or not lp.exists():
            self._correction_status("No cell labels file found."); return
        self._correction_status("Loading...")
        czp, nzp = self._cell_zavg_path(), self._nuc_zavg_path()

        @thread_worker(connect={
            "returned": self._on_labels_loaded,
            "errored": lambda e: self._correction_status(f"Error: {e}"),
        })
        def _w():
            labels = read_full_tracked_stack(lp)
            cz = np.asarray(tifffile.imread(str(czp)), dtype=np.float32) if czp and czp.exists() else None
            nz = np.asarray(tifffile.imread(str(nzp)), dtype=np.float32) if nzp and nzp.exists() else None
            return labels, cz, nz
        _w()

    def _on_labels_loaded(self, result) -> None:
        labels, cz, nz = result
        self._show_layer(_TRACKED_CELL_LAYER, labels, {}, self.viewer.add_labels)
        for img, name, cmap in (
            (self._broadcast_ref(cz, labels.shape), _CELL_ZAVG_LAYER, "gray"),
            (self._broadcast_ref(nz, labels.shape), _NUC_ZAVG_LAYER, "bop orange"),
        ):
            if img is None: continue
            if name in self.viewer.layers:
                self.viewer.layers[name].data = img
            else:
                self.viewer.add_image(img, name=name, colormap=cmap, blending="additive")
        self._correction_status(f"Loaded {labels.shape}.")
        self.correction_widget.activate_layer(self.viewer.layers[_TRACKED_CELL_LAYER])

    def _on_save_labels(self) -> None:
        lp = self._cell_labels_path()
        if lp is None:
            self._correction_status("No project open."); return
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._correction_status("No labels layer."); return
        data = np.asarray(self.viewer.layers[_TRACKED_CELL_LAYER].data)
        if data.ndim != 3:
            self._correction_status("Labels not 3D."); return
        lp.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(lp), data.astype(np.uint32, copy=False), compression="zlib")
        self._files_widget.refresh(self._pos_dir)
        self._correction_status(f"Saved {data.shape[0]} frames → {lp.name}.")

    # ── Fill Holes ────────────────────────────────────────────────── # ← NEW

    def _on_fill_holes(self) -> None:
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._correction_status("No cell labels loaded."); return
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
        radius = int(self.hole_radius_spin.value())
        frames = self._correction_frame_indices(layer)

        changed_frames = 0
        changed_pixels = 0
        for t in frames:
            seg2d = self.correction_widget._frame_view(layer, t)
            before = seg2d.copy()
            result = fill_label_holes(seg2d, radius=radius)
            np.copyto(seg2d, result)
            diff = int(np.sum(before != seg2d))
            if diff:
                changed_frames += 1
                changed_pixels += diff
                self.correction_widget._record_history(layer, t, before)

        if changed_pixels:
            layer.refresh()
            if self.correction_widget._selected_label:
                t_now = self._current_t()
                self.correction_widget._update_highlight(
                    t_now, self.correction_widget._selected_label,
                )
            self._correction_status(
                f"Filled holes in {changed_frames} frame(s), "
                f"{changed_pixels} px changed. Unsaved."
            )
        else:
            self._correction_status("No interior holes found.")

    # ── Fix Semi Holes ────────────────────────────────────────────── # ← NEW

    def _on_fix_semiholes(self) -> None:
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._correction_status("No cell labels loaded."); return
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
        radius = int(self.hole_radius_spin.value())
        max_opening = int(self.semihole_opening_spin.value())
        frames = self._correction_frame_indices(layer)

        changed_frames = 0
        changed_pixels = 0
        for t in frames:
            seg2d = self.correction_widget._frame_view(layer, t)
            before = seg2d.copy()
            result = fix_label_semiholes(
                seg2d, radius=radius, max_opening=max_opening,
            )
            np.copyto(seg2d, result)
            diff = int(np.sum(before != seg2d))
            if diff:
                changed_frames += 1
                changed_pixels += diff
                self.correction_widget._record_history(layer, t, before)

        if changed_pixels:
            layer.refresh()
            if self.correction_widget._selected_label:
                t_now = self._current_t()
                self.correction_widget._update_highlight(
                    t_now, self.correction_widget._selected_label,
                )
            self._correction_status(
                f"Fixed semiholes in {changed_frames} frame(s), "
                f"{changed_pixels} px changed. Unsaved."
            )
        else:
            self._correction_status("No semiholes found.")

    # ── Clean Up (movie-wide) ─────────────────────────────────────── # ← NEW

    def _get_nuclear_labels(self) -> np.ndarray | None:
        """Try viewer layer first, then fall back to disk."""
        if "Tracked: Nucleus" in self.viewer.layers:
            return np.asarray(self.viewer.layers["Tracked: Nucleus"].data)
        nuc_path = self._nuc_labels_path()
        if nuc_path is not None and nuc_path.exists():
            return np.asarray(tifffile.imread(str(nuc_path)))
        return None

    def _on_cleanup(self) -> None:
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._correction_status("No cell labels loaded."); return
        nuc_data = self._get_nuclear_labels()
        if nuc_data is None:
            self._correction_status(
                "Nuclear labels not found (viewer or disk)."
            ); return

        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
        cell_data = np.asarray(layer.data).copy()

        try:
            stats = cleanup_movie(cell_data, nuc_data)
        except ValueError as exc:
            self._correction_status(str(exc)); return

        layer.data = cell_data

        total = (
            stats["fragments_cleared"]
            + stats["cells_relabeled"]
            + stats["orphans_removed"]
        )
        if total:
            if self.correction_widget._selected_label:
                t_now = self._current_t()
                self.correction_widget._update_highlight(
                    t_now, self.correction_widget._selected_label,
                )
            self._correction_status(
                f"Cleanup: {stats['fragments_cleared']} fragment px, "
                f"{stats['cells_relabeled']} relabeled, "
                f"{stats['orphans_removed']} orphans removed. "
                f"No undo — save or reload to revert. Unsaved."
            )
        else:
            self._correction_status("Cleanup: nothing to change.")

    # ── Expand Cell ───────────────────────────────────────────────────

    def _foreground_for_expand(self) -> np.ndarray | None:
        if _CELL_FOREGROUND_LAYER in self.viewer.layers:
            return np.asarray(self.viewer.layers[_CELL_FOREGROUND_LAYER].data)
        fp = self._foreground_path()
        if fp is None or not fp.exists():
            return None
        fg = np.asarray(tifffile.imread(str(fp)))
        self._show_layer(_CELL_FOREGROUND_LAYER, fg, {}, self.viewer.add_labels)
        return fg

    def _on_expand_cell(self) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._correction_status("No labels loaded."); return
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
        if self.correction_widget._layer is not layer:
            self._correction_status("Labels not active for correction."); return
        lid = int(self.correction_widget._selected_label)
        if lid == 0:
            self._correction_status("No cell selected."); return
        labels = np.asarray(layer.data)
        if labels.ndim < 3:
            self._correction_status("Labels not 3D."); return
        t = self._current_time_index(labels.shape[0])
        seg2d = self.correction_widget._frame_view(layer, t)
        if not np.any(seg2d == lid):
            self._correction_status(f"Cell {lid} absent at t={t}."); return

        fg = self._foreground_for_expand()
        if fg is None:
            self._correction_status("Foreground mask not found."); return
        if fg.shape != labels.shape:
            self._correction_status(f"Foreground shape mismatch."); return
        fg2d = fg[t]
        while fg2d.ndim > 2:
            if fg2d.shape[0] != 1:
                self._correction_status("Foreground frame shape unsupported."); return
            fg2d = fg2d[0]

        before = seg2d.copy()
        try:
            added = expand_label_to_foreground(
                seg2d, fg2d, lid, max_distance=int(self.expand_max_px_spin.value()),
            )
        except ValueError as exc:
            self._correction_status(str(exc)); return
        if added == 0:
            if not bool(np.any((fg2d > 0) & (before == lid))):
                self._correction_status(f"Cell {lid} doesn't touch foreground at t={t}.")
            else:
                self._correction_status(f"No expansion for cell {lid} at t={t}.")
            return
        self.correction_widget._record_history(layer, t, before)
        layer.refresh()
        self.correction_widget._update_highlight(t, lid)
        self._correction_status(f"Expanded cell {lid} at t={t} by {added} px. Unsaved.")