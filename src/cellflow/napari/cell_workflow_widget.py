"""Cell segmentation workflow widget for CellFlow.

Flat layout — action buttons in a two-column grid at top, one collapsible
parameter panel, correction section at bottom.

Stages:
  1. Flow Filtering → ``filtered_dp.tif``
  2. Foreground Masks → ``foreground_masks.tif``
  3. Contours → ``contours.tif``, ``foreground_scores.tif``
  4. Segmentation → ``tracked_labels.tif`` (initialize + auto-commit)
  5. Correction (load / save / fill holes / fix semiholes / cleanup / expand)
"""
from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._widget_helpers import tool_btn as _tool_btn
from cellflow.correction.labels import best_overlapping_label
from cellflow.napari.cell_correction_widget import CellCorrectionWidget
from cellflow.napari.cell_params_widget import CellParamsWidget
from cellflow.napari.ui_style import (
    stage_header_label,
    status_label,
)
from cellflow.napari.widgets import (
    CollapsibleSection,
    PipelineFilesWidget,
    make_pipeline_files_header,
)
from cellflow.segmentation import (
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
        self._running_stage: str | None = None

        self._setup_ui()
        self._connect_signals()

    # ================================================================
    # UI
    # ================================================================
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)
        # Shrink to content height so collapsed sections don't leave a tall
        # empty strip below the last row. Matches nucleus_workflow_widget.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # ── Pipeline files (single deduplicated panel) ────────────────
        self._files_widget = PipelineFilesWidget(
            [
                ("Inputs", [
                    ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
                    ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
                    ("1_cellpose/cell_foreground.tif", "Cell foreground"),
                    ("1_cellpose/nucleus_foreground.tif", "Nucleus foreground"),
                    ("2_nucleus/tracked_labels.tif", "Nucleus tracked labels"),
                ]),
                ("Intermediates", [
                    ("3_cell/filtered_dp.tif", "Filtered flow vectors"),
                    ("3_cell/foreground_masks.tif", "Foreground masks"),
                    ("3_cell/contours.tif", "Contours"),
                    ("3_cell/foreground_scores.tif", "Foreground scores"),
                ]),
                ("Output", [
                    ("3_cell/tracked_labels.tif", "Cell tracked labels"),
                ]),
            ],
            viewer=self.viewer,
        )
        self._pipeline_files_section = CollapsibleSection(
            "Pipeline Files",
            self._files_widget,
            expanded=False,
        )
        (
            self.pipeline_files_header,
            self.pipeline_files_header_lbl,
            self.pipeline_files_toggle_btn,
        ) = make_pipeline_files_header(
            self._pipeline_files_section,
            stage_key="cell",
            parent=self,
        )
        root.addWidget(self.pipeline_files_header)
        root.addWidget(self._pipeline_files_section)

        # ── Pipeline parameters and per-stage rows ───────────────────
        self.cell_params_widget = CellParamsWidget(self)
        self._install_params_aliases()
        # CellParamsWidget acts as a controller here. Its visible section is
        # reparented into this layout, so keep the owner widget hidden to avoid
        # an unmanaged default rectangle intercepting header clicks.
        self.cell_params_widget.hide()

        self._build_pipeline_stage_rows(root)

        self.pipeline_status_lbl = _make_status()
        root.addWidget(self.pipeline_status_lbl)
        self.pipeline_progress_bar = _make_progress()
        root.addWidget(self.pipeline_progress_bar)

        # ── Correction section (child widget) ────────────────────────
        self.cell_correction_widget = CellCorrectionWidget(
            self.viewer,
            pos_dir_provider=lambda: self._pos_dir,
            files_widget_refresh_callback=lambda pd: self._files_widget.refresh(pd),
        )
        self._install_correction_aliases()
        # CellCorrectionWidget owns behavior; visible pieces are reparented
        # here to mirror the nucleus workflow structure.
        self.cell_correction_widget.hide()
        root.addWidget(self.correction_header)
        root.addWidget(self.correction_mode_section)

    # -- Parameters --------------------------------------------------------

    def _build_pipeline_stage_rows(self, root: QVBoxLayout) -> None:
        """Build nucleus-style pipeline stage rows with inline params."""
        self.flow_params_btn = _tool_btn(
            "⚙", "Show parameters for this stage.", checkable=True
        )
        self.flow_run_btn = _tool_btn("▶", "Run flow filtering.")
        self.foreground_params_btn = _tool_btn(
            "⚙", "Show parameters for this stage.", checkable=True
        )
        self.foreground_run_btn = _tool_btn("▶", "Run foreground mask generation.")
        self.contour_params_btn = _tool_btn(
            "⚙", "Show parameters for this stage.", checkable=True
        )
        self.contour_preview_btn = _tool_btn(
            "▷", "Preview contours for the current frame."
        )
        self.contour_run_btn = _tool_btn("▶", "Run contour map generation.")
        self.segmentation_params_btn = _tool_btn(
            "⚙", "Show parameters for this stage.", checkable=True
        )
        self.segmentation_run_btn = _tool_btn("▶", "Run cell segmentation.")

        # Backward-compatible preferred alias for existing callers.
        self.filter_flow_btn = self.flow_run_btn
        self.build_foreground_btn = self.foreground_run_btn
        self.preview_contour_btn = self.contour_preview_btn
        self.build_contour_btn = self.contour_run_btn
        self.segment_btn = self.segmentation_run_btn

        for section in (
            self.flow_filter_section,
            self.foreground_section,
            self.contour_section,
            self.segmentation_section,
        ):
            section.set_header_visible(False)
            section.collapse()

        for params_btn, section in (
            (self.flow_params_btn, self.flow_filter_section),
            (self.foreground_params_btn, self.foreground_section),
            (self.contour_params_btn, self.contour_section),
            (self.segmentation_params_btn, self.segmentation_section),
        ):
            params_btn.toggled.connect(
                lambda checked, section=section: section._toggle.setChecked(checked)
            )

        root.addLayout(self._stage_row(
            self._stage_label("Flow filtering"),
            self.flow_params_btn,
            self.flow_run_btn,
        ))
        root.addWidget(self.flow_filter_section)
        root.addLayout(self._stage_row(
            self._stage_label("Foreground masks"),
            self.foreground_params_btn,
            self.foreground_run_btn,
        ))
        root.addWidget(self.foreground_section)
        root.addLayout(self._stage_row(
            self._stage_label("Contours"),
            self.contour_params_btn,
            self.contour_preview_btn,
            self.contour_run_btn,
        ))
        root.addWidget(self.contour_section)
        root.addLayout(self._stage_row(
            self._stage_label("Segmentation"),
            self.segmentation_params_btn,
            self.segmentation_run_btn,
        ))
        root.addWidget(self.segmentation_section)

    @staticmethod
    def _stage_label(text: str) -> QLabel:
        return stage_header_label(QLabel(text), "cell")

    @staticmethod
    def _stage_row(label: QLabel, *trailing: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(label)
        row.addStretch(1)
        for widget in trailing:
            row.addWidget(widget)
        return row

    def _install_params_aliases(self) -> None:
        """Install compatibility aliases for all controls owned by CellParamsWidget."""
        p = self.cell_params_widget
        self.flow_filter_section = p.flow_filter_section
        self.foreground_section = p.foreground_section
        self.contour_section = p.contour_section
        self.segmentation_section = p.segmentation_section
        # Flow filtering
        self.ff_median_time_spin = p.ff_median_time_spin
        self.ff_median_space_spin = p.ff_median_space_spin
        self.ff_gauss_time_spin = p.ff_gauss_time_spin
        self.ff_gauss_space_spin = p.ff_gauss_space_spin
        # Foreground
        self.fg_cellprob_threshold_spin = p.fg_cellprob_threshold_spin
        # Contour sweep
        self.cp_min_spin = p.cp_min_spin
        self.cp_max_spin = p.cp_max_spin
        self.cp_step_spin = p.cp_step_spin
        # Contour flow-following
        self.ff_flow_weight_spin = p.ff_flow_weight_spin
        self.ff_step_scale_spin = p.ff_step_scale_spin
        self.ff_max_iter_spin = p.ff_max_iter_spin
        # Gamma averaging
        self.gamma_min_spin = p.gamma_min_spin
        self.gamma_max_spin = p.gamma_max_spin
        self.gamma_step_spin = p.gamma_step_spin
        # Temporal stabilization
        self.memory_tau_spin = p.memory_tau_spin
        self.memory_floor_spin = p.memory_floor_spin
        # Segmentation ICM
        self.alpha_unary_spin = p.alpha_unary_spin
        self.lambda_s_spin = p.lambda_s_spin
        self.beta_s_spin = p.beta_s_spin
        self.lambda_t_spin = p.lambda_t_spin
        self.gamma_unary_spin = p.gamma_unary_spin
        self.n_workers_spin = p.n_workers_spin

    # -- Correction aliases -----------------------------------------------

    def _install_correction_aliases(self) -> None:
        """Install compatibility aliases for all controls owned by CellCorrectionWidget."""
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
        self.flow_run_btn.clicked.connect(self._on_flow_run_btn_clicked)
        self.foreground_run_btn.clicked.connect(self._on_foreground_run_btn_clicked)
        self.contour_preview_btn.clicked.connect(self._on_preview_contours)
        self.contour_run_btn.clicked.connect(self._on_contour_run_btn_clicked)
        self.segmentation_run_btn.clicked.connect(
            self._on_segmentation_run_btn_clicked
        )

    def _on_flow_run_btn_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()
        else:
            self._on_filter_flow()

    def _on_foreground_run_btn_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()
        else:
            self._on_build_foreground()

    def _on_contour_run_btn_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()
        else:
            self._on_build_contours()

    def _on_segmentation_run_btn_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()
        else:
            self._on_segment()

    # ================================================================
    # Path helpers
    # ================================================================
    def _p(self, *parts: str) -> Path | None:
        return self._pos_dir.joinpath(*parts) if self._pos_dir else None

    def _prob_path(self):          return self._p("1_cellpose", "cell_prob_3dt.tif")
    def _dp_path(self):            return self._p("1_cellpose", "cell_dp_3dt.tif")
    def _filtered_dp_path(self):   return self._p("3_cell", "filtered_dp.tif")
    def _foreground_path(self):    return self._p("3_cell", "foreground_masks.tif")
    def _contours_path(self):      return self._p("3_cell", "contours.tif")
    def _fg_scores_path(self):     return self._p("3_cell", "foreground_scores.tif")
    def _nuc_labels_path(self):    return self._p("2_nucleus", "tracked_labels.tif")
    def _cell_labels_path(self):   return self._p("3_cell", "tracked_labels.tif")

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
        # Prefer the [Correction] layer (active when correction mode is on);
        # fall back to the pipeline-side Tracked: Cell.
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
    def _status(self, msg: str) -> None:
        self.pipeline_status_lbl.setText(msg)
        self.pipeline_status_lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    def _correction_status(self, msg: str) -> None:
        self.cell_correction_widget._correction_status(msg)

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

    def _set_running_stage(self, stage_key: str | None) -> None:
        """Update per-stage run/cancel state.

        ``None`` means idle.  Any stage key means that row owns cancellation
        and the other rows are disabled until the worker returns or errors.
        """
        self._running_stage = stage_key
        rows = {
            "flow": (
                self.flow_params_btn,
                self.flow_run_btn,
                "Run flow filtering.",
            ),
            "foreground": (
                self.foreground_params_btn,
                self.foreground_run_btn,
                "Run foreground mask generation.",
            ),
            "contour": (
                self.contour_params_btn,
                self.contour_run_btn,
                "Run contour map generation.",
            ),
            "segmentation": (
                self.segmentation_params_btn,
                self.segmentation_run_btn,
                "Run cell segmentation.",
            ),
        }
        if stage_key is None:
            for params_btn, run_btn, tooltip in rows.values():
                params_btn.setEnabled(True)
                run_btn.setEnabled(True)
                run_btn.setText("▶")
                run_btn.setToolTip(tooltip)
            self.contour_preview_btn.setEnabled(True)
            return

        for key, (params_btn, run_btn, _tooltip) in rows.items():
            if key == stage_key:
                params_btn.setEnabled(True)
                run_btn.setEnabled(True)
                run_btn.setText("✕")
                run_btn.setToolTip("Cancel.")
            else:
                params_btn.setEnabled(False)
                run_btn.setEnabled(False)
        self.contour_preview_btn.setEnabled(False)

    def _set_pipeline_buttons_enabled(self, enabled: bool) -> None:
        """Backward-compatible shim for older tests/callers."""
        if enabled:
            self._set_running_stage(None)
            return
        for btn in (
            self.flow_params_btn,
            self.flow_run_btn,
            self.foreground_params_btn,
            self.foreground_run_btn,
            self.contour_params_btn,
            self.contour_preview_btn,
            self.contour_run_btn,
            self.segmentation_params_btn,
            self.segmentation_run_btn,
        ):
            btn.setEnabled(False)

    def _on_cancel(self) -> None:
        stage_to_worker = {
            "flow": self._ff_worker,
            "foreground": self._foreground_worker,
            "contour": self._contour_worker,
            "segmentation": self._initialize_worker,
        }
        worker = stage_to_worker.get(self._running_stage)
        if worker is not None and hasattr(worker, "quit"):
            worker.quit()
        self._ff_worker = None if self._running_stage == "flow" else self._ff_worker
        self._foreground_worker = (
            None if self._running_stage == "foreground" else self._foreground_worker
        )
        self._contour_worker = (
            None if self._running_stage == "contour" else self._contour_worker
        )
        self._initialize_worker = (
            None if self._running_stage == "segmentation" else self._initialize_worker
        )
        self._clear_progress()
        self._set_running_stage(None)
        self._status("Cancelled.")

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
    # 1. Flow Filtering
    # ================================================================
    def _flow_filter_params(self):
        return self.cell_params_widget.flow_filter_params()

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
        fdp = self._filtered_dp_path()
        if not self._require(
            (prob_path, "cell_prob_3dt.tif"),
            (dp_path, "cell_dp_3dt.tif"),
        ):
            return

        params = self._flow_filter_params()
        pos_dir = self._pos_dir

        def _done(result):
            self._ff_worker = None
            self._set_running_stage(None)
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
            self._set_running_stage(None)
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
            return mag

        self._status("Filtering flow...")
        self._set_running_stage("flow")
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
            self._set_running_stage(None)
            self._clear_progress()
            self._show_layer(_CELL_FOREGROUND_LAYER, result, {}, self.viewer.add_labels)
            self._files_widget.refresh(pos_dir)
            self._status("Foreground masks complete.")

        def _error(exc):
            self._foreground_worker = None
            self._set_running_stage(None)
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
        self._set_running_stage("foreground")
        self._foreground_worker = _worker()

    # ================================================================
    # 3. Contour Maps
    # ================================================================
    def _cellprob_thresholds(self) -> list[float]:
        return self.cell_params_widget.cellprob_thresholds()

    def _gammas(self) -> list[float]:
        return self.cell_params_widget.gammas()

    def _contour_ff_params(self):
        return self.cell_params_widget.contour_ff_params()

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
        ct_path, sc_path = self._contours_path(), self._fg_scores_path()
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
            self._set_running_stage(None)
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
            self._set_running_stage(None)
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
        self._set_running_stage("contour")
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
            (self._contours_path(), "contours.tif"),
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
            self._set_running_stage(None)
            self._clear_progress()
            self._files_widget.refresh(pos_dir)
            self._status(
                f"Segmentation complete — {state.n_labels} labels, "
                f"saved to {output_path.name}."
            )

        def _error(exc):
            self._initialize_worker = None
            self._set_running_stage(None)
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
        self._set_running_stage("segmentation")
        self._initialize_worker = _worker()
