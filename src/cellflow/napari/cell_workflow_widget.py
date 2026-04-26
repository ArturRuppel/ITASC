"""Cell workflow widget for hypothesis generation and tracking in CellFlow v2."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cellflow.database.hypotheses import (
    HypothesisRecord,
    SeededWatershedSweepSpec,
    build_seeded_watershed_parameter_sets,
    delete_hypothesis_parameter,
    iter_seeded_watershed_records,
    list_hypotheses,
    read_full_hypothesis_stack,
    read_hypothesis_labels,
    write_hypothesis_sweep_h5,
)
from cellflow.database.tracked import (
    read_full_tracked_stack,
    write_tracked_frame,
)
from cellflow.database.validation import (
    invalidate_frame,
    is_validated,
    read_validated_frames,
    validate_frame,
)
from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget
from cellflow.segmentation import SeededWatershedParams, compute_seeded_watershed
from cellflow.tracking import propagate_one_frame
from cellflow.tracking.retracker import retrack_frame

logger = logging.getLogger(__name__)

_PREVIEW_LAYER  = "Preview: Cell"
_HYP_LAYER      = "Hypothesis: Cell"
_TRACKED_LAYER  = "Tracked: Cell"
_CELL_ZAVG_LAYER = "Cell z-avg"
_NUC_ZAVG_LAYER  = "Nucleus z-avg"


class CellWorkflowWidget(QWidget):
    """Cell hypothesis generation and tracking management."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._stop_flag: bool = False
        self._sweep_worker = None
        self._current_db_p: int | None = None
        self._db_param_map: dict[tuple[float, float], int] = {}
        self._db_fg_vals: list[float] = []
        self._db_compactness_vals: list[float] = []
        self._setup_ui()
        self._connect_signals()

    # ──────────────────────────────────────────────────────────────────────────
    # UI setup
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        SPIN_MAX_W = 70

        def _compact(spin, w=SPIN_MAX_W):
            spin.setMaximumWidth(w)
            spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            return spin

        def _compact_btn(btn):
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            return btn

        # ── Inputs ────────────────────────────────────────────────────────
        self.input_files = PipelineFilesWidget([
            ("Inputs", [
                ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
                ("1_cellpose/cell_dp_3dt.tif",   "Cell dp 3D+t"),
                ("2_nucleus/tracked_labels.tif",  "Nucleus tracked labels"),
            ]),
        ])
        layout.addWidget(self.input_files)

        # ── 1. Hypothesis Generation ──────────────────────────────────────
        _gen_inner = QWidget()
        gen_lay = QVBoxLayout(_gen_inner)
        gen_lay.setContentsMargins(4, 4, 4, 4)
        gen_lay.setSpacing(6)
        gen_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Basin selector
        basin_row = QHBoxLayout()
        basin_row.addWidget(QLabel("Basin:"))
        self.basin_combo = QComboBox()
        self.basin_combo.addItems(["Prob Map", "Flow Magnitude"])
        self.basin_combo.setToolTip(
            "Intensity map used as the watershed basin.\n"
            "Prob Map: sigmoid of cellpose probability logits.\n"
            "Flow Magnitude: L2 magnitude of dp vectors (computed on the fly).\n"
            "Foreground mask is always derived from sigmoid(prob) regardless of choice."
        )
        basin_row.addWidget(self.basin_combo)
        basin_row.addStretch()
        self.overwrite_check = QCheckBox("Overwrite existing")
        basin_row.addWidget(self.overwrite_check)
        gen_lay.addLayout(basin_row)

        self.gen_tabs = QTabWidget()

        # Tab: Tuning
        tuning_tab = QWidget()
        tuning_lay = QVBoxLayout(tuning_tab)

        tuning_form = QFormLayout()
        tuning_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        tuning_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        tuning_form.setHorizontalSpacing(8)
        tuning_form.setVerticalSpacing(4)

        self.single_fg_threshold = QDoubleSpinBox()
        self.single_fg_threshold.setRange(0.01, 0.99)
        self.single_fg_threshold.setValue(0.5)
        self.single_fg_threshold.setDecimals(2)
        self.single_fg_threshold.setSingleStep(0.05)
        self.single_fg_threshold.setToolTip(
            "Sigmoid foreground probability cutoff — pixels below this are excluded "
            "from the segmentation mask. Seeds whose centroid falls outside are dropped."
        )
        tuning_form.addRow("Foreground Threshold:", _compact(self.single_fg_threshold))

        self.single_compactness = QDoubleSpinBox()
        self.single_compactness.setRange(0.0, 10.0)
        self.single_compactness.setValue(0.0)
        self.single_compactness.setDecimals(2)
        self.single_compactness.setSingleStep(0.1)
        self.single_compactness.setToolTip(
            "Watershed compactness parameter — higher values produce rounder cells "
            "(penalises long, narrow regions). 0 = standard watershed."
        )
        tuning_form.addRow("Compactness:", _compact(self.single_compactness))
        tuning_lay.addLayout(tuning_form)

        tuning_btn_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.save_db_btn = QPushButton("Save to DB")
        tuning_btn_row.addWidget(_compact_btn(self.preview_btn))
        tuning_btn_row.addWidget(_compact_btn(self.save_db_btn))
        tuning_lay.addLayout(tuning_btn_row)
        self.gen_tabs.addTab(tuning_tab, "Tuning")

        # Tab: Sweep
        sweep_tab = QWidget()
        sweep_lay = QVBoxLayout(sweep_tab)

        def _sweep_row(label, d_min, d_max, d_step, decimals=2):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            min_s = QDoubleSpinBox()
            max_s = QDoubleSpinBox()
            step_s = QDoubleSpinBox()
            for s in (min_s, max_s, step_s):
                s.setRange(0.0, 20.0)
                s.setDecimals(decimals)
                s.setMaximumWidth(62)
                s.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            min_s.setValue(d_min)
            max_s.setValue(d_max)
            step_s.setValue(d_step)
            row.addWidget(QLabel("min")); row.addWidget(min_s)
            row.addWidget(QLabel("max")); row.addWidget(max_s)
            row.addWidget(QLabel("step")); row.addWidget(step_s)
            row.addStretch()
            sweep_lay.addLayout(row)
            return min_s, max_s, step_s

        self.sweep_fg_thr      = _sweep_row("Foreground Thr", 0.4, 0.6, 0.05)
        self.sweep_compactness = _sweep_row("Compactness",    0.0, 0.5, 0.1)

        workers_row = QHBoxLayout()
        workers_row.addWidget(QLabel("Workers:"))
        self.sweep_n_workers = QSpinBox()
        self.sweep_n_workers.setRange(1, max(1, os.cpu_count() or 1))
        self.sweep_n_workers.setValue(1)
        self.sweep_n_workers.setToolTip("Parallel threads for the sweep.")
        workers_row.addWidget(_compact(self.sweep_n_workers))
        workers_row.addStretch()
        sweep_lay.addLayout(workers_row)

        sweep_btn_row = QHBoxLayout()
        self.run_sweep_btn    = QPushButton("Run Sweep")
        self.cancel_sweep_btn = QPushButton("Cancel")
        self.cancel_sweep_btn.setEnabled(False)
        sweep_btn_row.addWidget(_compact_btn(self.run_sweep_btn))
        sweep_btn_row.addWidget(_compact_btn(self.cancel_sweep_btn))
        sweep_lay.addLayout(sweep_btn_row)

        self.sweep_progress_bar = QProgressBar()
        self.sweep_progress_bar.setRange(0, 100)
        self.sweep_progress_bar.setValue(0)
        self.sweep_progress_bar.setVisible(False)
        sweep_lay.addWidget(self.sweep_progress_bar)

        self.gen_tabs.addTab(sweep_tab, "Sweep")
        gen_lay.addWidget(self.gen_tabs)

        self.gen_section = CollapsibleSection(
            "1. Hypothesis Generation", _gen_inner, expanded=False
        )
        layout.addWidget(self.gen_section)

        # ── 2. Database Browser ──────────────────────────────────────────
        _db_inner = QWidget()
        db_lay = QVBoxLayout(_db_inner)
        db_lay.setContentsMargins(4, 4, 4, 4)
        db_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        hdr_row = QHBoxLayout()
        self.db_activate_btn = QPushButton("Activate")
        self.db_activate_btn.setCheckable(True)
        self.db_activate_btn.setChecked(False)
        self.db_activate_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        hdr_row.addWidget(self.db_activate_btn)
        hdr_row.addStretch()
        self.db_refresh_btn = QPushButton()
        self.db_refresh_btn.setToolTip("Refresh database browser")
        self.db_refresh_btn.setIcon(
            self.style().standardIcon(self.style().StandardPixmap.SP_BrowserReload)
        )
        self.db_refresh_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        hdr_row.addWidget(self.db_refresh_btn)
        db_lay.addLayout(hdr_row)

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("FG Thr:"))
        self.db_fg_thr_spin = QDoubleSpinBox()
        self.db_fg_thr_spin.setRange(0.01, 0.99)
        self.db_fg_thr_spin.setValue(0.5)
        self.db_fg_thr_spin.setDecimals(2)
        self.db_fg_thr_spin.setSingleStep(0.05)
        self.db_fg_thr_spin.setEnabled(False)
        params_row.addWidget(_compact(self.db_fg_thr_spin))
        params_row.addWidget(QLabel("Compactness:"))
        self.db_compactness_spin = QDoubleSpinBox()
        self.db_compactness_spin.setRange(0.0, 10.0)
        self.db_compactness_spin.setValue(0.0)
        self.db_compactness_spin.setDecimals(2)
        self.db_compactness_spin.setSingleStep(0.1)
        self.db_compactness_spin.setEnabled(False)
        params_row.addWidget(_compact(self.db_compactness_spin))
        params_row.addStretch(1)
        db_lay.addLayout(params_row)

        self.db_info_lbl = QLabel("—")
        self.db_info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        db_lay.addWidget(self.db_info_lbl)

        db_btn_row = QHBoxLayout()
        self.set_seed_btn = QPushButton("Set as Tracking Seed")
        db_btn_row.addWidget(_compact_btn(self.set_seed_btn))
        db_lay.addLayout(db_btn_row)

        db_del_row = QHBoxLayout()
        self.del_stack_btn = QPushButton("Remove Stack")
        self.del_stack_btn.setStyleSheet(
            "QPushButton { color: #cc3333; }"
            "QPushButton:hover { background-color: #4a1111; color: white; }"
        )
        db_del_row.addWidget(_compact_btn(self.del_stack_btn))
        db_lay.addLayout(db_del_row)

        self.db_section = CollapsibleSection(
            "2. Database Browser", _db_inner, expanded=False
        )
        layout.addWidget(self.db_section)

        # ── 3. Automated Search ──────────────────────────────────────────
        _search_inner = QWidget()
        search_lay = QVBoxLayout(_search_inner)
        search_lay.setContentsMargins(4, 4, 4, 4)
        search_lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        def _weight_spin(default):
            w = QDoubleSpinBox()
            w.setRange(0.0, 10.0)
            w.setValue(default)
            w.setSingleStep(0.5)
            w.setDecimals(1)
            return w

        search_form = QFormLayout()
        search_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        search_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        search_form.setHorizontalSpacing(8)
        search_form.setVerticalSpacing(4)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0, 1)
        self.iou_spin.setValue(0.5)
        self.iou_spin.setSingleStep(0.1)
        search_form.addRow("IoU Threshold:", _compact(self.iou_spin))

        self.dist_spin = QDoubleSpinBox()
        self.dist_spin.setRange(0, 1000)
        self.dist_spin.setValue(20.0)
        search_form.addRow("Max Dist (px):", _compact(self.dist_spin))

        self.vel_sigma_spin = QDoubleSpinBox()
        self.vel_sigma_spin.setRange(1, 500)
        self.vel_sigma_spin.setValue(25.0)
        self.vel_sigma_spin.setSingleStep(5.0)
        search_form.addRow("Velocity σ (px):", _compact(self.vel_sigma_spin))

        self.iou_weight_spin = _weight_spin(1.0)
        search_form.addRow("IoU Weight:", _compact(self.iou_weight_spin))

        self.area_weight_spin = _weight_spin(1.0)
        search_form.addRow("Area Weight:", _compact(self.area_weight_spin))

        self.vel_weight_spin = _weight_spin(1.0)
        search_form.addRow("Velocity Weight:", _compact(self.vel_weight_spin))

        self.pos_weight_spin = _weight_spin(0.0)
        search_form.addRow("Position Weight:", _compact(self.pos_weight_spin))

        self.unmatched_spin = QDoubleSpinBox()
        self.unmatched_spin.setRange(0.0, 1.0)
        self.unmatched_spin.setValue(0.1)
        self.unmatched_spin.setSingleStep(0.01)
        search_form.addRow("Unmatched Score:", _compact(self.unmatched_spin))

        search_lay.addLayout(search_form)

        prop_row = QHBoxLayout()
        self.prop_next_btn = QPushButton("Propagate Next")
        self.prop_all_btn  = QPushButton("Propagate All")
        self.stop_btn      = QPushButton("Stop")
        prop_row.addWidget(_compact_btn(self.prop_next_btn))
        prop_row.addWidget(_compact_btn(self.prop_all_btn))
        prop_row.addWidget(_compact_btn(self.stop_btn))
        search_lay.addLayout(prop_row)

        save_tracked_row = QHBoxLayout()
        self.save_tracked_btn = QPushButton("Save Tracked Labels")
        save_tracked_row.addWidget(_compact_btn(self.save_tracked_btn))
        search_lay.addLayout(save_tracked_row)

        load_tracked_row = QHBoxLayout()
        self.load_tracked_btn = QPushButton("Load Tracked Labels")
        load_tracked_row.addWidget(_compact_btn(self.load_tracked_btn))
        search_lay.addLayout(load_tracked_row)

        reassign_row = QHBoxLayout()
        self.reassign_ids_btn = QPushButton("Reassign IDs")
        reassign_row.addWidget(_compact_btn(self.reassign_ids_btn))
        search_lay.addLayout(reassign_row)

        self.search_section = CollapsibleSection(
            "3. Automated Search", _search_inner, expanded=False
        )
        layout.addWidget(self.search_section)

        # ── 4. Manual Correction ──────────────────────────────────────────
        _corr_inner = QWidget()
        _corr_inner_lay = QVBoxLayout(_corr_inner)
        _corr_inner_lay.setContentsMargins(0, 0, 0, 0)
        _corr_inner_lay.setSpacing(4)

        retrack_row = QHBoxLayout()
        self.retrack_btn = QPushButton("Retrack Frame")
        retrack_row.addWidget(_compact_btn(self.retrack_btn))
        self.validate_btn = QPushButton("Validate Frame")
        self.validate_btn.setCheckable(True)
        retrack_row.addWidget(_compact_btn(self.validate_btn))
        _corr_inner_lay.addLayout(retrack_row)

        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            inspector_first=True,
        )
        _corr_inner_lay.addWidget(self.correction_widget)

        self.correction_section = CollapsibleSection(
            "4. Manual Correction", _corr_inner, expanded=False
        )
        layout.addWidget(self.correction_section)

        # ── Status label ──────────────────────────────────────────────────
        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        layout.addWidget(self.status_lbl)

        # ── Outputs ───────────────────────────────────────────────────────
        self.output_files = PipelineFilesWidget([
            ("Outputs", [
                ("3_cell/hypotheses.h5", "Hypotheses DB"),
            ]),
        ])
        layout.addWidget(self.output_files)

    # ──────────────────────────────────────────────────────────────────────────
    # Signal wiring
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.preview_btn.clicked.connect(self._on_preview)
        self.save_db_btn.clicked.connect(self._on_save_db)
        self.run_sweep_btn.clicked.connect(self._on_run_sweep)
        self.cancel_sweep_btn.clicked.connect(self._on_cancel_sweep)
        self.db_fg_thr_spin.valueChanged.connect(self._on_db_param_changed)
        self.db_compactness_spin.valueChanged.connect(self._on_db_param_changed)
        self.db_activate_btn.toggled.connect(self._on_db_activate_toggled)
        self.db_refresh_btn.clicked.connect(lambda: self._refresh_db_browser())
        self.set_seed_btn.clicked.connect(self._on_set_seed)
        self.del_stack_btn.clicked.connect(self._on_remove_stack)
        self.prop_next_btn.clicked.connect(self._on_propagate_next)
        self.prop_all_btn.clicked.connect(self._on_propagate_all)
        self.stop_btn.clicked.connect(lambda: setattr(self, "_stop_flag", True))
        self.save_tracked_btn.clicked.connect(self._on_save_tracked)
        self.load_tracked_btn.clicked.connect(self._on_load_tracked)
        self.reassign_ids_btn.clicked.connect(self._on_reassign_ids)
        self.retrack_btn.clicked.connect(self._on_retrack_frame)
        self.validate_btn.toggled.connect(self._on_validate_toggled)
        self.viewer.dims.events.current_step.connect(self._on_dims_step_changed)

    # ──────────────────────────────────────────────────────────────────────────
    # Public refresh
    # ──────────────────────────────────────────────────────────────────────────

    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self.input_files.refresh(pos_dir)
        self.output_files.refresh(pos_dir)
        if pos_dir is None:
            self.correction_widget.deactivate()
            return
        self._refresh_db_browser()
        self._refresh_validate_btn()

    # ──────────────────────────────────────────────────────────────────────────
    # Path helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _hyp_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "hypotheses.h5" if self._pos_dir else None

    def _tracked_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "tracked_labels.tif" if self._pos_dir else None

    def _prob_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_prob_3dt.tif" if self._pos_dir else None

    def _dp_path(self) -> Path | None:
        return self._pos_dir / "1_cellpose" / "cell_dp_3dt.tif" if self._pos_dir else None

    def _nucleus_tracked_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    def _cell_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "cell_zavg.tif" if self._pos_dir else None

    def _nucleus_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "nucleus_zavg.tif" if self._pos_dir else None

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _current_t(self) -> int:
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) >= 1 else 0

    def _current_z(self) -> int:
        step = self.viewer.dims.current_step
        return int(step[1]) if len(step) >= 2 else 0

    def _basin_str(self) -> str:
        return "prob" if self.basin_combo.currentIndex() == 0 else "flow_mag"

    def _tuning_params(self) -> SeededWatershedParams:
        return SeededWatershedParams(
            basin=self._basin_str(),
            foreground_threshold=self.single_fg_threshold.value(),
            compactness=self.single_compactness.value(),
        )

    def _sweep_spec(self) -> SeededWatershedSweepSpec:
        return SeededWatershedSweepSpec(
            basin=self._basin_str(),
            foreground_threshold=self.sweep_fg_thr[0].value(),
            foreground_threshold_min=self.sweep_fg_thr[0].value(),
            foreground_threshold_max=self.sweep_fg_thr[1].value(),
            foreground_threshold_step=self.sweep_fg_thr[2].value(),
            compactness=self.sweep_compactness[0].value(),
            compactness_min=self.sweep_compactness[0].value(),
            compactness_max=self.sweep_compactness[1].value(),
            compactness_step=self.sweep_compactness[2].value(),
        )

    def _set_status(self, msg: str) -> None:
        self.status_lbl.setText(msg)
        logger.info(msg)

    def _load_inputs(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray] | None:
        """Load prob, dp (optional for prob basin), and nucleus stacks. Returns None on error."""
        prob_path = self._prob_path()
        dp_path   = self._dp_path()
        nuc_path  = self._nucleus_tracked_path()

        if prob_path is None or not prob_path.exists():
            self._set_status(f"Missing: {prob_path}")
            return None
        if nuc_path is None or not nuc_path.exists():
            self._set_status(f"Missing nucleus tracked labels: {nuc_path}")
            return None

        prob = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
        nucleus = np.asarray(tifffile.imread(str(nuc_path)))

        dp: np.ndarray | None = None
        if self._basin_str() == "flow_mag":
            if dp_path is None or not dp_path.exists():
                self._set_status(f"Flow Magnitude basin selected but missing: {dp_path}")
                return None
            dp = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)

        # Ensure T axis
        if prob.ndim == 3:
            prob = prob[np.newaxis]
        if nucleus.ndim == 3:
            nucleus = nucleus[np.newaxis]
        if dp is not None and dp.ndim == 4:
            dp = dp[np.newaxis]

        return prob, dp, nucleus

    def _update_tracked_display(self, labels: np.ndarray, t: int | None = None) -> None:
        if _TRACKED_LAYER in self.viewer.layers and t is not None:
            layer = self.viewer.layers[_TRACKED_LAYER]
            if layer.data.ndim == 3:
                if t < layer.data.shape[0]:
                    new_data = layer.data.copy()
                    new_data[t] = labels
                    layer.data = new_data
                    return
                new_data = np.concatenate(
                    [layer.data, labels[np.newaxis].astype(layer.data.dtype)], axis=0
                )
                layer.data = new_data
                return
        display = labels[np.newaxis].copy() if labels.ndim == 2 else labels
        self._update_layer(_TRACKED_LAYER, display)

    def _update_layer(self, name: str, data: np.ndarray) -> None:
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            self.viewer.add_labels(data, name=name)

    def _refresh_db_browser(self) -> None:
        self._db_param_map = {}
        self._db_fg_vals = []
        self._db_compactness_vals = []
        self._current_db_p = None
        self.db_fg_thr_spin.setEnabled(False)
        self.db_compactness_spin.setEnabled(False)
        self.db_info_lbl.setText("—")

        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            self.status_lbl.setText("Hypothesis DB: not found.")
            return
        try:
            n_p, params_by_p = list_hypotheses(hyp_path)
        except Exception as e:
            logger.warning("Could not read hypotheses.h5: %s", e)
            self.status_lbl.setText(f"Hypothesis DB: read error — {e}")
            return

        sw_entries = {
            p: info for p, info in params_by_p.items()
            if str(info.get("method", "")) == "seeded_watershed"
        }
        if not sw_entries:
            self.status_lbl.setText(f"Hypothesis DB: {n_p} parameter set(s) (no browsable entries).")
            return

        fg_set: set[float] = set()
        compactness_set: set[float] = set()
        for p_idx, info in sw_entries.items():
            fg = round(float(info.get("foreground_threshold", 0.5)), 4)
            c  = round(float(info.get("compactness", 0.0)), 4)
            fg_set.add(fg)
            compactness_set.add(c)
            self._db_param_map[(fg, c)] = p_idx

        self._db_fg_vals = sorted(fg_set)
        self._db_compactness_vals = sorted(compactness_set)
        self._apply_db_panel()
        self.status_lbl.setText(f"Hypothesis DB: {n_p} parameter set(s).")

    def _apply_db_panel(self) -> None:
        if not self._db_fg_vals:
            return

        self.db_fg_thr_spin.blockSignals(True)
        self.db_fg_thr_spin.setMinimum(self._db_fg_vals[0])
        self.db_fg_thr_spin.setMaximum(self._db_fg_vals[-1])
        step = round(self._db_fg_vals[1] - self._db_fg_vals[0], 4) if len(self._db_fg_vals) > 1 else 0.05
        self.db_fg_thr_spin.setSingleStep(step)
        self.db_fg_thr_spin.setValue(self._db_fg_vals[0])
        self.db_fg_thr_spin.setEnabled(True)
        self.db_fg_thr_spin.blockSignals(False)

        self.db_compactness_spin.blockSignals(True)
        self.db_compactness_spin.setMinimum(self._db_compactness_vals[0])
        self.db_compactness_spin.setMaximum(self._db_compactness_vals[-1])
        step_c = round(self._db_compactness_vals[1] - self._db_compactness_vals[0], 4) if len(self._db_compactness_vals) > 1 else 0.1
        self.db_compactness_spin.setSingleStep(step_c)
        self.db_compactness_spin.setValue(self._db_compactness_vals[0])
        self.db_compactness_spin.setEnabled(len(self._db_compactness_vals) > 1 or len(self._db_fg_vals) > 1)
        self.db_compactness_spin.blockSignals(False)

        self._update_db_info_lbl()

    def _lookup_db_p(self) -> int | None:
        if not self._db_param_map:
            return None
        fg = round(self.db_fg_thr_spin.value(), 4)
        c  = round(self.db_compactness_spin.value(), 4)
        if self._db_fg_vals:
            fg = round(min(self._db_fg_vals, key=lambda x: abs(x - fg)), 4)
        if self._db_compactness_vals:
            c = round(min(self._db_compactness_vals, key=lambda x: abs(x - c)), 4)
        return self._db_param_map.get((fg, c))

    def _update_db_info_lbl(self) -> None:
        p = self._lookup_db_p()
        self._current_db_p = p
        self.db_info_lbl.setText(f"p={p:03d}" if p is not None else "—")

    def _on_db_param_changed(self) -> None:
        self._update_db_info_lbl()
        if self.db_activate_btn.isChecked() and self._current_db_p is not None:
            self._load_db_stack(self._current_db_p)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Hypothesis generation
    # ──────────────────────────────────────────────────────────────────────────

    def _on_preview(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        inputs = self._load_inputs()
        if inputs is None:
            return
        prob, dp, nucleus = inputs

        t = min(self._current_t(), prob.shape[0] - 1)
        z = min(self._current_z(), prob.shape[1] - 1)
        params = self._tuning_params()

        dp_2d = dp[t, z] if dp is not None else None
        try:
            labels = compute_seeded_watershed(prob[t, z], dp_2d, nucleus[t, z], params)
        except Exception as e:
            self._set_status(f"Preview failed: {e}")
            return

        self._update_layer(_PREVIEW_LAYER, labels)
        self._set_status(
            f"Preview t={t} z={z}: {int(labels.max())} cells "
            f"(fg_thr={params.foreground_threshold:.2f}, compactness={params.compactness:.2f})"
        )

    def _on_save_db(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        inputs = self._load_inputs()
        if inputs is None:
            return
        prob, dp, nucleus = inputs

        params    = self._tuning_params()
        overwrite = self.overwrite_check.isChecked()
        output_path = self._hyp_path()
        pos_dir     = self._pos_dir

        spec = SeededWatershedSweepSpec(
            basin=params.basin,
            foreground_threshold=params.foreground_threshold,
            foreground_threshold_min=params.foreground_threshold,
            foreground_threshold_max=params.foreground_threshold,
            foreground_threshold_step=0.05,
            compactness=params.compactness,
            compactness_min=params.compactness,
            compactness_max=params.compactness,
            compactness_step=0.1,
        )

        @thread_worker(connect={"returned": self._on_save_done, "errored": self._on_worker_error})
        def _worker():
            records = iter_seeded_watershed_records(prob, dp, nucleus, spec)
            write_hypothesis_sweep_h5(output_path, records, overwrite=overwrite, n_t=None, n_p=1)
            return pos_dir

        self._set_status("Saving to DB…")
        _worker()

    def _on_save_done(self, pos_dir: Path) -> None:
        self.output_files.refresh(pos_dir)
        self._set_status("Saved to hypotheses.h5.")
        self.refresh(pos_dir)

    def _on_run_sweep(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        inputs = self._load_inputs()
        if inputs is None:
            return
        prob, dp, nucleus = inputs

        spec      = self._sweep_spec()
        n_workers = self.sweep_n_workers.value()
        overwrite = self.overwrite_check.isChecked()
        output_path = self._hyp_path()
        pos_dir     = self._pos_dir

        params_list = build_seeded_watershed_parameter_sets(spec)
        n_t = prob.shape[0]
        total = n_t * len(params_list)

        def _on_sweep_done(result):
            self._sweep_worker = None
            self._set_sweep_buttons_running(False)
            self._on_save_done(result)

        def _on_sweep_aborted():
            self._sweep_worker = None
            self._set_sweep_buttons_running(False)
            self._set_status("Sweep cancelled.")

        def _on_sweep_error(exc):
            self._sweep_worker = None
            self._set_sweep_buttons_running(False)
            self._on_worker_error(exc)

        @thread_worker(connect={
            "yielded":  self._on_sweep_progress,
            "returned": _on_sweep_done,
            "aborted":  _on_sweep_aborted,
            "errored":  _on_sweep_error,
        })
        def _worker():
            collected: list[HypothesisRecord] = []
            for done, record in enumerate(
                iter_seeded_watershed_records(prob, dp, nucleus, spec, n_workers=n_workers), 1
            ):
                collected.append(record)
                yield (done, total, f"Sweep {done}/{total}…")
            write_hypothesis_sweep_h5(output_path, iter(collected), overwrite=overwrite)
            return pos_dir

        self._set_status(f"Running sweep ({len(params_list)} param sets × {n_t} frames)…")
        self._set_sweep_buttons_running(True)
        self.sweep_progress_bar.setRange(0, total)
        self.sweep_progress_bar.setValue(0)
        self._sweep_worker = _worker()

    def _on_sweep_progress(self, data) -> None:
        if isinstance(data, tuple):
            done, total, msg = data
            self.sweep_progress_bar.setValue(done)
            self._set_status(msg)
        else:
            self._set_status(str(data))

    def _set_sweep_buttons_running(self, running: bool) -> None:
        self.run_sweep_btn.setEnabled(not running)
        self.cancel_sweep_btn.setEnabled(running)
        self.sweep_progress_bar.setVisible(running)
        if not running:
            self.sweep_progress_bar.setValue(0)

    def _on_cancel_sweep(self) -> None:
        if self._sweep_worker is not None:
            self._sweep_worker.quit()

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Database Browser
    # ──────────────────────────────────────────────────────────────────────────

    def _on_db_activate_toggled(self, active: bool) -> None:
        self.db_activate_btn.setText("Deactivate" if active else "Activate")
        if active and self._current_db_p is not None:
            self._load_db_stack(self._current_db_p)

    def _load_db_stack(self, p: int) -> None:
        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            return
        cell_zavg_path = self._cell_zavg_path()
        nuc_zavg_path  = self._nucleus_zavg_path()
        self._set_status(f"Loading p={p}…")

        @thread_worker(connect={"returned": self._on_load_stack_done, "errored": self._on_worker_error})
        def _worker():
            stack = read_full_hypothesis_stack(hyp_path, p)
            cell_zavg = (
                np.asarray(tifffile.imread(str(cell_zavg_path)), dtype=np.float32)
                if cell_zavg_path and cell_zavg_path.exists() else None
            )
            nuc_zavg = (
                np.asarray(tifffile.imread(str(nuc_zavg_path)), dtype=np.float32)
                if nuc_zavg_path and nuc_zavg_path.exists() else None
            )
            return p, stack, cell_zavg, nuc_zavg

        _worker()

    def _on_load_stack_done(self, result: tuple) -> None:
        p, stack, cell_zavg, nuc_zavg = result
        nt = stack.shape[0]
        if _HYP_LAYER in self.viewer.layers:
            self.viewer.layers[_HYP_LAYER].data = stack
        else:
            self.viewer.add_labels(stack, name=_HYP_LAYER)
        n_cells = int(stack.max()) if stack.size > 0 else 0
        self.db_info_lbl.setText(f"p={p:03d}  |  {n_cells} cells")
        self._set_status(f"Loaded p={p} → {stack.shape} into napari.")

        for zavg_data, layer_name, cmap in (
            (cell_zavg, _CELL_ZAVG_LAYER, "gray"),
            (nuc_zavg,  _NUC_ZAVG_LAYER,  "bop orange"),
        ):
            if zavg_data is None:
                continue
            if zavg_data.ndim == 2:
                zavg_data = np.broadcast_to(zavg_data[np.newaxis], (nt,) + zavg_data.shape).copy()
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = zavg_data
            else:
                self.viewer.add_image(zavg_data, name=layer_name, colormap=cmap, blending="additive")

    def _on_set_seed(self) -> None:
        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_status("No hypothesis DB found.")
            return
        p = self._current_db_p
        if p is None:
            self._set_status("No parameter set selected in the DB browser.")
            return
        t = self._current_t()
        try:
            volume = read_hypothesis_labels(hyp_path, t, p)  # (Z, Y, X)
            z = min(self._current_z(), volume.shape[0] - 1)
            slice_2d = volume[z]
            tracked_path = self._tracked_path()
            write_tracked_frame(tracked_path, t, slice_2d)
            self._update_tracked_display(slice_2d, t=t)
            self._set_status(f"Hypothesis p={p} z={z} set as tracking seed at t={t}.")
        except Exception as e:
            self._set_status(f"Error setting seed: {e}")

    def _on_remove_stack(self) -> None:
        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_status("No hypothesis DB found.")
            return
        p = self._current_db_p
        if p is None:
            self._set_status("No parameter set selected in the DB browser.")
            return
        try:
            delete_hypothesis_parameter(hyp_path, p)
        except Exception as e:
            self._set_status(f"Remove stack failed: {e}")
            return
        if _HYP_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_HYP_LAYER])
        self._current_db_p = None
        self._set_status(f"Removed p={p}.")
        self.refresh(self._pos_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Automated search / propagation
    # ──────────────────────────────────────────────────────────────────────────

    def _on_propagate_next(self) -> None:
        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_status("No hypothesis DB found.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_status("No tracked layer loaded. Set a seed first.")
            return
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3:
            self._set_status("Tracked layer is not a 3D stack.")
            return
        t = self._current_t()
        if t >= layer.data.shape[0]:
            self._set_status(f"No tracked frame at t={t}. Set a seed first.")
            return

        current_labels = np.asarray(layer.data[t])
        prev_labels = np.asarray(layer.data[t - 1]) if t > 0 else None

        try:
            next_frame, winner = propagate_one_frame(
                hyp_path, current_labels, t + 1, prev_labels,
                iou_threshold=self.iou_spin.value(),
                max_dist_px=self.dist_spin.value(),
                velocity_sigma_px=self.vel_sigma_spin.value(),
                iou_weight=self.iou_weight_spin.value(),
                area_weight=self.area_weight_spin.value(),
                velocity_weight=self.vel_weight_spin.value(),
                pos_weight=self.pos_weight_spin.value(),
                unmatched_score=self.unmatched_spin.value(),
            )
        except Exception as e:
            self._set_status(f"Propagation failed: {e}")
            return

        if next_frame is None:
            self._set_status(f"No suitable hypothesis found for t={t + 1}.")
            return

        self._update_tracked_display(next_frame, t=t + 1)
        step = list(self.viewer.dims.current_step)
        step[0] = t + 1
        self.viewer.dims.current_step = tuple(step)
        self._set_status(f"Propagated t={t}→{t + 1} using p={winner}. Unsaved.")

    def _on_propagate_all(self) -> None:
        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_status("No hypothesis DB found.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_status("No tracked layer loaded. Set a seed first.")
            return
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3:
            self._set_status("Tracked layer is not a 3D stack.")
            return
        t_start = self._current_t()
        if t_start >= layer.data.shape[0]:
            self._set_status(f"No tracked frame at t={t_start}. Set a seed first.")
            return

        initial_labels = np.asarray(layer.data[t_start])
        prev_labels    = np.asarray(layer.data[t_start - 1]) if t_start > 0 else None

        iou_thr   = self.iou_spin.value()
        max_dist  = self.dist_spin.value()
        vel_sigma = self.vel_sigma_spin.value()
        iou_w     = self.iou_weight_spin.value()
        area_w    = self.area_weight_spin.value()
        vel_w     = self.vel_weight_spin.value()
        pos_w     = self.pos_weight_spin.value()
        unmatch_s = self.unmatched_spin.value()
        self._stop_flag = False

        @thread_worker(connect={"yielded": self._on_prop_progress, "finished": self._on_prop_done, "errored": self._on_worker_error})
        def _worker():
            current = initial_labels
            prev    = prev_labels
            t = t_start
            while not self._stop_flag:
                next_frame, winner = propagate_one_frame(
                    hyp_path, current, t + 1, prev,
                    iou_threshold=iou_thr,
                    max_dist_px=max_dist,
                    velocity_sigma_px=vel_sigma,
                    iou_weight=iou_w,
                    area_weight=area_w,
                    velocity_weight=vel_w,
                    pos_weight=pos_w,
                    unmatched_score=unmatch_s,
                )
                if next_frame is None:
                    yield (t, None, None)
                    break
                yield (t, next_frame, winner)
                prev    = current
                current = next_frame
                t += 1

        self._set_status("Propagating…")
        _worker()

    def _on_prop_progress(self, result: tuple) -> None:
        t, next_frame, winner = result
        if next_frame is None:
            self._set_status(f"Propagation stopped at t={t}: no suitable hypothesis.")
        else:
            self._set_status(f"Propagated t={t}→{t + 1} (p={winner}). Unsaved.")
            self._update_tracked_display(next_frame, t=t + 1)
            step = list(self.viewer.dims.current_step)
            step[0] = t + 1
            self.viewer.dims.current_step = tuple(step)

    def _on_prop_done(self) -> None:
        self._set_status("Propagation complete.")

    def _on_save_tracked(self) -> None:
        tracked_path = self._tracked_path()
        if tracked_path is None:
            self._set_status("No project open.")
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_status("No tracked layer to save.")
            return
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3:
            self._set_status("Tracked layer is not a 3D stack.")
            return
        n = layer.data.shape[0]
        for t in range(n):
            write_tracked_frame(tracked_path, t, np.asarray(layer.data[t]))
        self._set_status(f"Saved {n} frame(s) to {tracked_path.name}.")

    def _on_load_tracked(self) -> None:
        tracked_path   = self._tracked_path()
        cell_zavg_path = self._cell_zavg_path()
        nuc_zavg_path  = self._nucleus_zavg_path()
        if tracked_path is None or not tracked_path.exists():
            self._set_status("No tracked labels file found.")
            return
        self._set_status("Loading tracked labels…")

        @thread_worker(connect={"returned": self._on_load_tracked_done, "errored": self._on_worker_error})
        def _worker():
            stack = read_full_tracked_stack(tracked_path)
            cell_zavg = (
                np.asarray(tifffile.imread(str(cell_zavg_path)), dtype=np.float32)
                if cell_zavg_path and cell_zavg_path.exists() else None
            )
            nuc_zavg = (
                np.asarray(tifffile.imread(str(nuc_zavg_path)), dtype=np.float32)
                if nuc_zavg_path and nuc_zavg_path.exists() else None
            )
            return stack, cell_zavg, nuc_zavg

        _worker()

    def _on_load_tracked_done(self, result: tuple) -> None:
        stack, cell_zavg, nuc_zavg = result
        nt = stack.shape[0]
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = stack
        else:
            self.viewer.add_labels(stack, name=_TRACKED_LAYER)

        for zavg_data, layer_name, cmap in (
            (cell_zavg, _CELL_ZAVG_LAYER, "gray"),
            (nuc_zavg,  _NUC_ZAVG_LAYER,  "bop orange"),
        ):
            if zavg_data is None:
                continue
            if zavg_data.ndim == 2:
                zavg_data = np.broadcast_to(zavg_data[np.newaxis], (nt,) + zavg_data.shape).copy()
            else:
                zavg_data = zavg_data
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = zavg_data
            else:
                self.viewer.add_image(zavg_data, name=layer_name, colormap=cmap, blending="additive")

        self._set_status(f"Loaded tracked stack {stack.shape} into napari.")
        layer = self.viewer.layers[_TRACKED_LAYER]
        self.correction_widget.activate_layer(layer)
        self.correction_section.expand()

    def _on_reassign_ids(self) -> None:
        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_status("No tracked layer loaded.")
            return
        stack = np.asarray(self.viewer.layers[_TRACKED_LAYER].data)
        self._set_status("Reassigning cell IDs to contiguous range…")

        @thread_worker(connect={"returned": self._on_reassign_ids_done, "errored": self._on_worker_error})
        def _worker():
            unique_ids = np.unique(stack)
            unique_ids = unique_ids[unique_ids != 0]
            if unique_ids.size == 0:
                return stack, 0
            lut = np.zeros(int(unique_ids.max()) + 1, dtype=np.uint32)
            for new_id, old_id in enumerate(unique_ids, start=1):
                lut[old_id] = new_id
            return lut[stack], len(unique_ids)

        _worker()

    def _on_reassign_ids_done(self, result: tuple) -> None:
        remapped, n_cells = result
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = remapped
        self._set_status(f"Reassigned {n_cells} cell IDs to contiguous range 1–{n_cells}. Unsaved.")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Manual correction
    # ──────────────────────────────────────────────────────────────────────────

    def _on_dims_step_changed(self, event=None) -> None:
        self._refresh_validate_btn()

    def _refresh_validate_btn(self) -> None:
        if self._pos_dir is None:
            self.validate_btn.setChecked(False)
            return
        t = self._current_t()
        validated = is_validated(self._pos_dir, t)
        self.validate_btn.blockSignals(True)
        self.validate_btn.setChecked(validated)
        self.validate_btn.blockSignals(False)

    def _on_validate_toggled(self, checked: bool) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        t = self._current_t()
        if checked:
            validate_frame(self._pos_dir, t)
            self._set_status(f"Frame t={t} marked as validated.")
        else:
            invalidate_frame(self._pos_dir, t)
            self._set_status(f"Frame t={t} validation removed.")

    def _on_retrack_frame(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        t = self._current_t()
        if is_validated(self._pos_dir, t):
            self._set_status(f"Frame t={t} is validated — unvalidate it first to retrack.")
            return

        validated = sorted(
            [v for v in read_validated_frames(self._pos_dir) if v < t], reverse=True
        )
        t_ref = validated[0] if validated else (t - 1)
        if t_ref < 0:
            self._set_status("No reference frame available (t=0 has no predecessor).")
            return

        if _TRACKED_LAYER not in self.viewer.layers:
            self._set_status("No tracked layer loaded.")
            return
        layer = self.viewer.layers[_TRACKED_LAYER]
        if layer.data.ndim != 3 or t >= layer.data.shape[0] or t_ref >= layer.data.shape[0]:
            self._set_status(f"Frame t={t} or reference t={t_ref} not in tracked layer.")
            return

        ref_labels = np.asarray(layer.data[t_ref])
        tgt_labels = np.asarray(layer.data[t])

        remapped = retrack_frame(ref_labels, tgt_labels, max_dist_px=self.dist_spin.value())
        new_ids = set(int(i) for i in np.unique(remapped) if i != 0)
        ref_ids = set(int(i) for i in np.unique(ref_labels) if i != 0)
        n_matched = len(new_ids & ref_ids)
        n_new     = len(new_ids - ref_ids)

        self._update_tracked_display(remapped, t=t)
        self._set_status(
            f"Retracked t={t} using t={t_ref}: {n_matched} matched, {n_new} new ID(s). Unsaved."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Error handler
    # ──────────────────────────────────────────────────────────────────────────

    def _on_worker_error(self, exc: Exception) -> None:
        self._set_status(f"Error: {exc}")
        logger.exception("Worker error", exc_info=exc)

    # ──────────────────────────────────────────────────────────────────────────
    # State persistence
    # ──────────────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        return {
            "overwrite":   self.overwrite_check.isChecked(),
            "basin":       self.basin_combo.currentIndex(),
            "tuning": {
                "fg_threshold": self.single_fg_threshold.value(),
                "compactness":  self.single_compactness.value(),
            },
            "sweep": {
                "fg_thr_min":        self.sweep_fg_thr[0].value(),
                "fg_thr_max":        self.sweep_fg_thr[1].value(),
                "fg_thr_step":       self.sweep_fg_thr[2].value(),
                "compactness_min":   self.sweep_compactness[0].value(),
                "compactness_max":   self.sweep_compactness[1].value(),
                "compactness_step":  self.sweep_compactness[2].value(),
                "n_workers":         self.sweep_n_workers.value(),
            },
            "db_browser": {
                "fg_threshold": self.db_fg_thr_spin.value(),
                "compactness":  self.db_compactness_spin.value(),
            },
            "search": {
                "iou_threshold":    self.iou_spin.value(),
                "max_dist_px":      self.dist_spin.value(),
                "velocity_sigma_px": self.vel_sigma_spin.value(),
                "iou_weight":       self.iou_weight_spin.value(),
                "area_weight":      self.area_weight_spin.value(),
                "velocity_weight":  self.vel_weight_spin.value(),
                "pos_weight":       self.pos_weight_spin.value(),
                "unmatched_score":  self.unmatched_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        if "overwrite" in state:
            self.overwrite_check.setChecked(state["overwrite"])
        if "basin" in state:
            self.basin_combo.setCurrentIndex(state["basin"])
        if "tuning" in state:
            t = state["tuning"]
            if "fg_threshold" in t: self.single_fg_threshold.setValue(t["fg_threshold"])
            if "compactness"  in t: self.single_compactness.setValue(t["compactness"])
        if "sweep" in state:
            sw = state["sweep"]
            if "fg_thr_min"       in sw: self.sweep_fg_thr[0].setValue(sw["fg_thr_min"])
            if "fg_thr_max"       in sw: self.sweep_fg_thr[1].setValue(sw["fg_thr_max"])
            if "fg_thr_step"      in sw: self.sweep_fg_thr[2].setValue(sw["fg_thr_step"])
            if "compactness_min"  in sw: self.sweep_compactness[0].setValue(sw["compactness_min"])
            if "compactness_max"  in sw: self.sweep_compactness[1].setValue(sw["compactness_max"])
            if "compactness_step" in sw: self.sweep_compactness[2].setValue(sw["compactness_step"])
            if "n_workers"        in sw: self.sweep_n_workers.setValue(sw["n_workers"])
        if "db_browser" in state:
            db = state["db_browser"]
            if "fg_threshold" in db: self.db_fg_thr_spin.setValue(db["fg_threshold"])
            if "compactness"  in db: self.db_compactness_spin.setValue(db["compactness"])
        if "search" in state:
            se = state["search"]
            if "iou_threshold"     in se: self.iou_spin.setValue(se["iou_threshold"])
            if "max_dist_px"       in se: self.dist_spin.setValue(se["max_dist_px"])
            if "velocity_sigma_px" in se: self.vel_sigma_spin.setValue(se["velocity_sigma_px"])
            if "iou_weight"        in se: self.iou_weight_spin.setValue(se["iou_weight"])
            if "area_weight"       in se: self.area_weight_spin.setValue(se["area_weight"])
            if "velocity_weight"   in se: self.vel_weight_spin.setValue(se["velocity_weight"])
            if "pos_weight"        in se: self.pos_weight_spin.setValue(se["pos_weight"])
            if "unmatched_score"   in se: self.unmatched_spin.setValue(se["unmatched_score"])
