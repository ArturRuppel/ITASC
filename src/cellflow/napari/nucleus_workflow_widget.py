"""Nucleus workflow widget for hypothesis generation and tracking in CellFlow v2."""
from __future__ import annotations

import logging
import shlex
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cellflow.database.hypotheses import (
    NucleusHypothesisSweepSpec,
    HypothesisRecord,
    build_parameter_sets,
    delete_hypothesis_parameter,
    iter_cellpose_flow_records_from_stacks,
    iter_hypothesis_records_from_stacks,
    list_hypotheses,
    read_full_hypothesis_stack,
    read_hypothesis_labels,
    write_hypothesis_sweep_h5,
    zero_hypothesis_slice,
)
from cellflow.database.tracked import (
    read_full_tracked_stack,
    read_tracked_frame,
    tracked_frame_exists,
    tracked_n_frames,
    write_tracked_frame,
)
from cellflow.database.validation import (
    invalidate_frame,
    is_validated,
    read_validated_frames,
    validate_frame,
)
from cellflow.segmentation import CellposeFlowHypothesisParams, NucleusHypothesisParams, compute_hypothesis_labels
from cellflow.tracking import propagate_one_frame
from cellflow.tracking.retracker import retrack_frame

logger = logging.getLogger(__name__)

_PREVIEW_LAYER = "Preview: Nucleus"
_HYP_LAYER = "Hypothesis: Nucleus"
_TRACKED_LAYER = "Tracked: Nucleus"
_PROB_LAYER = "Probability: Nucleus"
_CELL_ZAVG_LAYER = "Cell z-avg"
_NUC_ZAVG_LAYER = "Nucleus z-avg"


class NucleusWorkflowWidget(QWidget):
    """Nucleus hypothesis generation and tracking management."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._stop_flag: bool = False
        self._sweep_worker = None
        self._current_db_p: int | None = None
        self._setup_ui()
        self._connect_signals()

    # ──────────────────────────────────────────────────────────────────────────
    # UI setup (unchanged from original)
    # ──────────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        # ── 1. Hypothesis Generation ──────────────────────────────────────
        gen_group = QGroupBox("1. Hypothesis Generation")
        gen_lay = QVBoxLayout(gen_group)
        gen_lay.setSpacing(6)

        shared_lay = QVBoxLayout()

        row_seeds = QHBoxLayout()
        row_seeds.addWidget(QLabel("Seed Source:"))
        self.seed_source_combo = QComboBox()
        self.seed_source_combo.addItems(["Peak local max", "Active Layer", "Disk (Corrected)"])
        row_seeds.addWidget(self.seed_source_combo)
        shared_lay.addLayout(row_seeds)

        row_min_size = QHBoxLayout()
        row_min_size.addWidget(QLabel("Min Cell Size (px):"))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 100000)
        self.min_size_spin.setValue(0)
        self.min_size_spin.setToolTip("Remove connected regions smaller than this many pixels (0 = keep all)")
        row_min_size.addWidget(self.min_size_spin)
        shared_lay.addLayout(row_min_size)

        self.overwrite_check = QCheckBox("Overwrite existing in DB")
        self.overwrite_check.setChecked(False)
        shared_lay.addWidget(self.overwrite_check)
        gen_lay.addLayout(shared_lay)

        self.gen_tabs = QTabWidget()

        # Tab 1: Single ("Tuning")
        single_tab = QWidget()
        single_lay = QVBoxLayout(single_tab)

        def _add_single_param(label, min_val, max_val, default, step, decimals=1):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox() if decimals > 0 else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(default)
            if decimals > 0:
                spin.setDecimals(decimals)
                spin.setSingleStep(step)
            row.addWidget(spin)
            single_lay.addLayout(row)
            return spin

        row_z = QHBoxLayout()
        row_z.addWidget(QLabel("Z Slice:"))
        self.z_slice_spin = QSpinBox()
        self.z_slice_spin.setRange(0, 99)
        self.z_slice_spin.setValue(0)
        row_z.addWidget(self.z_slice_spin)
        single_lay.addLayout(row_z)

        self.single_thr = _add_single_param("Threshold (%)", 0.0, 100.0, 30.0, 1.0)
        self.single_cmp = _add_single_param("Compactness", 0.0, 1.0, 0.0, 0.01, 2)
        self.single_sigma = _add_single_param("Smooth Sigma", 0.0, 10.0, 0.5, 0.1, 1)
        self.single_seed_dist = _add_single_param("Seed Dist", 1, 500, 5, 1, 0)

        btn_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.save_db_btn = QPushButton("Save to DB")
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.save_db_btn)
        single_lay.addLayout(btn_row)

        self.gen_tabs.addTab(single_tab, "Tuning (Single)")

        # Tab 2: Parameter Sweep ("Batch")
        sweep_tab = QWidget()
        sweep_lay = QVBoxLayout(sweep_tab)

        def _add_sweep_row(label, d_min, d_max, d_step, decimals=1):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            min_s = QDoubleSpinBox() if decimals > 0 else QSpinBox()
            max_s = QDoubleSpinBox() if decimals > 0 else QSpinBox()
            step_s = QDoubleSpinBox() if decimals > 0 else QSpinBox()
            for s in (min_s, max_s, step_s):
                if "Seed Dist" in label:
                    s.setRange(1, 500)
                else:
                    s.setRange(0, 100 if "Threshold" in label else 10)
                if decimals > 0:
                    s.setDecimals(decimals)
            min_s.setValue(d_min)
            max_s.setValue(d_max)
            step_s.setValue(d_step)
            row.addWidget(QLabel("min"))
            row.addWidget(min_s)
            row.addWidget(QLabel("max"))
            row.addWidget(max_s)
            row.addWidget(QLabel("step"))
            row.addWidget(step_s)
            sweep_lay.addLayout(row)
            return min_s, max_s, step_s

        self.sweep_z_slice = _add_sweep_row("Z Slice", 0, 5, 1, 0)
        self.sweep_thr = _add_sweep_row("Threshold (%)", 10, 50, 10)
        self.sweep_cmp = _add_sweep_row("Compactness", 0, 0.1, 0.05, 2)
        self.sweep_sigma = _add_sweep_row("Smooth Sigma", 0, 1.0, 0.5, 1)
        self.sweep_seed_dist = _add_sweep_row("Seed Dist", 5, 20, 5, 0)

        sweep_btn_row = QHBoxLayout()
        self.run_sweep_btn = QPushButton("Run Batch Sweep")
        self.run_terminal_btn = QPushButton("Run in Terminal")
        self.cancel_sweep_btn = QPushButton("Cancel")
        self.cancel_sweep_btn.setEnabled(False)
        sweep_btn_row.addWidget(self.run_sweep_btn)
        sweep_btn_row.addWidget(self.run_terminal_btn)
        sweep_btn_row.addWidget(self.cancel_sweep_btn)
        sweep_lay.addLayout(sweep_btn_row)

        self.gen_tabs.addTab(sweep_tab, "Batch (Sweep)")

        # Tab 3: Cellpose Native
        cp_tab = QWidget()
        cp_lay = QVBoxLayout(cp_tab)

        cp_note = QLabel(
            "Requires nucleus_dp_3dt.tif alongside nucleus_prob_3dt.tif. "
            "Flow Threshold 0 = keep all masks; increase to filter low-quality masks."
        )
        cp_note.setWordWrap(True)
        cp_note.setStyleSheet("font-size: 8pt; color: #aaaaaa;")
        cp_lay.addWidget(cp_note)

        def _add_cp_param(label, min_val, max_val, default, step, decimals):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox() if decimals > 0 else QSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(default)
            if decimals > 0:
                spin.setDecimals(decimals)
                spin.setSingleStep(step)
            else:
                spin.setSingleStep(int(step))
            row.addWidget(spin)
            cp_lay.addLayout(row)
            return spin

        row_cp_z = QHBoxLayout()
        row_cp_z.addWidget(QLabel("Z Slice:"))
        self.cp_z_spin = QSpinBox()
        self.cp_z_spin.setRange(0, 99)
        self.cp_z_spin.setValue(0)
        row_cp_z.addWidget(self.cp_z_spin)
        cp_lay.addLayout(row_cp_z)

        self.cp_cellprob_spin = _add_cp_param("Cell Prob Threshold", -10.0, 10.0, 0.0, 0.5, 1)
        self.cp_flow_spin = _add_cp_param("Flow Threshold", 0.0, 1.0, 0.0, 0.05, 2)
        self.cp_niter_spin = _add_cp_param("Niter (0=auto)", 0, 1000, 0, 50, 0)

        cp_btn_row = QHBoxLayout()
        self.cp_preview_btn = QPushButton("Preview")
        self.cp_save_btn = QPushButton("Save to DB")
        cp_btn_row.addWidget(self.cp_preview_btn)
        cp_btn_row.addWidget(self.cp_save_btn)
        cp_lay.addLayout(cp_btn_row)

        self.gen_tabs.addTab(cp_tab, "Cellpose Native")
        gen_lay.addWidget(self.gen_tabs)
        layout.addWidget(gen_group)

        # ── 2. Database Browser ──────────────────────────────────────────
        db_group = QGroupBox("2. Database Browser")
        db_lay = QVBoxLayout(db_group)

        hdr_row = QHBoxLayout()
        self.db_activate_btn = QPushButton("Activate")
        self.db_activate_btn.setCheckable(True)
        self.db_activate_btn.setChecked(False)
        self.db_activate_btn.setToolTip("Activate database browser — enables live data loading")
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

        self.db_tree = QTreeWidget()
        self.db_tree.setHeaderHidden(True)
        self.db_tree.setColumnCount(1)
        self.db_tree.setIndentation(14)
        self.db_tree.setMinimumHeight(120)
        self.db_tree.setToolTip("Click a parameter set to load it")
        db_lay.addWidget(self.db_tree)

        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Display Z:"))
        self.db_z_spin = QSpinBox()
        self.db_z_spin.setRange(0, 99)
        self.db_z_spin.setValue(0)
        self.db_z_spin.setToolTip("Z-slice to display (only affects multi-z results)")
        z_row.addWidget(self.db_z_spin)
        z_row.addStretch()
        db_lay.addLayout(z_row)

        db_btn_row = QHBoxLayout()
        self.set_seed_btn = QPushButton("Set as Tracking Seed")
        db_btn_row.addWidget(self.set_seed_btn)
        db_lay.addLayout(db_btn_row)

        db_del_row = QHBoxLayout()
        self.del_slice_btn = QPushButton("Delete Current Slice")
        self.del_slice_btn.setToolTip(
            "Zero out the current (t, z) plane for the selected parameter set"
        )
        self.del_stack_btn = QPushButton("Remove Stack")
        self.del_stack_btn.setToolTip(
            "Delete all timepoints for the selected parameter set from the DB"
        )
        self.del_stack_btn.setStyleSheet(
            "QPushButton { color: #cc3333; }"
            "QPushButton:hover { background-color: #4a1111; color: white; }"
        )
        db_del_row.addWidget(self.del_slice_btn)
        db_del_row.addWidget(self.del_stack_btn)
        db_lay.addLayout(db_del_row)
        layout.addWidget(db_group)

        # ── 3. Automated Search ──────────────────────────────────────────
        search_group = QGroupBox("3. Automated Search")
        search_lay = QVBoxLayout(search_group)

        row_iou = QHBoxLayout()
        row_iou.addWidget(QLabel("IoU Threshold:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0, 1)
        self.iou_spin.setValue(0.5)
        self.iou_spin.setSingleStep(0.1)
        row_iou.addWidget(self.iou_spin)
        search_lay.addLayout(row_iou)

        row_dist = QHBoxLayout()
        row_dist.addWidget(QLabel("Max Dist (µm):"))
        self.dist_spin = QDoubleSpinBox()
        self.dist_spin.setRange(0, 1000)
        self.dist_spin.setValue(20.0)
        row_dist.addWidget(self.dist_spin)
        search_lay.addLayout(row_dist)

        row_vel = QHBoxLayout()
        row_vel.addWidget(QLabel("Velocity σ (px):"))
        self.vel_sigma_spin = QDoubleSpinBox()
        self.vel_sigma_spin.setRange(1, 500)
        self.vel_sigma_spin.setValue(25.0)
        self.vel_sigma_spin.setSingleStep(5.0)
        self.vel_sigma_spin.setToolTip(
            "Gaussian sigma for velocity consistency scoring. "
            "Larger values = more permissive, disabled on first frame."
        )
        row_vel.addWidget(self.vel_sigma_spin)
        search_lay.addLayout(row_vel)

        def _weight_spin(default):
            w = QDoubleSpinBox()
            w.setRange(0.0, 10.0)
            w.setValue(default)
            w.setSingleStep(0.5)
            w.setDecimals(1)
            return w

        row_iou_w = QHBoxLayout()
        row_iou_w.addWidget(QLabel("IoU Weight:"))
        self.iou_weight_spin = _weight_spin(1.0)
        self.iou_weight_spin.setToolTip("Exponent on IoU term. 0 = ignored, 1 = linear, >1 = amplified.")
        row_iou_w.addWidget(self.iou_weight_spin)
        search_lay.addLayout(row_iou_w)

        row_area_w = QHBoxLayout()
        row_area_w.addWidget(QLabel("Area Weight:"))
        self.area_weight_spin = _weight_spin(1.0)
        self.area_weight_spin.setToolTip("Exponent on area ratio term.")
        row_area_w.addWidget(self.area_weight_spin)
        search_lay.addLayout(row_area_w)

        row_vel_w = QHBoxLayout()
        row_vel_w.addWidget(QLabel("Velocity Weight:"))
        self.vel_weight_spin = _weight_spin(1.0)
        self.vel_weight_spin.setToolTip("Exponent on velocity Gaussian term. No effect on frame 0.")
        row_vel_w.addWidget(self.vel_weight_spin)
        search_lay.addLayout(row_vel_w)

        prop_row = QHBoxLayout()
        self.prop_next_btn = QPushButton("Propagate Next")
        self.prop_all_btn = QPushButton("Propagate All")
        self.stop_btn = QPushButton("Stop")
        prop_row.addWidget(self.prop_next_btn)
        prop_row.addWidget(self.prop_all_btn)
        prop_row.addWidget(self.stop_btn)
        search_lay.addLayout(prop_row)

        self.load_tracked_btn = QPushButton("Load Tracked Labels")
        self.load_tracked_btn.setToolTip(
            "Load full tracked label stack with cell/nucleus z-avg into napari"
        )
        search_lay.addWidget(self.load_tracked_btn)
        layout.addWidget(search_group)

        # ── 4. Manual Correction Integration ──────────────────────────────
        corr_group = QGroupBox("4. Manual Correction")
        corr_lay = QVBoxLayout(corr_group)
        self.jump_corr_btn = QPushButton("Correct Current Frame")
        self.jump_corr_btn.setStyleSheet("font-weight: bold; min-height: 28px;")
        corr_lay.addWidget(self.jump_corr_btn)

        # Retrack row
        retrack_row = QHBoxLayout()
        self.retrack_btn = QPushButton("Retrack Frame")
        self.retrack_btn.setToolTip(
            "Remap cell IDs in the current frame to match the nearest preceding "
            "validated frame (or t-1 if none exists).  Validated frames are "
            "protected and cannot be retracted."
        )
        retrack_row.addWidget(self.retrack_btn)
        self.validate_btn = QPushButton("Validate Frame")
        self.validate_btn.setCheckable(True)
        self.validate_btn.setToolTip(
            "Mark the current frame as a trusted ID anchor.  Validated frames "
            "are skipped by Retrack and used as reference when retracking neighbours."
        )
        retrack_row.addWidget(self.validate_btn)
        corr_lay.addLayout(retrack_row)

        layout.addWidget(corr_group)

        # Status label (added at bottom for feedback)
        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        layout.addWidget(self.status_lbl)

    # ──────────────────────────────────────────────────────────────────────────
    # Signal wiring
    # ──────────────────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.preview_btn.clicked.connect(self._on_preview)
        self.save_db_btn.clicked.connect(self._on_save_db)
        self.run_sweep_btn.clicked.connect(self._on_run_sweep)
        self.run_terminal_btn.clicked.connect(self._on_run_terminal)
        self.cancel_sweep_btn.clicked.connect(self._on_cancel_sweep)
        self.db_tree.currentItemChanged.connect(self._on_tree_item_selected)
        self.db_z_spin.valueChanged.connect(self._on_db_z_changed)
        self.set_seed_btn.clicked.connect(self._on_set_seed)
        self.db_activate_btn.toggled.connect(self._on_db_activate_toggled)
        self.db_refresh_btn.clicked.connect(lambda: self.refresh(self._pos_dir))
        self.del_slice_btn.clicked.connect(self._on_delete_slice)
        self.del_stack_btn.clicked.connect(self._on_remove_stack)
        self.cp_preview_btn.clicked.connect(self._on_preview_cellpose_native)
        self.cp_save_btn.clicked.connect(self._on_save_cellpose_native_db)
        self.prop_next_btn.clicked.connect(self._on_propagate_next)
        self.prop_all_btn.clicked.connect(self._on_propagate_all)
        self.stop_btn.clicked.connect(lambda: setattr(self, "_stop_flag", True))
        self.load_tracked_btn.clicked.connect(self._on_load_tracked)
        self.jump_corr_btn.clicked.connect(self._on_jump_correction)
        self.retrack_btn.clicked.connect(self._on_retrack_frame)
        self.validate_btn.toggled.connect(self._on_validate_toggled)
        self.viewer.dims.events.current_step.connect(self._on_dims_step_changed)

    # ──────────────────────────────────────────────────────────────────────────
    # Public refresh (called by main_widget on project change)
    # ──────────────────────────────────────────────────────────────────────────

    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        if pos_dir is None:
            return

        self._refresh_db_tree()
        self._refresh_validate_btn()


    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _current_t(self) -> int:
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) >= 1 else 0

    def _hyp_path(self) -> Path | None:
        if self._pos_dir is None:
            return None
        return self._pos_dir / "2_nucleus" / "hypotheses.h5"

    def _tracked_path(self) -> Path | None:
        if self._pos_dir is None:
            return None
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif"

    def _prob_path(self) -> Path | None:
        if self._pos_dir is None:
            return None
        return self._pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif"

    def _dp_path(self) -> Path | None:
        if self._pos_dir is None:
            return None
        return self._pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif"

    def _cell_zavg_path(self) -> Path | None:
        if self._pos_dir is None:
            return None
        return self._pos_dir / "0_input" / "cell_zavg.tif"

    def _nucleus_zavg_path(self) -> Path | None:
        if self._pos_dir is None:
            return None
        return self._pos_dir / "0_input" / "nucleus_zavg.tif"

    def _get_nz(self) -> int:
        """Return the Z dimension from any loaded 4D napari layer or the hypothesis DB."""
        for layer in self.viewer.layers:
            if hasattr(layer, "data") and layer.data.ndim == 4:
                return layer.data.shape[1]
        hyp_path = self._hyp_path()
        if hyp_path and hyp_path.exists():
            try:
                return read_hypothesis_labels(hyp_path, 0, 0).shape[0]
            except Exception:
                pass
        return 1

    def _update_tracked_display(
        self,
        labels: np.ndarray,
        t: int | None = None,
        tracked_path: Path | None = None,
    ) -> None:
        """Update the tracked labels layer with a single (Y, X) frame at timepoint t.

        If the layer already holds a (T, Y, X) stack and t is within bounds, updates
        that frame in-place. If t is out of bounds (new frame extends the stack), reloads
        the full stack from tracked_path. Falls back to a single-frame (1, Y, X) layer
        if no path is available.
        """
        if _TRACKED_LAYER in self.viewer.layers and t is not None:
            layer = self.viewer.layers[_TRACKED_LAYER]
            if layer.data.ndim == 3:
                if t < layer.data.shape[0]:
                    new_data = layer.data.copy()
                    new_data[t] = labels
                    layer.data = new_data
                    return
                if tracked_path is not None and tracked_path.exists():
                    layer.data = read_full_tracked_stack(tracked_path)
                    return
        display = labels[np.newaxis].copy() if labels.ndim == 2 else labels
        self._update_layer(_TRACKED_LAYER, display)

    def _update_layer(self, name: str, data: np.ndarray) -> None:
        """Add or replace a Labels layer in the viewer."""
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
        else:
            self.viewer.add_labels(data, name=name)

    def _refresh_db_tree(self) -> None:
        """Repopulate the hypothesis tree from the DB on disk."""
        self.db_tree.clear()
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

        watershed_entries = [(p, info) for p, info in sorted(params_by_p.items())
                             if str(info.get("method", "watershed")) != "cellpose_flow"]
        cellpose_entries = [(p, info) for p, info in sorted(params_by_p.items())
                            if str(info.get("method", "watershed")) == "cellpose_flow"]

        _SKIP_ATTRS = {"parameter_index", "parameter_json", "label_shape", "label_dtype"}

        def _add_group(group_label: str, entries: list) -> None:
            if not entries:
                return
            group_item = QTreeWidgetItem([f"{group_label}  ({len(entries)})"])
            group_item.setData(0, Qt.ItemDataRole.UserRole, None)
            group_item.setFlags(group_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            font = group_item.font(0)
            font.setBold(True)
            group_item.setFont(0, font)
            self.db_tree.addTopLevelItem(group_item)
            for p_idx, info in entries:
                method = str(info.get("method", "watershed"))
                if method == "cellpose_flow":
                    summary = (
                        f"p{p_idx:03d}  "
                        f"prob={float(info.get('cellprob_threshold', 0)):.2f}  "
                        f"flow={float(info.get('flow_threshold', 0)):.2f}  "
                        f"n={int(info.get('niter', 0))}"
                    )
                else:
                    summary = (
                        f"p{p_idx:03d}  "
                        f"z={int(info.get('z_slice', 0))}  "
                        f"thr={float(info.get('threshold_pct', 0)):.0f}%  "
                        f"σ={float(info.get('smooth_sigma', 0)):.2f}  "
                        f"dist={int(info.get('seed_distance', 0))}"
                    )
                p_item = QTreeWidgetItem([summary])
                p_item.setData(0, Qt.ItemDataRole.UserRole, p_idx)
                for k, v in sorted(info.items()):
                    if k in _SKIP_ATTRS:
                        continue
                    child = QTreeWidgetItem([f"  {k}:  {v}"])
                    child.setData(0, Qt.ItemDataRole.UserRole, None)
                    child.setFlags(child.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                    p_item.addChild(child)
                group_item.addChild(p_item)
            group_item.setExpanded(True)

        _add_group("Watershed", watershed_entries)
        _add_group("Cellpose Native", cellpose_entries)

        self.status_lbl.setText(f"Hypothesis DB: {n_p} parameter set(s).")

    def _single_params(self) -> NucleusHypothesisParams:
        seed_source = self.seed_source_combo.currentText()
        src = "auto" if seed_source == "Peak local max" else "layer"
        return NucleusHypothesisParams(
            basin="prob",
            threshold_pct=self.single_thr.value(),
            compactness=self.single_cmp.value(),
            smooth_sigma=self.single_sigma.value(),
            seed_source=src,
            seed_distance=self.single_seed_dist.value(),
            min_size=self.min_size_spin.value(),
            z_slice=self.z_slice_spin.value(),
        )

    def _set_status(self, msg: str) -> None:
        self.status_lbl.setText(msg)
        logger.info(msg)

    # ──────────────────────────────────────────────────────────────────────────
    # Button handlers
    # ──────────────────────────────────────────────────────────────────────────

    def _on_preview(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        prob_path = self._prob_path()
        if prob_path is None or not prob_path.exists():
            self._set_status(f"Missing: {prob_path}")
            return

        try:
            prob_stack = tifffile.imread(str(prob_path))  # (T, Z, Y, X) or (Z, Y, X)
            prob_stack = np.asarray(prob_stack, dtype=np.float32)
        except Exception as e:
            self._set_status(f"Could not read prob file: {e}")
            return

        if prob_stack.ndim == 3:
            prob_stack = prob_stack[np.newaxis]

        z = self.z_slice_spin.value()
        if z >= prob_stack.shape[1]:
            self._set_status(f"z={z} out of range (stack has {prob_stack.shape[1]} slices)")
            return

        # Load the 2DT movie at the selected z-slice into the viewer
        prob_2dt = prob_stack[:, z]  # (T, Y, X)
        if _PROB_LAYER in self.viewer.layers:
            self.viewer.layers[_PROB_LAYER].data = prob_2dt
        else:
            self.viewer.add_image(prob_2dt, name=_PROB_LAYER, colormap="inferno", blending="additive")

        t = self._current_t()
        if t >= prob_stack.shape[0]:
            self._set_status(f"t={t} out of range for prob stack {prob_stack.shape}")
            return

        prob_2d = prob_stack[t, z]

        markers = None
        seed_source = self.seed_source_combo.currentText()
        if seed_source == "Active Layer":
            active = self.viewer.layers.selection.active
            if active is not None and hasattr(active, "data"):
                layer_data = np.asarray(active.data)
                if layer_data.ndim == 2:
                    markers = layer_data.astype(np.int32)
                elif layer_data.ndim >= 3:
                    markers = layer_data[0].astype(np.int32)

        params = self._single_params()
        try:
            prob_t = prob_stack[t]  # (Z, Y, X)
            basin_3d = 1.0 / (1.0 + np.exp(-prob_t))
            global_lo = float(np.min(basin_3d))
            global_hi = float(np.max(basin_3d))
            result_2d = compute_hypothesis_labels(prob_2d, None, markers, params, global_lo=global_lo, global_hi=global_hi)
        except Exception as e:
            self._set_status(f"Segmentation failed: {e}")
            return

        self._update_layer(_PREVIEW_LAYER, result_2d)
        self._set_status(f"Previewing t={t}, z={z}.")

    def _on_save_db(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        prob_path = self._prob_path()
        if prob_path is None or not prob_path.exists():
            self._set_status(f"Missing: {prob_path}")
            return

        overwrite = self.overwrite_check.isChecked()
        params = self._single_params()
        spec = NucleusHypothesisSweepSpec(
            threshold=params.threshold_pct,
            compactness=params.compactness,
            smooth_sigma=params.smooth_sigma,
            seed_source=params.seed_source,
            seed_distance=params.seed_distance,
            min_size=params.min_size,
        )
        output_path = self._pos_dir / "2_nucleus" / "hypotheses.h5"
        pos_dir = self._pos_dir

        @thread_worker(connect={"returned": self._on_save_db_done, "errored": self._on_worker_error})
        def _worker():
            prob_stack = tifffile.imread(str(prob_path))
            prob_stack = np.asarray(prob_stack, dtype=np.float32)
            records = iter_hypothesis_records_from_stacks(prob_stack, None, None, spec)
            write_hypothesis_sweep_h5(output_path, records, overwrite=overwrite, n_t=None, n_p=1)
            return pos_dir

        self._set_status("Saving to DB…")
        _worker()

    def _on_save_db_done(self, pos_dir: Path) -> None:
        self._set_status("Saved to hypotheses.h5.")
        self.refresh(pos_dir)

    def _on_preview_cellpose_native(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._set_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_status(f"Missing: {dp_path}")
            return

        t = self._current_t()
        z = self.cp_z_spin.value()
        params = CellposeFlowHypothesisParams(
            cellprob_threshold=self.cp_cellprob_spin.value(),
            flow_threshold=self.cp_flow_spin.value(),
            min_size=self.min_size_spin.value(),
            niter=int(self.cp_niter_spin.value()),
        )

        @thread_worker(connect={"returned": self._on_preview_cellpose_native_done, "errored": self._on_worker_error})
        def _worker():
            from cellflow.segmentation import compute_cellpose_flow_hypothesis
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            dp_stack = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis]
            if dp_stack.ndim == 4:
                dp_stack = dp_stack[np.newaxis]
            z_clamped = min(z, prob_stack.shape[1] - 1)
            t_clamped = min(t, prob_stack.shape[0] - 1)
            prob_slice = prob_stack[t_clamped, z_clamped:z_clamped + 1]  # (1, Y, X)
            dp_slice = dp_stack[t_clamped, z_clamped:z_clamped + 1]      # (1, 2, Y, X)
            p_min = float(prob_slice.min())
            p_max = float(prob_slice.max())
            info = f"prob=[{p_min:.2f}, {p_max:.2f}] dp_shape={dp_slice.shape}"
            labels_3d = compute_cellpose_flow_hypothesis(prob_slice, dp_slice, params)
            return labels_3d[0], t_clamped, z_clamped, info  # (Y, X), t, z, info

        self._set_status(f"Previewing Cellpose native t={t}, z={z}…")
        _worker()

    def _on_preview_cellpose_native_done(self, result: tuple) -> None:
        labels_2d, t, z, info = result
        self._update_layer(_PREVIEW_LAYER, labels_2d)
        self._set_status(f"Cellpose native t={t}, z={z}: {int(labels_2d.max())} cells. {info}")

    def _on_save_cellpose_native_db(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        prob_path = self._prob_path()
        dp_path = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._set_status(f"Missing: {prob_path}")
            return
        if dp_path is None or not dp_path.exists():
            self._set_status(f"Missing: {dp_path}")
            return

        params = CellposeFlowHypothesisParams(
            cellprob_threshold=self.cp_cellprob_spin.value(),
            flow_threshold=self.cp_flow_spin.value(),
            min_size=self.min_size_spin.value(),
            niter=int(self.cp_niter_spin.value()),
        )
        overwrite = self.overwrite_check.isChecked()
        output_path = self._pos_dir / "2_nucleus" / "hypotheses.h5"
        pos_dir = self._pos_dir

        @thread_worker(connect={"returned": self._on_save_cellpose_native_done, "errored": self._on_worker_error})
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            dp_stack = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
            records = iter_cellpose_flow_records_from_stacks(prob_stack, dp_stack, params)
            write_hypothesis_sweep_h5(output_path, records, overwrite=overwrite, n_t=None, n_p=1)
            return pos_dir

        self._set_status("Running Cellpose native segmentation…")
        _worker()

    def _on_save_cellpose_native_done(self, pos_dir: Path) -> None:
        self._set_status("Cellpose native hypothesis saved to hypotheses.h5.")
        self.refresh(pos_dir)

    def _on_db_activate_toggled(self, active: bool) -> None:
        self.db_activate_btn.setText("Deactivate" if active else "Activate")
        if active:
            self._current_db_p = None
            self._on_tree_item_selected(self.db_tree.currentItem(), None)

    def _on_tree_item_selected(self, current: QTreeWidgetItem | None, _previous) -> None:
        if not self.db_activate_btn.isChecked():
            return
        if current is None:
            return
        p = current.data(0, Qt.ItemDataRole.UserRole)
        if p is None:
            return
        if p != self._current_db_p:
            self._current_db_p = p
            self._load_db_stack(p)

    def _on_db_z_changed(self) -> None:
        if not self.db_activate_btn.isChecked():
            return
        if self._current_db_p is not None:
            self._load_db_stack(self._current_db_p)

    def _load_db_stack(self, p: int) -> None:
        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            return
        self._set_status(f"Loading p={p}…")

        @thread_worker(connect={"returned": self._on_load_stack_done, "errored": self._on_worker_error})
        def _worker():
            return p, read_full_hypothesis_stack(hyp_path, p)

        _worker()

    def _on_load_stack_done(self, result: tuple) -> None:
        p, stack = result
        if stack.ndim == 4:
            z = min(self.db_z_spin.value(), stack.shape[1] - 1)
            stack = stack[:, z]
        if _HYP_LAYER in self.viewer.layers:
            self.viewer.layers[_HYP_LAYER].data = stack
        else:
            self.viewer.add_labels(stack, name=_HYP_LAYER)
        self._set_status(f"Loaded p={p} → {stack.shape} into napari.")

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
            z = min(self.db_z_spin.value(), volume.shape[0] - 1)
            slice_2d = volume[z]  # (Y, X)
            tracked_path = self._tracked_path()
            write_tracked_frame(tracked_path, t, slice_2d)
            self._update_tracked_display(slice_2d, t=t)
            self._set_status(f"Hypothesis p={p} set as tracking seed at t={t}.")
        except Exception as e:
            self._set_status(f"Error setting seed: {e}")

    def _on_delete_slice(self) -> None:
        hyp_path = self._hyp_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_status("No hypothesis DB found.")
            return
        p = self._current_db_p
        if p is None:
            self._set_status("No parameter set selected in the DB browser.")
            return
        z = self.db_z_spin.value()
        try:
            zero_hypothesis_slice(hyp_path, z, p)
        except Exception as e:
            self._set_status(f"Delete slice failed: {e}")
            return
        self._set_status(f"Zeroed z={z} across all frames, p={p}.")
        if _HYP_LAYER in self.viewer.layers:
            self.viewer.layers[_HYP_LAYER].data[:] = 0
            self.viewer.layers[_HYP_LAYER].refresh()

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

    def _on_propagate_next(self) -> None:
        hyp_path = self._hyp_path()
        tracked_path = self._tracked_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_status("No hypothesis DB found.")
            return
        if tracked_path is None or not tracked_path.exists():
            self._set_status("No tracked labels file. Set a tracking seed first.")
            return

        t = self._current_t()
        if not tracked_frame_exists(tracked_path, t):
            self._set_status(f"No tracked frame at t={t}. Set a seed first.")
            return

        # Flush any in-memory edits to disk so propagation reads the current state.
        if _TRACKED_LAYER in self.viewer.layers:
            layer = self.viewer.layers[_TRACKED_LAYER]
            if layer.data.ndim == 3 and t < layer.data.shape[0]:
                write_tracked_frame(tracked_path, t, np.asarray(layer.data[t]))

        try:
            winner = propagate_one_frame(
                hyp_path, tracked_path, t,
                iou_threshold=self.iou_spin.value(),
                max_dist_px=self.dist_spin.value(),
                velocity_sigma_px=self.vel_sigma_spin.value(),
                iou_weight=self.iou_weight_spin.value(),
                area_weight=self.area_weight_spin.value(),
                velocity_weight=self.vel_weight_spin.value(),
            )
        except Exception as e:
            self._set_status(f"Propagation failed: {e}")
            return

        if winner is None:
            self._set_status(f"No suitable hypothesis found for t={t+1}.")
            return

        try:
            labels = read_tracked_frame(tracked_path, t + 1)
            self._update_tracked_display(labels, t=t + 1, tracked_path=tracked_path)
            # Advance viewer to next timepoint
            step = list(self.viewer.dims.current_step)
            step[0] = t + 1
            self.viewer.dims.current_step = tuple(step)
        except Exception as e:
            self._set_status(f"Could not load t={t+1}: {e}")
            return

        self._set_status(f"Propagated t={t}→{t+1} using p={winner}.")

    def _on_propagate_all(self) -> None:
        hyp_path = self._hyp_path()
        tracked_path = self._tracked_path()
        if hyp_path is None or not hyp_path.exists():
            self._set_status("No hypothesis DB found.")
            return
        if tracked_path is None or not tracked_path.exists():
            self._set_status("No tracked labels file. Set a tracking seed first.")
            return

        n_p, _ = list_hypotheses(hyp_path)
        if n_p == 0:
            self._set_status("Hypothesis DB is empty.")
            return

        t_start = self._current_t()
        iou_thr = self.iou_spin.value()
        max_dist = self.dist_spin.value()
        vel_sigma = self.vel_sigma_spin.value()
        iou_w = self.iou_weight_spin.value()
        area_w = self.area_weight_spin.value()
        vel_w = self.vel_weight_spin.value()
        self._stop_flag = False

        # Flush any in-memory edits to disk before the worker thread reads them.
        if _TRACKED_LAYER in self.viewer.layers:
            layer = self.viewer.layers[_TRACKED_LAYER]
            if layer.data.ndim == 3 and t_start < layer.data.shape[0]:
                write_tracked_frame(tracked_path, t_start, np.asarray(layer.data[t_start]))

        @thread_worker(connect={"yielded": self._on_prop_progress, "finished": self._on_prop_done, "errored": self._on_worker_error})
        def _worker():
            t = t_start
            while not self._stop_flag:
                if not tracked_frame_exists(tracked_path, t):
                    break
                winner = propagate_one_frame(
                    hyp_path, tracked_path, t, iou_thr, max_dist,
                    velocity_sigma_px=vel_sigma,
                    iou_weight=iou_w,
                    area_weight=area_w,
                    velocity_weight=vel_w,
                )
                if winner is None:
                    yield (t, None)
                    break
                yield (t, winner)
                t += 1

        self._set_status("Propagating…")
        _worker()

    def _on_prop_progress(self, result: tuple[int, int | None]) -> None:
        t, winner = result
        if winner is None:
            self._set_status(f"Propagation stopped at t={t}: no suitable hypothesis.")
        else:
            self._set_status(f"Propagated t={t}→{t+1} (p={winner})")
            try:
                tracked_path = self._tracked_path()
                labels = read_tracked_frame(tracked_path, t + 1)
                self._update_tracked_display(labels, t=t + 1, tracked_path=tracked_path)
                step = list(self.viewer.dims.current_step)
                step[0] = t + 1
                self.viewer.dims.current_step = tuple(step)
            except Exception:
                pass

    def _on_prop_done(self) -> None:
        self._set_status("Propagation complete.")

    def _on_load_tracked(self) -> None:
        tracked_path = self._tracked_path()
        if tracked_path is None or not tracked_path.exists():
            self._set_status("No tracked labels file found.")
            return

        cell_zavg_path = self._cell_zavg_path()
        nuc_zavg_path = self._nucleus_zavg_path()
        self._set_status("Loading tracked labels…")

        @thread_worker(connect={"returned": self._on_load_tracked_done, "errored": self._on_worker_error})
        def _worker():
            stack = read_full_tracked_stack(tracked_path)  # (T, Y, X)
            cell_zavg = (
                np.asarray(tifffile.imread(str(cell_zavg_path)), dtype=np.float32)
                if cell_zavg_path and cell_zavg_path.exists()
                else None
            )
            nuc_zavg = (
                np.asarray(tifffile.imread(str(nuc_zavg_path)), dtype=np.float32)
                if nuc_zavg_path and nuc_zavg_path.exists()
                else None
            )
            return stack, cell_zavg, nuc_zavg

        _worker()

    def _on_load_tracked_done(self, result: tuple) -> None:
        stack, cell_zavg, nuc_zavg = result  # stack is (T, Y, X)
        nt = stack.shape[0]
        if _TRACKED_LAYER in self.viewer.layers:
            self.viewer.layers[_TRACKED_LAYER].data = stack
        else:
            self.viewer.add_labels(stack, name=_TRACKED_LAYER)

        for zavg_data, layer_name, cmap in (
            (cell_zavg, _CELL_ZAVG_LAYER, "gray"),
            (nuc_zavg, _NUC_ZAVG_LAYER, "bop orange"),
        ):
            if zavg_data is None:
                continue
            # zavg is (Y, X) or (T, Y, X) — broadcast to (T, Y, X) to match the 2D tracked stack
            if zavg_data.ndim == 2:
                broadcast_zavg = np.broadcast_to(zavg_data[np.newaxis], (nt,) + zavg_data.shape).copy()
            else:
                broadcast_zavg = zavg_data
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = broadcast_zavg
            else:
                self.viewer.add_image(broadcast_zavg, name=layer_name, colormap=cmap, blending="additive")

        self._set_status(f"Loaded tracked stack {stack.shape} into napari.")

    def _on_run_sweep(self) -> None:
        if self._pos_dir is None:
            self._set_status("No project open.")
            return
        prob_path = self._prob_path()
        if prob_path is None or not prob_path.exists():
            self._set_status(f"Missing: {prob_path}")
            return

        seed_source = self.seed_source_combo.currentText()
        src = "auto" if seed_source == "Peak local max" else "layer"
        spec = NucleusHypothesisSweepSpec(
            threshold=self.sweep_thr[0].value(),
            threshold_min=self.sweep_thr[0].value(),
            threshold_max=self.sweep_thr[1].value(),
            threshold_step=self.sweep_thr[2].value(),
            compactness=self.sweep_cmp[0].value(),
            compactness_min=self.sweep_cmp[0].value(),
            compactness_max=self.sweep_cmp[1].value(),
            compactness_step=self.sweep_cmp[2].value(),
            smooth_sigma=self.sweep_sigma[0].value(),
            smooth_min=self.sweep_sigma[0].value(),
            smooth_max=self.sweep_sigma[1].value(),
            smooth_step=self.sweep_sigma[2].value(),
            seed_source=src,
            seed_distance=self.sweep_seed_dist[0].value(),
            seed_distance_min=self.sweep_seed_dist[0].value(),
            seed_distance_max=self.sweep_seed_dist[1].value(),
            seed_distance_step=self.sweep_seed_dist[2].value(),
            min_size=self.min_size_spin.value(),
            z_slice=self.sweep_z_slice[0].value(),
            z_slice_min=self.sweep_z_slice[0].value(),
            z_slice_max=self.sweep_z_slice[1].value(),
            z_slice_step=self.sweep_z_slice[2].value(),
        )
        overwrite = self.overwrite_check.isChecked()
        output_path = self._pos_dir / "2_nucleus" / "hypotheses.h5"
        pos_dir = self._pos_dir

        def _on_sweep_done(result):
            self._sweep_worker = None
            self._set_sweep_buttons_running(False)
            self._on_save_db_done(result)

        def _on_sweep_aborted():
            self._sweep_worker = None
            self._set_sweep_buttons_running(False)
            self._set_status("Sweep cancelled.")

        def _on_sweep_error(exc):
            self._sweep_worker = None
            self._set_sweep_buttons_running(False)
            self._on_worker_error(exc)

        @thread_worker(connect={
            "yielded": self._set_status,
            "returned": _on_sweep_done,
            "aborted": _on_sweep_aborted,
            "errored": _on_sweep_error,
        })
        def _worker():
            import json as _json
            prob_stack = tifffile.imread(str(prob_path))
            prob_stack = np.asarray(prob_stack, dtype=np.float32)

            params_list = build_parameter_sets(spec)
            if not overwrite and output_path.exists():
                try:
                    _, existing = list_hypotheses(output_path)
                    existing_jsons = {
                        attrs["parameter_json"]
                        for attrs in existing.values()
                        if "parameter_json" in attrs
                    }
                    params_list = [
                        p for p in params_list
                        if _json.dumps(p.to_dict(), sort_keys=True) not in existing_jsons
                    ]
                except Exception:
                    pass  # unreadable/empty file — proceed with full sweep

            n_full = len(build_parameter_sets(spec))
            n_skip = n_full - len(params_list)

            if not params_list:
                yield f"Sweep: all {n_full} parameter set(s) already present, nothing to do."
                return pos_dir

            if n_skip:
                yield f"Sweep: skipping {n_skip} existing, computing {len(params_list)} new…"

            n_t = prob_stack.shape[0] if prob_stack.ndim == 4 else 1
            total = n_t * len(params_list)
            collected: list[HypothesisRecord] = []
            for done, record in enumerate(
                iter_hypothesis_records_from_stacks(prob_stack, None, None, spec, params_list=params_list), 1
            ):
                collected.append(record)
                yield f"Sweep {done}/{total}…"
            write_hypothesis_sweep_h5(output_path, iter(collected), overwrite=overwrite)
            return pos_dir

        self._set_status("Running sweep…")
        self._set_sweep_buttons_running(True)
        self._sweep_worker = _worker()

    def _set_sweep_buttons_running(self, running: bool) -> None:
        self.run_sweep_btn.setEnabled(not running)
        self.run_terminal_btn.setEnabled(not running)
        self.cancel_sweep_btn.setEnabled(running)

    def _on_cancel_sweep(self) -> None:
        if self._sweep_worker is not None:
            self._sweep_worker.quit()

    def _on_run_terminal(self) -> None:
        import sys
        import tempfile

        if self._pos_dir is None:
            self._set_status("No project open.")
            return

        prob_path = self._prob_path()
        output_path = self._pos_dir / "2_nucleus" / "hypotheses.h5"
        seed_source = self.seed_source_combo.currentText()
        src = "auto" if seed_source == "Peak local max" else "layer"

        overwrite_flag = self.overwrite_check.isChecked()
        python_code = (
            "from cellflow.database.hypotheses import (\n"
            "    NucleusHypothesisSweepSpec, iter_hypothesis_records_from_stacks,\n"
            "    build_parameter_sets, list_hypotheses, write_hypothesis_sweep_h5)\n"
            "import json, pathlib, tifffile, numpy as np\n"
            f"prob = tifffile.imread({str(prob_path)!r}).astype('float32')\n"
            f"output_path = pathlib.Path({str(output_path)!r})\n"
            f"overwrite = {overwrite_flag!r}\n"
            f"spec = NucleusHypothesisSweepSpec(\n"
            f"    threshold={self.sweep_thr[0].value()},\n"
            f"    threshold_min={self.sweep_thr[0].value()}, threshold_max={self.sweep_thr[1].value()},\n"
            f"    threshold_step={self.sweep_thr[2].value()},\n"
            f"    compactness={self.sweep_cmp[0].value()},\n"
            f"    compactness_min={self.sweep_cmp[0].value()}, compactness_max={self.sweep_cmp[1].value()},\n"
            f"    compactness_step={self.sweep_cmp[2].value()},\n"
            f"    smooth_sigma={self.sweep_sigma[0].value()},\n"
            f"    smooth_min={self.sweep_sigma[0].value()}, smooth_max={self.sweep_sigma[1].value()},\n"
            f"    smooth_step={self.sweep_sigma[2].value()},\n"
            f"    seed_source={src!r},\n"
            f"    seed_distance={self.sweep_seed_dist[0].value()},\n"
            f"    seed_distance_min={self.sweep_seed_dist[0].value()}, seed_distance_max={self.sweep_seed_dist[1].value()},\n"
            f"    seed_distance_step={self.sweep_seed_dist[2].value()},\n"
            f"    min_size={self.min_size_spin.value()},\n"
            f"    z_slice={self.sweep_z_slice[0].value()},\n"
            f"    z_slice_min={self.sweep_z_slice[0].value()}, z_slice_max={self.sweep_z_slice[1].value()},\n"
            f"    z_slice_step={self.sweep_z_slice[2].value()},\n"
            ")\n"
            "params_list = build_parameter_sets(spec)\n"
            "if not overwrite and output_path.exists():\n"
            "    try:\n"
            "        _, existing = list_hypotheses(output_path)\n"
            "        existing_jsons = {attrs['parameter_json'] for attrs in existing.values() if 'parameter_json' in attrs}\n"
            "        params_list = [p for p in params_list if json.dumps(p.to_dict(), sort_keys=True) not in existing_jsons]\n"
            "    except Exception:\n"
            "        pass\n"
            "n_full = len(build_parameter_sets(spec))\n"
            "n_skip = n_full - len(params_list)\n"
            "if not params_list:\n"
            "    print(f'Sweep: all {n_full} parameter set(s) already present, nothing to do.')\n"
            "else:\n"
            "    if n_skip:\n"
            "        print(f'Sweep: skipping {n_skip} existing, computing {len(params_list)} new…', flush=True)\n"
            "    n_t = prob.shape[0] if prob.ndim == 4 else 1\n"
            "    total = n_t * len(params_list)\n"
            "    records = []\n"
            "    for done, rec in enumerate(iter_hypothesis_records_from_stacks(prob, None, None, spec, params_list=params_list), 1):\n"
            "        records.append(rec)\n"
            "        print(f'Sweep {done}/{total}…', flush=True)\n"
            "    write_hypothesis_sweep_h5(str(output_path), iter(records), overwrite=overwrite)\n"
            "    print('Done.')\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="cellflow_sweep_", delete=False
        ) as tmp:
            tmp.write(python_code)
            tmp_path = tmp.name

        python_exe = sys.executable
        cmd = f"{shlex.quote(python_exe)} {shlex.quote(tmp_path)}"
        try:
            from cellflow.napari.utils import launch_in_terminal
            launch_in_terminal(cmd)
            self._set_status("Command launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(cmd)
            self._set_status("Copied command to clipboard (terminal launch unavailable).")

    def _on_dims_step_changed(self, event=None) -> None:
        """Refresh the Validate button state when the viewed frame changes."""
        self._refresh_validate_btn()

    def _refresh_validate_btn(self) -> None:
        """Sync the Validate button checked state to disk without emitting signals."""
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
        tracked_path = self._tracked_path()
        if tracked_path is None or not tracked_path.exists():
            self._set_status("No tracked labels file found.")
            return

        t = self._current_t()

        if is_validated(self._pos_dir, t):
            self._set_status(f"Frame t={t} is validated — unvalidate it first to retrack.")
            return

        if not tracked_frame_exists(tracked_path, t):
            self._set_status(f"No tracked frame at t={t}.")
            return

        # Find the nearest preceding validated frame; fall back to t-1.
        validated = sorted(
            [v for v in read_validated_frames(self._pos_dir) if v < t],
            reverse=True,
        )
        t_ref = validated[0] if validated else (t - 1)

        if t_ref < 0:
            self._set_status("No reference frame available (t=0 has no predecessor).")
            return

        if not tracked_frame_exists(tracked_path, t_ref):
            self._set_status(f"Reference frame t={t_ref} does not exist yet.")
            return

        # Flush in-memory edits for both frames before reading from disk.
        if _TRACKED_LAYER in self.viewer.layers:
            layer = self.viewer.layers[_TRACKED_LAYER]
            if layer.data.ndim == 3:
                for flush_t in (t, t_ref):
                    if flush_t < layer.data.shape[0]:
                        write_tracked_frame(tracked_path, flush_t, np.asarray(layer.data[flush_t]))

        try:
            ref_labels = read_tracked_frame(tracked_path, t_ref)
            tgt_labels = read_tracked_frame(tracked_path, t)
        except Exception as e:
            self._set_status(f"Could not read frames: {e}")
            return

        remapped = retrack_frame(ref_labels, tgt_labels, max_dist_px=self.dist_spin.value())

        tgt_ids = set(int(i) for i in np.unique(tgt_labels) if i != 0)
        new_ids = set(int(i) for i in np.unique(remapped) if i != 0)
        ref_ids = set(int(i) for i in np.unique(ref_labels) if i != 0)
        n_matched = len(new_ids & ref_ids)
        n_new = len(new_ids - ref_ids)

        write_tracked_frame(tracked_path, t, remapped)
        self._update_tracked_display(remapped, t=t, tracked_path=tracked_path)
        self._set_status(
            f"Retracked t={t} using t={t_ref} as reference: "
            f"{n_matched} matched, {n_new} new ID(s) assigned."
        )

    def _on_jump_correction(self) -> None:
        self._set_status("Manual correction widget not yet connected.")

    def _on_worker_error(self, exc: Exception) -> None:
        self._set_status(f"Error: {exc}")
        logger.exception("Worker error", exc_info=exc)

    # ──────────────────────────────────────────────────────────────────────────
    # State persistence
    # ──────────────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Return the current UI state as a dictionary."""
        return {
            "seed_source": self.seed_source_combo.currentText(),
            "overwrite": self.overwrite_check.isChecked(),
            "min_size": self.min_size_spin.value(),
            "z_slice": self.z_slice_spin.value(),
            "single": {
                "threshold": self.single_thr.value(),
                "compactness": self.single_cmp.value(),
                "sigma": self.single_sigma.value(),
                "seed_dist": self.single_seed_dist.value(),
            },
            "sweep": {
                "thr_min": self.sweep_thr[0].value(),
                "thr_max": self.sweep_thr[1].value(),
                "thr_step": self.sweep_thr[2].value(),
                "cmp_min": self.sweep_cmp[0].value(),
                "cmp_max": self.sweep_cmp[1].value(),
                "cmp_step": self.sweep_cmp[2].value(),
                "sigma_min": self.sweep_sigma[0].value(),
                "sigma_max": self.sweep_sigma[1].value(),
                "sigma_step": self.sweep_sigma[2].value(),
                "seed_dist_min": self.sweep_seed_dist[0].value(),
                "seed_dist_max": self.sweep_seed_dist[1].value(),
                "seed_dist_step": self.sweep_seed_dist[2].value(),
                "z_slice_min": self.sweep_z_slice[0].value(),
                "z_slice_max": self.sweep_z_slice[1].value(),
                "z_slice_step": self.sweep_z_slice[2].value(),
            },
            "db_browser": {
                "z_slice": self.db_z_spin.value(),
            },
            "search": {
                "iou_threshold": self.iou_spin.value(),
                "max_dist_um": self.dist_spin.value(),
                "velocity_sigma_px": self.vel_sigma_spin.value(),
                "iou_weight": self.iou_weight_spin.value(),
                "area_weight": self.area_weight_spin.value(),
                "velocity_weight": self.vel_weight_spin.value(),
            },
        }

    def set_state(self, state: dict) -> None:
        """Update the UI state from a dictionary."""
        if "seed_source" in state:
            self.seed_source_combo.setCurrentText(state["seed_source"])
        if "overwrite" in state:
            self.overwrite_check.setChecked(state["overwrite"])
        if "min_size" in state:
            self.min_size_spin.setValue(state["min_size"])
        if "z_slice" in state:
            self.z_slice_spin.setValue(state["z_slice"])

        if "single" in state:
            s = state["single"]
            if "threshold" in s: self.single_thr.setValue(s["threshold"])
            if "compactness" in s: self.single_cmp.setValue(s["compactness"])
            if "sigma" in s: self.single_sigma.setValue(s["sigma"])
            if "seed_dist" in s: self.single_seed_dist.setValue(s["seed_dist"])

        if "sweep" in state:
            sw = state["sweep"]
            if "thr_min" in sw: self.sweep_thr[0].setValue(sw["thr_min"])
            if "thr_max" in sw: self.sweep_thr[1].setValue(sw["thr_max"])
            if "thr_step" in sw: self.sweep_thr[2].setValue(sw["thr_step"])
            if "cmp_min" in sw: self.sweep_cmp[0].setValue(sw["cmp_min"])
            if "cmp_max" in sw: self.sweep_cmp[1].setValue(sw["cmp_max"])
            if "cmp_step" in sw: self.sweep_cmp[2].setValue(sw["cmp_step"])
            if "sigma_min" in sw: self.sweep_sigma[0].setValue(sw["sigma_min"])
            if "sigma_max" in sw: self.sweep_sigma[1].setValue(sw["sigma_max"])
            if "sigma_step" in sw: self.sweep_sigma[2].setValue(sw["sigma_step"])
            if "seed_dist_min" in sw: self.sweep_seed_dist[0].setValue(sw["seed_dist_min"])
            if "seed_dist_max" in sw: self.sweep_seed_dist[1].setValue(sw["seed_dist_max"])
            if "seed_dist_step" in sw: self.sweep_seed_dist[2].setValue(sw["seed_dist_step"])
            if "z_slice_min" in sw: self.sweep_z_slice[0].setValue(sw["z_slice_min"])
            if "z_slice_max" in sw: self.sweep_z_slice[1].setValue(sw["z_slice_max"])
            if "z_slice_step" in sw: self.sweep_z_slice[2].setValue(sw["z_slice_step"])

        if "db_browser" in state:
            db = state["db_browser"]
            if "z_slice" in db:
                self.db_z_spin.setValue(db["z_slice"])

        if "search" in state:
            se = state["search"]
            if "iou_threshold" in se: self.iou_spin.setValue(se["iou_threshold"])
            if "max_dist_um" in se: self.dist_spin.setValue(se["max_dist_um"])
            if "velocity_sigma_px" in se: self.vel_sigma_spin.setValue(se["velocity_sigma_px"])
            if "iou_weight" in se: self.iou_weight_spin.setValue(se["iou_weight"])
            if "area_weight" in se: self.area_weight_spin.setValue(se["area_weight"])
            if "velocity_weight" in se: self.vel_weight_spin.setValue(se["velocity_weight"])
