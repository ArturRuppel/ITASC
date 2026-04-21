"""Nucleus-anchored cell segmentation widget (step 4 — 4_cell_segmentation).

Uses Euler integration along a blend of the Cellpose flow field and an N-body
gravitational field computed from nuclear centroids (Gravity-Flow algorithm).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import tifffile
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from napari.qt.threading import thread_worker

from cellflow.core.paths import stage_dir
from cellflow.napari.runners.terminal import launch_in_terminal
from cellflow.napari.log_viewer import StageLogViewer
from cellflow.napari.registry import get_state
from cellflow.napari.widgets import PipelineFilesWidget


def cellpose_cell_dir(root_dir, pos):
    return stage_dir(root_dir, pos, "cellpose_cell")


def cell_segmentation_dir(root_dir, pos):
    return stage_dir(root_dir, pos, "cell_segmentation")


class CellSegmentationConfig:
    """Configuration for Euler (Gravity-Flow) cell segmentation."""

    def __init__(
        self,
        cellpose_prob_threshold: float = 0.0,
        flow_smoothing_sigma: float = 0.0,
        flow_step_scale: float = 0.2,
        euler_max_steps: int = 100,
        capture_radius: float = 3.0,
        flow_weight: float = 0.5,
        gravity_falloff: float = 2.0,
    ):
        self.cellpose_prob_threshold = cellpose_prob_threshold
        self.flow_smoothing_sigma = flow_smoothing_sigma
        self.flow_step_scale = flow_step_scale
        self.euler_max_steps = euler_max_steps
        self.capture_radius = capture_radius
        self.flow_weight = flow_weight
        self.gravity_falloff = gravity_falloff

    def model_dump(self) -> dict:
        return {
            "cellpose_prob_threshold": self.cellpose_prob_threshold,
            "flow_smoothing_sigma":    self.flow_smoothing_sigma,
            "flow_step_scale":         self.flow_step_scale,
            "euler_max_steps":         self.euler_max_steps,
            "capture_radius":          self.capture_radius,
            "flow_weight":          self.flow_weight,
            "gravity_falloff":         self.gravity_falloff,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CellSegmentationConfig":
        data = dict(data)
        for k in ("method", "algorithm", "flow_scale", "max_iterations", "uniform_growth_rate",
                  "foreground_mask_sigma", "foreground_mask_threshold",
                  "foreground_mask_postprocess_steps", "postprocess_steps",
                  "opening_radius", "closing_radius", "boundary_smoothness", "fill_holes_threshold"):
            data.pop(k, None)
        valid = {k for k in cls.__init__.__code__.co_varnames if k != "self"}
        data = {k: v for k, v in data.items() if k in valid}
        return cls(**data)


class WatershedConfig:
    """Configuration for nucleus-seeded watershed cell segmentation."""

    def __init__(
        self,
        cellpose_prob_threshold: float = 0.0,
        compactness: float = 0.0,
        prob_smoothing_sigma: float = 0.0,
        basin: str = "prob",
    ):
        self.cellpose_prob_threshold = cellpose_prob_threshold
        self.compactness = compactness
        self.prob_smoothing_sigma = prob_smoothing_sigma
        self.basin = basin

    def model_dump(self) -> dict:
        return {
            "cellpose_prob_threshold": self.cellpose_prob_threshold,
            "compactness": self.compactness,
            "prob_smoothing_sigma": self.prob_smoothing_sigma,
            "basin": self.basin,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WatershedConfig":
        valid = {"cellpose_prob_threshold", "compactness", "prob_smoothing_sigma", "basin"}
        return cls(**{k: v for k, v in data.items() if k in valid})


# ── Module-level data helpers ─────────────────────────────────────────────────

def _load_nuclear_labels(root_dir: Path | str, pos: int) -> np.ndarray | None:
    try:
        path = stage_dir(root_dir, pos, "correction") / "nuclear_labels_corrected.tif"
        if path.exists():
            return tifffile.imread(str(path)).astype(np.int32)
    except Exception:
        pass
    return None


def _load_cellpose_data(root_dir: Path | str, pos: int, t: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        cell_dir = cellpose_cell_dir(root_dir, pos)
        flow_path = cell_dir / "cell_dp.tif"
        flow = None
        if flow_path.exists():
            flow = tifffile.imread(str(flow_path)).astype(np.float32)
            if flow.ndim == 4:
                flow = np.transpose(flow, (0, 2, 3, 1))
                flow = flow[t]
            elif flow.ndim == 3:
                flow = np.transpose(flow, (1, 2, 0))
            else:
                flow = None

        prob_path = cell_dir / "cell_prob.tif"
        prob = None
        if prob_path.exists():
            prob = tifffile.imread(str(prob_path)).astype(np.float32)
            if prob.ndim == 3:
                prob = prob[t]

        return flow, prob
    except Exception:
        return None, None


# ── Module-level run helpers ──────────────────────────────────────────────────

def run_segmentation(
    root_dir: str | Path,
    pos: int,
    config: CellSegmentationConfig,
    overwrite: bool = True,
) -> Generator:
    """Run Euler (Gravity-Flow) cell segmentation for a full stack."""
    from cellflow.cellpose.processing.gravity_flow import gravity_flow_segmentation

    root_dir = Path(root_dir)
    out_dir = cell_segmentation_dir(root_dir, pos)
    out_path = out_dir / "cell_labels.tif"
    if out_path.exists() and not overwrite:
        return str(out_path)

    nuclear_labels = _load_nuclear_labels(root_dir, pos)
    if nuclear_labels is None or nuclear_labels.ndim != 3:
        print("Could not load nuclear labels (T, H, W)")
        return None

    T = nuclear_labels.shape[0]

    try:
        cell_dir = cellpose_cell_dir(root_dir, pos)
        flow_stack = tifffile.imread(str(cell_dir / "cell_dp.tif")).astype(np.float32)
        if flow_stack.ndim == 4:
            flow_stack = np.transpose(flow_stack, (0, 2, 3, 1))  # (T, H, W, 2)
        else:
            flow_stack = np.transpose(flow_stack, (1, 2, 0))[np.newaxis]

        prob_path = cell_dir / "cell_prob.tif"
        prob_stack = tifffile.imread(str(prob_path)).astype(np.float32) if prob_path.exists() else None
        if prob_stack is not None and prob_stack.ndim == 2:
            prob_stack = prob_stack[np.newaxis]
    except Exception as e:
        print(f"Error loading flow stack: {e}")
        return None

    cell_labels_stack = []

    for t in range(T):
        try:
            nuc_t = nuclear_labels[t]
            flow_t = flow_stack[t] if flow_stack.shape[0] > 1 else flow_stack[0]
            prob_t = prob_stack[t] if (prob_stack is not None and prob_stack.shape[0] > 1) else (prob_stack[0] if prob_stack is not None else None)

            cell_labels = gravity_flow_segmentation(
                nuc_t,
                flow_t,
                cellpose_prob=prob_t,
                flow_step_scale=config.flow_step_scale,
                cellpose_prob_threshold=config.cellpose_prob_threshold,
                flow_smoothing_sigma=config.flow_smoothing_sigma,
                max_iterations=config.euler_max_steps,
                capture_radius=config.capture_radius,
                flow_weight=config.flow_weight,
                gravity_falloff=config.gravity_falloff,
            )
            cell_labels_stack.append(cell_labels)
        except Exception as e:
            print(f"Error at t{t:03d}: {e}")
            cell_labels_stack.append(np.zeros_like(nuclear_labels[t], dtype=np.int32))

        yield (t + 1, T, f"t{t:03d}")

    stack = np.stack(cell_labels_stack, axis=0).astype(np.int32)
    out_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), stack, compression="zlib", metadata={"axes": "TYX"})
    return str(out_path)


def run_watershed_segmentation(
    root_dir: str | Path,
    pos: int,
    config: WatershedConfig,
    overwrite: bool = True,
) -> Generator:
    """Run nucleus-seeded watershed cell segmentation for a full stack."""
    from cellflow.cellpose.processing.gravity_flow import prob_watershed_segmentation

    root_dir = Path(root_dir)
    out_dir = cell_segmentation_dir(root_dir, pos)
    out_path = out_dir / "cell_labels.tif"
    if out_path.exists() and not overwrite:
        return str(out_path)

    nuclear_labels = _load_nuclear_labels(root_dir, pos)
    if nuclear_labels is None or nuclear_labels.ndim != 3:
        print("Could not load nuclear labels (T, H, W)")
        return None

    T = nuclear_labels.shape[0]

    try:
        cell_dir = cellpose_cell_dir(root_dir, pos)
        prob_path = cell_dir / "cell_prob.tif"
        if not prob_path.exists():
            print("cell_prob.tif not found")
            return None
        prob_stack = tifffile.imread(str(prob_path)).astype(np.float32)
        if prob_stack.ndim == 2:
            prob_stack = prob_stack[np.newaxis]
    except Exception as e:
        print(f"Error loading prob stack: {e}")
        return None

    flow_stack = None
    if config.basin == "flow_mag":
        try:
            flow_path = cellpose_cell_dir(root_dir, pos) / "cell_dp.tif"
            if not flow_path.exists():
                print("cell_dp.tif not found — required for flow_mag basin")
                return None
            fs = tifffile.imread(str(flow_path)).astype(np.float32)
            flow_stack = np.transpose(fs, (0, 2, 3, 1)) if fs.ndim == 4 else np.transpose(fs, (1, 2, 0))[np.newaxis]
        except Exception as e:
            print(f"Error loading flow stack: {e}")
            return None

    cell_labels_stack = []
    for t in range(T):
        try:
            nuc_t = nuclear_labels[t]
            prob_t = prob_stack[t] if prob_stack.shape[0] > 1 else prob_stack[0]
            flow_t = None
            if flow_stack is not None:
                flow_t = flow_stack[t] if flow_stack.shape[0] > 1 else flow_stack[0]
            cell_labels = prob_watershed_segmentation(
                nuc_t, prob_t,
                cellpose_prob_threshold=config.cellpose_prob_threshold,
                compactness=config.compactness,
                prob_smoothing_sigma=config.prob_smoothing_sigma,
                flow_field=flow_t,
                basin=config.basin,
            )
            cell_labels_stack.append(cell_labels)
        except Exception as e:
            print(f"Error at t{t:03d}: {e}")
            cell_labels_stack.append(np.zeros_like(nuclear_labels[t], dtype=np.int32))
        yield (t + 1, T, f"t{t:03d}")

    stack = np.stack(cell_labels_stack, axis=0).astype(np.int32)
    out_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), stack, compression="zlib", metadata={"axes": "TYX"})
    return str(out_path)


# ── Widget ────────────────────────────────────────────────────────────────────

class CellSegmentationWidget(QWidget):
    """Widget for Euler (Gravity-Flow) nucleus-anchored cell segmentation."""

    run_started = Signal()

    def __init__(self, viewer: "napari.Viewer", *, log_viewer=None) -> None:
        super().__init__()
        self.viewer = viewer
        self._state = get_state(viewer)
        self._seg_worker = None

        self._inner_layout = QVBoxLayout(self)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setAlignment(Qt.AlignTop)

        lay = self._inner_layout

        # ── Files ─────────────────────────────────────────────────────────
        self._files_widget = PipelineFilesWidget([
            ("Input", [
                ("3_correction/nuclear_labels_corrected.tif", "Corrected labels"),
            ]),
            ("Output", [
                ("4_cell_segmentation/cell_labels.tif", "Cell labels"),
            ]),
        ])
        lay.addWidget(self._files_widget)

        # ── Method tabs ────────────────────────────────────────────────────
        self._method_tabs = QTabWidget()
        self._method_tabs.addTab(self._build_gravity_tab(), "Gravity Flow")
        self._method_tabs.addTab(self._build_watershed_tab(), "Watershed")
        lay.addWidget(self._method_tabs)

        # ── Preview ────────────────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview frame"))
        self._seg_frame_spin = QSpinBox()
        self._seg_frame_spin.setRange(0, 1000)
        self._seg_frame_spin.setValue(0)
        row.addWidget(self._seg_frame_spin)
        lay.addLayout(row)

        self._seg_preview_btn = QPushButton("Preview")
        self._seg_preview_btn.clicked.connect(self._seg_on_preview)
        lay.addWidget(self._seg_preview_btn)

        # ── Run ────────────────────────────────────────────────────────────
        self._seg_overwrite_chk = QCheckBox("Overwrite existing files")
        self._seg_overwrite_chk.setStyleSheet("color: white;")
        lay.addWidget(self._seg_overwrite_chk)

        row = QHBoxLayout()
        self._seg_run_btn = QPushButton("Run Segmentation")
        self._seg_run_btn.clicked.connect(self._seg_on_run)
        row.addWidget(self._seg_run_btn)
        self._seg_term_btn = QPushButton("Run in Terminal")
        self._seg_term_btn.clicked.connect(self._seg_on_run_terminal)
        row.addWidget(self._seg_term_btn)
        self._seg_cancel_btn = QPushButton("Cancel")
        self._seg_cancel_btn.setEnabled(False)
        self._seg_cancel_btn.clicked.connect(self._seg_on_cancel)
        row.addWidget(self._seg_cancel_btn)
        lay.addLayout(row)

        self._seg_load_btn = QPushButton("Load Results")
        self._seg_load_btn.clicked.connect(self._seg_on_load_results)
        lay.addWidget(self._seg_load_btn)

        self._seg_progress = QProgressBar()
        self._seg_progress.setVisible(False)
        lay.addWidget(self._seg_progress)
        self._seg_status = QLabel("")
        lay.addWidget(self._seg_status)

        # ── Save Corrected Cell Labels ─────────────────────────────────────
        self._save_labels_btn = QPushButton("Save Corrected Cell Labels")
        self._save_labels_btn.clicked.connect(self._on_save_corrected_labels)
        lay.addWidget(self._save_labels_btn)
        self._save_labels_status = QLabel("")
        lay.addWidget(self._save_labels_status)

        if log_viewer is not None:
            self._log_viewer = log_viewer
        else:
            self._log_viewer = StageLogViewer(self._state)
            lay.addWidget(self._log_viewer)

        self._state.pipeline_schema_changed.connect(self._sync_project_dir)
        self._state.position_changed.connect(self._sync_project_dir)
        self._sync_project_dir()

    # ── Tab builders ──────────────────────────────────────────────────────

    def _build_gravity_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(4, 4, 4, 4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Cellprob threshold"))
        self._seg_prob_threshold_spin = QDoubleSpinBox()
        self._seg_prob_threshold_spin.setRange(-100.0, 100.0)
        self._seg_prob_threshold_spin.setSingleStep(1.0)
        self._seg_prob_threshold_spin.setDecimals(1)
        self._seg_prob_threshold_spin.setValue(0.0)
        row.addWidget(self._seg_prob_threshold_spin)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Flow step scale"))
        self._euler_flow_step_spin = QDoubleSpinBox()
        self._euler_flow_step_spin.setRange(0.01, 2.0)
        self._euler_flow_step_spin.setSingleStep(0.05)
        self._euler_flow_step_spin.setDecimals(2)
        self._euler_flow_step_spin.setValue(0.2)
        row.addWidget(self._euler_flow_step_spin)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Max steps"))
        self._euler_max_steps_spin = QSpinBox()
        self._euler_max_steps_spin.setRange(1, 2000)
        self._euler_max_steps_spin.setSingleStep(10)
        self._euler_max_steps_spin.setValue(100)
        row.addWidget(self._euler_max_steps_spin)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Capture radius (px)"))
        self._euler_capture_radius_spin = QDoubleSpinBox()
        self._euler_capture_radius_spin.setRange(0.5, 20.0)
        self._euler_capture_radius_spin.setSingleStep(0.5)
        self._euler_capture_radius_spin.setDecimals(1)
        self._euler_capture_radius_spin.setValue(3.0)
        row.addWidget(self._euler_capture_radius_spin)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Flow weight"))
        self._euler_flow_weight_spin = QDoubleSpinBox()
        self._euler_flow_weight_spin.setRange(0.0, 1.0)
        self._euler_flow_weight_spin.setSingleStep(0.05)
        self._euler_flow_weight_spin.setDecimals(2)
        self._euler_flow_weight_spin.setValue(0.5)
        row.addWidget(self._euler_flow_weight_spin)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Gravity falloff"))
        self._euler_gravity_falloff_spin = QDoubleSpinBox()
        self._euler_gravity_falloff_spin.setRange(0.5, 5.0)
        self._euler_gravity_falloff_spin.setSingleStep(0.5)
        self._euler_gravity_falloff_spin.setDecimals(1)
        self._euler_gravity_falloff_spin.setValue(2.0)
        row.addWidget(self._euler_gravity_falloff_spin)
        row.addStretch()
        lay.addLayout(row)

        lay.addStretch()
        return tab

    def _build_watershed_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(4, 4, 4, 4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Basin image"))
        self._ws_basin_combo = QComboBox()
        self._ws_basin_combo.addItems(["Probability", "Flow magnitude"])
        row.addWidget(self._ws_basin_combo)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Cellprob threshold"))
        self._ws_prob_threshold_spin = QDoubleSpinBox()
        self._ws_prob_threshold_spin.setRange(-100.0, 100.0)
        self._ws_prob_threshold_spin.setSingleStep(1.0)
        self._ws_prob_threshold_spin.setDecimals(1)
        self._ws_prob_threshold_spin.setValue(0.0)
        row.addWidget(self._ws_prob_threshold_spin)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Compactness"))
        self._ws_compactness_spin = QDoubleSpinBox()
        self._ws_compactness_spin.setRange(0.0, 10.0)
        self._ws_compactness_spin.setSingleStep(0.01)
        self._ws_compactness_spin.setDecimals(3)
        self._ws_compactness_spin.setValue(0.0)
        row.addWidget(self._ws_compactness_spin)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Prob smoothing sigma"))
        self._ws_prob_smoothing_spin = QDoubleSpinBox()
        self._ws_prob_smoothing_spin.setRange(0.0, 10.0)
        self._ws_prob_smoothing_spin.setSingleStep(0.5)
        self._ws_prob_smoothing_spin.setDecimals(1)
        self._ws_prob_smoothing_spin.setValue(0.0)
        row.addWidget(self._ws_prob_smoothing_spin)
        row.addStretch()
        lay.addLayout(row)

        lay.addStretch()
        return tab

    # ── Path helpers ──────────────────────────────────────────────────────

    def _get_root_dir(self) -> str | None:
        project_dir = self._state.project_dir
        if project_dir is None:
            return None
        return str(project_dir)

    def _sync_project_dir(self) -> None:
        root = self._get_root_dir()
        if root is None:
            self._files_widget.refresh(None)
            return
        pos = self._state.current_position
        self._files_widget.refresh(Path(root) / f"pos{pos:02d}")

    # ── Config helpers ────────────────────────────────────────────────────

    def _active_method(self) -> str:
        return "watershed" if self._method_tabs.currentIndex() == 1 else "gravity_flow"

    def _build_gravity_config(self) -> CellSegmentationConfig:
        return CellSegmentationConfig(
            cellpose_prob_threshold=self._seg_prob_threshold_spin.value(),
            flow_step_scale=self._euler_flow_step_spin.value(),
            euler_max_steps=self._euler_max_steps_spin.value(),
            capture_radius=self._euler_capture_radius_spin.value(),
            flow_weight=self._euler_flow_weight_spin.value(),
            gravity_falloff=self._euler_gravity_falloff_spin.value(),
        )

    def _apply_gravity_config(self, cfg: CellSegmentationConfig) -> None:
        self._seg_prob_threshold_spin.setValue(cfg.cellpose_prob_threshold)
        self._euler_flow_step_spin.setValue(cfg.flow_step_scale)
        self._euler_max_steps_spin.setValue(cfg.euler_max_steps)
        self._euler_capture_radius_spin.setValue(cfg.capture_radius)
        self._euler_flow_weight_spin.setValue(cfg.flow_weight)
        self._euler_gravity_falloff_spin.setValue(cfg.gravity_falloff)

    def _build_watershed_config(self) -> WatershedConfig:
        return WatershedConfig(
            cellpose_prob_threshold=self._ws_prob_threshold_spin.value(),
            compactness=self._ws_compactness_spin.value(),
            prob_smoothing_sigma=self._ws_prob_smoothing_spin.value(),
            basin="flow_mag" if self._ws_basin_combo.currentIndex() == 1 else "prob",
        )

    def _apply_watershed_config(self, cfg: WatershedConfig) -> None:
        self._ws_prob_threshold_spin.setValue(cfg.cellpose_prob_threshold)
        self._ws_compactness_spin.setValue(cfg.compactness)
        self._ws_prob_smoothing_spin.setValue(cfg.prob_smoothing_sigma)
        self._ws_basin_combo.setCurrentIndex(1 if cfg.basin == "flow_mag" else 0)

    # ── Save Corrected Cell Labels ─────────────────────────────────────────

    def _on_save_corrected_labels(self) -> None:
        root_dir = self._get_root_dir()
        if not root_dir:
            self._save_labels_status.setText("No project open.")
            return

        active = self.viewer.layers.selection.active
        if active is None or not hasattr(active, "data"):
            self._save_labels_status.setText("No active layer selected.")
            return

        pos = int(self._state.current_position)
        data = np.asarray(active.data).astype(np.int32)

        out_dir = cell_segmentation_dir(root_dir, pos)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cell_labels.tif"

        try:
            axes = "TYX" if data.ndim == 3 else "YX"
            tifffile.imwrite(str(out_path), data, compression="zlib", metadata={"axes": axes})
            self._save_labels_status.setText(f"Saved cell labels to {out_path.name}")
            self._sync_project_dir()
        except Exception as e:
            self._save_labels_status.setText(f"Error: {e}")

    # ── Preview ────────────────────────────────────────────────────────────

    def _seg_on_preview(self) -> None:
        root_dir = self._get_root_dir()
        if not root_dir:
            self._seg_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._state.current_position)
        frame = int(self._seg_frame_spin.value())
        method = self._active_method()

        self._seg_status.setText(f"Processing frame {frame}…")

        try:
            root_dir_path = Path(root_dir)

            nuclear_labels = _load_nuclear_labels(root_dir_path, pos)
            if nuclear_labels is None:
                self._seg_status.setText("Could not load nuclear labels.")
                return

            nuc_t = nuclear_labels[frame]

            if method == "gravity_flow":
                cfg = self._build_gravity_config()
                flow_t, prob_t = _load_cellpose_data(root_dir_path, pos, frame)
                if flow_t is None:
                    self._seg_status.setText("Could not load cellpose flow.")
                    return
                from cellflow.cellpose.processing.gravity_flow import gravity_flow_segmentation
                cell_labels = gravity_flow_segmentation(
                    nuc_t,
                    flow_t,
                    cellpose_prob=prob_t,
                    flow_step_scale=cfg.flow_step_scale,
                    cellpose_prob_threshold=cfg.cellpose_prob_threshold,
                    max_iterations=cfg.euler_max_steps,
                    capture_radius=cfg.capture_radius,
                    flow_weight=cfg.flow_weight,
                    gravity_falloff=cfg.gravity_falloff,
                )
            else:
                cfg = self._build_watershed_config()
                flow_t, prob_t = _load_cellpose_data(root_dir_path, pos, frame)
                if prob_t is None:
                    self._seg_status.setText("Could not load cellpose probability map.")
                    return
                if cfg.basin == "flow_mag" and flow_t is None:
                    self._seg_status.setText("Could not load cellpose flow (needed for flow_mag basin).")
                    return
                from cellflow.cellpose.processing.gravity_flow import prob_watershed_segmentation
                cell_labels = prob_watershed_segmentation(
                    nuc_t, prob_t,
                    cellpose_prob_threshold=cfg.cellpose_prob_threshold,
                    compactness=cfg.compactness,
                    prob_smoothing_sigma=cfg.prob_smoothing_sigma,
                    flow_field=flow_t,
                    basin=cfg.basin,
                )

            # --- Load background images ---
            raw_dir = stage_dir(root_dir_path, pos, "raw_import")
            cell_img_path = raw_dir / "cell" / "cell_zavg.tif"
            nuc_img_path  = raw_dir / "nucleus" / "nucleus_zavg.tif"

            cell_layer_name   = "Cell avg"
            nuc_layer_name    = "Nucleus avg"
            labels_layer_name = "Preview: Cell Segmentation"

            if cell_img_path.exists():
                cell_img = tifffile.imread(str(cell_img_path))
                cell_img_t = cell_img[frame] if cell_img.ndim == 3 else cell_img
                if cell_layer_name in self.viewer.layers and self.viewer.layers[cell_layer_name].data.ndim == cell_img_t.ndim:
                    self.viewer.layers[cell_layer_name].data = cell_img_t
                else:
                    if cell_layer_name in self.viewer.layers:
                        self.viewer.layers.remove(cell_layer_name)
                    self.viewer.add_image(cell_img_t, name=cell_layer_name, colormap="gray")

            if nuc_img_path.exists():
                nuc_img = tifffile.imread(str(nuc_img_path))
                nuc_img_t = nuc_img[frame] if nuc_img.ndim == 3 else nuc_img
                if nuc_layer_name in self.viewer.layers and self.viewer.layers[nuc_layer_name].data.ndim == nuc_img_t.ndim:
                    layer = self.viewer.layers[nuc_layer_name]
                    layer.data = nuc_img_t
                    layer.colormap = "bop orange"
                    layer.blending = "additive"
                else:
                    if nuc_layer_name in self.viewer.layers:
                        self.viewer.layers.remove(nuc_layer_name)
                    self.viewer.add_image(
                        nuc_img_t,
                        name=nuc_layer_name,
                        colormap="bop orange",
                        blending="additive",
                    )

            cell_labels_u32 = cell_labels.astype(np.uint32)
            if labels_layer_name in self.viewer.layers and self.viewer.layers[labels_layer_name].data.ndim == cell_labels_u32.ndim:
                self.viewer.layers[labels_layer_name].data = cell_labels_u32
            else:
                if labels_layer_name in self.viewer.layers:
                    self.viewer.layers.remove(labels_layer_name)
                self.viewer.add_labels(cell_labels_u32, name=labels_layer_name)

            # Reorder: cell_avg at bottom, nucleus_avg above it, labels on top
            def _layer_index(name):
                try:
                    return self.viewer.layers.index(self.viewer.layers[name])
                except KeyError:
                    return None

            for target_idx, name in enumerate([cell_layer_name, nuc_layer_name]):
                idx = _layer_index(name)
                if idx is not None and idx != target_idx:
                    self.viewer.layers.move(idx, target_idx)

            n_cells = len(np.unique(cell_labels)) - 1
            self._seg_status.setText(f"Preview: {n_cells} cells.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._seg_status.setText(f"Preview error: {e}")

    # ── Run ────────────────────────────────────────────────────────────────

    def _seg_on_run(self) -> None:
        root_dir = self._get_root_dir()
        if not root_dir:
            self._seg_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._state.current_position)
        method = self._active_method()
        overwrite = self._seg_overwrite_chk.isChecked()

        self._seg_run_btn.setEnabled(False)
        self._seg_term_btn.setEnabled(False)
        self._seg_cancel_btn.setEnabled(True)
        self._seg_progress.setVisible(True)
        self._seg_progress.setValue(0)
        self._seg_status.setText("Running segmentation…")

        if method == "gravity_flow":
            cfg = self._build_gravity_config()

            @thread_worker(
                connect={
                    "yielded": self._seg_on_progress,
                    "finished": self._seg_on_finished,
                    "errored": self._seg_on_error,
                }
            )
            def _work():
                from cellflow.core.logging import StageLogger
                from cellflow.core.paths import log_path
                with StageLogger(log_path(root_dir, pos), "cell_segmentation"):
                    for update in run_segmentation(root_dir, pos, cfg, overwrite=overwrite):
                        yield update
        else:
            cfg = self._build_watershed_config()

            @thread_worker(
                connect={
                    "yielded": self._seg_on_progress,
                    "finished": self._seg_on_finished,
                    "errored": self._seg_on_error,
                }
            )
            def _work():
                from cellflow.core.logging import StageLogger
                from cellflow.core.paths import log_path
                with StageLogger(log_path(root_dir, pos), "cell_segmentation"):
                    for update in run_watershed_segmentation(root_dir, pos, cfg, overwrite=overwrite):
                        yield update

        self.run_started.emit()
        self._seg_worker = _work()
        self._seg_worker.aborted.connect(self._seg_on_cancelled)

    def _seg_on_run_terminal(self) -> None:
        root_dir = self._get_root_dir()
        if not root_dir:
            self._seg_status.setText("No project open. Create or open a project first.")
            return

        method = self._active_method()
        if method == "watershed":
            self._seg_status.setText("Terminal run not available for Watershed — use 'Run Segmentation'.")
            return

        pos = int(self._state.current_position)
        cfg = self._build_gravity_config()
        overwrite_flag = "--overwrite" if self._seg_overwrite_chk.isChecked() else ""

        stage_cfg = {
            "flow_step_scale":         cfg.flow_step_scale,
            "cellpose_prob_threshold": cfg.cellpose_prob_threshold,
            "flow_smoothing_sigma":    cfg.flow_smoothing_sigma,
            "max_iterations":          cfg.euler_max_steps,
            "capture_radius":          cfg.capture_radius,
            "flow_weight":             cfg.flow_weight,
            "gravity_falloff":         cfg.gravity_falloff,
        }
        cfg_path = Path(tempfile.mktemp(suffix="_euler_cfg.json"))
        cfg_path.write_text(json.dumps(stage_cfg, indent=2))

        cmd = (
            f"\"{sys.executable}\" -m cellflow.cellpose.stages.cell_segmentation"
            f" --root-dir \"{root_dir}\""
            f" --pos {pos}"
            f" --config \"{cfg_path}\""
            f" {overwrite_flag}"
        ).strip()

        try:
            launch_in_terminal(cmd)
            self._seg_status.setText("Launched segmentation in terminal.")
        except Exception as e:
            self._seg_status.setText(f"Terminal launch error: {e}")

    # ── Progress / finished / error callbacks ──────────────────────────────

    def _seg_on_progress(self, update: tuple) -> None:
        done, total, label = update
        self._seg_progress.setMaximum(max(total, 1))
        self._seg_progress.setValue(done)
        pct = int(100 * done / total) if total > 0 else 0
        self._seg_status.setText(f"Segmentation: {done}/{total} frames ({pct}%)")

    def _seg_on_finished(self) -> None:
        self._seg_run_btn.setEnabled(True)
        self._seg_term_btn.setEnabled(True)
        self._seg_cancel_btn.setEnabled(False)
        self._seg_progress.setVisible(False)
        self._seg_status.setText("Done — Segmentation complete.")
        self._seg_worker = None
        self._log_viewer.refresh()

    def _seg_on_error(self, exc: Exception) -> None:
        self._seg_run_btn.setEnabled(True)
        self._seg_term_btn.setEnabled(True)
        self._seg_cancel_btn.setEnabled(False)
        self._seg_progress.setVisible(False)
        self._seg_status.setText(f"Error: {exc}")
        self._seg_worker = None
        self._log_viewer.refresh()

    def _seg_on_cancelled(self) -> None:
        self._seg_run_btn.setEnabled(True)
        self._seg_term_btn.setEnabled(True)
        self._seg_cancel_btn.setEnabled(False)
        self._seg_progress.setVisible(False)
        self._seg_status.setText("Cancelled.")
        self._seg_worker = None

    def _seg_on_cancel(self) -> None:
        if self._seg_worker:
            self._seg_worker.quit()

    # ── Load results ───────────────────────────────────────────────────────

    def _seg_on_load_results(self) -> None:
        root_dir = self._get_root_dir()
        if not root_dir:
            self._seg_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._state.current_position)
        out_dir = cell_segmentation_dir(root_dir, pos)
        labels_path = out_dir / "cell_labels.tif"

        if not labels_path.exists():
            self._seg_status.setText("No cell labels found.")
            return

        cell_stack = tifffile.imread(str(labels_path)).astype(np.uint32)

        raw_dir = stage_dir(root_dir, pos, "raw_import")
        cell_img_path = raw_dir / "cell" / "cell_zavg.tif"
        nuc_img_path  = raw_dir / "nucleus" / "nucleus_zavg.tif"

        cell_layer_name   = "Cell avg"
        nuc_layer_name    = "Nucleus avg"
        labels_layer_name = "Cell labels"

        if cell_img_path.exists():
            cell_img = tifffile.imread(str(cell_img_path))
            if cell_layer_name in self.viewer.layers and self.viewer.layers[cell_layer_name].data.ndim == cell_img.ndim:
                self.viewer.layers[cell_layer_name].data = cell_img
            else:
                if cell_layer_name in self.viewer.layers:
                    self.viewer.layers.remove(cell_layer_name)
                self.viewer.add_image(cell_img, name=cell_layer_name, colormap="gray")

        if nuc_img_path.exists():
            nuc_img = tifffile.imread(str(nuc_img_path))
            if nuc_layer_name in self.viewer.layers and self.viewer.layers[nuc_layer_name].data.ndim == nuc_img.ndim:
                layer = self.viewer.layers[nuc_layer_name]
                layer.data = nuc_img
                layer.colormap = "bop orange"
                layer.blending = "additive"
            else:
                if nuc_layer_name in self.viewer.layers:
                    self.viewer.layers.remove(nuc_layer_name)
                self.viewer.add_image(
                    nuc_img,
                    name=nuc_layer_name,
                    colormap="bop orange",
                    blending="additive",
                )

        if labels_layer_name in self.viewer.layers and self.viewer.layers[labels_layer_name].data.ndim == cell_stack.ndim:
            self.viewer.layers[labels_layer_name].data = cell_stack
        else:
            if labels_layer_name in self.viewer.layers:
                self.viewer.layers.remove(labels_layer_name)
            self.viewer.add_labels(cell_stack, name=labels_layer_name)

        # Reorder: cell_avg at bottom, nucleus_avg above it, labels on top
        def _layer_index(name):
            try:
                return self.viewer.layers.index(self.viewer.layers[name])
            except KeyError:
                return None

        for target_idx, name in enumerate([cell_layer_name, nuc_layer_name]):
            idx = _layer_index(name)
            if idx is not None and idx != target_idx:
                self.viewer.layers.move(idx, target_idx)

        self._seg_status.setText(f"Loaded cell_labels.tif  shape={cell_stack.shape}")

    # ── get_params / set_params ────────────────────────────────────────────

    def get_params(self) -> dict:
        return {
            "method": self._active_method(),
            "gravity_flow": self._build_gravity_config().model_dump(),
            "watershed": self._build_watershed_config().model_dump(),
            "seg_overwrite": self._seg_overwrite_chk.isChecked(),
        }

    def set_params(self, data: dict) -> None:
        method = data.get("method")

        # Support old flat format (pre-tab) — map directly into gravity_flow config
        if "gravity_flow" not in data and "watershed" not in data:
            cfg_data = {k: v for k, v in data.items() if k not in ("seg_overwrite", "pp_overwrite", "method")}
            if cfg_data:
                self._apply_gravity_config(CellSegmentationConfig.from_dict(cfg_data))
        else:
            if "gravity_flow" in data:
                self._apply_gravity_config(CellSegmentationConfig.from_dict(data["gravity_flow"]))
            if "watershed" in data:
                self._apply_watershed_config(WatershedConfig.from_dict(data["watershed"]))

        if method == "watershed":
            self._method_tabs.setCurrentIndex(1)
        elif method == "gravity_flow":
            self._method_tabs.setCurrentIndex(0)

        if "seg_overwrite" in data:
            self._seg_overwrite_chk.setChecked(bool(data["seg_overwrite"]))
