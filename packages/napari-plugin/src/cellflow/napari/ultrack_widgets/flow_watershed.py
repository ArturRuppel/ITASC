"""Nucleus-anchored cell segmentation widget (step 4 — 4_cell_segmentation).

Reads corrected 2D nuclear labels from the correction stage
(3_correction/nuclear_labels_corrected.tif) and Cellpose 2D flow/probability
maps from the cellpose cell stage (1_cellpose/cell/) to grow cell bodies
around each nucleus using a flow-guided expansion algorithm.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import tifffile
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari.qt.threading import thread_worker

from cellflow.core.paths import stage_dir
from cellflow.napari.runners.terminal import launch_in_terminal
from cellflow.napari.log_viewer import StageLogViewer
from cellflow.napari.registry import get_state


def cellpose_cell_dir(root_dir, pos):
    return stage_dir(root_dir, pos, "cellpose_cell")


def cell_segmentation_dir(root_dir, pos):
    return stage_dir(root_dir, pos, "cell_segmentation")


class FlowWatershedConfig:
    """Configuration for flow watershed segmentation."""

    def __init__(
        self,
        flow_scale: float = 1.0,
        cellpose_prob_threshold: float = 0.0,
        flow_smoothing_sigma: float = 0.0,
        max_iterations: int = 50,
        uniform_growth_rate: float = 0.2,
        opening_radius: int = 1,
        closing_radius: int = 1,
        boundary_smoothness: float = 0.5,
        fill_holes_threshold: float = 0.5,
    ):
        self.flow_scale = flow_scale
        self.cellpose_prob_threshold = cellpose_prob_threshold
        self.flow_smoothing_sigma = flow_smoothing_sigma
        self.max_iterations = max_iterations
        self.uniform_growth_rate = uniform_growth_rate
        self.opening_radius = opening_radius
        self.closing_radius = closing_radius
        self.boundary_smoothness = boundary_smoothness
        self.fill_holes_threshold = fill_holes_threshold

    def model_dump(self) -> dict:
        return {
            "flow_scale": self.flow_scale,
            "cellpose_prob_threshold": self.cellpose_prob_threshold,
            "flow_smoothing_sigma": self.flow_smoothing_sigma,
            "max_iterations": self.max_iterations,
            "uniform_growth_rate": self.uniform_growth_rate,
            "opening_radius": self.opening_radius,
            "closing_radius": self.closing_radius,
            "boundary_smoothness": self.boundary_smoothness,
            "fill_holes_threshold": self.fill_holes_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FlowWatershedConfig:
        # Drop 'method' if present in loaded config for backwards compatibility
        data.pop("method", None)
        return cls(**data)


def _load_nuclear_labels(root_dir: Path | str, pos: int) -> np.ndarray | None:
    """Load corrected 2D nuclear labels from the correction stage output."""
    try:
        correction_labels_path = (
            stage_dir(root_dir, pos, "correction") / "nuclear_labels_corrected.tif"
        )
        if correction_labels_path.exists():
            return tifffile.imread(str(correction_labels_path)).astype(np.int32)
    except Exception:
        pass
    return None


def _load_cellpose_data(root_dir: Path | str, pos: int, t: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load cellpose flow and probability from s1b for a given timepoint."""
    try:
        cell_dir = cellpose_cell_dir(root_dir, pos)

        # Load flow field (if available)
        flow_path = cell_dir / "cell_dp.tif"
        flow = None
        if flow_path.exists():
            flow = tifffile.imread(str(flow_path)).astype(np.float32)
            # Handle shape: cellpose outputs (T, 2, H, W), need to transpose to (T, H, W, 2) or (H, W, 2)
            if flow.ndim == 4:
                # Transpose from (T, 2, H, W) to (T, H, W, 2)
                flow = np.transpose(flow, (0, 2, 3, 1))
                flow = flow[t]  # Get timepoint t (now shape H, W, 2)
            elif flow.ndim == 3:
                # If already single timepoint, transpose from (2, H, W) to (H, W, 2)
                flow = np.transpose(flow, (1, 2, 0))
            else:
                flow = None

        # Load probability field
        prob_path = cell_dir / "cell_prob.tif"
        prob = None
        if prob_path.exists():
            prob = tifffile.imread(str(prob_path)).astype(np.float32)
            if prob.ndim == 3:
                prob = prob[t]  # Get timepoint t

        return flow, prob
    except Exception:
        return None, None


def run_segmentation_only(
    root_dir: str | Path,
    pos: int,
    config: FlowWatershedConfig,
) -> Generator:
    """
    Run flow watershed segmentation only (no postprocessing) for a full stack.
    Yields (done, total, label) tuples for progress reporting.
    Returns path to saved raw labels file.
    """
    from cellflow.cellpose.processing.flow_watershed import flow_guided_watershed

    root_dir = Path(root_dir)

    # Load nuclear labels
    nuclear_labels = _load_nuclear_labels(root_dir, pos)
    if nuclear_labels is None:
        print("Could not load nuclear labels")
        return None

    if nuclear_labels.ndim != 3:
        print(f"Expected (T, H, W), got {nuclear_labels.shape}")
        return None

    T = nuclear_labels.shape[0]
    cell_labels_stack = []

    # Load cellpose data
    flow_full, prob_full = _load_cellpose_data(root_dir, pos, 0)
    if flow_full is None:
        print(f"Could not load cellpose flow")
        return None

    # Process each timepoint
    for t in range(T):
        try:
            nuc_t = nuclear_labels[t]

            # Get flow/prob for this timepoint
            if flow_full.ndim == 4 and flow_full.shape[0] == T:
                flow_t = flow_full[t]
            else:
                flow_t = flow_full

            if prob_full is not None and prob_full.ndim == 3 and prob_full.shape[0] == T:
                prob_t = prob_full[t]
            else:
                prob_t = prob_full

            # Run segmentation (no postprocessing)
            cell_labels = flow_guided_watershed(
                nuc_t,
                flow_t,
                cellpose_prob=prob_t,
                flow_scale=config.flow_scale,
                cellpose_prob_threshold=config.cellpose_prob_threshold,
                flow_smoothing_sigma=config.flow_smoothing_sigma,
                max_iterations=config.max_iterations,
                uniform_growth_rate=config.uniform_growth_rate,
            )

            cell_labels_stack.append(cell_labels)

        except Exception as e:
            print(f"Error at t{t:03d}: {e}")
            cell_labels_stack.append(np.zeros_like(nuclear_labels[t], dtype=np.int32))

        yield (t + 1, T, f"t{t:03d}")

    # Save raw labels
    stack = np.stack(cell_labels_stack, axis=0).astype(np.int32)
    out_dir = cell_segmentation_dir(root_dir, pos)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cell_labels_raw.tif"
    tifffile.imwrite(
        str(out_path),
        stack,
        compression="zlib",
        metadata={"axes": "TYX"},
    )
    return str(out_path)


def run_postprocessing_only(
    root_dir: str | Path,
    pos: int,
    config: FlowWatershedConfig,
) -> Generator:
    """
    Run postprocessing on existing raw labels.
    Yields (done, total, label) tuples for progress reporting.
    Returns path to saved final labels file.
    """
    from cellflow.cellpose.processing.flow_watershed_postproc import postprocess_flow_watershed

    root_dir = Path(root_dir)
    out_dir = cell_segmentation_dir(root_dir, pos)
    raw_path = out_dir / "cell_labels_raw.tif"

    # Load raw labels
    if not raw_path.exists():
        print(f"Could not find raw labels at {raw_path}")
        return None

    raw_labels = tifffile.imread(str(raw_path)).astype(np.int32)
    if raw_labels.ndim != 3:
        print(f"Expected (T, H, W), got {raw_labels.shape}")
        return None

    T = raw_labels.shape[0]
    processed_stack = []

    # Load cellpose probability for trimming
    prob_full = None
    try:
        cell_dir = cellpose_cell_dir(root_dir, pos)
        prob_path = cell_dir / "cell_prob.tif"
        if prob_path.exists():
            prob_full = tifffile.imread(str(prob_path)).astype(np.float32)
    except Exception:
        pass

    # Process each timepoint
    for t in range(T):
        try:
            raw_t = raw_labels[t]

            # Get prob for this timepoint
            if prob_full is not None and prob_full.ndim == 3 and prob_full.shape[0] == T:
                prob_t = prob_full[t]
            else:
                prob_t = prob_full

            # Apply postprocessing
            processed = postprocess_flow_watershed(
                raw_t,
                cellpose_prob=prob_t,
                opening_radius=config.opening_radius,
                closing_radius=config.closing_radius,
                boundary_smoothness=config.boundary_smoothness,
                fill_holes_threshold=config.fill_holes_threshold,
            )

            processed_stack.append(processed)

        except Exception as e:
            print(f"Error at t{t:03d}: {e}")
            processed_stack.append(np.zeros_like(raw_labels[t], dtype=np.int32))

        yield (t + 1, T, f"t{t:03d}")

    # Save final labels
    stack = np.stack(processed_stack, axis=0).astype(np.int32)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cell_labels.tif"
    tifffile.imwrite(
        str(out_path),
        stack,
        compression="zlib",
        metadata={"axes": "TYX"},
    )
    return str(out_path)


def run_full_pipeline(
    root_dir: str | Path,
    pos: int,
    config: FlowWatershedConfig,
) -> Generator:
    """
    Run flow watershed segmentation + postprocessing for a full stack.
    Yields (done, total, label) tuples for progress reporting.
    Returns path to saved final labels file.
    """
    from cellflow.cellpose.processing.flow_watershed import flow_guided_watershed
    from cellflow.cellpose.processing.flow_watershed_postproc import postprocess_flow_watershed

    root_dir = Path(root_dir)

    # Load nuclear labels
    nuclear_labels = _load_nuclear_labels(root_dir, pos)
    if nuclear_labels is None:
        print("Could not load nuclear labels")
        return None

    if nuclear_labels.ndim != 3:
        print(f"Expected (T, H, W), got {nuclear_labels.shape}")
        return None

    T = nuclear_labels.shape[0]
    cell_labels_stack = []

    # Load cellpose data
    flow_full, prob_full = _load_cellpose_data(root_dir, pos, 0)
    if flow_full is None:
        print(f"Could not load cellpose flow")
        return None

    # Process each timepoint
    for t in range(T):
        try:
            nuc_t = nuclear_labels[t]

            # Get flow/prob for this timepoint
            if flow_full.ndim == 4 and flow_full.shape[0] == T:
                flow_t = flow_full[t]
            else:
                flow_t = flow_full

            if prob_full is not None and prob_full.ndim == 3 and prob_full.shape[0] == T:
                prob_t = prob_full[t]
            else:
                prob_t = prob_full

            # Run segmentation
            cell_labels = flow_guided_watershed(
                nuc_t,
                flow_t,
                cellpose_prob=prob_t,
                flow_scale=config.flow_scale,
                cellpose_prob_threshold=config.cellpose_prob_threshold,
                flow_smoothing_sigma=config.flow_smoothing_sigma,
                max_iterations=config.max_iterations,
                uniform_growth_rate=config.uniform_growth_rate,
            )

            # Apply post-processing
            cell_labels = postprocess_flow_watershed(
                cell_labels,
                cellpose_prob=prob_t,
                opening_radius=config.opening_radius,
                closing_radius=config.closing_radius,
                boundary_smoothness=config.boundary_smoothness,
                fill_holes_threshold=config.fill_holes_threshold,
            )

            cell_labels_stack.append(cell_labels)

        except Exception as e:
            print(f"Error at t{t:03d}: {e}")
            cell_labels_stack.append(np.zeros_like(nuclear_labels[t], dtype=np.int32))

        yield (t + 1, T, f"t{t:03d}")

    # Save outputs
    stack = np.stack(cell_labels_stack, axis=0).astype(np.int32)
    out_dir = cell_segmentation_dir(root_dir, pos)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Also save raw labels if not present
    raw_path = out_dir / "cell_labels_raw.tif"
    final_path = out_dir / "cell_labels.tif"

    # Re-run just segmentation to get raw labels for saving
    if not raw_path.exists():
        cell_labels_raw_stack = []
        for t in range(T):
            try:
                nuc_t = nuclear_labels[t]
                flow_t = flow_full[t] if flow_full.ndim == 4 else flow_full
                prob_t = prob_full[t] if prob_full is not None and prob_full.ndim == 3 else prob_full

                cell_labels_raw = flow_guided_watershed(
                    nuc_t,
                    flow_t,
                    cellpose_prob=prob_t,
                    flow_scale=config.flow_scale,
                    cellpose_prob_threshold=config.cellpose_prob_threshold,
                    flow_smoothing_sigma=config.flow_smoothing_sigma,
                    max_iterations=config.max_iterations,
                    uniform_growth_rate=config.uniform_growth_rate,
                )
                cell_labels_raw_stack.append(cell_labels_raw)
            except Exception:
                cell_labels_raw_stack.append(np.zeros_like(nuclear_labels[t], dtype=np.int32))

        raw_stack = np.stack(cell_labels_raw_stack, axis=0).astype(np.int32)
        tifffile.imwrite(
            str(raw_path),
            raw_stack,
            compression="zlib",
            metadata={"axes": "TYX"},
        )

    tifffile.imwrite(
        str(final_path),
        stack,
        compression="zlib",
        metadata={"axes": "TYX"},
    )
    return str(final_path)


class FlowGuidedSegmentationWidget(QWidget):
    """Widget for flow-guided watershed cell segmentation with independent segmentation and postprocessing stages."""

    def __init__(self, viewer: "napari.Viewer") -> None:
        super().__init__()
        self.viewer = viewer
        self._state = get_state(viewer)
        self._seg_worker = None
        self._pp_worker = None
        self._all_worker = None

        # ── Outer scroll area ────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        self._inner_layout = QVBoxLayout()
        self._inner_layout.setAlignment(Qt.AlignTop)
        inner.setLayout(self._inner_layout)
        scroll.setWidget(inner)

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.setLayout(outer)

        lay = self._inner_layout

        # ── Project info (derived from state) ────────────────────────────
        self._project_label = QLabel("No project open.")
        self._project_label.setStyleSheet("color: white; font-size: 8pt;")
        self._project_label.setWordWrap(True)
        lay.addWidget(self._project_label)

        row = QHBoxLayout()
        row.addWidget(QLabel("Position"))
        self._pos_spin = QSpinBox()
        self._pos_spin.setRange(0, 1000)
        self._pos_spin.setValue(0)
        self._pos_spin.valueChanged.connect(self._sync_project_dir)
        row.addWidget(self._pos_spin)
        row.addStretch()
        lay.addLayout(row)

        # ── Save / Load all parameters ───────────────────────────────────
        row = QHBoxLayout()
        save_btn = QPushButton("Save All Parameters…")
        save_btn.clicked.connect(self._on_save_all_params)
        row.addWidget(save_btn)
        load_btn = QPushButton("Load All Parameters…")
        load_btn.clicked.connect(self._on_load_all_params)
        row.addWidget(load_btn)
        lay.addLayout(row)

        # ── Build stage sections ─────────────────────────────────────────
        lay.addWidget(self._build_segmentation_section())
        lay.addWidget(self._build_postprocessing_section())

        # ── Run Full Pipeline ────────────────────────────────────────────
        row = QHBoxLayout()
        self._all_run_btn = QPushButton("Run Full Pipeline (Segmentation → Post-processing)")
        self._all_run_btn.clicked.connect(self._all_on_run)
        row.addWidget(self._all_run_btn)
        self._all_term_btn = QPushButton("Run in Terminal")
        self._all_term_btn.clicked.connect(self._all_on_run_terminal)
        row.addWidget(self._all_term_btn)
        self._all_cancel_btn = QPushButton("Cancel")
        self._all_cancel_btn.setEnabled(False)
        self._all_cancel_btn.clicked.connect(self._all_on_cancel)
        row.addWidget(self._all_cancel_btn)
        lay.addLayout(row)

        self._all_progress = QProgressBar()
        self._all_progress.setVisible(False)
        lay.addWidget(self._all_progress)

        self._all_status = QLabel("")
        lay.addWidget(self._all_status)

        self._log_viewer = StageLogViewer(self._state)
        lay.addWidget(self._log_viewer)

        # Connect project-change signal
        self._state.pipeline_schema_changed.connect(self._sync_project_dir)
        self._sync_project_dir()

    # ── Project-derived path helpers ─────────────────────────────────────

    def _get_root_dir(self) -> str | None:
        project_dir = self._state.project_dir
        if project_dir is None:
            return None
        return str(project_dir)

    def _sync_project_dir(self) -> None:
        """Update project info label when project or position changes."""
        root = self._get_root_dir()
        if root is None:
            self._project_label.setText(
                "No project open — create or open one via the Project panel."
            )
            return
        pos = self._pos_spin.value()
        self._project_label.setText(
            f"Root: {root}  |  Position: pos{pos:02d}"
        )

    def _build_segmentation_section(self) -> QGroupBox:
        """Build the Segmentation section."""
        grp = QGroupBox("Segmentation")
        lay = QVBoxLayout()

        # Preview frame
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview frame"))
        self._seg_frame_spin = QSpinBox()
        self._seg_frame_spin.setRange(0, 1000)
        self._seg_frame_spin.setValue(0)
        row.addWidget(self._seg_frame_spin)
        lay.addLayout(row)

        # Parameters
        row = QHBoxLayout()
        row.addWidget(QLabel("Flow scale (blend factor)"))
        self._seg_flow_scale_spin = QDoubleSpinBox()
        self._seg_flow_scale_spin.setRange(0.0, 3.0)
        self._seg_flow_scale_spin.setSingleStep(0.1)
        self._seg_flow_scale_spin.setDecimals(2)
        self._seg_flow_scale_spin.setValue(1.0)
        row.addWidget(self._seg_flow_scale_spin)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Cellpose prob threshold"))
        self._seg_prob_threshold_spin = QDoubleSpinBox()
        self._seg_prob_threshold_spin.setRange(-100.0, 100.0)
        self._seg_prob_threshold_spin.setSingleStep(1.0)
        self._seg_prob_threshold_spin.setDecimals(1)
        self._seg_prob_threshold_spin.setValue(0.0)
        row.addWidget(self._seg_prob_threshold_spin)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Flow smoothing (σ)"))
        self._seg_smoothing_spin = QDoubleSpinBox()
        self._seg_smoothing_spin.setRange(0.0, 5.0)
        self._seg_smoothing_spin.setSingleStep(0.1)
        self._seg_smoothing_spin.setDecimals(2)
        self._seg_smoothing_spin.setValue(0.0)
        row.addWidget(self._seg_smoothing_spin)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Max iterations"))
        self._seg_max_iter_spin = QSpinBox()
        self._seg_max_iter_spin.setRange(1, 2000)
        self._seg_max_iter_spin.setSingleStep(10)
        self._seg_max_iter_spin.setValue(50)
        row.addWidget(self._seg_max_iter_spin)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Uniform growth rate"))
        self._seg_uniform_growth_spin = QDoubleSpinBox()
        self._seg_uniform_growth_spin.setRange(0.0, 1.0)
        self._seg_uniform_growth_spin.setSingleStep(0.05)
        self._seg_uniform_growth_spin.setDecimals(2)
        self._seg_uniform_growth_spin.setValue(0.2)
        row.addWidget(self._seg_uniform_growth_spin)
        row.addStretch()
        lay.addLayout(row)

        # Preview button
        self._seg_preview_btn = QPushButton("Preview")
        self._seg_preview_btn.clicked.connect(self._seg_on_preview)
        lay.addWidget(self._seg_preview_btn)

        # Overwrite
        self._seg_overwrite_chk = QCheckBox("Overwrite existing files")
        lay.addWidget(self._seg_overwrite_chk)

        # Buttons
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

        # Load results
        self._seg_load_btn = QPushButton("Load Results")
        self._seg_load_btn.clicked.connect(self._seg_on_load_results)
        lay.addWidget(self._seg_load_btn)

        # Progress
        self._seg_progress = QProgressBar()
        self._seg_progress.setVisible(False)
        lay.addWidget(self._seg_progress)
        self._seg_status = QLabel("")
        lay.addWidget(self._seg_status)

        grp.setLayout(lay)
        return grp

    def _build_postprocessing_section(self) -> QGroupBox:
        """Build the Post-processing section."""
        grp = QGroupBox("Post-processing")
        lay = QVBoxLayout()

        # Preview frame
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview frame"))
        self._pp_frame_spin = QSpinBox()
        self._pp_frame_spin.setRange(0, 1000)
        self._pp_frame_spin.setValue(0)
        row.addWidget(self._pp_frame_spin)
        lay.addLayout(row)

        # Parameters
        row = QHBoxLayout()
        row.addWidget(QLabel("Opening radius"))
        self._pp_opening_spin = QSpinBox()
        self._pp_opening_spin.setRange(0, 10)
        self._pp_opening_spin.setValue(1)
        row.addWidget(self._pp_opening_spin)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Closing radius"))
        self._pp_closing_spin = QSpinBox()
        self._pp_closing_spin.setRange(0, 10)
        self._pp_closing_spin.setValue(1)
        row.addWidget(self._pp_closing_spin)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Boundary smoothness"))
        self._pp_boundary_spin = QDoubleSpinBox()
        self._pp_boundary_spin.setRange(0.0, 1.0)
        self._pp_boundary_spin.setSingleStep(0.05)
        self._pp_boundary_spin.setDecimals(2)
        self._pp_boundary_spin.setValue(0.5)
        row.addWidget(self._pp_boundary_spin)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Fill holes threshold"))
        self._pp_prob_trim_spin = QDoubleSpinBox()
        self._pp_prob_trim_spin.setRange(0.0, 1.0)
        self._pp_prob_trim_spin.setSingleStep(0.05)
        self._pp_prob_trim_spin.setDecimals(2)
        self._pp_prob_trim_spin.setValue(0.5)
        row.addWidget(self._pp_prob_trim_spin)
        row.addStretch()
        lay.addLayout(row)

        # Preview button
        self._pp_preview_btn = QPushButton("Preview")
        self._pp_preview_btn.clicked.connect(self._pp_on_preview)
        lay.addWidget(self._pp_preview_btn)

        # Overwrite
        self._pp_overwrite_chk = QCheckBox("Overwrite existing files")
        lay.addWidget(self._pp_overwrite_chk)

        # Buttons
        row = QHBoxLayout()
        self._pp_run_btn = QPushButton("Run Post-processing")
        self._pp_run_btn.clicked.connect(self._pp_on_run)
        row.addWidget(self._pp_run_btn)
        self._pp_term_btn = QPushButton("Run in Terminal")
        self._pp_term_btn.clicked.connect(self._pp_on_run_terminal)
        row.addWidget(self._pp_term_btn)
        self._pp_cancel_btn = QPushButton("Cancel")
        self._pp_cancel_btn.setEnabled(False)
        self._pp_cancel_btn.clicked.connect(self._pp_on_cancel)
        row.addWidget(self._pp_cancel_btn)
        lay.addLayout(row)

        # Load results
        self._pp_load_btn = QPushButton("Load Results")
        self._pp_load_btn.clicked.connect(self._pp_on_load_results)
        lay.addWidget(self._pp_load_btn)

        # Progress
        self._pp_progress = QProgressBar()
        self._pp_progress.setVisible(False)
        lay.addWidget(self._pp_progress)
        self._pp_status = QLabel("")
        lay.addWidget(self._pp_status)

        grp.setLayout(lay)
        return grp

    # ════════════════════════════════════════════════════════════════════
    # Config helpers
    # ════════════════════════════════════════════════════════════════════

    def _build_config(self) -> FlowWatershedConfig:
        """Build config from current UI state."""
        return FlowWatershedConfig(
            flow_scale=self._seg_flow_scale_spin.value(),
            cellpose_prob_threshold=self._seg_prob_threshold_spin.value(),
            flow_smoothing_sigma=self._seg_smoothing_spin.value(),
            max_iterations=self._seg_max_iter_spin.value(),
            uniform_growth_rate=self._seg_uniform_growth_spin.value(),
            opening_radius=self._pp_opening_spin.value(),
            closing_radius=self._pp_closing_spin.value(),
            boundary_smoothness=self._pp_boundary_spin.value(),
            fill_holes_threshold=self._pp_prob_trim_spin.value(),
        )

    def _apply_config(self, cfg: FlowWatershedConfig) -> None:
        """Apply config to UI."""
        self._seg_flow_scale_spin.setValue(cfg.flow_scale)
        self._seg_prob_threshold_spin.setValue(cfg.cellpose_prob_threshold)
        self._seg_smoothing_spin.setValue(cfg.flow_smoothing_sigma)
        self._seg_max_iter_spin.setValue(cfg.max_iterations)
        self._seg_uniform_growth_spin.setValue(cfg.uniform_growth_rate)
        self._pp_opening_spin.setValue(cfg.opening_radius)
        self._pp_closing_spin.setValue(cfg.closing_radius)
        self._pp_boundary_spin.setValue(cfg.boundary_smoothness)
        self._pp_prob_trim_spin.setValue(cfg.fill_holes_threshold)

    # ════════════════════════════════════════════════════════════════════
    # Segmentation section callbacks
    # ════════════════════════════════════════════════════════════════════

    def _seg_on_preview(self) -> None:
        """Preview single frame segmentation."""
        root_dir = self._get_root_dir()
        if not root_dir:
            self._seg_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._pos_spin.value())
        frame = int(self._seg_frame_spin.value())
        cfg = self._build_config()

        self._seg_status.setText(f"Processing frame {frame}…")

        try:
            from cellflow.cellpose.processing.flow_watershed import flow_guided_watershed
            root_dir_path = Path(root_dir)

            # Load nuclear labels
            nuclear_labels = _load_nuclear_labels(root_dir_path, pos)
            if nuclear_labels is None:
                self._seg_status.setText("Could not load nuclear labels.")
                return

            # Load cellpose data
            flow_full, prob_full = _load_cellpose_data(root_dir_path, pos, frame)
            if flow_full is None:
                self._seg_status.setText("Could not load cellpose flow.")
                return

            nuc_t = nuclear_labels[frame]

            # Get flow/prob for this frame
            if flow_full.ndim == 4 and flow_full.shape[0] == nuclear_labels.shape[0]:
                flow_t = flow_full[frame]
            else:
                flow_t = flow_full

            if prob_full is not None and prob_full.ndim == 3 and prob_full.shape[0] == nuclear_labels.shape[0]:
                prob_t = prob_full[frame]
            else:
                prob_t = prob_full

            # Run segmentation
            cell_labels = flow_guided_watershed(
                nuc_t,
                flow_t,
                cellpose_prob=prob_t,
                flow_scale=cfg.flow_scale,
                cellpose_prob_threshold=cfg.cellpose_prob_threshold,
                flow_smoothing_sigma=cfg.flow_smoothing_sigma,
                max_iterations=cfg.max_iterations,
                uniform_growth_rate=cfg.uniform_growth_rate,
            )

            # Display in napari
            flow_mag = np.sqrt(flow_t[..., 0]**2 + flow_t[..., 1]**2)

            # Clear existing layers
            while len(self.viewer.layers) > 0:
                self.viewer.layers.pop()

            self.viewer.add_image(nuc_t, name="Nuclear Labels")
            self.viewer.add_image(prob_t, name="Cellpose Probability")
            self.viewer.add_image(flow_mag, name="Cellpose Flow Magnitude")
            self.viewer.add_labels(cell_labels, name="Cell Segmentation (raw)")

            n_cells = len(np.unique(cell_labels)) - 1
            self._seg_status.setText(f"Preview complete. {n_cells} cells found.")

        except Exception as e:
            print(f"Preview error: {e}")
            import traceback
            traceback.print_exc()
            self._seg_status.setText(f"Preview error: {e}")

    def _seg_on_run(self) -> None:
        """Run segmentation in background."""
        root_dir = self._get_root_dir()
        if not root_dir:
            self._seg_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._pos_spin.value())
        cfg = self._build_config()

        self._seg_run_btn.setEnabled(False)
        self._seg_term_btn.setEnabled(False)
        self._seg_cancel_btn.setEnabled(True)
        self._seg_progress.setVisible(True)
        self._seg_progress.setValue(0)
        self._seg_status.setText("Running segmentation…")

        @thread_worker(
            connect={
                "yielded": self._seg_on_progress,
                "finished": self._seg_on_finished,
                "errored": self._seg_on_error,
            }
        )
        def _work():
            for update in run_segmentation_only(root_dir, pos, cfg):
                yield update

        self._seg_worker = _work()
        self._seg_worker.aborted.connect(self._seg_on_cancelled)

    def _seg_on_run_terminal(self) -> None:
        """Launch segmentation in terminal."""
        root_dir = self._get_root_dir()
        if not root_dir:
            self._seg_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._pos_spin.value())
        cfg = self._build_config()
        overwrite_flag = "--overwrite" if self._seg_overwrite_chk.isChecked() else ""

        # Save config to temp file
        cfg_path = Path(tempfile.mktemp(suffix="_fw_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))

        cmd = (
            f"python -m cellflow.cellpose.stages.flow_watershed"
            f" --root-dir \"{root_dir}\""
            f" --pos {pos}"
            f" --config \"{cfg_path}\""
            f" --mode seg-only"
            f" {overwrite_flag}"
        ).strip()

        try:
            launch_in_terminal(cmd)
            self._seg_status.setText("Launched segmentation in terminal.")
        except Exception as e:
            self._seg_status.setText(f"Terminal launch error: {e}")

    def _seg_on_progress(self, update: tuple) -> None:
        """Handle progress updates from segmentation worker."""
        done, total, label = update
        self._seg_progress.setMaximum(max(total, 1))
        self._seg_progress.setValue(done)
        pct = int(100 * done / total) if total > 0 else 0
        self._seg_status.setText(f"Segmentation: {done}/{total} frames ({pct}%)")

    def _seg_on_finished(self) -> None:
        """Callback after segmentation completes."""
        self._seg_run_btn.setEnabled(True)
        self._seg_term_btn.setEnabled(True)
        self._seg_cancel_btn.setEnabled(False)
        self._seg_progress.setVisible(False)

        if self._seg_worker and hasattr(self._seg_worker, 'result'):
            result = self._seg_worker.result
            if result is not None:
                out_path = Path(result)
                cell_stack = tifffile.imread(str(out_path)).astype(np.uint32)
                self.viewer.add_labels(cell_stack, name="cells (raw)")
                self._seg_status.setText(f"Done. Saved to {out_path.name}")
            else:
                self._seg_status.setText("Processing failed.")
        else:
            self._seg_status.setText("Done — Segmentation complete.")

        self._seg_worker = None
        self._log_viewer.refresh()

    def _seg_on_error(self, exc: Exception) -> None:
        """Callback on segmentation error."""
        self._seg_run_btn.setEnabled(True)
        self._seg_term_btn.setEnabled(True)
        self._seg_cancel_btn.setEnabled(False)
        self._seg_progress.setVisible(False)
        self._seg_status.setText(f"Error: {exc}")
        self._seg_worker = None
        self._log_viewer.refresh()

    def _seg_on_cancelled(self) -> None:
        """Callback when segmentation is cancelled."""
        self._seg_run_btn.setEnabled(True)
        self._seg_term_btn.setEnabled(True)
        self._seg_cancel_btn.setEnabled(False)
        self._seg_progress.setVisible(False)
        self._seg_status.setText("Cancelled.")
        self._seg_worker = None

    def _seg_on_cancel(self) -> None:
        """Cancel segmentation worker."""
        if self._seg_worker:
            self._seg_worker.quit()

    def _seg_on_load_results(self) -> None:
        """Load segmentation results from disk."""
        root_dir = self._get_root_dir()
        if not root_dir:
            self._seg_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._pos_spin.value())
        out_dir = cell_segmentation_dir(root_dir, pos)
        raw_path = out_dir / "cell_labels_raw.tif"

        if not raw_path.exists():
            self._seg_status.setText("No cell_labels_raw.tif found.")
            return

        cell_stack = tifffile.imread(str(raw_path)).astype(np.uint32)
        nuc_labels = _load_nuclear_labels(root_dir, pos)

        # Clear and reload
        while len(self.viewer.layers) > 0:
            self.viewer.layers.pop()

        if nuc_labels is not None:
            self.viewer.add_labels(nuc_labels.astype(np.uint32), name="nuclei", opacity=0.3)
        self.viewer.add_labels(cell_stack, name="cells (raw)")

        self._seg_status.setText(f"Loaded raw segmentation, shape={cell_stack.shape}")

    # ════════════════════════════════════════════════════════════════════
    # Post-processing section callbacks
    # ════════════════════════════════════════════════════════════════════

    def _pp_on_preview(self) -> None:
        """Preview post-processing on a single frame."""
        root_dir = self._get_root_dir()
        if not root_dir:
            self._pp_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._pos_spin.value())
        frame = int(self._pp_frame_spin.value())
        cfg = self._build_config()

        self._pp_status.setText(f"Processing frame {frame}…")

        try:
            from cellflow.cellpose.processing.flow_watershed_postproc import postprocess_flow_watershed

            out_dir = cell_segmentation_dir(root_dir, pos)
            raw_path = out_dir / "cell_labels_raw.tif"

            if not raw_path.exists():
                self._pp_status.setText("Raw segmentation not found. Run segmentation first.")
                return

            # Load raw labels
            raw_labels = tifffile.imread(str(raw_path)).astype(np.int32)
            if frame >= raw_labels.shape[0]:
                self._pp_status.setText(f"Frame {frame} out of range (max {raw_labels.shape[0]-1})")
                return

            raw_t = raw_labels[frame]

            # Load cellpose probability for trimming
            prob_t = None
            try:
                cell_dir = cellpose_cell_dir(root_dir, pos)
                prob_path = cell_dir / "cell_prob.tif"
                if prob_path.exists():
                    prob_full = tifffile.imread(str(prob_path)).astype(np.float32)
                    if prob_full.ndim == 3:
                        prob_t = prob_full[frame]
                    else:
                        prob_t = prob_full
            except Exception:
                pass

            # Apply postprocessing
            pp_t = postprocess_flow_watershed(
                raw_t,
                cellpose_prob=prob_t,
                opening_radius=cfg.opening_radius,
                closing_radius=cfg.closing_radius,
                boundary_smoothness=cfg.boundary_smoothness,
                fill_holes_threshold=cfg.fill_holes_threshold,
            )

            # Display in napari
            while len(self.viewer.layers) > 0:
                self.viewer.layers.pop()

            self.viewer.add_labels(raw_t, name="Raw Segmentation", opacity=0.5)
            self.viewer.add_labels(pp_t, name="Post-processed")

            n_cells_raw = len(np.unique(raw_t)) - 1
            n_cells_pp = len(np.unique(pp_t)) - 1
            self._pp_status.setText(
                f"Preview complete. Raw: {n_cells_raw} cells, Post-processed: {n_cells_pp} cells."
            )

        except Exception as e:
            print(f"Preview error: {e}")
            import traceback
            traceback.print_exc()
            self._pp_status.setText(f"Preview error: {e}")

    def _pp_on_run(self) -> None:
        """Run post-processing in background."""
        root_dir = self._get_root_dir()
        if not root_dir:
            self._pp_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._pos_spin.value())
        cfg = self._build_config()

        self._pp_run_btn.setEnabled(False)
        self._pp_term_btn.setEnabled(False)
        self._pp_cancel_btn.setEnabled(True)
        self._pp_progress.setVisible(True)
        self._pp_progress.setValue(0)
        self._pp_status.setText("Running post-processing…")

        @thread_worker(
            connect={
                "yielded": self._pp_on_progress,
                "finished": self._pp_on_finished,
                "errored": self._pp_on_error,
            }
        )
        def _work():
            for update in run_postprocessing_only(root_dir, pos, cfg):
                yield update

        self._pp_worker = _work()
        self._pp_worker.aborted.connect(self._pp_on_cancelled)

    def _pp_on_run_terminal(self) -> None:
        """Launch post-processing in terminal."""
        root_dir = self._get_root_dir()
        if not root_dir:
            self._pp_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._pos_spin.value())
        cfg = self._build_config()
        overwrite_flag = "--overwrite" if self._pp_overwrite_chk.isChecked() else ""

        # Save config to temp file
        cfg_path = Path(tempfile.mktemp(suffix="_fw_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))

        cmd = (
            f"python -m cellflow.cellpose.stages.flow_watershed"
            f" --root-dir \"{root_dir}\""
            f" --pos {pos}"
            f" --config \"{cfg_path}\""
            f" --mode postprocess-only"
            f" {overwrite_flag}"
        ).strip()

        try:
            launch_in_terminal(cmd)
            self._pp_status.setText("Launched post-processing in terminal.")
        except Exception as e:
            self._pp_status.setText(f"Terminal launch error: {e}")

    def _pp_on_progress(self, update: tuple) -> None:
        """Handle progress updates from post-processing worker."""
        done, total, label = update
        self._pp_progress.setMaximum(max(total, 1))
        self._pp_progress.setValue(done)
        pct = int(100 * done / total) if total > 0 else 0
        self._pp_status.setText(f"Post-processing: {done}/{total} frames ({pct}%)")

    def _pp_on_finished(self) -> None:
        """Callback after post-processing completes."""
        self._pp_run_btn.setEnabled(True)
        self._pp_term_btn.setEnabled(True)
        self._pp_cancel_btn.setEnabled(False)
        self._pp_progress.setVisible(False)

        if self._pp_worker and hasattr(self._pp_worker, 'result'):
            result = self._pp_worker.result
            if result is not None:
                out_path = Path(result)
                cell_stack = tifffile.imread(str(out_path)).astype(np.uint32)
                self.viewer.add_labels(cell_stack, name="cells")
                self._pp_status.setText(f"Done. Saved to {out_path.name}")
            else:
                self._pp_status.setText("Processing failed.")
        else:
            self._pp_status.setText("Done — Post-processing complete.")

        self._pp_worker = None
        self._log_viewer.refresh()

    def _pp_on_error(self, exc: Exception) -> None:
        """Callback on post-processing error."""
        self._pp_run_btn.setEnabled(True)
        self._pp_term_btn.setEnabled(True)
        self._pp_cancel_btn.setEnabled(False)
        self._pp_progress.setVisible(False)
        self._pp_status.setText(f"Error: {exc}")
        self._pp_worker = None
        self._log_viewer.refresh()

    def _pp_on_cancelled(self) -> None:
        """Callback when post-processing is cancelled."""
        self._pp_run_btn.setEnabled(True)
        self._pp_term_btn.setEnabled(True)
        self._pp_cancel_btn.setEnabled(False)
        self._pp_progress.setVisible(False)
        self._pp_status.setText("Cancelled.")
        self._pp_worker = None

    def _pp_on_cancel(self) -> None:
        """Cancel post-processing worker."""
        if self._pp_worker:
            self._pp_worker.quit()

    def _pp_on_load_results(self) -> None:
        """Load post-processing results from disk."""
        root_dir = self._get_root_dir()
        if not root_dir:
            self._pp_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._pos_spin.value())
        out_dir = cell_segmentation_dir(root_dir, pos)
        final_path = out_dir / "cell_labels.tif"

        if not final_path.exists():
            self._pp_status.setText("No cell_labels.tif found.")
            return

        cell_stack = tifffile.imread(str(final_path)).astype(np.uint32)
        nuc_labels = _load_nuclear_labels(root_dir, pos)

        # Clear and reload
        while len(self.viewer.layers) > 0:
            self.viewer.layers.pop()

        if nuc_labels is not None:
            self.viewer.add_labels(nuc_labels.astype(np.uint32), name="nuclei", opacity=0.3)
        self.viewer.add_labels(cell_stack, name="cells")

        self._pp_status.setText(f"Loaded final segmentation, shape={cell_stack.shape}")

    # ════════════════════════════════════════════════════════════════════
    # Full pipeline callbacks
    # ════════════════════════════════════════════════════════════════════

    def _all_on_run(self) -> None:
        """Run full pipeline in background."""
        root_dir = self._get_root_dir()
        if not root_dir:
            self._all_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._pos_spin.value())
        cfg = self._build_config()

        self._all_run_btn.setEnabled(False)
        self._all_term_btn.setEnabled(False)
        self._all_cancel_btn.setEnabled(True)
        self._all_progress.setVisible(True)
        self._all_progress.setValue(0)
        self._all_status.setText("Running full pipeline…")

        @thread_worker(
            connect={
                "yielded": self._all_on_progress,
                "finished": self._all_on_finished,
                "errored": self._all_on_error,
            }
        )
        def _work():
            for update in run_full_pipeline(root_dir, pos, cfg):
                yield update

        self._all_worker = _work()
        self._all_worker.aborted.connect(self._all_on_cancelled)

    def _all_on_run_terminal(self) -> None:
        """Launch full pipeline in terminal."""
        root_dir = self._get_root_dir()
        if not root_dir:
            self._all_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._pos_spin.value())
        cfg = self._build_config()
        overwrite_flag = "--overwrite" if self._seg_overwrite_chk.isChecked() else ""

        # Save config to temp file
        cfg_path = Path(tempfile.mktemp(suffix="_fw_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))

        cmd = (
            f"python -m cellflow.cellpose.stages.flow_watershed"
            f" --root-dir \"{root_dir}\""
            f" --pos {pos}"
            f" --config \"{cfg_path}\""
            f" {overwrite_flag}"
        ).strip()

        try:
            launch_in_terminal(cmd)
            self._all_status.setText("Launched full pipeline in terminal.")
        except Exception as e:
            self._all_status.setText(f"Terminal launch error: {e}")

    def _all_on_progress(self, update: tuple) -> None:
        """Handle progress updates from full pipeline worker."""
        done, total, label = update
        self._all_progress.setMaximum(max(total, 1))
        self._all_progress.setValue(done)
        pct = int(100 * done / total) if total > 0 else 0
        self._all_status.setText(f"Pipeline: {done}/{total} frames ({pct}%)")

    def _all_on_finished(self) -> None:
        """Callback after full pipeline completes."""
        self._all_run_btn.setEnabled(True)
        self._all_term_btn.setEnabled(True)
        self._all_cancel_btn.setEnabled(False)
        self._all_progress.setVisible(False)

        if self._all_worker and hasattr(self._all_worker, 'result'):
            result = self._all_worker.result
            if result is not None:
                out_path = Path(result)
                cell_stack = tifffile.imread(str(out_path)).astype(np.uint32)
                self.viewer.add_labels(cell_stack, name="cells")
                self._all_status.setText(f"Done. Saved to {out_path.name}")
            else:
                self._all_status.setText("Processing failed.")
        else:
            self._all_status.setText("Done — Full pipeline complete.")

        self._all_worker = None
        self._log_viewer.refresh()

    def _all_on_error(self, exc: Exception) -> None:
        """Callback on full pipeline error."""
        self._all_run_btn.setEnabled(True)
        self._all_term_btn.setEnabled(True)
        self._all_cancel_btn.setEnabled(False)
        self._all_progress.setVisible(False)
        self._all_status.setText(f"Error: {exc}")
        self._all_worker = None
        self._log_viewer.refresh()

    def _all_on_cancelled(self) -> None:
        """Callback when full pipeline is cancelled."""
        self._all_run_btn.setEnabled(True)
        self._all_term_btn.setEnabled(True)
        self._all_cancel_btn.setEnabled(False)
        self._all_progress.setVisible(False)
        self._all_status.setText("Cancelled.")
        self._all_worker = None

    def _all_on_cancel(self) -> None:
        """Cancel full pipeline worker."""
        if self._all_worker:
            self._all_worker.quit()

    # ════════════════════════════════════════════════════════════════════
    # Save / Load all parameters
    # ════════════════════════════════════════════════════════════════════

    def _on_save_all_params(self) -> None:
        """Save all configuration parameters to JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save parameters", "", "JSON files (*.json)"
        )
        if not path:
            return
        cfg = self._build_config()
        Path(path).write_text(json.dumps(cfg.model_dump(), indent=2))

    def _on_load_all_params(self) -> None:
        """Load all configuration parameters from JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load parameters", "", "JSON files (*.json)"
        )
        if not path:
            return
        data = json.loads(Path(path).read_text())
        cfg = FlowWatershedConfig.from_dict(data)
        self._apply_config(cfg)
