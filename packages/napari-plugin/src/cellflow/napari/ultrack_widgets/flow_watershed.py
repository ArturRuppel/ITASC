"""Nucleus-anchored cell segmentation widget (step 4 — 4_cell_segmentation).

Reads corrected 2D nuclear labels from the correction stage
(3_correction/nuclear_labels_corrected.tif) and Cellpose 2D flow/probability
maps from the cellpose cell stage (1_cellpose/cell/) to grow cell bodies
around each nucleus using a flow-guided expansion algorithm.
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
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
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


_DEFAULT_POSTPROCESS_STEPS: list[dict] = [
    {"type": "open",            "radius":      1   },
    {"type": "close",           "radius":      1   },
    {"type": "smooth_boundary", "smoothness":  0.5 },
]


class FlowWatershedConfig:
    """Configuration for flow watershed segmentation."""

    def __init__(
        self,
        flow_scale: float = 1.0,
        cellpose_prob_threshold: float = 0.0,
        flow_smoothing_sigma: float = 0.0,
        max_iterations: int = 50,
        uniform_growth_rate: float = 0.2,
        flow_mag_scale: float = 3.0,
        postprocess_steps: list | None = None,
        foreground_mask_sigma: float = 2.0,
        foreground_mask_threshold: float = 0.1,
        foreground_mask_postprocess_steps: list | None = None,
    ):
        self.flow_scale = flow_scale
        self.cellpose_prob_threshold = cellpose_prob_threshold
        self.flow_smoothing_sigma = flow_smoothing_sigma
        self.max_iterations = max_iterations
        self.uniform_growth_rate = uniform_growth_rate
        self.flow_mag_scale = flow_mag_scale
        self.postprocess_steps = postprocess_steps if postprocess_steps is not None \
            else [dict(s) for s in _DEFAULT_POSTPROCESS_STEPS]
        self.foreground_mask_sigma = foreground_mask_sigma
        self.foreground_mask_threshold = foreground_mask_threshold
        self.foreground_mask_postprocess_steps = foreground_mask_postprocess_steps \
            if foreground_mask_postprocess_steps is not None else []

    def model_dump(self) -> dict:
        return {
            "flow_scale":                          self.flow_scale,
            "cellpose_prob_threshold":             self.cellpose_prob_threshold,
            "flow_smoothing_sigma":                self.flow_smoothing_sigma,
            "max_iterations":                      self.max_iterations,
            "uniform_growth_rate":                 self.uniform_growth_rate,
            "flow_mag_scale":                      self.flow_mag_scale,
            "postprocess_steps":                   self.postprocess_steps,
            "foreground_mask_sigma":               self.foreground_mask_sigma,
            "foreground_mask_threshold":           self.foreground_mask_threshold,
            "foreground_mask_postprocess_steps":   self.foreground_mask_postprocess_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FlowWatershedConfig":
        data = dict(data)
        data.pop("method", None)  # old field — ignore

        # Migrate legacy flat postprocess params to pipeline steps
        if "postprocess_steps" not in data:
            steps: list[dict] = []
            opening = data.pop("opening_radius",      1)
            closing = data.pop("closing_radius",      1)
            smooth  = data.pop("boundary_smoothness", 0.5)
            data.pop("fill_holes_threshold", None)
            if opening > 0: steps.append({"type": "open",            "radius":     opening})
            if closing > 0: steps.append({"type": "close",           "radius":     closing})
            if smooth  > 0: steps.append({"type": "smooth_boundary", "smoothness": smooth})
            data["postprocess_steps"] = steps or [dict(s) for s in _DEFAULT_POSTPROCESS_STEPS]
        else:
            # Remove legacy keys if they somehow appear alongside new key
            for k in ("opening_radius", "closing_radius", "boundary_smoothness", "fill_holes_threshold"):
                data.pop(k, None)
            # Strip legacy tissue_mask steps — masking is now the Foreground Mask widget
            data["postprocess_steps"] = [
                s for s in data["postprocess_steps"] if s.get("type") != "tissue_mask"
            ]

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


def _load_tissue_image(root_dir: Path | str, pos: int) -> np.ndarray | None:
    """Load cell_zavg.tif from 0_input/cell/ for tissue masking."""
    try:
        path = stage_dir(root_dir, pos, "raw_import") / "cell" / "cell_zavg.tif"
        if path.exists():
            return tifffile.imread(str(path)).astype(np.float32)
    except Exception:
        pass
    return None


def _load_foreground_mask(root_dir: Path | str, pos: int) -> np.ndarray | None:
    """Load cell_foreground.tif from 4_cell_segmentation/ if it exists."""
    try:
        path = cell_segmentation_dir(root_dir, pos) / "cell_foreground.tif"
        if path.exists():
            return tifffile.imread(str(path)).astype(np.uint8)
    except Exception:
        pass
    return None


def run_foreground_mask_only(
    root_dir: str | Path,
    pos: int,
    sigma: float = 2.0,
    threshold: float = 0.1,
    postprocess_steps: list[dict] | None = None,
) -> Generator:
    """
    Compute and save the foreground mask stack to cell_foreground.tif.
    Applies optional binary mask postprocessing steps after thresholding.
    Yields (done, total, label) tuples for progress reporting.
    Returns path to saved mask file.
    """
    from cellflow.cellpose.processing.flow_watershed_postproc import (
        compute_tissue_foreground_mask,
        run_mask_postprocess_pipeline,
    )

    root_dir = Path(root_dir)
    tissue_full = _load_tissue_image(root_dir, pos)
    if tissue_full is None:
        print("Could not load tissue image (cell_zavg.tif)")
        return None

    T = tissue_full.shape[0] if tissue_full.ndim == 3 else 1
    steps = postprocess_steps or []
    masks = []

    for t in range(T):
        tissue_t = tissue_full[t] if tissue_full.ndim == 3 else tissue_full
        mask = compute_tissue_foreground_mask(tissue_t, sigma=sigma, threshold=threshold)
        if steps:
            mask = run_mask_postprocess_pipeline(mask, steps)
        masks.append(mask.astype(np.uint8))
        yield (t + 1, T, f"t{t:03d}")

    stack = np.stack(masks, axis=0).astype(np.uint8)
    out_dir = cell_segmentation_dir(root_dir, pos)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cell_foreground.tif"
    tifffile.imwrite(str(out_path), stack, compression="zlib", metadata={"axes": "TYX"})
    return str(out_path)


def run_segmentation_only(
    root_dir: str | Path,
    pos: int,
    config: FlowWatershedConfig,
    overwrite: bool = True,
) -> Generator:
    """
    Run flow watershed segmentation only (no postprocessing) for a full stack.
    Yields (done, total, label) tuples for progress reporting.
    Returns path to saved raw labels file.
    """
    from cellflow.cellpose.processing.flow_watershed import flow_guided_watershed

    root_dir = Path(root_dir)
    out_dir = cell_segmentation_dir(root_dir, pos)
    out_path = out_dir / "cell_labels_raw.tif"
    if out_path.exists() and not overwrite:
        print(f"Output exists, skipping (overwrite=False): {out_path}")
        return str(out_path)

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

    foreground_full = _load_foreground_mask(root_dir, pos)

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

            foreground_t = foreground_full[t] if (foreground_full is not None and foreground_full.ndim == 3 and foreground_full.shape[0] == T) else foreground_full

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
                flow_mag_scale=config.flow_mag_scale,
            )

            # Cut off cells that expanded outside the tissue foreground
            if foreground_t is not None:
                cell_labels[~foreground_t.astype(bool)] = 0

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
    overwrite: bool = True,
) -> Generator:
    """
    Run postprocessing on existing raw labels.
    Yields (done, total, label) tuples for progress reporting.
    Returns path to saved final labels file.
    """
    from cellflow.cellpose.processing.flow_watershed_postproc import run_postprocess_pipeline

    root_dir = Path(root_dir)
    out_dir = cell_segmentation_dir(root_dir, pos)
    raw_path = out_dir / "cell_labels_raw.tif"
    out_path = out_dir / "cell_labels.tif"

    if out_path.exists() and not overwrite:
        print(f"Output exists, skipping (overwrite=False): {out_path}")
        return str(out_path)

    if not raw_path.exists():
        print(f"Could not find raw labels at {raw_path}")
        return None

    raw_labels = tifffile.imread(str(raw_path)).astype(np.int32)
    if raw_labels.ndim != 3:
        print(f"Expected (T, H, W), got {raw_labels.shape}")
        return None

    T = raw_labels.shape[0]
    processed_stack = []

    tissue_full = _load_tissue_image(root_dir, pos)
    foreground_full = _load_foreground_mask(root_dir, pos)
    steps = config.postprocess_steps

    for t in range(T):
        try:
            raw_t = raw_labels[t]
            tissue_t = tissue_full[t] if (tissue_full is not None and tissue_full.ndim == 3 and tissue_full.shape[0] == T) else tissue_full
            foreground_t = foreground_full[t] if (foreground_full is not None and foreground_full.ndim == 3 and foreground_full.shape[0] == T) else foreground_full
            processed = run_postprocess_pipeline(raw_t, steps, tissue_image=tissue_t, foreground_mask=foreground_t)
            processed_stack.append(processed)
        except Exception as e:
            print(f"Error at t{t:03d}: {e}")
            processed_stack.append(np.zeros_like(raw_labels[t], dtype=np.int32))

        yield (t + 1, T, f"t{t:03d}")

    stack = np.stack(processed_stack, axis=0).astype(np.int32)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cell_labels.tif"
    tifffile.imwrite(str(out_path), stack, compression="zlib", metadata={"axes": "TYX"})
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
    from cellflow.cellpose.processing.flow_watershed_postproc import run_postprocess_pipeline

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

    tissue_full = _load_tissue_image(root_dir, pos)
    foreground_full = _load_foreground_mask(root_dir, pos)

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

            tissue_t = tissue_full[t] if (tissue_full is not None and tissue_full.ndim == 3 and tissue_full.shape[0] == T) else tissue_full
            foreground_t = foreground_full[t] if (foreground_full is not None and foreground_full.ndim == 3 and foreground_full.shape[0] == T) else foreground_full

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
                flow_mag_scale=config.flow_mag_scale,
            )

            # Cut off cells that expanded outside the tissue foreground
            if foreground_t is not None:
                cell_labels[~foreground_t.astype(bool)] = 0

            # Apply post-processing
            cell_labels = run_postprocess_pipeline(
                cell_labels,
                config.postprocess_steps,
                tissue_image=tissue_t,
                foreground_mask=foreground_t,
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
                prob_t = prob_full[t] if (prob_full is not None and prob_full.ndim == 3) else prob_full
                fg_t = foreground_full[t] if (foreground_full is not None and foreground_full.ndim == 3 and foreground_full.shape[0] == T) else foreground_full

                cell_labels_raw = flow_guided_watershed(
                    nuc_t,
                    flow_t,
                    cellpose_prob=prob_t,
                    flow_scale=config.flow_scale,
                    cellpose_prob_threshold=config.cellpose_prob_threshold,
                    flow_smoothing_sigma=config.flow_smoothing_sigma,
                    max_iterations=config.max_iterations,
                    uniform_growth_rate=config.uniform_growth_rate,
                    flow_mag_scale=config.flow_mag_scale,
                )
                if fg_t is not None:
                    cell_labels_raw[~fg_t.astype(bool)] = 0
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
    """Widget for flow-guided watershed cell segmentation."""

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

        self._files_widget = PipelineFilesWidget([
            ("Input", [
                ("3_correction/nuclear_labels_corrected.tif", "Corrected labels"),
                ("4_cell_segmentation/cell_foreground.tif",   "Foreground mask"),
            ]),
            ("Output", [
                ("4_cell_segmentation/cell_labels_raw.tif", "Cell labels raw"),
                ("4_cell_segmentation/cell_labels.tif",     "Cell labels"),
            ]),
        ])
        lay.addWidget(self._files_widget)

        # ── Load Foreground ──────────────────────────────────────────────
        self._load_fg_btn = QPushButton("Load Foreground from Active Layer")
        self._load_fg_btn.clicked.connect(self._on_load_foreground)
        lay.addWidget(self._load_fg_btn)
        self._load_fg_status = QLabel("")
        lay.addWidget(self._load_fg_status)

        # ── Segmentation controls ────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Preview frame"))
        self._seg_frame_spin = QSpinBox()
        self._seg_frame_spin.setRange(0, 1000)
        self._seg_frame_spin.setValue(0)
        row.addWidget(self._seg_frame_spin)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Cellprob threshold:"))
        self._seg_prob_threshold_spin = QDoubleSpinBox()
        self._seg_prob_threshold_spin.setRange(-100.0, 100.0)
        self._seg_prob_threshold_spin.setSingleStep(1.0)
        self._seg_prob_threshold_spin.setDecimals(1)
        self._seg_prob_threshold_spin.setValue(0.0)
        row.addWidget(self._seg_prob_threshold_spin)
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

        row = QHBoxLayout()
        row.addWidget(QLabel("Flow magnitude scale"))
        self._seg_flow_mag_scale_spin = QDoubleSpinBox()
        self._seg_flow_mag_scale_spin.setRange(0.0, 20.0)
        self._seg_flow_mag_scale_spin.setSingleStep(0.5)
        self._seg_flow_mag_scale_spin.setDecimals(1)
        self._seg_flow_mag_scale_spin.setValue(3.0)
        row.addWidget(self._seg_flow_mag_scale_spin)
        row.addStretch()
        lay.addLayout(row)

        self._seg_preview_btn = QPushButton("Preview")
        self._seg_preview_btn.clicked.connect(self._seg_on_preview)
        lay.addWidget(self._seg_preview_btn)

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

        # ── Save Corrected Cell Labels ───────────────────────────────────
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

    # ── Project-derived path helpers ─────────────────────────────────────

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

    # ════════════════════════════════════════════════════════════════════
    # Config helpers
    # ════════════════════════════════════════════════════════════════════

    def _build_config(self) -> FlowWatershedConfig:
        return FlowWatershedConfig(
            cellpose_prob_threshold=self._seg_prob_threshold_spin.value(),
            max_iterations=self._seg_max_iter_spin.value(),
            uniform_growth_rate=self._seg_uniform_growth_spin.value(),
            flow_mag_scale=self._seg_flow_mag_scale_spin.value(),
        )

    def _apply_config(self, cfg: FlowWatershedConfig) -> None:
        self._seg_prob_threshold_spin.setValue(cfg.cellpose_prob_threshold)
        self._seg_max_iter_spin.setValue(cfg.max_iterations)
        self._seg_uniform_growth_spin.setValue(cfg.uniform_growth_rate)
        self._seg_flow_mag_scale_spin.setValue(cfg.flow_mag_scale)

    # ════════════════════════════════════════════════════════════════════
    # Load Foreground callback
    # ════════════════════════════════════════════════════════════════════

    def _on_load_foreground(self) -> None:
        root_dir = self._get_root_dir()
        if not root_dir:
            self._load_fg_status.setText("No project open.")
            return

        active = self.viewer.layers.selection.active
        if active is None or not hasattr(active, "data"):
            self._load_fg_status.setText("No active layer selected.")
            return

        pos = int(self._state.current_position)
        data = np.asarray(active.data).astype(np.uint8)

        out_dir = cell_segmentation_dir(root_dir, pos)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cell_foreground.tif"

        try:
            axes = "TYX" if data.ndim == 3 else "YX"
            tifffile.imwrite(str(out_path), data, compression="zlib", metadata={"axes": axes})
            self._load_fg_status.setText(f"Saved foreground mask to {out_path.name}")
            self._sync_project_dir()
        except Exception as e:
            self._load_fg_status.setText(f"Error: {e}")

    # ════════════════════════════════════════════════════════════════════
    # Save Corrected Cell Labels callback
    # ════════════════════════════════════════════════════════════════════

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

    # ════════════════════════════════════════════════════════════════════
    # Segmentation callbacks
    # ════════════════════════════════════════════════════════════════════

    def _seg_on_preview(self) -> None:
        root_dir = self._get_root_dir()
        if not root_dir:
            self._seg_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._state.current_position)
        frame = int(self._seg_frame_spin.value())
        cfg = self._build_config()

        self._seg_status.setText(f"Processing frame {frame}…")

        try:
            from cellflow.cellpose.processing.flow_watershed import flow_guided_watershed
            root_dir_path = Path(root_dir)

            nuclear_labels = _load_nuclear_labels(root_dir_path, pos)
            if nuclear_labels is None:
                self._seg_status.setText("Could not load nuclear labels.")
                return

            flow_full, prob_full = _load_cellpose_data(root_dir_path, pos, frame)
            if flow_full is None:
                self._seg_status.setText("Could not load cellpose flow.")
                return

            nuc_t = nuclear_labels[frame]
            T = nuclear_labels.shape[0]

            flow_t = flow_full[frame] if (flow_full.ndim == 4 and flow_full.shape[0] == T) else flow_full
            prob_t = prob_full[frame] if (prob_full is not None and prob_full.ndim == 3 and prob_full.shape[0] == T) else prob_full

            cell_labels = flow_guided_watershed(
                nuc_t,
                flow_t,
                cellpose_prob=prob_t,
                flow_scale=cfg.flow_scale,
                cellpose_prob_threshold=cfg.cellpose_prob_threshold,
                flow_smoothing_sigma=cfg.flow_smoothing_sigma,
                max_iterations=cfg.max_iterations,
                uniform_growth_rate=cfg.uniform_growth_rate,
                flow_mag_scale=cfg.flow_mag_scale,
            )

            foreground_full = _load_foreground_mask(root_dir_path, pos)
            foreground_t = foreground_full[frame] if (foreground_full is not None and foreground_full.ndim == 3 and foreground_full.shape[0] == T) else foreground_full
            if foreground_t is not None:
                cell_labels[~foreground_t.astype(bool)] = 0

            flow_mag = np.sqrt(flow_t[..., 0]**2 + flow_t[..., 1]**2)

            while len(self.viewer.layers) > 0:
                self.viewer.layers.pop()

            self.viewer.add_image(nuc_t, name="Nuclear Labels")
            self.viewer.add_image(prob_t, name="Cellpose Probability")
            self.viewer.add_image(flow_mag, name="Cellpose Flow Magnitude")
            if foreground_t is not None:
                self.viewer.add_labels(foreground_t.astype(np.uint8), name="Foreground Mask")
            self.viewer.add_labels(cell_labels, name="Cell Segmentation (raw)")

            n_cells = len(np.unique(cell_labels)) - 1
            fg_note = " (foreground mask applied)" if foreground_t is not None else " (no foreground mask)"
            self._seg_status.setText(f"Preview complete. {n_cells} cells found{fg_note}.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._seg_status.setText(f"Preview error: {e}")

    def _seg_on_run(self) -> None:
        root_dir = self._get_root_dir()
        if not root_dir:
            self._seg_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._state.current_position)
        cfg = self._build_config()
        overwrite = self._seg_overwrite_chk.isChecked()

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
            from cellflow.core.logging import StageLogger
            from cellflow.core.paths import log_path
            with StageLogger(log_path(root_dir, pos), "cell_segmentation"):
                for update in run_segmentation_only(root_dir, pos, cfg, overwrite=overwrite):
                    yield update

        self.run_started.emit()
        self._seg_worker = _work()
        self._seg_worker.aborted.connect(self._seg_on_cancelled)

    def _seg_on_run_terminal(self) -> None:
        root_dir = self._get_root_dir()
        if not root_dir:
            self._seg_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._state.current_position)
        cfg = self._build_config()
        overwrite_flag = "--overwrite" if self._seg_overwrite_chk.isChecked() else ""

        cfg_path = Path(tempfile.mktemp(suffix="_fw_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))

        cmd = (
            f"\"{sys.executable}\" -m cellflow.cellpose.stages.flow_watershed"
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

    def _seg_on_load_results(self) -> None:
        root_dir = self._get_root_dir()
        if not root_dir:
            self._seg_status.setText("No project open. Create or open a project first.")
            return

        pos = int(self._state.current_position)
        out_dir = cell_segmentation_dir(root_dir, pos)
        raw_path = out_dir / "cell_labels_raw.tif"

        if not raw_path.exists():
            self._seg_status.setText("No cell_labels_raw.tif found.")
            return

        cell_stack = tifffile.imread(str(raw_path)).astype(np.uint32)
        nuc_labels = _load_nuclear_labels(root_dir, pos)

        while len(self.viewer.layers) > 0:
            self.viewer.layers.pop()

        if nuc_labels is not None:
            self.viewer.add_labels(nuc_labels.astype(np.uint32), name="nuclei", opacity=0.3)
        self.viewer.add_labels(cell_stack, name="cells (raw)")

        self._seg_status.setText(f"Loaded raw segmentation, shape={cell_stack.shape}")

    # ════════════════════════════════════════════════════════════════════
    # get_params / set_params
    # ════════════════════════════════════════════════════════════════════

    def get_params(self) -> dict:
        return self._build_config().model_dump()

    def set_params(self, data: dict) -> None:
        cfg_data = {k: v for k, v in data.items() if k not in ("seg_overwrite", "pp_overwrite")}
        self._apply_config(FlowWatershedConfig.from_dict(cfg_data))
        if "seg_overwrite" in data:
            self._seg_overwrite_chk.setChecked(bool(data["seg_overwrite"]))
