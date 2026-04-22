"""Cluster Cellpose panels for nucleus and cell inputs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

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
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget

from napari.qt.threading import thread_worker

from cellflow.core.paths import stage_dir
from cellflow.napari.log_viewer import StageLogViewer
from cellflow.napari.registry import get_state

try:
    from cellflow.cellpose.config import CellposeConfig
    from cellflow.napari.runners.terminal import launch_in_terminal
    from cellflow.cellpose.stages.nucleus_3d import (
        discover_input_files,
        run as run_s01a,
    )
    from cellflow.cellpose.stages.cell_2d import run as run_s01b
    _CELLPOSE_PIPELINE_AVAILABLE = True
except ImportError:
    _CELLPOSE_PIPELINE_AVAILABLE = False


class CellposeWidget(QWidget):
    """Widget for the cluster-side Cellpose step."""

    run_started = Signal()

    def __init__(self, viewer: "napari.Viewer", *, log_viewer=None) -> None:
        super().__init__()
        self.viewer = viewer
        self._state = get_state(viewer)
        self._worker_s01a = None
        self._worker_s01a_preview = None
        self._worker_s01b = None
        self._worker_s01b_preview = None
        self._s01b_run_terminal_btn = None

        if not _CELLPOSE_PIPELINE_AVAILABLE:
            layout = QVBoxLayout(self)
            msg = QLabel(
                "Cellpose pipeline package is not installed.\n"
                "Install with: pip install cellflow-napari[pipeline]"
            )
            msg.setWordWrap(True)
            layout.addWidget(msg)
            layout.addStretch()
            return

        inner_layout = QVBoxLayout(self)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setAlignment(Qt.AlignTop)

        # Two collapsible sub-sections for s01a and s01b
        self._s01a_widget = self._create_s01a_widget()
        self._s01b_widget = self._create_s01b_widget()
        self._s01a_section = CollapsibleSection("Cluster Cellpose", self._s01a_widget, expanded=False)
        self._s01b_section = CollapsibleSection("Cluster Cellpose", self._s01b_widget, expanded=False)
        inner_layout.addWidget(self._s01a_section)
        inner_layout.addWidget(self._s01b_section)

        if log_viewer is not None:
            self._log_viewer = log_viewer
        else:
            self._log_viewer = StageLogViewer(self._state)
            inner_layout.addWidget(self._log_viewer)

        # Connect project-change and position-change signals
        self._state.pipeline_schema_changed.connect(self._sync_project_dir)
        self._state.position_changed.connect(self._sync_project_dir)
        self._sync_project_dir()

    # ── Project-derived path helpers ─────────────────────────────────────

    def _get_project_dir(self) -> Path | None:
        return self._state.project_dir

    def _s01a_input_dir(self, pos: int) -> Path | None:
        root = self._get_project_dir()
        if root is None:
            return None
        return stage_dir(root, pos, "raw_import")

    def _s01a_output_dir(self, pos: int) -> Path | None:
        root = self._get_project_dir()
        if root is None:
            return None
        return stage_dir(root, pos, "cellpose_cluster")

    def _s01b_root_dir(self) -> Path | None:
        return self._get_project_dir()

    def _sync_project_dir(self) -> None:
        """Refresh file-status rows when the project or position changes."""
        root = self._get_project_dir()
        pos = self._state.current_position

        if root is None:
            self._s01a_files_widget.refresh(None)
            self._s01b_files_widget.refresh(None)
            return

        pos_dir = Path(root) / f"pos{pos:02d}"
        self._s01a_files_widget.refresh(pos_dir)
        self._s01b_files_widget.refresh(pos_dir)

    def _create_s01a_widget(self) -> QWidget:
        """Create the s01a (nucleus) panel."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # ── File status (derived from project) ──────────────────────────
        self._s01a_files_widget = PipelineFilesWidget([
            ("Input", [
                ("0_input/nucleus_4d.tif", "Nucleus 4D stack"),
                ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
            ]),
            ("Output", [
                ("1_cellpose/nucleus_dp.tif",   "Nucleus DP"),
                ("1_cellpose/nucleus_prob.tif", "Nucleus prob"),
                ("1_cellpose/nucleus_dp_zavg.tif",   "Nucleus DP z-avg"),
                ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
            ]),
        ])
        layout.addWidget(self._s01a_files_widget)

        # ── Model parameters ─────────────────────────────────────────────
        model_group = QGroupBox("Model parameters")
        mg_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Model"))
        self._s01a_model_combo = QComboBox()
        self._s01a_model_combo.addItems(["nuclei", "cpsam", "cyto3", "cyto2", "cyto"])
        self._s01a_model_combo.setEditable(True)
        row.addWidget(self._s01a_model_combo)
        mg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Diameter (px):"))
        self._s01a_diameter_spin = QDoubleSpinBox()
        self._s01a_diameter_spin.setRange(0.0, 500.0)
        self._s01a_diameter_spin.setSingleStep(1.0)
        self._s01a_diameter_spin.setDecimals(1)
        self._s01a_diameter_spin.setValue(17.0)
        row.addWidget(self._s01a_diameter_spin)
        mg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Anisotropy (Z/XY voxel ratio)"))
        self._s01a_anisotropy_spin = QDoubleSpinBox()
        self._s01a_anisotropy_spin.setRange(0.1, 20.0)
        self._s01a_anisotropy_spin.setSingleStep(0.1)
        self._s01a_anisotropy_spin.setDecimals(2)
        self._s01a_anisotropy_spin.setValue(1.0)
        row.addWidget(self._s01a_anisotropy_spin)
        mg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min size (voxels)"))
        self._s01a_min_size_spin = QSpinBox()
        self._s01a_min_size_spin.setRange(0, 10_000_000)
        self._s01a_min_size_spin.setSingleStep(100)
        self._s01a_min_size_spin.setValue(500)
        row.addWidget(self._s01a_min_size_spin)
        mg_layout.addLayout(row)

        self._s01a_use_gpu_check = QCheckBox("Use GPU")
        self._s01a_use_gpu_check.setChecked(True)
        mg_layout.addWidget(self._s01a_use_gpu_check)

        model_group.setLayout(mg_layout)
        layout.addWidget(model_group)

        # ── Preprocessing ────────────────────────────────────────────────
        preproc_group = QGroupBox("Preprocessing")
        pp_layout = QVBoxLayout()

        row = QHBoxLayout()
        self._s01a_gamma_check = QCheckBox("Gamma correction")
        self._s01a_gamma_check.toggled.connect(self._on_s01a_gamma_toggled)
        row.addWidget(self._s01a_gamma_check)
        self._s01a_gamma_spin = QDoubleSpinBox()
        self._s01a_gamma_spin.setRange(0.1, 5.0)
        self._s01a_gamma_spin.setSingleStep(0.1)
        self._s01a_gamma_spin.setDecimals(2)
        self._s01a_gamma_spin.setValue(1.0)
        self._s01a_gamma_spin.setEnabled(False)
        row.addWidget(self._s01a_gamma_spin)
        pp_layout.addLayout(row)

        preproc_group.setLayout(pp_layout)
        layout.addWidget(preproc_group)

        # ── Overwrite ────────────────────────────────────────────────────
        self._s01a_overwrite_check = QCheckBox("Overwrite existing files")
        self._s01a_overwrite_check.setStyleSheet("color: white;")
        layout.addWidget(self._s01a_overwrite_check)

        # ── Run buttons ──────────────────────────────────────────────────
        row = QHBoxLayout()
        self._s01a_run_btn = QPushButton("Run Cellpose")
        self._s01a_run_btn.clicked.connect(self._on_s01a_run)
        row.addWidget(self._s01a_run_btn)
        self._s01a_run_terminal_btn = QPushButton("Run in Terminal")
        self._s01a_run_terminal_btn.clicked.connect(self._on_s01a_run_terminal)
        row.addWidget(self._s01a_run_terminal_btn)
        layout.addLayout(row)

        # ── Frame selector + action buttons ─────────────────────────────
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        self._s01a_frame_spin = QSpinBox()
        self._s01a_frame_spin.setRange(0, 9999)
        self._s01a_frame_spin.setValue(0)
        frame_row.addWidget(self._s01a_frame_spin)
        frame_row.addStretch()
        layout.addLayout(frame_row)

        action_row = QHBoxLayout()
        self._s01a_preview_btn = QPushButton("Preview")
        self._s01a_preview_btn.clicked.connect(self._on_s01a_preview)
        action_row.addWidget(self._s01a_preview_btn)
        self._s01a_load_results_btn = QPushButton("Load Results")
        self._s01a_load_results_btn.clicked.connect(self._on_s01a_load_results)
        action_row.addWidget(self._s01a_load_results_btn)
        layout.addLayout(action_row)

        # ── Progress ─────────────────────────────────────────────────────
        self._s01a_progress = QProgressBar()
        self._s01a_progress.setVisible(False)
        layout.addWidget(self._s01a_progress)

        self._s01a_status_label = QLabel("")
        layout.addWidget(self._s01a_status_label)

        widget.setLayout(layout)
        return widget

    def _create_s01b_widget(self) -> QWidget:
        """Create the s01b (cell) panel."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)

        # ── File status (derived from project) ──────────────────────────
        self._s01b_files_widget = PipelineFilesWidget([
            ("Input", [
                ("0_input/cell_4d.tif",  "Cell 4D stack"),
            ]),
            ("Output", [
                ("1_cellpose/cell_dp.tif",          "Cell DP (z-slices)"),
                ("1_cellpose/cell_prob.tif",        "Cell prob (z-slices)"),
                ("1_cellpose/cell_dp_zavg.tif",     "Cell DP avg"),
                ("1_cellpose/cell_prob_zavg.tif",   "Cell prob avg"),
            ]),
        ])
        layout.addWidget(self._s01b_files_widget)

        # ── Model parameters ─────────────────────────────────────────────
        model_group = QGroupBox("Model parameters")
        mg_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Model"))
        self._s01b_model_combo = QComboBox()
        self._s01b_model_combo.addItems(["cpsam", "cyto3", "cyto2", "cyto", "nuclei"])
        self._s01b_model_combo.setEditable(True)
        self._s01b_model_combo.setCurrentText("cpsam")
        row.addWidget(self._s01b_model_combo)
        mg_layout.addLayout(row)

        row = QHBoxLayout()
        diam_label = QLabel("Diameter (px):")
        diam_label.setToolTip(
            "Expected cell diameter in pixels. 0 = no rescaling (recommended for cpsam).\n\n"
            "With cpsam, any non-zero value rescales the image by 30/diameter before\n"
            "inference. cpsam is scale-invariant and works best on the raw image — leave\n"
            "this at 0. The diameter field is a Cellpose 2/3 legacy parameter."
        )
        row.addWidget(diam_label)
        self._s01b_diameter_spin = QDoubleSpinBox()
        self._s01b_diameter_spin.setRange(0.0, 500.0)
        self._s01b_diameter_spin.setSingleStep(1.0)
        self._s01b_diameter_spin.setDecimals(1)
        self._s01b_diameter_spin.setValue(0.0)
        self._s01b_diameter_spin.setSpecialValueText("Auto")
        self._s01b_diameter_spin.setToolTip(diam_label.toolTip())
        row.addWidget(self._s01b_diameter_spin)
        mg_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min size (voxels)"))
        self._s01b_min_size_spin = QSpinBox()
        self._s01b_min_size_spin.setRange(0, 10_000_000)
        self._s01b_min_size_spin.setSingleStep(100)
        self._s01b_min_size_spin.setValue(200)
        row.addWidget(self._s01b_min_size_spin)
        mg_layout.addLayout(row)

        self._s01b_use_gpu_check = QCheckBox("Use GPU")
        self._s01b_use_gpu_check.setChecked(True)
        mg_layout.addWidget(self._s01b_use_gpu_check)

        model_group.setLayout(mg_layout)
        layout.addWidget(model_group)

        # ── Preprocessing ────────────────────────────────────────────────
        preproc_group = QGroupBox("Preprocessing")
        pp_layout = QVBoxLayout()

        row = QHBoxLayout()
        self._s01b_gamma_check = QCheckBox("Gamma correction")
        self._s01b_gamma_check.toggled.connect(self._on_s01b_gamma_toggled)
        row.addWidget(self._s01b_gamma_check)
        self._s01b_gamma_spin = QDoubleSpinBox()
        self._s01b_gamma_spin.setRange(0.1, 5.0)
        self._s01b_gamma_spin.setSingleStep(0.1)
        self._s01b_gamma_spin.setDecimals(2)
        self._s01b_gamma_spin.setValue(1.0)
        self._s01b_gamma_spin.setEnabled(False)
        row.addWidget(self._s01b_gamma_spin)
        pp_layout.addLayout(row)

        preproc_group.setLayout(pp_layout)
        layout.addWidget(preproc_group)

        # ── Overwrite ────────────────────────────────────────────────────
        self._s01b_overwrite_check = QCheckBox("Overwrite existing files")
        self._s01b_overwrite_check.setStyleSheet("color: white;")
        layout.addWidget(self._s01b_overwrite_check)

        # ── Run buttons ──────────────────────────────────────────────────
        row = QHBoxLayout()
        self._s01b_run_btn = QPushButton("Run Cellpose")
        self._s01b_run_btn.clicked.connect(self._on_s01b_run)
        row.addWidget(self._s01b_run_btn)
        self._s01b_run_terminal_btn = QPushButton("Run in Terminal")
        self._s01b_run_terminal_btn.clicked.connect(self._on_s01b_run_terminal)
        row.addWidget(self._s01b_run_terminal_btn)
        layout.addLayout(row)

        # ── Frame selector + action buttons ─────────────────────────────
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame:"))
        self._s01b_frame_spin = QSpinBox()
        self._s01b_frame_spin.setRange(0, 9999)
        self._s01b_frame_spin.setValue(0)
        frame_row.addWidget(self._s01b_frame_spin)
        frame_row.addStretch()
        layout.addLayout(frame_row)

        action_row = QHBoxLayout()
        self._s01b_preview_btn = QPushButton("Preview")
        self._s01b_preview_btn.clicked.connect(self._on_s01b_preview)
        action_row.addWidget(self._s01b_preview_btn)
        self._s01b_load_results_btn = QPushButton("Load Results")
        self._s01b_load_results_btn.clicked.connect(self._on_s01b_load_results)
        action_row.addWidget(self._s01b_load_results_btn)
        layout.addLayout(action_row)

        # ── Progress ─────────────────────────────────────────────────────
        self._s01b_progress = QProgressBar()
        self._s01b_progress.setVisible(False)
        layout.addWidget(self._s01b_progress)

        self._s01b_status_label = QLabel("")
        layout.addWidget(self._s01b_status_label)

        widget.setLayout(layout)
        return widget

    # ── Helpers ──────────────────────────────────────────────────────────

    def _on_s01a_gamma_toggled(self, checked: bool) -> None:
        self._s01a_gamma_spin.setEnabled(checked)

    def _on_s01b_gamma_toggled(self, checked: bool) -> None:
        self._s01b_gamma_spin.setEnabled(checked)

    def _build_s01a_config(self) -> CellposeConfig:
        gamma = self._s01a_gamma_spin.value() if self._s01a_gamma_check.isChecked() else None
        return CellposeConfig(
            model=self._s01a_model_combo.currentText(),
            diameter=self._s01a_diameter_spin.value(),
            anisotropy=self._s01a_anisotropy_spin.value(),
            min_size=self._s01a_min_size_spin.value(),
            use_gpu=self._s01a_use_gpu_check.isChecked(),
            gamma=gamma,
        )

    def _build_s01b_config(self) -> CellposeConfig:
        gamma = self._s01b_gamma_spin.value() if self._s01b_gamma_check.isChecked() else None
        return CellposeConfig(
            model=self._s01b_model_combo.currentText(),
            diameter=self._s01b_diameter_spin.value(),
            anisotropy=0.0,  # Not used for 2D
            min_size=self._s01b_min_size_spin.value(),
            use_gpu=self._s01b_use_gpu_check.isChecked(),
            gamma=gamma,
        )

    def _apply_s01a_config(self, cfg: CellposeConfig) -> None:
        self._s01a_model_combo.setCurrentText(cfg.model)
        self._s01a_diameter_spin.setValue(cfg.diameter)
        self._s01a_anisotropy_spin.setValue(cfg.anisotropy)
        self._s01a_min_size_spin.setValue(cfg.min_size)
        self._s01a_use_gpu_check.setChecked(cfg.use_gpu)
        if cfg.gamma is not None:
            self._s01a_gamma_check.setChecked(True)
            self._s01a_gamma_spin.setValue(cfg.gamma)
        else:
            self._s01a_gamma_check.setChecked(False)

    def _apply_s01b_config(self, cfg: CellposeConfig) -> None:
        self._s01b_model_combo.setCurrentText(cfg.model)
        self._s01b_diameter_spin.setValue(cfg.diameter)
        self._s01b_min_size_spin.setValue(cfg.min_size)
        self._s01b_use_gpu_check.setChecked(cfg.use_gpu)
        if cfg.gamma is not None:
            self._s01b_gamma_check.setChecked(True)
            self._s01b_gamma_spin.setValue(cfg.gamma)
        else:
            self._s01b_gamma_check.setChecked(False)

    # ── s01a Run inline ──────────────────────────────────────────────────

    def _on_s01a_run(self) -> None:
        pos = self._state.current_position
        input_dir = self._s01a_input_dir(pos)
        output_dir = self._s01a_output_dir(pos)
        if input_dir is None or output_dir is None:
            self._s01a_status_label.setText("No project open. Create or open a project first.")
            return
        cfg = self._build_s01a_config()
        overwrite = self._s01a_overwrite_check.isChecked()
        self._s01a_run_btn.setEnabled(False)
        self._s01a_run_terminal_btn.setEnabled(False)
        self._s01a_progress.setVisible(True)
        self._s01a_progress.setValue(0)
        self._s01a_status_label.setText("Starting…")

        input_dir_str = str(input_dir)
        output_dir_str = str(output_dir)
        log_file = output_dir.parent.parent / "pipeline.log"

        @thread_worker(
            connect={
                "yielded": self._on_s01a_progress,
                "finished": self._on_s01a_finished,
                "errored": self._on_s01a_error,
            }
        )
        def _work():
            from cellflow.core.logging import StageLogger
            with StageLogger(log_file, "cellpose_nucleus"):
                for update in run_s01a(input_dir_str, output_dir_str, cfg, overwrite=overwrite):
                    yield update

        self.run_started.emit()
        self._worker_s01a = _work()

    # ── s01b Run inline ──────────────────────────────────────────────────

    def _on_s01b_run(self) -> None:
        root_dir = self._s01b_root_dir()
        if root_dir is None:
            self._s01b_status_label.setText("No project open. Create or open a project first.")
            return
        pos = self._state.current_position
        cfg = self._build_s01b_config()
        overwrite = self._s01b_overwrite_check.isChecked()
        self._s01b_run_btn.setEnabled(False)
        if self._s01b_run_terminal_btn:
            self._s01b_run_terminal_btn.setEnabled(False)
        self._s01b_progress.setVisible(True)
        self._s01b_progress.setValue(0)
        self._s01b_status_label.setText("Starting…")

        root_dir_str = str(root_dir)

        @thread_worker(
            connect={
                "yielded": self._on_s01b_progress,
                "finished": self._on_s01b_finished,
                "errored": self._on_s01b_error,
            }
        )
        def _work():
            from cellflow.core.logging import StageLogger
            from cellflow.core.paths import log_path
            with StageLogger(log_path(root_dir_str, pos), "cellpose_cell"):
                for update in run_s01b(root_dir_str, pos, cfg, overwrite=overwrite):
                    yield update

        self.run_started.emit()
        self._worker_s01b = _work()

    # ── s01a Run in terminal ─────────────────────────────────────────────

    def _on_s01a_run_terminal(self) -> None:
        pos = self._state.current_position
        input_dir = self._s01a_input_dir(pos)
        output_dir = self._s01a_output_dir(pos)
        if input_dir is None or output_dir is None:
            self._s01a_status_label.setText("No project open. Create or open a project first.")
            return
        cfg = self._build_s01a_config()
        cfg_path = Path(tempfile.mktemp(suffix="_cp_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))
        overwrite_flag = "--overwrite" if self._s01a_overwrite_check.isChecked() else ""
        cmd = (
            f"python -m cellflow.cellpose.stages.nucleus_3d"
            f" --input-dir \"{input_dir}\""
            f" --output-dir \"{output_dir}\""
            f" --config \"{cfg_path}\""
            f" {overwrite_flag}"
        ).strip()
        try:
            launch_in_terminal(cmd)
            self._s01a_status_label.setText("Launched Cellpose stage in terminal.")
        except Exception as e:
            self._s01a_status_label.setText(f"Terminal launch error: {e}")

    # ── s01b Run in terminal ─────────────────────────────────────────────

    def _on_s01b_run_terminal(self) -> None:
        root_dir = self._s01b_root_dir()
        if root_dir is None:
            self._s01b_status_label.setText("No project open. Create or open a project first.")
            return
        pos = self._state.current_position
        cfg = self._build_s01b_config()
        cfg_path = Path(tempfile.mktemp(suffix="_cp_config.json"))
        cfg_path.write_text(json.dumps(cfg.model_dump(), indent=2))
        overwrite_flag = "--overwrite" if self._s01b_overwrite_check.isChecked() else ""
        cmd = (
            f"python -m cellflow.cellpose.stages.cell_2d"
            f" --root-dir \"{root_dir}\""
            f" --pos {pos}"
            f" --config \"{cfg_path}\""
            f" {overwrite_flag}"
        ).strip()
        try:
            launch_in_terminal(cmd)
            self._s01b_status_label.setText("Launched Cellpose cell stage in terminal.")
        except Exception as e:
            self._s01b_status_label.setText(f"Terminal launch error: {e}")

    # ── s01a Progress / finished / error callbacks ────────────────────────

    def _on_s01a_progress(self, update: tuple) -> None:
        done, total, label = update
        self._s01a_progress.setMaximum(max(total, 1))
        self._s01a_progress.setValue(done)
        self._s01a_status_label.setText(f"Processing {label} [{done}/{total}]")

    def _on_s01a_finished(self) -> None:
        self._s01a_run_btn.setEnabled(True)
        self._s01a_run_terminal_btn.setEnabled(True)
        self._s01a_progress.setVisible(False)
        self._s01a_status_label.setText("Done — Cellpose outputs written.")
        self._worker_s01a = None
        self._load_s01a_prob_stack()
        self._log_viewer.refresh()

    def _on_s01a_error(self, exc: Exception) -> None:
        self._s01a_run_btn.setEnabled(True)
        self._s01a_run_terminal_btn.setEnabled(True)
        self._s01a_progress.setVisible(False)
        self._s01a_status_label.setText(f"Error: {exc}")
        self._worker_s01a = None
        self._log_viewer.refresh()

    # ── s01b Progress / finished / error callbacks ────────────────────────

    def _on_s01b_progress(self, update: tuple) -> None:
        done, total, label = update
        self._s01b_progress.setMaximum(max(total, 1))
        self._s01b_progress.setValue(done)
        label_str = f"{label:03d}" if isinstance(label, int) else label
        self._s01b_status_label.setText(f"Processing t{label_str} [{done}/{total}]")

    def _on_s01b_finished(self) -> None:
        self._s01b_run_btn.setEnabled(True)
        if self._s01b_run_terminal_btn:
            self._s01b_run_terminal_btn.setEnabled(True)
        self._s01b_progress.setVisible(False)
        self._s01b_status_label.setText("Done — Cellpose outputs written.")
        self._worker_s01b = None
        self._load_s01b_results()
        self._log_viewer.refresh()

    def _on_s01b_error(self, exc: Exception) -> None:
        self._s01b_run_btn.setEnabled(True)
        if self._s01b_run_terminal_btn:
            self._s01b_run_terminal_btn.setEnabled(True)
        self._s01b_progress.setVisible(False)
        self._s01b_status_label.setText(f"Error: {exc}")
        self._worker_s01b = None
        self._log_viewer.refresh()

    # ── s01a Load results ────────────────────────────────────────────────

    def _load_s01a_prob_stack(self) -> None:
        pos = self._state.current_position
        output_dir = self._s01a_output_dir(pos)
        if output_dir is None:
            return
        prob_file = output_dir / "nucleus_prob.tif"
        if not prob_file.exists():
            return
        prob = tifffile.imread(str(prob_file))
        self.viewer.add_image(prob, name="nucleus_cellprob", colormap="inferno")
        self._s01a_status_label.setText(
            f"Loaded {prob_file.name}  shape={prob.shape}"
        )

    def _on_s01a_load_results(self) -> None:
        pos = self._state.current_position
        input_dir = self._s01a_input_dir(pos)
        output_dir = self._s01a_output_dir(pos)
        if input_dir is None:
            self._s01a_status_label.setText("No project open.")
            return

        loaded = []

        nuc_input_file = input_dir / "nucleus_4d.tif"
        if nuc_input_file.exists():
            data = tifffile.imread(str(nuc_input_file))
            self.viewer.add_image(data, name=f"nucleus_input_pos{pos:02d}", colormap="gray")
            loaded.append("input")

        if output_dir is not None and output_dir.is_dir():
            prob_file = output_dir / "nucleus_prob.tif"
            if prob_file.exists():
                prob = tifffile.imread(str(prob_file))
                self.viewer.add_image(prob, name=f"nucleus_prob_pos{pos:02d}", colormap="inferno")
                loaded.append("prob")

            dp_file = output_dir / "nucleus_dp.tif"
            if dp_file.exists():
                dp = tifffile.imread(str(dp_file))
                # dp: (T, 3, Z, Y, X) — split vector components
                for c in range(dp.shape[1]):
                    self.viewer.add_image(dp[:, c], name=f"nucleus_dp_{c}_pos{pos:02d}", colormap="RdBu")
                loaded.append("dp")

        if loaded:
            self._s01a_status_label.setText(f"Loaded: {', '.join(loaded)}")
        else:
            self._s01a_status_label.setText("No files found.")

    # ── s01b Load results ────────────────────────────────────────────────

    def _load_s01b_results(self) -> None:
        root_dir = self._s01b_root_dir()
        if root_dir is None:
            return
        pos = self._state.current_position
        cell_dir = stage_dir(root_dir, pos, "cellpose_cell")
        # Prefer z-averaged prob for quick status load
        prob_file = cell_dir / "cell_prob_zavg.tif"
        if not prob_file.exists():
            prob_file = cell_dir / "cell_prob.tif"
        if not prob_file.exists():
            return
        prob = tifffile.imread(str(prob_file))
        self.viewer.add_image(prob, name=f"cell_prob_pos{pos:02d}", colormap="inferno")
        self._s01b_status_label.setText(f"Loaded {prob_file.name}  shape={prob.shape}")

    def _on_s01b_load_results(self) -> None:
        from cellflow.cellpose.stages.raw_import import nucleus_zavg_path, cell_zavg_path

        root_dir = self._s01b_root_dir()
        if root_dir is None:
            self._s01b_status_label.setText("No project open.")
            return
        pos = self._state.current_position

        loaded = []

        path_nuc = nucleus_zavg_path(root_dir, pos)
        path_cell = cell_zavg_path(root_dir, pos)
        if path_nuc.exists():
            self.viewer.add_image(
                tifffile.imread(str(path_nuc)),
                name=f"nucleus_input_pos{pos:02d}",
                colormap="gray",
            )
            loaded.append("nucleus input")
        if path_cell.exists():
            self.viewer.add_image(
                tifffile.imread(str(path_cell)),
                name=f"cell_input_pos{pos:02d}",
                colormap="gray",
            )
            loaded.append("cell input")

        cell_dir = stage_dir(root_dir, pos, "cellpose_cell")

        # Per-z-slice outputs (T, Z, H, W) and (T, Z, 2, H, W)
        prob_zslices = cell_dir / "cell_prob.tif"
        dp_zslices   = cell_dir / "cell_dp.tif"
        if prob_zslices.exists():
            self.viewer.add_image(
                tifffile.imread(str(prob_zslices)),
                name=f"cell_prob_zslices_pos{pos:02d}",
                colormap="inferno",
            )
            loaded.append("cell prob (z-slices)")
        if dp_zslices.exists():
            dp = tifffile.imread(str(dp_zslices))  # (T, Z, 2, H, W)
            for c in range(dp.shape[2]):
                self.viewer.add_image(dp[:, :, c], name=f"cell_dp_{c}_zslices_pos{pos:02d}", colormap="RdBu")
            loaded.append("cell dp (z-slices)")

        # Z-averaged outputs
        prob_zavg = cell_dir / "cell_prob_zavg.tif"
        dp_zavg   = cell_dir / "cell_dp_zavg.tif"
        if prob_zavg.exists():
            self.viewer.add_image(
                tifffile.imread(str(prob_zavg)),
                name=f"cell_prob_zavg_pos{pos:02d}",
                colormap="inferno",
                visible=False,
            )
            loaded.append("cell prob avg")
        if dp_zavg.exists():
            dp = tifffile.imread(str(dp_zavg))  # (T, 2, H, W)
            for c in range(dp.shape[1]):
                self.viewer.add_image(dp[:, c], name=f"cell_dp_{c}_zavg_pos{pos:02d}", colormap="RdBu", visible=False)
            loaded.append("cell dp avg")

        if loaded:
            self._s01b_status_label.setText(f"Loaded: {', '.join(loaded)}")
        else:
            self._s01b_status_label.setText("No files found.")

    # ── s01b Preview ─────────────────────────────────────────────────────

    def _on_s01b_preview(self) -> None:
        from cellflow.cellpose.stages.raw_import import cell_3d_path, nucleus_3d_path

        root_dir = self._s01b_root_dir()
        if root_dir is None:
            self._s01b_status_label.setText("No project open. Create or open a project first.")
            return

        pos = self._state.current_position
        t = self._s01b_frame_spin.value()

        path_cell = cell_3d_path(root_dir, pos, t)
        path_nuc  = nucleus_3d_path(root_dir, pos, t)
        if not path_cell.exists() or not path_nuc.exists():
            self._s01b_status_label.setText(
                f"Input z-stacks not found for t={t}. Run data prep first."
            )
            return

        cfg = self._build_s01b_config()
        self._s01b_preview_btn.setEnabled(False)
        self._s01b_status_label.setText(f"Running per-z-slice preview for t={t}…")

        @thread_worker(
            connect={
                "yielded":  self._on_s01b_preview_status,
                "returned": self._on_s01b_preview_returned,
                "errored":  self._on_s01b_preview_error,
            }
        )
        def _work():
            import numpy as np
            from cellflow.cellpose.stages.cell_2d import _load_model, _apply_gamma

            yield f"Loading z-stacks for t={t}…"
            cell_z = tifffile.imread(str(path_cell)).astype(np.float32)  # (Z, H, W)
            nuc_z  = tifffile.imread(str(path_nuc)).astype(np.float32)   # (Z, H, W)
            Z = cell_z.shape[0]

            yield f"Loading model '{cfg.model}'…"
            model = _load_model(cfg.model, cfg.use_gpu)
            z_dp_list:   list[np.ndarray] = []
            z_prob_list: list[np.ndarray] = []
            try:
                for z in range(Z):
                    yield f"Running inference z={z}/{Z}…"
                    img = np.stack([cell_z[z], nuc_z[z]], axis=-1)  # (H, W, 2)
                    if cfg.gamma is not None and cfg.gamma != 1.0:
                        _apply_gamma(img, cfg.gamma)
                    _, flows, _ = model.eval(
                        img,
                        diameter=cfg.diameter if cfg.diameter > 0 else None,
                        min_size=cfg.min_size,
                    )
                    z_dp_list.append(flows[1].astype(np.float32))    # (2, H, W)
                    z_prob_list.append(flows[2].astype(np.float32))  # (H, W)
            finally:
                del model
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            dp   = np.stack(z_dp_list,   axis=0)  # (Z, 2, H, W)
            prob = np.stack(z_prob_list, axis=0)  # (Z, H, W)
            return t, dp, prob

        self._worker_s01b_preview = _work()

    def _on_s01b_preview_status(self, msg: str) -> None:
        self._s01b_status_label.setText(msg)

    def _on_s01b_preview_returned(self, result: tuple) -> None:
        import numpy as np

        self._s01b_preview_btn.setEnabled(True)
        self._worker_s01b_preview = None
        t, dp, prob = result
        # dp: (Z, 2, H, W)  prob: (Z, H, W)

        # Per-z-slice prob (Z, H, W) — browsable stack
        name = "Preview: cell_prob (z-slices)"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = prob
        else:
            self.viewer.add_image(prob, name=name, colormap="inferno")

        # Per-z-slice dp components (Z, H, W)
        for c in range(dp.shape[1]):
            name = f"Preview: cell_dp_{c} (z-slices)"
            if name in self.viewer.layers:
                self.viewer.layers[name].data = dp[:, c]
            else:
                self.viewer.add_image(dp[:, c], name=name, colormap="RdBu", visible=False)

        # Z-averaged prob and dp
        prob_avg = prob.mean(axis=0)
        name = "Preview: cell_prob (z-avg)"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = prob_avg
        else:
            self.viewer.add_image(prob_avg, name=name, colormap="inferno", visible=False)

        dp_avg = dp.mean(axis=0)  # (2, H, W)
        for c in range(dp_avg.shape[0]):
            name = f"Preview: cell_dp_{c} (z-avg)"
            if name in self.viewer.layers:
                self.viewer.layers[name].data = dp_avg[c]
            else:
                self.viewer.add_image(dp_avg[c], name=name, colormap="RdBu", visible=False)

        Z = dp.shape[0]
        self._s01b_status_label.setText(
            f"Preview done  t={t}  Z={Z}  prob={prob.shape}  dp={dp.shape}"
        )

    def _on_s01b_preview_error(self, exc: Exception) -> None:
        self._s01b_preview_btn.setEnabled(True)
        self._worker_s01b_preview = None
        self._s01b_status_label.setText(f"Preview error: {exc}")

    # ── s01a Preview ─────────────────────────────────────────────────────

    def _on_s01a_preview(self) -> None:
        pos = self._state.current_position
        input_dir = self._s01a_input_dir(pos)
        if input_dir is None:
            self._s01a_status_label.setText("No project open. Create or open a project first.")
            return

        t = self._s01a_frame_spin.value()
        in_path = input_dir / "nucleus_4d.tif"
        if not in_path.exists():
            self._s01a_status_label.setText("No nucleus_4d.tif input found.")
            return
        cfg = self._build_s01a_config()
        self._s01a_preview_btn.setEnabled(False)
        self._s01a_status_label.setText(f"Running nucleus preview for frame {t}…")

        @thread_worker(
            connect={
                "yielded":  self._on_s01a_preview_status,
                "returned": self._on_s01a_preview_returned,
                "errored":  self._on_s01a_preview_error,
            }
        )
        def _work():
            from cellflow.cellpose.stages.nucleus_3d import _load_model

            yield f"Loading frame {t}…"
            stack = tifffile.imread(str(in_path)).astype(np.float32)
            if stack.ndim != 4:
                raise ValueError(f"Expected nucleus_4d.tif with shape (T, Z, Y, X), got {stack.shape}")
            if t >= stack.shape[0]:
                raise IndexError(f"Frame {t} out of range (0–{stack.shape[0] - 1})")
            img = stack[t]
            gamma = cfg.gamma
            if gamma is not None and gamma != 1.0:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (
                        ((img - img_min) / (img_max - img_min)) ** gamma
                        * (img_max - img_min)
                        + img_min
                    )

            yield f"Loading model '{cfg.model}'…"
            model = _load_model(cfg.model, cfg.use_gpu)
            try:
                yield "Running nucleus inference…"
                _, flows, _ = model.eval(
                    img,
                    do_3D=True,
                    z_axis=0,
                    diameter=cfg.diameter if cfg.diameter > 0 else None,
                    anisotropy=cfg.anisotropy,
                    min_size=cfg.min_size,
                )
                dp = flows[1].astype(np.float32)    # (3, Z, Y, X)
                prob = flows[2].astype(np.float32)  # (Z, Y, X)
            finally:
                del model
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            return t, dp, prob

        self._worker_s01a_preview = _work()

    def _on_s01a_preview_returned(self, result: tuple) -> None:
        self._s01a_preview_btn.setEnabled(True)
        self._worker_s01a_preview = None
        t, dp, prob = result

        prob_name = "Preview: nucleus_prob"
        if prob_name in self.viewer.layers:
            self.viewer.layers[prob_name].data = prob
        else:
            self.viewer.add_image(prob, name=prob_name, colormap="inferno")

        # dp: (3, Z, Y, X) — split into one layer per vector component
        for c in range(dp.shape[0]):
            name = f"Preview: nucleus_dp_{c}"
            if name in self.viewer.layers:
                self.viewer.layers[name].data = dp[c]
            else:
                self.viewer.add_image(dp[c], name=name, colormap="RdBu")

        self._s01a_status_label.setText(
            f"Preview done  t={t}  prob={prob.shape}  dp={dp.shape}"
        )

    def _on_s01a_preview_status(self, msg: str) -> None:
        self._s01a_status_label.setText(msg)

    def _on_s01a_preview_error(self, exc: Exception) -> None:
        self._s01a_preview_btn.setEnabled(True)
        self._worker_s01a_preview = None
        self._s01a_status_label.setText(f"Preview error: {exc}")

    # ── get_params / set_params ──────────────────────────────────────────

    def get_params(self) -> dict:
        result = {"cellpose_cluster": {}}
        if not _CELLPOSE_PIPELINE_AVAILABLE:
            return result
        cfg_a = self._build_s01a_config()
        result["cellpose_cluster"]["nucleus"] = {
            **cfg_a.model_dump(),
            "overwrite": self._s01a_overwrite_check.isChecked(),
        }
        cfg_b = self._build_s01b_config()
        result["cellpose_cluster"]["cell"] = {
            **cfg_b.model_dump(),
            "overwrite": self._s01b_overwrite_check.isChecked(),
        }
        return result

    def set_params(self, data: dict) -> None:
        if not _CELLPOSE_PIPELINE_AVAILABLE:
            return
        cluster = data.get("cellpose_cluster")
        if cluster:
            if "nucleus" in cluster:
                d = cluster["nucleus"]
                cfg = CellposeConfig(**{k: v for k, v in d.items() if k != "overwrite"})
                self._apply_s01a_config(cfg)
                if "overwrite" in d:
                    self._s01a_overwrite_check.setChecked(bool(d["overwrite"]))
            if "cell" in cluster:
                d = cluster["cell"]
                cfg = CellposeConfig(**{k: v for k, v in d.items() if k != "overwrite"})
                self._apply_s01b_config(cfg)
                if "overwrite" in d:
                    self._s01b_overwrite_check.setChecked(bool(d["overwrite"]))
            return
        if "cellpose_nucleus" in data:
            d = data["cellpose_nucleus"]
            cfg = CellposeConfig(**{k: v for k, v in d.items() if k != "overwrite"})
            self._apply_s01a_config(cfg)
            if "overwrite" in d:
                self._s01a_overwrite_check.setChecked(bool(d["overwrite"]))
        if "cellpose_cell" in data:
            d = data["cellpose_cell"]
            cfg = CellposeConfig(**{k: v for k, v in d.items() if k != "overwrite"})
            self._apply_s01b_config(cfg)
            if "overwrite" in d:
                self._s01b_overwrite_check.setChecked(bool(d["overwrite"]))
