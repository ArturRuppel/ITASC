"""Seeded watershed hypothesis sweep widget."""

from __future__ import annotations

import json
import sys
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

from napari.qt.threading import thread_worker

from cellflow.core.paths import stage_dir
from cellflow.napari.runners.terminal import launch_in_terminal
from cellflow.napari.log_viewer import StageLogViewer
from cellflow.napari.registry import get_state
from cellflow.napari.widgets import PipelineFilesWidget


def _seeded_ws_dir(root_dir, pos):
    return stage_dir(root_dir, pos, "seeded_watershed")


def _correction_dir(root_dir, pos):
    return stage_dir(root_dir, pos, "correction")


def _cellpose_cell_dir(root_dir, pos):
    return stage_dir(root_dir, pos, "cellpose_cell")


class SeededWatershedWidget(QWidget):
    run_started = Signal()

    def __init__(self, viewer: "napari.Viewer", *, log_viewer=None) -> None:
        super().__init__()
        self.viewer = viewer
        self._state = get_state(viewer)
        self._worker = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setAlignment(Qt.AlignTop)

        # ── Files ─────────────────────────────────────────────────────────
        self._files_widget = PipelineFilesWidget([
            ("Input", [
                ("3_correction/nuclear_labels_corrected.tif", "Nucleus labels"),
                ("1_cellpose/cell/cell_prob_zavg.tif", "Cell probability"),
            ]),
            ("Output", [
                ("4_seeded_watershed/foreground.tif", "Foreground map"),
                ("4_seeded_watershed/contours.tif", "Contours map"),
                ("4_seeded_watershed/hypotheses/", "Hypothesis label stacks"),
            ]),
        ])
        lay.addWidget(self._files_widget)

        # ── Weight source ──────────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Weight source"))
        self._weight_source_combo = QComboBox()
        self._weight_source_combo.addItems(["prob", "flow_mag", "both"])
        row.addWidget(self._weight_source_combo)
        row.addStretch()
        lay.addLayout(row)

        # ── Cellprob sweep ─────────────────────────────────────────────────
        cp_group = QGroupBox("Cellprob threshold sweep")
        cp_lay = QVBoxLayout(cp_group)
        cp_lay.setContentsMargins(6, 4, 6, 4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min"))
        self._cp_min_spin = self._dspin(-100.0, 100.0, 0.0, step=0.5)
        row.addWidget(self._cp_min_spin)
        row.addWidget(QLabel("Max"))
        self._cp_max_spin = self._dspin(-100.0, 100.0, 0.0, step=0.5)
        row.addWidget(self._cp_max_spin)
        row.addWidget(QLabel("Step"))
        self._cp_step_spin = self._dspin(0.0, 20.0, 0.5, step=0.5)
        row.addWidget(self._cp_step_spin)
        cp_lay.addLayout(row)
        lay.addWidget(cp_group)

        # ── Compactness ────────────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Compactness"))
        self._compactness_spin = self._dspin(0.0, 10.0, 0.0, step=0.01, decimals=3)
        row.addWidget(self._compactness_spin)
        row.addStretch()
        lay.addLayout(row)

        # ── Smooth sigma ───────────────────────────────────────────────────
        sm_group = QGroupBox("Smooth sigma sweep")
        sm_lay = QVBoxLayout(sm_group)
        sm_lay.setContentsMargins(6, 4, 6, 4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min"))
        self._sm_min_spin = self._dspin(0.0, 20.0, 1.0, step=0.5)
        row.addWidget(self._sm_min_spin)
        row.addWidget(QLabel("Max"))
        self._sm_max_spin = self._dspin(0.0, 20.0, 1.0, step=0.5)
        row.addWidget(self._sm_max_spin)
        row.addWidget(QLabel("Step"))
        self._sm_step_spin = self._dspin(0.0, 10.0, 0.5, step=0.5)
        row.addWidget(self._sm_step_spin)
        sm_lay.addLayout(row)
        lay.addWidget(sm_group)

        # ── Contour smoothing ──────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Contour smooth sigma"))
        self._contour_sigma_spin = self._dspin(0.0, 10.0, 0.5, step=0.25)
        row.addWidget(self._contour_sigma_spin)
        row.addStretch()
        lay.addLayout(row)

        # ── Workers ────────────────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("Workers"))
        self._workers_spin = QSpinBox()
        self._workers_spin.setRange(1, 64)
        self._workers_spin.setValue(1)
        row.addWidget(self._workers_spin)
        row.addStretch()
        lay.addLayout(row)

        # ── Options ────────────────────────────────────────────────────────
        self._save_hyp_chk = QCheckBox("Save all hypothesis stacks")
        self._save_hyp_chk.setChecked(True)
        lay.addWidget(self._save_hyp_chk)

        self._overwrite_chk = QCheckBox("Overwrite existing files")
        self._overwrite_chk.setStyleSheet("color: white;")
        lay.addWidget(self._overwrite_chk)

        # ── Run buttons ────────────────────────────────────────────────────
        row = QHBoxLayout()
        self._run_btn = QPushButton("Run Sweep")
        self._run_btn.clicked.connect(self._on_run)
        row.addWidget(self._run_btn)
        self._term_btn = QPushButton("Run in Terminal")
        self._term_btn.clicked.connect(self._on_run_terminal)
        row.addWidget(self._term_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        row.addWidget(self._cancel_btn)
        lay.addLayout(row)

        self._load_btn = QPushButton("Load Results")
        self._load_btn.clicked.connect(self._on_load_results)
        lay.addWidget(self._load_btn)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        lay.addWidget(self._status)

        if log_viewer is not None:
            self._log_viewer = log_viewer
        else:
            self._log_viewer = StageLogViewer(self._state)
            lay.addWidget(self._log_viewer)

        self._state.pipeline_schema_changed.connect(self._sync_project_dir)
        self._state.position_changed.connect(self._sync_project_dir)
        self._sync_project_dir()

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _dspin(lo, hi, val, *, step=0.1, decimals=2) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        s.setValue(val)
        return s

    def _root_dir(self) -> str | None:
        return str(self._state.project_dir) if self._state.project_dir else None

    def _sync_project_dir(self) -> None:
        root = self._root_dir()
        if root is None:
            self._files_widget.refresh(None)
            return
        pos = self._state.current_position
        self._files_widget.refresh(Path(root) / f"pos{pos:02d}")

    def _build_config(self) -> dict:
        return {
            "weight_source": self._weight_source_combo.currentText(),
            "cellprob_min": self._cp_min_spin.value(),
            "cellprob_max": self._cp_max_spin.value(),
            "cellprob_step": self._cp_step_spin.value(),
            "cellprob_threshold": self._cp_min_spin.value(),
            "compactness": self._compactness_spin.value(),
            "smooth_min": self._sm_min_spin.value(),
            "smooth_max": self._sm_max_spin.value(),
            "smooth_step": self._sm_step_spin.value(),
            "smooth_sigma": self._sm_min_spin.value(),
            "smooth_contour_sigma": self._contour_sigma_spin.value(),
            "save_all_hypotheses": self._save_hyp_chk.isChecked(),
            "n_workers": self._workers_spin.value(),
        }

    def _apply_config(self, data: dict) -> None:
        if "weight_source" in data:
            idx = self._weight_source_combo.findText(data["weight_source"])
            if idx >= 0:
                self._weight_source_combo.setCurrentIndex(idx)
        if "cellprob_min" in data:
            self._cp_min_spin.setValue(data["cellprob_min"])
        if "cellprob_max" in data:
            self._cp_max_spin.setValue(data["cellprob_max"])
        if "cellprob_step" in data:
            self._cp_step_spin.setValue(data["cellprob_step"])
        if "compactness" in data:
            self._compactness_spin.setValue(data["compactness"])
        if "smooth_min" in data:
            self._sm_min_spin.setValue(data["smooth_min"])
        if "smooth_max" in data:
            self._sm_max_spin.setValue(data["smooth_max"])
        if "smooth_step" in data:
            self._sm_step_spin.setValue(data["smooth_step"])
        if "smooth_contour_sigma" in data:
            self._contour_sigma_spin.setValue(data["smooth_contour_sigma"])
        if "save_all_hypotheses" in data:
            self._save_hyp_chk.setChecked(bool(data["save_all_hypotheses"]))
        if "n_workers" in data:
            self._workers_spin.setValue(int(data["n_workers"]))

    # ── Run ────────────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        root = self._root_dir()
        if not root:
            self._status.setText("No project open.")
            return

        pos = int(self._state.current_position)
        cfg_dict = self._build_config()
        overwrite = self._overwrite_chk.isChecked()

        self._run_btn.setEnabled(False)
        self._term_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText("Running sweep…")

        @thread_worker(
            connect={
                "yielded": self._on_progress,
                "finished": self._on_finished,
                "errored": self._on_error,
            }
        )
        def _work():
            from cellflow.cellpose.config import SeededWatershedConfig
            from cellflow.cellpose.stages.seeded_watershed import run
            from cellflow.core.logging import StageLogger
            from cellflow.core.paths import log_path

            cfg = SeededWatershedConfig(**cfg_dict)
            input_dir = _cellpose_cell_dir(root, pos)
            nuc_path = _correction_dir(root, pos) / "nuclear_labels_corrected.tif"
            out_dir = _seeded_ws_dir(root, pos)

            with StageLogger(log_path(root, pos), "seeded_watershed"):
                for update in run(input_dir, nuc_path, out_dir, cfg, overwrite=overwrite):
                    yield update

        self.run_started.emit()
        self._worker = _work()
        self._worker.aborted.connect(self._on_cancelled)

    def _on_run_terminal(self) -> None:
        root = self._root_dir()
        if not root:
            self._status.setText("No project open.")
            return

        pos = int(self._state.current_position)
        cfg_dict = self._build_config()
        overwrite_flag = "--overwrite" if self._overwrite_chk.isChecked() else ""

        cfg_path = Path(tempfile.mktemp(suffix="_sw_cfg.json"))
        cfg_path.write_text(json.dumps(cfg_dict, indent=2))

        input_dir = str(_cellpose_cell_dir(root, pos))
        nuc_path = str(_correction_dir(root, pos) / "nuclear_labels_corrected.tif")
        out_dir = str(_seeded_ws_dir(root, pos))

        cmd = (
            f"\"{sys.executable}\" -m cellflow.cellpose.stages.seeded_watershed"
            f" --input-dir \"{input_dir}\""
            f" --nucleus-labels \"{nuc_path}\""
            f" --output-dir \"{out_dir}\""
            f" --config \"{cfg_path}\""
            f" {overwrite_flag}"
        ).strip()

        try:
            launch_in_terminal(cmd)
            self._status.setText("Launched sweep in terminal.")
        except Exception as e:
            self._status.setText(f"Terminal launch error: {e}")

    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.quit()

    # ── Progress / finished / error ────────────────────────────────────────

    def _on_progress(self, update: tuple) -> None:
        done, total, label = update
        self._progress.setMaximum(max(total, 1))
        self._progress.setValue(done)
        pct = int(100 * done / total) if total > 0 else 0
        self._status.setText(f"{done}/{total} ({pct}%) — {label}")

    def _on_finished(self) -> None:
        self._run_btn.setEnabled(True)
        self._term_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._status.setText("Done.")
        self._worker = None
        self._log_viewer.refresh()

    def _on_error(self, exc: Exception) -> None:
        self._run_btn.setEnabled(True)
        self._term_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._status.setText(f"Error: {exc}")
        self._worker = None
        self._log_viewer.refresh()

    def _on_cancelled(self) -> None:
        self._run_btn.setEnabled(True)
        self._term_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._status.setText("Cancelled.")
        self._worker = None

    # ── Load results ───────────────────────────────────────────────────────

    def _on_load_results(self) -> None:
        root = self._root_dir()
        if not root:
            self._status.setText("No project open.")
            return

        pos = int(self._state.current_position)
        out_dir = _seeded_ws_dir(root, pos)
        fg_path = out_dir / "foreground.tif"
        ct_path = out_dir / "contours.tif"

        if not fg_path.exists():
            self._status.setText("No results found — run the sweep first.")
            return

        try:
            fg = tifffile.imread(str(fg_path))
            ct = tifffile.imread(str(ct_path))

            for name, data, cmap, blending in [
                ("SW: Foreground", fg, "gray", "translucent"),
                ("SW: Contours", ct, "magenta", "additive"),
            ]:
                if name in self.viewer.layers:
                    self.viewer.layers[name].data = data
                else:
                    self.viewer.add_image(data, name=name, colormap=cmap, blending=blending)

            hyp_dir = out_dir / "hypotheses"
            hyp_files = sorted(hyp_dir.glob("hypothesis_*.tif")) if hyp_dir.exists() else []
            self._status.setText(
                f"Loaded foreground + contours. {len(hyp_files)} hypothesis stack(s) in {hyp_dir.name}/."
            )
        except Exception as e:
            self._status.setText(f"Load error: {e}")

    # ── Config persistence ─────────────────────────────────────────────────

    def get_params(self) -> dict:
        return {**self._build_config(), "overwrite": self._overwrite_chk.isChecked()}

    def set_params(self, data: dict) -> None:
        self._apply_config(data)
        if "overwrite" in data:
            self._overwrite_chk.setChecked(bool(data["overwrite"]))
