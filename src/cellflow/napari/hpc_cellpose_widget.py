"""Standalone widget for launching the HPC Cellpose pipeline."""
from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path
from typing import Any

from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.ui_style import action_button, compact_spinbox, status_label
from cellflow.napari.utils import launch_in_terminal


DEFAULT_PIPELINE_SCRIPT = Path(
    "/home/aruppel/Projects/HPC/cellpose_full/run_pipeline.sh"
)
DEFAULT_CONFIG_PATH = Path(
    "/home/aruppel/Projects/HPC/cellpose_full/cellpose_full.json"
)


class HpcCellposeWidget(QWidget):
    """Widget for configuring and launching the external Cellpose pipeline."""

    def __init__(self, viewer: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.pipeline_script_path = DEFAULT_PIPELINE_SCRIPT

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        form = QFormLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)
        layout.addLayout(form)

        self.input_dir_edit = QLineEdit()
        self.input_dir_browse_btn = QPushButton("Browse...")
        form.addRow("Input dir:", self._path_row(self.input_dir_edit, self.input_dir_browse_btn))

        self.output_dir_edit = QLineEdit()
        self.output_dir_browse_btn = QPushButton("Browse...")
        form.addRow(
            "Output dir:",
            self._path_row(self.output_dir_edit, self.output_dir_browse_btn),
        )

        self.config_path_edit = QLineEdit(str(DEFAULT_CONFIG_PATH))
        self.config_path_browse_btn = QPushButton("Browse...")
        form.addRow(
            "Config:",
            self._path_row(self.config_path_edit, self.config_path_browse_btn),
        )

        self.nuclei_input_edit = QLineEdit("nucleus_3dt.tif")
        form.addRow("Nuclei input:", self.nuclei_input_edit)

        self.cells_input_edit = QLineEdit("cell_3dt.tif")
        form.addRow("Cells input:", self.cells_input_edit)

        self.frames_edit = QLineEdit("all")
        form.addRow("Frames:", self.frames_edit)

        self.nuclei_do_3d_check = QCheckBox("Nuclei 3D")
        form.addRow("", self.nuclei_do_3d_check)

        self.nuclei_anisotropy_spin = self._double_spin(0.01, 100.0, 1.5, 2)
        form.addRow("Nuclei anisotropy:", self.nuclei_anisotropy_spin)

        self.nuclei_diameter_spin = self._int_spin(0, 10000, 25)
        form.addRow("Nuclei diameter:", self.nuclei_diameter_spin)

        self.nuclei_size_spin = self._int_spin(0, 1000000, 0)
        form.addRow("Nuclei size:", self.nuclei_size_spin)

        self.nuclei_gamma_spin = self._double_spin(0.01, 100.0, 1.0, 2)
        form.addRow("Nuclei gamma:", self.nuclei_gamma_spin)

        self.cells_size_spin = self._int_spin(0, 1000000, 0)
        form.addRow("Cells size:", self.cells_size_spin)

        self.cells_gamma_spin = self._double_spin(0.01, 100.0, 1.0, 2)
        form.addRow("Cells gamma:", self.cells_gamma_spin)

        self.max_concurrent_jobs_spin = self._int_spin(1, 1000, 4)
        form.addRow("Max concurrent jobs:", self.max_concurrent_jobs_spin)

        self.remote_user_edit = QLineEdit("aruppel")
        form.addRow("Remote user:", self.remote_user_edit)

        self.remote_host_edit = QLineEdit("maestro.pasteur.fr")
        form.addRow("Remote host:", self.remote_host_edit)

        self.input_status_lbl = QLabel("")
        status_label(self.input_status_lbl, muted=True)
        layout.addWidget(self.input_status_lbl)

        self.run_btn = QPushButton("Run in Terminal")
        action_button(self.run_btn, expand=True)
        layout.addWidget(self.run_btn)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        status_label(self.status_lbl)
        layout.addWidget(self.status_lbl)

        self.input_dir_browse_btn.clicked.connect(self._browse_input_dir)
        self.output_dir_browse_btn.clicked.connect(self._browse_output_dir)
        self.config_path_browse_btn.clicked.connect(self._browse_config_path)
        self.run_btn.clicked.connect(self._on_run_terminal)

        for field in (
            self.input_dir_edit,
            self.nuclei_input_edit,
            self.cells_input_edit,
        ):
            field.textChanged.connect(self._update_input_status)

        self._update_input_status()

    def refresh(self, pos_dir: Path | str | None) -> None:
        """Derive default local pipeline paths from a selected position directory."""
        if pos_dir is None or str(pos_dir) == "[no project]":
            self.input_dir_edit.clear()
            self.output_dir_edit.clear()
            self._set_status("No project open.")
            self._update_input_status()
            return

        pos_path = Path(pos_dir)
        self.input_dir_edit.setText(str(pos_path / "0_input"))
        self.output_dir_edit.setText(str(pos_path / "1_cellpose"))
        self._set_status("")
        self._update_input_status()

    def get_state(self) -> dict[str, Any]:
        """Return the current control values."""
        return {
            "input_dir": self.input_dir_edit.text(),
            "output_dir": self.output_dir_edit.text(),
            "config_path": self.config_path_edit.text(),
            "nuclei_input": self.nuclei_input_edit.text(),
            "cells_input": self.cells_input_edit.text(),
            "frames": self.frames_edit.text(),
            "nuclei_do_3d": self.nuclei_do_3d_check.isChecked(),
            "nuclei_anisotropy": self.nuclei_anisotropy_spin.value(),
            "nuclei_diameter": self.nuclei_diameter_spin.value(),
            "nuclei_size": self.nuclei_size_spin.value(),
            "nuclei_gamma": self.nuclei_gamma_spin.value(),
            "cells_size": self.cells_size_spin.value(),
            "cells_gamma": self.cells_gamma_spin.value(),
            "max_concurrent_jobs": self.max_concurrent_jobs_spin.value(),
            "remote_user": self.remote_user_edit.text(),
            "remote_host": self.remote_host_edit.text(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Update controls from a previously saved state dictionary."""
        if "input_dir" in state:
            self.input_dir_edit.setText(str(state["input_dir"]))
        if "output_dir" in state:
            self.output_dir_edit.setText(str(state["output_dir"]))
        if "config_path" in state:
            self.config_path_edit.setText(str(state["config_path"]))
        if "nuclei_input" in state:
            self.nuclei_input_edit.setText(str(state["nuclei_input"]))
        if "cells_input" in state:
            self.cells_input_edit.setText(str(state["cells_input"]))
        if "frames" in state:
            self.frames_edit.setText(str(state["frames"]))
        if "nuclei_do_3d" in state:
            self.nuclei_do_3d_check.setChecked(bool(state["nuclei_do_3d"]))
        if "nuclei_anisotropy" in state:
            self.nuclei_anisotropy_spin.setValue(float(state["nuclei_anisotropy"]))
        if "nuclei_diameter" in state:
            self.nuclei_diameter_spin.setValue(int(state["nuclei_diameter"]))
        if "nuclei_size" in state:
            self.nuclei_size_spin.setValue(int(state["nuclei_size"]))
        if "nuclei_gamma" in state:
            self.nuclei_gamma_spin.setValue(float(state["nuclei_gamma"]))
        if "cells_size" in state:
            self.cells_size_spin.setValue(int(state["cells_size"]))
        if "cells_gamma" in state:
            self.cells_gamma_spin.setValue(float(state["cells_gamma"]))
        if "max_concurrent_jobs" in state:
            self.max_concurrent_jobs_spin.setValue(int(state["max_concurrent_jobs"]))
        if "remote_user" in state:
            self.remote_user_edit.setText(str(state["remote_user"]))
        if "remote_host" in state:
            self.remote_host_edit.setText(str(state["remote_host"]))
        self._update_input_status()

    def build_runtime_config(self) -> dict[str, Any]:
        """Build the temporary JSON payload consumed by the pipeline script."""
        return {
            "input_dir": self.input_dir_edit.text().strip(),
            "frames": self.frames_edit.text().strip() or "all",
            "nuclei": {
                "input": self.nuclei_input_edit.text().strip(),
                "do_3d": self.nuclei_do_3d_check.isChecked(),
                "anisotropy": self.nuclei_anisotropy_spin.value(),
                "diameter": self.nuclei_diameter_spin.value(),
                "size": self.nuclei_size_spin.value(),
                "gamma": self.nuclei_gamma_spin.value(),
            },
            "cells": {
                "input": self.cells_input_edit.text().strip(),
                "size": self.cells_size_spin.value(),
                "gamma": self.cells_gamma_spin.value(),
            },
        }

    def build_command(self, config_path: Path | str) -> str:
        """Build the shell command used for terminal launch."""
        parts = [
            "bash",
            str(self.pipeline_script_path),
            "--input-dir",
            self.input_dir_edit.text().strip(),
            "--output-dir",
            self.output_dir_edit.text().strip(),
            "--config",
            str(config_path),
            "--nuclei-input",
            self.nuclei_input_edit.text().strip(),
            "--cells-input",
            self.cells_input_edit.text().strip(),
            "--max-concurrent-jobs",
            str(self.max_concurrent_jobs_spin.value()),
            "--remote-user",
            self.remote_user_edit.text().strip(),
            "--remote-host",
            self.remote_host_edit.text().strip(),
        ]
        return " ".join(shlex.quote(part) for part in parts)

    def _on_run_terminal(self) -> None:
        error = self._validation_error()
        if error:
            self._set_status(error)
            return

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            suffix=".json",
            prefix="cellflow_hpc_cellpose_",
        ) as tmp:
            json.dump(self.build_runtime_config(), tmp, indent=2, sort_keys=True)
            tmp_path = Path(tmp.name)

        command = self.build_command(tmp_path)
        try:
            launch_in_terminal(command)
            self._set_status("HPC Cellpose command launched in terminal.")
        except Exception:
            QApplication.clipboard().setText(command)
            self._set_status("Terminal launch failed; command copied to clipboard.")

    def _validation_error(self) -> str | None:
        script_path = Path(self.pipeline_script_path)
        config_path = Path(self.config_path_edit.text().strip())
        input_dir = Path(self.input_dir_edit.text().strip())
        nuclei_input = self.nuclei_input_edit.text().strip()
        cells_input = self.cells_input_edit.text().strip()

        if not str(input_dir):
            return "Error: No project open."
        if not script_path.is_file():
            return f"Error: Pipeline script missing: {script_path}"
        if not config_path.is_file():
            return f"Error: Config file missing: {config_path}"
        if not input_dir.is_dir():
            return f"Error: Input directory missing: {input_dir}"
        if self.max_concurrent_jobs_spin.value() < 1:
            return "Error: Invalid max concurrent jobs."
        if not nuclei_input or not (input_dir / nuclei_input).is_file():
            return f"Error: Missing nuclei input: {input_dir / nuclei_input}"
        if not cells_input or not (input_dir / cells_input).is_file():
            return f"Error: Missing cells input: {input_dir / cells_input}"
        return None

    def _update_input_status(self) -> None:
        input_dir_text = self.input_dir_edit.text().strip()
        if not input_dir_text:
            self.input_status_lbl.setText("Inputs: no input directory selected.")
            return

        input_dir = Path(input_dir_text)
        nuclei_path = input_dir / self.nuclei_input_edit.text().strip()
        cells_path = input_dir / self.cells_input_edit.text().strip()
        nuclei_status = "ok" if nuclei_path.is_file() else "missing"
        cells_status = "ok" if cells_path.is_file() else "missing"
        self.input_status_lbl.setText(
            f"Inputs: nuclei {nuclei_status}; cells {cells_status}."
        )

    def _browse_input_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if path:
            self.input_dir_edit.setText(path)

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_edit.setText(path)

    def _browse_config_path(self) -> None:
        path = QFileDialog.getOpenFileName(
            self,
            "Select Pipeline Config",
            self.config_path_edit.text(),
            "JSON (*.json);;All Files (*)",
        )[0]
        if path:
            self.config_path_edit.setText(path)

    def _set_status(self, message: str) -> None:
        self.status_lbl.setText(message)
        self.status_lbl.setVisible(bool(message))

    @staticmethod
    def _path_row(line_edit: QLineEdit, button: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return row

    @staticmethod
    def _int_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return compact_spinbox(spin)

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(0.1)
        spin.setValue(value)
        return compact_spinbox(spin)
