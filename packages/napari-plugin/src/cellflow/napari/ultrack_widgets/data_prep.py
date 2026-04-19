"""Data Preparation tab — s00 raw NDTiff export."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari.qt.threading import thread_worker

from cellflow.cellpose.config import DatasetConfig
from cellflow.cellpose.stages.raw_import import discover_metadata, run as run_s00
from cellflow.napari.log_viewer import StageLogViewer
from cellflow.napari.registry import get_state
from cellflow.napari.runners.terminal import launch_in_terminal
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget


class DataPrepWidget(QWidget):
    """Widget for exporting raw NDTiff data to per-timepoint TIFFs."""

    run_started = Signal()

    def __init__(self, viewer: "napari.Viewer", *, log_viewer=None) -> None:
        super().__init__()
        self.viewer = viewer
        self._state = get_state(viewer)
        self._worker = None
        self._meta_worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignTop)

        # ── Project file status (output files for current position) ──────
        self._files_widget = PipelineFilesWidget([
            ("Output", [
                ("0_input/cell/cell_zavg.tif",      "Cell avg"),
                ("0_input/nucleus/nucleus_zavg.tif", "Nucleus avg"),
            ]),
        ])
        layout.addWidget(self._files_widget)

        # ── Parameters accordion section ─────────────────────────────────
        params_inner = QWidget()
        params_layout = QVBoxLayout(params_inner)
        params_layout.setContentsMargins(4, 4, 4, 4)
        params_layout.setSpacing(4)

        # NDTiff path
        params_layout.addWidget(QLabel("NDTiff directory"))
        row = QHBoxLayout()
        self._ndtiff_edit = QLineEdit()
        self._ndtiff_edit.setPlaceholderText("/path/to/ndtiff_dataset")
        row.addWidget(self._ndtiff_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_ndtiff)
        row.addWidget(browse_btn)
        pull_btn = QPushButton("Pull Metadata")
        pull_btn.clicked.connect(self._pull_metadata)
        row.addWidget(pull_btn)
        params_layout.addLayout(row)

        # Metadata display
        meta_row = QHBoxLayout()
        self._px_label = QLabel("Pixel size: —")
        self._px_label.setStyleSheet("color: grey; font-size: 8pt;")
        meta_row.addWidget(self._px_label)
        self._dt_label = QLabel("Interval: —")
        self._dt_label.setStyleSheet("color: grey; font-size: 8pt;")
        meta_row.addWidget(self._dt_label)
        meta_row.addStretch()
        params_layout.addLayout(meta_row)

        # Positions
        params_layout.addWidget(QLabel("Positions (comma-separated, e.g. 0,1,2)"))
        self._positions_edit = QLineEdit("0")
        params_layout.addWidget(self._positions_edit)

        # XY downsample
        row = QHBoxLayout()
        row.addWidget(QLabel("XY downsample factor"))
        self._xy_spin = QSpinBox()
        self._xy_spin.setRange(1, 16)
        self._xy_spin.setValue(3)
        row.addWidget(self._xy_spin)
        params_layout.addLayout(row)

        self._params_section = CollapsibleSection("Parameters", params_inner, expanded=True)
        layout.addWidget(self._params_section)

        # ── Overwrite ────────────────────────────────────────────────────
        self._overwrite_check = QCheckBox("Overwrite existing files")
        self._overwrite_check.setStyleSheet("color: white;")
        layout.addWidget(self._overwrite_check)

        # ── Run buttons ──────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Export")
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)
        self._term_btn = QPushButton("Run in Terminal")
        self._term_btn.clicked.connect(self._on_run_in_terminal)
        btn_row.addWidget(self._term_btn)
        layout.addLayout(btn_row)

        # ── Progress ─────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        if log_viewer is not None:
            self._log_viewer = log_viewer
        else:
            self._log_viewer = StageLogViewer(self._state)
            layout.addWidget(self._log_viewer)

        # ── Debounce timer for auto-pull on path change ───────────────────
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._pull_metadata)
        self._ndtiff_edit.textChanged.connect(self._debounce_timer.start)

        # Connect project/position change signals
        self._state.pipeline_schema_changed.connect(self._sync_project_dir)
        self._state.position_changed.connect(self._sync_project_dir)

        # Populate from any already-open project
        self._sync_project_dir()

    # ── Project sync ─────────────────────────────────────────────────────────

    def _sync_project_dir(self) -> None:
        """Auto-fill NDTiff dir from schema.input_dir and refresh file-status rows."""
        project_dir = self._state.project_dir
        schema = self._state.pipeline_schema

        if project_dir is None:
            self._files_widget.refresh(None)
            return

        from pathlib import Path
        pos = self._state.current_position
        self._files_widget.refresh(Path(project_dir) / f"pos{pos:02d}")

        # Auto-fill NDTiff if schema has input_dir and field is empty
        if (
            schema is not None
            and schema.input_dir
            and not self._ndtiff_edit.text().strip()
        ):
            self._ndtiff_edit.setText(schema.input_dir)

    # ── Browsing ─────────────────────────────────────────────────────────────

    def _browse_ndtiff(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select NDTiff directory")
        if d:
            self._ndtiff_edit.setText(d)

    # ── Metadata pull ─────────────────────────────────────────────────────────

    def _pull_metadata(self) -> None:
        """Discover metadata from the NDTiff dataset in a background thread."""
        path = self._ndtiff_edit.text().strip()
        if not path:
            return

        # Cancel any in-flight worker
        if self._meta_worker is not None:
            try:
                self._meta_worker.quit()
            except Exception:
                pass
            self._meta_worker = None

        self._status_label.setText("Reading metadata…")

        @thread_worker(
            connect={
                "returned": self._on_metadata,
                "errored": self._on_metadata_error,
            }
        )
        def _work():
            return discover_metadata(path)

        self._meta_worker = _work()

    def _on_metadata(self, result: dict) -> None:
        self._meta_worker = None
        positions = result.get("positions", [])
        px = result.get("pixel_size_um")
        dt = result.get("time_interval_s")

        self._positions_edit.setText(",".join(str(p) for p in positions))

        if px is not None:
            self._px_label.setText(f"Pixel size: {px:.4g} µm")
        else:
            self._px_label.setText("Pixel size: —")

        if dt is not None:
            self._dt_label.setText(f"Interval: {dt:.4g} s")
        else:
            self._dt_label.setText("Interval: —")

        self._status_label.setText(
            f"Metadata: {len(positions)} position(s) found."
        )

    def _on_metadata_error(self, exc: Exception) -> None:
        self._meta_worker = None
        self._status_label.setText(f"Metadata error: {exc}")

    # ── Parsing ──────────────────────────────────────────────────────────────

    def _parse_int_list(self, text: str) -> list[int]:
        return [int(x.strip()) for x in text.split(",") if x.strip()]

    def _get_root_dir(self) -> str | None:
        project_dir = self._state.project_dir
        if project_dir is None:
            return None
        return str(project_dir)

    def _build_config(self) -> DatasetConfig:
        root_dir = self._get_root_dir()
        if root_dir is None:
            raise ValueError("No project open. Create or open a project first.")
        return DatasetConfig(
            ndtiff_path=self._ndtiff_edit.text().strip(),
            root_dir=root_dir,
            positions=self._parse_int_list(self._positions_edit.text()),
            xy_downsample=self._xy_spin.value(),
        )

    # ── Run ──────────────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        try:
            config = self._build_config()
        except Exception as e:
            self._status_label.setText(f"Config error: {e}")
            return

        if not config.ndtiff_path:
            self._status_label.setText("Please set the NDTiff directory.")
            return
        if not config.positions:
            self._status_label.setText("Please specify at least one position.")
            return

        self._run_btn.setEnabled(False)
        self._term_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText("Starting…")

        positions = list(config.positions)
        overwrite = self._overwrite_check.isChecked()

        self._n_positions = len(positions)

        @thread_worker(
            connect={
                "yielded": self._on_progress,
                "finished": self._on_finished,
                "errored": self._on_error,
            }
        )
        def _work():
            for pos in positions:
                for done, total, label in run_s00(config, pos, overwrite=overwrite):
                    yield (pos, done, total, label)

        self.run_started.emit()
        self._worker = _work()

    def _on_progress(self, update: tuple) -> None:
        pos, done, total, label = update
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        self._status_label.setText(f"pos{pos:02d} — {label} [{done}/{total}]")

    def _on_finished(self) -> None:
        self._run_btn.setEnabled(True)
        self._term_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"Done — exported {self._n_positions} position(s).")
        self._worker = None
        self._log_viewer.refresh()
        self._try_load_metadata()

    def _try_load_metadata(self) -> None:
        """Read pixel size / time interval from run_params.json (pos 0) and push to state."""
        project_dir = self._state.project_dir
        if project_dir is None:
            return
        from cellflow.core.paths import stage_dir
        params_path = stage_dir(project_dir, 0, "raw_import") / "run_params.json"
        if not params_path.exists():
            return
        try:
            data = json.loads(params_path.read_text(encoding="utf-8"))
            px = data.get("pixel_size_um")
            dt_s = data.get("time_interval_s")
            if px is not None:
                self._state.pixel_size = float(px)
            if dt_s is not None:
                self._state.time_interval = float(dt_s)
        except Exception:
            pass

    # ── Run in Terminal ───────────────────────────────────────────────────────

    def _on_run_in_terminal(self) -> None:
        try:
            config = self._build_config()
        except Exception as e:
            self._status_label.setText(f"Config error: {e}")
            return

        if not config.ndtiff_path:
            self._status_label.setText("Please set the NDTiff directory.")
            return
        if not config.positions:
            self._status_label.setText("Please specify at least one position.")
            return

        overwrite = self._overwrite_check.isChecked()
        parts = []
        for pos in config.positions:
            cmd = (
                f"python -m cellflow.cellpose.stages.raw_import"
                f" --ndtiff-path {shlex.quote(config.ndtiff_path)}"
                f" --root-dir {shlex.quote(config.root_dir)}"
                f" --pos {pos}"
                f" --xy-downsample {config.xy_downsample}"
            )
            if overwrite:
                cmd += " --overwrite"
            parts.append(cmd)

        full_cmd = " && ".join(parts)
        try:
            launch_in_terminal(full_cmd)
            self._status_label.setText("Launched export in terminal.")
        except Exception as e:
            self._status_label.setText(f"Terminal error: {e}")

    def _on_error(self, exc: Exception) -> None:
        self._run_btn.setEnabled(True)
        self._term_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"Error: {exc}")
        self._worker = None
        self._log_viewer.refresh()

    def get_params(self) -> dict:
        return {
            "ndtiff_path": self._ndtiff_edit.text().strip(),
            "positions": self._positions_edit.text().strip(),
            "xy_downsample": self._xy_spin.value(),
            "overwrite": self._overwrite_check.isChecked(),
        }

    def set_params(self, data: dict) -> None:
        if data.get("ndtiff_path"):
            self._ndtiff_edit.setText(str(data["ndtiff_path"]))
        if "positions" in data:
            self._positions_edit.setText(str(data["positions"]))
        if "xy_downsample" in data:
            self._xy_spin.setValue(int(data["xy_downsample"]))
        if "overwrite" in data:
            self._overwrite_check.setChecked(bool(data["overwrite"]))
