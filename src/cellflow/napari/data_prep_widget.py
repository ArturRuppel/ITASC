"""Widget for exporting and preparing raw data."""
from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
import shlex

import napari
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QCheckBox,
    QProgressBar,
    QSpinBox,
)
from qtpy.QtCore import Qt, QTimer
from napari.qt.threading import thread_worker

from cellflow.napari.widgets import PipelineFilesWidget
from cellflow.napari.ui_style import muted_label, status_label
from cellflow.core.data_prep import DatasetConfig, discover_metadata, run as run_prep
from cellflow.napari.utils import launch_in_terminal

if TYPE_CHECKING:
    from cellflow.napari.main_widget import CellFlowMainWidget


class DataPrepWidget(QWidget):
    """Widget for exporting and preparing raw data."""

    def __init__(self, viewer: napari.Viewer, main_widget: CellFlowMainWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.main_widget = main_widget
        self._worker = None
        self._meta_worker = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # NDTiff path
        layout.addWidget(QLabel("NDTiff Directory:"))
        row = QHBoxLayout()
        self.ndtiff_edit = QLineEdit()
        self.ndtiff_edit.setPlaceholderText("/path/to/ndtiff_dataset")
        row.addWidget(self.ndtiff_edit)
        self.browse_btn = QPushButton("Browse...")
        row.addWidget(self.browse_btn)
        self.pull_btn = QPushButton("Pull Metadata")
        row.addWidget(self.pull_btn)
        layout.addLayout(row)

        # Metadata display (placeholders)
        meta_row = QHBoxLayout()
        self.px_label = QLabel("Pixel size: —")
        muted_label(self.px_label)
        meta_row.addWidget(self.px_label)
        self.dt_label = QLabel("Interval: —")
        muted_label(self.dt_label)
        meta_row.addWidget(self.dt_label)
        meta_row.addStretch()
        layout.addLayout(meta_row)

        # Positions
        layout.addWidget(QLabel("Positions (e.g. 0,1,2):"))
        self.pos_edit = QLineEdit("0")
        layout.addWidget(self.pos_edit)

        # XY downsample
        ds_row = QHBoxLayout()
        ds_row.addWidget(QLabel("XY Downsample:"))
        self.ds_spin = QSpinBox()
        self.ds_spin.setRange(1, 16)
        self.ds_spin.setValue(2)
        ds_row.addWidget(self.ds_spin)
        layout.addLayout(ds_row)

        # Frame range
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frames:"))
        frame_row.addWidget(QLabel("Start"))
        self.frame_start_spin = QSpinBox()
        self.frame_start_spin.setRange(0, 999999)
        self.frame_start_spin.setValue(0)
        frame_row.addWidget(self.frame_start_spin)
        frame_row.addWidget(QLabel("End"))
        self.frame_end_spin = QSpinBox()
        self.frame_end_spin.setRange(-1, 999999)
        self.frame_end_spin.setValue(-1)
        self.frame_end_spin.setSpecialValueText("last")
        frame_row.addWidget(self.frame_end_spin)
        layout.addLayout(frame_row)

        # Overwrite
        self.overwrite_check = QCheckBox("Overwrite existing files")
        layout.addWidget(self.overwrite_check)

        # Run buttons
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Export")
        self.term_btn = QPushButton("Run in Terminal")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.term_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        # Progress & Status
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status_label = QLabel("")
        status_label(self.status_label)
        layout.addWidget(self.status_label)

        # ── Project file status ──────────────────────────────────────────
        self.files_tracker = PipelineFilesWidget([
            ("Outputs", [
                ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
                ("0_input/cell_zavg.tif", "Cell z-avg"),
                ("0_input/NLS_zavg.tif", "NLS z-avg"),
                ("0_input/nucleus_3dt.tif", "Nucleus 3D+t"),
                ("0_input/cell_3dt.tif", "Cell 3D+t"),
                ("0_input/NLS_3dt.tif", "NLS 3D+t"),
                ("0_input/z_shift.csv", "Z shift CSV"),
            ]),
        ], viewer=self.viewer)
        layout.addWidget(self.files_tracker)

        # Connect signals
        self.browse_btn.clicked.connect(self._on_browse)
        self.pull_btn.clicked.connect(self._on_pull_metadata)
        self.run_btn.clicked.connect(self._on_run)
        self.term_btn.clicked.connect(self._on_run_in_terminal)
        self.cancel_btn.clicked.connect(self._on_cancel)
        
        # Debounce timer for auto-pull
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(500)
        self._debounce_timer.timeout.connect(self._on_pull_metadata)
        self.ndtiff_edit.textChanged.connect(self._debounce_timer.start)

        # Auto-refresh on project change
        self.main_widget.refresh_requested.connect(self.refresh)

    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select NDTiff Directory")
        if path:
            self.ndtiff_edit.setText(path)

    def _on_pull_metadata(self) -> None:
        path = self.ndtiff_edit.text().strip()
        if not path or not Path(path).exists():
            return

        if self._meta_worker is not None:
            return

        self.status_label.setText("Reading metadata...")

        @thread_worker(connect={"returned": self._on_metadata_returned, "errored": self._on_error})
        def _work():
            return discover_metadata(path)
        
        self._meta_worker = _work()

    def _on_metadata_returned(self, result: dict) -> None:
        self._meta_worker = None
        positions = result.get("positions", [])
        px = result.get("pixel_size_um")
        dt = result.get("time_interval_s")

        self.pos_edit.setText(",".join(str(p) for p in positions))
        if px:
            self.px_label.setText(f"Pixel size: {px:.4g} µm")
            self.main_widget.px_edit.setText(f"{px:.4g}")
        if dt:
            self.dt_label.setText(f"Interval: {dt/60:.4g} min")
            self.main_widget.dt_edit.setText(f"{dt/60:.4g}")
        
        self.status_label.setText(f"Metadata: {len(positions)} positions found.")

    def _on_run(self) -> None:
        root_dir = self.main_widget.path_label.text()
        if not root_dir or root_dir == "[no project]":
            self.status_label.setText("Error: No project open.")
            return

        ndtiff_path = self.ndtiff_edit.text().strip()
        if not ndtiff_path:
            self.status_label.setText("Error: No NDTiff directory.")
            return

        try:
            positions = [int(p.strip()) for p in self.pos_edit.text().split(",") if p.strip()]
        except ValueError:
            self.status_label.setText("Error: Invalid positions.")
            return

        config = DatasetConfig(
            ndtiff_path=ndtiff_path,
            root_dir=root_dir,
            positions=positions,
            xy_downsample=self.ds_spin.value(),
            frame_start=self.frame_start_spin.value(),
            frame_end=self.frame_end_spin.value(),
        )

        self.run_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status_label.setText("Starting export...")

        overwrite = self.overwrite_check.isChecked()

        @thread_worker(connect={"yielded": self._on_progress, "finished": self._on_finished, "errored": self._on_error})
        def _work():
            for pos in positions:
                for done, total, label in run_prep(config, pos, overwrite=overwrite):
                    yield (pos, done, total, label)

        self._worker = _work()
        self._worker.aborted.connect(self._on_aborted)

    def _on_progress(self, update: tuple) -> None:
        pos, done, total, label = update
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.status_label.setText(f"pos{pos:02d} — {label} [{done}/{total}]")

    def _on_finished(self) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress.setVisible(False)
        self.status_label.setText("Export finished.")
        self._worker = None
        self.refresh(self.main_widget.path_label.text())

    def _on_error(self, exc: Exception) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress.setVisible(False)
        self.status_label.setText(f"Error: {exc}")
        self._worker = None
        self._meta_worker = None

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.quit()

    def _on_aborted(self) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        self.progress.setVisible(False)
        self.status_label.setText("Cancelled.")
        self._worker = None

    def _on_run_in_terminal(self) -> None:
        root_dir = self.main_widget.path_label.text()
        if not root_dir or root_dir == "[no project]":
            self.status_label.setText("Error: No project open.")
            return

        ndtiff_path = self.ndtiff_edit.text().strip()
        overwrite = self.overwrite_check.isChecked()
        ds = self.ds_spin.value()
        positions = self.pos_edit.text().strip()
        frame_start = self.frame_start_spin.value()
        frame_end = self.frame_end_spin.value()

        python_code = (
            "from cellflow.core.data_prep import DatasetConfig, run\n"
            f"config = DatasetConfig(ndtiff_path={ndtiff_path!r}, root_dir={root_dir!r}, "
            f"positions=[{positions}], xy_downsample={ds}, "
            f"frame_start={frame_start}, frame_end={frame_end})\n"
            "for pos in config.positions:\n"
            "    print(f'--- pos{pos} ---', flush=True)\n"
            f"    for d, t, l in run(config, pos, overwrite={overwrite}):\n"
            "        print(f'  {l} [{d}/{t}]', flush=True)"
        )
        cmd = f"python -c {shlex.quote(python_code)}"
        
        try:
            launch_in_terminal(cmd)
            self.status_label.setText("Launched in terminal.")
        except Exception as e:
            self.status_label.setText(f"Terminal error: {e}")

    def get_state(self) -> dict:
        """Return the current UI state as a dictionary."""
        return {
            "ndtiff_path": self.ndtiff_edit.text(),
            "positions": self.pos_edit.text(),
            "xy_downsample": self.ds_spin.value(),
            "frame_start": self.frame_start_spin.value(),
            "frame_end": self.frame_end_spin.value(),
            "overwrite": self.overwrite_check.isChecked(),
        }

    def set_state(self, state: dict) -> None:
        """Update the UI state from a dictionary."""
        if "ndtiff_path" in state:
            self.ndtiff_edit.setText(state["ndtiff_path"])
        if "positions" in state:
            self.pos_edit.setText(state["positions"])
        if "xy_downsample" in state:
            self.ds_spin.setValue(state["xy_downsample"])
        if "frame_start" in state:
            self.frame_start_spin.setValue(state["frame_start"])
        if "frame_end" in state:
            self.frame_end_spin.setValue(state["frame_end"])
        if "overwrite" in state:
            self.overwrite_check.setChecked(state["overwrite"])

    def refresh(self, pos_dir: Path | str | None) -> None:
        """Update file status display."""
        if pos_dir is None or str(pos_dir) == "[no project]":
            self.files_tracker.refresh(None)
            return
        
        p = Path(pos_dir)
        if not p.name.startswith("pos"):
            pos = self.main_widget.pos_spin.value()
            p = p / f"pos{pos:02d}"
        
        self.files_tracker.refresh(p)
