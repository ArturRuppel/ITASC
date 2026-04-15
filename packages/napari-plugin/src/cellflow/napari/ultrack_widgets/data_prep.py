"""Data Preparation tab — s00 raw NDTiff export."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari.qt.threading import thread_worker

from cellflow.cellpose.config import DatasetConfig
from cellflow.cellpose.stages.raw_import import run as run_s00
from cellflow.napari.log_viewer import StageLogViewer
from cellflow.napari.registry import get_state


class DataPrepWidget(QWidget):
    """Widget for exporting raw NDTiff data to per-timepoint TIFFs."""

    def __init__(self, viewer: "napari.Viewer") -> None:
        super().__init__()
        self.viewer = viewer
        self._state = get_state(viewer)
        self._worker = None

        # ── Outer scroll area ────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        inner.setLayout(layout)
        scroll.setWidget(inner)

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.setLayout(outer)

        # ── Project info ─────────────────────────────────────────────────
        self._project_label = QLabel("")
        self._project_label.setStyleSheet("color: white; font-size: 8pt;")
        self._project_label.setWordWrap(True)
        layout.addWidget(self._project_label)

        # ── NDTiff path ──────────────────────────────────────────────────
        layout.addWidget(QLabel("NDTiff directory"))
        row = QHBoxLayout()
        self._ndtiff_edit = QLineEdit()
        self._ndtiff_edit.setPlaceholderText("/path/to/ndtiff_dataset")
        row.addWidget(self._ndtiff_edit)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_ndtiff)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Positions ────────────────────────────────────────────────────
        layout.addWidget(QLabel("Positions (comma-separated, e.g. 0,1,2)"))
        self._positions_edit = QLineEdit("0")
        layout.addWidget(self._positions_edit)

        # ── Timepoints ───────────────────────────────────────────────────
        self._tp_all_check = QCheckBox("All timepoints")
        self._tp_all_check.setChecked(True)
        self._tp_all_check.toggled.connect(self._on_tp_toggle)
        layout.addWidget(self._tp_all_check)

        row = QHBoxLayout()
        row.addWidget(QLabel("Timepoints (comma-separated)"))
        self._tp_edit = QLineEdit()
        self._tp_edit.setPlaceholderText("0,1,2,3,4")
        self._tp_edit.setEnabled(False)
        row.addWidget(self._tp_edit)
        layout.addLayout(row)

        # ── XY downsample ────────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("XY downsample factor"))
        self._xy_spin = QSpinBox()
        self._xy_spin.setRange(1, 16)
        self._xy_spin.setValue(3)
        row.addWidget(self._xy_spin)
        layout.addLayout(row)

        # ── Overwrite ────────────────────────────────────────────────────
        self._overwrite_check = QCheckBox("Overwrite existing files")
        layout.addWidget(self._overwrite_check)

        # ── Run button ───────────────────────────────────────────────────
        self._run_btn = QPushButton("Export Raw Data")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        # ── Progress ─────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        self._log_viewer = StageLogViewer(self._state)
        layout.addWidget(self._log_viewer)

        # Connect project change signal
        self._state.pipeline_schema_changed.connect(self._sync_project_dir)

        # Populate from any already-open project
        self._sync_project_dir()

    # ── Project sync ─────────────────────────────────────────────────────

    def _sync_project_dir(self) -> None:
        """Auto-fill NDTiff dir from schema.input_dir and update project info label."""
        project_dir = self._state.project_dir
        schema = self._state.pipeline_schema

        if project_dir is None:
            self._project_label.setText(
                "No project open — create or open one via the Project panel."
            )
            return

        self._project_label.setText(
            f"Output root: {project_dir}"
        )

        # Auto-fill NDTiff if schema has input_dir and field is empty
        if (
            schema is not None
            and schema.input_dir
            and not self._ndtiff_edit.text().strip()
        ):
            self._ndtiff_edit.setText(schema.input_dir)

    # ── Browsing ─────────────────────────────────────────────────────────

    def _browse_ndtiff(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select NDTiff directory")
        if d:
            self._ndtiff_edit.setText(d)

    def _on_tp_toggle(self, checked: bool) -> None:
        self._tp_edit.setEnabled(not checked)

    # ── Parsing ──────────────────────────────────────────────────────────

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
        tp = None if self._tp_all_check.isChecked() else self._parse_int_list(self._tp_edit.text())
        return DatasetConfig(
            ndtiff_path=self._ndtiff_edit.text().strip(),
            root_dir=root_dir,
            positions=self._parse_int_list(self._positions_edit.text()),
            timepoints=tp,
            xy_downsample=self._xy_spin.value(),
        )

    # ── Run ──────────────────────────────────────────────────────────────

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

        self._worker = _work()

    def _on_progress(self, update: tuple) -> None:
        pos, done, total, label = update
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        self._status_label.setText(f"pos{pos:02d} — {label} [{done}/{total}]")

    def _on_finished(self) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"Done — exported {self._n_positions} position(s).")
        self._worker = None
        self._log_viewer.refresh()

    def _on_error(self, exc: Exception) -> None:
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"Error: {exc}")
        self._worker = None
        self._log_viewer.refresh()
