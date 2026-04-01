"""Dataset management tab widget for cellflow."""
import logging
from pathlib import Path
from typing import Optional

from qtpy.QtCore import QThread, Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .registry import get_state
from .workers import IOWorker

logger = logging.getLogger(__name__)


class DataBankWidget(QWidget):
    """Dataset management tab: tissue table, metadata, save/load/new, dashboard."""

    show_tissue_requested = Signal(int)  # tissue_id

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._state = get_state(napari_viewer)
        self._thread: Optional[QThread] = None
        self._worker = None
        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        _container = QWidget()
        layout = QVBoxLayout(_container)
        scroll.setWidget(_container)

        # --- Dataset metadata ---
        meta_group = QGroupBox("Dataset metadata")
        meta_layout = QVBoxLayout()

        cond_row = QHBoxLayout()
        cond_row.addWidget(QLabel("Condition:"))
        self.condition_edit = QLineEdit()
        self.condition_edit.setPlaceholderText("e.g. WT, vim_KO")
        cond_row.addWidget(self.condition_edit)
        meta_layout.addLayout(cond_row)

        px_row = QHBoxLayout()
        px_row.addWidget(QLabel("Pixel size (\u00b5m/px):"))
        self.pixel_size_edit = QLineEdit()
        self.pixel_size_edit.setPlaceholderText("optional")
        px_row.addWidget(self.pixel_size_edit)
        meta_layout.addLayout(px_row)

        dt_row = QHBoxLayout()
        dt_row.addWidget(QLabel("Time interval (s):"))
        self.time_interval_edit = QLineEdit()
        self.time_interval_edit.setPlaceholderText("optional")
        dt_row.addWidget(self.time_interval_edit)
        meta_layout.addLayout(dt_row)

        meta_group.setLayout(meta_layout)
        layout.addWidget(meta_group)

        # --- Dataset actions ---
        ds_btn_row = QHBoxLayout()
        self.new_btn = QPushButton("New")
        self.load_btn = QPushButton("Load...")
        self.save_btn = QPushButton("Save...")
        ds_btn_row.addWidget(self.new_btn)
        ds_btn_row.addWidget(self.load_btn)
        ds_btn_row.addWidget(self.save_btn)
        layout.addLayout(ds_btn_row)

        # --- Tissue table ---
        self.tissues_label = QLabel("No dataset")
        layout.addWidget(self.tissues_label)

        self.tissue_table = QTableWidget(0, 4)
        self.tissue_table.setHorizontalHeaderLabels(["ID", "Frames", "T1s", "Note"])
        self.tissue_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.tissue_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.tissue_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.tissue_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch
        )
        self.tissue_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.tissue_table.setSelectionMode(QTableWidget.SingleSelection)
        layout.addWidget(self.tissue_table)

        tissue_btn_row = QHBoxLayout()
        self.show_tissue_btn = QPushButton("Show in viewer")
        self.remove_tissue_btn = QPushButton("Remove")
        tissue_btn_row.addWidget(self.show_tissue_btn)
        tissue_btn_row.addWidget(self.remove_tissue_btn)
        layout.addLayout(tissue_btn_row)

        # --- Dashboard ---
        self.open_dashboard_btn = QPushButton("Open Dashboard \u2197")
        self.open_dashboard_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }"
        )
        self.open_dashboard_btn.setToolTip("Launch the analysis dashboard in your browser")
        layout.addWidget(self.open_dashboard_btn)

        # --- Progress + status ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        layout.addWidget(self.cancel_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

    def _connect_signals(self):
        self.new_btn.clicked.connect(self._new_dataset)
        self.load_btn.clicked.connect(self._load_dataset)
        self.save_btn.clicked.connect(self._save_dataset)
        self.show_tissue_btn.clicked.connect(self._show_selected)
        self.remove_tissue_btn.clicked.connect(self._remove_selected)
        self.open_dashboard_btn.clicked.connect(self._open_dashboard)
        self.cancel_btn.clicked.connect(self._cancel_io_worker)

        self.condition_edit.editingFinished.connect(self._apply_metadata)
        self.pixel_size_edit.editingFinished.connect(self._apply_metadata)
        self.time_interval_edit.editingFinished.connect(self._apply_metadata)

        self.tissue_table.cellChanged.connect(self._on_note_changed)

        self._state.dataset_changed.connect(self._on_dataset_changed)

    # ------------------------------------------------------------------
    # Dataset change handler
    # ------------------------------------------------------------------

    def _on_dataset_changed(self):
        self._refresh_table()

    def _refresh_table(self):
        ds = self._state.dataset
        self.tissue_table.blockSignals(True)
        self.tissue_table.setRowCount(0)

        if ds is None or ds.n_tissues == 0:
            self.tissues_label.setText("No dataset" if ds is None else "No tissues yet")
            self.tissue_table.blockSignals(False)
            return

        self.tissues_label.setText(f"{ds.n_tissues} tissue(s)")

        # Populate metadata fields if they are empty (e.g. after first add or load)
        if ds.condition and not self.condition_edit.text():
            self.condition_edit.setText(ds.condition)
        if ds.pixel_size is not None and not self.pixel_size_edit.text():
            self.pixel_size_edit.setText(str(ds.pixel_size))
        if ds.time_interval is not None and not self.time_interval_edit.text():
            self.time_interval_edit.setText(str(ds.time_interval))

        for tid in ds.tissue_ids:
            series = ds.tissues[tid]
            n_t1 = len(series.t1_events)
            note = series.metadata.get("note", "")
            row = self.tissue_table.rowCount()
            self.tissue_table.insertRow(row)

            id_item = QTableWidgetItem(str(tid))
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            self.tissue_table.setItem(row, 0, id_item)

            frames_item = QTableWidgetItem(str(series.num_frames))
            frames_item.setFlags(frames_item.flags() & ~Qt.ItemIsEditable)
            self.tissue_table.setItem(row, 1, frames_item)

            t1_item = QTableWidgetItem(str(n_t1))
            t1_item.setFlags(t1_item.flags() & ~Qt.ItemIsEditable)
            self.tissue_table.setItem(row, 2, t1_item)

            note_item = QTableWidgetItem(note)
            self.tissue_table.setItem(row, 3, note_item)

        self.tissue_table.blockSignals(False)

    def _on_note_changed(self, row: int, col: int):
        if col != 3:
            return
        ds = self._state.dataset
        if ds is None:
            return
        tid_item = self.tissue_table.item(row, 0)
        note_item = self.tissue_table.item(row, 3)
        if tid_item is None or note_item is None:
            return
        tid = int(tid_item.text())
        if tid in ds.tissues:
            ds.tissues[tid].metadata["note"] = note_item.text()

    def _apply_metadata(self):
        """Write edited metadata fields back to the dataset."""
        ds = self._state.dataset
        if ds is None:
            return
        ds.condition = self.condition_edit.text().strip()
        px = self._parse_float(self.pixel_size_edit.text())
        dt = self._parse_float(self.time_interval_edit.text())
        if px is not None:
            ds.pixel_size = px
        if dt is not None:
            ds.time_interval = dt

    # ------------------------------------------------------------------
    # Tissue actions
    # ------------------------------------------------------------------

    def _get_selected_tid(self) -> Optional[int]:
        row = self.tissue_table.currentRow()
        if row < 0:
            return None
        id_item = self.tissue_table.item(row, 0)
        if id_item is None:
            return None
        return int(id_item.text())

    def _show_selected(self):
        tid = self._get_selected_tid()
        if tid is None:
            self.status_label.setText("Select a tissue row first.")
            return
        self.show_tissue_requested.emit(tid)

    def _remove_selected(self):
        tid = self._get_selected_tid()
        if tid is None:
            self.status_label.setText("Select a tissue row first.")
            return
        self._state.remove_tissue(tid)
        self.status_label.setText(f"Removed tissue {tid}.")

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------

    def _new_dataset(self):
        ds = self._state.dataset
        if ds is not None and ds.n_tissues > 0:
            reply = QMessageBox.question(
                self, "New Dataset",
                f"Discard current dataset ({ds.n_tissues} tissue(s))?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        self._state.dataset = None
        self.condition_edit.clear()
        self.pixel_size_edit.clear()
        self.time_interval_edit.clear()
        self.status_label.setText("New dataset created.")

    def _load_dataset(self):
        ds = self._state.dataset
        if ds is not None and ds.n_tissues > 0:
            reply = QMessageBox.question(
                self, "Load Dataset",
                f"Replace current dataset ({ds.n_tissues} tissue(s)) with loaded one?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        path = QFileDialog.getExistingDirectory(self, "Load Dataset From")
        if not path:
            return
        self._run_io_worker(IOWorker("load", path), self._on_load_finished)

    def _on_load_finished(self, dataset):
        self._state.dataset = dataset
        self._finish_io_worker()

        # Populate metadata fields from loaded dataset
        self.condition_edit.setText(dataset.condition or "")
        if dataset.pixel_size is not None:
            self.pixel_size_edit.setText(str(dataset.pixel_size))
        else:
            self.pixel_size_edit.clear()
        if dataset.time_interval is not None:
            self.time_interval_edit.setText(str(dataset.time_interval))
        else:
            self.time_interval_edit.clear()

        self.status_label.setText(
            f"Loaded: {dataset.n_tissues} tissue(s), "
            f"condition: {dataset.condition or '(none)'}"
        )

    def _save_dataset(self):
        ds = self._state.dataset
        if ds is None or ds.n_tissues == 0:
            self.status_label.setText("No dataset to save.")
            return
        self._apply_metadata()
        path = QFileDialog.getExistingDirectory(self, "Save Dataset To")
        if not path:
            return
        self._run_io_worker(IOWorker("save", path, ds), self._on_save_finished)

    def _on_save_finished(self, _):
        self._finish_io_worker()
        self.status_label.setText("Dataset saved.")

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _open_dashboard(self):
        ds = self._state.dataset
        if ds is None or ds.n_tissues == 0:
            self.status_label.setText("No dataset to open in dashboard.")
            return
        try:
            import tempfile
            tmpdir = Path(tempfile.mkdtemp(prefix="tissuegraph_dashboard_"))
            from ..utils.io import save_dataset
            save_dataset(ds, tmpdir)

            import subprocess
            import sys
            subprocess.Popen(
                [sys.executable, "-m", "cellflow.dashboard", str(tmpdir)],
            )
            self.status_label.setText("Dashboard launched in browser.")
        except ImportError:
            self.status_label.setText(
                "Dashboard requires dash+plotly. Install with: "
                "pip install cellflow[dashboard]"
            )
        except Exception as exc:
            self.status_label.setText(f"Failed to launch dashboard: {exc}")

    # ------------------------------------------------------------------
    # IO worker management
    # ------------------------------------------------------------------

    def _run_io_worker(self, worker, on_finished):
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    self._thread.quit()
                    self._thread.wait()
            except RuntimeError:
                pass
            self._thread = None
            self._worker = None

        self._thread = QThread()
        self._worker = worker
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        if hasattr(self._worker, "progress"):
            self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self.new_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.cancel_btn.setVisible(True)
        self.status_label.setText("Working...")

        self._thread.start()

    def _finish_io_worker(self):
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.new_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

    def _on_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _on_error(self, exc):
        self._finish_io_worker()
        self.status_label.setText(f"Error: {exc}")
        logger.exception("IO operation failed", exc_info=exc)

    def _cancel_io_worker(self):
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    self._thread.quit()
                    self._thread.wait()
            except RuntimeError:
                pass
            self._thread = None
            self._worker = None
        self._finish_io_worker()
        self.status_label.setText("Cancelled.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_float(text: str):
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
