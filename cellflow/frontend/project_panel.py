"""Project panel — fixed strip above the tab widget.

Shows the current project file, Load / Save buttons, dataset metadata
(pixel size, time interval, condition), and the tissue table that was
previously in the Database tab.  All metadata values are kept in sync
with ViewerState so every tab reads the same values without touching
this widget directly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .registry import ViewerState

logger = logging.getLogger(__name__)


class ProjectPanel(QWidget):
    """Panel fixed above the tab widget: file ops, metadata, tissue table."""

    show_tissue_requested = Signal(int)  # tissue_id

    def __init__(self, viewer, state: ViewerState):
        super().__init__()
        self.viewer = viewer
        self._state = state
        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)
        self.setLayout(layout)

        # ── Row 1: file operations ────────────────────────────────────
        file_row = QHBoxLayout()
        file_row.setSpacing(6)

        self._path_label = QLabel("No project")
        self._path_label.setStyleSheet("color: palette(mid); font-size: 9pt;")
        self._path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._path_label.setToolTip("Current project file (.h5)")
        file_row.addWidget(self._path_label)

        self._load_btn = QPushButton("Load…")
        self._load_btn.setFixedWidth(52)
        self._load_btn.setToolTip("Load a project from a .h5 file")
        file_row.addWidget(self._load_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setFixedWidth(44)
        self._save_btn.setToolTip("Save the current project to a .h5 file")
        file_row.addWidget(self._save_btn)

        self._save_as_btn = QPushButton("Save as…")
        self._save_as_btn.setFixedWidth(68)
        self._save_as_btn.setToolTip("Save to a new .h5 file")
        file_row.addWidget(self._save_as_btn)

        layout.addLayout(file_row)

        # ── Row 2: metadata ───────────────────────────────────────────
        meta_row = QHBoxLayout()
        meta_row.setSpacing(6)

        meta_row.addWidget(QLabel("px (µm):"))
        self._px_edit = QLineEdit()
        self._px_edit.setFixedWidth(56)
        self._px_edit.setPlaceholderText("—")
        self._px_edit.setToolTip("Pixel size in µm/px")
        meta_row.addWidget(self._px_edit)

        meta_row.addWidget(QLabel("dt (s):"))
        self._dt_edit = QLineEdit()
        self._dt_edit.setFixedWidth(56)
        self._dt_edit.setPlaceholderText("—")
        self._dt_edit.setToolTip("Time interval between frames in seconds")
        meta_row.addWidget(self._dt_edit)

        meta_row.addWidget(QLabel("Condition:"))
        self._condition_edit = QLineEdit()
        self._condition_edit.setPlaceholderText("e.g. WT, vim_KO")
        self._condition_edit.setToolTip("Experimental condition label")
        meta_row.addWidget(self._condition_edit)

        layout.addLayout(meta_row)

        # ── Dataset group ─────────────────────────────────────────────
        ds_group = QGroupBox("Dataset")
        ds_layout = QVBoxLayout()
        ds_layout.setSpacing(4)

        self._tissues_label = QLabel("No dataset")
        ds_layout.addWidget(self._tissues_label)

        self._tissue_table = QTableWidget(0, 4)
        self._tissue_table.setHorizontalHeaderLabels(["ID", "Frames", "T1s", "Note"])
        self._tissue_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self._tissue_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self._tissue_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self._tissue_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch
        )
        self._tissue_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._tissue_table.setSelectionMode(QTableWidget.SingleSelection)
        self._tissue_table.setMaximumHeight(150)
        ds_layout.addWidget(self._tissue_table)

        tissue_btn_row = QHBoxLayout()
        self._show_tissue_btn = QPushButton("Show in viewer")
        self._remove_tissue_btn = QPushButton("Remove")
        tissue_btn_row.addWidget(self._show_tissue_btn)
        tissue_btn_row.addWidget(self._remove_tissue_btn)
        tissue_btn_row.addStretch()

        self._dashboard_btn = QPushButton("Open Dashboard ↗")
        self._dashboard_btn.setToolTip("Launch the analysis dashboard in your browser")
        tissue_btn_row.addWidget(self._dashboard_btn)

        ds_layout.addLayout(tissue_btn_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        ds_layout.addWidget(self._status_label)

        ds_group.setLayout(ds_layout)
        layout.addWidget(ds_group)

    def _connect_signals(self):
        self._load_btn.clicked.connect(self._on_load)
        self._save_btn.clicked.connect(self._on_save)
        self._save_as_btn.clicked.connect(self._on_save_as)

        self._px_edit.editingFinished.connect(self._on_metadata_edited)
        self._dt_edit.editingFinished.connect(self._on_metadata_edited)
        self._condition_edit.editingFinished.connect(self._on_metadata_edited)

        self._show_tissue_btn.clicked.connect(self._show_selected)
        self._remove_tissue_btn.clicked.connect(self._remove_selected)
        self._dashboard_btn.clicked.connect(self._open_dashboard)
        self._tissue_table.cellChanged.connect(self._on_note_changed)

        self._state.metadata_changed.connect(self._sync_from_state)
        self._state.project_changed.connect(self._sync_path_label)
        self._state.dataset_changed.connect(self._refresh_table)

    # ------------------------------------------------------------------
    # State → UI sync
    # ------------------------------------------------------------------

    def _sync_from_state(self):
        """Refresh editable fields from ViewerState (without re-emitting)."""
        for widget in (self._px_edit, self._dt_edit, self._condition_edit):
            widget.blockSignals(True)

        px = self._state.pixel_size
        dt = self._state.time_interval
        self._px_edit.setText(str(px) if px is not None else "")
        self._dt_edit.setText(str(dt) if dt is not None else "")
        self._condition_edit.setText(self._state.condition)

        for widget in (self._px_edit, self._dt_edit, self._condition_edit):
            widget.blockSignals(False)

    def _sync_path_label(self):
        p = self._state.project_path
        if p:
            self._path_label.setText(Path(p).name)
            self._path_label.setToolTip(p)
        else:
            self._path_label.setText("No project")
            self._path_label.setToolTip("")

    def _refresh_table(self):
        ds = self._state.dataset
        self._tissue_table.blockSignals(True)
        self._tissue_table.setRowCount(0)

        if ds is None or ds.n_tissues == 0:
            self._tissues_label.setText("No dataset" if ds is None else "No tissues yet")
            self._tissue_table.blockSignals(False)
            return

        self._tissues_label.setText(f"{ds.n_tissues} tissue(s)")

        # Populate metadata fields from dataset if they are currently empty.
        if ds.condition and not self._condition_edit.text():
            self._condition_edit.setText(ds.condition)
            self._state.condition = ds.condition
        if ds.pixel_size is not None and not self._px_edit.text():
            self._px_edit.setText(str(ds.pixel_size))
            self._state.pixel_size = ds.pixel_size
        if ds.time_interval is not None and not self._dt_edit.text():
            self._dt_edit.setText(str(ds.time_interval))
            self._state.time_interval = ds.time_interval

        for tid in ds.tissue_ids:
            series = ds.tissues[tid]
            n_t1 = len(series.t1_events)
            note = series.metadata.get("note", "")
            row = self._tissue_table.rowCount()
            self._tissue_table.insertRow(row)

            id_item = QTableWidgetItem(str(tid))
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            self._tissue_table.setItem(row, 0, id_item)

            frames_item = QTableWidgetItem(str(series.num_frames))
            frames_item.setFlags(frames_item.flags() & ~Qt.ItemIsEditable)
            self._tissue_table.setItem(row, 1, frames_item)

            t1_item = QTableWidgetItem(str(n_t1))
            t1_item.setFlags(t1_item.flags() & ~Qt.ItemIsEditable)
            self._tissue_table.setItem(row, 2, t1_item)

            note_item = QTableWidgetItem(note)
            self._tissue_table.setItem(row, 3, note_item)

        self._tissue_table.blockSignals(False)

    # ------------------------------------------------------------------
    # UI → State sync
    # ------------------------------------------------------------------

    def _on_metadata_edited(self):
        px = _parse_float(self._px_edit.text())
        dt = _parse_float(self._dt_edit.text())
        condition = self._condition_edit.text().strip()
        self._state.pixel_size = px
        self._state.time_interval = dt
        self._state.condition = condition

        ds = self._state.dataset
        if ds is not None:
            if px is not None:
                ds.pixel_size = px
            if dt is not None:
                ds.time_interval = dt
            ds.condition = condition

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load CellFlow project", "", "CellFlow project (*.h5);;All files (*)"
        )
        if not path:
            return
        self._load_from_path(path)

    def _load_from_path(self, path: str):
        from ..utils.io import load_project
        try:
            result = load_project(path)
        except Exception as exc:
            logger.error("Failed to load project: %s", exc)
            self._path_label.setText(f"Error: {exc}")
            return

        self._state.pixel_size = result["pixel_size"]
        self._state.time_interval = result["time_interval"]
        self._state.condition = result["condition"]
        self._state.dataset = result["dataset"]
        self._state.project_path = path

        labels = result.get("labels")
        if labels is not None:
            self._load_labels_into_viewer(labels, Path(path).stem)

    def _load_labels_into_viewer(self, labels: np.ndarray, name: str):
        import napari.layers
        layer_name = f"{name}_labels"
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = labels
        else:
            self.viewer.add_labels(labels, name=layer_name)

    def _on_save(self):
        path = self._state.project_path
        if not path:
            self._on_save_as()
            return
        self._save_to_path(path)

    def _on_save_as(self):
        default = self._state.project_path or "project.h5"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CellFlow project", default,
            "CellFlow project (*.h5);;All files (*)"
        )
        if not path:
            return
        if not path.endswith(".h5"):
            path += ".h5"
        self._save_to_path(path)

    def _save_to_path(self, path: str):
        from ..utils.io import save_project
        labels = self._collect_labels()
        try:
            save_project(
                path,
                labels=labels,
                dataset=self._state.dataset,
                pixel_size=self._state.pixel_size,
                time_interval=self._state.time_interval,
                condition=self._state.condition,
            )
        except Exception as exc:
            logger.error("Failed to save project: %s", exc)
            self._path_label.setText(f"Save error: {exc}")
            return
        self._state.project_path = path

    def _collect_labels(self) -> Optional[np.ndarray]:
        import napari.layers
        active = self.viewer.layers.selection.active
        if active is not None and isinstance(active, napari.layers.Labels):
            return np.asarray(active.data)
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                return np.asarray(layer.data)
        return None

    # ------------------------------------------------------------------
    # Tissue table actions
    # ------------------------------------------------------------------

    def _get_selected_tid(self) -> Optional[int]:
        row = self._tissue_table.currentRow()
        if row < 0:
            return None
        id_item = self._tissue_table.item(row, 0)
        if id_item is None:
            return None
        return int(id_item.text())

    def _show_selected(self):
        tid = self._get_selected_tid()
        if tid is None:
            self._status_label.setText("Select a tissue row first.")
            return
        self.show_tissue_requested.emit(tid)

    def _remove_selected(self):
        tid = self._get_selected_tid()
        if tid is None:
            self._status_label.setText("Select a tissue row first.")
            return
        self._state.remove_tissue(tid)
        self._status_label.setText(f"Removed tissue {tid}.")

    def _on_note_changed(self, row: int, col: int):
        if col != 3:
            return
        ds = self._state.dataset
        if ds is None:
            return
        tid_item = self._tissue_table.item(row, 0)
        note_item = self._tissue_table.item(row, 3)
        if tid_item is None or note_item is None:
            return
        tid = int(tid_item.text())
        if tid in ds.tissues:
            ds.tissues[tid].metadata["note"] = note_item.text()

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _open_dashboard(self):
        ds = self._state.dataset
        if ds is None or ds.n_tissues == 0:
            self._status_label.setText("No dataset to open in dashboard.")
            return
        try:
            import tempfile
            tmp_h5 = Path(tempfile.mktemp(prefix="tissuegraph_dashboard_", suffix=".h5"))
            from ..utils.io import save_dataset
            save_dataset(ds, tmp_h5)

            import subprocess
            import sys
            subprocess.Popen([sys.executable, "-m", "cellflow.dashboard", str(tmp_h5)])
            self._status_label.setText("Dashboard launched in browser.")
        except ImportError:
            self._status_label.setText(
                "Dashboard requires dash+plotly. Install with: "
                "pip install cellflow[dashboard]"
            )
        except Exception as exc:
            self._status_label.setText(f"Failed to launch dashboard: {exc}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_float(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return None
