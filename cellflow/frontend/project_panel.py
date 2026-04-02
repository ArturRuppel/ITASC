"""Project panel — fixed strip above the tab widget.

Shows the current project file, Load / Save buttons, dataset metadata
(pixel size, time interval, condition), and the tissue table.

When a multi-file manifest (.cfproj) is loaded a collapsible "Project
files" section appears listing all .h5 files with the active one marked.

The "Add to dataset" / "Discard" buttons are enabled only when the Edge
Analysis pipeline has a finished result ready (state.preview_series is
not None).  Clicking "Add" commits the result to the dataset and signals
the analysis widget to clean up its visualization layers.
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
    QListWidget,
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
        self._manifest = None        # Optional[ProjectManifest]
        self._manifest_path: Optional[str] = None
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
        file_row.setSpacing(4)

        self._path_label = QLabel("No project")
        self._path_label.setStyleSheet("color: palette(mid); font-size: 9pt;")
        self._path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._path_label.setToolTip("Current project file")
        file_row.addWidget(self._path_label)

        self._load_btn = QPushButton("Load…")
        self._load_btn.setFixedWidth(52)
        self._load_btn.setToolTip("Load a .h5 project or .cfproj manifest")
        file_row.addWidget(self._load_btn)

        self._new_manifest_btn = QPushButton("New manifest…")
        self._new_manifest_btn.setToolTip(
            "Create a new multi-file manifest (.cfproj) that assembles several "
            ".h5 files into one logical dataset"
        )
        file_row.addWidget(self._new_manifest_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setFixedWidth(44)
        self._save_btn.setToolTip("Save the current project")
        file_row.addWidget(self._save_btn)

        self._save_as_btn = QPushButton("Save as…")
        self._save_as_btn.setFixedWidth(68)
        self._save_as_btn.setToolTip("Save to a new file")
        file_row.addWidget(self._save_as_btn)

        layout.addLayout(file_row)

        # ── Row 2: metadata ───────────────────────────────────────────
        meta_row = QHBoxLayout()
        meta_row.setSpacing(4)

        meta_row.addWidget(QLabel("px (µm):"))
        self._px_edit = QLineEdit()
        self._px_edit.setFixedWidth(52)
        self._px_edit.setPlaceholderText("—")
        self._px_edit.setToolTip("Pixel size in µm/px")
        meta_row.addWidget(self._px_edit)

        meta_row.addWidget(QLabel("dt (s):"))
        self._dt_edit = QLineEdit()
        self._dt_edit.setFixedWidth(52)
        self._dt_edit.setPlaceholderText("—")
        self._dt_edit.setToolTip("Time interval between frames in seconds")
        meta_row.addWidget(self._dt_edit)

        meta_row.addWidget(QLabel("Condition:"))
        self._condition_edit = QLineEdit()
        self._condition_edit.setPlaceholderText("e.g. WT")
        self._condition_edit.setToolTip("Experimental condition label")
        meta_row.addWidget(self._condition_edit)

        layout.addLayout(meta_row)

        # ── Project files (collapsible, hidden unless manifest loaded) ─
        self._files_toggle_btn = QPushButton("▶ Project files")
        self._files_toggle_btn.setCheckable(True)
        self._files_toggle_btn.setChecked(False)
        self._files_toggle_btn.setFlat(True)
        self._files_toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; padding: 2px 0; }"
        )
        self._files_toggle_btn.setVisible(False)
        layout.addWidget(self._files_toggle_btn)

        self._files_content = QWidget()
        files_cl = QVBoxLayout()
        files_cl.setContentsMargins(0, 0, 0, 0)
        files_cl.setSpacing(2)

        self._files_list = QListWidget()
        self._files_list.setMaximumHeight(100)
        self._files_list.setToolTip("Double-click a file to make it active")
        files_cl.addWidget(self._files_list)

        files_btn_row = QHBoxLayout()
        self._add_file_btn = QPushButton("Add .h5…")
        self._add_file_btn.setToolTip("Add an existing .h5 file to this manifest")
        self._remove_file_btn = QPushButton("Remove")
        self._remove_file_btn.setToolTip("Remove the selected file from this manifest")
        files_btn_row.addWidget(self._add_file_btn)
        files_btn_row.addWidget(self._remove_file_btn)
        files_btn_row.addStretch()
        files_cl.addLayout(files_btn_row)

        self._files_content.setLayout(files_cl)
        self._files_content.setVisible(False)
        layout.addWidget(self._files_content)

        # ── Dataset group ─────────────────────────────────────────────
        ds_group = QGroupBox("Dataset")
        ds_layout = QVBoxLayout()
        ds_layout.setContentsMargins(6, 4, 6, 4)
        ds_layout.setSpacing(3)

        # Header row: tissue count + Add / Discard / Dashboard
        ds_header = QHBoxLayout()
        ds_header.setSpacing(4)

        self._tissues_label = QLabel("No dataset")
        self._tissues_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        ds_header.addWidget(self._tissues_label)

        self._add_preview_btn = QPushButton("Add ✓")
        self._add_preview_btn.setFixedWidth(56)
        self._add_preview_btn.setEnabled(False)
        self._add_preview_btn.setToolTip("Add the finished analysis result to the dataset")
        ds_header.addWidget(self._add_preview_btn)

        self._discard_preview_btn = QPushButton("Discard ✗")
        self._discard_preview_btn.setFixedWidth(64)
        self._discard_preview_btn.setEnabled(False)
        self._discard_preview_btn.setToolTip("Discard the finished analysis result")
        ds_header.addWidget(self._discard_preview_btn)

        self._dashboard_btn = QPushButton("Dashboard ↗")
        self._dashboard_btn.setToolTip("Launch the analysis dashboard in your browser")
        ds_header.addWidget(self._dashboard_btn)

        ds_layout.addLayout(ds_header)

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
        self._tissue_table.setMaximumHeight(110)
        ds_layout.addWidget(self._tissue_table)

        tissue_btn_row = QHBoxLayout()
        tissue_btn_row.setSpacing(4)
        self._show_tissue_btn = QPushButton("Show in viewer")
        self._remove_tissue_btn = QPushButton("Remove")
        tissue_btn_row.addWidget(self._show_tissue_btn)
        tissue_btn_row.addWidget(self._remove_tissue_btn)
        tissue_btn_row.addStretch()
        ds_layout.addLayout(tissue_btn_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 8pt;")
        ds_layout.addWidget(self._status_label)

        ds_group.setLayout(ds_layout)
        layout.addWidget(ds_group)

    def _connect_signals(self):
        self._load_btn.clicked.connect(self._on_load)
        self._new_manifest_btn.clicked.connect(self._on_new_manifest)
        self._save_btn.clicked.connect(self._on_save)
        self._save_as_btn.clicked.connect(self._on_save_as)

        self._px_edit.editingFinished.connect(self._on_metadata_edited)
        self._dt_edit.editingFinished.connect(self._on_metadata_edited)
        self._condition_edit.editingFinished.connect(self._on_metadata_edited)

        self._files_toggle_btn.toggled.connect(self._on_files_toggled)
        self._files_list.itemDoubleClicked.connect(self._on_file_double_clicked)
        self._add_file_btn.clicked.connect(self._on_add_file)
        self._remove_file_btn.clicked.connect(self._on_remove_file)

        self._add_preview_btn.clicked.connect(self._on_add_preview)
        self._discard_preview_btn.clicked.connect(self._on_discard_preview)
        self._show_tissue_btn.clicked.connect(self._show_selected)
        self._remove_tissue_btn.clicked.connect(self._remove_selected)
        self._dashboard_btn.clicked.connect(self._open_dashboard)
        self._tissue_table.cellChanged.connect(self._on_note_changed)

        self._state.metadata_changed.connect(self._sync_from_state)
        self._state.project_changed.connect(self._sync_path_label)
        self._state.dataset_changed.connect(self._refresh_table)
        self._state.preview_changed.connect(self._on_preview_changed)

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
        if self._manifest_path:
            m_name = Path(self._manifest_path).name
            p = self._state.project_path
            h5_name = Path(p).name if p else "—"
            self._path_label.setText(f"{m_name}  ›  {h5_name}")
            self._path_label.setToolTip(
                f"Manifest: {self._manifest_path}\nActive: {p or '—'}"
            )
        else:
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

    def _on_preview_changed(self):
        has = self._state.preview_series is not None
        self._add_preview_btn.setEnabled(has)
        self._discard_preview_btn.setEnabled(has)
        if has:
            s = self._state.preview_series
            n_t1 = len(s.t1_events)
            n_trajs = len(s.edge_trajectories)
            self._add_preview_btn.setToolTip(
                f"Add analysis result to dataset\n"
                f"({s.num_frames} frames, {n_t1} T1s, {n_trajs} trajectories)"
            )
        else:
            self._add_preview_btn.setToolTip("No analysis result pending")

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
            self, "Load CellFlow project", "",
            "CellFlow files (*.h5 *.cfproj);;"
            "CellFlow project (*.h5);;"
            "CellFlow manifest (*.cfproj);;"
            "All files (*)",
        )
        if not path:
            return
        if path.endswith(".cfproj"):
            self._load_manifest_from_path(path)
        else:
            self._manifest = None
            self._manifest_path = None
            self._files_toggle_btn.setVisible(False)
            self._files_content.setVisible(False)
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
        self._state.project_path = path  # triggers _sync_path_label

        labels = result.get("labels")
        if labels is not None:
            self._load_labels_into_viewer(labels, Path(path).stem)

    def _load_manifest_from_path(self, path: str):
        from ..utils.io import load_manifest
        try:
            manifest = load_manifest(path)
        except Exception as exc:
            logger.error("Failed to load manifest: %s", exc)
            self._status_label.setText(f"Error loading manifest: {exc}")
            return

        self._manifest = manifest
        self._manifest_path = path
        self._files_toggle_btn.setVisible(True)
        self._update_files_toggle_label()

        active = manifest.active_entry
        if active:
            h5_path = self._resolve_entry_path(active.path)
            if Path(h5_path).exists():
                self._load_from_path(h5_path)
                return
            self._status_label.setText(f"Active file not found: {h5_path}")

        self._sync_path_label()

    def _load_labels_into_viewer(self, labels: np.ndarray, name: str):
        import napari.layers
        layer_name = f"{name}_labels"
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = labels
        else:
            self.viewer.add_labels(labels, name=layer_name)

    def _on_new_manifest(self):
        current_h5 = self._state.project_path
        default_dir = str(Path(current_h5).parent) if current_h5 else ""
        default = str(Path(default_dir) / "project.cfproj") if default_dir else "project.cfproj"
        path, _ = QFileDialog.getSaveFileName(
            self, "Create new manifest", default,
            "CellFlow manifest (*.cfproj);;All files (*)",
        )
        if not path:
            return
        if not path.endswith(".cfproj"):
            path += ".cfproj"

        from ..utils.io import ProjectEntry, ProjectManifest, save_manifest
        entries = []
        if current_h5 and current_h5.endswith(".h5"):
            entries.append(ProjectEntry(
                path=_make_entry_path(current_h5, path),
                display_name=Path(current_h5).stem,
            ))

        manifest = ProjectManifest(entries=entries, active_index=0)
        try:
            save_manifest(path, manifest)
        except Exception as exc:
            logger.error("Failed to save manifest: %s", exc)
            self._status_label.setText(f"Error saving manifest: {exc}")
            return

        self._manifest = manifest
        self._manifest_path = path
        self._files_toggle_btn.setVisible(True)
        self._update_files_toggle_label()
        self._sync_path_label()

    def _on_save(self):
        path = self._state.project_path
        if not path:
            self._on_save_as()
            return
        self._save_to_path(path)
        self._save_manifest_if_active()

    def _on_save_as(self):
        if self._manifest_path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save manifest as", self._manifest_path,
                "CellFlow manifest (*.cfproj);;All files (*)",
            )
            if not path:
                return
            if not path.endswith(".cfproj"):
                path += ".cfproj"
            self._manifest_path = path
            self._save_manifest_if_active()
            self._sync_path_label()
        else:
            default = self._state.project_path or "project.h5"
            path, _ = QFileDialog.getSaveFileName(
                self, "Save CellFlow project", default,
                "CellFlow project (*.h5);;All files (*)",
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

    def _save_manifest_if_active(self):
        if self._manifest is None or self._manifest_path is None:
            return
        from ..utils.io import save_manifest
        try:
            save_manifest(self._manifest_path, self._manifest)
        except Exception as exc:
            logger.error("Failed to save manifest: %s", exc)
            self._status_label.setText(f"Manifest save error: {exc}")

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
    # Add / Discard pending analysis
    # ------------------------------------------------------------------

    def _on_add_preview(self):
        series = self._state.preview_series
        if series is None:
            return
        self._state.ensure_dataset(
            condition=self._state.condition,
            pixel_size=self._state.pixel_size,
            time_interval=self._state.time_interval,
        )
        tid = self._state.add_tissue(series)
        self._state.preview_series = None   # triggers cleanup in analysis widget
        self._status_label.setText(f"Added tissue {tid} to dataset.")

    def _on_discard_preview(self):
        if self._state.preview_series is None:
            return
        self._state.preview_series = None   # triggers cleanup in analysis widget
        self._status_label.setText("Analysis result discarded.")

    # ------------------------------------------------------------------
    # Manifest / project-files list
    # ------------------------------------------------------------------

    def _on_files_toggled(self, checked: bool):
        self._files_content.setVisible(checked)
        if checked:
            self._refresh_files_list()
        self._update_files_toggle_label()

    def _update_files_toggle_label(self):
        checked = self._files_toggle_btn.isChecked()
        arrow = "▼" if checked else "▶"
        if self._manifest and self._manifest.entries:
            n = len(self._manifest.entries)
            active_path = self._manifest.entries[self._manifest.active_index].path
            active_name = Path(active_path).name
            suffix = f" ({n} file{'s' if n != 1 else ''}, {active_name} active)"
        else:
            suffix = " (empty)"
        self._files_toggle_btn.setText(f"{arrow} Project files{suffix}")

    def _refresh_files_list(self):
        self._files_list.clear()
        if self._manifest is None:
            return
        for i, entry in enumerate(self._manifest.entries):
            active = (i == self._manifest.active_index)
            prefix = "►  " if active else "    "
            label = f"{prefix}{Path(entry.path).name}"
            if entry.display_name and entry.display_name != Path(entry.path).stem:
                label += f"  —  {entry.display_name}"
            self._files_list.addItem(label)
            if active:
                self._files_list.setCurrentRow(i)

    def _on_file_double_clicked(self, item):
        row = self._files_list.row(item)
        self._switch_active_file(row)

    def _switch_active_file(self, index: int):
        if self._manifest is None or not (0 <= index < len(self._manifest.entries)):
            return
        self._manifest.active_index = index
        self._refresh_files_list()
        self._update_files_toggle_label()
        self._save_manifest_if_active()

        entry = self._manifest.entries[index]
        h5_path = self._resolve_entry_path(entry.path)
        if Path(h5_path).exists():
            self._load_from_path(h5_path)
        else:
            self._status_label.setText(f"File not found: {h5_path}")
            self._sync_path_label()

    def _on_add_file(self):
        if self._manifest is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Add .h5 file to manifest", "",
            "CellFlow project (*.h5);;All files (*)",
        )
        if not path:
            return
        from ..utils.io import ProjectEntry
        entry = ProjectEntry(
            path=_make_entry_path(path, self._manifest_path),
            display_name=Path(path).stem,
        )
        self._manifest.entries.append(entry)
        self._refresh_files_list()
        self._update_files_toggle_label()
        self._save_manifest_if_active()

    def _on_remove_file(self):
        if self._manifest is None:
            return
        row = self._files_list.currentRow()
        if row < 0:
            self._status_label.setText("Select a file to remove.")
            return
        self._manifest.entries.pop(row)
        if self._manifest.active_index >= len(self._manifest.entries):
            self._manifest.active_index = max(0, len(self._manifest.entries) - 1)
        self._refresh_files_list()
        self._update_files_toggle_label()
        self._save_manifest_if_active()

    def _resolve_entry_path(self, entry_path: str) -> str:
        p = Path(entry_path)
        if p.is_absolute():
            return str(p)
        if self._manifest_path:
            return str(Path(self._manifest_path).parent / p)
        return str(p)

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


def _make_entry_path(h5_path: str, manifest_path: Optional[str]) -> str:
    """Return *h5_path* relative to the manifest directory, or absolute as fallback."""
    if not manifest_path:
        return h5_path
    try:
        return str(Path(h5_path).relative_to(Path(manifest_path).parent))
    except ValueError:
        return h5_path
