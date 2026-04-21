"""Project panel — fixed strip above the tab widget.

Three sections:

  Metadata  — pixel size, time interval, condition (always visible)
  Project   — collapsible; project open/new, import, pipeline file tracker
  Dataset   — catalog of saved tissues with summary table
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .registry import CatalogEntry, DatasetCatalog, TissueData, ViewerState
from .widgets import CollapsibleSection, _PipelineFileRow, _file_info

logger = logging.getLogger(__name__)

# Pipeline files to track, grouped by display stage.
# Each entry: (path_relative_to_pos_dir, display_name, loadable)
#   loadable = "image" | "labels" | None  (None → no Load button)
_TRACKED_FILE_GROUPS: list[tuple[str, list[tuple[str, str, "str | None"]]]] = [
    ("Input Export", [
        ("0_input/nucleus",                  "Nucleus 3D (frames)", None),
        ("0_input/nucleus/nucleus_zavg.tif", "Nucleus avg",         "image"),
        ("0_input/cell",                     "Cell 3D (frames)",    None),
        ("0_input/cell/cell_zavg.tif",       "Cell avg",            "image"),
    ]),
    ("Cellpose Nuclei", [
        ("1_cellpose/nucleus",               "Output directory",   None),
    ]),
    ("Cellpose Cells", [
        ("1_cellpose/cell/cell_dp.tif",          "Cell DP (z-slices)", None),
        ("1_cellpose/cell/cell_prob.tif",         "Cell prob (z-slices)", "image"),
        ("1_cellpose/cell/cell_dp_zavg.tif",     "Cell DP avg",        None),
        ("1_cellpose/cell/cell_prob_zavg.tif",   "Cell prob avg",      "image"),
    ]),
    ("Ultrack", [
        ("2_ultrack/foreground.tif",         "Foreground",         "image"),
        ("2_ultrack/contours.tif",           "Contours",           "image"),
        ("2_ultrack/data.db",                "Ultrack DB",         None),
        ("2_ultrack/tracks.csv",             "Tracks CSV",         None),
        ("2_ultrack/tracked_labels.tif",     "Tracked labels",     None),
        ("2_ultrack/nuclear_labels_2d.tif",  "Nuclear labels 2D",  "labels"),
    ]),
    ("Correction", [
        ("3_correction/nuclear_labels_corrected.tif", "Corrected labels", "labels"),
    ]),
    ("Cell Segmentation", [
        ("4_cell_segmentation/cell_labels.tif", "Cell labels", "labels"),
    ]),
    ("Analysis", [
        ("5_analysis/graph.h5",     "Graph",    None),
        ("5_analysis/topology.npz", "Topology", None),
    ]),
]


class ProjectPanel(QWidget):
    """Panel fixed above the tab widget: pipeline file tracker, metadata, dataset table."""

    def __init__(self, viewer, state: ViewerState):
        super().__init__()
        self.viewer = viewer
        self._state = state
        self._build_ui()
        self._connect_signals()
        self._refresh_catalog()
        self._sync_from_state()
        self._refresh_pipeline_status()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)
        self.setLayout(root)

        # ── Metadata row (always visible at top) ──────────────────────
        meta_row = QHBoxLayout()
        meta_row.setSpacing(4)
        meta_row.addWidget(QLabel("px (µm):"))
        self._px_edit = QLineEdit()
        self._px_edit.setFixedWidth(52)
        self._px_edit.setPlaceholderText("—")
        self._px_edit.setToolTip("Pixel size in µm/px")
        meta_row.addWidget(self._px_edit)
        meta_row.addWidget(QLabel("dt (min):"))
        self._dt_edit = QLineEdit()
        self._dt_edit.setFixedWidth(52)
        self._dt_edit.setPlaceholderText("—")
        self._dt_edit.setToolTip("Time interval between frames in minutes")
        meta_row.addWidget(self._dt_edit)
        meta_row.addWidget(QLabel("Condition:"))
        self._condition_edit = QLineEdit()
        self._condition_edit.setPlaceholderText("e.g. WT")
        self._condition_edit.setToolTip("Experimental condition label")
        meta_row.addWidget(self._condition_edit)
        meta_row.addWidget(QLabel("Pos:"))
        self._pipeline_pos_spin = QSpinBox()
        self._pipeline_pos_spin.setRange(0, 999)
        self._pipeline_pos_spin.setValue(0)
        self._pipeline_pos_spin.setFixedWidth(54)
        self._pipeline_pos_spin.setToolTip("Current pipeline position (shared across all widgets)")
        meta_row.addWidget(self._pipeline_pos_spin)
        root.addLayout(meta_row)

        # ── Project section (merged: project init + data state) ───────
        self._project_inner = QWidget()
        self._project_inner.setStyleSheet(
            "QLabel { color: white; } "
            "QPushButton { color: white; } "
            "QSpinBox { color: white; }"
        )
        pg_layout = QVBoxLayout(self._project_inner)
        pg_layout.setContentsMargins(4, 2, 0, 4)
        pg_layout.setSpacing(3)

        # Project buttons — parented here but placed in CellFlowWidget's top row
        self._new_project_btn = QPushButton("New Project…")
        self._new_project_btn.setToolTip("Create a new pipeline project directory")
        self._open_project_btn = QPushButton("Open Project…")
        self._open_project_btn.setToolTip("Open an existing pipeline project directory")
        # Project path label stays in the collapsible section
        self._project_dir_label = QLabel("[no project]")
        self._project_dir_label.setStyleSheet("font-size: 9pt;")
        self._project_dir_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._project_dir_label.setMinimumWidth(0)
        pg_layout.addWidget(self._project_dir_label)

        # Row: refresh button
        refresh_row = QHBoxLayout()
        refresh_row.setSpacing(4)
        refresh_row.addStretch()
        self._refresh_files_btn = QPushButton("↺ Refresh")
        self._refresh_files_btn.setFixedWidth(76)
        self._refresh_files_btn.setToolTip("Refresh pipeline file status")
        refresh_row.addWidget(self._refresh_files_btn)
        pg_layout.addLayout(refresh_row)

        # ── Pipeline file status rows (scrollable) ────────────────────
        files_container = QWidget()
        files_vlayout = QVBoxLayout(files_container)
        files_vlayout.setContentsMargins(0, 0, 0, 0)
        files_vlayout.setSpacing(0)

        self._pipeline_file_rows: list = []

        for group_name, entries in _TRACKED_FILE_GROUPS:
            grp_lbl = QLabel(group_name)
            grp_lbl.setStyleSheet(
                "font-size: 8pt; font-weight: bold; padding: 2px 4px 1px 4px; "
                "background: palette(alternateBase); color: white;"
            )
            files_vlayout.addWidget(grp_lbl)
            for rel_path, display_name, loadable in entries:
                row = _PipelineFileRow(rel_path, display_name, loadable)
                self._pipeline_file_rows.append(row)
                files_vlayout.addWidget(row)

        files_vlayout.addStretch()

        files_scroll = QScrollArea()
        files_scroll.setWidget(files_container)
        files_scroll.setWidgetResizable(True)
        files_scroll.setMinimumHeight(100)
        files_scroll.setFrameShape(QFrame.NoFrame)
        pg_layout.addWidget(files_scroll)

        self._project_section = CollapsibleSection(
            "Data Overview", self._project_inner, expanded=False
        )
        # Visually distinct from processing stage widgets: blue-tinted header
        self._project_section._toggle.setStyleSheet(
            "QToolButton { font-weight: bold; font-size: 10pt; border: none; "
            "padding: 2px; color: #7db8e8; }"
        )
        root.addWidget(self._project_section)

        # ── Dataset section ───────────────────────────────────────────
        ds_inner = QWidget()
        ds_layout = QVBoxLayout(ds_inner)
        ds_layout.setContentsMargins(6, 4, 6, 4)
        ds_layout.setSpacing(3)

        # Header: path label + Load / Save
        ds_path_row = QHBoxLayout()
        ds_path_row.setSpacing(4)
        self._catalog_path_label = QLabel("[no dataset]")
        self._catalog_path_label.setStyleSheet("font-size: 9pt;")
        self._catalog_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        ds_path_row.addWidget(self._catalog_path_label)
        self._ds_load_btn = QPushButton("Load dataset…")
        self._ds_load_btn.setToolTip("Load a .cfproj dataset file")
        ds_path_row.addWidget(self._ds_load_btn)
        self._ds_save_btn = QPushButton("Save dataset…")
        self._ds_save_btn.setToolTip("Save the dataset catalog to a .cfproj file")
        ds_path_row.addWidget(self._ds_save_btn)
        ds_layout.addLayout(ds_path_row)

        # Entry count toggle label
        self._catalog_count_label = QLabel("▶ 0 entries")
        self._catalog_count_label.setStyleSheet("font-weight: bold; font-size: 9pt;")
        ds_layout.addWidget(self._catalog_count_label)

        # Summary table
        self._catalog_table = QTableWidget(0, 5)
        self._catalog_table.setHorizontalHeaderLabels(
            ["Name", "Frames", "Cells", "T1s", "Condition"]
        )
        hh = self._catalog_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.Stretch)
        self._catalog_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._catalog_table.setSelectionMode(QTableWidget.SingleSelection)
        self._catalog_table.setMinimumHeight(60)
        ds_layout.addWidget(self._catalog_table)

        # Action buttons
        ds_btn_row = QHBoxLayout()
        ds_btn_row.setSpacing(4)
        self._show_in_viewer_btn = QPushButton("Load to widget")
        self._show_in_viewer_btn.setToolTip(
            "Load the selected entry into the working tissue, then push labels to viewer"
        )
        self._remove_from_ds_btn = QPushButton("Remove from dataset")
        self._remove_from_ds_btn.setToolTip("Remove the selected entry from the catalog")
        self._dashboard_btn = QPushButton("Dashboard ↗")
        self._dashboard_btn.setToolTip("Launch the analysis dashboard in your browser")
        ds_btn_row.addWidget(self._show_in_viewer_btn)
        ds_btn_row.addWidget(self._remove_from_ds_btn)
        ds_btn_row.addStretch()
        ds_btn_row.addWidget(self._dashboard_btn)
        ds_layout.addLayout(ds_btn_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 8pt;")
        ds_layout.addWidget(self._status_label)

        # Wrap in collapsible section — placed at the bottom of the plugin by analysis_widget.py
        self.dataset_widget = CollapsibleSection("Dataset", ds_inner, expanded=False)

    def _connect_signals(self):
        # Pipeline project buttons
        self._new_project_btn.clicked.connect(self._on_new_project)
        self._open_project_btn.clicked.connect(self._on_open_project)
        self._state.pipeline_schema_changed.connect(self._refresh_pipeline_status)

        # Auto-refresh pipeline status every 3 s when a project is open
        self._pipeline_timer = QTimer(self)
        self._pipeline_timer.setInterval(3000)
        self._pipeline_timer.timeout.connect(self._refresh_pipeline_status)
        self._pipeline_timer.start()

        # Project section toggle + pipeline file controls
        self._project_section._toggle.toggled.connect(self._on_project_section_toggled)
        self._pipeline_pos_spin.valueChanged.connect(self._on_pos_spin_changed)
        self._refresh_files_btn.clicked.connect(self._refresh_pipeline_files)
        self._state.pipeline_schema_changed.connect(self._refresh_pipeline_files)
        self._state.position_changed.connect(self._on_state_position_changed)

        # Metadata
        self._px_edit.editingFinished.connect(self._on_metadata_edited)
        self._dt_edit.editingFinished.connect(self._on_metadata_edited)
        self._condition_edit.editingFinished.connect(self._on_metadata_edited)

        # Dataset buttons
        self._ds_load_btn.clicked.connect(self._on_load_catalog)
        self._ds_save_btn.clicked.connect(self._on_save_catalog)
        self._show_in_viewer_btn.clicked.connect(self._on_show_in_viewer)
        self._remove_from_ds_btn.clicked.connect(self._on_remove_from_catalog)
        self._dashboard_btn.clicked.connect(self._on_open_dashboard)

        # State signals
        self._state.catalog_changed.connect(self._refresh_catalog)
        self._state.metadata_changed.connect(self._sync_from_state)
        # legacy compat
        self._state.dataset_changed.connect(self._refresh_catalog)

    # ------------------------------------------------------------------
    # Pipeline project handlers
    # ------------------------------------------------------------------

    def _on_new_project(self) -> None:
        from .new_project_dialog import NewProjectDialog
        dlg = NewProjectDialog(self.viewer, self._state, parent=self)
        dlg.exec_()

    def _on_open_project(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Open Pipeline Project Directory"
        )
        if d:
            self._state.set_project_dir(d)

    def _on_pos_spin_changed(self, value: int) -> None:
        self._state.current_position = value
        self._refresh_pipeline_files()

    def _on_state_position_changed(self) -> None:
        pos = self._state.current_position
        if self._pipeline_pos_spin.value() != pos:
            self._pipeline_pos_spin.setValue(pos)

    def _refresh_pipeline_status(self) -> None:
        """Update the project path label and refresh pipeline file status."""
        project_dir = self._state.project_dir
        if project_dir is None:
            self._project_dir_label.setText("[no project]")
        else:
            short = "/".join(project_dir.parts[-2:]) if len(project_dir.parts) >= 2 else str(project_dir)
            self._project_dir_label.setText(short)
            self._project_dir_label.setToolTip(str(project_dir))
        self._refresh_pipeline_files()

    # ------------------------------------------------------------------
    # Tissue toggle + pipeline file status
    # ------------------------------------------------------------------

    def _on_project_section_toggled(self, checked: bool) -> None:
        if checked:
            self._refresh_pipeline_files()

    def _refresh_pipeline_files(self) -> None:
        """Check on-disk presence and metadata for every tracked pipeline file."""
        if not self._project_inner.isVisible():
            return

        project_dir = self._state.project_dir
        if project_dir is None:
            for row in self._pipeline_file_rows:
                row.set_no_project()
            return

        pos = self._pipeline_pos_spin.value()
        pos_path = project_dir / f"pos{pos:02d}"

        for row in self._pipeline_file_rows:
            full_path = pos_path / row._rel_path
            if full_path.exists():
                info = _file_info(full_path)
                row.set_present(info)
                row._full_path = full_path
                if row._load_btn is not None:
                    try:
                        row._load_btn.clicked.disconnect()
                    except Exception:
                        pass
                    loadable = row._loadable
                    p = full_path
                    row._load_btn.clicked.connect(
                        lambda _, p=p, lt=loadable: self._load_pipeline_file(p, lt)
                    )
            else:
                row.set_missing()

    def _load_pipeline_file(self, path: Path, layer_type: str) -> None:
        """Load a pipeline TIFF file into the napari viewer."""
        try:
            import tifffile
            data = tifffile.imread(str(path))
        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)
            return
        name = path.name
        if layer_type == "labels":
            if name in self.viewer.layers:
                self.viewer.layers[name].data = data
            else:
                self.viewer.add_labels(data, name=name)
        else:
            if name in self.viewer.layers:
                self.viewer.layers[name].data = data
            else:
                self.viewer.add_image(data, name=name)

    # ------------------------------------------------------------------
    # State → UI
    # ------------------------------------------------------------------

    def _refresh_catalog(self):
        catalog = self._state.catalog
        # also reflect legacy dataset if catalog is empty
        entries = catalog.entries

        # path label
        if catalog.path:
            self._catalog_path_label.setText(Path(catalog.path).name)
            self._catalog_path_label.setToolTip(catalog.path)
        else:
            self._catalog_path_label.setText("[no dataset]")
            self._catalog_path_label.setToolTip("")

        n = len(entries)
        self._catalog_count_label.setText(f"▶ {n} entr{'y' if n == 1 else 'ies'}")

        self._catalog_table.blockSignals(True)
        self._catalog_table.setRowCount(0)

        for entry in entries:
            row = self._catalog_table.rowCount()
            self._catalog_table.insertRow(row)

            name = entry.display_name or Path(entry.path).stem
            name_item = QTableWidgetItem(name)
            name_item.setToolTip(entry.path)
            self._catalog_table.setItem(row, 0, name_item)

            s = entry.summary
            frames_item = QTableWidgetItem(str(s.get("n_frames", "—")))
            frames_item.setFlags(frames_item.flags() & ~Qt.ItemIsEditable)
            self._catalog_table.setItem(row, 1, frames_item)

            cells_item = QTableWidgetItem(str(s.get("avg_cells", "—")))
            cells_item.setFlags(cells_item.flags() & ~Qt.ItemIsEditable)
            self._catalog_table.setItem(row, 2, cells_item)

            t1_item = QTableWidgetItem(str(s.get("n_t1_events", "—")))
            t1_item.setFlags(t1_item.flags() & ~Qt.ItemIsEditable)
            self._catalog_table.setItem(row, 3, t1_item)

            cond_item = QTableWidgetItem(entry.condition)
            cond_item.setFlags(cond_item.flags() & ~Qt.ItemIsEditable)
            self._catalog_table.setItem(row, 4, cond_item)

        self._catalog_table.blockSignals(False)

    def _sync_from_state(self):
        for w in (self._px_edit, self._dt_edit, self._condition_edit):
            w.blockSignals(True)
        px = self._state.pixel_size
        dt_s = self._state.time_interval
        dt_min = dt_s / 60.0 if dt_s is not None else None
        self._px_edit.setText(str(px) if px is not None else "")
        self._dt_edit.setText(str(dt_min) if dt_min is not None else "")
        self._condition_edit.setText(self._state.condition)
        for w in (self._px_edit, self._dt_edit, self._condition_edit):
            w.blockSignals(False)

    # ------------------------------------------------------------------
    # UI → State: metadata
    # ------------------------------------------------------------------

    def _on_metadata_edited(self):
        px = _parse_float(self._px_edit.text())
        dt_min = _parse_float(self._dt_edit.text())
        dt_s = dt_min * 60.0 if dt_min is not None else None
        condition = self._condition_edit.text().strip()
        self._state.pixel_size = px
        self._state.time_interval = dt_s
        self._state.condition = condition

    # ------------------------------------------------------------------
    # Dataset catalog operations
    # ------------------------------------------------------------------

    def _on_load_catalog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load dataset", "",
            "CellFlow dataset (*.cfproj);;All files (*)",
        )
        if not path:
            return
        from cellflow.utils.io import load_catalog
        try:
            catalog = load_catalog(path)
        except Exception as exc:
            logger.error("Failed to load catalog: %s", exc)
            self._status_label.setText(f"Error: {exc}")
            return

        self._state._catalog = catalog
        self._state.catalog_changed.emit()

        # Populate metadata from catalog if currently empty
        if catalog.pixel_size is not None and self._state.pixel_size is None:
            self._state.pixel_size = catalog.pixel_size
        if catalog.time_interval is not None and self._state.time_interval is None:
            self._state.time_interval = catalog.time_interval
        if catalog.condition and not self._state.condition:
            self._state.condition = catalog.condition

    def _on_save_catalog(self):
        catalog = self._state.catalog
        default = catalog.path or "dataset.cfproj"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save dataset", default,
            "CellFlow dataset (*.cfproj);;All files (*)",
        )
        if not path:
            return
        if not path.endswith(".cfproj"):
            path += ".cfproj"

        # Sync metadata into catalog before saving
        catalog.pixel_size = self._state.pixel_size
        catalog.time_interval = self._state.time_interval
        catalog.condition = self._state.condition

        from cellflow.utils.io import save_catalog
        try:
            save_catalog(path, catalog)
        except Exception as exc:
            logger.error("Failed to save catalog: %s", exc)
            self._status_label.setText(f"Save error: {exc}")
            return

        self._state._catalog.path = path
        self._state.catalog_changed.emit()
        self._status_label.setText(f"Saved: {Path(path).name}")

    def _on_show_in_viewer(self):
        row = self._catalog_table.currentRow()
        if row < 0:
            self._status_label.setText("Select a row first.")
            return
        catalog = self._state.catalog
        if row >= len(catalog.entries):
            return
        entry = catalog.entries[row]
        h5_path = self._resolve_entry_path(entry.path)
        from cellflow.utils.io import load_tissue
        try:
            tissue = load_tissue(h5_path)
        except Exception as exc:
            logger.error("Failed to load tissue: %s", exc)
            self._status_label.setText(f"Error: {exc}")
            return

        # Load into working tissue state first
        self._state._tissue = tissue
        self._state.tissue_changed.emit()

        # Pull metadata from series if not already set
        if tissue.series:
            s = tissue.series
            if self._state.pixel_size is None and s.pixel_size is not None:
                self._state.pixel_size = s.pixel_size
            if self._state.time_interval is None and s.time_interval is not None:
                self._state.time_interval = s.time_interval

        # Auto-push labels to viewer
        name = entry.display_name or Path(h5_path).stem
        if tissue.labels is not None:
            self._load_labels_into_viewer(tissue.labels, name)
            tissue.labels_layer = f"{name}_labels"
        if tissue.nuclear_labels is not None:
            nuc_layer_name = f"{name}_nuclear_labels"
            if nuc_layer_name in self.viewer.layers:
                self.viewer.layers[nuc_layer_name].data = tissue.nuclear_labels
            else:
                self.viewer.add_labels(tissue.nuclear_labels, name=nuc_layer_name)
            tissue.nuclear_labels_layer = nuc_layer_name

        if tissue.labels is not None or tissue.nuclear_labels is not None:
            self._status_label.setText(f"Loaded: {name}")
        else:
            self._status_label.setText("Loaded (no labels in file).")

    def _on_remove_from_catalog(self):
        row = self._catalog_table.currentRow()
        if row < 0:
            self._status_label.setText("Select a row first.")
            return
        catalog = self._state.catalog
        if row >= len(catalog.entries):
            return
        name = catalog.entries[row].display_name or str(row)
        self._state.remove_from_catalog(row)
        self._status_label.setText(f"Removed '{name}' from dataset.")

    def _on_open_dashboard(self):
        catalog = self._state.catalog
        # Build a TissueGraphDataset from catalog entries for the dashboard
        from cellflow.utils.structures import TissueGraphDataset
        ds = TissueGraphDataset(
            condition=self._state.condition,
            pixel_size=self._state.pixel_size,
            time_interval=self._state.time_interval,
        )
        for i, entry in enumerate(catalog.entries):
            try:
                series = catalog.get_series(i)
                ds.tissues[i] = series
            except Exception as exc:
                logger.warning("Could not load series for entry %d: %s", i, exc)

        # Fall back to legacy dataset
        if not ds.tissues and self._state.dataset is not None:
            ds = self._state.dataset

        if not ds.tissues:
            self._status_label.setText("No dataset to open in dashboard.")
            return

        try:
            import tempfile
            tmp_h5 = Path(tempfile.mktemp(prefix="cellflow_dashboard_", suffix=".h5"))
            from cellflow.utils.io import save_dataset
            save_dataset(ds, tmp_h5)
            import subprocess, sys
            subprocess.Popen([sys.executable, "-m", "cellflow.dashboard", str(tmp_h5)])
            self._status_label.setText("Dashboard launched in browser.")
        except ImportError:
            self._status_label.setText(
                "Dashboard requires dash+plotly. Install with: pip install cellflow[dashboard]"
            )
        except Exception as exc:
            self._status_label.setText(f"Failed to launch dashboard: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_labels_into_viewer(self, labels: np.ndarray, name: str):
        import napari.layers
        layer_name = f"{name}_labels"
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = labels
        else:
            self.viewer.add_labels(labels, name=layer_name)

    def _resolve_entry_path(self, entry_path: str) -> str:
        p = Path(entry_path)
        if p.is_absolute():
            return str(p)
        catalog_path = self._state.catalog.path
        if catalog_path:
            return str(Path(catalog_path).parent / p)
        return str(p)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_float(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return None


def _make_entry_path(h5_path: str, catalog_path: Optional[str]) -> str:
    """Return *h5_path* relative to the catalog directory, or absolute as fallback."""
    if not catalog_path:
        return h5_path
    try:
        return str(Path(h5_path).relative_to(Path(catalog_path).parent))
    except ValueError:
        return h5_path


