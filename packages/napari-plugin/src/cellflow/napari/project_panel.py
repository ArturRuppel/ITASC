"""Project panel — fixed strip above the tab widget.

Three sections:

  Tissue    — collapsible; pipeline file tracker (presence, shape/dtype, load)
  Metadata  — pixel size, time interval, condition
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
    QGroupBox,
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .registry import CatalogEntry, DatasetCatalog, TissueData, ViewerState

logger = logging.getLogger(__name__)

# Pipeline files to track, grouped by display stage.
# Each entry: (path_relative_to_pos_dir, display_name, loadable)
#   loadable = "image" | "labels" | None  (None → no Load button)
_TRACKED_FILE_GROUPS: list[tuple[str, list[tuple[str, str, "str | None"]]]] = [
    ("Input Export", [
        ("0_input/nucleus",                  "Nucleus 3D (frames)", None),
        ("0_input/nucleus/nucleus_zavg.tif", "Nucleus avg",         "image"),
        ("0_input/cell/cell_zavg.tif",       "Cell avg",            "image"),
    ]),
    ("Cellpose Nuclei", [
        ("1_cellpose/nucleus",               "Output directory",   None),
    ]),
    ("Cellpose Cells", [
        ("1_cellpose/cell/cell_dp.tif",      "Cell DP",            None),
        ("1_cellpose/cell/cell_prob.tif",    "Cell prob",          "image"),
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
        ("4_cell_segmentation/cell_labels_raw.tif", "Cell labels raw", "labels"),
        ("4_cell_segmentation/cell_labels.tif",     "Cell labels",     "labels"),
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

        # ── Pipeline Project section ──────────────────────────────────
        pipeline_group = QGroupBox("Pipeline Project")
        pg_layout = QVBoxLayout()
        pg_layout.setContentsMargins(6, 4, 6, 4)
        pg_layout.setSpacing(3)

        # Row: buttons + project path label
        proj_btn_row = QHBoxLayout()
        proj_btn_row.setSpacing(4)
        self._new_project_btn = QPushButton("New Project…")
        self._new_project_btn.setToolTip("Create a new pipeline project directory")
        self._open_project_btn = QPushButton("Open Project…")
        self._open_project_btn.setToolTip("Open an existing pipeline project directory")
        self._project_dir_label = QLabel("[no project]")
        self._project_dir_label.setStyleSheet("font-size: 9pt;")
        self._project_dir_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._project_dir_label.setMinimumWidth(0)
        proj_btn_row.addWidget(self._new_project_btn)
        proj_btn_row.addWidget(self._open_project_btn)
        proj_btn_row.addWidget(self._project_dir_label)
        pg_layout.addLayout(proj_btn_row)

        # Row: import button + status
        import_row = QHBoxLayout()
        import_row.setSpacing(4)
        self._import_pipeline_btn = QPushButton("Import from pipeline…")
        self._import_pipeline_btn.setToolTip(
            "Load the tracked labels from the pipeline output for pos 0\n"
            "and auto-fill pixel size / time interval from the schema."
        )
        self._import_pipeline_btn.setEnabled(False)
        self._import_status_label = QLabel("")
        self._import_status_label.setStyleSheet("font-size: 9pt;")
        self._import_status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        import_row.addWidget(self._import_pipeline_btn)
        import_row.addWidget(self._import_status_label)
        pg_layout.addLayout(import_row)

        # Stage status table: Name | Status | Last Run
        self._pipeline_table = QTableWidget(0, 3)
        self._pipeline_table.setHorizontalHeaderLabels(["Stage", "Status", "Last Run"])
        hh = self._pipeline_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._pipeline_table.setMaximumHeight(110)
        self._pipeline_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._pipeline_table.setSelectionMode(QTableWidget.NoSelection)
        pg_layout.addWidget(self._pipeline_table)

        # Resume banner — shown when manifest has failed/running stages
        self._resume_banner = QLabel("")
        self._resume_banner.setWordWrap(True)
        self._resume_banner.setStyleSheet(
            "QLabel { color: #cc8800; font-size: 9pt; padding: 3px; "
            "border: 1px solid #cc8800; border-radius: 3px; }"
        )
        self._resume_banner.setVisible(False)
        pg_layout.addWidget(self._resume_banner)

        pipeline_group.setLayout(pg_layout)
        root.addWidget(pipeline_group)

        # ── Data State section (collapsible) ─────────────────────────
        self._tissue_toggle = QToolButton()
        self._tissue_toggle.setText("Data State")
        self._tissue_toggle.setArrowType(Qt.DownArrow)
        self._tissue_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._tissue_toggle.setCheckable(True)
        self._tissue_toggle.setChecked(True)
        self._tissue_toggle.setStyleSheet(
            "QToolButton { font-weight: bold; font-size: 10pt; border: none; "
            "padding: 2px; color: white; }"
        )
        root.addWidget(self._tissue_toggle)

        self._tissue_body = QWidget()
        self._tissue_body.setStyleSheet(
            "QLabel { color: white; } "
            "QPushButton { color: white; } "
            "QSpinBox { color: white; }"
        )
        tg_layout = QVBoxLayout(self._tissue_body)
        tg_layout.setContentsMargins(6, 0, 6, 4)
        tg_layout.setSpacing(3)
        root.addWidget(self._tissue_body)

        # ── Position selector + refresh ───────────────────────────────
        pos_row = QHBoxLayout()
        pos_row.setSpacing(4)
        pos_row.addWidget(QLabel("Position:"))
        self._pipeline_pos_spin = QSpinBox()
        self._pipeline_pos_spin.setRange(0, 99)
        self._pipeline_pos_spin.setValue(0)
        self._pipeline_pos_spin.setFixedWidth(54)
        self._pipeline_pos_spin.setToolTip("Position index for pipeline file status")
        pos_row.addWidget(self._pipeline_pos_spin)
        pos_row.addStretch()
        self._refresh_files_btn = QPushButton("↺ Refresh")
        self._refresh_files_btn.setFixedWidth(76)
        self._refresh_files_btn.setToolTip("Refresh pipeline file status")
        pos_row.addWidget(self._refresh_files_btn)
        tg_layout.addLayout(pos_row)

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
        files_scroll.setFixedHeight(210)
        files_scroll.setFrameShape(QFrame.NoFrame)
        tg_layout.addWidget(files_scroll)

        # ── Metadata section ──────────────────────────────────────────
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
        root.addLayout(meta_row)

        # ── Dataset section ───────────────────────────────────────────
        ds_group = QGroupBox("Dataset")
        ds_layout = QVBoxLayout()
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
        self._catalog_table.setMaximumHeight(130)
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

        ds_group.setLayout(ds_layout)
        # Exposed so the parent widget can place the dataset panel below the tab widget.
        self.dataset_widget = ds_group

    def _connect_signals(self):
        # Pipeline project buttons
        self._new_project_btn.clicked.connect(self._on_new_project)
        self._open_project_btn.clicked.connect(self._on_open_project)
        self._import_pipeline_btn.clicked.connect(self._on_import_from_pipeline)
        self._state.pipeline_schema_changed.connect(self._refresh_pipeline_status)
        self._state.pipeline_schema_changed.connect(self._update_import_btn_state)

        # Auto-refresh pipeline status every 3 s when a project is open
        self._pipeline_timer = QTimer(self)
        self._pipeline_timer.setInterval(3000)
        self._pipeline_timer.timeout.connect(self._refresh_pipeline_status)
        self._pipeline_timer.start()

        # Tissue toggle + pipeline file controls
        self._tissue_toggle.toggled.connect(self._on_tissue_toggle)
        self._pipeline_pos_spin.valueChanged.connect(self._refresh_pipeline_files)
        self._refresh_files_btn.clicked.connect(self._refresh_pipeline_files)
        self._state.pipeline_schema_changed.connect(self._refresh_pipeline_files)

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

    def _update_import_btn_state(self) -> None:
        """Enable the import button only when a project with a schema is open."""
        schema = self._state.pipeline_schema
        self._import_pipeline_btn.setEnabled(schema is not None)

    def _on_import_from_pipeline(self) -> None:
        """Load tracked labels from the pipeline output and populate state."""
        schema = self._state.pipeline_schema
        project_dir = self._state.project_dir
        if schema is None or project_dir is None:
            self._import_status_label.setText("No project open.")
            return

        # Resolve the tracked-labels path for pos 0.
        # Try the schema interface first; fall back to the conventional path.
        from cellflow.core.paths import stage_dir
        label_path = None
        iface = schema.interfaces.get("tracking.output.tracked_labels")
        if iface and iface.path_template:
            try:
                rel = iface.path_template.format(pos=0, stem="tracked_labels")
                candidate = project_dir / rel
                if candidate.exists():
                    label_path = candidate
            except Exception:
                pass

        if label_path is None:
            # Conventional fallback: <root>/pos00/2_ultrack/tracked_labels.tif
            candidate = stage_dir(project_dir, 0, "tracking") / "tracked_labels.tif"
            if candidate.exists():
                label_path = candidate

        if label_path is None:
            self._import_status_label.setText("tracked_labels.tif not found for pos 0.")
            return

        # Validate TIFF header before loading
        try:
            import tifffile
            with tifffile.TiffFile(str(label_path)) as tf:
                page = tf.pages[0]
                shape = (len(tf.pages),) + page.shape if len(tf.pages) > 1 else page.shape
        except Exception as exc:
            self._import_status_label.setText(f"Invalid TIFF: {exc}")
            return

        # Load the label array
        try:
            import tifffile
            labels = tifffile.imread(str(label_path))
        except Exception as exc:
            self._import_status_label.setText(f"Load error: {exc}")
            return

        # Push into state
        self._state.set_tissue_labels(labels, layer_name=label_path.name)
        self.viewer.add_labels(labels, name=label_path.name)

        # Auto-fill metadata from schema
        meta = schema.metadata
        if meta.pixel_size_um is not None:
            self._state.pixel_size = meta.pixel_size_um
        if meta.time_interval_s is not None:
            self._state.time_interval = meta.time_interval_s

        self._import_status_label.setText(
            f"Loaded {label_path.name} ({labels.shape})"
        )

    def _refresh_pipeline_status(self) -> None:
        """Read manifest for pos 0 and update the pipeline status table."""
        project_dir = self._state.project_dir
        schema = self._state.pipeline_schema

        if project_dir is None:
            self._project_dir_label.setText("[no project]")
            self._pipeline_table.setRowCount(0)
            return

        # Show a short path (last 2 components) to avoid widget expanding horizontally.
        short = "/".join(project_dir.parts[-2:]) if len(project_dir.parts) >= 2 else str(project_dir)
        self._project_dir_label.setText(short)
        self._project_dir_label.setToolTip(str(project_dir))

        # Determine which stages to show (from schema, or all installed)
        if schema is not None:
            stage_names = list(schema.stages)
        else:
            from ._plugin import STAGE_ORDER, STAGES
            installed = set(STAGES.keys())
            stage_names = [s for s in STAGE_ORDER if s in installed]
            stage_names += sorted(installed - set(STAGE_ORDER))

        if not stage_names:
            self._pipeline_table.setRowCount(0)
            return

        # Load manifest for pos 0
        from cellflow.core.paths import manifest_path
        from cellflow.core.manifest import PipelineManifest
        mpath = manifest_path(project_dir, 0)
        manifest = PipelineManifest.load(mpath)

        from ._plugin import STAGE_DISPLAY_NAMES
        _STATUS_BADGE = {
            "complete": "✓",
            "running":  "↻",
            "failed":   "✗",
            "stale":    "⚠",
            "pending":  "–",
        }
        _STATUS_COLOR = {
            "complete": "#4CAF50",
            "running":  "#2196F3",
            "failed":   "#F44336",
            "stale":    "#FF9800",
            "pending":  "#9E9E9E",
        }

        self._pipeline_table.setRowCount(len(stage_names))
        incomplete_stages = []
        for row, name in enumerate(stage_names):
            record = manifest.stages.get(name)
            status = record.status if record else "pending"
            badge = _STATUS_BADGE.get(status, "–")
            color = _STATUS_COLOR.get(status, "#9E9E9E")
            last_run = (record.finished_at or "").replace("T", " ")[:16] if record else ""

            display = STAGE_DISPLAY_NAMES.get(name, name)
            self._pipeline_table.setItem(row, 0, QTableWidgetItem(display))
            badge_item = QTableWidgetItem(badge)
            badge_item.setTextAlignment(Qt.AlignCenter)
            badge_item.setForeground(__import__("qtpy.QtGui", fromlist=["QColor"]).QColor(color))
            self._pipeline_table.setItem(row, 1, badge_item)
            self._pipeline_table.setItem(row, 2, QTableWidgetItem(last_run))

            if status in ("failed", "running"):
                incomplete_stages.append((display, status))

        # Show resume banner when there are failed or interrupted stages
        if incomplete_stages:
            parts = ", ".join(
                f"{n} ({s})" for n, s in incomplete_stages
            )
            self._resume_banner.setText(
                f"⚠ Incomplete stages detected — open the relevant tab to resume:\n{parts}"
            )
            self._resume_banner.setVisible(True)
        else:
            self._resume_banner.setVisible(False)

        # Keep the pipeline file list in sync with the manifest timer
        self._refresh_pipeline_files()

    # ------------------------------------------------------------------
    # Tissue toggle + pipeline file status
    # ------------------------------------------------------------------

    def _on_tissue_toggle(self, checked: bool) -> None:
        self._tissue_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._tissue_body.setVisible(checked)
        if checked:
            self._refresh_pipeline_files()

    def _refresh_pipeline_files(self) -> None:
        """Check on-disk presence and metadata for every tracked pipeline file."""
        if not self._tissue_body.isVisible():
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


class _PipelineFileRow(QWidget):
    """One pipeline file status row: icon | name | info | [load btn]"""

    def __init__(self, rel_path: str, display_name: str, loadable: "str | None"):
        super().__init__()
        self._rel_path = rel_path
        self._loadable = loadable
        self._full_path: "Path | None" = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 1, 4, 1)
        lay.setSpacing(4)

        self._icon_lbl = QLabel("○")
        self._icon_lbl.setFixedWidth(14)
        self._icon_lbl.setAlignment(Qt.AlignCenter)
        self._icon_lbl.setStyleSheet("font-size: 9pt; color: palette(mid);")
        lay.addWidget(self._icon_lbl)

        name_lbl = QLabel(rel_path)
        name_lbl.setFixedWidth(200)
        name_lbl.setStyleSheet("font-size: 8pt; color: white;")
        name_lbl.setToolTip(display_name)
        lay.addWidget(name_lbl)

        self._info_lbl = QLabel("—")
        self._info_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._info_lbl.setStyleSheet("font-size: 8pt; color: white;")
        lay.addWidget(self._info_lbl)

        if loadable is not None:
            self._load_btn = QPushButton("↑")
            self._load_btn.setFixedWidth(24)
            self._load_btn.setFixedHeight(18)
            self._load_btn.setToolTip("Load into napari viewer")
            self._load_btn.setEnabled(False)
            lay.addWidget(self._load_btn)
        else:
            self._load_btn = None

    def set_present(self, info_text: str) -> None:
        self._icon_lbl.setText("✓")
        self._icon_lbl.setStyleSheet("font-size: 9pt; font-weight: bold; color: #4CAF50;")
        self._info_lbl.setText(info_text)
        self._info_lbl.setStyleSheet("font-size: 8pt; color: white;")
        if self._load_btn:
            self._load_btn.setEnabled(True)

    def set_missing(self) -> None:
        self._icon_lbl.setText("✗")
        self._icon_lbl.setStyleSheet("font-size: 9pt; color: #9E9E9E;")
        self._info_lbl.setText("missing")
        self._info_lbl.setStyleSheet("font-size: 8pt; color: #9E9E9E;")
        self._full_path = None
        if self._load_btn:
            self._load_btn.setEnabled(False)

    def set_no_project(self) -> None:
        self._icon_lbl.setText("○")
        self._icon_lbl.setStyleSheet("font-size: 9pt; color: #9E9E9E;")
        self._info_lbl.setText("—")
        self._info_lbl.setStyleSheet("font-size: 8pt; color: #9E9E9E;")
        self._full_path = None
        if self._load_btn:
            self._load_btn.setEnabled(False)


def _file_info(path: Path) -> str:
    """Return a concise shape/dtype or size string for a pipeline output file."""
    if path.is_dir():
        tif_files = sorted(path.glob("*.tif"))
        n = len(tif_files)
        if n == 0:
            return "0 .tif files"
        first = tif_files[0]
        name_str = first.name if n == 1 else f"{first.name} … (+{n - 1})"
        # Validate names: detect expected pattern from directory name
        dir_name = path.name  # e.g. "nucleus" inside 1_cellpose
        # For cellpose nucleus output: files should be nucleus_3d_t*_dp.tif / *_prob.tif
        import fnmatch
        parent_name = path.parent.name if path.parent else ""
        if parent_name == "1_cellpose" and dir_name == "nucleus":
            expected = [f for f in tif_files
                        if fnmatch.fnmatch(f.name, "nucleus_3d_t*_dp.tif")
                        or fnmatch.fnmatch(f.name, "nucleus_3d_t*_prob.tif")]
            if len(expected) < n:
                name_str += " ⚠"
        # Shape of first file
        try:
            import tifffile
            with tifffile.TiffFile(str(first)) as tf:
                s = tf.series[0] if tf.series else None
                shape_str = "×".join(str(d) for d in s.shape) if s else "?"
        except Exception:
            shape_str = "?"
        return f"{n} files: {name_str} ({shape_str})"
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(str(path)) as tf:
                s = tf.series[0] if tf.series else None
                if s is not None:
                    shape_str = "×".join(str(d) for d in s.shape)
                    return f"{shape_str} {s.dtype}"
        except Exception:
            pass
        return "?"
    if suffix == ".db":
        kb = path.stat().st_size // 1024
        return f"{kb} KB"
    if suffix == ".csv":
        try:
            with open(path) as f:
                n = max(0, sum(1 for _ in f) - 1)
            return f"{n} rows"
        except Exception:
            return "?"
    if suffix in (".h5", ".hdf5"):
        kb = path.stat().st_size // 1024
        return f"{kb} KB"
    if suffix == ".npz":
        try:
            data = np.load(str(path), allow_pickle=False)
            keys = list(data.keys())
            data.close()
            return ", ".join(keys) if keys else "empty"
        except Exception:
            pass
        return "?"
    kb = path.stat().st_size // 1024
    return f"{kb} KB"
