"""Project panel — fixed strip above the tab widget.

Three sections:

  Tissue    — active working tissue (load/save/capture/push/clear per field)
  Metadata  — pixel size, time interval, condition
  Dataset   — catalog of saved tissues with summary table
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy.QtCore import Qt
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

from .registry import CatalogEntry, DatasetCatalog, TissueData, ViewerState

logger = logging.getLogger(__name__)


class ProjectPanel(QWidget):
    """Panel fixed above the tab widget: tissue ops, metadata, dataset table."""

    def __init__(self, viewer, state: ViewerState):
        super().__init__()
        self.viewer = viewer
        self._state = state
        self._build_ui()
        self._connect_signals()
        self._refresh_tissue()
        self._refresh_catalog()
        self._sync_from_state()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)
        self.setLayout(root)

        # ── Tissue section ────────────────────────────────────────────
        tissue_group = QGroupBox("Tissue")
        tg_layout = QVBoxLayout()
        tg_layout.setContentsMargins(6, 4, 6, 4)
        tg_layout.setSpacing(3)

        # Row 0: file path + Load / Save / Add to dataset
        path_row = QHBoxLayout()
        path_row.setSpacing(4)
        self._tissue_path_label = QLabel("[unsaved]")
        self._tissue_path_label.setStyleSheet("font-size: 9pt;")
        self._tissue_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        path_row.addWidget(self._tissue_path_label)
        self._tissue_load_btn = QPushButton("Load tissue…")
        self._tissue_load_btn.setToolTip("Load image, labels and analysis from an .h5 file")
        path_row.addWidget(self._tissue_load_btn)
        self._tissue_save_btn = QPushButton("Save tissue…")
        self._tissue_save_btn.setToolTip("Save current tissue to an .h5 file")
        path_row.addWidget(self._tissue_save_btn)
        self._add_to_dataset_btn = QPushButton("Add to dataset…")
        self._add_to_dataset_btn.setEnabled(False)
        self._add_to_dataset_btn.setToolTip(
            "Tissue must be saved first.\n"
            "Adds the saved .h5 to the catalog and clears the working tissue."
        )
        path_row.addWidget(self._add_to_dataset_btn)
        tg_layout.addLayout(path_row)

        # Row 1: Image field
        self._image_row = _FieldRow("Image:", self)
        tg_layout.addLayout(self._image_row)

        # Row 1b: Secondary image field (optional — for two-channel segmentation)
        self._image2_row = _FieldRow("Image 2 (opt.):", self)
        tg_layout.addLayout(self._image2_row)

        # Row 2: Segmentation (labels) field
        self._labels_row = _FieldRow("Segmentation:", self)
        tg_layout.addLayout(self._labels_row)

        # Row 3: Edge Analysis field (series; no capture button — comes from pipeline)
        self._analysis_row = _FieldRow("Edge Analysis:", self, has_capture=False)
        tg_layout.addLayout(self._analysis_row)

        # Row 4: ForSys/pressure inference field (no capture — comes from pipeline)
        self._forsys_row = _FieldRow("ForSys:", self, has_capture=False)
        tg_layout.addLayout(self._forsys_row)

        tissue_group.setLayout(tg_layout)
        root.addWidget(tissue_group)

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
        # Tissue buttons
        self._tissue_load_btn.clicked.connect(self._on_load_tissue)
        self._tissue_save_btn.clicked.connect(self._on_save_tissue)
        self._add_to_dataset_btn.clicked.connect(self._on_add_to_dataset)

        # Image field
        self._image_row.capture_btn.clicked.connect(self._on_capture_image)
        self._image_row.to_layer_btn.clicked.connect(self._on_image_to_layer)
        self._image_row.clear_btn.clicked.connect(self._on_clear_image)

        # Image 2 field
        self._image2_row.capture_btn.clicked.connect(self._on_capture_image2)
        self._image2_row.to_layer_btn.clicked.connect(self._on_image2_to_layer)
        self._image2_row.clear_btn.clicked.connect(self._on_clear_image2)

        # Labels field
        self._labels_row.capture_btn.clicked.connect(self._on_capture_labels)
        self._labels_row.to_layer_btn.clicked.connect(self._on_labels_to_layer)
        self._labels_row.clear_btn.clicked.connect(self._on_clear_labels)

        # Edge Analysis field (no capture — generated by pipeline)
        self._analysis_row.to_layer_btn.clicked.connect(self._on_analysis_to_layer)
        self._analysis_row.clear_btn.clicked.connect(self._on_clear_analysis)

        # ForSys field (no capture — generated by pipeline)
        self._forsys_row.to_layer_btn.clicked.connect(self._on_forsys_to_layer)
        self._forsys_row.clear_btn.clicked.connect(self._on_clear_forsys)

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
        self._state.tissue_changed.connect(self._refresh_tissue)
        self._state.catalog_changed.connect(self._refresh_catalog)
        self._state.metadata_changed.connect(self._sync_from_state)
        # legacy compat
        self._state.dataset_changed.connect(self._refresh_catalog)

    # ------------------------------------------------------------------
    # State → UI
    # ------------------------------------------------------------------

    def _refresh_tissue(self):
        tissue = self._state.tissue
        # path label
        if tissue.path:
            self._tissue_path_label.setText(Path(tissue.path).name)
            self._tissue_path_label.setToolTip(tissue.path)
        else:
            self._tissue_path_label.setText("[unsaved]")
            self._tissue_path_label.setToolTip("")

        # image row
        if tissue.image is not None:
            arr = tissue.image
            shape_str = " × ".join(str(d) for d in arr.shape)
            self._image_row.set_status(shape_str)
        else:
            self._image_row.set_status("not loaded")
        self._image_row.to_layer_btn.setEnabled(tissue.image is not None)
        self._image_row.clear_btn.setEnabled(tissue.image is not None)

        # image2 row
        if tissue.image2 is not None:
            arr = tissue.image2
            shape_str = " × ".join(str(d) for d in arr.shape)
            self._image2_row.set_status(shape_str)
        else:
            self._image2_row.set_status("not loaded")
        self._image2_row.to_layer_btn.setEnabled(tissue.image2 is not None)
        self._image2_row.clear_btn.setEnabled(tissue.image2 is not None)

        # labels row
        if tissue.labels is not None:
            arr = tissue.labels
            shape_str = " × ".join(str(d) for d in arr.shape)
            self._labels_row.set_status(shape_str)
        else:
            self._labels_row.set_status("not loaded")
        self._labels_row.to_layer_btn.setEnabled(tissue.labels is not None)
        self._labels_row.clear_btn.setEnabled(tissue.labels is not None)

        # edge analysis row
        if tissue.series is not None:
            s = tissue.series
            n_t1 = len(s.t1_events)
            n_traj = len(s.edge_trajectories)
            self._analysis_row.set_status(f"{n_t1} T1s  {n_traj} traj.")
        else:
            self._analysis_row.set_status("not loaded")
        self._analysis_row.to_layer_btn.setEnabled(tissue.series is not None)
        self._analysis_row.clear_btn.setEnabled(tissue.series is not None)

        # forsys row
        if tissue.forsys is not None:
            self._forsys_row.set_status("loaded")
        else:
            self._forsys_row.set_status("not loaded")
        self._forsys_row.to_layer_btn.setEnabled(tissue.forsys is not None)
        self._forsys_row.clear_btn.setEnabled(tissue.forsys is not None)

        # "Add to dataset" requires a saved path
        self._add_to_dataset_btn.setEnabled(bool(tissue.path))

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
    # Tissue file operations
    # ------------------------------------------------------------------

    def _on_load_tissue(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load tissue", "",
            "CellFlow tissue (*.h5);;All files (*)",
        )
        if not path:
            return
        from ..utils.io import load_tissue
        try:
            tissue = load_tissue(path)
        except Exception as exc:
            logger.error("Failed to load tissue: %s", exc)
            self._status_label.setText(f"Error: {exc}")
            return

        # Set layer name before emitting so the correction widget can find it immediately
        if tissue.labels is not None:
            tissue.labels_layer = f"{Path(path).stem}_labels"

        self._state._tissue = tissue
        self._state.tissue_changed.emit()

        # Push metadata from the loaded tissue (H5 metadata group takes priority over series)
        if self._state.pixel_size is None and tissue.pixel_size is not None:
            self._state.pixel_size = tissue.pixel_size
        if self._state.time_interval is None and tissue.time_interval is not None:
            self._state.time_interval = tissue.time_interval
        if not self._state.condition and tissue.condition:
            self._state.condition = tissue.condition

        # Load labels into viewer
        if tissue.labels is not None:
            self._load_labels_into_viewer(tissue.labels, Path(path).stem)

    def _on_save_tissue(self):
        tissue = self._state.tissue
        default = tissue.path or "tissue.h5"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save tissue", default,
            "CellFlow tissue (*.h5);;All files (*)",
        )
        if not path:
            return
        if not path.endswith(".h5"):
            path += ".h5"

        # Sync current UI metadata into tissue before writing
        tissue.pixel_size = self._state.pixel_size
        tissue.time_interval = self._state.time_interval
        tissue.condition = self._state.condition

        from ..utils.io import save_tissue
        try:
            save_tissue(path, tissue)
        except Exception as exc:
            logger.error("Failed to save tissue: %s", exc)
            self._status_label.setText(f"Save error: {exc}")
            return

        self._state._tissue.path = path
        self._state.tissue_changed.emit()
        self._status_label.setText(f"Saved: {Path(path).name}")

    def _on_add_to_dataset(self):
        tissue = self._state.tissue
        if not tissue.path:
            self._status_label.setText("Save the tissue first.")
            return

        from ..utils.io import read_tissue_summary
        try:
            summary = read_tissue_summary(tissue.path)
        except Exception as exc:
            logger.warning("Could not read summary from %s: %s", tissue.path, exc)
            summary = {}

        catalog = self._state.catalog
        # Resolve to relative path if we have a catalog file
        h5_path = tissue.path
        if catalog.path:
            h5_path = _make_entry_path(h5_path, catalog.path)

        entry = CatalogEntry(
            path=h5_path,
            display_name=Path(tissue.path).stem,
            condition=summary.pop("condition", "") or self._state.condition,
            summary=summary,
        )
        self._state.add_to_catalog(entry)
        self._state.clear_tissue()
        self._status_label.setText(f"Added {Path(tissue.path).name} to dataset.")

    # ------------------------------------------------------------------
    # Tissue field buttons
    # ------------------------------------------------------------------

    def _on_capture_image(self):
        import napari.layers
        for layer in reversed(self.viewer.layers):
            if isinstance(layer, napari.layers.Image):
                self._state.set_tissue_image(np.asarray(layer.data), layer.name)
                self._status_label.setText(f"Captured image from '{layer.name}'.")
                return
        self._status_label.setText("No Image layer found.")

    def _on_image_to_layer(self):
        tissue = self._state.tissue
        if tissue.image is None:
            return
        name = tissue.image_layer or "tissue_image"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = tissue.image
        else:
            self.viewer.add_image(tissue.image, name=name)

    def _on_clear_image(self):
        self._state._tissue.image = None
        self._state._tissue.image_layer = None
        self._state.tissue_changed.emit()

    def _on_capture_image2(self):
        import napari.layers
        # Capture the topmost Image layer that is NOT the primary image
        primary_name = self._state.tissue.image_layer
        for layer in reversed(self.viewer.layers):
            if isinstance(layer, napari.layers.Image) and layer.name != primary_name:
                self._state.set_tissue_image2(np.asarray(layer.data), layer.name)
                self._status_label.setText(f"Captured secondary image from '{layer.name}'.")
                return
        self._status_label.setText("No secondary Image layer found.")

    def _on_image2_to_layer(self):
        tissue = self._state.tissue
        if tissue.image2 is None:
            return
        name = tissue.image2_layer or "tissue_image2"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = tissue.image2
        else:
            self.viewer.add_image(tissue.image2, name=name)

    def _on_clear_image2(self):
        self._state._tissue.image2 = None
        self._state._tissue.image2_layer = None
        self._state.tissue_changed.emit()

    def _on_capture_labels(self):
        import napari.layers
        active = self.viewer.layers.selection.active
        if active is not None and isinstance(active, napari.layers.Labels):
            self._state.set_tissue_labels(np.asarray(active.data), active.name)
            self._status_label.setText(f"Captured labels from '{active.name}'.")
            return
        for layer in reversed(self.viewer.layers):
            if isinstance(layer, napari.layers.Labels):
                self._state.set_tissue_labels(np.asarray(layer.data), layer.name)
                self._status_label.setText(f"Captured labels from '{layer.name}'.")
                return
        self._status_label.setText("No Labels layer found.")

    def _on_labels_to_layer(self):
        tissue = self._state.tissue
        if tissue.labels is None:
            return
        name = tissue.labels_layer or "tissue_labels"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = tissue.labels
        else:
            self.viewer.add_labels(tissue.labels, name=name)

    def _on_clear_labels(self):
        self._state._tissue.labels = None
        self._state._tissue.labels_layer = None
        self._state.tissue_changed.emit()

    def _on_analysis_to_layer(self):
        # Push analysis visualization — delegate to analysis widget via series on state
        tissue = self._state.tissue
        if tissue.series is None:
            return
        # Re-emit tissue_changed; the analysis widget picks it up to redraw layers
        self._state.tissue_changed.emit()
        self._status_label.setText("Analysis pushed to viewer layers.")

    def _on_clear_analysis(self):
        self._state.set_tissue_series(None)
        self._status_label.setText("Analysis cleared.")

    def _on_forsys_to_layer(self):
        tissue = self._state.tissue
        if tissue.forsys is None:
            return
        name = tissue.forsys_layer or "tissue_forsys"
        if name in self.viewer.layers:
            self.viewer.layers[name].data = tissue.forsys
        else:
            self.viewer.add_image(tissue.forsys, name=name)

    def _on_clear_forsys(self):
        self._state._tissue.forsys = None
        self._state._tissue.forsys_layer = None
        self._state.tissue_changed.emit()
        self._status_label.setText("ForSys cleared.")

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
        from ..utils.io import load_catalog
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

        from ..utils.io import save_catalog
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
        from ..utils.io import load_tissue
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
        if tissue.labels is not None:
            name = entry.display_name or Path(h5_path).stem
            self._load_labels_into_viewer(tissue.labels, name)
            tissue.labels_layer = f"{name}_labels"
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
        from ..utils.structures import TissueGraphDataset
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
            from ..utils.io import save_dataset
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

class _FieldRow(QHBoxLayout):
    """One data-field row: label | status | [Load] | [To layer] | [Clear]"""

    def __init__(self, label_text: str, parent_widget: QWidget, has_capture: bool = True):
        super().__init__()
        self.setSpacing(4)

        lbl = QLabel(label_text)
        lbl.setFixedWidth(95)
        self.addWidget(lbl)

        self._status = QLabel("not loaded")
        self._status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._status.setStyleSheet("font-size: 9pt;")
        self.addWidget(self._status)

        if has_capture:
            self.capture_btn = QPushButton("Load")
            self.capture_btn.setFixedWidth(54)
            self.capture_btn.setToolTip("Pull from active napari layer into internal storage")
            self.addWidget(self.capture_btn)
        else:
            # placeholder so callers can always reference capture_btn
            self.capture_btn = QPushButton()
            self.capture_btn.setVisible(False)

        self.to_layer_btn = QPushButton("To layer")
        self.to_layer_btn.setFixedWidth(64)
        self.to_layer_btn.setEnabled(False)
        self.to_layer_btn.setToolTip("Push internal data to a napari layer")
        self.addWidget(self.to_layer_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedWidth(50)
        self.clear_btn.setEnabled(False)
        self.clear_btn.setToolTip("Remove from internal storage (napari layer unaffected)")
        self.addWidget(self.clear_btn)

    def set_status(self, text: str) -> None:
        self._status.setText(text)


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
