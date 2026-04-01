"""Project bar widget — fixed strip above the tab widget.

Shows the current project file, New / Load / Save buttons, and the
pixel-size / time-interval fields that used to live only in the
Database tab.  All values are kept in sync with ViewerState so every
tab reads the same metadata without touching this widget directly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QSizePolicy,
)

from .registry import ViewerState

logger = logging.getLogger(__name__)


class ProjectBar(QWidget):
    """Thin horizontal bar: [Project: path] [New] [Load] [Save]  px [  ]  dt [  ]"""

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
        layout = QHBoxLayout()
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)
        self.setLayout(layout)

        # Project path label
        self._path_label = QLabel("No project")
        self._path_label.setStyleSheet("color: palette(mid); font-size: 9pt;")
        self._path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._path_label.setToolTip("Current project file (.h5)")
        layout.addWidget(self._path_label)

        # Action buttons
        self._new_btn = QPushButton("New")
        self._new_btn.setFixedWidth(44)
        self._new_btn.setToolTip("Clear the current project (keeps viewer layers)")
        layout.addWidget(self._new_btn)

        self._load_btn = QPushButton("Load…")
        self._load_btn.setFixedWidth(52)
        self._load_btn.setToolTip("Load a project from a .h5 file")
        layout.addWidget(self._load_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setFixedWidth(44)
        self._save_btn.setToolTip("Save the current project to a .h5 file")
        layout.addWidget(self._save_btn)

        self._save_as_btn = QPushButton("Save as…")
        self._save_as_btn.setFixedWidth(68)
        self._save_as_btn.setToolTip("Save to a new .h5 file")
        layout.addWidget(self._save_as_btn)

        # Separator
        sep = QLabel("|")
        sep.setStyleSheet("color: palette(mid);")
        layout.addWidget(sep)

        # Pixel size
        layout.addWidget(QLabel("px (µm):"))
        self._px_edit = QLineEdit()
        self._px_edit.setFixedWidth(56)
        self._px_edit.setPlaceholderText("—")
        self._px_edit.setToolTip("Pixel size in µm/px")
        layout.addWidget(self._px_edit)

        # Time interval
        layout.addWidget(QLabel("dt (s):"))
        self._dt_edit = QLineEdit()
        self._dt_edit.setFixedWidth(56)
        self._dt_edit.setPlaceholderText("—")
        self._dt_edit.setToolTip("Time interval between frames in seconds")
        layout.addWidget(self._dt_edit)

    def _connect_signals(self):
        self._new_btn.clicked.connect(self._on_new)
        self._load_btn.clicked.connect(self._on_load)
        self._save_btn.clicked.connect(self._on_save)
        self._save_as_btn.clicked.connect(self._on_save_as)

        self._px_edit.editingFinished.connect(self._on_metadata_edited)
        self._dt_edit.editingFinished.connect(self._on_metadata_edited)

        # Keep fields in sync if another widget updates the state
        self._state.metadata_changed.connect(self._sync_from_state)
        self._state.project_changed.connect(self._sync_path_label)

    # ------------------------------------------------------------------
    # State → UI sync
    # ------------------------------------------------------------------

    def _sync_from_state(self):
        """Refresh editable fields from ViewerState (without re-emitting)."""
        self._px_edit.blockSignals(True)
        self._dt_edit.blockSignals(True)
        px = self._state.pixel_size
        dt = self._state.time_interval
        self._px_edit.setText(str(px) if px is not None else "")
        self._dt_edit.setText(str(dt) if dt is not None else "")
        self._px_edit.blockSignals(False)
        self._dt_edit.blockSignals(False)

    def _sync_path_label(self):
        p = self._state.project_path
        if p:
            self._path_label.setText(Path(p).name)
            self._path_label.setToolTip(p)
        else:
            self._path_label.setText("No project")
            self._path_label.setToolTip("")

    # ------------------------------------------------------------------
    # UI → State sync
    # ------------------------------------------------------------------

    def _on_metadata_edited(self):
        px = _parse_float(self._px_edit.text())
        dt = _parse_float(self._dt_edit.text())
        self._state.pixel_size = px
        self._state.time_interval = dt

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _on_new(self):
        self._state.project_path = None
        self._state.dataset = None
        self._state.pixel_size = None
        self._state.time_interval = None
        self._state.condition = ""
        self._px_edit.clear()
        self._dt_edit.clear()

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

        # Push metadata into state
        self._state.pixel_size = result["pixel_size"]
        self._state.time_interval = result["time_interval"]
        self._state.condition = result["condition"]
        self._state.dataset = result["dataset"]
        self._state.project_path = path

        # Load labels into viewer if present
        labels = result.get("labels")
        if labels is not None:
            self._load_labels_into_viewer(labels, Path(path).stem)

    def _load_labels_into_viewer(self, labels: np.ndarray, name: str):
        import napari.layers
        layer_name = f"{name}_labels"
        # Overwrite existing layer with the same name if present
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

        # Collect current labels from the active Labels layer
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
        """Return label data from the active Labels layer, or None."""
        import napari.layers
        active = self.viewer.layers.selection.active
        if active is not None and isinstance(active, napari.layers.Labels):
            return np.asarray(active.data)
        # Fall back to first Labels layer
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                return np.asarray(layer.data)
        return None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_float(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return None
