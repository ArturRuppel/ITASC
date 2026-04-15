"""New Project wizard dialog.

Lets the user pick an experiment root directory, choose which pipeline
stages to include, and optionally enter experiment metadata.  On
acceptance it writes ``pipeline_schema.json``, creates the directory
skeleton, and updates the viewer state.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from cellflow.core.paths import STAGE_DIRS, schema_path
from cellflow.core.schema import PipelineSchema
from ._plugin import STAGE_DISPLAY_NAMES, STAGE_ORDER, STAGES


class NewProjectDialog(QDialog):
    """Dialog to create a new CellFlow pipeline project.

    Parameters
    ----------
    viewer:
        Active napari viewer instance.
    state:
        Shared :class:`~cellflow.napari.registry.ViewerState`.
    parent:
        Qt parent widget.
    """

    def __init__(self, viewer, state, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._state = state
        self.setWindowTitle("New Pipeline Project")
        self.setMinimumWidth(480)

        self._root_path: Optional[Path] = None
        self._checkboxes: dict[str, QCheckBox] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Experiment root ───────────────────────────────────────────
        root_group = QGroupBox("Experiment Root Directory")
        root_layout = QHBoxLayout()
        root_group.setLayout(root_layout)

        self._root_edit = QLineEdit()
        self._root_edit.setPlaceholderText("/path/to/experiment_root")
        self._root_edit.setReadOnly(True)
        root_layout.addWidget(self._root_edit)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_root)
        root_layout.addWidget(browse_btn)
        layout.addWidget(root_group)

        # ── Stage selection ───────────────────────────────────────────
        stage_group = QGroupBox("Pipeline Stages (select stages to enable)")
        stage_outer = QVBoxLayout()
        stage_group.setLayout(stage_outer)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMaximumHeight(220)

        container = QWidget()
        cb_layout = QVBoxLayout(container)
        cb_layout.setSpacing(3)

        installed = set(STAGES.keys())
        ordered = [s for s in STAGE_ORDER if s in installed]
        ordered += sorted(installed - set(STAGE_ORDER))

        for name in ordered:
            display = STAGE_DISPLAY_NAMES.get(name, name)
            cb = QCheckBox(display)
            cb.setChecked(True)
            cb.setObjectName(name)
            self._checkboxes[name] = cb
            cb_layout.addWidget(cb)

        if not ordered:
            cb_layout.addWidget(QLabel("No stages installed."))

        cb_layout.addStretch()
        scroll.setWidget(container)
        stage_outer.addWidget(scroll)
        layout.addWidget(stage_group)

        # ── Metadata (optional) ───────────────────────────────────────
        meta_group = QGroupBox("Metadata (optional)")
        meta_layout = QHBoxLayout()
        meta_group.setLayout(meta_layout)

        meta_layout.addWidget(QLabel("px (µm):"))
        self._px_edit = QLineEdit()
        self._px_edit.setFixedWidth(62)
        self._px_edit.setPlaceholderText("—")
        meta_layout.addWidget(self._px_edit)

        meta_layout.addWidget(QLabel("dt (min):"))
        self._dt_edit = QLineEdit()
        self._dt_edit.setFixedWidth(62)
        self._dt_edit.setPlaceholderText("—")
        meta_layout.addWidget(self._dt_edit)

        meta_layout.addStretch()
        layout.addWidget(meta_group)

        # ── Buttons ───────────────────────────────────────────────────
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: red; font-size: 9pt;")
        layout.addWidget(self._status_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse_root(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select Experiment Root Directory"
        )
        if d:
            self._root_path = Path(d)
            self._root_edit.setText(d)

    def _on_accept(self) -> None:
        if self._root_path is None:
            self._status_label.setText("Please choose an experiment root directory.")
            return

        selected_stages = [
            name for name, cb in self._checkboxes.items() if cb.isChecked()
        ]
        if not selected_stages:
            self._status_label.setText("Please select at least one stage.")
            return

        try:
            self._create_project(self._root_path, selected_stages)
        except Exception as exc:
            self._status_label.setText(f"Error: {exc}")
            return

        self.accept()

    # ------------------------------------------------------------------
    # Project creation
    # ------------------------------------------------------------------

    def _create_project(self, root: Path, stages: List[str]) -> None:
        """Write schema, create directory skeleton, update state."""
        # Build metadata
        pixel_size: Optional[float] = None
        time_interval: Optional[float] = None
        try:
            pixel_size = float(self._px_edit.text())
        except (ValueError, AttributeError):
            pass
        try:
            time_interval = float(self._dt_edit.text())
        except (ValueError, AttributeError):
            pass

        # Build schema
        from cellflow.core.schema import PipelineMetadata
        schema = PipelineSchema(
            stages=stages,
            metadata=PipelineMetadata(
                pixel_size_um=pixel_size,
                time_interval_s=time_interval,
            ),
        )

        # Save schema
        root.mkdir(parents=True, exist_ok=True)
        schema.save(schema_path(root))

        # Create a single pos00 skeleton so users can see the structure
        _create_pos_skeleton(root, pos=0, stages=stages)

        # Update shared state
        self._state.set_project_dir(root)
        if pixel_size is not None:
            self._state.pixel_size = pixel_size
        if time_interval is not None:
            self._state.time_interval = time_interval / 60.0  # store in minutes


def _create_pos_skeleton(root: Path, pos: int, stages: List[str]) -> None:
    """Create the directory skeleton for one position."""
    from cellflow.core.paths import pos_dir, stage_dir
    pos_dir(root, pos).mkdir(parents=True, exist_ok=True)
    for stage_name in stages:
        if stage_name in STAGE_DIRS:
            stage_dir(root, pos, stage_name).mkdir(parents=True, exist_ok=True)
