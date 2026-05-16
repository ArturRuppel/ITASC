"""Main widget for the CellFlow napari plugin."""
from __future__ import annotations

import json
from pathlib import Path

import napari
from qtpy.QtCore import Qt, QSize, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.contact_analysis_widget import ContactAnalysisWidget
from cellflow.napari.cell_workflow_widget import CellWorkflowWidget
from cellflow.napari.data_panel_widget import ProjectStatusPanel
from cellflow.napari.hpc_cellpose_widget import HpcCellposeWidget
from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget
from cellflow.napari.widgets import (
    CollapsibleSection,
    PipelineFilesWidget,
    pipeline_status_from_files,
)
from cellflow.napari.ui_style import icon_button, muted_label, stage_accent, tiny_button


class _CellposePanel(QWidget):
    """Informational panel for external Cellpose output."""

    def __init__(self, viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        description = QLabel(
            "Cellpose runs externally on the cluster. This panel only documents "
            "the expected input/output files and loads them into napari."
        )
        description.setWordWrap(True)
        muted_label(description)
        layout.addWidget(description)

        self.input_files_tracker = PipelineFilesWidget([
            ("Inputs", [
                ("0_input/nucleus_3dt.tif", "Nucleus 3D+t"),
                ("0_input/cell_3dt.tif", "Cell 3D+t"),
            ]),
        ], viewer=self.viewer)
        layout.addWidget(self.input_files_tracker)

        self.hpc_cellpose_widget = HpcCellposeWidget(self.viewer)
        self.hpc_cellpose_section = CollapsibleSection(
            "HPC Cellpose", self.hpc_cellpose_widget, expanded=False
        )
        layout.addWidget(self.hpc_cellpose_section)

        self.output_files_tracker = PipelineFilesWidget([
            ("Outputs", [
                ("1_cellpose/nucleus_prob_3dt.tif", "Nucleus prob 3D+t"),
                ("1_cellpose/nucleus_prob_zavg.tif", "Nucleus prob z-avg"),
                ("1_cellpose/nucleus_dp_3dt.tif", "Nucleus dp 3D+t"),
                ("1_cellpose/cell_prob_3dt.tif", "Cell prob 3D+t"),
                ("1_cellpose/cell_prob_zavg.tif", "Cell prob z-avg"),
                ("1_cellpose/cell_dp_3dt.tif", "Cell dp 3D+t"),
            ]),
        ], viewer=self.viewer)
        layout.addWidget(self.output_files_tracker)

        self._pos_dir: Path | None = None

    def refresh(self, pos_dir: Path | None) -> None:
        """Update file status display."""
        self._pos_dir = pos_dir
        self.input_files_tracker.refresh(pos_dir)
        self.hpc_cellpose_widget.refresh(pos_dir)
        self.output_files_tracker.refresh(pos_dir)


class CellFlowMainWidget(QWidget):
    """The unified workflow-based UI for CellFlow."""

    refresh_requested = Signal(object)  # emits pos_dir: Path | None

    def __init__(self, napari_viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = napari_viewer

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # ── Project Info (Top Level) ──────────────────────────────────
        self._setup_project_ui(main_layout)

        # Main scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_widget = QWidget()
        self.scroll_widget.setMinimumWidth(0)
        self.scroll_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(2, 2, 2, 2)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.scroll_widget)

        main_layout.addWidget(self.scroll)

        # Add sections
        self.data_panel = ProjectStatusPanel(self.viewer)
        self.data_section = CollapsibleSection(
            "Project Status",
            self.data_panel,
            expanded=False,

            accent_color=stage_accent("project_status"),
        )

        self._cellpose_widget = _CellposePanel(self.viewer)
        self.cellpose_section = CollapsibleSection(
            "Cellpose",
            self._cellpose_widget,
            expanded=False,

            accent_color=stage_accent("cellpose"),
        )
        self.hpc_cellpose_widget = self._cellpose_widget.hpc_cellpose_widget

        self.nucleus_workflow_widget = NucleusWorkflowWidget(self.viewer)
        self.nucleus_section = CollapsibleSection(
            "Nucleus Segmentation & Tracking",
            self.nucleus_workflow_widget,
            expanded=False,

            accent_color=stage_accent("nucleus"),
        )

        self.cell_workflow_widget = CellWorkflowWidget(self.viewer)
        self.cell_section = CollapsibleSection(
            "Cell Segmentation",
            self.cell_workflow_widget,
            expanded=False,

            accent_color=stage_accent("cell"),
        )
        self._connect_label_selection_sync()

        self.contact_analysis_widget = ContactAnalysisWidget(self.viewer)
        self.contact_analysis_section = CollapsibleSection(
            "Contact Analysis",
            self.contact_analysis_widget,
            expanded=False,

            accent_color=stage_accent("contact_analysis"),
        )

        self.scroll_layout.addWidget(self.data_section)
        self.scroll_layout.addWidget(self.cellpose_section)
        self.scroll_layout.addWidget(self.nucleus_section)
        self.scroll_layout.addWidget(self.cell_section)
        self.scroll_layout.addWidget(self.contact_analysis_section)

        for section in (
            self.data_section,
            self.cellpose_section,
            self.nucleus_section,
            self.cell_section,
            self.contact_analysis_section,
        ):
            section.set_status("not_started")

        # Add stretch at the end
        self.scroll_layout.addStretch()

        # Connect signals
        self.project_btn.clicked.connect(lambda: self._on_set_project_directory())
        self.save_btn.clicked.connect(lambda: self._on_save_config())
        self.save_as_btn.clicked.connect(lambda: self._on_save_config_as())
        self.load_btn.clicked.connect(lambda: self._on_load_config())
        self.load_from_btn.clicked.connect(lambda: self._on_load_config_from())
        
        self.refresh_btn.clicked.connect(lambda: self._refresh_all())
        self.pos_spin.valueChanged.connect(lambda: self._refresh_all())

    def _connect_label_selection_sync(self) -> None:
        """Synchronize selected cell/nucleus IDs across correction widgets."""
        if hasattr(self.nucleus_workflow_widget, "set_selection_callback"):
            self.nucleus_workflow_widget.set_selection_callback(
                lambda t, label: self.cell_workflow_widget.select_matching_cell_label(t, label)
            )
        if hasattr(self.cell_workflow_widget, "set_selection_callback"):
            self.cell_workflow_widget.set_selection_callback(
                lambda t, label: self.nucleus_workflow_widget.select_matching_nucleus_label(t, label)
            )

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(int(hint.width() * 1.5), hint.height())

    def _setup_project_ui(self, layout: QVBoxLayout) -> None:
        """Create the top-level project metadata and buttons."""
        proj_widget = QWidget()
        proj_lay = QVBoxLayout(proj_widget)
        proj_lay.setContentsMargins(0, 0, 0, 0)
        proj_lay.setSpacing(4)

        # Row 1: Metadata
        meta_row = QHBoxLayout()
        meta_row.setSpacing(4)
        
        meta_row.addWidget(QLabel("px:"))
        self.px_edit = QLineEdit()
        self.px_edit.setFixedWidth(40)
        meta_row.addWidget(self.px_edit)

        meta_row.addWidget(QLabel("dt:"))
        self.dt_edit = QLineEdit()
        self.dt_edit.setFixedWidth(40)
        meta_row.addWidget(self.dt_edit)

        meta_row.addWidget(QLabel("C:"))
        self.cond_edit = QLineEdit()
        meta_row.addWidget(self.cond_edit)

        meta_row.addWidget(QLabel("P:"))
        self.pos_spin = QSpinBox()
        self.pos_spin.setRange(0, 99)
        self.pos_spin.setFixedWidth(40)
        meta_row.addWidget(self.pos_spin)
        
        self.refresh_btn = QPushButton("↺")
        icon_button(self.refresh_btn)
        self.refresh_btn.setToolTip("Refresh all status")
        meta_row.addWidget(self.refresh_btn)
        
        proj_lay.addLayout(meta_row)

        # Row 2: Project Actions
        project_row = QHBoxLayout()
        project_row.setSpacing(4)
        self.project_btn = QPushButton("Project Directory...")
        tiny_button(self.project_btn)
        project_row.addWidget(self.project_btn)
        proj_lay.addLayout(project_row)

        # Row 3: Config Actions
        config_row = QHBoxLayout()
        config_row.setSpacing(4)
        self.save_btn = QPushButton("Save Config")
        self.save_as_btn = QPushButton("Save Config As...")
        self.load_btn = QPushButton("Load Config")
        self.load_from_btn = QPushButton("Load Config From...")
        
        for btn in (self.save_btn, self.save_as_btn, self.load_btn, self.load_from_btn):
            tiny_button(btn)
            config_row.addWidget(btn)
        proj_lay.addLayout(config_row)

        # Row 4: Path Label
        self.path_label = QLabel("[no project]")
        muted_label(self.path_label)
        self.path_label.setWordWrap(True)
        proj_lay.addWidget(self.path_label)

        layout.addWidget(proj_widget)

    def _on_set_project_directory(self) -> None:
        """Set the project directory and load config if present."""
        path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if path:
            p = Path(path)
            self.path_label.setText(str(p))
            self.path_label.setToolTip(str(p))
            
            # Look for config file
            config_path = p / "cellflow_config.json"
            if config_path.exists():
                self._load_config(str(config_path))
            
            self._refresh_all()

    def get_state(self) -> dict:
        """Return the current UI state as a dictionary."""
        return {
            "metadata": {
                "pixel_size_um": self.px_edit.text(),
                "time_interval_s": self.dt_edit.text(),
                "condition": self.cond_edit.text(),
                "position": self.pos_spin.value(),
            },
            "hpc_cellpose": self.hpc_cellpose_widget.get_state(),
            "nucleus": self.nucleus_workflow_widget.get_state(),
            "cell": self.cell_workflow_widget.get_state(),
        }

    def set_state(self, state: dict) -> None:
        """Update the UI state from a dictionary."""
        if "metadata" in state:
            m = state["metadata"]
            if "pixel_size_um" in m: self.px_edit.setText(str(m["pixel_size_um"]))
            if "time_interval_s" in m: self.dt_edit.setText(str(m["time_interval_s"]))
            if "condition" in m: self.cond_edit.setText(str(m["condition"]))
            if "position" in m: self.pos_spin.setValue(int(m["position"]))

        if "hpc_cellpose" in state:
            self.hpc_cellpose_widget.set_state(state["hpc_cellpose"])
        
        if "nucleus" in state:
            self.nucleus_workflow_widget.set_state(state["nucleus"])
        
        if "cell" in state:
            self.cell_workflow_widget.set_state(state["cell"])

    def _on_save_config(self) -> None:
        """Save current configuration to project directory."""
        path_text = self.path_label.text()
        if not path_text or path_text == "[no project]":
            return
        
        config_path = Path(path_text) / "cellflow_config.json"
        self._save_config(str(config_path))

    def _on_save_config_as(self) -> None:
        """Save current configuration to a specific file."""
        path = QFileDialog.getSaveFileName(self, "Save Config As", filter="JSON (*.json)")[0]
        if path:
            self._save_config(path)

    def _on_load_config(self) -> None:
        """Load configuration from project directory."""
        path_text = self.path_label.text()
        if not path_text or path_text == "[no project]":
            return
        
        config_path = Path(path_text) / "cellflow_config.json"
        if config_path.exists():
            self._load_config(str(config_path))
        else:
            print(f"Config not found: {config_path}")

    def _on_load_config_from(self) -> None:
        """Load configuration from a specific file."""
        path = QFileDialog.getOpenFileName(self, "Load Config From", filter="JSON (*.json)")[0]
        if path:
            self._load_config(path)

    def _save_config(self, path: str) -> None:
        """Save state to a JSON file."""
        state = self.get_state()
        try:
            with open(path, "w") as f:
                json.dump(state, f, indent=4)
            print(f"Config saved to {path}")
        except Exception as e:
            print(f"Error saving config: {e}")

    def _load_config(self, path: str) -> None:
        """Load state from a JSON file."""
        try:
            with open(path, "r") as f:
                state = json.load(f)
            self.set_state(state)
            print(f"Config loaded from {path}")
        except Exception as e:
            print(f"Error loading config: {e}")

    def _refresh_all(self) -> None:
        """Refresh file status in all child widgets."""
        path_text = self.path_label.text()
        if path_text and path_text != "[no project]":
            pos = self.pos_spin.value()
            pos_dir = Path(path_text) / f"pos{pos:02d}"
        else:
            pos_dir = None

        self.data_panel.refresh(pos_dir)
        self._cellpose_widget.refresh(pos_dir)
        self.nucleus_workflow_widget.refresh(pos_dir)
        self.cell_workflow_widget.refresh(pos_dir)
        self.contact_analysis_widget.refresh(pos_dir)
        self._update_section_statuses()
        # Emit signal for other widgets
        self.refresh_requested.emit(pos_dir)

    def _update_section_statuses(self) -> None:
        """Refresh stage-status dots from on-disk file presence."""
        cellpose = pipeline_status_from_files(
            self._cellpose_widget.output_files_tracker, done_group="Outputs"
        )
        nucleus = pipeline_status_from_files(
            self.nucleus_workflow_widget._files_widget, done_group="Output"
        )
        cell = pipeline_status_from_files(
            self.cell_workflow_widget._files_widget, done_group="Output"
        )
        contact = pipeline_status_from_files(
            self.contact_analysis_widget._files_widget, done_group="Output"
        )

        self.cellpose_section.set_status(cellpose)
        self.nucleus_section.set_status(nucleus)
        self.cell_section.set_status(cell)
        self.contact_analysis_section.set_status(contact)

        # Project Status rolls up the three essential pipeline outputs
        # (nucleus labels, cell labels, contact analysis). Cellpose counts
        # toward "in progress" but not toward "done" — it's an intermediate.
        essentials = (nucleus, cell, contact)
        if all(s == "done" for s in essentials):
            project_status = "done"
        elif any(s != "not_started" for s in (cellpose, *essentials)):
            project_status = "in_progress"
        else:
            project_status = "not_started"
        self.data_section.set_status(project_status)
