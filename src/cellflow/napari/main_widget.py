"""Main widget for the CellFlow napari plugin."""
from __future__ import annotations

import json
from pathlib import Path

import napari
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.cellpose_widget import CellposeWidget
from cellflow.napari.contact_analysis_widget import ContactAnalysisWidget
from cellflow.napari.cell_workflow_widget import CellWorkflowWidget
from cellflow.napari.data_panel_widget import ProjectStatusPanel
from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget
from cellflow.napari.widgets import (
    CollapsibleSection,
    pipeline_status_from_files,
)
from cellflow.napari._widget_helpers import tool_btn
from cellflow.napari.ui_style import (
    active_theme_name,
    icon_button,
    muted_label,
    refresh_stage_header_labels,
    set_active_theme,
    stage_accent,
    theme_names,
    tiny_button,
)


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

        self._cellpose_widget = CellposeWidget(self.viewer)
        self.cellpose_section = CollapsibleSection(
            "Cellpose",
            self._cellpose_widget,
            expanded=False,

            accent_color=stage_accent("cellpose"),
        )

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
        self._connect_correction_position_lock()

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
        self._setup_theme_selector(main_layout)

        # Connect signals
        self.project_btn.clicked.connect(lambda: self._on_set_project_directory())
        self.save_btn.clicked.connect(lambda: self._on_save_config())
        self.save_as_btn.clicked.connect(lambda: self._on_save_config_as())
        self.load_btn.clicked.connect(lambda: self._on_load_config())
        self.load_from_btn.clicked.connect(lambda: self._on_load_config_from())
        
        self.refresh_btn.clicked.connect(lambda: self._refresh_all())
        self.pos_spin.valueChanged.connect(lambda: self._refresh_all())
        self._sync_position_controls_enabled()

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

    def _connect_correction_position_lock(self) -> None:
        """Disable top-level position changes while correction mode is active."""
        for workflow in (self.nucleus_workflow_widget, self.cell_workflow_widget):
            button = getattr(workflow, "correction_active_btn", None)
            if button is not None:
                button.toggled.connect(
                    lambda _checked=False: self._sync_position_controls_enabled()
                )

    def _correction_mode_active(self) -> bool:
        for workflow in (self.nucleus_workflow_widget, self.cell_workflow_widget):
            button = getattr(workflow, "correction_active_btn", None)
            if button is not None and button.isChecked():
                return True
        return False

    def _sync_position_controls_enabled(self) -> None:
        if not hasattr(self, "_position_spin_idle_tooltip"):
            self._position_spin_idle_tooltip = self.pos_spin.toolTip()
        active = self._correction_mode_active()
        self.pos_spin.setEnabled(not active)
        if active:
            self.pos_spin.setToolTip(
                "Position cannot be changed while correction mode is active. "
                "Exit correction mode before switching positions."
            )
        else:
            self.pos_spin.setToolTip(self._position_spin_idle_tooltip)

    def _setup_theme_selector(self, layout: QVBoxLayout) -> None:
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addStretch()

        self.theme_btn = tool_btn("◐", "Theme")
        self.theme_btn.setObjectName("theme_selector_button")
        self.theme_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        self.theme_menu = QMenu(self.theme_btn)
        self._theme_actions = {}
        for name in theme_names():
            action = self.theme_menu.addAction(name)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, theme_name=name: self._on_theme_selected(theme_name)
            )
            self._theme_actions[name] = action
        self.theme_btn.setMenu(self.theme_menu)
        self._sync_theme_menu_state()

        footer.addWidget(self.theme_btn)
        layout.addLayout(footer)

    def _on_theme_selected(self, name: str) -> None:
        set_active_theme(name)
        self._apply_theme_accents()
        self._sync_theme_menu_state()

    def _apply_theme_accents(self) -> None:
        section_stage_keys = (
            (self.data_section, "project_status"),
            (self.cellpose_section, "cellpose"),
            (self.nucleus_section, "nucleus"),
            (self.cell_section, "cell"),
            (self.contact_analysis_section, "contact_analysis"),
        )
        for section, stage_key in section_stage_keys:
            section.set_accent_color(stage_accent(stage_key))
        refresh_stage_header_labels(self)

    def _sync_theme_menu_state(self) -> None:
        current = active_theme_name()
        for name, action in self._theme_actions.items():
            action.setChecked(name == current)
        self.theme_btn.setToolTip(f"Theme: {current}")

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
            "cellpose": self._cellpose_widget.get_state(),
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

        if "cellpose" in state:
            self._cellpose_widget.set_state(state["cellpose"])

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
            with open(path) as f:
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
