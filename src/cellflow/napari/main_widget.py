"""Main widget for the CellFlow napari plugin."""
from __future__ import annotations

import json
from pathlib import Path

import napari
from napari.utils.notifications import show_error, show_info, show_warning
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.cellpose_widget import CellposeWidget
from cellflow.napari.aggregate_quantification_widget import AggregateQuantificationWidget
from cellflow.napari.cell_workflow_widget import CellWorkflowWidget
from cellflow.napari.data_panel_widget import ProjectStatusPanel
from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget
from cellflow.napari.widgets import (
    CollapsibleSection,
    pipeline_status_from_files,
)
from cellflow.napari._widget_helpers import tool_btn
from cellflow.napari.ui_gate import ControlClass, UiGate
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

        # The selected position folder — the unit of work. ``None`` until the
        # user picks one. This folder *is* ``pos_dir``; there is no separate
        # project root or position index.
        self._pos_dir: Path | None = None

        # Single app-wide UI gate shared by all sections. It is the one source
        # of truth for control enablement: viewer-owner mutual exclusion (only
        # one of correction / db-browser / live preview at a time) and the
        # context-change guard for folder selection / config loading.
        self.gate = UiGate(self)

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

        self._cellpose_widget = CellposeWidget(self.viewer, gate=self.gate)
        self.cellpose_section = CollapsibleSection(
            "Cellpose",
            self._cellpose_widget,
            expanded=False,

            accent_color=stage_accent("cellpose"),
        )

        self.nucleus_workflow_widget = NucleusWorkflowWidget(self.viewer, gate=self.gate)
        self.nucleus_section = CollapsibleSection(
            "Nucleus Segmentation & Tracking",
            self.nucleus_workflow_widget,
            expanded=False,

            accent_color=stage_accent("nucleus"),
        )

        self.cell_workflow_widget = CellWorkflowWidget(self.viewer, gate=self.gate)
        self.cell_section = CollapsibleSection(
            "Cell Segmentation",
            self.cell_workflow_widget,
            expanded=False,

            accent_color=stage_accent("cell"),
        )
        self._connect_label_selection_sync()

        self.contact_analysis_widget = AggregateQuantificationWidget(self.viewer, gate=self.gate)
        self.contact_analysis_section = CollapsibleSection(
            "Results",
            self.contact_analysis_widget,
            expanded=False,

            accent_color=stage_accent("contact_analysis"),
        )

        # Viewer-activity banner sits at the top level, above Project Status, so
        # the "exit the active mode" hint is visible regardless of which section
        # holds the active viewer owner (correction / db-browser / live preview).
        self.viewer_activity_banner = QLabel("")
        self.viewer_activity_banner.setWordWrap(True)
        self.viewer_activity_banner.setVisible(False)
        self.viewer_activity_banner.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        self.viewer_activity_banner.setStyleSheet(
            "QLabel { font-weight: 700; padding: 4px 6px; "
            "border: 1px solid #f9e2af; background: rgba(249, 226, 175, 35); }"
        )
        self.scroll_layout.addWidget(self.viewer_activity_banner)

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
        self.project_btn.clicked.connect(lambda: self._on_set_position_folder())
        self.save_btn.clicked.connect(lambda: self._on_save_config())
        self.save_as_btn.clicked.connect(lambda: self._on_save_config_as())
        self.load_btn.clicked.connect(lambda: self._on_load_config())
        self.load_from_btn.clicked.connect(lambda: self._on_load_config_from())

        self.refresh_btn.clicked.connect(lambda: self._refresh_all())
        self._register_gate_controls()

        self.gate.changed.connect(self._update_activity_banner)
        self._update_activity_banner()

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

    def _register_gate_controls(self) -> None:
        """Register top-level controls with the app-wide UI gate.

        Folder selection / config-load swap the underlying data, so they are
        ``CONTEXT_CHANGING``: they stay enabled, but clicking one while a viewer
        owner (correction / live preview / db-browser) is active first offers to
        exit that owner (see ``_change_context``). Save Config is harmless and
        needs no gating.
        """
        for control in (
            self.project_btn,
            self.load_btn,
            self.load_from_btn,
        ):
            self.gate.register(control, ControlClass.CONTEXT_CHANGING)
        self.gate.recompute()

    def _set_viewer_activity_banner(self, text: str) -> None:
        visible = bool(text)
        if self.viewer_activity_banner.text() != text:
            self.viewer_activity_banner.setText(text)
        if self.viewer_activity_banner.isVisible() != visible:
            self.viewer_activity_banner.setVisible(visible)

    def _update_activity_banner(self) -> None:
        label = self.gate.owner_label()
        if label:
            self._set_viewer_activity_banner(
                f"{label[0].upper()}{label[1:]} active. "
                "Exit it to use disabled workflow controls."
            )
        else:
            self._set_viewer_activity_banner("")

    def _change_context(self, action) -> bool:
        """Run *action*, offering to exit the active viewer owner first.

        Returns ``True`` if the action ran, ``False`` if the user declined.
        """
        return self.gate.confirm_context_change(self, action)

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

        self.refresh_btn = QPushButton("↺")
        icon_button(self.refresh_btn)
        self.refresh_btn.setToolTip("Refresh all status")
        meta_row.addWidget(self.refresh_btn)
        
        proj_lay.addLayout(meta_row)

        # Row 2: Project Actions
        project_row = QHBoxLayout()
        project_row.setSpacing(4)
        self.project_btn = QPushButton("Position Folder...")
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
        self.path_label = QLabel("[no folder]")
        muted_label(self.path_label)
        self.path_label.setWordWrap(True)
        proj_lay.addWidget(self.path_label)

        layout.addWidget(proj_widget)

    def _on_set_position_folder(self) -> None:
        """Pick a position folder and load its config if present.

        The chosen folder *is* ``pos_dir`` — the unit of work. Any folder with
        the stage layout (``0_input/`` … ``aggregate_quantification/``) is valid;
        the child widgets no-op on missing subdirs, so no validation is needed.
        """
        path = QFileDialog.getExistingDirectory(self, "Select Position Folder")
        if not path:
            return

        def action() -> None:
            p = Path(path)
            self._pos_dir = p
            self.path_label.setText(str(p))
            self.path_label.setToolTip(str(p))

            # Config travels with the data: look for it inside the folder.
            config_path = p / "cellflow_config.json"
            if config_path.exists():
                self._load_config(str(config_path))

            self._refresh_all()

        self._change_context(action)

    def get_state(self) -> dict:
        """Return the current UI state as a dictionary."""
        return {
            "metadata": {
                "pixel_size_um": self.px_edit.text(),
                "time_interval_s": self.dt_edit.text(),
                "condition": self.cond_edit.text(),
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
            # ``position`` from legacy project-root configs is intentionally
            # ignored: the picked folder now carries identity.

        if "cellpose" in state:
            self._cellpose_widget.set_state(state["cellpose"])

        if "nucleus" in state:
            self.nucleus_workflow_widget.set_state(state["nucleus"])

        if "cell" in state:
            self.cell_workflow_widget.set_state(state["cell"])

    def _on_save_config(self) -> None:
        """Save current configuration into the position folder."""
        if self._pos_dir is None:
            return

        config_path = self._pos_dir / "cellflow_config.json"
        self._save_config(str(config_path))

    def _on_save_config_as(self) -> None:
        """Save current configuration to a specific file."""
        path = QFileDialog.getSaveFileName(self, "Save Config As", filter="JSON (*.json)")[0]
        if path:
            self._save_config(path)

    def _on_load_config(self) -> None:
        """Load configuration from the position folder."""
        if self._pos_dir is None:
            return

        config_path = self._pos_dir / "cellflow_config.json"
        if config_path.exists():
            self._change_context(lambda: self._load_config(str(config_path)))
        else:
            print(f"Config not found: {config_path}")

    def _on_load_config_from(self) -> None:
        """Load configuration from a specific file."""
        path = QFileDialog.getOpenFileName(self, "Load Config From", filter="JSON (*.json)")[0]
        if path:
            self._change_context(lambda: self._load_config(path))

    def _save_config(self, path: str) -> None:
        """Save state to a JSON file."""
        state = self.get_state()
        try:
            with open(path, "w") as f:
                json.dump(state, f, indent=4)
            show_info(f"Config saved to {path}")
        except Exception as e:
            show_error(f"Error saving config: {e}")

    def _load_config(self, path: str) -> None:
        """Load state from a JSON file."""
        # Defense-in-depth: loading rewrites position + every section's params,
        # which would corrupt an in-progress correction. Callers route through
        # ``_change_context`` (which exits the owner first); refuse any path
        # that reaches here while a viewer owner is still active.
        if not self.gate.can_change_context():
            show_warning("Refusing to load config while a viewer mode is active.")
            return
        try:
            with open(path) as f:
                state = json.load(f)
            self.set_state(state)
            show_info(f"Config loaded from {path}")
        except Exception as e:
            show_error(f"Error loading config: {e}")

    def _refresh_all(self) -> None:
        """Refresh file status in all child widgets."""
        pos_dir = self._pos_dir

        self.data_panel.refresh(pos_dir)
        self._cellpose_widget.refresh(pos_dir)
        self.nucleus_workflow_widget.refresh(pos_dir)
        self.cell_workflow_widget.refresh(pos_dir)
        # The contact piece is position-agnostic; the orchestrator maps the
        # staged layout onto its explicit working context.
        if pos_dir is not None:
            self.contact_analysis_widget.set_context(
                cell_labels=pos_dir / "3_cell" / "tracked_labels.tif",
                nucleus_labels=pos_dir / "2_nucleus" / "tracked_labels.tif",
                out_path=pos_dir / "aggregate_quantification" / "contact_analysis.h5",
                status_root=pos_dir,
            )
        else:
            self.contact_analysis_widget.set_context(
                cell_labels=None, nucleus_labels=None, out_path=None, status_root=None
            )
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
