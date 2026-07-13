"""Main widget for the CellFlow napari plugin."""
from __future__ import annotations

import json
from pathlib import Path

import napari
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.contact_analysis.catalog import (
    CONTACT_ANALYSIS_RELPATH,
    columns_from_levels,
    load_catalog,
    merge_catalog_records,
    relative_levels,
    save_catalog,
)
from cellflow.napari._experiments_panel import ExperimentsPanel
from cellflow.napari._icons import stage_action_icon
from cellflow.napari.aggregate_widget import AggregateWidget
from cellflow.napari._stage_loader import load_stage
from cellflow.napari._stage_status import position_stage_status
from cellflow.napari.cellpose_widget import CellposeWidget
from cellflow.napari.contact_analysis_widget import ContactAnalysisWidget
from cellflow.napari.cell_workflow_widget import CellWorkflowWidget
from cellflow.napari.nucleus_workflow_widget import NucleusWorkflowWidget
from cellflow.napari.widgets import (
    CollapsibleSection,
    PipelineFilesWidget,
    pipeline_status_from_files,
)
from cellflow.napari._widget_helpers import tool_btn
from cellflow.napari.ui_gate import ControlClass, UiGate
from cellflow.napari.ui_style import (
    active_theme_name,
    muted_accent,
    muted_label,
    refresh_stage_header_labels,
    set_active_theme,
    stage_accent,
    theme_names,
)


#: Innermost nesting levels seeded with the recognized identity axes (innermost =
#: position_id), so the common ``condition / experiment_id / pos`` layout names
#: itself; deeper levels get generic ``level_k`` placeholders.
_SEED_LEVEL_NAMES = ("condition", "experiment_id", "position_id")

#: Committed-output paths relative to a position folder (see
#: ``cellflow.napari._paths.NucleusArtifactPaths.cell_labels`` / ``.nucleus_labels``
#: — the files the finalize/"commit" button writes), stamped onto catalog rows so
#: a project CSV saved from the full app carries everything the aggregate needs.
_CELL_LABELS_RELPATH = "cell_labels.tif"
_NUCLEUS_LABELS_RELPATH = "nucleus_labels.tif"

#: Seconds per minute — the Setup frame-length field is entered in minutes but the
#: backend (dynamics, persisted ``metadata.time_interval_s``) works in seconds.
_SECONDS_PER_MINUTE = 60.0


def _positive_float_or_none(value) -> float | None:
    """*value* as a positive float, or ``None`` when blank / non-numeric / ≤ 0."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _min_str_to_s_str(text) -> str:
    """A minutes field text → its seconds string for persistence (blank stays blank)."""
    minutes = _positive_float_or_none(text)
    return "" if minutes is None else _format_calibration(minutes * _SECONDS_PER_MINUTE)


def _s_str_to_min_str(text) -> str:
    """A persisted seconds value → its minutes string for the field (blank stays blank)."""
    seconds = _positive_float_or_none(text)
    return "" if seconds is None else _format_calibration(seconds / _SECONDS_PER_MINUTE)


def _format_calibration(value: float) -> str:
    """Compact numeric text (no trailing-zero noise), matching the panel's own."""
    return f"{value:g}"


def _seed_level_names(depth: int) -> list[str]:
    """Default name for each of *depth* nesting levels (anchored at the root)."""
    names = [f"level_{i + 1}" for i in range(depth)]
    for offset, seed in enumerate(reversed(_SEED_LEVEL_NAMES), start=1):
        if offset <= depth:
            names[depth - offset] = seed
    return names


def _discover_positions(root: str, input_names: dict[str, str]) -> list[dict]:
    """Find CellFlow position folders under *root* by their raw input files.

    A *position* is any folder containing at least one named raw input — the
    nucleus and/or cell image (each a bare file name or a path relative to the
    position folder, e.g. ``0_input/nucleus_3dt.tif``; the ``_3dt`` suffix varies
    with dimensionality — ``2d`` / ``2dt`` / ``3dt``). Each match becomes a panel
    entry whose columns are derived from its nesting under *root* (folder-derived
    columns, seeded with the recognized identity axes)."""
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    names = [n for n in input_names.values() if n]
    if not names:
        return []

    by_position: dict[Path, set[str]] = {}
    for name in names:
        rel = Path(name)
        for match in sorted(root_path.rglob(rel.name)):
            if not match.is_file():
                continue
            if len(rel.parts) > 1 and match.parts[-len(rel.parts):] != rel.parts:
                continue
            pos = match
            for _ in rel.parts:
                pos = pos.parent
            by_position.setdefault(pos.resolve(), set()).add(name)

    entries: list[dict] = []
    for pos in sorted(by_position):
        try:
            levels = relative_levels(root_path, pos)
        except ValueError:
            # The chosen root sits inside the position (the user picked the
            # position's own ``0_input`` folder, so the position is root's
            # parent): there is no nesting under root. Add it plainly, its own
            # folder name as the identity.
            levels = ()
        columns = columns_from_levels(_seed_level_names(len(levels)), levels)
        columns.setdefault("position_id", levels[-1] if levels else pos.name)
        entries.append(
            {"key": str(pos), "columns": columns, "payload": {"position_path": pos}}
        )
    return entries


class CellFlowMainWidget(QWidget):
    """The unified workflow-based UI for CellFlow."""

    refresh_requested = Signal(object)  # emits pos_dir: Path | None

    #: Preferred dock width on first open. This is only a *hint* — napari uses
    #: it to size the dock initially, but the user can still drag it narrower
    #: because minimumSizeHint (driven by the child widgets) is left untouched.
    _PREFERRED_WIDTH = 480

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(max(hint.width(), self._PREFERRED_WIDTH), hint.height())

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

        self.contact_analysis_widget = ContactAnalysisWidget(self.viewer, gate=self.gate)
        self.contact_analysis_section = CollapsibleSection(
            "Results",
            self.contact_analysis_widget,
            expanded=False,

            accent_color=stage_accent("contact_analysis"),
        )

        self.aggregate_widget = AggregateWidget()
        self.aggregate_section = CollapsibleSection(
            "Aggregate",
            self.aggregate_widget,
            expanded=False,
            accent_color=stage_accent("aggregate"),
        )

        # Positions panel (napariTFM ExperimentsList parity): discover a study
        # root, add its position folders to a list, and select one to drive the
        # detail sections below. The five stage sections ARE the selected
        # position's detail pane — selecting a row sets ``_pos_dir``.
        self._positions_panel = ExperimentsPanel(
            title="Data folders",
            input_fields=[
                ("nucleus", "Nucleus image", "0_input/nucleus_3dt.tif"),
                ("cell", "Cell image", "0_input/cell_3dt.tif"),
            ],
            discover_fn=_discover_positions,
            status_fn=lambda payload: position_stage_status(
                payload.get("position_path") if payload else None
            ),
            show_calibration=True,  # pixel size / frame length live in Setup
            show_run=False,  # batch Run selected is a later (stage-spine) pass
        )
        self._positions_panel.active_changed.connect(self._on_active_position)
        self._positions_panel.discover_requested.connect(self._on_discover_positions)
        self._positions_panel.stage_load_requested.connect(self._on_position_stage_load)
        self._positions_panel.records_changed.connect(self._refresh_aggregate)
        # Editing the Setup calibration after folders are added must re-stamp the
        # aggregate records, or param-gated quantities filled in afterwards stay grey.
        self._positions_panel.calibration_changed.connect(
            lambda *_: self._refresh_aggregate()
        )

        # Viewer-activity banner sits at the top level, above the stage
        # sections, so the "exit the active mode" hint is visible regardless of which section
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

        self.scroll_layout.addWidget(self._positions_panel)
        self.scroll_layout.addWidget(self.cellpose_section)
        self.scroll_layout.addWidget(self.nucleus_section)
        self.scroll_layout.addWidget(self.cell_section)
        self.scroll_layout.addWidget(self.contact_analysis_section)
        self.scroll_layout.addWidget(self.aggregate_section)

        # The five stage sections ARE the selected-position detail pane: they
        # stay hidden until a position is active (progressive disclosure — an
        # empty workspace shows only the toolbar + the positions panel).
        self._stage_sections = (
            self.cellpose_section,
            self.nucleus_section,
            self.cell_section,
            self.contact_analysis_section,
        )
        for section in self._stage_sections:
            section.set_status("not_started")

        # The aggregate capstone is project-level, not a per-position detail
        # pane: it is NOT in ``_stage_sections`` (so ``_update_disclosure``
        # never hides it on selection changes). Seed its own initial state,
        # hidden until the catalog has positions.
        self.aggregate_section.setVisible(False)
        self.aggregate_section.set_status(self.aggregate_widget.section_status())

        # Add stretch at the end
        self.scroll_layout.addStretch()
        self._setup_theme_selector(main_layout)

        # Connect signals
        self.save_as_btn.clicked.connect(lambda: self._on_save_config_as())
        self.load_from_btn.clicked.connect(lambda: self._on_load_config_from())
        self.save_project_btn.clicked.connect(lambda: self._on_save_project())
        self.load_project_btn.clicked.connect(lambda: self._on_load_project())

        self.refresh_btn.clicked.connect(lambda: self._refresh_all())

        # Config travels with the folder: snapshot params into the folder on every
        # run. The stage widgets' run buttons are the trigger (cancel re-clicks
        # re-save the same config, which is harmless).
        for widget_obj, attr in (
            (self._cellpose_widget, "nucleus_run_btn"),
            (self._cellpose_widget, "cell_run_btn"),
            (self.nucleus_workflow_widget, "seg_run_btn"),
            (self.nucleus_workflow_widget, "db_run_btn"),
            (self.nucleus_workflow_widget, "solve_run_btn"),
            (self.cell_workflow_widget, "run_btn"),
        ):
            getattr(widget_obj, attr).clicked.connect(self._autosave_config)

        self._register_gate_controls()

        self.gate.changed.connect(self._update_activity_banner)
        self._update_activity_banner()
        self._update_disclosure()
        self._connect_status_trackers()

    def _update_disclosure(self) -> None:
        """Reveal the stage sections only once a position is active.

        Empty workspace → only the toolbar and the positions panel show; the
        five stage sections (the selected-position detail pane) appear the moment
        a row is selected and hide again when the selection is cleared.
        """
        tuning = self._pos_dir is not None
        for section in self._stage_sections:
            section.setVisible(tuning)

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
            self.load_project_btn,
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
        """The top toolbar: a title + compact Project / Params action buttons.

        Calibration (pixel size / frame length) has moved into the positions
        panel's Setup section. Data folders are added via the panel's
        Discover-and-select flow, so the toolbar carries just the project
        catalog load/save, the standalone params-file load/save, and a global
        refresh.
        """
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        title = QLabel("CellFlow")
        title.setStyleSheet("font-weight: bold;")
        row.addWidget(title)

        row.addWidget(self._toolbar_group_label("Project"))
        # The load/save pair act on the PROJECT catalog CSV (the Data folders
        # list + classification columns), not the per-folder config.
        self.load_project_btn = self._toolbar_icon_btn(
            "load", "Load a project catalog (CSV) into the Data folders list"
        )
        self.save_project_btn = self._toolbar_icon_btn(
            "save", "Save the Data folders list to a project catalog (CSV)"
        )
        for btn in (self.load_project_btn, self.save_project_btn):
            row.addWidget(btn)

        row.addWidget(self._toolbar_group_label("Params"))
        # Per-folder config now autosaves on run; these move a tuned param SET
        # between experiments as a standalone file.
        self.load_from_btn = self._toolbar_icon_btn("load", "Load config from a file…")
        self.save_as_btn = self._toolbar_icon_btn("save", "Save config to a file…")
        for btn in (self.load_from_btn, self.save_as_btn):
            row.addWidget(btn)

        row.addStretch()
        self.refresh_btn = self._toolbar_icon_btn("reset", "Refresh all status")
        row.addWidget(self.refresh_btn)

        layout.addWidget(bar)

        self.path_label = QLabel("[no folder]")
        muted_label(self.path_label)
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

    #: Toolbar icon geometry — matches napariTFM's title-row buttons.
    _TOOLBAR_ICON_SIZE = 18
    _TOOLBAR_ICON_STROKE = 2.0

    def _toolbar_icon_btn(self, icon_name: str, tooltip: str) -> QToolButton:
        """A compact, icon-only, auto-raised toolbar button (napariTFM parity)."""
        button = QToolButton()
        accent = stage_accent("project_status")
        button.setIcon(
            stage_action_icon(
                icon_name,
                muted_accent(accent),
                disabled_color=muted_accent(muted_accent(accent)),
                size=self._TOOLBAR_ICON_SIZE,
                stroke_width=self._TOOLBAR_ICON_STROKE,
            )
        )
        button.setIconSize(QSize(self._TOOLBAR_ICON_SIZE, self._TOOLBAR_ICON_SIZE))
        button.setToolTip(tooltip)
        button.setAutoRaise(True)
        return button

    def _toolbar_group_label(self, text: str) -> QLabel:
        label = QLabel(text)
        muted_label(label)
        return label

    def _on_discover_positions(self) -> None:
        """Find-data-folders button: pick a parent directory and scan it."""
        path = QFileDialog.getExistingDirectory(
            self, "Select a parent folder to scan for data folders"
        )
        if path:
            self._positions_panel.discover(path)

    def _on_active_position(self, payload) -> None:
        """A positions-panel row was activated: make it the working ``pos_dir``.

        The selected position folder *is* ``pos_dir`` — the detail sections below
        retarget to it, exactly as the manual folder picker does. Routed through
        ``_change_context`` so an active viewer owner is offered an exit first.
        """
        if not payload or not payload.get("position_path"):
            return
        pos = Path(payload["position_path"])

        def action() -> None:
            self._pos_dir = pos
            self._update_disclosure()
            self.path_label.setText(str(pos))
            self.path_label.setToolTip(str(pos))
            config_path = pos / "cellflow_config.json"
            if config_path.exists():
                self._load_config(str(config_path))
            self._refresh_all()

        self._change_context(action)

    def _on_position_stage_load(self, payload, stage: str) -> None:
        """Click a row's rail dot → load that stage's output(s) into the viewer."""
        if not payload or not payload.get("position_path"):
            return
        load_stage(self.viewer, Path(payload["position_path"]), stage)

    def get_state(self) -> dict:
        """Return the current UI state as a dictionary."""
        calibration = self._positions_panel.calibration_values()
        return {
            "metadata": {
                "pixel_size_um": calibration.get("pixel_size_um", ""),
                # Panel holds minutes; persist the backend's seconds.
                "time_interval_s": _min_str_to_s_str(calibration.get("time_interval_min", "")),
            },
            "cellpose": self._cellpose_widget.get_state(),
            "nucleus": self.nucleus_workflow_widget.get_state(),
            "cell": self.cell_workflow_widget.get_state(),
        }

    def set_state(self, state: dict) -> None:
        """Update the UI state from a dictionary."""
        if "metadata" in state:
            m = state["metadata"]
            # Calibration now lives in the positions panel's Setup section. The
            # config stores seconds; the panel field is minutes, so translate.
            self._positions_panel.set_calibration_values(
                {
                    "pixel_size_um": m.get("pixel_size_um", ""),
                    "time_interval_min": _s_str_to_min_str(m.get("time_interval_s", "")),
                }
            )
            # Legacy ``condition`` / ``position`` keys are intentionally ignored:
            # condition is a folder-derived per-position column, and the selected
            # folder carries position identity.

        if "cellpose" in state:
            self._cellpose_widget.set_state(state["cellpose"])

        if "nucleus" in state:
            self.nucleus_workflow_widget.set_state(state["nucleus"])

        if "cell" in state:
            self.cell_workflow_widget.set_state(state["cell"])

    def _autosave_config(self) -> None:
        """Write the current config into the active folder (quiet) on each run."""
        if self._pos_dir is None:
            return
        self._save_config(str(self._pos_dir / "cellflow_config.json"), quiet=True)

    def _catalog_record_for_position(self, position_path: Path, columns: dict) -> dict:
        """A catalog record for one data folder, stamped with committed output paths.

        Panel rows carry only ``position_path`` + classification ``columns``; the
        aggregate catalog also needs the contact-analysis ``.h5`` and the two label
        images. Those are the *committed* outputs — ``cell_labels.tif`` /
        ``nucleus_labels.tif`` (what the finalize/"commit" button writes; see
        ``NucleusArtifactPaths.cell_labels`` / ``.nucleus_labels``) and the
        canonical ``4_contact_analysis/contact_analysis.h5`` — not the pre-commit
        working paths under ``2_nucleus`` / ``3_cell``. Fill their defaults here
        (whether or not the folder has been processed yet).
        """
        pos = Path(position_path)
        return {
            "position_path": pos,
            "contact_analysis_path": pos / CONTACT_ANALYSIS_RELPATH,
            "cell_tracked_labels_path": pos / _CELL_LABELS_RELPATH,
            "nucleus_tracked_labels_path": pos / _NUCLEUS_LABELS_RELPATH,
            "columns": dict(columns or {}),
        }

    def _catalog_records_for_panel(self, panel_records) -> list[dict]:
        """Catalog records (committed output paths + calibration) for panel rows.

        The Setup calibration (pixel size, frame length) is stamped onto every
        record so the aggregate's pooled cheap quantities (shape, dynamics) compute
        in physical units and light up in the panel; without it they stay greyed.
        """
        calibration = self._calibration_params()
        records = []
        for rec in panel_records:
            record = self._catalog_record_for_position(
                rec["position_path"], rec.get("columns", {})
            )
            record.update(calibration)
            records.append(record)
        return records

    def _calibration_params(self) -> dict[str, float]:
        """Setup calibration as backend ``{param: float}``, dropping blank /
        non-positive entries (an unset field contributes nothing rather than a zero).

        The frame-length field is entered in minutes; the backend's dynamics work
        in seconds, so it is converted to ``time_interval_s`` here.
        """
        values = self._positions_panel.calibration_values()
        params: dict[str, float] = {}
        pixel_size = _positive_float_or_none(values.get("pixel_size_um", ""))
        if pixel_size is not None:
            params["pixel_size_um"] = pixel_size
        minutes = _positive_float_or_none(values.get("time_interval_min", ""))
        if minutes is not None:
            params["time_interval_s"] = minutes * 60.0
        return params

    def _refresh_aggregate(self) -> None:
        """Feed the project-level catalog to the capstone; show it once positions exist."""
        records = self._catalog_records_for_panel(self._positions_panel.records())
        self.aggregate_widget.set_records(records)
        self.aggregate_section.setVisible(bool(records))
        self.aggregate_section.set_status(self.aggregate_widget.section_status())

    def _on_save_config_as(self) -> None:
        """Save current configuration to a specific file."""
        path = QFileDialog.getSaveFileName(self, "Save Config As", filter="JSON (*.json)")[0]
        if path:
            self._save_config(path)

    def _on_load_config_from(self) -> None:
        """Load configuration from a specific file."""
        path = QFileDialog.getOpenFileName(self, "Load Config From", filter="JSON (*.json)")[0]
        if path:
            self._change_context(lambda: self._load_config(path))

    def _on_save_project(self) -> None:
        """Write the Data folders catalog to a CSV the aggregate studio can run."""
        if not self._positions_panel.keys():
            show_warning("No data folders to save: the catalog is empty.")
            return
        path = QFileDialog.getSaveFileName(
            self, "Save project catalog", "catalog.csv", filter="CSV (*.csv)"
        )[0]
        if not path:
            return
        # getSaveFileName does not always append the filter suffix on Linux/Qt.
        if not path.lower().endswith(".csv"):
            path = f"{path}.csv"
        records = self._catalog_records_for_panel(self._positions_panel.records())
        try:
            save_catalog(Path(path), records)
            show_info(f"Project saved to {path}")
        except Exception as e:
            show_error(f"Error saving project: {e}")

    def _on_load_project(self) -> None:
        """Load a project catalog CSV, merging its rows into the Data folders list."""
        path = QFileDialog.getOpenFileName(
            self, "Load project catalog", filter="CSV (*.csv)"
        )[0]
        if not path:
            return

        def action() -> None:
            try:
                loaded = load_catalog(Path(path))
            except Exception as e:
                show_error(f"Error loading project: {e}")
                return
            merged = merge_catalog_records(self._positions_panel.records(), loaded)
            entries = [
                {
                    "key": str(rec.get("position_path") or rec["id"]),
                    "columns": dict(rec.get("columns") or {}),
                    "payload": {"position_path": Path(rec["position_path"])}
                    if rec.get("position_path")
                    else {"position_path": None},
                }
                for rec in merged
            ]
            self._positions_panel.set_records(entries)
            show_info(f"Project loaded: {len(entries)} data folder(s).")

        self._change_context(action)

    def _save_config(self, path: str, *, quiet: bool = False) -> None:
        """Save state to a JSON file. ``quiet`` suppresses the success toast."""
        state = self.get_state()
        try:
            with open(path, "w") as f:
                json.dump(state, f, indent=4)
            if not quiet:
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

    def _connect_status_trackers(self) -> None:
        """Recompute aggregated status whenever any stage's tracker refreshes.

        Every stage widget refreshes its ``PipelineFilesWidget`` after a run,
        commit, or finalize (the moments its on-disk output changes). Hooking
        each tracker's ``refreshed`` signal is what keeps the section dots and
        the catalog rail live: without it they stay frozen at whatever they
        showed when the position was last selected, so committing a label or
        running a contact analysis left the circles stale until a manual
        Refresh. Connecting the whole tree (via ``findChildren``) catches nested
        trackers too, e.g. the divergence-maps widget inside Cellpose.
        """
        for tracker in self.findChildren(PipelineFilesWidget):
            tracker.refreshed.connect(self._on_stage_files_refreshed)

    def _on_stage_files_refreshed(self) -> None:
        """A stage's on-disk output changed → repaint section dots + rail.

        Suppressed during ``_refresh_all`` (which drives every tracker and then
        recomputes once itself) so a bulk refresh doesn't re-read the catalog
        per tracker.
        """
        if getattr(self, "_refreshing_all", False):
            return
        self._update_section_statuses()
        self._positions_panel.refresh_statuses()

    def _refresh_all(self) -> None:
        """Refresh file status in all child widgets."""
        pos_dir = self._pos_dir

        self._refreshing_all = True
        try:
            self._cellpose_widget.refresh(pos_dir)
            self.nucleus_workflow_widget.refresh(pos_dir)
            self.cell_workflow_widget.refresh(pos_dir)
            # The contact piece is position-agnostic; the orchestrator maps the
            # staged layout onto its explicit working context.
            if pos_dir is not None:
                self.contact_analysis_widget.set_context(
                    cell_labels=pos_dir / "3_cell" / "tracked_labels.tif",
                    nucleus_labels=pos_dir / "2_nucleus" / "tracked_labels.tif",
                    out_path=pos_dir / CONTACT_ANALYSIS_RELPATH,
                    status_root=pos_dir,
                )
            else:
                self.contact_analysis_widget.set_context(
                    cell_labels=None, nucleus_labels=None, out_path=None, status_root=None
                )
            self._update_section_statuses()
        finally:
            self._refreshing_all = False
        # The catalog rail reads on-disk status per row directly; keep it in
        # step with the section dots on every full refresh.
        self._positions_panel.refresh_statuses()
        self._refresh_aggregate()
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
