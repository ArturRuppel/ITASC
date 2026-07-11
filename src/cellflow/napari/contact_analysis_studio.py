"""Contact Analysis studio: a position catalog + per-position view + analysis plugins.

This is the merged standalone Contact Analysis tool. It is built from two parts:

* a **catalog** of positions (autodiscover a study tree, add loose ``.h5`` files,
  load/save a CSV catalog);
* a **plugins** section that hosts every analysis *plugin* fed with the
  currently-selected catalog rows — each plugin is one collapsible whose header
  is its on/off control. The per-position visualizer is itself a plugin
  (``visualize_contacts``), as is all cross-position aggregation (see
  :mod:`cellflow.napari.contact_analysis.plugins`).

This module is full-install only (it depends on the napari analysis-plugin
package, which the standalone ``cellflow-aggregate`` wheel does not ship); the
standalone wheel falls back to the bare
:class:`ContactAnalysisWidget` (see ``make_contact_analysis_widget``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from napari.qt.threading import thread_worker
from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cellflow.contact_analysis.quantifier import (
    available_quantifiers,
)
from cellflow.contact_analysis.catalog import (
    columns_from_levels,
    discover_catalog_entries,
    discovered_level_depth,
    load_catalog,
    merge_catalog_records,
    relative_levels,
    save_catalog,
)
from cellflow.contact_analysis.pipeline import author_config, run
from cellflow.contact_analysis.shape_tables import catalogue_root
from cellflow.napari._experiments_panel import ExperimentsPanel
from cellflow.napari._stage_loader import load_stage
from cellflow.napari._stage_status import position_stage_status
from cellflow.napari.contact_analysis.plugins import AnalysisContext
from cellflow.napari.contact_analysis_params import SharedParamsWidget
from cellflow.napari.contact_analysis_run_area import RunArea
from cellflow.napari.contact_analysis_widget import _ProgressEmitter
from cellflow.napari.studio_plugins import (
    PluginEntry,
    available_tool_plugins,
    records_satisfying,
)
from cellflow.napari._icons import stage_action_icon
from cellflow.napari.ui_style import (
    action_button,
    muted_accent,
    stage_accent,
    status_label,
)
from cellflow.napari.widgets import CollapsibleSection


class _PluginSection(NamedTuple):
    """One plugin's collapsible section + its body widget.

    Every available plugin gets a section up front; the section's own collapse
    header is the on/off control (there is no separate checkbox), and expanding
    it reveals the body.
    """

    entry: PluginEntry
    section: QWidget
    body: QWidget


def _inputs_label(record: dict) -> str:
    """Compact ``cell·nuc`` summary of which source inputs a position has."""
    parts = []
    if record.get("cell_tracked_labels_path"):
        parts.append("cell")
    if record.get("nucleus_tracked_labels_path"):
        parts.append("nuc")
    return "·".join(parts) if parts else "—"


#: The innermost nesting levels are seeded with the recognized identity axes
#: (innermost = position_id), so the common ``condition / experiment_id / pos``
#: layout names itself; deeper levels get generic ``level_k`` placeholders.
_SEED_LEVEL_NAMES = ("condition", "experiment_id", "position_id")


def _seed_level_names(depth: int) -> list[str]:
    """Default name for each of *depth* nesting levels (anchored at the root)."""
    names = [f"level_{i + 1}" for i in range(depth)]
    for offset, seed in enumerate(reversed(_SEED_LEVEL_NAMES), start=1):
        if offset <= depth:
            names[depth - offset] = seed
    return names


def _contacts_reader():
    """The contacts quantifier's reader, used by analysis-plugin contexts.

    Sourced from the registry (contacts is the only quantity today); falls back
    to the contacts reader import so a context loader is always available.
    """
    for cls in available_quantifiers():
        if cls.quantity_id == "contacts":
            return cls().read
    from cellflow.contact_analysis.contacts.reader import (
        read_position_contacts,
    )

    return read_position_contacts


class ContactAnalysisStudioWidget(QWidget):
    """Position catalogue + embedded per-position view + a flat plugin list."""

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        #: Cache so plugins don't each re-open the same HDF5.
        self._analysis_cache: dict[Path, Any] = {}
        #: Reader feeding analysis-plugin contexts (contacts is the only quantity today).
        self._read_quantity = _contacts_reader()
        #: Flat plugin list: every available plugin as its own collapsible.
        self._plugin_entries: list[PluginEntry] = []
        self._plugin_sections: dict[str, _PluginSection] = {}
        #: Background full-pipeline run (author config, then pipeline.run()).
        self._run_worker = None
        self._run_emitter = _ProgressEmitter(self)
        self._run_emitter.progress.connect(self._on_run_progress)

        # The two regions can grow tall (embedded visualizer + several mounted
        # plugin collapsibles), so the whole studio scrolls vertically.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        scroll_content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        scroll.setWidget(scroll_content)
        outer.addWidget(scroll)

        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignTop)

        # One Catalogue region: the shared ExperimentsPanel (Setup → Discover →
        # Add → list) + a small Load/Save/Clear action row beneath it. The panel
        # owns the displayed catalog; the studio reads it for scope / run / save.
        catalogue = QWidget()
        cat_col = QVBoxLayout(catalogue)
        cat_col.setContentsMargins(0, 0, 0, 0)
        cat_col.setSpacing(4)
        self._build_catalogue_section(cat_col)
        layout.addWidget(CollapsibleSection("Catalogue", catalogue, expanded=True))

        # One shared parameter bar above both areas: builds and plots read the
        # same pixel size / frame interval (plus plot-only FOV / shuffles).
        self._shared_params = SharedParamsWidget()
        self._shared_params.changed.connect(self._push_context)
        self._params_section = CollapsibleSection(
            "Parameters", self._shared_params, expanded=True
        )
        layout.addWidget(self._params_section)

        # Tools on top, then the Run area.
        # (Plotting is Iris-only — no in-napari Plot area.)
        self._build_tools_section(layout)
        self._build_run_section(layout)

        # Progressive disclosure: Parameters / Tools / Run only make sense once
        # the catalogue holds positions, so they stay hidden until it does — an
        # empty studio shows only the Catalogue front door.
        self._gated_sections = (
            self._params_section,
            self._tools_section,
            self._run_section,
        )
        self._panel.records_changed.connect(self._update_disclosure)

        self._reload_plugins()
        self._push_context()
        self._update_disclosure()

    def _update_disclosure(self) -> None:
        """Reveal the downstream sections only once the catalogue is non-empty."""
        has_rows = bool(self._panel.keys())
        for section in self._gated_sections:
            section.setVisible(has_rows)

    # ----------------------------------------------------------------- catalog UI
    def _build_catalogue_section(self, layout) -> None:
        """Mount the shared ExperimentsPanel + a Load/Save/Clear action row.

        The panel runs the filesystem-centric flow (Setup → Find data folders → an
        additive, de-duped list with a per-folder status rail). It owns the
        displayed catalog and its own
        Delete-selected button; the studio only reads it (for scope / run / save)
        and reacts to its signals. Discovery inputs default to the committed final
        outputs (``cell_labels.tif`` / ``nucleus_labels.tif``) so a finalized
        data folder is auto-discovered; override to a working stage file when
        uncommitted.
        """
        self._panel = ExperimentsPanel(
            title="Data folders",
            input_fields=[
                ("cell", "Cell labels", "cell_labels.tif"),
                ("nucleus", "Nucleus labels", "nucleus_labels.tif"),
            ],
            discover_fn=self._discover,
            status_fn=self._status,
            show_calibration=False,
            show_run=False,
            show_output_dir=False,
        )
        self._panel.discover_requested.connect(self._on_discover_requested)
        self._panel.selection_changed.connect(self._push_context)
        self._panel.records_changed.connect(self._push_context)
        self._panel.stage_load_requested.connect(self._on_stage_load_requested)
        layout.addWidget(self._panel)

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(2)
        load_btn = QPushButton("Load CSV…")
        action_button(load_btn)
        load_btn.clicked.connect(self._on_load_csv)
        save_btn = QPushButton("Save CSV…")
        action_button(save_btn)
        save_btn.clicked.connect(self._on_save_csv)
        clear_btn = QPushButton("Clear")
        action_button(clear_btn)
        clear_btn.clicked.connect(self._on_clear_catalog)
        for btn in (load_btn, save_btn, clear_btn):
            actions_row.addWidget(btn)
        layout.addLayout(actions_row)

        self._catalog_status_lbl = QLabel("")
        self._catalog_status_lbl.setWordWrap(True)
        status_label(self._catalog_status_lbl)
        layout.addWidget(self._catalog_status_lbl)

    # -------------------------------------------------------- panel host contract
    def _discover(self, root: str, input_names: dict) -> list[dict]:
        """Panel ``discover_fn``: every data folder found under *root*.

        Each entry carries folder-derived columns (one per named nesting level,
        seeded with the recognized identity axes) plus a ``position_id`` pinned to
        the innermost folder name so folders keep distinct identities.
        """
        cell = input_names.get("cell") or None
        nucleus = input_names.get("nucleus") or None
        if not cell and not nucleus:
            return []
        entries = discover_catalog_entries(root, cell_name=cell, nucleus_name=nucleus)
        staged: list[dict] = []
        for entry in entries:
            levels = relative_levels(root, entry["position_path"])
            columns = columns_from_levels(_seed_level_names(len(levels)), levels)
            columns.setdefault("position_id", levels[-1] if levels else entry["id"])
            staged.append(
                {
                    "key": str(entry["position_path"]),
                    "columns": columns,
                    "payload": entry,
                }
            )
        return staged

    def _status(self, payload) -> dict:
        """Panel ``status_fn``: the per-stage rail status for one row's payload."""
        pos_dir = payload.get("position_path") if isinstance(payload, dict) else None
        return position_stage_status(pos_dir)

    def _on_discover_requested(self) -> None:
        """Find-data-folders button: pick a parent, guard uniform depth, then add."""
        root = QFileDialog.getExistingDirectory(
            self, "Select a parent folder to scan for data folders"
        )
        if not root:
            return
        names = self._panel.input_names()
        cell = names.get("cell") or None
        nucleus = names.get("nucleus") or None
        try:
            entries = discover_catalog_entries(root, cell_name=cell, nucleus_name=nucleus)
        except Exception as exc:
            self._set_catalog_status(f"Discover error: {exc}")
            return
        if entries and discovered_level_depth(
            root, [e["position_path"] for e in entries]
        ) is None:
            self._set_catalog_status(
                "Discovered positions sit at differing folder depths; the levels "
                "cannot line up. Scan a root where every position is equally nested."
            )
            return
        self._panel.discover(root)

    def _on_stage_load_requested(self, payload, stage: str) -> None:
        """A rail dot click → load that stage's output(s) for the position."""
        pos_dir = payload.get("position_path") if isinstance(payload, dict) else None
        if pos_dir is None:
            ident = payload.get("id", "position") if isinstance(payload, dict) else "position"
            self._set_catalog_status(f"{ident}: no canonical folder, cannot load.")
            return
        loaded = load_stage(self.viewer, pos_dir, stage)
        if loaded:
            self._set_catalog_status(f"Loaded {', '.join(loaded)}.")
        elif stage == "contacts":
            self._set_catalog_status(
                "Contact analysis has no image to load — use Visualize Contacts."
            )
        else:
            self._set_catalog_status(f"Nothing to load for {stage} yet.")

    def _all_records(self) -> list[dict]:
        """Every committed row as a normalized catalog record (row order)."""
        return merge_catalog_records([], self._panel.records())

    def _records_in_scope(self) -> list[dict]:
        """Selected rows define scope; an empty selection means the whole catalog."""
        return merge_catalog_records(
            [], self._panel.selected_records() or self._panel.records()
        )

    def _build_tools_section(self, layout) -> None:
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(2)
        header.addWidget(QLabel("Use a tool:"))
        header.addStretch(1)
        reload_btn = QPushButton()
        reload_btn.setIcon(
            stage_action_icon("reset", muted_accent(stage_accent("contact_analysis")), size=16)
        )
        reload_btn.setIconSize(QSize(16, 16))
        reload_btn.setToolTip("Re-scan for tools and metrics.")
        action_button(reload_btn)
        reload_btn.clicked.connect(self._reload_plugins)
        header.addWidget(reload_btn)
        col.addLayout(header)

        # One collapsible per available tool; its header is the on/off control.
        # Pin them to the top (like the main app's stage list) so an expanded
        # tool grows to its content instead of the sections sharing — and
        # spreading into — the leftover vertical space.
        self._plugin_host = QWidget()
        self._plugin_host_layout = QVBoxLayout(self._plugin_host)
        self._plugin_host_layout.setContentsMargins(0, 0, 0, 0)
        self._plugin_host_layout.setSpacing(0)
        self._plugin_host_layout.setAlignment(Qt.AlignTop)
        col.addWidget(self._plugin_host)

        self._tools_section = CollapsibleSection("Tools", container, expanded=False)
        layout.addWidget(self._tools_section)

    def _build_run_section(self, layout) -> None:
        """The single Run area: author catalog.csv + config.toml from the whole
        catalogue and drive ``pipeline.run`` on a worker. Re-created in
        :meth:`_reload_run_area` so a runtime-registered quantity appears."""
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        self._run_host = QWidget()
        self._run_host_layout = QVBoxLayout(self._run_host)
        self._run_host_layout.setContentsMargins(0, 0, 0, 0)
        self._run_host_layout.setSpacing(0)
        self._run_area: RunArea | None = None
        col.addWidget(self._run_host)
        self._run_section = CollapsibleSection("Run", container, expanded=True)
        layout.addWidget(self._run_section)

    def _reload_run_area(self) -> None:
        """(Re)create the Run area body from the quantifier registry."""
        while self._run_host_layout.count():
            item = self._run_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._run_area = RunArea(
            save_callback=self._on_run_save,
            run_callback=self._on_run_execute,
        )
        self._run_host_layout.addWidget(self._run_area)
        self._push_context_to(self._run_area)

    def _author_run_config(self, choices) -> Path:
        """Write catalog.csv + config.toml for the whole catalogue; return config path.

        When the user picked an output directory, the catalog/config and the pooled
        tables all live there (the config's ``out_dir`` is ``"."``, so tables land
        flat beside the config). Blank falls back to the catalogue root.
        """
        chosen = (choices.out_dir or "").strip()
        records = self._all_records()
        project_dir = Path(chosen) if chosen else catalogue_root(records)
        params = self._shared_params.build_params()
        return author_config(
            project_dir,
            records,
            tables_dir="." if chosen else None,
            quantities=choices.quantities,
            params=params,
        )

    def _on_run_save(self, choices) -> None:
        if not self._panel.keys():
            self._run_area.set_status("Add positions to the catalogue first.")
            return
        try:
            config_path = self._author_run_config(choices)
        except Exception as exc:
            self._run_area.set_status(f"Save error: {exc}")
            return
        self._run_area.set_status(f"Wrote {config_path.name} + catalog.csv.")

    def _on_run_execute(self, choices) -> None:
        if self._run_worker is not None:
            self._run_area.set_status("A run is already in progress.")
            return
        if not self._panel.keys():
            self._run_area.set_status("Add positions to the catalogue first.")
            return
        try:
            config_path = self._author_run_config(choices)
        except Exception as exc:
            self._run_area.set_status(f"Save error: {exc}")
            return
        self._run_area.set_status("Running pipeline…")
        emit = self._run_emitter.progress.emit

        @thread_worker(
            connect={"returned": self._on_run_done, "errored": self._on_run_error}
        )
        def _worker():
            return run(config_path, progress_cb=emit)

        self._run_worker = _worker()

    def _on_run_progress(self, done: int, total: int, label: str) -> None:
        self._run_area.set_status(f"Running: {done}/{total} {label}")

    def _on_run_done(self, written: dict) -> None:
        self._run_worker = None
        n = len(written)
        self._run_area.set_status(
            f"Wrote {n} pooled table{'s' if n != 1 else ''}."
            if n
            else "Run finished; no tables written."
        )

    def _on_run_error(self, exc: Exception) -> None:
        self._run_worker = None
        self._run_area.set_status(f"Run error: {exc}")

    # ----------------------------------------------------------- catalog actions
    def _set_catalog_status(self, message: str) -> None:
        self._catalog_status_lbl.setText(message)

    def _on_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load catalog CSV", filter="CSV files (*.csv)"
        )
        if path:
            self._load_csv_from(Path(path))

    def _load_csv_from(self, path: Path) -> None:
        """Merge a CSV catalog into the panel's current rows."""
        try:
            loaded = load_catalog(path)
        except Exception as exc:
            self._set_catalog_status(f"Load error: {exc}")
            return
        merged = merge_catalog_records(self._panel.records(), loaded)
        entries = [
            {
                "key": str(rec.get("position_path") or rec["id"]),
                "columns": dict(rec.get("columns") or {}),
                "payload": rec,
            }
            for rec in merged
        ]
        self._panel.set_records(entries)
        self._set_catalog_status(
            f"Catalog: {len(merged)} position(s) (loaded {len(loaded)} from CSV)."
        )

    def _on_save_csv(self) -> None:
        if not self._panel.keys():
            self._set_catalog_status("Nothing to save: the catalog is empty.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save catalog CSV", filter="CSV files (*.csv)"
        )
        if path:
            # getSaveFileName does not always append the filter's suffix (notably
            # on Linux/Qt), so ensure the file actually ends up as a .csv.
            if not path.lower().endswith(".csv"):
                path = f"{path}.csv"
            self._save_csv_to(Path(path))

    def _save_csv_to(self, path: Path) -> None:
        records = self._all_records()
        try:
            save_catalog(path, records)
        except Exception as exc:
            self._set_catalog_status(f"Save error: {exc}")
            return
        self._set_catalog_status(f"Saved {len(records)} record(s) to {path.name}.")

    def _on_clear_catalog(self) -> None:
        self._panel.set_records([])
        self._analysis_cache.clear()
        self._set_catalog_status("Catalog cleared.")

    # --------------------------------------------------------------- scope
    def _scoped_records(self) -> list[dict]:
        """In-scope records stamped with the shared build params (pixel size /
        frame interval), so building and gating both see a manual override."""
        records = self._records_in_scope()
        shared = getattr(self, "_shared_params", None)
        return shared.stamp(records) if shared is not None else records

    # -------------------------------------------------------------- plugin hosting
    def _reload_plugins(self) -> None:
        """Rebuild the tool collapsibles and the Run area from the registries.

        Each tool gets one collapsed :class:`CollapsibleSection`; its header is
        the on/off control (no separate checkbox), and expanding it reveals the
        body. The Run area is a single body with one quantity checkbox per
        quantifier. Both are fed the catalogue scope whether expanded or not, so
        the view is always current the moment it is opened.
        """
        self._reload_run_area()

        while self._plugin_host_layout.count():
            item = self._plugin_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._plugin_sections = {}

        self._plugin_entries = available_tool_plugins()
        if not self._plugin_entries:
            placeholder = QLabel("No tools found.")
            status_label(placeholder, muted=True)
            self._plugin_host_layout.addWidget(placeholder)
            self._plugin_host_layout.addStretch()
            return
        for entry in self._plugin_entries:
            body = entry.factory(self.viewer)
            section = CollapsibleSection(entry.display_name, body, expanded=False)
            section._toggle.toggled.connect(
                lambda checked, e=entry: self._on_plugin_toggled(e, checked)
            )
            self._plugin_host_layout.addWidget(section)
            self._plugin_sections[entry.plugin_id] = _PluginSection(
                entry=entry, section=section, body=body
            )
            self._push_context_to(body)
        # Trailing stretch keeps the collapsibles tight at the top; the clear
        # loop above takes it back out on the next reload.
        self._plugin_host_layout.addStretch()
        self._update_plugin_availability()

    def _on_plugin_toggled(self, entry: PluginEntry, checked: bool) -> None:
        # Expanding is the "activate" gesture: refresh the body with the current
        # scope so it is up to date even if the scope changed while collapsed.
        if not checked:
            return
        plugin = self._plugin_sections.get(entry.plugin_id)
        if plugin is not None:
            self._push_context_to(plugin.body)

    def _load_analysis(self, path: Path) -> Any:
        path = Path(path)
        cached = self._analysis_cache.get(path)
        if cached is None:
            cached = self._read_quantity(path)
            self._analysis_cache[path] = cached
        return cached

    def _push_context(self) -> None:
        """Feed every tool + the Run area the current scope, refresh gating."""
        for plugin in self._plugin_sections.values():
            self._push_context_to(plugin.body)
        run_area = getattr(self, "_run_area", None)
        if run_area is not None:
            self._push_context_to(run_area)
        self._update_plugin_availability()

    def _push_context_to(self, body: QWidget) -> None:
        set_context = getattr(body, "set_context", None)
        if not callable(set_context):
            return
        set_context(
            AnalysisContext(
                records=self._scoped_records(),
                viewer=self.viewer,
                loader=self._load_analysis,
            )
        )

    def _update_plugin_availability(self) -> None:
        """Disable a plugin's header when no in-scope position has its inputs."""
        scope = self._scoped_records()
        for entry in self._plugin_entries:
            plugin = self._plugin_sections.get(entry.plugin_id)
            if plugin is None:
                continue
            satisfied = bool(records_satisfying(entry.requires, scope))
            toggle = plugin.section._toggle
            # Keep an already-expanded plugin collapsible even if it falls out of
            # scope (so it is never trapped open).
            toggle.setEnabled(satisfied or plugin.section.is_expanded)
            toggle.setToolTip(
                ""
                if satisfied
                else f"No in-scope position has: {', '.join(entry.requires)}"
            )
