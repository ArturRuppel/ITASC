"""Aggregate Quantification studio: a position catalog + per-position view + analysis plugins.

This is the merged standalone Aggregate Quantification tool. It is built from two parts:

* a **catalog** of positions (autodiscover a study tree, add loose ``.h5`` files,
  load/save a CSV catalog);
* a **plugins** section that hosts every analysis *plugin* fed with the
  currently-selected catalog rows — each plugin is one collapsible whose header
  is its on/off control. The per-position visualizer is itself a plugin
  (``visualize_contacts``), as is all cross-position aggregation (see
  :mod:`cellflow.napari.aggregate_quantification.plugins`).

This module is full-install only (it depends on the napari analysis-plugin
package, which the standalone ``cellflow-aggregate`` wheel does not ship); the
standalone wheel falls back to the bare
:class:`AggregateQuantificationWidget` (see ``make_aggregate_quantification_widget``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.quantifier import (
    PositionInputs,
    Quantifier,
    available_quantifiers,
)
from cellflow.aggregate_quantification.catalog import (
    discover_catalog_entries,
    load_catalog,
    merge_catalog_records,
    save_catalog,
)
from cellflow.napari.aggregate_quantification.plugins import AnalysisContext
from cellflow.napari.aggregate_quantification_widget import _ProgressEmitter
from cellflow.napari.studio_plugins import (
    PluginEntry,
    available_studio_plugins,
    output_for_record,
    position_inputs_from_record,
    records_satisfying,
)
from cellflow.napari.ui_style import action_button, status_label
from cellflow.napari.widgets import CollapsibleSection


class _BuildPlan(NamedTuple):
    """One position queued for a build: its inputs + chosen artifact path."""

    inputs: PositionInputs
    output: Path


class _BuildResult(NamedTuple):
    position: str
    status: str  # "built" | "failed"


class _PluginSection(NamedTuple):
    """One plugin's collapsible section + its body widget.

    Every available plugin gets a section up front; the section's own collapse
    header is the on/off control (there is no separate checkbox), and expanding
    it reveals the body.
    """

    entry: "PluginEntry"
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


def _contacts_reader():
    """The contacts quantifier's reader, used by analysis-plugin contexts.

    Sourced from the registry (contacts is the only quantity today); falls back
    to the contacts reader import so a context loader is always available.
    """
    for cls in available_quantifiers():
        if cls.quantity_id == "contacts":
            return cls().read
    from cellflow.aggregate_quantification.contacts.reader import (
        read_position_contact_analysis,
    )

    return read_position_contact_analysis


class AggregateQuantificationStudioWidget(QWidget):
    """Position catalogue + embedded per-position view + a flat plugin list."""

    _TABLE_COLUMNS = ("condition", "date", "id", "inputs", "notes", "status")

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        #: Full catalog (normalized records from cellflow.aggregate_quantification.catalog).
        self._records: list[dict] = []
        #: Last discovered (not-yet-added) collection awaiting metadata.
        self._pending_entries: list[dict] = []
        #: Cache so plugins don't each re-open the same HDF5.
        self._analysis_cache: dict[Path, Any] = {}
        #: Reader feeding analysis-plugin contexts (contacts is the only quantity today).
        self._read_quantity = _contacts_reader()
        #: Flat plugin list: every available plugin as its own collapsible.
        self._plugin_entries: list[PluginEntry] = []
        self._plugin_sections: dict[str, _PluginSection] = {}
        #: Background build triggered by a builder plugin.
        self._build_worker = None
        self._build_emitter = _ProgressEmitter(self)
        self._build_emitter.progress.connect(self._on_build_progress)

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

        # One Catalogue region: discover/add (nested, collapsed) + the positions
        # table + the per-position visualizer (nested). Plugins are the second
        # region.
        catalogue = QWidget()
        cat_col = QVBoxLayout(catalogue)
        cat_col.setContentsMargins(0, 0, 0, 0)
        cat_col.setSpacing(4)
        self._build_discover_section(cat_col)
        self._build_catalog_section(cat_col)
        layout.addWidget(CollapsibleSection("Catalogue", catalogue, expanded=True))

        self._build_plugins_section(layout)

        self._reload_plugins()
        self._refresh_table()

    # ----------------------------------------------------------------- catalog UI
    def _make_field_row(self, layout, label: str, default: str = "", placeholder: str = "") -> QLineEdit:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        lbl = QLabel(label)
        lbl.setFixedWidth(150)
        edit = QLineEdit()
        if default:
            edit.setText(default)
        if placeholder:
            edit.setPlaceholderText(placeholder)
        row.addWidget(lbl)
        row.addWidget(edit, 1)
        layout.addLayout(row)
        return edit

    def _build_discover_section(self, layout) -> None:
        """Discover positions by file name / relative path, then add as a collection."""
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        root_row = QHBoxLayout()
        root_row.setContentsMargins(0, 0, 0, 0)
        root_row.setSpacing(2)
        self._root_edit = QLineEdit()
        self._root_edit.setPlaceholderText("Root folder to scan…")
        browse_btn = QPushButton("Browse…")
        action_button(browse_btn)
        browse_btn.clicked.connect(self._on_browse_root)
        root_row.addWidget(QLabel("Root:"))
        root_row.addWidget(self._root_edit, 1)
        root_row.addWidget(browse_btn)
        col.addLayout(root_row)

        # Names or relative paths (relative to each position folder). All inputs
        # are optional — a position is any folder with at least one of them. The
        # contact analysis is the derived output path (built later via a builder
        # plugin, not on add).
        self._cell_name_edit = self._make_field_row(
            col, "Cell labels (optional):", placeholder="e.g. 3_cell/tracked_labels.tif"
        )
        self._nucleus_name_edit = self._make_field_row(
            col, "Nucleus labels (optional):", placeholder="e.g. 2_nucleus/tracked_labels.tif"
        )
        self._contact_name_edit = self._make_field_row(
            col, "Contact analysis (output):", default="contact_analysis.h5"
        )

        discover_btn = QPushButton("Discover")
        discover_btn.setToolTip(
            "Find every folder under the root that contains at least one of the "
            "named inputs; derive the contact-analysis output path."
        )
        action_button(discover_btn, expand=True)
        discover_btn.clicked.connect(self._on_discover)
        col.addWidget(discover_btn)

        self._discovered_list = QListWidget()
        self._discovered_list.setMaximumHeight(120)
        col.addWidget(self._discovered_list)

        # Metadata applied to the whole discovered collection before adding.
        self._condition_edit = self._make_field_row(col, "Condition:")
        self._date_edit = self._make_field_row(col, "Date:")
        self._notes_edit = self._make_field_row(col, "Notes:")

        self._add_btn = QPushButton("Add to catalogue")
        action_button(self._add_btn, expand=True)
        self._add_btn.clicked.connect(self._on_add_to_catalogue)
        col.addWidget(self._add_btn)

        self._discover_status_lbl = QLabel("")
        self._discover_status_lbl.setWordWrap(True)
        status_label(self._discover_status_lbl)
        col.addWidget(self._discover_status_lbl)

        # Setup-y and occasional → nested and collapsed by default.
        layout.addWidget(CollapsibleSection("Discover & add", container, expanded=False))

    def _build_catalog_section(self, layout) -> None:
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        self._table = QTableWidget(0, len(self._TABLE_COLUMNS))
        self._table.setHorizontalHeaderLabels([c.title() for c in self._TABLE_COLUMNS])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        col.addWidget(self._table, 1)

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(2)
        load_btn = QPushButton("Load CSV…")
        action_button(load_btn)
        load_btn.clicked.connect(self._on_load_csv)
        save_btn = QPushButton("Save CSV…")
        action_button(save_btn)
        save_btn.clicked.connect(self._on_save_csv)
        remove_btn = QPushButton("Remove selected")
        remove_btn.setToolTip("Drop the selected position(s) from the catalogue.")
        action_button(remove_btn)
        remove_btn.clicked.connect(self._on_remove_selected)
        clear_btn = QPushButton("Clear")
        action_button(clear_btn)
        clear_btn.clicked.connect(self._on_clear_catalog)
        for btn in (load_btn, save_btn, remove_btn, clear_btn):
            actions_row.addWidget(btn)
        col.addLayout(actions_row)

        self._catalog_status_lbl = QLabel("")
        self._catalog_status_lbl.setWordWrap(True)
        status_label(self._catalog_status_lbl)
        col.addWidget(self._catalog_status_lbl)

        # The positions table is the always-visible core of the Catalogue region.
        layout.addWidget(container)

    def _build_plugins_section(self, layout) -> None:
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(2)
        header.addWidget(QLabel("Expand a plugin to use it:"))
        header.addStretch(1)
        reload_btn = QPushButton("↻")
        reload_btn.setToolTip("Re-scan for plugins.")
        action_button(reload_btn)
        reload_btn.clicked.connect(self._reload_plugins)
        header.addWidget(reload_btn)
        col.addLayout(header)

        # One collapsible per available plugin; its header is the on/off control.
        # Pin them to the top (like the main app's stage list) so an expanded
        # plugin grows to its content instead of the sections sharing — and
        # spreading into — the leftover vertical space.
        self._plugin_host = QWidget()
        self._plugin_host_layout = QVBoxLayout(self._plugin_host)
        self._plugin_host_layout.setContentsMargins(0, 0, 0, 0)
        self._plugin_host_layout.setSpacing(0)
        self._plugin_host_layout.setAlignment(Qt.AlignTop)
        col.addWidget(self._plugin_host)

        layout.addWidget(CollapsibleSection("Plugins", container, expanded=True))

    # ----------------------------------------------------------- catalog actions
    def _set_catalog_status(self, message: str) -> None:
        self._catalog_status_lbl.setText(message)

    def _on_browse_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select root folder to scan")
        if path:
            self._root_edit.setText(path)

    def _set_discover_status(self, message: str) -> None:
        self._discover_status_lbl.setText(message)

    def _on_discover(self) -> None:
        root = self._root_edit.text().strip()
        cell = self._cell_name_edit.text().strip()
        nucleus = self._nucleus_name_edit.text().strip()
        if not root:
            self._set_discover_status("Enter a root folder to scan first.")
            return
        if not cell and not nucleus:
            self._set_discover_status("Enter at least one input file name (cell or nucleus).")
            return
        try:
            entries = discover_catalog_entries(
                root,
                cell_name=cell or None,
                contact_name=self._contact_name_edit.text().strip() or "contact_analysis.h5",
                nucleus_name=nucleus or None,
            )
        except Exception as exc:  # noqa: BLE001 - surface discovery errors in the UI
            self._set_discover_status(f"Discover error: {exc}")
            return
        self._pending_entries = entries
        self._populate_discovered()

    def _populate_discovered(self) -> None:
        self._discovered_list.clear()
        for entry in self._pending_entries:
            have_nucleus = entry.get("nucleus_tracked_labels_path") is not None
            contact = entry.get("contact_analysis_path")
            built = contact is not None and Path(contact).is_file()
            badges = ["built" if built else "missing"]
            if have_nucleus:
                badges.append("nucleus")
            self._discovered_list.addItem(f"{entry['id']}  [{', '.join(badges)}]")
        n = len(self._pending_entries)
        if not n:
            self._set_discover_status("No matching positions found under the root.")
        else:
            self._set_discover_status(
                f"Discovered {n} position(s); set condition / date / notes and add."
            )

    def _on_add_to_catalogue(self) -> None:
        """Register the discovered positions — building is a builder-plugin action."""
        if not self._pending_entries:
            self._set_discover_status("Discover positions before adding.")
            return
        condition = self._condition_edit.text().strip() or "unknown_condition"
        date = self._date_edit.text().strip() or "unknown_date"
        notes = self._notes_edit.text().strip()
        annotated = [
            {**entry, "condition": condition, "date": date, "notes": notes}
            for entry in self._pending_entries
        ]
        added = len(annotated)
        self._merge_records(annotated, source=f"added {added} from discovery")
        self._pending_entries = []
        self._discovered_list.clear()
        self._set_discover_status(
            f"Added {added} position(s). Check a builder plugin to compute quantities."
        )

    # ----------------------------------------------------------------- building
    def _run_quantity_build(
        self, quantifier: Quantifier, records: list[dict], overwrite: bool
    ) -> None:
        """Build *quantifier* for the in-scope *records* (invoked by a builder plugin)."""
        if self._build_worker is not None:
            self._set_catalog_status("A build is already running.")
            return
        jobs: list[_BuildPlan] = []
        for record in records:
            inputs = position_inputs_from_record(record)
            if not quantifier.can_build(inputs):
                continue
            out = output_for_record(quantifier, record)
            if not overwrite and quantifier.is_built(out):
                continue
            jobs.append(_BuildPlan(inputs=inputs, output=out))
        if not jobs:
            self._set_catalog_status(
                "Nothing to build — inputs missing or already built (try Recompute)."
            )
            return
        self._begin_build(quantifier, jobs)

    def _begin_build(self, quantifier: Quantifier, jobs: list[_BuildPlan]) -> None:
        self._set_catalog_status(
            f"Computing {quantifier.display_name} for {len(jobs)} position(s)…"
        )
        emit = self._build_emitter.progress.emit

        @thread_worker(
            connect={"returned": self._on_build_done, "errored": self._on_build_error}
        )
        def _worker():
            results: list[_BuildResult] = []
            total = len(jobs)
            for index, plan in enumerate(jobs, start=1):
                name = plan.inputs.position_dir.name
                emit(index, total, name)
                try:
                    quantifier.build(plan.inputs, plan.output)
                    results.append(_BuildResult(name, "built"))
                except Exception:  # noqa: BLE001 - reported per-position, not fatal
                    results.append(_BuildResult(name, "failed"))
            return results

        self._build_worker = _worker()

    def _on_build_progress(self, done: int, total: int, label: str) -> None:
        self._set_catalog_status(f"Computing: {done}/{total} {label}")

    def _on_build_done(self, results: list) -> None:
        self._build_worker = None
        built = sum(1 for r in results if r.status == "built")
        failed = sum(1 for r in results if r.status == "failed")
        # Re-normalize so each position's status reflects freshly built files.
        self._records = merge_catalog_records(self._records, [])
        self._refresh_table()
        self._set_catalog_status(
            f"Built {built}" + (f", {failed} failed" if failed else "") + "."
        )

    def _on_build_error(self, exc: Exception) -> None:
        self._build_worker = None
        self._set_catalog_status(f"Build error: {exc}")

    def _on_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load catalog CSV", filter="CSV files (*.csv)"
        )
        if path:
            self._load_csv_from(Path(path))

    def _load_csv_from(self, path: Path) -> None:
        try:
            loaded = load_catalog(path)
        except Exception as exc:  # noqa: BLE001 - surface load errors in the UI
            self._set_catalog_status(f"Load error: {exc}")
            return
        self._merge_records(loaded, source=f"loaded {len(loaded)} from CSV")

    def _on_save_csv(self) -> None:
        if not self._records:
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
        try:
            save_catalog(path, self._records)
        except Exception as exc:  # noqa: BLE001 - surface save errors in the UI
            self._set_catalog_status(f"Save error: {exc}")
            return
        self._set_catalog_status(f"Saved {len(self._records)} record(s) to {path.name}.")

    def _on_remove_selected(self) -> None:
        rows = self._selected_rows()
        if not rows:
            self._set_catalog_status("Select one or more positions to remove first.")
            return
        drop = {row for row in rows if 0 <= row < len(self._records)}
        for record in (self._records[row] for row in drop):
            path = record.get("contact_analysis_path")
            if path is not None:
                self._analysis_cache.pop(Path(path), None)
        removed = len(drop)
        self._records = [
            record for row, record in enumerate(self._records) if row not in drop
        ]
        self._refresh_table()
        self._set_catalog_status(
            f"Removed {removed} position(s); {len(self._records)} remaining."
        )

    def _on_clear_catalog(self) -> None:
        self._records = []
        self._analysis_cache.clear()
        self._refresh_table()
        self._set_catalog_status("Catalog cleared.")

    def _merge_records(self, incoming: list[dict], *, source: str) -> None:
        self._records = merge_catalog_records(self._records, incoming)
        self._refresh_table()
        self._set_catalog_status(f"Catalog: {len(self._records)} position(s) ({source}).")

    # --------------------------------------------------------------- table + scope
    def _refresh_table(self) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._records))
        for row, record in enumerate(self._records):
            values = (
                str(record.get("condition", "")),
                str(record.get("date", "")),
                str(record.get("id", "")),
                _inputs_label(record),
                str(record.get("notes", "")),
                str(record.get("contact_analysis_status", "")),
            )
            for col, value in enumerate(values):
                self._table.setItem(row, col, QTableWidgetItem(value))
        self._table.blockSignals(False)
        self._push_context()

    def _selected_rows(self) -> list[int]:
        return sorted({idx.row() for idx in self._table.selectedIndexes()})

    def _records_in_scope(self) -> list[dict]:
        """Selected rows define scope; an empty selection means the whole catalog."""
        selected_rows = self._selected_rows()
        if not selected_rows:
            return list(self._records)
        return [self._records[row] for row in selected_rows if 0 <= row < len(self._records)]

    def _on_selection_changed(self) -> None:
        # The single-position visualizer is now the Visualize Contacts plugin; it
        # reads the in-scope rows from the context like every other plugin.
        self._push_context()

    # -------------------------------------------------------------- plugin hosting
    def _reload_plugins(self) -> None:
        """Rebuild the plugin collapsibles from the available plugins.

        Each plugin gets one collapsed :class:`CollapsibleSection`; its header is
        the on/off control (no separate checkbox), and expanding it reveals the
        body. Bodies are fed the catalogue scope whether expanded or not, so the
        view is always current the moment it is opened.
        """
        while self._plugin_host_layout.count():
            item = self._plugin_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._plugin_sections = {}

        self._plugin_entries = available_studio_plugins(
            build_callback=self._run_quantity_build
        )
        if not self._plugin_entries:
            placeholder = QLabel("No plugins found.")
            status_label(placeholder, muted=True)
            self._plugin_host_layout.addWidget(placeholder)
            self._plugin_host_layout.addStretch()
            return
        for entry in self._plugin_entries:
            body = entry.factory(self.viewer)
            # A group plugin that owns a quantity's build delegates execution to
            # the studio's centralized (threaded, status-refreshed) build path.
            set_build_callback = getattr(body, "set_build_callback", None)
            if callable(set_build_callback):
                set_build_callback(self._run_quantity_build)
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
        """Feed every plugin the current scope and refresh availability."""
        for plugin in self._plugin_sections.values():
            self._push_context_to(plugin.body)
        self._update_plugin_availability()

    def _push_context_to(self, body: QWidget) -> None:
        set_context = getattr(body, "set_context", None)
        if not callable(set_context):
            return
        set_context(
            AnalysisContext(
                records=self._records_in_scope(),
                viewer=self.viewer,
                loader=self._load_analysis,
            )
        )

    def _update_plugin_availability(self) -> None:
        """Disable a plugin's header when no in-scope position has its inputs."""
        scope = self._records_in_scope()
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
