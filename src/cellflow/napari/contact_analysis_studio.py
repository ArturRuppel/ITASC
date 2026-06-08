"""Contact Analysis studio: a position catalog + per-position view + analysis plugins.

This is the merged standalone Contact Analysis tool. It is built from three parts:

* a **catalog** of positions (autodiscover a study tree, add loose ``.h5`` files,
  load/save a CSV catalog);
* a **per-position contact view** — an embedded :class:`ContactAnalysisWidget`
  driven by the single selected position (visualize + compute-if-missing);
* an **analysis** section that hosts a meta-analysis *plugin* fed with the
  currently-selected catalog rows. All cross-position aggregation lives in
  plugins (see :mod:`cellflow.napari.meta_plugins`).

This module is full-install only (it depends on :mod:`cellflow.meta`); the
standalone ``cellflow-contact`` wheel falls back to the bare
:class:`ContactAnalysisWidget` (see ``make_contact_analysis_widget``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from cellflow.meta.catalog import (
    discover_catalog_entries,
    load_meta_catalog,
    merge_catalog_records,
    save_meta_catalog,
)
from cellflow.napari.contact_analysis_widget import ContactAnalysisWidget
from cellflow.napari.meta_plugins import (
    MetaAnalysisPlugin,
    MetaContext,
    available_meta_plugins,
)
from cellflow.napari.ui_style import action_button, status_label
from cellflow.napari.widgets import CollapsibleSection

try:  # pragma: no cover - standalone-packaging boundary
    from cellflow.contact_analysis.reader import read_position_contact_analysis
except ImportError:  # pragma: no cover - tests monkeypatch this when absent
    def read_position_contact_analysis(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("cellflow.contact_analysis.reader is unavailable")


class ContactAnalysisStudioWidget(QWidget):
    """Position catalog + embedded per-position contact view + analysis plugins."""

    _TABLE_COLUMNS = ("condition", "date", "id", "notes", "status")

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        #: Full catalog (normalized records from cellflow.meta.catalog).
        self._records: list[dict] = []
        #: Last discovered (not-yet-added) collection awaiting metadata.
        self._pending_entries: list[dict] = []
        #: Cache so plugins don't each re-open the same HDF5.
        self._analysis_cache: dict[Path, Any] = {}
        self._active_plugin: QWidget | None = None
        self._plugin_classes: list[type[MetaAnalysisPlugin]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self._build_discover_section(layout)
        self._build_catalog_section(layout)
        self._build_contact_view_section(layout)
        self._build_analysis_section(layout)

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

        # Names or relative paths (relative to each position folder).
        self._contact_name_edit = self._make_field_row(
            col, "Contact analysis:", default="contact_analysis.h5"
        )
        self._cell_name_edit = self._make_field_row(
            col, "Cell labels (optional):", placeholder="e.g. 3_cell/tracked_labels.tif"
        )
        self._nucleus_name_edit = self._make_field_row(
            col, "Nucleus labels (optional):", placeholder="e.g. 2_nucleus/tracked_labels.tif"
        )

        discover_btn = QPushButton("Discover")
        discover_btn.setToolTip(
            "Find every folder under the root that contains the contact-analysis "
            "file; associate co-located cell / nucleus labels."
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

        layout.addWidget(CollapsibleSection("Discover & add", container, expanded=True))

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
        clear_btn = QPushButton("Clear")
        action_button(clear_btn)
        clear_btn.clicked.connect(self._on_clear_catalog)
        for btn in (load_btn, save_btn, clear_btn):
            actions_row.addWidget(btn)
        col.addLayout(actions_row)

        self._catalog_status_lbl = QLabel("")
        self._catalog_status_lbl.setWordWrap(True)
        status_label(self._catalog_status_lbl)
        col.addWidget(self._catalog_status_lbl)

        layout.addWidget(CollapsibleSection("Catalogue", container, expanded=True))

    def _build_contact_view_section(self, layout) -> None:
        """Embed the per-position contact visualizer, driven by single selection."""
        self._contact_widget = ContactAnalysisWidget(viewer=self.viewer, standalone=False)
        # The embedded widget's own "Pipeline Files" panel is an orchestrator
        # concept; here the catalog table is the position source instead.
        self._contact_widget.pipeline_files_header.setVisible(False)
        self._contact_widget._pipeline_files_section.setVisible(False)
        layout.addWidget(
            CollapsibleSection("Contact view", self._contact_widget, expanded=True)
        )

    def _build_analysis_section(self, layout) -> None:
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        selector_row = QHBoxLayout()
        selector_row.setContentsMargins(0, 0, 0, 0)
        selector_row.setSpacing(2)
        selector_row.addWidget(QLabel("Analysis:"))
        self._plugin_combo = QComboBox()
        self._plugin_combo.currentIndexChanged.connect(self._on_plugin_changed)
        selector_row.addWidget(self._plugin_combo, 1)
        reload_btn = QPushButton("↻")
        reload_btn.setToolTip("Re-scan for meta-analysis plugins.")
        action_button(reload_btn)
        reload_btn.clicked.connect(self._reload_plugins)
        selector_row.addWidget(reload_btn)
        col.addLayout(selector_row)

        self._plugin_host = QWidget()
        self._plugin_host_layout = QVBoxLayout(self._plugin_host)
        self._plugin_host_layout.setContentsMargins(0, 0, 0, 0)
        self._plugin_host_layout.setSpacing(0)
        col.addWidget(self._plugin_host, 1)

        layout.addWidget(CollapsibleSection("Analysis", container, expanded=True))

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
        contact = self._contact_name_edit.text().strip()
        if not root:
            self._set_discover_status("Enter a root folder to scan first.")
            return
        if not contact:
            self._set_discover_status("Enter the contact-analysis file name first.")
            return
        try:
            entries = discover_catalog_entries(
                root,
                contact_name=contact,
                cell_name=self._cell_name_edit.text().strip() or None,
                nucleus_name=self._nucleus_name_edit.text().strip() or None,
            )
        except Exception as exc:  # noqa: BLE001 - surface discovery errors in the UI
            self._set_discover_status(f"Discover error: {exc}")
            return
        self._pending_entries = entries
        self._populate_discovered()

    def _populate_discovered(self) -> None:
        self._discovered_list.clear()
        for entry in self._pending_entries:
            have_cell = entry.get("cell_tracked_labels_path") is not None
            have_nucleus = entry.get("nucleus_tracked_labels_path") is not None
            extras = []
            if have_cell:
                extras.append("cell")
            if have_nucleus:
                extras.append("nucleus")
            suffix = f"  (+{', '.join(extras)})" if extras else ""
            self._discovered_list.addItem(f"{entry['id']}{suffix}")
        n = len(self._pending_entries)
        self._set_discover_status(
            f"Discovered {n} position(s); set condition / date / notes and add."
            if n
            else "No matching positions found under the root."
        )

    def _on_add_to_catalogue(self) -> None:
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
        self._set_discover_status(f"Added {added} position(s) to the catalogue.")

    def _on_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load catalog CSV", filter="CSV files (*.csv)"
        )
        if path:
            self._load_csv_from(Path(path))

    def _load_csv_from(self, path: Path) -> None:
        try:
            loaded = load_meta_catalog(path)
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
            self._save_csv_to(Path(path))

    def _save_csv_to(self, path: Path) -> None:
        try:
            save_meta_catalog(path, self._records)
        except Exception as exc:  # noqa: BLE001 - surface save errors in the UI
            self._set_catalog_status(f"Save error: {exc}")
            return
        self._set_catalog_status(f"Saved {len(self._records)} record(s) to {path.name}.")

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

    def _focused_record(self) -> dict | None:
        """The single position the contact view targets (the first selected row)."""
        rows = self._selected_rows()
        if len(rows) != 1:
            return None
        row = rows[0]
        return self._records[row] if 0 <= row < len(self._records) else None

    def _on_selection_changed(self) -> None:
        self._update_contact_view()
        self._push_context()

    def _update_contact_view(self) -> None:
        """Point the embedded contact view at the single selected position.

        The catalog rows from a study scan carry the cell/nucleus label paths;
        rows from a loose ``.h5`` add or a reloaded CSV carry only the ``.h5``,
        so those support showing an existing result but not compute-on-demand.
        """
        record = self._focused_record()
        if record is None:
            self._contact_widget.set_context(
                cell_labels=None, nucleus_labels=None, out_path=None
            )
            return
        self._contact_widget.set_context(
            cell_labels=record.get("cell_tracked_labels_path"),
            nucleus_labels=record.get("nucleus_tracked_labels_path"),
            out_path=record.get("contact_analysis_path"),
            status_root=record.get("position_path"),
        )

    # -------------------------------------------------------------- plugin hosting
    def _reload_plugins(self) -> None:
        self._plugin_classes = available_meta_plugins()
        self._plugin_combo.blockSignals(True)
        self._plugin_combo.clear()
        for plugin_cls in self._plugin_classes:
            self._plugin_combo.addItem(plugin_cls.display_name)
        self._plugin_combo.blockSignals(False)
        if self._plugin_classes:
            self._plugin_combo.setCurrentIndex(0)
            self._mount_plugin(0)
        else:
            self._mount_plugin(None)

    def _on_plugin_changed(self, index: int) -> None:
        self._mount_plugin(index if index >= 0 else None)

    def _mount_plugin(self, index: int | None) -> None:
        # Tear down the previously-mounted plugin.
        if self._active_plugin is not None:
            self._plugin_host_layout.removeWidget(self._active_plugin)
            self._active_plugin.deleteLater()
            self._active_plugin = None

        if index is None or not (0 <= index < len(self._plugin_classes)):
            placeholder = QLabel("No meta-analysis plugins found.")
            status_label(placeholder, muted=True)
            self._active_plugin = placeholder
            self._plugin_host_layout.addWidget(placeholder)
            return

        plugin_cls = self._plugin_classes[index]
        widget = plugin_cls(self.viewer)
        self._active_plugin = widget
        self._plugin_host_layout.addWidget(widget)
        self._push_context()

    def _load_analysis(self, path: Path) -> Any:
        path = Path(path)
        cached = self._analysis_cache.get(path)
        if cached is None:
            cached = read_position_contact_analysis(path)
            self._analysis_cache[path] = cached
        return cached

    def _push_context(self) -> None:
        """Feed the active plugin the current scope, if it accepts a context."""
        plugin = self._active_plugin
        set_context = getattr(plugin, "set_context", None)
        if not callable(set_context):
            return
        ctx = MetaContext(
            records=self._records_in_scope(),
            viewer=self.viewer,
            loader=self._load_analysis,
        )
        set_context(ctx)
