"""Aggregate Quantification studio: a position catalog + per-position view + analysis plugins.

This is the merged standalone Aggregate Quantification tool. It is built from three parts:

* a **catalog** of positions (autodiscover a study tree, add loose ``.h5`` files,
  load/save a CSV catalog);
* a **per-position quantity view** — an embedded :class:`AggregateQuantificationWidget`
  driven by the single selected position (visualize + compute-if-missing);
* an **analysis** section that hosts a meta-analysis *plugin* fed with the
  currently-selected catalog rows. All cross-position aggregation lives in
  plugins (see :mod:`cellflow.napari.meta_plugins`).

This module is full-install only (it depends on :mod:`cellflow.meta`); the
standalone ``cellflow-aggregate`` wheel falls back to the bare
:class:`AggregateQuantificationWidget` (see ``make_aggregate_quantification_widget``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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

from cellflow.aggregate_quantification import ContactBatchJob, run_contact_batch
from cellflow.meta.catalog import (
    discover_catalog_entries,
    load_meta_catalog,
    merge_catalog_records,
    save_meta_catalog,
)
from cellflow.napari.aggregate_quantification_widget import AggregateQuantificationWidget, _ProgressEmitter
from cellflow.napari.meta_plugins import (
    MetaAnalysisPlugin,
    MetaContext,
    available_meta_plugins,
)
from cellflow.napari.ui_style import action_button, status_label
from cellflow.napari.widgets import CollapsibleSection

try:  # pragma: no cover - standalone-packaging boundary
    from cellflow.aggregate_quantification.contacts.reader import read_position_contact_analysis
except ImportError:  # pragma: no cover - tests monkeypatch this when absent
    def read_position_contact_analysis(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("cellflow.aggregate_quantification.contacts.reader is unavailable")


class AggregateQuantificationStudioWidget(QWidget):
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
        #: Background build of missing contact analyses triggered by "Add".
        self._build_worker = None
        #: Annotated records held while their contact analyses are computed.
        self._pending_annotated: list[dict] | None = None
        self._build_emitter = _ProgressEmitter(self)
        self._build_emitter.progress.connect(self._on_build_progress)

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

        # Names or relative paths (relative to each position folder). Cell labels
        # are the anchor; the contact analysis is the derived output computed from
        # them (built on add when missing, or always with the checkbox below).
        self._cell_name_edit = self._make_field_row(
            col, "Cell labels:", placeholder="e.g. 3_cell/tracked_labels.tif"
        )
        self._nucleus_name_edit = self._make_field_row(
            col, "Nucleus labels (optional):", placeholder="e.g. 2_nucleus/tracked_labels.tif"
        )
        self._contact_name_edit = self._make_field_row(
            col, "Contact analysis (output):", default="contact_analysis.h5"
        )
        self._recompute_cb = QCheckBox("Always recompute contact analysis")
        self._recompute_cb.setToolTip(
            "Recompute the contact analysis from the cell labels for every "
            "position, overwriting any existing result. When unchecked, only "
            "missing analyses are computed."
        )
        col.addWidget(self._recompute_cb)

        discover_btn = QPushButton("Discover")
        discover_btn.setToolTip(
            "Find every folder under the root that contains the cell-labels file; "
            "derive the contact-analysis output path and associate co-located "
            "nucleus labels."
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

        layout.addWidget(CollapsibleSection("Catalogue", container, expanded=True))

    def _build_contact_view_section(self, layout) -> None:
        """Embed the per-position contact visualizer, driven by single selection."""
        self._contact_widget = AggregateQuantificationWidget(viewer=self.viewer, standalone=False)
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
        cell = self._cell_name_edit.text().strip()
        if not root:
            self._set_discover_status("Enter a root folder to scan first.")
            return
        if not cell:
            self._set_discover_status("Enter the cell-labels file name first.")
            return
        try:
            entries = discover_catalog_entries(
                root,
                cell_name=cell,
                contact_name=self._contact_name_edit.text().strip() or "contact_analysis.h5",
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
            have_nucleus = entry.get("nucleus_tracked_labels_path") is not None
            contact = entry.get("contact_analysis_path")
            built = contact is not None and Path(contact).is_file()
            badges = ["built" if built else "missing"]
            if have_nucleus:
                badges.append("nucleus")
            self._discovered_list.addItem(f"{entry['id']}  [{', '.join(badges)}]")
        n = len(self._pending_entries)
        missing = sum(
            1
            for e in self._pending_entries
            if not (e.get("contact_analysis_path") and Path(e["contact_analysis_path"]).is_file())
        )
        if not n:
            self._set_discover_status("No matching positions found under the root.")
        else:
            note = (
                f" ({missing} need the contact analysis computed on add)"
                if missing
                else ""
            )
            self._set_discover_status(
                f"Discovered {n} position(s); set condition / date / notes and add.{note}"
            )

    def _on_add_to_catalogue(self) -> None:
        if self._build_worker is not None:
            return
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

        # The contact analysis is derived from the cell labels: build it for any
        # position whose .h5 is missing, or for every position when "always
        # recompute" is checked. Positions without cell labels can't be built and
        # are added as-is (showing an existing result only).
        overwrite = self._recompute_cb.isChecked()
        jobs = [job for entry in annotated if (job := self._build_job(entry, overwrite))]
        if jobs:
            self._begin_build(annotated, jobs, overwrite=overwrite)
        else:
            self._finish_add(annotated)

    def _build_job(self, entry: dict, overwrite: bool) -> ContactBatchJob | None:
        """A build job for *entry* when its contact analysis must be computed."""
        cell = entry.get("cell_tracked_labels_path")
        out = entry.get("contact_analysis_path")
        if cell is None or out is None:
            return None
        if not overwrite and Path(out).is_file():
            return None
        nucleus = entry.get("nucleus_tracked_labels_path")
        return ContactBatchJob(
            group_dir=Path(entry.get("position_path") or Path(out).parent),
            cell_labels=Path(cell),
            output=Path(out),
            nucleus_labels=Path(nucleus) if nucleus else None,
        )

    def _begin_build(self, annotated: list[dict], jobs: list, *, overwrite: bool) -> None:
        self._pending_annotated = annotated
        self._add_btn.setEnabled(False)
        self._set_discover_status(
            f"Computing contact analysis for {len(jobs)} position(s)…"
        )

        @thread_worker(
            connect={"returned": self._on_build_done, "errored": self._on_build_error}
        )
        def _worker():
            return run_contact_batch(
                jobs, overwrite=overwrite, progress_cb=self._build_emitter.progress.emit
            )

        self._build_worker = _worker()

    def _on_build_progress(self, done: int, total: int, label: str) -> None:
        self._set_discover_status(f"Computing contact analysis: {done}/{total} {label}")

    def _on_build_done(self, results: list) -> None:
        self._build_worker = None
        self._add_btn.setEnabled(True)
        annotated = self._pending_annotated or []
        self._pending_annotated = None
        built = sum(1 for r in results if r.status == "built")
        failed = sum(1 for r in results if r.status == "failed")
        extra = f" (computed {built}" + (f", {failed} failed" if failed else "") + ")"
        self._finish_add(annotated, extra=extra)

    def _on_build_error(self, exc: Exception) -> None:
        self._build_worker = None
        self._add_btn.setEnabled(True)
        self._pending_annotated = None
        self._set_discover_status(f"Build error: {exc}")

    def _finish_add(self, annotated: list[dict], *, extra: str = "") -> None:
        # _merge_records re-normalizes, so contact_analysis_status reflects any
        # files just built.
        added = len(annotated)
        self._merge_records(annotated, source=f"added {added} from discovery")
        self._pending_entries = []
        self._discovered_list.clear()
        self._set_discover_status(f"Added {added} position(s) to the catalogue.{extra}")

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
            # getSaveFileName does not always append the filter's suffix (notably
            # on Linux/Qt), so ensure the file actually ends up as a .csv.
            if not path.lower().endswith(".csv"):
                path = f"{path}.csv"
            self._save_csv_to(Path(path))

    def _save_csv_to(self, path: Path) -> None:
        try:
            save_meta_catalog(path, self._records)
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
