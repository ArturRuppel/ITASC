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
package, which the standalone ``cellflow-contact-analysis`` wheel does not ship); the
standalone wheel falls back to the bare
:class:`ContactAnalysisWidget` (see ``make_contact_analysis_widget``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
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
from cellflow.napari._catalog_table import CatalogRowSpec, CatalogTable
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
from cellflow.napari.ui_style import action_button, status_label
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
        #: Full catalog (normalized records from cellflow.contact_analysis.catalog).
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

        # One shared parameter bar above both areas: builds and plots read the
        # same pixel size / frame interval (plus plot-only FOV / shuffles).
        self._shared_params = SharedParamsWidget()
        self._shared_params.changed.connect(self._push_context)
        layout.addWidget(
            CollapsibleSection("Parameters", self._shared_params, expanded=True)
        )

        # Tools on top, then the Run area.
        # (Plotting is Iris-only — no in-napari Plot area.)
        self._build_tools_section(layout)
        self._build_run_section(layout)

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
        """Discover positions by file name, name each folder-nesting level, then add.

        Each position inherits its metadata from its own path: a named nesting
        level becomes a column whose value is that folder's name (folder-derived
        columns). Levels named with the recognized identity axes (``condition`` /
        ``experiment_id`` / ``position_id``) drive pooling; any other level name is
        an extra descriptor. A manual *+ Add column* unit supplies batch-wide
        constant tags not encoded in the folder tree.
        """
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        #: Discovered-but-not-added entries; each carries its folder ``levels``.
        self._pending_entries = []
        #: Manual (name, value) constant-tag fields, copied onto every added row.
        self._manual_column_fields: list[tuple[QLineEdit, QLineEdit]] = []
        #: Per-level name fields, rebuilt to the discovered depth.
        self._level_name_fields: list[QLineEdit] = []

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

        # Names or relative paths (relative to each position folder). These are the
        # discovery **inputs** — a position is any folder with at least one of them.
        # Contact analysis (like every shape / NLS quantity) is a derived Contact
        # Analysis output, built later via a builder plugin — never an input here.
        # Default to the committed final outputs (nucleus_labels.tif /
        # cell_labels.tif) so a finalized position is auto-discovered; override to
        # a working stage file (e.g. 3_cell/tracked_labels.tif) when uncommitted.
        self._cell_name_edit = self._make_field_row(
            col, "Cell labels (optional):",
            placeholder="e.g. cell_labels.tif (final) or 3_cell/tracked_labels.tif",
        )
        self._cell_name_edit.setText("cell_labels.tif")
        self._nucleus_name_edit = self._make_field_row(
            col, "Nucleus labels (optional):",
            placeholder="e.g. nucleus_labels.tif (final) or 2_nucleus/tracked_labels.tif",
        )
        self._nucleus_name_edit.setText("nucleus_labels.tif")
        for edit in (self._cell_name_edit, self._nucleus_name_edit):
            edit.textChanged.connect(lambda _t: self._update_discover_tooltip())

        self._discover_btn = QPushButton("Discover")
        action_button(self._discover_btn, expand=True)
        self._discover_btn.clicked.connect(self._on_discover)
        col.addWidget(self._discover_btn)
        self._update_discover_tooltip()

        self._discovered_list = QListWidget()
        self._discovered_list.setMaximumHeight(120)
        col.addWidget(self._discovered_list)

        # Levels editor: one name field per detected nesting level, anchored at the
        # root. Editable before committing; renaming re-derives that column on the
        # staged rows. Rebuilt to the discovered depth in ``_rebuild_level_fields``.
        col.addWidget(QLabel("Folder levels (root → position):"))
        self._levels_box = QVBoxLayout()
        self._levels_box.setContentsMargins(12, 0, 0, 0)
        self._levels_box.setSpacing(2)
        col.addLayout(self._levels_box)
        self._levels_hint = QLabel("Discover first to name the folder levels.")
        status_label(self._levels_hint, muted=True)
        col.addWidget(self._levels_hint)

        # Manual constant columns (batch-wide tags not in the folder tree).
        self._manual_columns_box = QVBoxLayout()
        self._manual_columns_box.setContentsMargins(12, 0, 0, 0)
        self._manual_columns_box.setSpacing(2)
        col.addLayout(self._manual_columns_box)
        add_col_btn = QPushButton("+ Add column")
        add_col_btn.setToolTip(
            "Tag every added position with a constant column not in the folder tree, "
            "e.g. operator or treatment."
        )
        action_button(add_col_btn)
        add_col_btn.clicked.connect(lambda: self._add_manual_column())
        col.addWidget(add_col_btn)

        self._add_btn = QPushButton("Add to catalogue")
        action_button(self._add_btn, expand=True)
        self._add_btn.setEnabled(False)
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

        # Custom-widget rows (not a QTableWidget) so each position carries a live
        # status rail. The table is internally bounded + scrolls; the whole studio
        # already lives in a scroll area.
        self._table = CatalogTable()
        self._table.selectionChanged.connect(self._on_selection_changed)
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
        reload_btn = QPushButton("↻")
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

        layout.addWidget(CollapsibleSection("Tools", container, expanded=False))

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
        layout.addWidget(CollapsibleSection("Run", container, expanded=True))

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
        project_dir = Path(chosen) if chosen else catalogue_root(self._records)
        params = self._shared_params.build_params()
        return author_config(
            project_dir,
            self._records,
            tables_dir="." if chosen else None,
            quantities=choices.quantities,
            params=params,
        )

    def _on_run_save(self, choices) -> None:
        if not self._records:
            self._run_area.set_status("Add positions to the catalogue first.")
            return
        try:
            config_path = self._author_run_config(choices)
        except Exception as exc:  # noqa: BLE001 - surface authoring errors in the UI
            self._run_area.set_status(f"Save error: {exc}")
            return
        self._run_area.set_status(f"Wrote {config_path.name} + catalog.csv.")

    def _on_run_execute(self, choices) -> None:
        if self._run_worker is not None:
            self._run_area.set_status("A run is already in progress.")
            return
        if not self._records:
            self._run_area.set_status("Add positions to the catalogue first.")
            return
        try:
            config_path = self._author_run_config(choices)
        except Exception as exc:  # noqa: BLE001 - surface authoring errors in the UI
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

    def _on_browse_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select root folder to scan")
        if path:
            self._root_edit.setText(path)

    def _set_discover_status(self, message: str) -> None:
        self._discover_status_lbl.setText(message)

    def _discover_tooltip_text(self) -> str:
        """Plain-language scan description naming the filled input-file fields."""
        names = [
            edit.text().strip()
            for edit in (self._cell_name_edit, self._nucleus_name_edit)
            if edit.text().strip()
        ]
        if not names:
            return "Enter at least one input file name, then scan a root folder."
        joined = names[0] if len(names) == 1 else f"{' and '.join(names)}"
        return (
            "Find every folder under the root that contains "
            f"{joined}, and stage each as a position."
        )

    def _update_discover_tooltip(self) -> None:
        self._discover_btn.setToolTip(self._discover_tooltip_text())

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
                nucleus_name=nucleus or None,
            )
        except Exception as exc:  # noqa: BLE001 - surface discovery errors in the UI
            self._set_discover_status(f"Discover error: {exc}")
            return

        # Each entry carries its nesting levels (root → position folder) so a
        # named level maps to that folder's name per position.
        for entry in entries:
            entry["levels"] = list(relative_levels(root, entry["position_path"]))
        depth = discovered_level_depth(root, [e["position_path"] for e in entries])
        if entries and depth is None:
            self._pending_entries = []
            self._discovered_list.clear()
            self._rebuild_level_fields(0)
            self._add_btn.setEnabled(False)
            self._set_discover_status(
                "Discovered positions sit at differing folder depths; the levels "
                "cannot line up. Scan a root where every position is equally nested."
            )
            return

        self._pending_entries = entries
        self._rebuild_level_fields(depth or 0)
        self._populate_discovered()

    def _rebuild_level_fields(self, depth: int) -> None:
        """(Re)build one name field per nesting level, seeded by recognized axes."""
        while self._levels_box.count():
            item = self._levels_box.takeAt(0)
            sub = item.layout()
            if sub is not None:
                while sub.count():
                    w = sub.takeAt(0).widget()
                    if w is not None:
                        w.deleteLater()
            elif item.widget() is not None:
                item.widget().deleteLater()
        self._level_name_fields = []
        if depth <= 0:
            self._levels_hint.setVisible(True)
            return
        self._levels_hint.setVisible(False)
        for level, seed in enumerate(_seed_level_names(depth), start=1):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(2)
            lbl = QLabel(f"Level {level}:")
            lbl.setFixedWidth(60)
            edit = QLineEdit(seed)
            edit.textChanged.connect(lambda _t: self._reapply_levels())
            row.addWidget(lbl)
            row.addWidget(edit, 1)
            self._levels_box.addLayout(row)
            self._level_name_fields.append(edit)

    def _level_names(self) -> list[str]:
        return [edit.text().strip() for edit in self._level_name_fields]

    def _add_manual_column(self, name: str = "", value: str = "") -> None:
        """Append a manual (name, value) constant-column field to the batch."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        name_edit = QLineEdit(name)
        name_edit.setPlaceholderText("column")
        value_edit = QLineEdit(value)
        value_edit.setPlaceholderText("value")
        for edit in (name_edit, value_edit):
            edit.textChanged.connect(lambda _t: self._reapply_levels())
        row.addWidget(name_edit, 1)
        row.addWidget(value_edit, 1)
        self._manual_columns_box.addLayout(row)
        self._manual_column_fields.append((name_edit, value_edit))

    def _manual_columns(self) -> dict[str, str]:
        columns: dict[str, str] = {}
        for name_edit, value_edit in self._manual_column_fields:
            name = name_edit.text().strip()
            if name:
                columns[name] = value_edit.text().strip()
        return columns

    def _columns_for_entry(self, entry: dict) -> dict[str, str]:
        """Folder-derived columns for one entry, plus the manual constant tags.

        ``position_id`` always resolves to the innermost folder name so positions
        keep distinct identities even if the user renames that level away."""
        segments = tuple(entry.get("levels", ()))
        columns = columns_from_levels(self._level_names(), segments)
        if segments:
            columns.setdefault("position_id", segments[-1])
        columns.update(self._manual_columns())
        return columns

    def _reapply_levels(self) -> None:
        """Re-derive the staged preview as level names / manual tags change."""
        if self._pending_entries:
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
            columns = self._columns_for_entry(entry)
            caption = "  ·  ".join(
                f"{k}={v}" for k, v in columns.items() if k != "position_id" and v
            )
            pid = columns.get("position_id", entry["id"])
            label = f"{pid}  [{', '.join(badges)}]"
            if caption:
                label = f"{label}    {caption}"
            self._discovered_list.addItem(label)
        n = len(self._pending_entries)
        self._add_btn.setEnabled(n > 0)
        if not n:
            self._set_discover_status("No matching positions found under the root.")
        else:
            self._set_discover_status(
                f"{n} position(s) discovered — name the levels and Add to catalogue."
            )

    def _on_add_to_catalogue(self) -> None:
        """Register the discovered positions — building is a builder-plugin action."""
        if not self._pending_entries:
            self._set_discover_status("Discover positions before adding.")
            return
        annotated = [
            {**entry, "columns": self._columns_for_entry(entry)}
            for entry in self._pending_entries
        ]
        added = len(annotated)
        self._merge_records(annotated, source=f"added {added} from discovery")
        self._pending_entries = []
        self._discovered_list.clear()
        self._add_btn.setEnabled(False)
        self._set_discover_status(
            f"Added {added} position(s). Check a builder plugin to compute quantities."
        )

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
    def _group_caption(self, record: dict) -> str:
        """Bold-separator caption for a record's batch: its columns bar position_id."""
        columns = record.get("columns") or {}
        return "  ·  ".join(
            f"{k}: {v}" for k, v in columns.items() if k != "position_id" and v
        )

    def _refresh_table(self) -> None:
        """Rebuild the catalog rows: group separators + one status-rail row each.

        Each position's rail is derived from its canonical position directory
        (``position_path``); a row with none (a hand-authored CSV row with no
        common root) gets an all-``unknown`` rail.
        """
        specs: list[CatalogRowSpec] = []
        last_caption: str | None = None
        for ridx, record in enumerate(self._records):
            caption = self._group_caption(record)
            if caption and caption != last_caption:
                specs.append(CatalogRowSpec(record_index=None, caption=caption))
            last_caption = caption
            specs.append(
                CatalogRowSpec(
                    record_index=ridx,
                    values=(
                        str(record.get("condition", "")),
                        str(record.get("date", "")),
                        str(record.get("id", "")),
                        _inputs_label(record),
                        str(record.get("notes", "")),
                    ),
                    status=position_stage_status(record.get("position_path")),
                )
            )
        self._table.set_rows(specs)
        #: Display-row → record-index map (``None`` for separators); mirrors the
        #: table's own mapping so tests and callers keep a single source.
        self._row_to_record = self._table.row_to_record
        self._push_context()

    def _selected_rows(self) -> list[int]:
        """Selected *record* indices — separator rows never select."""
        return self._table.selected_record_indices()

    def _records_in_scope(self) -> list[dict]:
        """Selected rows define scope; an empty selection means the whole catalog."""
        selected_rows = self._selected_rows()
        if not selected_rows:
            return list(self._records)
        return [self._records[row] for row in selected_rows if 0 <= row < len(self._records)]

    def _scoped_records(self) -> list[dict]:
        """In-scope records stamped with the shared build params (pixel size /
        frame interval), so building and gating both see a manual override."""
        records = self._records_in_scope()
        shared = getattr(self, "_shared_params", None)
        return shared.stamp(records) if shared is not None else records

    def _on_selection_changed(self) -> None:
        # The single-position visualizer is now the Visualize Contacts plugin; it
        # reads the in-scope rows from the context like every other plugin.
        self._push_context()

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
