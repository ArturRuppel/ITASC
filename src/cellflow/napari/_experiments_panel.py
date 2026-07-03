"""The shared experiments/positions panel (napariTFM ExperimentsList parity).

One panel, two homes: the main CellFlow app and the standalone Contact Analysis
distro both mount this. It reproduces napariTFM's initialization ritual —

    Setup (calibration + input-file names + optional output dir)
      → Discover a root  → stage matches as dimmed preview rows
      → Add to list      → committed rows with an editable folder-nesting header,
                            an accent select-bar, a per-position status rail, and
                            an overall-status chip
      → Run selected / Workers, and a running count

— but speaks CellFlow: the rail is :class:`~cellflow.napari._status_rail.StatusRail`
(the five-state commit-contract vocabulary), the styling is CellFlow's own designed
surface tokens, and *what* Discover scans for / *how* a row maps to a host record is
injected by the host (``discover_fn`` / ``status_fn``).

The panel owns the displayed catalog (the ordered rows + their editable columns);
the host reads :meth:`payloads` for scope / run / save and reacts to the signals.
Each row carries an opaque ``payload`` (the host's record dict) threaded through
untouched, plus a ``columns`` dict keyed by the shared, editable column names.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterable

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._stage_status import (
    DONE,
    STAGES,
    STALE,
    UNKNOWN,
    WORKING,
)
from cellflow.napari._status_rail import StatusRail
from cellflow.napari.ui_style import (
    COMPACT_SPACING,
    TEXT_DIM,
    TEXT_MID,
    experiment_name_color,
    experiment_row_style,
    experiment_status_color,
    mono_input_style,
    stage_accent,
)
from cellflow.napari.widgets import CollapsibleSection

# Free-text calibration fields (soft validator, not spinbox stepping).
_CALIBRATION_SPECS = (
    ("pixel_size_um", "Pixel Size (µm)", 0.0001, 1000.0),
    ("time_interval_s", "Frame Length (s)", 0.0, 1e9),
)
_INPUT_DECIMALS = 6

# Fixed widths of the non-column cells, shared by the header and every data row so
# the editable column headers line up over their value cells.
_SELBAR_W = 3
_RAIL_W = 60
_CHIP_W = 56

#: Overall-status word → chip text (kept short so the chip column stays narrow).
_CHIP_TEXT = {"run": "run", "done": "done", "queued": "queued"}


def overall_status(status: dict[str, str]) -> str:
    """Collapse a per-stage status map into a single chip word.

    ``done`` when every *known* stage is committed/present; ``run`` when the
    position has some real progress (a committed, working, or stale stage) but is
    not finished; ``queued`` otherwise (nothing on disk, or an all-``unknown`` row
    with no canonical root).
    """
    real = [v for v in status.values() if v != UNKNOWN]
    if real and all(v == DONE for v in real):
        return "done"
    if any(v in (DONE, WORKING, STALE) for v in real):
        return "run"
    return "queued"


def _format_value(value) -> str:
    """Compact text for a calibration value — no trailing-zero noise."""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value or "")


class ExperimentRow(QWidget):
    """One position: accent select-bar, per-column value cells, rail, status chip.

    A *committed* row shows its rail + chip and selects/activates on click. A
    *preview* row (a discovered, not-yet-added position) is dimmed + italic, hides
    the rail/chip, and toggles a delete-selection on click.
    """

    clicked = Signal(str, int)  # key, modifier flag: 0 plain, 1 ctrl, 2 shift
    dot_clicked = Signal(str, str)  # key, stage — one rail dot's on-demand load

    def __init__(
        self,
        key: str,
        values: list[str] | None = None,
        *,
        preview: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._key = key
        self._preview = preview
        self._selected = False
        # The row paints its own (styled) background — selected rows lift.
        self.setObjectName("experiment_row")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(7, 5, 10, 5)
        layout.setSpacing(COMPACT_SPACING + 4)

        self._selbar = QFrame()
        self._selbar.setFixedWidth(_SELBAR_W)
        self._selbar.setStyleSheet("background: transparent;")
        layout.addWidget(self._selbar)

        cells = list(values) if values else [key]
        self._value_labels: list[QLabel] = []
        for text in cells:
            label = QLabel(str(text))
            layout.addWidget(label, 1)
            self._value_labels.append(label)

        self.rail = StatusRail()
        self.rail.setFixedWidth(_RAIL_W)
        self.rail.dotClicked.connect(
            lambda stage: self.dot_clicked.emit(self._key, stage)
        )
        layout.addWidget(self.rail)

        self._chip = QLabel("queued")
        self._chip.setFixedWidth(_CHIP_W)
        self._chip.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._chip.setStyleSheet(f"color: {experiment_status_color('queued')};")
        layout.addWidget(self._chip)

        if self._preview:
            # A not-yet-committed folder has no status to show.
            self.rail.setVisible(False)
            self._chip.setVisible(False)
            for value_label in self._value_labels:
                value_label.setStyleSheet(f"color: {TEXT_DIM}; font-style: italic;")

        self.set_selected(False)

    @property
    def key(self) -> str:
        return self._key

    @property
    def is_preview(self) -> bool:
        return self._preview

    def is_selected(self) -> bool:
        return self._selected

    def set_selected(self, on: bool) -> None:
        self._selected = on
        accent = stage_accent("nucleus")
        self._selbar.setStyleSheet(
            f"background: {accent};" if on else "background: transparent;"
        )
        self.setStyleSheet(experiment_row_style(on, accent))
        if not self._preview:
            color = experiment_name_color(on)
            for label in self._value_labels:
                label.setStyleSheet(f"color: {color};")

    def set_status(self, status: dict[str, str]) -> None:
        """Repaint the rail + derive the chip word from the full status map."""
        self.rail.set_status(status)
        word = overall_status(status)
        self._chip.setText(_CHIP_TEXT[word])
        self._chip.setStyleSheet(f"color: {experiment_status_color(word)};")

    def mousePressEvent(self, event) -> None:
        mods = event.modifiers()
        if mods & Qt.ControlModifier:
            flag = 1
        elif mods & Qt.ShiftModifier:
            flag = 2
        else:
            flag = 0
        self.clicked.emit(self._key, flag)
        super().mousePressEvent(event)


class ExperimentsPanel(QWidget):
    """Setup + Discover→Commit + an editable-column list of position rows.

    Host contract:

    * ``discover_fn(root, input_names) -> list[entry]`` — scan a root; each *entry*
      is ``{"key": str, "columns": {name: value}, "payload": dict}`` (``key`` is the
      row identity, typically the position folder path).
    * ``status_fn(payload) -> {stage: state}`` — the per-stage rail status.

    Signals let the host stay the model of record for persistence/scope while the
    panel owns the list UI, Setup, discovery staging, and selection.
    """

    active_changed = Signal(object)      # payload | None — the single active row
    selection_changed = Signal()         # multi-selection changed (read selected_payloads)
    records_changed = Signal()           # committed row set changed
    run_requested = Signal(list, int)    # (selected payloads, num_workers)
    stage_load_requested = Signal(object, str)  # (payload, stage)
    calibration_changed = Signal(str, str)      # (name, text)
    discover_requested = Signal()               # Discover button clicked
    output_dir_requested = Signal()             # output-dir button clicked

    def __init__(
        self,
        *,
        title: str = "Positions",
        input_fields: Iterable[tuple[str, str, str]] = (),
        discover_fn: Callable[[str, dict[str, str]], list[dict]] | None = None,
        status_fn: Callable[[dict], dict[str, str]] | None = None,
        show_calibration: bool = True,
        show_output_dir: bool = False,
        show_manual_columns: bool = False,
        show_run: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self._input_fields = list(input_fields)
        self._discover_fn = discover_fn
        self._status_fn = status_fn
        self._show_calibration = show_calibration
        self._show_output_dir = show_output_dir
        self._show_manual_columns = show_manual_columns
        self._show_run = show_run
        #: Batch-wide constant-tag (name, value) fields, copied onto every row a
        #: Discover→Commit adds — for descriptors not encoded in the folder tree.
        self._manual_fields: list[tuple[QLineEdit, QLineEdit]] = []

        # Committed rows: ordered keys + per-key {columns, payload}; the shared,
        # editable column names line up over the value cells.
        self._paths: list[str] = []
        self._records: dict[str, dict] = {}
        self._column_names: list[str] = []
        self._rows: list[ExperimentRow] = []
        self._preview_rows: list[ExperimentRow] = []

        # Staged (discovered, not-yet-committed) entries + their delete-selection.
        self._discovered: list[dict] = []
        self._discovered_selected: set[str] = set()

        # Committed-row selection: a single active row plus a multi-selection for
        # delete/run; ``_anchor`` is the Shift-range pivot.
        self._active: str | None = None
        self._selected_paths: set[str] = set()
        self._anchor: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(COMPACT_SPACING)

        heading = QLabel(title)
        heading.setObjectName("experiments_panel_label")
        heading.setStyleSheet(f"color: {TEXT_MID}; font-weight: bold;")
        layout.addWidget(heading)

        self.setup_section = self._build_setup_section()
        layout.addWidget(self.setup_section)

        self._staging_label = QLabel("")
        self._staging_label.setStyleSheet(f"color: {TEXT_DIM};")
        self._staging_label.setVisible(False)
        layout.addWidget(self._staging_label)

        # Rows in a bounded scroll region: a long list scrolls internally instead
        # of shoving the action bar off-screen. The editable column header is the
        # first item inside, so it always aligns with the value cells below.
        self._rows_box = QVBoxLayout()
        self._rows_box.setContentsMargins(0, 0, 0, 0)
        self._rows_box.setSpacing(2)
        self._rows_box.setAlignment(Qt.AlignTop)
        rows_container = QWidget()
        rows_container.setLayout(self._rows_box)
        self._rows_scroll = QScrollArea()
        self._rows_scroll.setObjectName("experiments_rows_scroll")
        self._rows_scroll.setWidgetResizable(True)
        self._rows_scroll.setMinimumHeight(220)
        self._rows_scroll.setMaximumHeight(480)
        self._rows_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._rows_scroll.setWidget(rows_container)
        layout.addWidget(self._rows_scroll)

        layout.addLayout(self._build_list_actions())
        if self._show_run:
            layout.addLayout(self._build_run_actions())

        self._meta = QLabel("")
        self._meta.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self._meta)

        self._rebuild_table()
        self._update_meta()

    # -- setup: calibration + input names + optional output dir ----------
    def _build_setup_section(self) -> CollapsibleSection:
        inner = QWidget()
        box = QVBoxLayout(inner)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(COMPACT_SPACING)
        if self._show_calibration:
            box.addLayout(self._build_calibration_row())
        box.addLayout(self._build_input_config())
        if self._show_manual_columns:
            box.addLayout(self._build_manual_columns())
        if self._show_output_dir:
            box.addLayout(self._build_output_dir_row())
        return CollapsibleSection("Setup", inner, expanded=True, title_color=TEXT_MID)

    def _build_calibration_row(self) -> QHBoxLayout:
        self.calibration_controls: dict[str, QLineEdit] = {}
        cal = QHBoxLayout()
        cal.setContentsMargins(0, 0, 0, 0)
        cal.setSpacing(COMPACT_SPACING + 4)
        for name, label, min_val, max_val in _CALIBRATION_SPECS:
            field = QLineEdit()
            validator = QDoubleValidator(min_val, max_val, _INPUT_DECIMALS, field)
            validator.setNotation(QDoubleValidator.StandardNotation)
            field.setValidator(validator)
            field.setObjectName(f"experiments_calibration_{name}")
            field.setStyleSheet(mono_input_style())
            field.editingFinished.connect(
                lambda n=name, c=field: self.calibration_changed.emit(n, c.text())
            )
            self.calibration_controls[name] = field

            caption = QLabel(label)
            caption.setStyleSheet(f"color: {TEXT_MID};")
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(1)
            cell.addWidget(caption)
            cell.addWidget(field)
            cal.addLayout(cell, 1)
        return cal

    def calibration_values(self) -> dict[str, str]:
        if not self._show_calibration:
            return {}
        return {n: c.text().strip() for n, c in self.calibration_controls.items()}

    def set_calibration_values(self, values: dict) -> None:
        if not self._show_calibration:
            return
        for name, control in self.calibration_controls.items():
            if name in values:
                control.blockSignals(True)
                control.setText(_format_value(values[name]))
                control.blockSignals(False)

    def _build_input_config(self) -> QVBoxLayout:
        """Input file names — the discovery requirements, copied to each scan."""
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(COMPACT_SPACING)
        files = QFormLayout()
        files.setContentsMargins(0, 0, 0, 0)
        files.setSpacing(2)
        self.input_name_fields: dict[str, QLineEdit] = {}
        for key, label, default in self._input_fields:
            field = QLineEdit(default)
            field.setObjectName(f"experiments_input_{key}")
            field.setStyleSheet(mono_input_style())
            self.input_name_fields[key] = field
            name = QLabel(label)
            name.setStyleSheet(f"color: {TEXT_MID};")
            files.addRow(name, field)
        box.addLayout(files)
        return box

    def input_names(self) -> dict[str, str]:
        """Current non-blank input file names, keyed by field key."""
        return {
            key: field.text().strip()
            for key, field in self.input_name_fields.items()
            if field.text().strip()
        }

    def _build_manual_columns(self) -> QVBoxLayout:
        """Batch-wide constant tags copied onto every Discover→Commit row.

        For descriptors not encoded in the folder tree (operator, treatment).
        Editing a field re-renders the staged previews; the values are baked into
        each row's columns at Add-to-list time.
        """
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        self._manual_box = QVBoxLayout()
        self._manual_box.setContentsMargins(12, 0, 0, 0)
        self._manual_box.setSpacing(2)
        box.addLayout(self._manual_box)
        add_btn = QToolButton()
        add_btn.setObjectName("experiments_add_column_button")
        add_btn.setText("+ Add column")
        add_btn.setToolTip(
            "Tag every added position with a constant column not in the folder "
            "tree, e.g. operator or treatment."
        )
        add_btn.clicked.connect(lambda: self.add_manual_column())
        box.addWidget(add_btn)
        return box

    def add_manual_column(self, name: str = "", value: str = "") -> None:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        name_edit = QLineEdit(name)
        name_edit.setPlaceholderText("column")
        name_edit.setStyleSheet(mono_input_style())
        value_edit = QLineEdit(value)
        value_edit.setPlaceholderText("value")
        value_edit.setStyleSheet(mono_input_style())
        for edit in (name_edit, value_edit):
            edit.textChanged.connect(lambda _t: self._rebuild_table())
        row.addWidget(name_edit, 1)
        row.addWidget(value_edit, 1)
        self._manual_box.addLayout(row)
        self._manual_fields.append((name_edit, value_edit))

    def manual_columns(self) -> dict[str, str]:
        """Current non-blank batch-wide constant columns (name → value)."""
        cols: dict[str, str] = {}
        for name_edit, value_edit in self._manual_fields:
            name = name_edit.text().strip()
            if name:
                cols[name] = value_edit.text().strip()
        return cols

    def _build_output_dir_row(self) -> QHBoxLayout:
        out = QHBoxLayout()
        out.setContentsMargins(0, 0, 0, 0)
        self.output_dir_button = QToolButton()
        self.output_dir_button.setObjectName("experiments_output_dir_button")
        self.output_dir_button.setText("Add custom output directory")
        self.output_dir_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.output_dir_button.clicked.connect(self.output_dir_requested)
        self.output_dir_label = QLabel("")
        self.output_dir_label.setObjectName("experiments_output_dir_label")
        self.output_dir_label.setStyleSheet(f"color: {TEXT_DIM};")
        self.output_dir_label.setVisible(False)
        out.addWidget(self.output_dir_button)
        out.addWidget(self.output_dir_label, 1)
        return out

    def set_output_dir_text(self, text: str) -> None:
        if not self._show_output_dir:
            return
        self.output_dir_label.setText(text)
        self.output_dir_label.setToolTip(text)
        self.output_dir_label.setVisible(bool(text))
        self.output_dir_button.setText(
            "Change output directory" if text else "Add custom output directory"
        )

    # -- list + run actions ----------------------------------------------
    def _build_list_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.discover_btn = QToolButton()
        self.discover_btn.setObjectName("experiments_discover_button")
        self.discover_btn.setText("Discover")
        self.discover_btn.clicked.connect(self.discover_requested)
        row.addWidget(self.discover_btn)

        self.commit_btn = QToolButton()
        self.commit_btn.setObjectName("experiments_commit_button")
        self.commit_btn.setText("Add to list")
        self.commit_btn.setEnabled(False)
        self.commit_btn.clicked.connect(self.commit_discovered)
        row.addWidget(self.commit_btn)

        self.delete_btn = QToolButton()
        self.delete_btn.setObjectName("experiments_delete_button")
        self.delete_btn.setText("Delete selected")
        self.delete_btn.setToolTip(
            "Remove the selected rows (Ctrl/Shift-click to select several)"
        )
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self.delete_selected)
        row.addWidget(self.delete_btn)
        row.addStretch()
        return row

    def _build_run_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.run_btn = QToolButton()
        self.run_btn.setObjectName("experiments_run_button")
        self.run_btn.setText("Run selected")
        self.run_btn.setToolTip(
            "Run the selected rows (Ctrl/Shift-click for several, Ctrl+A for all)"
        )
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._on_run_clicked)
        row.addWidget(self.run_btn)
        row.addStretch()

        workers_label = QLabel("Workers:")
        workers_label.setStyleSheet(f"color: {TEXT_DIM};")
        row.addWidget(workers_label)
        self.workers_spinbox = QSpinBox()
        self.workers_spinbox.setObjectName("experiments_workers_spinbox")
        self.workers_spinbox.setRange(1, os.cpu_count() or 1)
        self.workers_spinbox.setValue(1)
        self.workers_spinbox.setToolTip("How many positions Run processes in parallel")
        row.addWidget(self.workers_spinbox)
        return row

    def num_workers(self) -> int:
        return self.workers_spinbox.value() if self._show_run else 1

    def _on_run_clicked(self) -> None:
        self.run_requested.emit(self.selected_payloads(), self.num_workers())

    # -- discovery (two-step Discover → Commit) --------------------------
    def discover(self, root: str) -> list[dict]:
        """Step 1: stage the entries ``discover_fn`` finds under *root*.

        A second call *replaces* the current staged set. Staging never mutates the
        committed list — the Add-to-list step does.
        """
        if self._discover_fn is None:
            return []
        self._discovered = list(self._discover_fn(root, self.input_names()))
        self._discovered_selected = set()
        self._update_staging()
        self._rebuild_table()
        return list(self._discovered)

    def discovered(self) -> list[dict]:
        return list(self._discovered)

    def commit_discovered(self) -> None:
        """Step 2: add every staged entry as a committed row.

        Batch-wide manual columns are baked into each entry's columns here, so a
        committed row keeps its own snapshot even if the fields change later.
        """
        if not self._discovered:
            return
        manual = self.manual_columns()
        entries = []
        for entry in self._discovered:
            merged = dict(entry.get("columns") or {})
            merged.update(manual)
            entries.append({**entry, "columns": merged})
        self._discovered = []
        self._discovered_selected = set()
        self._add_entries(entries)
        self._update_staging()

    def _update_staging(self) -> None:
        n = len(self._discovered)
        self.commit_btn.setEnabled(n > 0)
        self._staging_label.setText(
            f"{n} folder{'s' if n != 1 else ''} discovered — Add to list" if n else ""
        )
        self._staging_label.setVisible(n > 0)

    # -- committed-row model ---------------------------------------------
    def column_names(self) -> list[str]:
        return list(self._column_names)

    def rename_column(self, index: int, new_name: str) -> None:
        """Rename column *index* table-wide, carrying each row's value across."""
        if not 0 <= index < len(self._column_names):
            return
        old = self._column_names[index]
        new = new_name.strip()
        if not new or new == old:
            return
        self._column_names[index] = new
        for key in self._paths:
            cols = self._records[key]["columns"]
            if old in cols:
                cols[new] = cols.pop(old)
        self._rebuild_table()
        self.records_changed.emit()

    def _ensure_columns(self, names: Iterable[str]) -> None:
        for name in names:
            if name and name not in self._column_names:
                self._column_names.append(name)

    def _add_entries(self, entries: list[dict]) -> None:
        """Append entries as committed rows (de-duped by key), extending columns."""
        was_empty = not self._paths
        added = False
        for entry in entries:
            key = str(entry["key"])
            if key in self._records:
                continue
            cols = dict(entry.get("columns") or {})
            self._ensure_columns(cols.keys())
            self._records[key] = {"columns": cols, "payload": entry.get("payload")}
            self._paths.append(key)
            added = True
        if not added:
            return
        self._rebuild_table()
        self._update_meta()
        self.records_changed.emit()
        if was_empty:
            self.set_active(self._paths[0])
            self.setup_section.collapse()

    def set_records(self, entries: list[dict]) -> None:
        """Replace the whole committed list from host-owned records (CSV / load).

        Each entry is ``{"key", "columns", "payload"}``. The shared column header is
        rebuilt from the union of column names (first-seen order).
        """
        self._paths = []
        self._records = {}
        self._column_names = []
        self._selected_paths = set()
        for entry in entries:
            key = str(entry["key"])
            if key in self._records:
                continue
            cols = dict(entry.get("columns") or {})
            self._ensure_columns(cols.keys())
            self._records[key] = {"columns": cols, "payload": entry.get("payload")}
            self._paths.append(key)
        if self._active not in self._paths:
            self._active = None
        self._rebuild_table()
        self._update_meta()
        self.setup_section.expand() if not self._paths else self.setup_section.collapse()
        self.records_changed.emit()

    def payloads(self) -> list[object]:
        """Every committed row's payload, in row order."""
        return [self._records[key]["payload"] for key in self._paths]

    def selected_payloads(self) -> list[object]:
        return [
            self._records[key]["payload"]
            for key in self._paths
            if key in self._selected_paths
        ]

    def active_payload(self) -> object:
        return self._records[self._active]["payload"] if self._active else None

    def _row_record(self, key: str) -> object:
        """One row's payload with the *live* editable columns merged in.

        When the payload is a dict (the catalog-record case), the row's current
        column values — including in-header renames — override its ``columns`` bag,
        so a host that reads :meth:`records` always sees what the header shows.
        """
        rec = self._records[key]
        payload = rec["payload"]
        if isinstance(payload, dict):
            return {**payload, "columns": dict(rec["columns"])}
        return payload

    def records(self) -> list[dict]:
        """Every committed row as ``payload + live columns`` (row order)."""
        return [self._row_record(key) for key in self._paths]

    def selected_records(self) -> list[dict]:
        """In-scope rows as ``payload + live columns`` (empty if nothing selected)."""
        return [
            self._row_record(key) for key in self._paths if key in self._selected_paths
        ]

    def active_record(self) -> object:
        return self._row_record(self._active) if self._active else None

    def keys(self) -> list[str]:
        return list(self._paths)

    # -- selection --------------------------------------------------------
    def set_active(self, key: str | None, *, selection=None) -> None:
        if key is not None and key not in self._paths:
            return
        changed = key != self._active
        self._active = key
        if selection is None:
            self._selected_paths = {key} if key else set()
        else:
            self._selected_paths = {k for k in selection if k in self._paths}
        self._apply_selection_styles()
        self._update_action_buttons()
        if changed:
            self.active_changed.emit(self.active_payload())
        self.selection_changed.emit()

    def select_all(self) -> None:
        self._selected_paths = set(self._paths)
        self._apply_selection_styles()
        self._update_action_buttons()
        self.selection_changed.emit()

    def clear_selection(self) -> None:
        self._active = None
        self._selected_paths = set()
        self._anchor = None
        self._apply_selection_styles()
        self._update_action_buttons()
        self.selection_changed.emit()

    def delete_selected(self) -> None:
        """Delete staged previews first (if any selected), else committed rows."""
        if self._discovered_selected:
            self._discovered = [
                e for e in self._discovered if str(e["key"]) not in self._discovered_selected
            ]
            self._discovered_selected = set()
            self._update_staging()
            self._rebuild_table()
            self._update_action_buttons()
            return
        if not self._selected_paths:
            return
        remaining = [k for k in self._paths if k not in self._selected_paths]
        if self._active in self._selected_paths:
            self._active = None
        self._selected_paths = set()
        self._paths = remaining
        self._records = {k: self._records[k] for k in remaining}
        self._rebuild_table()
        self._update_meta()
        self.records_changed.emit()
        self.selection_changed.emit()

    def _on_row_clicked(self, key: str, flag: int) -> None:
        self.setFocus()
        if flag == 1:  # Ctrl: toggle
            selection = set(self._selected_paths)
            selection.discard(key) if key in selection else selection.add(key)
            self._anchor = key
            active = key if key in selection else next(iter(selection), None)
            self.set_active(active, selection=selection)
        elif flag == 2 and self._anchor in self._paths:  # Shift: range
            lo, hi = sorted((self._paths.index(self._anchor), self._paths.index(key)))
            self.set_active(key, selection=set(self._paths[lo : hi + 1]))
        else:  # plain: single select + activate
            self._anchor = key
            self.set_active(key)

    def _on_preview_clicked(self, key: str, _flag: int) -> None:
        self._discovered_selected.symmetric_difference_update({key})
        for row in self._preview_rows:
            row.set_selected(row.key in self._discovered_selected)
        self._update_action_buttons()

    def _apply_selection_styles(self) -> None:
        for row in self._rows:
            row.set_selected(row.key in self._selected_paths)

    def _update_action_buttons(self) -> None:
        self.delete_btn.setEnabled(
            bool(self._selected_paths) or bool(self._discovered_selected)
        )
        if self._show_run:
            self.run_btn.setEnabled(bool(self._selected_paths))

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and (
            self._selected_paths or self._discovered_selected
        ):
            self.delete_selected()
            event.accept()
            return
        if event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier:
            self.select_all()
            event.accept()
            return
        super().keyPressEvent(event)

    # -- status -----------------------------------------------------------
    def refresh_statuses(self) -> None:
        """Re-read every committed row's stage status and repaint rail + chip."""
        if self._status_fn is None:
            return
        for row in self._rows:
            payload = self._records[row.key]["payload"]
            try:
                status = self._status_fn(payload)
            except Exception:
                status = {stage: UNKNOWN for stage in STAGES}
            row.set_status(status)

    def _on_row_dot_clicked(self, key: str, stage: str) -> None:
        payload = self._records.get(key, {}).get("payload")
        self.stage_load_requested.emit(payload, stage)

    # -- rendering --------------------------------------------------------
    def _build_header_widget(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("experiments_table_header")
        row = QHBoxLayout(widget)
        row.setContentsMargins(7, 5, 10, 5)
        row.setSpacing(COMPACT_SPACING + 4)

        lead = QWidget()
        lead.setFixedWidth(_SELBAR_W)
        row.addWidget(lead)

        self._header_fields: list[QLineEdit] = []
        if self._column_names:
            for index, name in enumerate(self._column_names):
                field = QLineEdit(name)
                field.setObjectName(f"experiments_column_header_{index}")
                field.setStyleSheet(mono_input_style())
                field.editingFinished.connect(
                    lambda i=index, f=field: self.rename_column(i, f.text())
                )
                row.addWidget(field, 1)
                self._header_fields.append(field)
        else:
            placeholder = QLineEdit("Folder")
            placeholder.setEnabled(False)
            placeholder.setStyleSheet(mono_input_style())
            row.addWidget(placeholder, 1)

        rail = QWidget()
        rail.setFixedWidth(_RAIL_W)
        row.addWidget(rail)
        chip = QWidget()
        chip.setFixedWidth(_CHIP_W)
        row.addWidget(chip)
        return widget

    def _rebuild_table(self) -> None:
        while self._rows_box.count():
            item = self._rows_box.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._rows = []
        self._preview_rows = []

        self._rows_box.addWidget(self._build_header_widget())

        for key in self._paths:
            cols = self._records[key]["columns"]
            values = [cols.get(name, "") for name in self._column_names]
            row = ExperimentRow(key, values or None)
            row.clicked.connect(self._on_row_clicked)
            row.dot_clicked.connect(self._on_row_dot_clicked)
            row.set_selected(key in self._selected_paths)
            self._rows_box.addWidget(row)
            self._rows.append(row)

        manual = self.manual_columns()
        for entry in self._discovered:
            key = str(entry["key"])
            cols = dict(entry.get("columns") or {})
            cols.update(manual)
            values = [cols.get(name, "") for name in self._column_names] if self._column_names else list(cols.values())
            row = ExperimentRow(key, values or None, preview=True)
            row.clicked.connect(self._on_preview_clicked)
            row.set_selected(key in self._discovered_selected)
            self._rows_box.addWidget(row)
            self._preview_rows.append(row)

        self._rows_scroll.setVisible(bool(self._paths) or bool(self._discovered))
        self.refresh_statuses()

    def _update_meta(self) -> None:
        n = len(self._paths)
        self._meta.setText(f"{n} position{'s' if n != 1 else ''}")
        self._update_action_buttons()
