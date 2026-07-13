"""The shared experiments/positions panel (napariTFM ExperimentsList parity).

One panel, two homes: the main CellFlow app and the standalone Contact Analysis
distro both mount this. Descended from napariTFM's ExperimentsList, but reshaped
around a filesystem-centric flow —

    Setup (calibration + input-file names + optional output dir)
      → Find data folders  → one additive scan of a parent root; every matching
                             data folder is added straight to the list (deduped),
                             each a committed row with an editable folder-nesting
                             header, an accent select-bar, a per-position status
                             rail, and an overall-status chip
      → tag selected rows  → set a condition column on the current selection, the
                             grouping columns of the aggregate tidy table
      → Run selected / Workers, and a running count

— and speaks CellFlow: the rail is :class:`~cellflow.napari._status_rail.StatusRail`
(the five-state commit-contract vocabulary), the styling is CellFlow's own designed
surface tokens, and *what* a scan looks for / *how* a row maps to a host record is
injected by the host (``discover_fn`` / ``status_fn``).

The panel owns the displayed catalog (the ordered rows + their editable columns);
the host reads :meth:`payloads` for scope / run / save and reacts to the signals.
Each row carries an opaque ``payload`` (the host's record dict) threaded through
untouched, plus a ``columns`` dict keyed by the shared, editable column names.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QTextBrowser,
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
    action_button_style,
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
    # Entered in minutes (long timelapses read naturally that way); the host
    # converts to the backend's seconds-based ``time_interval_s`` at the boundary.
    ("time_interval_min", "Frame Length (min)", 0.0, 1e9),
)
_INPUT_DECIMALS = 6

# Fixed widths of the non-column cells, shared by the header and every data row so
# the editable column headers line up over their value cells.
_SELBAR_W = 3
_RAIL_W = 60
_CHIP_W = 56

#: Overall-status word → chip text (kept short so the chip column stays narrow).
_CHIP_TEXT = {"run": "run", "done": "done", "queued": "queued"}

#: The `?` quickstart, distilled from docs/manual/workflow.md. Self-contained
#: (no browser / hosted-docs dependency: the standalone distros mount this too).
_QUICKSTART_HTML = """
<h3>One folder per movie</h3>
<p>Each movie or field of view lives in its own <i>data folder</i> holding the raw
nucleus and cell images, and every stage writes its results back into that same
folder: the folder on disk is the source of truth for results. The list of data
folders and how you classify them (conditions, replicates) is your <i>project</i>,
which you save to and reload from a <b>project catalog</b> (a CSV). That catalog is
also what drives aggregate quantification across the whole set.</p>

<h3>A worked example</h3>
<p>Say your images sit like this, two conditions and three fields of view:</p>
<pre>experiment/
  WT/
    pos01/   nucleus.tif   cell.tif
    pos02/   nucleus.tif   cell.tif
  KO/
    pos01/   nucleus.tif   cell.tif</pre>
<p>Name the two files in <b>Setup</b>, then point <b>Find data folders</b> at
<code>experiment/</code>. CellFlow adds the three folders that hold both images and
reads the <code>WT</code> / <code>KO</code> / <code>pos..</code> nesting into
columns. Select the two <code>WT</code> rows and set <code>condition&nbsp;=&nbsp;WT</code>;
select the <code>KO</code> row and set <code>condition&nbsp;=&nbsp;KO</code>. Those
columns become the grouping columns of the aggregate table.</p>

<h3>Where results go</h3>
<p><b>Run</b> processes the selected folders and writes each stage back inside the
data folder:</p>
<pre>pos01/
  0_input   1_cellpose   2_nucleus   3_cell   4_contact_analysis</pre>
<p>The status rail on each row shows how far that folder got. Run <b>Find data
folders</b> again on another parent to add more: the list accumulates and skips any
folder already listed.</p>
"""


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


def _count_above_root(entries: list[dict], root: str) -> int:
    """How many found entries resolve to a folder *above* the scanned *root*.

    A folder-path key is "above root" when root is neither it nor one of its
    ancestors: the sign of a too-deep pick (the user chose a data folder's own
    subfolder, so its position resolves to root's parent). Keys that aren't
    filesystem paths are ignored.
    """
    try:
        root_resolved = Path(root).resolve()
    except (OSError, ValueError):
        return 0
    n = 0
    for entry in entries:
        try:
            key = Path(str(entry["key"])).resolve()
        except (OSError, ValueError):
            continue
        if key != root_resolved and root_resolved not in key.parents:
            n += 1
    return n


class ExperimentRow(QWidget):
    """One data folder: accent select-bar, per-column value cells, rail, status chip.

    Shows its rail + chip and selects/activates on click.
    """

    clicked = Signal(str, int)  # key, modifier flag: 0 plain, 1 ctrl, 2 shift
    dot_clicked = Signal(str, str)  # key, stage — one rail dot's on-demand load

    def __init__(
        self,
        key: str,
        values: list[str] | None = None,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._key = key
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

        self.set_selected(False)

    @property
    def key(self) -> str:
        return self._key

    def is_selected(self) -> bool:
        return self._selected

    def set_selected(self, on: bool) -> None:
        self._selected = on
        accent = stage_accent("nucleus")
        self._selbar.setStyleSheet(
            f"background: {accent};" if on else "background: transparent;"
        )
        self.setStyleSheet(experiment_row_style(on, accent))
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
    """Setup + an additive Find + an editable-column list of data-folder rows.

    Host contract:

    * ``discover_fn(root, input_names) -> list[entry]`` — scan a root; each *entry*
      is ``{"key": str, "columns": {name: value}, "payload": dict}`` (``key`` is the
      row identity, typically the data-folder path).
    * ``status_fn(payload) -> {stage: state}`` — the per-stage rail status.

    Signals let the host stay the model of record for persistence/scope while the
    panel owns the list UI, Setup, folder discovery, and selection.
    """

    active_changed = Signal(object)      # payload | None — the single active row
    selection_changed = Signal()         # multi-selection changed (read selected_payloads)
    records_changed = Signal()           # committed row set changed
    run_requested = Signal(list, int)    # (selected payloads, num_workers)
    stage_load_requested = Signal(object, str)  # (payload, stage)
    calibration_changed = Signal(str, str)      # (name, text)
    discover_requested = Signal()               # Find-data-folders button clicked
    output_dir_requested = Signal()             # output-dir button clicked

    def __init__(
        self,
        *,
        title: str = "Data folders",
        input_fields: Iterable[tuple[str, str, str]] = (),
        discover_fn: Callable[[str, dict[str, str]], list[dict]] | None = None,
        status_fn: Callable[[dict], dict[str, str]] | None = None,
        show_calibration: bool = True,
        show_output_dir: bool = False,
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
        self._show_run = show_run

        # Committed rows: ordered keys + per-key {columns, payload}; the shared,
        # editable column names line up over the value cells.
        self._paths: list[str] = []
        self._records: dict[str, dict] = {}
        self._column_names: list[str] = []
        self._rows: list[ExperimentRow] = []

        # Committed-row selection: a single active row plus a multi-selection for
        # delete/run; ``_anchor`` is the Shift-range pivot.
        self._active: str | None = None
        self._selected_paths: set[str] = set()
        self._anchor: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(COMPACT_SPACING)

        layout.addLayout(self._build_heading(title))

        self.setup_section = self._build_setup_section()
        layout.addWidget(self.setup_section)

        # One status line, double duty: the empty-state call to action when the
        # list is empty, transient "Added N…" / dry-scan feedback after a Find.
        self._hint = QLabel("")
        self._hint.setObjectName("experiments_hint_label")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self._hint)

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

    # -- heading + quickstart --------------------------------------------
    def _build_heading(self, title: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        heading = QLabel(title)
        heading.setObjectName("experiments_panel_label")
        heading.setStyleSheet(f"color: {TEXT_MID}; font-weight: bold;")
        row.addWidget(heading)
        row.addStretch()
        self.help_btn = QToolButton()
        self.help_btn.setObjectName("experiments_help_button")
        self.help_btn.setText("?")
        self.help_btn.setToolTip("Quickstart: how CellFlow manages your data")
        self.help_btn.clicked.connect(self._show_quickstart)
        row.addWidget(self.help_btn)
        return row

    def _show_quickstart(self) -> None:
        dialog = QDialog(self)
        dialog.setObjectName("experiments_quickstart_dialog")
        dialog.setWindowTitle("CellFlow quickstart")
        dialog.setMinimumWidth(460)
        box = QVBoxLayout(dialog)
        body = QTextBrowser()
        body.setOpenExternalLinks(False)
        body.setHtml(_QUICKSTART_HTML)
        box.addWidget(body)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        box.addWidget(buttons)
        dialog.exec_()

    # -- setup: calibration + input names + optional output dir ----------
    def _build_setup_section(self) -> CollapsibleSection:
        inner = QWidget()
        box = QVBoxLayout(inner)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(COMPACT_SPACING)
        if self._show_calibration:
            box.addLayout(self._build_calibration_row())
        box.addLayout(self._build_input_config())
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
        self.discover_btn.setText("Find data folders…")
        self.discover_btn.setToolTip(
            "Pick a parent directory; every folder under it holding the Setup "
            "image files is added to the list (run again to add more)."
        )
        self.discover_btn.setStyleSheet(action_button_style())
        self.discover_btn.clicked.connect(self.discover_requested)
        row.addWidget(self.discover_btn)

        self.delete_btn = QToolButton()
        self.delete_btn.setObjectName("experiments_delete_button")
        self.delete_btn.setText("Delete selected")
        self.delete_btn.setToolTip(
            "Remove the selected rows (Ctrl/Shift-click to select several)"
        )
        self.delete_btn.setEnabled(False)
        self.delete_btn.setStyleSheet(action_button_style())
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
        self.run_btn.setStyleSheet(action_button_style())
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
        self.workers_spinbox.setToolTip("How many data folders Run processes in parallel")
        row.addWidget(self.workers_spinbox)
        return row

    def num_workers(self) -> int:
        return self.workers_spinbox.value() if self._show_run else 1

    def _on_run_clicked(self) -> None:
        self.run_requested.emit(self.selected_payloads(), self.num_workers())

    # -- discovery (one additive Find) -----------------------------------
    def discover(self, root: str) -> list[dict]:
        """Scan *root* via ``discover_fn`` and add every match straight to the list.

        Additive and de-duped by key: re-running against another root accumulates
        and never disturbs existing (possibly edited) rows. Returns the newly added
        entries and leaves a transient hint naming what happened.
        """
        if self._discover_fn is None:
            return []
        found = list(self._discover_fn(root, self.input_names()))
        added = self._add_entries(found)
        if added:
            message = (
                f"Added {added} data folder{'s' if added != 1 else ''}."
                + ("" if added == len(found) else f" ({len(found) - added} already listed.)")
            )
            above = _count_above_root(found, root)
            if above:
                message += (
                    f" {'It sits' if above == 1 else f'{above} sit'} above the folder "
                    "you picked: point Find at the parent directory to scan a whole batch."
                )
            self._refresh_hint(message)
        else:
            self._refresh_hint(
                f"No new data folders found under {root} — "
                "check the image filenames in Setup."
                if not found
                else f"All {len(found)} found data folders were already listed."
            )
        return found

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

    def remove_column(self, index: int) -> None:
        """Drop column *index* table-wide, discarding each row's value for it.

        A folder-derived column re-appears on the next Discover; this only
        removes it from the current committed list.
        """
        if not 0 <= index < len(self._column_names):
            return
        name = self._column_names.pop(index)
        for key in self._paths:
            self._records[key]["columns"].pop(name, None)
        self._rebuild_table()
        self.records_changed.emit()

    def _ensure_columns(self, names: Iterable[str]) -> None:
        for name in names:
            if name and name not in self._column_names:
                self._column_names.append(name)

    def _add_entries(self, entries: list[dict]) -> int:
        """Append entries as committed rows (de-duped by key), extending columns.

        Returns the number of rows actually added (skips keys already listed).
        """
        was_empty = not self._paths
        added = 0
        for entry in entries:
            key = str(entry["key"])
            if key in self._records:
                continue
            cols = dict(entry.get("columns") or {})
            self._ensure_columns(cols.keys())
            self._records[key] = {"columns": cols, "payload": entry.get("payload")}
            self._paths.append(key)
            added += 1
        if not added:
            return 0
        self._rebuild_table()
        self._update_meta()
        self.records_changed.emit()
        if was_empty:
            self.set_active(self._paths[0])
            self.setup_section.collapse()
        return added

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
        """Remove the selected committed rows."""
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

    def _apply_selection_styles(self) -> None:
        for row in self._rows:
            row.set_selected(row.key in self._selected_paths)

    def _update_action_buttons(self) -> None:
        has_sel = bool(self._selected_paths)
        self.delete_btn.setEnabled(has_sel)
        if self._show_run:
            self.run_btn.setEnabled(has_sel)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self._selected_paths:
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
                cell = QWidget()
                cell_row = QHBoxLayout(cell)
                cell_row.setContentsMargins(0, 0, 0, 0)
                cell_row.setSpacing(2)
                field = QLineEdit(name)
                field.setObjectName(f"experiments_column_header_{index}")
                field.setStyleSheet(mono_input_style())
                field.editingFinished.connect(
                    lambda i=index, f=field: self.rename_column(i, f.text())
                )
                cell_row.addWidget(field, 1)
                remove = QToolButton()
                remove.setObjectName(f"experiments_remove_column_{index}")
                remove.setText("×")
                remove.setToolTip(f"Remove the '{name}' column")
                remove.setAutoRaise(True)
                remove.setFixedWidth(16)
                remove.setStyleSheet(f"color: {TEXT_DIM}; border: none;")
                remove.clicked.connect(lambda _=False, i=index: self.remove_column(i))
                cell_row.addWidget(remove)
                row.addWidget(cell, 1)
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

        self._rows_scroll.setVisible(bool(self._paths))
        self.refresh_statuses()

    def _refresh_hint(self, message: str | None = None) -> None:
        """The one status line: a transient *message*, else the empty-state CTA."""
        if message:
            self._hint.setText(message)
        elif not self._paths:
            self._hint.setText(
                "Add folders with cell and nucleus images to start. "
                "Click ? for how CellFlow stores its work."
            )
        else:
            self._hint.setText("")

    def _update_meta(self) -> None:
        n = len(self._paths)
        self._meta.setText(f"{n} data folder{'s' if n != 1 else ''}")
        self._refresh_hint()
        self._update_action_buttons()
