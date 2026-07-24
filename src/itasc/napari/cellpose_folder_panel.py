"""Folder-mode panel for the standalone Cellpose *segment + track* widget.

The standalone tool is layer-bound: you bind Channel 1 / Channel 2 to image
layers already in the viewer and save results by hand. This panel adds the full
app's *project-walk* convenience *on top of the same interactive tool* — point it
at a project root, it discovers the position folders, and you walk them one at a
time: select a position (its images load and bind), segment/track/correct
interactively, then save the masks back into that folder under a name you chose.

It is **not** unattended batch: every position is still segmented and corrected by
a human, exactly as in layer mode. The list buys navigation and persistence,
nothing more.

Why this reimplements discovery + the status dot instead of reusing the app's
:class:`~itasc.napari._experiments_panel.ExperimentsPanel` /
``_discover_positions`` / :class:`~itasc.napari._status_rail.StatusRail`: those
live in the app / aggregate half and pull in :mod:`itasc.contact_analysis`, which
the ``itasc-cellpose`` wheel deliberately does not ship. This panel leans only on
``itasc-core``-shipped primitives (:class:`StandalonePathsMixin`,
:class:`CollapsibleSection`, the ``ui_style`` tokens) so it composes inside the
standalone distribution.

The four name fields do double duty — the two inputs are the discovery pattern,
the two outputs are the save target — and persist across sessions via QSettings.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from itasc.napari._standalone_paths import StandalonePathsMixin
from itasc.napari.ui_style import (
    TEXT_DIM,
    TEXT_MID,
    action_button_style,
    mono_input_style,
)
from itasc.napari.widgets import CollapsibleSection

#: QSettings scope for the four persisted name fields.
_SETTINGS_APP = "cellpose-segment-track"

# Per-position status vocabulary. Segment + track is a single stage here, so the
# app's four-dot rail collapses to one dot with three states.
MISSING = "missing"  # no output file on disk for this position
WORKING = "working"  # the active position, worked but not yet saved
SAVED = "saved"      # the output file(s) exist on disk

#: state → (fill, border), matching _status_rail's palette so the dot reads the
#: same as the app's rail on napari's dark theme.
_DOT_COLORS: dict[str, tuple[str, str]] = {
    MISSING: ("transparent", "#7a7a7a"),
    WORKING: ("#e0a020", "#a8791a"),  # amber — being worked, not saved
    SAVED: ("#3aa84a", "#2c7d38"),    # green — on disk
}
_DOT_PX = 12

_ROLE_ENTRY = Qt.UserRole  # the discovery entry dict stored on each list item


def _dot_icon(state: str) -> QIcon:
    """A small filled/hollow circle for one position's status, as a list icon."""
    pixmap = QPixmap(_DOT_PX, _DOT_PX)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    fill, border = _DOT_COLORS.get(state, _DOT_COLORS[MISSING])
    painter.setPen(QColor(border))
    painter.setBrush(Qt.transparent if fill == "transparent" else QColor(fill))
    painter.drawEllipse(1, 1, _DOT_PX - 3, _DOT_PX - 3)
    painter.end()
    return QIcon(pixmap)


def discover_folders(
    root: str | Path, ch1_name: str, ch2_name: str
) -> list[dict]:
    """Find position folders under *root* by their Channel 1 (and Channel 2) files.

    Each name is a bare file name or a path relative to the position folder
    (e.g. ``0_input/nucleus.tif``); the basename is ``rglob``-ed and the trailing
    parts must match, mirroring the app's ``_discover_positions`` — but without its
    ``itasc.contact_analysis`` column-derivation, which this distro doesn't ship.

    **Channel 1 is required, Channel 2 optional**: a folder is a position only when
    it holds the Channel 1 file. Its Channel 2 is bound only when that file sits in
    the same folder; otherwise the position runs single-channel. Returns entries
    ``{"position": Path, "ch1": Path, "ch2": Path | None, "rel": str}`` sorted by
    path, where ``rel`` is the folder path shown in the list (relative to *root*).
    """
    root_path = Path(root)
    if not root_path.is_dir() or not ch1_name:
        return []

    ch1_rel = Path(ch1_name)
    ch2_rel = Path(ch2_name) if ch2_name else None

    entries: list[dict] = []
    seen: set[Path] = set()
    for match in sorted(root_path.rglob(ch1_rel.name)):
        if not match.is_file():
            continue
        if len(ch1_rel.parts) > 1 and match.parts[-len(ch1_rel.parts):] != ch1_rel.parts:
            continue
        position = match
        for _ in ch1_rel.parts:
            position = position.parent
        position = position.resolve()
        if position in seen:
            continue
        seen.add(position)

        ch2_path: Path | None = None
        if ch2_rel is not None:
            candidate = position / ch2_rel
            if candidate.is_file():
                ch2_path = candidate

        try:
            rel = str(position.relative_to(root_path.resolve()))
        except ValueError:
            rel = position.name
        entries.append(
            {"position": position, "ch1": match, "ch2": ch2_path, "rel": rel or position.name}
        )
    return entries


class CellposeFolderPanel(StandalonePathsMixin, QWidget):
    """Setup (four name fields) + Find + a one-dot-per-position list + Save.

    Walking a project: fill the input names, *Find data folders…* under a root,
    click a position to load + bind its images, work it in the shared body below,
    then *Save masks to folder*. The host wires :attr:`positionActivated` to its
    load-and-bind seam and :attr:`saveRequested` to its mask-writer, then calls
    :meth:`refresh_statuses` so the dot flips to *saved*.
    """

    positionActivated = Signal(object)  # entry dict | None
    saveRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: dict[str, dict] = {}  # position-str → entry
        self._order: list[str] = []
        self._active: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(self._build_setup_section())

        # Find / Delete row.
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        self.discover_btn = QToolButton()
        self.discover_btn.setText("Find data folders…")
        self.discover_btn.setToolTip(
            "Pick a parent directory; every folder under it holding the Channel 1 "
            "input file is added to the list (run again to add more)."
        )
        self.discover_btn.setStyleSheet(action_button_style())
        self.discover_btn.clicked.connect(self._on_discover)
        actions.addWidget(self.discover_btn)
        self.delete_btn = QToolButton()
        self.delete_btn.setText("Delete selected")
        self.delete_btn.setStyleSheet(action_button_style())
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self._on_delete)
        actions.addWidget(self.delete_btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        # The position list — one colored dot + the folder path per row.
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_widget.currentItemChanged.connect(self._on_current_changed)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget)

        self._count = QLabel("0 data folders")
        self._count.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self._count)

        self.save_btn = QPushButton("Save masks to folder")
        self.save_btn.setToolTip(
            "Write the tracked masks for the selected position to its folder, "
            "under the Channel 1 / Channel 2 output names."
        )
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.saveRequested)
        layout.addWidget(self.save_btn)

        self._load_settings()

    # ------------------------------------------------------------ setup UI
    def _build_setup_section(self) -> CollapsibleSection:
        body = QWidget()
        form = QFormLayout(body)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(2)
        self._fields: dict[str, QLineEdit] = {}
        for key, label, placeholder in (
            ("ch1_input", "Channel 1 input", "e.g. 0_input/nucleus.tif (required)"),
            ("ch2_input", "Channel 2 input", "e.g. 0_input/cell.tif (optional)"),
            ("ch1_output", "Channel 1 output", "e.g. nucleus_labels.tif"),
            ("ch2_output", "Channel 2 output", "e.g. cell_labels.tif"),
        ):
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.setStyleSheet(mono_input_style())
            edit.editingFinished.connect(self._save_settings)
            self._fields[key] = edit
            name = QLabel(label)
            name.setStyleSheet(f"color: {TEXT_MID};")
            form.addRow(name, edit)
        return CollapsibleSection("Data folders", body, expanded=True, title_color=TEXT_MID)

    # --------------------------------------------------------- field access
    def input_names(self) -> tuple[str, str]:
        """(Channel 1, Channel 2) input names — the discovery pattern."""
        return (
            self._fields["ch1_input"].text().strip(),
            self._fields["ch2_input"].text().strip(),
        )

    def output_names(self) -> tuple[str, str]:
        """(Channel 1, Channel 2) output names — the save target."""
        return (
            self._fields["ch1_output"].text().strip(),
            self._fields["ch2_output"].text().strip(),
        )

    def _load_settings(self) -> None:
        self._load_path_settings(_SETTINGS_APP, self._fields)

    def _save_settings(self) -> None:
        self._save_path_settings(_SETTINGS_APP, self._fields)

    # ------------------------------------------------------------ discovery
    def _on_discover(self) -> None:
        ch1_name, ch2_name = self.input_names()
        if not ch1_name:
            self._count.setText("Enter the Channel 1 input name first.")
            return
        root = QFileDialog.getExistingDirectory(self, "Pick a project root to scan")
        if not root:
            return
        found = discover_folders(root, ch1_name, ch2_name)
        added = self._add_entries(found)
        if not found:
            self._count.setText(
                f"No folders with '{ch1_name}' found under {root}."
            )
        elif not added:
            self._count.setText(
                f"All {len(found)} found folders were already listed."
            )
        else:
            self._refresh_count()

    def _add_entries(self, entries: list[dict]) -> int:
        """Append entries (de-duped by position path); returns the number added."""
        added = 0
        for entry in entries:
            key = str(entry["position"])
            if key in self._entries:
                continue
            self._entries[key] = entry
            self._order.append(key)
            item = QListWidgetItem(_dot_icon(self._status_for(key)), entry["rel"])
            item.setData(_ROLE_ENTRY, key)
            self.list_widget.addItem(item)
            added += 1
        return added

    # ------------------------------------------------------------ selection
    def _on_current_changed(self, current, _previous) -> None:
        self._active = current.data(_ROLE_ENTRY) if current is not None else None
        self.save_btn.setEnabled(self._active is not None)
        entry = self._entries.get(self._active) if self._active else None
        self.refresh_statuses()
        self.positionActivated.emit(entry)

    def _on_selection_changed(self) -> None:
        self.delete_btn.setEnabled(bool(self.list_widget.selectedItems()))

    def active_entry(self) -> dict | None:
        return self._entries.get(self._active) if self._active else None

    def _on_delete(self) -> None:
        for item in self.list_widget.selectedItems():
            key = item.data(_ROLE_ENTRY)
            self._entries.pop(key, None)
            if key in self._order:
                self._order.remove(key)
            if key == self._active:
                self._active = None
                self.save_btn.setEnabled(False)
            self.list_widget.takeItem(self.list_widget.row(item))
        self._refresh_count()

    # -------------------------------------------------------------- status
    def _status_for(self, key: str) -> str:
        """One position's dot state from output-file existence + active-ness."""
        entry = self._entries[key]
        ch1_out, _ch2_out = self.output_names()
        if ch1_out and (entry["position"] / ch1_out).is_file():
            return SAVED
        if key == self._active:
            return WORKING
        return MISSING

    def refresh_statuses(self) -> None:
        """Repaint every row's dot (call after a save, or on selection change)."""
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            key = item.data(_ROLE_ENTRY)
            item.setIcon(_dot_icon(self._status_for(key)))

    def _refresh_count(self) -> None:
        n = len(self._order)
        self._count.setText(f"{n} data folder{'s' if n != 1 else ''}")
