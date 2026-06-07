"""Shared input/output path-picker scaffolding for standalone napari pieces.

The independently-installable CellFlow pieces (``cellflow-tracking``,
``cellflow-segmentation``) each expose a standalone seam: when run without the
orchestrator they show their own labeled file/dir pickers instead of being
driven through ``refresh(pos_dir)``. This mixin holds the parts that are
identical across those seams — the labeled line-edit + *Browse* row builder, the
file/dir browse handlers, and QSettings persistence — leaving each piece to
declare its own fields and implement its piece-specific "apply these paths to my
workspace" step.

Ships in ``cellflow-core`` (qtpy-only, no CellFlow dependencies) so both the
tracking and segmentation distributions compose on it.
"""
from __future__ import annotations

from qtpy.QtCore import QSettings
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

#: Default file filter for the image inputs the pieces consume.
IMAGE_FILTER = "Images (*.tif *.tiff);;All files (*)"


class StandalonePathsMixin:
    """Reusable path-picker helpers for a standalone workflow widget.

    Mixed into a ``QWidget`` subclass: ``self`` is the dialog parent.
    """

    def _add_path_row(
        self,
        column: QVBoxLayout,
        label: str,
        placeholder: str,
        on_browse,
        on_edited,
    ) -> QLineEdit:
        """Build a ``[label] [line edit] [Browse…]`` row and return the edit."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setMinimumWidth(80)
        row.addWidget(lbl)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.editingFinished.connect(on_edited)
        row.addWidget(edit, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(on_browse)
        row.addWidget(browse_btn)
        column.addLayout(row)
        return edit

    def _browse_file_into(
        self, edit: QLineEdit, title: str, on_done, *, file_filter: str = IMAGE_FILTER
    ) -> None:
        """Pick a file into ``edit`` and run ``on_done`` if one was chosen."""
        path, _ = QFileDialog.getOpenFileName(self, title, filter=file_filter)
        if path:
            edit.setText(path)
            on_done()

    def _browse_dir_into(self, edit: QLineEdit, title: str, on_done) -> None:
        """Pick a directory into ``edit`` and run ``on_done`` if one was chosen."""
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            edit.setText(path)
            on_done()

    def _load_path_settings(self, app: str, fields: dict[str, QLineEdit]) -> None:
        """Populate ``fields`` (key → edit) from ``QSettings('cellflow', app)``."""
        s = QSettings("cellflow", app)
        for key, edit in fields.items():
            value = s.value(key, "", type=str)
            if value:
                edit.setText(value)

    def _save_path_settings(self, app: str, fields: dict[str, QLineEdit]) -> None:
        """Persist ``fields`` (key → edit) into ``QSettings('cellflow', app)``."""
        s = QSettings("cellflow", app)
        for key, edit in fields.items():
            s.setValue(key, edit.text().strip())
