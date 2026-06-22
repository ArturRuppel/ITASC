"""Curation tool: author the exclusion table by eye, over the contact display.

The image-linked judgement ("scrub through a position, see a bad frame, exclude
it with a note") is what re-earns napari for Aggregate Quantification. This
plugin embeds the contact-visualization display for one selected position and
turns exclude actions into rows of the curation CSV — via the Qt-free
:class:`~cellflow.napari.aggregate_quantification.curation_controller.CurationController`,
which auto-saves so the table is always the source of truth. No plots live here
(Iris owns plotting).
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification.shape_tables import catalogue_root
from cellflow.napari.aggregate_quantification.curation_controller import (
    CurationController,
)
from cellflow.napari.aggregate_quantification.plugins import (
    AnalysisContext,
    AnalysisPlugin,
)
from cellflow.napari.ui_style import action_button, status_label

#: Qt.UserRole — stores the row's frame (int) or None (whole position) on a list item.
_FRAME_ROLE = 256


class CurationWidget(AnalysisPlugin):
    """Mark a frame / a whole position excluded with a reason, over the image."""

    plugin_id = "curation"
    display_name = "Curation"

    def __init__(self, viewer=None, parent: QWidget | None = None) -> None:
        super().__init__(viewer=viewer, parent=parent)
        from cellflow.napari.aggregate_quantification_widget import (
            AggregateQuantificationWidget,
        )

        self._experiment_id: str | None = None
        self._position_id: str | None = None
        self._controller: CurationController | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self._scope_lbl = QLabel("Select a single position to curate.")
        self._scope_lbl.setWordWrap(True)
        layout.addWidget(self._scope_lbl)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("curation.csv (defaults beside the tables)…")
        self._path_edit.editingFinished.connect(self._on_path_edited)
        layout.addLayout(self._labelled_row("Curation file:", self._path_edit))

        self._view = AggregateQuantificationWidget(viewer=viewer, standalone=False)
        self._view.pipeline_files_header.setVisible(False)
        self._view._pipeline_files_section.setVisible(False)
        layout.addWidget(self._view, 1)

        self._reason_edit = QLineEdit()
        self._reason_edit.setPlaceholderText("Reason (required)…")
        self._reason_edit.textChanged.connect(self._update_enabled)
        layout.addLayout(self._labelled_row("Reason:", self._reason_edit))

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(4)
        self._exclude_frame_btn = QPushButton("Exclude this frame")
        action_button(self._exclude_frame_btn, expand=True)
        self._exclude_frame_btn.clicked.connect(self._on_exclude_frame)
        self._exclude_position_btn = QPushButton("Exclude this position")
        action_button(self._exclude_position_btn, expand=True)
        self._exclude_position_btn.clicked.connect(self._on_exclude_position)
        actions.addWidget(self._exclude_frame_btn)
        actions.addWidget(self._exclude_position_btn)
        layout.addLayout(actions)

        layout.addWidget(QLabel("Exclusions for this position:"))
        self._exclusions_list = QListWidget()
        self._exclusions_list.setMaximumHeight(120)
        layout.addWidget(self._exclusions_list)
        self._remove_btn = QPushButton("Remove selected")
        action_button(self._remove_btn)
        self._remove_btn.clicked.connect(self._on_remove_selected)
        layout.addWidget(self._remove_btn)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        status_label(self._status_lbl)
        layout.addWidget(self._status_lbl)

        self._update_enabled()

    @staticmethod
    def _labelled_row(label: str, edit: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        lbl = QLabel(label)
        lbl.setFixedWidth(90)
        row.addWidget(lbl)
        row.addWidget(edit, 1)
        return row

    def _current_frame(self) -> int:
        dims = getattr(self.viewer, "dims", None)
        step = getattr(dims, "current_step", (0,))
        return int(step[0]) if step else 0

    def _has_single_position(self) -> bool:
        return self._position_id is not None and self._controller is not None

    def _reason(self) -> str:
        return self._reason_edit.text().strip()

    def set_context(self, ctx: AnalysisContext) -> None:
        if ctx.viewer is not None:
            self.viewer = ctx.viewer
        records = list(ctx.records)
        if len(records) == 1:
            record = records[0]
            self._experiment_id = str(record.get("experiment_id", ""))
            self._position_id = str(record.get("id", ""))
            self._ensure_controller(records)
            self._update_display(record)
            self._scope_lbl.setText(f"Curating position: {self._position_id}")
        else:
            self._experiment_id = None
            self._position_id = None
            self._controller = None
            self._clear_display()
            self._scope_lbl.setText(
                "Select a single position to curate."
                if not records
                else f"{len(records)} positions selected — pick exactly one to curate."
            )
        self._refresh_exclusions()
        self._update_enabled()

    def _ensure_controller(self, records: list[dict]) -> None:
        text = self._path_edit.text().strip()
        if text:
            path = Path(text)
        else:
            path = catalogue_root(records) / "curation.csv"
            self._path_edit.setText(str(path))
        self._controller = CurationController(path)

    def _on_path_edited(self) -> None:
        text = self._path_edit.text().strip()
        if text:
            self._controller = CurationController(Path(text))
            self._refresh_exclusions()
            self._update_enabled()

    def _update_display(self, record: dict) -> None:
        self._view.set_context(
            cell_labels=record.get("cell_tracked_labels_path"),
            nucleus_labels=record.get("nucleus_tracked_labels_path"),
            out_path=record.get("contact_analysis_path"),
            status_root=record.get("position_path"),
        )

    def _clear_display(self) -> None:
        self._view.set_context(
            cell_labels=None, nucleus_labels=None, out_path=None, status_root=None
        )

    def _update_enabled(self) -> None:
        can_act = self._has_single_position() and bool(self._reason())
        self._exclude_frame_btn.setEnabled(can_act)
        self._exclude_position_btn.setEnabled(can_act)
        self._remove_btn.setEnabled(self._has_single_position())

    def _on_exclude_frame(self) -> None:
        if not self._has_single_position() or not self._reason():
            return
        frame = self._current_frame()
        self._controller.exclude_frame(
            experiment_id=self._experiment_id,
            position_id=self._position_id,
            frame=frame,
            reason=self._reason(),
        )
        self._status_lbl.setText(f"Status: excluded frame {frame}.")
        self._refresh_exclusions()

    def _on_exclude_position(self) -> None:
        if not self._has_single_position() or not self._reason():
            return
        self._controller.exclude_position(
            experiment_id=self._experiment_id,
            position_id=self._position_id,
            reason=self._reason(),
        )
        self._status_lbl.setText("Status: excluded whole position.")
        self._refresh_exclusions()

    def _on_remove_selected(self) -> None:
        if not self._has_single_position():
            return
        item = self._exclusions_list.currentItem()
        if item is None:
            return
        frame = item.data(_FRAME_ROLE)
        self._controller.remove(
            experiment_id=self._experiment_id,
            position_id=self._position_id,
            frame=frame,
        )
        self._status_lbl.setText("Status: removed exclusion.")
        self._refresh_exclusions()

    def _refresh_exclusions(self) -> None:
        self._exclusions_list.clear()
        if not self._has_single_position():
            return
        rows = self._controller.exclusions_for(
            experiment_id=self._experiment_id, position_id=self._position_id
        )
        import pandas as pd

        for _, row in rows.iterrows():
            frame = row["frame"]
            if pd.isna(frame):
                text = f"whole position — {row['exclusion_reason']}"
                payload = None
            else:
                text = f"frame {int(frame)} — {row['exclusion_reason']}"
                payload = int(frame)
            item = QListWidgetItem(text)
            item.setData(_FRAME_ROLE, payload)
            self._exclusions_list.addItem(item)
