"""Napari widget for patching a contact-analysis H5 with NLS classifications."""
from __future__ import annotations

from pathlib import Path

from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow_utils.contact_analysis.nls_classification import (
    NLSClassificationSummary,
    patch_position_contact_analysis_nls_classes,
)
from cellflow.napari.ui_style import action_button, status_label


class NLSClassificationWidget(QWidget):
    """Run NLS-high/NLS-low classification for a chosen contact-analysis H5 file."""

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._classify_worker = None
        self._classify_completion_pending = False
        self._classify_error_pending = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)
        layout.addLayout(form)

        self.h5_edit = QLineEdit()
        self.h5_browse_btn = QPushButton("Browse...")
        form.addRow("Contact analysis H5:", self._path_row(self.h5_edit, self.h5_browse_btn))

        self.nls_edit = QLineEdit()
        self.nls_browse_btn = QPushButton("Browse...")
        form.addRow("NLS image:", self._path_row(self.nls_edit, self.nls_browse_btn))

        self.labels_edit = QLineEdit()
        self.labels_browse_btn = QPushButton("Browse...")
        form.addRow("Nucleus labels:", self._path_row(self.labels_edit, self.labels_browse_btn))

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        status_label(self.status_lbl)
        layout.addWidget(self.status_lbl)

        self.classify_btn = QPushButton("Classify NLS Tracks")
        action_button(self.classify_btn, expand=True)
        layout.addWidget(self.classify_btn)

        layout.addStretch()

        self.h5_browse_btn.clicked.connect(self._browse_h5)
        self.nls_browse_btn.clicked.connect(self._browse_nls)
        self.labels_browse_btn.clicked.connect(self._browse_labels)
        self.classify_btn.clicked.connect(self._on_classify)

        for field in (self.h5_edit, self.nls_edit, self.labels_edit):
            field.textChanged.connect(self._update_status)

        self._update_status()

    @property
    def contact_analysis_path(self) -> Path | None:
        return self._path_or_none(self.h5_edit)

    @property
    def nls_zavg_path(self) -> Path | None:
        return self._path_or_none(self.nls_edit)

    @property
    def nucleus_labels_path(self) -> Path | None:
        return self._path_or_none(self.labels_edit)

    def _path_or_none(self, edit: QLineEdit) -> Path | None:
        text = edit.text().strip()
        return Path(text) if text else None

    def _inputs_ready(self) -> bool:
        return all(
            path is not None and path.exists()
            for path in (
                self.contact_analysis_path,
                self.nls_zavg_path,
                self.nucleus_labels_path,
            )
        )

    def _update_status(self) -> None:
        self._update_action_states()
        missing = []
        if self.contact_analysis_path is None or not self.contact_analysis_path.exists():
            missing.append("contact analysis")
        if self.nls_zavg_path is None or not self.nls_zavg_path.exists():
            missing.append("NLS image")
        if self.nucleus_labels_path is None or not self.nucleus_labels_path.exists():
            missing.append("nucleus labels")
        if missing:
            self._set_status(f"Status: missing {', '.join(missing)}.")
            return
        if not self.status_lbl.text() or self.status_lbl.text().startswith("Status: missing"):
            self._set_status("Status: ready.")

    def _update_action_states(self) -> None:
        running = self._classify_worker is not None
        self.classify_btn.setEnabled(self._inputs_ready() and not running)

    def _set_classify_running(self, running: bool) -> None:
        self._update_action_states()

    def _set_status(self, message: str) -> None:
        self.status_lbl.setText(message)

    def _on_classify_done(self, summary: NLSClassificationSummary) -> None:
        self._classify_completion_pending = True
        self._classify_worker = None
        self._set_classify_running(False)
        self._set_status(
            "Status: classified "
            f"{summary.track_count} tracks "
            f"(high={summary.high_track_count}, low={summary.low_track_count}) "
            f"into {summary.h5_path.name}"
        )
        self._update_action_states()

    def _on_classify_error(self, exc: Exception) -> None:
        self._classify_error_pending = True
        self._classify_worker = None
        self._set_classify_running(False)
        self._set_status(f"Status: error: {exc}")
        self._update_action_states()

    def _on_classify(self) -> None:
        if not self._inputs_ready():
            self._update_status()
            return

        contact_analysis_path = self.contact_analysis_path
        nls_zavg_path = self.nls_zavg_path
        nucleus_labels_path = self.nucleus_labels_path
        if contact_analysis_path is None or nls_zavg_path is None or nucleus_labels_path is None:
            self._update_status()
            return

        self._classify_completion_pending = False
        self._classify_error_pending = False
        self._set_status("Status: classifying NLS tracks...")
        self._classify_worker = object()
        self._set_classify_running(True)

        @thread_worker(
            connect={
                "returned": self._on_classify_done,
                "errored": self._on_classify_error,
            }
        )
        def _worker():
            return patch_position_contact_analysis_nls_classes(
                contact_analysis_path,
                nls_zavg_path,
                nucleus_labels_path,
            )

        worker = _worker()
        self._classify_worker = worker
        if self._classify_completion_pending or self._classify_error_pending:
            self._classify_worker = None
            self._classify_completion_pending = False
            self._classify_error_pending = False
            self._update_action_states()

    def _browse_h5(self) -> None:
        path = QFileDialog.getOpenFileName(
            self,
            "Select Contact Analysis H5",
            self.h5_edit.text(),
            "HDF5 (*.h5 *.hdf5);;All Files (*)",
        )[0]
        if path:
            self.h5_edit.setText(path)

    def _browse_nls(self) -> None:
        path = QFileDialog.getOpenFileName(
            self,
            "Select NLS Image",
            self.nls_edit.text(),
            "TIFF (*.tif *.tiff);;All Files (*)",
        )[0]
        if path:
            self.nls_edit.setText(path)

    def _browse_labels(self) -> None:
        path = QFileDialog.getOpenFileName(
            self,
            "Select Nucleus Tracked Labels",
            self.labels_edit.text(),
            "TIFF (*.tif *.tiff);;All Files (*)",
        )[0]
        if path:
            self.labels_edit.setText(path)

    @staticmethod
    def _path_row(line_edit: QLineEdit, button: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return row
