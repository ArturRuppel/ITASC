"""Napari widget for patching position contact-analysis files with NLS classifications."""
from __future__ import annotations

from pathlib import Path

from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from cellflow.contact_analysis.nls_classification import (
    NLSClassificationSummary,
    patch_position_contact_analysis_nls_classes,
)
from cellflow.napari.ui_style import action_button, status_label


class NLSClassificationWidget(QWidget):
    """Run NLS-high/NLS-low classification for the active position contact-analysis file."""

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._classify_worker = None
        self._classify_completion_pending = False
        self._classify_error_pending = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        status_label(self.status_lbl)
        layout.addWidget(self.status_lbl)

        self.classify_btn = QPushButton("Classify NLS Tracks")
        action_button(self.classify_btn, expand=True)
        layout.addWidget(self.classify_btn)

        layout.addStretch()

        self.classify_btn.clicked.connect(self._on_classify)
        self.refresh(None)

    @property
    def nls_zavg_path(self) -> Path | None:
        return self._pos_dir / "0_input" / "NLS_zavg.tif" if self._pos_dir else None

    @property
    def nucleus_labels_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    @property
    def contact_analysis_path(self) -> Path | None:
        return self._pos_dir / "4_contact_analysis" / "contact_analysis.h5" if self._pos_dir else None

    def refresh(self, pos_dir: Path | str | None) -> None:
        self._pos_dir = Path(pos_dir) if pos_dir is not None else None
        self._update_status()

    def _inputs_ready(self) -> bool:
        return (
            self.nls_zavg_path is not None
            and self.nls_zavg_path.exists()
            and self.nucleus_labels_path is not None
            and self.nucleus_labels_path.exists()
            and self.contact_analysis_path is not None
            and self.contact_analysis_path.exists()
        )

    def _update_status(self) -> None:
        self._update_action_states()
        if self._pos_dir is None:
            self._set_status("Status: no project open.")
            return
        if not self._inputs_ready():
            missing = []
            if self.nls_zavg_path is None or not self.nls_zavg_path.exists():
                missing.append("NLS image")
            if self.nucleus_labels_path is None or not self.nucleus_labels_path.exists():
                missing.append("nucleus labels")
            if self.contact_analysis_path is None or not self.contact_analysis_path.exists():
                missing.append("contact analysis")
            self._set_status(f"Status: missing {', '.join(missing)}.")
            return
        if not self.status_lbl.text() or self.status_lbl.text().startswith("Status: missing"):
            self._set_status("Status: ready.")

    def _update_action_states(self) -> None:
        running = self._classify_worker is not None
        self.classify_btn.setEnabled(self._inputs_ready() and not running)

    def _set_classify_running(self, running: bool) -> None:
        self._update_action_states()
        if not running:
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
