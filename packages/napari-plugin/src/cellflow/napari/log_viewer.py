"""StageLogViewer — collapsible widget showing the last N lines of pipeline.log.

Embed this at the bottom of any stage widget to give users live visibility into
what the running stage is writing to ``pipeline.log``.

Usage::

    from .log_viewer import StageLogViewer
    from .registry import get_state

    self._log_viewer = StageLogViewer(get_state(viewer))
    layout.addWidget(self._log_viewer)
    # After a worker finishes, call:
    self._log_viewer.refresh()
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .registry import ViewerState


_MAX_LINES = 100  # keep last N lines visible


class StageLogViewer(QWidget):
    """Collapsible widget that displays the last ``max_lines`` of ``pipeline.log``.

    The widget subscribes to ``ViewerState.pipeline_schema_changed`` to
    automatically reload when the project directory changes.  After a stage
    worker finishes, call :meth:`refresh` to pull the latest entries.
    """

    def __init__(
        self,
        state: ViewerState,
        parent: Optional[QWidget] = None,
        max_lines: int = _MAX_LINES,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._max_lines = max_lines
        self._log_path: Optional[Path] = None
        self._expanded = False

        self._build_ui()
        self._connect_signals()
        self._on_project_changed()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(2)
        self.setLayout(outer)

        # ── Header row ────────────────────────────────────────────────
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(4)

        self._toggle_btn = QPushButton("▶ Pipeline Log")
        self._toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; border: none; "
            "font-weight: bold; color: palette(text); }"
            "QPushButton:hover { color: palette(highlight); }"
        )
        self._toggle_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._toggle_btn.clicked.connect(self._toggle)
        header_row.addWidget(self._toggle_btn)

        self._refresh_btn = QPushButton("↺")
        self._refresh_btn.setFixedWidth(26)
        self._refresh_btn.setToolTip("Reload pipeline.log from disk")
        self._refresh_btn.clicked.connect(self.refresh)
        header_row.addWidget(self._refresh_btn)

        self._log_path_label = QLabel("")
        self._log_path_label.setStyleSheet("font-size: 8pt; color: palette(mid);")
        self._log_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header_row.addWidget(self._log_path_label)

        outer.addLayout(header_row)

        # ── Collapsible log area ──────────────────────────────────────
        self._log_area = QGroupBox()
        self._log_area.setFlat(True)
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(0, 2, 0, 0)
        log_layout.setSpacing(0)

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setMinimumHeight(120)
        self._text_edit.setMaximumHeight(220)
        self._text_edit.setFont(
            __import__("qtpy.QtGui", fromlist=["QFont"]).QFont("Monospace", 8)
        )
        self._text_edit.setStyleSheet(
            "QTextEdit { background: palette(base); border: 1px solid palette(mid); "
            "border-radius: 3px; }"
        )
        log_layout.addWidget(self._text_edit)
        self._log_area.setLayout(log_layout)
        self._log_area.setVisible(False)
        outer.addWidget(self._log_area)

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        self._state.pipeline_schema_changed.connect(self._on_project_changed)

    def _on_project_changed(self) -> None:
        project_dir = self._state.project_dir
        if project_dir is None:
            self._log_path = None
            self._log_path_label.setText("")
            if self._expanded:
                self._text_edit.setPlainText("")
            return

        # pos 0 log
        self._log_path = project_dir / "pos00" / "pipeline.log"
        self._log_path_label.setText(
            str(self._log_path.relative_to(project_dir))
        )
        if self._expanded:
            self.refresh()

    # ------------------------------------------------------------------
    # Toggle
    # ------------------------------------------------------------------

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        arrow = "▼" if self._expanded else "▶"
        self._toggle_btn.setText(f"{arrow} Pipeline Log")
        self._log_area.setVisible(self._expanded)
        if self._expanded:
            self.refresh()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Read the last ``max_lines`` entries from ``pipeline.log`` and display them."""
        if not self._expanded:
            return

        if self._log_path is None or not self._log_path.exists():
            self._text_edit.setPlainText("(pipeline.log not found)")
            return

        try:
            lines = self._log_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            self._text_edit.setPlainText(f"Error reading log: {exc}")
            return

        tail = lines[-self._max_lines :]
        rendered = []
        for raw in tail:
            try:
                entry = json.loads(raw)
                ts = entry.get("ts", "")[:19]  # YYYY-MM-DDTHH:MM:SS
                stage = entry.get("stage", "")
                level = entry.get("level", "")
                msg = entry.get("message", "")
                rendered.append(f"[{ts}] {stage} {level}: {msg}")
            except (json.JSONDecodeError, ValueError):
                rendered.append(raw)

        self._text_edit.setPlainText("\n".join(rendered))
        # Scroll to bottom
        sb = self._text_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_log_path(self, path: Optional[Path]) -> None:
        """Override the log file path (e.g. when project_dir is not set)."""
        self._log_path = path
        if path is not None:
            self._log_path_label.setText(str(path))
        else:
            self._log_path_label.setText("")
        if self._expanded:
            self.refresh()
