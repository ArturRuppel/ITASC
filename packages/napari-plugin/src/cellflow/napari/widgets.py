"""Shared reusable Qt widgets for the CellFlow napari plugin."""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFrame, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """A labelled section with a toggle button that shows/hides its inner widget.

    The header is a :class:`QToolButton` with a right/down arrow. Clicking it
    expands or collapses ``inner``. A thin separator line sits below the header.

    Parameters
    ----------
    title:
        Text shown in the header button.
    inner:
        The widget to show/hide.
    expanded:
        Whether the section starts expanded (default ``True``).
    """

    def __init__(
        self,
        title: str,
        inner: QWidget,
        expanded: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._inner = inner
        self._base_title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header toggle button
        self._toggle = QToolButton()
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setText(title)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.setStyleSheet(
            "QToolButton { font-weight: bold; font-size: 10pt; border: none; "
            "padding: 2px; color: white; }"
        )
        self._toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self._toggle)

        # Thin separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        layout.addWidget(inner)
        inner.setVisible(expanded)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_title(self, title: str) -> None:
        """Update the header text (e.g. to append a status badge)."""
        self._base_title = title
        self._toggle.setText(title)

    @property
    def title(self) -> str:
        return self._base_title

    @property
    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def expand(self) -> None:
        self._toggle.setChecked(True)

    def collapse(self) -> None:
        self._toggle.setChecked(False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._inner.setVisible(checked)
