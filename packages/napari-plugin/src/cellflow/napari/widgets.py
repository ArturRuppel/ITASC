"""Shared reusable Qt widgets for the CellFlow napari plugin."""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFrame, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """A labelled section with a toggle button that shows/hides its inner widget.

    The header is a :class:`QToolButton` with a right/down arrow. Clicking it
    expands or collapses ``inner``. When expanded, the content is surrounded
    by a white border frame.

    Parameters
    ----------
    title:
        Text shown in the header button.
    inner:
        The widget to show/hide.
    expanded:
        Whether the section starts expanded (default ``False``).
    """

    def __init__(
        self,
        title: str,
        inner: QWidget,
        expanded: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._inner = inner
        self._base_title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
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

        # White-bordered frame that wraps inner content when expanded
        self._content_frame = QFrame()
        self._content_frame.setObjectName("collapsible_content")
        self._content_frame.setFrameShape(QFrame.NoFrame)
        self._content_frame.setStyleSheet(
            "QFrame#collapsible_content { border: 1px solid #666666; "
            "border-radius: 4px; margin: 0px 2px 2px 2px; }"
        )
        frame_layout = QVBoxLayout(self._content_frame)
        frame_layout.setContentsMargins(4, 4, 4, 4)
        frame_layout.setSpacing(0)
        frame_layout.addWidget(inner)
        self._content_frame.setVisible(expanded)
        layout.addWidget(self._content_frame)

        # Expanding policy when open so the section claims available vertical space
        # (and its internal scroll areas can grow); Preferred when closed so it
        # stays at toggle-button height.
        v_policy = QSizePolicy.Expanding if expanded else QSizePolicy.Preferred
        self.setSizePolicy(QSizePolicy.Preferred, v_policy)

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
        self._content_frame.setVisible(checked)
        v_policy = QSizePolicy.Expanding if checked else QSizePolicy.Preferred
        self.setSizePolicy(QSizePolicy.Preferred, v_policy)
