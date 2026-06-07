"""Qt view for the extend / swap candidate galleries.

Three :class:`CandidateColumn` blocks are stacked top-to-bottom — extend-backward
· swap · extend-forward — each a titled block whose clickable thumbnails (built
from a :class:`~cellflow.napari._correction_candidates.CandidateStrip`) flow
left→right and wrap onto new rows. The whole strip scrolls as one. Clicking a
thumbnail emits ``candidate_activated(which, key)`` so the controller can apply
that extend/swap; the pixels themselves come from the pure builder, so this stays
a thin view. ``rgb_to_qimage`` is the only import-safe symbol without a running
QApplication.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._correction_candidates import CandidateStrip, CandidateTile
from cellflow.napari._correction_track_accordion import rgb_to_qimage
from cellflow.napari._flow_layout import FlowLayout

_TILE_PX = 64
_TILE_PX_MIN = 20
_TILE_PX_MAX = 256
_ZOOM_STEP = 8       # px added/removed from the tile size per Ctrl+wheel notch

# Draggable handle between the stacked regions. Mirrors the workspace splitter
# that sizes the strip widths (see nucleus_correction_widget), but oriented
# vertically: a mid-grey grip inset against the dark dock background. Qt adds
# the border *on top of* the handle width for vertical handles (it insets it for
# horizontal ones), so to land the same ~10 px total thickness as the workspace
# grabbers we pair a 3 px border with a thin handle width (see _HANDLE_WIDTH).
_HANDLE_STYLE = (
    "QSplitter::handle:vertical {"
    " background: #5a606b;"
    " border-top: 3px solid #2e3440;"
    " border-bottom: 3px solid #2e3440;"
    " }"
    "QSplitter::handle:vertical:hover { background: #7a828f; }"
)
_HANDLE_WIDTH = 2


class _ClickableTile(QFrame):
    """One candidate thumbnail + caption that reports its key when clicked."""

    clicked = Signal(int)

    def __init__(
        self, tile: CandidateTile, *, tile_px: int, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._key = int(tile.key)
        self.setObjectName("candidateTile")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"{tile.caption} — click to apply")
        self.setStyleSheet(
            "QFrame#candidateTile { border: 1px solid transparent; }"
            "QFrame#candidateTile:hover { border: 1px solid #ffd246; }"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(1, 1, 1, 1)
        lay.setSpacing(0)

        pixmap = QPixmap.fromImage(rgb_to_qimage(tile.rgb)).scaledToHeight(
            tile_px, Qt.FastTransformation
        )
        image = QLabel()
        image.setPixmap(pixmap)
        image.setAlignment(Qt.AlignCenter)
        lay.addWidget(image)

    @property
    def key(self) -> int:
        return self._key

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit(self._key)


class CandidateColumn(QWidget):
    """A titled block whose candidate thumbnails flow left→right and wrap."""

    candidate_clicked = Signal(int)  # key

    def __init__(
        self, title: str = "", parent: QWidget | None = None, *, tile_px: int = _TILE_PX
    ) -> None:
        super().__init__(parent)
        self._tile_px = int(tile_px)
        self._title_text = title
        self._strip = CandidateStrip()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self._title = QLabel(title)
        self._title.setContentsMargins(4, 2, 4, 2)
        outer.addWidget(self._title)

        self._body = QWidget()
        self._body_lay = FlowLayout(self._body, margin=1, h_spacing=1, v_spacing=1)
        outer.addWidget(self._body)
        # Pin content to the top so a region dragged taller than its tiles
        # leaves empty space at the bottom rather than floating them centred.
        outer.addStretch(1)

    def set_strip(self, strip: CandidateStrip, *, title: str | None = None) -> None:
        """Replace the column's thumbnails (an empty strip shows a placeholder)."""
        self._strip = strip
        if title is not None:
            self._title_text = title
        self._rebuild()

    def set_tile_size(self, tile_px: int) -> None:
        """Change the thumbnail size and re-render the column's tiles."""
        tile_px = max(_TILE_PX_MIN, min(int(tile_px), _TILE_PX_MAX))
        if tile_px == self._tile_px:
            return
        self._tile_px = tile_px
        self._rebuild()

    def clear(self) -> None:
        self.set_strip(CandidateStrip())

    def tiles(self) -> list[_ClickableTile]:
        """The live tile widgets, for tests and hit-testing."""
        out: list[_ClickableTile] = []
        for i in range(self._body_lay.count()):
            widget = self._body_lay.itemAt(i).widget()
            if isinstance(widget, _ClickableTile):
                out.append(widget)
        return out

    def _clear_body(self) -> None:
        for i in reversed(range(self._body_lay.count())):
            item = self._body_lay.takeAt(i)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()

    def _rebuild(self) -> None:
        self._clear_body()
        n = len(self._strip.tiles)
        self._title.setText(
            f"{self._title_text} ({n})" if self._title_text else str(n)
        )
        if n == 0:
            placeholder = QLabel("—")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color: #707070;")
            self._body_lay.addWidget(placeholder)
            return
        for tile in self._strip.tiles:
            widget = _ClickableTile(tile, tile_px=self._tile_px, parent=self._body)
            widget.clicked.connect(self.candidate_clicked)
            self._body_lay.addWidget(widget)


class CandidateGalleryPanel(QWidget):
    """Three stacked candidate blocks: extend-backward · swap · extend-forward.

    The blocks stack top-to-bottom inside a vertical :class:`QSplitter`, so a
    draggable handle between each pair resizes the regions (the same grip style
    as the workspace strip-width splitter). Each block scrolls independently
    when its thumbnails overflow the height it has been given, and wraps them
    onto new rows (see :class:`CandidateColumn`).
    """

    EXTEND_BACKWARD = "extend_backward"
    SWAP = "swap"
    EXTEND_FORWARD = "extend_forward"

    # (which, key) — which column, and the candidate's routing key.
    candidate_activated = Signal(str, int)

    def __init__(self, parent: QWidget | None = None, *, tile_px: int = _TILE_PX) -> None:
        super().__init__(parent)
        self._tile_px = int(tile_px)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Vertical, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(_HANDLE_WIDTH)
        splitter.setStyleSheet(_HANDLE_STYLE)

        self._columns: dict[str, CandidateColumn] = {}
        for which, title in (
            (self.EXTEND_BACKWARD, "◀ Extend"),
            (self.SWAP, "Swap"),
            (self.EXTEND_FORWARD, "Extend ▶"),
        ):
            column = CandidateColumn(title, tile_px=tile_px)
            column.candidate_clicked.connect(
                lambda key, _which=which: self.candidate_activated.emit(_which, int(key))
            )
            self._columns[which] = column

            pane = _GalleryScroll(self)
            pane.setWidgetResizable(True)
            pane.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            pane.setWidget(column)
            splitter.addWidget(pane)

        # Start the three regions at equal height; the user drags from there.
        splitter.setSizes([1, 1, 1])
        self._splitter = splitter
        outer.addWidget(splitter)

    def column(self, which: str) -> CandidateColumn:
        return self._columns[which]

    def set_column(self, which: str, strip: CandidateStrip) -> None:
        self._columns[which].set_strip(strip)

    def set_tile_size(self, tile_px: int) -> None:
        """Resize every column's thumbnails (Ctrl+wheel zoom, like the film strip)."""
        tile_px = max(_TILE_PX_MIN, min(int(tile_px), _TILE_PX_MAX))
        if tile_px == self._tile_px:
            return
        self._tile_px = tile_px
        for column in self._columns.values():
            column.set_tile_size(tile_px)

    def clear(self) -> None:
        for column in self._columns.values():
            column.clear()


class _GalleryScroll(QScrollArea):
    """Per-region scroll area whose Ctrl+wheel zooms the gallery thumbnails."""

    def __init__(self, panel: CandidateGalleryPanel) -> None:
        super().__init__(panel)
        self._panel = panel

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.modifiers() & Qt.ControlModifier:
            step = _ZOOM_STEP if event.angleDelta().y() > 0 else -_ZOOM_STEP
            self._panel.set_tile_size(self._panel._tile_px + step)
            event.accept()
            return
        super().wheelEvent(event)


__all__ = ["CandidateColumn", "CandidateGalleryPanel"]
