"""Qt view for the extend / swap candidate galleries.

Three :class:`CandidateColumn`s sit side by side — extend-backward · swap ·
extend-forward — each a titled, scrollable stack of clickable thumbnails built
from a :class:`~cellflow.napari._correction_candidates.CandidateStrip`. Clicking
a thumbnail emits ``candidate_activated(which, key)`` so the controller can apply
that extend/swap; the pixels themselves come from the pure builder, so this stays
a thin view. ``rgb_to_qimage`` is the only import-safe symbol without a running
QApplication.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._correction_candidates import CandidateStrip, CandidateTile
from cellflow.napari._correction_film_strip import rgb_to_qimage

_TILE_PX = 72


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
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(1)

        pixmap = QPixmap.fromImage(rgb_to_qimage(tile.rgb)).scaledToHeight(
            tile_px, Qt.FastTransformation
        )
        image = QLabel()
        image.setPixmap(pixmap)
        image.setAlignment(Qt.AlignCenter)
        lay.addWidget(image)

        caption = QLabel(tile.caption)
        caption.setAlignment(Qt.AlignCenter)
        caption.setStyleSheet("color: #b0b0b0; font-size: 9px;")
        lay.addWidget(caption)

    @property
    def key(self) -> int:
        return self._key

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit(self._key)


class CandidateColumn(QWidget):
    """A titled, scrollable vertical stack of clickable candidate thumbnails."""

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

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._body = QWidget()
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(2, 2, 2, 2)
        self._body_lay.setSpacing(4)
        self._body_lay.addStretch(1)
        self._scroll.setWidget(self._body)
        outer.addWidget(self._scroll)

    def set_strip(self, strip: CandidateStrip, *, title: str | None = None) -> None:
        """Replace the column's thumbnails (an empty strip shows a placeholder)."""
        self._strip = strip
        if title is not None:
            self._title_text = title
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
        # Drop every widget but keep the trailing stretch item.
        for i in reversed(range(self._body_lay.count())):
            item = self._body_lay.itemAt(i)
            widget = item.widget()
            if widget is not None:
                self._body_lay.takeAt(i)
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
            self._body_lay.insertWidget(0, placeholder)
            return
        for tile in self._strip.tiles:
            widget = _ClickableTile(tile, tile_px=self._tile_px, parent=self._body)
            widget.clicked.connect(self.candidate_clicked)
            self._body_lay.insertWidget(self._body_lay.count() - 1, widget)


class CandidateGalleryPanel(QWidget):
    """Three candidate columns: extend-backward · swap · extend-forward."""

    EXTEND_BACKWARD = "extend_backward"
    SWAP = "swap"
    EXTEND_FORWARD = "extend_forward"

    # (which, key) — which column, and the candidate's routing key.
    candidate_activated = Signal(str, int)

    def __init__(self, parent: QWidget | None = None, *, tile_px: int = _TILE_PX) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

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
            row.addWidget(column)

    def column(self, which: str) -> CandidateColumn:
        return self._columns[which]

    def set_column(self, which: str, strip: CandidateStrip) -> None:
        self._columns[which].set_strip(strip)

    def clear(self) -> None:
        for column in self._columns.values():
            column.clear()


__all__ = ["CandidateColumn", "CandidateGalleryPanel"]
