"""Shared reusable Qt widgets for the CellFlow napari plugin."""
from __future__ import annotations

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .ui_style import (
    SECTION_MARGIN,
    TIGHT_SPACING,
    TINY_MARGIN,
    icon_button,
    muted_label,
    status_label,
)


class CollapsibleSection(QWidget):
    """A labelled section with a toggle button that shows/hides its inner widget."""

    def __init__(
        self,
        title: str,
        inner: QWidget,
        expanded: bool = False,
        parent: QWidget | None = None,
        title_color: str = "white",
    ) -> None:
        super().__init__(parent)
        self._inner = inner
        self._base_title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, TINY_MARGIN, 0, TINY_MARGIN)
        layout.setSpacing(0)

        # Header toggle button
        self._toggle = QToolButton()
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setText(title)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.setStyleSheet(
            f"QToolButton {{ font-weight: bold; font-size: 10pt; border: none; "
            f"padding: 2px; color: {title_color}; }}"
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
        frame_layout.setContentsMargins(
            SECTION_MARGIN, SECTION_MARGIN, SECTION_MARGIN, SECTION_MARGIN
        )
        frame_layout.setSpacing(TINY_MARGIN)
        frame_layout.addWidget(inner)

        self._content_frame.setVisible(expanded)
        layout.addWidget(self._content_frame)

        # Always Preferred policy — height is driven by scroll area's minimumHeight
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        if expanded:
            QTimer.singleShot(0, self._notify_layout_change)

    def set_title(self, title: str) -> None:
        """Update the header text."""
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

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content_frame.setVisible(checked)
        QTimer.singleShot(0, self._notify_layout_change)

    def _notify_layout_change(self) -> None:
        """Propagate geometry changes up the nested collapsible chain."""
        self.updateGeometry()
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, CollapsibleSection) and parent.is_expanded:
                parent.updateGeometry()
                QTimer.singleShot(0, parent._notify_layout_change)
                return
            parent.updateGeometry()
            parent = parent.parent()


# ---------------------------------------------------------------------------
# Pipeline file status rows
# ---------------------------------------------------------------------------

class _PipelineFileRow(QWidget):
    """One pipeline file status row: icon | rel-path | info | [load btn]"""

    def __init__(self, rel_path: str, display_name: str, loadable: str | None = None):
        super().__init__()
        self._rel_path = rel_path
        self._loadable = loadable
        self._full_path: "Path | None" = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(
            SECTION_MARGIN, TINY_MARGIN, SECTION_MARGIN, TINY_MARGIN
        )
        lay.setSpacing(TIGHT_SPACING)

        self._icon_lbl = QLabel("○")
        self._icon_lbl.setFixedWidth(14)
        self._icon_lbl.setAlignment(Qt.AlignCenter)
        muted_label(self._icon_lbl, size_pt=9)
        lay.addWidget(self._icon_lbl)

        name_lbl = QLabel(rel_path)
        name_lbl.setFixedWidth(200)
        status_label(name_lbl)
        name_lbl.setToolTip(display_name)
        lay.addWidget(name_lbl)

        self._info_lbl = QLabel("—")
        self._info_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        status_label(self._info_lbl)
        lay.addWidget(self._info_lbl)

        if loadable is not None:
            self._load_btn = QPushButton("↑")
            icon_button(self._load_btn, height=18)
            self._load_btn.setToolTip("Load into napari viewer")
            self._load_btn.setEnabled(False)
            lay.addWidget(self._load_btn)
        else:
            self._load_btn = None

    def set_present(self, info_text: str) -> None:
        self._icon_lbl.setText("✓")
        self._icon_lbl.setStyleSheet("font-size: 9pt; font-weight: bold; color: #4CAF50;")
        self._info_lbl.setText(info_text)
        status_label(self._info_lbl)
        if self._load_btn:
            self._load_btn.setEnabled(True)

    def set_missing(self) -> None:
        self._icon_lbl.setText("✗")
        muted_label(self._icon_lbl, size_pt=9)
        self._info_lbl.setText("missing")
        muted_label(self._info_lbl)
        self._full_path = None
        if self._load_btn:
            self._load_btn.setEnabled(False)

    def set_no_project(self) -> None:
        self._icon_lbl.setText("○")
        muted_label(self._icon_lbl, size_pt=9)
        self._info_lbl.setText("—")
        muted_label(self._info_lbl)
        self._full_path = None
        if self._load_btn:
            self._load_btn.setEnabled(False)


def _file_info(path: "Path") -> str:
    """Return a concise shape/dtype string for a pipeline output file."""
    if path.is_dir():
        return "Directory"
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(path) as tif:
                shape = tif.series[0].shape if tif.series else None
            if shape:
                return "×".join(str(d) for d in shape)
        except Exception:
            pass
        return "TIFF"
    if suffix in (".h5", ".hdf5"):
        try:
            import h5py
            shapes = []
            def _collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    shapes.append(f"{name}: " + "×".join(str(d) for d in obj.shape))
            with h5py.File(path, "r") as f:
                f.visititems(_collect)
            if shapes:
                return "; ".join(shapes[:2]) + ("…" if len(shapes) > 2 else "")
        except Exception:
            pass
        kb = path.stat().st_size // 1024
        return f"{kb} KB"
    return f"{path.stat().st_size // 1024} KB"


class PipelineFilesWidget(QWidget):
    """Compact file-status display for pipeline-stage widgets."""

    def __init__(
        self,
        groups: list[tuple[str, list[tuple[str, str]]]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._rows: list[_PipelineFileRow] = []

        for group_label, entries in groups:
            if group_label:
                hdr = QLabel(group_label)
                hdr.setStyleSheet(
                    "font-size: 7pt; font-weight: bold; padding: 1px 4px;"
                    " background: palette(alternateBase); color: palette(mid);"
                )
                lay.addWidget(hdr)
            for rel_path, display_name in entries:
                row = _PipelineFileRow(rel_path, display_name, loadable=None)
                self._rows.append(row)
                lay.addWidget(row)

    def refresh(self, pos_dir: "Path" | None) -> None:
        """Update all rows to reflect current on-disk state."""
        if pos_dir is None:
            for row in self._rows:
                row.set_no_project()
            return
        for row in self._rows:
            full_path = pos_dir / row._rel_path
            if full_path.exists():
                row._full_path = full_path
                row.set_present(_file_info(full_path))
            else:
                row.set_missing()
