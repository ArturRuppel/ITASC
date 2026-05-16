"""Shared reusable Qt widgets for the CellFlow napari plugin."""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor
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
    semantic_color,
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
        title_color: str | None = None,
        title_role: str = "stage",
        title_level: int = 1,
        accent_color: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._inner = inner
        self._base_title = title
        if title_color is None:
            title_color = semantic_color(title_role, title_level)
        self._default_title_color = title_color
        # An explicit accent_color marks this as the OUTER stage anchor: stripe
        # is thicker and the header text uses the full accent hue. Inner sections
        # leave accent_color=None and inherit a muted variant via parent walk.
        self._explicit_accent: str | None = accent_color
        self._effective_accent: str | None = accent_color
        self._is_outer_accent: bool = accent_color is not None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, TINY_MARGIN, 0, TINY_MARGIN)
        layout.setSpacing(0)

        # Header toggle button
        self._toggle = QToolButton()
        self._toggle.setObjectName("collapsible_toggle")
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setText(self._qt_display_text(title))
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self._toggle)

        self._content_frame = QFrame()
        self._content_frame.setObjectName("collapsible_content")
        self._content_frame.setFrameShape(QFrame.NoFrame)
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

        self._apply_accent_styles()
        # If no explicit accent, defer a parent walk to inherit one if present.
        if self._explicit_accent is None:
            QTimer.singleShot(0, self._maybe_inherit_accent)

        if expanded:
            QTimer.singleShot(0, self._notify_layout_change)

    def _apply_accent_styles(self) -> None:
        """(Re)apply header + content-frame stylesheets from current accent state."""
        accent = self._effective_accent
        if accent is None:
            title_color = self._default_title_color
            frame_qss = (
                "QFrame#collapsible_content { border: 1px solid #666666; "
                "border-radius: 4px; margin: 0px 2px 2px 2px; }"
            )
        else:
            title_color = accent if self._is_outer_accent else self._muted_accent(accent)
            # Stripe matches the header text colour; width is uniform across
            # outer and inner sections.
            frame_qss = (
                "QFrame#collapsible_content { "
                "border-top: 1px solid #666666; "
                "border-right: 1px solid #666666; "
                "border-bottom: 1px solid #666666; "
                f"border-left: 2px solid {title_color}; "
                "border-top-left-radius: 0px; "
                "border-bottom-left-radius: 0px; "
                "border-top-right-radius: 4px; "
                "border-bottom-right-radius: 4px; "
                "margin: 0px 2px 2px 2px; "
                "}"
            )
        self._toggle.setStyleSheet(
            "QToolButton#collapsible_toggle { "
            f"font-weight: bold; font-size: 10pt; border: none; "
            f"padding: 2px; color: {title_color}; "
            "}"
        )
        self._content_frame.setStyleSheet(frame_qss)

    def _maybe_inherit_accent(self) -> None:
        """Walk up the parent chain and pick up the nearest ancestor's accent."""
        if self._explicit_accent is not None:
            return
        ancestor_color = self._find_ancestor_accent_color()
        if ancestor_color is None or ancestor_color == self._effective_accent:
            return
        self._effective_accent = ancestor_color
        self._is_outer_accent = False
        self._apply_accent_styles()

    def _find_ancestor_accent_color(self) -> str | None:
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, CollapsibleSection):
                if parent._effective_accent is not None:
                    return parent._effective_accent
            parent = parent.parent()
        return None

    @staticmethod
    def _muted_accent(hex_str: str) -> str:
        """Desaturate and flatten lightness so inner headers read as 'same hue,
        quieter' rather than as a separate color."""
        c = QColor(hex_str)
        h, s, l, a = c.getHslF()
        new_s = max(0.0, s * 0.35)
        new_l = 0.55 + (l - 0.55) * 0.3
        new_l = max(0.0, min(1.0, new_l))
        c.setHslF(h, new_s, new_l, a)
        return c.name()

    def set_title(self, title: str) -> None:
        """Update the header text."""
        self._base_title = title
        self._toggle.setText(self._qt_display_text(title))

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

    @staticmethod
    def _qt_display_text(title: str) -> str:
        """Escape mnemonic markers so literal ampersands render correctly."""
        return title.replace("&", "&&")

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

    def __init__(
        self,
        rel_path: str,
        display_name: str,
        loadable: str | None = None,
        viewer=None,
        legacy_rel_path: str | None = None,
    ):
        super().__init__()
        self._rel_path = rel_path
        self._legacy_rel_path = legacy_rel_path
        self._loadable = loadable or self._infer_load_kind(rel_path)
        self._full_path: "Path | None" = None
        self._viewer = viewer

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

        self._load_btn = QPushButton("↑")
        icon_button(self._load_btn, width=18, height=18)
        self._load_btn.clicked.connect(self._on_load_clicked)
        self._load_btn.setEnabled(False)
        self._load_btn.setToolTip(self._load_tooltip())
        # Hide the button entirely when no viewer is wired in or file is not napari-loadable.
        self._load_btn.setVisible(viewer is not None and self._loadable is not None)
        lay.addWidget(self._load_btn)

    def set_present(self, info_text: str) -> None:
        self._icon_lbl.setText("✓")
        self._icon_lbl.setStyleSheet("font-size: 9pt; font-weight: bold; color: #4CAF50;")
        self._info_lbl.setText(info_text)
        status_label(self._info_lbl)
        self._update_load_button()

    def set_missing(self) -> None:
        self._icon_lbl.setText("✗")
        muted_label(self._icon_lbl, size_pt=9)
        self._info_lbl.setText("missing")
        muted_label(self._info_lbl)
        self._full_path = None
        self._load_btn.setEnabled(False)
        self._load_btn.setToolTip(self._load_tooltip(missing=True))

    def set_no_project(self) -> None:
        self._icon_lbl.setText("○")
        muted_label(self._icon_lbl, size_pt=9)
        self._info_lbl.setText("—")
        muted_label(self._info_lbl)
        self._full_path = None
        self._load_btn.setEnabled(False)
        self._load_btn.setToolTip(self._load_tooltip(no_project=True))

    def _update_load_button(self) -> None:
        if self._full_path is None or not self._full_path.exists():
            self._load_btn.setEnabled(False)
            self._load_btn.setToolTip(self._load_tooltip(missing=True))
            return

        if self._loadable in {"tracked", "labels", "tiff"}:
            self._load_btn.setEnabled(True)
            self._load_btn.setToolTip(self._load_tooltip())
            return

        self._load_btn.setEnabled(False)
        self._load_btn.setToolTip("No direct napari load action for this file.")

    def _on_load_clicked(self) -> None:
        if self._full_path is None or not self._full_path.exists():
            return

        self._load_file_into_viewer()

    def _load_file_into_viewer(self) -> None:
        viewer = self._viewer if self._viewer is not None else self._find_viewer()
        if viewer is None:
            return

        import tifffile

        data = tifffile.imread(str(self._full_path))
        layer_name = self._layer_name()
        use_labels = self._loadable in {"tracked", "labels"}

        if use_labels:
            if layer_name in viewer.layers:
                try:
                    viewer.layers[layer_name].data = data
                    return
                except Exception:
                    viewer.layers.remove(viewer.layers[layer_name])
            viewer.add_labels(data, name=layer_name)
        else:
            colormap = self._pick_colormap()
            if layer_name in viewer.layers:
                try:
                    viewer.layers[layer_name].data = data
                    return
                except Exception:
                    viewer.layers.remove(viewer.layers[layer_name])
            viewer.add_image(data, name=layer_name, colormap=colormap)

    def _pick_colormap(self) -> str:
        rel = self._rel_path
        name = Path(rel).name
        if rel.startswith("0_input/") or name.endswith(("_zavg.tif", "_3dt.tif")):
            return "gray"
        if (
            rel.startswith("1_cellpose/")
            or rel == "2_nucleus/contours.tif"
            or (rel.startswith("2_nucleus/") and name.startswith("foreground_"))
            or (rel.startswith("3_cell/") and name.startswith("foreground_"))
        ):
            return "inferno"
        return "gray"

    def _find_viewer(self):
        widget = self.parentWidget()
        while widget is not None:
            viewer = getattr(widget, "viewer", None)
            if viewer is not None and hasattr(viewer, "add_image") and hasattr(viewer, "add_labels"):
                return viewer
            widget = widget.parentWidget()
        return None

    def _layer_name(self) -> str:
        return Path(self._rel_path).with_suffix("").as_posix().replace("/", "_")

    def _load_tooltip(self, *, missing: bool = False, no_project: bool = False) -> str:
        if no_project:
            return "No project open."
        if missing:
            return "File is missing."
        if self._loadable in {"tracked", "labels"}:
            return "Load labels into napari."
        if self._loadable == "tiff":
            return "Load into napari viewer."
        return "No direct napari load action for this file."

    @staticmethod
    def _infer_load_kind(rel_path: str) -> str | None:
        name = Path(rel_path).name
        if name == "tracked_labels.tif":
            return "tracked"
        if name.endswith("_labels.tif") or ("labels" in name and name.endswith((".tif", ".tiff"))):
            return "labels"
        if name.endswith((".tif", ".tiff")):
            return "tiff"
        return None


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
        viewer=None,
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
            for entry in entries:
                rel_path, display_name = entry[0], entry[1]
                legacy = entry[2] if len(entry) > 2 else None
                row = _PipelineFileRow(rel_path, display_name, loadable=None, viewer=viewer, legacy_rel_path=legacy)
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
            if not full_path.exists() and row._legacy_rel_path is not None:
                legacy_path = pos_dir / row._legacy_rel_path
                if legacy_path.exists():
                    full_path = legacy_path
            if full_path.exists():
                row._full_path = full_path
                row.set_present(_file_info(full_path))
            else:
                row.set_missing()
