"""Shared reusable Qt widgets for the CellFlow napari plugin."""
from __future__ import annotations

from pathlib import Path

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
    muted_accent,
    muted_label,
    stage_header_action_button,
    stage_header_label,
    status_label,
)
from ._widget_helpers import tool_btn


class CollapsibleSection(QWidget):
    """A labelled section with a toggle button that shows/hides its inner widget."""

    def __init__(
        self,
        title: str,
        inner: QWidget,
        expanded: bool = False,
        parent: QWidget | None = None,
        title_color: str | None = None,
        accent_color: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._inner = inner
        self._base_title = title
        self._default_title_color: str | None = title_color
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

        self._status: str | None = None

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(4)
        header_row.addWidget(self._toggle, 1)
        layout.addLayout(header_row)

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
            font_size_pt = 10
            frame_qss = (
                "QFrame#collapsible_content { border: 1px solid #666666; "
                "border-radius: 4px; margin: 0px 2px 2px 2px; }"
            )
        else:
            if self._is_outer_accent:
                title_color = accent
                font_size_pt = 11
            else:
                title_color = self._muted_accent(accent)
                font_size_pt = 9
            # Only the coloured left stripe — no gray frame around the content,
            # since the stripe alone is enough to delimit the section.
            frame_qss = (
                "QFrame#collapsible_content { "
                "border: none; "
                f"border-left: 2px solid {title_color}; "
                "border-radius: 0px; "
                "margin: 0px 2px 2px 2px; "
                "}"
            )
        color_rule = f"color: {title_color}; " if title_color else ""
        self._toggle.setStyleSheet(
            "QToolButton#collapsible_toggle { "
            f"font-weight: bold; font-size: {font_size_pt}pt; border: none; "
            f"padding: 2px; {color_rule}"
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

    def set_accent_color(self, accent_color: str | None) -> None:
        """Set this section's explicit accent and refresh inherited child accents."""
        self._explicit_accent = accent_color
        self._effective_accent = accent_color
        self._is_outer_accent = accent_color is not None
        self._apply_accent_styles()
        self._refresh_descendant_inherited_accents()

    def _refresh_descendant_inherited_accents(self) -> None:
        for child in self.findChildren(CollapsibleSection):
            if child._explicit_accent is not None:
                continue
            ancestor_color = child._find_ancestor_accent_color()
            child._effective_accent = ancestor_color
            child._is_outer_accent = False
            child._apply_accent_styles()

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
        return muted_accent(hex_str)

    def set_header_visible(self, visible: bool) -> None:
        """Show or hide the built-in toggle header row."""
        self._toggle.setVisible(visible)

    def set_title(self, title: str) -> None:
        """Update the header text."""
        self._base_title = title
        self._toggle.setText(self._qt_display_text(title))

    def set_status(self, status: str | None) -> None:
        """Store section status for callers without rendering a header indicator."""
        self._status = status

    @property
    def status(self) -> str | None:
        return self._status

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


def make_pipeline_files_header(
    section: CollapsibleSection,
    *,
    stage_key: str,
    parent: QWidget | None = None,
) -> tuple[QWidget, QLabel, QToolButton]:
    """Create a compact external header for a pipeline files section."""
    section.set_header_visible(False)

    header = QWidget(parent)
    layout = QHBoxLayout(header)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    label = QLabel("Pipeline Files")
    stage_header_label(label, stage_key)

    button = tool_btn("🔍", "Show pipeline files.", checkable=True)
    stage_header_action_button(button, stage_key)
    button.setChecked(section.is_expanded)

    def _set_expanded(checked: bool) -> None:
        if checked:
            section.expand()
        else:
            section.collapse()
        button.setToolTip(
            "Hide pipeline files." if checked else "Show pipeline files."
        )

    button.toggled.connect(_set_expanded)
    section._toggle.toggled.connect(button.setChecked)

    layout.addWidget(label)
    layout.addWidget(button)
    layout.addStretch(1)
    return header, label, button


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
        self._full_path: Path | None = None
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
        """Colormap for non-label intensity TIFFs.

        Mirrors the dedicated cellpose/divergence viewers so a file looks the
        same however it's loaded: contour/divergence maps → ``magma``, cellpose
        probability → ``viridis``, flow/dp → ``inferno``, and raw input or
        foreground intensity → plain grayscale.
        """
        name = Path(self._rel_path).name
        if "contour" in name:
            return "magma"
        if "_prob" in name:
            return "viridis"
        if "_dp" in name or "flow" in name:
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
        if name == "foreground_masks.tif":
            return "labels"
        # atoms.tif is an int32 atom-ID image (stage ① output); load as labels.
        if name == "atoms.tif":
            return "labels"
        if name.endswith("_labels.tif") or ("labels" in name and name.endswith((".tif", ".tiff"))):
            return "labels"
        if name.endswith((".tif", ".tiff")):
            return "tiff"
        return None


def _file_info(path: Path) -> str:
    """Return a concise shape/dtype string for a pipeline output file."""
    if path.is_dir():
        return "Directory"
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(path) as tif:
                # The series carries the full N-D shape (T, Z, Y, X); a single
                # page is only the 2D Y×X plane, which would drop z and t.
                series = tif.series[0] if tif.series else None
                shape = tuple(series.shape) if series is not None else tuple(tif.pages[0].shape)
            return "×".join(str(d) for d in shape)
        except Exception:
            pass
        return "TIFF"
    if suffix in (".h5", ".hdf5"):
        return "HDF5"
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
        self._rows_by_group: dict[str, list[_PipelineFileRow]] = {}

        for group_label, entries in groups:
            if group_label:
                hdr = QLabel(group_label)
                hdr.setStyleSheet(
                    "font-size: 7pt; font-weight: bold; padding: 1px 4px;"
                    " background: palette(alternateBase); color: palette(mid);"
                )
                lay.addWidget(hdr)
            group_rows: list[_PipelineFileRow] = []
            for entry in entries:
                rel_path, display_name = entry[0], entry[1]
                legacy = entry[2] if len(entry) > 2 else None
                row = _PipelineFileRow(rel_path, display_name, loadable=None, viewer=viewer, legacy_rel_path=legacy)
                self._rows.append(row)
                group_rows.append(row)
                lay.addWidget(row)
            if group_label:
                self._rows_by_group[group_label] = group_rows

    def refresh(self, pos_dir: Path | None) -> None:
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

    def presence_count_by_group(self) -> dict[str, tuple[int, int]]:
        """Per-group (n_present, n_total) snapshot of current file state."""
        return {
            label: (
                sum(1 for r in rows if r._full_path is not None),
                len(rows),
            )
            for label, rows in self._rows_by_group.items()
        }


def pipeline_status_from_files(
    tracker: PipelineFilesWidget, *, done_group: str
) -> str:
    """Derive a "not_started" / "in_progress" / "done" status from on-disk files.

    `done` when ``done_group`` is fully present. `in_progress` when any file in
    an "Intermediates" group is present, or when ``done_group`` is partially
    populated. Otherwise `not_started`.
    """
    counts = tracker.presence_count_by_group()
    done_present, done_total = counts.get(done_group, (0, 0))
    if done_total > 0 and done_present == done_total:
        return "done"
    intermediates_present = counts.get("Intermediates", (0, 0))[0]
    if intermediates_present > 0 or done_present > 0:
        return "in_progress"
    return "not_started"
