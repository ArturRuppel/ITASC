from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
napari_pkg = types.ModuleType("cellflow.napari")
napari_pkg.__path__ = [str(package_root)]
sys.modules["cellflow.napari"] = napari_pkg

from cellflow.napari.ui_style import (
    DEFAULT_SPIN_WIDTH,
    FIELD_NOTES,
    SECTION_MARGIN,
    SOLARIZED_DARK,
    TIGHT_SPACING,
    TINY_MARGIN,
    action_button,
    checked_success_button,
    compact_spinbox,
    danger_button,
    icon_button,
    muted_label,
    parameter_heading,
    status_label,
    tiny_button,
)
from cellflow.napari.widgets import CollapsibleSection, PipelineFilesWidget


@pytest.fixture
def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_layout_constants_are_compact():
    assert TINY_MARGIN == 2
    assert SECTION_MARGIN == 4
    assert TIGHT_SPACING == 4
    assert DEFAULT_SPIN_WIDTH == 70


def test_theme_palettes_have_required_accent_keys():
    required_keys = {
        "rosewater",
        "flamingo",
        "pink",
        "mauve",
        "red",
        "maroon",
        "peach",
        "yellow",
        "green",
        "teal",
        "sky",
        "sapphire",
        "blue",
        "lavender",
    }

    assert set(SOLARIZED_DARK) == required_keys
    assert set(FIELD_NOTES) == required_keys


def test_compact_spinbox_sets_width_and_fixed_policy(_app):
    spin = QSpinBox()

    styled = compact_spinbox(spin)

    assert styled is spin
    assert spin.maximumWidth() == DEFAULT_SPIN_WIDTH
    assert spin.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
    assert spin.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed


def test_compact_spinbox_accepts_custom_width(_app):
    spin = QSpinBox()

    compact_spinbox(spin, width=56)

    assert spin.maximumWidth() == 56


def test_action_button_uses_fixed_or_expanding_horizontal_policy(_app):
    fixed = QPushButton("Run")
    expanding = QPushButton("Run all")

    assert action_button(fixed) is fixed
    assert fixed.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
    assert fixed.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed

    action_button(expanding, expand=True)
    assert expanding.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert expanding.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed
    assert expanding.styleSheet() == ""


def test_tiny_button_sets_small_style_and_fixed_vertical_policy(_app):
    button = QPushButton("...")

    assert tiny_button(button) is button

    style = button.styleSheet()
    assert "font-size" in style
    assert "padding" in style
    assert button.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed


def test_icon_button_sets_fixed_width_and_optional_height(_app):
    width_only = QPushButton()
    sized = QPushButton()

    assert icon_button(width_only) is width_only
    assert width_only.minimumWidth() == 24
    assert width_only.maximumWidth() == 24

    icon_button(sized, width=28, height=20)
    assert sized.minimumWidth() == 28
    assert sized.maximumWidth() == 28
    assert sized.minimumHeight() == 20
    assert sized.maximumHeight() == 20


def test_muted_label_uses_palette_mid_and_font_size(_app):
    label = QLabel("Metadata")

    assert muted_label(label, size_pt=9) is label

    style = label.styleSheet()
    assert "palette(mid)" in style
    assert "font-size: 9pt" in style


def test_status_label_sets_font_size_and_optional_italic(_app):
    plain = QLabel("Idle")
    italic = QLabel("Waiting")
    muted = QLabel("Muted")

    assert status_label(plain, size_pt=10) is plain
    assert "font-size: 10pt" in plain.styleSheet()
    assert "color" not in plain.styleSheet()
    assert "font-style" not in plain.styleSheet()

    status_label(italic, italic=True)
    assert "font-size: 8pt" in italic.styleSheet()
    assert "font-style: italic" in italic.styleSheet()

    status_label(muted, italic=True, muted=True)
    assert "palette(mid)" in muted.styleSheet()
    assert "font-style: italic" in muted.styleSheet()



def test_parameter_heading_uses_params_role_and_level(_app):
    label = QLabel("Contour")

    assert parameter_heading(label) is label

    style = label.styleSheet()
    assert "font-weight: 600" in style
    assert "color" not in style


def test_danger_button_keeps_native_button_style(_app):
    button = QPushButton("Delete")

    assert danger_button(button) is button

    assert button.styleSheet() == ""


def test_checked_success_button_keeps_native_button_style(_app):
    button = QPushButton("Activate")

    assert checked_success_button(button) is button

    assert button.styleSheet() == ""


def test_collapsible_section_does_not_tint_action_buttons(_app):
    inner = QWidget()
    layout = QVBoxLayout(inner)
    button = QPushButton("Run")
    action_button(button, expand=True)
    layout.addWidget(button)

    section = CollapsibleSection(
        "Pipeline",
        inner,
        expanded=True,
        accent_color="#89b4fa",
    )
    section.show()
    _app.processEvents()

    assert button.styleSheet() == ""

    section.deleteLater()


def test_collapsible_section_header_spans_available_width(_app):
    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    section = CollapsibleSection("Pipeline Files", QLabel("files"), expanded=False)
    layout.addWidget(section)

    wrapper.resize(360, 100)
    wrapper.show()
    _app.processEvents()

    toggle = section.findChild(QToolButton, "collapsible_toggle")
    assert toggle is not None
    assert toggle.width() == section.width()
    assert section.childAt(section.width() - 4, toggle.y() + toggle.height() // 2) is toggle
    assert toggle.text() == "Pipeline Files"

    wrapper.deleteLater()


def test_pipeline_files_widget_reflects_present_and_missing_states(_app, tmp_path):
    widget = PipelineFilesWidget(
        [
            (
                "Outputs",
                [
                    ("present.txt", "Present output"),
                    ("missing.txt", "Missing output"),
                ],
            )
        ]
    )
    rows = widget._rows

    widget.refresh(None)

    assert [row._info_lbl.text() for row in rows] == ["—", "—"]
    assert all("palette(mid)" in row._info_lbl.styleSheet() for row in rows)

    (tmp_path / "present.txt").write_text("data")
    widget.refresh(tmp_path)

    assert rows[0]._icon_lbl.text() == "✓"
    assert rows[0]._info_lbl.text() == "0 KB"
    assert rows[1]._icon_lbl.text() == "✗"
    assert rows[1]._info_lbl.text() == "missing"
    assert "palette(mid)" in rows[1]._info_lbl.styleSheet()

    widget.deleteLater()


def test_pipeline_files_widget_load_buttons_load_supported_files_directly(_app, tmp_path):
    class _FakeViewer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, tuple[int, ...]]] = []
            self.layers = {}

        def add_image(self, data, name=None, **kwargs):
            self.calls.append(("image", name, tuple(np.asarray(data).shape)))

        def add_labels(self, data, name=None, **kwargs):
            self.calls.append(("labels", name, tuple(np.asarray(data).shape)))

    wrapper = QWidget()
    wrapper.viewer = _FakeViewer()

    tifffile.imwrite(
        tmp_path / "nucleus_zavg.tif",
        np.zeros((4, 6), dtype=np.uint16),
        compression=None,
    )
    tifffile.imwrite(
        tmp_path / "tracked_labels.tif",
        np.zeros((2, 4, 6), dtype=np.uint32),
        compression=None,
    )

    widget = PipelineFilesWidget(
        [
            (
                "Outputs",
                [
                    ("nucleus_zavg.tif", "Nucleus z-avg"),
                    ("tracked_labels.tif", "Tracked labels"),
                ],
            )
        ],
        parent=wrapper,
    )
    widget.refresh(tmp_path)

    assert widget._rows[0]._load_btn.isEnabled()
    assert widget._rows[1]._load_btn.isEnabled()

    widget._rows[0]._load_btn.click()
    widget._rows[1]._load_btn.click()

    assert wrapper.viewer.calls == [
        ("image", "nucleus_zavg", (4, 6)),
        ("labels", "tracked_labels", (2, 4, 6)),
    ]

    widget.deleteLater()


def test_pipeline_files_widget_tracked_label_rows_do_not_delegate_to_workflow_loader(_app, tmp_path):
    class _FakeViewer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, tuple[int, ...]]] = []
            self.layers = {}

        def add_image(self, data, name=None, **kwargs):
            self.calls.append(("image", name, tuple(np.asarray(data).shape)))

        def add_labels(self, data, name=None, **kwargs):
            self.calls.append(("labels", name, tuple(np.asarray(data).shape)))

    class _WorkflowWidget(QWidget):
        viewer = _FakeViewer()

        def __init__(self, tracked_path: Path) -> None:
            super().__init__()
            self._tracked_path_value = tracked_path
            self.custom_loader_called = False

        def _tracked_path(self):
            return self._tracked_path_value

        def _on_load_tracked(self):
            self.custom_loader_called = True

    labels = np.zeros((2, 4, 6), dtype=np.uint32)
    tifffile.imwrite(tmp_path / "tracked_labels.tif", labels, compression=None)

    wrapper = _WorkflowWidget(tmp_path / "tracked_labels.tif")
    widget = PipelineFilesWidget([("Outputs", [("tracked_labels.tif", "Tracked labels")])], parent=wrapper)
    widget.refresh(tmp_path)

    widget._rows[0]._load_btn.click()

    assert wrapper.custom_loader_called is False
    assert wrapper.viewer.calls == [("labels", "tracked_labels", (2, 4, 6))]

    widget.deleteLater()
    wrapper.deleteLater()
