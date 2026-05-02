from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication, QLabel, QPushButton, QSizePolicy, QSpinBox, QWidget

package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"
napari_pkg = types.ModuleType("cellflow.napari")
napari_pkg.__path__ = [str(package_root)]
sys.modules["cellflow.napari"] = napari_pkg

from cellflow.napari.ui_style import (
    DEFAULT_SPIN_WIDTH,
    SECTION_MARGIN,
    TIGHT_SPACING,
    TINY_MARGIN,
    action_button,
    checked_success_button,
    compact_spinbox,
    danger_button,
    icon_button,
    muted_label,
    status_label,
    tiny_button,
)
from cellflow.database.hypotheses import HypothesisRecord, write_hypothesis_record
from cellflow.napari.widgets import PipelineFilesWidget
from cellflow.segmentation import NucleusHypothesisParams


@pytest.fixture
def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_layout_constants_are_compact():
    assert TINY_MARGIN == 2
    assert SECTION_MARGIN == 4
    assert TIGHT_SPACING == 4
    assert DEFAULT_SPIN_WIDTH == 70


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
    assert "font-style" not in plain.styleSheet()

    status_label(italic, italic=True)
    assert "font-size: 8pt" in italic.styleSheet()
    assert "font-style: italic" in italic.styleSheet()

    status_label(muted, italic=True, muted=True)
    assert "palette(mid)" in muted.styleSheet()
    assert "font-style: italic" in muted.styleSheet()


def test_danger_button_uses_semantic_red_and_hover_style(_app):
    button = QPushButton("Delete")

    assert danger_button(button) is button

    style = button.styleSheet()
    assert "QPushButton" in style
    assert "background-color" in style
    assert "#b00020" in style
    assert "QPushButton:hover" in style


def test_checked_success_button_styles_checked_state_green(_app):
    button = QPushButton("Activate")

    assert checked_success_button(button) is button

    style = button.styleSheet()
    assert "QPushButton:checked" in style
    assert "background-color" in style
    assert "#2e7d32" in style
    assert "font-weight: bold" in style


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


def test_hypotheses_h5_summary_shows_parameter_and_time_counts(_app, tmp_path):
    h5_path = tmp_path / "hypotheses.h5"
    labels = np.zeros((1, 12, 34), dtype=np.uint32)
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["version"] = 2
        h5.attrs["stage"] = "nucleus_hypotheses"
        h5.attrs["layout"] = "hypotheses/t{t:03d}/p{p:03d}/labels"
        for t in range(2):
            for p in range(2):
                write_hypothesis_record(
                    h5,
                    HypothesisRecord(
                        t=t,
                        p=p,
                        labels=labels,
                        params=NucleusHypothesisParams(z_slice=0),
                    ),
                )

    widget = PipelineFilesWidget(
        [("Outputs", [("hypotheses.h5", "Hypotheses DB")])]
    )

    widget.refresh(tmp_path)

    assert widget._rows[0]._info_lbl.text() == "2×2×12×34"

    widget.deleteLater()


def test_pipeline_files_widget_load_buttons_route_supported_rows_and_disable_hypotheses(_app, tmp_path):
    class _FakeViewer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, tuple[int, ...]]] = []

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
    with h5py.File(tmp_path / "hypotheses.h5", "w"):
        pass

    widget = PipelineFilesWidget(
        [
            (
                "Outputs",
                [
                    ("nucleus_zavg.tif", "Nucleus z-avg"),
                    ("tracked_labels.tif", "Tracked labels"),
                    ("hypotheses.h5", "Hypotheses DB"),
                ],
            )
        ],
        parent=wrapper,
    )
    widget.refresh(tmp_path)

    assert widget._rows[0]._load_btn.isEnabled()
    assert widget._rows[1]._load_btn.isEnabled()
    assert not widget._rows[2]._load_btn.isEnabled()

    widget._rows[0]._load_btn.click()
    widget._rows[1]._load_btn.click()

    assert wrapper.viewer.calls == [
        ("image", "nucleus_zavg", (4, 6)),
        ("labels", "tracked_labels", (2, 4, 6)),
    ]

    widget.deleteLater()
