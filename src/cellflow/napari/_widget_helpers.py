"""Small Qt widget factory helpers shared across napari workflow widgets."""
from __future__ import annotations

from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyleOption,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import (
    QLabeledDoubleRangeSlider,
    QLabeledDoubleSlider,
    QLabeledRangeSlider,
    QLabeledSlider,
)

from cellflow.napari.ui_style import action_button, parameter_heading, status_label


def separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #555;")
    return line


def heading(text: str) -> QLabel:
    lbl = QLabel(text)
    return parameter_heading(lbl)


def make_status() -> QLabel:
    lbl = QLabel("")
    lbl.setWordWrap(True)
    lbl.setVisible(False)
    status_label(lbl)
    return lbl


def make_progress() -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(True)
    bar.setVisible(False)
    return bar


def dspin(lo, hi, val, step=0.1, decimals=2, tooltip=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
    s.setDecimals(decimals); s.setToolTip(tooltip)
    return s


def ispin(lo, hi, val, step=1, tooltip=""):
    s = QSpinBox()
    s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
    s.setToolTip(tooltip)
    return s


def btn(text, tooltip=""):
    b = QPushButton(text)
    b.setToolTip(tooltip)
    action_button(b, expand=True)
    return b


def _patch_label_autosize(label) -> None:
    """Override the superqt SliderLabel's internal size calculation so it
    fits the widest value it can display, formatted with the configured
    decimals.

    superqt's stock ``_get_size`` widths labels from ``str(minimum())`` /
    ``str(maximum())`` — which drops trailing zeros (``str(1.0) == "1.0"``)
    and ignores decimals, so e.g. a (-10, 10) range with 1 decimal sizes
    the label for ``"-10.0"`` but then ``str(-10.0) == "-10.0"``... fine
    in that case, while a (0, 1) range with 2 decimals sizes for ``"1.0"``
    (3 chars) and clips the displayed ``"1.00"``. Style padding/font
    tweaks compound this. We replace ``_get_size`` on the instance so
    that every subsequent ``_update_size`` (rangeChanged, showEvent…)
    re-derives the size from the actual format width plus headroom for
    chrome and a possible minus sign.
    """
    def _get_size():
        dec = label.decimals() if hasattr(label, "decimals") else 0

        def _fmt(v):
            return f"{v:.{dec}f}" if dec else f"{int(v)}"
        lo, hi = label.minimum(), label.maximum()
        sample = max((_fmt(lo), _fmt(hi)), key=len)
        # ensure room for a minus sign even if both ends are non-negative
        # (the user may type one into the editable label).
        if not sample.startswith("-"):
            sample = "-" + sample
        fm = label.fontMetrics()
        prefix = label.prefix() or ""
        suffix = label.suffix() or ""
        w = fm.horizontalAdvance(prefix + sample + suffix) + 18
        h = label.sizeHint().height()
        opt = QStyleOption()
        return label.style().sizeFromContents(
            QStyle.ContentsType.CT_LineEdit, opt, QSize(w, h), label
        )

    label._get_size = _get_size
    label._update_size()


def _stack_slider_label_above(slider) -> None:
    """Repack a QLabeledSlider / QLabeledDoubleSlider so the editable value
    label sits centered above the slider track instead of beside it."""
    label = slider._label
    track = slider._slider
    label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    _patch_label_autosize(label)

    old_layout = slider.layout()
    if old_layout is not None:
        old_layout.removeWidget(label)
        old_layout.removeWidget(track)
        QWidget().setLayout(old_layout)
    label.setParent(slider)
    track.setParent(slider)
    vbox = QVBoxLayout()
    vbox.setContentsMargins(0, 0, 0, 0)
    vbox.setSpacing(0)
    vbox.addWidget(label, alignment=Qt.AlignmentFlag.AlignHCenter)
    vbox.addWidget(track)
    slider.setLayout(vbox)


def dslider(lo, hi, val, step=0.1, decimals=2, tooltip=""):
    """A horizontal QLabeledDoubleSlider — same call signature as `dspin`.

    The editable value label sits above the slider track for a compact,
    wide-track look."""
    s = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setToolTip(tooltip)
    s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    _stack_slider_label_above(s)
    return s


def islider(lo, hi, val, step=1, tooltip=""):
    """A horizontal QLabeledSlider — same call signature as `ispin`.

    The editable value label sits above the slider track."""
    s = QLabeledSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setToolTip(tooltip)
    s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    _stack_slider_label_above(s)
    return s


def _hide_range_edge_labels(slider) -> None:
    """Hide the edge labels of a QLabeled[Double]RangeSlider.

    The slider already renders an editable value label above each thumb
    (HandleLabelPosition.LabelsAbove, the superqt default), so the edge
    labels would just duplicate the same numbers at the track ends."""
    for name in ("_min_label", "_max_label"):
        lbl = getattr(slider, name, None)
        if lbl is not None:
            lbl.hide()


def _force_handle_label_width(slider) -> None:
    """Resize the handle labels of a range slider to fit the widest value
    they may display, robust to later rangeChanged/showEvent re-sizes."""
    for lbl in getattr(slider, "_handle_labels", []) or []:
        _patch_label_autosize(lbl)


def drslider(lo, hi, lo_val, hi_val, step=0.1, decimals=2, tooltip=""):
    """A horizontal QLabeledDoubleRangeSlider with one editable label above
    each thumb. Edge labels are hidden to avoid duplicating those values."""
    s = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue((lo_val, hi_val))
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setToolTip(tooltip)
    s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    _hide_range_edge_labels(s)
    _force_handle_label_width(s)
    return s


def irslider(lo, hi, lo_val, hi_val, step=1, tooltip=""):
    """A horizontal QLabeledRangeSlider (integer) with one editable label
    above each thumb. Edge labels are hidden to avoid duplicating those
    values."""
    s = QLabeledRangeSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue((lo_val, hi_val))
    s.setSingleStep(step)
    s.setToolTip(tooltip)
    s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    _hide_range_edge_labels(s)
    _force_handle_label_width(s)
    return s


class RangeThumbProxy:
    """A spinbox-like proxy (.value() / .setValue()) that reads and writes
    one thumb of a range slider, so existing call sites that used a
    standalone QSpinBox/QDoubleSpinBox per thumb keep working."""
    __slots__ = ("_slider", "_index")

    def __init__(self, slider, index: int) -> None:
        self._slider = slider
        self._index = index

    def value(self):
        return self._slider.value()[self._index]

    def setValue(self, v) -> None:
        vals = list(self._slider.value())
        vals[self._index] = v
        # Range sliders enforce min<=max; nudge the other thumb if needed.
        if self._index == 0 and v > vals[1]:
            vals[1] = v
        elif self._index == 1 and v < vals[0]:
            vals[0] = v
        self._slider.setValue(tuple(vals))


def tool_btn(glyph: str, tooltip: str = "", *, checkable: bool = False) -> QToolButton:
    """Compact icon-only QToolButton carrying a unicode glyph and a tooltip."""
    b = QToolButton()
    b.setText(glyph)
    b.setToolTip(tooltip)
    b.setCheckable(checkable)
    b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return b


def button_grid(*rows: tuple[QPushButton, ...]) -> QGridLayout:
    grid = QGridLayout()
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(4)
    for r, buttons in enumerate(rows):
        for c, button in enumerate(buttons):
            span = 2 - c if c == len(buttons) - 1 and len(buttons) == 1 else 1
            grid.addWidget(button, r, c, 1, span)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)
    return grid
