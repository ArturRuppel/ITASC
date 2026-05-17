"""Small Qt widget factory helpers shared across napari workflow widgets."""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider, QLabeledSlider

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


def _stack_slider_label_above(slider) -> None:
    """Repack a QLabeledSlider / QLabeledDoubleSlider so the editable value
    label sits centered above the slider track instead of beside it."""
    label = slider._label
    track = slider._slider
    label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

    # Force the editable label to be wide enough for the longest value
    # in the slider's range (including a minus sign and decimal places).
    decimals = getattr(label, "decimals", lambda: 0)()
    lo, hi = slider.minimum(), slider.maximum()

    def _fmt(v):
        return f"{v:.{decimals}f}" if decimals else f"{int(v)}"
    sample = max((_fmt(lo), _fmt(hi)), key=len)
    fm = label.fontMetrics()
    label.setMinimumWidth(fm.horizontalAdvance(sample) + 12)

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


def _stack_range_labels_above(slider) -> None:
    """Repack a QLabeledDoubleRangeSlider (in LabelIsValue mode) so the
    two editable thumb-value labels sit at the left/right edges above the
    track instead of beside it."""
    min_label = slider._min_label
    max_label = slider._max_label
    track = slider._slider

    decimals = getattr(min_label, "decimals", lambda: 0)()
    lo, hi = slider.minimum(), slider.maximum()

    def _fmt(v):
        return f"{v:.{decimals}f}" if decimals else f"{int(v)}"
    sample = max((_fmt(lo), _fmt(hi)), key=len)

    for lbl in (min_label, max_label):
        lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lbl.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        fm = lbl.fontMetrics()
        lbl.setMinimumWidth(fm.horizontalAdvance(sample) + 12)

    old_layout = slider.layout()
    if old_layout is not None:
        for child in (min_label, max_label, track):
            old_layout.removeWidget(child)
        QWidget().setLayout(old_layout)
    for child in (min_label, max_label, track):
        child.setParent(slider)

    hbox = QHBoxLayout()
    hbox.setContentsMargins(0, 0, 0, 0)
    hbox.setSpacing(0)
    hbox.addWidget(min_label, alignment=Qt.AlignmentFlag.AlignLeft)
    hbox.addStretch()
    hbox.addWidget(max_label, alignment=Qt.AlignmentFlag.AlignRight)

    vbox = QVBoxLayout()
    vbox.setContentsMargins(0, 0, 0, 0)
    vbox.setSpacing(0)
    vbox.addLayout(hbox)
    vbox.addWidget(track)
    slider.setLayout(vbox)


def drslider(lo, hi, lo_val, hi_val, step=0.1, decimals=2, tooltip=""):
    """A horizontal QLabeledDoubleRangeSlider with the two editable thumb
    values rendered as edge labels stacked above the track."""
    s = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue((lo_val, hi_val))
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setToolTip(tooltip)
    s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    s.setEdgeLabelMode(s.EdgeLabelMode.LabelIsValue)
    _stack_range_labels_above(s)
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
