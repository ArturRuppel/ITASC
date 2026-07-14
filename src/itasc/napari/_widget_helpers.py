"""Small Qt widget factory helpers shared across napari workflow widgets."""
from __future__ import annotations

from qtpy.QtCore import QEvent, QObject, QSize, Qt
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import (
    QLabeledDoubleSlider,
    QLabeledSlider,
)

from itasc.napari.ui_style import action_button, parameter_heading, status_label


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


def _slider_step_button(text: str, object_name: str, tooltip: str) -> QToolButton:
    button = QToolButton()
    button.setText(text)
    button.setObjectName(object_name)
    button.setToolTip(tooltip)
    button.setAutoRepeat(True)
    button.setFixedSize(18, 18)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return button


class _SliderStepButtonStateSyncer(QObject):
    def __init__(self, sync_button_state) -> None:
        super().__init__()
        self._sync_button_state = sync_button_state

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.EnabledChange:
            self._sync_button_state()
        return False


def _connect_slider_step_buttons(slider) -> tuple[QToolButton, QToolButton]:
    decrement = _slider_step_button(
        "-", "slider_decrement_button", "Decrease by one step"
    )
    increment = _slider_step_button(
        "+", "slider_increment_button", "Increase by one step"
    )

    def _set_stepped_value(direction: int) -> None:
        if not slider.isEnabled():
            return
        slider.setValue(slider.value() + direction * slider.singleStep())

    def _sync_button_state(*_args) -> None:
        enabled = slider.isEnabled()
        decrement.setEnabled(enabled and slider.value() > slider.minimum())
        increment.setEnabled(enabled and slider.value() < slider.maximum())

    decrement.clicked.connect(lambda: _set_stepped_value(-1))
    increment.clicked.connect(lambda: _set_stepped_value(1))
    slider.valueChanged.connect(_sync_button_state)
    slider.rangeChanged.connect(_sync_button_state)
    state_syncer = _SliderStepButtonStateSyncer(_sync_button_state)
    state_syncer.setParent(slider)
    slider.installEventFilter(state_syncer)
    slider._itasc_slider_step_button_state_syncer = state_syncer
    _sync_button_state()
    return decrement, increment


def add_slider_step_buttons(
    layout: QHBoxLayout, slider, track: QWidget | None = None
) -> tuple[QToolButton, QToolButton]:
    track = slider if track is None else track
    decrement, increment = _connect_slider_step_buttons(slider)
    layout.addWidget(decrement, alignment=Qt.AlignmentFlag.AlignVCenter)
    layout.addWidget(track)
    layout.addWidget(increment, alignment=Qt.AlignmentFlag.AlignVCenter)
    return decrement, increment


def _stack_slider_label_above(slider, *, step_buttons: bool = False) -> None:
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
    if step_buttons:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        add_slider_step_buttons(row, slider, track)
        vbox.addLayout(row)
    else:
        vbox.addWidget(track)
    slider.setLayout(vbox)


def dslider(lo, hi, val, step=0.1, decimals=2, tooltip="", *, step_buttons=True):
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
    s.setProperty("itasc_stack_section_label", True)
    _stack_slider_label_above(s, step_buttons=step_buttons)
    return s


def islider(lo, hi, val, step=1, tooltip="", *, step_buttons=True):
    """A horizontal QLabeledSlider — same call signature as `ispin`.

    The editable value label sits above the slider track."""
    s = QLabeledSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setToolTip(tooltip)
    s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    s.setProperty("itasc_stack_section_label", True)
    _stack_slider_label_above(s, step_buttons=step_buttons)
    return s


def tool_btn(glyph: str, tooltip: str = "", *, checkable: bool = False) -> QToolButton:
    """Compact icon-only QToolButton carrying a unicode glyph and a tooltip."""
    b = QToolButton()
    b.setText(glyph)
    b.setToolTip(tooltip)
    b.setCheckable(checkable)
    b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return b
