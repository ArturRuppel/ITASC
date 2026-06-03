"""Wrap-math coverage for the candidate-strip FlowLayout (offscreen Qt)."""
from __future__ import annotations

import pytest

pytest.importorskip("qtpy")
from qtpy.QtCore import QSize  # noqa: E402
from qtpy.QtWidgets import QApplication, QWidget  # noqa: E402

from cellflow.napari._flow_layout import FlowLayout  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


class _Box(QWidget):
    """A fixed 20×20 item so wrap maths are deterministic."""

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt override)
        return QSize(20, 20)


def test_items_wrap_onto_new_rows_when_narrow(_app):
    host = QWidget()
    layout = FlowLayout(host, margin=0, h_spacing=4, v_spacing=4)
    for _ in range(4):
        layout.addWidget(_Box())

    # Wide enough for all four on one row → one row tall (~20).
    wide = layout.heightForWidth(1000)
    # Room for ~two per row (2*20 + spacing) → two rows tall (~44).
    narrow = layout.heightForWidth(50)

    assert layout.count() == 4
    assert narrow > wide
    assert wide == pytest.approx(20, abs=2)


def test_take_at_removes_items(_app):
    host = QWidget()
    layout = FlowLayout(host)
    layout.addWidget(_Box())
    layout.addWidget(_Box())

    assert layout.count() == 2
    assert layout.takeAt(0) is not None
    assert layout.count() == 1
    assert layout.takeAt(5) is None  # out of range is a safe no-op
