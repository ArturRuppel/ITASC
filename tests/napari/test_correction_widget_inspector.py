"""Regression: the goto/step spinbox must survive show_inspector=False + GC.

The "Inspect cell" group can be dropped from the layout (show_inspector=False,
e.g. the nucleus workspace where the lineage canvas navigates instead), but the
Cell-ID spinbox and frames label still back the goto + Shift-± step logic. If
nothing keeps the un-added group alive, Python GCs it and deletes the C++
QSpinBox, so the next selection crashes with "wrapped C/C++ object ... deleted".
"""
from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication, QGroupBox  # noqa: E402

from cellflow.napari.correction_widget import CorrectionWidget  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _widget(show_inspector: bool) -> CorrectionWidget:
    # __init__ only builds Qt; the viewer is untouched until activation, so a
    # bare stand-in is enough to exercise the inspector wiring.
    return CorrectionWidget(SimpleNamespace(), show_inspector=show_inspector)


def test_goto_spinbox_survives_gc_when_inspector_hidden(_app):
    widget = _widget(show_inspector=False)
    gc.collect()  # would reap the orphaned group before the fix
    # The exact path from the crash report:
    old = widget._goto_cell_id.blockSignals(True)
    widget._goto_cell_id.setValue(5)
    widget._goto_cell_id.blockSignals(old)
    assert widget._goto_cell_id.value() == 5
    # The group is kept alive but never shown.
    assert not widget._inspect_group.isVisible()


def test_inspector_group_in_layout_when_shown(_app):
    widget = _widget(show_inspector=True)
    titles = [g.title() for g in widget.findChildren(QGroupBox)]
    assert "Inspect cell" in titles


def test_inspector_group_absent_from_tree_when_hidden(_app):
    widget = _widget(show_inspector=False)
    # Not parented into the widget tree (so it does not render), but still alive.
    titles = [g.title() for g in widget.findChildren(QGroupBox)]
    assert "Inspect cell" not in titles
    assert widget._inspect_group is not None
