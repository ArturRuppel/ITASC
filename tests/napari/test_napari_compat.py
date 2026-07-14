from __future__ import annotations

import os
import importlib
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QRect, Qt
from qtpy.QtWidgets import QApplication, QStyleOptionViewItem


class _NoSizeHintIndex:
    def __init__(self, *, loaded: bool | None = False) -> None:
        self.loaded = loaded

    def isValid(self) -> bool:
        return False

    def data(self, role):
        from napari._qt.containers.qt_layer_model import LoadedRole

        if role == LoadedRole:
            return self.loaded
        if role == Qt.ItemDataRole.SizeHintRole:
            return None
        return None


def test_layer_delegate_loading_indicator_tolerates_missing_size_hint():
    app = QApplication.instance() or QApplication([])
    sys.modules.pop("itasc.napari", None)
    importlib.import_module("itasc.napari")
    from napari._qt.containers._layer_delegate import LayerDelegate

    delegate = LayerDelegate()
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 200, 34)

    delegate._paint_loading(None, option, _NoSizeHintIndex(loaded=False))
    app.processEvents()


def test_padded_units_scale_extends_to_cover_displayed_dims():
    # Bug 28: a 2-tuple units scale on a layer whose displayed dims are [1, 2]
    # would overrun at index 2; pad with the no-calibration default (1.0).
    from itasc.napari._napari_compat import _padded_units_scale

    assert _padded_units_scale((1.0, 1.0), [1, 2]) == (1.0, 1.0, 1.0)
    # Already long enough -> returned unchanged (and as a tuple).
    assert _padded_units_scale((2.0, 3.0, 4.0), [1, 2]) == (2.0, 3.0, 4.0)
    assert _padded_units_scale((0.65, 0.65), [0, 1]) == (0.65, 0.65)
    # No displayed dims -> nothing to cover.
    assert _padded_units_scale((1.0,), []) == (1.0,)


def test_units_scale_guard_is_installed_on_import():
    sys.modules.pop("itasc.napari", None)
    importlib.import_module("itasc.napari")
    from napari._vispy.layers.base import VispyBaseLayer

    assert getattr(VispyBaseLayer, "_itasc_units_scale_guard", False) is True
