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
    sys.modules.pop("cellflow.napari", None)
    importlib.import_module("cellflow.napari")
    from napari._qt.containers._layer_delegate import LayerDelegate

    delegate = LayerDelegate()
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 200, 34)

    delegate._paint_loading(None, option, _NoSizeHintIndex(loaded=False))
    app.processEvents()
