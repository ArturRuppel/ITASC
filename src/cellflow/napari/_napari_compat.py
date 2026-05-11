"""Compatibility patches for napari/Qt integration edge cases."""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap


def _index_size_hint(index):
    if index is None or (hasattr(index, "isValid") and not index.isValid()):
        return None
    size_hint = index.data(Qt.ItemDataRole.SizeHintRole)
    if size_hint is None or not hasattr(size_hint, "height"):
        return None
    return size_hint


def patch_napari_layer_delegate() -> None:
    """Make napari's layer delegate tolerate transient indexes without size hints."""
    try:
        from napari._qt.containers._layer_delegate import LayerDelegate
        from napari._qt.containers.qt_layer_model import LoadedRole, ThumbnailRole
    except Exception:
        return

    if getattr(LayerDelegate, "_cellflow_missing_size_hint_patch", False):
        return

    def _paint_loading(self, painter, option, index):
        loaded = index.data(LoadedRole)
        if loaded:
            return

        size_hint = _index_size_hint(index)
        if size_hint is None:
            return

        self._load_movie.start()
        load_rect = option.rect.translated(4, 8)
        h = size_hint.height() - 16
        load_rect.setWidth(h)
        load_rect.setHeight(h)
        painter.drawPixmap(load_rect, self._load_movie.currentPixmap())

    def _paint_thumbnail(self, painter, option, index):
        loaded = index.data(LoadedRole)
        if not loaded:
            return

        size_hint = _index_size_hint(index)
        if size_hint is None:
            return

        all_loaded = index.model().sourceModel().all_loaded()
        if all_loaded:
            self._load_movie.setPaused(True)

        thumb_rect = option.rect.translated(-2, 2)
        h = size_hint.height() - 4
        thumb_rect.setWidth(h)
        thumb_rect.setHeight(h)
        image = index.data(ThumbnailRole)
        painter.drawPixmap(thumb_rect, QPixmap.fromImage(image))

    LayerDelegate._paint_loading = _paint_loading
    LayerDelegate._paint_thumbnail = _paint_thumbnail
    LayerDelegate._cellflow_missing_size_hint_patch = True

