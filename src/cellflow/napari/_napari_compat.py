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

    patch_vispy_units_scale_guard()


def patch_vispy_units_scale_guard() -> None:
    """Stop a canvas-draw ``IndexError`` from a layer whose per-axis units scale
    is shorter than its displayed dims.

    napari's ``VispyBaseLayer._on_matrix_change`` indexes
    ``self._world_to_layer_units_scale`` by ``self.layer._slice_input.displayed``.
    When physical units (e.g. an OME-TIFF's ``(pixel, pixel)``) reach a layer with
    more dimensions than the units tuple — a calibrated 2D stack shown in a 3D
    viewer — ``_recalculate_units_scale``'s ``strict=False`` zip yields a units
    scale shorter than ``dims_displayed``, and indexing ``[2]`` overruns:

        IndexError: tuple index out of range  (canvas.py → _on_matrix_change)

    The unresolved axes have no calibration, so their scale is just napari's
    default of ``1.0``. Pad the units scale up to the displayed-dim extent with
    ``1.0`` before delegating — semantically the identity, so the only behaviour
    change is that the draw no longer crashes."""
    try:
        from napari._vispy.layers.base import VispyBaseLayer
    except Exception:  # pragma: no cover - napari layout changed / unavailable
        return
    if getattr(VispyBaseLayer, "_cellflow_units_scale_guard", False):
        return

    original_on_matrix_change = VispyBaseLayer._on_matrix_change

    def _guarded_on_matrix_change(self):
        try:
            self._world_to_layer_units_scale = _padded_units_scale(
                self._world_to_layer_units_scale, self.layer._slice_input.displayed
            )
        except Exception:  # pragma: no cover - never let the guard itself break draw
            pass
        return original_on_matrix_change(self)

    VispyBaseLayer._on_matrix_change = _guarded_on_matrix_change
    VispyBaseLayer._cellflow_units_scale_guard = True


def _padded_units_scale(scale, displayed) -> tuple:
    """*scale* extended with ``1.0`` (the no-calibration default) so every index
    in *displayed* is in range; unchanged when it already covers them."""
    scale = tuple(scale)
    needed = (max(displayed) + 1) if len(displayed) else 0
    if len(scale) >= needed:
        return scale
    return scale + (1.0,) * (needed - len(scale))

