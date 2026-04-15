"""
Monkey-patches for napari bugs that affect CellFlow.

Each patch is applied once at import time and is guarded so it is a no-op if
the upstream code is already fixed (i.e. the target attribute no longer exists
or has been changed).
"""

# ── napari bug: LayerDelegate._paint_loading crashes when a model index
#    becomes stale during layer removal.  index.data(SizeHintRole) returns None
#    for a stale index, and the original code calls .height() on it.
#    Upstream fix: add a None-guard before using the return value.
try:
    from napari._qt.containers._layer_delegate import LayerDelegate

    _orig_paint_loading = LayerDelegate._paint_loading

    def _patched_paint_loading(self, painter, option, index):
        from PyQt6.QtCore import Qt
        if index.data(Qt.ItemDataRole.SizeHintRole) is None:
            return
        _orig_paint_loading(self, painter, option, index)

    LayerDelegate._paint_loading = _patched_paint_loading
except Exception:
    pass
