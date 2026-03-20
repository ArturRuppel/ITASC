"""Per-viewer shared state for inter-widget communication.

Each napari viewer gets one ViewerState instance, accessed via
``get_state(viewer)``.  Widgets use it to share the dataset and
coordinate updates without direct references to each other.

Typical usage inside a widget::

    from .registry import get_state

    state = get_state(self.viewer)
    state.dataset_changed.connect(self._on_dataset_changed)
    state.dataset = my_dataset          # triggers signal
    state.add_tissue(series)            # triggers signal
"""
from __future__ import annotations

import weakref
from typing import Dict, Optional

from qtpy.QtCore import QObject, Signal

from ..structures import TissueGraphDataset, TissueGraphTimeSeries


class ViewerState(QObject):
    """Shared state bound to a single napari viewer.

    Signals
    -------
    dataset_changed
        Emitted whenever the dataset is replaced or a tissue is
        added / removed.  Listeners should refresh their UI.
    """

    dataset_changed = Signal()

    def __init__(self, viewer) -> None:
        super().__init__()
        self._viewer_ref = weakref.ref(viewer)
        self._dataset: Optional[TissueGraphDataset] = None

    # -- viewer accessor ---------------------------------------------------

    @property
    def viewer(self):
        v = self._viewer_ref()
        if v is None:
            raise RuntimeError("Viewer has been closed")
        return v

    # -- dataset property --------------------------------------------------

    @property
    def dataset(self) -> Optional[TissueGraphDataset]:
        return self._dataset

    @dataset.setter
    def dataset(self, value: Optional[TissueGraphDataset]) -> None:
        self._dataset = value
        self.dataset_changed.emit()

    # -- convenience mutators that emit dataset_changed --------------------

    def ensure_dataset(self, **kwargs) -> TissueGraphDataset:
        """Return the existing dataset or create a new one."""
        if self._dataset is None:
            self._dataset = TissueGraphDataset(**kwargs)
            self.dataset_changed.emit()
        return self._dataset

    def add_tissue(self, series: TissueGraphTimeSeries) -> int:
        """Add *series* to the dataset and emit *dataset_changed*.

        Creates the dataset first if it does not exist yet.
        """
        ds = self.ensure_dataset()
        tid = ds.add_tissue(series)
        self.dataset_changed.emit()
        return tid

    def remove_tissue(self, tissue_id: int) -> None:
        if self._dataset is None:
            return
        self._dataset.remove_tissue(tissue_id)
        self.dataset_changed.emit()


# -- module-level registry ------------------------------------------------

_states: Dict[int, ViewerState] = {}


def get_state(viewer) -> ViewerState:
    """Return the shared :class:`ViewerState` for *viewer*, creating it
    on first access.  The entry is automatically removed when the viewer
    is garbage-collected.
    """
    vid = id(viewer)
    if vid in _states:
        state = _states[vid]
        # Verify the viewer is still alive
        try:
            _ = state.viewer
        except RuntimeError:
            del _states[vid]
        else:
            return state

    state = ViewerState(viewer)
    # Remove the entry when the viewer is garbage-collected.
    weakref.finalize(viewer, _states.pop, vid, None)
    _states[vid] = state
    return state
