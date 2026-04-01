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

from ..utils.structures import TissueGraphDataset, TissueGraphTimeSeries


class ViewerState(QObject):
    """Shared state bound to a single napari viewer.

    Signals
    -------
    dataset_changed
        Emitted whenever the dataset is replaced or a tissue is
        added / removed.  Listeners should refresh their UI.
    metadata_changed
        Emitted when pixel_size, time_interval, or condition changes.
    """

    dataset_changed = Signal()
    metadata_changed = Signal()
    project_changed = Signal()   # emitted when project_path changes

    def __init__(self, viewer) -> None:
        super().__init__()
        self._viewer_ref = weakref.ref(viewer)
        self._dataset: Optional[TissueGraphDataset] = None
        self._pixel_size: Optional[float] = None
        self._time_interval: Optional[float] = None
        self._condition: str = ""
        self._project_path: Optional[str] = None  # current .h5 file path

    # -- viewer accessor ---------------------------------------------------

    @property
    def viewer(self):
        v = self._viewer_ref()
        if v is None:
            raise RuntimeError("Viewer has been closed")
        return v

    # -- metadata properties -----------------------------------------------

    @property
    def pixel_size(self) -> Optional[float]:
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, value: Optional[float]) -> None:
        self._pixel_size = value
        self.metadata_changed.emit()

    @property
    def time_interval(self) -> Optional[float]:
        return self._time_interval

    @time_interval.setter
    def time_interval(self, value: Optional[float]) -> None:
        self._time_interval = value
        self.metadata_changed.emit()

    @property
    def condition(self) -> str:
        return self._condition

    @condition.setter
    def condition(self, value: str) -> None:
        self._condition = value or ""
        self.metadata_changed.emit()

    # -- project path ------------------------------------------------------

    @property
    def project_path(self) -> Optional[str]:
        return self._project_path

    @project_path.setter
    def project_path(self, value: Optional[str]) -> None:
        self._project_path = value
        self.project_changed.emit()

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
