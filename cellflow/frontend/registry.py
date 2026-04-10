"""Per-viewer shared state for inter-widget communication.

Each napari viewer gets one ViewerState instance, accessed via
``get_state(viewer)``.  Widgets use it to share the dataset and
coordinate updates without direct references to each other.

Typical usage inside a widget::

    from .registry import get_state

    state = get_state(self.viewer)
    state.tissue_changed.connect(self._on_tissue_changed)
    state.catalog_changed.connect(self._on_catalog_changed)
    state.set_tissue_series(series)   # triggers tissue_changed
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from qtpy.QtCore import QObject, Signal

from ..utils.structures import TissueGraphDataset, TissueGraphTimeSeries


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TissueData:
    """The single active working tissue (in-memory, independent of napari layers)."""
    image:   Optional[object]                   = None  # np.ndarray (T,H,W) or None
    labels:  Optional[object]                   = None  # np.ndarray (T,H,W) or None
    series:  Optional[TissueGraphTimeSeries]    = None
    forsys:  Optional[object]                   = None  # ForSys/pressure inference result
    path:    Optional[str]                      = None  # .h5 path if saved, else None

    image2:          Optional[object]  = None  # np.ndarray secondary channel for two-channel segmentation
    nuclear_labels:  Optional[object]  = None  # np.ndarray (T,H,W) nuclear segmentation
    # names of the linked napari layers (for regeneration / sync)
    image_layer:         Optional[str]  = None
    image2_layer:        Optional[str]  = None
    labels_layer:        Optional[str]  = None
    nuclear_labels_layer: Optional[str] = None
    forsys_layer:        Optional[str]  = None

    # project-level metadata read from the H5 file
    pixel_size:    Optional[float]  = None
    time_interval: Optional[float]  = None
    condition:     str              = ""


@dataclass
class CatalogEntry:
    """One row in the dataset catalog."""
    path:         str
    display_name: str  = ""
    condition:    str  = ""
    # summary written at add-time; never needs to open H5 files to show the table
    summary: Dict = field(default_factory=dict)
    # e.g. {"n_frames": 48, "avg_cells": 450, "n_t1_events": 12, "n_trajectories": 230}

    # runtime cache — populated on first access, never persisted
    _series_cache: Optional[TissueGraphTimeSeries] = field(default=None, repr=False)


@dataclass
class DatasetCatalog:
    """Ordered collection of saved tissue H5 files, lazy-loaded."""
    entries:       List[CatalogEntry] = field(default_factory=list)
    path:          Optional[str]      = None   # .cfproj file path
    pixel_size:    Optional[float]    = None
    time_interval: Optional[float]    = None
    condition:     str                = ""

    def get_series(self, index: int) -> TissueGraphTimeSeries:
        """Load from H5 if not cached, return from cache otherwise."""
        from ..utils.io import load_tissue
        entry = self.entries[index]
        if entry._series_cache is None:
            result = load_tissue(entry.path)
            entry._series_cache = result.series
        return entry._series_cache


# ---------------------------------------------------------------------------
# ViewerState
# ---------------------------------------------------------------------------

class ViewerState(QObject):
    """Shared state bound to a single napari viewer.

    Signals
    -------
    tissue_changed
        Emitted whenever any field of the active TissueData changes.
    catalog_changed
        Emitted whenever the DatasetCatalog changes (entry added/removed).
    metadata_changed
        Emitted when pixel_size, time_interval, or condition changes.
    """

    tissue_changed         = Signal()
    catalog_changed        = Signal()
    metadata_changed       = Signal()
    nuclear_labels_changed = Signal()

    # --- legacy signals kept for any remaining call-sites ---
    dataset_changed  = Signal()
    project_changed  = Signal()
    preview_changed  = Signal()

    def __init__(self, viewer) -> None:
        super().__init__()
        self._viewer_ref = weakref.ref(viewer)
        self._tissue:  TissueData     = TissueData()
        self._catalog: DatasetCatalog = DatasetCatalog()
        self._pixel_size: Optional[float] = None
        self._time_interval: Optional[float] = None
        self._condition: str = ""
        self._project_path: Optional[str] = None

        # --- legacy shim backing store ---
        self._dataset: Optional[TissueGraphDataset] = None
        self._preview_series: Optional[TissueGraphTimeSeries] = None

    # -- viewer accessor ---------------------------------------------------

    @property
    def viewer(self):
        v = self._viewer_ref()
        if v is None:
            raise RuntimeError("Viewer has been closed")
        return v

    # -- TissueData --------------------------------------------------------

    @property
    def tissue(self) -> TissueData:
        return self._tissue

    def set_tissue_labels(self, arr, layer_name: Optional[str] = None) -> None:
        self._tissue.labels = arr
        if layer_name is not None:
            self._tissue.labels_layer = layer_name
        self.tissue_changed.emit()

    def set_tissue_series(self, series: Optional[TissueGraphTimeSeries]) -> None:
        self._tissue.series = series
        self.tissue_changed.emit()
        # keep legacy preview_series in sync for widgets not yet migrated
        self._preview_series = series
        self.preview_changed.emit()

    def set_tissue_image(self, arr, layer_name: Optional[str] = None) -> None:
        self._tissue.image = arr
        if layer_name is not None:
            self._tissue.image_layer = layer_name
        self.tissue_changed.emit()

    def set_tissue_image2(self, arr, layer_name: Optional[str] = None) -> None:
        self._tissue.image2 = arr
        if layer_name is not None:
            self._tissue.image2_layer = layer_name
        self.tissue_changed.emit()

    def set_tissue_nuclear_labels(self, arr, layer_name: Optional[str] = None) -> None:
        self._tissue.nuclear_labels = arr
        if layer_name is not None:
            self._tissue.nuclear_labels_layer = layer_name
        self.nuclear_labels_changed.emit()
        self.tissue_changed.emit()

    def clear_tissue(self) -> None:
        self._tissue = TissueData()
        self.tissue_changed.emit()

    # -- DatasetCatalog ----------------------------------------------------

    @property
    def catalog(self) -> DatasetCatalog:
        return self._catalog

    def add_to_catalog(self, entry: CatalogEntry) -> None:
        self._catalog.entries.append(entry)
        self.catalog_changed.emit()
        self.dataset_changed.emit()

    def remove_from_catalog(self, index: int) -> None:
        del self._catalog.entries[index]
        self.catalog_changed.emit()
        self.dataset_changed.emit()

    # -- metadata properties -----------------------------------------------

    @property
    def pixel_size(self) -> Optional[float]:
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, value: Optional[float]) -> None:
        self._pixel_size = value
        self._catalog.pixel_size = value
        self.metadata_changed.emit()

    @property
    def time_interval(self) -> Optional[float]:
        return self._time_interval

    @time_interval.setter
    def time_interval(self, value: Optional[float]) -> None:
        self._time_interval = value
        self._catalog.time_interval = value
        self.metadata_changed.emit()

    @property
    def condition(self) -> str:
        return self._condition

    @condition.setter
    def condition(self, value: str) -> None:
        self._condition = value or ""
        self._catalog.condition = value or ""
        self.metadata_changed.emit()

    # -- project path (legacy, kept for ProjectPanel compat) ---------------

    @property
    def project_path(self) -> Optional[str]:
        return self._project_path

    @project_path.setter
    def project_path(self, value: Optional[str]) -> None:
        self._project_path = value
        self.project_changed.emit()

    # -- legacy: preview_series --------------------------------------------

    @property
    def preview_series(self) -> Optional[TissueGraphTimeSeries]:
        return self._preview_series

    @preview_series.setter
    def preview_series(self, value: Optional[TissueGraphTimeSeries]) -> None:
        self._preview_series = value
        self._tissue.series = value
        self.preview_changed.emit()
        self.tissue_changed.emit()

    # -- legacy: dataset ---------------------------------------------------

    @property
    def dataset(self) -> Optional[TissueGraphDataset]:
        return self._dataset

    @dataset.setter
    def dataset(self, value: Optional[TissueGraphDataset]) -> None:
        self._dataset = value
        self.dataset_changed.emit()

    def ensure_dataset(self, **kwargs) -> TissueGraphDataset:
        if self._dataset is None:
            self._dataset = TissueGraphDataset(**kwargs)
            self.dataset_changed.emit()
        return self._dataset

    def add_tissue(self, series: TissueGraphTimeSeries) -> int:
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
        try:
            _ = state.viewer
        except RuntimeError:
            del _states[vid]
        else:
            return state

    state = ViewerState(viewer)
    weakref.finalize(viewer, _states.pop, vid, None)
    _states[vid] = state
    return state
