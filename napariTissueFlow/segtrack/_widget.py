"""
Main napari plugin widget for napariSegTrack.

Contains three tabs:
  1. Segmentation – frame-by-frame or full-stack Cellpose segmentation
                    (Single Channel / Two Channel)
                    with Cellpose GUI correction support.
  2. Cell Bodies  – Voronoi expansion from nuclear Labels to cell body Labels,
                    with optional Lloyd's relaxation for regular cell shapes.
  3. Tracking     – LapTrack-based tracking from a Labels (or Image) layer,
                    optional Voronoi expansion, temporal correction.
"""

import napari
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from napariTissueFlow.segtrack._segmentation_tab import SegmentationTab
from napariTissueFlow.segtrack._voronoi_tab import VoronoiTab
from napariTissueFlow.segtrack._tracking_tab import TrackingTab


class SegTrackWidget(QWidget):
    """Tabbed plugin widget: Segmentation + Cell Bodies + Tracking."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        tabs = QTabWidget()
        tabs.addTab(SegmentationTab(viewer), "Segmentation")
        tabs.addTab(VoronoiTab(viewer),      "Cell Bodies")
        tabs.addTab(TrackingTab(viewer),     "Tracking")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(tabs)
