"""
Main napari plugin widget for napariSegTrack.

Tabs:
  1. Segmentation – frame-by-frame or full-stack Cellpose segmentation
                    (Single Channel / Two Channel)
                    with Cellpose GUI correction support.
  2. Tracking     – LapTrack-based tracking from a Labels layer,
                    temporal correction.
"""

import napari
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget

from cellflow.segtrack._segmentation_tab import SegmentationTab
from cellflow.segtrack._tracking_tab import TrackingTab


class SegTrackWidget(QWidget):
    """Tabbed plugin widget: Segmentation + Tracking."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer

        seg_tab = SegmentationTab(viewer)
        tabs = QTabWidget()
        tabs.addTab(seg_tab,                        "Segmentation")
        tabs.addTab(TrackingTab(viewer, seg_tab),   "Tracking")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(tabs)
