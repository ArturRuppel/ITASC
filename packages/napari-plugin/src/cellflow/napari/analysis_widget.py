"""Napari dock widget for cellflow.

Top-level QTabWidget that assembles all pipeline and analysis tabs.
Edge Analysis logic lives in EdgeAnalysisWidget; other tabs are
imported from their own modules.
"""
import logging
from typing import Optional

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .registry import get_state

logger = logging.getLogger(__name__)


class CellFlowWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._state = get_state(napari_viewer)

        # Poll manifest every 5 s to update pipeline tab status badges.
        self._badge_timer = QTimer(self)
        self._badge_timer.setInterval(5000)
        self._badge_timer.timeout.connect(self._refresh_tab_badges)
        self._badge_timer.start()

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer_layout)

        # Single scroll area around the whole plugin
        self._outer_scroll = QScrollArea()
        self._outer_scroll.setWidgetResizable(True)
        self._outer_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer_layout.addWidget(self._outer_scroll)

        _plugin_root = QWidget()
        _plugin_layout = QVBoxLayout(_plugin_root)
        _plugin_layout.setContentsMargins(0, 0, 0, 0)
        _plugin_layout.setSpacing(0)
        self._outer_scroll.setWidget(_plugin_root)

        # ── Project panel (fixed strip above all tabs) ────────────────────
        from .project_panel import ProjectPanel
        self._project_panel = ProjectPanel(self.viewer, self._state)
        _plugin_layout.addWidget(self._project_panel)

        self.tab_widget = QTabWidget()
        _plugin_layout.addWidget(self.tab_widget)

        _plugin_layout.addWidget(self._project_panel.dataset_widget)

        # ========== Pipeline stage tabs (in dataflow order) ==========
        from .ultrack_widgets.data_prep import DataPrepWidget
        from .ultrack_widgets.cellpose import CellposeWidget
        from .ultrack_widgets.ultrack_widget import UltrackAnalysisWidget
        from .correction_widget import CorrectionWidget
        from .tracking_widget import TrackingTab
        from .ultrack_widgets.flow_watershed import FlowGuidedSegmentationWidget
        from .edge_analysis_widget import EdgeAnalysisWidget
        from .forces_widget import ForcesWidget

        # 0_input
        self._data_prep_tab = DataPrepWidget(self.viewer)
        self.tab_widget.addTab(self._data_prep_tab, "Data Prep")

        # 1_cellpose/nucleus + 1_cellpose/cell
        self._cellpose_tab = CellposeWidget(self.viewer)
        self.tab_widget.addTab(self._cellpose_tab, "Cellpose")

        # 2_ultrack (contours intermediate + tracking)
        self._ultrack_tab = UltrackAnalysisWidget(self.viewer)
        self.tab_widget.addTab(self._ultrack_tab, "Ultrack")

        # 3_correction (manual correction loop)
        self._correction_widget = CorrectionWidget(self.viewer)
        self.tab_widget.addTab(self._correction_widget, "Correction")

        # LapTrack retracking — part of the correction loop, no stage dir
        self._tracking_tab = TrackingTab(self.viewer)
        self.tab_widget.addTab(self._tracking_tab, "Tracking")

        # 4_cell_segmentation
        self._cell_seg_tab = FlowGuidedSegmentationWidget(self.viewer)
        self.tab_widget.addTab(self._cell_seg_tab, "Cell Seg")

        # 5_analysis
        self._edge_analysis_widget = EdgeAnalysisWidget(self.viewer)
        self.tab_widget.addTab(self._edge_analysis_widget, "Edge Analysis")

        # ForSys (downstream, no pipeline dir yet)
        self._forces_widget = ForcesWidget(self.viewer)
        self.tab_widget.addTab(self._forces_widget, "ForSys")

    def _connect_signals(self):
        self._state.pipeline_schema_changed.connect(self._refresh_tab_badges)

    # ------------------------------------------------------------------
    # Pipeline tab status badges
    # ------------------------------------------------------------------

    @property
    def _TAB_STAGE_KEYS(self) -> dict:
        from ._plugin import TAB_STAGE_KEYS
        return TAB_STAGE_KEYS
    _STATUS_BADGE = {
        "complete": " ✓",
        "running":  " ↻",
        "failed":   " ✗",
        "stale":    " ⚠",
        "pending":  "",
    }

    def _refresh_tab_badges(self) -> None:
        project_dir = self._state.project_dir
        if project_dir is None:
            for i in range(self.tab_widget.count()):
                text = self.tab_widget.tabText(i)
                for badge in self._STATUS_BADGE.values():
                    if badge and text.endswith(badge):
                        self.tab_widget.setTabText(i, text[: -len(badge)])
            return

        try:
            from cellflow.core.paths import manifest_path
            from cellflow.core.manifest import PipelineManifest
            manifest = PipelineManifest.load(manifest_path(project_dir, 0))
        except Exception:
            return

        for i in range(self.tab_widget.count()):
            raw_title = self.tab_widget.tabText(i)
            base_title = raw_title
            for badge in self._STATUS_BADGE.values():
                if badge and base_title.endswith(badge):
                    base_title = base_title[: -len(badge)]
                    break

            keys = self._TAB_STAGE_KEYS.get(base_title)
            if keys is None:
                continue

            statuses = [
                manifest.stages[k].status
                if k in manifest.stages else "pending"
                for k in keys
            ]
            priority = ["failed", "running", "stale", "pending", "complete"]
            agg = next(
                (s for s in priority if s in statuses),
                "pending",
            )
            badge = self._STATUS_BADGE.get(agg, "")
            self.tab_widget.setTabText(i, base_title + badge)
