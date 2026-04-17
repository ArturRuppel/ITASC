"""Napari dock widget for cellflow.

All pipeline stages are rendered as a vertical accordion of CollapsibleSections.
"""
import logging

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .registry import get_state
from .widgets import CollapsibleSection

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

        # ========== Accordion sections (migrated from tabs) ==========
        from .ultrack_widgets.data_prep import DataPrepWidget
        from .ultrack_widgets.cellpose import CellposeWidget
        from .ultrack_widgets.ultrack_widget import UltrackAnalysisWidget

        # 0_input — accordion section, not a tab
        self._data_prep_widget = DataPrepWidget(self.viewer)
        self._data_prep_section = CollapsibleSection(
            "Prepare Input Data", self._data_prep_widget, expanded=False
        )
        _plugin_layout.addWidget(self._data_prep_section)

        # 1_cellpose — accordion section with two sub-sections (3D Nucleus / 2D Cell)
        self._cellpose_tab = CellposeWidget(self.viewer)
        self._cellpose_section = CollapsibleSection(
            "Cellpose", self._cellpose_tab, expanded=False
        )
        _plugin_layout.addWidget(self._cellpose_section)

        # 2_ultrack — accordion section with two sub-sections (Contours / Tracking)
        self._ultrack_tab = UltrackAnalysisWidget(self.viewer)
        self._ultrack_section = CollapsibleSection(
            "Ultrack", self._ultrack_tab, expanded=False
        )
        _plugin_layout.addWidget(self._ultrack_section)

        from .tracking_correction_widget import TrackingCorrectionWidget
        from .ultrack_widgets.flow_watershed import FlowGuidedSegmentationWidget
        from .edge_analysis_widget import EdgeAnalysisWidget
        from .forces_widget import ForcesWidget

        # 3_correction — LapTrack re-tracking + manual correction, shared layer
        self._tracking_correction_widget = TrackingCorrectionWidget(self.viewer)
        self._ultrack_tab.labels_loaded.connect(self._tracking_correction_widget._set_data_layer)
        self._correction_section = CollapsibleSection(
            "Correction", self._tracking_correction_widget, expanded=False
        )
        _plugin_layout.addWidget(self._correction_section)

        # 4_cell_segmentation
        self._cell_seg_tab = FlowGuidedSegmentationWidget(self.viewer)
        self._cell_seg_section = CollapsibleSection(
            "Flow Watershed", self._cell_seg_tab, expanded=False
        )
        _plugin_layout.addWidget(self._cell_seg_section)

        # 5_analysis
        self._edge_analysis_widget = EdgeAnalysisWidget(self.viewer)
        self._edge_analysis_section = CollapsibleSection(
            "Edge Analysis", self._edge_analysis_widget, expanded=False
        )
        _plugin_layout.addWidget(self._edge_analysis_section)

        # ForSys (downstream, no pipeline dir yet)
        self._forces_widget = ForcesWidget(self.viewer)
        self._forces_section = CollapsibleSection(
            "ForSys", self._forces_widget, expanded=False
        )
        _plugin_layout.addWidget(self._forces_section)

        # Dataset collapsible section sits at the very bottom.
        _plugin_layout.addWidget(self._project_panel.dataset_widget)

        # Stretch absorbs leftover space so sections pack to the top and never expand.
        _plugin_layout.addStretch(1)

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

    # All sections: maps base title → CollapsibleSection
    @property
    def _accordion_sections(self) -> "dict[str, CollapsibleSection]":
        return {
            "Prepare Input Data": self._data_prep_section,
            "Cellpose":           self._cellpose_section,
            "Ultrack":            self._ultrack_section,
            "Correction":         self._correction_section,
            "Flow Watershed":      self._cell_seg_section,
            "Edge Analysis":      self._edge_analysis_section,
            "ForSys":             self._forces_section,
        }

    def _refresh_tab_badges(self) -> None:
        stage_keys = self._TAB_STAGE_KEYS
        project_dir = self._state.project_dir

        if project_dir is None:
            for base_title, section in self._accordion_sections.items():
                section.set_title(base_title)
            return

        try:
            from cellflow.core.paths import manifest_path
            from cellflow.core.manifest import PipelineManifest
            manifest = PipelineManifest.load(manifest_path(project_dir, 0))
        except Exception:
            return

        priority = ["failed", "running", "stale", "pending", "complete"]

        for base_title, section in self._accordion_sections.items():
            keys = stage_keys.get(base_title)
            if keys is None:
                continue
            statuses = [
                manifest.stages[k].status if k in manifest.stages else "pending"
                for k in keys
            ]
            agg = next((s for s in priority if s in statuses), "pending")
            section.set_title(base_title + self._STATUS_BADGE.get(agg, ""))
