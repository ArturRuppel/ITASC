"""Napari dock widget for cellflow.

The workflow is rendered as a vertical accordion of the canonical six stages.
"""
import json
import logging
from pathlib import Path

from qtpy.QtCore import Qt, QSize, QTimer
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .log_viewer import StageLogViewer
from .registry import get_state
from .widgets import CollapsibleSection

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "cellflow_config.json"
_CONFIG_VERSION = 2


class CellFlowWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._state = get_state(napari_viewer)
        self._last_loaded_project_dir = None  # track which project's config was loaded

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

        # ── Top button rows: New/Open Project | Save/Load Config ─────────
        top_btn_row1 = QHBoxLayout()
        top_btn_row1.setContentsMargins(6, 2, 6, 0)
        top_btn_row1.setSpacing(4)
        top_btn_row1.addWidget(self._project_panel._new_project_btn)
        top_btn_row1.addWidget(self._project_panel._open_project_btn)
        _plugin_layout.addLayout(top_btn_row1)

        top_btn_row2 = QHBoxLayout()
        top_btn_row2.setContentsMargins(6, 4, 6, 2)
        top_btn_row2.setSpacing(4)
        self._save_cfg_btn = QPushButton("Save Config")
        self._save_cfg_btn.clicked.connect(self._save_config)
        top_btn_row2.addWidget(self._save_cfg_btn)
        self._load_cfg_btn = QPushButton("Load Config")
        self._load_cfg_btn.clicked.connect(self._load_config)
        top_btn_row2.addWidget(self._load_cfg_btn)
        self._save_cfg_as_btn = QPushButton("Save Config As\u2026")
        self._save_cfg_as_btn.clicked.connect(self._save_config_as)
        top_btn_row2.addWidget(self._save_cfg_as_btn)
        self._load_cfg_from_btn = QPushButton("Load Config From\u2026")
        self._load_cfg_from_btn.clicked.connect(self._load_config_from)
        top_btn_row2.addWidget(self._load_cfg_from_btn)
        _plugin_layout.addLayout(top_btn_row2)

        _plugin_layout.addWidget(self._project_panel)

        # ── Shared log viewer (passed to all subwidgets) ─────────────────
        self._log_viewer = StageLogViewer(self._state, expanded=True)

        # ========== Accordion sections ==========
        from .ultrack_widgets.data_prep import DataPrepWidget
        from .ultrack_widgets.cellpose import CellposeWidget
        from .ultrack_widgets.nucleus_hypotheses_widget import UltrackAnalysisWidget
        from .ultrack_widgets.seeded_tracker_widget import SeededTrackerWidget

        self._data_prep_widget = DataPrepWidget(self.viewer, log_viewer=self._log_viewer)
        self._data_prep_section = CollapsibleSection(
            "Prepare Input Data", self._data_prep_widget, expanded=False
        )
        _plugin_layout.addWidget(self._data_prep_section)

        self._cellpose_tab = CellposeWidget(self.viewer, log_viewer=self._log_viewer)
        self._cellpose_section = CollapsibleSection(
            "Cellpose Cluster", self._cellpose_tab, expanded=False
        )
        _plugin_layout.addWidget(self._cellpose_section)

        self._ultrack_tab = UltrackAnalysisWidget(self.viewer, log_viewer=self._log_viewer)
        self._ultrack_section = CollapsibleSection(
            "Nucleus Hypotheses", self._ultrack_tab, expanded=False
        )
        _plugin_layout.addWidget(self._ultrack_section)

        self._seeded_tracker_tab = SeededTrackerWidget(self.viewer, log_viewer=self._log_viewer)
        self._seeded_tracker_section = CollapsibleSection(
            "Seeded Tracking", self._seeded_tracker_tab, expanded=False
        )
        _plugin_layout.addWidget(self._seeded_tracker_section)

        from .tracking_correction_widget import TrackingCorrectionWidget
        from .ultrack_widgets.cell_segmentation import CellSegmentationWidget
        from .ultrack_widgets.seeded_watershed import SeededWatershedWidget
        from .edge_analysis_widget import EdgeAnalysisWidget
        from .forces_widget import ForcesWidget

        self._tracking_correction_widget = TrackingCorrectionWidget(self.viewer)
        self._ultrack_tab.labels_loaded.connect(self._tracking_correction_widget._set_data_layer)
        self._correction_section = CollapsibleSection(
            "Correction", self._tracking_correction_widget, expanded=False
        )
        _plugin_layout.addWidget(self._correction_section)

        self._cell_seg_tab = CellSegmentationWidget(self.viewer, log_viewer=self._log_viewer)
        self._seeded_ws_tab = SeededWatershedWidget(self.viewer, log_viewer=self._log_viewer)
        self._edge_analysis_widget = EdgeAnalysisWidget(self.viewer)
        self._forces_widget = ForcesWidget(self.viewer)

        cell_ultrack_content = QWidget()
        cell_ultrack_layout = QVBoxLayout(cell_ultrack_content)
        cell_ultrack_layout.setContentsMargins(0, 0, 0, 0)
        cell_ultrack_layout.setSpacing(6)
        cell_ultrack_layout.addWidget(self._cell_seg_tab)
        cell_ultrack_layout.addWidget(self._seeded_ws_tab)
        self._cell_ultrack_section = CollapsibleSection(
            "Cell Ultrack", cell_ultrack_content, expanded=False
        )
        _plugin_layout.addWidget(self._cell_ultrack_section)

        analysis_content = QWidget()
        analysis_layout = QVBoxLayout(analysis_content)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_layout.setSpacing(6)
        analysis_layout.addWidget(self._edge_analysis_widget)
        analysis_layout.addWidget(self._forces_widget)
        self._analysis_section = CollapsibleSection(
            "Analysis", analysis_content, expanded=False
        )
        _plugin_layout.addWidget(self._analysis_section)

        # Dataset collapsible section
        _plugin_layout.addWidget(self._project_panel.dataset_widget)

        # ── Shared log at the bottom ──────────────────────────────────────
        _plugin_layout.addWidget(self._log_viewer)

        _plugin_layout.addStretch(1)

    def sizeHint(self) -> QSize:
        return QSize(550, super().sizeHint().height())

    def _connect_signals(self):
        self._state.pipeline_schema_changed.connect(self._refresh_tab_badges)
        self._state.pipeline_schema_changed.connect(self._load_config_if_exists)

        # Auto-save config whenever any pipeline widget starts a run
        self._data_prep_widget.run_started.connect(self._autosave_config)
        self._cellpose_tab.run_started.connect(self._autosave_config)
        self._ultrack_tab.run_started.connect(self._autosave_config)
        self._seeded_tracker_tab.run_started.connect(self._autosave_config)
        self._cell_seg_tab.run_started.connect(self._autosave_config)
        self._seeded_ws_tab.run_started.connect(self._autosave_config)

    # ------------------------------------------------------------------
    # Config save / load
    # ------------------------------------------------------------------

    def _config_path(self) -> Path | None:
        project_dir = self._state.project_dir
        if project_dir is None:
            return None
        return Path(project_dir) / _CONFIG_FILENAME

    def _collect_config(self) -> dict:
        cfg = {"version": _CONFIG_VERSION}
        cfg["data_prep"] = self._data_prep_widget.get_params()
        cfg["cellpose_cluster"] = self._cellpose_tab.get_params()
        cfg["nucleus_hypotheses"] = self._ultrack_tab.get_params()
        cfg["correction"] = self._tracking_correction_widget.get_params()
        cfg["cell_ultrack"] = {
            "cell_segmentation": self._cell_seg_tab.get_params(),
            "seeded_watershed": self._seeded_ws_tab.get_params(),
        }
        cfg["analysis"] = {
            "edge_analysis": self._edge_analysis_widget.get_params(),
            "forces": self._forces_widget.get_params(),
        }
        return cfg

    def _apply_config(self, data: dict) -> None:
        if "data_prep" in data:
            self._data_prep_widget.set_params(data["data_prep"])
        if "cellpose_cluster" in data:
            self._cellpose_tab.set_params(data["cellpose_cluster"])
        else:
            self._cellpose_tab.set_params({
                k: data[k] for k in ("cellpose_nucleus", "cellpose_cell") if k in data
            })
        if "nucleus_hypotheses" in data:
            self._ultrack_tab.set_params(data["nucleus_hypotheses"])
        elif "nucleus_ultrack" in data:
            self._ultrack_tab.set_params(data["nucleus_ultrack"])
        elif "ultrack" in data:
            self._ultrack_tab.set_params(data["ultrack"])
        cell_ultrack = data.get("cell_ultrack")
        if cell_ultrack:
            self._cell_seg_tab.set_params(
                cell_ultrack.get("cell_segmentation")
                or cell_ultrack.get("segmentation")
                or {}
            )
            self._seeded_ws_tab.set_params(
                cell_ultrack.get("seeded_watershed")
                or cell_ultrack.get("watershed")
                or {}
            )
        else:
            seg_data = data.get("cell_segmentation") or data.get("flow_watershed")
            if seg_data:
                self._cell_seg_tab.set_params(seg_data)
            if "seeded_watershed" in data:
                self._seeded_ws_tab.set_params(data["seeded_watershed"])
        analysis = data.get("analysis")
        if analysis:
            if "edge_analysis" in analysis:
                self._edge_analysis_widget.set_params(analysis["edge_analysis"])
            if "forces" in analysis:
                self._forces_widget.set_params(analysis["forces"])
        else:
            if "edge_analysis" in data:
                self._edge_analysis_widget.set_params(data["edge_analysis"])
            if "forces" in data:
                self._forces_widget.set_params(data["forces"])
        if "correction" in data:
            self._tracking_correction_widget.set_params(data["correction"])

    def _write_config(self, path: Path) -> None:
        cfg = self._collect_config()
        path.write_text(json.dumps(cfg, indent=2))
        logger.info("Config saved: %s", path)

    def _read_config(self, path: Path) -> None:
        data = json.loads(path.read_text())
        self._apply_config(data)
        logger.info("Config loaded: %s", path)

    def _save_config(self) -> None:
        path = self._config_path()
        if path is None:
            return
        try:
            self._write_config(path)
        except Exception as e:
            logger.warning("Config save failed: %s", e)

    def _load_config(self) -> None:
        path = self._config_path()
        if path is None or not path.exists():
            return
        try:
            self._read_config(path)
        except Exception as e:
            logger.warning("Config load failed: %s", e)

    def _save_config_as(self) -> None:
        default = str(self._config_path() or "")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config As", default, "JSON files (*.json)"
        )
        if not path:
            return
        try:
            self._write_config(Path(path))
        except Exception as e:
            logger.warning("Config save failed: %s", e)

    def _load_config_from(self) -> None:
        default = str(self._config_path() or "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config From", default, "JSON files (*.json)"
        )
        if not path:
            return
        try:
            self._read_config(Path(path))
        except Exception as e:
            logger.warning("Config load failed: %s", e)

    def _autosave_config(self) -> None:
        path = self._config_path()
        if path is None:
            return
        try:
            self._write_config(path)
        except Exception as e:
            logger.warning("Config auto-save failed: %s", e)

    def _load_config_if_exists(self) -> None:
        project_dir = self._state.project_dir
        if project_dir is None or project_dir == self._last_loaded_project_dir:
            return
        path = Path(project_dir) / _CONFIG_FILENAME
        if not path.exists():
            self._last_loaded_project_dir = project_dir
            return
        try:
            self._read_config(path)
            self._last_loaded_project_dir = project_dir
        except Exception as e:
            logger.warning("Auto-load config failed: %s", e)

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

    @property
    def _accordion_sections(self) -> "dict[str, CollapsibleSection]":
        return {
            "Prepare Input Data": self._data_prep_section,
            "Cellpose Cluster":   self._cellpose_section,
            "Nucleus Hypotheses": self._ultrack_section,
            "Seeded Tracking":    self._seeded_tracker_section,
            "Correction":         self._correction_section,
            "Cell Ultrack":       self._cell_ultrack_section,
            "Analysis":           self._analysis_section,
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
