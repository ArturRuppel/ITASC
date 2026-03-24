"""Napari dock widget for napariTissueGraph.

Supports two workflows:
- Single-tissue: staged pipeline with visual QC at each step.
- Batch: build multiple tissues at once, add all to dataset.

The dataset accumulates tissues and can be saved/loaded.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QProgressBar,
    QListWidget,
    QLineEdit,
    QFileDialog,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QScrollArea,
    QCheckBox,
    QTabWidget,
)
from qtpy.QtCore import QThread, Qt, QTimer

from ..structures import TissueGraphTimeSeries
from ..core.graph import apply_track_map
from .visualization import (
    build_all_junction_lines,
    build_all_centroids,
    build_t1_markers,
    build_tracked_centroids,
    build_tracked_labels,
    build_trajectory_lines_with_features,
    build_tag_text_annotations,
)
from ..analysis.tagging import (
    tag_trajectory,
    untag_trajectory,
    tag_junction,
    untag_junction,
    get_all_tags,
    clear_tag,
)
from .workers import (
    PipelineStage,
    CellTrackingWorker,
    GraphExtractWorker,
    AnalysisWorker,
    BatchBuildWorker,
    IOWorker,
)
from .registry import get_state

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Widget
# ------------------------------------------------------------------

class TissueGraphWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._state = get_state(napari_viewer)
        self._batch_label_stacks: List[np.ndarray] = []
        self._batch_file_paths: List[str] = []

        # Pipeline state
        self._pipeline_stage = PipelineStage.IDLE
        self._preview_series: Optional[TissueGraphTimeSeries] = None
        self._current_label_stack: Optional[np.ndarray] = None
        self._current_track_map: Optional[Dict] = None
        self._source_layer = None  # original layer, hidden during QC
        self._tracked_labels_layer = None
        self._stage_layers: Dict[int, list] = {1: [], 2: [], 3: []}
        self._auto_pipeline = False  # flag for Run Pipeline chaining

        # Tagging state
        self._tagging_shapes_layer = None  # the Shapes layer used for tagging
        self._tagging_text_layer = None  # Points layer for tag text annotations
        self._tagging_series = None  # the series currently shown in the tagging layer
        self._cached_selection = []  # persists selection when layer loses focus
        self._show_only_tagged = False
        self._color_by_tags = False
        self._show_tag_labels = False

        # Poll the shapes layer selection every 200ms to keep the cache fresh.
        # napari clears selected_data when the mouse leaves the canvas, and
        # event-based approaches are unreliable across napari versions.
        self._selection_timer = QTimer(self)
        self._selection_timer.setInterval(200)
        self._selection_timer.timeout.connect(self._poll_selection)
        self._selection_timer.start()

        # Dataset inspection layers
        self._inspect_junction_layer = None
        self._inspect_centroid_layer = None
        self._inspect_t1_layer = None
        self._thread = None
        self._worker = None

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(outer_layout)

        self.tab_widget = QTabWidget()
        outer_layout.addWidget(self.tab_widget)

        # ========== Pipeline tab ==========
        pipeline_page = QWidget()
        page_layout = QVBoxLayout()
        page_layout.setContentsMargins(0, 0, 0, 0)
        pipeline_page.setLayout(page_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page_layout.addWidget(scroll)

        container = QWidget()
        layout = QVBoxLayout()
        container.setLayout(layout)
        scroll.setWidget(container)

        # --- Parameters ---
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout()

        px_row = QHBoxLayout()
        px_row.addWidget(QLabel("Pixel size (\u00b5m/px):"))
        self.pixel_size_edit = QLineEdit("")
        self.pixel_size_edit.setPlaceholderText("optional")
        px_row.addWidget(self.pixel_size_edit)
        param_layout.addLayout(px_row)

        dt_row = QHBoxLayout()
        dt_row.addWidget(QLabel("Time interval (s):"))
        self.time_interval_edit = QLineEdit("")
        self.time_interval_edit.setPlaceholderText("optional")
        dt_row.addWidget(self.time_interval_edit)
        param_layout.addLayout(dt_row)

        cond_row = QHBoxLayout()
        cond_row.addWidget(QLabel("Condition:"))
        self.condition_edit = QLineEdit("")
        self.condition_edit.setPlaceholderText("e.g. WT, vim_KO")
        cond_row.addWidget(self.condition_edit)
        param_layout.addLayout(cond_row)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # --- Stage 1: Cell Tracking ---
        self.stage1_toggle = QPushButton("\u25b6 Cell Tracking")
        self.stage1_toggle.setStyleSheet(
            "QPushButton { text-align: left; border: none; padding: 2px; }"
        )
        self.stage1_toggle.setCheckable(True)
        self.stage1_toggle.setChecked(False)
        layout.addWidget(self.stage1_toggle)

        self.stage1_params = QWidget()
        s1p_layout = QVBoxLayout()
        s1p_layout.setContentsMargins(10, 0, 0, 0)

        iou_row = QHBoxLayout()
        iou_row.addWidget(QLabel("Min IoU:"))
        self.min_iou_spin = QDoubleSpinBox()
        self.min_iou_spin.setMinimum(0.0)
        self.min_iou_spin.setMaximum(1.0)
        self.min_iou_spin.setSingleStep(0.05)
        self.min_iou_spin.setValue(0.3)
        iou_row.addWidget(self.min_iou_spin)
        s1p_layout.addLayout(iou_row)

        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Max area change:"))
        self.max_area_change_spin = QDoubleSpinBox()
        self.max_area_change_spin.setMinimum(0.0)
        self.max_area_change_spin.setMaximum(100.0)
        self.max_area_change_spin.setSingleStep(0.5)
        self.max_area_change_spin.setValue(0.0)
        self.max_area_change_spin.setToolTip(
            "Max area ratio between matched labels. "
            "0 = no limit. Try 2.0 to reject segmentation errors."
        )
        area_row.addWidget(self.max_area_change_spin)
        s1p_layout.addLayout(area_row)

        self.stage1_params.setLayout(s1p_layout)
        self.stage1_params.setVisible(False)
        layout.addWidget(self.stage1_params)

        self.stage1_btn = QPushButton("Run Tracking")
        layout.addWidget(self.stage1_btn)

        self.stage1_info = QLabel("")
        self.stage1_info.setWordWrap(True)
        layout.addWidget(self.stage1_info)

        # --- Stage 2: Graph Extraction ---
        self.stage2_toggle = QPushButton("\u25b6 Graph Extraction")
        self.stage2_toggle.setStyleSheet(
            "QPushButton { text-align: left; border: none; padding: 2px; }"
        )
        self.stage2_toggle.setCheckable(True)
        self.stage2_toggle.setChecked(False)
        layout.addWidget(self.stage2_toggle)

        self.stage2_params = QWidget()
        s2p_layout = QVBoxLayout()
        s2p_layout.setContentsMargins(10, 0, 0, 0)

        dr_row = QHBoxLayout()
        dr_row.addWidget(QLabel("Dilation radius:"))
        self.dilation_radius_spin = QSpinBox()
        self.dilation_radius_spin.setMinimum(1)
        self.dilation_radius_spin.setMaximum(20)
        self.dilation_radius_spin.setValue(1)
        self.dilation_radius_spin.setToolTip(
            "Radius for morphological dilation when detecting cell adjacency."
        )
        dr_row.addWidget(self.dilation_radius_spin)
        s2p_layout.addLayout(dr_row)

        mo_row = QHBoxLayout()
        mo_row.addWidget(QLabel("Min overlap (px):"))
        self.min_overlap_spin = QSpinBox()
        self.min_overlap_spin.setMinimum(1)
        self.min_overlap_spin.setMaximum(1000)
        self.min_overlap_spin.setValue(5)
        self.min_overlap_spin.setToolTip(
            "Minimum boundary overlap pixels to consider two cells adjacent."
        )
        mo_row.addWidget(self.min_overlap_spin)
        s2p_layout.addLayout(mo_row)

        mel_row = QHBoxLayout()
        mel_row.addWidget(QLabel("Min edge length (px):"))
        self.min_edge_length_spin = QDoubleSpinBox()
        self.min_edge_length_spin.setMinimum(0.0)
        self.min_edge_length_spin.setMaximum(1000.0)
        self.min_edge_length_spin.setSingleStep(1.0)
        self.min_edge_length_spin.setValue(0.0)
        self.min_edge_length_spin.setToolTip(
            "Minimum junction length to keep. Shorter junctions are discarded."
        )
        mel_row.addWidget(self.min_edge_length_spin)
        s2p_layout.addLayout(mel_row)

        self.filter_isolated_cb = QCheckBox("Tag border edges")
        self.filter_isolated_cb.setChecked(True)
        self.filter_isolated_cb.setToolTip(
            "Tag border/isolated edges as 'border' for downstream filtering."
        )
        s2p_layout.addWidget(self.filter_isolated_cb)

        mbel_row = QHBoxLayout()
        mbel_row.addWidget(QLabel("Min border edge (px):"))
        self.min_border_edge_spin = QDoubleSpinBox()
        self.min_border_edge_spin.setMinimum(0.0)
        self.min_border_edge_spin.setMaximum(1000.0)
        self.min_border_edge_spin.setSingleStep(1.0)
        self.min_border_edge_spin.setValue(5.0)
        self.min_border_edge_spin.setToolTip(
            "Minimum length for a border boundary segment to count as a real\n"
            "border edge. Increase to ignore small segmentation holes."
        )
        mbel_row.addWidget(self.min_border_edge_spin)
        s2p_layout.addLayout(mbel_row)

        self.stage2_params.setLayout(s2p_layout)
        self.stage2_params.setVisible(False)
        layout.addWidget(self.stage2_params)

        self.stage2_btn = QPushButton("Extract Graphs")
        layout.addWidget(self.stage2_btn)

        self.stage2_info = QLabel("")
        self.stage2_info.setWordWrap(True)
        layout.addWidget(self.stage2_info)

        # --- Stage 3: T1 + Edge Tracking ---
        self.stage3_toggle = QPushButton("\u25b6 T1 + Edge Tracking")
        self.stage3_toggle.setStyleSheet(
            "QPushButton { text-align: left; border: none; padding: 2px; }"
        )
        self.stage3_toggle.setCheckable(True)
        self.stage3_toggle.setChecked(False)
        layout.addWidget(self.stage3_toggle)

        self.stage3_params = QWidget()
        s3p_layout = QVBoxLayout()
        s3p_layout.setContentsMargins(10, 0, 0, 0)

        mjl_row = QHBoxLayout()
        mjl_row.addWidget(QLabel("Min junction length (px):"))
        self.min_junction_length_spin = QDoubleSpinBox()
        self.min_junction_length_spin.setMinimum(0.0)
        self.min_junction_length_spin.setMaximum(1000.0)
        self.min_junction_length_spin.setSingleStep(1.0)
        self.min_junction_length_spin.setValue(0.0)
        self.min_junction_length_spin.setToolTip(
            "Junctions shorter than this are ignored for T1 detection. "
            "Increase if noisy short edges cause false T1s."
        )
        mjl_row.addWidget(self.min_junction_length_spin)
        s3p_layout.addLayout(mjl_row)

        mtd_row = QHBoxLayout()
        mtd_row.addWidget(QLabel("Max T1 distance (px):"))
        self.max_t1_distance_spin = QDoubleSpinBox()
        self.max_t1_distance_spin.setMinimum(0.0)
        self.max_t1_distance_spin.setMaximum(10000.0)
        self.max_t1_distance_spin.setSingleStep(5.0)
        self.max_t1_distance_spin.setValue(0.0)
        self.max_t1_distance_spin.setToolTip(
            "Max distance between lost/gained edge midpoints to pair as T1. "
            "0 = no limit. Reduce to avoid pairing distant events."
        )
        mtd_row.addWidget(self.max_t1_distance_spin)
        s3p_layout.addLayout(mtd_row)

        mtf_row = QHBoxLayout()
        mtf_row.addWidget(QLabel("Min trajectory frames:"))
        self.min_traj_frames_spin = QSpinBox()
        self.min_traj_frames_spin.setMinimum(1)
        self.min_traj_frames_spin.setMaximum(10000)
        self.min_traj_frames_spin.setValue(1)
        self.min_traj_frames_spin.setToolTip(
            "Minimum frames a junction must exist to keep its trajectory."
        )
        mtf_row.addWidget(self.min_traj_frames_spin)
        s3p_layout.addLayout(mtf_row)

        mc_row = QHBoxLayout()
        mc_row.addWidget(QLabel("Min completeness:"))
        self.min_completeness_spin = QDoubleSpinBox()
        self.min_completeness_spin.setMinimum(0.0)
        self.min_completeness_spin.setMaximum(1.0)
        self.min_completeness_spin.setSingleStep(0.05)
        self.min_completeness_spin.setValue(0.0)
        self.min_completeness_spin.setToolTip(
            "Fraction of total frames a trajectory must span (0.0-1.0)."
        )
        mc_row.addWidget(self.min_completeness_spin)
        s3p_layout.addLayout(mc_row)

        mg_row = QHBoxLayout()
        mg_row.addWidget(QLabel("Max gap tolerance:"))
        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setMinimum(0)
        self.max_gap_spin.setMaximum(1000)
        self.max_gap_spin.setValue(0)
        self.max_gap_spin.setToolTip(
            "Max consecutive missing frames allowed in a trajectory. "
            "0 = no gaps allowed."
        )
        mg_row.addWidget(self.max_gap_spin)
        s3p_layout.addLayout(mg_row)

        self.stage3_params.setLayout(s3p_layout)
        self.stage3_params.setVisible(False)
        layout.addWidget(self.stage3_params)

        self.analyze_btn = QPushButton("Run Analysis")
        layout.addWidget(self.analyze_btn)

        self.stage3_info = QLabel("")
        self.stage3_info.setWordWrap(True)
        layout.addWidget(self.stage3_info)

        # --- Run Full Pipeline (one-click, chains all 3 stages) ---
        self.run_pipeline_btn = QPushButton("Run Full Pipeline")
        self.run_pipeline_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }"
        )
        layout.addWidget(self.run_pipeline_btn)

        # Active layer indicator (shown after pipeline starts)
        self.active_layer_label = QLabel("")
        self.active_layer_label.setStyleSheet("color: gray;")
        layout.addWidget(self.active_layer_label)

        # --- Add / Discard ---
        self.finalize_group = QGroupBox("Add / Discard")
        fin_layout = QHBoxLayout()
        self.add_to_dataset_btn = QPushButton("Add to Dataset")
        self.discard_btn = QPushButton("Discard")
        fin_layout.addWidget(self.add_to_dataset_btn)
        fin_layout.addWidget(self.discard_btn)
        self.finalize_group.setLayout(fin_layout)
        layout.addWidget(self.finalize_group)

        # --- Junction Tagging ---
        self.tagging_group = QGroupBox("Junction Tagging")
        tag_layout = QVBoxLayout()

        self.selection_label = QLabel("No junctions selected")
        tag_layout.addWidget(self.selection_label)

        tag_input_row = QHBoxLayout()
        self.tag_name_edit = QLineEdit()
        self.tag_name_edit.setPlaceholderText("Tag name (e.g. central)")
        tag_input_row.addWidget(self.tag_name_edit)
        tag_layout.addLayout(tag_input_row)

        tag_btn_row = QHBoxLayout()
        self.tag_selected_btn = QPushButton("Tag Selected")
        self.untag_selected_btn = QPushButton("Untag Selected")
        tag_btn_row.addWidget(self.tag_selected_btn)
        tag_btn_row.addWidget(self.untag_selected_btn)
        tag_layout.addLayout(tag_btn_row)

        self.color_by_tags_cb = QCheckBox("Color by tags")
        tag_layout.addWidget(self.color_by_tags_cb)
        self.show_only_tagged_cb = QCheckBox("Show only tagged junctions")
        tag_layout.addWidget(self.show_only_tagged_cb)
        self.show_tag_labels_cb = QCheckBox("Show tag labels")
        tag_layout.addWidget(self.show_tag_labels_cb)

        tag_layout.addWidget(QLabel("Tags:"))
        self.tag_list_widget = QListWidget()
        self.tag_list_widget.setMaximumHeight(80)
        tag_layout.addWidget(self.tag_list_widget)

        clear_tag_row = QHBoxLayout()
        self.clear_tag_btn = QPushButton("Clear Selected Tag")
        self.refresh_tags_btn = QPushButton("Refresh")
        clear_tag_row.addWidget(self.clear_tag_btn)
        clear_tag_row.addWidget(self.refresh_tags_btn)
        tag_layout.addLayout(clear_tag_row)

        self.tagging_group.setLayout(tag_layout)
        layout.addWidget(self.tagging_group)
        self.tagging_group.setVisible(False)

        # --- Batch mode ---
        self.batch_group = QGroupBox("Batch")
        batch_layout = QVBoxLayout()
        self.batch_load_btn = QPushButton("Load Files...")
        batch_layout.addWidget(self.batch_load_btn)
        self.batch_file_list = QListWidget()
        self.batch_file_list.setMaximumHeight(100)
        batch_layout.addWidget(self.batch_file_list)
        self.batch_file_info = QLabel("")
        self.batch_file_info.setWordWrap(True)
        batch_layout.addWidget(self.batch_file_info)
        btn_row = QHBoxLayout()
        self.build_batch_btn = QPushButton("Run Batch")
        self.batch_clear_btn = QPushButton("Clear")
        btn_row.addWidget(self.build_batch_btn)
        btn_row.addWidget(self.batch_clear_btn)
        batch_layout.addLayout(btn_row)
        self.batch_group.setLayout(batch_layout)
        layout.addWidget(self.batch_group)

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # --- Dataset section ---
        self.dataset_group = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout()

        self.dataset_summary_label = QLabel("No dataset")
        self.dataset_summary_label.setWordWrap(True)
        dataset_layout.addWidget(self.dataset_summary_label)

        tissue_row = QHBoxLayout()
        tissue_row.addWidget(QLabel("Tissue:"))
        self.tissue_spinner = QSpinBox()
        self.tissue_spinner.setMinimum(0)
        self.tissue_spinner.setMaximum(0)
        tissue_row.addWidget(self.tissue_spinner)
        self.tissue_info_label = QLabel("")
        tissue_row.addWidget(self.tissue_info_label)
        dataset_layout.addLayout(tissue_row)

        self.show_tissue_btn = QPushButton("Show Tissue")
        self.remove_tissue_btn = QPushButton("Remove Tissue")
        tissue_btn_row = QHBoxLayout()
        tissue_btn_row.addWidget(self.show_tissue_btn)
        tissue_btn_row.addWidget(self.remove_tissue_btn)
        dataset_layout.addLayout(tissue_btn_row)

        io_row = QHBoxLayout()
        self.save_btn = QPushButton("Save Dataset...")
        self.load_dataset_btn = QPushButton("Load Dataset...")
        self.new_dataset_btn = QPushButton("New")
        io_row.addWidget(self.save_btn)
        io_row.addWidget(self.load_dataset_btn)
        io_row.addWidget(self.new_dataset_btn)
        dataset_layout.addLayout(io_row)

        self.open_dashboard_btn = QPushButton("Open Dashboard...")
        self.open_dashboard_btn.setToolTip("Launch the analysis dashboard in your browser")
        dataset_layout.addWidget(self.open_dashboard_btn)

        self.dataset_group.setLayout(dataset_layout)
        layout.addWidget(self.dataset_group)

        layout.addStretch()

        self.tab_widget.addTab(pipeline_page, "Pipeline")

        # ========== Nuclear Tracks tab ==========
        from .tracks_widget import NuclearTracksWidget
        self._tracks_widget = NuclearTracksWidget(self.viewer)
        self.tab_widget.addTab(self._tracks_widget, "Nuclear Tracks")

        # ========== Forces tab ==========
        from .forces_widget import ForcesWidget
        self._forces_widget = ForcesWidget(self.viewer)
        self.tab_widget.addTab(self._forces_widget, "Forces")

        # Set initial button state
        self._update_pipeline_buttons()

    def _connect_signals(self):
        # Run Pipeline (one-click)
        self.run_pipeline_btn.clicked.connect(self._run_full_pipeline)

        # Stage parameter toggles
        self.stage1_toggle.toggled.connect(
            lambda checked: self._toggle_stage(1, checked)
        )
        self.stage2_toggle.toggled.connect(
            lambda checked: self._toggle_stage(2, checked)
        )
        self.stage3_toggle.toggled.connect(
            lambda checked: self._toggle_stage(3, checked)
        )

        # Pipeline (individual stages)
        self.stage1_btn.clicked.connect(self._run_stage1)
        self.stage2_btn.clicked.connect(self._run_stage2)
        self.analyze_btn.clicked.connect(self._run_analysis)
        self.add_to_dataset_btn.clicked.connect(self._add_preview_to_dataset)
        self.discard_btn.clicked.connect(self._discard_pipeline)

        # Tagging
        self.tag_selected_btn.clicked.connect(self._tag_selected)
        self.untag_selected_btn.clicked.connect(self._untag_selected)
        self.color_by_tags_cb.toggled.connect(self._toggle_color_by_tags)
        self.show_only_tagged_cb.toggled.connect(self._toggle_show_only_tagged)
        self.show_tag_labels_cb.toggled.connect(self._toggle_show_tag_labels)
        self.clear_tag_btn.clicked.connect(self._clear_selected_tag)
        self.refresh_tags_btn.clicked.connect(self._refresh_tagging_layer)

        # Batch
        self.batch_load_btn.clicked.connect(self._load_batch_files)
        self.build_batch_btn.clicked.connect(self._build_batch)
        self.batch_clear_btn.clicked.connect(self._clear_batch_files)

        # Dataset
        self.show_tissue_btn.clicked.connect(self._show_selected_tissue)
        self.remove_tissue_btn.clicked.connect(self._remove_current_tissue)
        self.save_btn.clicked.connect(self._save_dataset)
        self.load_dataset_btn.clicked.connect(self._load_dataset)
        self.new_dataset_btn.clicked.connect(self._new_dataset)
        self.open_dashboard_btn.clicked.connect(self._open_dashboard)

        # Registry: react to dataset changes from any widget
        self._state.dataset_changed.connect(self._update_dataset_ui)

    # ------------------------------------------------------------------
    # Pipeline stage gating
    # ------------------------------------------------------------------
    def _update_pipeline_buttons(self):
        stage = self._pipeline_stage
        # Allow re-running earlier stages from any later stage so users can
        # tweak parameters without having to discard and start over.
        self.run_pipeline_btn.setEnabled(True)
        self.stage1_btn.setEnabled(True)
        self.stage2_btn.setEnabled(stage.value >= PipelineStage.STAGE1_DONE.value)
        self.analyze_btn.setEnabled(stage.value >= PipelineStage.STAGE2_DONE.value)
        self.add_to_dataset_btn.setEnabled(stage == PipelineStage.STAGE3_DONE)
        self.discard_btn.setEnabled(stage != PipelineStage.IDLE)

    _stage_titles = {1: "Cell Tracking", 2: "Graph Extraction", 3: "T1 + Edge Tracking"}

    def _toggle_stage(self, stage: int, checked: bool):
        containers = {1: self.stage1_params, 2: self.stage2_params, 3: self.stage3_params}
        toggles = {1: self.stage1_toggle, 2: self.stage2_toggle, 3: self.stage3_toggle}
        containers[stage].setVisible(checked)
        arrow = "\u25bc" if checked else "\u25b6"
        toggles[stage].setText(f"{arrow} {self._stage_titles[stage]}")

    # ------------------------------------------------------------------
    # Label stack from viewer (uses active layer)
    # ------------------------------------------------------------------
    def _get_active_label_stack(self) -> Optional[np.ndarray]:
        """Return the data from the active Labels/Image layer, or None."""
        import napari

        active = self.viewer.layers.selection.active
        if active is None:
            self.status_label.setText("No active layer. Select a Labels or Image layer.")
            return None
        if not isinstance(active, (napari.layers.Labels, napari.layers.Image)):
            self.status_label.setText(
                f"Active layer '{active.name}' is not a Labels or Image layer."
            )
            return None

        data = active.data

        # Auto-convert Image layer data to integer labels
        if isinstance(active, napari.layers.Image):
            if not np.issubdtype(data.dtype, np.integer):
                data = np.round(data).astype(np.int32)
            logger.info("Converted Image layer '%s' to integer labels.", active.name)

        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if data.ndim != 3:
            self.status_label.setText(
                f"Layer must be 2D or 3D (T, H, W), got {data.ndim}D."
            )
            return None

        self.active_layer_label.setText(f"Layer: {active.name}")
        return data

    # ------------------------------------------------------------------
    # Run full pipeline (one-click)
    # ------------------------------------------------------------------
    def _run_full_pipeline(self):
        """Chain stages 1 -> 2 -> 3 and auto-add to dataset."""
        self._auto_pipeline = True
        self._run_stage1()

    # ------------------------------------------------------------------
    # Stage 1: Cell Tracking (IoU)
    # ------------------------------------------------------------------
    def _run_stage1(self):
        label_stack = self._get_active_label_stack()
        if label_stack is None:
            return

        self._discard_pipeline()
        self._current_label_stack = label_stack
        self._source_layer = self.viewer.layers.selection.active

        worker = CellTrackingWorker(
            label_stack,
            min_iou=self.min_iou_spin.value(),
            max_area_change=self.max_area_change_spin.value(),
        )
        self._run_worker(worker, self._on_stage1_done)

    def _on_stage1_done(self, track_map):
        self._current_track_map = track_map
        self._pipeline_stage = PipelineStage.STAGE1_DONE
        self._finish_worker()

        # Count tracks and breaks from the track_map
        all_track_ids = set()
        for frame_tracks in track_map.values():
            all_track_ids.update(frame_tracks.values())

        # Count births/deaths from track_map
        frame_indices = sorted(track_map.keys())
        track_first = {}
        track_last = {}
        for f_idx in frame_indices:
            for tid in track_map[f_idx].values():
                if tid not in track_first:
                    track_first[tid] = f_idx
                track_last[tid] = f_idx

        n_births = sum(1 for f in track_first.values() if f > frame_indices[0]) if frame_indices else 0
        n_deaths = sum(1 for f in track_last.values() if f < frame_indices[-1]) if frame_indices else 0

        self.stage1_info.setText(
            f"{len(all_track_ids)} tracks, {n_births} births, {n_deaths} deaths"
        )

        if not self._auto_pipeline:
            self.status_label.setText("Stage 1 complete. Inspect tracked labels, then extract graphs.")

            # Show tracked labels as napari Labels layer
            self._remove_stage_layers(1)
            tracked = build_tracked_labels(self._current_label_stack, track_map)
            layer = self.viewer.add_labels(
                tracked, name="[Pipeline] Tracked Labels",
            )
            self._tracked_labels_layer = layer
            self._stage_layers[1].append(layer)

            # Hide the source layer so its background color doesn't show through
            if self._source_layer is not None:
                try:
                    self._source_layer.visible = False
                except Exception:
                    pass

        self._update_pipeline_buttons()

        if self._auto_pipeline:
            QTimer.singleShot(0, self._run_stage2)

    # ------------------------------------------------------------------
    # Stage 2: Graph Extraction
    # ------------------------------------------------------------------
    def _run_stage2(self):
        if self._current_label_stack is None:
            return

        pixel_size = self._parse_float(self.pixel_size_edit.text())
        time_interval = self._parse_float(self.time_interval_edit.text())

        worker = GraphExtractWorker(
            self._current_label_stack,
            pixel_size=pixel_size,
            time_interval=time_interval,
            dilation_radius=self.dilation_radius_spin.value(),
            min_overlap_pixels=self.min_overlap_spin.value(),
            min_edge_length=self.min_edge_length_spin.value(),
            filter_isolated=self.filter_isolated_cb.isChecked(),
            min_border_edge_length=self.min_border_edge_spin.value(),
        )
        self._run_worker(worker, self._on_stage2_done)

    def _on_stage2_done(self, series):
        if self._current_track_map is not None:
            apply_track_map(series, self._current_track_map)

        self._preview_series = series
        self._pipeline_stage = PipelineStage.STAGE2_DONE
        self._finish_worker()

        total_cells = sum(len(f.cells) for f in series.frames.values())
        total_junctions = sum(len(f.junctions) for f in series.frames.values())

        # Count how many cells have tracking
        n_tracked = sum(
            1 for f in series.frames.values()
            for c in f.cells.values() if c.track_id is not None
        )
        self.stage2_info.setText(
            f"{series.num_frames} frames, {total_cells} cells ({n_tracked} tracked), "
            f"{total_junctions} junctions"
        )

        if not self._auto_pipeline:
            self.status_label.setText("Stage 2 complete. Inspect junctions & tracked centroids, then run analysis.")

            # Keep tracked labels layer visible underneath for context
            self._remove_stage_layers(2)

            # Show junctions
            lines, colors = build_all_junction_lines(series)
            if lines:
                layer = self.viewer.add_shapes(
                    lines, shape_type="path", edge_color=colors,
                    edge_width=2, name="[Pipeline] Junctions",
                )
                self._stage_layers[2].append(layer)

            # Show tracked centroids (graph cells only)
            tracked_pos, tracked_colors, _ = build_tracked_centroids(series)
            if len(tracked_pos) > 0:
                layer = self.viewer.add_points(
                    tracked_pos, size=5, face_color=tracked_colors,
                    name="[Pipeline] Tracked Centroids",
                )
                self._stage_layers[2].append(layer)

        self._update_pipeline_buttons()

        if self._auto_pipeline:
            QTimer.singleShot(0, self._run_analysis)

    # ------------------------------------------------------------------
    # Stage 3: T1 + Edge trajectories
    # ------------------------------------------------------------------
    def _run_analysis(self):
        if self._preview_series is None:
            return
        worker = AnalysisWorker(
            self._preview_series,
            min_junction_length=self.min_junction_length_spin.value(),
            max_t1_distance=self.max_t1_distance_spin.value(),
            min_traj_frames=self.min_traj_frames_spin.value(),
            min_completeness=self.min_completeness_spin.value(),
            max_gap=self.max_gap_spin.value(),
        )
        self._run_worker(worker, self._on_analysis_finished)

    def _on_analysis_finished(self, series):
        self._preview_series = series
        self._pipeline_stage = PipelineStage.STAGE3_DONE
        self._finish_worker()

        n_t1 = len(series.t1_events)
        n_trajs = len(series.edge_trajectories)

        self.stage3_info.setText(
            f"{n_t1} T1 events, {n_trajs} edge trajectories"
        )

        if self._auto_pipeline:
            self._auto_pipeline = False
            QTimer.singleShot(0, self._add_preview_to_dataset)
            return

        self.status_label.setText("Stage 3 complete. Inspect T1 events & trajectories, then add to dataset.")

        # Replace length-colored junctions with trajectory-colored junctions
        self._remove_stage_layers(3)

        # Remove stage 1 junction layer
        for l in list(self.viewer.layers):
            if l.name == "[Pipeline] Junctions":
                self.viewer.layers.remove(l)
        self._stage_layers[1] = [
            l for l in self._stage_layers[1]
            if l in self.viewer.layers
        ]

        # Build junction Shapes layer with features for tagging
        self._show_tagging_for_series(series, stage_layer=True)

        # T1 markers
        t1_positions = build_t1_markers(series.t1_events)
        if len(t1_positions) > 0:
            layer = self.viewer.add_points(
                t1_positions, size=12, face_color="red",
                symbol="star", name="[Pipeline] T1 Events",
            )
            self._stage_layers[3].append(layer)

        self._update_pipeline_buttons()

    # ------------------------------------------------------------------
    # Add / Discard
    # ------------------------------------------------------------------
    def _add_preview_to_dataset(self):
        if self._preview_series is None:
            return
        self._state.ensure_dataset(
            condition=self.condition_edit.text().strip(),
            pixel_size=self._parse_float(self.pixel_size_edit.text()),
            time_interval=self._parse_float(self.time_interval_edit.text()),
        )
        tid = self._state.add_tissue(self._preview_series)
        self._remove_all_stage_layers()
        self._preview_series = None
        self._current_label_stack = None
        self._current_track_map = None
        if self._tracked_labels_layer is not None:
            if self._tracked_labels_layer in self.viewer.layers:
                self.viewer.layers.remove(self._tracked_labels_layer)
            self._tracked_labels_layer = None
        self._pipeline_stage = PipelineStage.IDLE
        self._update_pipeline_buttons()
        self._clear_stage_info()
        self.status_label.setText(f"Added tissue {tid} to dataset.")

    def _discard_pipeline(self):
        self._remove_all_stage_layers()
        self._remove_tagging_layer()
        self._tagging_series = None
        self.tagging_group.setVisible(False)
        self._preview_series = None
        self._current_label_stack = None
        self._current_track_map = None
        if self._tracked_labels_layer is not None:
            if self._tracked_labels_layer in self.viewer.layers:
                self.viewer.layers.remove(self._tracked_labels_layer)
            self._tracked_labels_layer = None
        # Restore source layer visibility
        if self._source_layer is not None:
            try:
                self._source_layer.visible = True
            except Exception:
                pass
            self._source_layer = None
        self._pipeline_stage = PipelineStage.IDLE
        self._update_pipeline_buttons()
        self._clear_stage_info()

    def _clear_stage_info(self):
        self.active_layer_label.setText("")
        self.stage1_info.setText("")
        self.stage2_info.setText("")
        self.stage3_info.setText("")

    # ------------------------------------------------------------------
    # Batch file management
    # ------------------------------------------------------------------
    def _load_batch_files(self):
        """Load label files for batch processing."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Label Files",
            "", "Image files (*.tif *.tiff *.npy *.npz);;All files (*)",
        )
        if not files:
            return
        self._batch_label_stacks = []
        self._batch_file_paths = []
        self.batch_file_list.clear()
        errors = []
        for path in sorted(files):
            try:
                stack = self._load_label_file(path)
                self._batch_label_stacks.append(stack)
                self._batch_file_paths.append(path)
                self.batch_file_list.addItem(
                    f"{Path(path).name}  ({stack.shape[0]} frames, "
                    f"{stack.shape[1]}x{stack.shape[2]})"
                )
            except Exception as e:
                errors.append(f"{Path(path).name}: {e}")
        info = f"{len(self._batch_label_stacks)} file(s) loaded"
        if errors:
            info += f"\nErrors: {'; '.join(errors)}"
        self.batch_file_info.setText(info)

    @staticmethod
    def _load_label_file(path: str) -> np.ndarray:
        """Load a label stack from a .tif/.tiff or .npy/.npz file."""
        from pathlib import Path as _P
        ext = _P(path).suffix.lower()
        if ext in (".tif", ".tiff"):
            from tifffile import imread
            data = imread(path)
        elif ext == ".npy":
            data = np.load(path)
        elif ext == ".npz":
            npz = np.load(path)
            # Use first array in the archive
            data = npz[list(npz.keys())[0]]
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        if not np.issubdtype(data.dtype, np.integer):
            data = np.round(data).astype(np.int32)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
        return data

    def _clear_batch_files(self):
        self._batch_label_stacks = []
        self._batch_file_paths = []
        self.batch_file_list.clear()
        self.batch_file_info.setText("")

    # ------------------------------------------------------------------
    # Batch build (monolithic, no per-stage QC)
    # ------------------------------------------------------------------
    def _build_batch(self):
        if not self._batch_label_stacks:
            self.status_label.setText("Load label files first using 'Load Files...'.")
            return

        pixel_size = self._parse_float(self.pixel_size_edit.text())
        time_interval = self._parse_float(self.time_interval_edit.text())

        worker = BatchBuildWorker(
            label_stacks=self._batch_label_stacks,
            pixel_size=pixel_size,
            time_interval=time_interval,
            min_iou=self.min_iou_spin.value(),
            max_area_change=self.max_area_change_spin.value(),
            min_junction_length=self.min_junction_length_spin.value(),
            max_t1_distance=self.max_t1_distance_spin.value(),
            min_traj_frames=self.min_traj_frames_spin.value(),
            min_completeness=self.min_completeness_spin.value(),
            max_gap=self.max_gap_spin.value(),
        )
        self._run_worker(worker, self._on_batch_finished)

    def _on_batch_finished(self, series_list):
        self._state.ensure_dataset(
            condition=self.condition_edit.text().strip(),
            pixel_size=self._parse_float(self.pixel_size_edit.text()),
            time_interval=self._parse_float(self.time_interval_edit.text()),
        )
        for series in series_list:
            self._state.add_tissue(series)

        self._finish_worker()
        n = len(series_list)
        self.status_label.setText(f"Added {n} tissue(s) to dataset.")
        self._clear_batch_files()

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------
    def _update_dataset_ui(self):
        ds = self._state.dataset
        if ds is None or ds.n_tissues == 0:
            self.dataset_summary_label.setText("No dataset" if ds is None else "Dataset empty")
            self.tissue_spinner.setMinimum(0)
            self.tissue_spinner.setMaximum(0)
            return

        parts = [f"{ds.n_tissues} tissue(s)"]
        if ds.condition:
            parts.append(f"Condition: {ds.condition}")
        for tid in ds.tissue_ids:
            s = ds.tissues[tid]
            n_t1 = len(s.t1_events)
            parts.append(f"  T{tid}: {s.num_frames} frames, {n_t1} T1 events")
        self.dataset_summary_label.setText("\n".join(parts))

        ids = ds.tissue_ids
        self.tissue_spinner.setMinimum(min(ids))
        self.tissue_spinner.setMaximum(max(ids))

    def _show_selected_tissue(self):
        if self._state.dataset is None:
            return
        tid = self.tissue_spinner.value()
        if tid not in self._state.dataset.tissue_ids:
            self.status_label.setText(f"Tissue {tid} does not exist.")
            return
        self._remove_inspect_layers()
        series = self._state.dataset.tissues[tid]
        n_t1 = len(series.t1_events)
        n_trajs = len(series.edge_trajectories)
        total_cells = sum(len(f.cells) for f in series.frames.values())
        total_junctions = sum(len(f.junctions) for f in series.frames.values())
        self.tissue_info_label.setText(
            f"{series.num_frames}f, {total_cells}c, {total_junctions}j, {n_t1} T1"
        )
        self.status_label.setText(
            f"Tissue {tid}: {series.num_frames} frames, "
            f"{total_cells} cells, {total_junctions} junctions, "
            f"{n_t1} T1 events, {n_trajs} edge trajectories"
        )
        if series.edge_trajectories:
            # Use the tagging-enabled Shapes layer for junctions
            self._add_inspect_layers(series, skip_junctions=True)
            self._show_tagging_for_series(series)
        else:
            self._add_inspect_layers(series)

    def _remove_current_tissue(self):
        if self._state.dataset is None:
            return
        tid = self.tissue_spinner.value()
        if tid not in self._state.dataset.tissue_ids:
            self.status_label.setText(f"Tissue {tid} does not exist.")
            return
        self._state.remove_tissue(tid)
        self._remove_inspect_layers()
        self.status_label.setText(f"Removed tissue {tid}.")

    def _new_dataset(self):
        self._remove_inspect_layers()
        self._discard_pipeline()
        self._state.dataset = None
        self.status_label.setText("Created new dataset.")

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    def _open_dashboard(self):
        """Save dataset to a temp dir and launch the Dash dashboard."""
        ds = self._state.dataset
        if ds is None or ds.n_tissues == 0:
            self.status_label.setText("No dataset to open in dashboard.")
            return
        try:
            import tempfile
            tmpdir = Path(tempfile.mkdtemp(prefix="tissuegraph_dashboard_"))
            from ..core.io import save_dataset
            save_dataset(ds, tmpdir)

            import subprocess
            import sys
            subprocess.Popen(
                [sys.executable, "-m", "napariTissueGraph.dashboard", str(tmpdir)],
            )
            self.status_label.setText("Dashboard launched in browser.")
        except ImportError:
            self.status_label.setText(
                "Dashboard requires dash+plotly. Install with: "
                "pip install napariTissueGraph[dashboard]"
            )
        except Exception as exc:
            self.status_label.setText(f"Failed to launch dashboard: {exc}")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def _save_dataset(self):
        ds = self._state.dataset
        if ds is None or ds.n_tissues == 0:
            self.status_label.setText("No dataset to save.")
            return
        path = QFileDialog.getExistingDirectory(self, "Save Dataset To")
        if not path:
            return
        self._run_worker(
            IOWorker("save", path, ds), self._on_save_finished
        )

    def _on_save_finished(self, _):
        self._finish_worker()
        self.status_label.setText("Dataset saved.")

    def _load_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Load Dataset From")
        if not path:
            return
        self._run_worker(
            IOWorker("load", path), self._on_load_finished
        )

    def _on_load_finished(self, dataset):
        self._state.dataset = dataset  # triggers dataset_changed signal
        self._finish_worker()

        # Populate UI from loaded metadata
        if dataset.condition:
            self.condition_edit.setText(dataset.condition)
        if dataset.pixel_size is not None:
            self.pixel_size_edit.setText(str(dataset.pixel_size))
        if dataset.time_interval is not None:
            self.time_interval_edit.setText(str(dataset.time_interval))

        self.status_label.setText(
            f"Loaded dataset: {dataset.n_tissues} tissue(s), "
            f"condition: {dataset.condition or '(none)'}"
        )

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------
    def _run_worker(self, worker, on_finished):
        # Ensure any previous thread is fully stopped before starting a new one.
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    self._thread.quit()
                    self._thread.wait()
            except RuntimeError:
                pass  # C++ object already deleted by deleteLater
            self._thread = None
            self._worker = None

        self._thread = QThread()
        self._worker = worker
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        if hasattr(self._worker, "progress"):
            self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self.run_pipeline_btn.setEnabled(False)
        self.stage1_btn.setEnabled(False)
        self.stage2_btn.setEnabled(False)
        self.analyze_btn.setEnabled(False)
        self.build_batch_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Working...")

        self._thread.start()

    def _finish_worker(self):
        """Re-enable buttons and hide progress bar after worker completes."""
        self.progress_bar.setVisible(False)
        self.build_batch_btn.setEnabled(True)
        self._update_pipeline_buttons()

    def _on_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _on_error(self, exc):
        self._auto_pipeline = False
        self.progress_bar.setVisible(False)
        self.build_batch_btn.setEnabled(True)
        self._update_pipeline_buttons()
        self.status_label.setText(f"Error: {exc}")
        logger.exception("Worker failed", exc_info=exc)

    # ------------------------------------------------------------------
    # Napari layer management
    # ------------------------------------------------------------------
    def _remove_stage_layers(self, stage: int):
        for layer in self._stage_layers[stage]:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self._stage_layers[stage] = []

    def _remove_all_stage_layers(self):
        for stage in (1, 2, 3):
            self._remove_stage_layers(stage)

    def _remove_inspect_layers(self):
        for layer in (
            self._inspect_junction_layer,
            self._inspect_centroid_layer,
            self._inspect_t1_layer,
        ):
            if layer is not None and layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self._inspect_junction_layer = None
        self._inspect_centroid_layer = None
        self._inspect_t1_layer = None
        self._remove_tagging_layer()
        self._tagging_series = None
        self.tagging_group.setVisible(False)

    def _add_inspect_layers(
        self, series: TissueGraphTimeSeries, skip_junctions: bool = False,
    ):
        self._remove_inspect_layers()
        j, c, t = self._make_layers(
            series, prefix="", skip_junctions=skip_junctions,
        )
        self._inspect_junction_layer = j
        self._inspect_centroid_layer = c
        self._inspect_t1_layer = t

    def _make_layers(
        self,
        series: TissueGraphTimeSeries,
        prefix: str = "",
        skip_junctions: bool = False,
    ):
        """Create junction, centroid, and T1 layers for a tissue."""
        junction_layer = None
        centroid_layer = None
        t1_layer = None

        if not skip_junctions:
            lines, colors = build_all_junction_lines(series)
            if lines:
                junction_layer = self.viewer.add_shapes(
                    lines,
                    shape_type="path",
                    edge_color=colors,
                    edge_width=2,
                    name=f"{prefix}Junctions",
                )

        centroids = build_all_centroids(series)
        if len(centroids) > 0:
            centroid_layer = self.viewer.add_points(
                centroids,
                size=5,
                face_color="yellow",
                name=f"{prefix}Cell Centroids",
            )

        t1_positions = build_t1_markers(series.t1_events)
        if len(t1_positions) > 0:
            t1_layer = self.viewer.add_points(
                t1_positions,
                size=12,
                face_color="red",
                symbol="star",
                name=f"{prefix}T1 Events",
            )

        return junction_layer, centroid_layer, t1_layer

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------
    def _show_tagging_for_series(
        self, series: TissueGraphTimeSeries, stage_layer: bool = False,
    ):
        """Create or refresh the junction Shapes layer with features for tagging.

        Parameters
        ----------
        series : TissueGraphTimeSeries
            The series to visualize.
        stage_layer : bool
            If True, register the layer as a stage 3 layer (pipeline flow).
        """
        self._tagging_series = series
        self._remove_tagging_layer()
        self.tagging_group.setVisible(True)

        lines, colors, features = build_trajectory_lines_with_features(
            series,
            color_by_tags=self._color_by_tags,
            show_only_tagged=self._show_only_tagged,
        )
        if not lines:
            self._update_tag_list()
            return

        layer = self.viewer.add_shapes(
            lines,
            shape_type="path",
            edge_color=colors,
            edge_width=2,
            features=features,
            name="[Pipeline] Trajectories",
        )
        self._tagging_shapes_layer = layer
        self._cached_selection = []

        # Switch to SELECT mode so the user can click to select junctions.
        # In PAN_ZOOM (the default), hovering highlights shapes but clicking
        # does not populate selected_data.
        layer.mode = "select"

        if stage_layer:
            self._stage_layers[3].append(layer)

        # Add text annotations for tagged junctions
        self._add_tag_text_layer(series, stage_layer)

        self._update_tag_list()

    def _add_tag_text_layer(
        self, series: TissueGraphTimeSeries, stage_layer: bool = False,
    ):
        """Create a Points layer showing tag names next to tagged junctions."""
        self._remove_tag_text_layer()
        if not self._show_tag_labels:
            return

        import pandas as pd

        positions, texts, colors, features = build_tag_text_annotations(series)
        if len(positions) == 0:
            return

        features = features.copy()
        features["tag_label"] = texts
        text_layer = self.viewer.add_points(
            positions,
            size=8,
            face_color=colors,
            features=features,
            text={
                "string": "tag_label",
                "color": colors,
                "anchor": "upper_left",
                "size": 10,
            },
            name="[Pipeline] Tag Labels",
        )
        text_layer.mode = "select"

        @text_layer.mouse_drag_callbacks.append
        def _on_tag_click(layer, event):
            self._on_tag_text_clicked(layer)

        text_layer.bind_key("Delete", self._delete_tag_from_text_selection)
        text_layer.bind_key("Backspace", self._delete_tag_from_text_selection)

        self._tagging_text_layer = text_layer
        if stage_layer:
            self._stage_layers[3].append(text_layer)

    def _remove_tag_text_layer(self):
        if (
            self._tagging_text_layer is not None
            and self._tagging_text_layer in self.viewer.layers
        ):
            self.viewer.layers.remove(self._tagging_text_layer)
            for stage in self._stage_layers.values():
                if self._tagging_text_layer in stage:
                    stage.remove(self._tagging_text_layer)
        self._tagging_text_layer = None

    def _on_tag_text_clicked(self, layer):
        """When a tag text point is clicked, select the corresponding edge."""
        selected = set(layer.selected_data)
        if not selected or self._tagging_shapes_layer is None:
            return

        text_features = layer.features
        shapes_features = self._tagging_shapes_layer.features
        shape_indices = set()

        for pt_idx in selected:
            row = text_features.iloc[pt_idx]
            traj_id = int(row["trajectory_id"])
            pair_a = int(row["cell_pair_a"])
            pair_b = int(row["cell_pair_b"])
            frame = int(row["frame"])

            # Find matching shape(s) in the Shapes layer
            for i, srow in shapes_features.iterrows():
                if (
                    int(srow["trajectory_id"]) == traj_id
                    and int(srow["cell_pair_a"]) == pair_a
                    and int(srow["cell_pair_b"]) == pair_b
                    and int(srow["frame"]) == frame
                ):
                    shape_indices.add(i)

        if shape_indices:
            self._tagging_shapes_layer.selected_data = shape_indices
            self._cached_selection = list(shape_indices)
            n = len(shape_indices)
            self.selection_label.setText(f"{n} junction(s) selected")

    def _delete_tag_from_text_selection(self, layer):
        """Delete all tags from the junction under the selected text point."""
        selected = set(layer.selected_data)
        if not selected or self._tagging_series is None:
            return

        features = layer.features
        series = self._tagging_series
        count = 0

        for pt_idx in selected:
            row = features.iloc[pt_idx]
            traj_id = int(row["trajectory_id"])
            pair = (int(row["cell_pair_a"]), int(row["cell_pair_b"]))
            tags_str = row["tags"]
            tags_to_remove = [t.strip() for t in tags_str.split(",") if t.strip()]

            for tag_name in tags_to_remove:
                if traj_id != -1:
                    untag_trajectory(series, traj_id, tag_name)
                key = frozenset(pair)
                for frame in series.frames.values():
                    if key in frame.junctions:
                        untag_junction(frame, pair, tag_name)
            count += 1

        self.status_label.setText(f"Removed tags from {count} junction(s).")
        self._refresh_tagging_layer()

    def _remove_tagging_layer(self):
        self._remove_tag_text_layer()
        if (
            self._tagging_shapes_layer is not None
            and self._tagging_shapes_layer in self.viewer.layers
        ):
            self.viewer.layers.remove(self._tagging_shapes_layer)
            # Also remove from stage layers if present
            for stage in self._stage_layers.values():
                if self._tagging_shapes_layer in stage:
                    stage.remove(self._tagging_shapes_layer)
        self._tagging_shapes_layer = None

    def _poll_selection(self):
        """Poll the shapes layer selection and cache it.

        napari clears selected_data when the mouse leaves the canvas.
        By polling at 200ms we capture the selection while the user
        still has the mouse on the canvas, before it gets cleared.
        """
        if self._tagging_shapes_layer is None:
            return
        try:
            if self._tagging_shapes_layer not in self.viewer.layers:
                return
            current = set(self._tagging_shapes_layer.selected_data)
        except (RuntimeError, AttributeError):
            return
        if current:
            self._cached_selection = list(current)
            n = len(current)
            self.selection_label.setText(f"{n} junction(s) selected")

    def _get_selection(self):
        """Return the current or cached shape selection."""
        if self._tagging_shapes_layer is None:
            return []
        try:
            live = list(self._tagging_shapes_layer.selected_data)
        except (RuntimeError, AttributeError):
            live = []
        if live:
            return live
        return self._cached_selection

    def _refresh_tagging_layer(self):
        """Rebuild the tagging layer from the current series."""
        if self._tagging_series is not None:
            is_pipeline = self._pipeline_stage == PipelineStage.STAGE3_DONE
            self._show_tagging_for_series(
                self._tagging_series, stage_layer=is_pipeline,
            )

    def _tag_selected(self):
        """Apply a tag to all selected junctions in the Shapes layer."""
        tag_name = self.tag_name_edit.text().strip()
        if not tag_name:
            self.status_label.setText("Enter a tag name first.")
            return
        if self._tagging_shapes_layer is None or self._tagging_series is None:
            return

        selected = self._get_selection()
        if not selected:
            self.status_label.setText("Select junctions first by clicking them on the canvas.")
            return

        features = self._tagging_shapes_layer.features
        series = self._tagging_series
        count = 0

        for idx in selected:
            row = features.iloc[idx]
            traj_id = int(row["trajectory_id"])
            pair = (int(row["cell_pair_a"]), int(row["cell_pair_b"]))

            # Tag the trajectory if it exists
            if traj_id != -1:
                tag_trajectory(series, traj_id, tag_name)
                count += 1
            else:
                # Tag the junction directly in all frames where it appears
                key = frozenset(pair)
                for frame in series.frames.values():
                    if key in frame.junctions:
                        tag_junction(frame, pair, tag_name)
                count += 1

        self._cached_selection = []
        self.selection_label.setText("No junctions selected")
        self.status_label.setText(f"Tagged {count} junction(s) as '{tag_name}'.")
        self._refresh_tagging_layer()

    def _untag_selected(self):
        """Remove a tag from all selected junctions in the Shapes layer."""
        tag_name = self.tag_name_edit.text().strip()
        if not tag_name:
            self.status_label.setText("Enter a tag name first.")
            return
        if self._tagging_shapes_layer is None or self._tagging_series is None:
            return

        selected = self._get_selection()
        if not selected:
            self.status_label.setText("Select junctions first.")
            return

        features = self._tagging_shapes_layer.features
        series = self._tagging_series
        count = 0

        for idx in selected:
            row = features.iloc[idx]
            traj_id = int(row["trajectory_id"])
            pair = (int(row["cell_pair_a"]), int(row["cell_pair_b"]))

            if traj_id != -1:
                untag_trajectory(series, traj_id, tag_name)
                count += 1
            else:
                key = frozenset(pair)
                for frame in series.frames.values():
                    if key in frame.junctions:
                        untag_junction(frame, pair, tag_name)
                count += 1

        self._cached_selection = []
        self.selection_label.setText("No junctions selected")
        self.status_label.setText(f"Removed tag '{tag_name}' from {count} junction(s).")
        self._refresh_tagging_layer()

    def _toggle_color_by_tags(self, checked: bool):
        self._color_by_tags = checked
        self._refresh_tagging_layer()

    def _toggle_show_only_tagged(self, checked: bool):
        self._show_only_tagged = checked
        self._refresh_tagging_layer()

    def _toggle_show_tag_labels(self, checked: bool):
        self._show_tag_labels = checked
        self._refresh_tagging_layer()

    def _clear_selected_tag(self):
        """Remove the tag (chosen in the tag list) from the selected junctions.

        If no shapes are selected on the canvas, falls back to removing the
        tag from *all* junctions/trajectories in the series.
        """
        item = self.tag_list_widget.currentItem()
        if item is None or self._tagging_series is None:
            self.status_label.setText("Select a tag from the list first.")
            return

        # Extract tag name (format: "tag_name (N)")
        tag_name = item.text().rsplit(" (", 1)[0]

        selected = self._get_selection()
        if selected and self._tagging_shapes_layer is not None:
            # Remove the tag only from the selected shapes
            features = self._tagging_shapes_layer.features
            series = self._tagging_series
            count = 0
            for idx in selected:
                row = features.iloc[idx]
                traj_id = int(row["trajectory_id"])
                pair = (int(row["cell_pair_a"]), int(row["cell_pair_b"]))
                if traj_id != -1:
                    untag_trajectory(series, traj_id, tag_name)
                    count += 1
                else:
                    key = frozenset(pair)
                    for frame in series.frames.values():
                        if key in frame.junctions:
                            untag_junction(frame, pair, tag_name)
                    count += 1
            self._cached_selection = []
            self.selection_label.setText("No junctions selected")
            self.status_label.setText(
                f"Removed tag '{tag_name}' from {count} selected junction(s)."
            )
        else:
            # No selection — clear from everything
            count = clear_tag(self._tagging_series, tag_name)
            self.status_label.setText(
                f"Cleared tag '{tag_name}' from all {count} item(s)."
            )
        self._refresh_tagging_layer()

    def _update_tag_list(self):
        """Refresh the tag list widget from the current series."""
        self.tag_list_widget.clear()
        if self._tagging_series is None:
            return

        tags = get_all_tags(self._tagging_series)
        if not tags:
            return

        # Count unique edges per tag (deduplicated by cell pair).
        # An edge has the tag if it appears on its trajectory OR any of
        # its per-frame junctions.
        traj_tag_lookup: dict = {}  # frozenset(pair) -> set of tags from trajectory
        for traj in self._tagging_series.edge_trajectories.values():
            if traj.tags:
                for cp in traj.cell_pairs:
                    key = frozenset(cp)
                    traj_tag_lookup.setdefault(key, set()).update(traj.tags)

        for tag in sorted(tags):
            tagged_pairs: set = set()
            # From trajectories
            for pair_key, traj_tags in traj_tag_lookup.items():
                if tag in traj_tags:
                    tagged_pairs.add(pair_key)
            # From junctions (covers junction-level tags and edges without trajectories)
            for frame in self._tagging_series.frames.values():
                for jd in frame.junctions.values():
                    if tag in jd.tags:
                        tagged_pairs.add(frozenset(jd.cell_pair))
            self.tag_list_widget.addItem(f"{tag} ({len(tagged_pairs)})")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_float(text: str):
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def cleanup(self):
        """Clean up background thread and layers if running."""
        self._selection_timer.stop()
        self._remove_tagging_layer()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
