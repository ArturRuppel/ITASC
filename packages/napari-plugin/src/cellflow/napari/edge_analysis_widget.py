"""Edge Analysis tab widget.

Provides graph extraction, T1 detection, edge trajectory analysis,
and junction tagging on a segmentation label stack.

Pixel size / time interval are read from the shared ViewerState.
The resulting tissue series is pushed back into state when the user
accepts it from the Project Panel.
"""
from __future__ import annotations

import logging
from enum import auto, Enum
from typing import Dict, Optional

import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.backend.graph import extract_graphs_from_labels
from cellflow.backend.tagging import (
    clear_tag,
    get_all_tags,
    tag_junction,
    tag_trajectory,
    untag_junction,
    untag_trajectory,
)
from cellflow.backend.topology import detect_t1_events
from cellflow.backend.trajectories import build_edge_trajectories, filter_trajectories
from cellflow.utils.structures import TissueGraphTimeSeries
from .registry import get_state
from .visualization import (
    build_all_centroids,
    build_all_junction_lines,
    build_t1_markers,
    build_tag_text_annotations,
    build_trajectory_lines_with_features,
)

logger = logging.getLogger(__name__)


class _PipelineStage(Enum):
    IDLE = auto()
    STAGE2_DONE = auto()
    STAGE3_DONE = auto()


class EdgeAnalysisWidget(QWidget):
    """Tab widget for graph extraction, T1 detection, and junction tagging."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._state = get_state(napari_viewer)

        # Pipeline state
        self._pipeline_stage = _PipelineStage.IDLE
        self._preview_series: Optional[TissueGraphTimeSeries] = None
        self._current_label_stack: Optional[np.ndarray] = None
        self._source_layer = None
        self._tracked_labels_layer = None
        self._stage_layers: Dict[int, list] = {2: [], 3: []}

        # Tagging state
        self._tagging_shapes_layer = None
        self._tagging_text_layer = None
        self._tagging_series = None
        self._cached_selection: list = []
        self._show_only_tagged = False
        self._color_by_tags = False
        self._show_tag_labels = False

        # Worker handle (thread_worker)
        self._worker = None

        # Poll shapes-layer selection every 200 ms (napari clears it on mouse-leave)
        self._selection_timer = QTimer(self)
        self._selection_timer.setInterval(200)
        self._selection_timer.timeout.connect(self._poll_selection)
        self._selection_timer.start()

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignTop)

        # --- Analyse Tissue (graph extraction + T1 + edge tracking) ---
        self.stage2_toggle = QToolButton()
        self.stage2_toggle.setText("Run Analysis")
        self.stage2_toggle.setArrowType(Qt.RightArrow)
        self.stage2_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.stage2_toggle.setCheckable(True)
        self.stage2_toggle.setChecked(False)
        self.stage2_toggle.setStyleSheet("QToolButton { font-weight: bold; }")
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

        mbhs_row = QHBoxLayout()
        mbhs_row.addWidget(QLabel("Min bg hole (px\u00b2):"))
        self.min_bg_hole_spin = QSpinBox()
        self.min_bg_hole_spin.setMinimum(0)
        self.min_bg_hole_spin.setMaximum(100000)
        self.min_bg_hole_spin.setSingleStep(100)
        self.min_bg_hole_spin.setValue(500)
        self.min_bg_hole_spin.setToolTip(
            "Background regions smaller than this many pixels are treated as\n"
            "segmentation artifacts and ignored during border detection.\n"
            "Increase if small holes between cells create spurious border edges.\n"
            "Set to 0 to disable filtering."
        )
        mbhs_row.addWidget(self.min_bg_hole_spin)
        s2p_layout.addLayout(mbhs_row)

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
        s2p_layout.addLayout(mjl_row)

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
        s2p_layout.addLayout(mtd_row)

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
        s2p_layout.addLayout(mtf_row)

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
        s2p_layout.addLayout(mc_row)

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
        s2p_layout.addLayout(mg_row)

        self.stage2_params.setLayout(s2p_layout)
        self.stage2_params.setVisible(False)
        layout.addWidget(self.stage2_params)

        self.stage2_btn = QPushButton("Run Analysis")
        self.stage2_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }"
        )
        layout.addWidget(self.stage2_btn)

        self.stage2_info = QLabel("")
        self.stage2_info.setWordWrap(True)
        layout.addWidget(self.stage2_info)

        self.active_layer_label = QLabel("")
        self.active_layer_label.setStyleSheet("color: gray;")
        layout.addWidget(self.active_layer_label)

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

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        layout.addWidget(self.cancel_btn)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------
    def _connect_signals(self):
        self.stage2_toggle.toggled.connect(
            lambda checked: self._toggle_stage(checked)
        )
        self.stage2_btn.clicked.connect(self._run_stage2)

        # When series is cleared from state (add/discard via ProjectPanel),
        # reset the pipeline view.
        self._state.tissue_changed.connect(self._on_tissue_cleared)

        self.tag_selected_btn.clicked.connect(self._tag_selected)
        self.untag_selected_btn.clicked.connect(self._untag_selected)
        self.color_by_tags_cb.toggled.connect(self._toggle_color_by_tags)
        self.show_only_tagged_cb.toggled.connect(self._toggle_show_only_tagged)
        self.show_tag_labels_cb.toggled.connect(self._toggle_show_tag_labels)
        self.clear_tag_btn.clicked.connect(self._clear_selected_tag)
        self.refresh_tags_btn.clicked.connect(self._refresh_tagging_layer)
        self.cancel_btn.clicked.connect(self._cancel_worker)

    # ------------------------------------------------------------------
    # Pipeline stage gating
    # ------------------------------------------------------------------
    def _update_pipeline_buttons(self):
        self.stage2_btn.setEnabled(True)

    def _toggle_stage(self, checked: bool):
        self.stage2_params.setVisible(checked)
        self.stage2_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    # ------------------------------------------------------------------
    # Label stack from state
    # ------------------------------------------------------------------
    def _get_active_label_stack(self) -> Optional[np.ndarray]:
        if self._state.tissue.labels is None:
            self.status_label.setText(
                "No segmentation loaded. Load a tissue in the Project Panel first."
            )
            return None

        data = np.asarray(self._state.tissue.labels)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        if data.ndim != 3:
            self.status_label.setText(
                f"Labels must be 2D or 3D (T, H, W), got {data.ndim}D."
            )
            return None

        layer_name = self._state.tissue.labels_layer or "state"
        self.active_layer_label.setText(f"Layer: {layer_name}")
        return data

    # ------------------------------------------------------------------
    # Stage 2: graph extraction
    # ------------------------------------------------------------------
    def _run_stage2(self):
        label_stack = self._get_active_label_stack()
        if label_stack is None:
            return

        self._discard_pipeline()
        self._current_label_stack = label_stack

        pixel_size = self._state.pixel_size
        time_interval = self._state.time_interval
        dilation_radius = self.dilation_radius_spin.value()
        min_overlap_pixels = self.min_overlap_spin.value()
        min_edge_length = self.min_edge_length_spin.value()
        filter_isolated = self.filter_isolated_cb.isChecked()
        min_border_edge_length = self.min_border_edge_spin.value()
        min_bg_hole_size = self.min_bg_hole_spin.value()

        @thread_worker(connect={
            "returned": self._on_stage2_done,
            "errored":  self._on_error,
        })
        def _work():
            return extract_graphs_from_labels(
                label_stack,
                pixel_size=pixel_size,
                time_interval=time_interval,
                dilation_radius=dilation_radius,
                min_overlap_pixels=min_overlap_pixels,
                min_edge_length=min_edge_length,
                filter_isolated=filter_isolated,
                min_border_edge_length=min_border_edge_length,
                min_bg_hole_size=min_bg_hole_size,
            )

        self.stage2_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.cancel_btn.setVisible(True)
        self.status_label.setText("Extracting graphs…")
        self._worker = _work()

    def _on_stage2_done(self, series: TissueGraphTimeSeries):
        self._preview_series = series
        self._pipeline_stage = _PipelineStage.STAGE2_DONE

        total_cells = sum(len(f.cells) for f in series.frames.values())
        total_junctions = sum(len(f.junctions) for f in series.frames.values())
        self.status_label.setText(
            f"Graphs extracted ({series.num_frames} frames, "
            f"{total_cells} cells, {total_junctions} junctions). "
            f"Running analysis…"
        )

        # Stages 2 and 3 are always chained.
        QTimer.singleShot(0, self._run_analysis)

    # ------------------------------------------------------------------
    # Stage 3: T1 + edge trajectories
    # ------------------------------------------------------------------
    def _run_analysis(self):
        if self._preview_series is None:
            return

        series = self._preview_series
        min_junction_length = self.min_junction_length_spin.value()
        max_t1_distance = self.max_t1_distance_spin.value()
        if max_t1_distance == 0:
            max_t1_distance = float("inf")
        min_traj_frames = self.min_traj_frames_spin.value()
        min_completeness = self.min_completeness_spin.value()
        max_gap = self.max_gap_spin.value()

        @thread_worker(connect={
            "returned": self._on_analysis_finished,
            "errored":  self._on_error,
        })
        def _work():
            events = detect_t1_events(
                series,
                min_junction_length=min_junction_length,
                max_t1_distance=max_t1_distance,
            )
            build_edge_trajectories(series, events)
            if min_traj_frames > 1 or min_completeness > 0 or max_gap > 0:
                series.edge_trajectories = filter_trajectories(
                    series,
                    min_frames=min_traj_frames,
                    min_completeness=min_completeness,
                    max_gap=max_gap,
                )
            return series

        self.status_label.setText("Detecting T1 events and building trajectories…")
        self._worker = _work()

    def _on_analysis_finished(self, series: TissueGraphTimeSeries):
        self._preview_series = series
        self._state.set_tissue_series(series)
        self._pipeline_stage = _PipelineStage.STAGE3_DONE

        self._finish_worker()

        n_t1 = len(series.t1_events)
        n_trajs = len(series.edge_trajectories)
        self.stage2_info.setText(f"{n_t1} T1 events, {n_trajs} edge trajectories")
        self.status_label.setText(
            "Analysis complete. Inspect results, then use the Project panel to add to dataset."
        )

        self._remove_stage_layers(3)
        self._show_tagging_for_series(series, stage_layer=True)

        t1_positions = build_t1_markers(series.t1_events)
        if len(t1_positions) > 0:
            layer = self.viewer.add_points(
                t1_positions, size=12, face_color="red",
                symbol="star", name="[Pipeline] T1 Events",
            )
            self._stage_layers[3].append(layer)

        self._update_pipeline_buttons()

    # ------------------------------------------------------------------
    # Dataset inspection (triggered externally, e.g. from Database tab)
    # ------------------------------------------------------------------
    def show_tissue_from_databank(self, series: TissueGraphTimeSeries):
        """Display a tissue from the dataset catalog (read-only inspect)."""
        self._remove_inspect_layers()
        n_t1 = len(series.t1_events)
        n_trajs = len(series.edge_trajectories)
        total_cells = sum(len(f.cells) for f in series.frames.values())
        total_junctions = sum(len(f.junctions) for f in series.frames.values())
        self.status_label.setText(
            f"{total_cells} cells, {total_junctions} junctions, "
            f"{n_t1} T1 events, {n_trajs} edge trajectories"
        )
        if series.edge_trajectories:
            self._add_inspect_layers(series, skip_junctions=True)
            self._show_tagging_for_series(series)
        else:
            self._add_inspect_layers(series)

    # ------------------------------------------------------------------
    # Pipeline cleanup
    # ------------------------------------------------------------------
    def _on_tissue_cleared(self):
        """Called when state.tissue_changed fires; check if series was cleared."""
        if self._state.tissue.series is None and self._pipeline_stage != _PipelineStage.IDLE:
            self._do_pipeline_cleanup()
            self._pipeline_stage = _PipelineStage.IDLE
            self._update_pipeline_buttons()
            self._clear_stage_info()

    def _discard_pipeline(self):
        self._pipeline_stage = _PipelineStage.IDLE
        self._state.set_tissue_series(None)
        self._do_pipeline_cleanup()
        self._update_pipeline_buttons()
        self._clear_stage_info()

    def _do_pipeline_cleanup(self):
        self._remove_all_stage_layers()
        self._remove_tagging_layer()
        self._tagging_series = None
        self.tagging_group.setVisible(False)
        self._preview_series = None
        self._current_label_stack = None
        if self._tracked_labels_layer is not None:
            if self._tracked_labels_layer in self.viewer.layers:
                self.viewer.layers.remove(self._tracked_labels_layer)
            self._tracked_labels_layer = None
        if self._source_layer is not None:
            try:
                self._source_layer.visible = True
            except Exception:
                pass
            self._source_layer = None

    def _clear_stage_info(self):
        self.active_layer_label.setText("")
        self.stage2_info.setText("")

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------
    def _finish_worker(self):
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        self._update_pipeline_buttons()

    def _on_error(self, exc: Exception):
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)
        self._update_pipeline_buttons()
        self.status_label.setText(f"Error: {exc}")
        logger.exception("Worker failed", exc_info=exc)

    def _cancel_worker(self):
        if self._worker is not None:
            try:
                self._worker.quit()
            except Exception:
                pass
            self._worker = None
        self.cancel_btn.setVisible(False)
        self.progress_bar.setVisible(False)
        self._update_pipeline_buttons()
        self.status_label.setText("Cancelled.")

    # ------------------------------------------------------------------
    # Napari layer management
    # ------------------------------------------------------------------
    def _remove_stage_layers(self, stage: int):
        for layer in self._stage_layers[stage]:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self._stage_layers[stage] = []

    def _remove_all_stage_layers(self):
        for stage in (2, 3):
            self._remove_stage_layers(stage)

    # inspect layers (dataset view, read-only)
    _inspect_junction_layer = None
    _inspect_centroid_layer = None
    _inspect_t1_layer = None

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
        vis_j = self._inspect_junction_layer.visible if (self._inspect_junction_layer and self._inspect_junction_layer in self.viewer.layers) else True
        vis_c = self._inspect_centroid_layer.visible if (self._inspect_centroid_layer and self._inspect_centroid_layer in self.viewer.layers) else True
        vis_t = self._inspect_t1_layer.visible if (self._inspect_t1_layer and self._inspect_t1_layer in self.viewer.layers) else True
        self._remove_inspect_layers()
        j, c, t = self._make_layers(series, prefix="", skip_junctions=skip_junctions)
        if j is not None:
            j.visible = vis_j
        if c is not None:
            c.visible = vis_c
        if t is not None:
            t.visible = vis_t
        self._inspect_junction_layer = j
        self._inspect_centroid_layer = c
        self._inspect_t1_layer = t

    def _make_layers(
        self,
        series: TissueGraphTimeSeries,
        prefix: str = "",
        skip_junctions: bool = False,
    ):
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
                centroids, size=5, face_color="yellow",
                name=f"{prefix}Cell Centroids",
            )

        t1_positions = build_t1_markers(series.t1_events)
        if len(t1_positions) > 0:
            t1_layer = self.viewer.add_points(
                t1_positions, size=12, face_color="red",
                symbol="star", name=f"{prefix}T1 Events",
            )

        return junction_layer, centroid_layer, t1_layer

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------
    def _show_tagging_for_series(
        self, series: TissueGraphTimeSeries, stage_layer: bool = False,
    ):
        self._tagging_series = series
        vis_traj = self._tagging_shapes_layer.visible if (self._tagging_shapes_layer and self._tagging_shapes_layer in self.viewer.layers) else True
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
            visible=vis_traj,
        )
        self._tagging_shapes_layer = layer
        self._cached_selection = []
        layer.mode = "select"

        if stage_layer:
            self._stage_layers[3].append(layer)

        self._add_tag_text_layer(series, stage_layer)
        self._update_tag_list()

    def _add_tag_text_layer(
        self, series: TissueGraphTimeSeries, stage_layer: bool = False,
    ):
        vis_tags = self._tagging_text_layer.visible if (self._tagging_text_layer and self._tagging_text_layer in self.viewer.layers) else True
        self._remove_tag_text_layer()
        if not self._show_tag_labels:
            return

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
            visible=vis_tags,
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
            self.selection_label.setText(f"{len(shape_indices)} junction(s) selected")

    def _delete_tag_from_text_selection(self, layer):
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
            for stage in self._stage_layers.values():
                if self._tagging_shapes_layer in stage:
                    stage.remove(self._tagging_shapes_layer)
        self._tagging_shapes_layer = None

    def _poll_selection(self):
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
            self.selection_label.setText(f"{len(current)} junction(s) selected")

    def _get_selection(self):
        if self._tagging_shapes_layer is None:
            return []
        try:
            live = list(self._tagging_shapes_layer.selected_data)
        except (RuntimeError, AttributeError):
            live = []
        return live if live else self._cached_selection

    def _refresh_tagging_layer(self):
        if self._tagging_series is not None:
            is_pipeline = self._pipeline_stage == _PipelineStage.STAGE3_DONE
            self._show_tagging_for_series(self._tagging_series, stage_layer=is_pipeline)

    def _tag_selected(self):
        tag_name = self.tag_name_edit.text().strip()
        if not tag_name:
            self.status_label.setText("Enter a tag name first.")
            return
        if self._tagging_shapes_layer is None or self._tagging_series is None:
            return

        selected = self._get_selection()
        if not selected:
            self.status_label.setText(
                "Select junctions first by clicking them on the canvas."
            )
            return

        features = self._tagging_shapes_layer.features
        series = self._tagging_series
        count = 0

        for idx in selected:
            row = features.iloc[idx]
            traj_id = int(row["trajectory_id"])
            pair = (int(row["cell_pair_a"]), int(row["cell_pair_b"]))

            if traj_id != -1:
                tag_trajectory(series, traj_id, tag_name)
                count += 1
            else:
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
        item = self.tag_list_widget.currentItem()
        if item is None or self._tagging_series is None:
            self.status_label.setText("Select a tag from the list first.")
            return

        tag_name = item.text().rsplit(" (", 1)[0]
        selected = self._get_selection()

        if selected and self._tagging_shapes_layer is not None:
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
            count = clear_tag(self._tagging_series, tag_name)
            self.status_label.setText(
                f"Cleared tag '{tag_name}' from all {count} item(s)."
            )
        self._refresh_tagging_layer()

    def _update_tag_list(self):
        self.tag_list_widget.clear()
        if self._tagging_series is None:
            return

        tags = get_all_tags(self._tagging_series)
        if not tags:
            return

        traj_tag_lookup: dict = {}
        for traj in self._tagging_series.edge_trajectories.values():
            if traj.tags:
                for cp in traj.cell_pairs:
                    key = frozenset(cp)
                    traj_tag_lookup.setdefault(key, set()).update(traj.tags)

        for tag in sorted(tags):
            tagged_pairs: set = set()
            for pair_key, traj_tags in traj_tag_lookup.items():
                if tag in traj_tags:
                    tagged_pairs.add(pair_key)
            for frame in self._tagging_series.frames.values():
                for jd in frame.junctions.values():
                    if tag in jd.tags:
                        tagged_pairs.add(frozenset(jd.cell_pair))
            self.tag_list_widget.addItem(f"{tag} ({len(tagged_pairs)})")

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------
    def get_params(self) -> dict:
        return {
            "dilation_radius":      self.dilation_radius_spin.value(),
            "min_overlap":          self.min_overlap_spin.value(),
            "min_edge_length":      self.min_edge_length_spin.value(),
            "filter_isolated":      self.filter_isolated_cb.isChecked(),
            "min_border_edge_length": self.min_border_edge_spin.value(),
            "min_bg_hole_size":     self.min_bg_hole_spin.value(),
            "min_junction_length":  self.min_junction_length_spin.value(),
            "max_t1_distance":      self.max_t1_distance_spin.value(),
            "min_traj_frames":      self.min_traj_frames_spin.value(),
            "min_completeness":     self.min_completeness_spin.value(),
            "max_gap":              self.max_gap_spin.value(),
        }

    def set_params(self, data: dict) -> None:
        if "dilation_radius" in data:
            self.dilation_radius_spin.setValue(int(data["dilation_radius"]))
        if "min_overlap" in data:
            self.min_overlap_spin.setValue(int(data["min_overlap"]))
        if "min_edge_length" in data:
            self.min_edge_length_spin.setValue(float(data["min_edge_length"]))
        if "filter_isolated" in data:
            self.filter_isolated_cb.setChecked(bool(data["filter_isolated"]))
        if "min_border_edge_length" in data:
            self.min_border_edge_spin.setValue(float(data["min_border_edge_length"]))
        if "min_bg_hole_size" in data:
            self.min_bg_hole_spin.setValue(int(data["min_bg_hole_size"]))
        if "min_junction_length" in data:
            self.min_junction_length_spin.setValue(float(data["min_junction_length"]))
        if "max_t1_distance" in data:
            self.max_t1_distance_spin.setValue(float(data["max_t1_distance"]))
        if "min_traj_frames" in data:
            self.min_traj_frames_spin.setValue(int(data["min_traj_frames"]))
        if "min_completeness" in data:
            self.min_completeness_spin.setValue(float(data["min_completeness"]))
        if "max_gap" in data:
            self.max_gap_spin.setValue(int(data["max_gap"]))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup(self):
        self._selection_timer.stop()
        self._remove_tagging_layer()
        if self._worker is not None:
            try:
                self._worker.quit()
            except Exception:
                pass
            self._worker = None
