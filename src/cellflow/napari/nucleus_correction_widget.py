"""Correction section widget for the nucleus workflow."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker as _thread_worker
from napari.utils.colormaps import Colormap
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._widget_helpers import (
    btn as _btn,
    dspin as _dspin,
    heading as _heading,
    make_status as _make_status,
    tool_btn as _tool_btn,
)
from cellflow.napari._paths import NucleusArtifactPaths
from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.contact_analysis_visualization import (
    _categorical_colors,
    _nucleus_centroids_by_track,
    _rasterize_track_image,
)
from cellflow.database.tracked import (
    read_full_tracked_stack,
    write_tracked_frame,
)
from cellflow.database.validation import (
    add_anchor,
    add_correction,
    invalidate_track,
    is_track_validated,
    read_corrections,
    read_validated_cells_at_frame,
    read_validated_frames,
    read_validated_tracks,
    remap_validated_tracks,
    write_corrections,
)
from cellflow.napari.ui_style import (
    add_block_checkbox_row,
    add_block_pair_row,
    block_grid,
    compact_spinbox,
    danger_button,
    stage_header_label,
)
from cellflow.napari.validated_overlay_controller import (
    ValidatedOverlayController,
)
from cellflow.napari.widgets import CollapsibleSection
from cellflow.tracking_ultrack.corrections import Correction
from cellflow.tracking_ultrack.extend import extend_track_from_db as _extend_track_from_db
from cellflow.tracking_ultrack.swap_candidate import (
    SwapCandidate as _SwapCandidate,
    _SwapCursor,
    list_swap_candidates,
    step_larger as _step_larger,
    step_smaller as _step_smaller,
)
from cellflow.tracking_ultrack.retracker import retrack_frame_constrained

logger = logging.getLogger(__name__)

_TRACKED_LAYER = "Tracked: Nucleus"
_CORRECTION_TRACKED_LAYER = "[Correction] Nucleus Labels"
_CORRECTION_TRACK_LAYER = "[Correction] Nucleus tracks"
_CORRECTION_CELL_ZAVG_LAYER = "[Correction] Cell z-avg"
_CORRECTION_NUC_ZAVG_LAYER = "[Correction] Nucleus z-avg"
_CORRECTION_NLS_ZAVG_LAYER = "[Correction] NLS z-avg"
_NUCLEUS_TRACK_COLOR_SCALE = 0.65

_DEFAULT_DEPENDENCIES = {
    "extend_track_from_db": _extend_track_from_db,
    "read_corrections": lambda *args, **kwargs: read_corrections(*args, **kwargs),
    "read_validated_tracks": (
        lambda *args, **kwargs: read_validated_tracks(*args, **kwargs)
    ),
    "thread_worker": _thread_worker,
}


class NucleusCorrectionWidget(QWidget):
    """Qt controls for nucleus tracking correction workflows."""

    def __init__(
        self,
        viewer,
        *,
        edit_callback=None,
        pos_dir_provider: Callable[[], Path | None] | None = None,
        refresh_refinement_callback: Callable[[], None] | None = None,
        dependencies: dict[str, Callable] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir_provider = pos_dir_provider
        self._local_pos_dir: Path | None = None
        self._refresh_refinement_callback = refresh_refinement_callback or (lambda: None)
        self._dependencies = (
            _DEFAULT_DEPENDENCIES if dependencies is None
            else {**_DEFAULT_DEPENDENCIES, **dependencies}
        )
        self._edit_callback = edit_callback or self._on_cells_edited
        self._correction_owned_layers: set[str] = set()
        self._correction_view_state: dict | None = None
        self._swap_cursor: _SwapCursor | None = None
        self._validated_overlay = ValidatedOverlayController(
            self.viewer,
            tracked_layer_provider=self._correction_tracked_layer,
            pos_dir_provider=lambda: self._pos_dir,
            current_t_provider=self._current_t,
            owned_layers=self._correction_owned_layers,
        )
        self._setup_ui()

    @property
    def _pos_dir(self):
        if self._pos_dir_provider is not None:
            return self._pos_dir_provider()
        return self._local_pos_dir

    @_pos_dir.setter
    def _pos_dir(self, value) -> None:
        self._local_pos_dir = value

    def _dependency(self, name: str):
        return self._dependencies[name]

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        inner = QWidget(self)
        group_lay = QVBoxLayout(inner)
        group_lay.setContentsMargins(0, 0, 0, 0)
        group_lay.setSpacing(6)

        self.active_btn = _tool_btn(
            "⏻",
            "Activate correction mode and show correction layers and controls.",
            checkable=True,
        )
        self.active_btn.setToolTip(
            "Activate correction mode and show correction layers and controls."
        )
        self.params_btn = _tool_btn(
            "⚙", "Show correction parameters.", checkable=True
        )
        self.shortcuts_btn = _tool_btn(
            "📖", "Show correction shortcuts.", checkable=True
        )

        self.save_tracked_btn = _tool_btn(
            "💾", "Save corrected tracked nucleus labels to disk (S)."
        )
        self.extend_back_btn = _tool_btn(
            "◀", "Extend selected track one frame backward (A)."
        )
        self.extend_fwd_btn = _tool_btn(
            "▶", "Extend selected track one frame forward (D)."
        )
        self.retrack_back_btn = _tool_btn(
            "↶", "Retrack all labels backward from current frame (Q)."
        )
        self.retrack_fwd_btn = _tool_btn(
            "↷", "Retrack all labels forward from current frame (E)."
        )
        self.reassign_ids_btn = _tool_btn(
            "#", "Reassign cell IDs to contiguous range 1-N."
        )
        self.validate_track_btn = _tool_btn(
            "✓", "Lock selected cell geometry in every frame where it appears (V)."
        )
        self.anchor_here_btn = _tool_btn(
            "⚓", "Anchor selected cell identity at the current frame (B)."
        )
        self.remove_unvalidated_btn = _tool_btn(
            "🗑",
            "Remove nucleus label pixels not marked validated for their frame.",
        )
        danger_button(self.remove_unvalidated_btn)

        self.commit_btn = _btn(
            "Commit",
            "Reassign cell IDs, remove unvalidated labels, and save tracked labels.",
        )

        self.status_lbl = _make_status()

        self.validation_counter_lbl = QLabel("")
        self.validation_counter_lbl.setWordWrap(True)

        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
            show_cleanup=False,
        )
        self.correction_widget.set_edit_callback(self._edit_callback)
        self.correction_widget._status.setVisible(False)

        extend_retrack_inner = QWidget(self)
        extend_retrack_lay = QVBoxLayout(extend_retrack_inner)
        extend_retrack_lay.setContentsMargins(0, 0, 0, 0)
        extend_retrack_lay.setSpacing(6)
        extend_retrack_lay.addWidget(self.correction_widget._outline_btn)

        extend_retrack_lay.addWidget(_heading("Extend"))
        g = block_grid(horizontal_spacing=12)
        self.extend_max_dist_spin = _dspin(0, 500, 40.0, 1.0, 1)
        self.extend_area_weight_spin = _dspin(0, 10, 1.0, 0.1, 2)
        self.extend_iou_weight_spin = _dspin(0, 10, 1.0, 0.1, 2)
        self.extend_distance_weight_spin = _dspin(0, 10, 0.05, 0.01, 3)
        self.extend_overlap_penalty_spin = _dspin(0, 10, 1.0, 0.1, 2)
        self.extend_greedy_overwrite_check = QCheckBox("Greedy overwrite")
        add_block_pair_row(
            g,
            0,
            "Max\ndistance:",
            compact_spinbox(self.extend_max_dist_spin),
            "Area\nweight:",
            compact_spinbox(self.extend_area_weight_spin),
        )
        add_block_pair_row(
            g,
            1,
            "IoU\nweight:",
            compact_spinbox(self.extend_iou_weight_spin),
            "Distance\nweight:",
            compact_spinbox(self.extend_distance_weight_spin),
        )
        self.swap_radius_spin = _dspin(0, 500, 40.0, 1.0, 1)
        add_block_pair_row(
            g,
            2,
            "Overlap\npenalty:",
            compact_spinbox(self.extend_overlap_penalty_spin),
        )
        add_block_pair_row(g, 3, "Swap\nradius:", compact_spinbox(self.swap_radius_spin))
        add_block_checkbox_row(g, 4, self.extend_greedy_overwrite_check)
        extend_retrack_lay.addLayout(g)

        extend_retrack_lay.addWidget(_heading("Retrack"))
        g = block_grid(horizontal_spacing=12)
        self.retrack_max_dist_spin = _dspin(0, 500, 20.0, 1.0, 1)
        add_block_pair_row(g, 0, "Max\ndistance:", compact_spinbox(self.retrack_max_dist_spin))
        extend_retrack_lay.addLayout(g)
        self.extend_retrack_params_section = CollapsibleSection(
            "Extend / Retrack Parameters",
            extend_retrack_inner,
            expanded=False,

        )
        self._flatten_embedded_correction_section(self.extend_retrack_params_section)
        self.extend_retrack_params_section.setVisible(False)
        self.extend_params_section = self.extend_retrack_params_section
        self.retrack_params_section = self.extend_retrack_params_section
        group_lay.addWidget(self.extend_retrack_params_section)

        self.shortcuts_section = CollapsibleSection(
            "Correction Shortcuts",
            self._build_shortcuts_widget(),
            expanded=False,

        )
        self._flatten_embedded_correction_section(self.shortcuts_section)
        self.shortcuts_section.setVisible(False)
        group_lay.addWidget(self.shortcuts_section)
        group_lay.addWidget(self.correction_widget)
        self.correction_widget.setVisible(False)
        self.toolbar = self._build_correction_toolbar()
        self.toolbar.setVisible(False)
        group_lay.addWidget(self.toolbar)
        group_lay.addWidget(self.status_lbl)
        group_lay.addWidget(self.validation_counter_lbl)
        group_lay.addWidget(self.correction_widget._attrib_lbl)
        self.validation_counter_lbl.setVisible(False)
        self.correction_widget._attrib_lbl.setVisible(False)

        self.header = self._build_correction_header()

        self.section = CollapsibleSection(
            "Correction",
            inner,
            expanded=False,

        )
        self.section._toggle.setVisible(False)
        self.section._toggle.setEnabled(False)

        self.correction_active_btn = self.active_btn
        self.correction_shortcuts_btn = self.shortcuts_btn
        self.correction_status_lbl = self.status_lbl
        self.correction_mode_section = self.section
        self._correction_active_content_visible = False
        self._connect_signals()

    @staticmethod
    def _flatten_embedded_correction_section(section: CollapsibleSection) -> None:
        section.set_header_visible(False)
        section.layout().setContentsMargins(0, 0, 0, 0)
        section._content_frame.layout().setContentsMargins(0, 0, 0, 0)
        section._content_frame.setStyleSheet(
            "QFrame#collapsible_content { border: none; margin: 0px; }"
        )

    def _build_correction_header(self) -> QWidget:
        """Build the stage-style correction header with top-level controls."""
        header = QWidget(self)
        row = QHBoxLayout(header)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self.header_lbl = QLabel("Correction")
        stage_header_label(self.header_lbl, "nucleus")
        row.addWidget(self.header_lbl)
        row.addStretch(1)
        row.addWidget(self.shortcuts_btn)
        row.addWidget(self.params_btn)
        row.addWidget(self.active_btn)
        return header

    def _build_correction_toolbar(self) -> QWidget:
        """Build the active-only correction action toolbar."""
        toolbar = QWidget(self)
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        def _sep() -> QFrame:
            line = QFrame()
            line.setFrameShape(QFrame.VLine)
            line.setFrameShadow(QFrame.Sunken)
            return line

        groups: list[tuple] = [
            (self.save_tracked_btn,),
            (self.extend_back_btn, self.extend_fwd_btn),
            (self.retrack_back_btn, self.retrack_fwd_btn),
            (self.validate_track_btn, self.anchor_here_btn),
            (self.reassign_ids_btn, self.remove_unvalidated_btn),
        ]
        for i, group in enumerate(groups):
            if i > 0:
                row.addWidget(_sep())
            for b in group:
                row.addWidget(b)
        row.addStretch(1)
        return toolbar

    def _build_shortcuts_widget(self) -> QWidget:
        group = QGroupBox("Correction shortcuts")
        grid = QGridLayout(group)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)
        row = 0
        row = CorrectionWidget._add_shortcut_group(
            grid,
            "Track Workflow",
            [
                ("V", "Validate selected track"),
                ("B", "Anchor selected cell at current frame"),
                ("A / D", "Extend selected track backward / forward"),
                ("Q / E", "Retrack backward / forward"),
                ("Z / C", "Swap with smaller / larger hypothesis fragment"),
                ("S", "Save tracked labels"),
            ],
            start_row=row,
            is_first=True,
        )
        row = CorrectionWidget._add_shortcut_group(
            grid,
            "Selection",
            [
                ("Left-click", "Select / highlight cell"),
                ("Shift+Left / Shift+Right", "Previous / next cell"),
            ],
            start_row=row,
        )
        row = CorrectionWidget._add_shortcut_group(
            grid,
            "Manual Labels",
            [
                ("Middle-click or Delete", "Erase cell"),
                ("Ctrl+Left-click", "Merge selected with clicked cell"),
                ("Right-click variants", "Swap labels"),
                ("Shift+Left-drag", "Draw / extend cell path"),
                ("Shift+Right-drag", "Split by drawn line"),
            ],
            start_row=row,
        )
        row = CorrectionWidget._add_shortcut_group(
            grid, "History", [("Ctrl+Z", "Undo")], start_row=row
        )
        grid.setColumnStretch(1, 1)
        return group

    def _connect_signals(self) -> None:
        self.save_tracked_btn.clicked.connect(self._on_save_tracked)
        self.reassign_ids_btn.clicked.connect(self._on_reassign_ids)
        self.validate_track_btn.clicked.connect(self._on_validate_track)
        self.anchor_here_btn.clicked.connect(self._on_anchor_here)
        self.extend_back_btn.clicked.connect(self._on_extend_backward)
        self.extend_fwd_btn.clicked.connect(self._on_extend_forward)
        self.retrack_back_btn.clicked.connect(self._on_retrack_backward)
        self.retrack_fwd_btn.clicked.connect(self._on_retrack_forward)
        self.remove_unvalidated_btn.clicked.connect(
            self._on_remove_unvalidated_labels
        )
        self.commit_btn.clicked.connect(self._on_commit)
        self.params_btn.toggled.connect(
            self._on_correction_params_button_toggled
        )
        self.shortcuts_btn.toggled.connect(
            self._on_correction_shortcuts_button_toggled
        )
        self.active_btn.toggled.connect(self._on_correction_active_button_toggled)
        self.correction_widget._activate_btn.toggled.connect(
            self._on_correction_mode_toggled
        )
        self._install_correction_shortcuts()

    @staticmethod
    def _set_checked_without_signal(button, checked: bool) -> None:
        old = button.blockSignals(True)
        try:
            button.setChecked(checked)
        finally:
            button.blockSignals(old)

    def _sync_correction_panel_visibility(self) -> None:
        show_params = self.params_btn.isChecked()
        show_shortcuts = self.shortcuts_btn.isChecked()
        show_active = self._correction_active_content_visible

        self.extend_retrack_params_section.setVisible(show_params)
        self.shortcuts_section.setVisible(show_shortcuts)
        self.correction_widget.setVisible(show_active)
        self.toolbar.setVisible(show_active)
        self.validation_counter_lbl.setVisible(show_active)
        self.correction_widget._attrib_lbl.setVisible(show_active)

        if show_params or show_shortcuts or show_active:
            self.section.expand()
        else:
            self.section.collapse()

    def _on_correction_params_button_toggled(self, checked: bool) -> None:
        self.extend_retrack_params_section._toggle.setChecked(checked)
        if checked:
            self._set_checked_without_signal(self.shortcuts_btn, False)
            self.shortcuts_section._toggle.setChecked(False)
        self._sync_correction_panel_visibility()

    def _on_correction_shortcuts_button_toggled(self, checked: bool) -> None:
        self.shortcuts_section._toggle.setChecked(checked)
        if checked:
            self._set_checked_without_signal(self.params_btn, False)
            self.extend_retrack_params_section._toggle.setChecked(False)
        self._sync_correction_panel_visibility()

    @property
    def _paths(self) -> NucleusArtifactPaths | None:
        return NucleusArtifactPaths(self._pos_dir) if self._pos_dir else None

    def _tracked_path(self):
        return self._paths.tracked if self._paths else None

    def _cell_zavg_path(self):
        return self._paths.cell_zavg if self._paths else None

    def _nucleus_zavg_path(self):
        return self._paths.nucleus_zavg if self._paths else None

    def _cell_foreground_path(self):
        return self._paths.cell_foreground if self._paths else None

    def _nucleus_foreground_path(self):
        return self._paths.nucleus_foreground if self._paths else None

    def _nls_zavg_path(self):
        return self._paths.nls_zavg if self._paths else None

    def _ultrack_db_path(self):
        return self._paths.ultrack_db if self._paths else None

    def _current_t(self) -> int:
        step = self.viewer.dims.current_step
        return int(step[0]) if len(step) >= 1 else 0

    @staticmethod
    def _frame_view_2d(arr: np.ndarray, t: int) -> np.ndarray | None:
        if arr.ndim < 3 or t < 0 or t >= arr.shape[0]:
            return None
        view = arr[t]
        while view.ndim > 2:
            if view.shape[0] != 1:
                return None
            view = view[0]
        return view

    def _current_cell_ids(self, t: int) -> set[int]:
        layer = self._correction_tracked_layer()
        if layer is None:
            return set()
        frame = self._frame_view_2d(np.asarray(layer.data), t)
        if frame is None:
            return set()
        return set(int(value) for value in np.unique(frame)) - {0}

    def _correction_tracked_layer(self):
        if _CORRECTION_TRACKED_LAYER in self.viewer.layers:
            return self.viewer.layers[_CORRECTION_TRACKED_LAYER]
        if _TRACKED_LAYER in self.viewer.layers:
            return self.viewer.layers[_TRACKED_LAYER]
        return None

    def _capture_correction_view_state(self) -> None:
        selected = [layer.name for layer in self.viewer.layers.selection]
        active = self.viewer.layers.selection.active
        self._correction_view_state = {
            "visibility": {layer.name: bool(layer.visible) for layer in self.viewer.layers},
            "active": active.name if active is not None else None,
            "selected": selected,
        }

    def _restore_correction_view_state(self) -> None:
        state = self._correction_view_state or {}
        visibility = state.get("visibility", {})
        for name, visible in visibility.items():
            if name in self.viewer.layers:
                self.viewer.layers[name].visible = bool(visible)
        self.viewer.layers.selection.clear()
        for name in state.get("selected", ()):
            if name in self.viewer.layers:
                self.viewer.layers.selection.add(self.viewer.layers[name])
        active_name = state.get("active")
        if active_name in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[active_name]
        self._correction_view_state = None

    def _remove_correction_owned_layers(self) -> None:
        for name in list(self._correction_owned_layers):
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])
        self._correction_owned_layers.clear()

    def _add_correction_image_layer(self, data: np.ndarray, name: str, colormap: str) -> None:
        arr = np.asarray(data, dtype=np.float32)
        if colormap == "bop_blue":
            colormap = Colormap(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.25, 1.0, 1.0]],
                name="bop_blue",
            )
        self.viewer.add_image(arr, name=name, colormap=colormap, blending="minimum")
        self._correction_owned_layers.add(name)

    def _add_correction_track_layer(self, labels: np.ndarray) -> dict:
        labels = np.asarray(labels)
        label_ids = np.asarray(
            sorted(int(v) for v in np.unique(labels) if int(v) != 0)
        )
        label_colors = _categorical_colors(label_ids)
        color_map: dict[int | None, tuple[float, float, float, float] | str] = {
            None: "transparent",
            0: "transparent",
        }
        for label_id, color in zip(label_ids, label_colors, strict=True):
            rgba = np.asarray(color, dtype=np.float32).copy()
            rgba[:3] *= _NUCLEUS_TRACK_COLOR_SCALE
            color_map[int(label_id)] = tuple(float(c) for c in rgba)

        shape = (
            (1, int(labels.shape[0]), int(labels.shape[1]))
            if labels.ndim == 2
            else tuple(int(v) for v in labels.shape[:3])
        )
        track_image = _rasterize_track_image(
            _nucleus_centroids_by_track(labels),
            color_map,
            shape,
        )
        self.viewer.add_image(
            track_image,
            name=_CORRECTION_TRACK_LAYER,
            rgb=True,
            opacity=0.9,
            blending="additive",
        )
        self._correction_owned_layers.add(_CORRECTION_TRACK_LAYER)
        return color_map

    def _correction_status(self, msg: str) -> None:
        self.status_lbl.setText(msg)
        self.status_lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    def _on_save_tracked(self) -> None:
        tracked_path = self._tracked_path()
        if tracked_path is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer to save."); return
        if layer.data.ndim != 3:
            self._correction_status("Tracked layer is not a 3D stack."); return
        n = layer.data.shape[0]
        for t in range(n):
            write_tracked_frame(tracked_path, t, np.asarray(layer.data[t]))
        self._correction_status(f"Saved {n} frame(s) to {tracked_path.name}.")

    @staticmethod
    def _reassign_ids_stack(stack: np.ndarray) -> tuple[np.ndarray, int, dict[int, int]]:
        unique_ids = np.unique(stack)
        unique_ids = unique_ids[unique_ids != 0]
        if unique_ids.size == 0:
            return stack, 0, {}
        lut = np.zeros(int(unique_ids.max()) + 1, dtype=np.uint32)
        old_to_new: dict[int, int] = {}
        for new_id, old_id in enumerate(unique_ids, start=1):
            lut[old_id] = new_id
            old_to_new[int(old_id)] = new_id
        return lut[stack], len(unique_ids), old_to_new

    def _refresh_tracked_layer_from_disk(self) -> None:
        """Overwrite the 'Tracked: Nucleus' layer data from the saved TIFF.

        Called when correction mode is deactivated so the pipeline widget's
        re-solve reads the latest saved state rather than stale in-memory data.
        Does nothing if the file does not exist or if the layer is absent.
        """
        tracked_path = self._tracked_path()
        if tracked_path is None or not tracked_path.exists():
            return
        if _TRACKED_LAYER not in self.viewer.layers:
            return
        try:
            from cellflow.database.tracked import read_full_tracked_stack
            data = np.asarray(read_full_tracked_stack(tracked_path), dtype=np.uint32)
            self.viewer.layers[_TRACKED_LAYER].data = data
        except Exception:
            pass

    def _load_correction_layers_from_disk(self) -> bool:
        tracked_path = self._tracked_path()
        if tracked_path is None or not tracked_path.exists():
            self._correction_status("No tracked labels file found.")
            return False

        self._remove_correction_owned_layers()
        self._remove_other_correction_prefix_layers()
        stack = read_full_tracked_stack(tracked_path)

        for data, name, cmap in (
            (
                self._cell_foreground_path(),
                _CORRECTION_CELL_ZAVG_LAYER,
                "gray",
            ),
            (
                self._nucleus_foreground_path(),
                _CORRECTION_NUC_ZAVG_LAYER,
                "I Purple",
            ),
        ):
            if data is not None and data.exists():
                self._add_correction_image_layer(
                    np.asarray(tifffile.imread(str(data)), dtype=np.float32),
                    name,
                    cmap,
                )

        nls_path = self._nls_zavg_path()
        if nls_path is not None and nls_path.exists():
            self._add_correction_image_layer(
                np.asarray(tifffile.imread(str(nls_path)), dtype=np.float32),
                _CORRECTION_NLS_ZAVG_LAYER,
                "I Orange",
            )

        labels_layer = self.viewer.add_labels(
            stack,
            name=_CORRECTION_TRACKED_LAYER,
            blending="additive",
        )
        labels_layer.blending = "additive"
        self._correction_owned_layers.add(_CORRECTION_TRACKED_LAYER)
        color_map = self._add_correction_track_layer(stack)
        try:
            from napari.utils.colormaps import DirectLabelColormap
            labels_layer.colormap = DirectLabelColormap(color_dict=color_map)
        except Exception:
            pass

        self._correction_status(f"Loaded tracked stack {stack.shape} into correction mode.")
        return True

    def _remove_other_correction_prefix_layers(self) -> None:
        for layer in list(self.viewer.layers):
            if layer.name.startswith("[Correction]") and layer.name not in self._correction_owned_layers:
                if isinstance(layer, napari.layers.Labels):
                    self.viewer.layers.remove(layer)

    def _on_load_tracked(self) -> None:
        self._load_correction_layers_from_disk()

    def _on_reassign_ids(self) -> None:
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        stack = np.asarray(layer.data)
        self._correction_status("Reassigning cell IDs...")

        @self._dependency("thread_worker")(connect={
            "returned": self._on_reassign_ids_done,
            "errored": self._on_correction_worker_error,
        })
        def _worker():
            return self._reassign_ids_stack(stack)

        _worker()

    def _on_reassign_ids_done(self, result: tuple) -> None:
        remapped, n_cells, old_to_new = result
        layer = self._correction_tracked_layer()
        if layer is not None:
            layer.data = remapped
        if self._pos_dir is not None and old_to_new:
            remap_validated_tracks(self._pos_dir, old_to_new)
        self._correction_status(
            f"Reassigned {n_cells} cell IDs to range 1-{n_cells}. Unsaved."
        )

    def _commit_reassign_ids(self, layer) -> int:
        remapped, n_cells, old_to_new = self._reassign_ids_stack(np.asarray(layer.data))
        layer.data = remapped
        if self._pos_dir is not None and old_to_new:
            remap_validated_tracks(self._pos_dir, old_to_new)
        return int(n_cells)

    def _selected_correction_target(self) -> tuple[int, int, float, float] | None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return None
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return None
        cell_id = int(getattr(self.correction_widget, "_selected_label", 0) or 0)
        if cell_id == 0:
            self._correction_status("No cell selected (left-click first)."); return None
        t = self._current_t()
        data = np.asarray(layer.data)
        frame = self._frame_view_2d(data, t) if data.ndim >= 3 else data
        if frame is None or not np.any(frame == cell_id):
            self._correction_status(f"Cell {cell_id} not present at t={t}."); return None
        yy, xx = np.nonzero(frame == cell_id)
        return cell_id, t, float(np.mean(yy)), float(np.mean(xx))

    def _validated_correction_for_frame(
        self, cell_id: int, t: int, data: np.ndarray
    ) -> Correction | None:
        frame = self._frame_view_2d(data, t) if data.ndim >= 3 else data
        if frame is None or not np.any(frame == cell_id):
            return None
        yy, xx = np.nonzero(frame == cell_id)
        return Correction(
            cell_id=int(cell_id),
            t=int(t),
            kind="validated",
            y=float(np.mean(yy)),
            x=float(np.mean(xx)),
        )

    def _on_validate_track(self) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        cell_id = int(getattr(self.correction_widget, "_selected_label", 0) or 0)
        if cell_id == 0:
            self._correction_status("No cell selected (left-click first)."); return
        data = np.asarray(layer.data)
        frames = self._frames_with_cell(cell_id)
        if not frames:
            self._correction_status(f"Cell {cell_id} not present in tracked labels."); return
        for t in frames:
            correction = self._validated_correction_for_frame(cell_id, t, data)
            if correction is not None:
                add_correction(self._pos_dir, correction)
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        self._correction_status(
            f"Validated track {cell_id} across {len(frames)} frame(s)."
        )

    def _on_anchor_here(self) -> None:
        target = self._selected_correction_target()
        if target is None or self._pos_dir is None:
            return
        cell_id, t, y, x = target
        corrections = read_corrections(self._pos_dir)
        remaining = [
            correction
            for correction in corrections
            if not (
                int(correction.cell_id) == int(cell_id)
                and int(correction.t) == int(t)
                and correction.kind == "anchor"
            )
        ]
        if len(remaining) != len(corrections):
            write_corrections(self._pos_dir, remaining)
            self._refresh_validated_overlay()
            self._correction_status(f"Unanchored cell {cell_id} at t={t}.")
            return
        layer = self._correction_tracked_layer()
        if layer is None:
            add_correction(
                self._pos_dir,
                Correction(cell_id=cell_id, t=t, kind="anchor", y=y, x=x),
            )
            self._refresh_validated_overlay()
            self._correction_status(f"Anchored cell {cell_id} at t={t}.")
            return
        filled = add_anchor(
            self._pos_dir,
            cell_id=cell_id,
            t=t,
            y=y,
            x=x,
            tracked_labels=np.asarray(layer.data),
        )
        self._refresh_validated_overlay()
        suffix = f" (gap-filled {filled} frame(s))" if filled else ""
        self._correction_status(f"Anchored cell {cell_id} at t={t}.{suffix}")

    def _on_extend_backward(self) -> None:
        self._on_extend(direction="backward")

    def _on_extend_forward(self) -> None:
        self._on_extend(direction="forward")

    def _on_extend(self, direction: str) -> None:
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._correction_status("Extend: data.db not found - run DB Generation first."); return
        source_id = self.correction_widget._selected_label
        if not source_id:
            self._correction_status("Extend: no cell selected (left-click first)."); return

        t = self._current_t()
        tracked = np.asarray(layer.data)
        T = tracked.shape[0]

        target_frame = t + (1 if direction == "forward" else -1)
        if direction == "forward" and t >= T - 1:
            self._correction_status("Already at last frame."); return
        if direction == "backward" and t <= 0:
            self._correction_status("Already at first frame."); return
        if not np.any(tracked[t] == source_id):
            self._correction_status(f"Cell {source_id} not present at t={t}."); return

        read_validated_tracks_fn = self._dependency("read_validated_tracks")
        read_corrections_fn = self._dependency("read_corrections")
        validated_tracks = (
            read_validated_tracks_fn(self._pos_dir) if self._pos_dir is not None else {}
        )
        result = self._dependency("extend_track_from_db")(
            source_id=source_id, source_frame=t, direction=direction,
            tracked_labels=tracked, db_path=db_path,
            d_max=float(self.extend_max_dist_spin.value()),
            area_weight=float(self.extend_area_weight_spin.value()),
            iou_weight=float(self.extend_iou_weight_spin.value()),
            distance_weight=float(self.extend_distance_weight_spin.value()),
            overlap_penalty=float(self.extend_overlap_penalty_spin.value()),
            greedy_overwrite=self.extend_greedy_overwrite_check.isChecked(),
            validated_tracks=validated_tracks,
        )

        if result is None:
            self._correction_status(
                f"No candidate within {self.extend_max_dist_spin.value():g}px at t={target_frame}."
            ); return

        assignments = result.assignments or ()
        if not assignments:
            assignments = (SimpleNamespace(cell_id=source_id, mask_2d=result.mask_2d),)

        frame = layer.data[result.target_frame]

        protected_ids_at_target: set[int] = set()
        for cell_id, frames in validated_tracks.items():
            if result.target_frame in frames:
                protected_ids_at_target.add(cell_id)
        if self._pos_dir is not None:
            for c in read_corrections_fn(self._pos_dir):
                if c.kind == "anchor" and int(c.t) == int(result.target_frame):
                    protected_ids_at_target.add(int(c.cell_id))
        protected_mask = np.zeros_like(frame, dtype=bool)
        for pid in protected_ids_at_target:
            protected_mask |= (frame == pid)

        assignments = tuple(
            a for a in assignments
            if int(a.cell_id) not in protected_ids_at_target
        )
        if not assignments:
            self._correction_status(
                f"Extend skipped: cell {source_id} is protected at t={result.target_frame}."
            )
            return

        changed_ids = {int(a.cell_id) for a in assignments}
        for cid in changed_ids:
            frame[frame == cid] = 0
        if self.extend_greedy_overwrite_check.isChecked():
            for a in assignments:
                frame[a.mask_2d & ~protected_mask] = int(a.cell_id)
        else:
            for a in assignments:
                frame[a.mask_2d & (frame == 0)] = int(a.cell_id)
        layer.refresh()

        step = list(self.viewer.dims.current_step)
        step[0] = result.target_frame
        self.viewer.dims.current_step = tuple(step)

        self._correction_status(
            f"Extended cell {source_id} -> t={result.target_frame} "
            f"(dist={result.centroid_distance:.1f}px, area={result.area_ratio:.2f}, "
            f"iou={result.centroid_corrected_iou:.2f}, overlap={result.existing_overlap:.2f})"
        )

    def _on_swap_step(self, direction: str) -> None:
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._correction_status("data.db not found - run DB Generation first."); return
        source_id = self.correction_widget._selected_label
        if not source_id:
            self._correction_status("Swap: no cell selected (left-click first)."); return

        t = self._current_t()
        tracked = np.asarray(layer.data)
        source_mask = tracked[t] == source_id
        if not source_mask.any():
            self._correction_status(f"Cell {source_id} not present at t={t}."); return

        validated_tracks = (
            read_validated_tracks(self._pos_dir) if self._pos_dir is not None else {}
        )
        if source_id in validated_tracks:
            self._correction_status("Cannot swap a validated cell."); return

        if self._swap_cursor is not None and (
            self._swap_cursor.source_id != source_id
            or self._swap_cursor.frame != t
        ):
            self._swap_cursor = None

        if self._swap_cursor is None:
            from skimage.measure import regionprops as _regionprops
            props = _regionprops(source_mask.astype(np.uint8))
            if not props:
                self._correction_status("Cannot compute centroid for source cell."); return
            src_cy, src_cx = props[0].centroid
            src_area = int(props[0].area)
            source_centroid = (float(src_cy), float(src_cx))

            protected_ids: set[int] = set()
            for cell_id, frames in validated_tracks.items():
                if t in frames and cell_id != source_id:
                    protected_ids.add(cell_id)
            if self._pos_dir is not None:
                for c in read_corrections(self._pos_dir):
                    if c.kind == "anchor" and int(c.t) == t and int(c.cell_id) != source_id:
                        protected_ids.add(int(c.cell_id))
            protected_mask = (
                np.isin(tracked[t], list(protected_ids))
                if protected_ids
                else np.zeros(tracked.shape[1:], dtype=bool)
            )

            radius_px = float(self.swap_radius_spin.value())
            candidates = list_swap_candidates(
                db_path=db_path,
                frame=t,
                source_centroid=source_centroid,
                radius_px=radius_px,
                frame_shape=tuple(tracked.shape[1:]),
                protected_mask=protected_mask,
            )
            if not candidates:
                self._correction_status(
                    f"No swap candidates within {radius_px:g}px."
                ); return

            self._swap_cursor = _SwapCursor(
                source_id=source_id,
                frame=t,
                source_centroid=source_centroid,
                source_area=src_area,
                candidates=tuple(candidates),
                displayed_area=src_area,
                cursor=None,
                baseline_frame=tracked[t].copy(),
            )

        cursor = self._swap_cursor
        if direction == "smaller":
            idx = _step_smaller(cursor.candidates, cursor.displayed_area)
            no_move_msg = "No smaller candidate."
        else:
            idx = _step_larger(cursor.candidates, cursor.displayed_area)
            no_move_msg = "No larger candidate."

        if idx is None:
            self._correction_status(no_move_msg); return

        candidate = cursor.candidates[idx]
        validated_tracks_full = (
            read_validated_tracks(self._pos_dir) if self._pos_dir is not None else {}
        )
        self._apply_swap(layer, t, source_id, candidate, validated_tracks_full)
        cursor.cursor = idx
        cursor.displayed_area = candidate.area
        self._correction_status(
            f"Swapped cell {source_id} -> candidate {idx + 1}/{len(cursor.candidates)}"
            f" (area={candidate.area} px)"
        )

    def _apply_swap(self, layer, t: int, source_id: int, candidate: _SwapCandidate, validated_tracks: dict) -> None:
        frame = layer.data[t]
        before = frame.copy()

        cursor = self._swap_cursor
        if (
            cursor is not None
            and cursor.baseline_frame is not None
            and cursor.source_id == source_id
            and cursor.frame == t
            and cursor.baseline_frame.shape == frame.shape
        ):
            frame[:] = cursor.baseline_frame

        protected_ids: set[int] = set()
        for cell_id, frames in validated_tracks.items():
            if t in frames and cell_id != source_id:
                protected_ids.add(cell_id)
        if self._pos_dir is not None:
            for c in read_corrections(self._pos_dir):
                if c.kind == "anchor" and int(c.t) == t and int(c.cell_id) != source_id:
                    protected_ids.add(int(c.cell_id))
        protected_mask = (
            np.isin(frame, list(protected_ids))
            if protected_ids
            else np.zeros_like(frame, dtype=bool)
        )

        frame[frame == source_id] = 0
        paintable = candidate.mask_2d & ~protected_mask
        frame[paintable] = source_id

        self.correction_widget._record_history(layer, t, before)
        layer.refresh()

    def _on_retrack_forward(self) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._correction_status("Need >= 2 frames to retrack."); return
        t0 = int(self.viewer.dims.current_step[0])
        if t0 >= layer.data.shape[0] - 1:
            self._correction_status("Already at last frame."); return

        T = layer.data.shape[0]
        stack = layer.data.copy()
        fully_validated = read_validated_frames(self._pos_dir)
        reserved_ids = set(read_validated_tracks(self._pos_dir))
        n_retracked = n_skipped = 0
        for t in range(t0 + 1, T):
            if t in fully_validated:
                n_skipped += 1; continue
            locked = read_validated_cells_at_frame(self._pos_dir, t)
            stack[t] = retrack_frame_constrained(
                stack[t - 1], stack[t], locked,
                max_dist_px=float(self.retrack_max_dist_spin.value()),
                reserved_ids=reserved_ids,
            )
            n_retracked += 1
        layer.data = stack
        self._correction_status(
            f"Retracked forward from t={t0 + 1}: {n_retracked} updated, "
            f"{n_skipped} validated skipped. Unsaved."
        )

    def _on_retrack_backward(self) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        if layer.data.ndim != 3 or layer.data.shape[0] < 2:
            self._correction_status("Need >= 2 frames to retrack."); return
        t0 = int(self.viewer.dims.current_step[0])
        if t0 <= 0:
            self._correction_status("Already at first frame."); return

        stack = layer.data.copy()
        fully_validated = read_validated_frames(self._pos_dir)
        reserved_ids = set(read_validated_tracks(self._pos_dir))
        n_retracked = n_skipped = 0
        for t in range(t0 - 1, -1, -1):
            if t in fully_validated:
                n_skipped += 1; continue
            locked = read_validated_cells_at_frame(self._pos_dir, t)
            stack[t] = retrack_frame_constrained(
                stack[t + 1], stack[t], locked,
                max_dist_px=float(self.retrack_max_dist_spin.value()),
                reserved_ids=reserved_ids,
            )
            n_retracked += 1
        layer.data = stack
        self._correction_status(
            f"Retracked backward from t={t0 - 1}: {n_retracked} updated, "
            f"{n_skipped} validated skipped. Unsaved."
        )

    def _on_remove_unvalidated_labels(self) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer loaded."); return
        data = np.asarray(layer.data)
        if data.ndim < 2:
            self._correction_status("Tracked layer has no image data."); return

        validated_tracks = read_validated_tracks(self._pos_dir)
        frame_count = int(data.shape[0]) if data.ndim >= 3 else 1
        changed_pixels = changed_frames = 0
        for t in range(frame_count):
            frame = self._frame_view_2d(data, t) if data.ndim >= 3 else data
            if frame is None:
                self._correction_status("Tracked layer must be a time-first stack."); return
            validated_ids = {
                cid for cid, frames in validated_tracks.items() if t in frames
            }
            remove_mask = frame != 0
            if validated_ids:
                remove_mask &= ~np.isin(frame, list(validated_ids))
            n_remove = int(np.count_nonzero(remove_mask))
            if not n_remove: continue
            frame[remove_mask] = 0
            changed_pixels += n_remove
            changed_frames += 1

        if not changed_pixels:
            self._correction_status("No unvalidated labels found."); return
        layer.refresh()
        if self.correction_widget._selected_label:
            ct = self._current_t()
            if self.correction_widget._selected_label not in self._current_cell_ids(ct):
                self.correction_widget.select_label(ct, 0)
        self._refresh_validated_overlay()
        self._refresh_validation_counter()
        self._correction_status(
            f"Removed unvalidated labels in {changed_frames} frame(s), "
            f"{changed_pixels} px changed. Unsaved."
        )

    def _remove_unvalidated_from_layer(self, layer) -> tuple[int, int]:
        if self._pos_dir is None:
            return 0, 0
        data = np.asarray(layer.data)
        validated_tracks = read_validated_tracks(self._pos_dir)
        frame_count = int(data.shape[0]) if data.ndim >= 3 else 1
        changed_pixels = changed_frames = 0
        for t in range(frame_count):
            frame = self._frame_view_2d(data, t) if data.ndim >= 3 else data
            if frame is None:
                raise ValueError("Tracked layer must be a time-first stack.")
            validated_ids = {
                cid for cid, frames in validated_tracks.items() if t in frames
            }
            remove_mask = frame != 0
            if validated_ids:
                remove_mask &= ~np.isin(frame, list(validated_ids))
            n_remove = int(np.count_nonzero(remove_mask))
            if not n_remove:
                continue
            frame[remove_mask] = 0
            changed_pixels += n_remove
            changed_frames += 1
        if changed_pixels:
            layer.refresh()
            if self.correction_widget._selected_label:
                ct = self._current_t()
                if self.correction_widget._selected_label not in self._current_cell_ids(ct):
                    self.correction_widget.select_label(ct, 0)
            self._refresh_validated_overlay()
            self._refresh_validation_counter()
        return changed_frames, changed_pixels

    def _on_commit(self) -> None:
        tracked_path = self._tracked_path()
        if tracked_path is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No tracked layer to commit."); return
        if layer.data.ndim != 3:
            self._correction_status("Tracked layer is not a 3D stack."); return
        try:
            n_cells = self._commit_reassign_ids(layer)
            changed_frames, changed_pixels = self._remove_unvalidated_from_layer(layer)
        except Exception as exc:
            self._on_correction_worker_error(exc)
            return
        for t in range(int(layer.data.shape[0])):
            write_tracked_frame(tracked_path, t, np.asarray(layer.data[t]))
        self._correction_status(
            f"Committed {n_cells} cell(s); removed {changed_pixels} px in "
            f"{changed_frames} frame(s); saved to {tracked_path.name}."
        )

    def _on_correction_worker_error(self, exc: Exception) -> None:
        self._correction_status(f"Error: {exc}")
        logger.exception("Correction worker error", exc_info=exc)

    def _install_correction_shortcuts(self) -> None:
        specs = [
            ("A", lambda: self._on_extend(direction="backward")),
            ("D", lambda: self._on_extend(direction="forward")),
            ("Q", self._on_retrack_backward),
            ("E", self._on_retrack_forward),
            ("B", self._on_anchor_here),
            ("V", lambda: self._kb_toggle_cell_validation(None)),
            ("S", self._on_save_tracked),
            ("Z", lambda: self._on_swap_step(direction="smaller")),
            ("C", lambda: self._on_swap_step(direction="larger")),
        ]
        self._correction_shortcuts: list[QShortcut] = []
        for key, slot in specs:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.setEnabled(False)
            sc.activated.connect(slot)
            self._correction_shortcuts.append(sc)

    def _on_correction_active_button_toggled(self, active: bool) -> None:
        if active:
            self._capture_correction_view_state()
            for layer in list(self.viewer.layers):
                layer.visible = False

            if not self._load_correction_layers_from_disk():
                self._restore_correction_view_state()
                old = self.active_btn.blockSignals(True)
                try:
                    self.active_btn.setChecked(False)
                finally:
                    self.active_btn.blockSignals(old)
                self.correction_widget.deactivate()
                self._correction_active_content_visible = False
                self._sync_correction_panel_visibility()
                self._refresh_refinement_widget()
                return
            layer = self.viewer.layers[_CORRECTION_TRACKED_LAYER]
            layer.visible = True
            self.viewer.layers.selection.active = layer
            self.correction_widget.activate_layer(layer)
            self._set_checked_without_signal(self.params_btn, False)
            self._set_checked_without_signal(self.shortcuts_btn, False)
            self.extend_retrack_params_section._toggle.setChecked(False)
            self.shortcuts_section._toggle.setChecked(False)
            self._correction_active_content_visible = True
            self._sync_correction_panel_visibility()
            self._refresh_refinement_widget()
            return

        self._correction_active_content_visible = False
        self.correction_widget.deactivate()
        for sc in getattr(self, "_correction_shortcuts", []):
            sc.setEnabled(False)
        # Refresh the main Tracked layer from disk so a subsequent re-solve
        # picks up any corrections the user saved during this session.
        self._refresh_tracked_layer_from_disk()
        self._remove_correction_owned_layers()
        self._restore_correction_view_state()
        self._set_checked_without_signal(self.params_btn, False)
        self._set_checked_without_signal(self.shortcuts_btn, False)
        self.extend_retrack_params_section._toggle.setChecked(False)
        self.shortcuts_section._toggle.setChecked(False)
        self._sync_correction_panel_visibility()
        self._refresh_refinement_widget()

    def _refresh_refinement_widget(self) -> None:
        self._refresh_refinement_callback()

    def _on_correction_mode_toggled(self, active: bool) -> None:
        if not active:
            self._swap_cursor = None
        for sc in self._correction_shortcuts:
            sc.setEnabled(active)

    def _kb_toggle_cell_validation(self, _viewer) -> None:
        if self._pos_dir is None:
            return
        sel = self.correction_widget._selected_label
        if not sel:
            self._correction_status(
                "Validation toggle: no cell selected (left-click first)."
            ); return
        t = self._current_t()
        if sel not in self._current_cell_ids(t):
            self._correction_status(f"Cell {sel} not present at t={t}."); return
        frames = self._frames_with_cell(sel)
        if not frames:
            return
        if is_track_validated(self._pos_dir, sel):
            invalidate_track(self._pos_dir, sel)
            self._correction_status(
                f"Cell {sel} invalidated across {len(frames)} frame(s)."
            )
        else:
            layer = self._correction_tracked_layer()
            if layer is None:
                return
            data = np.asarray(layer.data)
            for frame in frames:
                correction = self._validated_correction_for_frame(sel, frame, data)
                if correction is not None:
                    add_correction(self._pos_dir, correction)
            self._correction_status(
                f"Cell {sel} validated across {len(frames)} frame(s)."
            )
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def on_dims_step_changed(self) -> None:
        self._swap_cursor = None
        self._refresh_validated_overlay()
        self._refresh_validation_counter()

    def _refresh_validated_overlay(self) -> None:
        self._validated_overlay.refresh_overlay(self._frame_view_2d)

    def _add_validated_overlay(self, data: np.ndarray) -> None:
        self._validated_overlay.add_overlay(data)

    def _place_validated_overlay_below_spotlight(self) -> None:
        self._validated_overlay.place_below_spotlight()

    def _refresh_validation_counter(self) -> None:
        self._validated_overlay.refresh_counter(self.validation_counter_lbl)

    def _on_cells_edited(self, t: int, changed_ids: set[int]) -> None:
        self._validated_overlay.on_cells_edited(
            t,
            changed_ids,
            frame_view_2d=self._frame_view_2d,
            counter_label=self.validation_counter_lbl,
        )

    def _frames_with_cell(self, cell_id: int) -> list[int]:
        return self._validated_overlay.frames_with_cell(cell_id)
