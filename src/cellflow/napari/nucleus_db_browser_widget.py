"""Ultrack database browser section for the nucleus workflow widget."""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._widget_helpers import (
    add_slider_step_buttons as _add_slider_step_buttons,
    tool_btn as _tool_btn,
)
from cellflow.napari.ui_style import (
    stage_header_action_button as _stage_header_action_button,
    stage_header_label as _stage_header_label,
)
from cellflow.napari.widgets import CollapsibleSection
from cellflow.tracking_ultrack.db_query import (
    annotation_name as _ultrack_db_annotation_name,
    link_annotation_counts as _query_ultrack_db_link_annotation_counts,
    node_annotation_metadata as _ultrack_db_node_annotation_metadata,
    node_mask_and_bbox as _node_mask_and_bbox,
    node_preview_metadata as _ultrack_db_node_preview_metadata,
    paint_nodes as _paint_ultrack_db_nodes,
    query_connected_nodes as _query_ultrack_db_connected_nodes,
    query_frame_range as _query_db_frame_range,
    query_middle_frame as _query_ultrack_db_middle_frame,
    query_union_color_classes as _query_union_color_classes,
    query_union_sizes as _query_union_sizes,
    render_union_partition as _render_union_partition,
    summary_text as _ultrack_db_summary_text,
)

logger = logging.getLogger(__name__)

_DATABASE_PREFIX = "[Database]"
_ULTRACK_DB_PREVIEW_LAYER = f"{_DATABASE_PREFIX} Ultrack DB Preview"
_ULTRACK_DB_SELECTION_LAYER = f"{_DATABASE_PREFIX} Ultrack DB Selection"
_ULTRACK_DB_ANNOTATION_LAYER = f"{_DATABASE_PREFIX} Ultrack DB Annotations"
_ULTRACK_DB_NUC_LAYER = f"{_DATABASE_PREFIX} Cellpose nucleus prob"
_ULTRACK_DB_ANNOTATION_COLORS = {
    0: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    1: np.array([0.0, 0.75, 0.0, 0.75], dtype=np.float32),
    2: np.array([1.0, 0.0, 0.0, 0.75], dtype=np.float32),
    3: np.array([0.5, 0.5, 0.5, 0.45], dtype=np.float32),
}


class NucleusUltrackDbBrowserWidget(QWidget):
    """Qt controls for browsing a generated Ultrack database."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.header = QWidget(parent)
        header_lay = QHBoxLayout(self.header)
        header_lay.setContentsMargins(0, 0, 0, 0)
        header_lay.setSpacing(4)
        self.header_lbl = QLabel("Database Browser")
        _stage_header_label(self.header_lbl, "nucleus")
        self.active_btn = _tool_btn(
            "⏻",
            "Activate database browser.",
            checkable=True,
        )
        self.active_btn.setChecked(False)
        _stage_header_action_button(self.active_btn, "nucleus")
        header_lay.addWidget(self.header_lbl)
        header_lay.addWidget(self.active_btn)
        header_lay.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.info_lbl = QLabel("—")
        self.info_lbl.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding,
        )
        lay.addWidget(self.info_lbl)

        # Vertical axis (primary): atom-union size — how many atoms are merged
        # into each candidate. 0 = individual atoms, higher = coarser merges.
        self.hierarchy_slider = QSlider(Qt.Horizontal)
        self.hierarchy_slider.setRange(0, 0)
        self.hierarchy_slider.setValue(0)
        self.hierarchy_slider.setToolTip(
            "Union size: number of atoms merged per candidate "
            "(lowest = individual atoms, higher = coarser merges)"
        )
        self.hierarchy_slider.setEnabled(False)
        self.height_lbl = QLabel("—")
        self.height_lbl.setFixedWidth(64)
        self.slider_row = QWidget()
        slider_lay = QHBoxLayout(self.slider_row)
        slider_lay.setContentsMargins(0, 0, 0, 0)
        slider_lay.setSpacing(2)
        _add_slider_step_buttons(slider_lay, self.hierarchy_slider)
        slider_lay.addWidget(self.height_lbl)
        lay.addWidget(self.slider_row)

        # Horizontal axis: which merge — at the chosen union size, scan the
        # non-overlapping merge groups (color classes) that together show every
        # candidate at least once.
        self.source_slider = QSlider(Qt.Horizontal)
        self.source_slider.setRange(0, 0)
        self.source_slider.setValue(0)
        self.source_slider.setToolTip(
            "Merge view: scan the different non-overlapping merge groups "
            "at the current union size"
        )
        self.source_slider.setEnabled(False)
        self.source_lbl = QLabel("—")
        self.source_lbl.setFixedWidth(64)
        self.source_slider_row = QWidget()
        source_slider_lay = QHBoxLayout(self.source_slider_row)
        source_slider_lay.setContentsMargins(0, 0, 0, 0)
        source_slider_lay.setSpacing(2)
        _add_slider_step_buttons(source_slider_lay, self.source_slider)
        source_slider_lay.addWidget(self.source_lbl)
        lay.addWidget(self.source_slider_row)

        self.prob_alpha_check = QCheckBox("Node prob transparency")
        self.prob_alpha_check.setToolTip("Modulate label opacity by node probability")
        self.prob_alpha_check.setEnabled(False)
        self.connected_focus_check = QCheckBox("Connected focus")
        self.connected_focus_check.setToolTip(
            "Focus the DB preview on a selected node and its temporal neighbors"
        )
        self.connected_focus_check.setEnabled(False)
        self.annotation_check = QCheckBox("Show DB annotations")
        self.annotation_check.setToolTip("Overlay DB REAL and FAKE annotations")
        self.annotation_check.setEnabled(False)
        for cb in (
            self.prob_alpha_check,
            self.connected_focus_check,
            self.annotation_check,
        ):
            lay.addWidget(cb)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setVisible(False)
        lay.addWidget(self.status_lbl)

        self.section: CollapsibleSection | None = None


class NucleusUltrackDbBrowserMixin:
    """Behavior for the nucleus Ultrack database browser section."""

    def _init_ultrack_db_browser_state(self) -> None:
        self._ultrack_db_preview_cache: dict = {}
        self._ultrack_db_summary_cache: dict[tuple, str] = {}
        self._ultrack_db_size_values_cache: dict[tuple, tuple[int, ...]] = {}
        self._ultrack_db_color_class_cache: dict[tuple, tuple[tuple[int, ...], ...]] = {}
        self._ultrack_db_frames_cache: dict[tuple, tuple[int, ...]] = {}
        self._ultrack_db_browser_active: bool = False
        self._ultrack_db_frame_initialized: bool = False
        self._ultrack_db_selected_node_id: int | None = None
        self._ultrack_db_selected_frame: int | None = None
        self._ultrack_db_label_to_node_id: dict[int, int] = {}
        self._ultrack_db_node_id_to_label: dict[int, int] = {}
        self._ultrack_db_node_annotations: dict[int, str] = {}
        self._ultrack_db_label_probabilities: dict[int, float] = {}
        self._ultrack_db_preview_labels: np.ndarray | None = None
        self._ultrack_db_preview_mouse_callback = None
        self._ultrack_db_preview_view_state: dict | None = None
        self._ultrack_db_refresh_timer = QTimer(self)
        self._ultrack_db_refresh_timer.setSingleShot(True)
        self._ultrack_db_refresh_timer.setInterval(150)
        self._ultrack_db_refresh_timer.timeout.connect(self._refresh_ultrack_db_browser)

    def _build_db_browser_section(self, root: QVBoxLayout) -> None:
        self.ultrack_db_browser_widget = NucleusUltrackDbBrowserWidget(self)
        self.ultrack_db_browser_widget.section = CollapsibleSection(
            "Database Browser",
            self.ultrack_db_browser_widget,
            expanded=False,
        )
        self.ultrack_db_browser_widget.section._toggle.setVisible(False)
        self.ultrack_db_browser_widget.section._toggle.setEnabled(False)
        self._alias_ultrack_db_browser_controls()

    def _alias_ultrack_db_browser_controls(self) -> None:
        browser = self.ultrack_db_browser_widget
        self.ultrack_db_browser_header = browser.header
        self.ultrack_db_browser_header_lbl = browser.header_lbl
        self.ultrack_db_browser_section = browser.section
        self.ultrack_db_info_lbl = browser.info_lbl
        self.ultrack_db_source_slider = browser.source_slider
        self.ultrack_db_source_lbl = browser.source_lbl
        self._ultrack_db_source_slider_row = browser.source_slider_row
        self.ultrack_db_hierarchy_slider = browser.hierarchy_slider
        self.ultrack_db_height_lbl = browser.height_lbl
        self._ultrack_db_slider_row = browser.slider_row
        self.ultrack_db_active_btn = browser.active_btn
        self.ultrack_db_prob_alpha_check = browser.prob_alpha_check
        self.ultrack_db_connected_focus_check = browser.connected_focus_check
        self.ultrack_db_annotation_check = browser.annotation_check
        self.ultrack_db_section_status_lbl = browser.status_lbl

    def _set_ultrack_db_status(self, msg: str) -> None:
        self.ultrack_db_section_status_lbl.setText(msg)
        self.ultrack_db_section_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _on_ultrack_db_browser_param_changed(self, *_args) -> None:
        self._ultrack_db_preview_cache.clear()

    def _schedule_ultrack_db_browser_refresh(self) -> None:
        self._ultrack_db_refresh_timer.start()

    def _on_ultrack_db_source_changed(self, value: int) -> None:
        """Horizontal axis: pick which merge group (color class) to display."""
        if not self._ultrack_db_browser_active:
            return
        max_merge = self.ultrack_db_source_slider.maximum()
        self._set_ultrack_db_merge_label(value, max_merge + 1)
        self._schedule_ultrack_db_browser_refresh()

    def _on_ultrack_db_slider_changed(self, value: int) -> None:
        """Vertical axis: pick the union size (number of atoms merged)."""
        if not self._ultrack_db_browser_active:
            return
        db_path = self._ultrack_db_path()
        sizes: tuple[int, ...] = ()
        if db_path is not None and db_path.exists():
            try:
                mtime_ns = db_path.stat().st_mtime_ns
                sizes = self._query_union_sizes(db_path, mtime_ns, self._current_t())
            except Exception:
                sizes = ()
        if sizes:
            index = min(max(int(value), 0), len(sizes) - 1)
            self._set_ultrack_db_size_label(index, sizes[index], len(sizes))
        else:
            self.ultrack_db_height_lbl.setText("—")
        self._schedule_ultrack_db_browser_refresh()

    def _on_ultrack_db_activate(self, checked: bool) -> None:
        if checked:
            db_path = self._ultrack_db_path()
            if db_path is None or not db_path.exists():
                self._ultrack_db_browser_active = False
                old = self.ultrack_db_active_btn.blockSignals(True)
                try:
                    self.ultrack_db_active_btn.setChecked(False)
                finally:
                    self.ultrack_db_active_btn.blockSignals(old)
                self._set_ultrack_db_controls_enabled(False)
                self.ultrack_db_browser_section.collapse()
                self._set_ultrack_db_status(
                    "data.db not found — run DB Generation first."
                )
                return

        self._ultrack_db_browser_active = checked
        self._set_ultrack_db_controls_enabled(checked)
        if checked:
            self._capture_ultrack_db_preview_view_state()
            self._hide_non_ultrack_db_preview_layers()
            self.ultrack_db_browser_section.expand()
            self._load_db_browser_nucleus_image()
            self._ultrack_db_frame_initialized = False
            self._refresh_ultrack_db_browser()
        else:
            self._remove_ultrack_db_browser_layers()
            self._restore_ultrack_db_preview_view_state()
            self.ultrack_db_browser_section.collapse()

    def _set_ultrack_db_controls_enabled(self, enabled: bool) -> None:
        self.ultrack_db_active_btn.setToolTip(
            "Deactivate database browser." if enabled else "Activate database browser."
        )
        self.ultrack_db_source_slider.setEnabled(enabled)
        self.ultrack_db_hierarchy_slider.setEnabled(enabled)
        self.ultrack_db_prob_alpha_check.setEnabled(enabled)
        self.ultrack_db_connected_focus_check.setEnabled(enabled)
        self.ultrack_db_annotation_check.setEnabled(enabled)

    def _remove_ultrack_db_browser_layers(self) -> None:
        self._remove_ultrack_db_preview_selector()
        for layer in list(self.viewer.layers):
            if self._is_ultrack_db_preview_layer(layer.name):
                self.viewer.layers.remove(layer)
        self._remove_db_browser_nucleus_image()
        self.ultrack_db_info_lbl.setText("—")
        self._set_ultrack_db_status("")

    @staticmethod
    def _is_ultrack_db_preview_layer(name: str) -> bool:
        return name.startswith(f"{_DATABASE_PREFIX} ")

    def _capture_ultrack_db_preview_view_state(self) -> None:
        selected = [layer.name for layer in self.viewer.layers.selection]
        active = self.viewer.layers.selection.active
        self._ultrack_db_preview_view_state = {
            "visibility": {
                layer.name: bool(layer.visible) for layer in self.viewer.layers
            },
            "active": active.name if active is not None else None,
            "selected": selected,
        }

    def _hide_non_ultrack_db_preview_layers(self) -> None:
        for layer in self.viewer.layers:
            if not self._is_ultrack_db_preview_layer(layer.name):
                layer.visible = False

    def _restore_ultrack_db_preview_view_state(self) -> None:
        state = self._ultrack_db_preview_view_state or {}
        for name, visible in state.get("visibility", {}).items():
            if name in self.viewer.layers:
                self.viewer.layers[name].visible = bool(visible)
        self.viewer.layers.selection.clear()
        for name in state.get("selected", ()):
            if name in self.viewer.layers:
                self.viewer.layers.selection.add(self.viewer.layers[name])
        active_name = state.get("active")
        if active_name in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[active_name]
        self._ultrack_db_preview_view_state = None

    def _load_db_browser_nucleus_image(self) -> None:
        if _ULTRACK_DB_NUC_LAYER in self.viewer.layers:
            return
        nuc_path = self._nucleus_foreground_path()
        if nuc_path is None or not nuc_path.exists():
            return
        import tifffile
        data = np.asarray(tifffile.imread(str(nuc_path)), dtype=np.float32)
        limits = np.percentile(data, [0.05, 99.5]) if data.size > 0 else None
        kwargs = {
            "name": _ULTRACK_DB_NUC_LAYER,
            "colormap": "bop orange",
            "blending": "minimum",
        }
        if limits is not None and limits[1] > limits[0]:
            kwargs["contrast_limits"] = [float(limits[0]), float(limits[1])]
        self.viewer.add_image(data, **kwargs)

    def _remove_db_browser_nucleus_image(self) -> None:
        if _ULTRACK_DB_NUC_LAYER in self.viewer.layers:
            self.viewer.layers.remove(_ULTRACK_DB_NUC_LAYER)

    def _ultrack_db_middle_frame(self, db_path: Path) -> int | None:
        return _query_ultrack_db_middle_frame(db_path)

    def _viewer_has_time_axis(self) -> bool:
        for layer in self.viewer.layers:
            data = getattr(layer, "data", None)
            if isinstance(data, np.ndarray) and data.ndim >= 3:
                return True
        return False

    def _query_db_frames(self, db_path: Path, mtime_ns: int) -> tuple[int, ...]:
        key = (str(db_path.resolve()), mtime_ns, "frames")
        cached = self._ultrack_db_frames_cache.get(key)
        if cached is not None:
            return cached
        result = _query_db_frame_range(db_path)
        self._ultrack_db_frames_cache[key] = result
        return result

    def _resolve_ultrack_db_partition(self, db_path: Path, mtime_ns: int, frame: int):
        """Configure both sliders for ``frame`` and return the normalized preview
        tuple for the current (union size, merge group), or ``None`` if the frame
        has no candidates.

        Vertical slider → union size (atoms merged per candidate); horizontal slider
        → which non-overlapping merge group (color class) of that size to paint.
        """
        sizes = self._configure_ultrack_db_size_slider(db_path, mtime_ns, frame)
        if not sizes:
            return None
        size_index = min(int(self.ultrack_db_hierarchy_slider.value()), len(sizes) - 1)
        union_size = sizes[size_index]
        classes = self._configure_ultrack_db_merge_slider(
            db_path, mtime_ns, frame, union_size
        )
        if not classes:
            return None
        color_index = min(int(self.ultrack_db_source_slider.value()), len(classes) - 1)
        color_node_ids = classes[color_index]
        key = self._ultrack_db_preview_cache_key(
            db_path, mtime_ns, frame, union_size, color_index,
        )
        cached = self._ultrack_db_preview_cache.get(key)
        if cached is None:
            cached = self._render_union_partition(db_path, frame, color_node_ids)
            self._ultrack_db_preview_cache[key] = cached
        return self._normalize_ultrack_db_preview(cached)

    def _load_full_db_stack(self, db_path: Path) -> None:
        """Create a 3D DB preview stack and render only the initial frame."""
        try:
            mtime_ns = db_path.stat().st_mtime_ns
            frames = self._query_db_frames(db_path, mtime_ns)
            if not frames:
                self._set_ultrack_db_status("No frames in database.")
                return

            mid_frame = frames[len(frames) // 2]
            self.ultrack_db_info_lbl.setText(
                self._cached_ultrack_db_summary_text(db_path, mtime_ns, mid_frame)
            )
            resolved = self._resolve_ultrack_db_partition(db_path, mtime_ns, mid_frame)
            if resolved is None:
                self._set_ultrack_db_status(f"No candidates for frame {mid_frame}.")
                return
            labels, _status, prob_dict, l2n, n2l, annots = resolved
            self._ultrack_db_label_to_node_id = l2n
            self._ultrack_db_node_id_to_label = n2l
            self._ultrack_db_node_annotations = annots
            self._ultrack_db_label_probabilities = prob_dict

            n_frames = frames[-1] + 1
            base_h, base_w = self._viewer_plane_shape()
            max_h = max(base_h, labels.shape[0], 1)
            max_w = max(base_w, labels.shape[1], 1)
            stack = np.zeros((n_frames, max_h, max_w), dtype=np.uint32)
            h, w = labels.shape
            stack[mid_frame, :h, :w] = labels

            self._update_labels_layer(_ULTRACK_DB_PREVIEW_LAYER, stack)
            self._ultrack_db_preview_labels = stack[mid_frame] if mid_frame < n_frames else stack[0]
            self._install_ultrack_db_preview_selector()
            self._restore_ultrack_db_preview_active()
            self._set_viewer_frame(mid_frame)
            self._set_ultrack_db_status(
                f"Loaded frame {mid_frame}/{frames[-1]} from database; "
                "other frames render when visited."
            )
        except Exception as e:
            self._set_ultrack_db_status(f"DB read error: {e}")
            logger.warning("DB browser stack load error: %s", e)

    def _update_ultrack_db_stack_frame(self, frame: int, labels: np.ndarray) -> bool:
        """Update a single frame slice in a 3D preview stack. Returns True if stack mode is active."""
        if _ULTRACK_DB_PREVIEW_LAYER not in self.viewer.layers:
            return False
        from napari.layers import Labels
        layer = self.viewer.layers[_ULTRACK_DB_PREVIEW_LAYER]
        if not isinstance(layer, Labels) or layer.data.ndim != 3:
            return False
        stack = layer.data
        t = int(frame)
        if t < 0 or t >= stack.shape[0]:
            return False
        if labels.shape[0] > stack.shape[1] or labels.shape[1] > stack.shape[2]:
            expanded = np.zeros(
                (
                    stack.shape[0],
                    max(stack.shape[1], labels.shape[0]),
                    max(stack.shape[2], labels.shape[1]),
                ),
                dtype=stack.dtype,
            )
            expanded[:, : stack.shape[1], : stack.shape[2]] = stack
            stack = expanded
        lh = labels.shape[0]
        lw = labels.shape[1]
        stack[t] = 0
        stack[t, :lh, :lw] = labels
        layer.data = stack
        return True

    def _refresh_ultrack_db_browser(self) -> None:
        if not self._ultrack_db_browser_active:
            return
        self.ultrack_db_info_lbl.setText("—")
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._set_ultrack_db_status("data.db not found — run DB Generation first.")
            return
        frame = self._current_t()
        if not self._ultrack_db_frame_initialized:
            self._ultrack_db_frame_initialized = True
            if not self._viewer_has_time_axis():
                # No movie open — load the whole database stack so napari gets
                # a time dimension and the user can navigate all frames.
                self._load_full_db_stack(db_path)
                return
            if frame == 0:
                mid = self._ultrack_db_middle_frame(db_path)
                if mid is not None and mid > 0:
                    frame = mid
                    self._set_viewer_frame(frame)
        try:
            mtime_ns = db_path.stat().st_mtime_ns
            self.ultrack_db_info_lbl.setText(
                self._cached_ultrack_db_summary_text(db_path, mtime_ns, frame)
            )
            resolved = self._resolve_ultrack_db_partition(db_path, mtime_ns, frame)
            if resolved is None:
                labels = self._empty_ultrack_db_preview()
                self._update_labels_layer(_ULTRACK_DB_PREVIEW_LAYER, labels)
                self._set_ultrack_db_status(f"No candidates for frame {frame}.")
                return
            labels, status, prob_dict, label_to_node_id, node_id_to_label, node_annotations = (
                resolved
            )
            self._ultrack_db_label_to_node_id = label_to_node_id
            self._ultrack_db_node_id_to_label = node_id_to_label
            self._ultrack_db_node_annotations = node_annotations
            self._ultrack_db_label_probabilities = prob_dict
            alpha_dict: dict[int, float] = {}
            if self.ultrack_db_connected_focus_check.isChecked():
                labels, status, alpha_dict = self._render_ultrack_db_connected_focus(
                    db_path, frame, labels, status, prob_dict,
                    label_to_node_id, node_id_to_label,
                )
            self._ultrack_db_preview_labels = labels.astype(np.uint32, copy=False)
            if not self._update_ultrack_db_stack_frame(frame, self._ultrack_db_preview_labels):
                self._update_ultrack_db_preview_layer(
                    self._ultrack_db_preview_labels, prob_dict, alpha_dict,
                )
            self._refresh_ultrack_db_annotation_visualization(
                self._ultrack_db_preview_labels, label_to_node_id, node_annotations,
            )
            status = self._append_ultrack_db_visible_annotation_status(
                status,
                self._ultrack_db_preview_labels,
                label_to_node_id,
                node_annotations,
            )
            self._install_ultrack_db_preview_selector()
            if not self.ultrack_db_connected_focus_check.isChecked():
                status = self._refresh_ultrack_db_selection_highlight(
                    self._ultrack_db_preview_labels, status, node_id_to_label, frame,
                )
            self._restore_ultrack_db_preview_active()
            self._set_ultrack_db_status(status)
        except Exception as e:
            self._set_ultrack_db_status(f"DB read error: {e}")
            logger.warning("DB browser error: %s", e)

    @staticmethod
    def _normalize_ultrack_db_preview(cached):
        if len(cached) == 2:
            labels, status = cached
            return labels, status, {}, {}, {}, {}
        if len(cached) == 3:
            labels, status, prob_dict = cached
            return labels, status, prob_dict, {}, {}, {}
        if len(cached) == 5:
            labels, status, prob_dict, l2n, n2l = cached
            return labels, status, prob_dict, l2n, n2l, {}
        labels, status, prob_dict, l2n, n2l, annots = cached
        return labels, status, prob_dict, l2n, n2l, annots

    def _update_ultrack_db_preview_layer(self, labels, prob_dict, alpha_dict=None):
        if alpha_dict:
            data = self._ultrack_db_alpha_rgba(labels, alpha_dict)
            self._update_image_layer(_ULTRACK_DB_PREVIEW_LAYER, data, rgb=True)
            return
        if self.ultrack_db_prob_alpha_check.isChecked() and prob_dict:
            data = self._ultrack_db_probability_rgba(labels, prob_dict)
            self._update_image_layer(_ULTRACK_DB_PREVIEW_LAYER, data, rgb=True)
            return
        self._update_labels_layer(_ULTRACK_DB_PREVIEW_LAYER, labels)

    def _update_ultrack_db_annotation_layer(self, labels, label_to_node_id, node_annotations):
        overlay = np.zeros_like(labels, dtype=np.uint8)
        for lid, nid in label_to_node_id.items():
            annot = node_annotations.get(int(nid), "UNKNOWN")
            if annot == "REAL":
                overlay[labels == int(lid)] = 1
            elif annot == "FAKE":
                overlay[labels == int(lid)] = 2
            else:
                overlay[labels == int(lid)] = 3
        if not np.any(overlay):
            if _ULTRACK_DB_ANNOTATION_LAYER in self.viewer.layers:
                self.viewer.layers.remove(_ULTRACK_DB_ANNOTATION_LAYER)
            return
        self._update_labels_layer(
            _ULTRACK_DB_ANNOTATION_LAYER,
            overlay,
            colormap=self._ultrack_db_annotation_colormap(),
        )

    def _refresh_ultrack_db_annotation_visualization(
        self, labels, label_to_node_id, node_annotations
    ) -> None:
        if not self.ultrack_db_annotation_check.isChecked():
            if _ULTRACK_DB_ANNOTATION_LAYER in self.viewer.layers:
                self.viewer.layers.remove(_ULTRACK_DB_ANNOTATION_LAYER)
            return
        self._update_ultrack_db_annotation_layer(
            labels, label_to_node_id, node_annotations,
        )

    @classmethod
    def _ultrack_db_visible_annotation_counts(
        cls, labels, label_to_node_id, node_annotations
    ) -> dict[str, int]:
        counts = {"REAL": 0, "FAKE": 0, "UNKNOWN": 0}
        if labels is None:
            return counts
        visible_labels = set(int(v) for v in np.unique(labels) if int(v) != 0)
        for display_label in visible_labels:
            node_id = label_to_node_id.get(display_label)
            if node_id is None:
                counts["UNKNOWN"] += 1
                continue
            annotation = cls._ultrack_db_annotation_name(
                node_annotations.get(int(node_id), "UNKNOWN")
            )
            counts[annotation] += 1
        return counts

    def _append_ultrack_db_visible_annotation_status(
        self, status, labels, label_to_node_id, node_annotations
    ) -> str:
        if not self.ultrack_db_annotation_check.isChecked():
            return status
        counts = self._ultrack_db_visible_annotation_counts(
            labels, label_to_node_id, node_annotations,
        )
        return (
            f"{status} Visible annotations: REAL {counts['REAL']}, "
            f"FAKE {counts['FAKE']}, UNKNOWN {counts['UNKNOWN']}."
        )

    @staticmethod
    def _ultrack_db_annotation_colormap():
        from napari.utils.colormaps import DirectLabelColormap

        return DirectLabelColormap(
            color_dict=defaultdict(
                lambda: _ULTRACK_DB_ANNOTATION_COLORS[3],
                _ULTRACK_DB_ANNOTATION_COLORS,
            ),
        )

    def _update_labels_layer(self, name: str, data: np.ndarray, *, colormap=None) -> None:
        from napari.layers import Labels
        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Labels):
            if colormap is not None:
                self.viewer.layers[name].colormap = colormap
            self.viewer.layers[name].data = data
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        kwargs = {"colormap": colormap} if colormap is not None else {}
        self.viewer.add_labels(data, name=name, **kwargs)

    def _update_image_layer(self, name: str, data: np.ndarray, *, rgb: bool = False) -> None:
        from napari.layers import Image
        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Image):
            self.viewer.layers[name].data = data
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        self.viewer.add_image(data, name=name, rgb=rgb, blending="translucent")

    @staticmethod
    def _ultrack_db_probability_rgba(labels, prob_dict):
        from napari.utils.colormaps import label_colormap
        rgba = np.zeros(labels.shape + (4,), dtype=np.float32)
        if labels.size == 0 or not prob_dict:
            return rgba
        probs = [float(v) for v in prob_dict.values()]
        min_p, max_p = min(probs), max(probs)
        denom = max(max_p - min_p, 1e-9)
        cmap = label_colormap(max(prob_dict.keys()) + 1)
        for lid, prob in prob_dict.items():
            mask = labels == int(lid)
            if not np.any(mask):
                continue
            color = np.asarray(cmap.map(int(lid)), dtype=np.float32)
            alpha = 0.15 + 0.85 * (float(prob) - min_p) / denom
            color[3] = float(np.clip(alpha, 0.15, 1.0))
            rgba[mask] = color
        return rgba

    @staticmethod
    def _ultrack_db_alpha_rgba(labels, alpha_dict):
        from napari.utils.colormaps import label_colormap
        rgba = np.zeros(labels.shape + (4,), dtype=np.float32)
        if labels.size == 0 or not alpha_dict:
            return rgba
        cmap = label_colormap(max(alpha_dict.keys()) + 1)
        for lid, alpha in alpha_dict.items():
            mask = labels == int(lid)
            if not np.any(mask):
                continue
            color = np.asarray(cmap.map(int(lid)), dtype=np.float32)
            color[3] = float(np.clip(alpha, 0.0, 1.0))
            rgba[mask] = color
        return rgba

    def _install_ultrack_db_preview_selector(self) -> None:
        if _ULTRACK_DB_PREVIEW_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[_ULTRACK_DB_PREVIEW_LAYER]
        self._remove_ultrack_db_preview_selector()

        def _on_drag(_layer, event):
            if getattr(event, "type", None) != "mouse_press":
                return
            if getattr(event, "button", None) != 1:
                return
            if getattr(event, "modifiers", set()):
                return
            labels = self._ultrack_db_preview_labels
            if labels is None or labels.size == 0:
                return
            pos = _layer.world_to_data(event.position)
            y, x = int(round(float(pos[-2]))), int(round(float(pos[-1])))
            if y < 0 or x < 0 or y >= labels.shape[-2] or x >= labels.shape[-1]:
                return
            display_label = int(labels[y, x])
            if display_label == 0:
                self._deselect_ultrack_db_node()
                yield
                return
            self._select_ultrack_db_preview_label(display_label, frame=self._current_t())
            yield

        layer.mouse_drag_callbacks.append(_on_drag)
        self._ultrack_db_preview_mouse_callback = _on_drag

    def _remove_ultrack_db_preview_selector(self) -> None:
        cb = self._ultrack_db_preview_mouse_callback
        if cb is None or _ULTRACK_DB_PREVIEW_LAYER not in self.viewer.layers:
            self._ultrack_db_preview_mouse_callback = None
            return
        layer = self.viewer.layers[_ULTRACK_DB_PREVIEW_LAYER]
        try:
            layer.mouse_drag_callbacks.remove(cb)
        except ValueError:
            pass
        self._ultrack_db_preview_mouse_callback = None

    def _select_ultrack_db_preview_label(self, display_label, *, frame=None):
        node_id = self._ultrack_db_label_to_node_id.get(int(display_label))
        if node_id is None:
            self._set_ultrack_db_status(f"No DB node mapped to label {display_label}.")
            self._clear_ultrack_db_highlight()
            return
        selected_frame = self._current_t() if frame is None else int(frame)
        self._ultrack_db_selected_node_id = int(node_id)
        self._ultrack_db_selected_frame = selected_frame
        self._update_ultrack_db_highlight(self._ultrack_db_preview_labels, int(display_label))
        annot = self._ultrack_db_node_annotations.get(int(node_id), "UNKNOWN")
        annot_suffix = "" if annot == "UNKNOWN" else f" [{annot}]"
        parts = [f"Selected node {node_id}{annot_suffix} at t={selected_frame}"]
        probability = self._ultrack_db_label_probabilities.get(int(display_label))
        if probability is not None:
            parts.append(f"p={float(probability):.3f}")
        db_path = self._ultrack_db_path()
        if db_path is not None and db_path.exists():
            link_counts = self._query_ultrack_db_link_annotation_counts(db_path, int(node_id))
            parts.append(
                f"links REAL {link_counts['REAL']}, "
                f"FAKE {link_counts['FAKE']}, UNKNOWN {link_counts['UNKNOWN']}"
            )
        self._set_ultrack_db_status(" | ".join(parts) + ".")
        if self.ultrack_db_connected_focus_check.isChecked():
            self._refresh_ultrack_db_browser()

    def _deselect_ultrack_db_node(self) -> None:
        if self._ultrack_db_selected_node_id is None:
            return
        self._ultrack_db_selected_node_id = None
        self._ultrack_db_selected_frame = None
        self._clear_ultrack_db_highlight()
        self._set_ultrack_db_status("")
        if self.ultrack_db_connected_focus_check.isChecked():
            self._refresh_ultrack_db_browser()

    def _refresh_ultrack_db_selection_highlight(self, labels, status, node_id_to_label, frame):
        sel = self._ultrack_db_selected_node_id
        if sel is None:
            self._clear_ultrack_db_highlight()
            return status
        dl = node_id_to_label.get(int(sel))
        if dl is None:
            self._clear_ultrack_db_highlight()
            annot = self._query_ultrack_db_node_annotation_for_status(node_id_to_label, sel)
            if annot in {"REAL", "FAKE"}:
                return (
                    f"{status} Selected node {sel} [{annot}] is hidden "
                    f"by annotation filter at frame {frame}."
                )
            return (
                f"{status} Selected node {sel} is hidden "
                f"at frame {frame} and the current union size / merge group."
            )
        self._update_ultrack_db_highlight(labels, int(dl))
        return status

    def _query_ultrack_db_node_annotation_for_status(self, node_id_to_label, selected_node_id):
        return self._ultrack_db_node_annotations.get(int(selected_node_id), "UNKNOWN")

    def _get_ultrack_db_highlight_layer(self):
        if _ULTRACK_DB_SELECTION_LAYER in self.viewer.layers:
            return self.viewer.layers[_ULTRACK_DB_SELECTION_LAYER]
        layer = self.viewer.add_shapes(
            name=_ULTRACK_DB_SELECTION_LAYER, ndim=2,
            edge_color="cyan", edge_width=2, face_color="transparent",
        )
        layer.visible = False
        # Adding a layer makes it the active selection in napari, which
        # would steal mouse events from the preview layer. Restore the
        # preview layer as active so its mouse_drag_callbacks keep firing.
        self._restore_ultrack_db_preview_active()
        return layer

    def _restore_ultrack_db_preview_active(self) -> None:
        if _ULTRACK_DB_PREVIEW_LAYER not in self.viewer.layers:
            return
        try:
            self.viewer.layers.selection.active = self.viewer.layers[_ULTRACK_DB_PREVIEW_LAYER]
        except Exception:
            pass

    def _update_ultrack_db_highlight(self, labels, display_label):
        layer = self._get_ultrack_db_highlight_layer()
        if labels is None or display_label == 0:
            layer.data = []
            layer.visible = False
            return
        mask = (labels == int(display_label)).astype(np.uint8)
        if not np.any(mask):
            layer.data = []
            layer.visible = False
            return
        from skimage.measure import find_contours
        contours = find_contours(mask, level=0.5)
        if not contours:
            layer.data = []
            layer.visible = False
            return
        layer.data = [max(contours, key=len)]
        layer.shape_type = ["polygon"]
        layer.visible = True

    def _clear_ultrack_db_highlight(self) -> None:
        if _ULTRACK_DB_SELECTION_LAYER not in self.viewer.layers:
            return
        layer = self.viewer.layers[_ULTRACK_DB_SELECTION_LAYER]
        layer.data = []
        layer.visible = False

    def _query_ultrack_db_connected_nodes(self, db_path, selected_node_id):
        return _query_ultrack_db_connected_nodes(db_path, selected_node_id)

    def _query_ultrack_db_link_annotation_counts(self, db_path, selected_node_id):
        return _query_ultrack_db_link_annotation_counts(db_path, selected_node_id)

    def _render_ultrack_db_connected_focus(
        self, db_path, frame, labels, status, prob_dict, label_to_node_id, node_id_to_label,
    ):
        sel_nid = self._ultrack_db_selected_node_id
        sel_frame = self._ultrack_db_selected_frame
        if sel_nid is None or sel_frame is None:
            self._clear_ultrack_db_highlight()
            return labels, f"{status} Click a DB preview node to focus links.", {}
        predecessors, successors = self._query_ultrack_db_connected_nodes(db_path, sel_nid)
        if frame == sel_frame:
            relation = "selected"
            allowed = {sel_nid: 1.0}
            if int(sel_nid) not in node_id_to_label:
                self._clear_ultrack_db_highlight()
                empty = np.zeros_like(labels, dtype=np.uint32)
                annot = self._ultrack_db_node_annotations.get(int(sel_nid), "UNKNOWN")
                suf = "" if annot == "UNKNOWN" else f" [{annot}]"
                return empty, (
                    f"Selected node {sel_nid}{suf} at t={sel_frame} is hidden."
                ), {}
        elif frame == sel_frame - 1:
            relation = "t-1"
            allowed = predecessors
        elif frame == sel_frame + 1:
            relation = "t+1"
            allowed = successors
        else:
            self._clear_ultrack_db_highlight()
            return np.zeros_like(labels, dtype=np.uint32), (
                f"Selected node {sel_nid} at t={sel_frame} | frame {frame}: outside focus."
            ), {}

        focused = np.zeros_like(labels, dtype=np.uint32)
        alpha_dict: dict[int, float] = {}
        for lid, nid in label_to_node_id.items():
            li, ni = int(lid), int(nid)
            if ni not in allowed:
                continue
            focused[labels == li] = li
            if self.ultrack_db_prob_alpha_check.isChecked():
                alpha_dict[li] = (
                    1.0 if ni == sel_nid
                    else self._ultrack_db_connected_alpha(li, prob_dict)
                )

        sel_label = node_id_to_label.get(int(sel_nid))
        if frame == sel_frame and sel_label is not None:
            self._update_ultrack_db_highlight(focused, int(sel_label))
        else:
            self._clear_ultrack_db_highlight()

        edge_vals = [
            float(v) for nid, v in allowed.items()
            if nid in node_id_to_label and nid != sel_nid
        ]
        edge_summary = (
            f" | edge range {min(edge_vals):.2f}-{max(edge_vals):.2f}" if edge_vals else ""
        )
        count = int(np.unique(focused[focused != 0]).size)
        annot = self._ultrack_db_node_annotations.get(int(sel_nid), "UNKNOWN")
        suf = "" if annot == "UNKNOWN" else f" [{annot}]"
        return focused, (
            f"Selected node {sel_nid}{suf} at t={sel_frame} | "
            f"{relation}: {count} connected{edge_summary}"
        ), alpha_dict

    def _ultrack_db_connected_alpha(self, label_id, prob_dict):
        alpha = 1.0
        if self.ultrack_db_prob_alpha_check.isChecked() and prob_dict:
            probs = [float(v) for v in prob_dict.values()]
            min_p, max_p = min(probs), max(probs)
            denom = max(max_p - min_p, 1e-9)
            prob = float(prob_dict.get(int(label_id), 1.0))
            alpha *= 0.15 + 0.85 * (prob - min_p) / denom
        return float(np.clip(alpha, 0.05, 1.0))

    def _ultrack_db_summary_text(self, db_path, frame):
        return _ultrack_db_summary_text(db_path, frame)

    def _cached_ultrack_db_summary_text(self, db_path, mtime_ns, frame):
        key = (str(db_path.resolve()), mtime_ns, int(frame))
        cached = self._ultrack_db_summary_cache.get(key)
        if cached is not None:
            return cached
        text = self._ultrack_db_summary_text(db_path, frame)
        self._ultrack_db_summary_cache[key] = text
        return text

    def _ultrack_db_preview_cache_key(self, db_path, mtime_ns, frame, union_size, color_index):
        return (
            str(db_path.resolve()), mtime_ns, int(frame),
            int(union_size), int(color_index),
        )

    def _query_union_sizes(self, db_path, mtime_ns, frame):
        key = (str(db_path.resolve()), mtime_ns, int(frame))
        cached = self._ultrack_db_size_values_cache.get(key)
        if cached is not None:
            return cached
        sizes = _query_union_sizes(db_path, int(frame))
        self._ultrack_db_size_values_cache[key] = sizes
        return sizes

    def _query_union_color_classes(self, db_path, mtime_ns, frame, union_size):
        key = (str(db_path.resolve()), mtime_ns, int(frame), int(union_size))
        cached = self._ultrack_db_color_class_cache.get(key)
        if cached is not None:
            return cached
        classes = _query_union_color_classes(db_path, int(frame), int(union_size))
        self._ultrack_db_color_class_cache[key] = classes
        return classes

    def _configure_ultrack_db_size_slider(self, db_path, mtime_ns, frame):
        """Vertical axis. Range over the distinct union sizes present in ``frame``."""
        sizes = self._query_union_sizes(db_path, mtime_ns, frame)
        maximum = max(len(sizes) - 1, 0)
        value = min(max(int(self.ultrack_db_hierarchy_slider.value()), 0), maximum)
        old = self.ultrack_db_hierarchy_slider.blockSignals(True)
        try:
            self.ultrack_db_hierarchy_slider.setRange(0, maximum)
            self.ultrack_db_hierarchy_slider.setValue(value)
        finally:
            self.ultrack_db_hierarchy_slider.blockSignals(old)
        if sizes:
            self._set_ultrack_db_size_label(value, sizes[value], len(sizes))
        else:
            self.ultrack_db_height_lbl.setText("—")
        return sizes

    def _configure_ultrack_db_merge_slider(self, db_path, mtime_ns, frame, union_size):
        """Horizontal axis. Range over the merge groups (color classes) of ``union_size``."""
        classes = self._query_union_color_classes(db_path, mtime_ns, frame, union_size)
        maximum = max(len(classes) - 1, 0)
        value = min(max(int(self.ultrack_db_source_slider.value()), 0), maximum)
        old = self.ultrack_db_source_slider.blockSignals(True)
        try:
            self.ultrack_db_source_slider.setRange(0, maximum)
            self.ultrack_db_source_slider.setValue(value)
        finally:
            self.ultrack_db_source_slider.blockSignals(old)
        if classes:
            self._set_ultrack_db_merge_label(value, len(classes))
        else:
            self.ultrack_db_source_lbl.setText("—")
        return classes

    def _set_ultrack_db_size_label(self, index, union_size, total):
        self.ultrack_db_height_lbl.setText(
            f"N={int(union_size)} ({index + 1}/{total})"
        )

    def _set_ultrack_db_merge_label(self, index, total):
        if total <= 0:
            self.ultrack_db_source_lbl.setText("—")
            return
        self.ultrack_db_source_lbl.setText(f"{index + 1}/{total}")

    def _render_union_partition(self, db_path, frame, color_node_ids):
        return _render_union_partition(
            db_path,
            frame,
            color_node_ids,
            plane_shape=self._viewer_plane_shape(),
        ).as_tuple()

    def _finalize_hierarchy_nodes(self, nodes, frame, *, empty_msg, status_suffix):
        from cellflow.tracking_ultrack.db_query import finalize_hierarchy_nodes

        return finalize_hierarchy_nodes(
            nodes,
            frame,
            plane_shape=self._viewer_plane_shape(),
            empty_msg=empty_msg,
            status_suffix=status_suffix,
        ).as_tuple()

    @staticmethod
    def _ultrack_db_annotation_name(value):
        return _ultrack_db_annotation_name(value)

    @staticmethod
    def _ultrack_db_node_preview_metadata(nodes):
        return _ultrack_db_node_preview_metadata(nodes)

    @staticmethod
    def _ultrack_db_node_annotation_metadata(nodes):
        return _ultrack_db_node_annotation_metadata(nodes)

    def _empty_ultrack_db_preview(self):
        return np.zeros(self._viewer_plane_shape(), dtype=np.uint32)

    def _viewer_plane_shape(self):
        for layer in self.viewer.layers:
            data = getattr(layer, "data", None)
            if isinstance(data, np.ndarray) and data.ndim >= 2:
                return tuple(int(v) for v in data.shape[-2:])
        return (1, 1)

    def _paint_ultrack_db_nodes(self, nodes):
        return _paint_ultrack_db_nodes(nodes, self._viewer_plane_shape())

    @staticmethod
    def _node_mask_and_bbox(node):
        return _node_mask_and_bbox(node)
