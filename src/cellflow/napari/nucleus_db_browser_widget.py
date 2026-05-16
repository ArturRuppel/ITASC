"""Ultrack database browser section for the nucleus workflow widget."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.widgets import CollapsibleSection
from cellflow.tracking_ultrack.db_query import (
    HierarchyCutState as _HierarchyCutState,
    annotation_name as _ultrack_db_annotation_name,
    node_annotation_metadata as _ultrack_db_node_annotation_metadata,
    node_mask_and_bbox as _node_mask_and_bbox,
    node_preview_metadata as _ultrack_db_node_preview_metadata,
    paint_nodes as _paint_ultrack_db_nodes,
    query_available_sources as _query_available_sources,
    query_connected_nodes as _query_ultrack_db_connected_nodes,
    query_distinct_heights as _query_distinct_heights,
    query_hierarchy_cut_states as _query_hierarchy_cut_states,
    query_middle_frame as _query_ultrack_db_middle_frame,
    render_hierarchy_cut as _render_hierarchy_cut,
    render_hierarchy_cut_state as _render_hierarchy_cut_state,
    summary_text as _ultrack_db_summary_text,
)

logger = logging.getLogger(__name__)

_ULTRACK_DB_PREVIEW_LAYER = "Ultrack DB Preview"
_ULTRACK_DB_SELECTION_LAYER = "Ultrack DB Selection"
_ULTRACK_DB_ANNOTATION_LAYER = "Ultrack DB Annotations"


class NucleusUltrackDbBrowserWidget(QWidget):
    """Qt controls for browsing a generated Ultrack database."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.info_lbl = QLabel("—")
        self.info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum,
        )
        lay.addWidget(self.info_lbl)

        self.source_slider = QSlider(Qt.Horizontal)
        self.source_slider.setRange(0, 0)
        self.source_slider.setValue(0)
        self.source_slider.setToolTip(
            "Select threshold source: 0 = lowest threshold, higher = more stringent"
        )
        self.source_slider.setEnabled(False)
        self.source_lbl = QLabel("all")
        self.source_lbl.setFixedWidth(48)
        self.source_slider_row = QWidget()
        source_slider_lay = QHBoxLayout(self.source_slider_row)
        source_slider_lay.setContentsMargins(0, 0, 0, 0)
        source_slider_lay.addWidget(self.source_slider)
        source_slider_lay.addWidget(self.source_lbl)
        lay.addWidget(self.source_slider_row)

        self.hierarchy_slider = QSlider(Qt.Horizontal)
        self.hierarchy_slider.setRange(0, 100)
        self.hierarchy_slider.setValue(50)
        self.hierarchy_slider.setToolTip(
            "Hierarchy cut level: 0 = most split, 1 = most merged"
        )
        self.hierarchy_slider.setEnabled(False)
        self.height_lbl = QLabel("0.50")
        self.height_lbl.setFixedWidth(48)
        self.slider_row = QWidget()
        slider_lay = QHBoxLayout(self.slider_row)
        slider_lay.setContentsMargins(0, 0, 0, 0)
        slider_lay.addWidget(self.hierarchy_slider)
        slider_lay.addWidget(self.height_lbl)
        lay.addWidget(self.slider_row)

        db_btn_row = QWidget()
        db_btn_lay = QHBoxLayout(db_btn_row)
        db_btn_lay.setContentsMargins(0, 0, 0, 0)
        db_btn_lay.setSpacing(4)
        self.active_btn = QPushButton("Activate Database Browser")
        self.active_btn.setCheckable(True)
        self.active_btn.setChecked(False)
        self.active_btn.setToolTip(
            "Load contour maps and foreground masks into viewer and enable DB preview"
        )
        self.refresh_btn = QPushButton()
        self.refresh_btn.setToolTip("Refresh Ultrack database browser")
        self.refresh_btn.setIcon(QIcon.fromTheme("view-refresh"))
        self.refresh_btn.setEnabled(False)
        db_btn_lay.addWidget(self.refresh_btn)
        lay.addWidget(db_btn_row)

        self.prob_alpha_check = QCheckBox("Node prob transparency")
        self.prob_alpha_check.setToolTip("Modulate label opacity by node probability")
        self.prob_alpha_check.setEnabled(False)
        self.connected_focus_check = QCheckBox("Connected focus")
        self.connected_focus_check.setToolTip(
            "Focus the DB preview on a selected node and its temporal neighbors"
        )
        self.connected_focus_check.setEnabled(False)
        self.edge_alpha_check = QCheckBox("Edge weight transparency")
        self.edge_alpha_check.setToolTip(
            "Modulate connected-neighbor opacity by link weight"
        )
        self.edge_alpha_check.setEnabled(False)
        self.show_validated_check = QCheckBox("Show validated nodes")
        self.show_validated_check.setChecked(True)
        self.show_validated_check.setEnabled(False)
        self.show_fake_check = QCheckBox("Show fake nodes")
        self.show_fake_check.setChecked(False)
        self.show_fake_check.setEnabled(False)
        for cb in (
            self.prob_alpha_check,
            self.connected_focus_check,
            self.edge_alpha_check,
            self.show_validated_check,
            self.show_fake_check,
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
        self._ultrack_db_height_values_cache: dict[tuple, tuple[float, ...]] = {}
        self._ultrack_db_cut_state_cache: dict[tuple, tuple[_HierarchyCutState, ...]] = {}
        self._ultrack_db_sources_cache: dict[tuple, tuple[int, ...]] = {}
        self._ultrack_db_browser_active: bool = False
        self._ultrack_db_frame_initialized: bool = False
        self._ultrack_db_selected_node_id: int | None = None
        self._ultrack_db_selected_frame: int | None = None
        self._ultrack_db_label_to_node_id: dict[int, int] = {}
        self._ultrack_db_node_id_to_label: dict[int, int] = {}
        self._ultrack_db_node_annotations: dict[int, str] = {}
        self._ultrack_db_preview_labels: np.ndarray | None = None
        self._ultrack_db_preview_mouse_callback = None

    def _build_db_browser_section(self, root: QVBoxLayout) -> None:
        self.ultrack_db_browser_widget = NucleusUltrackDbBrowserWidget(self)
        self.ultrack_db_browser_widget.section = CollapsibleSection(
            "Database Browser",
            self.ultrack_db_browser_widget,
            expanded=False,
            title_role="indicators",
            title_level=1,
        )
        self.ultrack_db_browser_widget.section._toggle.setVisible(False)
        self.ultrack_db_browser_widget.section._toggle.setEnabled(False)
        self._alias_ultrack_db_browser_controls()

    def _alias_ultrack_db_browser_controls(self) -> None:
        browser = self.ultrack_db_browser_widget
        self.ultrack_db_browser_section = browser.section
        self.ultrack_db_info_lbl = browser.info_lbl
        self.ultrack_db_source_slider = browser.source_slider
        self.ultrack_db_source_lbl = browser.source_lbl
        self._ultrack_db_source_slider_row = browser.source_slider_row
        self.ultrack_db_hierarchy_slider = browser.hierarchy_slider
        self.ultrack_db_height_lbl = browser.height_lbl
        self._ultrack_db_slider_row = browser.slider_row
        self.ultrack_db_active_btn = browser.active_btn
        self.ultrack_db_refresh_btn = browser.refresh_btn
        self.ultrack_db_prob_alpha_check = browser.prob_alpha_check
        self.ultrack_db_connected_focus_check = browser.connected_focus_check
        self.ultrack_db_edge_alpha_check = browser.edge_alpha_check
        self.ultrack_db_show_validated_check = browser.show_validated_check
        self.ultrack_db_show_fake_check = browser.show_fake_check
        self.ultrack_db_section_status_lbl = browser.status_lbl

    def _set_ultrack_db_status(self, msg: str) -> None:
        self.ultrack_db_section_status_lbl.setText(msg)
        self.ultrack_db_section_status_lbl.setVisible(bool(msg))
        logger.info(msg)

    def _on_ultrack_db_browser_param_changed(self, *_args) -> None:
        self._ultrack_db_preview_cache.clear()

    def _on_ultrack_db_source_changed(self, value: int) -> None:
        if not self._ultrack_db_browser_active:
            return
        max_source = self.ultrack_db_source_slider.maximum()
        if max_source > 0:
            self.ultrack_db_source_lbl.setText(f"{value}/{max_source}")
        else:
            self.ultrack_db_source_lbl.setText("all")
        self._ultrack_db_preview_cache.clear()
        QTimer.singleShot(150, self._refresh_ultrack_db_browser)

    def _on_ultrack_db_slider_changed(self, value: int) -> None:
        if not self._ultrack_db_browser_active:
            return
        db_path = self._ultrack_db_path()
        if db_path is not None and db_path.exists():
            try:
                mtime_ns = db_path.stat().st_mtime_ns
                heights = self._query_distinct_heights(db_path, mtime_ns)
                index = min(max(int(value), 0), max(len(heights) - 1, 0))
                if heights:
                    self._set_ultrack_db_height_label(index, heights[index], len(heights))
                else:
                    self.ultrack_db_height_lbl.setText("—")
            except Exception:
                self.ultrack_db_height_lbl.setText(str(value))
        else:
            self.ultrack_db_height_lbl.setText(str(value))
        self._ultrack_db_preview_cache.clear()
        QTimer.singleShot(150, self._refresh_ultrack_db_browser)

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
            self.ultrack_db_browser_section.expand()
            self._ultrack_db_frame_initialized = False
            self._refresh_ultrack_db_browser()
        else:
            self._remove_ultrack_db_browser_layers()
            self.ultrack_db_browser_section.collapse()

    def _set_ultrack_db_controls_enabled(self, enabled: bool) -> None:
        self.ultrack_db_active_btn.setText(
            "Deactivate Database Browser" if enabled else "Activate Database Browser"
        )
        self.ultrack_db_refresh_btn.setEnabled(enabled)
        self.ultrack_db_source_slider.setEnabled(enabled)
        self.ultrack_db_hierarchy_slider.setEnabled(enabled)
        self.ultrack_db_prob_alpha_check.setEnabled(enabled)
        self.ultrack_db_connected_focus_check.setEnabled(enabled)
        self.ultrack_db_edge_alpha_check.setEnabled(enabled)
        self.ultrack_db_show_validated_check.setEnabled(enabled)
        self.ultrack_db_show_fake_check.setEnabled(enabled)

    def _remove_ultrack_db_browser_layers(self) -> None:
        self._remove_ultrack_db_preview_selector()
        for name in (_ULTRACK_DB_PREVIEW_LAYER, _ULTRACK_DB_ANNOTATION_LAYER):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        if _ULTRACK_DB_SELECTION_LAYER in self.viewer.layers:
            self.viewer.layers.remove(_ULTRACK_DB_SELECTION_LAYER)
        self.ultrack_db_info_lbl.setText("—")
        self._set_ultrack_db_status("")

    def _ultrack_db_middle_frame(self, db_path: Path) -> int | None:
        return _query_ultrack_db_middle_frame(db_path)

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
            if frame == 0:
                mid = self._ultrack_db_middle_frame(db_path)
                if mid is not None and mid > 0:
                    frame = mid
                    self._set_viewer_frame(frame)
        try:
            self.ultrack_db_info_lbl.setText(self._ultrack_db_summary_text(db_path, frame))
            mtime_ns = db_path.stat().st_mtime_ns
            self._configure_ultrack_db_source_slider(db_path, mtime_ns)
            states = self._configure_ultrack_db_hierarchy_slider(db_path, mtime_ns, frame)
            if not states:
                labels = self._empty_ultrack_db_preview()
                self._update_layer(_ULTRACK_DB_PREVIEW_LAYER, labels)
                self._set_ultrack_db_status(f"No hierarchy states for frame {frame}.")
                return
            slider_int = int(self.ultrack_db_hierarchy_slider.value())
            state = states[slider_int]
            key = (
                str(db_path.resolve()), mtime_ns, frame, slider_int, state,
                self.ultrack_db_show_validated_check.isChecked(),
                self.ultrack_db_show_fake_check.isChecked(),
            )
            cached = self._ultrack_db_preview_cache.get(key)
            if cached is None:
                cached = self._render_hierarchy_cut_state(db_path, frame, state)
                self._ultrack_db_preview_cache[key] = cached
            labels, status, prob_dict, label_to_node_id, node_id_to_label, node_annotations = (
                self._normalize_ultrack_db_preview(cached)
            )
            self._ultrack_db_label_to_node_id = label_to_node_id
            self._ultrack_db_node_id_to_label = node_id_to_label
            self._ultrack_db_node_annotations = node_annotations
            alpha_dict: dict[int, float] = {}
            if self.ultrack_db_connected_focus_check.isChecked():
                labels, status, alpha_dict = self._render_ultrack_db_connected_focus(
                    db_path, frame, labels, status, prob_dict,
                    label_to_node_id, node_id_to_label,
                )
            self._ultrack_db_preview_labels = labels.astype(np.uint32, copy=False)
            self._update_ultrack_db_preview_layer(
                self._ultrack_db_preview_labels, prob_dict, alpha_dict,
            )
            self._update_ultrack_db_annotation_layer(
                self._ultrack_db_preview_labels, label_to_node_id, node_annotations,
            )
            self._install_ultrack_db_preview_selector()
            if not self.ultrack_db_connected_focus_check.isChecked():
                status = self._refresh_ultrack_db_selection_highlight(
                    self._ultrack_db_preview_labels, status, node_id_to_label, frame,
                )
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
        if not np.any(overlay):
            if _ULTRACK_DB_ANNOTATION_LAYER in self.viewer.layers:
                self.viewer.layers.remove(_ULTRACK_DB_ANNOTATION_LAYER)
            return
        self._update_labels_layer(_ULTRACK_DB_ANNOTATION_LAYER, overlay)

    def _update_labels_layer(self, name: str, data: np.ndarray) -> None:
        from napari.layers import Labels
        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Labels):
            self.viewer.layers[name].data = data
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        self.viewer.add_labels(data, name=name)

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
        self._set_ultrack_db_status(f"Selected node {node_id}{annot_suffix} at t={selected_frame}.")
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
                f"at frame {frame} and the current hierarchy threshold."
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
        return layer

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
            alpha_on = (
                self.ultrack_db_edge_alpha_check.isChecked()
                or self.ultrack_db_prob_alpha_check.isChecked()
            )
            if alpha_on:
                alpha_dict[li] = (
                    1.0 if ni == sel_nid
                    else self._ultrack_db_connected_alpha(li, float(allowed[ni]), prob_dict)
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

    def _ultrack_db_connected_alpha(self, label_id, edge_weight, prob_dict):
        alpha = 1.0
        if self.ultrack_db_edge_alpha_check.isChecked():
            alpha *= float(edge_weight)
        if self.ultrack_db_prob_alpha_check.isChecked() and prob_dict:
            probs = [float(v) for v in prob_dict.values()]
            min_p, max_p = min(probs), max(probs)
            denom = max(max_p - min_p, 1e-9)
            prob = float(prob_dict.get(int(label_id), 1.0))
            alpha *= 0.15 + 0.85 * (prob - min_p) / denom
        return float(np.clip(alpha, 0.05, 1.0))

    def _ultrack_db_summary_text(self, db_path, frame):
        return _ultrack_db_summary_text(db_path, frame)

    def _query_distinct_heights(self, db_path, mtime_ns):
        key = (str(db_path.resolve()), mtime_ns)
        cached = self._ultrack_db_height_values_cache.get(key)
        if cached is not None:
            return cached
        heights = _query_distinct_heights(db_path)
        self._ultrack_db_height_values_cache[key] = heights
        return heights

    def _query_hierarchy_cut_states(self, db_path, mtime_ns, frame):
        source_idx = self.ultrack_db_source_slider.value()
        max_source = self.ultrack_db_source_slider.maximum()
        source_key = int(source_idx) if max_source > 0 else None
        key = (str(db_path.resolve()), mtime_ns, frame, source_key)
        cached = self._ultrack_db_cut_state_cache.get(key)
        if cached is not None:
            return cached
        result = _query_hierarchy_cut_states(db_path, frame, source_index=source_key)
        self._ultrack_db_cut_state_cache[key] = result
        return result

    def _query_available_sources(self, db_path, mtime_ns):
        key = (str(db_path.resolve()), mtime_ns, "sources")
        cached = self._ultrack_db_sources_cache.get(key)
        if cached is not None:
            return cached
        sources = _query_available_sources(db_path)
        self._ultrack_db_sources_cache[key] = sources
        return sources

    def _configure_ultrack_db_source_slider(self, db_path, mtime_ns):
        sources = self._query_available_sources(db_path, mtime_ns)
        if not sources:
            self.ultrack_db_source_slider.setRange(0, 0)
            self.ultrack_db_source_lbl.setText("all")
            return False
        max_source = max(sources)
        current = min(max(int(self.ultrack_db_source_slider.value()), 0), max_source)
        old = self.ultrack_db_source_slider.blockSignals(True)
        try:
            self.ultrack_db_source_slider.setRange(0, max_source)
            self.ultrack_db_source_slider.setValue(current)
        finally:
            self.ultrack_db_source_slider.blockSignals(old)
        self.ultrack_db_source_lbl.setText(f"{current}/{max_source}")
        return len(sources) > 1

    def _configure_ultrack_db_hierarchy_slider(self, db_path, mtime_ns, frame):
        states = self._query_hierarchy_cut_states(db_path, mtime_ns, frame)
        maximum = max(len(states) - 1, 0)
        value = min(max(int(self.ultrack_db_hierarchy_slider.value()), 0), maximum)
        old = self.ultrack_db_hierarchy_slider.blockSignals(True)
        try:
            self.ultrack_db_hierarchy_slider.setRange(0, maximum)
            self.ultrack_db_hierarchy_slider.setValue(value)
        finally:
            self.ultrack_db_hierarchy_slider.blockSignals(old)
        if states:
            self._set_ultrack_db_height_label(value, states[value].height, len(states))
        else:
            self.ultrack_db_height_lbl.setText("—")
        return states

    def _set_ultrack_db_height_label(self, index, height, total):
        ht = "—" if height is None else f"{height:.2f}"
        self.ultrack_db_height_lbl.setText(f"i={index} h={ht} ({index + 1}/{total})")

    def _render_hierarchy_cut(self, db_path, frame, h_actual):
        return _render_hierarchy_cut(
            db_path,
            frame,
            h_actual,
            plane_shape=self._viewer_plane_shape(),
            show_validated=self.ultrack_db_show_validated_check.isChecked(),
            show_fake=self.ultrack_db_show_fake_check.isChecked(),
        ).as_tuple()

    def _render_hierarchy_cut_state(self, db_path, frame, state):
        return _render_hierarchy_cut_state(
            db_path,
            frame,
            state,
            plane_shape=self._viewer_plane_shape(),
            show_validated=self.ultrack_db_show_validated_check.isChecked(),
            show_fake=self.ultrack_db_show_fake_check.isChecked(),
        ).as_tuple()

    def _finalize_hierarchy_nodes(self, nodes, frame, *, empty_msg, status_suffix):
        from cellflow.tracking_ultrack.db_query import finalize_hierarchy_nodes

        return finalize_hierarchy_nodes(
            nodes,
            frame,
            plane_shape=self._viewer_plane_shape(),
            show_validated=self.ultrack_db_show_validated_check.isChecked(),
            show_fake=self.ultrack_db_show_fake_check.isChecked(),
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
