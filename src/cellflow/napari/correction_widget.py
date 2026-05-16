"""Label correction widget for CellFlow v2."""
from __future__ import annotations

import logging
import os
from typing import Callable

import napari
import napari.layers
import numpy as np
from napari.utils.notifications import show_error
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import distance_transform_edt
from skimage.measure import find_contours

from cellflow.correction.labels import (
    _label_at,
    draw_cell_path,
    erase_cell,
    clean_stranded_pixels,
    fill_label_holes,
    fix_label_semiholes,
    merge_cells,
    relabel_cell,
    split_draw,
    swap_labels,
)
from cellflow.napari.ui_style import (
    action_button,
    checked_success_button,
    danger_button,
    muted_label,
    status_label,
)

log = logging.getLogger("cellflow.correction")
if os.environ.get("CELLFLOW_DEBUG"):
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[cellflow.correction] %(levelname)s %(message)s"))
        log.addHandler(_h)

_DRAW_LAYER      = "[Correction] CorrectionDraw"
_HIGHLIGHT_LAYER = "[Correction] CellHighlight"
_SPOTLIGHT_LAYER = "[Correction] CellSpotlight"
_SPOTLIGHT_OPACITY = 0.7
_SPOTLIGHT_SCALE = 3.0


class CorrectionWidget(QWidget):
    """Dock widget for interactive label correction."""

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: QWidget | None = None,
        *,
        show_activate_btn: bool = True,
        show_shortcuts: bool = True,
        inspector_first: bool = False,
        spotlight: bool = True,           # ← NEW
        show_cleanup: bool = True,        # ← NEW
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._show_activate_btn = show_activate_btn
        self._show_shortcuts = show_shortcuts
        self._inspector_first = inspector_first
        self._spotlight = spotlight              # ← NEW
        self._show_cleanup = show_cleanup        # ← NEW

        self._layer: napari.layers.Labels | None = None

        self._selected_label: int = 0
        self._selected_pos = None
        self._selected_t: int = -1
        self._swap_first_pos = None
        self._swap_first_t: int = -1

        self._drag_callbacks: list = []
        self._bound_keys: list = []

        self._in_deactivate: bool = False

        self._saved_viewer_drag_cbs: list = []
        self._saved_layer_mode: str = "pan_zoom"
        self._saved_layer_contour: int = 0

        self._edit_callback: Callable[[int, set[int]], None] | None = None
        self._selection_callback: Callable[[int, int], None] | None = None

        self._setup_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)

        self._activate_btn = QPushButton("Activate on selected layer")
        self._activate_btn.setCheckable(True)
        self._activate_btn.setToolTip(
            "Enable interactive mouse callbacks for merging/splitting."
        )
        action_button(self._activate_btn, expand=True)
        checked_success_button(self._activate_btn)
        self._activate_btn.clicked.connect(self._toggle_active)
        if self._show_activate_btn:
            root.addWidget(self._activate_btn)

        attrib = QLabel(
            "Correction tools adapted from "
            '<a href="https://github.com/Image-Analysis-Hub/Epicure">Epicure</a>.'
            "<br>If you use these tools, please cite:<br>"
            '<a href="https://doi.org/10.64898/2026.03.27.714683">'
            "doi:10.64898/2026.03.27.714683</a>"
        )
        attrib.setOpenExternalLinks(True)
        attrib.setWordWrap(True)
        muted_label(attrib, size_pt=9)

        self._outline_btn = QCheckBox("Show outlines only")
        self._outline_btn.setEnabled(False)
        action_button(self._outline_btn, expand=True)
        self._outline_btn.toggled.connect(self._toggle_outline)
        root.addWidget(self._outline_btn)

        self._reset_mode_btn = QPushButton("⚠  Restore correction mode")
        self._reset_mode_btn.setVisible(False)
        action_button(self._reset_mode_btn, expand=True)
        danger_button(self._reset_mode_btn)
        self._reset_mode_btn.clicked.connect(self._reset_tool_mode)
        root.addWidget(self._reset_mode_btn)

        # ── cleanup section (wrapped in container) ────────────── # ← CHANGED
        self._cleanup_container = QWidget()                        # ← NEW
        _clay = QVBoxLayout(self._cleanup_container)               # ← NEW
        _clay.setContentsMargins(0, 0, 0, 0)                      # ← NEW
        _clay.setSpacing(6)                                        # ← NEW

        cleanup_label = QLabel("Artifact cleanup")
        muted_label(cleanup_label, size_pt=9)
        _clay.addWidget(cleanup_label)                             # ← CHANGED (was root)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope:"))
        self._cleanup_scope_combo = QComboBox()
        self._cleanup_scope_combo.addItems(["Current frame", "All frames"])
        self._cleanup_scope_combo.setToolTip(
            "Choose whether cleanup applies to the visible frame or the full label stack."
        )
        scope_row.addWidget(self._cleanup_scope_combo)
        _clay.addLayout(scope_row)                                 # ← CHANGED

        hole_row = QHBoxLayout()
        hole_row.addWidget(QLabel("Hole radius:"))
        self._hole_radius_spin = QSpinBox()
        self._hole_radius_spin.setRange(0, 999)
        self._hole_radius_spin.setValue(5)
        self._hole_radius_spin.setToolTip(
            "Maximum pixel distance for filling enclosed background gaps. Set to 0 to skip gap filling."
        )
        hole_row.addWidget(self._hole_radius_spin)
        _clay.addLayout(hole_row)                                  # ← CHANGED

        semihole_row = QHBoxLayout()
        semihole_row.addWidget(QLabel("Max opening:"))
        self._semihole_opening_spin = QSpinBox()
        self._semihole_opening_spin.setRange(0, 999)
        self._semihole_opening_spin.setValue(3)
        self._semihole_opening_spin.setToolTip(
            "Maximum border contact, in pixels, for semihole repair. Set to 0 to skip semihole repair."
        )
        semihole_row.addWidget(self._semihole_opening_spin)
        _clay.addLayout(semihole_row)                              # ← CHANGED

        self._fill_holes_btn = QPushButton("Fill Holes")
        self._fill_holes_btn.setEnabled(False)
        self._fill_holes_btn.setToolTip("Fill enclosed background gaps using the configured hole radius.")
        action_button(self._fill_holes_btn, expand=True)
        self._fill_holes_btn.clicked.connect(self._fill_holes)
        _clay.addWidget(self._fill_holes_btn)                      # ← CHANGED

        self._fix_semiholes_btn = QPushButton("Fix Semiholes")
        self._fix_semiholes_btn.setEnabled(False)
        self._fix_semiholes_btn.setToolTip(
            "Repair narrow border-connected gaps using the radius and max opening controls."
        )
        action_button(self._fix_semiholes_btn, expand=True)
        self._fix_semiholes_btn.clicked.connect(self._fix_semiholes)
        _clay.addWidget(self._fix_semiholes_btn)                   # ← CHANGED

        self._clean_fragments_btn = QPushButton("Clean Fragments")
        self._clean_fragments_btn.setEnabled(False)
        self._clean_fragments_btn.setToolTip("Remove disconnected same-label fragments without filling background holes.")
        action_button(self._clean_fragments_btn, expand=True)
        self._clean_fragments_btn.clicked.connect(self._clean_fragments)
        _clay.addWidget(self._clean_fragments_btn)                 # ← CHANGED

        if self._show_cleanup:
            root.addWidget(self._cleanup_container)

        self._status = QLabel("Inactive")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_label(self._status, italic=True, muted=True)
        root.addWidget(self._status)

        inspect_group = QGroupBox("Inspect cell")
        inspect_lay = QVBoxLayout(inspect_group)

        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Cell ID:"))
        self._goto_cell_id = QSpinBox()
        self._goto_cell_id.setRange(0, 999_999)
        self._goto_cell_id.setValue(0)
        self._goto_cell_id.setSpecialValueText("—")
        self._goto_cell_id.setEnabled(False)
        self._goto_cell_id.valueChanged.connect(self._goto_cell)
        id_row.addWidget(self._goto_cell_id)
        inspect_lay.addLayout(id_row)

        self._inspect_frames_label = QLabel("")
        self._inspect_frames_label.setWordWrap(True)
        muted_label(self._inspect_frames_label, size_pt=9)
        inspect_lay.addWidget(self._inspect_frames_label)

        ref_group = self.build_shortcuts_widget()

        if self._inspector_first:
            root.addWidget(inspect_group)
            if self._show_shortcuts:
                root.addWidget(ref_group)
        else:
            if self._show_shortcuts:
                root.addWidget(ref_group)
            root.addWidget(inspect_group)

        self._attrib_lbl = attrib
        if self._show_activate_btn:
            root.addWidget(self._attrib_lbl)
        root.addStretch()

    def build_shortcuts_widget(self) -> QWidget:
        group = QGroupBox("Correction shortcuts")
        lay = QVBoxLayout(group)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(8)
        self._add_shortcut_group(
            lay,
            "Selection",
            [
                ("Left-click", "Select / highlight cell"),
                ("Shift+Left / Shift+Right", "Previous / next cell"),
            ],
        )
        self._add_shortcut_group(
            lay,
            "Manual Labels",
            [
                ("Middle-click or Delete", "Erase cell"),
                ("Ctrl+Left-click", "Merge selected with clicked cell"),
                ("Right-click variants", "Swap labels"),
                ("Shift+Left-drag", "Draw / extend cell path"),
                ("Shift+Right-drag", "Split by drawn line"),
            ],
        )
        self._add_shortcut_group(lay, "History", [("Ctrl+Z", "Undo")])
        return group

    @staticmethod
    def _add_shortcut_group(
        lay: QVBoxLayout,
        title: str,
        rows: list[tuple[str, str]],
    ) -> None:
        title_lbl = QLabel(title)
        title_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        lay.addWidget(title_lbl)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Plain)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: rgba(255, 255, 255, 0.25); border: none;")
        lay.addWidget(sep)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)
        for row, (key, desc) in enumerate(rows):
            key_lbl = QLabel(key)
            key_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            grid.addWidget(key_lbl, row, 0)
            grid.addWidget(desc_lbl, row, 1)
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)

    # ── activation ────────────────────────────────────────────────────────────

    def _toggle_active(self, checked: bool) -> None:
        if checked:
            layer = self.viewer.layers.selection.active
            if layer is None:
                self._activate_btn.setChecked(False)
                self._set_status("Select a Labels layer first", error=True)
                return
            if not isinstance(layer, napari.layers.Labels):
                self._activate_btn.setChecked(False)
                self._set_status("Not a Labels layer", error=True)
                return
            self._activate(layer)
        else:
            self._deactivate()

    def _activate(self, layer: napari.layers.Labels) -> None:
        log.debug("activate: layer='%s' shape=%s", layer.name, layer.data.shape)
        self._layer = layer
        self._selected_label = 0
        self._selected_pos = None
        self._selected_t = -1
        self._swap_first_pos = None
        self._swap_first_t = -1

        if hasattr(self.viewer, "mouse_drag_callbacks"):
            self._saved_viewer_drag_cbs = list(self.viewer.mouse_drag_callbacks)
            self.viewer.mouse_drag_callbacks.clear()
        else:
            self._saved_viewer_drag_cbs = []

        self._saved_layer_mode = layer.mode
        self._saved_layer_contour = int(layer.contour)
        layer.mode = "pan_zoom"

        self.viewer.layers.selection.active = layer
        self._get_draw_layer()
        if self._spotlight:                                        # ← NEW
            self._get_spotlight_layer()
        self._get_highlight_layer()

        self.viewer.dims.events.current_step.connect(self._on_dims_change)
        layer.events.data.connect(self._on_layer_data_changed)
        layer.events.paint.connect(self._on_layer_data_changed)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        layer.events.mode.connect(self._on_layer_mode_change)

        self._register_callbacks()
        self._activate_btn.setText("Deactivate")
        self._outline_btn.setEnabled(True)
        self._set_cleanup_enabled(True)
        self._outline_btn.setChecked(True)
        self._toggle_outline(True)
        self._goto_cell_id.setEnabled(True)
        self._set_status(f"Active on '{layer.name}'")

    def _deactivate(self) -> None:
        if self._in_deactivate:
            return
        self._in_deactivate = True
        try:
            self._deactivate_impl()
        finally:
            self._in_deactivate = False

    def _deactivate_impl(self) -> None:
        log.debug("deactivate: layer='%s'", self._layer.name if self._layer else None)
        if self._layer is not None:
            self._remove_callbacks()

            for disconnect in [
                lambda: self.viewer.dims.events.current_step.disconnect(self._on_dims_change),
                lambda: self.viewer.layers.events.removed.disconnect(self._on_layer_removed),
                lambda: self._layer.events.data.disconnect(self._on_layer_data_changed),
                lambda: self._layer.events.paint.disconnect(self._on_layer_data_changed),
                lambda: self._layer.events.mode.disconnect(self._on_layer_mode_change),
            ]:
                try:
                    disconnect()
                except Exception:
                    pass

            try:
                self._layer.mode = self._saved_layer_mode
            except Exception:
                pass
            try:
                self._layer.contour = self._saved_layer_contour
            except Exception:
                pass

            if hasattr(self.viewer, "mouse_drag_callbacks"):
                self.viewer.mouse_drag_callbacks.clear()
                for cb in self._saved_viewer_drag_cbs:
                    self.viewer.mouse_drag_callbacks.append(cb)

        self._layer = None
        self._selected_label = 0
        self._selected_pos = None
        self._selected_t = -1
        self._swap_first_pos = None
        self._swap_first_t = -1
        self._saved_viewer_drag_cbs = []

        self._activate_btn.setText("Activate on selected layer")
        self._activate_btn.setChecked(False)
        self._outline_btn.setChecked(False)
        self._outline_btn.setEnabled(False)
        self._set_cleanup_enabled(False)
        self._goto_cell_id.setEnabled(False)
        self._goto_cell_id.setValue(0)
        self._inspect_frames_label.setText("")
        self._set_status("Inactive")
        self._cleanup_draw_layer()
        self._cleanup_highlight_layer()
        if self._spotlight:                                        # ← NEW
            self._cleanup_spotlight_layer()

    def activate_layer(self, layer: napari.layers.Labels) -> None:
        """Activate correction on a specific Labels layer (bypasses the UI button)."""
        if self._layer is not None:
            self._deactivate()
        self._activate(layer)
        self._activate_btn.setChecked(True)

    def deactivate(self) -> None:
        """Deactivate correction (public API)."""
        self._deactivate()

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status.setText(msg)
        if error:
            self._status.setStyleSheet("color: #b00020; font-style: italic;")
        else:
            status_label(self._status, italic=True, muted=True)

    def _set_cleanup_enabled(self, enabled: bool) -> None:
        for button in (
            self._fill_holes_btn,
            self._fix_semiholes_btn,
            self._clean_fragments_btn,
        ):
            button.setEnabled(enabled)

    def set_edit_callback(self, fn: Callable[[int, set[int]], None] | None) -> None:
        self._edit_callback = fn

    def set_selection_callback(self, fn: Callable[[int, int], None] | None) -> None:
        self._selection_callback = fn

    def select_label(self, t: int, label: int, *, notify: bool = True) -> None:
        self._update_highlight(t, label, notify=notify)

    def _cleanup_frame_indices(self) -> list[int]:
        if self._layer is None:
            return []
        if self._layer.data.ndim < 3:
            return [0]
        if self._cleanup_scope_combo.currentText() == "All frames":
            return list(range(int(self._layer.data.shape[0])))
        return [int(self.viewer.dims.current_step[0])]

    def _run_artifact_cleanup(
        self,
        operation_name: str,
        no_change_message: str,
        operation: Callable[[np.ndarray], None],
    ) -> None:
        if self._layer is None:
            self._set_status("No active labels layer", error=True)
            return
        try:
            changed_frames = 0
            changed_pixels = 0
            for t in self._cleanup_frame_indices():
                seg2d = self._frame_view(self._layer, t)
                before = seg2d.copy()
                operation(seg2d)
                changed = int(np.sum(before != seg2d))
                if not changed:
                    continue
                changed_frames += 1
                changed_pixels += changed
                self._record_history(self._layer, t, before)

            if changed_pixels:
                self._layer.refresh()
                current_t = (
                    int(self.viewer.dims.current_step[0])
                    if self._layer.data.ndim >= 3
                    else 0
                )
                if self._selected_label:
                    self._update_highlight(current_t, self._selected_label)
                self._set_status(
                    f"{operation_name} in {changed_frames} frame(s), {changed_pixels} px changed. Unsaved."
                )
            else:
                self._set_status(no_change_message)
        except Exception as exc:
            show_error(f"cleanup error: {exc}")

    def _fill_holes(self) -> None:
        radius = int(self._hole_radius_spin.value())
        self._run_artifact_cleanup(
            "Filled holes",
            "No holes found",
            lambda seg2d: np.copyto(seg2d, fill_label_holes(seg2d, radius=radius)),
        )

    def _fix_semiholes(self) -> None:
        radius = int(self._hole_radius_spin.value())
        max_opening = int(self._semihole_opening_spin.value())
        self._run_artifact_cleanup(
            "Fixed semiholes",
            "No semiholes found",
            lambda seg2d: np.copyto(
                seg2d,
                fix_label_semiholes(seg2d, radius=radius, max_opening=max_opening),
            ),
        )

    def _clean_fragments(self) -> None:
        self._run_artifact_cleanup(
            "Cleaned fragments",
            "No fragments found",
            lambda seg2d: clean_stranded_pixels(seg2d),
        )

    @staticmethod
    def _frame_view(layer, t: int) -> np.ndarray:
        if layer.data.ndim == 2:
            return layer.data
        v = layer.data[t]
        while v.ndim > 2:
            if v.shape[0] != 1:
                raise ValueError(f"non-singleton dim in frame slice: shape={v.shape}")
            v = v[0]
        return v

    def _next_free_label(self) -> int:
        if self._layer is None:
            return 1
        return int(np.max(self._layer.data)) + 1

    def _record_history(self, layer, t: int, before: np.ndarray) -> None:
        after = self._frame_view(layer, t)
        changed = np.where(before != after)
        if not changed[0].size:
            return
        n = changed[0].size
        extra = layer.data.ndim - 1 - 2
        parts = [np.full(n, t, dtype=layer.data.dtype)]
        parts.extend(np.zeros(n, dtype=layer.data.dtype) for _ in range(extra))
        parts.extend(changed)
        layer._save_history((tuple(parts), before[changed], after[changed]))
        if self._edit_callback is not None:
            ids = set(int(v) for v in before[changed]) | set(int(v) for v in after[changed])
            ids.discard(0)
            if ids:
                try:
                    self._edit_callback(t, ids)
                except Exception:
                    import logging as _logging
                    _logging.getLogger("cellflow.correction").exception("edit_callback failed")

    # ── draw layer ────────────────────────────────────────────────────────────

    def _get_draw_layer(self):
        if _DRAW_LAYER in self.viewer.layers:
            return self.viewer.layers[_DRAW_LAYER]
        dl = self.viewer.add_shapes(
            name=_DRAW_LAYER,
            ndim=2,
            edge_color="yellow",
            edge_width=1,
            face_color="transparent",
        )
        dl.visible = False
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer
        return dl

    def _cleanup_draw_layer(self) -> None:
        if _DRAW_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_DRAW_LAYER])

    # ── highlight layer ───────────────────────────────────────────────────────

    def _get_highlight_layer(self):
        if _HIGHLIGHT_LAYER in self.viewer.layers:
            return self.viewer.layers[_HIGHLIGHT_LAYER]
        hl = self.viewer.add_shapes(
            name=_HIGHLIGHT_LAYER,
            ndim=2,
            edge_color="cyan",
            edge_width=2,
            face_color="transparent",
        )
        hl.visible = False
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer
        return hl

    def _get_spotlight_layer(self):
        if _SPOTLIGHT_LAYER in self.viewer.layers:
            return self.viewer.layers[_SPOTLIGHT_LAYER]
        spotlight = self.viewer.add_image(
            np.zeros((1, 1, 4), dtype=np.float32),
            name=_SPOTLIGHT_LAYER,
            rgb=True,
            blending="translucent",
        )
        spotlight.visible = False
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer
        return spotlight

    def _notify_selection_changed(self, t: int, lab: int, previous_label: int) -> None:
        if lab == previous_label or self._selection_callback is None:
            return
        try:
            self._selection_callback(t, lab)
        except Exception:
            import logging as _logging
            _logging.getLogger("cellflow.correction").exception("selection_callback failed")

    def _update_highlight(self, t: int, lab: int, *, notify: bool = True) -> None:
        previous_label = self._selected_label
        self._selected_label = lab
        self._selected_t = t if lab != 0 else -1
        old = self._goto_cell_id.blockSignals(True)
        try:
            self._goto_cell_id.setValue(int(lab))
        finally:
            self._goto_cell_id.blockSignals(old)
        hl = self._get_highlight_layer()
        if lab == 0 or self._layer is None:
            hl.data = []
            hl.visible = False
            if self._spotlight:                                    # ← NEW
                self._clear_spotlight()
            if notify:
                self._notify_selection_changed(t, lab, previous_label)
            return
        seg2d = self._frame_view(self._layer, t)
        if not np.any(seg2d == lab):
            self._selected_label = 0
            self._selected_t = -1
            hl.data = []
            hl.visible = False
            if self._spotlight:                                    # ← NEW
                self._clear_spotlight()
            if notify:
                self._notify_selection_changed(t, 0, previous_label)
            return
        mask = (seg2d == lab).astype(np.uint8)
        contours = find_contours(mask, level=0.5)
        if not contours:
            self._selected_label = 0
            self._selected_t = -1
            hl.data = []
            hl.visible = False
            if self._spotlight:                                    # ← NEW
                self._clear_spotlight()
            if notify:
                self._notify_selection_changed(t, 0, previous_label)
            return
        if self._spotlight:                                        # ← NEW
            self._update_spotlight(mask.astype(bool))
        contour = max(contours, key=len)
        hl.data = [contour]
        hl.shape_type = ["polygon"]
        hl.visible = True
        self.viewer.layers.selection.active = self._layer
        if notify:
            self._notify_selection_changed(t, lab, previous_label)

    def _cleanup_highlight_layer(self) -> None:
        if _HIGHLIGHT_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_HIGHLIGHT_LAYER])

    def _update_spotlight(self, mask: np.ndarray) -> None:
        spotlight = self._get_spotlight_layer()
        outer_mask = self._scaled_mask(mask, scale=_SPOTLIGHT_SCALE)
        ring = outer_mask & ~mask
        alpha = np.full(mask.shape, _SPOTLIGHT_OPACITY, dtype=np.float32)
        if np.any(ring):
            inner_dist = distance_transform_edt(~mask)
            outer_dist = distance_transform_edt(outer_mask)
            denom = inner_dist + outer_dist
            ramp = np.divide(
                inner_dist,
                denom,
                out=np.zeros_like(inner_dist, dtype=np.float64),
                where=denom > 0,
            )
            alpha[ring] = (ramp[ring] * _SPOTLIGHT_OPACITY).astype(np.float32)
        alpha[mask] = 0.0
        data = np.zeros(mask.shape + (4,), dtype=np.float32)
        data[..., 3] = alpha
        spotlight.data = data
        spotlight.visible = True
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer

    @staticmethod
    def _scaled_mask(mask: np.ndarray, *, scale: float) -> np.ndarray:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return np.zeros_like(mask, dtype=bool)
        center = coords.mean(axis=0)
        yy, xx = np.indices(mask.shape)
        src_y = np.rint(center[0] + (yy - center[0]) / scale).astype(int)
        src_x = np.rint(center[1] + (xx - center[1]) / scale).astype(int)
        np.clip(src_y, 0, mask.shape[0] - 1, out=src_y)
        np.clip(src_x, 0, mask.shape[1] - 1, out=src_x)
        return mask[src_y, src_x]

    def _clear_spotlight(self) -> None:
        if _SPOTLIGHT_LAYER in self.viewer.layers:
            spotlight = self.viewer.layers[_SPOTLIGHT_LAYER]
            spotlight.data = np.zeros((1, 1, 4), dtype=np.float32)
            spotlight.visible = False

    def _cleanup_spotlight_layer(self) -> None:
        if _SPOTLIGHT_LAYER in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[_SPOTLIGHT_LAYER])

    def _on_dims_change(self, event=None) -> None:
        if not (self._selected_label and self._layer is not None):
            return
        step = self.viewer.dims.current_step
        if self._layer.data.ndim < 3 or len(step) < self._layer.data.ndim:
            return
        t = int(step[0])
        if t >= self._layer.data.shape[0]:
            return
        selected_label = self._selected_label
        selected_pos = self._selected_pos
        selected_t = self._selected_t
        self._update_highlight(t, selected_label, notify=False)
        self._selected_label = selected_label
        self._selected_pos = selected_pos
        self._selected_t = selected_t

    def _on_layer_data_changed(self, event=None) -> None:
        if not (self._selected_label and self._layer is not None):
            return
        step = self.viewer.dims.current_step
        if self._layer.data.ndim < 3 or len(step) < self._layer.data.ndim:
            return
        t = int(step[0])
        if t >= self._layer.data.shape[0]:
            return
        self._update_highlight(t, self._selected_label)

    def _on_layer_mode_change(self, event=None) -> None:
        if self._layer is None:
            return
        mode = getattr(event, "value", None) or self._layer.mode
        log.debug("_on_layer_mode_change: mode=%s", mode)
        if mode != "pan_zoom":
            self._reset_mode_btn.setVisible(True)
            self._set_status("Tool mode changed — corrections disabled", error=True)
        else:
            self._reset_mode_btn.setVisible(False)
            if self._layer is not None:
                self._set_status(f"Active on '{self._layer.name}'")

    def _on_layer_removed(self, event=None) -> None:
        removed = getattr(event, "value", None)
        removed_name = getattr(removed, "name", None)
        helper_names = {_DRAW_LAYER, _HIGHLIGHT_LAYER}             # ← CHANGED
        if self._spotlight:                                        # ← NEW
            helper_names.add(_SPOTLIGHT_LAYER)                     # ← NEW
        if removed is self._layer or removed_name in helper_names: # ← CHANGED
            log.debug("_on_layer_removed: '%s' removed, deactivating", removed_name)
            self._deactivate()

    def _reset_tool_mode(self) -> None:
        if self._layer is not None:
            self._layer.mode = "pan_zoom"

    # ── inspect cell ──────────────────────────────────────────────────────────

    def _goto_cell(self) -> None:
        lab = self._goto_cell_id.value()
        if lab == 0:
            step = self.viewer.dims.current_step
            t = int(step[0]) if (self._layer is not None and self._layer.data.ndim >= 3 and len(step) >= 1) else 0
            self._update_highlight(t, 0)
            self._inspect_frames_label.setText("")
            return
        if self._layer is None:
            return
        data = self._layer.data
        frames = [i for i in range(data.shape[0]) if np.any(data[i] == lab)]
        if not frames:
            self._inspect_frames_label.setText(f"Cell {lab} not found in any frame.")
            step = self.viewer.dims.current_step
            t = int(step[0]) if len(step) >= 1 else 0
            self._update_highlight(t, 0)
            return
        _MAX = 20
        if len(frames) <= _MAX:
            frames_str = ", ".join(str(f) for f in frames)
        else:
            shown = ", ".join(str(f) for f in frames[:_MAX])
            frames_str = f"{shown}, … ({len(frames)} frames total)"
        self._inspect_frames_label.setText(f"Frames: {frames_str}")
        step = self.viewer.dims.current_step
        t = int(step[0]) if len(step) >= 1 else 0
        self._update_highlight(t, lab)

    def _step_cell(self, direction: int) -> None:
        if self._layer is None:
            return
        data = self._layer.data
        all_ids = sorted(set(int(v) for v in np.unique(data)) - {0})
        if not all_ids:
            self._set_status("No cells in any frame")
            return
        step = self.viewer.dims.current_step
        t = int(step[0]) if len(step) >= 1 else 0
        if 0 <= t < data.shape[0]:
            frame_ids = sorted(set(int(v) for v in np.unique(data[t])) - {0})
        else:
            frame_ids = []
        cur = self._selected_label

        # Nothing selected: start on the current frame if it has any cells.
        if cur == 0 and frame_ids:
            nxt = frame_ids[0] if direction > 0 else frame_ids[-1]
            self._goto_cell_id.setValue(nxt)
            self._goto_cell()
            return

        # Cycle within the current frame first.
        if direction > 0:
            nxt = next((i for i in frame_ids if i > cur), None)
        else:
            nxt = next((i for i in reversed(frame_ids) if i < cur), None)
        if nxt is not None:
            self._goto_cell_id.setValue(nxt)
            self._goto_cell()
            return

        # Current frame exhausted: fall back to IDs on other frames.
        other_ids = [i for i in all_ids if i not in frame_ids]
        if other_ids:
            if direction > 0:
                nxt = next((i for i in other_ids if i > cur), other_ids[0])
            else:
                nxt = next((i for i in reversed(other_ids) if i < cur), other_ids[-1])
        else:
            nxt = frame_ids[0] if direction > 0 else frame_ids[-1]
        frames = [i for i in range(data.shape[0]) if np.any(data[i] == nxt)]
        if frames:
            step_list = list(self.viewer.dims.current_step)
            step_list[0] = frames[0]
            self.viewer.dims.current_step = tuple(step_list)
        self._goto_cell_id.setValue(nxt)
        self._goto_cell()

    # ── callback registration ─────────────────────────────────────────────────

    def _register_callbacks(self) -> None:
        layer = self._layer

        def key_delete(_layer):
            try:
                if self._selected_label == 0:
                    self._set_status("No cell selected — left-click a cell first")
                    return
                t = int(self.viewer.dims.current_step[0])
                seg2d = self._frame_view(_layer, t)
                before = seg2d.copy()
                if erase_cell(seg2d, label=self._selected_label):
                    self._record_history(_layer, t, before)
                    _layer.refresh()
                    self._update_highlight(t, 0)
                    self._set_status(f"Erased — Active on '{_layer.name}'")
            except Exception as exc:
                show_error(f"delete error: {exc}")

        def key_prev_cell(_layer):
            self._step_cell(-1)

        def key_next_cell(_layer):
            self._step_cell(1)

        for key, fn in [
            ("Delete", key_delete),
            ("Shift-Left", key_prev_cell),
            ("Shift-Right", key_next_cell),
        ]:
            layer.bind_key(key, fn, overwrite=True)
            self._bound_keys.append(key)

        def on_drag(_caller, event):
            _layer = self._layer
            if _layer is None:
                return
            try:
                if event.type != "mouse_press":
                    return

                t   = int(self.viewer.dims.current_step[0])
                btn = event.button
                mods = {m.name for m in event.modifiers}

                seg2d = self._frame_view(_layer, t)
                pos   = _layer.world_to_data(event.position)
                log.debug(
                    "on_drag: btn=%s mods=%s t=%d selected=%s",
                    btn, mods, t, self._selected_label,
                )

                if btn == 3 and not mods:
                    lab = _label_at(seg2d, pos)
                    if lab == 0:
                        return
                    before = seg2d.copy()
                    if erase_cell(seg2d, label=lab):
                        self._record_history(_layer, t, before)
                        _layer.refresh()
                        if lab == self._selected_label:
                            self._update_highlight(t, 0)
                        self._set_status(f"Erased — Active on '{_layer.name}'")
                    return

                if btn == 2 and mods == {"Control"}:
                    lab = _label_at(seg2d, pos)
                    if lab == 0:
                        self._set_status("Swap — click on a cell (not background)")
                        return
                    if (
                        self._selected_label != 0
                        and self._selected_pos is not None
                        and lab != self._selected_label
                    ):
                        before = seg2d.copy()
                        ok = swap_labels(seg2d, self._selected_pos, pos)
                        if ok:
                            self._record_history(_layer, t, before)
                            _layer.refresh()
                            self._selected_label = 0
                            self._selected_pos = None
                            self._selected_t = -1
                            self._update_highlight(t, 0)
                            self._set_status(f"Swapped — Active on '{_layer.name}'")
                        else:
                            self._set_status("Swap failed — click on two different cells")
                    else:
                        self._swap_first_pos = pos
                        self._swap_first_t = t
                        self._set_status(f"Swap — label {lab} selected, right-click second cell")
                    return

                if btn == 2 and not mods:
                    if self._swap_first_pos is not None:
                        if t != self._swap_first_t:
                            self._swap_first_pos = None
                            self._swap_first_t = -1
                            self._set_status("Frame changed — swap cancelled")
                        else:
                            before = seg2d.copy()
                            ok = swap_labels(seg2d, self._swap_first_pos, pos)
                            if ok:
                                self._record_history(_layer, t, before)
                                _layer.refresh()
                                self._swap_first_pos = None
                                self._swap_first_t = -1
                                self._set_status(f"Swapped — Active on '{_layer.name}'")
                            else:
                                self._set_status("Swap failed — click on two different cells")
                                self._swap_first_pos = None
                                self._swap_first_t = -1
                    elif self._selected_label != 0 and self._selected_t != -1:
                        before = seg2d.copy()
                        if t != self._selected_t:
                            ok = relabel_cell(seg2d, pos, self._selected_label)
                            msg_ok  = f"Relabelled → {self._selected_label} — Active on '{_layer.name}'"
                            msg_err = "Relabel failed — click on a different cell"
                        else:
                            ok = swap_labels(seg2d, self._selected_pos, pos)
                            msg_ok  = f"Swapped — Active on '{_layer.name}'"
                            msg_err = "Swap failed — click on a different cell"
                        if ok:
                            self._record_history(_layer, t, before)
                            _layer.refresh()
                            self._set_status(msg_ok)
                        else:
                            self._set_status(msg_err)
                    return

                if btn == 1 and mods == {"Control"}:
                    lab = _label_at(seg2d, pos)
                    if lab == 0:
                        self._set_status("Click on a cell, not background")
                        return
                    if (
                        self._selected_label != 0
                        and lab != self._selected_label
                        and np.any(seg2d == self._selected_label)
                    ):
                        before = seg2d.copy()
                        ok = merge_cells(
                            seg2d, pos, pos,
                            label_a=lab, label_b=self._selected_label,
                        )
                        self._set_status(
                            f"Merged — Active on '{_layer.name}'"
                            if ok else "Merge failed — labels not touching"
                        )
                        if ok:
                            self._record_history(_layer, t, before)
                        _layer.refresh()
                        self._selected_label = 0
                        self._selected_pos = None
                        self._selected_t = -1
                        self._update_highlight(t, _label_at(seg2d, pos))
                    return

                if btn == 1 and not mods:
                    self._swap_first_pos = None
                    self._swap_first_t = -1
                    lab = _label_at(seg2d, pos)
                    self._selected_pos = pos if lab != 0 else None
                    self._selected_t = t if lab != 0 else -1
                    self._update_highlight(t, lab)
                    if lab:
                        self._set_status(f"Selected label {lab} — Active on '{_layer.name}'")
                    else:
                        self._set_status(f"Active on '{_layer.name}'")
                    return

                if mods == {"Shift"} and btn == 2:
                    dl = self._get_draw_layer()
                    dl.data = []
                    dl.visible = True
                    pos_list = [_layer.world_to_data(event.position)]
                    yield
                    while event.type == "mouse_move":
                        pos_list.append(_layer.world_to_data(event.position))
                        if len(pos_list) % 3 == 0:
                            dl.data = [np.array([[p[-2], p[-1]] for p in pos_list])]
                            dl.shape_type = ["path"]
                        yield
                    pos_list.append(_layer.world_to_data(event.position))
                    dl.data = []
                    dl.visible = False
                    self.viewer.layers.selection.active = _layer
                    curlabel = self._selected_label if self._selected_label else None
                    before = seg2d.copy()
                    ok = split_draw(
                        seg2d,
                        pos_list,
                        curlabel=curlabel,
                        new_label=self._next_free_label(),
                    )
                    self._set_status(
                        f"Split — Active on '{_layer.name}'"
                        if ok else "Split draw failed — line did not divide the cell"
                    )
                    if ok:
                        self._record_history(_layer, t, before)
                    _layer.refresh()
                    self._update_highlight(t, self._selected_label)
                    return

                if mods == {"Shift"} and btn == 1:
                    dl = self._get_draw_layer()
                    dl.data = []
                    dl.visible = True
                    pos_list = [_layer.world_to_data(event.position)]
                    yield
                    while event.type == "mouse_move":
                        pos_list.append(_layer.world_to_data(event.position))
                        if len(pos_list) % 3 == 0:
                            dl.data = [np.array([[p[-2], p[-1]] for p in pos_list])]
                            dl.shape_type = ["path"]
                        yield
                    pos_list.append(_layer.world_to_data(event.position))
                    dl.data = []
                    dl.visible = False
                    self.viewer.layers.selection.active = _layer
                    curlabel = self._selected_label if self._selected_label else None
                    before = seg2d.copy()
                    ok = draw_cell_path(
                        seg2d,
                        pos_list,
                        curlabel=curlabel,
                        new_label=self._next_free_label(),
                    )
                    self._set_status(
                        f"Drew cell path — Active on '{_layer.name}'"
                        if ok else "Draw failed — stroke too short"
                    )
                    if ok:
                        self._record_history(_layer, t, before)
                    _layer.refresh()
                    self._update_highlight(t, self._selected_label)
                    return

            except Exception as exc:
                import traceback
                show_error(f"Correction error: {exc}\n{traceback.format_exc()}")

        self.viewer.mouse_drag_callbacks.append(on_drag)
        self._drag_callbacks.append(on_drag)

    def _remove_callbacks(self) -> None:
        for fn in self._drag_callbacks:
            try:
                self.viewer.mouse_drag_callbacks.remove(fn)
            except (ValueError, AttributeError):
                pass
        self._drag_callbacks.clear()
        for key in self._bound_keys:
            try:
                layer.bind_key(key, None)
            except Exception:
                pass
        self._bound_keys.clear()

    def _toggle_outline(self, checked: bool) -> None:
        if self._layer is None:
            self._outline_btn.setChecked(False)
            return
        self._layer.contour = 2 if checked else 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _image_frame(self, t: int) -> np.ndarray | None:
        for lyr in self.viewer.layers:
            if getattr(lyr, "name", None) == _SPOTLIGHT_LAYER:
                continue
            if isinstance(lyr, napari.layers.Image):
                d = lyr.data
                if d.ndim == 2:
                    return d
                v = d[t] if d.ndim >= 3 else d
                while v.ndim > 2:
                    if v.shape[0] != 1:
                        return None
                    v = v[0]
                return v
        return None
