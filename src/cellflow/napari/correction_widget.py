"""Label correction widget for CellFlow v2."""
from __future__ import annotations

import logging
import os
from collections.abc import Callable

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
from cellflow.napari._spotlight import spotlight_rgba as _spotlight_rgba
from cellflow.correction.labels import (
    _label_at,
    add_cell,
    carve_into_selected,
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
    add_block_pair_row,
    block_grid,
    checked_success_button,
    danger_button,
    muted_label,
    status_label,
)
from cellflow.napari._widget_helpers import islider as _islider

log = logging.getLogger("cellflow.correction")
if os.environ.get("CELLFLOW_DEBUG"):
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[cellflow.correction] %(levelname)s %(message)s"))
        log.addHandler(_h)

_DRAW_LAYER      = "[Correction] CorrectionDraw"
_SPOTLIGHT_LAYER = "[Correction] CellSpotlight"


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
        show_cleanup: bool = True,
        show_inspector: bool = True,
        show_spawn_controls: bool = True,
        contour_only: bool = False,
        highlight_style: str = "spotlight",
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._show_activate_btn = show_activate_btn
        self._show_shortcuts = show_shortcuts
        self._inspector_first = inspector_first
        self._show_cleanup = show_cleanup
        # The "Inspect cell" group is redundant where a lineage canvas provides
        # navigation; the spinbox/label still back the goto + Shift-±/step logic
        # even when the group is not added to the layout.
        self._show_inspector = show_inspector
        # The cell-correction panel is tied to the nucleus label set, so it must
        # not create / delete / renumber cells. ``contour_only`` restricts the
        # mouse toolkit to select + extend (Shift-left) + carve (Shift-right) and
        # hides the spawn radius control; ``highlight_style="border"`` outlines
        # the selection instead of dimming everything else.
        self._show_spawn_controls = show_spawn_controls
        self._contour_only = contour_only
        self._highlight_style = highlight_style

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
        self._selection_listeners: list[Callable[[int, int], None]] = []
        # Optional override for the spotlight cutout: given (t, lab, default_mask)
        # it may return a different boolean mask to highlight (e.g. the union of a
        # whole track's masks) or None to keep the default per-frame behavior.
        self._spotlight_mask_provider: (
            Callable[[int, int, np.ndarray], np.ndarray | None] | None
        ) = None
        self._protected_mask_callback: (
            Callable[[int, np.ndarray], np.ndarray | None] | None
        ) = None
        # Optional provider of a 2-D intensity frame (e.g. the nucleus image)
        # used to snap spawned cells to the underlying signal. None → always
        # stamp a plain disk.
        self._intensity_frame_callback: (
            Callable[[int], np.ndarray | None] | None
        ) = None

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

        cleanup_grid = block_grid(horizontal_spacing=12)
        self._cleanup_scope_combo = QComboBox()
        self._cleanup_scope_combo.addItems(["Current frame", "All frames"])
        self._cleanup_scope_combo.setToolTip(
            "Choose whether cleanup applies to the visible frame or the full label stack."
        )
        self._hole_radius_spin = _islider(0, 999, 5)
        self._hole_radius_spin.setToolTip(
            "Maximum pixel distance for filling enclosed background gaps. Set to 0 to skip gap filling."
        )
        self._semihole_opening_spin = _islider(0, 999, 3)
        self._semihole_opening_spin.setToolTip(
            "Maximum border contact, in pixels, for semihole repair. Set to 0 to skip semihole repair."
        )
        add_block_pair_row(
            cleanup_grid,
            0,
            "Scope:",
            self._cleanup_scope_combo,
            field_width=150,
        )
        add_block_pair_row(
            cleanup_grid,
            1,
            "Hole radius:",
            self._hole_radius_spin,
            "Max opening:",
            self._semihole_opening_spin,
        )
        _clay.addLayout(cleanup_grid)

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

        # ── spawn-cell controls ───────────────────────────────────────────────
        # Middle-clicking empty space spawns a cell; this sets its fallback size
        # when the nucleus signal is too weak to snap to. Wrapped in a container
        # so an embedder (e.g. the nucleus widget) can relocate it into its own
        # parameter panel; shown here when not relocated.
        self._cell_radius_spin = _islider(1, 999, 6)
        self._cell_radius_spin.setToolTip(
            "Radius (px) of a cell spawned by middle-clicking empty space. "
            "Used as a fallback size when the nucleus signal is too weak to snap to."
        )
        self._spawn_controls = QWidget()
        spawn_lay = QVBoxLayout(self._spawn_controls)
        spawn_lay.setContentsMargins(0, 0, 0, 0)
        spawn_lay.setSpacing(6)
        spawn_grid = block_grid(horizontal_spacing=12)
        add_block_pair_row(
            spawn_grid,
            0,
            "Cell radius:",
            self._cell_radius_spin,
        )
        spawn_lay.addLayout(spawn_grid)
        # Hidden in contour-only mode (no spawning), but kept alive as a member
        # so any code reading ``_cell_radius_spin`` still finds a live widget.
        if self._show_spawn_controls:
            root.addWidget(self._spawn_controls)

        self._status = QLabel("Inactive")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_label(self._status, italic=True, muted=True)
        root.addWidget(self._status)

        # Hold a reference even when the group is not added to the layout
        # (show_inspector=False): otherwise Python GCs the QGroupBox and its
        # children, deleting the C++ QSpinBox that goto/Shift-± still drive.
        inspect_group = QGroupBox("Inspect cell")
        self._inspect_group = inspect_group
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
            if self._show_inspector:
                root.addWidget(inspect_group)
            if self._show_shortcuts:
                root.addWidget(ref_group)
        else:
            if self._show_shortcuts:
                root.addWidget(ref_group)
            if self._show_inspector:
                root.addWidget(inspect_group)

        self._attrib_lbl = attrib
        if self._show_activate_btn:
            root.addWidget(self._attrib_lbl)
        root.addStretch()

    def build_shortcuts_widget(self) -> QWidget:
        group = QGroupBox("Correction shortcuts")
        grid = QGridLayout(group)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)
        row = 0
        row = self._add_shortcut_group(
            grid,
            "Selection",
            [
                ("Left-click", "Select / highlight cell"),
            ],
            start_row=row,
            is_first=True,
        )
        if self._contour_only:
            # Cell labels are tied to the nuclei: only contour edits are allowed.
            manual_rows = [
                ("Shift+Left-drag", "Extend the selected cell's contour"),
                ("Shift+Right-drag", "Draw a line through a neighbour to cut it and merge the near piece into the selected cell"),
            ]
        else:
            manual_rows = [
                ("Middle-click empty space", "Spawn new cell"),
                ("Middle-click on cell or Delete", "Erase cell"),
                ("Ctrl+Left-click", "Merge with the clicked cell (same frame)"),
                ("Ctrl+Middle-click", "Grow / link selected track here"),
                ("Ctrl+Right-click", "Swap with the clicked cell, or attach it to the selected track (other frame)"),
                ("Shift+Left-drag", "Draw / extend cell path"),
                ("Shift+Right-drag", "Split by drawn line"),
            ]
        row = self._add_shortcut_group(
            grid,
            "Contour Edits" if self._contour_only else "Manual Labels",
            manual_rows,
            start_row=row,
        )
        row = self._add_shortcut_group(
            grid, "History", [("Ctrl+Z", "Undo")], start_row=row
        )
        grid.setColumnStretch(1, 1)
        return group

    @staticmethod
    def _add_shortcut_group(
        grid: QGridLayout,
        title: str,
        rows: list[tuple[str, str]],
        *,
        start_row: int = 0,
        is_first: bool = False,
    ) -> int:
        row = start_row
        if not is_first:
            grid.setRowMinimumHeight(row, 6)
            row += 1
        title_lbl = QLabel(title)
        title_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        grid.addWidget(title_lbl, row, 0, 1, 2)
        row += 1
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Plain)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: rgba(255, 255, 255, 0.25); border: none;")
        grid.addWidget(sep, row, 0, 1, 2)
        row += 1
        for key, desc in rows:
            key_lbl = QLabel(key)
            key_lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
            key_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            sp = QSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding
            )
            sp.setHeightForWidth(True)
            desc_lbl.setSizePolicy(sp)
            grid.addWidget(key_lbl, row, 0)
            grid.addWidget(desc_lbl, row, 1)
            row += 1
        return row

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
        self._get_spotlight_layer()

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

    def add_selection_listener(self, fn: Callable[[int, int], None]) -> None:
        """Register an extra selection callback that survives ``set_selection_callback``.

        ``set_selection_callback`` is a single slot owned by the workflow widget,
        so registering there would clobber (or be clobbered by) it. Listeners
        accumulate instead, letting independent features (e.g. the track-path
        comet) react to selection changes without fighting over that slot.
        """
        if fn not in self._selection_listeners:
            self._selection_listeners.append(fn)

    def set_spotlight_mask_provider(
        self, fn: Callable[[int, int, np.ndarray], np.ndarray | None] | None
    ) -> None:
        self._spotlight_mask_provider = fn

    def set_highlight_style(self, style: str) -> None:
        """Switch the selection indicator between the dimming spotlight and a
        plain yellow border, re-rendering the current selection."""
        if style not in ("spotlight", "border"):
            raise ValueError(f"unknown highlight_style: {style!r}")
        if style == self._highlight_style:
            return
        self._highlight_style = style
        if self._selected_label and self._layer is not None and self._selected_t >= 0:
            self._update_highlight(
                self._selected_t, self._selected_label, notify=False
            )

    def set_protected_mask_callback(
        self,
        fn: Callable[[int, np.ndarray], np.ndarray | None] | None,
    ) -> None:
        self._protected_mask_callback = fn

    def set_intensity_frame_callback(
        self,
        fn: Callable[[int], np.ndarray | None] | None,
    ) -> None:
        """Provide a per-frame 2-D intensity image to guide spawned cells.

        ``fn(t)`` returns the nucleus/intensity frame for time ``t`` (matching
        the label frame shape) or None. When unset, middle-clicking empty space
        stamps a plain disk instead of snapping to the signal.
        """
        self._intensity_frame_callback = fn

    def _intensity_frame(self, t: int) -> np.ndarray | None:
        if self._intensity_frame_callback is None:
            return None
        try:
            frame = self._intensity_frame_callback(t)
        except Exception:
            return None
        return None if frame is None else np.asarray(frame)

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

    def _protected_mask(self, t: int, seg2d: np.ndarray) -> np.ndarray | None:
        if self._protected_mask_callback is None:
            return None
        mask = self._protected_mask_callback(t, seg2d)
        if mask is None:
            return None
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != seg2d.shape:
            raise ValueError(
                f"protected mask shape {mask.shape} does not match frame shape {seg2d.shape}"
            )
        return mask

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

    # ── selection spotlight ──────────────────────────────────────────────────

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
        if lab == previous_label:
            return
        import logging as _logging
        if self._selection_callback is not None:
            try:
                self._selection_callback(t, lab)
            except Exception:
                _logging.getLogger("cellflow.correction").exception(
                    "selection_callback failed"
                )
        for listener in list(self._selection_listeners):
            try:
                listener(t, lab)
            except Exception:
                _logging.getLogger("cellflow.correction").exception(
                    "selection_listener failed"
                )

    def _update_highlight(self, t: int, lab: int, *, notify: bool = True) -> None:
        """Spotlight the selected cell (or clear it when nothing is selected).

        The spotlight is the sole selection indicator: full brightness inside
        the selected cell's mask (or a provider-widened mask, e.g. a whole
        track's union), uniformly dimmed everywhere outside.
        """
        previous_label = self._selected_label
        self._selected_label = lab
        self._selected_t = t if lab != 0 else -1
        old = self._goto_cell_id.blockSignals(True)
        try:
            self._goto_cell_id.setValue(int(lab))
        finally:
            self._goto_cell_id.blockSignals(old)
        if lab == 0 or self._layer is None:
            self._clear_spotlight()
            if notify:
                self._notify_selection_changed(t, lab, previous_label)
            return
        seg2d = self._frame_view(self._layer, t)
        mask = seg2d == lab
        if not mask.any():
            self._selected_label = 0
            self._selected_t = -1
            self._clear_spotlight()
            if notify:
                self._notify_selection_changed(t, 0, previous_label)
            return
        # The border style outlines just the selected cell; only the dimming
        # spotlight widens to a provider mask (e.g. a whole track's union).
        if self._highlight_style == "border":
            highlight_mask = mask
        else:
            highlight_mask = self._resolve_spotlight_mask(t, lab, mask)
        self._update_spotlight(highlight_mask)
        self.viewer.layers.selection.active = self._layer
        if notify:
            self._notify_selection_changed(t, lab, previous_label)

    def _resolve_spotlight_mask(
        self, t: int, lab: int, default_mask: np.ndarray
    ) -> np.ndarray:
        """Let an optional provider widen the spotlight cutout (e.g. to a whole
        track's union), falling back to the per-frame ``default_mask``."""
        provider = self._spotlight_mask_provider
        if provider is None:
            return default_mask
        try:
            override = provider(t, lab, default_mask)
        except Exception:
            import logging as _logging
            _logging.getLogger("cellflow.correction").exception(
                "spotlight_mask_provider failed"
            )
            return default_mask
        if override is None:
            return default_mask
        override = np.asarray(override, dtype=bool)
        if override.shape != default_mask.shape or not override.any():
            return default_mask
        return override

    def _update_spotlight(self, mask: np.ndarray) -> None:
        # The mask is the selected cell, or the whole track's union when a
        # provider widens it. Two render styles:
        #   "spotlight" — full brightness inside the mask (alpha 0), uniformly
        #     dimmed everywhere outside it.
        #   "border" — a bright opaque ring around the mask, transparent
        #     elsewhere (leaves the rest of the frame untouched).
        spotlight = self._get_spotlight_layer()
        border = self._highlight_style == "border"
        data = _spotlight_rgba(mask, dim=not border, border=border)
        spotlight.data = data
        spotlight.visible = True
        if self._layer is not None:
            self.viewer.layers.selection.active = self._layer

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
        helper_names = {_DRAW_LAYER, _SPOTLIGHT_LAYER}
        if removed is self._layer or removed_name in helper_names:
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

    # ── selection-agnostic track edits ────────────────────────────────────────

    def _selected_label_pos(self, seg2d: np.ndarray) -> tuple[float, float] | None:
        """A representative ``(y, x)`` of the selected label within *seg2d*.

        The swap edits need a click point for the *selected* cell. Deriving it
        from the label itself lets them work no matter how the cell became
        selected — image click, lineage canvas, gallery, or the goto box. Only
        the image-click path ever recorded a click position, so depending on
        that stored position made these edits silently no-op for every other
        selection source. Returns ``None`` when the selected label is absent
        from this frame (so the caller can fall back to a link/relabel instead
        of a same-frame swap). The returned pixel is the one nearest the
        label's centroid, so it always lands inside the mask.
        """
        lab = self._selected_label
        if not lab:
            return None
        ys, xs = np.where(seg2d == lab)
        if ys.size == 0:
            return None
        cy, cx = ys.mean(), xs.mean()
        i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        return (float(ys[i]), float(xs[i]))

    def _spawn_into_selected(self, seg2d, pos, t: int, _layer) -> None:
        """Ctrl+middle-click: spawn a cell at *pos* carrying the selected ID.

        The new blob takes the selected cell's label, so it either *merges* into
        the selected cell (when that cell already lives in this frame) or *links*
        the selected track into this frame (when the selected cell is on another
        frame — typically the previous one the user just stepped from). The
        mechanics are identical — paint a fresh region with the selected ID —
        only the status wording differs.
        """
        sel = self._selected_label
        if not sel:
            self._set_status("Ctrl+middle-click: select a cell first")
            return
        if _label_at(seg2d, pos) != 0:
            self._set_status(
                "Ctrl+middle-click empty space to grow / link the selected cell"
            )
            return
        present_here = bool(np.any(seg2d == sel))
        before = seg2d.copy()
        ok = add_cell(
            seg2d,
            pos,
            new_label=sel,
            radius=int(self._cell_radius_spin.value()),
            image=self._intensity_frame(t),
            protected_mask=self._protected_mask(t, seg2d),
        )
        if not ok:
            self._set_status("Spawn failed — no room here")
            return
        self._record_history(_layer, t, before)
        _layer.refresh()
        self._update_highlight(t, sel)
        if present_here:
            self._set_status(f"Merged into cell {sel} — Active on '{_layer.name}'")
        else:
            self._set_status(f"Linked cell {sel} → t={t} — Active on '{_layer.name}'")

    def _ctrl_right_click_swap(self, seg2d, pos, t: int, _layer) -> None:
        """Ctrl+right-click: swap, attach to a track, or run a two-click swap.

        The selected cell's position is derived from its label, so the action
        fires however the cell was selected — not only after a left-click on it.
        Context picks the operation:

        * a pending two-click swap → this click is its second cell;
        * selected cell present in this frame → swap it with the clicked cell;
        * selected cell on another frame → attach the clicked cell to its track;
        * nothing usable selected → arm a two-click swap (this is the first cell;
          a following Ctrl+right-click picks the second).
        """
        lab = _label_at(seg2d, pos)
        if lab == 0:
            self._set_status("Swap — click on a cell (not background)")
            return

        # Finish a pending two-click swap.
        if self._swap_first_pos is not None:
            if t != self._swap_first_t:
                self._swap_first_pos = None
                self._swap_first_t = -1
                self._set_status("Frame changed — swap cancelled")
                return
            before = seg2d.copy()
            ok = swap_labels(seg2d, self._swap_first_pos, pos)
            self._swap_first_pos = None
            self._swap_first_t = -1
            if ok:
                self._record_history(_layer, t, before)
                _layer.refresh()
                self._set_status(f"Swapped — Active on '{_layer.name}'")
            else:
                self._set_status("Swap failed — click on two different cells")
            return

        sel_pos = self._selected_label_pos(seg2d)
        if self._selected_label != 0 and sel_pos is not None:
            # Selected cell is in this frame → swap it with the clicked cell.
            if lab == self._selected_label:
                return
            before = seg2d.copy()
            ok = swap_labels(seg2d, sel_pos, pos)
            if ok:
                self._record_history(_layer, t, before)
                _layer.refresh()
                # Keep the track selected after the swap (it now lives where the
                # clicked cell was), matching the attach-to-track path so the
                # selection survives whether or not the frame was occupied.
                self._update_highlight(t, self._selected_label)
                self._set_status(f"Swapped — Active on '{_layer.name}'")
            else:
                self._set_status("Swap failed — click on two different cells")
            return

        if self._selected_label != 0 and sel_pos is None:
            # Selected cell lives on another frame → attach the clicked cell to
            # its track by relabelling it into the selected ID.
            before = seg2d.copy()
            ok = relabel_cell(seg2d, pos, self._selected_label)
            if ok:
                self._record_history(_layer, t, before)
                _layer.refresh()
                self._update_highlight(t, self._selected_label)
                self._set_status(
                    f"Attached to track {self._selected_label} — "
                    f"Active on '{_layer.name}'"
                )
            else:
                self._set_status("Attach failed — click on a different cell")
            return

        # No usable selection → arm a two-click swap.
        self._swap_first_pos = pos
        self._swap_first_t = t
        self._set_status(
            f"Swap — label {lab} selected, Ctrl+right-click second cell"
        )

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

        # Contour-only mode forbids erasing a cell (it would desync from the
        # nuclei), so the Delete shortcut is not bound.
        key_bindings = [] if self._contour_only else [("Delete", key_delete)]
        for key, fn in key_bindings:
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

                if self._contour_only and not (
                    (btn == 1 and mods in (set(), {"Shift"}, {"Control", "Shift"}))
                    or (btn == 2 and mods == {"Shift"})
                ):
                    # Cell labels are tied to the nuclei — only select (left-click),
                    # extend (Shift+left-drag) and carve (Shift+right-drag) are
                    # allowed. Everything else would create/delete/renumber a cell.
                    return

                if btn == 3 and mods == {"Control"}:
                    self._spawn_into_selected(seg2d, pos, t, _layer)
                    return

                if btn == 3 and not mods:
                    lab = _label_at(seg2d, pos)
                    if lab == 0:
                        before = seg2d.copy()
                        ok = add_cell(
                            seg2d,
                            pos,
                            new_label=self._next_free_label(),
                            radius=int(self._cell_radius_spin.value()),
                            image=self._intensity_frame(t),
                            protected_mask=self._protected_mask(t, seg2d),
                        )
                        if ok:
                            self._record_history(_layer, t, before)
                            _layer.refresh()
                            self._set_status(f"Added cell — Active on '{_layer.name}'")
                        else:
                            self._set_status("Add cell failed — no room here")
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
                    self._ctrl_right_click_swap(seg2d, pos, t, _layer)
                    return

                if btn == 1 and mods == {"Control"}:
                    lab = _label_at(seg2d, pos)
                    if lab == 0:
                        self._set_status("Click on a cell, not background")
                        return
                    if self._selected_label == 0 or lab == self._selected_label:
                        return
                    if not np.any(seg2d == self._selected_label):
                        # Selected cell is on another frame → there is nothing
                        # here to merge with; attaching the clicked cell to its
                        # track is the Ctrl+right-click action instead.
                        self._set_status(
                            "Selected cell is on another frame — Ctrl+right-click "
                            "to attach the clicked cell to its track"
                        )
                        return
                    # Selected cell lives in this frame → merge the two cells.
                    before = seg2d.copy()
                    ok = merge_cells(
                        seg2d, pos, pos,
                        label_a=lab, label_b=self._selected_label,
                    )
                    self._set_status(
                        f"Merged — Active on '{_layer.name}'"
                        if ok else "Merge failed"
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
                    if self._contour_only:
                        if not self._selected_label:
                            self._set_status("Carve — select a cell first")
                            return
                        ok = carve_into_selected(
                            seg2d, pos_list, selected_label=self._selected_label,
                        )
                        self._set_status(
                            f"Carved into cell {self._selected_label} — "
                            f"Active on '{_layer.name}'"
                            if ok else
                            "Carve failed — draw a line all the way through a neighbour"
                        )
                    else:
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

                if btn == 1 and mods in ({"Shift"}, {"Control", "Shift"}):
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
                        protected_mask=self._protected_mask(t, seg2d),
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
                if self._layer is not None:
                    self._layer.bind_key(key, None)
            except Exception:
                pass
        self._bound_keys.clear()

    def _toggle_outline(self, checked: bool) -> None:
        if self._layer is None:
            self._outline_btn.setChecked(False)
            return
        self._layer.contour = 2 if checked else 0

    # ── helpers ───────────────────────────────────────────────────────────────

