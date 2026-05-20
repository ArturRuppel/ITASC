"""Correction section widget for the cell workflow."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile
from napari.qt.threading import thread_worker as _thread_worker
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from cellflow.correction.labels import (
    cleanup_movie,
    expand_label_to_foreground,
    fill_label_holes,
    fix_label_semiholes,
)
from cellflow.database.tracked import read_full_tracked_stack
from cellflow.napari._widget_helpers import (
    btn as _btn,
    button_grid as _button_grid,
    make_status as _make_status,
    tool_btn as _tool_btn,
)
from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.ui_style import (
    add_block_pair_row,
    block_grid,
    compact_spinbox,
    stage_header_label,
)
from cellflow.napari.widgets import CollapsibleSection
from cellflow.napari._widget_helpers import ispin as _ispin

logger = logging.getLogger(__name__)

# Layers created by the cell correction widget itself carry a [Correction]
# prefix so they can be cleanly removed on deactivate without clobbering the
# regular pipeline layers (Tracked: Cell, Cell z-avg, …) shown elsewhere.
_TRACKED_CELL_LAYER = "[Correction] Cell Labels"
_CELL_ZAVG_LAYER = "[Correction] Cell z-avg"
_NUC_ZAVG_LAYER = "[Correction] Nucleus z-avg"
_CELL_FOREGROUND_LAYER = "[Correction] Foreground Mask: Cell"

# Pipeline-side (non-correction) layer names — used as fallbacks where the
# user may have loaded layers manually, and as the target for the on-disk
# Tracked: Cell refresh when correction mode is exited.
_PIPELINE_TRACKED_CELL_LAYER = "Tracked: Cell"


class CellCorrectionWidget(QWidget):
    """Qt controls for cell tracking correction workflows.

    Owns the correction group-box UI, all correction-scoped button handlers,
    and the inline CorrectionWidget.  Does not hold a back-reference to the
    parent workflow widget.

    Parameters
    ----------
    viewer:
        The napari viewer instance.
    pos_dir_provider:
        Zero-argument callable that returns the current position directory
        (or None).  When None, the widget falls back to a locally stored
        ``_local_pos_dir``.
    files_widget_refresh_callback:
        Called with the current ``pos_dir`` after save operations so that the
        parent workflow can refresh its file status panel.
    parent:
        Optional Qt parent.
    """

    def __init__(
        self,
        viewer,
        *,
        pos_dir_provider: Callable[[], Path | None] | None = None,
        files_widget_refresh_callback: Callable[[Path | None], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir_provider = pos_dir_provider
        self._local_pos_dir: Path | None = None
        self._files_widget_refresh = files_widget_refresh_callback or (lambda _pd: None)
        self._correction_view_state: dict | None = None
        self._correction_owned_layers: set[str] = set()
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------ #
    # pos_dir property — delegate to provider if available                #
    # ------------------------------------------------------------------ #

    @property
    def _pos_dir(self) -> Path | None:
        if self._pos_dir_provider is not None:
            return self._pos_dir_provider()
        return self._local_pos_dir

    @_pos_dir.setter
    def _pos_dir(self, value: Path | None) -> None:
        self._local_pos_dir = value

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

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
            "Activate correction mode and show correction controls.",
            checkable=True,
        )
        self.active_btn.setToolTip(
            "Activate correction mode and show correction controls."
        )
        self.params_btn = _tool_btn(
            "⚙", "Show correction parameters.", checkable=True
        )
        self.shortcuts_btn = _tool_btn(
            "📖", "Show correction shortcuts.", checkable=True
        )

        # ── Action buttons (added later at bottom) ────────────────────
        self.load_labels_btn = _btn(
            "Load Labels", "Load tracked cell labels from disk.")
        self.save_labels_btn = _btn(
            "Save Labels", "Save tracked cell labels to disk.")
        self.fill_holes_btn = _btn(
            "Fill Holes",
            "Fill background holes fully enclosed within individual labels.",
        )
        self.fix_semiholes_btn = _btn(
            "Fix Semi Holes",
            "Bridge narrow channels in label boundaries and fill the pockets.",
        )
        self.cleanup_btn = _btn(
            "Clean Up",
            "All frames: clean fragments → resync to nuclear labels → remove orphans.",
        )
        self.expand_cell_btn = _btn(
            "Expand Cell",
            "Expand selected cell into adjacent foreground mask pixels.",
        )

        # ── Correction parameters (collapsible) ───────────────────────
        params_inner = QWidget(self)
        params_lay = QVBoxLayout(params_inner)
        params_lay.setContentsMargins(0, 0, 0, 0)
        params_lay.setSpacing(6)

        scope_row = QHBoxLayout()
        scope_lbl = QLabel("Scope:")
        scope_lbl.setToolTip("Applies to Fill Holes and Fix Semi Holes.")
        scope_row.addWidget(scope_lbl)
        self.correction_scope_combo = QComboBox()
        self.correction_scope_combo.addItems(["Current frame", "All frames"])
        self.correction_scope_combo.setToolTip(
            "Applies to Fill Holes and Fix Semi Holes. Clean Up always processes all frames."
        )
        scope_row.addWidget(self.correction_scope_combo)
        params_lay.addLayout(scope_row)

        g = block_grid(horizontal_spacing=12)
        self.hole_radius_spin = _ispin(
            0, 999, 5,
            tooltip="Max hole size (pixels) for fill / fix operations.",
        )
        self.semihole_opening_spin = _ispin(
            0, 999, 3,
            tooltip="Max channel width for semi-hole bridging.",
        )
        self.expand_max_px_spin = _ispin(
            0, 999, 25,
            tooltip="Max expansion distance in pixels.",
        )
        add_block_pair_row(g, 0,
            "Hole radius:", compact_spinbox(self.hole_radius_spin),
            "Max opening:", compact_spinbox(self.semihole_opening_spin))
        add_block_pair_row(g, 1,
            "Max expand px:", compact_spinbox(self.expand_max_px_spin))
        params_lay.addLayout(g)

        self.correction_params_section = CollapsibleSection(
            "Correction Parameters",
            params_inner,
            expanded=False,

        )
        self.correction_params_section.set_header_visible(False)
        self.correction_params_section.setVisible(False)
        group_lay.addWidget(self.correction_params_section)

        # ── Inline CorrectionWidget ───────────────────────────────────
        # Nucleus correction uses the default spotlight_scale (3.0). For cells
        # we want a noticeably larger spotlight halo (≈3× nucleus radius).
        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
            spotlight=True,
            spotlight_scale=9.0,
            show_cleanup=False,
        )

        self.correction_shortcuts_section = CollapsibleSection(
            "Correction Shortcuts",
            self.correction_widget.build_shortcuts_widget(),
            expanded=False,

        )
        self.correction_shortcuts_section.set_header_visible(False)
        self.correction_shortcuts_section.setVisible(False)
        group_lay.addWidget(self.correction_shortcuts_section)

        self.active_content = QWidget(self)
        active_lay = QVBoxLayout(self.active_content)
        active_lay.setContentsMargins(0, 0, 0, 0)
        active_lay.setSpacing(6)
        active_lay.addWidget(self.correction_widget)

        self.correction_status_lbl = _make_status()
        active_lay.addWidget(self.correction_status_lbl)

        active_lay.addLayout(_button_grid(
            (self.load_labels_btn, self.save_labels_btn),
            (self.fill_holes_btn, self.fix_semiholes_btn),
            (self.cleanup_btn, self.expand_cell_btn),
        ))
        active_lay.addWidget(self.correction_widget._attrib_lbl)
        self.active_content.setVisible(False)
        group_lay.addWidget(self.active_content)

        self.header = self._build_correction_header()

        self.section = CollapsibleSection(
            "Correction",
            inner,
            expanded=False,

        )
        self.section.set_header_visible(False)
        self.section._toggle.setEnabled(False)
        root.addWidget(self.header)
        root.addWidget(self.section)

        self.header_lbl = self.correction_header_lbl
        self.correction_active_btn = self.active_btn
        self.correction_params_btn = self.params_btn
        self.correction_shortcuts_btn = self.shortcuts_btn
        self.correction_mode_section = self.section

    def _build_correction_header(self) -> QWidget:
        header = QWidget(self)
        row = QHBoxLayout(header)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self.correction_header_lbl = QLabel("Correction")
        stage_header_label(self.correction_header_lbl, "cell")
        row.addWidget(self.correction_header_lbl)
        row.addStretch(1)
        row.addWidget(self.shortcuts_btn)
        row.addWidget(self.params_btn)
        row.addWidget(self.active_btn)
        return header

    # ------------------------------------------------------------------ #
    # Signal wiring                                                        #
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        self.load_labels_btn.clicked.connect(self._on_load_labels)
        self.save_labels_btn.clicked.connect(self._on_save_labels)
        self.fill_holes_btn.clicked.connect(self._on_fill_holes)
        self.fix_semiholes_btn.clicked.connect(self._on_fix_semiholes)
        self.cleanup_btn.clicked.connect(self._on_cleanup)
        self.expand_cell_btn.clicked.connect(self._on_expand_cell)
        self.active_btn.toggled.connect(self._on_active_button_toggled)
        self.params_btn.toggled.connect(self._on_params_button_toggled)
        self.shortcuts_btn.toggled.connect(self._on_shortcuts_button_toggled)

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
        show_active = self.active_btn.isChecked()

        self.correction_params_section.setVisible(show_params)
        self.correction_shortcuts_section.setVisible(show_shortcuts)
        self.active_content.setVisible(show_active)

        if show_params or show_shortcuts or show_active:
            self.section.expand()
        else:
            self.section.collapse()

    def _on_active_button_toggled(self, active: bool) -> None:
        if active:
            self._capture_correction_view_state()
            for layer in list(self.viewer.layers):
                layer.visible = False

            if not self._load_correction_layers_from_disk():
                self._restore_correction_view_state()
                self._set_checked_without_signal(self.active_btn, False)
                self._sync_correction_panel_visibility()
                return

            layer = self.viewer.layers[_TRACKED_CELL_LAYER]
            layer.visible = True
            for name in (_CELL_ZAVG_LAYER, _NUC_ZAVG_LAYER):
                if name in self.viewer.layers:
                    self.viewer.layers[name].visible = True
            self.viewer.layers.selection.active = layer
            self.correction_widget.activate_layer(layer)
            self._sync_correction_panel_visibility()
            return

        self.correction_widget.deactivate()
        self._refresh_tracked_layer_from_disk()
        self._remove_correction_owned_layers()
        self._restore_correction_view_state()
        self._sync_correction_panel_visibility()

    def _remove_correction_owned_layers(self) -> None:
        for name in list(self._correction_owned_layers):
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])
        self._correction_owned_layers.clear()

    def _correction_tracked_layer(self):
        """Return the [Correction] Cell Labels layer, or None if absent."""
        if _TRACKED_CELL_LAYER in self.viewer.layers:
            return self.viewer.layers[_TRACKED_CELL_LAYER]
        return None

    # ------------------------------------------------------------------ #
    # View state capture / restore (mirrors NucleusCorrectionWidget)     #
    # ------------------------------------------------------------------ #

    def _capture_correction_view_state(self) -> None:
        selected = [layer.name for layer in self.viewer.layers.selection]
        active = self.viewer.layers.selection.active
        self._correction_view_state = {
            "visibility": {
                layer.name: bool(layer.visible) for layer in self.viewer.layers
            },
            "active": active.name if active is not None else None,
            "selected": selected,
        }

    def _restore_correction_view_state(self) -> None:
        state = self._correction_view_state or {}
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
        self._correction_view_state = None

    def _load_correction_layers_from_disk(self) -> bool:
        lp = self._cell_labels_path()
        if lp is None or not lp.exists():
            self._correction_status("No cell labels file found.")
            return False
        try:
            labels = read_full_tracked_stack(lp)
        except Exception as exc:
            self._correction_status(f"Error reading cell labels: {exc}")
            return False
        czp, nzp = self._cell_foreground_path(), self._nuc_foreground_path()
        cz = (
            np.asarray(tifffile.imread(str(czp)), dtype=np.float32)
            if czp and czp.exists() else None
        )
        nz = (
            np.asarray(tifffile.imread(str(nzp)), dtype=np.float32)
            if nzp and nzp.exists() else None
        )
        self._apply_loaded_layers(labels, cz, nz)
        self._correction_status(f"Loaded cell label stack {labels.shape}.")
        return True

    def _refresh_tracked_layer_from_disk(self) -> None:
        """Reload the pipeline-side Tracked: Cell layer from disk on deactivate.

        Mirrors NucleusCorrectionWidget so a subsequent re-solve picks up any
        corrections saved during the session. The in-correction layer
        (``[Correction] Cell Labels``) is torn down separately via
        ``_remove_correction_owned_layers``.
        """
        lp = self._cell_labels_path()
        if lp is None or not lp.exists():
            return
        if _PIPELINE_TRACKED_CELL_LAYER not in self.viewer.layers:
            return
        try:
            data = np.asarray(read_full_tracked_stack(lp), dtype=np.uint32)
            self.viewer.layers[_PIPELINE_TRACKED_CELL_LAYER].data = data
        except Exception:
            pass

    def _on_params_button_toggled(self, checked: bool) -> None:
        self.correction_params_section._toggle.setChecked(checked)
        if checked:
            self._set_checked_without_signal(self.shortcuts_btn, False)
            self.correction_shortcuts_section._toggle.setChecked(False)
        self._sync_correction_panel_visibility()

    def _on_shortcuts_button_toggled(self, checked: bool) -> None:
        self.correction_shortcuts_section._toggle.setChecked(checked)
        if checked:
            self._set_checked_without_signal(self.params_btn, False)
            self.correction_params_section._toggle.setChecked(False)
        self._sync_correction_panel_visibility()

    # ------------------------------------------------------------------ #
    # Status helper                                                        #
    # ------------------------------------------------------------------ #

    def _correction_status(self, msg: str) -> None:
        self.correction_status_lbl.setText(msg)
        self.correction_status_lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    # ------------------------------------------------------------------ #
    # Path helpers (delegate to _pos_dir)                                 #
    # ------------------------------------------------------------------ #

    def _p(self, *parts: str) -> Path | None:
        return self._pos_dir.joinpath(*parts) if self._pos_dir else None

    def _cell_labels_path(self) -> Path | None:
        return self._p("3_cell", "tracked_labels.tif")

    def _cell_foreground_path(self) -> Path | None:
        return self._p("1_cellpose", "cell_foreground.tif")

    def _nuc_foreground_path(self) -> Path | None:
        return self._p("1_cellpose", "nucleus_foreground.tif")

    def _nuc_labels_path(self) -> Path | None:
        return self._p("2_nucleus", "tracked_labels.tif")

    def _foreground_path(self) -> Path | None:
        return self._p("3_cell", "foreground_masks.tif")

    # ------------------------------------------------------------------ #
    # Frame helpers                                                        #
    # ------------------------------------------------------------------ #

    def _current_t(self) -> int:
        step = getattr(getattr(self.viewer, "dims", None), "current_step", (0,))
        return int(step[0]) if len(step) >= 1 else 0

    def _current_time_index(self, max_t: int) -> int:
        return min(max(self._current_t(), 0), max(max_t - 1, 0))

    def _correction_frame_indices(self, layer) -> list[int]:
        """Return frame indices based on the correction scope combo."""
        if layer.data.ndim < 3:
            return [0]
        if self.correction_scope_combo.currentText() == "All frames":
            return list(range(int(layer.data.shape[0])))
        return [self._current_t()]

    # ------------------------------------------------------------------ #
    # Load labels                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _broadcast_ref(image, shape):
        if image is None:
            return None
        if image.ndim == 2 and len(shape) >= 3:
            return np.broadcast_to(image[np.newaxis], (shape[0],) + image.shape).copy()
        return image

    def _show_layer(self, name, data, kwargs, adder):
        if name in self.viewer.layers:
            try:
                self.viewer.layers[name].data = data
            except Exception:
                self.viewer.layers.remove(self.viewer.layers[name])
                adder(data, name=name, **kwargs)
        else:
            adder(data, name=name, **kwargs)

    def _on_load_labels(self) -> None:
        lp = self._cell_labels_path()
        if lp is None or not lp.exists():
            self._correction_status("No cell labels file found."); return
        self._correction_status("Loading...")
        czp, nzp = self._cell_foreground_path(), self._nuc_foreground_path()

        @_thread_worker(connect={
            "returned": self._on_labels_loaded,
            "errored": lambda e: self._correction_status(f"Error: {e}"),
        })
        def _w():
            labels = read_full_tracked_stack(lp)
            cz = np.asarray(tifffile.imread(str(czp)), dtype=np.float32) if czp and czp.exists() else None
            nz = np.asarray(tifffile.imread(str(nzp)), dtype=np.float32) if nzp and nzp.exists() else None
            return labels, cz, nz
        _w()

    def _apply_loaded_layers(self, labels, cz, nz) -> None:
        # Add the reference images first so the labels layer ends up at the
        # top of the layer stack (otherwise the z-avg images render over it).
        for img, name, cmap in (
            (self._broadcast_ref(cz, labels.shape), _CELL_ZAVG_LAYER, "gray"),
            (self._broadcast_ref(nz, labels.shape), _NUC_ZAVG_LAYER, "I Purple"),
        ):
            if img is None:
                continue
            if name in self.viewer.layers:
                self.viewer.layers[name].data = img
                self.viewer.layers[name].colormap = cmap
                self.viewer.layers[name].blending = "minimum"
            else:
                self.viewer.add_image(img, name=name, colormap=cmap, blending="minimum")
            self._correction_owned_layers.add(name)
        self._show_layer(_TRACKED_CELL_LAYER, labels, {}, self.viewer.add_labels)
        self._correction_owned_layers.add(_TRACKED_CELL_LAYER)

    def _on_labels_loaded(self, result) -> None:
        labels, cz, nz = result
        self._apply_loaded_layers(labels, cz, nz)
        self._correction_status(f"Loaded cell label stack {labels.shape}.")
        self.correction_widget.activate_layer(self.viewer.layers[_TRACKED_CELL_LAYER])

    # ------------------------------------------------------------------ #
    # Save labels                                                          #
    # ------------------------------------------------------------------ #

    def _on_save_labels(self) -> None:
        lp = self._cell_labels_path()
        if lp is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No labels layer."); return
        data = np.asarray(layer.data)
        if data.ndim != 3:
            self._correction_status("Labels not 3D."); return
        lp.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(lp), data.astype(np.uint32, copy=False), compression="zlib")
        self._files_widget_refresh(self._pos_dir)
        self._correction_status(f"Saved {data.shape[0]} frames → {lp.name}.")

    # ------------------------------------------------------------------ #
    # Fill Holes                                                           #
    # ------------------------------------------------------------------ #

    def _on_fill_holes(self) -> None:
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No cell labels loaded."); return
        radius = int(self.hole_radius_spin.value())
        frames = self._correction_frame_indices(layer)

        changed_frames = 0
        changed_pixels = 0
        for t in frames:
            seg2d = self.correction_widget._frame_view(layer, t)
            before = seg2d.copy()
            result = fill_label_holes(seg2d, radius=radius)
            np.copyto(seg2d, result)
            diff = int(np.sum(before != seg2d))
            if diff:
                changed_frames += 1
                changed_pixels += diff
                self.correction_widget._record_history(layer, t, before)

        if changed_pixels:
            layer.refresh()
            if self.correction_widget._selected_label:
                t_now = self._current_t()
                self.correction_widget._update_highlight(
                    t_now, self.correction_widget._selected_label,
                )
            self._correction_status(
                f"Filled holes in {changed_frames} frame(s), "
                f"{changed_pixels} px changed. Unsaved."
            )
        else:
            self._correction_status("No interior holes found.")

    # ------------------------------------------------------------------ #
    # Fix Semi Holes                                                       #
    # ------------------------------------------------------------------ #

    def _on_fix_semiholes(self) -> None:
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No cell labels loaded."); return
        radius = int(self.hole_radius_spin.value())
        max_opening = int(self.semihole_opening_spin.value())
        frames = self._correction_frame_indices(layer)

        changed_frames = 0
        changed_pixels = 0
        for t in frames:
            seg2d = self.correction_widget._frame_view(layer, t)
            before = seg2d.copy()
            result = fix_label_semiholes(
                seg2d, radius=radius, max_opening=max_opening,
            )
            np.copyto(seg2d, result)
            diff = int(np.sum(before != seg2d))
            if diff:
                changed_frames += 1
                changed_pixels += diff
                self.correction_widget._record_history(layer, t, before)

        if changed_pixels:
            layer.refresh()
            if self.correction_widget._selected_label:
                t_now = self._current_t()
                self.correction_widget._update_highlight(
                    t_now, self.correction_widget._selected_label,
                )
            self._correction_status(
                f"Fixed semiholes in {changed_frames} frame(s), "
                f"{changed_pixels} px changed. Unsaved."
            )
        else:
            self._correction_status("No semiholes found.")

    # ------------------------------------------------------------------ #
    # Clean Up                                                             #
    # ------------------------------------------------------------------ #

    def _get_nuclear_labels(self) -> np.ndarray | None:
        """Try viewer layer first, then fall back to disk."""
        if "Tracked: Nucleus" in self.viewer.layers:
            return np.asarray(self.viewer.layers["Tracked: Nucleus"].data)
        nuc_path = self._nuc_labels_path()
        if nuc_path is not None and nuc_path.exists():
            return np.asarray(tifffile.imread(str(nuc_path)))
        return None

    def _on_cleanup(self) -> None:
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No cell labels loaded."); return
        nuc_data = self._get_nuclear_labels()
        if nuc_data is None:
            self._correction_status(
                "Nuclear labels not found (viewer or disk)."
            ); return

        cell_data = np.asarray(layer.data).copy()

        try:
            stats = cleanup_movie(cell_data, nuc_data)
        except ValueError as exc:
            self._correction_status(str(exc)); return

        layer.data = cell_data

        total = (
            stats["fragments_cleared"]
            + stats["cells_relabeled"]
            + stats["orphans_removed"]
        )
        if total:
            if self.correction_widget._selected_label:
                t_now = self._current_t()
                self.correction_widget._update_highlight(
                    t_now, self.correction_widget._selected_label,
                )
            self._correction_status(
                f"Cleanup: {stats['fragments_cleared']} fragment px, "
                f"{stats['cells_relabeled']} relabeled, "
                f"{stats['orphans_removed']} orphans removed. "
                f"No undo — save or reload to revert. Unsaved."
            )
        else:
            self._correction_status("Cleanup: nothing to change.")

    # ------------------------------------------------------------------ #
    # Expand Cell                                                          #
    # ------------------------------------------------------------------ #

    def _foreground_for_expand(self) -> np.ndarray | None:
        if _CELL_FOREGROUND_LAYER in self.viewer.layers:
            return np.asarray(self.viewer.layers[_CELL_FOREGROUND_LAYER].data)
        fp = self._foreground_path()
        if fp is None or not fp.exists():
            return None
        fg = np.asarray(tifffile.imread(str(fp)))
        self._show_layer(_CELL_FOREGROUND_LAYER, fg, {}, self.viewer.add_labels)
        self._correction_owned_layers.add(_CELL_FOREGROUND_LAYER)
        return fg

    def _on_expand_cell(self) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        layer = self._correction_tracked_layer()
        if layer is None:
            self._correction_status("No labels loaded."); return
        if self.correction_widget._layer is not layer:
            self._correction_status("Labels not active for correction."); return
        lid = int(self.correction_widget._selected_label)
        if lid == 0:
            self._correction_status("No cell selected."); return
        labels = np.asarray(layer.data)
        if labels.ndim < 3:
            self._correction_status("Labels not 3D."); return
        t = self._current_time_index(labels.shape[0])
        seg2d = self.correction_widget._frame_view(layer, t)
        if not np.any(seg2d == lid):
            self._correction_status(f"Cell {lid} absent at t={t}."); return

        fg = self._foreground_for_expand()
        if fg is None:
            self._correction_status("Foreground mask not found."); return
        if fg.shape != labels.shape:
            self._correction_status("Foreground shape mismatch."); return
        fg2d = fg[t]
        while fg2d.ndim > 2:
            if fg2d.shape[0] != 1:
                self._correction_status("Foreground frame shape unsupported."); return
            fg2d = fg2d[0]

        before = seg2d.copy()
        try:
            added = expand_label_to_foreground(
                seg2d, fg2d, lid, max_distance=int(self.expand_max_px_spin.value()),
            )
        except ValueError as exc:
            self._correction_status(str(exc)); return
        if added == 0:
            if not bool(np.any((fg2d > 0) & (before == lid))):
                self._correction_status(f"Cell {lid} doesn't touch foreground at t={t}.")
            else:
                self._correction_status(f"No expansion for cell {lid} at t={t}.")
            return
        self.correction_widget._record_history(layer, t, before)
        layer.refresh()
        self.correction_widget._update_highlight(t, lid)
        self._correction_status(f"Expanded cell {lid} at t={t} by {added} px. Unsaved.")
