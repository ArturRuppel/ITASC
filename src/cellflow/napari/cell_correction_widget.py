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
    QPushButton,
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
)
from cellflow.napari.correction_widget import CorrectionWidget
from cellflow.napari.ui_style import (
    add_block_pair_row,
    block_grid,
    compact_spinbox,
)
from cellflow.napari.widgets import CollapsibleSection
from cellflow.napari._widget_helpers import ispin as _ispin

logger = logging.getLogger(__name__)

_TRACKED_CELL_LAYER = "Tracked: Cell"
_CELL_ZAVG_LAYER = "Cell z-avg"
_NUC_ZAVG_LAYER = "Nucleus z-avg"
_CELL_FOREGROUND_LAYER = "Foreground Mask: Cell"


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

        self.active_btn = QPushButton("Activate Correction")
        self.active_btn.setCheckable(True)
        self.active_btn.setToolTip(
            "Activate correction mode and show correction controls."
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
            title_role="params",
            title_level=2,
        )
        group_lay.addWidget(self.correction_params_section)

        # ── Inline CorrectionWidget ───────────────────────────────────
        self.correction_widget = CorrectionWidget(
            self.viewer,
            show_activate_btn=False,
            show_shortcuts=False,
            inspector_first=True,
            spotlight=False,
            show_cleanup=False,
        )

        self.correction_shortcuts_section = CollapsibleSection(
            "Correction Shortcuts",
            self.correction_widget.build_shortcuts_widget(),
            expanded=False,
            title_role="actions",
            title_level=2,
        )
        group_lay.addWidget(self.correction_shortcuts_section)
        group_lay.addWidget(self.correction_widget)

        self.correction_status_lbl = _make_status()
        group_lay.addWidget(self.correction_status_lbl)

        group_lay.addLayout(_button_grid(
            (self.load_labels_btn, self.save_labels_btn),
            (self.fill_holes_btn, self.fix_semiholes_btn),
            (self.cleanup_btn, self.expand_cell_btn),
        ))
        group_lay.addWidget(self.correction_widget._attrib_lbl)

        self.section = CollapsibleSection(
            "Correction",
            inner,
            expanded=False,
            title_role="stage",
            title_level=1,
        )
        self.section._toggle.setVisible(False)
        self.section._toggle.setEnabled(False)
        root.addWidget(self.active_btn)
        root.addWidget(self.section)

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

    def _on_active_button_toggled(self, active: bool) -> None:
        self.section.expand() if active else self.section.collapse()

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

    def _cell_prob_zavg_path(self) -> Path | None:
        return self._p("1_cellpose", "cell_prob_zavg.tif")

    def _nuc_prob_zavg_path(self) -> Path | None:
        return self._p("1_cellpose", "nucleus_prob_zavg.tif")

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
        czp, nzp = self._cell_prob_zavg_path(), self._nuc_prob_zavg_path()

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

    def _on_labels_loaded(self, result) -> None:
        labels, cz, nz = result
        self._show_layer(_TRACKED_CELL_LAYER, labels, {}, self.viewer.add_labels)
        for img, name, cmap in (
            (self._broadcast_ref(cz, labels.shape), _CELL_ZAVG_LAYER, "gray"),
            (self._broadcast_ref(nz, labels.shape), _NUC_ZAVG_LAYER, "I Orange"),
        ):
            if img is None:
                continue
            if name in self.viewer.layers:
                self.viewer.layers[name].data = img
                self.viewer.layers[name].colormap = cmap
                self.viewer.layers[name].blending = "minimum"
            else:
                self.viewer.add_image(img, name=name, colormap=cmap, blending="minimum")
        self._correction_status(f"Loaded cell label stack {labels.shape}.")
        self.correction_widget.activate_layer(self.viewer.layers[_TRACKED_CELL_LAYER])

    # ------------------------------------------------------------------ #
    # Save labels                                                          #
    # ------------------------------------------------------------------ #

    def _on_save_labels(self) -> None:
        lp = self._cell_labels_path()
        if lp is None:
            self._correction_status("No project open."); return
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._correction_status("No labels layer."); return
        data = np.asarray(self.viewer.layers[_TRACKED_CELL_LAYER].data)
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
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._correction_status("No cell labels loaded."); return
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
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
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._correction_status("No cell labels loaded."); return
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
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
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._correction_status("No cell labels loaded."); return
        nuc_data = self._get_nuclear_labels()
        if nuc_data is None:
            self._correction_status(
                "Nuclear labels not found (viewer or disk)."
            ); return

        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
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
        return fg

    def _on_expand_cell(self) -> None:
        if self._pos_dir is None:
            self._correction_status("No project open."); return
        if _TRACKED_CELL_LAYER not in self.viewer.layers:
            self._correction_status("No labels loaded."); return
        layer = self.viewer.layers[_TRACKED_CELL_LAYER]
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
