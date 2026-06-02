# src/cellflow/napari/nucleus_atom_extraction_widget.py
"""Atom Extraction section for the nucleus workflow widget (stage ①)."""
from __future__ import annotations

import logging

import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._widget_helpers import (
    dslider as _dslider,
    islider as _islider,
    tool_btn as _tool_btn,
)
from cellflow.napari.ui_style import (
    add_section_pair_row,
    section_grid,
    stage_header_action_button as _stage_header_action_button,
    stage_header_label as _stage_header_label,
)
from cellflow.napari.widgets import CollapsibleSection
from cellflow.tracking_ultrack.atoms import (
    AtomParams,
    extract_atoms_frame,
    extract_atoms_stack_with_maps,
    residual,
    write_atoms_tif,
)

logger = logging.getLogger(__name__)

_ATOM_PREFIX = "[Atoms]"
_ATOM_PREVIEW_LAYER = f"{_ATOM_PREFIX} atoms"
_ATOM_TERRITORY_LAYER = f"{_ATOM_PREFIX} territory"
_ATOM_FG_RESIDUAL_LAYER = f"{_ATOM_PREFIX} residual_foreground"
_ATOM_CONTOUR_RESIDUAL_LAYER = f"{_ATOM_PREFIX} residual_contour"
_ATOM_RIDGE_LAYER = f"{_ATOM_PREFIX} ridge"

_ATOM_MASK_OPACITY = 0.7

# The two tuning stages, each with its own layers. Each group's visibility
# checkbox flips exactly the layers it owns. The Foreground residual
# (→ territory) and the Contour residual (→ ridge → atoms watershed).
_ATOM_FG_GROUP_LAYERS = (_ATOM_FG_RESIDUAL_LAYER, _ATOM_TERRITORY_LAYER)
_ATOM_CONTOUR_GROUP_LAYERS = (
    _ATOM_CONTOUR_RESIDUAL_LAYER,
    _ATOM_RIDGE_LAYER,
    _ATOM_PREVIEW_LAYER,
)

# Fixed stack order, bottom → top: each mask sits directly above the residual
# image it is judged against. Also the add order, so napari stacks them this
# way on first creation.
_ATOM_LAYERS = (
    _ATOM_FG_RESIDUAL_LAYER,
    _ATOM_TERRITORY_LAYER,
    _ATOM_CONTOUR_RESIDUAL_LAYER,
    _ATOM_RIDGE_LAYER,
    _ATOM_PREVIEW_LAYER,
)


class NucleusAtomExtractionWidget(QWidget):
    """Qt controls for tuning atom extraction with a live preview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.header = QWidget(parent)
        header_lay = QHBoxLayout(self.header)
        header_lay.setContentsMargins(0, 0, 0, 0)
        header_lay.setSpacing(4)
        self.header_lbl = QLabel("Atom Extraction")
        _stage_header_label(self.header_lbl, "nucleus")
        self.params_btn = _tool_btn(
            "⚙", "Toggle atom extraction parameters.", checkable=True
        )
        self.params_btn.setChecked(False)
        _stage_header_action_button(self.params_btn, "nucleus")
        self.active_btn = _tool_btn(
            "◉", "Live atom preview (tune against the current frame).", checkable=True
        )
        self.active_btn.setChecked(False)
        _stage_header_action_button(self.active_btn, "nucleus")
        self.run_btn = _tool_btn(
            "▶", "Compute atoms for all frames, show them, and write atoms.tif."
        )
        _stage_header_action_button(self.run_btn, "nucleus")
        header_lay.addWidget(self.header_lbl)
        header_lay.addWidget(self.params_btn)
        header_lay.addWidget(self.active_btn)
        header_lay.addWidget(self.run_btn)
        header_lay.addStretch(1)

        self.fg_window_spin = _islider(
            3, 301, 51, tooltip="Foreground residual window (px, forced odd)."
        )
        self.fg_cutoff_spin = _dslider(
            0, 1, 0.002, 0.001, 3, "Territory threshold on the fg residual."
        )
        self.fg_strength_spin = _dslider(
            0, 1, 1.0, 0.05, 2,
            "Background-subtraction strength: 1 = full fg residual, "
            "0 = raw fg map (no flattening).",
        )
        self.contour_window_spin = _islider(
            3, 301, 51, tooltip="Contour residual window (px, forced odd)."
        )
        self.contour_floor_spin = _dslider(
            0, 1, 0.01, 0.001, 3, "Ridge noise floor on the contour residual."
        )
        self.contour_strength_spin = _dslider(
            0, 1, 1.0, 0.05, 2,
            "Background-subtraction strength: 1 = full contour residual, "
            "0 = raw contour map (no flattening).",
        )
        self.atom_min_area_spin = _islider(
            0, 5000, 100, tooltip="Atoms smaller than this merge into a neighbour."
        )

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setVisible(False)

        # Each tuning stage is a plain group whose checkbox header shows/hides
        # exactly that stage's layers. Foreground residual (→ territory):
        self.fg_visible_check = QCheckBox("Foreground")
        self.fg_visible_check.setChecked(True)
        self.fg_visible_check.setToolTip(
            "Show/hide the foreground layers (residual_foreground + territory)."
        )
        self.fg_visible_check.setStyleSheet("font-weight: bold;")
        fg_grid = section_grid()
        fg_grid.setContentsMargins(0, 0, 0, 0)
        add_section_pair_row(
            fg_grid, 0,
            "FG window:", self.fg_window_spin,
            "FG cutoff:", self.fg_cutoff_spin,
        )
        add_section_pair_row(fg_grid, 1, "FG strength:", self.fg_strength_spin)
        fg_grid_w = QWidget()
        fg_grid_w.setLayout(fg_grid)

        # Contour residual (→ ridge → atoms); atom_min_area only post-processes
        # the atoms, so it lives here too.
        self.contour_visible_check = QCheckBox("Contour")
        self.contour_visible_check.setChecked(False)
        self.contour_visible_check.setToolTip(
            "Show/hide the contour layers (residual_contour + ridge + atoms)."
        )
        self.contour_visible_check.setStyleSheet("font-weight: bold;")
        contour_grid = section_grid()
        contour_grid.setContentsMargins(0, 0, 0, 0)
        add_section_pair_row(
            contour_grid, 0,
            "Contour window:", self.contour_window_spin,
            "Contour floor:", self.contour_floor_spin,
        )
        add_section_pair_row(
            contour_grid, 1,
            "Contour strength:", self.contour_strength_spin,
            "Min area:", self.atom_min_area_spin,
        )
        contour_grid_w = QWidget()
        contour_grid_w.setLayout(contour_grid)

        inner_body = QWidget()
        inner_body_lay = QVBoxLayout(inner_body)
        inner_body_lay.setContentsMargins(0, 0, 0, 0)
        inner_body_lay.setSpacing(4)
        inner_body_lay.addWidget(self.fg_visible_check)
        inner_body_lay.addWidget(fg_grid_w)
        inner_body_lay.addWidget(self.contour_visible_check)
        inner_body_lay.addWidget(contour_grid_w)
        inner_body_lay.addWidget(self.status_lbl)

        self.section = CollapsibleSection("Atom Extraction Params", inner_body)
        self.section.set_header_visible(False)
        self.section.collapse()
        self.params_btn.toggled.connect(
            lambda checked: self.section._toggle.setChecked(checked)
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)


class NucleusAtomExtractionMixin:
    """Behavior for the Atom Extraction section.

    Host must provide: ``self.viewer``, ``self._current_t()``,
    ``self._atom_fg_path()``, ``self._atom_contour_path()``,
    ``self._atom_output_path()``.
    """

    def _init_atom_extraction_state(self) -> None:
        self._atom_preview_active = False
        # Image layers whose contrast limits have not yet been auto-set. We seed
        # contrast once, on the first real data a freshly created layer receives,
        # then leave it alone so the user's manual contrast survives refreshes.
        self._atom_image_needs_autocontrast: set[str] = set()
        # A compute is in flight (None when idle); rapid edits while one runs set
        # _atom_preview_pending so exactly one fresh pass fires when it returns.
        self._atom_preview_worker = None
        self._atom_preview_pending = False
        self._atom_refresh_timer = QTimer(self)
        self._atom_refresh_timer.setSingleShot(True)
        self._atom_refresh_timer.setInterval(150)
        self._atom_refresh_timer.timeout.connect(self._refresh_atom_preview)

    def _alias_atom_extraction_controls(self) -> None:
        w = self.atom_extraction_widget
        for spin in (w.fg_window_spin, w.fg_cutoff_spin, w.fg_strength_spin,
                     w.contour_window_spin, w.contour_floor_spin,
                     w.contour_strength_spin, w.atom_min_area_spin):
            spin.valueChanged.connect(self._on_atom_param_changed)
        w.active_btn.toggled.connect(self._on_atom_activate)
        w.run_btn.clicked.connect(self._run_atom_extraction)
        w.fg_visible_check.toggled.connect(
            lambda checked: self._set_atom_group_visible(_ATOM_FG_GROUP_LAYERS, checked)
        )
        w.contour_visible_check.toggled.connect(
            lambda checked: self._set_atom_group_visible(_ATOM_CONTOUR_GROUP_LAYERS, checked)
        )

    def _atom_params(self) -> AtomParams:
        w = self.atom_extraction_widget
        return AtomParams(
            fg_window=int(w.fg_window_spin.value()),
            fg_cutoff=float(w.fg_cutoff_spin.value()),
            fg_strength=float(w.fg_strength_spin.value()),
            contour_window=int(w.contour_window_spin.value()),
            contour_floor=float(w.contour_floor_spin.value()),
            contour_strength=float(w.contour_strength_spin.value()),
            atom_min_area=int(w.atom_min_area_spin.value()),
        )

    def _set_atom_status(self, msg: str) -> None:
        lbl = self.atom_extraction_widget.status_lbl
        lbl.setText(msg)
        lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    def _on_atom_param_changed(self, *_args) -> None:
        if self._atom_preview_active:
            self._atom_refresh_timer.start()

    def _on_atom_activate(self, checked: bool) -> None:
        self._atom_preview_active = bool(checked)
        if checked:
            # _refresh_atom_preview creates the layers synchronously (before the
            # worker starts), so the default visibility can be applied right away.
            self._refresh_atom_preview()
            self._apply_atom_default_visibility()
        else:
            self._atom_preview_pending = False
            for name in _ATOM_LAYERS:
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)
            self._atom_image_needs_autocontrast.clear()
            self._set_atom_status("")

    def _set_atom_group_visible(self, names, visible: bool) -> None:
        """Show/hide a tuning stage's layers together."""
        for name in names:
            if name in self.viewer.layers:
                self.viewer.layers[name].visible = bool(visible)

    def _apply_atom_default_visibility(self) -> None:
        """Default on activation/run: Foreground pair visible, Contour group hidden.

        Sets the checkboxes to match and applies the visibility directly (a
        ``setChecked`` to an unchanged state emits no signal, so we must apply it
        explicitly rather than relying on the toggle handler).
        """
        w = self.atom_extraction_widget
        w.fg_visible_check.setChecked(True)
        w.contour_visible_check.setChecked(False)
        self._set_atom_group_visible(_ATOM_FG_GROUP_LAYERS, True)
        self._set_atom_group_visible(_ATOM_CONTOUR_GROUP_LAYERS, False)

    def _read_frame(self, path, t: int) -> np.ndarray:
        return np.asarray(tifffile.imread(str(path), key=t), dtype=np.float32)

    def _atom_map_shape(self):
        """(T, Y, X) of the foreground map, from the TIFF header (no pixel load)."""
        fg_path = self._atom_fg_path()
        if fg_path is None:
            return None
        with tifffile.TiffFile(str(fg_path)) as tf:
            n_frames = len(tf.pages)
            y, x = tf.pages[0].shape[-2], tf.pages[0].shape[-1]
        return int(n_frames), int(y), int(x)

    def _refresh_atom_preview(self):
        """Recompute the current frame's preview off the GUI thread.

        The residual + watershed pass is too heavy to run inline — doing so froze
        the viewer on every slider tick. Instead we hand it to a ``thread_worker``
        and paint the result back on the main thread. While a pass is in flight,
        further edits just arm ``_atom_preview_pending`` so one fresh pass (with
        the latest params/frame) fires when the current one returns, coalescing a
        burst of slider moves into the minimum number of computes.

        The five preview layers are full ``(T, Y, X)`` stacks sized from the input
        maps and painted one frame at a time. Carrying the time axis gives the
        viewer a frame slider even when no movie is open — otherwise ``current_step``
        has no temporal entry and the preview is stuck on (and mislabels) frame 0.

        Returns the started worker (or ``None``) so callers/tests can await it.
        """
        if not self._atom_preview_active:
            return None
        fg_path = self._atom_fg_path()
        contour_path = self._atom_contour_path()
        if fg_path is None or contour_path is None:
            self._set_atom_status("Foreground/contour maps not found.")
            return None
        shape = self._atom_map_shape()
        if shape is None:
            self._set_atom_status("Foreground/contour maps not found.")
            return None
        self._ensure_atom_preview_stacks(shape)
        if self._atom_preview_worker is not None:
            self._atom_preview_pending = True
            return self._atom_preview_worker
        params = self._atom_params()
        n_frames = shape[0]
        t = max(0, min(self._current_t(), n_frames - 1))
        self._set_atom_status(f"Computing atoms for frame {t}…")

        @thread_worker(connect={
            "returned": self._on_atom_preview_done,
            "errored": self._on_atom_preview_error,
        })
        def _worker():
            fg = self._read_frame(fg_path, t)
            contour = self._read_frame(contour_path, t)
            residual_foreground = residual(fg, params.fg_window, params.fg_strength)
            territory = residual_foreground > params.fg_cutoff
            residual_contour = residual(contour, params.contour_window, params.contour_strength)
            atoms, ridge = extract_atoms_frame(
                residual_contour, territory,
                params.contour_floor, params.atom_min_area,
            )
            return (t, atoms, territory.astype(np.uint8),
                    residual_foreground, residual_contour, ridge)

        self._atom_preview_worker = _worker()
        return self._atom_preview_worker

    def _on_atom_preview_done(self, result) -> None:
        self._atom_preview_worker = None
        t, atoms, territory, residual_foreground, residual_contour, ridge = result
        if self._atom_preview_active:
            self._fill_atom_image_slice(_ATOM_FG_RESIDUAL_LAYER, t, residual_foreground)
            self._fill_atom_labels_slice(_ATOM_TERRITORY_LAYER, t, territory)
            self._fill_atom_image_slice(_ATOM_CONTOUR_RESIDUAL_LAYER, t, residual_contour)
            self._fill_atom_labels_slice(_ATOM_RIDGE_LAYER, t, ridge)
            self._fill_atom_labels_slice(_ATOM_PREVIEW_LAYER, t, atoms)
            self._set_atom_status(f"Frame {t}: {int(atoms.max())} atoms.")
        if self._atom_preview_pending and self._atom_preview_active:
            self._atom_preview_pending = False
            self._refresh_atom_preview()
        else:
            self._atom_preview_pending = False

    def _on_atom_preview_error(self, exc: Exception) -> None:
        self._atom_preview_worker = None
        self._atom_preview_pending = False
        self._set_atom_status(f"Atom preview failed: {exc}")
        logger.exception("Atom preview worker error", exc_info=exc)

    # ── preview stacks (one zero-filled (T, Y, X) layer per map) ─────────────

    def _ensure_atom_preview_stacks(self, shape) -> None:
        # Created bottom → top so napari stacks each mask directly above the
        # residual image it is judged against (§ fixed stack order).
        self._ensure_atom_image_stack(_ATOM_FG_RESIDUAL_LAYER, shape)
        self._ensure_atom_labels_stack(_ATOM_TERRITORY_LAYER, shape)
        self._ensure_atom_image_stack(_ATOM_CONTOUR_RESIDUAL_LAYER, shape)
        self._ensure_atom_labels_stack(_ATOM_RIDGE_LAYER, shape)
        self._ensure_atom_labels_stack(_ATOM_PREVIEW_LAYER, shape)

    def _ensure_atom_labels_stack(self, name: str, shape) -> None:
        from napari.layers import Labels
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if isinstance(layer, Labels) and tuple(layer.data.shape) == tuple(shape):
                return
            was_visible = layer.visible
            self.viewer.layers.remove(name)
        else:
            was_visible = True
        new_layer = self.viewer.add_labels(
            np.zeros(shape, dtype=np.int32), name=name, opacity=_ATOM_MASK_OPACITY
        )
        new_layer.visible = was_visible

    def _ensure_atom_image_stack(self, name: str, shape) -> None:
        if name in self.viewer.layers:
            layer = self.viewer.layers[name]
            if tuple(layer.data.shape) == tuple(shape):
                return
            was_visible = layer.visible
            self.viewer.layers.remove(name)
        else:
            was_visible = True
        new_layer = self.viewer.add_image(
            np.zeros(shape, dtype=np.float32), name=name,
            colormap="magma", blending="additive",
        )
        new_layer.visible = was_visible
        # A just-created image layer gets its contrast auto-set on the first real
        # data it receives; afterwards the user's contrast is left untouched.
        self._atom_image_needs_autocontrast.add(name)

    def _fill_atom_labels_slice(self, name: str, t: int, frame: np.ndarray) -> None:
        if name not in self.viewer.layers:
            return
        layer = self.viewer.layers[name]
        if layer.data.ndim != 3 or not 0 <= t < layer.data.shape[0]:
            return
        layer.data[t] = frame.astype(layer.data.dtype, copy=False)
        layer.refresh()

    def _fill_atom_image_slice(self, name: str, t: int, frame: np.ndarray) -> None:
        if name not in self.viewer.layers:
            return
        layer = self.viewer.layers[name]
        if layer.data.ndim != 3 or not 0 <= t < layer.data.shape[0]:
            return
        layer.data[t] = frame.astype(layer.data.dtype, copy=False)
        self._maybe_autocontrast(name, layer, frame)
        layer.refresh()

    def _maybe_autocontrast(self, name: str, layer, data: np.ndarray) -> None:
        """Seed a freshly created image layer's contrast once, from its first
        real data — then never again, so manual contrast survives refreshes."""
        if name not in self._atom_image_needs_autocontrast:
            return
        lo, hi = float(data.min()), float(data.max())
        if hi > lo:
            layer.contrast_limits = (lo, hi)
            self._atom_image_needs_autocontrast.discard(name)

    def _run_atom_extraction(self) -> None:
        fg_path = self._atom_fg_path()
        contour_path = self._atom_contour_path()
        out_path = self._atom_output_path()
        if fg_path is None or contour_path is None or out_path is None:
            self._set_atom_status("Foreground/contour maps not found.")
            return
        params = self._atom_params()
        self._set_atom_status("Computing atoms over all frames…")
        try:
            fg = np.asarray(tifffile.imread(str(fg_path)), dtype=np.float32)
            contour = np.asarray(tifffile.imread(str(contour_path)), dtype=np.float32)
            atoms, territory, residual_foreground, residual_contour, ridge = (
                extract_atoms_stack_with_maps(fg, contour, params)
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_atoms_tif(out_path, atoms, params)
        except Exception as exc:
            self._set_atom_status(f"Atom computation failed: {exc}")
            return
        # Reuse the preview layers, replacing each with its full (T, Y, X) stack.
        shape = atoms.shape
        self._ensure_atom_preview_stacks(shape)
        self.viewer.layers[_ATOM_PREVIEW_LAYER].data = atoms
        self.viewer.layers[_ATOM_TERRITORY_LAYER].data = territory.astype(np.int32)
        self.viewer.layers[_ATOM_RIDGE_LAYER].data = ridge.astype(np.int32)
        self._set_atom_image_stack(_ATOM_FG_RESIDUAL_LAYER, residual_foreground)
        self._set_atom_image_stack(_ATOM_CONTOUR_RESIDUAL_LAYER, residual_contour)
        self._apply_atom_default_visibility()
        self._set_atom_status(f"Wrote {atoms.shape[0]} frames → atoms.tif.")

    def _set_atom_image_stack(self, name: str, data: np.ndarray) -> None:
        layer = self.viewer.layers[name]
        data = np.asarray(data, dtype=np.float32)
        layer.data = data
        self._maybe_autocontrast(name, layer, data)
        layer.refresh()
