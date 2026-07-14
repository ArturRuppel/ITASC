# src/itasc/napari/nucleus_atom_extraction_widget.py
"""Atom Extraction section for the nucleus workflow widget (stage ①)."""
from __future__ import annotations

import dataclasses
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

from itasc.napari._preview_cache import FramePreviewCache
from itasc.napari._widget_helpers import (
    dslider as _dslider,
    islider as _islider,
    tool_btn as _tool_btn,
)
from itasc.napari.ui_style import (
    add_section_pair_row,
    section_grid,
    stage_header_action_button as _stage_header_action_button,
    stage_header_label as _stage_header_label,
)
from itasc.napari.ui_gate import ControlClass
from itasc.napari.widgets import CollapsibleSection
from itasc.tracking_ultrack.atoms import (
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

# The residual maps are napari Image layers; the rest are Labels.
_ATOM_IMAGE_LAYERS = (_ATOM_FG_RESIDUAL_LAYER, _ATOM_CONTOUR_RESIDUAL_LAYER)

# Single-frame compute levels, in increasing cost. The desired level is the max
# over the ticked Compute checkboxes; a cached frame is reused only when its
# stored level already covers the desired one. The contour stage's atom
# watershed runs on the foreground territory, so it implies the FG compute.
_ATOM_LEVEL_NONE = 0
_ATOM_LEVEL_FG = 1       # residual_foreground + territory (cheap)
_ATOM_LEVEL_CONTOUR = 2  # + residual_contour + ridge + atoms (heavier)


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
            3, 301, 20, tooltip="Contour residual window (px, forced odd)."
        )
        self.contour_floor_spin = _dslider(
            0, 1, 0.05, 0.001, 3, "Ridge noise floor on the contour residual."
        )
        self.contour_strength_spin = _dslider(
            0, 1, 0.16455696202531644, 0.05, 2,
            "Background-subtraction strength: 1 = full contour residual, "
            "0 = raw contour map (no flattening).",
        )
        self.atom_min_area_spin = _islider(
            0, 5000, 10, tooltip="Atoms smaller than this merge into a neighbour."
        )

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setVisible(False)

        # A "Compute:" checkbox row along the top — one box per tuning stage.
        # Each box gates whether that stage's layers are *computed at all* by the
        # live preview (not merely shown): the worker computes only the ticked
        # stages and an unticked stage's layers are removed. The contour stage's
        # atom watershed runs on the foreground territory, so ticking Contour
        # implies the (cheap) FG compute even when FG itself is unticked.
        self.fg_check = QCheckBox("Foreground")
        self.fg_check.setChecked(True)
        self.fg_check.setToolTip(
            "Compute + show the foreground layers (residual_foreground + territory)."
        )
        self.contour_check = QCheckBox("Contour")
        self.contour_check.setChecked(False)
        self.contour_check.setToolTip(
            "Compute + show the contour layers (residual_contour + ridge + atoms)."
        )
        compute_row = QWidget()
        compute_lay = QHBoxLayout(compute_row)
        compute_lay.setContentsMargins(8, 0, 4, 0)
        compute_lay.setSpacing(8)
        compute_lay.addWidget(QLabel("Compute:"))
        compute_lay.addWidget(self.fg_check)
        compute_lay.addWidget(self.contour_check)
        compute_lay.addStretch(1)

        # Foreground residual (→ territory) knobs.
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

        # Contour residual (→ ridge → atoms) knobs; atom_min_area only
        # post-processes the atoms, so it lives here too.
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
        inner_body_lay.addWidget(compute_row)
        inner_body_lay.addWidget(fg_grid_w)
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
    ``self._atom_output_path()``. Optionally ``self._files_widget`` +
    ``self._pos_dir`` — a run refreshes them so the atoms.tif status repaints.
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
        # Per-frame result cache (frame t → painted slices) keyed on the atom
        # params, so scrubbing back to a computed frame repaints instantly and
        # any param edit drops every cached frame. Mirrors the cell-segmentation
        # preview's cache; freed when the preview deactivates.
        self._atom_preview_cache = FramePreviewCache()
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
        w.fg_check.toggled.connect(
            lambda checked: self._on_atom_compute_toggled(_ATOM_FG_GROUP_LAYERS, checked)
        )
        w.contour_check.toggled.connect(
            lambda checked: self._on_atom_compute_toggled(_ATOM_CONTOUR_GROUP_LAYERS, checked)
        )

    def _register_atom_gate_controls(self) -> None:
        """Register the atom extraction section with the app-wide UI gate.

        The ◉ live atom preview is a mutually-exclusive viewer owner, shared
        across every section (correction, db browser, the cell/divergence live
        previews): turning it on must disable the others and vice versa. ▶ run
        rewrites the atom layers in the viewer, so it is blocked while any owner
        is active; ⚙ params just toggles a parameter panel and stays available.
        """
        g = self.gate
        w = self.atom_extraction_widget
        ready = self._atom_inputs_ready
        g.register_owner(
            "atom_preview",
            "atom live preview",
            exit_fn=lambda: w.active_btn.setChecked(False),
        )
        g.register(w.params_btn, ControlClass.HARMLESS)
        g.register(
            w.active_btn,
            ControlClass.VIEWER_OWNER,
            owner_token="atom_preview",
            when=ready,
        )
        g.register(w.run_btn, ControlClass.RUN_VIEWER, when=ready)

    def _atom_inputs_ready(self) -> bool:
        """True only when both input maps are set *and* present on disk.

        The live preview and ▶ run both read the foreground/contour TIFFs, so
        gating on path-not-None alone enables them for paths that don't exist
        (e.g. a stale foreground/contour pair restored from QSettings), and the
        first click then crashes in ``tifffile``. Requiring the files to exist
        keeps the controls disabled until there is something real to read.
        """
        fg = self._atom_fg_path()
        contour = self._atom_contour_path()
        return (
            fg is not None and fg.exists()
            and contour is not None and contour.exists()
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
        # The live preview owns the viewer; the gate derives cross-section
        # exclusivity (correction, db browser, other live previews) from this
        # claim.
        if checked:
            self.gate.claim_viewer("atom_preview")
            # Respect whatever the Compute checkboxes are set to — the refresh
            # creates and computes only the ticked stages' layers.
            self._refresh_atom_preview()
        else:
            self.gate.release_viewer("atom_preview")
            self._atom_preview_pending = False
            for name in _ATOM_LAYERS:
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)
            self._atom_image_needs_autocontrast.clear()
            self._atom_preview_cache.clear()
            self._set_atom_status("")

    def _on_atom_compute_toggled(self, group_layers, checked: bool) -> None:
        """Handle a Compute-checkbox toggle: drop unticked groups, refresh ticked.

        Unticking a stage removes its layers immediately. Then, while the preview
        is active, a refresh creates/paints any newly-ticked stage — computing
        only if the cached frame doesn't already cover the new level, so
        re-ticking a stage that was just computed repaints instantly.
        """
        if not checked:
            for name in group_layers:
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)
                self._atom_image_needs_autocontrast.discard(name)
        if self._atom_preview_active:
            self._refresh_atom_preview()

    def _atom_compute_groups(self):
        """(checkbox, owned layers, compute level) for each tuning stage."""
        w = self.atom_extraction_widget
        return (
            (w.fg_check, _ATOM_FG_GROUP_LAYERS, _ATOM_LEVEL_FG),
            (w.contour_check, _ATOM_CONTOUR_GROUP_LAYERS, _ATOM_LEVEL_CONTOUR),
        )

    def _desired_atom_level(self) -> int:
        """The single-frame compute level needed to fill every ticked stage."""
        level = _ATOM_LEVEL_NONE
        for check, _layers, need in self._atom_compute_groups():
            if check.isChecked():
                level = max(level, need)
        return level

    def _checked_atom_layer_names(self) -> list[str]:
        """Layer names owned by the currently-ticked Compute checkboxes."""
        names: list[str] = []
        for check, layers, _need in self._atom_compute_groups():
            if check.isChecked():
                names.extend(layers)
        return names

    def _read_frame(self, path, t: int) -> np.ndarray:
        return np.asarray(tifffile.imread(str(path), key=t), dtype=np.float32)

    def _atom_map_shape(self):
        """(T, Y, X) of the foreground map, from the TIFF header (no pixel load)."""
        fg_path = self._atom_fg_path()
        if fg_path is None or not fg_path.exists():
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
        params = self._atom_params()
        key = dataclasses.astuple(params)
        self._atom_preview_cache.sync(key)
        level = self._desired_atom_level()
        self._ensure_atom_preview_stacks(shape, self._checked_atom_layer_names())
        if level == _ATOM_LEVEL_NONE:
            self._set_atom_status("")
            return None
        n_frames = shape[0]
        t = max(0, min(self._current_t(), n_frames - 1))

        cached = self._atom_preview_cache.get(t)
        if cached is not None and cached[0] >= level:
            # Already computed to (at least) the needed level for these params —
            # instant repaint, even while a worker is busy on another frame.
            self._paint_atom_preview(t, cached[1])
            return None

        if self._atom_preview_worker is not None:
            self._atom_preview_pending = True
            return self._atom_preview_worker
        self._set_atom_status(f"Computing atoms for frame {t}…")

        @thread_worker(connect={
            "returned": self._on_atom_preview_done,
            "errored": self._on_atom_preview_error,
        })
        def _worker():
            # The FG residual + territory is always cheap and the atom watershed
            # needs the territory as its mask, so it runs at every level; the
            # contour stage (residual + ridge + atoms) runs only when reached.
            fg = self._read_frame(fg_path, t)
            residual_foreground = residual(fg, params.fg_window, params.fg_strength)
            territory = residual_foreground > params.fg_cutoff
            atoms = residual_contour = ridge = None
            if level >= _ATOM_LEVEL_CONTOUR:
                contour = self._read_frame(contour_path, t)
                residual_contour = residual(
                    contour, params.contour_window, params.contour_strength
                )
                atoms, ridge = extract_atoms_frame(
                    residual_contour, territory,
                    params.contour_floor, params.atom_min_area,
                )
            slices = (atoms, territory.astype(np.uint8),
                      residual_foreground, residual_contour, ridge)
            return key, t, level, slices

        self._atom_preview_worker = _worker()
        return self._atom_preview_worker

    def _on_atom_preview_done(self, result) -> None:
        self._atom_preview_worker = None
        key, t, level, slices = result
        self._atom_preview_cache.put(key, t, (level, slices))
        if self._atom_preview_active:
            self._paint_atom_preview(t, slices)
        if self._atom_preview_pending and self._atom_preview_active:
            self._atom_preview_pending = False
            self._refresh_atom_preview()
        else:
            self._atom_preview_pending = False

    def _paint_atom_preview(self, t: int, slices) -> None:
        """Fill frame ``t``'s slice for every stage the result supplies.

        The contour stage's slices are ``None`` when only the foreground level
        was computed; those are skipped (their layers don't exist either). Each
        ``_fill_*`` is itself a no-op for an absent layer, so painting a value
        whose stage is computed-but-unticked is harmless.
        """
        atoms, territory, residual_foreground, residual_contour, ridge = slices
        if residual_foreground is not None:
            self._fill_atom_image_slice(_ATOM_FG_RESIDUAL_LAYER, t, residual_foreground)
        if territory is not None:
            self._fill_atom_labels_slice(_ATOM_TERRITORY_LAYER, t, territory)
        if residual_contour is not None:
            self._fill_atom_image_slice(_ATOM_CONTOUR_RESIDUAL_LAYER, t, residual_contour)
        if ridge is not None:
            self._fill_atom_labels_slice(_ATOM_RIDGE_LAYER, t, ridge)
        if atoms is not None:
            self._fill_atom_labels_slice(_ATOM_PREVIEW_LAYER, t, atoms)
            self._set_atom_status(f"Frame {t}: {int(atoms.max())} atoms.")
        else:
            self._set_atom_status(f"Frame {t}: territory only.")

    def _on_atom_preview_error(self, exc: Exception) -> None:
        self._atom_preview_worker = None
        self._atom_preview_pending = False
        self._set_atom_status(f"Atom preview failed: {exc}")
        logger.exception("Atom preview worker error", exc_info=exc)

    # ── preview stacks (one zero-filled (T, Y, X) layer per map) ─────────────

    def _ensure_atom_preview_stacks(self, shape, names) -> None:
        # Walk the fixed bottom → top order and create only the requested layers,
        # so napari stacks each mask directly above the residual image it is
        # judged against (§ fixed stack order) even when a stage is unticked.
        wanted = set(names)
        for name in _ATOM_LAYERS:
            if name not in wanted:
                continue
            if name in _ATOM_IMAGE_LAYERS:
                self._ensure_atom_image_stack(name, shape)
            else:
                self._ensure_atom_labels_stack(name, shape)

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
        # The run computed every stage, so show them all: tick both Compute
        # boxes (so their layers exist) and replace each with its full (T, Y, X)
        # stack.
        w = self.atom_extraction_widget
        w.fg_check.setChecked(True)
        w.contour_check.setChecked(True)
        shape = atoms.shape
        self._ensure_atom_preview_stacks(shape, _ATOM_LAYERS)
        self.viewer.layers[_ATOM_PREVIEW_LAYER].data = atoms
        self.viewer.layers[_ATOM_TERRITORY_LAYER].data = territory.astype(np.int32)
        self.viewer.layers[_ATOM_RIDGE_LAYER].data = ridge.astype(np.int32)
        self._set_atom_image_stack(_ATOM_FG_RESIDUAL_LAYER, residual_foreground)
        self._set_atom_image_stack(_ATOM_CONTOUR_RESIDUAL_LAYER, residual_contour)
        self._set_atom_status(f"Wrote {atoms.shape[0]} frames → atoms.tif.")
        # atoms.tif is a tracked intermediate; refresh the host's Pipeline Files
        # so the section dot + catalog rail pick it up (see the host's
        # PipelineFilesWidget). Guarded because the mixin's host contract is loose.
        files_widget = getattr(self, "_files_widget", None)
        if files_widget is not None:
            files_widget.refresh(getattr(self, "_pos_dir", None))

    def _set_atom_image_stack(self, name: str, data: np.ndarray) -> None:
        layer = self.viewer.layers[name]
        data = np.asarray(data, dtype=np.float32)
        layer.data = data
        self._maybe_autocontrast(name, layer, data)
        layer.refresh()
