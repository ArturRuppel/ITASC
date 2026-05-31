# src/cellflow/napari/nucleus_atom_extraction_widget.py
"""Atom Extraction section for the nucleus workflow widget (stage ①)."""
from __future__ import annotations

import logging

import numpy as np
import tifffile
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._widget_helpers import (
    dslider as _dslider,
    heading as _heading,
    islider as _islider,
    tool_btn as _tool_btn,
)
from cellflow.napari.ui_style import (
    add_section_header,
    add_section_pair_row,
    section_grid,
    stage_header_action_button as _stage_header_action_button,
    stage_header_label as _stage_header_label,
)
from cellflow.napari.widgets import CollapsibleSection
from cellflow.tracking_ultrack.atoms import (
    AtomParams,
    extract_atoms_frame,
    extract_atoms_stack,
    residual,
    write_atoms_tif,
)

logger = logging.getLogger(__name__)

_ATOM_PREFIX = "[Atoms]"
_ATOM_PREVIEW_LAYER = f"{_ATOM_PREFIX} preview"
_ATOM_TERRITORY_LAYER = f"{_ATOM_PREFIX} territory"
_ATOM_RESIDUAL_LAYER = f"{_ATOM_PREFIX} residual_contour"


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
        self.active_btn = _tool_btn(
            "⏻", "Activate atom extraction preview.", checkable=True
        )
        self.active_btn.setChecked(False)
        _stage_header_action_button(self.active_btn, "nucleus")
        header_lay.addWidget(self.header_lbl)
        header_lay.addWidget(self.active_btn)
        header_lay.addStretch(1)

        inner = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(0, 0, 0, 0)
        inner.setLayout(grid)

        self.fg_window_spin = _islider(
            3, 301, 51, tooltip="Foreground residual window (px, forced odd)."
        )
        self.fg_cutoff_spin = _dslider(
            0, 1, 0.002, 0.001, 3, "Territory threshold on the fg residual."
        )
        self.contour_window_spin = _islider(
            3, 301, 51, tooltip="Contour residual window (px, forced odd)."
        )
        self.contour_floor_spin = _dslider(
            0, 1, 0.01, 0.001, 3, "Ridge noise floor on the contour residual."
        )
        self.atom_min_area_spin = _islider(
            0, 5000, 100, tooltip="Atoms smaller than this merge into a neighbour."
        )

        self.territory_overlay_check = QCheckBox("Territory overlay")
        self.residual_overlay_check = QCheckBox("Residual-contour overlay")
        self.compute_btn = QPushButton("Compute atoms (full stack)")
        self.compute_btn.setToolTip(
            "Run atom extraction over all frames and write atoms.tif."
        )
        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setVisible(False)

        row = 0
        add_section_header(grid, row, _heading("Residual")); row += 1
        add_section_pair_row(
            grid, row,
            "FG window:", self.fg_window_spin,
            "FG cutoff:", self.fg_cutoff_spin,
        ); row += 1
        add_section_pair_row(
            grid, row,
            "Contour window:", self.contour_window_spin,
            "Contour floor:", self.contour_floor_spin,
        ); row += 1
        add_section_header(grid, row, _heading("Atoms")); row += 1
        add_section_pair_row(grid, row, "Min area:", self.atom_min_area_spin); row += 1

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(inner)
        overlay_row = QWidget(self)
        overlay_lay = QHBoxLayout(overlay_row)
        overlay_lay.setContentsMargins(0, 0, 0, 0)
        overlay_lay.addWidget(self.territory_overlay_check)
        overlay_lay.addWidget(self.residual_overlay_check)
        lay.addWidget(overlay_row)
        lay.addWidget(self.compute_btn)
        lay.addWidget(self.status_lbl)

        self.section: CollapsibleSection | None = None


class NucleusAtomExtractionMixin:
    """Behavior for the Atom Extraction section.

    Host must provide: ``self.viewer``, ``self._current_t()``,
    ``self._atom_fg_path()``, ``self._atom_contour_path()``,
    ``self._atom_output_path()``.
    """

    def _init_atom_extraction_state(self) -> None:
        self._atom_preview_active = False
        self._atom_refresh_timer = QTimer(self)
        self._atom_refresh_timer.setSingleShot(True)
        self._atom_refresh_timer.setInterval(150)
        self._atom_refresh_timer.timeout.connect(self._refresh_atom_preview)

    def _alias_atom_extraction_controls(self) -> None:
        w = self.atom_extraction_widget
        for spin in (w.fg_window_spin, w.fg_cutoff_spin, w.contour_window_spin,
                     w.contour_floor_spin, w.atom_min_area_spin):
            spin.valueChanged.connect(self._on_atom_param_changed)
        w.territory_overlay_check.toggled.connect(self._on_atom_param_changed)
        w.residual_overlay_check.toggled.connect(self._on_atom_param_changed)
        w.active_btn.toggled.connect(self._on_atom_activate)
        w.compute_btn.clicked.connect(self._compute_atoms_full_stack)

    def _atom_params(self) -> AtomParams:
        w = self.atom_extraction_widget
        return AtomParams(
            fg_window=int(w.fg_window_spin.value()),
            fg_cutoff=float(w.fg_cutoff_spin.value()),
            contour_window=int(w.contour_window_spin.value()),
            contour_floor=float(w.contour_floor_spin.value()),
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
            self._refresh_atom_preview()
        else:
            for name in (_ATOM_PREVIEW_LAYER, _ATOM_TERRITORY_LAYER,
                         _ATOM_RESIDUAL_LAYER):
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)
            self._set_atom_status("")

    def _read_frame(self, path, t: int) -> np.ndarray:
        return np.asarray(tifffile.imread(str(path), key=t), dtype=np.float32)

    def _refresh_atom_preview(self) -> None:
        if not self._atom_preview_active:
            return
        fg_path = self._atom_fg_path()
        contour_path = self._atom_contour_path()
        if fg_path is None or contour_path is None:
            self._set_atom_status("Foreground/contour maps not found.")
            return
        params = self._atom_params()
        t = self._current_t()
        try:
            fg = self._read_frame(fg_path, t)
            contour = self._read_frame(contour_path, t)
        except (FileNotFoundError, IndexError) as exc:
            self._set_atom_status(f"Cannot read maps for frame {t}: {exc}")
            return
        territory = residual(fg, params.fg_window) > params.fg_cutoff
        residual_contour = residual(contour, params.contour_window)
        atoms = extract_atoms_frame(
            residual_contour, territory, params.contour_floor, params.atom_min_area
        )
        self._update_atom_labels_layer(_ATOM_PREVIEW_LAYER, atoms)
        w = self.atom_extraction_widget
        self._toggle_atom_overlay(
            _ATOM_TERRITORY_LAYER, territory.astype(np.uint8),
            w.territory_overlay_check.isChecked(), is_labels=True,
        )
        self._toggle_atom_overlay(
            _ATOM_RESIDUAL_LAYER, residual_contour,
            w.residual_overlay_check.isChecked(), is_labels=False,
        )
        n_atoms = int(atoms.max())
        self._set_atom_status(f"Frame {t}: {n_atoms} atoms.")

    def _update_atom_labels_layer(self, name: str, data: np.ndarray) -> None:
        from napari.layers import Labels
        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Labels):
            self.viewer.layers[name].data = data.astype(np.int32)
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        self.viewer.add_labels(data.astype(np.int32), name=name, opacity=0.55)

    def _toggle_atom_overlay(self, name, data, visible, *, is_labels) -> None:
        if not visible:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
            return
        if name in self.viewer.layers:
            self.viewer.layers[name].data = data
            self.viewer.layers[name].visible = True
            return
        if is_labels:
            self.viewer.add_labels(data.astype(np.uint8), name=name, opacity=0.3)
        else:
            self.viewer.add_image(data, name=name, colormap="magma",
                                  blending="additive")

    def _compute_atoms_full_stack(self) -> None:
        fg_path = self._atom_fg_path()
        contour_path = self._atom_contour_path()
        out_path = self._atom_output_path()
        if fg_path is None or contour_path is None or out_path is None:
            self._set_atom_status("Foreground/contour maps not found.")
            return
        params = self._atom_params()
        self._set_atom_status("Computing atoms over all frames…")
        fg = np.asarray(tifffile.imread(str(fg_path)), dtype=np.float32)
        contour = np.asarray(tifffile.imread(str(contour_path)), dtype=np.float32)
        atoms = extract_atoms_stack(fg, contour, params)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_atoms_tif(out_path, atoms, params)
        self._set_atom_status(
            f"Wrote {atoms.shape[0]} frames to {out_path.name} "
            f"({int(atoms.max())} atoms in last frame)."
        )
