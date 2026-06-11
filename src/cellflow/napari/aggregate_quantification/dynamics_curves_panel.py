"""Detached panel for the dynamics *curve* views: MSD, DAC, velocity correlation.

The per-cell / per-track distributions reuse the generic
:class:`~cellflow.napari.aggregate_quantification.plot_panel.PlotPanel`. The
three **curve** outputs (one ``(lag/separation, value)`` series per position)
have no per-object tidy table, so they get this small bespoke panel instead. It
is constructed from a snapshot — a list of ``(label, CurveSet)`` — the only
quantity-specific knowledge, supplied by the caller; it imports the Qt
matplotlib backend, so callers import it lazily behind a backend probe.

Each in-scope position contributes one line; lines are coloured by their group
label (condition by default). MSD is drawn log-log with the ensemble power-law
fit (``D``, ``α``) annotated per position; the directional autocorrelation and
velocity-correlation curves are drawn on linear axes (the latter with the
``1/e`` level marked).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from qtpy.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from cellflow.napari.ui_style import status_label
from cellflow.napari.widgets import CollapsibleSection

_INV_E = float(np.exp(-1.0))

#: View id -> human label.
_VIEWS = (
    ("msd", "MSD (mean-square displacement)"),
    ("dac", "Directional autocorrelation"),
    ("corr", "Velocity correlation C(r)"),
)


@dataclass(frozen=True)
class CurveSet:
    """One position's curve data + scalar fits, for the curves panel."""

    group: str
    msd_lag_s: np.ndarray
    msd_um2: np.ndarray
    msd_D_um2_per_s: float
    msd_alpha: float
    dac_lag_s: np.ndarray
    dac: np.ndarray
    dac_persistence_time_s: float
    corr_separation_um: np.ndarray
    corr: np.ndarray


class DynamicsCurvesPanel(QWidget):
    """A dock that overlays each in-scope position's MSD / DAC / C(r) curve."""

    def __init__(self, curves: list[CurveSet], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._curves = list(curves)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self._view_combo = QComboBox()
        for view_id, label in _VIEWS:
            self._view_combo.addItem(label, view_id)
        self._view_combo.currentIndexChanged.connect(lambda _=None: self._render())
        layout.addWidget(CollapsibleSection("Curve", self._view_combo, expanded=True))

        self._canvas = FigureCanvasQTAgg(Figure(figsize=(5, 4), tight_layout=True))
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)

        self._status = QLabel(f"{len(self._curves)} position(s) in snapshot.")
        status_label(self._status, muted=True)
        layout.addWidget(self._status)

        self._render()

    @property
    def _view(self) -> str:
        return self._view_combo.currentData() or "msd"

    def _render(self) -> None:
        fig = self._canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        view = self._view
        if not self._curves:
            ax.set_title("No built positions in scope.")
            self._canvas.draw_idle()
            return
        {"msd": self._draw_msd, "dac": self._draw_dac, "corr": self._draw_corr}[view](ax)
        ax.legend(fontsize="x-small", loc="best")
        self._canvas.draw_idle()

    def _draw_msd(self, ax) -> None:
        for c in self._curves:
            lag, msd = _finite_pairs(c.msd_lag_s, c.msd_um2, positive=True)
            if lag.size:
                label = f"{c.group}  (D={c.msd_D_um2_per_s:.3g}, α={c.msd_alpha:.2f})"
                ax.plot(lag, msd, marker=".", ms=3, lw=1, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("lag τ (s)")
        ax.set_ylabel("MSD (µm²)")
        ax.set_title("Ensemble time-averaged MSD")

    def _draw_dac(self, ax) -> None:
        for c in self._curves:
            lag, dac = _finite_pairs(c.dac_lag_s, c.dac)
            if lag.size:
                label = f"{c.group}  (P={c.dac_persistence_time_s:.3g} s)"
                ax.plot(lag, dac, marker=".", ms=3, lw=1, label=label)
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_xlabel("lag τ (s)")
        ax.set_ylabel("⟨û(t)·û(t+τ)⟩")
        ax.set_title("Directional autocorrelation")

    def _draw_corr(self, ax) -> None:
        for c in self._curves:
            sep, corr = _finite_pairs(c.corr_separation_um, c.corr)
            if sep.size:
                ax.plot(sep, corr, marker=".", ms=3, lw=1, label=c.group)
        ax.axhline(_INV_E, color="0.6", lw=0.8, ls="--", label="1/e")
        ax.axhline(0.0, color="0.85", lw=0.8)
        ax.set_xlabel("separation r (µm)")
        ax.set_ylabel("C(r)")
        ax.set_title("Velocity correlation (drift-subtracted)")


def _finite_pairs(
    x: np.ndarray, y: np.ndarray, *, positive: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if positive:
        mask &= (x > 0) & (y > 0)
    return x[mask], y[mask]
