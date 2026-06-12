"""One shared parameter bar for the whole Aggregate Quantification studio.

Pixel size and frame interval are needed by **builds** (cell/nucleus shape need
µm/px; dynamics needs µm/px + s/frame) *and* by **plots** (the potential
landscape's physical axis, the density field-of-view). Field-of-view area and the
z-score shuffle count are plot-only. Rather than duplicate these across the
builder and every plot, the studio hosts **one** :class:`SharedParamsWidget`
above both areas:

* :meth:`SharedParamsWidget.stamp` writes the build overrides
  (``pixel_size_um`` / ``time_interval_s``) onto catalogue records before a build,
  so a position whose metadata can't auto-resolve still builds.
* :meth:`SharedParamsWidget.plot_params` packages the plot-time tuning as a
  :class:`~cellflow.napari.aggregate_quantification.plots.PlotParams` for the Plot
  area.

Every field is "auto" by default (blank → per-position resolution); the
``changed`` signal lets the studio refresh build availability as values change.
"""
from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari.aggregate_quantification.plots import PlotParams


class SharedParamsWidget(QWidget):
    """The studio's single set of build-and-plot parameter fields."""

    #: Emitted on any field edit so the studio can re-gate builds / refresh.
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        col = QVBoxLayout(self)
        col.setContentsMargins(2, 2, 2, 2)
        col.setSpacing(2)

        self._pixel_size_edit = self._field(
            col,
            "Pixel size (µm/px):",
            placeholder="auto",
            tip="µm per pixel. Used by shape / dynamics builds and physical-unit "
            "plot axes (potential landscape, density). Blank auto-resolves per "
            "position from its config / label TIFF.",
        )
        self._frame_interval_edit = self._field(
            col,
            "Frame interval (s):",
            placeholder="auto",
            tip="Seconds per frame. Used by dynamics builds. Blank auto-resolves "
            "per position.",
        )
        self._fov_edit = self._field(
            col,
            "FOV area (mm²):",
            placeholder="auto",
            tip="Field-of-view area for the Density plot, applied to all positions. "
            "Blank uses each position's full image area.",
        )
        self._shuffles_edit = self._field(
            col,
            "Shuffles:",
            placeholder=str(PlotParams().shuffles),
            tip="Label permutations for the contact-type z-score null.",
        )

        for edit in (
            self._pixel_size_edit,
            self._frame_interval_edit,
            self._fov_edit,
            self._shuffles_edit,
        ):
            edit.textChanged.connect(lambda *_: self.changed.emit())

    def _field(self, layout, label: str, *, placeholder: str, tip: str) -> QLineEdit:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        lbl = QLabel(label)
        lbl.setFixedWidth(130)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setToolTip(tip)
        row.addWidget(lbl)
        row.addWidget(edit, 1)
        layout.addLayout(row)
        return edit

    # ----------------------------------------------------------------- plot side
    def plot_params(self) -> PlotParams:
        """Package the plot-time fields (blank/invalid → auto / default)."""
        shuffles = _parse_int(self._shuffles_edit.text())
        return PlotParams(
            pixel_size_um=_parse_positive(self._pixel_size_edit.text()),
            fov_area_mm2=_parse_positive(self._fov_edit.text()),
            shuffles=shuffles if shuffles and shuffles > 0 else PlotParams().shuffles,
        )

    # ---------------------------------------------------------------- build side
    def stamp(self, records: list[dict]) -> list[dict]:
        """Return *records* with build overrides applied, when any are set.

        A set pixel size / frame interval is written onto each record (the keys
        :func:`position_inputs_from_record` reads as explicit overrides), so a
        build uses the shared value instead of failing to auto-resolve. Records
        are copied; the originals are untouched.
        """
        pixel = _parse_positive(self._pixel_size_edit.text())
        interval = _parse_positive(self._frame_interval_edit.text())
        if pixel is None and interval is None:
            return list(records)
        stamped: list[dict] = []
        for record in records:
            updated = dict(record)
            if pixel is not None:
                updated["pixel_size_um"] = pixel
            if interval is not None:
                updated["time_interval_s"] = interval
            stamped.append(updated)
        return stamped


def _parse_positive(text: str) -> float | None:
    """A positive float from *text*, or ``None`` when blank / invalid (→ auto)."""
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _parse_int(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None
