"""One shared parameter bar for the whole Aggregate Quantification studio.

Pixel size and frame interval are needed by **builds** (cell/nucleus shape need
µm/px; dynamics needs µm/px + s/frame). Field-of-view area and the z-score
shuffle count are build knobs for specific metrics (density, contact-type
z-score). Rather than duplicate these across every build, the studio hosts
**one** :class:`SharedParamsWidget` above the build area:

* :meth:`SharedParamsWidget.stamp` writes the global ``pixel_size_um`` /
  ``time_interval_s`` onto every catalogue record before a build, so the build
  reads them off ``PositionInputs``. These are global-only — there is no
  per-position auto-resolution; a metric needing one is unbuildable until it is set.
* :meth:`SharedParamsWidget.build_params` packages the build-time tuning
  (shuffles, FOV area, pixel size, frame interval) as a plain ``dict`` for the
  Build area's gating and for quantifiers that opt in to build params.

Every field is blank (unset) by default; the ``changed`` signal lets the studio
refresh build availability as values change.
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

#: Default permutation count for the contact-type z-score null (was PlotParams's
#: default; the in-napari plot layer that defined PlotParams has been removed).
_DEFAULT_SHUFFLES = 1000


class SharedParamsWidget(QWidget):
    """The studio's single set of build parameter fields."""

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
            placeholder="required for shape / dynamics",
            tip="µm per pixel, applied to all positions. Required to build the "
            "shape / dynamics metrics and to scale physical-unit plot axes "
            "(signed contact length, density). No per-position auto-resolution — "
            "blank leaves those metrics unbuildable.",
        )
        self._frame_interval_edit = self._field(
            col,
            "Frame interval (s):",
            placeholder="required for dynamics",
            tip="Seconds per frame, applied to all positions. Required to build the "
            "dynamics metrics. No per-position auto-resolution — blank leaves them "
            "unbuildable.",
        )
        self._fov_edit = self._field(
            col,
            "FOV area (mm²):",
            placeholder="required for density",
            tip="Field-of-view area (mm²), applied to all positions. Required to "
            "build Cell density — there is no image-area fallback.",
        )
        self._shuffles_edit = self._field(
            col,
            "Shuffles:",
            placeholder=str(_DEFAULT_SHUFFLES),
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

    # ---------------------------------------------------------------- build side
    def build_params(self) -> dict:
        """The shared knobs the studio reads at build time.

        Two roles share this dict. **Gating**: the Build area's
        ``required_build_params`` check reads ``pixel_size_um`` / ``time_interval_s``
        / ``fov_area_mm2`` here to tell a metric's param chips set from unset and to
        enable its checkbox. **Build values**: a quantifier that opts in
        (``wants_build_params``) is handed this dict as its ``params`` — the
        contact-type z-score's ``shuffles`` and the density's ``fov_area_mm2``.
        Pixel size / frame interval values still reach shape/dynamics builds
        through :meth:`stamp` (they are ``PositionInputs`` fields), so they appear
        here only to drive gating. Blank → ``None`` (unset) for the positives.
        """
        shuffles = _parse_int(self._shuffles_edit.text())
        return {
            "shuffles": shuffles if shuffles and shuffles > 0 else _DEFAULT_SHUFFLES,
            "pixel_size_um": _parse_positive(self._pixel_size_edit.text()),
            "time_interval_s": _parse_positive(self._frame_interval_edit.text()),
            "fov_area_mm2": _parse_positive(self._fov_edit.text()),
        }

    def stamp(self, records: list[dict]) -> list[dict]:
        """Return *records* with the global px/Δt applied, when set.

        A set pixel size / frame interval is written onto each record (the keys
        :func:`position_inputs_from_record` carries into ``PositionInputs``), so a
        build reads the shared value. There is no per-position fallback. Records
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
