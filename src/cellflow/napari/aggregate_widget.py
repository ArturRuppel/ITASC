"""Aggregate capstone: pool every processed position into project-level tables.

The main app's project-level bookend to the per-position sections. Reads the same
catalog records the ``ExperimentsPanel`` builds, and drives the headless engine
(``author_config`` then ``pipeline.run``). Pool-only: it aggregates positions
whose ``contacts.h5`` already exists and never builds missing ones, so ``run`` is
load-and-pool with no per-position recompute. Plots live in Iris.
"""
from __future__ import annotations

from pathlib import Path

import tifffile
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.contact_analysis import author_config, run
from cellflow.contact_analysis.quantifier import available_quantifiers
from cellflow.contact_analysis.records import supported_quantities
from cellflow.contact_analysis.shape_tables import catalogue_root


def _lateral_pixels(path) -> int | None:
    """The ``Y * X`` pixel count of a label image's lateral field, or ``None``.

    Reads only the TIFF header (not the pixel data), so it is cheap to call on every
    records refresh. Returns ``None`` when *path* is missing, unreadable, or not at
    least 2-D. The field-of-view autofill multiplies this by ``pixel_size_um²``.
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with tifffile.TiffFile(p) as tif:
            shape = tif.series[0].shape
    except Exception:
        return None
    if len(shape) < 2:
        return None
    return int(shape[-2]) * int(shape[-1])


def pooled_quantifiers():
    """The registered quantifiers that pool into a project table, in display order.

    Only quantifiers declaring ``table_keys`` produce an aggregated CSV; producers
    (e.g. ``contacts``) are excluded. :func:`available_quantifiers` already returns
    them sorted by ``display_name``, which is the order the checkbox list shows.
    """
    return [cls for cls in available_quantifiers() if cls.table_keys]


def partition_ready(records):
    """Split catalog *records* into ``(ready, not_ready)`` by ``contacts.h5``.

    A record is *ready* when its ``contact_analysis_path`` exists on disk.
    """
    ready, not_ready = [], []
    for rec in records:
        path = rec.get("contact_analysis_path")
        if path is not None and Path(path).exists():
            ready.append(rec)
        else:
            not_ready.append(rec)
    return ready, not_ready


def _position_name(record) -> str:
    path = record.get("position_path")
    return Path(path).name if path else "(unknown)"


def pool_positions(ready_records, skipped_names, quantities=(), params=None):
    """Author the project artifacts for *ready_records* and run the engine.

    Writes ``catalog.csv`` + ``config.toml`` into the ready positions' common
    ancestor (:func:`catalogue_root`), then ``run``s the pipeline over them.
    *quantities* is the checked subset of pooled quantities to write (empty = every
    available table); it is authored into the config and restricts which tables the
    pool-only run emits. *params* are the shared build knobs (pixel size, frame
    length, FOV area) authored into the config's ``[params]`` so the pooled cheap
    quantities compute in physical units. Returns a result dict for the UI: the
    ``name -> path`` table map, the ``skipped`` position names, and the
    ``project_dir`` the tables landed under.
    """
    project_dir = catalogue_root(ready_records)
    config_path = author_config(
        project_dir, ready_records, quantities=tuple(quantities), params=params or None
    )
    # Pool-only: read each position's existing contacts.h5 and pool it (plus the
    # in-memory cheap quantities). build=False skips the producer's unconditional
    # rebuild, so ready positions are loaded, never recomputed.
    tables = run(config_path, build=False)
    return {
        "tables": tables,
        "skipped": list(skipped_names),
        "project_dir": project_dir,
    }


class AggregateWidget(QWidget):
    """Project-level capstone: pool every ready position into tidy tables.

    Fed catalog records via :meth:`set_records` (the same records the app's
    ``ExperimentsPanel`` builds). Pool-only: Run aggregates positions whose
    ``contacts.h5`` exists and reports the ones it skipped by name.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records: list[dict] = []
        self._worker = None
        self._has_run = False
        self._fov_user_edited = False

        layout = QVBoxLayout(self)
        self.subtitle = QLabel(
            "Pools every processed position into project-level tables."
        )
        self.subtitle.setWordWrap(True)
        self.readout = QLabel("No data folders yet.")
        self.readout.setWordWrap(True)
        self.quantities_label = QLabel("Quantities to pool:")
        # One checkbox per pooled quantity, all on by default. A box is greyed out
        # (disabled + unchecked) when no ready position carries the inputs/params it
        # needs, so the list also reads as "what this stage computes", and an
        # enabled box never promises a table the run would skip. See
        # :func:`~cellflow.contact_analysis.records.supported_quantities`.
        self._checks: dict[str, QCheckBox] = {}
        for cls in pooled_quantifiers():
            box = QCheckBox(cls.display_name)
            box.setChecked(True)
            box.setToolTip(cls.quantity_id)
            self._checks[cls.quantity_id] = box
        # Cell density's field-of-view area is a build param with no PositionInputs
        # field, so it gets its own input beside the density checkbox. Autofilled
        # from image size x pixel size (see :meth:`_maybe_autofill_fov`); the user
        # can override, after which autofill leaves it alone.
        self.fov_field = QDoubleSpinBox()
        # 6 decimals = 1 µm² resolution, so a small ROI or fine pixel size autofills
        # a representable area rather than rounding down to zero (which would grey
        # Cell density out).
        self.fov_field.setDecimals(6)
        self.fov_field.setRange(0.0, 1e9)
        self.fov_field.setSuffix(" mm²")
        self.fov_field.setToolTip(
            "Field-of-view area for Cell density. Autofilled from image size times "
            "pixel size; edit to override."
        )
        self.fov_field.valueChanged.connect(self._on_fov_changed)
        self.run_btn = QPushButton("Pool ready positions")
        self.run_btn.clicked.connect(self._on_run)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.results = QListWidget()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        for widget in (self.subtitle, self.readout, self.quantities_label):
            layout.addWidget(widget)
        for qid, box in self._checks.items():
            if qid == "cell_density":
                row = QHBoxLayout()
                row.addWidget(box)
                row.addWidget(QLabel("FOV area (mm²):"))
                row.addWidget(self.fov_field)
                row.addStretch()
                layout.addLayout(row)
            else:
                layout.addWidget(box)
        for widget in (self.run_btn, self.progress, self.results, self.status):
            layout.addWidget(widget)
        self._refresh_readout()

    # ------------------------------------------------------------------ inputs
    def set_records(self, records) -> None:
        """Replace the catalog records the readiness readout reflects."""
        self._records = list(records or [])
        self._maybe_autofill_fov()
        self._refresh_readout()

    def _current_params(self) -> dict:
        """The shared build knobs for greying + the run: calibration off the records
        (pixel size, frame length are global, so the first record is representative)
        plus the field-of-view area from :attr:`fov_field` (omitted when zero)."""
        params: dict[str, float] = {}
        if self._records:
            first = self._records[0]
            for key in ("pixel_size_um", "time_interval_s"):
                value = first.get(key)
                if value in (None, ""):
                    continue
                try:
                    params[key] = float(value)
                except (TypeError, ValueError):
                    pass
        fov = self.fov_field.value()
        if fov > 0:
            params["fov_area_mm2"] = fov
        return params

    def _maybe_autofill_fov(self) -> None:
        """Prefill the FOV field from ``image lateral pixels x pixel_size_um²`` (in
        mm²), reading a ready position's cell-label image. No-ops once the user has
        edited the field, when no pixel size is set, or when no image is readable."""
        if self._fov_user_edited:
            return
        pixel_size = self._current_params().get("pixel_size_um")
        if not pixel_size:
            return
        ready, _ = partition_ready(self._records)
        for record in ready:
            n_pixels = _lateral_pixels(record.get("cell_tracked_labels_path"))
            if n_pixels:
                fov_mm2 = n_pixels * pixel_size * pixel_size / 1e6
                self.fov_field.blockSignals(True)
                self.fov_field.setValue(fov_mm2)
                self.fov_field.blockSignals(False)
                return

    def _on_fov_changed(self, _value) -> None:
        """A user edit to the FOV field: remember it (so autofill backs off) and
        re-grey, since a non-zero area lifts Cell density into support."""
        self._fov_user_edited = True
        self._refresh_quantities(partition_ready(self._records)[0])

    def section_status(self) -> str:
        """Status for the enclosing section dot: not_started / in_progress / done."""
        ready, _ = partition_ready(self._records)
        if not ready:
            return "not_started"
        return "done" if self._has_run else "in_progress"

    # --------------------------------------------------------------- rendering
    def _refresh_readout(self) -> None:
        ready, not_ready = partition_ready(self._records)
        total = len(self._records)
        if total == 0:
            self.readout.setText("No data folders yet.")
        else:
            message = f"{len(ready)} of {total} positions analyzed"
            if not_ready:
                names = ", ".join(_position_name(r) for r in not_ready)
                message += f" — not yet ready: {names}"
            self.readout.setText(message)
        self._refresh_quantities(ready)
        self.run_btn.setEnabled(bool(ready) and self._worker is None)

    def _refresh_quantities(self, ready) -> None:
        """Enable a quantity's checkbox iff a ready position can produce its table.

        Greys out (disables + unchecks) any quantity whose required inputs/params
        are absent from every ready position; re-enables and re-checks one that
        becomes supported again. A deliberate uncheck of a *supported* quantity is
        preserved: only the supported/unsupported transition touches the check.
        """
        supported = supported_quantities(ready, params=self._current_params() or None)
        for qid, box in self._checks.items():
            was_enabled = box.isEnabled()
            now = qid in supported
            box.setEnabled(now)
            box.setToolTip(
                qid
                if now
                else f"{qid}: required inputs or parameters not available for any ready position"
            )
            if not now:
                box.setChecked(False)
            elif not was_enabled:
                box.setChecked(True)

    def _selected_quantities(self) -> tuple[str, ...]:
        """The checked quantities to author into the config.

        Collapses to ``()`` (= write every available table) when the user has left
        every *supported* quantity checked, so a default run writes a clean config;
        any deliberate uncheck yields the explicit checked subset instead.
        """
        checked = {qid for qid, box in self._checks.items() if box.isChecked()}
        supported = {qid for qid, box in self._checks.items() if box.isEnabled()}
        return () if checked == supported else tuple(sorted(checked))

    # --------------------------------------------------------------------- run
    def _on_run(self) -> None:
        ready, not_ready = partition_ready(self._records)
        if not ready:
            show_info("No analyzed positions to pool.")
            return
        skipped = [_position_name(r) for r in not_ready]
        quantities = self._selected_quantities()
        params = self._current_params()
        self.results.clear()
        self.status.setText("Pooling…")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.run_btn.setEnabled(False)

        @thread_worker(
            connect={"returned": self._on_done, "errored": self._on_error}
        )
        def _work():
            return pool_positions(ready, skipped, quantities, params)

        self._worker = _work()

    def _on_done(self, result: dict) -> None:
        self._worker = None
        self._has_run = True
        self.progress.setVisible(False)
        for name, path in sorted(result["tables"].items()):
            self.results.addItem(f"{name}: {path}")
        message = f"Pooled into {result['project_dir']}. Plots live in Iris."
        if result["skipped"]:
            message += f" Skipped (not analyzed): {', '.join(result['skipped'])}."
        self.status.setText(message)
        show_info(message)
        self._refresh_readout()

    def _on_error(self, exc: Exception) -> None:
        self._worker = None
        self.progress.setVisible(False)
        self.status.setText(f"Aggregate failed: {exc}")
        show_error(f"Aggregate failed: {exc}")
        self._refresh_readout()
