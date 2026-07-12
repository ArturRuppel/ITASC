"""Aggregate capstone: pool every processed position into project-level tables.

The main app's project-level bookend to the per-position sections. Reads the same
catalog records the ``ExperimentsPanel`` builds, and drives the headless engine
(``author_config`` then ``pipeline.run``). Pool-only: it aggregates positions
whose ``contacts.h5`` already exists and never builds missing ones, so ``run`` is
load-and-pool with no per-position recompute. Plots live in Iris.
"""
from __future__ import annotations

from pathlib import Path

from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import (
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.contact_analysis import author_config, run
from cellflow.contact_analysis.shape_tables import catalogue_root


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


def pool_positions(ready_records, skipped_names):
    """Author the project artifacts for *ready_records* and run the engine.

    Writes ``catalog.csv`` + ``config.toml`` into the ready positions' common
    ancestor (:func:`catalogue_root`), then ``run``s the pipeline over them.
    Returns a result dict for the UI: the ``name -> path`` table map, the
    ``skipped`` position names, and the ``project_dir`` the tables landed under.
    """
    project_dir = catalogue_root(ready_records)
    config_path = author_config(project_dir, ready_records, quantities=())
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

        layout = QVBoxLayout(self)
        self.subtitle = QLabel(
            "Pools every processed position into project-level tables."
        )
        self.subtitle.setWordWrap(True)
        self.readout = QLabel("No data folders yet.")
        self.readout.setWordWrap(True)
        self.run_btn = QPushButton("Pool ready positions")
        self.run_btn.clicked.connect(self._on_run)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.results = QListWidget()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        for widget in (
            self.subtitle,
            self.readout,
            self.run_btn,
            self.progress,
            self.results,
            self.status,
        ):
            layout.addWidget(widget)
        self._refresh_readout()

    # ------------------------------------------------------------------ inputs
    def set_records(self, records) -> None:
        """Replace the catalog records the readiness readout reflects."""
        self._records = list(records or [])
        self._refresh_readout()

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
        self.run_btn.setEnabled(bool(ready) and self._worker is None)

    # --------------------------------------------------------------------- run
    def _on_run(self) -> None:
        ready, not_ready = partition_ready(self._records)
        if not ready:
            show_info("No analyzed positions to pool.")
            return
        skipped = [_position_name(r) for r in not_ready]
        self.results.clear()
        self.status.setText("Pooling…")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.run_btn.setEnabled(False)

        @thread_worker(
            connect={"returned": self._on_done, "errored": self._on_error}
        )
        def _work():
            return pool_positions(ready, skipped)

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
