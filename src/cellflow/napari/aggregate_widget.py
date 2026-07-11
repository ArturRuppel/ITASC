"""Aggregate capstone: pool every processed position into project-level tables.

The main app's project-level bookend to the per-position sections. Reads the same
catalog records the ``ExperimentsPanel`` builds, and drives the headless engine
(``author_config`` then ``pipeline.run``). Pool-only: it aggregates positions
whose ``contacts.h5`` already exists and never builds missing ones, so ``run`` is
load-and-pool with no per-position recompute. Plots live in Iris.
"""
from __future__ import annotations

from pathlib import Path

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
    tables = run(config_path)
    return {
        "tables": tables,
        "skipped": list(skipped_names),
        "project_dir": project_dir,
    }
