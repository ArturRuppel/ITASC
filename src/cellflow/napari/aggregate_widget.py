"""Aggregate capstone: pool every processed position into project-level tables.

The main app's project-level bookend to the per-position sections. Reads the same
catalog records the ``ExperimentsPanel`` builds, and drives the headless engine
(``author_config`` then ``pipeline.run``). Pool-only: it aggregates positions
whose ``contacts.h5`` already exists and never builds missing ones, so ``run`` is
load-and-pool with no per-position recompute. Plots live in Iris.
"""
from __future__ import annotations

from pathlib import Path


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
