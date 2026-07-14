"""Small shared helpers for the aggregate-quantification builders.

These were copy-pasted byte-for-byte across the shape, contacts, and dynamics
builders; consolidated here so there is one definition of each.
"""
from __future__ import annotations

from collections.abc import Callable


def report_progress(
    progress_cb: Callable[[int, int, str], None] | None,
    done: int,
    total: int,
    message: str,
) -> None:
    """Forward a progress tick to *progress_cb* when one was supplied."""
    if progress_cb is not None:
        progress_cb(done, total, message)


def itasc_version() -> str:
    """The installed itasc version, or ``"unknown"`` if it can't be resolved."""
    try:
        from importlib.metadata import version

        return version("itasc")
    except Exception:
        return "unknown"
