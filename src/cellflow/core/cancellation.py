"""Cooperative-cancellation primitive shared across CellFlow pieces.

Long-running workers (segmentation, tracking, correction) poll a ``cancel``
callback and raise :class:`CancelledError` when it returns ``True``. The
exception lives in ``cellflow-core`` so the independently-installable pieces can
share one cancellation type without depending on the segmentation stage.
"""
from __future__ import annotations


class CancelledError(Exception):
    """Raised when a cooperative cancel signal is observed mid-computation."""


__all__ = ["CancelledError"]
