"""The shared aggregate-quantification provenance helpers."""
from __future__ import annotations

from cellflow.aggregate_quantification._provenance import (
    cellflow_version,
    report_progress,
)


def test_report_progress_forwards_when_callback_given():
    seen = []
    report_progress(lambda d, t, m: seen.append((d, t, m)), 2, 5, "hi")
    assert seen == [(2, 5, "hi")]


def test_report_progress_is_noop_without_callback():
    report_progress(None, 1, 1, "x")  # must not raise


def test_cellflow_version_returns_a_string():
    v = cellflow_version()
    assert isinstance(v, str) and v  # installed version or "unknown"
