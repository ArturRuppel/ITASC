"""Per-position contact analysis (cell-cell edges, T1 events, NLS classes)."""

from .batch import (
    ContactBatchJob,
    ContactBatchResult,
    discover_contact_batch_jobs,
    run_contact_batch,
)
from .build import (
    build_contact_analysis,
    build_position_contact_analysis,
    ensure_contact_analysis,
)

__all__ = [
    "build_contact_analysis",
    "build_position_contact_analysis",
    "ensure_contact_analysis",
    "ContactBatchJob",
    "ContactBatchResult",
    "discover_contact_batch_jobs",
    "run_contact_batch",
]
