"""Aggregate Quantification: pool per-position sources into plottable quantities.

This package hosts the quantifier seam. For now its one quantifier is contacts
(cell-cell edges, T1 events, NLS classes), whose public API is re-exported here
for stability; see :mod:`cellflow.aggregate_quantification.contacts`.
"""

from .contacts.batch import (
    ContactBatchJob,
    ContactBatchResult,
    discover_contact_batch_jobs,
    run_contact_batch,
)
from .contacts.build import (
    build_contact_analysis,
    build_position_contact_analysis,
    ensure_contact_analysis,
)
from .contacts.nls_classification import (
    NLSClassificationError,
    NLSClassificationSummary,
    TrackNLSMeasurement,
    auto_threshold,
    classify_by_threshold,
    classify_position_nls_to_csv,
    measure_track_nls_intensity,
    nls_classification_csv_path,
    read_nls_classification_csv,
    write_nls_classification_csv,
)

__all__ = [
    "build_contact_analysis",
    "build_position_contact_analysis",
    "ensure_contact_analysis",
    "ContactBatchJob",
    "ContactBatchResult",
    "discover_contact_batch_jobs",
    "run_contact_batch",
    "NLSClassificationError",
    "NLSClassificationSummary",
    "TrackNLSMeasurement",
    "auto_threshold",
    "classify_by_threshold",
    "classify_position_nls_to_csv",
    "measure_track_nls_intensity",
    "nls_classification_csv_path",
    "read_nls_classification_csv",
    "write_nls_classification_csv",
]
