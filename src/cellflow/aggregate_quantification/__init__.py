"""Aggregate Quantification: pool per-position sources into plottable quantities.

This package hosts the quantifier seam. The stable, napari-free surface that the
``cellflow-aggregate`` CLI calls and notebooks import is the four-stage
:mod:`~cellflow.aggregate_quantification.pipeline` — ``build_catalog`` →
``build_quantities`` → ``aggregate`` → ``export`` — re-exported here alongside the
catalogue load/save helpers and the quantifier registry.

The contacts quantifier's public API (cell-cell edges, T1 events, NLS classes) is
also re-exported for stability; see :mod:`cellflow.aggregate_quantification.contacts`.
"""

from .catalog import load_catalog, save_catalog
from .pipeline import aggregate, build_catalog, build_quantities, export
from .quantifier import available_quantifiers

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
    # Pipeline: the stable CLI / notebook surface.
    "build_catalog",
    "build_quantities",
    "aggregate",
    "export",
    "load_catalog",
    "save_catalog",
    "available_quantifiers",
    # Contacts quantifier public API.
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
