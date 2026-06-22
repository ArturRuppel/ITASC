"""Analyst-driven, replicate-level analyses over the aggregate tables.

The standard export (``iris_export``) ships one SuperPlot per metric; this package
captures the *cross-table, replicate-level* questions that need statistics the Iris
engine doesn't express — metric-vs-metric correlation with replicate-level inference,
and assortative clustering of a classifier vs a spatial null. Each report is
parameterized by column names defaulting to CellFlow's standard schema, so it
reproduces on a new experiment's ``aggregate_quantification/`` directory.

Run on a dataset::

    python -m cellflow.aggregate_quantification.analysis AGG_DIR OUT_DIR \
        --correlate cell_dynamics.speed_um_per_s neighbor_count.n_neighbors

The package ships no dataset's questions baked in: a concrete experiment's analysis
(its chosen correlations, bespoke titles, and the recorded findings) lives with that
experiment, as a small driver that imports these report functions.
"""
from .reports import label_clustering_report, metric_correlation_report
from .stats import (
    CorrelationResult,
    EnrichmentResult,
    homotypic_enrichment,
    replicate_correlation,
)

__all__ = [
    "CorrelationResult",
    "EnrichmentResult",
    "homotypic_enrichment",
    "label_clustering_report",
    "metric_correlation_report",
    "replicate_correlation",
]
