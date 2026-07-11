"""Contact Analysis: pool per-position sources into aggregate tidy tables.

This package hosts the quantifier seam. The stable, napari-free surface that the
``cellflow-aggregate`` CLI calls and notebooks import is the
:mod:`~cellflow.contact_analysis.pipeline` — ``build_catalog`` →
``build_quantities`` → ``aggregate`` — re-exported here alongside the catalogue
load/save helpers and the quantifier registry. Everything it produces is
**label-agnostic**: tidy CSVs with no subpopulation classification baked in.
Reduction to replicate-level summaries and plotting are downstream concerns
owned by Iris / the data repo, not this package.

The contacts quantifier's public API (cell-cell edges, T1 events) is also
re-exported for stability; see :mod:`cellflow.contact_analysis.contacts`.
"""

from .catalog import load_catalog, save_catalog
from .config import RunConfig, load_config
from .pipeline import (
    aggregate,
    author_config,
    build_catalog,
    build_quantities,
    run,
    select_quantifiers,
)
from .quantifier import available_quantifiers

from .contacts.batch import (
    ContactBatchJob,
    ContactBatchResult,
    discover_contact_batch_jobs,
    run_contact_batch,
)
from .contacts.build import (
    build_contacts,
    build_position_contacts,
    ensure_contacts,
)

__all__ = [
    # Pipeline: the stable CLI / notebook surface.
    "build_catalog",
    "build_quantities",
    "select_quantifiers",
    "aggregate",
    "author_config",
    "run",
    "load_catalog",
    "save_catalog",
    "RunConfig",
    "load_config",
    "available_quantifiers",
    # Contacts quantifier public API.
    "build_contacts",
    "build_position_contacts",
    "ensure_contacts",
    "ContactBatchJob",
    "ContactBatchResult",
    "discover_contact_batch_jobs",
    "run_contact_batch",
]
