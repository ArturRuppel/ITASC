"""Cell-cell contact quantifier: edges, T1 events, NLS classification.

The contacts domain logic that used to live at the top-level
``cellflow.contact_analysis`` package. It is one quantifier within
:mod:`cellflow.aggregate_quantification`; its public API is re-exported from the
package root for stability.
"""
from __future__ import annotations

from cellflow.aggregate_quantification.contacts.contact_labels import label_contacts
from cellflow.aggregate_quantification.contacts.signed_contact_length import (
    signed_central_junction_lengths,
)
from cellflow.aggregate_quantification.contacts.neighborhood import (
    cell_density,
    cell_neighbor_counts,
    contact_type_zscores,
    neighbor_enrichment,
)

__all__ = [
    "label_contacts",
    "signed_central_junction_lengths",
    "cell_neighbor_counts",
    "neighbor_enrichment",
    "contact_type_zscores",
    "cell_density",
]
