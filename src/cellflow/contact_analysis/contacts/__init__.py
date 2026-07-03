"""Cell-cell contact quantifier: edges, T1 events, neighbourhood & density.

The contacts domain logic that used to live at the top-level
``cellflow.contact_analysis`` package. It is one quantifier within
:mod:`cellflow.contact_analysis`; its public API is re-exported from the
package root for stability. All of it is **label-agnostic**.
"""
from __future__ import annotations

from cellflow.contact_analysis.contacts.contact_labels import label_contacts
from cellflow.contact_analysis.contacts.signed_contact_length import (
    signed_central_junction_lengths,
)
from cellflow.contact_analysis.contacts.neighborhood import (
    cell_density,
    cell_neighbor_counts,
)

__all__ = [
    "label_contacts",
    "signed_central_junction_lengths",
    "cell_neighbor_counts",
    "cell_density",
]
