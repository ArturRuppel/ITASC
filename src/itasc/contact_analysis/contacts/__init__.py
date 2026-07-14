"""Cell-cell contact quantifier: edges, T1 events, neighbourhood & density.

The contacts domain logic that used to live at the top-level
``itasc.contact_analysis`` package. It is one quantifier within
:mod:`itasc.contact_analysis`; its public API is re-exported from the
package root for stability. All of it is **label-agnostic**.
"""
from __future__ import annotations

from itasc.contact_analysis.contacts.contact_labels import label_contacts
from itasc.contact_analysis.contacts.signed_contact_length import (
    signed_central_junction_lengths,
)
from itasc.contact_analysis.contacts.neighborhood import (
    cell_density,
    cell_neighbor_counts,
)

__all__ = [
    "label_contacts",
    "signed_central_junction_lengths",
    "cell_neighbor_counts",
    "cell_density",
]
