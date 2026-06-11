"""Cell-cell contact quantifier: edges, T1 events, NLS classification.

The contacts domain logic that used to live at the top-level
``cellflow.contact_analysis`` package. It is one quantifier within
:mod:`cellflow.aggregate_quantification`; its public API is re-exported from the
package root for stability.
"""
from __future__ import annotations

from cellflow.aggregate_quantification.contacts.energetics import (
    signed_central_junction_lengths,
)

__all__ = ["signed_central_junction_lengths"]
