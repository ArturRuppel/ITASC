"""Signed-contact-length quantifier — signed T1 junction lengths for the potential.

A contacts-derived **pooled** quantity: :meth:`compute_object_table` returns the
signed central-junction length of each T1 event (negative on the losing side,
positive on the gaining side) in memory (no per-position artifact). Pooled and
Boltzmann-inverted downstream these reproduce the double-well potential. Lengths
are in µm when the position's pixel size resolves, else pixels. The
``contact_type`` transition label is normalized here (blank → ``"unlabelled"``) so
a group axis never sees an empty string.
"""
from __future__ import annotations

import numpy as np

from itasc.contact_analysis.contacts.signed_contact_length import (
    signed_central_junction_lengths,
)
from itasc.contact_analysis.quantifier import Quantifier
from itasc.contact_analysis.quantifiers import _contacts_derived as derived


class SignedContactLengthQuantifier(Quantifier):
    """Per-T1-event signed central junction length (the potential's samples)."""

    quantity_id = "signed_contact_length"
    display_name = "Signed contact length"
    requires = ("contact_analysis_path",)
    # One row per T1 event; ``contact_type`` / ``role`` are categorical axes that
    # key the table alongside the event id (only ``signed_length`` is a value
    # column).
    table_keys = ("frame", "t1_event_id", "role", "contact_type")

    def compute_object_table(self, inputs, *, params=None):
        analysis = derived.load_analysis(inputs)
        table = dict(
            signed_central_junction_lengths(
                analysis, pixel_size_um=inputs.pixel_size_um
            )
        )
        if "contact_type" in table:
            ct = np.asarray(table["contact_type"], dtype=object)
            ct[ct == ""] = "unlabelled"
            table["contact_type"] = ct
        return table
