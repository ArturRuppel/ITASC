"""Signed-contact-length quantifier — signed T1 junction lengths for the potential.

A contacts-derived Build product. Persists the signed central-junction length of
each T1 event (negative on the losing side, positive on the gaining side); pooled
and Boltzmann-inverted in the plot panel these reproduce the double-well potential.
Lengths are in µm when the position's pixel size resolves, else pixels. The
``contact_type`` transition label is normalized here (blank → ``"unlabelled"``) so
the plot's group axis never sees an empty string.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from cellflow.aggregate_quantification.contacts.signed_contact_length import (
    signed_central_junction_lengths,
)
from cellflow.aggregate_quantification.quantifier import PositionInputs, Quantifier
from cellflow.aggregate_quantification.quantifiers import _contacts_derived as derived


class SignedContactLengthQuantifier(Quantifier):
    """Per-T1-event signed central junction length (the potential's samples)."""

    quantity_id = "signed_contact_length"
    display_name = "Signed contact length"
    requires = ("contact_analysis_path",)
    default_output_name = "signed_contact_length.csv"
    # One row per T1 event; ``contact_type`` / ``role`` are categorical axes that
    # key the table alongside the event id (only ``signed_length`` is a value
    # column).
    table_keys = ("frame", "t1_event_id", "role", "contact_type")

    def build(
        self,
        inputs: PositionInputs,
        output_path: Path,
        *,
        params: dict | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> Path:
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
        return derived.persist(output_path, table)

    def read(self, output_path: Path) -> dict[str, np.ndarray]:
        return derived.read_derived_table(output_path)

    def object_table(self, output_path: Path) -> Mapping[str, np.ndarray]:
        return derived.read_derived_table(output_path)
