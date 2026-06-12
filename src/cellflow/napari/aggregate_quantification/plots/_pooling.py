"""Pool one product's per-position tables into a tidy DataFrame for plotting.

The plot area's generic statistical plots all need the same thing: read the
built ``object_table`` for a single ``quantity_id`` across the in-scope
positions, stamp catalogue metadata, left-join the NLS subpopulation
``class_label`` (by ``cell_id``) when a position has one, and pool the lot. This
module is that step, factored out of the old per-plugin pooling so every
:class:`~cellflow.napari.aggregate_quantification.plots.Plot` shares one
implementation.

Headless (no Qt): runs on a worker thread off the GUI and is unit-testable.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cellflow.aggregate_quantification.plotting import PositionSource, pool_object_tables
from cellflow.aggregate_quantification.quantifier import Quantifier, available_quantifiers

#: A per-position table extractor: ``(quantifier, output_path) -> tidy table``.
TableFn = Callable[[Quantifier, Path], Any]

#: Subpopulation column joined from the NLS sidecar CSV (by ``cell_id``); becomes
#: just another group-by column in the panel.
CLASS_COLUMN = "class_label"
#: The catalogue-metadata axes a plot can group / facet by.
METADATA_GROUPS = ("condition", "date", "position_id")
#: Column roles handed to the (quantity-agnostic) PlotPanel as group-by options.
GROUP_COLUMNS = (*METADATA_GROUPS, CLASS_COLUMN, "frame")


def quantifier_for(quantity_id: str) -> Quantifier:
    """The registered quantifier instance producing *quantity_id*."""
    for q_cls in available_quantifiers():
        if q_cls.quantity_id == quantity_id:
            return q_cls()
    raise KeyError(f"No registered quantifier produces {quantity_id!r}")


def iter_built(quantity_id: str, records: list[dict]) -> Iterator[tuple[dict, Path]]:
    """Yield ``(record, output_path)`` for each *records* entry whose
    *quantity_id* product is built. Shared by every pooling shape."""
    quantifier = quantifier_for(quantity_id)
    # Lazy import: studio_plugins lives in the Qt layer above this package; a
    # module-level import would risk a discovery-time cycle.
    from cellflow.napari.studio_plugins import output_for_record

    for record in records:
        path = output_for_record(quantifier, record)
        if quantifier.is_built(path):
            yield record, path


def pool_quantity(
    quantity_id: str,
    records: list[dict],
    *,
    table_fn: TableFn | None = None,
    join_class: bool = True,
) -> pd.DataFrame:
    """Pool the built *quantity_id* tables across *records* into one tidy frame.

    Reads each in-scope position's table (skipping those not built), stamps
    ``condition`` / ``date`` / ``position_id``, and — when *join_class* — left-joins
    the position's NLS ``class_label`` by ``cell_id`` (absent → ``unclassified``).
    *table_fn* selects the per-position table; it defaults to the quantifier's
    ``object_table`` but a plot may pass a different reader (e.g. the per-track
    summary). Runs off the GUI thread.
    """
    quantifier = quantifier_for(quantity_id)
    extract: TableFn = table_fn or (lambda q, path: q.object_table(path))

    sources: list[PositionSource] = []
    for record, path in iter_built(quantity_id, records):
        sources.append(
            PositionSource(
                metadata=position_metadata(record),
                table=extract(quantifier, path),
                join_table=_nls_join_table(record) if join_class else None,
                join_columns=(CLASS_COLUMN,),
            )
        )
    return pool_object_tables(sources)


def _nls_join_table(record: dict) -> dict[str, np.ndarray] | None:
    """A ``{cell_id, class_label}`` join table from the record's NLS sidecar CSV,
    or ``None`` when the position has no CSV (→ ``unclassified`` at pool time)."""
    from cellflow.aggregate_quantification.contacts.nls_classification import (
        nls_classification_csv_path,
        read_nls_classification_csv,
    )

    position_path = record.get("position_path")
    if not position_path:
        return None
    csv_path = nls_classification_csv_path(position_path)
    if not csv_path.is_file():
        return None
    labels = read_nls_classification_csv(csv_path)
    if not labels:
        return None
    return {
        "cell_id": np.asarray(list(labels), dtype=np.int64),
        CLASS_COLUMN: np.asarray(list(labels.values()), dtype=object),
    }


def position_metadata(record: dict) -> dict[str, Any]:
    """The catalogue-metadata columns stamped onto every pooled row."""
    return {
        "condition": str(record.get("condition", "")),
        "date": str(record.get("date", "")),
        "position_id": str(record.get("id", "")),
    }
