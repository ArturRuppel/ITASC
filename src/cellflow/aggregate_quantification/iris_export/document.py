"""Write an Iris ``.iris`` document by hand.

A ``.iris`` file is a ZIP of a Parquet table plus human-readable JSON parts
(manifest, schema, one file per analysis, provenance). This module reimplements
the reader's format (Iris ``document.py``, ``format_version`` 1.0) so CellFlow can
emit bundles **without importing the Iris engine** — the file format is the only
contract between the two projects. The format is documented and frozen in
``docs/superpowers/specs/2026-06-17-iris-export-design.md``; if it ever drifts,
the structural contract test (``tests/aggregate_quantification/test_iris_export``)
fails first.
"""
from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import pandas as pd

#: The ``.iris`` format version CellFlow targets. Iris refuses a document whose
#: ``format_version`` is *newer* than its own reader, so this must not exceed it.
FORMAT_VERSION = "1.0"


def write_iris(
    table: pd.DataFrame,
    schema: Mapping[str, Any],
    analyses: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    *,
    engine_snapshot: Mapping[str, Any] | None = None,
) -> bytes:
    """Serialize one table + its analyses into ``.iris`` (v1.0) bytes.

    The table is written as Parquet (zstd) so dtypes/nulls round-trip exactly with
    no re-inference on load; the sidecars stay JSON so the document is inspectable
    and diffable. ``analyses`` is an ordered sequence — each lands in its own
    ``analyses/NN-<id>.json`` part, numbered from 01.
    """
    parquet = io.BytesIO()
    table.to_parquet(parquet, index=False, engine="pyarrow", compression="zstd")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", _dumps({
            "format_version": FORMAT_VERSION,
            "modified": datetime.now(timezone.utc).isoformat(),
            "engine_snapshot": dict(engine_snapshot or {}),
        }))
        # Parquet is already zstd-compressed; store it raw rather than waste CPU
        # re-deflating incompressible bytes (matches the Iris writer).
        zf.writestr("data/table.parquet", parquet.getvalue(),
                    compress_type=zipfile.ZIP_STORED)
        zf.writestr("data/schema.json", _dumps(schema))
        for index, analysis in enumerate(analyses, 1):
            ident = analysis.get("id", "analysis")
            zf.writestr(f"analyses/{index:02d}-{ident}.json", _dumps(analysis))
        zf.writestr("provenance.json", _dumps(provenance))
    return buf.getvalue()


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2)
