"""Export CellFlow aggregate tables as Iris ``.iris`` bundles.

Discovers the known tidy CSVs in a data directory and writes one ``.iris``
document per table, each carrying the table plus its premade SuperPlot analyses.
Backend-only; usable from a headless / batch run or the CLI:

    python -m cellflow.aggregate_quantification.iris_export.export DATA_DIR
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .analyses import build_analyses
from .document import write_iris
from .schema import infer_schema

#: Table file stem → its finest object key (the swarm/box level). The spine is
#: ``date → position_id → <object_key> → frame``; for the non-cell tables the key
#: is category-/event-like, so the condition SuperPlot is a reasonable default the
#: user may re-pivot in Iris (see the design doc's caveat).
TABLE_OBJECT_KEYS = {
    "cells_by_frame": "cell_id",
    "cell_neighbors_by_frame": "cell_id",
    "contact_types_by_frame": "contact_type",
    "density_by_frame": "label",
    "edges_by_frame": "t1_event_id",
}

_EXPORTER = "cellflow.aggregate_quantification.iris_export"


def export_table(csv_path: Path | str, out_dir: Path | str,
                 object_key: str | None = None) -> Path:
    """Export one tidy CSV to ``<out_dir>/<stem>.iris``; return the written path.

    *object_key* defaults to the :data:`TABLE_OBJECT_KEYS` mapping for the file's
    stem; pass it explicitly for a table outside that mapping.
    """
    csv_path = Path(csv_path)
    stem = csv_path.stem
    object_key = object_key or TABLE_OBJECT_KEYS.get(stem)
    if object_key is None:
        raise ValueError(
            f"no object-key mapping for table {stem!r}; pass object_key explicitly"
        )

    df = _with_meta_columns(pd.read_csv(csv_path))
    schema = infer_schema(df)
    analyses = build_analyses(df, schema, object_key)
    provenance = {"source_csv": str(csv_path.resolve()), "exporter": _EXPORTER}
    data = write_iris(df, schema, analyses, provenance,
                      engine_snapshot={"producer": f"cellflow {_cellflow_version()}"})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.iris"
    out_path.write_bytes(data)
    return out_path


def export_dir(data_dir: Path | str, out_dir: Path | str | None = None) -> list[Path]:
    """Export every known table found in *data_dir*.

    Defaults the output to ``<data_dir>/iris/``. Returns the written paths in
    table order; tables not present are skipped.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir is not None else data_dir / "iris"
    written: list[Path] = []
    for stem in TABLE_OBJECT_KEYS:
        csv_path = data_dir / f"{stem}.csv"
        if csv_path.is_file():
            written.append(export_table(csv_path, out_dir))
    return written


def _with_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the stable ``id`` and ``excluded`` columns Iris keys rows on. Iris
    injects these on load when absent, but writing them gives stable row identity
    for click-to-exclude across reopen."""
    df = df.copy()
    if "id" not in df.columns:
        df.insert(0, "id", [str(i + 1) for i in range(len(df))])
    if "excluded" not in df.columns:
        df["excluded"] = False
    return df


def _cellflow_version() -> str:
    try:
        from cellflow import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - version is provenance only
        return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export CellFlow aggregate tables as Iris .iris bundles."
    )
    parser.add_argument("data_dir", type=Path, help="directory holding the tidy CSVs")
    parser.add_argument("-o", "--out-dir", type=Path, default=None,
                        help="output directory (default: DATA_DIR/iris)")
    args = parser.parse_args(argv)

    written = export_dir(args.data_dir, args.out_dir)
    if not written:
        print(f"no known tables found in {args.data_dir}")
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
