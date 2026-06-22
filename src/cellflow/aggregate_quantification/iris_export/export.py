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
from ..curation import apply_curation, filter_excluded

#: Tables :func:`export_dir` writes. The object-grain morphology + motility
#: tables for both segmented objects (cell and nucleus), each carrying premade
#: by-condition / by-class SuperPlots. Tables absent from a given run are skipped.
#: Other built tables (density, contacts, neighbour counts, shape-relational) are
#: still exportable via :func:`export_table` directly and deferred until their
#: object-grain plots are worth shipping (see the artifact-contract design doc, §8).
TABLES_TO_EXPORT = (
    "cell_shape",
    "nucleus_shape",
    "cell_dynamics",
    "nucleus_dynamics",
)

_EXPORTER = "cellflow.aggregate_quantification.iris_export"


def _object_key_for(stem: str) -> str | None:
    """The finest object key (swarm/box level) for the table named *stem* — the
    first non-``frame`` key of the quantifier whose ``quantity_id`` is *stem* — or
    ``None`` when *stem* names no aggregated quantity. The SuperPlot spine is
    ``experiment_id → position_id → <object_key> → frame``."""
    from ..shape_tables import shape_table_registry

    spec = shape_table_registry().get(stem)
    if spec is None:
        return None
    non_frame = [key for key in spec.keys if key != "frame"]
    return non_frame[0] if non_frame else None


def export_table(csv_path: Path | str, out_dir: Path | str,
                 object_key: str | None = None) -> Path:
    """Export one tidy CSV to ``<out_dir>/<stem>.iris``; return the written path.

    *object_key* defaults to the finest key of the quantifier the file's stem names
    (:func:`_object_key_for`); pass it explicitly for a table outside the registry.
    """
    csv_path = Path(csv_path)
    stem = csv_path.stem
    object_key = object_key or _object_key_for(stem)
    if object_key is None:
        raise ValueError(
            f"no object-key mapping for table {stem!r}; pass object_key explicitly"
        )

    return export_table_frame(
        pd.read_csv(csv_path), stem, out_dir,
        object_key=object_key, source={"source_csv": str(csv_path.resolve())},
    )


def export_table_frame(
    df: pd.DataFrame,
    stem: str,
    out_dir: Path | str,
    *,
    object_key: str | None = None,
    source: dict | None = None,
) -> Path:
    """Export an in-memory tidy *frame* to ``<out_dir>/<stem>.iris``.

    The frame-based core of :func:`export_table`: lets a caller (the pipeline) hand
    over an already-built frame so the ``.iris`` is written with no detour through
    disk. The bundle is a pure function of (frame + analysis spec) — no curation or
    other human judgment is baked in (Iris removed its exclusion mechanism; to drop
    rows, carry a boolean flag column upstream and filter on it in a reduce step).
    """
    object_key = object_key or _object_key_for(stem)
    if object_key is None:
        raise ValueError(
            f"no object-key mapping for table {stem!r}; pass object_key explicitly"
        )

    df = _with_meta_columns(df)
    schema = infer_schema(df)
    analyses = build_analyses(df, schema, object_key)
    provenance = {"exporter": _EXPORTER, **(source or {})}
    data = write_iris(df, schema, analyses, provenance,
                      engine_snapshot={"producer": f"cellflow {_cellflow_version()}"})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.iris"
    out_path.write_bytes(data)
    return out_path


def export_dir(
    data_dir: Path | str,
    out_dir: Path | str | None = None,
    *,
    curation: pd.DataFrame | None = None,
    curation_path: str | None = None,
) -> list[Path]:
    """Export every known table found in *data_dir*.

    Defaults the output to ``<data_dir>/iris/``. Returns the written paths in
    table order; tables not present are skipped.

    When *curation* is given (the parsed exclusion table), each table is read into
    a frame, marked by :func:`~cellflow.aggregate_quantification.curation.apply_curation`,
    and its excluded rows dropped by
    :func:`~cellflow.aggregate_quantification.curation.filter_excluded` before the
    ``.iris`` is written — so the bundle sees only kept data. The on-disk tidy
    CSVs are never touched; the bundle's provenance records *curation_path* and how
    many rows were dropped. With no *curation*, the unfiltered table is exported.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir is not None else data_dir / "iris"
    written: list[Path] = []
    for stem in TABLES_TO_EXPORT:
        csv_path = data_dir / f"{stem}.csv"
        if not csv_path.is_file():
            continue
        if curation is None:
            written.append(export_table(csv_path, out_dir))
            continue
        kept, n_dropped = filter_excluded(apply_curation(pd.read_csv(csv_path), curation))
        source = {
            "source_csv": str(csv_path.resolve()),
            "curation": {"file": curation_path, "rows_dropped": n_dropped},
        }
        written.append(export_table_frame(kept, stem, out_dir, source=source))
    return written


def _with_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the stable ``id`` column Iris keys rows on. Iris injects it on load
    when absent, but writing it gives stable row identity across reopen. (No
    ``excluded`` column: Iris no longer has a dedicated exclusion mechanism.)"""
    df = df.copy()
    if "id" not in df.columns:
        df.insert(0, "id", [str(i + 1) for i in range(len(df))])
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
