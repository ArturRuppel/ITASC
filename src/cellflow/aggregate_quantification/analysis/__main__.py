"""Run the analyst-driven reports over a dataset's aggregate tables.

    python -m cellflow.aggregate_quantification.analysis AGG_DIR [OUT_DIR] \
        [--correlate Y X ...]

AGG_DIR is an ``aggregate_quantification/`` directory of tidy CSVs; OUT_DIR defaults
to its parent's ``export/``. Writes figures under ``OUT_DIR/figures/analysis/`` and
``.iris`` docs under ``OUT_DIR/iris/``.

``label_clustering`` runs automatically whenever ``contact_type_zscore.csv`` is
present. Metric correlations are opt-in: pass ``--correlate Y X`` (repeatable, dotted
``table.column`` names) for each question you want — e.g.
``--correlate cell_dynamics.speed_um_per_s neighbor_count.n_neighbors``. Reports whose
source tables are missing are skipped with a note. The package ships *no* dataset's
questions baked in; a specific experiment's analysis is a small driver that imports
:mod:`cellflow.aggregate_quantification.analysis` and calls the report functions
directly (so it can pass bespoke titles/labels).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .reports import label_clustering_report, metric_correlation_report


def _has(agg: Path, *columns: str) -> bool:
    return all((agg / f"{c.split('.')[0]}.csv").is_file() for c in columns)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agg_dir", type=Path, help="aggregate_quantification/ directory")
    ap.add_argument("out_dir", type=Path, nargs="?", default=None,
                    help="output dir (default: AGG_DIR/../export)")
    ap.add_argument("--correlate", nargs=2, action="append", default=[],
                    metavar=("Y", "X"),
                    help="replicate-level correlation of Y on X (dotted "
                         "table.column names); repeatable")
    args = ap.parse_args(argv)
    agg = args.agg_dir
    out = args.out_dir or agg.parent / "export"

    did = []
    if (agg / "contact_type_zscore.csv").is_file():
        res = label_clustering_report(agg, out)
        homo, p = res["permutation null"].homotypic
        did.append(f"label_clustering (homotypic {homo:.2f}x, p={p:.3f}, "
                   f"{len(res)} null model(s))")
    else:
        print("skip label_clustering: contact_type_zscore.csv not found")

    for y, x in args.correlate:
        if _has(agg, x, y):
            r = metric_correlation_report(agg, out, x=x, y=y)
            did.append(f"correlation {y.split('.')[-1]} vs {x.split('.')[-1]} "
                       f"({', '.join(f'{k}:r={v[0]:+.2f}' for k, v in r.by_split.items())})")
        else:
            print(f"skip correlation {y} vs {x}: table(s) missing")

    print(f"\nwrote {len(did)} report(s) to {out}:")
    for d in did:
        print("  •", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
