"""Reproduce the recorded analyses for a dataset.

    python -m cellflow.aggregate_quantification.analysis AGG_DIR [OUT_DIR]

AGG_DIR is an ``aggregate_quantification/`` directory of tidy CSVs; OUT_DIR defaults
to its parent's ``export/``. Writes figures under ``OUT_DIR/figures/analysis/`` and
``.iris`` docs under ``OUT_DIR/iris/``. Reports whose source tables are missing are
skipped with a note, so it runs on any dataset that has the relevant quantities.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .reports import label_clustering_report, metric_correlation_report

#: The standard correlations to attempt (skipped when a column's table is absent).
_CORRELATIONS = [
    {"x": "neighbor_count.n_neighbors", "y": "cell_dynamics.speed_um_per_s",
     "x_label": "speed  vs  neighbour count",
     "title": "Denser neighbourhoods → slower cells\nper-replicate correlation, by class"},
]


def _has(agg: Path, *columns: str) -> bool:
    return all((agg / f"{c.split('.')[0]}.csv").is_file() for c in columns)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agg_dir", type=Path, help="aggregate_quantification/ directory")
    ap.add_argument("out_dir", type=Path, nargs="?", default=None,
                    help="output dir (default: AGG_DIR/../export)")
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

    for spec in _CORRELATIONS:
        if _has(agg, spec["x"], spec["y"]):
            r = metric_correlation_report(agg, out, **spec)
            did.append(f"correlation {spec['y'].split('.')[-1]} vs "
                       f"{spec['x'].split('.')[-1]} "
                       f"({', '.join(f'{k}:r={v[0]:+.2f}' for k, v in r.by_split.items())})")
        else:
            print(f"skip correlation {spec['y']} vs {spec['x']}: table(s) missing")

    print(f"\nwrote {len(did)} report(s) to {out}:")
    for d in did:
        print("  •", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
