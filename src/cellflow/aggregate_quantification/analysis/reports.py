"""Analyst-driven reports over a CellFlow ``aggregate_quantification/`` directory.

Each report reads the already-aggregated tidy CSVs, runs the replicate-level stats
(:mod:`.stats`), and writes figures (:mod:`.figures`) plus, optionally, an ``.iris``
doc for interactive viewing. Column names default to CellFlow's standard schema, so
pointing a report at a *new* experiment's aggregate dir reproduces the analysis.

The ``.iris`` docs carry a provenance **caveat**: their engine-computed p is pooled
(pseudoreplicated), so the valid inference is the replicate-level figure — never the
number the GUI prints. The figures are the scientific deliverable.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..iris_export.document import write_iris
from ..iris_export.export import _EXPORTER, _cellflow_version, _with_meta_columns
from ..iris_export.schema import infer_schema
from . import figures, stats

# Standard tidy-schema keys shared by every per-object table.
_CELL_KEYS = ("experiment_id", "position_id", "cell_id")
_FRAME_KEYS = ("experiment_id", "position_id", "frame", "cell_id")


def _table_of(column: str) -> str:
    """The aggregate table a prefixed value column lives in (``a.b`` -> ``a``)."""
    return column.split(".")[0]


def _merge_values(aggregate_dir: Path, value_cols: list[str], *,
                  split: str | None) -> pd.DataFrame:
    """Per-(cell,frame) frame carrying *value_cols* (each from its own table),
    merged on the frame keys. The *split* column (shared by every table) is taken
    from the first table only, so the merge never duplicates it."""
    merged: pd.DataFrame | None = None
    for table in dict.fromkeys(_table_of(c) for c in value_cols):
        df = pd.read_csv(aggregate_dir / f"{table}.csv")
        cols = list(_FRAME_KEYS) + [c for c in value_cols if _table_of(c) == table]
        if merged is None and split and split in df.columns:
            cols.append(split)          # split carried once, from the first table
        part = df[[c for c in cols if c in df.columns]]
        merged = part if merged is None else merged.merge(part, on=list(_FRAME_KEYS))
    return merged


def metric_correlation_report(
    aggregate_dir: Path | str,
    out_dir: Path | str,
    *,
    x: str,
    y: str,
    split: str | None = "class_label",
    replicate: str = "experiment_id",
    method: str = "spearman",
    x_label: str | None = None,
    title: str | None = None,
    stem: str | None = None,
    write_iris_doc: bool = True,
) -> stats.CorrelationResult:
    """Does *y* depend on *x*, and does the relationship differ by *split*?

    Two-stage replicate-level correlation + a figure; optionally an ``.iris`` of the
    per-cell scatter (all / per-split) for interactive viewing.
    """
    aggregate_dir, out_dir = Path(aggregate_dir), Path(out_dir)
    df = _merge_values(aggregate_dir, [x, y], split=split)
    res = stats.replicate_correlation(df, x, y, replicate=replicate, split=split,
                                      cell_keys=_CELL_KEYS, method=method)

    stem = stem or f"{_leaf(y)}_vs_{_leaf(x)}"
    figures.correlation_by_split(
        res, out_dir / "figures" / "analysis" / stem,
        x_label=x_label or f"{_leaf(y)} vs {_leaf(x)}",
        title=title or f"{_leaf(y)} vs {_leaf(x)} — per-replicate, by {split}")

    if write_iris_doc:
        _write_scatter_iris(df, x, y, split, out_dir / "iris" / f"{stem}.iris",
                            title or f"{_leaf(y)} vs {_leaf(x)}")
    return res


def label_clustering_report(
    aggregate_dir: Path | str,
    out_dir: Path | str,
    *,
    replicate: str = "experiment_id",
    write_iris_doc: bool = True,
) -> dict[str, stats.EnrichmentResult]:
    """Do cells cluster by label? Homotypic contact enrichment vs chance, from the
    permutation null (``contact_type_zscore``) and — when present — corroborated by
    the analytic null (``neighbor_enrichment``).
    """
    aggregate_dir, out_dir = Path(aggregate_dir), Path(out_dir)
    fig_dir = out_dir / "figures" / "analysis"
    results: dict[str, stats.EnrichmentResult] = {}

    ctz_path = aggregate_dir / "contact_type_zscore.csv"
    ctz = pd.read_csv(ctz_path)
    perm = stats.homotypic_enrichment(
        ctz, type_col="contact_type",
        observed="contact_type_zscore.observed_count",
        expected="contact_type_zscore.mean_null", replicate=replicate)
    results["permutation null"] = perm
    type_order, type_labels = _contact_type_order(perm.per_replicate["contact_type"])
    figures.enrichment_vs_chance(
        perm, fig_dir / "label_clustering", type_order=type_order,
        type_labels=type_labels, type_col="contact_type", replicate_col=replicate,
        title="Do cells cluster by label?\nhomotypic contacts vs a label-shuffle null")

    ne_path = aggregate_dir / "neighbor_enrichment.csv"
    if ne_path.is_file():
        ne = pd.read_csv(ne_path)
        ne["contact_type"] = ne["focal_label"] + "·" + ne["neighbor_label"]
        analytic = stats.homotypic_enrichment(
            ne, type_col="contact_type",
            observed="neighbor_enrichment.observed",
            expected="neighbor_enrichment.expected", replicate=replicate)
        results["analytic null"] = analytic
        collapsed = {k: _collapse_heterotypic(v) for k, v in results.items()}
        type_order, type_labels = _collapsed_type_order(collapsed)
        figures.enrichment_corroboration(
            collapsed, fig_dir / "label_clustering_corroboration",
            type_order=type_order, type_labels=type_labels,
            type_col="contact_type", replicate_col=replicate,
            title="Homotypic clustering is robust to the null model")

    if write_iris_doc:
        _write_clustering_iris(ctz, out_dir / "iris" / "label_clustering.iris",
                               replicate=replicate)
    return results


# ---------------------------------------------------------------- helpers / .iris


def _leaf(col: str) -> str:
    return col.split(".")[-1] if "." in col else col


def _contact_type_order(series: pd.Series):
    homo = sorted(t for t in series.unique() if _is_homo(t))
    het = sorted(t for t in series.unique() if not _is_homo(t))
    order = homo + het
    labels = {t: t.replace("·", "\n–") for t in order}
    return order, labels


def _is_homo(t: str) -> bool:
    parts = str(t).split("·")
    return len(parts) == 2 and parts[0] == parts[1]


def _collapsed_type_order(collapsed: dict[str, stats.EnrichmentResult]):
    """Axis order/labels for the corroboration figure: the homotypic types present
    in the (heterotypic-pooled) data, sorted, then ``heterotypic`` — derived from
    the data so it works for any class labels, not just a fixed pair."""
    any_rep = next(iter(collapsed.values())).per_replicate
    homo = sorted(t for t in any_rep["contact_type"].unique() if _is_homo(t))
    order = homo + ["heterotypic"]
    labels = {t: t.replace("·", "\n–") for t in homo}
    labels["heterotypic"] = "hetero-\ntypic"
    return order, labels


def _collapse_heterotypic(res: stats.EnrichmentResult) -> stats.EnrichmentResult:
    """Recompute a per-replicate frame with the two heterotypic directions pooled
    into a single ``heterotypic`` type, for the corroboration figure."""
    df = res.per_replicate.copy()
    df["contact_type"] = df["contact_type"].where(df["contact_type"].map(_is_homo),
                                                   "heterotypic")
    obs = [c for c in df.columns if c.endswith("observed") or c.endswith("observed_count")]
    exp = [c for c in df.columns if c.endswith("expected") or c.endswith("mean_null")]
    rep = df.columns[0]
    agg = (df.groupby([rep, "contact_type"], observed=True)[obs + exp].sum()
             .reset_index())
    agg["enrichment"] = agg[obs[0]] / agg[exp[0]]
    by_type = {}
    for t, sub in agg.groupby("contact_type", observed=True):
        by_type[str(t)] = stats._logratio_one_sample(sub["enrichment"].to_numpy())
    return stats.EnrichmentResult(agg, by_type, res.homotypic, res.heterotypic)


def _write_scatter_iris(df, x, y, split, out_path: Path, title: str) -> None:
    cell = (df.groupby([*_CELL_KEYS, *( [split] if split else [])], observed=True)
              [[x, y]].mean().reset_index().dropna(subset=[x, y]))
    # carry the replicate/split for the scatter; experiment_id already in cell keys
    tidy = _with_meta_columns(cell)
    schema = infer_schema(tidy)

    def scatter_spec(sid, ttl, level):
        spec = {"spec_version": "2.0", "id": sid, "title": ttl,
                "encodings": {"x": {"column": x}, "y": {"column": y},
                              "color": None, "size": None, "shape": None},
                "layers": [{"geom": "scatter", "params": {}},
                           {"geom": "regression", "params": {}}],
                "stats": {"alpha": 0.05}}
        if level and split:
            spec["reduce"] = {"steps": [{"kind": "filter", "conditions": [
                {"column": split, "op": "in", "value": [level]}]}]}
        return spec

    analyses = [scatter_spec(f"{_leaf(y)}_vs_{_leaf(x)}__all", f"{title} — all", None)]
    if split:
        for lv in sorted(cell[split].dropna().unique()):
            analyses.append(scatter_spec(f"{_leaf(y)}_vs_{_leaf(x)}__{lv}",
                                         f"{title} — {lv}", lv))
    prov = {"exporter": _EXPORTER, "analysis": f"{_leaf(y)}-vs-{_leaf(x)}",
            "grain": "one row per cell (frames averaged)",
            "caveat": ("EXPLORATORY scatter. The pooled per-cell p is "
                       "pseudoreplicated; valid inference is the replicate-level "
                       "correlation figure (see figures/analysis).")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(write_iris(tidy, schema, analyses, prov,
                         engine_snapshot={"producer": f"cellflow {_cellflow_version()}"}))


def _write_clustering_iris(ctz: pd.DataFrame, out_path: Path, *, replicate: str) -> None:
    z = "contact_type_zscore.z_score"
    tidy = _with_meta_columns(ctz.dropna(subset=[z]))
    schema = infer_schema(tidy)
    spec = {"spec_version": "2.0", "id": "clustering__z_by_contact_type",
            "title": "Label clustering — contact z-score by type (>0 = sorting)",
            "encodings": {"x": {"column": "contact_type"}, "y": {"column": z},
                          "color": {"column": replicate},
                          "shape": {"column": replicate}, "size": None},
            "hierarchy": {"spine": ["position_id", "frame"], "fn": {}},
            "layers": [{"geom": "violin", "level": "frame", "params": {}},
                       {"geom": "dot", "level": "", "params": {"layout": "swarm"}}],
            "stats": {}, "_describe_only": True}
    prov = {"exporter": _EXPORTER, "analysis": "label-clustering",
            "note": ("Homotypic z>0 = sorting, heterotypic <0 = mixing. Rigorous "
                     "replicate-level test in the label_clustering figures.")}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(write_iris(tidy, schema, [spec], prov,
                         engine_snapshot={"producer": f"cellflow {_cellflow_version()}"}))
