"""Replicate-level statistics for analyst-driven questions over the aggregate
tables — the inference the Iris engine's per-figure stats cannot express.

Both helpers honour the SuperPlots principle: the **biological replicate** is the
unit of inference (one summary per replicate, then a small-N test across them), so
the thousands of correlated cell-frames never inflate significance. Defaults match
CellFlow's standard tidy schema (``experiment_id`` replicates, ``class_label``
classifier), so pointing these at a new experiment's aggregate dir just works.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as ss


@dataclass(frozen=True)
class CorrelationResult:
    """Two-stage correlation of *y* on *x*.

    *per_replicate* is a tidy frame (one row per replicate, or per replicate ×
    split level) of the within-replicate correlation ``r``. *by_split* maps each
    split level to ``(mean_r, p)`` from a Fisher-z one-sample test vs 0 across
    replicates. *split_difference* (when a split is given) is the paired
    ``(delta_r, p)`` between the two split levels across replicates.
    """

    per_replicate: pd.DataFrame
    by_split: dict[str, tuple[float, float]]
    method: str
    split_difference: tuple[float, float] | None = None


def _fisher_one_sample(rs: np.ndarray) -> tuple[float, float]:
    """Mean r (back-transformed) and one-sample p that the correlation differs
    from zero, computed on Fisher-z-transformed r's across replicates."""
    z = np.arctanh(np.clip(rs, -0.999999, 0.999999))
    if len(z) < 2:
        return float(np.tanh(z.mean())) if len(z) else float("nan"), float("nan")
    t, p = ss.ttest_1samp(z, 0.0)
    return float(np.tanh(z.mean())), float(p)


def replicate_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    replicate: str = "experiment_id",
    split: str | None = "class_label",
    cell_keys: tuple[str, ...] = ("experiment_id", "position_id", "cell_id"),
    method: str = "spearman",
) -> CorrelationResult:
    """Does *y* depend on *x*? — answered the sound way for nested cell data.

    Stage 1 collapses temporal autocorrelation: average each cell over its frames
    (group by *cell_keys*). Stage 2 estimates the *method* correlation
    (``spearman`` | ``pearson``) within each *replicate* (and each *split* level,
    if given) — thousands of cells, so well-determined. Stage 3 infers at the
    replicate level: a Fisher-z one-sample test that the correlation is non-zero,
    and (with a split) a paired test of whether it differs between split levels.
    """
    corr = ss.spearmanr if method == "spearman" else _pearson
    # Stage 1 — one row per cell. Group by the cell identity plus the replicate and
    # split so both stay available as columns (dedup keeps it clean when, as usual,
    # the replicate is itself one of the cell keys).
    keys1 = list(dict.fromkeys(
        list(cell_keys) + [replicate] + ([split] if split else [])))
    cell = (df.groupby(keys1, observed=True)[[x, y]]
              .mean().reset_index().dropna(subset=[x, y]))

    group_cols = [replicate] + ([split] if split else [])
    rows = []
    for keys, g in cell.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        r = corr(g[x].to_numpy(), g[y].to_numpy())[0] if len(g) > 2 else np.nan
        rows.append((*keys, float(r), int(len(g))))
    per_rep = pd.DataFrame(rows, columns=group_cols + ["r", "n"])

    by_split: dict[str, tuple[float, float]] = {}
    if split:
        for level, sub in per_rep.groupby(split, observed=True):
            by_split[str(level)] = _fisher_one_sample(sub["r"].dropna().to_numpy())
    else:
        by_split["all"] = _fisher_one_sample(per_rep["r"].dropna().to_numpy())

    split_difference = None
    if split:
        wide = per_rep.pivot(index=replicate, columns=split, values="r").dropna()
        levels = list(wide.columns)
        if len(levels) == 2 and len(wide) >= 2:
            a, b = (np.arctanh(np.clip(wide[lv].to_numpy(), -0.999999, 0.999999))
                    for lv in levels)
            t, p = ss.ttest_rel(a, b)
            split_difference = (float(np.tanh(a.mean()) - np.tanh(b.mean())), float(p))

    return CorrelationResult(per_rep, by_split, method, split_difference)


def _pearson(a, b):
    return ss.pearsonr(a, b)


@dataclass(frozen=True)
class EnrichmentResult:
    """Per-replicate observed/expected enrichment per contact type, with a
    one-sample log-ratio test vs chance (ratio = 1) for each type and for the
    pooled homotypic class."""

    per_replicate: pd.DataFrame
    by_type: dict[str, tuple[float, float]]
    homotypic: tuple[float, float]
    heterotypic: tuple[float, float] = field(default=(float("nan"), float("nan")))


def _logratio_one_sample(ratios: np.ndarray) -> tuple[float, float]:
    z = np.log2(ratios)
    if len(z) < 2:
        return float(2 ** z.mean()) if len(z) else float("nan"), float("nan")
    t, p = ss.ttest_1samp(z, 0.0)
    return float(2 ** z.mean()), float(p)


def homotypic_enrichment(
    df: pd.DataFrame,
    *,
    type_col: str,
    observed: str,
    expected: str,
    replicate: str = "experiment_id",
    homotypic_types: tuple[str, ...] | None = None,
) -> EnrichmentResult:
    """Do cells cluster by label? — homotypic contact enrichment vs chance.

    Pools *observed* and *expected* counts per ``(replicate, type)`` (a ratio of
    sums, never an average of ratios), giving one enrichment per replicate per
    contact type. Tests ``log2(observed/expected)`` against 0 (chance) one-sample
    across replicates, per type and for the pooled homotypic class. *expected* is
    whatever null the source table carries (the permutation ``mean_null`` of
    ``contact_type_zscore`` or the analytic ``expected`` of ``neighbor_enrichment``),
    so the same function corroborates a result across independent nulls.

    *homotypic_types* names the same-label types (e.g. ``("aa", "bb")``); when
    omitted, a type is homotypic iff its two sides — split on ``·`` — are equal.
    """
    agg = df.groupby([replicate, type_col], observed=True)[[observed, expected]].sum()
    agg = agg.reset_index()
    agg["enrichment"] = agg[observed] / agg[expected]

    def _is_homo(t: str) -> bool:
        if homotypic_types is not None:
            return t in homotypic_types
        parts = str(t).split("·")
        return len(parts) == 2 and parts[0] == parts[1]

    agg["kind"] = np.where(agg[type_col].map(_is_homo), "homotypic", "heterotypic")

    by_type = {str(t): _logratio_one_sample(sub["enrichment"].to_numpy())
               for t, sub in agg.groupby(type_col, observed=True)}

    def _pooled(kind: str) -> tuple[float, float]:
        sub = agg[agg["kind"] == kind]
        if sub.empty:
            return float("nan"), float("nan")
        per_rep = (sub.groupby(replicate, observed=True)
                      .apply(lambda x: x[observed].sum() / x[expected].sum(),
                             include_groups=False))
        return _logratio_one_sample(per_rep.to_numpy())

    return EnrichmentResult(agg, by_type, _pooled("homotypic"), _pooled("heterotypic"))
