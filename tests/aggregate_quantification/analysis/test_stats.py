"""Replicate-level analysis statistics: the inference the Iris engine can't do.

These pin the two soundness guarantees: (1) correlation is estimated per replicate
and inferred across replicates (not pooled over pseudoreplicated cells), and
(2) homotypic enrichment is a ratio of pooled counts tested vs chance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cellflow.aggregate_quantification.analysis.stats import (
    homotypic_enrichment,
    replicate_correlation,
)


def _nested_corr_df(slope_sign: int, n_rep=3, n_cell=200, seed=0) -> pd.DataFrame:
    """Cells nested in experiments; within each experiment y depends on x with the
    given sign, plus a big per-experiment offset (the batch effect that pooling
    would mistake for signal)."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_rep):
        offset = 100.0 * e  # between-experiment shift
        for c in range(n_cell):
            x = rng.uniform(0, 10)
            for f in range(2):  # two frames per cell -> exercises the averaging
                y = offset + slope_sign * x + rng.normal(0, 1)
                rows.append({"experiment_id": f"E{e}", "position_id": "p",
                             "cell_id": c, "frame": f, "class_label": "a",
                             "x": x, "y": y})
    return pd.DataFrame(rows)


def test_replicate_correlation_recovers_within_replicate_sign():
    res = replicate_correlation(_nested_corr_df(+1), "x", "y", split=None)
    # one r per replicate, all strongly positive (the within-experiment signal)
    assert len(res.per_replicate) == 3
    assert (res.per_replicate["r"] > 0.7).all()
    mean_r, p = res.by_split["all"]
    assert mean_r > 0.7 and p < 0.05


def test_replicate_correlation_is_not_fooled_by_batch_offsets():
    """Pooling all cells would see the huge between-experiment offsets as a near-
    perfect positive correlation even when the within-replicate slope is negative.
    The two-stage estimate must report the true (negative) within-replicate sign."""
    df = _nested_corr_df(-1)
    pooled = np.corrcoef(df["x"], df["y"])[0, 1]
    res = replicate_correlation(df, "x", "y", split=None)
    mean_r, _ = res.by_split["all"]
    assert mean_r < -0.5          # true within-replicate relationship
    assert mean_r < pooled        # and it disagrees with the naive pooled r


def test_replicate_correlation_split_difference_paired():
    # identical relationship in both classes -> no split difference
    a = _nested_corr_df(+1, seed=1); a["class_label"] = "a"
    b = _nested_corr_df(+1, seed=2); b["class_label"] = "b"
    res = replicate_correlation(pd.concat([a, b]), "x", "y", split="class_label")
    assert set(res.by_split) == {"a", "b"}
    assert res.split_difference is not None
    _, p = res.split_difference
    assert p > 0.05               # classes don't differ


def _enrichment_df(homo_ratio: float, n_rep=3, seed=0) -> pd.DataFrame:
    """Synthetic contact-type counts: homotypic types over-represented vs the null
    by `homo_ratio` (with small per-replicate jitter), heterotypic depleted."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_rep):
        base = 1000
        j = 1.0 if homo_ratio == 1.0 else 1.0 + rng.normal(0, 0.01)
        rows += [
            {"experiment_id": f"E{e}", "type": "a·a",
             "obs": base * homo_ratio * j, "null": base},
            {"experiment_id": f"E{e}", "type": "b·b",
             "obs": base * homo_ratio * j, "null": base},
            {"experiment_id": f"E{e}", "type": "a·b",
             "obs": base * (2 - homo_ratio) * j, "null": base},
        ]
    return pd.DataFrame(rows)


def test_homotypic_enrichment_detects_sorting():
    res = homotypic_enrichment(_enrichment_df(1.06), type_col="type",
                               observed="obs", expected="null")
    ratio, p = res.homotypic
    assert ratio > 1.0 and p < 0.05            # sorting, significant
    het_ratio, _ = res.heterotypic
    assert het_ratio < 1.0                     # mixing depleted
    assert res.by_type["a·a"][0] > 1.0 and res.by_type["b·b"][0] > 1.0


def test_homotypic_enrichment_null_when_random():
    res = homotypic_enrichment(_enrichment_df(1.0), type_col="type",
                               observed="obs", expected="null")
    ratio, _ = res.homotypic
    assert abs(ratio - 1.0) < 1e-9             # exactly chance
