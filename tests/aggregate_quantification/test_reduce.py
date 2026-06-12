"""The reduce layer: filter / collapse primitives and the pipeline runner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellflow.aggregate_quantification.plotting import PlotSpec, reduce_to_units
from cellflow.aggregate_quantification.reduce import (
    Collapse,
    Filter,
    run_pipeline,
    unit_collapse_chain,
)


def _pooled() -> pd.DataFrame:
    """A small per-frame pooled table: 2 conditions × 1 position × 2 cells × 2
    frames, plus a per-cell ``class_label`` attribute."""
    rows = []
    rng = np.random.default_rng(0)
    for condition in ("ctrl", "drug"):
        for cell_id in (1, 2):
            label = "epithelial" if cell_id == 1 else "mesenchymal"
            for frame in (0, 1):
                rows.append(
                    {
                        "condition": condition,
                        "date": "d1",
                        "position_id": "p1",
                        "cell_id": cell_id,
                        "frame": frame,
                        "class_label": label,
                        "value": float(rng.uniform(0, 10)),
                    }
                )
    return pd.DataFrame(rows)


def test_empty_pipeline_is_identity():
    df = _pooled()
    out = run_pipeline(df, [])
    pd.testing.assert_frame_equal(out, df)


def test_filter_keeps_only_matching_rows():
    df = _pooled()
    out = run_pipeline(df, [Filter("class_label", "==", "epithelial")])
    assert set(out["class_label"]) == {"epithelial"}
    assert len(out) == 4  # 2 conditions × 1 cell × 2 frames


def test_filter_ordered_numeric():
    df = _pooled()
    out = run_pipeline(df, [Filter("frame", ">=", 1)])
    assert set(out["frame"]) == {1}


def test_filter_not_equal_and_missing_column_noop():
    df = _pooled()
    kept = run_pipeline(df, [Filter("class_label", "!=", "epithelial")])
    assert set(kept["class_label"]) == {"mesenchymal"}
    # A filter on a column the table lacks never narrows (and never raises).
    pd.testing.assert_frame_equal(run_pipeline(df, [Filter("nope", "==", "x")]), df)


def test_single_collapse_is_flat_group_by():
    df = _pooled()
    out = run_pipeline(df, [Collapse(by=("condition",), stat="mean")])
    assert len(out) == 2
    for condition in ("ctrl", "drug"):
        expected = df.loc[df["condition"] == condition, "value"].mean()
        got = out.loc[out["condition"] == condition, "value"].iloc[0]
        assert got == pytest.approx(expected)


def test_collapse_keeps_constant_attribute_drops_varying():
    df = _pooled()
    # Collapse to one row per cell: class_label is constant within a cell (kept);
    # condition varies across the cells we group only by cell_id, so it is dropped.
    out = run_pipeline(df, [Collapse(by=("condition", "cell_id"), stat="mean")])
    assert "class_label" in out.columns  # constant within (condition, cell_id)
    # frame is an identity key outside ``by`` → collapsed away, never averaged.
    assert "frame" not in out.columns

    varying = run_pipeline(df, [Collapse(by=("cell_id",), stat="mean")])
    assert "condition" not in varying.columns  # varies within a cell_id → dropped
    assert "class_label" in varying.columns  # still constant within a cell


def test_count_stat_reports_group_size():
    df = _pooled()
    out = run_pipeline(df, [Collapse(by=("condition",), stat="count")])
    assert set(out["n"]) == {4}


def test_step_order_matters():
    df = _pooled()
    filter_first = run_pipeline(
        df, [Filter("frame", "==", 0), Collapse(by=("condition",), stat="count")]
    )
    collapse_first = run_pipeline(
        df, [Collapse(by=("condition",), stat="count"), Filter("frame", "==", 0)]
    )
    # Filtering before counting halves the tally; collapsing first drops ``frame``
    # so the later filter is a no-op on the (frame-less) summary.
    assert set(filter_first["n"]) == {2}
    assert set(collapse_first["n"]) == {4}


@pytest.mark.parametrize("level", ["cell", "position", "date"])
@pytest.mark.parametrize("stat", ["mean", "median"])
def test_chained_collapse_matches_reduce_to_units(level, stat):
    """A chained single-rung collapse reproduces the level machinery's
    equal-weighted nested reduction (golden compare vs ``reduce_to_units``)."""
    df = _pooled()
    group = ("condition",)
    spec = PlotSpec(value="value", group_by=group, level=level, stat=stat)
    golden = reduce_to_units(df, spec)

    chain = unit_collapse_chain(("date", "position_id", "cell_id"), group, level, stat)
    got = run_pipeline(df, chain)

    keys = [c for c in ("condition", "date", "position_id", "cell_id") if c in golden.columns]
    a = golden[[*keys, "value"]].sort_values(keys).reset_index(drop=True)
    b = got[[*keys, "value"]].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
