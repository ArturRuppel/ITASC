"""The reduce layer: filter / collapse primitives and the pipeline runner."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cellflow.aggregate_quantification.plotting import PlotSpec, reduce_to_units
from cellflow.aggregate_quantification.reduce import (
    Collapse,
    Filter,
    _is_numeric,
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


def test_is_numeric_requires_all_values_numeric():
    """A mostly-categorical column with a single parseable value must not be
    treated as numeric (and silently averaged)."""
    assert _is_numeric(pd.Series([1.0, 2.0, 3.0]))
    assert _is_numeric(pd.Series([1.0, np.nan, 3.0]))  # NaN ignored
    assert not _is_numeric(pd.Series(["a", "b", "5"]))  # one parseable → still categorical
    assert not _is_numeric(pd.Series(["epithelial", "mesenchymal"]))


def test_collapse_does_not_average_mostly_categorical_column():
    """A class-like column where one label happens to be '5' must ride along as
    an attribute, not be coerced and averaged into a value column."""
    df = _pooled().copy()
    # Replace class_label with a column that has one numeric-looking value.
    df["grade"] = ["5" if i == 0 else "high" for i in range(len(df))]
    out = run_pipeline(df, [Collapse(by=("condition",), stat="mean")])
    # 'grade' is not constant within condition → dropped as a varying attribute,
    # and must NOT appear as an averaged numeric value column.
    assert "grade" not in out.columns or out["grade"].dtype == object


def test_filter_ordered_on_non_numeric_column_is_noop():
    """An ordered comparison against a fully non-numeric column is a config
    error; keep all rows (no-op) rather than silently dropping every row."""
    df = _pooled()
    out = run_pipeline(df, [Filter("class_label", ">", 0)])
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df.reset_index(drop=True))


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


def test_experiment_id_is_identity_not_a_value():
    """experiment_id is a catalogue identity axis: collapsing by it keeps it as an
    index column and never averages it (it would be meaningless to)."""
    df = _pooled()
    df["experiment_id"] = "E1"
    out = run_pipeline(df, [Collapse(by=("condition", "experiment_id"), stat="mean")])
    assert "experiment_id" in out.columns
    assert (out["experiment_id"] == "E1").all()
    # The value column is the only thing averaged; the identity axis is untouched.
    assert "value" in out.columns


def test_count_stat_reports_group_size():
    df = _pooled()
    out = run_pipeline(df, [Collapse(by=("condition",), stat="count")])
    assert set(out["n"]) == {4}


def test_mean_collapse_attaches_group_size_n():
    df = _pooled()
    # Every collapse (not just ``count``) carries ``n`` = the current group size, so
    # an ``n``-threshold filter works after a mean/median collapse too.
    out = run_pipeline(df, [Collapse(by=("condition",), stat="mean")])
    assert set(out["n"]) == {4}  # 1 position × 2 cells × 2 frames per condition
    # Whole-table (no ``by``) collapse → n = len(df).
    whole = run_pipeline(df, [Collapse(by=(), stat="median")])
    assert int(whole["n"].iloc[0]) == len(df)


def test_chained_collapse_recomputes_n_as_child_count():
    df = _pooled()
    # cell then position: ``n`` is recomputed to the per-position child (cell) count,
    # never the mean of the per-cell frame counts (which a reserved ``n`` would give).
    out = run_pipeline(df, [
        Collapse(by=("condition", "position_id", "cell_id"), stat="mean"),
        Collapse(by=("condition", "position_id"), stat="mean"),
    ])
    assert set(out["n"]) == {2}  # each (condition, position) pools 2 cells


def test_collapse_does_not_average_a_reserved_n_column():
    df = _pooled().assign(n=99)  # a pre-existing ``n`` must be overwritten, not averaged
    out = run_pipeline(df, [Collapse(by=("condition",), stat="mean")])
    assert set(out["n"]) == {4}  # recomputed group size, not the mean of 99


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
