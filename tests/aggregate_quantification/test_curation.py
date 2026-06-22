"""The curation artifact: human QC exclusions joined onto the measurement tables.

A separate, git-versioned tidy table — ``experiment_id, position_id, frame,
excluded, exclusion_reason`` — authored by hand and kept apart from the disposable
measurement tables. At export it is left-joined by the *natural* keys the tables
already carry: a frame-level exclusion matches ``(experiment_id, position_id,
frame)``; a whole-position exclusion is the ``frame``-is-NA row and matches
``(experiment_id, position_id)``. Rows with no entry default to kept; filter,
don't delete.
"""
from __future__ import annotations

import pandas as pd

from cellflow.aggregate_quantification.curation import (
    apply_curation,
    filter_excluded,
    read_curation,
)


def _table() -> pd.DataFrame:
    # Two positions of one experiment, three frames each.
    rows = []
    for position_id in ("p1", "p2"):
        for frame in (0, 1, 2):
            rows.append({
                "experiment_id": "EXP1",
                "position_id": position_id,
                "frame": frame,
                "cell_shape.area_um2": 10.0 * frame + 1.0,
            })
    return pd.DataFrame(rows)


def test_apply_curation_marks_frame_level_exclusion():
    cur = pd.DataFrame({
        "experiment_id": ["EXP1"],
        "position_id": ["p1"],
        "frame": [1],
        "excluded": [True],
        "exclusion_reason": ["out of focus"],
    })

    out = apply_curation(_table(), cur)

    # Only (EXP1, p1, frame 1) is excluded.
    marked = out[out["excluded"]]
    assert list(zip(marked["position_id"], marked["frame"])) == [("p1", 1)]
    assert list(out[out["excluded"]]["exclusion_reason"]) == ["out of focus"]
    # Everything else kept, no reason.
    assert (out.loc[~out["excluded"], "exclusion_reason"] == "").all()


def test_apply_curation_position_level_excludes_every_frame():
    # frame NA => whole position p2.
    cur = pd.DataFrame({
        "experiment_id": ["EXP1"],
        "position_id": ["p2"],
        "frame": [pd.NA],
        "excluded": [True],
        "exclusion_reason": ["debris"],
    })

    out = apply_curation(_table(), cur)

    excluded = out[out["excluded"]]
    assert set(excluded["position_id"]) == {"p2"}
    assert sorted(excluded["frame"]) == [0, 1, 2]  # all three frames
    assert (excluded["exclusion_reason"] == "debris").all()


def test_apply_curation_none_keeps_everything():
    out = apply_curation(_table(), None)
    assert not out["excluded"].any()
    assert (out["exclusion_reason"] == "").all()


def test_apply_curation_does_not_mutate_input():
    table = _table()
    apply_curation(table, pd.DataFrame({
        "experiment_id": ["EXP1"], "position_id": ["p1"],
        "frame": [0], "excluded": [True], "exclusion_reason": ["x"],
    }))
    assert "excluded" not in table.columns


def test_apply_curation_keys_compared_as_strings():
    # CSV round-trips ids as strings; a numeric-looking id must still match.
    # Single position so frame 0 is unique (matches exactly one row).
    table = pd.DataFrame({
        "experiment_id": ["EXP1", "EXP1", "EXP1"],
        "position_id": ["10", "10", "10"],
        "frame": [0, 1, 2],
        "cell_shape.area_um2": [1.0, 2.0, 3.0],
    })
    cur = pd.DataFrame({
        "experiment_id": ["EXP1"], "position_id": [10],  # int in curation
        "frame": [0], "excluded": [True], "exclusion_reason": ["x"],
    })
    out = apply_curation(table, cur)
    assert out["excluded"].sum() == 1


def test_filter_excluded_drops_marked_rows_and_marker_columns():
    table = _table()
    cur = pd.DataFrame({
        "experiment_id": ["EXP1"], "position_id": ["p1"],
        "frame": [pd.NA], "excluded": [True], "exclusion_reason": ["debris"],
    })
    marked = apply_curation(table, cur)

    kept, dropped = filter_excluded(marked)

    assert dropped == 3  # all of p1
    assert set(kept["position_id"]) == {"p2"}
    assert "excluded" not in kept.columns
    assert "exclusion_reason" not in kept.columns
    # Index is reset so the kept frame is clean.
    assert list(kept.index) == list(range(len(kept)))


def test_filter_excluded_no_marker_column_is_noop():
    table = _table()
    kept, dropped = filter_excluded(table)
    assert dropped == 0
    assert len(kept) == len(table)


def test_read_curation_missing_or_none_is_none(tmp_path):
    assert read_curation(None) is None
    assert read_curation(tmp_path / "nope.csv") is None


def test_read_curation_reads_csv(tmp_path):
    path = tmp_path / "curation.csv"
    pd.DataFrame({
        "experiment_id": ["EXP1"], "position_id": ["p1"],
        "frame": [pd.NA], "excluded": [True], "exclusion_reason": ["debris"],
    }).to_csv(path, index=False)

    cur = read_curation(path)

    assert cur is not None
    assert list(cur["position_id"]) == ["p1"]
    # The empty frame round-trips as NA, not the string "".
    assert cur["frame"].isna().all()
