"""Authoring ops for the curation table — the write side the curation tool needs.

The read/apply/filter side is covered in test_curation.py; these cover building
and editing the tidy exclusion table: append (frame-level and whole-position),
idempotent re-exclude, remove, and the CSV round-trip.
"""
from __future__ import annotations

import pandas as pd

from cellflow.aggregate_quantification.curation import (
    CURATION_COLUMNS,
    append_exclusion,
    empty_curation,
    read_curation,
    remove_exclusion,
    write_curation,
)


def test_empty_curation_has_schema_and_no_rows():
    cur = empty_curation()
    assert tuple(cur.columns) == CURATION_COLUMNS
    assert len(cur) == 0


def test_append_frame_level_exclusion():
    cur = append_exclusion(
        empty_curation(),
        experiment_id="EXP1", position_id="p1", frame=3, reason="out of focus",
    )
    assert len(cur) == 1
    row = cur.iloc[0]
    assert row["experiment_id"] == "EXP1"
    assert row["position_id"] == "p1"
    assert int(row["frame"]) == 3
    assert bool(row["excluded"]) is True
    assert row["exclusion_reason"] == "out of focus"


def test_append_position_level_exclusion_stores_frame_na():
    cur = append_exclusion(
        empty_curation(),
        experiment_id="EXP1", position_id="p2", frame=None, reason="debris",
    )
    assert len(cur) == 1
    assert pd.isna(cur.iloc[0]["frame"])
    assert cur.iloc[0]["exclusion_reason"] == "debris"


def test_append_is_idempotent_on_key_updating_reason():
    cur = append_exclusion(empty_curation(),
                           experiment_id="EXP1", position_id="p1", frame=3, reason="first")
    cur = append_exclusion(cur,
                           experiment_id="EXP1", position_id="p1", frame=3, reason="second")
    assert len(cur) == 1
    assert cur.iloc[0]["exclusion_reason"] == "second"


def test_append_frame_and_position_level_coexist():
    cur = append_exclusion(empty_curation(),
                           experiment_id="EXP1", position_id="p1", frame=3, reason="a")
    cur = append_exclusion(cur,
                           experiment_id="EXP1", position_id="p1", frame=None, reason="b")
    assert len(cur) == 2


def test_append_does_not_mutate_input():
    base = empty_curation()
    append_exclusion(base, experiment_id="EXP1", position_id="p1", frame=1, reason="x")
    assert len(base) == 0


def test_remove_frame_level_exclusion():
    cur = append_exclusion(empty_curation(),
                           experiment_id="EXP1", position_id="p1", frame=3, reason="a")
    cur = append_exclusion(cur,
                           experiment_id="EXP1", position_id="p1", frame=4, reason="b")
    out = remove_exclusion(cur, experiment_id="EXP1", position_id="p1", frame=3)
    assert len(out) == 1
    assert int(out.iloc[0]["frame"]) == 4


def test_remove_position_level_exclusion():
    cur = append_exclusion(empty_curation(),
                           experiment_id="EXP1", position_id="p1", frame=None, reason="a")
    cur = append_exclusion(cur,
                           experiment_id="EXP1", position_id="p1", frame=2, reason="b")
    out = remove_exclusion(cur, experiment_id="EXP1", position_id="p1", frame=None)
    assert len(out) == 1
    assert int(out.iloc[0]["frame"]) == 2


def test_remove_missing_key_is_noop():
    cur = append_exclusion(empty_curation(),
                           experiment_id="EXP1", position_id="p1", frame=3, reason="a")
    out = remove_exclusion(cur, experiment_id="EXP1", position_id="zzz", frame=3)
    assert len(out) == 1


def test_write_then_read_round_trips(tmp_path):
    cur = append_exclusion(empty_curation(),
                           experiment_id="EXP1", position_id="p1", frame=None, reason="debris")
    cur = append_exclusion(cur,
                           experiment_id="EXP1", position_id="p2", frame=5, reason="blurry")
    path = tmp_path / "curation.csv"
    write_curation(path, cur)
    back = read_curation(path)
    assert back is not None
    assert set(back["position_id"].astype(str)) == {"p1", "p2"}
    p1 = back[back["position_id"].astype(str) == "p1"].iloc[0]
    assert pd.isna(p1["frame"])


def test_write_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "curation.csv"
    write_curation(path, empty_curation())
    assert path.is_file()
