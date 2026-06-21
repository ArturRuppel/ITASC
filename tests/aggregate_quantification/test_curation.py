"""The curation join: human QC decisions left-joined into the export by ``id``.

Curation is the third artifact — ``id, excluded, qc_reason`` — git-versioned and
authored by hand, kept *separate* from the disposable measurement tables. At
export it is left-joined onto a table by the deterministic row ``id`` (the same
mechanism as the NLS class_label join), never mutating the measurement source.
"""
from __future__ import annotations

import pandas as pd

from cellflow.aggregate_quantification.curation import apply_curation, read_curation


def _table():
    return pd.DataFrame(
        {"id": ["r1", "r2", "r3"], "cell_shape.area_um2": [10.0, 20.0, 30.0]}
    )


def test_apply_curation_left_joins_excluded_and_reason():
    cur = pd.DataFrame(
        {"id": ["r2"], "excluded": [True], "qc_reason": ["debris"]}
    )

    out = apply_curation(_table(), cur)

    assert list(out["id"]) == ["r1", "r2", "r3"]
    # Rows absent from curation default to kept, no reason.
    assert list(out["excluded"]) == [False, True, False]
    assert list(out["qc_reason"]) == ["", "debris", ""]
    # Measurement columns pass through untouched.
    assert list(out["cell_shape.area_um2"]) == [10.0, 20.0, 30.0]


def test_apply_curation_none_defaults_all_kept():
    out = apply_curation(_table(), None)

    assert list(out["excluded"]) == [False, False, False]
    assert list(out["qc_reason"]) == ["", "", ""]


def test_apply_curation_does_not_mutate_input():
    table = _table()
    apply_curation(table, None)
    assert "excluded" not in table.columns


def test_read_curation_missing_file_is_none(tmp_path):
    assert read_curation(tmp_path / "nope.csv") is None
    assert read_curation(None) is None


def test_read_curation_reads_csv(tmp_path):
    path = tmp_path / "curation.csv"
    pd.DataFrame({"id": ["r1"], "excluded": [True], "qc_reason": ["blurry"]}).to_csv(
        path, index=False
    )

    cur = read_curation(path)

    assert cur is not None
    assert list(cur["id"]) == ["r1"]


def test_apply_curation_ignores_extra_curation_columns():
    """Only ``excluded`` / ``qc_reason`` are joined; stray curation columns and
    ids not present in the table are dropped, not crashed on."""
    cur = pd.DataFrame(
        {
            "id": ["r1", "ghost"],
            "excluded": [True, True],
            "qc_reason": ["x", "y"],
            "annotator": ["alice", "bob"],
        }
    )

    out = apply_curation(_table(), cur)

    assert "annotator" not in out.columns
    assert len(out) == 3  # 'ghost' (no matching measurement row) dropped
    assert list(out["excluded"]) == [True, False, False]
