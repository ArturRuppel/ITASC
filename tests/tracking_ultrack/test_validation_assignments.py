"""Tests for validated-mask → candidate assignment (pure, no Ultrack DB).

``_best_iou_assignments`` is a pure function over mask records and candidate
geometry, so it runs without ``ultrack`` installed (unlike the DB-backed
``inject_validated_nodes`` tests, which are gated behind an importorskip).
"""
from __future__ import annotations

import numpy as np

from itasc.tracking_ultrack.validation_nodes import (
    _best_iou_assignments,
    _MaskRecord,
)


def _record(cell_id: int, y0: int, x0: int, size: int = 5) -> _MaskRecord:
    return _MaskRecord(
        cell_id=cell_id,
        t=0,
        bbox=(y0, x0, y0 + size, x0 + size),
        mask=np.ones((size, size), dtype=bool),
        area=size * size,
        y=float(y0 + size / 2),
        x=float(x0 + size / 2),
    )


def _candidate(y0: int, x0: int, size: int = 5):
    return ((y0, x0, y0 + size, x0 + size), np.ones((size, size), dtype=bool), 2)


def test_record_without_overlap_gets_no_assignment():
    # A validated mask far from the only candidate must NOT be assigned it — that
    # would overwrite an unrelated node in place and make the validated cell
    # inherit its spatially-wrong temporal links. It must fall through to a fresh
    # reserved node (represented here by the absence of an assignment).
    record = _record(cell_id=3, y0=0, x0=0)
    candidates = {77: _candidate(y0=50, x0=50)}
    assert _best_iou_assignments([record], candidates) == {}


def test_overlapping_record_is_assigned():
    record = _record(cell_id=3, y0=50, x0=50)
    candidates = {77: _candidate(y0=50, x0=50)}
    assert _best_iou_assignments([record], candidates) == {0: 77}


def test_only_overlapping_record_is_matched_when_others_do_not_overlap():
    # Two records, one candidate that only overlaps the second record. The first
    # (non-overlapping) record must stay unassigned rather than greedily claim
    # the candidate that belongs to the second.
    records = [_record(cell_id=3, y0=0, x0=0), _record(cell_id=4, y0=50, x0=50)]
    candidates = {77: _candidate(y0=50, x0=50)}
    assert _best_iou_assignments(records, candidates) == {1: 77}


def test_each_record_takes_its_own_best_overlap():
    records = [_record(cell_id=3, y0=0, x0=0), _record(cell_id=4, y0=50, x0=50)]
    candidates = {77: _candidate(y0=0, x0=0), 88: _candidate(y0=50, x0=50)}
    assert _best_iou_assignments(records, candidates) == {0: 77, 1: 88}
