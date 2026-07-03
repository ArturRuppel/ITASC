from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cellflow.napari._catalog_table import CatalogRowSpec, CatalogTable
from cellflow.napari._stage_status import DONE, MISSING, STAGE_CELLPOSE


def _app():
    from napari.qt import get_qapp

    return get_qapp()


def _record_row(idx: int, id_: str, status=None) -> CatalogRowSpec:
    return CatalogRowSpec(
        record_index=idx,
        values=("cond", "date", id_, "cell", "notes"),
        status=status or {},
    )


def test_row_count_and_mapping_with_separators():
    _app()
    table = CatalogTable()
    table.set_rows(
        [
            CatalogRowSpec(record_index=None, caption="WT"),
            _record_row(0, "p1"),
            _record_row(1, "p2"),
            CatalogRowSpec(record_index=None, caption="KO"),
            _record_row(2, "p3"),
        ]
    )
    assert table.row_count() == 5
    assert table.row_to_record == [None, 0, 1, None, 2]


def test_programmatic_selection_excludes_separators():
    _app()
    table = CatalogTable()
    table.set_rows(
        [
            CatalogRowSpec(record_index=None, caption="WT"),
            _record_row(0, "p1"),
            _record_row(1, "p2"),
            CatalogRowSpec(record_index=None, caption="KO"),
            _record_row(2, "p3"),
        ]
    )
    table.select_records([2])
    assert table.selected_record_indices() == [2]

    table.select_all()
    assert table.selected_record_indices() == [0, 1, 2]

    table.clear_selection()
    assert table.selected_record_indices() == []


def test_selection_changed_signal_fires():
    _app()
    table = CatalogTable()
    table.set_rows([_record_row(0, "p1"), _record_row(1, "p2")])
    seen: list[list[int]] = []
    table.selectionChanged.connect(lambda: seen.append(table.selected_record_indices()))
    table.select_records([1])
    assert seen and seen[-1] == [1]


def test_set_rows_resets_stale_selection():
    _app()
    table = CatalogTable()
    table.set_rows([_record_row(0, "p1"), _record_row(1, "p2")])
    table.select_records([1])
    # Rebuilding with fewer rows must not leave a dangling selected index.
    table.set_rows([_record_row(0, "p1")])
    assert table.selected_record_indices() == []


def test_status_rail_reflects_spec_status():
    _app()
    table = CatalogTable()
    table.set_rows(
        [_record_row(0, "p1", status={STAGE_CELLPOSE: DONE})]
    )
    rail = table.record_row_widgets()[0].rail
    dot = next(d for d in rail.dots if d.stage == STAGE_CELLPOSE)
    assert dot.state == DONE
    # Absent stages default to their unknown/missing rendering, never crash.
    assert all(d.state in {DONE, MISSING, "unknown", "working", "stale"} for d in rail.dots)
