"""The studio's Aggregate area: the shape-table status list + Run delegation."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
from qtpy.QtWidgets import QApplication

from cellflow.aggregate_quantification.records import output_for_record
from cellflow.aggregate_quantification.shape_tables import aggregate, catalogue_root
from cellflow.aggregate_quantification.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.napari.aggregate_quantification_aggregate_area import AggregateArea


#: Kept alive for the module's lifetime so the QApplication is not GC'd between
#: constructing widgets (PyQt6 ties the C++ app to this Python object).
_APP = QApplication.instance() or QApplication([])


def _app():
    return _APP


def _record(tmp, pid):
    pdir = tmp / "ctrl" / pid
    pdir.mkdir(parents=True, exist_ok=True)
    return {"id": pid, "condition": "ctrl", "date": "d1", "position_path": pdir}


def _write_cell_shape(record):
    cs = CellShapeQuantifier()
    out = output_for_record(cs, record)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"frame": np.array([0, 0]), "cell_id": np.array([1, 2]), "area_um2": [1.0, 2.0]}
    ).to_csv(out, index=False)


def test_status_reflects_written_table(tmp_path):
    _app()
    rec = _record(tmp_path, "a")
    _write_cell_shape(rec)
    aggregate([rec], catalogue_root([rec]))

    area = AggregateArea(lambda recs: None)
    area.set_context(type("C", (), {"records": [rec]})())

    status_lbl, detail_lbl = area._rows["cells_by_frame"]
    assert status_lbl.text() == "built"
    assert "2 rows" in detail_lbl.text()
    # A table nothing was built for stays empty.
    assert area._rows["edges_by_frame"][0].text() == "—"


def test_run_button_delegates_scope(tmp_path):
    _app()
    rec = _record(tmp_path, "a")
    seen: list[list[dict]] = []
    area = AggregateArea(seen.append)
    area.set_context(type("C", (), {"records": [rec]})())
    area._on_run()
    assert seen == [[rec]]


def test_run_disabled_without_records():
    _app()
    area = AggregateArea(lambda recs: None)
    area.set_context(type("C", (), {"records": []})())
    assert not area._run_btn.isEnabled()
