"""The Reduce editor widget: state → collapse pipeline, add / remove / reorder."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from cellflow.aggregate_quantification.reduce import Collapse
from cellflow.napari.aggregate_quantification.reduce_editor import CollapsePipelineEditor

_APP = QApplication.instance() or QApplication([])


def _editor(columns=("condition", "position_id", "cell_id"), default=()):
    ed = CollapsePipelineEditor()
    ed.set_columns(list(columns), default)
    return ed


def test_default_pipeline_is_seeded():
    ed = _editor(default=(Collapse(by=("condition", "cell_id"), stat="mean"),))
    assert ed.pipeline() == (Collapse(by=("condition", "cell_id"), stat="mean"),)


def test_set_columns_drops_absent_by_columns():
    # A default referencing a column the table lacks is filtered to what fits.
    ed = _editor(columns=("condition", "position_id"),
                 default=(Collapse(by=("condition", "cell_id"), stat="mean"),))
    assert ed.pipeline() == (Collapse(by=("condition",), stat="mean"),)


def test_pipeline_orders_by_columns_and_skips_empty_steps():
    ed = _editor()
    ed._steps = [{"by": {"cell_id", "condition"}, "stat": "median"}, {"by": set(), "stat": "mean"}]
    # ``by`` follows the column order, and the all-empty step is skipped.
    assert ed.pipeline() == (Collapse(by=("condition", "cell_id"), stat="median"),)


def test_add_remove_and_reorder_emit_changed():
    ed = _editor(default=(Collapse(by=("cell_id",), stat="mean"),))
    fired = []
    ed.changed.connect(lambda: fired.append(True))

    ed._on_add()
    assert len(ed._steps) == 2 and fired

    ed._steps[1]["by"] = {"condition"}
    ed._move(1, -1)  # bring the new step to the front
    assert ed.pipeline()[0] == Collapse(by=("condition",), stat="mean")

    ed._remove(0)
    assert ed.pipeline() == (Collapse(by=("cell_id",), stat="mean"),)


def test_chained_collapse_climbs_levels():
    ed = _editor()
    ed._steps = [
        {"by": {"condition", "position_id", "cell_id"}, "stat": "mean"},
        {"by": {"condition", "position_id"}, "stat": "mean"},
    ]
    pipeline = ed.pipeline()
    assert len(pipeline) == 2
    assert pipeline[0].by == ("condition", "position_id", "cell_id")
    assert pipeline[1].by == ("condition", "position_id")
