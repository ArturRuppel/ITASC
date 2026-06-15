"""The Shape editor widget: state → filter/collapse pipeline, position-aware
columns, the row-count trail, and edit signals."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from cellflow.aggregate_quantification.reduce import Collapse, Filter
from cellflow.napari.aggregate_quantification.shape_editor import ShapePipelineEditor

_APP = QApplication.instance() or QApplication([])


def _editor(columns=("condition", "position_id", "cell_id"), default=(), categorical=None):
    ed = ShapePipelineEditor()
    ed.set_columns(list(columns), default, categorical=categorical or {})
    return ed


def test_default_pipeline_is_seeded():
    ed = _editor(default=(Collapse(by=("condition", "cell_id"), stat="mean"),))
    assert ed.pipeline() == (Collapse(by=("condition", "cell_id"), stat="mean"),)


def test_set_columns_drops_absent_by_columns():
    # A default referencing a column the table lacks is filtered to what fits.
    ed = _editor(columns=("condition", "position_id"),
                 default=(Collapse(by=("condition", "cell_id"), stat="mean"),))
    assert ed.pipeline() == (Collapse(by=("condition",), stat="mean"),)


def test_mixed_filter_and_collapse_pipeline():
    ed = _editor(
        columns=("condition", "class_label", "position_id", "cell_id"),
        default=(
            Filter("class_label", "==", "pos"),
            Collapse(by=("condition", "cell_id"), stat="median"),
        ),
    )
    pipeline = ed.pipeline()
    assert pipeline == (
        Filter("class_label", "==", "pos"),
        Collapse(by=("condition", "cell_id"), stat="median"),
    )


def test_empty_collapse_step_is_skipped():
    ed = _editor()
    ed._steps = [
        {"kind": "collapse", "by": {"cell_id", "condition"}, "stat": "median"},
        {"kind": "collapse", "by": set(), "stat": "mean"},
    ]
    # ``by`` follows the column order, and the all-empty step is skipped (a no-op,
    # not a whole-table collapse).
    assert ed.pipeline() == (Collapse(by=("condition", "cell_id"), stat="median"),)


def test_n_column_offered_only_after_a_collapse():
    ed = _editor()
    # Before any collapse, ``n`` is not a present column (so not filterable).
    assert "n" not in ed._columns_before(0)
    ed._steps = [
        {"kind": "collapse", "by": {"condition", "cell_id"}, "stat": "mean"},
        {"kind": "filter", "column": "n", "op": ">=", "value": "5"},
    ]
    # After the collapse, ``n`` is present (and the folded-away nesting key is gone).
    available = ed._columns_before(1)
    assert "n" in available
    assert "position_id" not in available  # collapsed away (not in the collapse by)
    # The post-collapse ``filter n >= 5`` therefore survives in the pipeline.
    assert ed.pipeline()[-1] == Filter("n", ">=", "5")


def test_filter_on_folded_column_is_dropped():
    ed = _editor()
    ed._steps = [
        {"kind": "collapse", "by": {"condition"}, "stat": "mean"},
        {"kind": "filter", "column": "cell_id", "op": ">=", "value": "3"},
    ]
    # ``cell_id`` was folded away by the collapse, so the later filter is dropped.
    assert ed.pipeline() == (Collapse(by=("condition",), stat="mean"),)


def test_categorical_filter_uses_value_dropdown():
    ed = _editor(
        default=(Filter("class_label", "==", "pos"),),
        columns=("class_label", "cell_id"),
        categorical={"class_label": ["neg", "pos"]},
    )
    # A categorical column renders a QComboBox of distinct values; a numeric one a
    # free QLineEdit. Probe the built value control for the single filter row.
    from qtpy.QtWidgets import QComboBox, QLineEdit
    cat_control = ed._build_value_control(0, "class_label")
    assert isinstance(cat_control, QComboBox)
    num_control = ed._build_value_control(0, "cell_id")
    assert isinstance(num_control, QLineEdit)


def test_add_remove_and_reorder_emit_changed():
    ed = _editor(default=(Collapse(by=("cell_id",), stat="mean"),))
    fired = []
    ed.changed.connect(lambda: fired.append(True))

    ed._on_add_collapse()
    assert len(ed._steps) == 2 and fired

    ed._steps[1]["by"] = {"condition"}
    ed._move(1, -1)  # bring the new step to the front
    assert ed.pipeline()[0] == Collapse(by=("condition",), stat="mean")

    ed._remove(0)
    assert ed.pipeline() == (Collapse(by=("cell_id",), stat="mean"),)


def test_collapse_by_multi_select():
    # A collapse groups by one *or more* axes: ``+`` appends a slot, each slot is
    # set via ``_set_collapse_slot``, and "(none)" clears it. Building the unique
    # combination (here position_id · cell_id) is what keeps cells that share a
    # cell_id across positions from being pooled.
    ed = _editor(
        columns=("condition", "position_id", "cell_id"),
        default=(Collapse(by=("cell_id",), stat="mean"),),
    )
    fired = []
    ed.changed.connect(lambda: fired.append(True))

    ed._add_collapse_slot(0)  # append an empty by slot
    ed._set_collapse_slot(0, 1, "position_id")
    # ``by`` follows column order, so the combination reads position_id · cell_id.
    assert ed.pipeline() == (Collapse(by=("position_id", "cell_id"), stat="mean"),)
    assert fired

    # "(none)" on the extra slot removes it, back to the single axis.
    ed._set_collapse_slot(0, 1, "")
    assert ed.pipeline() == (Collapse(by=("cell_id",), stat="mean"),)

    # Clearing the sole remaining slot makes the collapse a skipped no-op.
    ed._set_collapse_slot(0, 0, "")
    assert ed.pipeline() == ()


def test_add_filter_emits_and_appends():
    ed = _editor(default=(Collapse(by=("cell_id",), stat="mean"),))
    fired = []
    ed.changed.connect(lambda: fired.append(True))
    ed._on_add_filter()
    assert fired
    assert ed._steps[-1]["kind"] == "filter"


def test_chained_collapse_climbs_levels():
    ed = _editor()
    ed._steps = [
        {"kind": "collapse", "by": {"condition", "position_id", "cell_id"}, "stat": "mean"},
        {"kind": "collapse", "by": {"condition", "position_id"}, "stat": "mean"},
    ]
    pipeline = ed.pipeline()
    assert len(pipeline) == 2
    assert pipeline[0].by == ("condition", "position_id", "cell_id")
    assert pipeline[1].by == ("condition", "position_id")


def test_set_row_counts_renders_trail_in_pipeline_order():
    ed = _editor()
    ed._steps = [
        {"kind": "collapse", "by": {"condition", "cell_id"}, "stat": "mean"},
        {"kind": "collapse", "by": set(), "stat": "mean"},  # skipped (no by)
        {"kind": "collapse", "by": {"condition"}, "stat": "mean"},
    ]
    ed._rebuild()
    # Two active steps → two counts, mapped onto the contributing rows (0 and 2),
    # leaving the skipped row (1) blank.
    ed.set_row_counts([8, 2], start=12)
    assert "12" in ed._header.text()
    # Each active step reads before → after; the first step's "before" is the start
    # count, the next active step's "before" is the previous active step's output.
    assert ed._count_labels[0].text() == "12 → 8"
    assert ed._count_labels[1].text() == ""
    assert ed._count_labels[2].text() == "8 → 2"


def test_set_row_counts_emits_no_signal():
    ed = _editor(default=(Collapse(by=("cell_id",), stat="mean"),))
    fired = []
    ed.changed.connect(lambda: fired.append(True))
    ed.set_row_counts([5], start=10)
    assert fired == []  # display-only
