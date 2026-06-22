"""RunArea — the studio's single Run section (controls + RunChoices)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import QApplication

from cellflow.napari.aggregate_quantification_run_area import (
    RunArea,
    RunChoices,
    _group_label,
    _grouped_quantities,
)


def _app():
    return QApplication.instance() or QApplication([])


class _Ctx:
    def __init__(self, records):
        self.records = records


def _area():
    return RunArea(save_callback=lambda c: None, run_callback=lambda c: None)


def test_quantities_default_all_checked():
    app = _app()
    area = _area()
    choices = area.choices()
    assert isinstance(choices, RunChoices)
    assert len(choices.quantities) >= 1  # every registered quantifier, checked
    area.deleteLater()
    app.processEvents()


def test_quantities_grouped_by_required_input():
    """Quantifiers are grouped by the input they need, derived from `requires`."""
    groups = _grouped_quantities()
    by_label = {_group_label(req): [c.quantity_id for c in cs] for req, cs in groups}
    # Each known input kind becomes its own group with the right members.
    assert "contacts" in by_label["Cell labels"]
    assert "cell_shape" in by_label["Cell labels"]
    assert set(by_label["Nucleus labels"]) == {"nucleus_shape", "nucleus_dynamics"}
    assert "neighbor_count" in by_label["Contacts"]
    assert by_label["Cell labels + Nucleus labels"] == ["shape_relational"]
    # Single-input groups precede the combined one; every quantifier appears once.
    labels = list(by_label)
    assert labels.index("Cell labels") < labels.index("Cell labels + Nucleus labels")
    flat = [qid for ids in by_label.values() for qid in ids]
    assert len(flat) == len(set(flat))


def test_grouping_does_not_change_choices_membership():
    app = _app()
    area = _area()
    # Grouping is presentation-only: every quantifier still has a checkbox.
    flat = {c.quantity_id for _req, cs in _grouped_quantities() for c in cs}
    assert set(area._quantity_checks) == flat
    area.deleteLater()
    app.processEvents()


def test_out_dir_empty_by_default():
    app = _app()
    area = _area()
    assert area.choices().out_dir == ""
    area.deleteLater()
    app.processEvents()


def test_out_dir_reads_the_field():
    app = _app()
    area = _area()
    area._out_dir.setText("/data/COV2D")
    assert area.choices().out_dir == "/data/COV2D"
    area.deleteLater()
    app.processEvents()


def test_buttons_disabled_with_no_records():
    app = _app()
    area = _area()
    area.set_context(_Ctx([]))
    assert not area._run_btn.isEnabled()
    assert not area._save_btn.isEnabled()
    area.set_context(_Ctx([{"id": "p1"}]))
    assert area._run_btn.isEnabled()
    assert area._save_btn.isEnabled()
    area.deleteLater()
    app.processEvents()


def test_zero_quantities_disables_actions():
    app = _app()
    area = _area()
    area.set_context(_Ctx([{"id": "p1"}]))
    for cb in area._quantity_checks.values():
        cb.setChecked(False)
    assert not area._run_btn.isEnabled()
    area.deleteLater()
    app.processEvents()


def test_run_button_invokes_callback_with_choices():
    app = _app()
    seen = []
    area = RunArea(save_callback=lambda c: None,
                   run_callback=lambda c: seen.append(c))
    area.set_context(_Ctx([{"id": "p1"}]))
    area._run_btn.click()
    assert len(seen) == 1 and isinstance(seen[0], RunChoices)
    area.deleteLater()
    app.processEvents()


def test_save_button_invokes_callback_with_choices():
    app = _app()
    seen = []
    area = RunArea(save_callback=lambda c: seen.append(c),
                   run_callback=lambda c: None)
    area.set_context(_Ctx([{"id": "p1"}]))
    area._save_btn.click()
    assert len(seen) == 1 and isinstance(seen[0], RunChoices)
    area.deleteLater()
    app.processEvents()
