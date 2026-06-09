import numpy as np
import tifffile

from cellflow.aggregate_quantification.contacts.reader import PositionContactAnalysis
from cellflow.aggregate_quantification.quantifier import (
    PositionInputs,
    Quantifier,
    available_quantifiers,
)
from cellflow.aggregate_quantification.quantifiers.contacts import ContactsQuantifier


def test_available_quantifiers_discovers_contacts():
    classes = available_quantifiers()
    assert ContactsQuantifier in classes
    ids = {cls.quantity_id for cls in classes}
    assert "contacts" in ids


def test_subclassing_registers_quantifier():
    class _FakeQuantifier(Quantifier):
        quantity_id = "fake_for_test"
        display_name = "Fake (test)"

    try:
        assert _FakeQuantifier in available_quantifiers()
    finally:
        # Keep the global registry clean for other tests.
        from cellflow.aggregate_quantification import quantifier as mod

        mod._REGISTRY.pop("fake_for_test", None)


def test_intermediate_base_without_id_is_not_registered():
    before = set(available_quantifiers())

    class _Abstract(Quantifier):  # no quantity_id -> not a real plugin
        pass

    assert set(available_quantifiers()) == before


def test_can_build_respects_requires(tmp_path):
    q = ContactsQuantifier()
    with_cell = PositionInputs(
        position_dir=tmp_path, cell_labels_path=tmp_path / "cells.tif"
    )
    without_cell = PositionInputs(position_dir=tmp_path)
    assert q.can_build(with_cell) is True
    assert q.can_build(without_cell) is False


def test_contacts_quantifier_default_output_name(tmp_path):
    q = ContactsQuantifier()
    assert q.default_output(PositionInputs(position_dir=tmp_path)).name == "contact_analysis.h5"


def test_contacts_quantifier_build_and_read(tmp_path):
    frame = np.zeros((5, 6), dtype=np.uint16)
    frame[:, :3] = 1
    frame[:, 3:] = 2
    cell_stack = np.stack([frame, frame])
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, cell_stack)

    q = ContactsQuantifier()
    inputs = PositionInputs(position_dir=tmp_path, cell_labels_path=cell_path)
    out = q.default_output(inputs)

    assert q.is_built(out) is False
    written = q.build(inputs, out)
    assert written == out
    assert q.is_built(out) is True

    analysis = q.read(out)
    assert isinstance(analysis, PositionContactAnalysis)
    assert set(analysis.edges) >= {"frame", "kind"}
