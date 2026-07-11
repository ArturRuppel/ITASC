import numpy as np
import tifffile

from cellflow.contact_analysis.contacts.reader import PositionContactAnalysis
from cellflow.contact_analysis.quantifier import (
    PositionInputs,
    Quantifier,
    available_quantifiers,
)
from cellflow.contact_analysis.quantifiers.contacts import ContactsQuantifier
from cellflow.contact_analysis.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.contact_analysis.quantifiers.nucleus_shape import (
    NucleusShapeQuantifier,
)
from cellflow.contact_analysis.quantifiers.shape_relational import (
    ShapeRelationalQuantifier,
)


def test_available_quantifiers_discovers_contacts():
    classes = available_quantifiers()
    assert ContactsQuantifier in classes
    ids = {cls.quantity_id for cls in classes}
    assert "contacts" in ids


def test_available_quantifiers_discovers_shape_trio():
    ids = {cls.quantity_id for cls in available_quantifiers()}
    assert {"cell_shape", "nucleus_shape", "shape_relational"} <= ids


def test_shape_quantifiers_nest_outputs_under_aggregate_quantification(tmp_path):
    inputs = PositionInputs(position_dir=tmp_path)
    assert CellShapeQuantifier().default_output(inputs) == (
        tmp_path / "aggregate_quantification" / "cell_shape.csv"
    )
    assert NucleusShapeQuantifier().default_output(inputs) == (
        tmp_path / "aggregate_quantification" / "nucleus_shape.csv"
    )
    assert ShapeRelationalQuantifier().default_output(inputs) == (
        tmp_path / "aggregate_quantification" / "shape_relational.csv"
    )


def test_nucleus_shape_quantifier_requires_and_reads_nucleus_labels(tmp_path):
    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    nuc_path = tmp_path / "nuclei.tif"
    tifffile.imwrite(nuc_path, np.stack([frame, frame]))

    q = NucleusShapeQuantifier()
    # Pixel size is now a global build param, not a per-position file requirement.
    assert q.requires == ("nucleus_labels_path",)
    assert q.required_build_params == {"pixel_size_um": "pixel size (µm/px)"}
    # Cell labels alone do not satisfy a nucleus-shape build.
    assert q.can_build(
        PositionInputs(position_dir=tmp_path, cell_labels_path=nuc_path)
    ) is False
    inputs = PositionInputs(
        position_dir=tmp_path, nucleus_labels_path=nuc_path, pixel_size_um=1.0
    )
    out = q.default_output(inputs)
    assert q.can_build(inputs) is True

    written = q.build(inputs, out)
    assert written == out and out.name == "nucleus_shape.csv" and q.is_built(out)
    assert q.object_table(out)["cell_id"].tolist() == [1, 2, 1, 2]


def test_shape_relational_quantifier_requires_both_labels(tmp_path):
    q = ShapeRelationalQuantifier()
    # Pixel size is a global build param now; the file requirement is both stacks.
    assert q.requires == ("cell_labels_path", "nucleus_labels_path")
    assert q.required_build_params == {"pixel_size_um": "pixel size (µm/px)"}
    only_cell = PositionInputs(
        position_dir=tmp_path, cell_labels_path=tmp_path / "c.tif"
    )
    assert q.can_build(only_cell) is False
    both = PositionInputs(
        position_dir=tmp_path,
        cell_labels_path=tmp_path / "c.tif",
        nucleus_labels_path=tmp_path / "n.tif",
    )
    assert q.can_build(both) is True


def test_cell_shape_quantifier_default_output_name(tmp_path):
    q = CellShapeQuantifier()
    assert q.default_output(PositionInputs(position_dir=tmp_path)).name == "cell_shape.csv"


def test_cell_shape_quantifier_build_read_and_object_table(tmp_path):
    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, np.stack([frame, frame]))

    q = CellShapeQuantifier()
    inputs = PositionInputs(
        position_dir=tmp_path, cell_labels_path=cell_path, pixel_size_um=0.25
    )
    out = q.default_output(inputs)

    assert q.can_build(inputs) is True
    # No cell labels -> not buildable.
    assert q.can_build(PositionInputs(position_dir=tmp_path)) is False
    # Cell labels alone now satisfy the per-position file gate; pixel size is a
    # global build param the studio checks via required_build_params, not can_build.
    assert (
        q.can_build(PositionInputs(position_dir=tmp_path, cell_labels_path=cell_path))
        is True
    )
    assert q.is_built(out) is False

    written = q.build(inputs, out)
    assert written == out and q.is_built(out) is True

    table = q.read(out)
    assert table["cell_id"].tolist() == [1, 2, 1, 2]
    # object_table exposes the same tidy table to the plotting backend.
    object_table = q.object_table(out)
    assert object_table.keys() == table.keys()
    assert "circularity" in object_table
    assert "area_um2" in object_table


def test_subclassing_registers_quantifier():
    class _FakeQuantifier(Quantifier):
        quantity_id = "fake_for_test"
        display_name = "Fake (test)"

    try:
        assert _FakeQuantifier in available_quantifiers()
    finally:
        # Keep the global registry clean for other tests.
        from cellflow.contact_analysis import quantifier as mod

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


def test_compute_object_table_default_raises():
    from cellflow.contact_analysis.quantifier import PositionInputs, Quantifier

    class _Bare(Quantifier):
        quantity_id = ""  # not registered

    import pytest
    with pytest.raises(NotImplementedError):
        _Bare().compute_object_table(PositionInputs(position_dir=__import__("pathlib").Path(".")))


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
