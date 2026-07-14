import numpy as np
import tifffile

from itasc.contact_analysis.contacts.reader import PositionContactAnalysis
from itasc.contact_analysis.quantifier import (
    PositionInputs,
    Quantifier,
    available_quantifiers,
)
from itasc.contact_analysis.quantifiers.contacts import ContactsQuantifier
from itasc.contact_analysis.quantifiers.cell_shape import CellShapeQuantifier
from itasc.contact_analysis.quantifiers.nucleus_shape import (
    NucleusShapeQuantifier,
)
from itasc.contact_analysis.quantifiers.shape_relational import (
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
    assert q.can_build(inputs) is True
    # Pooled quantity: it computes its tidy table in memory (no artifact written).
    assert q.compute_object_table(inputs)["cell_id"].tolist() == [1, 2, 1, 2]


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


def test_cell_shape_quantifier_computes_tidy_table(tmp_path):
    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, np.stack([frame, frame]))

    q = CellShapeQuantifier()
    inputs = PositionInputs(
        position_dir=tmp_path, cell_labels_path=cell_path, pixel_size_um=0.25
    )

    assert q.can_build(inputs) is True
    # No cell labels -> not buildable.
    assert q.can_build(PositionInputs(position_dir=tmp_path)) is False
    # Cell labels alone satisfy the per-position file gate; pixel size is a global
    # build param checked via required_build_params, not can_build.
    assert (
        q.can_build(PositionInputs(position_dir=tmp_path, cell_labels_path=cell_path))
        is True
    )

    # Pooled quantity: compute_object_table returns the tidy table in memory.
    table = q.compute_object_table(inputs)
    assert table["cell_id"].tolist() == [1, 2, 1, 2]
    assert "circularity" in table
    assert "area_um2" in table


def test_subclassing_registers_quantifier():
    class _FakeQuantifier(Quantifier):
        quantity_id = "fake_for_test"
        display_name = "Fake (test)"

    try:
        assert _FakeQuantifier in available_quantifiers()
    finally:
        # Keep the global registry clean for other tests.
        from itasc.contact_analysis import quantifier as mod

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
    from itasc.contact_analysis.quantifier import PositionInputs, Quantifier

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


def test_output_subdir_is_stage_numbered():
    from itasc.contact_analysis.quantifier import OUTPUT_SUBDIR
    from itasc.contact_analysis.catalog import CONTACT_ANALYSIS_RELPATH
    # The dynamics quantifiers still persist under the stage-numbered folder…
    assert OUTPUT_SUBDIR == "4_contact_analysis"
    # …but the contacts h5 lives in the position base folder, beside the
    # committed labels (one homogeneous layout for the downstream-stable outputs).
    assert CONTACT_ANALYSIS_RELPATH == "contact_analysis.h5"


# --------------------------------------------------------- supported_quantities


def test_supported_quantities_gates_on_inputs_and_params(tmp_path):
    """A pooled quantity is 'supported' only when a record satisfies both its
    ``requires`` inputs and its ``required_build_params`` — the same gate
    ``build_table`` applies, so an enabled checkbox never promises an empty table."""
    from itasc.contact_analysis.records import supported_quantities

    # Cell labels + pixel size, but no nucleus, no FOV area, no contacts file.
    # The label input is gated on the file existing, so place it.
    cells = tmp_path / "cells.tif"
    cells.touch()
    rec = {
        "position_path": tmp_path / "posA",
        "cell_tracked_labels_path": cells,
        "pixel_size_um": 0.5,
    }
    supported = supported_quantities([rec])
    assert "cell_shape" in supported          # cell labels + pixel size present
    assert "nucleus_shape" not in supported   # no nucleus labels
    assert "shape_relational" not in supported  # needs both label sets
    assert "cell_density" not in supported    # no fov_area_mm2 param
    assert "neighbor_count" not in supported  # contacts.h5 does not exist
    assert "contacts" not in supported        # a producer — never pooled

    # Supplying the FOV area lifts cell_density into support.
    assert "cell_density" in supported_quantities([rec], params={"fov_area_mm2": 1.0})
