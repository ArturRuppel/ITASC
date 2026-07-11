"""compute_object_table for the shape quantifiers matches the build round-trip."""
import numpy as np
import tifffile

from cellflow.contact_analysis.quantifier import PositionInputs
from cellflow.contact_analysis.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.contact_analysis.quantifiers.nucleus_shape import NucleusShapeQuantifier


def _two_cell_stack(tmp_path, name):
    frame = np.zeros((12, 12), dtype=np.uint16)
    frame[1:5, 1:5] = 1
    frame[6:10, 6:10] = 2
    path = tmp_path / name
    tifffile.imwrite(path, frame[np.newaxis, ...])
    return path


def test_cell_shape_compute_matches_build(tmp_path):
    labels = _two_cell_stack(tmp_path, "cells.tif")
    q = CellShapeQuantifier()
    inputs = PositionInputs(position_dir=tmp_path, cell_labels_path=labels, pixel_size_um=0.5)
    out = q.default_output(inputs)
    q.build(inputs, out)
    expected = q.object_table(out)

    got = q.compute_object_table(inputs)

    assert set(got) == set(expected)
    for key in expected:
        np.testing.assert_allclose(np.asarray(got[key], float), np.asarray(expected[key], float))


def test_nucleus_shape_compute_matches_build(tmp_path):
    labels = _two_cell_stack(tmp_path, "nuclei.tif")
    q = NucleusShapeQuantifier()
    inputs = PositionInputs(position_dir=tmp_path, nucleus_labels_path=labels, pixel_size_um=0.5)
    out = q.default_output(inputs)
    q.build(inputs, out)
    expected = q.object_table(out)

    got = q.compute_object_table(inputs)

    assert set(got) == set(expected)
    for key in expected:
        np.testing.assert_allclose(np.asarray(got[key], float), np.asarray(expected[key], float))
