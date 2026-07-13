"""The shape quantifiers pool via compute_object_table over their label field."""
import numpy as np
import tifffile

from cellflow.contact_analysis.quantifier import PositionInputs
from cellflow.contact_analysis.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.contact_analysis.quantifiers.nucleus_shape import NucleusShapeQuantifier
from cellflow.contact_analysis.shape import DESCRIPTOR_COLUMNS, compute_object_shape


def _two_cell_stack(tmp_path, name):
    frame = np.zeros((12, 12), dtype=np.uint16)
    frame[1:5, 1:5] = 1
    frame[6:10, 6:10] = 2
    path = tmp_path / name
    tifffile.imwrite(path, frame[np.newaxis, ...])
    return path


def test_cell_shape_compute_object_table_reads_cell_labels(tmp_path):
    labels = _two_cell_stack(tmp_path, "cells.tif")
    q = CellShapeQuantifier()
    inputs = PositionInputs(position_dir=tmp_path, cell_labels_path=labels, pixel_size_um=0.5)

    got = q.compute_object_table(inputs)
    expected = compute_object_shape(labels, pixel_size_um=0.5, object_key="cell_id")

    assert list(got) == ["frame", "cell_id", *DESCRIPTOR_COLUMNS]
    assert set(got) == set(expected)
    for key in expected:
        np.testing.assert_allclose(np.asarray(got[key], float), np.asarray(expected[key], float))


def test_nucleus_shape_compute_object_table_reads_nucleus_labels(tmp_path):
    labels = _two_cell_stack(tmp_path, "nuclei.tif")
    q = NucleusShapeQuantifier()
    inputs = PositionInputs(position_dir=tmp_path, nucleus_labels_path=labels, pixel_size_um=0.5)

    got = q.compute_object_table(inputs)
    expected = compute_object_shape(labels, pixel_size_um=0.5, object_key="cell_id")

    assert list(got) == ["frame", "cell_id", *DESCRIPTOR_COLUMNS]
    assert set(got) == set(expected)
    for key in expected:
        np.testing.assert_allclose(np.asarray(got[key], float), np.asarray(expected[key], float))
