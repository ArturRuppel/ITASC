import math

import numpy as np
import tifffile

from cellflow.contact_analysis.shape.relational import (
    RELATIONAL_COLUMNS,
    compute_relational_table,
)


def _paired_frames():
    """Cell + nucleus frames sharing id 1, each with one unpaired extra id.

    Cell 1 is a 20x20 square (area 400); nucleus 1 a 10x10 square (area 100),
    shifted 1 px down/right so the centroid offset is non-zero. Cell 2 and
    nucleus 3 have no partner in the other stack -> dropped from the join.
    """
    cell = np.zeros((40, 40), dtype=np.uint16)
    cell[0:20, 0:20] = 1
    cell[30:40, 30:40] = 2  # unpaired cell

    nucleus = np.zeros((40, 40), dtype=np.uint16)
    nucleus[6:16, 6:16] = 1
    nucleus[0:5, 30:35] = 3  # unpaired nucleus
    return cell, nucleus


def _write_frames(tmp_path):
    cell, nucleus = _paired_frames()
    cell_path = tmp_path / "cells.tif"
    nuc_path = tmp_path / "nuclei.tif"
    tifffile.imwrite(cell_path, cell)
    tifffile.imwrite(nuc_path, nucleus)
    return cell_path, nuc_path


def test_compute_relational_ratios_and_offset(tmp_path):
    cell_path, nuc_path = _write_frames(tmp_path)

    table = compute_relational_table(cell_path, nuc_path, pixel_size_um=1.0)

    # Only the shared id 1 survives the inner join (unpaired ids dropped).
    assert table["frame"].tolist() == [0]
    assert table["cell_id"].tolist() == [1]
    assert math.isclose(table["nc_area_ratio"][0], 100 / 400)
    assert table["cell_area_um2"][0] == 400.0
    assert table["nucleus_area_um2"][0] == 100.0
    # nucleus is shifted, so the centroids do not coincide.
    assert table["centroid_offset_um"][0] > 0
    assert table["centroid_offset_norm"][0] > 0
    # orientation delta is folded into [0, pi/2].
    assert 0.0 <= table["orientation_delta"][0] <= math.pi / 2 + 1e-9
    assert np.isfinite(table["nc_perimeter_ratio"][0])
    assert np.isfinite(table["nc_major_axis_ratio"][0])


def test_compute_relational_columns_and_keys(tmp_path):
    cell_path, nuc_path = _write_frames(tmp_path)

    table = compute_relational_table(cell_path, nuc_path, pixel_size_um=1.0)

    assert list(table) == ["frame", "cell_id", *RELATIONAL_COLUMNS]
    assert table["frame"].dtype == np.int64
    assert table["cell_id"].dtype == np.int64


def test_compute_relational_scales_offset_with_pixel_size(tmp_path):
    cell_path, nuc_path = _write_frames(tmp_path)

    unit = compute_relational_table(cell_path, nuc_path, pixel_size_um=1.0)
    scaled = compute_relational_table(cell_path, nuc_path, pixel_size_um=0.5)

    # Offset is a length (scales by s); the ratio is dimensionless (invariant).
    assert np.allclose(scaled["centroid_offset_um"], unit["centroid_offset_um"] * 0.5)
    assert np.allclose(scaled["nc_area_ratio"], unit["nc_area_ratio"])
    assert np.allclose(scaled["centroid_offset_norm"], unit["centroid_offset_norm"])
