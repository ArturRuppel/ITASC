import math

import h5py
import numpy as np
import tifffile
from skimage.draw import disk

from cellflow.aggregate_quantification.cell_shape.build import (
    COLUMNS,
    build_cell_shape,
    read_cell_shape,
)


def _disk_and_bar_frame() -> np.ndarray:
    """A near-circular cell (label 1) and an elongated cell (label 2)."""
    frame = np.zeros((60, 80), dtype=np.uint16)
    rr, cc = disk((20, 20), 14)
    frame[rr, cc] = 1
    frame[28:32, 40:74] = 2  # 4 x 34 bar -> high aspect ratio, low circularity
    return frame


def test_build_cell_shape_writes_tidy_table_per_frame(tmp_path):
    frame = _disk_and_bar_frame()
    stack = np.stack([frame, frame])  # 2 frames
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, stack)
    out = tmp_path / "sub" / "cell_shape.h5"

    result = build_cell_shape(
        cell_labels_path=cell_path, output_path=out, source_path=tmp_path
    )

    assert result == out and out.exists()
    with h5py.File(out, "r") as h5:
        table = h5["shape/table"]
        assert list(table.keys()) == list(COLUMNS)
        assert table["frame"][:].tolist() == [0, 0, 1, 1]
        assert table["cell_id"][:].tolist() == [1, 2, 1, 2]

        prov = h5["provenance"].attrs
        assert prov["cell_tracked_labels_path"] == str(cell_path)
        assert prov["source_position_path"] == str(tmp_path)
        assert "created_at" in prov and "cellflow_version" in prov


def test_build_cell_shape_descriptor_values(tmp_path):
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, _disk_and_bar_frame())
    out = tmp_path / "cell_shape.h5"

    build_cell_shape(cell_labels_path=cell_path, output_path=out)
    table = read_cell_shape(out)

    # Row 0 is the disk, row 1 the elongated bar.
    disk_circ, bar_circ = table["circularity"][0], table["circularity"][1]
    disk_aspect, bar_aspect = table["aspect_ratio"][0], table["aspect_ratio"][1]

    assert disk_circ > 0.85  # a discretized disk is nearly circular
    assert bar_circ < disk_circ  # the bar is far from circular
    assert disk_aspect < 1.3  # roughly isotropic
    assert bar_aspect > 3.0  # clearly elongated
    # circularity is clamped to a physical maximum of 1.
    assert np.all(table["circularity"][~np.isnan(table["circularity"])] <= 1.0)
    # area sanity: the disk's pixel area is ~ pi r^2.
    assert abs(table["area"][0] - math.pi * 14 ** 2) / (math.pi * 14 ** 2) < 0.1


def test_build_cell_shape_guards_degenerate_region(tmp_path):
    frame = np.zeros((6, 6), dtype=np.uint16)
    frame[2, 3] = 1  # single pixel: zero perimeter and zero minor axis
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, frame)
    out = tmp_path / "cell_shape.h5"

    build_cell_shape(cell_labels_path=cell_path, output_path=out)
    table = read_cell_shape(out)

    assert table["cell_id"].tolist() == [1]
    assert math.isnan(table["aspect_ratio"][0])
    assert math.isnan(table["circularity"][0])
    assert table["area"][0] == 1.0


def test_build_cell_shape_reports_progress_in_order(tmp_path):
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, _disk_and_bar_frame())
    progress = []

    build_cell_shape(
        cell_labels_path=cell_path,
        output_path=tmp_path / "cell_shape.h5",
        progress_cb=lambda *a: progress.append(a),
    )

    assert progress == [
        (1, 3, "read labels"),
        (2, 3, "extract shape"),
        (3, 3, "write HDF5"),
    ]
