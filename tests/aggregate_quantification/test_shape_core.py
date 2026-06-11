import json
import math

import numpy as np
import pandas as pd
import tifffile
from skimage.draw import disk

from cellflow.aggregate_quantification.shape.core import (
    DESCRIPTOR_COLUMNS,
    build_object_shape,
    provenance_path,
    read_shape_table,
)


def _disk_and_bar_frame() -> np.ndarray:
    """A near-circular cell (label 1) and an elongated cell (label 2)."""
    frame = np.zeros((60, 80), dtype=np.uint16)
    rr, cc = disk((20, 20), 14)
    frame[rr, cc] = 1
    frame[28:32, 40:74] = 2  # 4 x 34 bar -> high aspect ratio, low circularity
    return frame


def test_build_object_shape_writes_tidy_csv_per_frame(tmp_path):
    frame = _disk_and_bar_frame()
    stack = np.stack([frame, frame])  # 2 frames
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, stack)
    out = tmp_path / "sub" / "cell_shape.csv"

    result = build_object_shape(
        cell_path,
        out,
        pixel_size_um=0.5,
        object_key="cell_id",
        source_path=tmp_path,
        quantity_id="cell_shape",
    )

    assert result == out and out.exists()
    df = pd.read_csv(out)
    assert list(df.columns) == ["frame", "cell_id", *DESCRIPTOR_COLUMNS]
    assert df["frame"].tolist() == [0, 0, 1, 1]
    assert df["cell_id"].tolist() == [1, 2, 1, 2]


def test_build_object_shape_writes_provenance_sidecar(tmp_path):
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, _disk_and_bar_frame())
    out = tmp_path / "cell_shape.csv"

    build_object_shape(
        cell_path,
        out,
        pixel_size_um=0.5,
        object_key="cell_id",
        source_path=tmp_path,
        quantity_id="cell_shape",
    )

    sidecar = provenance_path(out)
    assert sidecar == tmp_path / "cell_shape.provenance.json" and sidecar.exists()
    prov = json.loads(sidecar.read_text())
    assert prov["quantity_id"] == "cell_shape"
    assert prov["pixel_size_um"] == 0.5
    assert prov["source_position_path"] == str(tmp_path)
    assert prov["label_path"] == str(cell_path)
    assert prov["columns"] == ["frame", "cell_id", *DESCRIPTOR_COLUMNS]
    assert "created_at" in prov and "cellflow_version" in prov


def test_build_object_shape_descriptor_values(tmp_path):
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, _disk_and_bar_frame())
    out = tmp_path / "cell_shape.csv"

    build_object_shape(cell_path, out, pixel_size_um=1.0, object_key="cell_id")
    table = read_shape_table(out)

    # Row 0 is the disk, row 1 the elongated bar.
    disk_circ, bar_circ = table["circularity"][0], table["circularity"][1]
    disk_aspect, bar_aspect = table["aspect_ratio"][0], table["aspect_ratio"][1]

    assert disk_circ > 0.85  # a discretized disk is nearly circular
    assert bar_circ < disk_circ  # the bar is far from circular
    assert disk_aspect < 1.3  # roughly isotropic
    assert bar_aspect > 3.0  # clearly elongated
    # circularity is clamped to a physical maximum of 1.
    assert np.all(table["circularity"][~np.isnan(table["circularity"])] <= 1.0)
    # area sanity: at 1 µm/px the disk's µm² area is ~ pi r^2.
    assert abs(table["area_um2"][0] - math.pi * 14 ** 2) / (math.pi * 14 ** 2) < 0.1


def test_build_object_shape_scales_to_physical_units(tmp_path):
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, _disk_and_bar_frame())

    unit = read_shape_table(
        build_object_shape(cell_path, tmp_path / "unit.csv", pixel_size_um=1.0)
    )
    scaled = read_shape_table(
        build_object_shape(cell_path, tmp_path / "scaled.csv", pixel_size_um=0.5)
    )

    # Lengths scale by s, area by s^2; dimensionless ratios are invariant.
    assert np.allclose(scaled["perimeter_um"], unit["perimeter_um"] * 0.5)
    assert np.allclose(scaled["major_axis_length_um"], unit["major_axis_length_um"] * 0.5)
    assert np.allclose(scaled["centroid_x_um"], unit["centroid_x_um"] * 0.5)
    assert np.allclose(scaled["area_um2"], unit["area_um2"] * 0.25)
    assert np.allclose(scaled["aspect_ratio"], unit["aspect_ratio"], equal_nan=True)
    assert np.allclose(scaled["circularity"], unit["circularity"], equal_nan=True)


def test_read_shape_table_round_trips_keys_as_int(tmp_path):
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, _disk_and_bar_frame())
    table = read_shape_table(
        build_object_shape(cell_path, tmp_path / "s.csv", pixel_size_um=1.0)
    )
    assert table["frame"].dtype == np.int64
    assert table["cell_id"].dtype == np.int64
    assert table["area_um2"].dtype == float


def test_build_object_shape_rejects_nonpositive_pixel_size(tmp_path):
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, _disk_and_bar_frame())

    for bad in (0.0, -1.0):
        try:
            build_object_shape(cell_path, tmp_path / "out.csv", pixel_size_um=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for pixel_size_um={bad}")


def test_build_object_shape_guards_degenerate_region(tmp_path):
    frame = np.zeros((6, 6), dtype=np.uint16)
    frame[2, 3] = 1  # single pixel: zero perimeter and zero minor axis
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, frame)
    out = tmp_path / "cell_shape.csv"

    build_object_shape(cell_path, out, pixel_size_um=1.0)
    table = read_shape_table(out)

    assert table["cell_id"].tolist() == [1]
    assert math.isnan(table["aspect_ratio"][0])
    assert math.isnan(table["circularity"][0])
    assert table["area_um2"][0] == 1.0


def test_build_object_shape_reports_progress_in_order(tmp_path):
    cell_path = tmp_path / "cells.tif"
    tifffile.imwrite(cell_path, _disk_and_bar_frame())
    progress = []

    build_object_shape(
        cell_path,
        tmp_path / "cell_shape.csv",
        pixel_size_um=1.0,
        progress_cb=lambda *a: progress.append(a),
    )

    assert progress == [
        (1, 3, "read labels"),
        (2, 3, "extract shape"),
        (3, 3, "write CSV"),
    ]
