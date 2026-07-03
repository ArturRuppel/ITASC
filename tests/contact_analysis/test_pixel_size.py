import json

import numpy as np
import tifffile

from cellflow.contact_analysis.pixel_size import (
    pixel_size_from_config,
    pixel_size_from_tiff,
    resolve_pixel_size_um,
)


def _write_config(position_dir, pixel_size_um):
    config = {"metadata": {"pixel_size_um": pixel_size_um}}
    (position_dir / "cellflow_config.json").write_text(json.dumps(config))


def test_pixel_size_from_config_reads_metadata(tmp_path):
    _write_config(tmp_path, 0.325)
    assert pixel_size_from_config(tmp_path) == 0.325


def test_pixel_size_from_config_rejects_nonpositive_and_missing(tmp_path):
    assert pixel_size_from_config(tmp_path) is None  # no file
    _write_config(tmp_path, 0)
    assert pixel_size_from_config(tmp_path) is None
    (tmp_path / "cellflow_config.json").write_text("{not json")
    assert pixel_size_from_config(tmp_path) is None


def test_pixel_size_from_imagej_unit_and_resolution(tmp_path):
    path = tmp_path / "labels.tif"
    frame = np.zeros((4, 4), dtype=np.uint16)
    # ImageJ: resolution is pixels/unit, so 5 px/µm -> 0.2 µm/px.
    tifffile.imwrite(path, frame, imagej=True, resolution=(5.0, 5.0),
                     metadata={"unit": "micron"})
    assert abs(pixel_size_from_tiff(path) - 0.2) < 1e-9


def test_pixel_size_from_tiff_missing_returns_none(tmp_path):
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((4, 4), dtype=np.uint16))
    assert pixel_size_from_tiff(path) is None
    assert pixel_size_from_tiff(tmp_path / "nope.tif") is None


def test_resolve_prefers_config_over_tiff(tmp_path):
    path = tmp_path / "labels.tif"
    tifffile.imwrite(path, np.zeros((4, 4), dtype=np.uint16), imagej=True,
                     resolution=(5.0, 5.0), metadata={"unit": "micron"})
    _write_config(tmp_path, 0.5)
    # Config (0.5) wins over the TIFF tag (0.2).
    assert resolve_pixel_size_um(tmp_path, path) == 0.5


def test_resolve_falls_back_to_tiff_then_none(tmp_path):
    path = tmp_path / "labels.tif"
    tifffile.imwrite(path, np.zeros((4, 4), dtype=np.uint16), imagej=True,
                     resolution=(4.0, 4.0), metadata={"unit": "um"})
    assert abs(resolve_pixel_size_um(tmp_path, path) - 0.25) < 1e-9

    plain = tmp_path / "plain.tif"
    tifffile.imwrite(plain, np.zeros((4, 4), dtype=np.uint16))
    assert resolve_pixel_size_um(tmp_path / "empty", plain) is None
