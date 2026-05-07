"""Tests for raw data preparation exports."""
from __future__ import annotations

import csv
import sys
import types

import numpy as np
import tifffile

from cellflow.core.data_prep import DatasetConfig, run


class _FakeNDTiffDataset:
    image_height = 4
    image_width = 6
    axes = {"position": [0], "time": [0, 1, 2, 3, 4], "z": [0, 1, 2, 3, 4]}
    summary_metadata = {"PixelSizeUm": 0.25, "Interval_ms": 120000}

    def __init__(self, _path: str) -> None:
        pass

    def read_image(self, *, position: int, time: int, channel: int, z: int) -> np.ndarray:
        assert position == 0
        value = time * 100 + channel * 10 + z
        return np.full((self.image_height, self.image_width), value, dtype=np.uint16)


def _install_fake_ndtiff(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(Dataset=_FakeNDTiffDataset)
    monkeypatch.setitem(sys.modules, "ndtiff", fake_module)


def test_run_exports_only_selected_inclusive_frame_range(monkeypatch, tmp_path):
    _install_fake_ndtiff(monkeypatch)
    config = DatasetConfig(
        ndtiff_path="fake.ndtiff",
        root_dir=str(tmp_path),
        positions=[0],
        xy_downsample=1,
        frame_start=1,
        frame_end=3,
    )

    progress = list(run(config, 0, overwrite=True))

    pos_dir = tmp_path / "pos00" / "0_input"
    nucleus = tifffile.imread(pos_dir / "nucleus_zavg.tif")
    nucleus_3dt = tifffile.imread(pos_dir / "nucleus_3dt.tif")
    cells = tifffile.imread(pos_dir / "cell_zavg.tif")
    cells_3dt = tifffile.imread(pos_dir / "cell_3dt.tif")
    with (pos_dir / "z_shift.csv").open(newline="", encoding="utf-8") as fh:
        shift_times = [int(float(row["time"])) for row in csv.DictReader(fh)]

    assert nucleus.shape == (3, 4, 6)
    assert nucleus_3dt.shape == (3, 5, 4, 6)
    assert cells.shape == (3, 4, 6)
    assert cells_3dt.shape == (3, 5, 4, 6)
    assert shift_times == [1, 2, 3]
    assert [label for _done, _total, label in progress] == [
        "z-shift", "z-shift", "z-shift",
        "nucleus", "nucleus", "nucleus",
        "cell", "cell", "cell",
    ]
