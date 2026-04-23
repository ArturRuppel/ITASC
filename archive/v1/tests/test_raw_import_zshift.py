from __future__ import annotations

import csv
import sys
import types

import numpy as np

from cellflow.cellpose.config import DatasetConfig
from cellflow.cellpose.stages.raw_import import (
    _estimate_z_shift,
    _fit_double_sigmoid_profile,
    _shift_volume,
)
from cellflow.cellpose.stages import raw_import as raw_import_stage


def _double_sigmoid_fn(
    z: np.ndarray,
    *,
    center: float,
    span: float = 4.0,
    width: float = 0.8,
    offset: float = 10.0,
    amplitude: float = 30.0,
    slope: float = 0.0,
) -> np.ndarray:
    left_edge = center - 0.5 * span
    right_edge = center + 0.5 * span
    left = 1.0 / (1.0 + np.exp(-(z - left_edge) / width))
    right = 1.0 / (1.0 + np.exp(-(z - right_edge) / width))
    return offset + slope * (z - center) + amplitude * (left - right)


def test_fit_double_sigmoid_recovers_center():
    z = np.arange(16, dtype=np.float64)
    true_center = 7.25
    profile = _double_sigmoid_fn(z, center=true_center, span=4.5, width=0.7, slope=0.1)

    fitted, params = _fit_double_sigmoid_profile(profile)

    assert fitted.shape == profile.shape
    assert abs(params[2] - true_center) < 0.2


def test_estimate_z_shift_recovers_double_sigmoid_offset():
    z = np.arange(16, dtype=np.float64)
    reference = _double_sigmoid_fn(z, center=6.5, span=4.0, width=0.7)
    true_shift = 1.75
    target = 1.2 * _double_sigmoid_fn(z, center=6.5 + true_shift, span=4.0, width=0.7) + 0.35

    shift, scale, offset, mse = _estimate_z_shift(
        reference,
        target,
        max_shift_slices=4.0,
    )

    assert abs(shift - true_shift) < 0.2
    assert scale > 0.0
    assert mse < 1e-1
    assert np.isfinite(offset)


def test_shift_volume_moves_slices_linearly():
    volume = np.stack(
        [
            np.full((2, 2), 10, dtype=np.uint16),
            np.full((2, 2), 20, dtype=np.uint16),
            np.full((2, 2), 30, dtype=np.uint16),
            np.full((2, 2), 40, dtype=np.uint16),
        ],
        axis=0,
    )

    shifted = _shift_volume(volume, 1.0)

    assert shifted.shape == volume.shape
    assert shifted.dtype == np.uint16
    assert np.all(shifted[0] == 0)
    assert np.all(shifted[1] == 10)
    assert np.all(shifted[2] == 20)


def test_run_estimates_shift_from_488_channel(tmp_path, monkeypatch):
    z = np.arange(16, dtype=np.float64)

    def profile_405(t: int) -> np.ndarray:
        if t == 0:
            return _double_sigmoid_fn(z, center=6.0, span=4.0, width=0.7)
        return _double_sigmoid_fn(z, center=8.0, span=4.0, width=0.7)

    def profile_488(t: int) -> np.ndarray:
        if t == 0:
            return _double_sigmoid_fn(z, center=5.5, span=4.0, width=0.7)
        return _double_sigmoid_fn(z, center=6.75, span=4.0, width=0.7)

    class FakeDataset:
        def __init__(self, _path: str):
            self.axes = {"position": [0], "time": [0, 1], "z": list(range(16))}
            self.image_height = 2
            self.image_width = 2
            self.summary_metadata = {"PixelSizeUm": 1.0, "Interval_ms": 1.0}

        def read_image(self, position: int, time: int, channel: int, z: int):
            if position != 0:
                return None
            if channel == 1:
                value = profile_405(time)[z]
            elif channel == 2:
                value = profile_488(time)[z]
            else:
                value = 0.0
            return np.full(
                (self.image_height, self.image_width),
                int(round((value + 10.0) * 1000.0)),
                dtype=np.uint16,
            )

        def get_image_coordinates_list(self):
            return []

        def read_metadata(self, **_kwargs):
            return {}

    fake_ndtiff = types.ModuleType("ndtiff")
    fake_ndtiff.Dataset = FakeDataset
    monkeypatch.setitem(sys.modules, "ndtiff", fake_ndtiff)

    root_dir = tmp_path / "project"
    config = DatasetConfig(
        ndtiff_path="/fake/ndtiff",
        root_dir=str(root_dir),
        positions=[0],
        xy_downsample=1,
    )

    list(raw_import_stage.run(config, pos=0, overwrite=True))

    shift_path = root_dir / "pos00" / "0_input" / "z_shift.csv"
    with shift_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert abs(float(rows[1]["z_shift_slices"]) - 1.25) < 0.2
