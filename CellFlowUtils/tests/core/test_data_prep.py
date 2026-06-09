"""Tests for raw data preparation exports."""
from __future__ import annotations

import csv
import json
import sys
import types

import numpy as np
import tifffile

from cellflow_utils.core.data_prep import DatasetConfig, discover_metadata, run


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
        nucleus_channel=1,
        cell_channel=2,
        nls_channel=3,
    )

    progress = list(run(config, 0, overwrite=True))

    pos_dir = tmp_path / "pos00" / "0_input"
    nucleus = tifffile.imread(pos_dir / "nucleus_zavg.tif")
    nucleus_3dt = tifffile.imread(pos_dir / "nucleus_3dt.tif")
    cells = tifffile.imread(pos_dir / "cell_zavg.tif")
    cells_3dt = tifffile.imread(pos_dir / "cell_3dt.tif")
    nls = tifffile.imread(pos_dir / "NLS_zavg.tif")
    nls_3dt = tifffile.imread(pos_dir / "NLS_3dt.tif")
    with (pos_dir / "z_shift.csv").open(newline="", encoding="utf-8") as fh:
        shift_times = [int(float(row["time"])) for row in csv.DictReader(fh)]

    assert nucleus.shape == (3, 4, 6)
    assert nucleus_3dt.shape == (3, 5, 4, 6)
    assert cells.shape == (3, 4, 6)
    assert cells_3dt.shape == (3, 5, 4, 6)
    assert nls.shape == (3, 4, 6)
    assert nls_3dt.shape == (3, 5, 4, 6)
    np.testing.assert_array_equal(
        nucleus_3dt[0, :, 0, 0],
        np.array([110, 111, 112, 113, 114], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        cells_3dt[0, :, 0, 0],
        np.array([120, 121, 122, 123, 124], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        nls_3dt[0, :, 0, 0],
        np.array([130, 131, 132, 133, 134], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        nls[0, 0, 0],
        np.array(132, dtype=np.uint16),
    )
    assert shift_times == [1, 2, 3]
    assert [label for _done, _total, label in progress] == [
        "z-shift", "z-shift", "z-shift",
        "nucleus", "nucleus", "nucleus",
        "cell", "cell", "cell",
        "NLS", "NLS", "NLS",
    ]


def test_run_z_downsamples_3dt_outputs_by_averaging_adjacent_slices(monkeypatch, tmp_path):
    _install_fake_ndtiff(monkeypatch)
    config = DatasetConfig(
        ndtiff_path="fake.ndtiff",
        root_dir=str(tmp_path),
        positions=[0],
        xy_downsample=1,
        z_downsample=2,
        frame_start=1,
        frame_end=1,
    )

    list(run(config, 0, overwrite=True))

    pos_dir = tmp_path / "pos00" / "0_input"
    nls = tifffile.imread(pos_dir / "NLS_zavg.tif")
    nls_3dt = tifffile.imread(pos_dir / "NLS_3dt.tif")
    run_params = json.loads((pos_dir / "run_params.json").read_text(encoding="utf-8"))

    assert nls.shape == (1, 4, 6)
    assert nls_3dt.shape == (1, 3, 4, 6)
    np.testing.assert_array_equal(
        nls_3dt[0, :, 0, 0],
        np.array([120, 122, 124], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        nls[0, 0, 0],
        np.array(122, dtype=np.uint16),
    )
    assert run_params["z_downsample"] == 2


def test_dataset_config_defaults_to_april_28_channel_layout():
    config = DatasetConfig(ndtiff_path="fake.ndtiff", root_dir="/tmp/out", positions=[0])

    assert config.cell_channel == 1
    assert config.nls_channel == 2
    assert config.nucleus_channel == 3



def test_discover_metadata_includes_display_channel_names(monkeypatch, tmp_path):
    display_settings = {
        "map": {
            "ChannelSettings": {
                "array": [
                    {"Channel": {"scalar": "CSUTRANS"}},
                    {"Channel": {"scalar": "CSU405 "}},
                    {"Channel": {"scalar": "CSU488"}},
                    {"Channel": {"scalar": "CSU561"}},
                ]
            }
        }
    }
    (tmp_path / "DisplaySettings.json").write_text(json.dumps(display_settings), encoding="utf-8")

    class _DatasetWithPathMetadata(_FakeNDTiffDataset):
        def __init__(self, path: str) -> None:
            assert path == str(tmp_path)

    fake_module = types.SimpleNamespace(Dataset=_DatasetWithPathMetadata)
    monkeypatch.setitem(sys.modules, "ndtiff", fake_module)

    metadata = discover_metadata(str(tmp_path))

    assert metadata["channel_names"] == ["CSUTRANS", "CSU405", "CSU488", "CSU561"]
