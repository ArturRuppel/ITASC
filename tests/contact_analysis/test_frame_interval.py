import json

import numpy as np
import tifffile

from cellflow.contact_analysis.frame_interval import (
    resolve_time_interval_s,
    time_interval_from_config,
    time_interval_from_tiff,
)


def test_time_interval_from_config_reads_metadata(tmp_path):
    (tmp_path / "cellflow_config.json").write_text(
        json.dumps({"metadata": {"time_interval_s": 12.5}})
    )
    assert time_interval_from_config(tmp_path) == 12.5


def test_time_interval_from_config_missing_or_nonpositive(tmp_path):
    assert time_interval_from_config(tmp_path) is None  # no config file
    (tmp_path / "cellflow_config.json").write_text(
        json.dumps({"metadata": {"time_interval_s": 0}})
    )
    assert time_interval_from_config(tmp_path) is None


def test_time_interval_from_tiff_finterval(tmp_path):
    path = tmp_path / "labels.tif"
    tifffile.imwrite(
        path, np.zeros((2, 4, 4), dtype=np.uint16), imagej=True, metadata={"finterval": 30.0}
    )
    assert time_interval_from_tiff(path) == 30.0


def test_resolve_prefers_config_then_tiff(tmp_path):
    path = tmp_path / "labels.tif"
    tifffile.imwrite(
        path, np.zeros((2, 4, 4), dtype=np.uint16), imagej=True, metadata={"finterval": 30.0}
    )
    # No config -> falls back to the TIFF tag.
    assert resolve_time_interval_s(tmp_path, path) == 30.0
    # Config wins when present.
    (tmp_path / "cellflow_config.json").write_text(
        json.dumps({"metadata": {"time_interval_s": 5.0}})
    )
    assert resolve_time_interval_s(tmp_path, path) == 5.0


def test_resolve_returns_none_when_unknown(tmp_path):
    assert resolve_time_interval_s(tmp_path, None) is None
