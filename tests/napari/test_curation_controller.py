"""The Qt-free curation editing controller: load, mutate-and-autosave, query."""
from __future__ import annotations

import pandas as pd

from cellflow.napari.contact_analysis.curation_controller import (
    CurationController,
)
from cellflow.contact_analysis.curation import read_curation


def test_loads_empty_when_file_absent(tmp_path):
    ctrl = CurationController(tmp_path / "curation.csv")
    assert len(ctrl.curation) == 0


def test_loads_existing_file(tmp_path):
    path = tmp_path / "curation.csv"
    pd.DataFrame({
        "experiment_id": ["EXP1"], "position_id": ["p1"], "frame": [pd.NA],
        "excluded": [True], "exclusion_reason": ["debris"],
    }).to_csv(path, index=False)
    ctrl = CurationController(path)
    assert len(ctrl.curation) == 1


def test_exclude_frame_autosaves(tmp_path):
    path = tmp_path / "curation.csv"
    ctrl = CurationController(path)
    ctrl.exclude_frame(experiment_id="EXP1", position_id="p1", frame=3, reason="blurry")
    back = read_curation(path)
    assert back is not None and len(back) == 1
    assert int(back.iloc[0]["frame"]) == 3


def test_exclude_position_autosaves_with_na_frame(tmp_path):
    path = tmp_path / "curation.csv"
    ctrl = CurationController(path)
    ctrl.exclude_position(experiment_id="EXP1", position_id="p1", reason="all bad")
    back = read_curation(path)
    assert pd.isna(back.iloc[0]["frame"])


def test_remove_autosaves(tmp_path):
    path = tmp_path / "curation.csv"
    ctrl = CurationController(path)
    ctrl.exclude_frame(experiment_id="EXP1", position_id="p1", frame=3, reason="x")
    ctrl.remove(experiment_id="EXP1", position_id="p1", frame=3)
    back = read_curation(path)
    assert back is None or len(back) == 0


def test_exclusions_for_position(tmp_path):
    ctrl = CurationController(tmp_path / "curation.csv")
    ctrl.exclude_frame(experiment_id="EXP1", position_id="p1", frame=3, reason="x")
    ctrl.exclude_frame(experiment_id="EXP1", position_id="p2", frame=4, reason="y")
    got = ctrl.exclusions_for(experiment_id="EXP1", position_id="p1")
    assert list(got["position_id"].astype(str)) == ["p1"]


def test_is_frame_excluded(tmp_path):
    ctrl = CurationController(tmp_path / "curation.csv")
    ctrl.exclude_frame(experiment_id="EXP1", position_id="p1", frame=3, reason="x")
    assert ctrl.is_frame_excluded(experiment_id="EXP1", position_id="p1", frame=3)
    assert not ctrl.is_frame_excluded(experiment_id="EXP1", position_id="p1", frame=2)


def test_is_position_excluded(tmp_path):
    ctrl = CurationController(tmp_path / "curation.csv")
    ctrl.exclude_position(experiment_id="EXP1", position_id="p1", reason="x")
    assert ctrl.is_position_excluded(experiment_id="EXP1", position_id="p1")
    assert not ctrl.is_position_excluded(experiment_id="EXP1", position_id="p2")
