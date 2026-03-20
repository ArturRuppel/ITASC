"""Tests for the data query API (core/api.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import make_label_stack, make_track_positions

from napariTissueGraph.core.graph import build_from_labels, build_from_tracks
from napariTissueGraph.core.topology import detect_t1_events
from napariTissueGraph.core.io import save_dataset, load_dataset
from napariTissueGraph.analysis.trajectories import build_edge_trajectories
from napariTissueGraph.structures import (
    InputType,
    TissueGraphDataset,
    TissueGraphTimeSeries,
)
from napariTissueGraph.core.api import (
    _resolve_source,
    get_cells,
    get_junctions,
    get_trajectories,
    get_trajectory_summary,
    get_t1_events,
    get_time_since_last_t1,
    get_cell_history,
    get_neighbor_history,
    get_t1_rate,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def series_with_topology():
    """Build a TissueGraphTimeSeries with tracking, T1 detection, and trajectories."""
    stack = make_label_stack(n_frames=5, n_cells_side=4, image_size=200)
    series = build_from_labels(
        stack, pixel_size=0.65, time_interval=60.0,
    )
    events = detect_t1_events(series)
    trajectories = build_edge_trajectories(series, events)
    series.edge_trajectories = trajectories
    return series


@pytest.fixture
def dataset_single(series_with_topology):
    """Dataset with a single tissue."""
    ds = TissueGraphDataset(
        condition="WT",
        pixel_size=0.65,
        time_interval=60.0,
        input_type=InputType.SEGMENTATION,
    )
    ds.add_tissue(series_with_topology)
    return ds


@pytest.fixture
def dataset_multi():
    """Dataset with multiple tissues."""
    ds = TissueGraphDataset(
        condition="multi",
        pixel_size=0.65,
        time_interval=60.0,
        input_type=InputType.SEGMENTATION,
    )
    for i in range(2):
        stack = make_label_stack(n_frames=3 + i, n_cells_side=4, image_size=200)
        series = build_from_labels(stack, pixel_size=0.65, time_interval=60.0)
        events = detect_t1_events(series)
        trajectories = build_edge_trajectories(series, events)
        series.edge_trajectories = trajectories
        ds.add_tissue(series)
    return ds


# ------------------------------------------------------------------
# Source resolution
# ------------------------------------------------------------------

class TestResolveSource:
    def test_series_source(self, series_with_topology):
        items, multi = _resolve_source(series_with_topology)
        assert not multi
        assert len(items) == 1
        assert items[0][0] == 0
        assert items[0][1] is series_with_topology

    def test_dataset_single(self, dataset_single):
        items, multi = _resolve_source(dataset_single)
        assert not multi
        assert len(items) == 1

    def test_dataset_multi(self, dataset_multi):
        items, multi = _resolve_source(dataset_multi)
        assert multi
        assert len(items) == 2

    def test_path_source(self, dataset_single, tmp_path):
        save_path = tmp_path / "ds"
        save_dataset(dataset_single, save_path)
        items, multi = _resolve_source(save_path)
        assert not multi
        assert len(items) == 1

    def test_str_path_source(self, dataset_single, tmp_path):
        save_path = tmp_path / "ds"
        save_dataset(dataset_single, save_path)
        items, multi = _resolve_source(str(save_path))
        assert not multi
        assert len(items) == 1


# ------------------------------------------------------------------
# get_cells
# ------------------------------------------------------------------

class TestGetCells:
    def test_basic_columns(self, series_with_topology):
        df = get_cells(series_with_topology)
        expected_cols = {
            "frame", "cell_id", "track_id", "y", "x", "area",
            "perimeter", "shape_index", "num_neighbors", "pressure", "speed",
        }
        assert expected_cols == set(df.columns)
        assert len(df) > 0

    def test_no_tissue_id_for_single(self, series_with_topology):
        df = get_cells(series_with_topology)
        assert "tissue_id" not in df.columns

    def test_tissue_id_for_multi(self, dataset_multi):
        df = get_cells(dataset_multi)
        assert "tissue_id" in df.columns
        assert df.columns[0] == "tissue_id"

    def test_frame_filter(self, series_with_topology):
        df_all = get_cells(series_with_topology)
        df_filtered = get_cells(series_with_topology, frames={0, 1})
        assert set(df_filtered["frame"].unique()) <= {0, 1}
        assert len(df_filtered) < len(df_all)

    def test_neighbor_filter(self, series_with_topology):
        df = get_cells(series_with_topology, min_neighbors=4, max_neighbors=6)
        assert df["num_neighbors"].min() >= 4
        assert df["num_neighbors"].max() <= 6

    def test_area_filter(self, series_with_topology):
        df_all = get_cells(series_with_topology)
        median_area = df_all["area"].median()
        df = get_cells(series_with_topology, min_area=median_area)
        assert df["area"].min() >= median_area

    def test_track_id_filter(self, series_with_topology):
        df_all = get_cells(series_with_topology)
        valid_tracks = df_all["track_id"].dropna().unique()
        if len(valid_tracks) > 0:
            target = {int(valid_tracks[0])}
            df = get_cells(series_with_topology, track_ids=target)
            assert set(df["track_id"].unique()) == target

    def test_has_tracking_filter(self, series_with_topology):
        df_tracked = get_cells(series_with_topology, has_tracking=True)
        assert df_tracked["track_id"].notna().all()


# ------------------------------------------------------------------
# get_junctions
# ------------------------------------------------------------------

class TestGetJunctions:
    def test_basic_columns(self, series_with_topology):
        df = get_junctions(series_with_topology)
        expected_cols = {
            "frame", "cell_a", "cell_b", "length",
            "midpoint_y", "midpoint_x", "tension", "normal_stress", "tags",
        }
        assert expected_cols == set(df.columns)
        assert len(df) > 0

    def test_frame_filter(self, series_with_topology):
        df = get_junctions(series_with_topology, frames={0})
        assert set(df["frame"].unique()) == {0}

    def test_length_filter(self, series_with_topology):
        df_all = get_junctions(series_with_topology)
        median_len = df_all["length"].median()
        df = get_junctions(series_with_topology, min_length=median_len)
        assert df["length"].min() >= median_len

    def test_tag_filter(self, series_with_topology):
        # Tag some junctions
        first_frame = series_with_topology.frames[0]
        keys = list(first_frame.junctions.keys())
        if keys:
            first_frame.junctions[keys[0]].tags.add("test_tag")
            df = get_junctions(series_with_topology, tags={"test_tag"})
            assert len(df) >= 1
            # exclude_tags
            df_excl = get_junctions(series_with_topology, exclude_tags={"test_tag"})
            # The tagged junction should not appear in frame 0
            frame0 = df_excl[df_excl["frame"] == 0]
            pair = first_frame.junctions[keys[0]].cell_pair
            matching = frame0[
                (frame0["cell_a"] == pair[0]) & (frame0["cell_b"] == pair[1])
            ]
            assert len(matching) == 0

    def test_cell_ids_filter(self, series_with_topology):
        first_frame = series_with_topology.frames[0]
        some_cell = next(iter(first_frame.cells.keys()))
        df = get_junctions(series_with_topology, cell_ids={some_cell})
        # Every junction should involve some_cell
        assert ((df["cell_a"] == some_cell) | (df["cell_b"] == some_cell)).all()


# ------------------------------------------------------------------
# get_trajectories / get_trajectory_summary
# ------------------------------------------------------------------

class TestGetTrajectories:
    def test_basic_columns(self, series_with_topology):
        df = get_trajectories(series_with_topology)
        expected = {
            "trajectory_id", "frame", "cell_a", "cell_b",
            "signed_length", "abs_length", "tags", "name", "n_t1_events",
        }
        assert expected == set(df.columns)

    def test_min_frames_filter(self, series_with_topology):
        df_all = get_trajectories(series_with_topology)
        if df_all.empty:
            pytest.skip("No trajectories in test data")
        max_frames = df_all.groupby("trajectory_id").size().max()
        df = get_trajectories(series_with_topology, min_frames=max_frames)
        assert (df.groupby("trajectory_id").size() >= max_frames).all()

    def test_trajectory_ids_filter(self, series_with_topology):
        df_all = get_trajectories(series_with_topology)
        if df_all.empty:
            pytest.skip("No trajectories in test data")
        tid = df_all["trajectory_id"].iloc[0]
        df = get_trajectories(series_with_topology, trajectory_ids={tid})
        assert set(df["trajectory_id"].unique()) == {tid}

    def test_abs_length_positive(self, series_with_topology):
        df = get_trajectories(series_with_topology)
        if not df.empty:
            assert (df["abs_length"] >= 0).all()


class TestGetTrajectorySummary:
    def test_basic_columns(self, series_with_topology):
        df = get_trajectory_summary(series_with_topology)
        expected = {
            "trajectory_id", "n_frames", "first_frame", "last_frame",
            "mean_abs_length", "n_t1_events", "tags", "name",
        }
        assert expected == set(df.columns)

    def test_one_row_per_trajectory(self, series_with_topology):
        df = get_trajectory_summary(series_with_topology)
        if not df.empty:
            assert df["trajectory_id"].is_unique


# ------------------------------------------------------------------
# get_t1_events
# ------------------------------------------------------------------

class TestGetT1Events:
    def test_basic_columns(self, series_with_topology):
        df = get_t1_events(series_with_topology)
        if df.empty:
            pytest.skip("No T1 events in test data")
        expected = {
            "frame", "losing_a", "losing_b", "gaining_a", "gaining_b", "y", "x",
        }
        assert expected == set(df.columns)

    def test_frame_filter(self, series_with_topology):
        df_all = get_t1_events(series_with_topology)
        if df_all.empty:
            pytest.skip("No T1 events")
        target_frame = df_all["frame"].iloc[0]
        df = get_t1_events(series_with_topology, frames={target_frame})
        assert (df["frame"] == target_frame).all()

    def test_cell_ids_filter(self, series_with_topology):
        df_all = get_t1_events(series_with_topology)
        if df_all.empty:
            pytest.skip("No T1 events")
        # Filter by one of the cells from the first event
        target = {df_all["losing_a"].iloc[0]}
        df = get_t1_events(series_with_topology, cell_ids=target)
        assert len(df) >= 1


# ------------------------------------------------------------------
# get_time_since_last_t1
# ------------------------------------------------------------------

class TestGetTimeSinceLastT1:
    def test_basic_columns(self, series_with_topology):
        df = get_time_since_last_t1(series_with_topology)
        expected = {
            "frame", "cell_id", "frames_since_last_t1", "time_since_last_t1",
        }
        assert expected == set(df.columns)

    def test_values_non_negative(self, series_with_topology):
        df = get_time_since_last_t1(series_with_topology)
        valid = df["frames_since_last_t1"].dropna()
        if len(valid) > 0:
            assert (valid >= 0).all()

    def test_frame_filter(self, series_with_topology):
        df = get_time_since_last_t1(series_with_topology, frames={0})
        assert set(df["frame"].unique()) == {0}


# ------------------------------------------------------------------
# get_cell_history
# ------------------------------------------------------------------

class TestGetCellHistory:
    def test_returns_sorted_by_frame(self, series_with_topology):
        df_all = get_cells(series_with_topology, has_tracking=True)
        if df_all.empty:
            pytest.skip("No tracked cells")
        tid = int(df_all["track_id"].iloc[0])
        df = get_cell_history(series_with_topology, track_id=tid)
        assert list(df["frame"]) == sorted(df["frame"])

    def test_single_track(self, series_with_topology):
        df_all = get_cells(series_with_topology, has_tracking=True)
        if df_all.empty:
            pytest.skip("No tracked cells")
        tid = int(df_all["track_id"].iloc[0])
        df = get_cell_history(series_with_topology, track_id=tid)
        assert (df["track_id"] == tid).all()


# ------------------------------------------------------------------
# get_neighbor_history
# ------------------------------------------------------------------

class TestGetNeighborHistory:
    def test_basic_columns(self, series_with_topology):
        first_frame = series_with_topology.frames[0]
        cid = next(iter(first_frame.cells.keys()))
        df = get_neighbor_history(series_with_topology, cell_id=cid)
        assert set(df.columns) == {"frame", "cell_id", "neighbor_id"}
        assert (df["cell_id"] == cid).all()

    def test_frame_filter(self, series_with_topology):
        first_frame = series_with_topology.frames[0]
        cid = next(iter(first_frame.cells.keys()))
        df = get_neighbor_history(series_with_topology, cell_id=cid, frames={0})
        assert set(df["frame"].unique()) == {0}


# ------------------------------------------------------------------
# get_t1_rate
# ------------------------------------------------------------------

class TestGetT1Rate:
    def test_basic_columns(self, series_with_topology):
        df = get_t1_rate(series_with_topology)
        assert set(df.columns) == {"frame", "n_t1_events", "t1_rate"}

    def test_one_row_per_frame(self, series_with_topology):
        df = get_t1_rate(series_with_topology)
        assert len(df) == series_with_topology.num_frames

    def test_window(self, series_with_topology):
        df1 = get_t1_rate(series_with_topology, window=1)
        df3 = get_t1_rate(series_with_topology, window=3)
        # Window > 1 smooths the rate
        assert len(df1) == len(df3)

    def test_multi_tissue(self, dataset_multi):
        df = get_t1_rate(dataset_multi)
        assert "tissue_id" in df.columns


# ------------------------------------------------------------------
# Pressure field round-trip through IO
# ------------------------------------------------------------------

class TestPressureRoundTrip:
    def test_pressure_survives_save_load(self, tmp_path):
        stack = make_label_stack(n_frames=2, n_cells_side=4, image_size=200)
        series = build_from_labels(stack, pixel_size=0.65, time_interval=60.0)

        # Set pressure on some cells
        frame0 = series.frames[0]
        cell_ids = list(frame0.cells.keys())
        frame0.cells[cell_ids[0]].pressure = 1.5
        frame0.cells[cell_ids[1]].pressure = -0.3

        ds = TissueGraphDataset(condition="pressure_test")
        ds.add_tissue(series)

        save_path = tmp_path / "pressure_ds"
        save_dataset(ds, save_path)
        loaded = load_dataset(save_path)

        loaded_frame = loaded.tissues[0].frames[0]
        assert loaded_frame.cells[cell_ids[0]].pressure == pytest.approx(1.5)
        assert loaded_frame.cells[cell_ids[1]].pressure == pytest.approx(-0.3)
        # Cells without pressure should remain None
        assert loaded_frame.cells[cell_ids[2]].pressure is None

    def test_pressure_in_api(self, tmp_path):
        stack = make_label_stack(n_frames=2, n_cells_side=4, image_size=200)
        series = build_from_labels(stack, pixel_size=0.65, time_interval=60.0)

        frame0 = series.frames[0]
        cell_ids = list(frame0.cells.keys())
        frame0.cells[cell_ids[0]].pressure = 2.0

        df = get_cells(series)
        row = df[(df["frame"] == 0) & (df["cell_id"] == cell_ids[0])]
        assert row["pressure"].iloc[0] == pytest.approx(2.0)

    def test_all_none_pressure_not_stored(self, tmp_path):
        """When all pressures are None, the array key should not be in the NPZ."""
        stack = make_label_stack(n_frames=2, n_cells_side=4, image_size=200)
        series = build_from_labels(stack)
        ds = TissueGraphDataset()
        ds.add_tissue(series)

        save_path = tmp_path / "no_pressure"
        save_dataset(ds, save_path)

        with np.load(save_path / "tissue_000.npz") as npz:
            pressure_keys = [k for k in npz.files if "pressure" in k]
            assert len(pressure_keys) == 0
