"""Tests for save/load round-trip of TissueGraphDataset."""
import numpy as np
import pytest

from napariTissueGraph.core.graph import build_from_labels, build_from_tracks
from napariTissueGraph.core.io import (
    save_dataset,
    load_dataset,
    load_multiple_datasets,
    _serialize_ragged,
    _deserialize_ragged,
)
from napariTissueGraph.core.topology import detect_t1_events
from napariTissueGraph.analysis.trajectories import build_edge_trajectories
from napariTissueGraph.structures import (
    InputType,
    TissueGraphDataset,
    TissueGraphTimeSeries,
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from conftest import make_label_stack, make_track_positions


# ------------------------------------------------------------------
# Ragged array helpers
# ------------------------------------------------------------------

class TestRaggedSerialization:
    def test_round_trip(self):
        arrays = [
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([[7, 8]]),
            np.array([[9, 10], [11, 12], [13, 14], [15, 16]]),
        ]
        flat, offsets = _serialize_ragged(arrays)
        result = _deserialize_ragged(flat, offsets)
        assert len(result) == 3
        for orig, loaded in zip(arrays, result):
            np.testing.assert_array_equal(orig, loaded)

    def test_empty_list(self):
        flat, offsets = _serialize_ragged([])
        result = _deserialize_ragged(flat, offsets)
        assert len(result) == 0

    def test_single_array(self):
        arrays = [np.array([1, 2, 3])]
        flat, offsets = _serialize_ragged(arrays)
        result = _deserialize_ragged(flat, offsets)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0].ravel(), arrays[0])

    def test_with_empty_arrays(self):
        arrays = [
            np.array([[1, 2]]),
            np.empty((0, 2)),
            np.array([[3, 4], [5, 6]]),
        ]
        flat, offsets = _serialize_ragged(arrays)
        result = _deserialize_ragged(flat, offsets)
        assert len(result) == 3
        assert len(result[1]) == 0


# ------------------------------------------------------------------
# Round-trip: labels-based dataset
# ------------------------------------------------------------------

class TestSaveLoadLabels:
    @pytest.fixture
    def labels_dataset(self):
        stack = make_label_stack(n_frames=3, n_cells_side=4, image_size=200)
        series = build_from_labels(stack, pixel_size=0.65, time_interval=60.0)
        events = detect_t1_events(series)
        build_edge_trajectories(series, events)
        dataset = TissueGraphDataset(
            condition="WT",
            pixel_size=0.65,
            time_interval=60.0,
            input_type=InputType.SEGMENTATION,
            metadata={"experiment": "test"},
        )
        dataset.add_tissue(series)
        return dataset

    def test_round_trip(self, labels_dataset, tmp_path):
        save_path = tmp_path / "test_dataset"
        save_dataset(labels_dataset, save_path)

        loaded = load_dataset(save_path)

        # Dataset-level
        assert loaded.condition == "WT"
        assert loaded.pixel_size == 0.65
        assert loaded.time_interval == 60.0
        assert loaded.input_type == InputType.SEGMENTATION
        assert loaded.metadata == {"experiment": "test"}
        assert loaded.n_tissues == 1
        assert loaded.tissue_ids == [0]

    def test_tissue_frames(self, labels_dataset, tmp_path):
        save_path = tmp_path / "test_dataset"
        save_dataset(labels_dataset, save_path)
        loaded = load_dataset(save_path)

        orig = labels_dataset.tissues[0]
        loaded_s = loaded.tissues[0]

        assert loaded_s.num_frames == orig.num_frames
        assert loaded_s.frame_indices == orig.frame_indices

    def test_cells_preserved(self, labels_dataset, tmp_path):
        save_path = tmp_path / "test_dataset"
        save_dataset(labels_dataset, save_path)
        loaded = load_dataset(save_path)

        orig = labels_dataset.tissues[0]
        loaded_s = loaded.tissues[0]

        for f in orig.frame_indices:
            orig_frame = orig.frames[f]
            loaded_frame = loaded_s.frames[f]
            assert set(orig_frame.cells.keys()) == set(loaded_frame.cells.keys())
            for cid in orig_frame.cells:
                oc = orig_frame.cells[cid]
                lc = loaded_frame.cells[cid]
                np.testing.assert_allclose(oc.position, lc.position)
                assert abs(oc.area - lc.area) < 1e-10
                assert abs(oc.perimeter - lc.perimeter) < 1e-10
                assert abs(oc.shape_index - lc.shape_index) < 1e-10
                assert oc.num_neighbors == lc.num_neighbors
                assert oc.track_id == lc.track_id

    def test_junctions_preserved(self, labels_dataset, tmp_path):
        save_path = tmp_path / "test_dataset"
        save_dataset(labels_dataset, save_path)
        loaded = load_dataset(save_path)

        orig = labels_dataset.tissues[0]
        loaded_s = loaded.tissues[0]

        for f in orig.frame_indices:
            orig_frame = orig.frames[f]
            loaded_frame = loaded_s.frames[f]
            assert set(orig_frame.junctions.keys()) == set(loaded_frame.junctions.keys())
            for key in orig_frame.junctions:
                oj = orig_frame.junctions[key]
                lj = loaded_frame.junctions[key]
                assert abs(oj.length - lj.length) < 1e-10
                np.testing.assert_allclose(oj.midpoint, lj.midpoint)
                np.testing.assert_allclose(oj.coordinates, lj.coordinates)

    def test_graph_preserved(self, labels_dataset, tmp_path):
        save_path = tmp_path / "test_dataset"
        save_dataset(labels_dataset, save_path)
        loaded = load_dataset(save_path)

        orig = labels_dataset.tissues[0]
        loaded_s = loaded.tissues[0]

        for f in orig.frame_indices:
            orig_edges = set(tuple(sorted(e)) for e in orig.frames[f].graph.edges())
            loaded_edges = set(tuple(sorted(e)) for e in loaded_s.frames[f].graph.edges())
            assert orig_edges == loaded_edges

    def test_metadata_json_exists(self, labels_dataset, tmp_path):
        save_path = tmp_path / "test_dataset"
        save_dataset(labels_dataset, save_path)
        assert (save_path / "metadata.json").exists()
        assert (save_path / "tissue_000.npz").exists()


# ------------------------------------------------------------------
# Round-trip: tracks-based dataset
# ------------------------------------------------------------------

class TestSaveLoadTracks:
    @pytest.fixture
    def tracks_dataset(self):
        positions = make_track_positions(n_frames=5, nx=4, ny=4, spacing=20.0)
        series = build_from_tracks(
            positions, pixel_size=1.0, time_interval=30.0,
            image_shape=(100, 100),
        )
        events = detect_t1_events(series)
        build_edge_trajectories(series, events)
        dataset = TissueGraphDataset(
            condition="KO",
            pixel_size=1.0,
            time_interval=30.0,
            input_type=InputType.VORONOI,
        )
        dataset.add_tissue(series)
        return dataset

    def test_round_trip(self, tracks_dataset, tmp_path):
        save_path = tmp_path / "test_tracks"
        save_dataset(tracks_dataset, save_path)
        loaded = load_dataset(save_path)

        assert loaded.condition == "KO"
        assert loaded.n_tissues == 1
        assert loaded.input_type == InputType.VORONOI

        orig = tracks_dataset.tissues[0]
        loaded_s = loaded.tissues[0]
        assert loaded_s.num_frames == orig.num_frames

    def test_cell_vertices_preserved(self, tracks_dataset, tmp_path):
        save_path = tmp_path / "test_tracks"
        save_dataset(tracks_dataset, save_path)
        loaded = load_dataset(save_path)

        orig = tracks_dataset.tissues[0]
        loaded_s = loaded.tissues[0]

        for f in orig.frame_indices:
            for cid in orig.frames[f].cells:
                ov = orig.frames[f].cells[cid].vertices
                lv = loaded_s.frames[f].cells[cid].vertices
                if ov is not None:
                    assert lv is not None
                    np.testing.assert_allclose(ov, lv)
                else:
                    assert lv is None


# ------------------------------------------------------------------
# Multi-tissue dataset
# ------------------------------------------------------------------

class TestMultiTissueDataset:
    @pytest.fixture
    def multi_dataset(self):
        dataset = TissueGraphDataset(
            condition="mixed",
            pixel_size=0.65,
            time_interval=60.0,
            input_type=InputType.SEGMENTATION,
        )
        for i in range(3):
            stack = make_label_stack(n_frames=2 + i, n_cells_side=4, image_size=200)
            series = build_from_labels(stack)
            detect_t1_events(series)
            build_edge_trajectories(series, series.t1_events)
            dataset.add_tissue(series)
        return dataset

    def test_round_trip(self, multi_dataset, tmp_path):
        save_path = tmp_path / "multi"
        save_dataset(multi_dataset, save_path)
        loaded = load_dataset(save_path)

        assert loaded.n_tissues == 3
        assert loaded.tissue_ids == [0, 1, 2]

        for tid in multi_dataset.tissue_ids:
            orig = multi_dataset.tissues[tid]
            loaded_s = loaded.tissues[tid]
            assert loaded_s.num_frames == orig.num_frames


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_dataset(self, tmp_path):
        dataset = TissueGraphDataset(condition="empty")
        save_path = tmp_path / "empty"
        save_dataset(dataset, save_path)
        loaded = load_dataset(save_path)
        assert loaded.n_tissues == 0
        assert loaded.condition == "empty"

    def test_no_t1_events(self, tmp_path):
        """A tissue with no T1 events should still round-trip."""
        stack = make_label_stack(n_frames=2, n_cells_side=4, image_size=200)
        series = build_from_labels(stack)
        # Don't run T1 detection — no events
        dataset = TissueGraphDataset()
        dataset.add_tissue(series)

        save_path = tmp_path / "no_t1"
        save_dataset(dataset, save_path)
        loaded = load_dataset(save_path)
        assert len(loaded.tissues[0].t1_events) == 0
        assert len(loaded.tissues[0].edge_trajectories) == 0

    def test_none_pixel_size(self, tmp_path):
        dataset = TissueGraphDataset(pixel_size=None, time_interval=None)
        stack = make_label_stack(n_frames=2, n_cells_side=4, image_size=200)
        series = build_from_labels(stack)
        dataset.add_tissue(series)

        save_path = tmp_path / "none_params"
        save_dataset(dataset, save_path)
        loaded = load_dataset(save_path)
        assert loaded.pixel_size is None
        assert loaded.time_interval is None


# ------------------------------------------------------------------
# load_multiple_datasets
# ------------------------------------------------------------------

class TestLoadMultiple:
    def test_load_by_condition(self, tmp_path):
        for cond in ["WT", "KO"]:
            ds = TissueGraphDataset(condition=cond)
            stack = make_label_stack(n_frames=2, n_cells_side=4, image_size=200)
            series = build_from_labels(stack)
            ds.add_tissue(series)
            save_dataset(ds, tmp_path / cond)

        loaded = load_multiple_datasets([tmp_path / "WT", tmp_path / "KO"])
        assert set(loaded.keys()) == {"WT", "KO"}
        assert loaded["WT"].n_tissues == 1
        assert loaded["KO"].n_tissues == 1

    def test_load_by_dirname_when_no_condition(self, tmp_path):
        ds = TissueGraphDataset(condition="")
        stack = make_label_stack(n_frames=2, n_cells_side=4, image_size=200)
        series = build_from_labels(stack)
        ds.add_tissue(series)
        save_dataset(ds, tmp_path / "my_data")

        loaded = load_multiple_datasets([tmp_path / "my_data"])
        assert "my_data" in loaded


# ------------------------------------------------------------------
# Tags and names round-trip
# ------------------------------------------------------------------

class TestTagsRoundTrip:
    @pytest.fixture
    def tagged_dataset(self):
        positions = make_track_positions(n_frames=5, nx=4, ny=4, spacing=20.0)
        series = build_from_tracks(
            positions, pixel_size=1.0, time_interval=30.0,
            image_shape=(100, 100),
        )
        events = detect_t1_events(series)
        build_edge_trajectories(series, events)

        # Tag some junctions
        first_frame = series.frames[series.frame_indices[0]]
        for i, key in enumerate(first_frame.junctions):
            if i < 2:
                first_frame.junctions[key].tags.add("central")
            if i == 0:
                first_frame.junctions[key].tags.add("special")

        # Tag some trajectories
        traj_ids = sorted(series.edge_trajectories.keys())
        if traj_ids:
            series.edge_trajectories[traj_ids[0]].tags = {"central", "primary"}
            series.edge_trajectories[traj_ids[0]].name = "main_junction"
            if len(traj_ids) > 1:
                series.edge_trajectories[traj_ids[1]].tags = {"peripheral"}

        dataset = TissueGraphDataset(condition="tagged_test")
        dataset.add_tissue(series)
        return dataset

    def test_junction_tags_round_trip(self, tagged_dataset, tmp_path):
        save_path = tmp_path / "tagged"
        save_dataset(tagged_dataset, save_path)
        loaded = load_dataset(save_path)

        orig = tagged_dataset.tissues[0]
        loaded_s = loaded.tissues[0]

        first_frame_idx = orig.frame_indices[0]
        orig_frame = orig.frames[first_frame_idx]
        loaded_frame = loaded_s.frames[first_frame_idx]

        for key in orig_frame.junctions:
            assert orig_frame.junctions[key].tags == loaded_frame.junctions[key].tags

    def test_trajectory_tags_round_trip(self, tagged_dataset, tmp_path):
        save_path = tmp_path / "tagged"
        save_dataset(tagged_dataset, save_path)
        loaded = load_dataset(save_path)

        orig = tagged_dataset.tissues[0]
        loaded_s = loaded.tissues[0]

        for tid in orig.edge_trajectories:
            assert orig.edge_trajectories[tid].tags == loaded_s.edge_trajectories[tid].tags

    def test_trajectory_names_round_trip(self, tagged_dataset, tmp_path):
        save_path = tmp_path / "tagged"
        save_dataset(tagged_dataset, save_path)
        loaded = load_dataset(save_path)

        orig = tagged_dataset.tissues[0]
        loaded_s = loaded.tissues[0]

        for tid in orig.edge_trajectories:
            assert orig.edge_trajectories[tid].name == loaded_s.edge_trajectories[tid].name

    def test_empty_tags_round_trip(self, tmp_path):
        """Tissues with no user tags should round-trip cleanly.

        Border junctions may carry the auto-generated 'edge_border' tag, which
        is expected and should survive the round-trip.
        """
        stack = make_label_stack(n_frames=2, n_cells_side=4, image_size=200)
        series = build_from_labels(stack)
        detect_t1_events(series)
        build_edge_trajectories(series, series.t1_events)
        dataset = TissueGraphDataset()
        dataset.add_tissue(series)

        save_path = tmp_path / "no_tags"
        save_dataset(dataset, save_path)
        loaded = load_dataset(save_path)

        loaded_s = loaded.tissues[0]
        for frame in loaded_s.frames.values():
            for jd in frame.junctions.values():
                # Only auto-generated border tags allowed; no user-assigned tags
                assert jd.tags <= {"edge_border"}
        for traj in loaded_s.edge_trajectories.values():
            assert traj.tags == set()
            assert traj.name is None
