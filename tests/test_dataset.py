"""Tests for TissueGraphDataset and batch (4D) operations."""
import numpy as np
import pytest

from cellflow.utils.structures import (
    InputType,
    TissueGraphDataset,
    TissueGraphTimeSeries,
)
from cellflow.backend.graph import (
    build_from_labels,
    build_from_labels_4d,
)
from cellflow.backend.topology import detect_all_t1_events


class TestTissueGraphDataset:
    """Tests for the TissueGraphDataset dataclass."""

    def test_empty_dataset(self):
        ds = TissueGraphDataset()
        assert ds.n_tissues == 0
        assert ds.tissue_ids == []

    def test_add_tissue(self):
        ds = TissueGraphDataset(condition="WT")
        series = TissueGraphTimeSeries(frames={})
        tid = ds.add_tissue(series)
        assert tid == 0
        assert ds.n_tissues == 1
        assert ds.tissue_ids == [0]

    def test_add_multiple_tissues(self):
        ds = TissueGraphDataset()
        ds.add_tissue(TissueGraphTimeSeries(frames={}))
        ds.add_tissue(TissueGraphTimeSeries(frames={}))
        ds.add_tissue(TissueGraphTimeSeries(frames={}))
        assert ds.n_tissues == 3
        assert ds.tissue_ids == [0, 1, 2]

    def test_remove_tissue(self):
        ds = TissueGraphDataset()
        ds.add_tissue(TissueGraphTimeSeries(frames={}))
        ds.add_tissue(TissueGraphTimeSeries(frames={}))
        ds.remove_tissue(0)
        assert ds.n_tissues == 1
        assert ds.tissue_ids == [1]

    def test_remove_nonexistent_tissue_raises(self):
        ds = TissueGraphDataset()
        with pytest.raises(KeyError):
            ds.remove_tissue(0)

    def test_add_after_remove_increments_id(self):
        ds = TissueGraphDataset()
        ds.add_tissue(TissueGraphTimeSeries(frames={}))
        ds.add_tissue(TissueGraphTimeSeries(frames={}))
        ds.remove_tissue(0)
        tid = ds.add_tissue(TissueGraphTimeSeries(frames={}))
        assert tid == 2

    def test_metadata_fields(self):
        ds = TissueGraphDataset(
            condition="vim_KO",
            pixel_size=0.325,
            time_interval=60.0,
            input_type=InputType.SEGMENTATION,
            metadata={"experiment_date": "2026-03-16"},
        )
        assert ds.condition == "vim_KO"
        assert ds.pixel_size == 0.325
        assert ds.time_interval == 60.0
        assert ds.input_type == InputType.SEGMENTATION
        assert ds.metadata["experiment_date"] == "2026-03-16"

    def test_get_all_t1_events_empty(self):
        ds = TissueGraphDataset()
        ds.add_tissue(TissueGraphTimeSeries(frames={}))
        assert ds.get_all_t1_events() == []

    def test_get_all_edge_trajectories_empty(self):
        ds = TissueGraphDataset()
        ds.add_tissue(TissueGraphTimeSeries(frames={}))
        assert ds.get_all_edge_trajectories() == []


class TestBuildFromLabels4D:
    """Tests for build_from_labels_4d."""

    def test_basic_build(self, label_stack_4d):
        ds = build_from_labels_4d(label_stack_4d, condition="test")
        assert ds.n_tissues == 2
        assert ds.condition == "test"
        assert ds.input_type == InputType.SEGMENTATION
        for tid in ds.tissue_ids:
            series = ds.tissues[tid]
            assert series.num_frames == 3
            assert series.input_type == InputType.SEGMENTATION

    def test_pixel_size_propagated(self, label_stack_4d):
        ds = build_from_labels_4d(label_stack_4d, pixel_size=0.5)
        assert ds.pixel_size == 0.5
        for series in ds.tissues.values():
            assert series.pixel_size == 0.5

    def test_time_interval_propagated(self, label_stack_4d):
        ds = build_from_labels_4d(label_stack_4d, time_interval=120.0)
        assert ds.time_interval == 120.0
        for series in ds.tissues.values():
            assert series.time_interval == 120.0

    def test_progress_callback(self, label_stack_4d):
        calls = []
        ds = build_from_labels_4d(
            label_stack_4d,
            progress_callback=lambda frac, msg: calls.append((frac, msg)),
        )
        assert len(calls) == 2
        assert calls[0][0] == 0.0
        assert calls[1][0] == 0.5

    def test_single_tissue(self):
        from tests.conftest import make_label_stack_4d
        stack = make_label_stack_4d(n_tissues=1)
        ds = build_from_labels_4d(stack)
        assert ds.n_tissues == 1

    def test_graph_has_cells_and_junctions(self, label_stack_4d):
        ds = build_from_labels_4d(label_stack_4d)
        for series in ds.tissues.values():
            for frame in series.frames.values():
                assert len(frame.cells) > 0
                assert len(frame.junctions) > 0


class TestBuildFromLabels4DRagged:
    """Tests for build_from_labels_4d with variable-length tissues."""

    def test_list_input_basic(self, label_stacks_ragged):
        ds = build_from_labels_4d(label_stacks_ragged, condition="ragged")
        assert ds.n_tissues == 2
        assert ds.condition == "ragged"
        assert ds.input_type == InputType.SEGMENTATION

    def test_different_frame_counts(self, label_stacks_ragged):
        ds = build_from_labels_4d(label_stacks_ragged)
        frame_counts = [ds.tissues[tid].num_frames for tid in ds.tissue_ids]
        assert frame_counts == [5, 8]

    def test_pixel_size_propagated(self, label_stacks_ragged):
        ds = build_from_labels_4d(label_stacks_ragged, pixel_size=0.325)
        assert ds.pixel_size == 0.325
        for series in ds.tissues.values():
            assert series.pixel_size == 0.325

    def test_progress_callback(self, label_stacks_ragged):
        calls = []
        build_from_labels_4d(
            label_stacks_ragged,
            progress_callback=lambda frac, msg: calls.append((frac, msg)),
        )
        assert len(calls) == 2
        assert calls[0][0] == 0.0
        assert calls[1][0] == 0.5

    def test_graph_has_cells_and_junctions(self, label_stacks_ragged):
        ds = build_from_labels_4d(label_stacks_ragged)
        for series in ds.tissues.values():
            for frame in series.frames.values():
                assert len(frame.cells) > 0
                assert len(frame.junctions) > 0

    def test_invalid_stack_dimensionality(self):
        bad_stacks = [np.zeros((3, 50, 50), dtype=np.int32), np.zeros((50, 50), dtype=np.int32)]
        with pytest.raises(ValueError, match="3D"):
            build_from_labels_4d(bad_stacks)

    def test_invalid_4d_array_dimensionality(self):
        bad_array = np.zeros((3, 50, 50), dtype=np.int32)
        with pytest.raises(ValueError, match="4D"):
            build_from_labels_4d(bad_array)

    def test_t1_detection_on_ragged(self, label_stacks_ragged):
        ds = build_from_labels_4d(label_stacks_ragged)
        detect_all_t1_events(ds)
        for series in ds.tissues.values():
            assert isinstance(series.t1_events, list)
            assert isinstance(series.edge_trajectories, dict)


class TestDetectAllT1Events:
    """Tests for detect_all_t1_events on a dataset."""

    def test_runs_on_dataset(self, label_stack_4d):
        ds = build_from_labels_4d(label_stack_4d)
        detect_all_t1_events(ds)
        # Should not error; t1_events and edge_trajectories populated
        for series in ds.tissues.values():
            assert isinstance(series.t1_events, list)
            assert isinstance(series.edge_trajectories, dict)

    def test_progress_callback(self, label_stack_4d):
        ds = build_from_labels_4d(label_stack_4d)
        calls = []
        detect_all_t1_events(
            ds,
            progress_callback=lambda frac, msg: calls.append((frac, msg)),
        )
        assert len(calls) == 2

    def test_get_all_t1_events_after_detection(self, label_stack_4d):
        ds = build_from_labels_4d(label_stack_4d)
        detect_all_t1_events(ds)
        all_events = ds.get_all_t1_events()
        # Each event should be tagged with a tissue_id
        for tid, event in all_events:
            assert tid in ds.tissue_ids

    def test_get_all_edge_trajectories_after_detection(self, label_stack_4d):
        ds = build_from_labels_4d(label_stack_4d)
        detect_all_t1_events(ds)
        all_trajs = ds.get_all_edge_trajectories()
        for tid, traj in all_trajs:
            assert tid in ds.tissue_ids
            assert len(traj.frames) > 0
