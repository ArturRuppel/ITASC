"""Tests for staged pipeline: extract → track / track → extract equivalence."""
import numpy as np
import pytest

from napariTissueFlow.core.graph import (
    build_from_labels,
    extract_graphs_from_labels,
    assign_tracking_labels,
    apply_track_map,
    has_tracking,
)
from napariTissueFlow.core.label_tracking import assign_track_ids
from napariTissueFlow.structures import InputType


class TestStage1NoTracking:
    """Stage 1 functions should produce graphs with no track_ids."""

    def test_extract_graphs_from_labels_no_tracking(self, label_stack):
        series = extract_graphs_from_labels(label_stack)
        assert series.num_frames == 3
        assert series.input_type == InputType.SEGMENTATION
        for frame in series.frames.values():
            assert len(frame.cells) > 0
            for cell in frame.cells.values():
                assert cell.track_id is None

    def test_has_tracking_false_after_stage1(self, label_stack):
        series = extract_graphs_from_labels(label_stack)
        assert has_tracking(series) is False

    def test_extract_preserves_structure(self, label_stack):
        """Stage 1 should produce same cells/junctions as monolithic."""
        series_staged = extract_graphs_from_labels(label_stack)
        series_mono = build_from_labels(label_stack)

        for frame_idx in series_staged.frame_indices:
            assert len(series_staged.frames[frame_idx].cells) == len(series_mono.frames[frame_idx].cells)
            assert len(series_staged.frames[frame_idx].junctions) == len(series_mono.frames[frame_idx].junctions)


class TestStage2Tracking:
    """Stage 2 should assign track_ids to cells."""

    def test_assign_tracking_labels(self, label_stack):
        series = extract_graphs_from_labels(label_stack)
        assign_tracking_labels(series, label_stack)
        for frame in series.frames.values():
            for cell in frame.cells.values():
                assert cell.track_id is not None

    def test_has_tracking_true_after_stage2(self, label_stack):
        series = extract_graphs_from_labels(label_stack)
        assert has_tracking(series) is False
        assign_tracking_labels(series, label_stack)
        assert has_tracking(series) is True


class TestApplyTrackMap:
    """Test apply_track_map() applies pre-computed tracking correctly."""

    def test_apply_track_map_sets_track_ids(self, label_stack):
        series = extract_graphs_from_labels(label_stack)
        track_map = assign_track_ids(label_stack)
        apply_track_map(series, track_map)

        assert has_tracking(series) is True
        for frame_idx, frame in series.frames.items():
            frame_tracks = track_map.get(frame_idx, {})
            for cell_id, cell in frame.cells.items():
                if cell_id in frame_tracks:
                    assert cell.track_id == frame_tracks[cell_id]

    def test_apply_track_map_equivalent_to_assign_tracking(self, label_stack):
        """apply_track_map + assign_track_ids should match assign_tracking_labels."""
        # Method 1: assign_tracking_labels (the original monolithic way)
        series1 = extract_graphs_from_labels(label_stack)
        assign_tracking_labels(series1, label_stack)

        # Method 2: assign_track_ids + apply_track_map (the new staged way)
        series2 = extract_graphs_from_labels(label_stack)
        track_map = assign_track_ids(label_stack)
        apply_track_map(series2, track_map)

        for frame_idx in series1.frame_indices:
            cells1 = series1.frames[frame_idx].cells
            cells2 = series2.frames[frame_idx].cells
            assert set(cells1.keys()) == set(cells2.keys())
            for cell_id in cells1:
                assert cells1[cell_id].track_id == cells2[cell_id].track_id

    def test_apply_track_map_empty_map(self, label_stack):
        """Empty track map should leave all track_ids as None."""
        series = extract_graphs_from_labels(label_stack)
        apply_track_map(series, {})
        assert has_tracking(series) is False


class TestSegmentationPipelineOrder:
    """Segmentation pipeline should work as Track→Extract→Analyze."""

    def test_track_then_extract_matches_monolithic(self, label_stack):
        """Track→Extract+apply should produce same result as monolithic."""
        # Monolithic
        mono = build_from_labels(label_stack, min_iou=0.3)

        # Staged: Track first, then Extract + apply_track_map
        track_map = assign_track_ids(label_stack, min_iou=0.3)
        series = extract_graphs_from_labels(label_stack)
        apply_track_map(series, track_map)

        for frame_idx in mono.frame_indices:
            mono_cells = mono.frames[frame_idx].cells
            staged_cells = series.frames[frame_idx].cells

            assert set(mono_cells.keys()) == set(staged_cells.keys())
            for cell_id in mono_cells:
                assert mono_cells[cell_id].track_id == staged_cells[cell_id].track_id


class TestStagedEqualsMonolithic:
    """Staged pipeline should produce identical results to monolithic."""

    def test_staged_equals_monolithic_labels(self, label_stack):
        # Monolithic
        mono = build_from_labels(label_stack, min_iou=0.3)

        # Staged
        staged = extract_graphs_from_labels(label_stack)
        assign_tracking_labels(staged, label_stack, min_iou=0.3)

        for frame_idx in mono.frame_indices:
            mono_cells = mono.frames[frame_idx].cells
            staged_cells = staged.frames[frame_idx].cells

            assert set(mono_cells.keys()) == set(staged_cells.keys())
            for cell_id in mono_cells:
                assert mono_cells[cell_id].track_id == staged_cells[cell_id].track_id

