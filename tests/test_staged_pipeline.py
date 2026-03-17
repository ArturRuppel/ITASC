"""Tests for staged pipeline: extract (Stage 1) → track (Stage 2) → equivalence."""
import numpy as np
import pytest

from napariTissueGraph.core.graph import (
    build_from_labels,
    build_from_tracks,
    extract_graphs_from_labels,
    extract_graphs_from_tracks,
    assign_tracking_labels,
    has_tracking,
)
from napariTissueGraph.structures import InputType


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

    def test_extract_graphs_from_tracks_no_tracking(self, track_positions):
        series = extract_graphs_from_tracks(track_positions, image_shape=(100, 100))
        assert series.num_frames == 5
        assert series.input_type == InputType.VORONOI
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

    def test_staged_equals_monolithic_tracks(self, track_positions):
        image_shape = (100, 100)
        # Monolithic (no track_ids arg → no tracking)
        mono = build_from_tracks(track_positions, image_shape=image_shape)

        # Staged
        staged = extract_graphs_from_tracks(track_positions, image_shape=image_shape)

        for frame_idx in mono.frame_indices:
            assert len(mono.frames[frame_idx].cells) == len(staged.frames[frame_idx].cells)
            assert len(mono.frames[frame_idx].junctions) == len(staged.frames[frame_idx].junctions)
