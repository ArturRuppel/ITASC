"""Tests for core graph building from labels."""
import numpy as np
import pytest

from cellflow.backend.graph import build_from_labels
from cellflow.utils.structures import InputType


class TestBuildFromLabels:
    def test_basic_construction(self, label_stack):
        series = build_from_labels(label_stack)
        assert series.num_frames == 3
        assert series.input_type == InputType.SEGMENTATION

    def test_frames_have_cells(self, label_stack):
        series = build_from_labels(label_stack)
        for frame_idx, frame in series.frames.items():
            assert len(frame.cells) > 0, f"Frame {frame_idx} has no cells"

    def test_frames_have_junctions(self, label_stack):
        series = build_from_labels(label_stack)
        for frame_idx, frame in series.frames.items():
            assert len(frame.junctions) > 0, f"Frame {frame_idx} has no junctions"

    def test_junction_lengths_positive(self, label_stack):
        series = build_from_labels(label_stack)
        for frame in series.frames.values():
            for jd in frame.junctions.values():
                assert jd.length > 0

    def test_cell_areas_positive(self, label_stack):
        series = build_from_labels(label_stack)
        for frame in series.frames.values():
            for cd in frame.cells.values():
                assert cd.area > 0

    def test_graph_edges_match_junctions(self, label_stack):
        series = build_from_labels(label_stack)
        for frame in series.frames.values():
            # Border junctions (cell_id=0 in the pair) are stored in frame.junctions
            # for tagging purposes but do NOT appear as graph edges (background is
            # not a cell node). Only count cell-cell junctions for this comparison.
            cell_cell_junctions = sum(1 for pair in frame.junctions if 0 not in pair)
            assert frame.graph.number_of_edges() == cell_cell_junctions

    def test_single_frame(self, label_frame):
        series = build_from_labels(label_frame[np.newaxis, ...])
        assert series.num_frames == 1

    def test_pixel_size_stored(self, label_stack):
        series = build_from_labels(label_stack, pixel_size=0.65)
        assert series.pixel_size == 0.65
