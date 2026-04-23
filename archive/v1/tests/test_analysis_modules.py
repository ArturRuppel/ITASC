"""Tests for analysis module interface and built-in modules."""
import sys
from pathlib import Path

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from cellflow.backend.analysis_modules import (
    AnalysisModule,
    AnalysisResult,
    Parameter,
    ParamType,
    discover_modules,
)
from cellflow.backend.builtins import (
    JunctionLengthDistribution,
    T1TransitionRate,
    CellDistributions,
    EventTriggeredAveraging,
)
from cellflow.backend.graph import build_from_labels
from cellflow.backend.topology import detect_t1_events
from cellflow.backend.trajectories import build_edge_trajectories

from conftest import make_label_stack


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def series():
    """Build a TissueGraphTimeSeries from synthetic labels."""
    stack = make_label_stack(n_frames=5, n_cells_side=4, image_size=200)
    s = build_from_labels(stack, pixel_size=1.0, time_interval=1.0)
    detect_t1_events(s)
    build_edge_trajectories(s, s.t1_events)
    return s


# ------------------------------------------------------------------
# Base class / interface tests
# ------------------------------------------------------------------


class TestAnalysisModuleInterface:
    def test_builtin_modules_are_subclasses(self):
        for cls in [JunctionLengthDistribution, T1TransitionRate,
                    CellDistributions, EventTriggeredAveraging]:
            assert issubclass(cls, AnalysisModule)

    def test_module_has_required_properties(self):
        mod = JunctionLengthDistribution()
        assert isinstance(mod.name, str) and mod.name
        assert isinstance(mod.description, str) and mod.description
        params = mod.parameters()
        assert isinstance(params, list)
        assert all(isinstance(p, Parameter) for p in params)

    def test_validate_params_fills_defaults(self):
        mod = JunctionLengthDistribution()
        clean = mod.validate_params()
        assert clean["n_bins"] == 30
        assert clean["normalize"] is True

    def test_validate_params_clamps_range(self):
        mod = JunctionLengthDistribution()
        clean = mod.validate_params(n_bins=1000)
        assert clean["n_bins"] == 200  # max

    def test_validate_params_rejects_invalid_choice(self):
        mod = CellDistributions()
        clean = mod.validate_params(metric="nonexistent")
        assert clean["metric"] == "area"  # falls back to default


# ------------------------------------------------------------------
# Discovery tests
# ------------------------------------------------------------------


class TestDiscovery:
    def test_discover_finds_builtins(self):
        modules = discover_modules()
        assert "junction_length_distribution" in modules
        assert "t1_transition_rate" in modules
        assert "cell_distributions" in modules
        assert "event_triggered_averaging" in modules

    def test_discovered_modules_are_classes(self):
        modules = discover_modules()
        for name, cls in modules.items():
            assert issubclass(cls, AnalysisModule)


# ------------------------------------------------------------------
# JunctionLengthDistribution
# ------------------------------------------------------------------


class TestJunctionLengthDistribution:
    def test_compute_returns_result(self, series):
        mod = JunctionLengthDistribution()
        result = mod.compute(series)
        assert isinstance(result, AnalysisResult)
        assert "main" in result.tables
        assert not result.tables["main"].empty
        assert "length" in result.tables["main"].columns

    def test_compute_with_exclude_tags(self, series):
        mod = JunctionLengthDistribution()
        result = mod.compute(series, exclude_tags="edge_border")
        assert isinstance(result, AnalysisResult)

    def test_per_frame_summary(self, series):
        mod = JunctionLengthDistribution()
        result = mod.compute(series)
        assert "per_frame_summary" in result.tables
        summary = result.tables["per_frame_summary"]
        assert "frame" in summary.columns
        assert "mean_length" in summary.columns

    def test_metadata(self, series):
        mod = JunctionLengthDistribution()
        result = mod.compute(series)
        assert "n_junctions" in result.metadata
        assert "mean_length" in result.metadata

    def test_visualize_returns_plotly_figures(self, series):
        import plotly.graph_objects as go
        mod = JunctionLengthDistribution()
        result = mod.compute(series)
        figs = mod.visualize(result)
        assert len(figs) >= 1
        for fig in figs:
            assert isinstance(fig, go.Figure)


# ------------------------------------------------------------------
# T1TransitionRate
# ------------------------------------------------------------------


class TestT1TransitionRate:
    def test_compute_returns_result(self, series):
        mod = T1TransitionRate()
        result = mod.compute(series)
        assert isinstance(result, AnalysisResult)
        assert "rate" in result.tables
        assert "events" in result.tables

    def test_rate_table_has_columns(self, series):
        mod = T1TransitionRate()
        result = mod.compute(series)
        rate = result.tables["rate"]
        assert "frame" in rate.columns
        assert "t1_rate" in rate.columns

    def test_visualize(self, series):
        import plotly.graph_objects as go
        mod = T1TransitionRate()
        result = mod.compute(series)
        figs = mod.visualize(result)
        for fig in figs:
            assert isinstance(fig, go.Figure)


# ------------------------------------------------------------------
# CellDistributions
# ------------------------------------------------------------------


class TestCellDistributions:
    def test_compute_area(self, series):
        mod = CellDistributions()
        result = mod.compute(series, metric="area")
        assert result.metadata["metric"] == "area"
        assert not result.tables["main"].empty

    def test_compute_shape_index(self, series):
        mod = CellDistributions()
        result = mod.compute(series, metric="shape_index")
        assert result.metadata["metric"] == "shape_index"

    def test_compute_num_neighbors(self, series):
        mod = CellDistributions()
        result = mod.compute(series, metric="num_neighbors")
        assert result.metadata["metric"] == "num_neighbors"

    def test_visualize(self, series):
        import plotly.graph_objects as go
        mod = CellDistributions()
        result = mod.compute(series, metric="area")
        figs = mod.visualize(result, metric="area")
        assert len(figs) >= 1
        for fig in figs:
            assert isinstance(fig, go.Figure)


# ------------------------------------------------------------------
# EventTriggeredAveraging
# ------------------------------------------------------------------


class TestEventTriggeredAveraging:
    def test_compute_returns_result(self, series):
        mod = EventTriggeredAveraging()
        result = mod.compute(series, window_before=5, window_after=5)
        assert isinstance(result, AnalysisResult)
        assert "main" in result.tables
        assert "aligned" in result.tables

    def test_visualize(self, series):
        import plotly.graph_objects as go
        mod = EventTriggeredAveraging()
        result = mod.compute(series, window_before=5, window_after=5)
        figs = mod.visualize(result, window_before=5, window_after=5)
        for fig in figs:
            assert isinstance(fig, go.Figure)
