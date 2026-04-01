"""Tests for the Dash analysis dashboard."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import make_label_stack

from cellflow.backend.graph import build_from_labels
from cellflow.backend.topology import detect_t1_events
from cellflow.utils.io import save_dataset
from cellflow.backend.trajectories import build_edge_trajectories
from cellflow.utils.structures import TissueGraphDataset


@pytest.fixture
def saved_dataset(tmp_path):
    """Build and save a dataset, return the path."""
    stack = make_label_stack(n_frames=5, n_cells_side=4, image_size=200)
    series = build_from_labels(stack, pixel_size=1.0, time_interval=1.0)
    detect_t1_events(series)
    build_edge_trajectories(series, series.t1_events)
    ds = TissueGraphDataset(condition="test")
    ds.add_tissue(series)
    out = tmp_path / "test_dataset"
    save_dataset(ds, out)
    return str(out)


class TestDashAppCreation:
    def test_create_app_no_dataset(self):
        from cellflow.dashboard.app import create_app
        app = create_app()
        assert app is not None

    def test_create_app_with_dataset(self, saved_dataset):
        from cellflow.dashboard.app import create_app
        app = create_app(dataset_path=saved_dataset)
        assert app is not None

    def test_app_has_layout(self):
        from cellflow.dashboard.app import create_app
        app = create_app()
        assert app.layout is not None


class TestWidgetMapping:
    def test_int_param(self):
        from cellflow.dashboard.app import _make_input
        from cellflow.backend.analysis_modules import Parameter, ParamType
        p = Parameter(name="x", label="X", type=ParamType.INT, default=5, min=0, max=10)
        div = _make_input(p)
        assert div is not None

    def test_bool_param(self):
        from cellflow.dashboard.app import _make_input
        from cellflow.backend.analysis_modules import Parameter, ParamType
        p = Parameter(name="flag", label="Flag", type=ParamType.BOOL, default=True)
        div = _make_input(p)
        assert div is not None

    def test_choice_param(self):
        from cellflow.dashboard.app import _make_input
        from cellflow.backend.analysis_modules import Parameter, ParamType
        p = Parameter(name="m", label="M", type=ParamType.CHOICE, default="a", choices=["a", "b"])
        div = _make_input(p)
        assert div is not None

    def test_read_bool_value(self):
        from cellflow.dashboard.app import _read_widget_value
        from cellflow.backend.analysis_modules import Parameter, ParamType
        p = Parameter(name="flag", label="Flag", type=ParamType.BOOL, default=True)
        assert _read_widget_value(p, ["on"]) is True
        assert _read_widget_value(p, []) is False

    def test_read_int_value(self):
        from cellflow.dashboard.app import _read_widget_value
        from cellflow.backend.analysis_modules import Parameter, ParamType
        p = Parameter(name="x", label="X", type=ParamType.INT, default=5)
        assert _read_widget_value(p, 10) == 10
        assert isinstance(_read_widget_value(p, 10), int)


class TestParamExtraction:
    def test_extract_params_from_children(self):
        from cellflow.dashboard.app import _extract_params_from_children
        from cellflow.backend.analysis_modules import Parameter, ParamType

        specs = [
            Parameter(name="n_bins", label="Bins", type=ParamType.INT, default=30),
            Parameter(name="flag", label="Flag", type=ParamType.BOOL, default=False),
        ]
        # Simulate Dash component tree as dicts
        children = {
            "props": {
                "children": [
                    {
                        "props": {
                            "children": [
                                {"props": {"id": {"type": "param-input", "name": "n_bins"}, "value": 50}},
                                {"props": {"children": "help text"}},
                            ]
                        }
                    },
                    {
                        "props": {
                            "children": [
                                {"props": {"id": {"type": "param-input", "name": "flag"}, "value": ["on"]}},
                            ]
                        }
                    },
                ]
            }
        }
        result = _extract_params_from_children(specs, children)
        assert result["n_bins"] == 50
        assert result["flag"] is True

    def test_missing_params_get_defaults(self):
        from cellflow.dashboard.app import _extract_params_from_children
        from cellflow.backend.analysis_modules import Parameter, ParamType

        specs = [
            Parameter(name="n_bins", label="Bins", type=ParamType.INT, default=30),
        ]
        result = _extract_params_from_children(specs, [])
        assert result["n_bins"] == 30
