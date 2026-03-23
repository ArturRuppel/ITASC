"""Tests for ForSys force inference integration."""
import numpy as np
import pytest

from napariTissueGraph.core.forsys_adapter import forsys_available


# Skip all tests if forsys is not installed
pytestmark = pytest.mark.skipif(
    not forsys_available(), reason="forsys not installed"
)


def _build_series_from_labels():
    """Build a small TissueGraphTimeSeries from labels for testing."""
    from tests.conftest import make_label_stack
    from napariTissueGraph.core.graph import build_from_labels

    label_stack = make_label_stack(n_frames=1, n_cells_side=4, image_size=200)
    return build_from_labels(label_stack)


def _build_series_from_voronoi():
    """Build a small TissueGraphTimeSeries from Voronoi for testing."""
    from tests.conftest import make_track_positions
    from napariTissueGraph.core.graph import build_from_tracks

    positions = make_track_positions(n_frames=1, nx=4, ny=4, spacing=25.0)
    return build_from_tracks(positions, image_shape=(120, 120))


class TestForSysAdapter:
    """Test the geometry conversion layer."""

    def test_tissue_frame_to_forsys_voronoi(self):
        from napariTissueGraph.core.forsys_adapter import tissue_frame_to_forsys

        series = _build_series_from_voronoi()
        frame = list(series.frames.values())[0]
        fs_frame = tissue_frame_to_forsys(frame)

        assert len(fs_frame.vertices) > 0
        assert len(fs_frame.edges) > 0
        assert len(fs_frame.cells) > 0

    def test_tissue_frame_to_forsys_labels(self):
        from napariTissueGraph.core.forsys_adapter import tissue_frame_to_forsys

        series = _build_series_from_labels()
        frame = list(series.frames.values())[0]
        fs_frame = tissue_frame_to_forsys(frame)

        assert len(fs_frame.vertices) > 0
        assert len(fs_frame.edges) > 0
        assert len(fs_frame.cells) > 0

    def test_forsys_frame_has_big_edges(self):
        from napariTissueGraph.core.forsys_adapter import tissue_frame_to_forsys

        series = _build_series_from_voronoi()
        frame = list(series.frames.values())[0]
        fs_frame = tissue_frame_to_forsys(frame)

        # Frame.__post_init__ should auto-build big edges
        assert len(fs_frame.big_edges) > 0

    def test_no_vertices_raises(self):
        """Frames without cell vertices should raise ValueError."""
        from napariTissueGraph.core.forsys_adapter import tissue_frame_to_forsys
        from napariTissueGraph.structures import (
            CellData, JunctionData, TissueGraphFrame, InputType,
        )
        import networkx as nx

        # Build a frame with cells that have no vertices
        cells = {
            1: CellData(
                cell_id=1, position=np.array([10.0, 10.0]),
                area=100, perimeter=40, shape_index=4.0, num_neighbors=2,
            ),
        }
        frame = TissueGraphFrame(
            frame=0, graph=nx.Graph(), cells=cells,
            junctions={}, input_type=InputType.SEGMENTATION,
        )
        with pytest.raises(ValueError, match="No internal junctions"):
            tissue_frame_to_forsys(frame)


class TestForceInference:
    """Test the full inference pipeline."""

    def test_infer_forces_voronoi(self):
        from napariTissueGraph.core.mechanics import infer_forces

        series = _build_series_from_voronoi()
        infer_forces(series, method="static")

        # Check that some junctions got tension values
        n_tensions = sum(
            1 for f in series.frames.values()
            for jd in f.junctions.values() if jd.tension is not None
        )
        assert n_tensions > 0, "No tensions were inferred"

    def test_infer_forces_labels(self):
        from napariTissueGraph.core.mechanics import infer_forces

        series = _build_series_from_labels()
        infer_forces(series, method="static")

        n_tensions = sum(
            1 for f in series.frames.values()
            for jd in f.junctions.values() if jd.tension is not None
        )
        assert n_tensions > 0, "No tensions were inferred"

    def test_infer_pressures(self):
        from napariTissueGraph.core.mechanics import infer_forces

        series = _build_series_from_voronoi()
        infer_forces(series, method="static")

        n_pressures = sum(
            1 for f in series.frames.values()
            for cd in f.cells.values() if cd.pressure is not None
        )
        assert n_pressures > 0, "No pressures were inferred"

    def test_tensions_are_finite(self):
        from napariTissueGraph.core.mechanics import infer_forces

        series = _build_series_from_voronoi()
        infer_forces(series, method="static")

        for f in series.frames.values():
            for jd in f.junctions.values():
                if jd.tension is not None:
                    assert np.isfinite(jd.tension), f"Non-finite tension: {jd.tension}"

    def test_dynamic_not_implemented(self):
        from napariTissueGraph.core.mechanics import infer_forces

        series = _build_series_from_voronoi()
        with pytest.raises(NotImplementedError):
            infer_forces(series, method="dynamic")

    def test_invalid_method(self):
        from napariTissueGraph.core.mechanics import infer_forces

        series = _build_series_from_voronoi()
        with pytest.raises(ValueError):
            infer_forces(series, method="invalid")

    def test_multiframe(self):
        """Test inference across multiple frames."""
        from tests.conftest import make_track_positions
        from napariTissueGraph.core.graph import build_from_tracks
        from napariTissueGraph.core.mechanics import infer_forces

        positions = make_track_positions(n_frames=3, nx=4, ny=4, spacing=25.0)
        series = build_from_tracks(positions, image_shape=(120, 120))
        infer_forces(series, method="static")

        # Each frame should have some tensions
        for frame_idx, frame in series.frames.items():
            n = sum(1 for jd in frame.junctions.values() if jd.tension is not None)
            assert n > 0, f"Frame {frame_idx} has no tensions"


class TestResultPersistence:
    """Test that inferred values survive save/load."""

    def test_save_load_round_trip(self, tmp_path):
        from napariTissueGraph.core.mechanics import infer_forces
        from napariTissueGraph.core.io import save_dataset, load_dataset
        from napariTissueGraph.structures import TissueGraphDataset

        series = _build_series_from_voronoi()
        infer_forces(series, method="static")

        dataset = TissueGraphDataset()
        dataset.add_tissue(series)

        save_path = tmp_path / "test_forces"
        save_dataset(dataset, str(save_path))
        loaded = load_dataset(str(save_path))

        loaded_series = loaded.tissues[0]
        for frame_idx in series.frame_indices:
            orig_frame = series.frames[frame_idx]
            load_frame = loaded_series.frames[frame_idx]

            for key, orig_jd in orig_frame.junctions.items():
                if orig_jd.tension is not None:
                    load_jd = load_frame.junctions[key]
                    assert load_jd.tension is not None
                    assert abs(load_jd.tension - orig_jd.tension) < 1e-10

            for cid, orig_cd in orig_frame.cells.items():
                if orig_cd.pressure is not None:
                    load_cd = load_frame.cells[cid]
                    assert load_cd.pressure is not None
                    assert abs(load_cd.pressure - orig_cd.pressure) < 1e-10


class TestVisualization:
    """Test tension/pressure visualization builders."""

    def test_tension_colored_junctions(self):
        from napariTissueGraph.core.mechanics import infer_forces
        from napariTissueGraph.napari.visualization import build_tension_colored_junctions

        series = _build_series_from_voronoi()
        infer_forces(series, method="static")

        lines, colors = build_tension_colored_junctions(series)
        assert len(lines) > 0
        assert len(colors) == len(lines)
        assert colors.shape[1] == 4  # RGBA

    def test_pressure_colored_cells(self):
        from napariTissueGraph.core.mechanics import infer_forces
        from napariTissueGraph.napari.visualization import build_pressure_colored_cells

        series = _build_series_from_voronoi()
        infer_forces(series, method="static")

        polys, colors = build_pressure_colored_cells(series)
        assert len(polys) > 0
        assert len(colors) == len(polys)
        assert colors.shape[1] == 4

    def test_no_tensions_returns_empty(self):
        from napariTissueGraph.napari.visualization import build_tension_colored_junctions

        series = _build_series_from_voronoi()
        # No inference run — all tensions are None
        lines, colors = build_tension_colored_junctions(series)
        assert len(lines) == 0


class TestGracefulFailure:
    """Test behavior when forsys is not available."""

    def test_forsys_available_returns_bool(self):
        from napariTissueGraph.core.forsys_adapter import forsys_available
        assert isinstance(forsys_available(), bool)
