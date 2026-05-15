from __future__ import annotations

from pathlib import Path


def test_cellpose_status_panels_track_cell_flow_vectors():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    cellpose_source = (package_root / "main_widget.py").read_text()
    data_panel_source = (package_root / "data_panel_widget.py").read_text()

    assert "1_cellpose/cell_dp_3dt.tif" in cellpose_source
    assert "1_cellpose/cell_dp_3dt.tif" in data_panel_source


def test_cellpose_stage_places_hpc_controls_between_inputs_and_outputs():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    cellpose_source = (package_root / "main_widget.py").read_text()

    inputs_index = cellpose_source.index('"Inputs"')
    hpc_index = cellpose_source.index('"HPC Cellpose"')
    outputs_index = cellpose_source.index('"Outputs"')

    assert inputs_index < hpc_index < outputs_index
    assert "2b. HPC Cellpose" not in cellpose_source
