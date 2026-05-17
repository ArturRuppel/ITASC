from __future__ import annotations

from pathlib import Path


def test_cellpose_status_panels_track_cell_flow_vectors():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    cellpose_source = (package_root / "main_widget.py").read_text()
    data_panel_source = (package_root / "data_panel_widget.py").read_text()

    assert "1_cellpose/cell_dp_3dt.tif" in cellpose_source
    assert "1_cellpose/cell_dp_3dt.tif" in data_panel_source


def test_cellpose_status_panels_track_probability_zavgs():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    cellpose_source = (package_root / "main_widget.py").read_text()
    data_panel_source = (package_root / "data_panel_widget.py").read_text()

    for source in (cellpose_source, data_panel_source):
        assert "1_cellpose/cell_prob_zavg.tif" in source
        assert "1_cellpose/nucleus_prob_zavg.tif" in source


def test_segmentation_widgets_use_probability_zavgs_as_inputs():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    nucleus_source = (package_root / "nucleus_workflow_widget.py").read_text()
    cell_source = (package_root / "cell_workflow_widget.py").read_text()

    for source in (nucleus_source, cell_source):
        assert "1_cellpose/cell_prob_zavg.tif" in source
        assert "1_cellpose/nucleus_prob_zavg.tif" in source


def test_cellpose_stage_groups_pipeline_files_before_hpc_controls():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    cellpose_source = (package_root / "main_widget.py").read_text()

    pipeline_files_index = cellpose_source.index('"Pipeline Files"')
    hpc_index = cellpose_source.index('"HPC Cellpose"')

    assert pipeline_files_index < hpc_index
    assert '("Inputs", [' in cellpose_source
    assert '("Outputs", [' in cellpose_source
    assert "2b. HPC Cellpose" not in cellpose_source
