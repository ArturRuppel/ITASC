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


def test_hpc_cellpose_controls_live_in_data_preparation_entry_point():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    main_source = (package_root / "main_widget.py").read_text()
    data_prep_source = (package_root / "data_prep_standalone_widget.py").read_text()

    assert '"HPC Cellpose"' not in main_source
    assert '"HPC Cellpose"' in data_prep_source
    assert "HpcCellposeWidget" in data_prep_source
