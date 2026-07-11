from __future__ import annotations

from pathlib import Path


def test_cellpose_status_panels_track_cell_flow_vectors():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    cellpose_source = (package_root / "cellpose_widget.py").read_text()

    assert "1_cellpose/cell_dp_3dt.tif" in cellpose_source


def test_cellpose_status_panels_do_not_track_probability_zavgs():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    cellpose_source = (package_root / "cellpose_widget.py").read_text()

    assert "1_cellpose/cell_prob_zavg.tif" not in cellpose_source
    assert "1_cellpose/nucleus_prob_zavg.tif" not in cellpose_source
    assert "1_cellpose/cell_foreground.tif" in cellpose_source
    assert "1_cellpose/nucleus_foreground.tif" in cellpose_source


def test_segmentation_widgets_use_cellpose_foregrounds_as_inputs():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    nucleus_source = (package_root / "nucleus_workflow_widget.py").read_text()
    cell_source = (package_root / "cell_workflow_widget.py").read_text()

    assert "1_cellpose/nucleus_foreground.tif" in nucleus_source
    # The simplified cell widget consumes the cached cell divergence maps
    # (contours + foreground) produced upstream by DivergenceMapsWidget.
    assert "1_cellpose/cell_foreground.tif" in cell_source
    assert "1_cellpose/cell_contours.tif" in cell_source


def test_hpc_cellpose_controls_are_not_public_napari_entry_points():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    main_source = (package_root / "main_widget.py").read_text()
    manifest_source = (package_root / ".." / "napari.yaml").read_text()

    assert '"HPC Cellpose"' not in main_source
    assert "HpcCellposeWidget" not in manifest_source
