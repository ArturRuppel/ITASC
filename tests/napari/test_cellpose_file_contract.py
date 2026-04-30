from __future__ import annotations

from pathlib import Path


def test_cellpose_status_panels_track_cell_flow_vectors():
    package_root = Path(__file__).resolve().parents[2] / "src" / "cellflow" / "napari"

    cellpose_source = (package_root / "cellpose_widget.py").read_text()
    data_panel_source = (package_root / "data_panel_widget.py").read_text()

    assert "1_cellpose/cell_dp_3dt.tif" in cellpose_source
    assert "1_cellpose/cell_dp_3dt.tif" in data_panel_source
