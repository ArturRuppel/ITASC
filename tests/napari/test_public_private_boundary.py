from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_public_manifest_exposes_only_main_cellflow_widget():
    manifest_text = Path("src/cellflow/napari.yaml").read_text(encoding="utf-8")

    assert "cellflow.main_widget" in manifest_text
    assert "CellFlowWidget" in manifest_text
    assert "DataPrepStandaloneWidget" not in manifest_text
    assert "data_prep_widget" not in manifest_text


def test_public_napari_init_does_not_import_personal_widgets():
    for module_name in (
        "cellflow.napari",
        "cellflow.napari.data_prep_standalone_widget",
        "cellflow.napari.data_prep_widget",
        "cellflow.napari.hpc_cellpose_widget",
    ):
        sys.modules.pop(module_name, None)

    napari_module = importlib.import_module("cellflow.napari")

    assert napari_module.__all__ == ["CellFlowWidget"]
    assert not hasattr(napari_module, "DataPrepStandaloneWidget")
    assert "cellflow.napari.data_prep_standalone_widget" not in sys.modules
    assert "cellflow.napari.data_prep_widget" not in sys.modules
    assert "cellflow.napari.hpc_cellpose_widget" not in sys.modules


def test_pyproject_does_not_publish_personal_console_scripts():
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "cellflow-classify-nls" not in pyproject_text


def test_public_contact_analysis_widget_has_no_nls_widget_coupling():
    for module_name in (
        "cellflow.napari.contact_analysis_widget",
        "cellflow.napari.nls_classification_widget",
        "cellflow.contact_analysis.nls_classification",
    ):
        sys.modules.pop(module_name, None)

    module = importlib.import_module("cellflow.napari.contact_analysis_widget")

    assert not hasattr(module.ContactAnalysisWidget, "nls_classification_widget")
    assert "cellflow.napari.nls_classification_widget" not in sys.modules
    assert "cellflow.contact_analysis.nls_classification" not in sys.modules
