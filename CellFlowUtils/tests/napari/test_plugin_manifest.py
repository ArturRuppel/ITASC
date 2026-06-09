from __future__ import annotations

import importlib
from pathlib import Path


def test_manifest_exposes_private_utility_widgets():
    manifest_text = Path("src/cellflow_utils/napari.yaml").read_text(encoding="utf-8")

    assert "cellflow-utils.data_prep_widget" in manifest_text
    assert "cellflow-utils.hpc_cellpose_widget" in manifest_text
    assert "cellflow-utils.nls_classification_widget" in manifest_text
    assert "cellflow_utils.napari:DataPrepStandaloneWidget" in manifest_text
    assert "cellflow_utils.napari:HpcCellposeWidget" in manifest_text
    assert "cellflow_utils.napari:NLSClassificationWidget" in manifest_text


def test_plugin_exports_widget_classes():
    module = importlib.import_module("cellflow_utils.napari")

    assert module.__all__ == [
        "DataPrepStandaloneWidget",
        "HpcCellposeWidget",
        "NLSClassificationWidget",
    ]
    assert hasattr(module, "DataPrepStandaloneWidget")
    assert hasattr(module, "HpcCellposeWidget")
    assert hasattr(module, "NLSClassificationWidget")
