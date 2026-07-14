from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_public_manifest_exposes_only_main_itasc_widget():
    manifest_text = Path("src/itasc/napari.yaml").read_text(encoding="utf-8")

    assert "itasc.main_widget" in manifest_text
    assert "ITASCWidget" in manifest_text
    assert "DataPrepStandaloneWidget" not in manifest_text
    assert "data_prep_widget" not in manifest_text


def test_public_napari_init_does_not_import_personal_widgets():
    for module_name in (
        "itasc.napari",
        "itasc.napari.data_prep_standalone_widget",
        "itasc.napari.data_prep_widget",
        "itasc.napari.hpc_cellpose_widget",
    ):
        sys.modules.pop(module_name, None)

    napari_module = importlib.import_module("itasc.napari")

    assert napari_module.__all__ == ["ITASCWidget"]
    assert not hasattr(napari_module, "DataPrepStandaloneWidget")
    assert "itasc.napari.data_prep_standalone_widget" not in sys.modules
    assert "itasc.napari.data_prep_widget" not in sys.modules
    assert "itasc.napari.hpc_cellpose_widget" not in sys.modules


def test_pyproject_does_not_publish_personal_console_scripts():
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "itasc-classify-nls" not in pyproject_text


def test_public_contact_analysis_widget_has_no_nls_widget_coupling():
    for module_name in (
        "itasc.napari.contact_analysis_widget",
        "itasc.napari.nls_classification_widget",
        "itasc.contact_analysis.contacts.nls_classification",
    ):
        sys.modules.pop(module_name, None)

    module = importlib.import_module("itasc.napari.contact_analysis_widget")

    assert not hasattr(module.ContactAnalysisWidget, "nls_classification_widget")
    assert "itasc.napari.nls_classification_widget" not in sys.modules
    assert "itasc.contact_analysis.contacts.nls_classification" not in sys.modules
