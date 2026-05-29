from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib


def _run_isolated_import_probe() -> dict:
    script = """
import cellflow
import json
import sys

print(json.dumps({
    "all": cellflow.__all__,
    "has_version": hasattr(cellflow, "__version__"),
    "version": cellflow.__version__,
    "has_cellflow_widget": hasattr(cellflow, "CellFlowWidget"),
    "has_tracking_config": hasattr(cellflow, "TrackingConfig"),
    "imports": sorted(
        name for name in sys.modules
        if name in {"napari", "cellpose", "torch", "ultrack", "cellflow.napari"}
    ),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_top_level_public_api_is_intentionally_small() -> None:
    cellflow = _run_isolated_import_probe()

    assert cellflow["all"] == ["__version__"]
    assert cellflow["has_version"]
    assert not cellflow["has_cellflow_widget"]
    assert not cellflow["has_tracking_config"]


def test_top_level_version_matches_project_metadata() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    cellflow = importlib.import_module("cellflow")

    assert cellflow.__version__ == project["version"]


def test_top_level_import_does_not_import_ui_or_optional_workflow_modules() -> None:
    cellflow = _run_isolated_import_probe()

    assert cellflow["imports"] == []


def test_readme_documents_public_api_boundary() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "## Public API Boundary" in readme
    assert "`import cellflow` exposes only `__version__`" in readme
    assert "napari plugin" in readme
    assert "provisional" in readme
