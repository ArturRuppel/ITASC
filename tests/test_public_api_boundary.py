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
import itasc
import json
import sys

print(json.dumps({
    "all": itasc.__all__,
    "has_version": hasattr(itasc, "__version__"),
    "version": itasc.__version__,
    "has_itasc_widget": hasattr(itasc, "ITASCWidget"),
    "has_tracking_config": hasattr(itasc, "TrackingConfig"),
    "imports": sorted(
        name for name in sys.modules
        if name in {"napari", "cellpose", "torch", "ultrack", "itasc.napari"}
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
    itasc = _run_isolated_import_probe()

    assert itasc["all"] == ["__version__"]
    assert itasc["has_version"]
    assert not itasc["has_itasc_widget"]
    assert not itasc["has_tracking_config"]


def test_top_level_version_matches_project_metadata() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    itasc = importlib.import_module("itasc")

    assert itasc.__version__ == project["version"]


def test_top_level_import_does_not_import_ui_or_optional_workflow_modules() -> None:
    itasc = _run_isolated_import_probe()

    assert itasc["imports"] == []


def test_readme_documents_public_api_boundary() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "## Programmatic use" in readme
    assert "`import itasc` exposes only `__version__`" in readme
    assert "napari plugin" in readme
    assert "provisional" in readme


def test_active_code_does_not_reference_deprecated_h5_candidate_workflow() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    forbidden = (
        "hyp" + "otheses" + ".h5",
        "ingest_" + "hyp" + "otheses_to_db",
    )
    allowed_files = {
        Path("tests/test_public_api_boundary.py"),
    }

    offenders: list[str] = []
    for root in (repo_root / "src", repo_root / "tests"):
        for path in root.rglob("*.py"):
            rel_path = path.relative_to(repo_root)
            if rel_path in allowed_files:
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                if pattern in text:
                    offenders.append(f"{rel_path}: {pattern}")

    assert offenders == []
