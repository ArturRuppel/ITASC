"""Contract tests for the independently-installable CellFlow distributions.

The pilot extraction (``cellflow-contact`` + its ``cellflow-core`` substrate) is
built from the shared ``src/cellflow`` tree via hatchling ``force-include`` and
must compose with the orchestrator as PEP 420 namespace packages. These tests
pin that contract without requiring a build.
"""
from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

PACKAGES = Path(__file__).resolve().parents[1] / "packages"


def _load(pkg: str) -> dict:
    return tomllib.loads((PACKAGES / pkg / "pyproject.toml").read_text(encoding="utf-8"))


def test_core_and_contact_distributions_exist_and_name_themselves() -> None:
    assert _load("cellflow-core")["project"]["name"] == "cellflow-core"
    assert _load("cellflow-contact")["project"]["name"] == "cellflow-contact"


def test_extracted_wheels_omit_namespace_roots_for_pep420() -> None:
    # Neither distribution may ship cellflow/__init__.py or cellflow/napari/__init__.py,
    # or the namespace would stop composing across distributions.
    for pkg in ("cellflow-core", "cellflow-contact"):
        force_include = _load(pkg)["tool"]["hatch"]["build"]["targets"]["wheel"][
            "force-include"
        ]
        targets = set(force_include.values())
        assert "cellflow/__init__.py" not in targets
        assert "cellflow/napari/__init__.py" not in targets


def test_contact_depends_on_core_and_not_on_heavy_deps() -> None:
    deps = _load("cellflow-contact")["project"]["dependencies"]
    assert any(d == "cellflow-core" or d.startswith("cellflow-core") for d in deps)
    assert not any(d.startswith(("ultrack", "cellpose", "torch")) for d in deps)


def test_contact_registers_its_own_napari_manifest() -> None:
    project = _load("cellflow-contact")["project"]
    entry = project["entry-points"]["napari.manifest"]
    assert entry["cellflow-contact"] == "cellflow.contact_analysis:napari.yaml"

    manifest = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "cellflow"
        / "contact_analysis"
        / "napari.yaml"
    )
    text = manifest.read_text(encoding="utf-8")
    assert "name: cellflow-contact" in text
    assert (
        "cellflow.napari.contact_analysis_widget:make_contact_analysis_widget" in text
    )
