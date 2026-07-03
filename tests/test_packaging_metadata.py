from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib


def _pyproject() -> dict:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_heavy_workflow_dependencies_are_optional_extras() -> None:
    project = _pyproject()["project"]
    dependencies = project["dependencies"]
    extras = project["optional-dependencies"]

    assert not any(dep.startswith("cellpose") for dep in dependencies)
    assert not any(dep.startswith("torch") for dep in dependencies)
    assert not any(dep.startswith("ultrack") for dep in dependencies)

    assert "cellpose>=4.0" in extras["cellpose"]
    assert "torch" in extras["cellpose"]
    assert "torchvision" in extras["cellpose"]
    assert "ultrack" in extras["tracking"]
    assert "cellflow[cellpose,tracking]" in extras["all"]
    assert "tomli>=2.0; python_version < '3.11'" in extras["test"]
    assert "tomli>=2.0; python_version < '3.11'" in extras["dev"]


def _python_names(manifest_text: str) -> set[str]:
    return set(re.findall(r"python_name:\s*(\S+)", manifest_text))


def test_full_install_folds_extracted_pieces_into_single_napari_manifest() -> None:
    # A Python distribution may register exactly one napari manifest, and npe2
    # requires its name to match the distribution. The full cellflow install
    # therefore cannot re-register the standalone cellflow-contact-analysis /
    # cellflow-tracking manifests; instead it folds their widgets into the single
    # cellflow manifest so every extracted piece stays reachable from the full
    # install without installing the sub-distributions.
    manifests = _pyproject()["project"]["entry-points"]["napari.manifest"]
    assert manifests == {"cellflow": "cellflow:napari.yaml"}

    folded = (Path("src") / "cellflow" / "napari.yaml").read_text(encoding="utf-8")
    assert "name: cellflow" in folded
    folded_factories = _python_names(folded)

    # The orchestrator's own widget is registered alongside the folded pieces.
    assert "cellflow.napari:CellFlowWidget" in folded_factories

    # Every standalone piece's widget factory is mirrored verbatim in the folded
    # manifest, so the full install and a standalone install expose the same
    # callable for each piece.
    for piece in ("contact_analysis", "tracking_ultrack", "cellpose"):
        standalone = (
            Path("src") / "cellflow" / piece / "napari.yaml"
        ).read_text(encoding="utf-8")
        standalone_factories = _python_names(standalone)
        assert standalone_factories
        assert standalone_factories <= folded_factories


def test_publication_metadata_includes_repository_readme_and_classifiers() -> None:
    project = _pyproject()["project"]

    assert project["readme"] == "README.md"
    assert project["authors"] == [
        {"name": "Artur Ruppel", "email": "artur@ruppel.pro"}
    ]
    assert "Intended Audience :: Science/Research" in project["classifiers"]
    assert "Framework :: napari" in project["classifiers"]
    assert (
        "License :: OSI Approved :: GNU Affero General Public License v3"
        in project["classifiers"]
    )


def test_publication_metadata_exposes_project_urls() -> None:
    urls = _pyproject()["project"]["urls"]

    assert urls["Homepage"] == "https://github.com/ArturRuppel/CellFlow"
    assert urls["Repository"] == "https://github.com/ArturRuppel/CellFlow"
    assert urls["Issues"] == "https://github.com/ArturRuppel/CellFlow/issues"


def test_citation_file_contains_provisional_release_metadata() -> None:
    citation = Path("CITATION.cff").read_text(encoding="utf-8")

    assert "cff-version: 1.2.0" in citation
    assert 'title: "CellFlow"' in citation
    assert 'version: "0.2.0"' in citation
    assert "repository-code: https://github.com/ArturRuppel/CellFlow" in citation
    assert "email: artur@ruppel.pro" in citation


def test_sdist_excludes_internal_development_material() -> None:
    sdist = _pyproject()["tool"]["hatch"]["build"]["targets"]["sdist"]

    assert sdist["exclude"] == [
        "/docs/superpowers",
        "/notes",
        "/notebooks",
        "/scripts/__pycache__",
        "/dist",
        "/.claude",
        "/.qwen",
        "*.pyc",
        "*.nbi",
        "*.nbc",
    ]


def test_notebook_checkpoints_are_not_present_in_repository() -> None:
    checkpoint_files = [
        path
        for path in Path("notebooks").glob(".ipynb_checkpoints/**")
        if path.is_file()
    ]

    assert checkpoint_files == []


def test_ruff_rule_set_covers_low_noise_publication_hygiene() -> None:
    ruff = _pyproject()["tool"]["ruff"]
    lint = ruff["lint"]

    assert ruff["extend-exclude"] == [
        "docs/superpowers",
        "notes",
        "notebooks",
        "scripts/__pycache__",
    ]

    assert lint["select"] == [
        "E4",
        "E9",
        "F401",
        "F821",
        "F811",
        "UP",
    ]
