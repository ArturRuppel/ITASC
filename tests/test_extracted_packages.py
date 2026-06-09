"""Contract tests for the independently-installable CellFlow distributions.

The pilot extraction (``cellflow-aggregate`` + its ``cellflow-core`` substrate) is
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
    assert _load("cellflow-aggregate")["project"]["name"] == "cellflow-aggregate"
    assert _load("cellflow-tracking")["project"]["name"] == "cellflow-tracking"


def test_extracted_wheels_omit_namespace_roots_for_pep420() -> None:
    # No distribution may ship cellflow/__init__.py or cellflow/napari/__init__.py,
    # or the namespace would stop composing across distributions.
    for pkg in ("cellflow-core", "cellflow-aggregate", "cellflow-tracking"):
        force_include = _load(pkg)["tool"]["hatch"]["build"]["targets"]["wheel"][
            "force-include"
        ]
        targets = set(force_include.values())
        assert "cellflow/__init__.py" not in targets
        assert "cellflow/napari/__init__.py" not in targets


def test_tracking_depends_on_core_and_keeps_ultrack_optional() -> None:
    project = _load("cellflow-tracking")["project"]
    deps = project["dependencies"]
    assert any(d == "cellflow-core" or d.startswith("cellflow-core") for d in deps)
    # The heavy Ultrack solver is an extra, not a hard dependency, and the piece
    # must never depend on the contact piece or cellpose.
    assert not any(d.startswith(("ultrack", "cellpose", "torch", "cellflow-aggregate")) for d in deps)
    assert "ultrack" in project["optional-dependencies"]["solve"]


def test_tracking_does_not_ship_core_or_contact_owned_modules() -> None:
    # These namespace modules belong to sibling distributions; shipping them from
    # tracking too would collide at install time.
    targets = set(
        _load("cellflow-tracking")["tool"]["hatch"]["build"]["targets"]["wheel"][
            "force-include"
        ].values()
    )
    for owned_elsewhere in (
        "cellflow/napari/_track_render.py",  # core
        "cellflow/napari/_paths.py",  # core
        "cellflow/napari/widgets.py",  # core
        "cellflow/napari/contact_visualization.py",  # contact
    ):
        assert owned_elsewhere not in targets


def test_tracking_registers_its_own_napari_manifest() -> None:
    project = _load("cellflow-tracking")["project"]
    entry = project["entry-points"]["napari.manifest"]
    assert entry["cellflow-tracking"] == "cellflow.tracking_ultrack:napari.yaml"

    manifest = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "cellflow"
        / "tracking_ultrack"
        / "napari.yaml"
    )
    text = manifest.read_text(encoding="utf-8")
    assert "name: cellflow-tracking" in text
    assert (
        "cellflow.napari.nucleus_workflow_widget:make_nucleus_tracking_widget" in text
    )


def test_contact_depends_on_core_and_not_on_heavy_deps() -> None:
    deps = _load("cellflow-aggregate")["project"]["dependencies"]
    assert any(d == "cellflow-core" or d.startswith("cellflow-core") for d in deps)
    assert not any(d.startswith(("ultrack", "cellpose", "torch")) for d in deps)


def test_contact_registers_its_own_napari_manifest() -> None:
    project = _load("cellflow-aggregate")["project"]
    entry = project["entry-points"]["napari.manifest"]
    assert entry["cellflow-aggregate"] == "cellflow.aggregate_quantification:napari.yaml"

    manifest = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "cellflow"
        / "aggregate_quantification"
        / "napari.yaml"
    )
    text = manifest.read_text(encoding="utf-8")
    assert "name: cellflow-aggregate" in text
    assert (
        "cellflow.napari.aggregate_quantification_widget:make_aggregate_quantification_widget" in text
    )
