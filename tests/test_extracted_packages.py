"""Contract tests for the independently-installable ITASC distributions.

The pilot extraction (``itasc-aggregate`` + its ``itasc-core`` substrate) is
built from the shared ``src/itasc`` tree via hatchling ``force-include`` and
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
    assert _load("itasc-core")["project"]["name"] == "itasc-core"
    assert _load("itasc-aggregate")["project"]["name"] == "itasc-aggregate"
    assert _load("itasc-tracking")["project"]["name"] == "itasc-tracking"


def test_extracted_wheels_omit_namespace_roots_for_pep420() -> None:
    # No distribution may ship itasc/__init__.py or itasc/napari/__init__.py,
    # or the namespace would stop composing across distributions.
    for pkg in ("itasc-core", "itasc-aggregate", "itasc-tracking"):
        force_include = _load(pkg)["tool"]["hatch"]["build"]["targets"]["wheel"][
            "force-include"
        ]
        targets = set(force_include.values())
        assert "itasc/__init__.py" not in targets
        assert "itasc/napari/__init__.py" not in targets


def test_tracking_depends_on_core_and_keeps_ultrack_optional() -> None:
    project = _load("itasc-tracking")["project"]
    deps = project["dependencies"]
    assert any(d == "itasc-core" or d.startswith("itasc-core") for d in deps)
    # The heavy Ultrack solver is an extra, not a hard dependency, and the piece
    # must never depend on the contact piece or cellpose.
    assert not any(d.startswith(("ultrack", "cellpose", "torch", "itasc-aggregate")) for d in deps)
    assert "ultrack" in project["optional-dependencies"]["solve"]


def test_tracking_does_not_ship_core_or_contact_owned_modules() -> None:
    # These namespace modules belong to sibling distributions; shipping them from
    # tracking too would collide at install time.
    targets = set(
        _load("itasc-tracking")["tool"]["hatch"]["build"]["targets"]["wheel"][
            "force-include"
        ].values()
    )
    for owned_elsewhere in (
        "itasc/napari/_track_render.py",  # core
        "itasc/napari/_paths.py",  # core
        "itasc/napari/widgets.py",  # core
        "itasc/napari/contact_visualization.py",  # contact
    ):
        assert owned_elsewhere not in targets


def test_tracking_registers_its_own_napari_manifest() -> None:
    project = _load("itasc-tracking")["project"]
    entry = project["entry-points"]["napari.manifest"]
    assert entry["itasc-tracking"] == "itasc.tracking_ultrack:napari.yaml"

    manifest = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "itasc"
        / "tracking_ultrack"
        / "napari.yaml"
    )
    text = manifest.read_text(encoding="utf-8")
    assert "name: itasc-tracking" in text
    assert (
        "itasc.napari.nucleus_workflow_widget:make_nucleus_tracking_widget" in text
    )


def test_contact_depends_on_core_and_not_on_heavy_deps() -> None:
    deps = _load("itasc-aggregate")["project"]["dependencies"]
    assert any(d == "itasc-core" or d.startswith("itasc-core") for d in deps)
    assert not any(d.startswith(("ultrack", "cellpose", "torch")) for d in deps)


def test_contact_registers_its_own_napari_manifest() -> None:
    project = _load("itasc-aggregate")["project"]
    entry = project["entry-points"]["napari.manifest"]
    assert entry["itasc-aggregate"] == "itasc.contact_analysis:napari.yaml"

    manifest = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "itasc"
        / "contact_analysis"
        / "napari.yaml"
    )
    text = manifest.read_text(encoding="utf-8")
    assert "name: itasc-aggregate" in text
    # The standalone widget is the ITASC Aggregate app: the full catalog UI
    # restricted to the Contact Analysis stage (upstream stages omitted).
    assert "itasc.napari.main_widget:make_aggregate_app_widget" in text


def test_aggregate_wheel_ships_the_catalog_app_shell_without_upstream_widgets() -> None:
    # The standalone aggregate app is the main widget in contact-only mode, so the
    # wheel must ship the app shell (main widget, the Data-folders panel, the
    # status rail/status/loader cluster, the aggregate capstone, icons). It must
    # NOT ship the upstream stage widgets (their distributions supply those) nor
    # any core-owned module (that would collide with itasc-core at install).
    targets = set(
        _load("itasc-aggregate")["tool"]["hatch"]["build"]["targets"]["wheel"][
            "force-include"
        ].values()
    )
    for shipped in (
        "itasc/napari/main_widget.py",
        "itasc/napari/aggregate_widget.py",
        "itasc/napari/_experiments_panel.py",
        "itasc/napari/_status_rail.py",
        "itasc/napari/_stage_status.py",
        "itasc/napari/_stage_loader.py",
        "itasc/napari/_icons.py",
    ):
        assert shipped in targets
    for not_shipped in (
        "itasc/napari/cellpose_widget.py",
        "itasc/napari/nucleus_workflow_widget.py",
        "itasc/napari/cell_workflow_widget.py",
        "itasc/napari/widgets.py",  # core
        "itasc/napari/_paths.py",  # core
        "itasc/napari/ui_style.py",  # core
    ):
        assert not_shipped not in targets
