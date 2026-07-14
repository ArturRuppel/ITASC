from __future__ import annotations

from pathlib import Path


WORKFLOW_PATH = Path(".github/workflows/ci.yml")


def _workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def test_ci_workflow_exists() -> None:
    assert WORKFLOW_PATH.exists()


def test_ci_workflow_runs_on_push_and_pull_request() -> None:
    workflow = _workflow_text()

    assert "push:" in workflow
    assert "pull_request:" in workflow


def test_ci_workflow_tests_declared_python_range() -> None:
    workflow = _workflow_text()

    # Linux carries the full supported-Python sweep; Windows and macOS each run
    # one job so the pip-install path is exercised on every OS (JOSS readiness).
    assert "python-version: \"3.10\"" in workflow
    assert "python-version: \"3.13\"" in workflow
    assert "ubuntu-latest" in workflow
    assert "windows-latest" in workflow
    assert "macos-latest" in workflow
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow


def test_ci_workflow_runs_publication_readiness_checks() -> None:
    workflow = _workflow_text()

    assert "QT_QPA_PLATFORM: offscreen" in workflow
    assert "python -m pip install -e .[dev]" in workflow
    assert "python -m pip install build" in workflow
    assert "python -m ruff check ." in workflow
    assert "python -m pytest" in workflow
    assert "python -m build --sdist --wheel --outdir dist" in workflow
