from __future__ import annotations

import ast
import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _itasc_imports(script: Path) -> list[tuple[str, str | None]]:
    tree = ast.parse(script.read_text(), filename=str(script))
    imports: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "itasc" or alias.name.startswith("itasc."):
                    imports.append((alias.name, None))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "itasc" or node.module.startswith("itasc."):
                for alias in node.names:
                    if alias.name != "*":
                        imports.append((node.module, alias.name))
    return imports


def test_active_scripts_do_not_reference_missing_itasc_imports() -> None:
    missing: list[str] = []
    for script in sorted(SCRIPTS_DIR.glob("*.py")):
        for module_name, symbol_name in _itasc_imports(script):
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                missing.append(f"{script.relative_to(REPO_ROOT)}: {module_name} ({exc})")
                continue
            if symbol_name is not None and not hasattr(module, symbol_name):
                missing.append(
                    f"{script.relative_to(REPO_ROOT)}: {module_name}.{symbol_name}"
                )

    assert missing == [], "\n".join(missing)


def test_active_scripts_are_not_named_like_pytest_tests() -> None:
    assert sorted(path.name for path in SCRIPTS_DIR.glob("test_*.py")) == []
