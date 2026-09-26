"""Enforce the layer boundaries described in AGENTS.md by inspecting imports (static AST check)."""
import ast
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "src" / "openece"

_ALL_LAYERS = {
    "openece.analysis", "openece.io", "openece.instruments", "openece.measurements", "openece.recipes",
    "openece.validation", "openece.records", "openece.reporting", "openece.workflows", "openece.cli",
}
# Module prefix -> imports it must never contain.
FORBIDDEN = {
    "openece.analysis": _ALL_LAYERS - {"openece.analysis"} | {"matplotlib"},
    "openece.io": _ALL_LAYERS - {"openece.io"} | {"matplotlib"},
    "openece.instruments": _ALL_LAYERS - {"openece.instruments"} | {"matplotlib"},
    "openece.measurements": _ALL_LAYERS - {"openece.measurements", "openece.analysis"} | {"matplotlib"},
    "openece.validation": _ALL_LAYERS - {"openece.validation"} | {"matplotlib"},
    "openece.recipes": _ALL_LAYERS - {"openece.recipes", "openece.validation"} | {"matplotlib"},
    "openece.records": _ALL_LAYERS - {"openece.records"} | {"matplotlib"},
    "openece.reporting": _ALL_LAYERS - {"openece.reporting"},
    "openece.workflows": {"openece.cli"},
    "openece.cli": {"openece.analysis", "openece.instruments", "openece.measurements", "openece.reporting",
                    "matplotlib", "scipy"},
}


def module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE_DIR.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports_of(path: Path) -> set[str]:
    name = module_name(path)
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module
            found.add(target)
            found.update(f"{target}.{alias.name}" for alias in node.names)
    return found


MODULES = sorted(PACKAGE_DIR.rglob("*.py"))


@pytest.mark.parametrize("path", MODULES, ids=lambda p: module_name(p))
def test_layer_boundaries(path):
    name = module_name(path)
    rules = [forbidden for prefix, forbidden in FORBIDDEN.items() if name == prefix or name.startswith(prefix + ".")]
    violations = sorted(
        imported for imported in imports_of(path) for forbidden in rules for bad in forbidden
        if imported == bad or imported.startswith(bad + ".")
    )
    assert not violations, f"{name} must not import {violations} (see AGENTS.md)"


def test_every_module_is_covered_by_a_rule():
    uncovered = [
        module_name(p) for p in MODULES
        if module_name(p) not in ("openece", "openece.errors", "openece.units", "openece.__main__")
        and not any(module_name(p) == k or module_name(p).startswith(k + ".") for k in FORBIDDEN)
    ]
    assert not uncovered, f"add these modules to FORBIDDEN: {uncovered}"
