import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def isolated_runs_dir(tmp_path_factory, monkeypatch):
    """Default run location for every test: never the user's real data directory."""
    runs = tmp_path_factory.mktemp("default-runs")
    monkeypatch.setenv("OPENECE_RUNS_DIR", str(runs))
    return runs


@pytest.fixture
def write_text(tmp_path):
    """Write text with exactly the given characters (no newline translation)."""
    def _write(name: str, text: str, encoding: str = "utf-8") -> Path:
        path = tmp_path / name
        path.write_bytes(text.encode(encoding))
        return path
    return _write


@pytest.fixture(scope="session")
def generator():
    """The example-data generator script, imported as a module."""
    spec = importlib.util.spec_from_file_location("generate_examples", REPO_ROOT / "scripts" / "generate_examples.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
