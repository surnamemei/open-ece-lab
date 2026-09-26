from __future__ import annotations
from pathlib import Path
import yaml

REQUIRED = {"name", "stimulus", "requirements"}

def load_recipe(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    missing = REQUIRED - set(data or {})
    if missing:
        raise ValueError(f"recipe missing fields: {sorted(missing)}")
    return data
