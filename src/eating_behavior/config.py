"""Configuration loading and path resolution helpers.

This module only handles generic YAML loading. Business meanings such as model
paths, source snapshots, and identity routing are interpreted by callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path | None, base: Path | None = None) -> Path | None:
    """Resolve a config path against the project root unless it is absolute."""
    if value is None or value == "":
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or PROJECT_ROOT) / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and require a mapping at the root."""
    config_path = resolve_path(path)
    if config_path is None:
        return {}
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return data

