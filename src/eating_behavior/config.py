"""Configuration loading and path resolution helpers.

This module only handles generic YAML loading. Business meanings such as model
paths, source snapshots, and identity routing are interpreted by callers.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INTEGER_PATTERN = re.compile(r"[+-]?\d+\Z")


def config_bool(value: Any, *, field_name: str) -> bool:
    """Parse a configuration boolean without Python's truthy-string trap."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def config_float(value: Any, *, field_name: str) -> float:
    """Parse a finite numeric setting while rejecting YAML booleans."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a number") from None
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def config_int(value: Any, *, field_name: str) -> int:
    """Parse an integer setting without coercing bools or fractional values."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, not a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and _INTEGER_PATTERN.fullmatch(value.strip()):
        return int(value.strip())
    raise ValueError(f"{field_name} must be an integer")


def config_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    """Require a mapping-valued configuration section."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


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

