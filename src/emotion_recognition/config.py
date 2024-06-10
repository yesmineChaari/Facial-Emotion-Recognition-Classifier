"""Configuration loading for reproducible project workflows."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return its mapping."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must contain a mapping: {config_path}")
    return values