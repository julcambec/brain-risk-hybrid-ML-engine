"""YAML configuration loading and merging utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return its contents as a dict.

    Parameters
    ----------
    path : str | Path
        Path to a ``.yaml`` or ``.yml`` file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is empty or contains invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Config file is empty: {path}")

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(data).__name__}")
    return data


def merge_configs(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two configuration dicts, with *overrides* winning on conflicts.

    Nested dicts are merged recursively; all other types are replaced wholesale.

    Parameters
    ----------
    base : dict
        Default / base configuration.
    overrides : dict
        User-specified overrides.

    Returns
    -------
    dict[str, Any]
        Merged configuration (new dict — inputs are not mutated).
    """
    merged: dict[str, Any] = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
