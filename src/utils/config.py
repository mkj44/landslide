"""Configuration loading and validation utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration file into a Python dictionary.

    Args:
        config_path: File path to the YAML file.

    Returns:
        Dictionary with loaded configuration values.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        yaml.YAMLError: If the YAML syntax is invalid.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}
    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> Path:
    """Saves a Python dictionary as a YAML configuration file.

    Args:
        config: Configuration dictionary.
        save_path: Destination path.

    Returns:
        Path of the written configuration file.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    return path
