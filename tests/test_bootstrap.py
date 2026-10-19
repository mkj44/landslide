"""Unit tests for Phase 0 bootstrap, imports, seed, logging, and configs."""

import random
from pathlib import Path
import numpy as np
import pytest
import torch

from src.utils.seed import seed_everything
from src.utils.logging import detect_hardware, get_logger, get_default_device
from src.utils.config import load_config


def test_import_src_and_submodules():
    """Verify that src and all required submodules can be imported cleanly."""
    import src
    import src.data
    import src.features
    import src.models
    import src.training
    import src.evaluation
    import src.warning
    import src.utils

    assert src is not None
    assert src.data is not None
    assert src.features is not None
    assert src.models is not None
    assert src.training is not None
    assert src.evaluation is not None
    assert src.warning is not None
    assert src.utils is not None


def test_seed_reproducibility():
    """Verify that seed_everything yields identical pseudorandom outputs across runs."""
    seed_everything(1234)
    py_val_1 = random.random()
    np_val_1 = np.random.rand(5)
    torch_val_1 = torch.rand(5)

    seed_everything(1234)
    py_val_2 = random.random()
    np_val_2 = np.random.rand(5)
    torch_val_2 = torch.rand(5)

    assert py_val_1 == py_val_2
    np.testing.assert_allclose(np_val_1, np_val_2)
    assert torch.equal(torch_val_1, torch_val_2)


def test_hardware_detection():
    """Verify hardware detection runs and identifies device type and platform."""
    specs = detect_hardware()
    assert isinstance(specs, dict)
    assert "os" in specs
    assert "device_type" in specs
    assert specs["device_type"] in {"cuda", "mps", "cpu"}

    device = get_default_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "mps", "cpu"}


def test_config_loading():
    """Verify all YAML configs load cleanly without syntax errors."""
    for cfg_name in ["data.yaml", "detection.yaml", "susceptibility.yaml", "warning.yaml"]:
        cfg_path = Path("configs") / cfg_name
        assert cfg_path.exists(), f"Missing config file: {cfg_path}"
        cfg = load_config(cfg_path)
        assert isinstance(cfg, dict)
        assert len(cfg) > 0


def test_directory_structure():
    """Verify required project directories exist."""
    required_dirs = [
        "configs",
        "data/raw",
        "data/interim",
        "data/processed",
        "artifacts/splits",
        "artifacts/checkpoints",
        "artifacts/calibration",
        "results/figures",
        "results/tables",
        "results/logs",
        "results/alerts",
    ]
    for d in required_dirs:
        p = Path(d)
        assert p.is_dir(), f"Expected directory does not exist: {d}"
