"""Logging and hardware detection utilities for the Landslide ML pipeline."""

import logging
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch


def get_logger(name: str = "landslide", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Creates or retrieves a structured logger with formatting for console and file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        # Default project log file
        default_log_dir = Path("results/logs")
        default_log_dir.mkdir(parents=True, exist_ok=True)
        default_log_file = default_log_dir / "pipeline.log"
        file_handler = logging.FileHandler(str(default_log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def detect_hardware() -> Dict[str, Any]:
    """Detects available hardware (CUDA, Apple Silicon MPS, or CPU) and logs system specs."""
    logger = get_logger("hardware")
    specs: Dict[str, Any] = {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    if torch.cuda.is_available():
        specs["device_type"] = "cuda"
        specs["device_name"] = torch.cuda.get_device_name(0)
        specs["device_count"] = torch.cuda.device_count()
        specs["cuda_version"] = torch.version.cuda
        logger.info("Hardware Detected: NVIDIA GPU (%s, count=%d, CUDA=%s)",
                    specs["device_name"], specs["device_count"], specs["cuda_version"])
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        specs["device_type"] = "mps"
        specs["device_name"] = f"Apple Silicon Metal Performance Shaders ({platform.machine()})"
        specs["device_count"] = 1
        logger.info("Hardware Detected: Apple Silicon GPU via Metal Performance Shaders (MPS: %s)", specs["device_name"])
    else:
        specs["device_type"] = "cpu"
        specs["device_name"] = platform.processor() or "CPU"
        specs["device_count"] = os.cpu_count() or 1
        logger.info("Hardware Detected: CPU (%s, threads=%d)", specs["device_name"], specs["device_count"])

    return specs


def get_default_device() -> torch.device:
    """Returns the most capable torch.device available (cuda -> mps -> cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
