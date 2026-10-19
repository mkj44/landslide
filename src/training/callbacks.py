"""Training callbacks: Early Stopping and Model Checkpoint management."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch

from src.utils.logging import get_logger

logger = get_logger("callbacks")


class EarlyStopping:
    """Early stopping to terminate training when monitored metric stops improving."""

    def __init__(self, patience: int = 6, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return True

        improved = (
            (current_score > self.best_score + self.min_delta)
            if self.mode == "max"
            else (current_score < self.best_score - self.min_delta)
        )

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class ModelCheckpoint:
    """Saves the best model state dict based on validation score."""

    def __init__(self, filepath: str = "artifacts/checkpoints/best_model.pt", mode: str = "max"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")

    def step(self, score: float, model: torch.nn.Module, epoch: int, extra_info: Optional[Dict[str, Any]] = None) -> bool:
        is_best = (score > self.best_score) if self.mode == "max" else (score < self.best_score)
        if is_best:
            self.best_score = score
            save_payload = {
                "epoch": epoch,
                "score": score,
                "model_state_dict": model.state_dict(),
                "extra_info": extra_info or {},
            }
            torch.save(save_payload, self.filepath)
            logger.debug("Saved new best checkpoint to %s (score: %.4f)", self.filepath, score)
            return True
        return False
