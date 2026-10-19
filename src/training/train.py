"""Training engine for Stage A multi-spectral patch classifiers."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logging import get_logger
from src.utils.seed import seed_everything
from src.training.losses import get_loss_function
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.evaluation.metrics import compute_classification_metrics

logger = get_logger("training")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> float:
    """Runs a single training epoch and returns mean training loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device).float()

        optimizer.zero_grad()
        logits = model(images).view(-1)
        loss = criterion(logits, targets)

        loss.backward()
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluates model over validation dataloader.

    Returns:
        (val_loss, metrics_dict, y_prob, y_true)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_probs = []
    all_targets = []

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device).float()

        logits = model(images).view(-1)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_probs.extend(probs.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    val_loss = running_loss / max(total_samples, 1)
    y_prob = np.array(all_probs, dtype=float)
    y_true = np.array(all_targets, dtype=int)

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    return val_loss, metrics, y_prob, y_true


def train_patch_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 25,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    loss_strategy: str = "focal_loss",
    pos_weight: float = 1.5,
    early_stopping_patience: int = 6,
    gradient_clip: float = 1.0,
    freeze_epochs: int = 0,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float], np.ndarray, np.ndarray]:
    """Complete training loop for a deep learning patch classifier.

    Features:
      - Two-stage backbone freeze/unfreeze (if model supports it)
      - Cosine annealing learning rate scheduling
      - Focal loss or weighted cross-entropy
      - Early stopping on validation F1 score
      - Model checkpointing of the best validation epoch
      - History tracking for plotting learning curves

    Returns:
        history, best_val_metrics, best_val_probs, val_targets
    """
    if device is None:
        device = torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu")

    model = model.to(device)
    criterion = get_loss_function(strategy=loss_strategy, pos_weight=pos_weight)

    # Initial freeze handling
    if freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        logger.info("Freezing backbone for first %d epochs...", freeze_epochs)
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")
    checkpoint = ModelCheckpoint(filepath=checkpoint_path or "artifacts/checkpoints/best_model.pt", mode="max")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_accuracy": [],
        "val_roc_auc": [],
        "val_pr_auc": [],
        "lr": [],
    }

    best_val_metrics: Dict[str, float] = {}
    best_val_probs = np.array([])
    final_targets = np.array([])

    for epoch in range(1, num_epochs + 1):
        # Check if freeze stage ends
        if freeze_epochs > 0 and epoch == freeze_epochs + 1 and hasattr(model, "unfreeze_backbone"):
            logger.info("Unfreezing backbone for fine-tuning at epoch %d...", epoch)
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate * 0.5,
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(num_epochs - freeze_epochs, 1), eta_min=1e-6
            )

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip=gradient_clip,
        )

        val_loss, val_metrics, val_probs, val_targets = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Update history
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_f1"].append(val_metrics["f1"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["val_pr_auc"].append(val_metrics["pr_auc"])
        history["lr"].append(current_lr)

        # Monitored score for early stopping & checkpointing is F1 score
        score = val_metrics["f1"]
        is_best = checkpoint.step(score=score, model=model, epoch=epoch, extra_info=val_metrics)
        if is_best or epoch == 1:
            best_val_metrics = val_metrics
            best_val_probs = val_probs
            final_targets = val_targets

        improved = early_stopping(score)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered at epoch %d (best F1: %.4f)", epoch, early_stopping.best_score)
            break

    # Reload best model weights if saved
    if Path(checkpoint.filepath).exists():
        loaded = torch.load(checkpoint.filepath, map_location=device)
        model.load_state_dict(loaded["model_state_dict"])
        # Final pass with best weights
        _, best_val_metrics, best_val_probs, final_targets = evaluate_epoch(
            model, val_loader, criterion, device
        )

    return history, best_val_metrics, best_val_probs, final_targets
