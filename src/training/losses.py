"""Loss functions for Stage A detection models: Focal Loss and Weighted Cross Entropy."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary Focal Loss for combating severe class imbalance in detection.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:

        logits: Tensor of shape (B,) or (B, 1) raw unnormalized logits.
        targets: Tensor of shape (B,) binary targets (0 or 1).
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)

        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma

        loss = alpha_factor * modulating_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_function(strategy: str = "focal_loss", pos_weight: float = 1.0) -> nn.Module:
    """Factory for loss functions."""
    if strategy == "focal_loss":
        return FocalLoss(alpha=0.35, gamma=2.0)
    elif strategy == "class_weights":
        weight_tensor = torch.tensor([pos_weight])
        return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    else:
        return nn.BCEWithLogitsLoss()
