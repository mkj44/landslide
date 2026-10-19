"""Unit tests for Stage A detection models, metrics, losses, and adaptation."""

import pytest
import numpy as np
import torch
from PIL import Image

from src.models.cnn import ScratchLandslideCNN
from src.models.transfer import AdaptedResNet50, AdaptedEfficientNetB0, adapt_conv_weights
from src.training.losses import FocalLoss, get_loss_function
from src.evaluation.metrics import compute_classification_metrics
from src.models.clip_model import LandslideCLIP, CLIP_PROMPTS


def test_scratch_cnn_forward():
    """Verify ScratchLandslideCNN forward pass shapes with 14 and 3 channels."""
    # 14 channels
    model_14c = ScratchLandslideCNN(in_channels=14, base_channels=16)
    x_14c = torch.randn(2, 14, 128, 128)
    out_14c = model_14c(x_14c)
    assert out_14c.shape == (2, 1), f"Expected shape (2, 1), got {out_14c.shape}"

    # 3 channels (RGB)
    model_3c = ScratchLandslideCNN(in_channels=3, base_channels=16)
    x_3c = torch.randn(2, 3, 128, 128)
    out_3c = model_3c(x_3c)
    assert out_3c.shape == (2, 1), f"Expected shape (2, 1), got {out_3c.shape}"


def test_resnet50_adaptation_and_freeze():
    """Verify AdaptedResNet50 accepts 14 channels and freezes/unfreezes properly."""
    model = AdaptedResNet50(in_channels=14, pretrained=False)
    x = torch.randn(2, 14, 128, 128)
    out = model(x)
    assert out.shape == (2, 1), f"Expected shape (2, 1), got {out.shape}"

    # Test freeze
    model.freeze_backbone()
    assert not model.model.conv1.weight.requires_grad
    assert model.model.fc[1].weight.requires_grad

    # Test unfreeze
    model.unfreeze_backbone()
    assert model.model.conv1.weight.requires_grad
    assert model.model.fc[1].weight.requires_grad


def test_efficientnet_adaptation():
    """Verify AdaptedEfficientNetB0 accepts 14 channels and produces binary logits."""
    model = AdaptedEfficientNetB0(in_channels=14, pretrained=False)
    x = torch.randn(2, 14, 128, 128)
    out = model(x)
    assert out.shape == (2, 1), f"Expected shape (2, 1), got {out.shape}"

    # Test freeze
    model.freeze_backbone()
    assert not model.model.features[0][0].weight.requires_grad
    assert model.model.classifier[1].weight.requires_grad


def test_focal_loss():
    """Verify FocalLoss computes numerically valid non-negative losses."""
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.tensor([2.5, -1.8, 0.4, -0.2])
    targets = torch.tensor([1, 0, 1, 0])

    loss = criterion(logits, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_classification_metrics_completeness():
    """Verify compute_classification_metrics returns all required 10 metrics."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.6, 0.2, 0.4, 0.7, 0.3])

    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)

    expected_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]
    for k in expected_keys:
        assert k in metrics, f"Missing metric key: {k}"
        assert isinstance(metrics[k], float), f"Metric {k} must be float, got {type(metrics[k])}"

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_clip_patch_to_rgb():
    """Verify LandslideCLIP.patch_to_rgb creates a valid PIL RGB image from 14-band array."""
    dummy_14b = np.random.uniform(0, 5000, size=(128, 128, 14)).astype(np.float32)
    pil_img = LandslideCLIP.patch_to_rgb(dummy_14b)

    assert isinstance(pil_img, Image.Image)
    assert pil_img.size == (128, 128)
    assert pil_img.mode == "RGB"
