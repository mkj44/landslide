"""Custom Convolutional Neural Network from scratch for multi-channel satellite patch classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard Convolution block with Conv2d, BatchNorm2d, and LeakyReLU."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ScratchLandslideCNN(nn.Module):
    """Configurable multi-scale CNN architecture for landslide patch classification.

    Features:
      - Supports arbitrary input channels (3, 4, or 14).
      - Multi-layer ConvBlocks with BatchNorm and spatial downsampling.
      - Global Average Pooling (GAP) eliminating spatial dimensional bottlenecks.
      - Regularized Dropout head producing binary classification logits.
    """

    def __init__(
        self,
        in_channels: int = 14,
        base_channels: int = 32,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Stage 1: 128x128 -> 64x64
        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, base_channels, stride=1),
            ConvBlock(base_channels, base_channels, stride=2),
        )

        # Stage 2: 64x64 -> 32x32
        self.stage2 = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2, stride=1),
            ConvBlock(base_channels * 2, base_channels * 2, stride=2),
        )

        # Stage 3: 32x32 -> 16x16
        self.stage3 = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels * 4, stride=1),
            ConvBlock(base_channels * 4, base_channels * 4, stride=2),
        )

        # Stage 4: 16x16 -> 8x8
        self.stage4 = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8, stride=1),
            ConvBlock(base_channels * 8, base_channels * 8, stride=2),
        )

        # Global Average Pooling & Regularized Linear Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: Tensor of shape (B, in_channels, 128, 128).

        Returns:
            Logits of shape (B, 1).
        """
        feat = self.stage1(x)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)

        pooled = self.gap(feat).flatten(1)
        logits = self.head(pooled)
        return logits
