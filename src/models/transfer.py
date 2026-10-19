"""Transfer learning architectures: ResNet-50 and EfficientNet-B0 adapted for N-channel multi-spectral imagery."""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights


def adapt_conv_weights(old_conv: nn.Conv2d, new_channels: int) -> nn.Conv2d:
    """Adapts a 3-channel pretrained Conv2d layer to accept N channels.

    Preserves pretrained weights in the RGB channels and initializes additional
    multi-spectral channels using the mean of the pretrained channels scaled appropriately.
    """
    out_channels = old_conv.out_channels
    kernel_size = old_conv.kernel_size
    stride = old_conv.stride
    padding = old_conv.padding
    dilation = old_conv.dilation
    bias = old_conv.bias is not None

    new_conv = nn.Conv2d(
        new_channels, out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        dilation=dilation, bias=bias
    )

    with torch.no_grad():
        old_weight = old_conv.weight.data # (out_c, 3, k, k)
        new_weight = torch.zeros((out_channels, new_channels, *kernel_size), dtype=old_weight.dtype)

        # Map RGB bands (Sentinel-2 B4, B3, B2 in L4S are channels 3, 2, 1)
        # If new_channels >= 3, copy first 3 channels
        min_c = min(3, new_channels)
        new_weight[:, :min_c, :, :] = old_weight[:, :min_c, :, :]

        if new_channels > 3:
            # Channel mean
            channel_mean = old_weight.mean(dim=1, keepdim=True) # (out_c, 1, k, k)
            scale = 3.0 / new_channels
            for c in range(3, new_channels):
                new_weight[:, c:c+1, :, :] = channel_mean * scale

        new_conv.weight.data = new_weight
        if bias and old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()

    return new_conv


class AdaptedResNet50(nn.Module):
    """ResNet-50 adapted for N-channel multi-spectral satellite imagery with two-stage training."""

    def __init__(self, in_channels: int = 14, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)

        # Adapt first conv if channels != 3
        if in_channels != 3:
            base.conv1 = adapt_conv_weights(base.conv1, in_channels)

        # Replace classification head
        num_ftrs = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
        self.model = base

    def freeze_backbone(self):
        """Freezes all layers except the classification head."""
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes all parameters for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AdaptedEfficientNetB0(nn.Module):
    """EfficientNet-B0 adapted for N-channel multi-spectral satellite imagery with two-stage training."""

    def __init__(self, in_channels: int = 14, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = efficientnet_b0(weights=weights)

        # Adapt first conv
        if in_channels != 3:
            base.features[0][0] = adapt_conv_weights(base.features[0][0], in_channels)

        # Replace classifier
        num_ftrs = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
        self.model = base

    def freeze_backbone(self):
        """Freezes all feature extractor layers except the classifier head."""
        for name, param in self.model.features.named_parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes all parameters for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
