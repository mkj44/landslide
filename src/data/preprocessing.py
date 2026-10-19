"""Patch validation, configurable band subsets, training-only normalization, and augmentation."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch

from src.utils.logging import get_logger

logger = get_logger("preprocessing")

# Standard Landslide4Sense 14-band indices
BAND_INDICES = {
    "rgb": [3, 2, 1],                # B04 (Red), B03 (Green), B02 (Blue)
    "rgb_nir": [3, 2, 1, 7],         # B04, B03, B02, B08 (NIR)
    "all14": list(range(14)),        # All 14 multi-spectral + DEM + Slope bands
}


def validate_patch(
    img: np.ndarray,
    expected_shape: Tuple[int, ...] = (128, 128, 14),
    expected_dtype: Optional[np.dtype] = None,
) -> Tuple[bool, str]:
    """Validates tensor shape, data type, and detects NaN or Inf values.

    Returns:
        (is_valid, reason_if_invalid)
    """
    if img.shape != expected_shape:
        return False, f"Shape mismatch: expected {expected_shape}, got {img.shape}"
    if expected_dtype is not None and img.dtype != expected_dtype:
        return False, f"Dtype mismatch: expected {expected_dtype}, got {img.dtype}"
    if np.isnan(img).any():
        return False, "Contains NaN values"
    if np.isinf(img).any():
        return False, "Contains Inf values"
    return True, "Valid"


def extract_band_subset(img: np.ndarray, subset_name: str = "all14") -> np.ndarray:
    """Extracts a specific band subset from a 14-band image patch.

    Args:
        img: Array of shape (H, W, 14) or (C, H, W).
        subset_name: One of 'rgb', 'rgb_nir', or 'all14'.

    Returns:
        Array with selected channels.
    """
    if subset_name not in BAND_INDICES:
        raise ValueError(f"Unknown band subset: {subset_name}. Supported: {list(BAND_INDICES.keys())}")

    indices = BAND_INDICES[subset_name]
    if img.ndim == 3 and img.shape[-1] == 14:
        return img[:, :, indices]
    elif img.ndim == 3 and img.shape[0] == 14:
        return img[indices, :, :]
    else:
        raise ValueError(f"Unsupported image shape for band extraction: {img.shape}")


def compute_normalization_stats(
    train_patch_paths: List[Union[str, Path]],
    save_json_path: str = "artifacts/norm_stats.json",
    percentile_clip: bool = True,
) -> Dict[str, Any]:
    """Computes per-band mean, standard deviation, and 2nd/98th percentiles strictly from training patches.

    Args:
        train_patch_paths: List of paths to training image files (.h5).
        save_json_path: Destination path for normalization parameters.
        percentile_clip: Whether to compute 2nd and 98th percentiles for robust scaling.

    Returns:
        Dictionary containing per-band statistics.
    """
    import h5py

    logger.info("Computing per-band normalization stats on %d training patches...", len(train_patch_paths))
    n_bands = 14
    band_sums = np.zeros(n_bands, dtype=np.float64)
    band_sq_sums = np.zeros(n_bands, dtype=np.float64)
    total_pixels = 0

    all_channel_samples: List[List[float]] = [[] for _ in range(n_bands)]

    for p in train_patch_paths:
        with h5py.File(p, "r") as f:
            img = f["img"][:]
            is_valid, _ = validate_patch(img)
            if not is_valid:
                continue

            # Accumulate running sums
            for b in range(n_bands):
                c_vals = img[:, :, b]
                band_sums[b] += np.sum(c_vals)
                band_sq_sums[b] += np.sum(c_vals**2)
                # Sample 200 random pixels per patch for percentile estimation
                sample_idx = np.random.choice(c_vals.size, size=min(200, c_vals.size), replace=False)
                all_channel_samples[b].extend(c_vals.flatten()[sample_idx].tolist())

            total_pixels += img.shape[0] * img.shape[1]

    means = band_sums / total_pixels
    variances = (band_sq_sums / total_pixels) - (means**2)
    stds = np.sqrt(np.maximum(variances, 1e-8))

    p2 = [float(np.percentile(all_channel_samples[b], 2)) for b in range(n_bands)]
    p98 = [float(np.percentile(all_channel_samples[b], 98)) for b in range(n_bands)]

    stats = {
        "num_training_patches": len(train_patch_paths),
        "total_pixels_evaluated": int(total_pixels),
        "means": [float(m) for m in means],
        "stds": [float(s) for s in stds],
        "p2": p2,
        "p98": p98,
        "percentile_clipping_enabled": percentile_clip,
    }

    out_file = Path(save_json_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Saved training split normalization stats to %s", out_file)
    return stats


def apply_normalization(
    img: np.ndarray,
    norm_stats: Dict[str, Any],
    band_subset: str = "all14",
    clip_percentiles: bool = True,
) -> np.ndarray:
    """Applies precomputed training-set normalization to an image tensor.

    Args:
        img: Array of shape (H, W, C) or (C, H, W).
        norm_stats: Dictionary containing 'means', 'stds', 'p2', 'p98'.
        band_subset: Subset of bands present in img.
        clip_percentiles: Whether to clip values to [p2, p98] before z-score scaling.

    Returns:
        Normalized array with zero mean and unit variance.
    """
    indices = BAND_INDICES[band_subset]
    means = np.array([norm_stats["means"][i] for i in indices], dtype=np.float32)
    stds = np.array([norm_stats["stds"][i] for i in indices], dtype=np.float32)

    img_norm = img.astype(np.float32).copy()

    if clip_percentiles and "p2" in norm_stats and "p98" in norm_stats:
        p2 = np.array([norm_stats["p2"][i] for i in indices], dtype=np.float32)
        p98 = np.array([norm_stats["p98"][i] for i in indices], dtype=np.float32)

        if img_norm.ndim == 3 and img_norm.shape[-1] == len(indices):
            # (H, W, C)
            img_norm = np.clip(img_norm, p2, p98)
            img_norm = (img_norm - means) / (stds + 1e-6)
        elif img_norm.ndim == 3 and img_norm.shape[0] == len(indices):
            # (C, H, W)
            for b in range(len(indices)):
                img_norm[b] = np.clip(img_norm[b], p2[b], p98[b])
                img_norm[b] = (img_norm[b] - means[b]) / (stds[b] + 1e-6)
    else:
        if img_norm.ndim == 3 and img_norm.shape[-1] == len(indices):
            img_norm = (img_norm - means) / (stds + 1e-6)
        elif img_norm.ndim == 3 and img_norm.shape[0] == len(indices):
            for b in range(len(indices)):
                img_norm[b] = (img_norm[b] - means[b]) / (stds[b] + 1e-6)

    return img_norm


def augment_patch(
    img: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.5,
    rot90_prob: float = 0.5,
    brightness_jitter: float = 0.1,
    gaussian_noise_std: float = 0.01,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Applies train-only data augmentation synchronously to image tensor (C, H, W) and mask (H, W)."""
    # Random Horizontal Flip
    if torch.rand(1).item() < hflip_prob:
        img = torch.flip(img, dims=[-1])
        if mask is not None:
            mask = torch.flip(mask, dims=[-1])

    # Random Vertical Flip
    if torch.rand(1).item() < vflip_prob:
        img = torch.flip(img, dims=[-2])
        if mask is not None:
            mask = torch.flip(mask, dims=[-2])

    # Random 90-degree Rotation
    if torch.rand(1).item() < rot90_prob:
        k = int(torch.randint(1, 4, (1,)).item())
        img = torch.rot90(img, k=k, dims=[-2, -1])
        if mask is not None:
            mask = torch.rot90(mask, k=k, dims=[-2, -1])

    # Light Brightness / Contrast Jitter
    if brightness_jitter > 0:
        factor = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * brightness_jitter
        img = img * factor

    # Gaussian Noise (applied to optical bands only, DEM and slope left clean)
    if gaussian_noise_std > 0:
        noise = torch.randn_like(img) * gaussian_noise_std
        img = img + noise

    return img, mask
