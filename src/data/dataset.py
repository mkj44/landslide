"""PyTorch Dataset definitions for multi-spectral satellite patches and tabular feature vectors."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import extract_band_subset, apply_normalization, augment_patch


class LandslidePatchDataset(Dataset):
    """PyTorch Dataset for multi-spectral Landslide4Sense image patches."""

    def __init__(
        self,
        patch_ids: List[int],
        data_dir: str = "data/raw/landslide4sense",
        band_subset: str = "all14",
        norm_stats_file: Optional[str] = "artifacts/norm_stats.json",
        positive_threshold: float = 0.01,
        is_training: bool = False,
        augmentation_params: Optional[Dict[str, float]] = None,
    ):
        self.patch_ids = sorted(patch_ids)
        self.data_dir = Path(data_dir)
        self.band_subset = band_subset
        self.positive_threshold = positive_threshold
        self.is_training = is_training
        self.aug_params = augmentation_params or {}

        # Load normalization stats if provided
        self.norm_stats = None
        if norm_stats_file and Path(norm_stats_file).exists():
            with open(norm_stats_file, "r", encoding="utf-8") as f:
                self.norm_stats = json.load(f)

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch_id = self.patch_ids[idx]
        img_path = self.data_dir / "images" / "train" / f"image_{patch_id}.h5"
        mask_path = self.data_dir / "annotations" / "train" / f"mask_{patch_id}.h5"

        with h5py.File(img_path, "r") as f_i, h5py.File(mask_path, "r") as f_m:
            img = f_i["img"][:]     # (128, 128, 14)
            mask = f_m["mask"][:]   # (128, 128)

        # 1. Band subset extraction (H, W, C_sub)
        img_sub = extract_band_subset(img, self.band_subset)

        # 2. Per-band normalization using training statistics
        if self.norm_stats:
            img_sub = apply_normalization(img_sub, self.norm_stats, band_subset=self.band_subset)

        # Convert to channels-first tensor (C, H, W)
        img_tensor = torch.from_numpy(img_sub.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).long()

        # 3. Augmentation (training only)
        if self.is_training:
            img_tensor, mask_tensor = augment_patch(
                img=img_tensor,
                mask=mask_tensor,
                hflip_prob=self.aug_params.get("horizontal_flip_prob", 0.5),
                vflip_prob=self.aug_params.get("vertical_flip_prob", 0.5),
                rot90_prob=self.aug_params.get("random_rotate90_prob", 0.5),
                brightness_jitter=self.aug_params.get("brightness_contrast_jitter", 0.1),
                gaussian_noise_std=self.aug_params.get("gaussian_noise_std", 0.01),
            )

        # Compute patch-level binary classification label
        pos_ratio = (mask_tensor > 0).float().mean().item()
        label = 1 if pos_ratio >= self.positive_threshold else 0

        return img_tensor, torch.tensor(label, dtype=torch.long), mask_tensor


class TabularLandslideDataset(Dataset):
    """PyTorch Dataset for tabular features (Stage B susceptibility & risk)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
