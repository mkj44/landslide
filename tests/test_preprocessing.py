"""Unit tests for Phase 2 preprocessing, feature engineering, and leakage-safe splitting."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preprocessing import (
    validate_patch,
    extract_band_subset,
    apply_normalization,
    augment_patch,
)
from src.features.spectral_indices import compute_ndvi, compute_ndwi, compute_nbr
from src.data.dataset import LandslidePatchDataset


def test_splits_no_overlap():
    """Verify zero overlap between train, val, and test splits across Stage A and Stage B."""
    # Stage A
    with open("artifacts/splits/stage_a_splits.json", "r") as f:
        s_a = json.load(f)
    tr_a = set(s_a["train"])
    va_a = set(s_a["val"])
    te_a = set(s_a["test"])

    assert len(tr_a.intersection(va_a)) == 0, "Stage A Train and Val overlap!"
    assert len(tr_a.intersection(te_a)) == 0, "Stage A Train and Test overlap!"
    assert len(va_a.intersection(te_a)) == 0, "Stage A Val and Test overlap!"

    # Stage B Temporal
    with open("artifacts/splits/stage_b_temporal_split.json", "r") as f:
        s_b = json.load(f)
    tr_b_idx = set(s_b["train_indices"])
    te_b_idx = set(s_b["test_indices"])

    assert len(tr_b_idx.intersection(te_b_idx)) == 0, "Stage B Train and Test indices overlap!"
    assert max(s_b["train_years"]) <= s_b["cutoff_year"], "Train split contains post-cutoff years!"
    assert min(s_b["test_years"]) > s_b["cutoff_year"], "Test split contains pre-cutoff years!"


def test_spatial_group_separation():
    """Verify that in every Spatial Block CV fold, no grid cell appears in both train and val."""
    with open("artifacts/splits/stage_b_spatial_cv_splits.json", "r") as f:
        spatial_folds = json.load(f)

    df_b = pd.read_csv("data/processed/stage_b_features.csv")

    for fold_name, fold_data in spatial_folds.items():
        tr_idx = fold_data["train_indices"]
        va_idx = fold_data["val_indices"]

        # Ensure indices themselves don't overlap
        assert len(set(tr_idx).intersection(set(va_idx))) == 0

        # Ensure spatial grid cells don't overlap
        tr_cells = set(df_b.iloc[tr_idx]["grid_cell_id"].unique())
        va_cells = set(df_b.iloc[va_idx]["grid_cell_id"].unique())
        overlap = tr_cells.intersection(va_cells)
        assert len(overlap) == 0, f"Spatial leakage detected in {fold_name}! Leaked cells: {overlap}"


def test_normalization_stats_train_only():
    """Verify that normalization stats were derived strictly from the training split."""
    with open("artifacts/norm_stats.json", "r") as f:
        stats = json.load(f)
    with open("artifacts/splits/stage_a_splits.json", "r") as f:
        s_a = json.load(f)

    assert stats["num_training_patches"] == len(s_a["train"]), "Stats must match training patch count exactly."
    assert len(stats["means"]) == 14
    assert len(stats["stds"]) == 14
    for std in stats["stds"]:
        assert std > 0, "Standard deviation must be strictly positive."


def test_label_derivation_handmade_mask():
    """Verify patch classification label logic on synthetic ground truth masks with exact pixel counts."""
    # 128x128 mask has 16,384 total pixels
    total_px = 128 * 128

    thresholds = [0.001, 0.01, 0.05]

    def derive_label(mask: np.ndarray, tau: float) -> int:
        ratio = (mask > 0).sum() / total_px
        return int(ratio >= tau)

    # Case 1: Pure background (0 landslide pixels) -> 0 for all thresholds
    mask_0 = np.zeros((128, 128), dtype=np.uint8)
    for tau in thresholds:
        assert derive_label(mask_0, tau) == 0

    # Case 2: 100 pixels (100 / 16384 = 0.00610) -> 1 for 0.001, 0 for 0.01 & 0.05
    mask_100 = np.zeros((128, 128), dtype=np.uint8)
    mask_100.ravel()[:100] = 1
    assert derive_label(mask_100, 0.001) == 1
    assert derive_label(mask_100, 0.01) == 0
    assert derive_label(mask_100, 0.05) == 0

    # Case 3: 300 pixels (300 / 16384 = 0.01831) -> 1 for 0.001 and 0.01, 0 for 0.05
    mask_300 = np.zeros((128, 128), dtype=np.uint8)
    mask_300.ravel()[:300] = 1
    assert derive_label(mask_300, 0.001) == 1
    assert derive_label(mask_300, 0.01) == 1
    assert derive_label(mask_300, 0.05) == 0

    # Case 4: 1000 pixels (1000 / 16384 = 0.06103) -> 1 for all thresholds
    mask_1000 = np.zeros((128, 128), dtype=np.uint8)
    mask_1000.ravel()[:1000] = 1
    for tau in thresholds:
        assert derive_label(mask_1000, tau) == 1


def test_spectral_indices_bounds():
    """Verify NDVI, NDWI, and NBR calculations remain within [-1, 1]."""
    nir = np.array([0.8, 0.5, 0.1])
    red = np.array([0.2, 0.5, 0.9])
    green = np.array([0.3, 0.4, 0.2])
    swir2 = np.array([0.1, 0.4, 0.6])

    ndvi = compute_ndvi(nir, red)
    ndwi = compute_ndwi(green, nir)
    nbr = compute_nbr(nir, swir2)

    assert (ndvi >= -1.0).all() and (ndvi <= 1.0).all()
    assert (ndwi >= -1.0).all() and (ndwi <= 1.0).all()
    assert (nbr >= -1.0).all() and (nbr <= 1.0).all()


def test_patch_augmentation():
    """Verify that image augmentation preserves tensor shape and produces valid values."""
    img = torch.randn(14, 128, 128)
    mask = torch.randint(0, 2, (128, 128))

    aug_img, aug_mask = augment_patch(img, mask)
    assert aug_img.shape == (14, 128, 128)
    assert aug_mask.shape == (128, 128)
    assert not torch.isnan(aug_img).any()
    assert not torch.isinf(aug_img).any()
