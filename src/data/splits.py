"""Leakage-safe data splitting: spatial block cross-validation and temporal holdout."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold

from src.utils.logging import get_logger

logger = get_logger("splits")


def create_stage_a_splits(
    manifest_csv: str = "data/processed/landslide4sense_manifest.csv",
    save_path: str = "artifacts/splits/stage_a_splits.json",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
    stratify_col: str = "label_th_0.01",
) -> Dict[str, List[int]]:
    """Creates stratified train, validation, and test splits for Stage A patch classification.

    Ensures zero overlap between sets and preserves balanced class representation.
    """
    df = pd.read_csv(manifest_csv)
    rng = np.random.default_rng(random_seed)

    pos_ids = df[df[stratify_col] == 1]["patch_id"].values
    neg_ids = df[df[stratify_col] == 0]["patch_id"].values

    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    def split_ids(ids: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
        n = len(ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        tr = ids[:n_train].tolist()
        va = ids[n_train:n_train + n_val].tolist()
        te = ids[n_train + n_val:].tolist()
        return tr, va, te

    tr_pos, va_pos, te_pos = split_ids(pos_ids)
    tr_neg, va_neg, te_neg = split_ids(neg_ids)

    train_ids = sorted(tr_pos + tr_neg)
    val_ids = sorted(va_pos + va_neg)
    test_ids = sorted(te_pos + te_neg)

    # Verification: assert disjoint sets
    s_train, s_val, s_test = set(train_ids), set(val_ids), set(test_ids)
    assert len(s_train.intersection(s_val)) == 0, "Train and Val overlap in Stage A!"
    assert len(s_train.intersection(s_test)) == 0, "Train and Test overlap in Stage A!"
    assert len(s_val.intersection(s_test)) == 0, "Val and Test overlap in Stage A!"

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "counts": {
            "train_total": len(train_ids),
            "val_total": len(val_ids),
            "test_total": len(test_ids),
        }
    }

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    logger.info("Created Stage A splits: Train=%d, Val=%d, Test=%d (Saved to %s)",
                len(train_ids), len(val_ids), len(test_ids), out_p)
    return splits


def create_stage_b_spatial_cv_splits(
    features_csv: str = "data/processed/stage_b_features.csv",
    save_path: str = "artifacts/splits/stage_b_spatial_cv_splits.json",
    n_splits: int = 5,
    group_col: str = "grid_cell_id",
) -> Dict[str, Any]:
    """Creates Spatial Block Cross-Validation splits grouped by spatial grid cell.

    Prevents spatial autocorrelation leakage: all events within any grid cell are strictly
    assigned together to either train or validation in every fold.
    """
    df = pd.read_csv(features_csv)
    groups = df[group_col].values
    X = np.arange(len(df))
    y = df["label"].values

    gkf = GroupKFold(n_splits=n_splits)
    folds_dict = {}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        train_groups = set(df.iloc[train_idx][group_col].unique())
        val_groups = set(df.iloc[val_idx][group_col].unique())

        # Strict assertion: no shared grid cell across folds
        assert len(train_groups.intersection(val_groups)) == 0, f"Spatial leakage detected in Fold {fold_idx}!"

        folds_dict[f"fold_{fold_idx}"] = {
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "train_groups_count": len(train_groups),
            "val_groups_count": len(val_groups),
        }

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(folds_dict, f, indent=2)

    logger.info("Created %d Spatial Block CV folds grouped by %s (Saved to %s)",
                n_splits, group_col, out_p)
    return folds_dict


def create_stage_b_temporal_holdout_split(
    features_csv: str = "data/processed/stage_b_features.csv",
    save_path: str = "artifacts/splits/stage_b_temporal_split.json",
    cutoff_year: int = 2014,
) -> Dict[str, Any]:
    """Creates a temporal holdout split: trains on earlier years (<= cutoff_year) and tests on later years.

    Evaluates realistic operational forecasting performance on unseen future seasons.
    """
    df = pd.read_csv(features_csv)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["year"] = df["event_date"].dt.year

    train_mask = (df["year"] <= cutoff_year)
    test_mask = (df["year"] > cutoff_year)

    train_indices = np.where(train_mask)[0].tolist()
    test_indices = np.where(test_mask)[0].tolist()

    # Verify temporal separation
    train_years = set(df.iloc[train_indices]["year"].unique())
    test_years = set(df.iloc[test_indices]["year"].unique())
    assert max(train_years) <= cutoff_year, "Train set contains post-cutoff years!"
    assert min(test_years) > cutoff_year, "Test set contains pre-cutoff years!"

    split_dict = {
        "cutoff_year": int(cutoff_year),
        "train_years": [int(y) for y in sorted(list(train_years))],
        "test_years": [int(y) for y in sorted(list(test_years))],
        "train_indices": [int(i) for i in train_indices],
        "test_indices": [int(i) for i in test_indices],
        "train_count": len(train_indices),
        "test_count": len(test_indices),
        "train_positive_rate": float(df.iloc[train_indices]["label"].mean()),
        "test_positive_rate": float(df.iloc[test_indices]["label"].mean()),
    }

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2)

    logger.info("Created Temporal Holdout Split (Cutoff %d): Train=%d (%s), Test=%d (%s)",
                cutoff_year, len(train_indices), train_years, len(test_indices), test_years)
    return split_dict
