"""Baseline machine learning models: Random Forest and XGBoost on engineered patch features."""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from src.features.spectral_indices import compute_ndvi, compute_ndwi, compute_nbr
from src.evaluation.metrics import compute_classification_metrics


def extract_patch_tabular_features(
    patch_ids: List[int],
    data_dir: str = "data/raw/landslide4sense",
    positive_threshold: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extracts summary statistics (mean, std, min, max, p25, p75) and spectral indices from each 14-band patch."""
    base_p = Path(data_dir)
    features_list = []
    labels_list = []
    feature_names = []

    # Prepare feature names
    band_labels = [f"b{i+1}" for i in range(12)] + ["dem", "slope"]
    for b in band_labels:
        for stat in ["mean", "std", "min", "max", "p25", "p75"]:
            feature_names.append(f"{b}_{stat}")
    for idx_name in ["ndvi", "ndwi", "nbr"]:
        for stat in ["mean", "std", "p10", "p90"]:
            feature_names.append(f"{idx_name}_{stat}")

    for pid in patch_ids:
        img_p = base_p / "images" / "train" / f"image_{pid}.h5"
        mask_p = base_p / "annotations" / "train" / f"mask_{pid}.h5"

        with h5py.File(img_p, "r") as f_i, h5py.File(mask_p, "r") as f_m:
            img = f_i["img"][:]
            mask = f_m["mask"][:]

        vec = []
        # Per band summary stats
        for b in range(14):
            ch = img[:, :, b]
            vec.extend([
                float(np.mean(ch)),
                float(np.std(ch)),
                float(np.min(ch)),
                float(np.max(ch)),
                float(np.percentile(ch, 25)),
                float(np.percentile(ch, 75)),
            ])

        # Spectral indices stats
        ndvi = compute_ndvi(img[:, :, 7], img[:, :, 3])
        ndwi = compute_ndwi(img[:, :, 2], img[:, :, 7])
        nbr = compute_nbr(img[:, :, 7], img[:, :, 11])

        for arr in [ndvi, ndwi, nbr]:
            vec.extend([
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.percentile(arr, 10)),
                float(np.percentile(arr, 90)),
            ])

        ratio = (mask > 0).sum() / mask.size
        label = int(ratio >= positive_threshold)

        features_list.append(vec)
        labels_list.append(label)

    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=int), feature_names


class PatchRandomForest:
    """Random Forest baseline for multi-spectral patch classification."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 10, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1,
            class_weight="balanced",
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class PatchXGBoost:
    """Gradient Boosting baseline (HistGradientBoosting / XGBoost compatible) for multi-spectral patch classification."""

    def __init__(self, n_estimators: int = 150, max_depth: int = 5, learning_rate: float = 0.05, random_state: int = 42):
        self.model = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            class_weight="balanced",
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
