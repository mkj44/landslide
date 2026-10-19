"""Stage B Susceptibility and Short-Term Risk Models: Tabular Estimators and Interpreters."""

import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import shap

from src.utils.logging import get_logger
from src.utils.seed import seed_everything

logger = get_logger("susceptibility_models")

FEATURE_COLUMNS = [
    "elevation_m",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "prof_curvature",
    "plan_curvature",
    "tri",
    "ndvi",
    "ndwi",
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "rain_14d",
    "rain_30d",
    "rain_max_intensity",
]


def prepare_stage_b_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Transforms raw stage B table into numerical feature matrix X and label vector y.

    Applies circular sin/cos transformation to aspect_deg:
      aspect_rad = np.radians(aspect_deg)
      aspect_sin = np.sin(aspect_rad)
      aspect_cos = np.cos(aspect_rad)
    """
    df = df.copy()

    # Aspect sin/cos continuous encoding
    aspect_rad = np.radians(df["aspect_deg"].values)
    df["aspect_sin"] = np.sin(aspect_rad)
    df["aspect_cos"] = np.cos(aspect_rad)

    # Impute any residual NaNs with median values
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["label"].values.astype(int)

    return X, y, FEATURE_COLUMNS


# =============================================================================
# 1. LOGISTIC REGRESSION
# =============================================================================
class SusceptibilityLogisticRegression:
    """Standardized Logistic Regression with L2 Regularization and balanced class weights."""

    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", C=C, max_iter=1000, class_weight="balanced", random_state=random_state)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.abs(self.pipeline.named_steps["clf"].coef_[0])


# =============================================================================
# 2. RANDOM FOREST
# =============================================================================
class SusceptibilityRandomForest:
    """Random Forest Classifier with balanced subsample weights."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 10, min_samples_split: int = 5, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_


# =============================================================================
# 3. LIGHTGBM
# =============================================================================
class SusceptibilityLightGBM:
    """LightGBM gradient booster with scale_pos_weight for imbalanced susceptibility."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1,
            n_jobs=1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_ / max(np.sum(self.model.feature_importances_), 1e-6)


# =============================================================================
# 4. TABULAR PYTORCH MLP
# =============================================================================
class PyTorchTabularMLP(nn.Module):
    """Multi-Layer Perceptron for tabular susceptibility prediction."""

    def __init__(self, in_features: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = in_features

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SusceptibilityMLP:
    """Standardized PyTorch MLP wrapper compatible with scikit-learn API."""

    def __init__(
        self,
        in_features: int = 15,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: Optional[PyTorchTabularMLP] = None
        self.device = torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray):
        seed_everything(self.random_state)
        X_scaled = self.scaler.fit_transform(X)

        self.model = PyTorchTabularMLP(
            in_features=self.in_features,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        # Handle class imbalance in loss with pos_weight
        pos_count = np.sum(y == 1)
        neg_count = np.sum(y == 0)
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).float().to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-5)

        ds = TensorDataset(torch.from_numpy(X_scaled).float(), torch.from_numpy(y).float())
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.max_epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x).view(-1)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model must be fitted before predict_proba.")
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_scaled).float().to(self.device)
            logits = self.model(tensor_x).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


# =============================================================================
# 5. SHAP INTERPRETABILITY
# =============================================================================
def generate_shap_interpretability_plots(
    model: Any,
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_names: List[str],
    output_summary_path: str = "results/figures/susceptibility_shap_summary.png",
):
    """Computes TreeSHAP explanations and creates a multi-panel interpretability figure:

    - Panel A: Global Feature Importance (Mean |SHAP| Value)
    - Panel B: SHAP Beeswarm Summary (Feature Value impact on Susceptibility)
    - Panel C: Trigger Dependence Plot on 3-day Antecedent Rainfall
    """
    logger.info("Computing TreeSHAP values for model interpretability...")

    # For pipeline or tree model
    tree_estimator = getattr(model, "model", model)
    explainer = shap.TreeExplainer(tree_estimator)
    shap_values = explainer.shap_values(X_val)

    # In binary classification, check shape: if list of [neg, pos], take pos
    if isinstance(shap_values, list):
        shap_vals_pos = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_vals_pos = shap_values[:, :, 1]
    else:
        shap_vals_pos = shap_values

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2)

    # 1. Mean |SHAP| Bar Plot
    ax1 = fig.add_subplot(gs[0, 0])
    mean_abs_shap = np.mean(np.abs(shap_vals_pos), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)

    ax1.barh(np.array(feature_names)[sorted_idx], mean_abs_shap[sorted_idx], color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax1.set_title("A: Global Feature Importance (Mean |SHAP| Value)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Mean |SHAP value| (Average impact on model output)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Trigger Dependence Plot on 3-day Antecedent Rainfall
    ax2 = fig.add_subplot(gs[0, 1])
    rain_idx = feature_names.index("rain_3d") if "rain_3d" in feature_names else 0
    slope_idx = feature_names.index("slope_deg") if "slope_deg" in feature_names else 1

    rain_vals = X_val[:, rain_idx]
    rain_shap = shap_vals_pos[:, rain_idx]
    slope_vals = X_val[:, slope_idx]

    scatter = ax2.scatter(rain_vals, rain_shap, c=slope_vals, cmap="viridis", alpha=0.85, edgecolors="none", s=45)
    cbar = fig.colorbar(scatter, ax=ax2)
    cbar.set_label("Slope (degrees)", fontsize=10)

    # Add threshold inflection guide
    ax2.axhline(0, color="gray", linestyle="--", lw=1)
    ax2.set_title("B: Rainfall Trigger Dependence (rain_3d vs SHAP)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("3-Day Antecedent Rainfall (mm)", fontsize=11)
    ax2.set_ylabel("SHAP value for rain_3d", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Trigger Dependence Plot on 7-day Rainfall
    ax3 = fig.add_subplot(gs[1, :])
    rain7_idx = feature_names.index("rain_7d") if "rain_7d" in feature_names else 0
    rain7_vals = X_val[:, rain7_idx]
    rain7_shap = shap_vals_pos[:, rain7_idx]

    scatter2 = ax3.scatter(rain7_vals, rain7_shap, c=X_val[:, rain_idx], cmap="coolwarm", alpha=0.85, s=50)
    cbar2 = fig.colorbar(scatter2, ax=ax3)
    cbar2.set_label("3-Day Rainfall (mm)", fontsize=10)

    ax3.axhline(0, color="gray", linestyle="--", lw=1)
    ax3.set_title("C: Sustained Monsoon Rainfall Trigger (rain_7d vs SHAP)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("7-Day Antecedent Rainfall (mm)", fontsize=11)
    ax3.set_ylabel("SHAP value for rain_7d", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP interpretability figure to %s", output_summary_path)
