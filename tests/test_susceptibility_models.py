"""Unit tests for Stage B susceptibility models, feature transformations, calibration, and ECE."""

import pytest
import numpy as np
import pandas as pd

from src.models.susceptibility import (
    prepare_stage_b_features,
    SusceptibilityLogisticRegression,
    SusceptibilityRandomForest,
    SusceptibilityLightGBM,
    SusceptibilityMLP,
    FEATURE_COLUMNS,
)
from src.training.calibration import (
    compute_expected_calibration_error,
    calibrate_probabilities,
)


@pytest.fixture
def dummy_stage_b_data():
    """Generates synthetic tabular landslide data matching Stage B schema."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "elevation_m": np.random.uniform(50, 2000, n),
        "slope_deg": np.random.uniform(5, 60, n),
        "aspect_deg": np.random.uniform(0, 360, n),
        "prof_curvature": np.random.normal(0, 0.05, n),
        "plan_curvature": np.random.normal(0, 0.05, n),
        "tri": np.random.uniform(10, 80, n),
        "ndvi": np.random.uniform(0.1, 0.8, n),
        "ndwi": np.random.uniform(-0.3, 0.2, n),
        "rain_1d": np.random.uniform(0, 150, n),
        "rain_3d": np.random.uniform(10, 300, n),
        "rain_7d": np.random.uniform(20, 600, n),
        "rain_14d": np.random.uniform(30, 800, n),
        "rain_30d": np.random.uniform(50, 1200, n),
        "rain_max_intensity": np.random.uniform(5, 100, n),
        "label": np.random.binomial(1, 0.33, n),
    })
    return df


def test_prepare_stage_b_features(dummy_stage_b_data):
    """Verify feature matrix preparation and sin/cos aspect transformation."""
    X, y, feat_names = prepare_stage_b_features(dummy_stage_b_data)

    assert X.shape == (100, 15), f"Expected shape (100, 15), got {X.shape}"
    assert y.shape == (100,), f"Expected label shape (100,), got {y.shape}"
    assert "aspect_sin" in feat_names
    assert "aspect_cos" in feat_names

    # Check sin/cos range [-1, 1]
    sin_idx = feat_names.index("aspect_sin")
    cos_idx = feat_names.index("aspect_cos")
    assert np.all(X[:, sin_idx] >= -1.0) and np.all(X[:, sin_idx] <= 1.0)
    assert np.all(X[:, cos_idx] >= -1.0) and np.all(X[:, cos_idx] <= 1.0)


def test_susceptibility_models_predict(dummy_stage_b_data):
    """Verify all 4 susceptibility estimators produce valid probabilities in [0, 1]."""
    X, y, _ = prepare_stage_b_features(dummy_stage_b_data)
    X_train, y_train = X[:80], y[:80]
    X_val = X[80:]

    models = [
        SusceptibilityLogisticRegression(random_state=42),
        SusceptibilityRandomForest(n_estimators=30, random_state=42),
        SusceptibilityLightGBM(n_estimators=30, random_state=42),
        SusceptibilityMLP(in_features=15, max_epochs=5, random_state=42),
    ]

    for model in models:
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)

        assert probs.shape == (20,), f"Expected 20 probabilities, got {probs.shape}"
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities must be in [0, 1]"
        assert not np.isnan(probs).any(), "Probabilities should not contain NaN"


def test_expected_calibration_error():
    """Verify Expected Calibration Error (ECE) is within valid bounds [0, 1]."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.85, 0.15, 0.7, 0.3, 0.65, 0.2])

    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0, f"ECE must be in [0, 1], got {ece}"
    assert isinstance(ece, float)


def test_probability_calibration(dummy_stage_b_data):
    """Verify calibrate_probabilities produces calibrated predictions."""
    X, y, _ = prepare_stage_b_features(dummy_stage_b_data)
    X_train, y_train = X[:70], y[:70]
    X_val = X[70:]

    rf = SusceptibilityRandomForest(n_estimators=30, random_state=42)
    rf.fit(X_train, y_train)

    calibrator, cal_probs = calibrate_probabilities(rf.model, X_train, y_train, X_val, method="isotonic")
    assert cal_probs.shape == (30,)
    assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)
