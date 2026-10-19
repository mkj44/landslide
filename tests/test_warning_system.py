"""Unit tests for Stage C early warning system: cost thresholds, persistence, alert engine, and CLI."""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.warning.thresholds import (
    find_cost_sensitive_threshold,
    compute_target_recall_threshold,
    map_probability_to_risk_level,
)
from src.warning.persistence import apply_persistence_and_cooldown
from src.warning.alert_engine import EarlyWarningAlertEngine, MANDATORY_ACADEMIC_DISCLAIMER
from src.warning.mapping import plot_regional_risk_map


def test_cost_sensitive_threshold():
    """Verify cost-sensitive optimization shifts threshold lower to maximize recall under 10:1 miss penalty."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.7, 0.45, 0.35, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01])

    res = find_cost_sensitive_threshold(y_true, y_prob, cost_miss_to_false_alarm_ratio=10.0)
    assert 0.0 < res["threshold"] < 0.5, "Optimal threshold under 10:1 penalty should be below 0.5"
    assert res["recall"] >= 0.75, "High miss penalty should achieve high recall"
    assert "total_cost" in res


def test_risk_level_mapping():
    """Verify probability tiers map strictly to the 4 operational categories."""
    assert map_probability_to_risk_level(0.10)["risk_level"] == "Low"
    assert map_probability_to_risk_level(0.35)["risk_level"] == "Moderate"
    assert map_probability_to_risk_level(0.60)["risk_level"] == "High"
    assert map_probability_to_risk_level(0.85)["risk_level"] == "Critical"


def test_persistence_and_cooldown():
    """Verify temporal persistence suppresses single-day spikes and cooldown smoothly decays."""
    # Test 1: Single day spike with moderate probability should be suppressed by persistence
    levels = ["Low", "High", "Low"]
    probs = [0.10, 0.65, 0.10]
    smoothed = apply_persistence_and_cooldown(levels, probs, required_consecutive_days=2, cooldown_days=2)
    assert smoothed[1]["effective_risk_level"] == "Moderate"
    assert smoothed[1]["status"] == "persistence_suppressed"

    # Test 2: Extreme instantaneous spike override (P >= 0.85) triggers immediately
    levels = ["Low", "Critical", "Low"]
    probs = [0.10, 0.92, 0.10]
    smoothed2 = apply_persistence_and_cooldown(levels, probs, required_consecutive_days=2, cooldown_days=2)
    assert smoothed2[1]["effective_risk_level"] == "Critical"
    # Following day during cooldown must not drop straight to Low
    assert smoothed2[2]["effective_risk_level"] == "High"
    assert smoothed2[2]["status"] == "cooldown_active"


def test_alert_engine_generation(tmp_path):
    """Verify alert engine generates valid JSON records and CSV log rows with academic disclaimer."""
    engine = EarlyWarningAlertEngine(
        alerts_dir=str(tmp_path / "alerts"),
        log_csv_path=str(tmp_path / "log.csv"),
    )

    features = {
        "elevation_m": 750.0,
        "slope_deg": 32.0,
        "aspect_deg": 240.0,
        "prof_curvature": 0.05,
        "plan_curvature": -0.02,
        "tri": 42.0,
        "ndvi": 0.5,
        "ndwi": -0.1,
        "rain_1d": 45.0,
        "rain_3d": 130.0,
        "rain_7d": 320.0,
        "rain_14d": 450.0,
        "rain_30d": 800.0,
        "rain_max_intensity": 65.0,
    }

    record = engine.generate_alert(
        latitude=12.45,
        longitude=75.60,
        feature_dict=features,
        location_name="Coorg Slope 4",
        save_records=True,
    )

    assert record["alert_id"].startswith("ALERT-")
    assert record["risk_assessment"]["level"] in ["Low", "Moderate", "High", "Critical"]
    assert MANDATORY_ACADEMIC_DISCLAIMER in record["disclaimer"]

    # Verify JSON file written
    json_files = list((tmp_path / "alerts").glob("*.json"))
    assert len(json_files) == 1

    # Verify CSV log updated
    df_log = pd.read_csv(tmp_path / "log.csv")
    assert len(df_log) == 1
    assert df_log.iloc[0]["location_name"] == "Coorg Slope 4"


def test_plot_regional_risk_map(tmp_path):
    """Verify static geographic risk map generates a valid non-empty PNG file."""
    df_dummy = pd.DataFrame({
        "longitude": [75.2, 75.8, 76.1, 74.9],
        "latitude": [11.5, 12.1, 13.0, 14.2],
        "risk_level": ["Low", "Moderate", "High", "Critical"],
        "predicted_probability": [0.15, 0.42, 0.68, 0.91],
    })
    out_map = tmp_path / "test_map.png"
    plot_regional_risk_map(df_dummy, output_path=str(out_map))

    assert out_map.exists()
    assert out_map.stat().st_size > 10000, "Map PNG should be non-empty"
