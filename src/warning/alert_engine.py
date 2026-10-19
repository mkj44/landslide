"""Core Early Warning Alert Engine: probability inference, structured logging, and record generation."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import joblib
import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.warning.thresholds import map_probability_to_risk_level, DEFAULT_TIERS
from src.models.susceptibility import prepare_stage_b_features, FEATURE_COLUMNS

logger = get_logger("alert_engine")

MANDATORY_ACADEMIC_DISCLAIMER = (
    "DISCLAIMER: This system is a research prototype developed for academic and scientific evaluation. "
    "It is NOT an operational civil safety warning system and should NOT be used for evacuation or "
    "life-safety decisions."
)


class EarlyWarningAlertEngine:
    """Production early warning alert generator utilizing calibrated susceptibility models."""

    def __init__(
        self,
        model_path: str = "artifacts/checkpoints/best_susceptibility_model.joblib",
        alerts_dir: str = "results/alerts",
        log_csv_path: str = "results/alerts/warning_log.csv",
    ):
        self.model_path = Path(model_path)
        self.alerts_dir = Path(alerts_dir)
        self.log_csv_path = Path(log_csv_path)

        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}. Run Phase 4 training first.")

        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.feature_names = payload.get("feature_names", FEATURE_COLUMNS)
        logger.info("EarlyWarningAlertEngine initialized with model from %s", self.model_path)

    def predict_risk(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """Infers landslide probability and risk tier for a single location record."""
        # Compute aspect sin/cos if aspect_deg provided
        if "aspect_deg" in feature_dict and ("aspect_sin" not in feature_dict or "aspect_cos" not in feature_dict):
            rad = np.radians(feature_dict["aspect_deg"])
            feature_dict["aspect_sin"] = float(np.sin(rad))
            feature_dict["aspect_cos"] = float(np.cos(rad))

        # Vectorize in strict feature_names order
        vec = []
        for f in self.feature_names:
            vec.append(float(feature_dict.get(f, 0.0)))

        X = np.array([vec], dtype=np.float32)
        prob = float(self.model.predict_proba(X)[0, 1])

        risk_info = map_probability_to_risk_level(prob)
        return risk_info

    def generate_alert(
        self,
        latitude: float,
        longitude: float,
        feature_dict: Dict[str, float],
        location_name: Optional[str] = None,
        event_date: Optional[str] = None,
        save_records: bool = True,
    ) -> Dict[str, Any]:
        """Generates a complete, structured early warning alert record."""
        risk_info = self.predict_risk(feature_dict)
        now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        alert_id = f"ALERT-{uuid.uuid4().hex[:8].upper()}"

        alert_record = {
            "alert_id": alert_id,
            "timestamp_utc": now_str,
            "target_date": event_date or datetime.utcnow().strftime("%Y-%m-%d"),
            "location": {
                "name": location_name or "Western Ghats Monitoring Station",
                "latitude": round(latitude, 4),
                "longitude": round(longitude, 4),
            },
            "risk_assessment": {
                "level": risk_info["risk_level"],
                "probability": risk_info["probability"],
                "recommended_action": risk_info["recommended_action"],
                "color_code": risk_info["color"],
            },
            "key_trigger_conditions": {
                "3d_antecedent_rainfall_mm": round(feature_dict.get("rain_3d", 0.0), 1),
                "7d_antecedent_rainfall_mm": round(feature_dict.get("rain_7d", 0.0), 1),
                "terrain_slope_deg": round(feature_dict.get("slope_deg", 0.0), 1),
                "elevation_m": round(feature_dict.get("elevation_m", 0.0), 1),
                "vegetation_ndvi": round(feature_dict.get("ndvi", 0.0), 3),
            },
            "disclaimer": MANDATORY_ACADEMIC_DISCLAIMER,
        }

        if save_records:
            # 1. Save individual JSON record
            clean_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            json_file = self.alerts_dir / f"alert_{clean_ts}_{alert_id}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(alert_record, f, indent=2)

            # 2. Append row to warning_log.csv
            log_row = {
                "alert_id": alert_id,
                "timestamp_utc": now_str,
                "target_date": alert_record["target_date"],
                "location_name": alert_record["location"]["name"],
                "latitude": latitude,
                "longitude": longitude,
                "risk_level": risk_info["risk_level"],
                "probability": risk_info["probability"],
                "rain_3d_mm": alert_record["key_trigger_conditions"]["3d_antecedent_rainfall_mm"],
                "rain_7d_mm": alert_record["key_trigger_conditions"]["7d_antecedent_rainfall_mm"],
                "slope_deg": alert_record["key_trigger_conditions"]["terrain_slope_deg"],
                "recommended_action": risk_info["recommended_action"],
            }
            df_row = pd.DataFrame([log_row])
            write_header = not self.log_csv_path.exists()
            df_row.to_csv(self.log_csv_path, mode="a", header=write_header, index=False)

        return alert_record

    def process_batch(self, df: pd.DataFrame, save_csv: bool = True) -> pd.DataFrame:
        """Processes a batch DataFrame of candidate locations and appends risk assessment columns."""
        df_out = df.copy()
        X, _, _ = prepare_stage_b_features(df_out)
        probs = self.model.predict_proba(X)[:, 1]

        risk_levels = []
        actions = []

        for p in probs:
            r = map_probability_to_risk_level(p)
            risk_levels.append(r["risk_level"])
            actions.append(r["recommended_action"])

        df_out["predicted_probability"] = np.round(probs, 4)
        df_out["risk_level"] = risk_levels
        df_out["recommended_action"] = actions

        if save_csv:
            batch_output = self.alerts_dir / "batch_risk_assessment.csv"
            df_out.to_csv(batch_output, index=False)
            logger.info("Saved batch assessment (%d locations) to %s", len(df_out), batch_output)

        return df_out
