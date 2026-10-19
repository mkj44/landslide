import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from src.warning.alert_engine import EarlyWarningAlertEngine, MANDATORY_ACADEMIC_DISCLAIMER
from src.warning.mapping import plot_regional_risk_map
from scripts.warn import print_alert_banner, COLOR_BOLD, COLOR_RED, COLOR_YELLOW, COLOR_GREEN, COLOR_ORANGE, COLOR_RESET


def main():
    print(f"\n{COLOR_BOLD}=============================================================================={COLOR_RESET}")
    print(f"{COLOR_BOLD}         LANDSLIDE EARLY-WARNING & RISK ENGINE: DEMO RUN (Stage C)           {COLOR_RESET}")
    print(f"{COLOR_BOLD}=============================================================================={COLOR_RESET}")

    engine = EarlyWarningAlertEngine()

    # 1. Single Location High-Risk Monsoon Simulation
    sample_monsoon_features = {
        "elevation_m": 720.0,
        "slope_deg": 31.5,
        "aspect_deg": 240.0,
        "prof_curvature": 0.04,
        "plan_curvature": -0.02,
        "tri": 42.0,
        "ndvi": 0.38,
        "ndwi": -0.08,
        "rain_1d": 55.0,
        "rain_3d": 145.0,
        "rain_7d": 330.0,
        "rain_14d": 490.0,
        "rain_30d": 850.0,
        "rain_max_intensity": 72.0,
    }

    print("\n>>> [1/2] Generating High-Risk Alert Record for Western Ghats Field Station...")
    alert = engine.generate_alert(
        latitude=11.855,
        longitude=75.361,
        feature_dict=sample_monsoon_features,
        location_name="Kannur Sector 4",
        save_records=True,
    )
    print_alert_banner(alert)

    # 2. Regional Batch Assessment over 504 Monitoring Sites
    batch_csv = Path("data/processed/stage_b_features.csv")
    if batch_csv.exists():
        print(">>> [2/2] Generating Regional Batch Assessment & Spatial Risk Map...")
        df_batch = pd.read_csv(batch_csv)
        df_assessed = engine.process_batch(df_batch, save_csv=True)

        critical_n = int((df_assessed["risk_level"] == "Critical").sum())
        high_n = int((df_assessed["risk_level"] == "High").sum())
        mod_n = int((df_assessed["risk_level"] == "Moderate").sum())
        low_n = int((df_assessed["risk_level"] == "Low").sum())

        print("\n" + "=" * 65)
        print(" REGIONAL EARLY WARNING BATCH ASSESSMENT SUMMARY")
        print("=" * 65)
        print(f" Total Locations Evaluated : {len(df_assessed)}")
        print(f"   • {COLOR_RED}CRITICAL ALERTS{COLOR_RESET}         : {critical_n}")
        print(f"   • {COLOR_ORANGE}HIGH RISK WARNINGS{COLOR_RESET}      : {high_n}")
        print(f"   • {COLOR_YELLOW}MODERATE ADVISORIES{COLOR_RESET}    : {mod_n}")
        print(f"   • {COLOR_GREEN}LOW RISK NORMAL{COLOR_RESET}        : {low_n}")
        print("-" * 65)

        map_output = "results/figures/warning_risk_map.png"
        plot_regional_risk_map(df_assessed, output_path=map_output)
        print(f" Static risk map generated at : {map_output}")
        print(f" Alert logs saved to          : results/alerts/warning_log.csv")
        print(f" Batch CSV saved to           : results/alerts/batch_risk_assessment.csv")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
