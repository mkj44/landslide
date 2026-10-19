"""Command-Line Interface for Landslide Early Warning and Risk Assessment (Stage C).

Usage Examples:
  1. Single Coordinate Assessment:
     python scripts/warn.py --lat 11.855 --lon 75.361 --rain-3d 120 --rain-7d 280 --slope 28

  2. Scenario Rainfall Forecast Testing:
     python scripts/warn.py --lat 15.401 --lon 73.827 --forecast-rain 150

  3. Full Regional Batch Assessment & Static Mapping:
     python scripts/warn.py --batch data/processed/stage_b_features.csv --map
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from src.warning.alert_engine import EarlyWarningAlertEngine, MANDATORY_ACADEMIC_DISCLAIMER
from src.warning.mapping import plot_regional_risk_map

# ANSI Color codes for terminal UI
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[91m"
COLOR_YELLOW = "\033[93m"
COLOR_GREEN = "\033[92m"
COLOR_CYAN = "\033[96m"
COLOR_ORANGE = "\033[38;5;208m"

BADGE_COLORS = {
    "Critical": f"{COLOR_BOLD}{COLOR_RED}[CRITICAL EMERGENCY ALERT]{COLOR_RESET}",
    "High": f"{COLOR_BOLD}{COLOR_ORANGE}[HIGH RISK WARNING]{COLOR_RESET}",
    "Moderate": f"{COLOR_BOLD}{COLOR_YELLOW}[MODERATE ADVISORY]{COLOR_RESET}",
    "Low": f"{COLOR_BOLD}{COLOR_GREEN}[LOW RISK NORMAL]{COLOR_RESET}",
}


def print_alert_banner(alert_record: dict):
    """Prints a beautiful formatted early warning alert report to stdout."""
    loc = alert_record["location"]
    risk = alert_record["risk_assessment"]
    trig = alert_record["key_trigger_conditions"]
    level = risk["level"]
    badge = BADGE_COLORS.get(level, f"[{level.upper()}]")

    print("\n" + "=" * 78)
    print(f" {COLOR_BOLD}LANDSLIDE RISK ASSESSMENT & EARLY-WARNING ADVISORY (Stage C){COLOR_RESET}")
    print("=" * 78)
    print(f" Alert ID       : {alert_record['alert_id']}")
    print(f" Timestamp      : {alert_record['timestamp_utc']}")
    print(f" Target Date    : {alert_record['target_date']}")
    print(f" Location       : Lat {loc['latitude']:.4f}°N, Lon {loc['longitude']:.4f}°E ({loc['name']})")
    print("-" * 78)
    print(f" STATUS LEVEL   : {badge}")
    print(f" Probability    : {risk['probability']:.4f} ({risk['probability']*100:.1f}%)")
    print(f" Recommended    : {COLOR_BOLD}{risk['recommended_action']}{COLOR_RESET}")
    print("-" * 78)
    print(" Key Environmental & Trigger Conditions:")
    print(f"   • 3-Day Antecedent Rainfall : {trig['3d_antecedent_rainfall_mm']} mm")
    print(f"   • 7-Day Antecedent Rainfall : {trig['7d_antecedent_rainfall_mm']} mm")
    print(f"   • Terrain Slope Angle       : {trig['terrain_slope_deg']}°")
    print(f"   • Digital Elevation         : {trig['elevation_m']} m")
    print(f"   • Vegetation Density (NDVI) : {trig['vegetation_ndvi']}")
    print("-" * 78)
    print(f" {COLOR_YELLOW}{MANDATORY_ACADEMIC_DISCLAIMER}{COLOR_RESET}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Landslide Early Warning & Risk Assessment Engine (Stage C)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lat", type=float, default=11.855, help="Target latitude (degrees North)")
    parser.add_argument("--lon", type=float, default=75.361, help="Target longitude (degrees East)")
    parser.add_argument("--date", type=str, default=None, help="Target assessment date (YYYY-MM-DD)")
    parser.add_argument("--rain-3d", type=float, default=110.0, help="3-day antecedent rainfall in mm")
    parser.add_argument("--rain-7d", type=float, default=240.0, help="7-day antecedent rainfall in mm")
    parser.add_argument("--slope", type=float, default=28.0, help="Terrain slope angle in degrees")
    parser.add_argument("--elevation", type=float, default=650.0, help="Digital elevation in meters")
    parser.add_argument("--ndvi", type=float, default=0.45, help="Vegetation index NDVI")
    parser.add_argument("--forecast-rain", type=float, default=None, help="Scenario forecasted rainfall in mm")
    parser.add_argument("--batch", type=str, default=None, help="Path to CSV for regional batch processing")
    parser.add_argument("--map", action="store_true", help="Generate regional spatial risk map")
    parser.add_argument("--output-map", type=str, default="results/figures/warning_risk_map.png", help="Map PNG path")

    args = parser.parse_args()

    engine = EarlyWarningAlertEngine()

    if args.batch:
        batch_p = Path(args.batch)
        if not batch_p.exists():
            print(f"Error: Batch file {batch_p} does not exist.")
            sys.exit(1)

        print(f"\nProcessing regional batch assessment from {batch_p}...")
        df_batch = pd.read_csv(batch_p)
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
        print(f" Results logged to: results/alerts/batch_risk_assessment.csv")

        if args.map:
            print(f" Generating regional spatial risk map...")
            plot_regional_risk_map(df_assessed, output_path=args.output_map)
            print(f" Static risk map saved to: {args.output_map}")

        print(f"\n{COLOR_YELLOW}{MANDATORY_ACADEMIC_DISCLAIMER}{COLOR_RESET}\n")

    else:
        # Single Coordinate or Scenario Mode
        features = {
            "elevation_m": args.elevation,
            "slope_deg": args.slope,
            "aspect_deg": 225.0,
            "prof_curvature": -0.02,
            "plan_curvature": 0.01,
            "tri": 35.0,
            "ndvi": args.ndvi,
            "ndwi": -0.05,
            "rain_1d": args.rain_3d * 0.4 if not args.forecast_rain else args.forecast_rain * 0.6,
            "rain_3d": args.rain_3d if not args.forecast_rain else args.forecast_rain,
            "rain_7d": args.rain_7d if not args.forecast_rain else args.forecast_rain * 1.5,
            "rain_14d": args.rain_7d * 1.3,
            "rain_30d": args.rain_7d * 2.0,
            "rain_max_intensity": args.rain_3d * 0.5,
        }

        alert = engine.generate_alert(
            latitude=args.lat,
            longitude=args.lon,
            feature_dict=features,
            location_name="Western Ghats Field Station",
            event_date=args.date,
            save_records=True,
        )

        print_alert_banner(alert)
        print(f"Saved JSON alert record to: results/alerts/")
        print(f"Appended to log: results/alerts/warning_log.csv\n")


if __name__ == "__main__":
    main()
