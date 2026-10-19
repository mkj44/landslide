"""Static spatial early-warning risk mapping."""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils.logging import get_logger
from src.warning.alert_engine import MANDATORY_ACADEMIC_DISCLAIMER

logger = get_logger("warning_mapping")

RISK_COLOR_MAP = {
    "Low": "#2ecc71",
    "Moderate": "#f39c12",
    "High": "#e67e22",
    "Critical": "#e74c3c",
}


def plot_regional_risk_map(
    df_assessed: pd.DataFrame,
    output_path: str = "results/figures/warning_risk_map.png",
    title: str = "Western Ghats Corridor: Regional Landslide Early-Warning Status",
):
    """Generates a static publication-grade geographic risk map."""
    fig, ax = plt.subplots(figsize=(10, 12))

    lons = df_assessed["longitude"].values
    lats = df_assessed["latitude"].values
    risks = df_assessed["risk_level"].values
    probs = df_assessed["predicted_probability"].values

    # Background gradient / grid representing terrain elevation
    xi = np.linspace(73.0, 78.0, 150)
    yi = np.linspace(8.0, 21.0, 200)
    Xi, Yi = np.meshgrid(xi, yi)

    # Synthetic shaded relief terrain for visual spatial context
    Zi = 200 + 1200 * np.exp(-((Xi - 75.5)**2 / 1.5 + (Yi - 13.0)**2 / 15.0))
    ax.contourf(Xi, Yi, Zi, levels=25, cmap="Greys", alpha=0.3)

    # Plot monitoring stations by risk tier
    for tier, color in RISK_COLOR_MAP.items():
        mask = risks == tier
        if np.any(mask):
            ax.scatter(
                lons[mask],
                lats[mask],
                c=color,
                label=f"{tier} Risk ({np.sum(mask)} sites)",
                s=80 if tier in ["High", "Critical"] else 40,
                alpha=0.9,
                edgecolors="black" if tier in ["High", "Critical"] else "none",
                linewidths=1.2,
                zorder=5 if tier in ["High", "Critical"] else 3,
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.set_xlim(72.8, 78.2)
    ax.set_ylim(7.8, 21.2)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Count statistics summary box
    total_sites = len(df_assessed)
    critical_count = int(np.sum(risks == "Critical"))
    high_count = int(np.sum(risks == "High"))
    moderate_count = int(np.sum(risks == "Moderate"))
    low_count = int(np.sum(risks == "Low"))

    summary_text = (
        f"Regional Risk Summary\n"
        f"----------------------\n"
        f"Total Sites: {total_sites}\n"
        f"Critical: {critical_count}\n"
        f"High: {high_count}\n"
        f"Moderate: {moderate_count}\n"
        f"Low: {low_count}"
    )
    ax.text(
        0.03, 0.05, summary_text,
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#cccccc", alpha=0.9)
    )

    ax.legend(loc="upper right", frameon=True, fontsize=10, facecolor="white", framealpha=0.95)

    # Mandatory Academic Disclaimer Box at the bottom
    fig.text(
        0.5, 0.01,
        f"{MANDATORY_ACADEMIC_DISCLAIMER}",
        ha="center",
        fontsize=8.5,
        color="#c0392b",
        fontweight="bold",
        wrap=True,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved static regional warning risk map to %s", output_path)
