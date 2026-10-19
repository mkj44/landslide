#!/usr/bin/env python3
"""Preprocessing and feature engineering pipeline for Stage A and Stage B."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import h5py

from src.utils.logging import get_logger
from src.utils.seed import seed_everything
from src.data.splits import (
    create_stage_a_splits,
    create_stage_b_spatial_cv_splits,
    create_stage_b_temporal_holdout_split,
)
from src.data.preprocessing import (
    compute_normalization_stats,
    apply_normalization,
)
from src.features.feature_table import build_stage_b_feature_table

logger = get_logger("preprocess_pipeline")


def plot_class_distributions(
    stage_a_manifest: pd.DataFrame,
    stage_a_splits: dict,
    stage_b_df: pd.DataFrame,
    stage_b_temp_split: dict,
    output_png: str = "results/figures/class_distribution.png",
):
    """Generates a combined bar chart of class balances across all splits."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Stage A
    labels_a = []
    splits_a = []
    classes_a = []
    for split_name in ["train", "val", "test"]:
        pids = set(stage_a_splits[split_name])
        sub = stage_a_manifest[stage_a_manifest["patch_id"].isin(pids)]
        pos = int(sub["label_th_0.01"].sum())
        neg = len(sub) - pos
        splits_a.extend([f"Stage A\n{split_name.capitalize()}", f"Stage A\n{split_name.capitalize()}"])
        classes_a.extend(["Landslide (1)", "Non-Landslide (0)"])
        labels_a.extend([pos, neg])

    df_plot_a = pd.DataFrame({"Split": splits_a, "Class": classes_a, "Count": labels_a})
    sns.barplot(data=df_plot_a, x="Split", y="Count", hue="Class", palette=["#e74c3c", "#3498db"], ax=axes[0])
    axes[0].set_title("Stage A (Landslide4Sense) Split Balances", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Number of Patches", fontsize=10, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    for p in axes[0].patches:
        height = p.get_height()
        if height > 0:
            axes[0].annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    # Stage B
    train_idx = stage_b_temp_split["train_indices"]
    test_idx = stage_b_temp_split["test_indices"]
    tr_df = stage_b_df.iloc[train_idx]
    te_df = stage_b_df.iloc[test_idx]

    df_plot_b = pd.DataFrame({
        "Split": ["Stage B\nTrain (<=2014)", "Stage B\nTrain (<=2014)", "Stage B\nTest (2015-17)", "Stage B\nTest (2015-17)"],
        "Class": ["Landslide (1)", "Non-Landslide (0)", "Landslide (1)", "Non-Landslide (0)"],
        "Count": [
            int((tr_df["label"] == 1).sum()), int((tr_df["label"] == 0).sum()),
            int((te_df["label"] == 1).sum()), int((te_df["label"] == 0).sum()),
        ]
    })
    sns.barplot(data=df_plot_b, x="Split", y="Count", hue="Class", palette=["#e74c3c", "#3498db"], ax=axes[1])
    axes[1].set_title("Stage B (Western Ghats) Temporal Split Balances", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Number of Locations / Events", fontsize=10, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    for p in axes[1].patches:
        height = p.get_height()
        if height > 0:
            axes[1].annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    plt.suptitle("Class Distribution across Detection & Prediction Data Splits", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved class distribution figure to %s", output_png)


def plot_band_histograms(
    norm_stats: dict,
    train_patch_paths: list,
    output_png: str = "results/figures/band_histograms_norm.png",
):
    """Generates side-by-side histograms before and after normalization across representative bands."""
    bands_to_plot = [
        (1, "B02 (Blue)"),
        (3, "B04 (Red)"),
        (7, "B08 (NIR)"),
        (12, "DEM (m)"),
        (13, "Slope (deg)"),
    ]

    raw_vals = {b: [] for b, _ in bands_to_plot}
    norm_vals = {b: [] for b, _ in bands_to_plot}

    # Sample pixels across 20 training patches
    for p in train_patch_paths[:20]:
        with h5py.File(p, "r") as f:
            img = f["img"][:]
            img_norm = apply_normalization(img, norm_stats, band_subset="all14")
            for b, _ in bands_to_plot:
                raw_vals[b].extend(img[:, :, b].flatten()[::15].tolist())
                norm_vals[b].extend(img_norm[:, :, b].flatten()[::15].tolist())

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), dpi=300)

    for idx, (b, name) in enumerate(bands_to_plot):
        # Raw
        axes[0, idx].hist(raw_vals[b], bins=30, color="#2980b9", alpha=0.75, edgecolor="black", linewidth=0.5)
        axes[0, idx].set_title(f"Raw: {name}", fontsize=10, fontweight="bold")
        axes[0, idx].grid(True, linestyle="--", alpha=0.4)
        if idx == 0:
            axes[0, idx].set_ylabel("Raw Frequency", fontsize=10, fontweight="bold")

        # Normalized
        axes[1, idx].hist(norm_vals[b], bins=30, color="#27ae60", alpha=0.75, edgecolor="black", linewidth=0.5)
        axes[1, idx].set_title(f"Normalized (Z-Score)", fontsize=10, fontweight="bold")
        axes[1, idx].set_xlabel("Value", fontsize=9)
        axes[1, idx].grid(True, linestyle="--", alpha=0.4)
        if idx == 0:
            axes[1, idx].set_ylabel("Normalized Frequency", fontsize=10, fontweight="bold")

    plt.suptitle("Multi-Spectral Band Pixel Distributions Before and After Training-Split Normalization",
                 fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved band histograms figure to %s", output_png)


def plot_feature_distributions_and_correlations(
    stage_b_df: pd.DataFrame,
    violin_png: str = "results/figures/feature_distributions_by_class.png",
    corr_png: str = "results/figures/correlation_heatmap.png",
):
    """Generates violin plots of key features by class and correlation matrix heatmap."""
    key_features = [
        ("slope_deg", "Slope Angle (°)"),
        ("elevation_m", "Elevation (m)"),
        ("rain_3d", "3-Day Rainfall (mm)"),
        ("rain_7d", "7-Day Rainfall (mm)"),
        ("ndvi", "NDVI (Vegetation)"),
        ("tri", "Terrain Ruggedness (TRI)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
    palette = {0: "#3498db", 1: "#e74c3c"}

    for idx, (feat, label) in enumerate(key_features):
        ax = axes[idx // 3, idx % 3]
        sns.violinplot(
            data=stage_b_df, x="label", y=feat, hue="label",
            palette=palette, inner="quartile", ax=ax, legend=False
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-Landslide (0)", "Landslide (1)"], fontweight="bold")
        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Stage B Predictive Feature Distributions Stratified by Class",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    Path(violin_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(violin_png, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved feature distributions figure to %s", violin_png)

    # Correlation Matrix
    corr_cols = [
        "elevation_m", "slope_deg", "prof_curvature", "plan_curvature", "tri",
        "ndvi", "ndwi", "rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d", "label"
    ]
    corr_df = stage_b_df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(
        corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        linewidths=0.5, cbar_kws={"label": "Pearson Correlation Coefficient"}, ax=ax
    )
    ax.set_title("Feature Correlation Matrix (Conditioning, Triggers & Landslide Target)",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    Path(corr_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(corr_png, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved correlation heatmap figure to %s", corr_png)


def main():
    seed_everything(42)
    print("\n=================================================================")
    print("       PHASE 2: PREPROCESSING & FEATURE ENGINEERING PIPELINE     ")
    print("=================================================================\n")

    # 1. Stage A Splits (Train 70%, Val 15%, Test 15%)
    print("--- [1/5] Creating Stage A Leakage-Safe Splits ---")
    stage_a_manifest = pd.read_csv("data/processed/landslide4sense_manifest.csv")
    stage_a_splits = create_stage_a_splits(
        manifest_csv="data/processed/landslide4sense_manifest.csv",
        save_path="artifacts/splits/stage_a_splits.json",
        train_ratio=0.70,
        val_ratio=0.15,
        random_seed=42,
    )
    print(f"Stage A Splits Created: Train={len(stage_a_splits['train'])}, "
          f"Val={len(stage_a_splits['val'])}, Test={len(stage_a_splits['test'])}")

    # 2. Stage A Per-Band Normalization (Training Split Only)
    print("\n--- [2/5] Computing Training-Only Per-Band Normalization Stats ---")
    train_patch_paths = [
        Path(f"data/raw/landslide4sense/images/train/image_{pid}.h5")
        for pid in stage_a_splits["train"]
    ]
    norm_stats = compute_normalization_stats(
        train_patch_paths=train_patch_paths,
        save_json_path="artifacts/norm_stats.json",
        percentile_clip=True,
    )
    print("Calculated 14-band means and standard deviations from training split only.")
    print(f"Sample Band 4 (Red) Mean: {norm_stats['means'][3]:.3f}, Std: {norm_stats['stds'][3]:.3f}")
    print(f"Sample Band 8 (NIR) Mean: {norm_stats['means'][7]:.3f}, Std: {norm_stats['stds'][7]:.3f}")
    print(f"Sample DEM (B13) Mean: {norm_stats['means'][12]:.3f}, Std: {norm_stats['stds'][12]:.3f}")

    # 3. Stage B Feature Table Assembly
    print("\n--- [3/5] Building Stage B Predictive Feature Matrix ---")
    stage_b_features = build_stage_b_feature_table(
        events_csv="data/processed/stage_b_events.csv",
        output_parquet="data/processed/stage_b_features.parquet",
        grid_size_deg=0.5,
        random_seed=42,
    )
    print(f"Stage B Feature Matrix Shape: {stage_b_features.shape}")
    print(f"Features Generated: {[c for c in stage_b_features.columns if c not in ['event_id', 'latitude', 'longitude', 'event_date']]}")

    # 4. Stage B Leakage-Safe Splits: Spatial Block CV & Temporal Holdout
    print("\n--- [4/5] Constructing Spatial Block CV & Temporal Holdout Splits ---")
    spatial_cv = create_stage_b_spatial_cv_splits(
        features_csv="data/processed/stage_b_features.csv",
        save_path="artifacts/splits/stage_b_spatial_cv_splits.json",
        n_splits=5,
        group_col="grid_cell_id",
    )
    print("Spatial Block Cross-Validation: 5 independent geographic folds created.")

    temp_split = create_stage_b_temporal_holdout_split(
        features_csv="data/processed/stage_b_features.csv",
        save_path="artifacts/splits/stage_b_temporal_split.json",
        cutoff_year=2014,
    )
    print(f"Temporal Holdout Split: Train (<=2014) = {temp_split['train_count']} events, "
          f"Test (>2014: 2015-2017) = {temp_split['test_count']} events.")

    # 5. Publication-Grade Figures
    print("\n--- [5/5] Generating Diagnostic & Publication Figures ---")
    plot_class_distributions(stage_a_manifest, stage_a_splits, stage_b_features, temp_split)
    plot_band_histograms(norm_stats, train_patch_paths)
    plot_feature_distributions_and_correlations(stage_b_features)

    print("\n=================================================================")
    print("       PHASE 2 PREPROCESSING & FEATURE ENGINEERING COMPLETE      ")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
