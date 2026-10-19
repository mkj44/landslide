#!/usr/bin/env python3
"""Script to download, filter, inspect, and verify raw datasets for Stage A and Stage B."""

import json
import logging
from pathlib import Path
import pandas as pd

from src.utils.logging import get_logger
from src.utils.seed import seed_everything
from src.utils.config import load_config
from src.data.download import (
    download_nasa_glc,
    filter_glc_study_region,
    generate_negative_samples,
    inspect_landslide4sense_dataset,
    plot_sample_patch_grids,
    plot_study_region_events,
)

logger = get_logger("download_pipeline")


def main():
    seed_everything(42)
    config = load_config("configs/data.yaml")

    print("\n=================================================================")
    print("       PHASE 1: DATA ACQUISITION, FILTERING & INSPECTION         ")
    print("=================================================================\n")

    # 1. NASA GLC Download and Filtering
    print("--- [1/4] NASA Global Landslide Catalog Acquisition ---")
    glc_raw = download_nasa_glc()
    bbox = tuple(config["stage_b"]["study_region"]["bbox"])
    # Western Ghats corridor: [73.0, 8.0, 77.8, 20.5]
    wg_bbox = (73.0, 8.0, 77.8, 20.5)

    pos_df, glc_stats = filter_glc_study_region(
        csv_path=str(glc_raw),
        bbox=wg_bbox,
        min_accuracy_levels=["exact", "1km", "5km", "10km"],
    )

    print(f"Total global landslide records in NASA GLC: {glc_stats['total_global_events']}")
    print(f"Events within Western Ghats corridor {wg_bbox}: {glc_stats['events_in_bbox']}")
    print(f"Events retained with high spatial accuracy (< 10 km): {glc_stats['retained_count']}")
    print(f"Events dropped due to low accuracy (>= 25 km / unknown): {glc_stats['dropped_low_accuracy_count']}")
    print("\nDropped low-accuracy distribution:")
    for acc, count in glc_stats["dropped_accuracy_distribution"].items():
        print(f"  {acc}: {count}")

    print("\nRetained positive events per year (2007–2016):")
    for yr, count in glc_stats["yearly_counts"].items():
        print(f"  Year {yr}: {count} events")

    # 2. Stage B Negative Sampling
    print("\n--- [2/4] Stage B Negative Non-Landslide Sampling ---")
    neg_df = generate_negative_samples(
        pos_df=pos_df,
        bbox=wg_bbox,
        ratio=2.0,
        buffer_km=5.0,
        random_seed=42,
    )
    print(f"Generated {len(neg_df)} negative non-landslide control points.")
    print("Sampling strategy: Random points strictly outside a 5.0 km circular buffer")
    print("around all known positive landslides, matched on seasonal calendar distribution.")

    # Mark positive labels and combine for Stage B catalog
    pos_df["label"] = 1
    combined_stage_b = pd.concat([pos_df, neg_df], ignore_index=True)
    out_stage_b = Path("data/processed/stage_b_events.csv")
    out_stage_b.parent.mkdir(parents=True, exist_ok=True)
    combined_stage_b.to_csv(out_stage_b, index=False)
    print(f"Saved combined Stage B events table to {out_stage_b} (Total rows: {len(combined_stage_b)})")

    # 3. Stage A Landslide4Sense Inspection
    print("\n--- [3/4] Stage A Landslide4Sense Patch Dataset Inspection ---")
    l4s_df, l4s_stats = inspect_landslide4sense_dataset(
        data_dir="data/raw/landslide4sense",
        thresholds=[0.001, 0.01, 0.05],
    )
    print(f"Inspected {l4s_stats['total_patches']} multi-spectral image/mask pairs.")
    print(f"Image tensor dimensions: {l4s_stats['patch_shape']}, Dtype: {l4s_stats['image_dtype']}")
    print(f"Mask tensor dimensions: (128, 128), Dtype: {l4s_stats['mask_dtype']}")
    print(f"Corrupt/NaN patches: {l4s_stats['nan_patches']}, Inf patches: {l4s_stats['inf_patches']}")
    print("\nLandslide pixel ratio threshold analysis (Patch label = 1 if landslide pixels / total >= threshold):")
    for th, res in l4s_stats["threshold_evaluations"].items():
        print(f"  Threshold >= {float(th):5.3f}: Positives = {res['positives']:3d} ({res['positive_rate']:.1%}), Negatives = {res['negatives']:3d}")

    out_l4s = Path("data/processed/landslide4sense_manifest.csv")
    l4s_df.to_csv(out_l4s, index=False)
    print(f"Saved Landslide4Sense patch manifest to {out_l4s}")

    # 4. Publication-grade Static Visualizations
    print("\n--- [4/4] Generating Publication-Grade Figures ---")
    patch_grid_path = plot_sample_patch_grids(
        data_dir="data/raw/landslide4sense",
        output_png="results/figures/landslide4sense_sample_grid.png",
        num_samples=4,
    )
    print(f"Saved multi-spectral patch grid figure: {patch_grid_path}")

    scatter_map_path = plot_study_region_events(
        pos_df=pos_df,
        neg_df=neg_df,
        output_png="results/figures/study_region_events.png",
        bbox=wg_bbox,
    )
    print(f"Saved study region event scatter map: {scatter_map_path}")

    print("\n=================================================================")
    print("       PHASE 1 EXECUTION AND INSPECTION CHECKS COMPLETE          ")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
