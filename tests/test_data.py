"""Unit and integration tests for Phase 1 data acquisition, filtering, and negative sampling."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.download import (
    filter_glc_study_region,
    generate_negative_samples,
    inspect_landslide4sense_dataset,
)


def test_glc_filtering():
    """Verify that NASA GLC filtering enforces spatial bounding and accuracy thresholds."""
    csv_path = "data/raw/Global_Landslide_Catalog_Export.csv"
    assert Path(csv_path).exists(), "NASA GLC raw export must be present."

    wg_bbox = (73.0, 8.0, 77.8, 20.5)
    pos_df, stats = filter_glc_study_region(
        csv_path=csv_path,
        bbox=wg_bbox,
        min_accuracy_levels=["exact", "1km", "5km", "10km"],
    )

    assert len(pos_df) > 0, "Expected positive events in Western Ghats."
    assert stats["dropped_low_accuracy_count"] > 0, "Low-accuracy records must be logged and dropped."

    # Verify all retained points are inside bbox
    assert pos_df["longitude"].min() >= wg_bbox[0]
    assert pos_df["latitude"].min() >= wg_bbox[1]
    assert pos_df["longitude"].max() <= wg_bbox[2]
    assert pos_df["latitude"].max() <= wg_bbox[3]

    # Verify accuracy levels
    assert set(pos_df["location_accuracy"].unique()).issubset({"exact", "1km", "5km", "10km"})


def test_negative_sampling_buffer():
    """Verify that negative non-landslide points maintain strict buffer exclusion from positive events."""
    csv_path = "data/raw/Global_Landslide_Catalog_Export.csv"
    wg_bbox = (73.0, 8.0, 77.8, 20.5)
    pos_df, _ = filter_glc_study_region(csv_path=csv_path, bbox=wg_bbox)

    buffer_km = 5.0
    buffer_deg = buffer_km / 111.0
    neg_df = generate_negative_samples(pos_df=pos_df, bbox=wg_bbox, ratio=1.5, buffer_km=buffer_km, random_seed=42)

    assert len(neg_df) == int(len(pos_df) * 1.5)
    assert (neg_df["label"] == 0).all()

    # Test exclusion buffer on all pairs
    pos_coords = np.column_stack([pos_df["longitude"].values, pos_df["latitude"].values])
    for _, row in neg_df.iterrows():
        n_lon, n_lat = row["longitude"], row["latitude"]
        dists = np.hypot(pos_coords[:, 0] - n_lon, pos_coords[:, 1] - n_lat)
        assert np.min(dists) >= buffer_deg, f"Negative sample at ({n_lon}, {n_lat}) violated {buffer_km}km buffer!"


def test_landslide4sense_patches_and_labels():
    """Verify Landslide4Sense patch dimensions, dtypes, and multi-threshold label derivation."""
    df, stats = inspect_landslide4sense_dataset("data/raw/landslide4sense", thresholds=[0.001, 0.01, 0.05])

    assert stats["total_patches"] >= 50, "Expected at least 50 patches for testing."
    assert stats["patch_shape"] == (128, 128, 14), "Landslide4Sense patches must have shape (128, 128, 14)."
    assert stats["nan_patches"] == 0, "No NaN values allowed in raw patches."
    assert stats["inf_patches"] == 0, "No Inf values allowed in raw patches."

    # Verify monotonicity: positive count at 0.001 >= count at 0.01 >= count at 0.05
    c_001 = stats["threshold_evaluations"]["0.001"]["positives"]
    c_010 = stats["threshold_evaluations"]["0.01"]["positives"]
    c_050 = stats["threshold_evaluations"]["0.05"]["positives"]
    assert c_001 >= c_010 >= c_050


def test_figures_generated():
    """Verify publication figures are created on disk with valid size."""
    patch_grid = Path("results/figures/landslide4sense_sample_grid.png")
    scatter_map = Path("results/figures/study_region_events.png")

    assert patch_grid.exists() and patch_grid.stat().st_size > 100000, "Patch grid figure missing or too small."
    assert scatter_map.exists() and scatter_map.stat().st_size > 50000, "Scatter map figure missing or too small."
