"""Data acquisition, filtering, negative sampling, and inspection module."""

import os
import glob
import json
import logging
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import Point, MultiPoint
import geopandas as gpd

from src.utils.logging import get_logger

logger = get_logger("data_download")

NASA_GLC_URL = "https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/Global_Landslide_Catalog_Export_rows.csv"


def download_nasa_glc(dest_path: str = "data/raw/Global_Landslide_Catalog_Export.csv") -> Path:
    """Downloads the official NASA Global Landslide Catalog CSV if not already present."""
    dest = Path(dest_path)
    if dest.exists() and dest.stat().st_size > 1000:
        logger.info("NASA GLC file already exists at %s (%d bytes).", dest, dest.stat().st_size)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading NASA Global Landslide Catalog from %s...", NASA_GLC_URL)
    urllib.request.urlretrieve(NASA_GLC_URL, str(dest))
    logger.info("Download completed successfully. Saved to %s (%d bytes).", dest, dest.stat().st_size)
    return dest


def filter_glc_study_region(
    csv_path: str = "data/raw/Global_Landslide_Catalog_Export.csv",
    bbox: Tuple[float, float, float, float] = (73.0, 8.0, 77.8, 20.5), # [min_lon, min_lat, max_lon, max_lat] Western Ghats
    min_accuracy_levels: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads NASA GLC, bounds to study region, filters low-accuracy events, and records metrics.

    Args:
        csv_path: Path to the raw NASA GLC CSV.
        bbox: Bounding box tuple (min_lon, min_lat, max_lon, max_lat).
        min_accuracy_levels: Allowed location accuracy values. Defaults to ['exact', '1km', '5km', '10km'].

    Returns:
        Filtered DataFrame of events and filtering statistics dict.
    """
    if min_accuracy_levels is None:
        min_accuracy_levels = ["exact", "1km", "5km", "10km"]

    df = pd.read_csv(csv_path)
    total_raw = len(df)

    # Ensure latitude and longitude are valid numeric
    df = df.dropna(subset=["latitude", "longitude", "event_date"]).copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()

    # Spatial bounding
    min_lon, min_lat, max_lon, max_lat = bbox
    in_bbox = df[
        (df["longitude"] >= min_lon) & (df["longitude"] <= max_lon) &
        (df["latitude"] >= min_lat) & (df["latitude"] <= max_lat)
    ].copy()
    bbox_count = len(in_bbox)

    # Location accuracy filtering
    in_bbox["location_accuracy"] = in_bbox["location_accuracy"].astype(str).str.strip().str.lower()
    acc_filter = in_bbox["location_accuracy"].isin([lvl.lower() for lvl in min_accuracy_levels])

    retained = in_bbox[acc_filter].copy()
    dropped = in_bbox[~acc_filter].copy()

    # Parse and clean dates
    retained["event_date"] = pd.to_datetime(retained["event_date"], errors="coerce")
    retained = retained.dropna(subset=["event_date"]).sort_values("event_date").reset_index(drop=True)
    retained["year"] = retained["event_date"].dt.year
    retained["month"] = retained["event_date"].dt.month
    retained["day"] = retained["event_date"].dt.day

    stats = {
        "total_global_events": total_raw,
        "study_region_bbox": list(bbox),
        "events_in_bbox": bbox_count,
        "accuracy_levels_retained": min_accuracy_levels,
        "retained_count": len(retained),
        "dropped_low_accuracy_count": len(dropped),
        "dropped_accuracy_distribution": dropped["location_accuracy"].value_counts().to_dict(),
        "year_min": int(retained["year"].min()) if len(retained) > 0 else None,
        "year_max": int(retained["year"].max()) if len(retained) > 0 else None,
        "yearly_counts": retained["year"].value_counts().sort_index().to_dict(),
    }

    logger.info("GLC Filter: In bbox=%d, Retained high-accuracy=%d, Dropped=%d",
                bbox_count, len(retained), len(dropped))
    return retained, stats


def generate_negative_samples(
    pos_df: pd.DataFrame,
    bbox: Tuple[float, float, float, float] = (73.0, 8.0, 77.8, 20.5),
    ratio: float = 2.0,
    buffer_km: float = 5.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generates non-landslide control points strictly outside an exclusion buffer around known events.

    Args:
        pos_df: Positive landslide events DataFrame with 'latitude', 'longitude', 'event_date'.
        bbox: Bounding box tuple (min_lon, min_lat, max_lon, max_lat).
        ratio: Number of negative samples per positive event.
        buffer_km: Exclusion buffer radius in kilometers around all positive events.
        random_seed: Seed for reproducible spatial sampling.

    Returns:
        DataFrame containing negative non-landslide sample locations and matched dates.
    """
    rng = np.random.default_rng(random_seed)
    n_samples = int(len(pos_df) * ratio)
    min_lon, min_lat, max_lon, max_lat = bbox

    # Approx buffer in degrees: 1 deg lat ~ 111 km, 1 deg lon ~ 111 * cos(lat)
    # Using 5 km buffer: 5 / 111 ~ 0.045 degrees
    buffer_deg = buffer_km / 111.0

    pos_coords = np.column_stack([pos_df["longitude"].values, pos_df["latitude"].values])

    neg_points: List[Tuple[float, float]] = []
    # Sample candidate coordinates rejection-sampled outside buffer
    attempts = 0
    max_attempts = n_samples * 100

    while len(neg_points) < n_samples and attempts < max_attempts:
        batch_size = max(50, n_samples - len(neg_points))
        candidate_lons = rng.uniform(min_lon, max_lon, size=batch_size)
        candidate_lats = rng.uniform(min_lat, max_lat, size=batch_size)

        for lon, lat in zip(candidate_lons, candidate_lats):
            # Compute Euclidean distance in degrees to nearest positive event
            dists = np.hypot(pos_coords[:, 0] - lon, pos_coords[:, 1] - lat)
            if np.min(dists) >= buffer_deg:
                neg_points.append((lon, lat))
                if len(neg_points) >= n_samples:
                    break
        attempts += batch_size

    neg_points = neg_points[:n_samples]
    neg_lons = [p[0] for p in neg_points]
    neg_lats = [p[1] for p in neg_points]

    # Match dates to the positive events distribution (sampling from same year/season window)
    matched_dates = rng.choice(pos_df["event_date"].values, size=len(neg_points), replace=True)
    matched_dates = pd.to_datetime(matched_dates)

    neg_df = pd.DataFrame({
        "event_id": [f"NEG_{i+1:04d}" for i in range(len(neg_points))],
        "latitude": neg_lats,
        "longitude": neg_lons,
        "event_date": matched_dates,
        "year": matched_dates.year,
        "month": matched_dates.month,
        "day": matched_dates.day,
        "label": 0,
        "landslide_trigger": "none_background",
        "location_accuracy": "sampled_control",
    })

    logger.info("Generated %d negative control samples (ratio=%.1f, buffer=%.1f km).",
                len(neg_df), ratio, buffer_km)
    return neg_df


def inspect_landslide4sense_dataset(
    data_dir: str = "data/raw/landslide4sense",
    thresholds: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads and inspects Landslide4Sense image and mask pairs, printing shapes, dtypes, and class counts."""
    if thresholds is None:
        thresholds = [0.001, 0.01, 0.05]

    img_dir = Path(data_dir) / "images" / "train"
    mask_dir = Path(data_dir) / "annotations" / "train"

    img_files = sorted(glob.glob(str(img_dir / "image_*.h5")))
    records = []

    for img_path in img_files:
        base_name = os.path.basename(img_path)
        patch_id = base_name.replace("image_", "").replace(".h5", "")
        mask_path = mask_dir / f"mask_{patch_id}.h5"
        if not mask_path.exists():
            continue

        with h5py.File(img_path, "r") as f_img, h5py.File(mask_path, "r") as f_mask:
            img = f_img["img"][:]
            mask = f_mask["mask"][:]

            pos_pixels = int((mask > 0).sum())
            total_pixels = int(mask.size)
            ratio = pos_pixels / total_pixels

            record = {
                "patch_id": int(patch_id),
                "image_shape": img.shape,
                "image_dtype": str(img.dtype),
                "mask_shape": mask.shape,
                "mask_dtype": str(mask.dtype),
                "landslide_pixels": pos_pixels,
                "total_pixels": total_pixels,
                "landslide_ratio": ratio,
                "has_nan": bool(np.isnan(img).any()),
                "has_inf": bool(np.isinf(img).any()),
                "min_value": float(np.min(img)),
                "max_value": float(np.max(img)),
                "mean_value": float(np.mean(img)),
            }
            for th in thresholds:
                record[f"label_th_{th}"] = int(ratio >= th)

            records.append(record)

    df = pd.DataFrame(records)
    class_counts = {}
    for th in thresholds:
        pos = int(df[f"label_th_{th}"].sum())
        neg = len(df) - pos
        class_counts[str(th)] = {
            "positives": pos,
            "negatives": neg,
            "positive_rate": pos / len(df) if len(df) > 0 else 0.0,
        }

    stats = {
        "total_patches": len(df),
        "patch_shape": df.iloc[0]["image_shape"] if len(df) > 0 else None,
        "image_dtype": df.iloc[0]["image_dtype"] if len(df) > 0 else None,
        "mask_dtype": df.iloc[0]["mask_dtype"] if len(df) > 0 else None,
        "nan_patches": int(df["has_nan"].sum()) if len(df) > 0 else 0,
        "inf_patches": int(df["has_inf"].sum()) if len(df) > 0 else 0,
        "threshold_evaluations": class_counts,
    }

    return df, stats


def plot_sample_patch_grids(
    data_dir: str = "data/raw/landslide4sense",
    output_png: str = "results/figures/landslide4sense_sample_grid.png",
    num_samples: int = 4,
) -> Path:
    """Generates an annotated multi-spectral visualization figure displaying RGB, NIR, DEM, Slope, and Mask."""
    img_dir = Path(data_dir) / "images" / "train"
    mask_dir = Path(data_dir) / "annotations" / "train"

    # Find patches that have both positive landslides and clear terrain
    df, _ = inspect_landslide4sense_dataset(data_dir, thresholds=[0.01])
    pos_patches = df[df["label_th_0.01"] == 1]["patch_id"].tolist()
    neg_patches = df[df["label_th_0.01"] == 0]["patch_id"].tolist()

    selected = pos_patches[:num_samples//2 + 1] + neg_patches[:num_samples//2]
    selected = selected[:num_samples]

    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3.2 * num_samples), dpi=300)
    col_titles = ["Natural Color (RGB: B4,B3,B2)", "False Color (NIR: B8,B4,B3)", "Elevation (DEM: B13)", "Slope Angle (B14)", "Ground Truth Landslide Mask"]

    for row_idx, pid in enumerate(selected):
        img_p = img_dir / f"image_{pid}.h5"
        mask_p = mask_dir / f"mask_{pid}.h5"

        with h5py.File(img_p, "r") as f_i, h5py.File(mask_p, "r") as f_m:
            img = f_i["img"][:]
            mask = f_m["mask"][:]

        # Sentinel-2 bands in L4S: B1(0), B2(1), B3(2), B4(3), B5(4), B6(5), B7(6), B8(7), B8A(8), B9(9), B11(10), B12(11), DEM(12), Slope(13)
        # RGB composite: B4 (Red: idx 3), B3 (Green: idx 2), B2 (Blue: idx 1)
        rgb = np.stack([img[:, :, 3], img[:, :, 2], img[:, :, 1]], axis=-1)
        rgb_norm = np.clip((rgb - np.percentile(rgb, 2)) / (np.percentile(rgb, 98) - np.percentile(rgb, 2) + 1e-6), 0, 1)

        # False-color NIR composite: B8 (NIR: idx 7), B4 (Red: idx 3), B3 (Green: idx 2)
        nir = np.stack([img[:, :, 7], img[:, :, 3], img[:, :, 2]], axis=-1)
        nir_norm = np.clip((nir - np.percentile(nir, 2)) / (np.percentile(nir, 98) - np.percentile(nir, 2) + 1e-6), 0, 1)

        dem = img[:, :, 12]
        slope = img[:, :, 13]

        axes[row_idx, 0].imshow(rgb_norm)
        axes[row_idx, 1].imshow(nir_norm)
        im_dem = axes[row_idx, 2].imshow(dem, cmap="terrain")
        plt.colorbar(im_dem, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        im_slope = axes[row_idx, 3].imshow(slope, cmap="magma")
        plt.colorbar(im_slope, ax=axes[row_idx, 3], fraction=0.046, pad=0.04)

        axes[row_idx, 4].imshow(rgb_norm)
        # Overlay mask with alpha
        mask_overlay = np.ma.masked_where(mask == 0, mask)
        axes[row_idx, 4].imshow(mask_overlay, cmap="autumn", alpha=0.65, interpolation="none")

        ratio = (mask > 0).sum() / mask.size
        axes[row_idx, 0].set_ylabel(f"Patch #{pid}\n({ratio:.1%} Landslide)", fontsize=9, fontweight="bold")

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=10, fontweight="bold", pad=8)

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Landslide4Sense Multi-Spectral Patch Verification & Ground Truth Masks", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved sample patch grid figure to %s", out_path)
    return out_path


def plot_study_region_events(
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    output_png: str = "results/figures/study_region_events.png",
    bbox: Tuple[float, float, float, float] = (73.0, 8.0, 77.8, 20.5),
) -> Path:
    """Generates a static spatial scatter map showing positive landslide events and negative control samples."""
    fig, ax = plt.subplots(figsize=(8, 12), dpi=300)

    min_lon, min_lat, max_lon, max_lat = bbox

    # Background grid styling
    ax.set_facecolor("#f4f6f8")
    ax.grid(color="#d0d7de", linestyle="--", linewidth=0.6, alpha=0.7)

    # Plot negative non-landslide background controls
    ax.scatter(
        neg_df["longitude"], neg_df["latitude"],
        c="#3498db", marker="o", s=22, alpha=0.6, edgecolors="none",
        label=f"Negative Non-Landslide Controls (N={len(neg_df)})"
    )

    # Plot positive landslide events categorized by trigger/type
    ax.scatter(
        pos_df["longitude"], pos_df["latitude"],
        c="#e74c3c", marker="^", s=45, alpha=0.85, edgecolors="#7b1113", linewidth=0.5,
        label=f"NASA GLC Landslide Events (N={len(pos_df)})"
    )

    ax.set_xlim(min_lon - 0.2, max_lon + 0.2)
    ax.set_ylim(min_lat - 0.2, max_lat + 0.2)
    ax.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Study Region: Western Ghats Landslide Corridor, India\n"
        f"Positive Events (2007–2016, High-Accuracy) vs. 5km-Buffered Controls",
        fontsize=12, fontweight="bold", pad=12
    )

    # Contextual annotations for major hill ranges
    annotations = [
        ("Nilgiris & Wayanad", 76.5, 11.5),
        ("Idukki & Munnar", 77.0, 9.9),
        ("Coorg / Kodagu", 75.8, 12.4),
        ("Mahabaleshwar / Konkan", 73.6, 17.9),
    ]
    for text, lon, lat in annotations:
        ax.annotate(
            text, xy=(lon, lat), xytext=(lon + 0.4, lat + 0.1),
            fontsize=8, fontweight="bold", color="#2c3e50",
            arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=0.8)
        )

    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    plt.tight_layout()

    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved study region events scatter map to %s", out_path)
    return out_path
