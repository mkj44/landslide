# Landslide Pipeline Data Directory

This directory stores datasets for Detection (Stage A), Susceptibility/Risk Prediction (Stage B), and Warning (Stage C).

## Directory Structure

- `data/raw/`: Original, unmodified files downloaded from public data sources or APIs.
- `data/interim/`: Intermediate transformed, cleaned, aligned, or harmonized data.
- `data/processed/`: Modeling-ready patches, feature tables, split manifests, and spatial rasters.

---

## Data Sources & Provenance

### Stage A: Satellite & Aerial Detection

1. **Landslide4Sense (L4S)**
   - **Source URL**: https://www.iarai.ac.at/landslide4sense/ / Kaggle: `landslide4sense`
   - **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
   - **Download Date**: 2026-10-19
   - **Resolution / Dimensions**: 10m - 60m ground resolution, 128x128 pixel patches
   - **Bands / Variables (14 channels)**:
     - B1: Coastal aerosol (443 nm)
     - B2: Blue (490 nm)
     - B3: Green (560 nm)
     - B4: Red (665 nm)
     - B5: Vegetation Red Edge 1 (705 nm)
     - B6: Vegetation Red Edge 2 (740 nm)
     - B7: Vegetation Red Edge 3 (783 nm)
     - B8: NIR broad (842 nm)
     - B8A: Narrow NIR (865 nm)
     - B9: Water vapour (945 nm)
     - B11: SWIR 1 (1610 nm)
     - B12: SWIR 2 (2190 nm)
     - B13: Digital Elevation Model (DEM)
     - B14: Slope derived from DEM
   - **Mask**: Binary landslide mask (1 = landslide, 0 = background)
   - **Intended Use**: Stage A multi-spectral and optical classification; patch positive thresholding.

2. **Bijie Landslide Dataset**
   - **Source URL**: Kaggle / Open Access (Bijie City, Guizhou, China)
   - **License**: Research Open Access
   - **Download Date**: 2026-10-19
   - **Resolution**: Aerial optical imagery (RGB, sub-meter to meter resolution)
   - **Variables**: Optical RGB patches with class labels (landslide vs. non-landslide)
   - **Intended Use**: Stage A cross-dataset generalization evaluation.

---

### Stage B & C: Conditioning, Triggering & Ground Truth

1. **NASA Global Landslide Catalog (GLC)**
   - **Source URL**: https://data.nasa.gov/Earth-Science/Global-Landslide-Catalog-Export/h9d8-neg4
   - **License**: NASA Open Data Policy (Public Domain / CC0)
   - **Download Date**: 2026-10-19
   - **Variables**: `event_id`, `event_date`, `latitude`, `longitude`, `location_accuracy`, `landslide_category`, `landslide_trigger`, `fatality_count`, `injury_count`.
   - **Spatial Coverage**: Global event inventory (1988–present). Filtered to study region and high spatial accuracy (< 5 km / "exact").

2. **Digital Elevation Model (SRTM 30m / Copernicus GLO-30)**
   - **Source URL**: NASA Earthdata / OpenTopography / USGS
   - **License**: Public Domain / Copernicus Open Access
   - **Variables**: Elevation (m ASL). Derived terrain features: slope (degrees), aspect, planform/profile curvature, terrain ruggedness index (TRI).

3. **Rainfall Triggers (CHIRPS / NASA GPM IMERG)**
   - **Source URL**: Climate Hazards Center (CHIRPS) / NASA GES DISC (IMERG)
   - **License**: Public Domain / CC0
   - **Temporal Resolution**: Daily precipitation (mm/day)
   - **Variables**: Antecedent precipitation totals (1-day, 3-day, 7-day, 14-day, 30-day cumulative rainfall) and maximum daily intensity.

4. **Vegetation & Land Cover**
   - **Source URL**: ESA WorldCover 10m / Sentinel-2 L2A / MODIS NDVI
   - **License**: Creative Commons Attribution 4.0 International
   - **Variables**: NDVI (Normalized Difference Vegetation Index), land cover classification categories.

---

## Negative Sampling Strategy (Stage B)
To match positive GLC events:
1. Sample random non-landslide points strictly outside a 5 km buffer around all known historical events within the selected study region.
2. Match temporal observation windows (same years/months) and elevation distribution bins to prevent terrain or seasonal bias.
3. Sensitivity to buffer size (2 km, 5 km, 10 km) and elevation matching is evaluated during feature engineering.
