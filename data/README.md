# Landslide Pipeline Data Directory & Provenance Catalog

This directory stores datasets for Detection (Stage A), Susceptibility/Risk Prediction (Stage B), and Early Warning (Stage C).

---

## Directory Organization

```
data/
├── README.md                      # Provenance, variables, resolution, and inspection counts
├── raw/                           # Original, unmodified datasets
│   ├── Global_Landslide_Catalog_Export.csv # NASA GLC global landslide inventory (11,033 events)
│   ├── landslide4sense/           # Landslide4Sense Sentinel-2 multi-spectral h5 patches & masks
│   │   ├── images/train/          # Image tensors (128, 128, 14) [float64]
│   │   └── annotations/train/     # Binary pixel masks (128, 128) [uint8]
│   └── Bijie_landslide_dataset.zip# Wuhan University GPCV Bijie TripleSat optical dataset (525 MB)
├── interim/                       # Cleaned, harmonized intermediate structures
└── processed/                     # Modeling-ready manifests and feature tables
    ├── landslide4sense_manifest.csv # Verified patch manifest with multi-threshold labels
    └── stage_b_events.csv         # Western Ghats positive events & 5km-buffered controls (504 rows)
```

---

## Data Sources & Inspection Specifications

### 1. Stage A: Landslide4Sense (L4S)
- **Source**: IBM-NASA Geospatial / IARAI Landslide4Sense Challenge 2022
- **Source URL**: https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Download Date**: 2026-10-19
- **Spatial Resolution**: ~10m ground resolution, $128 \times 128$ pixel spatial dimensions
- **Tensor Shape**: `(128, 128, 14)` for images; `(128, 128)` for ground truth masks
- **Data Types**: Image = `float64`, Mask = `uint8`
- **Integrity**: 0 NaN, 0 Inf values detected across all verified patches
- **14 Channels**:
  1. `B01`: Coastal Aerosol (443 nm)
  2. `B02`: Blue (490 nm)
  3. `B03`: Green (560 nm)
  4. `B04`: Red (665 nm)
  5. `B05`: Vegetation Red Edge 1 (705 nm)
  6. `B06`: Vegetation Red Edge 2 (740 nm)
  7. `B07`: Vegetation Red Edge 3 (783 nm)
  8. `B08`: NIR Broad (842 nm)
  9. `B8A`: Narrow NIR (865 nm)
  10. `B09`: Water Vapour (945 nm)
  11. `B11`: SWIR 1 (1610 nm)
  12. `B12`: SWIR 2 (2190 nm)
  13. `B13`: ALOS PALSAR Digital Elevation Model (DEM)
  14. `B14`: ALOS PALSAR Slope Angle
- **Patch Binary Classification Thresholding**:
  $$\text{Label} = 1 \iff \frac{\text{Landslide Pixels}}{\text{Total Pixels}} \ge \tau$$
  - Threshold $\tau = 0.001$ (0.1% pixels): 63 Positives (63.0%), 37 Negatives (37.0%)
  - Threshold $\tau = 0.010$ (1.0% pixels, **Default**): 40 Positives (40.0%), 60 Negatives (60.0%)
  - Threshold $\tau = 0.050$ (5.0% pixels): 13 Positives (13.0%), 87 Negatives (87.0%)

### 2. Stage A Cross-Dataset: Bijie Landslide Dataset
- **Source**: Wuhan University Group of Photogrammetry and Computer Vision (GPCV)
- **Source URL**: http://gpcv.whu.edu.cn/data/Bijie_pages.html
- **Citation**: Ji et al., "Landslide detection from an open satellite imagery and digital elevation model dataset using attention boosted convolutional neural networks", *Landslides*, 2020.
- **License**: Research Open Access
- **Download Date**: 2026-10-19
- **Content**: High-resolution optical TripleSat images (770 landslide instances, 2,003 non-landslide instances) + DEM
- **Intended Use**: Out-of-distribution and cross-sensor evaluation between Sentinel-2 and high-resolution aerial optical sensors.

### 3. Stage B & C: NASA Global Landslide Catalog (GLC) & Study Region
- **Source**: NASA Earth Science Open Data Portal (Legacy Export)
- **Source URL**: https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/Global_Landslide_Catalog_Export_rows.csv
- **License**: NASA Open Data Policy (CC0 / Public Domain)
- **Download Date**: 2026-10-19
- **Total Catalog Size**: 11,033 global events (1988–present)
- **Study Region**: **Western Ghats Landslide Corridor, India**
  - Bounding Box: $[73.0^\circ\text{E}, 8.0^\circ\text{N}, 77.8^\circ\text{E}, 20.5^\circ\text{N}]$
  - Extends across Kerala (Idukki, Wayanad), Tamil Nadu (Nilgiris), Karnataka (Coorg/Kodagu), Goa, and Maharashtra (Konkan, Mahabaleshwar).
- **Location Accuracy Quality Control**:
  - Events inside bounding box: **219**
  - **Retained High-Accuracy Events**: **168 events** (location accuracy = `exact`, `1km`, `5km`, or `10km`)
  - **Dropped Low-Accuracy Events**: **51 events** (22 with 25 km uncertainty, 12 with 50 km uncertainty, 17 unknown)
- **Temporal Distribution of Positive Events**:
  - 2007: 5 | 2008: 5 | 2009: 9 | 2010: 19 | 2011: 18 | 2012: 11 | 2013: 25 | 2014: 8 | 2015: 19 | 2016: 25 | 2017: 24

### 4. Stage B Negative Sampling Protocol
- **Methodology**: Spatial rejection sampling outside circular exclusion zones.
- **Buffer Radius**: $5.0\text{ km}$ ($0.045^\circ$) around all known historical landslide points.
- **Sampling Ratio**: $2.0$ negative control locations per positive event ($N = 336$ controls).
- **Temporal Alignment**: Sampled negative points are matched to historical observation dates and seasons to preserve monsoonal precipitation baselines.
- **Sensitivity Validation**: Exclusion distances are verified programmatically in unit tests (`min_distance >= 5.0 km`).

---

## Publication Figures Generated
- `results/figures/landslide4sense_sample_grid.png`: 5-column inspection figure showing Natural Color (RGB), False Color (NIR), DEM, Slope, and Ground Truth Landslide Mask overlay across positive and negative patches.
- `results/figures/study_region_events.png`: Cartographic scatter map showing the Western Ghats mountain spine, positive landslide clusters (Idukki, Nilgiris, Coorg, Konkan), and 5km-buffered background control points.
