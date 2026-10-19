# Landslide Detection, Prediction, and Early-Warning ML Pipeline

> **IMPORTANT SCIENTIFIC FRAMING & DISCLAIMER**
>
> **Research Prototype Only — Not an Operational Safety System**:
> This repository is an academic and scientific research pipeline designed to investigate machine learning paradigms for natural hazard analysis. It is **not** an operational safety warning system and must never be used as the sole basis for life-safety, evacuation, or real-time disaster management decisions.
>
> **Physical & Temporal Distinction between Stages**:
> - **Stage A (Detection — Post-Event):** Optical and multi-spectral satellite imagery (such as Sentinel-2 or high-resolution aerial photography) captures scars and surface displacements that have **already occurred**. Detection classifies existing landslide scars from spatial texture and spectral reflectance changes.
> - **Stage B (Prediction / Susceptibility — Pre-Event & Dynamic Triggering):** Forecasting where and when landslides are likely to happen requires static geo-environmental conditioning factors (slope gradient, elevation, aspect, planform/profile curvature, lithology, vegetation density/NDVI, distance to drainage networks) coupled with a dynamic trigger—specifically, antecedent and short-term rainfall intensity and duration.
> - **Stage C (Early Warning Engine):** Evaluates calibrated probabilities and empirical rainfall threshold exceedances to issue multi-tiered risk alerts (Low, Moderate, High, Critical) with spatial persistence and cooldown logic.
>
> Stage A and Stage B address distinct physical phenomena and are rigorously evaluated on independent datasets with distinct metrics.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       Multi-Stage ML Architecture     |
                      +---------------------------------------+
                                          |
          +-------------------------------+-------------------------------+
          |                                                               |
    [ Stage A: Detection ]                                      [ Stage B: Prediction ]
  Sentinel-2 / Aerial Optical                                DEM, Curvature, NDVI, Rainfall
  - Landslide4Sense (14-band)                                - NASA Global Landslide Catalog
  - Bijie Optical (Generalization)                           - Antecedent Rainfall (1-30d)
  - Models: RF, CNN, ResNet,                                 - Models: LogReg, RF, XGBoost, MLP
    EfficientNet, Zero-Shot CLIP                             - Spatial Block CV + Temporal Split
          |                                                               |
          +-------------------------------+-------------------------------+
                                          |
                                [ Stage C: Warning ]
                          - Probability Calibration (Platt/Isotonic)
                          - Cost-Sensitive Thresholds (10:1 Miss Penalty)
                          - Rainfall Intensity-Duration (ID) Exceedance
                          - CLI, CSV/JSON Alert Logs, Static Risk Maps
```

---

## Directory Structure

```
├── configs/                  # Hyperparameter and pipeline YAML configs
│   ├── data.yaml             # Paths, study regions, band configs
│   ├── detection.yaml        # Stage A models & training parameters
│   ├── susceptibility.yaml   # Stage B tabular models & feature engineering
│   └── warning.yaml          # Stage C risk level thresholds & alert rules
├── data/
│   ├── README.md             # Dataset catalog, provenance, and citations
│   ├── raw/                  # Original downloaded archives
│   ├── interim/              # Aligned, clipped, and standardized data
│   └── processed/            # Patches, normalized tensors, tabular splits
├── src/
│   ├── data/                 # Downloaders, datasets, preprocessing, splitters
│   ├── features/             # Terrain, spectral indices, rainfall aggregations
│   ├── models/               # CNNs, transfer learning, CLIP, tabular estimators
│   ├── training/             # Loss functions, loops, callbacks, calibration
│   ├── evaluation/           # Metrics, confusion matrices, curves, Grad-CAM, SHAP
│   ├── warning/              # Decision rules, ID thresholds, alert engine, CLI
│   └── utils/                # Seed management, logging, config parsers, hardware
├── scripts/                  # Standalone execution pipelines (download, train, alert)
├── tests/                    # Unit and integration test suite (pytest)
├── artifacts/                # Model checkpoints, normalization stats, split manifests
└── results/                  # Final figures (300 DPI), tables, logs, and alert records
```

---

## Quickstart & Reproduction

### Prerequisites
- Python 3.10+ (Apple Silicon macOS or Linux x86_64/arm64)
- Git

### 1. Installation
```bash
git clone <repo-url> landslide-pipeline
cd landslide-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Environment & Tests
```bash
pytest tests/ -v
```

### 3. Run Pipeline Stages
```bash
make data             # Acquire and inspect datasets
make preprocess       # Preprocess and generate spatial CV splits
make detection        # Train Stage A detection models
make susceptibility   # Train Stage B susceptibility & risk models
make warning          # Run early-warning alert engine
make evaluate         # Generate evaluation tables, ROC/PR curves, confusion matrices
# Or run end-to-end:
make all
```
