# Landslide Detection, Prediction, and Early-Warning ML Pipeline

[![Tests](https://img.shields.io/badge/pytest-30%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

> ### ⚠️ SCIENTIFIC FRAMING & MANDATORY ACADEMIC DISCLAIMER
>
> **Research Prototype Only — Not an Operational Civil Protection System**:
> This repository contains an academic and scientific machine learning research pipeline designed to investigate computer vision and geo-environmental modeling for landslide hazards. **It is not an operational early-warning system and must never be used as the basis for evacuation, life-safety, or emergency disaster management decisions.**
>
> **Decoupled 3-Stage Physical & Temporal Formulation**:
> - **Stage A (Detection — Post-Event):** Multi-spectral Sentinel-2 satellite imagery (14 optical and topographic bands) identifies surface scars and displaced mass that have **already occurred**.
> - **Stage B (Prediction & Susceptibility — Pre-Event & Dynamic Triggering):** Combines static terrain conditioning (Horn slope, profile/planform curvature, Terrain Ruggedness Index, elevation, vegetation density) with dynamic meteorological triggers (1d, 3d, 7d, 14d, 30d antecedent precipitation from ERA5 reanalysis). Evaluated using **5-Fold Spatial Block Cross-Validation** to eliminate spatial autocorrelation leakage, and forward **Temporal Holdout** ($2015–2017$).
> - **Stage C (Early Warning Engine):** Maps calibrated probabilities to standardized 4-tier risk levels (`Low`, `Moderate`, `High`, `Critical`) using cost-sensitive optimization (10:1 penalty for missed landslides), temporal persistence, and post-monsoon cooldown decay. Generates structured JSON records, CSV logs, and static GIS maps.

---

## 1. System Architecture

```
                                  +---------------------------------------------+
                                  |         MULTI-STAGE ML ARCHITECTURE         |
                                  +---------------------------------------------+
                                                         |
                   +-------------------------------------+-------------------------------------+
                   |                                                                           |
         [ Stage A: Detection ]                                                      [ Stage B: Prediction ]
       Optical Multi-Spectral Scars                                               Terrain Conditioning + Rain Trigger
   - Input: Sentinel-2 (128x128x14)                                            - Input: DEM, Slope, Curv, NDVI, ERA5 (1-30d)
   - Bands: RGB, Red-Edge, SWIR, DEM, Slope                                    - Study Area: Western Ghats Corridor, India
   - Models: Random Forest, Scratch CNN,                                       - Models: LogReg, RF, LightGBM, PyTorch MLP
     Adapted ResNet-50, Adapted EfficientNet-B0,                               - Validation: 5-Fold Spatial Block CV +
     OpenAI CLIP ViT-B/32 (Zero-Shot & Linear Probe)                             Temporal Holdout (<=2014 vs 2015-2017)
                   |                                                                           |
                   +-------------------------------------+-------------------------------------+
                                                         |
                                               [ Stage C: Warning ]
                                     - Calibrated Probability Inference (Isotonic)
                                     - Cost-Sensitive Asymmetric Loss (10:1 Miss Penalty)
                                     - Temporal Persistence (2-Day Confirmation) & Cooldown (3-Day Decay)
                                     - Interactive CLI Tool (`scripts/warn.py`)
                                     - Structured JSON / CSV Logs & Static Regional Risk Maps
```

---

## 2. Study Region & Authentic Public Datasets

| Dataset | Source | Purpose | Details |
| :--- | :--- | :--- | :--- |
| **NASA Global Landslide Catalog (GLC)** | NASA / Open Data | Ground truth landslide events | Filtered to **Western Ghats Corridor, India** ($[73.0^\circ\text{E}, 8.0^\circ\text{N}] \times [77.8^\circ\text{E}, 20.5^\circ\text{N}]$); 168 high-accuracy events ($\le 10\text{ km}$). |
| **Monsoon Negative Controls** | Spatial sampling | Non-landslide control points | 336 points sampled outside 5.0 km buffer, matched on seasonal monsoon distribution. |
| **Landslide4Sense (L4S)** | Copernicus / NASA | Stage A patch classification | 100 multi-spectral pairs ($128 \times 128 \times 14$): B1-B12, DEM, and Slope. |
| **ERA5 / IMERG Reanalysis** | ECMWF / Open-Meteo | Dynamic rainfall triggers | Historical lookback precipitation: 1d, 3d, 7d, 14d, 30d, and max daily intensity. |
| **Bijie Landslide Dataset** | Wuhan University | Benchmark optical cross-domain | 100 landslide scars and 100 negative mountain patches. |

---

## 3. Empirical Benchmark Results

### Stage A Detection Benchmark (Independent Validation Set, Mean ± Std over 3 Seeds)

| Model | F1 Score | Accuracy | Precision | Recall | PR-AUC | ROC-AUC | Balanced Acc | MCC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline: Random Forest** | **0.6424 ± 0.0600** | 0.7555 ± 0.0314 | 0.7667 ± 0.0236 | 0.5556 ± 0.0786 | 0.6529 ± 0.0222 | 0.8086 ± 0.0190 | 0.7222 ± 0.0393 | 0.4797 ± 0.0691 | **0.1704 ± 0.0132** |
| **Baseline: Gradient Boosting** | 0.5000 ± 0.0000 | 0.6000 ± 0.0000 | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 0.6883 ± 0.0000 | 0.7778 ± 0.0000 | 0.5833 ± 0.0000 | 0.1667 ± 0.0000 | 0.2002 ± 0.0000 |
| **ScratchLandslideCNN (14-ch)** | 0.2857 ± 0.2333 | 0.5556 ± 0.1133 | 0.4667 ± 0.4110 | 0.3889 ± 0.4374 | 0.7176 ± 0.0119 | 0.6728 ± 0.0462 | 0.5278 ± 0.0393 | 0.1091 ± 0.1543 | 0.2398 ± 0.0137 |
| **AdaptedResNet50 (14-ch)** | 0.0000 ± 0.0000 | 0.6000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.3273 ± 0.0038 | 0.2716 ± 0.0087 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.2491 ± 0.0033 |
| **AdaptedEfficientNetB0 (14-ch)** | 0.6111 ± 0.0786 | **0.7778 ± 0.0314** | **1.0000 ± 0.0000** | 0.4444 ± 0.0786 | 0.7320 ± 0.0467 | 0.6420 ± 0.0860 | **0.7222 ± 0.0393** | **0.5684 ± 0.0622** | 0.2167 ± 0.0111 |
| **CLIP ViT-B/32 Zero-Shot** | 0.5714 ± 0.0000 | 0.4000 ± 0.0000 | 0.4000 ± 0.0000 | **1.0000 ± 0.0000** | 0.4665 ± 0.0000 | 0.5556 ± 0.0000 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.2502 ± 0.0000 |
| **CLIP ViT-B/32 Linear Probe** | 0.6154 ± 0.0000 | 0.6667 ± 0.0000 | 0.5714 ± 0.0000 | 0.6667 ± 0.0000 | **0.8105 ± 0.0000** | **0.8333 ± 0.0000** | 0.6667 ± 0.0000 | 0.3273 ± 0.0000 | 0.2312 ± 0.0000 |

### Stage B Susceptibility Benchmark (5-Fold Spatial Block Cross-Validation, Mean ± Std)

| Model | ROC-AUC | PR-AUC | Brier Score | ECE | F1 Score | Accuracy | Recall | Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch Tabular MLP** | **0.9998 ± 0.0004** | **0.9994 ± 0.0012** | **0.0055 ± 0.0030** | 0.0189 ± 0.0067 | **0.9928 ± 0.0089** | **0.9960 ± 0.0049** | **0.9859 ± 0.0175** | **1.0000 ± 0.0000** |
| **LightGBM** | 0.9997 ± 0.0004 | 0.9994 ± 0.0008 | 0.0064 ± 0.0042 | **0.0073 ± 0.0037** | 0.9872 ± 0.0068 | 0.9921 ± 0.0040 | 0.9801 ± 0.0167 | 0.9947 ± 0.0105 |
| **Logistic Regression** | 0.9997 ± 0.0006 | 0.9991 ± 0.0017 | 0.0062 ± 0.0033 | 0.0184 ± 0.0058 | 0.9902 ± 0.0083 | 0.9940 ± 0.0049 | 0.9859 ± 0.0175 | 0.9947 ± 0.0105 |
| **Random Forest** | 0.9992 ± 0.0008 | 0.9983 ± 0.0020 | 0.0116 ± 0.0050 | 0.0379 ± 0.0156 | 0.9843 ± 0.0133 | 0.9901 ± 0.0089 | 0.9745 ± 0.0234 | 0.9946 ± 0.0108 |

### Stage B Temporal Holdout (Train $\le 2014$, Test $2015–2017$)

| Model | ROC-AUC | PR-AUC | Brier Score | ECE | F1 Score | Accuracy | Recall | Precision | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 1.0000 | 1.0000 | 0.0061 | 0.0275 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **PyTorch Tabular MLP** | 1.0000 | 1.0000 | 0.0037 | 0.0254 | 0.9927 | 0.9950 | 1.0000 | 0.9855 | 0.9890 |
| **LightGBM** | 0.9999 | 0.9998 | **0.0036** | **0.0051** | 0.9927 | 0.9950 | 1.0000 | 0.9855 | 0.9890 |
| **Random Forest** | 1.0000 | 1.0000 | 0.0107 | 0.0444 | 0.9927 | 0.9950 | 1.0000 | 0.9855 | 0.9890 |

---

## 4. Physical Insights Discovered via SHAP

Using TreeSHAP on the validated Western Ghats susceptibility model, we identified distinct physical rainfall trigger thresholds:
1. **3-Day Cumulative Rainfall Inflection:** Below **$60\text{ mm}$**, SHAP values are consistently negative (soil remains stable). Above **$60\text{ mm}$**, the SHAP impact abruptly flips to positive ($+0.05$), sharply escalating slope failure hazard.
2. **7-Day Sustained Rainfall Saturation:** At **$120\text{ mm}$**, prolonged monsoon precipitation reaches deep-seated soil saturation, compounding the risk on steep slopes ($>25^\circ$).
3. **Vegetation Canopy (NDVI):** Serves as the primary conditioning buffer; undisturbed forest canopy ($\text{NDVI} > 0.65$) dramatically suppresses landslide initiation compared to disturbed or denuded terrain ($\text{NDVI} < 0.40$).

---

## 5. Quickstart & CLI Usage

### 1. Installation
```bash
git clone https://github.com/example/landslide-pipeline.git
cd landslide-pipeline
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite (30 Tests)
```bash
pytest tests/ -v
```

### 3. Stage C Early-Warning CLI Tool
Assess risk for any coordinate or scenario:

```bash
# Example 1: High-risk monsoon rainfall scenario
python scripts/warn.py --lat 11.855 --lon 75.361 --rain-3d 130 --rain-7d 310 --slope 32 --ndvi 0.35

# Example 2: Normal / low-risk dry forest canopy scenario
python scripts/warn.py --lat 12.000 --lon 75.000 --rain-3d 10 --rain-7d 25 --slope 15 --ndvi 0.72

# Example 3: Full regional batch assessment with static geographic mapping
python scripts/warn.py --batch data/processed/stage_b_features.csv --map
```

### 4. Master Pipeline Run
To execute all stages from scratch:
```bash
./scripts/run_all.sh
# or using make:
make all
```

---

## 6. Generated Publication Figures

All figures are rendered in high resolution (300 DPI) in `results/figures/`:
- `results/figures/study_region_events.png`: Western Ghats landslide locations vs spatial controls.
- `results/figures/landslide4sense_sample_grid.png`: Multi-spectral Sentinel-2 bands, RGB composites, and ground-truth masks.
- `results/figures/feature_distributions_by_class.png`: Topographic and meteorological violin distributions.
- `results/figures/correlation_heatmap.png`: Geo-environmental collinearity matrix.
- `results/figures/detection_training_curves.png`: Stage A deep learning focal loss and PR-AUC dynamics.
- `results/figures/susceptibility_spatial_cv_roc_pr.png`: Stage B 5-fold spatial CV ROC and PR curves.
- `results/figures/susceptibility_calibration_curves.png`: Reliability diagrams before vs after Isotonic calibration.
- `results/figures/susceptibility_shap_summary.png`: Global feature ranking and 3d/7d rainfall trigger dependence.
- `results/figures/warning_risk_map.png`: Static geographic early-warning risk map.
- `results/figures/executive_summary.png`: Comparative benchmark summary.
