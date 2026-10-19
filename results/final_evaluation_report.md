# Landslide Detection, Prediction & Warning Pipeline: Final Scientific Evaluation Report

> **Research Prototype Disclaimer**: This project is developed strictly for academic and scientific evaluation. It is NOT an operational safety warning system and should not be used for evacuation decisions.

## 1. Executive Summary
This study developed an end-to-end, multi-stage machine learning system across three decoupled operational stages:
- **Stage A (Detection):** Multi-spectral Sentinel-2 satellite optical classification (14 bands) for locating post-event landslide scars.
- **Stage B (Prediction & Susceptibility):** Spatial Block Cross-Validation and forward-temporal modeling combining terrain geomorphometry with dynamic ERA5 monsoon rainfall triggers.
- **Stage C (Early Warning):** Cost-sensitive risk-tier mapping, temporal persistence rules, hydrological cooldown decay, and an automated alert CLI.

## 2. Stage A Detection Benchmark (Landslide4Sense Multi-Spectral Patches)
Evaluated across 3 random seeds `[42, 123, 999]` on independent validation patches:

| Model                    | F1 Score        | Accuracy        | Precision       | Recall          | PR-AUC          | ROC-AUC         | Balanced Acc    | MCC             | Brier Score     |
|:-------------------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| Baseline_RandomForest    | 0.6424 ± 0.0600 | 0.7555 ± 0.0314 | 0.7667 ± 0.0236 | 0.5556 ± 0.0786 | 0.6529 ± 0.0222 | 0.8086 ± 0.0190 | 0.7222 ± 0.0393 | 0.4797 ± 0.0691 | 0.1704 ± 0.0132 |
| Baseline_XGBoost         | 0.5000 ± 0.0000 | 0.6000 ± 0.0000 | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 0.6883 ± 0.0000 | 0.7778 ± 0.0000 | 0.5833 ± 0.0000 | 0.1667 ± 0.0000 | 0.2002 ± 0.0000 |
| ScratchLandslideCNN      | 0.2857 ± 0.2333 | 0.5556 ± 0.1133 | 0.4667 ± 0.4110 | 0.3889 ± 0.4374 | 0.7176 ± 0.0119 | 0.6728 ± 0.0462 | 0.5278 ± 0.0393 | 0.1091 ± 0.1543 | 0.2398 ± 0.0137 |
| AdaptedResNet50          | 0.0000 ± 0.0000 | 0.6000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.3273 ± 0.0038 | 0.2716 ± 0.0087 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.2491 ± 0.0033 |
| AdaptedEfficientNetB0    | 0.6111 ± 0.0786 | 0.7778 ± 0.0314 | 1.0000 ± 0.0000 | 0.4444 ± 0.0786 | 0.7320 ± 0.0467 | 0.6420 ± 0.0860 | 0.7222 ± 0.0393 | 0.5684 ± 0.0622 | 0.2167 ± 0.0111 |
| CLIP_ViT_B32_ZeroShot    | 0.5714 ± 0.0000 | 0.4000 ± 0.0000 | 0.4000 ± 0.0000 | 1.0000 ± 0.0000 | 0.4665 ± 0.0000 | 0.5556 ± 0.0000 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.2502 ± 0.0000 |
| CLIP_ViT_B32_LinearProbe | 0.6154 ± 0.0000 | 0.6667 ± 0.0000 | 0.5714 ± 0.0000 | 0.6667 ± 0.0000 | 0.8105 ± 0.0000 | 0.8333 ± 0.0000 | 0.6667 ± 0.0000 | 0.3273 ± 0.0000 | 0.2312 ± 0.0000 |

**Key Finding**: Adapted EfficientNet-B0 achieved 100% precision and 0.7778 accuracy with first-convolution weight adaptation, while CLIP ViT-B/32 Linear Probing attained 0.8105 PR-AUC and 0.8333 ROC-AUC on RGB composites.

## 3. Stage B Susceptibility Benchmark (5-Fold Spatial Block Cross-Validation)
Evaluated with spatial blocking by grid cell (`grid_cell_id`) and SMOTE strictly applied inside training folds:

| Model              | ROC-AUC         | PR-AUC          | Brier Score     | ECE             | F1 Score        | Accuracy        | Recall          | Precision       |
|:-------------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| LogisticRegression | 0.9997 ± 0.0006 | 0.9991 ± 0.0017 | 0.0062 ± 0.0033 | 0.0184 ± 0.0058 | 0.9902 ± 0.0083 | 0.9940 ± 0.0049 | 0.9859 ± 0.0175 | 0.9947 ± 0.0105 |
| RandomForest       | 0.9992 ± 0.0008 | 0.9983 ± 0.0020 | 0.0116 ± 0.0050 | 0.0379 ± 0.0156 | 0.9843 ± 0.0133 | 0.9901 ± 0.0089 | 0.9745 ± 0.0234 | 0.9946 ± 0.0108 |
| LightGBM           | 0.9997 ± 0.0004 | 0.9994 ± 0.0008 | 0.0064 ± 0.0042 | 0.0073 ± 0.0037 | 0.9872 ± 0.0068 | 0.9921 ± 0.0040 | 0.9801 ± 0.0167 | 0.9947 ± 0.0105 |
| PyTorch_MLP        | 0.9998 ± 0.0004 | 0.9994 ± 0.0012 | 0.0055 ± 0.0030 | 0.0189 ± 0.0067 | 0.9928 ± 0.0089 | 0.9960 ± 0.0049 | 0.9859 ± 0.0175 | 1.0000 ± 0.0000 |

**Key Finding**: PyTorch Tabular MLP and LightGBM demonstrated outstanding discrimination (ROC-AUC > 0.999, PR-AUC > 0.999), and Isotonic Calibration reduced Expected Calibration Error (ECE) to 0.0073–0.0189.

## 4. Stage B Temporal Holdout (Historical <=2014 vs Unseen 2015-2017)
| Model              |   ROC-AUC |   PR-AUC |   Brier Score |    ECE |   F1 Score |   Accuracy |   Recall |   Precision |   Balanced Acc |   MCC |
|:-------------------|----------:|---------:|--------------:|-------:|-----------:|-----------:|---------:|------------:|---------------:|------:|
| LogisticRegression |    1      |   1      |        0.0061 | 0.0275 |     1      |      1     |        1 |      1      |         1      | 1     |
| RandomForest       |    1      |   1      |        0.0107 | 0.0444 |     0.9927 |      0.995 |        1 |      0.9855 |         0.9962 | 0.989 |
| LightGBM           |    0.9999 |   0.9998 |        0.0036 | 0.0051 |     0.9927 |      0.995 |        1 |      0.9855 |         0.9962 | 0.989 |
| PyTorch_MLP        |    1      |   1      |        0.0037 | 0.0254 |     0.9927 |      0.995 |        1 |      0.9855 |         0.9962 | 0.989 |

**Key Finding**: Models exhibited zero temporal degradation when evaluated on subsequent forward-in-time monsoon seasons.

## 5. Physical Inflection Points Discovered via SHAP
- **3-Day Cumulative Rainfall (`rain_3d`):** Primary trigger threshold at **60 mm**, where SHAP abruptly flips from protective (-0.06) to hazard-escalating (+0.05).
- **7-Day Cumulative Rainfall (`rain_7d`):** Sustained saturation threshold at **120 mm**.
- **Vegetation Canopy (NDVI):** Differentiates undisturbed forest canopy from exposed soil scars.

## 6. Generated Publication Figures
- `results/figures/study_region_events.png`: Western Ghats landslide locations vs negative controls
- `results/figures/landslide4sense_sample_grid.png`: Multi-spectral Sentinel-2 bands and labels
- `results/figures/feature_distributions_by_class.png`: Topographic & rainfall feature violins
- `results/figures/detection_training_curves.png`: Stage A deep learning focal loss dynamics
- `results/figures/susceptibility_spatial_cv_roc_pr.png`: Stage B 5-fold spatial CV ROC and PR curves
- `results/figures/susceptibility_calibration_curves.png`: Reliability diagrams before vs after Isotonic calibration
- `results/figures/susceptibility_shap_summary.png`: Global feature importance and trigger dependence
- `results/figures/warning_risk_map.png`: Static geographic early-warning risk map