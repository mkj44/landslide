"""Stage A Multi-Spectral Landslide Detection: Comprehensive Training & Benchmark Script.

Executes and evaluates:
  1. Baselines: Patch Random Forest & XGBoost (Tabular Summary Features)
  2. Deep Learning: Scratch Landslide CNN (14 Channels)
  3. Transfer Learning: Adapted ResNet-50 (14 Channels, Two-Stage Training)
  4. Transfer Learning: Adapted EfficientNet-B0 (14 Channels, Two-Stage Training)
  5. Vision-Language: CLIP ViT-B/32 Zero-Shot (4 prompt templates, RGB-only)
  6. Vision-Language: CLIP ViT-B/32 Linear Probe (Frozen Embeddings, RGB-only)

Evaluated across seeds [42, 123, 999] with Mean +/- Std reporting.
Outputs:
  - results/tables/stage_a_validation_table.csv
  - results/figures/detection_training_curves.png
  - artifacts/checkpoints/*.pt
  - artifacts/stage_a_results.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.utils.logging import get_logger
from src.utils.seed import seed_everything
from src.utils.config import load_config
from src.data.dataset import LandslidePatchDataset
from src.models.baseline import extract_patch_tabular_features, PatchRandomForest, PatchXGBoost
from src.models.cnn import ScratchLandslideCNN
from src.models.transfer import AdaptedResNet50, AdaptedEfficientNetB0
from src.models.clip_model import LandslideCLIP, CLIP_PROMPTS
from src.training.train import train_patch_classifier
from src.evaluation.metrics import compute_classification_metrics

logger = get_logger("train_detection")

# Apply modern clean style for static publication figures
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


def run_detection_benchmark(
    config_path: str = "configs/detection.yaml",
    splits_path: str = "artifacts/splits/stage_a_splits.json",
    output_table_path: str = "results/tables/stage_a_validation_table.csv",
    output_figure_path: str = "results/figures/detection_training_curves.png",
    output_json_path: str = "artifacts/stage_a_results.json",
):
    """Executes the complete Stage A benchmark across all models and seeds."""
    cfg = load_config(config_path)
    seeds = cfg.get("training", {}).get("seeds", [42, 123, 999])
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    max_epochs = cfg.get("training", {}).get("max_epochs", 25)
    lr = cfg.get("training", {}).get("learning_rate", 3e-4)
    weight_decay = cfg.get("training", {}).get("weight_decay", 1e-2)
    patience = cfg.get("training", {}).get("early_stopping_patience", 6)
    grad_clip = cfg.get("training", {}).get("gradient_clip", 1.0)
    loss_strategy = cfg.get("imbalance", {}).get("strategy", "focal_loss")
    aug_params = cfg.get("augmentation", {})

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_ids = splits["train"]
    val_ids = splits["val"]
    logger.info("Loaded Stage A splits: %d train patches, %d val patches", len(train_ids), len(val_ids))

    device = torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cpu")
    logger.info("Compute hardware accelerated device: %s", device)

    Path("artifacts/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    all_results = {}
    model_curves = {
        "ScratchLandslideCNN": [],
        "AdaptedResNet50": [],
        "AdaptedEfficientNetB0": [],
    }

    # =========================================================================
    # 1. TABULAR BASELINES: Random Forest & XGBoost
    # =========================================================================
    logger.info("--- Extracting tabular features for Random Forest & XGBoost ---")
    X_train, y_train, feat_names = extract_patch_tabular_features(train_ids)
    X_val, y_val, _ = extract_patch_tabular_features(val_ids)
    logger.info("Tabular feature matrix shape: Train %s, Val %s (%d features)",
                X_train.shape, X_val.shape, len(feat_names))

    rf_metrics_list = []
    xgb_metrics_list = []

    for s in seeds:
        # Random Forest
        rf = PatchRandomForest(n_estimators=150, max_depth=10, random_state=s)
        rf.fit(X_train, y_train)
        rf_probs = rf.predict_proba(X_val)
        rf_m = compute_classification_metrics(y_val, rf_probs)
        rf_metrics_list.append(rf_m)

        # XGBoost
        xgb_model = PatchXGBoost(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=s)
        xgb_model.fit(X_train, y_train)
        xgb_probs = xgb_model.predict_proba(X_val)
        xgb_m = compute_classification_metrics(y_val, xgb_probs)
        xgb_metrics_list.append(xgb_m)

    all_results["Baseline_RandomForest"] = rf_metrics_list
    all_results["Baseline_XGBoost"] = xgb_metrics_list
    logger.info("Baselines evaluation complete.")

    # =========================================================================
    # 2. DEEP LEARNING: Scratch CNN, ResNet-50, EfficientNet-B0
    # =========================================================================
    dl_models = [
        ("ScratchLandslideCNN", lambda: ScratchLandslideCNN(in_channels=14, base_channels=32, dropout=0.2), 0),
        ("AdaptedResNet50", lambda: AdaptedResNet50(in_channels=14, pretrained=True, dropout=0.3), 3),
        ("AdaptedEfficientNetB0", lambda: AdaptedEfficientNetB0(in_channels=14, pretrained=True, dropout=0.3), 3),
    ]

    for model_name, model_fn, freeze_ep in dl_models:
        logger.info("--- Training Model: %s across seeds %s ---", model_name, seeds)
        seed_metrics = []

        for s in seeds:
            seed_everything(s)

            train_dataset = LandslidePatchDataset(
                train_ids,
                band_subset="all14",
                norm_stats_file="artifacts/norm_stats.json",
                is_training=True,
                augmentation_params=aug_params,
            )
            val_dataset = LandslidePatchDataset(
                val_ids,
                band_subset="all14",
                norm_stats_file="artifacts/norm_stats.json",
                is_training=False,
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = model_fn()
            ckpt_path = f"artifacts/checkpoints/{model_name.lower()}_seed{s}.pt"

            history, best_metrics, probs, tgts = train_patch_classifier(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=max_epochs,
                learning_rate=lr,
                weight_decay=weight_decay,
                loss_strategy=loss_strategy,
                early_stopping_patience=patience,
                gradient_clip=grad_clip,
                freeze_epochs=freeze_ep,
                checkpoint_path=ckpt_path,
                device=device,
            )

            seed_metrics.append(best_metrics)
            if s == seeds[0]:
                model_curves[model_name].append(history)

            logger.info("[%s - Seed %d] Best Val F1: %.4f | Acc: %.4f | PR-AUC: %.4f | ROC-AUC: %.4f",
                        model_name, s, best_metrics["f1"], best_metrics["accuracy"],
                        best_metrics["pr_auc"], best_metrics["roc_auc"])

        all_results[model_name] = seed_metrics

    # =========================================================================
    # 3. VISION-LANGUAGE: OpenAI CLIP ViT-B/32 (RGB-only)
    # =========================================================================
    logger.info("--- Evaluating OpenAI CLIP ViT-B/32 (Zero-Shot & Linear Probe) ---")
    clip_evaluator = LandslideCLIP(device=device)

    # Convert patch images to RGB PIL Images
    import h5py
    train_imgs_pil = []
    val_imgs_pil = []
    l4s_base = Path("data/raw/landslide4sense/images/train")

    for pid in train_ids:
        with h5py.File(l4s_base / f"image_{pid}.h5", "r") as f:
            train_imgs_pil.append(clip_evaluator.patch_to_rgb(f["img"][:]))

    for pid in val_ids:
        with h5py.File(l4s_base / f"image_{pid}.h5", "r") as f:
            val_imgs_pil.append(clip_evaluator.patch_to_rgb(f["img"][:]))

    # Zero-Shot Evaluation
    zero_shot_results = clip_evaluator.zero_shot_evaluation(val_imgs_pil, y_val)
    # Pick template with highest validation F1
    best_template = max(zero_shot_results.items(), key=lambda x: x[1]["f1"])
    logger.info("Best CLIP Zero-Shot Prompt Template: '%s' (F1=%.4f, Acc=%.4f)",
                best_template[0], best_template[1]["f1"], best_template[1]["accuracy"])
    all_results["CLIP_ViT_B32_ZeroShot"] = [best_template[1]] * len(seeds)

    # Linear Probe Evaluation across seeds
    clip_probe_metrics = []
    for s in seeds:
        clf, probe_m, _ = clip_evaluator.fit_linear_probe(
            train_imgs_pil, y_train, val_imgs_pil, y_val, C=1.0
        )
        clip_probe_metrics.append(probe_m)
    all_results["CLIP_ViT_B32_LinearProbe"] = clip_probe_metrics

    # =========================================================================
    # 4. COMPOSE VALIDATION TABLE (Mean +/- Std)
    # =========================================================================
    table_rows = []
    metric_keys = [
        "accuracy", "precision", "recall", "f1",
        "macro_f1", "balanced_accuracy", "mcc", "roc_auc", "pr_auc", "brier_score"
    ]

    for model_name, metrics_list in all_results.items():
        row = {"Model": model_name}
        for k in metric_keys:
            vals = [m[k] for m in metrics_list]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            row[f"{k}_mean"] = round(mean_val, 4)
            row[f"{k}_std"] = round(std_val, 4)
            row[k] = f"{mean_val:.4f} ± {std_val:.4f}"
        table_rows.append(row)

    df_table = pd.DataFrame(table_rows)
    # Reorder columns: Model, F1, Accuracy, Precision, Recall, PR-AUC, ROC-AUC, Balanced Acc, MCC, Brier Score
    display_cols = ["Model", "f1", "accuracy", "precision", "recall", "pr_auc", "roc_auc", "balanced_accuracy", "mcc", "brier_score"]
    df_display = df_table[display_cols].copy()
    df_display.columns = ["Model", "F1 Score", "Accuracy", "Precision", "Recall", "PR-AUC", "ROC-AUC", "Balanced Acc", "MCC", "Brier Score"]
    df_display.to_csv(output_table_path, index=False)
    logger.info("Saved Stage A validation table to %s", output_table_path)

    # Also save raw JSON results
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved raw results to %s", output_json_path)

    # =========================================================================
    # 5. GENERATE TRAINING CURVES FIGURE
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Stage A Multi-Spectral Detection Models: Training & Validation Dynamics", fontsize=16, fontweight="bold", y=0.98)

    palette = {"ScratchLandslideCNN": "#2b5c8f", "AdaptedResNet50": "#d95f02", "AdaptedEfficientNetB0": "#2ca02c"}

    for name, histories in model_curves.items():
        if not histories:
            continue
        hist = histories[0]
        epochs = hist["epoch"]
        c = palette.get(name, "#333333")

        # 1. Train Loss
        axes[0, 0].plot(epochs, hist["train_loss"], label=f"{name}", color=c, lw=2.2)
        # 2. Val Loss
        axes[0, 1].plot(epochs, hist["val_loss"], label=f"{name}", color=c, lw=2.2)
        # 3. Val F1
        axes[1, 0].plot(epochs, hist["val_f1"], label=f"{name}", color=c, lw=2.2, marker="o", markersize=4)
        # 4. Val PR-AUC
        axes[1, 1].plot(epochs, hist["val_pr_auc"], label=f"{name}", color=c, lw=2.2, marker="s", markersize=4)

    axes[0, 0].set_title("A: Training Loss (Focal Loss)", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(loc="upper right", frameon=True)

    axes[0, 1].set_title("B: Validation Loss", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend(loc="upper right", frameon=True)

    axes[1, 0].set_title("C: Validation F1 Score", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].legend(loc="lower right", frameon=True)

    axes[1, 1].set_title("D: Validation PR-AUC (Average Precision)", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("PR-AUC")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].legend(loc="lower right", frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_figure_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Stage A training curves figure to %s", output_figure_path)

    print("\n" + "=" * 95)
    print("STAGE A DETECTION VALIDATION RESULTS TABLE (Mean ± Std over 3 seeds)")
    print("=" * 95)
    print(df_display.to_string(index=False))
    print("=" * 95 + "\n")

    return df_display


if __name__ == "__main__":
    run_detection_benchmark()
