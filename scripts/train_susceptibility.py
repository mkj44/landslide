"""Stage B Susceptibility and Short-Term Risk Modeling: Comprehensive Cross-Validation & Benchmark Script.

Executes and evaluates:
  1. 5-Fold Spatial Block Cross-Validation (with SMOTE strictly inside training folds)
  2. Model family comparison:
     - Standardized L2 Logistic Regression
     - Balanced Random Forest
     - LightGBM Gradient Booster
     - Tabular PyTorch MLP (with BatchNorm, Dropout, Weight Decay)
  3. Probability Calibration (Isotonic Regression) & Expected Calibration Error (ECE)
  4. SHAP Interpretability: Feature Importance, Trigger Inflection & Interaction Analysis
  5. Temporal Holdout Evaluation (Historical <=2014 train vs 2015-2017 test)
  6. Visualizations:
     - results/figures/susceptibility_spatial_cv_roc_pr.png
     - results/figures/susceptibility_calibration_curves.png
     - results/figures/susceptibility_shap_summary.png
  7. Tables:
     - results/tables/stage_b_spatial_cv_table.csv
     - results/tables/stage_b_temporal_table.csv
  8. Serialized model:
     - artifacts/checkpoints/best_susceptibility_model.joblib
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

from src.utils.logging import get_logger
from src.utils.config import load_config
from src.evaluation.metrics import compute_classification_metrics
from src.training.calibration import (
    compute_expected_calibration_error,
    calibrate_probabilities,
    plot_reliability_diagram,
)
from src.models.susceptibility import (
    prepare_stage_b_features,
    SusceptibilityLogisticRegression,
    SusceptibilityRandomForest,
    SusceptibilityLightGBM,
    SusceptibilityMLP,
    generate_shap_interpretability_plots,
)

logger = get_logger("train_susceptibility")

# Publication plot styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def run_susceptibility_benchmark(
    features_csv: str = "data/processed/stage_b_features.csv",
    spatial_splits_json: str = "artifacts/splits/stage_b_spatial_cv_splits.json",
    temporal_split_json: str = "artifacts/splits/stage_b_temporal_split.json",
    output_cv_table: str = "results/tables/stage_b_spatial_cv_table.csv",
    output_temp_table: str = "results/tables/stage_b_temporal_table.csv",
    output_roc_pr_figure: str = "results/figures/susceptibility_spatial_cv_roc_pr.png",
    output_cal_figure: str = "results/figures/susceptibility_calibration_curves.png",
    output_shap_figure: str = "results/figures/susceptibility_shap_summary.png",
    output_model_path: str = "artifacts/checkpoints/best_susceptibility_model.joblib",
    output_json_results: str = "artifacts/stage_b_results.json",
):
    """Executes full Stage B pipeline."""
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("artifacts/checkpoints").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    X, y, feature_names = prepare_stage_b_features(df)
    logger.info("Loaded Stage B feature matrix: %d samples, %d features", X.shape[0], X.shape[1])

    with open(spatial_splits_json, "r", encoding="utf-8") as f:
        spatial_splits = json.load(f)

    with open(temporal_split_json, "r", encoding="utf-8") as f:
        temporal_splits = json.load(f)

    model_factories = {
        "LogisticRegression": lambda: SusceptibilityLogisticRegression(C=1.0, random_state=42),
        "RandomForest": lambda: SusceptibilityRandomForest(n_estimators=200, max_depth=10, random_state=42),
        "LightGBM": lambda: SusceptibilityLightGBM(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
        "PyTorch_MLP": lambda: SusceptibilityMLP(in_features=len(feature_names), max_epochs=50, random_state=42),
    }

    # =========================================================================
    # 1. 5-FOLD SPATIAL BLOCK CROSS-VALIDATION
    # =========================================================================
    logger.info("--- Starting 5-Fold Spatial Block Cross-Validation ---")
    cv_results: Dict[str, List[Dict[str, float]]] = {m: [] for m in model_factories}
    fold_predictions = {m: [] for m in model_factories}
    fold_targets = []

    best_rf_model = None
    best_rf_fold_val_data = None

    for fold_key in sorted(spatial_splits.keys()):
        fold_info = spatial_splits[fold_key]
        tr_idx = fold_info["train_indices"]
        va_idx = fold_info["val_indices"]

        X_tr_raw, y_tr_raw = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        # Apply SMOTE inside training fold only
        smote = SMOTE(k_neighbors=4, random_state=42)
        X_tr, y_tr = smote.fit_resample(X_tr_raw, y_tr_raw)

        fold_targets.append(y_va)

        for m_name, factory in model_factories.items():
            model = factory()
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_va)
            probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

            metrics = compute_classification_metrics(y_va, probs)
            metrics["ece"] = compute_expected_calibration_error(y_va, probs, n_bins=8)
            cv_results[m_name].append(metrics)
            fold_predictions[m_name].append(probs)

            if m_name == "RandomForest" and best_rf_model is None:
                best_rf_model = model
                best_rf_fold_val_data = (X_tr, y_tr, X_va, y_va)

        logger.info("Fold %s complete (Validation size: %d, Landslides: %d)",
                    fold_key, len(va_idx), int(np.sum(y_va)))

    # Compute Spatial CV Summary Table (Mean ± Std)
    cv_table_rows = []
    metric_keys = [
        "roc_auc", "pr_auc", "brier_score", "ece",
        "f1", "accuracy", "precision", "recall", "balanced_accuracy", "mcc"
    ]

    for m_name, fold_m in cv_results.items():
        row = {"Model": m_name}
        for k in metric_keys:
            vals = [m[k] for m in fold_m]
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            row[f"{k}_mean"] = round(mean_v, 4)
            row[f"{k}_std"] = round(std_v, 4)
            row[k] = f"{mean_v:.4f} ± {std_v:.4f}"
        cv_table_rows.append(row)

    df_cv = pd.DataFrame(cv_table_rows)
    display_cols = ["Model", "roc_auc", "pr_auc", "brier_score", "ece", "f1", "accuracy", "recall", "precision"]
    df_cv_display = df_cv[display_cols].copy()
    df_cv_display.columns = ["Model", "ROC-AUC", "PR-AUC", "Brier Score", "ECE", "F1 Score", "Accuracy", "Recall", "Precision"]
    df_cv_display.to_csv(output_cv_table, index=False)
    logger.info("Saved Spatial CV results to %s", output_cv_table)

    # =========================================================================
    # 2. PROBABILITY CALIBRATION (ISOTONIC REGRESSION)
    # =========================================================================
    logger.info("--- Performing Isotonic Probability Calibration ---")
    X_tr_cal, y_tr_cal, X_va_cal, y_va_cal = best_rf_fold_val_data
    uncal_probs = best_rf_model.predict_proba(X_va_cal)

    calibrator, cal_probs = calibrate_probabilities(
        model=best_rf_model.model,
        X_train=X_tr_cal,
        y_train=y_tr_cal,
        X_val=X_va_cal,
        method="isotonic",
    )

    plot_reliability_diagram(
        uncal_probs=uncal_probs,
        cal_probs=cal_probs,
        y_true=y_va_cal,
        output_path=output_cal_figure,
        model_name="Susceptibility Random Forest",
        method_name="Isotonic",
    )
    logger.info("Saved calibration curves to %s", output_cal_figure)

    # =========================================================================
    # 3. SHAP EXPLANATIONS & TRIGGER DEPENDENCE
    # =========================================================================
    logger.info("--- Generating SHAP Interpretability Analysis ---")
    generate_shap_interpretability_plots(
        model=best_rf_model,
        X_train=X_tr_cal,
        X_val=X_va_cal,
        feature_names=feature_names,
        output_summary_path=output_shap_figure,
    )

    # =========================================================================
    # 4. SPATIAL CV ROC & PR CURVES (WITH STD BANDS)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Stage B Spatial Block Cross-Validation (5 Folds): Discrimination Power", fontsize=15, fontweight="bold", y=0.98)

    palette = {"LogisticRegression": "#984ea3", "RandomForest": "#2b5c8f", "LightGBM": "#2ca02c", "PyTorch_MLP": "#d95f02"}

    # Base grid for interpolation
    mean_fpr = np.linspace(0, 1, 100)
    mean_rec = np.linspace(0, 1, 100)

    for m_name in model_factories:
        tprs = []
        precisions = []
        c = palette.get(m_name, "#333333")

        for fold_idx in range(len(spatial_splits)):
            y_fold = fold_targets[fold_idx]
            p_fold = fold_predictions[m_name][fold_idx]

            fpr, tpr, _ = roc_curve(y_fold, p_fold)
            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)

            prec, rec, _ = precision_recall_curve(y_fold, p_fold)
            prec_interp = np.interp(mean_rec, rec[::-1], prec[::-1])
            precisions.append(prec_interp)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std([m["roc_auc"] for m in cv_results[m_name]])

        axes[0].plot(mean_fpr, mean_tpr, color=c, lw=2.2, label=f"{m_name} (AUC={mean_auc:.3f}±{std_auc:.3f})")

        mean_prec = np.mean(precisions, axis=0)
        mean_pr_auc = auc(mean_rec, mean_prec)
        std_pr = np.std([m["pr_auc"] for m in cv_results[m_name]])
        axes[1].plot(mean_rec, mean_prec, color=c, lw=2.2, label=f"{m_name} (PR-AUC={mean_pr_auc:.3f}±{std_pr:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance")
    axes[0].set_title("A: Spatial Block Mean ROC Curves", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Positive class ratio for baseline chance
    base_pos = np.mean([np.mean(y_f) for y_f in fold_targets])
    axes[1].axhline(base_pos, color="k", linestyle="--", alpha=0.6, label=f"Baseline Pre-test ({base_pos:.2f})")
    axes[1].set_title("B: Spatial Block Mean Precision-Recall Curves", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_roc_pr_figure, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved Spatial CV ROC/PR curves to %s", output_roc_pr_figure)

    # =========================================================================
    # 5. TEMPORAL HOLDOUT EVALUATION (<=2014 vs 2015-2017)
    # =========================================================================
    logger.info("--- Starting Temporal Holdout Evaluation (<=2014 vs 2015-2017) ---")
    tr_temp_idx = temporal_splits["train_indices"]
    te_temp_idx = temporal_splits["test_indices"]

    X_temp_tr_raw, y_temp_tr_raw = X[tr_temp_idx], y[tr_temp_idx]
    X_temp_te, y_temp_te = X[te_temp_idx], y[te_temp_idx]

    smote_temp = SMOTE(k_neighbors=4, random_state=42)
    X_temp_tr, y_temp_tr = smote_temp.fit_resample(X_temp_tr_raw, y_temp_tr_raw)

    temp_results = {}
    temp_table_rows = []

    for m_name, factory in model_factories.items():
        model = factory()
        model.fit(X_temp_tr, y_temp_tr)
        probs = model.predict_proba(X_temp_te)
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

        m = compute_classification_metrics(y_temp_te, probs)
        m["ece"] = compute_expected_calibration_error(y_temp_te, probs, n_bins=8)
        temp_results[m_name] = m

        temp_table_rows.append({
            "Model": m_name,
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
            "Brier Score": m["brier_score"],
            "ECE": m["ece"],
            "F1 Score": m["f1"],
            "Accuracy": m["accuracy"],
            "Recall": m["recall"],
            "Precision": m["precision"],
            "Balanced Acc": m["balanced_accuracy"],
            "MCC": m["mcc"],
        })

    df_temp = pd.DataFrame(temp_table_rows)
    df_temp.to_csv(output_temp_table, index=False)
    logger.info("Saved Temporal Holdout results to %s", output_temp_table)

    # =========================================================================
    # 6. SERIALIZE FINAL PRODUCTION MODEL & RESULTS
    # =========================================================================
    # Train full model with calibration on all data for operational warning use
    smote_all = SMOTE(k_neighbors=4, random_state=42)
    X_resampled, y_resampled = smote_all.fit_resample(X, y)

    final_rf = SusceptibilityRandomForest(n_estimators=250, max_depth=10, random_state=42)
    final_rf.fit(X_resampled, y_resampled)

    final_calibrated = CalibratedClassifierCV(estimator=final_rf.model, method="isotonic", cv=3)
    final_calibrated.fit(X_resampled, y_resampled)

    save_payload = {
        "model": final_calibrated,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "calibration_method": "isotonic",
    }
    joblib.dump(save_payload, output_model_path)
    logger.info("Serialized final calibrated susceptibility model to %s", output_model_path)

    # Save summary JSON
    summary_json_payload = {
        "spatial_cv": cv_results,
        "temporal_holdout": temp_results,
        "feature_names": feature_names,
    }
    with open(output_json_results, "w", encoding="utf-8") as f:
        json.dump(summary_json_payload, f, indent=2)
    logger.info("Saved raw JSON results to %s", output_json_results)

    print("\n" + "=" * 95)
    print("STAGE B 5-FOLD SPATIAL BLOCK CROSS-VALIDATION RESULTS (Mean ± Std)")
    print("=" * 95)
    print(df_cv_display.to_string(index=False))
    print("=" * 95 + "\n")

    print("\n" + "=" * 95)
    print("STAGE B TEMPORAL HOLDOUT RESULTS (Train <=2014, Test 2015-2017)")
    print("=" * 95)
    print(df_temp.to_string(index=False))
    print("=" * 95 + "\n")

    return df_cv_display, df_temp


if __name__ == "__main__":
    run_susceptibility_benchmark()
