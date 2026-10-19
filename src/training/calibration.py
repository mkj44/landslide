"""Model calibration utilities: Platt Scaling, Isotonic Regression, and Expected Calibration Error (ECE)."""

from typing import Dict, Any, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE) across uniform confidence bins.

    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / n_total) * np.abs(bin_acc - bin_conf)

    return float(round(ece, 4))


def calibrate_probabilities(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    method: str = "isotonic",
) -> Tuple[Any, np.ndarray]:
    """Fits a calibrator on X_train/y_train and outputs calibrated probabilities for X_val."""
    calibrator = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    calibrator.fit(X_train, y_train)
    cal_probs = calibrator.predict_proba(X_val)[:, 1]
    return calibrator, cal_probs


def plot_reliability_diagram(
    uncal_probs: np.ndarray,
    cal_probs: np.ndarray,
    y_true: np.ndarray,
    output_path: str = "results/figures/susceptibility_calibration_curves.png",
    model_name: str = "Random Forest",
    method_name: str = "Isotonic",
):
    """Plots a 2-panel reliability diagram: Calibration curves and Binned probability histograms."""
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, uncal_probs, n_bins=8, strategy="uniform")
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, cal_probs, n_bins=8, strategy="uniform")

    ece_uncal = compute_expected_calibration_error(y_true, uncal_probs, n_bins=8)
    ece_cal = compute_expected_calibration_error(y_true, cal_probs, n_bins=8)

    brier_uncal = brier_score_loss(y_true, uncal_probs)
    brier_cal = brier_score_loss(y_true, cal_probs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Reliability Curve
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)
    axes[0].plot(
        prob_pred_uncal,
        prob_true_uncal,
        "s-",
        color="#d95f02",
        lw=2,
        label=f"Uncalibrated (ECE={ece_uncal:.3f}, Brier={brier_uncal:.3f})",
    )
    axes[0].plot(
        prob_pred_cal,
        prob_true_cal,
        "o-",
        color="#2b5c8f",
        lw=2.2,
        label=f"Calibrated [{method_name}] (ECE={ece_cal:.3f}, Brier={brier_cal:.3f})",
    )
    axes[0].set_title(f"A: Reliability Diagram ({model_name})", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Mean Predicted Probability", fontsize=11)
    axes[0].set_ylabel("Fraction of Positive Landslides", fontsize=11)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Predicted Probability Distributions
    axes[1].hist(uncal_probs, range=(0, 1), bins=15, histtype="step", lw=2, color="#d95f02", label="Uncalibrated")
    axes[1].hist(cal_probs, range=(0, 1), bins=15, histtype="step", lw=2, color="#2b5c8f", label="Calibrated")
    axes[1].set_title("B: Predicted Probability Histogram", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Probability", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
