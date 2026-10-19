"""Cost-sensitive threshold optimization and risk level mapping for early-warning generation."""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score


DEFAULT_TIERS = {
    "Low": {"min_p": 0.0, "max_p": 0.25, "color": "#2ecc71", "action": "Routine monitoring"},
    "Moderate": {"min_p": 0.25, "max_p": 0.50, "color": "#f39c12", "action": "Advisory: heightened vigilance in steep terrain"},
    "High": {"min_p": 0.50, "max_p": 0.75, "color": "#e67e22", "action": "Warning: prepare response teams, inspect critical slopes"},
    "Critical": {"min_p": 0.75, "max_p": 1.01, "color": "#e74c3c", "action": "Emergency Alert: immediate evacuation of vulnerable slope zones"},
}


def find_cost_sensitive_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_miss_to_false_alarm_ratio: float = 10.0,
    n_thresholds: int = 100,
) -> Dict[str, Any]:
    """Finds the optimal binary classification threshold tau* minimizing expected operational loss:

      Loss(tau) = C_FP * FP(tau) + C_FN * FN(tau)
      where C_FP = 1.0 and C_FN = cost_miss_to_false_alarm_ratio.

    In disaster warning, a false negative (unwarned deadly landslide) is dramatically
    more catastrophic than a false positive (precautionary road closure / alert).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_loss = float("inf")
    best_tau = 0.5
    best_metrics = {}

    c_fp = 1.0
    c_fn = cost_miss_to_false_alarm_ratio

    for tau in thresholds:
        y_pred = (y_prob >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        total_loss = float(c_fp * fp + c_fn * fn)
        rec = float(tp / max(tp + fn, 1))
        prec = float(tp / max(tp + fp, 1))

        if total_loss < best_loss:
            best_loss = total_loss
            best_tau = float(tau)
            best_metrics = {
                "threshold": round(best_tau, 4),
                "total_cost": round(best_loss, 2),
                "recall": round(rec, 4),
                "precision": round(prec, 4),
                "fp_count": int(fp),
                "fn_count": int(fn),
                "cost_ratio": cost_miss_to_false_alarm_ratio,
            }

    return best_metrics


def compute_target_recall_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.90,
    n_thresholds: int = 100,
) -> float:
    """Finds the highest operational threshold that guarantees at least `target_recall`."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    viable_thresholds = []

    for tau in thresholds:
        y_pred = (y_prob >= tau).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec >= target_recall:
            viable_thresholds.append(tau)

    if viable_thresholds:
        return float(np.max(viable_thresholds))
    return 0.1


def map_probability_to_risk_level(prob: float, custom_tiers: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Maps a single predicted probability into a standardized 4-tier risk classification."""
    tiers = custom_tiers or DEFAULT_TIERS
    prob = float(np.clip(prob, 0.0, 1.0))

    for name, info in tiers.items():
        if info["min_p"] <= prob < info["max_p"] or (name == "Critical" and prob >= info["min_p"]):
            return {
                "risk_level": name,
                "probability": round(prob, 4),
                "color": info["color"],
                "recommended_action": info["action"],
            }

    return {
        "risk_level": "Low",
        "probability": round(prob, 4),
        "color": "#2ecc71",
        "recommended_action": "Routine monitoring",
    }
