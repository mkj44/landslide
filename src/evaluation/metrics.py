"""Comprehensive classification metrics computation for detection, susceptibility, and warning."""

from typing import Dict, Any, Union, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes all required evaluation metrics for binary classification.

    Metrics:
      - Accuracy
      - Precision
      - Recall (Sensitivity / POD)
      - F1 Score (Positive class)
      - Macro F1
      - Balanced Accuracy
      - Matthews Correlation Coefficient (MCC)
      - ROC-AUC
      - PR-AUC (Average Precision, crucial for imbalanced classes)
      - Brier Score (mean squared probability error)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Basic metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Matthews Correlation Coefficient
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = 0.0

    # ROC-AUC & PR-AUC
    # Check if both classes are present in y_true
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = float(np.mean(y_true))
    else:
        roc_auc = 0.5
        pr_auc = float(np.mean(y_true))

    # Brier Score
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "mcc": round(mcc, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
    }
