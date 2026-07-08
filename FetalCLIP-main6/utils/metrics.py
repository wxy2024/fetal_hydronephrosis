# utils/metrics.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) == 0:
        return np.nan

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for bin_idx in range(n_bins):
        lower = bin_edges[bin_idx]
        upper = bin_edges[bin_idx + 1]
        if bin_idx == n_bins - 1:
            in_bin = (y_prob >= lower) & (y_prob <= upper)
        else:
            in_bin = (y_prob >= lower) & (y_prob < upper)

        bin_count = int(in_bin.sum())
        if bin_count == 0:
            continue

        bin_accuracy = float(y_true[in_bin].mean())
        bin_confidence = float(y_prob[in_bin].mean())
        ece += (bin_count / len(y_true)) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5):
    """
    计算二分类指标。

    y_true: 真实标签，0/1
    y_prob: 预测概率，0~1
    """

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    brier_score = float(np.mean((y_prob - y_true) ** 2)) if len(y_true) > 0 else np.nan
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)

    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # confusion_matrix: [[TN, FP], [FN, TP]]
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # AUC 需要同时存在正负样本
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = None

    return {
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision_PPV": precision,
        "NPV": npv,
        "F1": f1,
        "AUC": auc,
        "BrierScore": brier_score,
        "ECE": ece,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
