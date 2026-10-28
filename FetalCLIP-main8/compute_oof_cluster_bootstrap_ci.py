r"""
Compute 95% confidence intervals from outer-fold OOF predictions.

Recommended for this project:
    outer OOF predictions + case_id cluster bootstrap percentile 95% CI

Example:
    python compute_oof_cluster_bootstrap_ci.py ^
        "D:\github_codes\fetal_hydronephrosis\FetalCLIP-main6\outputs_validation\...\5fold_predictions.csv"

Outputs are saved next to the input CSV:
    - 5fold_oof_case_cluster_bootstrap_95ci.csv
    - 5fold_oof_case_cluster_bootstrap_95ci_bootstrap_samples.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "Precision_PPV",
    "NPV",
    "F1",
    "AUC",
    "BrierScore",
    "ECE",
]


def auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC using average ranks for ties; avoids requiring sklearn."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(y_score, kind="mergesort")
    sorted_score = y_score[order]
    ranks_sorted = np.empty(len(y_score), dtype=float)

    i = 0
    while i < len(y_score):
        j = i + 1
        while j < len(y_score) and sorted_score[j] == sorted_score[i]:
            j += 1
        # Average 1-based rank for tied values in positions [i, j).
        ranks_sorted[i:j] = (i + 1 + j) / 2.0
        i = j

    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = ranks_sorted
    sum_pos_ranks = ranks[y_true == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_binary_metrics(
    data: pd.DataFrame,
    threshold: float = 0.5,
    label_col: str = "label",
    prob_col: str = "prob",
) -> dict[str, float | int]:
    y_true = data[label_col].to_numpy().astype(int)
    y_prob = data[prob_col].to_numpy().astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = int(len(y_true))

    brier_score = float(np.mean((y_prob - y_true) ** 2)) if n > 0 else np.nan
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)

    return {
        "Accuracy": (tp + tn) / n if n > 0 else np.nan,
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "Precision_PPV": tp / (tp + fp) if (tp + fp) > 0 else np.nan,
        "NPV": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        "F1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan,
        "AUC": auc_rank(y_true, y_prob),
        "BrierScore": brier_score,
        "ECE": ece,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "N": n,
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
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


def validate_input(
    df: pd.DataFrame,
    cluster_col: str,
    label_col: str,
    prob_col: str,
) -> None:
    required = {cluster_col, label_col, prob_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    labels = set(pd.Series(df[label_col]).dropna().astype(int).unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{label_col} must contain binary labels 0/1, got {sorted(labels)}")

    if df[cluster_col].isna().any():
        raise ValueError(f"{cluster_col} contains missing values.")
    if df[label_col].isna().any():
        raise ValueError(f"{label_col} contains missing values.")
    if df[prob_col].isna().any():
        raise ValueError(f"{prob_col} contains missing values.")


def case_cluster_bootstrap(
    df: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
    threshold: float,
    cluster_col: str,
    label_col: str,
    prob_col: str,
) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(seed)
    clusters = np.array(sorted(df[cluster_col].unique()))
    grouped = {cluster: group for cluster, group in df.groupby(cluster_col, sort=False)}

    rows = []
    skipped = 0
    for bootstrap_id in range(1, n_bootstrap + 1):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        boot_df = pd.concat(
            [grouped[cluster] for cluster in sampled_clusters],
            ignore_index=True,
        )
        metrics = compute_binary_metrics(
            boot_df,
            threshold=threshold,
            label_col=label_col,
            prob_col=prob_col,
        )
        if np.isnan(metrics["AUC"]):
            skipped += 1
            continue
        rows.append({"bootstrap_id": bootstrap_id, **metrics})

    return pd.DataFrame(rows), skipped


def summarize_ci(
    point_metrics: dict[str, float | int],
    bootstrap_df: pd.DataFrame,
    n_bootstrap_requested: int,
    n_bootstrap_used: int,
    skipped_auc: int,
    n_rows: int,
    n_clusters: int,
    threshold: float,
    seed: int,
    cluster_col: str,
) -> pd.DataFrame:
    rows = []
    for metric in DEFAULT_METRICS:
        values = pd.to_numeric(bootstrap_df[metric], errors="coerce").dropna()
        ci_low, ci_high = np.percentile(values.to_numpy(), [2.5, 97.5])
        rows.append(
            {
                "metric": metric,
                "estimate": point_metrics[metric],
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "estimate_with_95ci": (
                    f"{point_metrics[metric]:.3f} "
                    f"({ci_low:.3f}-{ci_high:.3f})"
                ),
                "method": "outer_oof_case_cluster_bootstrap_percentile",
                "cluster_col": cluster_col,
                "n_rows": n_rows,
                "n_clusters": n_clusters,
                "threshold": threshold,
                "seed": seed,
                "n_bootstrap_requested": n_bootstrap_requested,
                "n_bootstrap_used": n_bootstrap_used,
                "n_bootstrap_skipped_auc_single_class": skipped_auc,
                "TN": point_metrics["TN"],
                "FP": point_metrics["FP"],
                "FN": point_metrics["FN"],
                "TP": point_metrics["TP"],
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute percentile 95% CI from outer-fold OOF predictions using "
            "case_id cluster bootstrap."
        )
    )
    parser.add_argument("predictions_csv", type=Path, help="Path to 5fold_predictions.csv")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cluster-col", default="case_id")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--prob-col", default="prob")
    parser.add_argument(
        "--no-save-bootstrap-samples",
        action="store_true",
        help="Only save the CI summary CSV, not all bootstrap replicate metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_csv = args.predictions_csv.expanduser().resolve()
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Input file not found: {predictions_csv}")
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap should be at least 100; 10000 is recommended.")

    df = pd.read_csv(predictions_csv)
    validate_input(df, args.cluster_col, args.label_col, args.prob_col)

    point_metrics = compute_binary_metrics(
        df,
        threshold=args.threshold,
        label_col=args.label_col,
        prob_col=args.prob_col,
    )
    bootstrap_df, skipped_auc = case_cluster_bootstrap(
        df=df,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        threshold=args.threshold,
        cluster_col=args.cluster_col,
        label_col=args.label_col,
        prob_col=args.prob_col,
    )
    if bootstrap_df.empty:
        raise RuntimeError("No valid bootstrap replicates were produced.")

    ci_df = summarize_ci(
        point_metrics=point_metrics,
        bootstrap_df=bootstrap_df,
        n_bootstrap_requested=args.n_bootstrap,
        n_bootstrap_used=len(bootstrap_df),
        skipped_auc=skipped_auc,
        n_rows=len(df),
        n_clusters=df[args.cluster_col].nunique(),
        threshold=args.threshold,
        seed=args.seed,
        cluster_col=args.cluster_col,
    )

    output_dir = predictions_csv.parent
    ci_path = output_dir / "5fold_oof_case_cluster_bootstrap_95ci.csv"
    boot_path = output_dir / "5fold_oof_case_cluster_bootstrap_95ci_bootstrap_samples.csv"

    ci_df.to_csv(ci_path, index=False, encoding="utf-8-sig")
    if not args.no_save_bootstrap_samples:
        bootstrap_df.to_csv(boot_path, index=False, encoding="utf-8-sig")

    print("Saved 95% CI summary:")
    print(ci_path)
    if not args.no_save_bootstrap_samples:
        print("Saved bootstrap replicate metrics:")
        print(boot_path)
    print()
    print(ci_df[["metric", "estimate_with_95ci"]].to_string(index=False))


if __name__ == "__main__":
    main()
