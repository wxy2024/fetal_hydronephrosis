from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# 只需要修改这里：输入预测结果 CSV 路径
# CSV 中需要至少包含两列：
# label: 真实标签，0/1
# prob: 模型预测为阳性的概率，范围 0-1
# =========================
INPUT_CSV = Path(
    r"D:\github_codes\fetal_hydronephrosis\FetalCLIP-main6\output_add_metrics\results_63_dataset_123_fig_rcc_pooling_visit_direct_concat_td_conv_gru_modality_multimodal_temporal_baseline\5fold_predictions.csv"
)

# 输出目录：自动建立在输入 CSV 所在文件夹下
OUTPUT_DIR = INPUT_CSV.parent / "dca_analysis"

# 默认阈值范围：0.01 到 0.99，步长 0.01
# 如果临床上只关心某个区间，例如 0.05 到 0.60，可以改成 np.arange(0.05, 0.601, 0.01)
THRESHOLDS = np.arange(0.01, 0.991, 0.01)

# 可选临床展示范围
CLINICAL_THRESHOLD_MIN = 0.05
CLINICAL_THRESHOLD_MAX = 0.60


def setup_matplotlib():
    """设置绘图字体。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_and_check_data(input_csv):
    """读取预测结果，并检查 label 和 prob 是否满足 DCA 计算要求。"""
    if not input_csv.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = ["label", "prob"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"输入 CSV 缺少必需列: {missing_cols}")

    dca_df = df.copy()
    dca_df["label"] = pd.to_numeric(dca_df["label"], errors="coerce")
    dca_df["prob"] = pd.to_numeric(dca_df["prob"], errors="coerce")
    dca_df = dca_df.dropna(subset=["label", "prob"]).copy()

    unique_labels = sorted(dca_df["label"].unique().tolist())
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"label 列必须只包含 0/1，但当前发现: {unique_labels}")
    dca_df["label"] = dca_df["label"].astype(int)

    bad_prob = (dca_df["prob"] < 0) | (dca_df["prob"] > 1)
    if bad_prob.any():
        raise ValueError(f"prob 列应在 0 到 1 之间，但发现 {int(bad_prob.sum())} 个异常值")

    if len(dca_df) == 0:
        raise ValueError("去除缺失值后没有可用于 DCA 的样本")

    return dca_df


def calculate_pooled_dca(y_true, y_prob, thresholds):
    """计算总体 pooled DCA 曲线。

    净获益公式：
    Net Benefit = TP / N - FP / N * threshold / (1 - threshold)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    prevalence = y_true.mean()

    rows = []
    for pt in thresholds:
        # 在当前阈值下，预测概率大于等于阈值的样本被判为阳性/高风险
        pred_positive = y_prob >= pt

        tp = np.sum(pred_positive & (y_true == 1))
        fp = np.sum(pred_positive & (y_true == 0))
        tn = np.sum((~pred_positive) & (y_true == 0))
        fn = np.sum((~pred_positive) & (y_true == 1))

        # 假阳性惩罚权重：阈值越高，假阳性的代价越大
        weight = pt / (1 - pt)

        # 模型净获益
        nb_model = (tp / n) - (fp / n) * weight

        # Treat All：所有样本都判为阳性/都干预
        nb_all = prevalence - (1 - prevalence) * weight

        # Treat None：所有样本都判为阴性/都不干预，净获益恒为 0
        nb_none = 0.0

        rows.append(
            {
                "threshold": pt,
                "net_benefit_model": nb_model,
                "net_benefit_treat_all": nb_all,
                "net_benefit_treat_none": nb_none,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "n": int(n),
                "prevalence": prevalence,
            }
        )

    return pd.DataFrame(rows)


def set_auto_ylim(ax, result_df, max_threshold=0.80):
    """自动设置 y 轴范围，避免高阈值处 Treat All 过低导致主体曲线被压扁。"""
    visible = result_df[result_df["threshold"] <= max_threshold]
    y_values = pd.concat(
        [
            visible["net_benefit_model"],
            visible["net_benefit_treat_all"],
            visible["net_benefit_treat_none"],
        ]
    )
    y_min = min(-0.05, float(y_values.min()) - 0.03)
    y_max = max(0.05, float(y_values.max()) + 0.03)
    ax.set_ylim(y_min, y_max)


def plot_pooled_dca(dca_result, output_dir):
    """绘制并保存完整阈值范围的 pooled DCA 曲线。"""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    ax.plot(
        dca_result["threshold"],
        dca_result["net_benefit_model"],
        color="#1f77b4",
        linewidth=2.2,
        label="Model",
    )
    ax.plot(
        dca_result["threshold"],
        dca_result["net_benefit_treat_all"],
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label="Treat All",
    )
    ax.plot(
        dca_result["threshold"],
        dca_result["net_benefit_treat_none"],
        color="#333333",
        linestyle=":",
        linewidth=1.8,
        label="Treat None",
    )

    ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title("Pooled Decision Curve Analysis", fontsize=13)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, fontsize=11)
    ax.set_xlim(0.0, 1.0)
    set_auto_ylim(ax, dca_result, max_threshold=0.80)

    plt.tight_layout()

    png_path = output_dir / "pooled_dca_curve.png"
    pdf_path = output_dir / "pooled_dca_curve.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def plot_clinical_range_dca(dca_result, output_dir, threshold_min, threshold_max):
    """绘制并保存临床关注阈值范围内的 pooled DCA 曲线。"""
    clinical_result = dca_result[
        (dca_result["threshold"] >= threshold_min)
        & (dca_result["threshold"] <= threshold_max)
    ].copy()

    if len(clinical_result) == 0:
        raise ValueError("临床阈值范围内没有可绘制的数据，请检查阈值设置")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    ax.plot(
        clinical_result["threshold"],
        clinical_result["net_benefit_model"],
        color="#1f77b4",
        linewidth=2.2,
        label="Model",
    )
    ax.plot(
        clinical_result["threshold"],
        clinical_result["net_benefit_treat_all"],
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label="Treat All",
    )
    ax.plot(
        clinical_result["threshold"],
        clinical_result["net_benefit_treat_none"],
        color="#333333",
        linestyle=":",
        linewidth=1.8,
        label="Treat None",
    )

    ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title("Pooled DCA in Clinical Threshold Range", fontsize=13)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, fontsize=11)
    ax.set_xlim(threshold_min, threshold_max)

    y_values = pd.concat(
        [
            clinical_result["net_benefit_model"],
            clinical_result["net_benefit_treat_all"],
            clinical_result["net_benefit_treat_none"],
        ]
    )
    y_min = min(-0.05, float(y_values.min()) - 0.03)
    y_max = max(0.05, float(y_values.max()) + 0.03)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    png_path = output_dir / "pooled_dca_curve_clinical_range.png"
    pdf_path = output_dir / "pooled_dca_curve_clinical_range.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def main():
    setup_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"输入文件: {INPUT_CSV}")
    print(f"输出目录: {OUTPUT_DIR}")

    dca_df = load_and_check_data(INPUT_CSV)

    n_samples = len(dca_df)
    n_positive = int(dca_df["label"].sum())
    n_negative = int(n_samples - n_positive)
    prevalence = n_positive / n_samples

    print(f"样本数: {n_samples}")
    print(f"阳性数: {n_positive}")
    print(f"阴性数: {n_negative}")
    print(f"阳性率: {prevalence:.4f}")

    dca_result = calculate_pooled_dca(
        y_true=dca_df["label"].values,
        y_prob=dca_df["prob"].values,
        thresholds=THRESHOLDS,
    )

    csv_path = OUTPUT_DIR / "pooled_dca_curve_values.csv"
    dca_result.to_csv(csv_path, index=False, encoding="utf-8-sig")

    png_path, pdf_path = plot_pooled_dca(dca_result, OUTPUT_DIR)
    clinical_png_path, clinical_pdf_path = plot_clinical_range_dca(
        dca_result,
        OUTPUT_DIR,
        CLINICAL_THRESHOLD_MIN,
        CLINICAL_THRESHOLD_MAX,
    )

    print(f"DCA 计算结果已保存: {csv_path}")
    print(f"完整阈值范围 PNG 图片已保存: {png_path}")
    print(f"完整阈值范围 PDF 图片已保存: {pdf_path}")
    print(f"临床阈值范围 PNG 图片已保存: {clinical_png_path}")
    print(f"临床阈值范围 PDF 图片已保存: {clinical_pdf_path}")


if __name__ == "__main__":
    main()
