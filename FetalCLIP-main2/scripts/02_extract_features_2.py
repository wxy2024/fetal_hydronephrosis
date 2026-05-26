#你原来的脚本是“图像特征 + 文本特征 + image_mask + 阶段/时间特征”的形式，我这里改成“文本特征 + 阶段/年龄/时间特征”，完全不再调用 encode_images，也不再拼接图像特征和 image_mask。

# 这个版本的输出维度会从原来的：
# 图像特征维度 + 文本特征维度 + image_mask + 其他结构化时间特征
# 变成：
# 文本特征维度 + 5 个额外特征
# 其中 5 个额外特征分别是：
# phase_feat
# age_feat
# age_type_feat
# days_since_first_feat
# interval_feat


import sys
from pathlib import Path
import torch
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config import (
    RAW_ROOT,
    FEATURE_DIR,
    DEVICE,
    FETALCLIP_MODEL_NAME,
    FETALCLIP_CONFIG_PATH,
    FETALCLIP_WEIGHT_PATH,
)

from data.dataset import collect_case_side_samples
from fetalclip.fetalclip_wrapper import FetalCLIPFeatureExtractor
from utils.file_utils import ensure_dir


# =========================
# 新增：单独保存文本特征，避免覆盖原来的图像+文本特征
# =========================
TEXT_ONLY_FEATURE_DIR = FEATURE_DIR.parent / f"{FEATURE_DIR.name}_text_only"


def get_phase_flag(visit: dict) -> float:
    """
    根据 visit 中的 phase 或 phase_flag 得到阶段编码。
    prenatal 记为 0；
    postnatal / postoperative 记为 1。

    这样即使原始 meta.json 中没有 phase_flag，
    只要有 phase 字段，也不会报错。
    """
    if "phase_flag" in visit:
        return float(visit["phase_flag"])

    phase = str(visit.get("phase", "")).lower()

    if phase == "prenatal":
        return 0.0
    elif phase in ["postnatal", "postoperative"]:
        return 1.0
    else:
        # 如果阶段未知，默认按产后处理，也可以改成 -1.0
        return 1.0


def get_age_features(visit: dict, phase_flag: float):
    """
    提取年龄相关特征。

    prenatal:
        使用 gestational_age_days，表示孕龄天数。

    postnatal / postoperative:
        使用 postnatal_age_days，表示出生后天数。

    返回：
        age_feat: 归一化后的年龄特征
        age_type_feat: 0 表示孕期，1 表示产后
    """
    if phase_flag == 0.0:
        age_days = float(visit.get("gestational_age_days", 0.0))
        age_type = 0.0
    else:
        age_days = float(visit.get("postnatal_age_days", 0.0))
        age_type = 1.0

    # 简单归一化，保持与你原来的代码一致
    age_feat = torch.tensor([age_days / 300.0], dtype=torch.float32)
    age_type_feat = torch.tensor([age_type], dtype=torch.float32)

    return age_feat, age_type_feat


def parse_exam_date(visit: dict):
    """
    解析检查日期。
    优先使用 visit["exam_date"]。
    如果没有 exam_date，则尝试从 visit_id 前 8 位解析 YYYYMMDD。
    """
    exam_date_str = visit.get("exam_date", None)

    if exam_date_str is not None:
        try:
            return datetime.strptime(exam_date_str, "%Y-%m-%d")
        except ValueError:
            pass

    visit_id = str(visit.get("visit_id", ""))
    if len(visit_id) >= 8 and visit_id[:8].isdigit():
        try:
            return datetime.strptime(visit_id[:8], "%Y%m%d")
        except ValueError:
            return None

    return None


def get_time_features(visit_idx: int, visits: list, first_visit_date):
    """
    提取检查时间相关特征：
    1. days_since_first_feat：距离首次检查的天数；
    2. interval_feat：距离上一次检查的间隔天数。

    返回：
        days_since_first_feat
        interval_feat
        current_exam_date
    """
    current_exam_date = parse_exam_date(visits[visit_idx])

    if current_exam_date is None:
        days_since_first = 0.0
        interval_days = 0.0
        return (
            torch.tensor([days_since_first], dtype=torch.float32),
            torch.tensor([interval_days], dtype=torch.float32),
            first_visit_date,
        )

    if first_visit_date is None:
        first_visit_date = current_exam_date
        days_since_first = 0.0
        interval_days = 0.0
    else:
        days_since_first = float((current_exam_date - first_visit_date).days)

        if visit_idx == 0:
            interval_days = 0.0
        else:
            prev_exam_date = parse_exam_date(visits[visit_idx - 1])
            if prev_exam_date is None:
                interval_days = 0.0
            else:
                interval_days = float((current_exam_date - prev_exam_date).days)

    # 归一化方式与你原来的代码保持一致
    days_since_first_feat = torch.tensor([days_since_first / 1000.0], dtype=torch.float32)
    interval_feat = torch.tensor([interval_days / 100.0], dtype=torch.float32)

    return days_since_first_feat, interval_feat, first_visit_date


def main():
    ensure_dir(TEXT_ONLY_FEATURE_DIR)

    print("=" * 80)
    print("加载 FetalCLIP 文本特征提取器")
    print("本脚本不提取图像特征，不调用 encode_images")
    print(f"config: {FETALCLIP_CONFIG_PATH}")
    print(f"weight: {FETALCLIP_WEIGHT_PATH}")
    print(f"save_dir: {TEXT_ONLY_FEATURE_DIR}")
    print("=" * 80)

    # 仍然使用 FetalCLIP 的文本编码器
    # 但是后续只调用 encode_text，不调用 encode_images
    extractor = FetalCLIPFeatureExtractor(
        model_name=FETALCLIP_MODEL_NAME,
        config_path=FETALCLIP_CONFIG_PATH,
        weight_path=FETALCLIP_WEIGHT_PATH,
        device=DEVICE,
    )

    samples = collect_case_side_samples(RAW_ROOT)
    print(f"待提取单侧样本数量: {len(samples)}")

    for sample_idx, sample in enumerate(samples):
        sample_id = sample["sample_id"]
        save_path = TEXT_ONLY_FEATURE_DIR / f"{sample_id}.pt"

        print("\n" + "=" * 80)
        print(f"[{sample_idx + 1}/{len(samples)}] 处理样本: {sample_id}")
        print(f"label: {sample['label']}, num_visits: {sample['num_visits']}")
        print("=" * 80)

        visit_features = []
        visit_ids = []

        visits = sample["visits"]
        first_visit_date = None

        for visit_idx, visit in enumerate(visits):
            visit_id = visit.get("visit_id", f"visit_{visit_idx}")

            print(f"  - visit {visit_idx + 1}/{sample['num_visits']}: {visit_id}")

            # =========================
            # 1. 文本特征
            # =========================
            report_text = visit.get("report_text", "")

            if report_text is None:
                report_text = ""

            text_feat = extractor.encode_text(report_text).float()

            # 如果 encode_text 返回 [1, D]，压缩成 [D]
            if text_feat.dim() == 2:
                text_feat = text_feat.squeeze(0)

            # =========================
            # 2. 阶段特征
            # =========================
            phase_flag = get_phase_flag(visit)
            phase_feat = torch.tensor([phase_flag], dtype=torch.float32)

            # =========================
            # 3. 孕期/产后年龄特征
            # =========================
            age_feat, age_type_feat = get_age_features(visit, phase_flag)

            # =========================
            # 4. 检查时间特征
            # =========================
            days_since_first_feat, interval_feat, first_visit_date = get_time_features(
                visit_idx=visit_idx,
                visits=visits,
                first_visit_date=first_visit_date,
            )

            # =========================
            # 5. 拼接 visit 特征
            # 不再包含：
            #   img_feat
            #   image_mask
            # =========================
            visit_feat = torch.cat(
                [
                    text_feat,
                    phase_feat,
                    age_feat,
                    age_type_feat,
                    days_since_first_feat,
                    interval_feat,
                ],
                dim=0,
            )

            visit_features.append(visit_feat)
            visit_ids.append(visit_id)

        if len(visit_features) == 0:
            print(f"[警告] 样本没有任何有效 visit，跳过: {sample_id}")
            continue

        visit_features = torch.stack(visit_features, dim=0)

        save_obj = {
            "sample_id": sample["sample_id"],
            "case_id": sample["case_id"],
            "side": sample["side"],
            "label": int(sample["label"]),
            "num_visits": sample["num_visits"],
            "visit_ids": visit_ids,

            # 核心特征
            "visit_features": visit_features,

            # 额外记录，方便后续检查维度
            "feature_type": "text_only",
            "feature_order": [
                "text_feat",
                "phase_feat",
                "age_feat",
                "age_type_feat",
                "days_since_first_feat",
                "interval_feat",
            ],
            "text_dim": int(text_feat.shape[0]),
            "extra_dim": 5,
            "total_dim": int(visit_features.shape[1]),
        }

        torch.save(save_obj, save_path)

        print(f"已保存: {save_path}")
        print(f"visit_features shape: {tuple(visit_features.shape)}")

    print("\n全部文本特征提取完成。")


if __name__ == "__main__":
    main()