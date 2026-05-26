
# 对产后检查使用 postnatal_age_days，孕期使用 gestational_age_days。
# 增加 age_type_feat 标记孕期/产后。
# 增加检查日期编码：
# days_since_first_feat：与首次检查的间隔天数
# interval_feat：与上一检查的间隔天数

# 我在代码中用 # === 修改/新增 === 注释标明了改动位置。


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


def aggregate_image_features(image_features: torch.Tensor, text_dim: int) -> tuple:
    """
    对同一次检查内的多张图像做平均池化。
    如果 image_features 为 None，说明图像为空或读取失败。
    返回全零向量并设置 image_mask=0。
    """
    if image_features is None:
        img_feat = torch.zeros(text_dim, dtype=torch.float32)
        image_mask = torch.tensor([0.0], dtype=torch.float32)
    else:
        img_feat = image_features.mean(dim=0).float()
        image_mask = torch.tensor([1.0], dtype=torch.float32)
    return img_feat, image_mask


def main():
    ensure_dir(FEATURE_DIR)

    print("=" * 80)
    print("加载 FetalCLIP 特征提取器")
    print(f"config: {FETALCLIP_CONFIG_PATH}")
    print(f"weight: {FETALCLIP_WEIGHT_PATH}")
    print("=" * 80)

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
        save_path = FEATURE_DIR / f"{sample_id}.pt"

        print("\n" + "=" * 80)
        print(f"[{sample_idx + 1}/{len(samples)}] 处理样本: {sample_id}")
        print(f"label: {sample['label']}, num_visits: {sample['num_visits']}")
        print("=" * 80)

        visit_features = []
        visit_ids = []

        first_visit_date = None  # === 新增：记录首次检查日期 ===

        for visit_idx, visit in enumerate(sample["visits"]):
            print(f"  - visit {visit_idx + 1}/{sample['num_visits']}: {visit['visit_id']}")
            print(f"    num_images: {visit['num_images']}")

            # 1. 文本特征
            text_feat = extractor.encode_text(visit["report_text"])
            text_dim = text_feat.shape[0]

            # 2. 图像特征
            image_features = extractor.encode_images(visit["image_paths"])
            img_feat, image_mask = aggregate_image_features(image_features, text_dim=text_dim)

            # 3. 阶段特征
            phase_feat = torch.tensor([float(visit["phase_flag"])], dtype=torch.float32)

            # === 修改/新增：孕期 vs 产后年龄编码 + 标记 ===
            if visit["phase_flag"] == 0:  # prenatal
                age_days = float(visit.get("gestational_age_days", 0))
                age_type = 0.0  # 0表示孕期
            else:  # postnatal
                age_days = float(visit.get("postnatal_age_days", 0))
                age_type = 1.0  # 1表示产后

            age_feat = torch.tensor([age_days / 300.0], dtype=torch.float32)  # 简单归一化
            age_type_feat = torch.tensor([age_type], dtype=torch.float32)

            # === 修改/新增：检查日期编码 ===
            exam_date_str = visit.get("exam_date", None)
            if exam_date_str is not None:
                exam_date = datetime.strptime(exam_date_str, "%Y-%m-%d")
                if first_visit_date is None:
                    first_visit_date = exam_date
                    days_since_first = 0.0
                    interval_days = 0.0
                else:
                    days_since_first = (exam_date - first_visit_date).days
                    if visit_idx == 0:
                        interval_days = 0.0
                    else:
                        prev_date_str = sample["visits"][visit_idx - 1].get("exam_date", exam_date_str)
                        prev_date = datetime.strptime(prev_date_str, "%Y-%m-%d")
                        interval_days = (exam_date - prev_date).days
            else:
                days_since_first = 0.0
                interval_days = 0.0

            days_since_first_feat = torch.tensor([days_since_first / 1000.0], dtype=torch.float32)
            interval_feat = torch.tensor([interval_days / 100.0], dtype=torch.float32)

            # 4. 拼接 visit 特征
            visit_feat = torch.cat(
                [
                    img_feat,
                    text_feat,
                    image_mask,
                    phase_feat,
                    age_feat,
                    age_type_feat,       # === 新增
                    days_since_first_feat,  # === 新增
                    interval_feat,       # === 新增
                ],
                dim=0
            )

            visit_features.append(visit_feat)
            visit_ids.append(visit["visit_id"])

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
            "visit_features": visit_features,
        }

        torch.save(save_obj, save_path)
        print(f"已保存: {save_path}")
        print(f"visit_features shape: {tuple(visit_features.shape)}")

    print("\n全部特征提取完成。")


if __name__ == "__main__":
    main()