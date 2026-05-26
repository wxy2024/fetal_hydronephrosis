# 这个脚本负责提取 FetalCLIP 特征并保存。
# 保存后，每个单侧样本会对应一个 .pt 文件，例如：
# outputs/features/case_0001_left.pt
# outputs/features/case_0001_right.pt
# 每个 .pt 文件包含：
# {
#     "sample_id": ...,
#     "case_id": ...,
#     "side": ...,
#     "label": ...,
#     "visit_features": Tensor[T, D_visit],
#     "visit_ids": [...],
# }
# scripts/02_extract_features.py
# 执行 python scripts/02_extract_features.py


import sys
from pathlib import Path

import torch

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
    此时返回全零图像特征，并设置 image_mask=0。
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

        for visit_idx, visit in enumerate(sample["visits"]):
            print(f"  - visit {visit_idx + 1}/{sample['num_visits']}: {visit['visit_id']}")
            print(f"    num_images: {visit['num_images']}")

            # 1. 文本特征
            text_feat = extractor.encode_text(visit["report_text"])
            text_dim = text_feat.shape[0]

            # 2. 图像特征
            image_features = extractor.encode_images(visit["image_paths"])
            img_feat, image_mask = aggregate_image_features(image_features, text_dim=text_dim)

            # 3. 时间和阶段特征
            phase_feat = torch.tensor([float(visit["phase_flag"])], dtype=torch.float32)

            # 简单归一化：孕龄/存在天数除以 300，避免数值过大，不准确，计划弃用
            age_feat = torch.tensor(
                [float(visit["gestational_age_days"]) / 300.0],
                dtype=torch.float32
            )


            # 4. 检查级特征拼接
            # visit_feature = [图像特征, 文本特征, image_mask, phase, gestational_age_days]
            visit_feat = torch.cat(
                [
                    img_feat,
                    text_feat,
                    image_mask,
                    phase_feat,
                    age_feat,
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