import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config_online_attn import (
    RAW_ROOT,
    RESULT_DIR,
    DEVICE,
    FETALCLIP_MODEL_NAME,
    FETALCLIP_CONFIG_PATH,
    FETALCLIP_WEIGHT_PATH,
    SEED,
    NUM_EPOCHS,
    LR,
    WEIGHT_DECAY,
    DROPOUT,
    BATCH_SIZE,
    USE_POS_WEIGHT,
    USE_MEMORY_CACHE,
    SAVE_ATTENTION,
    ATTN_HIDDEN_DIM,
    GRU_HIDDEN_DIM,
    GRU_NUM_LAYERS,
    GRU_BIDIRECTIONAL,
    IMAGE_POOLING_TYPE,
    VISIT_FUSION_TYPE,
    FUSION_HIDDEN_DIM,

    CONTIFORMER_D_MODEL,
    CONTIFORMER_D_INNER,
    CONTIFORMER_N_LAYERS,
    CONTIFORMER_N_HEAD,
    CONTIFORMER_D_K,
    CONTIFORMER_D_V,
    CONTIFORMER_METHOD_ODE,
    CONTIFORMER_ATOL_ODE,
    CONTIFORMER_RTOL_ODE,
)

from data.dataset_online import collect_case_side_samples_online
from fetalclip.fetalclip_wrapper import FetalCLIPFeatureExtractor

##图像融合模块
#from models.attention_gru_online import AttentionGRUSideClassifier
#from models.gated_attention_mil_pooling import AttentionGRUSideClassifier
#from models.longitudinal_gru_classifier import ModularGRUSideClassifier


from models.longitudinal_contiformer_classifier import ModularContiFormerSideClassifier

from utils.metrics import compute_binary_metrics


def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RawSideDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_raw_samples(batch):
    return batch


def attach_fetalclip_features_in_memory(samples, extractor):
    """
    冻结 FetalCLIP，只在内存中提取图像/文本特征。
    不保存 outputs/features/*.pt。
    """
    print("\n" + "=" * 80)
    print("开始提取 FetalCLIP 特征到内存：不保存 .pt 文件")
    print("=" * 80)

    total_visits = sum(len(s["visits"]) for s in samples)
    done = 0

    for sample in samples:
        for visit in sample["visits"]:
            done += 1

            print(
                f"[{done}/{total_visits}] "
                f"{sample['sample_id']} | "
                f"{visit['visit_id']} | "
                f"images={visit['num_images']}"
            )

            visit["text_feature"] = extractor.encode_text(
                visit.get("report_text", "")
            )

            visit["image_features"] = extractor.encode_images(
                visit.get("image_paths", [])
            )

    print("内存特征提取完成。")


def infer_feature_dim(samples) -> int:
    for sample in samples:
        for visit in sample["visits"]:
            text_feat = visit.get("text_feature", None)

            if text_feat is not None:
                return int(text_feat.shape[0])

            image_feats = visit.get("image_features", None)

            if image_feats is not None and image_feats.numel() > 0:
                return int(image_feats.shape[1])

    raise RuntimeError(
        "无法推断 FetalCLIP 特征维度，请检查 report 或 image 是否能被读取。"
    )


def make_train_loader(train_items):
    train_dataset = RawSideDataset(train_items)

    train_labels = [int(x["label"]) for x in train_items]
    class_counts = np.bincount(train_labels, minlength=2)

    weights = []

    for label in train_labels:
        count = class_counts[label]

        if count > 0:
            weights.append(1.0 / count)
        else:
            weights.append(1.0)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(BATCH_SIZE, len(train_dataset)),
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_raw_samples,
    )

    return train_loader


def train_one_fold(
    train_items,
    test_items,
    feature_dim: int,
    fold_case_id: str,
):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    print("[Device Check] DEVICE from config:", DEVICE)
    print("[Device Check] torch.cuda.is_available():", torch.cuda.is_available())
    print("[Device Check] actual training device:", device)

    if torch.cuda.is_available():
        print("[Device Check] GPU name:", torch.cuda.get_device_name(0))

    train_loader = make_train_loader(train_items)

    # model = AttentionGRUSideClassifier(
    #     feature_dim=feature_dim,
    #     attn_hidden_dim=ATTN_HIDDEN_DIM,
    #     gru_hidden_dim=GRU_HIDDEN_DIM,
    #     gru_num_layers=GRU_NUM_LAYERS,
    #     dropout=DROPOUT,
    #     bidirectional=GRU_BIDIRECTIONAL,
    # ).to(device)
    model = ModularContiFormerSideClassifier(
        feature_dim=feature_dim,
        image_pooling_type=IMAGE_POOLING_TYPE,
        visit_fusion_type=VISIT_FUSION_TYPE,
        attn_hidden_dim=ATTN_HIDDEN_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        contiformer_d_model=CONTIFORMER_D_MODEL,
        contiformer_d_inner=CONTIFORMER_D_INNER,
        contiformer_n_layers=CONTIFORMER_N_LAYERS,
        contiformer_n_head=CONTIFORMER_N_HEAD,
        contiformer_d_k=CONTIFORMER_D_K,
        contiformer_d_v=CONTIFORMER_D_V,
        contiformer_method_ode=CONTIFORMER_METHOD_ODE,
        contiformer_atol_ode=CONTIFORMER_ATOL_ODE,
        contiformer_rtol_ode=CONTIFORMER_RTOL_ODE,
        dropout=DROPOUT,
    ).to(device)
    print("[Device Check] model device:", next(model.parameters()).device)

    if USE_POS_WEIGHT:
        train_labels = np.array([int(x["label"]) for x in train_items])
        num_pos = np.sum(train_labels == 1)
        num_neg = np.sum(train_labels == 0)

        if num_pos > 0:
            pos_weight_value = float(num_neg / max(num_pos, 1))
        else:
            pos_weight_value = 1.0

        pos_weight = torch.tensor(
            [pos_weight_value],
            dtype=torch.float32,
            device=device,
        )

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    model.train()

    for epoch in range(NUM_EPOCHS):
        losses = []

        for batch in train_loader:
            labels = torch.tensor(
                [float(x["label"]) for x in batch],
                dtype=torch.float32,
                device=device,
            )

            logits, _ = model(batch, return_attention=False)
            

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"loss={np.mean(losses):.4f}"
            )

    model.eval()

    fold_results = []

    with torch.no_grad():
        for item in test_items:
            logits, attention_batch = model(
                [item],
                return_attention=SAVE_ATTENTION,
            )

            logit_value = float(logits.item())
            prob = float(torch.sigmoid(logits).item())

            attention_json = ""

            if SAVE_ATTENTION:
                attention_json = json.dumps(
                    attention_batch[0],
                    ensure_ascii=False,
                )

            fold_results.append(
                {
                    "fold_case_id": fold_case_id,
                    "sample_id": item["sample_id"],
                    "case_id": item["case_id"],
                    "side": item["side"],
                    "label": int(item["label"]),
                    "num_visits": int(item["num_visits"]),
                    "visit_ids_sorted": "|".join(
                        [
                            str(v.get("visit_id", ""))
                            for v in item["visits"]
                        ]
                    ),
                    "prob": prob,
                    "logit": logit_value,
                    "pred": int(prob >= 0.5),
                    "image_attention_json": attention_json,
                }
            )

    return fold_results


def main():
    set_seed(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("读取 raw 数据")
    print(f"RAW_ROOT: {RAW_ROOT}")
    print("=" * 80)

    samples = collect_case_side_samples_online(RAW_ROOT)

    print(f"单侧样本数量: {len(samples)}")

    if len(samples) == 0:
        raise RuntimeError("没有读取到任何样本，请检查 RAW_ROOT 和数据结构。")

    print("\n" + "=" * 80)
    print("加载 FetalCLIP")
    print(f"config: {FETALCLIP_CONFIG_PATH}")
    print(f"weight: {FETALCLIP_WEIGHT_PATH}")
    print("=" * 80)

    extractor = FetalCLIPFeatureExtractor(
        model_name=FETALCLIP_MODEL_NAME,
        config_path=FETALCLIP_CONFIG_PATH,
        weight_path=FETALCLIP_WEIGHT_PATH,
        device=DEVICE,
    )

    if USE_MEMORY_CACHE:
        attach_fetalclip_features_in_memory(samples, extractor)
    else:
        raise NotImplementedError(
            "当前版本默认使用内存缓存。若要每个 epoch 在线编码，"
            "需要把 extractor 接入 model.forward。"
        )

    feature_dim = infer_feature_dim(samples)

    print(f"FetalCLIP feature_dim: {feature_dim}")

    case_ids = sorted(set([x["case_id"] for x in samples]))

    print(f"病例数量: {len(case_ids)}")
    print(f"病例列表: {case_ids}")

    print("\n检查前 5 个样本：")

    for item in samples[:5]:
        print(
            f"  {item['sample_id']}: "
            f"label={item['label']}, "
            f"num_visits={item['num_visits']}, "
            f"visits={[v['visit_id'] for v in item['visits']]}"
        )

    all_fold_results = []

    for fold_idx, test_case_id in enumerate(case_ids):
        print("\n" + "=" * 80)
        print(f"LOOCV Fold {fold_idx + 1}/{len(case_ids)}")
        print(f"Test case: {test_case_id}")
        print("=" * 80)

        train_items = [
            x for x in samples
            if x["case_id"] != test_case_id
        ]

        test_items = [
            x for x in samples
            if x["case_id"] == test_case_id
        ]

        train_labels = np.array([int(x["label"]) for x in train_items])

        print(f"训练样本数: {len(train_items)}")
        print(f"测试样本数: {len(test_items)}")
        print(
            f"训练集标签分布: "
            f"neg={sum(train_labels == 0)}, "
            f"pos={sum(train_labels == 1)}"
        )
        print(f"测试样本: {[x['sample_id'] for x in test_items]}")

        fold_results = train_one_fold(
            train_items=train_items,
            test_items=test_items,
            feature_dim=feature_dim,
            fold_case_id=test_case_id,
        )

        all_fold_results.extend(fold_results)

        for r in fold_results:
            print(
                f"  {r['sample_id']}: "
                f"label={r['label']}, "
                f"prob={r['prob']:.4f}, "
                f"pred={r['pred']}, "
                f"logit={r['logit']:.4f}"
            )

    y_true = [r["label"] for r in all_fold_results]
    y_prob = [r["prob"] for r in all_fold_results]

    metrics = compute_binary_metrics(
        y_true,
        y_prob,
        threshold=0.5,
    )

    print("\n" + "=" * 80)
    print("LOOCV  汇总结果")
    print("=" * 80)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    pred_df = pd.DataFrame(all_fold_results)
    pred_save_path = RESULT_DIR / "loocv_predictions.csv"
    pred_df.to_csv(
        pred_save_path,
        index=False,
        encoding="utf-8-sig",
    )

    metric_df = pd.DataFrame([metrics])
    metric_save_path = RESULT_DIR / "loocv_metrics.csv"
    metric_df.to_csv(
        metric_save_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n结果已保存:")
    print(f"predictions: {pred_save_path}")
    print(f"metrics: {metric_save_path}")


if __name__ == "__main__":
    main()