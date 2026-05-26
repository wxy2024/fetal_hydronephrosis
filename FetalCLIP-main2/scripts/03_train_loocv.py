# 这个脚本读取已经保存的特征，执行 LOOCV。

# 它会对每个单侧样本执行：
# visit_features: [T, D_visit]
# 然后做：
# mean pooling: [D_visit]
# max pooling:  [D_visit]
# last pooling: [D_visit]
# 最终得到：
# side_feature: [3 * D_visit]

# scripts/03_train_loocv.py
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config import (
    FEATURE_DIR,
    RESULT_DIR,
    NUM_EPOCHS,
    LR,
    WEIGHT_DECAY,
    DROPOUT,
    BATCH_SIZE,
    SEED,
    USE_POS_WEIGHT,
    DEVICE,
)
from models.baseline_model import SimpleSideClassifier
from utils.file_utils import ensure_dir
from utils.seed import set_seed
from utils.metrics import compute_binary_metrics

from torch.utils.data import WeightedRandomSampler


# 对输出进行温度校准用
class TempCalibrator(nn.Module):
    """
    Temperature scaling 用于校准模型输出概率
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # 初始温度 >1

    def forward(self, logits):
        return logits / self.temperature







class SideFeatureDataset(Dataset):
    """
    已提取特征后的单侧样本数据集。
    """

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        x = item["side_feature"].float()
        y = torch.tensor(float(item["label"]), dtype=torch.float32)

        return x, y, item["sample_id"], item["case_id"]



# 平均/最大/最新   三种方法融合   聚合单侧图像特征
def load_feature_items(feature_dir: Path):
    """
    读取 outputs/features 下所有 .pt 文件，并构造 side-level 特征。
    """

    pt_files = sorted(feature_dir.glob("*.pt"), key=lambda x: x.name)

    items = []

    for pt_path in pt_files:
        obj = torch.load(pt_path, map_location="cpu")

        visit_features = obj["visit_features"].float()  # [T, D]

        if visit_features.ndim != 2:
            print(f"[跳过] visit_features 维度异常: {pt_path}")
            continue

        if visit_features.shape[0] == 0:
            print(f"[跳过] 空 visit_features: {pt_path}")
            continue

        # mean pooling
        mean_feat = visit_features.mean(dim=0)

        # max pooling
        max_feat = visit_features.max(dim=0).values

        # last visit pooling
        last_feat = visit_features[-1]

        # side-level feature
        side_feature = torch.cat([mean_feat, max_feat, last_feat], dim=0)

        item = {
            "sample_id": obj["sample_id"],
            "case_id": obj["case_id"],
            "side": obj["side"],
            "label": int(obj["label"]),
            "num_visits": int(obj["num_visits"]),
            "side_feature": side_feature,
            "feature_path": str(pt_path),
        }

        items.append(item)

    return items





# 3头注意力池化   聚合单侧图像特征
# class AttentionPooling(nn.Module):
#     """
#     简单多头注意力池化，用于 visit_features 聚合
#     """

#     def __init__(self, feature_dim, num_heads=3):
#         super().__init__()
#         self.num_heads = num_heads
#         self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)

#     def forward(self, x):
#         # x: [T, D]
#         x = x.unsqueeze(0)  # [1, T, D]  batch_first=True
#         attn_output, _ = self.attn(x, x, x)  # [1, T, D]
#         pooled = attn_output.mean(dim=1).squeeze(0)  # [D]
#         return pooled

# def load_feature_items(feature_dir: Path):
#     """
#     使用 3 头注意力池化聚合 visit_features。
#     """

#     pt_files = sorted(feature_dir.glob("*.pt"), key=lambda x: x.name)
#     items = []

#     for pt_path in pt_files:
#         obj = torch.load(pt_path, map_location="cpu")
#         visit_features = obj["visit_features"].float()  # [T, D]

#         if visit_features.ndim != 2 or visit_features.shape[0] == 0:
#             print(f"[跳过] visit_features 异常或为空: {pt_path}")
#             continue

#         feature_dim = visit_features.shape[1]
#         attn_pool = AttentionPooling(feature_dim=feature_dim, num_heads=3)
#         side_feature = attn_pool(visit_features).detach()  # [D]

#         item = {
#             "sample_id": obj["sample_id"],
#             "case_id": obj["case_id"],
#             "side": obj["side"],
#             "label": int(obj["label"]),
#             "num_visits": int(obj["num_visits"]),
#             "side_feature": side_feature,
#             "feature_path": str(pt_path),
#         }
#         items.append(item)

#     return items



def train_one_fold(train_items, test_items, input_dim: int, fold_case_id: str):
    """
    训练一个 LOOCV fold。
    """

    
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # 普通数据加载器（不使用 WeightedRandomSampler）
    # train_dataset = SideFeatureDataset(train_items)
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=min(BATCH_SIZE, len(train_dataset)),
    #     shuffle=True,
    #     drop_last=False,
    # )


    # 使用WeightedRandomSampler的数据加载器   PS:使用WeightedRandomSampler确实有效提升模型性能
    train_dataset = SideFeatureDataset(train_items)

    # WeightedRandomSampler
    train_labels = [x["label"] for x in train_items]
    class_counts = np.bincount(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(BATCH_SIZE, len(train_dataset)),
        sampler=sampler,
        drop_last=False,
    )





    model = SimpleSideClassifier(
        input_dim=input_dim,
        hidden_dim=128,
        dropout=DROPOUT,
    ).to(device)

    # 类别不平衡权重
    if USE_POS_WEIGHT:
        train_labels = np.array([x["label"] for x in train_items])
        num_pos = np.sum(train_labels == 1)
        num_neg = np.sum(train_labels == 0)

        if num_pos > 0:
            pos_weight_value = float(num_neg / max(num_pos, 1))
        else:
            pos_weight_value = 1.0

        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
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
        epoch_losses = []

        for x, y, _, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(
                f"    Epoch [{epoch + 1}/{NUM_EPOCHS}], "
                f"loss={np.mean(epoch_losses):.4f}"
            )

    # 测试
    model.eval()
    fold_results = []


    # 使用温度校准

    # 初始化温度校准器
    # temp_calib = TempCalibrator().to(device)

    # # 使用训练集 logits 拟合温度
    # model.eval()
    # train_logits_list = []
    # train_labels_list = []

    # with torch.no_grad():
    #     for x, y, _, _ in train_loader:
    #         x = x.to(device)
    #         y = y.to(device)
    #         logits = model(x)
    #         train_logits_list.append(logits)
    #         train_labels_list.append(y)

    # train_logits = torch.cat(train_logits_list)
    # train_labels = torch.cat(train_labels_list)

    # # 使用简单优化器校准温度
    # optimizer = torch.optim.LBFGS([temp_calib.temperature], lr=0.01, max_iter=50)

    # criterion = nn.BCEWithLogitsLoss()

    # def eval():
    #     optimizer.zero_grad()
    #     logits_temp = train_logits / temp_calib.temperature
    #     loss = criterion(logits_temp, train_labels)
    #     loss.backward()
    #     return loss

    # optimizer.step(eval)

    # print(f"Calibrated temperature: {temp_calib.temperature.item():.3f}")

    # # 测试阶段使用校准后的 logits
    # with torch.no_grad():
    #     for item in test_items:
    #         x = item["side_feature"].unsqueeze(0).float().to(device)
    #         logit = model(x)
    #         logit_temp = temp_calib(logit)
    #         prob = torch.sigmoid(logit_temp).item()

    #         fold_results.append({
    #             "fold_case_id": fold_case_id,
    #             "sample_id": item["sample_id"],
    #             "case_id": item["case_id"],
    #             "side": item["side"],
    #             "label": int(item["label"]),
    #             "prob": float(prob),
    #             "pred": int(prob >= 0.5),  # 阈值可调整
    #         })


    # 原来的未使用温度校准的测试函数
    with torch.no_grad():
        for item in test_items:
            x = item["side_feature"].unsqueeze(0).float().to(device)
            logit = model(x)
            prob = torch.sigmoid(logit).item()

            fold_results.append({
                "fold_case_id": fold_case_id,
                "sample_id": item["sample_id"],
                "case_id": item["case_id"],
                "side": item["side"],
                "label": int(item["label"]),
                "prob": float(prob),
                "pred": int(prob >= 0.5),
            })

    return fold_results


def main():
    set_seed(SEED)
    ensure_dir(RESULT_DIR)

    items = load_feature_items(FEATURE_DIR)

    print("=" * 80)
    print(f"读取特征数量: {len(items)}")
    print("=" * 80)

    if len(items) == 0:
        raise RuntimeError("没有读取到任何特征，请先运行 scripts/02_extract_features.py")

    case_ids = sorted(set([x["case_id"] for x in items]))
    print(f"病例数量: {len(case_ids)}")
    print(f"病例列表: {case_ids}")

    input_dim = items[0]["side_feature"].shape[0]
    print(f"side_feature input_dim: {input_dim}")
    print("=" * 80)

    all_fold_results = []

    for fold_idx, test_case_id in enumerate(case_ids):
        print("\n" + "=" * 80)
        print(f"LOOCV Fold {fold_idx + 1}/{len(case_ids)}")
        print(f"Test case: {test_case_id}")
        print("=" * 80)

        train_items = [x for x in items if x["case_id"] != test_case_id]
        test_items = [x for x in items if x["case_id"] == test_case_id]

        print(f"训练样本数: {len(train_items)}")
        print(f"测试样本数: {len(test_items)}")
        print(f"测试样本: {[x['sample_id'] for x in test_items]}")

        fold_results = train_one_fold(
            train_items=train_items,
            test_items=test_items,
            input_dim=input_dim,
            fold_case_id=test_case_id,
        )

        all_fold_results.extend(fold_results)

        for r in fold_results:
            print(
                f"  {r['sample_id']}: "
                f"label={r['label']}, prob={r['prob']:.4f}, pred={r['pred']}"
            )

    # 汇总整体 out-of-fold 结果
    y_true = [r["label"] for r in all_fold_results]
    y_prob = [r["prob"] for r in all_fold_results]

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    print("\n" + "=" * 80)
    print("LOOCV 汇总结果")
    print("=" * 80)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 保存预测结果
    import pandas as pd

    pred_df = pd.DataFrame(all_fold_results)
    pred_save_path = RESULT_DIR / "loocv_predictions.csv"
    pred_df.to_csv(pred_save_path, index=False, encoding="utf-8-sig")

    metric_df = pd.DataFrame([metrics])
    metric_save_path = RESULT_DIR / "loocv_metrics.csv"
    metric_df.to_csv(metric_save_path, index=False, encoding="utf-8-sig")

    print("\n结果已保存:")
    print(f"  predictions: {pred_save_path}")
    print(f"  metrics: {metric_save_path}")


if __name__ == "__main__":
    main()
