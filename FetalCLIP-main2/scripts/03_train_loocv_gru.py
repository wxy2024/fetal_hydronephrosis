# scripts/03_train_loocv_gru.py
# 使用 GRU 对单侧肾脏的多次检查特征序列进行分类
#
# 每个 .pt 文件应包含：
# sample_id, case_id, side, label, num_visits, visit_ids, visit_features
#
# visit_features: [T, D_visit]
# visit_ids: 长度为 T 的列表，前 8 位为检查日期，例如 20250604
#
# 本脚本会：
# 1. 按 visit_ids 前 8 位日期对 visit_features 排序
# 2. 将排序后的 visit_features 作为序列输入 GRU
# 3. 使用 GRU 最后隐藏状态进行二分类
# 4. 按 case_id 执行 LOOCV，避免同一病例左右侧数据泄漏

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence

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

from utils.file_utils import ensure_dir
from utils.seed import set_seed
from utils.metrics import compute_binary_metrics


class GRUSideClassifier(nn.Module):
    """
    基于 GRU 的单侧肾脏序列分类模型。

    输入:
        x: [B, T, D]
            B 表示 batch size
            T 表示检查次数
            D 表示每次检查的特征维度，例如 1539

        lengths: [B]
            每个样本真实的检查次数，用于处理 padding

    输出:
        logits: [B]
            二分类 logit，后续用 BCEWithLogitsLoss
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * self.num_directions

        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_out_dim),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, lengths):
        """
        x: [B, T, D]
        lengths: [B]
        """

        # pack 后，GRU 不会把 padding 部分当作真实时间点处理
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, h_n = self.gru(packed)

        # h_n shape:
        # 单向 GRU: [num_layers, B, hidden_dim]
        # 双向 GRU: [num_layers * 2, B, hidden_dim]
        if self.bidirectional:
            # 最后一层 forward 和 backward 的 hidden state
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h_last = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_last = h_n[-1]

        logits = self.classifier(h_last).squeeze(-1)

        return logits


# 在输入GRU之前进行特征降维
# class GRUSideClassifier(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         proj_dim: int = 64,
#         hidden_dim: int = 32,
#         num_layers: int = 1,
#         dropout: float = 0.6,
#         bidirectional: bool = False,
#     ):
#         super().__init__()

#         self.input_proj = nn.Sequential(
#             nn.LayerNorm(input_dim),
#             nn.Linear(input_dim, proj_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#         self.bidirectional = bidirectional
#         self.num_directions = 2 if bidirectional else 1

#         self.gru = nn.GRU(
#             input_size=proj_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0.0,
#             bidirectional=bidirectional,
#         )

#         out_dim = hidden_dim * self.num_directions

#         self.classifier = nn.Sequential(
#             nn.LayerNorm(out_dim),
#             nn.Dropout(dropout),
#             nn.Linear(out_dim, 32),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(32, 1),
#         )

#     def forward(self, x, lengths):
#         # x: [B, T, D]
#         x = self.input_proj(x)  # [B, T, proj_dim]

#         packed = pack_padded_sequence(
#             x,
#             lengths.cpu(),
#             batch_first=True,
#             enforce_sorted=False,
#         )

#         _, h_n = self.gru(packed)

#         if self.bidirectional:
#             h_forward = h_n[-2]
#             h_backward = h_n[-1]
#             h_last = torch.cat([h_forward, h_backward], dim=1)
#         else:
#             h_last = h_n[-1]

#         logits = self.classifier(h_last).squeeze(-1)
#         return logits




class SideSequenceDataset(Dataset):
    """
    单侧肾脏序列数据集。
    每个样本是一侧肾脏的多次检查序列。
    """

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        x = item["visit_features"].float()  # [T, D]
        y = torch.tensor(float(item["label"]), dtype=torch.float32)

        return {
            "x": x,
            "y": y,
            "sample_id": item["sample_id"],
            "case_id": item["case_id"],
            "side": item["side"],
            "visit_ids": item["visit_ids"],
        }


def collate_side_sequences(batch):
    """
    将不同长度的 visit 序列 padding 到同一长度。

    输入 batch 中每个 x 的形状是 [T_i, D]
    输出 padded_x 的形状是 [B, T_max, D]
    """

    xs = [b["x"] for b in batch]
    ys = torch.stack([b["y"] for b in batch])

    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)

    batch_size = len(xs)
    max_len = int(lengths.max().item())
    input_dim = xs[0].shape[1]

    padded_x = torch.zeros(batch_size, max_len, input_dim, dtype=torch.float32)

    for i, x in enumerate(xs):
        t = x.shape[0]
        padded_x[i, :t, :] = x

    sample_ids = [b["sample_id"] for b in batch]
    case_ids = [b["case_id"] for b in batch]
    sides = [b["side"] for b in batch]
    visit_ids = [b["visit_ids"] for b in batch]

    return padded_x, lengths, ys, sample_ids, case_ids, sides, visit_ids


def sort_visits_by_date(visit_ids, visit_features):
    """
    根据 visit_ids 前 8 位日期对 visit_features 排序。

    visit_ids 示例:
        202506040840
        202412300188

    前 8 位:
        20250604
        20241230

    注意:
        必须对 visit_ids 和 visit_features 使用同一组排序索引，
        否则检查时间和特征会错位。
    """

    if visit_ids is None:
        return visit_ids, visit_features

    if len(visit_ids) != visit_features.shape[0]:
        print(
            f"[警告] visit_ids 数量与 visit_features 行数不一致: "
            f"len(visit_ids)={len(visit_ids)}, "
            f"visit_features.shape[0]={visit_features.shape[0]}，保持原顺序。"
        )
        return visit_ids, visit_features

    def get_date_key(v):
        v = str(v)
        if len(v) >= 8 and v[:8].isdigit():
            return v[:8]
        return "99999999"

    sorted_indices = sorted(
        range(len(visit_ids)),
        key=lambda i: (get_date_key(visit_ids[i]), str(visit_ids[i]), i),
    )

    sorted_visit_ids = [visit_ids[i] for i in sorted_indices]
    sorted_visit_features = visit_features[sorted_indices]

    return sorted_visit_ids, sorted_visit_features


def load_feature_items(feature_dir: Path):
    """
    读取 outputs/features 下所有 .pt 文件。

    与原来的 mean/max/last pooling 不同，这里不再构造 side_feature，
    而是保留排序后的 visit_features: [T, D]，直接输入 GRU。
    """

    pt_files = sorted(feature_dir.glob("*.pt"), key=lambda x: x.name)

    items = []

    for pt_path in pt_files:
        obj = torch.load(pt_path, map_location="cpu")

        if "visit_features" not in obj:
            print(f"[跳过] 缺少 visit_features: {pt_path}")
            continue

        visit_features = obj["visit_features"].float()  # [T, D]

        if visit_features.ndim != 2:
            print(f"[跳过] visit_features 维度异常: {pt_path}")
            continue

        if visit_features.shape[0] == 0:
            print(f"[跳过] 空 visit_features: {pt_path}")
            continue

        visit_ids = obj.get("visit_ids", None)

        if visit_ids is None:
            print(f"[警告] 缺少 visit_ids，保持原始顺序: {pt_path}")
            visit_ids = [f"visit_{i}" for i in range(visit_features.shape[0])]

        visit_ids, visit_features = sort_visits_by_date(
            visit_ids=visit_ids,
            visit_features=visit_features,
        )

        item = {
            "sample_id": obj["sample_id"],
            "case_id": obj["case_id"],
            "side": obj["side"],
            "label": int(obj["label"]),
            "num_visits": int(visit_features.shape[0]),
            "visit_ids": visit_ids,
            "visit_features": visit_features,
            "feature_path": str(pt_path),
        }

        items.append(item)

    return items


def make_train_loader(train_items):
    """
    构建训练 DataLoader。
    这里保留 WeightedRandomSampler，用于缓解类别不平衡。
    """

    train_dataset = SideSequenceDataset(train_items)

    train_labels = [int(x["label"]) for x in train_items]

    # minlength=2 确保即使某一折只有一个类别，也不会出现索引问题
    class_counts = np.bincount(train_labels, minlength=2)

    weights = []
    for label in train_labels:
        count = class_counts[label]
        if count == 0:
            weights.append(1.0)
        else:
            weights.append(1.0 / count)

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
        collate_fn=collate_side_sequences,
    )

    return train_loader


def train_one_fold(train_items, test_items, input_dim: int, fold_case_id: str):
    """
    训练一个 LOOCV fold。
    """

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    train_loader = make_train_loader(train_items)

    model = GRUSideClassifier(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=DROPOUT,
        bidirectional=True,
    ).to(device)

    # 输入gru之前进行特征降维
    # model = GRUSideClassifier(
    #     input_dim=input_dim,
    #     proj_dim=64,
    #     hidden_dim=32,
    #     num_layers=1,
    #     dropout=0.6,
    #     bidirectional=False,
    # ).to(device)

    # 类别不平衡权重
    if USE_POS_WEIGHT:
        train_labels = np.array([int(x["label"]) for x in train_items])
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

        for padded_x, lengths, y, _, _, _, _ in train_loader:
            padded_x = padded_x.to(device)      # [B, T_max, D]
            lengths = lengths.to(device)        # [B]
            y = y.to(device)                    # [B]

            logits = model(padded_x, lengths)   # [B]
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

    with torch.no_grad():
        for item in test_items:
            x = item["visit_features"].unsqueeze(0).float().to(device)  # [1, T, D]
            lengths = torch.tensor([item["num_visits"]], dtype=torch.long).to(device)

            logit = model(x, lengths)
            prob = torch.sigmoid(logit).item()

            ##保存logit看是是否logit 绝对值过大导致预测结果过于绝对
            logit_value = logit.item()

            fold_results.append({
                "fold_case_id": fold_case_id,
                "sample_id": item["sample_id"],
                "case_id": item["case_id"],
                "side": item["side"],
                "label": int(item["label"]),
                "num_visits": int(item["num_visits"]),
                "visit_ids_sorted": "|".join([str(v) for v in item["visit_ids"]]),
                "prob": float(prob),
                "logit": float(logit_value),
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

    input_dim = items[0]["visit_features"].shape[1]
    print(f"visit feature input_dim: {input_dim}")
    print("=" * 80)

    # 打印前几个样本，确认排序是否正确
    print("检查前 5 个样本的 visit_ids 排序结果:")
    for item in items[:5]:
        print(
            f"  {item['sample_id']}: "
            f"num_visits={item['num_visits']}, "
            f"visit_ids={item['visit_ids']}"
        )

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

        # 某些极小数据情况下，训练集中可能只有一个类别
        train_labels = [int(x["label"]) for x in train_items]
        print(
            f"训练集标签分布: "
            f"neg={sum(np.array(train_labels) == 0)}, "
            f"pos={sum(np.array(train_labels) == 1)}"
        )

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
                f"label={r['label']}, "
                f"num_visits={r['num_visits']}, "
                f"prob={r['prob']:.4f}, "
                f"pred={r['pred']}"
            )

    # 汇总整体 out-of-fold 结果
    y_true = [r["label"] for r in all_fold_results]
    y_prob = [r["prob"] for r in all_fold_results]

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    print("\n" + "=" * 80)
    print("LOOCV-GRU 汇总结果")
    print("=" * 80)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 保存预测结果
    pred_df = pd.DataFrame(all_fold_results)
    pred_save_path = RESULT_DIR / "loocv_gru_predictions.csv"
    pred_df.to_csv(pred_save_path, index=False, encoding="utf-8-sig")

    metric_df = pd.DataFrame([metrics])
    metric_save_path = RESULT_DIR / "loocv_gru_metrics.csv"
    metric_df.to_csv(metric_save_path, index=False, encoding="utf-8-sig")

    print("\n结果已保存:")
    print(f"  predictions: {pred_save_path}")
    print(f"  metrics: {metric_save_path}")


if __name__ == "__main__":
    main()