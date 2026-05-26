import os
import json
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import (
    INDEX_CSV_PATH, TEST_SIZE, RANDOM_SEED, BATCH_SIZE,
    NUM_EPOCHS, LR, WEIGHT_DECAY, DEVICE, NUM_CLASSES,
    LOG_DIR, SPLIT_JSON_PATH, BEST_MODEL_PATH, TEST_METRICS_JSON_PATH
)
from utils.seed import set_seed
from utils.metrics import compute_binary_metrics, print_binary_metrics, save_metrics
from datasets.visit_feature_dataset import VisitFeatureDataset
from models.mlp_classifier import MLPClassifier


def evaluate_model(model, dataloader, device):
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    metrics = compute_binary_metrics(all_labels, all_preds, all_probs)
    return metrics


def main():
    set_seed(RANDOM_SEED)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("1) 读取特征索引...")
    df = pd.read_csv(INDEX_CSV_PATH)

    print("2) 按 case_id 划分训练/测试...")
    case_ids = df["case_id"].unique()
    case_labels = df.groupby("case_id")["label"].first().loc[case_ids].values

    train_case_ids, test_case_ids = train_test_split(
        case_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=case_labels
    )

    split_info = {
        "train_case_ids": train_case_ids.tolist(),
        "test_case_ids": test_case_ids.tolist()
    }
    with open(SPLIT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    train_df = df[df["case_id"].isin(train_case_ids)].reset_index(drop=True)
    test_df = df[df["case_id"].isin(test_case_ids)].reset_index(drop=True)

    print(f"训练病例数: {len(train_case_ids)} | 测试病例数: {len(test_case_ids)}")
    print(f"训练检查数: {len(train_df)} | 测试检查数: {len(test_df)}")

    print("3) 构建 Dataset / DataLoader...")
    train_dataset = VisitFeatureDataset(train_df)
    test_dataset = VisitFeatureDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    sample_x, _ = train_dataset[0]
    input_dim = sample_x.shape[0]

    print("4) 构建模型...")
    model = MLPClassifier(input_dim=input_dim, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_acc = 0.0

    print("5) 开始训练...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_dataset)

        metrics = evaluate_model(model, test_loader, DEVICE)

        print(
            f"Epoch {epoch+1:02d} | "
            f"loss={avg_loss:.4f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"auc={metrics['roc_auc']:.4f}"
        )

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            save_metrics(metrics, TEST_METRICS_JSON_PATH)

    print("\n训练完成")
    print("最佳模型保存到:", BEST_MODEL_PATH)

    print("\n最佳模型对应测试集指标：")
    with open(TEST_METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        best_metrics = json.load(f)
    print_binary_metrics(best_metrics)


if __name__ == "__main__":
    main()