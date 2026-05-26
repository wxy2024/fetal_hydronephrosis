import json
import pandas as pd
import torch

from torch.utils.data import DataLoader

from config import (
    INDEX_CSV_PATH, BATCH_SIZE, DEVICE, NUM_CLASSES,
    SPLIT_JSON_PATH, BEST_MODEL_PATH, TEST_METRICS_JSON_PATH
)
from utils.metrics import compute_binary_metrics, print_binary_metrics, save_metrics
from datasets.visit_feature_dataset import VisitFeatureDataset
from models.mlp_classifier import MLPClassifier


def main():
    print("1) 读取特征索引...")
    df = pd.read_csv(INDEX_CSV_PATH)

    print("2) 读取数据划分...")
    with open(SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    test_case_ids = split_info["test_case_ids"]
    test_df = df[df["case_id"].isin(test_case_ids)].reset_index(drop=True)

    print(f"测试病例数: {len(set(test_case_ids))}")
    print(f"测试检查数: {len(test_df)}")

    print("3) 构建 Dataset / DataLoader...")
    test_dataset = VisitFeatureDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    sample_x, _ = test_dataset[0]
    input_dim = sample_x.shape[0]

    print("4) 加载模型...")
    model = MLPClassifier(input_dim=input_dim, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("5) 开始测试...")
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    metrics = compute_binary_metrics(all_labels, all_preds, all_probs)

    print("\n测试集结果：")
    print_binary_metrics(metrics)

    save_metrics(metrics, TEST_METRICS_JSON_PATH)
    print("\n测试指标已保存到:")
    print(TEST_METRICS_JSON_PATH)


if __name__ == "__main__":
    main()