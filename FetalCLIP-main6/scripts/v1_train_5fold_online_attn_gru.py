import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None

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
    TEMPORAL_BACKBONE,
)

from data.dataset_online import collect_case_side_samples_online
from fetalclip.fetalclip_wrapper import FetalCLIPFeatureExtractor
from models.longitudinal_gru_classifier import ModularGRUSideClassifier
from utils.metrics import compute_binary_metrics


N_SPLITS = 5


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
    print("\n" + "=" * 80)
    print("Extracting frozen FetalCLIP features into memory")
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
            visit["text_feature"] = extractor.encode_text(visit.get("report_text", ""))
            visit["image_features"] = extractor.encode_images(visit.get("image_paths", []))

    print("Feature extraction completed.")


def infer_feature_dim(samples) -> int:
    for sample in samples:
        for visit in sample["visits"]:
            text_feat = visit.get("text_feature", None)
            if text_feat is not None:
                return int(text_feat.shape[0])

            image_feats = visit.get("image_features", None)
            if image_feats is not None and image_feats.numel() > 0:
                return int(image_feats.shape[1])

    raise RuntimeError("Could not infer FetalCLIP feature dimension.")


def make_train_loader(train_items):
    train_dataset = RawSideDataset(train_items)
    train_labels = [int(x["label"]) for x in train_items]
    class_counts = np.bincount(train_labels, minlength=2)

    weights = []
    for label in train_labels:
        count = class_counts[label]
        weights.append(1.0 / count if count > 0 else 1.0)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )

    return DataLoader(
        train_dataset,
        batch_size=min(BATCH_SIZE, len(train_dataset)),
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_raw_samples,
    )


def train_one_fold(train_items, test_items, feature_dim: int, fold_id: str, test_case_ids):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    print("[Device Check] DEVICE from config:", DEVICE)
    print("[Device Check] torch.cuda.is_available():", torch.cuda.is_available())
    print("[Device Check] actual training device:", device)
    if torch.cuda.is_available():
        print("[Device Check] GPU name:", torch.cuda.get_device_name(0))

    train_loader = make_train_loader(train_items)

    model = ModularGRUSideClassifier(
        feature_dim=feature_dim,
        image_pooling_type=IMAGE_POOLING_TYPE,
        visit_fusion_type=VISIT_FUSION_TYPE,
        attn_hidden_dim=ATTN_HIDDEN_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        gru_hidden_dim=GRU_HIDDEN_DIM,
        gru_num_layers=GRU_NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=GRU_BIDIRECTIONAL,
        temporal_backbone=TEMPORAL_BACKBONE,
    ).to(device)
    print("[Device Check] model device:", next(model.parameters()).device)

    if USE_POS_WEIGHT:
        train_labels = np.array([int(x["label"]) for x in train_items])
        num_pos = np.sum(train_labels == 1)
        num_neg = np.sum(train_labels == 0)
        pos_weight_value = float(num_neg / max(num_pos, 1)) if num_pos > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
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
            print(f"  Epoch [{epoch + 1}/{NUM_EPOCHS}] loss={np.mean(losses):.4f}")

    model.eval()
    fold_results = []

    with torch.no_grad():
        for item in test_items:
            logits, attention_batch = model([item], return_attention=SAVE_ATTENTION)

            logit_value = float(logits.item())
            prob = float(torch.sigmoid(logits).item())

            attention_json = ""
            if SAVE_ATTENTION:
                attention_json = json.dumps(attention_batch[0], ensure_ascii=False)

            fold_results.append(
                {
                    "fold_id": fold_id,
                    "test_case_ids": "|".join(test_case_ids),
                    "sample_id": item["sample_id"],
                    "case_id": item["case_id"],
                    "side": item["side"],
                    "label": int(item["label"]),
                    "num_visits": int(item["num_visits"]),
                    "visit_ids_sorted": "|".join(
                        [str(v.get("visit_id", "")) for v in item["visits"]]
                    ),
                    "prob": prob,
                    "logit": logit_value,
                    "pred": int(prob >= 0.5),
                    "image_attention_json": attention_json,
                }
            )

    return fold_results


def make_case_level_stratified_folds(samples, n_splits=5, seed=42):
    case_ids = sorted(set(x["case_id"] for x in samples))
    case_labels = []
    for case_id in case_ids:
        labels = [int(x["label"]) for x in samples if x["case_id"] == case_id]
        case_labels.append(max(labels))

    case_df = pd.DataFrame({"case_id": case_ids, "label": case_labels})

    if StratifiedGroupKFold is None:
        print(
            "[Warning] sklearn StratifiedGroupKFold is unavailable. "
            "Using built-in case-level stratified round-robin split."
        )
        return make_case_level_stratified_folds_fallback(case_df, n_splits, seed)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    dummy_x = np.zeros(len(case_df))
    groups = case_df["case_id"].to_numpy()
    labels = case_df["label"].to_numpy()

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(dummy_x, labels, groups),
        start=1,
    ):
        train_cases = set(case_df.iloc[train_idx]["case_id"].tolist())
        test_cases = set(case_df.iloc[test_idx]["case_id"].tolist())
        folds.append(
            {
                "fold_idx": fold_idx,
                "fold_id": f"fold_{fold_idx}",
                "train_cases": train_cases,
                "test_cases": test_cases,
            }
        )
    return folds


def make_case_level_stratified_folds_fallback(case_df, n_splits=5, seed=42):
    rng = random.Random(seed)
    folds = [{"fold_idx": i + 1, "fold_id": f"fold_{i + 1}", "test_cases": set()} for i in range(n_splits)]

    for _, group in case_df.groupby("label"):
        case_list = group["case_id"].tolist()
        rng.shuffle(case_list)
        for idx, case_id in enumerate(case_list):
            folds[idx % n_splits]["test_cases"].add(case_id)

    all_cases = set(case_df["case_id"].tolist())
    for fold in folds:
        fold["train_cases"] = all_cases - fold["test_cases"]
    return folds


def summarize_fold_metrics(fold_metric_rows):
    metric_names = [
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "Precision_PPV",
        "NPV",
        "F1",
        "AUC",
    ]

    summary = {"num_folds": len(fold_metric_rows)}
    fold_metric_df = pd.DataFrame(fold_metric_rows)

    for metric_name in metric_names:
        values = pd.to_numeric(fold_metric_df[metric_name], errors="coerce").dropna()
        summary[f"{metric_name}_mean"] = float(values.mean()) if len(values) > 0 else None
        summary[f"{metric_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None

    for count_name in ["TN", "FP", "FN", "TP"]:
        summary[f"{count_name}_sum"] = int(fold_metric_df[count_name].sum())

    return summary


def main():
    set_seed(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Reading raw data")
    print(f"RAW_ROOT: {RAW_ROOT}")
    print("=" * 80)

    samples = collect_case_side_samples_online(RAW_ROOT)
    print(f"Side-level sample count: {len(samples)}")

    if len(samples) == 0:
        raise RuntimeError("No samples were loaded. Please check RAW_ROOT and dataset structure.")

    print("\n" + "=" * 80)
    print("Loading FetalCLIP")
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
            "This script expects USE_MEMORY_CACHE=True. "
            "Feature extraction is done once before cross-validation."
        )

    feature_dim = infer_feature_dim(samples)
    print(f"FetalCLIP feature_dim: {feature_dim}")

    case_ids = sorted(set(x["case_id"] for x in samples))
    print(f"Case count: {len(case_ids)}")
    print(f"Case list: {case_ids}")

    print("\nChecking first 5 samples:")
    for item in samples[:5]:
        print(
            f"  {item['sample_id']}: "
            f"label={item['label']}, "
            f"num_visits={item['num_visits']}, "
            f"visits={[v['visit_id'] for v in item['visits']]}"
        )

    folds = make_case_level_stratified_folds(samples, n_splits=N_SPLITS, seed=SEED)
    all_fold_results = []
    fold_metric_rows = []

    for fold in folds:
        fold_idx = fold["fold_idx"]
        fold_id = fold["fold_id"]
        train_cases = fold["train_cases"]
        test_cases = fold["test_cases"]
        test_case_ids = sorted(test_cases)

        print("\n" + "=" * 80)
        print(f"5-Fold CV {fold_idx}/{N_SPLITS}")
        print(f"Test cases: {test_case_ids}")
        print("=" * 80)

        train_items = [x for x in samples if x["case_id"] in train_cases]
        test_items = [x for x in samples if x["case_id"] in test_cases]

        train_labels = np.array([int(x["label"]) for x in train_items])
        test_labels = np.array([int(x["label"]) for x in test_items])

        print(f"Train samples: {len(train_items)}")
        print(f"Test samples: {len(test_items)}")
        print(f"Train labels: neg={sum(train_labels == 0)}, pos={sum(train_labels == 1)}")
        print(f"Test labels: neg={sum(test_labels == 0)}, pos={sum(test_labels == 1)}")
        print(f"Test samples: {[x['sample_id'] for x in test_items]}")

        fold_results = train_one_fold(
            train_items=train_items,
            test_items=test_items,
            feature_dim=feature_dim,
            fold_id=fold_id,
            test_case_ids=test_case_ids,
        )

        all_fold_results.extend(fold_results)

        fold_y_true = [r["label"] for r in fold_results]
        fold_y_prob = [r["prob"] for r in fold_results]
        fold_metrics = compute_binary_metrics(fold_y_true, fold_y_prob, threshold=0.5)
        fold_metrics_row = {
            "fold_id": fold_id,
            "test_case_ids": "|".join(test_case_ids),
            "train_samples": len(train_items),
            "test_samples": len(test_items),
            **fold_metrics,
        }
        fold_metric_rows.append(fold_metrics_row)

        for r in fold_results:
            print(
                f"  {r['sample_id']}: "
                f"label={r['label']}, "
                f"prob={r['prob']:.4f}, "
                f"pred={r['pred']}, "
                f"logit={r['logit']:.4f}"
            )

        print(f"Fold metrics: {fold_metrics}")

    y_true = [r["label"] for r in all_fold_results]
    y_prob = [r["prob"] for r in all_fold_results]
    pooled_metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    cv_metrics = summarize_fold_metrics(fold_metric_rows)

    print("\n" + "=" * 80)
    print("5-Fold CV summary: mean ± std across folds")
    print("=" * 80)

    for k, v in cv_metrics.items():
        print(f"{k}: {v}")

    print("\n" + "=" * 80)
    print("Pooled out-of-fold metrics")
    print("=" * 80)

    for k, v in pooled_metrics.items():
        print(f"{k}: {v}")

    pred_df = pd.DataFrame(all_fold_results)
    pred_save_path = RESULT_DIR / "5fold_predictions.csv"
    pred_df.to_csv(pred_save_path, index=False, encoding="utf-8-sig")

    fold_metric_df = pd.DataFrame(fold_metric_rows)
    fold_metric_save_path = RESULT_DIR / "5fold_fold_metrics.csv"
    fold_metric_df.to_csv(fold_metric_save_path, index=False, encoding="utf-8-sig")

    metric_df = pd.DataFrame([cv_metrics])
    metric_save_path = RESULT_DIR / "5fold_metrics.csv"
    metric_df.to_csv(metric_save_path, index=False, encoding="utf-8-sig")

    pooled_metric_df = pd.DataFrame([pooled_metrics])
    pooled_metric_save_path = RESULT_DIR / "5fold_pooled_metrics.csv"
    pooled_metric_df.to_csv(pooled_metric_save_path, index=False, encoding="utf-8-sig")

    print("\nResults saved:")
    print(f"predictions: {pred_save_path}")
    print(f"fold metrics: {fold_metric_save_path}")
    print(f"metrics mean/std: {metric_save_path}")
    print(f"pooled metrics: {pooled_metric_save_path}")


if __name__ == "__main__":
    main()
