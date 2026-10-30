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
    INNER_N_SPLITS,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
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
    INPUT_MODALITY,
    TEMPORAL_EXPERIMENT,
    TEMPORAL_MIN_VISITS,
    TEMPORAL_SHUFFLE_SEED,
    USE_EXPLICIT_TIME,
)

from data.dataset_online import collect_case_side_samples_online
from data.temporal_ablations import (
    apply_temporal_experiment,
    filter_multivisit_cases,
)
from fetalclip.fetalclip_wrapper import FetalCLIPFeatureExtractor
from models.longitudinal_gru_classifier import ModularGRUSideClassifier
from utils.metrics import compute_binary_metrics
from utils.checkpoint_io import save_final_checkpoint


from wx_config import APP_TOKEN, UID
import traceback
from datetime import datetime
import requests

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


def attach_fetalclip_features_in_memory(samples, extractor, input_modality):
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
            if input_modality in {"multimodal", "text_only"}:
                visit["text_feature"] = extractor.encode_text(
                    visit.get("report_text", "")
                )
            if input_modality in {"multimodal", "image_only"}:
                visit["image_features"] = extractor.encode_images(
                    visit.get("image_paths", [])
                )

    print("Feature extraction completed.")


def infer_feature_dim(samples, input_modality) -> int:
    for sample in samples:
        for visit in sample["visits"]:
            if input_modality in {"multimodal", "text_only"}:
                text_feat = visit.get("text_feature", None)
                if text_feat is not None:
                    return int(text_feat.shape[0])

            if input_modality in {"multimodal", "image_only"}:
                image_feats = visit.get("image_features", None)
                if image_feats is not None and image_feats.numel() > 0:
                    return int(image_feats.shape[1])

    raise RuntimeError("Could not infer FetalCLIP feature dimension.")


def make_train_loader(train_items, seed):
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
        generator=torch.Generator().manual_seed(seed),
    )

    return DataLoader(
        train_dataset,
        batch_size=min(BATCH_SIZE, len(train_dataset)),
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_raw_samples,
    )


def build_model(feature_dim: int, device: torch.device):
    return ModularGRUSideClassifier(
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
        input_modality=INPUT_MODALITY,
        use_explicit_time=USE_EXPLICIT_TIME,
    ).to(device)


def build_training_criterion(train_items, device: torch.device):
    if not USE_POS_WEIGHT:
        return nn.BCEWithLogitsLoss()

    train_labels = np.array([int(x["label"]) for x in train_items])
    num_pos = np.sum(train_labels == 1)
    num_neg = np.sum(train_labels == 0)
    pos_weight_value = float(num_neg / max(num_pos, 1)) if num_pos > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def compute_validation_loss(model, val_items, device: torch.device) -> float:
    """Unweighted mean BCE used consistently for inner-fold epoch selection."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    with torch.no_grad():
        for start in range(0, len(val_items), BATCH_SIZE):
            batch = val_items[start : start + BATCH_SIZE]
            labels = torch.tensor(
                [float(x["label"]) for x in batch],
                dtype=torch.float32,
                device=device,
            )
            logits, _ = model(batch, return_attention=False)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            total_count += len(batch)

    model.train()
    if total_count == 0:
        raise RuntimeError("Inner validation fold is empty.")
    return total_loss / total_count


def fit_model(
    train_items,
    feature_dim: int,
    seed: int,
    num_epochs: int,
    run_name: str,
    val_items=None,
    patience=None,
):
    """Train once; optionally select an epoch using an inner validation fold."""
    set_seed(seed)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    train_loader = make_train_loader(train_items, seed=seed)
    model = build_model(feature_dim=feature_dim, device=device)
    criterion = build_training_criterion(train_items, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_epoch = int(num_epochs)
    best_val_loss = None
    epochs_without_improvement = 0
    stopped_epoch = int(num_epochs)

    model.train()
    for epoch in range(num_epochs):
        losses = []
        for batch in train_loader:
            labels = torch.tensor(
                [float(x["label"]) for x in batch],
                dtype=torch.float32,
                device=device,
            )

            logits, _ = model(batch, return_attention=False)
            loss = criterion(logits, labels)

            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                bad_samples = [str(x.get("sample_id", "")) for x in batch]
                raise RuntimeError(
                    "Non-finite logits/loss detected before backward. "
                    f"samples={bad_samples}, "
                    f"logits={logits.detach().cpu().tolist()}, "
                    f"loss={float(loss.detach().cpu().item())}"
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses))
        val_loss = None
        if val_items is not None:
            val_loss = compute_validation_loss(model, val_items, device=device)
            improved = (
                best_val_loss is None
                or val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA
            )
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            message = (
                f"  {run_name} Epoch [{epoch + 1}/{num_epochs}] "
                f"train_loss={train_loss:.4f}"
            )
            if val_loss is not None:
                message += f" val_loss={val_loss:.4f} best_epoch={best_epoch}"
            print(message)

        if (
            val_items is not None
            and patience is not None
            and epochs_without_improvement >= patience
        ):
            stopped_epoch = epoch + 1
            print(
                f"  {run_name} early stopping at epoch {stopped_epoch}; "
                f"best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}"
            )
            break

    training_info = {
        "best_epoch": int(best_epoch),
        "best_val_loss": best_val_loss,
        "stopped_epoch": int(stopped_epoch),
    }
    return model, training_info


def select_epoch_with_inner_cv(train_items, feature_dim: int, outer_fold_id: str):
    """Select one epoch count without exposing the outer test fold."""
    outer_fold_number = int(outer_fold_id.rsplit("_", 1)[-1])
    inner_seed = SEED + outer_fold_number * 1000
    inner_folds = make_case_level_stratified_folds(
        train_items,
        n_splits=INNER_N_SPLITS,
        seed=inner_seed,
    )
    selection_rows = []
    best_epochs = []

    for inner_fold in inner_folds:
        inner_idx = inner_fold["fold_idx"]
        inner_train_cases = inner_fold["train_cases"]
        inner_val_cases = inner_fold["test_cases"]
        if inner_train_cases & inner_val_cases:
            raise RuntimeError(
                f"Case leakage detected in {outer_fold_id}/inner_{inner_idx}."
            )
        inner_train_items = [
            x for x in train_items if x["case_id"] in inner_train_cases
        ]
        inner_val_items = [
            x for x in train_items if x["case_id"] in inner_val_cases
        ]
        run_seed = inner_seed + inner_idx
        run_name = f"{outer_fold_id}/inner_{inner_idx}"

        print("\n" + "-" * 80)
        print(f"Nested epoch selection: {run_name}")
        print(
            f"inner train cases={len(inner_train_cases)}, "
            f"validation cases={len(inner_val_cases)}"
        )
        inner_model, info = fit_model(
            train_items=inner_train_items,
            feature_dim=feature_dim,
            seed=run_seed,
            num_epochs=NUM_EPOCHS,
            run_name=run_name,
            val_items=inner_val_items,
            patience=EARLY_STOPPING_PATIENCE,
        )
        best_epochs.append(info["best_epoch"])
        selection_rows.append(
            {
                "outer_fold_id": outer_fold_id,
                "inner_fold_id": f"inner_{inner_idx}",
                "seed": run_seed,
                "inner_train_case_count": len(inner_train_cases),
                "inner_val_case_count": len(inner_val_cases),
                "inner_train_sample_count": len(inner_train_items),
                "inner_val_sample_count": len(inner_val_items),
                "inner_train_case_ids": "|".join(sorted(inner_train_cases)),
                "inner_val_case_ids": "|".join(sorted(inner_val_cases)),
                **info,
            }
        )

        del inner_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_epoch = max(1, int(round(float(np.median(best_epochs)))))
    for row in selection_rows:
        row["selected_epoch_median"] = selected_epoch

    print(
        f"\n{outer_fold_id}: inner best epochs={best_epochs}; "
        f"selected median epoch={selected_epoch}"
    )
    return selected_epoch, selection_rows


def train_one_fold(train_items, test_items, feature_dim: int, fold_id: str, test_case_ids):
    fold_number = int(fold_id.rsplit("_", 1)[-1])
    fold_seed = SEED + fold_number
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    print("[Device Check] DEVICE from config:", DEVICE)
    print("[Device Check] torch.cuda.is_available():", torch.cuda.is_available())
    print("[Device Check] actual training device:", device)
    if torch.cuda.is_available():
        print("[Device Check] GPU name:", torch.cuda.get_device_name(0))

    selected_epoch, selection_rows = select_epoch_with_inner_cv(
        train_items=train_items,
        feature_dim=feature_dim,
        outer_fold_id=fold_id,
    )
    model, outer_training_info = fit_model(
        train_items=train_items,
        feature_dim=feature_dim,
        seed=fold_seed,
        num_epochs=selected_epoch,
        run_name=f"{fold_id}/outer_retrain",
    )
    if outer_training_info["stopped_epoch"] != selected_epoch:
        raise RuntimeError("Outer retraining did not complete the selected epoch count.")

    checkpoint_path = RESULT_DIR / fold_id / "final_model.pt"
    metadata = {
        "checkpoint_role": "final_outer_fold_model",
        "fold_id": fold_id,
        "seed": int(SEED),
        "fold_seed": int(fold_seed),
        "selected_epoch": int(selected_epoch),
        "train_case_ids": sorted({str(x["case_id"]) for x in train_items}),
        "test_case_ids": sorted({str(x["case_id"]) for x in test_items}),
        "train_sample_ids": sorted(str(x["sample_id"]) for x in train_items),
        "test_sample_ids": sorted(str(x["sample_id"]) for x in test_items),
        "model_kwargs": {
            "feature_dim": int(feature_dim),
            "image_pooling_type": IMAGE_POOLING_TYPE,
            "visit_fusion_type": VISIT_FUSION_TYPE,
            "attn_hidden_dim": ATTN_HIDDEN_DIM,
            "fusion_hidden_dim": FUSION_HIDDEN_DIM,
            "gru_hidden_dim": GRU_HIDDEN_DIM,
            "gru_num_layers": GRU_NUM_LAYERS,
            "dropout": DROPOUT,
            "bidirectional": GRU_BIDIRECTIONAL,
            "temporal_backbone": TEMPORAL_BACKBONE,
            "input_modality": INPUT_MODALITY,
            "use_explicit_time": USE_EXPLICIT_TIME,
        },
        "training": {"lr": LR, "weight_decay": WEIGHT_DECAY,
                     "batch_size": BATCH_SIZE, "use_pos_weight": USE_POS_WEIGHT,
                     "max_epochs": NUM_EPOCHS, "inner_n_splits": INNER_N_SPLITS,
                     "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                     "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA},
        "temporal_experiment": TEMPORAL_EXPERIMENT,
        "temporal_min_visits": TEMPORAL_MIN_VISITS,
        "temporal_shuffle_seed": TEMPORAL_SHUFFLE_SEED,
        "raw_root": str(RAW_ROOT),
        "fetalclip_config_path": str(FETALCLIP_CONFIG_PATH),
        "fetalclip_weight_path": str(FETALCLIP_WEIGHT_PATH),
        "fetalclip_model_name": FETALCLIP_MODEL_NAME,
        "fetalclip_frozen": True,
        "inner_epoch_selection": selection_rows,
        "torch_version": str(torch.__version__),
    }
    save_final_checkpoint(model, checkpoint_path, metadata)
    print(f"Saved final outer-fold checkpoint: {checkpoint_path}")

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
                    "input_modality": INPUT_MODALITY,
                    "temporal_experiment": TEMPORAL_EXPERIMENT,
                    "temporal_shuffle_seed": (
                        TEMPORAL_SHUFFLE_SEED
                        if TEMPORAL_EXPERIMENT == "shuffled"
                        else ""
                    ),
                    "test_case_ids": "|".join(test_case_ids),
                    "selected_epoch": selected_epoch,
                    "sample_id": item["sample_id"],
                    "case_id": item["case_id"],
                    "side": item["side"],
                    "label": int(item["label"]),
                    "num_visits": int(item["num_visits"]),
                    "visit_ids_chronological": "|".join(
                        item.get("chronological_visit_ids", [])
                    ),
                    "visit_ids_used": "|".join(
                        [str(v.get("visit_id", "")) for v in item["visits"]]
                    ),
                    "prob": prob,
                    "logit": logit_value,
                    "pred": int(prob >= 0.5),
                    "image_attention_json": attention_json,
                }
            )

    pd.DataFrame(fold_results).to_csv(
        RESULT_DIR / fold_id / "predictions.csv", index=False, encoding="utf-8-sig"
    )
    return fold_results, selection_rows, selected_epoch


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
        "BrierScore",
        "ECE",
    ]

    summary = {
        "temporal_experiment": TEMPORAL_EXPERIMENT,
        "temporal_shuffle_seed": (
            TEMPORAL_SHUFFLE_SEED
            if TEMPORAL_EXPERIMENT == "shuffled"
            else ""
        ),
        "num_folds": len(fold_metric_rows),
    }
    fold_metric_df = pd.DataFrame(fold_metric_rows)

    for metric_name in metric_names:
        values = pd.to_numeric(fold_metric_df[metric_name], errors="coerce").dropna()
        summary[f"{metric_name}_mean"] = float(values.mean()) if len(values) > 0 else None
        summary[f"{metric_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None

    for count_name in ["TN", "FP", "FN", "TP"]:
        summary[f"{count_name}_sum"] = int(fold_metric_df[count_name].sum())

    return summary



def send_wxpusher_message(
    title: str,
    content: str,
    timeout: int = 15,
) -> bool:
    """
    使用 WxPusher 向手机发送消息。

    返回：
        True：请求发送成功
        False：发送失败，但不会导致训练程序再次报错
    """
    api_url = "https://wxpusher.zjiecode.com/api/send/message"

    message = (
        f"## {title}\n\n"
        f"{content}\n\n"
        f"发送时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    payload = {
        "appToken": APP_TOKEN,
        "content": message,
        "contentType": 3,  # Markdown
        "uids": [UID],
    }

    try:
        response = requests.post(
            api_url,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        result = response.json()

        # WxPusher 正常成功时通常 code 为 1000
        if result.get("code") == 1000:
            print("WxPusher message sent successfully.")
            return True

        print(f"WxPusher returned an error: {result}")
        return False

    except Exception as exc:
        # 推送失败不应影响已经完成的实验结果
        print(f"Failed to send WxPusher message: {exc}")
        return False



def main():
    set_seed(SEED)
    # Do not mix newly trained weights with old predictions or a partial run.
    if RESULT_DIR.exists() and any(RESULT_DIR.iterdir()):
        raise FileExistsError(
            f"Result directory is not empty: {RESULT_DIR}. "
            "Set a new EXP_ID before training; existing results are preserved."
        )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Reading raw data")
    print(f"RAW_ROOT: {RAW_ROOT}")
    print(f"INPUT_MODALITY: {INPUT_MODALITY}")
    print(f"TEMPORAL_EXPERIMENT: {TEMPORAL_EXPERIMENT}")
    if TEMPORAL_EXPERIMENT == "shuffled":
        print(f"TEMPORAL_SHUFFLE_SEED: {TEMPORAL_SHUFFLE_SEED}")
    print(f"USE_EXPLICIT_TIME: {USE_EXPLICIT_TIME}")
    print(f"RESULT_DIR: {RESULT_DIR}")
    print("=" * 80)

    samples = collect_case_side_samples_online(RAW_ROOT)
    print(f"Side-level sample count before temporal cohort filter: {len(samples)}")

    samples, cohort_stats = filter_multivisit_cases(
        samples,
        min_visits=TEMPORAL_MIN_VISITS,
    )
    print(f"Temporal cohort filter: {cohort_stats}")

    samples = apply_temporal_experiment(
        samples,
        experiment=TEMPORAL_EXPERIMENT,
        shuffle_seed=TEMPORAL_SHUFFLE_SEED,
    )
    print(f"Side-level sample count after temporal cohort filter: {len(samples)}")

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
        attach_fetalclip_features_in_memory(samples, extractor, INPUT_MODALITY)
    else:
        raise NotImplementedError(
            "This script expects USE_MEMORY_CACHE=True. "
            "Feature extraction is done once before cross-validation."
        )

    feature_dim = infer_feature_dim(samples, INPUT_MODALITY)
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
    all_epoch_selection_rows = []

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

        fold_results, epoch_selection_rows, selected_epoch = train_one_fold(
            train_items=train_items,
            test_items=test_items,
            feature_dim=feature_dim,
            fold_id=fold_id,
            test_case_ids=test_case_ids,
        )

        all_fold_results.extend(fold_results)
        all_epoch_selection_rows.extend(epoch_selection_rows)

        fold_y_true = [r["label"] for r in fold_results]
        fold_y_prob = [r["prob"] for r in fold_results]
        fold_metrics = compute_binary_metrics(fold_y_true, fold_y_prob, threshold=0.5)
        fold_metrics_row = {
            "fold_id": fold_id,
            "temporal_experiment": TEMPORAL_EXPERIMENT,
            "temporal_shuffle_seed": (
                TEMPORAL_SHUFFLE_SEED
                if TEMPORAL_EXPERIMENT == "shuffled"
                else ""
            ),
            "test_case_ids": "|".join(test_case_ids),
            "train_samples": len(train_items),
            "test_samples": len(test_items),
            "selected_epoch": selected_epoch,
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
    pooled_metrics = {
        "temporal_experiment": TEMPORAL_EXPERIMENT,
        "temporal_shuffle_seed": (
            TEMPORAL_SHUFFLE_SEED
            if TEMPORAL_EXPERIMENT == "shuffled"
            else ""
        ),
        **compute_binary_metrics(y_true, y_prob, threshold=0.5),
    }
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

    epoch_selection_df = pd.DataFrame(all_epoch_selection_rows)
    epoch_selection_save_path = RESULT_DIR / "nested_cv_epoch_selection.csv"
    epoch_selection_df.to_csv(
        epoch_selection_save_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nResults saved:")
    print(f"predictions: {pred_save_path}")
    print(f"fold metrics: {fold_metric_save_path}")
    print(f"metrics mean/std: {metric_save_path}")
    print(f"pooled metrics: {pooled_metric_save_path}")
    print(f"nested CV epoch selection: {epoch_selection_save_path}")

    success_content = (
        f"实验已正常完成。\n\n"
        f"- 输入模态：{INPUT_MODALITY}\n"
        f"- 时间实验：{TEMPORAL_EXPERIMENT}\n"
        f"- 交叉验证：{N_SPLITS} 折\n"
        f"- 样本数量：{len(samples)}\n"
        f"- 病例数量：{len(case_ids)}\n"
        f"- 结果目录：`{RESULT_DIR}`\n\n"
        f"主要结果：\n"
        f"```text\n"
        f"{pooled_metrics}\n"
        f"```"
    )

    send_wxpusher_message(
        title="胎儿肾积水实验运行完成",
        content=success_content,
    )



if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nProgram was manually interrupted.")

        send_wxpusher_message(
            title="胎儿肾积水实验已手动停止",
            content="程序收到 KeyboardInterrupt，通常表示运行过程中按下了 Ctrl+C。",
        )

        raise

    except Exception as exc:
        error_traceback = traceback.format_exc()

        print("\nProgram failed:")
        print(error_traceback)

        # 避免异常信息过长
        max_length = 5000
        if len(error_traceback) > max_length:
            error_traceback = error_traceback[-max_length:]
            error_traceback = (
                "错误信息过长，以下仅显示最后一部分：\n\n"
                + error_traceback
            )

        send_wxpusher_message(
            title="胎儿肾积水实验运行失败",
            content=(
                f"程序运行过程中发生异常。\n\n"
                f"错误类型：`{type(exc).__name__}`\n\n"
                f"错误信息：`{exc}`\n\n"
                f"```text\n"
                f"{error_traceback}\n"
                f"```"
            ),
        )

        # 重新抛出异常，保留终端中的正常报错行为
        raise