import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import RAW_ROOT, FEATURE_DIR, DEVICE
from utils.clip_loader import load_fetalclip
from utils.seed import set_seed
from datasets.build_index import build_visit_index


@torch.no_grad()
def extract_visit_features(samples, model, preprocess, tokenizer, save_dir, device="cuda"):
    os.makedirs(save_dir, exist_ok=True)
    all_records = []

    for sample in tqdm(samples, desc="Extracting features"):
        case_id = sample["case_id"]
        label = sample["label"]
        visit_id = sample["visit_id"]
        image_paths = sample["image_paths"]
        report_path = sample["report_path"]

        # 1) 图像特征：每张图先编码，再平均池化
        imgs = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img = preprocess(img)
            imgs.append(img)

        imgs = torch.stack(imgs, dim=0).to(device)   # [N, C, H, W]
        img_feats = model.encode_image(imgs)         # [N, D]
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        visit_img_feat = img_feats.mean(dim=0)       # [D]
        visit_img_feat = visit_img_feat / visit_img_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # 2) 文本特征
        with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
            report_text = f.read().strip()

        if len(report_text) == 0:
            report_text = "empty report"

        text_tokens = tokenizer([report_text]).to(device)
        text_feat = model.encode_text(text_tokens).squeeze(0)   # [D]
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # 3) 拼接
        fused_feat = torch.cat([visit_img_feat, text_feat], dim=-1).cpu()   # [2D]

        # 4) 保存
        feature_name = f"{case_id}__{visit_id}.pt"
        feature_path = os.path.join(save_dir, feature_name)

        save_obj = {
            "case_id": case_id,
            "visit_id": visit_id,
            "label": label,
            "phase": sample["phase"],
            "exam_date": sample["exam_date"],
            "image_feature": visit_img_feat.cpu(),
            "text_feature": text_feat.cpu(),
            "fused_feature": fused_feat,
            "report_text": report_text,
            "image_paths": image_paths,
        }
        torch.save(save_obj, feature_path)

        all_records.append({
            "case_id": case_id,
            "visit_id": visit_id,
            "label": label,
            "phase": sample["phase"],
            "exam_date": sample["exam_date"],
            "feature_path": feature_path,
        })

    return all_records


def main():
    set_seed(42)
    os.makedirs(FEATURE_DIR, exist_ok=True)

    print("1) 构建检查级索引...")
    samples = build_visit_index(RAW_ROOT)
    print(f"共找到 {len(samples)} 次检查")

    print("2) 加载 FetalCLIP...")
    model, preprocess, tokenizer = load_fetalclip(device=DEVICE)

    print("3) 提取特征...")
    records = extract_visit_features(
        samples=samples,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        save_dir=FEATURE_DIR,
        device=DEVICE
    )

    print("4) 保存索引 CSV...")
    df = pd.DataFrame(records)
    csv_path = os.path.join(FEATURE_DIR, "visit_feature_index.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("已保存:", csv_path)


if __name__ == "__main__":
    main()