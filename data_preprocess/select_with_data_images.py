# -*- coding: utf-8 -*-
"""
使用 few-shot + qwen-vl-plus 将超声图像分成：
1. with_data_images: 含有效测量文字/数值
2. without_data_images: 不含有效测量文字/数值

和旧版本不同：
- 不再把结果统一复制到 OUTPUT_ROOT/positive 和 OUTPUT_ROOT/negative
- 而是在每个原始检查目录下新建：
    with_data_images/
    without_data_images/
- 分类结果总表保存在 TARGET_ROOT/classification_results.csv

目录示例：
原图：
case_0004/1prenatal/202411270182/images/0_0.jpg

分类后：
case_0004/1prenatal/202411270182/with_data_images/0_0.jpg
或
case_0004/1prenatal/202411270182/without_data_images/0_0.jpg
"""

import os
import re
import csv
import json
import time
import base64
import hashlib
import shutil
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI
from PIL import Image, ImageOps

try:
    import pydicom
except ImportError:
    pydicom = None


# =========================
# 1. 配置区
# =========================
API_KEY_ENV = "WXY_LM_API_KEY"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-vl-plus"

EXAMPLES_DIR = Path(r"G:\wxy_codes\Fetal_hydronephrosis\data_preprocess\examples")
TARGET_ROOT = Path(r"G:\wxy_codes\Fetal_hydronephrosis\fetal_hydro_dataset_3_report_合并 copy\raw")

# 是否递归扫描 TARGET_ROOT
RECURSIVE = True

# 只处理这些后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".dcm"}

# few-shot 每类最多使用多少张样例
MAX_POS_EXAMPLES = 8
MAX_NEG_EXAMPLES = 8

# 图像最长边缩放，减少请求体体积
MAX_IMAGE_SIDE = 640
MAX_EXAMPLE_SIDE = 512

# API 重试与节流
MAX_RETRY = 3
SLEEP_SEC = 0.5

# 是否跳过已经处理过的文件（依据 CSV）
SKIP_ALREADY_DONE = True

# 分类结果总表
CSV_PATH = TARGET_ROOT / "classification_results.csv"


# =========================
# 2. 初始化客户端
# =========================
api_key = os.getenv(API_KEY_ENV)
if not api_key:
    raise ValueError(f"未读取到环境变量 {API_KEY_ENV}，请先设置 API Key。")

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL
)


# =========================
# 3. 基础工具函数
# =========================
def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_array_to_uint8(arr):
    arr = arr.astype("float32")
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = arr * 0
    arr = (arr * 255).clip(0, 255).astype("uint8")
    return arr


def load_image_as_pil(img_path: Path) -> Image.Image:
    suffix = img_path.suffix.lower()

    if suffix == ".dcm":
        if pydicom is None:
            raise ImportError("检测到 .dcm 文件，但未安装 pydicom。请先 pip install pydicom")

        ds = pydicom.dcmread(str(img_path))
        arr = ds.pixel_array

        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr

        arr = normalize_array_to_uint8(arr)

        if arr.ndim == 2:
            img = Image.fromarray(arr)
        elif arr.ndim == 3:
            if arr.shape[0] in [3, 4] and arr.shape[-1] not in [3, 4]:
                arr = arr.transpose(1, 2, 0)
            if arr.shape[-1] == 1:
                arr = arr.squeeze(-1)
                img = Image.fromarray(arr)
            else:
                img = Image.fromarray(arr[..., :3])
        else:
            raise ValueError(f"不支持的 DICOM 维度: {arr.shape}")

        return img.convert("RGB")

    return Image.open(img_path).convert("RGB")


def resize_keep_ratio(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / m
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def enhance_for_small_text(img: Image.Image) -> Image.Image:
    img = ImageOps.autocontrast(img)
    return img


def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    bio = BytesIO()
    img.save(bio, format=fmt)
    raw = bio.getvalue()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def image_path_to_data_url(img_path: Path, max_side: int) -> str:
    img = load_image_as_pil(img_path)
    img = resize_keep_ratio(img, max_side=max_side)
    img = enhance_for_small_text(img)
    return pil_to_data_url(img, fmt="PNG")


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def extract_json(text: str) -> Dict[str, Any]:
    text = strip_code_fence(text)
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))

    raise ValueError(f"无法从模型输出解析 JSON：{text}")


# =========================
# 4. few-shot 样例读取
# =========================
def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def check_examples(examples_dir: Path):
    pos_dir = examples_dir / "positive"
    neg_dir = examples_dir / "negative"

    pos_files = list_images(pos_dir)
    neg_files = list_images(neg_dir)

    print(f"[样例] positive: {len(pos_files)} 张")
    print(f"[样例] negative: {len(neg_files)} 张")

    pos_names = {p.name for p in pos_files}
    neg_names = {p.name for p in neg_files}
    overlap_names = sorted(pos_names & neg_names)
    if overlap_names:
        print("[警告] positive 和 negative 中存在同名文件：")
        for x in overlap_names:
            print("  -", x)

    pos_md5 = {}
    neg_md5 = {}
    for p in pos_files:
        try:
            pos_md5[p] = md5_of_file(p)
        except Exception:
            pass
    for p in neg_files:
        try:
            neg_md5[p] = md5_of_file(p)
        except Exception:
            pass

    dup_pairs = []
    for p1, h1 in pos_md5.items():
        for p2, h2 in neg_md5.items():
            if h1 == h2:
                dup_pairs.append((p1.name, p2.name))

    if dup_pairs:
        print("[警告] positive 和 negative 中存在内容完全相同的图片：")
        for a, b in dup_pairs:
            print(f"  - positive/{a}  <->  negative/{b}")

    return pos_files[:MAX_POS_EXAMPLES], neg_files[:MAX_NEG_EXAMPLES]


# =========================
# 5. 构造 few-shot 提示
# =========================
def build_prompt() -> str:
    return """
你是“胎儿超声图像中有效测量文字检测器”。

任务：
根据正负样例和待分类图像，判断待分类图像是否包含“有效测量文字/数值”。

正样本（positive）的定义：
- 图中包含与卡尺测量直接对应的原始测量结果
- 常见形式例如：
  - 距离1 5.06 cm
  - 距离2 2.23 cm
  - 1D 37.5mm
  - 2D 18.7mm
  - 3D 39.8mm
  - 4D 15.9mm

负样本（negative）的定义：
- 只有机器界面文字、日期、时间、孕周、Hz、MI、TIS、探头参数
- 只有深度尺数字
- 只有器官图像本身，没有测量结果框
- 只有平均值/百分比，但没有原始测量结果

注意：
- 这里的“有文字”不是指任意文字，而是指“有效测量文字/数值”
- 不要把日期、时间、深度尺、机器参数当成 positive
- 只输出 JSON，不要输出 markdown，不要输出额外解释

输出格式必须严格为：
{
  "label": "positive 或 negative",
  "confidence": 0.0,
  "reason": "简短中文说明，不超过40字"
}
""".strip()


def build_fewshot_content(
    pos_examples: List[Path],
    neg_examples: List[Path],
    target_img: Path
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    content.append({"type": "text", "text": build_prompt()})

    for i, p in enumerate(pos_examples, start=1):
        content.append({
            "type": "text",
            "text": f"下面是正样例 positive #{i}：这张图包含有效测量文字/数值。"
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": image_path_to_data_url(p, max_side=MAX_EXAMPLE_SIDE)}
        })

    for i, p in enumerate(neg_examples, start=1):
        content.append({
            "type": "text",
            "text": f"下面是负样例 negative #{i}：这张图不包含有效测量文字/数值。"
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": image_path_to_data_url(p, max_side=MAX_EXAMPLE_SIDE)}
        })

    content.append({
        "type": "text",
        "text": f"现在请判断这张待分类图像（文件名：{target_img.name}）。"
    })
    content.append({
        "type": "image_url",
        "image_url": {"url": image_path_to_data_url(target_img, max_side=MAX_IMAGE_SIDE)}
    })

    return content


# =========================
# 6. 调用模型分类
# =========================
def classify_one_image(
    img_path: Path,
    pos_examples: List[Path],
    neg_examples: List[Path]
) -> Dict[str, Any]:
    content = build_fewshot_content(pos_examples, neg_examples, img_path)

    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个严格输出 JSON 的超声图像二分类助手。"
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )

            raw = resp.choices[0].message.content
            if isinstance(raw, list):
                text = ""
                for part in raw:
                    if isinstance(part, dict):
                        text += part.get("text", "")
                    else:
                        text += str(part)
            else:
                text = str(raw)

            data = extract_json(text)

            label = str(data.get("label", "negative")).strip().lower()
            if label not in {"positive", "negative"}:
                if "pos" in label or "有" in label:
                    label = "positive"
                else:
                    label = "negative"

            try:
                confidence = float(data.get("confidence", 0.0))
            except Exception:
                confidence = 0.0

            reason = str(data.get("reason", "")).strip()

            return {
                "file_name": img_path.name,
                "label": label,
                "confidence": confidence,
                "reason": reason,
                "status": "ok"
            }

        except Exception as e:
            if attempt == MAX_RETRY:
                return {
                    "file_name": img_path.name,
                    "label": "negative",
                    "confidence": 0.0,
                    "reason": f"API失败: {repr(e)}",
                    "status": "error"
                }
            time.sleep(1.5 * attempt)

    return {
        "file_name": img_path.name,
        "label": "negative",
        "confidence": 0.0,
        "reason": "未知错误",
        "status": "error"
    }


# =========================
# 7. 扫描目标图像
# =========================
def gather_target_images(root: Path) -> List[Path]:
    files: List[Path] = []

    if RECURSIVE:
        candidates = root.rglob("*")
    else:
        candidates = root.iterdir()

    for p in candidates:
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue

        p_str = str(p).lower()

        # 只处理原始 images 下的图，跳过新生成目录
        if "with_data_images" in p_str or "without_data_images" in p_str:
            continue

        # 为了稳妥，只收集路径中包含 images 的图片
        if "images" not in p_str:
            continue

        if str(EXAMPLES_DIR).lower() in p_str:
            continue

        files.append(p)

    return sorted(files)


def load_done_set(csv_path: Path) -> set:
    done = set()
    if not csv_path.exists():
        return done

    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row["source_path"])
    except Exception:
        pass
    return done


# =========================
# 8. 输出到原检查目录
# =========================
def get_exam_dir_from_source(src: Path) -> Path:
    """
    源图形如：
    case_xxx/phase/exam_id/images/xxx.jpg
    返回：
    case_xxx/phase/exam_id
    """
    if src.parent.name.lower() != "images":
        raise ValueError(f"源图片不在 images 目录下: {src}")
    return src.parent.parent


def copy_to_exam_output(src: Path, label: str):
    exam_dir = get_exam_dir_from_source(src)

    with_dir = exam_dir / "with_data_images"
    without_dir = exam_dir / "without_data_images"
    safe_mkdir(with_dir)
    safe_mkdir(without_dir)

    # 先清除旧分类，避免重复跑后同一张图同时存在于两个目录
    old_with = with_dir / src.name
    old_without = without_dir / src.name

    if old_with.exists():
        old_with.unlink()
    if old_without.exists():
        old_without.unlink()

    if label == "positive":
        dst = with_dir / src.name
    else:
        dst = without_dir / src.name

    shutil.copy2(src, dst)


def append_csv(csv_path: Path, row: Dict[str, Any]):
    file_exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_path",
                "exam_dir",
                "file_name",
                "label",
                "confidence",
                "reason",
                "status"
            ]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# =========================
# 9. 主函数
# =========================
def main():
    pos_examples, neg_examples = check_examples(EXAMPLES_DIR)
    if not pos_examples or not neg_examples:
        raise ValueError("positive 或 negative 样例为空，无法进行 few-shot 分类。")

    target_images = gather_target_images(TARGET_ROOT)
    print(f"[目标图像] 共 {len(target_images)} 张")

    done_set = load_done_set(CSV_PATH) if SKIP_ALREADY_DONE else set()

    for idx, img_path in enumerate(target_images, start=1):
        src_str = str(img_path)

        if SKIP_ALREADY_DONE and src_str in done_set:
            print(f"[跳过] ({idx}/{len(target_images)}) {img_path.name}")
            continue

        print(f"[处理中] ({idx}/{len(target_images)}) {img_path}")

        result = classify_one_image(img_path, pos_examples, neg_examples)

        exam_dir = get_exam_dir_from_source(img_path)
        copy_to_exam_output(img_path, result["label"])

        append_csv(CSV_PATH, {
            "source_path": src_str,
            "exam_dir": str(exam_dir),
            "file_name": img_path.name,
            "label": result["label"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "status": result["status"]
        })

        time.sleep(SLEEP_SEC)

    print("全部处理完成。")


if __name__ == "__main__":
    main()