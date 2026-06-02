# utils/file_utils.py
import json
from pathlib import Path
from typing import Dict, List, Optional


def read_json(json_path: Path) -> Dict:
    """读取 json 文件。"""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(txt_path: Path, default: str = "") -> str:
    """读取文本文件。"""
    if not txt_path.exists():
        return default

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def list_images(image_dir: Path, image_extensions: List[str]) -> List[Path]:
    """列出图像文件。"""
    if not image_dir.exists():
        return []

    image_extensions = [x.lower() for x in image_extensions]

    image_paths = []
    for p in image_dir.iterdir():
        if p.is_file() and p.suffix.lower() in image_extensions:
            image_paths.append(p)

    image_paths = sorted(image_paths, key=lambda x: x.name)
    return image_paths


def ensure_dir(path: Path):
    """创建文件夹。"""
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x, default: float = 0.0) -> float:
    """安全转换为 float。"""
    try:
        return float(x)
    except Exception:
        return default


def get_phase_flag(phase: str) -> float:
    """
    阶段编码：
    prenatal  -> 0
    postnatal -> 1
    其他       -> 0
    """
    phase = str(phase).lower()
    if "post" in phase:
        return 1.0
    return 0.0