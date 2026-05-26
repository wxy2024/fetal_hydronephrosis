# data/dataset.py
from pathlib import Path
from typing import Dict, List

from configs.config import (
    IMAGE_EXTENSIONS,
    USE_PRENATAL,
    USE_POSTNATAL,
    MAX_REPORT_CHARS,
)
from utils.file_utils import read_json, read_text, list_images, safe_float, get_phase_flag


def is_case_dir(path: Path) -> bool:
    """判断是否为 case_xxxx 文件夹。"""
    return path.is_dir() and path.name.startswith("case_")


def is_visit_dir(path: Path) -> bool:
    """
    判断是否为一次检查文件夹。
    你的检查文件夹通常是类似 202412300188，
    前 8 位是日期。
    """
    if not path.is_dir():
        return False

    name = path.name
    if len(name) < 8:
        return False

    return name[:8].isdigit()


def collect_case_side_samples(raw_root: Path) -> List[Dict]:
    """
    将数据组织为单侧样本。

    每个病例生成两个样本：
    1. case_xxxx_left
    2. case_xxxx_right
    """

    samples = []

    case_dirs = sorted([p for p in raw_root.iterdir() if is_case_dir(p)], key=lambda x: x.name)

    for case_dir in case_dirs:
        case_meta_path = case_dir / "case_meta.json"
        if not case_meta_path.exists():
            print(f"[跳过] 缺少 case_meta.json: {case_dir}")
            continue

        case_meta = read_json(case_meta_path)
        case_id = case_meta.get("case_id", case_dir.name)

        side_label_map = {
            "left": int(case_meta.get("left_label", -1)),
            "right": int(case_meta.get("right_label", -1)),
        }

        for side in ["left", "right"]:
            label = side_label_map[side]

            if label not in [0, 1]:
                print(f"[跳过] 标签异常: case={case_id}, side={side}, label={label}")
                continue

            visits = []

            phase_dirs = []

            if USE_PRENATAL:
                prenatal_dir = case_dir / "1prenatal"
                if prenatal_dir.exists():
                    phase_dirs.append(prenatal_dir)

            if USE_POSTNATAL:
                postnatal_dir = case_dir / "2postnatal"
                if postnatal_dir.exists():
                    phase_dirs.append(postnatal_dir)

            for phase_dir in phase_dirs:
                visit_dirs = sorted(
                    [p for p in phase_dir.iterdir() if is_visit_dir(p)],
                    key=lambda x: x.name
                )

                for visit_dir in visit_dirs:
                    meta_path = visit_dir / "meta.json"
                    if not meta_path.exists():
                        print(f"[警告] 缺少 meta.json: {visit_dir}")
                        continue

                    meta = read_json(meta_path)

                    side_dir = visit_dir / side
                    report_path = side_dir / "report.txt"
                    image_dir = side_dir / "images"

                    report_text = read_text(report_path, default="")
                    if len(report_text) > MAX_REPORT_CHARS:
                        report_text = report_text[:MAX_REPORT_CHARS]

                    image_paths = list_images(image_dir, IMAGE_EXTENSIONS)

                    visit_item = {
                        "case_id": case_id,
                        "side": side,
                        "visit_id": meta.get("visit_id", visit_dir.name),
                        "phase": meta.get("phase", phase_dir.name),
                        "phase_flag": get_phase_flag(meta.get("phase", phase_dir.name)),
                        "gestational_age_days": safe_float(meta.get("gestational_age_days", 0.0)),
                        "exam_date": meta.get("exam_date", ""),
                        "visit_dir": str(visit_dir),
                        "side_dir": str(side_dir),
                        "report_path": str(report_path),
                        "image_dir": str(image_dir),
                        "report_text": report_text,
                        "image_paths": [str(p) for p in image_paths],
                        "num_images": len(image_paths),
                    }

                    visits.append(visit_item)

            # 按 gestational_age_days 排序；如果都为 0，也可以用 visit_id 排序
            visits = sorted(
                visits,
                key=lambda x: (x["gestational_age_days"], x["visit_id"])
            )

            sample = {
                "sample_id": f"{case_id}_{side}",
                "case_id": case_id,
                "side": side,
                "label": label,
                "num_visits": len(visits),
                "visits": visits,
            }

            samples.append(sample)

    return samples