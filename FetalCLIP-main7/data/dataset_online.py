import json
from pathlib import Path
from typing import Dict, List

from configs.config_online_attn import (
    IMAGE_EXTENSIONS,
    USE_PRENATAL,
    USE_POSTNATAL,
    MAX_REPORT_CHARS,
)


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="ignore")


def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def get_phase_flag(phase: str) -> int:
    """
    prenatal -> 0
    postnatal -> 1
    """
    p = str(phase).lower()
    if "post" in p or "2postnatal" in p or "产后" in p:
        return 1
    return 0


def is_case_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("case_")


def is_visit_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    name = path.name
    return len(name) >= 8 and name[:8].isdigit()


def list_images(folder: Path, extensions: List[str]) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []

    exts = {e.lower() for e in extensions}
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]
    )


def collect_side_images(side_dir: Path) -> List[Path]:
    """
    兼容以下几种结构：

    side/images/*.png
    side/with_data_images/*.png
    side/without_data_images/*.png
    side/*.png
    """
    candidate_dirs = [
        side_dir / "images",
        side_dir / "with_data_images",
        side_dir / "without_data_images",
        side_dir,
    ]

    image_paths: List[Path] = []
    seen = set()

    for d in candidate_dirs:
        for p in list_images(d, IMAGE_EXTENSIONS):
            key = str(p.resolve())
            if key not in seen:
                image_paths.append(p)
                seen.add(key)

    return sorted(image_paths, key=lambda x: str(x))


def get_exam_date_from_meta_or_visit(meta: Dict, visit_dir: Path) -> str:
    exam_date = str(meta.get("exam_date", "")).strip()
    if exam_date:
        return exam_date

    # 若 meta 没有 exam_date，则用检查文件夹名前 8 位生成 YYYY-MM-DD
    name = visit_dir.name
    if len(name) >= 8 and name[:8].isdigit():
        return f"{name[:4]}-{name[4:6]}-{name[6:8]}"

    return ""


def collect_case_side_samples_online(raw_root: Path) -> List[Dict]:
    """
    直接从 raw 数据读取病例-侧别-检查-图像路径/文本/meta 信息。
    不读取也不保存 FetalCLIP 的 .pt 特征文件。

    返回：
    [
        {
            "sample_id": "case_xxx_left",
            "case_id": "case_xxx",
            "side": "left",
            "label": 0/1,
            "num_visits": T,
            "visits": [
                {
                    "visit_id": ...,
                    "phase_flag": 0/1,
                    "gestational_age_days": ...,
                    "postnatal_age_days": ...,
                    "exam_date": ...,
                    "report_text": ...,
                    "image_paths": [...]
                }
            ]
        }
    ]
    """
    raw_root = Path(raw_root)

    if not raw_root.exists():
        raise FileNotFoundError(f"RAW_ROOT 不存在: {raw_root}")

    samples: List[Dict] = []

    case_dirs = sorted(
        [p for p in raw_root.iterdir() if is_case_dir(p)],
        key=lambda x: x.name,
    )

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

        phase_dirs = []

        if USE_PRENATAL:
            prenatal_dir = case_dir / "1prenatal"
            if prenatal_dir.exists():
                phase_dirs.append((prenatal_dir, "prenatal"))

        if USE_POSTNATAL:
            postnatal_dir = case_dir / "2postnatal"
            if postnatal_dir.exists():
                phase_dirs.append((postnatal_dir, "postnatal"))

        for side in ["left", "right"]:
            label = side_label_map[side]

            if label not in [0, 1]:
                print(
                    f"[跳过] 标签异常: case={case_id}, "
                    f"side={side}, label={label}"
                )
                continue

            visits: List[Dict] = []

            for phase_dir, default_phase in phase_dirs:
                visit_dirs = sorted(
                    [p for p in phase_dir.iterdir() if is_visit_dir(p)],
                    key=lambda x: x.name,
                )

                for visit_dir in visit_dirs:
                    meta_path = visit_dir / "meta.json"

                    if not meta_path.exists():
                        print(f"[警告] 缺少 meta.json: {visit_dir}")
                        continue

                    meta = read_json(meta_path)

                    phase = meta.get("phase", default_phase)
                    phase_flag = get_phase_flag(phase)

                    side_dir = visit_dir / side
                    report_path = side_dir / "report.txt"

                    report_text = read_text(report_path, default="")
                    if len(report_text) > MAX_REPORT_CHARS:
                        report_text = report_text[:MAX_REPORT_CHARS]

                    image_paths = collect_side_images(side_dir)

                    visit_item = {
                        "case_id": case_id,
                        "side": side,
                        "visit_id": meta.get("visit_id", visit_dir.name),
                        "phase": phase,
                        "phase_flag": phase_flag,
                        "gestational_age_days": safe_float(
                            meta.get("gestational_age_days", 0.0)
                        ),
                        "postnatal_age_days": safe_float(
                            meta.get("postnatal_age_days", 0.0)
                        ),
                        "exam_date": get_exam_date_from_meta_or_visit(
                            meta, visit_dir
                        ),
                        "visit_dir": str(visit_dir),
                        "side_dir": str(side_dir),
                        "report_path": str(report_path),
                        "report_text": report_text,
                        "image_paths": [str(p) for p in image_paths],
                        "num_images": len(image_paths),
                    }

                    visits.append(visit_item)

            # 优先按检查日期排序；日期缺失时按 visit_id 排序。
            visits = sorted(
                visits,
                key=lambda x: (
                    str(x.get("exam_date", "")),
                    str(x.get("visit_id", "")),
                ),
            )

            if len(visits) == 0:
                print(f"[跳过] 无有效检查: case={case_id}, side={side}")
                continue

            sample = {
                "sample_id": f"{case_id}_{side}",
                "case_id": case_id,
                "side": side,
                "label": int(label),
                "num_visits": len(visits),
                "visits": visits,
            }

            samples.append(sample)

    return samples