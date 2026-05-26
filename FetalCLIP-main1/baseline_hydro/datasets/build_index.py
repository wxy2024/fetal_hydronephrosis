import json
from pathlib import Path

def build_visit_index(raw_root):
    raw_root = Path(raw_root)
    samples = []

    for case_dir in sorted(raw_root.glob("case_*")):
        if not case_dir.is_dir():
            continue

        case_meta_path = case_dir / "case_meta.json"
        if not case_meta_path.exists():
            continue

        with open(case_meta_path, "r", encoding="utf-8") as f:
            case_meta = json.load(f)

        case_id = case_meta["case_id"]
        label = int(case_meta["label"])

        for phase_dir in sorted(case_dir.iterdir()):
            if not phase_dir.is_dir():
                continue
            if phase_dir.name.startswith("__"):
                continue

            for visit_dir in sorted(phase_dir.iterdir()):
                if not visit_dir.is_dir():
                    continue

                images_dir = visit_dir / "images"
                report_path = visit_dir / "report.txt"
                meta_path = visit_dir / "meta.json"

                if not images_dir.exists() or not report_path.exists() or not meta_path.exists():
                    continue

                image_paths = sorted([
                    str(p) for p in images_dir.iterdir()
                    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ])

                if len(image_paths) == 0:
                    continue

                with open(meta_path, "r", encoding="utf-8") as f:
                    visit_meta = json.load(f)

                samples.append({
                    "case_id": case_id,
                    "label": label,
                    "phase_dir": phase_dir.name,
                    "visit_id": visit_meta.get("visit_id", visit_dir.name),
                    "phase": visit_meta.get("phase", ""),
                    "exam_date": visit_meta.get("exam_date", ""),
                    "meta_path": str(meta_path),
                    "report_path": str(report_path),
                    "image_paths": image_paths,
                })

    return samples