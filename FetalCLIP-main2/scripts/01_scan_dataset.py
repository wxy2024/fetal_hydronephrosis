# 这个脚本先检查数据结构有没有被正确读取。
# 会返回以下信息：
# 总共有多少个 case
# 总共有多少个单侧样本
# 每个样本有多少个检查时间点
# 每个检查时间点有多少张图像
# 每个 report.txt 是否被读到
# 执行 python scripts/01_scan_dataset.py

# scripts/01_scan_dataset.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config import RAW_ROOT
from data.dataset import collect_case_side_samples


def main():
    samples = collect_case_side_samples(RAW_ROOT)

    print("=" * 80)
    print(f"数据根目录: {RAW_ROOT}")
    print(f"单侧样本数量: {len(samples)}")
    print("=" * 80)

    case_ids = sorted(set([s["case_id"] for s in samples]))
    print(f"病例数量: {len(case_ids)}")
    print(f"病例列表: {case_ids}")
    print("=" * 80)

    for sample in samples:
        print(f"\n样本: {sample['sample_id']}")
        print(f"  case_id: {sample['case_id']}")
        print(f"  side: {sample['side']}")
        print(f"  label: {sample['label']}")
        print(f"  num_visits: {sample['num_visits']}")

        for i, visit in enumerate(sample["visits"]):
            print(f"    [{i + 1}] visit_id: {visit['visit_id']}")
            print(f"        phase: {visit['phase']}")
            print(f"        gestational_age_days: {visit['gestational_age_days']}")
            print(f"        exam_date: {visit['exam_date']}")
            print(f"        num_images: {visit['num_images']}")
            print(f"        report_path: {visit['report_path']}")
            print(f"        image_dir: {visit['image_dir']}")

            report_preview = visit["report_text"].replace("\n", " ")
            if len(report_preview) > 80:
                report_preview = report_preview[:80] + "..."
            print(f"        report_preview: {report_preview}")

    print("\n扫描完成。")


if __name__ == "__main__":
    main()