import subprocess
from pathlib import Path
from datetime import datetime

# =========================
# 1. 项目根目录
# =========================
PROJECT_ROOT = Path(r"G:\wxy_codes\Fetal_hydronephrosis\FetalCLIP-main4")

# =========================
# 2. 训练脚本
# =========================
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "v1_train_loocv_online_attn_gru.py"

# =========================
# 3. 配置文件列表
# 按顺序执行
# =========================
CONFIG_LIST = [
    PROJECT_ROOT / "configs" / "exp01_baseline.yaml",
    PROJECT_ROOT / "configs" / "exp02_gru_hidden128.yaml",
    PROJECT_ROOT / "configs" / "exp03_gru_hidden256.yaml",
    PROJECT_ROOT / "configs" / "exp04_no_image.yaml",
    PROJECT_ROOT / "configs" / "exp05_no_text.yaml",
]

# =========================
# 4. 日志文件夹
# =========================
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 5. 是否遇到错误后停止
# True：某个实验报错，后面的不再跑
# False：某个实验报错，继续跑后面的
# =========================
STOP_ON_ERROR = False


def run_one_experiment(config_path: Path):
    exp_name = config_path.stem
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{time_str}_{exp_name}.log"

    cmd = [
        "python",
        str(TRAIN_SCRIPT),
        "--config",
        str(config_path),
    ]

    print("=" * 80)
    print(f"Start experiment: {exp_name}")
    print(f"Config: {config_path}")
    print(f"Log: {log_path}")
    print("=" * 80)

    with open(log_path, "w", encoding="utf-8") as f:
        process = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if process.returncode == 0:
        print(f"[SUCCESS] {exp_name}")
    else:
        print(f"[FAILED] {exp_name}, return code = {process.returncode}")

    return process.returncode


def main():
    failed_experiments = []

    for config_path in CONFIG_LIST:
        if not config_path.exists():
            print(f"[SKIP] Config not found: {config_path}")
            failed_experiments.append(str(config_path))
            if STOP_ON_ERROR:
                break
            continue

        return_code = run_one_experiment(config_path)

        if return_code != 0:
            failed_experiments.append(str(config_path))
            if STOP_ON_ERROR:
                print("Stop because STOP_ON_ERROR=True")
                break

    print("\n" + "=" * 80)
    print("Experiment queue finished.")
    print("=" * 80)

    if failed_experiments:
        print("Failed experiments:")
        for item in failed_experiments:
            print(" -", item)
    else:
        print("All experiments finished successfully.")


if __name__ == "__main__":
    main()