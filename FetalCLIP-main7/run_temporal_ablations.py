import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "v1_train_5fold_online_attn_gru.py"
LOG_DIR = PROJECT_ROOT / "outputs" / "temporal_ablation_logs"

EXPERIMENTS = [
    ("baseline", None),
    ("first_only", None),
    ("last_only", None),
    ("no_explicit_time", None),
    *(("shuffled", seed) for seed in (42, 43, 44, 45, 46)),
]
STOP_ON_ERROR = True


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    for experiment, shuffle_seed in EXPERIMENTS:
        run_name = experiment
        if shuffle_seed is not None:
            run_name += f"_seed_{shuffle_seed}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"{timestamp}_{run_name}.log"
        env = os.environ.copy()
        env["TEMPORAL_EXPERIMENT"] = experiment
        if shuffle_seed is not None:
            env["TEMPORAL_SHUFFLE_SEED"] = str(shuffle_seed)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        print("=" * 80)
        print(f"Starting temporal ablation: {run_name}")
        print(f"Log: {log_path}")

        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                [sys.executable, str(TRAIN_SCRIPT)],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("Failed to capture training output.")

            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()

            return_code = process.wait()

        if return_code != 0:
            failures.append((run_name, return_code))
            print(f"Experiment failed: {run_name}, return code={return_code}")
            if STOP_ON_ERROR:
                break
        else:
            print(f"Experiment completed: {run_name}")

    if failures:
        details = ", ".join(f"{name}({code})" for name, code in failures)
        raise SystemExit(f"Temporal ablation failed: {details}")

    print("All temporal ablations completed successfully.")


if __name__ == "__main__":
    main()
