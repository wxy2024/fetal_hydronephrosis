import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "v1_train_5fold_online_attn_gru.py"
LOG_DIR = PROJECT_ROOT / "outputs" / "modality_ablation_logs"
MODALITIES = ("multimodal", "image_only", "text_only")
STOP_ON_ERROR = True


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    for modality in MODALITIES:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"{timestamp}_{modality}.log"
        env = os.environ.copy()
        env["INPUT_MODALITY"] = modality
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        print("=" * 80)
        print(f"Starting modality ablation: {modality}")
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
            failures.append((modality, return_code))
            print(f"Experiment failed: {modality}, return code={return_code}")
            if STOP_ON_ERROR:
                break
        else:
            print(f"Experiment completed: {modality}")

    if failures:
        details = ", ".join(f"{name}({code})" for name, code in failures)
        raise SystemExit(f"Modality ablation failed: {details}")

    print("All modality ablations completed successfully.")


if __name__ == "__main__":
    main()
