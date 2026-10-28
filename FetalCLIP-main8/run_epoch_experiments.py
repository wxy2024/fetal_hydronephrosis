import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "v1_train_5fold_online_attn_gru.py"

LOG_DIR = PROJECT_ROOT / "outputs" / "queue_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# True: any failed experiment stops the queue.
# False: continue running the remaining experiments after a failure.
STOP_ON_ERROR = True


def main():
    experiments = [
        (34 + index, num_epochs)
        for index, num_epochs in enumerate(range(5, 46, 5))
    ]

    failed_experiments = []

    for exp_id, num_epochs in experiments:
        print("=" * 80)
        print(f"Starting experiment: EXP_ID={exp_id}, NUM_EPOCHS={num_epochs}")

        env = os.environ.copy()
        env["EXP_ID"] = str(exp_id)
        env["NUM_EPOCHS"] = str(num_epochs)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / (
            f"exp_{exp_id}_epochs_{num_epochs}_{timestamp}.log"
        )
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
                raise RuntimeError("Failed to capture the training process output.")

            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()

            return_code = process.wait()

        if return_code == 0:
            print(f"Experiment completed: EXP_ID={exp_id}")
        else:
            failed_experiments.append((exp_id, num_epochs, return_code))
            print(
                f"Experiment failed: EXP_ID={exp_id}, "
                f"NUM_EPOCHS={num_epochs}, return code={return_code}"
            )
            if STOP_ON_ERROR:
                print("Stopping because STOP_ON_ERROR=True.")
                break

    print("=" * 80)
    print("Experiment queue finished.")
    if failed_experiments:
        print("Failed experiments:")
        for exp_id, num_epochs, return_code in failed_experiments:
            print(
                f"  EXP_ID={exp_id}, NUM_EPOCHS={num_epochs}, "
                f"return code={return_code}"
            )
        raise SystemExit(1)

    print("All experiments completed successfully.")


if __name__ == "__main__":
    main()
