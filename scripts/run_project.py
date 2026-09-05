"""Run the complete data, feature, and modeling workflow."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for script_name in ("run_pipeline.py", "build_training_dataset.py", "train_model.py"):
        print(f"Running {script_name}...")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / script_name)], check=True)


if __name__ == "__main__":
    main()
