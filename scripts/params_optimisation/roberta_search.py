import os
import shutil
import subprocess
import random
from datetime import datetime
from pathlib import Path

# Define random search space
learning_rates = [1e-5, 2e-5, 3e-5, 5e-5, 7e-5]
batch_sizes = [4, 8, 16]
dropouts = [0.0, 0.1, 0.2, 0.3, 0.5]
frozen_layers = [0, 2, 4, 6, 8, 10]

# Fixed parameters
train_file = "data/processed/train-test/train_set.csv"
val_file = "data/processed/train-test/val_set.csv"
model_output_base = Path("results/random_searches/roberta_search_runs")
model_output_base.mkdir(parents=True, exist_ok=True)
log_file_path = model_output_base / "search_log.txt"
script_path = "scripts/roberta/train.py"
model_name = "roberta-base"

# Create/open log file
with open(log_file_path, "a") as log_file:
    log_file.write(f"\n==== Random Search started at {datetime.now()} ====\n")

# Perform random search
num_trials = 50
for idx in range(num_trials):
    lr = random.choice(learning_rates)
    bs = random.choice(batch_sizes)
    dropout = random.choice(dropouts)
    frozen = random.choice(frozen_layers)

    run_name = f"roberta_rand{idx}_lr{lr}_bs{bs}_drop{dropout}_frozen{frozen}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_output_dir = model_output_base / run_name
    print(f"\n[RUN {idx+1}/{num_trials}] Starting: {run_name}")

    # Run training subprocess
    cmd = [
        "python", script_path,
        "--model_name", model_name,
        "--train_data_file", train_file,
        "--val_data_file", val_file,
        "--model_output_dir", str(model_output_base),
        "--num_of_epochs", "5",
        "--per_device_train_batch_size", str(bs),
        "--per_device_eval_batch_size", str(bs),
        "--learning_rate", str(lr),
        "--num_frozen_layers", str(frozen),
        "--max_length", "64",
        "--dropout", str(dropout),
        "--seed", "42",
        "--run_name", run_name
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    with open(log_file_path, "a") as log_file:
        log_file.write(f"\n--- Run: {run_name} ---\n")
        log_file.write(result.stdout)
        log_file.write(result.stderr)
        log_file.write(f"\n[INFO] Run completed at {datetime.now()}.\n")

    print(f"[INFO] Run complete. Cleaning up model directory: {run_output_dir}")

    # Remove model directory to save space
    if run_output_dir.exists():
        shutil.rmtree(run_output_dir)
        print(f"[INFO] Deleted {run_output_dir}")
    else:
        print(f"[WARNING] Expected output directory {run_output_dir} not found")

print(f"\n[FINISHED] Random search completed. Log saved to {log_file_path}")