import os
import shutil
import subprocess
import random
from datetime import datetime
from pathlib import Path
import json

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
eval_loss_path = model_output_base / "eval_loss.txt"
script_path = "scripts/roberta/train.py"
model_name = "roberta-base"

# Create/open log file
with open(log_file_path, "a") as log_file:
    log_file.write(f"\n==== Random Search started at {datetime.now()} ====\n")

num_trials = 100
tried_configs = set()
for idx in range(num_trials):
    lr = random.choice(learning_rates)
    bs = random.choice(batch_sizes)
    dropout = random.choice(dropouts)
    frozen = random.choice(frozen_layers)
    config_tuple = (lr, bs, dropout, frozen)
    if config_tuple in tried_configs:
        print(f"[SKIP] Duplicate config {config_tuple}, trying new one...")
        continue
    tried_configs.add(config_tuple)

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

    # Try to parse evaluation summary before cleanup
    summary_path = run_output_dir / "eval_summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
        eval_loss = summary.get("eval_loss", "N/A")
        eval_csv_path = model_output_base / "eval_loss.csv"
        eval_row = {
            "run_name": run_name,
            "learning_rate": lr,
            "batch_size": bs,
            "dropout": dropout,
            "frozen_layers": frozen,
            "eval_loss": eval_loss
        }
        import csv
        file_exists = eval_csv_path.exists()
        with open(eval_csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=eval_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(eval_row)
        summary_line = (
            f"[SUMMARY] Run {idx+1}/{num_trials}: "
            f"lr={lr}, bs={bs}, drop={dropout}, frozen={frozen} -> eval_loss={eval_loss}"
        )
    else:
        summary_line = f"[SUMMARY] Run {idx+1}/{num_trials}: No eval_summary.json found."

    # Log output, errors, and summary
    with open(log_file_path, "a") as log_file:
        log_file.write(f"\n--- Run: {run_name} ---\n")
        log_file.write(result.stdout)
        log_file.write(result.stderr)
        log_file.write(f"{summary_line}\n")
        log_file.write(f"[INFO] Run completed at {datetime.now()}.\n")

    print(summary_line)
    print(f"[INFO] Run complete. Cleaning up model directory: {run_output_dir}")

    # Remove model directory to save space
    if run_output_dir.exists():
        shutil.rmtree(run_output_dir)
        print(f"[INFO] Deleted {run_output_dir}")
    else:
        print(f"[WARNING] Expected output directory {run_output_dir} not found")

print(f"\n[FINISHED] Random search completed. Log saved to {log_file_path}")

# === Train best model using best config found ===
import pandas as pd

# Load all eval losses
eval_df = pd.read_csv(eval_loss_path, names=["run", "eval_loss"])
eval_df = eval_df.dropna()

# Extract best config
best_row = eval_df.loc[eval_df["eval_loss"].astype(float).idxmin()]
best_run_name = best_row["run"]

# Parse config from best_run_name
import re
config_match = re.search(r"lr([0-9.e-]+)_bs(\d+)_drop([0-9.]+)_frozen(\d+)", best_run_name)
if config_match:
    best_lr, best_bs, best_dropout, best_frozen = config_match.groups()

    # Prepare final training command with 10 epochs and final output dir
    final_cmd = [
        "python", script_path,
        "--model_name", model_name,
        "--train_data_file", train_file,
        "--val_data_file", val_file,
        "--model_output_dir", "results/model_training/roberta_binary/best_model",
        "--num_of_epochs", "10",
        "--per_device_train_batch_size", best_bs,
        "--per_device_eval_batch_size", best_bs,
        "--learning_rate", best_lr,
        "--dropout", best_dropout,
        "--num_frozen_layers", best_frozen,
    ]

    print(f"[BEST MODEL] Training best model with config: lr={best_lr}, bs={best_bs}, drop={best_dropout}, frozen={best_frozen}")
    result = subprocess.run(final_cmd, capture_output=True, text=True)

    # Log final training
    with open(log_file_path, "a") as log_file:
        log_file.write("\n=== Final Training of Best Model ===\n")
        log_file.write(result.stdout)
        log_file.write(result.stderr)
        log_file.write(f"[INFO] Completed at {datetime.now()}.\n")

else:
    print("❌ Failed to parse best config from run name.")