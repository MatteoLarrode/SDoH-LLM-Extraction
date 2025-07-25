import os
import json
import random
import subprocess
import shutil
from itertools import product
from pathlib import Path

# Add imports for evaluation
import torch
import pandas as pd
from sklearn.metrics import f1_score

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import helper functions and model loading
from scripts.llama.shared_utils.model import load_lora_llama
from scripts.llama.multilabel_direct_adverse.prepare_dataset import prepare_adverse_only_dataset_infer

def extract_list_output(output_text):
    start = output_text.find("<LIST>")
    end = output_text.find("</LIST>")
    if start != -1 and end != -1:
        return output_text[start:end+7]
    return "NO_LIST_FOUND"

def parse_labels(list_string):
    try:
        inner = list_string.replace("<LIST>", "").replace("</LIST>", "").strip()
        if not inner:
            return []
        return [label.strip() for label in inner.split(",")]
    except:
        return []

# Define search space
search_space = {
    "r": [4, 8, 16, 32],
    "lora_alpha": [16, 32, 64, 128],
    "lora_dropout": [0.0, 0.1],
    "learning_rate": [3e-5, 5e-5, 7e-5, 1e-4, 2e-4, 3e-4],
    #"per_device_train_batch_size": [4, 8],
    "per_device_train_batch_size": [8],
}

# Generate all possible combinations
all_combinations = list(product(
    search_space["r"],
    search_space["lora_alpha"],
    search_space["lora_dropout"],
    search_space["learning_rate"],
    search_space["per_device_train_batch_size"],
))

# Filter combinations according to constraints
valid_combinations = []
for r, alpha, dropout, lr, bs in all_combinations:
    if alpha not in [r * k for k in range(2, 9)]:
        continue
    if lr >= 2e-4 and bs > 16:
        continue
    if r <= 8 and dropout != 0.0:
        continue
    valid_combinations.append((r, alpha, dropout, lr, bs))

print(f"Total valid combinations: {len(valid_combinations)}")
# Randomly sample 50 unique combinations from valid ones
sampled_combinations = random.sample(valid_combinations, min(2, len(valid_combinations)))

results = []
run_dir = Path("llama_search_runs")
run_dir.mkdir(exist_ok=True)


# === Stage 1: Training and Evaluation Loop ===
for idx, (r, lora_alpha, lora_dropout, learning_rate, batch_size) in enumerate(sampled_combinations):
    config_name = f"run_{idx}_r{r}_alpha{lora_alpha}_drop{lora_dropout}_lr{learning_rate}_bs{batch_size}"
    output_dir = run_dir / config_name
    output_dir.mkdir(exist_ok=True)
    print(f"\n=== Running config {idx+1}/{len(sampled_combinations)}: {config_name} ===")
    cmd = [
        "python", "scripts/llama/multilabel_direct_adverse/train.py",
        "--train_data_file", "data/processed/train-test/train_set.csv",
        "--val_data_file", "data/processed/train-test/val_set.csv",
        f"--r={r}",
        f"--lora_alpha={lora_alpha}",
        f"--lora_dropout={lora_dropout}",
        f"--learning_rate={learning_rate}",
        "--num_train_epochs=1",
        f"--per_device_train_batch_size={batch_size}",
        f"--model_output_base={str(output_dir)}"
    ]
    print(f"🚀 Launching training subprocess with command:\n{' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Finished training for config: {config_name}")
    except subprocess.CalledProcessError as e:
        print(f"Run failed for config {config_name}: {e}")
        continue

    # Find subdirectory inside output_dir (assumes only one, from timestamped run)
    subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        model_subdir = subdirs[0]
    else:
        print(f"❌ Expected one subdirectory in {output_dir}, found {len(subdirs)}. Skipping.")
        continue
    eval_results_path = model_subdir / "eval_results.json"
    if eval_results_path.exists():
        print(f"🔍 Starting evaluation for config: {config_name}")
        with open(eval_results_path, "r") as f:
            eval_results = json.load(f)
        # Macro-F1 computation using generation on val set
        try:
            val_df = prepare_adverse_only_dataset_infer("data/processed/train-test/val_set.csv")
            print(f"📦 Loading model from adapter path: {model_subdir}")
            model, tokenizer = load_lora_llama(
                base_model_path="meta-llama/Llama-3.1-8B-Instruct",
                adapter_path=str(model_subdir),
                cache_dir="/data/resource/huggingface/hub",
                device=0
            )
            model.eval()
            from tqdm import tqdm
            predictions = []
            print("🧠 Generating predictions on validation prompts...")
            for prompt in tqdm(val_df["prompt"], desc="🔮 Generating validation predictions"):
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda:0")
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                input_len = inputs["input_ids"].shape[1]
                decoded = tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()
                predictions.append(extract_list_output(decoded))
            y_true = val_df["completion"].apply(parse_labels)
            y_pred = pd.Series(predictions).apply(parse_labels)
            from sklearn.preprocessing import MultiLabelBinarizer
            mlb = MultiLabelBinarizer().fit(y_true.tolist() + y_pred.tolist())
            y_true_bin = mlb.transform(y_true)
            y_pred_bin = mlb.transform(y_pred)
            macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro")
            print(f"📊 Macro-F1 score for config {config_name}: {macro_f1}")
            eval_results["macro_f1"] = macro_f1
            with open(eval_results_path, "w") as f:
                json.dump(eval_results, f, indent=2)
        except Exception as e:
            print(f"Error during macro-F1 evaluation for config {config_name}: {e}")
            macro_f1 = None
        val_loss = eval_results.get("eval_loss", None)
        result = {
            "config": {
                "r": r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": batch_size,
            },
            "macro_f1": macro_f1,
            "val_loss": val_loss,
            "eval_results_path": str(eval_results_path),
            "model_subdir": str(model_subdir)
        }
        results.append(result)
        print(f"val_loss: {val_loss}")
        print(f"macro_f1: {macro_f1}")
    else:
        print(f"eval_results.json not found for config {config_name}")

    print(f"🗑️ Deleting model files for config {config_name} to save space...")
    # Delete model files to save space
    for sub in output_dir.iterdir():
        if sub.is_file() or sub.is_dir():
            if sub.name != "eval_results.json":
                if sub.is_dir():
                    shutil.rmtree(sub)
                else:
                    sub.unlink()


# === Stage 2: Result Selection ===
def get_best(results, key, maximize=True):
    filtered = [r for r in results if r[key] is not None]
    if not filtered:
        return None
    return max(filtered, key=lambda x: x[key]) if maximize else min(filtered, key=lambda x: x[key])

best_f1 = get_best(results, "macro_f1", maximize=True)
best_loss = get_best(results, "val_loss", maximize=False)

print("\n=== Search Finished ===")
import pandas as pd
print("💾 Saving all results to search_results.csv")
df = pd.DataFrame(results)
df.to_csv(run_dir / "search_results.csv", index=False)
if best_f1:
    print(f"\nBest macro-F1: {best_f1['macro_f1']:.4f} with config: {best_f1['config']}")
    print(f"eval_results.json: {best_f1['eval_results_path']}")
else:
    print("No valid macro-F1 results found.")
if best_loss:
    print(f"\nBest val_loss: {best_loss['val_loss']:.4f} with config: {best_loss['config']}")
    print(f"eval_results.json: {best_loss['eval_results_path']}")
else:
    print("No valid val_loss results found.")

# === Stage 3: Re-run Best Model ===
if best_f1:
    print("\n=== Re-running best config (highest macro-F1) to save model ===")
    best_cfg = best_f1["config"]
    print("🔁 Re-running best config to save model permanently")
    print(f"Best config: {best_cfg}")
    final_dir = run_dir / "best_model"
    final_dir.mkdir(exist_ok=True)
    rerun_cmd = [
        "python", "scripts/llama/multilabel_direct_adverse/train.py",
        "--train_data_file", "data/processed/train-test/train_set.csv",
        "--val_data_file", "data/processed/train-test/val_set.csv",
        f"--r={best_cfg['r']}",
        f"--lora_alpha={best_cfg['lora_alpha']}",
        f"--lora_dropout={best_cfg['lora_dropout']}",
        f"--learning_rate={best_cfg['learning_rate']}",
        f"--per_device_train_batch_size={best_cfg['per_device_train_batch_size']}",
        f"--model_output_base={str(final_dir)}"
    ]
    subprocess.run(rerun_cmd, check=True)