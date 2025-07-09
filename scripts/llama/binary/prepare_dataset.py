import pandas as pd
from datasets import Dataset
from pathlib import Path
import sys

# Add project root to path to enable imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from scripts.llama.common.prompts import format_prompt_binary_sdoh

# === Load data ===
train_path = project_root / "data/processed/train-test/train_set.csv"
val_path = project_root / "data/processed/train-test/val_set.csv"

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

# === Map completions to binary labels ===
binary_label = lambda c: "<LIST>NoSDoH</LIST>" if "NoSDoH" in c else "<LIST>SDoH</LIST>"
train_df["completion"] = train_df["completion"].map(binary_label)
val_df["completion"] = val_df["completion"].map(binary_label)

# === Format prompts ===
train_df["text"] = train_df.apply(lambda row: format_prompt_binary_sdoh(row["Sentence"], row["completion"]), axis=1)
val_df["text"] = val_df.apply(lambda row: format_prompt_binary_sdoh(row["Sentence"], row["completion"]), axis=1)

# === Convert to Hugging Face datasets ===
train_dataset = Dataset.from_pandas(train_df[["text", "completion"]])
val_dataset = Dataset.from_pandas(val_df[["text", "completion"]])

print("✅ Dataset prepared")
print(train_df["completion"].value_counts())