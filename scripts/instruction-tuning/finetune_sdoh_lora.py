import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from datasets import Dataset
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# === CLI Argument Parsing ===
parser = argparse.ArgumentParser(description="Fine-tune LLaMA 3.1 with Unsloth for SDoH classification")
parser.add_argument('--sample', action='store_true', help="Use a 50-sample subset for quick training")
args = parser.parse_args()

# === Step 0: Set project root for imports ===
project_root = Path(__file__).resolve().parents[2]  # go up 2 levels from scripts/instruction-tuning/
sys.path.append(str(project_root))

from src.classification.prompt_creation_helpers import create_automated_prompt

# === Step 1: Load and prepare training data ===
print("[INFO] Loading training data...")
train_df = pd.read_csv(project_root / "data/processed/train-test/train_set.csv")

# === Step 2: Load tokenizer and format prompts ===
print("[INFO] Formatting prompts with LLaMA 3.1 chat template...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
train_df["prompt"] = train_df["Sentence"].apply(
    lambda s: create_automated_prompt(s, tokenizer=tokenizer, prompt_type="five_shot_basic")
)

# === Step 3: Prepare Dataset ===
train_df["text"] = train_df.apply(lambda row: f"{row['prompt']}{row['completion']}", axis=1)

if args.sample:
    print("[INFO] Using 50-sample subset for debugging...")
    train_df = train_df.sample(n=50, random_state=42).reset_index(drop=True)

formatted_dataset = Dataset.from_pandas(train_df[["text"]])

# === Step 4: Load model with Unsloth ===
print("[INFO] Loading model with Unsloth...")
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 512,
    dtype = None,
    load_in_4bit = True
)

# === Step 5: Patch model with LoRA ===
print("[INFO] Applying LoRA configuration...")
tokenizer.pad_token = tokenizer.eos_token
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# === Step 6: Define training arguments ===
print("[INFO] Setting up training configuration...")
training_args = TrainingArguments(
    output_dir = "results/finetune_lora_sdoh",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    num_train_epochs = 3,
    save_strategy = "epoch",
    save_total_limit = 2,
    learning_rate = 2e-4,
    bf16 = True,
    report_to = "none",
)

# === Step 7: Run training ===
print("[INFO] Beginning training...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formatted_dataset,
    dataset_text_field = "text",
    max_seq_length = 512,
    packing = True,
    dataset_num_proc = 2,
    args = training_args
)

trainer.train()
print("[INFO] Training complete.")