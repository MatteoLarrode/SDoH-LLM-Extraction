import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use nvtop Device 1 (A100)
from huggingface_hub import hf_hub_download
os.environ["HF_HUB_OFFLINE"] = "1"  # If model fully cached
import os
from datetime import datetime
import argparse
import pandas as pd
from collections import Counter
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from sklearn.metrics import f1_score
import numpy as np
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from scripts.llama.shared_utils.model import load_lora_llama
from scripts.llama.multilabel_direct_adverse.prepare_dataset import prepare_adverse_only_dataset

def main(args):
    # Format output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_name = f"{args.model_name.split('/')[-1]}_bs{args.per_device_train_batch_size}_lr{args.learning_rate}_epochs{args.num_train_epochs}_{timestamp}"
    model_output_dir = os.path.join("results/model_training/llama_multilabel_direct_adverse", output_dir_name)
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"📁 Model output directory: {model_output_dir}")

    # Load tokenizer and set pad token
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded and pad token set.")

    # Load model with LoRA
    model, _ = load_lora_llama(
        args.model_name,
        cache_dir=args.cache_dir,
        device=0,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    print("✅ LoRA model loaded.")

    # Load and preprocess datasets
    train_dataset = prepare_adverse_only_dataset(args.train_data_file)
    val_dataset = prepare_adverse_only_dataset(args.val_data_file)
    print(f"📊 Train set size: {len(train_dataset)}")
    print(f"Train set label counts: {Counter(train_dataset['completion'])}")
    print(f"📊 Validation set size: {len(val_dataset)}")
    print(f"Validation set label counts: {Counter(val_dataset['completion'])}")

    def tokenize(example):
        enc = tokenizer(
            example["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=512)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)
    print("✂️ Tokenization complete.")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(labels, preds, average='macro')
        return {"macro_f1": macro_f1}

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        label_names=["labels"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_dir=os.path.join(model_output_dir, "logs"),
        save_total_limit=1,
        load_best_model_at_end=True,
        bf16=True,
        report_to=[],
        run_name="llama3_lora_multilabel_adverse_sdoh"
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training complete.")
    
    results = trainer.evaluate()
    results_file = os.path.join(model_output_dir, "eval_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print("📊 Evaluation results:", results)

    if not args.search_mode:
        # Save model and tokenizer only if not in search mode
        trainer.save_model(model_output_dir)
        tokenizer.save_pretrained(model_output_dir)
        print("💾 Tokenizer and model saved.")
    print(f"✅ Model saved to {model_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA-8b-Instruct for multi-label *adverse-only* SDoH classification.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--train_data_file", type=str, required=True)
    parser.add_argument("--val_data_file", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="/data/resource/huggingface/hub")
    parser.add_argument("--model_output_base", type=str, default="results/model_training/llama_multilabel_direct_adverse")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--search_mode", action="store_true", help="Disable model saving for hyperparameter search")
    args = parser.parse_args()
    main(args)