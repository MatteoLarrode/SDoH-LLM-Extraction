import argparse
import pandas as pd
import torch
from transformers import (
    RobertaTokenizer, 
    RobertaConfig, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)
from dataset import BinarySDoHDataset, is_sdoh_label
from model import RobertaBinaryClassifierWithWeight
from datetime import datetime
import os
import json
from sklearn.metrics import precision_recall_fscore_support

def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.model_name}_bs{args.per_device_train_batch_size}_lr{args.learning_rate}_{timestamp}"
    output_dir = os.path.join(args.model_output_dir, run_name)

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    # Load and preprocess data
    train_df = pd.read_csv(args.train_data_file)
    val_df = pd.read_csv(args.val_data_file)
    print(f"[INFO] Loaded train shape: {train_df.shape}, validation shape: {val_df.shape}")

    train_df["binary_label"] = train_df["completion"].apply(is_sdoh_label)
    val_df["binary_label"] = val_df["completion"].apply(is_sdoh_label)
    print("[INFO] Label distribution (train):", train_df["binary_label"].value_counts().to_dict())

    # Compute class weights
    num_pos = train_df["binary_label"].sum()
    num_neg = len(train_df) - num_pos
    pos_weight_val = num_neg / num_pos

    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, local_files_only=True)
    train_dataset = BinarySDoHDataset(train_df, tokenizer, max_length=args.max_length)
    val_dataset = BinarySDoHDataset(val_df, tokenizer, max_length=args.max_length)

    sample = train_dataset[0]
    print("[DEBUG] Sample tokenized input keys:", sample.keys())
    print("[DEBUG] Input IDs (first 10):", sample["input_ids"][:10])
    print("[DEBUG] Label:", sample["labels"])

    print(f"[INFO] Using model: {args.model_name}")
    print(f"[INFO] Positive class weight: {pos_weight_val:.4f}")

    # Model
    config = RobertaConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir, local_files_only=True)
    model = RobertaBinaryClassifierWithWeight(config, pos_weight=pos_weight_val, dropout=args.dropout)
    model.roberta = model.roberta.from_pretrained(args.model_name, cache_dir=args.cache_dir, local_files_only=True)

    # Roberta Base has 12 layers, freeze bottom layers as specified
    num_frozen_layers = args.num_frozen_layers
    def freeze_bottom_layers(model, num_frozen_layers=num_frozen_layers):
        for name, param in model.roberta.named_parameters():
            if any(f"encoder.layer.{i}." in name for i in range(num_frozen_layers)):
                param.requires_grad = False

    freeze_bottom_layers(model, num_frozen_layers=num_frozen_layers)
    print(f"[INFO] Frozen bottom {num_frozen_layers} layers of RoBERTa encoder.")
    
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Total parameters: {total:,}")
        print(f"[INFO] Trainable parameters: {trainable:,} ({trainable / total:.2%})")
    
    count_parameters(model)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Convert logits to predictions (binary)
        preds = (torch.tensor(logits) > 0).int().numpy()
        labels = labels.astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        return {"precision": precision, "recall": recall, "f1": f1}

    # Training setup
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_of_epochs,
        logging_dir=os.path.join(output_dir, "logs"),
        # dataloader_num_workers=args.dataloader_num_workers,
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        report_to=[],
        run_name="binary_sdoh_classifier_roberta",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("[INFO] Trainer initialized. Starting training...")
    trainer.train()
    print("[INFO] Training complete.")

    # Evaluate and save summary metrics to JSON
    metrics = trainer.evaluate()
    summary = {
        "run_name": run_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.per_device_train_batch_size,
        "dropout": args.dropout,
        "num_frozen_layers": args.num_frozen_layers,
        "eval_loss": metrics.get("eval_loss"),
        "eval_precision": metrics.get("precision"),
        "eval_recall": metrics.get("recall"),
        "eval_f1": metrics.get("f1"),
        "timestamp": timestamp,
    }

    with open(os.path.join(output_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for binary SDoH classification.")

    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--train_data_file", type=str, required=True, help="Path to the training data file (CSV)")
    parser.add_argument("--val_data_file", type=str, required=True, help="Path to the validation data file (CSV)")
    parser.add_argument("--model_output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--num_of_epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, required=True, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, required=True, help="Evaluation batch size per device")
    # parser.add_argument("--gradient_accumulation_steps", type=int, required=True, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate")
    # parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--num_frozen_layers", type=int, default=10, help="Number of bottom RoBERTa layers to freeze")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--cache_dir", type=str, default="/data/resource/huggingface/hub", help="HuggingFace model cache directory")
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for this run (optional)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for classification head")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)