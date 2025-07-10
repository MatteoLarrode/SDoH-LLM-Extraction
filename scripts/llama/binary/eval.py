import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from scripts.llama.common.model import load_lora_llama
from scripts.llama.binary.prepare_dataset import prepare_binary_dataset_infer

def extract_list_output(output_text):
    start = output_text.find("<LIST>")
    end = output_text.find("</LIST>")
    if start != -1 and end != -1:
        return output_text[start:end+7]
    return "NO_LIST_FOUND"

def generate_response(prompt, model, tokenizer, max_new_tokens=64):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    return decoded.strip()

def main(args):
    model, tokenizer = load_lora_llama(
        base_model_path=args.model_dir,
        adapter_path=args.model_dir
    )

    test_df = prepare_binary_dataset_infer(args.test_data_file)
    if getattr(args, "head", False):
        test_df = test_df.head(10)

    # Generate predictions
    predictions = []
    for prompt in tqdm(test_df["text"], desc="Generating predictions"):
        output = generate_response(prompt, model, tokenizer)
        prediction = extract_list_output(output)
        predictions.append(prediction)

    test_df["generated_completion"] = predictions

    # Evaluate
    print("\n📊 Classification Report:")
    print(classification_report(
        test_df["completion"],
        test_df["generated_completion"],
        labels=["<LIST>NoSDoH</LIST>", "<LIST>SDoH</LIST>"],
        digits=2
    ))

    # Save results
    results_path = os.path.join(args.model_dir, "eval_predictions.csv")
    test_df[["Sentence", "text", "completion", "generated_completion"]].to_csv(results_path, index=False)
    print(f"\n✅ Predictions saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_file", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--head", action="store_true", help="Evaluate only on the first 10 rows")
    args = parser.parse_args()
    main(args)
