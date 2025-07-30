import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use nvtop Device 1 (A100)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.llama.shared_utils.model import load_lora_llama
from scripts.llama.shared_utils.prompts import build_sdoh_multilabel_present_or_not_prompt_infer

import time
import json

def extract_list_output(output_text):
    start = output_text.find("<LIST>")
    end = output_text.find("</LIST>")
    if start != -1 and end != -1:
        return output_text[start:end+7]
    return "NO_LIST_FOUND"

def generate_response(prompt, model, tokenizer, max_new_tokens=128):
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
        base_model_path="meta-llama/Llama-3.1-8B-Instruct",
        adapter_path=args.model_dir,
        cache_dir="/data/resource/huggingface/hub",
        device=0
    )

    df = pd.read_csv(args.dataset_path)
    df = df[df["referral_note"].notnull()].reset_index(drop=True)
    if args.head:
        df = df.head(20)

    from collections import defaultdict

    # Basic sentence splitter
    def split_into_sentences(text):
        return [s.strip() for s in text.split('.') if s.strip()]

    # Store predictions per CAS
    aggregated = defaultdict(list)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    num_sentences = 0
    num_cases = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing cases"):
        num_cases += 1
        case_id = row["case_ref"]
        note = row["referral_note"]
        sentences = split_into_sentences(note)
        sentence_prompts = [build_sdoh_multilabel_present_or_not_prompt_infer(s) for s in sentences]

        sentence_preds = []
        for prompt in sentence_prompts:
            num_sentences += 1
            output = generate_response(prompt, model, tokenizer)
            prediction = extract_list_output(output)
            sentence_preds.extend([
                label.strip() for label in prediction.replace("<LIST>", "").replace("</LIST>", "").split(",") if label.strip() and label.strip() != "NoSDoH"
            ])

        aggregated[case_id].append((note, sorted(set(sentence_preds))))

    # Construct result dataframe
    result_rows = []
    for case_id, entries in aggregated.items():
        note, label_list = entries[0]
        result_rows.append({
            "case_ref": case_id,
            "referral_note": note,
            "generated_labels": f"<LIST>{', '.join(label_list)}</LIST>" if label_list else "<LIST>NoSDoH</LIST>"
        })

    result_df = pd.DataFrame(result_rows)

    end_time = time.time()
    total_runtime = end_time - start_time
    avg_time_per_sentence = total_runtime / num_sentences if num_sentences else 0
    avg_time_per_case = total_runtime / num_cases if num_cases else 0
    max_memory_mb = torch.cuda.max_memory_allocated() / 1e6
    reserved_memory_mb = torch.cuda.max_memory_reserved() / 1e6

    metrics = {
        "total_runtime_sec": total_runtime,
        "cases_processed": num_cases,
        "sentences_processed": num_sentences,
        "avg_time_per_case_sec": avg_time_per_case,
        "avg_time_per_sentence_sec": avg_time_per_sentence,
        "peak_gpu_memory_mb": max_memory_mb,
        "peak_gpu_memory_reserved_mb": reserved_memory_mb
    }

    model_id = Path(args.model_dir).name
    data_id = Path(args.dataset_path).stem
    folder_name = f"{data_id}_{model_id}_predictions"
    output_folder = os.path.join("results/inference", folder_name)
    os.makedirs(output_folder, exist_ok=True)

    result_df.to_csv(os.path.join(output_folder, "predictions.csv"), index=False)

    with open(os.path.join(output_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Inference results saved to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to fine-tuned LoRA model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to CSV with referral notes")
    parser.add_argument("--head", action="store_true", help="Only run inference on the first 20 rows")
    args = parser.parse_args()
    main(args)