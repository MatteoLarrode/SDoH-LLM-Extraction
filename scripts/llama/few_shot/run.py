import os
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from scripts.llama.binary.prepare_dataset import prepare_binary_dataset_infer
from scripts.llama.shared_utils.prompts import build_sdoh_detection_prompt_infer

# Constants
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CACHE_DIR = "/data/resource/huggingface/hub"
RESULTS_PATH = "scripts/llama/few_shot/few_shot_eval_predictions.csv"

# Extract <LIST>...</LIST>
def extract_list_output(output_text):
    start = output_text.find("<LIST>")
    end = output_text.find("</LIST>")
    if start != -1 and end != -1:
        return output_text[start:end+7]
    return "NO_LIST_FOUND"

# Generate response
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

def main():
    # Load base model with 4-bit quant
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        cache_dir=CACHE_DIR,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    # Load test data and create prompts
    test_df = prepare_binary_dataset_infer("data/processed/train-test/test_set.csv")

    # Generate predictions
    predictions = []
    for prompt in tqdm(test_df["prompt"], desc="Generating predictions"):
        output = generate_response(prompt, model, tokenizer)
        prediction = extract_list_output(output)
        predictions.append(prediction)

    test_df["generated_completion"] = predictions

    # Evaluation
    print("\n📊 Classification Report:")
    print(classification_report(
        test_df["completion"],
        test_df["generated_completion"],
        labels=["<LIST>NoSDoH</LIST>", "<LIST>SDoH</LIST>"],
        digits=2
    ))

    # Save results
    test_df[["Sentence", "completion", "generated_completion"]].to_csv(RESULTS_PATH, index=False)
    print(f"\n✅ Few-shot predictions saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
