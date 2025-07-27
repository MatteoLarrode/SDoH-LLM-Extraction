import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use nvtop Device 1 (A100)
os.environ["WANDB_MODE"] = "disabled"

import torch
from transformers import RobertaTokenizer, RobertaConfig, Trainer
from tqdm import tqdm

from scripts.roberta.dataset import BinarySDoHDataset, is_sdoh_label
from scripts.roberta.model import RobertaBinaryClassifierWithWeight

from scripts.llama.shared_utils.model import load_lora_llama
from scripts.llama.multilabel_direct.prepare_dataset import prepare_multilabel_dataset_infer, strip_polarity

def run_roberta_binary_inference(test_data_file: str, model_dir: str, pos_weight: float):
    # Load test data
    test_df = pd.read_csv(test_data_file)
    test_df["binary_label"] = test_df["completion"].apply(is_sdoh_label)

    # Load tokenizer and config from trained model directory
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    config = RobertaConfig.from_pretrained(model_dir)

    # Load model with pos_weight
    model = RobertaBinaryClassifierWithWeight.from_pretrained(
        model_dir,
        config=config,
        pos_weight=pos_weight
    )

    # Dataset
    test_dataset = BinarySDoHDataset(test_df, tokenizer)

    # Trainer
    trainer = Trainer(model=model, tokenizer=tokenizer)
    outputs = trainer.predict(test_dataset)

    # Get predictions
    probs = torch.sigmoid(torch.tensor(outputs.predictions)).numpy().flatten()
    y_pred = (probs > 0.4).astype(int) # Updated threshold

    # Add predictions to DataFrame
    test_df["roberta_prob_sdoh"] = probs
    test_df["roberta_pred_sdoh"] = y_pred

    return test_df[["Sentence", "completion", "roberta_pred_sdoh", "roberta_prob_sdoh"]]

def extract_list_output(output_text: str) -> str:
    start = output_text.find("<LIST>")
    end = output_text.find("</LIST>")
    if start != -1 and end != -1:
        return output_text[start:end+7]
    return "NO_LIST_FOUND"

def generate_response(prompt: str, model, tokenizer, max_new_tokens=128) -> str:
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

def run_llama_on_flagged_sentences(df_flagged: pd.DataFrame, model_dir: str, cache_dir: str = "/data/resource/huggingface/hub"):
    # Load model
    model, tokenizer = load_lora_llama(
        base_model_path="meta-llama/Llama-3.1-8B-Instruct",
        adapter_path=model_dir,
        cache_dir=cache_dir,
        device=0  # CUDA_VISIBLE_DEVICES should be set externally
    )

    # Prepare prompts
    df_prompted = prepare_multilabel_dataset_infer(df_flagged.copy())

    # Generate predictions
    predictions = []
    for prompt in tqdm(df_prompted["prompt"], desc="LLaMA predictions"):
        output = generate_response(prompt, model, tokenizer)
        prediction = extract_list_output(output)
        predictions.append(prediction)

    df_prompted["generated_completion"] = predictions
    return df_prompted[["Sentence", "generated_completion"]]

def run_two_step_pipeline(
    test_data_file: str,
    roberta_model_dir: str,
    llama_model_dir: str,
    pos_weight: float,
    output_file: str
):
    # Set correct device
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use nvtop Device 1 (A100)

    # Step 1: RoBERTa
    roberta_outputs = run_roberta_binary_inference(
        test_data_file=test_data_file,
        model_dir=roberta_model_dir,
        pos_weight=pos_weight
    )

    # Step 2: LLaMA
    llama_df = run_llama_on_flagged_sentences(
        df_flagged=roberta_outputs[roberta_outputs["roberta_pred_sdoh"] == 1],
        model_dir=llama_model_dir
    )

    # Merge and fill
    final_df = roberta_outputs.merge(llama_df, on="Sentence", how="left")
    final_df["final_prediction"] = final_df.apply(
        lambda row: row["generated_completion"] if row["roberta_pred_sdoh"] == 1 else "<LIST>NoSDoH</LIST>",
        axis=1
    )

    # Strip polarity from completion in final predictions
    final_df["completion"] = final_df["completion"].apply(strip_polarity)

    # Save
    final_df.to_csv(output_file, index=False)
    print(f"✅ Saved two-step predictions to {output_file}")