"""
Script for running two-step SDoH inference (binary RoBERTa + multilabel LLaMA)
on a large dataset of referral notes in batches.

Steps:
1. Split each referral note into sentences.
2. Group all sentences into batches of size N (default 5000).
3. For each batch:
   a. Run binary RoBERTa model to flag sentences with SDoH content.
   b. Run LLaMA model on flagged sentences to predict SDoH types.
4. Save predictions batch-by-batch to avoid memory loss on crash.

Usage:
python infer_adverse_sdoh_twostep.py \
  --referral_path path/to/referrals.csv \
  --roberta_model_dir path/to/roberta_model \
  --llama_model_dir path/to/llama_model \
  --batch_size 5000 \
  --data_id experiment_name
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
import math
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Use nvtop Device 1 (A100)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.multistep_adverse.two_step_pipeline import run_two_step_pipeline

def sentence_splitter(text):
    """
    Splits input text into individual sentences by splitting on periods.
    
    Args:
        text (str): The full referral note.
    
    Returns:
        List[str]: List of non-empty, trimmed sentences.
    """
    # Split on periods and trim whitespace; filter out empty strings
    return [s.strip() for s in str(text).split('.') if s.strip()]

def prepare_batched_csvs(referral_path, output_dir, batch_size):
    """
    Batches the referral DataFrame by unique case_ref, preserving full notes.
    
    Args:
        referral_path (str): Path to CSV file with referral data.
        output_dir (str): Directory to save batch CSVs.
        batch_size (int): Number of cases per batch.
    
    Returns:
        List[str]: List of file paths to the saved batch CSVs.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(referral_path)
    df = df[df["Referral Notes (depersonalised)"].notnull()].reset_index(drop=True)
    
    # Keep only required columns for inference
    df = df[["case_ref", "Referral Notes (depersonalised)"]].copy()
    df = df.rename(columns={"Referral Notes (depersonalised)": "referral_note"})
    df["completion"] = "<LIST>"

    total = len(df)
    num_batches = math.ceil(total / batch_size)

    batch_paths = []
    for i in range(num_batches):
        batch_df = df.iloc[i*batch_size:(i+1)*batch_size]
        batch_path = os.path.join(output_dir, f"batch_{i:03d}.csv")
        batch_df.to_csv(batch_path, index=False)
        batch_paths.append(batch_path)

    return batch_paths

def main(args):
    """
    Main routine to prepare sentence batches and run the two-step model pipeline.
    
    Args:
        args: Command line arguments parsed by argparse.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join("results/inference/full_referrals", f"{args.data_id}_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)

    # Step 1: Prepare sentence batches
    batch_paths = prepare_batched_csvs(args.referral_path, os.path.join(base_output_dir, "batches"), args.batch_size)

    # Step 2: Run two-step inference on each batch
    for i, batch_path in enumerate(batch_paths):
        print(f"\n🚀 Running batch {i+1}/{len(batch_paths)}")
        out_path = os.path.join(base_output_dir, f"predictions_batch_{i:03d}.csv")
        if os.path.exists(out_path):
            print("⏭️ Already exists, skipping.")
            continue
        try:
            run_two_step_pipeline(
                test_data_file=batch_path,
                roberta_model_dir=args.roberta_model_dir,
                llama_model_dir=args.llama_model_dir,
                pos_weight=args.pos_weight,
                output_file=out_path
            )
        except Exception as e:
            print(f"❌ Failed on batch {i}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--referral_path", type=str, required=True, help="Path to input CSV with referral notes")
    parser.add_argument("--roberta_model_dir", type=str, required=True, help="Directory of trained binary RoBERTa model")
    parser.add_argument("--llama_model_dir", type=str, required=True, help="Directory of fine-tuned LLaMA model")
    parser.add_argument("--pos_weight", type=float, default=1.1757, help="Positive class weight for RoBERTa model")
    parser.add_argument("--batch_size", type=int, default=5000, help="Number of sentences per batch")
    parser.add_argument("--data_id", type=str, default="brc_referrals", help="Tag for output folder naming")
    args = parser.parse_args()
    main(args)