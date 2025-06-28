#!/usr/bin/env python3
import subprocess
import sys
import time
from pathlib import Path

# Configuration
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct", 
    "microsoft/Phi-4-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

PROMPTS = ["five_shot_basic", "five_shot_detailed"]
BATCH_SIZE = 50
START_INDEX = 0
END_INDEX = 200

def run_batch(model, prompt, start_idx, batch_size):
    """Run a single batch processing job"""
    cmd = [
        "python", "scripts/batch_process_notes.py",
        "--model_name", model,
        "--prompt_type", prompt,
        "--batch_size", str(batch_size),
        "--start_index", str(start_idx),
        "--output_dir", "results/comparison_batch"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr}"

def main():
    total_combinations = len(MODELS) * len(PROMPTS)
    current_combination = 0
    
    print(f"Starting comparison batch processing...")
    print(f"Models: {len(MODELS)}, Prompts: {len(PROMPTS)}")
    print(f"Total combinations: {total_combinations}")
    print(f"Processing notes {START_INDEX}-{END_INDEX}")
    
    start_time = time.time()
    
    for model in MODELS:
        for prompt in PROMPTS:
            current_combination += 1
            print(f"\n{'='*60}")
            print(f"Combination {current_combination}/{total_combinations}")
            print(f"Model: {model}")
            print(f"Prompt: {prompt}")
            print(f"{'='*60}")
            
            # Process in chunks
            for start in range(START_INDEX, END_INDEX, BATCH_SIZE):
                current_batch_size = min(BATCH_SIZE, END_INDEX - start)
                
                print(f"Processing chunk: notes {start} to {start + current_batch_size - 1}")
                
                success, output = run_batch(model, prompt, start, current_batch_size)
                
                if not success:
                    print(f"❌ FAILED: {output}")
                    print("Continuing with next combination...")
                    break
                else:
                    print("✅ Success")
            
            elapsed = time.time() - start_time
            remaining = total_combinations - current_combination
            eta = (elapsed / current_combination * remaining) if current_combination > 0 else 0
            
            print(f"Progress: {current_combination}/{total_combinations} "
                  f"({current_combination/total_combinations:.1%})")
            print(f"Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL PROCESSING COMPLETE!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved in: results/comparison_batch/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()