#!/usr/bin/env python3
"""
Batch processing script for SDoH extraction from referral notes

Usage:
    python batch_process_notes.py --model_name "meta-llama/Llama-3.1-8B-Instruct" \
                                 --prompt_type "five_shot_basic" \
                                 --level 1 \
                                 --batch_size 10 \
                                 --start_index 0 \
                                 --debug
"""

import argparse
import pandas as pd
import torch
from pathlib import Path
import sys
import json

# Add the project root to the path so we can import our modules
sys.path.append('.')

from utils.SDoH_classification_helpers import SDoHExtractor
from utils.batch_processing_helpers import BatchProcessor
from utils.model_helpers import load_instruction_model


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the referral notes dataset
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        DataFrame with referral notes
    """
    print(f"Loading data from: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records")
        
        # Check if the expected column exists
        expected_col = "Referral Notes (depersonalised)"
        if expected_col not in df.columns:
            print(f"Warning: Column '{expected_col}' not found.")
            print(f"Available columns: {list(df.columns)}")
        
            raise ValueError("Could not find referral notes column")
        
        return df, expected_col
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Batch process referral notes for SDoH extraction")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="Name or path of the model to use")
    parser.add_argument("--prompt_type", type=str, default="zero_shot_detailed",
                       choices=["zero_shot_basic", "zero_shot_detailed", "five_shot_basic", "five_shot_detailed"],
                       help="Type of prompt to use")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2],
                       help="Classification level (1 or 2)")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="data/processed/BRC_referrals_cleaned.csv",
                       help="Path to the referral notes CSV file")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Number of notes to process")
    parser.add_argument("--start_index", type=int, default=0,
                       help="Starting index in the dataset")
    
    # Processing arguments
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (includes prompts and raw responses)")
    parser.add_argument("--output_dir", type=str, default="results/batch_results",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (handled automatically by model_helpers)")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Load data
        df, note_column = load_data(args.data_path)
        
        # Validate start_index and batch_size
        if args.start_index >= len(df):
            raise ValueError(f"start_index ({args.start_index}) is >= dataset size ({len(df)})")
        
        effective_batch_size = min(args.batch_size, len(df) - args.start_index)
        print(f"Will process {effective_batch_size} notes starting from index {args.start_index}")
        
        # Load model and tokenizer using your helper
        tokenizer, model = load_instruction_model(args.model_name)
        
        if model is None or tokenizer is None:
            raise ValueError(f"Failed to load model: {args.model_name}")
        
        # Create SDoH extractor
        extractor = SDoHExtractor(
            model=model,
            tokenizer=tokenizer,
            prompt_type=args.prompt_type,
            level=args.level,
            debug=args.debug
        )
        
        # Create batch processor
        processor = BatchProcessor(output_dir=args.output_dir)
        
        # Process batch
        print(f"\nStarting batch processing...")
        print(f"Model: {args.model_name}")
        print(f"Prompt type: {args.prompt_type}")
        print(f"Level: {args.level}")
        print(f"Debug: {args.debug}")
        print("-" * 50)
        
        batch_results = processor.process_batch(
            df=df,
            extractor=extractor,
            note_column=note_column,
            batch_size=effective_batch_size,
            start_index=args.start_index
        )
        
        # Save results
        results_file = processor.save_results(batch_results)
        
        # Print summary
        print("\n" + "="*50)
        print("BATCH PROCESSING COMPLETE")
        print("="*50)
        
        metadata = batch_results["metadata"]
        stats = batch_results["summary_stats"]
        
        print(f"Batch ID: {metadata['batch_id']}")
        print(f"Processing time: {metadata['total_processing_time']:.2f} seconds")
        print(f"Notes processed: {stats['total_notes']}")
        print(f"Total sentences: {stats['total_sentences']}")
        print(f"Sentences with SDoH: {stats['sentences_with_sdoh']}")
        print(f"SDoH detection rate: {stats['sentences_with_sdoh']/stats['total_sentences']:.2%}")
        print(f"Unique factors found: {len(stats['unique_factors_found'])}")
        print(f"Results saved to: {results_file}")
        
        if stats['unique_factors_found']:
            print(f"\nTop factors found:")
            sorted_factors = sorted(stats['factor_frequencies'].items(), key=lambda x: x[1], reverse=True)
            for factor, count in sorted_factors[:5]:
                print(f"  {factor}: {count}")
        
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()