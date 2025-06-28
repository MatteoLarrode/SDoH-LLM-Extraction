#!/usr/bin/env python3
"""
Batch processing script for SDoH extraction with timing and memory monitoring

Usage:
    python batch_process_notes.py --model_name "meta-llama/Llama-3.1-8B-Instruct" \
                                 --prompt_type "five_shot_basic" \
                                 --batch_size 10 \
                                 --start_index 0 \
                                 --optimized
"""

import argparse
import pandas as pd
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append('.')

from src.classification.SDoH_classification_helpers import SDoHExtractor
from src.classification.batch_processing_helpers import BatchSDoHProcessor #, OptimisedBatchSDoHProcessor
from src.classification.model_helpers import load_instruction_model


def load_data(data_path: str) -> tuple[pd.DataFrame, str, str]:
    """Load the referral notes dataset"""
    print(f"Loading data from: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    
    # Check for the expected columns
    note_column = "Referral Notes (depersonalised)"
    case_ref_column = "Case Reference"
    
    missing_cols = []
    if note_column not in df.columns:
        missing_cols.append(note_column)
    if case_ref_column not in df.columns:
        missing_cols.append(case_ref_column)
        
    if missing_cols:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing columns: {missing_cols}")
    
    return df, note_column, case_ref_column


def main():
    parser = argparse.ArgumentParser(description="Enhanced batch process referral notes for SDoH extraction")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="Name or path of the model to use")
    parser.add_argument("--prompt_type", type=str, default="zero_shot_detailed",
                       choices=["zero_shot_basic", "zero_shot_detailed", "five_shot_basic", "five_shot_detailed"],
                       help="Type of prompt to use")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="data/processed/BRC_referrals_cleaned.csv",
                       help="Path to the referral notes CSV file")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Number of notes to process")
    parser.add_argument("--start_index", type=int, default=0,
                       help="Starting index in the dataset")
    
    # Processing arguments (to be added when model was optimised)
    # parser.add_argument("--optimized", action="store_true",
    #                    help="Use optimized processor with checkpointing")
    
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/batch_results",
                       help="Directory to save results")
    parser.add_argument("--output_filename", type=str, default=None,
                       help="Custom output filename (optional)")
    
    args = parser.parse_args()
    
    try:
        # Load data
        df, note_column, case_ref_column = load_data(args.data_path)
        
        # Validate indices
        if args.start_index >= len(df):
            raise ValueError(f"start_index ({args.start_index}) >= dataset size ({len(df)})")
        
        effective_batch_size = min(args.batch_size, len(df) - args.start_index)
        end_index = args.start_index + effective_batch_size - 1
        
        print(f"Processing notes {args.start_index}-{end_index} ({effective_batch_size} total)")
        
        # Load model
        print(f"Loading model: {args.model_name}")
        tokenizer, model = load_instruction_model(args.model_name)
        if model is None or tokenizer is None:
            raise ValueError(f"Failed to load model: {args.model_name}")
        
        # Create extractor
        extractor = SDoHExtractor(
            model=model,
            tokenizer=tokenizer,
            prompt_type=args.prompt_type,
            debug=False
        )
        
        # # Choose processor based on mode
        # if args.optimized:
        #     print(f"Using OPTIMIZED processor with checkpointing every {args.checkpoint_every} notes")
        #     processor = OptimisedBatchSDoHProcessor(
        #         output_dir=args.output_dir,
        #         checkpoint_every=args.checkpoint_every
        #     )
            
        #     # Process with optimized pipeline
        #     results_df = processor.process_batch_optimized(
        #         df=df,
        #         extractor=extractor,
        #         note_column=note_column,
        #         case_ref_column=case_ref_column,
        #         start_index=args.start_index,
        #         batch_size=effective_batch_size,
        #         enable_checkpoints=not args.disable_checkpoints
        #     )
        # else:
        print("Using STANDARD processor")
        processor = BatchSDoHProcessor(output_dir=args.output_dir)
        
        # Process with standard pipeline
        results_df = processor.process_to_dataframe(
            df=df,
            extractor=extractor,
            note_column=note_column,
            case_ref_column=case_ref_column,
            start_index=args.start_index,
            batch_size=effective_batch_size
        )
        
        # Generate filename if not provided
        if args.output_filename is None:
            model_short = args.model_name.split('/')[-1].replace('-', '_')
            filename = f"{model_short}_{args.prompt_type}_{args.start_index}-{end_index}.csv"
        else:
            filename = args.output_filename
        
        # Save results
        output_path = processor.save_dataframe(results_df, filename)
        
        # Print final summary
        print("\n" + "="*50)
        print("FINAL PROCESSING SUMMARY")
        print("="*50)
        
        total_sentences = len(results_df)
        sentences_with_sdoh = results_df['has_sdoh'].sum()
                
        # print(f"Processor type: {'OPTIMIZED' if args.optimized else 'STANDARD'}")
        print(f"Model: {args.model_name}")
        print(f"Prompt: {args.prompt_type}")
        print(f"Notes processed: {results_df['case_reference'].nunique()}")
        print(f"Total sentences: {total_sentences}")
        print(f"Sentences with SDoH: {sentences_with_sdoh} ({sentences_with_sdoh/total_sentences:.2%})")
        print(f"Average factors per sentence: {results_df['num_factors'].mean():.2f}")
        
        print(f"Results saved to: {output_path}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()