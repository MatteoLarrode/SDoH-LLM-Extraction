#!/usr/bin/env python3
"""
Annotation evaluation script for SDoH extraction using proper multi-label metrics

Usage:
    python scripts/evaluate_on_annotations.py --model_name "meta-llama/Llama-3.1-8B-Instruct" \
                                     --prompt_type "five_shot_basic" \
                                     --level 1 \
                                     --annotation_data "data/raw/BRC-Data/annotated_BRC_referrals.csv"
"""

import argparse
import pandas as pd
import sys
import os
import time
import psutil
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the path
sys.path.append('.')

from src.classification.SDoH_classification_helpers import SDoHExtractor
from src.classification.model_helpers import load_instruction_model
from utils.evaluation_helpers_lvl1 import (
    calculate_multilabel_metrics,
    print_multilabel_analysis,
    save_evaluation_results
)
from utils.evaluation_helpers_lvl2 import (
    calculate_level2_multilabel_metrics,
    print_level2_multilabel_analysis,
    save_level2_evaluation_results,
    parse_level1_with_adversity_to_level2
)

class AnnotationEvaluator:
    """Evaluator for annotated sentence-level SDoH data with proper multi-label metrics"""
    
    def __init__(self, output_dir: str = "results/annotation_evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = None
        self.process = psutil.Process()
    
    def load_annotation_data(self, annotation_path: str) -> pd.DataFrame:
        """Load and validate annotation data"""
        print(f"Loading annotation data from: {annotation_path}")
        
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        df = pd.read_csv(annotation_path)
        
        # Take first 4 columns and standardize names
        df = df.iloc[:, :4]
        df.columns = ['CAS', 'Sentence', 'Label', 'Adverse']
        
        # Clean data
        df = df.dropna(subset=['Sentence', 'Label'])
        df['Sentence'] = df['Sentence'].astype(str).str.strip()
        df['Label'] = df['Label'].astype(str).str.strip()
        
        # Remove empty sentences
        df = df[df['Sentence'] != '']
        
        print(f"Loaded {len(df)} annotated sentences from {df['CAS'].nunique()} cases")
        print(f"Label distribution:")
        print(df['Label'].value_counts().sort_index())
        
        return df
    
    def process_sentences(self, annotated_df: pd.DataFrame, extractor: SDoHExtractor, 
                         model_name: str, level: int) -> pd.DataFrame:
        """Process all annotated sentences and collect results"""
        
        print(f"\nProcessing {len(annotated_df)} sentences...")
        self.start_time = time.time()
        
        results = []
        
        for idx, row in annotated_df.iterrows():
            sentence_start = time.time()
            
            sentence = str(row['Sentence']).strip()
            if not sentence:
                continue
            
            # Extract SDoH from sentence
            try:
                extraction_result = extractor.extract_from_sentence(sentence)
                factors = extraction_result["sdoh_factors"]
                
                # Convert to string format for comparison (sorted for consistency)
                model_prediction = ", ".join(sorted(factors)) if factors and factors != ["NoSDoH"] else "NoSDoH"
                
                # Create result record
                result = {
                    'cas': row['CAS'],
                    'sentence_number': idx + 1,
                    'original_sentence': sentence,
                    'original_label': row['Label'],
                    'original_adverse': row.get('Adverse', ''),
                    'model_prediction': model_prediction,
                    'model_factors_list': factors,
                    'model_has_sdoh': factors != ["NoSDoH"] and bool(factors),
                    'num_model_factors': len(factors) if factors != ["NoSDoH"] else 0,
                    'model_name': model_name,
                    'prompt_type': extractor.prompt_type,
                    'level': extractor.level,
                    'processing_time_seconds': time.time() - sentence_start
                }

                 # For Level 2, add converted ground truth
                if level == 2:
                    gt_level2 = parse_level1_with_adversity_to_level2(
                        row['Label'], 
                        row.get('Adverse', '')
                    )
                    result['ground_truth_level2'] = ", ".join(gt_level2) if gt_level2 else "NoSDoH"
            
                
                results.append(result)
                
                # Progress indicator
                if (len(results)) % 10 == 0:
                    elapsed = time.time() - self.start_time
                    rate = len(results) / elapsed
                    print(f"Processed {len(results)} sentences ({rate:.1f} sentences/sec)")
                
            except Exception as e:
                print(f"Error processing sentence {idx}: {e}")
                continue
        
        return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SDoH extraction on annotated sentences using proper multi-label metrics")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="Name or path of the model to use")
    parser.add_argument("--prompt_type", type=str, default="zero_shot_detailed",
                       choices=["zero_shot_basic", "zero_shot_detailed", "five_shot_basic", "five_shot_detailed"],
                       help="Type of prompt to use")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2],
                       help="Classification level (1 or 2)")
    
    # Data arguments
    parser.add_argument("--annotation_data", type=str, 
                       default="data/raw/BRC-Data/annotated_BRC_referrals.csv",
                       help="Path to annotated data CSV file")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Number of sentences to sample for evaluation (default: all)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for sampling")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/annotation_evaluation",
                       help="Directory to save results")
    parser.add_argument("--save_mismatches_only", action="store_true",
                       help="Save only mismatched predictions for error analysis")
    
    # Processing arguments
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with detailed logging")
    
    args = parser.parse_args()
    
    try:
        print("="*70)
        if args.level == 2:
            print("SDoH LEVEL 2 EVALUATION (SDoH + ADVERSITY) WITH MULTI-LABEL METRICS")
        else:
            print("SDoH ANNOTATION EVALUATION WITH MULTI-LABEL METRICS")
        print("="*70)
        print(f"Model: {args.model_name}")
        print(f"Prompt: {args.prompt_type} (Level {args.level})")
        print(f"Annotation data: {args.annotation_data}")
        if args.sample_size:
            print(f"Sample size: {args.sample_size}")
        print("="*70)
        
        # Initialize evaluator
        evaluator = AnnotationEvaluator(output_dir=args.output_dir)
        
        # Load annotation data
        annotated_df = evaluator.load_annotation_data(args.annotation_data)
        
        # Sample if requested
        if args.sample_size and args.sample_size < len(annotated_df):
            annotated_df = annotated_df.sample(n=args.sample_size, random_state=args.random_seed)
            print(f"Sampled {len(annotated_df)} sentences for evaluation")
        
        # Load model
        print(f"\nLoading model: {args.model_name}")
        tokenizer, model = load_instruction_model(args.model_name)
        if model is None or tokenizer is None:
            raise ValueError(f"Failed to load model: {args.model_name}")
        
        # Create extractor
        extractor = SDoHExtractor(
            model=model,
            tokenizer=tokenizer,
            prompt_type=args.prompt_type,
            level=args.level,
            debug=args.debug
        )
        
        # Process sentences
        results_df = evaluator.process_sentences(annotated_df, extractor, args.model_name, args.level)
        
        if len(results_df) == 0:
            print("No sentences were successfully processed!")
            return
        
        # Calculate multi-label metrics, based on the level
        if args.level == 2:
            # Use Level 2 evaluation
            metrics, mlb, y_true, y_pred = calculate_level2_multilabel_metrics(results_df)
            
            # Print Level 2 analysis
            print_level2_multilabel_analysis(results_df, metrics, mlb, y_true, y_pred)
            
            # Save Level 2 results
            results_path, metrics_path = save_level2_evaluation_results(
                results_df, metrics, args.model_name, args.prompt_type, args.level, args.output_dir
            )
            
            if results_path:
                print("\n" + "="*70)
                print("LEVEL 2 EVALUATION COMPLETE")
                print("="*70)
                print(f"Key Results:")
                print(f"  Exact Level 2 Match: {metrics['combined_analysis']['exact_level2_match_rate']:.3f}")
                print(f"  SDoH-only Match:     {metrics['combined_analysis']['sdoh_only_match_rate']:.3f}")
                print(f"  Adversity Accuracy:  {metrics['adversity_only']['adversity_accuracy']:.3f}")
                print(f"  Level 2 Micro F1:    {metrics['level2_full']['label_based']['micro_f1']:.3f}")
                print("="*70)
        
        else:
            metrics, mlb, y_true, y_pred = calculate_multilabel_metrics(results_df)
            print_multilabel_analysis(results_df, metrics, mlb, y_true, y_pred)
            
            results_path, metrics_path = save_evaluation_results(
                results_df, metrics, args.model_name, args.prompt_type, args.level, args.output_dir
            )
            
            if results_path:
                print("\n" + "="*70)
                print("EVALUATION COMPLETE")
                print("="*70)
                print(f"Key Results:")
                print(f"  Exact Match Ratio: {metrics['example_based']['exact_match_ratio']:.3f}")
                print(f"  Example-based F1:  {metrics['example_based']['f1_score']:.3f}")
                print(f"  Macro F1:          {metrics['label_based']['macro_f1']:.3f}")
                print(f"  Micro F1:          {metrics['label_based']['micro_f1']:.3f}")
                print("="*70)
            else:
                print("Warning: Results could not be saved properly")
        
        # Filter results if requested
        if args.save_mismatches_only:
            # For multi-label, we'll save non-exact matches
            exact_matches = []
            for i in range(len(y_true)):
                true_labels = set(mlb.inverse_transform([y_true[i]])[0])
                pred_labels = set(mlb.inverse_transform([y_pred[i]])[0])
                if true_labels == pred_labels:
                    exact_matches.append(i)
            
            results_df = results_df.drop(results_df.index[exact_matches])
            print(f"\nSaving only {len(results_df)} non-exact matches for error analysis")
            
    except Exception as e:
        print(f"Error during save/summary: {e}")
        print("Evaluation completed but results may not be saved")

if __name__ == "__main__":
    main()