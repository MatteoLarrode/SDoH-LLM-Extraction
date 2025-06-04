#!/usr/bin/env python3
"""
Script to analyze and compare results from different batch processing runs

Usage:
    python analyze_batch_results.py --results_dir batch_results --output comparison_analysis.json
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import sys
from typing import List

sys.path.append('.')
from utils.batch_processing_helpers import ResultsAnalyzer


def find_result_files(results_dir: str, pattern: str = "*.json") -> List[str]:
    """Find all result files in the directory"""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"Results directory not found: {results_dir}")
    
    files = list(results_path.glob(pattern))
    if not files:
        raise ValueError(f"No result files found in {results_dir}")
    
    return [str(f) for f in files]


def load_and_validate_results(file_path: str) -> dict:
    """Load and validate a results file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Basic validation
        required_keys = ['metadata', 'notes', 'summary_stats']
        if not all(key in data for key in required_keys):
            raise ValueError(f"Invalid result file format: {file_path}")
        
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def convert_numpy_types(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def create_detailed_report(comparison_df: pd.DataFrame, analysis: dict) -> dict:
    """Create a detailed analysis report"""
    report = {
        "overview": {
            "total_experiments": len(analysis),
            "total_sentences_analyzed": int(len(comparison_df)),
            "unique_sentences": int(comparison_df['sentence_id'].nunique()),
            "models_tested": int(comparison_df['model_name'].nunique()),
            "prompt_types_tested": int(comparison_df['prompt_type'].nunique())
        },
        "model_performance": analysis,
        "sentence_level_insights": {},
        "factor_analysis": {}
    }
    
    # Analyze sentences that had different results across models
    sentence_variance = []
    for sentence_id in comparison_df['sentence_id'].unique():
        sentence_data = comparison_df[comparison_df['sentence_id'] == sentence_id]
        if len(sentence_data) > 1:  # Multiple models processed this sentence
            unique_results = sentence_data['sdoh_factors'].apply(lambda x: str(sorted(x))).nunique()
            if unique_results > 1:  # Different results
                sentence_variance.append({
                    "sentence_id": sentence_id,
                    "sentence_text": sentence_data.iloc[0]['sentence_text'],
                    "num_different_results": int(unique_results),
                    "models_tested": int(len(sentence_data))
                })
    
    report["sentence_level_insights"] = {
        "sentences_with_different_results": int(len(sentence_variance)),
        "high_variance_sentences": sorted(sentence_variance, 
                                        key=lambda x: x['num_different_results'], 
                                        reverse=True)[:10]
    }
    
    # Factor analysis across all models
    all_factors = []
    for factors_list in comparison_df[comparison_df['has_sdoh']]['sdoh_factors']:
        all_factors.extend([f for f in factors_list if f != "NoSDoH"])
    
    factor_freq = pd.Series(all_factors).value_counts()
    
    report["factor_analysis"] = {
        "total_factor_mentions": int(len(all_factors)),
        "unique_factors": int(len(factor_freq)),
        "most_common_factors": {k: int(v) for k, v in factor_freq.head(10).to_dict().items()},
        "rare_factors": factor_freq[factor_freq == 1].index.tolist()
    }
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare batch processing results")
    
    parser.add_argument("--results_dir", type=str, default="results/batch_results",
                       help="Directory containing result files")
    parser.add_argument("--output", type=str, default="results/comparison_analyses/comparison_analysis.json",
                       help="Output file for the analysis")
    parser.add_argument("--pattern", type=str, default="*.json",
                       help="File pattern to match result files")
    parser.add_argument("--sentence_id", type=str, default=None,
                       help="Specific sentence ID to analyze in detail")
    
    args = parser.parse_args()
    
    try:
        # Find result files
        result_files = find_result_files(args.results_dir, args.pattern)
        print(f"Found {len(result_files)} result files")
        
        # Load and validate results
        valid_files = []
        for file_path in result_files:
            data = load_and_validate_results(file_path)
            if data:
                valid_files.append(file_path)
                print(f"✓ Loaded: {Path(file_path).name}")
            else:
                print(f"✗ Failed: {Path(file_path).name}")
        
        if not valid_files:
            raise ValueError("No valid result files found")
        
        # Create batch processor to use its comparison methods
        from utils.batch_processing_helpers import BatchProcessor
        processor = BatchProcessor()
        
        # Create comparison dataset
        print("\nCreating comparison dataset...")
        comparison_df = processor.create_comparison_dataset(valid_files)
        print(f"Comparison dataset created with {len(comparison_df)} records")
        
        # Analyze results
        print("Analyzing results...")
        analyzer = ResultsAnalyzer()
        analysis = analyzer.compare_models(comparison_df)
        
        # Create detailed report
        detailed_report = create_detailed_report(comparison_df, analysis)
        
        # Convert numpy types to native Python types
        detailed_report = convert_numpy_types(detailed_report)
        
        # If specific sentence analysis requested
        if args.sentence_id:
            sentence_analysis = analyzer.sentence_level_comparison(comparison_df, args.sentence_id)
            detailed_report["specific_sentence_analysis"] = convert_numpy_types(sentence_analysis)
        
        # Save analysis
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, indent=2, ensure_ascii=False)
        
        print(f"\nAnalysis saved to: {args.output}")
        
        # Print summary
        print("\n" + "="*60)
        print("COMPARISON ANALYSIS SUMMARY")
        print("="*60)
        
        overview = detailed_report["overview"]
        print(f"Total experiments: {overview['total_experiments']}")
        print(f"Unique sentences analyzed: {overview['unique_sentences']}")
        print(f"Models tested: {overview['models_tested']}")
        print(f"Prompt types tested: {overview['prompt_types_tested']}")
        
        print(f"\nModel Performance:")
        for model_config, performance in analysis.items():
            print(f"  {model_config}:")
            print(f"    SDoH detection rate: {performance['sdoh_detection_rate']:.2%}")
            print(f"    Avg factors per sentence: {performance['avg_factors_per_sentence']:.2f}")
            print(f"    Unique factors found: {performance['unique_factors']}")
        
        insights = detailed_report["sentence_level_insights"]
        print(f"\nSentence-level insights:")
        print(f"  Sentences with different results across models: {insights['sentences_with_different_results']}")
        
        if insights['high_variance_sentences']:
            print(f"  Most controversial sentence: '{insights['high_variance_sentences'][0]['sentence_text'][:100]}...'")
        
        factor_analysis = detailed_report["factor_analysis"]
        print(f"\nFactor analysis:")
        print(f"  Total factor mentions: {factor_analysis['total_factor_mentions']}")
        print(f"  Unique factors: {factor_analysis['unique_factors']}")
        print(f"  Most common factor: {list(factor_analysis['most_common_factors'].keys())[0] if factor_analysis['most_common_factors'] else 'None'}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()