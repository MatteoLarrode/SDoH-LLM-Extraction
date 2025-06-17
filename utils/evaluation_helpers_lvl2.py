import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    multilabel_confusion_matrix, 
    classification_report,
    hamming_loss,
    jaccard_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)
from sklearn.preprocessing import MultiLabelBinarizer


def parse_level2_labels_to_list(label_str):
    """Parse Level 2 labels (SDoH-Adversity format) into list"""
    if pd.isna(label_str) or str(label_str).strip() == "" or str(label_str).strip() == "NoSDoH":
        return []
    return [label.strip() for label in str(label_str).split(',')]


def parse_level1_with_adversity_to_level2(sdoh_labels_str, adversity_str):
    """
    Convert Level 1 annotations + adversity column to Level 2 format
    
    Args:
        sdoh_labels_str: "FoodInsecurity, FinancialSituation"
        adversity_str: "adverse, adverse"
    
    Returns:
        List: ["FoodInsecurity-Adverse", "FinancialSituation-Adverse"]
    """
    if pd.isna(sdoh_labels_str) or str(sdoh_labels_str).strip() == "" or str(sdoh_labels_str).strip() == "NoSDoH":
        return []
    
    if pd.isna(adversity_str) or str(adversity_str).strip() == "":
        # If no adversity info, can't create level 2 labels
        return []
    
    sdoh_factors = [label.strip() for label in str(sdoh_labels_str).split(',')]
    adversity_labels = [label.strip() for label in str(adversity_str).split(',')]
    
    # Handle case where adversity list is shorter (pad with 'adverse' as default)
    while len(adversity_labels) < len(sdoh_factors):
        adversity_labels.append('adverse')
    
    # Create Level 2 labels
    level2_labels = []
    for i, sdoh in enumerate(sdoh_factors):
        if i < len(adversity_labels):
            # Standardize adversity format
            adv = adversity_labels[i].lower()
            if adv in ['adverse', 'adv']:
                adv_suffix = 'Adverse'
            elif adv in ['protective', 'nonadverse', 'non-adverse', 'prot']:
                adv_suffix = 'Protective'
            else:
                adv_suffix = 'Adverse'  # Default fallback
            
            level2_labels.append(f"{sdoh}-{adv_suffix}")
    
    return level2_labels


def extract_sdoh_and_adversity(level2_labels):
    """
    Extract SDoH factors and adversity classifications from Level 2 labels
    
    Args:
        level2_labels: ["FoodInsecurity-Adverse", "FinancialSituation-Protective"]
    
    Returns:
        tuple: (sdoh_factors, adversity_classifications)
               (["FoodInsecurity", "FinancialSituation"], ["Adverse", "Protective"])
    """
    if not level2_labels:
        return [], []
    
    sdoh_factors = []
    adversity_classifications = []
    
    for label in level2_labels:
        if '-' in label:
            sdoh, adversity = label.split('-', 1)
            sdoh_factors.append(sdoh.strip())
            adversity_classifications.append(adversity.strip())
        else:
            # Handle malformed labels
            sdoh_factors.append(label.strip())
            adversity_classifications.append('Adverse')  # Default
    
    return sdoh_factors, adversity_classifications


def calculate_level2_multilabel_metrics(results_df: pd.DataFrame) -> Tuple[Dict[str, Any], MultiLabelBinarizer, np.ndarray, np.ndarray]:
    """Calculate Level 2 multi-label classification metrics (SDoH + Adversity)"""
    
    # Parse Level 2 labels for both ground truth and predictions
    all_labels_gt = []
    all_labels_pred = []
    
    # Also track SDoH-only accuracy (ignoring adversity)
    sdoh_only_gt = []
    sdoh_only_pred = []
    
    for _, row in results_df.iterrows():
        # Convert annotations to Level 2 format if needed
        if 'original_adversity' in row and row['original_adversity'] is not None:
            # Convert from separate SDoH + adversity columns
            gt_labels = parse_level1_with_adversity_to_level2(
                row['original_label'], 
                row['original_adversity']
            )
        else:
            # Assume already in Level 2 format
            gt_labels = parse_level2_labels_to_list(row['original_label'])
        
        pred_labels = parse_level2_labels_to_list(row['model_prediction'])
        
        all_labels_gt.append(gt_labels)
        all_labels_pred.append(pred_labels)
        
        # Extract SDoH-only for separate analysis
        gt_sdoh, _ = extract_sdoh_and_adversity(gt_labels)
        pred_sdoh, _ = extract_sdoh_and_adversity(pred_labels)
        
        sdoh_only_gt.append(gt_sdoh)
        sdoh_only_pred.append(pred_sdoh)
    
    # Create multi-label binarizers
    mlb_full = MultiLabelBinarizer()
    mlb_sdoh = MultiLabelBinarizer()
    
    # Get all possible labels
    all_possible_labels_full = set()
    all_possible_labels_sdoh = set()
    
    for labels in all_labels_gt + all_labels_pred:
        all_possible_labels_full.update(labels)
    
    for labels in sdoh_only_gt + sdoh_only_pred:
        all_possible_labels_sdoh.update(labels)
    
    # Remove empty strings
    all_possible_labels_full.discard('')
    all_possible_labels_sdoh.discard('')
    
    # Fit binarizers
    mlb_full.fit([sorted(list(all_possible_labels_full))])
    mlb_sdoh.fit([sorted(list(all_possible_labels_sdoh))])
    
    # Transform to binary format
    y_true_full = mlb_full.transform(all_labels_gt)
    y_pred_full = mlb_full.transform(all_labels_pred)
    
    y_true_sdoh = mlb_sdoh.transform(sdoh_only_gt)
    y_pred_sdoh = mlb_sdoh.transform(sdoh_only_pred)
    
    # Calculate metrics for both full Level 2 and SDoH-only
    metrics = {}
    
    # 1. FULL LEVEL 2 METRICS (SDoH + Adversity)
    metrics['level2_full'] = calculate_multilabel_metrics_core(y_true_full, y_pred_full, mlb_full.classes_)
    
    # 2. SDOH-ONLY METRICS (ignoring adversity)
    metrics['sdoh_only'] = calculate_multilabel_metrics_core(y_true_sdoh, y_pred_sdoh, mlb_sdoh.classes_)
    
    # 3. ADVERSITY-ONLY METRICS (for matched SDoH factors)
    adversity_metrics = calculate_adversity_only_metrics(all_labels_gt, all_labels_pred)
    metrics['adversity_only'] = adversity_metrics
    
    # 4. COMBINED ANALYSIS
    metrics['combined_analysis'] = calculate_combined_analysis(
        all_labels_gt, all_labels_pred, y_true_full, y_pred_full, y_true_sdoh, y_pred_sdoh
    )
    
    # 5. STATISTICS
    metrics['statistics'] = {
        'total_samples': len(y_true_full),
        'total_level2_labels': len(mlb_full.classes_),
        'total_sdoh_labels': len(mlb_sdoh.classes_),
        'level2_labels_used': sorted(list(mlb_full.classes_)),
        'sdoh_labels_used': sorted(list(mlb_sdoh.classes_)),
        'label_cardinality_true_full': np.mean(np.sum(y_true_full, axis=1)),
        'label_cardinality_pred_full': np.mean(np.sum(y_pred_full, axis=1)),
        'label_cardinality_true_sdoh': np.mean(np.sum(y_true_sdoh, axis=1)),
        'label_cardinality_pred_sdoh': np.mean(np.sum(y_pred_sdoh, axis=1))
    }
    
    # Add processing time info if available
    if 'processing_time_seconds' in results_df.columns:
        metrics['timing'] = {
            'avg_processing_time': results_df['processing_time_seconds'].mean(),
            'total_processing_time': results_df['processing_time_seconds'].sum()
        }
    
    return metrics, mlb_full, y_true_full, y_pred_full


def calculate_multilabel_metrics_core(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    """Core multi-label metrics calculation"""
    
    # Example-based metrics
    exact_match = accuracy_score(y_true, y_pred)
    hamming_loss_score = hamming_loss(y_true, y_pred)
    
    # Calculate sample-wise metrics
    precision_samples = []
    recall_samples = []
    f1_samples = []
    jaccard_samples = []
    
    for i in range(len(y_true)):
        true_set = set(np.where(y_true[i] == 1)[0])
        pred_set = set(np.where(y_pred[i] == 1)[0])
        
        if len(pred_set) == 0 and len(true_set) == 0:
            precision_samples.append(1.0)
            recall_samples.append(1.0)
            f1_samples.append(1.0)
            jaccard_samples.append(1.0)
        elif len(pred_set) == 0:
            precision_samples.append(0.0)
            recall_samples.append(0.0)
            f1_samples.append(0.0)
            jaccard_samples.append(0.0)
        elif len(true_set) == 0:
            precision_samples.append(0.0)
            recall_samples.append(0.0)
            f1_samples.append(0.0)
            jaccard_samples.append(0.0)
        else:
            intersection = len(true_set & pred_set)
            union = len(true_set | pred_set)
            
            precision = intersection / len(pred_set)
            recall = intersection / len(true_set)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            jaccard = intersection / union if union > 0 else 0
            
            precision_samples.append(precision)
            recall_samples.append(recall)
            f1_samples.append(f1)
            jaccard_samples.append(jaccard)
    
    # Label-based metrics
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Per-label metrics
    class_report = classification_report(y_true, y_pred, target_names=class_names, 
                                       output_dict=True, zero_division=0)
    
    per_label_metrics = {}
    for label in class_names:
        if label in class_report:
            per_label_metrics[label] = {
                'precision': class_report[label]['precision'],
                'recall': class_report[label]['recall'],
                'f1_score': class_report[label]['f1-score'],
                'support': class_report[label]['support']
            }
    
    return {
        'example_based': {
            'exact_match_ratio': exact_match,
            'hamming_loss': hamming_loss_score,
            'precision': np.mean(precision_samples),
            'recall': np.mean(recall_samples),
            'f1_score': np.mean(f1_samples),
            'jaccard_index': np.mean(jaccard_samples)
        },
        'label_based': {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1
        },
        'per_label': per_label_metrics
    }


def calculate_adversity_only_metrics(all_labels_gt: List[List[str]], all_labels_pred: List[List[str]]) -> Dict[str, Any]:
    """Calculate metrics for adversity classification on matched SDoH factors"""
    
    adversity_matches = []
    total_adversity_comparisons = 0
    
    for gt_labels, pred_labels in zip(all_labels_gt, all_labels_pred):
        gt_sdoh, gt_adv = extract_sdoh_and_adversity(gt_labels)
        pred_sdoh, pred_adv = extract_sdoh_and_adversity(pred_labels)
        
        # Create dictionaries for easier lookup
        gt_dict = {sdoh: adv for sdoh, adv in zip(gt_sdoh, gt_adv)}
        pred_dict = {sdoh: adv for sdoh, adv in zip(pred_sdoh, pred_adv)}
        
        # Compare adversity for matching SDoH factors
        common_sdoh = set(gt_dict.keys()) & set(pred_dict.keys())
        
        for sdoh in common_sdoh:
            total_adversity_comparisons += 1
            if gt_dict[sdoh] == pred_dict[sdoh]:
                adversity_matches.append(1)
            else:
                adversity_matches.append(0)
    
    if total_adversity_comparisons == 0:
        adversity_accuracy = 0.0
    else:
        adversity_accuracy = sum(adversity_matches) / total_adversity_comparisons
    
    return {
        'adversity_accuracy': adversity_accuracy,
        'total_comparisons': total_adversity_comparisons,
        'correct_adversity_classifications': sum(adversity_matches)
    }


def calculate_combined_analysis(all_labels_gt, all_labels_pred, y_true_full, y_pred_full, y_true_sdoh, y_pred_sdoh):
    """Calculate combined analysis comparing different evaluation approaches"""
    
    num_samples = len(all_labels_gt)
    
    # Count different types of matches
    exact_level2_matches = 0
    sdoh_matches_only = 0
    partial_matches = 0
    no_matches = 0
    
    for i in range(num_samples):
        # Check Level 2 exact match
        level2_exact = np.array_equal(y_true_full[i], y_pred_full[i])
        
        # Check SDoH-only match
        sdoh_exact = np.array_equal(y_true_sdoh[i], y_pred_sdoh[i])
        
        if level2_exact:
            exact_level2_matches += 1
        elif sdoh_exact:
            sdoh_matches_only += 1
        elif np.any(y_true_sdoh[i] & y_pred_sdoh[i]):
            partial_matches += 1
        else:
            no_matches += 1
    
    return {
        'exact_level2_matches': exact_level2_matches,
        'exact_level2_match_rate': exact_level2_matches / num_samples,
        'sdoh_only_matches': sdoh_matches_only,
        'sdoh_only_match_rate': sdoh_matches_only / num_samples,
        'partial_matches': partial_matches,
        'partial_match_rate': partial_matches / num_samples,
        'no_matches': no_matches,
        'no_match_rate': no_matches / num_samples
    }


def print_level2_multilabel_analysis(results_df: pd.DataFrame, metrics: Dict[str, Any], 
                                   mlb: MultiLabelBinarizer, y_true: np.ndarray, y_pred: np.ndarray):
    """Print comprehensive Level 2 multi-label analysis"""
    
    print("\n" + "="*70)
    print("LEVEL 2 MULTI-LABEL CLASSIFICATION EVALUATION (SDoH + Adversity)")
    print("="*70)
    
    # Basic statistics
    stats = metrics['statistics']
    print(f"Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Total Level 2 labels: {stats['total_level2_labels']}")
    print(f"  Total SDoH labels: {stats['total_sdoh_labels']}")
    print(f"  Level 2 labels: {', '.join(stats['level2_labels_used'][:10])}...")  # Show first 10
    print(f"  SDoH labels: {', '.join(stats['sdoh_labels_used'])}")
    
    # Combined analysis
    print(f"\n" + "-"*50)
    print("COMBINED ANALYSIS")
    print("-"*50)
    combined = metrics['combined_analysis']
    print(f"Exact Level 2 matches (SDoH + Adversity): {combined['exact_level2_match_rate']:.3f} ({combined['exact_level2_matches']}/{stats['total_samples']})")
    print(f"SDoH-only matches (ignore adversity):     {combined['sdoh_only_match_rate']:.3f} ({combined['sdoh_only_matches']}/{stats['total_samples']})")
    print(f"Partial matches:                          {combined['partial_match_rate']:.3f} ({combined['partial_matches']}/{stats['total_samples']})")
    print(f"No matches:                               {combined['no_match_rate']:.3f} ({combined['no_matches']}/{stats['total_samples']})")
    
    # Level 2 Full metrics
    print(f"\n" + "-"*50)
    print("LEVEL 2 FULL METRICS (SDoH + Adversity)")
    print("-"*50)
    l2_full = metrics['level2_full']
    print(f"Exact Match Ratio: {l2_full['example_based']['exact_match_ratio']:.3f}")
    print(f"Macro F1:          {l2_full['label_based']['macro_f1']:.3f}")
    print(f"Micro F1:          {l2_full['label_based']['micro_f1']:.3f}")
    
    # SDoH-only metrics
    print(f"\n" + "-"*50)
    print("SDoH-ONLY METRICS (ignoring adversity)")
    print("-"*50)
    sdoh_only = metrics['sdoh_only']
    print(f"Exact Match Ratio: {sdoh_only['example_based']['exact_match_ratio']:.3f}")
    print(f"Macro F1:          {sdoh_only['label_based']['macro_f1']:.3f}")
    print(f"Micro F1:          {sdoh_only['label_based']['micro_f1']:.3f}")
    
    # Adversity-only metrics
    print(f"\n" + "-"*50)
    print("ADVERSITY-ONLY METRICS (for matched SDoH)")
    print("-"*50)
    adv_only = metrics['adversity_only']
    print(f"Adversity Accuracy: {adv_only['adversity_accuracy']:.3f}")
    print(f"Total Comparisons:  {adv_only['total_comparisons']}")
    print(f"Correct Classifications: {adv_only['correct_adversity_classifications']}")
    
    # Top performing Level 2 labels
    print(f"\n" + "-"*50)
    print("TOP PERFORMING LEVEL 2 LABELS")
    print("-"*50)
    per_label = metrics['level2_full']['per_label']
    sorted_labels = sorted(per_label.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    
    print(f"{'Label':<30} {'F1-Score':<10} {'Support':<8}")
    print("-" * 50)
    for label, label_metrics in sorted_labels[:10]:  # Top 10
        print(f"{label:<30} {label_metrics['f1_score']:<10.3f} {label_metrics['support']:<8.0f}")
    
    # Timing info
    if 'timing' in metrics:
        timing = metrics['timing']
        print(f"\nProcessing Time:")
        print(f"  Average per sentence: {timing['avg_processing_time']:.3f} seconds")
        print(f"  Total time: {timing['total_processing_time']:.1f} seconds")


def save_level2_evaluation_results(results_df: pd.DataFrame, metrics: Dict[str, Any], 
                                 model_name: str, prompt_type: str, level: int, 
                                 output_dir: str) -> Tuple[str, str]:
    """Save Level 2 evaluation results and metrics to files"""
    
    import json
    import time
    from pathlib import Path
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames
        model_short = model_name.split('/')[-1].replace('-', '_')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        results_filename = f"level2_eval_{model_short}_{prompt_type}_L{level}_{timestamp}.csv"
        metrics_filename = f"level2_metrics_{model_short}_{prompt_type}_L{level}_{timestamp}.json"
        
        # Save results DataFrame
        results_path = output_path / results_filename
        results_df.to_csv(results_path, index=False)
        print(f"Level 2 results saved to: {results_path}")
        
        # Convert metrics for JSON serialization
        def convert_for_json(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        metrics_json = convert_for_json(metrics)
        
        # Save metrics as JSON
        metrics_path = output_path / metrics_filename
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"Level 2 metrics saved to: {metrics_path}")
        
        return str(results_path), str(metrics_path)
        
    except Exception as e:
        print(f"Error saving Level 2 results: {e}")
        return "", ""