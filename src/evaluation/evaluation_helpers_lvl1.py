import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
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


def parse_labels_to_list(label_str):
    if pd.isna(label_str) or str(label_str).strip() == "":
        return []
    
    # Clean the input
    label_str = str(label_str).strip()
    
    # If it's exactly NoSDoH, return that as a single label
    if label_str == "NoSDoH":
        return ["NoSDoH"]
    
    # Otherwise, split by comma (and NoSDoH shouldn't appear here)
    return [label.strip() for label in label_str.split(',')]


def calculate_multilabel_metrics(results_df: pd.DataFrame) -> Tuple[Dict[str, Any], MultiLabelBinarizer, np.ndarray, np.ndarray]:
    """Calculate multi-label classification metrics using sklearn"""
    
    # Get all unique labels from both ground truth and predictions
    all_labels_gt = []
    all_labels_pred = []
    
    # Parse all labels
    for _, row in results_df.iterrows():
        gt_labels = parse_labels_to_list(row['original_label'])
        pred_labels = parse_labels_to_list(row['model_prediction'])
        
        all_labels_gt.append(gt_labels)
        all_labels_pred.append(pred_labels)
    
    # Create multi-label binarizer
    mlb = MultiLabelBinarizer()
    
    # Fit on all possible labels (union of ground truth and predictions)
    all_possible_labels = set()
    for labels in all_labels_gt + all_labels_pred:
        all_possible_labels.update(labels)
    
    # Remove empty set and sort for consistency
    all_possible_labels.discard('')
    mlb.fit([sorted(list(all_possible_labels))])
    
    # Transform to binary format
    y_true = mlb.transform(all_labels_gt)
    y_pred = mlb.transform(all_labels_pred)
    
    # Calculate all the metrics
    metrics = {}
    
    # 1. EXAMPLE-BASED METRICS (average across samples)
    
    # Exact Match Ratio (Subset Accuracy)
    exact_match = accuracy_score(y_true, y_pred)
    
    # Hamming Loss (fraction of wrong labels)
    hamming_loss_score = hamming_loss(y_true, y_pred)
    
    # Jaccard Index (Intersection over Union for each sample, then averaged)
    jaccard_samples = []
    precision_samples = []
    recall_samples = []
    f1_samples = []
    
    for i in range(len(y_true)):
        true_set = set(np.where(y_true[i] == 1)[0])
        pred_set = set(np.where(y_pred[i] == 1)[0])
        
        # Handle empty predictions
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
            recall_samples.append(1.0)  # No positives to miss
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
    
    # Example-based metrics
    metrics['example_based'] = {
        'exact_match_ratio': exact_match,
        'hamming_loss': hamming_loss_score,
        'precision': np.mean(precision_samples),
        'recall': np.mean(recall_samples),
        'f1_score': np.mean(f1_samples),
        'jaccard_index': np.mean(jaccard_samples)
    }
    
    # 2. LABEL-BASED METRICS (average across labels)
    
    # Macro averages (unweighted mean across labels)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Micro averages (aggregate the contributions across all labels)
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Weighted averages (weighted by support)
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics['label_based'] = {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }
    
    # 3. PER-LABEL METRICS
    per_label_metrics = {}
    label_names = mlb.classes_
    
    # Get classification report as dict
    class_report = classification_report(y_true, y_pred, target_names=label_names, 
                                       output_dict=True, zero_division=0)
    
    for i, label in enumerate(label_names):
        if label in class_report:
            per_label_metrics[label] = {
                'precision': class_report[label]['precision'],
                'recall': class_report[label]['recall'],
                'f1_score': class_report[label]['f1-score'],
                'support': class_report[label]['support']
            }
    
    metrics['per_label'] = per_label_metrics
    
    # 4. ADDITIONAL STATISTICS
    
    # Label cardinality (average number of labels per sample)
    label_cardinality_true = np.mean(np.sum(y_true, axis=1))
    label_cardinality_pred = np.mean(np.sum(y_pred, axis=1))
    
    # Label density (label cardinality / total possible labels)
    label_density_true = label_cardinality_true / len(label_names)
    label_density_pred = label_cardinality_pred / len(label_names)
    
    # Coverage (how many labels are predicted at least once)
    coverage_true = np.sum(np.sum(y_true, axis=0) > 0)
    coverage_pred = np.sum(np.sum(y_pred, axis=0) > 0)
    
    metrics['statistics'] = {
        'total_samples': len(y_true),
        'total_labels': len(label_names),
        'label_cardinality_true': label_cardinality_true,
        'label_cardinality_pred': label_cardinality_pred,
        'label_density_true': label_density_true,
        'label_density_pred': label_density_pred,
        'coverage_true': coverage_true,
        'coverage_pred': coverage_pred,
        'labels_used': sorted(list(label_names))
    }
    
    # 5. CONFUSION MATRICES PER LABEL
    conf_matrices = multilabel_confusion_matrix(y_true, y_pred)
    confusion_per_label = {}
    
    for i, label in enumerate(label_names):
        tn, fp, fn, tp = conf_matrices[i].ravel()
        confusion_per_label[label] = {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
    
    metrics['confusion_matrices'] = confusion_per_label
    
    # Add processing time info
    if 'processing_time_seconds' in results_df.columns:
        metrics['timing'] = {
            'avg_processing_time': results_df['processing_time_seconds'].mean(),
            'total_processing_time': results_df['processing_time_seconds'].sum()
        }
    
    return metrics, mlb, y_true, y_pred


def print_multilabel_analysis(results_df: pd.DataFrame, metrics: Dict[str, Any], 
                                   mlb: MultiLabelBinarizer, y_true: np.ndarray, y_pred: np.ndarray):
    """Print comprehensive multi-label analysis using proper metrics"""
    
    print("\n" + "="*70)
    print("PROPER MULTI-LABEL CLASSIFICATION EVALUATION")
    print("="*70)
    
    # Basic statistics
    stats = metrics['statistics']
    print(f"Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Total unique labels: {stats['total_labels']}")
    print(f"  Labels: {', '.join(stats['labels_used'])}")
    print(f"  Label cardinality (avg labels per sample):")
    print(f"    Ground truth: {stats['label_cardinality_true']:.2f}")
    print(f"    Predictions:  {stats['label_cardinality_pred']:.2f}")
    print(f"  Label density:")
    print(f"    Ground truth: {stats['label_density_true']:.3f}")
    print(f"    Predictions:  {stats['label_density_pred']:.3f}")
    
    # Example-based metrics
    print(f"\n" + "-"*50)
    print("EXAMPLE-BASED METRICS (averaged over samples)")
    print("-"*50)
    eb = metrics['example_based']
    print(f"Exact Match Ratio (Subset Accuracy): {eb['exact_match_ratio']:.3f}")
    print(f"Hamming Loss:                        {eb['hamming_loss']:.3f}")
    print(f"Example-based Precision:             {eb['precision']:.3f}")
    print(f"Example-based Recall:                {eb['recall']:.3f}")
    print(f"Example-based F1-Score:              {eb['f1_score']:.3f}")
    print(f"Jaccard Index:                       {eb['jaccard_index']:.3f}")
    
    # Label-based metrics
    print(f"\n" + "-"*50)
    print("LABEL-BASED METRICS (averaged over labels)")
    print("-"*50)
    lb = metrics['label_based']
    print(f"Macro-averaged:")
    print(f"  Precision: {lb['macro_precision']:.3f}")
    print(f"  Recall:    {lb['macro_recall']:.3f}")
    print(f"  F1-Score:  {lb['macro_f1']:.3f}")
    print(f"Micro-averaged:")
    print(f"  Precision: {lb['micro_precision']:.3f}")
    print(f"  Recall:    {lb['micro_recall']:.3f}")
    print(f"  F1-Score:  {lb['micro_f1']:.3f}")
    print(f"Weighted-averaged:")
    print(f"  Precision: {lb['weighted_precision']:.3f}")
    print(f"  Recall:    {lb['weighted_recall']:.3f}")
    print(f"  F1-Score:  {lb['weighted_f1']:.3f}")
    
    # Per-label performance
    print(f"\n" + "-"*50)
    print("PER-LABEL PERFORMANCE")
    print("-"*50)
    print(f"{'Label':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 65)
    
    # Sort by support for better readability
    per_label = metrics['per_label']
    sorted_labels = sorted(per_label.items(), key=lambda x: x[1]['support'], reverse=True)
    
    for label, label_metrics in sorted_labels:
        print(f"{label:<25} {label_metrics['precision']:<10.3f} "
              f"{label_metrics['recall']:<10.3f} {label_metrics['f1_score']:<10.3f} "
              f"{label_metrics['support']:<8.0f}")
    
    # Timing info
    if 'timing' in metrics:
        timing = metrics['timing']
        print(f"\nProcessing Time:")
        print(f"  Average per sentence: {timing['avg_processing_time']:.3f} seconds")
        print(f"  Total time: {timing['total_processing_time']:.1f} seconds")
    
    # Sample predictions analysis - FIXED VERSION
    print(f"\n" + "-"*50)
    print("SAMPLE PREDICTIONS ANALYSIS")
    print("-"*50)
    
    try:
        # Ensure we have numpy arrays
        if not isinstance(y_true, np.ndarray):
            y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray):
            y_pred = np.array(y_pred)
        
        # Show a few examples of each type
        exact_matches = []
        partial_matches = []
        no_matches = []
        
        # Use len() instead of .shape for compatibility
        num_samples = len(y_true) if hasattr(y_true, '__len__') else y_true.shape[0]
        
        for i in range(num_samples):
            true_labels = set(mlb.inverse_transform([y_true[i]])[0])
            pred_labels = set(mlb.inverse_transform([y_pred[i]])[0])
            
            if true_labels == pred_labels:
                exact_matches.append(i)
            elif len(true_labels & pred_labels) > 0:
                partial_matches.append(i)
            else:
                no_matches.append(i)
        
        print(f"Exact matches: {len(exact_matches)} ({len(exact_matches)/num_samples*100:.1f}%)")
        print(f"Partial matches: {len(partial_matches)} ({len(partial_matches)/num_samples*100:.1f}%)")
        print(f"No matches: {len(no_matches)} ({len(no_matches)/num_samples*100:.1f}%)")
        
        # Show examples
        for match_type, indices in [("EXACT", exact_matches[:2]), 
                                   ("PARTIAL", partial_matches[:2]), 
                                   ("NO MATCH", no_matches[:2])]:
            if indices:
                print(f"\n{match_type} Examples:")
                for idx in indices:
                    true_labels = list(mlb.inverse_transform([y_true[idx]])[0])
                    pred_labels = list(mlb.inverse_transform([y_pred[idx]])[0])
                    sentence = results_df.iloc[idx]['original_sentence'][:60]
                    
                    print(f"  Sentence: {sentence}...")
                    print(f"  True:     {true_labels if true_labels else ['NoSDoH']}")
                    print(f"  Pred:     {pred_labels if pred_labels else ['NoSDoH']}")
    
    except Exception as e:
        print(f"Error in sample analysis: {e}")
        print("Skipping sample predictions analysis...")


def save_evaluation_results(results_df: pd.DataFrame, metrics: Dict[str, Any], 
                          model_name: str, prompt_type: str, level: int, 
                          output_dir: str) -> Tuple[str, str]:
    """Save evaluation results and metrics to files"""
    
    import json
    import time
    from pathlib import Path
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames
        model_short = model_name.split('/')[-1].replace('-', '_')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        results_filename = f"annotation_eval_{model_short}_{prompt_type}_L{level}_{timestamp}.csv"
        metrics_filename = f"annotation_metrics_{model_short}_{prompt_type}_L{level}_{timestamp}.json"
        
        # Save results DataFrame
        results_path = output_path / results_filename
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")
        
        # Save metrics as JSON
        metrics_path = output_path / metrics_filename
        
        # Convert numpy types for JSON serialization
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
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"Metrics saved to: {metrics_path}")
        
        return str(results_path), str(metrics_path)
        
    except Exception as e:
        print(f"Error saving results: {e}")
        # Try to save at least the CSV
        try:
            fallback_path = Path(output_dir) / f"fallback_results_{timestamp}.csv"
            results_df.to_csv(fallback_path, index=False)
            print(f"Fallback results saved to: {fallback_path}")
            return str(fallback_path), ""
        except Exception as e2:
            print(f"Failed to save even fallback results: {e2}")
            return "", ""