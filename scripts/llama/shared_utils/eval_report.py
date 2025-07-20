import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score

def evaluate_multilabel_predictions(y_true, y_pred, output_dir):
    """
    Evaluate multi-label classification predictions and save performance metrics.

    Parameters:
    - y_true: List of lists. Each sublist contains the ground-truth labels for a sample.
    - y_pred: List of lists. Each sublist contains the predicted labels for a sample.
    - output_dir: Path to directory where evaluation results will be saved.

    Output:
    - Saves a CSV file named `eval_performance.csv` in output_dir containing:
        - Per-label precision, recall, and F1-score
        - Overall micro and macro F1 scores
    - Also prints the classification report to stdout.
    """
    # Binarize the labels
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    # Compute classification report as dict
    report_dict = classification_report(
        y_true_bin, y_pred_bin, target_names=mlb.classes_, output_dict=True
    )

    # Convert to DataFrame
    df_report = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={'index': 'label'})

    # Extract micro and macro F1 scores
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average="micro")
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro")

    # Append micro and macro F1 to the DataFrame
    summary_df = pd.DataFrame([
        {"label": "micro avg", "precision": micro_f1, "recall": "-", "f1-score": "-", "support": "-"},
        {"label": "macro avg", "precision": macro_f1, "recall": "-", "f1-score": "-", "support": "-"},
    ])

    output_df = pd.concat([
        df_report[df_report["label"].isin(mlb.classes_)],
        summary_df
    ], ignore_index=True)

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "eval_performance.csv")
    output_df.to_csv(output_csv_path, index=False)

    # Print full classification report
    print("\n📊 Multi-label Classification Report:\n")
    print(classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_))