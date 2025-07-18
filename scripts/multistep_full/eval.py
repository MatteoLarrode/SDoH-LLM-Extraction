import argparse
import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.multistep_full.two_step_pipeline import run_two_step_pipeline

def parse_labels(list_string):
    try:
        inner = list_string.replace("<LIST>", "").replace("</LIST>", "").strip()
        if not inner:
            return []
        return [label.strip() for label in inner.split(",")]
    except Exception:
        return []

def main(args):
    output_dir = "results/model_training/twostep_multi_label_full"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "two_step_predictions.csv")

    run_two_step_pipeline(
        test_data_file=args.test_data_file,
        roberta_model_dir=args.roberta_model_dir,
        llama_model_dir=args.llama_model_dir,
        pos_weight=args.pos_weight,
        output_file=output_file
    )

    df = pd.read_csv(output_file)
    if args.head:
        df = df.head(10)

    y_true = df["completion"].apply(parse_labels)
    y_pred = df["final_prediction"].apply(parse_labels)

    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    print("\n📊 Multi-label Classification Report:\n")
    print(classification_report(y_true_bin, y_pred_bin, target_names=mlb.classes_))

    df[["Sentence", "completion", "final_prediction"]].to_csv(
        os.path.join(output_dir, "eval_predictions.csv"), index=False
    )
    print(f"\n✅ Evaluation predictions saved to {output_dir}/eval_predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_file", type=str, required=True)
    parser.add_argument("--roberta_model_dir", type=str, required=True)
    parser.add_argument("--llama_model_dir", type=str, required=True)
    parser.add_argument("--pos_weight", type=float, required=True)
    parser.add_argument("--head", action="store_true", help="Evaluate only on the first 10 rows")
    args = parser.parse_args()
    main(args)