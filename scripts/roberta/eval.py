import argparse
import os
import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaConfig, Trainer
from dataset import BinarySDoHDataset, is_sdoh_label
from model import RobertaBinaryClassifierWithWeight
from sklearn.metrics import classification_report

os.environ["WANDB_MODE"] = "disabled"

def main(args):
    # Load test data
    test_df = pd.read_csv(args.test_data_file)
    test_df["binary_label"] = test_df["completion"].apply(is_sdoh_label)

    # Load tokenizer and config from trained model directory
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    config = RobertaConfig.from_pretrained(args.model_dir)

    # You must provide the pos_weight used during training
    model = RobertaBinaryClassifierWithWeight.from_pretrained(
        args.model_dir,
        config=config,
        pos_weight=args.pos_weight
    )

    # Prepare dataset
    test_dataset = BinarySDoHDataset(test_df, tokenizer)

    # Evaluate
    trainer = Trainer(model=model, tokenizer=tokenizer)
    outputs = trainer.predict(test_dataset)

    # Process predictions
    probs = torch.sigmoid(torch.tensor(outputs.predictions)).numpy().flatten()
    y_pred = (probs > 0.5).astype(int)
    y_true = test_df["binary_label"].values

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["NoSDoH", "Any SDoH"]))

    # Save results
    results_df = pd.DataFrame({
        "Sentence": test_df["Sentence"],
        "True Label": ["NoSDoH" if y == 0 else "Any SDoH" for y in y_true],
        "Predicted Label": ["NoSDoH" if y == 0 else "Any SDoH" for y in y_pred],
        "Prob_SDoH": probs
    })
    os.makedirs(args.model_dir, exist_ok=True)
    results_path = os.path.join(args.model_dir, "binary_predictions.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Predictions saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_file", type=str, required=True, help="Path to the test CSV file")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--pos_weight", type=float, required=True, help="Positive class weight used during training")
    args = parser.parse_args()

    main(args)