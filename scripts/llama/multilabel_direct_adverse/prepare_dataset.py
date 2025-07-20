import pandas as pd
from datasets import Dataset
from pathlib import Path
import sys

# Add project root to path to enable imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from scripts.llama.shared_utils.prompts import build_sdoh_adverse_only_prompt, build_sdoh_adverse_only_prompt_infer

def strip_protective_labels(label_string):
    # Convert "<LIST>Label-Adverse, Label-Protective</LIST>" to only Adverse; if none, then <LIST>NoSDoH</LIST>
    if label_string.startswith("<LIST>") and label_string.endswith("</LIST>"):
        label_content = label_string[6:-7]
        adverse_labels = [l.strip() for l in label_content.split(",") if "-Adverse" in l]
        return f"<LIST>{', '.join(adverse_labels)}</LIST>" if adverse_labels else "<LIST>NoSDoH</LIST>"
    return "<LIST>NoSDoH</LIST>"

def prepare_adverse_only_dataset(csv_path, prompt_builder=build_sdoh_adverse_only_prompt):
    df = pd.read_csv(csv_path)

    df["completion"] = df["completion"].apply(strip_protective_labels)
    df["text"] = df.apply(lambda row: prompt_builder(row["Sentence"], row["completion"]), axis=1)

    dataset = Dataset.from_pandas(df[["text", "completion"]])
    return dataset

def prepare_adverse_only_dataset_infer(csv_path, prompt_builder=build_sdoh_adverse_only_prompt_infer):
    df = pd.read_csv(csv_path)

    df["completion"] = df["completion"].apply(strip_protective_labels)
    df["prompt"] = df["Sentence"].apply(lambda s: prompt_builder(s))
    return df