import pandas as pd
from datasets import Dataset
from pathlib import Path
import sys

# Add project root to path to enable imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from scripts.llama.shared_utils.prompts import build_sdoh_multilabel_present_or_not_prompt, build_sdoh_multilabel_present_or_not_prompt_infer

def strip_polarity(label_string):
        # Input: "<LIST>Label1-Adverse, Label2-Protective</LIST>"
        # Output: "<LIST>Label1, Label2</LIST>"
        if label_string.startswith("<LIST>") and label_string.endswith("</LIST>"):
            label_content = label_string[6:-7]
            labels = [l.strip().split("-")[0] for l in label_content.split(",")]
            return f"<LIST>{', '.join(labels)}</LIST>"
        return label_string

def prepare_multilabel_dataset(csv_path, prompt_builder=build_sdoh_multilabel_present_or_not_prompt):
    df = pd.read_csv(csv_path)

    df["completion"] = df["completion"].apply(strip_polarity)
    df["text"] = df.apply(lambda row: prompt_builder(row["Sentence"], row["completion"]), axis=1)

    dataset = Dataset.from_pandas(df[["text", "completion"]])
    return dataset

def prepare_multilabel_dataset_infer(data, prompt_builder=build_sdoh_multilabel_present_or_not_prompt_infer):
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Expected a file path or DataFrame.")

    if "completion" in df.columns:
        df["completion"] = df["completion"].apply(strip_polarity)
    df["prompt"] = df["Sentence"].apply(lambda s: prompt_builder(s))
    return df