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

    if 'case_ref' in df.columns:
        dataset = Dataset.from_pandas(df[["case_ref", "text", "completion"]])
    else:
        dataset = Dataset.from_pandas(df[["text", "completion"]])
    return dataset

def prepare_adverse_only_dataset_infer(data, prompt_builder=build_sdoh_adverse_only_prompt_infer):
    """
    Prepares a dataset for inference on adverse-only SDoH extraction.

    Parameters:
        data (str or pd.DataFrame): Path to CSV file or DataFrame with at least a 'Sentence' column.
        prompt_builder (function): Function to create a prompt from a sentence.

    Returns:
        pd.DataFrame: Original data with an added 'prompt' column.
                      If 'completion' exists, it's cleaned using strip_protective_labels.
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Expected a file path or a DataFrame.")

    df["prompt"] = df["Sentence"].apply(prompt_builder)

    if "completion" in df.columns:
        df["completion"] = df["completion"].apply(strip_protective_labels)

    return df