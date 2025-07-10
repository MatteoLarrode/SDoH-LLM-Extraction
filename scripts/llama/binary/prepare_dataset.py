import pandas as pd
from datasets import Dataset
from pathlib import Path
import sys

# Add project root to path to enable imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from scripts.llama.shared_utils.prompts import build_sdoh_detection_prompt, build_sdoh_detection_prompt_infer

def prepare_binary_dataset(csv_path, prompt_builder=build_sdoh_detection_prompt):
    df = pd.read_csv(csv_path)

    binary_label = lambda c: "<LIST>NoSDoH</LIST>" if "NoSDoH" in c else "<LIST>SDoH</LIST>"
    df["completion"] = df["completion"].map(binary_label)
    df["text"] = df.apply(lambda row: prompt_builder(row["Sentence"], row["completion"]), axis=1)

    dataset = Dataset.from_pandas(df[["text", "completion"]])
    return dataset

def prepare_binary_dataset_infer(csv_path, prompt_builder=build_sdoh_detection_prompt_infer):
    df = pd.read_csv(csv_path)

    binary_label = lambda c: "<LIST>NoSDoH</LIST>" if "NoSDoH" in c else "<LIST>SDoH</LIST>"
    df["completion"] = df["completion"].map(binary_label)
    df["text"] = df["Sentence"].apply(lambda s: prompt_builder(s))
    return df