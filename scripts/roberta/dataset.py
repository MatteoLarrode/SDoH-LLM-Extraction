import torch
from torch.utils.data import Dataset

def is_sdoh_label(label_str):
    labels = label_str.strip("<LIST>").strip("</LIST>").split(",")
    return int(not (len(labels) == 1 and labels[0] == "NoSDoH"))

class BinarySDoHDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.texts = dataframe["Sentence"].tolist()
        self.has_labels = "binary_label" in dataframe.columns
        self.labels = dataframe["binary_label"].tolist() if self.has_labels else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {k: v.squeeze() for k, v in encoding.items()}
        if self.has_labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item