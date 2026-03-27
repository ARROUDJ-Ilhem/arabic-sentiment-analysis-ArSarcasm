"""
PyTorch Dataset class for Arabic sentiment tweets.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd


# 3 classes from ArSarcasm sentiment column
LABEL2ID = {'positive': 0, 'negative': 1, 'neutral': 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


class ArabicTweetDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_len: int = 128):
        self.texts  = dataframe['text_clean'].tolist()
        self.labels = dataframe['label_id'].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }


if __name__ == "__main__":
    print("Dataset smoke test:")
    print(f"  NUM_LABELS: {NUM_LABELS}")
    print(f"  LABEL2ID: {LABEL2ID}")
    print("  OK")