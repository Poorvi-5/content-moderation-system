# src/models/datasets.py
# PyTorch Dataset classes for text and image data.
# PyTorch's DataLoader calls __getitem__ repeatedly
# during training to feed batches into the model.

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class TextModerationDataset(Dataset):
    """
    Loads preprocessed text CSV and tokenizes on the fly.
    Each item returned is a dict with:
      - input_ids      : token IDs DistilBERT understands
      - attention_mask : 1 for real tokens, 0 for padding
      - label          : 0 (clean) or 1 (toxic)
    """

    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        self.df        = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Drop any rows with missing text
        self.df = self.df.dropna(subset=["clean_text", "label"])
        self.df = self.df.reset_index(drop=True)

        print(f"Loaded {len(self.df)} text samples from {csv_path}")

    def __len__(self):
        # PyTorch needs to know total dataset size
        return len(self.df)

    def __getitem__(self, idx):
        text  = str(self.df.loc[idx, "clean_text"])
        label = int(self.df.loc[idx, "label"])

        # Tokenizer converts raw text → token IDs
        # padding/truncation ensures all inputs are same length
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"   # return PyTorch tensors
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long)
        }


class ImageModerationDataset(Dataset):
    """
    Loads preprocessed image numpy arrays from disk.
    Each item returned is a dict with:
      - image : float tensor of shape (3, 224, 224)
      - label : 0 (safe) or 1 (flagged)
    """

    def __init__(self, metadata_csv: str):
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df.dropna(subset=["array_path", "label"])
        self.df = self.df.reset_index(drop=True)

        print(f"Loaded {len(self.df)} image samples from {metadata_csv}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        array_path = self.df.loc[idx, "array_path"]
        label      = int(self.df.loc[idx, "label"])

        # Load the preprocessed numpy array we saved in Phase 2
        arr = np.load(array_path).astype(np.float32)

        return {
            "image": torch.tensor(arr),
            "label": torch.tensor(label, dtype=torch.long)
        }