# src/data/preprocess_text.py
# What this file does:
# Takes raw Jigsaw CSV → cleans text → creates binary
# toxic/clean labels → saves to data/processed/text/
# This is what the DistilBERT model will train on

# src/data/preprocess_text.py

import pandas as pd
import re
import os

RAW_PATH       = "data/raw/text"
PROCESSED_PATH = "data/processed/text"

# Threshold: if >= 50% of annotators said toxic → label as 1
TOXICITY_THRESHOLD = 0.5


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text_data():
    print("Preprocessing text data...")
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    # civil_comments already has train/val/test splits
    train_df = pd.read_csv(f"{RAW_PATH}/train.csv")
    val_df   = pd.read_csv(f"{RAW_PATH}/val.csv")
    test_df  = pd.read_csv(f"{RAW_PATH}/test.csv")

    print(f"Raw train shape: {train_df.shape}")
    print(f"Columns: {list(train_df.columns)}")

    def process_split(df, name):
        # civil_comments has a 'text' column and 'toxicity' float column
        df["clean_text"] = df["text"].apply(clean_text)

        # Convert float toxicity score to binary label
        # 0.5 threshold = majority of annotators agreed it's toxic
        df["label"] = (df["toxicity"] >= TOXICITY_THRESHOLD).astype(int)

        # Remove very short texts after cleaning
        df = df[df["clean_text"].str.len() > 10]

        # Keep only what the model needs
        final = df[["clean_text", "label"]].copy()

        save_path = f"{PROCESSED_PATH}/{name}.csv"
        final.to_csv(save_path, index=False)

        print(f"\n{name}: {len(final)} rows")
        print(f"  Toxic:  {final['label'].sum()} ({final['label'].mean()*100:.1f}%)")
        print(f"  Clean:  {(final['label']==0).sum()}")
        return final

    process_split(train_df, "train")
    process_split(val_df,   "val")
    process_split(test_df,  "test")

    print(f"\nPreprocessed text saved to {PROCESSED_PATH}/")


if __name__ == "__main__":
    preprocess_text_data()