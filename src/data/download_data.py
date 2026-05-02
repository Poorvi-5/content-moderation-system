# src/data/download_data.py
# What this file does:
# Downloads real moderation datasets from HuggingFace
# Saves raw versions to data/raw/ folder
# Prints progress so you can see what's happening

# src/data/download_data.py

import os
import pandas as pd
from datasets import load_dataset


RAW_TEXT_PATH = "data/raw/text"
RAW_IMAGE_PATH = "data/raw/image"


def download_text_dataset():
    """
    Uses civil_comments dataset — 1.8M comments labeled
    for toxicity by Google. Already downloaded, this will
    just load from HuggingFace cache instantly.
    """
    print("Downloading text moderation dataset...")
    os.makedirs(RAW_TEXT_PATH, exist_ok=True)

    dataset = load_dataset("google/civil_comments")

    train_df = pd.DataFrame(dataset["train"])
    val_df   = pd.DataFrame(dataset["validation"])
    test_df  = pd.DataFrame(dataset["test"])

    train_df.to_csv(f"{RAW_TEXT_PATH}/train.csv", index=False)
    val_df.to_csv(f"{RAW_TEXT_PATH}/val.csv",     index=False)
    test_df.to_csv(f"{RAW_TEXT_PATH}/test.csv",   index=False)

    print(f"Text dataset saved.")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Columns: {list(train_df.columns)}")
    print(f"\nSample row:\n{train_df.iloc[0]}\n")


def download_image_dataset():
    """
    Uses CIFAR-10 — 60,000 images, always available,
    no auth needed. We simulate moderation labels:
    classes 3,4,5 = flagged (label=1), rest = safe (label=0).
    """
    print("Setting up image moderation dataset...")
    os.makedirs(RAW_IMAGE_PATH, exist_ok=True)

    dataset = load_dataset("uoft-cs/cifar10")

    train_data = dataset["train"]
    test_data  = dataset["test"]

    FLAGGED_CLASSES = {3, 4, 5}

    def build_records(split_data, split_name):
        records  = []
        split_dir = f"{RAW_IMAGE_PATH}/{split_name}"
        os.makedirs(split_dir, exist_ok=True)

        for i, item in enumerate(split_data):
            image       = item["img"]
            cifar_label = item["label"]
            mod_label   = 1 if cifar_label in FLAGGED_CLASSES else 0

            img_path = f"{split_dir}/img_{i}.png"
            image.save(img_path)

            records.append({
                "image_path":     img_path,
                "label":          mod_label,
                "original_class": cifar_label
            })

            if i % 2000 == 0:
                print(f"  [{split_name}] Saved {i}/{len(split_data)} images...")

        return records

    print("Saving train images...")
    train_records = build_records(train_data, "train")

    print("Saving test images...")
    test_records  = build_records(test_data, "test")

    train_df = pd.DataFrame(train_records)
    test_df  = pd.DataFrame(test_records)

    train_df.to_csv(f"{RAW_IMAGE_PATH}/train_labels.csv", index=False)
    test_df.to_csv(f"{RAW_IMAGE_PATH}/test_labels.csv",   index=False)

    print(f"\nImage dataset saved.")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Train label distribution:\n{train_df['label'].value_counts()}")


if __name__ == "__main__":
    download_text_dataset()
    download_image_dataset()
    print("\nAll datasets downloaded successfully.")