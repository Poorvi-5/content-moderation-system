# src/data/preprocess_image.py
# What this file does:
# Takes raw image dataset → resizes all images to 224x224
# → normalizes pixel values → saves processed metadata
# 224x224 is the standard input size for ResNet

# src/data/preprocess_image.py

from PIL import Image
import numpy as np
import pandas as pd
import os

RAW_PATH       = "data/raw/image"
PROCESSED_PATH = "data/processed/image"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
TARGET_SIZE   = (224, 224)


def preprocess_single_image(image_path: str) -> np.ndarray:
    """
    Loads image from disk, resizes to 224x224,
    normalizes using ImageNet stats, returns
    numpy array of shape (3, 224, 224)
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(TARGET_SIZE)

    arr = np.array(image, dtype=np.float32) / 255.0

    for i in range(3):
        arr[:, :, i] = (arr[:, :, i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]

    # PyTorch wants channels first: (3, 224, 224)
    arr = arr.transpose(2, 0, 1)
    return arr


def process_split(csv_path: str, split_name: str):
    print(f"Processing {split_name} images...")

    save_dir = f"{PROCESSED_PATH}/arrays/{split_name}"
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    records = []
    errors  = 0

    for i, row in df.iterrows():
        try:
            arr = preprocess_single_image(row["image_path"])

            array_path = f"{save_dir}/img_{i}.npy"
            np.save(array_path, arr)

            records.append({
                "array_path": array_path,
                "label": row["label"],
                "original_class": row["original_class"]
            })

        except Exception as e:
            errors += 1
            continue

        if i % 2000 == 0:
            print(f"  {i}/{len(df)} done...")

    meta_df = pd.DataFrame(records)
    meta_df.to_csv(f"{PROCESSED_PATH}/{split_name}_metadata.csv", index=False)

    print(f"  Done. {len(records)} processed, {errors} skipped.")
    print(f"  Label distribution:\n{meta_df['label'].value_counts()}")
    return meta_df


def preprocess_image_data():
    print("Preprocessing image data...")
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    process_split(f"{RAW_PATH}/train_labels.csv", "train")
    process_split(f"{RAW_PATH}/test_labels.csv",  "test")

    print(f"\nImage preprocessing complete. Saved to {PROCESSED_PATH}/")


if __name__ == "__main__":
    preprocess_image_data()