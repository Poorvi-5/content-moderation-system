# src/data/run_pipeline.py
# This is your one-click data pipeline runner.
# Run this file and it will:
# 1. Download all datasets
# 2. Preprocess text
# 3. Preprocess images
# In the right order, with error handling.

import sys
import os

# Make sure Python can find our src/ modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.download_data import download_text_dataset, download_image_dataset
from src.data.preprocess_text import preprocess_text_data
from src.data.preprocess_image import preprocess_image_data


def run_full_pipeline():
    print("=" * 50)
    print("CONTENT MODERATION — DATA PIPELINE")
    print("=" * 50)

    steps = [
        ("Downloading text dataset",  download_text_dataset),
        ("Downloading image dataset", download_image_dataset),
        ("Preprocessing text",        preprocess_text_data),
        ("Preprocessing images",      preprocess_image_data),
    ]

    for step_name, step_fn in steps:
        print(f"\n[STEP] {step_name}...")
        try:
            step_fn()
            print(f"[DONE] {step_name}")
        except Exception as e:
            print(f"[ERROR] {step_name} failed: {e}")
            raise

    print("\n" + "=" * 50)
    print("Pipeline complete. Data ready in data/processed/")
    print("=" * 50)


if __name__ == "__main__":
    run_full_pipeline()