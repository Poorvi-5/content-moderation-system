# src/api/predictor.py
# Functions that take raw input (text string, image bytes)
# and return a toxicity probability score (0.0 to 1.0).
# These are called by the FastAPI route handlers.

import io
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
TARGET_SIZE   = (224, 224)
MAX_LENGTH    = 128


def predict_text(text: str, tokenizer, model) -> float:
    """
    Takes raw text string.
    Returns probability (0.0-1.0) that text is toxic.
    """
    # Clean text slightly
    text = text.strip().lower()[:512]

    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask)
        # Convert logits to probabilities
        probs = torch.softmax(outputs.logits, dim=1)
        # Index 1 = probability of being toxic
        toxic_prob = probs[0][1].item()

    return round(toxic_prob, 4)


def predict_image(image_bytes: bytes, model) -> float:
    """
    Takes raw image bytes (from file upload).
    Returns probability (0.0-1.0) that image is flagged.
    """
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(TARGET_SIZE)

    # Normalize same way as training
    arr = np.array(image, dtype=np.float32) / 255.0
    for i in range(3):
        arr[:, :, i] = (arr[:, :, i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]

    # Shape: (1, 3, 224, 224) — batch of 1
    tensor = torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output     = model(tensor)
        probs      = torch.softmax(output, dim=1)
        flag_prob  = probs[0][1].item()

    return round(flag_prob, 4)