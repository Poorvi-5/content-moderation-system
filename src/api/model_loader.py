# src/api/model_loader.py
# Loads all trained models once when the server starts.
# Keeping models in memory means instant predictions —
# no disk reads on every API call.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import torch
from torchvision import models
from torch import nn
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────
TEXT_MODEL_PATH  = "models/text/distilbert_moderation.pt"
IMAGE_MODEL_PATH = "models/image/mobilenet_moderation.pt"


def load_text_model():
    """
    Loads DistilBERT tokenizer + fine-tuned weights.
    Returns (tokenizer, model) tuple.
    """
    print("Loading text model (DistilBERT)...")

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    model.load_state_dict(
        torch.load(TEXT_MODEL_PATH, map_location=DEVICE)
    )
    model = model.to(DEVICE)
    model.eval()

    print("Text model loaded.")
    return tokenizer, model


def load_image_model():
    print("Loading image model (ResNet18)...")
    from torchvision import models as tv_models
    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(
        torch.load("models/image/resnet_moderation.pt",
                   map_location=DEVICE)
    )
    model = model.to(DEVICE)
    model.eval()
    print("Image model loaded.")
    return model


# Global model instances — loaded once at startup
print("Initializing models...")
TEXT_TOKENIZER, TEXT_MODEL = load_text_model()
IMAGE_MODEL                = load_image_model()
print("All models ready.")