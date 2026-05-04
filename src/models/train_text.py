# src/models/train_text.py
# Fine-tunes DistilBERT for binary toxicity classification.
# Logs every metric to MLflow so you can compare runs.
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import os
import torch
import mlflow
import mlflow.pytorch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
from src.models.datasets import TextModerationDataset

# ── Config ────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
BATCH_SIZE   = 32
EPOCHS       = 3
LR           = 2e-5
MAX_LENGTH   = 128
SAVE_PATH    = "models/text/distilbert_moderation.pt"

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────


def evaluate(model, dataloader):
    """
    Runs model on a dataloader without updating weights.
    Returns accuracy and F1 score.
    This is called after every epoch on validation data.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():   # no gradient calculation needed
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)

            # outputs.logits shape: (batch_size, 2)
            # argmax gives index of highest score = predicted class
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary")
    return acc, f1


def train_text_model():
    print(f"Training on: {DEVICE}")
    os.makedirs("models/text", exist_ok=True)

    # ── Load tokenizer and model ──────────────────────
    print("Loading DistilBERT tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    # DistilBertForSequenceClassification adds a
    # classification head on top of DistilBERT automatically
    # num_labels=2 means binary classification
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    model = model.to(DEVICE)

    # ── Load datasets ─────────────────────────────────
    # We use a subset of training data for speed
    # In production you'd train on the full 1.8M rows
    print("Loading datasets...")
    train_dataset = TextModerationDataset(
        "data/processed/text/train.csv",
        tokenizer,
        MAX_LENGTH
    )
    val_dataset = TextModerationDataset(
        "data/processed/text/val.csv",
        tokenizer,
        MAX_LENGTH
    )

    # Use subset for faster training on local machine
    # Remove this in production / cloud training
    subset_size  = min(20000, len(train_dataset))
    train_subset = torch.utils.data.Subset(
        train_dataset,
        range(subset_size)
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ── Optimizer and scheduler ───────────────────────
    optimizer = AdamW(model.parameters(), lr=LR)

    # Linear warmup then linear decay — standard for BERT
    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ── MLflow experiment tracking ────────────────────
    mlflow.set_experiment("text-moderation")

    with mlflow.start_run(run_name="distilbert-finetune"):

        # Log all hyperparameters — visible in MLflow UI
        mlflow.log_params({
            "model":       MODEL_NAME,
            "epochs":      EPOCHS,
            "batch_size":  BATCH_SIZE,
            "lr":          LR,
            "max_length":  MAX_LENGTH,
            "train_size":  subset_size,
            "device":      str(DEVICE)
        })

        # ── Training loop ─────────────────────────────
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            correct    = 0
            total      = 0

            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print("-" * 40)

            for step, batch in enumerate(train_loader):
                input_ids      = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels         = batch["label"].to(DEVICE)

                # Forward pass — model calculates loss internally
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss

                # Backward pass — calculate gradients
                loss.backward()

                # Clip gradients to prevent exploding gradients
                # (common issue with transformer fine-tuning)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()

                # Track training accuracy
                preds    = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

                if step % 50 == 0:
                    print(f"  Step {step}/{len(train_loader)} "
                          f"| Loss: {loss.item():.4f}")

            # ── End of epoch metrics ──────────────────
            avg_loss   = total_loss / len(train_loader)
            train_acc  = correct / total
            val_acc, val_f1 = evaluate(model, val_loader)

            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss : {avg_loss:.4f}")
            print(f"  Train Acc  : {train_acc:.4f}")
            print(f"  Val Acc    : {val_acc:.4f}")
            print(f"  Val F1     : {val_f1:.4f}")

            # Log metrics to MLflow — one value per epoch
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "train_acc":  train_acc,
                "val_acc":    val_acc,
                "val_f1":     val_f1
            }, step=epoch)

        # ── Save model ────────────────────────────────
        torch.save(model.state_dict(), SAVE_PATH)
        mlflow.log_artifact(SAVE_PATH)

        print(f"\nModel saved to {SAVE_PATH}")
        print("Training complete. Check MLflow UI for metrics.")


if __name__ == "__main__":
    train_text_model()