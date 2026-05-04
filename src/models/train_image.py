# src/models/train_image.py
# Fine-tunes ResNet-18 for binary image classification.
# ResNet-18 is lightweight enough to train on a CPU
# in reasonable time while still being a real CNN.
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import os
import torch
import mlflow
import mlflow.pytorch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import Adam
from sklearn.metrics import f1_score, accuracy_score
from src.models.datasets import ImageModerationDataset

# ── Config ────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS     = 5
LR         = 1e-4
SAVE_PATH  = "models/image/resnet_moderation.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────


def build_resnet():
    """
    Loads pretrained ResNet-18 and replaces the final
    fully connected layer with our binary classifier.

    Original ResNet-18 fc layer: 512 → 1000 (ImageNet classes)
    Our replacement fc layer:    512 → 2   (safe / flagged)

    All other layers keep their ImageNet pretrained weights.
    We only train the final layer + fine-tune the rest.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer — this one will be trained
    num_features = model.fc.in_features   # 512 for ResNet-18
    model.fc = nn.Linear(num_features, 2) # binary output

    # Unfreeze last residual block + final layer for fine-tuning
    # Unfreezing too many layers on a small dataset = overfitting
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def evaluate(model, dataloader):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(images)
            preds   = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="binary")
    return acc, f1


def train_image_model():
    print(f"Training on: {DEVICE}")
    os.makedirs("models/image", exist_ok=True)

    # ── Load model ────────────────────────────────────
    print("Building ResNet-18 model...")
    model = build_resnet().to(DEVICE)

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # ── Load datasets ─────────────────────────────────
    print("Loading image datasets...")
    train_dataset = ImageModerationDataset(
        "data/processed/image/train_metadata.csv"
    )
    test_dataset = ImageModerationDataset(
        "data/processed/image/test_metadata.csv"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0   # 0 for Windows compatibility
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ── Loss and optimizer ────────────────────────────
    # CrossEntropyLoss is standard for classification
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    # ── MLflow tracking ───────────────────────────────
    mlflow.set_experiment("image-moderation")

    with mlflow.start_run(run_name="resnet18-finetune"):

        mlflow.log_params({
            "model":       "resnet18",
            "epochs":      EPOCHS,
            "batch_size":  BATCH_SIZE,
            "lr":          LR,
            "trainable_params": trainable,
            "device":      str(DEVICE)
        })

        best_val_f1 = 0.0

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            correct    = 0
            total      = 0

            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print("-" * 40)

            for step, batch in enumerate(train_loader):
                images = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                optimizer.zero_grad()

                outputs = model(images)
                loss    = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds    = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

                if step % 20 == 0:
                    print(f"  Step {step}/{len(train_loader)} "
                          f"| Loss: {loss.item():.4f}")

            avg_loss  = total_loss / len(train_loader)
            train_acc = correct / total
            val_acc, val_f1 = evaluate(model, test_loader)

            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss : {avg_loss:.4f}")
            print(f"  Train Acc  : {train_acc:.4f}")
            print(f"  Val Acc    : {val_acc:.4f}")
            print(f"  Val F1     : {val_f1:.4f}")

            mlflow.log_metrics({
                "train_loss": avg_loss,
                "train_acc":  train_acc,
                "val_acc":    val_acc,
                "val_f1":     val_f1
            }, step=epoch)

            # Save best model based on validation F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"  New best model saved (F1: {val_f1:.4f})")

        mlflow.log_artifact(SAVE_PATH)
        mlflow.log_metric("best_val_f1", best_val_f1)

        print(f"\nBest Val F1: {best_val_f1:.4f}")
        print(f"Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    train_image_model()