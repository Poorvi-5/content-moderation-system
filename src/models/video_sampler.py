# src/models/video_sampler.py
# Video moderation works by:
# 1. Sampling N frames from the video
# 2. Running each frame through ResNet image model
# 3. If > threshold% of frames are flagged → video flagged
# No separate training needed — reuses image model.

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import models
from torch import nn

IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
TARGET_SIZE    = (224, 224)
FRAME_SAMPLE_RATE = 2      # sample every 2 seconds
FLAG_THRESHOLD    = 0.3    # flag video if 30%+ frames flagged

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_model(weights_path: str):
    """Loads the trained ResNet model from disk."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE)
    )
    model = model.to(DEVICE)
    model.eval()
    return model


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """
    Takes a raw OpenCV frame (BGR numpy array)
    and converts it to a normalized tensor for ResNet.
    """
    # OpenCV loads as BGR — convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image     = Image.fromarray(frame_rgb)
    image     = image.resize(TARGET_SIZE)

    arr = np.array(image, dtype=np.float32) / 255.0
    for i in range(3):
        arr[:, :, i] = (arr[:, :, i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]

    arr = arr.transpose(2, 0, 1)  # (3, 224, 224)
    return torch.tensor(arr).unsqueeze(0)  # add batch dim


def moderate_video(video_path: str, model) -> dict:
    """
    Main video moderation function.
    Returns a dict with decision and frame-level details.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps if fps > 0 else 0

    # Sample one frame every FRAME_SAMPLE_RATE seconds
    sample_interval = int(fps * FRAME_SAMPLE_RATE)
    sample_interval = max(1, sample_interval)

    frame_results = []
    frame_idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            tensor = preprocess_frame(frame).to(DEVICE)

            with torch.no_grad():
                output    = model(tensor)
                prob      = torch.softmax(output, dim=1)
                flagged   = torch.argmax(output, dim=1).item()
                confidence = prob[0][flagged].item()

            frame_results.append({
                "frame_idx":  frame_idx,
                "flagged":    bool(flagged),
                "confidence": round(confidence, 3)
            })

        frame_idx += 1

    cap.release()

    if not frame_results:
        return {"error": "No frames could be sampled"}

    flagged_count  = sum(1 for f in frame_results if f["flagged"])
    flagged_ratio  = flagged_count / len(frame_results)
    video_flagged  = flagged_ratio >= FLAG_THRESHOLD

    return {
        "flagged":        video_flagged,
        "flagged_ratio":  round(flagged_ratio, 3),
        "frames_sampled": len(frame_results),
        "flagged_frames": flagged_count,
        "duration_s":     round(duration_s, 1),
        "decision":       "block" if video_flagged else "allow",
        "frame_details":  frame_results
    }