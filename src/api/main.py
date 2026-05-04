# src/api/main.py
# The main FastAPI application.
# This is what runs as your moderation service.
# Three endpoints: /moderate/text, /moderate/image, /moderate/video

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.model_loader import TEXT_TOKENIZER, TEXT_MODEL, IMAGE_MODEL
from src.api.predictor    import predict_text, predict_image
from src.api.risk_engine  import compute_risk_score
from src.models.video_sampler import load_image_model, moderate_video

# ── App setup ─────────────────────────────────────────
app = FastAPI(
    title="Content Moderation API",
    description="Real-time multimodal content moderation system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load video model (reuses image model weights)
VIDEO_MODEL = load_image_model("models/image/resnet_moderation.pt")


# ── Request/Response schemas ──────────────────────────
class TextRequest(BaseModel):
    text: str
    user_id: str = "anonymous"

class ModerationResponse(BaseModel):
    decision:        str
    risk_score:      float
    reason:          str
    modality_scores: dict
    latency_ms:      float


# ── Health check ──────────────────────────────────────
@app.get("/health")
def health():
    """Check if the API is alive and models are loaded."""
    return {
        "status":  "healthy",
        "models":  ["distilbert", "mobilenetv2"],
        "version": "1.0.0"
    }


# ── Text moderation endpoint ──────────────────────────
@app.post("/moderate/text", response_model=ModerationResponse)
def moderate_text(request: TextRequest):
    """
    Accepts a text string.
    Returns decision: allow / review / block
    with risk score and reasoning.
    """
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400,
                            detail="Text cannot be empty")

    start = time.time()

    # Get toxicity probability from DistilBERT
    text_score = predict_text(
        request.text,
        TEXT_TOKENIZER,
        TEXT_MODEL
    )

    # Compute final decision
    result = compute_risk_score({"text": text_score,
                                 "image": None,
                                 "video": None})

    latency = round((time.time() - start) * 1000, 2)

    return ModerationResponse(
        decision        = result["decision"],
        risk_score      = result["risk_score"],
        reason          = result["reason"],
        modality_scores = result["modality_scores"],
        latency_ms      = latency
    )


# ── Image moderation endpoint ─────────────────────────
@app.post("/moderate/image", response_model=ModerationResponse)
async def moderate_image(file: UploadFile = File(...)):
    """
    Accepts an image file upload.
    Returns decision with risk score.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,
                            detail="File must be an image")

    start = time.time()

    image_bytes = await file.read()
    image_score = predict_image(image_bytes, IMAGE_MODEL)

    result = compute_risk_score({"text":  None,
                                 "image": image_score,
                                 "video": None})

    latency = round((time.time() - start) * 1000, 2)

    return ModerationResponse(
        decision        = result["decision"],
        risk_score      = result["risk_score"],
        reason          = result["reason"],
        modality_scores = result["modality_scores"],
        latency_ms      = latency
    )


# ── Video moderation endpoint ─────────────────────────
@app.post("/moderate/video", response_model=ModerationResponse)
async def moderate_video_endpoint(file: UploadFile = File(...)):
    """
    Accepts a video file upload.
    Samples frames, runs image model on each.
    Returns decision with flagged frame ratio.
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400,
                            detail="File must be a video")

    start = time.time()

    # Save video to temp file — OpenCV needs a file path
    video_bytes = await file.read()
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4"
    ) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        result_raw   = moderate_video(tmp_path, VIDEO_MODEL)
        video_score  = result_raw.get("flagged_ratio", 0.0)

        result = compute_risk_score({"text":  None,
                                     "image": None,
                                     "video": video_score})
    finally:
        os.unlink(tmp_path)   # always clean up temp file

    latency = round((time.time() - start) * 1000, 2)

    return ModerationResponse(
        decision        = result["decision"],
        risk_score      = result["risk_score"],
        reason          = result["reason"],
        modality_scores = result["modality_scores"],
        latency_ms      = latency
    )


# ── Combined multimodal endpoint ──────────────────────
@app.post("/moderate/all")
async def moderate_all(
    text:  str        = None,
    image: UploadFile = File(None),
    video: UploadFile = File(None)
):
    """
    The most powerful endpoint — accepts any combination
    of text + image + video and fuses all scores together.
    """
    start  = time.time()
    scores = {"text": None, "image": None, "video": None}

    if text:
        scores["text"] = predict_text(
            text, TEXT_TOKENIZER, TEXT_MODEL
        )

    if image:
        image_bytes    = await image.read()
        scores["image"] = predict_image(image_bytes, IMAGE_MODEL)

    if video:
        video_bytes = await video.read()
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4"
        ) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        try:
            result_raw     = moderate_video(tmp_path, VIDEO_MODEL)
            scores["video"] = result_raw.get("flagged_ratio", 0.0)
        finally:
            os.unlink(tmp_path)

    result  = compute_risk_score(scores)
    latency = round((time.time() - start) * 1000, 2)

    return {**result, "latency_ms": latency}