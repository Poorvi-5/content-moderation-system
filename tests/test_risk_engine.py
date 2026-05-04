# tests/test_risk_engine.py
# These tests only test logic — no models needed.
# This is what the CI/CD pipeline runs on GitHub.
# Fast, reliable, no GPU or model files required.

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.risk_engine import compute_risk_score


def test_high_text_score_blocks():
    result = compute_risk_score({
        "text": 0.95, "image": None, "video": None
    })
    assert result["decision"] == "block"
    assert result["risk_score"] >= 0.75


def test_low_score_allows():
    result = compute_risk_score({
        "text": 0.05, "image": None, "video": None
    })
    assert result["decision"] == "allow"


def test_medium_score_reviews():
    result = compute_risk_score({
        "text": 0.60, "image": None, "video": None
    })
    assert result["decision"] == "review"


def test_score_always_between_0_and_1():
    for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = compute_risk_score({
            "text": score, "image": None, "video": None
        })
        assert 0.0 <= result["risk_score"] <= 1.0


def test_multimodal_fusion():
    result = compute_risk_score({
        "text": 0.9, "image": 0.8, "video": None
    })
    assert result["decision"] == "block"
    assert "text" in result["modality_scores"]
    assert "image" in result["modality_scores"]


def test_no_content_allows():
    result = compute_risk_score({
        "text": None, "image": None, "video": None
    })
    assert result["decision"] == "allow"


def test_modality_scores_returned():
    result = compute_risk_score({
        "text": 0.8, "image": 0.6, "video": None
    })
    assert "modality_scores" in result
    assert "text" in result["modality_scores"]
    assert "image" in result["modality_scores"]