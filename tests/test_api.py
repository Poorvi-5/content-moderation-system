# tests/test_api.py
# Automated tests for the moderation API.
# These run in the CI/CD pipeline on every code push.
# If any test fails, deployment is blocked automatically.

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """
    Creates a test client that talks to our API
    without actually starting a real server.
    scope=module means models load once for all tests
    — not once per test, which would be very slow.
    """
    from src.api.main import app
    return TestClient(app)


def test_health_check(client):
    """API should always return healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data


def test_text_moderation_clean(client):
    """Clean text should return allow decision."""
    response = client.post("/moderate/text", json={
        "text": "I love this community, everyone is so helpful!",
        "user_id": "test_user"
    })
    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert "risk_score" in data
    assert "latency_ms" in data
    assert data["risk_score"] >= 0.0
    assert data["risk_score"] <= 1.0
    assert data["decision"] in ["allow", "review", "block"]


def test_text_moderation_toxic(client):
    """Toxic text should return higher risk score than clean text."""
    clean_response = client.post("/moderate/text", json={
        "text": "Have a wonderful day!",
        "user_id": "test_user"
    })
    toxic_response = client.post("/moderate/text", json={
        "text": "I will destroy you, you worthless piece of garbage",
        "user_id": "test_user"
    })

    clean_score = clean_response.json()["risk_score"]
    toxic_score = toxic_response.json()["risk_score"]

    # Toxic text must score higher than clean text
    assert toxic_score > clean_score


def test_text_empty_input(client):
    """Empty text should return 400 error."""
    response = client.post("/moderate/text", json={
        "text": "",
        "user_id": "test_user"
    })
    assert response.status_code == 400


def test_risk_score_range(client):
    """Risk score must always be between 0 and 1."""
    texts = [
        "Hello world",
        "I hate everything",
        "Great job team",
        "You are stupid",
        "Thanks for your help"
    ]
    for text in texts:
        response = client.post("/moderate/text", json={
            "text": text,
            "user_id": "test_user"
        })
        score = response.json()["risk_score"]
        assert 0.0 <= score <= 1.0, \
            f"Score {score} out of range for: {text}"


def test_decision_values(client):
    """Decision must always be one of three valid values."""
    response = client.post("/moderate/text", json={
        "text": "some random text here",
        "user_id": "test_user"
    })
    assert response.json()["decision"] in ["allow", "review", "block"]


def test_risk_engine_directly():
    """Test the risk scoring logic in isolation."""
    from src.api.risk_engine import compute_risk_score

    # High text score should block
    result = compute_risk_score({
        "text": 0.95, "image": None, "video": None
    })
    assert result["decision"] == "block"
    assert result["risk_score"] >= 0.75

    # Low score should allow
    result = compute_risk_score({
        "text": 0.05, "image": None, "video": None
    })
    assert result["decision"] == "allow"

    # Medium score should review
    result = compute_risk_score({
        "text": 0.60, "image": None, "video": None
    })
    assert result["decision"] == "review"


def test_multimodal_fusion():
    """Test that multimodal scores are fused correctly."""
    from src.api.risk_engine import compute_risk_score

    # Combining high text + high image should still block
    result = compute_risk_score({
        "text": 0.90, "image": 0.85, "video": None
    })
    assert result["decision"] == "block"
    assert "text" in result["modality_scores"]
    assert "image" in result["modality_scores"]