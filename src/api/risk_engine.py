# src/api/risk_engine.py
# Takes scores from all three models and combines them
# into one final decision: allow, review, or block.
# This is called a "fusion layer" in production systems.

# Risk thresholds — tune these based on your use case
BLOCK_THRESHOLD  = 0.75   # above this → block immediately
REVIEW_THRESHOLD = 0.45   # above this → send to human review

# How much each modality contributes to final score
# Text is weighted highest because it's most reliable
WEIGHTS = {
    "text":  0.50,
    "image": 0.35,
    "video": 0.15
}


def compute_risk_score(scores: dict) -> dict:
    """
    scores = {
        "text":  0.92,   # probability of being toxic
        "image": 0.10,   # probability of being flagged
        "video": None    # None if not provided
    }
    Returns final score + decision.
    """
    active_scores  = {}
    active_weights = {}

    # Only include modalities that were actually provided
    for modality, score in scores.items():
        if score is not None:
            active_scores[modality]  = score
            active_weights[modality] = WEIGHTS[modality]

    if not active_scores:
        return {
            "risk_score": 0.0,
            "decision":   "allow",
            "reason":     "no content provided"
        }

    # Normalize weights so they sum to 1.0
    total_weight = sum(active_weights.values())
    normalized   = {k: v / total_weight
                    for k, v in active_weights.items()}

    # Weighted average of all active scores
    final_score = sum(
        active_scores[m] * normalized[m]
        for m in active_scores
    )
    final_score = round(final_score, 4)

    # Make decision based on thresholds
    if final_score >= BLOCK_THRESHOLD:
        decision = "block"
        reason   = f"risk score {final_score} exceeds block threshold"
    elif final_score >= REVIEW_THRESHOLD:
        decision = "review"
        reason   = f"risk score {final_score} requires human review"
    else:
        decision = "allow"
        reason   = f"risk score {final_score} within acceptable range"

    return {
        "risk_score":      final_score,
        "decision":        decision,
        "reason":          reason,
        "modality_scores": active_scores
    }