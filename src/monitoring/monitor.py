# src/monitoring/monitor.py
# Tracks every prediction the API makes in real time.
# Stores metrics to a local SQLite database.
# Detects when model behavior starts drifting.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sqlite3
import json
from datetime import datetime, timedelta
from collections import defaultdict

DB_PATH = "monitoring/metrics.db"


def init_db():
    """
    Creates the SQLite database and tables if they
    don't exist yet. Called once when server starts.
    """
    os.makedirs("monitoring", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    # Every prediction gets logged here
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT NOT NULL,
            modality     TEXT NOT NULL,
            risk_score   REAL NOT NULL,
            decision     TEXT NOT NULL,
            latency_ms   REAL NOT NULL,
            input_length INTEGER
        )
    """)

    # Human feedback on decisions
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp      TEXT NOT NULL,
            prediction_id  INTEGER,
            correct_label  INTEGER NOT NULL,
            model_decision TEXT NOT NULL,
            was_correct    INTEGER NOT NULL
        )
    """)

    # Drift alerts get stored here
    c.execute("""
        CREATE TABLE IF NOT EXISTS drift_alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            alert_type  TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            current_val REAL NOT NULL,
            baseline_val REAL NOT NULL,
            severity    TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("Monitoring database initialized.")


def log_prediction(modality: str, risk_score: float,
                   decision: str, latency_ms: float,
                   input_length: int = 0):
    """
    Logs every single prediction to the database.
    Called automatically by the API after each request.
    This is how we build up the monitoring data over time.
    """
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        INSERT INTO predictions
        (timestamp, modality, risk_score, decision,
         latency_ms, input_length)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        modality,
        risk_score,
        decision,
        latency_ms,
        input_length
    ))

    conn.commit()
    conn.close()


def log_feedback(prediction_id: int, correct_label: int,
                 model_decision: str):
    """
    Logs human moderator feedback.
    correct_label: 1=toxic/flagged, 0=clean/safe
    model_decision: what the model said (allow/block/review)
    """
    # Was the model correct?
    model_said_toxic = model_decision in ["block", "review"]
    was_correct      = int(model_said_toxic == bool(correct_label))

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        INSERT INTO feedback
        (timestamp, prediction_id, correct_label,
         model_decision, was_correct)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        prediction_id,
        correct_label,
        model_decision,
        was_correct
    ))

    conn.commit()
    conn.close()

    return {"logged": True, "was_correct": bool(was_correct)}


def get_metrics_summary(hours: int = 24) -> dict:
    """
    Returns a summary of the last N hours of predictions.
    This is what the /monitoring/metrics endpoint returns.
    """
    conn  = sqlite3.connect(DB_PATH)
    c     = conn.cursor()
    since = (datetime.now() - timedelta(hours=hours)).isoformat()

    # Total predictions
    c.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE timestamp > ?
    """, (since,))
    total = c.fetchone()[0]

    if total == 0:
        conn.close()
        return {"total_predictions": 0,
                "message": "No predictions in timeframe"}

    # Average risk score
    c.execute("""
        SELECT AVG(risk_score), AVG(latency_ms)
        FROM predictions WHERE timestamp > ?
    """, (since,))
    avg_risk, avg_latency = c.fetchone()

    # Decision distribution
    c.execute("""
        SELECT decision, COUNT(*) FROM predictions
        WHERE timestamp > ?
        GROUP BY decision
    """, (since,))
    decisions = dict(c.fetchall())

    # Block rate — key production metric
    block_rate = decisions.get("block", 0) / total

    # Average latency by modality
    c.execute("""
        SELECT modality, AVG(latency_ms)
        FROM predictions WHERE timestamp > ?
        GROUP BY modality
    """, (since,))
    latency_by_modality = dict(c.fetchall())

    conn.close()

    return {
        "timeframe_hours":      hours,
        "total_predictions":    total,
        "avg_risk_score":       round(avg_risk, 4),
        "avg_latency_ms":       round(avg_latency, 2),
        "block_rate":           round(block_rate, 4),
        "decision_distribution": decisions,
        "latency_by_modality":  latency_by_modality
    }


def detect_drift() -> dict:
    """
    Checks for two types of drift:
    1. Score drift — average risk score shifted significantly
    2. Latency drift — API is getting slower over time
    """
    conn     = sqlite3.connect(DB_PATH)
    c        = conn.cursor()
    now      = datetime.now()
    recent   = (now - timedelta(hours=6)).isoformat()
    previous = (now - timedelta(hours=12)).isoformat()

    c.execute("""
        SELECT AVG(risk_score), AVG(latency_ms), COUNT(*)
        FROM predictions WHERE timestamp > ?
    """, (recent,))
    recent_avg_score, recent_avg_latency, recent_count = c.fetchone()

    c.execute("""
        SELECT AVG(risk_score), AVG(latency_ms), COUNT(*)
        FROM predictions
        WHERE timestamp > ? AND timestamp <= ?
    """, (previous, recent))
    prev_avg_score, prev_avg_latency, prev_count = c.fetchone()

    conn.close()

    alerts = []

    if not recent_count or recent_count < 10:
        return {
            "drift_detected": False,
            "reason": "Not enough recent predictions to detect drift",
            "recent_count": recent_count or 0
        }

    if prev_count and prev_count >= 10:

        if prev_avg_score and prev_avg_score > 0:
            score_change = abs(
                recent_avg_score - prev_avg_score
            ) / prev_avg_score

            if score_change > 0.20:
                severity = "high" if score_change > 0.4 else "medium"
                alerts.append({
                    "type":       "score_drift",
                    "metric":     "avg_risk_score",
                    "current":    round(recent_avg_score, 4),
                    "baseline":   round(prev_avg_score, 4),
                    "change_pct": round(score_change * 100, 1),
                    "severity":   severity
                })

        if prev_avg_latency and prev_avg_latency > 0:
            latency_change = (
                recent_avg_latency - prev_avg_latency
            ) / prev_avg_latency

            if latency_change > 0.30:
                alerts.append({
                    "type":       "latency_drift",
                    "metric":     "avg_latency_ms",
                    "current":    round(recent_avg_latency, 2),
                    "baseline":   round(prev_avg_latency, 2),
                    "change_pct": round(latency_change * 100, 1),
                    "severity":   "medium"
                })

    drift_detected = len(alerts) > 0

    if drift_detected:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        for alert in alerts:
            c.execute("""
                INSERT INTO drift_alerts
                (timestamp, alert_type, metric_name,
                 current_val, baseline_val, severity)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                now.isoformat(),
                alert["type"],
                alert["metric"],
                alert["current"],
                alert["baseline"],
                alert["severity"]
            ))
        conn.commit()
        conn.close()

    return {
        "drift_detected":  drift_detected,
        "alerts":          alerts,
        "recent_count":    recent_count,
        "recommendation":  "trigger_retraining" if drift_detected
                           else "no_action_needed"
    }