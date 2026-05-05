# src/monitoring/retrain_trigger.py
# When drift is detected this module triggers retraining.
# In production this would submit a job to a cloud
# training cluster. Here it reruns training locally
# and logs the new run to MLflow automatically.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import mlflow
from datetime import datetime


def should_retrain() -> bool:
    """
    Checks drift detection results and decides
    whether retraining is needed.
    Returns True if retraining should be triggered.
    """
    from src.monitoring.monitor import detect_drift
    drift_result = detect_drift()

    if not drift_result["drift_detected"]:
        print("No drift detected. Retraining not needed.")
        return False

    # Check severity — only retrain on high severity
    alerts = drift_result.get("alerts", [])
    high_severity = any(
        a["severity"] == "high" for a in alerts
    )

    if high_severity:
        print(f"High severity drift detected: {alerts}")
        print("Retraining recommended.")
        return True

    print(f"Medium severity drift: {alerts}")
    print("Monitoring closely. No retraining yet.")
    return False


def trigger_retraining(reason: str = "drift_detected"):
    """
    Triggers the retraining pipeline.
    Logs the retraining event to MLflow for full
    audit trail — you can see every time the model
    was retrained and why.
    """
    print(f"\nTriggering retraining. Reason: {reason}")
    print("=" * 50)

    mlflow.set_experiment("retraining-log")

    with mlflow.start_run(run_name=f"retrain-{reason}"):

        mlflow.log_params({
            "trigger_reason": reason,
            "trigger_time":   datetime.now().isoformat(),
            "triggered_by":   "drift_detector"
        })

        try:
            # Import and run text model training
            print("Retraining text model...")
            from src.models.train_text import train_text_model
            train_text_model()
            mlflow.log_metric("text_retrain_success", 1)
            print("Text model retrained successfully.")

        except Exception as e:
            mlflow.log_metric("text_retrain_success", 0)
            mlflow.log_param("error", str(e))
            print(f"Retraining failed: {e}")
            raise

        mlflow.log_param("status", "completed")
        print("Retraining complete. New model saved.")
        print("=" * 50)


def run_retraining_check():
    """
    Main function — checks drift and retrains if needed.
    In production this runs on a schedule (every hour).
    You'd use Apache Airflow or a cron job to call this.
    """
    print(f"Running retraining check at {datetime.now()}")

    if should_retrain():
        trigger_retraining(reason="score_drift")
    else:
        print("System healthy. No action needed.")


if __name__ == "__main__":
    run_retraining_check()