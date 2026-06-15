"""
monitor.py

Generates Evidently AI monitoring reports for NBA Scout AI.
Compares a reference dataset (training data) vs current data (recent shots).
Produces:
- HTML drift + performance report
- JSON summary for the API and frontend
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "SHOT_DISTANCE", "LOC_X", "LOC_Y", "PERIOD", "TIME_REMAINING_SECS",
    "IS_THREE_POINTER", "IS_HOME", "ROLLING_FG_PCT", "ROLLING_ATTEMPTS",
    "ZONE_FG_PCT", "ACTION_TYPE", "SHOT_TYPE", "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE"
]
TARGET_COL = "SHOT_MADE_FLAG"


def load_reference_and_current(current_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split processed data into reference (training) and current (recent) sets.

    Args:
        current_ratio: Fraction of most recent data to treat as 'current'.

    Returns:
        Tuple of (reference_df, current_df).
    """
    df = pd.read_csv(PROCESSED_DIR / "shots_processed.csv", parse_dates=["GAME_DATE"])
    df = df.sort_values("GAME_DATE").dropna(subset=[TARGET_COL])

    available_features = [c for c in FEATURE_COLS if c in df.columns] + [TARGET_COL]
    df = df[available_features].fillna(0)

    split_idx = int(len(df) * (1 - current_ratio))
    reference = df.iloc[:split_idx].copy()
    current = df.iloc[split_idx:].copy()

    print(f"📊 Reference: {len(reference)} rows | Current: {len(current)} rows")
    return reference, current


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """
    Run Evidently data drift report and save to HTML.

    Args:
        reference: Reference (training-time) DataFrame.
        current: Current (recent) DataFrame.

    Returns:
        Dict summary of drift results.
    """
    report = Report(metrics=[
        DataDriftPreset(),
        DatasetMissingValuesMetric(),
    ])
    report.run(reference_data=reference, current_data=current)
    out_path = REPORTS_DIR / "monitoring_report.html"
    report.save_html(str(out_path))
    print(f"💾 Saved HTML report: {out_path}")

    result = report.as_dict()
    drift_metric = next(
        (m for m in result.get("metrics", []) if m.get("metric") == "DatasetDriftMetric"),
        {}
    )
    drift_result = drift_metric.get("result", {})

    return {
        "drift_detected": drift_result.get("dataset_drift", False),
        "number_of_drifted_columns": drift_result.get("number_of_drifted_columns", 0),
        "share_of_drifted_columns": round(drift_result.get("share_of_drifted_columns", 0.0), 3),
    }


def build_summary(reference: pd.DataFrame, current: pd.DataFrame, drift_info: dict) -> dict:
    """
    Build a compact JSON summary for the API monitoring endpoint.

    Args:
        reference: Reference DataFrame.
        current: Current DataFrame.
        drift_info: Dict from drift report.

    Returns:
        Summary dict saved to reports/monitoring_summary.json.
    """
    avg_confidence = float(np.abs(current[TARGET_COL] - 0.5).mean()) if len(current) > 0 else 0.0

    drift_status = "⚠️ Drift detected" if drift_info["drift_detected"] else "✅ No drift"
    summary_text = (
        f"{drift_status} — "
        f"{drift_info['number_of_drifted_columns']} of {len(reference.columns)} columns drifted "
        f"({drift_info['share_of_drifted_columns']:.0%}). "
        f"Current window: {len(current)} shots."
    )

    summary = {
        "status": "drift_detected" if drift_info["drift_detected"] else "healthy",
        "drift_detected": drift_info["drift_detected"],
        "total_predictions": len(current),
        "avg_confidence": round(avg_confidence, 3),
        "report_available": True,
        "summary": summary_text,
    }

    out_path = REPORTS_DIR / "monitoring_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"💾 Saved monitoring summary: {out_path}")
    print(f"📋 {summary_text}")
    return summary


def run_monitoring():
    """Run full monitoring pipeline: load data → drift report → save summary."""
    reference, current = load_reference_and_current()
    drift_info = run_drift_report(reference, current)
    build_summary(reference, current, drift_info)


if __name__ == "__main__":
    run_monitoring()
