"""
evaluate.py

Evaluates the best trained model.
Saves:
- Confusion matrix plot
- ROC curve plot
- Feature importance chart
- Zone-performance summary
- Text evaluation summary
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, classification_report
)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts() -> tuple:
    """
    Load test data and best model.

    Returns:
        Tuple of (X_test, y_test, model, processed_df).
    """
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    df = pd.read_csv(PROCESSED_DIR / "shots_processed.csv")
    return X_test, y_test, model, df


def plot_confusion_matrix(y_test, y_pred, out_path: Path):
    """
    Plot and save confusion matrix.

    Args:
        y_test: True labels.
        y_pred: Predicted labels.
        out_path: File path for saving the figure.
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Missed", "Made"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Shot Prediction")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"💾 Saved confusion matrix: {out_path}")


def plot_roc_curve(y_test, y_prob, out_path: Path):
    """
    Plot and save ROC curve.

    Args:
        y_test: True labels.
        y_prob: Predicted probabilities.
        out_path: File path for saving the figure.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Shot Prediction")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"💾 Saved ROC curve: {out_path}")


def plot_feature_importance(model, feature_names: list, out_path: Path):
    """
    Plot and save feature importance chart if available.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature column names.
        out_path: File path for saving the figure.
    """
    if not hasattr(model, "feature_importances_"):
        print("⚠️ Model has no feature_importances_ attribute. Skipping.")
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        color="steelblue"
    )
    ax.set_title("Top 15 Feature Importances")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"💾 Saved feature importance: {out_path}")


def plot_zone_performance(df: pd.DataFrame, out_path: Path):
    """
    Plot zone-by-zone FG% bar chart.

    Args:
        df: Processed DataFrame with SHOT_ZONE_BASIC and SHOT_MADE_FLAG.
        out_path: File path for saving the figure.
    """
    if "SHOT_ZONE_BASIC" not in df.columns:
        print("⚠️ SHOT_ZONE_BASIC not found. Skipping zone chart.")
        return
    zone_stats = (
        df.groupby("SHOT_ZONE_BASIC")["SHOT_MADE_FLAG"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "FG_PCT", "count": "Attempts"})
        .sort_values("FG_PCT", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(zone_stats.index, zone_stats["FG_PCT"] * 100, color="coral", edgecolor="white")
    for bar, (_, row) in zip(bars, zone_stats.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{row['Attempts']}x", ha="center", va="bottom", fontsize=8)
    ax.set_title("FG% by Zone (with attempt counts)")
    ax.set_ylabel("FG%")
    ax.set_xlabel("Shot Zone")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"💾 Saved zone performance: {out_path}")


def write_text_summary(y_test, y_pred, y_prob, model, feature_names: list):
    """
    Write a plain-text evaluation summary to reports/.

    Args:
        y_test, y_pred, y_prob: Test labels and predictions.
        model: Trained model.
        feature_names: Feature column names.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    summary = f"""
NBA Scout AI — Model Evaluation Summary
========================================
Model type  : {type(model).__name__}
Test samples: {len(y_test)}
Accuracy    : {accuracy_score(y_test, y_pred):.4f}
ROC-AUC     : {roc_auc_score(y_test, y_prob):.4f}

Classification Report:
{classification_report(y_test, y_pred, target_names=["Missed", "Made"])}

Interpretation:
The model predicts whether an individual shot will be made or missed.
ROC-AUC > 0.65 is considered meaningful for shot prediction given inherent variability.
Feature importance highlights which spatial and situational factors most influence shot outcome.
"""
    out_path = Path("reports/evaluation_summary.txt")
    out_path.write_text(summary)
    print(f"💾 Saved evaluation summary: {out_path}")
    print(summary)


def run_evaluation():
    """Run full evaluation pipeline."""
    X_test, y_test, model, df = load_artifacts()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    feature_names = list(X_test.columns)

    plot_confusion_matrix(y_test, y_pred, FIGURES_DIR / "confusion_matrix.png")
    plot_roc_curve(y_test, y_prob, FIGURES_DIR / "roc_curve.png")
    plot_feature_importance(model, feature_names, FIGURES_DIR / "feature_importance.png")
    plot_zone_performance(df, FIGURES_DIR / "zone_performance.png")
    write_text_summary(y_test, y_pred, y_prob, model, feature_names)


if __name__ == "__main__":
    run_evaluation()