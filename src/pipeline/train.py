"""
train.py

Trains and compares shot prediction models:
- Baseline: Logistic Regression
- Improved: XGBoost classifier

All experiments are logged with MLflow.
The best model (by ROC-AUC) is saved to models/best_model.joblib.
"""

import pandas as pd
import numpy as np
import joblib

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, classification_report
)
from xgboost import XGBClassifier

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
# AFTER — SQLite backend (MLflow 3.x compatible)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nba-shot-efficiency")


def load_splits() -> tuple:
    """
    Load train/test feature and target splits from disk.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) DataFrames/Series.
    """
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    print(f"📥 Loaded splits — Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Binary predicted labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        Dictionary of metric names and values.
    """
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
    }


def train_baseline(X_train, X_test, y_train, y_test) -> tuple[object, dict]:
    """
    Train a Logistic Regression baseline model and log it in MLflow.

    Returns:
        Tuple of (trained model, metrics dict).
    """
    with mlflow.start_run(run_name="baseline_logistic_regression"):
        params = {"C": 1.0, "max_iter": 500, "random_state": 42}
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"\n📊 Baseline (Logistic Regression): {metrics}")
        print(classification_report(y_test, y_pred, target_names=["Missed", "Made"]))
        return model, metrics


def train_xgboost(X_train, X_test, y_train, y_test) -> tuple[object, dict]:
    """
    Train an XGBoost classifier and log it in MLflow.

    Returns:
        Tuple of (trained model, metrics dict).
    """
    with mlflow.start_run(run_name="xgboost_classifier"):
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        print(f"\n📊 XGBoost: {metrics}")
        print(classification_report(y_test, y_pred, target_names=["Missed", "Made"]))
        return model, metrics


def save_best_model(baseline_model, xgb_model, baseline_metrics, xgb_metrics):
    """
    Compare ROC-AUC scores and save the best model to models/.

    Args:
        baseline_model: Trained logistic regression model.
        xgb_model: Trained XGBoost model.
        baseline_metrics: Metrics dict for baseline.
        xgb_metrics: Metrics dict for XGBoost.
    """
    if xgb_metrics["roc_auc"] >= baseline_metrics["roc_auc"]:
        best_model = xgb_model
        best_name = "XGBoost"
        best_metrics = xgb_metrics
    else:
        best_model = baseline_model
        best_name = "Logistic Regression"
        best_metrics = baseline_metrics

    out_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, out_path)
    print(f"\n🏆 Best model: {best_name} | ROC-AUC: {best_metrics['roc_auc']} | Saved to {out_path}")


def run_training():
    """Run full model training pipeline."""
    X_train, X_test, y_train, y_test = load_splits()

    print("\n🔵 Training Baseline Model...")
    baseline_model, baseline_metrics = train_baseline(X_train, X_test, y_train, y_test)

    print("\n🟠 Training XGBoost Model...")
    xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)

    save_best_model(baseline_model, xgb_model, baseline_metrics, xgb_metrics)

    print("\n📈 MLflow UI: run `mlflow ui` and open http://localhost:5000")


if __name__ == "__main__":
    run_training()