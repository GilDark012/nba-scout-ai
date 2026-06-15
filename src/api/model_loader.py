"""
model_loader.py

Loads the trained ML model and label encoders once at startup.
Provides a singleton accessor to avoid reloading on every request.
"""

import joblib
import os
from pathlib import Path

_model = None
_encoders = None

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
ENCODERS_PATH = os.getenv("ENCODERS_PATH", "data/processed/label_encoders.joblib")


def load_model():
    """
    Load and cache the best trained model from disk.

    Returns:
        Trained scikit-learn or XGBoost model.
    """
    global _model
    if _model is None:
        path = Path(MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")
        _model = joblib.load(path)
    return _model


def load_encoders() -> dict:
    """
    Load and cache label encoders for categorical features.

    Returns:
        Dict of {column_name: LabelEncoder}.
    """
    global _encoders
    if _encoders is None:
        path = Path(ENCODERS_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Encoders not found at {path}. Run feature_engineering.py first.")
        _encoders = joblib.load(path)
    return _encoders


def get_model():
    """Return cached model instance."""
    return load_model()


def get_encoders():
    """Return cached encoders instance."""
    return load_encoders()
