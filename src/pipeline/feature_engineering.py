"""
feature_engineering.py

Builds ML-ready features from cleaned shot chart data.
- Creates rolling form features (last N games shot efficiency)
- Creates zone-level efficiency aggregations
- Encodes categorical variables
- Builds and saves train/test splits

Target: SHOT_MADE_FLAG (binary classification — made=1, missed=0)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = ["ACTION_TYPE", "SHOT_TYPE", "SHOT_ZONE_BASIC",
                    "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE"]
NUMERIC_COLS = [
    "SHOT_DISTANCE", "LOC_X", "LOC_Y", "PERIOD",
    "TIME_REMAINING_SECS", "IS_THREE_POINTER", "IS_HOME",
    "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME", "CLOSE_DEF_DIST",
]
TARGET = "SHOT_MADE_FLAG"
ROLLING_WINDOW = 10  # number of games for rolling form


def add_rolling_form(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Add rolling shot efficiency features based on recent game history.
    Computes rolling FG% over the last N games per game date.

    Args:
        df: Clean shot chart DataFrame with GAME_DATE and SHOT_MADE_FLAG.
        window: Number of recent games to include in rolling window.

    Returns:
        DataFrame with ROLLING_FG_PCT and ROLLING_ATTEMPTS columns added.
    """
    df = df.copy().sort_values("GAME_DATE")

    game_stats = (
        df.groupby("GAME_DATE")["SHOT_MADE_FLAG"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "game_fg_pct", "count": "game_attempts"})
        .reset_index()
    )
    game_stats["ROLLING_FG_PCT"] = (
        game_stats["game_fg_pct"]
        .rolling(window=window, min_periods=1).mean().shift(1)
    )
    game_stats["ROLLING_ATTEMPTS"] = (
        game_stats["game_attempts"]
        .rolling(window=window, min_periods=1).mean().shift(1)
    )
    df = df.merge(
        game_stats[["GAME_DATE", "ROLLING_FG_PCT", "ROLLING_ATTEMPTS"]],
        on="GAME_DATE", how="left"
    )
    df["ROLLING_FG_PCT"] = df["ROLLING_FG_PCT"].fillna(df["SHOT_MADE_FLAG"].mean())
    df["ROLLING_ATTEMPTS"] = df["ROLLING_ATTEMPTS"].fillna(10.0)
    return df


def add_zone_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical zone-level efficiency as a feature.
    For each shot, computes the historical FG% for that shot zone.

    Args:
        df: DataFrame with SHOT_ZONE_BASIC and SHOT_MADE_FLAG.

    Returns:
        DataFrame with ZONE_FG_PCT column added.
    """
    df = df.copy()
    zone_eff = (
        df.groupby("SHOT_ZONE_BASIC")["SHOT_MADE_FLAG"]
        .mean()
        .rename("ZONE_FG_PCT")
    )
    df = df.join(zone_eff, on="SHOT_ZONE_BASIC")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns and save encoders.

    Args:
        df: DataFrame with raw categorical columns.

    Returns:
        Tuple of (encoded DataFrame, dict of label encoders).
    """
    encoders = {}
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the final feature matrix X and target vector y.

    Args:
        df: Fully processed DataFrame.

    Returns:
        Tuple of (X features DataFrame, y target Series).
    """
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS + ["ROLLING_FG_PCT", "ROLLING_ATTEMPTS", "ZONE_FG_PCT"]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0)
    y = df[TARGET]
    print(f"📐 Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y


def run_feature_engineering(clean_path: str = "data/processed/shots_clean.csv") -> dict:
    """
    Full feature engineering pipeline: load clean data, enrich, encode,
    split into train/test, and save all artifacts.

    Args:
        clean_path: Path to the cleaned CSV.

    Returns:
        Dict with keys 'X_train', 'X_test', 'y_train', 'y_test'.
    """
    df = pd.read_csv(clean_path)
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    else:
        # Create a synthetic GAME_DATE from GAME_ID or row order for rolling form
        print("⚠️  No GAME_DATE found — creating synthetic date from row order")
        df["GAME_DATE"] = pd.date_range(start="2014-10-01", periods=len(df), freq="1h")
    print(f"📥 Loaded {len(df)} clean shots")

    df = add_rolling_form(df)
    df = add_zone_efficiency(df)
    df, encoders = encode_categoricals(df)

    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✂️  Train: {len(X_train)} | Test: {len(X_test)}")

    # Save artifacts
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    df.to_csv(PROCESSED_DIR / "shots_processed.csv", index=False)
    joblib.dump(encoders, PROCESSED_DIR / "label_encoders.joblib")

    print("💾 Saved: X_train, X_test, y_train, y_test, shots_processed.csv, label_encoders.joblib")
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


if __name__ == "__main__":
    run_feature_engineering()
