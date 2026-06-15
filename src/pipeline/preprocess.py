"""
preprocess.py

Cleans raw NBA shot chart data.
Selects relevant features, handles nulls, engineers basic flags,
and prepares a clean dataset for feature engineering and modeling.

Target variable: SHOT_MADE_FLAG (1 = made, 0 = missed)
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Replace USEFUL_COLUMNS at the top of preprocess.py
USEFUL_COLUMNS = [
    "GAME_ID", "MATCHUP", "LOCATION", "PERIOD",
    "GAME_CLOCK", "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME",
    "SHOT_DIST", "PTS_TYPE", "SHOT_RESULT", "FGM",
    "SHOT_MADE_FLAG", "CLOSEST_DEFENDER", "CLOSE_DEF_DIST",
    "player_name", "player_id",
    # Standard names if already remapped
    "PLAYER_NAME", "SHOT_DISTANCE", "ACTION_TYPE",
    "SHOT_TYPE", "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA",
    "SHOT_ZONE_RANGE", "LOC_X", "LOC_Y",
]


def load_raw(filepath: str) -> pd.DataFrame:
    """
    Load raw shot chart CSV file into a DataFrame.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Raw DataFrame.
    """
    df = pd.read_csv(filepath)
    print(f"📥 Loaded {len(df)} rows from {filepath}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw shot chart data:
    - Filter to useful columns.
    - Drop rows with null values in critical columns.
    - Parse game dates to datetime.
    - Normalize string columns.

    Args:
        df: Raw shot chart DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    available = [c for c in USEFUL_COLUMNS if c in df.columns]
    df = df[available].copy()

    critical = ["SHOT_MADE_FLAG", "SHOT_DISTANCE", "PERIOD"]
    before = len(df)
    df = df.dropna(subset=[c for c in critical if c in df.columns])
    print(f"🧹 Dropped {before - len(df)} rows with nulls. Remaining: {len(df)}")

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y%m%d", errors="coerce")

    str_cols = ["ACTION_TYPE", "SHOT_TYPE", "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA",
                "SHOT_ZONE_RANGE", "PLAYER_NAME", "TEAM_NAME"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    df["SHOT_MADE_FLAG"] = df["SHOT_MADE_FLAG"].astype(int)
    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features compatible with the Kaggle NBA shot logs dataset.
    """
    df = df.copy()

    # Shot distance — use SHOT_DIST if LOC_X/LOC_Y not available
    if "SHOT_DIST" in df.columns and "SHOT_DISTANCE" not in df.columns:
        df["SHOT_DISTANCE"] = df["SHOT_DIST"]

    # 3-pointer flag from PTS_TYPE (2 or 3)
    if "PTS_TYPE" in df.columns and "IS_THREE_POINTER" not in df.columns:
        df["IS_THREE_POINTER"] = (df["PTS_TYPE"] == 3).astype(int)

    # Home/away from LOCATION ('H' = home, 'A' = away)
    if "LOCATION" in df.columns and "IS_HOME" not in df.columns:
        df["IS_HOME"] = (df["LOCATION"].str.upper() == "H").astype(int)

    # Convert GAME_CLOCK "MM:SS" → total seconds remaining
    if "GAME_CLOCK" in df.columns and "TIME_REMAINING_SECS" not in df.columns:
        def clock_to_secs(val):
            try:
                parts = str(val).split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except Exception:
                return 0
        df["TIME_REMAINING_SECS"] = df["GAME_CLOCK"].apply(clock_to_secs)

    # Fill missing LOC_X / LOC_Y with 0 if not present
    if "LOC_X" not in df.columns:
        df["LOC_X"] = 0
    if "LOC_Y" not in df.columns:
        df["LOC_Y"] = 0

    # Fill missing zone columns with placeholder
    for col in ["SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "ACTION_TYPE", "SHOT_TYPE"]:
        if col not in df.columns:
            df[col] = "Unknown"

    # Player name normalisation
    if "player_name" in df.columns and "PLAYER_NAME" not in df.columns:
        df["PLAYER_NAME"] = df["player_name"].str.strip().str.title()

    return df

def save_clean(df: pd.DataFrame, filename: str = "shots_clean.csv") -> Path:
    """
    Save the cleaned DataFrame to data/processed/.

    Args:
        df: Cleaned DataFrame.
        filename: Output filename.

    Returns:
        Path to the saved file.
    """
    out_path = PROCESSED_DATA_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"💾 Saved clean data: {out_path} ({len(df)} rows, {len(df.columns)} cols)")
    return out_path


def run_preprocess(raw_path: str, output_name: str = "shots_clean.csv") -> pd.DataFrame:
    """
    Full preprocessing pipeline: load → clean → add features → save.

    Args:
        raw_path: Path to raw CSV file.
        output_name: Output filename in data/processed/.

    Returns:
        Cleaned and enriched DataFrame.
    """
    df = load_raw(raw_path)
    df = clean(df)
    df = add_basic_features(df)
    save_clean(df, output_name)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess raw NBA shot data.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw CSV")
    parser.add_argument("--output", type=str, default="shots_clean.csv")
    args = parser.parse_args()
    run_preprocess(args.input, args.output)
