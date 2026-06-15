"""
fetch_data.py

Loads NBA shot chart data from a pre-downloaded Kaggle dataset.
Dataset: https://www.kaggle.com/datasets/dansbecker/nba-shot-logs
Place the file at: data/raw/shot_logs.csv

The live NBA API fetch is kept as an optional fallback (use_api=True)
but is disabled by default due to Cloudflare blocking on stats.nba.com.
"""

import argparse
import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV = RAW_DATA_DIR / "shot_logs.csv"


# ── Column mapping: Kaggle names → our project's standard names ──────────────
COLUMN_MAP = {
    "player_name":        "PLAYER_NAME",
    "shot_made_flag":     "SHOT_MADE_FLAG",
    "shot_distance":      "SHOT_DISTANCE",
    "loc_x":              "LOC_X",
    "loc_y":              "LOC_Y",
    "period":             "PERIOD",
    "minutes_remaining":  "MINUTES_REMAINING",
    "seconds_remaining":  "SECONDS_REMAINING",
    "action_type":        "ACTION_TYPE",
    "shot_type":          "SHOT_TYPE",
    "shot_zone_basic":    "SHOT_ZONE_BASIC",
    "shot_zone_area":     "SHOT_ZONE_AREA",
    "shot_zone_range":    "SHOT_ZONE_RANGE",
    "game_date":          "GAME_DATE",
    "htm":                "HTM",
    "vtm":                "VTM",
    "team_name":          "TEAM_NAME",
}


def load_from_kaggle(
    filepath: str = str(DEFAULT_CSV),
    player_name: str = None,
) -> pd.DataFrame:
    """
    Load NBA shot data from a pre-downloaded Kaggle CSV file.

    Args:
        filepath: Path to the shot_logs.csv file.
        player_name: Optional player name filter (partial match, case-insensitive).

    Returns:
        Standardised DataFrame ready for preprocessing.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"\n❌ File not found: {path}\n"
            f"   ➜ Download it from: https://www.kaggle.com/datasets/dansbecker/nba-shot-logs\n"
            f"   ➜ Place it at: data/raw/shot_logs.csv"
        )

    df = pd.read_csv(filepath)
    print(f"📥 Loaded {len(df)} rows from {filepath}")

    # Normalise column names to uppercase so the rest of the pipeline works
    df.columns = [c.strip().upper() for c in df.columns]

    # Apply column mapping for any columns that exist
    reverse_map = {v.upper(): v for v in COLUMN_MAP.keys()}
    rename = {
        col: COLUMN_MAP[reverse_map[col]]
        for col in df.columns
        if col in reverse_map
    }
    df = df.rename(columns=rename)

    # Filter to a specific player if requested
    if player_name and "PLAYER_NAME" in df.columns:
        mask = df["PLAYER_NAME"].str.contains(player_name, case=False, na=False)
        df = df[mask].copy()
        if len(df) == 0:
            raise ValueError(
                f"No data found for player '{player_name}'. "
                f"Check the name matches exactly what's in the dataset."
            )
        print(f"🏀 Filtered to '{player_name}': {len(df)} shots")
        # The dansbecker dataset uses FGM (0/1) instead of SHOT_MADE_FLAG — rename it
    if "FGM" in df.columns and "SHOT_MADE_FLAG" not in df.columns:
        df["SHOT_MADE_FLAG"] = df["FGM"].astype(int)
        print("✅ Created SHOT_MADE_FLAG from FGM column")

    # SHOT_RESULT is 'made'/'missed' — create flag if FGM not available
    elif "SHOT_RESULT" in df.columns and "SHOT_MADE_FLAG" not in df.columns:
        df["SHOT_MADE_FLAG"] = (df["SHOT_RESULT"].str.lower() == "made").astype(int)
        print("✅ Created SHOT_MADE_FLAG from SHOT_RESULT column")
    print(f"✅ Columns available: {list(df.columns)}")
    return df


def save_shot_data(df: pd.DataFrame, player_name: str = "all_players") -> Path:
    """
    Save the loaded DataFrame to data/raw/ as a standardised CSV.

    Args:
        df: Shot chart DataFrame.
        player_name: Used to name the output file.

    Returns:
        Path to the saved file.
    """
    safe_name = player_name.replace(" ", "_").lower()
    out_path = RAW_DATA_DIR / f"{safe_name}_shots.csv"
    df.to_csv(out_path, index=False)
    print(f"💾 Saved to: {out_path} ({len(df)} rows)")
    return out_path


def fetch_and_save(
    player_name: str = None,
    filepath: str = str(DEFAULT_CSV),
) -> pd.DataFrame:
    """
    Full pipeline: load from Kaggle CSV → optionally filter by player → save.

    Args:
        player_name: Optional player name to filter (e.g. 'LeBron James').
        filepath: Path to the Kaggle shot_logs.csv file.

    Returns:
        The loaded (and optionally filtered) DataFrame.
    """
    df = load_from_kaggle(filepath=filepath, player_name=player_name)
    label = player_name if player_name else "all_players"
    save_shot_data(df, label)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load NBA shot data from Kaggle CSV.")
    parser.add_argument("--player", type=str, default=None,
                        help="Filter by player name (e.g. 'LeBron James'). Leave empty to load all players.")
    parser.add_argument("--filepath", type=str, default=str(DEFAULT_CSV),
                        help="Path to shot_logs.csv")
    args = parser.parse_args()

    fetch_and_save(player_name=args.player, filepath=args.filepath)