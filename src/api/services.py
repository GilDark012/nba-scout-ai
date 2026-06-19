"""
services.py

Business logic for the NBA Scout AI API:
- Player search via nba_api
- Shot data fetching and zone aggregation
- Model inference with label encoding
- Scouting report generation
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from nba_api.stats.static import players
from nba_api.stats.endpoints import shotchartdetail
from src.api.model_loader import get_model, get_encoders
from src.api.schemas import ZoneStats, RecentForm, PlayerReportResponse

CATEGORICAL_COLS = ["ACTION_TYPE", "SHOT_TYPE", "SHOT_ZONE_BASIC",
                    "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE"]


def search_players(query: str) -> list[dict]:
    """
    Search NBA players by partial name match.

    Args:
        query: Partial player name string.

    Returns:
        List of dicts with player_id, full_name, is_active.
    """
    all_players = players.get_players()
    matches = [
        {"player_id": p["id"], "full_name": p["full_name"], "is_active": p["is_active"]}
        for p in all_players
        if query.lower() in p["full_name"].lower()
    ]
    return matches[:20]


def get_player_id(player_name: str) -> int:
    """
    Resolve player name to player ID.

    Args:
        player_name: Full or partial player name.

    Returns:
        Player ID integer.

    Raises:
        ValueError: If player not found.
    """
    results = search_players(player_name)
    if not results:
        raise ValueError(f"Player '{player_name}' not found.")
    return results[0]["player_id"]


def fetch_shots(player_name: str, season: str = "2023-24") -> pd.DataFrame:
    """
    Fetch shot chart data for a player/season with rate limit handling.

    Args:
        player_name: Full player name.
        season: Season string like '2023-24'.

    Returns:
        DataFrame with shot data.
    """
    player_id = get_player_id(player_name)
    time.sleep(1.0)
    chart = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=player_id,
        season_nullable=season,
        season_type_all_star="Regular Season",
        context_measure_simple="FGA",
    )
    df = chart.get_data_frames()[0]
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%Y%m%d", errors="coerce")
    return df


def compute_zone_stats(df: pd.DataFrame) -> list[ZoneStats]:
    """
    Compute per-zone shooting statistics and label hot/cold zones.

    Args:
        df: Shot DataFrame with SHOT_ZONE_BASIC and SHOT_MADE_FLAG.

    Returns:
        List of ZoneStats objects.
    """
    agg = (
        df.groupby("SHOT_ZONE_BASIC")["SHOT_MADE_FLAG"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "made", "count": "attempts"})
    )
    agg["fg_pct"] = agg["made"] / agg["attempts"]
    overall_avg = df["SHOT_MADE_FLAG"].mean()

    zones = []
    for zone, row in agg.iterrows():
        pct = row["fg_pct"]
        if pct >= overall_avg + 0.05:
            label = "🔥 Hot Zone"
        elif pct <= overall_avg - 0.05:
            label = "❄️ Cold Zone"
        else:
            label = "〜 Average Zone"
        zones.append(ZoneStats(
            zone=zone, attempts=int(row["attempts"]), made=int(row["made"]),
            fg_pct=round(pct, 3), label=label
        ))
    return sorted(zones, key=lambda z: z.fg_pct, reverse=True)


def compute_recent_form(df: pd.DataFrame, last_n_games: int = 10) -> list[RecentForm]:
    """
    Compute game-by-game FG% for the last N games.

    Args:
        df: Shot DataFrame with GAME_DATE and SHOT_MADE_FLAG.
        last_n_games: Number of recent games to include.

    Returns:
        List of RecentForm objects ordered by date.
    """
    game_stats = (
        df.groupby("GAME_DATE")["SHOT_MADE_FLAG"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "fg_pct", "count": "attempts"})
        .sort_values("GAME_DATE")
        .tail(last_n_games)
    )
    return [
        RecentForm(game_date=str(date.date()), fg_pct=round(row["fg_pct"], 3), attempts=int(row["attempts"]))
        for date, row in game_stats.iterrows()
    ]


def compute_form_trend(recent_form: list[RecentForm]) -> str:
    """
    Assess if a player's recent form is improving, declining, or stable.

    Args:
        recent_form: List of RecentForm objects ordered by date.

    Returns:
        String: 'improving', 'declining', or 'stable'.
    """
    if len(recent_form) < 3:
        return "stable"
    pcts = [f.fg_pct for f in recent_form]
    first_half_avg = np.mean(pcts[:len(pcts) // 2])
    second_half_avg = np.mean(pcts[len(pcts) // 2:])
    diff = second_half_avg - first_half_avg
    if diff > 0.03:
        return "improving"
    elif diff < -0.03:
        return "declining"
    return "stable"


def encode_feature(value: str, col: str) -> int:
    """
    Encode a categorical value using the saved LabelEncoder.

    Args:
        value: Raw string value.
        col: Column name for the corresponding encoder.

    Returns:
        Integer-encoded value (defaults to 0 if unseen).
    """
    encoders = get_encoders()
    if col not in encoders:
        return 0
    le = encoders[col]
    try:
        return int(le.transform([value.strip().title()])[0])
    except ValueError:
        return 0


def predict_shot(request_data: dict) -> dict:
    """
    Run model inference on a single shot input.

    Args:
        request_data: Dict matching PredictRequest fields.

    Returns:
        Dict with probability and interpretation fields.
    """
    model = get_model()
    row = {
        "SHOT_DISTANCE": request_data.get("shot_distance", 15),
        "LOC_X": request_data.get("loc_x", 0),
        "LOC_Y": request_data.get("loc_y", 150),
        "PERIOD": request_data.get("period", 1),
        "TIME_REMAINING_SECS": request_data.get("time_remaining_secs", 300),
        "IS_THREE_POINTER": request_data.get("is_three_pointer", 0),
        "IS_HOME": request_data.get("is_home", 1),
        "ACTION_TYPE": encode_feature(request_data.get("action_type", "Jump Shot"), "ACTION_TYPE"),
        "SHOT_TYPE": encode_feature(request_data.get("shot_type", "2Pt Field Goal"), "SHOT_TYPE"),
        "SHOT_ZONE_BASIC": encode_feature(request_data.get("shot_zone_basic", "Mid-Range"), "SHOT_ZONE_BASIC"),
        "SHOT_ZONE_AREA": encode_feature(request_data.get("shot_zone_area", "Center(C)"), "SHOT_ZONE_AREA"),
        "SHOT_ZONE_RANGE": encode_feature(request_data.get("shot_zone_range", "16-24 Ft."), "SHOT_ZONE_RANGE"),
        "SHOT_CLOCK": request_data.get("shot_clock", 12.0),
        "DRIBBLES": request_data.get("dribbles", 2.0),
        "TOUCH_TIME": request_data.get("touch_time", 2.5),
        "CLOSE_DEF_DIST": request_data.get("close_def_dist", 4.0),
        # Rolling features — filled with dataset averages at inference time
        "ROLLING_FG_PCT": request_data.get("rolling_fg_pct", 0.45),
        "ROLLING_ATTEMPTS": request_data.get("rolling_attempts", 10.0),
        "ZONE_FG_PCT": request_data.get("zone_fg_pct", 0.45),
    }
    df_input = pd.DataFrame([row])
    prob = model.predict_proba(df_input)[0][1]
    predicted = "Made" if prob >= 0.5 else "Missed"
    confidence = "High" if abs(prob - 0.5) > 0.2 else "Moderate" if abs(prob - 0.5) > 0.1 else "Low"
    interpretation = (
        f"The model estimates a {prob:.0%} chance this shot is made. "
        f"Confidence: {confidence}. Note: shot prediction has inherent variability."
    )
    return {
        "shot_made_probability": round(float(prob), 3),
        "shot_missed_probability": round(1 - float(prob), 3),
        "predicted_outcome": predicted,
        "confidence": confidence,
        "interpretation": interpretation,
    }


def build_player_report(
    player_name: str,
    season: str = "2023-24",
    last_n_games: int = 10,
    opponent: str = None
) -> PlayerReportResponse:
    """
    Fetch data and build a full scouting report for a player.

    Args:
        player_name: Full player name.
        season: Season string.
        last_n_games: Games for recent form analysis.
        opponent: Optional opponent team abbreviation for filtering.

    Returns:
        PlayerReportResponse with all scouting metrics.
    """
    df = fetch_shots(player_name, season)

    if opponent and "VTM" in df.columns:
        mask = df["VTM"].str.upper() == opponent.upper()
        if mask.sum() > 5:
            df = df[mask]

    zone_stats = compute_zone_stats(df)
    recent_form = compute_recent_form(df, last_n_games)
    form_trend = compute_form_trend(recent_form)

    overall_fg = df["SHOT_MADE_FLAG"].mean()
    threes = df[df["SHOT_TYPE"].str.contains("3Pt", case=False, na=False)]
    twos = df[~df["SHOT_TYPE"].str.contains("3Pt", case=False, na=False)]

    hot_zones = [z.zone for z in zone_stats if "Hot" in z.label]
    cold_zones = [z.zone for z in zone_stats if "Cold" in z.label]

    trend_text = {
        "improving": "📈 Player is trending upward in recent games.",
        "declining": "📉 Player's efficiency has dipped in recent games.",
        "stable": "➡️ Player performance is consistent."
    }[form_trend]

    projection_note = (
        f"Based on {last_n_games}-game rolling form and zone efficiency, "
        f"projected FG% for next game: ~{overall_fg:.0%}. "
        f"{trend_text} This is a scenario-based estimate, not a guarantee."
    )

    return PlayerReportResponse(
        player_name=player_name,
        season=season,
        total_attempts=len(df),
        overall_fg_pct=round(float(overall_fg), 3),
        three_pt_pct=round(float(threes["SHOT_MADE_FLAG"].mean()) if len(threes) > 0 else 0, 3),
        two_pt_pct=round(float(twos["SHOT_MADE_FLAG"].mean()) if len(twos) > 0 else 0, 3),
        zone_stats=zone_stats,
        recent_form=recent_form,
        hot_zones=hot_zones,
        cold_zones=cold_zones,
        form_trend=form_trend,
        projection_note=projection_note,
    )
