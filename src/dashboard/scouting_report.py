"""
scouting_report.py

Generates auto-written plain-English scouting reports from player stats.
Designed for non-technical users: coaches, scouts, journalists, fans.
"""


def generate_report(report_data: dict) -> str:
    """
    Generate a plain-English scouting report paragraph from structured data.

    Args:
        report_data: Dict with player stats from the /player-report API response.

    Returns:
        Multi-line string scouting report.
    """
    name = report_data.get("player_name", "This player")
    fg = report_data.get("overall_fg_pct", 0)
    three_pct = report_data.get("three_pt_pct", 0)
    two_pct = report_data.get("two_pt_pct", 0)
    hot_zones = report_data.get("hot_zones", [])
    cold_zones = report_data.get("cold_zones", [])
    form_trend = report_data.get("form_trend", "stable")
    attempts = report_data.get("total_attempts", 0)
    season = report_data.get("season", "this season")
    projection = report_data.get("projection_note", "")

    trend_phrases = {
        "improving": "Their recent form is trending upward — they appear to be finding their rhythm.",
        "declining": "Their efficiency has dipped in recent games — they may be going through a cold streak.",
        "stable": "Their performance has been consistent throughout the analyzed period.",
    }
    hot_phrase = (
        f"Their most dangerous areas are: **{', '.join(hot_zones)}**."
        if hot_zones else "No clearly dominant zones detected in this sample."
    )
    cold_phrase = (
        f"Defensively, opponents can shade them away from: **{', '.join(cold_zones)}**."
        if cold_zones else "No clearly weak zones identified."
    )

    report = f"""
**{name}** has taken **{attempts} shots** this {season}, converting at an overall rate of **{fg:.0%}**.
From two-point range, they're shooting **{two_pct:.0%}**, and from three-point range: **{three_pct:.0%}**.

{hot_phrase}
{cold_phrase}

{trend_phrases[form_trend]}

**Projection:** {projection}
""".strip()
    return report


def generate_bullet_insights(report_data: dict) -> list[str]:
    """
    Generate a short bulleted list of key insights for the scouting dashboard.

    Args:
        report_data: Dict with player stats.

    Returns:
        List of insight strings.
    """
    insights = []
    fg = report_data.get("overall_fg_pct", 0)
    three_pct = report_data.get("three_pt_pct", 0)
    hot_zones = report_data.get("hot_zones", [])
    cold_zones = report_data.get("cold_zones", [])
    form_trend = report_data.get("form_trend", "stable")
    attempts = report_data.get("total_attempts", 0)

    if fg >= 0.50:
        insights.append(f"✅ Above-average efficiency overall ({fg:.0%} FG%)")
    elif fg >= 0.43:
        insights.append(f"〜 League-average efficiency ({fg:.0%} FG%)")
    else:
        insights.append(f"⚠️ Below-average efficiency ({fg:.0%} FG% — may be volume shooting)")

    if three_pct >= 0.37:
        insights.append(f"🎯 Reliable three-point threat ({three_pct:.0%} from deep)")
    elif three_pct > 0:
        insights.append(f"❌ Contested threes advisable — shooting {three_pct:.0%} from range")

    if hot_zones:
        insights.append(f"🔥 High-efficiency zones: {', '.join(hot_zones)}")
    if cold_zones:
        insights.append(f"❄️ Force them here: {', '.join(cold_zones)}")

    if form_trend == "improving":
        insights.append("📈 Recent form is improving — trending toward peak efficiency")
    elif form_trend == "declining":
        insights.append("📉 Recent form is declining — may be fatigued or adjusting")

    if attempts < 50:
        insights.append("⚠️ Small sample size — interpret patterns with caution")

    return insights


def generate_defensive_focus(report_data: dict) -> str:
    """
    Generate a defensive strategy note for the scouting report.

    Args:
        report_data: Dict with player stats.

    Returns:
        Defensive focus string.
    """
    hot_zones = report_data.get("hot_zones", [])
    cold_zones = report_data.get("cold_zones", [])
    three_pct = report_data.get("three_pt_pct", 0)

    parts = []
    if hot_zones:
        parts.append(f"Close out hard in: {', '.join(hot_zones)}.")
    if cold_zones:
        parts.append(f"Concede contested attempts in: {', '.join(cold_zones)}.")
    if three_pct < 0.33:
        parts.append("Sag off on perimeter — they are not a reliable three-point shooter.")
    elif three_pct > 0.38:
        parts.append("Stay attached on the arc — they shoot efficiently from deep.")

    return " ".join(parts) if parts else "No specific defensive adjustments indicated by current data."
